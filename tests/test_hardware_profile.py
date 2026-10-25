"""The hardware profile behind the installers' one question (scripts/hardware_profile.py), and the installers'
parity on it: same flag, same question, same recommendation, same ComfyUI Desktop offer.

    python -m unittest tests.test_hardware_profile
"""
import json
import os
import shutil
import stat
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import hardware_profile as hp  # noqa: E402


def nv(name, mib):
    return {"name": name, "vram_mib": mib}


class Recommend(unittest.TestCase):
    def rec(self, **kw):
        prof = {"system": "windows", "machine": "amd64", "nvidia": [], "apple_silicon": False, "chip": "",
                "memory_gb": None, "other_gpu": ""}
        prof.update(kw)
        return hp.recommend(prof)["mode"]

    def test_nvidia_12gb_or_more_is_local(self):
        for mib in (12282, 12288, 16384, 24576, 32607):
            self.assertEqual(self.rec(nvidia=[nv("RTX", mib)]), "local", mib)

    def test_nvidia_6_to_12gb_is_hybrid(self):
        for mib in (6144, 8192, 8188, 10240, 11264):
            self.assertEqual(self.rec(nvidia=[nv("RTX", mib)]), "hybrid", mib)

    def test_a_weak_nvidia_gpu_is_cloud(self):
        for mib in (2048, 4096, 5120):
            self.assertEqual(self.rec(nvidia=[nv("GTX", mib)]), "cloud", mib)

    def test_no_or_non_nvidia_gpu_on_windows_is_cloud(self):
        self.assertEqual(self.rec(), "cloud")
        self.assertEqual(self.rec(other_gpu="AMD Radeon RX 7800 XT"), "cloud")
        self.assertEqual(self.rec(other_gpu="Intel(R) UHD Graphics"), "cloud")

    def test_apple_silicon_is_hybrid_and_intel_mac_is_cloud(self):
        self.assertEqual(self.rec(system="darwin", machine="arm64", apple_silicon=True, chip="Apple M2", memory_gb=16), "hybrid")
        self.assertEqual(self.rec(system="darwin", machine="x86_64", chip="Intel(R) Core(TM) i7"), "cloud")

    def test_the_biggest_gpu_counts(self):
        self.assertEqual(self.rec(nvidia=[nv("small", 4096), nv("big", 24576)]), "local")

    def test_the_reason_says_what_was_found(self):
        r = hp.recommend({"system": "darwin", "apple_silicon": True, "chip": "Apple M3 Pro", "memory_gb": 36, "nvidia": []})
        self.assertIn("Apple M3 Pro", r["reason"])
        self.assertIn("36 GB", r["reason"])


class Detect(unittest.TestCase):
    def test_nvidia_smi_output(self):
        got = hp.parse_nvidia_smi("NVIDIA GeForce RTX 4070, 12282\nNVIDIA GeForce RTX 3060, 6144\n")
        self.assertEqual(got, [nv("NVIDIA GeForce RTX 4070", 12282), nv("NVIDIA GeForce RTX 3060", 6144)])
        self.assertEqual(hp.parse_nvidia_smi(""), [])
        self.assertEqual(hp.parse_nvidia_smi("garbage"), [])

    def test_a_mac_reports_its_chip_and_unified_memory(self):
        def runner(cmd):
            return {"sysctl -n machdep.cpu.brand_string": "Apple M2 Pro\n",
                    "sysctl -n hw.memsize": str(32 * 1024 ** 3)}.get(" ".join(cmd), "")
        p = hp.detect("Darwin", "arm64", runner)
        self.assertTrue(p["apple_silicon"])
        self.assertEqual((p["chip"], p["memory_gb"]), ("Apple M2 Pro", 32))
        self.assertEqual(hp.recommend(p)["mode"], "hybrid")
        self.assertFalse(hp.detect("Darwin", "x86_64", runner)["apple_silicon"])

    def test_a_missing_tool_is_an_answer_not_a_crash(self):
        p = hp.detect("Linux", "x86_64", lambda cmd: "")
        self.assertEqual(hp.recommend(p)["mode"], "cloud")

    def test_the_real_machine_gives_a_valid_answer(self):
        self.assertIn(hp.profile()["mode"], hp.MODES)


class Settings(unittest.TestCase):
    def test_set_mode_merges_and_get_mode_reads(self):
        with tempfile.TemporaryDirectory() as d:
            f = os.path.join(d, "config", "settings.local.json")
            self.assertEqual(hp.current_mode(f), "")
            hp.set_mode(f, "hybrid")
            self.assertEqual(hp.current_mode(f), "hybrid")
            Path(f).write_text(json.dumps({"comfyui_url": "http://x", "execution": {"mode": "hybrid", "vram_margin_gb": 3}}))
            hp.set_mode(f, "cloud")
            data = json.loads(Path(f).read_text())
            self.assertEqual(data, {"comfyui_url": "http://x", "execution": {"mode": "cloud", "vram_margin_gb": 3}})

    def test_broken_json_is_left_alone(self):
        with tempfile.TemporaryDirectory() as d:
            f = os.path.join(d, "s.json")
            Path(f).write_text("{not json")
            with self.assertRaises(SystemExit):
                hp.set_mode(f, "cloud")
            self.assertEqual(Path(f).read_text(), "{not json")

    def test_a_bad_mode_is_refused(self):
        with self.assertRaises(ValueError):
            hp.set_mode("x.json", "turbo")


class InstallerParity(unittest.TestCase):
    def texts(self):
        return ((ROOT / "install_agent.ps1").read_text("utf-8", errors="replace"),
                (ROOT / "install_agent.sh").read_text("utf-8"))

    def test_both_have_the_flag_the_question_and_the_script(self):
        ps1, sh = self.texts()
        for ps_name, sh_name in (("$ExecutionMode", "--execution-mode"),):
            self.assertIn(ps_name, ps1)
            self.assertIn(sh_name, sh)
        for text in (ps1, sh):
            self.assertIn("hardware_profile.py", text)
            self.assertIn("Where should open-weight models", text)
            self.assertIn("[local/hybrid/cloud]", text)
            self.assertIn("--set-mode", text)
            self.assertIn("Partner API models", text)

    def test_both_ask_for_the_comfy_key_in_hybrid_and_cloud(self):
        for text in self.texts():
            self.assertIn("needed for", text)
            self.assertIn("COMFYUI_API_KEY", text)

    def test_both_offer_comfyui_desktop_with_the_official_routes(self):
        ps1, sh = self.texts()
        self.assertIn("Comfy.ComfyUI-Desktop", ps1)
        self.assertIn("brew install --cask comfy", ps1)
        self.assertIn("brew install --cask comfy", sh)
        for text in (ps1, sh):
            self.assertIn("https://www.comfy.org/download", text)
            self.assertIn("Install ComfyUI Desktop now?", text)
            self.assertIn("ComfyUI-Installs", text)

    def test_non_interactive_changes_nothing(self):
        ps1, sh = self.texts()
        self.assertIn("if ($NonInteractive) { return \"\" }", ps1)
        self.assertIn('[ "$NON_INTERACTIVE" = "1" ] && return 0', sh)

    def test_still_eight_stages(self):
        for text in self.texts():
            self.assertEqual(sum(f"{n} / 8" in text for n in range(1, 9)), 8)
            self.assertNotIn("9 / 9", text)


def _bash():
    """Git Bash on Windows (PATH's bash.exe there can be WSL, which cannot see these paths), else bash."""
    for c in ("C:/Program Files/Git/bin/bash.exe", "C:/Program Files (x86)/Git/bin/bash.exe"):
        if os.path.isfile(c):
            return c
    return shutil.which("bash") if os.name != "nt" else None


@unittest.skipUnless(_bash(), "no usable bash")
class ShQuestion(unittest.TestCase):
    """Run the shipping select_execution_mode with a stand-in venv python, answers on stdin."""

    def setUp(self):
        d = tempfile.TemporaryDirectory()
        self.addCleanup(d.cleanup)
        self.root = Path(d.name)
        (self.root / "scripts").mkdir()
        shutil.copy(ROOT / "scripts" / "hardware_profile.py", self.root / "scripts")
        py = self.root / ".venv" / "bin" / "python"
        py.parent.mkdir(parents=True)
        py.write_text(f'#!/usr/bin/env bash\nexec "{Path(sys.executable).as_posix()}" "$@"\n')
        py.chmod(py.stat().st_mode | stat.S_IEXEC)
        sh = (ROOT / "install_agent.sh").read_text("utf-8").replace("\r\n", "\n")
        body = "select_execution_mode() {" + sh.split("select_execution_mode() {", 1)[1].split("install_comfy_desktop() {", 1)[0]
        venv = "venv_python() { printf '%s/.venv/bin/python' \"$1\"; }\n"
        self.script = ("header(){ :; }; success(){ echo \"ok $1\" >&2; }; info(){ echo \"i $1\" >&2; }; "
                       "fail(){ echo \"! $1\" >&2; }; plain(){ echo \"$1\" >&2; }; die(){ echo \"$1\" >&2; exit 2; }\n"
                       f"{venv}{body}\n")
        self.settings = self.root / "config" / "settings.local.json"

    def run_q(self, answer, **env):
        script = self.script + (f'PROJECT_ROOT="{self.root.as_posix()}"; NON_INTERACTIVE="{env.get("ni", 0)}"; '
                                f'EXECUTION_MODE="{env.get("flag", "")}"\n'
                                f'select_execution_mode "{self.settings.as_posix()}"\n')
        out = subprocess.run([_bash(), "-c", script], input=answer, capture_output=True, text=True, timeout=60)
        return out.stdout.strip(), out.stderr

    def test_enter_takes_the_recommendation_and_saves_it(self):
        mode, err = self.run_q("\n")
        self.assertEqual(mode, hp.profile()["mode"])
        self.assertEqual(hp.current_mode(str(self.settings)), mode)
        self.assertIn("Where should open-weight models", err)
        self.assertIn("recommended:", err)

    def test_an_explicit_answer_wins(self):
        mode, _ = self.run_q("cloud\n")
        self.assertEqual((mode, hp.current_mode(str(self.settings))), ("cloud", "cloud"))

    def test_a_typo_falls_back_to_the_default(self):
        mode, err = self.run_q("turbo\n")
        self.assertEqual(mode, hp.profile()["mode"])
        self.assertIn("Unknown mode", err)

    def test_a_saved_mode_is_kept_on_enter(self):
        hp.set_mode(str(self.settings), "hybrid")
        mode, err = self.run_q("\n")
        self.assertEqual(mode, "hybrid")
        self.assertIn("keep hybrid", err)

    def test_non_interactive_asks_nothing_and_writes_nothing(self):
        mode, err = self.run_q("", ni=1)
        self.assertEqual(mode, "")
        self.assertFalse(self.settings.exists())

    def test_the_flag_skips_the_question(self):
        mode, err = self.run_q("", flag="hybrid", ni=1)
        self.assertEqual(mode, "hybrid")
        self.assertEqual(hp.current_mode(str(self.settings)), "hybrid")
        self.assertNotIn("Where should", err)


if __name__ == "__main__":
    unittest.main()
