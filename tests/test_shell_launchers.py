"""The macOS launchers, checked with a real bash.

Windows has one here too: Git for Windows ships bash, so these run on the machine
that writes them rather than waiting for a Mac to disagree. Where bash genuinely
is not present the suite skips rather than passing quietly — a launcher that was
never parsed is not a launcher anybody should trust.

The bodies live in ``tests/shell/*.sh`` and source their functions out of the real
scripts, so what runs here is what ships, not a copy that can drift.
"""
import shutil
import subprocess
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SHELL_DIR = Path(__file__).resolve().parent / "shell"
BASH = shutil.which("bash")

LAUNCHERS = ("run_agent.sh", "install_agent.sh")


@unittest.skipIf(BASH is None, "no bash on PATH")
class ShellSyntax(unittest.TestCase):
    """`bash -n` on both launchers: they parse before they are ever trusted."""

    def test_scripts_parse(self):
        for name in LAUNCHERS:
            with self.subTest(script=name):
                out = subprocess.run([BASH, "-n", str(ROOT / name)],
                                     capture_output=True, text=True)
                self.assertEqual(out.returncode, 0, out.stderr)

    def test_help_runs_without_touching_anything(self):
        """--help must not need a venv, a network, or a ComfyUI.

        It is also the only whole-script path that is safe to execute in a test, and
        it exercises the argument parser every other invocation goes through.
        """
        for name in LAUNCHERS:
            with self.subTest(script=name):
                out = subprocess.run([BASH, str(ROOT / name), "--help"],
                                     capture_output=True, text=True, timeout=60,
                                     cwd=str(ROOT))
                self.assertEqual(out.returncode, 0, out.stderr)
                self.assertIn("Usage:", out.stdout)
                self.assertIn("--help", out.stdout)

    def test_unknown_option_is_rejected(self):
        for name in LAUNCHERS:
            with self.subTest(script=name):
                out = subprocess.run([BASH, str(ROOT / name), "--nonsense"],
                                     capture_output=True, text=True, timeout=60,
                                     cwd=str(ROOT))
                self.assertEqual(out.returncode, 2)

    def test_no_bash_4_only_syntax(self):
        """macOS ships bash 3.2, and the newer forms are a syntax error there.

        `bash -n` on a modern bash cannot catch this — it parses them happily. So
        the constructs are named explicitly instead.
        """
        banned = (
            ("declare -A", "associative arrays are bash 4"),
            ("${!", "indirect/key expansion is bash 4"),
            ("mapfile", "mapfile is bash 4"),
            ("readarray", "readarray is bash 4"),
            (",,}", "${var,,} lowercasing is bash 4"),
            ("^^}", "${var^^} uppercasing is bash 4"),
            ("&>>", "&>> append-both is bash 4"),
        )
        for name in LAUNCHERS:
            # Comments stripped first: these scripts NAME the constructs they avoid,
            # in the header that explains why. Scanning the prose flags the very
            # documentation that keeps the rule alive.
            code = "\n".join(
                ln for ln in (ROOT / name).read_text(encoding="utf-8").splitlines()
                if not ln.lstrip().startswith("#"))
            for token, why in banned:
                with self.subTest(script=name, token=token):
                    self.assertNotIn(token, code, why)


@unittest.skipIf(BASH is None, "no bash on PATH")
class InstallerHelpers(unittest.TestCase):
    """The .env reader/writer, secret masking and ComfyUI detection."""

    def test_helpers(self):
        out = subprocess.run([BASH, str(SHELL_DIR / "installer_helpers_test.sh"), str(ROOT)],
                             capture_output=True, text=True, timeout=300)
        self.assertEqual(out.returncode, 0, out.stdout + out.stderr)
        self.assertIn("0 failed", out.stdout)


@unittest.skipIf(BASH is None, "no bash on PATH")
@unittest.skipIf(shutil.which("git") is None, "no git on PATH")
class UpdateRepo(unittest.TestCase):
    """The startup update against real repositories.

    This is the code that touches the user's working copy, so it is worth the
    seconds: it builds actual git repos and checks that local work survives a
    fast-forward, a genuine collision is stashed rather than dropped, and a diverged
    branch is left exactly as it was.
    """

    def test_update_repo(self):
        out = subprocess.run([BASH, str(SHELL_DIR / "update_repo_test.sh"), str(ROOT)],
                             capture_output=True, text=True, timeout=600)
        self.assertEqual(out.returncode, 0, out.stdout + out.stderr)
        self.assertIn("0 failed", out.stdout)


@unittest.skipIf(BASH is None, "no bash on PATH")
class FreePort(unittest.TestCase):
    """The only step in the launcher that destroys something.

    It picks a PID by matching a command line and sends it SIGKILL, so the case
    that matters most is the one where it must NOT: a stranger holding the port is
    reported and left alone. lsof/ps/kill are stubbed, so no real process is ever
    signalled by this test.
    """

    def test_free_port(self):
        out = subprocess.run([BASH, str(SHELL_DIR / "free_port_test.sh"), str(ROOT)],
                             capture_output=True, text=True, timeout=300)
        self.assertEqual(out.returncode, 0, out.stdout + out.stderr)
        self.assertIn("0 failed", out.stdout)


class LauncherParity(unittest.TestCase):
    """Both launchers must offer the same knobs, or the docs lie on one platform."""

    def test_run_agent_options_match(self):
        ps1 = (ROOT / "run_agent.ps1").read_text(encoding="utf-8", errors="replace")
        sh = (ROOT / "run_agent.sh").read_text(encoding="utf-8")
        for ps_name, sh_name in (("$Port", "--port"),
                                 ("$BindHost", "--host"),
                                 ("$Debug", "--debug"),
                                 ("$NoUpdate", "--no-update"),
                                 ("$LlmQueryTemplates", "--llm-query-templates"),
                                 ("$LlmAssembleWorkflow", "--llm-assemble-workflow")):
            with self.subTest(option=sh_name):
                self.assertIn(ps_name, ps1)
                self.assertIn(sh_name, sh)

    def test_install_agent_options_match(self):
        ps1 = (ROOT / "install_agent.ps1").read_text(encoding="utf-8", errors="replace")
        sh = (ROOT / "install_agent.sh").read_text(encoding="utf-8")
        for ps_name, sh_name in (("$ComfyUIPath", "--comfyui-path"),
                                 ("$ParentDir", "--parent-dir"),
                                 ("$SkipMcp", "--skip-mcp"),
                                 ("$SkipComfyNode", "--skip-comfy-node"),
                                 ("$NonInteractive", "--non-interactive")):
            with self.subTest(option=sh_name):
                self.assertIn(ps_name, ps1)
                self.assertIn(sh_name, sh)

    def test_both_installers_cover_the_same_seven_stages(self):
        ps1 = (ROOT / "install_agent.ps1").read_text(encoding="utf-8", errors="replace")
        sh = (ROOT / "install_agent.sh").read_text(encoding="utf-8")
        for stage in ("1 / 7", "2 / 7", "3 / 7", "4 / 7", "5 / 7", "6 / 7", "7 / 7"):
            with self.subTest(stage=stage):
                self.assertIn(stage, ps1)
                self.assertIn(stage, sh)

    def test_both_installers_ask_for_the_same_secrets(self):
        ps1 = (ROOT / "install_agent.ps1").read_text(encoding="utf-8", errors="replace")
        sh = (ROOT / "install_agent.sh").read_text(encoding="utf-8")
        for key in ("HF_TOKEN", "ANTHROPIC_API_KEY", "COMFYUI_API_KEY", "DASHSCOPE_API_KEY"):
            with self.subTest(key=key):
                self.assertIn(key, ps1)
                self.assertIn(key, sh)

    def test_the_sh_installer_does_not_reference_the_ps1_launcher(self):
        # Telling a Mac user to run run_agent.ps1 is the exact failure this port
        # exists to remove.
        sh = (ROOT / "install_agent.sh").read_text(encoding="utf-8")
        self.assertNotIn("run_agent.ps1", sh)


if __name__ == "__main__":
    unittest.main()
