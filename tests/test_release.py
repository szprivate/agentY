"""A release is a set: four repositories at known commits, and pinned packages.

Before this, every machine took the newest commit of each repository's main
branch whenever it happened to start, so the four were compatible only by luck
of timing. scripts/make_release.py records the set (release.toml), pins the
Python packages (requirements.lock) and moves each repository's `stable` branch;
the launchers follow `stable` unless told to follow `dev`.
"""

import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import make_release as mr  # noqa: E402
import sync_deps as sd  # noqa: E402

GIT = shutil.which("git")


class TheManifest(unittest.TestCase):

    REPOS = {"agentY-core": {"tag": "v1.2.0", "commit": "a" * 40},
             "agentY-comfyuiConnect": {"tag": "v1.2.0", "commit": "b" * 40},
             "agentY-mcp": {"tag": "v1.2.0", "commit": "c" * 40}}

    def test_what_is_written_is_what_is_read(self):
        text = mr.render_manifest("1.2.0", "2026-10-07", self.REPOS)
        back = mr.parse_manifest(text)
        self.assertEqual(back["version"], "1.2.0")
        self.assertEqual(back["date"], "2026-10-07")
        self.assertEqual(back["repos"], self.REPOS)

    def test_it_is_real_toml(self):
        try:
            import tomllib
        except ImportError:
            self.skipTest("no tomllib before Python 3.11")
        data = tomllib.loads(mr.render_manifest("1.2.0", "2026-10-07", self.REPOS))
        self.assertEqual(data["repos"]["agentY-core"]["commit"], "a" * 40)

    def test_a_missing_file_is_no_release_not_an_error(self):
        self.assertIsNone(mr.read_manifest(ROOT / "no-such-release.toml"))

    def test_a_bad_version_is_refused_before_anything_is_touched(self):
        for bad in ("1.2", "v1.2.0", "one", ""):
            self.assertEqual(mr.release(bad, dry_run=True), 2)


class TheChannel(unittest.TestCase):

    def _settings(self, data):
        d = tempfile.TemporaryDirectory()
        self.addCleanup(d.cleanup)
        path = Path(d.name) / "settings.local.json"
        path.write_text(json.dumps(data), encoding="utf-8")
        return path

    def test_stable_is_the_default(self):
        self.assertEqual(mr.channel(env={}, settings=self._settings({})), "stable")
        self.assertEqual(mr.channel(env={}, settings=Path("nowhere.json")), "stable")

    def test_settings_and_environment_can_say_dev(self):
        self.assertEqual(mr.channel(env={}, settings=self._settings({"update_channel": "dev"})), "dev")
        self.assertEqual(mr.channel(env={"AGENTY_UPDATE_CHANNEL": "DEV"}, settings=self._settings({})), "dev")
        self.assertEqual(mr.channel(env={"AGENTY_UPDATE_CHANNEL": "stable"},
                                    settings=self._settings({"update_channel": "dev"})), "stable")

    def test_the_shipped_default_is_stable_and_every_script_reads_it_the_same_way(self):
        self.assertIn('update_channel = "stable"',
                      (ROOT / "config" / "settings.default.toml").read_text(encoding="utf-8"))
        for name in ("run_agent.ps1", "install_agent.ps1", "run_agent.sh", "install_agent.sh"):
            text = (ROOT / name).read_text(encoding="utf-8", errors="replace")
            with self.subTest(script=name):
                self.assertIn("AGENTY_UPDATE_CHANNEL", text)
                self.assertIn("update_channel", text)
                self.assertIn("origin/stable", text)


@unittest.skipUnless(GIT, "needs git")
class Drift(unittest.TestCase):

    def test_a_sibling_at_another_commit_is_named(self):
        with tempfile.TemporaryDirectory() as d:
            repo = Path(d) / "agentY-core"
            repo.mkdir()
            run = lambda *a: subprocess.run([GIT, "-c", "user.name=t", "-c", "user.email=t@t", *a],  # noqa: E731
                                            cwd=str(repo), capture_output=True, text=True, check=True).stdout.strip()
            run("init", "-q", ".")
            (repo / "a.txt").write_text("1")
            run("add", "-A"); run("commit", "-qm", "one")
            head = run("rev-parse", "HEAD")
            dirs = {"agentY-core": repo}
            same = {"version": "1.0.0", "repos": {"agentY-core": {"tag": "v1.0.0", "commit": head}}}
            other = {"version": "1.0.0", "repos": {"agentY-core": {"tag": "v1.0.0", "commit": "f" * 40}}}
            self.assertEqual(mr.drift(same, dirs), [])
            self.assertEqual(mr.drift(other, dirs), [("agentY-core", head[:7], "f" * 7)])
            absent = {"version": "1.0.0", "repos": {"agentY-mcp": {"tag": "v1.0.0", "commit": "f" * 40}}}
            self.assertEqual(mr.drift(absent, dirs), [], "a repository that is not installed is not drift")


class TheLock(unittest.TestCase):

    def test_installs_use_the_lock_as_constraints_when_there_is_one(self):
        from unittest import mock
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            with mock.patch.object(sd, "ROOT", root), mock.patch.object(sd.shutil, "which", lambda n: "uv"):
                self.assertEqual(sd.install_command("py"),
                                 ["uv", "pip", "install", "--python", "py", "-r", "requirements.txt"])
                (root / "requirements.lock").write_text("requests==2.0\n")
                self.assertEqual(sd.install_command("py")[-2:], ["-c", "requirements.lock"])
            with mock.patch.object(sd, "ROOT", root), mock.patch.object(sd.shutil, "which", lambda n: None):
                self.assertEqual(sd.install_command("py"),
                                 ["py", "-m", "pip", "install", "-r", "requirements.txt", "-c", "requirements.lock"])

    def test_a_changed_lock_is_a_reason_to_reinstall(self):
        self.assertIn(ROOT / "requirements.lock", sd.dep_files())

    def test_both_installers_install_against_it_and_pin_torch_to_it(self):
        ps1 = (ROOT / "install_agent.ps1").read_text(encoding="utf-8", errors="replace")
        sh = (ROOT / "install_agent.sh").read_text(encoding="utf-8")
        self.assertIn("-c requirements.lock", ps1)
        self.assertIn("-c requirements.lock", sh)
        self.assertIn('Get-LockedSpec -Dir $Dir -Package "torch"', ps1)

    def test_the_committed_lock_pins_what_requirements_names(self):
        lock = ROOT / "requirements.lock"
        if not lock.is_file():
            self.skipTest("no release has been made from this checkout yet")
        text = lock.read_text(encoding="utf-8")
        self.assertNotIn("-e ", text, "an editable line is not a constraint")
        for name in ("torch", "strands-agents", "mem0ai", "flask", "fastembed"):
            with self.subTest(package=name):
                self.assertRegex(text, rf"(?m)^{name}==\d")

    def test_the_committed_manifest_names_every_sibling(self):
        manifest = mr.read_manifest()
        if manifest is None:
            self.skipTest("no release has been made from this checkout yet")
        self.assertRegex(manifest["version"], r"^\d+\.\d+\.\d+$")
        for name in mr.SIBLINGS:
            with self.subTest(repo=name):
                self.assertRegex(manifest["repos"][name]["commit"], r"^[0-9a-f]{40}$")
                self.assertEqual(manifest["repos"][name]["tag"], "v" + manifest["version"])


if __name__ == "__main__":
    unittest.main()
