"""install_agent.ps1, actually run.

The installer is the first thing a new machine meets and the thing people re-run
to update an old one, and for a long time nothing ran it but people. Two faults
got through that way: under StrictMode it died at the ComfyUI stage on any
settings.local.json without a comfyui_url (every fresh install, once the embedder
stage started writing that file), and it never updated agentY itself - a re-run
brought agenty_core to the newest commit under an old agentY, and said every
checkout was "up to date" even when the pull had refused.

So these run the real script in a throwaway tree. Only two things are faked: `uv
pip` (no downloads) and `git clone` (no network); every other git call is real.
"""

import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
POWERSHELL = shutil.which("powershell") or shutil.which("pwsh")
GIT = shutil.which("git")

FAKE_UV = """@echo off
if "%1"=="pip" ( echo [fake uv] pip & exit /b 0 )
if "%1"=="venv" ( "{python}" -m venv --without-pip .venv & exit /b %ERRORLEVEL% )
echo uv 0.0.0-fake
"""
# `clone` only makes its target folder - the last argument, found by walking them,
# because cmd splits `-c core.longpaths=true` at the "=" and moves every index.
FAKE_GIT = """@echo off
if "%1"=="clone" goto clone
"{git}" %*
exit /b %ERRORLEVEL%
:clone
set "last="
:next
if "%~1"=="" goto made
set "last=%~1"
shift
goto next
:made
mkdir "%last%"
echo [fake git] clone
exit /b 0
"""


def _git(cwd, *args):
    return subprocess.run([GIT, "-c", "user.name=t", "-c", "user.email=t@t", "-c", "core.autocrlf=false",
                           *args], cwd=str(cwd), capture_output=True, text=True, check=True).stdout.strip()


@unittest.skipUnless(sys.platform == "win32" and POWERSHELL and GIT, "needs Windows PowerShell and git")
class InstallerRun(unittest.TestCase):

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="agy-inst-"))
        self.addCleanup(shutil.rmtree, self.tmp, True)
        (self.tmp / "bin").mkdir()
        (self.tmp / "bin" / "uv.cmd").write_text(FAKE_UV.format(python=sys.executable))
        (self.tmp / "bin" / "git.cmd").write_text(FAKE_GIT.format(git=GIT))
        self.agent = self.tmp / "agentY"
        self.core = self.tmp / "agenty_core"
        self.comfy = self.tmp / "comfy"
        (self.comfy / "custom_nodes").mkdir(parents=True)

    def seed(self, folder):
        """What the installer needs of an agentY checkout."""
        (folder / "config").mkdir(parents=True, exist_ok=True)
        (folder / "scripts").mkdir(exist_ok=True)
        for name in ("install_agent.ps1", ".env_example"):
            shutil.copy(ROOT / name, folder / name)
        (folder / "requirements.txt").write_text("requests\n")
        (folder / "scripts" / "check_env.py").write_text("print('check_env stub ok')\n")

    def plain_tree(self, settings=None):
        self.seed(self.agent)
        (self.core / ".git").mkdir(parents=True)          # "present", with nothing to pull
        (self.core / "pyproject.toml").write_text("[project]\n")
        if settings is not None:
            (self.agent / "config" / "settings.local.json").write_text(settings, encoding="utf-8")

    def run_installer(self, *args, answers="\n" * 14):
        env = dict(os.environ, PATH=str(self.tmp / "bin") + os.pathsep + os.environ["PATH"])
        env.pop("AGENTY_INSTALLER_RESTARTED", None)
        return subprocess.run(
            [POWERSHELL, "-NoProfile", "-ExecutionPolicy", "Bypass", "-File",
             str(self.agent / "install_agent.ps1"), "-SkipMcp", "-SkipTorch", *args],
            cwd=str(self.agent), env=env, input=answers, capture_output=True, text=True, timeout=300)

    def settings(self):
        return json.loads((self.agent / "config" / "settings.local.json").read_text(encoding="utf-8-sig"))

    def test_a_fresh_interactive_install_runs_to_the_end(self):
        """Enter at every prompt. The ComfyUI stage reads comfyui_url from a
        settings file the embedder stage has just written without one."""
        self.plain_tree()
        out = self.run_installer("-ComfyUIPath", str(self.comfy))
        self.assertEqual(out.returncode, 0, out.stdout + out.stderr)
        self.assertIn("Setup complete", out.stdout)
        self.assertNotIn("cannot be found on this object", out.stdout + out.stderr)
        self.assertIn(self.settings()["memory"]["embedder"]["preset"], ("local", "ollama"))
        node = self.comfy / "custom_nodes" / "agentY-comfyuiConnect"
        recorded = json.loads((node / ".agenty_host.json").read_text(encoding="utf-8"))["project_root"]
        self.assertTrue(os.path.samefile(recorded, self.agent), recorded)

    def test_a_rerun_keeps_what_the_settings_file_holds(self):
        before = {"comfyui_dir": "D:\\comfy", "models": {"orchestrator": ["a", "b"]}, "auto_update": False,
                  "memory": {"embedder": {"preset": "local"}, "enabled": True}}
        self.plain_tree(json.dumps(before))
        out = self.run_installer("-ComfyUIPath", str(self.comfy), answers="\n" * 6 + "http://10.0.0.5:8188\n" * 8)
        self.assertEqual(out.returncode, 0, out.stdout + out.stderr)
        after = self.settings()
        self.assertEqual(after.pop("comfyui_url", None), "http://10.0.0.5:8188")
        self.assertEqual(after, before)

    def test_a_settings_file_it_cannot_read_is_left_alone(self):
        self.plain_tree("{ not json")
        out = self.run_installer("-ComfyUIPath", str(self.comfy))
        self.assertEqual(out.returncode, 0, out.stdout + out.stderr)
        self.assertIn("not valid JSON", out.stdout)
        self.assertEqual((self.agent / "config" / "settings.local.json").read_text(encoding="utf-8"), "{ not json")

    def test_a_rerun_updates_agentY_itself_around_local_changes(self):
        """An old checkout: a tracked file the agent rewrote that the remote also
        changed, other local work, agenty_core one commit behind - and a newer
        installer on the remote, which must be the one that finishes the job."""
        seed = self.tmp / "seed"
        seed.mkdir()
        _git(seed, "init", "-q", "-b", "main", ".")
        self.seed(seed)
        (seed / "config" / "models.json").write_text('{"v": 1}\n')
        (seed / "notes.txt").write_text("theirs\n")
        _git(seed, "add", "-A"); _git(seed, "commit", "-qm", "A")
        _git(self.tmp, "clone", "-q", "--bare", str(seed), "agentY.git")
        _git(self.tmp, "clone", "-q", str(self.tmp / "agentY.git"), "agentY")
        (seed / "config" / "models.json").write_text('{"v": 2}\n')
        script = seed / "install_agent.ps1"
        script.write_bytes(script.read_bytes().replace(b'"  agentY stack installer"',
                                                       b'"  agentY stack installer (NEW ONE)"'))
        _git(seed, "commit", "-qam", "B"); _git(seed, "push", "-q", str(self.tmp / "agentY.git"), "main")

        coreseed = self.tmp / "coreseed"
        coreseed.mkdir()
        _git(coreseed, "init", "-q", "-b", "main", ".")
        (coreseed / "pyproject.toml").write_text("[project]\n")
        _git(coreseed, "add", "-A"); _git(coreseed, "commit", "-qm", "c1")
        _git(self.tmp, "clone", "-q", "--bare", str(coreseed), "core.git")
        _git(self.tmp, "clone", "-q", str(self.tmp / "core.git"), "agenty_core")
        (coreseed / "new.txt").write_text("x\n")
        _git(coreseed, "add", "-A"); _git(coreseed, "commit", "-qm", "c2")
        _git(coreseed, "push", "-q", str(self.tmp / "core.git"), "main")

        (self.agent / "config" / "models.json").write_text('{"v": "local scan"}\n')   # collides
        (self.agent / "notes.txt").write_text("mine\n")                                # does not
        (self.agent / "untracked.txt").write_text("scratch\n")

        out = self.run_installer("-NonInteractive", "-SkipComfyNode")
        self.assertEqual(out.returncode, 0, out.stdout + out.stderr)
        self.assertIn("The installer itself was updated", out.stdout)
        self.assertIn("(NEW ONE)", out.stdout)
        self.assertIn("Setup complete", out.stdout)
        self.assertEqual(_git(self.agent, "log", "--format=%s", "-1"), "B")
        self.assertEqual(_git(self.core, "log", "--format=%s", "-1"), "c2")
        self.assertEqual(json.loads((self.agent / "config" / "models.json").read_text()), {"v": 2})
        self.assertEqual((self.agent / "notes.txt").read_text(), "mine\n")
        self.assertEqual((self.agent / "untracked.txt").read_text(), "scratch\n")
        self.assertIn("agentY installer", _git(self.agent, "stash", "list"), "the local version is kept")

    def test_a_pull_that_cannot_happen_is_not_called_up_to_date(self):
        """Local commits the remote does not have: nothing to fast-forward to."""
        seed = self.tmp / "seed"
        seed.mkdir()
        _git(seed, "init", "-q", "-b", "main", ".")
        self.seed(seed)
        _git(seed, "add", "-A"); _git(seed, "commit", "-qm", "A")
        _git(self.tmp, "clone", "-q", "--bare", str(seed), "agentY.git")
        _git(self.tmp, "clone", "-q", str(self.tmp / "agentY.git"), "agentY")
        (seed / "remote.txt").write_text("r\n")
        _git(seed, "add", "-A"); _git(seed, "commit", "-qm", "B")
        _git(seed, "push", "-q", str(self.tmp / "agentY.git"), "main")
        (self.agent / "local.txt").write_text("l\n")
        _git(self.agent, "add", "-A"); _git(self.agent, "commit", "-qm", "mine")
        (self.core / ".git").mkdir(parents=True)
        (self.core / "pyproject.toml").write_text("[project]\n")

        out = self.run_installer("-NonInteractive", "-SkipComfyNode")
        self.assertEqual(out.returncode, 0, out.stdout + out.stderr)
        self.assertIn("NOT updated", out.stdout)
        self.assertNotIn("agentY up to date", out.stdout)
        self.assertEqual(_git(self.agent, "log", "--format=%s", "-1"), "mine")


class InstallerText(unittest.TestCase):
    """What can be held without running it, on any platform."""

    def test_both_installers_update_agentY_before_its_siblings(self):
        for name, call in (("install_agent.ps1", 'Update-Checkout -Name "agentY"'),
                           ("install_agent.sh", 'update_checkout "agentY"')):
            text = (ROOT / name).read_text(encoding="utf-8", errors="replace")
            with self.subTest(installer=name):
                self.assertIn(call, text)
                self.assertLess(text.index(call), text.index("github.com/szprivate/agenty_core.git"))

    def test_neither_installer_pulls_blind(self):
        for name in ("install_agent.ps1", "install_agent.sh"):
            with self.subTest(installer=name):
                self.assertNotIn("pull --ff-only }", (ROOT / name).read_text(encoding="utf-8", errors="replace"))
                self.assertNotIn('pull --ff-only ||', (ROOT / name).read_text(encoding="utf-8", errors="replace"))

    def test_the_sh_installer_parses(self):
        bash = shutil.which("bash")
        if not bash:
            self.skipTest("no bash on PATH")
        out = subprocess.run([bash, "-n", str(ROOT / "install_agent.sh")], capture_output=True, text=True)
        self.assertEqual(out.returncode, 0, out.stderr)


if __name__ == "__main__":
    unittest.main()
