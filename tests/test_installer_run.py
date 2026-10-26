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
        self.addCleanup(self._drop_links)      # runs first: never delete through a link
        (self.tmp / "bin").mkdir()
        (self.tmp / "bin" / "uv.cmd").write_text(FAKE_UV.format(python=sys.executable))
        (self.tmp / "bin" / "git.cmd").write_text(FAKE_GIT.format(git=GIT))
        self.agent = self.tmp / "agentY"
        self.core = self.tmp / "agentY-core"
        self.comfy = self.tmp / "comfy"
        (self.comfy / "custom_nodes").mkdir(parents=True)

    def _drop_links(self):
        sys.path.insert(0, str(ROOT / "scripts"))
        import core_dir
        for name in ("agenty_core", "agentY-core"):
            if core_dir.is_link(self.tmp / name):
                core_dir.remove_link(self.tmp / name)

    def seed(self, folder):
        """What the installer needs of an agentY checkout."""
        (folder / "config").mkdir(parents=True, exist_ok=True)
        (folder / "scripts").mkdir(exist_ok=True)
        for name in ("install_agent.ps1", ".env_example"):
            shutil.copy(ROOT / name, folder / name)
        shutil.copy(ROOT / "scripts" / "core_dir.py", folder / "scripts" / "core_dir.py")
        shutil.copy(ROOT / "scripts" / "hardware_profile.py", folder / "scripts" / "hardware_profile.py")
        (folder / "requirements.txt").write_text("requests\n")
        (folder / "scripts" / "check_env.py").write_text("print('check_env stub ok')\n")

    def plain_tree(self, settings=None):
        self.seed(self.agent)
        (self.core / ".git").mkdir(parents=True)          # "present", with nothing to pull
        (self.core / "pyproject.toml").write_text("[project]\n")
        if settings is not None:
            (self.agent / "config" / "settings.local.json").write_text(settings, encoding="utf-8")

    def run_installer(self, *args, answers="\n" * 14, channel="dev"):
        env = dict(os.environ, PATH=str(self.tmp / "bin") + os.pathsep + os.environ["PATH"])
        env.pop("AGENTY_INSTALLER_RESTARTED", None)
        env["AGENTY_UPDATE_CHANNEL"] = channel
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

    def test_declining_the_comfyui_installs_nothing_and_says_so(self):
        """GitHub issue 79: 'n' to the ComfyUI offered must not end with the node installed somewhere else."""
        self.plain_tree()
        out = self.run_installer("-ComfyUIPath", str(self.comfy), answers="\n" * 6 + "n\n" + "\n" * 8)   # the 7th prompt is the ComfyUI one
        self.assertEqual(out.returncode, 0, out.stdout + out.stderr)
        self.assertIn("Setup complete", out.stdout)
        self.assertFalse((self.comfy / "custom_nodes" / "agentY-comfyuiConnect").exists())
        self.assertIn("NOT installed (no ComfyUI was chosen)", out.stdout)
        self.assertIn("No ComfyUI was chosen", out.stdout)

    def test_a_rerun_keeps_what_the_settings_file_holds(self):
        before = {"comfyui_dir": "D:\\comfy", "models": {"orchestrator": ["a", "b"]}, "auto_update": False,
                  "memory": {"embedder": {"preset": "local"}, "enabled": True}}
        self.plain_tree(json.dumps(before))
        out = self.run_installer("-ComfyUIPath", str(self.comfy), answers="\n" * 7 + "http://10.0.0.5:8188\n" * 8)
        self.assertEqual(out.returncode, 0, out.stdout + out.stderr)
        after = self.settings()
        self.assertEqual(after.pop("comfyui_url", None), "http://10.0.0.5:8188")
        # The one new question saves its answer (the recommendation, on Enter); nothing else moves.
        self.assertIn(after.pop("execution", {}).get("mode"), ("local", "hybrid", "cloud"))
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
        _git(self.tmp, "clone", "-q", str(self.tmp / "core.git"), "agentY-core")
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

    def test_an_install_from_before_the_rename_is_moved_to_the_new_folder_name(self):
        """agenty_core next to agentY, from when the repository had that name:
        the checkout ends up as agentY-core, and the old path still resolves."""
        self.seed(self.agent)
        legacy = self.tmp / "agenty_core"
        (legacy / ".git").mkdir(parents=True)
        (legacy / "pyproject.toml").write_text("[project]" + chr(10))
        (legacy / "agenty_core").mkdir()
        (legacy / "agenty_core" / "__init__.py").write_text("")
        (legacy / "saved_template.json").write_text("mine")

        out = self.run_installer("-NonInteractive", "-SkipComfyNode")
        self.assertEqual(out.returncode, 0, out.stdout + out.stderr)
        self.assertIn("is now agentY-core", out.stdout)
        self.assertNotIn("[fake git] clone", out.stdout, "the existing checkout is used, not cloned again")
        self.assertTrue((self.core / "agenty_core" / "__init__.py").is_file())
        self.assertEqual((self.core / "saved_template.json").read_text(), "mine")
        self.assertEqual((legacy / "saved_template.json").read_text(), "mine", "the old path still resolves")
        self.assertNotEqual(os.path.realpath(legacy), os.path.abspath(legacy), "…through a link")

    def _three_commits(self):
        """agentY cloned at A; the remote's `stable` is at B, its main at C."""
        seed = self.tmp / "seed"
        seed.mkdir()
        _git(seed, "init", "-q", "-b", "main", ".")
        self.seed(seed)
        (seed / "version.txt").write_text("A\n")
        _git(seed, "add", "-A"); _git(seed, "commit", "-qm", "A")
        _git(self.tmp, "clone", "-q", "--bare", str(seed), "agentY.git")
        _git(self.tmp, "clone", "-q", str(self.tmp / "agentY.git"), "agentY")
        (seed / "version.txt").write_text("B\n")
        _git(seed, "commit", "-qam", "B")
        _git(seed, "push", "-q", str(self.tmp / "agentY.git"), "main", "main:stable")
        (seed / "version.txt").write_text("C\n")
        _git(seed, "commit", "-qam", "C")
        _git(seed, "push", "-q", str(self.tmp / "agentY.git"), "main")
        (self.core / ".git").mkdir(parents=True)
        (self.core / "pyproject.toml").write_text("[project]\n")

    def test_the_stable_channel_stops_at_the_release(self):
        self._three_commits()
        out = self.run_installer("-NonInteractive", "-SkipComfyNode", channel="stable")
        self.assertEqual(out.returncode, 0, out.stdout + out.stderr)
        self.assertIn("Release channel: stable", out.stdout)
        self.assertEqual((self.agent / "version.txt").read_text().strip(), "B")
        self.assertEqual(_git(self.agent, "rev-parse", "--abbrev-ref", "@{u}"), "origin/stable")

    def test_the_dev_channel_takes_every_commit(self):
        self._three_commits()
        out = self.run_installer("-NonInteractive", "-SkipComfyNode", channel="dev")
        self.assertEqual(out.returncode, 0, out.stdout + out.stderr)
        self.assertEqual((self.agent / "version.txt").read_text().strip(), "C")

    def test_a_checkout_past_the_release_is_left_where_it_is(self):
        self._three_commits()
        _git(self.agent, "pull", "-q", "--ff-only")                    # at C, ahead of stable
        out = self.run_installer("-NonInteractive", "-SkipComfyNode", channel="stable")
        self.assertEqual(out.returncode, 0, out.stdout + out.stderr)
        self.assertIn("ahead of the stable release", out.stdout)
        self.assertEqual((self.agent / "version.txt").read_text().strip(), "C")

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


PS_HARNESS = r"""
$ErrorActionPreference = 'Stop'
Set-StrictMode -Version 3.0
$Script:OnWindows = $true
function Write-Success { param($m) }
function Write-Info { param($m) }
function Write-Fail { param($m) [Console]::Error.WriteLine("FAILMSG $m") }
function Write-Host { param($Object, $ForegroundColor) [Console]::Error.WriteLine("LIST $Object") }
function Read-Answer { param([string]$Prompt) $v = [Console]::In.ReadLine(); if ($null -eq $v) { return "" } return [string]$v }
. '__BLOCK__'
$base = '__BASE__'
function Get-ComfyCandidatePaths { return @("$base\home\ComfyUI", "$base\home\Documents\ComfyUI", "$base\home\ComfyUI-Installs", "$base\nothing") }
function Get-ComfyDesktopBasePaths { return @("$base\elsewhere\mydata", "$base\home\Documents\ComfyUI") }
function Get-ComfyProcessLines { return @("C:\py\python.exe $base\proc\ComfyUI\main.py --port 8188", "python other.py") }
__BODY__
"""


@unittest.skipUnless(sys.platform == "win32" and POWERSHELL, "needs Windows PowerShell")
class InstallerComfyChoicePs(unittest.TestCase):
    """install_agent.ps1's ComfyUI discovery and choice, run on their own against fake folders (issue 79)."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="agy-comfy-"))
        self.addCleanup(shutil.rmtree, self.tmp, True)
        for rel in ("home/ComfyUI", "home/Documents/ComfyUI", "home/ComfyUI-Installs/desk1",
                    "elsewhere/mydata", "proc/ComfyUI", "single/ComfyUI"):
            (self.tmp / rel / "custom_nodes").mkdir(parents=True)
        (self.tmp / "home/ComfyUI/comfyui_version.py").write_text('__version__ = "0.3.60"\n')
        text = (ROOT / "install_agent.ps1").read_text(encoding="utf-8")
        block = text.split("# == ComfyUI discovery", 1)[1].split("# == end ComfyUI discovery", 1)[0]
        self.block = self.tmp / "block.ps1"
        self.block.write_text("# ==" + block, encoding="utf-8")

    def ps(self, body, answers=""):
        script = (PS_HARNESS.replace("__BLOCK__", str(self.block)).replace("__BASE__", str(self.tmp))
                  .replace("__BODY__", body))
        f = self.tmp / "run.ps1"
        f.write_text(script, encoding="utf-8")
        out = subprocess.run([POWERSHELL, "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(f)],
                             input=answers, capture_output=True, text=True, timeout=120)
        self.assertEqual(out.returncode, 0, out.stdout + out.stderr)
        return out.stdout.strip().splitlines(), out.stderr

    def test_every_install_is_found_once_with_a_label(self):
        lines, _ = self.ps('$a = @(Find-AllComfyUI ""); $a.Count; ($a | % { $_.Label }) -join "|"')
        self.assertEqual(lines[0], "5")                       # the basePath equal to Documents\ComfyUI counts once
        self.assertIn("v0.3.60", lines[1])
        self.assertIn("data folder only", lines[1])

    def test_the_user_picks_by_number(self):
        lines, _ = self.ps('$r = Select-ComfyUI -Hint "" -NonInteractive $false; $r.Path; $r.Found', answers="2\n")
        self.assertEqual(lines[1], "5")
        self.assertTrue(lines[0].startswith(str(self.tmp)))
        first, _ = self.ps('$r = Select-ComfyUI -Hint "" -NonInteractive $false; $r.Path', answers="1\n")
        self.assertNotEqual(first[0], lines[0])

    def test_declining_everything_picks_nothing(self):
        for answer in ("\n", "s\n", "n\n", "9\n", "0\n"):
            with self.subTest(answer=answer):
                lines, _ = self.ps('$r = Select-ComfyUI -Hint "" -NonInteractive $false; "[" + $r.Path + "]"', answers=answer)
                self.assertEqual(lines[-1], "[]")

    def test_a_typed_path_is_used_only_when_it_is_a_comfyui(self):
        good = str(self.tmp / "proc" / "ComfyUI")
        lines, _ = self.ps('$r = Select-ComfyUI -Hint "" -NonInteractive $false; $r.Path', answers=good + "\n")
        self.assertEqual(lines[0], good)
        (self.tmp / "notcomfy").mkdir()
        lines, err = self.ps('$r = Select-ComfyUI -Hint "" -NonInteractive $false; "[" + $r.Path + "]"',
                             answers=str(self.tmp / "notcomfy") + "\n")
        self.assertEqual(lines[-1], "[]")
        self.assertIn("look like a ComfyUI", err)

    def test_a_single_install_is_confirmed_and_can_be_declined(self):
        one = self.tmp / "single" / "ComfyUI"
        body = ('function Get-ComfyCandidatePaths { return @("%s") }; function Get-ComfyDesktopBasePaths { return @() }; '
                'function Get-ComfyProcessLines { return @() }; '
                '$r = Select-ComfyUI -Hint "" -NonInteractive $false; "[" + $r.Path + "]"' % one)
        yes, _ = self.ps(body, answers="\n")
        self.assertEqual(yes[-1], "[%s]" % one)
        no, _ = self.ps(body, answers="n\n")
        self.assertEqual(no[-1], "[]")

    def test_non_interactive_never_chooses_among_several(self):
        lines, _ = self.ps('$r = Select-ComfyUI -Hint "" -NonInteractive $true; "[" + $r.Path + "]"')
        self.assertEqual(lines[-1], "[]")
        path = str(self.tmp / "elsewhere" / "mydata")
        lines, _ = self.ps('$r = Select-ComfyUI -Hint "%s" -NonInteractive $true; $r.Path' % path)
        self.assertEqual(lines[-1], path)


class InstallerText(unittest.TestCase):
    """What can be held without running it, on any platform."""

    def test_both_installers_update_agentY_before_its_siblings(self):
        for name, call in (("install_agent.ps1", 'Update-Checkout -Name "agentY"'),
                           ("install_agent.sh", 'update_checkout "agentY"')):
            text = (ROOT / name).read_text(encoding="utf-8", errors="replace")
            with self.subTest(installer=name):
                self.assertIn(call, text)
                self.assertLess(text.index(call), text.index("github.com/szprivate/agentY-core.git"))

    def test_neither_installer_pulls_blind(self):
        for name in ("install_agent.ps1", "install_agent.sh"):
            with self.subTest(installer=name):
                self.assertNotIn("pull --ff-only }", (ROOT / name).read_text(encoding="utf-8", errors="replace"))
                self.assertNotIn('pull --ff-only ||', (ROOT / name).read_text(encoding="utf-8", errors="replace"))

    def test_both_installers_list_the_installs_and_never_fall_back(self):
        ps1 = (ROOT / "install_agent.ps1").read_text(encoding="utf-8", errors="replace")
        sh = (ROOT / "install_agent.sh").read_text(encoding="utf-8")
        for name, text in (("ps1", ps1), ("sh", sh)):
            with self.subTest(installer=name):
                self.assertIn("Install the sidebar node into which one?", text)
                self.assertIn("NOT installed (no ComfyUI was chosen)", text)
                self.assertIn("basePath", text)            # ComfyUI Desktop's config.json
                self.assertIn("ComfyUI-Installs", text)
                self.assertIn("main.py", text)             # running processes
                self.assertNotIn("Find-ComfyDesktop", text)
                self.assertNotIn("find_comfy_desktop", text)

    def test_the_sh_installer_parses(self):
        bash = shutil.which("bash")
        if not bash:
            self.skipTest("no bash on PATH")
        out = subprocess.run([bash, "-n", str(ROOT / "install_agent.sh")], capture_output=True, text=True)
        self.assertEqual(out.returncode, 0, out.stderr)


if __name__ == "__main__":
    unittest.main()
