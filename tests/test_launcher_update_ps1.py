"""run_agent.ps1's update step, run for real against git repositories.

The launcher cannot be started in a test (it ends by serving the agent), so its
functions are lifted out of the script by PowerShell's own parser and called
directly. What is checked is what a release channel promises: a stable machine
stops at the release, a dev machine takes every commit, and neither goes
backwards.
"""

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

HARNESS = r'''
param([string]$Script, [string]$Repo, [string]$Channel)
$ErrorActionPreference = "Stop"
$ast = [System.Management.Automation.Language.Parser]::ParseFile($Script, [ref]$null, [ref]$null)
$fns = $ast.FindAll({ param($n) $n -is [System.Management.Automation.Language.FunctionDefinitionAst] }, $false)
foreach ($f in $fns) { Invoke-Expression $f.Extent.Text }
$ProjectRoot = Split-Path -Parent $Script
$env:AGENTY_UPDATE_CHANNEL = $Channel
$Script:Channel = Get-UpdateChannel
Write-Host "channel=$($Script:Channel)"
$changed = Update-Repo -Name "repo" -Dir $Repo
Write-Host "deps=$changed"
'''


def _git(cwd, *args):
    return subprocess.run([GIT, "-c", "user.name=t", "-c", "user.email=t@t", "-c", "core.autocrlf=false",
                           *args], cwd=str(cwd), capture_output=True, text=True, check=True).stdout.strip()


@unittest.skipUnless(sys.platform == "win32" and POWERSHELL and GIT, "needs Windows PowerShell and git")
class LauncherChannels(unittest.TestCase):

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="agy-run-"))
        self.addCleanup(shutil.rmtree, self.tmp, True)
        (self.tmp / "harness.ps1").write_text(HARNESS, encoding="utf-8")
        seed = self.tmp / "seed"
        seed.mkdir()
        _git(seed, "init", "-q", "-b", "main", ".")
        (seed / "version.txt").write_text("A\n")
        (seed / "requirements.txt").write_text("requests\n")
        _git(seed, "add", "-A"); _git(seed, "commit", "-qm", "A")
        _git(self.tmp, "clone", "-q", "--bare", str(seed), "remote.git")
        _git(self.tmp, "clone", "-q", str(self.tmp / "remote.git"), "repo")     # the machine, at A
        (seed / "version.txt").write_text("B\n")
        (seed / "requirements.txt").write_text("requests\nflask\n")
        _git(seed, "commit", "-qam", "B")
        _git(seed, "push", "-q", str(self.tmp / "remote.git"), "main", "main:stable")   # release at B
        (seed / "version.txt").write_text("C\n")
        _git(seed, "commit", "-qam", "C")
        _git(seed, "push", "-q", str(self.tmp / "remote.git"), "main")                  # dev at C
        self.repo = self.tmp / "repo"

    def update(self, channel):
        out = subprocess.run(
            [POWERSHELL, "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(self.tmp / "harness.ps1"),
             "-Script", str(ROOT / "run_agent.ps1"), "-Repo", str(self.repo), "-Channel", channel],
            capture_output=True, text=True, timeout=120, env=dict(os.environ))
        self.assertEqual(out.returncode, 0, out.stdout + out.stderr)
        return out.stdout

    def version(self):
        return (self.repo / "version.txt").read_text().strip()

    def test_stable_stops_at_the_release_and_reports_changed_dependencies(self):
        out = self.update("stable")
        self.assertIn("channel=stable", out)
        self.assertEqual(self.version(), "B")
        self.assertIn("deps=", out)
        self.assertRegex(out, r"deps=\S", "requirements.txt changed between A and B")
        self.assertEqual(_git(self.repo, "rev-parse", "--abbrev-ref", "@{u}"), "origin/stable")

    def test_dev_takes_every_commit(self):
        self.update("dev")
        self.assertEqual(self.version(), "C")

    def test_stable_never_goes_backwards(self):
        _git(self.repo, "pull", "-q", "--ff-only")                 # at C, past the release
        out = self.update("stable")
        self.assertEqual(self.version(), "C")
        self.assertIn("ahead of the stable release", out)

    def test_switching_back_to_dev_follows_the_branch_again(self):
        self.update("stable")
        self.assertEqual(self.version(), "B")
        self.update("dev")
        self.assertEqual(self.version(), "C")
        self.assertEqual(_git(self.repo, "rev-parse", "--abbrev-ref", "@{u}"), "origin/main")

    def test_anything_but_dev_is_stable(self):
        self.assertIn("channel=stable", self.update("nightly"))


if __name__ == "__main__":
    unittest.main()
