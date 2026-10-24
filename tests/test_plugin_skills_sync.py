"""The agentY-mcp plugin follows agentY: skill mirror, release hookup, installer step.

    python -m unittest tests.test_plugin_skills_sync
"""
import importlib.util
import json
import re
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import make_release as mr  # noqa: E402
import sync_plugin_skills as sps  # noqa: E402

_spec = importlib.util.spec_from_file_location("optional_deps", ROOT / "scripts" / "optional_deps.py")
od = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(od)


def _skill(root: Path, name: str, body: str = "x", extra: dict = None):
    d = root / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "SKILL.md").write_text(f"---\nname: {name}\ndescription: d\n---\n{body}\n", encoding="utf-8")
    for rel, text in (extra or {}).items():
        (d / rel).parent.mkdir(parents=True, exist_ok=True)
        (d / rel).write_text(text, encoding="utf-8")


class Mirror(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.src = Path(tmp.name) / "agentY" / "skills"
        self.mcp = Path(tmp.name) / "agentY-mcp"
        self.src.mkdir(parents=True)
        self.mcp.mkdir()
        _skill(self.src, "alpha", extra={"references/a.md": "ref"})
        _skill(self.src, "beta")
        _skill(self.src, "hostonly")
        (self.src / "folder-of-skills" / "inner").mkdir(parents=True)
        self.exclude = {"hostonly": "uses panel tools", "folder-of-skills": "no SKILL.md of its own"}

    def test_first_sync_copies_everything_but_the_excluded(self):
        p = sps.sync(self.src, self.mcp, self.exclude)
        self.assertEqual(p["add"], ["alpha", "beta"])
        self.assertTrue((self.mcp / "skills" / "alpha" / "references" / "a.md").is_file())
        self.assertFalse((self.mcp / "skills" / "hostonly").exists())
        manifest = json.loads((self.mcp / "skills.sync.json").read_text(encoding="utf-8"))
        self.assertEqual(sorted(manifest["skills"]), ["alpha", "beta"])
        self.assertEqual(manifest["excluded"], ["folder-of-skills", "hostonly"])

    def test_in_sync_after_a_sync(self):
        sps.sync(self.src, self.mcp, self.exclude)
        self.assertTrue(sps.is_clean(sps.changes(self.src, self.mcp, self.exclude)))

    def test_update_add_and_remove_are_mirrored(self):
        sps.sync(self.src, self.mcp, self.exclude)
        (self.src / "alpha" / "SKILL.md").write_text("---\nname: alpha\ndescription: d\n---\nchanged\n", encoding="utf-8")
        shutil.rmtree(self.src / "beta")
        _skill(self.src, "gamma")
        p = sps.changes(self.src, self.mcp, self.exclude)
        self.assertEqual((p["add"], p["update"], p["remove"]), (["gamma"], ["alpha"], ["beta"]))
        sps.sync(self.src, self.mcp, self.exclude)
        self.assertEqual(sorted(d.name for d in (self.mcp / "skills").iterdir()), ["alpha", "gamma"])
        self.assertTrue(sps.is_clean(sps.changes(self.src, self.mcp, self.exclude)))

    def test_a_hand_edit_in_the_plugin_is_drift(self):
        sps.sync(self.src, self.mcp, self.exclude)
        (self.mcp / "skills" / "beta" / "SKILL.md").write_text("edited", encoding="utf-8")
        self.assertEqual(sps.changes(self.src, self.mcp, self.exclude)["update"], ["beta"])

    def test_line_endings_are_not_drift(self):
        sps.sync(self.src, self.mcp, self.exclude)
        f = self.mcp / "skills" / "beta" / "SKILL.md"
        f.write_bytes(f.read_bytes().replace(b"\r\n", b"\n").replace(b"\n", b"\r\n"))
        self.assertTrue(sps.is_clean(sps.changes(self.src, self.mcp, self.exclude)))

    def test_folder_and_name_must_match(self):
        (self.src / "beta" / "SKILL.md").write_text("---\nname: other\ndescription: d\n---\n", encoding="utf-8")
        with self.assertRaises(ValueError):
            sps.source_skills(self.src, self.exclude)

    def test_a_folder_without_skill_md_needs_an_exclude_reason(self):
        with self.assertRaises(ValueError):
            sps.source_skills(self.src, {"hostonly": "x"})

    def test_a_stale_exclude_entry_is_an_error(self):
        with self.assertRaises(ValueError):
            sps.source_skills(self.src, {**self.exclude, "gone": "was removed"})

    def test_report_names_tools_the_server_does_not_offer(self):
        (self.src / "alpha" / "SKILL.md").write_text(
            "---\nname: alpha\ndescription: d\nallowed-tools: get_node_schema, show_in_panel\n---\n", encoding="utf-8")
        self.assertEqual(sps.report(self.src, self.exclude, {"get_node_schema"}), [("alpha", ["show_in_panel"])])


class TheShippedConfig(unittest.TestCase):
    def test_every_exclude_has_a_reason_and_exists(self):
        cfg = sps.load_config()
        for name, reason in cfg["exclude"].items():
            self.assertTrue(reason.strip(), name)
            self.assertTrue((ROOT / "skills" / name).exists(), f"{name} is excluded but not in skills/")

    def test_agentY_skills_can_be_mirrored(self):
        skills = sps.source_skills(ROOT / "skills", sps.load_config()["exclude"])
        self.assertIn("prompting", skills)
        self.assertNotIn("canvas-layout", skills)

    def test_shipped_skills_with_non_mcp_tools_are_noted(self):
        mcp = ROOT.parent / "agentY-mcp"
        if not (mcp / "src" / "mcp_server.py").is_file():
            self.skipTest("agentY-mcp is not next to agentY")
        cfg = sps.load_config()
        offered = {"list_skills", "read_skill"}
        dirs = [mcp / "src" / "tools", ROOT.parent / "agentY-core" / "agenty_core" / "tools"]
        for d in dirs:
            for f in d.glob("*.py") if d.is_dir() else []:
                offered |= set(re.findall(r"^def (\w+)\(", f.read_text(encoding="utf-8"), re.M))
        unnoted = [n for n, _extra in sps.report(ROOT / "skills", cfg["exclude"], offered) if n not in cfg["notes"]]
        self.assertEqual(unnoted, [], "these shipped skills name tools the MCP server lacks; add a note (or an exclude) in "
                         "config/plugin_skills.json: " + ", ".join(unnoted))


class ReleaseHookup(unittest.TestCase):
    def _git_repo(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        repo = Path(tmp.name)
        subprocess.run(["git", "init", "-q", str(repo)], check=True)
        (repo / "skills").mkdir()
        (repo / "skills" / "old.txt").write_text("x", encoding="utf-8")
        subprocess.run(["git", "-C", str(repo), "add", "."], check=True)
        subprocess.run(["git", "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "i"], check=True)
        return repo

    @unittest.skipIf(shutil.which("git") is None, "no git on PATH")
    def test_uncommitted_plugin_files_stop_the_release(self):
        repo = self._git_repo()
        (repo / "skills" / "old.txt").write_text("somebody else's work", encoding="utf-8")
        ok, lines = mr.plugin_prepare("1.5.0", repo, dry_run=False)
        self.assertFalse(ok)
        self.assertIn("uncommitted", lines[0])
        self.assertEqual((repo / "skills" / "old.txt").read_text(encoding="utf-8"), "somebody else's work")

    def test_release_runs_the_plugin_step_before_the_commits_are_read(self):
        text = (ROOT / "scripts" / "make_release.py").read_text(encoding="utf-8")
        self.assertLess(text.index("plugin_prepare(version"), text.index("repos = {n: {"))

    def test_dry_run_reports_and_writes_nothing(self):
        mcp = ROOT.parent / "agentY-mcp"
        if not (mcp / ".venv").exists():
            self.skipTest("agentY-mcp is not installed next to agentY")
        dirty = subprocess.run(["git", "-C", str(mcp), "status", "--porcelain", "--", *mr.PLUGIN_PATHS],
                               capture_output=True, text=True).stdout
        if dirty.strip():
            self.skipTest("agentY-mcp has uncommitted plugin files")
        before = subprocess.run(["git", "-C", str(mcp), "status", "--porcelain"], capture_output=True, text=True).stdout
        ok, lines = mr.plugin_prepare("9.9.9", mcp, dry_run=True)
        after = subprocess.run(["git", "-C", str(mcp), "status", "--porcelain"], capture_output=True, text=True).stdout
        self.assertEqual(before, after)
        self.assertTrue(any("would" in line for line in lines), lines)


class InstallerStep(unittest.TestCase):
    def test_it_is_offered_and_starts_unticked(self):
        self.assertIn("claude_code_plugin", od.OPTIONAL)
        with mock.patch.object(od, "read_choices", return_value={}), mock.patch("builtins.print") as pr:
            od.main(["list"])
        rows = {c.args[0].split("|")[0]: c.args[0].split("|")[3] for c in pr.call_args_list}
        self.assertEqual(rows["claude_code_plugin"], "0")
        self.assertEqual(rows["reverse_image_search"], "")

    def test_the_launcher_never_runs_it(self):
        with mock.patch.object(od, "read_choices", return_value={"claude_code_plugin": True}):
            self.assertEqual(od.wanted_and_missing(), [])

    def test_with_the_cli_it_adds_the_marketplace_and_installs(self):
        with mock.patch.object(od.shutil, "which", return_value="/bin/claude"), \
             mock.patch.object(od.subprocess, "run") as run, \
             mock.patch.object(Path, "is_file", return_value=True):
            run.return_value.returncode = 0
            self.assertEqual(od.install("claude_code_plugin"), 0)
        cmds = [c.args[0] for c in run.call_args_list]
        self.assertEqual(cmds[0][1:4], ["plugin", "marketplace", "add"])
        self.assertTrue(cmds[0][4].endswith("agentY-mcp"))
        self.assertEqual(cmds[1][1:], ["plugin", "install", "agenty@agentY"])

    def test_without_the_cli_the_two_commands_are_printed(self):
        with mock.patch.object(od.shutil, "which", return_value=None), \
             mock.patch.object(od.subprocess, "run") as run, \
             mock.patch.object(Path, "is_file", return_value=True), \
             mock.patch("builtins.print") as pr:
            self.assertEqual(od.install("claude_code_plugin"), 0)
        self.assertFalse(run.called)
        out = "\n".join(str(c.args[0]) for c in pr.call_args_list)
        self.assertIn("claude plugin marketplace add", out)
        self.assertIn("claude plugin install agenty@agentY", out)

    def test_a_failing_command_prints_them_and_reports_failure(self):
        with mock.patch.object(od.shutil, "which", return_value="/bin/claude"), \
             mock.patch.object(od.subprocess, "run") as run, \
             mock.patch.object(Path, "is_file", return_value=True), \
             mock.patch("builtins.print") as pr:
            run.return_value.returncode = 1
            self.assertEqual(od.install("claude_code_plugin"), 1)
        self.assertIn("claude plugin install agenty@agentY", "\n".join(str(c.args[0]) for c in pr.call_args_list))


if __name__ == "__main__":
    unittest.main()
