"""What is committed has to work without what is only on this disk.

Agents committed parts of src/pipeline.py and left other parts in the working
tree. The tests ran against the working tree and passed; the commit did not even
parse, and v1.8.7 to v1.8.10 were released with it. Skill folders tracked as
upscale-ultimateSD while the disk (case-blind) said upscale-ultimatesd were the
same kind of thing. scripts/check_committed_tree.py checks HEAD as `git archive`
exports it, and make_release.py refuses to tag a tree that fails it.

    python -m unittest discover -s tests
"""

import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import check_committed_tree as cct  # noqa: E402
import make_release as mr  # noqa: E402

GIT = shutil.which("git")


def _is_checkout() -> bool:
    if GIT is None:
        return False
    done = subprocess.run([GIT, "-C", str(ROOT), "rev-parse", "--verify", "-q", "HEAD"], capture_output=True)
    return done.returncode == 0


def _git(repo, *args):
    subprocess.run([GIT, "-C", str(repo), *args], check=True, capture_output=True)


@unittest.skipUnless(_is_checkout(), "not a git checkout")
class ThisCheckoutsHead(unittest.TestCase):
    """The committed agentY, not the working tree, compiles and imports."""

    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory(prefix="agenty_head_")
        cls.root = cct.export(ROOT, "HEAD", Path(cls._tmp.name) / "agentY")

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def test_every_committed_python_file_compiles(self):
        self.assertEqual(cct.compile_errors(self.root), [])

    def test_every_committed_skill_folder_is_named_as_its_skill(self):
        self.assertEqual(cct.skill_name_errors(self.root), [])

    def test_the_host_entry_modules_import(self):
        self.assertEqual(cct.import_errors(self.root), [])


@unittest.skipIf(GIT is None, "no git on PATH")
class TheCheck(unittest.TestCase):

    def _repo(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        repo = Path(tmp.name) / "repo"
        repo.mkdir()
        _git(repo, "init", "-q")
        _git(repo, "config", "user.email", "t@example.com")
        _git(repo, "config", "user.name", "t")
        return repo, Path(tmp.name)

    def test_a_syntax_error_fixed_only_on_disk_is_found(self):
        repo, tmp = self._repo()
        (repo / "src").mkdir()
        (repo / "src" / "m.py").write_text("async def f():\n    yield 1\n    return 2\n", encoding="utf-8")
        _git(repo, "add", "-A")
        _git(repo, "commit", "-q", "-m", "half")
        (repo / "src" / "m.py").write_text("async def f():\n    yield 1\n", encoding="utf-8")  # the uncommitted half
        found = cct.compile_errors(cct.export(repo, "HEAD", tmp / "out"))
        self.assertEqual(len(found), 1)
        self.assertTrue(found[0].startswith("src/m.py:3: SyntaxError"), found)

    def test_a_skill_folder_differing_only_in_case_is_found(self):
        repo, tmp = self._repo()
        (repo / "skills" / "upscale-SD").mkdir(parents=True)
        (repo / "skills" / "upscale-SD" / "SKILL.md").write_text("---\nname: upscale-sd\n---\n", encoding="utf-8")
        (repo / "skills" / "fine").mkdir()
        (repo / "skills" / "fine" / "SKILL.md").write_text("---\nname: fine\n---\n", encoding="utf-8")
        _git(repo, "add", "-A")
        _git(repo, "commit", "-q", "-m", "skills")
        self.assertEqual(cct.skill_name_errors(cct.export(repo, "HEAD", tmp / "out")),
                         ["skills/upscale-SD: frontmatter name is 'upscale-sd'"])

    def test_an_import_that_fails_is_reported(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        root = Path(tmp.name)
        (root / "pkg").mkdir()
        (root / "pkg" / "__init__.py").write_text("", encoding="utf-8")
        (root / "pkg" / "bad.py").write_text("import a_module_nobody_has\n", encoding="utf-8")
        (root / "pkg" / "good.py").write_text("X = 1\n", encoding="utf-8")
        self.assertEqual(cct.import_errors(root, modules=("pkg.good",)), [])
        found = cct.import_errors(root, modules=("pkg.bad",))
        self.assertEqual(len(found), 1)
        self.assertIn("a_module_nobody_has", found[0])


class TheRelease(unittest.TestCase):

    def test_a_broken_committed_tree_stops_the_release_before_the_plugin_step(self):
        dirs = {n: Path("/nowhere") / n for n in ("agentY", *mr.SIBLINGS)}
        with mock.patch.object(mr, "repo_dirs", return_value=dirs), \
             mock.patch.object(mr, "_git", side_effect=lambda _p, *a, **_k: "origin/dev" if a[:2] == ("branch", "-r") else ""), \
             mock.patch.object(cct, "check_stack", return_value=["agentY HEAD: src/pipeline.py:1: SyntaxError"]), \
             mock.patch.object(mr, "plugin_prepare") as plugin, \
             mock.patch("builtins.print"):
            self.assertEqual(mr.release("9.9.9", dry_run=True), 2)
        plugin.assert_not_called()

    def test_the_check_runs_before_anything_is_tagged(self):
        text = (ROOT / "scripts" / "make_release.py").read_text(encoding="utf-8")
        self.assertLess(text.index("check_committed_tree.check_stack(dirs)"), text.index("ok, plugin_lines = plugin_prepare("))


if __name__ == "__main__":
    unittest.main()
