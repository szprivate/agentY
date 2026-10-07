"""The tool layer's folder is agentY-core; an old install is moved over, safely.

The repository was renamed from agenty_core. A machine set up before that has
the folder under the old name, and requirements.txt now installs
``-e ../agentY-core`` - so the first start after the update has to get from one
layout to the other without losing the checkout (it holds saved templates) and
without breaking whatever still points at the old path.
"""

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import core_dir  # noqa: E402


def _checkout(folder: Path, marker="x") -> Path:
    (folder / "agenty_core").mkdir(parents=True)
    (folder / "agenty_core" / "__init__.py").write_text("")
    (folder / "saved_template.json").write_text(marker)
    return folder


class Migrating(unittest.TestCase):

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.parent = Path(self._tmp.name)
        self.new = self.parent / "agentY-core"
        self.old = self.parent / "agenty_core"

    def tearDown(self):
        # links first: cleanup must never walk through one into the real folder
        for p in (self.old, self.new):
            if core_dir.is_link(p):
                core_dir.remove_link(p)

    def test_an_old_folder_is_renamed_and_a_link_keeps_its_old_path_working(self):
        _checkout(self.old, "mine")
        self.assertEqual(core_dir.migrate(self.parent), "renamed")
        self.assertTrue(core_dir.is_checkout(self.new))
        self.assertEqual((self.new / "saved_template.json").read_text(), "mine")
        self.assertTrue(core_dir.is_link(self.old))
        self.assertEqual((self.old / "saved_template.json").read_text(), "mine",
                         "paths that still say agenty_core must keep resolving")
        self.assertEqual(core_dir.find(self.parent, env={}), self.new)

    def test_it_is_safe_to_run_again(self):
        _checkout(self.old)
        core_dir.migrate(self.parent)
        self.assertEqual(core_dir.migrate(self.parent), "current")
        self.assertTrue(core_dir.is_checkout(self.new))
        self.assertTrue(core_dir.is_link(self.old))

    def test_a_folder_that_cannot_be_renamed_is_linked_instead(self):
        """Windows refuses the rename while a program has a file in it open."""
        _checkout(self.old, "busy")
        with mock.patch.object(core_dir.os, "rename", side_effect=PermissionError("in use")):
            self.assertEqual(core_dir.migrate(self.parent), "linked")
        self.assertTrue(core_dir.is_checkout(self.old), "the folder itself is untouched")
        self.assertTrue(core_dir.is_link(self.new))
        self.assertEqual((self.new / "saved_template.json").read_text(), "busy")
        self.assertEqual(core_dir.find(self.parent, env={}), self.new)

    def test_the_rename_is_finished_on_a_later_start(self):
        _checkout(self.old, "later")
        with mock.patch.object(core_dir.os, "rename", side_effect=PermissionError("in use")):
            core_dir.migrate(self.parent)
        self.assertEqual(core_dir.migrate(self.parent), "renamed")
        self.assertTrue(core_dir.is_checkout(self.new))
        self.assertTrue(core_dir.is_link(self.old))
        self.assertEqual((self.new / "saved_template.json").read_text(), "later")

    def test_still_busy_on_the_later_start_keeps_the_link(self):
        _checkout(self.old)
        with mock.patch.object(core_dir.os, "rename", side_effect=PermissionError("in use")):
            core_dir.migrate(self.parent)
            self.assertEqual(core_dir.migrate(self.parent), "linked")
        self.assertTrue(core_dir.is_link(self.new))
        self.assertTrue(core_dir.is_checkout(self.old))

    def test_a_new_install_and_an_empty_machine_are_left_alone(self):
        self.assertEqual(core_dir.migrate(self.parent), "current")
        self.assertFalse(self.new.exists())
        _checkout(self.new)
        self.assertEqual(core_dir.migrate(self.parent), "current")
        self.assertFalse(os.path.lexists(str(self.old)), "no link is made where none is needed")

    def test_both_folders_present_touches_neither(self):
        _checkout(self.new, "new")
        _checkout(self.old, "old")
        self.assertEqual(core_dir.migrate(self.parent), "current")
        self.assertEqual((self.old / "saved_template.json").read_text(), "old")
        self.assertEqual((self.new / "saved_template.json").read_text(), "new")

    def test_something_else_under_the_new_name_is_not_overwritten(self):
        _checkout(self.old)
        self.new.mkdir()
        (self.new / "notes.txt").write_text("not a checkout")
        self.assertEqual(core_dir.migrate(self.parent), "failed")
        self.assertEqual((self.new / "notes.txt").read_text(), "not a checkout")
        self.assertTrue(core_dir.is_checkout(self.old))

    def test_only_a_link_is_ever_removed(self):
        _checkout(self.new)
        with self.assertRaises(OSError):
            core_dir.remove_link(self.new)
        self.assertTrue(core_dir.is_checkout(self.new))


class Finding(unittest.TestCase):

    def test_the_new_name_is_preferred_and_the_old_one_still_found(self):
        with tempfile.TemporaryDirectory() as d:
            parent = Path(d)
            self.assertIsNone(core_dir.find(parent, env={}))
            _checkout(parent / "agenty_core")
            self.assertEqual(core_dir.find(parent, env={}), parent / "agenty_core")
            _checkout(parent / "agentY-core")
            self.assertEqual(core_dir.find(parent, env={}), parent / "agentY-core")

    def test_the_override_wins(self):
        with tempfile.TemporaryDirectory() as d:
            parent = Path(d)
            _checkout(parent / "agentY-core")
            _checkout(parent / "elsewhere")
            self.assertEqual(core_dir.find(parent, env={"AGENTY_CORE_DIR": str(parent / "elsewhere")}),
                             parent / "elsewhere")

    def test_this_machine_has_one(self):
        self.assertIsNotNone(core_dir.find())


class EverythingAgrees(unittest.TestCase):
    """The places that name the folder, by reading them."""

    def test_requirements_installs_the_new_name(self):
        text = (ROOT / "requirements.txt").read_text(encoding="utf-8")
        self.assertIn("\n-e ../agentY-core\n", text)
        self.assertNotIn("\n-e ../agenty_core\n", text)

    def test_the_dependency_sync_moves_the_folder_before_it_installs(self):
        text = (ROOT / "scripts" / "sync_deps.py").read_text(encoding="utf-8")
        self.assertLess(text.index("core_dir.migrate()"), text.index("install_command(sys.executable)"))

    def test_launchers_and_installers_clone_and_update_the_new_name(self):
        for name in ("install_agent.ps1", "install_agent.sh"):
            text = (ROOT / name).read_text(encoding="utf-8", errors="replace")
            with self.subTest(script=name):
                self.assertIn("github.com/szprivate/agentY-core.git", text)
                self.assertNotIn("szprivate/agenty_core", text)
                self.assertIn("core_dir.py", text)
        for name in ("run_agent.ps1", "run_agent.sh"):
            text = (ROOT / name).read_text(encoding="utf-8", errors="replace")
            with self.subTest(script=name):
                self.assertIn("agentY-core", text)
                self.assertIn("agenty_core", text, "an install not yet moved over is still updated")

    def test_the_import_fallbacks_look_under_both_names(self):
        for rel in ("src/__init__.py", "scripts/check_env.py"):
            text = (ROOT / rel).read_text(encoding="utf-8")
            with self.subTest(file=rel):
                self.assertIn('"agentY-core"', text)
                self.assertIn('"agenty_core"', text)


if __name__ == "__main__":
    unittest.main()
