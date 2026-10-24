"""scripts/install_deadline_plugin.py against a temporary 'repository'. The real one is never touched."""

import importlib.util
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

_SPEC = importlib.util.spec_from_file_location(
    "install_deadline_plugin", Path(__file__).resolve().parent.parent / "scripts" / "install_deadline_plugin.py")
inst = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(inst)


class InstallTest(unittest.TestCase):

    def setUp(self):
        d = tempfile.TemporaryDirectory()
        self.addCleanup(d.cleanup)
        self.repo = Path(d.name) / "deadline"
        self.repo.mkdir()
        self.target = self.repo / "custom" / "plugins" / "ComfyUI"
        self.log = []

    def _install(self, **kw):
        return inst.install(self.repo, log=self.log.append, **kw)

    def test_the_source_has_the_three_plugin_files(self):
        for f in inst.PLUGIN_FILES:
            self.assertTrue((inst.SOURCE / f).is_file(), f)

    def test_a_dry_run_writes_nothing(self):
        p = self._install(dry_run=True)
        self.assertFalse(self.target.exists())
        self.assertEqual(sorted(p["new"]), sorted(inst.PLUGIN_FILES))
        self.assertIn("dry run", "\n".join(self.log))

    def test_a_fresh_install_copies_the_files_exactly(self):
        self._install()
        for f in inst.PLUGIN_FILES:
            self.assertEqual((self.target / f).read_bytes(), (inst.SOURCE / f).read_bytes())
        self.assertFalse(list(self.target.rglob("__pycache__")))

    def test_installing_twice_is_a_no_op(self):
        self._install()
        self.log.clear()
        p = self._install()
        self.assertIn("already up to date", "\n".join(self.log))
        self.assertEqual((p["new"], p["changed"], p["removed"]), ([], [], []))
        self.assertFalse((self.repo / "custom" / "_backup").exists())

    def test_a_changed_plugin_is_backed_up_outside_the_plugins_folder(self):
        self._install()
        (self.target / "ComfyUI.param").write_text("edited on the farm")
        p = self._install()
        self.assertEqual(p["changed"], ["ComfyUI.param"])
        backups = list((self.repo / "custom" / "_backup").iterdir())
        self.assertEqual(len(backups), 1)
        self.assertEqual((backups[0] / "ComfyUI.param").read_text(), "edited on the farm")
        self.assertEqual((self.target / "ComfyUI.param").read_bytes(),
                         (inst.SOURCE / "ComfyUI.param").read_bytes())
        self.assertEqual(sorted(x.name for x in (self.repo / "custom" / "plugins").iterdir()), ["ComfyUI"])

    def test_a_file_the_source_no_longer_has_goes_with_the_backup(self):
        self._install()
        (self.target / "old_helper.py").write_text("x")
        p = self._install()
        self.assertEqual(p["removed"], ["old_helper.py"])
        self.assertFalse((self.target / "old_helper.py").exists())

    def test_an_unreachable_repository_stops_with_a_message(self):
        with self.assertRaises(SystemExit) as cm:
            inst.install(self.repo / "nope", log=self.log.append)
        self.assertIn("not reachable", str(cm.exception))

    def test_uninstall_moves_it_aside_rather_than_deleting(self):
        self._install()
        self.assertTrue(inst.uninstall(self.repo, log=self.log.append))
        self.assertFalse(self.target.exists())
        self.assertEqual(len(list((self.repo / "custom" / "_backup").iterdir())), 1)
        self.assertFalse(inst.uninstall(self.repo, log=self.log.append))

    def test_the_repository_comes_from_the_flag_then_the_setting_then_deadlinecommand(self):
        self.assertEqual(inst.find_repository("X:/repo"), ("X:/repo", "--repository"))
        with mock.patch.dict(os.environ, {"DEADLINE_REPOSITORY": "Y:/repo"}):
            self.assertEqual(inst.find_repository(""), ("Y:/repo", "DEADLINE_REPOSITORY"))
        with mock.patch.dict(os.environ, {"DEADLINE_REPOSITORY": ""}), \
                mock.patch.object(inst, "_env", return_value=""), \
                mock.patch.object(inst, "_deadlinecommand", return_value=""):
            path, why = inst.find_repository("")
        self.assertEqual(path, "")
        self.assertIn("--repository", why)

    def test_main_prints_the_pool_commands_and_never_creates_a_pool(self):
        calls = []
        with mock.patch("subprocess.run", side_effect=lambda *a, **k: calls.append(a)), \
                mock.patch("builtins.print") as pr:
            rc = inst.main(["--repository", str(self.repo)])
        self.assertEqual(rc, 0)
        self.assertEqual(calls, [], "installing must not run deadlinecommand when the repository is given")
        printed = "\n".join(str(c.args[0]) for c in pr.call_args_list if c.args)
        self.assertIn("-AddPool comfyui", printed)


if __name__ == "__main__":
    unittest.main()
