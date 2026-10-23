"""Optional packages: the installer's answer is kept, and the launcher honours it.

    python -m unittest tests.test_optional_deps
"""
import importlib.util
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("optional_deps", ROOT / "scripts" / "optional_deps.py")
od = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(od)

NAME = "reverse_image_search"


class ChoiceTest(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.path = Path(tmp.name) / "config" / "optional_deps.json"

    def test_unset_by_default(self):
        self.assertEqual(od.read_choices(self.path), {})

    def test_round_trip(self):
        od.write_choice(NAME, True, self.path)
        self.assertEqual(od.read_choices(self.path), {NAME: True})
        od.write_choice(NAME, False, self.path)
        self.assertEqual(od.read_choices(self.path), {NAME: False})

    def test_garbage_file_reads_as_unset(self):
        self.path.parent.mkdir(parents=True)
        self.path.write_text("{not json", encoding="utf-8")
        self.assertEqual(od.read_choices(self.path), {})


class EnsureTest(unittest.TestCase):
    def _run(self, choice, present):
        with mock.patch.object(od, "read_choices", return_value=choice), \
             mock.patch.object(od, "installed", return_value=present), \
             mock.patch.object(od.subprocess, "run") as run:
            run.return_value.returncode = 0
            od.ensure()
            return run

    def test_wanted_and_present_runs_no_pip(self):
        self.assertFalse(self._run({NAME: True}, True).called)

    def test_wanted_and_missing_installs(self):
        run = self._run({NAME: True}, False)
        self.assertTrue(run.called)
        self.assertIn("playwright>=1.40", run.call_args[0][0])

    def test_declined_or_unset_never_installs(self):
        self.assertFalse(self._run({NAME: False}, False).called)
        self.assertFalse(self._run({}, False).called)

    def test_requirements_txt_does_not_carry_it(self):
        text = (ROOT / "requirements.txt").read_text(encoding="utf-8").lower()
        self.assertNotIn("playwright", text)


if __name__ == "__main__":
    unittest.main()
