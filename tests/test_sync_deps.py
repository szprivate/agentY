"""The launchers bring the venv in line with requirements.txt on every start.

They used to reinstall only when their own update step pulled a dependency
change. A `git pull` done by hand never told them, so a machine set up before
OpenCV and scenedetect were added started without both, with only a warning.
scripts/sync_deps.py now decides from the environment itself: a changed
dependency file (by the update, or since this venv was last installed), or a
required package that will not import.

    python -m unittest discover -s tests
"""

import importlib.util
import io
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("sync_deps", ROOT / "scripts" / "sync_deps.py")
sd = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sd)


class ReasonsTest(unittest.TestCase):

    def test_nothing_to_do(self):
        self.assertEqual(sd.reasons("abc", "abc", [], ""), [])

    def test_a_venv_seen_for_the_first_time_is_not_reinstalled(self):
        self.assertEqual(sd.reasons(None, "abc", [], ""), [])

    def test_changed_files_since_the_last_install(self):
        self.assertEqual(len(sd.reasons("old", "new", [], "")), 1)

    def test_a_missing_package_is_named(self):
        why = sd.reasons("abc", "abc", ["cv2", "scenedetect"], "")
        self.assertEqual(why, ["missing: cv2, scenedetect"])

    def test_the_update_step_saying_so(self):
        self.assertIn("agentY", sd.reasons("abc", "abc", [], "agentY")[0])


class FingerprintTest(unittest.TestCase):

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.dir = Path(tmp.name)

    def _fp(self, data: bytes):
        path = self.dir / "requirements.txt"
        path.write_bytes(data)
        return sd.fingerprint((path,))

    def test_line_endings_do_not_count(self):
        self.assertEqual(self._fp(b"a\nb\n"), self._fp(b"a\r\nb\r\n"))

    def test_a_new_requirement_does(self):
        self.assertNotEqual(self._fp(b"a\n"), self._fp(b"a\nscenedetect>=0.7.0\n"))

    def test_an_absent_file_is_not_an_error(self):
        self.assertTrue(sd.fingerprint((self.dir / "nope.toml",)))


class MainTest(unittest.TestCase):

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.stamp = Path(tmp.name) / ".agenty-deps"
        self.enterContext(mock.patch.object(sd, "stamp_path", lambda: self.stamp))
        self.enterContext(mock.patch.object(sd, "fingerprint", lambda files=None: "fp-now"))
        self.enterContext(mock.patch.object(sd.check_env, "main", lambda argv: 0))
        self.enterContext(mock.patch.object(sd.shutil, "which", lambda name: "C:/uv.exe"))

    def _run(self, missing, *, code=0, argv=("--quiet",)):
        calls = []

        def run(cmd, cwd=None):
            calls.append((cmd, cwd))
            return mock.Mock(returncode=code)

        out = io.StringIO()
        with mock.patch.object(sd, "missing_required", lambda: missing), \
             mock.patch.object(sd.subprocess, "run", run), redirect_stdout(out):
            result = sd.main(list(argv))
        return result, calls, out.getvalue()

    def test_missing_packages_are_installed_into_this_venv(self):
        _, calls, out = self._run(["cv2"])
        self.assertEqual(len(calls), 1)
        cmd, cwd = calls[0]
        self.assertEqual(cmd, ["uv", "pip", "install", "--python", sys.executable,
                               "-r", "requirements.txt"])
        self.assertEqual(cwd, str(sd.ROOT), "-e ../agenty_core is relative to agentY")
        self.assertIn("missing: cv2", out)
        self.assertEqual(self.stamp.read_text().strip(), "fp-now")

    def test_a_hand_pulled_version_bump_is_installed(self):
        self.stamp.write_text("fp-before\n")
        _, calls, _ = self._run([])
        self.assertEqual(len(calls), 1)

    def test_a_failed_install_leaves_the_stamp_alone(self):
        self.stamp.write_text("fp-before\n")
        _, calls, out = self._run(["cv2"], code=1)
        self.assertEqual(len(calls), 1)
        self.assertEqual(self.stamp.read_text().strip(), "fp-before", "so the next start tries again")
        self.assertIn("returned 1", out)

    def test_a_first_look_with_nothing_missing_only_remembers(self):
        _, calls, out = self._run([])
        self.assertEqual(calls, [])
        self.assertEqual(self.stamp.read_text().strip(), "fp-now")
        self.assertEqual(out, "", "quiet means quiet")

    def test_in_line_means_nothing_happens(self):
        self.stamp.write_text("fp-now\n")
        _, calls, out = self._run([])
        self.assertEqual((calls, out), ([], ""))

    def test_an_empty_changed_list_is_no_reason(self):
        self.stamp.write_text("fp-now\n")
        _, calls, _ = self._run([], argv=("--quiet", "--changed="))
        self.assertEqual(calls, [])

    def test_without_uv_it_uses_this_venvs_pip(self):
        with mock.patch.object(sd.shutil, "which", lambda name: None):
            _, calls, _ = self._run(["cv2"])
        self.assertEqual(calls[0][0][:3], [sys.executable, "-m", "pip"])


class LaunchersTest(unittest.TestCase):
    """Both launchers hand the decision to sync_deps.py, with the venv's python."""

    def test_windows(self):
        text = (ROOT / "run_agent.ps1").read_text(encoding="utf-8")
        self.assertIn("scripts\\sync_deps.py", text)
        self.assertIn("& $venvPy $syncDeps", text)

    def test_macos_and_linux(self):
        text = (ROOT / "run_agent.sh").read_text(encoding="utf-8")
        self.assertIn("scripts/sync_deps.py", text)
        self.assertIn('"$VENV_PY" "$PROJECT_ROOT/scripts/sync_deps.py"', text)


if __name__ == "__main__":
    unittest.main()
