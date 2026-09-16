"""A defaults file that will not parse is reported, not swallowed.

settings.default.toml holds every committed setting. When it failed to parse,
the loader returned an empty dict and said nothing, so the settings panel showed
only what settings.local.json happened to contain: on one machine, a Canvas
section with a single checkbox and no hint that six others had not loaded. A
UTF-8 byte-order mark is enough to cause it (TOML rejects it at line 1, column 1),
and some Windows editors add one on save.

    python -m unittest discover -s tests
"""

import tempfile
import unittest
from pathlib import Path
from unittest import mock

from src.utils import settings as st


class DefaultsProblemTest(unittest.TestCase):

    def _load(self, data: bytes):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "settings.default.toml"
            path.write_bytes(data)
            with mock.patch.object(st, "_DEFAULT_PATH", path):
                return st.load_defaults(), st.defaults_problem()

    def test_a_good_file_has_no_problem(self):
        self.assertEqual(self._load(b"canvas_full_graph = false\n"),
                         ({"canvas_full_graph": False}, ""))

    def test_windows_line_endings_are_fine(self):
        self.assertEqual(self._load(b"a = 1\r\nb = 2\r\n")[0], {"a": 1, "b": 2})

    def test_a_byte_order_mark_is_named_and_logged(self):
        with self.assertLogs("agentY.settings", "WARNING"):
            data, problem = self._load(b"\xef\xbb\xbfa = 1\n")
        self.assertEqual(data, {})
        self.assertIn("byte-order mark", problem)
        self.assertIn("git checkout", problem, "it says how to get the file back")

    def test_a_broken_file_says_why(self):
        with self.assertLogs("agentY.settings", "WARNING"):
            data, problem = self._load(b"a = 1\na = 2\n")
        self.assertEqual(data, {})
        self.assertIn("could not be read", problem)
        self.assertNotIn("byte-order mark", problem)

    def test_a_missing_file_is_reported_too(self):
        with mock.patch.object(st, "_DEFAULT_PATH", Path(tempfile.gettempdir()) / "nope" / "x.toml"), \
             self.assertLogs("agentY.settings", "WARNING"):
            self.assertEqual(st.load_defaults(), {})
            self.assertIn("missing", st.defaults_problem())

    def test_fixing_the_file_clears_the_problem(self):
        with self.assertLogs("agentY.settings", "WARNING"):
            self._load(b"\xef\xbb\xbfa = 1\n")
        self.assertEqual(self._load(b"a = 1\n")[1], "")

    def test_the_real_defaults_file_parses(self):
        """The committed file itself must never be the problem."""
        st.load_defaults()
        self.assertEqual(st.defaults_problem(), "")


if __name__ == "__main__":
    unittest.main()
