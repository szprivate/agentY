"""The preset scripts run as plain scripts (the installers call ``python scripts/mcp_presets.py``).

Another folder called ``scripts`` earlier on the path (a dependency's stray top-level folder, a second
checkout) used to win ``from scripts import mcp_setup`` and the installer's MCP setup stopped with
"cannot import name 'mcp_setup' from 'scripts' (unknown location)".
"""
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


class ForeignScriptsFolder(unittest.TestCase):
    def _run(self, *args):
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "scripts").mkdir()          # a namespace "scripts" with nothing in it
            env = dict(os.environ, PYTHONPATH=tmp)
            return subprocess.run([sys.executable, *args], cwd=tmp, env=env,
                                  capture_output=True, text=True, timeout=120)

    def test_preset_setup_still_imports_its_runner(self):
        r = self._run(str(ROOT / "scripts" / "mcp_presets.py"), "setup", "blender", "--dry-run")
        self.assertNotIn("ImportError", r.stderr)
        self.assertNotIn("cannot import name", r.stderr)

    def test_recipe_draft_still_imports(self):
        r = self._run(str(ROOT / "scripts" / "mcp_draft.py"), "--help")
        self.assertEqual(r.returncode, 0, r.stderr)


if __name__ == "__main__":
    unittest.main()
