"""MCP servers start on any machine, not only the one mcp.json was written on.

config/mcp.json is tracked, and it named the bEpic Worlds pack by this
machine's path (D:/AI/comfyui/custom_nodes/ComfyUI-bEpicWorlds, run by
D:/ai/comfyui/.venv/Scripts/python.exe). On a Mac that folder doesn't exist,
so the server failed to start with "the repo can't be found". Now the entry
says ${PYTHON} and ${BEPIC_WORLDS_DIR}, worked out on the machine that runs it.

    python -m unittest tests.test_mcp_portable_paths
"""

import json
import os
import re
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from src.tools import mcp_tools as mt

ROOT = Path(__file__).resolve().parent.parent


class Base(unittest.TestCase):
    def setUp(self):
        mt._BUILTIN_CACHE.clear()
        self.env = mock.patch.dict(os.environ, {}, clear=False)
        self.env.start()
        for var in ("BEPIC_WORLDS_DIR", "COMFYUI_PATH", "COMFYUI_DIR", "PYTHON"):
            os.environ.pop(var, None)

    def tearDown(self):
        self.env.stop()
        mt._BUILTIN_CACHE.clear()


class TheSharedConfig(unittest.TestCase):

    def test_the_worlds_server_names_no_machines_path(self):
        # (This machine's own copy may differ from the shared one; the Worlds
        # entry, when there, must be the portable one.)
        cfg = json.loads((ROOT / "config" / "mcp.json").read_text(encoding="utf-8"))
        worlds = cfg["servers"].get("bepic_worlds")
        if worlds:
            self.assertEqual(worlds["command"], "${PYTHON}")
            self.assertEqual(worlds["cwd"], "${BEPIC_WORLDS_DIR}")
            self.assertIsNone(re.search(r"[A-Za-z]:[\\/]", json.dumps(worlds)))


class FindingThePack(Base):

    def _comfy(self, root: Path):
        nodes = root / "custom_nodes"
        pack = nodes / "comfyui-bepicworlds"            # the registry's lower-case name
        (pack / "bepic_worlds").mkdir(parents=True)
        (pack / "bepic_worlds" / "mcp_server.py").write_text("")
        (nodes / "other_pack").mkdir()
        return nodes, pack

    def test_found_in_the_running_comfyuis_custom_nodes_whatever_its_folder_is_called(self):
        with tempfile.TemporaryDirectory() as d:
            nodes, pack = self._comfy(Path(d))
            client = mock.Mock(get=lambda path: {"custom_nodes": [str(nodes)]}, base_url="http://host:8188")
            with mock.patch("agenty_core.utils.comfyui_client.get_client", return_value=client):
                self.assertEqual(mt._expand("${BEPIC_WORLDS_DIR}"), str(pack))
                self.assertEqual(mt._expand("${COMFYUI_URL}"), "http://host:8188")
            self.assertEqual(mt._expand("${PYTHON}"), __import__("sys").executable)

    def test_comfyui_path_works_without_a_running_comfyui(self):
        with tempfile.TemporaryDirectory() as d:
            _nodes, pack = self._comfy(Path(d))
            os.environ["COMFYUI_PATH"] = d
            with mock.patch("agenty_core.utils.comfyui_client.get_client", side_effect=RuntimeError("down")):
                self.assertEqual(mt._stdio_cwd({"cwd": "${BEPIC_WORLDS_DIR}"}), str(pack))

    def test_an_explicit_setting_wins(self):
        with tempfile.TemporaryDirectory() as d:
            os.environ["BEPIC_WORLDS_DIR"] = d
            self.assertEqual(mt._stdio_cwd({"cwd": "${BEPIC_WORLDS_DIR}"}), d)

    def test_not_found_is_a_clear_error_not_another_machines_path(self):
        with mock.patch("agenty_core.utils.comfyui_client.get_client", side_effect=RuntimeError("down")):
            with self.assertRaises(FileNotFoundError) as ctx:
                mt._stdio_cwd({"cwd": "${BEPIC_WORLDS_DIR}"})
        self.assertIn("set BEPIC_WORLDS_DIR", str(ctx.exception))

    def test_a_missing_pack_leaves_the_other_servers_running(self):
        cfg = {"servers": {"bepic_worlds": {"enabled": True, "transport": "stdio", "command": "${PYTHON}",
                                            "args": ["-m", "bepic_worlds.mcp_server"], "cwd": "${BEPIC_WORLDS_DIR}"}}}
        with mock.patch.object(mt, "load_mcp_config", return_value=cfg), \
                mock.patch("agenty_core.utils.comfyui_client.get_client", side_effect=RuntimeError("down")), \
                mock.patch.dict(mt._CLIENTS, {}, clear=True):
            self.assertEqual(mt.load_mcp_tools(), [])
        self.assertIn("BEPIC_WORLDS_DIR", mt._STATUS["bepic_worlds"])


class Bundles(Base):

    def test_a_relative_folder_is_agentys_own(self):
        rel = "config"
        self.assertEqual(mt._stdio_cwd({"cwd": rel}), str(ROOT / rel))

    def test_a_bundle_recorded_on_another_machine_runs_from_its_own_folder(self):
        sc = {"cwd": "D:\\\\AI\\\\agentY\\\\config\\\\mcp_bundles\\\\blender" if os.name != "nt" else "Q:/nowhere/blender",
              "bundle": {"dir": "config"}}
        self.assertEqual(mt._stdio_cwd(sc), str(ROOT / "config"))


if __name__ == "__main__":
    unittest.main()
