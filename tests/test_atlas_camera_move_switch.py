"""The paid Atlas camera move is experimental and off by default ([atlas] camera_move_enabled)."""
import os
import unittest
from unittest import mock

from src.utils import settings


class AtlasCameraMoveSwitchTest(unittest.TestCase):
    def test_default_is_off(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("AGENTY_ATLAS_CAMERA_MOVE", None)
            with mock.patch.object(settings, "load_settings", return_value={}):
                self.assertFalse(settings.atlas_camera_move_enabled())
            with mock.patch.object(settings, "load_settings", return_value={"atlas": {"camera_move_enabled": True}}):
                self.assertTrue(settings.atlas_camera_move_enabled())

    def test_default_toml_has_it_off(self):
        self.assertIs(settings.load_defaults()["atlas"]["camera_move_enabled"], False)

    def test_env_wins(self):
        with mock.patch.object(settings, "load_settings", return_value={"atlas": {"camera_move_enabled": True}}):
            with mock.patch.dict(os.environ, {"AGENTY_ATLAS_CAMERA_MOVE": "0"}):
                self.assertFalse(settings.atlas_camera_move_enabled())

    def test_tool_not_offered_when_off(self):
        from src.tools import ORCHESTRATOR_TOOLS, tool_packs
        names = {tool_packs.name_of(t) for t in ORCHESTRATOR_TOOLS}
        self.assertNotIn("atlas_render_camera_move", names)
        self.assertNotIn("atlas_render_camera_move", tool_packs._PACK_OF)

    def test_host_route_says_disabled(self):
        import inspect
        from src.utils import agentY_server
        self.assertIn("atlas_camera_move_enabled", inspect.getsource(agentY_server))


if __name__ == "__main__":
    unittest.main()
