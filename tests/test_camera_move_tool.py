"""render_camera_move is wired into the orchestrator, its worlds pack and its skill scope."""
import unittest


class Registered(unittest.TestCase):
    def test_tool_pack_and_skill(self):
        import src.tools as tools
        from src import agent
        from src.tools import tool_packs
        self.assertIn("render_camera_move", tools._SHARED_CORE_TOOLS)
        names = {tool_packs.name_of(t) for t in tools.ORCHESTRATOR_TOOLS}
        self.assertIn("render_camera_move", names)
        self.assertIn("render_camera_move", tool_packs.PACKS["worlds"][1])
        self.assertIn("camera-move-control", agent._ORCH_SKILL_NAMES)
        self.assertTrue(agent._skill_sources(["camera-move-control"]))


if __name__ == "__main__":
    unittest.main()
