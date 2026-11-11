"""animate_character is wired into the orchestrator, its characters pack and its skill scope."""
import unittest


class Registered(unittest.TestCase):
    def test_tool_pack_and_skill(self):
        import src.tools as tools
        from src import agent
        from src.tools import tool_packs
        self.assertIn("animate_character", tools._SHARED_CORE_TOOLS)
        names = {tool_packs.name_of(t) for t in tools.ORCHESTRATOR_TOOLS}
        self.assertIn("animate_character", names)
        self.assertIn("animate_character", tool_packs.PACKS["characters"][1])
        self.assertIn("character-animation", agent._ORCH_SKILL_NAMES)
        self.assertTrue(agent._skill_sources(["character-animation"]))


if __name__ == "__main__":
    unittest.main()
