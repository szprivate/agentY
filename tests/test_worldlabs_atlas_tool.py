"""The World Labs Atlas tools are wired into the orchestrator, the worlds pack, the skill scope and the key list."""
import unittest

_NAMES = ["atlas_create_world", "atlas_expand_world", "atlas_rebuild_world_mesh", "atlas_splat_to_mesh",
          "atlas_fetch", "atlas_list"]


class Registered(unittest.TestCase):
    def test_tools_pack_skill_and_key(self):
        import src.tools as tools
        from src import agent
        from src.tools import tool_packs
        from src.utils import agentY_server
        picked = {tool_packs.name_of(t) for t in tools.ORCHESTRATOR_TOOLS}
        for n in _NAMES:
            self.assertIn(n, tools._SHARED_CORE_TOOLS)
            self.assertIn(n, picked)
            self.assertIn(n, tool_packs.PACKS["worlds"][1])
        self.assertIn("worldlabs-atlas", agent._ORCH_SKILL_NAMES)
        self.assertTrue(agent._skill_sources(["worldlabs-atlas"]))
        self.assertIn("WORLDLABS_API_KEY", agentY_server._KNOWN_ENV_KEYS)


if __name__ == "__main__":
    unittest.main()
