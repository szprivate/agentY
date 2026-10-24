"""The Deadline farm tools are wired into the orchestrator, their own pack, the skill scope and the settings keys."""
import unittest

_NAMES = ["submit_to_deadline", "deadline_job_status", "deadline_farm_info"]


class Registered(unittest.TestCase):
    def test_tools_pack_skill_and_keys(self):
        import src.tools as tools
        from src import agent
        from src.tools import tool_packs
        from src.utils import agentY_server
        picked = {tool_packs.name_of(t) for t in tools.ORCHESTRATOR_TOOLS}
        for n in _NAMES:
            self.assertIn(n, tools._SHARED_CORE_TOOLS)
            self.assertIn(n, picked)
            self.assertIn(n, tool_packs.PACKS["farm"][1])
        self.assertIn("deadline-farm", agent._ORCH_SKILL_NAMES)
        self.assertTrue(agent._skill_sources(["deadline-farm"]))
        for k in ("DEADLINE_COMFYUI_POOL", "DEADLINE_OUTPUT_DIR", "DEADLINE_WEBSERVICE_URL",
                  "DEADLINE_ROUTE_RUNS", "DEADLINE_FETCH_DIR", "DEADLINE_MAX_WAIT_MINUTES",
                  "SLACK_ALLOWED_CHANNELS", "SLACK_APPROVERS"):
            self.assertIn(k, agentY_server._KNOWN_ENV_KEYS)


if __name__ == "__main__":
    unittest.main()
