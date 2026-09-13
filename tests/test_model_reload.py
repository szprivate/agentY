"""Changing a model takes effect without restarting the host.

Most agents are built once, when the host starts, so a model picked in Settings did
nothing until a restart, and /switch_model rebuilt four of them and said the rest
would "apply on the next agent start". Every role resolves its model through
role_model, so a fingerprint before a change and one after name exactly the agents
that are now wrong; those are rebuilt in place. A running turn is still using its
agents, so a switch made mid-turn waits for the turn to end.

    python -m unittest discover -s tests
"""

import unittest
from types import SimpleNamespace
from unittest import mock

import src.agent as agents
from src.utils import agentY_server as srv
from src.utils import model_reload as mr


def fp(**over):
    """A fingerprint where every role runs the same model, with *over* changed."""
    base = {role: "claude,claude-haiku-4-5"
            for deps in mr.LIVE_AGENTS.values() for role in deps}
    base.update({"qa_checker": "claude,claude-sonnet-4-6", "coder": "claude,claude-sonnet-4-6",
                 mr.PROVIDERS: "p1"})
    base.update(over)
    return base


class WhatChanged(unittest.TestCase):

    def test_nothing_changed_is_nothing_to_rebuild(self):
        self.assertEqual(mr.changed_agents(fp(), fp()), [])

    def test_a_model_change_names_the_agent_it_belongs_to(self):
        self.assertEqual(mr.changed_agents(fp(), fp(orchestrator="dashscope,qwen3.6-plus")),
                         ["orchestrator"])

    def test_a_fallback_moves_everything_that_leans_on_it(self):
        changed = mr.changed_agents(fp(), fp(llm_functions="ollama,qwen3.5:9b"))
        self.assertEqual(sorted(changed), ["info", "planner", "query_templates", "search_web"])

    def test_the_repair_specialists_follow_the_assembler(self):
        changed = mr.changed_agents(fp(), fp(assemble_workflow="claude,claude-sonnet-4-6"))
        self.assertEqual(sorted(changed), ["fix_workflow_assembly", "generate_new_workflow"])

    def test_a_per_call_role_needs_no_rebuild(self):
        """QA, learnings and coder are built fresh on every call already."""
        self.assertEqual(mr.changed_agents(fp(), fp(qa_checker="openai,gpt-5")), [])

    def test_a_provider_setting_rebuilds_everything(self):
        self.assertEqual(mr.changed_agents(fp(), fp(**{mr.PROVIDERS: "p2"})),
                         list(mr.LIVE_AGENTS))

    def test_a_changed_api_key_is_a_client_change(self):
        self.assertTrue(mr.env_affects_clients(["DASHSCOPE_API_KEY"]))
        self.assertTrue(mr.env_affects_clients(["anthropic_api_key"]))
        self.assertFalse(mr.env_affects_clients(["SLACK_BOT_TOKEN", "HF_TOKEN"]))
        self.assertFalse(mr.env_affects_clients([]))


class Fingerprinting(unittest.TestCase):

    def fingerprint(self, models, llm):
        with mock.patch.object(agents, "role_model",
                               side_effect=lambda role, *a, **k: models.get(role, "")), \
                mock.patch.object(agents, "_settings", return_value={"llm": llm}):
            return mr.fingerprint()

    def test_it_reads_every_role(self):
        out = self.fingerprint({"orchestrator": "claude,x"}, {})
        self.assertEqual(out["orchestrator"], "claude,x")
        for role in agents._ROLE_TIERS:
            self.assertIn(role, out)

    def test_a_tier_change_is_a_model_change_not_a_provider_change(self):
        a = self.fingerprint({}, {"tiers": {"orchestrator": "claude,x"},
                                  "dashscope": {"base_url": "u"}})
        b = self.fingerprint({}, {"tiers": {"orchestrator": "claude,y"},
                                  "dashscope": {"base_url": "u"}})
        self.assertEqual(a[mr.PROVIDERS], b[mr.PROVIDERS])

    def test_an_endpoint_change_is_a_provider_change(self):
        a = self.fingerprint({}, {"dashscope": {"base_url": "https://one"}})
        b = self.fingerprint({}, {"dashscope": {"base_url": "https://two"}})
        self.assertNotEqual(a[mr.PROVIDERS], b[mr.PROVIDERS])


class _Pipeline:
    def __init__(self):
        self._orchestrator_agent = SimpleNamespace(
            name="old", messages=[{"role": "user", "content": [{"text": "a cat"}]}])
        self._delegation_tools = ["prepare_workflow"]
        self._researcher = "old"
        self._info_agent = "old"
        self._planner_agent = "old"
        self._search_web_agent = "old"
        self._fix_agent = "old"
        self._generate_agent = "old"
        self.vision = []
        self.video = []

    def set_orchestrator(self, agent):
        self._orchestrator_agent = agent

    def _init_vision_agent(self, *, strict=False):
        self.vision.append(strict)

    def _init_video_agent(self, *, strict=False):
        self.video.append(strict)


class Rebuilding(unittest.TestCase):

    def setUp(self):
        self.pipe = _Pipeline()
        self.built = []

        def factory(label):
            def make(**kwargs):
                self.built.append((label, kwargs))
                return SimpleNamespace(name="new-" + label, messages=[])
            return make

        for attr, label in (("create_orchestrator_agent", "orchestrator"),
                            ("create_query_templates_agent", "query_templates"),
                            ("create_info_agent", "info"),
                            ("create_planner_agent", "planner"),
                            ("create_search_web_agent", "search_web")):
            patcher = mock.patch.object(agents, attr, side_effect=factory(label))
            patcher.start()
            self.addCleanup(patcher.stop)

    def test_the_orchestrator_keeps_the_conversation(self):
        """Switching model mid-conversation must not make the agent forget it."""
        self.assertEqual(mr.reload_live_agents(self.pipe, ["orchestrator"]),
                         (["orchestrator"], {}))
        self.assertEqual(self.pipe._orchestrator_agent.name, "new-orchestrator")
        self.assertEqual(self.pipe._orchestrator_agent.messages,
                         [{"role": "user", "content": [{"text": "a cat"}]}])

    def test_the_orchestrator_is_given_its_delegation_tools(self):
        mr.reload_live_agents(self.pipe, ["orchestrator"])
        self.assertEqual(self.built[0][1]["extra_tools"], ["prepare_workflow"])

    def test_specialists_are_swapped_in_place(self):
        mr.reload_live_agents(self.pipe, ["query_templates", "info", "planner", "search_web"])
        self.assertEqual(
            (self.pipe._researcher.name, self.pipe._info_agent.name,
             self.pipe._planner_agent.name, self.pipe._search_web_agent.name),
            ("new-query_templates", "new-info", "new-planner", "new-search_web"))

    def test_lazy_specialists_are_dropped_to_be_built_on_next_use(self):
        mr.reload_live_agents(self.pipe, ["fix_workflow_assembly", "generate_new_workflow"])
        self.assertIsNone(self.pipe._fix_agent)
        self.assertIsNone(self.pipe._generate_agent)

    def test_vision_and_video_are_rebuilt_strictly(self):
        mr.reload_live_agents(self.pipe, ["vision_agent", "video_agent"])
        self.assertEqual((self.pipe.vision, self.pipe.video), ([True], [True]))

    def test_a_failed_rebuild_keeps_the_old_agent(self):
        with mock.patch.object(agents, "create_info_agent", side_effect=RuntimeError("no key")):
            rebuilt, failures = mr.reload_live_agents(self.pipe, ["info", "planner"])
        self.assertEqual(self.pipe._info_agent, "old")
        self.assertEqual(rebuilt, ["planner"])
        self.assertIn("no key", failures["info"])

    def test_a_role_that_is_not_held_live_is_skipped(self):
        self.assertEqual(mr.reload_live_agents(self.pipe, ["qa_checker"]), ([], {}))


class DuringATurn(unittest.TestCase):

    BEFORE = fp()
    AFTER = fp(orchestrator="dashscope,qwen3.6-plus")

    def setUp(self):
        self.registry = {}
        for target, name, value in ((srv, "_run_registry", self.registry),
                                    (srv, "_agent_ref", SimpleNamespace()),
                                    (srv, "_pending_model_before", None),
                                    (srv, "_pending_model_force", False)):
            patcher = mock.patch.object(target, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)
        patcher = mock.patch.object(mr, "fingerprint", return_value=self.AFTER)
        patcher.start()
        self.addCleanup(patcher.stop)
        patcher = mock.patch.object(mr, "reload_live_agents", return_value=(["orchestrator"], {}))
        self.reload = patcher.start()
        self.addCleanup(patcher.stop)

    def test_idle_it_applies_at_once(self):
        self.assertEqual(srv._apply_model_change(self.BEFORE)["state"], "applied")
        self.reload.assert_called_once()
        self.assertEqual(self.reload.call_args.args[1], ["orchestrator"])

    def test_during_a_turn_it_waits_for_the_turn_to_end(self):
        self.registry["r1"] = {"thread_id": "t1"}
        self.assertEqual(srv._apply_model_change(self.BEFORE)["state"], "deferred")
        self.reload.assert_not_called()
        self.registry.clear()
        srv._apply_pending_model_change()
        self.reload.assert_called_once()
        self.assertEqual(self.reload.call_args.args[1], ["orchestrator"])

    def test_two_switches_in_one_turn_are_measured_from_the_first(self):
        self.registry["r1"] = {"thread_id": "t1"}
        srv._apply_model_change(self.BEFORE)
        srv._apply_model_change(self.AFTER)      # a second switch, same turn
        self.registry.clear()
        srv._apply_pending_model_change()
        self.assertEqual(self.reload.call_args.args[1], ["orchestrator"])

    def test_a_pending_change_is_applied_once(self):
        self.registry["r1"] = {"thread_id": "t1"}
        srv._apply_model_change(self.BEFORE)
        self.registry.clear()
        srv._apply_pending_model_change()
        srv._apply_pending_model_change()
        self.reload.assert_called_once()

    def test_a_changed_key_rebuilds_every_live_agent(self):
        srv._apply_model_change(self.BEFORE, force=True)
        self.assertEqual(self.reload.call_args.args[1], list(mr.LIVE_AGENTS))

    def test_without_a_pipeline_there_is_nothing_to_rebuild(self):
        with mock.patch.object(srv, "_agent_ref", None):
            self.assertEqual(srv._apply_model_change(self.BEFORE)["state"], "no_pipeline")
        self.reload.assert_not_called()


class TheSwitchModelReply(unittest.TestCase):
    """What /switch_model tells you — it used to promise a restart."""

    def reply(self, change):
        with mock.patch("src.utils.settings.set_local"), \
                mock.patch("src.utils.settings.load_settings", return_value={}), \
                mock.patch.object(mr, "fingerprint", return_value=fp()), \
                mock.patch.object(srv, "_apply_model_change", return_value=change):
            [event] = srv._switch_model(["orchestrator", "claude,claude-haiku-4-5"])
        return event["data"]

    def test_it_says_the_change_is_live(self):
        said = self.reply({"state": "applied", "rebuilt": ["orchestrator"], "failures": {}})
        self.assertIn("Live now", said)
        self.assertNotIn("next agent start", said)

    def test_mid_turn_it_says_when_it_will_land(self):
        said = self.reply({"state": "deferred", "rebuilt": [], "failures": {}})
        self.assertIn("the moment it finishes", said)

    def test_a_failed_rebuild_is_reported(self):
        said = self.reply({"state": "applied", "rebuilt": [],
                           "failures": {"orchestrator": "bad key"}})
        self.assertIn("bad key", said)


if __name__ == "__main__":
    unittest.main()
