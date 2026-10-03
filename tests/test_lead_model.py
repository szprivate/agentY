"""Reasoning switched per agent, and the Lead tier a lead conversation runs on.

role_thinking (per role → per tier → off, the providers' own switches forcing it
on), what each provider is then asked for, role_model("lead") falling back to the
orchestrator, the pipeline putting the lead's model under its orchestrator and
back, which turns count as a lead's, and a reasoning change rebuilding only the
agents it concerns.
"""

import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

import src.agent as agents


def _llm(**llm):
    return mock.patch.object(agents, "_settings", return_value={"llm": llm})


class ReasoningPerAgent(unittest.TestCase):

    def test_off_unless_asked(self):
        with _llm():
            self.assertFalse(agents.role_thinking("orchestrator"))
            self.assertFalse(agents.role_thinking("query_templates"))

    def test_a_tier_switches_its_roles(self):
        with _llm(thinking={"research_assembly": True}):
            self.assertTrue(agents.role_thinking("query_templates"))
            self.assertTrue(agents.role_thinking("brain"))          # assemble_workflow's agent
            self.assertFalse(agents.role_thinking("info"))

    def test_a_role_overrides_its_tier_or_inherits(self):
        with _llm(thinking={"research_assembly": True},
                  thinking_roles={"query_templates": "off", "info": "on", "planner": ""}):
            self.assertFalse(agents.role_thinking("query_templates"))
            self.assertTrue(agents.role_thinking("assemble_workflow"))
            self.assertTrue(agents.role_thinking("info"))
            self.assertFalse(agents.role_thinking("planner"))       # "" = as its tier (off)

    def test_the_defaults_think_for_the_lead_only(self):
        from src.utils.settings import load_settings
        llm = (load_settings() or {}).get("llm") or {}
        defaults = dict(llm.get("thinking") or {})
        with _llm(thinking=defaults, thinking_roles={}):
            self.assertTrue(agents.role_thinking("lead"))
            for role in ("orchestrator", "query_templates", "info", "vision_agent",
                         "qa_checker", "coder"):
                self.assertFalse(agents.role_thinking(role), role)


class WhatEachProviderIsAskedFor(unittest.TestCase):

    def build(self, llm_name, think, provider_switch=None, **kw):
        llm = {"thinking_roles": {"info": "on" if think else "off"}}
        if provider_switch:
            llm.update(provider_switch)
        env = {"DASHSCOPE_API_KEY": "x", "OPENAI_API_KEY": "x", "GEMINI_API_KEY": "x",
               "ANTHROPIC_API_KEY": "x"}
        with _llm(**llm), mock.patch.dict(os.environ, env):
            for var in ("ANTHROPIC_THINK", "DASHSCOPE_ENABLE_THINKING", "OLLAMA_THINK"):
                os.environ.pop(var, None)
            model, _ = agents._build_model(role="info", llm=llm_name, system_prompt="sys",
                                           dashscope_model=kw.get("model", "m"),
                                           anthropic_model="claude-haiku-4-5")
        return model.config

    def test_dashscope(self):
        self.assertTrue(self.build("dashscope", True)["params"]["extra_body"]["enable_thinking"])
        self.assertFalse(self.build("dashscope", False)["params"]["extra_body"]["enable_thinking"])

    def test_the_provider_switch_still_turns_it_on_for_everyone(self):
        cfg = self.build("dashscope", False, {"dashscope": {"enable_thinking": True}})
        self.assertTrue(cfg["params"]["extra_body"]["enable_thinking"])

    def test_anthropic(self):
        self.assertIn("thinking", self.build("claude", True)["params"])
        self.assertNotIn("thinking", self.build("claude", False)["params"])

    def test_openai_and_gemini(self):
        self.assertEqual(self.build("openai", True)["params"].get("reasoning_effort"), "medium")
        self.assertNotIn("reasoning_effort", self.build("openai", False)["params"])
        self.assertEqual(self.build("gemini", True)["params"].get("reasoning_effort"), "medium")


class TheLeadTier(unittest.TestCase):

    def test_blank_means_the_orchestrators_model(self):
        with _llm(tiers={"orchestrator": "dashscope,qwen-a", "lead": ""}), \
                mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("ORCHESTRATOR_LLM", None)
            self.assertEqual(agents.role_model("lead"), "dashscope,qwen-a")

    def test_its_own_model_when_set(self):
        with _llm(tiers={"orchestrator": "dashscope,qwen-a", "lead": "claude,claude-sonnet-4-5"}):
            self.assertEqual(agents.role_model("lead"), "claude,claude-sonnet-4-5")

    def test_it_is_offered_as_a_tier(self):
        self.assertIn("lead", agents.TIER_LABELS)
        self.assertEqual(agents._ROLE_TIERS["lead"], "lead")


class _ModelA:
    def __init__(self, name): self.name = name


class _ModelB(_ModelA):
    pass


def _pipeline_with(model):
    from src.pipeline import Pipeline
    p = Pipeline.__new__(Pipeline)
    agent = SimpleNamespace(model=model, system_prompt="sys", _cost_meta={"model_id": "orch"},
                            messages=[{"role": "assistant", "content": [
                                {"reasoningContent": {"reasoningText": {"text": "hmm"}}},
                                {"text": "answer"}]}])
    p._orchestrator_agent = agent
    p._orch_model = model
    p._lead_model = None
    p._lead_active = False
    return p, agent


class TheOrchestratorTakesTheLeadsModel(unittest.TestCase):

    def test_swapped_in_for_a_lead_turn_and_back_after(self):
        orch = _ModelA("orch")
        lead = _ModelA("lead")
        p, agent = _pipeline_with(orch)
        with mock.patch.object(agents, "build_lead_model",
                               return_value=(lead, "dashscope", "qwen-lead")) as build:
            self.assertTrue(p.use_lead_model(True))
            self.assertIs(agent.model, lead)
            self.assertEqual(agent._cost_meta["model_id"], "qwen-lead")
            self.assertTrue(p.use_lead_model(True))          # built once
            self.assertEqual(build.call_count, 1)
            self.assertFalse(p.use_lead_model(False))
            self.assertIs(agent.model, orch)
            self.assertEqual(agent._cost_meta["model_id"], "orch")
        # Same provider: the history keeps its reasoning.
        self.assertIn("reasoningContent", agent.messages[0]["content"][0])

    def test_another_provider_gets_the_history_without_reasoning(self):
        p, agent = _pipeline_with(_ModelA("orch"))
        with mock.patch.object(agents, "build_lead_model",
                               return_value=(_ModelB("lead"), "claude", "claude-sonnet")):
            p.use_lead_model(True)
        self.assertEqual(agent.messages[0]["content"], [{"text": "answer"}])

    def test_a_lead_model_that_cannot_be_built_leaves_the_orchestrators(self):
        orch = _ModelA("orch")
        p, agent = _pipeline_with(orch)
        with mock.patch.object(agents, "build_lead_model", side_effect=RuntimeError("no key")):
            self.assertFalse(p.use_lead_model(True))
        self.assertIs(agent.model, orch)

    def test_a_settings_change_drops_it(self):
        p, agent = _pipeline_with(_ModelA("orch"))
        with mock.patch.object(agents, "build_lead_model",
                               return_value=(_ModelA("lead"), "dashscope", "x")):
            p.use_lead_model(True)
        p.drop_lead_model()
        self.assertIsNone(p._lead_model)
        self.assertEqual(agent.model.name, "orch")


class WhichTurnsAreALeads(unittest.TestCase):

    def setUp(self):
        from src.utils import conversation_store as cs
        self.cs = cs
        tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.addCleanup(tmp.cleanup)
        env = mock.patch.dict(os.environ, {
            "AGENTY_CONVERSATION_DB": os.path.join(tmp.name, "c.sqlite")})
        env.start()
        self.addCleanup(env.stop)
        cs._INITIALISED = False
        self.addCleanup(setattr, cs, "_INITIALISED", False)

    def lead_turn(self, thread_id, text, origin="panel"):
        import queue
        from agenty_core.utils import turn_scope
        from src.pipeline import Pipeline
        from src.utils import turn_bus
        rid = f"r-{thread_id}-{origin}-{len(text)}"
        turn_bus.tee(queue.Queue(), request_id=rid, thread_id=thread_id, origin=origin)
        token = turn_scope.enter(turn_scope.Scope(rid, thread_id))
        try:
            return Pipeline._is_lead_turn(text)
        finally:
            turn_scope.leave(token)

    def test_asking_for_shot_conversations(self):
        t = self.cs.create_thread()
        self.assertTrue(self.lead_turn(t, "Work on these six shots in parallel, one conversation each"))
        self.assertFalse(self.lead_turn(t, "make the shot brighter"))

    def test_a_conversation_with_shots_and_its_wake_turns(self):
        lead, shot = self.cs.create_thread(), self.cs.create_thread()
        self.cs.set_shot(shot, lead, "sh010")
        self.assertTrue(self.lead_turn(lead, "how is it going?"))
        self.assertTrue(self.lead_turn(self.cs.create_thread(), "x", origin="shots"))
        # A shot is never a lead, whatever it is asked.
        self.assertFalse(self.lead_turn(shot, "use parallel shots and agents"))


class AReasoningChangeRebuildsOnlyItsAgents(unittest.TestCase):

    def test_fingerprint(self):
        from src.utils import model_reload as mr
        base = {"tiers": {"orchestrator": "dashscope,a", "research_assembly": "dashscope,b",
                          "fast_utility": "dashscope,c"}, "thinking": {}}
        with _llm(**base):
            before = mr.fingerprint()
        on = dict(base, thinking={"fast_utility": True})
        with _llm(**on):
            after = mr.fingerprint()
        changed = mr.changed_agents(before, after)
        self.assertIn("info", changed)
        self.assertNotIn("orchestrator", changed)
        self.assertNotIn("fix_workflow_assembly", changed)   # research_assembly only
        lead_on = dict(base, thinking={"lead": True})
        with _llm(**lead_on):
            self.assertEqual(mr.changed_agents(before, mr.fingerprint()), ["lead"])

    def test_the_lead_builder_drops_the_cached_model(self):
        from src.utils import model_reload as mr
        p = SimpleNamespace(drop_lead_model=mock.Mock())
        rebuilt, failures = mr.reload_live_agents(p, ["lead"])
        self.assertEqual(rebuilt, ["lead"])
        p.drop_lead_model.assert_called_once()


class StartingAShotMakesALead(unittest.TestCase):

    def test_became_lead_is_called(self):
        from src.utils import conversation_store as cs, shots
        tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.addCleanup(tmp.cleanup)
        with mock.patch.dict(os.environ, {"AGENTY_CONVERSATION_DB": os.path.join(tmp.name, "c.sqlite")}):
            cs._INITIALISED = False
            self.addCleanup(setattr, cs, "_INITIALISED", False)
            saved = dict(shots._hooks)
            self.addCleanup(shots._hooks.update, saved)
            became = mock.Mock()
            shots._hooks.update(start_turn=lambda *a, **k: "r1", is_running=lambda t: False,
                                became_lead=became)
            lead = cs.create_thread()
            self.assertTrue(shots.start_shot(lead, "sh010", "brief")["ok"])
        became.assert_called_once_with(lead)


if __name__ == "__main__":
    unittest.main()
