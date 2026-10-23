"""The orchestrator can switch its own model; a rule can tell it when to ask.

The hard constraint: the routing pass adds ZERO model calls. A per-message
classifier was built and reverted (1e05c1f / edea633); here the rule check reads
the name of a tool call that already happened, so these tests stand in for the
model factory and assert it is never asked to build one unless a switch is
actually made.

    uv run --with pytest python -m pytest -q tests/test_model_routing.py
"""

import unittest
from types import SimpleNamespace
from unittest import mock

import src.agent as agents
from src.utils import costs
from src.utils import model_routing as mr
from src.utils import tool_permissions as tp
from agenty_core.utils import turn_scope

CATALOG = {
    "Anthropic": [["claude,claude-opus-5-5", "Claude Opus 5.5"],
                  ["claude,claude-haiku-4-5", "Claude Haiku 4.5"]],
    "Alibaba (DashScope)": [["dashscope,deepseek-v4.1-flash", "DeepSeek V4.1 Flash"],
                            ["dashscope,qwen3-coder-plus", "Qwen3 Coder Plus"]],
}
RULES = [
    {"when": {"skill": "model-from-reference"}, "role": "orchestrator",
     "suggest": "claude,claude-opus-5-5", "why": "3D needs a stronger model"},
    {"when": {"tool_pack": "dcc"}, "role": "orchestrator",
     "suggest": ["claude,claude-ghost-9", "claude,claude-opus-5-5"], "why": "dcc builds"},
    {"when": {"tool": "web_search"}, "role": "orchestrator",
     "suggest": "claude,claude-not-there", "why": "unavailable model"},
]


def make_agent(spec="dashscope,deepseek-v4.1-flash", messages=None):
    provider, _, model_id = spec.partition(",")
    return SimpleNamespace(
        model=SimpleNamespace(tag="base"), system_prompt="p", messages=messages or [],
        _cost_meta={"provider": provider, "model_id": model_id, "is_ollama": False},
        _is_claude=True)


class Base(unittest.TestCase):
    def setUp(self):
        self.routing = {"auto_switch": False, "rules": RULES}
        mr._OVERRIDES.clear()
        mr._NOTIFIED.clear()
        self.built = []

        def build(**kw):
            self.built.append(kw)
            model_id = kw["dashscope_model"]
            return SimpleNamespace(tag=f"{kw['llm']},{model_id}"), model_id

        patches = [
            mock.patch.object(mr, "_catalog", lambda: CATALOG),
            mock.patch.object(mr, "_settings", lambda: self.routing),
            mock.patch.object(agents, "_build_model", build),
            mock.patch.object(mr, "current_models",
                              lambda: {"tiers": {"orchestrator": "dashscope,deepseek-v4.1-flash",
                                                 "vision": "dashscope,deepseek-v4.1-flash"},
                                       "roles": {"vision_agent": "dashscope,deepseek-v4.1-flash"}}),
        ]
        for p in patches:
            p.start()
            self.addCleanup(p.stop)
        token = turn_scope.enter(turn_scope.Scope("req", "thread-1"))
        self.addCleanup(turn_scope.leave, token)


class ListModels(Base):
    def test_only_configured_providers_with_vision_and_price(self):
        got = mr.available_models()
        self.assertEqual({m["id"] for m in got}, {s[0] for g in CATALOG.values() for s in g})
        by = {m["id"]: m for m in got}
        self.assertTrue(by["claude,claude-opus-5-5"]["vision"])
        self.assertFalse(by["dashscope,qwen3-coder-plus"]["vision"])
        self.assertEqual(by["claude,claude-opus-5-5"]["price_per_mtok"]["in"], 4.0)

    def test_a_provider_without_a_key_is_absent(self):
        with mock.patch.object(mr, "_catalog", lambda: {"Anthropic": CATALOG["Anthropic"]}):
            self.assertNotIn("dashscope,deepseek-v4.1-flash", {m["id"] for m in mr.available_models()})

    def test_role_narrows_to_vision_candidates(self):
        out = mr.list_models(make_agent(), "vision")
        self.assertTrue(out["needs_vision"])
        self.assertNotIn("dashscope,qwen3-coder-plus", out["candidates"])
        self.assertEqual(out["running"], "dashscope,deepseek-v4.1-flash")


class Switch(Base):
    def setUp(self):
        super().setUp()
        self.agent = make_agent()

    def switch(self, **kw):
        args = dict(role="orchestrator", model="claude,claude-opus-5-5", reason="r",
                    scope="conversation", confirm=False)
        args.update(kw)
        return mr.switch_model(self.agent, **args)

    def test_unknown_model_is_refused_with_the_closest(self):
        out = self.switch(model="claude,claude-opus-5-6")
        self.assertFalse(out["ok"])
        self.assertIn("claude,claude-opus-5-5", out["closest"])
        self.assertEqual(self.built, [])

    def test_text_only_refused_when_the_conversation_carries_images(self):
        self.agent.messages = [{"role": "user", "content": [
            {"text": "look"}, {"image": {"format": "png", "source": {"bytes": b"x"}}}]}]
        out = self.switch(model="dashscope,qwen3-coder-plus", confirm=True)
        self.assertFalse(out["ok"])
        self.assertIn("images", out["error"])
        self.assertEqual(self.agent.model.tag, "base")

    def test_image_inside_a_tool_result_counts(self):
        self.agent.messages = [{"role": "user", "content": [{"toolResult": {"content": [
            {"image": {"format": "jpeg", "source": {"bytes": b"x"}}}]}}]}]
        self.assertFalse(self.switch(model="dashscope,qwen3-coder-plus", confirm=True)["ok"])

    def test_text_only_is_fine_without_images(self):
        self.assertTrue(self.switch(model="dashscope,qwen3-coder-plus", confirm=True)["applied"])

    def test_a_vision_role_never_gets_a_text_only_model(self):
        out = self.switch(role="vision", model="dashscope,qwen3-coder-plus", scope="settings", confirm=True)
        self.assertFalse(out["ok"])

    def test_without_confirm_it_is_only_a_proposal(self):
        out = self.switch()
        self.assertTrue(out["ok"])
        self.assertFalse(out["applied"])
        self.assertEqual(out["proposal"]["to"], "claude,claude-opus-5-5")
        self.assertEqual(self.built, [])
        self.assertEqual(self.agent.model.tag, "base")

    def test_confirm_switches_the_seat_and_the_cost_meta(self):
        before = costs.get_model_prices_for(self.agent)
        out = self.switch(confirm=True)
        self.assertTrue(out["applied"])
        self.assertEqual(self.agent.model.tag, "claude,claude-opus-5-5")
        self.assertEqual(self.agent._cost_meta["model_id"], "claude-opus-5-5")
        after = costs.get_model_prices_for(self.agent)
        self.assertNotEqual(before, after)
        self.assertEqual(after, (4.0 / 1e6, 20.0 / 1e6))
        self.assertEqual(mr.conversation_spec(), "claude,claude-opus-5-5")
        self.assertIn("next model call", out["takes_effect"])

    def test_default_reverts_model_and_cost_meta(self):
        self.switch(confirm=True)
        out = self.switch(model="default")
        self.assertTrue(out["reverted"])
        self.assertEqual(self.agent.model.tag, "base")
        self.assertEqual(self.agent._cost_meta["model_id"], "deepseek-v4.1-flash")
        self.assertEqual(mr.conversation_spec(), "")

    def test_the_override_follows_its_conversation_only(self):
        self.switch(confirm=True)
        mr.sync_thread(self.agent, "thread-2")           # another conversation, same pipeline
        self.assertEqual(self.agent.model.tag, "base")
        mr.sync_thread(self.agent, "thread-1")           # back
        self.assertEqual(self.agent.model.tag, "claude,claude-opus-5-5")

    def test_conversation_scope_moves_only_the_orchestrator(self):
        self.assertFalse(self.switch(role="vision", confirm=True)["ok"])

    def test_vision_capability_follows_the_conversation_override(self):
        from src.utils import agentY_server as srv
        self.switch(model="dashscope,qwen3-coder-plus", confirm=True)
        self.assertFalse(srv._orchestrator_supports_vision())
        self.switch(model="claude,claude-opus-5-5", confirm=True)
        self.assertTrue(srv._orchestrator_supports_vision())

    def test_settings_scope_goes_through_the_switch_model_path(self):
        from src.utils import agentY_server as srv
        seen = []
        said = ("saved. A conversation is running, so the switch reaches it the moment "
                "its turn finishes")
        with mock.patch.object(srv, "_switch_model",
                               lambda a: seen.append(a) or [{"type": "system", "data": said}]):
            out = self.switch(scope="settings", confirm=True)
        self.assertEqual(seen, [["orchestrator", "claude,claude-opus-5-5"]])
        self.assertTrue(out["applied"])
        self.assertEqual(out["takes_effect"], "when this turn ends")
        self.assertEqual(mr.conversation_spec(), "")

    def test_settings_scope_refuses_text_only_when_any_conversation_has_images(self):
        with mock.patch.object(mr, "_any_conversation_has_images", lambda a: True):
            self.assertFalse(self.switch(scope="settings", model="dashscope,qwen3-coder-plus",
                                         confirm=True)["ok"])


class Approval(Base):
    def test_the_ask_list_covers_switch_model_and_an_old_list_cannot_undo_it(self):
        self.assertIn("switch_model", tp.DEFAULT_ASK_TOOLS)
        use = {"confirm": True, "model": "claude,claude-opus-5-5"}
        self.assertTrue(tp.must_ask("switch_model", use, []))                   # even an empty list
        self.assertFalse(tp.must_ask("switch_model", {"model": "x"}, ["switch_model"]))   # a proposal
        self.assertFalse(tp.must_ask("switch_model", {"confirm": True, "model": "default"}, []))
        self.assertTrue(tp.must_ask("run_script", {}, ["run_script"]))

    def test_auto_switch_skips_the_prompt(self):
        self.routing = {"auto_switch": True, "rules": RULES}
        self.assertFalse(tp.must_ask("switch_model", {"confirm": True, "model": "a,b"}, ["switch_model"]))

    def test_the_prompt_says_what_will_happen(self):
        text = tp.describe("switch_model", {"role": "orchestrator", "model": "claude,claude-opus-5-5",
                                            "scope": "conversation", "reason": "3D"})
        self.assertIn("claude,claude-opus-5-5", text)
        self.assertIn("3D", text)


class Rules(Base):
    def setUp(self):
        super().setUp()
        self.agent = make_agent()

    def check(self, tool, **inp):
        return mr.check_rules(self.agent, mr.triggers_of(tool, inp))

    def test_a_skill_fires_once_with_the_notice_and_no_model_is_built(self):
        notes = self.check("skills", skill_name="model-from-reference")
        self.assertEqual(len(notes), 1)
        self.assertIn("claude,claude-opus-5-5 is recommended (3D needs a stronger model)", notes[0])
        self.assertIn("switch_model", notes[0])
        self.assertEqual(self.check("skills", skill_name="model-from-reference"), [])   # never repeated
        self.assertEqual(self.built, [])                                                # zero model calls
        self.assertEqual(self.agent.model.tag, "base")

    def test_a_pack_load_and_a_direct_held_tool_both_count_but_speak_once(self):
        self.assertEqual(len(self.check("load_tools", pack="dcc")), 1)
        self.assertEqual(self.check("dcc_silhouette_check"), [])

    def test_the_first_available_candidate_wins(self):
        self.assertIn("claude,claude-opus-5-5", self.check("load_tools", pack="dcc")[0])

    def test_a_rule_whose_model_is_not_available_says_nothing(self):
        self.assertEqual(self.check("web_search"), [])
        self.assertEqual(self.built, [])

    def test_nothing_when_already_on_the_suggested_model(self):
        self.agent = make_agent("claude,claude-opus-5-5")
        self.assertEqual(self.check("skills", skill_name="model-from-reference"), [])

    def test_an_unrelated_call_costs_nothing(self):
        self.assertEqual(self.check("queue"), [])

    def test_a_conversation_with_images_is_not_pointed_at_a_text_only_model(self):
        self.agent.messages = [{"role": "user", "content": [{"image": {"format": "png", "source": {}}}]}]
        self.routing = {"rules": [{"when": {"skill": "s"}, "suggest": "dashscope,qwen3-coder-plus",
                                   "why": "w"}]}
        self.assertEqual(self.check("skills", skill_name="s"), [])

    def test_auto_switch_switches_and_says_so(self):
        self.routing = {"auto_switch": True, "rules": RULES}
        notes = self.check("skills", skill_name="model-from-reference")
        self.assertIn("switched this conversation to claude,claude-opus-5-5", notes[0])
        self.assertEqual(self.agent.model.tag, "claude,claude-opus-5-5")
        self.assertEqual(len(self.built), 1)                 # the switch, nothing else

    def test_a_malformed_rule_is_skipped(self):
        self.routing = {"rules": ["x", {"when": {}}, {"when": {"skill": "s"}},
                                  {"when": {"skill": "s"}, "suggest": "nocomma"}]}
        self.assertEqual(mr.rules(), [])

    def test_the_hook_appends_to_the_tool_result(self):
        event = SimpleNamespace(
            agent=self.agent,
            tool_use={"name": "skills", "input": {"skill_name": "model-from-reference"}},
            result={"content": [{"text": "skill body"}], "status": "success"})
        mr.ModelRoutingHook()._on_after(event)
        self.assertEqual(len(event.result["content"]), 2)
        self.assertTrue(event.result["content"][1]["text"].startswith("Routing:"))
        self.assertEqual(self.built, [])

    def test_each_conversation_hears_a_rule_once(self):
        t = [("skill", "model-from-reference")]
        self.assertEqual(len(mr.check_rules(self.agent, t)), 1)
        token = turn_scope.enter(turn_scope.Scope("r2", "thread-2"))
        try:
            self.assertEqual(len(mr.check_rules(self.agent, t)), 1)
        finally:
            turn_scope.leave(token)


class Reload(unittest.TestCase):
    def test_editing_the_rules_does_not_rebuild_every_agent(self):
        from src.utils import model_reload
        base = {"llm": {"history_window": 12, "routing": {"auto_switch": False, "rules": []}}}
        edited = {"llm": {"history_window": 12, "routing": {"auto_switch": True, "rules": RULES}}}
        prints = []
        for settings in (base, edited):
            with mock.patch.object(agents, "_settings", lambda s=settings: s),                  mock.patch.object(agents, "role_model", lambda *a, **k: "claude,x"),                  mock.patch.object(agents, "role_thinking", lambda *a, **k: False):
                prints.append(model_reload.fingerprint())
        self.assertEqual(model_reload.changed_agents(*prints), [])


class Shipped(unittest.TestCase):
    def test_the_default_rules_parse_and_name_a_known_trigger(self):
        import tomllib
        from pathlib import Path
        data = tomllib.loads(Path("config/settings.default.toml").read_text(encoding="utf-8"))
        routing = data["llm"]["routing"]
        self.assertFalse(routing["auto_switch"])
        kinds = {(next(iter(r["when"])), next(iter(r["when"].values()))) for r in routing["rules"]}
        self.assertEqual(kinds, {("skill", "model-from-reference"), ("tool_pack", "dcc")})


if __name__ == "__main__":
    unittest.main()
