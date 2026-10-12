"""The mid-turn limit follows a local model's window - and only a local model's.

A hosted model keeps the configured limit to the token: its window is large, and
trimming earlier would only break the provider's prompt cache.
"""
import unittest
from types import SimpleNamespace
from unittest import mock

from src.utils import compaction as cp

CFG = {"hard_budget_tokens": 60_000}


class OllamaModel:
    def __init__(self, num_ctx=None):
        self._cfg = {"options": {"num_ctx": num_ctx}} if num_ctx else {}

    def get_config(self):
        return self._cfg


class OpenAIModel:
    def get_config(self):
        return {"options": {"num_ctx": 8192}, "model_id": "deepseek-v4.1-flash"}


class AnthropicModel:
    pass


def _agent(model, prompt_chars=40_000, tool_chars=80_000):
    reg = SimpleNamespace(get_all_tool_specs=lambda: [{"name": "t", "description": "x" * tool_chars}])
    return SimpleNamespace(model=model, system_prompt="p" * prompt_chars, tool_registry=reg)


class Hosted(unittest.TestCase):

    def test_a_hosted_model_keeps_the_configured_limit(self):
        for model in (OpenAIModel(), AnthropicModel(), None):
            with self.subTest(model=type(model).__name__):
                self.assertEqual(cp.hard_budget_for(_agent(model), CFG), 60_000)

    def test_whatever_its_prompt_and_tools_weigh(self):
        self.assertEqual(cp.hard_budget_for(_agent(OpenAIModel(), 400_000, 800_000), CFG), 60_000)

    def test_and_whatever_the_ollama_setting_says(self):
        with mock.patch("src.utils.settings.load_settings",
                        return_value={"llm": {"ollama": {"num_ctx": 4096}}}):
            self.assertEqual(cp.hard_budget_for(_agent(OpenAIModel()), CFG), 60_000)


class Local(unittest.TestCase):

    def test_the_limit_is_what_the_window_has_left(self):
        # 65536 - (40000 + ~80040) / 4 - 4096 answer reserve
        got = cp.hard_budget_for(_agent(OllamaModel(65_536)), CFG)
        self.assertTrue(31_000 < got < 31_500, got)

    def test_never_more_than_the_configured_limit(self):
        self.assertEqual(cp.hard_budget_for(_agent(OllamaModel(262_144)), CFG), 60_000)

    def test_never_less_than_the_floor(self):
        self.assertEqual(cp.hard_budget_for(_agent(OllamaModel(32_768)), CFG), cp._LOCAL_FLOOR)

    def test_the_setting_stands_in_when_the_model_does_not_say(self):
        with mock.patch("src.utils.settings.load_settings",
                        return_value={"llm": {"ollama": {"num_ctx": 65_536}}}):
            got = cp.hard_budget_for(_agent(OllamaModel()), CFG)
        self.assertTrue(31_000 < got < 31_500, got)


class TheHook(unittest.TestCase):
    """The same history: aged for a local model, left alone for a hosted one."""

    def _messages(self):
        big = "x" * 8_000
        msgs = [{"role": "user", "content": [{"text": "build it"}]}]
        for i in range(24):                      # ~48k tokens of tool output
            msgs.append({"role": "assistant", "content": [{"toolUse": {"toolUseId": f"u{i}", "name": "get_node_schema", "input": {}}}]})
            msgs.append({"role": "user", "content": [{"toolResult": {"toolUseId": f"u{i}", "content": [{"text": big}]}}]})
        return msgs

    def _run(self, model):
        agent = _agent(model)
        agent.messages = self._messages()
        before = cp.estimate_tokens(agent.messages)
        with mock.patch.object(cp, "settings", return_value=dict(cp.DEFAULTS, enabled=True)), \
                mock.patch.object(cp, "archive"):
            cp.CompactionHookProvider(role="agent")._before_model(SimpleNamespace(agent=agent))
        return before, cp.estimate_tokens(agent.messages)

    def test_hosted_history_is_untouched_below_the_configured_limit(self):
        before, after = self._run(OpenAIModel())
        self.assertLess(before, 60_000)
        self.assertEqual(after, before)

    def test_local_history_is_aged_once_the_window_is_short(self):
        before, after = self._run(OllamaModel(65_536))
        self.assertLess(after, before)


if __name__ == "__main__":
    unittest.main()
