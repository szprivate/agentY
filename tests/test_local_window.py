"""A call that would not fit a local model's window is made to fit - gently.

Only for a model on Ollama, only when the call really would not fit, one tool
result at a time, oldest first, the newest answer of each tool last. A hosted
model's history is never touched by any of this.
"""
import unittest
from types import SimpleNamespace
from unittest import mock

from src.utils import compaction as cp


class OllamaModel:
    def __init__(self, num_ctx=None, max_tokens=None):
        self._cfg = {"options": {"num_ctx": num_ctx}} if num_ctx else {}
        if max_tokens:
            self._cfg["max_tokens"] = max_tokens

    def get_config(self):
        return self._cfg


class OpenAIModel:
    def get_config(self):
        return {"options": {"num_ctx": 4096}, "model_id": "deepseek-v4.1-flash"}


def _agent(model, messages, prompt_chars=40_000, tool_chars=80_000):
    reg = SimpleNamespace(get_all_tool_specs=lambda: [{"name": "t", "description": "x" * tool_chars}])
    return SimpleNamespace(model=model, system_prompt="p" * prompt_chars, tool_registry=reg,
                           messages=list(messages))


def _call(i, name, size, text=None):
    body = text if text is not None else (f"{name} #{i} " + "x" * size)
    return [{"role": "assistant", "content": [{"toolUse": {"toolUseId": f"u{i}", "name": name, "input": {"n": i}}}]},
            {"role": "user", "content": [{"toolResult": {"toolUseId": f"u{i}", "status": "success",
                                                         "content": [{"text": body}]}}]}]


def _history(calls):
    msgs = [{"role": "user", "content": [{"text": "Build a depth ControlNet workflow for street.png."}]}]
    for i, (name, size) in enumerate(calls):
        msgs += _call(i, name, size)
    return msgs


def _texts(msgs):
    return [b["toolResult"]["content"][0]["text"] for m in msgs for b in m.get("content", []) if "toolResult" in b]


class Fitting(unittest.TestCase):

    KW = dict(keep_recent=4, tool_chars=1_500, tool_head=500)

    def test_a_history_that_fits_is_returned_as_it_is(self):
        msgs = _history([("hf_file", 4_000)] * 3)
        out, changed = cp.fit_to_window(msgs, 50_000, **self.KW)
        self.assertEqual(out, msgs)
        self.assertEqual(changed, [])

    def test_it_stops_as_soon_as_it_fits_and_starts_with_the_oldest(self):
        msgs = _history([("hf_file", 8_000)] * 10)                 # ~20k tokens
        room = cp.estimate_tokens(msgs) - 3_000                    # two results too many
        out, changed = cp.fit_to_window(msgs, room, **self.KW)
        cut = [t.startswith(cp.TRIM_MARK) for t in _texts(out)]
        self.assertEqual(cut, [True, True] + [False] * 8)
        self.assertEqual(len(changed), 2)
        self.assertLessEqual(cp.estimate_tokens(out), room)

    def test_the_newest_answer_of_each_tool_goes_last(self):
        """The report of the workflow just built is what the turn works from: the
        identical lookups around it are cut first, however old it is."""
        calls = [("prepare_workflow", 9_000)] + [("hf_file", 8_000)] * 8 + [("get_node_schema", 8_000)]
        msgs = _history(calls)
        out, _ = cp.fit_to_window(msgs, cp.estimate_tokens(msgs) - 9_000, **self.KW)
        texts = _texts(out)
        self.assertFalse(texts[0].startswith(cp.TRIM_MARK), "the prepare_workflow report was cut")
        self.assertTrue(all(t.startswith(cp.TRIM_MARK) for t in texts[1:6]))

    def test_only_when_nothing_else_is_left_are_those_cut_too(self):
        calls = [("prepare_workflow", 20_000), ("hf_file", 3_000), ("check_model", 3_000),
                 ("hf_file", 3_000), ("hf_file", 3_000)]
        msgs = _history(calls)
        out, _ = cp.fit_to_window(msgs, 3_000, **self.KW)
        self.assertTrue(_texts(out)[0].startswith(cp.TRIM_MARK))

    def test_the_last_messages_and_the_request_are_never_touched(self):
        msgs = _history([("hf_file", 8_000)] * 6)
        out, _ = cp.fit_to_window(msgs, 1_000, **self.KW)
        self.assertEqual(out[0], msgs[0])
        self.assertEqual(out[-4:], msgs[-4:])

    def test_what_was_cut_already_is_not_cut_again(self):
        msgs = _history([("hf_file", 8_000)] * 6)
        once, _ = cp.fit_to_window(msgs, 4_000, **self.KW)
        twice, changed = cp.fit_to_window(once, 1_000, **self.KW)
        self.assertEqual(changed, [])
        self.assertEqual(twice, once)


class TheHook(unittest.TestCase):

    def _run(self, model, messages, **cfg):
        agent = _agent(model, messages)
        settings = dict(cp.DEFAULTS, enabled=True, **cfg)
        with mock.patch.object(cp, "settings", return_value=settings), mock.patch.object(cp, "archive"):
            cp.CompactionHookProvider(role="agent")._before_model(SimpleNamespace(agent=agent))
        return agent.messages

    def test_a_hosted_models_history_is_left_exactly_as_it_was(self):
        msgs = _history([("hf_file", 8_000)] * 24)                 # ~48k tokens: under the hosted limit
        self.assertEqual(self._run(OpenAIModel(), msgs), msgs)
        self.assertEqual(self._run(None, msgs), msgs)

    def test_a_local_model_gets_a_history_that_fits_its_window(self):
        msgs = _history([("hf_file", 8_000)] * 24)
        out = self._run(OllamaModel(65_536), msgs)
        room = 65_536 - cp.fixed_tokens(_agent(None, [])) - cp._ANSWER_RESERVE
        self.assertLessEqual(cp.estimate_tokens(out), room)
        self.assertLess(sum(t.startswith(cp.TRIM_MARK) for t in _texts(out)), 24)

    def test_a_local_model_with_room_to_spare_is_left_alone(self):
        msgs = _history([("hf_file", 8_000)] * 6)
        self.assertEqual(self._run(OllamaModel(131_072), msgs), msgs)

    def test_the_answers_allowance_comes_off_the_window(self):
        """Ollama lets a prompt be num_ctx - num_predict long: 32,768 with 16,384
        for the answer cut a 47,461-token call down to 16,386."""
        self.assertEqual(cp.answer_reserve(_agent(OllamaModel(65_536, 16_384), [])), 16_384)
        self.assertEqual(cp.answer_reserve(_agent(OllamaModel(65_536), [])), cp._ANSWER_RESERVE)
        msgs = _history([("hf_file", 8_000)] * 12)                 # ~24k: fits 65k, not 65k less 16k
        self.assertEqual(self._run(OllamaModel(65_536), msgs), msgs)
        out = self._run(OllamaModel(65_536, 16_384), msgs)
        self.assertGreater(sum(t.startswith(cp.TRIM_MARK) for t in _texts(out)), 0)

    def test_a_window_the_call_cannot_fit_is_said_once(self):
        told = []
        cp._warned.clear()
        with mock.patch("src.utils.status_bus.notify", side_effect=lambda text, **kw: told.append(text)):
            msgs = _history([("hf_file", 2_000)] * 3)
            self.assertEqual(self._run(OllamaModel(32_768, 16_384), msgs), msgs)     # nothing to cut: 30k of prompt
            self._run(OllamaModel(32_768, 16_384), msgs)
        self.assertEqual(len(told), 1)
        self.assertIn("32,768", told[0])
        self.assertIn("16,384 of them are kept for the answer", told[0])
        self.assertIn("no user query found", told[0])
        self.assertIn("llm.ollama.num_ctx", told[0])

    def test_an_output_allowance_that_eats_the_window_is_named_as_the_cause(self):
        """Max Output Tokens set to 90,384 of a 98,304 window: the advice is to
        lower that, not to find a card for a 262,144 window."""
        text = cp.window_warning(98_304, 90_384, 31_471, 58_051)
        self.assertIn("because of the output allowance", text)
        self.assertIn("Max Output Tokens", text)
        self.assertIn("16,384", text)
        self.assertNotIn("262,144", text)
        # a window that really is too small still says so
        self.assertIn("llm.ollama.num_ctx", cp.window_warning(32_768, 16_384, 30_000, 20_000))

    def test_a_call_that_fits_says_nothing(self):
        told = []
        cp._warned.clear()
        with mock.patch("src.utils.status_bus.notify", side_effect=lambda text, **kw: told.append(text)):
            self._run(OllamaModel(98_304, 16_384), _history([("hf_file", 8_000)] * 24))
            self._run(OpenAIModel(), _history([("hf_file", 8_000)] * 24))
        self.assertEqual(told, [])

    def test_the_setting_turns_it_off(self):
        msgs = _history([("hf_file", 8_000)] * 24)
        self.assertEqual(self._run(OllamaModel(65_536), msgs, fit_local_window=False), msgs)


class TheWindow(unittest.TestCase):

    def test_only_an_ollama_model_has_one(self):
        self.assertEqual(cp.ollama_window(_agent(OpenAIModel(), [])), 0)
        self.assertEqual(cp.ollama_window(_agent(None, [])), 0)
        self.assertEqual(cp.ollama_window(_agent(OllamaModel(98_304), [])), 98_304)

    def test_the_setting_stands_in_when_the_model_does_not_say(self):
        with mock.patch("src.utils.settings.load_settings",
                        return_value={"llm": {"ollama": {"num_ctx": 65_536}}}):
            self.assertEqual(cp.ollama_window(_agent(OllamaModel(), [])), 65_536)


if __name__ == "__main__":
    unittest.main()
