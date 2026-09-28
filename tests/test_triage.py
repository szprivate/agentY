"""Triage: which model answers this message.

The rules that matter here are the ones whose failure is invisible — a seat whose
provider has no key, a text-only seat handed a message with an image in it, a
classifier that answered in prose — because all of them look like "it ran fine"
from the outside.

    python -m unittest discover -s tests
"""

import asyncio
import os
import unittest
from unittest import mock

from src.utils import triage

CLAUDE = "claude,claude-haiku-4-5"
SONNET = "claude,claude-sonnet-4-5"
QWEN_TEXT = "dashscope,qwen3.6-flash"
QWEN_VL = "dashscope,qwen3-vl-flash"


def settings(**triage_cfg) -> dict:
    cfg = {"enabled": True, "simple": CLAUDE, "complex": SONNET,
           "classifier": CLAUDE, "min_confidence": 0.6}
    cfg.update(triage_cfg)
    return {"llm": {"triage": cfg}}


class Fixture(unittest.TestCase):
    """Settings and environment under our control, and no remembered decisions."""

    seat = CLAUDE          # what role_model("orchestrator") resolves to
    classifier = CLAUDE    # what role_model("triage") resolves to
    env = {"ANTHROPIC_API_KEY": "k", "DASHSCOPE_API_KEY": "k"}

    def configure(self, **triage_cfg) -> None:
        merged = settings(**triage_cfg)
        patches = [
            mock.patch("src.utils.settings.load_settings", return_value=merged),
            mock.patch("src.agent.role_model",
                       side_effect=lambda role, **_kw: (self.classifier
                                                        if role == "triage" else self.seat)),
            mock.patch.dict(os.environ, self.env, clear=False),
        ]
        for p in patches:
            p.start()
            self.addCleanup(p.stop)

    def setUp(self) -> None:
        triage.reset()
        self.addCleanup(triage.reset)
        env = mock.patch.dict(os.environ, {}, clear=False)
        env.start()
        self.addCleanup(env.stop)
        os.environ.pop("AGENTY_TRIAGE", None)


class ShortcutTest(Fixture):

    def test_work_that_needs_no_reading(self):
        for kwargs in ({"has_hooks": True}, {"has_media": True}, {"dry_run": True},
                       {"pending_approval": True}):
            self.assertEqual(triage.shortcut("yes", **kwargs), triage.COMPLEX, kwargs)

    def test_an_acknowledgement(self):
        for text in ("thanks!", "Thank you", "perfect 🙏", "ok", "nice, thanks"):
            self.assertEqual(triage.shortcut(text), triage.SIMPLE, text)

    def test_a_short_message_that_is_not_small_talk(self):
        for text in ("ok now render all four rooms", "redo it", "make it brighter",
                     "what did that cost?"):
            self.assertIsNone(triage.shortcut(text), text)

    def test_an_answer_to_a_question_is_never_shortcut_to_simple(self):
        # "yes" alone sets running whatever the agent asked about. The gate catches
        # the two cases agentY knows it is waiting on; the rest go to the classifier.
        self.assertEqual(triage.shortcut("yes", pending_approval=True), triage.COMPLEX)
        for text in ("yes", "no", "go ahead", "yes please", "do it"):
            self.assertIsNone(triage.shortcut(text), text)


class SeatTest(Fixture):

    def test_an_explicit_seat(self):
        self.configure()
        self.assertEqual(triage.seat_for(triage.SIMPLE), CLAUDE)
        self.assertEqual(triage.seat_for(triage.COMPLEX), SONNET)

    def test_a_blank_seat_follows_the_orchestrator_tier(self):
        self.seat = QWEN_TEXT
        self.configure(complex="")
        self.assertEqual(triage.seat_for(triage.COMPLEX), QWEN_TEXT)

    def test_a_seat_whose_provider_has_no_key_follows_the_tier(self):
        # The committed defaults name Anthropic models; a DashScope-only install
        # must not be re-pointed at one it cannot call.
        self.seat = QWEN_TEXT
        self.env = {"DASHSCOPE_API_KEY": "k"}
        self.configure()
        with mock.patch.dict(os.environ, {"ANTHROPIC_API_KEY": ""}):
            self.assertEqual(triage.seat_for(triage.SIMPLE), QWEN_TEXT)

    def test_ollama_needs_no_key(self):
        self.configure(simple="ollama,qwen3:0.6b")
        self.assertEqual(triage.seat_for(triage.SIMPLE), "ollama,qwen3:0.6b")


class ClassifierChoiceTest(Fixture):

    def test_an_unreachable_provider_is_declined(self):
        self.classifier = "openai,gpt-4o"
        self.configure(classifier="")
        self.assertEqual(triage.classifier_spec(), "")

    def test_the_setting_beats_the_tier(self):
        self.classifier = "dashscope,qwen3.7-flash"
        self.configure(classifier=CLAUDE)
        self.assertEqual(triage.classifier_spec(), CLAUDE)


class ParseTest(unittest.TestCase):

    def test_plain_fenced_and_reasoned(self):
        for raw in ('{"complexity": "complex", "confidence": 0.9, "why": "video plan"}',
                    '```json\n{"complexity":"complex","confidence":0.9,"why":"video plan"}\n```',
                    '<think>hmm</think>{"complexity":"complex","confidence":0.9,'
                    '"why":"video plan"}'):
            self.assertEqual(triage._parse(raw), (triage.COMPLEX, 0.9, "video plan"))

    def test_nothing_usable(self):
        for raw in ("", "it depends", '{"complexity": "medium"}', "{oops",
                    '{"complexity": "simple", "confidence": "high"}'):
            weight, confidence, _why = triage._parse(raw)
            self.assertTrue(weight == "" or confidence == 0.0, raw)


class Answer:
    """A classifier that answers with *raw*, or raises/hangs instead."""

    def __init__(self, raw="", exc=None, hang=False):
        self.raw, self.exc, self.hang = raw, exc, hang
        self.messages = None

    async def chat(self, messages, *, json_format=False):  # noqa: ARG002
        self.messages = messages
        if self.exc is not None:
            raise self.exc
        if self.hang:
            await asyncio.sleep(5)
        return self.raw


class DecideTest(Fixture):

    def answer(self, **kwargs) -> Answer:
        stub = Answer(**kwargs)
        p = mock.patch("src.utils.llm_functions.LLMFunctions.for_spec", return_value=stub)
        p.start()
        self.addCleanup(p.stop)
        q = mock.patch("src.utils.triage._prompt", return_value="judge it")
        q.start()
        self.addCleanup(q.stop)
        return stub

    def decide(self, text, **kwargs):
        return asyncio.run(triage.decide(text, **kwargs))

    def test_off_means_nothing_happens(self):
        self.configure(enabled=False)
        self.answer(raw='{"complexity":"complex","confidence":1}')
        self.assertIsNone(self.decide("build me a five-shot video"))

    def test_two_identical_seats_are_not_worth_a_call(self):
        self.configure(simple=CLAUDE, complex=CLAUDE)
        stub = self.answer(raw='{"complexity":"complex","confidence":1}')
        self.assertIsNone(self.decide("build me a five-shot video"))
        self.assertIsNone(stub.messages)  # never asked

    def test_a_shortcut_skips_the_call(self):
        self.configure()
        stub = self.answer(raw='{"complexity":"complex","confidence":1}')
        got = self.decide("thanks!")
        self.assertEqual((got.complexity, got.seat, got.source),
                         (triage.SIMPLE, CLAUDE, "shortcut"))
        self.assertIsNone(stub.messages)

    def test_the_classifier_decides_the_rest(self):
        self.configure()
        self.answer(raw='{"complexity":"complex","confidence":0.9,"why":"five shots"}')
        got = self.decide("do the bedroom, the kitchen and two exteriors")
        self.assertEqual((got.complexity, got.seat, got.why, got.source),
                         (triage.COMPLEX, SONNET, "five shots", "classifier"))

    def test_a_low_confidence_reading_is_thrown_away(self):
        self.configure(min_confidence=0.8)
        self.answer(raw='{"complexity":"simple","confidence":0.5}')
        self.assertIsNone(self.decide("what did that cost?"))

    def test_a_failure_leaves_the_seat_alone(self):
        self.configure()
        self.answer(exc=RuntimeError("502 from the endpoint"))
        self.assertIsNone(self.decide("what did that cost?"))

    def test_a_timeout_leaves_the_seat_alone(self):
        self.configure(timeout=1.0)
        self.answer(hang=True)
        with mock.patch.object(triage, "_timeout", return_value=0.01):
            self.assertIsNone(self.decide("what did that cost?"))

    def test_the_previous_turn_is_shown_to_the_classifier(self):
        self.configure()
        stub = self.answer(raw='{"complexity":"complex","confidence":0.9}')
        self.decide("render the bedroom", conversation="t1")
        self.decide("make it brighter", conversation="t1")
        sent = stub.messages[-1]["content"]
        self.assertIn("render the bedroom", sent)
        self.assertIn("complex", sent)

    def test_conversations_do_not_see_each_other(self):
        self.configure()
        stub = self.answer(raw='{"complexity":"complex","confidence":0.9}')
        self.decide("render the bedroom", conversation="t1")
        self.decide("what is a LoRA?", conversation="t2")
        self.assertNotIn("render the bedroom", stub.messages[-1]["content"])


class VisionGuardTest(Fixture):
    """An embedded image may never be handed to a model that cannot read one."""

    def test_a_text_only_simple_seat_is_refused_for_an_image_turn(self):
        self.configure(simple=QWEN_TEXT, complex=QWEN_VL)
        got = asyncio.run(triage.decide("thanks!", has_media=True, needs_vision=True))
        # has_media already forces complex; the point is the seat that comes back.
        self.assertEqual(got.seat, QWEN_VL)

    def test_neither_seat_can_see_falls_back_to_the_configured_tier(self):
        self.seat = QWEN_VL
        self.configure(simple=QWEN_TEXT, complex="dashscope,qwen3.7-max")
        got = asyncio.run(triage.decide("what is in this?", has_media=True,
                                        needs_vision=True))
        self.assertEqual(got.seat, QWEN_VL)
        self.assertIn("read images", got.why)

    def test_a_path_in_the_text_is_work_but_needs_no_eyes(self):
        self.configure(simple=QWEN_TEXT, complex=QWEN_TEXT + "x")
        got = asyncio.run(triage.decide("upscale D:/x/a.png", has_media=True,
                                        needs_vision=False))
        self.assertEqual(got.complexity, triage.COMPLEX)
        self.assertEqual(got.seat, QWEN_TEXT + "x")  # not vision-filtered


class Agentish:
    """Just enough of a Strands Agent for the swap."""

    def __init__(self, spec=CLAUDE):
        provider, _, model = spec.partition(",")
        self.model = object()
        self.system_prompt = "you are the orchestrator"
        self.messages = [{"role": "user", "content": [{"text": "hi"}]}]
        self.tool_names = ["run_research"]
        self._cost_meta = {"provider": provider, "model_id": model,
                           "is_ollama": provider == "ollama"}


class RetargetTest(unittest.TestCase):

    def test_the_model_is_replaced_and_nothing_else_is(self):
        from src import agent as agents
        a = Agentish(CLAUDE)
        before_messages, before_tools = a.messages, a.tool_names
        built = object()
        with mock.patch.object(agents, "build_model", return_value=(built, "qwen-plus")) as bm:
            landed = agents.retarget_agent(a, "dashscope,qwen-plus", role="orchestrator")
        self.assertEqual(landed, "dashscope,qwen-plus")
        self.assertIs(a.model, built)
        self.assertEqual(a._cost_meta, {"provider": "dashscope", "model_id": "qwen-plus",
                                        "is_ollama": False})
        self.assertIs(a.messages, before_messages)
        self.assertIs(a.tool_names, before_tools)
        # The system prompt travels with the model (Anthropic caches it there).
        self.assertEqual(bm.call_args.kwargs["system_prompt"], "you are the orchestrator")

    def test_a_spec_with_no_model_is_refused(self):
        from src import agent as agents
        with self.assertRaises(ValueError):
            agents.retarget_agent(Agentish(), "claude")

    def test_the_spec_of_a_built_agent(self):
        from src import agent as agents
        self.assertEqual(agents.agent_spec(Agentish(SONNET)), SONNET)
        self.assertEqual(agents.agent_spec(object()), "")


class ApplyTest(Fixture):

    def decision(self, seat, complexity=triage.COMPLEX) -> triage.Decision:
        return triage.Decision(complexity=complexity, seat=seat, why="because",
                               confidence=1.0, source="classifier")

    def test_the_same_seat_is_not_rebuilt(self):
        self.configure()
        agent = Agentish(CLAUDE)
        with mock.patch("src.agent.retarget_agent") as retarget:
            self.assertEqual(triage.apply_to(agent, self.decision(CLAUDE)), "")
        retarget.assert_not_called()

    def test_a_different_seat_is_applied(self):
        self.configure()
        agent = Agentish(CLAUDE)
        with mock.patch("src.agent.retarget_agent", return_value=SONNET) as retarget:
            self.assertEqual(triage.apply_to(agent, self.decision(SONNET)), SONNET)
        retarget.assert_called_once_with(agent, SONNET, role="orchestrator")

    def test_a_failed_swap_keeps_the_working_model(self):
        self.configure()
        agent = Agentish(CLAUDE)
        with mock.patch("src.agent.retarget_agent", side_effect=RuntimeError("no key")):
            self.assertEqual(triage.apply_to(agent, self.decision(SONNET)), "")

    def test_no_decision_does_nothing(self):
        self.configure()
        self.assertEqual(triage.apply_to(Agentish(), None), "")
        self.assertEqual(triage.apply_to(None, self.decision(SONNET)), "")


class PipelineWiringTest(unittest.TestCase):
    """The seat is chosen BETWEEN turns, in stream_async, before the orchestrator runs."""

    def test_stream_async_triages_before_it_streams(self):
        import inspect
        from src.pipeline import Pipeline
        src = inspect.getsource(Pipeline.stream_async)
        self.assertIn("_triage_seat", src)
        self.assertLess(src.index("_triage_seat"), src.index("_astream_orchestrator"))

    def test_an_embedded_image_is_told_apart_from_a_path(self):
        import inspect
        from src.pipeline import Pipeline
        src = inspect.getsource(Pipeline._triage_seat)
        self.assertIn("needs_vision=embedded", src)


if __name__ == "__main__":
    unittest.main()
