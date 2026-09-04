"""A judge that cannot see asks someone who can.

From the report: *"the qa agent should be able to forward any vision runs to the
vision agent (same as what the orchestrator does)."*

The orchestrator has never read pixels. It calls `analyze_image` in `describe`
mode, the vision agent looks, and the orchestrator reasons over the text. That is
why pointing the orchestrator tier at a text-only model works fine.

`qa_judge` had no such route. Handed an image, a text-only model raises
("Unexpected item type in content."), and `check_output` recognised the blindness
but could only report it — returning a pass that meant nothing, forever, for every
output. Now it relays: the output (and the briefing's references) go to the vision
agent, the descriptions come back, and the judge answers the same question about
them. The verdict is real, and marked `secondhand` because it was reached from
someone else's description rather than the image.

    python -m unittest discover -s tests
"""

import unittest
from unittest import mock

from src.utils import qa

# The real error, from the session test_blind_vision_model.py was written from.
BLIND = RuntimeError(
    'BadRequestError: data: {"error":{"code":"invalid_parameter_error",'
    '"message":"The provided messages input is invalid. The error info is '
    '[Unexpected item type in content.].","type":"invalid_request_error"}}')

VERDICT = ('{"verdict": "fail", "summary": "reads as pregnant", '
           '"checks": [{"criterion": "not pregnant", "result": "fail", '
           '"note": "visible bump"}]}')

# Set by the check_set helper so a test can inspect the describer afterwards.
_LAST_ANALYZE: list = [None]

BRIEFING = qa.QaBriefing(criteria="a trans woman in her 40s, not pregnant",
                         sources=("canvas qa hook",))


class _BlindJudge:
    """Raises on image blocks the way a text-only endpoint does; answers text."""

    def __init__(self, reply=VERDICT):
        self.messages = []
        self.reply = reply
        self.seen: list = []

    def __call__(self, payload):
        self.seen.append(payload)
        if any(isinstance(b, dict) and "image" in b for b in payload):
            raise BLIND
        return self.reply


def _vision_says(text="A woman in her forties, slim, no pregnancy visible."):
    """A stand-in for `analyze_image` in describe mode."""
    return mock.Mock(return_value={"status": "success",
                                   "content": [{"text": text}]})


def _run_check(*, judge=None, analyze=None, has_vision=True, briefing=BRIEFING):
    """One `check_output` against a blind judge, with the relay stubbed around it."""
    judge = judge or _BlindJudge()
    analyze = analyze or _vision_says()
    from src.tools import image_handling as ih
    with mock.patch.object(qa, "_output_blocks",
                           return_value=([{"image": {}}], "an image")), \
         mock.patch.object(qa, "_output_frame_paths",
                           return_value=(["out.png"], "the GENERATED OUTPUT image")), \
         mock.patch.object(qa, "_image_block", return_value=None), \
         mock.patch.object(qa, "_judge_question", return_value=("QUESTION", [])), \
         mock.patch.object(qa, "qa_settings",
                           return_value={"video_frames": 1, "max_references": 2}), \
         mock.patch.object(ih, "_vision_agent", object() if has_vision else None), \
         mock.patch.object(ih, "analyze_image", analyze):
        return qa.check_output("out.png", briefing, agent=judge), judge, analyze


class TheRelay(unittest.TestCase):
    """`check_output` — a blind judge reaches the image through the vision agent."""

    def _check(self, **kw):
        return _run_check(**kw)

    def test_a_blind_judge_now_returns_a_real_verdict(self):
        res, _judge, _a = self._check()
        self.assertFalse(res.blind, "the relay worked; this is not an unchecked pass")
        self.assertFalse(res.passed)
        self.assertEqual(res.failed_criteria(), ["not pregnant — visible bump"])

    def test_the_verdict_is_marked_second_hand(self):
        # It IS a check, and it is not the same as having looked. Both matter.
        res, _judge, _a = self._check()
        self.assertTrue(res.secondhand)
        self.assertIn("judged from a vision agent's description", res.render())

    def test_it_went_through_the_vision_agent(self):
        _res, _judge, analyze = self._check()
        self.assertTrue(analyze.called)
        self.assertEqual(analyze.call_args.kwargs["mode"], "describe")
        self.assertEqual(analyze.call_args.kwargs["file_path"], "out.png")

    def test_the_describer_is_told_what_the_check_turns_on(self):
        # A generic "describe this image" spends its words on mood and omits the
        # one detail the briefing turns on.
        _res, _judge, analyze = self._check()
        self.assertIn("not pregnant", analyze.call_args.kwargs["question"])

    def test_the_judge_is_given_the_description_and_told_it_is_second_hand(self):
        _res, judge, _a = self._check()
        text = " ".join(b.get("text", "") for b in judge.seen[-1]
                        if isinstance(b, dict))
        self.assertIn("no pregnancy visible", text)
        self.assertIn("You cannot see images", text)
        self.assertIn("mark a criterion `n/a`", text)

    def test_the_second_call_carries_no_pixels(self):
        # Sending the image again to the model that just rejected it is the retry
        # loop this whole path exists to avoid.
        _res, judge, _a = self._check()
        self.assertTrue(any("image" in b for b in judge.seen[0] if isinstance(b, dict)))
        self.assertFalse(any("image" in b for b in judge.seen[-1] if isinstance(b, dict)))

    def test_references_are_described_too(self):
        brief = qa.QaBriefing(criteria="match the reference lighting",
                              reference_paths=("ref.png",), sources=("t",))
        _res, judge, analyze = self._check(briefing=brief)
        self.assertIn("ref.png", [c.kwargs.get("file_path")
                                  for c in analyze.call_args_list])
        text = " ".join(b.get("text", "") for b in judge.seen[-1]
                        if isinstance(b, dict))
        self.assertIn("DESCRIPTION OF REFERENCE IMAGE 1", text)


class WhenTheRelayIsNotThere(unittest.TestCase):
    """It must degrade to the honest old answer, never to a made-up verdict."""

    def _check(self, **kw):
        return _run_check(**kw)[0]

    def test_no_vision_agent_registered_reports_blindness(self):
        res = self._check(has_vision=False)
        self.assertTrue(res.blind)
        self.assertTrue(res.passed, "our misconfiguration must not fail their work")
        self.assertFalse(res.secondhand)

    def test_a_blind_vision_model_is_not_a_fallback(self):
        # Relaying to a second model that also cannot see is the same failure.
        refused = mock.Mock(return_value={"status": "error",
                                          "content": [{"text": "not multimodal"}]})
        res = self._check(analyze=refused)
        self.assertTrue(res.blind)
        self.assertFalse(res.secondhand)

    def test_a_vision_agent_that_raises_is_survivable(self):
        res = self._check(analyze=mock.Mock(side_effect=RuntimeError("boom")))
        self.assertTrue(res.blind)

    def test_the_advice_names_both_halves_of_the_problem(self):
        said: list = []
        with mock.patch("src.utils.status_bus.emit", said.append):
            self._check(has_vision=False)
        self.assertEqual(len(said), 1)
        self.assertIn("no vision agent was available", said[0])

    def test_a_transient_failure_never_reaches_the_relay(self):
        # A timeout is doubt, not blindness. Describing the image and judging the
        # description would answer a question nobody asked.
        from src.tools import image_handling as ih
        agent = mock.Mock()
        agent.messages = []
        agent.side_effect = RuntimeError("Read timed out")
        analyze = _vision_says()
        with mock.patch.object(qa, "_output_blocks",
                               return_value=([{"image": {}}], "an image")), \
             mock.patch.object(qa, "_image_block", return_value=None), \
             mock.patch.object(qa, "qa_settings",
                               return_value={"video_frames": 1, "max_references": 2}), \
             mock.patch.object(ih, "_vision_agent", object()), \
             mock.patch.object(ih, "analyze_image", analyze):
            res = qa.check_output("out.png", BRIEFING, agent=agent)
        self.assertFalse(analyze.called)
        self.assertFalse(res.blind)
        self.assertIn("timed out", res.error)


class TheSetJudgeRelaysToo(unittest.TestCase):
    """`check_set` asks a question a description answers well: do these match?"""

    def _check(self, analyze=None, judge=None):
        judge = judge or _BlindJudge(
            '{"verdict": "pass", "summary": "one grade", "checks": []}')
        analyze = analyze or _vision_says()
        from src.tools import image_handling as ih
        with mock.patch.object(qa, "_output_blocks",
                               return_value=([{"image": {}}], "an image")), \
             mock.patch.object(qa, "_image_block", return_value=None), \
             mock.patch.object(qa, "load_qa_prompts",
                               return_value={"question": "Q {{CRITERIA}}"}), \
             mock.patch.object(qa, "qa_settings",
                               return_value={"video_frames": 1, "max_references": 2,
                                             "max_outputs": 6}), \
             mock.patch.object(ih, "_vision_agent", object()), \
             mock.patch.object(ih, "analyze_image", analyze):
            _LAST_ANALYZE[0] = analyze
            return qa.check_set(["a.png", "b.png"], BRIEFING, agent=judge), judge

    def test_a_blind_set_judge_still_returns_a_verdict(self):
        res, _judge = self._check()
        self.assertTrue(res.passed)
        self.assertTrue(res.secondhand)
        self.assertEqual(res.error, "")

    def test_every_output_is_described(self):
        _res, _judge = self._check()
        # Both outputs, and nothing invented for a file that was never read.
        self.assertEqual(sorted(c.kwargs["file_path"] for c in
                                _LAST_ANALYZE[0].call_args_list), ["a.png", "b.png"])

    def test_it_still_asks_about_the_set_not_the_stills(self):
        _res, judge = self._check()
        text = " ".join(b.get("text", "") for b in judge.seen[-1]
                        if isinstance(b, dict))
        self.assertIn("AS A SET", text)
        self.assertIn("DESCRIPTION OF OUTPUT 1", text)
        self.assertIn("DESCRIPTION OF OUTPUT 2", text)

    def test_no_relay_keeps_the_old_unavailable_result(self):
        refused = mock.Mock(return_value={"status": "error", "content": []})
        res, _judge = self._check(analyze=refused)
        self.assertTrue(res.passed)
        self.assertFalse(res.secondhand)
        self.assertIn("Unexpected item type", res.error)


if __name__ == "__main__":
    unittest.main()
