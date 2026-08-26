"""A model that cannot see must say so, not fail cryptically or pass silently.

From a real session. The vision tier and `qa_judge` were both pointed at
`dashscope,qwen3.7-max`, which is not multimodal. Handed an image, DashScope
answers:

    invalid_parameter_error — The provided messages input is invalid.
    The error info is [Unexpected item type in content.]

Nothing in that mentions vision, and the two callers made it worse in opposite
directions:

* `analyze_image` reported "the vision agent call failed … Retry analyze_image
  for this file". So the agent retried, re-uploaded, copied the file to a temp
  path, simplified the question, and retried again — the one instruction that
  could never work — before telling the user only that analysis was failing.
* `qa.check_output` treats any exception as doubt and PASSES, because a judge
  that cannot be reached must not condemn the user's work. With a blind judge
  that is not doubt: a silver hatchback passed "must show a RED SPORTS CAR on a
  racetrack", and every output would have passed, forever, silently.

Both are now told apart from a transient failure and named for what they are.

    python -m unittest discover -s tests
"""

import unittest
from unittest import mock

from src.utils.vision_capability import blind_model_message, looks_blind

# The real message, from the run this comes from.
DASHSCOPE = ('BadRequestError: data: {"error":{"code":"invalid_parameter_error",'
             '"param":null,"message":"The provided messages input is invalid. '
             'The error info is [Unexpected item type in content.].",'
             '"type":"invalid_request_error"}}')


class DetectionTest(unittest.TestCase):

    def test_the_error_this_came_from_is_recognised(self):
        self.assertTrue(looks_blind(DASHSCOPE))

    def test_other_providers_phrasings_are_recognised(self):
        for text in ("This model does not support image input",
                     "400: image_url is not supported by this model",
                     "Error: model does not support images",
                     "Unsupported content type: image"):
            self.assertTrue(looks_blind(text), text)

    def test_an_exception_object_is_accepted_not_just_a_string(self):
        self.assertTrue(looks_blind(RuntimeError(DASHSCOPE)))

    def test_a_transient_failure_is_not_mistaken_for_blindness(self):
        """Telling someone to change a working setting is the costly wrong answer."""
        for text in ("Read timed out after 60s",
                     "429 Too Many Requests — rate limit exceeded",
                     "Connection reset by peer",
                     "500 Internal Server Error",
                     "insufficient_quota: You exceeded your current quota",
                     ""):
            self.assertFalse(looks_blind(text), text)

    def test_nothing_at_all_is_not_blindness(self):
        self.assertFalse(looks_blind(None))

    def test_the_message_names_the_setting_and_forbids_retrying(self):
        msg = blind_model_message("qa_judge", "qwen3.7-max", DASHSCOPE)
        self.assertIn("qa_judge", msg)
        self.assertIn("qwen3.7-max", msg)
        self.assertIn("not multimodal", msg)
        self.assertIn("Do not retry", msg)
        self.assertIn("settings.local.json", msg)

    def test_it_still_reads_without_a_model_name(self):
        msg = blind_model_message("vision")
        self.assertIn("vision", msg)
        self.assertNotIn("()", msg, "an empty model name left brackets behind")

    def test_it_is_printable_on_a_windows_console(self):
        """These strings reach a cp1252 terminal via print(); one crashed it."""
        blind_model_message("vision", "m", DASHSCOPE).encode("cp1252")

    def test_only_a_real_model_name_is_ever_quoted(self):
        """Whatever ends up here is printed back as a setting to go and change."""
        from src.utils.vision_capability import model_name

        good = mock.Mock()
        good.model = mock.Mock(config={"model_id": "  qwen3-vl-flash  "})
        self.assertEqual(model_name(good), "qwen3-vl-flash")

        for bad in (mock.Mock(),                     # config is itself a Mock
                    mock.Mock(model=None),           # no model at all
                    mock.Mock(model=mock.Mock(config={})),          # no id
                    mock.Mock(model=mock.Mock(config={"model_id": 7}))):
            self.assertEqual(model_name(bad), "", repr(bad))


class QaJudgeTest(unittest.TestCase):
    """A blind judge passes — but never quietly."""

    def _check(self, exc):
        from src.utils import qa
        agent = mock.Mock()
        agent.messages = []
        agent.side_effect = exc
        with mock.patch.object(qa, "_output_blocks",
                               return_value=([{"image": {}}], "an image")), \
             mock.patch.object(qa, "_image_block", return_value=None), \
             mock.patch.object(qa, "qa_settings",
                               return_value={"video_frames": 1, "max_references": 2}):
            return qa.check_output("out.png", qa.QaBriefing(criteria="a red car",
                                                            sources=("t",)),
                                   agent=agent)

    def test_a_blind_judge_is_flagged_as_blind(self):
        res = self._check(RuntimeError(DASHSCOPE))
        self.assertTrue(res.blind, "a pass that means nothing looked like a real one")
        self.assertIn("not multimodal", res.error)

    def test_a_blind_judge_still_does_not_condemn_the_work(self):
        """Our misconfiguration must not fail the user's render."""
        self.assertTrue(self._check(RuntimeError(DASHSCOPE)).passed)

    def test_a_transient_failure_keeps_the_old_behaviour(self):
        res = self._check(RuntimeError("Read timed out"))
        self.assertTrue(res.passed)
        self.assertFalse(res.blind)
        self.assertIn("timed out", res.error)

    def test_a_working_judge_is_never_blind(self):
        from src.utils import qa
        agent = mock.Mock()
        agent.messages = []
        agent.return_value = '{"passed": true, "summary": "fine", "checks": []}'
        with mock.patch.object(qa, "_output_blocks",
                               return_value=([{"image": {}}], "an image")), \
             mock.patch.object(qa, "_image_block", return_value=None), \
             mock.patch.object(qa, "qa_settings",
                               return_value={"video_frames": 1, "max_references": 2}):
            res = qa.check_output("out.png", qa.QaBriefing(criteria="a red car",
                                                           sources=("t",)),
                                  agent=agent)
        self.assertFalse(res.blind)


class AnalyzeImageTest(unittest.TestCase):
    """The tool must not send the agent round the retry loop that caused this."""

    def _analyse(self, exc):
        from src.tools import image_handling as ih
        agent = mock.Mock()
        agent.messages = []
        agent.side_effect = exc
        agent.model = mock.Mock(config={"model_id": "qwen3.7-max"})
        with mock.patch.object(ih, "_vision_agent", agent), \
             mock.patch.object(ih, "_ensure_vision_pool") as pool, \
             mock.patch("src.utils.agentY_server._orchestrator_supports_vision",
                        return_value=False):
            pool.return_value.borrow.return_value.__enter__ = lambda s: agent
            pool.return_value.borrow.return_value.__exit__ = lambda s, *a: False
            out = ih.analyze_image(file_path=str(_png_on_disk()),
                                   question="what is this?", mode="describe")
        return " ".join(c.get("text", "") for c in out.get("content", []))

    def test_a_blind_model_is_named_and_retrying_is_forbidden(self):
        text = self._analyse(RuntimeError(DASHSCOPE))
        self.assertIn("not multimodal", text)
        self.assertIn("Do not retry", text)
        self.assertNotIn("Retry analyze_image", text,
                         "the advice that produced three useless retries")

    def test_a_transient_failure_still_says_retry(self):
        text = self._analyse(RuntimeError("Read timed out"))
        self.assertIn("Retry analyze_image", text)
        self.assertNotIn("not multimodal", text)


_TMP = None


def _png_on_disk():
    """A real 1x1 PNG on disk — analyze_image reads and re-encodes it."""
    global _TMP
    import tempfile
    from pathlib import Path
    if _TMP is None:
        _TMP = tempfile.TemporaryDirectory()
    p = Path(_TMP.name) / "probe.png"
    if not p.exists():
        p.write_bytes(
            b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
            b"\x08\x06\x00\x00\x00\x1f\x15\xc4\x89\x00\x00\x00\nIDATx\x9cc\x00\x01"
            b"\x00\x00\x05\x00\x01\r\n-\xb4\x00\x00\x00\x00IEND\xaeB`\x82")
    return p


if __name__ == "__main__":
    unittest.main()
