"""A provider refusing the content is not a broken workflow.

The repair specialist cannot fix one and shouldn't see one: the graph is correct,
the request reached the model, and the model's own filter said no. Sent to the
fixer it spends a repair budget rewriting a workflow that was already right and
then reports a defect that does not exist.

What does work is running it again — every one of these filters is probabilistic
and none of these APIs is deterministic. So a refusal re-rolls the seed and
re-runs, a bounded number of times, and when that stops being worth it the agent
is told plainly that this needs rewording, not repairing.

Every signature here was read out of ComfyUI's own API-node source.

    python -m unittest discover -s tests
"""

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from pipeline_stub import pipeline_stub
from src.pipeline import Pipeline
from src.utils.content_policy import Rejection, classify, exhausted


class ClassifyTest(unittest.TestCase):
    """The error text each provider actually produces."""

    def test_seedream_and_seedance(self):
        # nodes_bytedance.py: "ByteDance request failed. Code: <code>, message: …"
        r = classify("ByteDance request failed. Code: InputTextSensitiveContentDetected, "
                     "message: The text may contain sensitive information.")
        self.assertEqual((r.provider, r.stage), ("ByteDance", "input"))
        r = classify("ByteDance request failed. Code: OutputImageSensitiveContentDetected, "
                     "message: The generated image was filtered.")
        self.assertEqual((r.provider, r.stage), ("ByteDance", "output"))
        # The one spelled out in ComfyUI's source, for Seedance's audio track.
        r = classify("OutputAudioSensitiveContentDetected.PolicyViolation")
        self.assertEqual(r.provider, "ByteDance")

    def test_nano_banana_and_the_rest_of_gemini(self):
        r = classify("Gemini API blocked the request. Reason: SAFETY (…)")
        self.assertEqual((r.provider, r.stage), ("Gemini", "input"))
        r = classify("Gemini API blocked the request. Reasons: ['IMAGE_PROHIBITED_CONTENT']")
        self.assertEqual(r.provider, "Gemini")
        self.assertEqual(classify("Content filtered by Google's Responsible AI "
                                  "practices: 58061214 (1 videos filtered.)").provider,
                         "Google Veo")

    def test_gpt_image(self):
        # Surfaced by the shared client as "API Error: … (Type: …)".
        r = classify("API Error: Your request was rejected as a result of our safety "
                     "system. (Type: image_generation_user_error)")
        self.assertEqual((r.provider, r.stage), ("OpenAI", "input"))
        self.assertEqual(classify("API Error: content_policy_violation").provider, "OpenAI")

    def test_the_rest_of_the_roster(self):
        for text, provider in [
            ("Task failed with status: Content Moderated", "Black Forest Labs"),
            ("Task failed with status: Request Moderated", "Black Forest Labs"),
            ("The generation was blocked by Ideogram's content safety filter.", "Ideogram"),
            ("The generated image was flagged for content policy violation.", "Reve"),
            ("finish_reason: CONTENT_FILTERED", "Stability"),
            ("Kling request failed. Code: 1301, Message: Trigger content security policy",
             "Kling"),
        ]:
            with self.subTest(provider=provider):
                self.assertEqual(classify(text).provider, provider)

    def test_the_prose_bytedance_returns_as_a_plain_400(self):
        """From the 20:54 run — it only matched the generic 'copyright' catch-all."""
        r = classify("API Error: The request failed because the output image may be "
                     "related to copyright restrictions. Request id: 0217867339703 "
                     "(Type: BadRequest)")
        self.assertEqual((r.provider, r.stage), ("ByteDance", "output"),
                         "it says which stage refused — read it rather than guessing")
        r = classify("API Error: The request failed because the input image may "
                     "contain sensitive content. (Type: BadRequest)")
        self.assertEqual((r.provider, r.stage), ("ByteDance", "input"))

    def test_an_unnamed_provider_can_still_report_the_stage(self):
        r = classify("Generation stopped: the output image was blocked by review.")
        self.assertEqual(r.stage, "output")
        self.assertEqual(r.provider, "")

    def test_a_provider_that_words_it_its_own_way_still_lands(self):
        r = classify("Generation refused: the request violates our content policy.")
        self.assertIsNotNone(r)
        self.assertEqual(r.stage, "unknown")
        self.assertIsNotNone(classify("blocked: possible copyrighted character"))

    def test_an_ordinary_failure_is_left_alone(self):
        """These must reach the fixer — they are what it is for."""
        for text in [
            "Prompt outputs failed validation: Required input is missing: image",
            "Error occurred when executing KSampler: mat1 and mat2 shapes cannot be multiplied",
            "Value not in list: ckpt_name: 'sdxl.safetensors' not in []",
            "Payment Required: Please add credits to your account to use this node.",
            "Rate Limit Exceeded: The server returned 429 after all retry attempts.",
            "ComfyUI execution failed",
        ]:
            with self.subTest(text=text):
                self.assertIsNone(classify(text), text)

    def test_it_reads_the_executors_error_dict(self):
        r = classify({"error": "Error occurred when executing ByteDanceImageNode:",
                      "details": {"exception_message": "ByteDance request failed. Code: "
                                                       "OutputImageSensitiveContentDetected"}})
        self.assertEqual(r.provider, "ByteDance")

    def test_nothing_at_all_is_not_a_refusal(self):
        self.assertIsNone(classify(""))
        self.assertIsNone(classify({}))
        self.assertIsNone(classify(None))


class BudgetTest(unittest.TestCase):
    def test_an_output_side_refusal_is_worth_more_tries_than_an_input_one(self):
        """A prompt the provider read and refused reads the same the second time."""
        self.assertGreater(Rejection("X", "output", "q").retries(),
                           Rejection("X", "input", "q").retries())

    def test_the_budget_can_be_overridden(self):
        with mock.patch.dict("os.environ", {"AGENTY_POLICY_RETRIES": "0"}):
            self.assertEqual(Rejection("X", "output", "q").retries(), 0)

    def test_what_it_says_when_the_retries_are_spent(self):
        out = exhausted(Rejection("OpenAI", "input", "safety system"), 1)
        self.assertEqual(out["status"], "rejected")
        self.assertEqual(out["kind"], "content_policy")
        self.assertIn("NOT a workflow defect", out["error"])
        self.assertIn("Reword the prompt", out["what_to_do"])
        self.assertIn("Do not send this to the repair specialist", out["do_not"])

    def test_an_output_refusal_gets_different_advice(self):
        out = exhausted(Rejection("Ideogram", "output", "blocked"), 2)
        self.assertIn("what came out was not", out["what_to_do"])


class RetryTest(unittest.TestCase):
    """The loop: re-roll and re-run, then stop and say so."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.wf = self.tmp / "canvas_000.json"
        self.wf.write_text(json.dumps({
            "1": {"class_type": "ByteDanceImageNode",
                  "inputs": {"prompt": "a hero", "seed": 42}},
            "2": {"class_type": "KSampler", "inputs": {"noise_seed": 7}},
            "3": {"class_type": "SaveImage", "inputs": {"images": ["1", 0]}},
        }), encoding="utf-8")

    def _seeds(self):
        g = json.loads(self.wf.read_text(encoding="utf-8"))
        return g["1"]["inputs"]["seed"], g["2"]["inputs"]["noise_seed"]

    def test_a_refusal_re_rolls_and_asks_for_a_re_run(self):
        pipe = pipeline_stub()
        rej = Rejection("ByteDance", "output", "OutputImageSensitiveContentDetected")
        out = Pipeline._retry_after_refusal(pipe, str(self.wf), rej)
        self.assertEqual(out["status"], "ready")
        self.assertEqual(out["retried_after"], "content_policy")
        self.assertNotEqual(self._seeds(), (42, 7), "a re-run of the same seed is waste")

    def test_it_gives_up_after_the_budget_and_does_not_call_it_a_defect(self):
        pipe = pipeline_stub()
        rej = Rejection("ByteDance", "output", "OutputImageSensitiveContentDetected")
        seen = [Pipeline._retry_after_refusal(pipe, str(self.wf), rej)["status"]
                for _ in range(4)]
        self.assertEqual(seen, ["ready", "ready", "rejected", "rejected"])
        self.assertIn("NOT a workflow defect",
                      Pipeline._retry_after_refusal(pipe, str(self.wf), rej)["error"])

    def test_each_workflow_has_its_own_budget(self):
        pipe = pipeline_stub()
        other = self.tmp / "canvas_001.json"
        other.write_text(self.wf.read_text(encoding="utf-8"), encoding="utf-8")
        # An OUTPUT refusal: the stage is incidental to what this checks, and the
        # input stage no longer has a budget to spend (re-sending the same bytes
        # to the same classifier was never going to answer differently).
        rej = Rejection("ByteDance", "output", "OutputImageSensitiveContentDetected")
        self.assertEqual(Pipeline._retry_after_refusal(pipe, str(self.wf), rej)["status"],
                         "ready")
        self.assertEqual(Pipeline._retry_after_refusal(pipe, str(other), rej)["status"],
                         "ready", "one member's refusal is not the other's")

    def test_a_graph_with_no_seed_is_still_re_run(self):
        """OpenAI's seed says 'not implemented yet in backend' — the retry is the point."""
        self.wf.write_text(json.dumps({
            "1": {"class_type": "OpenAIGPTImage2", "inputs": {"prompt": "a hero"}}}),
            encoding="utf-8")
        out = Pipeline._retry_after_refusal(pipeline_stub(), str(self.wf),
                                            Rejection("OpenAI", "output", "blocked"))
        self.assertEqual(out["status"], "ready")

    def test_an_unreadable_file_fails_honestly(self):
        out = Pipeline._retry_after_refusal(pipeline_stub(), str(self.tmp / "gone.json"),
                                            Rejection("X", "output", "q"))
        self.assertEqual(out["status"], "failed")


class HealPathTest(unittest.TestCase):
    """The fixer must never be handed one of these."""

    def test_a_refusal_never_reaches_the_repair_agent(self):
        import asyncio
        pipe = pipeline_stub()
        tmp = Path(tempfile.mkdtemp()) / "wf.json"
        tmp.write_text(json.dumps({"1": {"class_type": "X", "inputs": {"seed": 1}}}),
                       encoding="utf-8")
        err = {"error": "ByteDance request failed. Code: OutputImageSensitiveContentDetected",
               "details": {}}
        with mock.patch("src.pipeline.create_fix_workflow_assembly_agent") as agent:
            out = asyncio.run(Pipeline._heal_exec_failure(pipe, str(tmp), err))
        agent.assert_not_called()
        self.assertEqual(out["status"], "ready")

    def test_an_ordinary_failure_still_goes_to_the_repair_agent(self):
        import asyncio
        pipe = pipeline_stub()
        pipe._run_fix_workflow_assembly = mock.AsyncMock(return_value={"status": "ready"})
        err = {"error": "Required input is missing: image", "details": {}}
        with mock.patch("src.pipeline.create_fix_workflow_assembly_agent") as agent:
            asyncio.run(Pipeline._heal_exec_failure(pipe, "C:/tmp/wf.json", err))
        agent.assert_called_once()
        pipe._run_fix_workflow_assembly.assert_awaited_once()


if __name__ == "__main__":
    unittest.main()
