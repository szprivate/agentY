"""A turn that ends on "let me do X:" and does nothing is sent back.

The endings below are the real ones from the local-orchestrator benchmark of
2026-10-12 (qwen3.6:27b): each closed a turn in which nothing was built.
"""
import re
import unittest
from types import SimpleNamespace
from unittest import mock

from src.utils import intent_guard as ig

SEEN = [
    "Good, I have SD 1.5 and SDXL checkpoints available. Since the request is a classic img2img with denoise control, I'll use an SDXL checkpoint which works well for this. Let me also check what's in the input image to write a good prompt.  \ud83e\uddea DRY RUN \u2014 nothing was submitted to ComfyUI, and nothing was built either: no run was reached this turn.",
    "The workflow is valid. Now I'll signal it ready and then insert it into the canvas (dry run mode means no submission). Let me also verify the input file was staged properly first \u2014 street.png should already be in ComfyUI's input dir from my earlier `upload_image`.  Let me double-check:  \ud83e\uddea DRY RUN \u2014 nothing was submitted to ComfyUI, and nothing was built either: no run was reached this turn.",
    "Now I understand the current wiring. Let me restructure this properly for inpainting:  1. Remove Canny (#4), ControlNetLoader (#5), and ControlNetApplyAdvanced (#8) \u2014 not needed for alpha-mask inpainting 2. Add `InpaintModelConditioning` node that takes the positive/negative conditioning, VAE, image (from scaling), mask (from LoadImage slot 1), and outputs conditioned positive/negative + latent 3. Wire KSampler to use InpaintModelConditioning's outputs instead of ControlNetApplyAdvanced  Let me do this step by step:  \ud83e\uddea DRY RUN \u2014 nothing was submitted to ComfyUI, and nothing was built either: no run was reached this turn.",
    "I need to find an SD 1.5 checkpoint that's already installed. Let me look more broadly:  \ud83e\uddea DRY RUN \u2014 nothing was submitted to ComfyUI, and nothing was built either: no run was reached this turn.",
    "\ud83e\uddea DRY RUN \u2014 nothing was submitted to ComfyUI, and nothing was built either: no run was reached this turn.",
    "I'll fix this by: 1. Adding a PreviewImage node 2. Wiring VAEDecode output to it 3. Then separately deleting the SaveImage node  Let me do add + connect first, then delete:  \ud83e\uddea DRY RUN \u2014 nothing was submitted to ComfyUI, and nothing was built either: no run was reached this turn.",
    "Now I understand the full wiring. The issue is that `KSamplerAdvanced` nodes require their inputs to be connected, and looking at the validation errors:  The problem says it needs a **KSampler** node (not KSamplerAdvanced), but what's actually on the canvas are two `KSamplerAdvanced` nodes with proper base\u2192refiner chaining. The wiring looks correct from what I can see - #10 feeds into #11, and #11 feeds VAE decode.  Wait - let me re-read the error message more carefully. It says \"Required node missing: the workflow needs a KSampler node.\" This suggests the validator is looking for a standard `KSampler`, not `KSamplerAdvanced`. The issue might be that `KSamplerAdvanced` has different input requirements than `KSampler`.  Let me check what inputs are actually wired and identify which connections might be broken. Let me look at all nodes:  \ud83e\uddea DRY RUN \u2014 nothing was submitted to ComfyUI, and nothing was built either: no run was reached this turn.",
    "Now I have a clear picture. The current wiring is:  ``` #2 LoadImage (IMAGE\u2192#3, MASK\u2192nothing)     \u2192 #3 ImageScale \u2192 #4 Canny \u2192 (#8 ControlNetApplyAdvanced for conditioning)                                      \u2199 also \u2190 positive from #6, negative from #7                                       \u2193                                  #10 KSampler (latent_image\u2190empty latent #9)                                      \u2193                                   #11 VAEDecode \u2192 #12 SaveImage ```  **Problems:** - LoadImage has a MASK output (#2 slot 1) that nothing reads \u2014 the mask is just ignored. It's doing full repaint, not inpainting of *only* masked areas. - The KSampler feeds an `EmptyLatentImage` instead of VAE-encoded content of what to keep. Without proper latent conditioning from the source image, it doesn't know which regions are already there and which need repainting \u2014 this produces a completely different output than inpaint.  Let me fix the wiring by adding proper SD1.5 inpainting nodes:  \ud83e\uddea DRY RUN \u2014 nothing was submitted to ComfyUI, and nothing was built either: no run was reached this turn.",
    "I need to fix the prompt and also verify against an SD 1.5 ControlNet reference workflow:  \ud83e\uddea DRY RUN \u2014 nothing was submitted to ComfyUI, and nothing was built either: no run was reached this turn.",
    "I see \u2014 I need to be more specific about which input on node #6 to disconnect. Let me try with `input` specified:  \ud83e\uddea DRY RUN \u2014 nothing was submitted to ComfyUI, and nothing was built either: no run was reached this turn."
]

FINISHED = [
    "Built from the SD 1.5 text-to-image template with dreamshaper_8; it saves to output/bench.",
    "The workflow is on the canvas. Let me know if you want a different sampler.",
    "Which checkpoint should I use - SD 1.5 or SDXL?",
    "Done. I swapped SaveImage for PreviewImage and removed the old node.",
    "I could not find an SD 1.5 inpainting model. Let me know whether to download one.",
    "Two things changed:\n- the sampler is now dpmpp_2m\n- steps went from 20 to 30.",
]


class Unfinished(unittest.TestCase):

    def test_every_ending_seen_in_the_benchmark_is_caught(self):
        for text in SEEN:
            # The benchmark quoted the whole reply; the runtime's dry-run line at
            # its end is not the agent's.
            text = re.sub(r"\s*[^\x00-\x7f]*\s*DRY RUN.*$", "", text, flags=re.S)
            with self.subTest(ending=text[-60:]):
                self.assertTrue(ig.unfinished(text))

    def test_a_finished_answer_is_left_alone(self):
        for text in FINISHED:
            with self.subTest(answer=text[:50]):
                self.assertEqual(ig.unfinished(text), "")

    def test_it_names_the_sentence_that_was_not_followed(self):
        self.assertEqual(ig.unfinished("I found the cause. Let me look more broadly:"),
                         "Let me look more broadly:")
        self.assertEqual(ig.unfinished("Both are installed. Now let me check the input image."),
                         "Now let me check the input image.")

    def test_what_the_runtime_appended_is_not_the_agents_last_word(self):
        text = "Let me try with `input` specified:\n\n\U0001f9ea DRY RUN - nothing was submitted to ComfyUI."
        self.assertEqual(ig.unfinished(text), "Let me try with `input` specified:")


class LastAnswer(unittest.TestCase):

    def test_the_closing_text_of_the_turn(self):
        msgs = [{"role": "user", "content": [{"text": "build it"}]},
                {"role": "assistant", "content": [{"text": "Let me "}, {"text": "check:"}]}]
        self.assertEqual(ig.last_answer(msgs), "Let me check:")

    def test_a_turn_that_ended_on_a_tool_is_not_judged(self):
        call = {"role": "assistant", "content": [{"text": "Let me check:"}, {"toolUse": {"name": "x"}}]}
        self.assertIsNone(ig.last_answer([call]))
        self.assertIsNone(ig.last_answer([call, {"role": "user", "content": [{"toolResult": {}}]}]))
        self.assertIsNone(ig.last_answer([]))


class InThePipeline(unittest.TestCase):
    """Pipeline._unfinished_intent_input: when the note is sent, and when not."""

    def _pipe(self, text):
        from src.pipeline import Pipeline
        pipe = Pipeline.__new__(Pipeline)
        pipe._verbose = False
        pipe._orchestrator_agent = SimpleNamespace(messages=[
            {"role": "assistant", "content": [{"text": text}]}])
        return pipe

    def _ask(self, text, done=0, signalled=(), setting=True):
        with mock.patch("src.utils.workflow_signal.peek", return_value=list(signalled)), \
                mock.patch("src.utils.settings.load_settings", return_value={"intent_guard": setting}):
            return self._pipe(text)._unfinished_intent_input(done)

    def test_an_announced_step_gets_the_note_naming_it(self):
        note = self._ask("I need to find a checkpoint. Let me look more broadly:")
        self.assertIn("Let me look more broadly:", note)
        self.assertIn("RUNTIME NOTE", note)

    def test_a_finished_answer_gets_none(self):
        self.assertEqual(self._ask("Built and filed. Let me know what to change."), "")

    def test_not_after_a_workflow_was_signalled(self):
        self.assertEqual(self._ask("Let me run it:", signalled=["a.json"]), "")

    def test_not_more_often_than_the_limit(self):
        self.assertEqual(self._ask("Let me look:", done=ig.MAX_REDRIVES), "")
        self.assertTrue(self._ask("Let me look:", done=ig.MAX_REDRIVES - 1))

    def test_the_setting_turns_it_off(self):
        self.assertEqual(self._ask("Let me look:", setting=False), "")


if __name__ == "__main__":
    unittest.main()
