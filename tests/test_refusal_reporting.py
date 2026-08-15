"""When a provider refuses an input, say WHICH one — and don't re-send it.

From a real run: seven reference frames generated, the video refused four seconds
in with ByteDance's `InputImageSensitiveContentDetected.PolicyViolation` — "the
input image may be related to copyright restrictions". One of the seven was a
recognisable franchise character. Two things were wrong with how that landed.

It was re-sent. An input refusal is not the bet the others are: an output refusal
is a probabilistic filter scoring a fresh roll, and rolling again genuinely often
passes. An input refusal is the provider reading what was SENT. Re-running ships
the identical bytes to the identical classifier — the seed cannot reach a
reference image — so the retry was pure latency, and on a video model that is
minutes of it.

And it named nothing. The provider identifies the offending image by its own asset
id, which means nothing locally, so "a reference image was refused" left all seven
under suspicion. The graph knows better: which slot each arrived on, which file it
came from, and — from the prompt's own `@imageN` table — which character it is.

    python -m unittest discover -s tests
"""

import unittest

from src.pipeline import Pipeline
from src.utils import content_policy as cp
from src.utils.canvas_hooks import describe_references, reference_inputs

REFUSAL = ('Polling aborted due to error: Task failed: {"error": {"code": '
           '"InputImageSensitiveContentDetected.PolicyViolation", "message": "The '
           'request failed because the input image may be related to copyright '
           'restrictions."}}')


def graph(files=("ref_00042_.png", "ref_00043_.png", "ref_00044_.png")):
    """The shape the collector path builds: collector -> expander -> numbered slots."""
    g = {"367": {"class_type": "AgentYImageCollector",
                 "inputs": {"files": "\n".join(f"C:/out/{f}" for f in files)}},
         "385": {"class_type": "AgentYImageBatchExpand",
                 "inputs": {"images": ["367", 0]}},
         "384": {"class_type": "ByteDance2ReferenceNode", "inputs": {
             "model.prompt": ("REFERENCE ASSIGNMENT TABLE:\n"
                              "@image1 = TANIHO (HERO) — samurai in red armor\n"
                              "@image2 = HEDI MAZIU (MENTOR) — elder Japanese man\n"
                              "@image3 = APE — colossal giant ape, charcoal fur\n"
                              "\nSTYLE: cool-blue palette")}}}
    for i in range(len(files)):
        g["384"]["inputs"][f"model.reference_images.image_{i + 1}"] = ["385", i]
    return g


class InputRefusalIsNotReRunTest(unittest.TestCase):

    def test_it_is_recognised_as_an_input_refusal(self):
        rej = cp.classify(REFUSAL)
        self.assertIsNotNone(rej)
        self.assertEqual((rej.provider, rej.stage), ("ByteDance", "input"))

    def test_an_input_refusal_is_not_re_sent(self):
        """The seed cannot reach a reference image."""
        self.assertEqual(cp.classify(REFUSAL).retries(), 0)

    def test_an_output_refusal_still_gets_its_rolls(self):
        """That one IS a fresh generation, and frequently passes."""
        out = cp.classify("The request failed because the output image may be "
                          "related to copyright restrictions")
        self.assertEqual(out.stage, "output")
        self.assertGreaterEqual(out.retries(), 2)

    def test_it_says_re_running_was_not_attempted_rather_than_that_it_failed(self):
        told = cp.exhausted(cp.classify(REFUSAL), 0)
        self.assertIn("Re-running was not attempted", told["error"])
        self.assertIn("NOT a workflow defect", told["error"])
        self.assertNotIn("further attempt", told["error"])

    def test_the_advice_is_to_replace_the_reference(self):
        told = cp.exhausted(cp.classify(REFUSAL), 0)
        self.assertIn("reference image", told["what_to_do"])
        self.assertIn("Do not send this to the repair specialist", told["do_not"])


class WhichReferenceTest(unittest.TestCase):

    def test_each_slot_is_resolved_back_through_the_expander_to_a_file(self):
        refs = reference_inputs(graph(), "384")
        self.assertEqual([r["slot"] for r in refs],
                         ["image_1", "image_2", "image_3"])
        self.assertEqual([r["path"].rsplit("/", 1)[-1] for r in refs],
                         ["ref_00042_.png", "ref_00043_.png", "ref_00044_.png"])

    def test_the_prompt_s_own_table_names_the_subject(self):
        block = describe_references(graph(), "384")
        self.assertIn("image_1 = ref_00042_.png — TANIHO (HERO)", block)
        self.assertIn("image_3 = ref_00044_.png — APE", block)

    def test_it_says_the_order_is_the_upload_order(self):
        """Which is what makes the list a shortlist rather than a set."""
        self.assertIn("stops at the first one it refuses",
                      describe_references(graph(), "384"))

    def test_a_slot_wired_straight_to_a_loader_is_read_off_its_widget(self):
        g = {"7": {"class_type": "LoadImage", "inputs": {"image": "ben.png"}},
             "348": {"class_type": "ByteDanceSeedreamNodeV2",
                     "inputs": {"model.images.image_1": ["7", 0]}}}
        self.assertEqual(reference_inputs(g, "348")[0]["path"], "ben.png")

    def test_a_node_with_no_references_says_nothing(self):
        g = {"1": {"class_type": "KSampler", "inputs": {"seed": 1}}}
        self.assertEqual(describe_references(g, "1"), "")
        self.assertEqual(describe_references(g, "nope"), "")


class DistinguishingLabelsTest(unittest.TestCase):
    """Every variant was named after the style guide they all shared."""

    STYLE = ("STYLE GUIDE: Monochromatic cool-blue palette, deep navy shadows to "
             "steel blues. ")

    def _labels(self, tails):
        return Pipeline._variant_labels([{"348.prompt": self.STYLE + t} for t in tails])

    def test_the_shared_opening_is_dropped(self):
        got = self._labels(["TANIHO heroic samurai amid battlefield",
                            "HEDI MAZIU aged mentor weathered face",
                            "APE massive knuckle-walking giant"])
        self.assertEqual(got, ["TANIHO heroic samurai amid battlefield",
                               "HEDI MAZIU aged mentor weathered face",
                               "APE massive knuckle-walking giant"])
        self.assertEqual(len(set(got)), 3)

    def test_a_batch_that_shares_nothing_is_untouched(self):
        self.assertEqual(
            Pipeline._variant_labels([{"a.prompt": "a red car"},
                                      {"a.prompt": "a blue boat"}]),
            ["a red car", "a blue boat"])

    def test_identical_prompts_are_left_whole_rather_than_reduced_to_a_tail(self):
        got = self._labels(["", ""])
        self.assertTrue(all(g.startswith("STYLE GUIDE") for g in got), got)

    def test_a_single_variant_has_nothing_to_share_with(self):
        got = self._labels(["only one"])
        self.assertTrue(got[0].startswith("STYLE GUIDE"), got)

    def test_it_will_not_cut_down_to_a_fragment(self):
        """Two prompts differing by one character keep their full text."""
        got = self._labels(["A", "B"])
        self.assertTrue(all(len(g) > 20 for g in got), got)

    def test_a_seed_sweep_still_has_no_label(self):
        self.assertEqual(Pipeline._variant_labels([{"a.seed": 1}, {"a.seed": 2}]),
                         ["", ""])

    def test_nothing_at_all(self):
        self.assertEqual(Pipeline._variant_labels([]), [])


if __name__ == "__main__":
    unittest.main()
