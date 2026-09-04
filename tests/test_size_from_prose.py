"""Saying the shape in words, and finding the knob that sets it.

From the report: *"the qa checker correctly identified that the aspect ratio of
the generated image was wrong, but had no way of correcting it — all it would
have had to do would be to set the `size_preset` on the node to a different
value. That might be a general problem with comfyui — there's no standard in
regards how parameters are called."*

Two gaps, and they only close together.

**The requirement never arrived.** `qa_repair` is handed `technical`, which comes
from the briefing node's dropdowns and nothing else. A briefing that said "16:9"
in prose was measured and failed — the ratio is computed either way — and then
the retry rerolled the seed and rewrote the prompt, neither of which has ever
changed an image's dimensions. `infer_technical` reads the requirement out of the
words, so the repair path hears about it.

**And the knob is not always called what we expect.** `governing_params` matched
a fixed list of names. There is no naming standard, so the next node pack is free
to call it `output_size` and be missed while its options sit there reading
`1024x1024, 1280x720`. It now asks the options when the name misses.

Either alone leaves the reported run broken: a requirement nobody can hear, or a
knob nobody can find. The last test here is the two of them meeting.

    python -m unittest discover -s tests
"""

import unittest
from unittest import mock

from src.utils.qa_checks import infer_technical
from src.utils.qa_repair import (_sizing_kind, apply_fix, describe_fix,
                                 governing_params, plan_fixes)


class ReadingTheRequirementOutOfTheWords(unittest.TestCase):
    """`infer_technical` — narrow on purpose."""

    def test_a_stated_ratio_is_read(self):
        self.assertEqual(infer_technical("make it 16:9"), {"aspect_ratio": "16:9"})

    def test_a_pixel_size_states_both(self):
        self.assertEqual(infer_technical("a 1920x1080 hero shot"),
                         {"aspect_ratio": "16:9", "resolution": "1080p"})

    def test_named_resolutions_are_read(self):
        self.assertEqual(infer_technical("shoot at 4K")["resolution"], "2160p (4K)")
        self.assertEqual(infer_technical("1080p please")["resolution"], "1080p")

    def test_an_odd_size_invents_nothing(self):
        # 1000x777 is no named ratio. The closest guess is still a guess.
        self.assertEqual(infer_technical("1000x777"), {})

    def test_two_different_ratios_yield_none(self):
        """Which one they meant is a question for the user, not a coin toss."""
        self.assertEqual(infer_technical("16:9 for the hero, 9:16 for the story"), {})

    def test_the_largest_stated_resolution_wins(self):
        # "at least" is a floor, so the biggest floor satisfies the others.
        self.assertEqual(infer_technical("720p minimum, ideally 4K")["resolution"],
                         "2160p (4K)")

    def test_prose_that_states_no_shape_is_left_alone(self):
        for text in ("the subject reads as a trans woman in her 40s",
                     "cinematic, moody, shallow depth of field",
                     "match the reference for lighting and mood",
                     ""):
            self.assertEqual(infer_technical(text), {}, text)

    def test_a_grid_is_not_an_aspect_ratio(self):
        # The false positive that would silently reshape a render.
        self.assertEqual(infer_technical("a 2x3 grid of variations"), {})
        self.assertEqual(infer_technical("batch of 8x4 tiles"), {})

    def test_several_texts_are_read_together(self):
        # The retry reads the failed criteria, which arrive as separate strings.
        self.assertEqual(infer_technical("too soft", "aspect ratio 16:9"),
                         {"aspect_ratio": "16:9"})

    def test_a_verdict_line_states_two_ratios_and_means_one(self):
        """The wanted shape and the observed one, in the judge's own format.

        `failed_criteria()` renders "<criterion> - <note>", and the note says
        what it GOT. Read whole, that is two ratios and infers nothing — which is
        why the retry splits the note off before asking. Pinned here because the
        split lives at the caller and this is what makes it necessary.
        """
        line = "aspect ratio 16:9 — the image is 1:1"
        self.assertEqual(infer_technical(line), {})
        self.assertEqual(infer_technical(line.split(" — ", 1)[0]),
                         {"aspect_ratio": "16:9"})


class ItNeverOverridesAnExplicitChoice(unittest.TestCase):
    """A dropdown is a decision; prose is a reading of one."""

    def _resolve(self, directive, technical=None):
        from src.utils.qa import resolve_briefing
        hook = {"hook_node_id": "49", "purpose": "qa", "directive": directive,
                "anchors": []}
        if technical:
            hook["technical"] = technical
        with mock.patch("src.utils.qa.qa_settings",
                        return_value={"enabled": True, "forced_off": False}):
            return resolve_briefing(hooks=[hook], thread_id="")

    def test_prose_reaches_the_technical_spec(self):
        self.assertEqual(self._resolve("must be 16:9").technical,
                         {"aspect_ratio": "16:9"})

    def test_the_dropdown_wins_when_they_disagree(self):
        got = self._resolve("must be 16:9", {"aspect_ratio": "4:3"})
        self.assertEqual(got.technical["aspect_ratio"], "4:3")

    def test_the_merge_order_is_what_makes_that_true(self):
        """Not the ambiguity rule, which happens to cover the same case.

        `briefing_from_hooks` appends `describe(technical)` to the criteria, so a
        briefing with the dropdown set says BOTH ratios in its text and infers
        nothing at all. That is a second reason the dropdown survives, and it
        would keep this passing with the precedence reversed — so pin the
        precedence on its own, with criteria that state one ratio cleanly.
        """
        from src.utils.qa import QaBriefing, _infer_technical
        got = _infer_technical(QaBriefing(criteria="must be 16:9",
                                          technical={"aspect_ratio": "4:3"},
                                          sources=("t",)))
        self.assertEqual(got.technical["aspect_ratio"], "4:3")

    def test_inference_still_fills_a_key_the_dropdown_left_unset(self):
        from src.utils.qa import QaBriefing, _infer_technical
        got = _infer_technical(QaBriefing(criteria="16:9 at 1080p",
                                          technical={"sharpness": "sharp"},
                                          sources=("t",)))
        self.assertEqual(got.technical, {"sharpness": "sharp",
                                         "aspect_ratio": "16:9",
                                         "resolution": "1080p"})

    def test_it_is_not_restated_in_the_criteria(self):
        # It is already there in the user's own words; twice reads as two.
        got = self._resolve("must be 16:9")
        self.assertEqual(got.criteria.strip(), "must be 16:9")


class TheRetryReadsTheVerdict(unittest.TestCase):
    """Where the requirement lives in the user's message, not the briefing.

    QA is given the request as well as the criteria, so it can fail an output on
    a shape the briefing never mentions — which is how the reported run went. The
    verdict is then the only place downstream that knows, so the retry reads it.
    `_qa_retry` has no test harness (it wants a pipeline, a graph on disk and a
    model), so this pins the two things that make the reading correct.
    """

    def _source(self) -> str:
        import inspect

        from src import pipeline
        return inspect.getsource(pipeline.Pipeline._qa_retry)

    def test_the_verdict_is_read_at_all(self):
        self.assertIn("infer_technical", self._source())

    def test_the_note_is_split_off_before_reading(self):
        src = self._source()
        self.assertIn('split(" — ", 1)[0]', src)

    def test_the_briefing_still_outranks_the_verdict(self):
        # `{**inferred, **technical}` — the deliberate statement wins.
        self.assertIn("{**infer_technical(*wanted), **technical}", self._source())


class FindingTheKnobWithoutKnowingItsName(unittest.TestCase):
    """`_sizing_kind` — the options say what the parameter is."""

    def test_a_size_menu_is_recognised_under_any_name(self):
        self.assertEqual(
            _sizing_kind("output_size", ["1024x1024", "1280x720", "1920x1080"]),
            "size")

    def test_a_ratio_menu_is_recognised_under_any_name(self):
        self.assertEqual(_sizing_kind("image_shape", ["16:9", "9:16", "1:1"]),
                         "ratio")

    def test_an_unrelated_menu_is_left_alone(self):
        self.assertEqual(_sizing_kind("sampler_name", ["euler", "ddim"]), "")

    def test_one_stray_ratio_does_not_recruit_a_menu(self):
        self.assertEqual(_sizing_kind("mode", ["fast", "slow", "1:1", "auto"]), "")

    def test_names_that_are_never_sizing_are_excluded(self):
        # `batch_size` offering 1:1 and 2:1 must not be read as a shape control.
        self.assertEqual(_sizing_kind("batch_size", ["1:1", "2:1"]), "")

    def test_a_single_option_is_not_a_menu(self):
        self.assertEqual(_sizing_kind("size", ["1024x1024"]), "")

    def test_a_long_menu_needs_most_of_it_to_parse(self):
        # Three shapes buried in a twelve-item menu of something else is not a
        # sizing control; the proportion rule is what says so, since the "at
        # least two" floor is already met.
        opts = ["16:9", "1:1", "4:3"] + [f"preset_{i}" for i in range(9)]
        self.assertEqual(_sizing_kind("mode", opts), "")

    def test_a_menu_that_is_mostly_shapes_still_counts(self):
        opts = ["16:9", "9:16", "1:1", "4:3", "custom"]
        self.assertEqual(_sizing_kind("mode", opts), "ratio")


# A generator that carries its own sizing widget under a name nothing knows,
# feeding a saver. This is the shape of the reported graph.
GRAPH = {
    "1": {"class_type": "MysteryGen",
          "inputs": {"prompt": "a portrait", "output_size": "1024x1024"}},
    "2": {"class_type": "SaveImage", "inputs": {"images": ["1", 0]}},
}
SCHEMA = {
    "MysteryGen": {"input": {"required": {
        "prompt": ["STRING", {}],
        "output_size": [["1024x1024", "1280x720", "1920x1080", "1080x1920"], {}],
    }}},
    "SaveImage": {"input": {"required": {"images": ["IMAGE", {}]}}},
}


class TheTwoHalvesMeeting(unittest.TestCase):
    """The reported run, end to end: prose in, the right widget set."""

    def _schema_of(self, cls):
        return SCHEMA.get(cls, {})

    def test_the_unknown_widget_is_found(self):
        found = governing_params(GRAPH, self._schema_of)
        self.assertTrue(found)
        self.assertEqual(found[0]["param"], "output_size")
        self.assertEqual(found[0]["kind"], "size")
        self.assertEqual(found[0]["by"], "options")

    def test_a_prose_briefing_now_fixes_the_graph(self):
        technical = infer_technical("the render must be 16:9")
        fixes, problems = plan_fixes(GRAPH, technical, self._schema_of)
        self.assertEqual(problems, [])
        self.assertEqual(len(fixes), 1)

        graph = {k: {**v, "inputs": dict(v["inputs"])} for k, v in GRAPH.items()}
        self.assertTrue(apply_fix(graph, fixes[0]))
        self.assertEqual(graph["1"]["inputs"]["output_size"], "1280x720")
        self.assertIn("output_size", describe_fix(fixes[0]))

    def test_the_cheapest_option_that_qualifies_is_chosen(self):
        # 1280x720 and 1920x1080 are both 16:9; spending the larger one is our
        # choice to make on someone else's bill.
        technical = infer_technical("16:9")
        fixes, _ = plan_fixes(GRAPH, technical, self._schema_of)
        self.assertEqual(fixes[0]["to"], "1280x720")

    def test_neither_half_alone_would_have_done_it(self):
        # No prose read → nothing asked → nothing planned.
        self.assertEqual(plan_fixes(GRAPH, {}, self._schema_of), ([], []))
        # No options fallback → the widget is invisible → nothing to plan against.
        with mock.patch("src.utils.qa_repair._sizing_kind", return_value=""):
            fixes, problems = plan_fixes(GRAPH, {"aspect_ratio": "16:9"},
                                         self._schema_of)
        self.assertEqual(fixes, [])
        self.assertTrue(problems)

    def test_a_shape_the_graph_cannot_make_is_reported_not_forced(self):
        technical = {"aspect_ratio": "2.39:1"}
        fixes, problems = plan_fixes(GRAPH, technical, self._schema_of)
        self.assertEqual(fixes, [])
        self.assertEqual(problems[0]["control"], "aspect_ratio")
        self.assertIn("output_size", problems[0]["why"])


if __name__ == "__main__":
    unittest.main()
