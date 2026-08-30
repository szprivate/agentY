"""Technical QA requirements, settled by arithmetic instead of by a model.

Half of a QA briefing is not a matter of opinion. *16:9. At least 1080p. Not a
soft render.* Each is decided by a number the file already yields, and asking a
vision model instead is worse three ways: it can be wrong, it costs a round trip,
and the same image can pass on Tuesday and fail on Wednesday.

So the `agentY qa briefing` node puts those on dropdowns and switches, they are
evaluated here, and the model is told the answers and told not to re-judge them.

    python -m unittest discover -s tests
"""

import unittest

from src.utils.qa import QaBriefing, briefing_from_hooks
from src.utils.qa_checks import describe, evaluate, render_for_model


def facts(**over):
    """Measured facts for a 1920x1080 output that is sharp, clean and well exposed."""
    base = {
        "width": 1920, "height": 1080,
        "sharpness": {"score": 300.0, "band": "very sharp",
                      "sharpest_region": 900.0, "sharpest_band": "very sharp"},
        "noise": {"sigma": 1.0, "band": "clean"},
        "exposure": {"mean": 120.0, "contrast": 50.0,
                     "clipped_black": 0.0, "clipped_white": 0.0},
    }
    base.update(over)
    return base


def verdicts(spec, f):
    return {r["criterion"]: r["result"] for r in evaluate(spec, f)}


class AspectRatioTest(unittest.TestCase):

    def test_a_matching_ratio_passes(self):
        self.assertEqual(verdicts({"aspect_ratio": "16:9"}, facts()),
                         {"aspect ratio 16:9": "pass"})

    def test_the_wrong_way_round_fails(self):
        f = facts(width=1080, height=1920)
        self.assertEqual(verdicts({"aspect_ratio": "16:9"}, f)["aspect ratio 16:9"],
                         "fail")

    def test_a_near_miss_is_allowed(self):
        """1312x736 is 1.7826 where 16:9 is 1.7778 — a rounding, not a mistake.

        Real renders land on sizes divisible by 8 or 64; failing them for the
        fourth decimal would fail almost everything agentY produces.
        """
        f = facts(width=1312, height=736)
        self.assertEqual(verdicts({"aspect_ratio": "16:9"}, f)["aspect ratio 16:9"],
                         "pass")

    def test_a_ratio_that_is_genuinely_different_still_fails(self):
        f = facts(width=1600, height=1000)          # 1.6, not 1.778
        self.assertEqual(verdicts({"aspect_ratio": "16:9"}, f)["aspect ratio 16:9"],
                         "fail")

    def test_the_note_carries_both_numbers(self):
        row = evaluate({"aspect_ratio": "16:9"}, facts(width=1080, height=1920))[0]
        self.assertIn("1080x1920", row["note"])
        self.assertIn("1.778", row["note"])


class ResolutionTest(unittest.TestCase):

    def test_the_short_side_is_what_counts(self):
        """'1080p' means 1080 tall for landscape and 1080 WIDE for portrait."""
        self.assertEqual(verdicts({"resolution": "1080p"}, facts())["at least 1080p"],
                         "pass")
        portrait = facts(width=1080, height=1920)
        self.assertEqual(verdicts({"resolution": "1080p"}, portrait)["at least 1080p"],
                         "pass")

    def test_a_smaller_output_fails(self):
        f = facts(width=1312, height=736)
        row = evaluate({"resolution": "1080p"}, f)[0]
        self.assertEqual(row["result"], "fail")
        self.assertIn("736", row["note"])


class SharpnessTest(unittest.TestCase):

    def test_a_sharp_render_passes(self):
        self.assertEqual(len(evaluate({"sharpness": "must be sharp"}, facts())), 1)
        self.assertEqual(evaluate({"sharpness": "must be sharp"}, facts())[0]["result"],
                         "pass")

    def test_a_uniformly_soft_render_fails(self):
        f = facts(sharpness={"score": 11.0, "band": "very soft",
                             "sharpest_region": 27.0, "sharpest_band": "soft"})
        row = evaluate({"sharpness": "must be sharp"}, f)[0]
        self.assertEqual(row["result"], "fail")
        self.assertIn("whole frame is soft", row["note"])

    def test_a_shallow_depth_of_field_passes(self):
        """The case this check would otherwise get exactly backwards.

        A portrait with a soft background reads soft overall. Failing it would
        reject the picture the briefing usually wanted.
        """
        f = facts(sharpness={"score": 40.0, "band": "soft",
                             "sharpest_region": 900.0, "sharpest_band": "very sharp"})
        row = evaluate({"sharpness": "must be sharp"}, f)[0]
        self.assertEqual(row["result"], "pass")
        self.assertIn("depth of field", row["note"])


class GrainAndExposureTest(unittest.TestCase):

    def test_a_grainy_output_fails_a_clean_requirement(self):
        f = facts(noise={"sigma": 12.0, "band": "very grainy"})
        self.assertEqual(evaluate({"grain": "must be clean"}, f)[0]["result"], "fail")

    def test_light_grain_still_counts_as_clean(self):
        f = facts(noise={"sigma": 2.5, "band": "light grain"})
        self.assertEqual(evaluate({"grain": "must be clean"}, f)[0]["result"], "pass")

    def test_a_blown_output_fails(self):
        f = facts(exposure={"mean": 200, "contrast": 30,
                            "clipped_black": 0.0, "clipped_white": 0.35})
        row = evaluate({"no_clipping": True}, f)[0]
        self.assertEqual(row["result"], "fail")
        self.assertIn("35.0% blown", row["note"])

    def test_a_little_clipping_is_allowed(self):
        """A light source in frame clips. A threshold, not zero."""
        f = facts(exposure={"mean": 120, "contrast": 50,
                            "clipped_black": 0.005, "clipped_white": 0.01})
        self.assertEqual(evaluate({"no_clipping": True}, f)[0]["result"], "pass")


class VideoTest(unittest.TestCase):

    def test_black_frames_and_stalls_are_caught(self):
        f = {"width": 1920, "height": 1080, "black_frames": 2, "frames_sampled": 9,
             "frozen_pairs": 3}
        got = verdicts({"no_black_frames": True, "no_stalled_motion": True}, f)
        self.assertEqual(got["no black frames"], "fail")
        self.assertEqual(got["the clip must not stall"], "fail")

    def test_a_clean_clip_passes_both(self):
        f = {"width": 1920, "height": 1080, "black_frames": 0, "frames_sampled": 9,
             "frozen_pairs": 0}
        got = verdicts({"no_black_frames": True, "no_stalled_motion": True}, f)
        self.assertEqual(set(got.values()), {"pass"})

    def test_video_checks_do_not_fire_on_a_still(self):
        """A still has no frames to be black, and must not fail for it."""
        self.assertEqual(evaluate({"no_black_frames": True, "no_stalled_motion": True},
                                  facts()), [])


class NothingAskedTest(unittest.TestCase):

    def test_controls_left_alone_check_nothing(self):
        spec = {"aspect_ratio": "any", "resolution": "any", "sharpness": "any",
                "grain": "any", "no_clipping": False, "no_black_frames": False,
                "no_stalled_motion": False}
        self.assertEqual(evaluate(spec, facts()), [])

    def test_an_unmeasurable_file_yields_no_verdicts(self):
        """Doubt does not condemn — that is QA's whole disposition."""
        self.assertEqual(evaluate({"aspect_ratio": "16:9", "sharpness": "must be sharp"},
                                  {}), [])

    def test_a_missing_fact_skips_only_its_own_check(self):
        f = {"width": 1920, "height": 1080}          # no sharpness measured
        got = verdicts({"aspect_ratio": "16:9", "sharpness": "must be sharp"}, f)
        self.assertEqual(got, {"aspect ratio 16:9": "pass"})

    def test_an_unknown_key_is_ignored_rather_than_fatal(self):
        """A graph saved by a newer node must still run here."""
        got = verdicts({"aspect_ratio": "16:9", "something_new": "yes"}, facts())
        self.assertEqual(got, {"aspect ratio 16:9": "pass"})

    def test_junk_in_place_of_a_spec_is_survivable(self):
        for junk in (None, "", [], "16:9"):
            self.assertEqual(evaluate(junk, facts()), [], repr(junk))


class TellingTheModelTest(unittest.TestCase):

    def test_the_settled_results_are_handed_over_as_decided(self):
        text = render_for_model(evaluate({"aspect_ratio": "16:9"}, facts()))
        self.assertIn("ALREADY DECIDED", text)
        self.assertIn("Do not re-judge", text)
        self.assertIn("aspect ratio 16:9: PASS", text)

    def test_nothing_settled_says_nothing(self):
        self.assertEqual(render_for_model([]), "")

    def test_the_requirements_are_written_into_the_criteria(self):
        """The briefing is what the user reads back; a missing line looks dropped."""
        text = describe({"aspect_ratio": "16:9", "resolution": "1080p",
                         "sharpness": "must be sharp", "no_clipping": True})
        for expected in ("16:9", "1080p", "must be sharp", "blown highlights"):
            self.assertIn(expected, text)

    def test_unset_controls_are_not_written_into_the_criteria(self):
        self.assertEqual(describe({"aspect_ratio": "any", "no_clipping": False}), "")


class FromTheNodeTest(unittest.TestCase):
    """The node reaches the briefing as a qa hook, because that is what it is."""

    def _hook(self, **over):
        h = {"hook_node_id": "9", "purpose": "qa", "directive": "match the reference",
             "technical": {"aspect_ratio": "16:9", "sharpness": "must be sharp",
                           "grain": "any", "no_clipping": True},
             "retries": 2, "anchors": []}
        h.update(over)
        return h

    def test_the_controls_reach_the_briefing(self):
        b = briefing_from_hooks([self._hook()])
        self.assertEqual(b.technical, {"aspect_ratio": "16:9",
                                       "sharpness": "must be sharp",
                                       "no_clipping": True})

    def test_the_prose_and_the_controls_both_reach_the_criteria(self):
        b = briefing_from_hooks([self._hook()])
        self.assertIn("match the reference", b.criteria)
        self.assertIn("aspect ratio 16:9", b.criteria)

    def test_the_retry_count_is_taken_from_the_node(self):
        self.assertEqual(briefing_from_hooks([self._hook()]).retry_budget, 2)

    def test_written_retry_syntax_still_wins(self):
        """A briefing that says `retry: 5` in its own words meant it."""
        b = briefing_from_hooks([self._hook(directive="looks right. retry: 5")])
        self.assertEqual(b.retry_budget, 5)

    def test_a_hand_written_qa_hook_is_unaffected(self):
        b = briefing_from_hooks([{"purpose": "qa", "directive": "warm light",
                                  "anchors": []}])
        self.assertEqual(b.technical, {})
        self.assertEqual(b.criteria, "warm light")

    def test_two_briefings_merge_their_controls(self):
        a = QaBriefing(criteria="a", technical={"aspect_ratio": "16:9"})
        b = QaBriefing(criteria="b", technical={"grain": "must be clean"})
        merged = a.merged_with(b)
        self.assertEqual(merged.technical,
                         {"aspect_ratio": "16:9", "grain": "must be clean"})

    def test_the_nearer_briefing_wins_a_disagreement(self):
        a = QaBriefing(criteria="a", technical={"aspect_ratio": "16:9"})
        b = QaBriefing(criteria="b", technical={"aspect_ratio": "1:1"})
        self.assertEqual(a.merged_with(b).technical["aspect_ratio"], "16:9")

    def test_a_briefing_of_only_controls_still_counts_as_a_briefing(self):
        """Otherwise ticking boxes and writing nothing turns QA off."""
        self.assertTrue(QaBriefing(technical={"aspect_ratio": "16:9"}))
        self.assertFalse(QaBriefing())


if __name__ == "__main__":
    unittest.main()
