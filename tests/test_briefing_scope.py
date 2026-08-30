"""A briefing can name the stage it judges, instead of judging the whole run.

A chain that makes reference frames and then animates them has two stages with
genuinely different standards: the stills want 16:9 and a sharp render, the clip
wants no black frames and no stall. Merging both briefings and applying the pair
to everything is how a still gets failed for not moving.

So a briefing node's `out` wires into a hook's anchor and means *"I judge that
stage"*. What is held down here:

**Unwired still means everything.** That is the common case and the previous
behaviour, and a graph nobody has scoped must behave exactly as it did.

**A briefing naming another stage must not judge this one.** That is the whole
point, and the tempting failure — falling back to "well, use the merged one" —
would silently restore exactly what scoping was for.

**And it can only ever narrow.** Where the stage isn't known, or QA came from
`/qa` rather than the canvas, the turn's briefing still applies. Scoping must
never be the reason an output goes unchecked.

    python -m unittest discover -s tests
"""

import unittest
from unittest import mock

from src.utils.qa import QaBriefing, briefing_for_hook, briefing_from_hooks


def brief(node_id, *, directive="", applies_to=None, **technical):
    """A qa hook as the panel sends one, optionally scoped to some hooks."""
    h = {"hook_node_id": str(node_id), "purpose": "qa", "directive": directive,
         "technical": dict(technical), "anchors": []}
    if applies_to is not None:
        h["applies_to"] = [str(x) for x in applies_to]
    return h


def work(node_id):
    """A hook that does work, which is what a briefing scopes itself to."""
    return {"hook_node_id": str(node_id), "purpose": "make_workflow",
            "directive": "make something", "anchors": []}


class ScopeTest(unittest.TestCase):

    def test_an_unwired_briefing_judges_every_stage(self):
        hooks = [work("5"), work("30"), brief("9", directive="warm light")]
        for hook_id in ("5", "30"):
            b = briefing_for_hook(hooks, hook_id)
            self.assertIn("warm light", b.criteria, hook_id)

    def test_a_briefing_naming_a_stage_judges_only_that_one(self):
        hooks = [work("5"), work("30"),
                 brief("9", directive="16:9 stills", applies_to=["5"])]
        self.assertIn("16:9 stills", briefing_for_hook(hooks, "5").criteria)
        self.assertIsNone(briefing_for_hook(hooks, "30"))

    def test_two_stages_can_have_different_standards(self):
        """The case the feature exists for."""
        hooks = [
            work("5"), work("30"),
            brief("9", directive="sharp stills", applies_to=["5"],
                  aspect_ratio="16:9", sharpness="must be sharp"),
            brief("10", directive="no stalls", applies_to=["30"],
                  no_black_frames=True, no_stalled_motion=True),
        ]
        stills = briefing_for_hook(hooks, "5")
        clip = briefing_for_hook(hooks, "30")
        self.assertEqual(stills.technical,
                         {"aspect_ratio": "16:9", "sharpness": "must be sharp"})
        self.assertEqual(clip.technical,
                         {"no_black_frames": True, "no_stalled_motion": True})
        self.assertIn("sharp stills", stills.criteria)
        self.assertNotIn("no stalls", stills.criteria)

    def test_one_briefing_can_name_several_stages(self):
        hooks = [work("5"), work("30"), work("40"),
                 brief("9", directive="house style", applies_to=["5", "40"])]
        self.assertIn("house style", briefing_for_hook(hooks, "5").criteria)
        self.assertIn("house style", briefing_for_hook(hooks, "40").criteria)
        self.assertIsNone(briefing_for_hook(hooks, "30"))

    def test_a_global_and_a_scoped_briefing_both_apply(self):
        hooks = [work("5"),
                 brief("9", directive="no text anywhere"),
                 brief("10", directive="16:9", applies_to=["5"])]
        b = briefing_for_hook(hooks, "5")
        self.assertIn("no text anywhere", b.criteria)
        self.assertIn("16:9", b.criteria)

    def test_the_briefing_naming_the_stage_wins_a_disagreement(self):
        """The statement about this stage beats the one about the graph."""
        hooks = [work("5"),
                 brief("9", aspect_ratio="1:1"),
                 brief("10", aspect_ratio="16:9", applies_to=["5"])]
        self.assertEqual(briefing_for_hook(hooks, "5").technical["aspect_ratio"],
                         "16:9")

    def test_an_empty_applies_to_is_the_same_as_unwired(self):
        """The panel sends [] for a briefing whose `out` goes nowhere."""
        hooks = [work("5"), brief("9", directive="warm", applies_to=[])]
        self.assertIn("warm", briefing_for_hook(hooks, "5").criteria)

    def test_a_scope_of_blanks_is_no_scope(self):
        """A stale or half-deleted link leaves an entry with nothing in it.

        Counting that as a scope would silently retire the briefing: it would name
        a stage that does not exist and so judge nothing, anywhere.
        """
        hooks = [work("5"), brief("9", directive="warm", applies_to=["", "  "])]
        self.assertIn("warm", briefing_for_hook(hooks, "5").criteria)

    def test_a_blank_hook_id_takes_only_the_unscoped_ones(self):
        hooks = [brief("9", directive="global"),
                 brief("10", directive="scoped", applies_to=["5"])]
        b = briefing_for_hook(hooks, "")
        self.assertIn("global", b.criteria)
        self.assertNotIn("scoped", b.criteria)

    def test_no_briefings_at_all_is_still_no_briefing(self):
        self.assertIsNone(briefing_for_hook([work("5")], "5"))
        self.assertIsNone(briefing_for_hook([], "5"))
        self.assertIsNone(briefing_for_hook(None, "5"))

    def test_junk_among_the_hooks_is_survivable(self):
        hooks = [None, "nonsense", 7, brief("9", directive="warm")]
        self.assertIn("warm", briefing_for_hook(hooks, "5").criteria)

    def test_an_unscoped_canvas_is_unchanged(self):
        """Nothing wired: the scoped lookup and the old one must agree."""
        hooks = [work("5"), brief("9", directive="warm", aspect_ratio="16:9"),
                 brief("10", directive="sharp", sharpness="must be sharp")]
        a = briefing_for_hook(hooks, "5")
        b = briefing_from_hooks(hooks)
        self.assertEqual(a.criteria, b.criteria)
        self.assertEqual(a.technical, b.technical)


class PipelineTest(unittest.TestCase):
    """Where the pipeline chooses which briefing a stage is judged against."""

    def _pipeline(self, hooks, turn_briefing):
        from src.pipeline import Pipeline
        p = object.__new__(Pipeline)
        p._verbose = False
        p._canvas_hooks = hooks
        p._qa_briefing = turn_briefing
        return p

    def test_a_scoped_stage_gets_its_own_briefing(self):
        hooks = [work("5"), brief("9", directive="16:9 stills", applies_to=["5"])]
        p = self._pipeline(hooks, QaBriefing(criteria="everything"))
        with mock.patch("src.utils.agentY_server._resolve_media_ref", lambda *a, **k: None):
            self.assertIn("16:9 stills", p._briefing_for("5").criteria)

    def test_a_stage_no_briefing_names_is_judged_by_nothing(self):
        """NOT by the merged briefing — that is what scoping exists to prevent."""
        hooks = [work("30"), brief("9", directive="16:9 stills", applies_to=["5"])]
        p = self._pipeline(hooks, QaBriefing(criteria="16:9 stills"))
        with mock.patch("src.utils.agentY_server._resolve_media_ref", lambda *a, **k: None):
            self.assertIsNone(p._briefing_for("30"))

    def test_a_slash_qa_briefing_still_applies_with_nothing_on_the_canvas(self):
        """It was never scoped, so scoping must not take it away."""
        p = self._pipeline([work("5")], QaBriefing(criteria="from /qa"))
        with mock.patch("src.utils.agentY_server._resolve_media_ref", lambda *a, **k: None):
            self.assertEqual(p._briefing_for("5").criteria, "from /qa")

    def test_no_hook_id_falls_back_to_the_turn_briefing(self):
        """The queued path has no stage to hand; it must not lose QA for that."""
        hooks = [work("5"), brief("9", directive="16:9", applies_to=["5"])]
        turn = QaBriefing(criteria="16:9")
        self.assertIs(self._pipeline(hooks, turn)._briefing_for(""), turn)

    def test_a_broken_lookup_never_costs_the_output_its_check(self):
        hooks = [work("5"), brief("9", directive="16:9", applies_to=["5"])]
        turn = QaBriefing(criteria="everything")
        p = self._pipeline(hooks, turn)
        with mock.patch("src.utils.qa.briefing_for_hook",
                        side_effect=RuntimeError("boom")):
            self.assertIs(p._briefing_for("5"), turn)


if __name__ == "__main__":
    unittest.main()
