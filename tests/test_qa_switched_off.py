"""A briefing nobody is going to read.

From the report: *"I tried to add a qa node to a workflow, but the orchestrator
just said: QA hook 49 — not mine to run. After the render, the QA agent compares
the output against the reference … So the qa checker agent actually never ran."*

It never ran because `qa.enabled` was false in settings. Nothing was broken; the
switch was off. What was broken is that **nothing said so**. `resolve_briefing`
returns None both for "nobody wrote a briefing" and for "one is wired to the
canvas but QA is switched off", and the `[CANVAS HOOKS]` block promised a QA pass
either way — so the orchestrator faithfully told the user a QA agent would compare
the render against the reference, and no line anywhere admitted that it wouldn't.

    python -m unittest discover -s tests
"""

import unittest
from unittest import mock

from src.utils.canvas_hooks import describe_hooks

QA_HOOK = {
    "hook_node_id": "49",
    "purpose": "qa",
    "directive": "reads as a trans woman in her 40s, not pregnant; match the "
                 "reference for lighting and mood",
    "anchors": [],
}

# The sentence that made the promise. Its absence is the fix.
PROMISE = "A separate QA agent applies it to every image/video the run produces"


def _hooks_block(*, enabled: bool) -> str:
    with mock.patch("src.utils.qa.qa_settings",
                    return_value={"enabled": enabled, "max_retries": 1,
                                  "max_outputs": 6, "max_references": 4,
                                  "video_frames": 3, "briefing_dir": "./config/qa/"}):
        return describe_hooks([dict(QA_HOOK)])


class TheBlockSaysWhichWorldItIsIn(unittest.TestCase):
    """`describe_hooks` is the orchestrator's only account of what QA will do."""

    def test_a_switched_off_briefing_does_not_promise_a_judge(self):
        block = _hooks_block(enabled=False)
        self.assertNotIn(PROMISE, block)
        self.assertIn("NOTHING WILL CHECK", block)

    def test_it_says_where_the_switch_is(self):
        # A user told "QA did not run" and not told why has to go looking.
        self.assertIn("Settings ▸ qa ▸ enabled", _hooks_block(enabled=False))

    def test_the_agent_is_told_to_pass_that_on(self):
        block = _hooks_block(enabled=False)
        self.assertIn("say plainly that QA did NOT run", block)
        self.assertIn("Do not describe a check that will not happen", block)

    def test_a_switched_on_briefing_still_promises_one(self):
        block = _hooks_block(enabled=True)
        self.assertIn(PROMISE, block)
        self.assertNotIn("NOTHING WILL CHECK", block)

    def test_the_criteria_survive_either_way(self):
        # Switched off, satisfying the briefing up front is the ONLY thing left
        # standing behind it — so dropping the criteria would be the worse bug.
        for enabled in (True, False):
            with self.subTest(enabled=enabled):
                block = _hooks_block(enabled=enabled)
                self.assertIn("QA hook 49", block)
                self.assertIn("not pregnant", block)

    def test_neither_branch_offers_the_node_as_work(self):
        for enabled in (True, False):
            with self.subTest(enabled=enabled):
                block = _hooks_block(enabled=enabled)
                self.assertIn("these are NOT work for you", block)
                self.assertIn("apply_canvas_hooks them", block)

    def test_an_unreadable_settings_file_keeps_the_normal_wording(self):
        # Erring the other way would silence a briefing that is on by default.
        from src.utils.canvas_hooks import _qa_will_run
        with mock.patch("src.utils.qa.qa_settings", side_effect=RuntimeError("boom")):
            self.assertTrue(_qa_will_run())


class TheUserHearsAboutItToo(unittest.TestCase):
    """The panel's own line, independent of whatever the orchestrator writes."""

    def _notices(self, *, enabled: bool, hooks: list, thread_briefing=None) -> list:
        from src.utils import agentY_server as srv
        said: list = []
        cfg = {"enabled": enabled, "max_retries": 1, "max_outputs": 6,
               "max_references": 4, "video_frames": 3, "briefing_dir": "./config/qa/"}
        with mock.patch("src.utils.qa.qa_settings", return_value=cfg), \
             mock.patch("src.utils.qa.briefing_from_thread",
                        return_value=thread_briefing), \
             mock.patch.object(srv.status_bus, "notify", said.append):
            srv._warn_qa_is_off(hooks, "t1")
        return said

    def test_a_shelved_canvas_briefing_is_announced(self):
        said = self._notices(enabled=False, hooks=[dict(QA_HOOK)])
        self.assertEqual(len(said), 1)
        self.assertIn("QA is switched off", said[0])
        self.assertIn("will NOT be checked", said[0])

    def test_a_shelved_thread_briefing_counts_as_well(self):
        from src.utils.qa import QaBriefing
        said = self._notices(enabled=False, hooks=[],
                             thread_briefing=QaBriefing(criteria="sharp",
                                                        sources=("thread",)))
        self.assertEqual(len(said), 1)

    def test_no_briefing_no_warning(self):
        # Off with nothing to judge against is not a problem worth a line.
        self.assertEqual(self._notices(enabled=False, hooks=[]), [])

    def test_switched_on_says_nothing_here(self):
        # The active-briefing line is the caller's job; this must not double up.
        self.assertEqual(self._notices(enabled=True, hooks=[dict(QA_HOOK)]), [])

    def test_a_broken_lookup_never_costs_the_turn(self):
        from src.utils import agentY_server as srv
        with mock.patch("src.utils.qa.qa_settings", side_effect=RuntimeError("boom")), \
             mock.patch.object(srv.status_bus, "notify",
                               side_effect=AssertionError("should not speak")):
            srv._warn_qa_is_off([dict(QA_HOOK)], "t1")  # must not raise


if __name__ == "__main__":
    unittest.main()
