"""Changing what was made, while the chain is stopped on it.

A review halt was a yes/no gate: continue with what you have, or throw the run
away. Every execution tool was shut, including the ones a revision is made of —
so "regenerate the third one, warmer", which is the single most obvious thing to
say at a stop, was refused by the machinery and encouraged by the prompt at the
same time.

What the halt actually protects is the chain ADVANCING: the stages after the
hook are the expensive ones. So the line is *when*, not *what* — work that runs
inline, this turn, with its result going back into the collector, is the review
doing its job. Work queued to run after the turn is the chain advancing, and
that still waits.

    python -m unittest discover -s tests
"""

import json
import unittest
from unittest import mock

from pipeline_stub import pipeline_stub, tools
from src.utils.review_gate import ReviewHalt, execution_refusal


def _halted(**over):
    over.setdefault("_review_halt", ReviewHalt(
        hook_node_id="11", collector_key="agentY_review_11",
        produced=("a.png", "b.png"), question="which?", remaining=("12",)))
    over.setdefault("_review_reply", "")
    return pipeline_stub(**over)


class TheGateTest(unittest.TestCase):
    """`inline` is the whole distinction."""

    def test_queued_work_still_waits_for_the_user(self):
        self.assertIsNotNone(_halted()._review_gate_refusal(announce=False))

    def test_inline_work_is_allowed_while_the_stop_stands(self):
        self.assertIsNone(_halted()._review_gate_refusal(announce=False, inline=True))

    def test_a_continue_opens_everything(self):
        pipe = _halted(_review_reply="continue")
        self.assertIsNone(pipe._review_gate_refusal(announce=False))
        self.assertIsNone(pipe._review_gate_refusal(announce=False, inline=True))

    def test_with_no_halt_nothing_is_gated(self):
        pipe = pipeline_stub(_review_halt=None)
        self.assertIsNone(pipe._review_gate_refusal(announce=False))

    def test_a_stop_does_not_open_the_gate(self):
        """`stop` ends the run; it does not mean 'go ahead'."""
        self.assertIsNotNone(
            _halted(_review_reply="stop")._review_gate_refusal(announce=False))


class WhatTheRefusalSaysTest(unittest.TestCase):
    """An agent told only "no" ends the turn, and the user is left with two
    moves when they asked for a third."""

    def setUp(self):
        self.out = execution_refusal(ReviewHalt(hook_node_id="11",
                                                produced=("a.png",)))

    def test_it_says_a_change_can_be_made_now(self):
        self.assertIn("CHANGE", self.out["what_to_do"])

    def test_it_names_the_tools_that_still_work(self):
        for tool in ("run_workflow_now", "run_now=True", "iterate_step"):
            self.assertIn(tool, self.out["what_to_do"], tool)

    def test_it_says_what_is_actually_shut(self):
        self.assertIn("QUEUING", self.out["what_to_do"])

    def test_it_is_still_not_reported_as_a_failure(self):
        self.assertIn("not", self.out["do_not"].lower())
        self.assertIn("pause", self.out["do_not"])


class ThroughTheToolsTest(unittest.TestCase):
    """The rule where it is actually applied."""

    def _call(self, pipe, name, **kw):
        import asyncio
        return json.loads(asyncio.run(tools(pipe)[name](**kw)))

    def test_queuing_a_hook_run_is_refused(self):
        out = self._call(_halted(), "apply_canvas_hooks", resolutions=[])
        self.assertIn("stopped at review hook", out["error"])

    def test_running_one_inline_is_not(self):
        """`run_now=True` is a revision: it renders now and the user sees it."""
        pipe = _halted()

        async def ran(paths, notes, labels=None, hook_id=""):
            return json.dumps({"status": "ran"})

        pipe._run_canvas_batch = ran
        out = self._call(pipe, "apply_canvas_hooks", resolutions=[], run_now=True)
        self.assertNotIn("stopped at review hook", json.dumps(out))

    def test_iterate_step_is_open_because_it_IS_the_feedback_loop(self):
        out = self._call(_halted(), "iterate_step", prompt="warmer")
        self.assertNotIn("stopped at review hook", json.dumps(out))

    def test_run_workflow_now_is_open(self):
        out = self._call(_halted(), "run_workflow_now", workflow_path="nope.json")
        self.assertNotIn("stopped at review hook", json.dumps(out))

    def test_the_end_of_turn_executor_is_still_held(self):
        """Queued work runs after the turn, which is the chain advancing — the
        one thing the stop exists to prevent."""
        pipe = _halted()
        self.assertIsNotNone(pipe._review_gate_refusal(announce=False))


class TheHaltSurvivesTheLoopTest(unittest.TestCase):
    """Revising is not continuing. Ten rounds of it is the stop doing its job."""

    def test_a_revision_turn_leaves_the_halt_standing(self):
        from src.utils.models import AgentSession
        pipe = _halted(_session=AgentSession(session_id="t"), _review_armed=None)
        pipe._arm_review_halt()
        self.assertEqual(pipe._session.review_halt["hook_node_id"], "11")

    def test_an_answered_halt_is_spent(self):
        from src.utils.models import AgentSession
        pipe = _halted(_session=AgentSession(session_id="t"), _review_armed=None,
                       _review_reply="continue")
        pipe._arm_review_halt()
        self.assertIsNone(pipe._session.review_halt)


class ThePromptTeachesItTest(unittest.TestCase):
    """The machinery allows the loop; the prompt is what makes it happen."""

    def setUp(self):
        from src.pipeline import _orch_partial
        self.text = _orch_partial("review_halt")

    def test_the_loop_is_described(self):
        self.assertIn("Revising during the halt", self.text)
        self.assertIn("as many times as they like", self.text)

    def test_it_covers_more_than_images(self):
        for kind in ("images", "video", "audio"):
            self.assertIn(kind, self.text, kind)

    def test_it_says_which_tools_still_run(self):
        self.assertIn("run_now=True", self.text)
        self.assertIn("iterate_step", self.text)

    def test_the_existing_workflow_is_preferred_over_a_side_graph(self):
        self.assertIn("Prefer the workflow that is already on the canvas", self.text)
        self.assertIn("separate graph", self.text)

    def test_a_side_graph_hands_its_result_back(self):
        self.assertIn("scaffolding", self.text)

    def test_it_no_longer_tells_the_agent_everything_is_refused(self):
        """The old wording — "the execution tools will refuse anyway" — is what
        made an agent stop trying when the user asked for a change."""
        self.assertNotIn("The execution tools will refuse anyway", self.text)


if __name__ == "__main__":
    unittest.main()
