"""The plan is said out loud before it runs — and waits only where asked.

Two rules that share a module and must not share behaviour. Announcing the plan
is unconditional and lives in the prompts. *Waiting* for a yes happens only where
someone asked to be asked, and that part is enforced: the tools that would run
work refuse until the user has answered, so a gated turn ends with the plan
instead of with a finished batch nobody sanctioned.

The sharpest risk is the false positive. A hook directive that says "wait for all
the references to be generated" is a CONDITIONAL hook — what it waits on is an
outcome, not a person — and reading it as an approval request would stall every
film-hook graph on the canvas. Those directives are tested here, as non-matches.

    python -m unittest discover -s tests
"""

import asyncio
import json
import unittest

from pipeline_stub import pipeline_stub, tools as _tools
from src.pipeline import Pipeline
from src.utils.models import AgentSession
from src.utils.plan_gate import (ApprovalRequest, execution_refusal, find_approval_request,
                                 plan_note, waived)
from src.utils.workflow_signal import clear_and_get, set_execution_hold


def _found(text, label="the user's message"):
    return find_approval_request([(label, text)])


class DetectionTest(unittest.TestCase):
    def test_the_shapes_people_actually_write(self):
        for text in [
            "Build the sequence, but wait for my go-ahead before you run anything.",
            "Show me the plan first, then wait for my ok.",
            "Ask me first before you start generating.",
            "Check with me before you queue anything.",
            "Before you start, get my approval.",
            "Don't run anything until I approve it.",
            "My sign-off is required before any generation.",
            "This needs my confirmation first.",
            "Let me approve the shot list.",
            "Hold until I say so.",
            "Present the plan to me and wait.",
        ]:
            with self.subTest(text=text):
                self.assertIsNotNone(_found(text), text)

    def test_a_conditional_hook_is_not_an_approval_request(self):
        """The directives from the film graph: they wait on outcomes, not people."""
        for directive in [
            "Wait for all the references to be generated. If ANY reference generation "
            "failed - STOP, and ask the user for advice.",
            "Only continue once every shot exists.",
            "Wait until all three references are complete, then build the sheet.",
            "If any of them failed, abort and tell me.",
            "Do not proceed if the upscale errored.",
            "Halt if the script names no characters.",
        ]:
            with self.subTest(directive=directive):
                self.assertIsNone(_found(directive, "hook 30's directive"), directive)

    def test_ordinary_instructions_stay_ordinary(self):
        for text in [
            "Break the story down into single shots, max 2 seconds per shot.",
            "Before generating, describe the style and colour of the frame.",
            "Render a non-stop dolly move across the courtyard.",
            "Ask the user for advice if the reference is missing.",
            "Give me five variations of the alley at night.",
            "Wait — make the second one warmer.",
        ]:
            with self.subTest(text=text):
                self.assertIsNone(_found(text), text)

    def test_it_quotes_the_sentence_that_asked_not_the_two_matched_words(self):
        req = _found("Use the Kling template. Show me the plan first and wait for my "
                     "ok, including the prompts. Then render at 1080p.")
        self.assertIn("Show me the plan first", req.quote)
        self.assertIn("including the prompts", req.quote,
                      "the whole ask has to survive, or the agent honours half of it")
        self.assertNotIn("Kling template", req.quote)
        self.assertNotIn("1080p", req.quote)

    def test_the_first_source_wins_and_is_named(self):
        req = find_approval_request([
            ("the user's message", "nothing to see here"),
            ("hook 30's directive", "Ask me first before you start."),
            ("the project's memory", "Wait for my approval."),
        ])
        self.assertEqual(req.source, "hook 30's directive")

    def test_empty_sources_are_skipped(self):
        self.assertIsNone(find_approval_request([("x", ""), ("y", None), ("z", "   ")]))


class WaiverTest(unittest.TestCase):
    def test_the_user_can_suspend_their_own_rule(self):
        for text in ["just do it", "go ahead", "no need to ask this time",
                     "don't ask, run them all", "run it without asking"]:
            with self.subTest(text=text):
                self.assertTrue(waived(text), text)

    def test_a_reply_that_is_only_agreement_is_the_approval(self):
        for text in ["yes", "yes please", "ok", "Perfect.", "approved",
                     "sounds good", "do it", "yep, that's the one"]:
            with self.subTest(text=text):
                self.assertTrue(waived(text), text)

    def test_a_yes_carrying_a_fresh_request_is_not(self):
        """"Yes, and …" is new work — it gets the same plan treatment as any."""
        self.assertFalse(waived("yes, and now build the whole 12-shot sequence from "
                                "the script with the new character sheet"))

    def test_an_ordinary_request_is_not_a_waiver(self):
        for text in ["make me a poster of a lighthouse", "run the batch after the plan",
                     "generate five variations",
                     # "just <verb>" is how people phrase a small request, not a
                     # licence to skip the plan they asked for.
                     "just make it warmer", "just generate the second shot again",
                     "don't stop until all five exist"]:
            with self.subTest(text=text):
                self.assertFalse(waived(text), text)


class _Stub:
    """A Pipeline reduced to the plan-gate surface."""

    _verbose = False
    _detect_plan_approval = Pipeline._detect_plan_approval
    _plan_gate_refusal = Pipeline._plan_gate_refusal
    _arm_plan_gate = Pipeline._arm_plan_gate

    def __init__(self, hooks=(), gate_open=False, approval=None):
        self._canvas_hooks = list(hooks)
        self._plan_gate_open = gate_open
        self._plan_approval = approval
        self._plan_gate_fired = False
        self._session = AgentSession(session_id="t")


def _hook(hid, directive):
    return {"hook_node_id": str(hid), "purpose": "inline_parameter", "directive": directive}


class SourcesTest(unittest.TestCase):
    def test_a_hook_node_can_ask(self):
        req = _Stub(hooks=[_hook(30, "Show me the plan and wait for my go before "
                                     "you run anything.")])._detect_plan_approval("render it")
        self.assertEqual(req.source, "hook 30's directive")

    def test_the_project_memory_can_ask(self):
        req = _Stub()._detect_plan_approval("render it", "policy: ask me first before "
                                                         "you start a paid render.")
        self.assertEqual(req.source, "the project's memory")

    def test_this_turn_can_waive_a_standing_hook_rule(self):
        stub = _Stub(hooks=[_hook(30, "Ask me first before you start.")])
        self.assertIsNotNone(stub._detect_plan_approval("render the sequence"))
        self.assertIsNone(stub._detect_plan_approval("just do it, no need to ask"),
                          "their own rule is theirs to suspend for a turn")

    def test_nobody_asked_means_no_gate(self):
        self.assertIsNone(_Stub(hooks=[_hook(30, "Caption the image in anchor_0.")])
                          ._detect_plan_approval("make me a poster"))


class GateStateTest(unittest.TestCase):
    def setUp(self):
        self.req = ApprovalRequest(source="hook 30's directive", quote="Ask me first.")

    def test_no_request_means_the_gate_is_open(self):
        self.assertIsNone(_Stub()._plan_gate_refusal(announce=False))

    def test_a_pending_request_refuses(self):
        out = _Stub(approval=self.req)._plan_gate_refusal(announce=False)
        self.assertIn("has not answered", out["error"])
        self.assertEqual(out["asked_by"], "hook 30's directive")
        self.assertIn("numbered list", out["what_to_do"])
        self.assertIn("Do not report this as a failure", out["do_not"])

    def test_an_answered_request_lets_the_work_through(self):
        self.assertIsNone(_Stub(approval=self.req, gate_open=True)
                          ._plan_gate_refusal(announce=False))

    def test_a_gate_that_stopped_something_hands_the_ball_to_the_user(self):
        stub = _Stub(approval=self.req)
        stub._plan_gate_refusal()                 # a tool asked to run; it was refused
        stub._arm_plan_gate()
        self.assertTrue(stub._session.plan_awaiting_reply,
                        "their next message is what re-opens the gate")

    def test_a_gated_turn_that_ran_nothing_leaves_the_gate_shut(self):
        """A question about the graph is not a plan put to the user."""
        stub = _Stub(approval=self.req)
        stub._arm_plan_gate()
        self.assertFalse(stub._session.plan_awaiting_reply,
                         "or chatting once would buy a free pass on the next request")

    def test_a_turn_that_ran_re_closes_the_gate_for_the_next_request(self):
        stub = _Stub(approval=self.req, gate_open=True)
        stub._arm_plan_gate()
        self.assertFalse(stub._session.plan_awaiting_reply)

    def test_an_ungated_turn_never_arms_it(self):
        stub = _Stub()
        stub._session.plan_awaiting_reply = True
        stub._arm_plan_gate()
        self.assertTrue(stub._session.plan_awaiting_reply,
                        "an untouched flag belongs to whoever set it")

    def test_the_signal_hold_arms_it_too(self):
        """signal_workflow_ready is refused inside the mailbox, out of `self`'s sight."""
        from src.tools.workflow_handoff import signal_workflow_ready
        stub = _Stub(approval=self.req)
        set_execution_hold(execution_refusal(self.req))
        self.addCleanup(set_execution_hold, None)
        self.addCleanup(clear_and_get)
        signal_workflow_ready("C:/tmp/whatever.json")
        stub._arm_plan_gate()
        self.assertTrue(stub._session.plan_awaiting_reply)

    def test_the_flag_survives_a_session_round_trip(self):
        s = AgentSession(session_id="t")
        s.plan_awaiting_reply = True
        self.assertTrue(AgentSession(**s.model_dump()).plan_awaiting_reply)

    def test_an_older_stored_session_still_loads(self):
        old = {"session_id": "t", "chat_summaries": [], "current_output_paths": []}
        self.assertFalse(AgentSession(**old).plan_awaiting_reply)


def _pipe(**over):
    """The Pipeline stand-in, with a plan waiting to be approved unless told otherwise."""
    over.setdefault("_plan_approval", ApprovalRequest(
        source="the user's message", quote="Ask me first before you start."))
    return pipeline_stub(**over)


def _call(tool, **kw):
    return json.loads(asyncio.run(tool(**kw)))


class ThroughTheToolsTest(unittest.TestCase):
    """The part that matters: nothing runs, and the agent is told why in-turn."""

    def setUp(self):
        clear_and_get()
        set_execution_hold(None)
        self.addCleanup(clear_and_get)
        self.addCleanup(set_execution_hold, None)

    def test_apply_canvas_hooks_queues_nothing_while_the_plan_is_pending(self):
        out = _call(_tools(_pipe())["apply_canvas_hooks"], resolutions=[
            {"target_node_id": "1", "param": "seed", "mode": "sweep_seed", "count": 3}])
        self.assertIn("error", out)
        self.assertIn("not yet", out["error"])
        self.assertEqual(clear_and_get(), [], "a refused call must queue nothing")

    def test_the_same_call_goes_through_once_they_have_answered(self):
        out = _call(_tools(_pipe(_plan_gate_open=True))["apply_canvas_hooks"], resolutions=[
            {"target_node_id": "1", "param": "seed", "mode": "sweep_seed", "count": 3}])
        self.assertEqual(out.get("status"), "queued")
        self.assertEqual(len(clear_and_get()), 3)

    def test_run_workflow_now_refuses_too(self):
        out = _call(_tools(_pipe())["run_workflow_now"], workflow_path="C:/tmp/stage2.json")
        self.assertIn("not yet", out["error"])

    def test_iterate_step_refuses_before_it_touches_the_graph(self):
        out = _call(_tools(_pipe())["iterate_step"], prompt="make it warmer")
        self.assertIn("not yet", out["error"])

    def test_signal_workflow_ready_refuses_through_the_mailbox_hold(self):
        from src.tools.workflow_handoff import signal_workflow_ready
        set_execution_hold(execution_refusal(
            ApprovalRequest(source="the user's message", quote="Wait for my go.")))
        out = json.loads(signal_workflow_ready("C:/tmp/whatever.json"))
        self.assertIn("not yet", out["error"])
        self.assertEqual(clear_and_get(), [])

    def test_lifting_the_hold_lets_a_real_path_through(self):
        import tempfile
        from pathlib import Path
        from src.tools.workflow_handoff import signal_workflow_ready
        p = Path(tempfile.mkdtemp()) / "wf.json"
        p.write_text("{}", encoding="utf-8")
        set_execution_hold(None)
        self.assertEqual(json.loads(signal_workflow_ready(str(p)))["status"], "ready")
        self.assertEqual(len(clear_and_get()), 1)

    def test_the_keep_live_canvas_run_is_held_as_well(self):
        """No tool queues it — a producer's injection does — so it stops here."""
        pipe = _pipe(_canvas_keeplive_run=True)
        self.assertEqual(Pipeline._pending_execution_paths(pipe), [])
        self.assertFalse(pipe._canvas_keeplive_run)

    def test_an_approved_turn_runs_the_keep_live_canvas_normally(self):
        pipe = _pipe(_canvas_keeplive_run=True, _plan_gate_open=True)
        self.assertEqual(Pipeline._pending_execution_paths(pipe), [])
        self.assertTrue(pipe._canvas_keeplive_run, "nothing to hold: they said yes")


class PlanNoteTest(unittest.TestCase):
    """run_planner hands the instruction back with the plan it is about."""

    def test_an_ungated_plan_is_announced_and_carried_out(self):
        note = plan_note(None)
        self.assertIn("say this plan to the user", note)
        self.assertIn("announcement, not a question", note)
        self.assertNotIn("STOP", note)

    def test_a_gated_plan_stops_after_saying_it(self):
        note = plan_note(ApprovalRequest(source="hook 30's directive",
                                         quote="Wait for my go."))
        self.assertIn("STOP", note)
        self.assertIn("hook 30's directive", note)
        self.assertIn("Wait for my go.", note)

    def test_an_answered_gate_reads_like_an_ungated_one(self):
        note = plan_note(ApprovalRequest(source="x", quote="y"), answered=True)
        self.assertIn("get on with step 1", note)
        self.assertNotIn("STOP", note)


if __name__ == "__main__":
    unittest.main()
