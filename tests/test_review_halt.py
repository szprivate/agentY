"""Stop between two stages, and let the user pick what goes on to the next one.

A chain that makes reference frames and then feeds them into a video runs the
whole way through, every time. The video is the expensive half, and by the time
you have seen the references it has already been paid for.

A `review` hook is a break in that chain. The stage before it runs, what it made
is gathered into an `agentY image collector` on the canvas, and the turn ENDS
with the question put to the user. They edit that node — drop rows, add their
own files, reorder — and say continue or stop.

The load-bearing decision, and most of what is tested here: **the answer lives on
the canvas, not in the session.** Only the flag is remembered. The list is read
off the collector at resume, because the user is expected to edit it while the
chain is stopped — that is the entire point — and anything cached at halt time is
a list that has probably been overtaken by then.

    python -m unittest discover -s tests
"""

import json
import unittest
from unittest import mock

from pipeline_stub import pipeline_stub, tools
from src.pipeline import Pipeline
from src.utils.canvas_hooks import _is_qa, _is_review, describe_hooks, gated_by_review
from src.utils.models import AgentSession
from src.utils.review_gate import ReviewHalt, execution_refusal, read_reply


def _hook(hid, purpose, prev=None, directive="do the thing"):
    h = {"hook_node_id": str(hid), "purpose": purpose, "directive": directive,
         "anchors": [], "targets": []}
    if prev is not None:
        h["prev_hook_id"] = str(prev)
        h["prev_hook_ids"] = [str(prev)]
        h["anchors"] = [{"node_id": str(prev), "to_input": "anchors.anchor0"}]
    return h


# refs (10) -> review (11) -> video (12)
def _chain():
    return [_hook(10, "make_workflow", directive="one reference per character"),
            _hook(11, "review", prev=10, directive="which of these should the video use?"),
            _hook(12, "make_workflow", prev=11, directive="animate the chosen refs")]


class PurposeTest(unittest.TestCase):

    def test_review_is_its_own_purpose_now(self):
        """It used to be a tolerant alias for `qa`, which would swallow it whole."""
        h = _hook(11, "review")
        self.assertTrue(_is_review(h))
        self.assertFalse(_is_qa(h), "a review hook is not a QA briefing")

    def test_qa_keeps_its_own_spellings(self):
        for p in ("qa", "quality", "check", "qa_check"):
            with self.subTest(p=p):
                self.assertTrue(_is_qa(_hook(9, p)))
                self.assertFalse(_is_review(_hook(9, p)))

    def test_the_spellings_a_user_might_type(self):
        for p in ("review", "halt", "pause", "check_in"):
            with self.subTest(p=p):
                self.assertTrue(_is_review(_hook(11, p)))


class GatingTest(unittest.TestCase):
    """Which hooks a review hook stands in front of."""

    def test_everything_downstream_is_gated(self):
        self.assertEqual(gated_by_review(_chain()), {"12"})

    def test_it_gates_transitively(self):
        chain = _chain() + [_hook(13, "make_workflow", prev=12)]
        self.assertEqual(gated_by_review(chain), {"12", "13"})

    def test_the_stage_before_it_is_not_gated(self):
        """It has to run — a review of nothing is not a review."""
        self.assertNotIn("10", gated_by_review(_chain()))

    def test_an_independent_branch_is_not_gated(self):
        """Two chains on one canvas are two things; stopping one is not stopping both."""
        other = [_hook(20, "make_workflow"), _hook(21, "make_workflow", prev=20)]
        self.assertEqual(gated_by_review(_chain() + other), {"12"})

    def test_no_review_hook_gates_nothing(self):
        self.assertEqual(gated_by_review([_hook(10, "make_workflow"),
                                          _hook(12, "make_workflow", prev=10)]), set())


class ReplyTest(unittest.TestCase):

    def test_the_two_answers(self):
        for text in ("continue", "Continue", "continue with these", "proceed",
                     "carry on", "yes", "ok", "go ahead"):
            with self.subTest(text=text):
                self.assertEqual(read_reply(text), "continue")
        for text in ("stop", "STOP", "no", "cancel", "abort", "discard"):
            with self.subTest(text=text):
                self.assertEqual(read_reply(text), "stop")

    def test_anything_else_is_not_an_answer(self):
        """Reading a request as a yes spends the budget nobody approved."""
        for text in ("make the third one warmer", "what did the second one cost?",
                     "why is the ape missing?", ""):
            with self.subTest(text=text):
                self.assertEqual(read_reply(text), "")

    def test_an_answer_that_carries_its_edit_still_answers(self):
        """Saying it is quicker than editing the node, and people do."""
        self.assertEqual(read_reply("continue, but drop the second one"), "continue")
        self.assertEqual(read_reply("stop, they're all wrong"), "stop")

    def test_a_long_message_opening_with_continue_is_not_swallowed(self):
        long = ("continue " + "and also please rewrite the whole style guide "
                "so that every shot is warmer and the grade is different " * 2)
        self.assertEqual(read_reply(long), "",
                         "that is a fresh instruction, not an answer to the halt")


class BlockTest(unittest.TestCase):
    """What the agent is told when the graph has a review hook in it."""

    def test_the_review_hook_is_announced_with_its_question(self):
        block = describe_hooks(_chain(), {})
        self.assertIn("REVIEW HOOK", block)
        self.assertIn("halt_for_review", block)
        self.assertIn("which of these should the video use?", block)

    def test_the_gated_hooks_are_named_as_not_this_turn(self):
        block = describe_hooks(_chain(), {})
        self.assertIn("NOT this turn", block)
        self.assertIn("hook(s) 12", block)

    def test_it_says_to_read_the_collector_as_it_stands_then(self):
        self.assertIn("as it stands THEN", describe_hooks(_chain(), {}))

    def test_a_graph_without_one_gains_nothing(self):
        block = describe_hooks([_hook(10, "make_workflow")], {})
        self.assertNotIn("REVIEW HOOK", block)


class GateTest(unittest.TestCase):
    """The execution tools while a halt is unanswered."""

    def setUp(self):
        self.halt = ReviewHalt(hook_node_id="11", collector_key="agentY_review_11",
                               produced=("a.png", "b.png"))

    def test_a_refusal_is_a_pause_not_a_failure(self):
        out = execution_refusal(self.halt)
        self.assertIn("not yet", out["error"])
        self.assertIn("Do not report this as a failure", out["do_not"])

    def test_it_says_to_read_the_collector_fresh_on_resume(self):
        self.assertIn("AT THAT POINT", execution_refusal(self.halt)["after"])

    def test_an_unanswered_halt_shuts_the_run_tools(self):
        pipe = pipeline_stub(_review_halt=self.halt, _review_reply="")
        self.assertIsNotNone(Pipeline._review_gate_refusal(pipe, announce=False))

    def test_a_continue_opens_them(self):
        pipe = pipeline_stub(_review_halt=self.halt, _review_reply="continue")
        self.assertIsNone(Pipeline._review_gate_refusal(pipe, announce=False))

    def test_a_stop_does_NOT_open_them(self):
        """Stop means stop — it is not permission to run the rest."""
        pipe = pipeline_stub(_review_halt=self.halt, _review_reply="stop")
        self.assertIsNotNone(Pipeline._review_gate_refusal(pipe, announce=False))

    def test_no_halt_refuses_nothing(self):
        self.assertIsNone(Pipeline._review_gate_refusal(pipeline_stub(), announce=False))

    def test_the_keep_live_run_is_held_too(self):
        """It is queued by an injection, so the tool refusals never see it."""
        from src.utils.workflow_signal import clear_and_get
        self.addCleanup(clear_and_get)
        pipe = pipeline_stub(_review_halt=self.halt, _review_reply="",
                             _canvas_keeplive_run=True)
        self.assertEqual(Pipeline._pending_execution_paths(pipe), [])
        self.assertFalse(pipe._canvas_keeplive_run)


class HaltToolTest(unittest.TestCase):

    def _pipe(self, **over):
        over.setdefault("_canvas_hooks", _chain())
        over.setdefault("_chain_output_paths", ["C:/out/ref_1.png", "C:/out/ref_2.png"])
        return pipeline_stub(**over)

    def _call(self, pipe, **kw):
        import asyncio
        kw.setdefault("hook_node_id", "11")
        return json.loads(asyncio.run(tools(pipe)["halt_for_review"](**kw)))

    def test_it_arms_the_halt_and_reports_what_was_collected(self):
        pipe = self._pipe()
        out = self._call(pipe)
        self.assertEqual(out["status"], "halted")
        self.assertEqual(out["collected"], 2)
        self.assertEqual(out["not_run"], ["12"])
        self.assertIsNotNone(pipe._review_armed)
        self.assertEqual(pipe._review_armed.hook_node_id, "11")

    def test_it_pushes_the_collector_to_the_canvas(self):
        from src.utils.canvas_patch import clear, drain
        clear()
        self.addCleanup(clear)
        self._call(self._pipe())
        ops = [e for e in drain() if e.get("op") == "review_collector"]
        self.assertEqual(len(ops), 1)
        self.assertEqual(ops[0]["files"], ["C:/out/ref_1.png", "C:/out/ref_2.png"])
        self.assertEqual(ops[0]["hook_node_id"], "11")

    def test_it_defaults_to_what_this_turn_produced(self):
        out = self._call(self._pipe())
        self.assertEqual(out["files"], ["C:/out/ref_1.png", "C:/out/ref_2.png"])

    def test_explicit_outputs_win(self):
        out = self._call(self._pipe(), outputs=["C:/out/only.png"])
        self.assertEqual(out["files"], ["C:/out/only.png"])

    def test_halting_on_nothing_is_refused(self):
        """Run the stage first — there is no reviewing a run that hasn't happened."""
        pipe = self._pipe(_chain_output_paths=[])
        out = self._call(pipe)
        self.assertIn("nothing to review", out["error"])
        self.assertIsNone(pipe._review_armed)

    def test_it_refuses_a_hook_that_is_not_a_review_hook(self):
        out = self._call(self._pipe(), hook_node_id="10")
        self.assertIn("not a review hook", out["error"])
        self.assertIn("stop_hook_run", out["error"])

    def test_it_refuses_an_id_that_is_not_on_the_canvas(self):
        self.assertIn("no hook", self._call(self._pipe(), hook_node_id="999")["error"])

    def test_a_dry_run_walks_straight_past_it(self):
        """Its outputs are stand-ins; choosing between files that don't exist is
        no kind of review, and it would leave a halt armed over a run that never
        happened."""
        pipe = self._pipe(_dry_run=True)
        out = self._call(pipe)
        self.assertEqual(out["status"], "skipped")
        self.assertIsNone(pipe._review_armed)

    def test_duplicate_paths_collapse(self):
        out = self._call(self._pipe(), outputs=["a.png", "a.png", "b.png"])
        self.assertEqual(out["files"], ["a.png", "b.png"])


class RoundTripTest(unittest.TestCase):
    """The halt across the turn boundary: armed, restored, answered, spent."""

    @staticmethod
    def _pipe(**over):
        over.setdefault("_session", AgentSession(session_id="t"))
        return pipeline_stub(**over)

    def test_an_armed_halt_is_written_to_the_session(self):
        pipe = self._pipe(_review_armed=ReviewHalt(
            hook_node_id="11", collector_key="agentY_review_11", produced=("a.png",),
            question="which?", remaining=("12",)))
        Pipeline._arm_review_halt(pipe)
        self.assertEqual(pipe._session.review_halt["hook_node_id"], "11")
        self.assertEqual(pipe._session.review_halt["remaining"], ["12"])

    def test_it_comes_back_on_the_next_turn(self):
        pipe = self._pipe()
        pipe._session.review_halt = {"hook_node_id": "11", "collector_key": "k",
                                     "produced": ["a.png", "b.png"], "question": "which?",
                                     "remaining": ["12"]}
        halt = Pipeline._restore_review_halt(pipe)
        self.assertEqual(halt.hook_node_id, "11")
        self.assertEqual(halt.count(), 2)

    def test_an_answered_halt_is_spent(self):
        pipe = self._pipe(_review_halt=ReviewHalt(hook_node_id="11"),
                          _review_reply="continue")
        Pipeline._arm_review_halt(pipe)
        self.assertIsNone(pipe._session.review_halt)

    def test_an_unanswered_halt_survives_an_unrelated_message(self):
        """Asking something else mid-review is ordinary; losing the stop is not."""
        pipe = self._pipe(_review_halt=ReviewHalt(hook_node_id="11",
                                                  collector_key="agentY_review_11"),
                          _review_reply="")
        Pipeline._arm_review_halt(pipe)
        self.assertEqual(pipe._session.review_halt["hook_node_id"], "11")

    def test_a_stop_is_also_spent(self):
        pipe = self._pipe(_review_halt=ReviewHalt(hook_node_id="11"),
                          _review_reply="stop")
        Pipeline._arm_review_halt(pipe)
        self.assertIsNone(pipe._session.review_halt)

    def test_the_flag_survives_a_session_round_trip(self):
        s = AgentSession(session_id="t")
        s.review_halt = {"hook_node_id": "11", "collector_key": "k"}
        self.assertEqual(AgentSession(**s.model_dump()).review_halt["hook_node_id"], "11")

    def test_an_older_stored_session_still_loads(self):
        old = {"session_id": "t", "chat_summaries": [], "current_output_paths": []}
        self.assertIsNone(AgentSession(**old).review_halt)

    def test_a_junk_payload_is_ignored_rather_than_crashing_the_turn(self):
        for junk in ({}, {"hook_node_id": ""}, "nonsense", 7):
            with self.subTest(junk=junk):
                pipe = self._pipe()
                pipe._session.review_halt = junk
                self.assertIsNone(Pipeline._restore_review_halt(pipe))


def _halted(files="C:/out/a.png\nC:/out/c.png", node_id="77", base=None,
            reply="continue", wired=True):
    """A pipeline mid-halt, with the ballot wired into the review hook's anchor.

    Wired, because that is how the ballot is actually found: the collector is
    created in the BROWSER, so litegraph assigns its id there and the server never
    sees it. What the server does get, every turn, is each hook's anchors — id,
    type and widget values — which is the same channel a QA hook's references
    arrive on.
    """
    hooks = _chain()
    if wired:
        hooks[1]["anchors"] = [{"node_id": node_id, "type": "AgentYImageCollector",
                                "to_input": "anchors.anchor0",
                                "widgets": {"files": files}}]
    return pipeline_stub(
        _canvas_hooks=hooks,
        _review_halt=ReviewHalt(hook_node_id="11", collector_key="agentY_review_11",
                                produced=("a.png", "b.png", "c.png")),
        _review_reply=reply,
        _canvas_base_prompt=base if base is not None else {})


class TheAnswerIsOnTheCanvasTest(unittest.TestCase):
    """The point of the whole design: resume reads the node, not the memory."""

    def test_it_reads_the_collector_as_the_user_left_it(self):
        self.assertEqual(Pipeline._review_collector_files(_halted()),
                         ["C:/out/a.png", "C:/out/c.png"])

    def test_what_it_held_at_the_halt_is_NOT_the_answer(self):
        """Three were produced; the user kept two. Replaying three ignores them."""
        pipe = _halted()
        self.assertEqual(len(Pipeline._review_collector_files(pipe)), 2)
        self.assertEqual(pipe._review_halt.count(), 3, "the record still says three")

    def test_a_file_the_user_added_themselves_comes_through(self):
        pipe = _halted("C:/out/a.png\nD:/mine/hand_painted.png")
        self.assertIn("D:/mine/hand_painted.png", Pipeline._review_collector_files(pipe))

    def test_their_order_is_kept(self):
        pipe = _halted("C:/out/c.png\nC:/out/a.png")
        self.assertEqual(Pipeline._review_collector_files(pipe),
                         ["C:/out/c.png", "C:/out/a.png"])

    def test_it_falls_back_to_the_captured_graph(self):
        """An anchor can arrive without its widget values; the graph still has them."""
        pipe = _halted(files="", base={"77": {"class_type": "AgentYImageCollector",
                                              "inputs": {"files": "C:/out/b.png"}}})
        self.assertEqual(Pipeline._review_collector_files(pipe), ["C:/out/b.png"])

    def test_a_deleted_collector_reads_as_empty_rather_than_as_the_old_list(self):
        self.assertEqual(Pipeline._review_collector_files(_halted(wired=False)), [])

    def test_an_emptied_collector_reads_as_empty_too(self):
        self.assertEqual(Pipeline._review_collector_files(_halted(files="")), [])

    def test_blank_lines_and_quotes_are_tidied(self):
        pipe = _halted('\n"C:/out/a.png"\n\n  C:/out/b.png  \n')
        self.assertEqual(Pipeline._review_collector_files(pipe),
                         ["C:/out/a.png", "C:/out/b.png"])

    def test_the_answer_follows_the_WIRE_not_a_remembered_id(self):
        """Wire a different collector in and that is the one that is read.

        Which is what someone rearranging their graph mid-review would expect,
        and is not something a remembered node id could ever notice.
        """
        pipe = _halted(node_id="99", files="C:/out/mine.png")
        self.assertEqual(Pipeline._review_collector_files(pipe), ["C:/out/mine.png"])
        self.assertEqual((Pipeline._review_collector(pipe) or {})["node_id"], "99")


class TheIdSeamTest(unittest.TestCase):
    """halt_for_review and the resume path have to agree on what the ballot IS.

    They did not. The halt stored a synthetic key (`agentY_review_11`) because the
    node does not exist yet when it is armed; resume looked a node up BY THAT KEY
    and of course never found one, so every continue read an empty collector and
    reported that the user had deleted it. Both halves were tested, separately,
    and both passed — the bug lived in the seam between them.
    """

    def test_the_halt_does_not_pretend_to_know_a_node_id(self):
        import asyncio
        pipe = pipeline_stub(_canvas_hooks=_chain(),
                             _chain_output_paths=["C:/out/a.png"])
        asyncio.run(tools(pipe)["halt_for_review"](hook_node_id="11"))
        self.assertFalse(hasattr(pipe._review_armed, "collector_node_id"),
                         "there is no id to know — litegraph assigns it in the browser")
        self.assertEqual(pipe._review_armed.collector_key, "agentY_review_11")

    def test_the_key_is_what_the_frontend_is_given_to_reuse_the_node_by(self):
        import asyncio
        from src.utils.canvas_patch import clear, drain
        clear()
        self.addCleanup(clear)
        pipe = pipeline_stub(_canvas_hooks=_chain(),
                             _chain_output_paths=["C:/out/a.png"])
        asyncio.run(tools(pipe)["halt_for_review"](hook_node_id="11"))
        op = next(e for e in drain() if e.get("op") == "review_collector")
        self.assertEqual(op["collector_key"], pipe._review_armed.collector_key)

    def test_armed_here_resolves_there(self):
        """The end-to-end seam: arm a halt, restore it, resolve the ballot."""
        import asyncio
        pipe = pipeline_stub(_canvas_hooks=_chain(), _session=AgentSession(session_id="t"),
                             _chain_output_paths=["C:/out/a.png", "C:/out/b.png"])
        asyncio.run(tools(pipe)["halt_for_review"](hook_node_id="11"))
        Pipeline._arm_review_halt(pipe)

        # Next turn: the frontend has created the node and reports it as an anchor.
        nxt = pipeline_stub(_session=pipe._session, _review_reply="continue")
        nxt._canvas_hooks = _chain()
        nxt._canvas_hooks[1]["anchors"] = [
            {"node_id": "412", "type": "AgentYImageCollector",
             "widgets": {"files": "C:/out/b.png"}}]
        nxt._review_halt = Pipeline._restore_review_halt(nxt)
        self.assertIsNotNone(nxt._review_halt)
        self.assertEqual(Pipeline._review_collector_files(nxt), ["C:/out/b.png"],
                         "the user kept one of the two")


class WritingToTheBallotTest(unittest.TestCase):
    """Replacing a reference, not just dropping one.

    "Regenerate the third one and carry on" is the natural request during a
    review, and it dead-ends if the agent can only write to nodes the user has
    SELECTED — they would have to go and click the node first. So the collector a
    live halt is waiting on is writable without a selection. Nothing else is:
    an edit to someone's graph should be one they pointed at.
    """

    def setUp(self):
        # Pin the mode. Without this these read the MACHINE's settings and pass or
        # fail depending on whether whoever ran them has canvas_full_graph on —
        # which is exactly the kind of test that goes green on one desk and red on
        # another. Selection-only is the default, and the mode the ballot
        # exemption has to work in (see the both-modes test below).
        self.enterContext(mock.patch("src.utils.canvas_view.full_graph_visible",
                                     return_value=False))

    def _call(self, pipe, node_id, files):
        import asyncio
        return json.loads(asyncio.run(
            tools(pipe)["set_canvas_node_params"](node_id=node_id,
                                                  params={"files": files})))

    def test_the_halted_ballot_takes_a_write_with_nothing_selected(self):
        pipe = _halted()
        out = self._call(pipe, "77", "C:/out/a.png\nC:/out/new.png")
        self.assertEqual(out["status"], "applied")
        self.assertEqual(pipe._canvas_selection, [], "nothing was selected")

    def test_the_write_reaches_the_canvas(self):
        from src.utils.canvas_patch import clear, drain
        clear()
        self.addCleanup(clear)
        self._call(_halted(), "77", "C:/out/new.png")
        ops = [e for e in drain() if str(e.get("node_id")) == "77"]
        self.assertEqual(ops[0]["params"], {"files": "C:/out/new.png"})

    def test_the_exemption_is_the_ballot_and_nothing_else(self):
        """At the default setting (selection-only), everything else is refused."""
        out = self._call(_halted(), "50", "whatever")
        self.assertIn("not in the current canvas selection", out["error"])

    def test_the_ballot_exemption_does_not_outlive_the_halt(self):
        """It matters because the ballot is created in the BROWSER mid-turn, so it
        can be absent from the graph captured at the start of the turn — which is
        why it is its own exemption rather than something canvas_full_graph covers."""
        pipe = pipeline_stub(_canvas_hooks=_chain(), _review_halt=None)
        out = self._call(pipe, "77", "C:/out/new.png")
        self.assertIn("not in the current canvas selection", out["error"])

    def test_the_ballot_is_writable_in_BOTH_modes(self):
        """The exemption is not about visibility — it is about the node existing.

        The ballot is created in the browser mid-turn, so it can be missing from
        the graph captured at the start of the turn whether or not the agent is
        allowed to see that whole graph.
        """
        for full in (False, True):
            with self.subTest(canvas_full_graph=full):
                with mock.patch("src.utils.canvas_view.full_graph_visible",
                                return_value=full):
                    out = self._call(_halted(), "77", "C:/out/a.png")
                self.assertEqual(out["status"], "applied")

    def test_the_agent_is_told_the_node_id_it_may_write_to(self):
        from src.utils.review_gate import halt_state
        block = halt_state(ReviewHalt(hook_node_id="11", produced=("a.png",)), "77")
        self.assertIn("collector node 77", block)

    def test_a_ballot_that_is_gone_is_named_as_gone(self):
        from src.utils.review_gate import halt_state
        block = halt_state(ReviewHalt(hook_node_id="11", produced=("a.png",)), "")
        self.assertIn("no longer wired", block)


class RenumberTest(unittest.TestCase):
    """Dropping a row moves every row after it up a slot.

    The wiring follows by itself — expand_image_batches wires only as many slots
    as there are files. The @imageN table in the prompt does NOT: it is prose, and
    it is still whatever was written before the edit. Get it wrong and the video
    comes back with the ape doing the mentor's beat, with no error anywhere.

    So the bindings are shown as the slots they WILL be, with each file's own
    role beside it, at the moment they have to be rewritten.
    """

    def test_the_list_is_rendered_as_the_slots_it_becomes(self):
        from src.utils.review_gate import binding_table
        out = binding_table(["C:/out/ref_42.png", "C:/out/ref_44.png"])
        self.assertIn("@image1 / image_1 = ref_42.png", out)
        self.assertIn("@image2 / image_2 = ref_44.png", out)

    def test_the_role_rides_along_so_the_shift_is_visible(self):
        from src.utils.review_gate import binding_table
        out = binding_table(["a.png", "c.png"],
                            {"a.png": "TANIHO (HERO)", "c.png": "APE"})
        self.assertIn("@image1 / image_1 = a.png — TANIHO (HERO)", out)
        self.assertIn("@image2 / image_2 = c.png — APE",
                      out, "APE was @image3 before the cut — that is the whole point")

    def test_a_file_with_no_role_is_left_honestly_bare(self):
        from src.utils.review_gate import binding_table
        self.assertEqual(binding_table(["mine.png"]), "    @image1 / image_1 = mine.png")

    def test_windows_separators_are_read_too(self):
        from src.utils.review_gate import binding_table
        self.assertIn("= ref_42.png", binding_table([r"C:\out\ref_42.png"]))

    def test_nothing_renders_as_nothing(self):
        from src.utils.review_gate import binding_table
        self.assertEqual(binding_table([]), "")

    def test_the_rule_says_what_the_number_now_means(self):
        from src.utils.review_gate import renumber_note
        note = renumber_note()
        self.assertIn("REWRITE", note)
        self.assertIn("as it stands NOW", note)
        self.assertIn("reports no error", note)

    def test_a_partial_continue_is_told_to_renumber(self):
        from src.utils.review_gate import resumed_note
        self.assertIn("RENUMBER", resumed_note(kept=2, dropped=1))

    def test_an_untouched_list_is_not_nagged_about_it(self):
        """Nothing moved, so there is nothing to rewrite — say less."""
        from src.utils.review_gate import resumed_note
        self.assertNotIn("RENUMBER", resumed_note(kept=3, dropped=0))

    def test_roles_come_from_the_sidecars_the_files_already_carry(self):
        with mock.patch("src.utils.output_tags.role_of_file",
                        side_effect=lambda p: "APE" if p.endswith("c.png") else ""):
            got = Pipeline._output_roles(pipeline_stub(), ["a.png", "c.png"])
        self.assertEqual(got, {"c.png": "APE"})

    def test_an_unreadable_sidecar_costs_nothing(self):
        with mock.patch("src.utils.output_tags.role_of_file", side_effect=OSError("nope")):
            self.assertEqual(Pipeline._output_roles(pipeline_stub(), ["a.png"]), {})

    def test_the_partial_spells_the_renumbering_out(self):
        from src.pipeline import _ORCH_PARTIALS_DIR
        text = (_ORCH_PARTIALS_DIR / "review_halt.md").read_text(encoding="utf-8")
        self.assertIn("Renumber the reference table", text)
        self.assertIn("@image2` means **the second line as it stands now**", text)


class PromptTest(unittest.TestCase):

    def test_the_partial_exists_and_says_the_precedence_rule(self):
        from src.pipeline import _ORCH_PARTIALS_DIR
        text = (_ORCH_PARTIALS_DIR / "review_halt.md").read_text(encoding="utf-8")
        self.assertIn("halt_for_review", text)
        self.assertIn("Their words beat the node", text)
        self.assertIn("stop_hook_run", text, "the two must not be confused")

    def test_the_halt_block_tells_it_to_re_read_the_node(self):
        from src.utils.review_gate import halt_state
        block = halt_state(ReviewHalt(hook_node_id="11", collector_key="k",
                                      produced=("a.png",), remaining=("12",)), "77")
        self.assertIn("read the node as it stands NOW", block)
        self.assertIn("77", block)
        self.assertIn("12", block)


if __name__ == "__main__":
    unittest.main()
