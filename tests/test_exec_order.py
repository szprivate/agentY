"""Order is the execution wire.

The panel reports, for each stage, the stage(s) it runs after (``via_hook_ids``,
read off the ``exec`` wire) apart from the stage(s) whose value it reads
(``prev_hook_ids``, the ``anchor`` wires). These are the same pipeline as
tests/test_review_stops.py, drawn the new way:

    start 10 -> hook 4 -> review 12 -> break 11 -+-> hook 18 -> start 2 -> hook 8 -> review 9 -> break 3
                                                 +-> hook 29 -> start 24 -> hook 27 -> qa 28 -> break 25

Hook 8 writes prompts into image node 7, whose save node 30 is wired into review
9's anchor: data. That 9 comes after 8 is said by the wire, not guessed from it.
"""
import unittest

from src.utils import canvas_hooks as ch
from src.utils import hook_flow as hf


def stage(hid, purpose, after=(), reads=(), anchors=(), targets=(), **more):
    return {"hook_node_id": hid, "purpose": purpose,
            "exec_prev_ids": list(after), "via_hook_ids": list(after), "exec_wired": True,
            "prev_hook_ids": list(reads), "prev_hook_id": (reads[0] if reads else None),
            "anchors": [{"node_id": a, "type": t} for a, t in anchors],
            "targets": [{"node_id": n, "to_input": i, "type": "OpenAIGPTImageNodeV2"}
                        for n, i in targets], **more}


def canvas():
    return [
        stage("10", "loop_start"),
        stage("4", "text_only", after=["10"], directive="screenplay"),
        stage("12", "human_review", after=["4"], reads=["4"]),
        stage("11", "loop_break", after=["12"], condition="until the story was approved"),
        stage("18", "text_only", after=["11"], reads=["4"], directive="characters"),
        stage("2", "loop_start", after=["18"]),
        stage("8", "set_parameter", after=["2"], reads=["18"], targets=[("7", "prompt")],
              directive="3 prompts per character"),
        stage("9", "human_review", after=["8"], anchors=[("30", "bEpicSendToViewer")]),
        stage("3", "loop_break", after=["9"], condition="one image per character approved"),
        stage("29", "text_only", after=["11"], reads=["4"], directive="places"),
        stage("24", "loop_start", after=["29"]),
        stage("27", "set_parameter", after=["24"], reads=["29"], targets=[("26", "prompt")],
              directive="3 prompts per place"),
        stage("28", "qa", after=["27"], directive="the flat reads as one cramped space",
              applies_to=["27"]),
        stage("25", "loop_break", after=["28"], max_rounds=4),
    ]


class TheWireIsTheOrder(unittest.TestCase):
    def setUp(self):
        self.flow = hf.plan(canvas())
        self.by_id = {h["hook_node_id"]: h for h in self.flow.hooks}

    def test_loop_nodes_come_out_and_the_stages_join_up_across_them(self):
        self.assertNotIn("11", self.by_id)
        self.assertEqual(self.by_id["18"]["via_hook_ids"], ["12"])     # through break 11
        self.assertEqual(self.by_id["8"]["via_hook_ids"], ["18"])      # through start 2
        self.assertEqual(self.by_id["4"]["via_hook_ids"], [])          # start 10 leads nowhere back

    def test_what_a_stage_reads_is_kept_apart_from_when_it_runs(self):
        self.assertEqual(self.by_id["18"]["prev_hook_ids"], ["4"])
        self.assertEqual(self.by_id["9"]["prev_hook_ids"], [])
        self.assertEqual(self.by_id["9"]["via_hook_ids"], ["8"])

    def test_loops_are_what_lies_between_a_start_and_a_break_on_the_wire(self):
        loops = {lp.break_id: lp for lp in self.flow.loops}
        self.assertEqual(loops["11"].members, ["4", "12"])
        self.assertEqual(loops["3"].members, ["8", "9"])
        self.assertEqual((loops["11"].start_id, loops["3"].start_id, loops["25"].start_id),
                         ("10", "2", "24"))
        self.assertEqual(self.flow.problems, [])

    def test_a_human_review_in_a_loop_judges_it(self):
        loops = {lp.break_id: lp for lp in self.flow.loops}
        self.assertEqual(loops["3"].review_id, "9")
        self.assertEqual(loops["3"].qa_id, "")

    def test_an_agent_review_in_a_loop_is_the_judging_not_a_stage(self):
        loop = next(lp for lp in self.flow.loops if lp.break_id == "25")
        self.assertEqual(loop.members, ["27"])
        self.assertEqual(loop.qa_id, "28")
        self.assertEqual(loop.review_id, "")
        text = "\n".join(hf.loop_lines(self.flow))
        self.assertIn("The agent review (node 28) in this loop is part of the judging", text)

    def test_the_stop_gates_what_runs_after_it_on_the_wire(self):
        waits = ch.reviews_before(self.flow.hooks)
        self.assertEqual(waits["18"], ["12"])
        self.assertEqual(waits["8"], ["12"])
        self.assertEqual(waits["27"], ["12"])
        self.assertNotIn("9", waits["27"])                 # the other branch's review
        self.assertEqual(ch.stage_before_review(self.flow.hooks, "9"), ["8"])

    def test_a_split_wire_is_two_branches(self):
        groups = [sorted(g) for g in hf.branches([h for h in self.flow.hooks
                                                  if h["hook_node_id"] in ("8", "9", "27")])]
        self.assertIn(["8", "9"], groups)
        self.assertIn(["27"], groups)


class WhenTheWireAndTheDataDisagree(unittest.TestCase):
    def test_reading_a_value_that_is_produced_later_is_reported(self):
        hooks = [stage("1", "set_parameter", reads=["2"], directive="use the caption"),
                 stage("2", "text_only", after=["1"], directive="write a caption")]
        problems = hf.plan(hooks + [stage("9", "loop_start")]).problems
        self.assertTrue(any("hook 1 reads the value of hook 2" in p and "runs 1 first" in p
                            for p in problems), problems)

    def test_reading_a_value_produced_earlier_is_fine(self):
        hooks = [stage("2", "text_only", directive="write a caption"),
                 stage("1", "set_parameter", after=["2"], reads=["2"], directive="use it"),
                 stage("9", "loop_start")]
        self.assertFalse([p for p in hf.plan(hooks).problems if "reads the value" in p])


class NoGuessingBesideAWire(unittest.TestCase):
    GRAPH = {"7": {"class_type": "X", "inputs": {"prompt": ""}},
             "30": {"class_type": "Save", "inputs": {"input": ["7", 0]}}}

    def test_a_wired_canvas_is_not_second_guessed_from_its_data(self):
        hooks = [stage("8", "set_parameter", targets=[("7", "prompt")], directive="p"),
                 stage("9", "human_review", anchors=[("30", "Save")])]      # NOT wired after 8
        out = {h["hook_node_id"]: h for h in ch.link_through_nodes(hooks, self.GRAPH)}
        self.assertEqual(out["9"]["via_hook_ids"], [])

    def test_a_canvas_with_no_wire_at_all_still_gets_the_guess(self):
        hooks = [{**stage("8", "set_parameter", targets=[("7", "prompt")], directive="p"),
                  "exec_wired": False},
                 {**stage("9", "human_review", anchors=[("30", "Save")]), "exec_wired": False}]
        out = {h["hook_node_id"]: h for h in ch.link_through_nodes(hooks, self.GRAPH)}
        self.assertEqual(out["9"]["via_hook_ids"], ["8"])


class TheWireIsNotData(unittest.TestCase):
    def graph(self):
        return {
            "8": {"class_type": "AgentYHook", "inputs": {"exec": ["2", 0], "directive": "p"}},
            "2": {"class_type": "AgentYLoopStart", "inputs": {}},
            "9": {"class_type": "AgentYReview", "inputs": {"exec": ["8", 1], "anchors.anchor0": ["30", 0]}},
            "3": {"class_type": "AgentYLoopBreak", "inputs": {"exec": ["9", 1]}},
            "7": {"class_type": "X", "inputs": {"prompt": ["8", 0]}},
            "30": {"class_type": "Save", "inputs": {"input": ["7", 0]}},
            "40": {"class_type": "Other", "inputs": {"exec": ["8", 1]}},
        }

    def test_it_is_taken_out_of_every_stage_node(self):
        g = ch.strip_exec_links(self.graph())
        for nid in ("8", "9", "3"):
            self.assertNotIn("exec", g[nid]["inputs"])
        self.assertEqual(g["9"]["inputs"]["anchors.anchor0"], ["30", 0])   # data stays

    def test_another_nodes_input_of_that_name_is_left_alone(self):
        self.assertIn("exec", ch.strip_exec_links(self.graph())["40"]["inputs"])

    def test_no_graph_is_not_an_error(self):
        self.assertIsNone(ch.strip_exec_links(None))

    def test_a_review_node_comes_out_of_the_graph_like_a_hook(self):
        g = ch.strip_exec_links(self.graph())
        g["50"] = {"class_type": "Next", "inputs": {"image": ["9", 0]}}
        clean, removed = ch.splice_hook_nodes(g)
        self.assertIn("9", removed)
        self.assertNotIn("9", clean)
        self.assertEqual(clean["50"]["inputs"]["image"], ["30", 0])   # its anchor passes through


class ThePurposesByTheirNewNames(unittest.TestCase):
    def test_text_only_is_a_text_hook(self):
        self.assertTrue(ch._is_text({"purpose": "text_only"}))
        self.assertTrue(ch._is_text({"purpose": "text"}))           # an older graph

    def test_set_parameter_is_the_producer(self):
        hook = {"purpose": "set_parameter"}
        self.assertFalse(ch._is_text(hook) or ch._is_standin(hook) or ch._is_review(hook)
                         or ch._is_qa(hook))

    def test_a_review_nodes_default_title_is_not_a_question(self):
        self.assertEqual(ch.hook_title({"title": "agentY review"}), "")
        self.assertEqual(ch.hook_title({"title": "pick two for the video"}), "pick two for the video")

    def test_the_notes_box_is_the_question_ahead_of_the_title(self):
        hooks = [{"hook_node_id": "4", "purpose": "text_only", "directive": "write it",
                  "exec_wired": True, "via_hook_ids": []},
                 {"hook_node_id": "12", "purpose": "human_review", "title": "Story check",
                  "directive": "is the ending right?", "via_hook_ids": ["4"], "exec_wired": True}]
        block = ch.describe_hooks(hooks, {})
        self.assertIn('put to the user: "is the ending right?"', block)


class BranchesOffTheWire(unittest.TestCase):
    """A wire that splits is branches; each is worked in a conversation of its own."""

    def setUp(self):
        self.flow = hf.plan(canvas())
        self.par = hf.parallel(self.flow)

    def test_the_trunk_is_what_runs_before_the_split(self):
        self.assertEqual(self.par.trunk, ["4", "12"])

    def test_each_arm_is_a_branch_with_everything_after_it(self):
        self.assertEqual([b.members for b in self.par.branches],
                         [["18", "8", "9"], ["29", "27", "28"]])
        self.assertEqual({b.after for b in self.par.branches}, {"12"})

    def test_a_branch_is_handed_its_loop_nodes_with_its_stages(self):
        # start_shot(hook_ids=...) scopes the canvas to these: without the loop
        # nodes the branch would not know its own stages repeat.
        self.assertEqual(self.par.branches[0].scope, ["18", "8", "9", "2", "3"])
        self.assertEqual(self.par.branches[1].scope, ["29", "27", "28", "24", "25"])

    def test_it_knows_which_branches_a_person_reviews(self):
        self.assertEqual([b.reviews for b in self.par.branches], [["9"], []])

    def test_a_single_chain_is_not_parallel(self):
        chain = [stage("1", "text_only", directive="a"), stage("2", "text_only", after=["1"], directive="b")]
        self.assertFalse(hf.parallel(hf.plan(chain)))
        self.assertEqual(hf.parallel_lines(hf.plan(chain)), [])

    def test_the_lead_is_told_to_review_here_and_answer_into_the_same_conversations(self):
        text = "\n".join(hf.parallel_lines(self.flow))
        self.assertIn("the execution wire splits into 2 branches", text)
        self.assertIn("the user reviews HERE", text)
        self.assertIn("hook_ids=['18', '8', '9', '2', '3']", text)
        self.assertIn("halt_for_review(<id>, outputs=[those files])", text)
        self.assertIn("several stops stand at once", text)
        self.assertIn("message_shot(name, …), never a new start_shot", text)
        self.assertIn("- trunk (yours, first): hook 4 \"screenplay\"; hook 12", text)

    def test_a_pipeline_with_no_review_in_its_branches_is_not_told_about_stops(self):
        hooks = [stage("1", "text_only", directive="t"),
                 stage("2", "make_workflow", after=["1"], directive="a"),
                 stage("3", "make_workflow", after=["1"], directive="b")]
        text = "\n".join(hf.parallel_lines(hf.plan(hooks + [stage("9", "loop_start")])))
        self.assertIn("splits into 2 branches", text)
        self.assertNotIn("halt_for_review", text)


class WhereBranchesMeet(unittest.TestCase):
    def setUp(self):
        self.hooks = canvas() + [stage("70", "join", after=["3", "25"]),
                                 stage("71", "make_workflow", after=["70"], directive="animate")]
        self.flow = hf.plan(self.hooks)
        self.by_id = {h["hook_node_id"]: h for h in self.flow.hooks}

    def test_the_stage_after_a_join_runs_after_every_wire_into_it(self):
        self.assertNotIn("70", self.by_id)                       # the join does no work
        self.assertEqual(self.by_id["71"]["via_hook_ids"], ["9", "28"])

    def test_it_belongs_to_no_branch_and_waits_for_all(self):
        par = hf.parallel(self.flow)
        self.assertEqual(par.joined, ["71"])
        self.assertEqual([b.members for b in par.branches], [["18", "8", "9"], ["29", "27", "28"]])
        self.assertIn("wait for ALL of them", "\n".join(hf.parallel_lines(self.flow)))

    def test_it_stays_shut_until_every_branchs_review_is_answered(self):
        waits = ch.reviews_before(self.flow.hooks)
        self.assertEqual(sorted(waits["71"]), ["12", "9"])

    def test_a_join_with_one_wire_is_reported(self):
        hooks = [stage("1", "text_only", directive="a"), stage("70", "join", after=["1"]),
                 stage("2", "text_only", after=["70"], directive="b")]
        self.assertTrue(any("join 70 has fewer than two" in p_ for p_ in hf.plan(hooks).problems))


if __name__ == "__main__":
    unittest.main()
