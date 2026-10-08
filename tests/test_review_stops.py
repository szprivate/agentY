"""Review stops on a pipeline whose stages run through real nodes.

The canvas these are modelled on: a screenplay, reviewed in a loop; then two
branches, each writing prompts into an image node whose save node feeds a review
hook that sits in a loop of its own.

    start 10 -> hook 4 -> review 12 -> break 11 -+-> hook 18 -> start 2 -> hook 8
                                                 |      -> [7 image] -> [30 save] -> review 9 -> break 3
                                                 +-> hook 29 -> start 24 -> hook 27
                                                        -> [26 image] -> [31 save] -> review 28 -> break 25
"""
import unittest

from src.utils import canvas_hooks as ch
from src.utils import hook_flow as hf
from src.utils import review_gate as rg


def hook(hid, purpose, prev=(), anchors=(), targets=(), **more):
    return {"hook_node_id": hid, "purpose": purpose, "prev_hook_ids": list(prev),
            "prev_hook_id": (prev[0] if prev else None),
            "anchors": [{"node_id": a, "type": t} for a, t in anchors],
            "targets": [{"node_id": n, "to_input": i, "type": "OpenAIGPTImageNodeV2"}
                        for n, i in targets], **more}


def canvas():
    hooks = [
        hook("10", "loop_start"),
        hook("4", "text", prev=["10"], anchors=[("10", "AgentYLoopStart")], directive="screenplay"),
        hook("12", "human_review", prev=["4"], anchors=[("4", "AgentYHook")]),
        hook("11", "loop_break", prev=["12"], anchors=[("12", "AgentYHook")],
             condition="until the story was approved", max_rounds=10),
        hook("18", "text", prev=["11"], anchors=[("11", "AgentYLoopBreak")], directive="characters"),
        hook("2", "loop_start", prev=["18"], anchors=[("18", "AgentYHook")]),
        hook("8", "inline_parameter", prev=["2"], anchors=[("2", "AgentYLoopStart")],
             targets=[("7", "prompt")], directive="3 prompts per character"),
        hook("9", "human_review", anchors=[("30", "bEpicSendToViewer")]),
        hook("3", "loop_break", prev=["9"], anchors=[("9", "AgentYHook")],
             condition="one image per character approved", max_rounds=5),
        hook("29", "text", prev=["11"], anchors=[("11", "AgentYLoopBreak")], directive="places"),
        hook("24", "loop_start", prev=["29"], anchors=[("29", "AgentYHook")]),
        hook("27", "inline_parameter", prev=["24"], anchors=[("24", "AgentYLoopStart")],
             targets=[("26", "prompt")], directive="3 prompts per place"),
        hook("28", "human_review", anchors=[("31", "bEpicSendToViewer")]),
        hook("25", "loop_break", prev=["28"], anchors=[("28", "AgentYHook")],
             condition="one image per place approved", max_rounds=5),
    ]
    graph = {
        "7": {"class_type": "OpenAIGPTImageNodeV2", "inputs": {"prompt": ["8", 0]}},
        "30": {"class_type": "bEpicSendToViewer", "inputs": {"input": ["7", 0]}},
        "26": {"class_type": "OpenAIGPTImageNodeV2", "inputs": {"prompt": ""}},
        "31": {"class_type": "bEpicSendToViewer", "inputs": {"input": ["26", 0]}},
    }
    return hooks, graph


def planned():
    hooks, graph = canvas()
    return hf.plan(ch.link_through_nodes(hooks, graph))


class ThroughRealNodes(unittest.TestCase):
    def test_a_review_follows_the_stage_whose_save_node_it_reads(self):
        hooks, graph = canvas()
        linked = {h["hook_node_id"]: h for h in ch.link_through_nodes(hooks, graph)}
        self.assertEqual(linked["9"].get("via_hook_ids"), ["8"])     # hook node in the graph
        self.assertEqual(linked["28"].get("via_hook_ids"), ["27"])   # spliced out: by target
        self.assertFalse(linked["8"].get("via_hook_ids"))

    def test_no_graph_changes_nothing(self):
        hooks, _ = canvas()
        self.assertTrue(all("via_hook_ids" not in h for h in ch.link_through_nodes(hooks, None)))

    def test_the_loop_takes_the_stage_and_finds_its_start(self):
        flow = planned()
        loops = {lp.break_id: lp for lp in flow.loops}
        self.assertEqual(loops["3"].members, ["8", "9"])
        self.assertEqual(loops["3"].start_id, "2")
        self.assertEqual(loops["25"].members, ["27", "28"])
        self.assertEqual(loops["11"].members, ["4", "12"])
        self.assertEqual(flow.problems, [])

    def test_the_value_a_hook_reads_is_still_only_the_direct_one(self):
        by_id = {h["hook_node_id"]: h for h in planned().hooks}
        self.assertEqual(by_id["9"]["prev_hook_ids"], [])
        self.assertIsNone(by_id["9"]["prev_hook_id"])
        self.assertEqual(by_id["9"]["via_hook_ids"], ["8"])
        self.assertEqual(by_id["18"]["prev_hook_ids"], ["12"])


class WhichStopComesFirst(unittest.TestCase):
    def setUp(self):
        self.hooks = planned().hooks
        self.waits = ch.reviews_before(self.hooks)

    def test_every_stage_knows_the_reviews_in_front_of_it(self):
        self.assertNotIn("4", self.waits)
        self.assertEqual(self.waits["18"], ["12"])
        self.assertEqual(self.waits["8"], ["12"])
        self.assertEqual(self.waits["9"], ["12"])
        self.assertEqual(self.waits["27"], ["12"])

    def test_a_branch_does_not_wait_on_the_other_branchs_review(self):
        self.assertNotIn("9", self.waits["27"])
        self.assertNotIn("28", self.waits["8"])

    def test_a_review_stops_on_the_stage_before_it(self):
        self.assertEqual(ch.stage_before_review(self.hooks, "12"), ["4"])
        self.assertEqual(ch.stage_before_review(self.hooks, "9"), ["8"])

    def test_the_block_names_the_unanswered_stop_and_what_waits(self):
        block = ch.describe_hooks(self.hooks, {}, flow=planned())
        self.assertIn("ENFORCED", block)
        self.assertIn("review hook 12 stops on the work of hook(s) 4", block)
        self.assertIn("NOT this turn — hook(s) 18, 27, 29, 8 ", block)

    def test_an_answered_review_opens_only_what_stood_behind_it(self):
        block = ch.describe_hooks(self.hooks, {}, flow=planned(), passed_reviews=["12"])
        self.assertIn("review hook 12 → ALREADY ANSWERED", block)
        self.assertNotIn("NOT this turn", block)      # 8 and 27 are open; nothing is behind 9/28


class LoopsThePersonJudges(unittest.TestCase):
    def test_a_loop_with_a_review_in_it_is_the_users_to_end(self):
        flow = planned()
        self.assertEqual({lp.break_id: lp.review_id for lp in flow.loops},
                         {"3": "9", "11": "12", "25": "28"})
        text = "\n".join(hf.loop_lines(flow))
        self.assertIn("JUDGED BY THE USER at review hook 12", text)
        self.assertIn("halt_for_review(12)", text)

    def test_a_loop_without_one_is_still_judged_by_loop_check(self):
        hooks = [hook("1", "loop_start"), hook("2", "make_workflow", prev=["1"], directive="x"),
                 hook("3", "loop_break", prev=["2"], condition="sharp")]
        flow = hf.plan(hooks)
        self.assertEqual(flow.loops[0].review_id, "")
        self.assertNotIn("JUDGED BY THE USER", "\n".join(hf.loop_lines(flow)))


class TheRefusal(unittest.TestCase):
    def test_it_says_where_to_stop_and_that_nothing_failed(self):
        out = rg.ahead_refusal(["18", "8"], "12", ["4"])
        self.assertIn("review hook 12", out["error"])
        self.assertIn('halt_for_review("12")', out["what_to_do"])
        self.assertIn("hook(s) 4", out["what_to_do"])
        self.assertIn("nothing failed", out["do_not"])

    def test_a_text_halt_has_no_collector(self):
        halt = rg.ReviewHalt(hook_node_id="12", text_hooks=("4",), remaining=("18", "29"))
        self.assertTrue(halt.is_text())
        state = rg.halt_state(halt)
        self.assertIn("no collector", state)
        self.assertIn("place_canvas_text", state)
        self.assertIn("text is waiting", halt.describe())

    def test_a_file_halt_is_not_a_text_halt(self):
        halt = rg.ReviewHalt(hook_node_id="9", produced=("a.png",), text_hooks=())
        self.assertFalse(halt.is_text())
        self.assertIn("1 output", halt.describe())



class AnAnswerSaidTwice(unittest.TestCase):
    """"Approved - proceed" did not lift the stop: only one-phrase answers did."""

    def test_two_plain_yeses_are_a_continue(self):
        for text in ("Approved - proceed", "yes, go ahead", "ok. continue please",
                     "looks good, proceed", "perfect - go", "yes please",
                     "Approved - proceed\n\ncontinue"):
            with self.subTest(text=text):
                self.assertEqual(rg.read_reply(text), "continue")

    def test_two_plain_noes_are_a_stop(self):
        self.assertEqual(rg.read_reply("no - stop"), "stop")

    def test_a_yes_with_an_instruction_is_still_an_instruction(self):
        for text in ("approved, but make the third one warmer",
                     "approved and then make it blue", "yes and no", "no, make it shorter"):
            with self.subTest(text=text):
                self.assertEqual(rg.read_reply(text), "")

    def test_a_loose_word_alone_is_not_an_answer(self):
        for text in ("good", "great", "make it shorter"):
            with self.subTest(text=text):
                self.assertEqual(rg.read_reply(text), "")

if __name__ == "__main__":
    unittest.main()
