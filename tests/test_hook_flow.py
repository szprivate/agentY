"""Loops and parallel branches of a hook pipeline, built on the canvas.

Two nodes mark a loop - loop start and loop break - and the hooks wired between
them repeat until the break's condition is met, judged by the QA agent, or its
rounds run out. Hooks that share no wire are separate branches and are worked on
at the same time. A hook run's workflows go into the open graph, a large one
folded into a subgraph.
"""

import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from agenty_core.utils import turn_scope

from pipeline_stub import pipeline_stub, tools
from src.utils import hook_flow as hf
from src.utils import loop_judge as lj
from src.utils.canvas_hooks import describe_hooks, splice_hook_nodes
from test_canvas_edit import SCHEMAS, _graph
from test_insert_workflow import BUILT


def _hook(hid, purpose="make_workflow", prev=(), directive="", **kw):
    prev = [str(p) for p in prev]
    return {"hook_node_id": str(hid), "purpose": purpose, "directive": directive or f"stage {hid}",
            "title": "", "prev_hook_ids": prev, "prev_hook_id": prev[0] if prev else None,
            "prev_links": [{"from_hook_id": p, "from_output_slot": 0, "to_input": "anchors.anchor0"}
                           for p in prev],
            "anchors": [], "targets": [], **kw}


LOAD = {"node_id": "3", "type": "LoadImage", "title": "", "widgets": {"image": "ref.png"},
        "from_output_slot": 0, "from_output_type": "IMAGE", "to_input": "anchors.anchor0"}


def _pipeline():
    """load -> start(10) -> make(11) -> refine(12) -> break(13) -> animate(14)."""
    return [
        _hook(10, hf.LOOP_START, anchors=[LOAD]),
        _hook(11, prev=[10], directive="put the dancer on the stage"),
        _hook(12, prev=[11], directive="fix the hands"),
        _hook(13, hf.LOOP_BREAK, prev=[12], directive="", condition="the pose matches the reference",
              max_rounds=4, forward="best", title="agentY loop break"),
        _hook(14, prev=[13], directive="animate it"),
    ]


class ReadingTheCanvas(unittest.TestCase):

    def setUp(self):
        self.flow = hf.plan(_pipeline())

    def test_the_hooks_between_start_and_break_are_the_body(self):
        (loop,) = self.flow.loops
        self.assertEqual(loop.members, ["11", "12"])
        self.assertEqual((loop.start_id, loop.break_id), ("10", "13"))
        self.assertEqual((loop.condition, loop.max_rounds, loop.forward),
                         ("the pose matches the reference", 4, "best"))
        self.assertEqual(self.flow.problems, [])

    def test_the_flow_nodes_are_gone_and_the_chain_is_whole(self):
        self.assertEqual([h["hook_node_id"] for h in self.flow.hooks], ["11", "12", "14"])
        after = next(h for h in self.flow.hooks if h["hook_node_id"] == "14")
        self.assertEqual(after["prev_hook_ids"], ["12"], "the stage after the break follows the body")
        self.assertEqual(after["prev_links"][0]["from_hook_id"], "12")

    def test_what_is_wired_into_the_loop_start_reaches_the_first_stage(self):
        first = self.flow.hooks[0]
        self.assertEqual(first["prev_hook_ids"], [])
        self.assertEqual([a["node_id"] for a in first["anchors"]], ["3"])
        self.assertEqual(first["anchor_node_id"], "3")

    def test_a_pipeline_without_flow_nodes_is_handed_back_as_it_came(self):
        hooks = [_hook(1), _hook(2, prev=[1])]
        flow = hf.plan(hooks)
        self.assertEqual(flow.hooks, hooks)
        self.assertEqual(flow.loops, [])

    def test_what_is_wired_wrong_is_said(self):
        empty = hf.plan([_hook(10, hf.LOOP_START), _hook(13, hf.LOOP_BREAK, prev=[10], condition="x")])
        self.assertIn("nothing is wired between", empty.problems[0])
        no_start = hf.plan([_hook(11), _hook(13, hf.LOOP_BREAK, prev=[11], condition="x")])
        self.assertEqual(no_start.loops[0].members, ["11"])
        self.assertIn("no loop start", no_start.problems[0])
        lonely = hf.plan([_hook(10, hf.LOOP_START), _hook(11, prev=[10])])
        self.assertIn("no loop break", lonely.problems[0])

    def test_rounds_and_forward_fall_back_to_something_sane(self):
        self.assertEqual([hf.clamp_rounds(v) for v in (0, 99, "x", None, 5)], [1, 10, 3, 3, 5])
        self.assertEqual([hf.forward_mode(v) for v in ("ALL", "nonsense", "")], ["all", "best", "best"])

    def test_the_loop_nodes_come_out_of_the_graph_like_hooks(self):
        # They carry only the execution wire now, so there is nothing of theirs to
        # pass through: they go, and what is left is the graph that renders.
        prompt = {
            "3": {"class_type": "LoadImage", "inputs": {"image": "ref.png"}},
            "10": {"class_type": "AgentYLoopStart", "inputs": {}},
            "13": {"class_type": "AgentYLoopBreak", "inputs": {"exec": ["10", 0], "condition": "x"}},
            "20": {"class_type": "PreviewImage", "inputs": {"images": ["3", 0]}},
        }
        clean, removed = splice_hook_nodes(prompt)
        self.assertEqual(sorted(removed), ["10", "13"])
        self.assertEqual(sorted(clean), ["20", "3"])
        self.assertEqual(clean["20"]["inputs"]["images"], ["3", 0])


def _result(path, passed, missed=(), score=0.5):
    return {"path": path, "passed": passed, "missed": list(missed), "summary": "", "score": score}


class ARound(unittest.TestCase):

    def setUp(self):
        self.loop = hf.Loop(break_id="13", members=["11"], condition="c", max_rounds=3)
        self.state = hf.new_state(self.loop)

    def test_a_miss_goes_round_again_with_what_was_missed(self):
        v = hf.record_round(self.state, [_result("a.png", False, ["pose is mirrored"], 0.4),
                                         _result("b.png", False, ["pose is mirrored", "blurry"], 0.9)], self.loop)
        self.assertFalse(v["finished"])
        self.assertEqual(v["missed"], ["pose is mirrored", "blurry"])
        self.assertEqual(v["closest"], "a.png", "fewest objections first, then the score")
        self.assertEqual(v["rounds_left"], 2)

    def test_meeting_the_condition_ends_it_and_names_what_goes_on(self):
        hf.record_round(self.state, [_result("a.png", False, ["x"])], self.loop)
        v = hf.record_round(self.state, [_result("b.png", True, score=0.2), _result("c.png", True, score=0.8)],
                            self.loop)
        self.assertTrue(v["finished"] and v["condition_met"])
        self.assertEqual(v["forward"], ["c.png"])
        self.assertEqual(self.state["outcome"], "met")

    def test_all_that_pass_and_all(self):
        results = [_result("a.png", True), _result("b.png", False, ["x"]), _result("c.png", True)]
        self.assertEqual(hf.choose(results, hf.FORWARD_PASSING), ["a.png", "c.png"])
        self.assertEqual(hf.choose(results, hf.FORWARD_ALL), ["a.png", "b.png", "c.png"])
        self.assertEqual(hf.choose([_result("b.png", False, ["x"])], hf.FORWARD_PASSING), ["b.png"],
                         "nothing passed: the best attempt still goes on")

    def test_out_of_rounds_sends_on_the_best_attempt_of_any_round(self):
        hf.record_round(self.state, [_result("r1.png", False, ["a"], 0.9)], self.loop)
        hf.record_round(self.state, [_result("r2.png", False, ["a", "b"], 0.9)], self.loop)
        v = hf.record_round(self.state, [_result("r3.png", False, ["a", "b", "c"], 0.9)], self.loop)
        self.assertTrue(v["finished"])
        self.assertFalse(v["condition_met"])
        self.assertEqual(v["forward"], ["r1.png"])
        self.assertEqual(self.state["outcome"], "out_of_rounds")


class TheJudge(unittest.TestCase):

    def test_the_condition_is_judged_with_the_stage_briefings(self):
        from src.utils.qa import QaBriefing
        loop = hf.Loop(break_id="13", condition="pose matches\nno text in frame")
        house = QaBriefing(criteria="warm skin tones", reference_paths=("ref.png",), sources=("hook",))
        briefing = lj.loop_briefing(loop, [house])
        self.assertEqual(briefing.criteria.splitlines(), ["pose matches", "no text in frame", "warm skin tones"])
        self.assertEqual(briefing.reference_paths, ("ref.png",))

    def test_one_verdict_per_file(self):
        seen = []

        def check(path, briefing, request=""):
            seen.append((path, briefing.criteria))
            bad = path == "b.png"
            return SimpleNamespace(passed=not bad, summary="s", error="",
                                   failed_criteria=lambda: ["pose is off"] if bad else [])

        out = lj.judge(["a.png", "b.png"], hf.Loop(break_id="13", condition="pose matches"),
                       check=check, score=lambda p: {"score": 0.7})
        self.assertEqual([(r["path"], r["passed"], r["missed"]) for r in out],
                         [("a.png", True, []), ("b.png", False, ["pose is off"])])
        self.assertEqual(seen[0], ("a.png", "pose matches"))
        self.assertEqual(out[0]["score"], 0.7)

    def test_a_judge_that_breaks_does_not_condemn_the_work(self):
        def check(*a, **k):
            raise RuntimeError("model down")
        (r,) = lj.judge(["a.png"], hf.Loop(break_id="13", condition="x"), check=check, score=lambda p: {})
        self.assertTrue(r["passed"])
        self.assertIn("model down", r["error"])


class WhatTheAgentIsTold(unittest.TestCase):

    def setUp(self):
        self.flow = hf.plan(_pipeline())
        self.block = describe_hooks(self.flow.hooks, {}, flow=self.flow, into_canvas=True)

    def test_the_loop_its_body_and_its_end(self):
        self.assertIn('finished when: "the pose matches the reference"; at most 4 round(s)', self.block)
        self.assertIn("body stage 1: hook 11", self.block)
        self.assertIn("body stage 2: hook 12", self.block)
        self.assertIn("loop_check(break_node_id", self.block)
        self.assertNotIn("PRODUCER hook 13", self.block, "a loop node is not work")

    def test_the_stage_after_the_break_is_still_part_of_the_chain(self):
        self.assertIn("MAKE-WORKFLOW CHAINS", self.block)
        self.assertIn('Stage 3 [INPUT = stage 2', self.block)

    def test_workflows_go_into_the_open_graph(self):
        self.assertIn("insert_workflow_into_canvas(workflow_path, hook_node_id=", self.block)
        off = describe_hooks(self.flow.hooks, {}, flow=self.flow, into_canvas=False)
        self.assertNotIn("ON THE CANVAS", off)

    def test_one_chain_is_not_parallel(self):
        self.assertNotIn("PARALLEL BRANCHES", self.block)

    def test_chains_that_share_no_wire_are_worked_on_at_once(self):
        hooks = [_hook(1, directive="hero shot"), _hook(2, prev=[1], directive="animate hero"),
                 _hook(5, directive="villain shot"), _hook(7, "text", directive="write a caption")]
        self.assertEqual(hf.branches(hooks), [["1", "2"], ["5"], ["7"]])
        block = describe_hooks(hooks, {}, flow=hf.plan(hooks), into_canvas=True)
        self.assertIn("PARALLEL BRANCHES — the execution wire splits into 2 branches", block)
        self.assertIn("start_shot(name, briefing, hook_ids=[...])", block)
        self.assertIn("- branch 1: hook 1 \"hero shot\"; hook 2 \"animate hero\" — hook_ids=['1', '2']",
                      block)


class TheTools(unittest.TestCase):

    def setUp(self):
        self.enterContext(mock.patch("src.utils.canvas_view.full_graph_visible", return_value=False))
        self.enterContext(mock.patch("src.agent._load_settings", return_value={"hook_subgraph_min_nodes": 6}))
        self.enterContext(mock.patch("src.utils.preflight._schema", side_effect=lambda c: SCHEMAS.get(c, {})))
        from src.utils.canvas_patch import clear
        clear()
        self.addCleanup(clear)
        token = turn_scope.enter(turn_scope.Scope("req", "thread"))
        self.addCleanup(turn_scope.leave, token)
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.flow = hf.plan(_pipeline())

    def _file(self, name, text="x"):
        p = Path(self.tmp.name) / name
        p.write_text(text, encoding="utf-8")
        return str(p)

    def _pipe(self, **kw):
        return pipeline_stub(_canvas_graph=_graph(), _canvas_flow=self.flow,
                             _canvas_hooks=self.flow.hooks, _loop_states={}, _forwarded={}, **kw)

    def _check(self, pipe, verdicts, files):
        def judge(paths, loop, briefings, request=""):
            return [{"path": p, "passed": verdicts[Path(p).name][0],
                     "missed": list(verdicts[Path(p).name][1]), "summary": "", "score": 0.5}
                    for p in paths]
        with mock.patch("src.utils.loop_judge.judge", judge):
            return json.loads(asyncio.run(tools(pipe)["loop_check"](break_node_id="13", outputs=files)))

    def test_a_loop_runs_until_the_judge_is_satisfied(self):
        from src.utils.canvas_patch import drain
        pipe = self._pipe()
        a, b = self._file("a.png"), self._file("b.png")
        first = self._check(pipe, {"a.png": (False, ["pose is mirrored"])}, [a])
        self.assertFalse(first["finished"])
        self.assertEqual((first["round"], first["rounds_left"]), (1, 3))
        second = self._check(pipe, {"b.png": (True, [])}, [b])
        self.assertTrue(second["finished"] and second["condition_met"])
        self.assertEqual(second["forward"], [b])
        self.assertEqual(pipe._forwarded["12"], [b], "the body's last stage forwards the chosen file")
        states = [p for p in drain() if p.get("op") == "flow_state"]
        self.assertEqual([s["state"] for s in states], ["running", "met"])
        again = self._check(pipe, {"b.png": (True, [])}, [b])
        self.assertIn("already ended", again["note"])

    def test_it_refuses_what_it_cannot_judge(self):
        pipe = self._pipe()
        out = json.loads(asyncio.run(tools(pipe)["loop_check"](break_node_id="99", outputs=["x"])))
        self.assertEqual(out["loops"], ["13"])
        out = json.loads(asyncio.run(tools(pipe)["loop_check"](break_node_id="13", outputs=["Z:/nope.png"])))
        self.assertIn("none of these files exist", out["error"])
        self.assertEqual(pipe._loop_states["13"]["round"], 0, "a refused call is not a round")

    def test_the_agent_names_what_goes_on(self):
        pipe = self._pipe(_hook_products={"11": ["W:/a.png", "W:/b.png"]})
        call = tools(pipe)["forward_outputs"]
        out = json.loads(asyncio.run(call(hook_node_id="11", outputs=["W:/b.png"], reason="sharper")))
        self.assertEqual(out["forwarded"], ["W:/b.png"])
        self.assertEqual(pipe._forwarded["11"], ["W:/b.png"])
        out = json.loads(asyncio.run(call(hook_node_id="11", outputs=["W:/other.png"])))
        self.assertIn("not produced by that stage", out["error"])

    def _insert(self, pipe, **kw):
        from src.utils.canvas_patch import drain
        path = self._file("built.json", json.dumps(BUILT))
        out = json.loads(asyncio.run(tools(pipe)["insert_workflow_into_canvas"](workflow_path=path, **kw)))
        self.assertEqual(out["status"], "applied", out)
        return next(p for p in drain() if p.get("op") == "edit_graph")

    def test_a_stage_workflow_is_placed_for_its_hook_and_a_large_one_folded(self):
        patch = self._insert(self._pipe(), hook_node_id="11", reason="Dancer on stage")
        self.assertEqual(patch["block"]["stage"],
                         {"hook_node_id": "11", "name": "Dancer on stage", "collapse": True})

    def test_a_small_one_stays_flat(self):
        with mock.patch("src.agent._load_settings", return_value={"hook_subgraph_min_nodes": 12}):
            patch = self._insert(self._pipe(), hook_node_id="11")
        self.assertFalse(patch["block"]["stage"]["collapse"])

    def test_outside_a_hook_run_nothing_changes(self):
        patch = self._insert(self._pipe())
        self.assertNotIn("stage", patch.get("block") or {})


if __name__ == "__main__":
    unittest.main()
