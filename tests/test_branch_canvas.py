"""Branches of a pipeline do not wait on each other for the canvas.

Two branches ran side by side until one of them set a value: it then held the
canvas, and the other was told "another conversation is editing the open canvas"
until that turn ended. And the value itself went nowhere - a branch's turn has
no page to carry a patch - so the branch had lost it by its next turn.

A branch's values are now its own, and reach the canvas through its lead.
"""
import json
import os
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

from src.pipeline import Pipeline
from src.utils import branch_canvas as bc
from src.utils import canvas_lease
from src.utils import conversation_store as cs

ROOT = Path(__file__).resolve().parent.parent


def value(node, **params):
    return {"node_id": node, "params": params, "node_title": "gen"}


class Store(unittest.TestCase):
    def setUp(self):
        self._dir = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.addCleanup(self._dir.cleanup)
        patcher = mock.patch.dict(os.environ, {
            "AGENTY_CONVERSATION_DB": str(Path(self._dir.name) / "c.sqlite")})
        patcher.start()
        self.addCleanup(patcher.stop)
        cs._INITIALISED = False
        self.addCleanup(setattr, cs, "_INITIALISED", False)
        for tid in ("lead", "b1", "b2"):
            cs.create_thread("t", thread_id=tid)


class WhatIsAValue(unittest.TestCase):
    def test_setting_parameters_and_placing_a_text_are(self):
        self.assertTrue(bc.is_value(value("7", prompt="a")))
        self.assertTrue(bc.is_value({"op": "place_text", "hook_node_id": "4", "text": "x"}))

    def test_changing_the_graph_is_not(self):
        for op in ("edit_graph", "delete_nodes", "set_mode", "insert_workflow"):
            self.assertFalse(bc.is_value({"op": op, "params": {}}), op)


class TheBranchKeepsItsValues(Store):
    def test_they_are_there_on_its_next_turn(self):
        bc.note("b1", value("7", prompt="MARA, 34"))
        canvas = {"7": {"class_type": "Gen", "inputs": {"prompt": "", "seed": 1}}}
        self.assertEqual(bc.overlay(canvas, "b1"), 1)
        self.assertEqual(canvas["7"]["inputs"], {"prompt": "MARA, 34", "seed": 1})

    def test_a_later_write_to_the_same_node_keeps_the_other_parameters(self):
        bc.note("b1", value("7", prompt="first", seed=5))
        bc.note("b1", value("7", prompt="second"))
        self.assertEqual(bc.values("b1"), [dict(value("7", prompt="second", seed=5))])

    def test_another_branch_does_not_get_them(self):
        bc.note("b1", value("7", prompt="mine"))
        canvas = {"7": {"class_type": "Gen", "inputs": {"prompt": "theirs"}}}
        self.assertEqual(bc.overlay(canvas, "b2"), 0)
        self.assertEqual(canvas["7"]["inputs"]["prompt"], "theirs")

    def test_a_node_the_user_deleted_and_an_input_they_wired_are_left_alone(self):
        bc.note("b1", value("7", prompt="p"))
        bc.note("b1", value("9", prompt="p"))
        canvas = {"7": {"class_type": "Gen", "inputs": {"prompt": ["3", 0]}}}
        bc.overlay(canvas, "b1")
        self.assertEqual(canvas, {"7": {"class_type": "Gen", "inputs": {"prompt": ["3", 0]}}})

    def test_a_change_to_the_graph_is_never_kept(self):
        bc.note("b1", {"op": "delete_nodes", "node_ids": ["7"]})
        self.assertEqual(bc.values("b1"), [])


class TheyReachTheCanvasThroughTheLead(Store):
    def test_each_branchs_writes_wait_for_the_lead(self):
        bc.queue("lead", value("7", prompt="a"), "characters")
        bc.queue("lead", value("26", prompt="b"), "places")
        got = bc.take("lead")
        self.assertEqual([(p["node_id"], p["from_branch"]) for p in got],
                         [("7", "characters"), ("26", "places")])

    def test_taken_once(self):
        bc.queue("lead", value("7", prompt="a"), "characters")
        bc.take("lead")
        self.assertEqual(bc.take("lead"), [])

    def test_the_last_value_of_a_loop_is_the_one_delivered(self):
        for n in range(4):
            bc.queue("lead", value("7", prompt=f"round {n}"), "characters")
        self.assertEqual([p["params"] for p in bc.take("lead")], [{"prompt": "round 3"}])

    def test_they_survive_a_restart(self):
        bc.queue("lead", value("7", prompt="a"), "characters")
        cs._INITIALISED = False                      # a new process opens the same file
        self.assertEqual(bc.waiting("lead"), 1)


def me(branch=None, thread="b1", canvas=True):
    obj = types.SimpleNamespace(_verbose=False)
    obj._branch_info = lambda: branch
    obj._thread_id_now = lambda: thread
    return obj


BRANCH = {"thread_id": "b1", "lead_id": "lead", "name": "characters"}


class TheLock(unittest.TestCase):
    def setUp(self):
        canvas_lease.release(canvas_lease.holder())
        self.addCleanup(lambda: canvas_lease.release(canvas_lease.holder()))

    def scoped(self, thread):
        scope = types.SimpleNamespace(thread_id=thread)
        return mock.patch("agenty_core.utils.turn_scope.current", return_value=scope)

    def test_a_branch_setting_a_value_does_not_take_it(self):
        self.assertEqual(Pipeline._canvas_lease_refusal(me(BRANCH), structural=False), "")
        self.assertEqual(canvas_lease.holder(), "")

    def test_so_the_next_branch_is_not_refused(self):
        Pipeline._canvas_lease_refusal(me(BRANCH), structural=False)
        other = dict(BRANCH, thread_id="b2", name="places")
        self.assertEqual(Pipeline._canvas_lease_refusal(me(other, "b2"), structural=False), "")

    def test_a_branch_may_not_change_the_graph(self):
        said = json.loads(Pipeline._canvas_lease_refusal(me(BRANCH)))
        self.assertEqual(said["status"], "not_in_a_branch")
        self.assertIn("the lead reads it and makes the change", said["what_to_do"])
        self.assertEqual(canvas_lease.holder(), "")

    def test_an_ordinary_conversation_still_holds_the_canvas(self):
        with self.scoped("one"):
            self.assertEqual(Pipeline._canvas_lease_refusal(me(None, "one")), "")
        with self.scoped("two"):
            said = json.loads(Pipeline._canvas_lease_refusal(me(None, "two"), structural=False))
        self.assertEqual(said["status"], "canvas_in_use")

    def test_which_tools_are_which(self):
        src = (ROOT / "src" / "pipeline.py").read_text(encoding="utf-8")

        def opening(tool):
            body = src.split(f"        async def {tool}(", 1)[1].split("        @_tool", 1)[0]
            return body.split("_held = ", 1)[1].split("\n", 1)[0]

        for tool in ("set_canvas_node_params", "place_canvas_text", "refine_canvas_until"):
            self.assertEqual(opening(tool), "self._canvas_lease_refusal(structural=False)", tool)
        for tool in ("delete_canvas_nodes", "edit_canvas_graph", "insert_workflow_into_canvas",
                     "set_canvas_node_mode"):
            self.assertEqual(opening(tool), "self._canvas_lease_refusal()", tool)


class WhereAWriteGoes(Store):
    def pushed(self, obj, patch):
        with mock.patch("src.utils.canvas_patch.push") as push:
            said = Pipeline._send_canvas_value(obj, patch)
        return said, [c.args[0] for c in push.call_args_list]

    def test_from_the_lead_it_goes_to_the_canvas(self):
        said, pushed = self.pushed(me(None, "lead"), value("7", prompt="a"))
        self.assertEqual((said, pushed), ("", [value("7", prompt="a")]))

    def test_from_a_branch_it_is_kept_and_queued_and_not_pushed(self):
        said, pushed = self.pushed(me(BRANCH), value("7", prompt="a"))
        self.assertEqual(pushed, [])
        self.assertIn("lead's next turn", said)
        self.assertEqual(bc.values("b1"), [value("7", prompt="a")])
        self.assertEqual(bc.waiting("lead"), 1)

    def test_the_lead_reads_them_as_on_the_canvas_and_sends_them_on(self):
        bc.queue("lead", value("7", prompt="from the branch"), "characters")
        lead = me(None, "lead")
        canvas = {"7": {"class_type": "Gen", "inputs": {"prompt": ""}}}
        lead._branch_values_waiting = Pipeline._take_branch_values(lead, canvas)
        self.assertEqual(canvas["7"]["inputs"]["prompt"], "from the branch")
        with mock.patch("src.utils.canvas_patch.push") as push:
            self.assertEqual(Pipeline._deliver_branch_values(lead), 1)
        self.assertEqual(push.call_args.args[0]["from_branch"], "characters")
        self.assertEqual(Pipeline._deliver_branch_values(lead), 0)

    def test_with_no_canvas_open_they_go_on_waiting(self):
        bc.queue("lead", value("7", prompt="a"), "characters")
        self.assertEqual(Pipeline._take_branch_values(me(None, "lead"), None), [])
        self.assertEqual(bc.waiting("lead"), 1)

    def test_a_branch_delivers_nothing_itself(self):
        bc.queue("b1", value("7", prompt="a"), "x")
        self.assertEqual(Pipeline._take_branch_values(me(BRANCH), {"7": {"inputs": {}}}), [])

    def test_the_turn_does_both_at_its_start(self):
        src = (ROOT / "src" / "pipeline.py").read_text(encoding="utf-8")
        start = src.split("canvas_prompt = _no_exec(canvas_prompt)", 1)[1].split("_hook_flow.plan(", 1)[0]
        self.assertIn("_bc.overlay(canvas_prompt, self._thread_id_now())", start)
        self.assertIn("self._branch_values_waiting = self._take_branch_values(canvas_prompt)", start)
        self.assertIn("        _clear_canvas_patch()\n        # What this conversation's branches "
                      "wrote while it was not looking.\n        self._deliver_branch_values()", src)


class APageWatchingTheLead(unittest.TestCase):
    """A lead woken by its branches has no page of its own: a watcher gets its values."""

    def setUp(self):
        from src.utils import agentY_server as srv
        self.srv = srv
        srv._watch_sent.clear()
        self.addCleanup(srv._watch_sent.clear)

    def test_a_value_is_delivered_once(self):
        ev = {"type": "canvas_patch", **value("7", prompt="a"), "from_branch": "characters"}
        self.assertTrue(self.srv._watch_delivers("run1", 3, ev))
        self.assertFalse(self.srv._watch_delivers("run1", 3, ev))

    def test_a_placed_text_too(self):
        ev = {"type": "canvas_patch", "op": "place_text", "hook_node_id": "4", "text": "x"}
        self.assertTrue(self.srv._watch_delivers("run1", 4, ev))

    def test_a_change_to_the_graph_is_not(self):
        for op in ("edit_graph", "delete_nodes", "set_mode"):
            ev = {"type": "canvas_patch", "op": op, "params": {"a": 1}}
            self.assertFalse(self.srv._watch_delivers("run1", 5, ev), op)


class WhatEachIsTold(unittest.TestCase):
    def test_the_branch(self):
        src = (ROOT / "src" / "utils" / "shots.py").read_text(encoding="utf-8")
        self.assertIn("Values \"\n        \"on your own stages are yours to set", src)
        self.assertIn("say in your report exactly what \"\n        \"should change", src)

    def test_the_lead(self):
        from src.utils import hook_flow as hf

        def stage(hid, purpose, after=()):
            return {"hook_node_id": hid, "purpose": purpose, "exec_wired": True,
                    "via_hook_ids": list(after), "exec_prev_ids": list(after),
                    "prev_hook_ids": [], "anchors": [], "directive": "d",
                    "targets": [{"node_id": "9" + hid, "to_input": "prompt"}]}
        flow = hf.plan([stage("1", "text_only"), stage("2", "set_parameter", ["1"]),
                        stage("3", "set_parameter", ["1"])])
        text = "\n".join(hf.parallel_lines(flow))
        self.assertIn("THE CANVAS IS YOURS TO CHANGE, not a branch's", text)


if __name__ == "__main__":
    unittest.main()
