"""The common changes go through change_graph first: a hand-wiring call that one
change_graph call covers is handed back once, and runs when sent again."""

import json
import unittest
from types import SimpleNamespace

from src.utils import splice_first as sf


def call(name, **inp):
    agent = call.agent
    return SimpleNamespace(agent=agent, tool_use={"name": name, "input": inp}, cancel_tool=False)


def fresh(tools=("change_graph", "edit_canvas_graph", "update_workflow")):
    call.agent = SimpleNamespace(tool_registry=SimpleNamespace(registry={t: object() for t in tools}))
    return sf.SpliceFirstHook()


class WhichCallsAmountToACommonChange(unittest.TestCase):

    def test_adding_a_controlnet_on_the_canvas(self):
        ops = [{"op": "add", "class_type": "ControlNetLoader", "ref": "l"},
               {"op": "add", "class_type": "ControlNetApplyAdvanced", "ref": "a"},
               {"op": "add", "class_type": "LoadImage", "ref": "i"},
               {"op": "connect", "from": "l", "to": "a", "input": "control_net"}]
        self.assertEqual(sf.operation_for("edit_canvas_graph", {"ops": ops}),
                         ("controlnet", ["ControlNetLoader", "ControlNetApplyAdvanced"]))

    def test_adding_freeu_to_a_file_with_the_list_as_text(self):
        got = sf.operation_for("update_workflow", {"workflow_path": "w.json",
                                                   "add_nodes": json.dumps([{"class_type": "FreeU_V2"}])})
        self.assertEqual(got, ("patch_model", ["FreeU_V2"]))

    def test_wiring_and_values_alone_are_nobodys_business(self):
        self.assertEqual(sf.operation_for("edit_canvas_graph", {"ops": [{"op": "connect", "from": "1", "to": "2",
                                                                         "input": "model"}]}), ("", []))
        self.assertEqual(sf.operation_for("update_workflow", {"patches": "[]", "add_nodes": "[]"}), ("", []))
        self.assertEqual(sf.operation_for("set_canvas_node_params", {"node_id": "3"}), ("", []))

    def test_a_node_of_no_common_change_is_left_alone(self):
        self.assertEqual(sf.operation_for("edit_canvas_graph", {"ops": [{"op": "add", "class_type": "FluxGuidance"}]}),
                         ("", []))

    def test_two_changes_in_one_call_are_somebodys_own_design(self):
        ops = [{"op": "add", "class_type": "FreeU_V2"}, {"op": "add", "class_type": "LatentUpscaleBy"}]
        self.assertEqual(sf.operation_for("edit_canvas_graph", {"ops": ops}), ("", []))


class TheRedirect(unittest.TestCase):

    def test_the_first_hand_call_is_handed_back_with_the_one_call_that_does_it(self):
        hook = fresh()
        ev = call("update_workflow", workflow_path="w.json", add_nodes=[{"class_type": "FreeU_V2"}])
        hook._on_tool(ev)
        self.assertIn("NOT RUN", ev.cancel_tool)
        self.assertIn('change_graph(operation="patch_model", params={"class_type": "FreeU_V2"}, workflow_path="w.json")',
                      ev.cancel_tool)

    def test_the_same_call_again_runs(self):
        hook = fresh()
        first = call("edit_canvas_graph", ops=[{"op": "add", "class_type": "LatentUpscaleBy"}])
        hook._on_tool(first)
        again = call("edit_canvas_graph", ops=[{"op": "add", "class_type": "LatentUpscaleBy"}])
        hook._on_tool(again)
        self.assertTrue(first.cancel_tool)
        self.assertFalse(again.cancel_tool)

    def test_after_change_graph_was_called_for_it_hand_work_is_finishing_work(self):
        hook = fresh()
        hook._on_tool(call("change_graph", operation="controlnet", params={}))
        ev = call("edit_canvas_graph", ops=[{"op": "add", "class_type": "ControlNetLoader"}])
        hook._on_tool(ev)
        self.assertFalse(ev.cancel_tool)

    def test_another_operation_is_still_redirected(self):
        hook = fresh()
        hook._on_tool(call("change_graph", operation="controlnet", params={}))
        ev = call("edit_canvas_graph", ops=[{"op": "add", "class_type": "FreeU_V2"}])
        hook._on_tool(ev)
        self.assertTrue(ev.cancel_tool)

    def test_a_new_turn_starts_over(self):
        hook = fresh()
        hook._on_tool(call("edit_canvas_graph", ops=[{"op": "add", "class_type": "FreeU_V2"}]))
        hook._on_turn(SimpleNamespace(agent=call.agent))
        ev = call("edit_canvas_graph", ops=[{"op": "add", "class_type": "FreeU_V2"}])
        hook._on_tool(ev)
        self.assertTrue(ev.cancel_tool)

    def test_an_agent_without_change_graph_is_never_redirected(self):
        hook = fresh(tools=("update_workflow",))
        ev = call("update_workflow", add_nodes=[{"class_type": "FreeU_V2"}])
        hook._on_tool(ev)
        self.assertFalse(ev.cancel_tool)

    def test_a_call_something_else_already_stopped_is_left(self):
        hook = fresh()
        ev = call("edit_canvas_graph", ops=[{"op": "add", "class_type": "FreeU_V2"}])
        ev.cancel_tool = "denied"
        hook._on_tool(ev)
        self.assertEqual(ev.cancel_tool, "denied")


if __name__ == "__main__":
    unittest.main()
