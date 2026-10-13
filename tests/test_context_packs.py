"""Tools that belong to a kind of turn are in the list on that kind of turn.

The stages of a hook pipeline and the changing of an open graph are not rare -
they are simply not every turn's. Whether a turn is of that kind is known before
it starts, so the pipeline pins the pack and the model never has to ask; on
every other turn their definitions are not sent.
"""

import unittest
from types import SimpleNamespace

from pipeline_stub import pipeline_stub
from src.tools import tool_packs as tp


def _pins(**over):
    agent = SimpleNamespace()
    base = dict(_orchestrator_agent=agent, _canvas_hooks=[], _review_halt=None, _loop_states={},
                _canvas_graph={}, _canvas_selection=[])
    base.update(over)
    p = pipeline_stub(**base)
    p._pin_tool_packs(over.pop("lead", False) if "lead" in over else False)
    return agent._tool_packs_pinned


class WhichTurnsGetWhat(unittest.TestCase):

    def test_a_plain_turn_gets_neither(self):
        self.assertEqual(_pins(), set())

    def test_hook_nodes_on_the_canvas_bring_the_hook_tools(self):
        self.assertEqual(_pins(_canvas_hooks=[{"hook_node_id": "4", "purpose": "make_workflow"}]), {"hooks"})

    def test_a_review_that_is_waiting_or_a_loop_that_is_running_does_too(self):
        self.assertIn("hooks", _pins(_review_halt=SimpleNamespace(hook_node_id="9")))
        self.assertIn("hooks", _pins(_loop_states={"11": {"round": 1}}))

    def test_a_graph_that_came_with_the_message_brings_the_graph_tools(self):
        self.assertEqual(_pins(_canvas_graph={"1": {"class_type": "KSampler", "inputs": {}}}), {"graph"})
        self.assertEqual(_pins(_canvas_selection=[{"id": 3}]), {"graph"})

    def test_working_in_the_open_graph_does_too(self):
        session = SimpleNamespace(current_output_paths=[], open_graph_mode=True, inserted_workflows={},
                                  generated_images=[])
        self.assertEqual(_pins(_session=session), {"graph"})

    def test_both_at_once(self):
        got = _pins(_canvas_hooks=[{"hook_node_id": "4"}], _canvas_graph={"1": {}})
        self.assertEqual(got, {"hooks", "graph"})


class ThePacks(unittest.TestCase):

    def test_what_is_in_them(self):
        self.assertEqual(set(tp.PACKS["hooks"][1]), {"apply_canvas_hooks", "stop_hook_run", "halt_for_review",
                                                     "release_review", "loop_check", "forward_outputs",
                                                     "place_canvas_text"})
        self.assertEqual(set(tp.PACKS["graph"][1]), {"edit_canvas_graph", "delete_canvas_nodes", "run_canvas"})
        self.assertEqual(tp.CONTEXT_PACKS, ("hooks", "graph"))

    def test_what_works_on_a_built_workflow_too_stays_in_every_list(self):
        """set_canvas_node_params and get_canvas_node take a workflow_path: they
        are how a save node of a workflow that was just built is set and read."""
        for name in ("set_canvas_node_params", "get_canvas_node", "insert_workflow_into_canvas",
                     "open_workflow_in_canvas", "name_outputs", "work_in_open_graph", "run_workflow_now"):
            self.assertNotIn(name, tp._PACK_OF, name)

    def test_they_are_listed_in_the_prompt_so_one_can_still_be_asked_for(self):
        held = {"hooks": [SimpleNamespace(tool_name=n) for n in tp.PACKS["hooks"][1]],
                "graph": [SimpleNamespace(tool_name=n) for n in tp.PACKS["graph"][1]]}
        text = tp.catalogue(held)
        self.assertIn("`apply_canvas_hooks`", text)
        self.assertIn("`run_canvas`", text)
        self.assertIn("already in your list whenever", text)


if __name__ == "__main__":
    unittest.main()
