"""Removing nodes — the one canvas edit that destroys something.

Everything else the agent does to a canvas adds or overwrites, and an overwrite
still leaves a node to look at. A deletion leaves a hole, and the hole is often
somewhere else: the node that went was feeding two others, and the graph now
fails validation for a reason nobody watching the delete would connect to it.

So the tool answers "what am I about to lose?" before it goes: what each node
actually IS (an id is not something anyone can picture) and which inputs
elsewhere lose their feed. And it is wrapped in ComfyUI's undo hooks, which the
extension had never called for any of its edits.

    python -m unittest discover -s tests
"""

import json
import unittest
from unittest import mock

from pipeline_stub import pipeline_stub, tools
from src.utils.canvas_view import deletion_impact


def _graph():
    return {
        "3": {"class_type": "KSampler",
              "inputs": {"model": ["4", 0], "positive": ["6", 0], "steps": 30},
              "_meta": {"title": "Main sampler"}},
        "4": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": "x.safetensors"}},
        "6": {"class_type": "CLIPTextEncode", "inputs": {"text": "a cat", "clip": ["4", 1]}},
        "9": {"class_type": "SaveImage", "inputs": {"filename_prefix": "out",
                                                    "images": ["3", 0]}},
    }


class ImpactTest(unittest.TestCase):

    def test_it_says_what_the_nodes_actually_are(self):
        got = deletion_impact(_graph(), ["3"])
        self.assertEqual(got["found"][0]["class_type"], "KSampler")
        self.assertEqual(got["found"][0]["title"], "Main sampler")

    def test_it_names_every_input_that_loses_its_feed(self):
        """Deleting the checkpoint breaks the sampler AND the text encode."""
        orphaned = deletion_impact(_graph(), ["4"])["orphaned"]
        self.assertEqual({(o["node_id"], o["input"]) for o in orphaned},
                         {("3", "model"), ("6", "clip")})

    def test_a_link_between_two_doomed_nodes_is_not_an_orphan(self):
        """Delete both ends and nothing is left dangling — do not cry wolf.

        3 feeds 9; removing the pair leaves nobody looking for either.
        """
        self.assertEqual(deletion_impact(_graph(), ["3", "9"])["orphaned"], [])

    def test_a_surviving_node_reports_every_input_it_lost(self):
        """Deleting 4 and 6 strands BOTH of the sampler's inputs, not just one."""
        orphaned = deletion_impact(_graph(), ["4", "6"])["orphaned"]
        self.assertEqual({(o["node_id"], o["input"], o["was_fed_by"]) for o in orphaned},
                         {("3", "model", "4"), ("3", "positive", "6")})

    def test_a_leaf_breaks_nothing(self):
        self.assertEqual(deletion_impact(_graph(), ["9"])["orphaned"], [])

    def test_ids_that_are_not_there_come_back_as_missing(self):
        got = deletion_impact(_graph(), ["3", "999"])
        self.assertEqual([f["node_id"] for f in got["found"]], ["3"])
        self.assertEqual(got["missing"], ["999"])

    def test_no_graph_is_not_a_crash(self):
        self.assertEqual(deletion_impact(None, ["3"])["found"], [])


class ToolTest(unittest.TestCase):

    def setUp(self):
        self.enterContext(mock.patch("src.utils.canvas_view.full_graph_visible",
                                     return_value=True))
        from src.utils.canvas_patch import clear
        clear()
        self.addCleanup(clear)

    def _pipe(self, **over):
        over.setdefault("_canvas_graph", _graph())
        return pipeline_stub(**over)

    def _call(self, pipe, ids, **kw):
        import asyncio
        return json.loads(asyncio.run(
            tools(pipe)["delete_canvas_nodes"](node_ids=ids, **kw)))

    def test_a_delete_reports_what_went_and_what_broke(self):
        out = self._call(self._pipe(), ["4"])
        self.assertEqual(out["status"], "deleted")
        self.assertEqual(out["deleted"][0]["class_type"], "CheckpointLoaderSimple")
        self.assertEqual(len(out["orphaned_inputs"]), 2)
        self.assertIn("will not run until they are rewired", out["message"])

    def test_it_reaches_the_canvas(self):
        from src.utils.canvas_patch import drain
        self._call(self._pipe(), ["9"])
        op = next(e for e in drain() if e.get("op") == "delete_nodes")
        self.assertEqual(op["node_ids"], ["9"])

    def test_a_clean_delete_does_not_warn_about_damage_it_did_not_do(self):
        out = self._call(self._pipe(), ["9"])
        self.assertEqual(out["orphaned_inputs"], [])
        self.assertNotIn("rewired", out["message"])

    def test_undo_is_mentioned_because_the_user_will_want_it(self):
        self.assertIn("Ctrl+Z", self._call(self._pipe(), ["9"])["message"])

    def test_nothing_to_delete_is_refused_rather_than_reported_as_success(self):
        self.assertIn("at least one node id", self._call(self._pipe(), [])["error"])

    def test_ids_that_are_on_no_canvas_are_refused(self):
        self.assertIn("no node(s) 999", self._call(self._pipe(), ["999"])["error"])

    def test_a_partly_valid_list_deletes_what_exists_and_says_what_did_not(self):
        out = self._call(self._pipe(), ["9", "999"])
        self.assertEqual(out["count"], 1)
        self.assertEqual(out["not_found"], ["999"])

    def test_duplicates_collapse(self):
        self.assertEqual(self._call(self._pipe(), ["9", "9", "9"])["count"], 1)

    def test_a_runaway_delete_is_capped(self):
        out = self._call(self._pipe(), [str(i) for i in range(40)])
        self.assertIn("more than this tool will delete", out["error"])
        self.assertIn("theirs to make", out["what_to_do"])

    def test_the_cap_does_not_fire_at_the_boundary(self):
        from src.pipeline import _MAX_CANVAS_DELETE
        pipe = self._pipe(_canvas_graph={str(i): {"class_type": "PreviewImage",
                                                  "inputs": {}}
                                         for i in range(_MAX_CANVAS_DELETE)})
        out = self._call(pipe, [str(i) for i in range(_MAX_CANVAS_DELETE)])
        self.assertEqual(out["count"], _MAX_CANVAS_DELETE)

    def test_the_reason_travels_to_the_canvas(self):
        from src.utils.canvas_patch import drain
        self._call(self._pipe(), ["9"], reason="you asked for the saver to go")
        op = next(e for e in drain() if e.get("op") == "delete_nodes")
        self.assertEqual(op["reason"], "you asked for the saver to go")


class TheBallotIsNotDeletableTest(unittest.TestCase):
    """A live review's collector IS the choice the user is in the middle of."""

    def setUp(self):
        self.enterContext(mock.patch("src.utils.canvas_view.full_graph_visible",
                                     return_value=True))

    def _pipe(self):
        from src.utils.review_gate import ReviewHalt
        hooks = [{"hook_node_id": "11", "purpose": "human_review", "directive": "which?",
                  "targets": [], "anchors": [
                      {"node_id": "77", "type": "AgentYImageCollector",
                       "widgets": {"files": "C:/out/a.png\nC:/out/b.png"}}]}]
        graph = dict(_graph())
        graph["77"] = {"class_type": "AgentYImageCollector",
                       "inputs": {"files": "C:/out/a.png"}}
        return pipeline_stub(_canvas_graph=graph, _canvas_hooks=hooks,
                             _review_halt=ReviewHalt(hook_node_id="11"))

    def _call(self, pipe, ids):
        import asyncio
        return json.loads(asyncio.run(
            tools(pipe)["delete_canvas_nodes"](node_ids=ids)))

    def test_deleting_it_mid_halt_is_refused(self):
        out = self._call(self._pipe(), ["77"])
        self.assertIn("stopped on", out["error"])
        self.assertIn("that is `stop`", out["what_to_do"])

    def test_other_nodes_are_still_deletable_during_a_halt(self):
        self.assertEqual(self._call(self._pipe(), ["9"])["status"], "deleted")

    def test_with_no_halt_it_is_an_ordinary_node(self):
        pipe = self._pipe()
        pipe._review_halt = None
        self.assertEqual(self._call(pipe, ["77"])["status"], "deleted")


class SelectionOnlyTest(unittest.TestCase):
    """With canvas_full_graph off, deleting is restricted exactly as editing is."""

    def setUp(self):
        self.enterContext(mock.patch("src.utils.canvas_view.full_graph_visible",
                                     return_value=False))

    def _call(self, pipe, ids):
        import asyncio
        return json.loads(asyncio.run(
            tools(pipe)["delete_canvas_nodes"](node_ids=ids)))

    def test_an_unselected_node_cannot_be_deleted(self):
        pipe = pipeline_stub(_canvas_graph=_graph())
        out = self._call(pipe, ["9"])
        self.assertIn("not in the current canvas selection", out["error"])
        self.assertIn("select what they want removed", out["what_to_do"])

    def test_a_selected_node_can_be(self):
        pipe = pipeline_stub(_canvas_graph=_graph(),
                             _canvas_selection=[{"id": "9", "type": "SaveImage"}])
        self.assertEqual(self._call(pipe, ["9"])["status"], "deleted")

    def test_one_selected_and_one_not_deletes_neither(self):
        """Half a delete is worse than none — it is the half nobody checked."""
        from src.utils.canvas_patch import clear, drain
        clear()
        self.addCleanup(clear)
        pipe = pipeline_stub(_canvas_graph=_graph(),
                             _canvas_selection=[{"id": "9", "type": "SaveImage"}])
        out = self._call(pipe, ["9", "4"])
        self.assertIn("not in the current canvas selection", out["error"])
        self.assertEqual([e for e in drain() if e.get("op") == "delete_nodes"], [])


if __name__ == "__main__":
    unittest.main()
