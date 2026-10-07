"""A built workflow goes INTO the graph the user has open, wired.

"Generate this with Seedream, Nano Banana and GPT Image. Place all nodes into
the currently open canvas." That produced three workflows in three tabs, and
the user's own graph untouched: there was no step that puts a built workflow
into the open graph, so the agent built each one as usual and it opened beside
theirs. insert_workflow_into_canvas adds a built workflow to the open graph,
taking the wiring from the workflow itself.
"""

import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from agenty_core.utils import turn_scope

from pipeline_stub import pipeline_stub, tools
from src.utils import canvas_edit as ce
from test_canvas_edit import SCHEMAS, _graph

# What prepare_workflow hands back: API format, ids as strings, links as [id, slot].
BUILT = {
    "10": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": "a.safetensors"}},
    "11": {"class_type": "CLIPTextEncode", "inputs": {"text": "an old man in a desert", "clip": ["10", 1]}},
    "12": {"class_type": "EmptyLatentImage", "inputs": {"width": 512}},
    "13": {"class_type": "KSampler", "inputs": {
        "model": ["10", 0], "positive": ["11", 0], "negative": ["11", 0], "latent_image": ["12", 0],
        "seed": 7, "steps": 20, "denoise": 1.0, "sampler_name": "euler"}},
    "14": {"class_type": "VAEDecode", "inputs": {"samples": ["13", 0], "vae": ["10", 2]},
           "_meta": {"title": "decode"}},
    "15": {"class_type": "SaveImage", "inputs": {"images": ["14", 0], "filename_prefix": "raven"}},
}
LINKS = 8


def _links(graph):
    return sum(1 for n in graph.values() for v in (n.get("inputs") or {}).values() if ce._is_link(v))


class TheOps(unittest.TestCase):

    def test_every_node_and_every_wire_of_the_workflow(self):
        ops = ce.ops_for_workflow(BUILT)
        adds = [o for o in ops if o["op"] == "add"]
        wires = [o for o in ops if o["op"] == "connect"]
        self.assertEqual(sorted(o["class_type"] for o in adds), sorted(n["class_type"] for n in BUILT.values()))
        self.assertEqual(len(wires), LINKS)
        self.assertEqual(ops.index(wires[0]), len(adds), "all nodes exist before the first wire")

    def test_values_travel_as_params_and_links_do_not(self):
        sampler = next(o for o in ce.ops_for_workflow(BUILT) if o["class_type"] == "KSampler")
        self.assertEqual(sampler["params"], {"seed": 7, "steps": 20, "denoise": 1.0, "sampler_name": "euler"})

    def test_a_node_is_placed_beside_the_one_that_feeds_it(self):
        ops = ce.ops_for_workflow(BUILT)
        seen = set()
        for o in (o for o in ops if o["op"] == "add"):
            if "near" in o:
                self.assertIn(o["near"], seen, "its neighbour must already have been added")
            seen.add(o["ref"])
        loader = next(o for o in ops if o["class_type"] == "CheckpointLoaderSimple")
        self.assertNotIn("near", loader)

    def test_a_saved_canvas_graph_is_refused_not_guessed_at(self):
        with self.assertRaises(ValueError):
            ce.ops_for_workflow({"nodes": [], "links": []})
        with self.assertRaises(ValueError):
            ce.ops_for_workflow({})

    def test_a_loop_in_the_file_does_not_hang_it(self):
        loop = {"1": {"class_type": "A", "inputs": {"x": ["2", 0]}},
                "2": {"class_type": "A", "inputs": {"x": ["1", 0]}}}
        self.assertEqual(len([o for o in ce.ops_for_workflow(loop) if o["op"] == "add"]), 2)


class OntoTheGraph(unittest.TestCase):

    def test_it_lands_beside_what_is_there_with_all_its_wires(self):
        before = _graph()
        res = ce.plan(before, ce.ops_for_workflow(BUILT), SCHEMAS)
        self.assertTrue(res["ok"], res["errors"])
        after = res["graph"]
        self.assertEqual(len(after), len(before) + len(BUILT))
        self.assertEqual(_links(after), _links(before) + LINKS)
        for nid, node in before.items():
            self.assertEqual(after[nid], node, "the user's own nodes are untouched")

    def test_three_models_are_three_workflows_in_one_graph(self):
        graph = _graph()
        for _ in range(3):
            res = ce.plan(graph, ce.ops_for_workflow(BUILT), SCHEMAS)
            self.assertTrue(res["ok"], res["errors"])
            graph = res["graph"]
        self.assertEqual(len(graph), len(_graph()) + 3 * len(BUILT))
        self.assertEqual(_links(graph), _links(_graph()) + 3 * LINKS)

    def test_an_empty_canvas_takes_it_too(self):
        res = ce.plan({}, ce.ops_for_workflow(BUILT), SCHEMAS)
        self.assertTrue(res["ok"], res["errors"])
        self.assertEqual(_links(res["graph"]), LINKS)


class TheTool(unittest.TestCase):

    def setUp(self):
        self.enterContext(mock.patch("src.utils.canvas_view.full_graph_visible", return_value=False))
        self.enterContext(mock.patch("src.utils.preflight._schema", side_effect=lambda c: SCHEMAS.get(c, {})))
        from src.utils.canvas_patch import clear
        clear()
        self.addCleanup(clear)
        token = turn_scope.enter(turn_scope.Scope("req", "thread"))
        self.addCleanup(turn_scope.leave, token)
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.path = Path(tmp.name) / "built.json"
        self.path.write_text(json.dumps(BUILT), encoding="utf-8")

    def _call(self, pipe, **kw):
        return json.loads(asyncio.run(tools(pipe)["insert_workflow_into_canvas"](**kw)))

    def test_it_reaches_the_canvas_wired_without_anything_selected(self):
        from src.utils.canvas_patch import drain
        pipe = pipeline_stub(_canvas_graph=_graph(), _canvas_selection=[])
        out = self._call(pipe, workflow_path=str(self.path), reason="raven, model A")
        self.assertEqual(out["status"], "applied", out)
        self.assertEqual((out["nodes_added"], out["wires_set"]), (len(BUILT), LINKS))
        patch = next(e for e in drain() if e.get("op") == "edit_graph")
        self.assertEqual(sum(o["op"] == "connect" for o in patch["ops"]), LINKS)
        self.assertEqual(_links(pipe._canvas_graph), _links(_graph()) + LINKS)

    def test_it_does_not_count_as_an_edit_that_blocks_the_next_build(self):
        """Three models means build, insert, build, insert… - an insert must not
        make the next prepare_workflow refuse as "you already edited the canvas"."""
        pipe = pipeline_stub(_canvas_graph=_graph(), _canvas_selection=[], _canvas_edits=[])
        self._call(pipe, workflow_path=str(self.path))
        self.assertEqual(pipe._canvas_edits, [])

    def test_running_it_afterwards_does_not_open_it_in_a_tab_as_well(self):
        from src import executor
        self.assertFalse(executor._already_on_canvas(self.path))
        self._call(pipeline_stub(_canvas_graph=_graph(), _canvas_selection=[]), workflow_path=str(self.path))
        self.assertTrue(executor._already_on_canvas(self.path))
        self.assertFalse(executor._already_on_canvas(self.path.with_name("other.json")))

    def test_an_open_but_empty_canvas_is_a_canvas(self):
        pipe = pipeline_stub(_canvas_graph={}, _canvas_selection=[], _open_workflows=[{"name": "Unsaved"}])
        self.assertEqual(self._call(pipe, workflow_path=str(self.path))["status"], "applied")

    def test_no_canvas_at_all_says_what_to_do_instead(self):
        out = self._call(pipeline_stub(_canvas_graph={}, _canvas_selection=[], _open_workflows=[]),
                         workflow_path=str(self.path))
        self.assertIn("open_workflow_in_canvas", out["what_to_do"])

    def test_a_file_that_is_not_a_built_workflow_changes_nothing(self):
        self.path.write_text(json.dumps({"nodes": [], "links": []}), encoding="utf-8")
        pipe = pipeline_stub(_canvas_graph=_graph(), _canvas_selection=[])
        out = self._call(pipe, workflow_path=str(self.path))
        self.assertIn("error", out)
        self.assertEqual(pipe._canvas_graph, _graph())


class TheRule(unittest.TestCase):

    def test_the_orchestrator_is_told_and_has_the_tool(self):
        prompt = (Path(__file__).resolve().parents[1] / "config" / "system_prompts"
                  / "system_prompt.orchestrator.md").read_text(encoding="utf-8")
        section = prompt.split("### Into the graph the user has open", 1)[1].split("###", 1)[0]
        for word in ("insert_workflow_into_canvas", "prepare_workflow", "run_workflow_now", "same"):
            self.assertIn(word, section)
        self.assertIn("insert_workflow_into_canvas", tools(pipeline_stub(_canvas_graph=_graph())))


if __name__ == "__main__":
    unittest.main()
