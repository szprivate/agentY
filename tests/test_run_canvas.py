"""Asked to run the open workflow, the agent runs the open workflow.

There was no way to: "run it" on a graph that was already on the canvas made the
agent write its own script that posted to ComfyUI, after hunting for a workflow
file to load. run_canvas runs what is on the canvas, as it is, with nothing
built, searched for or read first.

Also here: a conversation that works in the open graph is not told to offer
"want me to graph the generated workflows?" - they are on the canvas already.
"""

import asyncio
import inspect
import json
import unittest
from pathlib import Path
from unittest import mock

from agenty_core.utils import turn_scope

from pipeline_stub import pipeline_stub, tools
from src import executor, pipeline
from test_canvas_edit import _graph

ROOT = Path(__file__).resolve().parents[1]


class RunningTheOpenGraph(unittest.TestCase):

    def setUp(self):
        token = turn_scope.enter(turn_scope.Scope("req", "thread"))
        self.addCleanup(turn_scope.leave, token)
        self.seen = {}

        async def _spy(wf, *a, **kw):
            self.seen["path"] = str(wf)
            self.seen["graph"] = json.loads(Path(wf).read_text(encoding="utf-8"))
            kw["collected_paths"].append("W:/out/1.png")
            yield "done"

        self.enterContext(mock.patch("src.executor.execute_workflow", _spy))

    def _run(self, pipe, **kw):
        return json.loads(asyncio.run(tools(pipe)["run_canvas"](**kw)))

    def test_it_runs_the_graph_exactly_as_it_is_on_the_canvas(self):
        pipe = pipeline_stub(_canvas_graph=_graph())
        out = self._run(pipe)
        self.assertEqual(out.get("status"), "done", out)
        self.assertEqual(out["outputs"], ["W:/out/1.png"])
        self.assertEqual(self.seen["graph"], _graph())

    def test_a_change_made_this_turn_is_in_what_runs(self):
        pipe = pipeline_stub(_canvas_graph=_graph())
        pipe._canvas_graph["5"]["inputs"]["steps"] = 33
        self._run(pipe)
        self.assertEqual(self.seen["graph"]["5"]["inputs"]["steps"], 33)

    def test_part_of_it_is_those_nodes_and_what_feeds_them(self):
        pipe = pipeline_stub(_canvas_graph=_graph())
        self._run(pipe, node_ids=["#6"])
        self.assertEqual(sorted(self.seen["graph"]), ["1", "2", "4", "5", "6"])

    def test_it_is_not_opened_in_a_tab_as_well(self):
        self._run(pipeline_stub(_canvas_graph=_graph()))
        self.assertTrue(executor._already_on_canvas(self.seen["path"]))

    def test_an_unknown_node_or_no_canvas_runs_nothing(self):
        pipe = pipeline_stub(_canvas_graph=_graph())
        self.assertIn("no node 99", self._run(pipe, node_ids=["99"])["error"])
        self.assertIn("no graph is open", self._run(pipeline_stub(_canvas_graph={}))["error"])
        self.assertEqual(self.seen, {})


class WhatTheAgentIsTold(unittest.TestCase):

    def setUp(self):
        self.prompt = (ROOT / "config" / "system_prompts" / "system_prompt.orchestrator.md").read_text(encoding="utf-8")
        self.src = inspect.getsource(pipeline)

    def test_run_the_open_workflow_means_run_canvas_and_nothing_first(self):
        section = self.prompt.split("### Running what is open", 1)[1].split("###", 1)[0]
        self.assertIn("run_canvas()", section)
        self.assertIn("Do not search", section)
        doc = self.src.split("async def run_canvas(", 1)[1].split('"""', 2)[1]
        self.assertIn("do not call", doc)
        self.assertIn("run_canvas", tools(pipeline_stub(_canvas_graph=_graph())))

    def test_in_the_open_graph_there_is_no_offer_to_graph_what_is_already_there(self):
        self.assertIn("if not _autoload() and not self._open_graph_mode():", self.src)
        self.assertIn("never offer to graph, load or show it", pipeline._OPEN_GRAPH_NOTE)
        insert = self.src.split("async def insert_workflow_into_canvas(", 1)[1].split("@_tool", 1)[0]
        self.assertIn("do NOT offer to graph", insert)

    def test_the_mode_note_names_run_canvas(self):
        self.assertIn("run_canvas()", pipeline._OPEN_GRAPH_NOTE)


if __name__ == "__main__":
    unittest.main()
