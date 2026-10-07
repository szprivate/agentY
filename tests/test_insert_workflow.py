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
        # whatever this machine's own settings say about working in the open graph
        self.enterContext(mock.patch("src.agent._load_settings", return_value={}))
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


class StayingInTheGraph(unittest.TestCase):
    """Once a workflow has been inserted, the conversation keeps working there."""

    def setUp(self):
        self.enterContext(mock.patch("src.utils.canvas_view.full_graph_visible", return_value=False))
        # whatever this machine's own settings say about working in the open graph
        self.enterContext(mock.patch("src.agent._load_settings", return_value={}))
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

    def _pipe(self, **over):
        from src.utils.models import AgentSession
        return pipeline_stub(_canvas_graph=_graph(), _canvas_selection=[],
                             _session=AgentSession(session_id="thread"), **over)

    def _call(self, pipe, name, **kw):
        return json.loads(asyncio.run(tools(pipe)[name](**kw)))

    def test_inserting_switches_the_conversation_over_and_it_is_saved_with_it(self):
        from src.utils.models import AgentSession
        pipe = self._pipe()
        self.assertFalse(pipe._open_graph_mode())
        self._call(pipe, "insert_workflow_into_canvas", workflow_path=str(self.path))
        self.assertTrue(pipe._open_graph_mode())
        restored = AgentSession(**pipe._session.model_dump())      # what a restart reads back
        self.assertTrue(restored.open_graph_mode)

    def test_in_that_mode_it_may_change_nodes_nobody_selected(self):
        """What it inserted two turns ago is not selected now."""
        pipe = self._pipe()
        refused = self._call(pipe, "delete_canvas_nodes", node_ids=["7"])
        self.assertIn("not in the current", refused.get("error", ""))
        pipe._session.open_graph_mode = True
        self.assertTrue(pipe._canvas_full_graph())

    def test_the_user_can_switch_it_off_and_on(self):
        pipe = self._pipe()
        pipe._session.open_graph_mode = True
        self.assertFalse(self._call(pipe, "work_in_open_graph", on=False)["open_graph_mode"])
        self.assertFalse(pipe._open_graph_mode())
        self.assertTrue(self._call(pipe, "work_in_open_graph", on=True)["open_graph_mode"])
        self.assertTrue(pipe._open_graph_mode())

    def test_every_turn_of_such_a_conversation_is_told(self):
        import inspect
        from src import pipeline
        src = inspect.getsource(pipeline)
        told = src.split("if self._open_graph_mode():", 1)[1][:80]
        self.assertIn("pin = _OPEN_GRAPH_NOTE", told)
        for word in ("insert_workflow_into_canvas", "set_canvas_node_mode", "delete_canvas_nodes",
                     "edit_canvas_graph", "set_canvas_node_params", "work_in_open_graph(on=false)"):
            self.assertIn(word, pipeline._OPEN_GRAPH_NOTE)

    def test_a_new_conversation_starts_the_way_the_setting_says(self):
        from src.utils.models import AgentSession
        self.assertIsNone(AgentSession(session_id="x").open_graph_mode, "not decided yet")
        pipe = self._pipe()
        with mock.patch("src.agent._load_settings", return_value={}):
            self.assertFalse(pipe._open_graph_mode())
        with mock.patch("src.agent._load_settings", return_value={"work_in_open_graph": True}):
            self.assertTrue(pipe._open_graph_mode())

    def test_what_the_conversation_decided_beats_the_setting(self):
        pipe = self._pipe()
        with mock.patch("src.agent._load_settings", return_value={"work_in_open_graph": True}):
            self._call(pipe, "work_in_open_graph", on=False)
            self.assertFalse(pipe._open_graph_mode())
        pipe2 = self._pipe()
        pipe2._session.open_graph_mode = True
        with mock.patch("src.agent._load_settings", return_value={"work_in_open_graph": False}):
            self.assertTrue(pipe2._open_graph_mode())

    def test_the_setting_ships_off_and_is_in_the_defaults(self):
        text = (Path(__file__).resolve().parents[1] / "config" / "settings.default.toml").read_text(encoding="utf-8")
        self.assertIn("work_in_open_graph = false", text)


class RunningWhatIsOnTheCanvas(unittest.TestCase):
    """An inserted workflow runs as it is on the canvas NOW, not as the file was.

    The run this is for: three models inserted, all rendered square; the agent set
    16:9 on the canvas nodes, re-ran the three workflow FILES, got square again,
    and concluded the models ignore the setting. The files had never been told.
    """

    def setUp(self):
        self.enterContext(mock.patch("src.utils.canvas_view.full_graph_visible", return_value=False))
        # whatever this machine's own settings say about working in the open graph
        self.enterContext(mock.patch("src.agent._load_settings", return_value={}))
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
        from src.utils.models import AgentSession
        self.pipe = pipeline_stub(_canvas_graph=_graph(), _canvas_selection=[],
                                  _session=AgentSession(session_id="thread"))
        out = json.loads(asyncio.run(tools(self.pipe)["insert_workflow_into_canvas"](
            workflow_path=str(self.path))))
        self.ids = out["node_ids"]
        self.sampler = next(i for i, c in self.ids.items() if c == "KSampler")

    def _version(self):
        path, from_canvas = self.pipe._canvas_version_of(str(self.path))
        return json.loads(Path(path).read_text(encoding="utf-8")), from_canvas, path

    def test_a_value_changed_on_the_canvas_is_what_runs(self):
        self.pipe._canvas_graph[self.sampler]["inputs"]["steps"] = 33        # set on the canvas
        graph, from_canvas, path = self._version()
        self.assertTrue(from_canvas)
        self.assertNotEqual(path, str(self.path), "the built file itself is left as it was")
        self.assertEqual(graph[self.sampler]["inputs"]["steps"], 33)
        self.assertEqual(json.loads(self.path.read_text(encoding="utf-8"))["13"]["inputs"]["steps"], 20)

    def test_only_that_workflow_runs_not_the_users_own_nodes(self):
        graph, _from_canvas, _path = self._version()
        self.assertEqual(sorted(graph), sorted(self.ids))
        self.assertTrue(set(graph).isdisjoint(_graph()))

    def test_what_was_wired_in_front_of_it_since_comes_along(self):
        """A node the user (or the agent) put upstream is part of what runs."""
        loader = next(i for i, c in self.ids.items() if c == "CheckpointLoaderSimple")
        self.pipe._canvas_graph[self.sampler]["inputs"]["model"] = ["1", 0]   # the user's own loader
        graph, _from_canvas, _path = self._version()
        self.assertIn("1", graph)
        self.assertIn(loader, graph, "still feeds the text encoder and the decode")

    def test_running_it_does_not_open_a_tab_for_the_canvas_version_either(self):
        from src import executor
        _graph_, _from_canvas, path = self._version()
        self.assertTrue(executor._already_on_canvas(path))

    def test_it_is_remembered_with_the_conversation(self):
        from src.utils.models import AgentSession
        restored = AgentSession(**self.pipe._session.model_dump())
        self.assertEqual(sorted(restored.inserted_workflows[str(self.path.resolve())]), sorted(self.ids))

    def test_a_workflow_that_was_never_inserted_runs_as_its_file(self):
        other = self.path.with_name("other.json")
        other.write_text(json.dumps(BUILT), encoding="utf-8")
        self.assertEqual(self.pipe._canvas_version_of(str(other)), (str(other), False))

    def test_without_a_canvas_this_turn_the_file_runs(self):
        self.pipe._canvas_graph = {}
        self.assertEqual(self.pipe._canvas_version_of(str(self.path)), (str(self.path), False))

    def test_nodes_that_are_gone_from_the_graph_fall_back_to_the_file(self):
        for nid in self.ids:
            self.pipe._canvas_graph.pop(nid, None)
        self.assertEqual(self.pipe._canvas_version_of(str(self.path)), (str(self.path), False))

    def test_run_workflow_now_goes_through_it(self):
        import inspect
        from src import pipeline
        body = inspect.getsource(pipeline).split("async def run_workflow_now(", 1)[1].split("@_tool", 1)[0]
        self.assertIn("workflow_path, _from_canvas = self._canvas_version_of(workflow_path)", body)
        self.assertLess(body.index("_canvas_version_of"), body.index("_execute_workflow("))


class Subgraph(unittest.TestCase):

    def test_the_named_nodes_and_everything_upstream(self):
        got = ce.subgraph(_graph(), ["6"])
        self.assertEqual(sorted(got), ["1", "2", "4", "5", "6"])
        self.assertNotIn("7", got, "downstream of it is not needed for it to run")

    def test_unknown_ids_are_left_out_and_nothing_is_an_empty_graph(self):
        self.assertEqual(sorted(ce.subgraph(_graph(), ["4", "99"])), ["4"])
        self.assertEqual(ce.subgraph(_graph(), ["99"]), {})
        self.assertEqual(ce.subgraph(None, ["1"]), {})

    def test_it_is_a_copy(self):
        graph = _graph()
        ce.subgraph(graph, ["5"])["5"]["inputs"]["steps"] = 1
        self.assertEqual(graph["5"]["inputs"]["steps"], 20)


class BypassAndMute(unittest.TestCase):

    def setUp(self):
        self.enterContext(mock.patch("src.utils.canvas_view.full_graph_visible", return_value=True))
        self.enterContext(mock.patch("src.agent._load_settings", return_value={}))
        from src.utils.canvas_patch import clear
        clear()
        self.addCleanup(clear)

    def _call(self, pipe, **kw):
        return json.loads(asyncio.run(tools(pipe)["set_canvas_node_mode"](**kw)))

    def test_it_reaches_the_canvas(self):
        from src.utils.canvas_patch import drain
        pipe = pipeline_stub(_canvas_graph=_graph(), _canvas_selection=[], _canvas_edits=[])
        out = self._call(pipe, node_ids=["6", "#7"], mode="bypass", reason="skip the decode")
        self.assertEqual(out["status"], "applied")
        patch = next(e for e in drain() if e.get("op") == "set_mode")
        self.assertEqual((patch["node_ids"], patch["mode"]), (["6", "7"], "bypass"))

    def test_a_node_that_is_not_there_cannot_be_switched_off(self):
        pipe = pipeline_stub(_canvas_graph=_graph(), _canvas_selection=[], _canvas_edits=[])
        self.assertIn("no node 99", self._call(pipe, node_ids=["99"], mode="mute")["error"])

    def test_one_that_was_switched_off_can_be_switched_back_on(self):
        """A bypassed node is not in the graph that would run, so it is not in
        the turn's copy either - re-enabling it must not be refused for that."""
        pipe = pipeline_stub(_canvas_graph=_graph(), _canvas_selection=[], _canvas_edits=[])
        self.assertEqual(self._call(pipe, node_ids=["99"], mode="active")["status"], "applied")

    def test_an_unknown_mode_is_refused(self):
        pipe = pipeline_stub(_canvas_graph=_graph(), _canvas_selection=[], _canvas_edits=[])
        self.assertIn("unknown mode", self._call(pipe, node_ids=["6"], mode="off")["error"])

    def test_selection_only_still_applies_outside_the_mode(self):
        with mock.patch("src.utils.canvas_view.full_graph_visible", return_value=False):
            pipe = pipeline_stub(_canvas_graph=_graph(), _canvas_selection=[], _canvas_edits=[])
            self.assertIn("not in the current canvas selection",
                          self._call(pipe, node_ids=["6"], mode="bypass")["error"])


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
