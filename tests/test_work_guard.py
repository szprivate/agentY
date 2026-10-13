"""What the validator says is acted on: at the hand-off, at the end of the turn, in the canvas tool.

It ran after every edit and said what was wrong, and that was all. A small
orchestrator (qwen3.5:9b) read "error" seven times and ended its turn; read
"applied ... some inputs are still unwired" and ended its turn.
"""

import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from pipeline_stub import pipeline_stub, tools
from src.utils import canvas_edit as ce
from src.utils import work_guard as wg

INFO = {
    "CheckpointLoaderSimple": {"input": {"required": {"ckpt_name": [["a.safetensors"]]}},
                               "output": ["MODEL", "CLIP", "VAE"]},
    "LoadImage": {"input": {"required": {"image": [["x.png"]]}}, "output": ["IMAGE", "MASK"]},
    "VAEEncode": {"input": {"required": {"pixels": ["IMAGE"], "vae": ["VAE"]}}, "output": ["LATENT"]},
    "ControlNetApply": {"input": {"required": {"control_net": ["CONTROL_NET"]}}, "output": ["CONDITIONING"]},
}
GRAPH = {"4": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": "a.safetensors"}},
         "14": {"class_type": "LoadImage", "inputs": {"image": "x.png"}},
         "19": {"class_type": "VAEEncode", "inputs": {}},
         "20": {"class_type": "ControlNetApply", "inputs": {}}}


def _use(tid, name):
    return {"role": "assistant", "content": [{"toolUse": {"toolUseId": tid, "name": name, "input": {}}}]}


def _res(tid, data):
    return {"role": "user", "content": [{"toolResult": {"toolUseId": tid, "content": [{"text": json.dumps(data)}]}}]}


class WhereAWireCouldComeFrom(unittest.TestCase):

    def test_the_outputs_in_the_graph_that_carry_the_type(self):
        got = ce.wire_candidates(GRAPH, "19", INFO)
        self.assertEqual(got["vae (VAE)"], ["node 4 (CheckpointLoaderSimple) output 2"])
        self.assertEqual(got["pixels (IMAGE)"], ["node 14 (LoadImage) output 0"])

    def test_nothing_where_the_graph_makes_no_such_type(self):
        self.assertEqual(ce.wire_candidates(GRAPH, "20", INFO), {})      # a node has to be added first


class TheLastEditOfTheTurn(unittest.TestCase):

    def test_an_edit_that_did_not_validate(self):
        msgs = [_use("a", "update_workflow"), _res("a", {"status": "ok", "workflow_path": "w.json"}),
                _use("b", "get_node_schema"), _res("b", {"status": "error"}),
                _use("c", "update_workflow"),
                _res("c", {"status": "error", "workflow_path": "w.json",
                           "local_errors": ["Node 19 (VAEEncode): required input 'vae' is unconnected (expects a VAE link)."],
                           "node_errors": ["patch 0: node '99' not found."]})]
        got = wg.last_file_edit(msgs)
        self.assertFalse(got["ok"])
        self.assertEqual(got["workflow_path"], "w.json")
        self.assertEqual(len(got["errors"]), 2)

    def test_the_last_one_counts(self):
        msgs = [_use("a", "update_workflow"), _res("a", {"status": "error", "workflow_path": "w.json",
                                                         "local_errors": ["x"]}),
                _use("b", "update_workflow"), _res("b", {"status": "ok", "workflow_path": "w.json"})]
        self.assertTrue(wg.last_file_edit(msgs)["ok"])

    def test_only_this_turns(self):
        msgs = [_use("a", "update_workflow"), _res("a", {"status": "error", "workflow_path": "w.json",
                                                         "local_errors": ["x"]}),
                {"role": "user", "content": [{"text": "thanks"}]}]
        self.assertIsNone(wg.last_file_edit(msgs, start=2))
        self.assertIsNone(wg.last_file_edit([], 0))

    def test_the_note_names_each_input_and_where_it_could_come_from(self):
        note = wg.graph_note({"19": ["vae (VAE)", "pixels (IMAGE)"]},
                             {"19": {"vae (VAE)": ["node 4 (CheckpointLoaderSimple) output 2"]}})
        self.assertIn("node 19: `vae (VAE)` has nothing wired into it - could come from node 4", note)
        self.assertIn("node 19: `pixels (IMAGE)` has nothing wired into it", note)


class AtTheEndOfTheTurn(unittest.TestCase):

    def _pipe(self, **over):
        agent = SimpleNamespace(messages=over.pop("messages", []))
        base = dict(_orchestrator_agent=agent, _canvas_unwired={}, _work_repaired=None)
        base.update(over)
        return pipeline_stub(**base)

    def _ask(self, pipe, done=0, start=0):
        with mock.patch("src.utils.settings.load_settings", return_value={}), \
                mock.patch("src.utils.workflow_signal.peek", return_value=getattr(pipe, "_signalled", [])):
            return asyncio.run(pipe._unfinished_work_input(done, start))

    def test_a_turn_that_left_nothing_broken_ends(self):
        self.assertEqual(self._ask(self._pipe()), "")

    def test_unfed_inputs_on_the_canvas_send_it_back_with_the_list(self):
        pipe = self._pipe(_canvas_unwired={"19": ["vae (VAE)"]}, _canvas_graph=GRAPH)
        with mock.patch("src.pipeline._CanvasSchemas", return_value=INFO):
            note = self._ask(pipe)
        self.assertIn("node 19: `vae (VAE)`", note)
        self.assertIn("node 4 (CheckpointLoaderSimple) output 2", note)
        self.assertIn("edit_canvas_graph", note)

    def test_not_more_often_than_the_limit(self):
        pipe = self._pipe(_canvas_unwired={"19": ["vae (VAE)"]}, _canvas_graph=GRAPH)
        self.assertEqual(self._ask(pipe, done=wg.MAX_REDRIVES), "")

    def test_a_file_that_did_not_validate_goes_to_the_repair_first(self):
        msgs = [_use("c", "update_workflow"),
                _res("c", {"status": "error", "workflow_path": "w.json", "local_errors": ["Node 19: unconnected"]})]
        pipe = self._pipe(messages=msgs)
        calls = []

        async def repair(path, problems=None, **kw):
            calls.append((path, problems))
            return {"status": "ready", "workflow_path": path}
        pipe._run_fix_workflow_assembly = repair
        note = self._ask(pipe)
        self.assertEqual(calls, [("w.json", ["Node 19: unconnected"])])
        self.assertIn("validates now", note)
        self.assertIn("signal_workflow_ready", note)
        self.assertEqual(self._ask(pipe), "")                 # once per file

    def test_and_says_so_when_the_repair_could_not(self):
        msgs = [_use("c", "update_workflow"),
                _res("c", {"status": "error", "workflow_path": "w.json", "local_errors": ["Node 19: unconnected"]})]
        pipe = self._pipe(messages=msgs)

        async def repair(path, problems=None, **kw):
            return {"status": "needs_fix", "workflow_path": path, "problems": ["still: vae"]}
        pipe._run_fix_workflow_assembly = repair
        note = self._ask(pipe)
        self.assertIn("could not clear", note)
        self.assertIn("- still: vae", note)

    def test_a_workflow_that_was_handed_off_is_left_alone(self):
        msgs = [_use("c", "update_workflow"),
                _res("c", {"status": "error", "workflow_path": "w.json", "local_errors": ["x"]})]
        pipe = self._pipe(messages=msgs)
        pipe._signalled = ["other.json"]
        self.assertEqual(self._ask(pipe), "")


class AtTheHandOff(unittest.TestCase):

    def setUp(self):
        from agenty_core.utils import turn_scope
        from src.utils import workflow_signal as ws
        self.ws = ws
        ws.consume() if hasattr(ws, "consume") else None
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / "w.json"
        self.path.write_text(json.dumps({"1": {"class_type": "VAEEncode", "inputs": {}}}), encoding="utf-8")

    def _signal(self, errors):
        from src.tools import workflow_handoff as wh
        fn = getattr(wh.signal_workflow_ready, "func", None) or wh.signal_workflow_ready._tool_func
        with mock.patch.object(wh, "_validation_errors", return_value=errors), \
                mock.patch.object(wh, "_dead_nodes", return_value=([], [])), \
                mock.patch("src.utils.workflow_signal.execution_hold", return_value=None):
            return json.loads(fn(str(self.path)))

    def test_a_workflow_that_does_not_validate_is_handed_back_with_the_list(self):
        got = self._signal(["Node 1 (VAEEncode): required input 'vae' is unconnected (expects a VAE link)."])
        self.assertEqual(got["status"], "not_ready")
        self.assertIn("unconnected", got["problems"][0])
        self.assertNotIn(str(self.path.resolve()), self.ws.peek())

    def test_twice_and_then_it_goes_through(self):
        errors = ["Node 1 (VAEEncode): required input 'vae' is unconnected."]
        self.assertEqual(self._signal(errors)["status"], "not_ready")
        self.assertEqual(self._signal(errors)["status"], "not_ready")
        self.assertIn(self._signal(errors)["status"], ("ready", "dry_run"))
        self.assertIn(str(self.path.resolve()), self.ws.peek())

    def test_one_that_validates_is_queued_as_before(self):
        self.assertIn(self._signal([])["status"], ("ready", "dry_run"))


class TheCanvasTool(unittest.TestCase):

    def test_an_edit_that_leaves_inputs_unfed_says_so_in_its_status(self):
        pipe = pipeline_stub(_canvas_graph=dict(GRAPH), _canvas_selection=[], _canvas_edits=[],
                             _canvas_touched=set(), _canvas_unwired={})
        pipe._canvas_full_graph = lambda: True
        with mock.patch("src.pipeline._CanvasSchemas", return_value=INFO):
            out = json.loads(asyncio.run(tools(pipe)["edit_canvas_graph"](
                ops=[{"op": "add", "class_type": "VAEEncode", "ref": "e"}], reason="img2img")))
            self.assertEqual(out["status"], "applied_incomplete")
            new = out["added"]["e"]
            self.assertIn("vae (VAE)", out["still_unwired"][new])
            self.assertEqual(out["wire_from"][new]["vae (VAE)"], ["node 4 (CheckpointLoaderSimple) output 2"])
            self.assertIn("NOT finished", out["message"])
            self.assertIn(new, pipe._canvas_unwired)
            # ... and the edit that wires them clears the turn's list.
            out = json.loads(asyncio.run(tools(pipe)["edit_canvas_graph"](
                ops=[{"op": "connect", "from": "4", "output": 2, "to": new, "input": "vae"},
                     {"op": "connect", "from": "14", "output": 0, "to": new, "input": "pixels"}])))
        self.assertEqual(out.get("status"), "applied", out)
        self.assertNotIn(new, pipe._canvas_unwired)


if __name__ == "__main__":
    unittest.main()
