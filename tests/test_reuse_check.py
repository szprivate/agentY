"""A run that would make a pass already on disk is held once; asked again, it runs.

Pinned on real files in a scratch ComfyUI layout: a depth pass of plate.mp4
(frames 0-96) exists. Running the same graph is held with a message naming the
file; the same call again goes through; a different frame range, a render and a
UI-format file are never held; the switch turns it off. Also the host's
/agentY/deps route and the file_dependencies tool.

    PYTHONPATH=.:tests python -m unittest tests.test_reuse_check
"""

import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path

from agenty_core.utils import dependencies as dp
from agenty_core.utils import provenance as pv
from src.utils import reuse_check as rc


def depth_graph(frames=97):
    return {
        "1": {"class_type": "VHS_LoadVideo", "inputs": {"video": "plate.mp4", "frame_load_cap": frames}},
        "2": {"class_type": "DepthAnythingV2Preprocessor",
              "inputs": {"image": ["1", 0], "ckpt_name": "depth_anything_v2_vitl.pth"}},
        "3": {"class_type": "VHS_VideoCombine", "inputs": {"images": ["2", 0], "filename_prefix": "depth"}},
    }


class _Event:
    def __init__(self, name, inp):
        self.tool_use = {"name": name, "input": inp}
        self.cancel_tool = False


class ReuseCheck(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.inp, self.out = self.tmp / "input", self.tmp / "output"
        self.inp.mkdir()
        self.out.mkdir()
        dp.set_roots({"input": str(self.inp), "output": str(self.out)})
        self.plate = self.inp / "plate.mp4"
        self.plate.write_bytes(b"plate" * 999)
        self.depth = self.out / "depth_00001.mp4"
        self.depth.write_bytes(b"depth" * 99)
        pv.write(self.depth, depth_graph(), output_node="3")
        os.environ.pop("AGENTY_REUSE_CHECK", None)

    def tearDown(self):
        dp.set_roots(None)
        os.environ.pop("AGENTY_REUSE_CHECK", None)
        shutil.rmtree(self.tmp, ignore_errors=True)

    def wf(self, graph, name="wf.json"):
        p = self.tmp / name
        p.write_text(json.dumps(graph))
        return str(p)

    def test_held_once_then_runs(self):
        hook = rc.ReuseCheckHook()
        hook._on_start()
        ev = _Event("signal_workflow_ready", {"workflow_path": self.wf(depth_graph())})
        hook._on_before(ev)
        self.assertIsInstance(ev.cancel_tool, str)
        self.assertIn("depth of plate.mp4 frames 0-96 already exists", ev.cancel_tool)
        self.assertIn(str(self.depth.resolve()), ev.cancel_tool)
        again = _Event("signal_workflow_ready", dict(ev.tool_use["input"]))
        hook._on_before(again)
        self.assertFalse(again.cancel_tool)

    def test_not_held_when_nothing_matches(self):
        hook = rc.ReuseCheckHook()
        for ev in (_Event("run_workflow_now", {"workflow_path": self.wf(depth_graph(frames=49), "w49.json")}),
                   _Event("run_workflow_now", {"workflow_path": self.wf({"nodes": [], "links": []}, "ui.json")}),
                   _Event("edit_canvas_graph", {"workflow_path": self.wf(depth_graph(), "other.json")})):
            hook._on_before(ev)
            self.assertFalse(ev.cancel_tool, ev.tool_use)

    def test_switched_off(self):
        os.environ["AGENTY_REUSE_CHECK"] = "0"
        ev = _Event("signal_workflow_ready", {"workflow_path": self.wf(depth_graph())})
        rc.ReuseCheckHook()._on_before(ev)
        self.assertFalse(ev.cancel_tool)

    def test_file_dependencies_tool(self):
        from src.tools.file_dependencies import file_dependencies
        fn = getattr(file_dependencies, "_tool_func", None) or getattr(file_dependencies, "original_function", None) \
            or file_dependencies
        got = json.loads(fn(str(self.plate)))
        self.assertEqual(got["status"], "ok")
        self.assertEqual([m["kind"] for m in got["made_from_this"]], ["depth"])
        got = json.loads(fn(str(self.depth)))
        self.assertEqual(got["kind"], "depth")
        self.assertEqual([m["file"] for m in got["made_from"]], ["plate.mp4"])
        self.assertEqual(got["stale"], "")

    def test_deps_route(self):
        from route_client import authorised_client
        c = authorised_client()
        r = c.get("/agentY/deps", query_string={"path": str(self.depth)})
        self.assertEqual(r.status_code, 200, r.get_data(as_text=True))
        body = r.get_json()
        self.assertEqual({n["file"] for n in body["nodes"]}, {"plate.mp4", "depth_00001.mp4"})
        self.assertEqual(len(body["edges"]), 1)
        self.assertEqual(c.get("/agentY/deps").status_code, 400)
        self.assertEqual(c.get("/agentY/deps", query_string={"path": str(self.tmp / "nope.mp4")}).status_code, 404)


if __name__ == "__main__":
    unittest.main()
