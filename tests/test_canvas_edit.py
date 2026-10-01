"""Adding nodes and rewiring the open graph — the edit agentY could not make.

It could set values and delete nodes, so "add a hires-fix pass" or "preview
instead of saving" ended in click-by-click instructions for the user, a menu of
options, or a request for a file path. Benchmarked, three of fifteen first-attempt
failures were exactly that. ``edit_canvas_graph`` adds nodes and changes wires,
checked against the classes' real inputs and outputs, all or nothing.

And the turn's copy of the canvas now follows every edit pushed in the turn:
``get_canvas_node`` used to read the start-of-turn snapshot, so the agent saw its
own write as reverted, wrote it again, and finally told the user (and its memory)
that canvas edits don't stick.

    python -m unittest discover -s tests
"""

import asyncio
import json
import unittest
from unittest import mock

from pipeline_stub import pipeline_stub, tools
from src.utils import canvas_edit as ce

SCHEMAS = {
    "CheckpointLoaderSimple": {"input": {"required": {"ckpt_name": [["a.safetensors"], {}]}},
                               "output": ["MODEL", "CLIP", "VAE"], "output_name": ["MODEL", "CLIP", "VAE"]},
    "KSampler": {"input": {"required": {
        "model": ["MODEL"], "positive": ["CONDITIONING"], "negative": ["CONDITIONING"],
        "latent_image": ["LATENT"], "seed": ["INT", {"default": 0}], "steps": ["INT", {"default": 20}],
        "denoise": ["FLOAT", {"default": 1.0}], "sampler_name": [["euler", "dpmpp_2m"], {}]}},
        "output": ["LATENT"], "output_name": ["LATENT"]},
    "LatentUpscaleBy": {"input": {"required": {"samples": ["LATENT"], "scale_by": ["FLOAT", {"default": 1.5}],
                                               "upscale_method": [["nearest-exact", "bilinear"], {}]}},
                        "output": ["LATENT"], "output_name": ["LATENT"]},
    "VAEDecode": {"input": {"required": {"samples": ["LATENT"], "vae": ["VAE"]}},
                  "output": ["IMAGE"], "output_name": ["IMAGE"]},
    "SaveImage": {"input": {"required": {"images": ["IMAGE"], "filename_prefix": ["STRING", {"default": "x"}]}},
                  "output": [], "output_node": True},
    "PreviewImage": {"input": {"required": {"images": ["IMAGE"]}}, "output": [], "output_node": True},
    "EmptyLatentImage": {"input": {"required": {"width": ["INT", {"default": 512}]}},
                         "output": ["LATENT"], "output_name": ["LATENT"]},
    "CLIPTextEncode": {"input": {"required": {"text": ["STRING", {}], "clip": ["CLIP"]}},
                       "output": ["CONDITIONING"], "output_name": ["CONDITIONING"]},
}


def _graph():
    return {
        "1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": "a.safetensors"}},
        "2": {"class_type": "CLIPTextEncode", "inputs": {"text": "a cat", "clip": ["1", 1]}},
        "4": {"class_type": "EmptyLatentImage", "inputs": {"width": 512}},
        "5": {"class_type": "KSampler", "inputs": {"model": ["1", 0], "positive": ["2", 0], "negative": ["2", 0],
                                                   "latent_image": ["4", 0], "seed": 1, "steps": 20, "denoise": 1.0, "sampler_name": "euler"}},
        "6": {"class_type": "VAEDecode", "inputs": {"samples": ["5", 0], "vae": ["1", 2]}},
        "7": {"class_type": "SaveImage", "inputs": {"images": ["6", 0], "filename_prefix": "out"}},
    }


HIRES = [
    {"op": "add", "class_type": "LatentUpscaleBy", "ref": "up", "params": {"scale_by": 1.5}, "near": "5"},
    {"op": "add", "class_type": "KSampler", "ref": "ks2", "params": {"denoise": 0.5}},
    {"op": "connect", "from": "5", "output": "LATENT", "to": "up", "input": "samples"},
    {"op": "connect", "from": "up", "to": "ks2", "input": "latent_image"},
    {"op": "connect", "from": "1", "output": "MODEL", "to": "ks2", "input": "model"},
    {"op": "connect", "from": "2", "to": "ks2", "input": "positive"},
    {"op": "connect", "from": "2", "to": "ks2", "input": "negative"},
    {"op": "connect", "from": "ks2", "to": "6", "input": "samples"},
]


class PlanTest(unittest.TestCase):

    def test_a_hires_pass_is_added_and_wired(self):
        r = ce.plan(_graph(), HIRES, SCHEMAS)
        self.assertTrue(r["ok"], r["errors"])
        up, ks2 = r["added"]["up"], r["added"]["ks2"]
        g = r["graph"]
        self.assertEqual(g[up]["inputs"]["samples"], ["5", 0])
        self.assertEqual(g[ks2]["inputs"]["latent_image"], [up, 0])
        self.assertEqual(g["6"]["inputs"]["samples"], [ks2, 0])
        self.assertEqual(g[ks2]["inputs"]["denoise"], 0.5)
        self.assertEqual(g[ks2]["inputs"]["steps"], 20, "unset widgets keep the class default")
        self.assertEqual(ce.unwired_required(g, ks2, SCHEMAS), [])

    def test_new_ids_follow_the_highest_on_the_graph(self):
        r = ce.plan(_graph(), HIRES[:2], SCHEMAS)
        self.assertEqual(sorted(r["added"].values()), ["8", "9"])

    def test_one_bad_op_changes_nothing(self):
        g = _graph()
        r = ce.plan(g, [HIRES[0], {"op": "connect", "from": "1", "output": 0, "to": "6", "input": "samples"}], SCHEMAS)
        self.assertFalse(r["ok"])
        self.assertIn("is MODEL, but #6.samples takes LATENT", r["errors"][0])
        self.assertEqual(r["graph"], g)
        self.assertEqual(r["ops"], [])

    def test_the_errors_say_what_to_use_instead(self):
        r = ce.plan(_graph(), [
            {"op": "add", "class_type": "NotANode"},
            {"op": "add", "class_type": "KSampler", "params": {"sampler_name": "dpmpp"}},
            {"op": "add", "class_type": "KSampler", "params": {"colour": 1}},
            {"op": "connect", "from": "5", "output": 3, "to": "6", "input": "samples"},
            {"op": "disconnect", "node": "7", "input": "filename_prefix"},
            {"op": "rewire"}], SCHEMAS)
        text = "\n".join(r["errors"])
        self.assertIn("'NotANode' is not installed", text)
        self.assertIn("close: ['dpmpp_2m']", text)
        self.assertIn("has no input 'colour'", text)
        self.assertIn("outputs: 0=LATENT(LATENT)", text)
        self.assertIn("is not wired", text)
        self.assertIn("use add, connect or disconnect", text)

    def test_a_swap_is_add_connect_disconnect(self):
        r = ce.plan(_graph(), [{"op": "add", "class_type": "PreviewImage", "ref": "pv"},
                               {"op": "connect", "from": "6", "to": "pv", "input": "images"},
                               {"op": "disconnect", "node": "7", "input": "images"}], SCHEMAS)
        self.assertTrue(r["ok"], r["errors"])
        self.assertNotIn("images", r["graph"]["7"]["inputs"])

    def test_replaying_the_patch_gives_the_same_graph(self):
        """The panel and a headless client replay the pushed ops; they must land
        where the plan said."""
        r = ce.plan(_graph(), HIRES, SCHEMAS)
        self.assertEqual(ce.apply_patch(_graph(), {"op": "edit_graph", "ops": r["ops"]}), r["graph"])

    def test_v3_dynamic_members_can_be_wired(self):
        schemas = dict(SCHEMAS, ComfyMathExpression={
            "input": {"required": {"expression": ["STRING", {}],
                                   "values": ["COMFY_AUTOGROW_V3", {"names": ["a", "b"]}]}},
            "output": ["FLOAT"], "output_name": ["FLOAT"]})
        r = ce.plan(_graph(), [{"op": "add", "class_type": "ComfyMathExpression", "ref": "m"},
                               {"op": "connect", "from": "5", "to": "m", "input": "values.a"}], schemas)
        self.assertTrue(r["ok"], r["errors"])


class ToolTest(unittest.TestCase):

    def setUp(self):
        self.enterContext(mock.patch("src.utils.canvas_view.full_graph_visible", return_value=True))
        self.enterContext(mock.patch("src.utils.preflight._schema", side_effect=lambda c: SCHEMAS.get(c, {})))
        from src.utils.canvas_patch import clear
        clear()
        self.addCleanup(clear)

    def _pipe(self):
        return pipeline_stub(_canvas_graph=_graph(), _canvas_selection=[])

    def _call(self, pipe, name, **kw):
        return json.loads(asyncio.run(tools(pipe)[name](**kw)))

    def test_it_reaches_the_canvas_and_the_turns_copy(self):
        from src.utils.canvas_patch import drain
        pipe = self._pipe()
        out = self._call(pipe, "edit_canvas_graph", ops=HIRES, reason="hires fix")
        self.assertEqual(out["status"], "applied")
        self.assertNotIn("still_unwired", out)
        patch = next(e for e in drain() if e.get("op") == "edit_graph")
        self.assertEqual(len(patch["ops"]), len(HIRES))
        ks2 = out["added"]["ks2"]
        node = self._call(pipe, "get_canvas_node", node_id="6")
        self.assertEqual(node["wired_inputs"]["samples"], f"from #{ks2} output 0")

    def test_an_unfinished_edit_says_what_is_still_unwired(self):
        out = self._call(self._pipe(), "edit_canvas_graph",
                         ops=[{"op": "add", "class_type": "KSampler", "ref": "k"}])
        self.assertIn("model (MODEL)", out["still_unwired"][out["added"]["k"]])
        self.assertIn("will not run", out["message"])

    def test_a_rejected_edit_changes_nothing(self):
        from src.utils.canvas_patch import drain
        pipe = self._pipe()
        out = self._call(pipe, "edit_canvas_graph", ops=[{"op": "add", "class_type": "NotANode"}])
        self.assertEqual(out["status"], "rejected")
        self.assertEqual(pipe._canvas_graph, _graph())
        self.assertFalse([e for e in drain() if e.get("op") == "edit_graph"])

    def test_a_value_written_this_turn_reads_back(self):
        """The bug that taught the agent edits don't stick."""
        pipe = self._pipe()
        self._call(pipe, "set_canvas_node_params", node_id="1", params={"ckpt_name": "b.safetensors"})
        self.assertEqual(self._call(pipe, "get_canvas_node", node_id="1")["values"]["ckpt_name"],
                         "b.safetensors")

    def test_a_deleted_node_is_gone_from_the_turns_copy(self):
        pipe = self._pipe()
        self._call(pipe, "delete_canvas_nodes", node_ids=["7"])
        self.assertNotIn("7", pipe._canvas_graph)


if __name__ == "__main__":
    unittest.main()
