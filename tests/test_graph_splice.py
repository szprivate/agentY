"""The common changes to a graph are worked out in code, not wired by a model.

Each operation is applied to a plain text-to-image graph through the same
``canvas_edit.plan`` the canvas tool uses, and the result must be a graph in
which every required input of every node is fed.
"""

import json
import unittest
from pathlib import Path

from src.utils import canvas_edit as ce
from src.utils import graph_splice as gs

INFO = json.loads((Path(__file__).parent / "fixtures_object_info_splice.json").read_text(encoding="utf-8"))
INFO["ControlNetLoader"]["input"]["required"]["control_net_name"] = [
    ["SD1.5\\control_v11p_sd15_canny.safetensors", "SD1.5\\control_v11f1p_sd15_depth.safetensors"]]
INFO["UpscaleModelLoader"]["input"]["required"]["model_name"] = [
    "COMBO", {"options": ["RealESRGAN_x4plus.safetensors", "4x-UltraSharp.pth"]}]
INFO["LoadImage"]["input"]["required"]["image"] = [["input.png", "pose.png"]]
INFO["LoraLoader"]["input"]["required"]["lora_name"] = [["detail.safetensors", "style\\ink.safetensors"]]


def t2i():
    return {
        "4": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": "CHORD\\chord_v1.safetensors"}},
        "5": {"class_type": "EmptyLatentImage", "inputs": {"width": 512, "height": 512, "batch_size": 1}},
        "6": {"class_type": "CLIPTextEncode", "inputs": {"text": "a cat", "clip": ["4", 1]}},
        "7": {"class_type": "CLIPTextEncode", "inputs": {"text": "blurry", "clip": ["4", 1]}},
        "3": {"class_type": "KSampler", "inputs": {"model": ["4", 0], "seed": 1, "steps": 20, "cfg": 7.0,
                                                    "sampler_name": "euler", "scheduler": "simple",
                                                    "positive": ["6", 0], "negative": ["7", 0],
                                                    "latent_image": ["5", 0], "denoise": 1.0}},
        "8": {"class_type": "VAEDecode", "inputs": {"samples": ["3", 0], "vae": ["4", 2]}},
        "9": {"class_type": "SaveImage", "inputs": {"images": ["8", 0], "filename_prefix": "out"}},
    }


def apply(graph, operation, params=None):
    """Splice, then apply the way the tool does: plan, set, delete."""
    got = gs.splice(graph, operation, params or {}, INFO)
    if not got["ok"]:
        return got, None
    res = ce.plan(graph, got["ops"], INFO)
    assert res["ok"], res["errors"]
    g = res["graph"]
    for nid, values in got["set"].items():
        g[nid]["inputs"].update(values)
    for nid in got["delete"]:
        g.pop(nid, None)
    return got, g


def unfed(graph):
    return {n: u for n in graph for u in [ce.unwired_required(graph, n, INFO)] if u}


def classes(graph):
    return sorted(n["class_type"] for n in graph.values())


class Finding(unittest.TestCase):

    def test_the_sampler_is_found_by_what_it_takes(self):
        self.assertEqual(gs.samplers(t2i(), INFO), ["3"])

    def test_a_second_pass_comes_after_the_first(self):
        _, g = apply(t2i(), "hires_fix")
        found = gs.samplers(g, INFO)
        self.assertEqual(found[0], "3")
        self.assertEqual(len(found), 2)


class ImageToImage(unittest.TestCase):

    def test_the_image_is_encoded_with_the_graphs_vae_and_feeds_the_sampler(self):
        got, g = apply(t2i(), "image_to_image", {"image": "input.png", "denoise": 0.6})
        self.assertEqual(unfed(g), {})
        enc = next(n for n, v in g.items() if v["class_type"] == "VAEEncode")
        self.assertEqual(g["3"]["inputs"]["latent_image"], [enc, 0])
        self.assertEqual(g[enc]["inputs"]["vae"], ["4", 2])
        self.assertEqual(g["3"]["inputs"]["denoise"], 0.6)
        self.assertNotIn("5", g)                       # the empty latent nothing reads any more
        self.assertIn("LoadImage", classes(g))

    def test_an_image_that_is_not_there_is_said_not_guessed(self):
        got, g = apply(t2i(), "image_to_image", {"image": "nowhere.png"})
        self.assertFalse(got["ok"])
        self.assertIn("upload", got["error"])


class ControlNet(unittest.TestCase):

    def test_both_conditionings_go_through_it(self):
        got, g = apply(t2i(), "controlnet", {"control_net_name": "canny", "image": "pose.png",
                                             "preprocessor": "canny", "strength": 0.8})
        self.assertEqual(unfed(g), {})
        cn = next(n for n, v in g.items() if v["class_type"] == "ControlNetApplyAdvanced")
        self.assertEqual(g["3"]["inputs"]["positive"], [cn, 0])
        self.assertEqual(g["3"]["inputs"]["negative"], [cn, 1])
        self.assertEqual(g[cn]["inputs"]["positive"], ["6", 0])
        self.assertEqual(g[cn]["inputs"]["strength"], 0.8)
        loader = next(v for v in g.values() if v["class_type"] == "ControlNetLoader")
        self.assertEqual(loader["inputs"]["control_net_name"], "SD1.5\\control_v11p_sd15_canny.safetensors")
        canny = next(n for n, v in g.items() if v["class_type"] == "Canny")
        self.assertEqual(g[cn]["inputs"]["image"], [canny, 0])

    def test_a_model_that_is_not_installed_is_said(self):
        got, _ = apply(t2i(), "controlnet", {"control_net_name": "openpose_xl", "image": "pose.png"})
        self.assertFalse(got["ok"])
        self.assertIn("ControlNet", got["error"])


class Patches(unittest.TestCase):

    def test_freeu_sits_between_the_model_and_the_sampler(self):
        got, g = apply(t2i(), "patch_model", {"class_type": "FreeU_V2", "b1": 1.1})
        self.assertEqual(unfed(g), {})
        fu = next(n for n, v in g.items() if v["class_type"] == "FreeU_V2")
        self.assertEqual(g[fu]["inputs"]["model"], ["4", 0])
        self.assertEqual(g["3"]["inputs"]["model"], [fu, 0])
        self.assertEqual(g[fu]["inputs"]["b1"], 1.1)

    def test_a_lora_goes_on_the_model_and_on_the_text_encoders(self):
        got, g = apply(t2i(), "lora", {"lora_name": "ink", "strength_model": 0.7})
        self.assertEqual(unfed(g), {})
        lo = next(n for n, v in g.items() if v["class_type"] == "LoraLoader")
        self.assertEqual(g[lo]["inputs"]["lora_name"], "style\\ink.safetensors")
        self.assertEqual(g["3"]["inputs"]["model"], [lo, 0])
        self.assertEqual(g["6"]["inputs"]["clip"], [lo, 1])
        self.assertEqual(g["7"]["inputs"]["clip"], [lo, 1])

    def test_a_node_that_is_no_model_patch_is_refused(self):
        got, _ = apply(t2i(), "patch_model", {"class_type": "VAEDecode"})
        self.assertFalse(got["ok"])


class HiresFix(unittest.TestCase):

    def test_a_second_pass_on_the_upscaled_latent_feeds_the_decoder(self):
        got, g = apply(t2i(), "hires_fix", {"scale_by": 2, "denoise": 0.45})
        self.assertEqual(unfed(g), {})
        second = next(n for n, v in g.items() if v["class_type"] == "KSampler" and n != "3")
        up = next(n for n, v in g.items() if v["class_type"] == "LatentUpscaleBy")
        self.assertEqual(g[up]["inputs"]["samples"], ["3", 0])
        self.assertEqual(g[up]["inputs"]["scale_by"], 2.0)
        self.assertEqual(g[second]["inputs"]["latent_image"], [up, 0])
        self.assertEqual(g[second]["inputs"]["denoise"], 0.45)
        self.assertEqual(g[second]["inputs"]["model"], ["4", 0])
        self.assertEqual(g[second]["inputs"]["steps"], 20)             # the first pass's own values
        self.assertEqual(g["8"]["inputs"]["samples"], [second, 0])


class Output(unittest.TestCase):

    def test_an_upscale_model_goes_between_the_decoder_and_the_output(self):
        got, g = apply(t2i(), "model_upscale", {"model_name": "RealESRGAN_x4plus"})
        self.assertEqual(unfed(g), {})
        up = next(n for n, v in g.items() if v["class_type"] == "ImageUpscaleWithModel")
        self.assertEqual(g[up]["inputs"]["image"], ["8", 0])
        self.assertEqual(g["9"]["inputs"]["images"], [up, 0])

    def test_preview_in_place_of_save(self):
        got, g = apply(t2i(), "preview_output")
        self.assertEqual(unfed(g), {})
        self.assertNotIn("SaveImage", classes(g))
        pv = next(v for v in g.values() if v["class_type"] == "PreviewImage")
        self.assertEqual(pv["inputs"]["images"], ["8", 0])

    def test_and_back(self):
        _, g = apply(t2i(), "preview_output")
        got, g = apply(g, "save_output", {"filename_prefix": "final"})
        self.assertNotIn("PreviewImage", classes(g))
        self.assertEqual(next(v for v in g.values() if v["class_type"] == "SaveImage")["inputs"]["filename_prefix"],
                         "final")


class OneAfterAnother(unittest.TestCase):

    def test_they_compose(self):
        g = t2i()
        for op, params in (("image_to_image", {"image": "input.png"}),
                           ("controlnet", {"control_net_name": "depth", "image": "pose.png"}),
                           ("patch_model", {"class_type": "FreeU_V2"}),
                           ("hires_fix", {}), ("model_upscale", {}), ("preview_output", {})):
            got, g = apply(g, op, params)
            self.assertTrue(got["ok"], (op, got))
            self.assertEqual(unfed(g), {}, op)
        cn = next(n for n, v in g.items() if v["class_type"] == "ControlNetApplyAdvanced")
        for s in gs.samplers(g, INFO):                  # both passes are guided and patched
            self.assertEqual(g[s]["inputs"]["positive"], [cn, 0])
            self.assertEqual(g[g[s]["inputs"]["model"][0]]["class_type"], "FreeU_V2")


class WhatItWillNotDo(unittest.TestCase):

    def test_an_unknown_operation_lists_the_ones_there_are(self):
        got = gs.splice(t2i(), "make it pretty", {}, INFO)
        self.assertFalse(got["ok"])
        self.assertIn("controlnet", got["operations"])

    def test_a_graph_without_a_sampler_is_left_alone(self):
        g = {"1": {"class_type": "LoadImage", "inputs": {"image": "input.png"}},
             "2": {"class_type": "SaveImage", "inputs": {"images": ["1", 0], "filename_prefix": "x"}}}
        for op in ("image_to_image", "controlnet", "hires_fix"):
            self.assertFalse(gs.splice(g, op, {"image": "input.png", "control_net_name": "canny"}, INFO)["ok"], op)

    def test_nothing_is_changed_by_asking(self):
        g = t2i()
        before = json.dumps(g, sort_keys=True)
        gs.splice(g, "controlnet", {"control_net_name": "canny", "image": "pose.png"}, INFO)
        self.assertEqual(json.dumps(g, sort_keys=True), before)


class TheTool(unittest.TestCase):
    """change_graph: the same operations, on the open canvas and on a workflow file."""

    def _pipe(self):
        from pipeline_stub import pipeline_stub
        pipe = pipeline_stub(_canvas_graph=t2i(), _canvas_selection=[], _canvas_edits=[],
                             _canvas_touched=set(), _canvas_unwired={})
        pipe._canvas_full_graph = lambda: True
        return pipe

    def _call(self, pipe, **kw):
        import asyncio
        from unittest import mock
        from pipeline_stub import tools
        with mock.patch("src.pipeline._CanvasSchemas", return_value=INFO):
            return json.loads(asyncio.run(tools(pipe)["change_graph"](**kw)))

    def test_on_the_canvas_it_leaves_a_complete_graph(self):
        pipe = self._pipe()
        out = self._call(pipe, operation="patch_model", params={"class_type": "FreeU_V2"})
        self.assertEqual(out["status"], "ok", out)
        g = pipe._canvas_graph
        fu = next(n for n, v in g.items() if v["class_type"] == "FreeU_V2")
        self.assertEqual(g["3"]["inputs"]["model"], [fu, 0])
        self.assertEqual(pipe._canvas_unwired, {})

    def test_params_may_arrive_as_text(self):
        out = self._call(self._pipe(), operation="hires_fix", params='{"scale_by": 2}')
        self.assertEqual(out["status"], "ok", out)

    def test_what_it_cannot_do_changes_nothing_and_says_so(self):
        pipe = self._pipe()
        before = json.dumps(pipe._canvas_graph, sort_keys=True)
        out = self._call(pipe, operation="controlnet", params={"control_net_name": "nope", "image": "pose.png"})
        self.assertEqual(out["status"], "cannot")
        self.assertIn("by hand", out["note"])
        self.assertEqual(json.dumps(pipe._canvas_graph, sort_keys=True), before)

    def test_on_a_workflow_file(self):
        import tempfile
        from unittest import mock
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "wf.json"
            path.write_text(json.dumps(t2i()), encoding="utf-8")
            with mock.patch("agenty_core.tools.comfyui._edit_target", return_value=path),                     mock.patch("src.tools.validate_workflow", new=lambda p: json.dumps({"valid": True, "local_errors": []})):
                out = self._call(self._pipe(), operation="image_to_image",
                                 params={"image": "input.png", "denoise": 0.6}, workflow_path=str(path))
            self.assertEqual(out["status"], "ok", out)
            g = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(unfed(g), {})
        self.assertEqual(g["3"]["inputs"]["denoise"], 0.6)
        self.assertNotIn("5", g)


if __name__ == "__main__":
    unittest.main()
