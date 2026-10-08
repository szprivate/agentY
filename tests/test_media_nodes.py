"""The load and save node for each kind of file is the user's to choose.

Settings > Load & save nodes offers, per kind, the loaders and savers this
ComfyUI has - bEpic's "Send to Image Viewer" among them. A chosen saver replaces
the one a workflow was built with when it can take the same connection; a chosen
loader is what results are dropped onto the canvas with.
"""

import tomllib
import unittest
from pathlib import Path
from unittest import mock

from src.utils import media_loaders, media_nodes as mn

ROOT = Path(__file__).resolve().parents[1]


def _node(required=None, optional=None, output=(), output_node=False, display_name=""):
    return {"input": {"required": required or {}, "optional": optional or {}},
            "output": list(output), "output_node": output_node, "display_name": display_name}


# Shaped after the real nodes (object_info of this ComfyUI, trimmed).
INFO = {
    "SaveImage": _node({"images": ["IMAGE", {}], "filename_prefix": ["STRING", {"default": "ComfyUI"}]},
                       output_node=True, display_name="Save Image"),
    "PreviewImage": _node({"images": ["IMAGE", {}]}, output_node=True),
    "bEpic_imageSave": _node({"images": ["IMAGE", {}], "filename_prefix": ["STRING", {"default": "x"}],
                              "file_format": [["exr", "png"], {"default": "exr"}]},
                             {"audio": ["AUDIO", {}]}, output_node=True),
    "bEpicSendToViewer": _node({"input": ["*", {}], "tab_name": ["STRING", {"default": "viewer"}],
                                "save_to_output": ["BOOLEAN", {"default": False}],
                                "file_format": [["png", "jpg", "mp4", "exr"], {"default": "png"}],
                                "filename_prefix": ["STRING", {"default": "viewer"}]},
                               output=["*"], output_node=True, display_name="Send to Image Viewer"),
    "SaveVideo": _node({"video": ["VIDEO", {}], "filename_prefix": ["STRING", {"default": "video/ComfyUI"}]},
                       output_node=True),
    "VHS_VideoCombine": _node({"images": ["IMAGE", {}], "frame_rate": ["FLOAT", {"default": 8}],
                               "filename_prefix": ["STRING", {"default": "AnimateDiff"}]},
                              {"audio": ["AUDIO", {}]}, output_node=True),
    "SaveAudio": _node({"audio": ["AUDIO", {}], "filename_prefix": ["STRING", {"default": "audio"}]},
                       output_node=True),
    "LoadImage": _node({"image": [["a.png"], {}]}, output=["IMAGE", "MASK"], display_name="Load Image"),
    "VHS_LoadImagePath": _node({"image": ["STRING", {}], "custom_width": ["INT", {"default": 0}]},
                               {"vae": ["VAE", {}]}, output=["IMAGE", "MASK"]),
    "bepic_imageLoad": _node({"image_path": ["STRING", {}]}, output=["IMAGE", "MASK"],
                             display_name="bEpic Image Load"),
    "LoadVideo": _node({"file": [["a.mp4"], {}]}, output=["VIDEO"]),
    "VHS_LoadVideoPath": _node({"video": ["STRING", {}]}, output=["IMAGE", "INT", "AUDIO"],
                               display_name="Load Video (Path)"),
    "LoadAudio": _node({"audio": [["a.wav"], {}]}, output=["AUDIO"]),
    "LoadImageMask": _node({"image": [["a.png"], {}]}, output=["MASK"]),
    "ImageScale": _node({"image": ["IMAGE", {}], "width": ["INT", {}]}, output=["IMAGE"]),
    "KSampler": _node({"model": ["MODEL", {}]}, output=["LATENT"]),
}


def _ids(entries):
    return [e["id"] for e in entries]


class WhatCanBeChosen(unittest.TestCase):

    def setUp(self):
        self.c = mn.choices(INFO)

    def test_image_savers_include_the_viewer_and_leave_previews_out(self):
        self.assertEqual(_ids(self.c["image"]["save"]), ["SaveImage", "bEpicSendToViewer", "bEpic_imageSave"])

    def test_a_node_that_takes_anything_is_offered_for_every_kind(self):
        for kind in mn.KINDS:
            self.assertIn("bEpicSendToViewer", _ids(self.c[kind]["save"]))

    def test_video_and_audio_savers(self):
        self.assertEqual(_ids(self.c["video"]["save"]), ["SaveVideo", "VHS_VideoCombine", "bEpicSendToViewer"])
        self.assertEqual(_ids(self.c["audio"]["save"]), ["SaveAudio", "bEpicSendToViewer"])

    def test_loaders_by_what_they_hand_out(self):
        self.assertEqual(_ids(self.c["image"]["load"]), ["LoadImage", "VHS_LoadImagePath", "bepic_imageLoad"])
        self.assertEqual(_ids(self.c["video"]["load"]), ["LoadVideo", "VHS_LoadVideoPath"],
                         "a video loader that hands out frames is still a video loader")
        self.assertEqual(_ids(self.c["audio"]["load"]), ["LoadAudio"])

    def test_the_label_is_the_name_on_the_node_with_its_class(self):
        viewer = next(e for e in self.c["image"]["save"] if e["id"] == "bEpicSendToViewer")
        self.assertEqual(viewer["label"], "Send to Image Viewer  (bEpicSendToViewer)")

    def test_a_comfyui_that_cannot_be_asked_offers_nothing(self):
        with mock.patch.dict(mn._choices_cache, {}, clear=True), \
                mock.patch("agenty_core.tools.comfyui._get_object_info", side_effect=OSError("down")):
            self.assertEqual(mn.choices(), {})


def _settings(**chosen):
    return mock.patch("src.agent._load_settings", return_value={"media_nodes": chosen})


def _graph(saver="SaveImage", **inputs):
    base = {"images": ["8", 0], "filename_prefix": "agent/images/cube"}
    base.update(inputs)
    return {"8": {"class_type": "VAEDecode", "inputs": {}},
            "9": {"class_type": saver, "inputs": base, "_meta": {"title": "Save Image"}}}


def _apply(graph):
    return mn.apply_savers(graph, schema_of=lambda c: INFO.get(c, {}))


def _defaults(cls):
    return {n: s[1]["default"] for n, s in INFO[cls]["input"]["required"].items()
            if isinstance(s[1], dict) and "default" in s[1]}


class SwappingTheSaver(unittest.TestCase):

    def setUp(self):
        self.enterContext(mock.patch.object(mn, "_default_inputs", _defaults))

    def test_nothing_chosen_nothing_changes(self):
        graph = _graph()
        with _settings(image_save=""):
            self.assertEqual(_apply(graph), {})
        self.assertEqual(graph["9"]["class_type"], "SaveImage")

    def test_the_viewer_takes_the_image_the_name_and_is_told_to_save(self):
        graph = _graph()
        with _settings(image_save="bEpicSendToViewer"):
            self.assertEqual(_apply(graph), {"9": ("SaveImage", "bEpicSendToViewer")})
        node = graph["9"]
        self.assertEqual(node["class_type"], "bEpicSendToViewer")
        self.assertEqual(node["inputs"]["input"], ["8", 0])
        self.assertEqual(node["inputs"]["filename_prefix"], "agent/images/cube")
        self.assertIs(node["inputs"]["save_to_output"], True)
        self.assertEqual(node["inputs"]["tab_name"], "viewer", "its other inputs are at their defaults")
        self.assertNotIn("images", node["inputs"])
        self.assertNotIn("_meta", node, "the old node's title is not carried over")

    def test_the_node_keeps_its_id(self):
        graph = _graph()
        with _settings(image_save="bEpic_imageSave"):
            _apply(graph)
        self.assertEqual(list(graph), ["8", "9"])
        self.assertEqual(graph["9"]["inputs"]["images"], ["8", 0])

    def test_a_wire_both_nodes_know_travels_too(self):
        graph = _graph("VHS_VideoCombine", audio=["3", 0])
        with _settings(video_save="bEpicSendToViewer", image_save="bEpic_imageSave"):
            self.assertEqual(_apply(graph), {"9": ("VHS_VideoCombine", "bEpicSendToViewer")})
        self.assertEqual(graph["9"]["inputs"]["file_format"], "mp4", "frames saved as a video stay a video")

    def test_a_saver_that_cannot_take_the_wire_is_not_used(self):
        """SaveVideo takes a VIDEO; VHS_VideoCombine was saving IMAGE frames."""
        graph = _graph("VHS_VideoCombine")
        with _settings(video_save="SaveVideo"):
            self.assertEqual(_apply(graph), {})
        self.assertEqual(graph["9"]["class_type"], "VHS_VideoCombine")

    def test_an_image_choice_does_not_touch_a_video_saver(self):
        graph = _graph("VHS_VideoCombine")
        with _settings(image_save="SaveImage"):
            self.assertEqual(_apply(graph), {})

    def test_a_choice_this_comfyui_does_not_have_is_ignored(self):
        graph = _graph()
        with _settings(image_save="NodeFromARemovedPack"):
            self.assertEqual(_apply(graph), {})

    def test_it_is_done_once(self):
        graph = _graph()
        with _settings(image_save="bEpicSendToViewer"):
            _apply(graph)
            self.assertEqual(_apply(graph), {}, "the executor and the canvas insert both call it")

    def test_a_wired_file_name_stays_wired(self):
        graph = _graph(filename_prefix=["4", 0])
        with _settings(image_save="bEpic_imageSave"):
            _apply(graph)
        self.assertEqual(graph["9"]["inputs"]["filename_prefix"], ["4", 0])


class TheLoader(unittest.TestCase):

    def test_the_chosen_loader_is_tried_first_and_the_rest_stay_as_fallback(self):
        with _settings(image_load="bepic_imageLoad"):
            self.assertEqual(media_loaders.candidates("image"),
                             ["bepic_imageLoad", "VHS_LoadImagePath", "LoadImage"])
            self.assertEqual(media_loaders.candidates("video"), media_loaders.CANDIDATES["video"])
        with _settings():
            self.assertEqual(media_loaders.candidates("image"), media_loaders.CANDIDATES["image"])

    def test_which_shape_a_chosen_loader_is_comes_from_its_own_widget(self):
        self.assertEqual(mn.file_widget(INFO["bepic_imageLoad"]), ("image_path", True))
        self.assertEqual(mn.file_widget(INFO["LoadImage"]), ("image", False))
        with mock.patch.object(mn, "_schema", lambda c: INFO.get(c, {})):
            self.assertTrue(media_loaders.takes_absolute_path("bepic_imageLoad"))
            self.assertTrue(media_loaders.takes_absolute_path("VHS_LoadVideoPath"))
            self.assertFalse(media_loaders.takes_absolute_path("LoadImage"))


class WhereItIsWired(unittest.TestCase):

    def test_the_settings_start_on_automatic(self):
        defaults = tomllib.loads((ROOT / "config" / "settings.default.toml").read_text(encoding="utf-8"))
        self.assertEqual(defaults["media_nodes"], {f"{k}_{r}": "" for k in mn.KINDS for r in ("load", "save")})
        self.assertIn("update_channel", defaults, "the table must not swallow the settings after it")

    def test_every_submission_and_every_canvas_insert_passes_through(self):
        executor = (ROOT / "src" / "executor.py").read_text(encoding="utf-8")
        submit = executor.split("def _submit_workflow(", 1)[1].split("\ndef ", 1)[0]
        self.assertLess(submit.index("media_nodes.apply_savers(workflow)"),
                        submit.index("output_context.stamp(workflow)"),
                        "the chosen saver is in place before it is named")
        pipeline = (ROOT / "src" / "pipeline.py").read_text(encoding="utf-8")
        insert = pipeline.split("async def insert_workflow_into_canvas(", 1)[1].split("@_tool", 1)[0]
        self.assertLess(insert.index("_mn.apply_savers(_built)"), insert.index("_ce.ops_for_workflow(_built)"))

    def test_the_settings_page_is_given_the_candidates(self):
        server = (ROOT / "src" / "utils" / "agentY_server.py").read_text(encoding="utf-8")
        self.assertIn('"media_node_choices": _media_node_choices(),', server)


if __name__ == "__main__":
    unittest.main()
