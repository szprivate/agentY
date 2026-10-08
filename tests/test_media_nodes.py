"""The load and save node for each kind of file is the user's to choose.

Settings > Load & save nodes offers, per kind, the loaders and savers this
ComfyUI has - bEpic's "Send to Image Viewer" among them. A chosen saver replaces
the one a workflow was built with when it can take the same connection; a chosen
loader is what results are dropped onto the canvas with.
"""

import asyncio
import json
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
                                "file_format": [["png", "jpg", "mp4", "mov", "exr", "glb"], {"default": "png"}],
                                "fps": ["FLOAT", {"default": 24.0, "bepic_video_formats": ["mp4", "mov"],
                                                  "bepic_model_formats": ["glb"]}],
                                "filename_prefix": ["STRING", {"default": "viewer"}],
                                "is_sequence": ["BOOLEAN", {"default": False}],
                                "first_frame_number": ["INT", {"default": 1001}],
                                "padding": ["INT", {"default": 4}]},
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
    # 3D: an input lists every type it takes, comma-separated.
    "SaveGLB": _node({"mesh": ["MESH,FILE_3D_GLB,FILE_3D_OBJ,FILE_3D", {}],
                      "filename_prefix": ["STRING", {"default": "mesh/ComfyUI"}]},
                     output_node=True, display_name="Save 3D Model"),
    "Save3DAdvanced": _node({"model_3d": ["FILE_3D_GLB,FILE_3D_OBJ,FILE_3D", {}],
                             "filename_prefix": ["STRING", {"default": "3d/ComfyUI"}]},
                            {"camera_info": ["LOAD3D_CAMERA", {}]}, output_node=True),
    "SaveGaussianSplat": _node({"model_3d": ["FILE_3D_SPLAT_ANY,FILE_3D_PLY", {}],
                                "filename_prefix": ["STRING", {"default": "3d/ComfyUI"}]},
                               output_node=True, display_name="Save Splat"),
    "Load3D": _node({"model_file": [["3d/a.glb"], {}], "image": ["LOAD_3D", {}], "width": ["INT", {}]},
                    output=["IMAGE", "MASK", "STRING", "FILE_3D"], display_name="Load 3D & Animation"),
    "AYON Load 3D Model": _node({"ayon_container_info": ["STRING", {}]}, output=["FILE_3D"]),
    "MeshGen": _node({"image": ["IMAGE", {}]}, output=["MESH"]),
    "Glb3DGen": _node({"image": ["IMAGE", {}]}, output=["FILE_3D_GLB"]),
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


class ThreeD(unittest.TestCase):
    """3D inputs take several wire types; which one is on the wire decides."""

    def setUp(self):
        self.enterContext(mock.patch.object(mn, "_default_inputs", _defaults))
        self.c = mn.choices(INFO)

    def _graph(self, source, saver="SaveGLB", wire="mesh"):
        return {"8": {"class_type": source, "inputs": {}},
                "9": {"class_type": saver, "inputs": {wire: ["8", 0], "filename_prefix": "agent/models/chair"}}}

    def test_savers_are_found_by_what_they_take(self):
        self.assertEqual(_ids(self.c["model"]["save"]),
                         ["Save3DAdvanced", "SaveGLB", "SaveGaussianSplat", "bEpicSendToViewer"])
        self.assertNotIn("SaveGaussianSplat", _ids(self.c["image"]["save"]))

    def test_the_loader_is_the_one_with_a_model_file_and_a_viewport_widget(self):
        self.assertEqual(_ids(self.c["model"]["load"]), ["Load3D"])
        self.assertEqual(mn.file_widget(INFO["Load3D"]), ("model_file", False))
        self.assertNotIn("Load3D", _ids(self.c["image"]["load"]), "it hands out images too")

    def test_the_viewer_takes_a_mesh(self):
        graph = self._graph("MeshGen")
        with _settings(model_save="bEpicSendToViewer"):
            self.assertEqual(_apply(graph), {"9": ("SaveGLB", "bEpicSendToViewer")})
        node = graph["9"]["inputs"]
        self.assertEqual((node["input"], node["filename_prefix"], node["save_to_output"]),
                         (["8", 0], "agent/models/chair", True))
        self.assertEqual(node["file_format"], "png", "left alone: the viewer keeps a model's own format")

    def test_a_file_saver_is_not_put_where_a_mesh_was_being_saved(self):
        """SaveGLB takes a MESH or a file; Save 3D (Advanced) only a file."""
        graph = self._graph("MeshGen")
        with _settings(model_save="Save3DAdvanced"):
            self.assertEqual(_apply(graph), {})
        self.assertEqual(graph["9"]["class_type"], "SaveGLB")

    def test_it_is_when_the_wire_carries_a_file(self):
        graph = self._graph("Glb3DGen")
        with _settings(model_save="Save3DAdvanced"):
            self.assertEqual(_apply(graph), {"9": ("SaveGLB", "Save3DAdvanced")})
        self.assertEqual(graph["9"]["inputs"]["model_3d"], ["8", 0])

    def test_a_wire_nobody_can_name_needs_a_saver_that_takes_all_the_old_one_did(self):
        graph = self._graph("NodeFromAnUnknownPack")
        with _settings(model_save="Save3DAdvanced"):
            self.assertEqual(_apply(graph), {})
        with _settings(model_save="bEpicSendToViewer"):
            self.assertEqual(list(_apply(graph)), ["9"])

    def test_a_3d_result_is_placed_on_the_canvas_only_with_a_chosen_loader(self):
        from src.utils import agentY_server as srv
        with _settings():
            self.assertEqual(srv._drop_kind("W:/out/chair.glb"), "file")
            self.assertEqual(srv._drop_kind("W:/out/voice.wav"), "file")
            self.assertEqual(srv._drop_kind("W:/out/a.png"), "image")
        with _settings(model_load="Load3D", audio_load="LoadAudio"):
            self.assertEqual(srv._drop_kind("W:/out/chair.glb"), "model")
            self.assertEqual(srv._drop_kind("W:/out/voice.wav"), "audio")
            self.assertEqual(srv._NODE_CANDIDATES.get("model", []), ["Load3D"])

    def test_a_staged_model_goes_where_load_3d_looks(self):
        import tempfile
        from src.utils import agentY_server as srv
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "out" / "chair.glb"
            src.parent.mkdir()
            src.write_text("x", encoding="utf-8")
            with mock.patch.object(srv, "_comfy_input_dir", lambda: Path(tmp) / "input"):
                self.assertEqual(srv._stage_into_comfy_input(str(src), "3d"), "3d/chair.glb")
                self.assertTrue((Path(tmp) / "input" / "3d" / "chair.glb").exists())
                self.assertEqual(srv._stage_into_comfy_input(str(src)), "chair.glb")


class HowAClipIsSaved(unittest.TestCase):
    """A video file or an image sequence: the setting decides, the workflow's
    frame rate travels, and what the agent sets for one run is not undone."""

    def setUp(self):
        self.enterContext(mock.patch.object(mn, "_default_inputs", _defaults))

    def _frames(self, **inputs):
        return _graph("VHS_VideoCombine", frame_rate=16, **inputs)

    def _video(self):
        return {"8": {"class_type": "VideoGen", "inputs": {}},
                "9": {"class_type": "SaveVideo", "inputs": {"video": ["8", 0], "filename_prefix": "agent/videos/clip"}}}

    def test_which_formats_a_node_can_write_a_sequence_in(self):
        self.assertEqual(mn.sequence_formats(INFO["bEpicSendToViewer"]), ["png", "jpg", "exr"])
        self.assertEqual(mn.sequence_formats(INFO["SaveVideo"]), [])
        self.assertEqual(mn.choices(INFO)["video"]["sequence_formats"],
                         {"bEpicSendToViewer": ["png", "jpg", "exr"]})

    def test_the_workflows_frame_rate_goes_with_it(self):
        graph = self._frames()
        with _settings(video_save="bEpicSendToViewer"):
            _apply(graph)
        self.assertEqual(graph["9"]["inputs"]["fps"], 16.0)

    def test_a_video_file_by_default_whichever_wire_it_came_on(self):
        frames, video = self._frames(), self._video()
        with _settings(video_save="bEpicSendToViewer"):
            _apply(frames), _apply(video)
        for graph in (frames, video):
            self.assertEqual(graph["9"]["inputs"]["file_format"], "mp4")
            self.assertIs(graph["9"]["inputs"]["is_sequence"], False)
        self.assertEqual(video["9"]["inputs"]["input"], ["8", 0])

    def test_an_image_sequence_when_that_is_the_setting(self):
        frames, video = self._frames(), self._video()
        with _settings(video_save="bEpicSendToViewer", video_as="sequence", sequence_format="exr"):
            _apply(frames), _apply(video)
        for graph in (frames, video):
            node = graph["9"]["inputs"]
            self.assertEqual((node["file_format"], node["is_sequence"]), ("exr", True))
            self.assertEqual((node["first_frame_number"], node["padding"]), (1001, 4))

    def test_a_format_the_node_cannot_write_falls_back_to_png(self):
        graph = self._frames()
        with _settings(video_save="bEpicSendToViewer", video_as="sequence", sequence_format="dpx"):
            _apply(graph)
        self.assertEqual(graph["9"]["inputs"]["file_format"], "png")

    def test_a_node_that_cannot_write_a_sequence_is_left_writing_video(self):
        graph = self._video()
        with _settings(video_save="SaveVideo", video_as="sequence"):
            self.assertEqual(_apply(graph), {})
        self.assertNotIn("is_sequence", graph["9"]["inputs"])

    def test_an_image_saver_is_not_touched_by_the_video_setting(self):
        graph = _graph()
        with _settings(image_save="bEpicSendToViewer", video_as="sequence", sequence_format="exr"):
            _apply(graph)
        self.assertEqual(graph["9"]["inputs"]["file_format"], "png")
        self.assertIs(graph["9"]["inputs"]["is_sequence"], False)

    def test_what_the_agent_set_for_one_run_survives_the_run(self):
        """Built with the chosen node, changed by the agent, then submitted."""
        graph = self._frames()
        with _settings(video_save="bEpicSendToViewer"):
            _apply(graph)                                   # at build
            graph["9"]["inputs"].update(file_format="exr", is_sequence=True, first_frame_number=1)
            self.assertEqual(_apply(graph), {})             # at submission
        self.assertEqual(graph["9"]["inputs"]["file_format"], "exr")
        self.assertEqual(graph["9"]["inputs"]["first_frame_number"], 1)

    def test_the_agent_is_told_what_it_can_set(self):
        graph = self._frames()
        with _settings(video_save="bEpicSendToViewer"):
            _apply(graph)
            (saver,) = mn.describe_savers(graph, schema_of=lambda c: INFO.get(c, {}))
        self.assertEqual((saver["node_id"], saver["node"]), ("9", "bEpicSendToViewer"))
        self.assertEqual(saver["values"]["fps"], 16.0)
        self.assertEqual(saver["options"]["file_format"], ["png", "jpg", "mp4", "mov", "exr", "glb"])
        self.assertNotIn("input", saver["values"], "a wire is not a value")
        with _settings():
            self.assertEqual(mn.describe_savers(graph, schema_of=lambda c: INFO.get(c, {})), [])


class TheAgentsOwnTools(unittest.TestCase):
    """The node tools the agent already has, pointed at a built workflow."""

    def setUp(self):
        import tempfile
        from agenty_core.utils import turn_scope
        token = turn_scope.enter(turn_scope.Scope("req", "thread"))
        self.addCleanup(turn_scope.leave, token)
        self.enterContext(mock.patch.object(mn, "_default_inputs", _defaults))
        self.enterContext(mock.patch.object(mn, "_schema", lambda c: INFO.get(c, {})))
        self.enterContext(mock.patch("src.utils.preflight._schema", side_effect=lambda c: INFO.get(c, {})))
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.path = Path(tmp.name) / "clip.json"
        self.path.write_text(json.dumps(_graph("VHS_VideoCombine", frame_rate=16)), encoding="utf-8")

    def _tool(self, name, **kw):
        from pipeline_stub import pipeline_stub, tools
        return json.loads(asyncio.run(tools(pipeline_stub())[name](**kw)))

    def _saved(self):
        return json.loads(self.path.read_text(encoding="utf-8"))["9"]

    def test_a_built_workflow_comes_back_with_the_chosen_saver_and_its_values(self):
        from pipeline_stub import pipeline_stub
        result = {"status": "ready", "workflow_path": str(self.path)}
        with _settings(video_save="bEpicSendToViewer"):
            pipeline_stub()._apply_chosen_savers(result)
        self.assertEqual(self._saved()["class_type"], "bEpicSendToViewer")
        (saver,) = result["save_nodes"]
        self.assertEqual((saver["node_id"], saver["values"]["file_format"]), ("9", "mp4"))
        self.assertIn("set_canvas_node_params(node_id, {...}, workflow_path=", result["save_note"])

    def test_nothing_chosen_nothing_added(self):
        from pipeline_stub import pipeline_stub
        result = {"status": "ready", "workflow_path": str(self.path)}
        with _settings():
            pipeline_stub()._apply_chosen_savers(result)
        self.assertNotIn("save_nodes", result)
        self.assertEqual(self._saved()["class_type"], "VHS_VideoCombine")

    def test_save_this_one_as_an_exr_sequence_from_frame_1(self):
        with _settings(video_save="bEpicSendToViewer"):
            from pipeline_stub import pipeline_stub
            pipeline_stub()._apply_chosen_savers({"workflow_path": str(self.path)})
            seen = self._tool("get_canvas_node", node_id="9", workflow_path=str(self.path))
            self.assertEqual(seen["values"]["file_format"], "mp4")
            out = self._tool("set_canvas_node_params", node_id="9", workflow_path=str(self.path),
                             params={"file_format": "exr", "is_sequence": True, "first_frame_number": 1})
        self.assertEqual(out["status"], "applied", out)
        node = self._saved()["inputs"]
        self.assertEqual((node["file_format"], node["is_sequence"], node["first_frame_number"]), ("exr", True, 1))
        self.assertEqual(node["fps"], 16.0, "what it did not name stays")

    def test_a_value_the_node_does_not_take_is_refused_and_nothing_is_written(self):
        before = self.path.read_text(encoding="utf-8")
        out = self._tool("set_canvas_node_params", node_id="9", workflow_path=str(self.path),
                         params={"file_format": "exr"})
        self.assertEqual(out["status"], "rejected", out)
        out = self._tool("set_canvas_node_params", node_id="9", workflow_path=str(self.path),
                         params={"images": "x.png"})
        self.assertIn("wired from another node", out["errors"][0])
        self.assertEqual(self.path.read_text(encoding="utf-8"), before)

    def test_a_node_or_a_file_that_is_not_there_is_said(self):
        out = self._tool("get_canvas_node", node_id="77", workflow_path=str(self.path))
        self.assertIn("no node '77'", out["error"])
        out = self._tool("set_canvas_node_params", node_id="9", params={"fps": 12},
                         workflow_path=str(self.path) + ".missing")
        self.assertIn("cannot read", out["error"])


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
        want = {f"{k}_{r}": "" for k in mn.KINDS for r in ("load", "save")}
        want.update(video_as="video", sequence_format="png")
        self.assertEqual(defaults["media_nodes"], want)
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
