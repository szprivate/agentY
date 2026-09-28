"""The World Builder builds a world around one model of the whole picture, and
gives it a real sky.

The user asked for three things: the whole world built around a 3D model of
the original picture (with the engine theirs to choose, including any other
fitting template in the library), the ground meeting that model at the right
height and scale, and web HDRIs before generated skies. The routes and
ComfyUI are faked here; what is checked is what the tools send and choose.

    python -m unittest tests.test_world_scene_model
"""

import json
import pathlib
import unittest
from unittest import mock

import src.tools.worlds as W

SLOTS = {"slots": {"scene_model": {"default": "s3d_trellis2", "templates": {
    "s3d_trellis2": {"about": "TRELLIS.2", "frame": "object", "textured": True, "cost": "local"},
    "s3d_sharp": {"about": "SHARP", "frame": "camera_cv", "textured": True, "cost": "local"},
    "s3d_meshy": {"about": "Meshy", "frame": "object", "textured": True, "cost": "api"}}}},
    "choices": {"scene_model": "s3d_trellis2"}}
CATALOG = {"api_rodin_image_to_model": "Rodin", "api_tripo_text_to_model": "text", "api_meshy_image_to_model": "covered",
           "3d_moge_perspective_to_mesh": "covered", "api_tripo_multiview_to_model": "multiview",
           "image_to_model_hunyuan3d_2_1": "local hunyuan", "flux_dev": "not 3d"}
SHARP_WF = {"1": {"class_type": "LoadImage", "inputs": {"image": "x"}, "_meta": {"title": "IN:image"}},
            "3": {"class_type": "SharpPredict", "inputs": {"focal_length_mm": 30.0},
                  "_meta": {"title": "IN:focal_length_mm.focal_length_mm"}},
            "7": {"class_type": "SaveGLB", "inputs": {"filename_prefix": "p"}, "_meta": {"title": "OUT:mesh"}}}
WORLD = {"items": [{"id": "refcam", "kind": "camera", "fov": 50.0, "resolution": [1600, 900]},
                   {"id": "scene", "kind": "model", "scene_model": {"size_m": [30, 12, 20], "error_m": 0.4, "coverage": 0.7}}]}


class Routes:
    """The pack's routes as the tools see them; records what was posted."""

    def __init__(self):
        self.posts = []

    def get(self, path, **params):
        if path == "/bepic_worlds/slots":
            return SLOTS
        if path == "/bepic_worlds/slot_template":
            return json.loads(json.dumps(SHARP_WF))
        if path == "/bepic_worlds/world":
            return WORLD
        if path == "/bepic_worlds/hdri_search":
            self.search = params
            return {"matches": [{"id": "sky_a", "name": "Sky A", "thumbnail": "t", "page": "p"},
                                {"id": "sky_b", "name": "Sky B", "thumbnail": "t", "page": "p"}]}
        raise AssertionError(path)

    def post(self, path, body, timeout=600):
        self.posts.append((path, body))
        if path == "/bepic_worlds/stage_reference":
            return {"reference": {"filename": "ref.png", "subfolder": "worlds_refs", "type": "input"}}
        return {"version": 5, "sun": {"azimuth": 150, "elevation": 30, "shadows": True}}


class Base(unittest.TestCase):
    def setUp(self):
        self.r = Routes()
        self.patches = [mock.patch.object(W, "_get", self.r.get), mock.patch.object(W, "_post", self.r.post),
                        mock.patch.object(W, "_progress", lambda m: None),
                        mock.patch("agenty_core.tools.comfyui.get_workflow_catalog", lambda: json.dumps(CATALOG))]
        for p in self.patches:
            p.start()

    def tearDown(self):
        for p in self.patches:
            p.stop()


class TheUserChoosesTheEngine(Base):

    def test_options_are_the_packs_and_the_librarys_image_to_3d(self):
        engines = json.loads(W.world_scene_model_options())["engines"]
        self.assertIn("s3d_sharp", engines)
        self.assertEqual(engines["s3d_sharp"]["frame"], "camera_cv")
        # other fitting templates are offered too …
        self.assertIn("api_rodin_image_to_model", engines)
        self.assertEqual(engines["api_rodin_image_to_model"]["cost"], "api")
        self.assertIn("image_to_model_hunyuan3d_2_1", engines)
        # … but not text-to-3D, multiview, what the pack already covers, or non-3D
        for name in ("api_tripo_text_to_model", "api_tripo_multiview_to_model", "api_meshy_image_to_model",
                     "3d_moge_perspective_to_mesh", "flux_dev"):
            self.assertNotIn(name, engines)

    def test_the_prompts_make_it_a_question_for_the_user(self):
        root = pathlib.Path(__file__).resolve().parent.parent / "config" / "system_prompts"
        wb = (root / "system_prompt.world_builder.md").read_text(encoding="utf-8")
        orch = (root / "system_prompt.orchestrator.md").read_text(encoding="utf-8")
        self.assertIn("QUESTION FOR THE USER", wb)
        self.assertIn("world_scene_model_options", wb)
        self.assertIn("QUESTION FOR THE USER", orch)


class TheSceneModel(Base):

    def test_a_camera_engine_gets_the_worlds_lens_and_is_placed_by_the_camera(self):
        ran = {}

        def run(prompt, label, timeout=1800):
            ran["prompt"] = prompt
            return {"7": {"3d": [{"filename": "scene.glb", "subfolder": "worlds", "type": "output"}]}}

        with mock.patch.object(W, "_run", run):
            out = json.loads(W.world_scene_model(name="street", engine="s3d_sharp"))
        self.assertEqual(out["status"], "ok", out)
        # 50° vertical over 16:9 → 79.3° across → 18 / tan(39.7°) ≈ 21.7 mm (35 mm equivalent)
        self.assertAlmostEqual(ran["prompt"]["3"]["inputs"]["focal_length_mm"], 21.7, delta=0.1)
        edit = [b for p, b in self.r.posts if p == "/bepic_worlds/edit"][0]
        op = edit["ops"][0]
        self.assertEqual(op["op"], "set_scene_model")
        self.assertEqual(op["frame"], "camera_cv")
        self.assertEqual(op["glb"]["filename"], "scene.glb")
        self.assertNotIn("height_m", op)

    def test_a_known_height_and_turn_are_passed_on(self):
        run = lambda prompt, label, timeout=1800: {"7": {"3d": [{"filename": "s.glb", "subfolder": "", "type": "output"}]}}
        with mock.patch.object(W, "_run", run):
            W.world_scene_model(name="street", engine="s3d_meshy", height_m=11.0, yaw=90)
        op = [b for p, b in self.r.posts if p == "/bepic_worlds/edit"][0]["ops"][0]
        self.assertEqual((op["frame"], op["height_m"], op["yaw"]), ("object", 11.0, 90.0))

    def test_an_unknown_engine_names_the_choices(self):
        out = json.loads(W.world_scene_model(name="street", engine="nope"))
        self.assertEqual(out["status"], "error")
        self.assertIn("s3d_trellis2", out["error"])


class TheSkyComesFromTheWebFirst(Base):

    def setUp(self):
        super().setUp()
        self.patches += [mock.patch.object(W, "_download", lambda ref, folder="": "ref.png"),
                         mock.patch.object(W, "_contact_sheet", lambda ref, thumbs, out: out),
                         mock.patch.object(W.requests, "get", lambda url, timeout=30: mock.Mock(
                             content=b"png", raise_for_status=lambda: None))]
        for p in self.patches[-3:]:
            p.start()

    def _eyes(self, *answers):
        it = iter(answers)
        return mock.patch.object(W, "_describe", lambda ref, q, folder: next(it))

    def test_the_hdri_the_eyes_pick_is_applied_with_open_sky(self):
        with self._eyes('{"indoor": false, "weather": "overcast", "time_of_day": "midday", "words": "grey sky"}', "2"):
            out = json.loads(W.world_environment(name="street"))
        self.assertEqual((out["status"], out["hdri"]), ("ok", "sky_b"))
        self.assertEqual(self.r.search["open_sky"], "true")
        self.assertEqual(self.r.search["weather"], "overcast")
        applied = [b for p, b in self.r.posts if p == "/bepic_worlds/hdri"]
        self.assertEqual(applied[0]["id"], "sky_b")

    def test_nothing_fits_so_a_sky_is_generated(self):
        made = {}
        with self._eyes('{"indoor": false, "words": "a purple alien sky"}', "0"), \
                mock.patch.object(W, "world_make_sky",
                                  lambda name, description="": (made.update(d=description), '{"status": "ok"}')[1]):
            out = json.loads(W.world_environment(name="street"))
        self.assertEqual(out["source"], "generated")
        self.assertEqual(made["d"], "a purple alien sky")
        self.assertFalse([p for p, _ in self.r.posts if p == "/bepic_worlds/hdri"])

    def test_an_interior_is_left_alone(self):
        with self._eyes('{"indoor": true, "words": "a car park"}'):
            out = json.loads(W.world_environment(name="street"))
        self.assertFalse(out["applied"])


if __name__ == "__main__":
    unittest.main()
