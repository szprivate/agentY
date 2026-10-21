"""An explicit "local" request is a hard constraint (#54); all-in-one comparison
templates are not offered to automatic assembly (#47)."""

import json
import unittest

import src.tools as tools_mod
from src.pipeline import Pipeline


def _model(name, api, media, files):
    return {"model": name, "uses_api_nodes": api, "member_files": list(files),
            "user_intent": {"media": media}}


TASKS = [
    {"task": "API / Partner Nodes - Reference to Video", "models": [
        _model("MiniMax H3", True, "video", ["api_minimax_hailuo03_r2v"])]},
    {"task": "Reference to Video", "models": [
        _model("MiniMax H3", False, "video", ["video_minimax_h3_r2v"])]},
    {"task": "Image Edit", "models": [
        _model("Flux 2 Klein", False, "image",
               ["image_flux2_klein_edit", "templates-all_in_one-image_edit_models"])]},
]
LIVE = {
    "api_minimax_hailuo03_r2v": "Hailuo 03 reference to video via the partner API.",
    "video_minimax_h3_r2v": "Local MiniMax H3 reference to video.",
    "image_flux2_klein_edit": "Flux 2 Klein edit.",
    "templates-all_in_one-image_edit_models": "Several edit models side by side.",
}


class LocalConstraintTest(unittest.TestCase):
    def setUp(self):
        self.p = object.__new__(Pipeline)
        self.p._recipe_tasks_cache = TASKS
        self._orig = tools_mod.get_workflow_catalog
        tools_mod.get_workflow_catalog = lambda: json.dumps(LIVE)

    def tearDown(self):
        tools_mod.get_workflow_catalog = self._orig

    def _catalog(self, request):
        return self.p._format_template_catalog(request)

    def test_local_wording(self):
        for q in ("the local MiniMax H3 workflow", "run it on my GPU",
                  "use an open-source model", "open source video"):
            self.assertTrue(self.p._wants_local(q), q)
        self.assertFalse(self.p._wants_local("make a MiniMax H3 video"))

    def test_named_model_with_local_drops_the_api_variant(self):
        cat = self._catalog("use the local MiniMax H3 reference-to-video workflow")
        self.assertIn("video_minimax_h3_r2v", cat)
        self.assertNotIn("api_minimax_hailuo03_r2v", cat)

    def test_without_local_both_stay(self):
        cat = self._catalog("use the MiniMax H3 reference-to-video workflow")
        self.assertIn("video_minimax_h3_r2v", cat)
        self.assertIn("api_minimax_hailuo03_r2v", cat)

    def test_local_scope_is_cached_apart_from_the_api_one(self):
        k1 = self.p._scope_recipes("local", "video", "MiniMax H3")[0]
        k2 = self.p._scope_recipes("api", "video", "MiniMax H3")[0]
        self.assertNotEqual(k1, k2)

    def test_template_is_api(self):
        self.assertTrue(self.p._template_is_api("api_minimax_hailuo03_r2v"))
        self.assertTrue(self.p._template_is_api("api_minimax_hailuo03_r2v.json"))
        self.assertFalse(self.p._template_is_api("video_minimax_h3_r2v"))

    def test_all_in_one_template_is_not_offered(self):
        cat = self._catalog("edit this image with flux 2 klein")
        self.assertIn("image_flux2_klein_edit", cat)
        self.assertNotIn("all_in_one", cat)


if __name__ == "__main__":
    unittest.main()
