"""Tests for deterministic catalog scoping (Pipeline.resolve_catalog_scope et al).

Self-contained: a fake recipe tree and a fake live catalog are injected, so these
never touch the real corpus and stay fast.

    python -m unittest discover -s tests
"""

import json
import unittest

import src.tools as tools_mod
from src.pipeline import Pipeline


def _model(name, api, media, files):
    return {"model": name, "uses_api_nodes": api, "member_files": list(files),
            "user_intent": {"media": media}}


TASKS = [
    {"task": "API / Partner Nodes - Image Edit", "models": [
        _model("Nano-Banana", True, "image", ["api_nano_banana_edit", "api_nano_banana_t2i"]),
        _model("Magnific", True, "image", ["api_magnific_image_relight"]),
        _model("Generic", True, "image", ["api_some_generic_edit"]),
    ]},
    {"task": "API / Partner Nodes - Image to Video", "models": [
        _model("Kling", True, "video", ["api_kling_o3_i2v"]),
        _model("WAN 2.2", True, "video", ["api_wan22_i2v"]),
    ]},
    {"task": "Image to Video", "models": [
        _model("WAN 2.2", False, "video", ["video_wan2_2_14B_i2v", "video_wan2_2_14B_flf2v"]),
        _model("Anima", False, "image", ["image_anima_preview"]),
    ]},
]
LIVE = {
    "api_nano_banana_edit": "Edit an image with Nano Banana.",
    "api_nano_banana_t2i": "Generate an image with Nano Banana.",
    "api_magnific_image_relight": "Relight an image with Magnific.",
    "api_some_generic_edit": "Generic partner image edit.",
    "api_kling_o3_i2v": "Kling image to video.",
    "api_wan22_i2v": "WAN 2.2 image to video via the partner API.",
    "video_wan2_2_14B_i2v": "Local WAN 2.2 image to video.",
    "video_wan2_2_14B_flf2v": "Local WAN 2.2 first/last frame to video.",
    "image_anima_preview": "Anima text to image.",
    "orphan_template": "Referenced by no recipe at all.",
}


class CatalogScopeTest(unittest.TestCase):
    def setUp(self):
        self.p = object.__new__(Pipeline)      # the catalog path is a pure read
        self.p._recipe_tasks_cache = TASKS
        self._orig = tools_mod.get_workflow_catalog
        tools_mod.get_workflow_catalog = lambda: json.dumps(LIVE)

    def tearDown(self):
        tools_mod.get_workflow_catalog = self._orig

    # -- resolution ------------------------------------------------------- #
    def test_partner_api_is_the_default_execution(self):
        self.assertEqual(self.p.resolve_catalog_scope("make me an image")[0], "api")

    def test_explicit_local_flips_execution(self):
        for q in ("a local wan 2.2 workflow", "offline please", "no api nodes"):
            self.assertEqual(self.p.resolve_catalog_scope(q)[0], "local", q)

    def test_media_from_request(self):
        cases = {"make a clip": "video", "upscale this photo": "image",
                 "turn it into a 3d mesh": "3d", "write me a song": "audio",
                 "do the thing": None}
        for q, want in cases.items():
            self.assertEqual(self.p.resolve_catalog_scope(q)[1], want, q)

    def test_media_from_staged_input_when_request_is_silent(self):
        self.assertEqual(self.p.resolve_catalog_scope("make something", "image")[1], "image")

    def test_model_matched_across_separator_styles(self):
        for q in ("wan2.2 workflow", "WAN 2.2 please", "a wan-2-2 graph"):
            self.assertEqual(self.p.resolve_catalog_scope(q)[2], "WAN 2.2", q)

    def test_longest_model_name_wins(self):
        # "WAN 2.2" must beat a bare "WAN" prefix match.
        self.p._catalog_models_cache = None
        self.p._recipe_tasks_cache = TASKS + [
            {"task": "Text to Video", "models": [_model("WAN", False, "video", ["video_wan"])]}]
        self.assertEqual(self.p.resolve_catalog_scope("wan 2.2 i2v")[2], "WAN 2.2")
        self.p._recipe_tasks_cache, self.p._catalog_models_cache = TASKS, None

    def test_word_boundaries_are_respected(self):
        # The classic trap: a stripped view fuses words and "animation" matches "Anima".
        self.assertIsNone(self.p.resolve_catalog_scope("make an animation")[2])

    def test_generic_is_never_matched_as_a_model(self):
        self.assertIsNone(self.p.resolve_catalog_scope("a generic image edit")[2])

    # -- scoping ---------------------------------------------------------- #
    def test_named_model_outranks_the_api_default(self):
        """The v1 bug: "build me a wan 2.2 workflow" keyed API+WAN 2.2, returned the
        single partner recipe and hid both local ones."""
        key, tasks, note = self.p._scope_recipes(*self.p.resolve_catalog_scope(
            "can you build me a wan 2.2. workflow"))
        self.assertEqual(key, "model:WAN 2.2")
        files = {f for t in tasks for m in t["models"] for f in m["member_files"]}
        self.assertIn("video_wan2_2_14B_i2v", files)     # local kept
        self.assertIn("api_wan22_i2v", files)            # partner kept
        self.assertIn("WAN 2.2", note)

    def test_named_model_survives_a_wrong_media_guess(self):
        # "shot" reads as video; Magnific is image-only. The model must still win.
        key, tasks, _ = self.p._scope_recipes(*self.p.resolve_catalog_scope(
            "relight this shot with magnific"))
        self.assertEqual(key, "model:Magnific")
        self.assertEqual([m["model"] for t in tasks for m in t["models"]], ["Magnific"])

    def test_falls_back_to_the_execution_media_bucket(self):
        key, tasks, note = self.p._scope_recipes(*self.p.resolve_catalog_scope(
            "make me a video from this"))
        self.assertEqual(key, "api:video")
        self.assertTrue(all(m["uses_api_nodes"] for t in tasks for m in t["models"]))
        self.assertIn("partner-API video", note)

    def test_unresolvable_request_falls_back_to_the_index(self):
        key, tasks, _ = self.p._scope_recipes(*self.p.resolve_catalog_scope(
            "I need something for the client"))
        self.assertEqual(key, "index")
        self.assertIsNone(tasks)

    def test_empty_bucket_widens_to_the_index_rather_than_rendering_nothing(self):
        key, tasks, _ = self.p._scope_recipes("local", "audio", None)
        self.assertEqual(key, "index")
        self.assertIsNone(tasks)

    # -- rendering -------------------------------------------------------- #
    def test_full_render_keeps_the_lossless_other_bucket(self):
        block = self.p._format_recipe_catalog()
        self.assertIn("## Other", block)
        self.assertIn("orphan_template", block)

    def test_scoped_render_drops_other_and_names_the_filter(self):
        _, tasks, note = self.p._scope_recipes("api", "image", "Nano-Banana")
        block = self.p._format_recipe_catalog(tasks, scope_note=note)
        self.assertNotIn("## Other", block)
        self.assertNotIn("orphan_template", block)
        self.assertIn("api_nano_banana_edit", block)
        self.assertIn("filtered to", block)
        self.assertIn("get_workflow_catalog", block)   # the way out stays signposted

    def test_scoped_render_lists_strictly_fewer_leaves(self):
        # Asserted on leaf count, not character count: the header is fixed overhead
        # and dominates a fixture this small. The token ratio that actually matters
        # is a property of the real corpus, not of ten fake templates.
        def leaves(block):
            return {ln.strip().split(":")[0].lstrip("- ")
                    for ln in block.splitlines() if ln.startswith("    - ")}

        full = leaves(self.p._format_recipe_catalog())
        _, tasks, note = self.p._scope_recipes("api", "image", "Magnific")
        scoped = leaves(self.p._format_recipe_catalog(tasks, scope_note=note))
        self.assertEqual(scoped, {"api_magnific_image_relight"})
        self.assertLess(len(scoped), len(full))
        self.assertTrue(scoped < full)   # scoping only ever removes

    def test_index_names_models_but_no_templates(self):
        idx = self.p._format_catalog_index()
        self.assertIn("[API] Kling", idx)
        self.assertNotIn("api_kling_o3_i2v", idx)

    def test_block_cache_is_keyed_per_scope(self):
        self.p._catalog_block_cache = None
        a = self.p._format_template_catalog("upscale this with magnific")
        b = self.p._format_template_catalog("make a kling video")
        self.assertNotEqual(a, b)
        self.assertEqual(set(self.p._catalog_block_cache), {"model:Magnific", "model:Kling"})
        self.assertEqual(self.p._format_template_catalog("relight with magnific please"), a)

    def test_no_request_still_renders_the_full_tree(self):
        self.p._catalog_block_cache = None
        self.assertIn("## Other", self.p._format_template_catalog())


if __name__ == "__main__":
    unittest.main()
