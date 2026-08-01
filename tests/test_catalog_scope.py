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


TASK_TASKS = TASKS + [
    {"task": "Text to Video", "models": [_model("WAN", False, "video", ["video_wan_t2v"])]},
    {"task": "API / Partner Nodes - Text to Video", "models": [
        _model("Veo", True, "video", ["api_veo3"])]},
    {"task": "Video to Video", "models": [_model("WAN VACE", False, "video", ["video_vace"])]},
    {"task": "Upscale", "models": [_model("RealESRGAN", False, "image", ["image_upscale"])]},
    {"task": "API / Partner Nodes - Upscale", "models": [
        _model("Topaz", True, "image", ["api_topaz_image_enhance"])]},
    {"task": "Inpaint / Outpaint", "models": [
        _model("Flux", False, "image", ["image_inpaint_flux"])]},
]


class TaskKeyTest(unittest.TestCase):
    """The scope level between 'model named' and 'media bucket'."""

    def setUp(self):
        self.p = object.__new__(Pipeline)
        self.p._recipe_tasks_cache = TASK_TASKS
        self._orig = tools_mod.get_workflow_catalog
        tools_mod.get_workflow_catalog = lambda: json.dumps(LIVE)

    def tearDown(self):
        tools_mod.get_workflow_catalog = self._orig

    def test_direction_uses_what_the_user_has(self):
        """The naive matcher's bug: "make a video" is Text to Video, not Video to
        Video — both task-name tokens are "video", so wording alone cannot tell
        them apart. What separates them is whether anything is staged."""
        self.assertEqual(sorted(self.p._resolve_tasks("make a video", "video", "")),
                         ["API / Partner Nodes - Text to Video", "Text to Video"])
        self.assertEqual(sorted(self.p._resolve_tasks("make a video", "video", "image")),
                         ["API / Partner Nodes - Image to Video", "Image to Video"])

    def test_keywords_are_matched_as_stems(self):
        for q in ("upscale this", "image upscaling templates", "it was upscaled"):
            self.assertEqual(sorted(self.p._resolve_tasks(q, "image", "")),
                             ["API / Partner Nodes - Upscale", "Upscale"], q)

    def test_a_named_capability_beats_the_direction_guess(self):
        # "which image upscaling templates" reads as Text to Image by direction;
        # Upscale is what was actually asked for.
        hits = self.p._resolve_tasks(
            "Which image upscaling workflow templates do I have?", "image", "")
        self.assertNotIn("Text to Image", hits)
        self.assertIn("Upscale", hits)

    def test_phrases_are_matched_whole(self):
        self.assertIn("Inpaint / Outpaint",
                      self.p._resolve_tasks("remove object from this photo", "image", "image"))

    def test_vague_verbs_match_nothing(self):
        for q in ("change the prompt in this node", "make something nice", "do the thing"):
            self.assertEqual(self.p._resolve_tasks(q, None, ""), [], q)

    def test_no_task_match_falls_through_to_the_media_bucket(self):
        # No loader task exists, so a loader request must NOT be forced into one.
        tasks = self.p._resolve_tasks("a workflow that loads all of these images",
                                      "image", "image")
        key, sel, _ = self.p._scope_recipes("api", "image", None, tasks)
        self.assertEqual(key, "api:image")

    def test_scope_keeps_both_api_and_local_variants(self):
        tasks = self.p._resolve_tasks("upscale this", "image", "")
        key, sel, note = self.p._scope_recipes("api", "image", None, tasks)
        self.assertTrue(key.startswith("task:"))
        names = {t["task"] for t in sel}
        self.assertEqual(names, {"Upscale", "API / Partner Nodes - Upscale"})
        self.assertIn("Upscale", note)

    def test_a_named_model_still_outranks_the_task(self):
        tasks = self.p._resolve_tasks("upscale this with topaz", "image", "")
        key, _, _ = self.p._scope_recipes("api", "image", "Topaz", tasks)
        self.assertEqual(key, "model:Topaz")


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


FLF_TASKS = TASK_TASKS + [
    {"task": "API / Partner Nodes - First / Last Frame to Video", "models": [
        _model("Generic", True, "video",
               ["api_seedance2_0_flf2v", "api_bytedance_seedance1_5_flf2v"]),
        _model("Kling", True, "video", ["api_kling_v3_flf2v"]),
    ]},
    {"task": "First / Last Frame to Video", "models": [
        _model("WAN 2.2", False, "video", ["video_wan2_2_14B_flf2v"])]},
    # Shares a prefix with api_bytedance_seedance1_5_flf2v above — naming the
    # long one must not drag this one's task in with it.
    {"task": "API / Partner Nodes - Image Edit", "models": [
        _model("Generic", True, "image", ["api_bytedance_seed"])]},
]


class NamedTemplateAndFrameWordingTest(unittest.TestCase):
    """A first/last-frame request must reach the flf2v templates.

    Both routes into that task were blind: the task name is not "<in> to <out>",
    so direction could never match it, and no wording mapped onto it — so "a
    video from a start and an end frame" resolved to Text to Video, and the
    researcher reported (correctly, for the scope it was handed) that no flf2v
    template existed. Naming one outright did not help either, because a
    template name is not a task name.
    """

    def setUp(self):
        self.p = object.__new__(Pipeline)
        self.p._recipe_tasks_cache = FLF_TASKS

    def _tasks(self, request, staged=""):
        media = self.p.resolve_catalog_scope(request, staged)[1]
        return self.p._resolve_tasks(request, media, staged)

    def test_naming_a_template_wins_outright(self):
        self.assertEqual(
            self._tasks("Use api_seedance2_0_flf2v to make a 5s video"),
            ["API / Partner Nodes - First / Last Frame to Video"])

    def test_naming_a_template_beats_the_media_direction(self):
        # A staged image + "video" would otherwise read as Image to Video.
        self.assertEqual(
            self._tasks("use api_seedance2_0_flf2v for a video", "master_image"),
            ["API / Partner Nodes - First / Last Frame to Video"])

    def test_a_name_inside_a_longer_name_does_not_match(self):
        tasks = self._tasks("use api_bytedance_seedance1_5_flf2v please")
        self.assertEqual(tasks, ["API / Partner Nodes - First / Last Frame to Video"])

    def test_trailing_punctuation_is_still_a_boundary(self):
        self.assertEqual(self._tasks("please use api_seedance2_0_flf2v."),
                         ["API / Partner Nodes - First / Last Frame to Video"])

    def test_short_names_never_match_prose(self):
        self.p._recipe_tasks_cache = [
            {"task": "Tiny", "models": [_model("X", True, "image", ["edit"])]}]
        self.assertEqual(self.p._tasks_naming_templates("please edit this photo"), [])

    def test_frame_wording_reaches_the_task_without_a_name(self):
        self.assertEqual(
            sorted(self._tasks("make a video from a start frame and an end frame")),
            ["API / Partner Nodes - First / Last Frame to Video",
             "First / Last Frame to Video"])

    def test_first_and_last_frame_phrasing(self):
        self.assertIn("API / Partner Nodes - First / Last Frame to Video",
                      self._tasks("video from the first and last frame"))

    def test_plain_video_request_is_unchanged(self):
        self.assertEqual(sorted(self._tasks("make me a video of a cat")),
                         ["API / Partner Nodes - Text to Video", "Text to Video"])

    def test_empty_request_names_nothing(self):
        self.assertEqual(self.p._tasks_naming_templates(""), [])


if __name__ == "__main__":
    unittest.main()
