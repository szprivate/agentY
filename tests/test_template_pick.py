"""The researcher's template pick is held to what the request asks for.

Regression: "animate each starting frame with Seedance 2.5 first-frame, 480p, no audio, cut into
one film" must NOT pick template_seedance2_storyboard_to_video (it draws an 8-panel board,
drives Seedance 2.0 through a reference node, and renders 720p with audio).

    python -m unittest tests.test_template_pick
"""
import unittest

from src.utils import template_pick as tp

REQUEST = ("animate each starting frame with Seedance 2.5 first-frame, 480p, no audio, "
           "cut into one film")
CATALOG = (
    "## API / Partner Nodes - Image to Video\n"
    "  Seedance [API]:\n"
    "    - template_seedance2_storyboard_to_video: Generate an 8-panel storyboard from a simple text\n"
    "    - api_seedance2_5_flf2v: Generate a smooth video transition between a start frame and an end frame\n"
    "    - api_seedance2_0_flf2v: Upload the first and last frames of your video.\n"
    "    - not_a_template_in_the_corpus: whatever\n"
)


def _corpus_has(name):
    from agenty_core.utils import template_fit
    return template_fit.facts_for_template(name) is not None


class Regression(unittest.TestCase):
    def setUp(self):
        if not _corpus_has("template_seedance2_storyboard_to_video"):
            self.skipTest("official corpus not available")

    def test_the_storyboard_template_is_rejected_with_a_reason(self):
        why = tp.reject_reason("template_seedance2_storyboard_to_video", REQUEST)
        self.assertIn("storyboard", why)
        self.assertIn("2.0", why)

    def test_naming_the_template_yourself_overrides(self):
        req = REQUEST + " using template_seedance2_storyboard_to_video"
        self.assertTrue(tp.names_template(req, "template_seedance2_storyboard_to_video"))
        self.assertEqual(tp.reject_reason("template_seedance2_storyboard_to_video", req, user_named=True), "")

    def test_the_catalog_marks_the_wrong_ones_and_keeps_the_right_one_clean(self):
        out = tp.annotate_catalog(CATALOG, REQUEST).split("\n")
        by = {l.split(":")[0].strip("- ").strip(): l for l in out if l.startswith("    - ")}
        self.assertIn("NOT FOR THIS REQUEST", by["template_seedance2_storyboard_to_video"])
        self.assertIn("NOT FOR THIS REQUEST", by["api_seedance2_0_flf2v"])
        self.assertNotIn("NOT FOR THIS REQUEST", by["api_seedance2_5_flf2v"])
        self.assertIn("adjust", by["api_seedance2_5_flf2v"])        # 720p -> 480p
        self.assertEqual(by["not_a_template_in_the_corpus"], "    - not_a_template_in_the_corpus: whatever")

    def test_the_right_template_gets_its_adjustments(self):
        adj = tp.adjustments("api_seedance2_5_flf2v", REQUEST)
        self.assertTrue(any("480p" in a for a in adj))

    def test_no_request_no_annotation(self):
        self.assertEqual(tp.annotate_catalog(CATALOG, ""), CATALOG)

    def test_build_new_is_never_rejected(self):
        self.assertEqual(tp.reject_reason("build_new", REQUEST), "")


class SavedJobs(unittest.TestCase):
    def test_nothing_saved_nothing_shown(self):
        from unittest import mock
        with mock.patch("agenty_core.tools.local_templates.match_local", return_value=[]):
            self.assertEqual(tp.saved_jobs_block(REQUEST), "")


if __name__ == "__main__":
    unittest.main()


class InTheCatalogTheResearcherReads(unittest.TestCase):
    """Pipeline._format_template_catalog applies both per request, after its per-scope cache."""

    def setUp(self):
        if not _corpus_has("template_seedance2_storyboard_to_video"):
            self.skipTest("official corpus not available")
        import json
        import src.tools as tools_mod
        from src.pipeline import Pipeline
        self.p = object.__new__(Pipeline)
        self.p._recipe_tasks_cache = [
            {"task": "API / Partner Nodes - Image to Video", "models": [
                {"model": "Seedance", "uses_api_nodes": True, "user_intent": {"media": "video"},
                 "member_files": ["template_seedance2_storyboard_to_video", "api_seedance2_5_flf2v"]}]}]
        live = {"template_seedance2_storyboard_to_video": "8-panel storyboard then Seedance 2.0.",
                "api_seedance2_5_flf2v": "Seedance 2.5 first and last frame to video."}
        self._orig = tools_mod.get_workflow_catalog
        tools_mod.get_workflow_catalog = lambda: json.dumps(live)
        self.addCleanup(setattr, tools_mod, "get_workflow_catalog", self._orig)

    def test_the_block_is_marked_for_this_request_only(self):
        block = self.p._format_template_catalog(REQUEST, "image")
        line = next(l for l in block.split("\n") if "template_seedance2_storyboard_to_video:" in l)
        self.assertIn("NOT FOR THIS REQUEST", line)
        good = next(l for l in block.split("\n") if "api_seedance2_5_flf2v:" in l)
        self.assertNotIn("NOT FOR THIS REQUEST", good)
        # the cached scope block itself stays request-free
        plain = self.p._format_template_catalog("make a storyboard with Seedance 2.0", "image")
        self.assertNotIn("NOT FOR THIS REQUEST", next(
            l for l in plain.split("\n") if "template_seedance2_storyboard_to_video:" in l) or "")

    def test_matching_saved_jobs_come_first(self):
        from unittest import mock
        fake = [{"name": "local_seedance_ff", "description": "[API] Saved job via Seedance.", "score": 0.9,
                 "output_kind": "video", "verdict": "fit", "adjust": [],
                 "parameters": [{"node_id": "2", "input": "image", "kind": "image", "label": "x", "value": "a.png"}]}]
        with mock.patch("agenty_core.tools.local_templates.match_local", return_value=fake):
            block = self.p._format_template_catalog(REQUEST, "image")
        self.assertTrue(block.startswith("YOUR SAVED JOBS"))
        self.assertIn("local_seedance_ff", block)


class LocalRequests(unittest.TestCase):
    def test_an_explicit_local_ask_excludes_partner_jobs(self):
        from unittest import mock
        with mock.patch("agenty_core.tools.local_templates.catalog_block", return_value="") as cb:
            tp.saved_jobs_block("make this locally, offline, no api")
            tp.saved_jobs_block("make a video with Seedance 2.5")
        self.assertIs(cb.call_args_list[0].kwargs["local_only"], True)
        self.assertIs(cb.call_args_list[1].kwargs["local_only"], False)
