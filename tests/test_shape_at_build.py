"""A ratio named in the request is set on the node that decides the shape.

"Generate this in 16:9 with Seedream, Nano Banana 2 and GPT Image" built three
workflows that rendered 1024x1024, 1024x1024 and 2160x3840. The ratio was in
the request, the briefing and each workflow's title; it was written nowhere it
counted. The builder only filled ``width`` / ``height`` inputs, and these nodes
set their shape from a menu that belongs to the chosen model
(``model.size_preset``, ``model.aspect_ratio``, ``model.size``) - members of a
dynamic combo, which the shape search did not look inside either.
"""

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from pipeline_stub import pipeline_stub
from src import pipeline
from src.pipeline import Pipeline
from src.utils import qa_repair as qr


def _dynamic(members: dict, key: str = "the-model") -> list:
    return ["COMFY_DYNAMICCOMBO_V3", {"options": [{"key": key, "inputs": {"required": members}}]}]


SCHEMAS = {
    "Seedream": {"input": {"required": {"model": _dynamic({
        "size_preset": ["COMBO", {"options": ["(1K) 1024x1024 (1:1)", "(1K) 1312x736 (16:9)",
                                              "(1K) 736x1312 (9:16)", "Custom"]}],
        "width": ["INT", {"default": 2048, "min": 1024}],
        "height": ["INT", {"default": 2048, "min": 1024}]})}},
        "output": ["IMAGE"]},
    "Banana": {"input": {"required": {"model": _dynamic({
        "aspect_ratio": ["COMBO", {"options": ["auto", "1:1", "16:9", "9:16"]}],
        "resolution": ["COMBO", {"options": ["1K", "2K", "4K"]}]})}},
        "output": ["IMAGE"]},
    "GptImage": {"input": {"required": {"model": _dynamic({
        "size": ["COMBO", {"options": ["auto", "1024x1024", "1024x1536", "2160x3840",
                                       "3840x2160", "2048x1152", "Custom"]}],
        "custom_width": ["INT", {"default": 1024}]})}},
        "output": ["IMAGE"]},
    "Latent": {"input": {"required": {"width": ["INT", {"default": 1024}], "height": ["INT", {"default": 1024}]}},
               "output": ["LATENT"]},
    "SaveImage": {"input": {"required": {"images": ["IMAGE"]}}, "output": [], "output_node": True},
}


def _schema(cls):
    return SCHEMAS.get(cls, {})


def _wf(cls, inputs):
    return {"1": {"class_type": cls, "inputs": {"model": "the-model", **inputs}},
            "2": {"class_type": "SaveImage", "inputs": {"images": ["1", 0]}}}


class InsideTheModelsOwnMenu(unittest.TestCase):

    def _fix(self, wf):
        fixes, problems = qr.plan_fixes(wf, {"aspect_ratio": "16:9"}, _schema)
        self.assertEqual(problems, [])
        self.assertEqual(len(fixes), 1)
        return fixes[0]

    def test_a_size_preset_that_belongs_to_the_model(self):
        fix = self._fix(_wf("Seedream", {"model.size_preset": "(1K) 1024x1024 (1:1)",
                                         "model.width": 2048, "model.height": 2048}))
        self.assertEqual((fix["param"], fix["to"]), ("model.size_preset", "(1K) 1312x736 (16:9)"))

    def test_a_ratio_that_belongs_to_the_model(self):
        fix = self._fix(_wf("Banana", {"model.aspect_ratio": "1:1", "model.resolution": "1K"}))
        self.assertEqual((fix["param"], fix["to"]), ("model.aspect_ratio", "16:9"))

    def test_a_size_list_and_the_cheapest_one_that_fits(self):
        fix = self._fix(_wf("GptImage", {"model.size": "2160x3840", "model.custom_width": 1024}))
        self.assertEqual((fix["param"], fix["to"]), ("model.size", "2048x1152"))

    def test_writing_it_lands_on_the_dotted_name(self):
        wf = _wf("Banana", {"model.aspect_ratio": "1:1", "model.resolution": "1K"})
        self.assertTrue(qr.apply_fix(wf, self._fix(wf)))
        self.assertEqual(wf["1"]["inputs"]["model.aspect_ratio"], "16:9")
        self.assertEqual(qr.plan_fixes(wf, {"aspect_ratio": "16:9"}, _schema), ([], []))

    def test_the_members_are_those_of_the_option_that_is_chosen(self):
        two = ["COMFY_DYNAMICCOMBO_V3", {"options": [
            {"key": "a", "inputs": {"required": {"aspect_ratio": ["COMBO", {"options": ["1:1", "16:9"]}]}}},
            {"key": "b", "inputs": {"required": {"steps": ["INT", {}]}}}]}]
        self.assertIn("model.aspect_ratio", qr._with_members({"model": two}, {"model": "a"}))
        self.assertNotIn("model.aspect_ratio", qr._with_members({"model": two}, {"model": "b"}))

    def test_plain_width_and_height_still_work(self):
        wf = {"1": {"class_type": "Latent", "inputs": {"width": 1024, "height": 1024}},
              "2": {"class_type": "SaveImage", "inputs": {"images": ["1", 0]}}}
        fixes, _problems = qr.plan_fixes(wf, {"aspect_ratio": "16:9"}, _schema)
        self.assertEqual(sorted(fixes[0]["to"]), ["height", "width"])


class TheRatioInTheRequest(unittest.TestCase):

    def test_it_is_read_from_the_words(self):
        for text, want in (("Black and white, 16:9 widescreen.", "16:9"),
                           ("portrait 9:16 for a phone", "9:16"),
                           ("2.39:1 scope, please", "2.39:1"),
                           ("a 16 : 9 frame", "16:9")):
            with self.subTest(text=text):
                self.assertEqual(pipeline._ratio_named_in(text), want)

    def test_a_time_a_pixel_size_or_two_ratios_are_not_one_ratio(self):
        for text in ("meet at 12:30", "make it 1024x1024", "16:9 or 9:16, you choose", "", None,
                     "version 1.2:3.4"):
            with self.subTest(text=text):
                self.assertEqual(pipeline._ratio_named_in(text), "")


class AtBuildTime(unittest.TestCase):

    def setUp(self):
        self.enterContext(mock.patch("src.utils.preflight._schema", side_effect=_schema))
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.path = Path(tmp.name) / "built.json"
        self.pipe = pipeline_stub()
        self.pipe._fit_requested_shape = Pipeline._fit_requested_shape.__get__(self.pipe)

    def _build(self, wf, request):
        self.path.write_text(json.dumps(wf), encoding="utf-8")
        result = {"status": "ready", "workflow_path": str(self.path)}
        self.pipe._fit_requested_shape(result, request)
        return result, json.loads(self.path.read_text(encoding="utf-8"))

    def test_the_built_file_carries_the_ratio(self):
        result, wf = self._build(_wf("Seedream", {"model.size_preset": "(1K) 1024x1024 (1:1)"}),
                                 "Seedream, black and white, 16:9 widescreen")
        self.assertEqual(wf["1"]["inputs"]["model.size_preset"], "(1K) 1312x736 (16:9)")
        self.assertEqual(result["shape"]["aspect_ratio"], "16:9")
        self.assertIn("model.size_preset", result["shape"]["set"][0])

    def test_no_ratio_in_the_request_changes_nothing(self):
        before = _wf("Seedream", {"model.size_preset": "(1K) 1024x1024 (1:1)"})
        result, wf = self._build(before, "a red apple")
        self.assertEqual(wf, before)
        self.assertNotIn("shape", result)

    def test_already_right_is_left_alone(self):
        before = _wf("Banana", {"model.aspect_ratio": "16:9"})
        result, wf = self._build(before, "16:9 please")
        self.assertEqual(wf, before)
        self.assertNotIn("shape", result)

    def test_a_graph_with_nothing_that_sets_its_shape_says_so(self):
        wf = {"1": {"class_type": "SaveImage", "inputs": {}}}
        result, after = self._build(wf, "16:9 please")
        self.assertEqual(after, wf)
        self.assertIn("NOT set", result["shape"]["note"])

    def test_a_build_that_is_not_ready_is_not_touched(self):
        result = {"status": "needs_fix", "workflow_path": str(self.path)}
        self.pipe._fit_requested_shape(result, "16:9")
        self.assertNotIn("shape", result)

    def test_prepare_workflow_does_it_before_it_answers(self):
        import inspect
        body = inspect.getsource(pipeline).split("async def prepare_workflow(", 1)[1].split("@_tool", 1)[0]
        self.assertLess(body.index("self._fit_requested_shape(result, request)"),
                        body.index("self._attach_built_summary(result)"))


if __name__ == "__main__":
    unittest.main()
