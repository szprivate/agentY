"""Fixing what a parameter decides, instead of re-rolling and hoping.

QA says an output is the wrong shape with certainty. The retry that followed
could do nothing about it: it rerolls the seed and rewrites the positive prompt,
and neither has ever changed an image's dimensions. So a 1:1 render against a
16:9 briefing burned a paid generation and came back failing with the identical
number.

What is held down here:

**The right node.** A graph can name a size in several places, and only one is
the shape the picture is *made* at. Rescaling on the way to the saver would
satisfy the ruler and misreport the render, so the walk takes the furthest
governing node upstream — the latent, not the resize.

**One shape, one decision.** Ratio and resolution planned separately overwrite
each other: 1024x1024 becomes 1344x768 for 16:9, then 1088x1088 for 1080p, and
the second edit throws the first away.

**Doing nothing, when nothing is wrong.** A graph already at 1920x1080 must not
be "fixed" to 1088 by a snapping rule. An edit reported on a correct graph
teaches people to distrust the ones that are real.

**And declining, when nothing can be done.** Sharpness and grain are properties
of a picture, not settings. Saying so is the point: a retry that cannot work
should not be paid for.

Schemas are stubbed. What needs testing is the choosing, not ComfyUI — though the
stubs are the real declarations, copied from a running install.

    python -m unittest discover -s tests
"""

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from src.utils.qa_repair import (apply_fix, describe_fix, evaluate_shape,
                                 governing_params, plan_fixes, ratio_of,
                                 short_side_of)

# Copied from a running ComfyUI's /object_info, trimmed to what is read here.
SCHEMAS = {
    "EmptyLatentImage": {"input": {"required": {
        "width": ["INT", {"default": 512}], "height": ["INT", {"default": 512}],
        "batch_size": ["INT", {"default": 1}]}}},
    "ImageScale": {"input": {"required": {
        "image": ["IMAGE"], "width": ["INT", {}], "height": ["INT", {}],
        "upscale_method": [["nearest-exact", "bilinear"]], "crop": [["disabled"]]}}},
    "KSampler": {"input": {"required": {
        "latent_image": ["LATENT"], "seed": ["INT", {}], "steps": ["INT", {}]}}},
    "VAEDecode": {"input": {"required": {"samples": ["LATENT"]}}},
    "SaveImage": {"input": {"required": {"images": ["IMAGE"]}}},
    "SaveVideo": {"input": {"required": {"video": ["VIDEO"]}}},
    "KlingTextToVideoNode": {"input": {"required": {
        "prompt": ["STRING", {}],
        "aspect_ratio": ["COMBO", {"default": "16:9",
                                   "options": ["16:9", "9:16", "1:1"]}],
        "mode": ["COMBO", {"options": ["std", "pro"]}]}}},
    "ByteDanceSeedreamNode": {"input": {"required": {
        "prompt": ["STRING", {}],
        "size_preset": ["COMBO", {"options": ["1024x1024", "1280x720", "1920x1080",
                                              "720x1280", "2048x2048"]}]}}},
}


def schema_of(cls):
    return SCHEMAS.get(str(cls), {})


def sd(width=1024, height=1024, scale=(512, 512)):
    """A diffusion chain: latent → sampler → decode → resize → save."""
    return {
        "1": {"class_type": "EmptyLatentImage",
              "inputs": {"width": width, "height": height, "batch_size": 1}},
        "2": {"class_type": "KSampler", "inputs": {"latent_image": ["1", 0], "seed": 7}},
        "3": {"class_type": "VAEDecode", "inputs": {"samples": ["2", 0]}},
        "4": {"class_type": "ImageScale",
              "inputs": {"image": ["3", 0], "width": scale[0], "height": scale[1]}},
        "5": {"class_type": "SaveImage", "inputs": {"images": ["4", 0]}},
    }


def kling(ratio="1:1"):
    return {"10": {"class_type": "KlingTextToVideoNode",
                   "inputs": {"prompt": "a car", "aspect_ratio": ratio}},
            "11": {"class_type": "SaveVideo", "inputs": {"video": ["10", 0]}}}


def fixes_for(graph, technical):
    return plan_fixes(graph, technical, schema_of)


class ReadingLabelsTest(unittest.TestCase):

    def test_a_ratio_label_is_a_number(self):
        self.assertAlmostEqual(ratio_of("16:9"), 16 / 9, places=3)
        self.assertAlmostEqual(ratio_of("1024x576"), 16 / 9, places=3)
        self.assertAlmostEqual(ratio_of("2.39:1"), 2.39, places=3)

    def test_a_label_that_names_no_shape_is_none(self):
        for junk in ("pro", "", None, "std", "16:0"):
            self.assertIsNone(ratio_of(junk), repr(junk))

    def test_a_size_label_gives_its_short_side(self):
        self.assertEqual(short_side_of("1920x1080"), 1080)
        self.assertEqual(short_side_of("720x1280"), 720)
        self.assertIsNone(short_side_of("16:9"))


class WhichNodeTest(unittest.TestCase):

    def test_the_latent_outranks_the_resize(self):
        """The size the picture is MADE at, not the one it is squashed to."""
        rows = governing_params(sd(), schema_of)
        self.assertEqual([r["node_id"] for r in rows], ["1", "4"])
        self.assertGreater(rows[0]["depth"], rows[1]["depth"])

    def test_a_generator_carrying_the_parameter_itself_is_the_one(self):
        rows = governing_params(kling(), schema_of)
        self.assertEqual([(r["node_id"], r["kind"], r["param"]) for r in rows],
                         [("10", "ratio", "aspect_ratio")])

    def test_the_order_is_computed_not_inherited_from_the_graph(self):
        """The same graph written the other way round must choose the same node."""
        g = sd()
        backwards = {k: g[k] for k in reversed(list(g))}
        self.assertEqual([r["node_id"] for r in governing_params(backwards, schema_of)],
                         ["1", "4"])

    def test_a_wired_menu_is_not_a_knob_to_turn_either(self):
        """Same reason as width/height: something upstream is deciding it."""
        g = kling()
        g["10"]["inputs"]["aspect_ratio"] = ["12", 0]
        self.assertEqual(governing_params(g, schema_of), [])

    def test_a_wired_size_is_not_a_knob_to_turn(self):
        """Something upstream is deciding it; overwriting would drop that link."""
        g = sd()
        g["1"]["inputs"]["width"] = ["9", 0]
        self.assertNotIn("1", [r["node_id"] for r in governing_params(g, schema_of)])

    def test_a_graph_of_nodes_we_cannot_ask_about_yields_nothing(self):
        self.assertEqual(governing_params(sd(), lambda cls: {}), [])

    def test_junk_is_survivable(self):
        for junk in (None, "", [], 7):
            self.assertEqual(governing_params(junk, schema_of), [], repr(junk))


class ChoosingTheValueTest(unittest.TestCase):

    def test_a_square_latent_is_reshaped_to_the_ratio_asked_for(self):
        (fix,), un = fixes_for(sd(), {"aspect_ratio": "16:9"})
        self.assertEqual(fix["node_id"], "1")
        w, h = fix["to"]["width"], fix["to"]["height"]
        self.assertAlmostEqual(w / h, 16 / 9, places=2)
        self.assertEqual(un, [])

    def test_pixel_count_is_roughly_preserved(self):
        """The user chose a render size; reshaping should not double the bill."""
        (fix,), _ = fixes_for(sd(1024, 1024), {"aspect_ratio": "16:9"})
        before, after = 1024 * 1024, fix["to"]["width"] * fix["to"]["height"]
        self.assertLess(abs(after - before) / before, 0.25)

    def test_both_requirements_are_met_at_once(self):
        """Planned separately they overwrite each other."""
        (fix,), _ = fixes_for(sd(), {"aspect_ratio": "16:9", "resolution": "1080p"})
        w, h = fix["to"]["width"], fix["to"]["height"]
        self.assertAlmostEqual(w / h, 16 / 9, places=2)
        self.assertGreaterEqual(min(w, h), 1080)

    def test_a_size_already_correct_is_left_alone(self):
        self.assertEqual(fixes_for(sd(1920, 1080),
                                   {"aspect_ratio": "16:9", "resolution": "1080p"}),
                         ([], []))

    def test_a_size_within_tolerance_is_already_correct(self):
        """1920x1088 is 1.7647 where 16:9 is 1.7778 — inside what QA accepts.

        Reshaping it would report a fix for a graph that was going to pass, and
        the tolerance has to be the SAME one the check uses or the two disagree
        about the identical picture.
        """
        self.assertEqual(fixes_for(sd(1920, 1088), {"aspect_ratio": "16:9"}), ([], []))

    def test_a_size_outside_tolerance_is_not(self):
        self.assertNotEqual(fixes_for(sd(1920, 1200), {"aspect_ratio": "16:9"}), ([], []))

    def test_raising_the_resolution_keeps_the_shape(self):
        (fix,), _ = fixes_for(sd(1280, 720), {"resolution": "1080p"})
        w, h = fix["to"]["width"], fix["to"]["height"]
        self.assertGreaterEqual(min(w, h), 1080)
        self.assertAlmostEqual(w / h, 1280 / 720, places=2)

    def test_computed_sides_stay_usable(self):
        """A latent side that is not a multiple of 8 does not render at all."""
        for want in ({"aspect_ratio": "2.39:1"}, {"aspect_ratio": "3:2"},
                     {"resolution": "2160p (4K)"}, {"aspect_ratio": "9:16"}):
            (fix,), _ = fixes_for(sd(), want)
            for side in fix["to"].values():
                self.assertEqual(side % 8, 0, f"{want} -> {fix['to']}")
                self.assertGreaterEqual(side, 256)

    def test_a_menu_is_read_rather_than_computed(self):
        (fix,), _ = fixes_for(kling(), {"aspect_ratio": "16:9"})
        self.assertEqual((fix["param"], fix["from"], fix["to"]),
                         ("aspect_ratio", "1:1", "16:9"))

    def test_a_menu_already_on_the_right_option_is_left_alone(self):
        self.assertEqual(fixes_for(kling("16:9"), {"aspect_ratio": "16:9"}), ([], []))

    def test_a_menu_with_nothing_that_fits_is_reported_not_forced(self):
        _fixes, un = fixes_for(kling(), {"aspect_ratio": "2.39:1"})
        self.assertEqual(un, ["aspect_ratio"])

    def test_the_cheapest_option_that_qualifies_wins(self):
        """Spending more than was asked for is not ours to decide."""
        g = {"1": {"class_type": "ByteDanceSeedreamNode",
                   "inputs": {"prompt": "x", "size_preset": "1024x1024"}},
             "2": {"class_type": "SaveImage", "inputs": {"images": ["1", 0]}}}
        (fix,), _ = fixes_for(g, {"aspect_ratio": "16:9"})
        self.assertEqual(fix["to"], "1280x720")

    def test_a_size_menu_can_answer_both_requirements(self):
        g = {"1": {"class_type": "ByteDanceSeedreamNode",
                   "inputs": {"prompt": "x", "size_preset": "1024x1024"}},
             "2": {"class_type": "SaveImage", "inputs": {"images": ["1", 0]}}}
        (fix,), _ = fixes_for(g, {"aspect_ratio": "16:9", "resolution": "1080p"})
        self.assertEqual(fix["to"], "1920x1080")


class WhatItWillNotDoTest(unittest.TestCase):

    def test_a_picture_property_is_never_treated_as_a_setting(self):
        for control, want in (("sharpness", "must be sharp"), ("grain", "must be clean"),
                              ("no_clipping", True), ("likeness", "must match the reference face"),
                              ("no_black_frames", True)):
            self.assertEqual(fixes_for(sd(), {control: want}), ([], []),
                             f"{control} is not a parameter")

    def test_an_unset_control_asks_for_nothing(self):
        self.assertEqual(fixes_for(sd(), {"aspect_ratio": "any", "resolution": ""}),
                         ([], []))
        self.assertEqual(fixes_for(sd(), {}), ([], []))

    def test_a_graph_with_no_size_anywhere_reports_it(self):
        g = {"1": {"class_type": "KSampler", "inputs": {"seed": 1}},
             "2": {"class_type": "SaveImage", "inputs": {"images": ["1", 0]}}}
        _fixes, un = fixes_for(g, {"aspect_ratio": "16:9"})
        self.assertEqual(un, ["aspect_ratio"])

    def test_the_downstream_resize_is_never_the_answer(self):
        """Rescaling satisfies the ruler and misreports the render."""
        (fix,), _ = fixes_for(sd(), {"aspect_ratio": "16:9"})
        self.assertNotEqual(fix["node_id"], "4")


class ApplyingTest(unittest.TestCase):

    def test_a_dims_fix_lands_on_the_node(self):
        g = sd()
        (fix,), _ = fixes_for(g, {"aspect_ratio": "16:9"})
        self.assertTrue(apply_fix(g, fix))
        self.assertEqual(g["1"]["inputs"]["width"], fix["to"]["width"])
        self.assertEqual(g["4"]["inputs"]["width"], 512)      # resize untouched

    def test_a_menu_fix_lands_on_the_node(self):
        g = kling()
        (fix,), _ = fixes_for(g, {"aspect_ratio": "16:9"})
        self.assertTrue(apply_fix(g, fix))
        self.assertEqual(g["10"]["inputs"]["aspect_ratio"], "16:9")

    def test_applying_to_a_node_that_is_gone_is_not_a_crash(self):
        self.assertFalse(apply_fix(sd(), {"node_id": "999", "param": "width", "to": 1}))
        self.assertFalse(apply_fix(sd(), {}))

    def test_the_description_names_the_node_and_the_reason(self):
        (fix,), _ = fixes_for(sd(), {"aspect_ratio": "16:9"})
        text = describe_fix(fix)
        self.assertIn("node 1", text)
        self.assertIn("EmptyLatentImage", text)
        self.assertIn("16:9", text)
        self.assertEqual(describe_fix({}), "")


class BeforeTheRunTest(unittest.TestCase):
    """The executor fits the graph before submitting, and says what it changed."""

    def _fit(self, graph, technical):
        from src.executor import _fit_to_briefing
        from src.utils.qa import QaBriefing
        d = Path(tempfile.mkdtemp())
        wf = d / "workflow.json"
        wf.write_text(json.dumps(graph), encoding="utf-8")
        briefing = QaBriefing(criteria="x", technical=dict(technical))
        with mock.patch("src.utils.preflight._schema", schema_of):
            path, lines = _fit_to_briefing(str(wf), briefing)
        return wf, Path(path), lines

    def test_the_graph_is_fitted_before_anything_is_paid_for(self):
        src, out, lines = self._fit(sd(), {"aspect_ratio": "16:9"})
        self.assertNotEqual(src, out)
        fitted = json.loads(out.read_text(encoding="utf-8"))
        w, h = fitted["1"]["inputs"]["width"], fitted["1"]["inputs"]["height"]
        self.assertAlmostEqual(w / h, 16 / 9, places=2)
        self.assertTrue(any("Fitted" in ln for ln in lines))

    def test_the_original_workflow_is_left_exactly_as_it_was(self):
        src, out, _ = self._fit(sd(), {"aspect_ratio": "16:9"})
        self.assertEqual(json.loads(src.read_text(encoding="utf-8")), sd())
        self.assertNotEqual(src, out)

    def test_a_graph_already_right_is_submitted_unchanged(self):
        src, out, lines = self._fit(sd(1920, 1080), {"aspect_ratio": "16:9"})
        self.assertEqual(src, out)
        self.assertEqual(lines, [])

    def test_a_requirement_nothing_governs_is_warned_about_not_forced(self):
        g = {"1": {"class_type": "KSampler", "inputs": {"seed": 1}},
             "2": {"class_type": "SaveImage", "inputs": {"images": ["1", 0]}}}
        src, out, lines = self._fit(g, {"aspect_ratio": "16:9"})
        self.assertEqual(src, out)
        self.assertTrue(any("nothing in this graph sets it" in ln for ln in lines))

    def test_no_briefing_means_no_fitting(self):
        from src.executor import _fit_to_briefing
        d = Path(tempfile.mkdtemp())
        wf = d / "w.json"
        wf.write_text(json.dumps(sd()), encoding="utf-8")
        self.assertEqual(_fit_to_briefing(str(wf), None), (str(wf), []))

    def test_an_unreadable_workflow_submits_unchanged_rather_than_failing(self):
        from src.executor import _fit_to_briefing
        from src.utils.qa import QaBriefing
        b = QaBriefing(criteria="x", technical={"aspect_ratio": "16:9"})
        self.assertEqual(_fit_to_briefing("/no/such/file.json", b),
                         ("/no/such/file.json", []))


class AfterAFailedCheckTest(unittest.TestCase):
    """The retry: fix the parameter, or decline rather than pay to be told again."""

    def _retry(self, graph, technical, failures=("aspect ratio 16:9 — wrong shape",)):
        import asyncio

        from src.pipeline import Pipeline
        from src.utils.qa import QaBriefing
        d = Path(tempfile.mkdtemp())
        wf = d / "workflow.json"
        wf.write_text(json.dumps(graph), encoding="utf-8")
        p = object.__new__(Pipeline)
        p._verbose = False
        p._qa_briefing = QaBriefing(criteria="x", technical=dict(technical))
        p._last_brainbriefing_json = "{}"
        p._reroll_seeds = lambda g: False        # isolate the parameter lever
        with mock.patch("src.utils.preflight._schema", schema_of):
            out = asyncio.run(p._qa_retry(str(wf), {"fail_details": [{"failed": list(failures)}]}))
        return wf, out

    def test_it_changes_the_parameter_that_decides_the_shape(self):
        _wf, out = self._retry(sd(), {"aspect_ratio": "16:9"})
        self.assertEqual(out["status"], "ready")
        fixed = json.loads(Path(out["workflow_path"]).read_text(encoding="utf-8"))
        w, h = fixed["1"]["inputs"]["width"], fixed["1"]["inputs"]["height"]
        self.assertAlmostEqual(w / h, 16 / 9, places=2)

    def test_the_rejected_workflow_is_kept_for_comparison(self):
        wf, out = self._retry(sd(), {"aspect_ratio": "16:9"})
        self.assertNotEqual(str(wf), out["workflow_path"])
        self.assertEqual(json.loads(wf.read_text(encoding="utf-8")), sd())

    def test_it_declines_when_nothing_in_the_graph_decides_the_failure(self):
        """The verdict is already known; a generation cannot change it."""
        g = {"1": {"class_type": "KSampler", "inputs": {"seed": 1}},
             "2": {"class_type": "SaveImage", "inputs": {"images": ["1", 0]}}}
        _wf, out = self._retry(g, {"aspect_ratio": "16:9"})
        self.assertEqual(out["status"], "failed")
        self.assertIn("not retrying", out["error"])
        self.assertIn("aspect ratio", out["error"])

    def test_a_soft_failure_still_gets_the_old_levers(self):
        """Sharpness is not a setting, but a different sample may well be sharper."""
        from src.pipeline import Pipeline
        import asyncio
        from src.utils.qa import QaBriefing
        d = Path(tempfile.mkdtemp())
        wf = d / "w.json"
        wf.write_text(json.dumps(sd()), encoding="utf-8")
        p = object.__new__(Pipeline)
        p._verbose = False
        p._qa_briefing = QaBriefing(criteria="x", technical={"sharpness": "must be sharp"})
        p._last_brainbriefing_json = "{}"
        p._reroll_seeds = lambda g: True
        with mock.patch("src.utils.preflight._schema", schema_of):
            out = asyncio.run(p._qa_retry(str(wf), {"fail_details": [{"failed": ["soft"]}]}))
        self.assertEqual(out["status"], "ready")


if __name__ == "__main__":
    unittest.main()
