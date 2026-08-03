"""Tests for annotate_image's argument handling and SAM3 path discovery.

Everything here is the deterministic half: parsing what the model passed in, and
finding files on disk. No GPU, no model load, no ComfyUI — the grounding call
itself is exercised live, not here.

    python -m unittest discover -s tests
"""

import json
import os
import tempfile
import unittest
from unittest import mock

from src.tools import annotate as A
from src.utils import image_locate as L


class SplitTargetsTest(unittest.TestCase):
    def test_a_single_phrase(self):
        self.assertEqual(A._split_targets("bolt"), ["bolt"])

    def test_comma_separated(self):
        self.assertEqual(A._split_targets("bolt, seaweed, sky"), ["bolt", "seaweed", "sky"])

    def test_a_json_array(self):
        self.assertEqual(A._split_targets('["red car", "traffic light"]'),
                         ["red car", "traffic light"])

    def test_a_real_list(self):
        self.assertEqual(A._split_targets(["a", "b"]), ["a", "b"])

    def test_semicolons_win_over_commas(self):
        # "a, b; c" is two targets, one of which contains a comma.
        self.assertEqual(A._split_targets("big, red car; traffic light"),
                         ["big, red car", "traffic light"])

    def test_blank_and_none(self):
        self.assertEqual(A._split_targets(""), [])
        self.assertEqual(A._split_targets(None), [])
        self.assertEqual(A._split_targets("  "), [])

    def test_empty_entries_are_dropped(self):
        self.assertEqual(A._split_targets("bolt, , sky"), ["bolt", "sky"])

    def test_malformed_json_falls_back_to_splitting(self):
        self.assertEqual(A._split_targets('["bolt", "sky"'), ['["bolt"', '"sky"'])


class ParseRegionsTest(unittest.TestCase):
    def test_pixel_boxes(self):
        got = A._parse_regions('[{"box":[10,20,110,120],"label":"x"}]', 500, 500)
        self.assertEqual(len(got), 1)
        self.assertEqual(got[0].box, [10.0, 20.0, 110.0, 120.0])
        self.assertEqual(got[0].label, "x")

    def test_fractional_boxes_are_scaled(self):
        got = A._parse_regions('[{"box":[0.1,0.2,0.5,0.6]}]', 1000, 500)
        self.assertEqual(got[0].box, [100.0, 100.0, 500.0, 300.0])

    def test_a_bare_array_is_accepted(self):
        got = A._parse_regions("[[10,20,110,120]]", 500, 500)
        self.assertEqual(got[0].box, [10.0, 20.0, 110.0, 120.0])

    def test_a_single_object_is_accepted(self):
        got = A._parse_regions('{"box":[1,2,3,4]}', 500, 500)
        self.assertEqual(len(got), 1)

    def test_bbox_2d_spelling(self):
        # What a Qwen-VL style grounding reply calls it.
        got = A._parse_regions('[{"bbox_2d":[10,20,110,120]}]', 500, 500)
        self.assertEqual(got[0].box, [10.0, 20.0, 110.0, 120.0])

    def test_already_parsed_input(self):
        got = A._parse_regions([{"box": [1, 2, 3, 4]}], 500, 500)
        self.assertEqual(len(got), 1)

    def test_blank_gives_nothing(self):
        self.assertEqual(A._parse_regions("", 100, 100), [])
        self.assertEqual(A._parse_regions(None, 100, 100), [])

    def test_bad_json_raises(self):
        with self.assertRaises(ValueError):
            A._parse_regions("{not json", 100, 100)

    def test_a_short_box_raises(self):
        with self.assertRaises(ValueError):
            A._parse_regions('[{"box":[1,2]}]', 100, 100)

    def test_a_non_list_raises(self):
        with self.assertRaises(ValueError):
            A._parse_regions('"just a string"', 100, 100)

    def test_a_full_frame_fractional_box_is_not_mistaken_for_pixels(self):
        got = A._parse_regions('[{"box":[0,0,1,1]}]', 800, 600)
        self.assertEqual(got[0].box, [0.0, 0.0, 800.0, 600.0])


class ClampBoxTest(unittest.TestCase):
    def test_a_box_outside_the_frame_is_pulled_in(self):
        # Grounding really does return slightly negative coordinates.
        self.assertEqual(A._clamp_box([-1.7, -0.7, 150.8, 386.4], 500, 500),
                         [0.0, 0.0, 150.8, 386.4])

    def test_an_oversized_box_is_capped(self):
        self.assertEqual(A._clamp_box([10, 10, 900, 900], 500, 400),
                         [10.0, 10.0, 500.0, 400.0])

    def test_an_inside_box_is_untouched(self):
        self.assertEqual(A._clamp_box([10, 20, 30, 40], 500, 500),
                         [10.0, 20.0, 30.0, 40.0])


class OutputSinkTest(unittest.TestCase):
    def tearDown(self):
        A.set_output_sink(None)

    def test_publishing_calls_the_sink(self):
        seen = []
        A.set_output_sink(seen.append)
        self.assertTrue(A._publish("x.png"))
        self.assertEqual(seen, ["x.png"])

    def test_no_sink_is_not_an_error(self):
        A.set_output_sink(None)
        self.assertFalse(A._publish("x.png"))

    def test_a_raising_sink_does_not_break_the_tool(self):
        def boom(_):
            raise RuntimeError("panel is gone")
        A.set_output_sink(boom)
        self.assertFalse(A._publish("x.png"))


class ConfigTest(unittest.TestCase):
    def test_env_var_beats_settings(self):
        with mock.patch.dict(os.environ, {"AGENTY_SAM3_DEVICE": "cpu"}):
            self.assertEqual(L._device(), "cpu")

    def test_an_unknown_device_falls_back_to_detection(self):
        with mock.patch.dict(os.environ, {"AGENTY_SAM3_DEVICE": "quantum"}):
            self.assertIn(L._device(), ("cpu", "cuda"))

    def test_idle_unload_reads_the_env(self):
        with mock.patch.dict(os.environ, {"AGENTY_SAM3_IDLE_UNLOAD": "45"}):
            self.assertEqual(L.idle_unload_seconds(), 45.0)

    def test_a_junk_idle_value_uses_the_default(self):
        with mock.patch.dict(os.environ, {"AGENTY_SAM3_IDLE_UNLOAD": "soon"}):
            self.assertEqual(L.idle_unload_seconds(), L._DEFAULT_IDLE_UNLOAD_S)

    def test_an_explicit_checkpoint_that_does_not_exist_resolves_to_none(self):
        with mock.patch.dict(os.environ, {"AGENTY_SAM3_CHECKPOINT": "Z:/nope.safetensors"}):
            self.assertIsNone(L.checkpoint_path())

    def test_an_explicit_checkpoint_that_exists_is_used(self):
        with tempfile.NamedTemporaryFile(suffix=".safetensors", delete=False) as fh:
            fh.write(b"x")
            path = fh.name
        try:
            with mock.patch.dict(os.environ, {"AGENTY_SAM3_CHECKPOINT": path}):
                self.assertEqual(L.checkpoint_path(), path)
        finally:
            os.unlink(path)

    def test_no_absolute_machine_paths_are_hard_coded(self):
        # A path baked in here would work on one machine and nowhere else.
        import inspect
        src = inspect.getsource(L)
        for needle in ("D:/AI", "D:\\\\AI", "C:/Users", "/home/"):
            self.assertNotIn(needle, src, f"hard-coded path {needle!r} in image_locate")

    def test_comfy_roots_are_relative_to_the_checkout(self):
        roots = L._comfy_roots()
        self.assertTrue(roots)
        self.assertTrue(all(isinstance(r, str) and r for r in roots))

    def test_availability_explains_itself_when_the_checkpoint_is_missing(self):
        with mock.patch.object(L, "checkpoint_path", return_value=None):
            ok, why = L.availability()
            self.assertFalse(ok)
            self.assertIn("sam3.safetensors", why)


if __name__ == "__main__":
    unittest.main()
