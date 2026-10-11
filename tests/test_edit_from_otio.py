"""A cut comes back from an OpenTimelineIO file.

The user can cut a film themselves - in the bEpic viewer's Edit panel, or in an
editing program - and the result is a timeline file. agentY takes that cut in
(the "load" operation) and renders it, instead of holding on to its own.
"""
import json
import unittest
from pathlib import Path

from src.utils import edit_timeline as et

ROOT = Path(__file__).resolve().parent.parent
FACTS = {"D:/s/a.mp4": {"duration": 4.0}, "//srv/s/b.mp4": {"duration": 5.0},
         "D:/s/c.mp4": {"duration": 6.0}}


def cut(fps=25):
    return et.apply(et.new(fps, 1280, 720), [
        *({"op": "add", "src": s, "label": n} for s, n in zip(FACTS, ("wide", "close", "hall"))),
        {"op": "trim", "clip": "c2", "in": 1, "out": 4},
        {"op": "transition", "clip": "c3", "seconds": 1},
        {"op": "sound", "clip": "c1", "volume": 0.5},
        {"op": "audio", "src": "D:/m/bed.wav", "gain": 0.3, "fade_out": 2},
        {"op": "sound", "on": False},
        {"op": "text", "preset": "lower_third", "text": "Mara\nKitchen", "start": 2, "length": 3,
         "params": {"size": 40}}], FACTS)


class WhatWasWrittenComesBack(unittest.TestCase):
    def test_the_whole_timeline_at_any_frame_rate(self):
        for fps in (24, 25, 30):
            tl = cut(fps)
            self.assertEqual(et.from_otio(et.to_otio(tl), tl), tl, fps)

    def test_clips_are_whole_again_with_their_dissolve(self):
        back = et.from_otio(et.to_otio(cut()))
        self.assertEqual([(c["in"], c["out"], c["transition"]) for c in back["clips"]],
                         [(0.0, 4.0, 0.0), (1.0, 4.0, 0.0), (0.0, 6.0, 1.0)])

    def test_a_dissolve_of_an_odd_number_of_frames_keeps_all_of_them(self):
        # 25 frames: two rounded halves would be 24, and the film a frame short.
        video = et.to_otio(cut())["tracks"]["children"][0]["children"]
        mix = next(c for c in video if c["OTIO_SCHEMA"] == "Transition.1")
        self.assertEqual(mix["in_offset"]["value"] + mix["out_offset"]["value"], 25.0)
        self.assertEqual(et.duration(et.from_otio(et.to_otio(cut()))), et.duration(cut()))

    def test_what_the_file_cannot_say_is_kept_from_the_timeline_it_replaces(self):
        doc = et.to_otio(cut())
        doc["metadata"] = {}                                   # a file from an editing program
        back = et.from_otio(doc, et.new(30, 3840, 2160))
        self.assertEqual((back["fps"], back["width"], back["height"]), (30.0, 3840, 2160))

    def test_the_sound_settings_travel_with_it(self):
        back = et.from_otio(et.to_otio(cut()))
        self.assertEqual(back["audio"], {"src": "D:/m/bed.wav", "in": 0.0, "gain": 0.3, "fade_out": 2.0})
        self.assertEqual((back["clip_sound"], back["clips"][0]["volume"]), (False, 0.5))

    def test_a_network_path_and_a_drive_path_both_survive(self):
        back = et.from_otio(et.to_otio(cut()))
        self.assertEqual([c["src"] for c in back["clips"]], list(FACTS))


class ACutFromSomewhereElse(unittest.TestCase):
    def test_a_marker_with_no_preset_becomes_a_caption(self):
        doc = et.to_otio(cut())
        doc["tracks"]["markers"] = [{"OTIO_SCHEMA": "Marker.2", "name": "note from the editor",
                                     "marked_range": {"OTIO_SCHEMA": "TimeRange.1",
                                                      "start_time": {"OTIO_SCHEMA": "RationalTime.1", "rate": 25, "value": 50},
                                                      "duration": {"OTIO_SCHEMA": "RationalTime.1", "rate": 25, "value": 75}},
                                     "metadata": {}}]
        text = et.from_otio(doc)["overlays"][0]
        self.assertEqual((text["preset"], text["text"], text["start"], text["length"]),
                         ("caption", "note from the editor", 2.0, 3.0))

    def test_a_gap_and_a_clip_with_no_file_are_left_out(self):
        doc = et.to_otio(cut())
        video = doc["tracks"]["children"][0]["children"]
        video.insert(0, {"OTIO_SCHEMA": "Gap.1", "source_range": video[0]["source_range"]})
        video[1]["media_reference"] = {"OTIO_SCHEMA": "MissingReference.1"}
        self.assertEqual([c["label"] for c in et.from_otio(doc)["clips"]], ["close", "hall"])

    def test_what_is_not_a_timeline_is_said(self):
        for bad in ({}, {"OTIO_SCHEMA": "Clip.1"}, []):
            with self.assertRaisesRegex(et.EditError, "not an OpenTimelineIO timeline"):
                et.from_otio(bad)
        empty = et.to_otio(cut())
        empty["tracks"]["children"][0]["children"] = []
        with self.assertRaisesRegex(et.EditError, "no clips with a file"):
            et.from_otio(empty)

    def test_a_cut_saved_from_the_viewer_says_so(self):
        doc = et.to_otio(cut())
        self.assertEqual(et.viewer_stamp(doc), "")
        doc["metadata"]["bepic"] = {"edited": "2026-10-12T08:00:00+00:00"}
        self.assertEqual(et.viewer_stamp(doc), "2026-10-12T08:00:00+00:00")


class TheLoadOperation(unittest.TestCase):
    SRC = (ROOT / "src" / "pipeline.py").read_text(encoding="utf-8")

    def setUp(self):
        self.tool = self.SRC.split("        async def edit_timeline(", 1)[1].split("        @_tool", 1)[0]

    def test_it_comes_first_and_the_rest_is_applied_to_what_was_loaded(self):
        self.assertIn('str(ops[0].get("op") or "").lower() == "load"', self.tool)
        self.assertLess(self.tool.index("timeline = _et.from_otio(doc, timeline)"),
                        self.tool.index("timeline = _et.apply(timeline, list(ops), known)"))
        self.assertIn("ops = list(ops[1:])", self.tool)

    def test_only_a_timeline_file_that_exists(self):
        self.assertIn("if not path.lower().endswith(\".otio\") or not os.path.isfile(path):", self.tool)

    def test_the_agent_is_told_when_the_user_changed_the_cut_in_the_viewer(self):
        body = self.SRC.split("    def _edit_states(self)", 1)[1].split("\n    def ", 1)[0]
        self.assertIn("THE USER CHANGED THIS CUT IN THE VIEWER", body)
        self.assertIn("their cut, not yours", body)
        check = self.SRC.split("    def _edit_changed_in_viewer(self", 1)[1].split("\n    def ", 1)[0]
        self.assertIn('f"edit_loaded:{hook_node_id}"', check)

    def test_once_loaded_it_is_not_announced_again(self):
        self.assertIn("self._edit_note_loaded(str(hook_node_id), _et.viewer_stamp(doc))", self.tool)

    def test_the_stage_is_told_the_user_can_cut_it_themselves(self):
        from src.utils import canvas_hooks as ch
        text = "\n".join(ch.edit_lines([{"hook_node_id": "150", "purpose": "edit"}], lambda h: "none"))
        self.assertIn("THE USER CAN CUT IT THEMSELVES", text)
        self.assertIn('{"op": "load", "file":', text)


if __name__ == "__main__":
    unittest.main()
