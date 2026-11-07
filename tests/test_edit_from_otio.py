"""A cut comes back from an OpenTimelineIO file.

The user can cut a film themselves - in the ComfyUI Image Viewer's Edit panel, or in an
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

    def test_a_retimed_or_looped_clip_comes_in_plain_and_is_named(self):
        doc = et.to_otio(cut())
        video = doc["tracks"]["children"][0]["children"]
        clips = [c for c in video if c["OTIO_SCHEMA"].startswith("Clip")]
        clips[0].setdefault("metadata", {})["imageviewer"] = {"speed": 0.5}
        clips[0].setdefault("effects", []).append({"OTIO_SCHEMA": "LinearTimeWarp.1", "name": "",
                                                   "effect_name": "LinearTimeWarp", "metadata": {}, "time_scalar": 0.5})
        clips[1].setdefault("metadata", {})["imageviewer"] = {"loop": "pingpong", "span": 1.0}
        plain = et.from_otio(et.to_otio(cut()))
        with self.assertLogs("src.utils.edit_timeline", level="WARNING"):
            back = et.from_otio(doc)
        self.assertEqual(back["clips"], plain["clips"])        # the plain clips: from the in point for their length
        dropped = et.otio_dropped(doc)
        self.assertEqual(len(dropped), 2)
        self.assertIn("50 %", dropped[0])
        self.assertIn("pingpong", dropped[1])
        self.assertEqual(et.otio_dropped(et.to_otio(cut())), [])
        # Keyframes from the viewer's Inspector: named, played without them.
        doc = et.to_otio(cut())
        first = [c for c in doc["tracks"]["children"][0]["children"] if c["OTIO_SCHEMA"].startswith("Clip")][0]
        first.setdefault("metadata", {})["imageviewer"] = {"fx": {"op": 0.5, "keys": {"op": [{"t": 0, "v": 0}, {"t": 1, "v": 1}], "vol": [{"t": 0, "v": 1}]}}}
        self.assertTrue(any("keyframes (op, vol)" in d for d in et.otio_dropped(doc)))

    def test_audio_fades_and_crossfades_are_named_but_locks_are_not(self):
        doc = et.to_otio(cut())
        clips = [c for c in doc["tracks"]["children"][0]["children"] if c["OTIO_SCHEMA"].startswith("Clip")]
        clips[0].setdefault("metadata", {})["imageviewer"] = {"afade_in": 1.0, "afade_out": 0}
        clips[1].setdefault("metadata", {})["imageviewer"] = {
            "afade_out": 0.5, "axfade": {"len": 1, "curve": "+3", "align": "centre"}}
        doc["tracks"]["children"][0].setdefault("metadata", {}).setdefault("imageviewer", {}).update(
            {"lock": True, "auto_select": False})
        dropped = et.otio_dropped(doc)
        self.assertTrue(any("audio crossfade into" in d and "(1s) left out" in d for d in dropped), dropped)
        self.assertTrue(any("audio fade in - left out" in d for d in dropped), dropped)
        self.assertTrue(any("audio fade out - left out" in d for d in dropped), dropped)
        self.assertFalse(any("lock" in d for d in dropped), dropped)
        self.assertEqual(len(dropped), 3)
        et.from_otio(doc)                                   # still loads

    def test_sound_effects_are_named_unless_bypassed(self):
        doc = et.to_otio(cut())
        track = doc["tracks"]["children"][0]
        clips = [c for c in track["children"] if c["OTIO_SCHEMA"].startswith("Clip")]
        clips[0].setdefault("metadata", {})["imageviewer"] = {"afx": [
            {"type": "eq", "on": True, "p": {}}, {"type": "reverb", "on": True, "p": {}},
            {"type": "delay", "on": False, "p": {}}]}
        track.setdefault("metadata", {}).setdefault("imageviewer", {})["afx"] = [
            {"type": "comp", "on": True, "p": {}}, {"type": "limit", "on": False, "p": {}}]
        dropped = et.otio_dropped(doc)
        self.assertTrue(any(d.endswith(": sound effects (EQ, Reverb) - left out") for d in dropped), dropped)
        self.assertTrue(any(d.startswith("track ") and d.endswith(": sound effects (Compressor) - left out")
                            for d in dropped), dropped)
        self.assertEqual(len(dropped), 2)
        et.from_otio(doc)                                   # still loads

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
        doc["metadata"]["imageviewer"] = {"edited": "2026-10-12T08:00:00+00:00"}
        self.assertEqual(et.viewer_stamp(doc), "2026-10-12T08:00:00+00:00")


class GapsAndSoundSplitOffInTheViewer(unittest.TestCase):
    """The viewer's Edit panel can leave gaps on V1 and split a film's sound onto a sound track."""

    @staticmethod
    def rt(v):
        return {"OTIO_SCHEMA": "RationalTime.1", "rate": 25, "value": v}

    def gap(self, frames):
        return {"OTIO_SCHEMA": "Gap.1", "name": "", "metadata": {}, "effects": [], "markers": [],
                "source_range": {"OTIO_SCHEMA": "TimeRange.1", "start_time": self.rt(0), "duration": self.rt(frames)}}

    def test_a_gap_on_v1_is_closed_up_and_said(self):
        doc = et.to_otio(cut())
        video = doc["tracks"]["children"][0]["children"]
        video.insert(1, self.gap(50))                       # 2 s of nothing after "wide"
        back = et.from_otio(doc)
        self.assertEqual(back["clips"], et.from_otio(et.to_otio(cut()))["clips"])
        self.assertEqual(et.duration(back), et.duration(cut()))
        dropped = et.otio_dropped(doc)
        self.assertEqual(len(dropped), 1)
        self.assertIn("gap of 2s before close", dropped[0])
        # And written out again: no gap (this timeline has none), the film as long as its clips.
        again = et.to_otio(back)["tracks"]["children"][0]["children"]
        self.assertFalse([c for c in again if c["OTIO_SCHEMA"].startswith("Gap")])

    def test_a_sound_split_off_its_film_is_heard_with_its_picture_again(self):
        doc = et.to_otio(cut())
        video = doc["tracks"]["children"][0]["children"]
        first = video[0]
        first["metadata"]["imageviewer"] = {"link": "L1", "unlinked": True, "mute": True}
        sound = json.loads(json.dumps(first))
        sound["metadata"] = {"imageviewer": {"link": "L1", "unlinked": True}}
        a1 = next(t for t in doc["tracks"]["children"] if t["kind"] == "Audio")
        a2 = {"OTIO_SCHEMA": "Track.1", "name": "A2", "kind": "Audio", "metadata": {}, "source_range": None,
              "effects": [], "markers": [], "children": [self.gap(12), sound]}
        v2 = {"OTIO_SCHEMA": "Track.1", "name": "V2", "kind": "Video", "metadata": {}, "source_range": None,
              "effects": [], "markers": [], "children": [json.loads(json.dumps(video[-1]))]}
        doc["tracks"]["children"] = [doc["tracks"]["children"][0], v2, a1, a2]
        back = et.from_otio(doc)
        plain = et.from_otio(et.to_otio(cut()))
        self.assertEqual(back["clips"], plain["clips"])     # the film keeps its own sound (volume 0.5 as it was)
        self.assertEqual(back["audio"], plain["audio"])     # A2 is not taken for the bed
        dropped = et.otio_dropped(doc)
        self.assertTrue(any("split off" in d and "+0.48s" in d for d in dropped), dropped)
        self.assertTrue(any("on V2 - left out" in d for d in dropped), dropped)
        self.assertFalse(any("A2 - left out" in d for d in dropped), dropped)
        # Its sound clip deleted in the viewer: the film stays silent.
        doc["tracks"]["children"] = doc["tracks"]["children"][:3]
        self.assertEqual(et.from_otio(doc)["clips"][0]["volume"], 0.0)
        # Another sound on a further track is said to be left out.
        a2["children"] = [self.gap(5), {**sound, "metadata": {}}]
        doc["tracks"]["children"].append(a2)
        self.assertTrue(any("A2 - left out" in d for d in et.otio_dropped(doc)))


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



class OldViewerNames(unittest.TestCase):
    """A cut saved while the viewer was "bEpic" (before 1.18) reads as one saved now."""

    def test_old_notes_and_titles_get_the_new_names(self):
        doc = {"metadata": {"bepic": {"edited": "2026-10-12T08:00:00+00:00"}},
               "tracks": {"children": [{"metadata": {"bepic": {"lock": True}},
                                        "children": [{"media_reference": {"generator_kind": "bepic.title"}}]}]}}
        self.assertEqual(et.viewer_stamp(doc), "2026-10-12T08:00:00+00:00")
        self.assertEqual(doc["tracks"]["children"][0]["metadata"], {"imageviewer": {"lock": True}})
        self.assertEqual(doc["tracks"]["children"][0]["children"][0]["media_reference"]["generator_kind"],
                         "imageviewer.title")

if __name__ == "__main__":
    unittest.main()
