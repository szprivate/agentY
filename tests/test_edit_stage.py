"""The edit stage: an edit is a document the agent changes and the host renders.

`agentY edit` cuts finished shots into one film. The agent keeps a timeline and
changes it with small operations, so "shot 3 shorter" is one trim and not a new
cut; the host checks it against the files, renders it with ffmpeg and writes an
OpenTimelineIO file for whoever takes it further.
"""
import json
import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

from src.utils import canvas_hooks as ch
from src.utils import edit_render as er
from src.utils import edit_timeline as et

ROOT = Path(__file__).resolve().parent.parent
FACTS = {"a.mp4": {"duration": 4.0}, "b.mp4": {"duration": 5.0}, "c.mp4": {"duration": 6.0},
         "still.png": {"duration": 0.0}}


def cut():
    return et.apply(et.new(), [{"op": "add", "src": "a.mp4", "label": "wide"},
                               {"op": "add", "src": "b.mp4", "in": 1, "out": 4},
                               {"op": "add", "src": "c.mp4", "in": 0, "length": 2}], FACTS)


class BuildingACut(unittest.TestCase):
    def test_a_clip_added_whole_takes_its_files_length(self):
        tl = cut()
        self.assertEqual([(c["id"], c["in"], c["out"]) for c in tl["clips"]],
                         [("c1", 0.0, 4.0), ("c2", 1.0, 4.0), ("c3", 0.0, 2.0)])
        self.assertEqual(et.duration(tl), 9.0)

    def test_without_knowing_its_length_the_end_has_to_be_said(self):
        with self.assertRaisesRegex(et.EditError, "say where the clip ends"):
            et.apply(et.new(), [{"op": "add", "src": "unknown.mp4"}], {})

    def test_a_still_stands_for_a_while(self):
        tl = et.apply(et.new(), [{"op": "add", "src": "still.png"}], FACTS)
        self.assertEqual(et.clip_length(tl["clips"][0]), 3.0)

    def test_a_clip_can_go_in_at_a_position(self):
        tl = et.apply(cut(), [{"op": "add", "src": "still.png", "length": 1, "at": 1}], FACTS)
        self.assertEqual([c["src"] for c in tl["clips"]][0], "still.png")


class ANoteIsASmallChange(unittest.TestCase):
    def test_shot_three_shorter(self):
        tl = et.apply(cut(), [{"op": "trim", "clip": "c3", "out": 1.5}])
        self.assertEqual(et.clip_length(tl["clips"][2]), 1.5)
        self.assertEqual([et.clip_length(c) for c in tl["clips"][:2]], [4.0, 3.0])   # untouched

    def test_a_clip_is_also_named_by_its_position(self):
        tl = et.apply(cut(), [{"op": "trim", "clip": 3, "out": 1.5}])
        self.assertEqual(tl["clips"][2]["out"], 1.5)

    def test_swap_two(self):
        tl = et.apply(cut(), [{"op": "move", "clip": "c3", "to": 1}])
        self.assertEqual([c["id"] for c in tl["clips"]], ["c3", "c1", "c2"])

    def test_another_take_keeps_its_place_and_length(self):
        tl = et.apply(cut(), [{"op": "replace", "clip": "c2", "src": "c.mp4"}], FACTS)
        c = tl["clips"][1]
        self.assertEqual((c["id"], c["src"], c["in"], c["out"]), ("c2", "c.mp4", 0.0, 3.0))

    def test_removing_one(self):
        self.assertEqual([c["id"] for c in et.apply(cut(), [{"op": "remove", "clip": "c1"}])["clips"]],
                         ["c2", "c3"])

    def test_the_original_is_never_changed_and_a_bad_list_changes_nothing(self):
        before = cut()
        with self.assertRaisesRegex(et.EditError, r"operation 2 \(trim\): there is no clip 'c9'"):
            et.apply(before, [{"op": "remove", "clip": "c1"}, {"op": "trim", "clip": "c9", "out": 1}])
        self.assertEqual(len(before["clips"]), 3)

    def test_an_unknown_operation_is_named(self):
        with self.assertRaisesRegex(et.EditError, "unknown operation"):
            et.apply(cut(), [{"op": "wipe"}])


class Dissolves(unittest.TestCase):
    def test_a_dissolve_is_taken_out_of_both_clips(self):
        tl = et.apply(cut(), [{"op": "transition", "clip": "c2", "seconds": 0.5}])
        self.assertEqual(et.duration(tl), 8.5)
        self.assertEqual(et.starts(tl), [0.0, 3.5, 6.5])

    def test_the_first_clip_has_nothing_to_dissolve_from(self):
        with self.assertRaisesRegex(et.EditError, "nothing before it"):
            et.apply(cut(), [{"op": "transition", "clip": "c1", "seconds": 0.5}])

    def test_zero_makes_it_a_cut_again(self):
        tl = et.apply(cut(), [{"op": "transition", "clip": "c2", "seconds": 0.5},
                              {"op": "transition", "clip": "c2", "seconds": 0}])
        self.assertEqual(et.summary(tl)["clips"][1]["into_it"], "cut")


class FittingALength(unittest.TestCase):
    def test_every_clip_gives_the_same_share_from_its_end(self):
        tl = et.apply(cut(), [{"op": "fit", "seconds": 4.5}])
        self.assertAlmostEqual(et.duration(tl), 4.5, places=3)
        self.assertEqual([c["in"] for c in tl["clips"]], [0.0, 1.0, 0.0])
        self.assertEqual([round(et.clip_length(c), 3) for c in tl["clips"]], [2.0, 1.5, 1.0])

    def test_with_a_dissolve_in_it(self):
        tl = et.apply(cut(), [{"op": "transition", "clip": "c2", "seconds": 0.5},
                              {"op": "fit", "seconds": 6}])
        self.assertAlmostEqual(et.duration(tl), 6.0, places=3)

    def test_a_film_already_that_short_is_left_alone(self):
        self.assertEqual(et.duration(et.apply(cut(), [{"op": "fit", "seconds": 60}])), 9.0)


class WhatStopsARender(unittest.TestCase):
    def test_nothing_when_the_cut_is_sound(self):
        self.assertEqual(et.problems(cut(), FACTS), [])

    def test_no_clips(self):
        self.assertEqual(et.problems(et.new(), {}), ["the timeline has no clips yet"])

    def test_a_clip_that_runs_past_the_end_of_its_file(self):
        tl = et.apply(cut(), [{"op": "trim", "clip": "c1", "out": 7}])
        self.assertIn("clip c1 (a.mp4) ends at 7s but the file is only 4s long", et.problems(tl, FACTS))

    def test_a_file_that_is_gone(self):
        facts = dict(FACTS, **{"b.mp4": None})
        self.assertIn("clip c2 (b.mp4): the file is missing or is not a picture or a video",
                      et.problems(cut(), facts))

    def test_a_dissolve_as_long_as_a_clip(self):
        tl = et.apply(cut(), [{"op": "transition", "clip": "c3", "seconds": 2}])
        self.assertTrue(any("dissolve into clip c3" in p for p in et.problems(tl, FACTS)))


class TheCommand(unittest.TestCase):
    def setUp(self):
        self.tl = et.apply(cut(), [{"op": "transition", "clip": "c2", "seconds": 0.5},
                                   {"op": "audio", "src": "bed.wav", "gain": 0.5}])
        self.cmd = er.command(self.tl, "out.mp4")
        self.graph = self.cmd[self.cmd.index("-filter_complex") + 1]

    def test_each_clip_is_read_from_its_in_point_for_its_length(self):
        self.assertEqual(self.cmd[self.cmd.index("b.mp4") - 5:self.cmd.index("b.mp4") + 1],
                         ["-ss", "1", "-t", "3", "-i", "b.mp4"])

    def test_everything_is_brought_to_one_size_and_rate_without_stretching(self):
        self.assertEqual(self.graph.count("scale=1920:1080:force_original_aspect_ratio=decrease"), 3)
        self.assertEqual(self.graph.count("fps=24"), 3)

    def test_a_dissolve_is_an_xfade_at_the_right_moment_and_a_cut_a_concat(self):
        self.assertIn("[v0][v1]xfade=transition=fade:duration=0.5:offset=3.5[j1]", self.graph)
        self.assertIn("[j1][v2]concat=n=2:v=1:a=0[j2]", self.graph)

    def test_the_sound_lies_under_the_whole_film_and_fades_out(self):
        self.assertIn("[3:a]atrim=duration=8.5,asetpts=PTS-STARTPTS,volume=0.5,"
                      "afade=t=out:st=7.5:d=1[aout]", self.graph)

    def test_a_still_is_held(self):
        tl = et.apply(et.new(), [{"op": "add", "src": "still.png", "length": 2}], FACTS)
        cmd = er.command(tl, "o.mp4")
        self.assertEqual(cmd[cmd.index("still.png") - 5:cmd.index("still.png")],
                         ["-loop", "1", "-t", "2", "-i"])

    def test_mov_is_prores(self):
        self.assertIn("prores_ks", er.command(self.tl, "out.mov", "mov"))

    def test_no_sound_bed_no_sound(self):
        self.assertIn("-an", er.command(cut(), "out.mp4"))


class HandingItOn(unittest.TestCase):
    """OpenTimelineIO, written as the JSON it is."""

    def setUp(self):
        self.tl = et.apply(cut(), [{"op": "transition", "clip": "c2", "seconds": 0.5},
                                   {"op": "audio", "src": "//srv/music/bed.wav"}])
        self.otio = et.to_otio(self.tl, "final cut")
        self.video, self.audio = self.otio["tracks"]["children"]

    def test_the_shape(self):
        self.assertEqual(self.otio["OTIO_SCHEMA"], "Timeline.1")
        self.assertEqual([t["kind"] for t in self.otio["tracks"]["children"]], ["Video", "Audio"])
        self.assertEqual([c["OTIO_SCHEMA"] for c in self.video["children"]],
                         ["Clip.1", "Transition.1", "Clip.1", "Clip.1"])

    def test_a_dissolve_takes_half_from_each_side(self):
        a, t, b, _c = self.video["children"]
        frames = lambda c: c["source_range"]["duration"]["value"]  # noqa: E731
        self.assertEqual((frames(a), frames(b)), (90.0, 66.0))          # 3.75 s and 2.75 s at 24
        self.assertEqual(b["source_range"]["start_time"]["value"], 30.0)   # 1.0 + 0.25
        self.assertEqual(t["in_offset"]["value"], 6.0)

    def test_the_lengths_add_up_to_the_film(self):
        total = sum(c["source_range"]["duration"]["value"] for c in self.video["children"]
                    if c["OTIO_SCHEMA"] == "Clip.1")
        self.assertEqual(total / 24, et.duration(self.tl))

    def test_files_are_urls_a_network_path_too(self):
        self.assertEqual(self.video["children"][0]["media_reference"]["target_url"], "file:///a.mp4")
        self.assertEqual(self.audio["children"][0]["media_reference"]["target_url"],
                         "file://srv/music/bed.wav")


@unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "ffmpeg is not installed")
class ARealRender(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dir = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        ff = shutil.which("ffmpeg")

        def make(name, src, seconds, size):
            out = os.path.join(cls.dir.name, name)
            subprocess.run([ff, "-y", "-loglevel", "error", "-f", "lavfi", "-i",
                            f"{src}=size={size}:rate=30:duration={seconds}", "-pix_fmt", "yuv420p", out],
                           check=True)
            return out
        cls.a = make("a.mp4", "testsrc", 2, "640x360")
        cls.b = make("b.mp4", "testsrc2", 2, "320x320")

    @classmethod
    def tearDownClass(cls):
        cls.dir.cleanup()

    def test_what_a_file_is(self):
        fact = er.probe(self.a)
        self.assertEqual((fact["duration"], fact["fps"], fact["width"], fact["audio"]),
                         (2.0, 30.0, 640, False))
        self.assertIsNone(er.probe(os.path.join(self.dir.name, "gone.mp4")))

    def test_the_film_is_as_long_as_the_timeline_says(self):
        tl = et.apply(et.new(24, 640, 360), [{"op": "add", "src": self.a, "in": 0.5},
                                             {"op": "add", "src": self.b},
                                             {"op": "transition", "clip": "c2", "seconds": 0.5}],
                      er.facts_for({"clips": [{"src": self.a}, {"src": self.b}]}))
        self.assertEqual(et.problems(tl, er.facts_for(tl)), [])
        out = er.render(tl, os.path.join(self.dir.name, "v001", "cut.mp4"))
        got = er.probe(out)
        self.assertAlmostEqual(got["duration"], et.duration(tl), delta=0.1)
        self.assertEqual((got["width"], got["height"], got["fps"]), (640, 360, 24.0))
        otio = er.write_otio(tl, os.path.join(self.dir.name, "v001", "cut.otio"))
        self.assertEqual(json.loads(Path(otio).read_text(encoding="utf-8"))["OTIO_SCHEMA"], "Timeline.1")

    def test_ffmpegs_own_words_come_back_when_it_fails(self):
        tl = et.apply(et.new(), [{"op": "add", "src": os.path.join(self.dir.name, "gone.mp4"),
                                  "in": 0, "out": 1}])
        with self.assertRaisesRegex(er.RenderError, "ffmpeg failed"):
            er.render(tl, os.path.join(self.dir.name, "bad.mp4"))


def stage(**more):
    return {"hook_node_id": "150", "purpose": "edit", "title": "final cut",
            "directive": "open on the kitchen", "anchors": [], "targets": [],
            "prev_hook_ids": ["140"], "edit": {"width": 1920, "height": 1080, "fps": 24,
                                               "length": 30, "format": "mov"}, **more}


class WhatTheAgentIsTold(unittest.TestCase):
    def test_it_is_an_edit_stage_and_not_a_producer(self):
        text = ch.describe_hooks([stage()], {}, edits={"150": "5 clip(s), 23.5 s"})
        self.assertIn("EDIT stages — each cuts the shots it is given into ONE film", text)
        self.assertIn('- EDIT stage 150 "final cut"', text)
        self.assertNotIn("PRODUCER hook 150", text)

    def test_its_settings_brief_and_where_the_timeline_stands(self):
        text = ch.describe_hooks([stage()], {}, edits={"150": "5 clip(s), 23.5 s"})
        self.assertIn("film: 1920x1080 at 24 fps, mov, 30 s long (fit the cut to it)", text)
        self.assertIn("timeline now: 5 clip(s), 23.5 s", text)
        self.assertIn('brief: "open on the kitchen"', text)

    def test_no_timeline_and_no_brief(self):
        text = ch.describe_hooks([stage(directive="", edit={"length": 0})], {})
        self.assertIn("as long as the shots make it", text)
        self.assertIn("timeline now: none yet", text)
        self.assertIn("no brief: the shots in story order, straight cuts", text)

    def test_the_rhythm_reference_and_the_sound_are_named_as_files(self):
        ref = [{"node_id": "9", "type": "VHS_LoadVideoPath", "widgets": {"video": "//srv/ref/spot.mp4"}}]
        audio = [{"node_id": "10", "type": "LoadAudio", "widgets": {"audio": "//srv/music/bed.wav"}}]
        text = ch.describe_hooks([stage(reference=ref, audio=audio)], {})
        self.assertIn("RHYTHM REFERENCE: //srv/ref/spot.mp4", text)
        self.assertIn("AUDIO under the film: //srv/music/bed.wav", text)

    def test_files_named_by_an_entry(self):
        entries = [{"widgets": {"files": "D:/a.png\nD:/b.mp4\nnot a file", "seed": 3}}, None]
        self.assertEqual(ch.entry_files(entries), ["D:/a.png", "D:/b.mp4"])

    def test_the_node_is_taken_out_of_the_graph_that_runs(self):
        self.assertIn("AgentYEdit", ch._HOOK_CLASSES)
        g = {"1": {"class_type": "SaveImage", "inputs": {}},
             "150": {"class_type": "AgentYEdit", "inputs": {"anchors.anchor0": ["1", 0], "exec": ["9", 1]}},
             "151": {"class_type": "PreviewAny", "inputs": {"source": ["150", 0]}}}
        clean, removed = ch.splice_hook_nodes(g, [stage()])
        self.assertEqual(removed, ["150"])
        self.assertNotIn("150", clean)


class TheTools(unittest.TestCase):
    SRC = (ROOT / "src" / "pipeline.py").read_text(encoding="utf-8")

    def tool(self, name):
        return self.SRC.split(f"        async def {name}(", 1)[1].split("        @_tool", 1)[0]

    def test_the_orchestrator_has_them(self):
        self.assertIn("edit_timeline, render_edit,", self.SRC)

    def test_the_timeline_is_kept_with_the_run(self):
        self.assertIn('_cs.kv_set(lead, f"edit:{hook_node_id}", timeline)', self.SRC)

    def test_a_bad_operation_changes_nothing_and_says_why(self):
        body = self.tool("edit_timeline")
        self.assertLess(body.index("timeline = _et.apply(timeline, list(ops), known)"),
                        body.index("self._edit_save(str(hook_node_id), timeline)"))
        self.assertIn('return json.dumps({"error": f"not changed - {exc}",', body)

    def test_a_render_is_checked_first_and_is_the_stages_product(self):
        body = self.tool("render_edit")
        self.assertLess(body.index("wrong = _et.problems(timeline, facts)"),
                        body.index("_er.render, timeline, out, fmt"))
        self.assertIn("products = self._record_hook_products(hid, [out])", body)
        self.assertIn("self._session.current_output_paths.append(out)", body)
        self.assertIn("version = self._version_stage(hid, [], same_round) or 1", body)

    def test_it_waits_behind_a_review_like_any_stage(self):
        for name in ("edit_timeline", "render_edit"):
            self.assertIn("self._review_ahead_refusal(", self.tool(name), name)

    def test_a_dry_run_renders_nothing(self):
        body = self.tool("render_edit")
        self.assertLess(body.index("if self._dry_run:"), body.index("_er.render, timeline, out, fmt"))


if __name__ == "__main__":
    unittest.main()
