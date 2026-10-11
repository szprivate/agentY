"""Looking at material before cutting it.

A cut made from file names and lengths is made blind. Three measurements give
the agent something to cut by: a contact sheet of a clip, the shot lengths of a
reference film, and the beats of a piece of music.
"""
import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.utils import edit_look as el
from src.utils import edit_render as er
from src.utils import edit_timeline as et

ROOT = Path(__file__).resolve().parent.parent
FACTS = {"a": {"duration": 4.0}, "b": {"duration": 5.0}, "c": {"duration": 6.0}}


def cut():
    return et.apply(et.new(), [{"op": "add", "src": s} for s in "abc"], FACTS)


class TheRhythmOfAnotherFilm(unittest.TestCase):
    def test_cuts_become_shot_lengths(self):
        self.assertEqual(el.lengths_from_cuts([1.5, 4.0, 5.0], 8.0), [1.5, 2.5, 1.0, 3.0])

    def test_one_cut_seen_twice_is_one_cut(self):
        self.assertEqual(el.lengths_from_cuts([2.0, 2.05, 5.0], 8.0), [2.0, 3.0, 3.0])

    def test_a_film_that_was_not_cut_is_one_shot(self):
        self.assertEqual(el.lengths_from_cuts([], 8.0), [8.0])

    def test_the_clips_take_those_lengths_in_order(self):
        tl = et.apply(cut(), [{"op": "rhythm", "lengths": [1.0, 2.0, 0.5]}], FACTS)
        self.assertEqual([et.clip_length(c) for c in tl["clips"]], [1.0, 2.0, 0.5])

    def test_the_pattern_repeats_and_a_short_clip_keeps_what_it_has(self):
        tl = et.apply(cut(), [{"op": "rhythm", "lengths": [1.0, 9.0]}], FACTS)
        self.assertEqual([et.clip_length(c) for c in tl["clips"]], [1.0, 5.0, 1.0])

    def test_no_lengths_is_said(self):
        with self.assertRaisesRegex(et.EditError, "give the reference's shot lengths"):
            et.apply(cut(), [{"op": "rhythm", "lengths": []}], FACTS)


class CuttingOnTheBeat(unittest.TestCase):
    def test_a_cut_moves_to_the_beat_near_it(self):
        tl = et.apply(cut(), [{"op": "snap", "times": [3.8, 9.1]}], FACTS)
        self.assertEqual(et.starts(tl), [0.0, 3.8, 8.8])       # first cut moved, the next measured from it

    def test_a_cut_with_no_beat_near_it_stays(self):
        tl = et.apply(cut(), [{"op": "snap", "times": [20.0]}], FACTS)
        self.assertEqual(et.starts(tl), [0.0, 4.0, 9.0])

    def test_within_says_how_far_a_cut_may_move(self):
        tl = et.apply(cut(), [{"op": "snap", "times": [3.2], "within": 1.0}], FACTS)
        self.assertEqual(et.starts(tl)[1], 3.2)

    def test_a_clip_is_never_made_longer_than_its_file(self):
        tl = et.apply(cut(), [{"op": "snap", "times": [4.4], "within": 1.0}], FACTS)
        self.assertEqual(et.starts(tl)[1], 4.0)                # a is only 4 s long

    def test_the_end_of_the_film_is_not_a_cut(self):
        tl = et.apply(cut(), [{"op": "snap", "times": [14.6], "within": 1.0}], FACTS)
        self.assertEqual(et.duration(tl), 15.0)


def clicks(bpm, seconds=12.0, first=0.25, rate=22050):
    x = np.zeros(int(rate * seconds), dtype="float32")
    t = first
    while t < seconds - 0.05:
        i = int(t * rate)
        n = int(0.04 * rate)
        x[i:i + n] = np.sin(2 * np.pi * 880 * np.arange(n) / rate)
        t += 60.0 / bpm
    return x


class FindingTheBeat(unittest.TestCase):
    def test_the_tempo_of_a_steady_pulse(self):
        for bpm in (96, 120, 150):
            got, _beats = el.tempo_and_beats(el.onset_strength(clicks(bpm)))
            self.assertAlmostEqual(got, bpm, delta=0.6, msg=f"{bpm} bpm read as {got}")

    def test_120_is_not_read_as_60(self):
        self.assertGreater(el.tempo_and_beats(el.onset_strength(clicks(120)))[0], 100)

    def test_the_beats_fall_where_the_clicks_are_all_the_way_through(self):
        _bpm, beats = el.tempo_and_beats(el.onset_strength(clicks(120, first=0.25)))
        self.assertAlmostEqual(beats[0], 0.25, delta=0.03)
        # No drift: the twentieth beat is still on its click.
        self.assertAlmostEqual(beats[20], 0.25 + 20 * 0.5, delta=0.04)

    def test_noise_has_no_beat(self):
        for seed in range(4):
            noise = np.random.default_rng(seed).normal(0, 0.2, 22050 * 8).astype("float32")
            self.assertEqual(el.tempo_and_beats(el.onset_strength(noise)), (0.0, []), seed)

    def test_silence_and_a_snippet_have_none_either(self):
        self.assertEqual(el.tempo_and_beats(el.onset_strength(np.zeros(22050 * 6, dtype="float32"))),
                         (0.0, []))
        self.assertEqual(el.tempo_and_beats(el.onset_strength(clicks(120, seconds=0.5))), (0.0, []))


class AContactSheet(unittest.TestCase):
    def test_frames_are_taken_from_the_middle_of_equal_parts(self):
        self.assertEqual(el.sheet_times(8.0, 8), [0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5])
        self.assertEqual(el.sheet_times(3.0, 2), [0.75, 2.25])


@unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "ffmpeg is not installed")
class OnRealFiles(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dir = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        ff = shutil.which("ffmpeg")
        parts = []
        for name, src, seconds in (("a", "smptebars", 1.5), ("b", "testsrc2", 2.5), ("c", "testsrc", 2.0)):
            out = os.path.join(cls.dir.name, name + ".mp4")
            subprocess.run([ff, "-y", "-loglevel", "error", "-f", "lavfi", "-i",
                            f"{src}=size=320x180:rate=24:duration={seconds}", "-pix_fmt", "yuv420p", out],
                           check=True)
            parts.append(out)
        tl = et.apply(et.new(24, 320, 180), [{"op": "add", "src": p} for p in parts],
                      {p: er.probe(p) for p in parts})
        cls.film = er.render(tl, os.path.join(cls.dir.name, "film.mp4"))
        cls.click = os.path.join(cls.dir.name, "click.wav")
        subprocess.run([ff, "-y", "-loglevel", "error", "-f", "lavfi", "-i",
                        "aevalsrc='sin(2*PI*880*t)*lt(mod(t+0.25\\,0.5)\\,0.04)':d=10:s=44100",
                        cls.click], check=True)

    @classmethod
    def tearDownClass(cls):
        cls.dir.cleanup()

    def test_a_films_shots(self):
        got = el.shots_of(self.film)
        self.assertEqual(got["shots"], [1.5, 2.5, 2.0])
        self.assertEqual(got["cuts"], [1.5, 4.0])

    def test_a_sheet_is_a_picture_with_its_times(self):
        got = el.contact_sheet(self.film, os.path.join(self.dir.name, "sheets", "film.jpg"))
        self.assertEqual(got["times"], el.sheet_times(6.0, 8))
        self.assertGreater(os.path.getsize(got["sheet"]), 2000)
        fact = er.probe(got["sheet"])
        self.assertGreater(fact["width"], fact["height"])       # four across, two down

    def test_a_tracks_beats(self):
        got = el.beats(self.click)
        self.assertAlmostEqual(got["bpm"], 120, delta=0.6)
        self.assertAlmostEqual(got["beats"][0], 0.25, delta=0.03)

    def test_a_file_with_no_sound_or_no_picture_is_said(self):
        with self.assertRaisesRegex(er.RenderError, "has no sound"):
            el.beats(self.film)
        with self.assertRaisesRegex(er.RenderError, "is not a video"):
            el.shots_of(self.click)


class TheTools(unittest.TestCase):
    SRC = (ROOT / "src" / "pipeline.py").read_text(encoding="utf-8")

    def test_the_orchestrator_has_them(self):
        self.assertIn("probe_media, read_rhythm,\n                 read_beats,", self.SRC)

    def test_a_sheet_is_made_only_when_asked_and_only_of_a_video(self):
        tool = self.SRC.split("        async def probe_media(", 1)[1].split("        @_tool", 1)[0]
        self.assertIn("contact_sheet: bool = False", tool)
        self.assertIn('if _sheets is not None and fact.get("video") and not fact.get("still"):', tool)

    def test_the_agent_is_told_to_look_first(self):
        from src.utils import canvas_hooks as ch
        text = "\n".join(ch.edit_lines([{"hook_node_id": "150", "purpose": "edit"}], lambda h: "none"))
        self.assertIn("probe_media([files], contact_sheet=True)", text)
        self.assertIn("read_rhythm(file)", text)
        self.assertIn("read_beats(file)", text)


if __name__ == "__main__":
    unittest.main()
