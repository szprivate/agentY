"""The shots' own sound goes into the edit.

The first edit stage laid a sound bed under the film and dropped whatever the
shots themselves carried - dialogue, room tone, a door. A clip's sound now comes
with it: cut where the picture cuts, crossfaded where it dissolves, silent where
a clip has none, and mixed with the bed.
"""
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

from src.utils import edit_render as er
from src.utils import edit_timeline as et

FACTS = {"a.mp4": {"duration": 4.0, "audio": True}, "b.mp4": {"duration": 4.0, "audio": False},
         "c.mp4": {"duration": 4.0, "audio": True}, "s.png": {"duration": 0.0, "audio": False}}


def cut(*ops):
    return et.apply(et.new(), [{"op": "add", "src": s} for s in ("a.mp4", "b.mp4", "c.mp4")] + list(ops),
                    FACTS)


def graph(tl, **kw):
    cmd = er.command(tl, "o.mp4", facts=FACTS, **kw)
    return cmd, cmd[cmd.index("-filter_complex") + 1]


class WhichClipsAreHeard(unittest.TestCase):
    def test_the_ones_that_have_sound(self):
        self.assertEqual(er.clips_with_sound(cut(), FACTS), [True, False, True])

    def test_none_when_it_is_switched_off_for_the_film(self):
        self.assertEqual(er.clips_with_sound(cut({"op": "sound", "on": False}), FACTS), [False] * 3)

    def test_not_a_clip_turned_down_to_nothing(self):
        self.assertEqual(er.clips_with_sound(cut({"op": "sound", "clip": "c1", "volume": 0}), FACTS),
                         [False, False, True])

    def test_without_knowing_the_files_nothing_is_assumed(self):
        self.assertEqual(er.clips_with_sound(cut(), None), [False] * 3)


class OnTheTimeline(unittest.TestCase):
    def test_it_is_on_unless_switched_off(self):
        self.assertEqual(et.summary(cut())["clip_sound"], "on")
        self.assertEqual(et.summary(cut({"op": "sound", "on": False}))["clip_sound"], "off")

    def test_one_clips_volume(self):
        tl = cut({"op": "sound", "clip": "c3", "volume": 0.5})
        self.assertEqual(et.summary(tl)["clips"][2]["volume"], 0.5)
        self.assertNotIn("volume", et.summary(tl)["clips"][0])

    def test_every_clips_volume_at_once(self):
        self.assertEqual([c["volume"] for c in cut({"op": "sound", "volume": 0.3})["clips"]], [0.3] * 3)

    def test_what_is_refused(self):
        with self.assertRaisesRegex(et.EditError, "out of range"):
            cut({"op": "sound", "clip": "c1", "volume": 9})
        with self.assertRaisesRegex(et.EditError, "give the clip's volume"):
            cut({"op": "sound", "clip": "c1"})
        with self.assertRaisesRegex(et.EditError, "say on: true/false"):
            cut({"op": "sound"})


class JoinedAsThePictureIs(unittest.TestCase):
    def test_every_clip_gives_sound_as_long_as_itself(self):
        _cmd, g = graph(cut())
        self.assertIn("[0:a]aresample=48000,aformat=sample_fmts=fltp:channel_layouts=stereo,"
                      "atrim=duration=4,asetpts=PTS-STARTPTS,apad=whole_dur=4,volume=1[a0]", g)
        # No sound of its own: silence, so what follows is not pulled forward.
        self.assertIn("anullsrc=r=48000:cl=stereo,atrim=duration=4[a1]", g)

    def test_a_cut_is_a_join_and_a_dissolve_a_crossfade_of_the_same_length(self):
        _cmd, g = graph(cut({"op": "transition", "clip": "c3", "seconds": 1}))
        self.assertIn("[a0][a1]concat=n=2:v=0:a=1[s1]", g)
        self.assertIn("[s1][a2]acrossfade=d=1:c1=tri:c2=tri[s2]", g)
        self.assertIn("xfade=transition=fade:duration=1:offset=7", g)

    def test_a_clips_volume_is_applied(self):
        _cmd, g = graph(cut({"op": "sound", "clip": "c3", "volume": 0.5}))
        self.assertIn("apad=whole_dur=4,volume=0.5[a2]", g)

    def test_alone_it_is_the_films_sound(self):
        cmd, _g = graph(cut())
        maps = [cmd[i + 1] for i, a in enumerate(cmd) if a == "-map"]
        self.assertEqual(maps, ["[j2]", "[s2]"])
        self.assertIn("aac", cmd)

    def test_with_a_bed_the_two_are_mixed_as_they_are(self):
        cmd, g = graph(cut({"op": "audio", "src": "bed.wav", "gain": 0.25}))
        self.assertIn("volume=0.25,afade=t=out:st=11:d=1[bed]", g)
        self.assertIn("[s2][bed]amix=inputs=2:duration=first:normalize=0[aout]", g)
        self.assertEqual([cmd[i + 1] for i, a in enumerate(cmd) if a == "-map"], ["[j2]", "[aout]"])

    def test_switched_off_only_the_bed_is_left(self):
        _cmd, g = graph(cut({"op": "sound", "on": False}, {"op": "audio", "src": "bed.wav"}))
        self.assertNotIn("amix", g)
        self.assertNotIn("anullsrc", g)
        self.assertIn("[aout]", g)

    def test_no_sound_anywhere_is_a_silent_film(self):
        tl = et.apply(et.new(), [{"op": "add", "src": "b.mp4"}, {"op": "add", "src": "s.png", "length": 2}],
                      FACTS)
        cmd, g = graph(tl)
        self.assertIn("-an", cmd)
        self.assertNotIn("anullsrc", g)

    def test_a_dissolve_straight_after_a_cut_can_be_rendered(self):
        # A cut's join and a dissolve disagreed about their timebase; ffmpeg
        # refused the whole film.
        _cmd, g = graph(cut({"op": "transition", "clip": "c3", "seconds": 1}))
        self.assertEqual(g.count("setpts=PTS-STARTPTS,settb=AVTB[v"), 3)


@unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "ffmpeg is not installed")
class Heard(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dir = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        cls.ff = shutil.which("ffmpeg")

        def clip(name, tone=None):
            out = str(Path(cls.dir.name) / name)
            cmd = [cls.ff, "-y", "-loglevel", "error", "-f", "lavfi", "-i",
                   "testsrc2=size=160x90:rate=12:duration=2"]
            if tone:
                cmd += ["-f", "lavfi", "-i", f"sine=frequency={tone}:duration=2", "-c:a", "aac"]
            subprocess.run(cmd + ["-pix_fmt", "yuv420p", "-shortest", out], check=True)
            return out
        cls.a, cls.b, cls.c = clip("a.mp4", 440), clip("b.mp4"), clip("c.mp4", 880)
        cls.tl = et.apply(et.new(12, 160, 90), [
            {"op": "add", "src": cls.a}, {"op": "add", "src": cls.b}, {"op": "add", "src": cls.c},
            {"op": "transition", "clip": "c3", "seconds": 0.5},
            {"op": "sound", "clip": "c3", "volume": 0.5}], {p: er.probe(p) for p in (cls.a, cls.b, cls.c)})
        cls.film = er.render(cls.tl, str(Path(cls.dir.name) / "film.mp4"))

    @classmethod
    def tearDownClass(cls):
        cls.dir.cleanup()

    def level(self, start, end):
        import numpy as np
        raw = subprocess.run([self.ff, "-loglevel", "error", "-ss", str(start), "-t", str(end - start),
                              "-i", self.film, "-vn", "-ac", "1", "-ar", "8000", "-f", "s16le", "-"],
                             capture_output=True).stdout
        x = np.frombuffer(raw, dtype="<i2").astype("float32") / 32768.0
        return float(np.sqrt((x ** 2).mean())) if len(x) else 0.0

    def test_the_film_has_sound_and_is_as_long_as_the_picture(self):
        fact = er.probe(self.film)
        self.assertTrue(fact["audio"])
        self.assertAlmostEqual(fact["duration"], et.duration(self.tl), delta=0.15)

    def test_each_clips_sound_is_where_its_picture_is(self):
        loud, silent, half = self.level(0.4, 1.6), self.level(2.3, 3.2), self.level(4.2, 5.2)
        self.assertGreater(loud, 0.04)
        self.assertLess(silent, 0.003)
        self.assertAlmostEqual(half / loud, 0.5, delta=0.08)

    def test_switched_off_there_is_none(self):
        off = et.apply(self.tl, [{"op": "sound", "on": False}])
        self.assertFalse(er.probe(er.render(off, str(Path(self.dir.name) / "off.mp4")))["audio"])


if __name__ == "__main__":
    unittest.main()
