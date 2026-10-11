"""Text on a film: overlay clips, drawn from presets.

The model does not write animation for a title. It picks a preset - CSS that
says how the text looks and moves - and fills in the words, the moment and a
few settings. The page is built by the host, a browser draws it frame by frame
with no background, and ffmpeg lays it over the picture.
"""
import json
import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from src.utils import edit_render as er
from src.utils import edit_timeline as et
from src.utils import text_presets as tp
from src.utils import text_render as tr

ROOT = Path(__file__).resolve().parent.parent
FACTS = {"a.mp4": {"duration": 4.0}, "b.mp4": {"duration": 4.0}}


def cut(*ops):
    return et.apply(et.new(), [{"op": "add", "src": "a.mp4"}, {"op": "add", "src": "b.mp4"}, *ops], FACTS)


TITLE = {"op": "text", "preset": "title_rise", "text": "A quiet morning", "start": 0.5, "length": 2.5}


class TheBuiltInPresets(unittest.TestCase):
    def setUp(self):
        self.presets = tp.load()

    def test_the_four_kinds_the_plan_names_are_there(self):
        for name in ("title_rise", "title_letters", "lower_third", "caption", "end_card"):
            self.assertIn(name, self.presets)

    def test_every_one_passes_the_same_check_an_added_one_must(self):
        for name, preset in self.presets.items():
            self.assertEqual(tp.problems(preset), [], name)

    def test_the_agent_chooses_from_a_list_without_the_css(self):
        row = next(r for r in tp.listing() if r["name"] == "lower_third")
        self.assertNotIn("css", row)
        self.assertIn("naming a person", row["use_for"])
        self.assertEqual(row["params"]["align"], "left")


class APresetIsOnlyCss(unittest.TestCase):
    GOOD = {"name": "pop", "split": "words", "params": {"size": 80},
            "css": ".u{animation:p var(--in) both}@keyframes p{from{opacity:0}to{opacity:1}}"}

    def bad(self, **change):
        return tp.problems(dict(self.GOOD, **change))

    def test_a_plain_one_is_fine(self):
        self.assertEqual(tp.problems(self.GOOD), [])

    def test_nothing_that_runs(self):
        self.assertTrue(self.bad(css=self.GOOD["css"] + "</style><script>fetch('//x')</script>"))
        self.assertTrue(self.bad(css=".u{behavior:url(x.htc)}@keyframes p{to{opacity:1}}"))
        self.assertTrue(self.bad(css=".u{width:expression(alert(1))}@keyframes p{to{opacity:1}}"))

    def test_nothing_fetched(self):
        self.assertIn("the css contains an @import", self.bad(css="@import 'x.css';" + self.GOOD["css"]))
        self.assertTrue(any("url()" in p for p in
                            self.bad(css=".u{background:url(http://x/y.png)}@keyframes p{to{opacity:1}}")))

    def test_it_has_to_animate_something_and_be_whole(self):
        self.assertIn("the css animates nothing (no @keyframes)", self.bad(css=".u{color:red}"))
        self.assertIn("the css has unbalanced braces", self.bad(css=self.GOOD["css"] + "}"))

    def test_a_setting_holds_a_value_and_nothing_more(self):
        self.assertTrue(self.bad(params={"color": "red;} body{display:none"}))
        self.assertTrue(self.bad(params={"font": "x</style>"}))
        self.assertTrue(self.bad(split="sentences"))
        self.assertTrue(self.bad(name="Pop Title!"))


class ThePage(unittest.TestCase):
    def page(self, name="title_rise", text="A quiet morning", params=None, length=2.5):
        return tp.page(tp.get(name), text, length, 1920, 1080, params)

    def test_the_text_is_cut_the_way_the_preset_says(self):
        self.assertEqual(tp.units("A quiet morning", "words"), ["A ", "quiet ", "morning"])
        self.assertEqual(tp.units("Mara\nKitchen", "lines"), ["Mara", "Kitchen"])
        self.assertEqual(len(tp.units("Flat", "letters")), 4)
        self.assertEqual(tp.units("as it is", "none"), ["as it is"])

    def test_each_unit_is_numbered_for_the_stagger(self):
        html = self.page()
        self.assertIn('<span class="u" style="--i:0">A&nbsp;</span>', html)
        self.assertIn('<span class="u" style="--i:2">morning</span>', html)
        self.assertIn("--n:3;", html)

    def test_the_users_words_cannot_be_markup(self):
        html = self.page("caption", '<script>alert(1)</script> & "more"')
        self.assertNotIn("<script>", html)
        self.assertIn("&lt;script&gt;alert(1)&lt;/script&gt; &amp;", html)

    def test_settings_arrive_as_properties_and_seconds_are_seconds(self):
        html = self.page(params={"size": 60, "color": "#ffcc33", "in": 1.2})
        self.assertIn("--size:60px;", html)
        self.assertIn("--color:#ffcc33;", html)
        self.assertIn("--in:1.2s;", html)
        self.assertIn("--len:2.5s;", html)

    def test_position_and_alignment_place_the_block(self):
        html = self.page("lower_third", "Mara\nKitchen")
        self.assertIn("left:7%;top:82%;transform:translate(0,-50%);text-align:left", html)

    def test_a_setting_the_preset_does_not_have_is_dropped_and_a_bad_value_ignored(self):
        html = self.page(params={"glow": "9px", "color": "red;}body{display:none"})
        self.assertNotIn("glow", html)
        self.assertNotIn("display:none", html)

    def test_a_text_too_short_for_its_way_in_and_out_shares_the_time(self):
        html = self.page("end_card", "The End", length=1.15)
        self.assertIn("--in:0.7s;--out:0.45s;", html)

    def test_the_page_carries_no_script_and_reaches_nowhere(self):
        for name in tp.load():
            html = self.page(name)
            self.assertNotIn("<script", html.lower(), name)
            self.assertNotIn("http", html.lower(), name)


class AddedPresets(unittest.TestCase):
    def setUp(self):
        self._dir = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.addCleanup(self._dir.cleanup)
        env = mock.patch.dict(os.environ, {"AGENTY_TEXT_PRESETS": self._dir.name})
        env.start()
        self.addCleanup(env.stop)

    def test_one_that_is_kept_is_there_next_time(self):
        tp.save(dict(APresetIsOnlyCss.GOOD, source="https://example.org/pen", licence="MIT",
                     author="someone"))
        got = tp.get("pop")
        self.assertEqual((got["source"], got["licence"], got["builtin"]),
                         ("https://example.org/pen", "MIT", False))
        row = next(r for r in tp.listing() if r["name"] == "pop")
        self.assertTrue(row["added"])

    def test_a_broken_file_in_the_folder_is_passed_over(self):
        Path(self._dir.name, "bad.json").write_text("{not json", encoding="utf-8")
        Path(self._dir.name, "evil.json").write_text(json.dumps(
            dict(APresetIsOnlyCss.GOOD, name="evil", css="</style><script>x</script>")), encoding="utf-8")
        self.assertNotIn("evil", tp.load())
        self.assertIn("title_rise", tp.load())

    def test_it_can_be_taken_away_again(self):
        tp.save(dict(APresetIsOnlyCss.GOOD))
        self.assertTrue(tp.remove("pop"))
        self.assertIsNone(tp.get("pop"))


class OnTheTimeline(unittest.TestCase):
    def test_a_text_is_added_with_its_moment(self):
        tl = cut(TITLE)
        self.assertEqual(tl["overlays"], [{"id": "t1", "preset": "title_rise", "text": "A quiet morning",
                                           "start": 0.5, "length": 2.5, "params": {}}])
        self.assertEqual(et.summary(tl)["texts"][0]["to"], 3.0)

    def test_a_note_changes_the_one_that_is_there(self):
        tl = et.apply(cut(TITLE), [{"op": "text", "text_id": "t1", "start": 1.0,
                                    "params": {"size": 120}}])
        self.assertEqual((tl["overlays"][0]["start"], tl["overlays"][0]["text"],
                          tl["overlays"][0]["params"]), (1.0, "A quiet morning", {"size": 120}))
        self.assertEqual(len(tl["overlays"]), 1)

    def test_it_can_be_removed(self):
        self.assertEqual(et.apply(cut(TITLE), [{"op": "text_remove", "text_id": "t1"}])["overlays"], [])

    def test_words_and_a_preset_are_both_needed(self):
        with self.assertRaisesRegex(et.EditError, "give the words"):
            cut({"op": "text", "preset": "caption"})
        with self.assertRaisesRegex(et.EditError, "name the preset"):
            cut({"op": "text", "text": "hello"})

    def test_what_stops_a_render(self):
        tl = cut({**TITLE, "preset": "sparkle"}, {**TITLE, "start": 9.0}, {**TITLE, "start": 7.0})
        wrong = et.problems(tl, FACTS, list(tp.load()))
        self.assertTrue(any("names the preset 'sparkle', which does not exist" in p for p in wrong))
        self.assertTrue(any("after the film has ended (8s)" in p for p in wrong))
        self.assertTrue(any("runs to 9.5s but the film ends at 8s" in p for p in wrong))

    def test_clear_takes_the_texts_too(self):
        self.assertEqual(et.apply(cut(TITLE), [{"op": "clear"}])["overlays"], [])


class LaidOverTheFilm(unittest.TestCase):
    def setUp(self):
        self.tl = cut({"op": "audio", "src": "bed.wav"})
        self.cmd = er.command(self.tl, "o.mp4", texts=[{"pattern": "T/f_%05d.png", "start": 4.5,
                                                         "length": 3.0}])
        self.graph = self.cmd[self.cmd.index("-filter_complex") + 1]

    def test_the_frames_are_read_at_the_films_rate(self):
        i = self.cmd.index("T/f_%05d.png")
        self.assertEqual(self.cmd[i - 5:i], ["-framerate", "24", "-start_number", "1", "-i"])

    def test_it_is_moved_to_its_moment_and_only_shown_then(self):
        self.assertIn("[2:v]format=rgba,trim=duration=3,setpts=PTS-STARTPTS+4.5/TB[t0]", self.graph)
        self.assertIn("overlay=0:0:eof_action=pass:enable='between(t,4.5,7.5)'[o0]", self.graph)

    def test_the_sound_is_still_found_behind_the_texts(self):
        self.assertIn("[3:a]aresample", self.graph)
        self.assertEqual(self.cmd[self.cmd.index("-map") + 1], "[o0]")

    def test_no_texts_is_the_same_command_as_before(self):
        self.assertEqual(er.command(self.tl, "o.mp4"), er.command(self.tl, "o.mp4", texts=[]))


class TheBrowser(unittest.TestCase):
    def test_frames_for_a_length(self):
        self.assertEqual((tr.frame_count(1.5, 24), tr.frame_count(0.01, 24)), (36, 1))

    def test_none_on_the_machine_is_said_plainly(self):
        with mock.patch.object(tr, "browser_path", return_value=""):
            with self.assertRaisesRegex(tr.TextRenderError, "no Chrome or Edge was found"):
                tr.render_frames("<html></html>", "x", 10, 10, 1)

    def test_it_is_started_shut_off_from_the_network_and_from_other_machines(self):
        src = Path(tr.__file__).read_text(encoding="utf-8")
        self.assertIn('"--host-resolver-rules=MAP * ~NOTFOUND"', src)
        self.assertIn('"--remote-debugging-address=127.0.0.1"', src)
        self.assertIn('"--disable-extensions"', src)

    def test_nothing_plays_in_real_time(self):
        self.assertIn("a.pause(); a.currentTime = ms;", tr._SEEK)


@unittest.skipUnless(tr.browser_path() and shutil.which("ffmpeg"), "needs a Chromium browser and ffmpeg")
class DrawnForReal(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dir = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)

    @classmethod
    def tearDownClass(cls):
        cls.dir.cleanup()

    def test_frames_are_transparent_and_the_text_arrives(self):
        from PIL import Image
        html = tp.page(tp.get("title_rise"), "Morning", 1.0, 480, 270, {"size": 60})
        got = tr.render_frames(html, os.path.join(self.dir.name, "t"), 480, 270, 1.0, 12)
        self.assertEqual(got["frames"], 12)
        first = Image.open(got["pattern"] % 1)
        mid = Image.open(got["pattern"] % 7)
        self.assertEqual((first.mode, first.size), ("RGBA", (480, 270)))
        self.assertEqual(first.getpixel((2, 2))[3], 0)                 # no background
        self.assertGreater(mid.getchannel("A").getextrema()[1], 200)   # the word is there

    def test_a_film_with_a_text_on_it(self):
        from PIL import Image
        ff = shutil.which("ffmpeg")
        black = os.path.join(self.dir.name, "k.mp4")
        subprocess.run([ff, "-y", "-loglevel", "error", "-f", "lavfi", "-i",
                        "color=c=black:size=480x270:rate=12:duration=3", "-pix_fmt", "yuv420p", black],
                       check=True)
        tl = et.apply(et.new(12, 480, 270), [
            {"op": "add", "src": black},
            {"op": "text", "preset": "caption", "text": "Hello", "start": 1.0, "length": 1.0}],
            {black: er.probe(black)})
        film = er.render(tl, os.path.join(self.dir.name, "f.mp4"))

        def lit(t):
            png = os.path.join(self.dir.name, "x.png")
            subprocess.run([ff, "-y", "-loglevel", "error", "-ss", str(t), "-i", film,
                            "-frames:v", "1", png], check=True)
            return Image.open(png).convert("L").point(lambda v: 255 if v > 60 else 0).getbbox()
        self.assertIsNone(lit(0.5))
        self.assertIsNotNone(lit(1.5))
        self.assertIsNone(lit(2.5))


class TheTools(unittest.TestCase):
    SRC = (ROOT / "src" / "pipeline.py").read_text(encoding="utf-8")

    def test_the_agent_can_see_the_presets_and_is_told_to_use_them_as_they_are(self):
        self.assertIn("list_text_presets, ", self.SRC)
        tool = self.SRC.split("        async def list_text_presets(", 1)[1].split("        @_tool", 1)[0]
        self.assertIn("You do not write animation", tool)

    def test_a_render_checks_the_presets_exist(self):
        tool = self.SRC.split("        async def render_edit(", 1)[1].split("        @_tool", 1)[0]
        self.assertIn("wrong = _et.problems(timeline, facts, list(_tp.load()))", tool)


if __name__ == "__main__":
    unittest.main()
