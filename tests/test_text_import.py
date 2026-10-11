"""Adding a text preset from a web page.

Pages of text effects are full of things worth having as titles - and none of
them is a title: they run on scroll, follow the mouse, loop for ever. So a demo
is read, never copied. The agent writes a preset from it, the host checks it and
shows the user a test clip, and it is kept only on the user's word, with where
it came from and under what licence.
"""
import os
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from src.utils import text_import as ti
from src.utils import text_presets as tp
from src.utils import text_render as tr

ROOT = Path(__file__).resolve().parent.parent

INDEX = """
<h3><a href="/x">Staggered Bars Reveal</a></h3>
<p>by <img alt="Alvaro Montoro"></p>
<a href="https://codepen.io/alvaromontoro/pen/abpBmxz">Demo</a>
<h3>Text Shimmer</h3>
<a href="https://codepen.io/someone/full/Exqbaem">Demo</a>
<a href="https://codepen.io/someone/pen/Exqbaem">again, the same pen</a>
<h3>Neon</h3><a href="https://codepen.io/third_one/pen/JVWKZE">Demo</a>
"""

PREVIEW = """<html><head><title>CodePen</title>
<style>.referer-warning{background:black}</style></head><body>
<div class="referer-warning">Do not enter passwords</div>
<iframe id="result" srcdoc="<!DOCTYPE html><html><head><title>Header idea</title>
<style>h1{animation:slide 2s infinite}@keyframes slide{from{opacity:0}to{opacity:1}}</style></head>
<body><header><h1 class=&quot;big&quot;>&amp;Aacute;lvaro</h1></header>
<script>window.console = window.console || function(t) {};</script>
<script src=&quot;https://cdn.example/gsap.min.js&quot;></script>
<script>document.addEventListener('mousemove', follow)</script></body></html>"></iframe>
</body></html>"""


class ACollectionPage(unittest.TestCase):
    def setUp(self):
        self.demos = ti.demo_links(INDEX)

    def test_every_demo_once_in_page_order(self):
        self.assertEqual([d["url"] for d in self.demos], [
            "https://codepen.io/alvaromontoro/pen/abpBmxz",
            "https://codepen.io/someone/pen/Exqbaem",
            "https://codepen.io/third_one/pen/JVWKZE"])

    def test_with_who_made_it_and_the_heading_before_it(self):
        self.assertEqual((self.demos[0]["author"], self.demos[0]["title"]),
                         ("alvaromontoro", "Staggered Bars Reveal"))
        self.assertEqual(self.demos[2]["title"], "Neon")

    def test_a_page_with_demos_is_an_index_and_a_pen_is_a_demo(self):
        with mock.patch.object(ti, "fetch", return_value=INDEX):
            self.assertEqual(ti.read("https://example.org/effects")["kind"], "index")
        with mock.patch.object(ti, "fetch", return_value=PREVIEW) as fetch:
            got = ti.read("https://codepen.io/alvaromontoro/pen/abpBmxz")
        self.assertEqual(got["kind"], "demo")
        # Read from the preview host: the site itself turns automated requests away.
        self.assertEqual(fetch.call_args.args[0], "https://cdpn.io/alvaromontoro/fullcpgrid/abpBmxz")


class OneDemo(unittest.TestCase):
    def setUp(self):
        with mock.patch.object(ti, "fetch", return_value=PREVIEW):
            self.demo = ti.read_demo("https://codepen.io/alvaromontoro/pen/abpBmxz")

    def test_it_is_the_demo_and_not_the_page_around_it(self):
        self.assertIn("@keyframes slide", self.demo["css"])
        self.assertNotIn("referer-warning", self.demo["css"])
        self.assertNotIn("Do not enter passwords", self.demo["markup"])
        self.assertIn('<h1 class="big">', self.demo["markup"])

    def test_the_hosts_own_script_is_not_counted_as_the_demos(self):
        self.assertNotIn("window.console", self.demo["script"])
        self.assertIn("mousemove", self.demo["script"])

    def test_what_drives_it_is_said(self):
        self.assertEqual(self.demo["driven_by"], ["the mouse", "an endless loop"])
        self.assertTrue(self.demo["uses_script"])
        self.assertEqual(self.demo["libraries"], ["https://cdn.example/gsap.min.js"])

    def test_where_it_came_from_and_under_what_licence(self):
        self.assertEqual((self.demo["source"], self.demo["author"], self.demo["title"]),
                         ("https://codepen.io/alvaromontoro/pen/abpBmxz", "alvaromontoro", "Header idea"))
        self.assertIn("MIT", self.demo["licence"])

    def test_a_long_stylesheet_is_cut_and_says_so(self):
        got = ti.parts_of("<style>" + "a{b:c}" * 5000 + "</style><body></body>")
        self.assertLess(len(got["css"]), 14100)
        self.assertIn("more characters */", got["css"])


class WhatMayBeFetched(unittest.TestCase):
    def test_only_the_public_web(self):
        for url in ("http://127.0.0.1:5000/agentY/runs", "http://localhost:8188/", "file:///C:/x",
                    "ftp://example.org/x", "", "http://192.168.1.10/", "http://10.0.0.1/"):
            with self.assertRaises(ti.ImportError_, msg=url):
                ti.fetch(url)

    def test_a_redirect_to_this_machine_is_refused_too(self):
        src = Path(ti.__file__).read_text(encoding="utf-8")
        self.assertIn("if final.hostname and not _public(final.hostname):", src)

    def test_a_name_that_does_not_resolve_is_not_public(self):
        self.assertFalse(ti._public("no-such-host.invalid"))


class ThePipelinesTools(unittest.TestCase):
    SRC = (ROOT / "src" / "pipeline.py").read_text(encoding="utf-8")

    def tool(self, name):
        return self.SRC.split(f"        async def {name}(", 1)[1].split("        @_tool", 1)[0]

    def test_the_orchestrator_has_them(self):
        self.assertIn("find_text_demos, try_text_preset,\n                 keep_text_preset,", self.SRC)

    def test_a_preset_is_checked_before_anything_is_drawn(self):
        body = self.tool("try_text_preset")
        self.assertLess(body.index("wrong = _tp.problems(preset)"), body.index("_ti.test_clip, preset"))
        self.assertIn('"what_to_do": "Fix them and call this again. Nothing was shown."', body)

    def test_trying_shows_a_clip_and_keeps_nothing(self):
        body = self.tool("try_text_preset")
        self.assertIn("self._show_again(", body)
        self.assertNotIn("_tp.save(", body)
        self.assertIn("It is NOT in the library yet", body)

    def test_a_built_in_cannot_be_replaced(self):
        self.assertIn("is a built-in preset - give yours", self.tool("try_text_preset"))

    def test_it_is_kept_only_on_the_users_words_in_this_turn(self):
        body = self.tool("keep_text_preset")
        self.assertIn("heard = self._users_words()", body)
        self.assertIn("why = _quote(user_said, heard)", body)
        self.assertLess(body.index("why = _quote(user_said, heard)"), body.index("path = _tp.save(preset)"))
        self.assertLess(body.index("if not heard.strip():"), body.index("path = _tp.save(preset)"))

    def test_only_one_that_was_shown_can_be_kept(self):
        body = self.tool("keep_text_preset")
        self.assertIn("is waiting - show the user a", body)
        self.assertLess(body.index("if preset is None:"), body.index("path = _tp.save(preset)"))

    def test_the_source_and_licence_travel_with_it(self):
        body = self.tool("try_text_preset")
        for field in ('"source": str(source or "").strip()', '"author": str(author or "").strip()',
                      '"licence": str(licence or "").strip()'):
            self.assertIn(field, body)

    def test_the_agent_is_told_a_demo_is_not_copied(self):
        self.assertIn("you never copy it", self.tool("find_text_demos"))
        rules = self.tool("try_text_preset")
        self.assertIn("It runs ON THE CLOCK, once", rules)
        self.assertIn("one that follows the mouse is not a preset", rules)


@unittest.skipUnless(tr.browser_path() and shutil.which("ffmpeg"), "needs a Chromium browser and ffmpeg")
class ATestClip(unittest.TestCase):
    def test_a_written_preset_becomes_a_film_to_look_at(self):
        from src.utils import edit_render as er
        preset = {"name": "bars_reveal", "split": "words",
                  "params": {"size": 80, "in": 0.5, "out": 0.3, "stagger": 0.1},
                  "css": ".u{display:inline-block;clip-path:inset(0 100% 0 0);"
                         "animation:br-in var(--in) ease-out calc(var(--i) * var(--stagger)) forwards}"
                         ".text{animation:br-out var(--out) linear calc(var(--len) - var(--out)) forwards}"
                         "@keyframes br-in{to{clip-path:inset(0 0 0 0)}}@keyframes br-out{to{opacity:0}}"}
        self.assertEqual(tp.problems(preset), [])
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            out = ti.test_clip(preset, "Three short words", os.path.join(tmp, "clip.mp4"),
                               seconds=1.5, width=640, height=360, fps=12)
            fact = er.probe(out)
        self.assertEqual((fact["duration"], fact["width"], fact["height"]), (1.5, 640, 360))


if __name__ == "__main__":
    unittest.main()
