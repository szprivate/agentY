"""A baked stage needs a name, not a paragraph.

Baking exists to turn a multi-step agent task into a native workflow you can read
and re-run. The tool asked the agent for a subgraph name and offered "e.g. the
hook's directive" as the example — so it got the directive, in full, on every
stage: a collapsed node, a breadcrumb and a node-library entry each labelled with
a paragraph about volumetric light and 35mm film grain. The one artifact whose
job was to make the chain legible was the least legible thing on the canvas.

The contract now asks for two to five words. This is what happens when it doesn't
get them: the name is derived rather than trusted, because a bad name is a worse
name and a pasted directive is a broken canvas.

    python -m unittest discover -s tests
"""

import unittest

from src.utils.subgraph_bake import short_name


class ShortNameTest(unittest.TestCase):

    def test_a_good_name_is_left_exactly_as_it_is(self):
        for name in ("Upscale 2x + grain", "Animate the scene",
                     "Extract last frame", "Relight to golden hour"):
            with self.subTest(name=name):
                self.assertEqual(short_name(name), name)

    def test_a_whole_directive_becomes_the_thing_it_asks_for(self):
        self.assertEqual(
            short_name("generate a cinematic wide shot of a lighthouse at dusk, "
                       "moody volumetric light, 35mm film grain, shallow depth of field"),
            "Generate a cinematic wide shot")

    def test_leading_filler_is_dropped_before_the_clause_is_taken(self):
        """'Then, animate the scene' splits on that comma into 'Then' — a name for nothing."""
        self.assertEqual(short_name("Then, animate the scene"), "Animate the scene")
        self.assertEqual(short_name("Please upscale the image"), "Upscale the image")
        self.assertEqual(short_name("Your task is to relight the frame"), "Relight the frame")

    def test_it_never_ends_on_a_dangling_word(self):
        """Cutting at five words lands on one often; a trailing 'and' reads as damage."""
        for directive in ("upscale the image 2x and add a subtle film grain",
                          "extract the last frame of the video",
                          "write a caption for this image describing the mood",
                          "animate this into a 5 second clip"):
            with self.subTest(directive=directive):
                got = short_name(directive)
                self.assertTrue(got)
                last = got.split()[-1].lower()
                self.assertNotIn(last, {"and", "of", "for", "a", "the", "into", "this"})
                self.assertFalse(last.isdigit(), got)

    def test_it_stays_within_what_a_collapsed_node_can_show(self):
        long_one = "Supercalifragilistic transformation pipeline reconstruction assembly"
        self.assertLessEqual(len(short_name(long_one)), 42)
        self.assertLessEqual(len(short_name("a " * 200)), 42)

    def test_it_never_cuts_a_word_in_half(self):
        got = short_name("Supercalifragilistic transformation pipeline reconstruction")
        for word in got.split():
            self.assertIn(word, "Supercalifragilistic transformation pipeline reconstruction")

    def test_nothing_usable_falls_back_rather_than_naming_it_blank(self):
        for junk in ("", None, "   ", "the of and to", ",,,"):
            with self.subTest(junk=junk):
                self.assertEqual(short_name(junk), "Baked stage")
        self.assertEqual(short_name("", fallback="Stage 3"), "Stage 3")

    def test_the_builder_shortens_what_it_is_handed(self):
        """Not only the tool: the builder is where the name reaches the canvas."""
        from src.utils.subgraph_bake import build_baked_workflow
        stage = {"graph": {"nodes": [{"id": 1, "type": "SaveImage", "inputs": [],
                                      "outputs": [{"name": "IMAGE", "type": "IMAGE"}]}],
                           "links": []},
                 "name": "animate the scene into a five second clip with a slow push-in",
                 "inputs": [], "outputs": []}
        baked = build_baked_workflow([stage])
        got = baked["definitions"]["subgraphs"][0]["name"]
        self.assertEqual(got, "Animate the scene")

    def test_the_tool_no_longer_offers_the_directive_as_the_example(self):
        """It is what produced the paragraph names in the first place."""
        import inspect
        from src.tools import bake
        doc = inspect.getsource(bake)
        self.assertNotIn("e.g. the hook's directive", doc)
        self.assertIn("2 to 5 words", doc)


if __name__ == "__main__":
    unittest.main()
