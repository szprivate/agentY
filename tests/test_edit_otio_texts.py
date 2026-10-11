"""The texts of an edit go into its OpenTimelineIO file.

A text is over the film at a moment, not inside a clip, so it is written as a
marker on the whole timeline: an editor sees a note of what goes there, and the
bEpic viewer draws the markers as a track of their own over the scrub bar.
"""
import unittest

from src.utils import edit_timeline as et

FACTS = {"a.mp4": {"duration": 4.0}, "b.mp4": {"duration": 4.0}}


def cut(*ops):
    return et.apply(et.new(), [{"op": "add", "src": "a.mp4"}, {"op": "add", "src": "b.mp4"}, *ops], FACTS)


class TextsAreMarkers(unittest.TestCase):
    def setUp(self):
        tl = cut({"op": "text", "preset": "lower_third", "text": "Mara Lindqvist\nKitchen",
                  "start": 4.5, "length": 3.0, "params": {"size": 40}})
        self.stack = et.to_otio(tl)["tracks"]
        self.marker = self.stack["markers"][0]

    def test_on_the_timeline_not_in_a_clip(self):
        self.assertEqual(self.marker["OTIO_SCHEMA"], "Marker.2")
        for track in self.stack["children"]:
            for child in track["children"]:
                self.assertEqual(child.get("markers", []), [])

    def test_at_its_moment_for_its_length(self):
        rng = self.marker["marked_range"]
        self.assertEqual((rng["start_time"]["value"], rng["duration"]["value"], rng["start_time"]["rate"]),
                         (108.0, 72.0, 24.0))

    def test_named_by_its_first_line_with_the_whole_text_kept(self):
        self.assertEqual(self.marker["name"], "Mara Lindqvist")
        note = self.marker["metadata"]["agentY"]
        self.assertEqual((note["kind"], note["text"], note["preset"], note["params"]),
                         ("text", "Mara Lindqvist\nKitchen", "lower_third", {"size": 40}))

    def test_a_cut_with_no_texts_has_no_markers(self):
        self.assertEqual(et.to_otio(cut())["tracks"]["markers"], [])


if __name__ == "__main__":
    unittest.main()
