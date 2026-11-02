"""Titles the bEpic viewer saves (clips on extra video tracks) are read back as texts."""
import unittest

from src.utils import edit_timeline as et

FACTS = {"a.mp4": {"duration": 4.0}, "b.mp4": {"duration": 4.0}}
RATE = 24.0


def rt(s):
    return {"OTIO_SCHEMA": "RationalTime.1", "rate": RATE, "value": s * RATE}


def rng(a, d):
    return {"OTIO_SCHEMA": "TimeRange.1", "start_time": rt(a), "duration": rt(d)}


def title(words, length, agenty=None, **extra):
    meta = {"bepic": {"title": {"text": words, "font": "Inter", "size": 8}},
            "agentY": agenty if agenty is not None else {"kind": "text", "text": words}}
    clip = {"OTIO_SCHEMA": "Clip.1", "name": words.splitlines()[0], "metadata": meta, "effects": [], "markers": [],
            "source_range": rng(0, length),
            "media_reference": {"OTIO_SCHEMA": "GeneratorReference.1", "generator_kind": "bepic.title",
                                "name": "", "metadata": {}}}
    clip.update(extra)
    return clip


def gap(d):
    return {"OTIO_SCHEMA": "Gap.1", "name": "", "metadata": {}, "source_range": rng(0, d)}


def track(name, children):
    return {"OTIO_SCHEMA": "Track.1", "name": name, "kind": "Video", "metadata": {}, "children": children}


def doc(extra_tracks, markers=()):
    base = et.to_otio(et.apply(et.new(), [{"op": "add", "src": "a.mp4"}, {"op": "add", "src": "b.mp4"}], FACTS))
    stack = base["tracks"]
    stack["children"].extend(extra_tracks)
    stack["markers"] = list(markers)
    return base


class Titles(unittest.TestCase):
    def test_from_agenty_marker_keeps_preset(self):
        d = doc([track("V2", [gap(2.0), title("Mara\nKitchen", 3.0,
                {"kind": "text", "text": "Mara\nKitchen", "preset": "lower_third", "params": {"size": 40}})])])
        o = et.from_otio(d)["overlays"]
        self.assertEqual(len(o), 1)
        self.assertEqual((o[0]["preset"], o[0]["text"], o[0]["start"], o[0]["length"], o[0]["params"]),
                         ("lower_third", "Mara\nKitchen", 2.0, 3.0, {"size": 40}))

    def test_viewer_made_title_is_caption(self):
        o = et.from_otio(doc([track("V2", [title("Hello", 1.5)])]))["overlays"]
        self.assertEqual((o[0]["preset"], o[0]["text"], o[0]["params"]), ("caption", "Hello", {}))

    def test_disabled_left_out(self):
        d = doc([track("V2", [title("Off", 1.0, enabled=False), gap(1.0), title("Gone", 1.0)])])
        d["tracks"]["children"][-1]["children"][2]["metadata"]["bepic"]["disabled"] = True
        self.assertEqual(et.from_otio(d)["overlays"], [])
        self.assertFalse(any("titles' look" in w for w in et.otio_dropped(d)))

    def test_time_order_with_markers(self):
        mk = {"OTIO_SCHEMA": "Marker.2", "name": "Old", "marked_range": rng(1.0, 1.0),
              "metadata": {"agentY": {"kind": "text", "text": "Old"}}}
        d = doc([track("V2", [gap(4.0), title("Late", 1.0)]), track("V3", [title("Early", 1.0)])], [mk])
        o = et.from_otio(d)["overlays"]
        self.assertEqual([x["text"] for x in o], ["Early", "Old", "Late"])
        self.assertEqual([x["id"] for x in o], ["t1", "t2", "t3"])

    def test_old_cut_with_markers_unchanged(self):
        tl = et.apply(et.new(), [{"op": "add", "src": "a.mp4"}, {"op": "add", "src": "b.mp4"},
                                 {"op": "text", "preset": "lower_third", "text": "Hi", "start": 1, "length": 2}], FACTS)
        d = et.to_otio(tl)
        self.assertEqual(et.otio_dropped(d), [])
        o = et.from_otio(d)["overlays"]
        self.assertEqual((o[0]["preset"], o[0]["text"], o[0]["start"], o[0]["length"]), ("lower_third", "Hi", 1.0, 2.0))

    def test_dropped_wording(self):
        out = et.otio_dropped(doc([track("V2", [title("Hello", 1.0)])]))
        self.assertEqual(len(out), 1)
        self.assertIn("titles' look (font, style, position, outline, shadow, box) is not taken over", out[0])

    def test_real_clip_on_extra_track_still_left_out(self):
        c = title("x", 1.0)
        c["media_reference"] = {"OTIO_SCHEMA": "ExternalReference.1", "target_url": "file:///c.mp4"}
        c["name"] = "c.mp4"
        self.assertTrue(any("left out" in w for w in et.otio_dropped(doc([track("V2", [c])]))))


if __name__ == "__main__":
    unittest.main()
