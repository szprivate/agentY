"""What is pinned as "the user was explicit" comes from the user, in this turn.

In a pipeline run the block was filled from text nobody typed. A branch's
briefing and a lead's wake-up report were read as the user's message, so a
template whose name shared a word with them was "explicitly asked for", and the
files a report listed became "the user provided input file(s)". Those files were
then remembered by the conversation and demanded again on every later turn -
one attached image was pinned on 87 turns running.
"""
import types
import unittest
from pathlib import Path

from src.pipeline import Pipeline

ROOT = Path(__file__).resolve().parent.parent


def me(origin="panel", remembered=(), fresh=None, named=None, hooks=None, graph=None,
       families=()):
    obj = types.SimpleNamespace(
        _session=types.SimpleNamespace(last_user_input_images=list(remembered),
                                       named_families=list(families)),
        _canvas_graph=graph, _canvas_hooks=hooks)
    if fresh is not None:
        obj._fresh_input_images = list(fresh)
    obj._turn_origin = lambda: origin
    obj._user_opened_turn = lambda: Pipeline._user_opened_turn(obj)
    obj._match_named_templates = lambda text: named
    return obj


def pinned(obj, text="make it brighter"):
    return "\n".join(Pipeline._extract_hard_constraints(obj, text))


class WhoOpenedTheTurn(unittest.TestCase):
    def test_the_user_did_from_the_panel_or_slack(self):
        for origin in ("panel", "slack", ""):
            self.assertTrue(Pipeline._user_opened_turn(me(origin)), origin)

    def test_a_lead_or_a_branch_did_not(self):
        for origin in ("lead", "shots"):
            self.assertFalse(Pipeline._user_opened_turn(me(origin)), origin)


class TextNobodyTyped(unittest.TestCase):
    NAMED = ("medium", ["audio_stable_audio_3_medium"])

    def test_a_branch_briefing_pins_nothing(self):
        obj = me("lead", remembered=["//srv/out/a_v188.png"], fresh=[], named=self.NAMED)
        self.assertEqual(pinned(obj, "Medium shot of JONAS, use //srv/out/a_v188.png"), "")

    def test_a_wake_up_report_pins_nothing(self):
        obj = me("shots", remembered=["//srv/out/a_v188.png"], fresh=[], named=self.NAMED)
        self.assertEqual(pinned(obj, "[SHOTS REPORTING BACK] medium ... a_v188.png"), "")

    def test_the_same_words_from_the_user_still_pin(self):
        obj = me("panel", fresh=[], named=("magnific", ["api_magnific_image_relight"]))
        self.assertIn('"api_magnific_image_relight" template', pinned(obj, "use magnific"))


class FilesBelongToTheTurnTheyCameWith(unittest.TestCase):
    def test_a_file_attached_now_is_pinned(self):
        obj = me(remembered=["D:/in/street.png"], fresh=["D:/in/street.png"])
        self.assertIn("The user provided input file(s): street.png", pinned(obj))

    def test_a_file_from_an_earlier_turn_is_not_demanded_again(self):
        obj = me(remembered=["D:/in/street.png"], fresh=[])
        self.assertNotIn("provided input file", pinned(obj, "approved, go on"))

    def test_the_conversation_still_remembers_it(self):
        obj = me(remembered=["D:/in/street.png"], fresh=[])
        pinned(obj, "now make a video from it")
        self.assertEqual(obj._session.last_user_input_images, ["D:/in/street.png"])

    def test_a_caller_that_says_nothing_about_this_turn_is_read_as_before(self):
        obj = me(remembered=["D:/in/street.png"])              # no _fresh_input_images
        self.assertIn("street.png", pinned(obj))


class TheFamilyOfAnEarlierRequest(unittest.TestCase):
    def test_it_holds_for_a_follow_up_on_a_plain_canvas(self):
        obj = me(fresh=[], graph={"nodes": [1]}, families=["LTX-2"])
        self.assertIn("named the model family: LTX-2", pinned(obj, "fix these problems"))

    def test_it_does_not_hold_over_a_canvas_of_stages(self):
        obj = me(fresh=[], graph={"nodes": [1]}, hooks=[{"hook_node_id": "4"}],
                 families=["LTX-2"])
        self.assertNotIn("model family", pinned(obj, "approved, go on"))


class TheTurnSaysWhatCameWithIt(unittest.TestCase):
    def test_the_server_says_so_for_every_turn(self):
        src = (ROOT / "src" / "utils" / "agentY_server.py").read_text(encoding="utf-8")
        self.assertIn("pipeline._fresh_input_images = list(image_paths or [])", src)

    def test_paths_in_a_report_are_not_taken_as_inputs(self):
        src = (ROOT / "src" / "pipeline.py").read_text(encoding="utf-8")
        scan = src.split("user_text = self._extract_text(user_input)", 1)[1].split("synth = ", 1)[0]
        self.assertIn('self._turn_origin() != "shots"', scan)
        self.assertIn("self._fresh_input_images = []", scan)


if __name__ == "__main__":
    unittest.main()
