"""A memorizing hook answers once, and keeps answering until something changes.

The case is a hook that reads an image and writes a description, wired into a
graph the user iterates on all afternoon: the same vision call, the same answer,
twenty times, for a picture that never moved. With ``memorize`` on, the value is
stored against a fingerprint of everything feeding the hook and put straight back
into the graph next time.

Which makes the fingerprint the whole feature. It has to move when the inputs move
— a different image, a rewire, an edit three nodes upstream, a changed directive —
and it has to stay still when nothing that matters changed, including where the
value is delivered. Both halves are tested here, because a cache that invalidates
too eagerly is just a slower run, and one that invalidates too rarely is wrong.

    python -m unittest discover -s tests
"""

import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from pipeline_stub import pipeline_stub
from src.pipeline import Pipeline
from src.utils import hook_cache as hc
from src.utils.canvas_hooks import describe_hooks


def _graph():
    """A hook reading one loaded image, writing into a prompt node."""
    return {
        "10": {"class_type": "LoadImage", "inputs": {"image": "hero.png"}},
        "11": {"class_type": "ImageScale", "inputs": {"image": ["10", 0], "width": 1024}},
        "20": {"class_type": "CLIPTextEncode", "inputs": {"text": ""}},
        "21": {"class_type": "SaveImage", "inputs": {"images": ["20", 0],
                                                     "filename_prefix": "out"}},
    }


def _hook(memorize=True, directive="Describe the STYLE of the wired image."):
    return {
        "hook_node_id": "30", "purpose": "text", "directive": directive,
        "memorize": memorize, "freeze": False,
        "anchors": [{"node_id": "11", "to_input": "anchors.anchor0", "from_output_slot": 0}],
        "targets": [{"node_id": "20", "to_input": "text", "to_input_type": "STRING"}],
    }


class FingerprintTest(unittest.TestCase):
    def setUp(self):
        # Never touch the real project store, and never call a live ComfyUI.
        self.enterContext(mock.patch("src.utils.hook_cache._file_stamp", return_value=""))

    def test_the_same_hook_and_graph_key_the_same(self):
        self.assertEqual(hc.fingerprint(_hook(), _graph()),
                         hc.fingerprint(_hook(), _graph()))

    def test_a_different_image_upstream_releases_it(self):
        g = _graph()
        g["10"]["inputs"]["image"] = "villain.png"
        self.assertNotEqual(hc.fingerprint(_hook(), _graph()),
                            hc.fingerprint(_hook(), g),
                            "the answer was about the other picture")

    def test_an_edit_further_upstream_releases_it_too(self):
        g = _graph()
        g["11"]["inputs"]["width"] = 512      # two nodes back from the hook
        self.assertNotEqual(hc.fingerprint(_hook(), _graph()), hc.fingerprint(_hook(), g))

    def test_rewiring_the_anchor_releases_it(self):
        h = _hook()
        h["anchors"] = [{"node_id": "10", "to_input": "anchors.anchor0",
                         "from_output_slot": 0}]
        self.assertNotEqual(hc.fingerprint(_hook(), _graph()), hc.fingerprint(h, _graph()))

    def test_changing_the_prompt_releases_it(self):
        self.assertNotEqual(
            hc.fingerprint(_hook(), _graph()),
            hc.fingerprint(_hook(directive="Describe the COLOUR instead."), _graph()))

    def test_changing_a_setting_releases_it(self):
        h = _hook()
        h["freeze"] = True
        self.assertNotEqual(hc.fingerprint(_hook(), _graph()), hc.fingerprint(h, _graph()))

    def test_moving_the_output_elsewhere_releases_it(self):
        h = _hook()
        h["targets"] = [{"node_id": "22", "to_input": "text"}]
        self.assertNotEqual(hc.fingerprint(_hook(), _graph()), hc.fingerprint(h, _graph()))

    def test_a_downstream_change_does_NOT_release_it(self):
        """Renaming the save prefix doesn't change what the picture looks like."""
        g = _graph()
        g["21"]["inputs"]["filename_prefix"] = "something_else"
        self.assertEqual(hc.fingerprint(_hook(), _graph()), hc.fingerprint(_hook(), g))

    def test_the_memorize_toggle_itself_is_not_in_the_key(self):
        """Off has to resolve to the key On wrote under, or it could never forget."""
        self.assertEqual(hc.fingerprint(_hook(memorize=True), _graph()),
                         hc.fingerprint(_hook(memorize=False), _graph()))

    def test_a_cycle_does_not_hang(self):
        g = {"1": {"class_type": "A", "inputs": {"x": ["2", 0]}},
             "2": {"class_type": "B", "inputs": {"y": ["1", 0]}}}
        h = _hook()
        h["anchors"] = [{"node_id": "1"}]
        self.assertTrue(hc.fingerprint(h, g))

    def test_a_file_that_changed_behind_its_name_releases_it(self):
        with mock.patch("src.utils.hook_cache._file_stamp",
                        side_effect=["100:1", "250:9"]):   # same name, new bytes
            before = hc.fingerprint(_hook(), _graph())
            after = hc.fingerprint(_hook(), _graph())
        self.assertNotEqual(before, after,
                            "ComfyUI's input dir is where hero.png gets overwritten")


class StoreTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.enterContext(mock.patch.object(hc, "cache_dir",
                                            side_effect=lambda create=False: self.tmp))

    def test_a_value_survives_the_round_trip(self):
        hc.write("abc", "a warm, low-contrast alley at night", hook="30")
        got = hc.read("abc")
        self.assertEqual(got["value"], "a warm, low-contrast alley at night")
        self.assertEqual(got["hook"], "30")
        self.assertIn("when", got)

    def test_a_miss_is_just_none(self):
        self.assertIsNone(hc.read("nothing-here"))

    def test_forgetting_removes_it(self):
        hc.write("abc", "x")
        self.assertTrue(hc.forget("abc"))
        self.assertIsNone(hc.read("abc"))
        self.assertFalse(hc.forget("abc"), "forgetting twice is not an error")

    def test_an_empty_value_is_not_stored(self):
        self.assertFalse(hc.write("abc", "   "))

    def test_no_project_store_means_no_cache_and_no_crash(self):
        with mock.patch.object(hc, "cache_dir", return_value=None):
            self.assertFalse(hc.write("abc", "x"))
            self.assertIsNone(hc.read("abc"))
            self.assertFalse(hc.forget("abc"))

    def test_the_toggle_is_read_the_way_the_frontend_sends_it(self):
        self.assertTrue(hc.memorizing({"memorize": True}))
        self.assertTrue(hc.memorizing({"memorize": "true"}))
        self.assertFalse(hc.memorizing({"memorize": False}))
        self.assertFalse(hc.memorizing({}), "a hook from an older frontend never caches")


class TurnTest(unittest.TestCase):
    """What the pipeline does with it at the start of a turn."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.enterContext(mock.patch.object(hc, "cache_dir",
                                            side_effect=lambda create=False: self.tmp))
        self.enterContext(mock.patch("src.utils.hook_cache._file_stamp", return_value=""))

    @staticmethod
    def _pipe(hooks, graph=None):
        return pipeline_stub(_canvas_base_prompt=graph or _graph(), _canvas_hooks=hooks)

    def _apply(self, pipe):
        Pipeline._apply_hook_cache(pipe)

    def test_a_hit_is_put_back_into_the_graph_without_asking_anyone(self):
        hooks = [_hook()]
        pipe = self._pipe(hooks)
        self._apply(pipe)                                    # miss: nothing stored yet
        self.assertNotIn("_cached", hooks[0])
        hc.write(hooks[0]["_cache_key"], "a warm, low-contrast alley")

        hooks2 = [_hook()]
        pipe2 = self._pipe(hooks2)
        self._apply(pipe2)
        self.assertEqual(hooks2[0]["_cached"]["value"], "a warm, low-contrast alley")
        self.assertEqual(hooks2[0]["_cached"]["targets"], ["20"])
        self.assertEqual(pipe2._canvas_base_prompt["20"]["inputs"]["text"],
                         "a warm, low-contrast alley",
                         "the value has to be IN the graph, not just reported")

    def test_a_changed_input_is_a_miss(self):
        hooks = [_hook()]
        pipe = self._pipe(hooks)
        self._apply(pipe)
        hc.write(hooks[0]["_cache_key"], "the old answer")

        g = _graph()
        g["10"]["inputs"]["image"] = "villain.png"
        hooks2 = [_hook()]
        self._apply(self._pipe(hooks2, g))
        self.assertNotIn("_cached", hooks2[0])

    def test_switching_the_toggle_off_releases_what_was_stored(self):
        hooks = [_hook()]
        self._apply(self._pipe(hooks))
        key = hooks[0]["_cache_key"]
        hc.write(key, "the answer")

        self._apply(self._pipe([_hook(memorize=False)]))
        self.assertIsNone(hc.read(key), "off is how the user forces a fresh result")

    def test_the_value_is_stored_when_the_agent_places_it(self):
        import asyncio
        hooks = [_hook()]
        pipe = self._pipe(hooks)
        self._apply(pipe)
        tool = {t.tool_name: t for t in Pipeline._build_delegation_tools(pipe)}
        out = json.loads(asyncio.run(tool["place_canvas_text"](
            hook_node_id="30", text="a warm, low-contrast alley")))
        self.assertEqual(out["status"], "placed")
        self.assertEqual(hc.read(hooks[0]["_cache_key"])["value"],
                         "a warm, low-contrast alley")

    def test_a_hook_that_does_not_memorize_stores_nothing(self):
        import asyncio
        hooks = [_hook(memorize=False)]
        pipe = self._pipe(hooks)
        self._apply(pipe)
        tool = {t.tool_name: t for t in Pipeline._build_delegation_tools(pipe)}
        asyncio.run(tool["place_canvas_text"](hook_node_id="30", text="an answer"))
        self.assertIsNone(hc.read(hooks[0]["_cache_key"]))


class BlockTest(unittest.TestCase):
    """How a cached hook reads to the agent — it must not be offered as work."""

    def test_a_cached_hook_is_reported_as_done_not_assigned(self):
        h = _hook()
        h["_cached"] = {"value": "a warm, low-contrast alley", "targets": ["20"],
                        "when": "2026-08-14T10:00:00"}
        block = describe_hooks([h], _graph())
        self.assertIn("ALREADY DONE", block)
        self.assertIn("a warm, low-contrast alley", block)
        self.assertIn("filled node(s) 20", block)
        self.assertNotIn("TEXT hook 30", block, "it must not also be listed as work")
        self.assertIn("apply_canvas_hooks(resolutions=[])", block,
                      "with every hook cached, running the graph is all that's left")

    def test_a_consumer_is_handed_the_remembered_value_not_a_promise(self):
        producer = _hook()
        producer["_cached"] = {"value": "warm sodium light", "targets": [], "when": ""}
        consumer = {
            "hook_node_id": "31", "purpose": "text",
            "directive": "Write a prompt using the style from the previous hook.",
            "anchors": [{"node_id": "30"}], "targets": [{"node_id": "20",
                                                         "to_input": "text"}],
        }
        block = describe_hooks([producer, consumer], _graph())
        self.assertIn('the remembered value of hook 30: "warm sodium light"', block)
        self.assertNotIn("the value you produce for hook 30", block)

    def test_an_ordinary_graph_gains_nothing(self):
        self.assertNotIn("ALREADY DONE", describe_hooks([_hook()], _graph()))


if __name__ == "__main__":
    unittest.main()
