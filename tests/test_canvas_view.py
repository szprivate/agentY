"""The agent sees the whole canvas, and a selection stops being a permission.

Selection used to be the only way in: the agent saw the nodes the user had
selected and could write to exactly those. That made selecting a *permission*,
which is not what it is — it is a way of POINTING ("this one") — and it turned
"set the sampler to 30 steps" into "first go and click the sampler".

The graph was already being sent on every turn (`graphToPrompt` is what ComfyUI
runs on every Queue, so it costs nothing extra). It was simply never described.

Cost is why this is an index rather than a dump: a real 20-node graph is ~700
tokens as raw JSON and ~75 as one line per node, so values are truncated and the
block is capped — with `get_canvas_node` for the exact value, which is what makes
truncating safe to do at all.

    python -m unittest discover -s tests
"""

import json
import unittest

from pipeline_stub import pipeline_stub, tools
from src.utils.canvas_view import describe_canvas, node_detail, node_line


def _graph():
    return {
        "3": {"class_type": "KSampler",
              "inputs": {"seed": 42, "steps": 30, "cfg": 6.5, "sampler_name": "euler",
                         "model": ["4", 0], "positive": ["6", 0]},
              "_meta": {"title": "Main sampler"}},
        "4": {"class_type": "CheckpointLoaderSimple",
              "inputs": {"ckpt_name": "sdxl.safetensors"}},
        "6": {"class_type": "CLIPTextEncode",
              "inputs": {"text": "a cinematic wide shot of Tokyo at night, "
                                 "neon reflections in the rain, shallow depth of field, "
                                 "35mm, cool blue grade", "clip": ["4", 1]}},
        "9": {"class_type": "SaveImage",
              "inputs": {"filename_prefix": "out", "images": ["3", 0]},
              "_meta": {"title": "SaveImage"}},
    }


class LineTest(unittest.TestCase):

    def test_a_node_reads_as_id_type_title_and_values(self):
        line = node_line("3", _graph()["3"])
        self.assertTrue(line.startswith('#3 KSampler "Main sampler" — '))
        self.assertIn("steps=30", line)
        self.assertIn("cfg=6.5", line)

    def test_wired_inputs_are_left_out(self):
        """They are links; set_canvas_node_params writes values, not wiring."""
        line = node_line("3", _graph()["3"])
        self.assertNotIn("model=", line)
        self.assertNotIn("positive=", line)

    def test_a_default_title_is_not_repeated_as_a_name(self):
        """ComfyUI stores the class as the title until the user changes it."""
        self.assertEqual(node_line("9", _graph()["9"]).split(" — ")[0], "#9 SaveImage")

    def test_a_long_value_is_cut_and_says_so(self):
        line = node_line("6", _graph()["6"])
        self.assertIn("…", line)
        self.assertIn("a cinematic wide shot of Tokyo", line)

    def test_a_node_with_nothing_to_show_is_just_its_head(self):
        self.assertEqual(node_line("1", {"class_type": "PreviewImage",
                                         "inputs": {"images": ["3", 0]}}),
                         "#1 PreviewImage")

    def test_values_only_can_be_asked_for(self):
        self.assertEqual(node_line("3", _graph()["3"], values=False),
                         '#3 KSampler "Main sampler"')


class BlockTest(unittest.TestCase):

    def test_every_node_is_listed(self):
        block = describe_canvas(_graph())
        for nid in ("#3", "#4", "#6", "#9"):
            self.assertIn(nid, block)

    def test_it_says_selection_is_not_permission(self):
        block = describe_canvas(_graph())
        self.assertIn("does NOT have to be selected", block)
        self.assertIn("not permission to touch it", block)

    def test_a_selected_node_is_marked(self):
        block = describe_canvas(_graph(), ["6"])
        self.assertIn("← SELECTED", block)
        self.assertEqual(block.count("← SELECTED"), 1)

    def test_it_warns_against_editing_a_truncated_value(self):
        self.assertIn("never edit a value you have only seen truncated",
                      describe_canvas(_graph()))

    def test_it_says_editing_does_not_queue_the_graph(self):
        self.assertIn("does NOT queue the graph", describe_canvas(_graph()))

    def test_no_canvas_is_no_block(self):
        self.assertEqual(describe_canvas(None), "")
        self.assertEqual(describe_canvas({}), "")

    def test_a_normal_graph_is_described_in_full(self):
        block = describe_canvas(_graph())
        self.assertNotIn("too many nodes", block)
        self.assertNotIn("more node(s) not listed", block)
        self.assertLess(len(block), 1200, "a small graph must stay a small block")

    def test_a_big_graph_degrades_to_names_rather_than_growing_forever(self):
        big = {str(i): {"class_type": "CLIPTextEncode",
                        "inputs": {"text": "x" * 400}} for i in range(120)}
        block = describe_canvas(big)
        self.assertIn("too many nodes to describe in full", block)
        self.assertIn("#119 CLIPTextEncode", block, "still findable by id and type")

    def test_a_huge_graph_is_bounded_even_as_a_bare_list(self):
        """Listing 400 nodes by id costs ~2k tokens on EVERY turn, mostly about
        nodes nobody will touch. get_canvas_node still reaches them all."""
        huge = {str(i): {"class_type": "CLIPTextEncode",
                         "inputs": {"text": "x" * 400}} for i in range(400)}
        block = describe_canvas(huge)
        self.assertLess(len(block), 12000, "a whole turn cannot be this block")
        self.assertIn("more node(s) not listed", block)
        self.assertIn("400 node(s)", block, "the true total is still stated")

    def test_junk_entries_do_not_break_it(self):
        self.assertIn("#3", describe_canvas({"3": _graph()["3"], "x": None, "y": 7}))


class DetailTest(unittest.TestCase):
    """What a truncated line could not show — which is what makes truncation safe."""

    def test_the_full_value_comes_back_untruncated(self):
        got = node_detail(_graph(), "6")
        self.assertEqual(got["values"]["text"], _graph()["6"]["inputs"]["text"])
        self.assertNotIn("…", got["values"]["text"])

    def test_wired_inputs_are_reported_as_links_not_values(self):
        got = node_detail(_graph(), "3")
        self.assertEqual(got["wired_inputs"]["model"], "from #4 output 0")
        self.assertNotIn("model", got["values"])

    def test_a_missing_node_is_none(self):
        self.assertIsNone(node_detail(_graph(), "999"))
        self.assertIsNone(node_detail(None, "3"))


class ThroughTheToolsTest(unittest.TestCase):

    def _pipe(self, **over):
        over.setdefault("_canvas_graph", _graph())
        return pipeline_stub(**over)

    def _get(self, pipe, node_id):
        import asyncio
        return json.loads(asyncio.run(tools(pipe)["get_canvas_node"](node_id=node_id)))

    def _set(self, pipe, node_id, params):
        import asyncio
        return json.loads(asyncio.run(
            tools(pipe)["set_canvas_node_params"](node_id=node_id, params=params)))

    def test_an_unselected_node_can_be_read(self):
        got = self._get(self._pipe(), "6")
        self.assertEqual(got["class_type"], "CLIPTextEncode")
        self.assertIn("Tokyo", got["values"]["text"])

    def test_an_unselected_node_can_be_written(self):
        """The whole point: no clicking required."""
        pipe = self._pipe()
        out = self._set(pipe, "3", {"steps": 40})
        self.assertEqual(out["status"], "applied")
        self.assertEqual(pipe._canvas_selection, [], "nothing was selected")

    def test_the_edit_reaches_the_canvas(self):
        from src.utils.canvas_patch import clear, drain
        clear()
        self.addCleanup(clear)
        self._set(self._pipe(), "3", {"steps": 40})
        op = next(e for e in drain() if str(e.get("node_id")) == "3")
        self.assertEqual(op["params"], {"steps": 40})

    def test_a_selected_node_still_works_the_way_it_did(self):
        pipe = self._pipe(_canvas_selection=[
            {"id": "3", "type": "KSampler", "widgets": {"steps": 30}}])
        self.assertEqual(self._set(pipe, "3", {"steps": 40})["status"], "applied")

    def test_a_node_that_is_not_on_the_canvas_is_refused(self):
        out = self._set(self._pipe(), "999", {"steps": 40})
        self.assertIn("no node '999' on the open canvas", out["error"])
        self.assertIn("node_ids_on_canvas", out)

    def test_reading_a_node_that_is_not_there_says_so(self):
        self.assertIn("no node '999'", self._get(self._pipe(), "999")["error"])

    def test_no_canvas_this_turn_is_not_a_crash(self):
        pipe = self._pipe(_canvas_graph={})
        self.assertIn("error", self._get(pipe, "3"))
        self.assertIn("error", self._set(pipe, "3", {"steps": 40}))

    def test_an_empty_params_map_is_still_refused(self):
        self.assertIn("non-empty", self._set(self._pipe(), "3", {})["error"])


if __name__ == "__main__":
    unittest.main()
