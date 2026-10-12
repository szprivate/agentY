"""Changing the graph that is open - the ways a turn went wrong, and what stops each.

All four are from the local-orchestrator runs of 2026-10-12 (complex suite, the
two "modify this workflow" cases):

* a NEW workflow was built when the request was to change the open one;
* the edit list arrived as JSON text and was refused, three times running;
* a plan was announced as a numbered list, and the turn ended;
* an edit was rejected, and the turn ended with nothing changed.
"""
import json
import unittest
from types import SimpleNamespace
from unittest import mock

from src.utils import intent_guard as ig

GRAPH = {
    "1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": "sd15.safetensors"}},
    "4": {"class_type": "EmptyLatentImage", "inputs": {"width": 512, "height": 512, "batch_size": 1}},
    "6": {"class_type": "VAEDecode", "inputs": {"samples": ["5", 0], "vae": ["1", 2]}},
    "7": {"class_type": "SaveImage", "inputs": {"images": ["6", 0], "filename_prefix": "x"}},
}


class AnAnnouncedPlan(unittest.TestCase):

    SEEN = ("Good — there's a RealESRGAN x4 model available. Since it's x4, I'll use scale factor 2 "
            "to get the requested 2x upscale (using half of what the model supports). Now let me make "
            "all the edits:\n\n1. Set batch_size=4 on EmptyLatentImage\n2. Add UpscaleModelLoader + "
            "ImageUpscaleWithModel after VAEDecode\n3. Rewire SaveImage to the upscaled image")

    def test_a_plan_that_was_announced_and_not_carried_out(self):
        self.assertEqual(ig.unfinished(self.SEEN), "Now let me make all the edits:")

    def test_also_when_it_arrives_on_one_line(self):
        flat = self.SEEN.replace("\n", " ")
        self.assertEqual(ig.unfinished(flat), "Now let me make all the edits:")

    def test_a_report_that_ends_on_a_list_is_a_report(self):
        for text in ("Two things changed:\n- the sampler is now dpmpp_2m\n- steps went from 20 to 30.",
                     "Done. The graph now has:\n1. a batch of 4\n2. a 2x upscale before the save.",
                     "Built it. What I changed:\n- batch_size 4\n- an upscale pass"):
            with self.subTest(text=text[:30]):
                self.assertEqual(ig.unfinished(text), "")


def _pipe(graph=None, answer="Right - that input does not exist. Let me look at it again."):
    from src.pipeline import Pipeline
    pipe = Pipeline.__new__(Pipeline)
    pipe._verbose = False
    pipe._canvas_graph = dict(graph if graph is not None else GRAPH)
    pipe._canvas_edits = []
    pipe._canvas_hooks = None
    pipe._autoloop = None
    pipe._canvas_edit_rejected = None
    pipe._canvas_edit_first_said = False
    pipe._session = SimpleNamespace(open_graph_mode=True)
    pipe._orchestrator_agent = SimpleNamespace(messages=[{"role": "assistant", "content": [{"text": answer}]}])
    return pipe


def _ask(pipe, done=0, signalled=()):
    with mock.patch("src.utils.workflow_signal.peek", return_value=list(signalled)), \
            mock.patch("src.utils.settings.load_settings", return_value={"intent_guard": True}):
        return pipe._unfinished_intent_input(done)


class ATurnThatEndsOnARejectedEdit(unittest.TestCase):

    ERRORS = ["op 2: ImageUpscaleWithModel has no input 'scale_by' (it has: image, upscale_model)."]

    def test_it_is_sent_back_with_the_errors(self):
        pipe = _pipe(answer="The model upscales by its own factor, so there is no scale_by.")
        pipe._canvas_edit_rejected = list(self.ERRORS)
        note = _ask(pipe)
        self.assertIn("REJECTED", note)
        self.assertIn("scale_by", note)

    def test_once_per_refusal(self):
        pipe = _pipe(answer="There is no scale_by.")
        pipe._canvas_edit_rejected = list(self.ERRORS)
        self.assertTrue(_ask(pipe))
        self.assertEqual(_ask(pipe, done=1), "")

    def test_a_question_to_the_user_is_theirs_to_answer(self):
        pipe = _pipe(answer="That node has no scale input. Should I resize after the upscale instead?")
        pipe._canvas_edit_rejected = list(self.ERRORS)
        self.assertEqual(_ask(pipe), "")

    def test_an_edit_that_then_went_through_ends_the_matter(self):
        pipe = _pipe(answer="Done: batch of 4, and a 2x upscale before the save.")
        self.assertEqual(_ask(pipe), "")


class ANewWorkflowWhereAChangeWasAsked(unittest.TestCase):

    def _refusal(self, pipe, setting=True):
        with mock.patch("src.utils.settings.load_settings", return_value={"edit_open_graph_first": setting}):
            return pipe._canvas_edit_first_refusal()

    def test_the_first_request_is_answered_with_the_open_graph(self):
        pipe = _pipe()
        got = self._refusal(pipe)
        self.assertEqual(got["status"], "graph_is_open")
        self.assertIn("edit_canvas_graph", got["message"])

    def test_the_second_goes_through(self):
        pipe = _pipe()
        self._refusal(pipe)
        self.assertIsNone(self._refusal(pipe))

    def test_not_when_the_canvas_is_empty_or_was_already_changed(self):
        self.assertIsNone(self._refusal(_pipe(graph={})))
        edited = _pipe()
        edited._canvas_edits = ["node 4: batch_size=4"]
        self.assertIsNone(self._refusal(edited))

    def test_not_outside_the_open_graph_mode_nor_in_a_hook_turn(self):
        own_tab = _pipe()
        own_tab._session = SimpleNamespace(open_graph_mode=False)
        self.assertIsNone(self._refusal(own_tab))
        hooks = _pipe()
        hooks._canvas_hooks = [{"node_id": "9"}]
        self.assertIsNone(self._refusal(hooks))

    def test_the_setting_turns_it_off(self):
        self.assertIsNone(self._refusal(_pipe(), setting=False))


class TheEditListAsText(unittest.TestCase):
    """edit_canvas_graph takes its list as JSON text too (the tool's own type says so)."""

    def test_the_tool_accepts_text(self):
        import re
        from pathlib import Path
        src = (Path(__file__).resolve().parent.parent / "src" / "pipeline.py").read_text(encoding="utf-8")
        self.assertRegex(src, r"async def edit_canvas_graph\(ops: list \| str,")
        body = src[src.index("async def edit_canvas_graph("):]
        body = body[:body.index("@_tool")]
        self.assertLess(body.index("if isinstance(ops, str):"), body.index("_ce.plan(graph, ops, schemas)"))
        self.assertIn("ops = json.loads(ops)", body)


class ADeprecatedNode(unittest.TestCase):

    def _schema(self, info):
        from agenty_core.tools import comfyui
        client = SimpleNamespace(get=lambda path: {"N": info})
        with mock.patch.object(comfyui, "get_client", return_value=client):
            return json.loads(comfyui._get_node_schema_one("N"))

    def test_it_says_so_outright(self):
        got = self._schema({"display_name": "Apply ControlNet (DEPRECATED)", "deprecated": True, "input": {}})
        self.assertTrue(got["deprecated"])
        self.assertIn("replaces it", got["note"])

    def test_a_current_node_carries_no_such_note(self):
        got = self._schema({"display_name": "Apply ControlNet", "input": {}})
        self.assertNotIn("deprecated", got)
        self.assertNotIn("note", got)


if __name__ == "__main__":
    unittest.main()
