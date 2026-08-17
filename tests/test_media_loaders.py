"""Which loader node a finished file lands in, and what gets written into it.

Two shapes exist on a ComfyUI canvas and they take different things: a core
`LoadImage` names a file inside ComfyUI's input directory, a VHS `(Path)` loader
holds an absolute path and reads the original where it was written. The pairing
is the whole point — a node handed the other shape's value looks completely
normal on the canvas and fails only when it runs, which is late and confusing.

    python -m unittest discover -s tests
"""

import json
import unittest
from unittest import mock

from pipeline_stub import pipeline_stub, tools
from src.utils.media_loaders import CANDIDATES, candidates, takes_absolute_path


class ChoiceTest(unittest.TestCase):

    def test_the_path_loader_is_preferred_when_it_exists(self):
        """No copy, and the node points at the file the run actually produced."""
        self.assertEqual(candidates("image")[0], "VHS_LoadImagePath")
        self.assertEqual(candidates("video")[0], "VHS_LoadVideoPath")

    def test_the_core_node_is_still_there_to_fall_back_to(self):
        """The frontend takes the first one REGISTERED, so a ComfyUI without the
        pack installed keeps exactly the behaviour it always had."""
        self.assertIn("LoadImage", candidates("image"))
        self.assertIn("VHS_LoadVideo", candidates("video"))

    def test_an_unknown_kind_offers_nothing_rather_than_guessing(self):
        self.assertEqual(candidates("audio"), [])
        self.assertEqual(candidates(""), [])

    def test_the_list_handed_out_is_a_copy(self):
        """A caller that sorts or trims its list must not edit everyone else's."""
        candidates("image").clear()
        self.assertTrue(candidates("image"))

    def test_the_server_sends_the_frontend_this_same_list(self):
        from src.utils.agentY_server import _NODE_CANDIDATES
        self.assertIs(_NODE_CANDIDATES, CANDIDATES)


class ShapeTest(unittest.TestCase):

    def test_every_vhs_path_loader_reads_as_one(self):
        for name in ("VHS_LoadImagePath", "VHS_LoadVideoPath",
                     "VHS_LoadVideoFFmpegPath", "VHS_LoadImagesPath"):
            with self.subTest(name=name):
                self.assertTrue(takes_absolute_path(name))

    def test_the_name_loaders_do_not(self):
        for name in ("LoadImage", "LoadVideo", "VHS_LoadVideo", "VHS_LoadImages",
                     "AgentYImageCollector"):
            with self.subTest(name=name):
                self.assertFalse(takes_absolute_path(name))

    def test_nothing_at_all_is_a_name_loader(self):
        """Unknown is the safe way to be wrong: the staged copy always exists."""
        for value in (None, "", "   "):
            self.assertFalse(takes_absolute_path(value))


PRODUCED = "D:/out/refined_00007_.png"


class IterateFeedbackTest(unittest.TestCase):
    """`iterate_step` writes the running result back into the wired loader.

    Whichever shape the user wired, the value written has to be one that node can
    read — otherwise the loop points at nothing from the second step on, and the
    user sees a refine loop that quietly stops refining.
    """

    def _pipe(self, loader):
        graph = {
            "5": {"class_type": "CLIPTextEncode", "inputs": {"text": "a cat"}},
            "7": {"class_type": loader, "inputs": {"image": "start.png"}},
        }
        hooks = [{
            "hook_node_id": "20", "purpose": "iterate", "directive": "refine it",
            "anchors": [{"node_id": "7", "type": loader, "to_input": "anchors.anchor0"}],
            "targets": [{"node_id": "5", "to_input": "text", "type": "CLIPTextEncode"}],
        }]
        return pipeline_stub(_canvas_base_prompt=graph, _canvas_hooks=hooks,
                             _canvas_graph=graph)

    def _run(self, loader):
        import asyncio
        from src.utils import canvas_patch
        canvas_patch.clear()
        self.addCleanup(canvas_patch.clear)

        async def _fake_run(*a, **kw):
            kw["collected_paths"].append(PRODUCED)
            yield "done"

        pipe = self._pipe(loader)
        with mock.patch("src.executor.execute_workflow", _fake_run), \
             mock.patch("src.tools.image_handling._upload_one",
                        return_value={"name": "refined_00007_.png"}) as up:
            out = json.loads(asyncio.run(tools(pipe)["iterate_step"](prompt="warmer")))
        patch = next((e for e in canvas_patch.drain() if e.get("node_id") == "7"), None)
        return pipe, out, patch, up

    def test_a_path_loader_is_given_the_path_and_nothing_is_staged(self):
        _pipe, out, patch, up = self._run("VHS_LoadImagePath")
        self.assertEqual(out["status"], "done")
        self.assertEqual(patch["params"]["image"], PRODUCED)
        self.assertEqual(up.call_count, 0, "a path loader reads the original in place")

    def test_a_name_loader_is_given_the_staged_name(self):
        _pipe, out, patch, up = self._run("LoadImage")
        self.assertEqual(out["status"], "done")
        self.assertEqual(patch["params"]["image"], "refined_00007_.png")
        self.assertEqual(up.call_count, 1, "it can only see the input directory")

    def test_the_graph_that_ran_is_untouched_by_the_difference(self):
        """Only the reference shape changes; the step itself is the same step."""
        for loader in ("VHS_LoadImagePath", "LoadImage"):
            with self.subTest(loader=loader):
                _pipe, out, _patch, _up = self._run(loader)
                self.assertEqual(out["generation"], 1)
                self.assertEqual(out["output"], PRODUCED)

    def test_the_history_remembers_a_ref_the_node_can_still_read(self):
        """A go-back writes an OLD entry's ref into that same node, so an entry
        recorded in the wrong shape breaks a turn or three later."""
        pipe, _out, _patch, _up = self._run("VHS_LoadImagePath")
        self.assertEqual(pipe._iterate_history[-1]["input_ref"], PRODUCED)
        pipe, _out, _patch, _up = self._run("LoadImage")
        self.assertEqual(pipe._iterate_history[-1]["input_ref"], "refined_00007_.png")


if __name__ == "__main__":
    unittest.main()
