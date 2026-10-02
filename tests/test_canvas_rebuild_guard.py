"""A canvas fix is not followed by a rebuild that throws it away.

Asked to correct a workflow on the canvas, the orchestrator set the value there
and then called prepare_workflow "to build it again". That builds from the
template, not from the canvas: a duration corrected from 120 s to 30 s came
back as 120 s three attempts in a row. prepare_workflow now says so, once.

    python -m unittest discover -s tests
"""

import asyncio
import json
import unittest

from pipeline_stub import pipeline_stub, tools


def _graph():
    return {"98": {"class_type": "EmptyAceStep1.5LatentAudio", "inputs": {"seconds": 120, "batch_size": 1}}}


class _Reached(Exception):
    """prepare_workflow got past the guard (the stub has no researcher)."""


def _pipe(**over):
    def _lease():
        raise _Reached()
    return pipeline_stub(_canvas_graph=_graph(), _canvas_full_graph=lambda: True,
                         _researcher_lease=_lease, **over)


def _call(pipe, name, **kw):
    return json.loads(asyncio.run(tools(pipe)[name](**kw)))


class RebuildGuardTest(unittest.TestCase):

    def test_prepare_after_a_canvas_edit_is_stopped_and_names_the_edit(self):
        pipe = _pipe()
        self.assertEqual(_call(pipe, "set_canvas_node_params", node_id="98", params={"seconds": 30})["status"],
                         "applied")
        out = _call(pipe, "prepare_workflow", request="the same graph, 30 seconds", staged_inputs=[])
        self.assertEqual(out["status"], "canvas_already_edited")
        self.assertIn("seconds=30", out["canvas_edits"][0])

    def test_it_is_said_once_so_a_real_second_request_goes_through(self):
        pipe = _pipe()
        _call(pipe, "set_canvas_node_params", node_id="98", params={"seconds": 30})
        _call(pipe, "prepare_workflow", request="x", staged_inputs=[])
        with self.assertRaises(_Reached):
            _call(pipe, "prepare_workflow", request="a new picture of a cat", staged_inputs=[])

    def test_no_canvas_edit_no_guard(self):
        with self.assertRaises(_Reached):
            _call(_pipe(), "prepare_workflow", request="a cat", staged_inputs=[])

    def test_hook_turns_write_to_the_canvas_and_still_prepare(self):
        pipe = _pipe(_canvas_hooks=[{"node_id": "1"}])
        _call(pipe, "set_canvas_node_params", node_id="98", params={"seconds": 30})
        with self.assertRaises(_Reached):
            _call(pipe, "prepare_workflow", request="x", staged_inputs=[])


if __name__ == "__main__":
    unittest.main()
