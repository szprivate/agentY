"""Photographing the canvas: the tool, and what it tells the agent afterwards.

The picture is drawn by the browser (only it has the user's layout, colours and
collapsed nodes), so the tool's own job is small and entirely about honesty:
decode what the page sent, put it somewhere, and describe it accurately enough
that the agent does not oversell it.

The one that matters is `detail`. Below zoom 0.5 LiteGraph draws no node text at
all — measured on the live canvas by sampling `low_quality` across zooms — so a
big graph comes back as shape-and-wiring with every label missing. A tool that
reported that as "here is your workflow" would be handing the user a picture and
a false description of it.

    python -m unittest discover -s tests
"""

import asyncio
import base64
import json
import unittest
from pathlib import Path
from unittest import mock

from pipeline_stub import pipeline_stub, tools

# A real 1x1 PNG: the tool decodes what it is given, so a fake string would test
# the error path instead of the happy one.
PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmM"
    "IQAAAABJRU5ErkJggg==")
DATA_URL = "data:image/png;base64," + base64.b64encode(PNG).decode()


def page(**over):
    """What the browser sends back for a screenshot probe."""
    reply = {"data_url": DATA_URL, "width": 1200, "height": 800, "scale": 0.9,
             "nodes": 12, "detail": "full", "scoped": False,
             "workflow": "my_workflow"}
    reply.update(over)
    return reply


def run_tool(pipe, reply, **kwargs):
    """Call screenshot_canvas with the page answering *reply*."""
    with mock.patch("src.utils.canvas_probe.request", return_value=reply) as req:
        out = json.loads(asyncio.run(tools(pipe)["screenshot_canvas"](**kwargs)))
    return out, req


class CaptureTest(unittest.TestCase):

    def setUp(self):
        self.pipe = pipeline_stub()
        self._written = []

    def tearDown(self):
        for p in self._written:
            try:
                Path(p).unlink()
            except OSError:
                pass

    def _run(self, reply, **kwargs):
        out, req = run_tool(self.pipe, reply, **kwargs)
        if out.get("path"):
            self._written.append(out["path"])
        return out, req

    def test_the_image_reaches_disk_intact(self):
        out, _ = self._run(page())
        self.assertEqual(out["status"], "captured")
        path = Path(out["path"])
        self.assertTrue(path.exists())
        self.assertEqual(path.read_bytes(), PNG, "the PNG was mangled on the way")
        self.assertEqual(out["size_bytes"], len(PNG))

    def test_it_is_filed_apart_from_generated_media(self):
        """A picture OF the work, not a piece of it: it must not read as output."""
        out, _ = self._run(page())
        parts = Path(out["path"]).parts
        self.assertIn("screenshots", parts)
        self.assertNotIn("output_images", parts)

    def test_the_workflow_names_the_file(self):
        out, _ = self._run(page(workflow="0193_cilia_ai_v002"))
        self.assertIn("0193_cilia_ai_v002", Path(out["path"]).name)

    def test_a_hostile_workflow_name_cannot_escape_the_folder(self):
        out, _ = self._run(page(workflow="../../etc/passwd"))
        parts = Path(out["path"]).parts
        self.assertIn("screenshots", parts)
        self.assertNotIn("..", parts)

    def test_an_overview_says_the_text_is_missing(self):
        """The honesty case. Below zoom 0.5 LiteGraph draws no node labels."""
        out, _ = self._run(page(detail="overview", scale=0.31,
                                note="…shows its layout and wiring but NO node text."))
        self.assertEqual(out["detail"], "overview")
        self.assertIn("NO node text", out["note"])

    def test_a_full_capture_carries_no_warning_note(self):
        out, _ = self._run(page())
        self.assertEqual(out["detail"], "full")
        self.assertNotIn("note", out)

    def test_the_selection_is_asked_for_and_reported(self):
        out, req = self._run(page(scoped=True, nodes=4), only_selected=True)
        self.assertEqual(req.call_args.args[1], {"only_selected": True})
        self.assertEqual(out["scope"], "selected nodes")
        self.assertIn("selection", Path(out["path"]).name)

    def test_the_whole_graph_is_the_default(self):
        out, req = self._run(page())
        self.assertEqual(req.call_args.args[1], {})
        self.assertEqual(out["scope"], "whole graph")

    def test_a_closed_tab_is_reported_and_not_retried(self):
        out, _ = self._run({"error": "the ComfyUI page did not answer in 20s — "
                                     "it is probably closed", "timeout": True})
        self.assertIn("closed", out["error"])
        self.assertIn("Do not retry", out["what_to_do"])
        self.assertNotIn("path", out)

    def test_a_page_error_is_passed_through_plainly(self):
        out, _ = self._run({"error": "the canvas is empty — there is nothing to show"})
        self.assertIn("nothing to show", out["error"])
        self.assertNotIn("what_to_do", out, "an empty canvas is not a closed tab")

    def test_a_reply_with_no_image_is_an_error_not_an_empty_file(self):
        out, _ = self._run({"width": 100})
        self.assertIn("no image data", out["error"])

    def test_undecodable_data_does_not_write_a_broken_png(self):
        out, _ = self._run(page(data_url="data:image/png;base64,!!!not base64!!!"))
        self.assertIn("could not be decoded", out["error"])
        self.assertNotIn("path", out)

    def test_a_dry_run_draws_nothing(self):
        pipe = pipeline_stub(_dry_run=True)
        with mock.patch("src.utils.canvas_probe.request") as req:
            out = json.loads(asyncio.run(tools(pipe)["screenshot_canvas"]()))
        self.assertIn("dry run", out["error"])
        req.assert_not_called()

    def test_it_points_at_how_to_send_the_picture(self):
        out, _ = self._run(page())
        self.assertIn("send_to_slack", out["message"])


class ToolRegistrationTest(unittest.TestCase):

    def test_the_orchestrator_actually_has_it(self):
        self.assertIn("screenshot_canvas", tools(pipeline_stub()))

    def test_the_docstring_warns_against_using_it_to_read_values(self):
        """A picture is the wrong way to answer "what is the seed"."""
        doc = tools(pipeline_stub())["screenshot_canvas"].__doc__ or ""
        self.assertIn("do NOT need this", doc.replace("You ", "").replace("you ", ""))
        self.assertIn("get_canvas_node", doc)


if __name__ == "__main__":
    unittest.main()
