"""A downloaded image has to appear, without anyone building a workflow for it.

From a real turn. Asked to research a car from every angle, the agent identified
it, searched, and downloaded eight good reference photos into ComfyUI's input
directory. Then, with no way to SHOW them, it tried `update_workflow` with an
image path as the workflow (error), and followed that with:

    prepare_workflow("Create a simple workflow with 9 LoadImage nodes arranged
    on the canvas for viewing reference images … No generation needed")

Building a graph to look at pictures it already had on disk.

The first fix published what the Reference Scout reported in its manifest, which
did not help here at all: the orchestrator has `download_image` in its OWN
toolset and never went through `run_web_search`. So the publish belongs at the
download itself, where every caller passes — the orchestrator directly, and the
scout inside the specialist.

`download_image` lives in the shared layer, which has no canvas. The host injects
a sink (as `annotate_image` already does) and the MCP server leaves it unset.

    python -m unittest discover -s tests
"""

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from agenty_core.tools import image_io

PNG = (b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
       b"\x08\x06\x00\x00\x00\x1f\x15\xc4\x89\x00\x00\x00\nIDATx\x9cc\x00\x01"
       b"\x00\x00\x05\x00\x01\r\n-\xb4\x00\x00\x00\x00IEND\xaeB`\x82")


class _Resp:
    status_code = 200

    def __init__(self, data, content_type="image/png"):
        self.content = data
        self.headers = {"content-type": content_type}

    def raise_for_status(self):
        pass


class DownloadPublishesTest(unittest.TestCase):
    """The download itself announces the file, whoever asked for it."""

    def setUp(self):
        self._dir = tempfile.TemporaryDirectory()
        self.input_dir = Path(self._dir.name)
        self.published: list = []
        image_io.set_output_sink(self.published.append)

    def tearDown(self):
        image_io.set_output_sink(None)
        self._dir.cleanup()

    def _download(self, name="ref.png", subfolder="", data=PNG, client=None,
                  content_type="image/png"):
        """Run download_image against a stubbed web and a stubbed ComfyUI."""
        if client is None:
            client = mock.Mock()
            client.post.return_value = {"name": name, "subfolder": subfolder,
                                        "type": "input"}
        with mock.patch.object(image_io, "requests") as req, \
             mock.patch.object(image_io, "get_client", return_value=client), \
             mock.patch.object(image_io, "comfy_input_dir",
                               return_value=str(self.input_dir)):
            req.get.return_value = _Resp(data, content_type)
            return json.loads(image_io.download_image("https://example.com/x.png"))

    def test_a_download_is_announced_to_the_host(self):
        out = self._download()
        self.assertNotIn("error", out)
        self.assertEqual(self.published, [out["saved_to"]],
                         "nothing was published — the picture would never show")

    def test_what_is_published_is_the_file_on_disk(self):
        """A loader node needs a real path, not the URL or the bare name."""
        out = self._download(name="ref.png")
        self.assertEqual(self.published[0], str(self.input_dir / "ref.png"))

    def test_a_subfolder_is_part_of_the_published_path(self):
        """The car run put everything in `golf_mk1_refs/`; the path must say so."""
        self._download(name="a.png", subfolder="golf_mk1_refs")
        self.assertEqual(self.published[0],
                         str(self.input_dir / "golf_mk1_refs" / "a.png"))

    def test_every_download_gets_announced(self):
        for n in ("a.png", "b.png", "c.png"):
            self._download(name=n)
        self.assertEqual(len(self.published), 3)

    def test_a_failed_download_announces_nothing(self):
        with mock.patch.object(image_io, "requests") as req:
            req.get.side_effect = RuntimeError("403 Forbidden")
            out = json.loads(image_io.download_image("https://example.com/x.png"))
        self.assertIn("error", out)
        self.assertEqual(self.published, [],
                         "a broken loader node would have been dropped")

    def test_something_that_is_not_really_an_image_is_not_shown(self):
        """A hotlink block or login page served at `.../photo.png`.

        The format is read from the content-type or the URL extension, so such a
        response can still come back looking like a PNG. Whether the download
        reports that is its own business — but it must not become a canvas node
        that shows nothing and fails when the graph runs.
        """
        self._download(data=b"<html>not an image</html>")
        self.assertEqual(self.published, [],
                         "a node pointing at an HTML page would have been dropped")

    def test_a_sink_that_throws_does_not_break_the_download(self):
        """Showing the picture is secondary to having it."""
        image_io.set_output_sink(mock.Mock(side_effect=RuntimeError("panel gone")))
        out = self._download()
        self.assertNotIn("error", out)
        self.assertTrue(out["saved_to"])

    def test_with_no_host_listening_it_just_downloads(self):
        """The MCP server has no canvas and registers no sink."""
        image_io.set_output_sink(None)
        out = self._download()
        self.assertNotIn("error", out)

    def test_an_unresolvable_input_dir_publishes_nothing(self):
        """No path means no node; better silent than pointing at nowhere."""
        client = mock.Mock()
        client.post.return_value = {"name": "ref.png", "subfolder": "",
                                    "type": "input"}
        with mock.patch.object(image_io, "requests") as req, \
             mock.patch.object(image_io, "get_client", return_value=client), \
             mock.patch.object(image_io, "comfy_input_dir", return_value=None):
            req.get.return_value = _Resp(PNG)
            out = json.loads(image_io.download_image("https://example.com/x.png"))
        self.assertNotIn("error", out)
        self.assertEqual(self.published, [])


class PipelineWiringTest(unittest.TestCase):
    """The sink is only useful if the pipeline actually registers it."""

    def test_the_pipeline_registers_the_download_sink(self):
        import inspect

        from src.pipeline import Pipeline
        src = inspect.getsource(Pipeline.__init__)
        self.assertIn("_set_download_sink(self._register_output_path)", src,
                      "downloads are published nowhere — the sink is never set")

    def test_the_tool_says_downloading_is_showing(self):
        """Or the agent builds a workflow of LoadImage nodes to display them."""
        doc = image_io.download_image.__doc__ or ""
        self.assertIn("canvas", doc)
        self.assertIn("do not follow it", doc.lower())


if __name__ == "__main__":
    unittest.main()
