"""The other half: a registered path has to come out as a canvas node.

`test_web_search_outputs` proves the tool REGISTERS what the scout staged. That is
only the first of three links, and registering into a list nobody drains would
look identical from there:

    run_web_search  -> session.current_output_paths     (tested there)
    _check_outputs  -> SSE {"type": "output", ...}      (tested HERE)
    panel           -> injectNode -> loader node        (the panel's own path,
                                                         shared with generated
                                                         media)

So this drives a real turn through `_run_pipeline_turn` with a stand-in pipeline
that registers a file the way the tool does, and reads the SSE queue the panel
would receive.

The one thing genuinely different about a web download is that it is ALREADY in
ComfyUI's input directory — `download_image` uploads it there — where a generated
file has to be copied in. That is the case worth pinning: staging must recognise
the file is already home and hand back its name rather than copying it onto
itself or renaming it.

    python -m unittest discover -s tests
"""

import queue
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from src.utils import agentY_server as srv


class _FakePipeline:
    """Registers a file mid-turn, exactly as `run_web_search` now does."""

    def __init__(self, session, paths):
        self._session = session
        self._paths = paths

    async def stream_async(self, user_input, **kwargs):
        for p in self._paths:
            if p not in self._session.current_output_paths:
                self._session.current_output_paths.append(p)
        # A text chunk: the turn has to say something for the panel to render,
        # and it is also what makes the emitter sweep for new outputs mid-turn.
        yield {"data": "Found it."}

    # Everything else the turn touches on the pipeline.
    def _pending_execution_paths(self):
        return []


class ReachesTheCanvasTest(unittest.TestCase):

    def setUp(self):
        self._dir = tempfile.TemporaryDirectory()
        self.root = Path(self._dir.name)
        # Stand in for ComfyUI's input directory — where download_image puts it.
        self.input_dir = self.root / "input"
        self.input_dir.mkdir()
        self.session = SimpleNamespace(current_output_paths=[], session_id="t1")

    def tearDown(self):
        self._dir.cleanup()

    def _turn(self, paths):
        """Run one turn and return the SSE events the panel would receive."""
        out_q: queue.Queue = queue.Queue()
        pipe = _FakePipeline(self.session, [str(p) for p in paths])
        with mock.patch.object(srv, "_agent_ref", pipe), \
             mock.patch.object(srv, "_comfy_input_dir", return_value=self.input_dir), \
             mock.patch.object(srv, "_resolve_qa_briefing", return_value=None), \
             mock.patch.object(srv.cs, "add_gallery_image"), \
             mock.patch.object(srv.cs, "add_message"), \
             mock.patch.object(srv, "_maybe_autotitle", create=True), \
             mock.patch.object(srv, "_save_state", create=True):
            srv._run_pipeline_turn("t1", "find me a diner", [], out_q,
                                   "req1", {"emitted": False})
        events = []
        while True:
            try:
                ev = out_q.get_nowait()
            except queue.Empty:
                break
            if ev is None:
                break
            events.append(ev)
        return events

    def _outputs(self, events):
        return [e for e in events if e.get("type") == "output"]

    def test_a_downloaded_reference_comes_out_as_a_canvas_output(self):
        """The whole point: the panel is told to drop a node for it."""
        img = self.input_dir / "diner_a1b2c3.jpg"
        img.write_bytes(b"\x89PNG\r\n\x1a\n")

        outs = self._outputs(self._turn([img]))

        self.assertEqual(len(outs), 1, "no output event — nothing would be dropped")
        ev = outs[0]
        self.assertEqual(ev["kind"], "image")
        self.assertEqual(ev["path"], str(img))
        # `filename` is what the loader node's widget gets. Without it the node
        # has nothing to load.
        self.assertEqual(ev["filename"], "diner_a1b2c3.jpg")
        self.assertTrue(ev["node_candidates"], "no node type to drop")

    def test_a_file_already_in_the_input_dir_is_not_copied_or_renamed(self):
        """The web-download case. A generated file is copied in; this one is home.

        Copying it onto itself would fail, and renaming it would hand the node a
        name for a file the manifest does not mention.
        """
        img = self.input_dir / "diner_a1b2c3.jpg"
        img.write_bytes(b"\x89PNG\r\n\x1a\n")

        outs = self._outputs(self._turn([img]))

        self.assertEqual(outs[0]["filename"], "diner_a1b2c3.jpg")
        self.assertEqual([p.name for p in self.input_dir.iterdir()],
                         ["diner_a1b2c3.jpg"], "a duplicate was made")

    def test_several_references_each_get_their_own_node(self):
        imgs = []
        for name in ("diner_1.jpg", "diner_2.png"):
            f = self.input_dir / name
            f.write_bytes(b"\x89PNG\r\n\x1a\n")
            imgs.append(f)

        outs = self._outputs(self._turn(imgs))

        self.assertEqual([e["filename"] for e in outs],
                         ["diner_1.jpg", "diner_2.png"])

    def test_a_path_that_vanished_is_not_announced(self):
        """A file deleted between staging and the sweep must not become a node."""
        outs = self._outputs(self._turn([self.input_dir / "never_written.jpg"]))
        self.assertEqual(outs, [])

    def test_the_same_reference_is_announced_once(self):
        """The sweep runs on every chunk; a node per chunk would be a mess."""
        img = self.input_dir / "diner.jpg"
        img.write_bytes(b"\x89PNG\r\n\x1a\n")

        outs = self._outputs(self._turn([img, img]))

        self.assertEqual(len(outs), 1)


if __name__ == "__main__":
    unittest.main()
