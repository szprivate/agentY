"""show_in_viewer and "open the edit": an edit render opens the viewer's Edit panel.

The viewer transport (ComfyUI's HTTP API) and ComfyUI's folders are stubs.

    python -m unittest tests.test_viewer_open
"""

import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from src.utils import agentY_server as srv
from src.utils import media_sink as ms


class OpenEdit(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.addCleanup(tmp.cleanup)
        self.base = Path(tmp.name)
        self.input = self.base / "input"
        self.output = self.base / "output"
        self.input.mkdir()
        self.output.mkdir()
        for bucket in (ms._intent, ms._viewer_sent):
            bucket.clear()
        for p in (mock.patch.object(srv, "_comfy_input_dir", return_value=self.input),
                  mock.patch.dict(os.environ, {"AGENTY_VIEWER_PUSH": "1"}, clear=False)):
            p.start()
            self.addCleanup(p.stop)
        self.sent = mock.Mock(return_value={"ok": True, "shown": 1, "edit": True})
        p = mock.patch("agenty_core.tools.show_in_viewer.open_in_viewer", self.sent)
        p.start()
        self.addCleanup(p.stop)

    def film(self, folder, name="cut_v001.mp4"):
        film = folder / name
        film.write_bytes(b"x")
        otio = film.with_suffix(".otio")
        otio.write_text("{}")
        return str(film), str(otio)

    def test_film_in_comfy_output_is_opened_in_place_with_edit_panel(self):
        film, otio = self.film(self.output)
        res = srv.open_edit_in_viewer(film, otio, "My edit")
        self.assertTrue(res["ok"])
        args, kw = self.sent.call_args
        self.assertEqual(args[0], [film])
        self.assertEqual(kw["panel"], "edit")
        self.assertTrue(kw["focus"])
        self.assertEqual(kw["group"], "My edit")

    def test_film_elsewhere_is_staged_with_its_cut(self):
        elsewhere = self.base / "elsewhere"
        elsewhere.mkdir()
        film, otio = self.film(elsewhere)
        srv.open_edit_in_viewer(film, otio)
        shown = Path(self.sent.call_args.args[0][0])
        self.assertEqual(shown.parent, self.input / srv.STAGE_SUBFOLDER)
        self.assertTrue(shown.is_file())
        self.assertTrue(shown.with_suffix(".otio").is_file())

    def test_the_generic_delivery_does_not_push_it_again(self):
        film, otio = self.film(self.output)
        ms.note(film, "My edit", "edit")
        srv.open_edit_in_viewer(film, otio)
        self.assertIsNone(ms.take(film))
        self.assertFalse(ms.push_to_viewer(film, wait=True))

    def test_switched_off(self):
        film, otio = self.film(self.output)
        with mock.patch.dict(os.environ, {"AGENTY_VIEWER_OPEN_EDIT": "0"}):
            res = srv.open_edit_in_viewer(film, otio)
        self.assertIn("skipped", res)
        self.sent.assert_not_called()

    def test_a_missing_viewer_never_raises(self):
        film, otio = self.film(self.output)
        self.sent.side_effect = RuntimeError("down")
        self.assertIn("error", srv.open_edit_in_viewer(film, otio))


class Registered(unittest.TestCase):
    def test_tool_is_in_the_orchestrator_and_skill_is_scoped(self):
        import src.tools as tools
        self.assertIn("show_in_viewer", tools._SHARED_CORE_TOOLS)
        from src import agent
        self.assertIn("show-in-viewer", agent._ORCH_SKILL_NAMES)
        self.assertTrue(agent._skill_sources(["show-in-viewer"]))


if __name__ == "__main__":
    unittest.main()
