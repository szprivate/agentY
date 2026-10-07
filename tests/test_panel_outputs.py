"""Results are shown in the chat panel, numbered the way the agent knows them.

The panel used to carry text only: a finished picture landed on the canvas and
the chat named its file. Now each image and video is shown in the chat with its
number in the conversation, and that number is the one the agent is told - there
is one numbering (the conversation's stored output list), not a list the panel
keeps and another the agent keeps.
"""

import asyncio
import inspect
import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from pipeline_stub import pipeline_stub, tools
from src.pipeline import Pipeline
from src.utils import agentY_server as srv
from src.utils import conversation_store as cs
from src.utils.models import GeneratedImage

ROOT = Path(__file__).resolve().parents[1]


class _WithFiles(unittest.TestCase):

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.dir = Path(tmp.name)
        self.files = []
        for name in ("a.png", "b.png", "c.mp4"):
            p = self.dir / name
            p.write_bytes(b"x")
            self.files.append(str(p))

    def _pipe(self, rows, session_images=()):
        pipe = pipeline_stub(_session=SimpleNamespace(session_id="thread-1",
                                                      generated_images=list(session_images)))
        pipe._shown_outputs = Pipeline._shown_outputs.__get__(pipe)
        pipe._format_image_gallery = Pipeline._format_image_gallery.__get__(pipe)
        self.enterContext(mock.patch.object(cs, "get_gallery", return_value=list(rows)))
        return pipe


class OneNumbering(_WithFiles):

    def test_the_agent_is_told_the_numbers_the_panel_prints(self):
        rows = [{"idx": 1, "path": self.files[0], "caption": "hero"},
                {"idx": 2, "path": self.files[1], "caption": ""},
                {"idx": 3, "path": self.files[2], "caption": "the clip"}]
        pipe = self._pipe(rows)
        shown = pipe._shown_outputs()
        self.assertEqual([(o["number"], o["kind"]) for o in shown], [(1, "image"), (2, "image"), (3, "video")])
        block = pipe._format_image_gallery()
        self.assertIn(f"  1. {self.files[0]}  — hero", block)
        self.assertIn(f"  3. {self.files[2]}  — the clip", block)
        self.assertIn("printed on each picture", block)

    def test_a_deleted_file_leaves_a_gap_it_does_not_renumber(self):
        """Picture #3 stays #3 when #2's file is gone: it has a 3 on it."""
        os.remove(self.files[1])
        rows = [{"idx": i + 1, "path": p, "caption": ""} for i, p in enumerate(self.files)]
        self.assertEqual([o["number"] for o in self._pipe(rows)._shown_outputs()], [1, 3])

    def test_the_sessions_caption_fills_in_where_the_stored_one_is_empty(self):
        rows = [{"idx": 1, "path": self.files[0], "caption": ""}]
        mine = [GeneratedImage(index=9, path=self.files[0], caption="a rainy alley", turn=1)]
        shown = self._pipe(rows, mine)._shown_outputs()
        self.assertEqual((shown[0]["number"], shown[0]["caption"]), (1, "a rainy alley"))

    def test_without_a_stored_list_the_sessions_own_stands_in(self):
        mine = [GeneratedImage(index=1, path=self.files[0], caption="x", turn=1)]
        self.assertEqual(self._pipe([], mine)._shown_outputs()[0]["number"], 1)

    def test_nothing_generated_is_no_block_at_all(self):
        self.assertEqual(self._pipe([])._format_image_gallery(), "")

    def test_a_restart_keeps_the_stored_numbers(self):
        src = inspect.getsource(srv)
        self.assertIn('GeneratedImage(index=int(g.get("idx") or i + 1)', src)


class AskingForTheList(_WithFiles):

    def test_the_tool_gives_the_list_as_it_is_now(self):
        rows = [{"idx": 4, "path": self.files[0], "caption": "hero"}]
        pipe = self._pipe(rows)
        self.enterContext(mock.patch.object(srv, "_show_outputs_in_panel", return_value=True))
        out = json.loads(asyncio.run(tools(pipe)["list_outputs"]()))
        self.assertEqual(out, {"outputs": [{"number": 4, "path": self.files[0], "caption": "hero",
                                            "kind": "image"}], "shown_in_panel": True})

    def test_the_prompt_tells_the_agent_about_the_numbers(self):
        prompt = (ROOT / "config" / "system_prompts" / "system_prompt.orchestrator.md").read_text(encoding="utf-8")
        section = prompt.split("### Results in the chat panel", 1)[1].split("###", 1)[0]
        for word in ("number", "list_outputs", "GENERATED IN THIS THREAD"):
            self.assertIn(word, section)


class TheSwitch(unittest.TestCase):

    def test_on_by_default_and_the_setting_or_environment_turns_it_off(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("AGENTY_PANEL_OUTPUTS", None)
            with mock.patch("src.agent._load_settings", return_value={}):
                self.assertTrue(srv._show_outputs_in_panel())
            with mock.patch("src.agent._load_settings", return_value={"show_outputs_in_panel": False}):
                self.assertFalse(srv._show_outputs_in_panel())
            with mock.patch.dict(os.environ, {"AGENTY_PANEL_OUTPUTS": "0"}), \
                 mock.patch("src.agent._load_settings", return_value={}):
                self.assertFalse(srv._show_outputs_in_panel())

    def test_the_default_is_shipped_on(self):
        self.assertIn("show_outputs_in_panel = true",
                      (ROOT / "config" / "settings.default.toml").read_text(encoding="utf-8"))

    def test_every_output_event_carries_its_number_and_the_switch(self):
        src = inspect.getsource(srv)
        self.assertIn("index = cs.add_gallery_image(thread_id, p, role)", src)
        self.assertEqual(src.count('"show": _show_outputs_in_panel()'), 3,
                         "a turn's outputs, web references, and a background result")


if __name__ == "__main__":
    unittest.main()
