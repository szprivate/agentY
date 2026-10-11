"""show_in_panel: a picture the user should be looking at is put in front of them.

A result was shown once, when it was made, in the conversation that made it.
There was no way to show it again - so a lead never showed what its branches
made (they had been shown in conversations nobody reads), and asked the user to
choose between pictures described in words.
"""
import os
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

from src.pipeline import Pipeline
from src.utils import conversation_store as cs

ROOT = Path(__file__).resolve().parent.parent


class Files(unittest.TestCase):
    def setUp(self):
        self._dir = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.addCleanup(self._dir.cleanup)
        env = mock.patch.dict(os.environ, {
            "AGENTY_CONVERSATION_DB": os.path.join(self._dir.name, "c.sqlite")})
        env.start()
        self.addCleanup(env.stop)
        cs._INITIALISED = False
        self.addCleanup(setattr, cs, "_INITIALISED", False)
        cs.create_thread("t", thread_id="lead")

    def file(self, name):
        p = os.path.join(self._dir.name, name)
        Path(p).write_bytes(b"x")
        return p

    def me(self, own=()):
        session = types.SimpleNamespace(session_id="lead", generated_images=[],
                                        current_output_paths=list(own))
        obj = types.SimpleNamespace(_session=session, _verbose=False, _chain_output_paths=[])
        for name in ("_shown_outputs", "_show_items", "_show_again", "_show_unseen"):
            setattr(obj, name, getattr(Pipeline, name).__get__(obj))
        obj._SHOW_NUMBER = Pipeline._SHOW_NUMBER
        obj._SHOW_IMAGE_EXTS = Pipeline._SHOW_IMAGE_EXTS
        return obj

    def pushed(self, fn, *args):
        with mock.patch("src.utils.canvas_patch.push") as push:
            out = fn(*args)
        return out, [c.args[0] for c in push.call_args_list]


class WhatIsNamed(Files):
    def test_a_file_shown_before_keeps_its_number(self):
        a = self.file("a.png")
        cs.add_gallery_image("lead", self.file("first.png"))
        cs.add_gallery_image("lead", a, "the location")
        items, problems = self.me()._show_items([a])
        self.assertEqual((items, problems),
                         ([{"path": a, "number": 2, "caption": "the location"}], []))

    def test_it_can_be_named_by_that_number(self):
        a = self.file("a.png")
        cs.add_gallery_image("lead", a)
        for said in (1, "1", "#1", "image 1", "Image #1"):
            items, _ = self.me()._show_items([said])
            self.assertEqual([i["path"] for i in items], [a], said)

    def test_a_file_from_a_branch_has_no_number_yet(self):
        b = self.file("mara_v003.png")
        items, _ = self.me()._show_items([b])
        self.assertEqual(items, [{"path": b, "number": None, "caption": ""}])

    def test_what_cannot_be_shown_is_said(self):
        doc = self.file("notes.txt")
        items, problems = self.me()._show_items([7, os.path.join(self._dir.name, "gone.png"), doc])
        self.assertEqual(items, [])
        self.assertIn("no output with that number", problems[0])
        self.assertIn("no such file", problems[1])
        self.assertIn("not an image or a video", problems[2])

    def test_a_video_is_shown_too_and_a_file_named_twice_once(self):
        v = self.file("shot.mp4")
        items, _ = self.me()._show_items([v, v])
        self.assertEqual([i["path"] for i in items], [v])


class Showing(Files):
    def test_a_new_file_gets_the_conversations_next_number(self):
        cs.add_gallery_image("lead", self.file("first.png"))
        b = self.file("mara.png")
        obj = self.me()
        sent, pushed = self.pushed(obj._show_again, obj._show_items([b])[0], "approved characters")
        self.assertEqual(sent[0]["number"], 2)
        self.assertEqual([g["path"] for g in cs.get_gallery("lead")][-1], b)
        self.assertEqual(pushed, [{"op": "show_again", "items": sent,
                                   "caption": "approved characters"}])

    def test_showing_it_again_does_not_number_it_again(self):
        b = self.file("mara.png")
        obj = self.me()
        self.pushed(obj._show_again, obj._show_items([b])[0])
        sent, _ = self.pushed(obj._show_again, obj._show_items([b])[0])
        self.assertEqual(sent[0]["number"], 1)
        self.assertEqual(len(cs.get_gallery("lead")), 1)


class AReviewStop(Files):
    """halt_for_review shows what it asks about."""

    def test_what_a_branch_made_is_shown(self):
        b = self.file("mara.png")
        n, pushed = self.pushed(self.me()._show_unseen, [b], "review 135")
        self.assertEqual(n, 1)
        self.assertEqual(pushed[0]["caption"], "review 135")

    def test_what_this_conversation_already_showed_is_not_shown_twice(self):
        a = self.file("a.png")
        cs.add_gallery_image("lead", a)
        self.assertEqual(self.pushed(self.me()._show_unseen, [a]), (0, []))

    def test_nor_what_this_turn_made_and_the_host_is_about_to_show(self):
        a = self.file("a.png")
        self.assertEqual(self.pushed(self.me(own=[a])._show_unseen, [a]), (0, []))
        self.assertEqual(cs.get_gallery("lead"), [])

    def test_the_stop_asks_for_it(self):
        src = (ROOT / "src" / "pipeline.py").read_text(encoding="utf-8")
        tool = src.split("        async def halt_for_review(", 1)[1].split("        @_tool", 1)[0]
        self.assertIn('self._show_unseen(files, f"review {hid}")', tool)


class TheTool(unittest.TestCase):
    SRC = (ROOT / "src" / "pipeline.py").read_text(encoding="utf-8")

    def test_the_orchestrator_has_it(self):
        self.assertIn("list_outputs, show_in_panel, ", self.SRC)

    def test_it_places_and_generates_nothing(self):
        tool = self.SRC.split("        async def show_in_panel(", 1)[1].split("        @_tool", 1)[0]
        for forbidden in ("_canvas_lease_refusal", "_execute_workflow", "set_canvas"):
            self.assertNotIn(forbidden, tool)

    def test_the_agent_is_told_when_to_use_it(self):
        prompt = (ROOT / "config" / "system_prompts" / "system_prompt.orchestrator.md").read_text(encoding="utf-8")
        self.assertIn("`show_in_panel(files)` puts pictures in", prompt)
        from src.utils import hook_flow
        self.assertIn("show_in_panel(<its files>) before you speak of it",
                      Path(hook_flow.__file__).read_text(encoding="utf-8"))


class TheHost(unittest.TestCase):
    SRC = (ROOT / "src" / "utils" / "agentY_server.py").read_text(encoding="utf-8")

    def test_it_becomes_cards_and_never_a_canvas_patch(self):
        translate = self.SRC.split("        cp = event.get(\"canvas_patch\")", 1)[1].split("return\n", 2)
        self.assertIn('if cp.get("op") == "show_again":', translate[0])
        self.assertIn("_show_again(cp)", translate[0])

    def test_a_card_is_never_dropped_on_the_canvas_and_always_shown(self):
        body = self.SRC.split("    def _show_again(patch: dict) -> None:", 1)[1].split("    def _translate", 1)[0]
        self.assertIn('"drop": False', body)
        self.assertIn('"show": True', body)
        self.assertIn('"again": f"{req_id}:{again[\'n\']}"', body)
        self.assertIn('"index": it.get("number")', body)

    def test_showing_again_does_not_copy_the_file_again(self):
        from src.utils import agentY_server as srv
        with tempfile.TemporaryDirectory() as tmp:
            src_dir, in_dir = Path(tmp) / "out", Path(tmp) / "input"
            src_dir.mkdir(); in_dir.mkdir()
            f = src_dir / "mara.png"
            f.write_bytes(b"picture")
            with mock.patch.object(srv, "_comfy_input_dir", return_value=in_dir):
                first = srv._staged_copy(str(f))
                second = srv._staged_copy(str(f))
            self.assertEqual((first, second), ("mara.png", "mara.png"))
            self.assertEqual(sorted(p.name for p in in_dir.iterdir()), ["mara.png"])

    def test_a_different_file_of_the_same_name_is_still_copied(self):
        from src.utils import agentY_server as srv
        with tempfile.TemporaryDirectory() as tmp:
            src_dir, in_dir = Path(tmp) / "out", Path(tmp) / "input"
            src_dir.mkdir(); in_dir.mkdir()
            (in_dir / "mara.png").write_bytes(b"an older, longer picture")
            f = src_dir / "mara.png"
            f.write_bytes(b"picture")
            with mock.patch.object(srv, "_comfy_input_dir", return_value=in_dir):
                staged = srv._staged_copy(str(f))
            self.assertNotEqual(staged, "mara.png")
            self.assertEqual((in_dir / staged).read_bytes(), b"picture")


if __name__ == "__main__":
    unittest.main()
