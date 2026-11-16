"""The two agent faults of the promo run of 2026-11-15, read from its logs.

1. Circling: the agent judged a cached preview of its input frame after every
   change of denoise (run_workflow_now listed that preview first).
2. Stale prompt: an old brief sent by mistake and stopped took one of the turns
   compaction keeps word for word, and the night's prompts went into a summary.
"""

import json
import os
import tempfile
import time
import unittest
from pathlib import Path

from src.utils import compaction as C
from src.utils import run_outputs as RO


def user(text):
    return {"role": "user", "content": [{"text": text}]}


def assistant_text(text):
    return {"role": "assistant", "content": [{"text": text}]}


def call(name, tid, **inp):
    return {"role": "assistant", "content": [{"toolUse": {"name": name, "toolUseId": tid, "input": inp}}]}


def result(tid, text="ok"):
    return {"role": "user", "content": [{"toolResult": {"toolUseId": tid, "content": [{"text": text}]}}]}


class RunOutputs(unittest.TestCase):

    def test_saved_results_come_first_and_previews_and_cached_files_are_named(self):
        with tempfile.TemporaryDirectory() as d:
            out = Path(d, "output", "agent"); out.mkdir(parents=True)
            tmp = Path(d, "temp"); tmp.mkdir()
            preview = tmp / "ComfyUI_temp_hlkla_00001_.png"
            preview.write_bytes(b"x")
            old = time.time() - 600
            os.utime(preview, (old, old))           # a cached node's old file
            saved = out / "images_00004_.png"
            started = time.time()
            saved.write_bytes(b"y")
            rows = RO.describe([str(preview), str(saved)], started)
        self.assertEqual([r["kind"] for r in rows], ["saved", "preview"])
        self.assertEqual(rows[0]["file"], "images_00004_.png")
        self.assertEqual(rows[0]["subfolder"], "agent")
        self.assertTrue(rows[0]["fresh"])
        self.assertFalse(rows[1]["fresh"])
        note = RO.note(rows)
        self.assertIn("images_00004_.png", note)
        self.assertIn("not results", note)
        self.assertIn("Not rewritten by this run", note)

    def test_a_plain_fresh_result_needs_no_note(self):
        with tempfile.TemporaryDirectory() as d:
            started = time.time()
            p = Path(d, "output", "a.png"); p.parent.mkdir(); p.write_bytes(b"z")
            self.assertEqual(RO.note(RO.describe([str(p)], started)), "")


class AnalyzeTheSameFileTwice(unittest.TestCase):

    def test_a_second_look_at_an_unchanged_file_says_so(self):
        from src.tools import image_handling as ih
        with tempfile.TemporaryDirectory() as d:
            p = Path(d, "ComfyUI_temp_1.png"); p.write_bytes(b"x")
            self.assertEqual(ih._unchanged_note(str(p)), "")
            self.assertIn("has not changed", ih._unchanged_note(str(p)))
            p.write_bytes(b"xy")
            self.assertEqual(ih._unchanged_note(str(p)), "")


class AStoppedTurnDoesNotCount(unittest.TestCase):

    def history(self, wf):
        return [
            user("take 1: render the hills"),
            call("run_workflow_now", "a", workflow_path=wf), result("a"),
            assistant_text("done, take 1"),
            user("take 2: the hills again with these prompts"),
            call("run_workflow_now", "b", workflow_path=wf), result("b"),
            assistant_text("done, take 2"),
            user("OLD CANYON BRIEF: races forwards at about 50 km/h ... sandstone canyon"),
            user("Ignore the message before. Same prompt as your last render, word for word."),
        ]

    def test_the_stopped_brief_is_left_out_and_take_two_is_kept(self):
        msgs = self.history("/nowhere.json")
        self.assertEqual(len(C.turn_starts(msgs)), 4)
        counted = C.counted_turn_starts(msgs)
        self.assertEqual([msgs[i]["content"][0]["text"][:6] for i in counted],
                         ["take 1", "take 2", "Ignore"])
        new, removed = C.summarise_older(msgs, keep_turns=2, summarise=lambda m: "summary")
        texts = json.dumps(new)
        self.assertIn("take 2: the hills again", texts)
        self.assertEqual(len(removed), 4)    # only take 1 went into the summary

    def test_the_summary_keeps_the_last_render_prompts_word_for_word(self):
        with tempfile.TemporaryDirectory() as d:
            wf = Path(d, "ltx2_5_layout_to_render.json")
            hills = "A Spectral Blue GLC races fast at 100 km/h through eroded sandstone hills"
            wf.write_text(json.dumps({"5014:2483": {"class_type": "CLIPTextEncode",
                                                    "inputs": {"text": hills, "clip": ["1", 0]}},
                                      "7": {"class_type": "KSampler", "inputs": {"seed": 1}}}))
            msgs = [
                user("take 2"), call("run_workflow_now", "b", workflow_path=str(wf)),
                result("b"), assistant_text("done"),
                user("take 3 prep"), assistant_text("ok"),
                user("render again, same prompt as last time"),
            ]
            new, removed = C.summarise_older(msgs, keep_turns=2, summarise=lambda m: "summary")
        head = new[0]["content"][0]["text"]
        self.assertIn("LAST RENDER", head)
        self.assertIn(hills, head)
        self.assertIn("node 5014:2483 text", head)

    def test_no_render_block_when_a_kept_turn_has_a_newer_run(self):
        msgs = [
            user("one"), call("run_workflow_now", "a", workflow_path="/old.json"), result("a"),
            assistant_text("done"),
            user("two"), call("run_workflow_now", "b", workflow_path="/new.json"), result("b"),
            assistant_text("done"),
            user("three"),
        ]
        new, _ = C.summarise_older(msgs, keep_turns=2, summarise=lambda m: "s")
        self.assertNotIn("LAST RENDER", new[0]["content"][0]["text"])


if __name__ == "__main__":
    unittest.main()
