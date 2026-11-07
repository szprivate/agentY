"""The plan the agent wrote, kept for the panel's plan overview.

Asking the agent for a plan left the overview saying there was nothing to show:
it listed a hook pipeline's stages only. The host now keeps the latest plan the
agent put to the user, from its reply or from run_planner.
"""
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from src.utils import conversation_store as cs
from src.utils import proposed_plan as pp

ROOT = Path(__file__).resolve().parent.parent

ASKED = """Here's the plan:

1. **Analyse** image 4 — describe the character
   - with the vision model
2. Write the synopsis
3) Generate 5 shots with `Flux`

Shall I go ahead?"""


class FromTheReply(unittest.TestCase):
    def test_a_list_after_words_about_a_plan(self):
        got = pp.from_reply(ASKED)
        self.assertEqual([s["text"] for s in got["steps"]],
                         ["Analyse image 4 — describe the character", "Write the synopsis",
                          "Generate 5 shots with Flux"])
        self.assertEqual(got["steps"][0]["detail"], "with the vision model")
        self.assertEqual(got["steps"][2]["detail"], "")      # not "Shall I go ahead?"
        self.assertEqual(got["title"], "")                     # "Here's the plan" adds nothing
        self.assertEqual(got["source"], "reply")

    def test_any_numbered_list_is_not_a_plan(self):
        self.assertIsNone(pp.from_reply("Found these:\n1. a.png\n2. b.png"))

    def test_but_it_is_when_the_turn_was_about_one(self):
        got = pp.from_reply("Found these:\n1. a.png\n2. b.png", wanted=True)
        self.assertEqual([s["text"] for s in got["steps"]], ["a.png", "b.png"])
        self.assertEqual(got["title"], "Found these")

    def test_heading_style_steps(self):
        got = pp.from_reply("## Short film\n\n### Step 1: Screenplay\nfive shots\n\n"
                            "### Step 2: Characters\nthree of them\n\nOK?", wanted=True)
        self.assertEqual([(s["text"], s["detail"]) for s in got["steps"]],
                         [("Screenplay", "five shots"), ("Characters", "three of them")])

    def test_the_numbering_has_to_run_in_order(self):
        got = pp.from_reply("The plan:\n1. a\n2. b\n4. c\n1. x", wanted=True)
        self.assertEqual([s["text"] for s in got["steps"]], ["a", "b"])

    def test_one_item_is_no_list_and_code_is_skipped(self):
        self.assertIsNone(pp.from_reply("The plan:\n1. only", wanted=True))
        self.assertIsNone(pp.from_reply("The plan:\n```\n1. a\n2. b\n```", wanted=True))


class FromThePlanner(unittest.TestCase):
    def test_its_steps(self):
        got = pp.from_planner('{"steps": [{"request": "Generate a portrait.", '
                              '"description": "Generate base portrait", "kind": "generation"}, '
                              '{"request": "Upscale it.", "description": "Upscale it."}]}')
        self.assertEqual(got["steps"], [{"text": "Generate base portrait", "detail": "Generate a portrait."},
                                        {"text": "Upscale it.", "detail": ""}])
        self.assertEqual(got["source"], "planner")

    def test_not_json(self):
        self.assertIsNone(pp.from_planner("sorry"))


class KeptWithTheConversation(unittest.TestCase):
    def setUp(self):
        self._dir = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.addCleanup(self._dir.cleanup)
        env = mock.patch.dict(os.environ, {
            "AGENTY_CONVERSATION_DB": os.path.join(self._dir.name, "c.sqlite")})
        env.start()
        self.addCleanup(env.stop)
        cs._INITIALISED = False
        self.addCleanup(setattr, cs, "_INITIALISED", False)
        cs.create_thread("t", thread_id="t")

    def test_the_user_asking_for_a_plan(self):
        pp.record_turn("t", "make a plan for a short film", "Sure:\n1. a\n2. b")
        self.assertEqual([s["text"] for s in pp.get("t")["steps"]], ["a", "b"])

    def test_a_turn_without_one_leaves_the_last_plan(self):
        pp.record_turn("t", "make a plan", "1. a\n2. b")
        pp.record_turn("t", "what is in image 3?", "A cat.")
        pp.record_turn("t", "list the files", "1. x.png\n2. y.png")
        self.assertEqual([s["text"] for s in pp.get("t")["steps"]], ["a", "b"])

    def test_the_planner_when_the_reply_did_not_spell_it_out(self):
        pp.record_turn("t", "do it all", "Done, five pictures are on the canvas.",
                       planner_raw='{"steps": [{"request": "r1", "description": "one"}, '
                                   '{"request": "r2", "description": "two"}]}')
        self.assertEqual(pp.get("t")["source"], "planner")

    def test_a_plan_up_for_approval(self):
        pp.record_turn("t", "make five shots, ask me first", "1. a\n2. b", awaiting_approval=True)
        self.assertEqual(len(pp.get("t")["steps"]), 2)

    def test_a_newer_plan_replaces_it_and_clear_forgets_it(self):
        pp.record_turn("t", "plan", "1. a\n2. b")
        pp.record_turn("t", "new plan", "1. c\n2. d\n3. e")
        self.assertEqual(len(pp.get("t")["steps"]), 3)
        pp.clear("t")
        self.assertIsNone(pp.get("t"))


class Wiring(unittest.TestCase):
    def test_the_host_records_it_at_the_end_of_the_turn_and_serves_it(self):
        src = (ROOT / "src" / "utils" / "agentY_server.py").read_text(encoding="utf-8")
        self.assertIn("_pp.record_turn(thread_id,", src)
        self.assertLess(src.index('cs.add_message(thread_id, "assistant", text)'),
                        src.index("_pp.record_turn(thread_id,"))
        self.assertIn('"proposal": proposal', src)
        self.assertIn('@app.route("/agentY/threads/<tid>/plan/proposal", methods=["DELETE"])', src)

    def test_run_planner_leaves_its_steps_for_the_host(self):
        src = (ROOT / "src" / "pipeline.py").read_text(encoding="utf-8")
        body = src.split("async def run_planner(", 1)[1].split("@_tool", 1)[0]
        self.assertIn("self._planner_raw = ", body)
        self.assertIn('self._planner_raw = ""', src)


if __name__ == "__main__":
    unittest.main()
