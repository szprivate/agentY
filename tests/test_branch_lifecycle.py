"""A branch ends: closed when its work is approved, replaced when the approach changes.

A branch conversation used to live for as long as its lead did. One whose work
was long approved could be messaged again and would start working on settled
stages; and when the user changed the approach, the old branch was handed the
new plan on top of a conversation full of the old one.
"""
import os
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

from src.utils import conversation_store as cs
from src.utils import run_ledger as rl
from src.utils import shots

ROOT = Path(__file__).resolve().parent.parent


class Run(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.addCleanup(tmp.cleanup)
        env = mock.patch.dict(os.environ, {
            "AGENTY_CONVERSATION_DB": os.path.join(tmp.name, "c.sqlite")})
        env.start()
        self.addCleanup(env.stop)
        cs._INITIALISED = False
        self.addCleanup(setattr, cs, "_INITIALISED", False)
        shots._reset_for_tests()
        self.addCleanup(shots._reset_for_tests)
        self.running, self.stopped, self.started = set(), [], []

        def start(tid, text, origin="", dry_run=False):
            self.started.append(tid)
            return "rid"
        shots.configure(start_turn=start, is_running=lambda t: t in self.running,
                        stop_thread=lambda t: self.stopped.append(t) or True,
                        interject=lambda t, m: False, became_lead=lambda t: None)
        cs.create_thread("lead", thread_id="lead")
        self.chars = shots.start_shot("lead", "characters", "make them", hook_ids=["8", "135"])["thread_id"]
        self.places = shots.start_shot("lead", "places", "make them", hook_ids=["9", "118", "136"])["thread_id"]
        for tid in (self.chars, self.places):
            cs.set_shot_status(tid, "done")

    def state(self, tid):
        return cs.shot_of(tid)["status"]


class Closing(Run):
    def test_a_closed_branch_takes_no_more_work(self):
        self.assertEqual(shots.close_shot("lead", "characters")["status"], "closed")
        said = shots.message_shot("lead", "characters", "one more round")
        self.assertFalse(said["ok"])
        self.assertIn("is closed and takes no more work", said["error"])
        self.assertEqual(self.started.count(self.chars), 1)        # only its first turn

    def test_its_running_turn_is_stopped_so_it_holds_no_agent(self):
        self.running.add(self.chars)
        shots.close_shot("lead", "characters")
        self.assertEqual(self.stopped, [self.chars])

    def test_its_conversation_stays_and_says_what_happened(self):
        shots.close_shot("lead", "characters")
        last = cs.get_thread(self.chars)["messages"][-1]["content"]
        self.assertIn("This branch is closed — all of its work is approved", last)

    def test_a_turn_still_winding_down_does_not_reopen_it(self):
        shots.close_shot("lead", "characters")
        turn = types.SimpleNamespace(request_id="r1", thread_id=self.chars, origin="lead")
        shots._on_event({"type": "turn_start"}, turn)
        shots._on_event({"type": "done"}, turn)
        self.assertEqual(self.state(self.chars), "closed")

    def test_closing_twice_is_not_an_error(self):
        shots.close_shot("lead", "characters")
        self.assertEqual(shots.close_shot("lead", "characters")["note"], "it had already ended")

    def test_the_other_branch_is_untouched(self):
        shots.close_shot("lead", "characters")
        self.assertTrue(shots.message_shot("lead", "places", "warmer light")["ok"])


class Replacing(Run):
    def test_the_old_branch_is_marked_and_the_name_is_free(self):
        shots.close_shot("lead", "places", replaced=True, why="one master shot instead of per-shot plates")
        self.assertEqual(self.state(self.places), "replaced")
        fresh = shots.start_shot("lead", "places", "the new approach", hook_ids=["201", "202"])
        self.assertTrue(fresh["ok"])
        self.assertNotEqual(fresh["thread_id"], self.places)

    def test_messages_go_to_the_fresh_one(self):
        shots.close_shot("lead", "places", replaced=True)
        fresh = shots.start_shot("lead", "places", "the new approach", hook_ids=["201"])["thread_id"]
        cs.set_shot_status(fresh, "done")
        shots.message_shot("lead", "places", "go on")
        self.assertEqual(self.started[-1], fresh)

    def test_ended_branches_do_not_count_against_the_limit(self):
        shots.close_shot("lead", "places", replaced=True)
        with mock.patch.object(shots, "MAX_SHOTS_PER_LEAD", 2):
            self.assertTrue(shots.start_shot("lead", "places", "again", hook_ids=["201"])["ok"])

    def test_the_old_conversation_says_it_was_replaced(self):
        shots.close_shot("lead", "places", replaced=True, why="one master shot instead")
        last = cs.get_thread(self.places)["messages"][-1]["content"]
        self.assertIn("This branch was replaced: one master shot instead", last)


class ClosedByItself(Run):
    """The lead's turn closes a branch once everything it had to do is approved."""

    def test_all_stages_approved(self):
        self.assertEqual(shots.close_approved("lead", ["8", "135"], ["8", "135", "118", "136"]),
                         ["characters"])
        self.assertEqual(self.state(self.places), "done")

    def test_not_while_one_of_its_stages_is_open(self):
        self.assertEqual(shots.close_approved("lead", ["8"], ["8", "135", "118", "136"]), [])

    def test_a_loop_node_is_nothing_it_owes(self):
        # 9 is the loop start around the places stage: the ledger does not count it.
        self.assertEqual(shots.close_approved("lead", ["118", "136"], ["8", "135", "118", "136"]),
                         ["places"])

    def test_not_while_it_is_working_or_has_not_reported(self):
        self.running.add(self.chars)
        self.assertEqual(shots.close_approved("lead", ["8", "135"], ["8", "135"]), [])
        self.running.clear()
        cs.set_shot_status(self.chars, "stopped")
        self.assertEqual(shots.close_approved("lead", ["8", "135"], ["8", "135"]), [])

    def test_a_shot_with_no_stages_of_a_canvas_is_never_closed_this_way(self):
        plain = shots.start_shot("lead", "sh010", "build it")["thread_id"]
        cs.set_shot_status(plain, "done")
        self.assertEqual(shots.close_approved("lead", ["999"], ["999"]), [])

    def test_what_the_ledger_counts(self):
        rl.sync("lead", [{"hook_node_id": "9", "purpose": "loop_start"},
                         {"hook_node_id": "118", "purpose": "set_parameter", "title": "place"},
                         {"hook_node_id": "136", "purpose": "human_review"},
                         {"hook_node_id": "140", "purpose": "qa"}], {})
        rl.produced("lead", "118", outputs=["a.png"])
        rl.approved("lead", "136", ["118"], ["a.png"])
        approved, known = rl.approved_ids("lead")
        self.assertEqual((sorted(approved), sorted(known)), (["118", "136"], ["118", "136"]))

    def test_the_leads_turn_does_it(self):
        src = (ROOT / "src" / "pipeline.py").read_text(encoding="utf-8")
        start = src.split("    def _ledger_turn_start(self)", 1)[1].split("\n    def ", 1)[0]
        self.assertTrue(start.rstrip().endswith("self._close_finished_branches()"))
        body = src.split("    def _close_finished_branches(self)", 1)[1].split("\n    def ", 1)[0]
        self.assertIn("if self._branch_info():\n            return []", body)
        self.assertIn("_shots.close_approved(lead, approved, known)", body)


class WhatAFreshBranchIsTold(Run):
    def setUp(self):
        super().setUp()
        hook = lambda i, p, t: {"hook_node_id": i, "purpose": p, "title": t}  # noqa: E731
        rl.sync("lead", [hook("4", "text_only", "screenplay"), hook("131", "human_review", "read it"),
                         hook("8", "set_parameter", "characters"), hook("135", "human_review", "pick"),
                         hook("201", "set_parameter", "master shot")], {})
        rl.produced("lead", "4", text_path="s_v002.md")
        rl.approved("lead", "131", ["4"], [])
        rl.produced("lead", "8", outputs=["mara.png"])
        rl.approved("lead", "135", ["8"], ["mara.png"])

    def test_its_own_stages_then_what_is_settled_elsewhere(self):
        text = rl.index("lead", only=["201"])
        self.assertIn('- 201 "master shot" [set / sweep parameter]: not done yet', text)
        settled = text.split("SETTLED ELSEWHERE IN THE RUN", 1)[1]
        self.assertIn('- 4 "screenplay"', settled)
        self.assertIn("text → s_v002.md", settled)
        self.assertIn("goes on (1): mara.png", settled)

    def test_the_reviews_themselves_are_not_repeated(self):
        settled = rl.index("lead", only=["201"]).split("SETTLED ELSEWHERE IN THE RUN", 1)[1]
        self.assertNotIn('"pick"', settled)

    def test_a_branch_with_nothing_done_yet_still_gets_what_is_settled(self):
        self.assertIn("SETTLED ELSEWHERE", rl.index("lead", only=["201"]))

    def test_the_lead_gets_one_list(self):
        self.assertNotIn("SETTLED ELSEWHERE", rl.index("lead"))


class WhatTheLeadIsTold(unittest.TestCase):
    def test_how_a_branch_ends(self):
        src = (ROOT / "src" / "utils" / "hook_flow.py").read_text(encoding="utf-8")
        self.assertIn("A BRANCH ENDS.", src)
        self.assertIn("close_shot(name, \"\n        \"replaced=True), then start_shot a fresh branch", src)

    def test_the_lead_has_the_tool_and_a_branch_does_not_start_shots(self):
        tools = (ROOT / "src" / "tools" / "__init__.py").read_text(encoding="utf-8")
        self.assertEqual(tools.count("    close_shot,\n"), 2)
        agent = (ROOT / "src" / "agent.py").read_text(encoding="utf-8")
        self.assertIn("close_shot as _cls", agent)
        self.assertIn("id(_ssn), id(_cls)}", agent)

    def test_the_plan_matches_a_branch_to_the_conversation_at_work(self):
        src = (ROOT / "src" / "utils" / "run_ledger.py").read_text(encoding="utf-8")
        self.assertIn('key=lambda s: s.get("status") in ("closed", "replaced")', src)


if __name__ == "__main__":
    unittest.main()
