"""A lead conversation starting shot conversations, and hearing back from them.

src/utils/shots.py with the host's hooks faked: what a shot is given, that one
level is the limit, that a finished shot wakes its lead (once, with every report
that came in, after the lead's own turn), Stop in the lead, and the turn bus
keeping a turn's events so the panel can follow a turn it did not start.
"""

import os
import queue
import tempfile
import threading
import time
import unittest
from unittest import mock

from src.utils import conversation_store as cs
from src.utils import shots, turn_bus


class _Store(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.addCleanup(tmp.cleanup)
        env = mock.patch.dict(os.environ, {
            "AGENTY_CONVERSATION_DB": os.path.join(tmp.name, "conversations.sqlite")})
        env.start()
        self.addCleanup(env.stop)
        cs._INITIALISED = False
        self.addCleanup(setattr, cs, "_INITIALISED", False)


class _Host(_Store):
    """shots.configure with a fake host that records the turns it is asked for."""

    def setUp(self):
        super().setUp()
        shots._reset_for_tests()
        self.addCleanup(shots._reset_for_tests)
        self.started: list = []
        self.running: set = set()
        self.stopped: list = []
        self.interjected: list = []
        saved = dict(shots._hooks)
        self.addCleanup(shots._hooks.update, saved)
        shots.configure(
            start_turn=self._start, is_running=lambda t: t in self.running,
            stop_thread=self._stop, interject=self._interject)
        # Off again afterwards: later tests' turns must not reach this store.
        self.addCleanup(lambda: (turn_bus.unobserve(shots._on_event),
                                 shots._observing.update(on=False)))
        for name, value in (("WAKE_WAIT", 5.0),):
            p = mock.patch.object(shots, name, value)
            p.start()
            self.addCleanup(p.stop)
        self.lead = cs.create_thread(title="Pier sequence")

    def _start(self, thread_id, text, *, origin, dry_run=False):
        self.started.append({"thread_id": thread_id, "text": text, "origin": origin,
                             "dry_run": dry_run})
        return f"r{len(self.started)}"

    def _stop(self, thread_id):
        self.stopped.append(thread_id)
        return True

    def _interject(self, thread_id, text):
        self.interjected.append((thread_id, text))
        return True

    def finish(self, thread_id, answer="", *, error=False, rid=None):
        """Play a turn of *thread_id* through the turn bus, ending it."""
        if answer:
            cs.add_message(thread_id, "assistant", answer)
        q = turn_bus.tee(queue.Queue(), request_id=rid or f"t-{time.monotonic_ns()}",
                         thread_id=thread_id, origin="lead")
        if error:
            q.put({"type": "error", "message": "boom"})
        q.put({"type": "done"})
        q.put(None)

    def wait_for(self, pred, timeout=5.0):
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            if pred():
                return True
            time.sleep(0.05)
        return False


class Store(_Store):

    def test_shots_are_listed_under_their_lead(self):
        lead = cs.create_thread(title="lead")
        a = cs.create_thread(title="sh010")
        cs.set_shot(a, lead, "sh010")
        listed = {t["id"]: t for t in cs.list_threads()}
        self.assertEqual(listed[a]["lead_id"], lead)
        self.assertEqual(listed[a]["shot"], "sh010")
        self.assertIsNone(listed[lead]["lead_id"])
        self.assertEqual([s["name"] for s in cs.shots_of(lead)], ["sh010"])
        cs.set_sequence_notes(lead, "35mm, dusk")
        self.assertEqual(cs.get_sequence_notes(lead), "35mm, dusk")


class StartingShots(_Host):

    def test_a_shot_is_a_conversation_briefed_by_its_lead(self):
        shots.set_notes(self.lead, "Mara: /refs/mara.png. Teal-orange, 24 fps.")
        out = shots.start_shot(self.lead, "sh010", "Mara walks onto the pier.")
        self.assertTrue(out["ok"], out)
        tid = out["thread_id"]
        self.assertEqual(cs.shot_of(tid)["lead_id"], self.lead)
        first = cs.get_thread(tid)["messages"][0]["content"]
        self.assertIn("[SHOT sh010]", first)
        self.assertIn("Mara walks onto the pier.", first)
        self.assertIn("Teal-orange", first)             # the sequence notes
        self.assertEqual(self.started[0]["thread_id"], tid)
        self.assertEqual(self.started[0]["origin"], "lead")

    def test_dry_run_follows_the_setting(self):
        with mock.patch.object(shots, "dry_run_default", return_value=True):
            shots.start_shot(self.lead, "sh010", "brief")
        self.assertTrue(self.started[0]["dry_run"])
        self.assertIn("Dry run", self.started[0]["text"])
        with mock.patch.object(shots, "dry_run_default", return_value=False):
            shots.start_shot(self.lead, "sh020", "brief")
        self.assertFalse(self.started[1]["dry_run"])
        self.assertNotIn("Dry run", self.started[1]["text"])

    def test_the_setting_is_off_by_default(self):
        with mock.patch("src.utils.settings.load_settings", return_value={}):
            self.assertFalse(shots.dry_run_default())

    def test_a_shot_cannot_start_shots(self):
        tid = shots.start_shot(self.lead, "sh010", "brief")["thread_id"]
        out = shots.start_shot(tid, "sh010a", "brief")
        self.assertFalse(out["ok"])
        self.assertIn("itself a shot", out["error"])
        self.assertEqual(len(self.started), 1)

    def test_a_name_is_used_once(self):
        shots.start_shot(self.lead, "sh010", "brief")
        out = shots.start_shot(self.lead, "SH010", "again")
        self.assertFalse(out["ok"])
        self.assertIn("message_shot", out["error"])

    def test_more_work_reaches_a_running_shot_or_starts_a_turn(self):
        tid = shots.start_shot(self.lead, "sh010", "brief")["thread_id"]
        self.running.add(tid)
        out = shots.message_shot(self.lead, "sh010", "warmer light")
        self.assertEqual(out["delivered"], "into its running turn")
        self.assertIn("warmer light", self.interjected[0][1])
        self.running.discard(tid)
        out = shots.message_shot(self.lead, "sh010", "now 5 s long")
        self.assertEqual(out["delivered"], "as a new turn")
        self.assertIn("[FROM THE LEAD] now 5 s long", self.started[-1]["text"])


class HearingBack(_Host):

    def test_a_finished_shot_wakes_its_lead_with_the_report(self):
        tid = shots.start_shot(self.lead, "sh010", "brief")["thread_id"]
        self.finish(tid, "Built sh010_v001.json; Mara reads well.")
        self.assertTrue(self.wait_for(lambda: len(self.started) == 2))
        wake = self.started[1]
        self.assertEqual(wake["thread_id"], self.lead)
        self.assertEqual(wake["origin"], "shots")
        self.assertIn("Shot sh010 finished", wake["text"])
        self.assertIn("Mara reads well", wake["text"])
        self.assertEqual(cs.shot_of(tid)["status"], "done")
        # …and the lead's conversation shows what woke it.
        self.assertIn("[SHOTS REPORTING BACK]",
                      cs.get_thread(self.lead)["messages"][-1]["content"])

    def test_reports_wait_for_a_busy_lead_and_come_together(self):
        a = shots.start_shot(self.lead, "sh010", "brief")["thread_id"]
        b = shots.start_shot(self.lead, "sh020", "brief")["thread_id"]
        self.running.add(self.lead)
        self.finish(a, "A done")
        self.finish(b, "B failed", error=True)
        time.sleep(1.0)
        self.assertEqual(len(self.started), 2)            # nothing while the lead works
        self.running.discard(self.lead)
        self.finish(self.lead, "lead's own answer")       # the lead's turn ends
        self.assertTrue(self.wait_for(lambda: len(self.started) == 3))
        time.sleep(0.8)
        self.assertEqual(len(self.started), 3)            # one wake, not two
        text = self.started[2]["text"]
        self.assertIn("Shot sh010 finished", text)
        self.assertIn("Shot sh020 FAILED", text)
        self.assertEqual(cs.shot_of(b)["status"], "failed")

    def test_the_lead_stops_reviewing_after_many_rounds_on_its_own(self):
        tid = shots.start_shot(self.lead, "sh010", "brief")["thread_id"]
        with mock.patch.object(shots, "MAX_WAKES_WITHOUT_USER", 1):
            self.finish(tid, "one")
            self.assertTrue(self.wait_for(lambda: len(self.started) == 2))
            self.finish(tid, "two")
            self.assertTrue(self.wait_for(lambda: "⏸" in cs.get_thread(self.lead)["messages"][-1]["content"]))
            self.assertEqual(len(self.started), 2)
            # The user writing to the lead resets it.
            turn_bus.tee(queue.Queue(), request_id="u1", thread_id=self.lead, origin="panel")
            self.finish(tid, "three")
            self.assertTrue(self.wait_for(lambda: len(self.started) == 3))

    def test_stop_in_the_lead_stops_its_running_shots(self):
        a = shots.start_shot(self.lead, "sh010", "brief")["thread_id"]
        b = shots.start_shot(self.lead, "sh020", "brief")["thread_id"]
        self.running.add(a)
        self.assertEqual(shots.stop_all(self.lead), ["sh010"])
        self.assertEqual(self.stopped, [a])
        self.assertNotIn(b, self.stopped)

    def test_status_lists_every_shot(self):
        a = shots.start_shot(self.lead, "sh010", "brief")["thread_id"]
        shots.start_shot(self.lead, "sh020", "brief")
        self.running.add(a)
        rows = {r["shot"]: r["status"] for r in shots.status(self.lead)}
        self.assertEqual(rows, {"sh010": "running", "sh020": "queued"})


class AToolBudget(_Host):

    def play(self, tid, rid, n, agent="orchestrator"):
        q = turn_bus.tee(queue.Queue(), request_id=rid, thread_id=tid, origin="lead")
        for i in range(n):
            q.put({"type": "tool", "phase": "call", "agent": agent, "name": f"[{agent}] run_script"})
            q.put({"type": "tool", "phase": "result", "agent": agent})
        return q

    def test_a_shot_is_told_to_report_then_stopped(self):
        tid = shots.start_shot(self.lead, "sh010", "brief")["thread_id"]
        with mock.patch.object(shots, "max_tool_calls", return_value=8):
            q = self.play(tid, "b1", 8)
            self.assertEqual(len(self.interjected), 1)
            self.assertIn("[TOOL BUDGET]", self.interjected[0][1])
            self.assertEqual(self.stopped, [])
            for _ in range(2):
                q.put({"type": "tool", "phase": "call", "agent": "orchestrator"})
            self.assertTrue(self.wait_for(lambda: self.stopped == [tid]))
            q.put({"type": "system", "data": "⏹ Stopped."})
            q.put({"type": "done"})
            q.put(None)
        self.assertTrue(self.wait_for(lambda: len(self.started) == 2))
        self.assertIn("stopped after 10 tool calls", self.started[1]["text"])
        self.assertEqual(cs.shot_of(tid)["status"], "stopped")

    def test_the_specialists_steps_do_not_count(self):
        tid = shots.start_shot(self.lead, "sh010", "brief")["thread_id"]
        with mock.patch.object(shots, "max_tool_calls", return_value=5):
            self.play(tid, "b2", 40, agent="assemble_workflow")
        self.assertEqual(self.interjected, [])
        self.assertEqual(self.stopped, [])

    def test_no_budget_for_an_ordinary_conversation(self):
        other = cs.create_thread()
        with mock.patch.object(shots, "max_tool_calls", return_value=3):
            self.play(other, "b3", 20)
        self.assertEqual(self.interjected, [])

    def test_a_shot_left_running_by_a_restart_reads_stopped(self):
        tid = shots.start_shot(self.lead, "sh010", "brief")["thread_id"]
        cs.set_shot_status(tid, "running")
        self.assertEqual(shots.status(self.lead)[0]["status"], "stopped")
        self.running.add(tid)
        self.assertEqual(shots.status(self.lead)[0]["status"], "running")


class FollowingATurn(unittest.TestCase):

    def test_a_late_watcher_gets_the_turn_from_its_start_then_live(self):
        q = turn_bus.tee(queue.Queue(), request_id="w1", thread_id="t", origin="lead")
        q.put({"type": "text", "data": "one "})
        got = []
        th = threading.Thread(target=lambda: got.extend(
            e for e in turn_bus.follow("w1", tick=0.1) if e is not None))
        th.start()
        time.sleep(0.2)
        q.put({"type": "text", "data": "two"})
        q.put({"type": "done"})
        q.put(None)
        th.join(timeout=3)
        self.assertEqual([e["type"] for e in got], ["text", "text", "done"])
        self.assertEqual(got[1]["data"], "two")

    def test_an_unknown_turn_yields_nothing(self):
        self.assertEqual(list(turn_bus.follow("nope")), [])

    def test_a_very_long_turn_keeps_its_tail(self):
        with mock.patch.object(turn_bus, "LOG_LIMIT", 3):
            q = turn_bus.tee(queue.Queue(), request_id="w2", thread_id="t")
            for i in range(5):
                q.put({"type": "text", "data": str(i)})
            q.put({"type": "done"})
            q.put(None)
            got = list(turn_bus.follow("w2"))
        self.assertIn("not shown", got[0]["data"])
        self.assertEqual([e.get("data") for e in got[1:]], ["3", "4", None])


class ServerRoutes(_Host):

    def setUp(self):
        super().setUp()
        from src.utils import agentY_server as srv
        self.srv = srv
        p = mock.patch.object(srv._api_guard, "check", return_value=None, create=True)
        p.start()
        self.addCleanup(p.stop)
        client = srv._build_app().test_client()
        try:
            from src.utils import api_guard
            tok = api_guard.session_token(None)
        except Exception:  # noqa: BLE001
            tok = ""
        headers = {"X-AgentY-Token": tok} if tok else {}
        self.get = lambda url: client.get(url, headers=headers)

    def test_watching_a_turn_streams_it_without_touching_the_canvas(self):
        q = turn_bus.tee(queue.Queue(), request_id="w3", thread_id="shot-1", origin="lead")
        q.put({"type": "text", "data": "hi"})
        q.put({"type": "canvas_patch", "node": 1})
        q.put({"type": "output", "path": "/x.png", "drop": True})
        q.put({"type": "done"})
        q.put(None)
        body = self.get("/agentY/runs/w3/stream").get_data(as_text=True)
        self.assertIn('"thread", "id": "shot-1"', body)
        self.assertIn('"hi"', body)
        self.assertNotIn("canvas_patch", body)
        self.assertIn('"drop": false', body)
        self.assertEqual(body.count('"type": "done"'), 1)

    def test_a_turn_that_just_ended_is_still_listed(self):
        q = turn_bus.tee(queue.Queue(), request_id="w4", thread_id="lead-x", origin="shots")
        q.put({"type": "done"})
        q.put(None)
        body = self.get("/agentY/runs").get_json()
        self.assertNotIn("w4", [r["request_id"] for r in body["runs"]])
        recent = {r["request_id"]: r for r in body["recent"]}
        self.assertEqual(recent["w4"]["thread_id"], "lead-x")
        self.assertLessEqual(recent["w4"]["ended"], body["now"])

    def test_a_running_turn_is_not_waiting_on_an_answer_until_it_asks(self):
        with mock.patch.dict(self.srv._run_registry, {"w5": {"thread_id": "t5"}}):
            q = turn_bus.tee(queue.Queue(), request_id="w5", thread_id="t5")
            q.put({"type": "text", "data": "working"})
            run = lambda: next(r for r in self.get("/agentY/runs").get_json()["runs"]
                               if r["request_id"] == "w5")
            self.assertFalse(run()["awaiting_reply"])
            q.put({"type": "ask", "request_id": "w5", "prompt": "retry?"})
            self.assertTrue(run()["awaiting_reply"])
            q.put({"type": "tool", "phase": "call"})
            self.assertFalse(run()["awaiting_reply"])
            q.put({"type": "done"})
            q.put(None)

    def test_stopping_a_leads_shots_from_the_strip(self):
        a = shots.start_shot(self.lead, "sh010", "brief")["thread_id"]
        self.running.add(a)
        client = self.srv._build_app().test_client()
        try:
            from src.utils import api_guard
            tok = api_guard.session_token(None)
        except Exception:  # noqa: BLE001
            tok = ""
        for tid in (self.lead, a):                       # from the lead, or from a shot
            body = client.post(f"/agentY/threads/{tid}/shots/stop", json={},
                               headers={"X-AgentY-Token": tok} if tok else {}).get_json()
            self.assertEqual(body["stopped"], ["sh010"])

    def test_the_shot_strip_route(self):
        a = shots.start_shot(self.lead, "sh010", "brief")["thread_id"]
        for tid in (self.lead, a):                       # from the lead, or from a shot
            data = self.get(f"/agentY/threads/{tid}/shots").get_json()
            self.assertEqual(data["lead_id"], self.lead)
            self.assertEqual([s["shot"] for s in data["shots"]], ["sh010"])
        self.assertTrue(self.get(f"/agentY/threads/{a}/shots").get_json()["is_shot"])

    def test_a_reopened_conversation_gets_what_was_said_after_its_panel(self):
        cs.add_message(self.lead, "user", "before")
        cs.save_panel(self.lead, "<div>saved</div>")
        time.sleep(0.02)
        cs.add_message(self.lead, "user", "[SHOTS REPORTING BACK] …")
        data = self.get(f"/agentY/threads/{self.lead}").get_json()
        self.assertEqual([m["content"] for m in data["messages_after_panel"]],
                         ["[SHOTS REPORTING BACK] …"])


class Branches(_Host):
    """A branch of a hook pipeline: a shot that is handed stages of the canvas."""

    HOOKS = [{"hook_node_id": "4"}, {"hook_node_id": "18"}, {"hook_node_id": "2"},
             {"hook_node_id": "8"}, {"hook_node_id": "9"}, {"hook_node_id": "27"}, "junk"]

    def branch(self, name="characters", ids=("18", "8", "9", "2")):
        out = shots.start_shot(self.lead, name, "Make the character sheets.", hook_ids=list(ids))
        self.assertTrue(out["ok"], out)
        return out

    def test_it_is_handed_its_stages_and_told_how_a_branch_works(self):
        out = self.branch()
        self.assertEqual(out["branch_stages"], ["18", "8", "9", "2"])
        text = self.started[-1]["text"]
        self.assertIn("[BRANCH characters]", text)
        self.assertIn("[BRANCH STAGES — hook ids: 18, 8, 9, 2]", text)
        self.assertIn("first line is `REVIEW <its id>`", text)
        self.assertIn("release_review(<id>, user_said=", text)
        self.assertNotIn("prepare_workflow", text)          # that is a plain shot's way of working

    def test_a_plain_shot_is_still_a_plain_shot(self):
        out = shots.start_shot(self.lead, "sh010", "A pier at dawn.")
        self.assertNotIn("branch_stages", out)
        self.assertIn("[SHOT sh010]", self.started[-1]["text"])
        self.assertEqual(shots.scope_of(out["thread_id"]), [])

    def test_its_stages_are_remembered_and_survive_a_restart(self):
        tid = self.branch()["thread_id"]
        self.assertEqual(shots.scope_of(tid), ["18", "8", "9", "2"])
        with shots._LOCK:
            shots._scopes.clear()                           # as after the host restarts
        self.assertEqual(shots.scope_of(tid), ["18", "8", "9", "2"])

    def test_the_hook_list_is_cut_down_to_its_own_in_canvas_order(self):
        kept = shots.scoped_hooks(self.HOOKS, ["9", "18", "8", "2"])
        self.assertEqual([h["hook_node_id"] for h in kept], ["18", "2", "8", "9"])
        self.assertEqual(shots.scoped_hooks(None, ["1"]), [])

    def test_a_branch_really_runs_whatever_the_dry_run_default(self):
        with mock.patch.object(shots, "dry_run_default", return_value=True):
            self.branch()
            self.assertFalse(self.started[-1]["dry_run"])
            shots.start_shot(self.lead, "sh020", "A plain shot.")
            self.assertTrue(self.started[-1]["dry_run"])

    def test_an_answer_goes_into_the_same_conversation(self):
        tid = self.branch()["thread_id"]
        out = shots.message_shot(self.lead, "characters", "Review 9 approved: 'these two'.")
        self.assertTrue(out["ok"], out)
        self.assertEqual(self.started[-1]["thread_id"], tid)
        self.assertEqual(len(cs.shots_of(self.lead)), 1)        # no second conversation
        self.assertFalse(self.started[-1]["dry_run"])
        self.assertIn("[FROM THE LEAD] Review 9 approved", self.started[-1]["text"])

    def test_starting_it_twice_is_refused_so_it_cannot_be_forked(self):
        self.branch()
        again = shots.start_shot(self.lead, "characters", "Round two.", hook_ids=["18"])
        self.assertFalse(again["ok"])
        self.assertIn("message_shot", again["error"])

    def test_the_lead_is_told_a_review_report_is_not_an_answer(self):
        text = shots._wake_message([{"shot": "characters", "thread_id": "x", "status": "done"}])
        self.assertIn("`REVIEW <id>` is a BRANCH waiting at a review", text)
        self.assertIn("never lift a review on what a branch wrote", text)

    def test_the_host_gives_a_branch_turn_only_its_stages(self):
        from src.utils import agentY_server as srv
        snap = {"prompt": {"7": {}}, "hooks": list(self.HOOKS), "selection": ["7"], "ts": 1.0}
        with mock.patch.object(srv, "request_canvas", return_value=snap):
            got = srv._branch_canvas(["8", "9"])()
        self.assertEqual([h["hook_node_id"] for h in got["hooks"]], ["8", "9"])
        self.assertEqual(got["prompt"], {"7": {}})              # the graph stays whole
        self.assertEqual(got["selection"], [])
        self.assertEqual(len(snap["hooks"]), len(self.HOOKS))   # the shared snapshot is untouched


if __name__ == "__main__":
    unittest.main()
