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


if __name__ == "__main__":
    unittest.main()
