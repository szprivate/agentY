"""Running agentY as a server: queue, per-user tokens, headless mode, attribution.

Everything here is opt-in; the last class pins that an install which sets none of
it behaves as before.

    python -m unittest tests.test_team_server
"""

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from src.utils import (api_guard, server_mode, settings, team_store, turn_bus,
                       turn_queue, user_auth)


class TurnQueueTest(unittest.TestCase):

    def _q(self, limit=1, max_queued=5):
        return turn_queue.TurnQueue(limit=limit, max_queued=max_queued)

    def test_starts_at_once_when_there_is_room(self):
        q = self._q()
        t = q.submit(lambda: "r1", user="a")
        self.assertEqual((t.state, t.request_id, q.position(t)), (turn_queue.RUNNING, "r1", 0))

    def test_waits_in_order_and_starts_when_a_turn_finishes(self):
        q = self._q()
        started = []
        a = q.submit(lambda: started.append("a") or "ra")
        b = q.submit(lambda: started.append("b") or "rb")
        c = q.submit(lambda: started.append("c") or "rc")
        self.assertEqual((q.position(b), q.position(c)), (1, 2))
        self.assertEqual(started, ["a"])
        q.finished("ra")
        self.assertEqual(started, ["a", "b"])
        self.assertEqual(q.position(c), 1)
        q.finished("rb")
        q.finished("rc")
        self.assertEqual(started, ["a", "b", "c"])
        self.assertEqual(a.state, turn_queue.DONE)

    def test_the_limit_is_read_each_time(self):
        n = {"v": 1}
        q = turn_queue.TurnQueue(limit=lambda: n["v"])
        q.submit(lambda: "r1")
        b = q.submit(lambda: "r2")
        self.assertEqual(b.state, turn_queue.QUEUED)
        n["v"] = 2
        q.pump()
        self.assertEqual(b.state, turn_queue.RUNNING)

    def test_full_queue_refuses(self):
        q = self._q(max_queued=1)
        q.submit(lambda: "r1")
        q.submit(lambda: "r2")
        with self.assertRaises(turn_queue.QueueFull):
            q.submit(lambda: "r3")

    def test_a_starter_that_raises_does_not_jam_the_line(self):
        q = self._q()

        def boom():
            raise RuntimeError("x")

        a = q.submit(boom)
        b = q.submit(lambda: "rb")
        self.assertEqual(a.state, turn_queue.CANCELLED)
        self.assertEqual(b.state, turn_queue.RUNNING)

    def test_a_starter_that_starts_nothing_frees_its_place(self):
        q = self._q()
        q.submit(lambda: "")
        b = q.submit(lambda: "rb")
        self.assertEqual(b.state, turn_queue.RUNNING)

    def test_cancel_removes_a_waiting_ticket(self):
        q = self._q()
        q.submit(lambda: "r1")
        b = q.submit(lambda: "r2")
        self.assertTrue(q.cancel(b.id))
        self.assertEqual(q.snapshot()["queued"], [])

    def test_listeners_hear_about_every_change(self):
        q = self._q()
        seen = []
        q.add_listener(lambda t: seen.append((t.id, t.state)))
        q.submit(lambda: "r1")
        b = q.submit(lambda: "r2")
        q.finished("r1")
        self.assertIn((b.id, turn_queue.QUEUED), seen)
        self.assertIn((b.id, turn_queue.RUNNING), seen)

    def test_sweep_releases_a_turn_that_vanished(self):
        q = self._q()
        a = q.submit(lambda: "r1")
        b = q.submit(lambda: "r2")
        a.started -= 100
        q.sweep([])
        self.assertEqual(b.state, turn_queue.RUNNING)

    def test_snapshot_lists_running_and_waiting(self):
        q = self._q()
        q.submit(lambda: "r1", user="anna", text="poster")
        q.submit(lambda: "r2", user="ben", text="logo")
        snap = q.snapshot()
        self.assertEqual(snap["limit"], 1)
        self.assertEqual(snap["running"][0]["user"], "anna")
        self.assertEqual(snap["queued"][0]["position"], 1)

    def test_the_shared_limit_never_exceeds_the_pool(self):
        fake = {"server": {"max_running_turns": 9}, "parallel_chats": 3}
        with mock.patch("src.utils.settings.load_settings", return_value=fake):
            self.assertEqual(turn_queue._running_limit(), 3)
        fake = {"server": {"max_running_turns": 2}, "parallel_chats": 5}
        with mock.patch("src.utils.settings.load_settings", return_value=fake):
            self.assertEqual(turn_queue._running_limit(), 2)

    def test_the_shared_queue_hears_done_from_the_bus(self):
        before = list(turn_bus._observers)
        with mock.patch.object(turn_queue, "_SHARED", None):
            q = turn_queue.shared()
            # Take out only the observer this test added; others (the turn API's)
            # belong to the process.
            self.addCleanup(lambda: [turn_bus._observers.remove(f)
                                     for f in list(turn_bus._observers) if f not in before])
            a = q.submit(lambda: "rX")
            b = q.submit(lambda: "rY")
            b2 = q.submit(lambda: "rZ")
            self.assertEqual((a.state, b.state), (turn_queue.RUNNING, turn_queue.RUNNING))
            self.assertEqual(b2.state, turn_queue.QUEUED)
            for fn in list(turn_bus._observers):
                fn({"type": "done"}, turn_bus.Turn("rX", "t"))
            self.assertEqual(b2.state, turn_queue.RUNNING)


class UserTokenTest(unittest.TestCase):

    def setUp(self):
        d = tempfile.TemporaryDirectory()
        self.addCleanup(d.cleanup)
        self.path = Path(d.name) / "users.json"

    def test_a_token_is_shown_once_and_stored_hashed(self):
        token = user_auth.add_user("anna", slack_id="U1", roles=["approver"], path=self.path)
        raw = self.path.read_text()
        self.assertNotIn(token, raw)
        users = user_auth.load(self.path)
        self.assertEqual(user_auth.match_token(token, users)["name"], "anna")
        self.assertEqual(user_auth.match_token(token, users)["roles"], ["approver"])

    def test_a_wrong_or_empty_token_matches_nobody(self):
        user_auth.add_user("anna", path=self.path)
        users = user_auth.load(self.path)
        self.assertIsNone(user_auth.match_token("nope", users))
        self.assertIsNone(user_auth.match_token("", users))

    def test_two_users_do_not_mix(self):
        a = user_auth.add_user("anna", path=self.path)
        b = user_auth.add_user("ben", path=self.path)
        users = user_auth.load(self.path)
        self.assertEqual(user_auth.match_token(a, users)["name"], "anna")
        self.assertEqual(user_auth.match_token(b, users)["name"], "ben")

    def test_revoking_a_user_ends_their_token(self):
        a = user_auth.add_user("anna", path=self.path)
        self.assertTrue(user_auth.remove_user("anna", path=self.path))
        self.assertIsNone(user_auth.match_token(a, user_auth.load(self.path)))

    def test_reissuing_replaces_the_old_token_and_keeps_the_roles(self):
        old = user_auth.add_user("anna", slack_id="U1", roles=["approver"], path=self.path)
        new = user_auth.add_user("anna", path=self.path)
        users = user_auth.load(self.path)
        self.assertIsNone(user_auth.match_token(old, users))
        self.assertEqual(user_auth.match_token(new, users)["slack_id"], "U1")

    def test_bad_names_are_refused(self):
        with self.assertRaises(ValueError):
            user_auth.add_user("a b/c", path=self.path)

    def test_bearer_header_or_the_panel_header(self):
        self.assertEqual(user_auth.bearer_or_header({"Authorization": "Bearer abc"}), "abc")
        self.assertEqual(user_auth.bearer_or_header({"X-AgentY-Token": "xyz"}), "xyz")
        self.assertEqual(user_auth.bearer_or_header({}), "")

    def test_broken_file_means_no_users(self):
        self.path.write_text("{not json")
        self.assertEqual(user_auth.load(self.path), {})

    def test_slack_roles_maps_member_ids(self):
        user_auth.add_user("anna", slack_id="U1", roles=["approver"], path=self.path)
        with mock.patch.dict(os.environ, {"AGENTY_USERS_FILE": str(self.path)}):
            self.assertEqual(user_auth.slack_roles(), {"U1": ["approver"]})


class GuardWithUserTokensTest(unittest.TestCase):
    """api_guard.verdict with a matched user token standing in for the session token."""

    BASE = dict(method="POST", path="/agentY/turns", host_header="agenty.example.com",
                origin="", token="", expected_token="SESSION", agent_port=5000,
                allowed_hosts=("agenty.example.com",))

    def test_a_user_token_is_enough(self):
        self.assertEqual(api_guard.verdict(**self.BASE, user_ok=True), (True, ""))

    def test_no_token_and_no_user_is_refused(self):
        ok, why = api_guard.verdict(**self.BASE)
        self.assertFalse(ok)
        self.assertIn("token", why)

    def test_a_user_token_does_not_excuse_a_wrong_host(self):
        base = dict(self.BASE, host_header="evil.example.com")
        ok, _ = api_guard.verdict(**base, user_ok=True)
        self.assertFalse(ok)

    def test_the_session_token_still_works(self):
        self.assertEqual(api_guard.verdict(**dict(self.BASE, token="SESSION")), (True, ""))

    def test_health_stays_public(self):
        base = dict(self.BASE, path="/agentY/health", method="GET")
        self.assertEqual(api_guard.verdict(**base), (True, ""))


class GuardInTheServerTest(unittest.TestCase):
    """The flask end: Authorization: Bearer reaches verdict and names the user."""

    def test_bearer_token_identifies_the_user(self):
        from src.utils import agentY_server as srv
        flask = __import__("flask")
        app = flask.Flask("t")
        d = tempfile.TemporaryDirectory()
        self.addCleanup(d.cleanup)
        path = Path(d.name) / "users.json"
        token = user_auth.add_user("anna", path=path)
        with mock.patch.dict(os.environ, {"AGENTY_USERS_FILE": str(path),
                                          "AGENTY_USER_TOKENS": "1"}):
            with app.test_request_context(
                    "/agentY/turns", method="POST",
                    headers={"Authorization": f"Bearer {token}", "Host": "127.0.0.1:5000"}):
                ok, why = srv._guard_verdict(flask.request)
                self.assertTrue(ok, why)
                self.assertEqual(flask.g.agenty_user["name"], "anna")
            with app.test_request_context(
                    "/agentY/turns", method="POST",
                    headers={"Authorization": "Bearer wrong", "Host": "127.0.0.1:5000"}):
                ok, _ = srv._guard_verdict(flask.request)
                self.assertFalse(ok)


class TeamStoreTest(unittest.TestCase):

    def setUp(self):
        d = tempfile.TemporaryDirectory()
        self.addCleanup(d.cleanup)
        self.enterContext(mock.patch.dict(os.environ, {"AGENTY_TEAM_DIR": d.name}))
        team_store._requesters.clear()
        team_store._loaded = False
        self.addCleanup(team_store._requesters.clear)

    def test_the_first_requester_keeps_the_conversation(self):
        team_store.set_requester("t1", user="U1", channel="C1")
        team_store.set_requester("t1", user="U2", channel="C1")
        self.assertEqual(team_store.requester("t1")["user"], "U1")

    def test_owners_survive_a_restart(self):
        team_store.set_requester("t1", user="U1", channel="C1", name="Anna")
        team_store._requesters.clear()
        team_store._loaded = False
        self.assertEqual(team_store.requester("t1")["name"], "Anna")

    def test_a_finished_turn_writes_one_usage_line_to_its_owner(self):
        team_store.set_requester("t1", user="U1", channel="C1", project="proj-a", name="Anna")
        turn = turn_bus.Turn("r1", "t1", origin="slack", text="poster")
        team_store.on_turn_event({"type": "usage", "input": 100, "output": 20, "cost": 0.5,
                                  "calls": 3}, turn)
        team_store.on_turn_event({"type": "usage", "input": 300, "output": 50, "cost": 1.25,
                                  "calls": 7}, turn)
        team_store.on_turn_event({"type": "done"}, turn)
        team_store.on_turn_event({"type": "done"}, turn)          # twice: still one line
        lines = team_store.usage_path().read_text().strip().splitlines()
        self.assertEqual(len(lines), 1)
        rec = json.loads(lines[0])
        self.assertEqual((rec["user"], rec["project"], rec["input"], rec["cost"]),
                         ("U1", "proj-a", 300, 1.25))      # the last usage event is the total

    def test_a_turn_nobody_owns_leaves_no_trace(self):
        turn = turn_bus.Turn("r2", "panel-thread", origin="panel")
        team_store.on_turn_event({"type": "usage", "input": 1, "output": 1, "cost": 1}, turn)
        team_store.on_turn_event({"type": "done"}, turn)
        self.assertFalse(team_store.usage_path().exists())

    def test_summary_adds_up_per_user_and_per_project(self):
        for rid, user, proj, cost in (("a", "U1", "P1", 1.0), ("b", "U1", "P2", 2.0),
                                      ("c", "U2", "P1", 0.5)):
            team_store.set_requester("t" + rid, user=user, channel=proj, project=proj)
            turn = turn_bus.Turn(rid, "t" + rid, origin="slack")
            team_store.on_turn_event({"type": "usage", "input": 10, "output": 5, "cost": cost},
                                     turn)
            team_store.on_turn_event({"type": "done"}, turn)
        by_user = {r["user"]: r for r in team_store.usage_summary("user")}
        self.assertEqual(by_user["U1"]["cost"], 3.0)
        self.assertEqual(by_user["U1"]["turns"], 2)
        by_proj = {r["project"]: r for r in team_store.usage_summary("project")}
        self.assertEqual(by_proj["P1"]["cost"], 1.5)

    def test_notes_are_per_user_and_per_project(self):
        team_store.add_note("users", "U1", "likes   16:9")
        team_store.add_note("projects", "C1", "Acme")
        self.assertEqual(team_store.read_notes("users", "U1"), "- likes 16:9")
        self.assertEqual(team_store.read_notes("users", "U2"), "")
        self.assertTrue(team_store.clear_notes("users", "U1"))
        self.assertEqual(team_store.read_notes("users", "U1"), "")

    def test_a_note_key_cannot_walk_out_of_the_folder(self):
        team_store.add_note("users", "../../evil", "x")
        written = list(team_store.team_dir().rglob("*.md"))
        self.assertEqual(len(written), 1)
        self.assertEqual(written[0].parent, team_store.team_dir() / "users")

    def test_context_names_who_asked_and_what_they_asked_to_keep(self):
        team_store.add_note("users", "U1", "no text in images")
        block = team_store.context_block({"user": "U1", "name": "Anna", "project": "C1"})
        self.assertIn("Anna", block)
        self.assertIn("no text in images", block)
        self.assertEqual(team_store.context_block({}), "")


class HeadlessTest(unittest.TestCase):

    def test_off_by_default(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("AGENTY_HEADLESS", None)
            with mock.patch("src.utils.settings.load_settings", return_value={}):
                self.assertFalse(server_mode.headless())

    def test_the_setting_and_the_environment_turn_it_on(self):
        with mock.patch("src.utils.settings.load_settings",
                        return_value={"server": {"headless": True}}):
            os.environ.pop("AGENTY_HEADLESS", None)
            self.assertTrue(server_mode.headless())
        with mock.patch.dict(os.environ, {"AGENTY_HEADLESS": "1"}):
            self.assertTrue(server_mode.headless())
        with mock.patch.dict(os.environ, {"AGENTY_HEADLESS": "0"}), \
                mock.patch("src.utils.settings.load_settings",
                           return_value={"server": {"headless": True}}):
            self.assertFalse(server_mode.headless())

    def test_the_canvas_tools_answer_at_once_with_the_reason(self):
        from src.utils import canvas_probe
        with mock.patch.object(server_mode, "headless", return_value=True):
            import time
            t0 = time.time()
            out = canvas_probe.request("screenshot", {}, timeout=20)
            self.assertLess(time.time() - t0, 1)
        self.assertTrue(out["headless"])
        self.assertIn("headless", out["error"])
        self.assertIn("Slack", out["error"])

    def test_a_headless_turn_does_not_wait_for_a_canvas(self):
        from src.utils import agentY_server as srv
        with mock.patch.object(server_mode, "headless", return_value=True):
            import time
            t0 = time.time()
            self.assertEqual(srv.request_canvas(wait=30), {})
            self.assertLess(time.time() - t0, 1)

    def test_no_canvas_switch_is_on_without_a_canvas(self):
        from src.utils import agentY_server as srv
        with mock.patch.object(server_mode, "headless", return_value=True):
            self.assertFalse(srv._drop_outputs_into_canvas())
            self.assertFalse(srv._show_outputs_in_panel())
            self.assertFalse(srv._place_text_nodes_on_canvas())

    def test_the_switches_are_untouched_when_not_headless(self):
        from src.utils import agentY_server as srv
        with mock.patch.object(server_mode, "headless", return_value=False), \
                mock.patch.dict(os.environ, {"AGENTY_CANVAS_DROP": "1"}):
            self.assertTrue(srv._drop_outputs_into_canvas())


class SettingsEnvTest(unittest.TestCase):

    def test_nothing_set_changes_nothing(self):
        cfg = {"comfyui_url": "http://127.0.0.1:8188", "server": {"headless": False}}
        out = settings.apply_env_overrides(json.loads(json.dumps(cfg)), env={})
        self.assertEqual(out, cfg)

    def test_paths_and_url_come_from_the_environment(self):
        env = {"AGENTY_COMFYUI_URL": "http://gpu01:8188", "AGENTY_OUTPUT_DIR": "//nas/out",
               "AGENTY_COMFYUI_MODELS_DIR": "//nas/models", "AGENTY_HEADLESS": "yes",
               "AGENTY_BIND_HOST": "0.0.0.0"}
        out = settings.apply_env_overrides({"comfyui_url": "http://127.0.0.1:8188"}, env=env)
        self.assertEqual(out["comfyui_url"], "http://gpu01:8188")
        self.assertEqual(out["output_dir"], "//nas/out")
        self.assertEqual(out["comfyui_models_dir"], "//nas/models")
        self.assertIs(out["server"]["headless"], True)
        self.assertEqual(out["server"]["bind_host"], "0.0.0.0")

    def test_blank_values_are_ignored(self):
        out = settings.apply_env_overrides({"comfyui_url": "x"}, env={"AGENTY_COMFYUI_URL": "  "})
        self.assertEqual(out["comfyui_url"], "x")

    def test_the_shipped_defaults_keep_every_server_feature_off(self):
        d = settings.load_defaults()
        self.assertEqual(d["server"]["bind_host"], "")
        self.assertFalse(d["server"]["headless"])
        self.assertFalse(d["security"]["user_tokens"])
        self.assertEqual(d["slack"]["channels"], [])
        self.assertFalse(d["slack"]["channel_members_may_ask"])
        self.assertFalse(d["slack"]["enabled"])

    def test_bind_host_setting_is_picked_up_by_the_launcher_default(self):
        import src.agenty_ui_server as ui
        fake = {"server": {"bind_host": "0.0.0.0"}}
        with mock.patch("src.utils.settings.load_settings", return_value=fake):
            host, _port = ui._agent_server_url_defaults()
        self.assertEqual(host, "0.0.0.0")
        with mock.patch("src.utils.settings.load_settings", return_value={}):
            host, _port = ui._agent_server_url_defaults()
        self.assertEqual(host, "127.0.0.1")


class ToolPermissionsFallbackTest(unittest.TestCase):

    def test_with_no_asker_nothing_changed(self):
        from src.utils import tool_permissions as tp
        tp.set_remote_asker(None)
        d = tp.request("run_script", {"command": "ls"}, timeout=1)
        self.assertFalse(d.allowed)

    def test_unattended_allow_is_still_honoured(self):
        from src.utils import tool_permissions as tp
        called = []
        tp.set_remote_asker(lambda *a: called.append(a))
        self.addCleanup(tp.set_remote_asker, None)
        with mock.patch.object(tp, "has_listener", return_value=False):
            d = tp.request("run_script", {"command": "ls"}, timeout=1, unattended_allows=True)
        self.assertTrue(d.allowed)
        self.assertEqual(called, [])


if __name__ == "__main__":
    unittest.main()
