"""A turn's canvas is the canvas of the panel it belongs to - never "whichever
panel answered last".

A script that drove agentY through the turn API, with a headless browser of its
own open on ComfyUI, had the agent's canvas work land in the user's own tab: the
host raised one global "post me the graph" flag, the first page to answer filled
one global cache, and the edits (and the workflow loads, which ComfyUI broadcasts
to every page) were aimed at that. Now every panel names itself and the
conversation it shows, and a turn asks only the panel its target names.

Two simulated panels: the USER's (showing conversation U) and the SCRIPT's
(showing conversation A).

    python -m unittest tests.test_canvas_target
"""

import threading
import time
import unittest
from unittest import mock

from agenty_core.utils import turn_scope

from src.utils import agentY_server as srv
from src.utils import canvas_probe, canvas_target, server_mode, turn_bus


def _graph(tag):
    return {"1": {"class_type": "KSampler", "inputs": {"seed": tag}}}


class _Panels:
    """Simulated pages: each heartbeats, and answers `want_canvas` for itself."""

    def __init__(self):
        self.stop = threading.Event()
        self.threads = []
        self.answered = []

    def open(self, client, thread_id, tag):
        canvas_target.touch(client, thread_id)

        def run():
            while not self.stop.is_set():
                canvas_target.touch(client, thread_id)
                if srv.canvas_wanted(client):
                    self.answered.append(client)
                    srv.remember_canvas(_graph(tag), [], [], client=client)
                time.sleep(0.02)
        t = threading.Thread(target=run, daemon=True)
        t.start()
        self.threads.append(t)

    def close(self):
        self.stop.set()
        for t in self.threads:
            t.join(2)


class _Base(unittest.TestCase):
    def setUp(self):
        canvas_target._reset_for_tests()
        srv._canvas_cache.clear()
        srv._canvas_by_client.clear()
        srv._canvas_wanted_by.clear()
        srv._turn_canvas.clear()
        for c in (srv._canvas_cache, srv._canvas_by_client, srv._canvas_wanted_by, srv._turn_canvas):
            self.addCleanup(c.clear)
        self.addCleanup(canvas_target._reset_for_tests)
        self.enterContext(mock.patch.object(server_mode, "headless", return_value=False))
        self.panels = _Panels()
        self.addCleanup(self.panels.close)
        self.panels.open("user-tab", "U", 1)
        self.panels.open("script-tab", "A", 2)


class WhichPanelTest(_Base):

    def test_a_conversation_nobody_shows_has_no_canvas_and_asks_nobody(self):
        got = srv.canvas_provider_for("B")()
        self.assertEqual(got, {})
        time.sleep(0.15)
        self.assertEqual(self.panels.answered, [], "no page was asked for its graph")
        self.assertFalse(srv.canvas_wanted())

    def test_the_panel_showing_the_conversation_is_the_one_asked(self):
        got = srv.canvas_provider_for("A")()
        self.assertEqual(got["client"], "script-tab")
        self.assertEqual(got["prompt"], _graph(2))
        self.assertEqual(set(self.panels.answered), {"script-tab"}, "the user's tab was never asked")

    def test_the_users_own_tab_is_not_reachable_from_another_conversation(self):
        got = srv.canvas_provider_for("A", "user-tab")()      # names the user's tab for conversation A
        self.assertEqual(got, {})
        self.assertEqual(self.panels.answered, [])

    def test_none_means_none_even_with_a_panel_open(self):
        self.assertEqual(srv.canvas_provider_for("A", "none")(), {})
        self.assertEqual(self.panels.answered, [])

    def test_a_named_panel_must_show_the_conversation(self):
        self.assertEqual(srv.canvas_provider_for("A", "script-tab")()["client"], "script-tab")

    def test_a_closed_panel_is_not_asked(self):
        with mock.patch.object(canvas_target, "PANEL_TTL", -1.0):
            self.assertEqual(srv.canvas_provider_for("A")(), {})

    def test_two_tabs_on_one_conversation_the_named_one_wins(self):
        self.panels.open("script-tab-2", "A", 3)
        self.assertEqual(srv.canvas_provider_for("A", "script-tab-2")()["prompt"], _graph(3))

    def test_a_panels_snapshot_never_serves_another_panels_turn(self):
        srv.remember_canvas(_graph(99), [], [], client="user-tab")
        got = srv.canvas_provider_for("A")()
        self.assertEqual(got["prompt"], _graph(2))


class TurnWiringTest(_Base):
    """_run_pipeline_stream: what the turn is told about its canvas."""

    def _run(self, thread_id, origin, provider):
        seen = {}

        def fake_turn(*a, **kw):
            seen["scope"] = dict(turn_scope.current()._slots)
            seen["prompt"] = kw.get("canvas_prompt")

        with mock.patch.object(srv, "_take_pipeline", return_value=object()), \
                mock.patch.object(srv, "_run_pipeline_turn", fake_turn), \
                mock.patch.object(srv, "_wd", mock.MagicMock()), \
                mock.patch.object(srv, "_pool", None), \
                mock.patch.object(turn_bus, "tee", lambda q, **k: q):
            srv._run_pipeline_stream(thread_id, "go", [], mock.MagicMock(), "rid-" + thread_id,
                                     origin=origin, canvas_provider=provider)
        return seen

    def test_an_api_turn_with_no_panel_is_marked_detached(self):
        seen = self._run("B", "api", srv.canvas_provider_for("B"))
        self.assertTrue(seen["scope"].get("canvas_detached"))
        self.assertIsNone(seen["prompt"])
        self.assertNotIn("rid-B", srv._turn_canvas)

    def test_an_api_turn_with_its_panel_is_bound_to_it(self):
        seen = self._run("A", "api", srv.canvas_provider_for("A"))
        self.assertEqual(seen["scope"].get("canvas_client"), "script-tab")
        self.assertFalse(seen["scope"].get("canvas_detached"))
        self.assertEqual(seen["prompt"], _graph(2))
        self.assertEqual(srv._turn_canvas["rid-A"], "script-tab")

    def test_a_slack_turn_gets_a_canvas_only_when_a_panel_shows_it(self):
        self.assertTrue(self._run("B", "slack", srv.canvas_provider_for("B"))["scope"].get("canvas_detached"))
        self.assertEqual(self._run("A", "slack", srv.canvas_provider_for("A"))["scope"].get("canvas_client"),
                         "script-tab")

    def test_a_headless_host_has_no_canvas_for_anyone(self):
        with mock.patch.object(server_mode, "headless", return_value=True):
            seen = self._run("A", "api", None)
        self.assertTrue(seen["scope"].get("canvas_detached"))

    def test_a_panel_turn_is_not_marked_detached(self):
        seen = self._run("A", "panel", None)
        self.assertFalse(seen["scope"].get("canvas_detached"))

    def test_no_canvas_says_what_to_do_instead(self):
        from src.pipeline import _no_graph_msg
        tok = turn_scope.enter(turn_scope.Scope("r", "B"))
        try:
            turn_scope.current().set("canvas_detached", "conversation")
            msg = _no_graph_msg()
        finally:
            turn_scope.leave(tok)
        self.assertIn("no canvas attached to this turn", msg)
        self.assertIn("open_workflow_in_canvas", msg)
        self.assertIn("no ComfyUI panel is showing this conversation", msg)
        self.assertEqual(_no_graph_msg(), "no graph is open on the canvas this turn.")


class ApiRequestTest(unittest.TestCase):

    def test_the_request_carries_the_target_to_the_turn(self):
        ran = []
        with mock.patch.object(srv, "_run_pipeline_stream", lambda *a, **k: ran.append(k)), \
                mock.patch.object(srv, "_drain_queue", lambda q: None), \
                mock.patch.object(srv, "_use_api_queue", return_value=False), \
                mock.patch.object(server_mode, "headless", return_value=False), \
                mock.patch.object(srv.cs, "get_thread", lambda t: {"id": t}), \
                mock.patch.object(srv.cs, "add_message", lambda *a, **k: 1), \
                mock.patch.object(srv, "running_threads", lambda: []):
            srv._api_start_turn("hi", [], "conv", canvas="none")
            srv._api_start_turn("hi", [], "conv")
            for t in threading.enumerate():
                if t.name == "agentY-api-turn":
                    t.join(2)
        targets = sorted(str(k["canvas_provider"].target or "") for k in ran)
        self.assertEqual(targets, ["", "none"])         # "" = the default: the conversation's panel


class EditDeliveryTest(unittest.TestCase):
    """Which page a watched turn's canvas patches go to."""

    EDIT = {"op": "edit_graph", "ops": [{"op": "add"}]}

    def setUp(self):
        srv._turn_canvas.clear()
        self.addCleanup(srv._turn_canvas.clear)

    def _turn(self, origin):
        return turn_bus.Turn(request_id="r1", thread_id="A", origin=origin)

    def test_only_the_owning_page_gets_an_api_turns_edits_and_only_once(self):
        srv._turn_canvas["r1"] = "script-tab"
        t = self._turn("api")
        self.assertFalse(srv._watch_may_edit(t, "r1", 0, self.EDIT, "user-tab"))
        self.assertFalse(srv._watch_may_edit(t, "r1", 0, self.EDIT, ""))
        self.assertTrue(srv._watch_may_edit(t, "r1", 0, self.EDIT, "script-tab"))
        self.assertFalse(srv._watch_may_edit(t, "r1", 0, self.EDIT, "script-tab"), "not replayed")

    def test_a_turn_with_no_canvas_edits_no_page(self):
        t = self._turn("slack")
        for client in ("user-tab", "script-tab", ""):
            self.assertFalse(srv._watch_may_edit(t, "r2", 0, self.EDIT, client))

    def test_a_panel_turn_is_unchanged(self):
        t = self._turn("panel")
        self.assertFalse(srv._watch_may_edit(t, "r3", 0, self.EDIT, "user-tab"))      # structure: never replayed
        self.assertTrue(srv._watch_may_edit(t, "r3", 1, {"op": "review_released"}, "user-tab"))


class ProbeTest(unittest.TestCase):

    def setUp(self):
        canvas_probe.clear()
        self.addCleanup(canvas_probe.clear)
        self.enterContext(mock.patch.object(server_mode, "headless", return_value=False))

    def _ask(self, slots):
        out = {}
        tok = turn_scope.enter(turn_scope.Scope("r", "t"))
        for k, v in slots.items():
            turn_scope.current().set(k, v)

        def go():
            out["r"] = canvas_probe.request("screenshot", {}, timeout=1.5)
        th = threading.Thread(target=turn_scope.bind(go))
        th.start()
        turn_scope.leave(tok)
        return th, out

    def test_a_probe_goes_to_the_turns_page_only(self):
        th, out = self._ask({"canvas_client": "script-tab"})
        time.sleep(0.15)
        self.assertIsNone(canvas_probe.take("user-tab"))
        probe = canvas_probe.take("script-tab")
        self.assertEqual(probe["kind"], "screenshot")
        canvas_probe.reply(probe["probe_id"], {"ok": True})
        th.join(3)
        self.assertEqual(out["r"], {"ok": True})

    def test_a_turn_with_no_canvas_asks_no_page(self):
        th, out = self._ask({"canvas_detached": True})
        th.join(3)
        self.assertTrue(out["r"].get("no_canvas"))
        self.assertIsNone(canvas_probe.take("user-tab"))

    def test_an_unbound_panel_turn_is_asked_of_whoever_polls(self):
        th, out = self._ask({})
        time.sleep(0.15)
        probe = canvas_probe.take("")
        self.assertIsNotNone(probe)
        canvas_probe.reply(probe["probe_id"], {"ok": 1})
        th.join(3)


class AckTest(unittest.TestCase):
    """An edit is only "applied" once a page says so."""

    def setUp(self):
        from src.utils import canvas_ack
        self.ack = canvas_ack
        canvas_ack._reset_for_tests()
        self.addCleanup(canvas_ack._reset_for_tests)
        tok = turn_scope.enter(turn_scope.Scope("rq", "A"))
        self.addCleanup(turn_scope.leave, tok)

    def _push(self, bound=True, op="edit_graph"):
        from src.utils import canvas_patch
        if bound:
            turn_scope.current().set("canvas_client", "script-tab")
        return canvas_patch.push({"op": op, "ops": []})

    def test_a_patch_of_a_bound_turn_is_tracked_and_an_unbound_one_is_not(self):
        self.assertTrue(self._push(bound=True))
        turn_scope.current().set("canvas_client", "")
        self.assertEqual(self._push(bound=False), "")

    def test_no_page_confirms_so_the_tool_says_not_applied(self):
        import asyncio
        from src.pipeline import _patch_unconfirmed
        pid = self._push()
        with mock.patch.object(self.ack, "wait", lambda p, t: {"ok": False, "timeout": True}):
            bad = asyncio.run(_patch_unconfirmed(pid))
        self.assertEqual(bad["status"], "not_applied")
        self.assertIn("NOT applied to any canvas", bad["error"])
        self.assertIn("open_workflow_in_canvas", bad["what_to_do"])

    def test_a_confirmed_patch_is_applied(self):
        import asyncio
        from src.pipeline import _patch_unconfirmed
        pid = self._push()
        threading.Timer(0.1, lambda: self.ack.ack(pid, True, applied=3)).start()
        self.assertIsNone(asyncio.run(_patch_unconfirmed(pid)))

    def test_a_page_that_could_not_apply_it_says_why(self):
        import asyncio
        from src.pipeline import _patch_unconfirmed
        pid = self._push()
        threading.Timer(0.1, lambda: self.ack.ack(pid, False, missed=["no workflow is open on this page"])).start()
        bad = asyncio.run(_patch_unconfirmed(pid))
        self.assertIn("no workflow is open on this page", bad["error"])

    def test_a_patch_given_up_on_is_never_applied_late(self):
        pid = self._push()
        self.ack.wait(pid, 0.01)                          # times out: expired
        srv._turn_canvas["rq"] = "script-tab"
        self.addCleanup(srv._turn_canvas.clear)
        t = turn_bus.Turn(request_id="rq", thread_id="A", origin="api")
        self.assertFalse(srv._watch_may_edit(t, "rq", 0, {"op": "edit_graph", "patch_id": pid}, "script-tab"))

    def test_an_untracked_patch_counts_as_confirmed(self):
        import asyncio
        from src.pipeline import _patch_unconfirmed
        self.assertIsNone(asyncio.run(_patch_unconfirmed("")))


if __name__ == "__main__":
    unittest.main()
