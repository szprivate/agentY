"""The word `stop`, through the actual HTTP route the panel posts to.

The routing rule has its own tests. This asks the question the user asks: press
Stop while a review halt is standing, and does the host survive? Nothing else in
the suite drives `/agentY/chat`, and the rule is applied there — a fix that is
correct in `_is_command` and mis-wired in the route is exactly as broken as no
fix at all.

    python -m unittest discover -s tests
"""

import json
import unittest
from unittest import mock

from src.utils import agentY_server as srv


class ChatRouteTest(unittest.TestCase):

    def setUp(self):
        self.commands = []
        self.turns = []

        def fake_command(thread_id, text, canvas_prompt=None):
            self.commands.append(text)
            return [{"type": "system", "data": "command ran"}]

        def fake_stream(thread_id, message, image_paths, q, rid, **kw):
            self.turns.append(message)
            q.put({"type": "done"})
            q.put(None)

        self.enterContext(mock.patch.object(srv, "_handle_command", fake_command))
        self.enterContext(mock.patch.object(srv, "_run_pipeline_stream", fake_stream))
        self.enterContext(mock.patch.object(srv.cs, "get_thread",
                                            lambda tid: {"id": tid, "messages": []}))
        self.enterContext(mock.patch.object(srv.cs, "add_message", lambda *a, **k: 1))
        self.enterContext(mock.patch.object(srv.cs, "save_panel", lambda *a, **k: None))
        app = srv._build_app()
        app.config["TESTING"] = True
        self.client = app.test_client()

    def _post(self, message, halted, thread_id="t1"):
        self.asked = []
        def fake_pending(tid):
            self.asked.append(tid)
            return halted
        with mock.patch.object(srv, "_halt_pending", fake_pending):
            r = self.client.post("/agentY/chat",
                                 data=json.dumps({"thread_id": thread_id,
                                                  "message": message}),
                                 content_type="application/json")
            r.get_data()          # drain the SSE generator
        return r

    # ── the bug ───────────────────────────────────────────────────────────────
    def test_stop_during_a_halt_does_not_reach_the_shutdown_command(self):
        self._post("stop", halted=True)
        self.assertEqual(self.commands, [], "it went to the command handler")
        self.assertEqual(self.turns, ["stop"], "and never reached the halt as an answer")

    def test_the_action_bar_continue_button_is_not_a_command_either(self):
        self._post("continue", halted=True)
        self.assertEqual(self.commands, [])
        self.assertEqual(self.turns, ["continue"])

    # ── and nothing else moved ────────────────────────────────────────────────
    def test_stop_with_no_halt_is_still_the_command(self):
        self._post("stop", halted=False)
        self.assertEqual(self.commands, ["stop"])
        self.assertEqual(self.turns, [])

    def test_a_slash_command_still_works_mid_halt(self):
        self._post("/images", halted=True)
        self.assertEqual(self.commands, ["/images"])

    def test_other_bare_commands_still_work_mid_halt(self):
        self._post("unload", halted=True)
        self.assertEqual(self.commands, ["unload"])

    def test_an_ordinary_message_is_still_a_turn(self):
        self._post("make it warmer", halted=True)
        self.assertEqual(self.commands, [])
        self.assertEqual(self.turns, ["make it warmer"])

    def test_the_route_answers_rather_than_erroring(self):
        self.assertEqual(self._post("stop", halted=True).status_code, 200)

    def test_it_asks_about_the_thread_the_message_came_from(self):
        """A halt on one conversation must not excuse `stop` on another — and
        asking about the wrong thread fails in the dangerous direction."""
        self._post("stop", halted=True, thread_id="t-other")
        self.assertEqual(self.asked, ["t-other"])


if __name__ == "__main__":
    unittest.main()
