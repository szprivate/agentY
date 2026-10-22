"""The turn API: another program asks agentY to run a turn and gets one answer.

Two layers. ``summarize`` is pure and is checked against hand-written events. The
routes are checked through Flask's test client with the pipeline replaced by a
stub that puts events on the real turn bus, so what is tested is the wiring: the
guard (token), the start, the long-poll, the busy refusal, and that the turn is
labelled so the panel can follow it.
"""
import time
import unittest

from src.utils import turn_api, turn_bus

try:
    import flask  # noqa: F401
    HAVE_FLASK = True
except ImportError:  # pragma: no cover
    HAVE_FLASK = False


def ev(_t, **kw):
    return {"type": _t, **kw}


class Summarize(unittest.TestCase):
    def test_a_finished_turn(self):
        r = turn_api.summarize([
            ev("text", data="Done. "), ev("text", data="One image."),
            ev("tool", phase="call", id="a", agent="orchestrator", name="[orchestrator] prepare_workflow", input={"x": 1}),
            ev("tool", phase="result", id="a", agent="orchestrator", name="[orchestrator] prepare_workflow", result="ok: ready"),
            ev("tool", phase="call", id="b", agent="orchestrator", name="[orchestrator] run", input="q"),
            ev("tool", phase="result", id="b", agent="orchestrator", name="[orchestrator] run", result="error: boom"),
            ev("output", kind="image", path="C:/out/a.png", name="a.png", role="result"),
            ev("output", kind="image", path="C:/out/a.png", name="a.png"),
            ev("canvas_patch", op="place_text", node_id=7, text="x" * 500),
            ev("canvas_patch", op="flow_state", node_id=7, state="running"),
            ev("usage", input=10, output=5, cost=0.01, calls=2),
            ev("done"),
        ], request_id="r", thread_id="t", started=1.0, finished=4.0)
        self.assertEqual(r["status"], "done")
        self.assertEqual(r["reply"], "Done. One image.")
        self.assertEqual([(t["name"], t["status"]) for t in r["tool_calls"]],
                         [("prepare_workflow", "ok"), ("run", "error")])
        self.assertEqual(len(r["files"]), 1, "the same file is listed once")
        self.assertEqual([c["op"] for c in r["canvas"]], ["place_text"], "bookkeeping patches are left out")
        self.assertEqual(r["usage"]["cost"], 0.01)
        self.assertEqual(r["seconds"], 3.0)
        self.assertIsNone(r["waiting"])

    def test_running_until_done(self):
        self.assertEqual(turn_api.summarize([ev("text", data="hi")])["status"], "running")

    def test_an_open_question_is_waiting(self):
        events = [ev("ask", request_id="r9", kind="approval", prompt="Approve?")]
        r = turn_api.summarize(events, request_id="r", asking=True)
        self.assertEqual(r["status"], "waiting")
        self.assertEqual(r["waiting"]["kind"], "approval")
        self.assertEqual(r["waiting"]["request_id"], "r9")

    def test_an_answered_question_is_not_waiting(self):
        r = turn_api.summarize([ev("ask", kind="approval", prompt="?"), ev("text", data="ok")], asking=False)
        self.assertEqual(r["status"], "running")

    def test_an_open_review_is_waiting_even_after_the_turn_ended(self):
        r = turn_api.summarize([ev("canvas_patch", op="review_collector", node_id=3), ev("done")])
        self.assertEqual(r["status"], "waiting")
        self.assertEqual(r["waiting"]["kind"], "review")
        r = turn_api.summarize([ev("canvas_patch", op="review_collector", node_id=3),
                                ev("canvas_patch", op="review_released"), ev("done")])
        self.assertEqual(r["status"], "done")

    def test_error_without_a_reply_is_an_error(self):
        r = turn_api.summarize([ev("error", message="nope"), ev("done")])
        self.assertEqual(r["status"], "error")
        self.assertEqual(r["errors"], ["nope"])


@unittest.skipUnless(HAVE_FLASK, "Flask is not installed")
class Routes(unittest.TestCase):
    HOST = {"Host": "127.0.0.1:5001"}

    @classmethod
    def setUpClass(cls):
        from src.utils import agentY_server as srv, api_guard
        cls.srv, cls.guard = srv, api_guard
        srv.set_bound_port(5001)
        api_guard._token = "test-token"
        cls.app = srv._build_app()
        cls.app.config["TESTING"] = True

    @classmethod
    def tearDownClass(cls):
        cls.guard._token = None

    def setUp(self):
        self.started = []
        srv = self.srv
        self._real = (srv._run_pipeline_stream, srv.cs, srv.running_threads)
        outer = self

        def fake_stream(thread_id, message, image_paths, q, rid, **kw):
            outer.started.append((thread_id, message, list(image_paths), kw.get("origin")))
            q = turn_bus.tee(q, request_id=rid, thread_id=thread_id,
                             origin=kw.get("origin", "panel"), text=message)
            q.put({"type": "text", "data": "pong"})
            q.put({"type": "output", "kind": "image", "path": "C:/x.png", "name": "x.png"})
            q.put({"type": "done"})
            q.put(None)

        class FakeCS:
            threads = {"t1": {"id": "t1"}}

            def get_thread(self, tid):
                return self.threads.get(tid)

            def create_thread(self, title="New chat", **kw):
                self.threads["new"] = {"id": "new", "title": title}
                return "new"

            def add_message(self, *a, **kw):
                pass

        srv._run_pipeline_stream = fake_stream
        srv.cs = FakeCS()
        srv.running_threads = lambda: []

    def tearDown(self):
        self.srv._run_pipeline_stream, self.srv.cs, self.srv.running_threads = self._real

    def post(self, body, token="test-token", **headers):
        h = dict(self.HOST, **headers)
        if token:
            h[self.guard.TOKEN_HEADER] = token
        with self.app.test_client() as c:
            return c.post("/agentY/turns", json=body, headers=h)

    def get(self, path, token="test-token", **headers):
        h = dict(self.HOST, **headers)
        if token:
            h[self.guard.TOKEN_HEADER] = token
        with self.app.test_client() as c:
            return c.get(path, headers=h)

    def test_refused_without_the_token(self):
        self.assertEqual(self.post({"message": "hi"}, token="").status_code, 403)
        self.assertEqual(self.post({"message": "hi"}, token="wrong").status_code, 403)
        self.assertEqual(self.get("/agentY/turns/x", token="").status_code, 403)
        self.assertEqual(self.started, [])

    def test_refused_from_a_foreign_browser_origin_even_with_the_token(self):
        r = self.post({"message": "hi"}, Origin="https://evil.example")
        self.assertEqual(r.status_code, 403)
        self.assertEqual(self.started, [])

    def test_start_then_read_the_result(self):
        r = self.post({"message": "ping"})
        self.assertEqual(r.status_code, 202)
        body = r.get_json()
        self.assertEqual(body["conversation"], "new")
        res = self.get(f"/agentY/turns/{body['turn_id']}?wait=5").get_json()
        self.assertEqual(res["status"], "done")
        self.assertEqual(res["reply"], "pong")
        self.assertEqual(res["files"][0]["path"], "C:/x.png")
        self.assertEqual(res["origin"], "api", "labelled, so the panel can tell whose it is")
        self.assertEqual(self.started[0][3], "api")

    def test_the_finished_summary_outlives_the_bus(self):
        rid = self.post({"message": "ping"}).get_json()["turn_id"]
        turn_bus.wait(rid, 5)
        time.sleep(0.05)
        turn_bus._logs.pop(rid, None)
        turn_bus._active.pop(rid, None)
        res = self.get(f"/agentY/turns/{rid}").get_json()
        self.assertEqual(res["reply"], "pong")

    def test_an_existing_conversation_is_addressed(self):
        r = self.post({"message": "again", "conversation": "t1"})
        self.assertEqual(r.get_json()["conversation"], "t1")

    def test_unknown_conversation_busy_conversation_and_bad_input(self):
        self.assertEqual(self.post({"message": "x", "conversation": "nope"}).status_code, 400)
        self.assertEqual(self.post({"message": "  "}).status_code, 400)
        self.assertEqual(self.post({"message": "x", "attachments": ["C:/definitely/missing.png"]}).status_code, 400)
        self.assertEqual(self.post({"message": "x", "attachments": "no"}).status_code, 400)
        self.srv.running_threads = lambda: ["t1"]
        self.assertEqual(self.post({"message": "x", "conversation": "t1"}).status_code, 409)

    def test_unknown_turn(self):
        self.assertEqual(self.get("/agentY/turns/doesnotexist").status_code, 404)

    def test_the_list(self):
        self.post({"message": "ping"})
        rows = self.get("/agentY/turns").get_json()["turns"]
        self.assertTrue(rows)


if __name__ == "__main__":
    unittest.main()
