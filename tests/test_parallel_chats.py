"""Several conversations at once, each on a pipeline of its own.

A host used to run one turn at a time: its pipeline holds a turn's state on
itself and its agents refuse a second call. Running conversations side by side
needs two things, pinned here:

* a pool that hands each running conversation its own pipeline (and never two
  to one conversation), building more when needed and making a turn wait when
  every one is taken;
* per-turn buffers. Canvas patches, tool activity, progress, the workflows a turn
  hands the executor, its dry-run flag, its execution errors, the prompts it
  queued in ComfyUI — all of these lived in module variables, so two turns would
  have read, drained and stopped each other's.

    python -m unittest discover -s tests
"""

import asyncio
import json
import threading
import time
import unittest
from types import SimpleNamespace
from unittest import mock

from agenty_core.utils import turn_scope
from src.utils.pipeline_pool import ConversationBusy, PipelinePool, PoolTimeout


def in_scope(scope, fn, *args):
    tok = turn_scope.enter(scope)
    try:
        return fn(*args)
    finally:
        turn_scope.leave(tok)


class Pool(unittest.TestCase):

    def setUp(self):
        self.built = []

        def factory():
            p = SimpleNamespace(n=len(self.built) + 2)
            self.built.append(p)
            return p
        self.first = SimpleNamespace(n=1)
        self.pool = PipelinePool(self.first, factory=factory, max_size=3)

    def test_two_conversations_get_two_pipelines(self):
        a = self.pool.acquire("chat-a")
        b = self.pool.acquire("chat-b")
        self.assertIsNot(a, b)
        self.assertEqual(sorted(self.pool.running_threads()), ["chat-a", "chat-b"])

    def test_a_conversation_never_gets_a_second_one(self):
        self.pool.acquire("chat-a")
        with self.assertRaises(ConversationBusy):
            self.pool.acquire("chat-a")

    def test_a_conversation_goes_back_to_the_pipeline_that_knows_it(self):
        a = self.pool.acquire("chat-a")
        b = self.pool.acquire("chat-b")
        self.pool.release(a)
        self.pool.release(b)
        self.assertIs(self.pool.acquire("chat-b"), b)
        self.assertIs(self.pool.acquire("chat-a"), a)

    def test_when_every_slot_is_taken_a_turn_waits_for_one(self):
        held = [self.pool.acquire(f"chat-{i}") for i in range(3)]
        told = []
        with self.assertRaises(PoolTimeout):
            self.pool.acquire("chat-x", timeout=0.2, on_wait=lambda: told.append(1))
        self.assertEqual(told, [1])
        threading.Timer(0.1, self.pool.release, args=(held[1],)).start()
        self.assertIs(self.pool.acquire("chat-x", timeout=2), held[1])

    def test_a_failed_build_frees_its_slot(self):
        pool = PipelinePool(self.first, factory=lambda: (_ for _ in ()).throw(RuntimeError("x")),
                            max_size=2)
        pool.acquire("chat-a")
        with self.assertRaises(RuntimeError):
            pool.acquire("chat-b")
        self.assertEqual(pool._building, 0)


class TurnBuffers(unittest.TestCase):
    """What one turn pushes, only that turn reads."""

    def setUp(self):
        self.a = turn_scope.Scope("req-a", "chat-a")
        self.b = turn_scope.Scope("req-b", "chat-b")

    def test_canvas_patches_tool_activity_and_progress(self):
        from agenty_core.utils import progress_signal
        from src.utils import canvas_patch, tool_activity
        in_scope(self.a, canvas_patch.push, {"node_id": "5", "params": {"steps": 30}})
        in_scope(self.a, tool_activity.push, {"name": "edit"})
        in_scope(self.a, progress_signal.push, "⬇️ 50%")
        self.assertEqual(in_scope(self.b, canvas_patch.drain), [])
        self.assertEqual(in_scope(self.b, tool_activity.drain), [])
        self.assertEqual(in_scope(self.b, progress_signal.drain), [])
        # A reader in another thread (the SSE stream) names the scope it reads for.
        self.assertEqual(canvas_patch.drain(self.a), [{"node_id": "5", "params": {"steps": 30}}])
        self.assertEqual(in_scope(self.a, tool_activity.drain), [{"name": "edit"}])
        self.assertEqual(progress_signal.drain(self.a), ["⬇️ 50%"])

    def test_one_conversations_dry_run_is_not_anothers(self):
        from src.utils import dry_run
        in_scope(self.a, dry_run.arm, True)
        self.assertTrue(in_scope(self.a, dry_run.active))
        self.assertFalse(in_scope(self.b, dry_run.active))

    def test_workflows_handed_to_the_executor_stay_in_their_turn(self):
        from src.utils import workflow_signal
        in_scope(self.a, workflow_signal.append_workflow_path, "a.json")
        in_scope(self.b, workflow_signal.append_workflow_path, "b.json")
        self.assertEqual(in_scope(self.a, workflow_signal.clear_and_get), ["a.json"])
        self.assertEqual(in_scope(self.b, workflow_signal.clear_and_get), ["b.json"])

    def test_execution_errors_and_output_roles(self):
        from src import executor
        from src.utils import output_tags
        in_scope(self.a, executor._record_exec_error, {}, "a.json", "boom")
        self.assertEqual(in_scope(self.b, executor.get_and_clear_exec_errors), [])
        self.assertEqual(len(in_scope(self.a, executor.get_and_clear_exec_errors)), 1)
        in_scope(self.a, output_tags.set_run_role, "hero portrait")
        in_scope(self.b, output_tags.clear)              # b's turn starting
        self.assertEqual(in_scope(self.a, lambda: output_tags._st().run_role), "hero portrait")

    def test_the_patch_failure_allowance_is_per_turn(self):
        from agenty_core.tools import comfyui
        in_scope(self.a, lambda: setattr(comfyui._patch_guard(), "count", 2))
        self.assertEqual(in_scope(self.b, lambda: comfyui._patch_guard().count), 0)

    def test_context_reaches_a_tool_run_in_a_worker_thread(self):
        """Strands runs a sync tool through asyncio.to_thread, which copies context."""
        from src.utils import canvas_patch

        async def turn():
            await asyncio.to_thread(canvas_patch.push, {"node_id": "9"})
        in_scope(self.a, lambda: asyncio.run(turn()))
        self.assertEqual(canvas_patch.drain(self.a), [{"node_id": "9"}])

    def test_two_turns_running_at_once(self):
        from src.utils import canvas_patch
        got = {}

        def run(scope, n):
            def body():
                for i in range(200):
                    canvas_patch.push({"from": scope.thread_id, "i": i})
                    if i % 50 == 0:
                        time.sleep(0.001)
                got[scope.thread_id] = canvas_patch.drain()
            in_scope(scope, body)
        threads = [threading.Thread(target=run, args=(s, n)) for n, s in enumerate((self.a, self.b))]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertTrue(all(e["from"] == "chat-a" for e in got["chat-a"]))
        self.assertTrue(all(e["from"] == "chat-b" for e in got["chat-b"]))
        self.assertEqual(len(got["chat-a"]), 200)


class StopOnlyStopsItsOwn(unittest.TestCase):

    def setUp(self):
        from agenty_core import queue_ledger
        self.ledger = queue_ledger
        queue_ledger.clear()
        self.addCleanup(queue_ledger.clear)
        in_scope(turn_scope.Scope("req-a", "chat-a"), queue_ledger.remember, "pa1")
        in_scope(turn_scope.Scope("req-a", "chat-a"), queue_ledger.remember, "pa2")
        in_scope(turn_scope.Scope("req-b", "chat-b"), queue_ledger.remember, "pb1")

    def client(self, running, pending):
        posted = []
        c = SimpleNamespace(
            get=lambda path: {"queue_running": [[0, pid] for pid in running],
                              "queue_pending": [[0, pid] for pid in pending]},
            post=lambda path, json_data=None: posted.append(json_data))
        return c, posted

    def test_a_conversations_stop_removes_only_its_prompts(self):
        c, posted = self.client(running=["pb1"], pending=["pa1", "pa2", "user-1"])
        out = self.ledger.cancel_ours(c, owner="chat-a")
        self.assertEqual(posted, [{"delete": ["pa1", "pa2"]}])
        self.assertFalse(out["running_is_ours"], "chat-b's render must not be interrupted")
        self.assertTrue(self.ledger.is_ours("pb1"))

    def test_without_an_owner_every_agent_prompt_goes_as_before(self):
        c, posted = self.client(running=[], pending=["pa1", "pb1", "user-1"])
        self.ledger.cancel_ours(c)
        self.assertEqual(posted, [{"delete": ["pa1", "pb1"]}])

    def test_the_server_stops_by_conversation(self):
        from src.utils import agentY_server as srv
        with mock.patch("agenty_core.queue_ledger.cancel_ours",
                        return_value={"ok": True, "running": ["pb1"], "running_is_ours": False}) as cancel, \
                mock.patch("agenty_core.utils.comfyui_client.get_client") as client:
            report = srv._interrupt_comfy(owner="chat-a")
        self.assertEqual(cancel.call_args.kwargs["owner"], "chat-a")
        client.return_value.post.assert_not_called()
        self.assertFalse(report["interrupted_running"])


class OneConversationEditsTheCanvas(unittest.TestCase):
    """The open canvas is shared; the first conversation to change it holds it."""

    def setUp(self):
        from src.utils import canvas_lease
        self.lease = canvas_lease
        canvas_lease.release(canvas_lease.holder())
        self.addCleanup(lambda: canvas_lease.release(canvas_lease.holder()))
        canvas_lease.set_alive_check(lambda: ["chat-a", "chat-b"])
        self.addCleanup(canvas_lease.set_alive_check, None)

    def edit(self, scope):
        from pipeline_stub import pipeline_stub, tools
        from src.utils.canvas_patch import clear
        graph = {"5": {"class_type": "KSampler", "inputs": {"steps": 20}}}
        pipe = pipeline_stub(_canvas_graph=graph, _canvas_selection=[])
        schema = {"input": {"required": {"steps": ["INT", {"default": 20, "min": 1, "max": 100}]}}}
        with mock.patch("src.utils.canvas_view.full_graph_visible", return_value=True),                 mock.patch("src.utils.preflight._schema", return_value=schema):
            out = in_scope(scope, lambda: asyncio.run(tools(pipe)["set_canvas_node_params"](
                node_id="5", params={"steps": 30})))
        in_scope(scope, clear)
        return json.loads(out)

    def test_a_second_conversation_is_told_to_leave_it_alone(self):
        a, b = turn_scope.Scope("ra", "chat-a"), turn_scope.Scope("rb", "chat-b")
        self.assertEqual(self.edit(a)["status"], "applied")
        refused = self.edit(b)
        self.assertEqual(refused["status"], "canvas_in_use")
        self.assertIn("prepare_workflow", refused["what_to_do"])
        self.assertEqual(self.edit(a)["status"], "applied", "the holder keeps editing")
        self.lease.release("chat-a")                     # a's turn ended
        self.assertEqual(self.edit(b)["status"], "applied")

    def test_a_holder_that_is_no_longer_running_does_not_keep_it(self):
        self.lease.claim("chat-gone")
        self.assertEqual(self.lease.claim("chat-b"), "")

    def test_outside_a_conversation_there_is_no_lease(self):
        self.lease.claim("chat-a")
        self.assertEqual(self.lease.claim(""), "")


class ServerTurns(unittest.TestCase):

    def setUp(self):
        from src.utils import agentY_server as srv
        self.srv = srv
        self.first = SimpleNamespace(n=1)
        self.pool = PipelinePool(self.first, factory=lambda: SimpleNamespace(n=2), max_size=1)
        for name, value in (("_pool", self.pool), ("_agent_ref", self.first), ("_run_registry", {})):
            patcher = mock.patch.object(srv, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)

    def drained(self, q):
        out = []
        while not q.empty():
            out.append(q.get())
        return out

    def test_a_turn_for_a_running_conversation_is_refused(self):
        import queue
        self.pool.acquire("chat-a")
        q, finished = queue.Queue(), {"emitted": False}
        self.assertIsNone(self.srv._take_pipeline("chat-a", "r2", q, finished))
        events = self.drained(q)
        self.assertEqual(events[0]["type"], "error")
        self.assertIsNone(events[-1])

    def test_a_waiting_turn_says_so_and_can_be_stopped(self):
        import queue
        self.pool.acquire("chat-a")                      # the only slot
        q, finished = queue.Queue(), {"emitted": False}
        result = {}
        t = threading.Thread(target=lambda: result.setdefault(
            "p", self.srv._take_pipeline("chat-b", "r2", q, finished)))
        t.start()
        time.sleep(0.3)
        self.assertTrue(self.srv._cancel_run("r2"))
        t.join(timeout=5)
        self.assertIsNone(result["p"])
        texts = [e.get("data", "") for e in self.drained(q) if isinstance(e, dict)]
        self.assertTrue(any("agents are busy" in x for x in texts))
        self.assertTrue(any("Stopped" in x for x in texts))

    def test_the_panel_hears_which_conversations_are_running(self):
        self.pool.acquire("chat-a")
        app = self.srv._build_app()
        with mock.patch.object(self.srv.cs, "list_threads",
                               return_value=[{"id": "chat-a"}, {"id": "chat-b"}]), \
                mock.patch.object(self.srv._api_guard, "check", return_value=None, create=True):
            client = app.test_client()
            health = client.get("/agentY/health").get_json()
            self.assertEqual(health["running_threads"], ["chat-a"])
            listed = client.get("/agentY/threads", headers=self._auth()).get_json()
        self.assertEqual({t["id"]: t["running"] for t in listed}, {"chat-a": True, "chat-b": False})

    def _auth(self):
        try:
            from src.utils import api_guard
            tok = api_guard.session_token(None) if hasattr(api_guard, "session_token") else ""
        except Exception:  # noqa: BLE001
            tok = ""
        return {"X-AgentY-Token": tok} if tok else {}


if __name__ == "__main__":
    unittest.main()
