"""Stop stops the whole run — and only the parts of it that are ours.

Two failures met here, both reported from a real session.

The first: Stop interrupted the job that was running and ComfyUI started the next
one. A run is usually several prompts deep (a batch member per variant, a repaired
graph queued behind the original), so stopping it meant pressing Stop once per
remaining item and watching the GPU work through a queue nobody wanted any more.

The second is why the cure cannot be "clear the queue": the user queues their own
work in the same ComfyUI. A stop button that threw away somebody's overnight batch
because an agent happened to be running is not a button anyone presses twice.
"""

import unittest

from agenty_core import queue_ledger


class FakeClient:
    """Enough of the ComfyUI client to answer /queue and record the deletes."""

    def __init__(self, running=(), pending=()):
        self.running = list(running)
        self.pending = list(pending)
        self.posted = []
        self.api_key = ""

    def get(self, path):
        if path == "/queue":
            return {
                "queue_running": [[0, pid, {}, {}, []] for pid in self.running],
                "queue_pending": [[i + 1, pid, {}, {}, []]
                                  for i, pid in enumerate(self.pending)],
            }
        raise AssertionError(f"unexpected GET {path}")

    def post(self, path, json_data=None):
        self.posted.append((path, json_data or {}))
        return {}


class TheLedger(unittest.TestCase):
    def setUp(self):
        queue_ledger.clear()

    def tearDown(self):
        queue_ledger.clear()

    def test_it_remembers_what_we_submitted(self):
        queue_ledger.remember("abc")
        self.assertTrue(queue_ledger.is_ours("abc"))
        self.assertFalse(queue_ledger.is_ours("xyz"))

    def test_remember_returns_the_id_so_it_can_wrap_a_submit(self):
        self.assertEqual(queue_ledger.remember("abc"), "abc")

    def test_blank_ids_are_not_recorded(self):
        """A submission that failed returns nothing, and "" must never be a key —
        it would match every entry whose prompt id we could not read."""
        queue_ledger.remember("")
        queue_ledger.remember(None)
        self.assertEqual(queue_ledger.ours(), [])
        self.assertFalse(queue_ledger.is_ours(""))

    def test_it_does_not_grow_without_bound(self):
        for i in range(queue_ledger._MAX + 50):
            queue_ledger.remember(f"p{i}")
        self.assertLessEqual(len(queue_ledger.ours()), queue_ledger._MAX)
        self.assertTrue(queue_ledger.is_ours(f"p{queue_ledger._MAX + 49}"),
                        "the newest must survive; the oldest are the ones to drop")

    def test_reading_a_queue_entry(self):
        self.assertEqual(queue_ledger.prompt_id_of([3, "abc", {}, {}, []]), "abc")
        self.assertEqual(queue_ledger.prompt_id_of({"prompt_id": "abc"}), "abc")

    def test_an_entry_it_cannot_read_yields_no_id(self):
        """Defensive on purpose: this is the one place a shape change in ComfyUI's
        API would turn "stop the agent's jobs" into "delete something else"."""
        for junk in ([], [7], "abc", None, 12):
            with self.subTest(junk=junk):
                self.assertEqual(queue_ledger.prompt_id_of(junk), "")


class CancellingOurs(unittest.TestCase):
    def setUp(self):
        queue_ledger.clear()

    def tearDown(self):
        queue_ledger.clear()

    def test_the_users_queued_jobs_are_left_alone(self):
        """The whole reason this is a ledger and not a POST /queue {clear: true}."""
        queue_ledger.remember("agent-1")
        queue_ledger.remember("agent-2")
        client = FakeClient(running=["agent-1"],
                            pending=["agent-2", "user-1", "user-2"])
        report = queue_ledger.cancel_ours(client)
        self.assertEqual(report["deleted"], ["agent-2"])
        self.assertEqual(report["kept"], 2)
        self.assertEqual(client.posted, [("/queue", {"delete": ["agent-2"]})])

    def test_every_one_of_ours_goes_in_a_single_call(self):
        """Deleting one at a time leaves the queue advancing between requests, so
        an item can start running in the gap and be missed entirely."""
        for pid in ("a", "b", "c"):
            queue_ledger.remember(pid)
        client = FakeClient(pending=["a", "user", "b", "c"])
        queue_ledger.cancel_ours(client)
        self.assertEqual(len(client.posted), 1)
        self.assertEqual(client.posted[0][1]["delete"], ["a", "b", "c"])

    def test_nothing_of_ours_means_no_delete_at_all(self):
        client = FakeClient(pending=["user-1", "user-2"])
        report = queue_ledger.cancel_ours(client)
        self.assertEqual(report["deleted"], [])
        self.assertEqual(report["kept"], 2)
        self.assertEqual(client.posted, [], "an empty delete list is still a write")

    def test_it_says_whether_the_running_job_is_ours(self):
        """What lets the caller leave somebody else's render running."""
        queue_ledger.remember("agent-1")
        self.assertTrue(queue_ledger.cancel_ours(
            FakeClient(running=["agent-1"]))["running_is_ours"])
        self.assertFalse(queue_ledger.cancel_ours(
            FakeClient(running=["user-1"]))["running_is_ours"])

    def test_an_unreadable_queue_is_reported_as_such(self):
        """Not as "nothing of ours is queued". The caller has to be able to tell
        those apart: with no queue to read it has no basis for deciding whose job
        is running, and must fall back to interrupting."""
        class Broken(FakeClient):
            def get(self, path):
                raise OSError("ComfyUI is not running")

        report = queue_ledger.cancel_ours(Broken())
        self.assertFalse(report["ok"])
        self.assertEqual(report["deleted"], [])

    def test_a_delete_that_fails_does_not_look_like_a_success(self):
        queue_ledger.remember("agent-1")

        class Refuses(FakeClient):
            def post(self, path, json_data=None):
                raise OSError("gone")

        report = queue_ledger.cancel_ours(Refuses(pending=["agent-1"]))
        self.assertTrue(report["ok"])
        self.assertEqual(report["deleted"], [])

    def test_deleted_prompts_are_forgotten(self):
        """Prompt ids are reused by nobody, but a ledger that only grows would
        eventually evict live ids to make room for dead ones."""
        queue_ledger.remember("agent-1")
        queue_ledger.cancel_ours(FakeClient(pending=["agent-1"]))
        self.assertFalse(queue_ledger.is_ours("agent-1"))


class TheStopRoute(unittest.TestCase):
    """What the panel's Stop button actually does, in the server."""

    def setUp(self):
        from src.utils import agentY_server as srv
        self.srv = srv
        queue_ledger.clear()
        self.calls = []

        class Recording(FakeClient):
            def post(inner, path, json_data=None):     # noqa: N805
                self.calls.append(path)
                return FakeClient.post(inner, path, json_data)

        self.Recording = Recording

    def tearDown(self):
        queue_ledger.clear()

    def _run(self, client):
        from unittest import mock
        with mock.patch("agenty_core.utils.comfyui_client.get_client",
                        return_value=client):
            return self.srv._interrupt_comfy()

    def test_it_interrupts_and_clears_in_one_press(self):
        queue_ledger.remember("agent-1")
        queue_ledger.remember("agent-2")
        client = self.Recording(running=["agent-1"], pending=["agent-2"])
        report = self._run(client)
        self.assertEqual(report["deleted"], ["agent-2"])
        self.assertIn("/interrupt", self.calls)

    def test_it_does_not_interrupt_somebody_elses_render(self):
        """A stop meant for the agent must not end a job the user queued."""
        client = self.Recording(running=["user-1"], pending=["user-2"])
        report = self._run(client)
        self.assertFalse(report["interrupted_running"])
        self.assertNotIn("/interrupt", self.calls)

    def test_an_unreachable_comfyui_still_gets_an_interrupt(self):
        """In doubt, stop. The person pressed Stop, and a stop that does nothing
        is a worse failure than an interrupt nobody needed."""
        class Broken(FakeClient):
            def get(inner, path):                      # noqa: N805
                raise OSError("down")

            def post(inner, path, json_data=None):     # noqa: N805
                self.calls.append(path)
                return {}

        report = self._run(Broken())
        self.assertTrue(report["interrupted_running"])
        self.assertIn("/interrupt", self.calls)

    def test_an_empty_queue_is_still_interrupted(self):
        # Nothing queued and nothing running that we can see: harmless, and it
        # covers the gap between reading the queue and acting on it.
        report = self._run(self.Recording())
        self.assertTrue(report["interrupted_running"])


class AStoppedJobIsNotABrokenOne(unittest.TestCase):
    """The second half of the same report.

    The agent printed "Skipping (not a workflow error)" — and then repaired the
    workflow anyway. The executor's own path was right; the resume path was not.
    It handed the interrupt back to the agent as an ordinary tool result, and
    `{"interrupted": true, "error": "Execution interrupted"}` reads to an agent
    exactly like a workflow that failed. So it fixed it, and re-queued the graph
    the user had just cancelled.
    """

    def note(self, result):
        from src.pipeline import interrupted_turn_note
        return interrupted_turn_note(result)

    def test_an_interrupt_ends_the_turn_with_a_reason(self):
        line = self.note({"interrupted": True, "error": "Execution interrupted"})
        self.assertTrue(line)
        self.assertIn("nothing to repair", line)

    def test_it_says_what_to_do_next(self):
        """An agent that just goes quiet after a stop looks broken."""
        self.assertIn("run it again",
                      self.note({"interrupted": True}))

    def test_a_real_failure_is_not_treated_as_a_stop(self):
        """The distinction has to cut both ways: an execution error still has to
        reach the repair path, or a genuinely broken graph is never fixed."""
        for result in ({"error": "ComfyUI execution failed"},
                       {"error": "boom", "details": {"node_id": "7"}},
                       {"interrupted": False, "error": "x"}):
            with self.subTest(result=result):
                self.assertEqual(self.note(result), "")

    def test_a_normal_history_is_not_a_stop(self):
        self.assertEqual(self.note({"7": {"outputs": {}}}), "")

    def test_junk_is_not_a_stop(self):
        for junk in (None, "", [], "interrupted", 0):
            with self.subTest(junk=junk):
                self.assertEqual(self.note(junk), "")

    def test_the_resume_path_checks_before_it_resumes(self):
        """Order is the whole fix: the guard has to come BEFORE the tool result
        is handed back, or the agent has already been told the workflow failed."""
        from pathlib import Path
        source = (Path(__file__).resolve().parents[1] / "src" / "pipeline.py"
                  ).read_text(encoding="utf-8")
        guard = source.rindex("interrupted_turn_note(")
        resume = source.index('"interruptResponse"')
        self.assertLess(guard, resume,
                        "the interrupt guard must precede the resume it prevents")


if __name__ == "__main__":
    unittest.main()
