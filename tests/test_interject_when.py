"""A message sent into a running turn is told when it will be read.

The turn bus knows which tool a turn is in; /agentY/interject answers with it
(and a download's progress), so the panel can say "it reads this at its next
check" instead of leaving the agent looking deaf through a 26 GB download.
"""

import json
import queue
import unittest
from unittest import mock

from src.utils import interject_bus, turn_bus


class TheToolATurnIsIn(unittest.TestCase):

    def test_the_outermost_running_call(self):
        q = turn_bus.tee(queue.Queue(), request_id="ct1", thread_id="t")
        self.assertIsNone(turn_bus.current_tool("ct1"))
        q.put({"type": "tool", "phase": "call", "id": "a", "agent": "orchestrator",
               "name": "[orchestrator] prepare_workflow"})
        q.put({"type": "tool", "phase": "call", "id": "b", "agent": "assemble_workflow",
               "name": "[assemble_workflow] validate_workflow"})
        self.assertEqual(turn_bus.current_tool("ct1")["name"], "prepare_workflow")
        q.put({"type": "tool", "phase": "result", "id": "b"})
        q.put({"type": "tool", "phase": "result", "id": "a"})
        self.assertIsNone(turn_bus.current_tool("ct1"))
        q.put({"type": "done"})
        q.put(None)


class TheSenderIsTold(unittest.TestCase):

    def setUp(self):
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
        self.post = lambda body: client.post("/agentY/interject", json=body,
                                             headers={"X-AgentY-Token": tok} if tok else {})

    def test_during_a_download_it_says_how_far_it_got(self):
        q = turn_bus.tee(queue.Queue(), request_id="ij1", thread_id="t1")
        interject_bus.open_run("ij1", "t1")
        self.addCleanup(interject_bus.close_run, "ij1")
        q.put({"type": "tool", "phase": "call", "id": "d", "agent": "orchestrator",
               "name": "[orchestrator] download_hf_model"})
        progress = [{"job_id": "j", "file": "gemma.safetensors", "done_gb": 15.2,
                     "total_gb": 26.3, "eta_min": 14.7}]
        with mock.patch("agenty_core.tools.huggingface.download_progress", return_value=progress):
            body = self.post({"request_id": "ij1", "text": "where does it go?"}).get_json()
        self.assertTrue(body["ok"])
        self.assertEqual(body["current_step"]["name"], "download_hf_model")
        self.assertEqual(body["current_step"]["download"]["total_gb"], 26.3)
        q.put({"type": "done"})
        q.put(None)

    def test_between_steps_there_is_nothing_to_wait_for(self):
        turn_bus.tee(queue.Queue(), request_id="ij2", thread_id="t2")
        interject_bus.open_run("ij2", "t2")
        self.addCleanup(interject_bus.close_run, "ij2")
        body = self.post({"request_id": "ij2", "text": "hi"}).get_json()
        self.assertIsNone(body["current_step"])

    def test_a_finished_download_is_announced(self):
        with mock.patch.object(self.srv.status_bus, "notify") as notify:
            self.srv._download_finished({"filename": "gemma.safetensors",
                                         "result": json.dumps({"ok": True, "path": "D:/m/clip/gemma.safetensors"})})
            self.srv._download_finished({"filename": "x", "result": json.dumps({"ok": True, "skipped": True})})
        self.assertEqual(notify.call_count, 1)
        self.assertIn("Download finished: gemma.safetensors", notify.call_args.args[0])


if __name__ == "__main__":
    unittest.main()
