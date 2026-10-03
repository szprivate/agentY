"""A signalled workflow can be taken back before the turn ends, and `queue` shows it.

A signal is submitted only when the turn ends, so ComfyUI's queue was empty when
the agent looked, it told the user nothing would run — and the workflow ran.
"""

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from src.tools import workflow_handoff as wh
from src.utils import workflow_signal as ws


class TakingASignalBack(unittest.TestCase):

    def setUp(self):
        ws.clear_and_get()
        self.addCleanup(ws.clear_and_get)
        d = tempfile.mkdtemp()
        self.a = Path(d, "a.json")
        self.b = Path(d, "b.json")
        for p in (self.a, self.b):
            p.write_text("{}", encoding="utf-8")
        for p in (self.a, self.b):
            ws.append_workflow_path(str(p.resolve()))

    def test_one_is_withdrawn(self):
        out = json.loads(wh.withdraw_workflow._tool_func(str(self.a)))
        self.assertTrue(out["ok"])
        self.assertEqual(out["withdrawn"], [str(self.a.resolve())])
        self.assertEqual(ws.peek(), [str(self.b.resolve())])

    def test_blank_withdraws_all(self):
        out = json.loads(wh.withdraw_workflow._tool_func(""))
        self.assertEqual(len(out["withdrawn"]), 2)
        self.assertEqual(ws.peek(), [])
        self.assertFalse(json.loads(wh.withdraw_workflow._tool_func(""))["ok"])

    def test_the_queue_answer_names_what_is_waiting(self):
        with mock.patch("agenty_core.tools.comfyui.queue",
                        return_value=json.dumps({"queue_running": [], "queue_pending": []})):
            out = json.loads(wh.queue._tool_func("status"))
        self.assertEqual(out["queue_pending"], [])
        self.assertEqual(len(out["signalled_this_turn"]), 2)
        self.assertIn("withdraw_workflow", out["note"])
        ws.clear_and_get()
        with mock.patch("agenty_core.tools.comfyui.queue", return_value='{"queue_running": []}'):
            self.assertNotIn("signalled_this_turn", json.loads(wh.queue._tool_func("status")))

    def test_the_orchestrator_has_both(self):
        from src.tools import ORCHESTRATOR_TOOLS
        names = {getattr(t, "tool_name", None) for t in ORCHESTRATOR_TOOLS}
        self.assertIn("withdraw_workflow", names)
        self.assertIn("queue", names)
        from src.tools import queue
        self.assertIs(queue, wh.queue)


if __name__ == "__main__":
    unittest.main()
