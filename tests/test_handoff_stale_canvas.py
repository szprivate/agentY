"""A fix made on the canvas is not undone by signalling an older file.

Asked to fix the failing workflow on the canvas, the orchestrator replaced the
broken loader there (edit_canvas_graph) and then signalled the workflow FILE it
had built a turn earlier. The file is not the canvas: the old graph would have
run, and the benchmark saw the same DualCLIPLoader error three times over
(CX15, night_20261029). So a file older than the turn, signalled after the turn
changed the canvas, is handed back once.

    python -m unittest discover -s tests -t .
"""

import json
import os
import tempfile
import time
import unittest
from unittest import mock

from src.tools import workflow_handoff as handoff
from src.utils import workflow_signal


def signal(path: str) -> dict:
    fn = getattr(handoff.signal_workflow_ready, "original", None) \
        or getattr(handoff.signal_workflow_ready, "__wrapped__", None) \
        or handoff.signal_workflow_ready
    return json.loads(fn(path))


class Fixture(unittest.TestCase):

    def setUp(self) -> None:
        workflow_signal.clear_and_get()
        workflow_signal.set_execution_hold(None)
        self.addCleanup(workflow_signal.clear_and_get)
        self.addCleanup(workflow_signal.set_canvas_probe, None, 0.0)
        for p in (mock.patch.object(handoff, "_validation_errors", return_value=[]),
                  mock.patch.object(handoff, "_dead_nodes", return_value=([], []))):
            p.start()
            self.addCleanup(p.stop)
        d = tempfile.mkdtemp()
        self.path = os.path.join(d, "wf.json")
        with open(self.path, "w", encoding="utf-8") as f:
            f.write("{}")
        self.edits: list = []
        self.started = time.time()
        os.utime(self.path, (self.started - 600, self.started - 600))   # built a turn ago

    def turn(self) -> None:
        workflow_signal.set_canvas_probe(lambda: self.edits, self.started)


class AFileFromAnEarlierTurn(Fixture):

    def test_after_canvas_edits_is_handed_back_once(self):
        self.turn()
        self.edits.append("deleted node(s) 42")
        res = signal(self.path)
        self.assertEqual(res["status"], "not_ready")
        self.assertEqual(res["error_kind"], "file_predates_canvas_edits")
        self.assertEqual(res["canvas_edits"], ["deleted node(s) 42"])
        self.assertEqual(workflow_signal.peek(), [])
        self.assertEqual(signal(self.path)["status"], "ready")         # once only

    def test_without_canvas_edits_goes_through(self):
        self.turn()
        self.assertEqual(signal(self.path)["status"], "ready")

    def test_without_a_probe_goes_through(self):
        self.assertEqual(signal(self.path)["status"], "ready")


class AFileWrittenThisTurn(Fixture):

    def test_goes_through_beside_canvas_edits(self):
        self.turn()
        self.edits.append("rewired / added nodes")
        os.utime(self.path, (self.started + 1, self.started + 1))
        self.assertEqual(signal(self.path)["status"], "ready")


if __name__ == "__main__":
    unittest.main()
