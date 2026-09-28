"""Nothing is submitted while part of it will never run.

The build gate covers workflows built from scratch. This covers every other way a
workflow reaches ComfyUI — a patched template above all, which is where the real
one came from: a `VAEDecodeAudio` whose output never reached `CreateVideo.audio`
(an optional input, so validation passed) turned a model whose whole point is
synchronised audio into a silent video, and the run reported success.

So `signal_workflow_ready` looks once more before the queue. It hands the graph
back ONCE — the agent still has the turn and can wire or remove — and then lets it
through with a note: something useless in a graph must never become a render that
never happens.

    python -m unittest discover -s tests
"""

import json
import unittest
from unittest import mock

from src.tools import workflow_handoff as handoff
from src.utils import workflow_signal

DEAD = [{"node_id": "12", "class_type": "VAEDecodeAudio",
         "title": "Decode Audio Latent", "problem": "nothing reads this node's output …"}]


def signal(path: str) -> dict:
    """Call the tool the way the pipeline does, through its @tool wrapper."""
    fn = getattr(handoff.signal_workflow_ready, "original", None) \
        or getattr(handoff.signal_workflow_ready, "__wrapped__", None) \
        or handoff.signal_workflow_ready
    return json.loads(fn(path))


class Fixture(unittest.TestCase):

    def setUp(self) -> None:
        workflow_signal.clear_and_get()          # queue + refusal memory
        workflow_signal.set_execution_hold(None)
        self.addCleanup(workflow_signal.clear_and_get)
        exists = mock.patch("pathlib.Path.exists", return_value=True)
        resolve = mock.patch("pathlib.Path.resolve",
                             side_effect=lambda: __import__("pathlib").Path("wf.json"))
        for p in (exists, resolve):
            p.start()
            self.addCleanup(p.stop)

    def detector(self, *rounds):
        """Stub the shared detector; each call answers the next round."""
        answers = list(rounds)

        def check(_path):
            found = answers.pop(0) if answers else []
            return found, [d["problem"] for d in found]

        p = mock.patch("agenty_core.tools.assembly_deterministic.dead_nodes_in_file", check)
        p.start()
        self.addCleanup(p.stop)


class ACleanWorkflow(Fixture):

    def test_goes_straight_into_the_queue(self):
        self.detector([])
        res = signal("wf.json")
        self.assertEqual(res["status"], "ready")
        self.assertEqual(workflow_signal.peek(), ["wf.json"])


class AWorkflowWithDeadNodes(Fixture):

    def test_is_handed_back_the_first_time_and_not_queued(self):
        self.detector(DEAD)
        res = signal("wf.json")
        self.assertEqual(res["status"], "not_ready")
        self.assertEqual(res["dead_nodes"], DEAD)
        self.assertEqual(workflow_signal.peek(), [])          # nothing to run yet
        # It must say what to do, in the terms the fix is made in.
        self.assertIn("update_workflow", res["fix"])
        self.assertIn("signal_workflow_ready again", res["fix"])

    def test_the_second_signal_runs_anyway(self):
        # A graph carrying something useless must never become a run that never
        # happens — the note is the report, the queue still gets the workflow.
        self.detector(DEAD, DEAD)
        self.assertEqual(signal("wf.json")["status"], "not_ready")
        notes: list = []
        with mock.patch.object(handoff, "_note", notes.append):
            res = signal("wf.json")
        self.assertEqual(res["status"], "ready")
        self.assertEqual(workflow_signal.peek(), ["wf.json"])
        self.assertTrue(any("never execute" in n for n in notes), notes)
        self.assertTrue(any("VAEDecodeAudio" in n for n in notes), notes)

    def test_a_fixed_workflow_is_queued_on_the_second_signal(self):
        self.detector(DEAD, [])
        self.assertEqual(signal("wf.json")["status"], "not_ready")
        self.assertEqual(signal("wf.json")["status"], "ready")
        self.assertEqual(workflow_signal.peek(), ["wf.json"])

    def test_the_refusal_memory_lasts_exactly_one_handoff(self):
        self.detector(DEAD, DEAD, DEAD)
        self.assertEqual(signal("wf.json")["status"], "not_ready")
        self.assertEqual(signal("wf.json")["status"], "ready")
        workflow_signal.clear_and_get()            # the pipeline takes the queue
        self.assertEqual(signal("wf.json")["status"], "not_ready")


class WhenTheCheckCannotRun(Fixture):

    def test_a_detector_that_throws_never_stops_a_render(self):
        p = mock.patch("agenty_core.tools.assembly_deterministic.dead_nodes_in_file",
                       side_effect=OSError("comfyui is down"))
        p.start()
        self.addCleanup(p.stop)
        res = signal("wf.json")
        self.assertEqual(res["status"], "ready")
        self.assertEqual(workflow_signal.peek(), ["wf.json"])


class OtherRefusalsStillComeFirst(Fixture):

    def test_a_plan_hold_beats_the_dead_node_check(self):
        self.detector(DEAD)
        workflow_signal.set_execution_hold({"status": "held", "message": "approve first"})
        self.addCleanup(workflow_signal.set_execution_hold, None)
        res = signal("wf.json")
        self.assertEqual(res["status"], "held")
        # And the refusal memory was not spent on a call that never got that far.
        workflow_signal.set_execution_hold(None)
        self.assertEqual(signal("wf.json")["status"], "not_ready")

    def test_a_missing_file_is_still_its_own_error(self):
        self.detector(DEAD)
        with mock.patch("pathlib.Path.exists", return_value=False):
            res = signal("gone.json")
        self.assertIn("not found", res["error"])


if __name__ == "__main__":
    unittest.main()
