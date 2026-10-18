"""A cancelled run is not a broken workflow, and must not be repaired.

Seen on 2026-10-18: a two-variant canvas sweep was globally interrupted about a
second after submission, while the API node's call was in flight. ComfyUI
reported it as ``ProcessingInterrupted: Task cancelled`` — an ordinary
``execution_error``, because a node that is actively working raises rather than
emitting ``execution_interrupted``. The monitor flags the interruptions it
witnesses, but a member can reach the heal decision by a path that never got the
chance, and then the repair specialist is sent after a graph that ran fine until
somebody stopped it. It made 109 tool calls, 67 of them ``get_node_source``, and
one ``update_workflow``, until the user pressed stop.
"""
import unittest

from src import executor


def _failure(exc_type: str, message: str, **extra) -> dict:
    """What the monitor hands back for a node that raised *exc_type*."""
    return {"error": "ComfyUI execution failed",
            "details": {"node_id": "15", "node_type": "OpenAIGPTImageNodeV2",
                        "exception_type": exc_type, "exception_message": message,
                        "traceback": []},
            **extra}


class FlaggedByTheMonitorTest(unittest.TestCase):
    """The flag the progress monitor sets still settles it on its own."""

    def test_the_flag_alone_is_enough(self):
        self.assertTrue(executor.is_interruption(
            {"interrupted": True, "error": "Execution interrupted"}))

    def test_the_flag_wins_over_details_that_look_like_a_defect(self):
        self.assertTrue(executor.is_interruption(
            {"interrupted": True, "details": {"exception_type": "ValueError"}}))


class CarriedOnlyInTheDetailsTest(unittest.TestCase):
    """The cases the flag missed — the exception is all the evidence there is."""

    def test_an_api_node_cancelled_mid_call(self):
        """GPT Image, Kling and friends raise this instead of emitting an event."""
        self.assertTrue(executor.is_interruption(
            _failure("ProcessingInterrupted", "Task cancelled")))

    def test_a_core_node_interrupted(self):
        self.assertTrue(executor.is_interruption(
            _failure("InterruptProcessingException", "")))

    def test_the_message_alone_is_enough(self):
        """An exception type that names nothing still says 'task cancelled'."""
        self.assertTrue(executor.is_interruption(
            _failure("RuntimeError", "Task cancelled")))


class ARealDefectIsStillRepairedTest(unittest.TestCase):
    """The point of the check is to skip cancellations, not to stop healing."""

    def test_a_node_that_genuinely_failed(self):
        self.assertFalse(executor.is_interruption(
            _failure("ValueError", "size must be one of the allowed options")))

    def test_a_missing_model(self):
        self.assertFalse(executor.is_interruption(
            _failure("FileNotFoundError", "model not found")))

    def test_a_failure_with_no_details_at_all(self):
        """Nothing to classify means nothing reclassified — heal as before."""
        self.assertFalse(executor.is_interruption(
            {"details": {}, "error": "stream ended without a result"}))

    def test_a_timeout_is_not_a_cancellation(self):
        self.assertFalse(executor.is_interruption(
            {"error": "ComfyUI job did not finish within 600s"}))


class JunkIsSurvivableTest(unittest.TestCase):
    def test_nothing_at_all(self):
        self.assertFalse(executor.is_interruption(None))

    def test_not_a_dict(self):
        self.assertFalse(executor.is_interruption("interrupted"))

    def test_details_that_are_not_a_dict(self):
        self.assertFalse(executor.is_interruption({"details": "nonsense"}))


if __name__ == "__main__":
    unittest.main()
