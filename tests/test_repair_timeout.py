"""A repair that runs out of time is a verdict, not something to try again.

The executor retries a heal that comes back ``failed`` — same file, same prompt,
a brand-new agent with its messages cleared. A specialist that could not
converge in its window does not converge in the next one either; it just
replays the same exploration from zero. On 2026-10-18 two variants timed out at
120s, were retried from a blank slate, and burned four minutes and ~890k input
tokens each before the user pressed stop.

So a timeout comes back ``rejected`` — the status ``content_policy.exhausted``
already uses for "trying again cannot help", which the executor retires without
a retry. Any other failure stays ``failed``, because a transient error is worth
another go.
"""
import asyncio
import unittest
from unittest import mock

EXEC_ERROR = {"error": "ComfyUI execution failed",
              "details": {"node_id": "15", "node_type": "OpenAIGPTImageNodeV2",
                          "exception_type": "ValueError",
                          "exception_message": "bad size"}}


async def _repair(behaviour, timeout=0.05):
    """Run one repair turn against an agent whose ``invoke_async`` misbehaves."""
    from src.pipeline import Pipeline

    agent = mock.Mock()
    agent.messages = []
    agent.invoke_async = behaviour

    me = mock.Mock()
    me._FIX_ASSEMBLY_TIMEOUT = timeout
    me._limit_violations.return_value = None

    return await Pipeline._run_fix_workflow_assembly(
        me, "/tmp/wf.json", exec_error=EXEC_ERROR, agent=agent)


class TheRepairTimeoutIsTerminalTest(unittest.IsolatedAsyncioTestCase):

    async def test_a_timed_out_repair_is_rejected_not_failed(self):
        async def never_converges(prompt):
            await asyncio.sleep(5)

        res = await _repair(never_converges)
        self.assertEqual(res["status"], "rejected")

    async def test_the_reason_tells_the_caller_retrying_cannot_help(self):
        """A bare "timed out" invites another attempt; this says not to."""
        async def never_converges(prompt):
            await asyncio.sleep(5)

        res = await _repair(never_converges)
        self.assertIn("without converging", res["error"])
        self.assertIn("would not differ", res["error"])

    async def test_the_workflow_it_repaired_is_still_named(self):
        async def never_converges(prompt):
            await asyncio.sleep(5)

        res = await _repair(never_converges)
        self.assertEqual(res["workflow_path"], "/tmp/wf.json")


class OnlyATimeoutIsAVerdictTest(unittest.TestCase):
    """Anything else stays retryable — the point is not to stop healing."""

    def test_another_kind_of_failure_is_still_failed(self):
        async def explodes(prompt):
            raise RuntimeError("connection reset by peer")

        res = asyncio.run(_repair(explodes))
        self.assertEqual(res["status"], "failed")
        self.assertIn("connection reset", res["error"])


if __name__ == "__main__":
    unittest.main()
