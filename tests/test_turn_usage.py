"""The panel's live usage line: what a turn has spent so far.

While a turn ran, the panel said nothing about its cost - token counts went to
the terminal and, afterwards, to the usage viewer. Every agent's token hook now
reports to the turn it runs in, and the stream sends a `usage` event whenever
the totals move.
"""

import unittest
from types import SimpleNamespace
from unittest import mock

from agenty_core.utils import turn_scope

from src import agent as agent_mod
from src.utils import turn_usage


class _InTurn(unittest.TestCase):

    def setUp(self):
        self.scope = turn_scope.Scope("req-1", "thread-1")
        token = turn_scope.enter(self.scope)
        self.addCleanup(turn_scope.leave, token)


class Counting(_InTurn):

    def test_nothing_is_sent_before_anything_was_used(self):
        self.assertIsNone(turn_usage.take_changed())

    def test_it_adds_up_and_is_sent_once_per_change(self):
        turn_usage.add(1000, 50, 600, 0, 0.01)
        turn_usage.add(2000, 150, 1400, 100, 0.02)
        got = turn_usage.take_changed()
        self.assertEqual(got, {"input": 3000, "output": 200, "cache_read": 2000, "cache_write": 100,
                               "cache_hit": 0.667, "cost": 0.03, "calls": 2})
        self.assertIsNone(turn_usage.take_changed(), "unchanged: not sent again")
        turn_usage.add(10, 1)
        self.assertEqual(turn_usage.take_changed()["input"], 3010)

    def test_a_model_without_a_price_has_no_cost_rather_than_a_free_one(self):
        turn_usage.add(500, 20, d_cost=None)
        self.assertIsNone(turn_usage.take_changed()["cost"])
        turn_usage.add(500, 20, d_cost=0.004)
        self.assertEqual(turn_usage.take_changed()["cost"], 0.004)

    def test_a_counter_that_went_backwards_never_subtracts(self):
        turn_usage.add(1000, 100, d_cost=0.01)
        turn_usage.add(-400, -10, -5, -5, -1.0)
        got = turn_usage.take_changed()
        self.assertEqual((got["input"], got["output"], got["cost"]), (1000, 100, 0.01))

    def test_each_turn_counts_its_own(self):
        turn_usage.add(1000, 100)
        other = turn_scope.Scope("req-2", "thread-2")
        token = turn_scope.enter(other)
        try:
            self.assertIsNone(turn_usage.take_changed(), "another conversation starts at nothing")
            turn_usage.add(7, 1)
            self.assertEqual(turn_usage.take_changed()["input"], 7)
        finally:
            turn_scope.leave(token)
        self.assertEqual(turn_usage.take_changed()["input"], 1000)

    def test_the_stream_reads_a_turn_it_is_not_running_in(self):
        turn_usage.add(42, 1)
        token = turn_scope.enter(turn_scope.Scope("elsewhere"))
        try:
            self.assertEqual(turn_usage.take_changed(self.scope)["input"], 42)
        finally:
            turn_scope.leave(token)


def _agent(inp, out, cr=0, cw=0):
    usage = {"inputTokens": inp, "outputTokens": out, "cacheReadInputTokens": cr, "cacheWriteInputTokens": cw}
    return SimpleNamespace(event_loop_metrics=SimpleNamespace(accumulated_usage=usage))


class TheHookReports(_InTurn):

    def setUp(self):
        super().setUp()
        self.hook = agent_mod.TokenUsageHookProvider(role="orchestrator")
        # $1 per 1000 input tokens, so the arithmetic is readable
        patcher = mock.patch.object(agent_mod, "compute_cost_from_usage",
                                    lambda usage, agent: (usage["inputTokens"] / 1000.0, 0))
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_only_what_is_new_since_the_last_report(self):
        self.hook._report_live(_agent(1000, 50, 400), calls=1)
        self.hook._report_live(_agent(1000, 50, 400))            # nothing new: nothing added
        self.hook._report_live(_agent(3000, 90, 1400), calls=1)
        got = turn_usage.take_changed()
        self.assertEqual((got["input"], got["output"], got["cache_read"], got["calls"]), (3000, 90, 1400, 2))
        self.assertEqual(got["cost"], 3.0)

    def test_an_agent_whose_counter_started_over_is_counted_from_zero(self):
        """A pooled agent's totals reset between turns; its first report of a
        new turn is all new, not a negative difference."""
        self.hook._report_live(_agent(5000, 300), calls=1)
        turn_usage.take_changed()
        self.hook._report_live(_agent(800, 40), calls=1)
        got = turn_usage.take_changed()
        self.assertEqual((got["input"], got["output"]), (5800, 340))
        self.assertEqual(got["cost"], 5.8)

    def test_two_agents_in_one_turn_are_one_total(self):
        specialist = agent_mod.TokenUsageHookProvider(role="vision_agent")
        self.hook._report_live(_agent(1000, 10), calls=1)
        specialist._report_live(_agent(400, 5), calls=1)
        got = turn_usage.take_changed()
        self.assertEqual((got["input"], got["calls"]), (1400, 2))

    def test_a_broken_agent_object_never_breaks_the_turn(self):
        self.hook._report_live(SimpleNamespace())       # no metrics at all
        self.assertIsNone(turn_usage.take_changed())

    def test_it_is_registered_where_the_totals_can_have_changed(self):
        from strands.hooks.events import AfterInvocationEvent, AfterToolCallEvent, BeforeModelCallEvent
        seen = []
        registry = SimpleNamespace(add_callback=lambda event, fn: seen.append(event))
        self.hook.register_hooks(registry)
        for event in (AfterToolCallEvent, BeforeModelCallEvent, AfterInvocationEvent):
            self.assertIn(event, seen)


class TheStreamSendsIt(unittest.TestCase):

    def test_the_pump_puts_usage_on_the_queue_when_it_moved(self):
        import inspect
        from src.utils import agentY_server
        src = inspect.getsource(agentY_server)
        body = src.split("def _flush_activity() -> None:", 1)[1].split("async def _pump()", 1)[0]
        self.assertIn("_turn_usage.take_changed()", body)
        self.assertIn('out_q.put({"type": "usage", **_used})', body)


if __name__ == "__main__":
    unittest.main()
