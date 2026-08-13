"""Tests for stopping a canvas-hook run on a hook's own condition (stop_hook_run).

A hook directive can make the rest of the run conditional — "if ANY reference
generation failed, STOP and ask the user for advice". stop_hook_run is how the
agent obeys that: nothing queued this turn is executed, the tools that would run
something refuse afterwards, and anything queued anyway is dropped at the handoff.

The tool closures are built with a stand-in `self` (they only touch a handful of
Pipeline attributes), so no agent, model or ComfyUI is involved.

    python -m unittest discover -s tests
"""

import asyncio
import json
import unittest
from types import SimpleNamespace
from unittest import mock

from src.pipeline import Pipeline
from src.utils.workflow_signal import append_workflow_path, clear_and_get


def _stub(**over):
    """A minimal stand-in for the Pipeline the tool closures are bound to."""
    base = dict(
        _hook_run_stopped=None,
        _canvas_keeplive_run=False,
        _canvas_base_prompt={"1": {"class_type": "KSampler", "inputs": {"seed": 1}}},
        _canvas_hooks=[],
        _verbose=False,
        _session=SimpleNamespace(current_output_paths=[]),
        _last_brainbriefing_json="{}",
        _chain_output_paths=[],
        _qa_briefing=None,
        _qa_retry=None,
        _heal_exec_failure=lambda *a, **k: None,
        _limit_handbacks={},
    )
    base.update(over)
    ns = SimpleNamespace(**base)
    # Real Pipeline methods — bind them, don't reimplement them, so these tests
    # keep exercising what the tool actually calls (including the hard-limit check
    # that stands between build_batch and the queue).
    ns._run_canvas_batch = Pipeline._run_canvas_batch.__get__(ns)
    ns._batch_limit_refusal = Pipeline._batch_limit_refusal.__get__(ns)
    ns._count_handback = Pipeline._count_handback.__get__(ns)
    return ns


def _tools(pipe):
    """Name -> callable for the orchestrator's delegation tools."""
    return {t.tool_name: t for t in Pipeline._build_delegation_tools(pipe)}


def _call(tool, **kwargs):
    return json.loads(asyncio.run(tool(**kwargs)))


class StopHookRunTest(unittest.TestCase):
    def setUp(self):
        clear_and_get()          # the mailbox is process-global
        self.addCleanup(clear_and_get)

    # ── the stop itself ──────────────────────────────────────────────────────
    def test_stop_records_the_reason_and_the_question(self):
        pipe = _stub()
        out = _call(_tools(pipe)["stop_hook_run"],
                    reason="2 of 3 reference generations failed",
                    question="Re-run the failures or change the prompt?")
        self.assertEqual(out["status"], "stopped")
        self.assertEqual(pipe._hook_run_stopped["reason"],
                         "2 of 3 reference generations failed")
        self.assertIn("Re-run the failures", pipe._hook_run_stopped["question"])
        # The agent is told, in the result, to stop calling tools.
        self.assertIn("Do NOT", out["message"])
        self.assertIn("Re-run the failures", out["message"])

    def test_stop_discards_what_was_already_queued(self):
        append_workflow_path("C:/tmp/canvas_000.json")
        append_workflow_path("C:/tmp/canvas_001.json")
        pipe = _stub()
        out = _call(_tools(pipe)["stop_hook_run"], reason="reference 2 failed")
        self.assertEqual(out["discarded_queued_workflows"], 2)
        self.assertEqual(clear_and_get(), [], "the mailbox must be empty after a stop")

    def test_stop_cancels_a_pending_keep_live_run(self):
        pipe = _stub(_canvas_keeplive_run=True)
        _call(_tools(pipe)["stop_hook_run"], reason="script named no characters")
        self.assertFalse(pipe._canvas_keeplive_run)

    def test_a_stop_without_a_reason_is_refused(self):
        pipe = _stub()
        out = _call(_tools(pipe)["stop_hook_run"], reason="   ")
        self.assertIn("error", out)
        self.assertIsNone(pipe._hook_run_stopped, "an empty reason must not stop the run")

    # ── nothing runs after a stop ────────────────────────────────────────────
    def test_apply_canvas_hooks_refuses_after_a_stop(self):
        pipe = _stub()
        tools = _tools(pipe)
        _call(tools["stop_hook_run"], reason="reference 2 failed")
        out = _call(tools["apply_canvas_hooks"], resolutions=[
            {"target_node_id": "1", "param": "seed", "mode": "sweep_seed", "count": 3}])
        self.assertIn("error", out)
        self.assertIn("reference 2 failed", out["error"])
        self.assertEqual(clear_and_get(), [], "a refused call must queue nothing")

    def test_run_workflow_now_refuses_after_a_stop(self):
        pipe = _stub()
        tools = _tools(pipe)
        _call(tools["stop_hook_run"], reason="only 1 of 5 shots exists")
        out = _call(tools["run_workflow_now"], workflow_path="C:/tmp/stage2.json")
        self.assertIn("error", out)
        self.assertIn("only 1 of 5 shots exists", out["error"])

    def test_the_same_tools_work_when_nothing_stopped_the_run(self):
        pipe = _stub()
        out = _call(_tools(pipe)["apply_canvas_hooks"], resolutions=[
            {"target_node_id": "1", "param": "seed", "mode": "sweep_seed", "count": 3}])
        self.assertEqual(out.get("status"), "queued")
        self.assertEqual(out.get("count"), 3)
        self.assertEqual(len(clear_and_get()), 3)

    # ── the end-of-turn handoff ──────────────────────────────────────────────
    def test_handoff_executes_the_queue_normally(self):
        append_workflow_path("C:/tmp/a.json")
        pipe = _stub()
        self.assertEqual(Pipeline._pending_execution_paths(pipe), ["C:/tmp/a.json"])

    def test_handoff_executes_nothing_after_a_stop(self):
        pipe = _stub(_hook_run_stopped={"reason": "x", "question": "", "discarded": 0},
                     _canvas_keeplive_run=True)
        # Queued despite the stop (an agent that kept going anyway).
        append_workflow_path("C:/tmp/late.json")
        self.assertEqual(Pipeline._pending_execution_paths(pipe), [])
        self.assertFalse(pipe._canvas_keeplive_run, "keep-live must not run either")
        self.assertEqual(clear_and_get(), [],
                         "the mailbox must still be drained, or it leaks into the next turn")

    # ── "stop, but let what I queued finish" ─────────────────────────────────
    def test_keep_queued_leaves_the_queue_alone(self):
        append_workflow_path("C:/tmp/ref_000.json")
        append_workflow_path("C:/tmp/ref_001.json")
        pipe = _stub()
        out = _call(_tools(pipe)["stop_hook_run"],
                    reason="hook 30 needs the references first",
                    keep_queued=True)
        self.assertEqual(out["kept_queued_workflows"], 2)
        self.assertEqual(out["discarded_queued_workflows"], 0)
        self.assertIn("will still run", out["message"])
        # Still stopped — just not cancelled.
        self.assertTrue(pipe._hook_run_stopped["keep_queued"])
        self.assertEqual(Pipeline._pending_execution_paths(pipe),
                         ["C:/tmp/ref_000.json", "C:/tmp/ref_001.json"])

    def test_keep_queued_still_blocks_further_work(self):
        pipe = _stub()
        tools = _tools(pipe)
        _call(tools["stop_hook_run"], reason="references first", keep_queued=True)
        out = _call(tools["run_workflow_now"], workflow_path="C:/tmp/stage2.json")
        self.assertIn("error", out)

    # ── run_now: the results a condition can actually read ───────────────────
    def test_run_now_executes_instead_of_queueing(self):
        ran = {}

        async def fake_batch(paths, *a, **kw):
            ran["paths"] = list(paths)
            ran["repair"] = kw.get("repair_fn")
            kw["collected_paths"].extend(["C:/out/ref_0.png", "C:/out/ref_1.png"])
            yield "queued 2"

        pipe = _stub()
        with mock.patch("src.pipeline._execute_workflows_batch", fake_batch), \
             mock.patch("src.pipeline._clear_exec_errors"), \
             mock.patch("src.pipeline._get_exec_errors", return_value=[]):
            out = _call(_tools(pipe)["apply_canvas_hooks"], run_now=True, resolutions=[
                {"target_node_id": "1", "param": "seed", "mode": "sweep_seed", "count": 2}])
        self.assertEqual(out["status"], "ran")
        self.assertEqual(out["failed_count"], 0)
        self.assertTrue(all(v["ok"] for v in out["variants"]))
        self.assertEqual(len(ran["paths"]), 2, "both variants were executed")
        self.assertIsNotNone(ran["repair"], "failures should still be healed first")
        self.assertEqual(clear_and_get(), [], "run_now must not ALSO queue them")
        # Outputs survive the end-of-turn reset.
        self.assertEqual(pipe._chain_output_paths, ["C:/out/ref_0.png", "C:/out/ref_1.png"])

    def test_run_now_reports_which_variant_failed(self):
        written = {}

        async def fake_batch(paths, *a, **kw):
            written["paths"] = list(paths)
            yield "done"

        pipe = _stub()
        with mock.patch("src.pipeline._execute_workflows_batch", fake_batch), \
             mock.patch("src.pipeline._clear_exec_errors"), \
             mock.patch("src.pipeline._get_exec_errors") as errs:
            errs.side_effect = lambda: [{"workflow_path": written["paths"][1],
                                         "error": "SeedreamNode: upstream 500"}]
            out = _call(_tools(pipe)["apply_canvas_hooks"], run_now=True, resolutions=[
                {"target_node_id": "1", "param": "seed", "mode": "sweep_seed", "count": 3}])
        self.assertEqual(out["failed_count"], 1)
        self.assertEqual([v["ok"] for v in out["variants"]], [True, False, True])
        self.assertIn("upstream 500", out["variants"][1]["error"])
        # The agent is pointed at the decision it now has to make.
        self.assertIn("stop_hook_run", out["message"])

    def test_run_now_surfaces_an_executor_crash_rather_than_claiming_success(self):
        async def boom(paths, *a, **kw):
            raise RuntimeError("ComfyUI unreachable")
            yield  # pragma: no cover — generator marker

        pipe = _stub()
        with mock.patch("src.pipeline._execute_workflows_batch", boom), \
             mock.patch("src.pipeline._clear_exec_errors"):
            out = _call(_tools(pipe)["apply_canvas_hooks"], run_now=True, resolutions=[
                {"target_node_id": "1", "param": "seed", "mode": "sweep_seed", "count": 2}])
        self.assertEqual(out["status"], "error")
        self.assertIn("ComfyUI unreachable", out["error"])

    def test_the_stop_does_not_survive_into_the_next_turn(self):
        """A stop is per-turn state: the next turn starts able to run again."""
        import inspect
        src = inspect.getsource(Pipeline._astream_orchestrator)
        self.assertIn("self._hook_run_stopped = None", src,
                      "the per-turn reset must clear the stop")


if __name__ == "__main__":
    unittest.main()
