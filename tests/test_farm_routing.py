"""Runs routed to the Deadline farm: off by default, and when on, the executor's own flow
(prepare, cost gate, progress, output collection, healing) is the one that runs - only the queueing differs.

The farm is a fake (Deadline's status calls are stubbed); ComfyUI must not be touched.

    python -m unittest tests.test_farm_routing
"""

import asyncio
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from src import executor
from agenty_core.tools import deadline_farm as df
from agenty_core.utils import farm_run

PROMPT = {"1": {"class_type": "KSampler", "inputs": {"seed": 1}},
          "2": {"class_type": "SaveImage", "inputs": {"images": ["1", 0], "filename_prefix": "x"}}}


def run(coro):
    return asyncio.run(coro)


_REAL_SLEEP = asyncio.sleep


async def lines_of(gen):
    return [x async for x in gen]


class _Case(unittest.TestCase):
    def setUp(self):
        d = tempfile.TemporaryDirectory()
        self.addCleanup(d.cleanup)
        self.dir = Path(d.name)
        self.wf = self.dir / "wf.json"
        self.wf.write_text(json.dumps(PROMPT))
        self.result = self.dir / "share" / "x_t0000.png"
        self.result.parent.mkdir()
        self.result.write_bytes(b"png")
        self.enterContext(mock.patch.dict(os.environ, {"DEADLINE_ROUTE_RUNS": "", "DEADLINE_FETCH_DIR": ""}))


class SwitchTest(_Case):

    def test_off_by_default(self):
        with mock.patch("src.utils.settings.load_settings", return_value={}):
            self.assertFalse(executor._routed_to_farm())

    def test_the_setting_turns_it_on(self):
        with mock.patch("src.utils.settings.load_settings",
                        return_value={"deadline": {"route_runs": True}}):
            self.assertTrue(executor._routed_to_farm())

    def test_the_environment_can_turn_it_off_again(self):
        with mock.patch("src.utils.settings.load_settings",
                        return_value={"deadline": {"route_runs": True}}), \
                mock.patch.dict(os.environ, {"DEADLINE_ROUTE_RUNS": "0"}):
            self.assertFalse(executor._routed_to_farm())

    def test_the_shipped_default_is_off(self):
        from src.utils import settings
        self.assertIs(settings.load_defaults()["deadline"]["route_runs"], False)

    def test_a_farm_reference_gets_the_farm_stream(self):
        gen = executor._job_stream("farm:J1", "cid")
        self.assertEqual(type(gen).__qualname__, farm_run.stream("farm:J1").__class__.__qualname__)
        run(gen.aclose())

    def test_a_partner_credit_check_is_skipped_for_the_farm(self):
        self.assertEqual(run(executor._settle_partner("farm:J1")), "")


class SubmitTest(_Case):

    def setUp(self):
        super().setUp()
        # The graph-preparation steps ask ComfyUI for its node list; there is none here.
        for target in ("agenty_core.tools.comfyui.normalize_autogrow_wires",
                       "src.utils.media_nodes.apply_savers", "src.utils.viewer_group.stamp",
                       "src.utils.output_context.stamp"):
            self.enterContext(mock.patch(target, return_value=False))

    def _gate(self):
        return (mock.patch("agenty_core.partner_credits.gate_submission", return_value=0),
                mock.patch("src.utils.partner_cost_host.ensure_registered"))

    def test_routed_submission_never_posts_to_comfyui(self):
        client = mock.Mock()
        g1, g2 = self._gate()
        with g1, g2, mock.patch("agenty_core.utils.comfyui_client.get_client", return_value=client), \
                mock.patch.object(executor, "_routed_to_farm", return_value=True), \
                mock.patch.object(farm_run, "submit", return_value="farm:J1") as sub:
            ref = executor._submit_workflow(str(self.wf), client_id="c")
        self.assertEqual(ref, "farm:J1")
        client.post.assert_not_called()
        self.assertEqual(sub.call_args.kwargs["name"], "wf")
        self.assertIn("2", sub.call_args.args[0])             # the prepared prompt, not the path

    def test_the_cost_gate_still_runs_before_anything_is_sent(self):
        from agenty_core.partner_credits import CostGateError
        with mock.patch("agenty_core.partner_credits.gate_submission",
                        side_effect=CostGateError("confirm_cost", "too expensive", {}, None)), \
                mock.patch("src.utils.partner_cost_host.ensure_registered"), \
                mock.patch.object(executor, "_routed_to_farm", return_value=True), \
                mock.patch.object(farm_run, "submit") as sub:
            with self.assertRaises(CostGateError):
                executor._submit_workflow(str(self.wf))
        sub.assert_not_called()

    def test_a_refusing_farm_is_a_submission_failure_like_any_other(self):
        g1, g2 = self._gate()
        with g1, g2, mock.patch.object(executor, "_routed_to_farm", return_value=True), \
                mock.patch.object(df, "submit",
                                  side_effect=df.FarmError("pool 'comfyui' does not exist", "config")):
            with self.assertRaises(RuntimeError) as cm:
                executor._submit_workflow(str(self.wf))
        self.assertIn("does not exist", str(cm.exception))


class RunTest(_Case):
    """execute_workflow end to end against a fake farm."""

    def _run(self, states):
        it = iter(states)
        collected: list = []
        touched = {"vram": 0, "history": 0}
        with mock.patch.object(executor, "_routed_to_farm", return_value=True), \
                mock.patch.object(executor, "_submit_workflow", return_value="farm:J1"), \
                mock.patch.object(executor, "_free_vram_for_comfyui",
                                  lambda: touched.__setitem__("vram", touched["vram"] + 1)), \
                mock.patch.object(executor, "_clear_comfyui_history",
                                  lambda: touched.__setitem__("history", touched["history"] + 1)), \
                mock.patch.object(executor, "_copy_workflow_to_user_dir"), \
                mock.patch.object(executor, "_record_exec_error") as rec, \
                mock.patch.object(df, "job_status", side_effect=lambda j, w=0: next(it)), \
                mock.patch.object(farm_run.asyncio, "sleep", new=lambda s: _REAL_SLEEP(0)):
            lines = run(lines_of(executor.execute_workflow(
                str(self.wf), "{}", collected_paths=collected, verbose=False)))
        return lines, collected, touched, rec

    def test_a_finished_farm_job_delivers_its_files_like_a_local_run(self):
        lines, collected, touched, _ = self._run([
            {"state": "Queued", "tasks": {}},
            {"state": "Rendering", "progress": 0.5, "tasks": {"completed": 0, "total": 1}},
            {"state": "Completed", "outputs": [str(self.result)]}])
        text = "\n".join(l for l in lines if isinstance(l, str))
        self.assertIn("Deadline farm", text)
        self.assertIn("waiting for a free worker", text)
        self.assertIn("rendering", text)
        self.assertIn(f"Output: `{self.result}`", text)
        self.assertIn("✅ Done", text)
        self.assertEqual(collected, [str(self.result)])
        self.assertEqual(touched, {"vram": 0, "history": 0}, "no local ComfyUI housekeeping for a farm run")

    def test_a_failed_farm_job_is_recorded_with_its_node_for_the_repair(self):
        err = "ComfyUI execution_error in KSampler (node 1): RuntimeError: boom"
        lines, collected, _, rec = self._run([{"state": "Failed", "errors": err}])
        self.assertEqual(collected, [])
        text = "\n".join(l for l in lines if isinstance(l, str))
        self.assertIn("execution error", text)
        details = rec.call_args.args[0]
        self.assertEqual((details["node_id"], details["node_type"]), ("1", "KSampler"))

    def test_a_job_suspended_in_monitor_is_not_repaired(self):
        lines, _, _, rec = self._run([{"state": "Suspended"}])
        text = "\n".join(l for l in lines if isinstance(l, str))
        self.assertIn("interrupted", text.lower())
        rec.assert_not_called()


class NotRoutedTest(_Case):
    """With routing off, the executor takes the local path exactly as before."""

    def test_local_run_still_streams_comfyui_and_cleans_up(self):
        touched = []

        async def fake_stream(pid, cid, **kw):
            yield "▶ Execution started"
            yield {"history": {pid: {"outputs": {"2": {"images": [
                {"filename": self.result.name, "subfolder": "", "type": "output"}]}}}}}

        with mock.patch.object(executor, "_routed_to_farm", return_value=False), \
                mock.patch.object(executor, "_submit_workflow", return_value="pid1"), \
                mock.patch.object(executor, "_free_vram_for_comfyui", lambda: touched.append("vram")), \
                mock.patch.object(executor, "_clear_comfyui_history", lambda: touched.append("hist")), \
                mock.patch.object(executor, "_copy_workflow_to_user_dir"), \
                mock.patch.object(executor, "_resolve_output_path", return_value=self.result), \
                mock.patch("agenty_core.utils.comfyui_progress.stream_comfyui_job", fake_stream):
            lines = run(lines_of(executor.execute_workflow(str(self.wf), "{}", verbose=False)))
        text = "\n".join(l for l in lines if isinstance(l, str))
        self.assertIn("Submitting workflow to ComfyUI", text)
        self.assertNotIn("farm", text.lower())
        self.assertEqual(touched, ["vram", "hist"])


if __name__ == "__main__":
    unittest.main()
