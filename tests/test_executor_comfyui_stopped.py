"""ComfyUI dying mid-run: the executor says so, names the partial file, and does not repair.

Seen 2026-10-27: ComfyUI crashed while writing an LTX video; run_workflow_now waited
7.8 hours on the dead port. The wait now ends (agenty_core.utils.comfyui_liveness);
this checks what the executor does with that ending.
"""
import asyncio
import unittest
from unittest import mock

from src import executor


async def _stopped_stream(*a, **kw):
    yield "▶ Execution started"
    yield {"error": "ComfyUI stopped responding / crashed during the run: its API has not answered "
                    "for 61s. Partial output on disk: C:/out/v.mp4 (TRUNCATED / unplayable: no moov atom).",
           "comfyui_stopped": True,
           "details": {"prompt_id": "p1", "reason": "comfyui_unreachable", "partial_outputs": []}}


class ExecutorStoppedTest(unittest.TestCase):
    def test_reports_the_crash_and_does_not_record_a_repair(self):
        async def go():
            out = []
            with mock.patch.object(executor, "_job_stream", _stopped_stream), \
                 mock.patch.object(executor, "_submit_workflow", lambda *a, **k: "p1"), \
                 mock.patch.object(executor, "_free_vram_for_comfyui", lambda: None), \
                 mock.patch.object(executor, "_clear_comfyui_history", lambda: None), \
                 mock.patch.object(executor, "_fit_to_briefing", lambda p, q: (p, [])), \
                 mock.patch.object(executor, "_settle_partner", mock.AsyncMock(return_value="")), \
                 mock.patch.object(executor, "_record_exec_error") as rec:
                async for line in executor.execute_workflow("wf.json", "{}", verbose=False):
                    out.append(line)
            return out, rec
        out, rec = asyncio.run(go())
        bad = [l for l in out if isinstance(l, str) and l.startswith("❌")]
        self.assertEqual(len(bad), 1, out)
        self.assertIn("stopped responding / crashed", bad[0])
        self.assertIn("no moov atom", bad[0])
        rec.assert_not_called()


if __name__ == "__main__":
    unittest.main()
