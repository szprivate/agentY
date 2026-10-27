"""A finished job with no output files says why.

run_workflow_now answered {"status": "done", "outputs": []} "Workflow ran but
produced no output files." for runs that errored, were served from ComfyUI's cache
or went through a viewer node that only buffers; the agent then re-ran the same
graph. why_no_outputs reads the history and the graph and names the cause.

    python -m unittest tests.test_why_no_outputs
"""
import json
import tempfile
import unittest
from pathlib import Path

from src.executor import NO_OUTPUT_MARKER, _process_completed_job, why_no_outputs


def hist(status_str="success", messages=(), outputs=None):
    return {"p1": {"status": {"status_str": status_str, "completed": True,
                              "messages": list(messages)},
                   "outputs": outputs or {}}}


def graph(save):
    d = Path(tempfile.mkdtemp())
    p = d / "w.json"
    p.write_text(json.dumps({"23": {"class_type": "bEpicSendToViewer",
                                    "inputs": {"save_to_output": save}}}), encoding="utf-8")
    return str(p)


class WhyNoOutputs(unittest.TestCase):
    def test_execution_error_is_surfaced(self):
        h = hist("error", [["execution_error", {"node_id": "9", "node_type": "KSampler",
                                                "exception_message": "CUDA out of memory"}]])
        out = why_no_outputs(h)
        self.assertIn("CUDA out of memory", out)
        self.assertIn("KSampler", out)

    def test_all_cached(self):
        h = hist(messages=[["execution_cached", {"nodes": ["23", "14"]}]],
                 outputs={"23": {"images": []}})
        self.assertIn("cached - nothing new was run", why_no_outputs(h))

    def test_viewer_with_save_off(self):
        self.assertIn("save_to_output", why_no_outputs(hist(), graph(False)))

    def test_output_node_ran_without_files(self):
        out = why_no_outputs(hist(outputs={"23": {"text": ["x"]}}), graph(True))
        self.assertIn("ran but produced no", out)

    def test_no_output_node_ran(self):
        self.assertIn("No output node ran", why_no_outputs(hist(), graph(True)))

    def test_the_executor_line_carries_the_reason(self):
        async def go():
            return [x async for x in _process_completed_job(
                hist(), "p1", {}, workflow_path=graph(False), verbose=False,
                collected_paths=[])]
        import asyncio
        lines = asyncio.run(go())
        self.assertEqual(len(lines), 1)
        self.assertIn(NO_OUTPUT_MARKER, lines[0])
        self.assertIn("save_to_output", lines[0].split(NO_OUTPUT_MARKER, 1)[1])


if __name__ == "__main__":
    unittest.main()
