"""run_python_node: a snippet run in an agentY python node, fed from the canvas.

Only the nodes the snippet reads from are submitted — the whole canvas would
also run the user's save nodes and generations — and what the node reports
back (its output lines, its files, its error) is read off the run's history.

    python -m unittest discover -s tests
"""

import unittest
from unittest import mock

from src.utils import python_node as py

BASE = {
    "1": {"class_type": "LoadImage", "inputs": {"image": "a.png"}},
    "2": {"class_type": "ImageScale", "inputs": {"image": ["1", 0], "width": 64}},
    "3": {"class_type": "SaveImage", "inputs": {"images": ["2", 0]}},
    "4": {"class_type": "KSampler", "inputs": {"seed": 1}},
    "5:3": {"class_type": "VAEDecode", "inputs": {}},
}


class InputsTest(unittest.TestCase):

    def test_ids_and_slots(self):
        got = py.parse_inputs(["2", {"node_id": "1", "output": 1}], BASE)
        self.assertEqual(got, [("2", 0), ("1", 1)])

    def test_a_subgraph_id_is_an_id_not_a_slot(self):
        self.assertEqual(py.parse_inputs(["5:3"], BASE), [("5:3", 0)])

    def test_a_node_not_in_the_captured_graph_is_refused(self):
        with self.assertRaisesRegex(py.PythonNodeError, "in0: node 99"):
            py.parse_inputs(["99"], BASE)

    def test_no_graph_means_no_inputs(self):
        self.assertEqual(py.parse_inputs(None, None), [])
        with self.assertRaises(py.PythonNodeError):
            py.parse_inputs(["1"], None)

    def test_a_bad_slot(self):
        with self.assertRaisesRegex(py.PythonNodeError, "slot number"):
            py.parse_inputs([{"node_id": "1", "output": "x"}], BASE)


class PromptTest(unittest.TestCase):

    def test_only_what_the_snippet_reads_is_submitted(self):
        prompt = py.build_prompt("outputs=[in0]", [("2", 0)], BASE, "size")
        self.assertEqual(set(prompt), {"1", "2", py.RUN_NODE_ID})
        node = prompt[py.RUN_NODE_ID]
        self.assertEqual(node["class_type"], "AgentYPython")
        self.assertEqual(node["inputs"]["inputs.in0"], ["2", 0])
        self.assertEqual(node["inputs"]["code"], "outputs=[in0]")
        self.assertEqual(node["_meta"]["title"], "size")

    def test_no_inputs_runs_the_node_alone(self):
        self.assertEqual(set(py.build_prompt("outputs=[1]", [], BASE)), {py.RUN_NODE_ID})

    def test_the_canvas_graph_is_not_changed(self):
        py.build_prompt("x", [("2", 0)], BASE)["2"]["inputs"]["width"] = 1
        self.assertEqual(BASE["2"]["inputs"]["width"], 64)


class ResultTest(unittest.TestCase):

    def test_lines_and_files(self):
        entry = {"status": {"completed": True}, "outputs": {py.RUN_NODE_ID: {
            "text": ["out0: 8"],
            "images": [{"filename": "p.png", "subfolder": "agentY_python", "type": "output"}],
        }}}
        got = py.read_result(entry)
        self.assertEqual(got["outputs"], ["out0: 8"])
        self.assertEqual(got["files"][0]["filename"], "p.png")
        self.assertIsNone(py.error_of(entry))

    def test_the_snippet_error(self):
        entry = {"status": {"status_str": "error", "messages": [
            ["execution_start", {}],
            ["execution_error", {"exception_message": "AgentYPython snippet error: boom\n"}],
        ]}}
        self.assertEqual(py.error_of(entry), "AgentYPython snippet error: boom")


class RunTest(unittest.TestCase):

    def _client(self, post, history):
        return mock.Mock(api_key="", post=mock.Mock(side_effect=post),
                         get=mock.Mock(side_effect=history))

    def test_waits_for_the_run(self):
        done = {"p1": {"status": {"completed": True}, "outputs": {}}}
        client = self._client(lambda *a, **k: {"prompt_id": "p1"}, [{}, done])
        with mock.patch("agenty_core.utils.comfyui_client.get_client", return_value=client):
            self.assertIs(py.run({"x": {}}, poll=0), done["p1"])

    def test_a_refusal_says_why(self):
        def post(*a, **k):
            raise RuntimeError("400: Required input is missing: code")
        client = self._client(post, [])
        with mock.patch("agenty_core.utils.comfyui_client.get_client", return_value=client):
            with self.assertRaisesRegex(py.PythonNodeError, "Required input"):
                py.run({"x": {}}, poll=0)


if __name__ == "__main__":
    unittest.main()
