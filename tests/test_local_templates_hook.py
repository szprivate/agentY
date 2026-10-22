"""A run that finished clean is kept as a machine-local template - and only then.

    python -m unittest tests.test_local_templates_hook
"""
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from agenty_core.tools import comfyui as C
from agenty_core.tools import local_templates as L
from src.utils import local_templates_hook as H

GRAPH = {
    "1": {"class_type": "OpenAIGPTImage1", "inputs": {"prompt": "a red fox in the snow at dawn", "quality": "high"}},
    "2": {"class_type": "SaveImage", "inputs": {"images": ["1", 0], "filename_prefix": "agent/images/fox"}},
}
REQUEST = "make a poster image of a red fox in the snow with GPT Image"


class Hook(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.TemporaryDirectory()
        self.addCleanup(self.d.cleanup)
        self.root = Path(self.d.name)
        for target, val in (("_custom_templates_dir", self.root), ("_official_templates_dir", self.root / "o")):
            p = mock.patch.object(C, target, return_value=val)
            p.start()
            self.addCleanup(p.stop)
        self.wf = self.root / "wf.json"
        self.wf.write_text(json.dumps(GRAPH))

    def test_a_clean_run_is_kept_as_auto(self):
        name = H.record(str(self.wf), REQUEST, ["out.png"])
        self.assertTrue(name)
        entry = L._read_entries()[0]
        self.assertEqual(entry["source"], "auto")
        self.assertEqual(entry["output_kind"], "image")

    def test_a_run_without_outputs_is_not(self):
        self.assertEqual(H.record(str(self.wf), REQUEST, []), "")
        self.assertEqual(L._read_entries(), [])

    def test_a_one_word_request_is_not(self):
        self.assertEqual(H.record(str(self.wf), "fox", ["o.png"]), "")

    def test_hook_nodes_and_oversize_graphs_are_not(self):
        hooked = {**GRAPH, "3": {"class_type": "agentY hook", "inputs": {}}}
        self.assertIn("hook", H.eligible(hooked, REQUEST, ["o.png"]))
        big = {str(i): {"class_type": "KSampler", "inputs": {}} for i in range(H.MAX_NODES + 1)}
        self.assertIn("nodes", H.eligible(big, REQUEST, ["o.png"]))
        self.assertIn("work", H.eligible({"1": {"class_type": "LoadImage", "inputs": {}}}, REQUEST, ["o.png"]))

    def test_the_setting_switches_it_off(self):
        with mock.patch.object(H, "enabled", return_value=False):
            self.assertEqual(H.record(str(self.wf), REQUEST, ["o.png"]), "")

    def test_the_same_job_twice_is_one_template(self):
        H.record(str(self.wf), REQUEST, ["o.png"])
        H.record_turn([str(self.wf)], REQUEST, ["o2.png"])
        self.assertEqual(len(L._read_entries()), 1)
        self.assertEqual(L._read_entries()[0]["uses"], 2)

    def test_auto_jobs_need_stronger_wording_to_match(self):
        H.record(str(self.wf), REQUEST, ["o.png"])
        self.assertEqual(len(L.match_local("make a poster image of a red fox in the snow with GPT Image")), 1)
        self.assertEqual(L.match_local("an image of a fox, GPT Image"), [])


if __name__ == "__main__":
    unittest.main()
