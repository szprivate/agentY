"""A fitting template whose only problem is missing model files comes back
assembled with the list - it is not sent to the repair specialist."""

import asyncio
import json
import unittest
from unittest import mock

from src import pipeline


class WhichProblemsAreOnlyMissingModels(unittest.TestCase):

    def test_all_missing_models(self):
        probs = ["Node 12 (UNETLoader): model 'ltx-2.5-22b-dev.safetensors' is not installed. Get it first ...",
                 "Node 14 (LoraLoaderModelOnly): model 'ltx-2.5-ic-lora-sdr-to-hdr.safetensors' is not installed. ...",
                 "Node 15 (UNETLoader): model 'ltx-2.5-22b-dev.safetensors' is not installed. Get it first ..."]
        self.assertEqual(pipeline._only_missing_models(probs),
                         ["ltx-2.5-22b-dev.safetensors", "ltx-2.5-ic-lora-sdr-to-hdr.safetensors"])

    def test_a_real_defect_among_them_still_goes_to_repair(self):
        probs = ["Node 12 (UNETLoader): model 'a.safetensors' is not installed.",
                 "Node 3 (KSampler): missing required input 'model'."]
        self.assertEqual(pipeline._only_missing_models(probs), [])

    def test_no_problems(self):
        self.assertEqual(pipeline._only_missing_models([]), [])


class TheAssembly(unittest.TestCase):

    def test_missing_models_are_handed_back_not_repaired(self):
        p = pipeline.Pipeline.__new__(pipeline.Pipeline)
        p._verbose = False
        briefing = mock.Mock()
        briefing.template.name = "ltx2_5_vfx_sdr_to_hdr"
        briefing.model_dump_json.return_value = "{}"
        res = {"status": "error", "workflow_path": "/w/ltx_abc.json",
               "problems": ["Node 12 (UNETLoader): model 'ltx-2.5-22b-dev.safetensors' is not installed. Get it first"],
               "server_errors": {}}
        repair = mock.AsyncMock(return_value={"status": "ready"})
        with mock.patch("src.tools.get_workflow_template", lambda n: json.dumps({"workflow_path": "/w/ltx.json"})), \
             mock.patch("src.tools.apply_brainbriefing", lambda w, b: json.dumps(res)), \
             mock.patch.object(pipeline.Pipeline, "_own_copy", staticmethod(lambda w: w)), \
             mock.patch.object(pipeline.Pipeline, "_run_fix_workflow_assembly", repair, create=True):
            got = asyncio.run(p._assemble_deterministic(briefing))
        repair.assert_not_called()
        self.assertEqual(got["status"], "missing_models")
        self.assertEqual(got["missing_models"], ["ltx-2.5-22b-dev.safetensors"])
        self.assertEqual(got["workflow_path"], "/w/ltx_abc.json")
        self.assertIn("download_hf_model", got["guidance"])

    def test_the_orchestrator_is_told_what_to_do_with_it(self):
        from pathlib import Path
        prompt = (Path(pipeline.__file__).resolve().parents[1] / "config" / "system_prompts"
                  / "system_prompt.orchestrator.md").read_text(encoding="utf-8")
        self.assertIn("**`missing_models`**", prompt)


if __name__ == "__main__":
    unittest.main()
