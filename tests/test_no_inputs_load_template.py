"""No input given: the template is loaded as it is (its loaders wait for the
user's file), not sent to the repair specialist because ComfyUI does not list the
template's sample file."""

import asyncio
import json
import unittest
from unittest import mock

from src import pipeline

WARN = ("Node 3 (LoadVideo) input 'file': 'sky_citadel_ring_collapse.mp4' is not in ComfyUI's input folder (yet). "
        "Kept as written, NOT replaced by another file. Upload it (upload_image) before running, or the "
        "run fails / uses the wrong file.")
SERVER = {"error": {"type": "prompt_outputs_failed_validation"},
          "node_errors": {"3": {"class_type": "LoadVideo", "errors": [
              {"type": "value_not_in_list", "message": "Value not in list",
               "details": "file: 'sky_citadel_ring_collapse.mp4' not in []"}]}}}


def assemble(res):
    p = pipeline.Pipeline.__new__(pipeline.Pipeline)
    p._verbose = False
    briefing = mock.Mock()
    briefing.template.name = "api_ltx_video_sdr2hdr"
    briefing.model_dump_json.return_value = "{}"
    repair = mock.AsyncMock(return_value={"status": "ready"})
    with mock.patch("src.tools.get_workflow_template", lambda n: json.dumps({"workflow_path": "/w/t.json"})), \
         mock.patch("src.tools.apply_brainbriefing", lambda w, b: json.dumps(res)), \
         mock.patch.object(pipeline.Pipeline, "_own_copy", staticmethod(lambda w: w)), \
         mock.patch.object(pipeline.Pipeline, "_run_fix_workflow_assembly", repair, create=True):
        got = asyncio.run(p._assemble_deterministic(briefing))
    return got, repair


class TheAssembly(unittest.TestCase):

    def test_a_missing_input_file_loads_the_template(self):
        got, repair = assemble({"status": "error", "workflow_path": "/w/t_1.json", "problems": [],
                                "server_errors": SERVER, "warnings": [WARN]})
        repair.assert_not_called()
        self.assertEqual(got["status"], "needs_inputs")
        self.assertEqual(got["inputs_needed"], [{"node_id": "3", "class_type": "LoadVideo",
                                                 "file": "input 'file': 'sky_citadel_ring_collapse.mp4'"}])
        self.assertIn("open_workflow_in_canvas", got["guidance"])

    def test_missing_models_and_inputs_together(self):
        server = json.loads(json.dumps(SERVER))
        server["node_errors"]["7"] = {"class_type": "UNETLoader", "errors": [
            {"type": "value_not_in_list", "details": "unet_name: 'ltx-2.5-22b.safetensors' not in [...]"}]}
        got, repair = assemble({"status": "error", "workflow_path": "/w/t_1.json",
                                "problems": ["Node 7 (UNETLoader): model 'ltx-2.5-22b.safetensors' is not installed."],
                                "server_errors": server, "warnings": [WARN]})
        repair.assert_not_called()
        self.assertEqual(got["status"], "missing_models")
        self.assertEqual(got["missing_models"], ["ltx-2.5-22b.safetensors"])
        self.assertEqual(len(got["inputs_needed"]), 1)

    def test_a_real_server_error_still_goes_to_repair(self):
        server = {"error": {"type": "x"}, "node_errors": {"5": {"errors": [
            {"type": "value_bigger_than_max", "details": "bit_depth 16 > 10"}]}}}
        got, repair = assemble({"status": "error", "workflow_path": "/w/t_1.json", "problems": [],
                                "server_errors": server, "warnings": [WARN]})
        repair.assert_called_once()

    def test_a_file_error_on_a_loader_that_is_not_waiting_goes_to_repair(self):
        got, repair = assemble({"status": "error", "workflow_path": "/w/t_1.json", "problems": [],
                                "server_errors": SERVER, "warnings": []})
        repair.assert_called_once()

    def test_the_orchestrator_knows_the_status(self):
        from pathlib import Path
        prompt = (Path(pipeline.__file__).resolve().parents[1] / "config" / "system_prompts"
                  / "system_prompt.orchestrator.md").read_text(encoding="utf-8")
        self.assertIn("**`needs_inputs`**", prompt)


if __name__ == "__main__":
    unittest.main()


class AMissingNodePack(unittest.TestCase):
    """Mac, 2026-11-15: the LTXVideo pack was not installed; the template went to
    repair, the pack was git-cloned and 25 GB downloaded for "don't run it"."""

    def test_is_handed_back_loaded_with_what_is_missing(self):
        server = {"error": {"type": "missing_node_type",
                            "message": "Node 'LTXVHDRDecodePostprocess' not found. The custom node may not be installed."},
                  "node_errors": {}}
        with mock.patch.object(pipeline, "_classes_not_installed",
                               lambda p: ["LTXICLoRALoaderModelOnly", "LTXVHDRDecodePostprocess"]):
            got, repair = assemble({"status": "error", "workflow_path": "/w/t_1.json",
                                    "problems": ["Node 7 (UNETLoader): model 'ltx-2.5-22b.safetensors' is not installed."],
                                    "server_errors": server, "warnings": [WARN]})
        repair.assert_not_called()
        self.assertEqual(got["status"], "missing_nodes")
        self.assertEqual(got["missing_nodes"], ["LTXVHDRDecodePostprocess", "LTXICLoRALoaderModelOnly"])
        self.assertEqual(got["missing_models"], ["ltx-2.5-22b.safetensors"])
        for word in ("ASK before installing", "never git clone", "open_workflow_in_canvas"):
            self.assertIn(word, got["guidance"])

    def test_an_unknown_class_found_locally(self):
        got, repair = assemble({"status": "error", "workflow_path": "/w/t_1.json",
                                "problems": ["Node 4: unknown class_type 'LTXVLoadConditioning'."],
                                "server_errors": {}})
        repair.assert_not_called()
        self.assertEqual(got["missing_nodes"], ["LTXVLoadConditioning"])

    def test_missing_models_alone_ask_before_downloading(self):
        got, _ = assemble({"status": "error", "workflow_path": "/w/t_1.json",
                           "problems": ["Node 7 (UNETLoader): model 'a.safetensors' is not installed."],
                           "server_errors": {}})
        self.assertEqual(got["status"], "missing_models")
        self.assertIn("ASK before", got["guidance"])
