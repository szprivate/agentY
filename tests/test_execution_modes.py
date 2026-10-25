"""Execution modes in agentY: the setting, the executor's routing seam, partner nodes in every mode, the tools,
the skill and the agent wiring. No ComfyUI, no Comfy Cloud, no credits.

    python -m unittest tests.test_execution_modes
"""
import asyncio
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from src import executor
from agenty_core import partner_credits as pc
from agenty_core.utils import cloud_run, execution_mode as em, exec_route, farm_run

PARTNER = {"1": {"class_type": "SeedreamNode", "inputs": {"prompt": "x"}},
           "2": {"class_type": "SaveImage", "inputs": {"images": ["1", 0], "filename_prefix": "x"}}}
MIXED = {**PARTNER, "3": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": "sdxl.safetensors"}}}


class _Case(unittest.TestCase):
    def setUp(self):
        d = tempfile.TemporaryDirectory()
        self.addCleanup(d.cleanup)
        self.dir = Path(d.name)
        self.wf = self.dir / "wf.json"
        self.wf.write_text(json.dumps(PARTNER))
        em.set_turn_override(None)
        self.addCleanup(em.set_turn_override, None)
        self.enterContext(mock.patch.dict(os.environ, {"AGENTY_EXECUTION_MODE": "", "DEADLINE_ROUTE_RUNS": ""}))
        # never let a test learn ComfyUI's folders (the executor caches them for the process)
        self.addCleanup(executor._reset_comfyui_dir_cache)
        self.enterContext(mock.patch.object(executor, "_output_dir", return_value=self.dir / "out"))
        for target in ("agenty_core.tools.comfyui.normalize_autogrow_wires",
                       "src.utils.media_nodes.apply_savers", "src.utils.viewer_group.stamp",
                       "src.utils.output_context.stamp"):
            self.enterContext(mock.patch(target, return_value=False))
        self.enterContext(mock.patch("src.utils.partner_cost_host.ensure_registered"))

    def mode(self, mode):
        return mock.patch("src.utils.settings.load_settings", return_value={"execution": {"mode": mode}})


class Setting(unittest.TestCase):
    def test_the_shipped_default_is_local(self):
        from src.utils import settings
        ex = settings.load_defaults()["execution"]
        self.assertEqual(ex["mode"], "local")
        self.assertGreaterEqual(ex["min_balance_usd"], 0)

    def test_existing_installs_without_the_table_stay_local(self):
        with mock.patch("src.utils.settings.load_settings", return_value={}):
            executor._register_execution_mode()
            self.assertEqual(em.effective_mode(), "local")


class LocalModeIsUnchanged(_Case):
    """local = today's path: partner nodes go through the local ComfyUI with the credit checks; no routing."""

    def _submit(self, wf_obj, gate=None):
        self.wf.write_text(json.dumps(wf_obj))
        client = mock.Mock()
        client.api_key = ""
        client.post.return_value = {"prompt_id": "P1"}
        with self.mode("local"), mock.patch("agenty_core.partner_credits.gate_submission",
                                            side_effect=gate or (lambda *a, **k: {"paid": False})) as g, \
                mock.patch("agenty_core.utils.comfyui_client.get_client", return_value=client), \
                mock.patch.object(executor, "_routed_to_farm", return_value=False), \
                mock.patch.object(exec_route, "decide") as decide, \
                mock.patch.object(cloud_run, "submit") as cloud, \
                mock.patch("src.utils.ollama_unload.release"):
            try:
                ref = executor._submit_workflow(str(self.wf), client_id="c")
            except pc.CostGateError as exc:
                return exc, g, client, decide, cloud
            return ref, g, client, decide, cloud

    def test_partner_only_runs_locally_with_the_credit_check(self):
        ref, gate, client, decide, cloud = self._submit(PARTNER)
        self.assertEqual(ref, "P1")
        gate.assert_called_once()                    # the partner-credit check is still applied
        client.post.assert_called_once()              # ... and it went to the local ComfyUI
        decide.assert_not_called()
        cloud.assert_not_called()

    def test_mixed_runs_locally(self):
        ref, gate, client, decide, cloud = self._submit(MIXED)
        self.assertEqual(ref, "P1")
        cloud.assert_not_called()
        decide.assert_not_called()

    def test_the_credit_check_still_refuses_in_local_mode(self):
        err = pc.CostGateError("confirm_cost", "too expensive", {}, None)
        ref, gate, client, decide, cloud = self._submit(PARTNER, gate=err)
        self.assertIsInstance(ref, pc.CostGateError)
        client.post.assert_not_called()
        cloud.assert_not_called()


class HybridAndCloud(_Case):
    def _run(self, mode, backend):
        decision = exec_route.Decision(backend=backend, mode=mode, open_weight=True)
        self.wf.write_text(json.dumps(MIXED))
        client = mock.Mock()
        client.api_key = ""
        client.post.return_value = {"prompt_id": "P1"}
        with self.mode(mode), mock.patch("agenty_core.partner_credits.gate_submission", return_value={"paid": False}), \
                mock.patch("agenty_core.utils.comfyui_client.get_client", return_value=client), \
                mock.patch.object(executor, "_routed_to_farm", return_value=False), \
                mock.patch.object(exec_route, "decide", return_value=decision) as decide, \
                mock.patch.object(cloud_run, "submit", return_value="cloud:C1") as cloud, \
                mock.patch("src.utils.ollama_unload.release"):
            ref = executor._submit_workflow(str(self.wf), client_id="c")
        return ref, client, decide, cloud

    def test_hybrid_sends_to_the_cloud_when_decided(self):
        ref, client, decide, cloud = self._run("hybrid", "comfy_cloud")
        self.assertEqual(ref, "cloud:C1")
        client.post.assert_not_called()
        self.assertTrue(cloud.call_args.kwargs["open_weight"])
        self.assertEqual(Path(cloud.call_args.kwargs["dest_dir"]).name, "comfy_cloud")

    def test_hybrid_stays_local_when_decided(self):
        ref, client, decide, cloud = self._run("hybrid", "local")
        self.assertEqual(ref, "P1")
        cloud.assert_not_called()

    def test_the_turn_override_wins_over_the_setting(self):
        em.set_turn_override("cloud")
        ref, client, decide, cloud = self._run("local", "comfy_cloud")
        self.assertEqual(ref, "cloud:C1")

    def test_a_refusal_is_a_cost_gate_style_answer(self):
        d = exec_route.Decision(mode="hybrid", missing_cloud_models=["a.safetensors"], hint="download it")
        err = exec_route.RouteRefused("nowhere", d)
        self.assertIsInstance(err, pc.CostGateError)
        self.assertEqual(err.payload()["error_kind"], "cannot_run_here")
        with self.mode("hybrid"), mock.patch("agenty_core.partner_credits.gate_submission", return_value={}), \
                mock.patch.object(exec_route, "decide", side_effect=err):
            with self.assertRaises(pc.CostGateError):
                executor._submit_workflow(str(self.wf))

    def test_a_broken_probe_never_stops_a_run_that_can_go_local(self):
        with self.mode("hybrid"), mock.patch.object(exec_route, "decide", side_effect=ValueError("probe")):
            self.assertIsNone(executor._route_decision(PARTNER))

    def test_cloud_mode_does_not_touch_the_local_comfyui_first(self):
        with self.mode("cloud"):
            self.assertTrue(executor._cloud_only_mode())
        with self.mode("hybrid"):
            self.assertFalse(executor._cloud_only_mode())


class Streams(_Case):
    def test_a_cloud_reference_gets_the_cloud_stream(self):
        gen = executor._job_stream("cloud:J1", "cid")
        self.assertEqual(type(gen).__qualname__, cloud_run.stream("cloud:J1").__class__.__qualname__)
        asyncio.run(gen.aclose())

    def test_a_cloud_run_says_where_it_ran_and_what_it_cost(self):
        with mock.patch.object(cloud_run, "settle_line", return_value="📍 ran_on: comfy_cloud · cost $0.12"):
            line = asyncio.run(executor._settle_partner("cloud:J1"))
        self.assertIn("comfy_cloud", line)
        self.assertIn("$0.12", line)

    def test_a_local_run_in_hybrid_says_so_and_in_local_mode_stays_quiet(self):
        with self.mode("hybrid"), mock.patch("agenty_core.partner_credits.settle", return_value=None):
            self.assertIn("ran_on: local", asyncio.run(executor._settle_partner("P1")))
        with self.mode("local"), mock.patch("agenty_core.partner_credits.settle", return_value=None):
            self.assertEqual(asyncio.run(executor._settle_partner("P1")), "")


class Wiring(unittest.TestCase):
    def test_tools_skill_and_group(self):
        import src.tools as tools
        from src import agent
        from src.tools import tool_packs
        picked = {tool_packs.name_of(t) for t in tools.ORCHESTRATOR_TOOLS}
        for n in ("set_execution_mode_for_turn", "get_execution_mode"):
            self.assertIn(n, tools._SHARED_CORE_TOOLS)
            self.assertIn(n, picked)
        self.assertIn("execution-modes", agent._ORCH_SKILL_NAMES)
        self.assertTrue(agent._skill_sources(["execution-modes"]))

    def test_the_skill_does_not_call_local_no_partner_models(self):
        text = (Path(__file__).resolve().parent.parent / "skills" / "execution-modes" / "SKILL.md").read_text("utf-8")
        self.assertIn("Partner (API) nodes", text)
        self.assertIn("local ComfyUI in every mode", text)
        self.assertNotIn("no partner", text.lower())
        said = {"sam3_masks": "SAM3", "splat_training": "train_splat", "lora_training_local": "LoRA",
                "custom_nodes": "custom nodes", "canvas_only": "canvas-only"}
        for lim in em.CLOUD_LIMITS:                      # the one list is what the skill says
            self.assertIn(said[lim["id"]].lower(), text.lower())

    def test_the_env_keys_are_known(self):
        from src.utils import agentY_server
        self.assertIn("AGENTY_EXECUTION_MODE", agentY_server._KNOWN_ENV_KEYS)


if __name__ == "__main__":
    unittest.main()
