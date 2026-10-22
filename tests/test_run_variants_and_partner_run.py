"""The orchestrator's run_variants and partner_run: one call, the host's own execution path.

    python -m unittest discover -s tests -p "test_run_variants_and_partner_run.py"
"""
import asyncio
import json
import unittest
from unittest import mock

from pipeline_stub import pipeline_stub as _stub, tools as _tools


def _call(tool, **kwargs):
    return json.loads(asyncio.run(tool(**kwargs)))


def _fake_executor(outputs_by_path, seen=None):
    async def execute_workflow(path, brief, user_message="", verbose=False, collected_paths=None, **kw):
        if seen is not None:
            seen.append(path)
        yield f"running {path}"
        collected_paths.extend(outputs_by_path.get(path, [f"{path}.png"]))
    return execute_workflow


class RunVariants(unittest.TestCase):
    def prepared(self, n=3, status="ok"):
        paths = [f"C:/wf/v{i}.json" for i in range(n)]
        return {"status": status, "ready": paths,
                "variants": [{"index": i + 1, "label": f"v{i}", "workflow_path": p, "status": "ok", "problems": []}
                             for i, p in enumerate(paths)]}

    def test_all_variants_run_in_one_call_and_are_staged(self):
        pipe = _stub()
        seen = []
        prep = self.prepared(3)
        with mock.patch("agenty_core.tools.variants.prepare_variants", return_value=prep), \
             mock.patch("src.executor.execute_workflow", _fake_executor({}, seen)):
            out = _call(_tools(pipe)["run_variants"], workflow_path="C:/wf/base.json", variants=[{}, {}, {}])
        self.assertTrue(out["ran"])
        self.assertEqual(sorted(seen), sorted(prep["ready"]))
        self.assertEqual(len(out["outputs"]), 3)
        self.assertEqual([v["outputs"] for v in out["variants"]], [[f"{p}.png"] for p in prep["ready"]])
        self.assertEqual(len(pipe._session.current_output_paths), 3)
        self.assertEqual(len(pipe._chain_output_paths), 3)

    def test_run_false_builds_only(self):
        pipe = _stub()
        with mock.patch("agenty_core.tools.variants.prepare_variants", return_value=self.prepared(2)), \
             mock.patch("src.executor.execute_workflow", side_effect=AssertionError("must not run")):
            out = _call(_tools(pipe)["run_variants"], workflow_path="b", variants=[{}, {}], run=False)
        self.assertFalse(out["ran"])

    def test_a_failed_build_is_returned(self):
        pipe = _stub()
        with mock.patch("agenty_core.tools.variants.prepare_variants",
                        return_value={"status": "error", "error": "no variants given"}):
            out = _call(_tools(pipe)["run_variants"], workflow_path="b", variants=[])
        self.assertEqual(out["status"], "error")
        self.assertFalse(out["ran"])

    def test_one_failing_variant_does_not_sink_the_others(self):
        pipe = _stub()

        async def flaky(path, brief, user_message="", verbose=False, collected_paths=None, **kw):
            if path.endswith("v1.json"):
                raise RuntimeError("boom")
            collected_paths.append(path + ".png")
            yield "ok"

        with mock.patch("agenty_core.tools.variants.prepare_variants", return_value=self.prepared(3)), \
             mock.patch("src.executor.execute_workflow", flaky):
            out = _call(_tools(pipe)["run_variants"], workflow_path="b", variants=[{}, {}, {}])
        self.assertEqual(len(out["outputs"]), 2)
        self.assertIn("v1", out["errors"])

    def test_it_is_refused_after_a_hook_stop(self):
        pipe = _stub(_hook_run_stopped={"reason": "stopped"})
        out = _call(_tools(pipe)["run_variants"], workflow_path="b", variants=[{}])
        self.assertIn("error", out)

    def test_a_dry_run_runs_nothing(self):
        pipe = _stub(_dry_run=True)
        with mock.patch("agenty_core.tools.variants.prepare_variants", return_value=self.prepared(2)), \
             mock.patch("src.executor.execute_workflow", side_effect=AssertionError("must not run")), \
             mock.patch.object(pipe, "_dry_run_one", return_value="would run", create=True):
            out = _call(_tools(pipe)["run_variants"], workflow_path="b", variants=[{}, {}])
        self.assertFalse(out["ran"])


class PartnerRun(unittest.TestCase):
    READY = {"status": "ready", "workflow_path": "C:/wf/partner_x.json", "node": "OpenAIGPTImage1",
             "loaders": [], "warnings": []}

    def test_it_builds_then_runs_through_run_workflow_now(self):
        pipe = _stub()
        seen = []
        with mock.patch("agenty_core.tools.partner_run.prepare_partner_workflow", return_value=self.READY), \
             mock.patch("src.executor.execute_workflow", _fake_executor({"C:/wf/partner_x.json": ["fox.png"]}, seen)), \
             mock.patch("src.utils.local_templates_hook.record", return_value=""):
            out = _call(_tools(pipe)["partner_run"], node_type="GPT Image 2", settings={"prompt": "a fox"})
        self.assertEqual(seen, ["C:/wf/partner_x.json"])
        self.assertTrue(out["ran"])
        self.assertEqual(out["run"]["outputs"], ["fox.png"])
        self.assertEqual(pipe._chain_output_paths, ["fox.png"])

    def test_a_build_error_never_runs(self):
        pipe = _stub()
        with mock.patch("agenty_core.tools.partner_run.prepare_partner_workflow",
                        return_value={"status": "error", "errors": ["'quality'='ultra' is not an option"]}), \
             mock.patch("src.executor.execute_workflow", side_effect=AssertionError("must not run")):
            out = _call(_tools(pipe)["partner_run"], node_type="x", settings={"quality": "ultra"})
        self.assertEqual(out["status"], "error")

    def test_run_false_returns_the_workflow_path(self):
        pipe = _stub()
        with mock.patch("agenty_core.tools.partner_run.prepare_partner_workflow", return_value=self.READY):
            out = _call(_tools(pipe)["partner_run"], node_type="x", run=False)
        self.assertEqual(out["workflow_path"], "C:/wf/partner_x.json")
        self.assertFalse(out["ran"])

    def test_a_successful_run_is_offered_to_the_local_template_store(self):
        pipe = _stub()
        with mock.patch("agenty_core.tools.partner_run.prepare_partner_workflow", return_value=self.READY), \
             mock.patch("src.executor.execute_workflow", _fake_executor({"C:/wf/partner_x.json": ["fox.png"]})), \
             mock.patch("src.utils.local_templates_hook.record", return_value="local_fox") as rec:
            _call(_tools(pipe)["partner_run"], node_type="OpenAIGPTImage1", name="fox", settings={"prompt": "a fox"})
        rec.assert_called_once()
        self.assertEqual(rec.call_args[0][0], "C:/wf/partner_x.json")
        self.assertIn("a fox", rec.call_args[0][1])


class Wiring(unittest.TestCase):
    def test_both_are_core_tools_wrapped_here_and_offered_by_the_pipeline(self):
        import src.tools as T
        for name in ("partner_run", "run_variants"):
            self.assertIn(name, T._SHARED_CORE_TOOLS)
        from src.pipeline import Pipeline
        names = {t.tool_name for t in Pipeline._build_delegation_tools(_stub())}
        self.assertTrue({"partner_run", "run_variants"} <= names)


if __name__ == "__main__":
    unittest.main()
