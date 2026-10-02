"""A template's description is what it is found by — so it is never filler.

The researcher sees a template's name and the first 60 characters of its
description. The old writer stored "Local generation via ComfyUI Model. …
Processes and generates content using ComfyUI workflows." whenever its model call
failed, which on a machine configured by tier was always. Pinned here: no answer
means no description; the agent's own line is used when it gives one; node pack
examples are described once per file; and the utility client follows the tiers.

    python -m unittest discover -s tests
"""
import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from pipeline_stub import pipeline_stub, tools
from src.utils import workflow_admin as wa
from src.utils import workflow_describe as wd

GOOD = "[Local] Segmentation via SAM3. 1 image -> 1 image. Text-prompted masks."
API = {"1": {"class_type": "KSampler", "inputs": {}}}


class Clean(unittest.TestCase):

    def test_a_proper_line_passes_and_is_tidied(self):
        self.assertEqual(wd.clean(f'  "{GOOD}"\n'), GOOD)

    def test_what_is_not_a_description_becomes_nothing(self):
        for bad in ("", "ok", "Sure! Here is the description you asked for, hope it helps you a lot.",
                    "[Local] generation via ComfyUI Model. text input → 1 image output. Processes and "
                    "generates content using ComfyUI workflows.",
                    "[Local] " + "x" * 500):
            self.assertEqual(wd.clean(bad), "", bad[:40])


class Describe(unittest.TestCase):

    def test_the_model_is_given_the_graphs_facts(self):
        seen = []
        out = wd.describe(API, "my_wf", pack="Pack", workflow="basic",
                          ask=lambda p: seen.append(p) or GOOD)
        self.assertEqual(out, GOOD)
        self.assertIn("my_wf", seen[0])
        self.assertIn("custom node pack Pack", seen[0])

    def test_a_failing_or_rambling_model_yields_nothing(self):
        def boom(_p):
            raise RuntimeError("404")
        self.assertEqual(wd.describe(API, "x", ask=boom), "")
        self.assertEqual(wd.describe(API, "x", ask=lambda p: "I cannot tell."), "")


class RegisterWorkflow(unittest.TestCase):

    def _register(self, **kw):
        with tempfile.TemporaryDirectory() as d, \
                mock.patch.object(wa, "_templates_dir", return_value=Path(d)), \
                mock.patch.object(wa, "parse_workflow") as parse, \
                mock.patch.object(wa, "_generate_description", return_value="generated") as gen:
            res = wa.register_workflow(API, "my_wf", regenerate=False, **kw)
        return res, parse, gen

    def test_the_callers_description_is_used_and_nothing_is_generated(self):
        res, parse, gen = self._register(description="  [Local] Upscale via\n ESRGAN.  ")
        self.assertEqual(res["description"], "[Local] Upscale via ESRGAN.")
        self.assertEqual(parse.call_args.kwargs["description"], "[Local] Upscale via ESRGAN.")
        gen.assert_not_called()

    def test_without_one_it_is_generated(self):
        res, _parse, gen = self._register()
        self.assertEqual(res["description"], "generated")
        gen.assert_called_once()

    def test_the_generator_stores_nothing_when_the_model_fails(self):
        with mock.patch.object(wd, "describe", return_value=""):
            self.assertEqual(wa._generate_description(API, "x"), "")

    def test_the_agents_tool_passes_its_description_on(self):
        pipe = pipeline_stub(_canvas_base_prompt=API)
        with mock.patch.object(wa, "register_workflow",
                               return_value={"name": "w", "template_file": "f", "description": "d",
                                             "recipes": {}}) as reg:
            asyncio.run(tools(pipe)["add_canvas_workflow"](name="w", description="[Local] X via Y."))
        self.assertEqual(reg.call_args.kwargs["description"], "[Local] X via Y.")


class NodePackExamples(unittest.TestCase):

    def setUp(self):
        from agenty_core.templates_sync import NODE_PACK_DIR
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.dir = Path(self.tmp.name) / NODE_PACK_DIR
        self.dir.mkdir(parents=True)
        for n in ("A__one", "A__two"):
            (self.dir / f"{n}.json").write_text(json.dumps(API), "utf-8")
        (self.dir / "index.json").write_text(json.dumps([{"moduleName": "A", "templates": [
            {"name": "A__one", "title": "one", "description": "graph", "description_source": "graph"},
            {"name": "A__two", "title": "two", "description": "graph", "description_source": "graph"},
        ]}]), "utf-8")

    def _entries(self):
        return {t["name"]: t for g in json.loads((self.dir / "index.json").read_text("utf-8"))
                for t in g["templates"]}

    def test_each_example_is_described_once(self):
        calls = []

        def one(data, name, *, pack="", workflow=""):
            calls.append((name, pack, workflow))
            return GOOD if name == "A__one" else ""

        res = wd.describe_node_pack_examples(self.tmp.name, describe_one=one, workers=1)
        self.assertEqual((res["written"], res["failed"]), (["A__one"], ["A__two"]))
        got = self._entries()
        self.assertEqual((got["A__one"]["description"], got["A__one"]["description_source"]), (GOOD, "llm"))
        self.assertEqual(got["A__two"]["description"], "graph")          # kept, tried again later
        self.assertIn(("A__one", "A", "one"), calls)

        calls.clear()
        wd.describe_node_pack_examples(self.tmp.name, describe_one=one, workers=1)
        self.assertEqual([c[0] for c in calls], ["A__two"])              # the written one is skipped

    def test_a_changed_file_is_described_again(self):
        one = lambda data, name, *, pack="", workflow="": GOOD  # noqa: E731
        wd.describe_node_pack_examples(self.tmp.name, describe_one=one, workers=1)
        (self.dir / "A__one.json").write_text(json.dumps({"2": {"class_type": "X", "inputs": {}}}), "utf-8")
        res = wd.describe_node_pack_examples(self.tmp.name, describe_one=one, workers=1)
        self.assertEqual(res["written"], ["A__one"])

    def test_no_mirror_is_not_an_error(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertEqual(wd.describe_node_pack_examples(d)["written"], [])


class UtilityClientFollowsTheTiers(unittest.TestCase):

    def test_a_blank_role_inherits_its_tier(self):
        from src.utils import llm_functions as lf
        with mock.patch("src.agent.role_model", return_value="dashscope,qwen-flash"):
            llm = lf.LLMFunctions.from_settings()
        self.assertEqual((llm.provider, llm.model), ("dashscope", "qwen-flash"))


class Startup(unittest.TestCase):

    def test_it_can_be_switched_off_and_a_failure_is_not_fatal(self):
        from src import agenty_ui_server as U
        calls = []
        self.assertIsNone(U._describe_node_pack_examples(lambda: calls.append(1),
                                                         settings={"describe_node_pack_examples": False}))
        self.assertIsNone(U._describe_node_pack_examples(lambda: calls.append(1),
                                                         settings={"sync_node_pack_examples": False}))
        self.assertEqual(calls, [])

        def boom():
            raise RuntimeError("no model")
        self.assertIsNone(U._describe_node_pack_examples(boom, settings={}))

    def test_nothing_written_rebuilds_nothing(self):
        from src import agenty_ui_server as U
        with mock.patch("src.utils.workflow_admin.regenerate_recipes") as regen:
            U._describe_node_pack_examples(lambda: {"written": [], "failed": ["a"]}, settings={})
        regen.assert_not_called()


if __name__ == "__main__":
    unittest.main()
