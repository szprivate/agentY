"""Build everything, submit nothing.

A hook chain is a piece of reasoning wearing a pile of paid API calls. Until now
the only way to find out whether a five-hook chain wired up the way you meant was
to run it and read the invoice afterwards.

A dry run keeps every step except the last one. The hooks are answered, the values
written, the variants built to disk as real workflow files — and where ComfyUI
would be handed the graph, each variant is answered with a **stand-in**: a path,
and no file. What makes it worth anything is that the chain does not stop there:
the hook whose directive is "take the references you just made and queue one video
per shot" still gets something where the references were, which is the half most
worth testing.

Two promises hold this together, and both are tested here: nothing is submitted,
by any route; and everything a real run would have said about the batch is still
said, in the same shape, so the agent's next move is the one it would really make.

    python -m unittest discover -s tests
"""

import asyncio
import json
import os
import unittest
from types import SimpleNamespace
from unittest import mock

from pipeline_stub import pipeline_stub, tools
from src.pipeline import Pipeline
from src.utils import dry_run


class StandInTest(unittest.TestCase):
    """What a graph would have produced, said in paths."""

    def setUp(self):
        dry_run.arm(True)
        self.addCleanup(dry_run.reset)

    def test_it_is_off_until_it_is_armed(self):
        dry_run.reset()
        self.assertFalse(dry_run.active())
        dry_run.arm(True)
        self.assertTrue(dry_run.active())

    def test_the_extension_follows_what_the_graph_makes(self):
        video = {"1": {"class_type": "KlingImage2VideoNode"}, "2": {"class_type": "SaveVideo"}}
        image = {"1": {"class_type": "KSampler"}, "2": {"class_type": "SaveImage"}}
        self.assertEqual(dry_run.media_kind(video), "video")
        self.assertEqual(dry_run.media_kind(image), "image")
        self.assertTrue(dry_run.stand_ins(video, "wf.json")[0].endswith(".mp4"))
        self.assertTrue(dry_run.stand_ins(image, "wf.json")[0].endswith(".png"))

    def test_each_writer_gets_the_kind_it_writes(self):
        """A reference sweep came back named like clips, because one video node
        anywhere in the graph made the whole thing "video"."""
        both = {"348": {"class_type": "ByteDanceSeedreamNodeV2"},
                "349": {"class_type": "SaveImage"},
                "283": {"class_type": "ByteDanceSeedanceNode"},
                "284": {"class_type": "SaveVideo"}}
        got = dry_run.stand_ins(both, "wf.json")
        self.assertEqual(sorted(p.rsplit(".", 1)[1] for p in got), ["mp4", "png"])

    def test_a_scoped_reference_stage_is_all_images(self):
        ref = {"348": {"class_type": "ByteDanceSeedreamNodeV2"},
               "349": {"class_type": "SaveImage"}}
        [only] = dry_run.stand_ins(ref, "wf.json", label="Ben")
        self.assertTrue(only.endswith(".png"), only)

    def test_a_stand_in_is_named_after_the_variant(self):
        """Five references are five different things; DRY-RUN_003.png says none of it."""
        [path] = dry_run.stand_ins({"1": {"class_type": "SaveImage"}}, "wf.json",
                                   label="Ben, grey suit, late 40s", index=3)
        self.assertIn("DRY-RUN_003", os.path.basename(path))
        self.assertIn("ben-grey-suit", path)
        self.assertEqual(dry_run.stands_for(path), "Ben, grey suit, late 40s")

    def test_no_file_is_ever_written(self):
        for p in dry_run.stand_ins({"1": {"class_type": "SaveImage"}}, "wf.json"):
            self.assertFalse(os.path.exists(p), p)

    def test_a_stand_in_is_recognised_by_the_registry_and_by_its_name(self):
        [path] = dry_run.stand_ins({"1": {"class_type": "SaveImage"}}, "wf.json")
        self.assertTrue(dry_run.is_stand_in(path))
        # Retyped by an agent with the other slash — still not a render.
        self.assertTrue(dry_run.is_stand_in(path.replace("\\", "/")))
        self.assertFalse(dry_run.is_stand_in(r"C:\output\ComfyUI_00042_.png"))
        self.assertFalse(dry_run.is_stand_in(""))

    def test_the_summary_names_every_graph_it_built(self):
        dry_run.record("C:/tmp/canvas_000.json", ["C:/tmp/DRY-RUN_001_ben.png"], label="Ben")
        dry_run.record("C:/tmp/canvas_001.json", ["C:/tmp/DRY-RUN_002_ana.png"], label="Ana")
        text = dry_run.summary()
        self.assertIn("2 workflow(s) built, nothing submitted", text)
        self.assertIn("canvas_000.json", text)
        self.assertIn("Ana", text)

    def test_a_dry_run_that_built_nothing_says_nothing(self):
        self.assertEqual(dry_run.summary(), "")

    def test_the_turn_disarms_it(self):
        dry_run.record("wf.json", ["x.png"])
        dry_run.reset()
        self.assertFalse(dry_run.active())
        self.assertEqual(dry_run.runs(), [])


class NothingIsSubmittedTest(unittest.TestCase):
    """The promise, checked at each door out of the process."""

    def setUp(self):
        dry_run.arm(True)
        self.addCleanup(dry_run.reset)
        # Filing the built graph is exercised in GraphsWhatItBuiltTest; here it
        # would only spend a ComfyUI connection timeout per test.
        p = mock.patch("agenty_core.tools.comfyui.open_workflow_in_canvas")
        self.graphed = p.start()
        self.addCleanup(p.stop)

    @staticmethod
    def _apply(pipe, resolutions, **kw):
        return asyncio.run(tools(pipe)["apply_canvas_hooks"](resolutions, **kw))

    def _pipe(self):
        return pipeline_stub(
            _dry_run=True,
            _canvas_base_prompt={
                "1": {"class_type": "CLIPTextEncode", "inputs": {"text": ""}},
                "2": {"class_type": "SaveImage", "inputs": {"images": ["1", 0]}},
            })

    def test_the_batch_is_built_and_queued_nowhere(self):
        with mock.patch("src.utils.workflow_signal.append_workflow_path") as queued:
            out = json.loads(self._apply(self._pipe(), [
                {"target_node_id": "1", "param": "text", "mode": "value_list",
                 "values": ["a cat", "a dog"]}]))
        queued.assert_not_called()
        self.assertEqual(out["status"], "dry_run")
        self.assertEqual(out["count"], 2)

    def test_the_graphs_it_built_are_real_files_on_disk(self):
        """The whole point of building them: you can open one and check the wiring."""
        from pathlib import Path
        out = json.loads(self._apply(self._pipe(), [
            {"target_node_id": "1", "param": "text", "mode": "value_list",
             "values": ["a cat", "a dog"]}]))
        built = [json.loads(Path(v["workflow"]).read_text(encoding="utf-8"))
                 for v in out["variants"]]
        for graph in built:
            self.assertIn("2", graph)          # the whole graph, not only what changed
        self.assertEqual([g["1"]["inputs"]["text"] for g in built], ["a cat", "a dog"])

    def test_every_variant_gets_stand_ins_paired_with_what_made_it(self):
        out = json.loads(self._apply(self._pipe(), [
            {"target_node_id": "1", "param": "text", "mode": "value_list",
             "values": ["Ben, grey suit", "Ana, red coat"]}]))
        first, second = out["variants"]
        self.assertEqual(first["made_from"]["1.text"], "Ben, grey suit")
        self.assertIn("ben-grey-suit", first["outputs"][0])
        self.assertIn("ana-red-coat", second["outputs"][0])
        self.assertEqual(len(out["outputs"]), 2)

    def test_run_now_costs_nothing_and_never_executes(self):
        """A conditional directive is only testable if run_now still answers."""
        with mock.patch("src.pipeline._execute_workflows_batch") as ex:
            out = json.loads(self._apply(self._pipe(), [
                {"target_node_id": "1", "param": "text", "mode": "value_list",
                 "values": ["a cat"]}], run_now=True))
        ex.assert_not_called()
        self.assertEqual(out["status"], "dry_run")
        self.assertTrue(out["outputs"])

    def test_the_answer_tells_the_agent_to_carry_on_rather_than_stop(self):
        out = json.loads(self._apply(self._pipe(), []))
        self.assertIn("as if every variant had succeeded", out["message"])
        self.assertIn("stand-ins", out["message"])
        self.assertIn("do not call stop_hook_run", out["message"].lower())

    def test_a_chained_stage_is_answered_with_paths(self):
        """run_workflow_now exists to feed the next stage; a refusal ends the chain."""
        import tempfile
        from pathlib import Path
        wf = Path(tempfile.mkdtemp()) / "stage1.json"
        wf.write_text(json.dumps({"1": {"class_type": "SaveImage"}}), encoding="utf-8")
        pipe = self._pipe()
        with mock.patch("src.executor.execute_workflow") as ex:
            out = json.loads(asyncio.run(tools(pipe)["run_workflow_now"](str(wf))))
        ex.assert_not_called()
        self.assertEqual(out["status"], "dry_run")
        self.assertTrue(dry_run.is_stand_in(out["outputs"][0]))

    def test_an_iterate_step_is_refused_rather_than_faked(self):
        """Its result is written back into the user's own node and kept across turns."""
        pipe = pipeline_stub(_dry_run=True, _canvas_hooks=[
            {"hook_node_id": "9", "purpose": "iterate", "anchors": [],
             "directive": "refine"}])
        out = json.loads(asyncio.run(tools(pipe)["iterate_step"]("warmer light")))
        self.assertIn("DRY RUN", out["error"])
        self.assertIn("full run", out["error"])

    def test_a_signalled_workflow_says_dry_run_instead_of_ready(self):
        import tempfile
        from pathlib import Path
        from src.tools.workflow_handoff import signal_workflow_ready
        wf = Path(tempfile.mkdtemp()) / "assembled.json"
        wf.write_text("{}", encoding="utf-8")
        with mock.patch("src.utils.workflow_signal.execution_hold", return_value=None):
            out = json.loads(signal_workflow_ready(str(wf)))
        self.assertEqual(out["status"], "dry_run")
        self.assertIn("NOT be submitted", out["message"])


class GraphsWhatItBuiltTest(unittest.TestCase):
    """A dry run that shows you nothing built the graph for nobody.

    A real run files what it submits into the Workflows sidebar on its way to
    ``/prompt`` — inside the executor, which a dry run never reaches. So it files
    them here instead, and it does NOT swap them onto the open canvas unless the
    user asked for that: the graph they have open is the one under test.
    """

    def setUp(self):
        dry_run.arm(True)
        self.addCleanup(dry_run.reset)
        p = mock.patch("agenty_core.tools.comfyui.open_workflow_in_canvas")
        self.canvas = p.start()
        self.addCleanup(p.stop)

    def _sweep(self, autoload):
        pipe = pipeline_stub(
            _dry_run=True,
            _canvas_base_prompt={"1": {"class_type": "CLIPTextEncode", "inputs": {"text": ""}},
                                 "2": {"class_type": "SaveImage", "inputs": {}}})
        with mock.patch("src.executor._autoload_workflows_into_canvas", return_value=autoload):
            return json.loads(asyncio.run(tools(pipe)["apply_canvas_hooks"]([
                {"target_node_id": "1", "param": "text", "mode": "value_list",
                 "values": ["a", "b", "c"]}])))

    def test_it_files_every_variant_under_its_own_name(self):
        """Filing one representative hid the rest.

        The variants of a reference sweep are three different characters, not one
        graph three times, and "did it make them all?" is the first thing anyone
        checks — so all three are filed, each named after what makes it itself.
        """
        out = self._sweep(False)
        self.assertEqual(out["count"], 3)
        self.assertEqual(self.canvas.call_count, 3)
        self.assertEqual(len(out["graphed_as"]), 3)
        self.assertEqual(len(set(out["graphed_as"])), 3, "two variants filed as one name")
        for name in out["graphed_as"]:
            self.assertTrue(name.startswith("agent/dryrun_"), name)

    def test_it_does_not_take_away_the_graph_under_test(self):
        self._sweep(False)
        self.assertIs(self.canvas.call_args.kwargs["push_to_canvas"], False)

    def test_a_user_who_asked_for_auto_graphing_still_gets_it(self):
        self._sweep(True)
        self.assertIs(self.canvas.call_args.kwargs["push_to_canvas"], True)

    def test_a_long_chain_cannot_bury_the_sidebar(self):
        pipe = pipeline_stub(_dry_run=True)
        with mock.patch("src.executor._autoload_workflows_into_canvas", return_value=False):
            for i in range(Pipeline._DRY_GRAPH_CAP + 4):
                pipe._graph_dry_build(f"stage{i}.json")
        self.assertEqual(self.canvas.call_count, Pipeline._DRY_GRAPH_CAP)

    def test_a_comfyui_that_is_down_costs_the_run_nothing(self):
        self.canvas.side_effect = RuntimeError("connection refused")
        out = self._sweep(False)
        self.assertEqual(out["status"], "dry_run")
        self.assertEqual(out["graphed_as"], [])
        self.assertEqual(out["count"], 3, "the graphs are still BUILT when filing fails")


class TheChainKeepsGoingTest(unittest.TestCase):
    """A stand-in has to survive every check a real output would pass.

    The video hook's whole job is to take what the reference hook produced and
    queue one generation per shot. In a dry run what it receives is stand-ins — so
    any check that treats "no file on disk" as an error refuses the batch and kills
    the chain at exactly the hook the dry run existed to exercise. Which is what
    happened: the collector rejected its own lines.
    """

    def setUp(self):
        dry_run.arm(True)
        self.addCleanup(dry_run.reset)

    def _collector(self, paths):
        return {"60": {"class_type": "AgentYImageCollector",
                       "inputs": {"files": "\n".join(paths)}}}

    def test_a_collector_full_of_stand_ins_is_not_a_broken_list(self):
        from src.utils.canvas_hooks import missing_collector_files
        refs = dry_run.stand_ins({"1": {"class_type": "SaveImage"}}, "wf.json",
                                 label="Ben") + \
            dry_run.stand_ins({"1": {"class_type": "SaveImage"}}, "wf2.json",
                              label="Ana", index=2)
        self.assertEqual(missing_collector_files(self._collector(refs)), [])

    def test_a_genuinely_wrong_path_is_still_caught(self):
        """The check earns its keep on real runs; the exemption is only for these."""
        from src.utils.canvas_hooks import missing_collector_files
        bad = missing_collector_files(self._collector([r"C:\nope\ben.png"]))
        self.assertEqual(len(bad), 1)
        self.assertEqual(bad[0]["missing"], [r"C:\nope\ben.png"])

    def test_the_batch_is_not_refused_over_them(self):
        pipe = pipeline_stub()
        self.assertIsNone(pipe._collector_refusal([self._collector(
            dry_run.stand_ins({"1": {"class_type": "SaveImage"}}, "wf.json"))]))


class LeavesNothingBehindTest(unittest.TestCase):
    """A rehearsal that changes the next real run is not a rehearsal."""

    def test_a_memorizing_hook_does_not_store_what_it_answered(self):
        """It would be derived from a stand-in, and served silently to a real run."""
        hook = {"hook_node_id": "5", "purpose": "text", "directive": "describe it",
                "anchors": [], "targets": [], "memorize": True, "_cache_key": "abc123"}
        for dry, expected in ((True, False), (False, True)):
            with self.subTest(dry_run=dry):
                pipe = pipeline_stub(_dry_run=dry, _canvas_hooks=[hook],
                                     _canvas_base_prompt={})
                with mock.patch("src.utils.hook_cache.write") as wrote, \
                     mock.patch("src.utils.hook_cache.memorizing", return_value=True), \
                     mock.patch("src.utils.canvas_patch.push"):
                    asyncio.run(tools(pipe)["place_canvas_text"]("5", "a wide grey room"))
                self.assertEqual(wrote.called, expected)


class ToolsThatWouldOpenOneTest(unittest.TestCase):
    """A stand-in must read as a skipped step, never as a failure to repair."""

    def setUp(self):
        dry_run.arm(True)
        self.addCleanup(dry_run.reset)
        [self.path] = dry_run.stand_ins({"1": {"class_type": "SaveImage"}}, "wf.json",
                                        label="the hero sheet")

    def test_analysis_answers_instead_of_reporting_a_missing_file(self):
        from src.tools.image_handling import analyze_image
        out = analyze_image(file_path=self.path)
        self.assertEqual(out["status"], "ok")
        text = out["content"][0]["text"]
        self.assertIn("DRY RUN", text)
        self.assertIn("the hero sheet", text)

    def test_video_analysis_does_the_same(self):
        from src.tools.video_handling import analyze_video
        out = analyze_video(file_path=self.path)
        self.assertEqual(out["status"], "ok")
        self.assertIn("DRY RUN", out["content"][0]["text"])

    def test_staging_one_answers_with_the_name_a_real_upload_would(self):
        from src.tools.image_handling import _upload_one
        out = _upload_one(self.path)
        self.assertNotIn("error", out)
        self.assertEqual(out["name"], os.path.basename(self.path))
        self.assertIn("DRY RUN", out["note"])

    def test_a_real_file_is_untouched_by_any_of_this(self):
        from src.tools.image_handling import analyze_image
        out = analyze_image(file_path="C:/nowhere/real.png")
        self.assertEqual(out["status"], "error")


class TheTurnTest(unittest.TestCase):
    """How the flag reaches the tools, and what the user is left holding."""

    def test_the_orchestrator_is_told_before_anything_else(self):
        pipe = pipeline_stub(_dry_run=True, _canvas_hooks=[], _canvas_base_prompt=None,
                             _session=SimpleNamespace(current_output_paths=[],
                                                      last_user_input_images=[],
                                                      generated_images=[]))
        for name in ("_extract_hard_constraints", "_describe_canvas_selection",
                     "_get_memory_context", "_get_project_memory_context",
                     "_detect_plan_approval", "_format_image_gallery",
                     "_annotate_attachments", "_prepend_gallery"):
            setattr(pipe, name, mock.Mock(return_value="" if name != "_extract_hard_constraints"
                                          else []))
        pipe._detect_plan_approval = mock.Mock(return_value=None)
        pipe._prepend_gallery = lambda t: t
        pipe._annotate_attachments = lambda a, b: str(a)
        built = Pipeline._build_orchestrator_input(pipe, "run it", "run it")
        self.assertTrue(built.startswith("[DRY RUN"), built[:80])
        self.assertIn("no graph is submitted", built.replace("\n", " "))

    def test_the_partial_lives_in_a_file_not_in_the_code(self):
        from src.pipeline import _ORCH_PARTIALS_DIR
        self.assertTrue((_ORCH_PARTIALS_DIR / "dry_run.md").is_file())

    def test_it_travels_from_the_panel_to_the_pipeline(self):
        import inspect
        from src.utils import agentY_server
        self.assertIn('body.get("dry_run")', inspect.getsource(agentY_server))
        self.assertIn("dry_run=dry_run", inspect.getsource(agentY_server._run_pipeline_turn))
        self.assertIn("dry_run: bool = False",
                      inspect.getsource(Pipeline.stream_async))

    def test_the_end_of_turn_batch_never_reaches_the_executor(self):
        """A signalled workflow arrives there without passing a tool that could stop it."""
        import inspect
        src = inspect.getsource(Pipeline._astream_orchestrator)
        self.assertIn("if self._dry_run and workflow_paths:", src)
        self.assertIn("workflow_paths = []", src)

    def test_the_user_is_told_what_would_have_happened(self):
        import inspect
        src = inspect.getsource(Pipeline._astream_orchestrator)
        self.assertIn("_dry_mod.summary()", src)


if __name__ == "__main__":
    unittest.main()
