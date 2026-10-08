"""Output files follow the context on the canvas.

With a sequence and a shot named - by an AYON context node, else an agentY
context node - every file a run produces is saved as
<sequence>/<shot>_<suffix>_v###.<ext> under ComfyUI's output folder. The agent
supplies the suffix, and can name another shot or asset for part of a run.
"""

import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from agenty_core.utils import turn_scope

from pipeline_stub import pipeline_stub, tools
from src.utils import output_context as oc

AYON = {"class_type": "AYONContext", "inputs": {"context": json.dumps(
    {"context": {"folder_path": "/03_sequences/spec/spec_0210"}, "instances": []})}}
OURS = {"class_type": "AgentYContext", "inputs": {"sequence": "characters", "shot": "male lead"}}


def _graph(prefix="agent/images/astronaut_dune", cls="SaveImage"):
    return {"1": {"class_type": "KSampler", "inputs": {"seed": 1}},
            "9": {"class_type": cls, "inputs": {"images": ["1", 0], "filename_prefix": prefix}}}


class Scoped(unittest.TestCase):

    def setUp(self):
        token = turn_scope.enter(turn_scope.Scope("req", "thread"))
        self.addCleanup(turn_scope.leave, token)


class ReadingTheCanvas(unittest.TestCase):

    def test_an_agenty_context_node_names_sequence_and_shot(self):
        self.assertEqual(oc.from_canvas({"5": OURS}),
                         {"sequence": "characters", "shot": "male_lead", "source": "agentY context node 5"})

    def test_an_ayon_context_wins_and_is_split_like_bepics_get_path(self):
        ctx = oc.from_canvas({"5": OURS, "6": AYON})
        self.assertEqual((ctx["sequence"], ctx["shot"]), ("spec", "spec_0210"))
        self.assertEqual(ctx["source"], "AYON context node")

    def test_the_publish_shape_takes_the_active_instance(self):
        publish = {"instances": [{"folderPath": "/seq/a/a_010"}, {"folderPath": "/seq/b/b_020", "active": True}]}
        self.assertEqual(oc.ayon_folder_path(publish), "/seq/b/b_020")
        self.assertEqual(oc.split_folder_path("only_shot"), ("", "only_shot"))

    def test_a_graph_that_names_nothing_has_no_context(self):
        self.assertEqual(oc.from_canvas(_graph()), {})
        self.assertEqual(oc.from_canvas({"5": {"class_type": "AgentYContext", "inputs": {"sequence": "", "shot": ""}}}), {})
        self.assertEqual(oc.from_canvas(None), {})


class BeforeARun(Scoped):

    def test_without_a_context_nothing_changes(self):
        graph = _graph()
        self.assertEqual(oc.stamp(graph), {})
        self.assertEqual(graph["9"]["inputs"]["filename_prefix"], "agent/images/astronaut_dune")
        self.assertEqual(oc.describe(), "")

    def test_the_saver_is_named_for_the_context_and_what_the_builder_called_it(self):
        oc.set_current(sequence="spec", shot="spec_0210")
        graph = _graph()
        self.assertEqual(oc.stamp(graph), {"9": "spec/spec_0210_astronaut_dune"})

    def test_a_generic_stem_becomes_the_kind_of_file(self):
        oc.set_current(sequence="spec", shot="spec_0210")
        image, video = _graph("ComfyUI"), _graph("agent/videos/video", "SaveVideo")
        oc.stamp(image), oc.stamp(video)
        self.assertEqual(image["9"]["inputs"]["filename_prefix"], "spec/spec_0210_image")
        self.assertEqual(video["9"]["inputs"]["filename_prefix"], "spec/spec_0210_video")

    def test_the_agents_suffix_is_used_when_it_gave_one(self):
        oc.set_current(sequence="spec", shot="spec_0210", suffix="start frame")
        graph = _graph()
        oc.stamp(graph)
        self.assertEqual(graph["9"]["inputs"]["filename_prefix"], "spec/spec_0210_start_frame")

    def test_a_prefix_that_names_its_own_folder_or_is_wired_is_left_alone(self):
        oc.set_current(sequence="spec", shot="spec_0210")
        own, wired = _graph("characters/woman_ref"), _graph()
        wired["9"]["inputs"]["filename_prefix"] = ["4", 0]
        self.assertEqual(oc.stamp(own), {})
        self.assertEqual(oc.stamp(wired), {})
        self.assertEqual(own["9"]["inputs"]["filename_prefix"], "characters/woman_ref")

    def test_a_shot_without_a_sequence_goes_to_the_output_root(self):
        self.assertEqual(oc.prefix_for({"sequence": "", "shot": "sh010"}, "ref"), "sh010_ref")

    def test_the_agent_is_told_the_rule_and_how_to_alternate_it(self):
        oc.set_current(sequence="spec", shot="spec_0210", source="AYON context node")
        note = oc.describe()
        self.assertIn("spec/spec_0210_startframe_v001.png", note)
        self.assertIn("name_outputs(suffix=", note)
        self.assertIn("SEVERAL shots", note)


class AFileComesBack(Scoped):

    def setUp(self):
        super().setUp()
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.dir = Path(tmp.name) / "spec"
        self.dir.mkdir()
        oc.set_current(sequence="spec", shot="spec_0210", suffix="startframe")
        oc.stamp(_graph())

    def _saved(self, name):
        p = self.dir / name
        p.write_text("x", encoding="utf-8")
        return p

    def test_comfyuis_counter_becomes_the_version(self):
        out = oc.finalize(self._saved("spec_0210_startframe_00001_.png"))
        self.assertEqual(out.name, "spec_0210_startframe_v001.png")
        self.assertTrue(out.exists())

    def test_the_version_is_the_next_free_one(self):
        self._saved("spec_0210_startframe_v001.png")
        self._saved("spec_0210_startframe_v007.png")
        out = oc.finalize(self._saved("spec_0210_startframe_00001_.png"))
        self.assertEqual(out.name, "spec_0210_startframe_v008.png")

    def test_each_file_of_a_batch_gets_its_own_version(self):
        a = oc.finalize(self._saved("spec_0210_startframe_00001_.png"))
        b = oc.finalize(self._saved("spec_0210_startframe_00002_.png"))
        self.assertEqual([a.name, b.name], ["spec_0210_startframe_v001.png", "spec_0210_startframe_v002.png"])

    def test_a_video_and_its_preview_share_a_version(self):
        video = oc.finalize(self._saved("spec_0210_startframe_00001.mp4"))
        frame = oc.finalize(self._saved("spec_0210_startframe_00001.png"))
        self.assertEqual([video.name, frame.name],
                         ["spec_0210_startframe_v001.mp4", "spec_0210_startframe_v001.png"])

    def test_a_file_this_turn_did_not_name_keeps_its_name(self):
        other = self._saved("somebody_elses_00001_.png")
        self.assertEqual(oc.finalize(other), other)
        elsewhere = Path(self.dir.parent) / "spec_0210_startframe_00001_.png"
        elsewhere.write_text("x", encoding="utf-8")
        self.assertEqual(oc.finalize(elsewhere), elsewhere, "same name, wrong folder")

    def test_an_already_versioned_file_is_not_renamed_again(self):
        done = oc.finalize(self._saved("spec_0210_startframe_00001_.png"))
        self.assertEqual(oc.finalize(done), done)


class TheTool(Scoped):

    def setUp(self):
        super().setUp()
        from src.utils.canvas_patch import clear
        clear()
        self.addCleanup(clear)
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.tmp = Path(tmp.name)

    def _wf(self, name="a.json"):
        p = self.tmp / name
        p.write_text(json.dumps(_graph()), encoding="utf-8")
        return p

    def _call(self, pipe=None, **kw):
        return json.loads(asyncio.run(tools(pipe or pipeline_stub())["name_outputs"](**kw)))

    def _prefix(self, path):
        return json.loads(path.read_text(encoding="utf-8"))["9"]["inputs"]["filename_prefix"]

    def test_one_request_two_assets(self):
        """ "create images of the main male character and the woman" """
        oc.set_current(sequence="characters", shot="cast")
        man, woman = self._wf("man.json"), self._wf("woman.json")
        self._call(suffix="ref", shot="male lead", workflow_path=str(man))
        out = self._call(suffix="ref", shot="woman", workflow_path=str(woman))
        self.assertEqual(self._prefix(man), "characters/male_lead_ref")
        self.assertEqual(self._prefix(woman), "characters/woman_ref")
        self.assertEqual(out["files"], "characters/woman_ref_v###.<ext>")
        self.assertEqual(oc.current()["shot"], "cast", "naming one workflow does not move the context")

    def test_named_workflows_keep_their_names_at_submission_and_get_versions(self):
        oc.set_current(sequence="characters", shot="cast")
        wf = self._wf()
        self._call(suffix="ref", shot="woman", workflow_path=str(wf))
        graph = json.loads(wf.read_text(encoding="utf-8"))
        self.assertEqual(oc.stamp(graph), {})
        folder = self.tmp / "characters"
        folder.mkdir()
        saved = folder / "woman_ref_00001_.png"
        saved.write_text("x", encoding="utf-8")
        self.assertEqual(oc.finalize(saved).name, "woman_ref_v001.png")

    def test_without_a_workflow_it_sets_what_follows(self):
        oc.set_current(sequence="spec", shot="spec_0210")
        out = self._call(suffix="end frame", shot="spec_0220")
        self.assertEqual(out["files"], "spec/spec_0220_end_frame_v###.<ext>")
        graph = _graph()
        oc.stamp(graph)
        self.assertEqual(graph["9"]["inputs"]["filename_prefix"], "spec/spec_0220_end_frame")

    def test_it_works_with_no_context_node_when_the_agent_names_the_shot(self):
        wf = self._wf()
        self.assertIn("no sequence or shot", self._call(suffix="ref", workflow_path=str(wf))["error"])
        self._call(suffix="ref", shot="hero", sequence="assets", workflow_path=str(wf))
        self.assertEqual(self._prefix(wf), "assets/hero_ref")
        self.assertTrue(oc.active(), "so the file is versioned when it comes back")

    def test_a_workflow_on_the_canvas_is_renamed_there_too(self):
        from src.utils.canvas_patch import drain
        oc.set_current(sequence="spec", shot="spec_0210")
        wf = self._wf()
        live = {"40": {"class_type": "KSampler", "inputs": {}},
                "41": {"class_type": "SaveImage", "inputs": {"filename_prefix": "agent/images/x"}}}
        pipe = pipeline_stub(_canvas_graph=live)
        pipe._session.inserted_workflows[str(wf.resolve())] = ["40", "41"]
        out = self._call(pipe, suffix="startframe", workflow_path=str(wf))
        self.assertEqual(out["canvas_nodes"], ["41"])
        self.assertEqual(live["41"]["inputs"]["filename_prefix"], "spec/spec_0210_startframe")
        self.assertEqual([p["params"] for p in drain()], [{"filename_prefix": "spec/spec_0210_startframe"}])


class InTheExecutor(unittest.TestCase):

    def test_every_submission_and_every_collected_file_passes_through(self):
        src = (Path(__file__).resolve().parents[1] / "src" / "executor.py").read_text(encoding="utf-8")
        submit = src.split("def _submit_workflow(", 1)[1].split("\ndef ", 1)[0]
        self.assertIn("output_context.stamp(workflow)", submit)
        self.assertLess(submit.index("output_context.stamp(workflow)"), submit.index('client.post("/prompt"'))
        self.assertIn("resolved = output_context.finalize(resolved)", src)


if __name__ == "__main__":
    unittest.main()
