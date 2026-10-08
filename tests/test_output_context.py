"""Output files follow the context on the canvas.

With a sequence and a shot named - by an AYON context node, else an agentY
context node - every file a run produces is saved in bEpic's layout,
<Sequence>/<Shot>/<images|videos>/v###/<shot>_v###_<suffix>, under ComfyUI's
output folder. The agent supplies the suffix, and can name another shot or
asset for part of a run.

The whole name goes into the save node before the run. An earlier version
saved first and renamed afterwards, which left ComfyUI's history pointing at a
file that was no longer there.
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
    """A turn, with an empty output folder of its own."""

    def setUp(self):
        token = turn_scope.enter(turn_scope.Scope("req", "thread"))
        self.addCleanup(turn_scope.leave, token)
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.out = Path(tmp.name)
        self.enterContext(mock.patch.object(oc, "_output_root", lambda: self.out))

    def _existing(self, rel):
        p = self.out / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("x", encoding="utf-8")


class ReadingTheCanvas(unittest.TestCase):

    def test_an_agenty_context_node_names_sequence_and_shot(self):
        self.assertEqual(oc.from_canvas({"5": OURS}),
                         {"sequence": "characters", "shot": "male_lead", "source": "agentY context node 5"})

    def test_an_ayon_context_wins_and_is_split_like_bepics_get_path(self):
        ctx = oc.from_canvas({"5": OURS, "6": AYON})
        self.assertEqual((ctx["sequence"], ctx["shot"]), ("spec", "spec_0210"))
        self.assertEqual(ctx["source"], "AYON context node")

    def test_the_node_the_ayon_addon_writes(self):
        """ayon_comfyui imprints {"context", "instances", "containers"} into the
        `ayon_context_info` widget of an "AYON Context" node; the folder is on
        the instances, as camelCase `folderPath`."""
        info = {"context": {"publish_attributes": {}},
                "instances": [{"productType": "image", "folderPath": "/03_sequences/spec/spec_0210",
                               "task": "comp", "active": True, "instance_id": "abc"}],
                "containers": []}
        graph = {"12": {"class_type": "AYON Context", "inputs": {"ayon_context_info": json.dumps(info)}},
                 "9": {"class_type": "SaveImage", "inputs": {"filename_prefix": "ComfyUI"}}}
        self.assertEqual(oc.from_canvas(graph),
                         {"sequence": "spec", "shot": "spec_0210", "source": "AYON context node"})

    def test_an_ayon_node_with_nothing_set_yet_falls_back_to_the_agenty_one(self):
        empty = {"class_type": "AYON Context", "inputs": {"ayon_context_info": ""}}
        bare = {"class_type": "AYON Context", "inputs": {"ayon_context_info": json.dumps(
            {"context": {}, "instances": [], "containers": []})}}
        for node in (empty, bare):
            self.assertEqual(oc.from_canvas({"12": node, "5": OURS})["source"], "agentY context node 5")

    def test_the_publish_shape_takes_the_active_instance(self):
        publish = {"instances": [{"folderPath": "/seq/a/a_010"}, {"folderPath": "/seq/b/b_020", "active": True}]}
        self.assertEqual(oc.ayon_folder_path(publish), "/seq/b/b_020")
        self.assertEqual(oc.split_folder_path("only_shot"), ("", "only_shot"))

    def test_a_graph_that_names_nothing_has_no_context(self):
        self.assertEqual(oc.from_canvas(_graph()), {})
        self.assertEqual(oc.from_canvas({"5": {"class_type": "AgentYContext", "inputs": {"sequence": "", "shot": ""}}}), {})
        self.assertEqual(oc.from_canvas(None), {})


class TheLayout(unittest.TestCase):

    def test_it_is_the_path_bepics_set_path_builds(self):
        """_bepic_build_paths("spec", "SPEC_0210", 12): pathImages + suffix."""
        ctx = {"sequence": "spec", "shot": "SPEC_0210"}
        self.assertEqual(oc.prefix_for(ctx, "startframe", "images", 12),
                         "spec/SPEC_0210/images/v012/spec_0210_v012_startframe")
        self.assertEqual(oc.prefix_for(ctx, "startframe", "videos", 12),
                         "spec/SPEC_0210/videos/v012/spec_0210_v012_startframe")

    def test_a_shot_without_a_sequence_sits_in_the_output_root(self):
        self.assertEqual(oc.prefix_for({"sequence": "", "shot": "sh010"}, "ref"),
                         "sh010/images/v001/sh010_v001_ref")


class BeforeARun(Scoped):

    def test_without_a_context_nothing_changes(self):
        graph = _graph()
        self.assertEqual(oc.stamp(graph), {})
        self.assertEqual(graph["9"]["inputs"]["filename_prefix"], "agent/images/astronaut_dune")
        self.assertEqual(oc.describe(), "")

    def test_the_save_node_gets_the_whole_name_version_included(self):
        oc.set_current(sequence="spec", shot="spec_0210")
        graph = _graph()
        self.assertEqual(oc.stamp(graph), {"9": "spec/spec_0210/images/v001/spec_0210_v001_astronaut_dune"})

    def test_a_video_goes_to_videos_and_a_generic_stem_becomes_the_kind(self):
        oc.set_current(sequence="spec", shot="spec_0210")
        image, video = _graph("ComfyUI"), _graph("agent/videos/video", "SaveVideo")
        oc.stamp(image), oc.stamp(video)
        self.assertEqual(image["9"]["inputs"]["filename_prefix"], "spec/spec_0210/images/v001/spec_0210_v001_image")
        self.assertEqual(video["9"]["inputs"]["filename_prefix"], "spec/spec_0210/videos/v001/spec_0210_v001_video")

    def test_the_agents_suffix_is_used_when_it_gave_one(self):
        oc.set_current(sequence="spec", shot="spec_0210", suffix="start frame")
        graph = _graph()
        oc.stamp(graph)
        self.assertEqual(graph["9"]["inputs"]["filename_prefix"],
                         "spec/spec_0210/images/v001/spec_0210_v001_start_frame")

    def test_the_version_is_the_first_with_no_such_file_yet(self):
        self._existing("spec/spec_0210/images/v001/spec_0210_v001_startframe_00001_.png")
        self._existing("spec/spec_0210/videos/v004/spec_0210_v004_startframe_00001.mp4")
        self._existing("spec/spec_0210/images/v009/spec_0210_v009_endframe_00001_.png")
        ctx = oc.set_current(sequence="spec", shot="spec_0210")
        self.assertEqual(oc.next_version(ctx, "startframe"), 5)
        self.assertEqual(oc.next_version(ctx, "endframe"), 10)
        self.assertEqual(oc.next_version(ctx, "plate"), 1)

    def test_two_generations_named_alike_in_one_turn_do_not_share_a_version(self):
        ctx = oc.set_current(sequence="spec", shot="spec_0210")
        self.assertEqual([oc.next_version(ctx, "ref"), oc.next_version(ctx, "ref")], [1, 2])

    def test_the_image_and_the_video_of_one_graph_share_a_version(self):
        oc.set_current(sequence="spec", shot="spec_0210", suffix="take")
        graph = _graph()
        graph["10"] = {"class_type": "SaveVideo", "inputs": {"filename_prefix": "agent/videos/x"}}
        named = oc.stamp(graph)
        self.assertEqual(named, {"9": "spec/spec_0210/images/v001/spec_0210_v001_take",
                                 "10": "spec/spec_0210/videos/v001/spec_0210_v001_take"})

    def test_running_the_same_graph_again_keeps_its_name(self):
        """A retry or a loop round: same version, ComfyUI's counter counts the takes."""
        oc.set_current(sequence="spec", shot="spec_0210")
        graph = _graph()
        first = dict(oc.stamp(graph))
        self.assertEqual(oc.stamp(graph), {})
        self.assertEqual(graph["9"]["inputs"]["filename_prefix"], first["9"])

    def test_a_prefix_that_names_its_own_folder_or_is_wired_is_left_alone(self):
        oc.set_current(sequence="spec", shot="spec_0210")
        own, dotted, wired = _graph("renders/final"), _graph("./renders/final"), _graph()
        wired["9"]["inputs"]["filename_prefix"] = ["4", 0]
        self.assertEqual([oc.stamp(own), oc.stamp(dotted), oc.stamp(wired)], [{}, {}, {}])
        self.assertEqual(own["9"]["inputs"]["filename_prefix"], "renders/final")

    def test_the_agent_is_told_the_rule_and_how_to_alternate_it(self):
        oc.set_current(sequence="spec", shot="spec_0210", source="AYON context node")
        note = oc.describe()
        self.assertIn("spec/spec_0210/images/v001/spec_0210_v001_startframe_00001_.png", note)
        self.assertIn("name_outputs(suffix=", note)
        self.assertIn("SEVERAL shots", note)


class TheTool(Scoped):

    def setUp(self):
        super().setUp()
        from src.utils.canvas_patch import clear
        clear()
        self.addCleanup(clear)

    def _wf(self, name="a.json"):
        p = self.out / name
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
        self.assertEqual(self._prefix(man), "characters/male_lead/images/v001/male_lead_v001_ref")
        self.assertEqual(self._prefix(woman), "characters/woman/images/v001/woman_v001_ref")
        self.assertEqual(out["files"], "characters/woman/images/v001/woman_v001_ref")
        self.assertEqual(oc.current()["shot"], "cast", "naming one workflow does not move the context")

    def test_a_named_workflow_is_not_named_again_when_it_is_submitted(self):
        oc.set_current(sequence="characters", shot="cast")
        wf = self._wf()
        self._call(suffix="ref", shot="woman", workflow_path=str(wf))
        graph = json.loads(wf.read_text(encoding="utf-8"))
        self.assertEqual(oc.stamp(graph), {})
        self.assertEqual(graph["9"]["inputs"]["filename_prefix"], "characters/woman/images/v001/woman_v001_ref")

    def test_without_a_workflow_it_sets_what_follows(self):
        oc.set_current(sequence="spec", shot="spec_0210")
        out = self._call(suffix="end frame", shot="spec_0220")
        self.assertEqual(out["files"], "spec/spec_0220/<images|videos>/v###/spec_0220_v###_end_frame")
        graph = _graph()
        oc.stamp(graph)
        self.assertEqual(graph["9"]["inputs"]["filename_prefix"],
                         "spec/spec_0220/images/v001/spec_0220_v001_end_frame")

    def test_it_works_with_no_context_node_when_the_agent_names_the_shot(self):
        wf = self._wf()
        self.assertIn("no sequence or shot", self._call(suffix="ref", workflow_path=str(wf))["error"])
        self._call(suffix="ref", shot="hero", sequence="assets", workflow_path=str(wf))
        self.assertEqual(self._prefix(wf), "assets/hero/images/v001/hero_v001_ref")

    def test_a_workflow_on_the_canvas_gets_the_same_name_there(self):
        from src.utils.canvas_patch import drain
        oc.set_current(sequence="spec", shot="spec_0210")
        wf = self._wf()
        live = {"40": {"class_type": "KSampler", "inputs": {}},
                "41": {"class_type": "SaveImage", "inputs": {"filename_prefix": "agent/images/x"}}}
        pipe = pipeline_stub(_canvas_graph=live)
        pipe._session.inserted_workflows[str(wf.resolve())] = ["40", "41"]
        out = self._call(pipe, suffix="startframe", workflow_path=str(wf))
        want = "spec/spec_0210/images/v001/spec_0210_v001_startframe"
        self.assertEqual(out["canvas_nodes"], ["41"])
        self.assertEqual(self._prefix(wf), want)
        self.assertEqual(live["41"]["inputs"]["filename_prefix"], want, "same version as the file")
        self.assertEqual([p["params"] for p in drain()], [{"filename_prefix": want}])


class InTheExecutor(unittest.TestCase):

    def setUp(self):
        self.src = (Path(__file__).resolve().parents[1] / "src" / "executor.py").read_text(encoding="utf-8")

    def test_every_submission_is_named_before_it_is_sent(self):
        submit = self.src.split("def _submit_workflow(", 1)[1].split("\ndef ", 1)[0]
        self.assertIn("output_context.stamp(workflow)", submit)
        self.assertLess(submit.index("output_context.stamp(workflow)"), submit.index('client.post("/prompt"'))

    def test_nothing_is_renamed_after_the_run(self):
        """The file ComfyUI saved is the file its history shows."""
        self.assertNotIn("finalize", self.src)
        self.assertFalse(hasattr(oc, "finalize"))
        module = Path(oc.__file__).read_text(encoding="utf-8")
        self.assertNotIn("os.replace", module)
        self.assertNotIn("os.rename", module)


if __name__ == "__main__":
    unittest.main()
