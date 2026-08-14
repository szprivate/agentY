"""Which variant produced which file — as data, not as an assumption about order.

The case: hook one generates a reference frame per character, hook two feeds them
to a video model that addresses them by position (`@image1`, `@image2`). That only
works if the agent knows which file is Anna and which is Ben.

It used to get a flat list of output paths and had to assume they came back in the
order they went in. They usually do — ComfyUI runs the queue serially — right up
until one member fails, is repaired, and is re-queued behind the others. Then two
references transpose and the video renders perfectly, starring the wrong people.

So the executor records which member produced each file, and each member is named
after the value that made it different *before* it runs — the panel drops (and
titles) a node the instant the file appears, so naming it afterwards is too late.

    python -m unittest discover -s tests
"""

import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from pipeline_stub import pipeline_stub, tools as _tools
from src.pipeline import Pipeline
from src.utils import output_tags as ot
from src.utils.canvas_hooks import build_batch
from src.utils.workflow_signal import clear_and_get


def _graph():
    return {
        "6": {"class_type": "CLIPTextEncode", "inputs": {"text": ""}},
        "3": {"class_type": "KSampler", "inputs": {"seed": 1, "positive": ["6", 0]}},
        "9": {"class_type": "SaveImage", "inputs": {"images": ["3", 0]}},
    }


def _hook(directive="Create one reference frame per character.\nrole: character reference"):
    return {"hook_node_id": "30", "purpose": "inline_parameter", "directive": directive,
            "anchors": [],
            "targets": [{"node_id": "6", "to_input": "text", "to_input_type": "STRING"}]}


CHARACTERS = ["Anna, red coat, 30s, freckles",
              "Ben, grey suit, late 40s",
              "Cleo, shaved head, silver jacket"]


class LabelTest(unittest.TestCase):
    """build_batch says what each variant was made from."""

    def test_every_variant_reports_its_own_values(self):
        labels = []
        prompts, _ = build_batch(_graph(), [
            {"target_node_id": "6", "param": "text", "mode": "value_list",
             "values": CHARACTERS}], labels=labels)
        self.assertEqual(len(prompts), 3)
        self.assertEqual([lb["6.text"] for lb in labels], CHARACTERS)

    def test_a_product_reports_both_axes(self):
        labels = []
        build_batch(_graph(), [
            {"target_node_id": "6", "param": "text", "mode": "value_list",
             "values": CHARACTERS[:2]},
            {"target_node_id": "3", "param": "seed", "mode": "sweep_seed", "count": 2},
        ], labels=labels)
        self.assertEqual(len(labels), 4)
        self.assertEqual({"6.text", "3.seed"}, set(labels[0]))

    def test_asking_for_no_labels_costs_nothing(self):
        prompts, _ = build_batch(_graph(), [
            {"target_node_id": "6", "param": "text", "mode": "value_list",
             "values": CHARACTERS}])
        self.assertEqual(len(prompts), 3)


class NamingTest(unittest.TestCase):
    """A variant is named after what makes it a different thing, not a different roll."""

    def test_the_prompt_names_it_not_the_seed(self):
        self.assertEqual(
            Pipeline._variant_label({"3.seed": 771, "6.text": "Anna, red coat"}),
            "Anna, red coat")

    def test_a_seed_only_sweep_has_nothing_to_name(self):
        self.assertEqual(Pipeline._variant_label({"3.seed": 771}), "")

    def test_any_string_will_do_when_no_slot_looks_like_a_prompt(self):
        self.assertEqual(Pipeline._variant_label({"9.filename_prefix": "shot_02"}),
                         "shot_02")

    def test_the_hooks_stated_role_qualifies_each_name(self):
        ot.clear()
        self.addCleanup(ot.clear)
        pipe = pipeline_stub(_canvas_hooks=[_hook()])
        paths = [f"C:/tmp/canvas_{i:03d}.json" for i in range(3)]
        labels = [{"6.text": c} for c in CHARACTERS]
        Pipeline._name_variants(pipe, paths, labels, _hook())
        ot.note_source("C:/out/ref_b.png", paths[1])
        self.assertEqual(ot.role_for("C:/out/ref_b.png"),
                         "character reference: Ben, grey suit, late 40s")
        self.assertTrue(ot.meta_for("C:/out/ref_b.png")["declared"],
                        "the user named the role, so its outputs get a ref note")


class PairingTest(unittest.TestCase):
    """The whole point: file → the value that produced it, whatever the order."""

    def setUp(self):
        ot.clear()
        self.addCleanup(ot.clear)
        self.paths = [f"C:/tmp/canvas_{i:03d}.json" for i in range(3)]
        self.labels = [{"6.text": c} for c in CHARACTERS]

    def test_each_variant_carries_its_own_outputs(self):
        ot.note_source("C:/out/ref_anna.png", self.paths[0])
        ot.note_source("C:/out/ref_ben.png", self.paths[1])
        ot.note_source("C:/out/ref_cleo.png", self.paths[2])
        report = Pipeline._variant_report(self.paths, self.labels)
        self.assertEqual(report[1]["made_from"]["6.text"], CHARACTERS[1])
        self.assertEqual(report[1]["outputs"], ["C:/out/ref_ben.png"])

    def test_a_healed_member_finishing_last_does_not_transpose_anything(self):
        """Ben failed, was repaired, and came back after Cleo — the pairing holds."""
        ot.note_source("C:/out/a.png", self.paths[0])
        ot.note_source("C:/out/c.png", self.paths[2])
        ot.note_source("C:/out/b.png", self.paths[1])      # healed, arrives last
        report = Pipeline._variant_report(self.paths, self.labels)
        self.assertEqual(report[1]["outputs"], ["C:/out/b.png"])
        self.assertEqual(report[2]["outputs"], ["C:/out/c.png"])

    def test_a_variant_that_produced_nothing_says_so(self):
        ot.note_source("C:/out/a.png", self.paths[0])
        report = Pipeline._variant_report(self.paths, self.labels,
                                          {self.paths[1]: "upstream 500"})
        self.assertNotIn("outputs", report[1])
        self.assertFalse(report[1]["ok"])
        self.assertEqual(report[1]["error"], "upstream 500")

    def test_several_files_from_one_variant_stay_together(self):
        ot.note_source("C:/out/a_00.png", self.paths[0])
        ot.note_source("C:/out/a_01.png", self.paths[0])
        self.assertEqual(Pipeline._variant_report(self.paths, self.labels)[0]["outputs"],
                         ["C:/out/a_00.png", "C:/out/a_01.png"])

    def test_a_long_prompt_is_trimmed_in_the_report_but_not_on_disk(self):
        long = "Anna, " + "red coat and freckles, " * 20
        report = Pipeline._variant_report(self.paths[:1], [{"6.text": long}])
        self.assertLessEqual(len(report[0]["made_from"]["6.text"]), 90)


class ThroughTheToolTest(unittest.TestCase):
    """End to end on apply_canvas_hooks, queued and run_now."""

    def setUp(self):
        ot.clear()
        clear_and_get()
        self.addCleanup(ot.clear)
        self.addCleanup(clear_and_get)

    def _pipe(self):
        return pipeline_stub(_canvas_base_prompt=_graph(), _canvas_hooks=[_hook()])

    def _resolutions(self):
        return [{"target_node_id": "6", "param": "text", "mode": "value_list",
                 "values": CHARACTERS}]

    def test_a_queued_batch_names_its_members_before_they_run(self):
        pipe = self._pipe()
        out = json.loads(asyncio.run(
            _tools(pipe)["apply_canvas_hooks"](resolutions=self._resolutions())))
        self.assertEqual(out["status"], "queued")
        self.assertEqual([v["made_from"]["6.text"] for v in out["variants"]], CHARACTERS)
        # The name is in place before the file exists — which is the only moment
        # that works, since the panel titles the node the instant it appears.
        queued = clear_and_get()
        ot.note_source("C:/out/x.png", queued[2])
        self.assertEqual(ot.role_for("C:/out/x.png"),
                         "character reference: Cleo, shaved head, silver jacket")

    def test_run_now_reports_the_pairing_it_actually_observed(self):
        produced = {}

        async def fake_batch(paths, *a, **kw):
            # Executed out of order, as a healed member would be.
            for i in (1, 2, 0):
                p = f"C:/out/gen_{i}.png"
                kw["collected_paths"].append(p)
                ot.note_source(p, paths[i])
                produced[paths[i]] = p
            yield "ran"

        pipe = self._pipe()
        with mock.patch("src.pipeline._execute_workflows_batch", fake_batch), \
             mock.patch("src.pipeline._clear_exec_errors"), \
             mock.patch("src.pipeline._get_exec_errors", return_value=[]):
            out = json.loads(asyncio.run(_tools(pipe)["apply_canvas_hooks"](
                run_now=True, resolutions=self._resolutions())))
        self.assertEqual(out["status"], "ran")
        for i, character in enumerate(CHARACTERS):
            self.assertEqual(out["variants"][i]["made_from"]["6.text"], character)
            self.assertEqual(out["variants"][i]["outputs"], [produced[out["variants"][i]["workflow"]]])
        self.assertIn("never the position in the flat list", out["message"])

    def test_running_the_canvas_as_it_stands_has_nothing_to_name(self):
        pipe = self._pipe()
        out = json.loads(asyncio.run(_tools(pipe)["apply_canvas_hooks"](resolutions=[])))
        self.assertEqual(out["status"], "queued")
        self.assertEqual(out["variants"], [{"variant": 1, "workflow": out["variants"][0]["workflow"],
                                            "ok": True}])


class SidecarTest(unittest.TestCase):
    """The name has to survive to the turn that wires the frames into the video."""

    def test_the_reference_frame_tells_the_next_turn_who_it_is(self):
        ot.clear()
        self.addCleanup(ot.clear)
        tmp = Path(tempfile.mkdtemp())
        img = tmp / "ref_002.png"
        img.write_bytes(b"x")
        wf = "C:/tmp/canvas_001.json"
        ot.set_workflow_role(wf, "character reference: Ben, grey suit, late 40s")
        ot.note_source(img, wf)
        ot.role_for(img)                      # the server pump, as the file lands
        ot.clear()                            # …and a later turn, with nothing in memory
        self.assertEqual(ot.role_of_file(img),
                         "character reference: Ben, grey suit, late 40s")


if __name__ == "__main__":
    unittest.main()
