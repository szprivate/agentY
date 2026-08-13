"""Hard model limits: caught before the call, and handed to whoever can fix them.

Kling 3.0 Omni refuses a prompt over 2,500 characters and more than seven
reference images. ComfyUI raises on both from inside the node, so the failure
arrives as an execution error and lands on the repair specialist — the one agent
that cannot help, because shortening a prompt is rewriting it and choosing which
reference to drop is a creative decision. Every number checked here is read off
ComfyUI's own validators (comfy_api_nodes/nodes_kling.py).

    python -m unittest tests.test_model_limits
"""

import json
import unittest

from src.utils import model_limits as ml


def _node(cls, **inputs):
    return {"class_type": cls, "inputs": inputs}


def _kling_i2v(prompt="a slow dolly through fog", refs=None):
    """Kling 3.0 Omni image-to-video: prompt <= 2500, reference_images <= 7."""
    g = {"9": _node("KlingOmniProImageToVideoNode", prompt=prompt,
                    reference_images=["100", 0] if refs else None)}
    if refs:
        g.update(refs)
    return g


def _batch_of(n, start=100):
    """n LoadImages combined by ImageBatch, ending at node `start`."""
    g = {}
    prev = None
    for i in range(n):
        lid = str(start + 100 + i)
        g[lid] = _node("LoadImage", image=f"ref{i}.png", upload="image")
        if prev is None:
            prev = lid
            continue
        bid = str(start + i)
        g[bid] = _node("ImageBatch", image1=[prev, 0], image2=[lid, 0])
        prev = bid
    # The node the consumer wires to must be reachable as `start`.
    g[str(start)] = g.pop(prev) if prev != str(start) else g[str(start)]
    return g, str(start)


class PromptLengthTests(unittest.TestCase):
    def test_the_reported_case(self):
        over = ml.check_workflow(_kling_i2v("x" * 2600))
        self.assertEqual(len(over), 1)
        v = over[0]
        self.assertEqual((v.kind, v.field, v.limit, v.actual), ("text", "prompt", 2500, 2600))
        self.assertIn("Cut at least 100", v.describe())

    def test_exactly_at_the_limit_is_allowed(self):
        self.assertEqual(ml.check_workflow(_kling_i2v("x" * 2500)), [])

    def test_a_normal_prompt_says_nothing(self):
        self.assertEqual(ml.check_workflow(_kling_i2v()), [])

    def test_storyboard_slots_have_their_own_much_tighter_cap(self):
        g = {"9": _node("KlingOmniProTextToVideoNode", prompt="fine",
                        storyboard_1_prompt="ok", storyboard_4_prompt="y" * 600)}
        over = ml.check_workflow(g)
        self.assertEqual([(v.field, v.limit, v.actual) for v in over],
                         [("storyboard_4_prompt", 512, 600)])

    def test_image_to_video_is_tighter_than_text_to_video(self):
        # 800 characters is fine for Kling text-to-video and refused by image2video.
        self.assertEqual(ml.check_workflow({"1": _node("KlingTextToVideoNode", prompt="x" * 800)}), [])
        self.assertEqual(len(ml.check_workflow({"1": _node("KlingImage2VideoNode", prompt="x" * 800)})), 1)

    def test_an_unlisted_node_is_not_policed(self):
        # Unknown is not zero: a node with no known cap must never be flagged.
        self.assertEqual(ml.check_workflow({"1": _node("KSampler", text="x" * 99999)}), [])

    def test_a_linked_input_is_not_measured_as_text(self):
        g = {"9": _node("KlingOmniProImageToVideoNode", prompt=["7", 0]),
             "7": _node("PrimitiveStringMultiline", value="x" * 5000)}
        self.assertEqual(ml.check_workflow(g), [])


class ImageCountTests(unittest.TestCase):
    def test_eight_references_into_a_seven_image_node(self):
        refs, head = _batch_of(8)
        over = ml.check_workflow(_kling_i2v(refs=refs))
        self.assertEqual(len(over), 1)
        v = over[0]
        self.assertEqual((v.kind, v.field, v.limit, v.actual), ("images", "reference_images", 7, 8))
        self.assertIn("Drop 1", v.describe())

    def test_seven_is_allowed(self):
        refs, _ = _batch_of(7)
        self.assertEqual(ml.check_workflow(_kling_i2v(refs=refs)), [])

    def test_the_first_last_frame_node_stops_at_six(self):
        refs, _ = _batch_of(7)
        g = {"9": _node("KlingOmniProFirstLastFrameNode", prompt="ok",
                        reference_images=["100", 0])}
        g.update(refs)
        self.assertEqual([(v.limit, v.actual) for v in ml.check_workflow(g)], [(6, 7)])

    def test_a_collector_counts_its_files(self):
        g = {"9": _node("KlingOmniProImageToVideoNode", prompt="ok",
                        reference_images=["100", 0]),
             "100": _node("AgentYImageCollector",
                          files="\n".join(f"C:/in/{i}.png" for i in range(9)))}
        self.assertEqual([(v.limit, v.actual) for v in ml.check_workflow(g)], [(7, 9)])

    def test_a_collector_stepping_one_per_queue_counts_as_one(self):
        g = {"9": _node("KlingOmniProImageToVideoNode", prompt="ok",
                        reference_images=["100", 0]),
             "100": _node("AgentYImageCollector", load_incrementally="true",
                          files="\n".join(f"C:/in/{i}.png" for i in range(9)))}
        self.assertEqual(ml.check_workflow(g), [])

    def test_a_ref_note_on_the_wire_is_counted_through(self):
        refs, _ = _batch_of(8)
        g = {"9": _node("KlingOmniProImageToVideoNode", prompt="ok",
                        reference_images=["50", 0]),
             "50": _node("AgentYRefNote", input=["100", 0], role="the faces")}
        g.update(refs)
        self.assertEqual([(v.limit, v.actual) for v in ml.check_workflow(g)], [(7, 8)])

    def test_an_unknowable_source_gives_no_verdict(self):
        # Something that could emit any number: silence beats a wrong accusation.
        g = {"9": _node("KlingOmniProImageToVideoNode", prompt="ok",
                        reference_images=["100", 0]),
             "100": _node("SomeCustomBatchThing", n=40)}
        self.assertIsNone(ml.count_images_into(g, "9", "reference_images"))
        self.assertEqual(ml.check_workflow(g), [])

    def test_one_unknown_part_makes_the_whole_batch_unknown(self):
        g = {"9": _node("KlingOmniProImageToVideoNode", prompt="ok",
                        reference_images=["100", 0]),
             "100": _node("ImageBatch", image1=["101", 0], image2=["102", 0]),
             "101": _node("LoadImage", image="a.png"),
             "102": _node("SomeCustomBatchThing", n=40)}
        self.assertEqual(ml.check_workflow(g), [])

    def test_a_cycle_does_not_hang(self):
        g = {"9": _node("KlingOmniProImageToVideoNode", prompt="ok",
                        reference_images=["100", 0]),
             "100": _node("ImageBatch", image1=["101", 0]),
             "101": _node("ImageBatch", image1=["100", 0])}
        self.assertIsNone(ml.count_images_into(g, "9", "reference_images"))


class RuntimeErrorTests(unittest.TestCase):
    """When it already ran: recognise the model's own complaint."""

    def test_comfyuis_own_wording_for_a_long_prompt(self):
        err = {"details": {"exception_message":
               " Field 'prompt cannot be longer than 2500 characters; was 3120 characters long."}}
        self.assertIn("2500", ml.runtime_limit_error(err))

    def test_the_wording_for_too_many_images(self):
        err = {"details": {"exception_message": "A maximum of 3 input images is supported."}}
        self.assertTrue(ml.runtime_limit_error(err))

    def test_an_ordinary_failure_is_left_to_the_repair_specialist(self):
        err = {"details": {"exception_type": "OutOfMemoryError",
                           "exception_message": "CUDA out of memory"}}
        self.assertEqual(ml.runtime_limit_error(err), "")
        self.assertEqual(ml.runtime_limit_error(None), "")


class GuidanceTests(unittest.TestCase):
    def test_it_tells_the_orchestrator_what_to_do_with_it(self):
        text = ml.guidance(ml.check_workflow(_kling_i2v("x" * 3000)))
        self.assertIn("hard API limit", text)
        self.assertIn("repair specialist cannot fix it", text)
        self.assertIn("update_workflow", text)
        self.assertIn("signal_workflow_ready", text)
        self.assertIn("Do not re-run prepare_workflow", text)
        self.assertIn("Do NOT truncate mid-sentence", text)

    def test_it_works_from_a_runtime_message_alone(self):
        text = ml.guidance([], "A maximum of 3 input images is supported.")
        self.assertIn("the model reported", text)

    def test_the_one_line_summary_carries_the_reason(self):
        # The batch executor prints this when a heal fails; there is no
        # orchestrator listening there, so "still invalid" would be the whole story.
        line = ml.summary(ml.check_workflow(_kling_i2v("x" * 3000)))
        self.assertIn("KlingOmniProImageToVideoNode", line)
        self.assertIn("prompt 3000 chars > 2500", line)
        self.assertIn("not repair", line)
        self.assertEqual(len(line.splitlines()), 1)


class TableTests(unittest.TestCase):
    def test_the_table_is_loadable_and_covers_kling_omni(self):
        text, images, note = ml.limits_for("KlingOmniProImageToVideoNode")
        self.assertEqual(text["prompt"], 2500)
        self.assertEqual(images["reference_images"], 7)
        self.assertIn("Kling", note)

    def test_an_unknown_node_has_no_limits(self):
        self.assertEqual(ml.limits_for("KSampler"), ({}, {}, ""))

    def test_every_entry_is_well_formed(self):
        data = json.loads(ml._CONFIG.read_text(encoding="utf-8"))
        for entry in data["limits"]:
            self.assertTrue(entry.get("nodes"), entry)
            self.assertTrue(entry.get("text") or entry.get("images"), entry)
            for value in list((entry.get("text") or {}).values()) + \
                         list((entry.get("images") or {}).values()):
                self.assertIsInstance(value, int)
                self.assertGreater(value, 0)


class ScaffoldTests(unittest.TestCase):
    """Cheaper than handing it back: say the cap while the prompt is being written."""

    def test_the_cap_reaches_whoever_writes_the_prompt(self):
        from src.tools.briefing_scaffold import _resolve_prompt_nodes
        wf = {"9": _node("KlingOmniProImageToVideoNode", prompt="",
                         reference_images=["100", 0]),
              "100": _node("LoadImage", image="a.png")}
        nodes = _resolve_prompt_nodes(wf)[2]
        self.assertEqual(nodes[0]["max_chars"], 2500)
        self.assertEqual(nodes[0]["slot"], "prompt")

    def test_a_node_with_no_cap_carries_none(self):
        from src.tools.briefing_scaffold import _resolve_prompt_nodes
        wf = {"6": _node("CLIPTextEncode", text="", clip=["4", 1]),
              "3": _node("KSampler", positive=["6", 0])}
        self.assertNotIn("max_chars", _resolve_prompt_nodes(wf)[2][0])

    def test_it_survives_the_briefing_schema(self):
        from src.pipeline import PromptNode
        pn = PromptNode(node_id="9", role="positive", slot="prompt",
                        node="KlingOmniProImageToVideoNode", max_chars=2500)
        self.assertIn('"max_chars":2500', pn.model_dump_json().replace(" ", ""))


class HandBackTests(unittest.TestCase):
    """Where it matters: the turn stops going to the repair specialist."""

    def setUp(self):
        import tempfile
        from pathlib import Path
        self.path = Path(tempfile.mkdtemp(prefix="agenty-limits-")) / "wf.json"
        self.path.write_text(json.dumps(_kling_i2v("x" * 3000)), encoding="utf-8")

    @staticmethod
    def _stub():
        """A Pipeline stripped to what these paths touch, with a tripwire where the
        repair specialist would be started."""
        from src.pipeline import Pipeline

        class Stub:
            _verbose = False
            _FIX_ASSEMBLY_TIMEOUT = 1
            _limit_violations = Pipeline._limit_violations

            def _ensure_fix_agent(self):
                raise AssertionError("the repair specialist must not be started for this")
        return Stub()

    def test_assembly_hands_it_back_instead_of_calling_it_ready(self):
        from src.pipeline import Pipeline
        res = Pipeline._limit_violations(self._stub(), str(self.path))
        self.assertEqual(res["status"], "limit_exceeded")
        self.assertEqual(res["workflow_path"], str(self.path))
        self.assertEqual([v["field"] for v in res["violations"]], ["prompt"])
        self.assertIn("update_workflow", res["guidance"])
        self.assertIn("hard model limit", res["error"],
                      "the batch executor prints this one and nothing else")

    def test_the_repair_specialist_is_never_started_for_a_limit(self):
        import asyncio

        from src.pipeline import Pipeline
        res = asyncio.run(Pipeline._run_fix_workflow_assembly(
            self._stub(), str(self.path), problems=["something else"]))
        self.assertEqual(res["status"], "limit_exceeded")

    def test_a_runtime_complaint_alone_is_enough(self):
        import asyncio

        from src.pipeline import Pipeline
        self.path.write_text(json.dumps(_kling_i2v("short prompt")), encoding="utf-8")
        res = asyncio.run(Pipeline._run_fix_workflow_assembly(
            self._stub(), str(self.path),
            exec_error={"details": {"exception_message":
                        "A maximum of 3 input images is supported."}}))
        self.assertEqual(res["status"], "limit_exceeded")
        self.assertEqual(res["violations"], [], "the graph itself was within limits")
        self.assertIn("the model reported", res["guidance"])

    def test_an_ordinary_defect_still_reaches_the_specialist(self):
        import asyncio

        from src.pipeline import Pipeline
        self.path.write_text(json.dumps(_kling_i2v("short prompt")), encoding="utf-8")
        with self.assertRaises(AssertionError):   # the stub's tripwire
            asyncio.run(Pipeline._run_fix_workflow_assembly(
                self._stub(), str(self.path), problems=["missing input on node 4"]))


if __name__ == "__main__":
    unittest.main()
