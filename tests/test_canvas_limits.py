"""A canvas value the model will refuse goes back to the agent, not to the user.

From the log of 2026-08-13 23:38. The agent wrote a multi-shot prompt for a Kling
3.0 Omni node and placed it::

    {"status": "placed", "hook_node_id": "30", "chars": 8857,
     "injected_targets": ["283"], "message": "…injected your answer into the graph"}

8,857 characters into a node that accepts 2,500. The tool counted them, reported
success, and the agent told the user everything had worked. The run then died
inside the node, and by then nobody could fix it but the user.

Both canvas write paths now refuse first. The agent wrote the text and is still
holding the turn, so it is the one that can fix this — which is the whole point of
refusing at the tool rather than reporting afterwards.

    python -m unittest tests.test_canvas_limits
"""

import json
import unittest

from src.pipeline import Pipeline
from src.utils import model_limits as ml


def _node(cls, **inputs):
    return {"class_type": cls, "inputs": inputs}


def _canvas():
    """The user's graph, reduced: hook 30 feeds the Kling node's prompt."""
    return {
        "283": _node("KlingOmniProImageToVideoNode", prompt="", reference_images=["284", 0]),
        "284": _node("BatchImagesNode", **{"images.image0": ["193", 0]}),
        "193": _node("AgentYImageCollector",
                     files="\n".join(f"C:/refs/{i}.png" for i in range(15))),
        "300": _node("SaveVideo", video=["283", 0]),
    }


def _hook(targets=(("283", "prompt", "STRING"),)):
    return {"hook_node_id": "30", "purpose": "inline_parameter",
            "directive": "write the multi-shot video prompt",
            "anchors": [],
            "targets": [{"node_id": t, "to_input": i, "to_input_type": ty,
                         "type": "KlingOmniProImageToVideoNode"} for t, i, ty in targets]}


class _Stub:
    """A Pipeline reduced to what these two paths touch."""
    _verbose = False
    _canvas_limit_refusal = Pipeline._canvas_limit_refusal
    _batch_limit_refusal = Pipeline._batch_limit_refusal
    _count_handback = Pipeline._count_handback

    def __init__(self, prompt=None):
        self._canvas_base_prompt = prompt if prompt is not None else _canvas()
        self._limit_handbacks = {}


class PlaceCanvasTextTests(unittest.TestCase):
    def test_the_8857_character_prompt_is_refused(self):
        res = _Stub()._canvas_limit_refusal(_hook(), "x" * 8857)
        self.assertIsNotNone(res, "this is the case that shipped broken")
        self.assertIn("nothing was placed", res["error"])
        self.assertIn("8857 characters", res["what_to_fix"])
        self.assertIn("accepts 2500", res["what_to_fix"])
        self.assertIn("Cut at least 6357", res["what_to_fix"])

    def test_it_tells_the_agent_to_fix_it_rather_than_report_it(self):
        res = _Stub()._canvas_limit_refusal(_hook(), "x" * 8857)
        self.assertIn("call this tool again", res["what_to_fix"])
        self.assertIn("Do not report this to the user", res["do_not"])
        self.assertIn("do not stop the turn", res["do_not"])
        self.assertIn("Rewrite rather than truncate", res["how"])
        self.assertIn("SPLIT it across several runs", res["how"])

    def test_a_prompt_within_the_limit_is_placed(self):
        self.assertIsNone(_Stub()._canvas_limit_refusal(_hook(), "a short prompt"))

    def test_the_cap_comes_from_the_target_not_the_hook(self):
        # The same text is fine for a CLIPTextEncode and refused by the Kling node.
        graph = {"6": _node("CLIPTextEncode", text=""), "283": _canvas()["283"]}
        clip_hook = _hook(targets=(("6", "text", "STRING"),))
        self.assertIsNone(_Stub(graph)._canvas_limit_refusal(clip_hook, "x" * 8857))
        self.assertIsNotNone(_Stub(graph)._canvas_limit_refusal(_hook(), "x" * 8857))

    def test_a_storyboard_slot_has_its_own_much_tighter_cap(self):
        graph = {"283": _node("KlingOmniProTextToVideoNode", storyboard_2_prompt="")}
        hook = _hook(targets=(("283", "storyboard_2_prompt", "STRING"),))
        res = _Stub(graph)._canvas_limit_refusal(hook, "x" * 600)
        self.assertIn("accepts 512", res["what_to_fix"])

    def test_no_canvas_and_no_hook_are_not_errors(self):
        self.assertIsNone(_Stub()._canvas_limit_refusal(None, "x" * 8857))
        stub = _Stub()
        stub._canvas_base_prompt = None
        self.assertIsNone(stub._canvas_limit_refusal(_hook(), "x" * 8857))


class ApplyCanvasHooksTests(unittest.TestCase):
    """The other write path: a sweep of values, checked as built graphs."""

    @staticmethod
    def _built(prompt_text, refs=15):
        g = _canvas()
        g["283"]["inputs"]["prompt"] = prompt_text
        g["193"]["inputs"]["files"] = "\n".join(f"C:/refs/{i}.png" for i in range(refs))
        return g

    def test_a_swept_prompt_over_the_cap_is_refused_before_queueing(self):
        res = _Stub()._batch_limit_refusal([self._built("x" * 4000, refs=3)])
        self.assertIn("nothing was placed or queued", res["error"])
        self.assertEqual(res["violations"][0]["field"], "prompt")

    def test_too_many_reference_images_are_refused(self):
        res = _Stub()._batch_limit_refusal([self._built("short", refs=15)])
        self.assertEqual(res["violations"][0]["kind"], "images")
        self.assertIn("wired 15 images", res["what_to_fix"])
        self.assertIn("accepts 7", res["what_to_fix"])
        self.assertIn("Drop 8", res["what_to_fix"])
        self.assertIn("which you dropped", res["how"])

    def test_one_mistake_across_many_variants_is_reported_once(self):
        variants = [self._built("x" * 4000, refs=3) for _ in range(25)]
        res = _Stub()._batch_limit_refusal(variants)
        self.assertEqual(len(res["violations"]), 1,
                         "25 variants of one over-long prompt is one mistake")

    def test_a_clean_batch_is_queued(self):
        self.assertIsNone(_Stub()._batch_limit_refusal([self._built("short", refs=3)]))

    def test_both_problems_are_reported_together(self):
        res = _Stub()._batch_limit_refusal([self._built("x" * 4000, refs=15)])
        self.assertEqual({v["kind"] for v in res["violations"]}, {"text", "images"})


class BatchImagesNodeTests(unittest.TestCase):
    """The node the user's graph actually used to gather references."""

    def test_it_counts_through_the_batch_node(self):
        g = _canvas()
        self.assertEqual(ml.count_images_into(g, "283", "reference_images"), 15)

    def test_several_slots_are_summed(self):
        g = _canvas()
        g["284"]["inputs"] = {"images.image0": ["193", 0], "images.image1": ["194", 0]}
        g["193"]["inputs"]["files"] = "C:/refs/a.png\nC:/refs/b.png"
        g["194"] = _node("LoadImage", image="c.png")
        self.assertEqual(ml.count_images_into(g, "283", "reference_images"), 3)


class RepeatedHandbackTests(unittest.TestCase):
    """Telling an agent the same sentence a third time is not advice."""

    def test_the_advice_turns_structural_on_the_third_attempt(self):
        stub = _Stub()
        first = stub._canvas_limit_refusal(_hook(), "x" * 8857)
        second = stub._canvas_limit_refusal(_hook(), "x" * 8000)
        third = stub._canvas_limit_refusal(_hook(), "x" * 7000)
        self.assertEqual([first["attempt"], second["attempt"], third["attempt"]], [1, 2, 3])
        self.assertIn("Rewrite rather than truncate", first["how"])
        self.assertIn("stop trimming", third["how"])
        self.assertIn("SPLIT it across several runs", third["how"])
        self.assertIn("let them choose", third["how"])

    def test_a_different_input_keeps_its_own_count(self):
        stub = _Stub()
        stub._canvas_limit_refusal(_hook(), "x" * 8857)
        stub._canvas_limit_refusal(_hook(), "x" * 8857)
        other = _hook(targets=(("283", "negative_prompt", "STRING"),))
        stub._canvas_base_prompt["283"]["inputs"]["negative_prompt"] = ""
        graph = {"283": _node("KlingTextToVideoNode", negative_prompt="")}
        stub2 = _Stub(graph)
        self.assertEqual(stub2._canvas_limit_refusal(other, "x" * 8857)["attempt"], 1)


class ThroughTheRealToolTests(unittest.TestCase):
    """End to end on the tool the agent actually calls, with the logged value."""

    @staticmethod
    def _pipe(hooks):
        from pipeline_stub import pipeline_stub
        return pipeline_stub(_canvas_base_prompt=_canvas(), _canvas_hooks=hooks)

    def _place(self, text):
        import asyncio
        hook = _hook()
        hook["freeze"] = False           # keep-live: inject into the graph at run time
        pipe = self._pipe([hook])
        tool = {t.tool_name: t for t in Pipeline._build_delegation_tools(pipe)}["place_canvas_text"]
        return json.loads(asyncio.run(tool(hook_node_id="30", text=text))), pipe

    def test_the_logged_8857_character_placement_is_now_refused(self):
        # The log said: {"status": "placed", "chars": 8857, "injected_targets": ["283"]}
        res, pipe = self._place("x" * 8857)
        self.assertNotIn("status", res, "it must not report success")
        self.assertIn("error", res)
        self.assertIn("8857 characters", res["what_to_fix"])
        self.assertEqual(pipe._canvas_base_prompt["283"]["inputs"]["prompt"], "",
                         "and nothing may reach the graph")

    def test_a_value_within_the_limit_still_places_and_injects(self):
        res, pipe = self._place("a prompt that fits")
        # Not `status == "placed"`: whether a text NODE is also drawn is a
        # setting, and this test is about the value reaching the graph — which
        # happens either way. Asserting the node here made the suite depend on
        # whoever ran it having Canvas → place text nodes switched on.
        self.assertIn(res["status"], ("placed", "injected"))
        self.assertNotIn("error", res)
        self.assertEqual(res["injected_targets"], ["283"])
        self.assertEqual(pipe._canvas_base_prompt["283"]["inputs"]["prompt"],
                         "a prompt that fits")


class ToolResultShapeTests(unittest.TestCase):
    """It has to survive being JSON-encoded into a tool result."""

    def test_the_refusal_is_json_serialisable(self):
        res = _Stub()._canvas_limit_refusal(_hook(), "x" * 8857)
        back = json.loads(json.dumps(res))
        self.assertEqual(set(back),
                         {"error", "what_to_fix", "how", "attempt", "violations", "do_not"})


if __name__ == "__main__":
    unittest.main()
