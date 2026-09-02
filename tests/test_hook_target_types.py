"""One hook, two targets that want different KINDS of thing.

A hook's output can be wired into several inputs at once — on the run these are
drawn from, one hook fed an image-edit node's `prompt` (STRING) and its
`model.images.image_1` (IMAGE connection). Filling those is the agent's job, and
getting them the wrong way round is the failure that repeated for nine runs:

  - the block offered `connect one of 141` under the IMAGE input, where 141 was
    the STRING anchor — the only thing wired into that hook,
  - the agent concluded node ids were the currency and wrote `"141"` into the
    PROMPT, its own words being *"For the STRING target (`prompt`), I should pass
    the anchor node id (#141) as the value for all runs"*,
  - three generations were queued with the prompt `141`, reported `ok`,
  - and an earlier run resolved only the image input, so the prompt was filled by
    the hook's own value with nothing in the receipt to say so.

Every check here is on the data the graph already carries: the declared type of
each target input and of each anchor's output.

    python -m unittest discover -s tests
"""

import asyncio
import json
import unittest

from pipeline_stub import pipeline_stub, tools
from src.utils.canvas_hooks import (_target_context, misrouted_resolutions,
                                    target_input_types, type_satisfies,
                                    unresolved_targets)


def _hook(anchors=(), targets=(), hook_id="142"):
    return {
        "hook_node_id": hook_id,
        "purpose": "inline_parameter",
        "directive": "edit each reference image",
        "anchors": [{"node_id": nid, "from_output_type": ty} for nid, ty in anchors],
        "targets": [{"node_id": nid, "to_input": inp, "to_input_type": ty,
                     "type": "OpenAIGPTImageNodeV2"} for nid, inp, ty in targets],
    }


# The shape that failed: one STRING anchor, two targets of different kinds.
STRING_ANCHOR = _hook(
    anchors=[("141", "STRING")],
    targets=[("31", "prompt", "STRING"),
             ("31", "model.images.image_1", "IMAGE")])

# The shape that works: six loaders wired in, same two targets.
IMAGE_ANCHORS = _hook(
    anchors=[(str(n), "IMAGE") for n in (92, 93, 94, 95, 97, 98)],
    targets=[("106", "prompt", "STRING"),
             ("106", "model.images.image_1", "IMAGE")],
    hook_id="105")

CANVAS = {
    "31": {"class_type": "OpenAIGPTImageNodeV2", "inputs": {"prompt": ""}},
    "141": {"class_type": "AgentYText", "inputs": {"text": "add snowfall"}},
}


class TypeRuleTests(unittest.TestCase):

    def test_like_feeds_like(self):
        self.assertTrue(type_satisfies("IMAGE", "IMAGE"))
        self.assertFalse(type_satisfies("STRING", "IMAGE"))
        self.assertFalse(type_satisfies("IMAGE", "STRING"))

    def test_a_wildcard_satisfies_anything_either_way(self):
        for wild in ("*", "COMFY_MATCHTYPE_V3", "COMFY_MULTITYPE_V3"):
            with self.subTest(wild=wild):
                self.assertTrue(type_satisfies(wild, "IMAGE"))
                self.assertTrue(type_satisfies("IMAGE", wild))

    def test_unknown_is_allowed_rather_than_excluded(self):
        """A type the frontend did not report is not evidence of a mismatch, and
        excluding on missing information hides a choice that would have worked."""
        self.assertTrue(type_satisfies("", "IMAGE"))
        self.assertTrue(type_satisfies(None, "IMAGE"))
        self.assertTrue(type_satisfies("IMAGE", ""))


class CandidateOfferTests(unittest.TestCase):
    """Which anchors the block offers for a CONNECTION input."""

    def test_an_anchor_that_cannot_feed_the_input_is_not_offered(self):
        line = _target_context(STRING_ANCHOR)
        self.assertNotIn("connect one of 141", line,
                         "offered the text node as the image to connect")

    def test_and_the_block_says_why_there_is_nothing_to_choose(self):
        line = _target_context(STRING_ANCHOR)
        self.assertIn("nothing wired into this hook carries a IMAGE", line)
        # And it now says what WOULD fill it. Leading with "supply a node id" is
        # what sent a run to name the node that generates the references, which
        # wires the generator in and runs it again instead of reusing its output.
        self.assertIn("FILE PATH", line)
        self.assertIn("CONNECTION", line, "it is still marked as taking a wire")

    def test_matching_anchors_are_offered(self):
        line = _target_context(IMAGE_ANCHORS)
        self.assertIn("connect one of 92, 93, 94, 95, 97, 98", line)

    def test_the_text_target_is_never_given_a_node_id_list(self):
        """The candidate list is the connection input's, and saying it anywhere
        else is what taught the agent that node ids were the currency here."""
        prompt_part = _target_context(IMAGE_ANCHORS).split(";")[0]
        self.assertIn("`prompt`", prompt_part)
        self.assertNotIn("connect one of", prompt_part)

    def test_an_anchor_of_unknown_type_is_still_offered(self):
        hook = _hook(anchors=[("77", "")],
                     targets=[("31", "model.images.image_1", "IMAGE")])
        self.assertIn("connect one of 77", _target_context(hook))


class TargetTypeMapTests(unittest.TestCase):

    def test_every_target_keeps_its_declared_type(self):
        self.assertEqual(target_input_types([STRING_ANCHOR]),
                         {"31.prompt": "STRING",
                          "31.model.images.image_1": "IMAGE"})

    def test_no_hooks_is_an_empty_map_not_an_error(self):
        self.assertEqual(target_input_types(None), {})
        self.assertEqual(target_input_types([]), {})


class MisroutedTests(unittest.TestCase):

    def _res(self, param, values):
        return {"target_node_id": "31", "param": param,
                "mode": "value_list", "values": values}

    def test_a_node_id_written_into_a_prompt_is_caught(self):
        bad = misrouted_resolutions(
            CANVAS, [STRING_ANCHOR], [self._res("prompt", ["141", "141", "141"])])
        self.assertEqual(len(bad), 1, bad)
        self.assertIn("31.prompt", bad[0])
        self.assertIn("141", bad[0])
        self.assertIn("AgentYText", bad[0], "say WHAT node the id belongs to")

    def test_real_words_pass(self):
        self.assertEqual(misrouted_resolutions(
            CANVAS, [STRING_ANCHOR],
            [self._res("prompt", ["add a realistic winter snowfall"])]), [])

    def test_a_node_id_in_the_connection_input_is_exactly_right(self):
        self.assertEqual(misrouted_resolutions(
            CANVAS, [STRING_ANCHOR],
            [self._res("model.images.image_1", ["141"])]), [])

    def test_a_number_that_is_no_node_on_this_canvas_passes(self):
        self.assertEqual(misrouted_resolutions(
            CANVAS, [STRING_ANCHOR], [self._res("prompt", ["9999"])]), [])

    def test_a_hook_with_no_connection_target_is_not_checked(self):
        """Nothing here takes a node id, so there is no confusion to make and a
        value that happens to look like one is the user's business."""
        text_only = _hook(anchors=[("141", "STRING")],
                          targets=[("31", "prompt", "STRING")])
        self.assertEqual(misrouted_resolutions(
            CANVAS, [text_only], [self._res("prompt", ["141"])]), [])

    def test_one_bad_value_in_a_list_is_enough(self):
        bad = misrouted_resolutions(
            CANVAS, [STRING_ANCHOR],
            [self._res("prompt", ["a real prompt", "141"])])
        self.assertEqual(len(bad), 1)

    def test_it_is_reported_once_per_resolution_not_once_per_value(self):
        bad = misrouted_resolutions(
            CANVAS, [STRING_ANCHOR], [self._res("prompt", ["141", "141", "141"])])
        self.assertEqual(len(bad), 1)

    def test_nothing_to_check_is_no_problem(self):
        self.assertEqual(misrouted_resolutions(CANVAS, [STRING_ANCHOR], []), [])
        self.assertEqual(misrouted_resolutions(None, [STRING_ANCHOR],
                                               [self._res("prompt", ["141"])]), [])


class UnresolvedTargetTests(unittest.TestCase):

    def test_a_target_nobody_resolved_is_named(self):
        left = unresolved_targets([STRING_ANCHOR], [
            {"target_node_id": "31", "param": "model.images.image_1",
             "values": ["a.png"]}])
        self.assertEqual(left, ["31.prompt"])

    def test_both_resolved_leaves_nothing_to_say(self):
        self.assertEqual(unresolved_targets([STRING_ANCHOR], [
            {"target_node_id": "31", "param": "prompt", "values": ["words"]},
            {"target_node_id": "31", "param": "model.images.image_1",
             "values": ["a.png"]}]), [])

    def test_a_single_target_hook_says_nothing(self):
        """One target IS the hook's value — delivering it there is the whole
        design, and announcing it every run would be noise."""
        one = _hook(anchors=[("141", "STRING")],
                    targets=[("31", "prompt", "STRING")])
        self.assertEqual(unresolved_targets([one], []), [])

    def test_no_resolutions_at_all_still_names_the_several(self):
        self.assertEqual(sorted(unresolved_targets([STRING_ANCHOR], [])),
                         ["31.model.images.image_1", "31.prompt"])


class NothingIsQueuedTest(unittest.TestCase):
    """The gate, at the door it has to hold: apply_canvas_hooks itself.

    The run this comes from queued three generations with the prompt `141` and
    reported ``ok``. Catching it in a helper is worth nothing if the tool still
    says yes.
    """

    def _pipe(self):
        return pipeline_stub(_canvas_base_prompt=dict(CANVAS),
                             _canvas_hooks=[STRING_ANCHOR])

    @staticmethod
    def _apply(pipe, resolutions, **kw):
        return json.loads(
            asyncio.run(tools(pipe)["apply_canvas_hooks"](resolutions, **kw)))

    def test_a_node_id_in_the_prompt_stops_the_run(self):
        from unittest import mock
        with mock.patch("src.utils.workflow_signal.append_workflow_path") as queued:
            out = self._apply(self._pipe(), [
                {"target_node_id": "31", "param": "model.images.image_1",
                 "mode": "value_list", "values": ["a.png", "b.png", "c.png"],
                 "zip_group": "run"},
                {"target_node_id": "31", "param": "prompt", "mode": "value_list",
                 "values": ["141", "141", "141"], "zip_group": "run"}])
        queued.assert_not_called()
        self.assertIn("error", out)
        self.assertNotIn("status", out, "it must not also report a queued batch")
        self.assertTrue(any("31.prompt" in p for p in out["problems"]), out)
        self.assertIn("fix", out, "say what to do instead, or it just tries again")

    def test_the_same_batch_with_real_words_runs(self):
        out = self._apply(self._pipe(), [
            {"target_node_id": "31", "param": "model.images.image_1",
             "mode": "value_list", "values": ["a.png", "b.png"], "zip_group": "run"},
            {"target_node_id": "31", "param": "prompt", "mode": "value_list",
             "values": ["add snowfall", "add snowfall"], "zip_group": "run"}])
        self.assertNotIn("error", out)
        self.assertEqual(out.get("count"), 2)

    def test_an_unresolved_second_target_is_reported_in_the_notes(self):
        # `sweep: true` because two images on one numbered slot is otherwise
        # ambiguous and refused (see test_slot_fan_out) — one run each is what
        # this test means, and it now has to say so.
        out = self._apply(self._pipe(), [
            {"target_node_id": "31", "param": "model.images.image_1",
             "mode": "value_list", "values": ["a.png", "b.png"], "sweep": True}])
        self.assertNotIn("error", out)
        self.assertTrue(any("31.prompt" in n and "hook's own produced value" in n
                            for n in out.get("notes", [])), out.get("notes"))


if __name__ == "__main__":
    unittest.main()
