"""One call, one stage — and nothing in the graph that cannot affect the result.

Three findings from one dry run of a reference→video canvas.

The run produced **3 workflows, and every one of them held both the Seedream node
and the Seedance node**. It should have been several Seedream graphs (one per
reference) and a single Seedance graph. The cause is that the turn-level scope
keeps everything ANY hook reaches — right for the turn, wrong for one
``apply_canvas_hooks`` call, which resolves exactly one hook. Building that hook's
variants from the whole scope carries the other stage into every one of them, and
a five-reference sweep would have generated five videos nobody asked for.

And **every image and ref-note wired into the hook made it into the generated
workflows**, where they cannot affect anything: their only consumer was the hook,
and the hook is spliced out before the graph runs. A node with no consumer that is
not itself an output was never going to execute — ComfyUI walks back from output
nodes — so it is dead weight in a file the user is meant to be able to read.

    python -m unittest discover -s tests
"""

import asyncio
import json
import unittest
from unittest import mock

from pipeline_stub import pipeline_stub, tools
from src.utils.canvas_hooks import prune_dead_nodes, scope_to_hook


# A canvas with two stages joined ONLY through the hook chain: a Seedream branch
# that makes references, and a Seedance branch that makes the video. Plus the
# reference images and their ref note, which fed the hooks as context.
def canvas():
    return {
        "7":   {"class_type": "LoadImage", "inputs": {"image": "ben.png"}},
        "8":   {"class_type": "AgentYRefNote", "inputs": {"any": ["7", 0], "note": "Ben"}},
        # The dotted name is what the frontend really serialises a numbered
        # autogrow slot as (see any captured Seedream graph); the bare
        # `image_1` this used to say is not an address the node reads.
        "348": {"class_type": "ByteDanceSeedreamNodeV2",
                "inputs": {"prompt": "", "model.images.image_1": ["7", 0]}},
        "349": {"class_type": "SaveImage", "inputs": {"images": ["348", 0]}},
        "283": {"class_type": "ByteDanceSeedanceNode", "inputs": {"prompt": ""}},
        "284": {"class_type": "SaveVideo", "inputs": {"video": ["283", 0]}},
    }


def hook(hid, target, param="prompt", param_type="STRING", extra=()):
    """A hook whose output feeds *param* on *target*.

    ``param_type`` matters: it is what tells the pipeline whether the produced
    value is a literal or a WIRE. An IMAGE target is delivered as a connection,
    which is the case the reference sweep exercises.
    """
    targets = [{"node_id": target, "to_input": param, "to_input_type": param_type}]
    targets += [{"node_id": target, "to_input": p, "to_input_type": t}
                for p, t in extra]
    return {"hook_node_id": hid, "purpose": "inline_parameter", "directive": "go",
            "anchors": [{"node_id": "7", "to_input": "anchors.anchor"}],
            "targets": targets}


class SchemaOffline:
    """Answer the "is this class an output node?" question without ComfyUI.

    `is_terminal` is deliberately biased: a class it cannot ask about counts as an
    output, so with the server down NOTHING is ever pruned and every assertion
    below about dropped nodes fails. That bias is right in production — keeping a
    dead node costs a line of JSON, dropping a live one costs the render — and
    wrong here, where what is being tested is the pruning itself.

    None of these fixtures is an output node; the ones that are (`SaveImage`,
    `SaveVideo`, `PreviewImage`) are recognised by name before the schema is
    consulted at all, so a flat "no" is the whole of ComfyUI's answer.
    """

    def setUp(self):
        super().setUp()
        patch = mock.patch("src.utils.preflight._schema",
                           return_value={"output_node": False})
        patch.start()
        self.addCleanup(patch.stop)


class ScopeToOneStageTest(SchemaOffline, unittest.TestCase):

    def test_a_hook_keeps_only_the_branch_its_output_drives(self):
        scoped, dropped = scope_to_hook(canvas(), hook("5", "348"))
        self.assertIn("348", scoped)          # its own generator
        self.assertIn("349", scoped)          # and where that lands
        self.assertNotIn("283", scoped)       # not the other stage
        self.assertNotIn("284", scoped)
        # The ref note goes too: nothing reads it, so it is neither downstream of
        # this hook's target nor an ancestor of anything that survived.
        self.assertEqual(sorted(dropped), ["283", "284", "8"])
        self.assertIn("7", scoped)            # the image node 348 really reads

    def test_the_other_hook_keeps_the_other_branch(self):
        scoped, _ = scope_to_hook(canvas(), hook("30", "283"))
        self.assertIn("283", scoped)
        self.assertNotIn("348", scoped)

    def test_stages_genuinely_wired_together_stay_together(self):
        """Then the graph really IS one chain, and splitting it would be wrong."""
        wired = canvas()
        wired["283"]["inputs"]["image"] = ["348", 0]
        scoped, _dropped = scope_to_hook(wired, hook("5", "348"))
        self.assertIn("283", scoped)
        self.assertIn("284", scoped)

    def test_it_never_hands_back_a_graph_that_renders_nothing(self):
        headless = {"348": {"class_type": "ByteDanceSeedreamNodeV2", "inputs": {"prompt": ""}},
                    "283": {"class_type": "ByteDanceSeedanceNode", "inputs": {"prompt": ""}},
                    "284": {"class_type": "SaveVideo", "inputs": {"video": ["283", 0]}}}
        scoped, dropped = scope_to_hook(headless, hook("5", "348"))
        self.assertEqual(dropped, [])
        self.assertEqual(scoped, headless)

    def test_a_hook_with_no_target_scopes_nothing(self):
        h = hook("5", "348")
        h["targets"] = []
        scoped, dropped = scope_to_hook(canvas(), h)
        self.assertEqual(dropped, [])
        self.assertEqual(scoped, canvas())


class DeadNodesTest(SchemaOffline, unittest.TestCase):

    def test_a_reference_that_only_fed_the_hook_is_dropped(self):
        """Its consumer was spliced out; it cannot reach the render."""
        graph = canvas()
        del graph["348"]["inputs"]["model.images.image_1"]   # splicing removed it
        pruned, dropped = prune_dead_nodes(graph)
        self.assertNotIn("7", pruned)
        self.assertNotIn("8", pruned)                # the note goes with it
        self.assertEqual(sorted(dropped), ["7", "8"])

    def test_the_same_image_is_kept_when_something_real_uses_it(self):
        pruned, dropped = prune_dead_nodes(canvas())
        self.assertIn("7", pruned)                   # node 348 reads it
        self.assertEqual(dropped, ["8"])             # only the note is dead

    def test_it_never_drops_an_output_node(self):
        graph = {"1": {"class_type": "KSampler", "inputs": {}},
                 "2": {"class_type": "SaveImage", "inputs": {"images": ["1", 0]}},
                 "3": {"class_type": "PreviewImage", "inputs": {"images": ["1", 0]}}}
        pruned, dropped = prune_dead_nodes(graph)
        self.assertEqual(dropped, [])
        self.assertEqual(pruned, graph)

    def test_a_class_it_cannot_ask_about_is_kept(self):
        """Keeping a useless node costs a line of JSON; dropping a saver costs the render."""
        with mock.patch("src.utils.preflight._schema", return_value={}):
            graph = {"1": {"class_type": "SomeThirdPartyThing", "inputs": {}}}
            pruned, dropped = prune_dead_nodes(graph)
        self.assertEqual(dropped, [])
        self.assertEqual(pruned, graph)

    def test_a_healthy_graph_is_returned_untouched(self):
        graph = {"1": {"class_type": "KSampler", "inputs": {}},
                 "2": {"class_type": "SaveImage", "inputs": {"images": ["1", 0]}}}
        self.assertIs(prune_dead_nodes(graph)[0], graph)


class ThroughTheToolTest(unittest.TestCase):
    """What the agent actually gets back from one apply_canvas_hooks call."""

    @staticmethod
    def _apply(resolutions, hooks):
        pipe = pipeline_stub(_canvas_base_prompt=canvas(), _canvas_hooks=hooks,
                             _dry_run=True)
        with mock.patch("agenty_core.tools.comfyui.open_workflow_in_canvas"), \
             mock.patch("src.executor._autoload_workflows_into_canvas", return_value=False):
            return json.loads(asyncio.run(
                tools(pipe)["apply_canvas_hooks"](resolutions)))

    def _graphs(self, out):
        from pathlib import Path
        return [json.loads(Path(v["workflow"]).read_text(encoding="utf-8"))
                for v in out["variants"]]

    def test_a_reference_sweep_carries_no_video_node(self):
        out = self._apply(
            [{"target_node_id": "348", "param": "prompt", "mode": "value_list",
              "values": ["Ben", "Ana", "Cy"]}],
            [hook("5", "348"), hook("30", "283")])
        self.assertEqual(out["count"], 3)
        for graph in self._graphs(out):
            self.assertIn("348", graph)
            self.assertNotIn("283", graph, "the video stage rode along on a reference run")
            self.assertNotIn("284", graph)

    def test_the_context_images_do_not_ride_along_either(self):
        """They fed the hook, the hook is gone, and they cannot affect the result."""
        out = self._apply(
            [{"target_node_id": "283", "param": "prompt", "mode": "value_list",
              "values": ["a slow push-in"]}],
            [hook("30", "283")])
        graph, = self._graphs(out)
        self.assertIn("283", graph)
        self.assertNotIn("7", graph)
        self.assertNotIn("8", graph)

    def test_it_says_what_it_left_out(self):
        out = self._apply(
            [{"target_node_id": "348", "param": "prompt", "mode": "value_list",
              "values": ["Ben"]}],
            [hook("5", "348"), hook("30", "283")])
        joined = " ".join(str(n) for n in out["notes"])
        self.assertIn("scoped to hook 5", joined)

    def test_the_image_a_variant_selects_survives_the_trim(self):
        """The one that got away: trimming BEFORE the build broke the build.

        A hook feeding a connection input is spliced off it, so at base-graph time
        the reference images are orphans — their only consumer was the hook. The
        agent then selects one per variant BY NODE ID, and `as_connection` resolves
        that id against the graph. Trim first and the id names a node that is no
        longer there: the lookup falls through, the input is skipped, and the
        reference workflows come out with no images in them.
        """
        graph = canvas()
        del graph["348"]["inputs"]["model.images.image_1"]   # what splicing leaves
        for nid, fn in (("9", "ana.png"), ("11", "cy.png")):
            graph[nid] = {"class_type": "LoadImage", "inputs": {"image": fn}}
        pipe = pipeline_stub(_canvas_base_prompt=graph, _dry_run=True,
                             _canvas_hooks=[hook(
                                 "5", "348",
                                 extra=[("model.images.image_1", "IMAGE")])])
        with mock.patch("agenty_core.tools.comfyui.open_workflow_in_canvas"), \
             mock.patch("src.executor._autoload_workflows_into_canvas", return_value=False):
            out = json.loads(asyncio.run(tools(pipe)["apply_canvas_hooks"]([
                {"target_node_id": "348", "param": "prompt", "mode": "value_list",
                 "values": ["Ben", "Ana", "Cy"], "zip_group": "shot"},
                {"target_node_id": "348", "mode": "value_list",
                 "param": "model.images.image_1",
                 "values": ["7", "9", "11"], "zip_group": "shot"}])))
        self.assertEqual(out["count"], 3)
        picked = []
        for graph_v in self._graphs(out):
            wire = graph_v["348"]["inputs"].get("model.images.image_1")
            self.assertIsInstance(wire, list, "the selected image was not wired in")
            picked.append(wire[0])
            self.assertIn(wire[0], graph_v, "wired to a node that is not in the graph")
        self.assertEqual(picked, ["7", "9", "11"])

    def test_each_variant_keeps_only_the_reference_it_uses(self):
        """The other two are unwired in THIS variant, so they are dead in it."""
        graph = canvas()
        del graph["348"]["inputs"]["model.images.image_1"]
        graph["9"] = {"class_type": "LoadImage", "inputs": {"image": "ana.png"}}
        pipe = pipeline_stub(_canvas_base_prompt=graph, _dry_run=True,
                             _canvas_hooks=[hook(
                                 "5", "348", param="model.images.image_1",
                                 param_type="IMAGE")])
        with mock.patch("agenty_core.tools.comfyui.open_workflow_in_canvas"), \
             mock.patch("src.executor._autoload_workflows_into_canvas", return_value=False):
            out = json.loads(asyncio.run(tools(pipe)["apply_canvas_hooks"]([
                {"target_node_id": "348", "mode": "value_list",
                 "param": "model.images.image_1",
                 # Two variants, one reference each — which on a numbered slot is
                 # now one of two readings, so it has to be said (the other is one
                 # run using both). See test_slot_fan_out.
                 "values": ["7", "9"], "sweep": True}])))
        first, second = self._graphs(out)
        self.assertIn("7", first)
        self.assertNotIn("9", first)
        self.assertIn("9", second)
        self.assertNotIn("7", second)

    def test_running_the_canvas_as_it_stands_is_left_whole(self):
        """No resolutions means "run what I drew" — there is no one stage to pick."""
        out = self._apply([], [hook("5", "348"), hook("30", "283")])
        graph, = self._graphs(out)
        for nid in ("348", "283", "284"):
            self.assertIn(nid, graph)


if __name__ == "__main__":
    unittest.main()
