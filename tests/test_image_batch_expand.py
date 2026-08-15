"""A batch aimed at a slot that takes one image.

An `agentY image collector` emits its files as a single IMAGE batch. The API model
nodes take references in numbered single-image slots. Wire one into the other and
the node reads the FIRST image and ignores the rest: five references handed in, a
render built from one, and nothing anywhere reports an error.

The rewrite has one right answer, so it is done in code rather than asked of the
agent — which had the node available and did not reach for it. The wire is routed
through an `agentY expand image batch` and fanned across the slots the target
actually declares.

"Actually declares" is the whole safety story. The names are not at the top of the
schema: Seedream keeps them under a dynamic combo, `model` → options → `images`
(COMFY_AUTOGROW_V3) → template.names, which is why the prompt addresses one as
`model.images.image_1`. Nothing is wired that the schema has not named.

    python -m unittest discover -s tests
"""

import unittest
from unittest import mock

from src.utils.canvas_hooks import autogrow_slots, expand_image_batches

# Trimmed to shape from the live /object_info for ByteDanceSeedreamNodeV2.
SEEDREAM = {
    "input": {"required": {
        "prompt": ["STRING", {}],
        "model": ["COMFY_DYNAMICCOMBO_V3", {"options": [
            {"key": "seedream 5.0 pro", "inputs": {"required": {
                "width": ["INT", {}],
                "images": ["COMFY_AUTOGROW_V3", {"template": {
                    "names": [f"image_{i}" for i in range(1, 11)], "min": 0}}],
            }}},
        ]}],
    }},
}
EXPANDER = {"input": {"required": {"images": ["IMAGE", {}]}}}
PLAIN = {"input": {"required": {"images": ["IMAGE", {}]}}}


def graph(files=("a.png", "b.png", "c.png"), slot="model.images.image_1",
          cls="ByteDanceSeedreamNodeV2"):
    return {"60": {"class_type": "AgentYImageCollector",
                   "inputs": {"files": "\n".join(files)}},
            "348": {"class_type": cls, "inputs": {"prompt": "x", slot: ["60", 0]}},
            "349": {"class_type": "SaveImage", "inputs": {"images": ["348", 0]}}}


def schemas(**over):
    """Patch the schema lookup; unknown classes answer {} (ComfyUI can't say)."""
    table = {"ByteDanceSeedreamNodeV2": SEEDREAM,
             "AgentYImageBatchExpand": EXPANDER}
    table.update(over)
    return mock.patch("src.utils.preflight._schema",
                      side_effect=lambda c: table.get(c, {}))


class SlotDiscoveryTest(unittest.TestCase):

    def test_the_names_are_found_under_the_dynamic_combo(self):
        with schemas():
            self.assertEqual(autogrow_slots("ByteDanceSeedreamNodeV2"),
                             {"model.images": [f"image_{i}" for i in range(1, 11)]})

    def test_a_class_comfyui_cannot_describe_yields_nothing(self):
        with schemas():
            self.assertEqual(autogrow_slots("SomethingElse"), {})


class ExpansionTest(unittest.TestCase):

    def test_the_batch_is_fanned_across_the_slots(self):
        with schemas():
            out, notes = expand_image_batches(graph())
        eid = out["348"]["inputs"]["model.images.image_1"][0]
        self.assertEqual(out[eid]["class_type"], "AgentYImageBatchExpand")
        self.assertEqual(out[eid]["inputs"]["images"], ["60", 0])
        for k in range(3):
            self.assertEqual(out["348"]["inputs"][f"model.images.image_{k + 1}"],
                             [eid, k])
        self.assertIn("wired to 3 slot(s)", notes[0])

    def test_the_collector_is_left_where_it_was(self):
        """It still holds the file list; only what reads it changed."""
        with schemas():
            out, _ = expand_image_batches(graph())
        self.assertEqual(out["60"]["inputs"]["files"], "a.png\nb.png\nc.png")

    def test_one_image_is_not_a_batch(self):
        with schemas():
            out, notes = expand_image_batches(graph(files=("only.png",)))
        self.assertEqual(notes, [])
        self.assertEqual(out["348"]["inputs"]["model.images.image_1"], ["60", 0])

    def test_a_plural_input_that_wants_the_batch_is_left_alone(self):
        with schemas(PreviewImage=PLAIN):
            out, notes = expand_image_batches({
                "60": {"class_type": "AgentYImageCollector",
                       "inputs": {"files": "a.png\nb.png"}},
                "9": {"class_type": "PreviewImage", "inputs": {"images": ["60", 0]}}})
        self.assertEqual(notes, [])
        self.assertEqual(out["9"]["inputs"]["images"], ["60", 0])

    def test_nothing_is_written_when_the_expander_is_not_installed(self):
        """A graph naming a node ComfyUI does not have fails validation."""
        with mock.patch("src.utils.preflight._schema",
                        side_effect=lambda c: {} if c == "AgentYImageBatchExpand"
                        else SEEDREAM):
            out, notes = expand_image_batches(graph())
        self.assertEqual(notes, [])
        self.assertEqual(out["348"]["inputs"]["model.images.image_1"], ["60", 0])

    def test_nothing_is_written_when_the_slots_are_not_declared(self):
        with schemas(ByteDanceSeedreamNodeV2={}):
            out, notes = expand_image_batches(graph())
        self.assertEqual(notes, [])
        self.assertEqual(out["348"]["inputs"]["model.images.image_1"], ["60", 0])

    def test_it_stops_at_the_expander_s_own_ceiling(self):
        with schemas():
            out, notes = expand_image_batches(
                graph(files=tuple(f"{i}.png" for i in range(10))))
        wired = [k for k in out["348"]["inputs"] if k.startswith("model.images.")]
        self.assertEqual(len(wired), 8)
        self.assertIn("2 more could not be placed", notes[0])

    def test_starting_part_way_along_only_uses_what_is_left(self):
        with schemas():
            out, _ = expand_image_batches(
                graph(files=("a.png", "b.png", "c.png"),
                      slot="model.images.image_9"))
        wired = sorted(k for k in out["348"]["inputs"] if k.startswith("model.images."))
        self.assertEqual(wired, ["model.images.image_10", "model.images.image_9"])

    def test_a_graph_with_no_collector_is_returned_untouched(self):
        plain = {"1": {"class_type": "KSampler", "inputs": {}},
                 "2": {"class_type": "SaveImage", "inputs": {"images": ["1", 0]}}}
        with schemas():
            out, notes = expand_image_batches(plain)
        self.assertIs(out, plain)
        self.assertEqual(notes, [])


if __name__ == "__main__":
    unittest.main()
