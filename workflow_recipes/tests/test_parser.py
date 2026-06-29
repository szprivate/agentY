"""Unit tests for the parser: UI-vs-API equivalence, subgraph expansion,
malformed-input robustness."""

import json
import os
import tempfile
import unittest

from workflow_recipes import parser as P


# A tiny text-to-image graph expressed two ways: UI and API. They describe the
# same structure (CheckpointLoader -> CLIPTextEncode -> KSampler -> VAEDecode ->
# SaveImage, with an EmptyLatentImage feeding the sampler) and must normalize to
# the same node classes and connection shape.
UI_WORKFLOW = {
    "nodes": [
        {"id": 1, "type": "CheckpointLoaderSimple", "widgets_values": ["sd.safetensors"]},
        {"id": 2, "type": "CLIPTextEncode", "widgets_values": ["a cat"]},
        {"id": 3, "type": "EmptyLatentImage", "widgets_values": [512, 512, 1]},
        {"id": 4, "type": "KSampler", "widgets_values": [42, "euler"]},
        {"id": 5, "type": "VAEDecode", "widgets_values": []},
        {"id": 6, "type": "SaveImage", "widgets_values": ["out"]},
    ],
    "links": [
        [10, 1, 0, 4, 0, "MODEL"],
        [11, 1, 1, 2, 0, "CLIP"],
        [12, 2, 0, 4, 1, "CONDITIONING"],
        [13, 3, 0, 4, 3, "LATENT"],
        [14, 4, 0, 5, 0, "LATENT"],
        [15, 1, 2, 5, 1, "VAE"],
        [16, 5, 0, 6, 0, "IMAGE"],
    ],
}

API_WORKFLOW = {
    "1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": "sd.safetensors"}},
    "2": {"class_type": "CLIPTextEncode", "inputs": {"text": "a cat", "clip": ["1", 1]}},
    "3": {"class_type": "EmptyLatentImage", "inputs": {"width": 512, "height": 512}},
    "4": {"class_type": "KSampler", "inputs": {
        "seed": 42, "model": ["1", 0], "positive": ["2", 0], "latent_image": ["3", 0]}},
    "5": {"class_type": "VAEDecode", "inputs": {"samples": ["4", 0], "vae": ["1", 2]}},
    "6": {"class_type": "SaveImage", "inputs": {"images": ["5", 0]}},
}

# Object info giving CheckpointLoaderSimple typed outputs so API edge types resolve.
OBJECT_INFO = {
    "CheckpointLoaderSimple": {"python_module": "nodes", "input": {}, "output": ["MODEL", "CLIP", "VAE"]},
    "CLIPTextEncode": {"python_module": "nodes", "input": {}, "output": ["CONDITIONING"]},
    "EmptyLatentImage": {"python_module": "nodes", "input": {}, "output": ["LATENT"]},
    "KSampler": {"python_module": "nodes", "input": {}, "output": ["LATENT"]},
    "VAEDecode": {"python_module": "nodes", "input": {}, "output": ["IMAGE"]},
    "SaveImage": {"python_module": "nodes", "input": {}, "output": []},
}


def _classes(graph):
    return sorted(n.class_type for n in graph.nodes.values())


def _conn_classes(graph):
    """Connection patterns as (src_class, dst_class, data_type) sets."""
    return {
        (graph.class_of(e.src_id), graph.class_of(e.dst_id), e.data_type)
        for e in graph.edges
    }


class TestFormatDetection(unittest.TestCase):
    def test_detect_ui(self):
        self.assertEqual(P.detect_format(UI_WORKFLOW), "ui")

    def test_detect_api(self):
        self.assertEqual(P.detect_format(API_WORKFLOW), "api")

    def test_detect_unknown(self):
        self.assertIsNone(P.detect_format({"foo": "bar"}))
        self.assertIsNone(P.detect_format([1, 2, 3]))


class TestUiApiEquivalence(unittest.TestCase):
    def setUp(self):
        self.ui = P.enrich(P.parse_ui(UI_WORKFLOW, "ui", "ui.json", "official"), OBJECT_INFO)
        self.api = P.enrich(P.parse_api(API_WORKFLOW, "api", "api.json", "official"), OBJECT_INFO)

    def test_same_node_classes(self):
        self.assertEqual(_classes(self.ui), _classes(self.api))

    def test_same_connection_patterns(self):
        # The UI form carries types directly; the API form infers them from
        # object_info. The (src_class -> dst_class, type) sets must match.
        self.assertEqual(_conn_classes(self.ui), _conn_classes(self.api))

    def test_api_edge_types_inferred(self):
        # No edge should remain UNKNOWN given full object_info.
        self.assertTrue(all(e.data_type != "UNKNOWN" for e in self.api.edges))


class TestSubgraphExpansion(unittest.TestCase):
    def test_single_collapsed_subgraph_is_expanded(self):
        # A workflow that is one subgraph node referencing a 3-node subgraph must
        # expand to the 3 real nodes, not stay a single opaque UUID node.
        sub_id = "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"
        wf = {
            "nodes": [{"id": 99, "type": sub_id, "widgets_values": []}],
            "links": [],
            "definitions": {"subgraphs": [{
                "id": sub_id,
                "inputNode": {"id": -10},
                "outputNode": {"id": -20},
                "inputs": [{"name": "prompt", "type": "STRING"}],
                "outputs": [{"name": "IMAGE", "type": "IMAGE"}],
                "nodes": [
                    {"id": 1, "type": "CLIPTextEncode", "widgets_values": ["x"]},
                    {"id": 2, "type": "KSampler", "widgets_values": []},
                    {"id": 3, "type": "VAEDecode", "widgets_values": []},
                ],
                "links": [
                    {"id": 1, "origin_id": 1, "origin_slot": 0, "target_id": 2, "target_slot": 1, "type": "CONDITIONING"},
                    {"id": 2, "origin_id": 2, "origin_slot": 0, "target_id": 3, "target_slot": 0, "type": "LATENT"},
                    {"id": 3, "origin_id": -10, "origin_slot": 0, "target_id": 1, "target_slot": 0, "type": "STRING"},
                    {"id": 4, "origin_id": 3, "origin_slot": 0, "target_id": -20, "target_slot": 0, "type": "IMAGE"},
                ],
            }]},
        }
        g = P.parse_ui(wf, "sg", "sg.json", "official")
        self.assertEqual(_classes(g), ["CLIPTextEncode", "KSampler", "VAEDecode"])
        # Internal edges preserved.
        self.assertIn(("CLIPTextEncode", "KSampler", "CONDITIONING"), _conn_classes(g))
        self.assertIn(("KSampler", "VAEDecode", "LATENT"), _conn_classes(g))
        # Explicit boundary ports recovered from the subgraph definition.
        self.assertEqual(g.boundary_inputs, [{"name": "prompt", "data_type": "STRING"}])
        self.assertEqual(g.boundary_outputs, [{"name": "IMAGE", "data_type": "IMAGE"}])

    def test_nested_subgraph_boundary_rewire(self):
        # An outer subgraph contains an inner subgraph; a link crossing the inner
        # boundary must become a direct real-to-real edge after flattening.
        inner = "11111111-1111-1111-1111-111111111111"
        outer = "22222222-2222-2222-2222-222222222222"
        wf = {
            "nodes": [{"id": 5, "type": outer, "widgets_values": []}],
            "links": [],
            "definitions": {"subgraphs": [
                {
                    "id": outer,
                    "inputNode": {"id": -10}, "outputNode": {"id": -20},
                    "inputs": [], "outputs": [{"name": "IMAGE", "type": "IMAGE"}],
                    "nodes": [
                        {"id": 1, "type": "KSampler", "widgets_values": []},
                        {"id": 2, "type": inner, "widgets_values": []},
                    ],
                    # KSampler LATENT -> inner subgraph input slot 0; inner output -> boundary.
                    "links": [
                        {"id": 1, "origin_id": 1, "origin_slot": 0, "target_id": 2, "target_slot": 0, "type": "LATENT"},
                        {"id": 2, "origin_id": 2, "origin_slot": 0, "target_id": -20, "target_slot": 0, "type": "IMAGE"},
                    ],
                },
                {
                    "id": inner,
                    "inputNode": {"id": -10}, "outputNode": {"id": -20},
                    "inputs": [{"name": "samples", "type": "LATENT"}],
                    "outputs": [{"name": "IMAGE", "type": "IMAGE"}],
                    "nodes": [{"id": 1, "type": "VAEDecode", "widgets_values": []}],
                    "links": [
                        {"id": 1, "origin_id": -10, "origin_slot": 0, "target_id": 1, "target_slot": 0, "type": "LATENT"},
                        {"id": 2, "origin_id": 1, "origin_slot": 0, "target_id": -20, "target_slot": 0, "type": "IMAGE"},
                    ],
                },
            ]},
        }
        g = P.parse_ui(wf, "nested", "nested.json", "official")
        self.assertEqual(_classes(g), ["KSampler", "VAEDecode"])
        # The cross-boundary KSampler(LATENT) -> VAEDecode edge must be present.
        self.assertIn(("KSampler", "VAEDecode", "LATENT"), _conn_classes(g))


class TestRobustness(unittest.TestCase):
    def test_malformed_json_skipped(self):
        with tempfile.TemporaryDirectory() as d:
            bad = os.path.join(d, "bad.json")
            with open(bad, "w", encoding="utf-8") as f:
                f.write("{ this is not json ")
            logs = []
            result = P.parse_file(bad, "official", {}, log=logs.append)
            self.assertIsNone(result)
            self.assertTrue(any("bad.json" in m for m in logs))

    def test_index_files_skipped(self):
        with tempfile.TemporaryDirectory() as d:
            idx = os.path.join(d, "index.json")
            with open(idx, "w", encoding="utf-8") as f:
                json.dump({"templates": []}, f)
            self.assertIsNone(P.parse_file(idx, "official", {}))

    def test_unresolved_classes_flagged(self):
        wf = {"nodes": [{"id": 1, "type": "TotallyCustomNode", "widgets_values": []}], "links": []}
        g = P.enrich(P.parse_ui(wf, "c", "c.json", "custom"), OBJECT_INFO)
        self.assertIn("TotallyCustomNode", g.unresolved_classes)


if __name__ == "__main__":
    unittest.main()
