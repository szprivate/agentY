"""Unit tests for recipe synthesis: invariant vs optional roles, the critical
paired-node case, boundary ports, and the annotation policy."""

import unittest

from workflow_recipes import parser as P
from workflow_recipes import recipe_builder as R
from workflow_recipes.cluster import Cluster


def _graph(nodes, links, name, source="official", object_info=None):
    wf = {"nodes": nodes, "links": links}
    g = P.parse_ui(wf, name, name + ".json", source)
    return P.enrich(g, object_info or {})


def _single_cluster(graphs):
    return [Cluster(members=list(range(len(graphs))), cohesion=1.0)]


# Core object_info so these graphs count as pure-core (no custom/unresolved).
CORE_INFO = {
    "UNETLoader": {"python_module": "nodes", "input": {}, "output": ["MODEL"]},
    "KSampler": {"python_module": "nodes", "input": {}, "output": ["LATENT"]},
    "VAEDecode": {"python_module": "nodes", "input": {}, "output": ["IMAGE"]},
    "SaveImage": {"python_module": "nodes", "input": {}, "output": []},
    "CLIPTextEncode": {"python_module": "nodes", "input": {}, "output": ["CONDITIONING"]},
    "LoraLoader": {"python_module": "nodes", "input": {}, "output": ["MODEL", "CLIP"]},
}


class TestPairedNodes(unittest.TestCase):
    """The critical case: a type that always uses TWO model loaders (e.g. a
    high-noise/low-noise WAN pair) must surface both as required, not collapse
    them into a single role."""

    def _two_loader_graph(self, name):
        # Two UNETLoaders, each feeding its own KSampler; both samplers feed a
        # shared VAEDecode -> SaveImage.
        nodes = [
            {"id": 1, "type": "UNETLoader", "widgets_values": ["high.safetensors"]},
            {"id": 2, "type": "UNETLoader", "widgets_values": ["low.safetensors"]},
            {"id": 3, "type": "KSampler", "widgets_values": [1]},
            {"id": 4, "type": "KSampler", "widgets_values": [2]},
            {"id": 5, "type": "VAEDecode", "widgets_values": []},
            {"id": 6, "type": "SaveImage", "widgets_values": ["out"]},
        ]
        links = [
            [1, 1, 0, 3, 0, "MODEL"],
            [2, 2, 0, 4, 0, "MODEL"],
            [3, 3, 0, 4, 3, "LATENT"],
            [4, 4, 0, 5, 0, "LATENT"],
            [5, 5, 0, 6, 0, "IMAGE"],
        ]
        return _graph(nodes, links, name, object_info=CORE_INFO)

    def test_two_loaders_both_required(self):
        graphs = [self._two_loader_graph("a"), self._two_loader_graph("b")]
        recipes = R.build_recipes(graphs, _single_cluster(graphs), object_info_available=True)
        recipe = recipes[0]
        loader = [e for e in recipe["required_node_roles"] if e["node_class"] == "UNETLoader"]
        self.assertEqual(len(loader), 1, "should be one aggregated entry for the class")
        entry = loader[0]
        # The pairing must be preserved: two instances are required, not one.
        self.assertTrue(entry.get("paired_or_multiple"))
        self.assertEqual(entry["min_instances"], 2)
        self.assertIn("2 required instances", entry["frequency"])

    def test_two_loaders_not_collapsed_when_one_member_has_one(self):
        # If one member has only a single loader, multiplicity is NOT guaranteed:
        # min_instances drops to 1 and it is no longer flagged as paired.
        two = self._two_loader_graph("two")
        one_nodes = [
            {"id": 1, "type": "UNETLoader", "widgets_values": ["x"]},
            {"id": 3, "type": "KSampler", "widgets_values": [1]},
            {"id": 5, "type": "VAEDecode", "widgets_values": []},
            {"id": 6, "type": "SaveImage", "widgets_values": ["o"]},
        ]
        one_links = [[1, 1, 0, 3, 0, "MODEL"], [4, 3, 0, 5, 0, "LATENT"], [5, 5, 0, 6, 0, "IMAGE"]]
        one = _graph(one_nodes, one_links, "one", object_info=CORE_INFO)
        graphs = [two, one]
        recipes = R.build_recipes(graphs, _single_cluster(graphs), object_info_available=True)
        entry = [e for e in recipes[0]["required_node_roles"] if e["node_class"] == "UNETLoader"][0]
        self.assertEqual(entry["min_instances"], 1)
        self.assertFalse(entry.get("paired_or_multiple", False))


class TestInvariantVsOptional(unittest.TestCase):
    def test_invariant_and_optional_split(self):
        # KSampler/VAEDecode/SaveImage in both; LoraLoader only in one.
        base_nodes = [
            {"id": 1, "type": "KSampler", "widgets_values": [1]},
            {"id": 2, "type": "VAEDecode", "widgets_values": []},
            {"id": 3, "type": "SaveImage", "widgets_values": ["o"]},
        ]
        base_links = [[1, 1, 0, 2, 0, "LATENT"], [2, 2, 0, 3, 0, "IMAGE"]]
        g1 = _graph(base_nodes, base_links, "g1", object_info=CORE_INFO)
        g2_nodes = base_nodes + [{"id": 4, "type": "LoraLoader", "widgets_values": ["l"]}]
        g2_links = base_links + [[3, 4, 0, 1, 0, "MODEL"]]
        g2 = _graph(g2_nodes, g2_links, "g2", object_info=CORE_INFO)
        graphs = [g1, g2]
        recipe = R.build_recipes(graphs, _single_cluster(graphs), object_info_available=True)[0]
        req_classes = {e["node_class"] for e in recipe["required_node_roles"]}
        opt_classes = {e["node_class"] for e in recipe["optional_node_roles"]}
        self.assertEqual(req_classes, {"KSampler", "VAEDecode", "SaveImage"})
        self.assertEqual(opt_classes, {"LoraLoader"})


class TestSelfContainedRecords(unittest.TestCase):
    def test_no_annotation_fields(self):
        # The database is self-contained: the old human-in-the-loop fields are gone.
        nodes = [{"id": 1, "type": "MyCustomThing", "widgets_values": []}]
        g = _graph(nodes, [], "c", source="custom", object_info={})
        recipe = R.build_recipes([g], _single_cluster([g]), object_info_available=True)[0]
        for gone in ("needs_annotation", "annotation_reason", "notes_for_annotation"):
            self.assertNotIn(gone, recipe)

    def test_custom_type_gets_synthesized_description(self):
        nodes = [{"id": 1, "type": "MyCustomThing", "widgets_values": []}]
        g = _graph(nodes, [], "c", source="custom", object_info={})  # unresolved
        recipe = R.build_recipes([g], _single_cluster([g]), object_info_available=True)[0]
        self.assertTrue(recipe["description"])                  # never blank
        self.assertEqual(recipe["description_source"], "synthesized")
        self.assertIn("MyCustomThing", recipe["unresolved_nodes"])
        self.assertIn("user_intent", recipe)

    def test_core_type_description_is_clean(self):
        nodes = [
            {"id": 1, "type": "KSampler", "widgets_values": [1]},
            {"id": 2, "type": "VAEDecode", "widgets_values": []},
            {"id": 3, "type": "SaveImage", "widgets_values": ["o"]},
        ]
        links = [[1, 1, 0, 2, 0, "LATENT"], [2, 2, 0, 3, 0, "IMAGE"]]
        g = _graph(nodes, links, "core", object_info=CORE_INFO)
        recipe = R.build_recipes([g], _single_cluster([g]), object_info_available=True)[0]
        self.assertTrue(recipe["description"])
        # No warning / draft language anywhere in the description.
        self.assertNotIn("DRAFT", recipe["description"])
        self.assertNotIn("FLAG", recipe["description"])


class TestNodeKnowledge(unittest.TestCase):
    def test_signature_and_usage(self):
        nodes = [
            {"id": 1, "type": "KSampler", "widgets_values": [1]},
            {"id": 2, "type": "VAEDecode", "widgets_values": []},
        ]
        g = _graph(nodes, [[1, 1, 0, 2, 0, "LATENT"]], "g1", object_info=CORE_INFO)
        recipes = R.build_recipes([g], _single_cluster([g]), object_info_available=True)
        nk = R.build_node_knowledge([g], recipes, CORE_INFO)
        by_class = {n["class"]: n for n in nk}
        self.assertIn("KSampler", by_class)
        ks = by_class["KSampler"]
        self.assertEqual(ks["outputs"], ["LATENT"])
        self.assertEqual(ks["role"], "sampler")
        self.assertEqual(ks["used_in_type_ids"], [recipes[0]["id"]])
        self.assertEqual(ks["occurrences"], 1)


class TestDeterminismAndIds(unittest.TestCase):
    def test_ids_unique_and_sorted_by_member_count(self):
        def mk(name):
            nodes = [
                {"id": 1, "type": "KSampler", "widgets_values": [1]},
                {"id": 2, "type": "VAEDecode", "widgets_values": []},
            ]
            return _graph(nodes, [[1, 1, 0, 2, 0, "LATENT"]], name, object_info=CORE_INFO)

        graphs = [mk("a"), mk("b"), mk("c")]
        clusters = [Cluster(members=[0, 1], cohesion=1.0), Cluster(members=[2], cohesion=1.0)]
        recipes = R.build_recipes(graphs, clusters, object_info_available=True)
        ids = [r["id"] for r in recipes]
        self.assertEqual(len(ids), len(set(ids)), "ids must be unique")
        # Larger cluster first.
        self.assertEqual(recipes[0]["member_count"], 2)
        self.assertEqual(recipes[1]["member_count"], 1)


if __name__ == "__main__":
    unittest.main()
