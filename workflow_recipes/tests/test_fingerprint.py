"""Unit tests for fingerprinting: stability under parameter changes, and role
classification."""

import copy
import unittest

from workflow_recipes import parser as P
from workflow_recipes import fingerprint as F
from workflow_recipes.tests.test_parser import UI_WORKFLOW, OBJECT_INFO


def _fp(wf):
    g = P.enrich(P.parse_ui(wf, "w", "w.json", "official"), OBJECT_INFO)
    return F.fingerprint(g)


class TestFingerprintStability(unittest.TestCase):
    def test_param_changes_do_not_change_fingerprint(self):
        # Same graph shape, different widget values -> identical fingerprint.
        base = _fp(UI_WORKFLOW)
        mutated = copy.deepcopy(UI_WORKFLOW)
        for node in mutated["nodes"]:
            node["widgets_values"] = ["COMPLETELY", "DIFFERENT", 9999]
        other = _fp(mutated)
        self.assertEqual(base.class_set, other.class_set)
        self.assertEqual(base.connection_set, other.connection_set)
        self.assertEqual(base.cluster_set, other.cluster_set)
        self.assertEqual(base.spine_set, other.spine_set)

    def test_node_id_renumbering_is_stable(self):
        # Renumbering node ids (keeping the same structure) must not change the
        # fingerprint - signatures are role/type based, not id based.
        renum = copy.deepcopy(UI_WORKFLOW)
        offset = 100
        for node in renum["nodes"]:
            node["id"] += offset
        for link in renum["links"]:
            link[1] += offset
            link[3] += offset
        self.assertEqual(_fp(UI_WORKFLOW).connection_set, _fp(renum).connection_set)


class TestRoleClassification(unittest.TestCase):
    def test_spine_roles_detected(self):
        fp = _fp(UI_WORKFLOW)
        self.assertIn("sampler", fp.spine_set)
        self.assertIn("model_loader", fp.spine_set)
        self.assertIn("vae_decode", fp.spine_set)
        self.assertIn("text_encode", fp.spine_set)

    def test_classify_known_roles(self):
        self.assertEqual(F.classify_role("KSampler"), "sampler")
        self.assertEqual(F.classify_role("UNETLoader"), "model_loader")
        self.assertEqual(F.classify_role("VAEDecode"), "vae_decode")
        self.assertEqual(F.classify_role("SaveImage"), "save_output")
        self.assertEqual(F.classify_role("SomethingWeird"), "other")


if __name__ == "__main__":
    unittest.main()
