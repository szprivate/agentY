"""Unit tests for clustering: threshold behavior on a synthetic set, and
determinism."""

import unittest
from collections import Counter

from workflow_recipes import cluster as C
from workflow_recipes import fingerprint as F


def make_fp(name, classes, connections, spine=None):
    """Build a Fingerprint directly from simple structural facts (no parsing)."""
    return F.Fingerprint(
        name=name,
        source="synthetic",
        class_set=frozenset(classes),
        class_multiset=Counter(classes),
        connection_set=frozenset(connections),
        cluster_set=frozenset(f"{a}->{b}" for a, b, _ in connections),
        spine_set=frozenset(spine or []),
        node_count=len(classes),
    )


# Two tight families plus one outlier.
A1 = make_fp("a1", ["KSampler", "VAEDecode", "CLIPTextEncode", "UNETLoader"],
             [("model_loader", "sampler", "MODEL"), ("sampler", "vae_decode", "LATENT")],
             spine=["sampler", "model_loader", "vae_decode"])
A2 = make_fp("a2", ["KSampler", "VAEDecode", "CLIPTextEncode", "UNETLoader", "LoraLoader"],
             [("model_loader", "sampler", "MODEL"), ("sampler", "vae_decode", "LATENT")],
             spine=["sampler", "model_loader", "vae_decode"])
B1 = make_fp("b1", ["LoadImage", "Upscale", "SaveImage"],
             [("image_loader", "upscale", "IMAGE"), ("upscale", "save_output", "IMAGE")],
             spine=[])
B2 = make_fp("b2", ["LoadImage", "Upscale", "SaveImage", "ImageScale"],
             [("image_loader", "upscale", "IMAGE"), ("upscale", "save_output", "IMAGE")],
             spine=[])
OUT = make_fp("z_outlier", ["AudioLoad", "AudioEncode", "SaveAudio"],
              [("other", "other", "AUDIO")], spine=[])

FPS = [A1, A2, B1, B2, OUT]
WEIGHTS = F.DEFAULT_WEIGHTS


def cluster_names(clusters):
    return sorted(sorted(f.name for f in [FPS[m] for m in c.members]) for c in clusters)


class TestThresholdClustering(unittest.TestCase):
    def setUp(self):
        self.matrix = C.pairwise_matrix(FPS, WEIGHTS)

    def test_families_group_outlier_isolated(self):
        clusters = C.agglomerate(FPS, self.matrix, threshold=0.6)
        names = cluster_names(clusters)
        self.assertIn(["a1", "a2"], names)
        self.assertIn(["b1", "b2"], names)
        self.assertIn(["z_outlier"], names)

    def test_high_threshold_splits_everything(self):
        clusters = C.agglomerate(FPS, self.matrix, threshold=0.999)
        # Near-identical-but-not-equal members should fall into singletons.
        self.assertEqual(len(clusters), len(FPS))

    def test_low_threshold_merges_more(self):
        loose = C.agglomerate(FPS, self.matrix, threshold=0.0)
        self.assertEqual(len(loose), 1)


class TestDeterminism(unittest.TestCase):
    def test_same_inputs_same_clusters(self):
        m1 = C.pairwise_matrix(FPS, WEIGHTS)
        m2 = C.pairwise_matrix(FPS, WEIGHTS)
        c1 = C.agglomerate(FPS, m1, threshold=0.6)
        c2 = C.agglomerate(FPS, m2, threshold=0.6)
        self.assertEqual(cluster_names(c1), cluster_names(c2))

    def test_input_order_independence(self):
        # Reversing input order must yield the same grouping (by names).
        rev = list(reversed(FPS))
        mrev = C.pairwise_matrix(rev, WEIGHTS)
        crev = C.agglomerate(rev, mrev, threshold=0.6)
        names_rev = sorted(sorted(rev[m].name for m in c.members) for c in crev)
        cfwd = C.agglomerate(FPS, C.pairwise_matrix(FPS, WEIGHTS), threshold=0.6)
        self.assertEqual(names_rev, cluster_names(cfwd))


class TestCategorySignal(unittest.TestCase):
    """The opt-in catalog-category signal: off by default, neutral when a
    category is unknown, and able to nudge similarity when enabled."""

    def _pair(self, cat_a, cat_b, weights):
        fa = make_fp("a", ["KSampler", "VAEDecode"],
                     [("model_loader", "sampler", "MODEL")])
        fb = make_fp("b", ["KSampler", "VAEDecode"],
                     [("model_loader", "sampler", "MODEL")])
        fa.category_set = frozenset({cat_a}) if cat_a else frozenset()
        fb.category_set = frozenset({cat_b}) if cat_b else frozenset()
        return C.similarity_breakdown(fa, fb, weights)

    def test_default_weight_zero_ignores_category(self):
        # With default weights (category 0), differing categories do not matter.
        w = dict(F.DEFAULT_WEIGHTS)
        same, _ = self._pair("Image Tools", "Image Tools", w)
        diff, per = self._pair("Image Tools", "Video Tools", w)
        self.assertEqual(same, diff)
        self.assertNotIn("category", per)   # weight 0 -> not scored

    def test_enabled_category_changes_similarity(self):
        w = dict(F.DEFAULT_WEIGHTS); w["category"] = 0.5
        same, per_same = self._pair("Image Tools", "Image Tools", w)
        diff, _ = self._pair("Image Tools", "Video Tools", w)
        self.assertIn("category", per_same)
        self.assertEqual(per_same["category"], 1.0)
        self.assertGreater(same, diff)      # same category scores higher

    def test_unknown_category_is_neutral(self):
        # If either side lacks a category, the signal is excluded (not scored 0),
        # so the combined score equals the purely-structural score.
        w = dict(F.DEFAULT_WEIGHTS); w["category"] = 0.5
        struct_only = dict(F.DEFAULT_WEIGHTS); struct_only["category"] = 0.0
        with_unknown, per = self._pair("Image Tools", None, w)
        baseline, _ = self._pair("Image Tools", None, struct_only)
        self.assertNotIn("category", per)
        self.assertAlmostEqual(with_unknown, baseline)


class TestJaccard(unittest.TestCase):
    def test_basic(self):
        self.assertEqual(C.jaccard(frozenset({1, 2}), frozenset({1, 2})), 1.0)
        self.assertEqual(C.jaccard(frozenset({1, 2}), frozenset({3, 4})), 0.0)
        self.assertAlmostEqual(C.jaccard(frozenset({1, 2}), frozenset({2, 3})), 1 / 3)

    def test_empty_sets_equal(self):
        self.assertEqual(C.jaccard(frozenset(), frozenset()), 1.0)


if __name__ == "__main__":
    unittest.main()
