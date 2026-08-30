"""A score for ranking outputs, and weights learned from what the user keeps.

Three things are being held down here, and they are different in kind.

**The score must not become a gate.** A weighted sum lets a strong feature pay
for a weak one, which is what ranking wants and the opposite of what "must be
16:9" wants. The gates live in qa_checks and nothing here may leak into them.

**A missing measurement is not a zero.** A video has no exposure and a still has
no motion. If absence read as zero, every clip would rank below every image for
reasons that are about the file type, not the picture.

**And the fit has to actually work.** A learned model quietly worse than the
guess it replaced is the characteristic failure of this kind of work, so the fit
is checked against a known ground truth it has to recover, and the installer is
checked to refuse anything that does not beat the defaults on held-out data.

    python -m unittest discover -s tests
"""

import json
import random
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from src.utils import fitness, preference_log
from src.utils.fitness import (DEFAULT_WEIGHTS, FEATURES, features, rank_files,
                               render_score, score)
from src.utils.fitness_fit import (active_keys, evaluate, fit, pair_accuracy,
                                   pairs_of, split, top1_accuracy)


def facts(**over):
    """A well-exposed, sharp, clean 1920x1080 still."""
    base = {
        "width": 1920, "height": 1080,
        "sharpness": {"score": 300.0, "band": "very sharp",
                      "sharpest_region": 900.0, "sharpest_band": "very sharp"},
        "noise": {"sigma": 1.0, "band": "clean"},
        "exposure": {"mean": 120.0, "contrast": 50.0,
                     "clipped_black": 0.0, "clipped_white": 0.0},
    }
    base.update(over)
    return base


def clip(**over):
    base = {"width": 1920, "height": 1080,
            "sharpness": {"score": 200.0, "band": "very sharp"},
            "noise": {"sigma": 1.0, "band": "clean"},
            "frames_sampled": 9, "black_frames": 0, "frozen_pairs": 0}
    base.update(over)
    return base


class FeatureTest(unittest.TestCase):

    def test_a_sharp_clean_still_scores_near_the_top_of_every_feature(self):
        f = features(facts())
        for k in ("detail", "focus", "cleanliness", "headroom"):
            self.assertGreater(f[k], 0.7, k)

    def test_a_soft_grainy_blown_render_scores_near_the_bottom(self):
        f = features(facts(
            sharpness={"score": 6.0, "band": "very soft", "sharpest_region": 15.0},
            noise={"sigma": 8.0, "band": "grainy"},
            exposure={"mean": 200.0, "contrast": 20.0,
                      "clipped_black": 0.0, "clipped_white": 0.30}))
        for k in ("detail", "focus", "cleanliness", "headroom"):
            self.assertLess(f[k], 0.3, k)

    def test_a_typical_render_lands_in_the_middle(self):
        """The curves are calibrated on 260 real outputs; the median must not pin.

        A normalisation that puts every real picture at one end has stopped
        distinguishing between them, which is the only thing it is for.
        """
        f = features(facts(
            sharpness={"score": 48.1, "band": "sharp", "sharpest_region": 121.3},
            noise={"sigma": 1.4, "band": "clean"},
            exposure={"mean": 70.2, "contrast": 47.4,
                      "clipped_black": 0.0, "clipped_white": 0.0}))
        for k in ("detail", "focus"):
            self.assertTrue(0.35 < f[k] < 0.7, f"{k}={f[k]}")
        self.assertTrue(0.55 < f["cleanliness"] < 0.85, f["cleanliness"])

    def test_a_still_has_no_motion_features_and_a_clip_no_exposure_ones(self):
        """Absent, not zero — see the module docstring."""
        self.assertNotIn("motion", features(facts()))
        self.assertNotIn("no_black", features(facts()))
        self.assertNotIn("headroom", features(clip()))
        self.assertNotIn("contrast", features(clip()))

    def test_a_frozen_black_clip_is_marked_down_on_both(self):
        f = features(clip(black_frames=3, frozen_pairs=8))
        self.assertEqual(f["motion"], 0.0)
        self.assertLess(f["no_black"], 0.7)

    def test_a_clean_clip_scores_full_marks_for_motion(self):
        f = features(clip())
        self.assertEqual(f["motion"], 1.0)
        self.assertEqual(f["no_black"], 1.0)

    def test_likeness_arrives_from_whichever_scorer_ran(self):
        f = features(facts(face_match={"available": True, "score": 0.91}))
        self.assertEqual(f["likeness"], 0.91)
        g = features(facts(subject_match={"available": True, "score": 0.4}))
        self.assertEqual(g["likeness"], 0.4)

    def test_a_likeness_that_could_not_be_measured_is_not_a_feature(self):
        f = features(facts(face_match={"available": False, "why": "no face"}))
        self.assertNotIn("likeness", f)

    def test_availability_is_the_authority_not_the_presence_of_a_number(self):
        """A stale score beside `available: False` must not become a feature.

        These dicts are read back out of the preference log as well as straight
        from the scorer, so the flag is the thing to trust.
        """
        f = features(facts(face_match={"available": False, "score": 0.9,
                                       "why": "no face detected in the output"}))
        self.assertNotIn("likeness", f)

    def test_junk_is_survivable(self):
        for junk in (None, "", [], 7, {}):
            self.assertEqual(features(junk), {}, repr(junk))

    def test_every_feature_it_can_produce_is_declared(self):
        """FEATURES is the fitter's vector order; a stray key would be dropped."""
        produced = set(features(facts(face_match={"available": True, "score": 0.5})))
        produced |= set(features(clip()))
        self.assertEqual(produced - set(FEATURES), set())


class ScoreTest(unittest.TestCase):

    def test_a_better_render_scores_higher(self):
        good = score(facts())["score"]
        bad = score(facts(sharpness={"score": 6.0, "band": "very soft",
                                     "sharpest_region": 15.0},
                          noise={"sigma": 8.0, "band": "grainy"}))["score"]
        self.assertGreater(good, bad)

    def test_an_unmeasurable_file_has_no_score_rather_than_a_zero(self):
        """Ranking it last would be a claim we cannot make."""
        self.assertEqual(score({}), {})

    def test_the_weights_are_renormalised_over_what_is_present(self):
        """Or a clip would rank below every still, for having no exposure."""
        s = score(clip())
        self.assertGreater(s["score"], 0.8)

    def test_a_zeroed_weight_cannot_move_the_score(self):
        a = score(facts(exposure={"mean": 40.0, "contrast": 20.0,
                                  "clipped_black": 0.0, "clipped_white": 0.0}))
        b = score(facts(exposure={"mean": 140.0, "contrast": 70.0,
                                  "clipped_black": 0.0, "clipped_white": 0.0}))
        self.assertEqual(a["score"], b["score"])

    def test_a_negative_learned_weight_still_orders_things(self):
        """A fit may learn "darker is better"; clamping at zero would flatten it."""
        w = dict(DEFAULT_WEIGHTS, brightness=-1.0, detail=0.0, focus=0.0,
                 cleanliness=0.0, headroom=0.0)
        dark = score(facts(exposure={"mean": 35.0, "contrast": 50.0,
                                     "clipped_black": 0.0, "clipped_white": 0.0}), w)
        bright = score(facts(exposure={"mean": 145.0, "contrast": 50.0,
                                       "clipped_black": 0.0, "clipped_white": 0.0}), w)
        self.assertGreater(dark["score"], bright["score"])
        self.assertGreater(dark["score"], 0.0)
        for s in (dark, bright):
            self.assertTrue(0.0 <= s["score"] <= 1.0)

    def test_all_weights_zero_yields_no_score_rather_than_a_divide_by_zero(self):
        self.assertEqual(score(facts(), {k: 0.0 for k in FEATURES}), {})

    def test_it_says_it_is_a_ranking_aid_wherever_it_is_shown(self):
        """The judge is handed thresholds either side of this one number."""
        text = render_score(score(facts()))
        self.assertIn("RANKING", text)
        self.assertIn("never a pass or a fail", text)
        self.assertEqual(render_score({}), "")


class WeightsFileTest(unittest.TestCase):

    def _with_file(self, payload):
        d = Path(tempfile.mkdtemp())
        f = d / "fitness_weights.json"
        f.write_text(json.dumps(payload), encoding="utf-8")
        return mock.patch.object(fitness, "WEIGHTS_FILE", f)

    def test_no_file_means_the_hand_set_weights(self):
        with mock.patch.object(fitness, "WEIGHTS_FILE", Path("/nope/nothing.json")):
            self.assertEqual(fitness.load_weights(), DEFAULT_WEIGHTS)

    def test_a_fitted_file_replaces_them(self):
        with self._with_file({"weights": {"detail": 0.9}}):
            self.assertEqual(fitness.load_weights()["detail"], 0.9)

    def test_a_broken_file_costs_accuracy_not_the_run(self):
        d = Path(tempfile.mkdtemp())
        f = d / "w.json"
        f.write_text("{not json", encoding="utf-8")
        with mock.patch.object(fitness, "WEIGHTS_FILE", f):
            self.assertEqual(fitness.load_weights(), DEFAULT_WEIGHTS)

    def test_a_file_naming_nothing_we_know_is_ignored(self):
        with self._with_file({"weights": {"vibes": 5.0}}):
            self.assertEqual(fitness.load_weights(), DEFAULT_WEIGHTS)

    def test_a_partial_file_keeps_the_defaults_for_the_rest(self):
        with self._with_file({"weights": {"detail": 0.9}}):
            got = fitness.load_weights()
        self.assertEqual(got["focus"], DEFAULT_WEIGHTS["focus"])


class RankFilesTest(unittest.TestCase):

    def _measured(self, table):
        def fake(path, is_video=False):
            return table.get(Path(path).name, {})
        return mock.patch("src.utils.image_facts.measure", fake)

    def test_the_best_file_comes_first(self):
        table = {"a.png": facts(sharpness={"score": 10.0, "sharpest_region": 20.0},
                                noise={"sigma": 7.0}),
                 "b.png": facts()}
        with self._measured(table):
            rows = rank_files(["a.png", "b.png"])
        self.assertEqual([r["name"] for r in rows], ["b.png", "a.png"])

    def test_a_file_that_cannot_be_measured_goes_last_unranked(self):
        """Unmeasurable is not worst — we simply do not know."""
        with self._measured({"good.png": facts()}):
            rows = rank_files(["broken.png", "good.png"])
        self.assertEqual([r["name"] for r in rows], ["good.png", "broken.png"])
        self.assertIsNone(rows[-1]["score"])

    def test_a_huge_batch_is_capped(self):
        with self._measured({}):
            rows = rank_files([f"{i}.png" for i in range(200)])
        self.assertEqual(len(rows), fitness.MAX_RANKED)

    def test_nothing_ranks_to_nothing(self):
        self.assertEqual(rank_files([]), [])
        self.assertEqual(fitness.render_ranking([]), "")

    def test_the_rendering_says_it_does_not_decide(self):
        with self._measured({"a.png": facts()}):
            text = fitness.render_ranking(rank_files(["a.png"]))
        self.assertIn("not a verdict", text)


# ── the fit ───────────────────────────────────────────────────────────────────

KEYS = ["detail", "focus", "cleanliness", "headroom", "contrast", "brightness"]
# A user who likes grain and dark pictures and is indifferent to focus — three
# things the hand-set weights get wrong, two of them by the sign.
TRUTH = {"detail": 0.3, "focus": 0.0, "cleanliness": -0.6, "headroom": 0.2,
         "contrast": 0.4, "brightness": -0.7}


def slates_from(truth, n, seed=7, keep=1):
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        size = rng.choice([4, 5, 6, 8])
        pool = [{k: round(rng.random(), 4) for k in KEYS} for _ in range(size)]
        pool.sort(key=lambda f: sum(truth[k] * f[k] for k in KEYS), reverse=True)
        out.append((pool[:keep], pool[keep:]))
    return out


class FitTest(unittest.TestCase):

    def test_it_recovers_a_preference_the_defaults_get_backwards(self):
        rows = slates_from(TRUTH, 120)
        train, test = split(rows)
        learned = fit(train, keys=active_keys(rows))
        base, new = evaluate(test, DEFAULT_WEIGHTS), evaluate(test, learned)
        self.assertGreater(new["pair_accuracy"], 0.9)
        self.assertGreater(new["pair_accuracy"], base["pair_accuracy"] + 0.3)
        # And the signs the defaults had wrong are now right.
        self.assertLess(learned["cleanliness"], 0)
        self.assertLess(learned["brightness"], 0)

    def test_a_single_kept_output_is_the_case_it_is_built_for(self):
        rows = slates_from(TRUTH, 60, keep=1)
        self.assertTrue(all(len(c) == 1 for c, _ in rows))
        self.assertGreater(top1_accuracy(rows, fit(rows, keys=KEYS)), 0.85)

    def test_two_kept_outputs_still_fit(self):
        rows = slates_from(TRUTH, 60, keep=2)
        self.assertGreater(pair_accuracy(pairs_of(rows), fit(rows, keys=KEYS)), 0.85)

    def test_a_two_item_slate_is_the_bradley_terry_case(self):
        """One kept, one rejected: the pairwise likelihood, unchanged."""
        rows = [([{"detail": 0.9}], [{"detail": 0.1}])] * 20
        learned = fit(rows, keys=["detail"], prior={"detail": 0.0}, l2=0.01)
        self.assertGreater(learned["detail"], 0.5)

    def test_the_prior_wins_when_there_is_almost_no_evidence(self):
        """Ten labels should nudge, not decide."""
        rows = slates_from(TRUTH, 3)
        learned = fit(rows, keys=KEYS)
        for k in KEYS:
            self.assertLess(abs(learned[k] - DEFAULT_WEIGHTS[k]), 0.35, k)

    def test_no_labels_returns_the_prior_untouched(self):
        self.assertEqual(fit([], keys=KEYS), DEFAULT_WEIGHTS)

    def test_a_feature_that_never_varies_is_left_alone(self):
        """It cannot be learned from labels that never move it."""
        rows = [([{"detail": 0.9, "focus": 0.5}], [{"detail": 0.1, "focus": 0.5}])] * 5
        self.assertEqual(active_keys(rows), ["detail"])

    def test_features_absent_from_the_labels_keep_their_defaults(self):
        learned = fit(slates_from(TRUTH, 40), keys=KEYS)
        self.assertEqual(learned["motion"], DEFAULT_WEIGHTS["motion"])
        self.assertEqual(learned["likeness"], DEFAULT_WEIGHTS["likeness"])

    def test_the_loss_goes_down_as_the_weights_get_better(self):
        from src.utils.fitness_fit import loss_and_grad
        rows = slates_from(TRUTH, 30)
        prior = [0.0] * len(KEYS)
        right = [TRUTH[k] * 8 for k in KEYS]
        wrong = [-TRUTH[k] * 8 for k in KEYS]
        good, _ = loss_and_grad(rows, right, KEYS, prior, l2=0.0)
        bad, _ = loss_and_grad(rows, wrong, KEYS, prior, l2=0.0)
        self.assertLess(good, bad)

    def test_the_gradient_agrees_with_the_loss_it_claims_to_be_of(self):
        """A sign slip here would train confidently in the wrong direction."""
        from src.utils.fitness_fit import loss_and_grad
        rows = slates_from(TRUTH, 8)
        w = [0.3, -0.2, 0.5, 0.1, -0.4, 0.2]
        base, grad = loss_and_grad(rows, w, KEYS, [0.0] * len(KEYS), l2=0.5)
        eps = 1e-6
        for j in range(len(KEYS)):
            bumped = list(w)
            bumped[j] += eps
            up, _ = loss_and_grad(rows, bumped, KEYS, [0.0] * len(KEYS), l2=0.5)
            self.assertAlmostEqual((up - base) / eps, grad[j], places=3, msg=KEYS[j])


class AccuracyTest(unittest.TestCase):

    def test_pair_accuracy_is_what_it_says(self):
        pairs = [({"a": 1.0}, {"a": 0.0}), ({"a": 0.0}, {"a": 1.0})]
        self.assertEqual(pair_accuracy(pairs, {"a": 1.0}), 0.5)
        self.assertEqual(pair_accuracy([pairs[0]], {"a": 1.0}), 1.0)

    def test_a_tie_is_half_not_a_win(self):
        """Otherwise weights that ignore everything would score a perfect 1.0."""
        pairs = [({"a": 1.0}, {"a": 0.0})]
        self.assertEqual(pair_accuracy(pairs, {"a": 0.0}), 0.5)

    def test_top1_asks_whether_the_best_scoring_one_was_kept(self):
        rows = [([{"a": 0.9}], [{"a": 0.1}, {"a": 0.2}])]
        self.assertEqual(top1_accuracy(rows, {"a": 1.0}), 1.0)
        self.assertEqual(top1_accuracy(rows, {"a": -1.0}), 0.0)

    def test_a_top1_tie_is_not_a_hit(self):
        """The model put a rejected output level with a kept one — it did not win."""
        rows = [([{"a": 0.5}], [{"a": 0.5}, {"a": 0.1}])]
        self.assertEqual(top1_accuracy(rows, {"a": 1.0}), 0.0)

    def test_a_holdout_that_rounds_to_nothing_still_holds_something_back(self):
        """Otherwise the fit is compared against data it was trained on."""
        rows = slates_from(TRUTH, 3)
        train, test = split(rows, holdout=0.15)
        self.assertEqual(len(test), 1)
        self.assertEqual(len(train), 2)

    def test_nothing_to_measure_is_zero_not_a_crash(self):
        self.assertEqual(pair_accuracy([], DEFAULT_WEIGHTS), 0.0)
        self.assertEqual(top1_accuracy([], DEFAULT_WEIGHTS), 0.0)

    def test_the_split_is_deterministic_and_keeps_everything(self):
        rows = slates_from(TRUTH, 20)
        a1, b1 = split(rows)
        a2, b2 = split(rows)
        self.assertEqual(len(a1) + len(b1), 20)
        self.assertEqual([id(r) for r in a1], [id(r) for r in a2])
        self.assertEqual([id(r) for r in b1], [id(r) for r in b2])

    def test_too_few_to_split_holds_nothing_back(self):
        rows = slates_from(TRUTH, 1)
        train, test = split(rows)
        self.assertEqual(len(train), 1)
        self.assertEqual(test, [])


if __name__ == "__main__":
    unittest.main()
