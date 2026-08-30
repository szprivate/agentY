"""Does the output look like the reference — as a number, not an impression.

"Same character" is the criterion people write most often and the one a vision
model is worst at: it will call any two dark-haired men the same person and any
two lighting setups a match. There is a real measurement for it — a face
embedding compared by cosine — and on this machine's own renders it separates
cleanly: the same character scores 0.95-0.98, different characters 0.09-0.54.

So the briefing gets a `likeness` control, the scorers run only when it is set,
and the verdict is settled by arithmetic like every other technical check.

The scorers themselves are stubbed here. Loading ArcFace costs ~30 s and
DreamSim ~100 s and 3 GB, and what needs testing is not whether a published
embedding works — it is everything around it: that the expensive path is not
entered when nobody asked, that a clip is compared frame by frame, that a
reference is examined once rather than once per frame, and that a comparison
which cannot be made does not condemn the output.

    python -m unittest discover -s tests
"""

import unittest
from unittest import mock

from src.utils import likeness
from src.utils.qa import QaBriefing, _likeness_facts, render_measurements
from src.utils.qa_checks import WANT_FACE, WANT_SUBJECT, evaluate


class Face:
    """A detected face, as far as this module is concerned: a bbox and a vector."""

    def __init__(self, vec, size=100.0):
        self.normed_embedding = vec
        self.bbox = (0.0, 0.0, size, size)


def vec(x):
    """A unit vector whose cosine against ``vec(1.0)`` is *x*."""
    import numpy as np
    y = (1.0 - x ** 2) ** 0.5
    return np.array([x, y], dtype="float32")


ALIKE = vec(1.0)


class CandidatesTest(unittest.TestCase):

    def test_one_path_is_one_candidate(self):
        self.assertEqual(likeness._candidates("a.png"), ["a.png"])

    def test_a_list_of_frames_is_kept(self):
        self.assertEqual(likeness._candidates(["a.png", "b.png"]), ["a.png", "b.png"])

    def test_nothing_is_no_candidates(self):
        for junk in (None, [], "", [None, ""]):
            self.assertEqual(likeness._candidates(junk), [], repr(junk))


class FaceMatchTest(unittest.TestCase):

    def _run(self, out_faces, ref_faces, output="out.png", refs=("ref.png",)):
        """Run face_match with detection stubbed to a lookup by path."""
        table = {}
        table.update(out_faces)
        table.update(ref_faces)
        seen = []

        def detect(_app, path):
            seen.append(path)
            return table.get(path)

        with mock.patch.object(likeness, "_face_analyser", return_value=object()), \
             mock.patch.object(likeness, "_largest_face", detect):
            return likeness.face_match(output, list(refs)), seen

    def test_the_same_person_scores_high_and_is_named(self):
        got, _ = self._run({"out.png": Face(ALIKE)}, {"ref.png": Face(vec(0.97))})
        self.assertTrue(got["available"])
        self.assertEqual(got["score"], 0.97)
        self.assertEqual(got["band"], "the same person")
        self.assertEqual(got["reference"], "ref.png")

    def test_a_different_person_is_named_as_one(self):
        got, _ = self._run({"out.png": Face(ALIKE)}, {"ref.png": Face(vec(0.12))})
        self.assertEqual(got["band"], "different person")

    def test_the_best_reference_wins_and_is_the_one_reported(self):
        """Several references mean several tries — the closest is the answer."""
        got, _ = self._run({"out.png": Face(ALIKE)},
                           {"a.png": Face(vec(0.2)), "b.png": Face(vec(0.96))},
                           refs=("a.png", "b.png"))
        self.assertEqual(got["score"], 0.96)
        self.assertEqual(got["reference"], "b.png")
        self.assertEqual(got["compared"], 2)

    def test_the_best_frame_of_a_clip_wins(self):
        """A character out of shot for two frames of three is still a match."""
        got, _ = self._run({"f1.png": Face(vec(0.1)), "f2.png": Face(vec(0.2)),
                            "f3.png": Face(vec(0.98))},
                           {"ref.png": Face(ALIKE)},
                           output=["f1.png", "f2.png", "f3.png"])
        self.assertEqual(got["score"], 0.98)

    def test_a_reference_is_examined_once_however_many_frames(self):
        """Detection is the expensive half; re-running it per frame is waste."""
        _got, seen = self._run({"f1.png": Face(vec(0.1)), "f2.png": Face(vec(0.2)),
                                "f3.png": Face(vec(0.9))},
                               {"ref.png": Face(ALIKE)},
                               output=["f1.png", "f2.png", "f3.png"])
        self.assertEqual(seen.count("ref.png"), 1)
        self.assertEqual(len([s for s in seen if s.startswith("f")]), 3)

    def test_no_face_in_the_output_is_reported_but_is_not_a_score(self):
        got, _ = self._run({"out.png": None}, {"ref.png": Face(ALIKE)})
        self.assertFalse(got["available"])
        self.assertIn("output", got["why"])

    def test_no_face_in_any_reference_is_reported_separately(self):
        """A landscape wired in as a reference is a different problem to name."""
        got, _ = self._run({"out.png": Face(ALIKE)}, {"ref.png": None})
        self.assertFalse(got["available"])
        self.assertIn("reference", got["why"])

    def test_a_reference_without_a_face_is_skipped_not_fatal(self):
        got, _ = self._run({"out.png": Face(ALIKE)},
                           {"a.png": None, "b.png": Face(vec(0.9))},
                           refs=("a.png", "b.png"))
        self.assertEqual(got["reference"], "b.png")
        self.assertEqual(got["compared"], 1)

    def test_no_references_asks_nothing_at_all(self):
        """Not a failure — a briefing with no reference did not ask the question."""
        with mock.patch.object(likeness, "_face_analyser",
                               side_effect=AssertionError("must not load")):
            self.assertEqual(likeness.face_match("out.png", []), {})

    def test_the_scorer_being_absent_yields_nothing_rather_than_failing(self):
        with mock.patch.object(likeness, "_face_analyser", return_value=None):
            self.assertEqual(likeness.face_match("out.png", ["ref.png"]), {})


class FaceBandTest(unittest.TestCase):
    """The bands are what the pass/fail bar is written against, so they are fixed."""

    def test_each_band_starts_where_it_says(self):
        for score, expected in ((0.0, "different person"),
                                (0.299, "different person"),
                                (0.30, "possibly the same"),
                                (0.499, "possibly the same"),
                                (0.50, "likely the same person"),
                                (0.699, "likely the same person"),
                                (0.70, "the same person"),
                                (1.0, "the same person")):
            self.assertEqual(likeness._band(score, likeness.FACE_BANDS,
                                            "the same person"), expected, score)


class SubjectMatchTest(unittest.TestCase):

    def test_a_matching_subject_reads_as_a_similarity_not_a_distance(self):
        """DreamSim counts down from identical; the score next to it counts up."""
        got, _ = _subject(distance=0.02)
        self.assertEqual(got["distance"], 0.02)
        self.assertEqual(got["score"], 0.98)
        self.assertEqual(got["band"], "the same subject")

    def test_an_unrelated_subject_is_named_as_one(self):
        got, _ = _subject(distance=0.8)
        self.assertEqual(got["band"], "a different subject")

    def test_a_distance_beyond_one_does_not_produce_a_negative_score(self):
        got, _ = _subject(distance=1.4)
        self.assertEqual(got["score"], 0.0)

    def test_the_closest_frame_of_a_clip_is_the_answer(self):
        got, calls = _subject(distance={"f1.png": 0.7, "f2.png": 0.05},
                              output=["f1.png", "f2.png"])
        self.assertEqual(got["distance"], 0.05)
        self.assertEqual(len(calls), 2)

    def test_the_closest_reference_is_the_answer(self):
        """Several references mean several tries — the nearest one is the score."""
        got, _ = _subject(distance=lambda _o, r: {"a.png": 0.62, "b.png": 0.04}[r],
                          refs=("a.png", "b.png"))
        self.assertEqual(got["distance"], 0.04)
        self.assertEqual(got["reference"], "b.png")
        self.assertEqual(got["compared"], 2)

    def test_no_references_never_loads_the_model(self):
        with mock.patch.object(likeness, "_dreamsim_model",
                               side_effect=AssertionError("must not load")):
            self.assertEqual(likeness.subject_match("out.png", []), {})

    def test_the_model_being_absent_yields_nothing(self):
        with mock.patch.object(likeness, "_dreamsim_model", return_value=None):
            self.assertEqual(likeness.subject_match("out.png", ["ref.png"]), {})

    def test_a_reference_that_cannot_be_read_is_skipped(self):
        got, _ = _subject(distance=0.1, bad_refs=["broken.png"])
        self.assertEqual(got["reference"], "ref.png")


def _subject(distance, output="out.png", refs=("ref.png",), bad_refs=()):
    """subject_match with DreamSim stubbed.

    *distance* is a flat float, a dict keyed by candidate (to vary by frame), or a
    callable taking the candidate and the reference.
    """
    calls = []

    def model(a, b):
        calls.append((a, b))
        if callable(distance):
            return distance(a, b)
        if isinstance(distance, dict):
            return distance[a]
        return distance

    def preprocess(im):
        if im in bad_refs:
            raise OSError("unreadable")
        return im

    with mock.patch.object(likeness, "_dreamsim_model", return_value=(model, preprocess)), \
         mock.patch("PIL.Image.open", _fake_open):
        got = likeness.subject_match(output, [*refs, *bad_refs])
    return got, calls


class _FakeImage:
    def __init__(self, name):
        self.name = name

    def convert(self, _mode):
        return self.name

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        return False


def _fake_open(path):
    return _FakeImage(str(path))


class RenderTest(unittest.TestCase):
    """The facts block is what the model reads; a bare number means nothing."""

    def test_a_face_score_says_what_it_is_and_which_way_round(self):
        line = likeness.render_likeness(
            {"face_match": {"available": True, "score": 0.97, "band": "the same person",
                            "reference": "hero.png"}})[0]
        self.assertIn("0.97", line)
        self.assertIn("hero.png", line)
        self.assertIn("the same person", line)
        self.assertIn("1.0 is identical", line)

    def test_a_comparison_that_could_not_be_made_says_why(self):
        line = likeness.render_likeness(
            {"face_match": {"available": False, "why": "no face detected in the output"}})[0]
        self.assertIn("not measurable", line)
        self.assertIn("no face", line)

    def test_a_subject_score_is_labelled_as_perceptual(self):
        line = likeness.render_likeness(
            {"subject_match": {"available": True, "score": 0.9,
                               "band": "the same subject", "reference": "set.png"}})[0]
        self.assertIn("subject match", line)
        self.assertIn("perceptual", line)

    def test_nothing_measured_renders_nothing(self):
        for facts in ({}, None, {"face_match": {}}, {"width": 100}):
            self.assertEqual(likeness.render_likeness(facts), [], repr(facts))

    def test_the_lines_reach_the_measured_block_the_judge_is_given(self):
        text = render_measurements({"width": 1920, "height": 1080,
                                    "face_match": {"available": True, "score": 0.97,
                                                   "band": "the same person",
                                                   "reference": "hero.png"}})
        self.assertIn("face match: 0.97", text)


class VerdictTest(unittest.TestCase):
    """Where the score turns into a pass or a fail."""

    def _facts(self, key, **over):
        m = {"available": True, "score": 0.9, "band": "the same person",
             "reference": "hero.png"}
        m.update(over)
        return {key: m}

    def test_a_match_passes(self):
        rows = evaluate({"likeness": WANT_FACE}, self._facts("face_match"))
        self.assertEqual(rows[0]["result"], "pass")
        self.assertEqual(rows[0]["criterion"], WANT_FACE)

    def test_a_stranger_fails_and_the_note_carries_the_number(self):
        rows = evaluate({"likeness": WANT_FACE},
                        self._facts("face_match", score=0.21,
                                    band="different person"))
        self.assertEqual(rows[0]["result"], "fail")
        self.assertIn("0.21", rows[0]["note"])
        self.assertIn("hero.png", rows[0]["note"])

    def test_the_bar_sits_above_the_uncertain_band(self):
        """0.54 was a DIFFERENT character on this machine's own renders.

        Anything the scorer calls only 'possibly the same' must not pass, or the
        check waves through exactly the failure it exists to catch.
        """
        rows = evaluate({"likeness": WANT_FACE},
                        self._facts("face_match", score=0.54,
                                    band="possibly the same"))
        self.assertEqual(rows[0]["result"], "fail")
        rows = evaluate({"likeness": WANT_FACE},
                        self._facts("face_match", score=0.71,
                                    band="likely the same person"))
        self.assertEqual(rows[0]["result"], "pass")

    def test_a_subject_check_reads_its_own_score(self):
        rows = evaluate({"likeness": WANT_SUBJECT},
                        self._facts("subject_match", band="clearly related"))
        self.assertEqual(rows[0]["result"], "pass")
        self.assertIn("subject similarity", rows[0]["note"])

    def test_a_loosely_similar_subject_is_not_a_match(self):
        rows = evaluate({"likeness": WANT_SUBJECT},
                        self._facts("subject_match", band="loosely similar"))
        self.assertEqual(rows[0]["result"], "fail")

    def test_the_face_score_does_not_answer_a_subject_question(self):
        """Each option reads its own scorer, or a wired-up face would pass both."""
        self.assertEqual(evaluate({"likeness": WANT_SUBJECT},
                                  self._facts("face_match")), [])

    def test_a_comparison_that_could_not_be_made_yields_no_verdict(self):
        """Doubt does not condemn. The written criterion still reaches the model."""
        rows = evaluate({"likeness": WANT_FACE},
                        {"face_match": {"available": False, "why": "no face"}})
        self.assertEqual(rows, [])

    def test_nothing_asked_checks_nothing(self):
        self.assertEqual(evaluate({"likeness": "any"}, self._facts("face_match")), [])


class OnlyWhenAskedTest(unittest.TestCase):
    """The expensive half must not run for a briefing that never mentioned it."""

    def _briefing(self, **tech):
        return QaBriefing(criteria="looks good", technical=dict(tech))

    def _never(self):
        """Watch both scorers without raising.

        A stub that raises proves nothing here: `_likeness_facts` catches every
        exception on purpose, so an assertion thrown inside it is swallowed and
        the test passes whether or not the scorer ran. The call count is the only
        honest witness.
        """
        return mock.patch.object(likeness, "face_match", return_value={"available": True})

    def test_no_likeness_control_means_no_scorer(self):
        with self._never() as fm:
            self.assertEqual(_likeness_facts("out.png", self._briefing(), ["ref.png"]), {})
        fm.assert_not_called()

    def test_no_reference_means_no_scorer(self):
        """There is nothing to compare against; loading 3 GB to learn that is waste."""
        with self._never() as fm:
            self.assertEqual(
                _likeness_facts("out.png", self._briefing(likeness=WANT_FACE), []), {})
        fm.assert_not_called()

    def test_an_image_is_compared_as_itself(self):
        with mock.patch.object(likeness, "face_match",
                               return_value={"available": True}) as fm:
            got = _likeness_facts("out.png", self._briefing(likeness=WANT_FACE),
                                  ["ref.png"])
        self.assertEqual(got, {"face_match": {"available": True}})
        self.assertEqual(fm.call_args[0][0], ["out.png"])
        self.assertEqual(fm.call_args[0][1], ["ref.png"])

    def test_a_clip_is_compared_as_its_frames(self):
        with mock.patch("src.utils.qa._likeness_frames",
                        return_value=["f1.png", "f2.png"]), \
             mock.patch.object(likeness, "face_match",
                               return_value={"available": True}) as fm:
            _likeness_facts("out.mp4", self._briefing(likeness=WANT_FACE), ["ref.png"])
        self.assertEqual(fm.call_args[0][0], ["f1.png", "f2.png"])

    def test_a_clip_that_cannot_be_sampled_asks_nothing(self):
        with mock.patch("src.utils.qa._likeness_frames", return_value=[]), \
             self._never() as fm:
            self.assertEqual(
                _likeness_facts("out.mp4", self._briefing(likeness=WANT_FACE),
                                ["ref.png"]), {})
        fm.assert_not_called()

    def test_the_subject_option_reaches_the_subject_scorer(self):
        with mock.patch.object(likeness, "subject_match",
                               return_value={"available": True}) as sm, \
             self._never() as fm:
            got = _likeness_facts("out.png", self._briefing(likeness=WANT_SUBJECT),
                                  ["ref.png"])
        self.assertEqual(got, {"subject_match": {"available": True}})
        self.assertEqual(sm.call_count, 1)
        fm.assert_not_called()

    def test_a_scorer_with_nothing_to_say_adds_no_fact(self):
        with mock.patch.object(likeness, "face_match", return_value={}):
            self.assertEqual(
                _likeness_facts("out.png", self._briefing(likeness=WANT_FACE),
                                ["ref.png"]), {})

    def test_a_scorer_that_blows_up_does_not_take_qa_with_it(self):
        with mock.patch.object(likeness, "face_match",
                               side_effect=RuntimeError("onnx died")):
            self.assertEqual(
                _likeness_facts("out.png", self._briefing(likeness=WANT_FACE),
                                ["ref.png"]), {})

    def test_an_unknown_option_from_a_newer_node_is_ignored(self):
        with self._never() as fm:
            self.assertEqual(
                _likeness_facts("out.png", self._briefing(likeness="must match the vibe"),
                                ["ref.png"]), {})
        fm.assert_not_called()


class TheWholeWayThroughTest(unittest.TestCase):
    """From the node's dropdown to the verdict, with the judge stubbed.

    Every piece below is covered on its own; what this holds down is that they are
    still joined. The score is worth nothing if `check_output` forgets to measure
    it, or measures it and never tells the model.
    """

    def _check(self, score, band, reply='{"verdict": "pass", "summary": "fine"}'):
        from src.utils import qa
        agent = mock.Mock()
        agent.messages = []
        agent.return_value = reply
        briefing = QaBriefing(criteria="the hero must be the hero",
                              reference_paths=("hero.png",),
                              technical={"likeness": WANT_FACE})
        with mock.patch.object(qa, "_output_blocks",
                               return_value=([{"image": {}}], "an image")), \
             mock.patch.object(qa, "_image_block", return_value={"image": {}}), \
             mock.patch.object(qa, "is_image", return_value=True), \
             mock.patch.object(qa, "measure_output",
                               return_value={"width": 1920, "height": 1080}), \
             mock.patch.object(qa, "qa_settings",
                               return_value={"video_frames": 1, "max_references": 2}), \
             mock.patch.object(likeness, "face_match",
                               return_value={"available": True, "score": score,
                                             "band": band, "reference": "hero.png",
                                             "compared": 1}):
            result = qa.check_output("out.png", briefing, request="a portrait",
                                     agent=agent)
        asked = agent.call_args[0][0][-1]["text"]
        return result, asked

    def test_the_score_reaches_the_model_and_the_verdict(self):
        result, asked = self._check(0.97, "the same person")
        self.assertIn("face match: 0.97", asked)
        self.assertIn("ALREADY DECIDED", asked)
        self.assertTrue(result.passed)
        row = [c for c in result.checks if c["criterion"] == WANT_FACE][0]
        self.assertEqual(row["result"], "pass")

    def test_a_stranger_fails_the_output_even_when_the_model_liked_it(self):
        """A number the model cannot see must not be talkable past."""
        result, _asked = self._check(0.18, "different person")
        self.assertFalse(result.passed)
        row = [c for c in result.checks if c["criterion"] == WANT_FACE][0]
        self.assertEqual(row["result"], "fail")
        self.assertIn("0.18", row["note"])


class WrittenCriteriaTest(unittest.TestCase):
    """The demand also has to reach the model, for the times it cannot be measured."""

    def test_the_likeness_demand_is_written_into_the_criteria(self):
        from src.utils.qa_checks import describe
        self.assertIn(WANT_FACE, describe({"likeness": WANT_FACE}))
        self.assertEqual(describe({"likeness": "any"}), "")

    def test_the_control_reaches_the_briefing_from_the_node(self):
        from src.utils.qa import briefing_from_hooks
        b = briefing_from_hooks([{"hook_node_id": "3", "purpose": "qa", "directive": "",
                                  "technical": {"likeness": WANT_FACE}, "anchors": []}])
        self.assertEqual(b.technical["likeness"], WANT_FACE)
        self.assertIn(WANT_FACE, b.criteria)


if __name__ == "__main__":
    unittest.main()
