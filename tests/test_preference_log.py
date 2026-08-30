"""The labels that make the ranking weights learnable, taken from real decisions.

A `review` hook already stops a chain, fills a collector with everything the
stage produced, and waits for the user to delete the ones they do not want. What
survives is what they chose. That is training data for
:mod:`src.utils.fitness`, produced as a side effect of a decision they were
making anyway — nobody has to be asked to label anything.

What is being held down here is mostly about restraint. Recording must never
change what runs, never fail a turn, and never invent a preference where the user
did not express one: keeping everything is not a choice between anything, and a
review answered with "stop" rejected the lot.

And the vectors, not the paths, are what is stored. Outputs get deleted and
folders get moved; a label whose evidence is gone teaches nothing.

    python -m unittest discover -s tests
"""

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from src.utils import preference_log
from src.utils.preference_log import (pairs, read_events, record_review, slates,
                                      summary)


def facts(sharp=300.0, sigma=1.0, mean=120.0):
    return {"width": 1920, "height": 1080,
            "sharpness": {"score": sharp, "band": "sharp",
                          "sharpest_region": sharp * 3},
            "noise": {"sigma": sigma, "band": "clean"},
            "exposure": {"mean": mean, "contrast": 50.0,
                         "clipped_black": 0.0, "clipped_white": 0.0}}


class LogFile(unittest.TestCase):
    """Every test writes to its own file; none touches the real log."""

    def setUp(self):
        self.dir = Path(tempfile.mkdtemp())
        self.log = self.dir / "preferences.jsonl"

    def _record(self, kept, dropped, table=None, **kw):
        table = table or {}
        return record_review(kept, dropped, path=self.log,
                             facts_by_path={k: v for k, v in table.items()}, **kw)


class RecordingTest(LogFile):

    def test_a_review_is_written_with_both_sides_and_their_numbers(self):
        n = self._record(["keep.png"], ["drop.png"],
                         {"keep.png": facts(300.0), "drop.png": facts(9.0)},
                         hook_node_id="12", question="which one?")
        self.assertEqual(n, 1)
        ev = read_events(self.log)[0]
        self.assertEqual(ev["hook_node_id"], "12")
        self.assertEqual(ev["question"], "which one?")
        self.assertEqual([c["name"] for c in ev["chosen"]], ["keep.png"])
        self.assertEqual([r["name"] for r in ev["rejected"]], ["drop.png"])
        self.assertIn("detail", ev["chosen"][0]["features"])

    def test_the_features_are_stored_not_just_the_paths(self):
        """The pictures will be deleted; the label has to survive that."""
        self._record(["a.png"], ["b.png"], {"a.png": facts(300.0), "b.png": facts(9.0)})
        ev = read_events(self.log)[0]
        self.assertGreater(ev["chosen"][0]["features"]["detail"],
                           ev["rejected"][0]["features"]["detail"])

    def test_keeping_everything_records_nothing(self):
        """Not a preference between anything."""
        self.assertEqual(self._record(["a.png", "b.png"], [], {"a.png": facts()}), 0)
        self.assertEqual(read_events(self.log), [])

    def test_rejecting_everything_records_nothing(self):
        self.assertEqual(self._record([], ["a.png"], {"a.png": facts()}), 0)
        self.assertEqual(read_events(self.log), [])

    def test_a_file_with_no_measurable_features_is_left_out(self):
        n = self._record(["a.png", "unreadable.png"], ["b.png"],
                         {"a.png": facts(), "unreadable.png": {}, "b.png": facts(9.0)})
        self.assertEqual(n, 1)
        self.assertEqual([c["name"] for c in read_events(self.log)[0]["chosen"]],
                         ["a.png"])

    def test_a_review_where_nothing_could_be_measured_is_not_written(self):
        self.assertEqual(self._record(["a.png"], ["b.png"],
                                      {"a.png": {}, "b.png": {}}), 0)
        self.assertEqual(read_events(self.log), [])

    def test_reviews_accumulate_rather_than_overwrite(self):
        for i in range(3):
            self._record([f"k{i}.png"], [f"d{i}.png"],
                         {f"k{i}.png": facts(), f"d{i}.png": facts(9.0)})
        self.assertEqual(len(read_events(self.log)), 3)

    def test_an_enormous_review_is_capped(self):
        table = {f"f{i}.png": facts() for i in range(200)}
        self._record([f"f{i}.png" for i in range(100)],
                     [f"f{i}.png" for i in range(100, 200)], table)
        ev = read_events(self.log)[0]
        self.assertEqual(len(ev["chosen"]), preference_log.MAX_PER_SIDE)
        self.assertEqual(len(ev["rejected"]), preference_log.MAX_PER_SIDE)

    def test_one_empty_side_is_noticed_before_anything_is_measured(self):
        """Measuring costs real milliseconds; there is nothing here to learn from."""
        with mock.patch("src.utils.image_facts.measure") as m:
            self.assertEqual(record_review(["a.png", "b.png"], [], path=self.log), 0)
            self.assertEqual(record_review([], ["a.png"], path=self.log), 0)
        m.assert_not_called()

    def test_a_side_that_measured_to_nothing_is_not_written_as_empty(self):
        """One usable side and one unreadable one is not a preference either.

        Checked against the FILE, not against what comes back: a one-sided row is
        filtered out on the way in, so an event written with an empty side is
        invisible to every reader here and would sit in the log for good.
        """
        for kept, dropped in ((["a.png"], ["broken.png"]),
                              (["broken.png"], ["a.png"])):
            self.assertEqual(self._record(kept, dropped,
                                          {"a.png": facts(), "broken.png": {}}), 0)
            self.assertFalse(self.log.exists() and self.log.read_text("utf-8").strip(),
                             "a row with an empty side was appended to the log")

    def test_an_unwritable_path_is_survivable(self):
        """A label is never worth a turn."""
        blocker = self.dir / "not-a-directory"
        blocker.write_text("", encoding="utf-8")
        self.assertEqual(
            record_review(["a.png"], ["b.png"], path=blocker / "sub" / "y.jsonl",
                          facts_by_path={"a.png": facts(), "b.png": facts(9.0)}), 0)

    def test_it_measures_for_itself_when_the_caller_has_no_numbers(self):
        with mock.patch("src.utils.image_facts.measure",
                        return_value=facts()) as m:
            record_review(["a.png"], ["b.mp4"], path=self.log)
        self.assertEqual(m.call_count, 2)
        self.assertFalse(m.call_args_list[0].kwargs["is_video"])
        self.assertTrue(m.call_args_list[1].kwargs["is_video"])


class ReadingTest(LogFile):

    def _write(self, rows):
        self.log.write_text("\n".join(json.dumps(r) for r in rows) + "\n",
                            encoding="utf-8")

    def test_no_log_yet_is_the_normal_case_not_an_error(self):
        self.assertEqual(read_events(self.dir / "nothing.jsonl"), [])
        self.assertEqual(slates(path=self.dir / "nothing.jsonl"), [])
        self.assertIn("no preference labels", summary(self.dir / "nothing.jsonl"))

    def test_a_corrupt_line_is_skipped_not_fatal(self):
        self.log.write_text(
            json.dumps({"chosen": [{"features": {"detail": 0.9}}],
                        "rejected": [{"features": {"detail": 0.1}}]})
            + "\n{ broken\n\n", encoding="utf-8")
        self.assertEqual(len(read_events(self.log)), 1)

    def test_a_row_missing_a_side_is_not_an_event(self):
        self._write([{"chosen": [{"features": {"detail": 0.9}}], "rejected": []},
                     {"chosen": [{"features": {"detail": 0.9}}],
                      "rejected": [{"features": {"detail": 0.1}}]}])
        self.assertEqual(len(read_events(self.log)), 1)


class SlateTest(unittest.TestCase):
    """The shape the fit reads: a choice from a field, not a pile of duels."""

    def _ev(self, chosen, rejected):
        return {"chosen": [{"features": f} for f in chosen],
                "rejected": [{"features": f} for f in rejected]}

    def test_one_review_is_one_slate_however_many_were_rejected(self):
        rows = slates([self._ev([{"detail": 0.9}],
                                [{"detail": 0.1}, {"detail": 0.2}, {"detail": 0.3}])])
        self.assertEqual(len(rows), 1)
        chosen, rejected = rows[0]
        self.assertEqual(len(chosen), 1)
        self.assertEqual(len(rejected), 3)

    def test_the_same_review_implies_one_pair_per_rejected_output(self):
        """Pairs are for MEASURING a fit, not making one — so they still exist."""
        self.assertEqual(
            len(pairs([self._ev([{"detail": 0.9}],
                                [{"detail": 0.1}, {"detail": 0.2}])])), 2)

    def test_only_features_shared_by_the_whole_slate_survive(self):
        """They go into one denominator; a feature some members lack cannot."""
        rows = slates([self._ev([{"detail": 0.9, "motion": 1.0}],
                                [{"detail": 0.1, "motion": 0.5},
                                 {"detail": 0.2}])])          # a still among clips
        chosen, _rejected = rows[0]
        self.assertEqual(sorted(chosen[0]), ["detail"])

    def test_a_slate_with_nothing_in_common_is_dropped(self):
        self.assertEqual(slates([self._ev([{"motion": 1.0}], [{"detail": 0.1}])]), [])

    def test_a_two_item_slate_is_exactly_the_pairwise_case(self):
        rows = slates([self._ev([{"detail": 0.9}], [{"detail": 0.1}])])
        self.assertEqual(len(rows[0][0]), 1)
        self.assertEqual(len(rows[0][1]), 1)

    def test_the_summary_counts_decisions_and_duels_separately(self):
        d = Path(tempfile.mkdtemp()) / "p.jsonl"
        d.write_text(json.dumps(self._ev([{"detail": 0.9}],
                                         [{"detail": 0.1}, {"detail": 0.2}])) + "\n",
                     encoding="utf-8")
        text = summary(d)
        self.assertIn("1 review", text)
        self.assertIn("1 usable slate", text)
        self.assertIn("2 implied preference pairs", text)


class FromAReviewTest(unittest.TestCase):
    """The pipeline's side: what the halt produced, minus what the user kept."""

    def _pipeline(self, produced, kept, reply="continue"):
        from src.pipeline import Pipeline
        from src.utils.review_gate import ReviewHalt
        p = object.__new__(Pipeline)
        p._verbose = False
        p._review_reply = reply
        p._review_halt = ReviewHalt(hook_node_id="12", produced=tuple(produced),
                                    question="which?")
        p._review_collector_files = lambda: list(kept)
        return p

    def test_what_they_deleted_is_the_rejected_side(self):
        p = self._pipeline(["a.png", "b.png", "c.png"], ["b.png"])
        with mock.patch("src.utils.preference_log.record_review",
                        return_value=2) as rec:
            p._record_review_preference("make me three")
        self.assertEqual(rec.call_args[0][0], ["b.png"])
        self.assertEqual(sorted(rec.call_args[0][1]), ["a.png", "c.png"])
        self.assertEqual(rec.call_args[1]["hook_node_id"], "12")
        self.assertEqual(rec.call_args[1]["request"], "make me three")

    def test_keeping_everything_records_nothing(self):
        p = self._pipeline(["a.png", "b.png"], ["a.png", "b.png"])
        with mock.patch("src.utils.preference_log.record_review") as rec:
            p._record_review_preference()
        rec.assert_not_called()

    def test_stop_records_nothing(self):
        """They rejected the lot — there is no preference between two things in it."""
        p = self._pipeline(["a.png", "b.png"], [], reply="stop")
        with mock.patch("src.utils.preference_log.record_review") as rec:
            p._record_review_preference()
        rec.assert_not_called()

    def test_an_emptied_collector_records_nothing(self):
        p = self._pipeline(["a.png", "b.png"], [])
        with mock.patch("src.utils.preference_log.record_review") as rec:
            p._record_review_preference()
        rec.assert_not_called()

    def test_a_file_swapped_in_by_hand_does_not_make_the_others_rejected(self):
        """They added their own; the produced ones they kept are still kept."""
        p = self._pipeline(["a.png", "b.png"], ["a.png", "D:/mine/own.png"])
        with mock.patch("src.utils.preference_log.record_review",
                        return_value=1) as rec:
            p._record_review_preference()
        self.assertEqual(rec.call_args[0][1], ["b.png"])

    def test_the_same_file_by_another_path_still_counts_as_kept(self):
        p = self._pipeline(["D:/out/a.png", "D:/out/b.png"], ["W:/copy/a.png"])
        with mock.patch("src.utils.preference_log.record_review",
                        return_value=1) as rec:
            p._record_review_preference()
        self.assertEqual(rec.call_args[0][1], ["D:/out/b.png"])

    def test_no_halt_records_nothing(self):
        p = self._pipeline(["a.png"], ["a.png"])
        p._review_halt = None
        with mock.patch("src.utils.preference_log.record_review") as rec:
            p._record_review_preference()
        rec.assert_not_called()

    def test_a_logging_failure_never_reaches_the_turn(self):
        """A review is the user's decision; our logging has no business failing it."""
        p = self._pipeline(["a.png", "b.png"], ["a.png"])
        with mock.patch("src.utils.preference_log.record_review",
                        side_effect=RuntimeError("disk full")):
            p._record_review_preference()       # must not raise


if __name__ == "__main__":
    unittest.main()
