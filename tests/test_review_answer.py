"""After a review is answered there is one list of what goes on, and one reading of who said so.

A lead conversation spent a quarter of an hour reasoning about which pictures
were approved. It had been given three answers: the files the user had just
approved, a collector holding an earlier run's pictures, and a second collector
reported as emptied by the user. Neither collector had been touched by anyone -
the stops were raised in turns no page was showing, so they never reached the
canvas - and the approval itself had been sent into a running turn and dropped.
"""
import types
import unittest
from pathlib import Path
from unittest import mock

from src.pipeline import Pipeline
from src.utils import interject_hook
from src.utils import review_gate as rg

V3 = ["//srv/out/v003/a_v003_MARA_00001_.png", "//srv/out/v003/a_v003_JONAS_00001_.png"]
OLD = ["//srv/out/v080/a_v080_bEpic_00001_.png", "//srv/out/v001/a_v001_MAYA_00001_.png"]


def me(collector_files, origin="panel", opening="", thread="t1"):
    obj = types.SimpleNamespace(_review_user_text=opening)
    obj._review_collector_files = lambda halt=None: list(collector_files)
    obj._turn_origin = lambda: origin
    obj._thread_id_now = lambda: thread
    return obj


class WhatGoesOn(unittest.TestCase):
    HALT = rg.ReviewHalt(hook_node_id="135", collector_key="k", produced=tuple(V3))

    def goes_on(self, collector_files):
        return Pipeline._review_goes_on(me(collector_files), self.HALT)

    def test_the_collector_is_the_answer_when_it_holds_this_reviews_files(self):
        files, where = self.goes_on(V3)
        self.assertEqual(files, V3)
        self.assertIn("the collector as it stands now", where)

    def test_rows_the_user_removed_stay_removed(self):
        files, where = self.goes_on(V3[:1])
        self.assertEqual(files, V3[:1])
        self.assertIn("1 of this round's output(s) removed by the user", where)

    def test_a_file_the_user_added_beside_them_is_kept(self):
        files, _ = self.goes_on(V3 + ["//srv/mine/own.png"])
        self.assertEqual(files, V3 + ["//srv/mine/own.png"])

    def test_a_collector_left_from_an_earlier_run_is_not_the_answer(self):
        files, where = self.goes_on(OLD)
        self.assertEqual(files, V3)
        self.assertIn("from an earlier run", where)
        self.assertIn("NOT the answer", where)

    def test_no_collector_on_the_canvas_means_this_rounds_outputs(self):
        files, where = self.goes_on([])
        self.assertEqual(files, V3)
        self.assertIn("No collector for this review is on the", where)

    def test_a_written_review_has_no_files(self):
        text = rg.ReviewHalt(hook_node_id="131", text_hooks=("4",))
        self.assertEqual(Pipeline._review_goes_on(me(OLD), text), ([], ""))

    def test_the_agent_is_told_it_is_settled(self):
        src = (Path(__file__).resolve().parent.parent / "src" / "pipeline.py").read_text(encoding="utf-8")
        self.assertIn("this is settled: do not weigh it against", src)
        self.assertNotIn("The collector is EMPTY or gone — say so and ask what", src)


class WhoseWordsCount(unittest.TestCase):
    def setUp(self):
        interject_hook._spoken.clear()
        self.addCleanup(interject_hook._spoken.clear)

    def test_the_message_that_opened_the_turn_counts(self):
        self.assertEqual(Pipeline._users_words(me([], opening="go ahead")), "go ahead")

    def test_a_branch_report_that_opened_the_turn_does_not(self):
        obj = me([], origin="shots", opening="[SHOTS REPORTING BACK] ... approved ...")
        self.assertEqual(Pipeline._users_words(obj), "")

    def test_what_the_user_sends_into_that_turn_does(self):
        obj = me([], origin="shots", opening="[SHOTS REPORTING BACK] ...")
        with mock.patch.object(interject_hook.interject_bus, "thread_id", return_value="t1"), \
                mock.patch("src.utils.conversation_store.add_message"):
            interject_hook._persist([{"text": "Nice - latest character images are approved."}])
        heard = Pipeline._users_words(obj)
        self.assertEqual(heard, "Nice - latest character images are approved.")
        self.assertEqual(rg.quote_check("latest character images are approved", heard), "")

    def test_another_conversations_words_are_not_heard(self):
        with mock.patch.object(interject_hook.interject_bus, "thread_id", return_value="other"), \
                mock.patch("src.utils.conversation_store.add_message"):
            interject_hook._persist([{"text": "approved"}])
        self.assertEqual(Pipeline._users_words(me([], origin="shots")), "")

    def test_they_are_forgotten_when_the_next_turn_begins(self):
        interject_hook._spoken["t1"] = ["approved"]
        interject_hook.clear_spoken("t1")
        self.assertEqual(interject_hook.spoken("t1"), [])

    def test_release_review_is_held_to_those_words(self):
        src = (Path(__file__).resolve().parent.parent / "src" / "pipeline.py").read_text(encoding="utf-8")
        tool = src.split("        async def release_review(", 1)[1].split("        @_tool", 1)[0]
        self.assertIn("heard = self._users_words()", tool)
        self.assertIn("if not heard.strip():", tool)
        self.assertEqual(tool.count("user_said, heard"), 3)
        self.assertNotIn('getattr(self, "_review_user_text", "")', tool.split("heard = self._users_words()", 1)[1]
                         .split("self._record_review_preference", 1)[0])


class AStopRaisedWithNoPageWatching(unittest.TestCase):
    """A lead woken by its branches has no page of its own: a watcher delivers its review."""

    def setUp(self):
        from src.utils import agentY_server as srv
        self.srv = srv
        srv._watch_sent.clear()
        self.addCleanup(srv._watch_sent.clear)

    def test_a_review_patch_goes_to_the_first_page_watching(self):
        ev = {"type": "canvas_patch", "op": "review_collector", "hook_node_id": "135"}
        self.assertTrue(self.srv._watch_delivers("run1", 7, ev))

    def test_and_only_once_so_a_later_look_does_not_undo_the_users_edits(self):
        ev = {"type": "canvas_patch", "op": "review_collector"}
        self.srv._watch_delivers("run1", 7, ev)
        self.assertFalse(self.srv._watch_delivers("run1", 7, ev))
        self.assertTrue(self.srv._watch_delivers("run2", 7, ev))      # another run's own

    def test_edits_to_the_graph_are_still_not_replayed(self):
        for op in ("edit_graph", "place_text", "delete_nodes", "set_mode", ""):
            self.assertFalse(self.srv._watch_delivers("run1", 1, {"type": "canvas_patch", "op": op}))

    def test_the_stream_asks_before_it_skips(self):
        src = Path(self.srv.__file__).read_text(encoding="utf-8")
        gen = src.split('    @app.route("/agentY/runs/<rid>/stream"', 1)[1].split("    @app.route(", 1)[0]
        self.assertIn('if kind == "canvas_patch" and not _watch_delivers(rid, index, ev):', gen)


if __name__ == "__main__":
    unittest.main()
