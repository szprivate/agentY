"""What a turn looks like once it reaches Slack.

The panel and Slack show the same turn; the split is the one the panel already
makes on screen. The answer and any generated media go where they will be seen;
tool calls, reasoning, plans and canvas edits go into the message's thread, which
is what a collapsible block is for. Progress rewrites one line rather than
posting a hundred.

Getting this wrong does not fail — it floods. A hook run that posts every event
as its own message is a channel nobody keeps open, which is the same as having no
second channel at all.

    python -m unittest discover -s tests
"""

import unittest

from src.utils.slack_render import TurnRender, clip, to_mrkdwn


def _render(**kw):
    return TurnRender(**kw)


def _feed(r, *events):
    out = []
    for ev in events:
        out.extend(r.feed(ev))
    return out


def _where(posts):
    return [p.where for p in posts]


class MrkdwnTest(unittest.TestCase):
    """Slack is not markdown, and the agent writes markdown."""

    def test_bold_loses_a_star(self):
        self.assertEqual(to_mrkdwn("a **bold** word"), "a *bold* word")

    def test_headings_become_bold_because_slack_has_none(self):
        self.assertEqual(to_mrkdwn("## Results\ntext"), "*Results*\ntext")

    def test_links_are_rewritten(self):
        self.assertEqual(to_mrkdwn("see [docs](https://x.dev/a)"),
                         "see <https://x.dev/a|docs>")

    def test_bullets_are_rewritten(self):
        self.assertEqual(to_mrkdwn("- one\n- two"), "• one\n• two")

    def test_code_fences_are_left_exactly_alone(self):
        src = "```\n**not bold** - not a bullet\n```"
        self.assertEqual(to_mrkdwn(src), src)

    def test_text_around_a_fence_is_still_converted(self):
        got = to_mrkdwn("**a**\n```\n**b**\n```\n**c**")
        self.assertEqual(got, "*a*\n```\n**b**\n```\n*c*")

    def test_a_lone_star_is_not_mangled(self):
        self.assertEqual(to_mrkdwn("2 * 3 = 6"), "2 * 3 = 6")


class ClipTest(unittest.TestCase):

    def test_short_text_is_untouched(self):
        self.assertEqual(clip("abc", 10), "abc")

    def test_long_text_says_how_much_was_cut(self):
        got = clip("x" * 100, 10)
        self.assertTrue(got.startswith("x" * 10))
        self.assertIn("+90 more characters", got)


class AnswerTest(unittest.TestCase):

    def test_the_answer_accumulates_into_one_message(self):
        r = _render()
        posts = _feed(r, {"type": "text", "data": "Ren"}, {"type": "text", "data": "dered."})
        self.assertEqual(_where(posts), ["answer", "answer"])
        self.assertEqual(r.answer, "Rendered.")

    def test_it_reads_as_still_going_until_it_is_done(self):
        r = _render()
        _feed(r, {"type": "text", "data": "working"})
        self.assertTrue(r.body().endswith("_…_"))
        _feed(r, {"type": "done"})
        self.assertFalse(r.body().endswith("_…_"))

    def test_a_turn_that_says_nothing_still_closes_its_message(self):
        """Silence at the end is indistinguishable from a crash."""
        posts = _feed(_render(), {"type": "done"})
        answer = [p for p in posts if p.where == "answer"]
        self.assertEqual(len(answer), 1)
        self.assertIn("nothing to report", answer[0].text)

    def test_a_turn_that_only_made_files_does_not_claim_it_said_nothing(self):
        r = _render()
        posts = _feed(r, {"type": "output", "path": "a.png", "name": "a.png"},
                      {"type": "done"})
        answer = [p for p in posts if p.where == "answer"][-1]
        self.assertNotIn("nothing to report", answer.text)

    def test_the_opening_says_where_the_turn_came_from(self):
        """A panel turn appearing in Slack unannounced reads as the bot talking
        to itself."""
        self.assertIn("panel", _render(origin="panel", started_by="go").opening())
        self.assertNotIn("panel", _render(origin="slack", started_by="go").opening())

    def test_the_opening_quotes_what_was_asked(self):
        self.assertIn("make it warmer",
                      _render(origin="slack", started_by="make it warmer").opening())

    def test_a_panel_turn_is_marked_on_the_ANSWER_too(self):
        """A fast turn never shows an opening message — text arrives first and
        the answer message is created from it. The attribution has to survive
        that, or it is lost exactly when the turn is quick."""
        r = _render(origin="panel", started_by="go")
        _feed(r, {"type": "text", "data": "Done."})
        self.assertIn("panel", r.body())

    def test_a_slack_turn_is_not_marked(self):
        """You know you asked; saying so every time is noise."""
        r = _render(origin="slack", started_by="go")
        _feed(r, {"type": "text", "data": "Done."})
        self.assertNotIn("panel", r.body())

    def test_nothing_is_rendered_after_done(self):
        r = _render()
        _feed(r, {"type": "done"})
        self.assertEqual(_feed(r, {"type": "text", "data": "late"}), [])


class WhatMustBeSeenTest(unittest.TestCase):
    """Media, questions and errors go where you do not have to go looking."""

    def test_generated_media_is_uploaded_to_the_channel(self):
        posts = _feed(_render(), {"type": "output", "path": "D:/out/a.png",
                                  "name": "a.png", "role": "hero sheet"})
        self.assertEqual(posts[0].where, "channel")
        self.assertEqual(posts[0].kind, "file")
        self.assertEqual(posts[0].path, "D:/out/a.png")
        self.assertIn("hero sheet", posts[0].text)

    def test_a_question_is_its_own_message(self):
        posts = _feed(_render(), {"type": "ask", "prompt": "Retry the failed one?"})
        self.assertEqual(posts[0].where, "channel")
        self.assertIn("Retry the failed one?", posts[0].text)
        self.assertIn("Reply here", posts[0].text)

    def test_an_error_is_not_buried_in_the_thread(self):
        posts = _feed(_render(), {"type": "error", "message": "ComfyUI refused it"})
        self.assertEqual(posts[0].where, "channel")
        self.assertIn("ComfyUI refused it", posts[0].text)

    def test_a_background_notification_brings_its_file_with_it(self):
        posts = _feed(_render(), {"type": "notify", "kind": "media",
                                  "toast": {"title": "Magnific", "body": "upscale done"},
                                  "output": {"path": "D:/out/up.png", "name": "up.png"}})
        self.assertEqual([p.kind for p in posts], ["text", "file"])
        self.assertIn("Magnific", posts[0].text)
        self.assertEqual(posts[1].path, "D:/out/up.png")


class ThreadTest(unittest.TestCase):
    """The detail, kept out of the way of the answer."""

    def test_tool_calls_go_to_the_thread(self):
        posts = _feed(_render(), {"type": "tool", "phase": "call", "id": "t1",
                                  "name": "run_research", "input": "{'q': 'x'}"})
        self.assertEqual(posts[0].where, "detail")
        self.assertIn("run_research", posts[0].text)

    def test_a_result_rewrites_its_call_instead_of_following_it(self):
        r = _render()
        call = _feed(r, {"type": "tool", "phase": "call", "id": "t1", "name": "go"})
        done = _feed(r, {"type": "tool", "phase": "result", "id": "t1", "name": "go",
                         "result": "4 images"})
        self.assertEqual(call[0].key, done[0].key, "same message, rewritten")
        self.assertEqual(done[0].where, "detail", "the result belongs in the thread too")
        self.assertIn("4 images", done[0].text)

    def test_a_failed_tool_is_marked_as_failed(self):
        posts = _feed(_render(), {"type": "tool", "phase": "result", "id": "t1",
                                  "name": "go", "result": "error: nope"})
        self.assertTrue(posts[0].text.startswith("⚠️"))

    def test_tools_can_be_turned_off(self):
        r = _render(show_tools=False)
        self.assertEqual(_feed(r, {"type": "tool", "phase": "call", "name": "go"}), [])

    def test_reasoning_is_one_message_not_a_thousand(self):
        """`think` arrives a few characters at a time; posting each is unusable."""
        r = _render()
        streamed = _feed(r, *[{"type": "think", "data": c} for c in "because reasons"])
        self.assertEqual(streamed, [])
        flushed = _feed(r, {"type": "step_end"})
        self.assertEqual(_where(flushed), ["detail"])
        self.assertIn("because reasons", flushed[0].text)

    def test_reasoning_left_open_still_arrives_at_the_end(self):
        r = _render()
        _feed(r, {"type": "think", "data": "half a thought"})
        posts = _feed(r, {"type": "done"})
        self.assertIn("half a thought", " ".join(p.text for p in posts))

    def test_reasoning_can_be_turned_off(self):
        r = _render(show_thinking=False)
        _feed(r, {"type": "think", "data": "hidden"})
        self.assertNotIn("hidden", " ".join(p.text for p in _feed(r, {"type": "done"})))

    def test_a_plan_is_numbered(self):
        posts = _feed(_render(), {"type": "plan", "steps": ["one", "two"]})
        self.assertEqual(posts[0].where, "detail")
        self.assertIn("1. one", posts[0].text)

    def test_a_canvas_edit_is_said_in_words(self):
        """The one part of a turn a phone genuinely cannot show."""
        posts = _feed(_render(), {"type": "canvas_patch", "op": "review_collector"})
        self.assertEqual(posts[0].where, "detail")
        self.assertIn("review node", posts[0].text)


class StatusTest(unittest.TestCase):
    """The transient line under the panel's composer, as one rewritten message."""

    def test_progress_rewrites_a_single_message(self):
        r = _render()
        a = _feed(r, {"type": "progress", "data": "step 1"})
        b = _feed(r, {"type": "progress", "data": "step 2"})
        self.assertEqual(a[0].key, "status")
        self.assertEqual(a[0].key, b[0].key)

    def test_qa_and_console_share_that_line(self):
        r = _render()
        for ev in ({"type": "qa", "data": "checking"}, {"type": "console", "data": "…"}):
            self.assertEqual(_feed(r, ev)[0].key, "status")

    def test_comfyui_running_sets_it_and_finishing_clears_it(self):
        r = _render()
        on = _feed(r, {"type": "exec", "state": "start"})
        off = _feed(r, {"type": "exec", "state": "end"})
        self.assertIn("ComfyUI", on[0].text)
        self.assertEqual(off[0].kind, "clear")

    def test_the_line_is_taken_down_when_the_turn_ends(self):
        posts = _feed(_render(), {"type": "done"})
        self.assertIn("clear", [p.kind for p in posts])

    def test_panel_bookkeeping_produces_nothing(self):
        r = _render()
        for t in ("thread", "request", "turn_start", "plan_step"):
            self.assertEqual(_feed(r, {"type": t}), [], t)

    def test_an_unknown_event_is_ignored_rather_than_crashing(self):
        self.assertEqual(_feed(_render(), {"type": "something_new_in_2027"}), [])


if __name__ == "__main__":
    unittest.main()
