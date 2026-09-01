"""Asking a person before the agent runs something irreversible.

Two things decide whether this is worth having, and both are tested here rather
than described: it must BLOCK (a prompt the tool races past is decoration), and
the answer it gives when nobody replies must be no.

The blocking is exercised with real threads, because the deadlock this design
avoids is a threading fact: the agent's thread is stopped inside the tool while
the question is outstanding, which is why the question travels on its own
connection instead of on the turn's event stream.
"""

import threading
import time
import unittest

from src.utils import tool_permissions as tp


def answer_after(delay, getter, **reply):
    """Answer whichever question is waiting, once one appears.

    Polls rather than sleeping a fixed time: a test that raced would fail on a
    loaded machine and pass on a quiet one, which is worse than no test.
    """
    def run():
        deadline = time.time() + 5.0
        while time.time() < deadline:
            req = getter()
            if req:
                time.sleep(delay)
                tp.answer(req["permission_id"], **reply)
                return
            time.sleep(0.01)
    t = threading.Thread(target=run, daemon=True)
    t.start()
    return t


class TheDefaultIsNo(unittest.TestCase):
    """Every way of not answering has to land on the same side."""

    def setUp(self):
        tp.reset_session()
        tp._last_poll = time.time()          # pretend a panel is polling

    def test_a_timeout_declines(self):
        started = time.time()
        decision = tp.request("run_script", {"command": "rm -rf /"}, timeout=1.0)
        self.assertFalse(decision.allowed)
        self.assertIn("nobody approved", decision.reason)
        self.assertGreaterEqual(time.time() - started, 1.0,
                                "it must actually wait, not fall straight through")

    def test_no_panel_means_no(self):
        """Nobody watching is a worse moment than usual to run an unreviewed
        command, so the absent person's answer is the safe one."""
        tp._last_poll = 0.0
        decision = tp.request("run_script", {"command": "x"}, timeout=1.0)
        self.assertFalse(decision.allowed)
        self.assertIn("no agentY panel is open", decision.reason)

    def test_an_unattended_agent_can_be_allowed_on_purpose(self):
        tp._last_poll = 0.0
        decision = tp.request("run_script", {"command": "x"}, timeout=1.0,
                              unattended_allows=True)
        self.assertTrue(decision.allowed)

    def test_an_explicit_no(self):
        answer_after(0, tp.take, allowed=False)
        decision = tp.request("run_script", {"command": "x"}, timeout=5.0)
        self.assertFalse(decision.allowed)


class TheAnswers(unittest.TestCase):
    def setUp(self):
        tp.reset_session()
        tp._last_poll = time.time()

    def test_allow_once_does_not_stick(self):
        answer_after(0, tp.take, allowed=True)
        self.assertTrue(tp.request("run_script", {"command": "a"}, timeout=5.0).allowed)
        self.assertEqual(tp.granted_for_session(), [])
        # The next call must ask again, so with nobody answering it declines.
        self.assertFalse(tp.request("run_script", {"command": "b"}, timeout=1.0).allowed)

    def test_allow_for_session_stops_asking(self):
        answer_after(0, tp.take, allowed=True, remember=True)
        self.assertTrue(tp.request("run_script", {"command": "a"}, timeout=5.0).allowed)
        self.assertEqual(tp.granted_for_session(), ["run_script"])
        # No answerer this time: it must not block at all.
        started = time.time()
        second = tp.request("run_script", {"command": "b"}, timeout=5.0)
        self.assertTrue(second.allowed)
        self.assertTrue(second.remembered)
        self.assertLess(time.time() - started, 1.0)

    def test_a_grant_covers_only_the_tool_it_was_given_for(self):
        answer_after(0, tp.take, allowed=True, remember=True)
        tp.request("run_script", {"command": "a"}, timeout=5.0)
        self.assertFalse(tp.request("iterate", {"python_call": "1"}, timeout=1.0).allowed)

    def test_a_restart_forgets_every_grant(self):
        """"For this session" must not outlive the session it was given in.

        The host calls reset_session() at startup for exactly this.
        """
        answer_after(0, tp.take, allowed=True, remember=True)
        tp.request("run_script", {"command": "a"}, timeout=5.0)
        tp.reset_session()
        self.assertEqual(tp.granted_for_session(), [])

    def test_answering_a_question_nobody_is_waiting_on(self):
        # The waiter timed out first, or the turn was stopped. Reported, not
        # raised: the panel should take the prompt down either way.
        self.assertFalse(tp.answer("no-such-id", True))


class TheQueue(unittest.TestCase):
    def setUp(self):
        tp.reset_session()
        tp._last_poll = time.time()

    def test_a_question_is_only_handed_out_once(self):
        """Two polls must not hand the same question to two places.

        The panel polls on an interval and ComfyUI can have several tabs open;
        without the reservation both would draw the same prompt and the second
        answer would land on nothing.
        """
        t = threading.Thread(
            target=lambda: tp.request("run_script", {"command": "x"}, timeout=3.0),
            daemon=True)
        t.start()
        deadline = time.time() + 3.0
        first = None
        while first is None and time.time() < deadline:
            first = tp.take()
            time.sleep(0.01)
        self.assertIsNotNone(first)
        self.assertIsNone(tp.take(), "the same question came back a second time")
        tp.answer(first["permission_id"], False)
        t.join(timeout=3.0)

    def test_a_reload_can_pick_up_an_unanswered_question(self):
        """A page that vanished mid-question must release it.

        Otherwise the waiter is stranded until its timeout with a live panel sat
        right there able to answer.
        """
        t = threading.Thread(
            target=lambda: tp.request("run_script", {"command": "x"}, timeout=12.0),
            daemon=True)
        t.start()
        deadline = time.time() + 3.0
        first = None
        while first is None and time.time() < deadline:
            first = tp.take()
            time.sleep(0.01)
        self.assertIsNotNone(first)
        tp._SERVED.clear()                       # what the reservation lapsing does
        again = tp.take()
        self.assertIsNotNone(again)
        self.assertEqual(again["permission_id"], first["permission_id"])
        tp.answer(first["permission_id"], False)
        t.join(timeout=3.0)

    def test_a_long_poll_returns_the_moment_a_question_arrives(self):
        """Held open on the server, not asked for on a timer.

        Every request carries the session token, which makes it a non-simple
        request — so the browser sends a CORS preflight first and a one-second
        poll costs TWO access-log lines a second. Waiting here instead makes the
        panel quieter AND faster to show the prompt.
        """
        started = time.time()
        threading.Timer(0.3, lambda: threading.Thread(
            target=lambda: tp.request("run_script", {"command": "x"}, timeout=5.0),
            daemon=True).start()).start()
        got = tp.take(wait=5.0)
        elapsed = time.time() - started
        self.assertIsNotNone(got)
        self.assertLess(elapsed, 2.0, "it waited for a tick instead of being woken")
        self.assertGreater(elapsed, 0.25, "it cannot have had the answer already")
        tp.answer(got["permission_id"], False)

    def test_a_long_poll_with_nothing_to_say_gives_up_on_time(self):
        started = time.time()
        self.assertIsNone(tp.take(wait=1.0))
        self.assertGreaterEqual(time.time() - started, 1.0)

    def test_waiting_counts_as_listening(self):
        """has_listener() is read while a poll is BLOCKED in here.

        Recording the poll on the way out would call the panel absent for exactly
        the time it was most present — and "no panel is open" declines the
        request.
        """
        tp._last_poll = 0.0
        done = threading.Event()
        threading.Thread(target=lambda: (tp.take(wait=1.5), done.set()),
                         daemon=True).start()
        time.sleep(0.4)
        self.assertTrue(tp.has_listener())
        done.wait(timeout=3.0)

    def test_polling_registers_a_listener(self):
        tp._last_poll = 0.0
        self.assertFalse(tp.has_listener())
        tp.take()
        self.assertTrue(tp.has_listener())


class WhatTheUserIsShown(unittest.TestCase):
    """The command itself, never a paraphrase.

    A prompt that summarised would be asking the user to trust the summary, which
    is the one thing an approval step exists not to do.
    """

    def test_run_script_shows_the_command_verbatim(self):
        self.assertEqual(
            tp.describe("run_script", {"command": "python ./x.py --flag"}),
            "python ./x.py --flag")

    def test_iterate_shows_the_code_and_the_count(self):
        out = tp.describe("iterate", {"python_call": "os.remove(p)", "iter": 40})
        self.assertIn("os.remove(p)", out)
        self.assertIn("40", out)

    def test_installing_a_node_pack_says_it_will_pip_install(self):
        out = tp.describe("install_custom_node", {"source": "https://x/y"})
        self.assertIn("https://x/y", out)
        self.assertIn("pip", out)

    def test_turning_off_pip_is_reflected(self):
        out = tp.describe("install_custom_node", {"source": "https://x/y", "run_pip": False})
        self.assertNotIn("pip", out)

    def test_an_unknown_tool_still_shows_its_input(self):
        # New tools get added to the ask-list before anyone writes a describer.
        # Showing the raw input beats showing nothing.
        self.assertIn("danger", tp.describe("some_new_tool", {"arg": "danger"}))


class TheDefaultList(unittest.TestCase):
    def test_it_is_the_tools_whose_effects_leave_this_process(self):
        """Not "dangerous" tools — irreversible ones.

        Reading a file, searching the web and editing a graph are recoverable.
        Running a program, evaluating Python, and installing code from a stranger
        are not.
        """
        self.assertEqual(set(tp.DEFAULT_ASK_TOOLS),
                         {"run_script", "iterate", "install_custom_node"})

    def test_the_settings_file_agrees_with_the_code(self):
        """A default that drifted would be a gate that quietly stopped covering
        something."""
        import tomllib
        from pathlib import Path
        root = Path(__file__).resolve().parents[1]
        with (root / "config" / "settings.default.toml").open("rb") as fh:
            security = tomllib.load(fh).get("security", {})
        self.assertEqual(set(security.get("ask_before_tools", [])),
                         set(tp.DEFAULT_ASK_TOOLS))
        self.assertEqual(security.get("unattended_tool_policy"), "deny")


if __name__ == "__main__":
    unittest.main()
