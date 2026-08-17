"""`stop` at a review halt is not `stop` the agent host.

From a real run: the review halt tells the user, in these words, to "say continue
— or stop to end the run here", and the action-bar button sends exactly that. But
a bare `stop` was also the slash command that shuts the whole host down, and the
command check ran first — so ending a run at a review hook killed the server.

The two readings are not close. One ends a chain and keeps everything it made;
the other takes the agent down mid-conversation, and nothing on screen explains
which one you asked for.

    python -m unittest discover -s tests
"""

import unittest
from unittest import mock

from src.utils import agentY_server as srv


def _is_command(text, halted=False):
    with mock.patch.object(srv, "_halt_pending", return_value=halted):
        return srv._is_command(text, "t1")


class DuringAHaltTest(unittest.TestCase):

    def test_stop_answers_the_halt_instead_of_killing_the_host(self):
        self.assertFalse(_is_command("stop", halted=True))

    def test_continue_is_not_a_command_either(self):
        self.assertFalse(_is_command("continue", halted=True))

    def test_case_and_spacing_do_not_change_it(self):
        for text in ("Stop", "  STOP  ", "stop"):
            self.assertFalse(_is_command(text, halted=True), text)

    def test_the_other_commands_still_work_mid_halt(self):
        """A halt is not a lock on the panel — only two words are ambiguous."""
        for text in ("restart", "unload", "images", "clearhistory", "resend"):
            self.assertTrue(_is_command(text, halted=True), text)

    def test_a_typed_slash_command_is_always_a_command(self):
        """`/stop` is unambiguous: nobody types a slash to answer a question."""
        self.assertTrue(_is_command("/stop", halted=True))
        self.assertTrue(_is_command("/qa some briefing", halted=True))


class WithNoHaltTest(unittest.TestCase):

    def test_stop_is_still_the_command_it_always_was(self):
        self.assertTrue(_is_command("stop", halted=False))

    def test_continue_was_never_a_command(self):
        self.assertFalse(_is_command("continue", halted=False))

    def test_ordinary_text_is_never_a_command(self):
        for text in ("stop the render please", "make it warmer", ""):
            self.assertFalse(_is_command(text, halted=False), text)


class HaltPendingTest(unittest.TestCase):
    """What "there is a halt" is read from — the thread's stored session."""

    def _state(self, state):
        return mock.patch.object(srv.cs, "load_state", return_value=state)

    def test_a_stored_halt_counts(self):
        with self._state({"agent_session": {"review_halt": {"hook_node_id": "475"}}}):
            self.assertTrue(srv._halt_pending("t1"))

    def test_an_answered_halt_does_not(self):
        with self._state({"agent_session": {"review_halt": None}}):
            self.assertFalse(srv._halt_pending("t1"))

    def test_a_thread_with_no_state_does_not(self):
        with self._state(None):
            self.assertFalse(srv._halt_pending("t1"))

    def test_a_halt_with_no_hook_id_is_not_a_halt(self):
        with self._state({"agent_session": {"review_halt": {}}}):
            self.assertFalse(srv._halt_pending("t1"))

    def test_a_store_that_throws_answers_no(self):
        """The safe reading: every command keeps working as it always did."""
        with mock.patch.object(srv.cs, "load_state", side_effect=RuntimeError("db")):
            self.assertFalse(srv._halt_pending("t1"))

    def test_and_so_stop_still_reaches_the_command(self):
        with mock.patch.object(srv.cs, "load_state", side_effect=RuntimeError("db")):
            self.assertTrue(srv._is_command("stop", "t1"))


if __name__ == "__main__":
    unittest.main()
