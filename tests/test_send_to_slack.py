"""The agent choosing to hand you a file.

The mirror already uploads what a run *produced*. This tool is for the rest —
the JSON it wrote, one frame out of sixty, a script, a log — and its whole value
is being usable when you are not at the machine, which is exactly when a silent
failure is least recoverable. So: every refusal says what to do instead, and the
tool is not offered at all where there is no Slack to send to (a tool nobody can
use is a standing token cost on every call and a standing invitation to try it).

    python -m unittest discover -s tests
"""

import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from pipeline_stub import pipeline_stub, tools
from src.pipeline import _MAX_SLACK_FILES


class FakeBridge:
    def __init__(self, result=None):
        self.calls = []
        self.result = result

    def send_files(self, paths, message=""):
        self.calls.append({"paths": list(paths), "message": message})
        if self.result is not None:
            return self.result
        return {"sent": list(paths), "missing": [], "too_large": []}


def _call(bridge, **kw):
    with mock.patch("src.utils.slack_bridge.current", return_value=bridge), \
         mock.patch("src.utils.slack_bridge.enabled", return_value=True):
        pipe = pipeline_stub()
        return json.loads(asyncio.run(tools(pipe)["send_to_slack"](**kw)))


class SendTest(unittest.TestCase):

    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)

    def _file(self, name):
        p = Path(self.dir.name) / name
        p.write_bytes(b"x")
        return str(p)

    def test_it_sends_and_says_what_went(self):
        b = FakeBridge()
        out = _call(b, paths=[self._file("shots.json")], message="the shot list")
        self.assertEqual(out["status"], "sent")
        self.assertEqual(b.calls[0]["message"], "the shot list")
        self.assertIn("Sent 1 file(s)", out["message"])

    def test_several_files_go_in_one_call(self):
        b = FakeBridge()
        paths = [self._file("a.png"), self._file("b.json")]
        _call(b, paths=paths)
        self.assertEqual(b.calls[0]["paths"], paths)

    def test_duplicates_collapse(self):
        b = FakeBridge()
        p = self._file("a.png")
        _call(b, paths=[p, p, p])
        self.assertEqual(b.calls[0]["paths"], [p])

    def test_nothing_to_send_is_refused_rather_than_reported_as_sent(self):
        b = FakeBridge()
        out = _call(b, paths=[])
        self.assertIn("no paths given", out["error"])
        self.assertEqual(b.calls, [])

    def test_blank_entries_do_not_count_as_paths(self):
        b = FakeBridge()
        self.assertIn("no paths given", _call(b, paths=["", "   "])["error"])

    def test_a_runaway_send_is_capped(self):
        """A DM is where someone reads one thing on a phone, not a folder."""
        b = FakeBridge()
        out = _call(b, paths=[self._file(f"{i}.png") for i in range(_MAX_SLACK_FILES + 1)])
        self.assertIn("more than this tool will send", out["error"])
        self.assertEqual(b.calls, [], "it must not send a partial pile either")

    def test_the_cap_does_not_fire_at_the_boundary(self):
        b = FakeBridge()
        out = _call(b, paths=[self._file(f"{i}.png") for i in range(_MAX_SLACK_FILES)])
        self.assertEqual(out["status"], "sent")

    def test_files_that_could_not_go_are_named_in_the_result(self):
        """The agent has to be able to tell the user, in the same breath."""
        b = FakeBridge({"sent": ["a.png"], "missing": ["gone.png"],
                        "too_large": ["huge.mp4"]})
        out = _call(b, paths=[self._file("a.png")])
        self.assertIn("gone.png", out["message"])
        self.assertIn("huge.mp4", out["message"])
        self.assertIn("give the user the path instead", out["message"])

    def test_a_bridge_level_error_is_passed_through(self):
        b = FakeBridge({"error": "the Slack bridge has nowhere to post"})
        self.assertIn("nowhere to post", _call(b, paths=[self._file("a.png")])["error"])


class NoSlackTest(unittest.TestCase):

    def test_with_the_bridge_down_it_hands_back_the_path(self):
        """Failing here is failing at the moment the user is away from the
        machine — the answer has to be something they can act on."""
        out = _call(None, paths=["D:/out/a.png"])
        self.assertIn("not running", out["error"])
        self.assertIn("give them the path", out["error"])
        self.assertEqual(out["paths"], ["D:/out/a.png"])

    def test_the_tool_is_not_offered_when_slack_is_off(self):
        with mock.patch("src.utils.slack_bridge.enabled", return_value=False):
            self.assertNotIn("send_to_slack", tools(pipeline_stub()))

    def test_and_is_offered_when_it_is_on(self):
        with mock.patch("src.utils.slack_bridge.enabled", return_value=True):
            self.assertIn("send_to_slack", tools(pipeline_stub()))

    def test_a_broken_settings_read_does_not_cost_the_whole_toolset(self):
        with mock.patch("src.utils.slack_bridge.enabled",
                        side_effect=RuntimeError("settings on fire")):
            names = tools(pipeline_stub())
        self.assertNotIn("send_to_slack", names)
        self.assertIn("run_workflow_now", names, "the rest of the tools survived")


if __name__ == "__main__":
    unittest.main()
