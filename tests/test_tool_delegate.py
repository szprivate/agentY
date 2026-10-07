"""The tool card says which agent really answered.

analyze_image is called by the orchestrator and answered by the vision agent.
The panel showed "[orchestrator] analyze_image" and nothing else, which read as
the orchestrator looking at every image itself - and the vision model in
Settings as a switch wired to nothing.
"""
import unittest
from types import SimpleNamespace
from unittest import mock

from agenty_core.utils import turn_scope

from src.agent import ToolActivityHookProvider
from src.tools import image_handling, video_handling
from src.utils import tool_activity
from src.utils.tool_delegate import delegate_for


def _agent(model_id):
    return SimpleNamespace(model=SimpleNamespace(config={"model_id": model_id}))


class WhoAnswers(unittest.TestCase):

    def setUp(self):
        self.enterContext(mock.patch.object(image_handling, "_vision_agent", _agent("qwen3-vl-flash")))
        self.enterContext(mock.patch.object(video_handling, "_video_agent", _agent("qwen3-vl-plus")))

    def test_an_image_is_described_by_the_vision_agent(self):
        want = {"agent": "vision", "model": "qwen3-vl-flash"}
        self.assertEqual(delegate_for("analyze_image", {"file_path": "a.png"}), want)
        self.assertEqual(delegate_for("analyze_image", {"file_path": "a.png", "mode": "describe"}), want)

    def test_full_mode_is_the_caller_looking_for_itself(self):
        self.assertIsNone(delegate_for("analyze_image", {"file_path": "a.png", "mode": "full"}))

    def test_a_video_is_read_by_the_video_agent(self):
        self.assertEqual(delegate_for("analyze_video", {"file_path": "a.mp4"}),
                         {"agent": "video", "model": "qwen3-vl-plus"})

    def test_any_other_tool_is_answered_by_whoever_called_it(self):
        self.assertIsNone(delegate_for("upload_image", {"file_path": "a.png"}))
        self.assertIsNone(delegate_for("view_image", {}))

    def test_no_agent_registered_is_nobody_to_credit(self):
        with mock.patch.object(image_handling, "_vision_agent", None):
            self.assertIsNone(delegate_for("analyze_image", {"file_path": "a.png"}))

    def test_an_unreadable_model_still_names_the_agent(self):
        with mock.patch.object(image_handling, "_vision_agent", object()):
            self.assertEqual(delegate_for("analyze_image", {}), {"agent": "vision", "model": ""})

    def test_odd_input_never_raises(self):
        self.assertEqual(delegate_for("analyze_image", None)["agent"], "vision")
        self.assertIsNone(delegate_for("analyze_image", {"mode": "FULL"}))


class OnTheCard(unittest.TestCase):

    def setUp(self):
        token = turn_scope.enter(turn_scope.Scope("req", "thread"))
        self.addCleanup(turn_scope.leave, token)
        self.enterContext(mock.patch.object(image_handling, "_vision_agent", _agent("qwen3-vl-flash")))

    def _call(self, name, tool_input):
        event = SimpleNamespace(tool_use={"toolUseId": "t1", "name": name, "input": tool_input})
        ToolActivityHookProvider("orchestrator")._on_before(event)
        return tool_activity.drain()[0]

    def test_the_call_carries_who_answers_it(self):
        card = self._call("analyze_image", {"file_path": "a.png"})
        self.assertEqual(card["agent"], "orchestrator")
        self.assertEqual(card["via"], {"agent": "vision", "model": "qwen3-vl-flash"})

    def test_an_ordinary_call_carries_nothing_extra(self):
        self.assertNotIn("via", self._call("upload_image", {"file_path": "a.png"}))


if __name__ == "__main__":
    unittest.main()
