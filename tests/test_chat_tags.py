"""``#tag`` in a chat message resolves to the canvas node or remembered file."""
import unittest
from unittest import mock

from src.utils.chat_tags import describe_chat_tags

CANVAS = {
    "12": {"class_type": "LoadImage", "inputs": {"image": "a.png"}},
    "13": {"class_type": "AgentYRefNote",
           "inputs": {"input": ["12", 0], "tag": "hero face", "role": "the face only"}},
}


class ChatTags(unittest.TestCase):
    def test_canvas_tag_resolves_to_its_node(self):
        out = describe_chat_tags("relight #hero_face please", CANVAS)
        self.assertIn("#hero_face -> node 12", out)
        self.assertIn("the face only", out)

    def test_memory_tag_resolves_to_its_file(self):
        ref = {"tag": "alley", "path": "D:/x/alley.png", "role": "lighting"}
        with mock.patch("src.utils.tag_memory.remembered_reference", return_value=ref):
            out = describe_chat_tags("use #alley", {})
        self.assertIn("D:/x/alley.png", out)

    def test_unknown_and_numeric_mentions_are_silent(self):
        with mock.patch("src.utils.tag_memory.remembered_reference", return_value=None):
            self.assertEqual(describe_chat_tags("post it #fyp, shot #3", CANVAS), "")
        self.assertEqual(describe_chat_tags("no tags here", CANVAS), "")


if __name__ == "__main__":
    unittest.main()
