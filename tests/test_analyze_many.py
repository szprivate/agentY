"""analyze_image(file_paths=[...]) looks at several images in ONE vision call.

    python -m unittest tests.test_analyze_many
"""
import os
import tempfile
import unittest
from unittest import mock

from PIL import Image

from src.tools import image_handling as ih


def _text(out):
    return out["content"][0]["text"]


class Many(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.TemporaryDirectory()
        self.addCleanup(self.d.cleanup)
        self.paths = []
        for i in range(3):
            p = os.path.join(self.d.name, f"shot{i}.png")
            Image.new("RGB", (80, 60), (i * 80, 0, 0)).save(p)
            self.paths.append(p)

    def test_three_images_make_one_vision_call_and_a_legend(self):
        calls = []
        real = ih.analyze_image

        def spy(**kw):
            calls.append(kw)
            return {"status": "success", "content": [{"text": "1: red. 2: dark. 3: brighter."}]}

        with mock.patch.object(ih, "analyze_image", side_effect=spy):
            out = ih._analyze_many(self.paths, "Which is brightest?", "describe")
        self.assertEqual(len(calls), 1)
        self.assertIn("contact sheet of 3 separate images", calls[0]["question"])
        self.assertIn("Which is brightest?", calls[0]["question"])
        self.assertIn("1 = " + self.paths[0], _text(out))
        self.assertIn("3: brighter", _text(out))
        self.assertFalse(os.path.exists(calls[0]["file_path"]), "temp sheet is removed")

    def test_the_public_tool_routes_a_list_to_the_sheet(self):
        with mock.patch.object(ih, "_analyze_many", return_value={"status": "success", "content": [{"text": "ok"}]}) as m:
            ih.analyze_image(file_paths=self.paths, question="q")
        m.assert_called_once()
        self.assertEqual(m.call_args[0][0], self.paths)

    def test_a_single_path_in_the_list_is_a_plain_call(self):
        with mock.patch.dict(os.environ, {"AGENTY_STUB_VISION": "1"}):
            out = ih.analyze_image(file_paths=[self.paths[0]])
        self.assertIn("shot0.png", _text(out))

    def test_unreadable_set_is_an_error_not_a_crash(self):
        out = ih._analyze_many([os.path.join(self.d.name, "no.png"), os.path.join(self.d.name, "no2.png")], "", "describe")
        self.assertEqual(out["status"], "error")


if __name__ == "__main__":
    unittest.main()
