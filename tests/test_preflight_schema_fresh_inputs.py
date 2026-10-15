"""preflight's per-class schema cache forgets itself when a file lands in the input dir.

LoadImage's menu is ComfyUI's input-dir listing at fetch time. Kept for the whole
session, it made insert_workflow_into_canvas refuse the photo the agent had just
uploaded ("'2f79780a_me.png' is not an option"): the graph ran, but never reached
the canvas (seen 2026-10-14, four runs out of five).

    python -m unittest tests.test_preflight_schema_fresh_inputs
"""
import unittest
from unittest import mock

from src.utils import preflight
from src.utils import canvas_edit


def _load_image(files):
    return {"input": {"required": {"image": [list(files), {"image_upload": True}]}},
            "output": ["IMAGE", "MASK"]}


class SchemaCacheFollowsTheInputDir(unittest.TestCase):
    def setUp(self):
        preflight._schema_cache.clear()
        preflight._schema_stamp = None
        self.files = ["old_me.png"]
        self.calls = 0

        def get(path):
            self.calls += 1
            return {"LoadImage": _load_image(self.files)}

        client = mock.Mock(get=get)
        self.p_client = mock.patch("agenty_core.utils.comfyui_client.get_client", return_value=client)
        self.p_client.start()
        self.stamp = (1, 1)
        self.p_stamp = mock.patch.object(preflight, "_input_dir_stamp", lambda: self.stamp)
        self.p_stamp.start()

    def tearDown(self):
        self.p_client.stop()
        self.p_stamp.stop()
        preflight._schema_cache.clear()
        preflight._schema_stamp = None

    def _menu(self):
        return preflight._schema("LoadImage")["input"]["required"]["image"][0]

    def test_cached_while_the_dir_is_unchanged(self):
        self._menu(); self._menu()
        self.assertEqual(self.calls, 1)

    def test_a_new_upload_is_an_option_at_once(self):
        self.assertEqual(self._menu(), ["old_me.png"])
        self.files = ["old_me.png", "2f79780a_me.png"]   # the agent uploads the photo
        self.stamp = (2, 1)                               # the input dir changed
        self.assertIn("2f79780a_me.png", self._menu())
        spec = preflight._schema("LoadImage")["input"]["required"]["image"]
        value, err = canvas_edit.coerce_value(spec, "2f79780a_me.png")
        self.assertIsNone(err)
        self.assertEqual(value, "2f79780a_me.png")

    def test_no_dir_on_this_machine_keeps_the_cache(self):
        self.stamp = None
        self._menu(); self._menu()
        self.assertEqual(self.calls, 1)


if __name__ == "__main__":
    unittest.main()
