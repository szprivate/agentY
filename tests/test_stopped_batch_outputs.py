"""A batch cut short still delivers what it had finished."""
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from src import executor


class Salvage(unittest.TestCase):
    def setUp(self):
        self.dir = Path(tempfile.mkdtemp(prefix="agenty_salvage_"))
        for name in ("a.png", "b.png"):
            (self.dir / name).write_bytes(b"x")
        self.history = {
            "p1": {"outputs": {"30": {"images": [{"filename": "a.png", "subfolder": "", "type": "output"}]}}},
            "p2": {"outputs": {"30": {"images": [{"filename": "b.png", "subfolder": "", "type": "output"},
                                                 {"filename": "gone.png", "subfolder": "", "type": "output"}]}}},
        }

    def run_it(self, collected):
        client = mock.Mock()
        client.get.return_value = self.history
        with mock.patch("agenty_core.utils.comfyui_client.get_client", return_value=client), \
                mock.patch.object(executor, "_resolve_output_path",
                                  side_effect=lambda f, s="", t="output", **k: self.dir / f):
            return executor._salvage_finished_outputs(collected)

    def test_finished_files_are_collected(self):
        collected: list = []
        self.assertEqual(self.run_it(collected), 2)
        self.assertEqual(sorted(Path(p).name for p in collected), ["a.png", "b.png"])

    def test_a_file_already_delivered_is_not_delivered_twice(self):
        collected = [str(self.dir / "a.png")]
        self.assertEqual(self.run_it(collected), 1)
        self.assertEqual(len(collected), 2)

    def test_a_file_that_is_not_on_disk_is_left_out(self):
        collected: list = []
        self.run_it(collected)
        self.assertNotIn("gone.png", [Path(p).name for p in collected])

    def test_no_list_to_fill_is_not_an_error(self):
        self.assertEqual(executor._salvage_finished_outputs(None), 0)

    def test_an_unreachable_comfyui_is_not_an_error(self):
        with mock.patch("agenty_core.utils.comfyui_client.get_client", side_effect=RuntimeError("down")):
            self.assertEqual(executor._salvage_finished_outputs([]), 0)


if __name__ == "__main__":
    unittest.main()
