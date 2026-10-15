"""The chat line a model download reports its progress with.

tqdm's own output used to be forwarded: with two downloads at once it is cursor
movement between two bars ("ESC[A"), and at a chat-line width it has no room for
the bar, so the panel showed "[A" rows and one new row per refresh. The line is
now made for the chat, and the panel keys its row by the file name it starts with.

    python -m unittest tests.test_download_progress_line
"""
import re
import unittest

from agenty_core.tools.huggingface import _progress_line

GB = 1024 ** 3
MB = 1024 ** 2


class DownloadProgressLine(unittest.TestCase):
    def test_line_names_the_file_and_carries_a_bar(self):
        line = _progress_line("z_image_turbo_bf16.safetensors", int(1.0 * GB), int(11.5 * GB), 12.8 * MB)
        # The panel's per-file key: "⬇️ <name> " followed by the bar.
        self.assertRegex(line, r"^⬇️ z_image_turbo_bf16\.safetensors [█░]{20} 9% · ")
        for part in ("1.0/11.5 GB", "12.8 MB/s", "min left"):
            self.assertIn(part, line)

    def test_no_terminal_escape_codes(self):
        line = _progress_line("qwen_3_4b.safetensors", 3 * GB, 8 * GB, 20 * MB)
        for ch in ("\x1b", "\r", "\n"):
            self.assertNotIn(ch, line)

    def test_finished_and_unknown_size(self):
        done = _progress_line("a.safetensors", 2 * GB, 2 * GB, None)
        self.assertIn("█" * 20, done)
        self.assertIn("100%", done)
        self.assertTrue(done.endswith("done"))
        unknown = _progress_line("b.bin", int(0.5 * GB), 0, 5 * MB)
        self.assertNotIn("█", unknown)
        self.assertIn("0.5 GB", unknown)
        self.assertNotIn("left", unknown)

    def test_short_remainder_in_seconds(self):
        self.assertTrue(_progress_line("c.bin", 99 * MB, 100 * MB, 1 * MB).endswith("1 s left"))

    def test_panel_key_pattern_matches(self):
        # Same pattern agent_chat.js keys a download's row by.
        line = _progress_line("qwen_3_4b.safetensors", GB, 8 * GB, 10 * MB)
        m = re.match(r"^⬇️ (\S+) [█░]", line)
        self.assertIsNotNone(m)
        self.assertEqual(m.group(1), "qwen_3_4b.safetensors")


if __name__ == "__main__":
    unittest.main()
