"""A sound the agent makes reaches the chat panel and the viewer.

Both load the copy staged in ComfyUI's input dir. A sound with no loader node
chosen for it (Settings > Load & save nodes) is a plain "file" to the canvas, and
files used not to be staged - so the panel had nothing to play. Now it is staged
whatever the canvas makes of it, and the viewer is sent it like a picture.
"""
import unittest
from pathlib import Path

from src.utils import agentY_server as srv

ROOT = Path(__file__).resolve().parent.parent
SOURCE = (ROOT / "src" / "utils" / "agentY_server.py").read_text(encoding="utf-8")


class Sounds(unittest.TestCase):
    def test_what_is_a_sound(self):
        for name in ("a.wav", "b.MP3", "c.flac", "d.ogg", "e.m4a", "f.aac", "g.opus"):
            self.assertTrue(srv._is_audio_path(name), name)
        for name in ("a.png", "b.mp4", "c.txt", "wav"):
            self.assertFalse(srv._is_audio_path(name), name)

    def test_a_turns_sound_is_staged_although_the_canvas_takes_it_as_a_file(self):
        check = SOURCE.split("    def _check_outputs() -> None:", 1)[1].split("    def _emit_paths", 1)[0]
        self.assertIn('if kind != "file" or _is_audio_path(p) else None)', check)

    def test_it_goes_to_the_viewer_like_a_picture(self):
        deliver = SOURCE.split("def _deliver_elsewhere(", 1)[1].split("\ndef ", 1)[0]
        self.assertIn("_is_audio_path(path)", deliver)

    def test_shown_again_it_is_a_sound(self):
        again = SOURCE.split("    def _show_again(patch: dict) -> None:", 1)[1].split("    def _translate", 1)[0]
        self.assertIn('"audio" if _is_audio_path(p)', again)


if __name__ == "__main__":
    unittest.main()
