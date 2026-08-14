"""A generated file remembers what it was for.

An output leaves as a path and arrives on the canvas as a loader node named after
its file — every time, the next turn had to look at the picture again to find out
what it was. Three records fix that, and they are tested here: the per-turn
registry the panel reads as each file lands, the sidecar that outlives the thread,
and the role the user states in the hook's own prompt, which is the one that earns
a ref note on their canvas.

    python -m unittest discover -s tests
"""

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from src.pipeline import Pipeline
from src.utils import output_tags as ot
from src.utils.canvas_hooks import _render_anchor, declared_output_role


def _file(tmp, name="shot_01.png", body=b"x"):
    p = Path(tmp) / name
    p.write_bytes(body)
    return p


class DeclaredRoleTest(unittest.TestCase):
    """What the user wrote in the hook's prompt — the only thing that decorates."""

    def test_the_forms_that_count_as_stating_a_role(self):
        for text, want in [
            ("Make a start frame per shot.\nrole: shot start frame", "shot start frame"),
            ("[role: hero character sheet] build the sheet", "hero character sheet"),
            ("Generate the refs and tag the outputs as \"alley night\"", "alley night"),
            ("Render them and label the results as mood boards", "mood boards"),
            ("ROLE = lighting reference", "lighting reference"),
        ]:
            with self.subTest(text=text):
                self.assertEqual(declared_output_role(text), want)

    def test_an_ordinary_directive_states_nothing(self):
        for text in [
            "Create one starting image for each of the prompts.",
            "Describe the style and colour of this frame.",
            "Break the story down into shots; keep the role of the hero consistent.",
        ]:
            with self.subTest(text=text):
                self.assertEqual(declared_output_role(text), "")

    def test_it_reads_a_hook_dict_too(self):
        self.assertEqual(declared_output_role({"directive": "role: key art"}), "key art")
        self.assertEqual(declared_output_role({}), "")


class RegistryTest(unittest.TestCase):
    def setUp(self):
        ot.clear()
        self.addCleanup(ot.clear)
        self.tmp = tempfile.mkdtemp()

    def test_the_run_names_what_it_produces(self):
        ot.set_run_role("shot start frame", declared=True, hook="30")
        p = _file(self.tmp)
        self.assertEqual(ot.role_for(p), "shot start frame")
        self.assertTrue(ot.meta_for(p)["declared"])

    def test_a_role_is_frozen_when_the_file_first_appears(self):
        """A chained turn moves on; the file produced by stage one must not."""
        ot.set_run_role("stage one")
        first = _file(self.tmp, "a.png")
        self.assertEqual(ot.role_for(first), "stage one")
        ot.set_run_role("stage three")
        second = _file(self.tmp, "b.png")
        self.assertEqual(ot.role_for(first), "stage one")
        self.assertEqual(ot.role_for(second), "stage three")

    def test_tagging_one_file_beats_the_run_default(self):
        ot.set_run_role("the batch")
        p = _file(self.tmp)
        ot.tag(p, "the hero sheet")
        self.assertEqual(ot.role_for(p), "the hero sheet")

    def test_no_role_no_record(self):
        p = _file(self.tmp)
        self.assertEqual(ot.role_for(p), "")
        self.assertFalse(ot.sidecar_path(p).exists(),
                         "an empty role must not litter the output dir")


class SidecarTest(unittest.TestCase):
    def setUp(self):
        ot.clear()
        self.addCleanup(ot.clear)
        self.tmp = tempfile.mkdtemp()

    def test_resolving_a_role_writes_the_record_beside_the_file(self):
        ot.set_run_role("hero character sheet", hook="30")
        p = _file(self.tmp)
        ot.role_for(p)
        body = json.loads(ot.sidecar_path(p).read_text(encoding="utf-8"))
        self.assertEqual(body["role"], "hero character sheet")
        self.assertEqual(body["hook"], "30")
        self.assertIn("when", body)

    def test_it_is_read_back_after_the_turn_that_wrote_it(self):
        p = _file(self.tmp)
        ot.write_sidecar(p, "the alley at night")
        ot.clear()                       # a new turn, or a new session entirely
        self.assertEqual(ot.role_of_file(p), "the alley at night")

    def test_a_file_with_no_record_says_nothing(self):
        self.assertEqual(ot.role_of_file(_file(self.tmp)), "")
        self.assertEqual(ot.read_sidecar(Path(self.tmp) / "nope.png"), {})

    def test_a_canvas_node_resolves_its_file_through_the_input_dir(self):
        p = _file(self.tmp, "staged.png")
        ot.write_sidecar(p, "shot 2 start frame")
        ot.clear()
        with mock.patch.object(ot, "input_dir", return_value=Path(self.tmp)):
            self.assertEqual(ot.role_of_canvas_file("staged.png"), "shot 2 start frame")
            # ComfyUI's "name [input]" suffix and subfolders both resolve.
            self.assertEqual(ot.role_of_canvas_file("staged.png [input]"),
                             "shot 2 start frame")

    def test_an_absolute_path_needs_no_comfyui_at_all(self):
        p = _file(self.tmp, "abs.png")
        ot.write_sidecar(p, "the reference")
        ot.clear()
        with mock.patch.object(ot, "input_dir", return_value=None):
            self.assertEqual(ot.role_of_canvas_file(str(p)), "the reference")


class AnchorRenderingTest(unittest.TestCase):
    """The payoff: the next turn reads the role instead of describing pixels."""

    def setUp(self):
        ot.clear()
        self.addCleanup(ot.clear)
        self.tmp = tempfile.mkdtemp()

    def test_the_recorded_role_reaches_the_hook_block(self):
        p = _file(self.tmp, "gen_04.png")
        ot.write_sidecar(p, "shot 4 start frame")
        ot.clear()
        with mock.patch.object(ot, "input_dir", return_value=Path(self.tmp)):
            line = _render_anchor("42", "LoadImage", {"image": "gen_04.png"})
        self.assertIn('← this is: "shot 4 start frame"', line)

    def test_a_ref_note_on_the_wire_still_wins(self):
        """The user's instruction for THIS wire outranks what the file says it is."""
        p = _file(self.tmp, "gen_05.png")
        ot.write_sidecar(p, "shot 5 start frame")
        ot.clear()
        with mock.patch.object(ot, "input_dir", return_value=Path(self.tmp)):
            line = _render_anchor("42", "LoadImage", {"image": "gen_05.png"},
                                  role="take only the colour grade")
        self.assertIn("USE THIS FOR", line)
        self.assertNotIn("shot 5 start frame", line)

    def test_a_node_title_is_used_when_there_is_no_record(self):
        line = _render_anchor("42", "LoadImage", {"image": "x.png"},
                              title="agentY · hero character sheet")
        self.assertIn('← this is: "hero character sheet"', line)

    def test_a_title_that_only_repeats_the_filename_says_nothing(self):
        line = _render_anchor("42", "LoadImage", {"image": "x.png"},
                              title="agentY · x.png")
        self.assertNotIn("this is:", line)

    def test_an_untitled_unrecorded_anchor_is_unchanged(self):
        line = _render_anchor("42", "LoadImage", {"image": "x.png"})
        self.assertEqual(line, "node 42 (LoadImage) inputs[image='x.png']")


class GalleryTest(unittest.TestCase):
    """Videos belong in it, and a hook turn's outputs are no longer nameless."""

    def setUp(self):
        ot.clear()
        self.addCleanup(ot.clear)

    @staticmethod
    def _pipe(paths):
        from types import SimpleNamespace
        ns = SimpleNamespace(
            _session=SimpleNamespace(current_output_paths=list(paths),
                                     generated_images=[], chat_summaries=[]),
            _last_brainbriefing_json="{}",
        )
        ns._caption_from_brief = Pipeline._caption_from_brief
        ns._register_generated_images = Pipeline._register_generated_images.__get__(ns)
        return ns

    def test_a_video_gets_a_gallery_number(self):
        pipe = self._pipe([r"C:\out\clip_01.mp4", r"C:\out\frame_01.png"])
        pipe._register_generated_images(None)
        got = [(g.index, g.path) for g in pipe._session.generated_images]
        self.assertEqual(len(got), 2, "'the second video' has to resolve to something")

    def test_the_caption_is_what_the_run_was_for(self):
        ot.set_run_role("shot 3 start frame")
        pipe = self._pipe([r"C:\out\a.png"])
        pipe._register_generated_images(None)   # no briefing: a canvas-hook turn
        self.assertEqual(pipe._session.generated_images[0].caption, "shot 3 start frame")

    def test_the_briefing_still_captions_a_template_run(self):
        pipe = self._pipe([r"C:\out\a.png"])
        pipe._register_generated_images(json.dumps({"prompt": {"positive": "a lighthouse"}}))
        self.assertEqual(pipe._session.generated_images[0].caption, "a lighthouse")


if __name__ == "__main__":
    unittest.main()
