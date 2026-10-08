"""Which loader node a finished file lands in, and what gets written into it.

Two shapes exist on a ComfyUI canvas and they take different things: a core
`LoadImage` names a file inside ComfyUI's input directory, a VHS `(Path)` loader
holds an absolute path and reads the original where it was written. The pairing
is the whole point — a node handed the other shape's value looks completely
normal on the canvas and fails only when it runs, which is late and confusing.

    python -m unittest discover -s tests
"""

import unittest
from unittest import mock

from src.utils.media_loaders import (CANDIDATES, candidates, takes_absolute_path,
                                     value_for)


class ChoiceTest(unittest.TestCase):

    def test_the_path_loader_is_preferred_when_it_exists(self):
        """No copy, and the node points at the file the run actually produced."""
        self.assertEqual(candidates("image")[0], "VHS_LoadImagePath")
        self.assertEqual(candidates("video")[0], "VHS_LoadVideoPath")

    def test_the_core_node_is_still_there_to_fall_back_to(self):
        """The frontend takes the first one REGISTERED, so a ComfyUI without the
        pack installed keeps exactly the behaviour it always had."""
        self.assertIn("LoadImage", candidates("image"))
        self.assertIn("VHS_LoadVideo", candidates("video"))

    def test_an_unknown_kind_offers_nothing_rather_than_guessing(self):
        self.assertEqual(candidates("audio"), [])
        self.assertEqual(candidates(""), [])

    def test_the_list_handed_out_is_a_copy(self):
        """A caller that sorts or trims its list must not edit everyone else's."""
        candidates("image").clear()
        self.assertTrue(candidates("image"))

    def test_the_server_sends_the_frontend_this_same_list(self):
        from unittest import mock
        from src.utils.agentY_server import _NODE_CANDIDATES
        with mock.patch("src.agent._load_settings", return_value={}):
            for kind in CANDIDATES:
                self.assertEqual(_NODE_CANDIDATES.get(kind, []), candidates(kind))
            self.assertEqual(_NODE_CANDIDATES.get("image", []), CANDIDATES["image"])
        # ... read when asked, so a loader chosen in Settings counts at once.
        with mock.patch("src.agent._load_settings",
                        return_value={"media_nodes": {"image_load": "bepic_imageLoad"}}):
            self.assertEqual(_NODE_CANDIDATES.get("image", [])[0], "bepic_imageLoad")


class ShapeTest(unittest.TestCase):

    def test_every_vhs_path_loader_reads_as_one(self):
        for name in ("VHS_LoadImagePath", "VHS_LoadVideoPath",
                     "VHS_LoadVideoFFmpegPath", "VHS_LoadImagesPath"):
            with self.subTest(name=name):
                self.assertTrue(takes_absolute_path(name))

    def test_the_name_loaders_do_not(self):
        for name in ("LoadImage", "LoadVideo", "VHS_LoadVideo", "VHS_LoadImages",
                     "AgentYImageCollector"):
            with self.subTest(name=name):
                self.assertFalse(takes_absolute_path(name))

    def test_nothing_at_all_is_a_name_loader(self):
        """Unknown is the safe way to be wrong: the staged copy always exists."""
        for value in (None, "", "   "):
            self.assertFalse(takes_absolute_path(value))


PRODUCED = "D:/out/refined_00007_.png"


class WritingAProducedFileIntoALoaderTest(unittest.TestCase):
    """A finished file written back into a loader the user wired.

    Whichever shape they wired, the value written has to be one that node can
    read. Handed the other shape it looks completely normal on the canvas and
    fails only when it runs — late, and nowhere near the decision that caused it.

    (This used to be driven through `iterate_step`, the hook loop that ran the
    graph one generation per turn. That purpose is retired; the pairing it relied
    on is still how every hook resolution fills an image input, so the subject is
    tested where the decision is actually made.)
    """

    def test_a_path_loader_is_given_the_path_and_nothing_is_staged(self):
        with mock.patch("agenty_core.tools.image_io.stage_image") as staged:
            self.assertEqual(value_for("VHS_LoadImagePath", PRODUCED), PRODUCED)
        staged.assert_not_called()

    def test_a_name_loader_is_given_the_staged_name(self):
        with mock.patch("agenty_core.tools.image_io.stage_image",
                        return_value={"name": "refined_00007_.png"}) as staged:
            self.assertEqual(value_for("LoadImage", PRODUCED), "refined_00007_.png")
        self.assertEqual(staged.call_count, 1, "it can only see the input directory")

    def test_a_bare_name_is_already_what_a_name_loader_wants(self):
        with mock.patch("agenty_core.tools.image_io.stage_image") as staged:
            self.assertEqual(value_for("LoadImage", "already_there.png"), "already_there.png")
        staged.assert_not_called()

    def test_a_file_that_cannot_be_staged_builds_no_node_at_all(self):
        """None means "do not wire this" — a loader pointing at nothing looks fine
        on the canvas and fails at run time."""
        with mock.patch("agenty_core.tools.image_io.stage_image", return_value={}):
            self.assertIsNone(value_for("LoadImage", PRODUCED))
        with mock.patch("agenty_core.tools.image_io.stage_image",
                        side_effect=OSError("input dir is read-only")):
            self.assertIsNone(value_for("LoadImage", PRODUCED))

    def test_the_hook_path_clones_the_user_own_loader_with_the_right_shape(self):
        """as_connection reuses the class the user wired, so the shape follows it."""
        from src.utils.canvas_hooks import as_connection
        for loader, expected, staging in (("VHS_LoadImagePath", PRODUCED, {}),
                                          ("LoadImage", "refined_00007_.png",
                                           {"name": "refined_00007_.png"})):
            with self.subTest(loader=loader):
                graph = {"7": {"class_type": loader, "inputs": {"image": "start.png"}},
                         "9": {"class_type": "KSampler", "inputs": {"image": ["7", 0]}}}
                with mock.patch("agenty_core.tools.image_io.stage_image",
                                return_value=staging):
                    link = as_connection(graph, PRODUCED, ["7", 0])
                self.assertIsNotNone(link, "the clone must be built")
                clone = graph[link[0]]
                self.assertEqual(clone["class_type"], loader)
                self.assertEqual(clone["inputs"]["image"], expected)
                self.assertEqual(graph["7"]["inputs"]["image"], "start.png",
                                 "the node the user wired is left alone")


if __name__ == "__main__":
    unittest.main()
