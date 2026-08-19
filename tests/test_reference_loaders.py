"""A `#tag` resolved into a wire has to land in a loader that can READ the file.

A remembered tag stores a full path — that is the whole point of remembering it,
since references live in output folders and on network shares, not in ComfyUI's
input directory. But core's `LoadImage` names a file INSIDE the input directory
and cannot take a path at all. Writing one into it builds a node that looks
completely right on the canvas, keeps the path the user can read on the widget,
and fails only when the graph runs.

So the choice of loader is a question about the MACHINE, not about the workflow:
a VHS `(Path)` loader reads the file where it lies, and an install without the
pack has to have the file copied into the input directory first. Both answers are
correct; picking the wrong one is silent.

Self-contained: ComfyUI is stubbed, so what these check is the decision, not the
server.

    python -m unittest discover -s tests
"""

import unittest
from unittest import mock

from src.utils.canvas_hooks import as_connection
from src.utils.media_loaders import image_loader_node, installed, value_for

# A remembered reference, as tag_memory stores one: absolute, and nowhere near
# ComfyUI's input directory.
REF = "W:/0207_omaze/02_build/comfy/output/rnd/RND_0500/image_FF_Car_Social_01_v02.png"

CORE_ONLY = {"LoadImage", "KSampler", "SaveImage"}
WITH_VHS = CORE_ONLY | {"VHS_LoadImagePath", "VHS_LoadVideoPath"}


def _installed(classes):
    """Stub what this ComfyUI has registered."""
    return mock.patch("agenty_core.tools.comfyui.registered_node_classes",
                      return_value=set(classes))


# The required widgets each loader declares, copied from a live /object_info.
# VHS_LoadImagePath's two size widgets are the point: ComfyUI rejects a prompt
# with a required input missing rather than applying the declared default.
DEFAULTS = {
    "LoadImage": {},                                    # only `image`, and no default
    "VHS_LoadImagePath": {"custom_width": 0, "custom_height": 0},
}


def _defaults(table=None):
    return mock.patch("agenty_core.tools.comfyui.node_default_inputs",
                      side_effect=lambda cls: dict((table or DEFAULTS).get(cls, {})))


def _staging(name="image_FF_Car_Social_01_v02.png"):
    """Stub the upload into ComfyUI's input directory."""
    return mock.patch("agenty_core.tools.image_io.stage_image",
                      return_value={"name": name, "subfolder": "", "type": "input"})


def _node(cls, **inputs):
    return {"class_type": cls, "inputs": inputs}


class InstalledTests(unittest.TestCase):

    def test_the_path_loader_wins_when_the_pack_is_there(self):
        with _installed(WITH_VHS):
            self.assertEqual(installed("image"), "VHS_LoadImagePath")

    def test_core_is_the_answer_when_it_is_not(self):
        with _installed(CORE_ONLY):
            self.assertEqual(installed("image"), "LoadImage")

    def test_an_unreachable_comfyui_answers_nothing_rather_than_guessing(self):
        """Empty means UNKNOWN. Answering "LoadImage" here would be a claim that
        the pack is absent, which is exactly what could not be established."""
        with _installed(set()):
            self.assertEqual(installed("image"), "")


class NodeDefaultsTests(unittest.TestCase):
    """Reading a node's required widgets straight off ``/object_info``.

    The shapes below are what ComfyUI really sends, trimmed: a widget is
    ``[type, {options}]``, and a combo declares its choices IN PLACE of a type.
    """

    OBJECT_INFO = {
        "VHS_LoadImagePath": {"input": {"required": {
            "image": ["STRING", {"placeholder": "X://insert/path/here.png"}],
            "custom_width": ["INT", {"default": 0, "min": 0, "max": 8192}],
            "custom_height": ["INT", {"default": 0, "min": 0, "max": 8192}],
        }, "optional": {"vae": ["VAE"]}}},
        "LoadImage": {"input": {"required": {
            "image": [["a.png", "b.png"], {"image_upload": True}],
        }}},
        "KSampler": {"input": {"required": {
            "model": ["MODEL"],
            "steps": ["INT", {"default": 20}],
            "sampler_name": [["euler", "dpmpp_2m"], {}],
            "scheduler": [["normal", "karras"], {"default": "karras"}],
        }}},
    }

    def _read(self, cls):
        from agenty_core.tools import comfyui
        with mock.patch.object(comfyui, "_get_object_info",
                               return_value=self.OBJECT_INFO):
            return comfyui.node_default_inputs(cls)

    def test_declared_defaults_are_taken(self):
        self.assertEqual(self._read("VHS_LoadImagePath"),
                         {"custom_width": 0, "custom_height": 0})

    def test_a_widget_with_no_default_is_left_for_the_caller(self):
        """The path widget declares none, and inventing one would put a
        placeholder where the caller is about to write the real file."""
        self.assertNotIn("image", self._read("VHS_LoadImagePath"))

    def test_a_combo_defaults_to_its_first_choice(self):
        self.assertEqual(self._read("LoadImage"), {"image": "a.png"})
        self.assertEqual(self._read("KSampler")["sampler_name"], "euler")

    def test_a_combo_with_a_stated_default_uses_that_instead(self):
        self.assertEqual(self._read("KSampler")["scheduler"], "karras")

    def test_inputs_that_take_a_wire_are_left_out(self):
        """A literal in a MODEL input is not a default, it is a type error."""
        self.assertNotIn("model", self._read("KSampler"))

    def test_optional_inputs_are_not_defaults_to_write(self):
        self.assertNotIn("vae", self._read("VHS_LoadImagePath"))

    def test_a_class_this_comfyui_does_not_have_reads_as_nothing(self):
        self.assertEqual(self._read("SomePackNobodyInstalled"), {})

    def test_an_unreachable_comfyui_reads_as_nothing(self):
        from agenty_core.tools import comfyui
        with mock.patch.object(comfyui, "_get_object_info",
                               side_effect=OSError("down")):
            self.assertEqual(comfyui.node_default_inputs("LoadImage"), {})


class ValueForTests(unittest.TestCase):

    def test_a_path_loader_is_handed_the_path_untouched(self):
        with _staging() as staged:
            self.assertEqual(value_for("VHS_LoadImagePath", REF), REF)
        staged.assert_not_called()

    def test_a_name_loader_stages_the_file_and_takes_the_copys_name(self):
        with _staging("image_FF_Car_Social_01_v02.png") as staged:
            self.assertEqual(value_for("LoadImage", REF),
                             "image_FF_Car_Social_01_v02.png")
        staged.assert_called_once()

    def test_a_bare_name_is_already_what_a_name_loader_wants(self):
        with _staging() as staged:
            self.assertEqual(value_for("LoadImage", "hero.png"), "hero.png")
        staged.assert_not_called()

    def test_a_failed_upload_is_none_rather_than_a_broken_value(self):
        with mock.patch("agenty_core.tools.image_io.stage_image",
                        return_value={"error": "File not found"}):
            self.assertIsNone(value_for("LoadImage", REF))

    def test_an_upload_that_raises_is_none_too(self):
        with mock.patch("agenty_core.tools.image_io.stage_image",
                        side_effect=OSError("comfyui is down")):
            self.assertIsNone(value_for("LoadImage", REF))


class ImageLoaderNodeTests(unittest.TestCase):

    def test_with_vhs_the_node_reads_the_file_where_it_lies(self):
        with _installed(WITH_VHS), _defaults(), _staging() as staged:
            node = image_loader_node(REF)
        self.assertEqual(node["class_type"], "VHS_LoadImagePath")
        self.assertEqual(node["inputs"]["image"], REF)
        self.assertNotIn("upload", node["inputs"],
                         "that widget belongs to core's picker, not a path loader")
        staged.assert_not_called()

    def test_the_nodes_other_required_widgets_are_written_too(self):
        """ComfyUI rejects a prompt with a required input missing rather than
        applying the node's own default, so a loader carrying only its path is a
        loader that never runs."""
        with _installed(WITH_VHS), _defaults(), _staging():
            node = image_loader_node(REF)
        self.assertEqual(node["inputs"]["custom_width"], 0)
        self.assertEqual(node["inputs"]["custom_height"], 0)

    def test_without_vhs_the_file_is_staged_and_the_copy_named(self):
        with _installed(CORE_ONLY), _defaults(), _staging("hero_ref.png") as staged:
            node = image_loader_node(REF)
        self.assertEqual(node["class_type"], "LoadImage")
        self.assertEqual(node["inputs"]["image"], "hero_ref.png")
        self.assertEqual(node["inputs"]["upload"], "image")
        staged.assert_called_once()

    def test_a_bare_name_needs_no_comfyui_at_all(self):
        """The value already names a file in the input directory. Asking what is
        installed would make a working case depend on a reachable server."""
        with _installed(set()) as asked, _staging() as staged:
            node = image_loader_node("hero.png")
        self.assertEqual(node, {"class_type": "LoadImage",
                                "inputs": {"image": "hero.png", "upload": "image"}})
        asked.assert_not_called()
        staged.assert_not_called()

    def test_nothing_is_built_when_the_file_cannot_be_reached_either_way(self):
        with _installed(set()):
            self.assertIsNone(image_loader_node(REF))

    def test_an_empty_path_builds_nothing(self):
        self.assertIsNone(image_loader_node(""))
        self.assertIsNone(image_loader_node("   "))


class AsConnectionLoaderTests(unittest.TestCase):
    """The path the user actually hit: a hook directive naming a remembered tag."""

    def _canvas(self):
        return {"9": _node("KSampler", seed=1),
                "43": _node("KlingVideo", first_frame=["9", 0])}

    def test_a_remembered_path_gets_a_path_loader_not_a_loadimage(self):
        g = self._canvas()
        with _installed(WITH_VHS), _defaults(), _staging():
            link = as_connection(g, REF, None)
        self.assertIsNotNone(link)
        node = g[link[0]]
        self.assertEqual(node["class_type"], "VHS_LoadImagePath")
        self.assertEqual(node["inputs"]["image"], REF)

    def test_without_the_pack_it_is_staged_into_the_input_directory(self):
        g = self._canvas()
        with _installed(CORE_ONLY), _defaults(), _staging("hero_ref.png"):
            link = as_connection(g, REF, None)
        node = g[link[0]]
        self.assertEqual(node["class_type"], "LoadImage")
        self.assertEqual(node["inputs"]["image"], "hero_ref.png",
                         "a LoadImage holding a path is the bug being fixed")

    def test_an_unusable_reference_leaves_the_input_alone(self):
        """`as_connection` returning None is the contract for "cannot be
        expressed as a wire" — the caller keeps the existing link rather than
        writing a literal, and no half-built loader is left on the canvas."""
        g = self._canvas()
        before = set(g)
        with _installed(set()):
            self.assertIsNone(as_connection(g, REF, None))
        self.assertEqual(set(g), before, "a node was added that cannot load")


class CloneLoaderTests(unittest.TestCase):
    """Cloning the user's own loader inherits what that loader can be handed."""

    def test_cloning_a_name_loader_stages_rather_than_writing_the_path(self):
        g = {"12": _node("LoadImage", image="already_there.png", upload="image"),
             "43": _node("KlingVideo", first_frame=["12", 0])}
        with _installed(CORE_ONLY), _staging("hero_ref.png") as staged:
            link = as_connection(g, REF, ["12", 0])
        clone = g[link[0]]
        self.assertEqual(clone["class_type"], "LoadImage", "kept the user's loader")
        self.assertEqual(clone["inputs"]["image"], "hero_ref.png")
        staged.assert_called_once()
        self.assertEqual(g["12"]["inputs"]["image"], "already_there.png",
                         "cloning must leave the original alone")

    def test_cloning_a_path_loader_keeps_the_path(self):
        g = {"12": _node("VHS_LoadImagePath", image="W:/refs/other.png"),
             "43": _node("KlingVideo", first_frame=["12", 0])}
        with _installed(WITH_VHS), _staging() as staged:
            link = as_connection(g, REF, ["12", 0])
        clone = g[link[0]]
        self.assertEqual(clone["class_type"], "VHS_LoadImagePath")
        self.assertEqual(clone["inputs"]["image"], REF)
        staged.assert_not_called()

    def test_a_clone_that_cannot_load_falls_back_to_a_loader_that_can(self):
        """Keeping the user's loader class is a preference, not a requirement.
        When the pack IS installed, a reference core cannot take is no reason to
        give up on the reference."""
        g = {"12": _node("LoadImage", image="already_there.png", upload="image"),
             "43": _node("KlingVideo", first_frame=["12", 0])}
        with _installed(WITH_VHS), _defaults(), mock.patch(
                "agenty_core.tools.image_io.stage_image",
                return_value={"error": "nope"}):
            link = as_connection(g, REF, ["12", 0])
        self.assertIsNotNone(link)
        self.assertEqual(g[link[0]]["class_type"], "VHS_LoadImagePath")
        self.assertEqual(g[link[0]]["inputs"]["image"], REF)

    def test_a_reference_no_loader_here_can_read_adds_nothing(self):
        g = {"12": _node("LoadImage", image="already_there.png", upload="image"),
             "43": _node("KlingVideo", first_frame=["12", 0])}
        before = set(g)
        with _installed(CORE_ONLY), mock.patch(
                "agenty_core.tools.image_io.stage_image",
                return_value={"error": "nope"}):
            self.assertIsNone(as_connection(g, REF, ["12", 0]))
        self.assertEqual(set(g), before, "a node was added that cannot load")


if __name__ == "__main__":
    unittest.main()
