"""Two ways a TRELLIS.2 build broke, both in what the agent read from the template.

1. Widget values. UnwrapMesh's template declares only its wired `resolution`
   widget, so the converter fell back to schema order and tried two readings.
   The "wired widgets keep their slot" reading also gave the wired `mesh` socket
   a slot, lost on fit, and the shifted reading shipped padding=2048 (clamped to
   16) and weld_distance=1: every vertex welded into one, and ApplyTextureToMesh
   died on "zero-size array to reduction operation minimum".

2. Model folders. The agent named Trellis2ShapeStage as the loading node, which
   no map knows, so trellis_2_int8_convrot.safetensors landed in checkpoints
   although the template says diffusion_models — twice, in its loader's model
   list and in its "Model Storage Location" note.

    python -m unittest tests.test_template_model_facts
"""

import unittest
from unittest import mock

from agenty_core.tools import comfyui as C
from agenty_core.tools import huggingface as H
from agenty_core.utils import template_models as T

UNWRAP_SCHEMA = {"required": {
    "mesh": ["MESH", {}],
    "segmenter": ["COMBO", {"options": ["pec", "adaptive"], "default": "pec"}],
    "resolution": ["INT", {"default": 1024, "min": 0, "max": 8192}],
    "padding": ["INT", {"default": 1, "min": 0, "max": 16}],
    "weld_distance": ["FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0}],
}}

GRAPH = {
    "nodes": [
        {"id": 1, "type": "Src", "inputs": [], "outputs": [], "widgets_values": []},
        {"id": 2, "type": "PrimitiveInt", "inputs": [], "widgets_values": [2048, "fixed"]},
        {"id": 196, "type": "UnwrapMesh", "widgets_values": ["pec", 2048, 1, 0.0002],
         "inputs": [{"name": "mesh", "type": "MESH", "link": 10},
                    {"name": "resolution", "type": "INT", "link": 11,
                     "widget": {"name": "resolution"}}]},
    ],
    "links": [[10, 1, 0, 196, 0, "MESH"], [11, 2, 0, 196, 1, "INT"]],
}

NOTE = """## Model Storage Location

```
📂 ComfyUI/
├── 📂 models/
│   ├── 📂 vae/
│   │   └── some_vae.safetensors
│   └── 📂 diffusion_models/
│       ├── note_only_model.safetensors
│       └── declared_model.safetensors
```
"""


class WiredWidgetsKeepTheirSlot(unittest.TestCase):

    def test_unwrap_mesh_values_stay_in_place(self):
        with mock.patch.object(C, "_get_object_info",
                               return_value={"UnwrapMesh": {"input": UNWRAP_SCHEMA}}):
            node = C._convert_graph_to_api(GRAPH)["196"]
        self.assertEqual(node["inputs"]["resolution"], ["2", 0])
        self.assertEqual(node["inputs"]["segmenter"], "pec")
        self.assertEqual(node["inputs"]["padding"], 1)
        self.assertEqual(node["inputs"]["weld_distance"], 0.0002)
        self.assertNotIn("unmapped_widgets", node.get("_meta", {}))


class TheTemplateSaysWhereAModelGoes(unittest.TestCase):

    def test_loader_list_and_note_are_both_read(self):
        wf = {"nodes": [
            {"type": "UNETLoader", "properties": {"models": [
                {"name": "declared_model.safetensors", "directory": "diffusion_models",
                 "url": "https://huggingface.co/Org/Repo/resolve/main/diffusion_models/declared_model.safetensors"}]}},
            {"type": "MarkdownNote", "widgets_values": [NOTE]},
        ]}
        found = T.models_in_workflow(wf)
        self.assertEqual(found["declared_model.safetensors"]["directory"], "diffusion_models")
        self.assertTrue(found["declared_model.safetensors"]["url"])
        self.assertEqual(found["note_only_model.safetensors"]["directory"], "diffusion_models")
        self.assertEqual(found["some_vae.safetensors"]["directory"], "vae")

    def test_the_real_trellis_template_is_indexed(self):
        hit = T.template_model_index().get("trellis_2_int8_convrot.safetensors")
        self.assertIsNotNone(hit)
        self.assertEqual(hit["directory"], "diffusion_models")

    def test_template_folder_beats_an_unknown_node(self):
        paths = {"diffusion_models": ["X:/models/diffusion_models", "X:/models/unet"],
                 "checkpoints": ["X:/models/checkpoints"]}
        with mock.patch.object(H, "_folder_paths", return_value=paths), \
             mock.patch.object(H, "_models_base_dir", return_value=__import__("pathlib").Path("X:/models")):
            d, src = H._resolve_download_dir("Trellis2ShapeStage", "", "trellis_2_int8_convrot.safetensors")
            self.assertEqual(d.as_posix(), "X:/models/diffusion_models")
            self.assertTrue(src.startswith("template"))
            # a file no template names still reads the HF repo path
            d, _ = H._resolve_download_dir("Trellis2ShapeStage", "", "unheard_of.safetensors",
                                           "diffusion_models")
            self.assertEqual(d.as_posix(), "X:/models/diffusion_models")


if __name__ == "__main__":
    unittest.main()
