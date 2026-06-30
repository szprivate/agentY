"""Unit tests for the canonical taxonomy classifier."""

import unittest

from src.tools.workflow_recipes import parser as P
from src.tools.workflow_recipes.taxonomy import TaxonomyClassifier

T = TaxonomyClassifier()


def _graph(name, classes, category=None, b_in=None, b_out=None, api=False):
    nodes = [{"id": i, "type": c, "widgets_values": []} for i, c in enumerate(classes)]
    g = P.parse_ui({"nodes": nodes, "links": []}, name, name + ".json", "official")
    g.category = category
    g.boundary_inputs = [{"name": t, "data_type": t} for t in (b_in or [])]
    g.boundary_outputs = [{"name": t, "data_type": t} for t in (b_out or [])]
    for n in g.nodes.values():
        n.is_api = api
    return g


class TestTaxonomy(unittest.TestCase):
    def test_api_partner_nodes_sub_split(self):
        # Partner workflows are bucketed under "API / Partner Nodes - <task>".
        g = _graph("api_kling_o3_i2v", ["KlingOmniProImageToVideoNode"],
                   b_in=["IMAGE"], b_out=["VIDEO"], api=True)
        self.assertEqual(T.classify(g), "API / Partner Nodes - Image to Video")

    def test_api_by_name_prefix(self):
        g = _graph("api_some_partner", ["SomeNode"])
        self.assertTrue(T.classify(g).startswith("API / Partner Nodes - "))

    def test_text_to_image(self):
        g = _graph("text_to_image_flux_1_dev", ["UNETLoader", "KSampler", "VAEDecode"],
                   category="Image generation and editing", b_in=["STRING"], b_out=["IMAGE"])
        self.assertEqual(T.classify(g), "Text to Image")

    def test_image_tools_from_catalog(self):
        # Index category is authoritative; a GLSL filter with no LoadImage node.
        g = _graph("brightness_and_contrast", ["GLSLShader"],
                   category="Image Tools", b_in=["IMAGE"], b_out=["IMAGE"])
        self.assertEqual(T.classify(g), "Image Tools")

    def test_image_to_video(self):
        g = _graph("image_to_video_wan_2_2", ["UNETLoader", "KSampler", "WanImageToVideo"],
                   b_in=["IMAGE"], b_out=["VIDEO"])
        self.assertEqual(T.classify(g), "Image to Video")

    def test_video_to_video(self):
        g = _graph("video_wan_vace_14B_v2v", ["UNETLoader", "KSampler", "VHS_VideoCombine"],
                   b_in=["VIDEO"], b_out=["VIDEO"])
        self.assertEqual(T.classify(g), "Video to Video")

    def test_partner_text_to_video_with_optional_image(self):
        # Veo-style: a partner video node with an OPTIONAL image input must stay
        # Text to Video, not be mistaken for Image to Video.
        veo = _graph("api_veo3", ["Veo3VideoGenerationNode", "LoadImage", "SaveVideo"],
                     b_out=["VIDEO"], api=True)
        self.assertEqual(T.classify(veo), "API / Partner Nodes - Text to Video")
        # A partner node whose CLASS name declares the task is honored.
        i2v = _graph("api_wan2_6_i2v", ["WanImageToVideoApi", "LoadImage"],
                     b_out=["VIDEO"], api=True)
        self.assertEqual(T.classify(i2v), "API / Partner Nodes - Image to Video")
        t2v = _graph("api_wan2_6_t2v", ["WanTextToVideoApi"], b_out=["VIDEO"], api=True)
        self.assertEqual(T.classify(t2v), "API / Partner Nodes - Text to Video")

    def test_local_image_input_still_i2v(self):
        # A LOCAL video graph with an image input is still Image to Video.
        g = _graph("video_wan2_2_14B_fun_camera",
                   ["UNETLoader", "KSampler", "LoadImage", "VHS_VideoCombine"],
                   b_out=["VIDEO"])
        self.assertEqual(T.classify(g), "Image to Video")

    def test_first_last_frame(self):
        g = _graph("video_wan2_2_14B_flf2v", ["UNETLoader", "KSampler", "VHS_VideoCombine"],
                   b_in=["IMAGE"], b_out=["VIDEO"])
        self.assertEqual(T.classify(g), "First / Last Frame to Video")

    def test_controlnet_generation_vs_depth_estimation(self):
        # "depth_to_image" is controlnet generation; "image_to_depth" is estimation.
        gen = _graph("depth_to_image_z_image_turbo", ["UNETLoader", "KSampler", "VAEDecode"],
                     category="Image generation and editing", b_in=["IMAGE"], b_out=["IMAGE"])
        est = _graph("image_to_depth_map_lotus", ["UNETLoader", "VAEDecode"],
                     category="Image generation and editing", b_in=["IMAGE"], b_out=["IMAGE"])
        self.assertEqual(T.classify(gen), "Image Edit with ControlNet")
        self.assertEqual(T.classify(est), "Preprocessors / Estimation")

    def test_inpaint(self):
        g = _graph("image_inpainting_qwen_image", ["UNETLoader", "KSampler", "VAEDecode"],
                   category="Image generation and editing", b_in=["IMAGE", "MASK"], b_out=["IMAGE"])
        self.assertEqual(T.classify(g), "Inpaint / Outpaint")

    def test_preprocessor_from_catalog(self):
        g = _graph("image_segmentation_sam3", ["SAM3"],
                   category="Conditioning & Preprocessors", b_in=["IMAGE"], b_out=["MASK"])
        self.assertEqual(T.classify(g), "Preprocessors / Estimation")

    def test_audio_and_3d(self):
        a = _graph("text_to_audio_ace_step_1_5", ["KSampler"], category="Audio", b_out=["AUDIO"])
        d = _graph("api_meshy_text_to_model", ["MeshyNode"], b_out=["MESH"], api=True)
        self.assertEqual(T.classify(a), "Audio")
        self.assertEqual(T.classify(d), "API / Partner Nodes - 3D")  # api, sub-split

    def test_text_tools(self):
        g = _graph("image_captioning_gemini", ["GeminiNode"], api=True)
        self.assertEqual(T.classify(g), "Text Tools")   # text task beats api


if __name__ == "__main__":
    unittest.main()
