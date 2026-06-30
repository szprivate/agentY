"""Unit tests for user-intent extraction: media / task / model family from
filenames, descriptions, and node signals; phrasing; determinism."""

import unittest

from src.tools.workflow_recipes import parser as P
from src.tools.workflow_recipes.intent import IntentClassifier

I = IntentClassifier()


def _graph(name, classes=None, description=None, media_type=None, widgets=None):
    classes = classes or ["KSampler"]
    nodes = [{"id": i, "type": c, "widgets_values": (widgets or {}).get(c, [])}
             for i, c in enumerate(classes)]
    g = P.parse_ui({"nodes": nodes, "links": []}, name, name + ".json", "custom")
    g.index_description = description
    g.media_type = media_type
    return g


class TestMediaTaskModel(unittest.TestCase):
    def test_wan_2_2_image_to_video(self):
        it = I.classify(_graph("video_wan2_2_14B_flf2v"))
        self.assertEqual(it.media, "video")
        self.assertEqual(it.task, "first_last_frame_to_video")
        self.assertIn("WAN 2.2", it.model_families)
        self.assertNotIn("WAN", it.model_families)   # generic suppressed by specific

    def test_text_to_image_flux(self):
        it = I.classify(_graph("text_to_image_flux_1_dev"))
        self.assertEqual(it.media, "image")
        self.assertEqual(it.task, "text_to_image")
        self.assertIn("Flux", it.model_families)

    def test_kling_i2v_from_description(self):
        g = _graph("api_kling_o3_i2v",
                   description="API image-to-video via Kling O3. 1 image -> 1 video output.")
        it = I.classify(g)
        self.assertEqual(it.media, "video")
        self.assertEqual(it.task, "image_to_video")
        self.assertIn("Kling", it.model_families)

    def test_model_family_from_loader_widget(self):
        # No model hint in the name; it comes from the model-loader widget value.
        g = _graph("my_graph", classes=["UNETLoader"],
                   widgets={"UNETLoader": ["wan2.2_i2v_high_noise.safetensors"]})
        it = I.classify(g)
        self.assertIn("WAN 2.2", it.model_families)

    def test_audio_and_3d(self):
        self.assertEqual(I.classify(_graph("text_to_audio_ace_step_1_5")).media, "audio")
        self.assertEqual(I.classify(_graph("api_meshy_image_to_model")).media, "3d")

    def test_captioning_is_text(self):
        self.assertEqual(I.classify(_graph("image_captioning_gemini")).media, "text")


class TestPhrasing(unittest.TestCase):
    def test_example_requests_match_user_phrasing(self):
        reqs = I.example_requests("video", "image_to_video", ["WAN 2.2"])
        self.assertIn("build a video workflow using WAN 2.2", reqs)

    def test_when_to_use_mentions_family(self):
        s = I.when_to_use("video", "text_to_video", ["WAN 2.2"])
        self.assertIn("WAN 2.2", s)
        self.assertTrue(s.endswith("."))


class TestDeterminism(unittest.TestCase):
    def test_stable(self):
        g = _graph("video_wan2_2_14B_flf2v")
        a = I.classify(g)
        b = I.classify(g)
        self.assertEqual(a.model_families, b.model_families)
        self.assertEqual((a.media, a.task), (b.media, b.task))


if __name__ == "__main__":
    unittest.main()
