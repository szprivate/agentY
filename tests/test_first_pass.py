"""Two deterministic mistakes behind most first-attempt failures, benchmarked.

1. **Template pinning on ordinary words.** A template counted as "named by the
   user" when all its name's words appeared in the request — so "Stable
   Diffusion 1.5" pinned ``audio_stable_audio_example``, "canny" pinned the
   Z-Image and SD3.5 canny templates (forbidding an SD 1.5 ControlNet), and
   "using both models" pinned ``upscale_using_model`` to a Wan video request.
   Each became a hard "MUST use" constraint. Brands still pin ("nano banana",
   "kling"); job words don't; a template written out by name always does.

2. **The named model family wasn't held.** "SDXL" came back ``ready`` loading an
   SD 1.5 checkpoint. A loader whose file is plainly another family is now
   rebound to an installed file of the named one.

    python -m unittest discover -s tests
"""

import unittest
from types import SimpleNamespace

from src.pipeline import Pipeline
from src.utils import model_family as mf

CATALOG = ["audio_stable_audio_example", "canny_to_image_z_image_turbo", "sd3.5_large_canny_controlnet_example",
           "upscale_using_model", "basic_switch_node", "film_grain", "image_sdxl_simple", "sdxl_refiner_prompt_example",
           "flux_fill_inpaint_example", "flux_dev_full_text_to_image", "NanoBanana2_text_to_image",
           "api_kling2_6_t2v", "depth_to_image_z_image_turbo"]


def _match(text):
    fake = SimpleNamespace(_brand_index_cache=None, _verbose=False)
    fake._template_brand_index = lambda: Pipeline._template_brand_index(fake)
    import json
    from unittest import mock
    with mock.patch("agenty_core.tools.comfyui.get_workflow_catalog",
                    return_value=json.dumps({n: "" for n in CATALOG})):
        out = Pipeline._match_named_templates(fake, text)
    return out[1] if out else None


class PinTest(unittest.TestCase):

    def test_job_words_do_not_pin(self):
        for text in ("Build a basic Stable Diffusion 1.5 text-to-image workflow",
                     "use a canny ControlNet on SD 1.5",
                     "estimate its depth map and use a depth ControlNet",
                     "a Wan 2.2 text-to-video using both the high and low models",
                     "switch the sampler to dpmpp_2m",
                     "a cinematic photo, 35mm film grain"):
            self.assertIsNone(_match(text), text)

    def test_brands_still_pin(self):
        self.assertEqual(_match("make it with nano banana"), ["NanoBanana2_text_to_image"])
        self.assertEqual(_match("animate it with kling"), ["api_kling2_6_t2v"])
        self.assertEqual(_match("a FLUX.1 fill inpainting workflow"), ["flux_fill_inpaint_example"])

    def test_a_template_written_out_by_name_pins_whatever_its_words(self):
        self.assertEqual(_match("run upscale_using_model on it"), ["upscale_using_model"])
        self.assertEqual(_match("build image_sdxl_simple please"), ["image_sdxl_simple"])


def _options(cls, name):
    return ["SD15\\v1-5-pruned-emaonly-fp16.safetensors", "SD15\\sd-v1-5-inpainting.ckpt",
            "SDXL\\sd_xl_base_1.0.safetensors", "SDXL\\sd_xl_refiner_1.0.safetensors",
            "juggernaut_reborn.safetensors"]


class FamilyTest(unittest.TestCase):

    def test_the_family_a_request_names(self):
        self.assertEqual(mf.named("Build an SDXL text-to-image workflow"), ["SDXL"])
        self.assertEqual(mf.named("a basic Stable Diffusion 1.5 workflow"), ["SD 1.5"])
        self.assertEqual(mf.named("FLUX.1 dev"), ["FLUX.1"])
        self.assertEqual(mf.named("Flux 2 dev"), ["FLUX.2"])
        self.assertEqual(mf.named("Wan 2.2 A14B text-to-video"), ["Wan 2.2"])
        self.assertEqual(mf.named("a red fox in the snow"), [])

    def test_an_sd15_checkpoint_in_an_sdxl_request_is_rebound(self):
        wf = {"15": {"class_type": "CheckpointLoaderSimple",
                     "inputs": {"ckpt_name": "SD15\\v1-5-pruned-emaonly-fp16.safetensors"}}}
        wrong = mf.check(wf, ["SDXL"])
        self.assertEqual(wrong[0]["is"], "SD 1.5")
        swaps, left = mf.repair(wf, wrong, _options, "an SDXL text-to-image")
        self.assertEqual(left, [])
        self.assertEqual(wf["15"]["inputs"]["ckpt_name"], "SDXL\\sd_xl_base_1.0.safetensors",
                         "the base model, not the refiner")

    def test_the_inpainting_file_is_not_picked_for_a_plain_request(self):
        wf = {"4": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": "SDXL\\sd_xl_base_1.0.safetensors"}}}
        mf.repair(wf, mf.check(wf, ["SD 1.5"]), _options, "SD 1.5 canny")
        self.assertEqual(wf["4"]["inputs"]["ckpt_name"], "SD15\\v1-5-pruned-emaonly-fp16.safetensors")

    def test_an_unrecognisable_finetune_is_left_alone(self):
        """A false alarm would swap out the user's own model."""
        wf = {"4": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": "juggernaut_reborn.safetensors"}}}
        self.assertEqual(mf.check(wf, ["SDXL"]), [])

    def test_no_family_installed_is_reported_not_guessed(self):
        wf = {"4": {"class_type": "UNETLoader", "inputs": {"unet_name": "FLUX1\\flux1-dev.safetensors"}}}
        swaps, left = mf.repair(wf, mf.check(wf, ["Qwen Image"]), _options, "")
        self.assertEqual((swaps, len(left)), ([], 1))
        self.assertEqual(wf["4"]["inputs"]["unet_name"], "FLUX1\\flux1-dev.safetensors")


if __name__ == "__main__":
    unittest.main()
