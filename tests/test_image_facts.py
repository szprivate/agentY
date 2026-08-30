"""Measured technical facts about an output, and what they must never claim.

QA already measures what a vision model guesses badly — dimensions, duration —
and hands the numbers over as authoritative. These are the same trade for the
complaints people actually make: *soft, grainy, blown out*.

Everything here is built from images whose answer is known by construction, so a
number is checked against a fact rather than against yesterday's output of the
same code. The bands come from 87 real renders and reference photos (score 10 to
457, quartiles 40 / 81 / 136), with a deliberate Gaussian blur landing near 1 —
below everything real, which is what makes the bottom band mean something.

    python -m unittest discover -s tests
"""

import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

from src.utils.image_facts import (SHARP_BANDS, image_quality, measure,
                                   render_quality)


def _detailed(seed=0, size=(700, 500)):
    """A picture-like image: shapes and a gradient, not blurred white noise.

    The fixture matters. Random noise has no edges to lose, so adding grain to it
    raises the score the way it never does on a photograph — measuring that would
    be measuring the fixture.
    """
    from PIL import ImageDraw
    w, h = size
    ramp = np.linspace(20, 220, w, dtype="float32")
    arr = np.repeat(ramp[None, :], h, axis=0)
    im = Image.fromarray(np.stack([arr] * 3, axis=-1).astype("uint8"))
    draw = ImageDraw.Draw(im)
    rng = np.random.default_rng(seed)
    for _ in range(40):
        x, y = rng.integers(0, w - 60), rng.integers(0, h - 60)
        s = int(rng.integers(18, 55))
        shade = int(rng.integers(0, 255))
        if rng.random() < 0.5:
            draw.rectangle([x, y, x + s, y + s], fill=(shade, shade, shade))
        else:
            draw.ellipse([x, y, x + s, y + s], fill=(shade, 255 - shade, shade))
    return im


def _flat(value=128, size=(700, 500)):
    return Image.fromarray(np.full((size[1], size[0], 3), value, dtype="uint8"))


class _Tmp(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._dir = tempfile.TemporaryDirectory()
        cls.root = Path(cls._dir.name)

    @classmethod
    def tearDownClass(cls):
        cls._dir.cleanup()

    def save(self, im, name):
        p = self.root / name
        im.save(p)
        return str(p)


class SharpnessTest(_Tmp):

    def test_blurring_an_image_lowers_its_score(self):
        sharp = image_quality(self.save(_detailed(), "sharp.png"))
        blurred = image_quality(
            self.save(_detailed().filter(ImageFilter.GaussianBlur(6)), "blur.png"))
        self.assertLess(blurred["sharpness"]["score"],
                        sharp["sharpness"]["score"] / 4,
                        "blur barely moved the number")

    def test_a_blurred_image_lands_in_a_soft_band(self):
        f = image_quality(
            self.save(_detailed().filter(ImageFilter.GaussianBlur(6)), "b2.png"))
        self.assertIn(f["sharpness"]["band"], ("very soft", "soft"))

    def test_grain_does_not_make_an_image_read_as_sharper(self):
        """The bug the denoised measure exists for.

        Raw variance-of-Laplacian counts grain as edge energy: adding sigma-18
        noise to a photo took it from 1394 to 4173, which would have told the
        judge a degraded frame was unusually crisp.
        """
        base = _detailed()
        arr = np.asarray(base).astype("float32")
        rng = np.random.default_rng(1)
        noisy = Image.fromarray(
            np.clip(arr + rng.normal(0, 18, arr.shape), 0, 255).astype("uint8"))
        clean_score = image_quality(self.save(base, "c.png"))["sharpness"]["score"]
        noisy_score = image_quality(self.save(noisy, "n.png"))["sharpness"]["score"]
        self.assertLess(noisy_score, clean_score * 1.25,
                        f"noise inflated sharpness ({clean_score} -> {noisy_score})")

    def test_the_same_picture_reads_the_same_at_two_resolutions(self):
        """A threshold tuned at one resolution has to survive another.

        The same source at 2400px and at 1200px — both above the working size, so
        both are resized down to it. Comparing a native small image against an
        upscaled copy would measure the resampling instead.
        """
        big = _detailed(size=(4000, 2800))
        small = big.resize((1100, 770), Image.LANCZOS)
        a = image_quality(self.save(big, "s1.png"))["sharpness"]["score"]
        b = image_quality(self.save(small, "s2.png"))["sharpness"]["score"]
        self.assertLess(abs(a - b) / max(a, b), 0.25,
                        f"resolution changed the reading ({a} vs {b})")

    def test_the_bands_are_ordered_and_cover_the_range(self):
        edges = [e for e, _ in SHARP_BANDS]
        self.assertEqual(edges, sorted(edges))
        self.assertEqual(len(set(n for _, n in SHARP_BANDS)), len(SHARP_BANDS))


class NoiseTest(_Tmp):

    def test_added_grain_raises_the_estimate(self):
        base = _flat()
        arr = np.asarray(base).astype("float32")
        rng = np.random.default_rng(2)
        noisy = Image.fromarray(
            np.clip(arr + rng.normal(0, 12, arr.shape), 0, 255).astype("uint8"))
        clean = image_quality(self.save(base, "flat.png"))["noise"]["sigma"]
        grainy = image_quality(self.save(noisy, "grain.png"))["noise"]["sigma"]
        self.assertLess(clean, 1.0)
        self.assertGreater(grainy, 5.0)
        self.assertIn(image_quality(self.save(noisy, "g2.png"))["noise"]["band"],
                      ("grainy", "very grainy"))


    def test_detail_is_not_counted_as_grain(self):
        """Edges are high-frequency too, which is the trap this estimate avoids.

        Taking the WORST tile would report a picture full of hard edges as noisy;
        grain lives in the flat areas, so the estimate reads a low percentile.
        """
        f = image_quality(self.save(_detailed(), "detail.png"))
        self.assertLess(f["noise"]["sigma"], 4.0,
                        "sharp detail was counted as grain")
        self.assertIn(f["noise"]["band"], ("clean", "light grain"))


    def test_one_textured_patch_does_not_make_the_frame_grainy(self):
        """Why the estimate reads a low percentile of tiles, not the worst one.

        Taking the worst tile reports every real photograph as noisy — measured
        across reference shots, the worst tile runs 10-20 where the quarter-point
        runs 0.4-5.6. A frame that is mostly clean with one busy area is clean.
        """
        arr = np.full((500, 700, 3), 128, dtype="uint8")
        rng = np.random.default_rng(7)
        patch = rng.normal(128, 40, (120, 160, 3))
        arr[20:140, 20:180] = np.clip(patch, 0, 255).astype("uint8")
        f = image_quality(self.save(Image.fromarray(arr), "patch.png"))
        self.assertLess(f["noise"]["sigma"], 2.0,
                        "one busy corner made the whole frame read as grainy")
        self.assertEqual(f["noise"]["band"], "clean")


class ExposureTest(_Tmp):

    def test_a_blown_image_reports_clipped_white(self):
        f = image_quality(self.save(_flat(255), "white.png"))
        self.assertGreater(f["exposure"]["clipped_white"], 0.9)
        self.assertEqual(f["exposure"]["clipped_black"], 0.0)

    def test_a_crushed_image_reports_clipped_black(self):
        f = image_quality(self.save(_flat(0), "black.png"))
        self.assertGreater(f["exposure"]["clipped_black"], 0.9)

    def test_a_mid_grey_image_clips_at_neither_end(self):
        f = image_quality(self.save(_flat(128), "mid.png"))
        self.assertEqual(f["exposure"]["clipped_white"], 0.0)
        self.assertEqual(f["exposure"]["clipped_black"], 0.0)
        self.assertAlmostEqual(f["exposure"]["mean"], 128.0, delta=1.0)


class RenderingTest(_Tmp):
    """What the judge is handed. Wording matters: it decides what gets failed."""

    def test_every_line_carries_a_number_and_a_band(self):
        lines = render_quality(image_quality(self.save(_detailed(), "r.png")))
        joined = " ".join(lines)
        self.assertIn("sharpness:", joined)
        self.assertIn("noise/grain:", joined)
        self.assertIn("exposure:", joined)

    def test_depth_of_field_is_only_claimed_when_somewhere_is_sharp(self):
        """A hazy render whose best region was also soft was described as having
        sharp parts — which would tell the judge the softness is deliberate."""
        soft = {"sharpness": {"score": 19.7, "band": "very soft",
                              "sharpest_region": 52.4, "sharpest_band": "soft"}}
        self.assertNotIn("depth of field", " ".join(render_quality(soft)))

        dof = {"sharpness": {"score": 40.0, "band": "soft",
                             "sharpest_region": 900.0, "sharpest_band": "very sharp"}}
        self.assertIn("depth of field", " ".join(render_quality(dof)))

    def test_clipping_is_only_mentioned_when_it_is_worth_mentioning(self):
        clean = {"exposure": {"mean": 120, "contrast": 50,
                              "clipped_black": 0.001, "clipped_white": 0.0}}
        self.assertNotIn("clipping", " ".join(render_quality(clean)))
        blown = {"exposure": {"mean": 200, "contrast": 40,
                              "clipped_black": 0.0, "clipped_white": 0.4}}
        self.assertIn("blown to white", " ".join(render_quality(blown)))

    def test_video_faults_are_named(self):
        lines = " ".join(render_quality(
            {"black_frames": 2, "frames_sampled": 9, "frozen_pairs": 3}))
        self.assertIn("black frames: 2 of 9", lines)
        self.assertIn("stall", lines)

    def test_nothing_measured_renders_nothing(self):
        self.assertEqual(render_quality({}), [])


class RobustnessTest(_Tmp):
    """A QA pass must survive whatever it is pointed at."""

    def test_a_missing_file_measures_to_nothing(self):
        self.assertEqual(image_quality(str(self.root / "nope.png")), {})

    def test_a_file_that_is_not_an_image_measures_to_nothing(self):
        p = self.root / "notes.txt"
        p.write_text("not a picture", encoding="utf-8")
        self.assertEqual(image_quality(str(p)), {})

    def test_a_one_pixel_image_does_not_raise(self):
        self.assertIsInstance(image_quality(self.save(_flat(size=(1, 1)), "px.png")),
                              dict)

    def test_a_greyscale_image_is_handled(self):
        im = Image.fromarray(np.full((80, 80), 100, dtype="uint8"), mode="L")
        self.assertIn("exposure", image_quality(self.save(im, "l.png")))

    def test_measure_routes_video_and_stills_apart(self):
        still = measure(self.save(_detailed(), "m.png"), is_video=False)
        self.assertIn("sharpness", still)
        self.assertEqual(measure(str(self.root / "none.mp4"), is_video=True), {})


class WiredIntoQaTest(_Tmp):
    """The facts are only useful if `measure_output` actually carries them."""

    def test_measure_output_includes_the_quality_facts(self):
        from src.utils.qa import measure_output
        facts = measure_output(self.save(_detailed(), "q.png"))
        self.assertIn("width", facts)                 # what it measured before
        self.assertIn("sharpness", facts)             # and what it measures now

    def test_the_facts_block_shows_them_to_the_judge(self):
        from src.utils.qa import measure_output, render_measurements
        block = render_measurements(measure_output(self.save(_detailed(), "q2.png")))
        self.assertIn("dimensions:", block)
        self.assertIn("sharpness:", block)

    def test_an_unmeasurable_file_still_produces_a_block(self):
        from src.utils.qa import measure_output, render_measurements
        p = self.root / "broken.png"
        p.write_bytes(b"not a png")
        block = render_measurements(measure_output(str(p)))
        self.assertIn("file:", block)                 # size still known
        self.assertNotIn("sharpness:", block)


if __name__ == "__main__":
    unittest.main()
