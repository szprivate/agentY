"""Staged files go into a loader of their own kind (briefing_scaffold.pair_inputs_with_loaders)."""
import unittest

from src.tools.briefing_scaffold import pair_inputs_with_loaders


class PairInputsTests(unittest.TestCase):
    LTX = [("5001", "LoadVideo"), ("9009", "LoadImage")]

    def test_a_look_frame_staged_first_still_goes_to_the_image_loader(self):
        got = pair_inputs_with_loaders([{"filename": "image_00009_.png", "role": "master_image"},
                                        {"filename": "canyon_previz.mp4", "role": "reference_image"}], self.LTX)
        self.assertEqual(got, [("9009", "LoadImage"), ("5001", "LoadVideo")])

    def test_a_picture_alone_never_lands_in_the_video_loader(self):
        got = pair_inputs_with_loaders([{"filename": "image_00009_.png"}], self.LTX)
        self.assertEqual(got, [("9009", "LoadImage")])

    def test_a_mask_goes_to_the_mask_loader(self):
        loaders = [("1", "LoadImageMask"), ("2", "LoadImage")]
        got = pair_inputs_with_loaders([{"filename": "a.png", "role": "master_image"},
                                        {"filename": "m.png", "role": "mask"}], loaders)
        self.assertEqual(got, [("2", "LoadImage"), ("1", "LoadImageMask")])

    def test_unknown_kinds_keep_the_old_order(self):
        got = pair_inputs_with_loaders([{"filename": "thing"}, {"filename": "other"}], self.LTX)
        self.assertEqual(got, self.LTX)

    def test_two_pictures_two_image_loaders_in_order(self):
        loaders = [("10", "LoadImage"), ("11", "LoadImage")]
        got = pair_inputs_with_loaders([{"filename": "a.png"}, {"filename": "b.jpg"}], loaders)
        self.assertEqual(got, loaders)


if __name__ == "__main__":
    unittest.main()
