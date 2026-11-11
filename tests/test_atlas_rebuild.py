import json
import os
import tempfile
import unittest

from src.utils import atlas_rebuild as ar


class AtlasRebuildTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.wd = os.path.realpath(os.path.join(self.tmp.name, "court"))
        os.makedirs(self.wd)
        with open(os.path.join(self.wd, "world.json"), "w") as f:
            json.dump({"name": "court", "frames": []}, f)

    def tearDown(self):
        self.tmp.cleanup()

    def test_takes_the_glb_or_the_folder(self):
        seen = []
        open(os.path.join(self.wd, "world.glb"), "wb").close()
        out = ar.rebuild({"world": os.path.join(self.wd, "world.glb")}, build=lambda wd: seen.append(wd) or {"glb": "x"})
        self.assertTrue(out["ok"])
        self.assertEqual(seen, [self.wd])
        ar.rebuild({"world": self.wd}, build=lambda wd: seen.append(wd) or {})
        self.assertEqual(seen[-1], self.wd)

    def test_refuses_a_folder_that_is_no_world(self):
        with self.assertRaises(ValueError):
            ar.rebuild({"world": self.tmp.name}, build=lambda wd: {})
        with self.assertRaises(ValueError):
            ar.rebuild("nope", build=lambda wd: {})


if __name__ == "__main__":
    unittest.main()
