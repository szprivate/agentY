"""Atlas Chisel pictures asked for by the viewer's 3D space: validated, made in the background, delivered."""
import json
import os
import tempfile
import time
import unittest

from src.utils.atlas_chisel_jobs import AtlasChiselJobs

CAM = {"viewer": {"camera_world": [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1], "fov": 50, "aspect": 16 / 9,
                  "gated": True}}


def wait(jobs, jid, timeout=5.0):
    t0 = time.time()
    while time.time() - t0 < timeout:
        j = jobs.get(jid)
        if j["state"] != "running":
            return jobs.public(j)
        time.sleep(0.02)
    raise AssertionError("job did not finish")


class ChiselJobTests(unittest.TestCase):
    def test_bad_requests(self):
        jobs = AtlasChiselJobs(run=lambda **kw: "{}")
        for body in (None, {}, {"world": "w", "camera": CAM}, {"world": "w", "prompt": "x"},
                     {"world": "w", "prompt": "x", "camera": {"camera_world": []}},
                     {"world": "w", "prompt": "x", "camera": CAM, "image_path": "/no/such.png"},
                     {"world": "w", "prompt": "x", "camera": CAM, "seed": "abc"}):
            with self.assertRaises(ValueError, msg=body):
                jobs.start(body)

    def test_mask_needs_its_picture(self):
        with tempfile.TemporaryDirectory() as d:
            m = os.path.join(d, "mask.png")
            open(m, "wb").write(b"x")
            with self.assertRaises(ValueError):
                AtlasChiselJobs(run=lambda **kw: "{}").start({"world": "w", "prompt": "x", "camera": CAM,
                                                               "mask_path": m})

    def test_masked_edit_is_run_with_the_viewer_camera(self):
        with tempfile.TemporaryDirectory() as d:
            img, mask = os.path.join(d, "view.png"), os.path.join(d, "mask.png")
            for p in (img, mask):
                open(p, "wb").write(b"x")
            seen = {}

            def run(**kw):
                seen.update(kw)
                return json.dumps({"status": "ok", "path": "C:/w/chisel/chisel_001.png", "prompt_used": "a fountain"})
            jobs = AtlasChiselJobs(run=run)
            job = jobs.start({"world": "C:/w", "camera": CAM, "prompt": "a fountain here", "image_path": img,
                              "mask_path": mask, "seed": "7", "enhance_prompt": False})
            self.assertEqual(job["mode"], "edit")
            done = wait(jobs, job["id"])
        self.assertEqual((done["state"], done["path"], done["prompt_used"]),
                         ("done", "C:/w/chisel/chisel_001.png", "a fountain"))
        self.assertEqual((seen["world"], seen["camera"], seen["image_path"], seen["mask_path"], seen["seed"],
                          seen["enhance_prompt"], seen["context"], seen["style_image"]),
                         ("C:/w", CAM, img, mask, 7, False, None, ""))

    def test_on_the_scene_depth_without_context_and_errors_are_kept(self):
        seen = {}
        jobs = AtlasChiselJobs(run=lambda **kw: seen.update(kw) or json.dumps(
            {"error": "out of credits", "error_kind": "billing", "operation_id": "op1"}))
        with tempfile.TemporaryDirectory() as d:
            depth = os.path.join(d, "chisel_depth.png")
            open(depth, "wb").write(b"x")
            for bad in ({}, {"depth_path": depth}, {"depth_path": depth, "z_min": 5, "z_max": 2},
                        {"depth_path": depth, "z_min": 0, "z_max": 2}):
                with self.assertRaises(ValueError, msg=bad):
                    jobs.start({"world": "C:/w", "camera": CAM, "prompt": "night", **bad})
            job = jobs.start({"world": "C:/w", "camera": CAM, "prompt": "night", "context": False,
                              "depth_path": depth, "z_min": "0.5", "z_max": 40, "voxel_size": 3})
            self.assertEqual(job["mode"], "depth")
            done = wait(jobs, job["id"])
        self.assertEqual((done["state"], done["message"], done["error_kind"], done["operation_id"]),
                         ("error", "out of credits", "billing", "op1"))
        self.assertEqual((seen["context"], seen["seed"], seen["depth_path"], seen["z_min"], seen["z_max"],
                          seen["voxel_size"], seen["image_path"]), ([], -1, depth, 0.5, 40.0, 0.4, ""))

if __name__ == "__main__":
    unittest.main()
