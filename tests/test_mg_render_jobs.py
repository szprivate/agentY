"""Motion-graphics renders asked for by the viewer's Motion panel, and the shipped example scenes."""
import json
import os
import time
import unittest

from src.utils.mg_render_jobs import MgRenderJobs

SKILL = os.path.join(os.path.dirname(__file__), "..", "skills", "motion-graphics-scene")


def wait(jobs, jid, timeout=5.0):
    t0 = time.time()
    while time.time() - t0 < timeout:
        j = jobs.get(jid)
        if j["state"] != "running":
            return jobs.public(j)
        time.sleep(0.02)
    raise AssertionError("job did not finish")


class JobTests(unittest.TestCase):
    def test_bad_requests(self):
        jobs = MgRenderJobs(run=lambda **kw: "{}")
        for body in (None, {}, {"scene_path": "x.json"}, {"scene_path": "a.mgscene.json", "format": "avi"},
                     {"scene_path": "a.mgscene.json", "scale": "big"}):
            with self.assertRaises(ValueError):
                jobs.start(body)

    def test_runs_in_the_background_and_delivers(self):
        seen, delivered = {}, []

        def run(**kw):
            seen.update(kw)
            return json.dumps({"path": "C:/x/a.mp4", "frames": 90, "width": 1920, "height": 1080, "fps": 30})
        jobs = MgRenderJobs(run=run, deliver=delivered.append)
        job = jobs.start({"scene_path": "C:/x/a.mgscene.json", "format": "WEBM", "scale": 5})
        done = wait(jobs, job["id"])
        self.assertEqual((done["state"], done["path"], done["frames"]), ("done", "C:/x/a.mp4", 90))
        self.assertEqual((seen["format"], seen["scale"]), ("webm", 1.0))
        self.assertEqual(delivered, ["C:/x/a.mp4"])

    def test_a_failed_render_reports_why(self):
        jobs = MgRenderJobs(run=lambda **kw: json.dumps({"error": "the scene is not valid"}))
        done = wait(jobs, jobs.start({"scene_path": "a.mgscene.json"})["id"])
        self.assertEqual((done["state"], done["error"]), ("error", "the scene is not valid"))


class SkillExamples(unittest.TestCase):
    def test_every_example_in_the_skill_is_a_valid_scene(self):
        from agenty_core.motion_graphics import schema
        folder = os.path.join(SKILL, "examples")
        names = [n for n in os.listdir(folder) if n.endswith(".mgscene.json")]
        self.assertGreaterEqual(len(names), 3)
        for n in names:
            r = schema.validate_scene(schema.load_scene(os.path.join(folder, n)))
            self.assertTrue(r["ok"], f"{n}: {r['errors']}")
            self.assertEqual(r["warnings"], [], n)


if __name__ == "__main__":
    unittest.main()
