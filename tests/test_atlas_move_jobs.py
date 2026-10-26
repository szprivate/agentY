"""Atlas camera moves asked for by the viewer: validated, a dry run answered at once, the real run in the background."""
import json
import time
import unittest

from src.utils.atlas_move_jobs import AtlasMoveJobs

M16 = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]


def body(**kw):
    b = {"world": "C:/w/interior2/world.glb", "name": "push", "model_world": M16, "quality": "standard",
         "keys": [{"t": 0, "camera_world": M16, "fov": 40, "aspect": 1.78, "gated": True},
                  {"t": 1, "camera_world": M16[:12] + [0, 0, -1, 1], "fov": 40, "aspect": 1.78, "gated": True}]}
    b.update(kw)
    return b


def wait(jobs, jid, timeout=5.0):
    t0 = time.time()
    while time.time() - t0 < timeout:
        j = jobs.get(jid)
        if j["state"] != "running":
            return jobs.public(j)
        time.sleep(0.02)
    raise AssertionError("job did not finish")


class MoveJobTests(unittest.TestCase):
    def test_bad_requests(self):
        jobs = AtlasMoveJobs(run=lambda **kw: "{}")
        for b in (None, {}, body(world=""), body(keys=[]), body(keys=[{"t": 0}, {"t": 1}]), body(quality="ultra"),
                  body(model_world=[1, 2]), body(keys=[{"t": 0, "camera_world": [1]}] * 2)):
            with self.assertRaises(ValueError):
                jobs.start(b)

    def test_dry_run_is_answered_at_once_and_spends_nothing(self):
        seen = {}

        def run(**kw):
            seen.update(kw)
            return json.dumps({"status": "dry_run", "spent": 0, "plan": {"tasks": 2}})
        out = AtlasMoveJobs(run=run).start(body(dry_run=True))
        self.assertEqual((out["dry_run"], out["plan"]["tasks"]), (True, 2))
        self.assertTrue(seen["dry_run"])
        self.assertFalse(seen["confirm_cost"])
        self.assertEqual(seen["keys"][1]["viewer"]["camera_world"][14], -1.0)
        self.assertEqual(seen["keys"][0]["viewer"]["model_world"], [float(x) for x in M16])

    def test_runs_in_the_background_with_progress(self):
        def run(on_progress=None, **kw):
            on_progress({"phase": "Atlas is painting views 1 of 2", "tasks_done": 0, "tasks_total": 2})
            on_progress({"phase": "Rendering", "tasks_done": 2, "tasks_total": 2})
            return json.dumps({"status": "ok", "path": "C:/out/push.mp4", "tasks_used": 2, "frames": 49})
        jobs = AtlasMoveJobs(run=run)
        job = jobs.start(body(confirm_cost=True))
        done = wait(jobs, job["id"])
        self.assertEqual((done["state"], done["path"], done["tasks_done"], done["tasks_total"]),
                         ("done", "C:/out/push.mp4", 2, 2))

    def test_the_cost_gate_error_passes_through(self):
        jobs = AtlasMoveJobs(run=lambda **kw: json.dumps({"error": "Not run: ...", "error_kind": "confirm_cost"}))
        done = wait(jobs, jobs.start(body())["id"])
        self.assertEqual((done["state"], done["error_kind"]), ("error", "confirm_cost"))


if __name__ == "__main__":
    unittest.main()
