"""Gaussian splats of Atlas worlds asked for by the viewer's Atlas menu: price first, then a background job."""
import json
import time
import unittest

from src.utils.atlas_splat_jobs import AtlasSplatJobs


def wait(jobs, jid, timeout=5.0):
    t0 = time.time()
    while time.time() - t0 < timeout:
        j = jobs.get(jid)
        if j["state"] != "running":
            return jobs.public(j)
        time.sleep(0.02)
    raise AssertionError("job did not finish")


class SplatJobTests(unittest.TestCase):
    def test_bad_requests(self):
        jobs = AtlasSplatJobs(run=lambda **kw: "{}")
        for body in (None, {}, {"world": ""}, {"world": "room"}):        # the last: no dry run, no confirm_cost
            with self.assertRaises(ValueError, msg=body):
                jobs.start(body)

    def test_dry_run_answers_the_plan_at_once_and_never_opens_the_viewer(self):
        seen = {}

        def run(**kw):
            seen.update(kw)
            return json.dumps({"status": "dry_run", "plan": {"frame_count": 5, "estimate": {"usd": 0.5}}})
        out = AtlasSplatJobs(run=run).start({"world": "room", "refine": False, "dry_run": True, "name": "my splat!"})
        self.assertEqual((out["state"], out["plan"]["frame_count"]), ("dry_run", 5))
        self.assertEqual((seen["dry_run"], seen["refine"], seen["open_in_viewer"], seen["name"]),
                         (True, False, False, "my_splat"))

    def test_dry_run_error_is_a_400(self):
        jobs = AtlasSplatJobs(run=lambda **kw: json.dumps({"error": "fewer than 2 views", "error_kind": "args"}))
        with self.assertRaises(ValueError):
            jobs.start({"world": "room", "dry_run": True})

    def test_confirmed_run_reports_the_files(self):
        seen = {}

        def run(**kw):
            seen.update(kw)
            return json.dumps({"status": "ok", "spz": "C:/w/splats/a.spz", "ply": "C:/w/splats/a.ply", "splats": 9})
        jobs = AtlasSplatJobs(run=run)
        done = wait(jobs, jobs.start({"world": "room", "confirm_cost": True})["id"])
        self.assertEqual((done["state"], done["spz"], done["ply"], done["splats"]),
                         ("done", "C:/w/splats/a.spz", "C:/w/splats/a.ply", 9))
        self.assertEqual((seen["confirm_cost"], seen["open_in_viewer"], seen["refine"]), (True, False, True))

    def test_a_failed_run_says_why(self):
        jobs = AtlasSplatJobs(run=lambda **kw: json.dumps({"error": "out of credits", "error_kind": "balance"}))
        done = wait(jobs, jobs.start({"world": "room", "confirm_cost": True})["id"])
        self.assertEqual((done["state"], done["message"], done["error_kind"]), ("error", "out of credits", "balance"))


if __name__ == "__main__":
    unittest.main()
