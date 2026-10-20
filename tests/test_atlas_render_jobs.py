"""Atlas renders asked for by the viewer: validated, run in the background, delivered outside any turn."""
import json
import time
import unittest

from src.utils.atlas_render_jobs import AtlasRenderJobs


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
        jobs = AtlasRenderJobs(run=lambda **kw: "{}")
        for body in (None, {}, {"world": "w"}, {"world": "w", "camera": {}}, {"world": "w", "camera": {"forward": 1}, "upscale": 3},
                     {"world": "", "camera": {"forward": 1}}):
            with self.assertRaises(ValueError):
                jobs.start(body)

    def test_runs_in_the_background_and_reports(self):
        seen = {}

        def run(**kw):
            seen.update(kw)
            return json.dumps({"status": "ok", "path": "C:/x/render_000_x2.png", "original": "C:/x/render_000.png",
                               "fov_deg": 50.0, "size": [2560, 1440]})
        jobs = AtlasRenderJobs(run=run)
        job = jobs.start({"world": "C:/w/world.glb", "camera": {"viewer": {"camera_world": [0] * 16}}, "upscale": "2",
                          "add_to_world": 1})
        done = wait(jobs, job["id"])
        self.assertEqual((done["state"], done["path"], done["size"]), ("done", "C:/x/render_000_x2.png", [2560, 1440]))
        self.assertEqual((seen["upscale"], seen["add_to_world"], seen["world"]), (2, True, "C:/w/world.glb"))

    def test_errors_pass_through_with_their_kind(self):
        jobs = AtlasRenderJobs(run=lambda **kw: json.dumps({"error": "out of credits", "error_kind": "balance"}))
        done = wait(jobs, jobs.start({"world": "w", "camera": {"forward": 1}})["id"])
        self.assertEqual((done["state"], done["error_kind"], done["message"]), ("error", "balance", "out of credits"))
        boom = AtlasRenderJobs(run=lambda **kw: (_ for _ in ()).throw(RuntimeError("x")))
        self.assertEqual(wait(boom, boom.start({"world": "w", "camera": {"forward": 1}})["id"])["state"], "error")

    def test_the_picture_goes_to_the_jobs_own_sink_not_a_turns(self):
        from agenty_core.tools import image_io
        got, other = [], []
        image_io.set_output_sink(other.append)           # whatever the host registered module-wide

        def run(**kw):
            image_io._publish("C:/x/render.png")
            return json.dumps({"status": "ok", "path": "C:/x/render.png"})
        try:
            jobs = AtlasRenderJobs(deliver=got.append, run=run)
            wait(jobs, jobs.start({"world": "w", "camera": {"forward": 1}})["id"])
        finally:
            image_io.set_output_sink(None)
        self.assertEqual(got, ["C:/x/render.png"])
        self.assertEqual(other, [])

    def test_the_route_is_registered(self):
        import inspect
        from src.utils import agentY_server
        src = inspect.getsource(agentY_server)
        self.assertIn('"/agentY/atlas_render"', src)


if __name__ == "__main__":
    unittest.main()
