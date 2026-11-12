"""New Atlas worlds asked for by the viewer's 3D space: validated, made in the background, opened in the viewer."""
import json
import os
import tempfile
import time
import unittest

from src.utils.atlas_create_jobs import AtlasCreateJobs


def wait(jobs, jid, timeout=5.0):
    t0 = time.time()
    while time.time() - t0 < timeout:
        j = jobs.get(jid)
        if j["state"] != "running":
            return jobs.public(j)
        time.sleep(0.02)
    raise AssertionError("job did not finish")


class CreateJobTests(unittest.TestCase):
    def test_bad_requests(self):
        jobs = AtlasCreateJobs(create=lambda **kw: "{}", open_world=lambda w: "{}")
        for body in (None, {}, {"name": "w"}, {"name": "", "prompt": "a room"},
                     {"name": "w", "prompt": "x", "aspect_ratio": "2:1"},
                     {"name": "w", "prompt": "x", "mesh_detail": 3},
                     {"name": "w", "image_paths": ["/no/such/file.png"]}):
            with self.assertRaises(ValueError, msg=body):
                jobs.start(body)

    def test_prompt_world_is_made_and_opened(self):
        seen, opened = {}, []

        def create(**kw):
            seen.update(kw)
            return json.dumps({"status": "ok", "world_dir": "C:/a/court", "glb": "C:/a/court/world.glb",
                               "frames": [{"id": 0}], "prompt_used": "a sunny courtyard"})

        def open_world(w):
            opened.append(w)
            return json.dumps({"status": "ok", "tab": "previz:atlas_court"})
        jobs = AtlasCreateJobs(create=create, open_world=open_world)
        job = jobs.start({"name": "court", "prompt": "a courtyard", "aspect_ratio": "4:3", "mesh_detail": "8"})
        done = wait(jobs, job["id"])
        self.assertEqual((done["state"], done["glb"], done["tab"], done["frames"]),
                         ("done", "C:/a/court/world.glb", "previz:atlas_court", 1))
        self.assertEqual((seen["prompt"], seen["image_paths"], seen["aspect_ratio"], seen["mesh_detail"]),
                         ("a courtyard", None, "4:3", 8))
        self.assertEqual(opened, ["C:/a/court"])

    def test_pictures_are_passed_and_open_failure_is_only_a_note(self):
        with tempfile.TemporaryDirectory() as d:
            img = os.path.join(d, "room.png")
            open(img, "wb").write(b"x")
            seen = {}
            jobs = AtlasCreateJobs(create=lambda **kw: seen.update(kw) or json.dumps({"world_dir": d, "glb": "g"}),
                                   open_world=lambda w: json.dumps({"error": "viewer too old"}))
            done = wait(jobs, jobs.start({"name": "room", "image_paths": [img]})["id"])
        self.assertEqual(seen["image_paths"], [img])
        self.assertEqual(done["state"], "done")
        self.assertIn("viewer too old", done["message"])

    def test_errors_pass_through_and_one_name_runs_once(self):
        jobs = AtlasCreateJobs(create=lambda **kw: json.dumps({"error": "out of credits", "error_kind": "balance"}),
                               open_world=lambda w: "{}")
        done = wait(jobs, jobs.start({"name": "w", "prompt": "x"})["id"])
        self.assertEqual((done["state"], done["error_kind"], done["message"]), ("error", "balance", "out of credits"))
        slow = AtlasCreateJobs(create=lambda **kw: time.sleep(0.3) or json.dumps({"world_dir": "d"}),
                               open_world=lambda w: "{}")
        first = slow.start({"name": "Hall", "prompt": "x"})
        with self.assertRaises(ValueError):
            slow.start({"name": "hall", "prompt": "y"})
        wait(slow, first["id"])

    def test_nothing_is_published_into_a_turn(self):
        from agenty_core.tools import image_io
        other = []
        image_io.set_output_sink(other.append)

        def create(**kw):
            image_io._publish("C:/x/view_000.png")
            return json.dumps({"world_dir": "d", "glb": "g"})
        try:
            jobs = AtlasCreateJobs(create=create, open_world=lambda w: "{}")
            wait(jobs, jobs.start({"name": "w", "prompt": "x"})["id"])
        finally:
            image_io.set_output_sink(None)
        self.assertEqual(other, [])


if __name__ == "__main__":
    unittest.main()


class EchoJobTests(unittest.TestCase):
    """engine "echo": SpAItial Echo, the second world generator."""

    def test_bad_echo_requests(self):
        jobs = AtlasCreateJobs(create_echo=lambda **kw: "{}")
        with tempfile.TemporaryDirectory() as d:
            a, b = os.path.join(d, "a.png"), os.path.join(d, "b.png")
            for p in (a, b):
                open(p, "wb").write(b"x")
            for body in ({"name": "w", "prompt": "x", "engine": "marble"},
                         {"name": "w", "image_paths": [a, b], "engine": "echo"},
                         {"name": "w", "prompt": "x", "engine": "echo", "model": "echo-3"},
                         {"name": "w", "prompt": "x", "engine": "echo", "is_pano": "maybe"},
                         {"name": "w", "prompt": "x", "engine": "echo", "mesh": "obj"}):
                with self.assertRaises(ValueError, msg=body):
                    jobs.start(body)

    def test_an_echo_world_from_a_picture_leaves_the_words_out(self):
        seen = {}

        def echo(**kw):
            seen.update(kw)
            return json.dumps({"status": "ok", "world_dir": "C:/e/room", "splat": "C:/e/room/splat.spz",
                               "mesh": "C:/e/room/mesh.ply", "viewer": {"tab": "previz:splat_room"}})
        with tempfile.TemporaryDirectory() as d:
            img = os.path.join(d, "room.png")
            open(img, "wb").write(b"x")
            jobs = AtlasCreateJobs(create=lambda **kw: self.fail("Atlas must not run"), create_echo=echo)
            job = jobs.start({"name": "room", "prompt": "words", "image_paths": [img], "engine": "echo",
                              "model": "echo-hq", "is_pano": "true", "mesh": "mesh"})
            self.assertIn("an hour", job["phase"])
            done = wait(jobs, job["id"])
        self.assertEqual((done["state"], done["engine"], done["splat"], done["mesh"], done["tab"]),
                         ("done", "echo", "C:/e/room/splat.spz", "C:/e/room/mesh.ply", "previz:splat_room"))
        self.assertEqual(seen, {"name": "room", "image": img, "prompt": "", "model": "echo-hq", "is_pano": "true",
                                "mesh": "mesh", "open_in_viewer": True})

    def test_an_echo_error_and_notes_reach_the_page(self):
        jobs = AtlasCreateJobs(create_echo=lambda **kw: json.dumps({"error": "too few credits", "error_kind": "balance",
                                                                    "request_id": "req_9"}))
        done = wait(jobs, jobs.start({"name": "p", "prompt": "a room", "engine": "echo"})["id"])
        self.assertEqual((done["state"], done["error_kind"], done["operation_id"]), ("error", "balance", "req_9"))
        jobs = AtlasCreateJobs(create_echo=lambda **kw: json.dumps({"world_dir": "d", "splat": "s", "mesh_error": "boom",
                                                                    "viewer": {"error": "no viewer"}}))
        done = wait(jobs, jobs.start({"name": "p", "prompt": "a room", "engine": "echo", "mesh": "mesh"})["id"])
        self.assertEqual(done["state"], "done")
        self.assertIn("mesh export failed: boom", done["message"])
        self.assertIn("no viewer", done["message"])
