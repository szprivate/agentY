"""A conversation that worked, read back as steps for a hook pipeline (conversation_as_pipeline)."""
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from src.utils import conversation_store as cs
from src.utils import conversation_pipeline as cp


def t2i(prompt: str, seed: int = 7) -> dict:
    return {
        "4": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": "flux_dev.safetensors"}},
        "6": {"class_type": "CLIPTextEncode", "inputs": {"text": prompt, "clip": ["4", 1]}},
        "7": {"class_type": "CLIPTextEncode", "inputs": {"text": "blurry", "clip": ["4", 1]}},
        "3": {"class_type": "KSampler", "inputs": {"seed": seed, "steps": 20, "model": ["4", 0],
                                                   "positive": ["6", 0], "negative": ["7", 0]}},
        "9": {"class_type": "SaveImage", "inputs": {"images": ["3", 0], "filename_prefix": "robot"}},
    }


def i2v(image_name: str) -> dict:
    return {
        "1": {"class_type": "LoadImage", "inputs": {"image": image_name}},
        "2": {"class_type": "WanImageToVideo", "inputs": {"start_image": ["1", 0]}},
        "3": {"class_type": "SaveVideo", "inputs": {"video": ["2", 0]}},
    }


class Base(unittest.TestCase):
    def setUp(self):
        self._dir = tempfile.TemporaryDirectory()
        self.addCleanup(self._dir.cleanup)
        self.root = Path(self._dir.name)
        env = mock.patch.dict(os.environ, {"AGENTY_CONVERSATION_DB": str(self.root / "c.sqlite")})
        env.start()
        self.addCleanup(env.stop)
        cs._INITIALISED = False
        self.addCleanup(setattr, cs, "_INITIALISED", False)
        self.wf_dir = self.root / "workflows"
        wd = mock.patch("agenty_core.tools.comfyui._workflows_dir", return_value=self.wf_dir)
        wd.start()
        self.addCleanup(wd.stop)
        self.tid = cs.create_thread("Robot portrait to video")
        self.clock = 1000.0

    def at(self, sql: str, args: tuple):
        self.clock += 10
        with cs._connect() as conn:
            conn.execute(sql, (*args, self.clock))

    def say(self, role: str, text: str):
        self.at("INSERT INTO messages(thread_id, role, content, created_at) VALUES (?,?,?,?)",
                (self.tid, role, text))

    def output(self, name: str, graph: dict | None, workflow_file: str = "") -> str:
        from agenty_core.utils import provenance as pv
        path = self.root / "out" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"x")
        if graph is not None:
            self.assertTrue(pv.write(path, graph, prompt_id="p", workflow_file=workflow_file))
        with cs._connect() as conn:
            idx = conn.execute("SELECT COALESCE(MAX(idx),0)+1 FROM gallery WHERE thread_id=?",
                               (self.tid,)).fetchone()[0]
        self.at("INSERT INTO gallery(thread_id, idx, path, caption, created_at) VALUES (?,?,?,?,?)",
                (self.tid, idx, str(path), ""))
        return str(path)


class Steps(Base):
    def setUp(self):
        super().setUp()
        wf = self.root / "robot_t2i.json"
        wf.write_text(json.dumps(t2i("x")), encoding="utf-8")
        self.say("user", "Make a portrait of a rusty robot")
        self.say("assistant", "Here are two portraits.")
        self.output("robot_a.png", t2i("a rusty robot, portrait", 1), str(wf))
        self.output("robot_b.png", t2i("a rusty robot, portrait", 2), str(wf))
        self.say("user", "Warmer light please")
        self.output("robot_c.png", t2i("a rusty robot, portrait, warm rim light", 3), str(wf))
        self.say("user", "Great. Animate it, 5 seconds")
        self.output("robot_c.mp4", i2v("robot_c.png [input]"))  # its workflow file is gone
        self.say("assistant", "Done - a 5 s clip.")
        self.say("user", "Perfect, thanks")
        self.got = cp.steps(self.tid)

    def test_each_user_message_opens_a_step_with_its_outputs(self):
        st = self.got["steps"]
        self.assertEqual([s["step"] for s in st], [1, 2, 3, 4])
        self.assertEqual([[o["n"] for o in s["outputs"]] for s in st], [[1, 2], [3], [4], []])
        self.assertEqual(st[0]["asked"], "Make a portrait of a rusty robot")
        self.assertEqual(st[0]["replied"], "Here are two portraits.")
        self.assertEqual(st[0]["reaction"], "Warmer light please")
        self.assertNotIn("reaction", st[-1])

    def test_an_output_names_its_workflow_and_prompt(self):
        o = self.got["steps"][1]["outputs"][0]
        self.assertTrue(o["workflow"].endswith("robot_t2i.json"))
        self.assertEqual(o["workflow_source"], "the workflow file that ran")
        self.assertEqual(o["params"]["positive"]["value"], "a rusty robot, portrait, warm rim light")
        self.assertEqual(o["params"]["positive"]["id"], "6.text")
        self.assertIn("flux_dev.safetensors", o["made_with"])

    def test_what_a_step_loaded_is_wired_back_to_the_output_it_came_from(self):
        video = self.got["steps"][2]["outputs"][0]
        self.assertEqual(video["reads"], [3])
        self.assertTrue(self.got["steps"][1]["outputs"][0]["used_by_a_later_step"])
        self.assertNotIn("used_by_a_later_step", self.got["steps"][0]["outputs"][0])
        self.assertTrue(self.got["steps"][2].get("last_step_with_outputs"))

    def test_a_lost_workflow_file_is_rebuilt_from_the_stored_graph(self):
        video = self.got["steps"][2]["outputs"][0]
        self.assertEqual(video["workflow_source"], "rebuilt from the graph stored with the file")
        self.assertTrue(Path(video["workflow"]).is_file())
        self.assertEqual(Path(video["workflow"]).parent, self.wf_dir)
        self.assertEqual(json.loads(Path(video["workflow"]).read_text())["1"]["inputs"]["image"],
                         "robot_c.png [input]")

    def test_workflows_list_where_each_was_used(self):
        by = {Path(w["workflow"]).name: w for w in self.got["workflows"]}
        self.assertEqual(by["robot_t2i.json"]["steps"], [1, 2])
        self.assertEqual(by["robot_t2i.json"]["outputs"], [1, 2, 3])

    def test_the_graph_never_goes_into_the_result(self):
        body = json.dumps(self.got, default=str)
        self.assertNotIn("CheckpointLoaderSimple", body)
        self.assertNotIn("loads", body)

    def test_only_the_newest_steps_of_a_long_conversation(self):
        got = cp.steps(self.tid, max_steps=2)
        self.assertEqual([s["step"] for s in got["steps"]], [3, 4])
        self.assertIn("newest 2 of 4", got["note"])


class OutputsWithoutARecord(Base):
    def test_a_file_with_no_provenance_says_so(self):
        self.say("user", "Make it with DashScope")
        self.output("ds.png", None)
        o = cp.steps(self.tid)["steps"][0]["outputs"][0]
        self.assertNotIn("workflow", o)
        self.assertIn("none", o["workflow_source"])


class TheTool(Base):
    def call(self, **kw):
        from src.tools.conversation_pipeline import conversation_as_pipeline
        return json.loads(conversation_as_pipeline(**kw))

    def test_by_title_words(self):
        self.say("user", "hello")
        got = self.call(conversation="Robot portrait")
        self.assertEqual(got["status"], "ok")
        self.assertEqual(got["conversation"], self.tid)
        self.assertIn("hook-pipeline", got["how_to"])

    def test_the_current_conversation_by_default(self):
        self.say("user", "hello")
        with mock.patch("src.utils.compaction._current_thread", return_value=self.tid):
            self.assertEqual(self.call()["conversation"], self.tid)

    def test_no_match(self):
        got = self.call(conversation="nothing like this")
        self.assertEqual(got["status"], "error")


class TheWiring(unittest.TestCase):
    def test_the_orchestrator_has_it_in_a_pack(self):
        from src.tools import ORCHESTRATOR_TOOLS
        from src.tools import tool_packs as tp
        self.assertIn("conversation_as_pipeline", {tp.name_of(t) for t in ORCHESTRATOR_TOOLS})
        self.assertEqual(tp._PACK_OF["conversation_as_pipeline"], "rebuild")

    def test_a_message_that_asks_for_it_pins_it_with_the_canvas_tools(self):
        from src.tools.tool_packs import packs_asked_for
        for text in ("turn this conversation into a hook pipeline",
                     "Rebuild what we just did as a pipeline",
                     "make the robot chat re-runnable as hooks",
                     "rebuild all steps as a hook chain"):
            self.assertEqual(packs_asked_for(text), {"rebuild", "graph"}, text)
        for text in ("build me a pipeline that upscales", "what hooks are on the canvas?",
                     "summarize this conversation", "install the impact pack custom nodes"):
            self.assertNotIn("rebuild", packs_asked_for(text), text)


if __name__ == "__main__":
    unittest.main()
