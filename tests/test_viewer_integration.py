"""The bEpic viewer as agentY's viewer: its tab, its stages, its tags.

Three things agentY does for the viewer (the viewer's own side - the history
menu, the grouping in the strip - lives in ComfyUI-ImageViewer):

* before a workflow is submitted its viewer save node is told ONE tab for all of
  the agent's pictures, the stage they belong to and, in a batch, what each is;
* "Tag for agentY" in the viewer becomes a project reference, resolved like the
  one an `agentY add tag` node writes;
* the viewer is the default saver and is installed beside the sidebar extension.

    python -m unittest tests.test_viewer_integration
"""

import json
import tempfile
import tomllib
import unittest
from pathlib import Path
from unittest import mock

from agenty_core.utils import project_memory as PM
from src.utils import output_tags, tag_memory, viewer_group
from src.utils import agentY_server as srv

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = {"input": {"required": {"input": ["*"], "tab_name": ["STRING"]},
                    "optional": {"group": ["STRING"], "label": ["STRING"]}}}
OLD_SCHEMA = {"input": {"required": {"input": ["*"], "tab_name": ["STRING"]}}}


def _graph(**inputs):
    return {"9": {"class_type": "bEpicSendToViewer", "inputs": {"input": ["4", 0], "tab_name": "", **inputs}},
            "4": {"class_type": "VAEDecode", "inputs": {}}}


def _settings(**media):
    return mock.patch("src.agent._load_settings", return_value={"media_nodes": media})


class OneTabGroupedByStage(unittest.TestCase):

    def setUp(self):
        output_tags.clear()
        self.addCleanup(output_tags.clear)

    def stamp(self, graph, path="wf.json", schema=SCHEMA, **media):
        with _settings(**media):
            return viewer_group.stamp(graph, path, schema_of=lambda c: schema)

    def test_the_stage_and_the_tab_are_written_into_the_node(self):
        output_tags.set_run_role("hero sheets", stage="Characters", hook="7")
        g = _graph()
        self.assertEqual(self.stamp(g), 1)
        inputs = g["9"]["inputs"]
        self.assertEqual((inputs["tab_name"], inputs["group"]), ("agentY", "Characters"))
        self.assertEqual(inputs["label"], "hero sheets")

    def test_a_batch_member_is_named_by_its_own_role(self):
        output_tags.set_run_role("Characters", stage="Characters")
        output_tags.set_workflow_role("anna.json", "Characters: Anna", stage="Characters", asset="Anna")
        g = _graph()
        self.stamp(g, "anna.json")
        self.assertEqual((g["9"]["inputs"]["group"], g["9"]["inputs"]["label"]), ("Characters", "Anna"))

    def test_a_stage_that_is_its_own_label_is_not_said_twice(self):
        output_tags.set_run_role("Places", stage="Places")
        g = _graph()
        self.stamp(g)
        self.assertEqual(g["9"]["inputs"]["label"], "")

    def test_a_run_with_no_stage_still_lands_in_the_one_tab(self):
        g = _graph()
        self.stamp(g)
        self.assertEqual((g["9"]["inputs"]["tab_name"], g["9"]["inputs"]["group"]), ("agentY", ""))

    def test_a_tab_the_user_named_is_left_alone(self):
        output_tags.set_run_role("x", stage="Characters")
        g = _graph(tab_name="my own")
        self.assertEqual(self.stamp(g), 0)
        self.assertEqual(g["9"]["inputs"]["tab_name"], "my own")
        self.assertNotIn("group", g["9"]["inputs"])

    def test_the_tab_name_is_a_setting_and_empty_turns_it_off(self):
        g = _graph()
        self.assertEqual(self.stamp(g, viewer_tab="Previews"), 1)
        self.assertEqual(g["9"]["inputs"]["tab_name"], "Previews")
        g = _graph()
        self.assertEqual(self.stamp(g, viewer_tab=""), 0)
        self.assertEqual(g["9"]["inputs"]["tab_name"], "")

    def test_an_older_viewer_without_the_group_input_is_not_touched(self):
        g = _graph()
        self.assertEqual(self.stamp(g, schema=OLD_SCHEMA), 0)
        self.assertEqual(g["9"]["inputs"]["tab_name"], "")

    def test_a_second_pass_changes_nothing(self):
        output_tags.set_run_role("x", stage="Places")
        g = _graph()
        self.stamp(g)
        self.assertEqual(self.stamp(g), 0)

    def test_the_executor_stamps_before_it_submits(self):
        src = (ROOT / "src" / "executor.py").read_text(encoding="utf-8")
        self.assertLess(src.index("media_nodes.apply_savers(workflow)"), src.index("viewer_group.stamp("))

    def test_the_pipeline_records_the_stage_a_picture_belongs_to(self):
        from src.pipeline import _stage_name
        self.assertEqual(_stage_name({"title": "Characters", "hook_node_id": "7"}), "Characters")
        self.assertEqual(_stage_name({"title": "agentY hook"}, "hero sheets"), "hero sheets")
        self.assertEqual(_stage_name(None), "")


class TaggingFromTheViewer(unittest.TestCase):

    def setUp(self):
        d = tempfile.TemporaryDirectory()
        self.addCleanup(d.cleanup)
        self.root = Path(d.name)
        self.enterContext(mock.patch.object(PM, "user_dir", lambda: self.root))
        self.pic = self.root / "anna.png"
        self.pic.write_bytes(b"png")

    def test_a_tagged_picture_resolves_like_a_tag_node_would(self):
        out = tag_memory.remember_file("#Hero Face", str(self.pic), "Anna, the face only")
        self.assertEqual(out["tag"], "Hero_Face")
        hit = tag_memory.resolve("hero_face", {})
        self.assertEqual(hit["source"], "memory")
        self.assertTrue(hit["path"].endswith("anna.png"))
        self.assertIn("the face only", hit["role"])
        self.assertIn("bEpic viewer", hit["text"])

    def test_tagging_again_replaces_it(self):
        tag_memory.remember_file("hero", str(self.pic), "one")
        tag_memory.remember_file("hero", str(self.pic), "two")
        self.assertEqual(tag_memory.remembered_reference("hero")["role"], "two")

    def test_what_cannot_be_tagged_says_why(self):
        for tag, path in (("", str(self.pic)), ("123", str(self.pic)),
                          ("hero", str(self.root / "gone.png")), ("hero", str(self.root))):
            with self.assertRaises(ValueError, msg=(tag, path)):
                tag_memory.remember_file(tag, path)
        txt = self.root / "notes.txt"
        txt.write_text("x")
        with self.assertRaises(ValueError):
            tag_memory.remember_file("hero", str(txt))

    def test_the_route(self):
        from tests.route_client import authorised_client
        app = srv._build_app()
        app.testing = True
        client = authorised_client(app)
        ok = client.post("/agentY/viewer/tag", json={"tag": "anna", "path": str(self.pic), "role": "lead"})
        self.assertEqual(ok.status_code, 200, ok.data)
        self.assertEqual(json.loads(ok.data)["tag"], "anna")
        self.assertTrue(PM.read_entry("anna"))
        bad = client.post("/agentY/viewer/tag", json={"tag": "anna", "path": str(self.root / "no.png")})
        self.assertEqual(bad.status_code, 400)
        self.assertFalse(json.loads(bad.data)["ok"])


class InstalledWithAgentY(unittest.TestCase):

    def test_the_viewer_is_the_default_saver(self):
        media = tomllib.loads((ROOT / "config" / "settings.default.toml").read_text(encoding="utf-8"))["media_nodes"]
        for kind in ("image", "video", "model"):
            self.assertEqual(media[f"{kind}_save"], "bEpicSendToViewer")
        self.assertEqual(media["viewer_tab"], "agentY")

    def test_the_installers_and_the_launchers_know_the_viewer(self):
        for name in ("install_agent.ps1", "install_agent.sh", "run_agent.ps1", "run_agent.sh"):
            text = (ROOT / name).read_text(encoding="utf-8")
            self.assertIn("ComfyUI-ImageViewer", text, name)
        for name in ("install_agent.ps1", "install_agent.sh"):
            self.assertIn("https://github.com/szprivate/ComfyUI-ImageViewer.git",
                          (ROOT / name).read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
