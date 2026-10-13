"""What Settings says about paths, and what a save may take out of .env.

A path setting is shown in full, as it comes to on this machine; ``output_dir``
is read against ComfyUI's output folder; ComfyUI's extra model folders are
reported; a key removed in Authentication leaves .env altogether.
"""

import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from src import executor
from src.utils import agentY_server as srv

FACTS = {
    "argv": ["main.py"], "root": r"D:\AI\comfyui",
    "user_dir": r"W:\proj\user_dir", "output_dir": r"W:\proj\output",
    "folders": {"checkpoints": [r"D:\AI\comfyui\models\checkpoints", r"L:\Models\checkpoints", r"W:\proj\output\checkpoints"],
                "loras": [r"D:\AI\comfyui\models\loras", r"L:\Models\loras"],
                "sam3": []},
}


class PathsInFull(unittest.TestCase):

    def setUp(self):
        self.root = Path(srv._project_root())

    def test_a_blank_setting_is_where_things_really_go(self):
        got = srv._setting_defaults({}, FACTS)
        self.assertEqual(got["comfyui_dir"], r"D:\AI\comfyui")
        self.assertEqual(Path(got["comfyui_models_dir"]), Path(r"D:\AI\comfyui\models"))
        self.assertEqual(Path(got["comfyui_user_dir"]), Path(r"W:\proj\user_dir"))
        self.assertEqual(Path(got["output_workflows_dir"]), Path(r"W:\proj\user_dir\default\workflows\agentY"))

    def test_a_relative_one_is_resolved_against_this_checkout(self):
        got = srv._setting_defaults({"conversation_db": "./memory/conversations.sqlite",
                                     "message_history_log": "./.logs/message_history.log"}, FACTS)
        self.assertEqual(Path(got["conversation_db"]), self.root / "memory" / "conversations.sqlite")
        self.assertEqual(Path(got["message_history_log"]), self.root / ".logs" / "message_history.log")
        self.assertTrue(Path(got["conversation_db"]).is_absolute())

    def test_output_dir_is_read_against_comfyuis_output_folder(self):
        self.assertEqual(Path(srv._setting_defaults({"output_dir": ""}, FACTS)["output_dir"]), Path(r"W:\proj\output"))
        self.assertEqual(Path(srv._setting_defaults({"output_dir": "fetched"}, FACTS)["output_dir"]),
                         Path(r"W:\proj\output\fetched"))
        self.assertEqual(Path(srv._setting_defaults({"output_dir": r"E:\elsewhere"}, FACTS)["output_dir"]),
                         Path(r"E:\elsewhere"))

    def test_without_comfyui_it_says_so_instead_of_guessing(self):
        empty = {"argv": [], "root": "", "user_dir": "", "output_dir": "", "folders": {}}
        got = srv._setting_defaults({}, empty)
        for key in ("comfyui_dir", "comfyui_user_dir", "output_dir", "output_workflows_dir"):
            self.assertIn("ComfyUI is running", got[key], key)

    def test_comfyuis_other_model_folders(self):
        self.assertEqual(srv._extra_model_dirs(FACTS), [r"L:\Models", r"W:\proj\output"])
        self.assertEqual(srv._extra_model_dirs({"root": "", "folders": {}}), [])


class WhereAFetchedResultGoes(unittest.TestCase):

    def _dir(self, setting, comfy):
        with mock.patch.object(executor, "_load_config", return_value={"output_dir": setting}), \
                mock.patch.object(executor, "_get_comfyui_output_dir", return_value=comfy):
            return executor._output_dir()

    def test_blank_is_comfyuis_output_folder_and_relative_is_inside_it(self):
        with tempfile.TemporaryDirectory() as tmp:
            comfy = Path(tmp).resolve()
            self.assertEqual(self._dir("", comfy), comfy)
            self.assertEqual(self._dir("fetched", comfy), comfy / "fetched")

    def test_an_absolute_path_is_used_as_it_is(self):
        with tempfile.TemporaryDirectory() as tmp, tempfile.TemporaryDirectory() as other:
            self.assertEqual(self._dir(other, Path(tmp)), Path(other).resolve())

    def test_a_comfyui_folder_that_is_not_on_this_machine_falls_back_to_the_checkout(self):
        gone = Path(tempfile.gettempdir()) / "agenty_no_such_comfy_output"
        got = self._dir("", gone)
        self.assertEqual(got, (executor._project_root() / "output_images").resolve())
        self.assertEqual(self._dir("", None), got)


class RemovingAKey(unittest.TestCase):

    def test_its_line_leaves_env_and_nothing_else_does(self):
        with tempfile.TemporaryDirectory() as tmp:
            env = Path(tmp) / ".env"
            env.write_text("# keys\nHF_TOKEN=abc\nMY_OLD_KEY=zzz\n\nOPENAI_API_KEY=def\n", encoding="utf-8")
            with mock.patch.object(srv, "_env_path", return_value=env), mock.patch.object(srv, "_note_key_ages"), \
                    mock.patch.dict(os.environ, {"MY_OLD_KEY": "zzz"}):
                removed = srv._remove_env_keys(["MY_OLD_KEY", "NOT_THERE", "bad name; rm"])
                self.assertNotIn("MY_OLD_KEY", os.environ)
            self.assertEqual(removed, ["MY_OLD_KEY"])
            self.assertEqual(env.read_text(encoding="utf-8"), "# keys\nHF_TOKEN=abc\n\nOPENAI_API_KEY=def\n")

    def test_a_save_that_names_keys_to_remove_removes_them(self):
        with tempfile.TemporaryDirectory() as tmp:
            env = Path(tmp) / ".env"
            env.write_text("A=1\nB=2\n", encoding="utf-8")
            with mock.patch.object(srv, "_env_path", return_value=env), mock.patch.object(srv, "_note_key_ages"), \
                    mock.patch.dict(os.environ, {}, clear=False):
                got = srv.settings_save({"env_remove": ["A"]})
                os.environ.pop("A", None)
            self.assertEqual(got.get("env_removed"), ["A"])
            self.assertEqual(env.read_text(encoding="utf-8"), "B=2\n")


class WhatCanBePicked(unittest.TestCase):

    def test_the_decision_models_are_offered_installed_or_not(self):
        from src.utils import decision_models as dm
        with mock.patch.object(dm, "installed", return_value=True), \
                mock.patch.dict(dm._installed, {"names": ("nimble:latest", "qwen3.8:27b", "nomic-embed-text:latest")}):
            got = srv._aux_model_choices({}, FACTS)
        ids = [c["id"] for c in got["decision.router_model"]]
        self.assertEqual(ids[:3], ["nimble", "clef-flash", "clef"])
        self.assertIn("qwen3.8:27b", ids)
        self.assertNotIn("nomic-embed-text", ids)                          # an embedder answers no questions
        labels = {c["id"]: c["label"] for c in got["decision.router_model"]}
        self.assertEqual(labels["nimble"], "nimble")
        self.assertIn("downloaded on first use", labels["clef"])
        self.assertEqual(got["decision.prompt_check_model"], got["decision.router_model"])

    def test_the_face_models_and_the_checkpoint(self):
        got = srv._aux_model_choices({}, FACTS)
        self.assertEqual(got["likeness.face_model"][0]["id"], "buffalo_l")
        self.assertEqual(got["annotate.sam3_checkpoint"][0]["id"], "")      # automatic comes first


if __name__ == "__main__":
    unittest.main()
