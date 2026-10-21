"""The Deadline "ComfyUI" plugin, run against a stub of Deadline's Python API: a worker without ComfyUI fails fast
with a clear message, seeds are randomised per task, ComfyUI's errors are turned into readable text."""

import importlib.util
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path

_PLUGIN = Path(__file__).resolve().parent.parent / "deadline" / "plugins" / "ComfyUI" / "ComfyUI.py"


class _Cb:
    def __iadd__(self, other):
        return self


class _Failed(Exception):
    pass


class _StubPlugin:
    config: dict = {}
    info: dict = {}

    def __init__(self):
        for n in ("InitializeProcessCallback", "StartJobCallback", "RenderTasksCallback", "EndJobCallback"):
            setattr(self, n, _Cb())
        self.logs = []

    def GetConfigEntryWithDefault(self, k, d):
        return self.config.get(k, d)

    def GetBooleanConfigEntryWithDefault(self, k, d):
        return bool(self.config.get(k, d))

    def GetPluginInfoEntryWithDefault(self, k, d):
        return self.info.get(k, d)

    def FailRender(self, msg):
        raise _Failed(msg)

    def LogInfo(self, m):
        self.logs.append(m)

    LogWarning = LogInfo

    def GetStartFrame(self):
        return 3

    def GetJob(self):
        return types.SimpleNamespace(JobId="job1")

    def IsCanceled(self):
        return False


def _load():
    dl = types.ModuleType("Deadline")
    plugins = types.ModuleType("Deadline.Plugins")
    plugins.DeadlinePlugin, plugins.PluginType = _StubPlugin, types.SimpleNamespace(Advanced=1)
    scripting = types.ModuleType("Deadline.Scripting")
    scripting.RepositoryUtils = types.SimpleNamespace(CheckPathMapping=lambda p: p)
    scripting.SystemUtils = types.SimpleNamespace(Sleep=lambda ms: None)
    sys.modules.update({"Deadline": dl, "Deadline.Plugins": plugins, "Deadline.Scripting": scripting})
    spec = importlib.util.spec_from_file_location("comfyui_deadline_plugin", _PLUGIN)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class Plugin(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mod = _load()

    def setUp(self):
        self.p = self.mod.GetDeadlinePlugin()
        self.p.config, self.p.info = {}, {}

    def test_not_installed_fails_fast(self):
        self.p.config = {"ComfyUIPath_" + self.mod.ComfyUIPlugin._os_name(): r"Z:\nowhere\ComfyUI"}
        with self.assertRaises(_Failed) as cm:
            self.p.StartJob()
        self.assertIn("ComfyUI not installed on", str(cm.exception))
        self.assertIn("does not exist", str(cm.exception))

    def test_no_path_configured_fails_fast(self):
        with self.assertRaises(_Failed) as cm:
            self.p.StartJob()
        self.assertIn("ComfyUI not installed on", str(cm.exception))

    def test_folder_without_main_py(self):
        with tempfile.TemporaryDirectory() as d:
            self.p.info = {"ComfyUIPath": d}
            with self.assertRaises(_Failed) as cm:
                self.p.StartJob()
        self.assertIn("no main.py", str(cm.exception))

    def test_seeds_randomised_per_task(self):
        with tempfile.TemporaryDirectory() as d:
            pf = Path(d) / "prompt.json"
            pf.write_text(json.dumps({"1": {"class_type": "KSampler", "inputs": {"seed": 1, "steps": 20, "model": ["2", 0]}},
                                      "2": {"class_type": "Noise", "inputs": {"noise_seed": 2}}}))
            self.p.GetDataFilename = lambda: str(pf)
            self.p.info = {"SeedMode": "randomize"}
            a = self.p._load_prompt(3)
            b = self.p._load_prompt(3)
            self.p.GetStartFrame = lambda: 4
            c = self.p._load_prompt(4)
        self.assertEqual(a, b)                                  # deterministic per task
        self.assertNotEqual(a["1"]["inputs"]["seed"], c["1"]["inputs"]["seed"])
        self.assertEqual(a["1"]["inputs"]["steps"], 20)
        self.assertEqual(a["1"]["inputs"]["model"], ["2", 0])

    def test_seeds_kept(self):
        with tempfile.TemporaryDirectory() as d:
            pf = Path(d) / "prompt.json"
            pf.write_text(json.dumps({"1": {"class_type": "KSampler", "inputs": {"seed": 1}}}))
            self.p.GetDataFilename = lambda: str(pf)
            self.assertEqual(self.p._load_prompt(0)["1"]["inputs"]["seed"], 1)

    def test_error_texts(self):
        status = {"status_str": "error", "messages": [["execution_error", {
            "node_type": "KSampler", "node_id": "7", "exception_type": "OOM", "exception_message": "out of memory\n"}]]}
        txt = self.p._exec_error(status)
        self.assertIn("KSampler", txt)
        self.assertIn("out of memory", txt)
        raw = json.dumps({"error": {"message": "Prompt outputs failed validation", "details": ""},
                          "node_errors": {"3": {"class_type": "CheckpointLoaderSimple",
                                                "errors": [{"message": "Value not in list", "details": "ckpt_name"}]}}}).encode()
        self.assertIn("CheckpointLoaderSimple", self.p._describe_reject(raw))

    def test_api_key_from_env_and_file(self):
        import os
        with tempfile.TemporaryDirectory() as d:
            kf = Path(d) / "k.txt"
            kf.write_text("filekey\n")
            self.p.config = {"ApiKeyFile": str(kf)}
            os.environ.pop("COMFY_ORG_API_KEY", None)
            self.assertEqual(self.p._api_key(), "filekey")
            os.environ["COMFY_ORG_API_KEY"] = "envkey"
            try:
                self.assertEqual(self.p._api_key(), "envkey")
            finally:
                os.environ.pop("COMFY_ORG_API_KEY", None)


if __name__ == "__main__":
    unittest.main()
