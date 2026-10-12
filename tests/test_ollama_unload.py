"""A local model gives the GPU back before ComfyUI renders."""
import unittest
from unittest import mock

from src.utils import ollama_unload as ou

LOCAL = {"llm": {"tiers": {"orchestrator": "ollama,qwen3.6:27b", "vision": "dashscope,qwen3.8-flash"}}}
HOSTED = {"llm": {"tiers": {"orchestrator": "dashscope,deepseek-v4.1-flash"}, "pipeline": {"learnings": ""}}}
GB = 1024 ** 3


class Wanted(unittest.TestCase):

    def test_only_when_a_role_runs_on_ollama(self):
        self.assertTrue(ou.wanted(LOCAL))
        self.assertFalse(ou.wanted(HOSTED))
        self.assertTrue(ou.wanted({"llm": {"tiers": {}, "pipeline": {"coder": "Ollama,qwen3-coder:30b"}}}))

    def test_the_switch_turns_it_off(self):
        cfg = {"llm": {**LOCAL["llm"], "ollama": {"unload_before_render": False}}}
        self.assertFalse(ou.wanted(cfg))

    def test_settings_that_make_no_sense_mean_no(self):
        self.assertFalse(ou.wanted({"llm": "nonsense"}))


class Release(unittest.TestCase):

    def setUp(self):
        ou._last = 0.0
        self.calls = []

    def _fake(self, loaded, fail_on=None):
        def call(url, body=None):
            self.calls.append((url, body))
            if fail_on and fail_on in url:
                raise OSError("down")
            return {"models": loaded} if url.endswith("/api/ps") else {}
        return mock.patch.object(ou, "_call", side_effect=call)

    def test_the_large_models_are_unloaded_and_the_embedder_is_left(self):
        loaded = [{"name": "qwen3.6:27b", "size_vram": 20 * GB}, {"name": "nomic-embed-text:latest", "size_vram": GB // 3}]
        with self._fake(loaded):
            got = ou.release("http://127.0.0.1:11434/", settings=LOCAL)
        self.assertEqual(got, ["qwen3.6:27b"])
        self.assertEqual(self.calls[-1], ("http://127.0.0.1:11434/api/generate",
                                          {"model": "qwen3.6:27b", "keep_alive": 0}))

    def test_nothing_is_asked_when_no_role_is_local(self):
        with self._fake([{"name": "x", "size_vram": 20 * GB}]):
            self.assertEqual(ou.release("http://h", settings=HOSTED), [])
        self.assertEqual(self.calls, [])

    def test_a_batch_asks_once(self):
        with self._fake([{"name": "x", "size_vram": 20 * GB}]):
            self.assertEqual(ou.release("http://h", settings=LOCAL), ["x"])
            self.assertEqual(ou.release("http://h", settings=LOCAL), [])
        self.assertEqual(len(self.calls), 2)

    def test_an_ollama_that_does_not_answer_is_not_an_error(self):
        with self._fake([], fail_on="/api/ps"):
            self.assertEqual(ou.release("http://h", settings=LOCAL), [])
        ou._last = 0.0
        with self._fake([{"name": "x", "size_vram": 20 * GB}], fail_on="/api/generate"):
            self.assertEqual(ou.release("http://h", settings=LOCAL), [])


class Wired(unittest.TestCase):
    """Both places a graph is queued for rendering ask first."""

    def test_the_executor_and_the_probe_release_before_they_queue(self):
        from pathlib import Path
        root = Path(__file__).resolve().parent.parent / "src"
        for name in ("executor.py", "utils/canvas_tap.py"):
            src = (root / name).read_text(encoding="utf-8")
            with self.subTest(file=name):
                self.assertLess(src.index("_release_gpu()"), src.index('client.post("/prompt"'))


if __name__ == "__main__":
    unittest.main()
