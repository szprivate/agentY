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
                self.assertLess(src.index("_release_gpu(prompt="), src.index('client.post("/prompt"'))


INFO = {"KlingOmniPro": {"python_module": "comfy_api_nodes.nodes_kling", "api_node": True},
        "KSampler": {"python_module": "nodes"}, "CheckpointLoaderSimple": {"python_module": "nodes"}}


class NeedsCard(unittest.TestCase):
    """The model steps aside only for a graph that uses the card."""

    def test_api_nodes_and_plumbing_do_not(self):
        graph = {"1": {"class_type": "LoadImage"}, "2": {"class_type": "KlingOmniPro"},
                 "3": {"class_type": "SaveVideo"}, "4": {"class_type": "AgentYHook"},
                 "5": {"class_type": "PrimitiveInt"}}
        self.assertFalse(ou.needs_card(graph, INFO))

    def test_one_local_node_does(self):
        graph = {"1": {"class_type": "KlingOmniPro"}, "2": {"class_type": "KSampler"}}
        self.assertTrue(ou.needs_card(graph, INFO))

    def test_what_is_not_known_does(self):
        self.assertTrue(ou.needs_card({"1": {"class_type": "SomePackUpscaler"}}, INFO))
        self.assertTrue(ou.needs_card({"1": {"inputs": {}}}, INFO))
        self.assertTrue(ou.needs_card({}, INFO))
        self.assertTrue(ou.needs_card(None, INFO))

    def test_release_leaves_the_model_for_a_graph_that_needs_no_card(self):
        ou._last = 0.0
        calls = []
        with mock.patch.object(ou, "_call", side_effect=lambda u, b=None: calls.append(u) or {"models": []}), \
                mock.patch.object(ou, "needs_card", return_value=False):
            self.assertEqual(ou.release("http://o", settings=LOCAL, prompt={"1": {"class_type": "KlingOmniPro"}}), [])
        self.assertEqual(calls, [])                     # Ollama was not even asked


class MakeRoom(unittest.TestCase):
    """ComfyUI lets go of the card before a local model works."""

    def setUp(self):
        ou._room_last = 0.0
        self.asked = []

    def _run(self, free_gb, queue=None, need_gb=25, settings=LOCAL):
        def comfy(path, body=None):
            self.asked.append(path)
            if path == "/system_stats":
                return {"devices": [{"type": "cuda", "vram_free": int(free_gb * GB)}]}
            if path == "/queue":
                return queue or {"queue_running": [], "queue_pending": []}
            return {}
        with mock.patch.object(ou, "_comfy", side_effect=comfy), \
                mock.patch("src.utils.ollama_window.need_bytes", return_value=int(need_gb * GB)):
            return ou.make_room("http://o", settings=settings)

    def test_it_asks_when_the_model_would_not_fit_beside_what_comfyui_holds(self):
        self.assertTrue(self._run(free_gb=5.7))                       # as measured: 26 GB held, 25 needed
        self.assertEqual(self.asked, ["/system_stats", "/queue", "/free"])

    def test_not_when_there_is_room(self):
        self.assertFalse(self._run(free_gb=30))
        self.assertEqual(self.asked, ["/system_stats"])

    def test_what_comfyui_calls_free_is_the_card_less_its_own(self):
        """So a model that is already loaded beside 10 GB of ComfyUI's is squeezed too."""
        self.assertTrue(self._run(free_gb=21.8))
        ou._room_last = 0.0
        self.assertFalse(self._run(free_gb=27.6))                     # as measured with the model loaded

    def test_never_while_comfyui_renders_or_has_something_waiting(self):
        self.assertFalse(self._run(free_gb=1, queue={"queue_running": [[1, "a"]], "queue_pending": []}))
        ou._room_last = 0.0
        self.assertFalse(self._run(free_gb=1, queue={"queue_running": [], "queue_pending": [[2, "b"]]}))
        self.assertNotIn("/free", self.asked)

    def test_not_for_hosted_models_and_not_with_the_switch_off(self):
        self.assertFalse(self._run(free_gb=1, settings=HOSTED))
        off = {"llm": {**LOCAL["llm"], "ollama": {"unload_before_render": False}}}
        self.assertFalse(self._run(free_gb=1, settings=off))
        self.assertEqual(self.asked, [])

    def test_it_is_not_asked_again_at_once(self):
        self.assertTrue(self._run(free_gb=1))
        self.assertFalse(self._run(free_gb=1))
        self.assertEqual(self.asked.count("/system_stats"), 1)

    def test_a_comfyui_that_does_not_answer_costs_nothing(self):
        with mock.patch.object(ou, "_comfy", side_effect=OSError("down")):
            self.assertFalse(ou.make_room("http://o", settings=LOCAL))

    def test_a_machine_without_a_card_is_left_alone(self):
        with mock.patch.object(ou, "_comfy", return_value={"devices": [{"type": "cpu", "vram_free": 0}]}):
            self.assertFalse(ou.make_room("http://o", settings=LOCAL))


if __name__ == "__main__":
    unittest.main()
