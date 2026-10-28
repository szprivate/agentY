"""Strata as a provider, and one GPU for it and ComfyUI (src/utils/strata.py).

Nothing here starts Strata or talks to ComfyUI: the server, the stop/start
commands and ComfyUI's queue are stubbed.
"""
import asyncio
import threading
import time
import tomllib
import unittest
from pathlib import Path
from unittest import mock

from src.utils import strata as st

ROOT = Path(__file__).resolve().parents[1]
USED = {"llm": {"tiers": {"orchestrator": "strata,qwen3.6-35b-a3b-ud-q4_k_xl"}, "strata": {}}}
UNUSED = {"llm": {"tiers": {"orchestrator": "dashscope,qwen3.8-max"}, "pipeline": {"coder": ""}}}
GRAPH = {"1": {"class_type": "KSampler", "inputs": {}}}


class _GateReset(unittest.TestCase):

    def setUp(self):
        st.GATE = st._Gate()
        self._built = st._built
        st._built = False
        self.events = []
        self.up = True

    def tearDown(self):
        st._built = self._built
        st.GATE = st._Gate()

    def _stubs(self, settings, *, busy=lambda: False, card=True):
        def stop(*a, **k):
            self.events.append("stop")
            self.up = False
            return True

        def start(*a, **k):
            self.events.append("start")
            self.up = True
            return 1.0

        return [
            mock.patch.object(st, "_settings", return_value=settings),
            mock.patch.object(st, "alive", side_effect=lambda *a, **k: self.up),
            mock.patch.object(st, "stop", side_effect=stop),
            mock.patch.object(st, "start", side_effect=start),
            mock.patch.object(st, "comfy_busy", side_effect=busy),
            mock.patch.object(st, "comfy_free", side_effect=lambda: self.events.append("free")),
            mock.patch.object(st, "_status", side_effect=lambda text, level="info": None),
            mock.patch("src.utils.ollama_unload.needs_card", return_value=card),
            mock.patch.object(st, "_QUEUE_POLL", 0.01),
        ]

    def _enter(self, patches):
        for p in patches:
            p.start()
            self.addCleanup(p.stop)


class Settings(unittest.TestCase):

    def test_the_default_settings_carry_a_strata_block(self):
        cfg = tomllib.loads((ROOT / "config" / "settings.default.toml").read_text(encoding="utf-8"))
        block = cfg["llm"]["strata"]
        self.assertEqual(block["base_url"], "http://127.0.0.1:8080/v1")
        self.assertIs(block["gpu_sharing"], True)
        self.assertIs(block["think"], False)
        self.assertIs(block["vision"], True)
        self.assertIn("--parallel 1", block["launch_args"])

    def test_config_defaults_and_env_overrides(self):
        with mock.patch.dict("os.environ", {"STRATA_BASE_URL": "", "STRATA_API_KEY": ""}):
            cfg = st.config({"llm": {}})
        self.assertEqual(cfg["base_url"], st.DEFAULT_URL)
        self.assertEqual(cfg["api_key"], "none")            # the OpenAI client needs one
        self.assertTrue(cfg["gpu_sharing"])
        with mock.patch.dict("os.environ", {"STRATA_BASE_URL": "http://box:9000/v1/", "STRATA_API_KEY": "k"}):
            cfg = st.config({"llm": {"strata": {"gpu_sharing": False}}})
        self.assertEqual(cfg["base_url"], "http://box:9000/v1")
        self.assertEqual(cfg["api_key"], "k")
        self.assertFalse(cfg["gpu_sharing"])

    def test_thinking_is_off_unless_asked(self):
        self.assertEqual(st.extra_body(False, {"llm": {}}),
                         {"chat_template_kwargs": {"enable_thinking": False}})
        self.assertEqual(st.extra_body(True, {"llm": {}}),
                         {"chat_template_kwargs": {"enable_thinking": True}})
        self.assertEqual(st.extra_body(False, {"llm": {"strata": {"think": True}}}),
                         {"chat_template_kwargs": {"enable_thinking": True}})

    def test_extra_body_from_the_settings_is_laid_over(self):
        cfg = {"llm": {"strata": {"extra_body": {"top_k": 20, "chat_template_kwargs": {"x": 1}}}}}
        self.assertEqual(st.extra_body(False, cfg),
                         {"chat_template_kwargs": {"enable_thinking": False, "x": 1}, "top_k": 20})

    def test_in_use_only_when_a_role_runs_on_strata(self):
        with mock.patch.object(st, "_built", False):
            self.assertTrue(st.in_use(USED))
            self.assertFalse(st.in_use(UNUSED))
            self.assertTrue(st.in_use({"llm": {"pipeline": {"coder": "Strata,x"}}}))
        with mock.patch.object(st, "_built", True):     # a conversation switched to it
            self.assertTrue(st.in_use(UNUSED))

    def test_the_launcher_keeps_windows_paths(self):
        with mock.patch.object(st, "_settings", return_value={"llm": {"strata": {
                "launcher": __file__, "launch_args": r"launch --data-dir D:\AI\models --parallel 1"}}}):
            argv = st._command(st.config()["launch_args"])
        self.assertIn(r"D:\AI\models", argv)
        self.assertEqual(argv[-2:], ["--parallel", "1"])

    def test_the_name_is_reserved_for_custom_providers(self):
        from src.utils import custom_providers as cp
        self.assertIn("strata", cp.reserved())
        self.assertTrue(cp.validate("strata", {"key_env": "A_KEY", "url_env": "A_URL"}))


class ModelFactory(unittest.TestCase):

    def setUp(self):
        saved = st._built
        self.addCleanup(setattr, st, "_built", saved)

    def _build(self, think=False):
        from src import agent as A
        with mock.patch.object(st, "_settings", return_value={"llm": {"strata": {}}}), \
                mock.patch.object(A, "role_thinking", return_value=think), \
                mock.patch.dict("os.environ", {"STRATA_THINK": "", "STRATA_BASE_URL": "", "STRATA_API_KEY": ""}):
            return A._build_model(role="orchestrator", llm="strata", system_prompt="hi",
                                  dashscope_model="qwen3.6-35b-a3b-ud-q4_k_xl")

    def test_strata_tier_builds_an_openai_client_with_thinking_off(self):
        model, model_id = self._build()
        self.assertEqual(model_id, "qwen3.6-35b-a3b-ud-q4_k_xl")
        cfg = model.get_config()
        self.assertEqual(cfg["params"]["extra_body"], {"chat_template_kwargs": {"enable_thinking": False}})
        self.assertEqual(model.client_args["base_url"], "http://127.0.0.1:8080/v1")
        self.assertGreaterEqual(model.client_args["timeout"], 1800)   # queued behind --parallel 1
        self.assertEqual(model.provider, "strata")
        self.assertTrue(st._built)

    def test_a_thinking_role_turns_it_on(self):
        model, _ = self._build(think=True)
        self.assertTrue(model.get_config()["params"]["extra_body"]["chat_template_kwargs"]["enable_thinking"])

    def test_a_strata_agent_costs_nothing(self):
        from src import agent as A
        from src.utils import costs
        model, model_id = self._build()
        agent = A._wrap_agent(role="orchestrator", llm="strata", model=model, model_id=model_id,
                              system_prompt="hi", tools=[])
        self.assertTrue(agent._cost_meta["is_ollama"])
        self.assertEqual(costs.get_model_prices_for(agent), (0.0, 0.0))

        class _Meta:
            _cost_meta = {"provider": "strata", "model_id": "x", "is_ollama": False}
        self.assertEqual(costs.get_model_prices_for(_Meta()), (0.0, 0.0))

    def test_cached_tokens_are_read_from_the_usage(self):
        model, _ = self._build()
        chunk = model.format_chunk({"chunk_type": "metadata", "data": mock.Mock(
            prompt_tokens=1000, completion_tokens=50, total_tokens=1050,
            prompt_tokens_details=mock.Mock(cached_tokens=800), completion_tokens_details=None,
            model_extra={})})
        usage = chunk["metadata"]["usage"]
        self.assertEqual(usage["inputTokens"], 1000)
        self.assertEqual(usage.get("cacheReadInputTokens"), 800)

    def test_strata_reads_images(self):
        from src.utils.vision_capability import supports_vision
        with mock.patch.object(st, "_settings", return_value={"llm": {"strata": {}}}):
            self.assertTrue(supports_vision("strata,qwen3.6-35b-a3b-ud-q4_k_xl"))
        with mock.patch.object(st, "_settings", return_value={"llm": {"strata": {"vision": False}}}):
            self.assertFalse(supports_vision("strata,some-text-model"))

    def test_a_custom_provider_sends_its_extra_body(self):
        from src import agent as A
        from src.utils import custom_providers as cp
        reg = {"vllm": {"label": "vLLM", "key_env": "VLLM_KEY", "url_env": "VLLM_URL", "vision": False,
                        "extra_body": {"chat_template_kwargs": {"enable_thinking": False}}}}
        with mock.patch.object(cp, "load", return_value={k: cp.normalize(k, v) for k, v in reg.items()}), \
                mock.patch.dict("os.environ", {"VLLM_KEY": "k", "VLLM_URL": "http://h/v1"}), \
                mock.patch.object(A, "role_thinking", return_value=False):
            model, _ = A._build_model(role="info", llm="vllm", system_prompt="", dashscope_model="m")
        self.assertEqual(model.get_config()["params"]["extra_body"],
                         {"chat_template_kwargs": {"enable_thinking": False}})


class GpuSharing(_GateReset):

    def test_a_render_stops_strata_when_it_is_used(self):
        self._enter(self._stubs(USED))
        self.assertTrue(st.yield_gpu(GRAPH))
        self.assertEqual(self.events, ["stop"])

    def test_nothing_happens_when_strata_is_not_used(self):
        self._enter(self._stubs(UNUSED))
        self.assertFalse(st.yield_gpu(GRAPH))
        self.assertEqual(self.events, [])

    def test_nothing_happens_with_gpu_sharing_off(self):
        cfg = {"llm": {**USED["llm"], "strata": {"gpu_sharing": False}}}
        self._enter(self._stubs(cfg))
        self.assertFalse(st.yield_gpu(GRAPH))
        self.assertEqual(self.events, [])

    def test_a_graph_that_needs_no_card_leaves_it_running(self):
        self._enter(self._stubs(USED, card=False))
        self.assertFalse(st.yield_gpu(GRAPH))
        self.assertEqual(self.events, [])

    def test_a_strata_that_is_not_running_is_not_stopped(self):
        self._enter(self._stubs(USED))
        self.up = False
        self.assertFalse(st.yield_gpu(GRAPH))
        self.assertEqual(self.events, [])

    def test_the_next_call_starts_it_again_after_the_render(self):
        polls = iter([True, True, False, False, False])
        self._enter(self._stubs(USED, busy=lambda: next(polls, False)))
        st.yield_gpu(GRAPH)
        st.GATE.acquire()
        self.assertEqual(self.events, ["stop", "free", "start"])
        self.assertEqual(st.GATE.inflight, 1)
        st.GATE.release()
        self.assertEqual(st.GATE.inflight, 0)

    def test_a_running_strata_lets_the_call_through(self):
        self._enter(self._stubs(USED))
        st.GATE.acquire()
        st.GATE.release()
        self.assertEqual(self.events, [])

    def test_the_stop_waits_for_the_call_in_flight(self):
        self._enter(self._stubs(USED))
        st.GATE.acquire()                       # a call is being answered
        done = threading.Event()
        t = threading.Thread(target=lambda: (st.yield_gpu(GRAPH), done.set()))
        t.start()
        time.sleep(0.3)
        self.assertEqual(self.events, [])       # not cut off
        self.assertFalse(done.is_set())
        st.GATE.release()
        t.join(5)
        self.assertEqual(self.events, ["stop"])

    def test_a_call_waits_while_a_graph_is_being_queued(self):
        self._enter(self._stubs(USED))
        self.up = False                          # stopped for a render ...
        st.GATE.submitting = 1                   # ... whose POST has not landed yet
        got = threading.Event()
        t = threading.Thread(target=lambda: (st.GATE.acquire(), got.set()))
        t.start()
        time.sleep(0.3)
        self.assertFalse(got.is_set())
        self.assertNotIn("start", self.events)
        with st.GATE.cond:
            st.GATE.submitting = 0
        t.join(5)
        self.assertTrue(got.is_set())
        self.assertEqual(self.events, ["free", "start"])
        st.GATE.release()

    def test_without_autostart_a_stopped_strata_is_an_error(self):
        cfg = {"llm": {**USED["llm"], "strata": {"autostart": False}}}
        self._enter(self._stubs(cfg))
        self.up = False
        with self.assertRaises(RuntimeError):
            st.GATE.acquire()

    def test_every_queued_graph_passes_the_hook(self):
        from agenty_core.utils import comfyui_client as cc
        self._enter(self._stubs(USED))
        st.install()
        self.assertIn(st._prompt_hook, cc._prompt_hooks)

        class _Resp:
            status_code = 200

            def json(self):
                return {"prompt_id": "p"}

            def raise_for_status(self):
                pass

        def post(*a, **k):
            self.events.append("post")
            self.assertEqual(st.GATE.submitting, 1)
            return _Resp()

        cc.forget_down()
        with mock.patch.object(cc.requests, "post", side_effect=post):
            cc.ComfyUIClient(base_url="http://127.0.0.1:1", api_key="").post("/prompt", json_data={"prompt": GRAPH})
        self.assertEqual(self.events, ["stop", "post"])
        self.assertEqual(st.GATE.submitting, 0)

    def test_the_browser_queue_route_is_a_no_op_when_unused(self):
        self._enter(self._stubs(UNUSED))
        self.assertEqual(st.yield_for_browser_queue(hold=0.01), {"used": False, "stopped": False})
        self.assertEqual(self.events, [])


class TheModelCall(_GateReset):

    def test_the_call_passes_the_gate_and_lets_go_after(self):
        self._enter(self._stubs(USED))
        self.up = False                          # stopped for a render that has finished
        with mock.patch.dict("os.environ", {"STRATA_BASE_URL": "", "STRATA_API_KEY": ""}):
            model = st.build_model("m", settings={"llm": {}})

        async def fake_stream(self_, *a, **k):
            events_ref.append(f"stream inflight={st.GATE.inflight}")
            yield {"x": 1}

        events_ref = self.events
        from strands.models.openai import OpenAIModel
        with mock.patch.object(OpenAIModel, "stream", fake_stream):
            async def run():
                return [e async for e in model.stream([])]
            got = asyncio.run(run())
        self.assertEqual(got, [{"x": 1}])
        self.assertEqual(self.events, ["free", "start", "stream inflight=1"])
        self.assertEqual(st.GATE.inflight, 0)


if __name__ == "__main__":
    unittest.main()
