"""Costs the provider states beat estimates, and the overview says which is which.

Provider facts these fakes follow (checked against the docs 2026-10):
* OpenRouter puts ``usage.cost`` (USD) on every response; the others (Anthropic,
  OpenAI, Gemini, DashScope) return token counts only.
* OpenAI / DashScope / OpenRouter count reasoning tokens inside completion_tokens;
  Gemini's OpenAI-compatible endpoint only inside total_tokens.
* Anthropic and OpenAI admin cost APIs are organisation-level and need admin keys.

    python -m unittest discover -s tests
"""
import os
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

from src.utils import cost_reconcile, usage_capture
from src.utils.costs import compute_cost_from_usage, cost_for_call


class _Model:
    def __init__(self, provider, model_id):
        self._cost_meta = {"provider": provider, "model_id": model_id,
                           "is_ollama": provider == "ollama"}


def _usage(prompt, completion, total=None, cached=0, cost=None, reasoning=0):
    u = types.SimpleNamespace(
        prompt_tokens=prompt, completion_tokens=completion,
        total_tokens=prompt + completion if total is None else total,
        prompt_tokens_details=types.SimpleNamespace(cached_tokens=cached),
        completion_tokens_details=types.SimpleNamespace(reasoning_tokens=reasoning))
    if cost is not None:
        u.cost = cost
    return u


class Extraction(unittest.TestCase):
    def test_cost_is_read_from_object_extra_and_dict(self):
        self.assertEqual(usage_capture.reported_cost(_usage(1, 1, cost=0.0123)), 0.0123)
        sdk_like = types.SimpleNamespace(model_extra={"cost": 0.5})
        self.assertEqual(usage_capture.reported_cost(sdk_like), 0.5)
        self.assertEqual(usage_capture.reported_cost({"cost": "0.25"}), 0.25)

    def test_providers_that_state_no_cost_give_none(self):
        self.assertIsNone(usage_capture.reported_cost(_usage(10, 5)))
        self.assertIsNone(usage_capture.reported_cost({"cost": None}))
        self.assertIsNone(usage_capture.reported_cost({"cost": "n/a"}))

    def test_gemini_thoughts_are_the_gap_in_total_tokens(self):
        self.assertEqual(usage_capture.thoughts_outside_completion(_usage(100, 20, total=170)), 50)
        self.assertEqual(usage_capture.thoughts_outside_completion(_usage(100, 20)), 0)


def _model(extra=None):
    base = usage_capture.capturing_openai_model()
    m = base.__new__(base)          # no client needed to format a chunk
    for k, v in (extra or {}).items():
        setattr(m, k, v)
    return m


class Capture(unittest.TestCase):
    def test_openrouter_style_cost_is_accumulated_per_call(self):
        m = _model()
        for c in (0.01, 0.02):
            chunk = m.format_chunk({"chunk_type": "metadata", "data": _usage(1000, 100, cached=400, cost=c)})
        self.assertAlmostEqual(m.reported_cost_total, 0.03)
        self.assertEqual((m.usage_calls, m.cost_calls), (2, 2))
        self.assertEqual(chunk["metadata"]["usage"]["cacheReadInputTokens"], 400)

    def test_a_call_without_cost_counts_but_adds_nothing(self):
        m = _model()
        m.format_chunk({"chunk_type": "metadata", "data": _usage(10, 5)})
        self.assertEqual((m.usage_calls, m.cost_calls, m.reported_cost_total), (1, 0, 0.0))

    def test_gemini_thinking_tokens_are_billed_as_output(self):
        m = _model({"thoughts_in_total": True})
        chunk = m.format_chunk({"chunk_type": "metadata", "data": _usage(100, 20, total=170)})
        self.assertEqual(chunk["metadata"]["usage"]["outputTokens"], 70)

    def test_openai_reasoning_is_already_in_completion(self):
        m = _model({"thoughts_in_total": True})   # even with the rule on: reasoning_tokens present
        chunk = m.format_chunk({"chunk_type": "metadata", "data": _usage(100, 80, reasoning=60)})
        self.assertEqual(chunk["metadata"]["usage"]["outputTokens"], 80)
        self.assertEqual(m.reasoning_total, 60)


class Sources(unittest.TestCase):
    U = {"inputTokens": 100_000, "outputTokens": 1000, "cacheReadInputTokens": 50_000}

    def test_reported_wins_and_says_so(self):
        cost, _t, src = cost_for_call(self.U, _Model("openai", "gpt-4o"), reported=0.42)
        self.assertEqual((cost, src), (0.42, "reported"))

    def test_without_a_stated_cost_it_is_an_estimate(self):
        cost, _t, src = cost_for_call(self.U, _Model("openai", "gpt-4o"))
        self.assertEqual(src, "estimated")
        self.assertAlmostEqual(cost, (50_000 * 2.5 + 50_000 * 2.5 * 0.5 + 1000 * 10) / 1e6)

    def test_local_models_are_free(self):
        self.assertEqual(cost_for_call(self.U, _Model("ollama", "qwen3"))[2], "free")

    def test_cached_input_rates_follow_the_provider(self):
        def cached_cost(provider, model):
            usage = {"inputTokens": 1_000_000, "cacheReadInputTokens": 1_000_000}
            return compute_cost_from_usage(usage, _Model(provider, model))[0]
        self.assertAlmostEqual(cached_cost("openai", "gpt-4o"), 1.25)             # 50% of 2.50
        self.assertAlmostEqual(cached_cost("openai", "o3"), 2.5)                  # 25% of 10
        self.assertAlmostEqual(cached_cost("google", "gemini-2.5-flash"), 0.03)   # 10% of 0.30
        self.assertAlmostEqual(cached_cost("google", "gemini-2.0-flash"), 0.025)  # 25% of 0.10


class Overview(unittest.TestCase):
    def test_log_lines_are_marked_reported_or_estimated(self):
        from src.utils import agentY_server as srv
        tail = "  total=1000in/100out/0cache_read/0cache_write"
        lines = [
            "2026-10-19 10:00:00 [orchestrator] tool=a delta=+1000in/+100out/+0cache_read/+0cache_write"
            + tail + " cost=$0.00/tokens=1100 src=reported dcost=0.004000 model=openai/or-model\n",
            "2026-10-19 10:00:01 [orchestrator] tool=b delta=+1000in/+100out/+0cache_read/+0cache_write"
            + tail + " cost=$0.01/tokens=2200 src=estimated model=claude/claude-haiku-4-5\n",
            # an old line with no src= is an estimate
            "2026-10-19 10:00:02 [orchestrator] tool=c delta=+1000in/+0out/+0cache_read/+0cache_write"
            + tail + " model=dashscope/qwen3.8-flash\n",
            "2026-10-19 10:00:03 [orchestrator] tool=d delta=+500in/+50out/+0cache_read/+0cache_write"
            + tail + " src=free model=ollama/qwen3\n",
        ]
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "t.log").write_text("".join(lines), encoding="utf-8")
            with mock.patch.object(srv, "_project_root", return_value=Path(d)), \
                 mock.patch("src.agent._load_settings", return_value={"tokens_usage_log": "t.log"}):
                res = srv._aggregate_token_usage(lambda ts: True)
        rows = {r["model"]: r for r in res["rows"]}
        self.assertEqual(rows["openai/or-model"]["cost_source"], "reported")
        self.assertAlmostEqual(rows["openai/or-model"]["cost"], 0.004)
        self.assertEqual(rows["claude/claude-haiku-4-5"]["cost_source"], "estimated")
        self.assertEqual(rows["dashscope/qwen3.8-flash"]["cost_source"], "estimated")
        self.assertEqual(rows["ollama/qwen3"]["cost_source"], "free")
        t = res["total"]
        self.assertEqual(t["cost_source"], "mixed")
        self.assertAlmostEqual(t["cost_reported"], 0.004)
        self.assertAlmostEqual(t["cost"], t["cost_reported"] + t["cost_estimated"], places=4)


class Reconcile(unittest.TestCase):
    def _get(self, url, params, headers):
        if "anthropic" in url:
            assert "x-api-key" in headers
            if params.get("page") is None:
                return {"data": [{"results": [{"amount": "123.45", "currency": "USD"}]}],
                        "has_more": True, "next_page": "p2"}
            return {"data": [{"results": [{"amount": "100", "currency": "USD"},
                                          {"amount": "5", "currency": "EUR"}]}],
                    "has_more": False, "next_page": None}
        assert headers["Authorization"].startswith("Bearer ")
        return {"data": [{"results": [{"amount": {"value": 0.06, "currency": "usd"}},
                                      {"amount": {"value": 0.04, "currency": "usd"}}]}],
                "has_more": False}

    def test_off_by_default(self):
        with mock.patch.object(cost_reconcile, "enabled", return_value=False):
            self.assertFalse(cost_reconcile.reconcile(1, 2)["enabled"])

    def test_both_providers_with_admin_keys(self):
        env = {"ANTHROPIC_ADMIN_KEY": "k1", "OPENAI_ADMIN_KEY": "k2"}
        with mock.patch.dict(os.environ, env):
            out = cost_reconcile.reconcile(1_700_000_000, 1_700_600_000, get=self._get, force=True)
        self.assertAlmostEqual(out["providers"]["anthropic"]["cost"], 2.2345)   # cents -> USD, EUR ignored
        self.assertAlmostEqual(out["providers"]["openai"]["cost"], 0.10)
        self.assertEqual(out["skipped"], {})

    def test_missing_key_is_skipped_and_failures_do_not_leak_the_key(self):
        def boom(*a, **k):
            raise RuntimeError("sk-secret")
        with mock.patch.dict(os.environ, {"ANTHROPIC_ADMIN_KEY": "sk-secret"}):
            os.environ.pop("OPENAI_ADMIN_KEY", None)
            os.environ.pop("OPENAI_ADMIN_API_KEY", None)
            out = cost_reconcile.reconcile(1, 2, get=boom, force=True)
        self.assertIn("openai", out["skipped"])
        self.assertNotIn("sk-secret", str(out))


class Hook(unittest.TestCase):
    def test_hook_logs_the_stated_cost(self):
        from src.agent import TokenUsageHookProvider
        m = _model()
        m.format_chunk({"chunk_type": "metadata", "data": _usage(1000, 100, cost=0.007)})
        m._cost_meta = {"provider": "openai", "model_id": "x", "is_ollama": False}
        agent = types.SimpleNamespace(
            model=m, event_loop_metrics=types.SimpleNamespace(accumulated_usage={
                "inputTokens": 1000, "outputTokens": 100,
                "cacheReadInputTokens": 0, "cacheWriteInputTokens": 0}))
        with tempfile.TemporaryDirectory() as d:
            hook = TokenUsageHookProvider(role="orchestrator")
            hook._log_path = Path(d) / "log.txt"
            ev = types.SimpleNamespace(agent=agent, tool_use={"name": "t"})
            with mock.patch("builtins.print"):
                hook._on_after_tool_call(ev)
            line = hook._log_path.read_text(encoding="utf-8")
        self.assertIn("src=reported dcost=0.007000", line)


if __name__ == "__main__":
    unittest.main()
