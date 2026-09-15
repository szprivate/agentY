"""Cost estimates that agree with the Alibaba Cloud bill.

The cost panel showed about ten times what the console billed: $35 for five days
the console put at roughly $3. Two causes, both pinned here:

* DashScope reports cached tokens INSIDE inputTokens (OpenAI-style prompt_tokens),
  and they were priced as if reported beside it, like Anthropic's: every cache
  hit was charged at the full input price and then again at 0.1x. With 90% of
  agentY's input served from cache, that more than quadrupled the figure.
* qwen3.8-flash was not in the price table and fell back to the qwen-plus-class
  default ($0.40 / $1.20), and the table held International rates where the
  console bills Global-deployment ones - $0.109 / $0.368 for qwen3.8-flash.

    python -m unittest discover -s tests
"""
import unittest

from src.utils.costs import compute_cost_from_usage, get_model_prices_for


class _Model:
    def __init__(self, provider, model_id):
        self._cost_meta = {"provider": provider, "model_id": model_id,
                           "is_ollama": provider == "ollama"}


FLASH = _Model("dashscope", "qwen3.8-flash")
MILLION = 1_000_000


def usage(inp, out, cache_read=0, cache_write=0):
    return {"inputTokens": inp, "outputTokens": out,
            "cacheReadInputTokens": cache_read, "cacheWriteInputTokens": cache_write}


class Prices(unittest.TestCase):

    def test_qwen3_8_flash_has_its_own_global_rate(self):
        inp, out = get_model_prices_for(FLASH)
        self.assertAlmostEqual(inp * MILLION, 0.109)
        self.assertAlmostEqual(out * MILLION, 0.368)

    def test_the_other_models_on_the_bill_are_priced(self):
        for model in ("qwen3.8-max", "qwen3.8-27b", "qwen3.7-plus", "qwen3-vl-flash"):
            inp, _out = get_model_prices_for(_Model("dashscope", model))
            # Not the qwen-plus-class fallback every unknown Qwen model got.
            self.assertNotAlmostEqual(inp * MILLION, 0.40, msg=model)


class CachedTokens(unittest.TestCase):

    def test_a_dashscope_cache_hit_is_billed_once_at_a_fifth_of_the_input_price(self):
        cost, tokens = compute_cost_from_usage(usage(100_000, 1_000, cache_read=90_000), FLASH)
        expected = (10_000 * 0.109 + 90_000 * 0.109 * 0.2 + 1_000 * 0.368) / MILLION
        self.assertAlmostEqual(cost, expected)
        self.assertEqual(tokens, 101_000)       # cached tokens are part of the 100k

    def test_anthropic_still_reports_cache_beside_input(self):
        sonnet = _Model("anthropic", "claude-sonnet-4-6")
        cost, tokens = compute_cost_from_usage(usage(10_000, 1_000, cache_read=90_000), sonnet)
        expected = (10_000 * 3.00 + 1_000 * 15.00 + 90_000 * 3.00 * 0.1) / MILLION
        self.assertAlmostEqual(cost, expected)
        self.assertEqual(tokens, 101_000)

    def test_a_breakdown_larger_than_the_input_is_read_as_separate(self):
        # Never subtract into negative fresh input.
        cost, _ = compute_cost_from_usage(usage(1_000, 0, cache_read=5_000), FLASH)
        self.assertGreater(cost, 0)

    def test_ollama_is_free(self):
        cost, _ = compute_cost_from_usage(usage(100_000, 1_000, cache_read=90_000),
                                          _Model("ollama", "qwen3:8b"))
        self.assertEqual(cost, 0)


class AgreesWithTheConsole(unittest.TestCase):
    """Token counts from agentY's log for whole console days (Beijing midnight to
    midnight), against what the console billed for qwen3.8-flash on those days."""

    DAYS = {
        # day: (input incl. cache, cache hits, output, console USD)
        "2026-09-11": (7_220_000, 6_000_000, 50_000, 0.32),
        "2026-09-14": (30_660_000, 28_410_000, 140_000, 0.865),
        "2026-09-15": (10_910_000, 9_670_000, 70_000, 0.375),
    }

    def test_each_day_lands_within_fifteen_percent(self):
        for day, (inp, hits, out, billed) in self.DAYS.items():
            cost, _ = compute_cost_from_usage(usage(inp, out, cache_read=hits), FLASH)
            self.assertLess(abs(cost - billed) / billed, 0.15, f"{day}: ${cost:.3f} vs ${billed}")

    def test_the_old_calculation_was_an_order_of_magnitude_off(self):
        inp, hits, out, billed = self.DAYS["2026-09-14"]
        old = inp * 0.40 / MILLION + out * 1.20 / MILLION + hits * 0.40 / MILLION * 0.1
        self.assertGreater(old / billed, 10)


if __name__ == "__main__":
    unittest.main()
