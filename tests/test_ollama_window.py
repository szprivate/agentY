"""The window a local model is loaded with is worked out from the graphics card.

``llm.ollama.num_ctx = 0`` (the default) means: the largest window with which
the largest local model the agents use still fits the card. A number in the
setting is used as it is.
"""

import unittest
from unittest import mock

from src.utils import ollama_window as ow

GB = 2 ** 30
# qwen3.8:27b as Ollama describes it: measured at 23.0 / 25.2 / 27.5 GB for 65,536 / 98,304 / 131,072.
QWEN = {"size": 17_741_872_154, "per_token": 2 * 65 * 5120 * 4 / 24 * 2 * ow.CACHE_FACTOR, "limit": 262_144}


class TheArithmetic(unittest.TestCase):

    def test_it_matches_what_was_measured(self):
        for window, measured in ((65_536, 23.0), (98_304, 25.2), (131_072, 27.5)):
            need = (QWEN["size"] + ow.OVERHEAD + window * QWEN["per_token"]) / GB
            self.assertAlmostEqual(need, measured, delta=0.6, msg=str(window))

    def test_the_largest_window_that_fits_the_card(self):
        self.assertEqual(ow.fit(32 * GB, QWEN), 98_304)          # 131,072 would need 27.8 of the 27.2 it may take
        self.assertEqual(ow.fit(40 * GB, QWEN), 196_608)
        self.assertEqual(ow.fit(48 * GB, QWEN), 262_144)         # the model's own window, where the card has room
        self.assertEqual(ow.fit(24 * GB, QWEN), 32_768)          # it does not fit at all: the smallest, not nothing

    def test_never_beyond_what_the_model_itself_takes(self):
        self.assertEqual(ow.fit(48 * GB, dict(QWEN, limit=40_000)), 32_768)

    def test_a_small_model_gets_the_whole_ladder(self):
        small = {"size": 5 * GB, "per_token": 30_000, "limit": 262_144}
        self.assertEqual(ow.fit(24 * GB, small), 262_144)
        self.assertEqual(ow.fit(24 * GB, dict(small, limit=131_072)), 131_072)   # never beyond the model's own


class WhatItComesTo(unittest.TestCase):

    def setUp(self):
        ow.forget()
        self.addCleanup(ow.forget)

    def _machine(self, card=32 * GB, models=("qwen3.8:27b",), facts=QWEN):
        return (mock.patch.object(ow, "card_bytes", return_value=card),
                mock.patch.object(ow, "local_models", return_value=list(models)),
                mock.patch.object(ow, "model_facts", return_value=facts))

    def test_zero_is_worked_out(self):
        a, b, c = self._machine()
        with a, b, c:
            self.assertEqual(ow.resolve(0), 98_304)
            self.assertEqual(ow.resolve("0"), 98_304)
            self.assertIn("qwen3.8:27b", ow.auto()["why"])

    def test_a_number_is_used_as_it_is(self):
        a, b, c = self._machine()
        with a, b, c:
            self.assertEqual(ow.resolve(40_000), 40_000)
            self.assertEqual(ow.resolve("65536"), 65_536)

    def test_it_is_measured_once(self):
        a, b, c = self._machine()
        with a as card, b, c:
            ow.resolve(0)
            ow.resolve(0)
            self.assertEqual(card.call_count, 1)

    def test_the_largest_local_model_decides(self):
        small = {"size": 5 * GB, "per_token": 30_000, "limit": 262_144}
        with mock.patch.object(ow, "card_bytes", return_value=32 * GB), \
                mock.patch.object(ow, "local_models", return_value=["tiny", "qwen3.8:27b"]), \
                mock.patch.object(ow, "model_facts", side_effect=lambda m, h: small if m == "tiny" else QWEN):
            self.assertEqual(ow.auto()["window"], 98_304)
            self.assertEqual(ow.auto()["model"], "qwen3.8:27b")

    def test_with_nothing_to_measure_it_is_a_window_the_orchestrator_fits(self):
        for kw in ({"card": 0}, {"models": ()}, {"facts": None}):
            ow.forget()
            a, b, c = self._machine(**kw)
            with a, b, c:
                got = ow.auto()
            self.assertEqual(got["window"], ow.FALLBACK, kw)
            self.assertTrue(got["why"], kw)

    def test_the_local_models_are_read_from_tiers_and_overrides(self):
        settings = {"llm": {"tiers": {"orchestrator": "ollama,qwen3.8:27b", "vision": "dashscope,qwen-vl"},
                            "pipeline": {"learnings": "ollama, tiny", "planner": ""}}}
        with mock.patch("src.utils.settings.load_settings", return_value=settings):
            self.assertEqual(ow.local_models(), ["qwen3.8:27b", "tiny"])

    def test_the_default_setting_is_zero(self):
        import tomllib
        from pathlib import Path
        root = Path(__file__).resolve().parent.parent
        with open(root / "config" / "settings.default.toml", "rb") as fh:
            cfg = tomllib.load(fh)
        self.assertEqual(cfg["llm"]["ollama"]["num_ctx"], 0)
        self.assertEqual(cfg["likeness"]["face_model"], "buffalo_l")


if __name__ == "__main__":
    unittest.main()
