from __future__ import annotations
import os
from typing import Dict, Optional, Tuple

# ---------------------------------------------------------------------------
# Utilities to compute token costs for runs.
#
# Pricing data sourced from official pricing pages (verified April 2026):
#   Anthropic : https://platform.claude.com/docs/en/about-claude/pricing
#   OpenAI    : https://developers.openai.com/api/docs/pricing
#   Google    : https://ai.google.dev/gemini-api/docs/pricing
#
# Rules:
#   - Ollama models are free (cost per token = 0).
#   - Lookup order: exact model-id -> prefix match -> env var -> hardcoded default.
#   - Env var overrides (prices in USD per single token):
#       COST_INPUT_TOKEN_<MODEL>         e.g. COST_INPUT_TOKEN_CLAUDE_SONNET_4_6
#       COST_OUTPUT_TOKEN_<MODEL>
#       COST_INPUT_TOKEN_PROVIDER_<P>    e.g. COST_INPUT_TOKEN_PROVIDER_ANTHROPIC
#       COST_OUTPUT_TOKEN_PROVIDER_<P>
# ---------------------------------------------------------------------------


def _mtok(inp: float, out: float) -> tuple:
    """Convert per-million-token prices (USD) to per-single-token prices."""
    return (inp / 1_000_000, out / 1_000_000)


# ---------------------------------------------------------------------------
# Anthropic / Claude
# Source: https://platform.claude.com/docs/en/about-claude/pricing  (Apr 2026)
# ---------------------------------------------------------------------------
ANTHROPIC_PRICES: Dict[str, Tuple[float, float]] = {
    # Claude 4.x
    "claude-opus-4-6":        _mtok( 5.00,  25.00),
    "claude-opus-4-5":        _mtok( 5.00,  25.00),
    "claude-opus-4-1":        _mtok(15.00,  75.00),
    "claude-opus-4":          _mtok(15.00,  75.00),
    "claude-sonnet-4-6":      _mtok( 3.00,  15.00),
    "claude-sonnet-4-5":      _mtok( 3.00,  15.00),
    "claude-sonnet-4":        _mtok( 3.00,  15.00),
    "claude-haiku-4-5":       _mtok( 1.00,   5.00),
    # Claude 3.x
    "claude-sonnet-3-7":      _mtok( 3.00,  15.00),
    "claude-haiku-3-5":       _mtok( 0.80,   4.00),
    "claude-opus-3":          _mtok(15.00,  75.00),
    "claude-haiku-3":         _mtok( 0.25,   1.25),
    # Legacy API aliases (snapshot-dated IDs matched via prefix fallback)
    "claude-3-5-sonnet":      _mtok( 3.00,  15.00),
    "claude-3-5-haiku":       _mtok( 0.80,   4.00),
    "claude-3-opus":          _mtok(15.00,  75.00),
    "claude-3-haiku":         _mtok( 0.25,   1.25),
}

# ---------------------------------------------------------------------------
# OpenAI / GPT
# Source: https://developers.openai.com/api/docs/pricing  (Apr 2026)
# ---------------------------------------------------------------------------
OPENAI_PRICES: Dict[str, Tuple[float, float]] = {
    # GPT-5.x (current generation)
    "gpt-5.4":                _mtok(  2.50,  15.00),
    "gpt-5.4-mini":           _mtok(  0.75,   4.50),
    "gpt-5.4-nano":           _mtok(  0.20,   1.25),
    "gpt-5.4-pro":            _mtok( 30.00, 180.00),
    "gpt-5.3-chat-latest":    _mtok(  1.75,  14.00),
    "gpt-5.3-codex":          _mtok(  1.75,  14.00),
    # GPT-4o family
    "gpt-4o":                 _mtok(  2.50,  10.00),
    "gpt-4o-mini":            _mtok(  0.15,   0.60),
    # GPT-4 Turbo / GPT-4
    "gpt-4-turbo":            _mtok( 10.00,  30.00),
    "gpt-4-turbo-preview":    _mtok( 10.00,  30.00),
    "gpt-4":                  _mtok( 30.00,  60.00),
    "gpt-4-32k":              _mtok( 60.00, 120.00),
    # GPT-3.5
    "gpt-3.5-turbo":          _mtok(  0.50,   1.50),
    "gpt-3.5-turbo-instruct": _mtok(  1.50,   2.00),
    # o-series reasoning models
    "o4-mini":                _mtok(  1.10,   4.40),
    "o3":                     _mtok( 10.00,  40.00),
    "o3-mini":                _mtok(  1.10,   4.40),
    "o1":                     _mtok( 15.00,  60.00),
    "o1-mini":                _mtok(  3.00,  12.00),
    "o1-preview":             _mtok( 15.00,  60.00),
}

# ---------------------------------------------------------------------------
# Google / Gemini
# Source: https://ai.google.dev/gemini-api/docs/pricing  (Apr 2026)
# ---------------------------------------------------------------------------
GEMINI_PRICES: Dict[str, Tuple[float, float]] = {
    # Gemini 3.x  (newest generation)
    "gemini-3.1-pro-preview":        _mtok(2.000,  12.00),
    "gemini-3.1-flash-lite-preview":  _mtok(0.250,   1.50),
    "gemini-3-flash-preview":         _mtok(0.500,   3.00),
    # Gemini 2.5
    "gemini-2.5-pro":                 _mtok(1.250,  10.00),
    "gemini-2.5-flash":               _mtok(0.300,   2.50),
    "gemini-2.5-flash-lite":          _mtok(0.100,   0.40),
    # Gemini 2.0
    "gemini-2.0-flash":               _mtok(0.100,   0.40),
    "gemini-2.0-flash-lite":          _mtok(0.075,   0.30),
    # Gemini 1.5
    "gemini-1.5-pro":                 _mtok(1.250,   5.00),
    "gemini-1.5-flash":               _mtok(0.075,   0.30),
    "gemini-1.5-flash-8b":            _mtok(0.03750, 0.15),
}

# Unified lookup table (all keys lower-cased)
_ALL_PRICES: Dict[str, Tuple[float, float]] = {
    **{k.lower(): v for k, v in ANTHROPIC_PRICES.items()},
    **{k.lower(): v for k, v in OPENAI_PRICES.items()},
    **{k.lower(): v for k, v in GEMINI_PRICES.items()},
}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _norm_env_name(s: str) -> str:
    return s.upper().replace(":", "_").replace("/", "_").replace("-", "_").replace(".", "_")


def _lookup_prices(model_id: str) -> Optional[Tuple[float, float]]:
    key = model_id.lower().strip()
    if key in _ALL_PRICES:
        return _ALL_PRICES[key]
    # Prefix match: handles snapshot-dated IDs like "claude-haiku-4-5-20251001"
    for table_key, prices in _ALL_PRICES.items():
        if key.startswith(table_key) or table_key.startswith(key):
            return prices
    return None


def _env_price(model_id: str, provider: str) -> Optional[Tuple[float, float]]:
    def _get_pair(suffix: str) -> Optional[Tuple[float, float]]:
        in_val = os.environ.get(f"COST_INPUT_TOKEN_{suffix}")
        out_val = os.environ.get(f"COST_OUTPUT_TOKEN_{suffix}")
        if in_val or out_val:
            try:
                return (float(in_val or 0), float(out_val or 0))
            except (ValueError, TypeError):
                pass
        return None

    if model_id:
        pair = _get_pair(_norm_env_name(model_id))
        if pair:
            return pair
    if provider:
        pair = _get_pair(f"PROVIDER_{_norm_env_name(provider)}")
        if pair:
            return pair
    return None


def _extract_meta(obj) -> Tuple[str, str, bool]:
    if hasattr(obj, "_brain") and obj._brain is not None:
        obj = obj._brain

    meta = getattr(obj, "_cost_meta", None)
    if isinstance(meta, dict):
        return meta.get("provider", ""), meta.get("model_id", ""), bool(meta.get("is_ollama", False))

    model = getattr(obj, "model", None)
    if model is not None:
        model_id = getattr(model, "model_id", None) or getattr(model, "id", None) or ""
        provider = getattr(model, "provider", "") or ""
        is_ollama = provider == "ollama"
        return provider, str(model_id or ""), is_ollama

    return "", "", False


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_model_prices_for(obj) -> Tuple[float, float]:
    """Return (input_usd_per_token, output_usd_per_token) for the model in *obj*.

    Returns (0.0, 0.0) for Ollama models.
    """
    provider, model_id, is_ollama = _extract_meta(obj)
    if is_ollama:
        return (0.0, 0.0)

    # 1. Environment variable overrides
    env = _env_price(model_id, provider)
    if env:
        return env

    # 2. Exact / prefix match in pricing tables
    if model_id:
        prices = _lookup_prices(model_id)
        if prices:
            return prices

    # 3. Provider-level defaults
    if provider in ("ollama",):
        return (0.0, 0.0)
    if "claude" in model_id.lower() or provider in ("claude", "anthropic"):
        return _mtok(3.00, 15.00)   # Sonnet-class default
    if "gemini" in model_id.lower() or provider in ("google", "gemini"):
        return _mtok(1.25, 10.00)   # Gemini 2.5 Pro default
    if "gpt" in model_id.lower() or provider in ("openai",):
        return _mtok(2.50, 15.00)   # GPT-5.4 default

    # Unknown hosted model - conservative estimate
    return _mtok(3.00, 15.00)


def compute_cost_from_usage(usage: dict, obj) -> Tuple[float, int]:
    """Compute total cost (USD) and total tokens from a usage dict and model obj.

    Cost = inputTokens * input_price + outputTokens * output_price.
    Returns (cost_in_dollars, total_tokens).
    """
    in_tok = int(usage.get("inputTokens", 0) or 0)
    out_tok = int(usage.get("outputTokens", 0) or 0)
    in_price, out_price = get_model_prices_for(obj)
    cost = in_tok * in_price + out_tok * out_price
    return cost, in_tok + out_tok
