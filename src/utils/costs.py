from __future__ import annotations
import json
import os
from pathlib import Path
from typing import Dict, Optional, Tuple

# ---------------------------------------------------------------------------
# Utilities to compute token costs for runs.
#
# Pricing data sourced from official pricing pages (verified April 2026;
# Qwen/DashScope added July 2026):
#   Anthropic : https://platform.claude.com/docs/en/about-claude/pricing
#   OpenAI    : https://developers.openai.com/api/docs/pricing
#   Google    : https://ai.google.dev/gemini-api/docs/pricing
#   Alibaba   : https://www.alibabacloud.com/help/en/model-studio/model-pricing
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

# ---------------------------------------------------------------------------
# Alibaba / Qwen (DashScope · Model Studio, OpenAI-compatible API)
# Source: https://help.aliyun.com/zh/model-studio/model-pricing (Sep 2026)
#
# **Global deployment** rates (全球) - what the Alibaba Cloud console actually bills
# for calls through dashscope-intl.aliyuncs.com. This table used to carry the
# *International* (国际) rates, which are 1.4x to 3.5x higher for the same model,
# and qwen3.8-flash was missing altogether (it fell back to the qwen-plus-class
# default of $0.40/$1.20). Checked against the console for 2026-09-11/14/15: the
# Global rates with the 20% cache-hit price below land within 10% of the bill.
#
# The docs list prices in CNY; converted at 7.34 CNY/USD, the rate the same page
# uses for its own USD-priced models (qwen-flash International $0.05 = 0.367 CNY).
# First input tier; non-thinking output. A model with no Global row keeps its
# International rate and says so.
# ---------------------------------------------------------------------------
QWEN_PRICES: Dict[str, Tuple[float, float]] = {
    # Flash — cost-optimized; the agentY pipeline's default across every stage
    "qwen3.8-flash":  _mtok(0.109, 0.368),   # 0.8 / 2.7 CNY
    "qwen3.6-flash":  _mtok(0.163, 0.981),   # 1.2 / 7.2 CNY
    "qwen3.5-flash":  _mtok(0.027, 0.272),   # 0.2 / 2 CNY
    "qwen-flash":     _mtok(0.020, 0.204),   # 0.15 / 1.5 CNY (<=128K)
    # Plus — balanced
    "qwen3.7-plus":   _mtok(0.272, 1.090),   # 2 / 8 CNY list price (a 20% promo was running)
    "qwen3.6-plus":   _mtok(0.272, 1.635),   # 2 / 12 CNY
    "qwen3.5-plus":   _mtok(0.109, 0.654),   # 0.8 / 4.8 CNY
    "qwen-plus":      _mtok(0.109, 0.272),   # 0.8 / 2 CNY (<=128K)
    # Max — flagship
    "qwen3.8-max":    _mtok(1.635, 4.905),   # 12 / 36 CNY
    "qwen3.7-max":    _mtok(1.635, 4.905),   # 12 / 36 CNY
    "qwen3-max":      _mtok(0.341, 1.362),   # 2.5 / 10 CNY
    "qwen-max":       _mtok(1.60, 6.40),     # no Global row: International rate
    # Open-weight
    "qwen3.8-27b":    _mtok(0.409, 1.635),   # 3 / 12 CNY (no Global row: mainland rate)
    # Turbo — legacy budget tier
    "qwen-turbo":     _mtok(0.05, 0.20),     # no Global row: International rate
    # Vision-language
    "qwen3-vl-flash": _mtok(0.020, 0.204),   # 0.15 / 1.5 CNY (<=32K)
    "qwen3-vl-plus":  _mtok(0.136, 1.362),   # 1 / 10 CNY (<=32K)
    "qwen-vl-max":    _mtok(0.80, 3.20),     # no Global row: International rate
    "qwen-vl-plus":   _mtok(0.21, 0.63),     # no Global row: International rate
}

# DashScope / Model Studio provider aliases (mirror of llm_functions._DASHSCOPE_PROVIDERS)
_DASHSCOPE_PROVIDERS = {"dashscope", "modelstudio", "qwen", "alibaba"}

# Unified lookup table (all keys lower-cased)
_ALL_PRICES: Dict[str, Tuple[float, float]] = {
    **{k.lower(): v for k, v in ANTHROPIC_PRICES.items()},
    **{k.lower(): v for k, v in OPENAI_PRICES.items()},
    **{k.lower(): v for k, v in GEMINI_PRICES.items()},
    **{k.lower(): v for k, v in QWEN_PRICES.items()},
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

# ---------------------------------------------------------------------------
# config/pricing.json — user-editable overrides (per MILLION tokens), also
# exposed in the ComfyUI settings UI. Overrides the built-in tables above so the
# viewer's cost column can track a specific endpoint (e.g. a eu-central-1 MaaS
# deployment) and models the tables don't ship (deepseek, kimi, …). Cached by
# mtime; malformed / non-positive entries are skipped so partial files are safe.
# ---------------------------------------------------------------------------
_PRICING_PATH = Path(__file__).resolve().parents[2] / "config" / "pricing.json"
_pricing_cache: Optional[dict] = None
_pricing_mtime: Optional[float] = None


def _load_pricing_overrides() -> dict:
    """Return ``{"models": {name: (in_per_tok, out_per_tok)}, "providers": {…}}``
    from config/pricing.json (values there are USD per million tokens)."""
    global _pricing_cache, _pricing_mtime
    try:
        mtime = _PRICING_PATH.stat().st_mtime
    except OSError:
        _pricing_cache, _pricing_mtime = {"models": {}, "providers": {}}, None
        return _pricing_cache
    if _pricing_cache is not None and _pricing_mtime == mtime:
        return _pricing_cache

    def _pair(d: dict):
        try:
            i, o = float(d.get("in", 0) or 0), float(d.get("out", 0) or 0)
        except (TypeError, ValueError):
            return None
        return _mtok(i, o) if (i > 0 or o > 0) else None

    models: dict = {}
    providers: dict = {}
    try:
        raw = json.loads(_PRICING_PATH.read_text(encoding="utf-8"))
        for name, d in (raw.get("models") or {}).items():
            if isinstance(d, dict) and (p := _pair(d)):
                models[str(name).lower().strip()] = p
        for prov, d in (raw.get("provider_defaults") or {}).items():
            if isinstance(d, dict) and (p := _pair(d)):
                providers[str(prov).lower().strip()] = p
    except Exception:  # noqa: BLE001 — a broken file must never break costing
        pass
    _pricing_cache = {"models": models, "providers": providers}
    _pricing_mtime = mtime
    return _pricing_cache


def get_model_prices_for(obj) -> Tuple[float, float]:
    """Return (input_usd_per_token, output_usd_per_token) for the model in *obj*.

    Returns (0.0, 0.0) for Ollama models. Precedence: env var >
    config/pricing.json (models) > built-in tables > config/pricing.json
    (provider_defaults) > built-in provider default.
    """
    provider, model_id, is_ollama = _extract_meta(obj)
    if is_ollama:
        return (0.0, 0.0)

    # 1. Environment variable overrides
    env = _env_price(model_id, provider)
    if env:
        return env

    over = _load_pricing_overrides()

    # 2. config/pricing.json model override (exact, then prefix)
    if model_id:
        key = model_id.lower().strip()
        if key in over["models"]:
            return over["models"][key]
        for mk, pv in over["models"].items():
            if key.startswith(mk) or mk.startswith(key):
                return pv

    # 3. Exact / prefix match in the built-in pricing tables
    if model_id:
        prices = _lookup_prices(model_id)
        if prices:
            return prices

    # 4. config/pricing.json provider default
    pk = (provider or "").lower().strip()
    if pk in over["providers"]:
        return over["providers"][pk]

    # 5. Provider-level defaults
    if provider in ("ollama",):
        return (0.0, 0.0)
    if "claude" in model_id.lower() or provider in ("claude", "anthropic"):
        return _mtok(3.00, 15.00)   # Sonnet-class default
    if "gemini" in model_id.lower() or provider in ("google", "gemini"):
        return _mtok(1.25, 10.00)   # Gemini 2.5 Pro default
    if "gpt" in model_id.lower() or provider in ("openai",):
        return _mtok(2.50, 15.00)   # GPT-5.4 default
    if "qwen" in model_id.lower() or provider.lower() in _DASHSCOPE_PROVIDERS:
        return _mtok(0.40, 1.20)    # qwen-plus-class default (Model Studio Intl)

    # Unknown hosted model - conservative estimate
    return _mtok(3.00, 15.00)


# ---------------------------------------------------------------------------
# Cached input
#
# Providers report cache hits in one of two ways, and the bill depends on which:
#
# * Anthropic reports them BESIDE ``inputTokens``, which counts only fresh input.
#   Hits bill at 0.1x the input price, cache creation at 1.25x.
# * OpenAI-compatible APIs - DashScope among them - report ``prompt_tokens``
#   INCLUDING the cached part, with ``cached_tokens`` as a breakdown of it (Strands
#   maps those to inputTokens and cacheReadInputTokens).
#
# This used to treat every provider like Anthropic: a DashScope turn's cached
# tokens were charged at the full input price as part of inputTokens, and then
# again at 0.1x on top. With 90% of agentY's input served from cache, that alone
# more than quadrupled the figure. DashScope's implicit cache - the automatic one
# agentY relies on - bills a hit at 20% of the input price and creates the cache
# at the normal input price (help.aliyun.com/zh/model-studio/context-cache).
# ---------------------------------------------------------------------------
_ANTHROPIC_PROVIDERS = {"anthropic", "claude", "bedrock"}

# (cache-hit multiplier, cache-creation multiplier), relative to the input price.
_CACHE_RATES_ANTHROPIC = (0.10, 1.25)
_CACHE_RATES_DASHSCOPE = (0.20, 1.00)
_CACHE_RATES_DEFAULT = (0.10, 1.00)


def _cache_accounting(obj) -> Tuple[float, float, bool]:
    """``(hit_multiplier, creation_multiplier, cached_included_in_input)`` for *obj*."""
    provider, model_id, _is_ollama = _extract_meta(obj)
    p = (provider or "").lower()
    m = (model_id or "").lower()
    if p in _ANTHROPIC_PROVIDERS or m.startswith("claude"):
        return (*_CACHE_RATES_ANTHROPIC, False)
    if p in _DASHSCOPE_PROVIDERS or m.startswith("qwen"):
        return (*_CACHE_RATES_DASHSCOPE, True)
    return (*_CACHE_RATES_DEFAULT, True)


def compute_cost_from_usage(usage: dict, obj) -> Tuple[float, int]:
    """Compute total cost (USD) and total tokens from a usage dict and model obj.

    Cost = fresh input   * input_price
         + outputTokens  * output_price
         + cache hits    * input_price * hit_multiplier
         + cache writes  * input_price * creation_multiplier

    where fresh input is ``inputTokens`` for Anthropic (which reports cache hits
    separately) and ``inputTokens - cacheReadInputTokens`` for OpenAI-compatible
    providers such as DashScope (which count them inside inputTokens). See the
    notes above for the multipliers. Ollama models price every term at 0.

    Returns (cost_in_dollars, total_tokens), each token counted once: cached
    tokens are added to the total only where the provider reports them separately.
    """
    in_tok = int(usage.get("inputTokens", 0) or 0)
    out_tok = int(usage.get("outputTokens", 0) or 0)
    cache_read = int(usage.get("cacheReadInputTokens", 0) or 0)
    cache_write = int(usage.get("cacheWriteInputTokens", 0) or 0)
    in_price, out_price = get_model_prices_for(obj)
    hit_mult, write_mult, included = _cache_accounting(obj)
    # A breakdown can never exceed what it breaks down; if it does, the provider
    # must be reporting the two separately after all.
    if included and cache_read + cache_write > in_tok:
        included = False
    fresh = in_tok - cache_read - cache_write if included else in_tok
    cost = (
        fresh * in_price
        + out_tok * out_price
        + cache_read * in_price * hit_mult
        + cache_write * in_price * write_mult
    )
    total = in_tok + out_tok + (0 if included else cache_read + cache_write)
    return cost, total
