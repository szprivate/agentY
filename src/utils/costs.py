"""Utilities to compute token costs for runs.

Rules:
- Ollama models are free (cost per token = 0).
- Look up cost via environment variables when available:
  * COST_PER_TOKEN_<MODEL_ID> (model id uppercased, ':' -> '_')
  * COST_PER_TOKEN_PROVIDER_<PROVIDER> (e.g. PROVIDER=ANTHROPIC)
- Fallbacks: Anthropic/claude default to 0.00003 $/token, others default to 0.00003.
"""
from __future__ import annotations

import os
from typing import Tuple


def _norm_model_env_name(model_id: str) -> str:
    return model_id.upper().replace(":", "_").replace("/", "_").replace("-", "_")


def _extract_meta(obj) -> Tuple[str, str, bool]:
    """Return (provider, model_id, is_ollama) inferred from *obj*.

    Tries several common shapes: pipeline (has _brain), Agent (has _cost_meta),
    or a model object with attribute `model_id`.
    """
    # Pipeline-like object with a brain agent
    if hasattr(obj, "_brain") and obj._brain is not None:
        obj = obj._brain

    # Prefers an attached _cost_meta set when creating the Agent
    meta = getattr(obj, "_cost_meta", None)
    if isinstance(meta, dict):
        return meta.get("provider", ""), meta.get("model_id", ""), bool(meta.get("is_ollama", False))

    # Try to introspect a model attribute
    model = getattr(obj, "model", None)
    if model is not None:
        model_id = getattr(model, "model_id", None) or getattr(model, "id", None) or ""
        provider = getattr(model, "provider", "") or ""
        # Best-effort check if it's Ollama
        is_ollama = provider == "ollama" or (isinstance(model_id, str) and model_id.lower().startswith("qwen"))
        return provider, str(model_id or ""), is_ollama

    # Last resort: empty
    return "", "", False


def get_cost_per_token_for(obj) -> float:
    """Return cost-per-token in dollars for the model used by *obj*.

    Priority:
      1. Environment var COST_PER_TOKEN_<MODEL_ID>
      2. Environment var COST_PER_TOKEN_PROVIDER_<PROVIDER>
      3. Ollama -> 0.0
      4. Defaults: 0.00003
    """
    provider, model_id, is_ollama = _extract_meta(obj)
    if is_ollama:
        return 0.0

    if model_id:
        env_name = f"COST_PER_TOKEN_{_norm_model_env_name(model_id)}"
        val = os.environ.get(env_name)
        if val:
            try:
                return float(val)
            except Exception:
                pass

    if provider:
        env_name = f"COST_PER_TOKEN_PROVIDER_{provider.upper()}"
        val = os.environ.get(env_name)
        if val:
            try:
                return float(val)
            except Exception:
                pass

    # Reasonable default for hosted models (per-token prices vary wildly).
    return 0.00003


def compute_cost_from_usage(usage: dict, obj) -> Tuple[float, int]:
    """Compute total cost (dollars) and total tokens from usage dict and model obj.

    Uses inputTokens + outputTokens as the total token count.
    """
    in_tok = int(usage.get("inputTokens", 0) or 0)
    out_tok = int(usage.get("outputTokens", 0) or 0)
    total = in_tok + out_tok
    price = get_cost_per_token_for(obj)
    return total * price, total
