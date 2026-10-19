"""The provider's own books, set beside agentY's token-log figures.

Optional (``[costs] reconcile`` in settings, off by default) and only for providers
that publish an account-level cost API a normal key cannot reach:

* Anthropic  ``GET /v1/organizations/cost_report``  needs an Admin API key
  (``ANTHROPIC_ADMIN_KEY``, ``sk-ant-admin...``); amounts are cents as decimal strings.
  https://platform.claude.com/docs/en/manage-claude/usage-cost-api
* OpenAI     ``GET /v1/organization/costs``         needs an Admin key
  (``OPENAI_ADMIN_KEY``); ``amount.value`` is in ``amount.currency`` (usd).
  https://platform.openai.com/docs/api-reference/usage/costs

These are ORGANISATION totals: they include everything else billed to the same
org/key owner, not only agentY. They are shown as a cross-check, never added to
the overview. Not covered (no API a plain key can call): Google Gemini (billing
export to BigQuery only), DashScope / Model Studio (Alibaba BSS OpenAPI, needs
Alibaba Cloud AccessKeys, not the DashScope key), Ollama (free), OpenRouter
(states the cost on every response, no reconciliation needed).
"""
from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Callable

_ANTHROPIC_URL = "https://api.anthropic.com/v1/organizations/cost_report"
_OPENAI_URL = "https://api.openai.com/v1/organization/costs"
_MAX_PAGES = 12


def enabled() -> bool:
    try:
        from src.utils.settings import load_settings  # noqa: PLC0415
        return bool(((load_settings() or {}).get("costs") or {}).get("reconcile", False))
    except Exception:  # noqa: BLE001
        return False


def _key(*names: str) -> str:
    for n in names:
        v = os.environ.get(n)
        if v:
            return v
    return ""


def _iso(ts: float) -> str:
    return datetime.fromtimestamp(ts, timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def anthropic_cost(from_ts: float, to_ts: float, key: str, get: Callable) -> float:
    """USD over the window from the Anthropic cost report (cents -> dollars)."""
    cents, page = 0.0, None
    for _ in range(_MAX_PAGES):
        params = {"starting_at": _iso(from_ts), "ending_at": _iso(to_ts), "limit": 31}
        if page:
            params["page"] = page
        body = get(_ANTHROPIC_URL, params=params,
                   headers={"x-api-key": key, "anthropic-version": "2023-06-01"})
        for bucket in body.get("data") or []:
            for res in bucket.get("results") or []:
                if str(res.get("currency", "USD")).upper() == "USD":
                    cents += float(res.get("amount") or 0)
        page = body.get("next_page")
        if not body.get("has_more") or not page:
            break
    return cents / 100.0


def openai_cost(from_ts: float, to_ts: float, key: str, get: Callable) -> float:
    """USD over the window from the OpenAI costs endpoint."""
    total, page = 0.0, None
    for _ in range(_MAX_PAGES):
        params = {"start_time": int(from_ts), "end_time": int(to_ts), "bucket_width": "1d", "limit": 180}
        if page:
            params["page"] = page
        body = get(_OPENAI_URL, params=params, headers={"Authorization": f"Bearer {key}"})
        for bucket in body.get("data") or []:
            for res in bucket.get("results") or []:
                amt = res.get("amount") or {}
                if str(amt.get("currency", "usd")).lower() == "usd":
                    total += float(amt.get("value") or 0)
        page = body.get("next_page")
        if not body.get("has_more") or not page:
            break
    return total


def _http_get(url: str, params: dict, headers: dict) -> dict:
    import httpx  # noqa: PLC0415
    r = httpx.get(url, params=params, headers=headers, timeout=20.0)
    r.raise_for_status()
    return r.json()


def reconcile(from_ts: float | None, to_ts: float | None, *, get: Callable = _http_get,
              force: bool = False) -> dict:
    """``{"enabled", "providers": {name: {"cost", "currency"}}, "skipped": {name: why}}``.

    Reads only (no spend). Never raises: a failing provider lands in ``skipped``.
    """
    if not (force or enabled()):
        return {"enabled": False, "providers": {}, "skipped": {}}
    import time as _t  # noqa: PLC0415
    to_ts = to_ts or _t.time()
    from_ts = from_ts or (to_ts - 7 * 86400)
    out: dict = {"enabled": True, "providers": {}, "skipped": {}}
    for name, keys, fn in (
        ("anthropic", ("ANTHROPIC_ADMIN_KEY", "ANTHROPIC_ADMIN_API_KEY"), anthropic_cost),
        ("openai", ("OPENAI_ADMIN_KEY", "OPENAI_ADMIN_API_KEY"), openai_cost),
    ):
        key = _key(*keys)
        if not key:
            out["skipped"][name] = f"needs an admin key in {keys[0]}"
            continue
        try:
            out["providers"][name] = {"cost": round(fn(from_ts, to_ts, key, get), 4), "currency": "USD"}
        except Exception as exc:  # noqa: BLE001 - never leak the key; say what failed
            out["skipped"][name] = f"request failed: {type(exc).__name__}"
    return out
