"""Central settings loader: committed TOML defaults ⊕ gitignored JSON local overrides.

Config is split so structural/default values can be committed while machine-specific
values (paths, model pins, private endpoints) stay local and uncommitted:

* ``config/settings.default.toml`` — committed, human-authored defaults (``#`` comments).
* ``config/settings.local.json``  — gitignored, per-machine overrides, deep-merged
  OVER the defaults. Written by the settings UI; safe to hand-edit.

Effective precedence for any value: environment variable (applied by each caller's
``_cfg``/``_get`` helper) > ``settings.local.json`` > ``settings.default.toml``.

Read is pure stdlib (``tomllib`` + ``json``); the local file is written back with
``json.dump`` (``set_local``), so no third-party TOML writer is needed. Comments live
only in the committed defaults, where they're human-authored and matter — the local
file is machine-owned overrides.
"""
from __future__ import annotations

import json
import threading
import tomllib
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_DEFAULT_PATH = _PROJECT_ROOT / "config" / "settings.default.toml"
_LOCAL_PATH = _PROJECT_ROOT / "config" / "settings.local.json"

_lock = threading.Lock()
_cache: dict | None = None


def default_path() -> Path:
    return _DEFAULT_PATH


def local_path() -> Path:
    return _LOCAL_PATH


def _deep_merge(base: dict, over: dict) -> dict:
    """Return *base* with *over* merged in, recursing into nested dicts (over wins).

    Non-dict values (scalars, lists) replace wholesale; only mappings are merged, so
    a local override sets exactly the leaves it names and inherits the rest.
    """
    out = dict(base)
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_defaults() -> dict:
    """Parse ``settings.default.toml`` (committed). Returns {} if absent/invalid."""
    try:
        with _DEFAULT_PATH.open("rb") as f:
            return tomllib.load(f)
    except FileNotFoundError:
        return {}
    except Exception:  # noqa: BLE001 — never let a bad file crash startup
        return {}


def load_local() -> dict:
    """Parse ``settings.local.json`` (gitignored). Returns {} if absent/invalid."""
    try:
        return json.loads(_LOCAL_PATH.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}
    except Exception:  # noqa: BLE001
        return {}


def load_settings(*, refresh: bool = False) -> dict:
    """Return the merged settings (defaults ⊕ local overrides), cached.

    Pass ``refresh=True`` to force a re-read (e.g. after the file changes on disk).
    """
    global _cache
    if _cache is not None and not refresh:
        return _cache
    with _lock:
        if _cache is not None and not refresh:
            return _cache
        _cache = _deep_merge(load_defaults(), load_local())
        return _cache


def invalidate() -> None:
    """Drop the cached merge so the next ``load_settings`` re-reads both files."""
    global _cache
    with _lock:
        _cache = None


def set_local(overrides: dict) -> dict:
    """Deep-merge *overrides* into ``settings.local.json`` and persist it.

    Used by the settings UI to record machine-specific overrides without touching the
    committed defaults. Creates the file if absent, then invalidates the cache.
    Returns the full new local dict.
    """
    with _lock:
        merged = _deep_merge(load_local(), overrides or {})
        _LOCAL_PATH.parent.mkdir(parents=True, exist_ok=True)
        _LOCAL_PATH.write_text(
            json.dumps(merged, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        global _cache
        _cache = None
        return merged
