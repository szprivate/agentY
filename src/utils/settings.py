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


# The port the chat host listens on when nothing else says otherwise, per platform.
#
# macOS is not 5000 because macOS does not leave 5000 free. ControlCenter's AirPlay
# Receiver listens on *:5000 (and *:7000) on a stock machine, and the reason this is
# worth a platform switch rather than a line in the README is that it ANSWERS:
# a 403 from `Server: AirTunes/...`, not a refused connection. A sidebar pointed at
# 5000 on a Mac therefore reports the host as down while the host is running
# perfectly well beside it, and every obvious diagnosis - is it started? is the port
# right? - says everything is fine.
#
# 5001 is the neighbour AirPlay does not take. Windows keeps 5000: nothing there
# claims it, and moving it would change the address under installs that already
# have it in a bookmark or a firewall rule.
_DEFAULT_AGENT_PORT = 5000
_PLATFORM_AGENT_PORT = {"darwin": 5001}


def default_agent_port(platform: str | None = None) -> int:
    """The shipped default chat-host port for *platform* (default: this machine).

    Takes the platform rather than reading it so both answers can be checked from
    either machine - a platform switch whose other branch nothing exercises is a
    platform switch that quietly rots.
    """
    import sys

    return _PLATFORM_AGENT_PORT.get(platform or sys.platform, _DEFAULT_AGENT_PORT)


def agent_server_url(platform: str | None = None, *,
                     defaults: dict | None = None,
                     local: dict | None = None) -> str:
    """The address the chat host serves on and the ComfyUI sidebar calls.

    Precedence, and the reason for each step:

    1. ``agent_server_url`` in settings.local.json - an explicit choice for THIS
       machine, and what the settings UI writes. It wins on every platform,
       including a Mac, or choosing a port in the UI would silently do nothing
       there.
    2. ``agent_server_url_macos`` in the committed defaults, on macOS only - the
       shipped answer to AirPlay holding 5000 (see _PLATFORM_AGENT_PORT).
    3. ``agent_server_url`` in the committed defaults - the cross-platform value.
    4. Failing all of that, loopback on this platform's default port.

    *defaults* and *local* are for tests; left alone they are read from disk.
    """
    import sys

    plat = platform or sys.platform
    if defaults is None:
        defaults = load_defaults()
    if local is None:
        local = load_local()

    chosen = str((local or {}).get("agent_server_url") or "").strip()
    if not chosen and plat == "darwin":
        chosen = str((defaults or {}).get("agent_server_url_macos") or "").strip()
    if not chosen:
        chosen = str((defaults or {}).get("agent_server_url") or "").strip()
    return chosen or f"http://127.0.0.1:{default_agent_port(plat)}"


def ollama_host() -> str:
    """The Ollama server URL every caller should use.

    One address, three consumers (the agents, the memory embedder, and the small
    llm_functions helper), so it belongs next to the other server URLs in
    Connections rather than buried in the Ollama tuning block. ``ollama_server_url``
    is that key; ``llm.ollama.host`` is still honoured for configs written before it
    existed, and ``OLLAMA_HOST`` overrides both.
    """
    import os

    env = (os.environ.get("OLLAMA_HOST") or "").strip()
    if env:
        return env
    cfg = load_settings()
    top = str(cfg.get("ollama_server_url") or "").strip()
    if top:
        return top
    legacy = str(((cfg.get("llm") or {}).get("ollama") or {}).get("host") or "").strip()
    return legacy or "http://localhost:11434"
