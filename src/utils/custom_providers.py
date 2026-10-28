"""Custom OpenAI-compatible LLM providers (Mercury, OpenRouter, Together, vLLM, LM Studio...).

A provider agentY does not know is registered in the settings as
``llm.custom_providers.<id> = {label, key_env, url_env, vision, extra_body?}``
(config/settings.local.json). The key and the URL stay in .env, under the
variables named by ``key_env`` and ``url_env``. A tier then names it like any other
provider: ``"mercury,model-id"``.

This module is pure on purpose: no import of src.agent at module level, so the
model factory, the host and the tests can all use it.
"""
from __future__ import annotations

import re

_ENV_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_ID = re.compile(r"^[a-z][a-z0-9_-]{0,31}$")

# Names a provider token already means. A custom provider must not shadow one: a
# tier saying "openai,gpt-4o" would silently go somewhere else. agent.py's alias
# sets are added to these at check time (reserved()).
_STATIC_RESERVED = frozenset({
    "claude", "anthropic", "dashscope", "modelstudio", "qwen", "alibaba", "openai", "gpt",
    "google", "gemini", "ollama", "strata", "bedrock", "azure", "vertex", "local", "none", "auto", "all",
})


def reserved() -> frozenset:
    names = set(_STATIC_RESERVED)
    try:
        from src import agent  # noqa: PLC0415
        names |= set(agent._DASHSCOPE_PROVIDERS) | set(agent._OPENAI_PROVIDERS) | set(agent._GEMINI_PROVIDERS)
    except Exception:  # noqa: BLE001 - the static list is the floor
        pass
    return frozenset(names)


def slug(name: str) -> str:
    """The id of a service name: lowercase, non-alphanumerics to "_"."""
    return re.sub(r"[^a-z0-9]+", "_", str(name or "").strip().lower()).strip("_")


def validate(pid: str, entry: dict) -> str:
    """Why *entry* cannot be registered under *pid*; "" when it can."""
    if not isinstance(entry, dict):
        return "the entry must be an object"
    if not _ID.match(str(pid)):
        return f"'{pid}' is not a usable provider id (a letter first, then letters, digits, - or _)"
    if str(pid) in reserved():
        return f"'{pid}' is the name of a built-in provider; pick another service name"
    for field in ("key_env", "url_env"):
        if not _ENV_NAME.match(str(entry.get(field) or "")):
            return f"{field} must be an environment variable name"
    return ""


def normalize(pid: str, entry: dict) -> dict:
    out = {"label": str(entry.get("label") or pid), "key_env": str(entry["key_env"]),
           "url_env": str(entry["url_env"]), "vision": bool(entry.get("vision"))}
    # Request fields of the provider's own, sent with every call (a vLLM or
    # llama.cpp server's chat_template_kwargs, say). Only an object is kept.
    if isinstance(entry.get("extra_body"), dict) and entry["extra_body"]:
        out["extra_body"] = dict(entry["extra_body"])
    return out


def load(settings: dict | None = None) -> dict:
    """The registered providers, {id: {label, key_env, url_env, vision}}. Invalid
    entries are left out. Reads the merged settings unless given a dict."""
    if settings is None:
        try:
            from src.utils.settings import load_settings  # noqa: PLC0415
            settings = load_settings()
        except Exception:  # noqa: BLE001
            return {}
    raw = ((settings or {}).get("llm") or {}).get("custom_providers")
    out = {}
    if isinstance(raw, dict):
        for pid, entry in raw.items():
            pid = str(pid).strip().lower()
            if isinstance(entry, dict) and not validate(pid, entry):
                out[pid] = normalize(pid, entry)
    return out


def get(pid: str, settings: dict | None = None) -> dict | None:
    return load(settings).get(str(pid or "").strip().lower())


def connection(pid: str, settings: dict | None = None) -> tuple[str, str]:
    """(base_url, api_key) of provider *pid* from the environment. Raises
    RuntimeError naming the missing variable."""
    import os  # noqa: PLC0415
    entry = get(pid, settings)
    if entry is None:
        raise RuntimeError(f"'{pid}' is not a registered custom provider")
    url = os.environ.get(entry["url_env"], "").strip()
    key = os.environ.get(entry["key_env"], "").strip()
    if not url:
        raise RuntimeError(f"{entry['label']}: {entry['url_env']} is not set - add the server URL "
                           f"in Settings > Authentication (+ Add API key) or in .env")
    if not key:
        raise RuntimeError(f"{entry['label']}: {entry['key_env']} is not set - add the API key "
                           f"in Settings > Authentication (+ Add API key) or in .env")
    return url, key


def env_names() -> set[str]:
    """Every environment variable the registered providers read."""
    return {v for e in load().values() for v in (e["key_env"], e["url_env"])}
