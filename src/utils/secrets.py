"""
Centralised secret access for agentY.

Reads secrets directly from the .env file using ``dotenv_values`` so they
are never injected into ``os.environ`` (and therefore not visible to child
processes, logging frameworks, or environment dumps).

Usage::

    from src.utils.secrets import get_secret

    api_key = get_secret("COMFYUI_API_KEY")

Known secrets
-------------
    COMFYUI_API_KEY     – ComfyUI / Comfy.org API key
    HF_TOKEN            – Hugging Face access token
    COMFYUI_MODELS_DIR  – Local ComfyUI models directory override
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Optional

from dotenv import dotenv_values


# Resolved once; __file__ is src/utils/secrets.py → project root is ../../
_ENV_FILE: Path = Path(__file__).parent.parent.parent / ".env"


@lru_cache(maxsize=1)
def _load() -> dict[str, Optional[str]]:
    """Load the .env file exactly once and cache the result."""
    if _ENV_FILE.exists():
        return dotenv_values(_ENV_FILE)
    return {}


def get_secret(key: str, default: str = "") -> str:
    """Return the value of *key* from the .env file.

    Args:
        key:     The name of the secret (e.g. ``"HF_TOKEN"``).
        default: Value to return when the key is absent or empty.

    Returns:
        The secret string, or *default* if not found.
    """
    value = _load().get(key)
    if value is None or value == "":
        return default
    return value
