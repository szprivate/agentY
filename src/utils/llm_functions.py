"""
agentY – Ollama/Qwen LLM helper.

Provides :class:`LLMFunctions`, a thin wrapper around the Ollama ``/api/chat``
endpoint that reads model and host from ``config/settings.json`` and exposes a
single coroutine method for making chat requests.

Typical usage
-------------
>>> llm = LLMFunctions.from_settings()
>>> raw = await llm.chat(messages, json_format=True)
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import httpx

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config helpers (internal)
# ---------------------------------------------------------------------------


def _load_settings() -> dict:
    path = Path(__file__).parent.parent.parent / "config" / "settings.json"
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {}


def _get_ollama_config() -> tuple[str, str]:
    """Return ``(model_name, host)`` from ``settings.json``."""
    settings = _load_settings()
    llm = settings.get("llm", {})
    model: str = llm.get("pipeline", {}).get("llm_functions", "qwen3:0.6b")
    host: str = llm.get("ollama", {}).get("host", "http://localhost:11434")
    return model, host


# ---------------------------------------------------------------------------
# LLMFunctions
# ---------------------------------------------------------------------------


class LLMFunctions:
    """Stateless Ollama chat client bound to a specific model and host.

    Parameters
    ----------
    model:
        Ollama model tag, e.g. ``"qwen3:0.6b"``.
    host:
        Ollama server base URL, e.g. ``"http://localhost:11434"``.
    """

    def __init__(self, model: str, host: str) -> None:
        self.model = model
        self.host = host

    @classmethod
    def from_settings(cls) -> "LLMFunctions":
        """Construct from ``config/settings.json`` (``llm.pipeline.llm_functions`` key)."""
        model, host = _get_ollama_config()
        return cls(model, host)

    async def chat(
        self,
        messages: list[dict[str, str]],
        *,
        json_format: bool = False,
    ) -> str:
        """POST to Ollama ``/api/chat`` and return the assistant message content.

        Parameters
        ----------
        messages:
            OpenAI-style message list, e.g.
            ``[{"role": "system", "content": "..."}, {"role": "user", "content": "..."}]``.
        json_format:
            When ``True``, requests JSON output via Ollama's ``format="json"`` flag.

        Returns
        -------
        str
            Raw text content of the assistant's reply.

        Raises
        ------
        httpx.HTTPStatusError
            On a non-2xx response from Ollama.
        """
        payload: dict[str, Any] = {
            "model":    self.model,
            "messages": messages,
            "stream":   False,
        }
        if json_format:
            payload["format"] = "json"

        async with httpx.AsyncClient() as client:
            resp = await client.post(
                f"{self.host}/api/chat",
                json=payload,
                timeout=60.0,
            )
        resp.raise_for_status()
        return resp.json()["message"]["content"]
