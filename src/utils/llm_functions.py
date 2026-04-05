"""
agentY – Ollama/Qwen LLM helper.

Provides :class:`LLMFunctions`, a thin wrapper around the Ollama ``/api/chat``
endpoint that reads model and host from ``config/settings.json`` and exposes
coroutine methods for text and vision chat requests.

Typical usage
-------------
>>> llm = LLMFunctions.from_settings()
>>> raw = await llm.chat(messages, json_format=True)

Vision usage (requires a multimodal model such as llava or qwen2.5-vl):
>>> llm_vis = LLMFunctions.for_vision()
>>> answer = await llm_vis.vision_chat("Does this image match the brief?", image_bytes)
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


def _get_triage_config() -> tuple[str, str]:
    """Return ``(model_name, host)`` for triage classification from ``settings.json``.

    Reads ``llm.pipeline.triage``; falls back to the hard-coded default
    ``qwen3:0.6b`` when not set.
    """
    settings = _load_settings()
    llm = settings.get("llm", {})
    model: str = llm.get("pipeline", {}).get("triage", "qwen3:0.6b")
    host: str = llm.get("ollama", {}).get("host", "http://localhost:11434")
    return model, host


def _get_vision_config() -> tuple[str, str]:
    """Return ``(vision_model, host)`` for multimodal analysis from ``settings.json``.

    Reads ``llm.pipeline.executor_vision_model``; falls back to ``llm_functions``
    model or the hard-coded default ``llava:latest`` when not set.
    """
    settings = _load_settings()
    llm = settings.get("llm", {})
    pipeline = llm.get("pipeline", {})
    model: str = (
        pipeline.get("executor_vision_model")
        or pipeline.get("llm_functions", "llava:latest")
    )
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

    @classmethod
    def for_triage(cls) -> "LLMFunctions":
        """Construct a triage-optimised instance (``llm.pipeline.triage`` key).

        Uses a really small Ollama model for fast, cheap intent classification.
        Falls back to ``qwen3:0.6b`` when the key is absent.
        """
        model, host = _get_triage_config()
        return cls(model, host)

    @classmethod
    def for_vision(cls) -> "LLMFunctions":
        """Construct a vision-capable instance (``llm.pipeline.executor_vision_model`` key)."""
        model, host = _get_vision_config()
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

    async def vision_chat(
        self,
        prompt: str,
        image_bytes: bytes,
        *,
        system: str = "",
    ) -> str:
        """Send *image_bytes* plus a text *prompt* to an Ollama multimodal model.

        The image is base64-encoded and sent in the Ollama ``images`` field.
        Requires a model that supports vision, e.g. ``llava``, ``qwen2.5-vl``,
        ``moondream``.  Configure via ``llm.pipeline.executor_vision_model`` in
        ``settings.json``.

        Parameters
        ----------
        prompt:
            Text instruction / question about the image.
        image_bytes:
            Raw bytes of the image (PNG, JPEG, etc.).
        system:
            Optional system message injected before the user turn.

        Returns
        -------
        str
            Raw text content of the model's reply.

        Raises
        ------
        httpx.HTTPStatusError
            On a non-2xx response from Ollama.
        """
        import base64

        b64 = base64.b64encode(image_bytes).decode("ascii")
        messages: list[dict] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt, "images": [b64]})

        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": False,
        }

        async with httpx.AsyncClient() as client:
            resp = await client.post(
                f"{self.host}/api/chat",
                json=payload,
                timeout=120.0,
            )
        resp.raise_for_status()
        return resp.json()["message"]["content"]
