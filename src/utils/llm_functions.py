"""
agentY – configurable LLM helper.

Provides :class:`LLMFunctions`, a thin, stateless chat/vision client for the
pipeline's cheap side calls (conversation summarisation, executor Vision-QA,
short chat titles). It is **provider-agnostic**: the model is chosen with the
usual ``provider,model`` spec (as everywhere else in agentY) read from
``config/settings.json``. A bare model tag with no comma is treated as an Ollama
model, matching the historical behaviour of this module.

Supported providers
--------------------
- ``ollama``                       → ``{host}/api/chat``              (local)
- ``dashscope`` / ``qwen`` / …     → ``{base_url}/chat/completions``  (OpenAI-compatible)
- ``claude`` / ``anthropic``       → ``api.anthropic.com/v1/messages``

``.host`` always resolves to the Ollama host (independent of the chat provider),
so Ollama-specific callers (e.g. model unload) keep working regardless of which
provider ``llm_functions`` points at.

Typical usage
-------------
>>> llm = LLMFunctions.from_settings()
>>> raw = await llm.chat(messages, json_format=True)

Vision usage (requires a vision-capable model for the chosen provider — e.g.
``ollama,llava``, ``dashscope,qwen-vl-max``, ``claude,claude-haiku-4-5``):
>>> llm_vis = LLMFunctions.for_vision()
>>> answer = await llm_vis.vision_chat("Does this image match the brief?", image_bytes)
"""

from __future__ import annotations

import base64
import json
import logging
import os
from pathlib import Path
from typing import Any

import httpx

logger = logging.getLogger(__name__)

# Provider aliases, kept in sync with src/agent.py so a spec written for the
# pipeline agents works here too.
_DASHSCOPE_PROVIDERS = {"dashscope", "modelstudio", "qwen", "alibaba"}
_ANTHROPIC_PROVIDERS = {"claude", "anthropic"}

_ANTHROPIC_URL = "https://api.anthropic.com/v1/messages"
_ANTHROPIC_VERSION = "2023-06-01"


# ---------------------------------------------------------------------------
# Config helpers (internal)
# ---------------------------------------------------------------------------


def _load_settings() -> dict:
    from src.utils.settings import load_settings
    return load_settings()


def _parse_spec(spec: str) -> tuple[str, str]:
    """Split a ``provider,model`` spec. A bare tag (no comma) is an Ollama model."""
    spec = (spec or "").strip()
    if "," in spec:
        provider, _, model = spec.partition(",")
        return provider.strip().lower(), model.strip()
    return "ollama", spec


def _guess_image_mime(data: bytes) -> str:
    """Best-effort image MIME sniff from magic bytes (defaults to image/png)."""
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return "image/png"
    if data[:2] == b"\xff\xd8":
        return "image/jpeg"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    if data[:6] in (b"GIF87a", b"GIF89a"):
        return "image/gif"
    return "image/png"


# ---------------------------------------------------------------------------
# LLMFunctions
# ---------------------------------------------------------------------------


class LLMFunctions:
    """Stateless multi-provider chat/vision client bound to one model.

    Construct via :meth:`from_settings` (text) or :meth:`for_vision` (multimodal);
    the model — and therefore the provider — comes from ``config/settings.json``.
    """

    def __init__(
        self,
        provider: str,
        model: str,
        *,
        host: str,
        base_url: str = "",
        api_key: str = "",
        max_tokens: int = 2048,
    ) -> None:
        self.provider = provider
        self.model = model
        self.host = host          # Ollama host — always resolved (used by /api/* callers)
        self.base_url = base_url  # OpenAI-compatible base URL (dashscope)
        self.api_key = api_key
        self.max_tokens = max_tokens

    # ── constructors ─────────────────────────────────────────────────────
    @classmethod
    def _from_spec(cls, spec: str, *, default_model: str, default_max_tokens: int) -> "LLMFunctions":
        settings = _load_settings()
        llm = settings.get("llm", {})
        provider, model = _parse_spec(spec)
        host = llm.get("ollama", {}).get("host", "http://localhost:11434")
        base_url, api_key = "", ""
        max_tokens = default_max_tokens

        if provider in _DASHSCOPE_PROVIDERS:
            ds = llm.get("dashscope", {})
            base_url = (os.environ.get("DASHSCOPE_BASE_URL")
                        or ds.get("base_url", "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"))
            api_key = os.environ.get("DASHSCOPE_API_KEY") or os.environ.get("ALIBABA_API_KEY") or ""
            max_tokens = int(ds.get("max_tokens", default_max_tokens) or default_max_tokens)
            model = model or ds.get("model", "qwen-plus")
        elif provider in _ANTHROPIC_PROVIDERS:
            an = llm.get("anthropic", {})
            api_key = os.environ.get("ANTHROPIC_API_KEY") or ""
            max_tokens = int(an.get("max_tokens", default_max_tokens) or default_max_tokens)
            model = model or an.get("model", "claude-haiku-4-5")
        else:
            provider = "ollama"
            model = model or default_model

        return cls(provider, model, host=host, base_url=base_url, api_key=api_key, max_tokens=max_tokens)

    @classmethod
    def from_settings(cls) -> "LLMFunctions":
        """Text client from ``llm.pipeline.llm_functions`` (``provider,model`` or bare Ollama tag)."""
        settings = _load_settings()
        spec = settings.get("llm", {}).get("pipeline", {}).get("llm_functions", "qwen3:0.6b")
        return cls._from_spec(spec, default_model="qwen3:0.6b", default_max_tokens=2048)

    @classmethod
    def for_vision(cls) -> "LLMFunctions":
        """Vision client from ``llm.pipeline.executor_vision_model`` (falls back to ``llm_functions``)."""
        settings = _load_settings()
        pipeline = settings.get("llm", {}).get("pipeline", {})
        spec = pipeline.get("executor_vision_model") or pipeline.get("llm_functions") or "llava:latest"
        return cls._from_spec(spec, default_model="llava:latest", default_max_tokens=1024)

    # ── public API ───────────────────────────────────────────────────────
    async def chat(self, messages: list[dict[str, Any]], *, json_format: bool = False) -> str:
        """Send an OpenAI-style ``messages`` list; return the assistant text.

        ``messages`` is ``[{"role": "system"|"user"|"assistant", "content": "…"}]``.
        ``json_format`` requests a JSON object where the provider supports it.
        """
        if self.provider in _DASHSCOPE_PROVIDERS:
            return await self._openai_chat(messages, json_format=json_format)
        if self.provider in _ANTHROPIC_PROVIDERS:
            return await self._anthropic_chat(messages, json_format=json_format)
        return await self._ollama_chat(messages, json_format=json_format)

    async def vision_chat(
        self,
        prompt: str,
        image_bytes: bytes,
        *,
        system: str = "",
        extra_images: "list[bytes] | None" = None,
    ) -> str:
        """Send *image_bytes* (+ any *extra_images*) plus a text *prompt* to a
        vision-capable model and return its reply. Requires a multimodal model
        for the chosen provider."""
        images = [image_bytes] + list(extra_images or [])
        if self.provider in _DASHSCOPE_PROVIDERS:
            return await self._openai_vision(prompt, images, system=system)
        if self.provider in _ANTHROPIC_PROVIDERS:
            return await self._anthropic_vision(prompt, images, system=system)
        return await self._ollama_vision(prompt, images, system=system)

    # ── Ollama backend ───────────────────────────────────────────────────
    async def _ollama_chat(self, messages: list[dict[str, Any]], *, json_format: bool) -> str:
        payload: dict[str, Any] = {"model": self.model, "messages": messages, "stream": False}
        if json_format:
            payload["format"] = "json"
        async with httpx.AsyncClient() as client:
            resp = await client.post(f"{self.host}/api/chat", json=payload, timeout=60.0)
        resp.raise_for_status()
        return resp.json()["message"]["content"]

    async def _ollama_vision(self, prompt: str, images: list[bytes], *, system: str) -> str:
        encoded = [base64.b64encode(b).decode("ascii") for b in images]
        messages: list[dict] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt, "images": encoded})
        payload = {"model": self.model, "messages": messages, "stream": False}
        async with httpx.AsyncClient() as client:
            resp = await client.post(f"{self.host}/api/chat", json=payload, timeout=120.0)
        resp.raise_for_status()
        return resp.json()["message"]["content"]

    # ── OpenAI-compatible backend (DashScope / Model Studio) ─────────────
    def _openai_url(self) -> str:
        return self.base_url.rstrip("/") + "/chat/completions"

    def _openai_headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}

    async def _openai_chat(self, messages: list[dict[str, Any]], *, json_format: bool) -> str:
        payload: dict[str, Any] = {
            "model": self.model, "messages": messages, "stream": False,
            "max_tokens": self.max_tokens,
        }
        if json_format:
            payload["response_format"] = {"type": "json_object"}
        async with httpx.AsyncClient() as client:
            resp = await client.post(self._openai_url(), json=payload, headers=self._openai_headers(), timeout=60.0)
        resp.raise_for_status()
        return (resp.json()["choices"][0]["message"].get("content") or "")

    async def _openai_vision(self, prompt: str, images: list[bytes], *, system: str) -> str:
        content: list[dict] = [{"type": "text", "text": prompt}]
        for b in images:
            b64 = base64.b64encode(b).decode("ascii")
            content.append({"type": "image_url",
                            "image_url": {"url": f"data:{_guess_image_mime(b)};base64,{b64}"}})
        messages: list[dict] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": content})
        payload = {"model": self.model, "messages": messages, "stream": False, "max_tokens": self.max_tokens}
        async with httpx.AsyncClient() as client:
            resp = await client.post(self._openai_url(), json=payload, headers=self._openai_headers(), timeout=120.0)
        resp.raise_for_status()
        return (resp.json()["choices"][0]["message"].get("content") or "")

    # ── Anthropic backend ────────────────────────────────────────────────
    def _anthropic_headers(self) -> dict[str, str]:
        return {"x-api-key": self.api_key, "anthropic-version": _ANTHROPIC_VERSION,
                "content-type": "application/json"}

    @staticmethod
    def _split_system(messages: list[dict[str, Any]]) -> tuple[str, list[dict]]:
        """Anthropic keeps the system prompt out of the message list."""
        system_parts: list[str] = []
        conv: list[dict] = []
        for m in messages:
            if m.get("role") == "system":
                system_parts.append(str(m.get("content", "")))
            else:
                conv.append({"role": m.get("role", "user"), "content": m.get("content", "")})
        return "\n\n".join(p for p in system_parts if p), conv

    async def _anthropic_chat(self, messages: list[dict[str, Any]], *, json_format: bool) -> str:
        system, conv = self._split_system(messages)
        if json_format and system:
            system += "\n\nReply with a single valid JSON object and nothing else."
        if not conv:
            conv = [{"role": "user", "content": system or "(no message)"}]
            system = ""
        payload: dict[str, Any] = {"model": self.model, "max_tokens": self.max_tokens, "messages": conv}
        if system:
            payload["system"] = system
        async with httpx.AsyncClient() as client:
            resp = await client.post(_ANTHROPIC_URL, json=payload, headers=self._anthropic_headers(), timeout=60.0)
        resp.raise_for_status()
        return "".join(blk.get("text", "") for blk in resp.json().get("content", [])
                       if isinstance(blk, dict) and blk.get("type") == "text")

    async def _anthropic_vision(self, prompt: str, images: list[bytes], *, system: str) -> str:
        content: list[dict] = []
        for b in images:
            content.append({"type": "image", "source": {"type": "base64",
                            "media_type": _guess_image_mime(b),
                            "data": base64.b64encode(b).decode("ascii")}})
        content.append({"type": "text", "text": prompt})
        payload: dict[str, Any] = {"model": self.model, "max_tokens": self.max_tokens,
                                   "messages": [{"role": "user", "content": content}]}
        if system:
            payload["system"] = system
        async with httpx.AsyncClient() as client:
            resp = await client.post(_ANTHROPIC_URL, json=payload, headers=self._anthropic_headers(), timeout=120.0)
        resp.raise_for_status()
        return "".join(blk.get("text", "") for blk in resp.json().get("content", [])
                       if isinstance(blk, dict) and blk.get("type") == "text")
