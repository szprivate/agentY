"""
agentY – A ComfyUI agent built on the Strands Agents SDK.

This module configures and exposes the Strands Agent instance with all
ComfyUI tools registered.
"""

import os
import subprocess
from pathlib import Path

import requests

from strands import Agent
from strands.models.anthropic import AnthropicModel as _BaseAnthropicModel
from strands.models.ollama import OllamaModel
from strands.agent.conversation_manager import SlidingWindowConversationManager

from src.tools import ALL_TOOLS


class AnthropicModel(_BaseAnthropicModel):
    """AnthropicModel with cache_control injected on the last tool.

    This causes Anthropic to cache the entire tools block on every request,
    reducing cached-token cost to 10 % of the normal input price after the
    first call (which pays the 1.25× cache-write surcharge).
    """

    def format_request(self, messages, tool_specs=None, system_prompt=None, tool_choice=None):  # type: ignore[override]
        req = super().format_request(messages, tool_specs, system_prompt, tool_choice)
        if req.get("tools"):
            *head, last = req["tools"]
            req["tools"] = head + [{**last, "cache_control": {"type": "ephemeral"}}]
        return req

# Map from resolved llm name → system-prompt markdown filename stem.
_SYSTEM_PROMPT_FILE: dict[str, str] = {
    "claude": "system_prompt.claude",
    "ollama": "system_prompt.qwencode",
}


def _load_system_prompt(llm: str) -> str:
    """Load the system prompt for *llm* from config/system_prompt.<model>.md."""
    stem = _SYSTEM_PROMPT_FILE.get(llm, f"system_prompt.{llm}")
    path = Path(__file__).parent.parent / "config" / f"{stem}.md"
    return path.read_text(encoding="utf-8")


def _ensure_ollama_model(model_id: str, host: str) -> None:
    """Pull *model_id* via ``ollama pull`` if it is not already present locally.

    Checks the Ollama REST API first; only pulls when the model is absent.
    Streams pull progress to stdout so the user can see download progress.
    """
    try:
        resp = requests.get(f"{host}/api/tags", timeout=10)
        resp.raise_for_status()
        local_names = {m["name"] for m in resp.json().get("models", [])}
        # Ollama stores names as "model:tag"; normalise the requested id the same way.
        normalised = model_id if ":" in model_id else f"{model_id}:latest"
        if normalised in local_names or model_id in local_names:
            print(f"[agentY] Ollama model '{model_id}' already present — skipping pull.")
            return
    except Exception as exc:  # noqa: BLE001
        print(f"[agentY] Warning: could not query Ollama tags ({exc}). Attempting pull anyway.")

    print(f"[agentY] Pulling Ollama model '{model_id}' …")
    try:
        subprocess.run(["ollama", "pull", model_id], check=True)
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"Failed to pull Ollama model '{model_id}': {exc}") from exc
    except FileNotFoundError:
        raise RuntimeError(
            "The 'ollama' CLI was not found on PATH. "
            "Install Ollama from https://ollama.com and ensure it is in PATH."
        )


def _build_model(llm: str, ollama_model: str | None = None):
    """Instantiate the requested LLM backend.

    Args:
        llm: ``'claude'`` (default) or ``'ollama'``.
        ollama_model: Optional Ollama model name override. Takes precedence
                      over the ``OLLAMA_MODEL`` env var when provided.
    """
    llm = llm.strip().lower()
    system_prompt = _load_system_prompt(llm)
    if llm == "ollama":
        model_id = ollama_model or os.environ.get("OLLAMA_MODEL", "qwen3-vl:30b")
        host = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
        _ensure_ollama_model(model_id, host)
        return OllamaModel(
            host=host,
            model_id=model_id,
        )
    # Default: claude
    # Pass system_prompt as a structured content block so Anthropic's
    # prompt-caching kicks in (cache_control="ephemeral"). params is
    # expanded last in AnthropicModel.format_request, so it overrides
    # the plain-string "system" key that Strands sets from system_prompt.
    return AnthropicModel(
        model_id=os.environ.get("ANTHROPIC_MODEL", "claude-haiku-4-5"),
        max_tokens=int(os.environ.get("ANTHROPIC_MAX_TOKENS", "4096")),
        params={
            "system": [
                {
                    "type": "text",
                    "text": system_prompt,
                    "cache_control": {"type": "ephemeral"},
                }
            ]
        },
    )


def create_agent(llm: str | None = None, ollama_model: str | None = None, **kwargs) -> Agent:
    """Create and return the agentY Strands Agent with all ComfyUI tools.

    Args:
        llm: Which LLM backend to use: ``'claude'`` (default) or ``'ollama'``.
             Falls back to the ``AGENT_LLM`` env var, then to ``'claude'``.
        ollama_model: Ollama model name to use when ``llm='ollama'``. Overrides
                      the ``OLLAMA_MODEL`` env var when provided.
        **kwargs: Extra keyword arguments forwarded to the Strands Agent
                  constructor (e.g. to override the model or system prompt).
    """
    resolved_llm = llm or os.environ.get("AGENT_LLM", "claude")
    model = _build_model(resolved_llm, ollama_model=ollama_model)
    print(f"[agentY] Using LLM backend: {resolved_llm} ({model.__class__.__name__})")
    # Limit conversation history to the last 40 messages (≈20 turns).
    # This prevents costs from compounding as history grows across long sessions.
    window_size = int(os.environ.get("AGENT_HISTORY_WINDOW", "40"))
    agent_kwargs = {
        "model": model,
        "system_prompt": _load_system_prompt(resolved_llm),
        "tools": ALL_TOOLS,
        "conversation_manager": SlidingWindowConversationManager(window_size=window_size),
    }
    agent_kwargs.update(kwargs)
    return Agent(**agent_kwargs)
