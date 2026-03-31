"""
agentY – A ComfyUI agent built on the Strands Agents SDK.

Two-agent pipeline:
  • Researcher  – Ollama (default) or any LLM; pattern-matching/resolution only.
                  Produces a brainbriefing JSON.
  • Brain       – Claude (default) or any LLM; workflow assembly, execution, QA.

Single-agent fallback (``create_agent``) is kept for backward compatibility.
"""

import json
import os
import subprocess
from pathlib import Path

import requests

from strands import Agent, AgentSkills
from strands.models.anthropic import AnthropicModel as _BaseAnthropicModel
from strands.models.ollama import OllamaModel
from strands.agent.conversation_manager import SlidingWindowConversationManager

from src.tools import ALL_TOOLS, RESEARCHER_TOOLS, BRAIN_TOOLS


# ---------------------------------------------------------------------------
# Settings loader – reads config/settings.json once; env vars always win.
# ---------------------------------------------------------------------------

def _load_settings() -> dict:
    """Return the parsed settings.json, or {} if the file is absent/invalid."""
    path = Path(__file__).parent.parent / "config" / "settings.json"
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {}


_SETTINGS: dict = {}  # populated lazily by _cfg()


def _settings() -> dict:
    global _SETTINGS
    if not _SETTINGS:
        _SETTINGS = _load_settings()
    return _SETTINGS


def _cfg(env_var: str, *settings_path: str, default: str | int = "") -> str | int:
    """Return a config value with priority: env var > settings.json > default.

    Args:
        env_var:       Name of the environment variable to check first.
        *settings_path: Sequence of keys to traverse in the ``llm`` block,
                        e.g. ``"pipeline", "researcher_ollama_model"``.
        default:       Hard-coded fallback when neither env var nor JSON key is set.
    """
    # 1. Environment variable wins
    val = os.environ.get(env_var)
    if val is not None:
        return int(val) if isinstance(default, int) else val

    # 2. Walk settings.json["llm"][...path...]
    node: dict | str | int = _settings().get("llm", {})
    for key in settings_path:
        if not isinstance(node, dict):
            break
        node = node.get(key, {})  # type: ignore[assignment]
    if node and not isinstance(node, dict):
        return int(node) if isinstance(default, int) else str(node)

    # 3. Hard-coded default
    return default


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


def _load_models() -> dict:
    """Return the parsed models.json, or {} if the file is absent/invalid."""
    path = Path(__file__).parent.parent / "config" / "models.json"
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {}


_MODELS: dict = {}  # populated lazily


def _models() -> dict:
    global _MODELS
    if not _MODELS:
        _MODELS = _load_models()
    return _MODELS


def _build_model_table() -> str:
    """Build a markdown model-reference section from models.json.

    Returns a ``## Models`` section with one table per category, ready to
    be spliced into any system prompt that contains ``{{MODEL_TABLE}}``.
    Returns an empty string if models.json is missing.
    """
    data = _models()
    if not data:
        return ""

    # Human-readable category titles in display order
    category_titles: dict[str, str] = {
        "unets":        "UNETs",
        "checkpoints":  "Checkpoints",
        "vae":          "VAE",
        "clip":         "CLIP",
        "controlnets":  "ControlNets",
        "loras":        "LoRAs",
    }

    lines: list[str] = [
        "## Models",
        "",
        "Use these paths verbatim. Never guess a path.",
        "Call `get_models_in_folder()` only for models not listed here.",
    ]

    for key, title in category_titles.items():
        entries = data.get(key)
        if not entries:
            continue
        col_w = max(len(k) for k in entries)
        lines.append("")
        lines.append(f"### {title}")
        lines.append(f"| {'shortname':<{col_w}} | path |")
        lines.append(f"|{'-' * (col_w + 2)}|------|")
        for shortname, path in entries.items():
            lines.append(f"| {shortname:<{col_w}} | {path} |")

    return "\n".join(lines)


# Map from resolved llm name → system-prompt markdown filename stem.
_SYSTEM_PROMPT_FILE: dict[str, str] = {
    "claude": "system_prompt.claude",
    "ollama": "system_prompt.qwencode",
    "researcher": "system_prompt.researcher",
    "brain": "system_prompt.brain",
}


def _load_system_prompt(llm: str) -> str:
    """Load the system prompt for *llm* and inject the model table."""
    stem = _SYSTEM_PROMPT_FILE.get(llm, f"system_prompt.{llm}")
    path = Path(__file__).parent.parent / "config" / f"{stem}.md"
    text = path.read_text(encoding="utf-8")
    if "{{MODEL_TABLE}}" in text:
        text = text.replace("{{MODEL_TABLE}}", _build_model_table())
    return text


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
        model_id = ollama_model or str(_cfg("OLLAMA_MODEL", "single_agent", "ollama_model", default="qwen3-vl:30b"))
        host = str(_cfg("OLLAMA_HOST", "ollama", "host", default="http://localhost:11434"))
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
        model_id=str(_cfg("ANTHROPIC_MODEL", "anthropic", "model", default="claude-haiku-4-5")),
        max_tokens=int(_cfg("ANTHROPIC_MAX_TOKENS", "anthropic", "max_tokens", default=4096)),
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


# ---------------------------------------------------------------------------
# Skills directory – lives at <project_root>/skills/
# ---------------------------------------------------------------------------
_SKILLS_DIR = Path(__file__).parent.parent / "skills"


def _make_agent(
    *,
    role: str,
    llm: str,
    system_prompt: str,
    tools: list,
    ollama_model: str | None = None,
    anthropic_model: str | None = None,
    max_tokens: int | None = None,
    plugins: list | None = None,
    **kwargs,
) -> Agent:
    """Internal helper that builds a model and wraps it in a Strands Agent.

    Args:
        role: Human-readable label used in log output (e.g. 'researcher', 'brain').
        llm: LLM backend – ``'claude'`` or ``'ollama'``.
        system_prompt: Full system prompt string.
        tools: List of @tool-decorated callables to give the agent.
        ollama_model: Override for the Ollama model ID.
        anthropic_model: Override for the Anthropic model ID.
        max_tokens: Override for Anthropic max_tokens.
        plugins: Optional list of Strands plugins (e.g. AgentSkills).
        **kwargs: Extra kwargs forwarded to the Strands Agent constructor.
    """
    llm = llm.strip().lower()
    if llm == "ollama":
        model_id = ollama_model or str(_cfg("OLLAMA_MODEL", "ollama", "model", default="qwen3-vl:30b"))
        host = str(_cfg("OLLAMA_HOST", "ollama", "host", default="http://localhost:11434"))
        _ensure_ollama_model(model_id, host)
        model = OllamaModel(host=host, model_id=model_id)
        print(f"[agentY:{role}] Using Ollama — {model_id}")
    else:
        model_id = anthropic_model or str(_cfg("ANTHROPIC_MODEL", "anthropic", "model", default="claude-haiku-4-5"))
        tokens = max_tokens or int(_cfg("ANTHROPIC_MAX_TOKENS", "anthropic", "max_tokens", default=4096))
        model = AnthropicModel(
            model_id=model_id,
            max_tokens=tokens,
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
        print(f"[agentY:{role}] Using Anthropic — {model_id}")

    window_size = int(_cfg("AGENT_HISTORY_WINDOW", "history_window", default=40))
    agent_kwargs: dict = {
        "model": model,
        "system_prompt": system_prompt,
        "tools": tools,
        "conversation_manager": SlidingWindowConversationManager(window_size=window_size),
    }
    if plugins:
        agent_kwargs["plugins"] = plugins
    agent_kwargs.update(kwargs)
    return Agent(**agent_kwargs)


# ---------------------------------------------------------------------------
# Public factory functions
# ---------------------------------------------------------------------------

def create_researcher_agent(
    llm: str | None = None,
    ollama_model: str | None = None,
    anthropic_model: str | None = None,
    **kwargs,
) -> Agent:
    """Create the Researcher agent.

    Defaults to Ollama (env: ``RESEARCHER_LLM``, then ``'ollama'``).
    Override the Ollama model with ``RESEARCHER_OLLAMA_MODEL`` or *ollama_model*.
    Override the Anthropic model with ``RESEARCHER_ANTHROPIC_MODEL`` or *anthropic_model*.

    Args:
        llm: ``'ollama'`` or ``'claude'``. Falls back to ``RESEARCHER_LLM`` env var.
        ollama_model: Ollama model override (e.g. ``'qwen3-coder:32b'``).
        anthropic_model: Anthropic model override (e.g. ``'claude-haiku-4-5'``).
        **kwargs: Forwarded to the Strands Agent constructor.
    """
    resolved_llm = llm or str(_cfg("RESEARCHER_LLM", "pipeline", "researcher_llm", default="ollama"))
    resolved_ollama = ollama_model or str(_cfg("RESEARCHER_OLLAMA_MODEL", "pipeline", "researcher_ollama_model", default="qwen3-coder:32b"))
    resolved_anthropic = anthropic_model or str(_cfg("RESEARCHER_ANTHROPIC_MODEL", "pipeline", "researcher_anthropic_model",
        default=_cfg("ANTHROPIC_MODEL", "anthropic", "model", default="claude-haiku-4-5")))
    system_prompt = _load_system_prompt("researcher")
    return _make_agent(
        role="researcher",
        llm=resolved_llm,
        system_prompt=system_prompt,
        tools=RESEARCHER_TOOLS,
        ollama_model=resolved_ollama,
        anthropic_model=resolved_anthropic,
        **kwargs,
    )


def create_brain_agent(
    llm: str | None = None,
    ollama_model: str | None = None,
    anthropic_model: str | None = None,
    **kwargs,
) -> Agent:
    """Create the Brain agent.

    Defaults to Claude (env: ``BRAIN_LLM``, then ``'claude'``).
    Override the Anthropic model with ``BRAIN_ANTHROPIC_MODEL`` or *anthropic_model*.
    Override the Ollama model with ``BRAIN_OLLAMA_MODEL`` or *ollama_model*.

    Args:
        llm: ``'claude'`` or ``'ollama'``. Falls back to ``BRAIN_LLM`` env var.
        anthropic_model: Anthropic model override (e.g. ``'claude-sonnet-4-5'``).
        ollama_model: Ollama model override.
        **kwargs: Forwarded to the Strands Agent constructor.
    """
    resolved_llm = llm or str(_cfg("BRAIN_LLM", "pipeline", "brain_llm", default="claude"))
    resolved_anthropic = anthropic_model or str(_cfg("BRAIN_ANTHROPIC_MODEL", "pipeline", "brain_anthropic_model",
        default=_cfg("ANTHROPIC_MODEL", "anthropic", "model", default="claude-haiku-4-5")))
    resolved_ollama = ollama_model or str(_cfg("BRAIN_OLLAMA_MODEL", "pipeline", "brain_ollama_model", default="qwen3-vl:30b"))
    system_prompt = _load_system_prompt("brain")

    # Load skills from the project-level skills/ directory.
    skills_plugins: list = []
    if _SKILLS_DIR.is_dir():
        skills_plugin = AgentSkills(skills=str(_SKILLS_DIR))
        skills_plugins.append(skills_plugin)
        loaded = [s.name for s in skills_plugin.get_available_skills()]
        if loaded:
            print(f"[agentY:brain] Loaded skills: {', '.join(loaded)}")

    return _make_agent(
        role="brain",
        llm=resolved_llm,
        system_prompt=system_prompt,
        tools=BRAIN_TOOLS,
        ollama_model=resolved_ollama,
        anthropic_model=resolved_anthropic,
        plugins=skills_plugins or None,
        **kwargs,
    )


def create_agent(llm: str | None = None, ollama_model: str | None = None, **kwargs) -> Agent:
    """Create and return the agentY Strands Agent with all ComfyUI tools.

    Legacy single-agent factory kept for backward compatibility.

    Args:
        llm: Which LLM backend to use: ``'claude'`` (default) or ``'ollama'``.
             Falls back to the ``AGENT_LLM`` env var, then to ``'claude'``.
        ollama_model: Ollama model name to use when ``llm='ollama'``. Overrides
                      the ``OLLAMA_MODEL`` env var when provided.
        **kwargs: Extra keyword arguments forwarded to the Strands Agent
                  constructor (e.g. to override the model or system prompt).
    """
    resolved_llm = llm or str(_cfg("AGENT_LLM", "single_agent", "llm", default="claude"))
    model = _build_model(resolved_llm, ollama_model=ollama_model)
    print(f"[agentY] Using LLM backend: {resolved_llm} ({model.__class__.__name__})")
    window_size = int(_cfg("AGENT_HISTORY_WINDOW", "history_window", default=40))

    # Load skills from the project-level skills/ directory.
    skills_plugins: list = []
    if _SKILLS_DIR.is_dir():
        skills_plugin = AgentSkills(skills=str(_SKILLS_DIR))
        skills_plugins.append(skills_plugin)
        loaded = [s.name for s in skills_plugin.get_available_skills()]
        if loaded:
            print(f"[agentY] Loaded skills: {', '.join(loaded)}")

    agent_kwargs = {
        "model": model,
        "system_prompt": _load_system_prompt(resolved_llm),
        "tools": ALL_TOOLS,
        "conversation_manager": SlidingWindowConversationManager(window_size=window_size),
    }
    if skills_plugins:
        agent_kwargs["plugins"] = skills_plugins
    agent_kwargs.update(kwargs)
    return Agent(**agent_kwargs)
