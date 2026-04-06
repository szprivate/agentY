"""
agentY – A ComfyUI agent built on the Strands Agents SDK.

Two-agent pipeline:
  • Researcher  – Ollama (default) or any LLM; pattern-matching/resolution only.
                  Produces a brainbriefing JSON.
  • Brain       – Claude (default) or any LLM; workflow assembly, execution, QA.
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
from strands.hooks.registry import HookRegistry
from strands.hooks.events import AfterToolCallEvent

from strands_tools import handoff_to_user

from src.utils.comfyui_interrupt_hook import ComfyUIInterruptHook

from src.tools import RESEARCHER_TOOLS, BRAIN_TOOLS, INFO_TOOLS, reset_patch_workflow_guard


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


def _parse_llm_setting(value: str) -> tuple[str, str]:
    """Split a 'provider,model' string into (provider, model).

    The model part is an empty string when the value contains no comma
    (e.g. when the value came from a plain RESEARCHER_LLM env var).
    """
    provider, _, model = value.partition(",")
    return provider.strip(), model.strip()


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
    "researcher": "system_prompt.researcher",
    "brain": "system_prompt.brain",
    "triage": "system_prompt.triage",
}


def _load_system_prompt(llm: str) -> str:
    """Load the system prompt for *llm* and inject the model table."""
    stem = _SYSTEM_PROMPT_FILE.get(llm, f"system_prompt.{llm}")
    path = Path(__file__).parent.parent / "config" / f"{stem}.md"
    print(f"[agentY] System prompt: {path.resolve()}")
    text = path.read_text(encoding="utf-8")
    if "{{MODEL_TABLE}}" in text:
        text = text.replace("{{MODEL_TABLE}}", _build_model_table())
    if "{{EXTERNAL_MODEL_DIR}}" in text:
        ext_dir = _models().get("external_model_dir", "")
        text = text.replace("{{EXTERNAL_MODEL_DIR}}", ext_dir)
    if "{{BRAINBRIEF_EXAMPLE}}" in text:
        example_path = Path(__file__).parent.parent / "config" / "brainbrief_example.json"
        if example_path.exists():
            example_text = example_path.read_text(encoding="utf-8")
            text = text.replace("{{BRAINBRIEF_EXAMPLE}}", example_text)
        else:
            print(f"[agentY] Warning: brainbrief_example.json not found at {example_path.resolve()}")
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



# Note: cost-estimation removed — only token counts are reported.


# ---------------------------------------------------------------------------
# Token-usage hook – prints token counts after every tool call
# ---------------------------------------------------------------------------

class TokenUsageHookProvider:
    """Prints a token-usage summary line after every tool call.

    Shows the delta (tokens consumed since the last report) and the
    running accumulated total so the operator can monitor costs in
    real time.
    """

    def __init__(self, role: str = "agent", is_ollama: bool = False) -> None:
        self._role = role
        self._is_ollama = is_ollama
        self._prev_in = 0
        self._prev_out = 0
        self._prev_cache_read = 0
        self._prev_cache_write = 0

    def register_hooks(self, registry: HookRegistry, **kwargs) -> None:  # noqa: ARG002
        registry.add_callback(AfterToolCallEvent, self._on_after_tool_call)

    def _on_after_tool_call(self, event: AfterToolCallEvent, **kwargs) -> None:  # noqa: ARG002
        try:
            usage = event.agent.event_loop_metrics.accumulated_usage
            in_tok = usage.get("inputTokens", 0)
            out_tok = usage.get("outputTokens", 0)
            cache_read = usage.get("cacheReadInputTokens", 0)
            cache_write = usage.get("cacheWriteInputTokens", 0)

            # Compute delta since last report
            d_in = in_tok - self._prev_in
            d_out = out_tok - self._prev_out
            d_cr = cache_read - self._prev_cache_read
            d_cw = cache_write - self._prev_cache_write
            self._prev_in = in_tok
            self._prev_out = out_tok
            self._prev_cache_read = cache_read
            self._prev_cache_write = cache_write

            tool_name = event.tool_use.get("name", "?")

            delta_parts = [f"+{d_in:,} in", f"+{d_out:,} out"]
            if d_cr:
                delta_parts.append(f"+{d_cr:,} cache hit")
            if d_cw:
                delta_parts.append(f"+{d_cw:,} cache write")

            total_parts = [f"{in_tok:,} in", f"{out_tok:,} out"]
            if cache_read:
                total_parts.append(f"{cache_read:,} cache hit")
            if cache_write:
                total_parts.append(f"{cache_write:,} cache write")
            # Cost estimation intentionally omitted; only token counts shown.

            print(
                f"\n\U0001fa99 [{self._role}] after {tool_name}: "
                f"{' / '.join(delta_parts)}  "
                f"(total: {' / '.join(total_parts)})"
            )
        except Exception:
            pass  # Never break the agent loop for cosmetic output


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
        "hooks": [TokenUsageHookProvider(role=role, is_ollama=(llm == "ollama"))],
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
    """Create the Researcher agent for experimental dual-agent pipeline.

    Defaults to Ollama (env: ``RESEARCHER_LLM``, then ``'ollama'``).
    Override the Ollama model with ``RESEARCHER_OLLAMA_MODEL`` or *ollama_model*.
    Override the Anthropic model with ``RESEARCHER_ANTHROPIC_MODEL`` or *anthropic_model*.

    Args:
        llm: ``'ollama'`` or ``'claude'``. Falls back to ``RESEARCHER_LLM`` env var.
        ollama_model: Ollama model override (e.g. ``'qwen3-coder:32b'``).
        anthropic_model: Anthropic model override (e.g. ``'claude-haiku-4-5'``).
        **kwargs: Forwarded to the Strands Agent constructor.
    """
    # Passing an Ollama model without an explicit LLM backend implies ollama.
    if ollama_model and llm is None:
        llm = "ollama"

    # Read combined 'provider,model' from settings (env var RESEARCHER_LLM still wins).
    _raw = str(_cfg("RESEARCHER_LLM", "pipeline", "researcher", default="ollama"))
    _settings_llm, _settings_model = _parse_llm_setting(_raw)
    resolved_llm = llm or _settings_llm or "ollama"

    # Model: CLI arg > provider-specific env var > model extracted from settings > hard default.
    if resolved_llm == "ollama":
        resolved_ollama = (
            ollama_model
            or os.environ.get("RESEARCHER_OLLAMA_MODEL")
            or _settings_model
            or "qwen3-coder:32b"
        )
        resolved_anthropic = (
            anthropic_model
            or os.environ.get("RESEARCHER_ANTHROPIC_MODEL")
            or str(_cfg("ANTHROPIC_MODEL", "anthropic", "model", default="claude-haiku-4-5"))
        )
    else:  # claude
        resolved_anthropic = (
            anthropic_model
            or os.environ.get("RESEARCHER_ANTHROPIC_MODEL")
            or _settings_model
            or str(_cfg("ANTHROPIC_MODEL", "anthropic", "model", default="claude-haiku-4-5"))
        )
        resolved_ollama = ollama_model or "qwen3-coder:32b"

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


def create_info_agent(
    ollama_model: str | None = None,
    **kwargs,
) -> Agent:
    """Create the Info agent — a lightweight Ollama agent that answers questions
    about available ComfyUI workflows, models, and capabilities.

    Defaults to the same Ollama model used by ``llm_functions``
    (``llm.pipeline.llm_functions`` in settings.json).
    Override with ``INFO_OLLAMA_MODEL`` env var or *ollama_model*.

    Args:
        ollama_model: Ollama model override (e.g. ``'qwen3.5:9b'``).
        **kwargs: Forwarded to the Strands Agent constructor.
    """
    resolved_model = (
        ollama_model
        or os.environ.get("INFO_OLLAMA_MODEL")
        or str(_cfg("INFO_OLLAMA_MODEL", "pipeline", "info", default=""))
        or str(_cfg("LLM_FUNCTIONS_MODEL", "pipeline", "llm_functions", default="qwen3.5:9b"))
    )
    system_prompt = _load_system_prompt("info")
    return _make_agent(
        role="info",
        llm="ollama",
        system_prompt=system_prompt,
        tools=INFO_TOOLS,
        ollama_model=resolved_model,
        **kwargs,
    )


def create_triage_agent(
    llm: str | None = None,
    ollama_model: str | None = None,
    anthropic_model: str | None = None,
    **kwargs,
) -> Agent:
    """Create the Triage agent — a stateless, tool-free intent classifier.

    Reads ``llm.pipeline.triage`` from settings.json (format: ``'provider,model'``,
    e.g. ``'ollama,qwen3:0.6b'`` or ``'claude,claude-haiku-4-5'``).
    Env var ``TRIAGE_LLM`` overrides the full setting; ``TRIAGE_OLLAMA_MODEL``
    or ``TRIAGE_ANTHROPIC_MODEL`` override just the model.

    The agent has no tools and no meaningful conversation history — it reads
    the user message (optionally prefixed with session context) and returns a
    JSON ``{"intent": "...", "confidence": 0.0–1.0}`` object.

    Args:
        llm: ``'ollama'`` or ``'claude'``. Falls back to ``TRIAGE_LLM`` env var.
        ollama_model: Ollama model override (e.g. ``'qwen3:0.6b'``).
        anthropic_model: Anthropic model override (e.g. ``'claude-haiku-4-5'``).
        **kwargs: Forwarded to the Strands Agent constructor.
    """
    if ollama_model and llm is None:
        llm = "ollama"

    # Read combined 'provider,model' from settings (env var TRIAGE_LLM still wins).
    _raw = str(_cfg("TRIAGE_LLM", "pipeline", "triage", default="ollama"))
    _settings_llm, _settings_model = _parse_llm_setting(_raw)
    resolved_llm = llm or _settings_llm or "ollama"

    if resolved_llm == "ollama":
        resolved_ollama = (
            ollama_model
            or os.environ.get("TRIAGE_OLLAMA_MODEL")
            or _settings_model
            or str(_cfg("LLM_FUNCTIONS_MODEL", "pipeline", "llm_functions", default="qwen3:0.6b"))
        )
        resolved_anthropic = (
            anthropic_model
            or os.environ.get("TRIAGE_ANTHROPIC_MODEL")
            or str(_cfg("ANTHROPIC_MODEL", "anthropic", "model", default="claude-haiku-4-5"))
        )
    else:  # claude
        resolved_anthropic = (
            anthropic_model
            or os.environ.get("TRIAGE_ANTHROPIC_MODEL")
            or _settings_model
            or str(_cfg("ANTHROPIC_MODEL", "anthropic", "model", default="claude-haiku-4-5"))
        )
        resolved_ollama = ollama_model or str(_cfg("LLM_FUNCTIONS_MODEL", "pipeline", "llm_functions", default="qwen3:0.6b"))

    system_prompt = _load_system_prompt("triage")
    agent = _make_agent(
        role="triage",
        llm=resolved_llm,
        system_prompt=system_prompt,
        tools=[handoff_to_user],
        ollama_model=resolved_ollama,
        anthropic_model=resolved_anthropic,
        **kwargs,
    )
    # Triage is single-turn and stateless — cap history to avoid stale
    # classification exchanges polluting future calls.
    agent.conversation_manager = SlidingWindowConversationManager(window_size=2)
    return agent


def create_brain_agent(
    llm: str | None = None,
    ollama_model: str | None = None,
    anthropic_model: str | None = None,
    **kwargs,
) -> Agent:
    """Create the Brain agent for experimental dual-agent pipeline.

    Defaults to Claude (env: ``BRAIN_LLM``, then ``'claude'``).
    Override the Anthropic model with ``BRAIN_ANTHROPIC_MODEL`` or *anthropic_model*.
    Override the Ollama model with ``BRAIN_OLLAMA_MODEL`` or *ollama_model*.

    Args:
        llm: ``'claude'`` or ``'ollama'``. Falls back to ``BRAIN_LLM`` env var.
        anthropic_model: Anthropic model override (e.g. ``'claude-sonnet-4-5'``).
        ollama_model: Ollama model override.
        **kwargs: Forwarded to the Strands Agent constructor.
    """
    # Passing an Ollama model without an explicit LLM backend implies ollama.
    if ollama_model and llm is None:
        llm = "ollama"
    # Reset the patch_workflow failure counter for each new brain session.
    reset_patch_workflow_guard()

    # Read combined 'provider,model' from settings (env var BRAIN_LLM still wins).
    _raw = str(_cfg("BRAIN_LLM", "pipeline", "brain", default="claude"))
    _settings_llm, _settings_model = _parse_llm_setting(_raw)
    resolved_llm = llm or _settings_llm or "claude"

    # Model: CLI arg > provider-specific env var > model extracted from settings > hard default.
    if resolved_llm == "claude":
        resolved_anthropic = (
            anthropic_model
            or os.environ.get("BRAIN_ANTHROPIC_MODEL")
            or _settings_model
            or str(_cfg("ANTHROPIC_MODEL", "anthropic", "model", default="claude-haiku-4-5"))
        )
        resolved_ollama = ollama_model or "qwen3-vl:30b"
    else:  # ollama
        resolved_ollama = (
            ollama_model
            or os.environ.get("BRAIN_OLLAMA_MODEL")
            or _settings_model
            or "qwen3-vl:30b"
        )
        resolved_anthropic = (
            anthropic_model
            or os.environ.get("BRAIN_ANTHROPIC_MODEL")
            or str(_cfg("ANTHROPIC_MODEL", "anthropic", "model", default="claude-haiku-4-5"))
        )
    system_prompt = _load_system_prompt("brain")

    # Load skills from the project-level skills/ directory.
    skills_plugins: list = []
    if _SKILLS_DIR.is_dir():
        skills_plugin = AgentSkills(skills=str(_SKILLS_DIR))
        skills_plugins.append(skills_plugin)
        loaded = [s.name for s in skills_plugin.get_available_skills()]
        if loaded:
            print(f"[agentY:brain] Loaded skills: {', '.join(loaded)}")

    # Merge the ComfyUI interrupt hook with any caller-supplied hooks so we
    # don't silently drop the TokenUsageHookProvider built by _make_agent.
    # We pass the combined list via kwargs; _make_agent's agent_kwargs.update()
    # will replace its default [TokenUsageHookProvider] with our explicit list.
    extra_hooks = kwargs.pop("hooks", [])
    brain_hooks = [TokenUsageHookProvider(role="brain"), ComfyUIInterruptHook(), *extra_hooks]

    return _make_agent(
        role="brain",
        llm=resolved_llm,
        system_prompt=system_prompt,
        tools=BRAIN_TOOLS,
        ollama_model=resolved_ollama,
        anthropic_model=resolved_anthropic,
        plugins=skills_plugins or None,
        hooks=brain_hooks,
        **kwargs,
    )


