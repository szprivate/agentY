"""
agentY – A ComfyUI agent built on the //eStrands Agents SDK.

Two-agent pipeline:
  • Query Templates  – Ollama (default) or any LLM; pattern-matching/resolution only.
                  Produces a brainbriefing JSON.
  • Assemble Workflow       – Claude (default) or any LLM; workflow assembly, execution, QA.
"""

import datetime
import json
import logging
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

# Removed `handoff_to_user` tool registration — not used by agents anymore.

from src.utils.comfyui_interrupt_hook import ComfyUIInterruptHook
from src.utils.costs import compute_cost_from_usage, _extract_meta

from src.tools import (
    QUERYTEMPLATES_TOOLS,
    ASSEMBLEWORKFLOW_TOOLS,
    ORCHESTRATOR_TOOLS,
    INFO_TOOLS,
    SEARCHWEB_TOOLS,
    PLANNER_TOOLS,
    LEARNINGS_TOOLS,
    VISION_AGENT_TOOLS,
    VIDEO_AGENT_TOOLS,
    CODER_TOOLS,
    FIX_WORKFLOW_ASSEMBLY_TOOLS,
    GENERATE_NEW_WORKFLOW_TOOLS,
    reset_patch_workflow_guard,
)
from src.steering import get_ASSEMBLEWORKFLOW_steering_handlers, get_QUERYTEMPLATES_steering_handlers


# ---------------------------------------------------------------------------
# Settings loader – reads config/settings.json once; env vars always win.
# ---------------------------------------------------------------------------

def _load_settings() -> dict:
    """Return the merged settings (TOML defaults ⊕ local JSON overrides), or {}.

    Delegates to :mod:`src.utils.settings`, the single settings loader for the app.
    """
    from src.utils.settings import load_settings
    return load_settings()


def _settings() -> dict:
    """The merged settings dict — the single shared cache (see src.utils.settings).

    Returns the SAME cached object on every call, so a runtime override written into
    it (e.g. by ``/switch_model``) is visible to every reader until the cache is
    invalidated (a settings save) or the process restarts.
    """
    return _load_settings()


def _cfg(env_var: str, *settings_path: str, default: str | int = "") -> str | int:
    """Return a config value with priority: env var > settings.json > default.

    Args:
        env_var:       Name of the environment variable to check first.
        *settings_path: Sequence of keys to traverse in the ``llm`` block,
                        e.g. ``"pipeline", "QUERYTEMPLATES_ollama_model"``.
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


# Which TIER each role inherits its model from when it has no explicit override.
#
# Thirteen per-role dropdowns were really expressing about five decisions — in a
# typical install every "cheap" role held the same value, every vision role held
# the same value, and so on. Tiers make that the default and leave the per-role
# override as the exception it should be.
#
# The groupings are deliberate, not cosmetic:
#   * research_assembly — reasoning over templates/graphs; the repair and
#     build-from-scratch specialists belong with the assembler they stand in for
#     (they already fell back to it, silently and invisibly, before this existed).
#   * fast_utility — short, cheap, high-frequency calls where a big model buys
#     nothing.
#   * vision — reads images the user supplied.
#   * qa_judge — kept OUT of `vision` on purpose: it judges finished work once per
#     output, and a weak judge either waves defects through or fails clean work and
#     triggers a pointless re-render. It is worth more than the input reader.
#   * coder / orchestrator — one role each; both usually want a specific model
#     rather than a shared tier.
_ROLE_TIERS: dict[str, str] = {
    "orchestrator": "orchestrator",
    "query_templates": "research_assembly",
    "assemble_workflow": "research_assembly",
    "fix_workflow_assembly": "research_assembly",
    "generate_new_workflow": "research_assembly",
    "info": "fast_utility",
    "search_web": "fast_utility",
    "planner": "fast_utility",
    "learnings": "fast_utility",
    "llm_functions": "fast_utility",
    "build_skill": "fast_utility",
    "executor_vision_model": "vision",
    "vision_agent": "vision",
    "video_agent": "vision",
    "qa_checker": "qa_judge",
    "coder": "coder",
}

# Human labels for the tier selectors (used by the settings UI via /agentY/settings).
TIER_LABELS: dict[str, str] = {
    "orchestrator": "Orchestrator — drives every turn",
    "research_assembly": "Research & assembly — templates, graph building, repair",
    "fast_utility": "Fast utility — short cheap calls (info, search, planner, …)",
    "vision": "Vision — reads input images and video",
    "qa_judge": "QA judge — grades finished outputs (worth a stronger model)",
    "coder": "Coder — writes scripts and custom nodes",
}


def role_model(role: str, default: str = "", env_var: str = "") -> str:
    """The ``'provider,model'`` for *role*: env var → override → tier → *default*.

    An empty per-role value means **inherit** rather than "unset", which is what
    lets the tier do its job; an existing settings.local.json that pins a role
    keeps winning, so nothing anyone has already configured changes meaning.
    """
    if env_var:
        val = os.environ.get(env_var)
        if val:
            return val
    explicit = str(_cfg("", "pipeline", role, default="") or "").strip()
    if explicit:
        return explicit
    tier = _ROLE_TIERS.get(role)
    if tier:
        inherited = str(_cfg("", "tiers", tier, default="") or "").strip()
        if inherited:
            return inherited
    return default


def _parse_llm_setting(value: str) -> tuple[str, str]:
    """Split a 'provider,model' string into (provider, model).

    The model part is an empty string when the value contains no comma
    (e.g. when the value came from a plain QUERYTEMPLATES_LLM env var).
    """
    provider, _, model = value.partition(",")
    return provider.strip(), model.strip()


# Provider tokens routed through Alibaba Model Studio (DashScope)'s
# OpenAI-compatible endpoint. Use any of them in settings.json, e.g.
# "query_templates": "dashscope,qwen-plus".
_DASHSCOPE_PROVIDERS = {"dashscope", "modelstudio", "qwen", "alibaba"}
# OpenAI and Google Gemini are also driven through the OpenAI-compatible client
# (Gemini via its OpenAI-compat endpoint), gated on their own API keys. Use e.g.
# "orchestrator": "openai,gpt-4o" or "orchestrator": "google,gemini-2.5-pro".
_OPENAI_PROVIDERS = {"openai", "gpt"}
_GEMINI_PROVIDERS = {"google", "gemini"}


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

    def format_chunk(self, event):  # type: ignore[override]
        """Surface Anthropic prompt-cache token counts in the usage metadata.

        The upstream Strands Anthropic model maps only ``input_tokens`` /
        ``output_tokens`` and silently drops ``cache_read_input_tokens`` and
        ``cache_creation_input_tokens``.  Because this model caches the system
        prompt and the entire tools block, the cached tokens — the bulk of input
        on warm calls, plus the 1.25x cache-write on the cold call — would
        otherwise be billed at $0.  We re-attach them as ``cacheReadInputTokens``
        / ``cacheWriteInputTokens`` so the Strands metrics layer accumulates them
        (it already sums those keys) and the cost accounting can price them.
        """
        chunk = super().format_chunk(event)
        if isinstance(event, dict) and event.get("type") == "metadata":
            usage = event.get("usage") or {}
            meta_usage = chunk.get("metadata", {}).get("usage")
            if isinstance(meta_usage, dict):
                cache_read = usage.get("cache_read_input_tokens")
                cache_write = usage.get("cache_creation_input_tokens")
                if cache_read is not None:
                    meta_usage["cacheReadInputTokens"] = int(cache_read or 0)
                if cache_write is not None:
                    meta_usage["cacheWriteInputTokens"] = int(cache_write or 0)
        return chunk


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

    # Legacy models.json carried curated shortname->path maps under these keys.
    # When they are absent, derive them from the auto-scanned `available`
    # inventory (which reflects EVERY ComfyUI search path, incl. extra drives
    # like L:/) so the researcher always sees what is actually installed.
    if not any(isinstance(data.get(k), dict) and data.get(k) for k in category_titles):
        available = data.get("available", {})
        _folder_map = {
            "unets":       ("diffusion_models", "unet", "unet_gguf"),
            "checkpoints": ("checkpoints",),
            "vae":         ("vae",),
            "clip":        ("text_encoders", "clip", "clip_gguf"),
            "controlnets": ("controlnet",),
            "loras":       ("loras",),
        }
        derived: dict[str, dict[str, str]] = {}
        for cat, folders in _folder_map.items():
            entries: dict[str, str] = {}
            for folder in folders:
                for path in available.get(folder, []):
                    if isinstance(path, str):
                        entries.setdefault(Path(path).name, path)
            if entries:
                derived[cat] = entries
        data = {**data, **derived}

    lines: list[str] = [
        "## Models",
        "",
        "Use these paths verbatim — they come from the Query Templates' brainbriefing.",
        "Do NOT check, download, or guess model paths yourself.",
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
    "query_templates": "system_prompt.query_templates",
    "assemble_workflow": "system_prompt.assemble_workflow",
    "orchestrator": "system_prompt.orchestrator",
    "planner": "system_prompt.planner",
    "info": "system_prompt.info",
    "search_web": "system_prompt.search_web",
    "learnings": "system_prompt.learnings",
    "qa_checker": "system_prompt.qaChecker",
    "vision_agent": "system_prompt.vision_agent",
    "video_agent": "system_prompt.video_agent",
    "coder": "system_prompt.coder",
}


def _load_system_prompt(llm: str) -> str:
    """Load the system prompt for *llm* and inject the model table."""
    # Allow override of system prompt filenames from config/settings.json.
    # Settings may provide exact filenames (with or without .md) under
    # the `system_prompts` mapping. Fall back to the built-in stems.
    cfg_map = _settings().get("system_prompts", {})
    configured = cfg_map.get(llm)
    if configured:
        stem = configured
    else:
        stem = _SYSTEM_PROMPT_FILE.get(llm, f"system_prompt.{llm}")

    # Accept either 'name' or 'name.md' in settings and normalize to a path.
    if stem.endswith(".md"):
        filename = stem
    else:
        filename = f"{stem}.md"
    config_dir = Path(__file__).parent.parent / "config"
    prompts_dir = config_dir / "system_prompts"
    # Prefer prompts in ./config/system_prompts/, fall back to ./config/ directly.
    candidate = prompts_dir / filename
    if candidate.exists():
        path = candidate
    else:
        path = config_dir / filename
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
    if "{{DECISION_EXAMPLE}}" in text:
        example_path = Path(__file__).parent.parent / "config" / "researcher_decision_example.json"
        if example_path.exists():
            text = text.replace("{{DECISION_EXAMPLE}}", example_path.read_text(encoding="utf-8"))
        else:
            print(f"[agentY] Warning: researcher_decision_example.json not found at {example_path.resolve()}")
    return text


_THINK_SUPPORT_CACHE: dict[str, bool] = {}


def _ollama_supports_thinking(model_id: str, host: str) -> bool:
    """Return True if the Ollama model advertises the 'thinking' capability.

    Passing `think` to a non-thinking model (e.g. qwen3-coder) is a 400 error,
    so callers gate the flag on this. Cached per model; on any lookup failure
    returns False (safest — never send an unsupported param)."""
    if model_id in _THINK_SUPPORT_CACHE:
        return _THINK_SUPPORT_CACHE[model_id]
    supported = False
    try:
        import ollama  # noqa: PLC0415
        resp = ollama.Client(host).show(model_id)
        caps = getattr(resp, "capabilities", None)
        if caps is None and isinstance(resp, dict):
            caps = resp.get("capabilities")
        supported = "thinking" in (caps or [])
    except Exception:  # noqa: BLE001
        supported = False
    _THINK_SUPPORT_CACHE[model_id] = supported
    return supported


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
    """Prints a token-usage summary line after every tool call and appends to
    ./logs/tokens_usage.log.

    Shows the delta (tokens consumed since the last report) and the
    running accumulated total so the operator can monitor costs in
    real time.
    """

    @staticmethod
    def _resolve_log_path() -> Path:
        _project_root = Path(__file__).parent.parent
        rel = _settings().get("tokens_usage_log", "./.logs/tokens_usage.log")
        return _project_root / rel

    _log_path: Path = Path(__file__).parent.parent / ".logs" / "tokens_usage.log"

    def __init__(self, role: str = "agent", is_ollama: bool = False) -> None:
        self.__class__._log_path = self._resolve_log_path()
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

            # Detect skill name for script-based skills (run_script)
            tool_display = tool_name
            try:
                import re as _re

                if "run_script" in (tool_name or "").lower():
                    tool_input = event.tool_use.get("input") or event.tool_use.get("arguments") or ""
                    cmd = ""
                    if isinstance(tool_input, dict):
                        cmd = tool_input.get("command") or ""
                    elif isinstance(tool_input, str):
                        cmd = tool_input
                    else:
                        try:
                            cmd = str(tool_input)
                        except Exception:
                            cmd = ""

                    m = _re.search(r"skills[\\/](?P<name>[a-z0-9\-]+)", cmd, _re.I)
                    if m:
                        skill_name = m.group("name")
                        tool_display = f"{tool_name} (skill:{skill_name})"
            except Exception:
                tool_display = tool_name

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

            # Per-tool token deltas are printed to the console for live
            # debugging only.  Cost is intentionally NOT shown here — the
            # single whole-generation cost is reported once at the end of the
            # turn (see chainlit_app / main.py).  Full per-call cost still goes
            # to the tokens_usage.log file below for offline analysis.
            summary_line = (
                f"\U0001fa99 [{self._role}] after {tool_display}: "
                f"{' / '.join(delta_parts)}  "
                f"(total: {' / '.join(total_parts)})"
            )
            print(f"\n{summary_line}")

            # ── Append to log file ────────────────────────────────────────
            try:
                self._log_path.parent.mkdir(parents=True, exist_ok=True)
                ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                # Try to compute cost for this accumulated usage; ignore failures
                cost_str = ""
                try:
                    cost_val, total_tokens = compute_cost_from_usage(usage, event.agent)
                    cost_str = f" cost=${cost_val:.2f}/tokens={total_tokens}"
                except Exception:
                    cost_str = ""

                # Record which model produced this usage so the Token Usage
                # overview can filter by model. Older log lines predate this
                # field; the parser falls back to the role for those.
                model_str = ""
                try:
                    _prov, _mid, _ = _extract_meta(event.agent)
                    if _prov or _mid:
                        model_str = f"{_prov}/{_mid}"
                except Exception:
                    model_str = ""

                log_entry = (
                    f"{ts} [{self._role}] tool={tool_display} "
                    f"delta=+{d_in}in/+{d_out}out/+{d_cr}cache_read/+{d_cw}cache_write"
                    f"  total={in_tok}in/{out_tok}out/{cache_read}cache_read/{cache_write}cache_write"
                    f"{cost_str} model={model_str}\n"
                )
                with self._log_path.open("a", encoding="utf-8") as f:
                    f.write(log_entry)
            except Exception:
                pass  # Never break the agent loop for file I/O errors
        except Exception:
            pass  # Never break the agent loop for cosmetic output


# ---------------------------------------------------------------------------
# Tool-activity hook – surfaces the agent's tool calls + results to the chat UI
# ---------------------------------------------------------------------------

def _summarize_tool_result(result: object, cap: int = 800) -> str:
    """Reduce a Strands ToolResult to a short display string.

    A ToolResult is ``{"status": ..., "content": [{"text"|"json": ...}, ...]}``.
    Joins the content blocks' text/json into one string, truncated to *cap*.
    """
    try:
        parts: list[str] = []
        content = result.get("content") if isinstance(result, dict) else None
        for block in (content or []):
            if not isinstance(block, dict):
                parts.append(str(block))
            elif "text" in block:
                parts.append(str(block["text"]))
            elif "json" in block:
                parts.append(json.dumps(block["json"], ensure_ascii=False))
            else:
                parts.append(str(block))
        text = " ".join(p for p in parts if p).strip()
        if not text and isinstance(result, dict):
            text = str(result.get("status", ""))
    except Exception:  # noqa: BLE001
        text = str(result)
    return _truncate_activity(text, cap)


def _truncate_activity(s: object, cap: int = 800) -> str:
    s = str(s)
    return s if len(s) <= cap else s[:cap] + f" …(+{len(s) - cap} chars)"


class ToolActivityHookProvider:
    """Pushes each tool call (name + input) and result to ``tool_activity`` so the
    chat UI can render what the agent is doing, inline in the conversation.

    ``role`` labels which agent is running the tool (e.g. ``orchestrator``,
    ``query_templates``, ``subagent``); the side panel renders it in brackets
    before the tool name, so it's clear which agent made each call.
    """

    def __init__(self, role: str = "agent") -> None:
        self._role = role

    def register_hooks(self, registry: HookRegistry, **kwargs) -> None:  # noqa: ARG002
        from strands.hooks.events import BeforeToolCallEvent, AfterToolCallEvent
        registry.add_callback(BeforeToolCallEvent, self._on_before)
        registry.add_callback(AfterToolCallEvent, self._on_after)

    def _on_before(self, event, **kwargs) -> None:  # noqa: ANN001, ARG002
        try:
            from src.utils.tool_activity import push
            tu = getattr(event, "tool_use", None) or {}
            push({
                "phase": "call",
                "id": tu.get("toolUseId", ""),
                "agent": self._role,
                "name": tu.get("name", "tool"),
                "input": _truncate_activity(tu.get("input", {})),
            })
        except Exception:  # noqa: BLE001
            pass

    def _on_after(self, event, **kwargs) -> None:  # noqa: ANN001, ARG002
        try:
            from src.utils.tool_activity import push
            tu = getattr(event, "tool_use", None) or {}
            exc = getattr(event, "exception", None)
            if exc is not None:
                summary = f"error: {exc}"
            else:
                summary = _summarize_tool_result(getattr(event, "result", None))
            push({
                "phase": "result",
                "id": tu.get("toolUseId", ""),
                "agent": self._role,
                "name": tu.get("name", "tool"),
                "result": summary,
            })
        except Exception:  # noqa: BLE001
            pass


class MagnificWatchHookProvider:
    """Auto-register async Magnific creations for background auto-drop.

    Magnific generation tools (``magnific__video_generate`` / ``image_generate`` /
    upscale, …) return immediately with a queued ``creations[].identifier``; the
    render finishes minutes later. On each ``AfterToolCallEvent`` for a
    ``magnific__`` tool, this inspects the result for queued creation ids and hands
    them to :func:`src.utils.magnific_watch.register_from_result`, which watches
    each to completion and drops the finished asset onto the canvas + pops a note.
    Deterministic — it never relies on the model to call a watch tool.
    """

    def register_hooks(self, registry: HookRegistry, **kwargs) -> None:  # noqa: ARG002
        from strands.hooks.events import AfterToolCallEvent
        registry.add_callback(AfterToolCallEvent, self._on_after)

    def _on_after(self, event, **kwargs) -> None:  # noqa: ANN001, ARG002
        try:
            tu = getattr(event, "tool_use", None) or {}
            name = tu.get("name", "")
            if not name.startswith("magnific__"):
                return
            if getattr(event, "exception", None) is not None:
                return
            from src.utils import magnific_watch
            n = magnific_watch.register_from_result(getattr(event, "result", None), tool=name)
            # Log unconditionally (even n==0) so a *missed* auto-drop is diagnosable:
            # n==0 means this magnific__ result carried no queued creation id to watch
            # (already terminal, unparseable, or a non-generating tool). Goes to the
            # persistent .logs/magnific_watch.log via the shared watcher logger.
            logging.getLogger("agentY.magnific_watch").info(
                "hook: %s → registered %d watcher(s)", name, n)
        except Exception as exc:  # noqa: BLE001
            logging.getLogger("agentY.magnific_watch").warning(
                "hook: register_from_result failed for a magnific__ tool: %s",
                exc, exc_info=True)


# ---------------------------------------------------------------------------
# Skills directory – lives at <project_root>/skills/
# ---------------------------------------------------------------------------
_SKILLS_DIR = Path(__file__).parent.parent / "skills"

# Runtime-authored ("scratch") skills the orchestrator writes via create_skill.
# Kept in a subdirectory so they are easy to identify/clean and never collide
# with the curated project skills. Registered as a second AgentSkills source
# (the skill loader only discovers immediate child dirs, so a nested subdir must
# be passed as its own source).
_SCRATCH_SKILLS_DIR = _SKILLS_DIR / "_scratch"

# Curated orchestrator-only skills, grouped in their own folder (e.g. self-extension
# moved out of the base system prompt). Like _scratch, it's a nested subdir of
# skills/, so the loader only sees its child skills when it's passed as its own
# source — hence it's added as a third AgentSkills root below.
_ORCH_SKILLS_DIR = _SKILLS_DIR / "orchestrator-skills"

# Per-agent skill scoping. AgentSkills(skills=…) accepts individual skill DIRS, not
# just parent roots, so each agent gets an explicit allowlist instead of the whole
# skills/ folder. This stops a builder from ever seeing (and mis-activating) an
# orchestrator/story skill and, more importantly, stops the router from seeing the
# assembly skills it must delegate — while shared skills (batch-handoff) simply
# appear in both lists.
#
# The workflow builders (query_templates / assemble_workflow / fix_workflow /
# generate_new). NOTE: 'output-paths' is deliberately NOT here — its media-kind
# routing rules are now baked into the assembly base prompts so they ALWAYS apply,
# and 'custom-node-from-github' is omitted because it is only ever baked into the
# coder subagent (via _load_subagent_skill), never activated from a listing.
_ASSEMBLY_SKILL_NAMES = [
    "annotation", "assemble-from-template", "assemble-new-workflow",
    "assemble-workflow-learnings",
    "flux-sampling", "image-batch", "kling-multishot", "prompting", "recipe",
    "upscale-ultimatesd", "video-gemini-motionpromptgeneration",
    "workflow-templates", "batch-handoff",
]
# Orchestrator-owned skills (router / writing / batch handoff / image prep). The
# self-extension skill lives under _ORCH_SKILLS_DIR and is added separately.
_ORCH_SKILL_NAMES = [
    "story-synopsis", "story-scene", "story-storyboard",
    "spawn-subagent", "batch-handoff", "image-downsize",
]


def _skill_sources(names: list[str]) -> list[str]:
    """Turn skill dir names under skills/ into explicit AgentSkills sources, keeping
    only those that exist on disk."""
    return [str(_SKILLS_DIR / n) for n in names if (_SKILLS_DIR / n / "SKILL.md").is_file()]


def _orchestrator_skill_sources() -> list[str]:
    """The orchestrator's AgentSkills sources: its owned skills + the grouped
    orchestrator-skills folder (self-extension) + the runtime _scratch dir. Used at
    build time AND by orchestration._rescan_skills so a live create_skill re-scan
    preserves the exact scoping instead of re-widening to the whole skills/ dir."""
    return _skill_sources(_ORCH_SKILL_NAMES) + [str(_ORCH_SKILLS_DIR), str(_SCRATCH_SKILLS_DIR)]


def _make_agent(
    *,
    role: str,
    llm: str,
    system_prompt: str,
    tools: list,
    ollama_model: str | None = None,
    anthropic_model: str | None = None,
    dashscope_model: str | None = None,
    max_tokens: int | None = None,
    plugins: list | None = None,
    **kwargs,
) -> Agent:
    """Internal helper that builds a model and wraps it in a Strands Agent.

    Args:
        role: Human-readable label used in log output (e.g. 'query_templates', 'assemble_workflow').
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
        from src.utils.settings import ollama_host as _ollama_host
        host = _ollama_host()
        # Ollama defaults num_ctx to ~4k, which truncates the large agent prompts
        # (query_templates/assemble_workflow carry the full model table) and yields
        # malformed brainbriefings. Give the local model a big context window.
        num_ctx = int(_cfg("OLLAMA_NUM_CTX", "ollama", "num_ctx", default=32768))
        # From-scratch builds emit a long tool-call sequence; a low output cap
        # trips MaxTokensReachedException, so raise num_predict/max_tokens too.
        ol_max_tokens = int(_cfg("OLLAMA_MAX_TOKENS", "ollama", "max_tokens", default=12288))
        # qwen3.6 is a reasoning model: its <think> chain-of-thought consumes output
        # tokens and, on complex brainbriefings, exhausts max_tokens -> the
        # unrecoverable MaxTokensReachedException (observed ~50% of text-to-image
        # recipes). The briefing/patch work is structured extraction, not deep
        # reasoning, so disable thinking. think=False is the root-cause fix; a mild
        # repeat_penalty guards residual repetition without degrading JSON (1.3 did).
        repeat_penalty = float(_cfg("OLLAMA_REPEAT_PENALTY", "ollama", "repeat_penalty", default=1.1))
        _think_cfg = _cfg("OLLAMA_THINK", "ollama", "think", default=False)
        think = _think_cfg if isinstance(_think_cfg, bool) else \
            str(_think_cfg).strip().lower() in ("1", "true", "yes", "on")
        _ensure_ollama_model(model_id, host)
        # Only pass the `think` flag to models that actually support thinking:
        # a non-thinking model (e.g. qwen3-coder) rejects it with a 400.
        _add_args = {}
        if _ollama_supports_thinking(model_id, host):
            _add_args["think"] = think
        model = OllamaModel(host=host, model_id=model_id, max_tokens=ol_max_tokens,
                            options={"num_ctx": num_ctx, "repeat_penalty": repeat_penalty},
                            additional_args=_add_args)
        print(f"[agentY:{role}] Using Ollama — {model_id} (num_ctx={num_ctx}, "
              f"max_tokens={ol_max_tokens}, repeat_penalty={repeat_penalty}, "
              f"think={_add_args.get('think', 'n/a')})")
    elif llm in _DASHSCOPE_PROVIDERS:
        # Alibaba Model Studio (DashScope) via its OpenAI-compatible endpoint.
        from strands.models.openai import OpenAIModel
        model_id = dashscope_model or str(_cfg("DASHSCOPE_MODEL", "dashscope", "model", default="qwen-plus"))
        base_url = str(_cfg("DASHSCOPE_BASE_URL", "dashscope", "base_url",
                            default="https://dashscope-intl.aliyuncs.com/compatible-mode/v1"))
        api_key = os.environ.get("DASHSCOPE_API_KEY") or os.environ.get("ALIBABA_API_KEY") or ""
        if not api_key:
            print(f"[agentY:{role}] WARNING: DASHSCOPE_API_KEY not set — Model Studio calls will fail.")
        ds_max_tokens = max_tokens or int(_cfg("DASHSCOPE_MAX_TOKENS", "dashscope", "max_tokens", default=8192))
        # Qwen3 "thinking" models (e.g. qwen3.7-plus) emit reasoning_content, which
        # the Chat Completions API rejects when echoed back on multi-turn requests
        # — and the pipeline is a tool-calling loop (multi-turn). The work here is
        # structured extraction/assembly, not deep reasoning, so disable thinking
        # by default. Override with DASHSCOPE_ENABLE_THINKING or
        # dashscope.enable_thinking in settings.json.
        _ds_think_raw = _cfg("DASHSCOPE_ENABLE_THINKING", "dashscope", "enable_thinking", default="false")
        _ds_think = str(_ds_think_raw).strip().lower() in ("1", "true", "yes", "on")
        model = OpenAIModel(
            client_args={"api_key": api_key, "base_url": base_url},
            model_id=model_id,
            params={"max_tokens": ds_max_tokens, "extra_body": {"enable_thinking": _ds_think}},
        )
        print(f"[agentY:{role}] Using Alibaba Model Studio (DashScope) — {model_id} (thinking={_ds_think})")
    elif llm in _OPENAI_PROVIDERS:
        # OpenAI proper, via the same OpenAI-compatible Strands client. The model id
        # threads in through ``dashscope_model`` (the shared openai-compatible model
        # slot the factories populate from settings).
        from strands.models.openai import OpenAIModel
        model_id = dashscope_model or str(_cfg("OPENAI_MODEL", "openai", "model", default="gpt-4o"))
        base_url = str(_cfg("OPENAI_BASE_URL", "openai", "base_url", default="https://api.openai.com/v1"))
        api_key = os.environ.get("OPENAI_API_KEY") or ""
        if not api_key:
            print(f"[agentY:{role}] WARNING: OPENAI_API_KEY not set — OpenAI calls will fail.")
        oc_max_tokens = max_tokens or int(_cfg("OPENAI_MAX_TOKENS", "openai", "max_tokens", default=8192))
        model = OpenAIModel(
            client_args={"api_key": api_key, "base_url": base_url},
            model_id=model_id,
            params={"max_tokens": oc_max_tokens},
        )
        print(f"[agentY:{role}] Using OpenAI — {model_id}")
    elif llm in _GEMINI_PROVIDERS:
        # Google Gemini through its OpenAI-compatible endpoint (same client). Key
        # from GEMINI_API_KEY or GOOGLE_API_KEY.
        from strands.models.openai import OpenAIModel
        model_id = dashscope_model or str(_cfg("GEMINI_MODEL", "google", "model", default="gemini-2.5-flash"))
        base_url = str(_cfg("GEMINI_BASE_URL", "google", "base_url",
                            default="https://generativelanguage.googleapis.com/v1beta/openai/"))
        api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY") or ""
        if not api_key:
            print(f"[agentY:{role}] WARNING: GEMINI_API_KEY/GOOGLE_API_KEY not set — Gemini calls will fail.")
        oc_max_tokens = max_tokens or int(_cfg("GEMINI_MAX_TOKENS", "google", "max_tokens", default=8192))
        model = OpenAIModel(
            client_args={"api_key": api_key, "base_url": base_url},
            model_id=model_id,
            params={"max_tokens": oc_max_tokens},
        )
        print(f"[agentY:{role}] Using Google Gemini — {model_id}")
    else:
        model_id = anthropic_model or str(_cfg("ANTHROPIC_MODEL", "anthropic", "model", default="claude-haiku-4-5"))
        tokens = max_tokens or int(_cfg("ANTHROPIC_MAX_TOKENS", "anthropic", "max_tokens", default=4096))
        _an_params: dict = {
            "system": [
                {
                    "type": "text",
                    "text": system_prompt,
                    "cache_control": {"type": "ephemeral"},
                }
            ]
        }
        # Extended thinking (reasoning). Off by default — parallels the Ollama
        # `think` and DashScope `enable_thinking` toggles so the switch is
        # available for every provider. When on, Claude reasons before answering;
        # the budget must be < max_tokens, so bump max_tokens if it's too small.
        _an_think_raw = _cfg("ANTHROPIC_THINK", "anthropic", "think", default=False)
        _an_think = _an_think_raw if isinstance(_an_think_raw, bool) else \
            str(_an_think_raw).strip().lower() in ("1", "true", "yes", "on")
        if _an_think:
            budget = max(1024, min(4096, tokens - 1024))
            if tokens <= budget:
                tokens = budget + 1024
            _an_params["thinking"] = {"type": "enabled", "budget_tokens": budget}
        model = AnthropicModel(
            model_id=model_id,
            max_tokens=tokens,
            # Disable Strands' native count_tokens API for context-window
            # estimation. On the warm path it calls count_tokens() with only the
            # messages AFTER the last assistant turn (event_loop.py); that slice
            # starts with a tool_result whose tool_use is in the excluded message,
            # which Anthropic rejects with HTTP 400 ("each tool_result must have a
            # corresponding tool_use"). The error is caught and falls back to local
            # estimation anyway, so the native call only adds a failing round-trip
            # and log noise on every tool call. Local estimation has no effect on
            # cost accounting (which reads real accumulated_usage from responses).
            use_native_token_count=False,
            params=_an_params,
        )
        print(f"[agentY:{role}] Using Anthropic — {model_id} (thinking={_an_think})")

    window_size = int(_cfg("AGENT_HISTORY_WINDOW", "history_window", default=40))
    agent_kwargs: dict = {
        "model": model,
        "system_prompt": system_prompt,
        "tools": tools,
        "conversation_manager": SlidingWindowConversationManager(window_size=window_size),
        "hooks": [TokenUsageHookProvider(role=role, is_ollama=(llm == "ollama"))],
        # Disable Strands' default PrintingCallbackHandler.  Both entry points
        # consume agents via stream_async — Chainlit renders the yielded events
        # in the web UI, and the CLI (Pipeline.run) collects them into the
        # printed response — so the built-in console echo only duplicates that
        # output.  Per-tool token usage is still logged via the hook above, and
        # triage/planner/etc. output is still written to message_history.log.
        # A caller may re-enable it by passing callback_handler=... in kwargs.
        "callback_handler": None,
    }
    if plugins:
        agent_kwargs["plugins"] = plugins
    agent_kwargs.update(kwargs)
    # Every agent — not just the orchestrator — reports its tool calls to the
    # shared tool_activity buffer, so the chat panel shows ANY tool call the
    # pipeline makes (delegate specialists, the executor, subagents), matching
    # what the CLI prints. Append centrally, de-duped, so a factory that already
    # supplied one (the orchestrator) isn't double-hooked.
    _hooks = list(agent_kwargs.get("hooks") or [])
    if not any(isinstance(h, ToolActivityHookProvider) for h in _hooks):
        _hooks.append(ToolActivityHookProvider(role=role))
        agent_kwargs["hooks"] = _hooks
    agent = Agent(**agent_kwargs)
    # Attach light-weight cost metadata so callers can compute run cost.
    try:
        agent._cost_meta = {
            "provider": llm,
            "model_id": model_id,
            "is_ollama": (llm == "ollama"),
        }
        agent._is_claude = (llm != "ollama")
    except Exception:
        pass
    return agent


# ---------------------------------------------------------------------------
# Public factory functions
# ---------------------------------------------------------------------------

def create_vision_agent(
    ollama_model: str | None = None,
    anthropic_model: str | None = None,
    **kwargs,
) -> Agent:
    """Create the Vision Agent – stateless, single-shot image analysis.

    Returns a fully configured Strands :class:`~strands.Agent` using the same
    Ollama vision model that the Executor uses for QA, but with no tools and
    a minimal history window so every call is independent.

    Configuration (in priority order):
    1. ``VISION_AGENT_MODEL`` env var
    2. ``llm.pipeline.vision_agent`` in settings.json (format: ``'provider,model'``)
    3. ``llm.pipeline.executor_vision_model`` – the shared vision model fallback
    4. Hard default: ``'gemma4:26b'``

    Tools: :data:`src.tools.VISION_AGENT_TOOLS` (empty – vision agent is
    stateless and performs no tool calls).

    Args:
        ollama_model:    Ollama model override.
        anthropic_model: Anthropic model override (if using Claude for vision).
        **kwargs:        Forwarded to the Strands Agent constructor.
    """
    # Read combined 'provider,model' from settings; VISION_AGENT_MODEL env var wins.
    _env_model = os.environ.get("VISION_AGENT_MODEL", "")
    _raw = str(role_model("vision_agent", default=""))
    if not _raw:
        _raw = str(role_model("executor_vision_model", default="ollama,gemma4:26b"))
        if "," not in _raw:
            _raw = f"ollama,{_raw}"
    _settings_llm, _settings_model = _parse_llm_setting(_raw)
    resolved_llm = _settings_llm or "ollama"

    if resolved_llm == "ollama":
        resolved_ollama = (
            ollama_model
            or _env_model
            or _settings_model
            or str(role_model("executor_vision_model", default="gemma4:26b"))
        )
        resolved_anthropic = (
            anthropic_model
            or str(_cfg("ANTHROPIC_MODEL", "anthropic", "model", default="claude-haiku-4-5"))
        )
    else:  # claude
        resolved_anthropic = (
            anthropic_model
            or _env_model
            or _settings_model
            or str(_cfg("ANTHROPIC_MODEL", "anthropic", "model", default="claude-haiku-4-5"))
        )
        resolved_ollama = ollama_model or "gemma4:26b"

    system_prompt = _load_system_prompt("vision_agent")
    agent = _make_agent(
        role="vision_agent",
        llm=resolved_llm,
        dashscope_model=_settings_model,
        system_prompt=system_prompt,
        tools=VISION_AGENT_TOOLS,
        ollama_model=resolved_ollama,
        anthropic_model=resolved_anthropic,
        **kwargs,
    )
    # Stateless: keep only the immediate exchange (mirrors Planner behaviour).
    agent.conversation_manager = SlidingWindowConversationManager(window_size=2)
    return agent


def create_video_agent(
    dashscope_model: str | None = None,
    ollama_model: str | None = None,
    anthropic_model: str | None = None,
    **kwargs,
) -> Agent:
    """Create the Video Agent — stateless, single-shot video understanding.

    Sampled video frames are handed to a vision-language model (default Qwen2.5-VL
    on Alibaba Model Studio / DashScope) via the ``analyze_video`` tool, which reads
    a frame sequence as a video. No tools, minimal history — every call is
    independent (mirrors the Vision Agent).

    Configuration (in priority order):
    1. ``VIDEO_AGENT_MODEL`` env var (``'provider,model'`` or bare model)
    2. ``llm.pipeline.video_agent`` in settings.json (``'provider,model'``)
    3. Hard default: ``'dashscope,qwen2.5-vl-72b-instruct'``

    Args:
        dashscope_model: DashScope/OpenAI-compatible model id override.
        ollama_model:    Ollama model override (if the setting selects ollama).
        anthropic_model: Anthropic model override (if the setting selects claude).
        **kwargs:        Forwarded to the Strands Agent constructor.
    """
    _env_model = os.environ.get("VIDEO_AGENT_MODEL", "")
    _raw = _env_model or str(role_model("video_agent", default="dashscope,qwen2.5-vl-72b-instruct"))
    if "," not in _raw:
        _raw = f"dashscope,{_raw}"
    _settings_llm, _settings_model = _parse_llm_setting(_raw)
    resolved_llm = _settings_llm or "dashscope"

    if resolved_llm == "ollama":
        resolved_ollama = ollama_model or _settings_model or "qwen3-vl:30b"
    else:
        resolved_ollama = ollama_model or "qwen3-vl:30b"
    if resolved_llm == "claude":
        resolved_anthropic = anthropic_model or _settings_model or str(
            _cfg("ANTHROPIC_MODEL", "anthropic", "model", default="claude-haiku-4-5"))
    else:
        resolved_anthropic = anthropic_model or str(
            _cfg("ANTHROPIC_MODEL", "anthropic", "model", default="claude-haiku-4-5"))

    system_prompt = _load_system_prompt("video_agent")
    agent = _make_agent(
        role="video_agent",
        llm=resolved_llm,
        # For dashscope / openai / gemini backends this becomes the model id.
        dashscope_model=dashscope_model or _settings_model,
        system_prompt=system_prompt,
        tools=VIDEO_AGENT_TOOLS,
        ollama_model=resolved_ollama,
        anthropic_model=resolved_anthropic,
        **kwargs,
    )
    # Stateless: keep only the immediate exchange.
    agent.conversation_manager = SlidingWindowConversationManager(window_size=2)
    return agent


def create_qa_agent(
    ollama_model: str | None = None,
    anthropic_model: str | None = None,
    **kwargs,
) -> Agent:
    """Create the QA agent — stateless, single-shot judging of a finished output.

    Reads one produced image (or a video's sampled frames) plus the user's QA
    briefing and returns a structured per-criterion verdict. Stateless and
    tool-less by design, exactly like the Vision Agent: every output is judged on
    its own evidence, so one harsh call can't sour the next, and there is no tool
    loop to wander off in while the user waits.

    This is the role where a **stronger** model earns its cost, which is why it has
    its own setting rather than sharing ``executor_vision_model``: it runs once per
    finished output, and a weak judge is worse than none — it waves through defects
    or fails clean work and triggers a pointless re-render. Must be multimodal.

    Configuration (in priority order):
    1. ``QA_AGENT_MODEL`` env var (``'provider,model'`` or a bare model name)
    2. ``llm.pipeline.qa_checker`` in settings
    3. ``llm.pipeline.executor_vision_model`` — the shared vision model fallback

    Args:
        ollama_model:    Ollama model override.
        anthropic_model: Anthropic model override.
        **kwargs:        Forwarded to the Strands Agent constructor.
    """
    _env_model = os.environ.get("QA_AGENT_MODEL", "")
    _raw = _env_model or str(role_model("qa_checker", default=""))
    if not _raw:
        _raw = str(role_model("executor_vision_model", default="claude,claude-haiku-4-5"))
    if "," not in _raw:
        _raw = f"claude,{_raw}"
    _settings_llm, _settings_model = _parse_llm_setting(_raw)
    resolved_llm = _settings_llm or "claude"

    if resolved_llm == "ollama":
        resolved_ollama = ollama_model or _settings_model or "gemma4:26b"
        resolved_anthropic = anthropic_model or str(
            _cfg("ANTHROPIC_MODEL", "anthropic", "model", default="claude-haiku-4-5"))
    else:
        resolved_ollama = ollama_model or "gemma4:26b"
        resolved_anthropic = anthropic_model or _settings_model or str(
            _cfg("ANTHROPIC_MODEL", "anthropic", "model", default="claude-haiku-4-5"))

    # The QA prompt file is sectioned (## system / ## question / …) because the
    # question templates are data for the caller, not part of the agent's role.
    from src.utils.qa import load_qa_prompts
    system_prompt = load_qa_prompts().get("system", "") or _load_system_prompt("qa_checker")
    agent = _make_agent(
        role="qa_checker",
        llm=resolved_llm,
        dashscope_model=_settings_model,
        system_prompt=system_prompt,
        tools=VISION_AGENT_TOOLS,  # none — judging is a single look, not an investigation
        ollama_model=resolved_ollama,
        anthropic_model=resolved_anthropic,
        **kwargs,
    )
    agent.conversation_manager = SlidingWindowConversationManager(window_size=2)
    return agent


def create_query_templates_agent(
    llm: str | None = None,
    ollama_model: str | None = None,
    anthropic_model: str | None = None,
    **kwargs,
) -> Agent:
    """Create the Query Templates agent for experimental dual-agent pipeline.

    Defaults to Ollama (env: ``QUERYTEMPLATES_LLM``, then ``'ollama'``).
    Override the Ollama model with ``QUERYTEMPLATES_OLLAMA_MODEL`` or *ollama_model*.
    Override the Anthropic model with ``QUERYTEMPLATES_ANTHROPIC_MODEL`` or *anthropic_model*.

    Args:
        llm: ``'ollama'`` or ``'claude'``. Falls back to ``QUERYTEMPLATES_LLM`` env var.
        ollama_model: Ollama model override (e.g. ``'qwen3-coder:32b'``).
        anthropic_model: Anthropic model override (e.g. ``'claude-haiku-4-5'``).
        **kwargs: Forwarded to the Strands Agent constructor.
    """
    # Passing an Ollama model without an explicit LLM backend implies ollama.
    if ollama_model and llm is None:
        llm = "ollama"

    # Read combined 'provider,model' from settings (env var QUERYTEMPLATES_LLM still wins).
    _raw = str(role_model("query_templates", default="ollama", env_var="QUERYTEMPLATES_LLM"))
    _settings_llm, _settings_model = _parse_llm_setting(_raw)
    resolved_llm = llm or _settings_llm or "ollama"

    # Model: CLI arg > provider-specific env var > model extracted from settings > hard default.
    if resolved_llm == "ollama":
        resolved_ollama = (
            ollama_model
            or os.environ.get("QUERYTEMPLATES_OLLAMA_MODEL")
            or _settings_model
            or "qwen3-coder:32b"
        )
        resolved_anthropic = (
            anthropic_model
            or os.environ.get("QUERYTEMPLATES_ANTHROPIC_MODEL")
            or str(_cfg("ANTHROPIC_MODEL", "anthropic", "model", default="claude-haiku-4-5"))
        )
    else:  # claude
        resolved_anthropic = (
            anthropic_model
            or os.environ.get("QUERYTEMPLATES_ANTHROPIC_MODEL")
            or _settings_model
            or str(_cfg("ANTHROPIC_MODEL", "anthropic", "model", default="claude-haiku-4-5"))
        )
        resolved_ollama = ollama_model or "qwen3-coder:32b"

    system_prompt = _load_system_prompt("query_templates")

    # Load only the assembly-family skills (not the orchestrator/story skills).
    QUERYTEMPLATES_skill_plugins: list = []
    _asm_sources = _skill_sources(_ASSEMBLY_SKILL_NAMES)
    if _asm_sources:
        skills_plugin = AgentSkills(skills=_asm_sources)
        QUERYTEMPLATES_skill_plugins.append(skills_plugin)
        loaded = [s.name for s in skills_plugin.get_available_skills()]
        if loaded:
            print(f"[agentY:researcher] Loaded skills: {', '.join(loaded)}")

    # Merge steering handlers with skill plugins.
    QUERYTEMPLATES_plugins = QUERYTEMPLATES_skill_plugins + get_QUERYTEMPLATES_steering_handlers()

    return _make_agent(
        role="query_templates",
        llm=resolved_llm,
        dashscope_model=_settings_model,
        system_prompt=system_prompt,
        tools=QUERYTEMPLATES_TOOLS,
        ollama_model=resolved_ollama,
        anthropic_model=resolved_anthropic,
        plugins=QUERYTEMPLATES_plugins or None,
        **kwargs,
    )


def create_planner_agent(
    llm: str | None = None,
    ollama_model: str | None = None,
    anthropic_model: str | None = None,
    **kwargs,
) -> Agent:
    """Create the Planner agent — a stateless, tool-free multi-step decomposer.

    The Planner receives a complex multi-step user request and breaks it into
    a sequence of atomic generation tasks expressed as individual user requests.
    It outputs a JSON object ``{"steps": [{"request": "...", "description": "..."}]}``.

    Reads ``llm.pipeline.planner`` from settings.json (format: ``'provider,model'``).
    Env var ``PLANNER_LLM`` overrides the full setting; ``PLANNER_OLLAMA_MODEL``
    or ``PLANNER_ANTHROPIC_MODEL`` override just the model.

    Defaults to the same backend/model as the Detect User Intent agent.

    Args:
        llm: ``'ollama'`` or ``'claude'``. Falls back to ``PLANNER_LLM`` env var.
        ollama_model: Ollama model override (e.g. ``'qwen3:0.6b'``).
        anthropic_model: Anthropic model override (e.g. ``'claude-haiku-4-5'``).
        **kwargs: Forwarded to the Strands Agent constructor.
    """
    if ollama_model and llm is None:
        llm = "ollama"

    # Read combined 'provider,model' from settings (env var PLANNER_LLM still wins).
    _raw = str(role_model("planner", default="ollama", env_var="PLANNER_LLM"))
    _settings_llm, _settings_model = _parse_llm_setting(_raw)
    resolved_llm = llm or _settings_llm or "ollama"

    if resolved_llm == "ollama":
        resolved_ollama = (
            ollama_model
            or os.environ.get("PLANNER_OLLAMA_MODEL")
            or _settings_model
            or str(role_model("llm_functions", default="qwen3.5:9b", env_var="LLM_FUNCTIONS_MODEL"))
        )
        resolved_anthropic = (
            anthropic_model
            or os.environ.get("PLANNER_ANTHROPIC_MODEL")
            or str(_cfg("ANTHROPIC_MODEL", "anthropic", "model", default="claude-haiku-4-5"))
        )
    else:  # claude
        resolved_anthropic = (
            anthropic_model
            or os.environ.get("PLANNER_ANTHROPIC_MODEL")
            or _settings_model
            or str(_cfg("ANTHROPIC_MODEL", "anthropic", "model", default="claude-haiku-4-5"))
        )
        resolved_ollama = (
            ollama_model
            or str(role_model("llm_functions", default="qwen3.5:9b", env_var="LLM_FUNCTIONS_MODEL"))
        )

    system_prompt = _load_system_prompt("planner")
    agent = _make_agent(
        role="planner",
        llm=resolved_llm,
        dashscope_model=_settings_model,
        system_prompt=system_prompt,
        tools=PLANNER_TOOLS,
        ollama_model=resolved_ollama,
        anthropic_model=resolved_anthropic,
        **kwargs,
    )
    # Planner is single-turn and stateless.
    agent.conversation_manager = SlidingWindowConversationManager(window_size=2)
    return agent


def create_info_agent(
    llm: str | None = None,
    ollama_model: str | None = None,
    anthropic_model: str | None = None,
    **kwargs,
) -> Agent:
    """Create the Info agent — a lightweight agent that answers questions
    about available ComfyUI workflows, models, and capabilities.

    Reads ``llm.pipeline.info`` from settings.json (format: ``'provider,model'``),
    e.g. ``'ollama,qwen3.5:9b'`` or ``'claude,claude-haiku-4-5'``. Env var
    ``INFO_LLM`` overrides the combined setting; ``INFO_OLLAMA_MODEL`` or
    ``INFO_ANTHROPIC_MODEL`` override the provider-specific model.

    Args:
        llm: ``'ollama'`` or ``'claude'``. Falls back to ``INFO_LLM`` env/settings.
        ollama_model: Ollama model override (e.g. ``'qwen3.5:9b'``).
        anthropic_model: Anthropic model override (e.g. ``'claude-haiku-4-5'``).
        **kwargs: Forwarded to the Strands Agent constructor.
    """
    if ollama_model and llm is None:
        llm = "ollama"

    # Read combined 'provider,model' from settings (env var INFO_LLM still wins).
    _raw = role_model(
        "info", env_var="INFO_LLM",
        default=role_model("llm_functions", default="qwen3.5:9b",
                           env_var="LLM_FUNCTIONS_MODEL"),
    )
    _settings_llm, _settings_model = _parse_llm_setting(_raw)
    resolved_llm = llm or _settings_llm or "ollama"

    system_prompt = _load_system_prompt("info")

    if resolved_llm == "ollama":
        resolved_ollama = (
            ollama_model
            or os.environ.get("INFO_OLLAMA_MODEL")
            or _settings_model
            or str(role_model("llm_functions", default="qwen3.5:9b", env_var="LLM_FUNCTIONS_MODEL"))
        )
        return _make_agent(
            role="info",
            llm="ollama",
            system_prompt=system_prompt,
            tools=INFO_TOOLS,
            ollama_model=resolved_ollama,
            **kwargs,
        )

    # Otherwise use Anthropic/Claude
    resolved_anthropic = (
        anthropic_model
        or os.environ.get("INFO_ANTHROPIC_MODEL")
        or _settings_model
        or str(_cfg("ANTHROPIC_MODEL", "anthropic", "model", default="claude-haiku-4-5"))
    )
    return _make_agent(
        role="info",
        llm=resolved_llm,
        dashscope_model=_settings_model,
        system_prompt=system_prompt,
        tools=INFO_TOOLS,
        anthropic_model=resolved_anthropic,
        **kwargs,
    )


def create_SEARCHWEB_agent(
    llm: str | None = None,
    ollama_model: str | None = None,
    anthropic_model: str | None = None,
    **kwargs,
) -> Agent:
    """Create the Reference Search Web agent — a focused web-reference gatherer.

    Given a request, it searches the web, downloads the best reference image(s),
    decides per reference whether it is best used as a direct image input or a
    textual description, and returns a JSON manifest. Shares the same web/image
    tools as the Info agent but with a focused prompt and structured output so the
    Storyboard director can reliably consume the result.

    Reads ``llm.pipeline.search_web`` from settings.json (format ``'provider,model'``);
    falls back to the Info-agent setting, then ``claude-haiku-4-5``. Env var
    ``SEARCHWEB_LLM`` overrides the combined setting; ``SEARCHWEB_OLLAMA_MODEL`` /
    ``SEARCHWEB_ANTHROPIC_MODEL`` override the provider-specific model.

    Args:
        llm: ``'claude'`` or ``'ollama'``. Falls back to ``SEARCHWEB_LLM`` env/settings.
        ollama_model: Ollama model override.
        anthropic_model: Anthropic model override.
        **kwargs: Forwarded to the Strands Agent constructor.
    """
    if ollama_model and llm is None:
        llm = "ollama"

    # Fall back to the Info-agent setting so no extra config is required.
    _info_default = str(role_model("info", default="claude,claude-haiku-4-5", env_var="INFO_LLM"))
    _raw = str(role_model("search_web", default=_info_default, env_var="SEARCHWEB_LLM"))
    _settings_llm, _settings_model = _parse_llm_setting(_raw)
    resolved_llm = llm or _settings_llm or "claude"

    system_prompt = _load_system_prompt("search_web")

    if resolved_llm == "ollama":
        resolved_ollama = (
            ollama_model
            or os.environ.get("SEARCHWEB_OLLAMA_MODEL")
            or _settings_model
            or str(role_model("llm_functions", default="qwen3.5:9b", env_var="LLM_FUNCTIONS_MODEL"))
        )
        agent = _make_agent(
            role="search_web",
            llm="ollama",
            system_prompt=system_prompt,
            tools=SEARCHWEB_TOOLS,
            ollama_model=resolved_ollama,
            **kwargs,
        )
    else:
        resolved_anthropic = (
            anthropic_model
            or os.environ.get("SEARCHWEB_ANTHROPIC_MODEL")
            or _settings_model
            or str(_cfg("ANTHROPIC_MODEL", "anthropic", "model", default="claude-haiku-4-5"))
        )
        agent = _make_agent(
            role="search_web",
            llm=resolved_llm,
            dashscope_model=_settings_model,
            system_prompt=system_prompt,
            tools=SEARCHWEB_TOOLS,
            anthropic_model=resolved_anthropic,
            **kwargs,
        )
    # Single-turn, stateless: each scouting request is independent.
    agent.conversation_manager = SlidingWindowConversationManager(window_size=6)
    return agent


def create_ASSEMBLEWORKFLOW_agent(
    llm: str | None = None,
    ollama_model: str | None = None,
    anthropic_model: str | None = None,
    **kwargs,
) -> Agent:
    """Create the Assemble Workflow agent for experimental dual-agent pipeline.

    Defaults to Claude (env: ``ASSEMBLEWORKFLOW_LLM``, then ``'claude'``).
    Override the Anthropic model with ``ASSEMBLEWORKFLOW_ANTHROPIC_MODEL`` or *anthropic_model*.
    Override the Ollama model with ``ASSEMBLEWORKFLOW_OLLAMA_MODEL`` or *ollama_model*.

    Args:
        llm: ``'claude'`` or ``'ollama'``. Falls back to ``ASSEMBLEWORKFLOW_LLM`` env var.
        anthropic_model: Anthropic model override (e.g. ``'claude-sonnet-4-5'``).
        ollama_model: Ollama model override.
        **kwargs: Forwarded to the Strands Agent constructor.
    """
    # Passing an Ollama model without an explicit LLM backend implies ollama.
    if ollama_model and llm is None:
        llm = "ollama"
    # Reset the patch_workflow failure counter for each new brain session.
    reset_patch_workflow_guard()

    # Read combined 'provider,model' from settings (env var ASSEMBLEWORKFLOW_LLM still wins).
    _raw = str(role_model("assemble_workflow", default="claude", env_var="ASSEMBLEWORKFLOW_LLM"))
    _settings_llm, _settings_model = _parse_llm_setting(_raw)
    resolved_llm = llm or _settings_llm or "claude"

    # Model: CLI arg > provider-specific env var > model extracted from settings > hard default.
    if resolved_llm == "claude":
        resolved_anthropic = (
            anthropic_model
            or os.environ.get("ASSEMBLEWORKFLOW_ANTHROPIC_MODEL")
            or _settings_model
            or str(_cfg("ANTHROPIC_MODEL", "anthropic", "model", default="claude-haiku-4-5"))
        )
        resolved_ollama = ollama_model or "qwen3-vl:30b"
    else:  # ollama
        resolved_ollama = (
            ollama_model
            or os.environ.get("ASSEMBLEWORKFLOW_OLLAMA_MODEL")
            or _settings_model
            or "qwen3-vl:30b"
        )
        resolved_anthropic = (
            anthropic_model
            or os.environ.get("ASSEMBLEWORKFLOW_ANTHROPIC_MODEL")
            or str(_cfg("ANTHROPIC_MODEL", "anthropic", "model", default="claude-haiku-4-5"))
        )
    system_prompt = _load_system_prompt("assemble_workflow")

    # Load only the assembly-family skills (not the orchestrator/story skills).
    skills_plugins: list = []
    _asm_sources = _skill_sources(_ASSEMBLY_SKILL_NAMES)
    if _asm_sources:
        skills_plugin = AgentSkills(skills=_asm_sources)
        skills_plugins.append(skills_plugin)
        loaded = [s.name for s in skills_plugin.get_available_skills()]
        if loaded:
            print(f"[agentY:brain] Loaded skills: {', '.join(loaded)}")

    # Merge skills plugins with steering handlers.
    ASSEMBLEWORKFLOW_plugins = skills_plugins + get_ASSEMBLEWORKFLOW_steering_handlers()

    # Merge the ComfyUI interrupt hook with any caller-supplied hooks so we
    # don't silently drop the TokenUsageHookProvider built by _make_agent.
    # We pass the combined list via kwargs; _make_agent's agent_kwargs.update()
    # will replace its default [TokenUsageHookProvider] with our explicit list.
    extra_hooks = kwargs.pop("hooks", [])
    ASSEMBLEWORKFLOW_hooks = [TokenUsageHookProvider(role="brain"), ComfyUIInterruptHook(), *extra_hooks]

    return _make_agent(
        role="brain",
        llm=resolved_llm,
        dashscope_model=_settings_model,
        system_prompt=system_prompt,
        tools=ASSEMBLEWORKFLOW_TOOLS,
        ollama_model=resolved_ollama,
        anthropic_model=resolved_anthropic,
        plugins=ASSEMBLEWORKFLOW_plugins or None,
        hooks=ASSEMBLEWORKFLOW_hooks,
        **kwargs,
    )


# Rename-compat aliases: pipeline.py, triage.py, and utils/agentY_server.py import
# the snake_case factory names; these three factories were left in the older
# ALLCAPS spelling. Alias so both spellings refer to the same function.
create_assemble_workflow_agent = create_ASSEMBLEWORKFLOW_agent
create_search_web_agent = create_SEARCHWEB_agent


def create_learnings_agent(
    llm: str | None = None,
    ollama_model: str | None = None,
    anthropic_model: str | None = None,
    **kwargs,
) -> Agent:
    """Create the Learnings agent — a stateless pattern-analyser.

    The Learnings agent receives a Assemble Workflow session transcript and extracts
    concise actionable learnings from repeated failure→fix patterns.
    It is typically invoked asynchronously after tasks where the Assemble Workflow used
    more than 5 tool calls.

    Reads ``llm.pipeline.learnings`` from settings.json (format: ``'provider,model'``).
    Env var ``LEARNINGS_LLM`` overrides the full setting.
    Defaults to ``'ollama,qwen3.5:9b'``.

    Args:
        llm: ``'ollama'`` or ``'claude'``. Falls back to ``LEARNINGS_LLM`` env var.
        ollama_model: Ollama model override.
        anthropic_model: Anthropic model override.
        **kwargs: Forwarded to the Strands Agent constructor.
    """
    if ollama_model and llm is None:
        llm = "ollama"

    _raw = str(role_model("learnings", default="ollama,qwen3.5:9b", env_var="LEARNINGS_LLM"))
    _settings_llm, _settings_model = _parse_llm_setting(_raw)
    resolved_llm = llm or _settings_llm or "ollama"

    if resolved_llm == "ollama":
        resolved_ollama = (
            ollama_model
            or os.environ.get("LEARNINGS_OLLAMA_MODEL")
            or _settings_model
            or "qwen3.5:9b"
        )
        resolved_anthropic = (
            anthropic_model
            or os.environ.get("LEARNINGS_ANTHROPIC_MODEL")
            or str(_cfg("ANTHROPIC_MODEL", "anthropic", "model", default="claude-haiku-4-5"))
        )
    else:  # claude
        resolved_anthropic = (
            anthropic_model
            or os.environ.get("LEARNINGS_ANTHROPIC_MODEL")
            or _settings_model
            or str(_cfg("ANTHROPIC_MODEL", "anthropic", "model", default="claude-haiku-4-5"))
        )
        resolved_ollama = ollama_model or "qwen3.5:9b"

    system_prompt = _load_system_prompt("learnings")
    agent = _make_agent(
        role="learnings",
        llm=resolved_llm,
        dashscope_model=_settings_model,
        system_prompt=system_prompt,
        tools=LEARNINGS_TOOLS,
        ollama_model=resolved_ollama,
        anthropic_model=resolved_anthropic,
        **kwargs,
    )
    # Learnings agent is single-turn and stateless.
    agent.conversation_manager = SlidingWindowConversationManager(window_size=2)
    return agent


def create_fix_workflow_assembly_agent(
    llm: str | None = None,
    ollama_model: str | None = None,
    anthropic_model: str | None = None,
    **kwargs,
) -> Agent:
    """Create the Fix Workflow Assembly agent — the consolidated workflow-repair
    specialist.

    Invoked on demand (never on the happy path) for two triggers:
      * assembly-time: ``apply_brainbriefing`` returned ``status:error`` with
        concrete ``problems``;
      * execution-time: ComfyUI failed to run the workflow (bad node/model).
    It diagnoses the failing node, patches the graph with a minimal change, and
    re-validates. It does not select templates or write prompts.

    Reads ``llm.pipeline.fix_workflow_assembly`` from settings.json
    (``'provider,model'``); falls back to the assemble_workflow (Brain) setting.
    Env var ``FIXWORKFLOWASSEMBLY_LLM`` overrides the full setting.
    """
    if ollama_model and llm is None:
        llm = "ollama"

    _ASSEMBLEWORKFLOW_default = str(role_model("assemble_workflow", default="claude,claude-haiku-4-5", env_var="ASSEMBLEWORKFLOW_LLM"))
    _raw = str(role_model("fix_workflow_assembly", default=_ASSEMBLEWORKFLOW_default, env_var="FIXWORKFLOWASSEMBLY_LLM"))
    _settings_llm, _settings_model = _parse_llm_setting(_raw)
    resolved_llm = llm or _settings_llm or "claude"

    if resolved_llm == "claude":
        resolved_anthropic = (
            anthropic_model
            or os.environ.get("FIXWORKFLOWASSEMBLY_ANTHROPIC_MODEL")
            or _settings_model
            or str(_cfg("ANTHROPIC_MODEL", "anthropic", "model", default="claude-haiku-4-5"))
        )
        resolved_ollama = ollama_model or "qwen3-coder:32b"
    else:  # ollama
        resolved_ollama = (
            ollama_model
            or os.environ.get("FIXWORKFLOWASSEMBLY_OLLAMA_MODEL")
            or _settings_model
            or "qwen3-coder:32b"
        )
        resolved_anthropic = (
            anthropic_model
            or os.environ.get("FIXWORKFLOWASSEMBLY_ANTHROPIC_MODEL")
            or str(_cfg("ANTHROPIC_MODEL", "anthropic", "model", default="claude-haiku-4-5"))
        )

    system_prompt = _load_system_prompt("fix_workflow_assembly")

    fx_plugins: list = []
    _asm_sources = _skill_sources(_ASSEMBLY_SKILL_NAMES)
    if _asm_sources:
        fx_plugins.append(AgentSkills(skills=_asm_sources))

    agent = _make_agent(
        role="fix_workflow_assembly",
        llm=resolved_llm,
        dashscope_model=_settings_model,
        system_prompt=system_prompt,
        tools=FIX_WORKFLOW_ASSEMBLY_TOOLS,
        ollama_model=resolved_ollama,
        anthropic_model=resolved_anthropic,
        plugins=fx_plugins or None,
        **kwargs,
    )
    # Short-lived repair turns — a small window keeps context lean.
    agent.conversation_manager = SlidingWindowConversationManager(window_size=4)
    return agent


def create_generate_new_workflow_agent(
    llm: str | None = None,
    ollama_model: str | None = None,
    anthropic_model: str | None = None,
    **kwargs,
) -> Agent:
    """Create the Generate New Workflow agent — builds a ComfyUI workflow from
    scratch when no template fits (``template.name == "build_new"``).

    Rare, on-demand: the researcher prefers templates, so this fires only when
    nothing matches. It follows the ``assemble-new-workflow`` skill — fetch the
    recipe, load the closest member template as a scaffold, and conform it to the
    recipe (nodes, wiring, boundary ports). It does not select a template for a
    normal request and does not submit for execution.

    Reads ``llm.pipeline.generate_new_workflow`` from settings.json; falls back to
    the assemble_workflow (Brain) setting. Env var ``GENERATENEWWORKFLOW_LLM``
    overrides the full setting.
    """
    if ollama_model and llm is None:
        llm = "ollama"

    _ASSEMBLEWORKFLOW_default = str(role_model("assemble_workflow", default="claude,claude-haiku-4-5", env_var="ASSEMBLEWORKFLOW_LLM"))
    _raw = str(role_model("generate_new_workflow", default=_ASSEMBLEWORKFLOW_default, env_var="GENERATENEWWORKFLOW_LLM"))
    _settings_llm, _settings_model = _parse_llm_setting(_raw)
    resolved_llm = llm or _settings_llm or "claude"

    if resolved_llm == "claude":
        resolved_anthropic = (
            anthropic_model
            or os.environ.get("GENERATENEWWORKFLOW_ANTHROPIC_MODEL")
            or _settings_model
            or str(_cfg("ANTHROPIC_MODEL", "anthropic", "model", default="claude-haiku-4-5"))
        )
        resolved_ollama = ollama_model or "qwen3-coder:32b"
    else:  # ollama
        resolved_ollama = (
            ollama_model
            or os.environ.get("GENERATENEWWORKFLOW_OLLAMA_MODEL")
            or _settings_model
            or "qwen3-coder:32b"
        )
        resolved_anthropic = (
            anthropic_model
            or os.environ.get("GENERATENEWWORKFLOW_ANTHROPIC_MODEL")
            or str(_cfg("ANTHROPIC_MODEL", "anthropic", "model", default="claude-haiku-4-5"))
        )

    system_prompt = _load_system_prompt("generate_new_workflow")

    gn_plugins: list = []
    _asm_sources = _skill_sources(_ASSEMBLY_SKILL_NAMES)
    if _asm_sources:
        gn_plugins.append(AgentSkills(skills=_asm_sources))

    agent = _make_agent(
        role="generate_new_workflow",
        llm=resolved_llm,
        dashscope_model=_settings_model,
        system_prompt=system_prompt,
        tools=GENERATE_NEW_WORKFLOW_TOOLS,
        ollama_model=resolved_ollama,
        anthropic_model=resolved_anthropic,
        plugins=gn_plugins or None,
        **kwargs,
    )
    return agent


def create_coder_agent(
    skill: str | None = None,
    llm: str | None = None,
    ollama_model: str | None = None,
    anthropic_model: str | None = None,
    **kwargs,
) -> Agent:
    """Create the general **coder** agent.

    A focused code-authoring agent: given a self-contained coding task it reads the
    relevant source/docs and writes or edits complete, importable code, then returns
    a concise summary. The general contract (read before you write, honest TODO
    stubs, never invent an API) lives in ``system_prompt.coder``; the domain
    knowledge for a *particular* task is supplied as a **skill** whose ``SKILL.md``
    body is baked into the prompt as the agent's procedure. For example
    ``create_coder_agent(skill="custom-node-from-github")`` turns a cloned model repo
    (staged by the ``create_custom_node`` tool) into a ComfyUI custom-node pack.

    Writing correct code is a demanding generation task, so this role reads
    ``llm.pipeline.coder`` from settings.json and, when unset, falls back to the
    **assemble_workflow** (builder) setting — guaranteed to resolve on any working
    install. Point ``pipeline.coder`` at a strong coding model (a Qwen/Kimi coder or a
    Claude Sonnet/Opus). Env var ``CODER_LLM`` overrides the full setting;
    ``CODER_OLLAMA_MODEL`` / ``CODER_ANTHROPIC_MODEL`` override the model.

    Args:
        skill: Optional skill name whose SKILL.md body is baked into the prompt as
            the agent's procedure (e.g. ``"custom-node-from-github"``).
        llm: ``'claude'`` | ``'ollama'`` | a DashScope provider. Falls back to settings.
        ollama_model: Ollama model override.
        anthropic_model: Anthropic model override.
        **kwargs: Forwarded to the Strands Agent constructor.
    """
    if ollama_model and llm is None:
        llm = "ollama"

    # Resolve the coding model: coder → builder (assemble_workflow) default, which is
    # always configured on a working install. Point pipeline.coder at a strong model.
    _builder_default = str(role_model("assemble_workflow", default="claude,claude-haiku-4-5", env_var="ASSEMBLEWORKFLOW_LLM"))
    _raw = str(role_model("coder", default=_builder_default, env_var="CODER_LLM"))
    _settings_llm, _settings_model = _parse_llm_setting(_raw)
    resolved_llm = llm or _settings_llm or "claude"

    if resolved_llm == "ollama":
        resolved_ollama = (
            ollama_model
            or os.environ.get("CODER_OLLAMA_MODEL")
            or _settings_model
            or "qwen3-coder:30b"
        )
        resolved_anthropic = (
            anthropic_model
            or os.environ.get("CODER_ANTHROPIC_MODEL")
            or str(_cfg("ANTHROPIC_MODEL", "anthropic", "model", default="claude-haiku-4-5"))
        )
    else:  # claude / dashscope
        resolved_anthropic = (
            anthropic_model
            or os.environ.get("CODER_ANTHROPIC_MODEL")
            or _settings_model
            or str(_cfg("ANTHROPIC_MODEL", "anthropic", "model", default="claude-haiku-4-5"))
        )
        resolved_ollama = ollama_model or "qwen3-coder:30b"

    system_prompt = _load_system_prompt("coder")
    skill_body = _load_subagent_skill(skill) if skill else ""
    if skill_body:
        system_prompt = (system_prompt.rstrip()
                         + "\n\n## Your procedure — follow this exactly\n\n"
                         + skill_body)
    agent = _make_agent(
        role="coder",
        llm=resolved_llm,
        dashscope_model=_settings_model,
        system_prompt=system_prompt,
        tools=CODER_TOOLS,
        ollama_model=resolved_ollama,
        anthropic_model=resolved_anthropic,
        **kwargs,
    )
    return agent


# ---------------------------------------------------------------------------
# Orchestrator — the free-agent entry point (replaces triage + rigid routing)
# ---------------------------------------------------------------------------

def create_orchestrator_agent(
    llm: str | None = None,
    ollama_model: str | None = None,
    anthropic_model: str | None = None,
    extra_tools: list | None = None,
    **kwargs,
) -> Agent:
    """Create the Orchestrator agent — a single, free agent that owns the turn.

    Instead of a triage classifier fanning requests out to fixed handlers, the
    orchestrator holds the full toolset directly, can delegate to the specialist
    agents (passed in as ``extra_tools`` by the pipeline), can spawn ad-hoc
    subagents, and can author skills at runtime. It decides for itself how to
    fulfil the user's intent.

    Reads ``llm.pipeline.orchestrator`` from settings.json (format
    ``'provider,model'``); defaults to ``claude,claude-haiku-4-5``. Env var
    ``ORCHESTRATOR_LLM`` overrides the combined setting;
    ``ORCHESTRATOR_OLLAMA_MODEL`` / ``ORCHESTRATOR_ANTHROPIC_MODEL`` override the
    provider-specific model.

    Args:
        llm: ``'claude'`` | ``'ollama'`` | a DashScope provider. Falls back to settings.
        ollama_model: Ollama model override.
        anthropic_model: Anthropic model override.
        extra_tools: Extra @tool callables to append (the pipeline's delegation tools).
        **kwargs: Forwarded to the Strands Agent constructor.

    The built agent carries ``agent._agentskills_plugin`` so the pipeline can wire
    ``create_skill`` to re-scan the live plugin.
    """
    if ollama_model and llm is None:
        llm = "ollama"
    # The orchestrator assembles workflows too, so reset the per-session guard.
    reset_patch_workflow_guard()

    _raw = str(role_model("orchestrator", default="claude,claude-haiku-4-5", env_var="ORCHESTRATOR_LLM"))
    _settings_llm, _settings_model = _parse_llm_setting(_raw)
    resolved_llm = llm or _settings_llm or "claude"

    if resolved_llm == "ollama":
        resolved_ollama = (
            ollama_model
            or os.environ.get("ORCHESTRATOR_OLLAMA_MODEL")
            or _settings_model
            or "qwen3-vl:30b"
        )
        resolved_anthropic = (
            anthropic_model
            or os.environ.get("ORCHESTRATOR_ANTHROPIC_MODEL")
            or str(_cfg("ANTHROPIC_MODEL", "anthropic", "model", default="claude-haiku-4-5"))
        )
    else:  # claude / dashscope
        resolved_anthropic = (
            anthropic_model
            or os.environ.get("ORCHESTRATOR_ANTHROPIC_MODEL")
            or _settings_model
            or str(_cfg("ANTHROPIC_MODEL", "anthropic", "model", default="claude-haiku-4-5"))
        )
        resolved_ollama = ollama_model or "qwen3-vl:30b"

    system_prompt = _load_system_prompt("orchestrator")

    # Load ONLY the orchestrator-owned skills (router/writing/self-extension) + the
    # runtime-authored scratch skills — never the assembly skills, which the router
    # must delegate. Missing dirs are safely skipped by the loader. Keep a reference
    # to the plugin (and stash the exact sources) so a live create_skill re-scan
    # preserves this scoping instead of re-widening to the whole skills/ dir.
    _orch_sources = _orchestrator_skill_sources()
    skills_plugin = AgentSkills(skills=_orch_sources)
    skills_plugin._agenty_sources = _orch_sources  # noqa: SLF001 — read back by _rescan_skills
    loaded = [s.name for s in skills_plugin.get_available_skills()]
    if loaded:
        print(f"[agentY:orchestrator] Loaded skills: {', '.join(loaded)}")

    tools = list(ORCHESTRATOR_TOOLS) + list(extra_tools or [])

    # Tools from configured MCP servers (config/mcp.json). Fully contained: any
    # failure (unreachable/unauthorized server, missing deps) is swallowed so the
    # orchestrator always builds. OAuth servers with no stored token are skipped
    # here (no browser at startup) — authorize once via agentY Settings.
    try:
        from src.tools.mcp_tools import load_mcp_tools as _load_mcp_tools
        _mcp_tools = _load_mcp_tools()
        if _mcp_tools:
            tools += _mcp_tools
            print(f"[agentY:orchestrator] Loaded {len(_mcp_tools)} MCP tool(s).")
    except Exception as _mcp_exc:  # noqa: BLE001
        print(f"[agentY:orchestrator] MCP tools skipped: {_mcp_exc}")

    extra_hooks = kwargs.pop("hooks", [])
    orch_hooks = [TokenUsageHookProvider(role="orchestrator"), ToolActivityHookProvider(role="orchestrator"),
                  MagnificWatchHookProvider(), ComfyUIInterruptHook(), *extra_hooks]

    agent = _make_agent(
        role="orchestrator",
        llm=resolved_llm,
        dashscope_model=_settings_model,
        system_prompt=system_prompt,
        tools=tools,
        ollama_model=resolved_ollama,
        anthropic_model=resolved_anthropic,
        plugins=[skills_plugin],
        hooks=orch_hooks,
        **kwargs,
    )
    agent._agentskills_plugin = skills_plugin
    return agent


# Meta-tool identities to exclude from a subagent's toolset (keeps subagents
# depth-1: they cannot spawn further subagents or author skills).
def _subagent_full_tools() -> list:
    from src.tools import (  # local import avoids import-time cycles
        ORCHESTRATOR_TOOLS as _OT,
        create_skill as _cs,
        list_skills as _ls,
        remove_skill as _rs,
        spawn_subagent as _sp,
        create_custom_node as _cn,
        list_generated_nodes as _lgn,
    )
    _meta = {id(_cs), id(_ls), id(_rs), id(_sp), id(_cn), id(_lgn)}
    return [t for t in _OT if id(t) not in _meta]


def _load_subagent_skill(name: str) -> str:
    """Return a skill's body (frontmatter stripped) to bake into a subagent's
    system prompt as its main procedure. Looks in the curated skills/ dir first,
    then the runtime _scratch dir. Returns '' if not found."""
    for base in (_SKILLS_DIR, _ORCH_SKILLS_DIR, _SCRATCH_SKILLS_DIR):
        try:
            p = base / name / "SKILL.md"
            if p.is_file():
                txt = p.read_text(encoding="utf-8")
                if txt.startswith("---"):
                    parts = txt.split("---", 2)
                    if len(parts) == 3:
                        txt = parts[2]
                return txt.strip()
        except Exception:  # noqa: BLE001
            continue
    return ""


def build_subagent(toolset: str = "full", model: str | None = None,
                   tools: list[str] | None = None, skill: str | None = None) -> Agent:
    """Build a fresh, single-use subagent with a curated toolset.

    Used by the ``spawn_subagent`` tool. Subagents are depth-1: the ``full``
    toolset excludes the self-extension meta-tools so a subagent cannot spawn
    further subagents.

    Args:
        toolset: research|assembly|info|web|vision|full (ignored when
            ``tools`` is given).
        model: Optional ``'provider,model'`` override.
        tools: Optional explicit list of tool NAMES — builds a lean, single-purpose
            agent with ONLY those tools (fewer tool defs = less context + better
            tool-selection for small models). Takes priority over ``toolset``.
        skill: Optional skill name whose body is baked into the subagent's system
            prompt as its procedure (its "main skill").

    Returns:
        A ready-to-invoke Strands Agent.
    """
    ts = (toolset or "full").strip().lower()
    prov: str | None = None
    mdl: str | None = None
    if model:
        _p, _, _m = model.partition(",")
        prov = (_p.strip().lower() or None)
        mdl = (_m.strip() or None)

    def _mk(anthropic=None, ollama=None):
        return {
            "llm": prov,
            "anthropic_model": anthropic or (mdl if prov in (None, "claude") else None),
            "ollama_model": ollama or (mdl if prov == "ollama" else None),
        }

    # Explicit minimal toolset (highest priority): a lean, single-purpose agent
    # with ONLY the named tools, optionally with a skill baked in as its procedure.
    if tools:
        avail = {getattr(t, "tool_name", getattr(t, "__name__", "")): t
                 for t in _subagent_full_tools()}
        chosen = [avail[n] for n in tools if n in avail]
        if not chosen:
            raise ValueError(f"none of the requested subagent tools exist: {tools}")
        resolved_llm = prov or str(role_model("orchestrator", default="claude", env_var="ORCHESTRATOR_LLM")).partition(",")[0].strip() or "claude"
        sp = (
            "You are a focused subagent handed a single, self-contained task by an "
            "orchestrator. Use ONLY the tools you were given to complete it, then "
            "return a concise result. For ComfyUI generation, assemble and validate "
            "the workflow and call signal_workflow_ready(workflow_path) as your final "
            "step — never submit_prompt. Do not ask clarifying questions; make "
            "reasonable assumptions."
        )
        skill_body = _load_subagent_skill(skill) if skill else ""
        if skill_body:
            sp += "\n\n## Your procedure — follow this exactly\n\n" + skill_body
        return _make_agent(
            role="subagent",
            llm=resolved_llm,
            dashscope_model=mdl or "",
            system_prompt=sp,
            tools=chosen,
            ollama_model=(mdl if resolved_llm == "ollama" else None) or "qwen3-vl:30b",
            anthropic_model=(mdl if resolved_llm not in ("ollama",) else None)
            or str(_cfg("ANTHROPIC_MODEL", "anthropic", "model", default="claude-haiku-4-5")),
            hooks=[TokenUsageHookProvider(role="subagent"), ComfyUIInterruptHook()],
        )

    if ts == "research":
        return create_query_templates_agent(**_mk())
    if ts == "assembly":
        return create_assemble_workflow_agent(**_mk())
    if ts == "info":
        return create_info_agent(**_mk())
    if ts == "web":
        return create_search_web_agent(**_mk())
    if ts == "vision":
        return create_vision_agent(anthropic_model=mdl if prov in (None, "claude") else None,
                                   ollama_model=mdl if prov == "ollama" else None)

    # "full": a general agent with the whole non-meta toolset.
    resolved_llm = prov or str(role_model("orchestrator", default="claude", env_var="ORCHESTRATOR_LLM")).partition(",")[0].strip() or "claude"
    system_prompt = (
        "You are a focused subagent working on a single, self-contained task handed "
        "to you by an orchestrator. Use your tools to complete it, then return a "
        "concise result. For ComfyUI generation, assemble and validate the workflow "
        "and call signal_workflow_ready(workflow_path) as your final step — never "
        "submit_prompt. Do not ask clarifying questions; make reasonable assumptions."
    )
    return _make_agent(
        role="subagent",
        llm=resolved_llm,
        dashscope_model=mdl or "",
        system_prompt=system_prompt,
        tools=_subagent_full_tools(),
        ollama_model=(mdl if resolved_llm == "ollama" else None) or "qwen3-vl:30b",
        anthropic_model=(mdl if resolved_llm not in ("ollama",) else None)
        or str(_cfg("ANTHROPIC_MODEL", "anthropic", "model", default="claude-haiku-4-5")),
        hooks=[TokenUsageHookProvider(role="subagent"), ComfyUIInterruptHook()],
    )

