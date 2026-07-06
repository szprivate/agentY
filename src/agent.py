"""
agentY – A ComfyUI agent built on the //eStrands Agents SDK.

Two-agent pipeline:
  • Query Templates  – Ollama (default) or any LLM; pattern-matching/resolution only.
                  Produces a brainbriefing JSON.
  • Assemble Workflow       – Claude (default) or any LLM; workflow assembly, execution, QA.
"""

import datetime
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

# Removed `handoff_to_user` tool registration — not used by agents anymore.

from src.utils.comfyui_interrupt_hook import ComfyUIInterruptHook
from src.utils.costs import compute_cost_from_usage

from src.tools import (
    QUERYTEMPLATES_TOOLS,
    ASSEMBLEWORKFLOW_TOOLS,
    INFO_TOOLS,
    STORY_TOOLS,
    SEARCHWEB_TOOLS,
    ERROR_CHECKER_TOOLS,
    PLANNER_TOOLS,
    DETECTUSERINTENT_TOOLS,
    LEARNINGS_TOOLS,
    VISION_AGENT_TOOLS,
    DOP_TOOLS,
    reset_patch_workflow_guard,
)
from src.steering import get_ASSEMBLEWORKFLOW_steering_handlers, get_QUERYTEMPLATES_steering_handlers


# ---------------------------------------------------------------------------
# Settings loader – reads config/settings.json once; env vars always win.
# ---------------------------------------------------------------------------

def _load_settings() -> dict:
    """Return the parsed settings.json, or {} if the file is absent/invalid."""
    path = Path(__file__).parent.parent / "config" / "settings.json"
    if path.exists():
        try:
            return json.loads("".join(ln for ln in path.read_text(encoding="utf-8").splitlines(keepends=True) if not ln.lstrip().startswith("//")))
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
    "query_templates.local": "system_prompt.query_templates.local",
    "assemble_workflow": "system_prompt.assemble_workflow",
    "assemble_workflow.local": "system_prompt.assemble_workflow.local",
    "detect_user_intent": "system_prompt.detect_user_intent",
    "planner": "system_prompt.planner",
    "info": "system_prompt.info",
    "story": "system_prompt.story",
    "search_web": "system_prompt.search_web",
    "dop": "system_prompt.dop",
    "learnings": "system_prompt.learnings",
    "error_checker": "system_prompt.error_checker",
    "qa_checker": "system_prompt.qaChecker",
    "vision_agent": "system_prompt.vision_agent",
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

                log_entry = (
                    f"{ts} [{self._role}] tool={tool_display} "
                    f"delta=+{d_in}in/+{d_out}out/+{d_cr}cache_read/+{d_cw}cache_write"
                    f"  total={in_tok}in/{out_tok}out/{cache_read}cache_read/{cache_write}cache_write"
                    f"{cost_str}\n"
                )
                with self._log_path.open("a", encoding="utf-8") as f:
                    f.write(log_entry)
            except Exception:
                pass  # Never break the agent loop for file I/O errors
        except Exception:
            pass  # Never break the agent loop for cosmetic output


# ---------------------------------------------------------------------------
# Skills directory – lives at <project_root>/skills/
# ---------------------------------------------------------------------------
_SKILLS_DIR = Path(__file__).parent.parent / "skills"

# Story-agent skills live in a separate directory so the ComfyUI agents
# (Brain / Researcher / Error-checker), which scan the whole _SKILLS_DIR, never
# see the story modes — and the Story agent never sees the ComfyUI skills.
_STORY_SKILLS_DIR = Path(__file__).parent.parent / "skills_story"


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
        host = str(_cfg("OLLAMA_HOST", "ollama", "host", default="http://localhost:11434"))
        # Ollama defaults num_ctx to ~4k, which truncates the large agent prompts
        # (the .local query_templates/assemble_workflow carry the full model table) and yields
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
        model = OpenAIModel(
            client_args={"api_key": api_key, "base_url": base_url},
            model_id=model_id,
            params={"max_tokens": ds_max_tokens},
        )
        print(f"[agentY:{role}] Using Alibaba Model Studio (DashScope) — {model_id}")
    else:
        model_id = anthropic_model or str(_cfg("ANTHROPIC_MODEL", "anthropic", "model", default="claude-haiku-4-5"))
        tokens = max_tokens or int(_cfg("ANTHROPIC_MAX_TOKENS", "anthropic", "max_tokens", default=4096))
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
    _raw = str(_cfg("", "pipeline", "vision_agent", default=""))
    if not _raw:
        _raw = str(_cfg("", "pipeline", "executor_vision_model", default="ollama,gemma4:26b"))
        if "," not in _raw:
            _raw = f"ollama,{_raw}"
    _settings_llm, _settings_model = _parse_llm_setting(_raw)
    resolved_llm = _settings_llm or "ollama"

    if resolved_llm == "ollama":
        resolved_ollama = (
            ollama_model
            or _env_model
            or _settings_model
            or str(_cfg("", "pipeline", "executor_vision_model", default="gemma4:26b"))
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
    _raw = str(_cfg("QUERYTEMPLATES_LLM", "pipeline", "query_templates", default="ollama"))
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

    system_prompt = _load_system_prompt("query_templates.local" if resolved_llm == "ollama" else "query_templates")

    # Load skills from the project-level skills/ directory.
    QUERYTEMPLATES_skill_plugins: list = []
    if _SKILLS_DIR.is_dir():
        skills_plugin = AgentSkills(skills=str(_SKILLS_DIR))
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
    # Falls back to the detect_user_intent setting so no extra config is required.
    _raw = str(_cfg("PLANNER_LLM", "pipeline", "planner",
                    default=str(_cfg("DETECTUSERINTENT_LLM", "pipeline", "detect_user_intent", default="ollama"))))
    _settings_llm, _settings_model = _parse_llm_setting(_raw)
    resolved_llm = llm or _settings_llm or "ollama"

    if resolved_llm == "ollama":
        resolved_ollama = (
            ollama_model
            or os.environ.get("PLANNER_OLLAMA_MODEL")
            or _settings_model
            or str(_cfg("LLM_FUNCTIONS_MODEL", "pipeline", "llm_functions", default="qwen3.5:9b"))
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
            or str(_cfg("LLM_FUNCTIONS_MODEL", "pipeline", "llm_functions", default="qwen3.5:9b"))
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
    _raw = str(
        _cfg(
            "INFO_LLM",
            "pipeline",
            "info",
            default=str(_cfg("LLM_FUNCTIONS_MODEL", "pipeline", "llm_functions", default="qwen3.5:9b")),
        )
    )
    _settings_llm, _settings_model = _parse_llm_setting(_raw)
    resolved_llm = llm or _settings_llm or "ollama"

    system_prompt = _load_system_prompt("info")

    if resolved_llm == "ollama":
        resolved_ollama = (
            ollama_model
            or os.environ.get("INFO_OLLAMA_MODEL")
            or _settings_model
            or str(_cfg("LLM_FUNCTIONS_MODEL", "pipeline", "llm_functions", default="qwen3.5:9b"))
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


def create_story_agent(
    llm: str | None = None,
    ollama_model: str | None = None,
    anthropic_model: str | None = None,
    **kwargs,
) -> Agent:
    """Create the Story agent — a creative writer with two skill-driven modes.

    The agent itself is a thin mode router (short system prompt); the detailed
    instructions for each mode live in ``skills_story/``:

    - ``story-synopsis`` (Mode A) — write a very short synopsis / logline.
    - ``story-scene``    (Mode B) — expand a synopsis into consistent scene
      descriptions for downstream start-frame + video generation.

    These skills are kept in a dedicated directory so the ComfyUI agents (which
    scan ``skills/``) never see them, and the Story agent never sees the ComfyUI
    skills.

    Reads ``llm.pipeline.story`` from settings.json (format: ``'provider,model'``),
    e.g. ``'claude,claude-haiku-4-5'`` or ``'ollama,qwen3.5:9b'``. Env var
    ``STORY_LLM`` overrides the combined setting; ``STORY_OLLAMA_MODEL`` or
    ``STORY_ANTHROPIC_MODEL`` override the provider-specific model.

    Defaults to Claude (``claude-haiku-4-5``) when no setting is present.

    Args:
        llm: ``'claude'`` or ``'ollama'``. Falls back to ``STORY_LLM`` env/settings.
        ollama_model: Ollama model override.
        anthropic_model: Anthropic model override (e.g. ``'claude-haiku-4-5'``).
        **kwargs: Forwarded to the Strands Agent constructor.
    """
    if ollama_model and llm is None:
        llm = "ollama"

    # Read combined 'provider,model' from settings (env var STORY_LLM still wins).
    _raw = str(_cfg("STORY_LLM", "pipeline", "story", default="claude,claude-haiku-4-5"))
    _settings_llm, _settings_model = _parse_llm_setting(_raw)
    resolved_llm = llm or _settings_llm or "claude"

    system_prompt = _load_system_prompt("story")

    # Load the story-only skills (Mode A / Mode B). Scoped to _STORY_SKILLS_DIR
    # so this agent sees only its two modes.
    story_plugins: list = []
    if _STORY_SKILLS_DIR.is_dir():
        skills_plugin = AgentSkills(skills=str(_STORY_SKILLS_DIR))
        story_plugins.append(skills_plugin)
        loaded = [s.name for s in skills_plugin.get_available_skills()]
        if loaded:
            print(f"[agentY:story] Loaded skills: {', '.join(loaded)}")

    if resolved_llm == "ollama":
        resolved_ollama = (
            ollama_model
            or os.environ.get("STORY_OLLAMA_MODEL")
            or _settings_model
            or str(_cfg("LLM_FUNCTIONS_MODEL", "pipeline", "llm_functions", default="qwen3.5:9b"))
        )
        return _make_agent(
            role="story",
            llm="ollama",
            system_prompt=system_prompt,
            tools=STORY_TOOLS,
            ollama_model=resolved_ollama,
            plugins=story_plugins or None,
            **kwargs,
        )

    # Otherwise use Anthropic/Claude.
    resolved_anthropic = (
        anthropic_model
        or os.environ.get("STORY_ANTHROPIC_MODEL")
        or _settings_model
        or str(_cfg("ANTHROPIC_MODEL", "anthropic", "model", default="claude-haiku-4-5"))
    )
    return _make_agent(
        role="story",
        llm=resolved_llm,
        dashscope_model=_settings_model,
        system_prompt=system_prompt,
        tools=STORY_TOOLS,
        anthropic_model=resolved_anthropic,
        plugins=story_plugins or None,
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

    Reads ``llm.pipeline.scout`` from settings.json (format ``'provider,model'``);
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
    _info_default = str(_cfg("INFO_LLM", "pipeline", "info", default="claude,claude-haiku-4-5"))
    _raw = str(_cfg("SEARCHWEB_LLM", "pipeline", "search_web", default=_info_default))
    _settings_llm, _settings_model = _parse_llm_setting(_raw)
    resolved_llm = llm or _settings_llm or "claude"

    system_prompt = _load_system_prompt("search_web")

    if resolved_llm == "ollama":
        resolved_ollama = (
            ollama_model
            or os.environ.get("SEARCHWEB_OLLAMA_MODEL")
            or _settings_model
            or str(_cfg("LLM_FUNCTIONS_MODEL", "pipeline", "llm_functions", default="qwen3.5:9b"))
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


def create_dop_agent(
    llm: str | None = None,
    ollama_model: str | None = None,
    anthropic_model: str | None = None,
    **kwargs,
) -> Agent:
    """Create the DoP (Director of Photography) agent — a stateless cinematographer.

    Given a **finished storyboard JSON spec or a single prompt/scene**, it applies
    concrete cinematography rules (lighting, composition, camera movement, colour)
    and returns the enriched result — the SAME storyboard JSON schema when handed a
    storyboard, or an enriched prompt when handed a single prompt. It writes text
    only and calls no tools.

    Two callers use it:
      • the Storyboard director — to enrich every start frame + shot before generation;
      • the Planner — as a ``dop`` step in a normal multi-step plan.

    Reads ``llm.pipeline.dop`` from settings.json (format: ``'provider,model'``);
    falls back to the Story-agent setting, then ``claude-haiku-4-5``. Env var
    ``DOP_LLM`` overrides the combined setting; ``DOP_OLLAMA_MODEL`` /
    ``DOP_ANTHROPIC_MODEL`` override the provider-specific model.

    Args:
        llm: ``'claude'`` or ``'ollama'``. Falls back to ``DOP_LLM`` env/settings.
        ollama_model: Ollama model override.
        anthropic_model: Anthropic model override.
        **kwargs: Forwarded to the Strands Agent constructor.
    """
    if ollama_model and llm is None:
        llm = "ollama"

    # Fall back to the Story-agent setting so no extra config is required.
    _story_default = str(_cfg("STORY_LLM", "pipeline", "story", default="claude,claude-haiku-4-5"))
    _raw = str(_cfg("DOP_LLM", "pipeline", "dop", default=_story_default))
    _settings_llm, _settings_model = _parse_llm_setting(_raw)
    resolved_llm = llm or _settings_llm or "claude"

    system_prompt = _load_system_prompt("dop")

    if resolved_llm == "ollama":
        resolved_ollama = (
            ollama_model
            or os.environ.get("DOP_OLLAMA_MODEL")
            or _settings_model
            or str(_cfg("LLM_FUNCTIONS_MODEL", "pipeline", "llm_functions", default="qwen3.5:9b"))
        )
        agent = _make_agent(
            role="dop",
            llm="ollama",
            system_prompt=system_prompt,
            tools=DOP_TOOLS,
            ollama_model=resolved_ollama,
            **kwargs,
        )
    else:
        resolved_anthropic = (
            anthropic_model
            or os.environ.get("DOP_ANTHROPIC_MODEL")
            or _settings_model
            or str(_cfg("ANTHROPIC_MODEL", "anthropic", "model", default="claude-haiku-4-5"))
        )
        agent = _make_agent(
            role="dop",
            llm=resolved_llm,
            dashscope_model=_settings_model,
            system_prompt=system_prompt,
            tools=DOP_TOOLS,
            anthropic_model=resolved_anthropic,
            **kwargs,
        )
    # Single-turn, stateless: each storyboard/prompt is enriched independently.
    agent.conversation_manager = SlidingWindowConversationManager(window_size=4)
    return agent


def create_DETECTUSERINTENT_agent(
    llm: str | None = None,
    ollama_model: str | None = None,
    anthropic_model: str | None = None,
    **kwargs,
) -> Agent:
    """Create the Detect User Intent agent — a stateless, tool-free intent classifier.

    Reads ``llm.pipeline.triage`` from settings.json (format: ``'provider,model'``,
    e.g. ``'ollama,qwen3:0.6b'`` or ``'claude,claude-haiku-4-5'``).
    Env var ``DETECTUSERINTENT_LLM`` overrides the full setting; ``DETECTUSERINTENT_OLLAMA_MODEL``
    or ``DETECTUSERINTENT_ANTHROPIC_MODEL`` override just the model.

    The agent has no tools and no meaningful conversation history — it reads
    the user message (optionally prefixed with session context) and returns a
    JSON ``{"intent": "...", "confidence": 0.0–1.0}`` object.

    Args:
        llm: ``'ollama'`` or ``'claude'``. Falls back to ``DETECTUSERINTENT_LLM`` env var.
        ollama_model: Ollama model override (e.g. ``'qwen3:0.6b'``).
        anthropic_model: Anthropic model override (e.g. ``'claude-haiku-4-5'``).
        **kwargs: Forwarded to the Strands Agent constructor.
    """
    if ollama_model and llm is None:
        llm = "ollama"

    # Read combined 'provider,model' from settings (env var DETECTUSERINTENT_LLM still wins).
    _raw = str(_cfg("DETECTUSERINTENT_LLM", "pipeline", "detect_user_intent", default="ollama"))
    _settings_llm, _settings_model = _parse_llm_setting(_raw)
    resolved_llm = llm or _settings_llm or "ollama"

    if resolved_llm == "ollama":
        resolved_ollama = (
            ollama_model
            or os.environ.get("DETECTUSERINTENT_OLLAMA_MODEL")
            or _settings_model
            or str(_cfg("LLM_FUNCTIONS_MODEL", "pipeline", "llm_functions", default="qwen3:0.6b"))
        )
        resolved_anthropic = (
            anthropic_model
            or os.environ.get("DETECTUSERINTENT_ANTHROPIC_MODEL")
            or str(_cfg("ANTHROPIC_MODEL", "anthropic", "model", default="claude-haiku-4-5"))
        )
    else:  # claude
        resolved_anthropic = (
            anthropic_model
            or os.environ.get("DETECTUSERINTENT_ANTHROPIC_MODEL")
            or _settings_model
            or str(_cfg("ANTHROPIC_MODEL", "anthropic", "model", default="claude-haiku-4-5"))
        )
        resolved_ollama = ollama_model or str(_cfg("LLM_FUNCTIONS_MODEL", "pipeline", "llm_functions", default="qwen3:0.6b"))

    system_prompt = _load_system_prompt("detect_user_intent")
    agent = _make_agent(
        role="detect_user_intent",
        llm=resolved_llm,
        dashscope_model=_settings_model,
        system_prompt=system_prompt,
        tools=DETECTUSERINTENT_TOOLS,
        ollama_model=resolved_ollama,
        anthropic_model=resolved_anthropic,
        **kwargs,
    )
    # Triage is single-turn and stateless — cap history to avoid stale
    # classification exchanges polluting future calls.
    agent.conversation_manager = SlidingWindowConversationManager(window_size=2)
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
    _raw = str(_cfg("ASSEMBLEWORKFLOW_LLM", "pipeline", "assemble_workflow", default="claude"))
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
    # Use the local-model variant of the Assemble Workflow system prompt for Ollama; the
    # standard prompt for Claude.  The local variant contains explicit step-by-step
    # patching instructions instead of skill-activation references.
    ASSEMBLEWORKFLOW_prompt_key = "assemble_workflow.local" if resolved_llm == "ollama" else "assemble_workflow"
    system_prompt = _load_system_prompt(ASSEMBLEWORKFLOW_prompt_key)

    # Load skills from the project-level skills/ directory.
    skills_plugins: list = []
    if _SKILLS_DIR.is_dir():
        skills_plugin = AgentSkills(skills=str(_SKILLS_DIR))
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
create_detect_user_intent_agent = create_DETECTUSERINTENT_agent


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

    _raw = str(_cfg("LEARNINGS_LLM", "pipeline", "learnings", default="ollama,qwen3.5:9b"))
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


def create_error_checker_agent(
    llm: str | None = None,
    ollama_model: str | None = None,
    anthropic_model: str | None = None,
    **kwargs,
) -> Agent:
    """Create the Error Checker agent — a single-turn post-execution log analyser.

    Runs after every ComfyUI workflow execution, fetches recent logs, and outputs
    a JSON verdict: ``ok``, ``error_fixable`` (with a concrete fix plan for the
    Brain), or ``error_unfixable`` (with a human-readable user message).

    Reads ``llm.pipeline.error_checker`` from settings.json (format:
    ``'provider,model'``).  Env var ``ERROR_CHECKER_LLM`` overrides the full
    setting; ``ERROR_CHECKER_OLLAMA_MODEL`` / ``ERROR_CHECKER_ANTHROPIC_MODEL``
    override just the model.  Defaults to the same model as the Brain.

    Args:
        llm: ``'claude'`` or ``'ollama'``. Falls back to ``ERROR_CHECKER_LLM`` env var.
        ollama_model: Ollama model override.
        anthropic_model: Anthropic model override.
        **kwargs: Forwarded to the Strands Agent constructor.
    """
    if ollama_model and llm is None:
        llm = "ollama"

    # Fall back to the brain setting so no extra config is needed out of the box.
    _ASSEMBLEWORKFLOW_default = str(_cfg("ASSEMBLEWORKFLOW_LLM", "pipeline", "assemble_workflow", default="claude,claude-haiku-4-5"))
    _raw = str(_cfg("ERROR_CHECKER_LLM", "pipeline", "error_checker", default=_ASSEMBLEWORKFLOW_default))
    _settings_llm, _settings_model = _parse_llm_setting(_raw)
    resolved_llm = llm or _settings_llm or "claude"

    if resolved_llm == "claude":
        resolved_anthropic = (
            anthropic_model
            or os.environ.get("ERROR_CHECKER_ANTHROPIC_MODEL")
            or _settings_model
            or str(_cfg("ANTHROPIC_MODEL", "anthropic", "model", default="claude-haiku-4-5"))
        )
        resolved_ollama = ollama_model or "qwen3.5:9b"
    else:  # ollama
        resolved_ollama = (
            ollama_model
            or os.environ.get("ERROR_CHECKER_OLLAMA_MODEL")
            or _settings_model
            or "qwen3.5:9b"
        )
        resolved_anthropic = (
            anthropic_model
            or os.environ.get("ERROR_CHECKER_ANTHROPIC_MODEL")
            or str(_cfg("ANTHROPIC_MODEL", "anthropic", "model", default="claude-haiku-4-5"))
        )

    system_prompt = _load_system_prompt("error_checker")

    # Load skills so the troubleshooting skill is available.
    ec_plugins: list = []
    if _SKILLS_DIR.is_dir():
        skills_plugin = AgentSkills(skills=str(_SKILLS_DIR))
        ec_plugins.append(skills_plugin)

    agent = _make_agent(
        role="error_checker",
        llm=resolved_llm,
        dashscope_model=_settings_model,
        system_prompt=system_prompt,
        tools=ERROR_CHECKER_TOOLS,
        ollama_model=resolved_ollama,
        anthropic_model=resolved_anthropic,
        plugins=ec_plugins or None,
        **kwargs,
    )
    # Single-turn — no persistent conversation history needed.
    agent.conversation_manager = SlidingWindowConversationManager(window_size=2)
    return agent

