#!/usr/bin/env python3
"""
agentY – Chainlit GUI entry point.

Launch with:
    chainlit run src/chainlit_app.py
    chainlit run src/chainlit_app.py -w      # auto-reload on file changes
    chainlit run src/chainlit_app.py --port 8080

The app reads model/pipeline configuration from config/settings.json and
.env, exactly the same as the console main.py entry point.
"""

from __future__ import annotations

import io
import os
import sys
from pathlib import Path

# ── Project root on sys.path ──────────────────────────────────────────────────
_project_root = Path(__file__).parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from dotenv import load_dotenv
load_dotenv(_project_root / ".env")

import chainlit as cl
# Chainlit auto-initialises its data layer from DATABASE_URL + APP_AWS_* env vars
# (set in .env). No manual _data_layer assignment needed.

from src.pipeline import create_pipeline
from src.utils.costs import compute_cost_from_usage
from src.utils.models import AgentSession
from src.utils.triage import triage as _run_triage, route as _route_intent

# Agent factories imported lazily inside the switch_model handler to avoid
# circular-import issues at module load time.


# ── Unload Ollama models before pipeline startup ──────────────────────────────

def _unload_ollama_models() -> None:
    """Send keep_alive=0 to Ollama for every model listed in config/settings.json."""
    import json as _json
    import urllib.request as _urlreq
    import urllib.error as _urlerr

    try:
        _cfg_path = _project_root / "config" / "settings.json"
        with open(_cfg_path, "r", encoding="utf-8") as _f:
            # Strip JSONC-style single-line comments before parsing.
            _lines = [ln for ln in _f if not ln.lstrip().startswith("//")]
            _cfg = _json.loads("".join(_lines))

        _llm = _cfg.get("llm", {})
        _host: str = _llm.get("ollama", {}).get("host", "http://localhost:11434")
        _pipeline_cfg: dict = _llm.get("pipeline", {})

        # Collect unique Ollama model names.
        # Values prefixed with "ollama," → strip prefix.
        # Bare values (no comma) → treat as Ollama model names.
        _models: set[str] = set()
        for _val in _pipeline_cfg.values():
            if not isinstance(_val, str):
                continue
            if _val.startswith("ollama,"):
                _models.add(_val.split(",", 1)[1])
            elif "," not in _val and _val:
                _models.add(_val)

        _url = f"{_host.rstrip('/')}/api/generate"
        for _model in sorted(_models):
            try:
                _payload = _json.dumps({"model": _model, "keep_alive": 0}).encode()
                _req = _urlreq.Request(
                    _url, data=_payload, method="POST",
                    headers={"Content-Type": "application/json"},
                )
                with _urlreq.urlopen(_req, timeout=5):
                    pass
                print(f"[chainlit] Unloaded Ollama model: {_model}")
            except _urlerr.URLError as _exc:
                print(f"[chainlit] Could not unload Ollama model '{_model}': {_exc}")
    except Exception as _exc:
        print(f"[chainlit] Ollama model unload skipped: {_exc}")


_unload_ollama_models()


# ── Module-level pipeline singleton ──────────────────────────────────────────
try:
    _pipeline = create_pipeline()
except Exception as _pipeline_exc:
    print(f"[chainlit] Failed to create pipeline at startup: {_pipeline_exc}")
    _pipeline = None


# Simple password auth callback for Chainlit (adjust as needed)
@cl.password_auth_callback
def auth_callback(username: str, password: str):
    _chainlit_user = os.environ.get("CHAINLIT_USERNAME", "yourname")
    _chainlit_pass = os.environ.get("CHAINLIT_PASSWORD", "yourpassword")
    if (username, password) == (_chainlit_user, _chainlit_pass):
        return cl.User(identifier=_chainlit_user, metadata={"role": "admin"})
    return None


# ── Content builder ───────────────────────────────────────────────────────────

def _build_content(text: str, image_paths: list[str]) -> list | str:
    """Convert user text + uploaded image paths into Strands content blocks.

    Mirrors the pattern used by agentY_server.py so the pipeline handles
    both plain-text prompts and multimodal (text + image) inputs correctly.
    Images are always downsized to satisfy Claude's 5 MB / 1568 px constraints.
    """
    if not image_paths:
        return text or "(no message)"

    from src.tools.image_handling import _downsize, _detect_format  # noqa: PLC0415

    blocks: list = []
    valid_paths: list[str] = []

    for path in image_paths:
        try:
            raw = Path(path).read_bytes()
            img_fmt = _detect_format(path) or "png"
            image_bytes = _downsize(raw, img_fmt)
            blocks.append({
                "image": {
                    "format": img_fmt,
                    "source": {"bytes": image_bytes},
                }
            })
            valid_paths.append(path)
        except Exception as exc:
            print(f"[chainlit] Could not load image '{path}': {exc}")

    if not blocks:
        return text or "(no message)"

    path_lines = "\n".join(
        f"  - {p}  [image, use this path for ComfyUI input]"
        for p in valid_paths
    )
    paths_info = (
        f"\n\nAttached image file paths (use these for ComfyUI):\n{path_lines}"
        if path_lines else ""
    )
    intro = text if text else "The user sent an image for processing."
    blocks.insert(0, {"text": intro + paths_info})
    return blocks


def _is_image_path(path: str) -> bool:
    return Path(path).suffix.lower() in {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp"}


def _is_video_path(path: str) -> bool:
    return Path(path).suffix.lower() in {".mp4", ".mov", ".avi", ".mkv", ".webm"}


def _is_file_output_path(path: str) -> bool:
    return Path(path).suffix.lower() in {".json"}


def _parse_think_chunk(chunk: str, state: dict) -> tuple[str, str]:
    """Split *chunk* into (normal_text, think_text) tracking <think>...</think> state.

    *state* is a mutable dict with keys ``in_think`` (bool) and ``buf`` (str
    lookahead for tags that span chunk boundaries).  Modified in-place.
    """
    OPEN, CLOSE = "<think>", "</think>"
    combined = state["buf"] + chunk
    state["buf"] = ""
    normal: list[str] = []
    think: list[str] = []
    while combined:
        if not state["in_think"]:
            idx = combined.find(OPEN)
            if idx == -1:
                for cut in range(min(len(OPEN) - 1, len(combined)), 0, -1):
                    if OPEN[:cut] == combined[-cut:]:
                        normal.append(combined[:-cut])
                        state["buf"] = combined[-cut:]
                        combined = ""
                        break
                else:
                    normal.append(combined)
                    combined = ""
            else:
                normal.append(combined[:idx])
                combined = combined[idx + len(OPEN):]
                state["in_think"] = True
        else:
            idx = combined.find(CLOSE)
            if idx == -1:
                for cut in range(min(len(CLOSE) - 1, len(combined)), 0, -1):
                    if CLOSE[:cut] == combined[-cut:]:
                        think.append(combined[:-cut])
                        state["buf"] = combined[-cut:]
                        combined = ""
                        break
                else:
                    think.append(combined)
                    combined = ""
            else:
                think.append(combined[:idx])
                combined = combined[idx + len(CLOSE):]
                state["in_think"] = False
    return "".join(normal), "".join(think)


# ── Chainlit lifecycle ────────────────────────────────────────────────────────

def _reset_pipeline_state(pipeline) -> None:
    """Wipe all per-conversation state from the shared pipeline singleton.

    Called when a new thread starts so no history from a previous chat leaks
    in.  Clears brain.messages, AgentSession, and cached researcher output.
    """
    brain = getattr(pipeline, "_brain", None)
    if brain is not None and hasattr(brain, "messages"):
        brain.messages.clear()

    existing_session = getattr(pipeline, "_session", None)
    session_id = getattr(existing_session, "session_id", "default") if existing_session else "default"
    pipeline._session = AgentSession(session_id=session_id)  # noqa: SLF001
    pipeline._last_brainbriefing_json = None  # noqa: SLF001


def _save_thread_state(pipeline) -> None:
    """Snapshot the current pipeline state into Chainlit's per-thread session.

    Called at the end of every on_message turn so the compressed brain summary
    and session metadata survive thread navigation (on_chat_resume).
    """
    brain = getattr(pipeline, "_brain", None)
    if brain is not None and hasattr(brain, "messages"):
        cl.user_session.set("brain_messages", list(brain.messages))

    session = getattr(pipeline, "_session", None)
    if session is not None:
        cl.user_session.set("agent_session", session.model_dump())

    cl.user_session.set(
        "last_brainbriefing_json",
        getattr(pipeline, "_last_brainbriefing_json", None),
    )


def _restore_thread_state(pipeline) -> None:
    """Restore pipeline state from Chainlit's per-thread session on resume.

    If the thread has no saved state (e.g. the very first resume before any
    message was processed), falls back to a clean reset.
    """
    brain_messages = cl.user_session.get("brain_messages")
    if brain_messages is None:
        _reset_pipeline_state(pipeline)
        return

    brain = getattr(pipeline, "_brain", None)
    if brain is not None and hasattr(brain, "messages"):
        brain.messages[:] = brain_messages

    agent_session_data = cl.user_session.get("agent_session")
    if agent_session_data is not None:
        try:
            pipeline._session = AgentSession(**agent_session_data)  # noqa: SLF001
        except Exception:
            pass

    pipeline._last_brainbriefing_json = cl.user_session.get(  # noqa: SLF001
        "last_brainbriefing_json"
    )


@cl.on_chat_start
async def on_chat_start() -> None:
    """Store the shared pipeline in the user session and greet the user."""
    if _pipeline is None:
        await cl.Message(
            content=f"❌ Failed to create pipeline:\n```\n{_pipeline_exc}\n```",
            author="system",
        ).send()
        return

    # Reset all per-conversation state so each new thread starts clean.
    _reset_pipeline_state(_pipeline)

    cl.user_session.set("pipeline", _pipeline)
    cl.user_session.set("awaiting_answer", False)


@cl.on_chat_resume
async def on_chat_resume(thread) -> None:  # noqa: ARG001
    """Called when the user navigates to an existing thread from the sidebar.

    Restores the compressed brain summary and session state that were saved at
    the end of the last turn in this thread, so the user can continue where
    they left off without losing context.  Falls back to a clean reset if no
    state was saved yet (e.g. a thread that was never completed).
    """
    if _pipeline is None:
        return

    _restore_thread_state(_pipeline)

    cl.user_session.set("pipeline", _pipeline)


@cl.on_message
async def on_message(message: cl.Message) -> None:
    """Handle an incoming user message, optionally with image attachments."""
    # ── Built-in commands ─────────────────────────────────────────────────
    _text = (message.content or "").strip()
    if _text.lower() in {"restart", "/restart"}:
        await cl.Message(content="🔄 Restarting agent…", author="system").send()
        try:
            new_pipeline = create_pipeline()
        except Exception as _exc:
            await cl.Message(
                content=f"❌ Failed to restart pipeline:\n```\n{_exc}\n```",
                author="system",
            ).send()
            return
        cl.user_session.set("pipeline", new_pipeline)
        cl.user_session.set("awaiting_answer", False)
        _reset_pipeline_state(new_pipeline)
        await cl.Message(content="✅ Agent restarted successfully.", author="system").send()
        return

    if _text.lower() in {"stop", "/stop", "!stop", "shutdown", "/shutdown"}:
        await cl.Message(content="🛑 Stopping agent…", author="system").send()
        import signal
        os.kill(os.getpid(), signal.SIGTERM)
        return

    if _text.lower() in {"unload", "/unload", "unload models", "!unload"}:
        await cl.Message(content="⏏️ Unloading Ollama models from VRAM…", author="system").send()
        try:
            from src.tools.agent_control import unload_ollama_models
            unloaded = unload_ollama_models()
            if unloaded:
                names = ", ".join(f"`{m}`" for m in unloaded)
                await cl.Message(
                    content=f"✅ Unloaded: {names}",
                    author="system",
                ).send()
            else:
                await cl.Message(
                    content="⚠️ No models were unloaded (Ollama unreachable or no models loaded).",
                    author="system",
                ).send()
        except Exception as _exc:
            await cl.Message(
                content=f"❌ Unload failed:\n```\n{_exc}\n```",
                author="system",
            ).send()
        return

    if _text.lower() in {"clear_vram", "/clear_vram", "clearvram", "/clearvram"}:
        await cl.Message(content="🧹 Clearing VRAM…", author="system").send()
        try:
            from src.tools.comfyui import free_memory as _free_memory
            import json as _json
            _result = _json.loads(_free_memory())
            if "error" not in _result:
                await cl.Message(
                    content="✅ VRAM cleared — models unloaded and GPU cache freed.",
                    author="system",
                ).send()
            else:
                await cl.Message(
                    content=f"❌ Clear VRAM failed:\n```\n{_result.get('error')}\n```",
                    author="system",
                ).send()
        except Exception as _exc:
            await cl.Message(
                content=f"❌ Clear VRAM failed:\n```\n{_exc}\n```",
                author="system",
            ).send()
        return

    if _text.lower() in {"clearhistory", "/clearhistory", "clear history", "/clear history"}:
        from chainlit.data import get_data_layer
        from chainlit.types import Pagination, ThreadFilter

        data_layer = get_data_layer()
        if data_layer is None:
            await cl.Message(
                content="⚠️ No data layer configured — history is not persisted.",
                author="system",
            ).send()
            return

        try:
            _user = cl.user_session.get("user")
            _user_id: str | None = getattr(_user, "id", None) if _user else None

            deleted_count = 0
            cursor: str | None = None
            while True:
                page: object = await data_layer.list_threads(
                    pagination=Pagination(first=100, cursor=cursor),
                    filters=ThreadFilter(userId=_user_id),
                )
                threads = getattr(page, "data", []) or []
                if not threads:
                    break
                for thread in threads:
                    tid = thread.get("id") if isinstance(thread, dict) else getattr(thread, "id", None)
                    if tid:
                        await data_layer.delete_thread(tid)
                        deleted_count += 1
                next_cursor = getattr(getattr(page, "pageInfo", None), "endCursor", None)
                has_next = getattr(getattr(page, "pageInfo", None), "hasNextPage", False)
                if not has_next or not next_cursor:
                    break
                cursor = next_cursor

            await cl.Message(
                content=f"🗑️ Cleared {deleted_count} thread(s) from history. Refresh the page to see the updated sidebar.",
                author="system",
            ).send()
        except Exception as _exc:
            await cl.Message(
                content=f"❌ Failed to clear history:\n```\n{_exc}\n```",
                author="system",
            ).send()
        return

    # ── /add_workflow <path_to_workflow.json> ─────────────────────────────────
    if _text.lower().startswith("/add_workflow"):
        _parts = _text.split(None, 1)
        if len(_parts) < 2:
            await cl.Message(
                content="⚠️ Usage: `/add_workflow <path_to_workflow.json>`",
                author="system",
            ).send()
            return
        _wf_path = Path(_parts[1].strip())
        if not _wf_path.exists():
            await cl.Message(
                content=f"❌ File not found: `{_wf_path}`",
                author="system",
            ).send()
            return
        try:
            import json as _json
            from src.utils.workflow_parser import parse_workflow as _parse_workflow, _custom_index_path
            with open(_wf_path, encoding="utf-8") as _f:
                _wf_data = _json.load(_f)
            _stem = _wf_path.stem
            _entry = _parse_workflow(_wf_data, name=_stem, update_index=True)
            # Update config/workflow_templates.json
            _templates_path = _project_root / "config" / "workflow_templates.json"
            if _templates_path.exists():
                _tpl = _json.loads(_templates_path.read_text(encoding="utf-8"))
            else:
                _tpl = {}
            if _stem not in _tpl:
                _tpl[_stem] = ""
                _templates_path.write_text(
                    _json.dumps(_tpl, indent=4, ensure_ascii=False) + "\n",
                    encoding="utf-8",
                )
                _tpl_msg = f" Added `{_stem}` to `config/workflow_templates.json`."
            else:
                _tpl_msg = f" `{_stem}` already present in `config/workflow_templates.json`."
            _idx = _custom_index_path()
            await cl.Message(
                content=f"✅ Workflow `{_stem}` added to `{_idx}`.{_tpl_msg}",
                author="system",
            ).send()
        except Exception as _exc:
            await cl.Message(
                content=f"❌ Failed to add workflow:\n```\n{_exc}\n```",
                author="system",
            ).send()
        return

    # ── /remove_workflow <name> ───────────────────────────────────────────────
    if _text.lower().startswith("/remove_workflow"):
        _parts = _text.split(None, 1)
        if len(_parts) < 2:
            await cl.Message(
                content="⚠️ Usage: `/remove_workflow <template_name>`",
                author="system",
            ).send()
            return
        _wf_name = _parts[1].strip()
        try:
            import json as _json
            from src.utils.workflow_parser import workflow_remove as _workflow_remove, _custom_index_path
            _idx = _workflow_remove(_wf_name)
            # Update config/workflow_templates.json
            _templates_path = _project_root / "config" / "workflow_templates.json"
            if _templates_path.exists():
                _tpl = _json.loads(_templates_path.read_text(encoding="utf-8"))
                if _wf_name in _tpl:
                    del _tpl[_wf_name]
                    _templates_path.write_text(
                        _json.dumps(_tpl, indent=4, ensure_ascii=False) + "\n",
                        encoding="utf-8",
                    )
                    _tpl_msg = f" Removed `{_wf_name}` from `config/workflow_templates.json`."
                else:
                    _tpl_msg = f" `{_wf_name}` not found in `config/workflow_templates.json`."
            else:
                _tpl_msg = ""
            await cl.Message(
                content=f"✅ Workflow `{_wf_name}` removed from `{_idx}`.{_tpl_msg}",
                author="system",
            ).send()
        except Exception as _exc:
            await cl.Message(
                content=f"❌ Failed to remove workflow:\n```\n{_exc}\n```",
                author="system",
            ).send()
        return

    # ── /switch_model <agent> <provider,model> ────────────────────────────────
    if _text.lower().startswith("/switch_model") or _text.lower().startswith("switch_model"):
        _parts = _text.split(None, 2)  # [cmd, agent_name, provider,model]
        _AGENTS = {"researcher", "brain", "info", "triage", "planner"}
        if len(_parts) < 3:
            await cl.Message(
                content=(
                    "⚠️ Usage: `/switch_model <agent> <provider,model>`\n\n"
                    f"Agents: `{', '.join(sorted(_AGENTS))}`\n"
                    "Examples:\n"
                    "- `/switch_model brain claude,claude-opus-4-7`\n"
                    "- `/switch_model researcher ollama,qwen3:14b`"
                ),
                author="system",
            ).send()
            return
        _agent_name = _parts[1].lower()
        _llm_spec = _parts[2].strip()
        if _agent_name not in _AGENTS:
            await cl.Message(
                content=f"❌ Unknown agent `{_agent_name}`. Valid agents: `{', '.join(sorted(_AGENTS))}`",
                author="system",
            ).send()
            return
        _pipeline = cl.user_session.get("pipeline")
        if _pipeline is None:
            await cl.Message(content="⚠️ Pipeline not initialised. Please reload the page.", author="system").send()
            return
        _provider, _, _model = _llm_spec.partition(",")
        _provider = _provider.strip().lower()
        _model = _model.strip()
        if _provider not in {"claude", "ollama"}:
            await cl.Message(
                content=f"❌ Unknown provider `{_provider}`. Use `claude` or `ollama`.",
                author="system",
            ).send()
            return
        await cl.Message(
            content=f"🔄 Switching `{_agent_name}` to `{_provider},{_model}`…",
            author="system",
        ).send()
        try:
            from src.agent import (
                create_researcher_agent,
                create_brain_agent,
                create_info_agent,
                create_triage_agent,
                create_planner_agent,
            )
            _kwargs = {"llm": _provider}
            if _model:
                if _provider == "ollama":
                    _kwargs["ollama_model"] = _model
                else:
                    _kwargs["anthropic_model"] = _model
            _factory_map = {
                "researcher": create_researcher_agent,
                "brain":      create_brain_agent,
                "info":       create_info_agent,
                "triage":     create_triage_agent,
                "planner":    create_planner_agent,
            }
            _new_agent = _factory_map[_agent_name](**_kwargs)
            _attr_map = {
                "researcher": "_researcher",
                "brain":      "_brain",
                "info":       "_info_agent",
                "triage":     "_triage_agent",
                "planner":    "_planner_agent",
            }
            setattr(_pipeline, _attr_map[_agent_name], _new_agent)
            _display = f"`{_provider},{_model}`" if _model else f"`{_provider}`"
            await cl.Message(
                content=f"✅ `{_agent_name}` now using {_display}.",
                author="system",
            ).send()
        except Exception as _exc:
            await cl.Message(
                content=f"❌ Failed to switch model:\n```\n{_exc}\n```",
                author="system",
            ).send()
        return

    pipeline = cl.user_session.get("pipeline")
    if pipeline is None:
        await cl.Message(
            content="⚠️ Pipeline not initialised. Please reload the page.",
            author="system",
        ).send()
        return

    # ── Context continuation: triage-aware history management ─────────────
    # If the agent posed a question in its last response, check whether the
    # user is answering it (follow-up) or starting a new request entirely.
    # For follow-ups the pipeline keeps brain.messages intact automatically;
    # for new requests we clear it here as well so the state is consistent.
    _awaiting_answer: bool = cl.user_session.get("awaiting_answer", False)
    if _awaiting_answer:
        _triage_agent = getattr(pipeline, "_triage_agent", None)
        _pip_session  = getattr(pipeline, "_session",      None)
        if _triage_agent is not None and _pip_session is not None:
            try:
                _tr      = await _run_triage(_text, _pip_session, {}, _triage_agent)
                _handler = _route_intent(_tr)
                if _handler in ("researcher", "planner"):
                    # User switched to a new topic → wipe stale history now.
                    # The pipeline will also clear it, but doing it here keeps
                    # the Chainlit layer's view of the session self-consistent.
                    _brain = getattr(pipeline, "_brain", None)
                    if _brain is not None and hasattr(_brain, "messages"):
                        _brain.messages.clear()
                    cl.user_session.set("awaiting_answer", False)
            except Exception as _tr_exc:
                print(f"[chainlit] Triage continuation check failed: {_tr_exc}")

    # ── Collect uploaded image paths ──────────────────────────────────────
    image_paths: list[str] = []
    if message.elements:
        for element in message.elements:
            # Chainlit attaches files as cl.File / cl.Image elements with a
            # `.path` attribute pointing to a server-side temp file.
            path: str | None = getattr(element, "path", None)
            if path and os.path.isfile(path):
                image_paths.append(path)

    # ── Persist uploaded image paths in session for future triage turns ───
    if image_paths:
        _pip_session = getattr(pipeline, "_session", None)
        if _pip_session is not None:
            _pip_session.last_user_input_images = image_paths

    # ── Build Strands-compatible content ──────────────────────────────────
    content = _build_content(message.content or "", image_paths)

    # ── Stream the pipeline response ──────────────────────────────────────
    session = getattr(pipeline, "_session", None)
    sent_paths: set[str] = set(getattr(session, "current_output_paths", []))

    # Response message created lazily so the researcher step appears above it.
    response_msg: cl.Message | None = None

    async def _ensure_response_msg() -> cl.Message:
        nonlocal response_msg
        if response_msg is None:
            response_msg = cl.Message(content="")
            await response_msg.send()
        return response_msg

    async def _flush_new_outputs() -> None:
        """Send new images, videos, and workflow files that appeared in session."""
        current: list[str] = list(getattr(session, "current_output_paths", []))
        new_paths = [p for p in current if p not in sent_paths and os.path.isfile(p)]
        if not new_paths:
            return
        imgs  = [cl.Image(path=p, name=Path(p).name, display="inline") for p in new_paths if _is_image_path(p)]
        vids  = [cl.Video(path=p, name=Path(p).name, display="inline") for p in new_paths if _is_video_path(p)]
        files = [cl.File(path=p,  name=Path(p).name, display="inline") for p in new_paths if _is_file_output_path(p)]
        if imgs:
            await cl.Message(content=f"🖼️ **{len(imgs)} image(s) ready:**", elements=imgs).send()
        if vids:
            await cl.Message(content=f"🎬 **{len(vids)} video(s) ready:**", elements=vids).send()
        if files:
            await cl.Message(content=f"📄 **{len(files)} file(s) ready:**", elements=files).send()
        sent_paths.update(new_paths)

    _STREAM_FLUSH_CHARS = 12
    _token_buf: list[str] = []
    _token_buf_len: int = 0
    _full_response_parts: list[str] = []

    async def _flush_token_buf() -> None:
        nonlocal _token_buf, _token_buf_len
        if _token_buf:
            msg = await _ensure_response_msg()
            await msg.stream_token("".join(_token_buf))
            _token_buf = []
            _token_buf_len = 0

    # ── Researcher streaming ───────────────────────────────────────────────
    _in_researcher: bool = False
    _researcher_step: cl.Step | None = None

    # ── Brain streaming ────────────────────────────────────────────────────
    _in_brain: bool = False
    _brain_step: cl.Step | None = None

    # ── Think-block parser state ───────────────────────────────────────────
    _think_state: dict = {"in_think": False, "buf": ""}
    _think_step: cl.Step | None = None

    # ── Planner task list ──────────────────────────────────────────────────
    _task_list: cl.TaskList | None = None
    _tasks: list[cl.Task] = []

    try:
        async for event in pipeline.stream_async(content):
            if not isinstance(event, dict):
                continue

            # ── QA failure ────────────────────────────────────────────────
            if event.get("qa_fail"):
                await _flush_token_buf()
                if response_msg:
                    await response_msg.update()

                fail_paths: list[str] = event.get("image_paths", [])
                new_qa = [p for p in fail_paths if p not in sent_paths and os.path.isfile(p)]
                if new_qa:
                    imgs = [cl.Image(path=p, name=Path(p).name, display="inline") for p in new_qa if _is_image_path(p)]
                    vids = [cl.Video(path=p, name=Path(p).name, display="inline") for p in new_qa if _is_video_path(p)]
                    if imgs:
                        await cl.Message(content=f"🖼️ **Output from failed step ({len(imgs)} image(s)):**", elements=imgs).send()
                    if vids:
                        await cl.Message(content=f"🎬 **Output from failed step ({len(vids)} video(s)):**", elements=vids).send()
                    sent_paths.update(new_qa)

                fail_details: list[dict] = event.get("fail_details", [])
                verdict_lines = "\n".join(
                    f"- `{Path(d['path']).name}`: {d['verdict']}" for d in fail_details
                )
                await cl.Message(
                    content=(
                        "⚠️ **QA check failed for this step.**\n\n"
                        + (verdict_lines + "\n\n" if verdict_lines else "")
                        + "Remaining plan steps have been skipped.\n\n"
                        "Reply **continue** to proceed with the next steps anyway, "
                        "or describe what should be changed."
                    ),
                    author="system",
                ).send()
                break

            # ── Researcher start — open streaming step ────────────────────
            if event.get("_researcher_start"):
                _in_researcher = True
                _researcher_step = cl.Step(name="🔍 Researcher", type="tool")
                await _researcher_step.send()
                continue

            # ── Researcher done — finalise streaming step ─────────────────
            if event.get("_researcher_done"):
                _in_researcher = False
                if _researcher_step is not None:
                    await _researcher_step.update()
                    _researcher_step = None
                continue

            # ── Brain start — open streaming step ────────────────────────
            if event.get("_brain_start"):
                _in_brain = True
                _brain_step = cl.Step(name="🧠 Brain", type="tool")
                await _brain_step.send()
                continue

            # ── Brain done — finalise streaming step ──────────────────────
            if event.get("_brain_done"):
                _in_brain = False
                if _brain_step is not None:
                    await _brain_step.update()
                    _brain_step = None
                continue

            # ── Plan ready — create task list ─────────────────────────────
            if event.get("_plan_ready"):
                _task_list = cl.TaskList()
                _task_list.status = "Running..."
                _tasks = [
                    cl.Task(title=s["description"], status=cl.TaskStatus.READY)
                    for s in event.get("steps", [])
                ]
                for t in _tasks:
                    await _task_list.add_task(t)
                await _task_list.send()
                continue

            # ── Step start — mark task running ────────────────────────────
            if event.get("_step_start"):
                _idx = event["idx"]
                if _task_list is not None and _idx < len(_tasks):
                    _tasks[_idx].status = cl.TaskStatus.RUNNING
                    await _task_list.send()
                continue

            # ── Step done — mark task complete ────────────────────────────
            if event.get("_step_done"):
                _idx = event["idx"]
                _failed = event.get("failed", False)
                if _task_list is not None and _idx < len(_tasks):
                    _tasks[_idx].status = cl.TaskStatus.FAILED if _failed else cl.TaskStatus.DONE
                    if _idx == len(_tasks) - 1 or _failed:
                        _task_list.status = "Failed" if _failed else "Done"
                    await _task_list.send()
                continue

            # ── Extended thinking (Anthropic reasoning blocks) ─────────────
            reasoning_text: str = event.get("reasoningText") or ""
            if reasoning_text:
                if _in_researcher:
                    if _researcher_step is not None:
                        await _researcher_step.stream_token(reasoning_text)
                elif _in_brain:
                    if _brain_step is not None:
                        await _brain_step.stream_token(reasoning_text)
                else:
                    if _think_step is None:
                        _think_step = cl.Step(name="💭 Thinking")
                        await _think_step.send()
                    await _think_step.stream_token(reasoning_text)
                continue

            # Reasoning block complete (signature marks end of thinking block)
            if event.get("reasoning_signature") is not None and not _in_researcher and not _in_brain:
                if _think_step is not None:
                    await _think_step.update()
                    _think_step = None
                continue

            # ── Normal data chunk ─────────────────────────────────────────
            chunk: str = event.get("data", "") or ""
            if not chunk:
                continue

            if _in_researcher:
                if _researcher_step is not None:
                    await _researcher_step.stream_token(chunk)
                continue

            if _in_brain:
                if _brain_step is not None:
                    await _brain_step.stream_token(chunk)
                continue

            # Split out <think>…</think> blocks
            _was_thinking = _think_state["in_think"]
            normal_text, think_text = _parse_think_chunk(chunk, _think_state)
            _now_thinking = _think_state["in_think"]

            if think_text:
                if _think_step is None:
                    _think_step = cl.Step(name="💭 Thinking")
                    await _think_step.send()
                await _think_step.stream_token(think_text)

            # Think block just closed — finalise step
            if _was_thinking and not _now_thinking and _think_step is not None:
                await _think_step.update()
                _think_step = None

            if normal_text:
                _full_response_parts.append(normal_text)
                _token_buf.append(normal_text)
                _token_buf_len += len(normal_text)
                _output_signal = any(kw in normal_text for kw in ("💾", "Saved", "executor"))
                if _token_buf_len >= _STREAM_FLUSH_CHARS or _output_signal:
                    await _flush_token_buf()
                    if _output_signal:
                        await _flush_new_outputs()

        await _flush_token_buf()
        # Finalise any still-open brain/think steps
        if _brain_step is not None:
            await _brain_step.update()
        if _think_step is not None:
            await _think_step.update()
        # Mark task list done if all steps completed
        if _task_list is not None and all(t.status == cl.TaskStatus.DONE for t in _tasks):
            _task_list.status = "Done"
            await _task_list.send()

    except Exception as exc:
        await _flush_token_buf()
        if _researcher_step is not None:
            await _researcher_step.update()
            _researcher_step = None
        if _brain_step is not None:
            await _brain_step.update()
            _brain_step = None
        msg = await _ensure_response_msg()
        await msg.stream_token(f"\n\n❌ Pipeline error: {exc}")

    if response_msg is not None:
        await response_msg.update()

    # ── Update awaiting_answer for the next turn ──────────────────────────
    _full_response = "".join(_full_response_parts)
    _tail = _full_response[-300:] if len(_full_response) > 300 else _full_response
    cl.user_session.set("awaiting_answer", "?" in _tail)

    # Final flush — catches any outputs that arrived with the last event.
    await _flush_new_outputs()

    # ── Persist thread state so on_chat_resume can restore it ────────────
    _save_thread_state(pipeline)

    # ── Token / cost summary ──────────────────────────────────────────────
    try:
        usage = pipeline.event_loop_metrics.accumulated_usage
        in_tok = usage.get("inputTokens", 0)
        out_tok = usage.get("outputTokens", 0)
        cache_read = usage.get("cacheReadInputTokens", 0)
        cache_write = usage.get("cacheWriteInputTokens", 0)

        parts = [f"{in_tok:,} in", f"{out_tok:,} out"]
        if cache_read:
            parts.append(f"{cache_read:,} cache hit")
        if cache_write:
            parts.append(f"{cache_write:,} cache write")
        summary = f"🪙 Tokens: {' | '.join(parts)}"

        try:
            if hasattr(pipeline, "compute_turn_cost"):
                cost_val, total_tokens = pipeline.compute_turn_cost()
            else:
                cost_val, total_tokens = compute_cost_from_usage(usage, pipeline)
            summary += f"  —  💵 ${cost_val:.4f}  ({total_tokens:,} total)"
        except Exception:
            pass

        await cl.Message(content=summary, author="system").send()
    except Exception:
        pass
