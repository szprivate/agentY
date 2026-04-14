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
try:
    import chainlit.data as cl_data
    from chainlit.data.sql_alchemy import SQLAlchemyDataLayer

    _conninfo = os.environ.get("DATABASE_URL")
    if _conninfo:
        cl_data._data_layer = SQLAlchemyDataLayer(conninfo=_conninfo)
    else:
        print("[chainlit] DATABASE_URL not set; skipping SQLAlchemyDataLayer init")
except Exception as _exc:  # pragma: no cover - non-critical init
    print(f"[chainlit] Could not initialise SQLAlchemyDataLayer: {_exc}")

from src.pipeline import create_pipeline
from src.utils.costs import compute_cost_from_usage
from src.utils.models import AgentSession
from src.utils.triage import triage as _run_triage, route as _route_intent


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
    """
    if not image_paths:
        return text or "(no message)"

    blocks: list = []
    valid_paths: list[str] = []

    for path in image_paths:
        try:
            from PIL import Image as PILImage  # noqa: PLC0415
            with PILImage.open(path) as img:
                buf = io.BytesIO()
                img.save(buf, format="PNG")
                image_bytes = buf.getvalue()
            blocks.append({
                "image": {
                    "format": "png",
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
    """Return True when *path* points to a supported image format."""
    return Path(path).suffix.lower() in {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp"}


# ── Chainlit lifecycle ────────────────────────────────────────────────────────

def _reset_pipeline_state(pipeline) -> None:
    """Wipe all per-conversation state from the shared pipeline singleton.

    Called whenever Chainlit starts or resumes a thread so that no history
    from a previous chat leaks into the new one.  Clears:
    - brain.messages         – in-memory LLM conversation history
    - pipeline._session      – AgentSession (chat_summaries, output paths, …)
    - pipeline._last_brainbriefing_json – cached researcher output
    """
    brain = getattr(pipeline, "_brain", None)
    if brain is not None and hasattr(brain, "messages"):
        brain.messages.clear()

    # Reset the AgentSession so triage no longer sees stale chat_summaries
    # from the previous thread, which would cause wrong follow-up routing.
    existing_session = getattr(pipeline, "_session", None)
    session_id = getattr(existing_session, "session_id", "default") if existing_session else "default"
    pipeline._session = AgentSession(session_id=session_id)  # noqa: SLF001

    # Clear the cached researcher JSON so the brain never inherits a
    # brainbriefing from an earlier thread.
    pipeline._last_brainbriefing_json = None  # noqa: SLF001


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

    Because the pipeline is a module-level singleton it cannot hold state for
    multiple threads simultaneously.  Reset everything to a clean slate so the
    resumed thread starts fresh rather than inheriting context from whatever
    was last active.
    """
    if _pipeline is None:
        return

    _reset_pipeline_state(_pipeline)

    cl.user_session.set("pipeline", _pipeline)
    cl.user_session.set("awaiting_answer", False)


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

    # ── Build Strands-compatible content ──────────────────────────────────
    content = _build_content(message.content or "", image_paths)

    # ── Stream the pipeline response ──────────────────────────────────────
    response_msg = cl.Message(content="")
    await response_msg.send()

    session = getattr(pipeline, "_session", None)
    sent_paths: set[str] = set(getattr(session, "current_output_paths", []))

    async def _flush_new_images() -> None:
        """Send any image paths that appeared in session since last check."""
        current: list[str] = list(getattr(session, "current_output_paths", []))
        new = [p for p in current if p not in sent_paths and os.path.isfile(p) and _is_image_path(p)]
        if new:
            elements = [cl.Image(path=p, name=Path(p).name, display="inline") for p in new]
            await cl.Message(
                content=f"🖼️ **{len(new)} image(s) ready:**",
                elements=elements,
            ).send()
            sent_paths.update(new)

    # Buffer tokens and flush in batches to reduce WebSocket round-trips.
    # Flushing every _STREAM_FLUSH_CHARS characters keeps latency low while
    # dramatically cutting the number of individual async sends vs. flushing
    # on every single token (which is what made Chainlit lag behind the CLI).
    _STREAM_FLUSH_CHARS = 12
    _token_buf: list[str] = []
    _token_buf_len: int = 0
    _full_response_parts: list[str] = []  # accumulates all chunks for post-response analysis

    async def _flush_token_buf() -> None:
        nonlocal _token_buf, _token_buf_len
        if _token_buf:
            await response_msg.stream_token("".join(_token_buf))
            _token_buf = []
            _token_buf_len = 0

    try:
        async for event in pipeline.stream_async(content):
            # ── QA failure from a new_planned_request step ────────────────
            if isinstance(event, dict) and event.get("qa_fail"):
                await _flush_token_buf()
                await response_msg.update()

                # Send images produced by the failed step.
                image_paths_from_event: list[str] = event.get("image_paths", [])
                new_qa_images = [
                    p for p in image_paths_from_event
                    if p not in sent_paths and os.path.isfile(p) and _is_image_path(p)
                ]
                if new_qa_images:
                    elements = [
                        cl.Image(path=p, name=Path(p).name, display="inline")
                        for p in new_qa_images
                    ]
                    await cl.Message(
                        content=f"🖼️ **Output from failed step ({len(new_qa_images)} image(s)):**",
                        elements=elements,
                    ).send()
                    sent_paths.update(new_qa_images)

                # Build a human-readable verdict summary.
                fail_details: list[dict] = event.get("fail_details", [])
                verdict_lines = "\n".join(
                    f"- `{Path(d['path']).name}`: {d['verdict']}"
                    for d in fail_details
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
                break  # stop consuming the generator — plan is paused

            # ── Normal streaming chunk ─────────────────────────────────────
            chunk: str = ""
            if isinstance(event, dict):
                chunk = event.get("data", "") or ""
            if chunk:
                _token_buf.append(chunk)
                _full_response_parts.append(chunk)
                _token_buf_len += len(chunk)
                # Flush when buffer is full enough, or immediately on lines
                # that signal image saves so the UI updates promptly.
                if _token_buf_len >= _STREAM_FLUSH_CHARS or "💾" in chunk or "Saved" in chunk or "executor" in chunk.lower():
                    await _flush_token_buf()
                    if "💾" in chunk or "Saved" in chunk or "executor" in chunk.lower():
                        await _flush_new_images()
        # Flush any remaining buffered tokens.
        await _flush_token_buf()
    except Exception as exc:
        await _flush_token_buf()
        await response_msg.stream_token(f"\n\n❌ Pipeline error: {exc}")

    await response_msg.update()

    # ── Update awaiting_answer for the next turn ──────────────────────────
    # If the brain's response contains a question in its closing section the
    # next user message is likely an answer rather than a brand-new request.
    _full_response = "".join(_full_response_parts)
    _tail = _full_response[-300:] if len(_full_response) > 300 else _full_response
    cl.user_session.set("awaiting_answer", "?" in _tail)

    # Final flush — catches any images that arrived with the last event.
    await _flush_new_images()

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
            cost_val, total_tokens = compute_cost_from_usage(usage, pipeline)
            summary += f"  —  💵 ${cost_val:.4f}  ({total_tokens:,} total)"
        except Exception:
            pass

        await cl.Message(content=summary, author="system").send()
    except Exception:
        pass
