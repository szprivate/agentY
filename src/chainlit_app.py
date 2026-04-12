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

from src.pipeline import create_pipeline
from src.utils.costs import compute_cost_from_usage


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

@cl.on_chat_start
async def on_chat_start() -> None:
    """Initialise a fresh pipeline for each user session."""
    await cl.Message(
        content="_⚙️ Initialising agentY pipeline — this may take a moment…_",
        author="system",
    ).send()

    try:
        pipeline = create_pipeline()
    except Exception as exc:
        await cl.Message(
            content=f"❌ Failed to create pipeline:\n```\n{exc}\n```",
            author="system",
        ).send()
        return

    cl.user_session.set("pipeline", pipeline)
    await cl.Message(
        content="✅ **agentY** is ready! Describe what you'd like to create or edit. "
                "You can attach images directly to your message.",
        author="system",
    ).send()


@cl.on_message
async def on_message(message: cl.Message) -> None:
    """Handle an incoming user message, optionally with image attachments."""
    pipeline = cl.user_session.get("pipeline")
    if pipeline is None:
        await cl.Message(
            content="⚠️ Pipeline not initialised. Please reload the page.",
            author="system",
        ).send()
        return

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

    try:
        async for event in pipeline.stream_async(content):
            chunk: str = ""
            if isinstance(event, dict):
                chunk = event.get("data", "") or ""
            if chunk:
                await response_msg.stream_token(chunk)
                # Check for newly saved images after every chunk that looks like
                # a save/executor line so we don't poll on every token.
                if "💾" in chunk or "Saved" in chunk or "executor" in chunk.lower():
                    await _flush_new_images()
    except Exception as exc:
        await response_msg.stream_token(f"\n\n❌ Pipeline error: {exc}")

    await response_msg.update()

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
