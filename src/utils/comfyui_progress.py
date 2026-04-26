"""
comfyui_progress – WebSocket-based live progress streamer for ComfyUI jobs.

Connects to ``ws://<comfyui>/ws?clientId=<uuid>`` and yields one-line status
updates (queue position, per-node progress bars, errors) as ComfyUI emits
events.  This is the sole completion path for ComfyUI jobs in agentY.

Yields:
    str  — human-readable status line (progress bar, node-start, etc.)
    dict — terminal result, exactly one of:
              {"history": <stripped_history_dict>}     on success
              {"error": str, "details"?: dict}         on error / timeout

The caller treats any dict yield as the end of the stream.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import AsyncGenerator
from urllib.parse import urlparse

logger = logging.getLogger("agentY.comfyui_progress")


def _bar(value: int, max_value: int, width: int = 20) -> str:
    if max_value <= 0:
        return ""
    pct = int(value / max_value * 100)
    filled = int(width * value / max_value)
    return f"[{'█' * filled}{'░' * (width - filled)}] {value}/{max_value} ({pct}%)"


def _check_history(client, prompt_id: str):
    """Inline check of /history/{prompt_id}.  Returns terminal dict or None."""
    from src.tools.comfyui import _strip_history

    try:
        raw = client.get(f"/history/{prompt_id}")
    except Exception as exc:
        logger.debug("history check failed: %s", exc)
        return None
    if not isinstance(raw, dict) or prompt_id not in raw:
        return None
    entry = raw[prompt_id]
    status_info = entry.get("status", {})
    if status_info.get("completed"):
        return {"history": _strip_history(raw)}
    if status_info.get("status_str") == "error":
        return {"error": "ComfyUI job failed", "details": _strip_history(raw)}
    return None


async def stream_comfyui_job(
    prompt_id: str,
    client_id: str,
    *,
    timeout: float = 30 * 60,
) -> AsyncGenerator:
    """Stream live progress for *prompt_id* via the ComfyUI WebSocket.

    Args:
        prompt_id: Returned by POST /prompt.
        client_id: The same client_id passed to /prompt; used to subscribe.
        timeout:   Hard cap on total wait time (seconds).

    Yields:
        Progress strings, then a single terminal dict.
    """
    import websockets

    from src.utils.comfyui_client import get_client

    client = get_client()
    parsed = urlparse(client.base_url)
    ws_scheme = "wss" if parsed.scheme == "https" else "ws"
    ws_url = f"{ws_scheme}://{parsed.netloc}/ws?clientId={client_id}"

    headers: list[tuple[str, str]] = []
    if client.api_key:
        headers.append(("Authorization", f"Bearer {client.api_key}"))

    # If the job already completed before we got here (common in batch flows),
    # short-circuit without opening a socket.
    pre = _check_history(client, prompt_id)
    if pre is not None:
        yield pre
        return

    last_progress_pct: int = -1
    last_emit_loop_t: float = 0.0
    elapsed: float = 0.0
    RECV_TIMEOUT = 5.0  # seconds — also drives periodic history fallback check

    try:
        connect_kwargs: dict = {"max_size": None}
        if headers:
            # websockets >= 12 uses additional_headers
            connect_kwargs["additional_headers"] = headers

        async with websockets.connect(ws_url, **connect_kwargs) as ws:
            while elapsed < timeout:
                try:
                    raw = await asyncio.wait_for(ws.recv(), timeout=RECV_TIMEOUT)
                except asyncio.TimeoutError:
                    # No event for RECV_TIMEOUT — re-check history in case we
                    # missed the completion event (e.g. server restart, or the
                    # prompt finished between our pre-check and ws.connect).
                    elapsed += RECV_TIMEOUT
                    fallback = _check_history(client, prompt_id)
                    if fallback is not None:
                        yield fallback
                        return
                    continue

                if isinstance(raw, (bytes, bytearray)):
                    # Binary preview frames — not used for progress.
                    continue

                try:
                    msg = json.loads(raw)
                except json.JSONDecodeError:
                    continue

                msg_type = msg.get("type")
                data = msg.get("data", {}) or {}
                msg_prompt_id = data.get("prompt_id")

                # Filter to our prompt where the message carries one.
                if msg_prompt_id and msg_prompt_id != prompt_id:
                    continue

                if msg_type == "status":
                    qrem = (
                        data.get("status", {})
                        .get("exec_info", {})
                        .get("queue_remaining")
                    )
                    if qrem is not None and qrem > 0:
                        yield f"⏳ Queue: {qrem} job(s) ahead"

                elif msg_type == "execution_start":
                    yield "▶ Execution started"
                    last_progress_pct = -1

                elif msg_type == "execution_cached":
                    cached = data.get("nodes", []) or []
                    if cached:
                        yield f"💾 {len(cached)} node(s) served from cache"

                elif msg_type == "executing":
                    node = data.get("node")
                    if node is None:
                        # null node = prompt finished (older protocol); confirm via history
                        fallback = _check_history(client, prompt_id)
                        if fallback is not None:
                            yield fallback
                            return
                    else:
                        last_progress_pct = -1
                        yield f"🎨 Running node {node}"

                elif msg_type == "progress":
                    value = int(data.get("value", 0) or 0)
                    max_v = int(data.get("max", 0) or 0)
                    node = data.get("node")
                    if max_v > 0:
                        pct = int(value / max_v * 100)
                        loop_t = asyncio.get_event_loop().time()
                        # Throttle: emit on first/last step, ≥10% jump, or ≥1s elapsed.
                        is_endpoint = value <= 1 or value >= max_v
                        big_jump = pct - last_progress_pct >= 10
                        time_due = loop_t - last_emit_loop_t >= 1.0
                        if is_endpoint or big_jump or time_due:
                            node_label = f" — node {node}" if node else ""
                            yield f"🎨 {_bar(value, max_v)}{node_label}"
                            last_progress_pct = pct
                            last_emit_loop_t = loop_t

                elif msg_type == "execution_success":
                    raw_hist = None
                    try:
                        raw_hist = client.get(f"/history/{prompt_id}")
                    except Exception as exc:
                        yield {"error": f"Could not fetch history: {exc}"}
                        return
                    from src.tools.comfyui import _strip_history
                    yield {"history": _strip_history(raw_hist)}
                    return

                elif msg_type == "execution_error":
                    err_msg = data.get("exception_message", "Unknown error")
                    node_type = data.get("node_type", "?")
                    node_id = data.get("node_id", "?")
                    yield f"❌ Error in {node_type} (node {node_id}): {err_msg}"
                    yield {
                        "error": "ComfyUI execution failed",
                        "details": {
                            "node_id": node_id,
                            "node_type": node_type,
                            "exception_type": data.get("exception_type", ""),
                            "exception_message": err_msg,
                            "traceback": data.get("traceback", []),
                        },
                    }
                    return

                elif msg_type == "execution_interrupted":
                    yield "🛑 Execution interrupted"
                    yield {"error": "Execution interrupted"}
                    return

            # Hard timeout
            yield {"error": f"WebSocket timeout after {timeout:.0f}s"}
            return

    except Exception as exc:
        logger.error("comfyui_progress: WebSocket failed for prompt_id=%s: %s", prompt_id, exc)
        yield {"error": f"WebSocket connection failed: {exc}"}
