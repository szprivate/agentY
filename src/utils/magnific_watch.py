"""Background watcher for asynchronous Magnific (MCP) creations.

Magnific generation is async: ``video_generate`` / ``image_generate`` / upscale
tools return immediately with ``{creations:[{identifier, status:"queued", …}]}``
and the render finishes minutes later on Magnific's servers. This module watches
a queued creation to completion **outside the agent turn** and — via an injected
completion handler — downloads the finished asset, then raises a structured
notification on :mod:`src.utils.notify_bus` so the sidebar drops it onto the
ComfyUI canvas and pops a note.

Wiring:

* :func:`register` is called (deterministically, no LLM reliance) from an
  ``AfterToolCallEvent`` hook whenever a Magnific tool result carries a queued
  creation identifier. It dedupes and starts one daemon poller thread per id.
* The server injects the download/stage step via :func:`set_completion_handler`
  at startup (it owns the ComfyUI-input staging + gallery helpers); this module
  stays free of any server/import cycle.

Polling uses ``creations_wait`` — the MCP-sanctioned way to get the final asset
URL — through :func:`src.tools.mcp_tools.call_mcp_tool` (direct client call, no
agent). Response shapes vary by tool, so parsing is deliberately defensive and
the raw body is logged for a first live run.
"""
from __future__ import annotations

import logging
import re
import threading
import time
from typing import Callable, Optional

logger = logging.getLogger("agentY.magnific_watch")

# ── tunables ─────────────────────────────────────────────────────────────────
_MAX_WATCH_SECONDS = 30 * 60      # give up after 30 min (avoid a zombie thread)
_WAIT_TIMEOUT_S = 240             # per creations_wait long-poll read timeout
_POLL_INTERVAL_S = 6              # gap between polls when the wait returns early

_DONE = {"done", "completed", "complete", "succeeded", "success", "finished",
         "ready", "generated"}
_FAIL = {"failed", "error", "errored", "cancelled", "canceled", "rejected"}

_VIDEO_EXT = (".mp4", ".webm", ".mov", ".m4v", ".mkv", ".gif")
_IMAGE_EXT = (".png", ".jpg", ".jpeg", ".webp", ".avif", ".bmp", ".tif", ".tiff")
_MEDIA_EXT = _VIDEO_EXT + _IMAGE_EXT

# A URL that points at the *asset* (not the app's creation-viewer page). The
# webUrl Magnific returns for sharing is www.magnific.com/app/creation/<id> — no
# media extension — so the extension test excludes it automatically.
_URL_RE = re.compile(r"https?://[^\s\"'<>)\]]+", re.IGNORECASE)

# ── injected completion handler ──────────────────────────────────────────────
# Signature: (identifier, asset_url, kind, meta) -> Optional[dict]
#   returns {"kind","path","filename","name","node_candidates"} once downloaded.
_completion_handler: Optional[Callable] = None

_LOCK = threading.Lock()
_active: set = set()   # identifiers currently being watched (dedupe)


def set_completion_handler(fn: Callable) -> None:
    """Inject the download+stage step (server-side; see agentY_server)."""
    global _completion_handler
    _completion_handler = fn


# ── public: register a creation to watch ─────────────────────────────────────

def register(identifier: str, *, tool: str = "", web_url: str = "",
             thread_id: str = "") -> bool:
    """Start watching *identifier* (idempotent). Returns True if a new watcher was
    started, False if one was already running for this id or the id is blank."""
    identifier = (identifier or "").strip()
    if not identifier:
        return False
    with _LOCK:
        if identifier in _active:
            return False
        _active.add(identifier)
    meta = {"tool": tool, "web_url": web_url, "thread_id": thread_id}
    threading.Thread(
        target=_watch, args=(identifier, meta),
        name=f"magnific-watch-{identifier[:8]}", daemon=True,
    ).start()
    logger.info("magnific_watch: watching creation %s (tool=%s)", identifier, tool or "?")
    return True


def register_from_result(result: object, *, tool: str = "") -> int:
    """Scan a Magnific tool result for queued creation ids and register each.

    Accepts the parsed JSON (dict/list) or a Strands ToolResult-ish dict; returns
    the number of new watchers started. Only creations in a non-terminal state
    (queued/processing/…) with an identifier are watched.
    """
    started = 0
    for ident, web_url in _iter_pending_creations(result):
        if register(ident, tool=tool, web_url=web_url):
            started += 1
    return started


# ── watcher thread ───────────────────────────────────────────────────────────

def _watch(identifier: str, meta: dict) -> None:
    try:
        from src.tools.mcp_tools import call_mcp_tool
    except Exception as exc:  # noqa: BLE001
        logger.warning("magnific_watch: cannot import call_mcp_tool: %s", exc)
        _discard(identifier)
        return

    deadline = time.time() + _MAX_WATCH_SECONDS
    first_raw_logged = False
    try:
        while time.time() < deadline:
            res = call_mcp_tool("magnific", "creations_wait",
                                {"identifier": identifier}, timeout_s=_WAIT_TIMEOUT_S)
            if not res.get("ok") and res.get("error"):
                logger.info("magnific_watch[%s]: wait error: %s", identifier, res["error"])
                time.sleep(_POLL_INTERVAL_S)
                continue
            body = res.get("json")
            if not first_raw_logged:
                logger.info("magnific_watch[%s]: first wait body: %s",
                            identifier, (res.get("text") or "")[:600])
                first_raw_logged = True

            status = _extract_status(body)
            asset_url = _extract_asset_url(body)

            if status in _FAIL:
                _emit_fail(identifier, meta, status)
                return
            if asset_url and (status in _DONE or status is None):
                # A media asset URL is the real completion signal; some tools omit
                # an explicit terminal status but only surface the asset when done.
                _emit_done(identifier, meta, asset_url)
                return
            if status in _DONE and not asset_url:
                logger.info("magnific_watch[%s]: done but no asset URL found in body",
                            identifier)
                _emit_fail(identifier, meta, "done-no-asset")
                return
            time.sleep(_POLL_INTERVAL_S)
        _emit_fail(identifier, meta, "timeout")
    except Exception as exc:  # noqa: BLE001
        logger.error("magnific_watch[%s]: watcher crashed: %s", identifier, exc, exc_info=True)
        _emit_fail(identifier, meta, "watcher-error")
    finally:
        _discard(identifier)


def _emit_done(identifier: str, meta: dict, asset_url: str) -> None:
    from src.utils import notify_bus

    kind = _kind_from_url(asset_url)
    output = None
    if _completion_handler is not None:
        try:
            output = _completion_handler(identifier, asset_url, kind, meta)
        except Exception as exc:  # noqa: BLE001
            logger.error("magnific_watch[%s]: completion handler failed: %s",
                         identifier, exc, exc_info=True)
            output = None

    if output and output.get("path"):
        name = output.get("name") or "Magnific result"
        notify_bus.emit({
            "kind": "media",
            "output": output,
            "toast": {
                "title": "Magnific result ready",
                "body": f"Added {output.get('kind', kind)} to the canvas: {name}",
                "url": meta.get("web_url", ""),
                "level": "success",
            },
        })
        logger.info("magnific_watch[%s]: dropped %s onto canvas", identifier,
                    output.get("path"))
    else:
        # Couldn't download/stage — still tell the user, with the viewer link.
        notify_bus.emit({
            "kind": "toast",
            "toast": {
                "title": "Magnific result ready",
                "body": "Your generation finished. Open it in the Magnific viewer.",
                "url": meta.get("web_url", asset_url),
                "level": "success",
            },
        })


def _emit_fail(identifier: str, meta: dict, reason: str) -> None:
    from src.utils import notify_bus

    human = {
        "timeout": "still not ready after 30 min — check the Magnific viewer.",
        "done-no-asset": "finished but no downloadable asset was returned.",
    }.get(reason, f"could not be completed ({reason}).")
    notify_bus.emit({
        "kind": "error",
        "toast": {
            "title": "Magnific generation",
            "body": f"A generation {human}",
            "url": meta.get("web_url", ""),
            "level": "error",
        },
    })
    logger.info("magnific_watch[%s]: gave up (%s)", identifier, reason)


def _discard(identifier: str) -> None:
    with _LOCK:
        _active.discard(identifier)


# ── defensive parsing ────────────────────────────────────────────────────────

def _iter_pending_creations(result: object):
    """Yield ``(identifier, web_url)`` for each non-terminal creation in *result*."""
    data = result
    if isinstance(result, dict) and "content" in result and "creations" not in result:
        # A Strands ToolResult wrapper — pull the JSON out of its content blocks.
        import json as _json
        for block in (result.get("content") or []):
            if isinstance(block, dict) and "json" in block:
                data = block["json"]
                break
            if isinstance(block, dict) and "text" in block:
                t = str(block["text"]).strip()
                if t[:1] in ("{", "["):
                    try:
                        data = _json.loads(t)
                        break
                    except Exception:  # noqa: BLE001
                        continue
    creations = None
    if isinstance(data, dict):
        creations = data.get("creations")
        if creations is None and data.get("identifier"):
            creations = [data]
    elif isinstance(data, list):
        creations = data
    for c in (creations or []):
        if not isinstance(c, dict):
            continue
        ident = c.get("identifier") or c.get("id")
        status = str(c.get("status") or c.get("state") or "").lower()
        if ident and status not in _DONE and status not in _FAIL:
            yield str(ident), str(c.get("webUrl") or c.get("web_url") or "")


def _first_creation(body: object) -> Optional[dict]:
    if isinstance(body, dict):
        creations = body.get("creations")
        if isinstance(creations, list) and creations:
            return creations[0] if isinstance(creations[0], dict) else None
        return body
    if isinstance(body, list) and body and isinstance(body[0], dict):
        return body[0]
    return None


def _extract_status(body: object) -> Optional[str]:
    c = _first_creation(body)
    if not isinstance(c, dict):
        return None
    s = c.get("status") or c.get("state")
    return str(s).lower() if s else None


def _extract_asset_url(body: object) -> Optional[str]:
    """Find the first media-asset URL anywhere in *body* (deep walk).

    Prefers an explicit asset field, then any http(s) URL whose path ends in a
    known media extension. The share/viewer webUrl is excluded (no extension).
    """
    if body is None:
        return None
    preferred_keys = ("assetUrl", "asset_url", "outputUrl", "output_url",
                      "resultUrl", "result_url", "downloadUrl", "download_url",
                      "videoUrl", "imageUrl", "fileUrl", "url", "video", "image",
                      "output", "asset", "result")

    def _is_media(u: str) -> bool:
        if not isinstance(u, str) or not u.lower().startswith("http"):
            return False
        path = u.split("?", 1)[0].lower()
        return path.endswith(_MEDIA_EXT)

    # Pass 1: preferred keys carrying a media URL.
    def _walk_pref(node):
        if isinstance(node, dict):
            for k in preferred_keys:
                v = node.get(k)
                if _is_media(v):
                    return v
                if isinstance(v, dict):
                    for cand in _URL_RE.findall(str(v)):
                        if _is_media(cand):
                            return cand
            for v in node.values():
                r = _walk_pref(v)
                if r:
                    return r
        elif isinstance(node, list):
            for v in node:
                r = _walk_pref(v)
                if r:
                    return r
        return None

    hit = _walk_pref(body)
    if hit:
        return hit

    # Pass 2: any media URL found by scanning the serialized body.
    import json as _json
    try:
        blob = _json.dumps(body)
    except Exception:  # noqa: BLE001
        blob = str(body)
    for cand in _URL_RE.findall(blob):
        if _is_media(cand):
            return cand
    return None


def _kind_from_url(url: str) -> str:
    path = (url or "").split("?", 1)[0].lower()
    if path.endswith(_VIDEO_EXT):
        return "video"
    return "image"
