"""
agentY bridge HTTP server — runs on localhost:5000.

Exposes two endpoints consumed by the ComfyUI frontend extension:

    GET  /agentY/pending_previews
        Returns a JSON array of pending preview-job descriptors.
        Each item is either:
          { job_id, label, origin_pos }          — inject a PreviewImage node
          { clear: true, job_id }                — remove the injected node

    POST /agentY/review
        Body: { node_id, message, image_paths, source }
        Dispatches the message + image(s) to the agentY agent directly.

Public helpers (called from agent tools):

    add_preview_job(job_id, label, origin_pos)  — enqueue a preview
    clear_preview_job(job_id)                   — signal the frontend to remove it
"""

import asyncio
import io
import logging
import os
import threading

logger = logging.getLogger("agentY.server")

# ── In-memory state ───────────────────────────────────────────────────────────

_lock = threading.Lock()
_pending_previews: dict[str, dict] = {}
_node_responses: dict[str, str] = {}   # node_id (str) -> accumulated agent text
_agent_ref = None


# ── Public helpers ─────────────────────────────────────────────────────────────

def add_preview_job(job_id: str, label: str, origin_pos: list | None = None) -> None:
    """Enqueue a preview job so the frontend injects a PreviewImage node."""
    with _lock:
        _pending_previews[job_id] = {
            "job_id": job_id,
            "label": label,
            "origin_pos": origin_pos or [100, 100],
        }
    logger.info("Preview job queued: %s (%s)", job_id, label)


def clear_preview_job(job_id: str) -> None:
    """Signal the frontend to remove the injected PreviewImage node for *job_id*."""
    with _lock:
        _pending_previews[job_id] = {"job_id": job_id, "clear": True}
    logger.info("Preview job marked for clearance: %s", job_id)


# ── Content builder ───────────────────────────────────────────────────────────

def _build_content(message: str, image_paths: list[str]) -> list | str:
    """Build a Strands-compatible content list from text + image file paths.

    Images are always downsized to satisfy Claude's 5 MB / 1568 px constraints.
    """
    if not image_paths:
        return message or "(no message)"

    from src.tools.image_handling import _downsize, _detect_format

    blocks: list = []

    for path in image_paths:
        try:
            from pathlib import Path as _Path
            raw = _Path(path).read_bytes()
            img_fmt = _detect_format(path) or "png"
            image_bytes = _downsize(raw, img_fmt)
            blocks.append({
                "image": {
                    "format": img_fmt,
                    "source": {"bytes": image_bytes},
                }
            })
            logger.info("Loaded review image: %s (%d bytes, after downsize)", path, len(image_bytes))
        except Exception as exc:
            logger.warning("Could not load image %s: %s", path, exc)

    if not blocks:
        return message or "(no message)"

    # Include the on-disk paths so the Researcher can reference them in the
    # BrainBriefing (ComfyUI loaders need the actual file path, not bytes).
    path_lines = "\n".join(f"  - {p}  [image, use this path for ComfyUI input]" for p in image_paths if os.path.exists(p))
    paths_info = f"\n\nAttached image file paths (use these for ComfyUI):\n{path_lines}" if path_lines else ""

    text = (message if message else "The user sent an image from ComfyUI for review.") + paths_info
    blocks.insert(0, {"text": text})
    return blocks


# ── Agent dispatch ────────────────────────────────────────────────────────────

def _dispatch_to_agent(message: str, image_paths: list[str], node_id: str | None) -> None:
    """Build content and run the agent entirely in a background thread.

    Image loading (PIL) and agent streaming are both off the Flask request
    thread so the HTTP response is returned immediately.
    """
    if _agent_ref is None:
        logger.error("No agent registered — call start_agentY_server(agent=...) first")
        return

    def _run():
        content = _build_content(message, image_paths)

        async def _stream():
            accumulated = []
            async for event in _agent_ref.stream_async(content):
                if "data" in event and event["data"]:
                    accumulated.append(event["data"])
            if node_id and accumulated:
                with _lock:
                    _node_responses[str(node_id)] = "".join(accumulated)
                logger.info(
                    "Stored response for node %s (%d chars)",
                    node_id, len(_node_responses[str(node_id)]),
                )

        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(_stream())
        except Exception as exc:
            logger.error("Agent error during ComfyUI review dispatch: %s", exc, exc_info=True)
        finally:
            loop.close()

    threading.Thread(target=_run, name="agentY-review-dispatch", daemon=True).start()


# ── Flask application ─────────────────────────────────────────────────────────

def _build_app():
    from flask import Flask, jsonify, request

    flask_app = Flask("agentY_bridge")
    flask_app.logger.disabled = True

    @flask_app.after_request
    def _add_cors(response):
        response.headers["Access-Control-Allow-Origin"] = "*"
        response.headers["Access-Control-Allow-Headers"] = "Content-Type"
        response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
        return response

    @flask_app.route("/agentY/health", methods=["GET"])
    def health():
        return jsonify({"status": "ok"})

    @flask_app.route("/agentY/pending_previews", methods=["GET", "OPTIONS"])
    def pending_previews():
        if request.method == "OPTIONS":
            return "", 204
        with _lock:
            jobs = list(_pending_previews.values())
            to_remove = [j["job_id"] for j in jobs if j.get("clear")]
            for jid in to_remove:
                _pending_previews.pop(jid, None)
        return jsonify(jobs)

    @flask_app.route("/agentY/review", methods=["POST", "OPTIONS"])
    def review():
        if request.method == "OPTIONS":
            return "", 204
        payload = request.get_json(silent=True) or {}
        message = payload.get("message", "")
        image_paths = payload.get("image_paths", [])
        node_id = payload.get("node_id", "?")

        logger.info(
            "Review request from node %s — message=%r, images=%d",
            node_id, message[:80] if message else "", len(image_paths),
        )

        _dispatch_to_agent(
            message=message,
            image_paths=image_paths,
            node_id=str(node_id) if node_id not in ("?", None) else None,
        )
        return jsonify({"status": "dispatched"})

    @flask_app.route("/agentY/node_responses", methods=["GET", "OPTIONS"])
    def node_responses():
        if request.method == "OPTIONS":
            return "", 204
        with _lock:
            data = dict(_node_responses)
            _node_responses.clear()
        return jsonify(data)

    return flask_app


# ── Server startup ─────────────────────────────────────────────────────────────

_server_thread: threading.Thread | None = None


def start_agentY_server(agent, host: str = "127.0.0.1", port: int = 5000) -> bool:
    """Start the agentY bridge HTTP server in a background daemon thread.

    Args:
        agent: The Strands Agent / pipeline callable that accepts stream_async().
        host:  Bind address (default 127.0.0.1).
        port:  Port (default 5000).

    Returns True if the server started (or was already running).
    """
    global _server_thread, _agent_ref

    _agent_ref = agent

    if _server_thread is not None and _server_thread.is_alive():
        logger.debug("agentY bridge server already running.")
        return True

    try:
        from flask import Flask  # noqa: F401
    except ImportError:
        logger.error(
            "Flask is not installed. Run: pip install flask\n"
            "The agentY ComfyUI bridge will be unavailable."
        )
        return False

    flask_app = _build_app()

    def _run():
        try:
            from werkzeug.serving import make_server
            srv = make_server(host, port, flask_app, threaded=True)
            logger.info("agentY bridge server ready on http://%s:%d", host, port)
            srv.serve_forever()
        except Exception as exc:
            logger.error("agentY bridge server crashed: %s", exc, exc_info=True)

    _server_thread = threading.Thread(target=_run, name="agentY-bridge-server", daemon=True)
    _server_thread.start()
    logger.info("agentY bridge server started on http://%s:%d", host, port)
    return True
