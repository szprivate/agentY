"""
Slack Events API server for agentY.

Runs a lightweight Flask app that:
  1. Responds to Slack URL-verification challenges.
  2. Receives event callbacks (DM messages) and feeds them to the agent.
  3. Replies via Slack using the agent's response.

An ngrok tunnel is started automatically so a public Request URL is
available for Slack Event Subscriptions configuration.

Environment variables (loaded from .env by main.py):
    SLACK_BOT_TOKEN   - Bot User OAuth Token (xoxb-...)
    SLACK_MEMBER_ID   - Default Slack member ID for DMs
    NGROK_AUTH_TOKEN   - (Optional) ngrok auth token for persistent tunnels
    SLACK_SIGNING_SECRET - (Optional) Slack signing secret for request verification
"""

import hashlib
import hmac
import json
import logging
import os
import re
import sys
import asyncio
import base64
import threading
import time
from typing import Optional

import requests as http_requests
from flask import Flask, Response, jsonify, request

# ---------------------------------------------------------------------------
# Set up module logger so messages actually print to console
# ---------------------------------------------------------------------------
logger = logging.getLogger("agentY.slack_server")
logger.setLevel(logging.DEBUG)
if not logger.handlers:
    _handler = logging.StreamHandler(sys.stdout)
    _handler.setLevel(logging.DEBUG)
    _handler.setFormatter(logging.Formatter(
        "[%(asctime)s] [SlackServer] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    ))
    logger.addHandler(_handler)

# ---------------------------------------------------------------------------
# Module-level references (set by start_slack_server)
# ---------------------------------------------------------------------------
_agent_ref = None          # Will hold the Strands Agent instance
_flask_app = Flask(__name__)
_ngrok_url: Optional[str] = None
_bot_user_id: Optional[str] = None  # filled on first boot to ignore own msgs

SLACK_SERVER_PORT = int(os.environ.get("SLACK_SERVER_PORT", "3000"))


# ---------------------------------------------------------------------------
# Slack request verification (optional but recommended)
# ---------------------------------------------------------------------------

def _verify_slack_signature(body: bytes, timestamp: str, signature: str) -> bool:
    """Verify the request actually came from Slack using the signing secret."""
    secret = os.environ.get("SLACK_SIGNING_SECRET", "")
    if not secret:
        return True  # skip verification when secret is not configured
    basestring = f"v0:{timestamp}:{body.decode('utf-8')}"
    computed = "v0=" + hmac.new(
        secret.encode(), basestring.encode(), hashlib.sha256
    ).hexdigest()
    return hmac.compare_digest(computed, signature)


# ---------------------------------------------------------------------------
# Resolve bot's own user-id so we can ignore our own messages
# ---------------------------------------------------------------------------

def _resolve_bot_user_id() -> Optional[str]:
    global _bot_user_id
    if _bot_user_id:
        return _bot_user_id
    try:
        from slack_sdk import WebClient
        client = WebClient(token=os.environ.get("SLACK_BOT_TOKEN", ""))
        resp = client.auth_test()
        _bot_user_id = resp.get("user_id")
        logger.info("Resolved bot user ID: %s", _bot_user_id)
        return _bot_user_id
    except Exception as exc:
        logger.warning("Could not resolve bot user ID: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Duplicate event guard (Slack can retry)
# ---------------------------------------------------------------------------

_seen_event_ids: set = set()
_SEEN_MAX = 500

# Matches Werkzeug / Flask access-log lines that occasionally leak into the
# Strands SDK streamed output via captured stdout, e.g.:
#   127.0.0.1 - - [29/Mar/2026 23:12:15] "POST /slack/events HTTP/1.1" 200 -
_ACCESS_LOG_RE = re.compile(
    r"^\d{1,3}(?:\.\d{1,3}){3}\s+-\s+-\s+\[.*?\]\s+\".*?\"\s+\d{3}.*$",
    re.MULTILINE,
)


def _is_duplicate(event_id: str) -> bool:
    if event_id in _seen_event_ids:
        return True
    _seen_event_ids.add(event_id)
    # prevent unbounded growth
    if len(_seen_event_ids) > _SEEN_MAX:
        _seen_event_ids.clear()
    return False


# ---------------------------------------------------------------------------
# Agent invocation (runs in its own thread so Flask can respond quickly)
# ---------------------------------------------------------------------------

# How often (seconds) to push partial text to Slack via chat_update
_STREAM_UPDATE_INTERVAL = 2.0

# Image MIME types we can handle as vision input
_IMAGE_EXTENSIONS = {
    "png": "png",
    "jpg": "jpeg",
    "jpeg": "jpeg",
    "gif": "gif",
    "webp": "webp",
}

# Video MIME types the vision model can process
_VIDEO_EXTENSIONS = {
    "mp4": "mp4",
    "mov": "mov",
    "mpeg": "mpeg",
    "mpg": "mpg",
    "webm": "webm",
    "mkv": "mkv",
    "flv": "flv",
    "wmv": "wmv",
    "3gp": "three_gp",
}


def _download_slack_file(url: str) -> Optional[bytes]:
    """Download a file from Slack using the bot token for auth."""
    token = os.environ.get("SLACK_BOT_TOKEN", "")
    try:
        resp = http_requests.get(
            url,
            headers={"Authorization": f"Bearer {token}"},
            timeout=30,
        )
        resp.raise_for_status()
        return resp.content
    except Exception as exc:
        logger.warning("Failed to download Slack file %s: %s", url, exc)
        return None


def _slack_downloads_dir() -> str:
    """Return (and create) the directory for files downloaded from Slack."""
    d = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "slack_downloads")
    os.makedirs(d, exist_ok=True)
    return d


def _build_content_blocks(text: str, files: list) -> list:
    """Build Strands content blocks from message text + Slack file attachments.

    Files are saved to ``<project>/slack_downloads/`` so the agent can reference
    them by path when submitting to ComfyUI or other tools.

    Returns a list of ContentBlock dicts suitable for ``agent.stream_async()``.
    If there are no processable media files, returns the plain text string.
    """
    blocks: list = []
    media_count = 0
    saved_paths: list[str] = []

    for f in files:
        mimetype = f.get("mimetype", "")
        filetype = f.get("filetype", "").lower()
        name = f.get("name", "unknown")

        # --- Try image first ------------------------------------------- #
        img_fmt = _IMAGE_EXTENSIONS.get(filetype)
        if not img_fmt and mimetype.startswith("image/"):
            sub = mimetype.split("/")[-1].lower()
            img_fmt = _IMAGE_EXTENSIONS.get(sub)

        if img_fmt:
            url = f.get("url_private_download") or f.get("url_private")
            if not url:
                logger.debug("No download URL for file %s", name)
                continue
            data = _download_slack_file(url)
            if not data:
                continue

            # Save to disk so agent tools can reference the file
            save_path = os.path.join(_slack_downloads_dir(), name)
            with open(save_path, "wb") as fp:
                fp.write(data)
            saved_paths.append(save_path)

            blocks.append({
                "image": {
                    "format": img_fmt,
                    "source": {"bytes": data},
                }
            })
            media_count += 1
            logger.info("Downloaded image '%s' (%s, %d bytes) -> %s", name, img_fmt, len(data), save_path)
            continue

        # --- Try video ------------------------------------------------- #
        vid_fmt = _VIDEO_EXTENSIONS.get(filetype)
        if not vid_fmt and mimetype.startswith("video/"):
            sub = mimetype.split("/")[-1].lower()
            vid_fmt = _VIDEO_EXTENSIONS.get(sub)

        if vid_fmt:
            url = f.get("url_private_download") or f.get("url_private")
            if not url:
                logger.debug("No download URL for file %s", name)
                continue
            data = _download_slack_file(url)
            if not data:
                continue

            # Save to disk
            save_path = os.path.join(_slack_downloads_dir(), name)
            with open(save_path, "wb") as fp:
                fp.write(data)
            saved_paths.append(save_path)

            blocks.append({
                "video": {
                    "format": vid_fmt,
                    "source": {"bytes": data},
                }
            })
            media_count += 1
            logger.info("Downloaded video '%s' (%s, %d bytes) -> %s", name, vid_fmt, len(data), save_path)
            continue

        logger.debug("Skipping unsupported file: %s (type=%s)", name, filetype)

    if media_count == 0:
        # No media - just return text directly (agent accepts str)
        return text  # type: ignore[return-value]

    # Build text that tells the agent where the files are on disk
    paths_info = "\n".join(f"  - {p}" for p in saved_paths)
    file_context = f"\n\n[Attached files saved to disk:\n{paths_info}\nUse these paths when tools need a file_path.]"

    if text:
        blocks.insert(0, {"text": text + file_context})
    else:
        blocks.insert(0, {"text": "The user sent the following file(s)." + file_context})

    return blocks


def _handle_message_async(content, channel: str, thread_ts: str, user: str):
    """Run the agent on *content*, streaming partial results to Slack.

    ``content`` can be a plain str or a list of Strands ContentBlock dicts
    (when the user sent images).

    1. Immediately posts a "Thinking..." placeholder message.
    2. Streams tokens from the agent via ``stream_async()``.
    3. Periodically updates the placeholder with accumulated text.
    4. Finalises the message when the agent finishes.
    """
    try:
        if _agent_ref is None:
            logger.error("Agent reference is not set; cannot process message.")
            return

        from slack_sdk import WebClient
        from src.tools.slack_tools import set_slack_channel_context, clear_slack_channel_context
        client = WebClient(token=os.environ.get("SLACK_BOT_TOKEN", ""))

        # Set channel context so agent tools send files to this channel
        set_slack_channel_context(channel, thread_ts)

        # -- Step 1: Post a placeholder --------------------------------- #
        log_preview = content if isinstance(content, str) else f"[multimodal: {len(content)} blocks]"
        logger.info("Processing Slack message from %s: %s", user, str(log_preview)[:120])

        placeholder = client.chat_postMessage(
            channel=channel,
            text=":hourglass_flowing_sand: Thinking...",
            thread_ts=thread_ts,
        )
        msg_ts = placeholder["ts"]
        logger.info("Posted placeholder (ts=%s) for user %s", msg_ts, user)

        # -- Step 2: Stream tokens from the agent ----------------------- #
        accumulated = []
        last_update_time = time.monotonic()

        def _push_update(final: bool = False):
            """Update the Slack message with accumulated text so far."""
            nonlocal last_update_time
            raw = "".join(accumulated)
            # Strip HTTP access-log lines that the Strands SDK occasionally
            # captures from stdout and injects into the streamed text.
            current_text = _ACCESS_LOG_RE.sub("", raw).strip()
            if not current_text:
                return
            suffix = "" if final else " :writing_hand:"
            try:
                client.chat_update(
                    channel=channel,
                    ts=msg_ts,
                    text=current_text + suffix,
                )
                last_update_time = time.monotonic()
            except Exception as update_exc:
                logger.warning("chat_update failed: %s", update_exc)

        async def _stream():
            nonlocal last_update_time
            last_was_text = False
            async for event in _agent_ref.stream_async(content):
                if "data" in event:
                    chunk = event["data"]
                    if chunk:
                        accumulated.append(chunk)
                        last_was_text = True
                        # Push an update if enough time has elapsed
                        if (time.monotonic() - last_update_time) >= _STREAM_UPDATE_INTERVAL:
                            _push_update()
                else:
                    last_was_text = False

        # Run the async stream in a new event loop (we're in a thread)
        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(_stream())
        finally:
            loop.close()

        # -- Step 3: Append token-usage summary then finalise ----------- #
        try:
            usage = _agent_ref.event_loop_metrics.accumulated_usage
            in_tok = usage.get("inputTokens", 0)
            out_tok = usage.get("outputTokens", 0)
            cache_read = usage.get("cacheReadInputTokens", 0)
            cache_write = usage.get("cacheWriteInputTokens", 0)
            logger.info(
                "Token usage — in: %d, out: %d, total: %d, cache_read: %d, cache_write: %d",
                in_tok, out_tok, in_tok + out_tok, cache_read, cache_write,
            )
            parts = [f"{in_tok:,} in", f"{out_tok:,} out"]
            if cache_read:
                parts.append(f"{cache_read:,} cache hit")
            if cache_write:
                parts.append(f"{cache_write:,} cache write")
            accumulated.append(f"\n\n_🪙 {' / '.join(parts)}_")
        except Exception as _tok_exc:
            logger.debug("Could not read token usage: %s", _tok_exc)

        _push_update(final=True)
        clear_slack_channel_context()
        logger.info("Streamed reply to %s in channel %s", user, channel)

    except Exception as exc:
        logger.error("Error handling Slack message: %s", exc, exc_info=True)
        try:
            from src.tools.slack_tools import clear_slack_channel_context
            clear_slack_channel_context()
        except Exception:
            pass
        # Attempt to notify the user about the error
        try:
            from slack_sdk import WebClient
            client = WebClient(token=os.environ.get("SLACK_BOT_TOKEN", ""))
            client.chat_postMessage(
                channel=channel,
                text=f"Sorry, I encountered an error processing your message: {exc}",
                thread_ts=thread_ts,
            )
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Flask routes
# ---------------------------------------------------------------------------

@_flask_app.route("/slack/events", methods=["POST"])
def slack_events():
    """Handle incoming Slack Events API requests."""
    body = request.get_data()
    logger.debug("Incoming request to /slack/events  (%d bytes)", len(body))

    # -- Optional signature verification --------------------------------- #
    ts = request.headers.get("X-Slack-Request-Timestamp", "")
    sig = request.headers.get("X-Slack-Signature", "")
    if not _verify_slack_signature(body, ts, sig):
        logger.warning("Slack signature verification FAILED")
        return Response("Unauthorized", status=401)

    payload = request.get_json(force=True)
    payload_type = payload.get("type", "")
    logger.debug("Payload type: %s", payload_type)

    # -- URL verification challenge -------------------------------------- #
    if payload_type == "url_verification":
        challenge = payload.get("challenge", "")
        logger.info("Responding to Slack URL verification challenge")
        return jsonify({"challenge": challenge})

    # -- Event callback -------------------------------------------------- #
    if payload_type == "event_callback":
        event_id = payload.get("event_id", "")
        if _is_duplicate(event_id):
            logger.debug("Duplicate event %s - skipping", event_id)
            return Response("OK", status=200)

        event = payload.get("event", {})
        event_type = event.get("type", "")
        channel_type = event.get("channel_type", "")
        logger.info(
            "Event: type=%s  channel_type=%s  user=%s  event_id=%s",
            event_type, channel_type, event.get("user", "?"), event_id,
        )

        # We care about messages - both DMs (im) and channels where bot is mentioned
        if event_type == "message":
            # Ignore bot's own messages, message_changed/deleted subtypes, etc.
            subtype = event.get("subtype")
            bot_id = event.get("bot_id")
            user = event.get("user", "")

            # Allow file_share (image uploads) through; block other subtypes
            _ALLOWED_SUBTYPES = {None, "file_share"}
            if subtype not in _ALLOWED_SUBTYPES:
                logger.debug("Ignoring message with subtype=%s", subtype)
                return Response("OK", status=200)
            if bot_id:
                logger.debug("Ignoring message from bot_id=%s", bot_id)
                return Response("OK", status=200)

            # Resolve our own bot user id
            bot_uid = _resolve_bot_user_id()
            if user == bot_uid:
                logger.debug("Ignoring own message from bot user %s", bot_uid)
                return Response("OK", status=200)

            text = event.get("text", "").strip()
            files = event.get("files", [])

            if not text and not files:
                logger.debug("Ignoring empty message")
                return Response("OK", status=200)

            channel = event.get("channel", "")
            thread_ts = event.get("ts", "")  # reply in thread under their msg

            # Build content: plain text or multimodal blocks with images
            if files:
                content = _build_content_blocks(text, files)
            else:
                content = text

            logger.info(
                ">> Processing message from user=%s channel=%s: %s",
                user, channel,
                text[:120] if text else f"[{len(files)} file(s)]",
            )

            # Process asynchronously so we respond to Slack within 3 s
            t = threading.Thread(
                target=_handle_message_async,
                args=(content, channel, thread_ts, user),
                daemon=True,
            )
            t.start()
        else:
            logger.debug("Unhandled event type: %s", event_type)

    return Response("OK", status=200)


@_flask_app.route("/health", methods=["GET"])
def health():
    """Simple health-check endpoint."""
    return jsonify({"status": "ok", "ngrok_url": _ngrok_url})


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def start_slack_server(agent) -> Optional[str]:
    """Start the Flask server + ngrok tunnel in background threads.

    Args:
        agent: The Strands Agent instance that will process messages.

    Returns:
        The public ngrok URL (e.g. ``https://xxxx.ngrok-free.app``) or
        ``None`` if the server could not be started.
    """
    global _agent_ref, _ngrok_url
    _agent_ref = agent

    # -- Start ngrok tunnel --------------------------------------------- #
    try:
        from pyngrok import ngrok, conf

        auth_token = os.environ.get("NGROK_AUTH_TOKEN", "")
        if auth_token:
            conf.get_default().auth_token = auth_token

        # pyngrok >= 7.x uses scheme="https"; older used bind_tls=True
        try:
            tunnel = ngrok.connect(
                addr=str(SLACK_SERVER_PORT),
                proto="http",
                bind_tls=True,
            )
        except TypeError:
            # Fallback for newer pyngrok API
            tunnel = ngrok.connect(
                addr=str(SLACK_SERVER_PORT),
                schemes=["https"],
            )

        _ngrok_url = tunnel.public_url
        # Ensure https
        if _ngrok_url and _ngrok_url.startswith("http://"):
            _ngrok_url = _ngrok_url.replace("http://", "https://", 1)

        logger.info("ngrok tunnel established: %s", _ngrok_url)
    except Exception as exc:
        logger.error("Failed to start ngrok tunnel: %s", exc, exc_info=True)
        print(f"[agentY] ERROR: Could not start ngrok tunnel: {exc}")
        print("[agentY] Install ngrok CLI and/or set NGROK_AUTH_TOKEN in .env")
        return None

    # -- Start Flask in a daemon thread --------------------------------- #
    def _run_flask():
        logger.info("Starting Flask server on 0.0.0.0:%d", SLACK_SERVER_PORT)
        _flask_app.run(
            host="0.0.0.0",
            port=SLACK_SERVER_PORT,
            debug=False,
            use_reloader=False,
        )

    flask_thread = threading.Thread(target=_run_flask, daemon=True)
    flask_thread.start()

    # Give Flask a moment to bind the port
    time.sleep(1)

    events_url = f"{_ngrok_url}/slack/events"
    print()
    print("=" * 60)
    print("  SLACK EVENT SUBSCRIPTIONS - Request URL")
    print("=" * 60)
    print(f"  {events_url}")
    print("=" * 60)
    print()
    print("  Paste this URL into your Slack app settings:")
    print("  https://api.slack.com/apps  ->  Event Subscriptions")
    print("  -> Enable Events -> Request URL")
    print()
    print("  Subscribe to bot events:")
    print("    - message.im  (messages in DMs)")
    print()
    print("  Also ensure these Bot Token Scopes are granted:")
    print("    - chat:write, im:history, im:read, im:write")
    print("    - files:read, files:write (for image/video tools)")
    print()

    return events_url


def get_ngrok_url() -> Optional[str]:
    """Return the current ngrok public URL, or None."""
    return _ngrok_url
