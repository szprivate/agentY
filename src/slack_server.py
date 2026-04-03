"""
Slack Socket Mode listener for agentY.

Connects to Slack via Socket Mode (WebSocket) instead of a public HTTP
endpoint, so no ngrok tunnel or port-forwarding is required.

Uses ``slack_sdk.socket_mode.SocketModeClient`` (the built-in SDK client)
rather than Slack Bolt, keeping the dependency set minimal.

Environment variables (loaded from .env by main.py):
    SLACK_BOT_TOKEN    - Bot User OAuth Token (xoxb-...)
    SLACK_APP_TOKEN    - App-level token with connections:write scope (xapp-...)
    SLACK_MEMBER_ID    - Default Slack member ID for DMs

Slack app configuration:
    Required Bot Token Scopes:
        chat:write, im:history, im:read, im:write,
        files:read, files:write, reactions:write
    Socket Mode:
        Enable "Socket Mode" in your Slack app settings.
        Create an App-Level Token with the connections:write scope.
"""

import io
import logging
import os
import re
import asyncio
import threading
import time
from typing import Optional

import requests as http_requests
from PIL import Image

# ---------------------------------------------------------------------------
# Set up module logger – writes to output/slack_server.log (not the console)
# ---------------------------------------------------------------------------
_LOG_FILE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "output",
    "slack_server.log",
)
os.makedirs(os.path.dirname(_LOG_FILE), exist_ok=True)

logger = logging.getLogger("agentY.slack_server")
logger.setLevel(logging.DEBUG)
if not logger.handlers:
    _handler = logging.FileHandler(_LOG_FILE, encoding="utf-8")
    _handler.setLevel(logging.DEBUG)
    _handler.setFormatter(logging.Formatter(
        "[%(asctime)s] [SlackServer] %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    ))
    logger.addHandler(_handler)
    logger.propagate = False  # don't bubble up to the root console handler

# ---------------------------------------------------------------------------
# Module-level references (set by start_slack_server)
# ---------------------------------------------------------------------------
_agent_ref = None          # Will hold the Strands Agent instance
_bot_user_id: Optional[str] = None  # filled on first message to ignore own msgs


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

# How often (seconds) to push partial text to Slack via chat_update.
# Set to a large value (e.g. 60) to effectively disable mid-stream updates
# and only post the final completed message.
_STREAM_UPDATE_INTERVAL = 0.5

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


# ---------------------------------------------------------------------------
# Image downsizing for Claude API limits
# ---------------------------------------------------------------------------

_MAX_IMAGE_BYTES = 5 * 1024 * 1024   # 5 MB hard limit
_OPTIMAL_LONG_EDGE = 1568            # Claude resizes beyond this anyway


def _downsize_image_bytes(data: bytes, img_fmt: str, save_path: str) -> bytes:
    """Downsize *data* in-memory so it fits Claude's 5 MB limit.

    Also caps the long edge at 1568 px for optimal quality (Claude
    server-side resizes anything larger, adding latency with no benefit).

    Overwrites *save_path* on disk with the resized version so the
    agent's file-path reference stays valid.

    Returns the (possibly smaller) image bytes.
    """
    if len(data) <= _MAX_IMAGE_BYTES:
        # Still apply long-edge cap for optimal quality
        img = Image.open(io.BytesIO(data))
        long_edge = max(img.width, img.height)
        if long_edge <= _OPTIMAL_LONG_EDGE:
            return data
        # Resize for quality but size is already OK
        ratio = _OPTIMAL_LONG_EDGE / long_edge
        new_w, new_h = int(img.width * ratio), int(img.height * ratio)
        img = img.resize((new_w, new_h), Image.LANCZOS)
    else:
        img = Image.open(io.BytesIO(data))
        long_edge = max(img.width, img.height)
        target_px = _OPTIMAL_LONG_EDGE
        if long_edge > target_px:
            ratio = target_px / long_edge
            new_w, new_h = int(img.width * ratio), int(img.height * ratio)
            img = img.resize((new_w, new_h), Image.LANCZOS)

    # Re-encode — try PNG first, fall back to JPEG if still too large
    pil_fmt = "PNG" if img_fmt == "png" else "JPEG"
    if img.mode == "RGBA" and pil_fmt == "JPEG":
        img = img.convert("RGB")

    buf = io.BytesIO()
    quality = 90
    while quality >= 20:
        buf.seek(0)
        buf.truncate()
        if pil_fmt == "JPEG":
            img.save(buf, format=pil_fmt, quality=quality, optimize=True)
        else:
            img.save(buf, format=pil_fmt, optimize=True)
        if buf.tell() <= _MAX_IMAGE_BYTES:
            break
        # Switch to JPEG if PNG is too large
        if pil_fmt == "PNG":
            pil_fmt = "JPEG"
            if img.mode == "RGBA":
                img = img.convert("RGB")
            continue
        quality -= 10

    result = buf.getvalue()

    # Overwrite on-disk copy
    with open(save_path, "wb") as fp:
        fp.write(result)

    logger.info(
        "Downsized image for Claude API: %d bytes -> %d bytes (%dx%d, %s)",
        len(data), len(result), img.width, img.height, pil_fmt,
    )
    return result


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

            # Downsize if needed to fit Claude's 5 MB / 1568px limits
            data = _downsize_image_bytes(data, img_fmt, save_path)

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
# Event routing
# ---------------------------------------------------------------------------

# file_share is a valid subtype when a user uploads a file with a caption.
_ALLOWED_SUBTYPES = {None, "file_share"}


def _route_message_event(event: dict, event_id: str = "") -> None:
    """Validate and dispatch a Slack message event to the agent thread.

    Encapsulates the same filtering logic that was previously in the Flask
    /slack/events route, now called from the Socket Mode event handler.
    """
    # Dedup check (Socket Mode can deliver duplicates on reconnect)
    if event_id and _is_duplicate(event_id):
        logger.debug("Duplicate event %s - skipping", event_id)
        return

    subtype = event.get("subtype")
    bot_id = event.get("bot_id")
    user = event.get("user", "")
    event_type = event.get("type", "")
    channel_type = event.get("channel_type", "")

    logger.info(
        "Event: type=%s  channel_type=%s  user=%s  event_id=%s",
        event_type, channel_type, user, event_id,
    )

    # Allow file_share (image uploads) through; block other subtypes
    if subtype not in _ALLOWED_SUBTYPES:
        logger.debug("Ignoring message with subtype=%s", subtype)
        return
    if bot_id:
        logger.debug("Ignoring message from bot_id=%s", bot_id)
        return

    # Resolve our own bot user id and ignore our own messages
    bot_uid = _resolve_bot_user_id()
    if user == bot_uid:
        logger.debug("Ignoring own message from bot user %s", bot_uid)
        return

    text = event.get("text", "").strip()
    files = event.get("files", [])

    if not text and not files:
        logger.debug("Ignoring empty message")
        return

    channel = event.get("channel", "")
    thread_ts = event.get("ts", "")  # reply in thread under their msg

    # Build content: plain text or multimodal blocks with images/video
    if files:
        content = _build_content_blocks(text, files)
    else:
        content = text

    logger.info(
        ">> Processing message from user=%s channel=%s: %s",
        user, channel,
        text[:120] if text else f"[{len(files)} file(s)]",
    )

    # Process asynchronously so Slack's ACK isn't delayed
    t = threading.Thread(
        target=_handle_message_async,
        args=(content, channel, thread_ts, user),
        daemon=True,
    )
    t.start()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def start_slack_server(agent) -> bool:
    """Start the Slack Socket Mode listener in a background thread.

    Uses ``slack_sdk.socket_mode.SocketModeClient`` directly (no Bolt).
    No public URL or ngrok tunnel is required — the listener connects to
    Slack over an outbound WebSocket using the App-Level Token.

    Args:
        agent: The Strands Agent instance that will process messages.

    Returns:
        True if the Socket Mode connection was established successfully,
        False otherwise.
    """
    global _agent_ref
    _agent_ref = agent

    bot_token = os.environ.get("SLACK_BOT_TOKEN", "")
    app_token = os.environ.get("SLACK_APP_TOKEN", "")

    if not bot_token:
        logger.error("SLACK_BOT_TOKEN is not set")
        return False
    if not app_token:
        logger.error(
            "SLACK_APP_TOKEN is not set. "
            "Create an App-Level Token with connections:write scope in your Slack app settings."
        )
        return False

    try:
        from slack_sdk.socket_mode import SocketModeClient
        from slack_sdk.socket_mode.request import SocketModeRequest
        from slack_sdk.socket_mode.response import SocketModeResponse
        from slack_sdk import WebClient

        web_client = WebClient(token=bot_token)

        def _process(client: SocketModeClient, req: SocketModeRequest) -> None:
            """Handle every Socket Mode envelope from Slack."""
            # Always acknowledge immediately so Slack doesn't retry
            client.send_socket_mode_response(
                SocketModeResponse(envelope_id=req.envelope_id)
            )

            if req.type != "events_api":
                logger.debug("Ignored Socket Mode request type: %s", req.type)
                return

            event = req.payload.get("event", {})
            event_id = req.payload.get("event_id", "")

            if event.get("type") == "message":
                _route_message_event(event, event_id=event_id)
            else:
                logger.debug("Ignored Slack event type: %s", event.get("type"))

        sm_client = SocketModeClient(
            app_token=app_token,
            web_client=web_client,
        )
        sm_client.socket_mode_request_listeners.append(_process)

        # connect() is non-blocking — it starts the WebSocket in a background thread
        sm_client.connect()
        logger.info("Slack Socket Mode client connected.")

        # Give the WebSocket a moment to complete the handshake
        time.sleep(2)

        if sm_client.is_connected():
            logger.info("Slack Socket Mode listener started successfully.")
            return True
        else:
            logger.error("Socket Mode client failed to connect within timeout.")
            return False

    except ImportError as exc:
        logger.error(
            "slack_sdk is not installed. Run: pip install slack-sdk\n%s", exc
        )
        return False
    except Exception as exc:
        logger.error("Failed to start Slack Socket Mode: %s", exc, exc_info=True)
        return False
