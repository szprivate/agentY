"""
Slack integration tools for agentY.

Provides @tool-decorated functions for sending and receiving messages,
images, and videos via Slack using the Slack Web API (slack_sdk).

When the agent is invoked from a Slack event (DM or channel), the
current channel context is set automatically so that file uploads and
replies go back to the originating conversation.

Environment variables:
    SLACK_BOT_TOKEN  - Bot User OAuth Token (xoxb-...)
    SLACK_MEMBER_ID  - Slack member ID of the primary user for DMs
"""

import json
import logging
import os
import threading
from pathlib import Path
from typing import Optional

from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError
from strands import tool

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Singleton Slack client
# ---------------------------------------------------------------------------

_slack_client: Optional[WebClient] = None
_dm_channel_id: Optional[str] = None

# ---------------------------------------------------------------------------
# Thread-local channel context (set by slack_server when processing events)
# ---------------------------------------------------------------------------
_channel_context = threading.local()


def set_slack_channel_context(channel_id: str, thread_ts: str = "") -> None:
    """Set the Slack channel context for the current thread.

    Called by the Slack event server before invoking the agent so that
    tools automatically send files/messages to the originating channel.
    """
    _channel_context.channel_id = channel_id
    _channel_context.thread_ts = thread_ts


def clear_slack_channel_context() -> None:
    """Clear the channel context after the agent finishes."""
    _channel_context.channel_id = None
    _channel_context.thread_ts = None


def _get_active_channel(channel_id: str = "") -> str:
    """Return the channel to use, preferring an explicit value, then context."""
    if channel_id:
        return channel_id
    ctx = getattr(_channel_context, "channel_id", None)
    if ctx:
        return ctx
    return ""


def _get_active_thread_ts(thread_ts: str = "") -> str:
    """Return the thread_ts to use, preferring explicit, then context."""
    if thread_ts:
        return thread_ts
    return getattr(_channel_context, "thread_ts", "") or ""


def _get_slack_client() -> WebClient:
    """Return a cached Slack WebClient, initialised from env vars."""
    global _slack_client
    if _slack_client is None:
        token = os.environ.get("SLACK_BOT_TOKEN")
        if not token:
            raise RuntimeError(
                "SLACK_BOT_TOKEN environment variable is not set. "
                "Cannot initialise Slack client."
            )
        _slack_client = WebClient(token=token)
    return _slack_client


def _slack_api_error_payload(exc: SlackApiError) -> dict:
    """Extract a consistent, human-readable dict from a SlackApiError.

    Includes the ``needed`` scope when the error is ``missing_scope`` so the
    caller knows exactly which OAuth scope to add to the Slack App.
    """
    resp = exc.response or {}
    error = resp.get("error", str(exc))
    payload: dict = {"ok": False, "error": error}
    if error == "missing_scope":
        needed = resp.get("needed", "")
        provided = resp.get("provided", "")
        payload["needed_scope"] = needed
        payload["provided_scopes"] = provided
        payload["fix"] = (
            f"Add the '{needed}' scope to your Slack App "
            "(https://api.slack.com/apps → OAuth & Permissions → Scopes), "
            "then reinstall the app to your workspace."
        )
    return payload


def _get_dm_channel(user_id: Optional[str] = None) -> str:
    """Open (or retrieve) a DM channel with the given user.

    If *user_id* is ``None``, falls back to ``SLACK_MEMBER_ID`` env var.
    Results are cached so subsequent calls don't hit the API again.
    """
    global _dm_channel_id
    uid = user_id or os.environ.get("SLACK_MEMBER_ID")
    if not uid:
        raise RuntimeError(
            "No user_id provided and SLACK_MEMBER_ID is not set."
        )

    # Only cache when using the default member
    if uid == os.environ.get("SLACK_MEMBER_ID") and _dm_channel_id:
        return _dm_channel_id

    client = _get_slack_client()
    resp = client.conversations_open(users=[uid])
    channel_id = resp["channel"]["id"]

    if uid == os.environ.get("SLACK_MEMBER_ID"):
        _dm_channel_id = channel_id
    return channel_id


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------


@tool
def slack_send_dm(
    text: str,
    user_id: str = "",
    channel_id: str = "",
    thread_ts: str = "",
) -> str:
    """Send a text message to a Slack channel or DM. Auto-routes to originating channel inside an event handler.

    Requires the ``chat:write`` OAuth scope on the bot token.
    Opening a DM via user_id also requires ``im:write``.
    If you see a ``missing_scope`` error, add the needed scope at
    https://api.slack.com/apps → OAuth & Permissions → Scopes, then reinstall.

    Args:
        text: Message body (mrkdwn).
        user_id: Member ID to DM (ignored if channel_id set).
        channel_id: Channel ID; uses event context if omitted.
        thread_ts: Parent ts for threaded reply.
    """
    try:
        channel = _get_active_channel(channel_id)
        if not channel:
            channel = _get_dm_channel(user_id or None)
        ts = _get_active_thread_ts(thread_ts)
        client = _get_slack_client()
        params: dict = {"channel": channel, "text": text}
        if ts:
            params["thread_ts"] = ts
        resp = client.chat_postMessage(**params)
        return json.dumps({
            "ok": True,
            "channel": resp["channel"],
            "ts": resp["ts"],
            "message": "Message sent successfully.",
        })
    except SlackApiError as exc:
        payload = _slack_api_error_payload(exc)
        logger.error("Slack API error in slack_send_dm: %s", payload)
        return json.dumps(payload)
    except Exception as exc:
        logger.error("Error in slack_send_dm: %s", exc, exc_info=True)
        return json.dumps({"ok": False, "error": str(exc)})


@tool
def slack_send_image(
    file_path: str,
    title: str = "",
    initial_comment: str = "",
    user_id: str = "",
    channel_id: str = "",
) -> str:
    """Upload an image file to a Slack channel or DM.

    Requires the ``files:write`` OAuth scope on the bot token.
    If you see a ``missing_scope`` error, add ``files:write`` to your Slack
    App's Bot Token Scopes and reinstall the app.

    Args:
        file_path: Local path to image.
        title: Title shown above image.
        initial_comment: Accompanying text.
        user_id: Member ID to DM (ignored if channel_id set).
        channel_id: Channel ID; uses event context if omitted.
    """
    try:
        path = Path(file_path)
        if not path.exists():
            return json.dumps({"ok": False, "error": f"File not found: {file_path}"})

        channel = _get_active_channel(channel_id)
        if not channel:
            channel = _get_dm_channel(user_id or None)
        thread_ts = _get_active_thread_ts()
        client = _get_slack_client()

        resp = client.files_upload_v2(
            channel=channel,
            file=str(path),
            title=title or path.name,
            initial_comment=initial_comment,
            **(dict(thread_ts=thread_ts) if thread_ts else {}),
        )

        # files_upload_v2 returns a slightly different shape
        file_info = resp.get("file", {})
        return json.dumps({
            "ok": True,
            "file_id": file_info.get("id", ""),
            "permalink": file_info.get("permalink", ""),
            "message": f"Image '{path.name}' sent successfully.",
        })
    except SlackApiError as exc:
        payload = _slack_api_error_payload(exc)
        logger.error("Slack API error in slack_send_image: %s", payload)
        return json.dumps(payload)
    except Exception as exc:
        logger.error("Error in slack_send_image: %s", exc, exc_info=True)
        return json.dumps({"ok": False, "error": str(exc)})


@tool
def slack_send_video(
    file_path: str,
    title: str = "",
    initial_comment: str = "",
    user_id: str = "",
    channel_id: str = "",
) -> str:
    """Upload a video file to a Slack channel or DM.

    Args:
        file_path: Local path to video.
        title: Title shown above video.
        initial_comment: Accompanying text.
        user_id: Member ID to DM (ignored if channel_id set).
        channel_id: Channel ID; uses event context if omitted.
    """
    try:
        path = Path(file_path)
        if not path.exists():
            return json.dumps({"ok": False, "error": f"File not found: {file_path}"})

        channel = _get_active_channel(channel_id)
        if not channel:
            channel = _get_dm_channel(user_id or None)
        thread_ts = _get_active_thread_ts()
        client = _get_slack_client()

        resp = client.files_upload_v2(
            channel=channel,
            file=str(path),
            title=title or path.name,
            initial_comment=initial_comment,
            **(dict(thread_ts=thread_ts) if thread_ts else {}),
        )

        file_info = resp.get("file", {})
        return json.dumps({
            "ok": True,
            "file_id": file_info.get("id", ""),
            "permalink": file_info.get("permalink", ""),
            "message": f"Video '{path.name}' sent successfully.",
        })
    except SlackApiError as exc:
        payload = _slack_api_error_payload(exc)
        logger.error("Slack API error in slack_send_video: %s", payload)
        return json.dumps(payload)
    except Exception as exc:
        logger.error("Error in slack_send_video: %s", exc, exc_info=True)
        return json.dumps({"ok": False, "error": str(exc)})


@tool
def slack_send_file(
    file_path: str,
    title: str = "",
    initial_comment: str = "",
    user_id: str = "",
    channel_id: str = "",
) -> str:
    """Upload and send any file (document, archive, etc.) to Slack. Prefer slack_send_image/video for media.

    Args:
        file_path: Local path to the file.
        title: Optional title for the file in Slack.
        initial_comment: Optional accompanying text.
        user_id: Slack member ID to DM (ignored if channel_id set).
        channel_id: Slack channel ID; uses event context if omitted.
    """
    try:
        path = Path(file_path)
        if not path.exists():
            return json.dumps({"ok": False, "error": f"File not found: {file_path}"})

        channel = _get_active_channel(channel_id)
        if not channel:
            channel = _get_dm_channel(user_id or None)
        thread_ts = _get_active_thread_ts()
        client = _get_slack_client()

        resp = client.files_upload_v2(
            channel=channel,
            file=str(path),
            title=title or path.name,
            initial_comment=initial_comment,
            **(dict(thread_ts=thread_ts) if thread_ts else {}),
        )

        file_info = resp.get("file", {})
        return json.dumps({
            "ok": True,
            "file_id": file_info.get("id", ""),
            "permalink": file_info.get("permalink", ""),
            "message": f"File '{path.name}' sent successfully.",
        })
    except SlackApiError as exc:
        payload = _slack_api_error_payload(exc)
        logger.error("Slack API error in slack_send_file: %s", payload)
        return json.dumps(payload)
    except Exception as exc:
        logger.error("Error in slack_send_file: %s", exc, exc_info=True)
        return json.dumps({"ok": False, "error": str(exc)})


@tool
def slack_read_messages(
    count: int = 10,
    user_id: str = "",
) -> str:
    """Read recent messages from a Slack DM conversation.

    Args:
        count: Number of messages to retrieve (max 100, default 10).
        user_id: Slack member ID to read DM from; defaults to SLACK_MEMBER_ID env var.
    """
    try:
        channel = _get_dm_channel(user_id or None)
        client = _get_slack_client()

        resp = client.conversations_history(
            channel=channel,
            limit=min(count, 100),
        )

        messages = []
        for msg in resp.get("messages", []):
            entry: dict = {
                "ts": msg.get("ts"),
                "user": msg.get("user", msg.get("bot_id", "unknown")),
                "text": msg.get("text", ""),
            }
            # Include file metadata if present
            if "files" in msg:
                entry["files"] = [
                    {
                        "id": f.get("id"),
                        "name": f.get("name"),
                        "mimetype": f.get("mimetype"),
                        "url_private": f.get("url_private"),
                        "permalink": f.get("permalink"),
                    }
                    for f in msg["files"]
                ]
            messages.append(entry)

        return json.dumps({"ok": True, "count": len(messages), "messages": messages})
    except SlackApiError as exc:
        logger.error("Slack API error in slack_read_messages: %s", exc.response["error"])
        return json.dumps({"ok": False, "error": exc.response["error"]})
    except Exception as exc:
        logger.error("Error in slack_read_messages: %s", exc, exc_info=True)
        return json.dumps({"ok": False, "error": str(exc)})


@tool
def slack_add_reaction(
    emoji: str,
    timestamp: str,
    user_id: str = "",
) -> str:
    """Add an emoji reaction to a Slack DM message.

    Args:
        emoji: Emoji name without colons e.g. 'thumbsup', 'fire'.
        timestamp: The ts of the message to react to.
        user_id: Slack member ID whose DM contains the message; defaults to SLACK_MEMBER_ID.
    """
    try:
        channel = _get_dm_channel(user_id or None)
        client = _get_slack_client()
        client.reactions_add(name=emoji, channel=channel, timestamp=timestamp)
        return json.dumps({"ok": True, "message": f"Reaction :{emoji}: added."})
    except SlackApiError as exc:
        logger.error("Slack API error in slack_add_reaction: %s", exc.response["error"])
        return json.dumps({"ok": False, "error": exc.response["error"]})
    except Exception as exc:
        logger.error("Error in slack_add_reaction: %s", exc, exc_info=True)
        return json.dumps({"ok": False, "error": str(exc)})


@tool
def slack_send_json(
    data: str,
    filename: str = "workflow.json",
    title: str = "",
    comment: str = "",
    channel_id: str = "",
) -> str:
    """Upload JSON as a Slack file snippet instead of pasting into messages.

    Args:
        data: JSON string or dict/list to send.
        filename: Filename in Slack (default 'workflow.json').
        title: Title above the snippet.
        comment: Message alongside the file.
        channel_id: Channel ID; auto-detected if omitted.
    """
    try:
        # Pretty-print if possible
        try:
            parsed = json.loads(data) if isinstance(data, str) else data
            pretty = json.dumps(parsed, indent=2, ensure_ascii=False)
        except (json.JSONDecodeError, TypeError):
            pretty = data  # send as-is if not valid JSON

        # Determine target channel
        target_channel = channel_id or _get_active_channel()
        if not target_channel:
            user_id = os.environ.get("SLACK_MEMBER_ID")
            if not user_id:
                return json.dumps({"ok": False, "error": "No channel context and SLACK_MEMBER_ID not set."})
            target_channel = _get_dm_channel(user_id)

        # Save to disk
        save_dir = os.path.join(os.getcwd(), "slack_downloads")
        os.makedirs(save_dir, exist_ok=True)
        # Ensure unique filename
        base, ext = os.path.splitext(filename)
        if not ext:
            ext = ".json"
        import time as _time
        local_path = os.path.join(save_dir, f"{base}_{int(_time.time())}{ext}")
        with open(local_path, "w", encoding="utf-8") as fh:
            fh.write(pretty)
        logger.info("Saved JSON to %s (%d bytes)", local_path, len(pretty))

        # Upload to Slack
        client = _get_slack_client()
        thread_ts = _get_active_thread_ts()
        upload_kwargs: dict = {
            "channel": target_channel,
            "file": local_path,
            "filename": f"{base}{ext}",
            "title": title or filename,
        }
        if comment:
            upload_kwargs["initial_comment"] = comment
        if thread_ts:
            upload_kwargs["thread_ts"] = thread_ts

        resp = client.files_upload_v2(**upload_kwargs)
        file_id = resp.get("file", {}).get("id", "unknown")
        logger.info("Uploaded JSON file to Slack: file_id=%s", file_id)
        return json.dumps({
            "ok": True,
            "file_path": local_path,
            "file_id": file_id,
            "message": f"JSON file uploaded to Slack as {filename}",
        })
    except SlackApiError as exc:
        payload = _slack_api_error_payload(exc)
        logger.error("Slack API error in slack_send_json: %s", payload)
        return json.dumps(payload)
    except Exception as exc:
        logger.error("Error in slack_send_json: %s", exc, exc_info=True)
        return json.dumps({"ok": False, "error": str(exc)})
