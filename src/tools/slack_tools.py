"""
Slack integration tools for agentY.

Provides @tool-decorated functions for sending and receiving messages,
images, and videos via Slack DMs using the Slack Web API (slack_sdk).

Environment variables:
    SLACK_BOT_TOKEN  – Bot User OAuth Token (xoxb-...)
    SLACK_MEMBER_ID  – Slack member ID of the primary user for DMs
"""

import json
import logging
import os
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
    thread_ts: str = "",
) -> str:
    """Send a direct message to a Slack user.

    If *user_id* is omitted the message is sent to the default user
    defined by the SLACK_MEMBER_ID environment variable.

    Args:
        text:      The message body.  Supports Slack mrkdwn formatting
                   (*bold*, _italic_, `code`, ```code blocks```, etc.).
        user_id:   (Optional) Slack member ID to DM.  Defaults to
                   SLACK_MEMBER_ID env var.
        thread_ts: (Optional) Timestamp of a parent message to reply in-thread.

    Returns:
        JSON string with the message timestamp and channel, or an error.
    """
    try:
        channel = _get_dm_channel(user_id or None)
        client = _get_slack_client()
        params: dict = {"channel": channel, "text": text}
        if thread_ts:
            params["thread_ts"] = thread_ts
        resp = client.chat_postMessage(**params)
        return json.dumps({
            "ok": True,
            "channel": resp["channel"],
            "ts": resp["ts"],
            "message": "Message sent successfully.",
        })
    except SlackApiError as exc:
        logger.error("Slack API error in slack_send_dm: %s", exc.response["error"])
        return json.dumps({"ok": False, "error": exc.response["error"]})
    except Exception as exc:
        logger.error("Error in slack_send_dm: %s", exc, exc_info=True)
        return json.dumps({"ok": False, "error": str(exc)})


@tool
def slack_send_image(
    file_path: str,
    title: str = "",
    initial_comment: str = "",
    user_id: str = "",
) -> str:
    """Upload and send an image file to a Slack user via DM.

    Args:
        file_path:       Absolute or relative path to the image file.
        title:           (Optional) A title shown above the image in Slack.
        initial_comment: (Optional) Accompanying text message.
        user_id:         (Optional) Slack member ID to DM. Defaults to
                         SLACK_MEMBER_ID env var.

    Returns:
        JSON result with file id and permalink, or an error.
    """
    try:
        path = Path(file_path)
        if not path.exists():
            return json.dumps({"ok": False, "error": f"File not found: {file_path}"})

        channel = _get_dm_channel(user_id or None)
        client = _get_slack_client()

        resp = client.files_upload_v2(
            channel=channel,
            file=str(path),
            title=title or path.name,
            initial_comment=initial_comment,
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
        logger.error("Slack API error in slack_send_image: %s", exc.response["error"])
        return json.dumps({"ok": False, "error": exc.response["error"]})
    except Exception as exc:
        logger.error("Error in slack_send_image: %s", exc, exc_info=True)
        return json.dumps({"ok": False, "error": str(exc)})


@tool
def slack_send_video(
    file_path: str,
    title: str = "",
    initial_comment: str = "",
    user_id: str = "",
) -> str:
    """Upload and send a video file to a Slack user via DM.

    Args:
        file_path:       Absolute or relative path to the video file.
        title:           (Optional) A title shown above the video in Slack.
        initial_comment: (Optional) Accompanying text message.
        user_id:         (Optional) Slack member ID to DM. Defaults to
                         SLACK_MEMBER_ID env var.

    Returns:
        JSON result with file id and permalink, or an error.
    """
    try:
        path = Path(file_path)
        if not path.exists():
            return json.dumps({"ok": False, "error": f"File not found: {file_path}"})

        channel = _get_dm_channel(user_id or None)
        client = _get_slack_client()

        resp = client.files_upload_v2(
            channel=channel,
            file=str(path),
            title=title or path.name,
            initial_comment=initial_comment,
        )

        file_info = resp.get("file", {})
        return json.dumps({
            "ok": True,
            "file_id": file_info.get("id", ""),
            "permalink": file_info.get("permalink", ""),
            "message": f"Video '{path.name}' sent successfully.",
        })
    except SlackApiError as exc:
        logger.error("Slack API error in slack_send_video: %s", exc.response["error"])
        return json.dumps({"ok": False, "error": exc.response["error"]})
    except Exception as exc:
        logger.error("Error in slack_send_video: %s", exc, exc_info=True)
        return json.dumps({"ok": False, "error": str(exc)})


@tool
def slack_send_file(
    file_path: str,
    title: str = "",
    initial_comment: str = "",
    user_id: str = "",
) -> str:
    """Upload and send any file to a Slack user via DM.

    Use this as a general-purpose file sender (documents, archives, etc.).
    For images or videos, prefer the dedicated slack_send_image /
    slack_send_video tools.

    Args:
        file_path:       Absolute or relative path to the file.
        title:           (Optional) A title for the file in Slack.
        initial_comment: (Optional) Accompanying text message.
        user_id:         (Optional) Slack member ID to DM. Defaults to
                         SLACK_MEMBER_ID env var.

    Returns:
        JSON result with file id and permalink, or an error.
    """
    try:
        path = Path(file_path)
        if not path.exists():
            return json.dumps({"ok": False, "error": f"File not found: {file_path}"})

        channel = _get_dm_channel(user_id or None)
        client = _get_slack_client()

        resp = client.files_upload_v2(
            channel=channel,
            file=str(path),
            title=title or path.name,
            initial_comment=initial_comment,
        )

        file_info = resp.get("file", {})
        return json.dumps({
            "ok": True,
            "file_id": file_info.get("id", ""),
            "permalink": file_info.get("permalink", ""),
            "message": f"File '{path.name}' sent successfully.",
        })
    except SlackApiError as exc:
        logger.error("Slack API error in slack_send_file: %s", exc.response["error"])
        return json.dumps({"ok": False, "error": exc.response["error"]})
    except Exception as exc:
        logger.error("Error in slack_send_file: %s", exc, exc_info=True)
        return json.dumps({"ok": False, "error": str(exc)})


@tool
def slack_read_messages(
    count: int = 10,
    user_id: str = "",
) -> str:
    """Read recent messages from the DM conversation with a Slack user.

    Args:
        count:   Number of recent messages to retrieve (max 100, default 10).
        user_id: (Optional) Slack member ID whose DM to read. Defaults to
                 SLACK_MEMBER_ID env var.

    Returns:
        JSON array of messages (newest first), each with ts, user, text,
        and optional files array — or an error.
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
    """Add an emoji reaction to a message in a DM conversation.

    Args:
        emoji:     Emoji name without colons (e.g. "thumbsup", "fire").
        timestamp: The ``ts`` value of the message to react to.
        user_id:   (Optional) Slack member ID whose DM contains the message.
                   Defaults to SLACK_MEMBER_ID env var.

    Returns:
        JSON success / error result.
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
