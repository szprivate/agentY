"""Slack as a second line into agentY — not a replacement for the sidebar.

The ComfyUI side panel stays exactly what it was. This adds a second way in and a
second way to watch, so a run started at the desk can be followed (and answered,
and steered) from a phone:

* **watching** — every turn is mirrored, including turns started in the panel.
  That is the point: you queue a video from the canvas, walk away, and see it
  finish. :mod:`src.utils.turn_bus` is what makes a turn visible to anyone but
  the browser that asked for it; :mod:`src.utils.slack_render` decides what each
  event looks like once it gets here.
* **talking** — a DM starts a turn in *the conversation the panel is already in*,
  so Slack is another window on one session rather than a second, forked one.

Three things a message can mean, resolved in this order, because getting it wrong
is worse than any of them:

1. the agent asked a question and is holding the turn open → this is the answer;
2. a turn is running → **interject** it (the panel does the same). Starting a
   second turn would run two of them through one pipeline singleton, which
   corrupts both;
3. otherwise → start a turn.

Transport is Socket Mode, so nothing has to be reachable from the internet: the
host opens an outbound WebSocket to Slack. Off unless both tokens are set.

Setup (see ``docs/slack.md``)::

    SLACK_BOT_TOKEN=xoxb-…     bot token; scopes: chat:write, files:write,
                               im:history, im:read, im:write, users:read
    SLACK_APP_TOKEN=xapp-…     app-level token with connections:write
    SLACK_ALLOWED_USERS=U123…  who may drive the agent (comma-separated)

``SLACK_ALLOWED_USERS`` is not optional and has no default. Anyone who can DM the
bot would otherwise be able to run generations, tools and scripts on this
machine, and "whoever installed the app" is not an access rule.
"""
from __future__ import annotations

import logging
import os
import queue
import threading
import time
from pathlib import Path

from src.utils import turn_bus
from src.utils.slack_render import Post, TurnRender, clip, to_mrkdwn

logger = logging.getLogger("agentY.slack")

# A Slack edit is a round trip and its own rate-limit bucket (chat.update is
# ~50/minute per workspace). Text streams in far faster than that, so the answer
# message is rewritten on a timer instead of per token.
_EDIT_INTERVAL = 1.5
# How long a mirrored turn's Slack state is kept after it ends, so a late reply
# ("continue") still finds the turn it belongs to.
_TURN_TTL = 900.0
_DOWNLOAD_DIRNAME = "slack_uploads"
# Slack lets one message carry ten files. Every one of them becomes an input the
# agent is expected to reason about, so the ceiling is what a turn can actually
# use rather than what Slack can carry.
_MAX_INBOUND_FILES = 10


def _env_list(name: str) -> list:
    return [p.strip() for p in (os.environ.get(name) or "").split(",") if p.strip()]


def enabled() -> bool:
    """The settings switch. **Off** unless turned on — a second channel that
    starts talking to a workspace on its own is not a feature."""
    return bool(_setting("enabled", False))


def configured() -> bool:
    """Both tokens present — i.e. the user has actually set this up."""
    return bool((os.environ.get("SLACK_BOT_TOKEN") or "").strip()
                and (os.environ.get("SLACK_APP_TOKEN") or "").strip())


# Slack issues two kinds of token and they are not interchangeable. The bot token
# (`xoxb-`) is the one everything about the app points you at; the app-level token
# (`xapp-`) is created separately, under Basic Information → App-Level Tokens, and
# is the ONLY one `apps.connections.open` accepts — the call that opens the Socket
# Mode WebSocket. Put the bot token in both fields, as is easy to do, and Slack
# answers `not_allowed_token_type`, which names neither the field nor the fix.
_TOKEN_PREFIX = {"SLACK_BOT_TOKEN": "xoxb-", "SLACK_APP_TOKEN": "xapp-"}


def token_complaint() -> str:
    """What is wrong with the tokens, in words, or "" when they look right.

    Checked before connecting because the alternative is an SDK exception at the
    bottom of a stack trace, and the answer to it is one sentence.
    """
    for name, prefix in _TOKEN_PREFIX.items():
        value = (os.environ.get(name) or "").strip()
        if not value:
            return f"{name} is not set."
        if value.startswith(prefix):
            continue
        other = next((n for n, p in _TOKEN_PREFIX.items()
                      if n != name and value.startswith(p)), "")
        if other:
            return (f"{name} holds a {value.split('-')[0]}- token, which is what "
                    f"{other} wants. The two are not interchangeable.")
        return (f"{name} should start with '{prefix}'. "
                + ("Create it under Basic Information → App-Level Tokens, with the "
                   "connections:write scope — it is not the bot token."
                   if name == "SLACK_APP_TOKEN" else
                   "It is the Bot User OAuth Token under OAuth & Permissions."))
    return ""


def _setting(key: str, default):
    try:
        from src.utils.settings import load_settings
        return (load_settings().get("slack") or {}).get(key, default)
    except Exception:  # noqa: BLE001 — settings must never break the bridge
        return default


class _Keyed:
    """The Slack message ids one turn is using."""

    def __init__(self):
        self.answer_ts = ""
        self.by_key: dict = {}


class SlackTurn:
    """One turn, as it appears in Slack: a message, its thread, and its files."""

    def __init__(self, bridge, turn, channel: str):
        self.bridge = bridge
        self.turn = turn
        self.channel = channel
        self.render = TurnRender(
            origin=turn.origin, started_by=turn.text,
            show_thinking=bool(_setting("show_thinking", True)),
            show_tools=bool(_setting("show_tools", True)))
        self.ids = _Keyed()
        self.ask_request_id = ""
        self.ended = 0.0
        self._last_edit = 0.0
        self._pending_answer = ""

    # ── outbound ──────────────────────────────────────────────────────────────
    def feed(self, event: dict) -> None:
        """Runs on the TURN's thread, so it does no I/O.

        Rendering is pure and cheap; every Slack call it implies is handed to the
        worker, in order. A hook run makes dozens of them, and doing them here
        would put a round trip to Slack between the agent and its next step — the
        panel would go slow because a phone was watching.

        The two flags below are set here rather than queued because an incoming
        DM is raced against them: a message that arrives while the agent is
        waiting is an *answer*, and finding that out a queue-drain later is too
        late.
        """
        kind = str(event.get("type") or "")
        if kind == "ask":
            self.ask_request_id = str(event.get("request_id") or "")
        for post in self.render.feed(event):
            self.bridge._call(self._apply, post)
        if kind == "done":
            self.bridge._call(self._flush_answer, True)
            self.ended = time.time()
            self.ask_request_id = ""

    def _apply(self, post: Post) -> None:
        """Runs on the worker thread — every Slack id in here is single-threaded."""
        if post.where == "answer":
            self._pending_answer = post.text
            self._flush_answer()
            return
        if post.kind == "file":
            self.bridge._do_upload(self.channel, post.path, post.text)
            return
        if post.where == "channel":
            self.bridge.post(self.channel, post.text)
            return
        # detail → a reply in this turn's thread
        thread_ts = self._ensure_answer()
        if not thread_ts:
            return
        if post.kind == "clear":
            ts = self.ids.by_key.pop(post.key, "")
            if ts:
                self.bridge._do_delete(self.channel, ts)
            return
        if not post.text.strip():
            return
        ts = self.ids.by_key.get(post.key) if post.key else None
        if ts:
            self.bridge._do_update(self.channel, ts, post.text)
            return
        new_ts = self.bridge.post(self.channel, post.text, thread_ts=thread_ts)
        if post.key and new_ts:
            self.ids.by_key[post.key] = new_ts

    def _ensure_answer(self) -> str:
        """The turn's own message, created on first need. Everything hangs off it."""
        if not self.ids.answer_ts:
            self.ids.answer_ts = self.bridge.post(self.channel, self.render.opening())
        return self.ids.answer_ts

    def _flush_answer(self, force: bool = False) -> None:
        if not self._pending_answer:
            return
        now = time.time()
        if not force and now - self._last_edit < _EDIT_INTERVAL:
            return
        self._last_edit = now
        text, self._pending_answer = self._pending_answer, ""
        if self.ids.answer_ts:
            self.bridge._do_update(self.channel, self.ids.answer_ts, text)
        else:
            self.ids.answer_ts = self.bridge.post(self.channel, text)

    def tick(self) -> None:
        """Let a throttled edit land even when no further event arrives.

        Called from the worker loop, which is also the only thread that runs
        :meth:`_apply` — so the edit and the posts it races with are ordered.
        """
        self._flush_answer()


class SlackBridge:
    """Owns the Slack connection, the mirrored turns, and what a message means.

    The Slack client is injected so every decision in here can be tested against
    a fake one — what gets said, what a message is taken to mean, who is allowed
    to say it — without a workspace or a network.
    """

    def __init__(self, client=None, *, start_turn=None, answer=None, interject=None,
                 allowed_users=None, default_channel: str = ""):
        self.client = client
        self._start_turn = start_turn
        self._answer = answer
        self._interject = interject
        self.allowed = list(allowed_users if allowed_users is not None
                            else _env_list("SLACK_ALLOWED_USERS"))
        self.default_channel = default_channel
        self.turns: dict = {}            # request_id -> SlackTurn
        self._lock = threading.RLock()
        self._out: "queue.Queue" = queue.Queue()
        self._stop = threading.Event()
        self._worker = None
        self._socket = None
        self.bot_user_id = ""

    # ── talking to Slack (all of it funnelled through one worker thread) ──────
    # Slack calls are network calls, and they happen on the turn's own thread
    # unless something moves them off it. This is that something: the turn hands
    # over a closure and carries on.
    def _call(self, fn, *args, **kwargs):
        self._out.put((fn, args, kwargs))

    def post(self, channel: str, text: str, thread_ts: str = "") -> str:
        """Post and return the message ts. Synchronous — the ts is needed to edit
        it later, and losing it means the next edit posts a duplicate instead."""
        if not self.client or not text.strip():
            return ""
        kw = {"channel": channel, "text": clip(text, 39000)}
        if thread_ts:
            kw["thread_ts"] = thread_ts
        try:
            resp = self.client.chat_postMessage(**kw)
            return str((resp or {}).get("ts") or "")
        except Exception:  # noqa: BLE001
            logger.exception("slack: chat_postMessage failed")
            return ""

    def _do_update(self, channel: str, ts: str, text: str) -> None:
        if not self.client or not ts:
            return
        self.client.chat_update(channel=channel, ts=ts, text=clip(text, 39000))

    def _do_delete(self, channel: str, ts: str) -> None:
        if not self.client or not ts:
            return
        try:
            self.client.chat_delete(channel=channel, ts=ts)
        except Exception:  # noqa: BLE001 — an already-gone message is not an error
            logger.debug("slack: chat_delete(%s) failed", ts, exc_info=True)

    def _do_upload(self, channel: str, path: str, caption: str = "") -> None:
        if not self.client or not path:
            return
        p = Path(path)
        if not p.is_file():
            self.post(channel, f"_{caption or p.name} — the file is not on disk any more._")
            return
        limit = int(_setting("max_upload_mb", 45)) * 1024 * 1024
        if p.stat().st_size > limit:
            # Slack would reject it; say where it is instead of failing silently.
            self.post(channel, f"_{caption or p.name} is too large to post here._\n`{p}`")
            return
        self.client.files_upload_v2(channel=channel, file=str(p),
                                    filename=p.name, initial_comment=caption or None)

    def _pump(self) -> None:
        while not self._stop.is_set():
            try:
                item = self._out.get(timeout=0.5)
            except queue.Empty:
                self._tick_turns()
                continue
            if item is None:
                break
            self._run(item)

    def _run(self, item) -> None:
        fn, args, kwargs = item
        try:
            fn(*args, **kwargs)
        except Exception:  # noqa: BLE001
            logger.exception("slack: %s failed", getattr(fn, "__name__", fn))

    def flush(self) -> None:
        """Do everything queued for Slack now, on this thread.

        The worker does this continuously; a shutdown (and a test) needs it to
        happen without one.
        """
        while True:
            try:
                item = self._out.get_nowait()
            except queue.Empty:
                return
            if item is None:
                return
            self._run(item)

    def _tick_turns(self) -> None:
        """Land throttled edits and forget turns nobody will reply to."""
        now = time.time()
        with self._lock:
            turns = list(self.turns.items())
        for rid, st in turns:
            st.tick()
            if st.ended and now - st.ended > _TURN_TTL:
                with self._lock:
                    self.turns.pop(rid, None)

    # ── watching every turn ───────────────────────────────────────────────────
    def on_turn_event(self, event: dict, turn) -> None:
        """Registered on the turn bus. Runs on the turn's thread — stays cheap."""
        channel = self.default_channel
        if not channel:
            return
        with self._lock:
            st = self.turns.get(turn.request_id)
            if st is None:
                if str(event.get("type")) == "done":
                    return       # nothing was ever shown; nothing to close
                st = SlackTurn(self, turn, channel)
                self.turns[turn.request_id] = st
        st.feed(event)

    # ── the agent handing something over on purpose ───────────────────────────
    def send_files(self, paths: list, message: str = "") -> dict:
        """Put files in the DM deliberately, rather than because a run made them.

        The mirror uploads what a run *produced*. This is for everything else the
        agent decides is worth having in your hand: the JSON it just wrote, one
        chosen frame out of sixty, a script, a log it wants you to look at.

        Every file is checked here rather than in the worker, so the agent is told
        what it actually sent while it can still say so in the same breath — a
        report that arrives after the turn is a report nobody reads.
        """
        if not self.client:
            return {"error": "the Slack bridge is not connected."}
        if not self.default_channel:
            return {"error": "the Slack bridge has nowhere to post — no DM was "
                             "opened (check SLACK_ALLOWED_USERS and the im:write "
                             "scope)."}
        limit = int(_setting("max_upload_mb", 45)) * 1024 * 1024
        sent, missing, too_large = [], [], []
        for raw in paths or []:
            p = Path(str(raw))
            if not p.is_file():
                missing.append(str(raw))
            elif p.stat().st_size > limit:
                too_large.append(str(p))
            else:
                sent.append(p)
        if message.strip():
            self._call(self.post, self.default_channel, to_mrkdwn(clip(message, 3000)))
        for p in sent:
            self._call(self._do_upload, self.default_channel, str(p), "")
        return {"sent": [str(p) for p in sent], "missing": missing,
                "too_large": too_large}

    # ── inbound ───────────────────────────────────────────────────────────────
    def route(self, user: str, text: str, files: list | None = None) -> dict:
        """What this message means, and what was done about it.

        Returns ``{"action": …}`` — ``answer``/``interject``/``turn``/``denied``/
        ``ignored`` — which is what the tests assert on and what the log records.
        """
        text = (text or "").strip()
        files = list(files or [])
        if not text and not files:
            return {"action": "ignored", "why": "empty"}
        if user and self.bot_user_id and user == self.bot_user_id:
            return {"action": "ignored", "why": "own message"}
        if not self.allowed:
            logger.warning("slack: a message arrived but SLACK_ALLOWED_USERS is "
                           "empty — refusing to act on it")
            return {"action": "denied", "why": "no allow-list configured"}
        if user not in self.allowed:
            return {"action": "denied", "why": "not in SLACK_ALLOWED_USERS"}

        pending = self._pending_ask()
        if pending is not None and text:
            rid, st = pending
            if self._answer and self._answer(rid, text):
                st.ask_request_id = ""
                return {"action": "answer", "request_id": rid}

        live = self._live_turn()
        if live is not None and text:
            rid, _st = live
            if self._interject and self._interject(rid, text):
                return {"action": "interject", "request_id": rid}

        if self._start_turn is None:
            return {"action": "ignored", "why": "no turn starter wired"}
        rid = self._start_turn(text, files)
        return {"action": "turn", "request_id": rid}

    def _pending_ask(self):
        with self._lock:
            for rid, st in self.turns.items():
                if st.ask_request_id:
                    return st.ask_request_id, st
        return None

    def _live_turn(self):
        running = {t.request_id for t in turn_bus.active()}
        with self._lock:
            for rid, st in self.turns.items():
                if rid in running and not st.ended:
                    return rid, st
        return None

    # ── lifecycle ─────────────────────────────────────────────────────────────
    def start_worker(self) -> None:
        if self._worker is None:
            self._worker = threading.Thread(target=self._pump, name="agentY-slack",
                                            daemon=True)
            self._worker.start()
        turn_bus.observe(self.on_turn_event)

    def stop(self) -> None:
        self._stop.set()
        turn_bus.unobserve(self.on_turn_event)
        self._out.put(None)
        if self._socket is not None:
            try:
                self._socket.close()
            except Exception:  # noqa: BLE001
                pass


# ── inbound Slack events ──────────────────────────────────────────────────────

# Attaching a picture is not a different kind of event to Slack — it is a message
# with a subtype. Rejecting every subtype (which is otherwise the right rule) drops
# exactly the message someone took a photo for, and drops it in total silence.
_ACTIONABLE_SUBTYPES = {"file_share"}


def is_actionable(event: dict) -> bool:
    """Whether a Slack event is a DM from a person that we should act on.

    Slack sends a great deal that looks like a message and is not: the bot's own
    posts (every mirrored turn would otherwise talk to itself, forever), edits and
    deletions of old messages, channel joins, and thread broadcasts. A person
    typing has no subtype — and a person *attaching something* has ``file_share``,
    which is the one exception.
    """
    if str((event or {}).get("type")) != "message":
        return False
    subtype = str(event.get("subtype") or "")
    if subtype and subtype not in _ACTIONABLE_SUBTYPES:
        return False
    if event.get("bot_id"):
        return False
    if str(event.get("channel_type") or "") != "im":
        return False
    return bool(str(event.get("user") or ""))


def download_files(client, event: dict, dest_dir) -> tuple:
    """Save what the user attached, and hand back ``(paths, skipped)``.

    A DM with a picture and "make this warmer" is one message to a person, so the
    attachment has to arrive as an INPUT — a path the agent can wire into a
    LoadImage or hand to the vision agent — rather than as something it is merely
    told about. Videos land the same way (see ``_build_content`` on the server:
    images are embedded as vision where the model can read them, videos are always
    listed as paths).

    Private Slack URLs need the bot token, which is why this cannot be a plain
    download. It streams to disk rather than through memory, because the thing
    someone films on a phone and sends is measured in hundreds of megabytes and
    the alternative is holding all of it at once.

    *skipped* carries a line per file that did not make it, so the caller can say
    so in Slack. Silence is the wrong answer to "I sent you a video": there is no
    canvas to look at and no terminal to check.
    """
    out, skipped = [], []
    dest = Path(dest_dir)
    files = list((event or {}).get("files") or [])
    if len(files) > _MAX_INBOUND_FILES:
        skipped.append(f"only the first {_MAX_INBOUND_FILES} of {len(files)} "
                       "attachments were taken")
        files = files[:_MAX_INBOUND_FILES]
    limit = int(_setting("max_download_mb", 250)) * 1024 * 1024
    for f in files:
        url = f.get("url_private_download") or f.get("url_private")
        name = str(f.get("name") or f.get("id") or "upload")
        if not url:
            skipped.append(f"{name} (Slack gave no download link — the bot may "
                           "lack the files:read scope)")
            continue
        size = int(f.get("size") or 0)
        if size and size > limit:
            skipped.append(f"{name} ({size / 1048576:.0f} MB, over the "
                           f"{limit // 1048576} MB limit)")
            continue
        try:
            import requests
            token = getattr(client, "token", "") or os.environ.get("SLACK_BOT_TOKEN", "")
            dest.mkdir(parents=True, exist_ok=True)
            p = dest / f"{int(time.time())}_{_safe_name(name)}"
            written = 0
            with requests.get(url, headers={"Authorization": f"Bearer {token}"},
                              timeout=120, stream=True) as r:
                r.raise_for_status()
                with p.open("wb") as fh:
                    for chunk in r.iter_content(chunk_size=1 << 20):
                        written += len(chunk)
                        if written > limit:
                            raise ValueError(
                                f"over the {limit // 1048576} MB limit")
                        fh.write(chunk)
            out.append(str(p))
        except Exception as exc:  # noqa: BLE001 — a failed download is not a failed message
            logger.exception("slack: could not download %s", name)
            skipped.append(f"{name} ({exc})")
            try:
                p.unlink(missing_ok=True)   # no half a video left on disk
            except Exception:  # noqa: BLE001
                pass
    return out, skipped


def _safe_name(name: str) -> str:
    """A Slack filename is whatever the sender's phone called it."""
    keep = "".join(c if (c.isalnum() or c in "._- ") else "_" for c in str(name))
    return (keep.strip() or "upload")[:100]


def _why(exc) -> str:
    """Slack's own error code, which is the searchable part of any failure here.

    A ``SlackApiError`` stringifies to a paragraph with the code buried in a dict
    at the end; the code alone is what tells you which of two tokens is wrong.
    """
    code = ""
    try:
        code = str((getattr(exc, "response", None) or {}).get("error") or "")
    except Exception:  # noqa: BLE001
        pass
    return code or f"{type(exc).__name__}: {exc}"


# Slack's codes, in the words of what to do about them. Every one of these has
# cost somebody an evening.
_CONNECT_HINTS = {
    "not_allowed_token_type":
        "SLACK_APP_TOKEN is not an app-level token. It is created separately from "
        "the bot token — Basic Information → App-Level Tokens → Generate Token and "
        "Scopes, with connections:write — and starts with 'xapp-'.",
    "missing_scope":
        "The app-level token exists but lacks the connections:write scope. Add it "
        "to the token (not to the bot scopes) and generate it again.",
    "invalid_auth":
        "SLACK_APP_TOKEN was not accepted at all — it may have been revoked, or "
        "belong to a different app than SLACK_BOT_TOKEN.",
    "account_inactive":
        "The app has been removed from the workspace. Reinstall it.",
}


def _complain(text: str) -> None:
    """Say it in the log AND in the panel.

    A setup mistake here has no other symptom: nothing connects, nothing errors
    where anyone is looking, and "off" and "misconfigured" are indistinguishable
    from the chair.
    """
    logger.warning("slack: %s", text)
    try:
        from src.utils import status_bus
        status_bus.notify("⚠️ " + text)
    except Exception:  # noqa: BLE001
        pass


_BRIDGE: SlackBridge | None = None


def current() -> "SlackBridge | None":
    return _BRIDGE


def start(*, start_turn, answer, interject, downloads_dir=None) -> bool:
    """Connect to Slack and start mirroring. False when it is not set up.

    The three callbacks are how this reaches the pipeline without importing the
    server that owns it (which imports this): ``start_turn(text, files) -> rid``,
    ``answer(request_id, text) -> bool``, ``interject(request_id, text) -> bool``.
    """
    global _BRIDGE
    if not enabled():
        return False
    complaint = token_complaint()
    if complaint:
        # Loud, and on the status bus so it reaches the panel: the bridge being
        # silently absent looks exactly like the bridge being off.
        _complain("Slack bridge not started — " + complaint)
        return False
    try:
        from slack_sdk import WebClient
        # The `builtin` client, deliberately: it is the SDK's dependency-free
        # synchronous one. The `websockets` and `aiohttp` backends are asyncio,
        # and this host has no event loop of its own to run one on.
        from slack_sdk.socket_mode.builtin import SocketModeClient
        from slack_sdk.socket_mode.response import SocketModeResponse
    except Exception as exc:  # noqa: BLE001
        logger.warning("slack: SDK unavailable (%s) — bridge not started", exc)
        return False

    allowed = _env_list("SLACK_ALLOWED_USERS") or list(_setting("allowed_users", []) or [])
    if not allowed:
        _complain("Slack bridge: SLACK_ALLOWED_USERS is empty — it will connect "
                  "but refuse every message. Anyone able to DM the bot could "
                  "otherwise run generations and tools on this machine.")

    web = WebClient(token=os.environ["SLACK_BOT_TOKEN"])
    bridge = SlackBridge(client=web, start_turn=start_turn, answer=answer,
                         interject=interject, allowed_users=allowed)
    try:
        bridge.bot_user_id = str((web.auth_test() or {}).get("user_id") or "")
    except Exception as exc:  # noqa: BLE001
        _complain(f"Slack rejected SLACK_BOT_TOKEN ({_why(exc)}). It is the Bot "
                  "User OAuth Token under OAuth & Permissions, and the app has to "
                  "be installed to the workspace.")
        return False

    # Where the mirror posts: the DM with the first allowed user. A turn started
    # in the panel has no Slack conversation of its own, so it needs somewhere to
    # go, and "the person who is allowed to drive it" is the only right answer.
    channel = str(_setting("channel", "") or "")
    if not channel and allowed:
        try:
            opened = web.conversations_open(users=allowed[0])
            channel = str(((opened or {}).get("channel") or {}).get("id") or "")
        except Exception:  # noqa: BLE001
            logger.exception("slack: could not open a DM with %s", allowed[0])
    bridge.default_channel = channel
    if not channel:
        logger.warning("slack: no channel to post in — mirroring is disabled")

    dest = Path(downloads_dir) if downloads_dir else Path.cwd() / "output" / _DOWNLOAD_DIRNAME
    socket = SocketModeClient(app_token=os.environ["SLACK_APP_TOKEN"], web_client=web)

    def _handle(client, req):  # noqa: ANN001
        try:
            client.send_socket_mode_response(SocketModeResponse(envelope_id=req.envelope_id))
        except Exception:  # noqa: BLE001
            logger.exception("slack: ack failed")
        if str(getattr(req, "type", "")) != "events_api":
            return
        event = ((getattr(req, "payload", None) or {}).get("event") or {})
        if not is_actionable(event):
            return
        # Off the socket thread: routing starts a turn, and a turn is minutes long.
        threading.Thread(
            target=_route_message, args=(bridge, web, event, dest),
            name="agentY-slack-msg", daemon=True).start()

    socket.socket_mode_request_listeners.append(_handle)
    bridge._socket = socket
    bridge.start_worker()
    try:
        socket.connect()
    except Exception as exc:  # noqa: BLE001
        _complain("Slack refused the Socket Mode connection (" + _why(exc) + "). "
                  + _CONNECT_HINTS.get(_why(exc),
                                       "Check that Socket Mode is enabled on the "
                                       "app and that SLACK_APP_TOKEN has the "
                                       "connections:write scope."))
        bridge.stop()
        return False
    _BRIDGE = bridge
    logger.info("slack: connected as %s, posting in %s", bridge.bot_user_id, channel or "(nowhere)")
    return True


def _route_message(bridge: SlackBridge, web, event: dict, dest) -> None:
    user = str(event.get("user") or "")
    text = str(event.get("text") or "")
    channel = str(event.get("channel") or "")
    # The DM the message came from is where the answer belongs, even if the
    # mirror was pointed somewhere else at startup.
    if channel and not bridge.default_channel:
        bridge.default_channel = channel
    files, skipped = ([], [])
    if event.get("files"):
        # Downloaded BEFORE the allow-list is consulted only in the sense that
        # route() is next: an unauthorised sender's files are fetched and then
        # their message is refused. Cheap to reorder if that ever matters.
        files, skipped = download_files(web, event, dest)
    result = bridge.route(user, text, files)
    action = result.get("action")
    if action == "denied":
        bridge.post(channel, "_Not authorised. Ask the owner of this agentY host "
                             "to add your Slack member id to `SLACK_ALLOWED_USERS`._")
    elif action == "interject":
        bridge.post(channel, "_Passed to the turn already running._")
    if skipped and action not in ("denied", "ignored"):
        # Said in Slack, not just logged: from a phone there is no canvas to look
        # at and no terminal to check, so an attachment that quietly did not
        # arrive looks exactly like an agent that ignored it.
        bridge.post(channel, "_Could not take: " + "; ".join(skipped) + "._")
    logger.info("slack: message from %s → %s (%s), %d file(s), %d skipped",
                user, action, result.get("why", ""), len(files), len(skipped))
