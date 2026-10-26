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

Setup (see ``docs/setup/slack.md``)::

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

from src.utils import conversation_store as cs
from src.utils import team_store
from src.utils import turn_queue
from src.utils import turn_bus
from src.utils.slack_render import Post, TurnRender, clip, split_message, to_mrkdwn

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
# How long a message may be, tried in order. Slack documents 40,000 characters,
# but chat.update refused a ~3,500-character answer with msg_too_long
# (2026-09-15), so a refused message is sent again, shorter, until it fits.
_MSG_LIMITS = (39000, 3000, 1200)


def _env_list(name: str) -> list:
    return [p.strip() for p in (os.environ.get(name) or "").split(",") if p.strip()]


# Where a message can come from. "im" is a DM (the original mode); the others are
# channels, which only answer an @mention.
CHANNEL_TYPES = frozenset({"channel", "group", "mpim"})

_YES_WORDS = frozenset({"yes", "y", "yep", "yeah", "ok", "okay", "approve", "approved",
                        "allow", "go", "sure", "confirm", "do it", "go ahead", "+1",
                        "👍", ":+1:", ":thumbsup:"})
_NO_WORDS = frozenset({"no", "n", "nope", "deny", "denied", "reject", "stop", "cancel",
                       "dont", "don't", "-1", "👎", ":-1:", ":thumbsdown:"})


def parse_decision(text: str):
    """True for a yes, False for a no, None when the message is neither.

    A question to the agent that happens to arrive while an approval is open must
    not be mistaken for an answer to it, so only a bare yes/no counts.
    """
    word = " ".join(str(text or "").lower().replace("!", " ").replace(".", " ").split())
    if word in _YES_WORDS:
        return True
    if word in _NO_WORDS:
        return False
    return None


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


def _setting_list(key: str, env: str) -> list:
    """A list from settings, plus the comma-separated environment variable."""
    out = [str(x).strip() for x in (_setting(key, []) or []) if str(x).strip()]
    for item in _env_list(env):
        if item not in out:
            out.append(item)
    return out


def _setting(key: str, default):
    try:
        from src.utils.settings import load_settings
        return (load_settings().get("slack") or {}).get(key, default)
    except Exception:  # noqa: BLE001 — settings must never break the bridge
        return default


class _Approval:
    """A yes/no a tool is waiting for in a Slack thread."""
    _n = 0

    def __init__(self, thread_id: str):
        _Approval._n += 1
        self.id = _Approval._n
        self.thread_id = str(thread_id)
        self.event = threading.Event()
        self.allowed = False
        self.by = ""


class _Keyed:
    """The Slack message ids one turn is using."""

    def __init__(self):
        self.answer_ts = ""
        self.by_key: dict = {}


class SlackTurn:
    """One turn, as it appears in Slack: a few replies inside its conversation.

    The Slack thread belongs to the CONVERSATION, not to the turn — one thread
    per agentY conversation, the way the side panel has one pane per chat. So a
    turn is at most three replies in it: the answer, the working-out, and the
    transient status line. Anything that must be *seen* rather than found is
    broadcast, which shows it in the DM while leaving it in the thread it
    belongs to.
    """

    def __init__(self, bridge, turn, channel: str, root_ts: str = "", mention: str = ""):
        self.bridge = bridge
        self.turn = turn
        self.channel = channel
        self.root_ts = root_ts
        # A turn asked for in a channel names its requester when it needs them
        # ("<@U123> …"), and never broadcasts: a team channel is not one person's DM.
        self.mention = mention
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
            self.bridge._call(self.bridge.finish_ack, self.turn.request_id)
            self.ended = time.time()
            self.ask_request_id = ""

    def _apply(self, post: Post) -> None:
        """Runs on the worker thread — every Slack id in here is single-threaded.

        Including the conversation's root message: creating it is a network call,
        and the one thing this bridge must never do is make one on the turn's own
        thread.
        """
        if not self.root_ts:
            # Which conversation this belongs to, created on its first word —
            # including for a turn started at the desk, so work from the panel
            # shows up under its own heading rather than loose in the DM.
            self.root_ts = self.bridge.conversation_root(
                self.turn.thread_id, self.turn.text)
            if not self.root_ts:
                return
        if post.where == "answer":
            self._pending_answer = post.text
            self._flush_answer()
            return
        if post.kind == "file":
            self.bridge._do_upload(self.channel, post.path, post.text,
                                   thread_ts=self.root_ts)
            return
        if post.where == "channel":
            if self.mention:
                self.bridge.post(self.channel, f"{self.mention} {post.text}",
                                 thread_ts=self.root_ts)
                return
            # Broadcast: in the DM where it will be noticed, and still filed
            # under the conversation it came from.
            self.bridge.post(self.channel, post.text, thread_ts=self.root_ts,
                             broadcast=True)
            return
        thread_ts = self.root_ts
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
            self.ids.answer_ts = self.bridge.post(self.channel, text,
                                                  thread_ts=self.root_ts)

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
                 allowed_users=None, default_channel: str = "",
                 allowed_channels=None, approvers=None, turns_queue=None,
                 members_may_ask=None, queue_dm=None):
        self.client = client
        self._start_turn = start_turn
        self._answer = answer
        self._interject = interject
        self.allowed = list(allowed_users if allowed_users is not None
                            else _env_list("SLACK_ALLOWED_USERS"))
        self.default_channel = default_channel
        # Channels (docs/setup/slack.md): where an @mention is answered, who besides the
        # requester may approve, and the line turns wait in. All empty by default,
        # which is the DM-only bridge it has always been.
        self.allowed_channels = list(allowed_channels if allowed_channels is not None
                                     else _setting_list("channels", "SLACK_ALLOWED_CHANNELS"))
        self.approvers = list(approvers if approvers is not None
                              else _setting_list("approvers", "SLACK_APPROVERS"))
        self.members_may_ask = bool(_setting("channel_members_may_ask", False)
                                    if members_may_ask is None else members_may_ask)
        self.queue = turns_queue
        self.queue_dm = bool(_setting("queue_dm_turns", False) if queue_dm is None else queue_dm)
        self._approvals: dict = {}       # approval id -> _Approval
        self._seen: dict = {}            # (channel, ts) -> time: Slack retries events
        self._names: dict = {}           # member id -> display name
        self._waiting_roots: dict = {}   # (channel, root_ts) -> queue Ticket
        self.turns: dict = {}            # request_id -> SlackTurn
        self.acks: dict = {}             # request_id -> (channel, ts) of the message it answers
        self._root_text_cache: dict = {}  # thread_id -> its root message text
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

    def post(self, channel: str, text: str, thread_ts: str = "",
             broadcast: bool = False) -> str:
        """Post and return the message ts. Synchronous — the ts is needed to edit
        it later, and losing it means the next edit posts a duplicate instead.

        ``broadcast`` is Slack's "also send to channel": the message stays in the
        conversation's thread but shows in the DM, which is what a question or an
        error needs and what an ordinary progress line must never do.
        """
        if not self.client or not text.strip():
            return ""
        kw = {"channel": channel}
        if thread_ts:
            kw["thread_ts"] = thread_ts
            if broadcast:
                kw["reply_broadcast"] = True
        try:
            resp = self._fitting(self.client.chat_postMessage, text, **kw)
            return str((resp or {}).get("ts") or "")
        except Exception:  # noqa: BLE001
            logger.exception("slack: chat_postMessage failed")
            return ""

    def _do_update(self, channel: str, ts: str, text: str) -> None:
        if not self.client or not ts:
            return
        self._fitting(self.client.chat_update, text, channel=channel, ts=ts)

    @staticmethod
    def _fitting(send, text: str, **kw):
        """Send *text*, cut shorter each time Slack answers msg_too_long."""
        for limit in _MSG_LIMITS:
            try:
                return send(text=clip(text, limit), **kw)
            except Exception as exc:  # noqa: BLE001
                if "msg_too_long" not in str(exc) or limit == _MSG_LIMITS[-1]:
                    raise
        return None

    def _do_delete(self, channel: str, ts: str) -> None:
        if not self.client or not ts:
            return
        try:
            self.client.chat_delete(channel=channel, ts=ts)
        except Exception:  # noqa: BLE001 — an already-gone message is not an error
            logger.debug("slack: chat_delete(%s) failed", ts, exc_info=True)

    def _do_upload(self, channel: str, path: str, caption: str = "",
                   thread_ts: str = "") -> None:
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
        kw = {"channel": channel, "file": str(p), "filename": p.name,
              "initial_comment": caption or None}
        if thread_ts:
            kw["thread_ts"] = thread_ts
        self.client.files_upload_v2(**kw)

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
        if self.queue is not None:
            try:
                self.queue.sweep([t.request_id for t in turn_bus.active()])
            except Exception:  # noqa: BLE001
                logger.exception("slack: queue sweep failed")
        with self._lock:
            turns = list(self.turns.items())
        for rid, st in turns:
            # Unguarded, one refused edit ended the worker thread, and with it
            # every Slack message until the host was restarted (2026-09-15).
            try:
                st.tick()
            except Exception:  # noqa: BLE001
                logger.exception("slack: a throttled edit failed")
            if st.ended and now - st.ended > _TURN_TTL:
                with self._lock:
                    self.turns.pop(rid, None)

    # ── one conversation, one thread ──────────────────────────────────────────
    def conversation_root(self, thread_id: str, seed: str = "") -> str:
        """The ts of the message that IS this conversation in Slack.

        Created the first time the conversation says anything, and remembered in
        the store — a restart that forgot it would open a second thread for a
        conversation that already has one, and nothing afterwards could tell the
        two apart.
        """
        try:
            bound = cs.get_slack_thread(thread_id)
        except Exception:  # noqa: BLE001
            bound = None
        if (bound and bound.get("root_ts") and bound.get("channel") != self.default_channel
                and bound.get("channel") in self.allowed_channels):
            # Asked for in a channel: the thread is the requester's own message,
            # which is not ours to retitle.
            return str(bound["root_ts"])
        if not self.default_channel:
            return ""
        if bound and bound.get("channel") == self.default_channel and bound.get("root_ts"):
            self._retitle(thread_id, bound["root_ts"])
            return str(bound["root_ts"])
        text = self._root_text(thread_id, seed)
        ts = self.post(self.default_channel, text)
        if not ts:
            return ""
        self._root_text_cache[thread_id] = text
        try:
            cs.set_slack_thread(thread_id, self.default_channel, ts)
        except Exception:  # noqa: BLE001
            logger.exception("slack: could not remember the thread for %s", thread_id)
        return ts

    def _root_text(self, thread_id: str, seed: str = "") -> str:
        """What the conversation is called, as its thread's opening message."""
        title = ""
        try:
            title = next((str(r.get("title") or "") for r in cs.list_threads(limit=200)
                          if r.get("id") == thread_id), "")
        except Exception:  # noqa: BLE001
            pass
        if title.strip().lower() in ("", "new chat", "slack"):
            title = clip(" ".join(str(seed or "").split()), 80) or "New chat"
        return "🧵 *" + to_mrkdwn(clip(title, 120)) + "*"

    def _retitle(self, thread_id: str, root_ts: str) -> None:
        """Follow the auto-title, which lands a moment after the first turn.

        Only when it actually changed: this is asked on every turn, and an edit
        per turn is a round trip and a rate-limit bucket for no new information.
        """
        text = self._root_text(thread_id)
        if self._root_text_cache.get(thread_id) == text:
            return
        self._root_text_cache[thread_id] = text
        self._call(self._do_update, self.default_channel, root_ts, text)

    # ── watching every turn ───────────────────────────────────────────────────
    def on_turn_event(self, event: dict, turn) -> None:
        """Registered on the turn bus. Runs on the turn's thread — stays cheap."""
        if str(event.get("type")) == "done" and self.queue is not None:
            # The next waiting request may start. Before anything that can return:
            # a turn that never showed anything still held a place in line.
            self.queue.finished(turn.request_id)
        with self._lock:
            st = self.turns.get(turn.request_id)
            if st is None:
                if str(event.get("type")) == "done":
                    return       # nothing was ever shown; nothing to close
                channel, root, mention = self._where(turn.thread_id)
                if not channel:
                    return
                st = SlackTurn(self, turn, channel, root_ts=root, mention=mention)
                self.turns[turn.request_id] = st
        st.feed(event)

    def _where(self, thread_id: str):
        """``(channel, root_ts, mention)`` a conversation speaks in.

        A conversation asked for in a channel has its own thread there. Every
        other turn - the panel's, a DM's - goes to the DM mirror, as it always did.
        """
        try:
            bound = cs.get_slack_thread(thread_id)
        except Exception:  # noqa: BLE001
            bound = None
        if (bound and bound.get("root_ts") and bound.get("channel") != self.default_channel
                and bound.get("channel") in self.allowed_channels):
            who = team_store.requester(thread_id)
            return (str(bound["channel"]), str(bound["root_ts"]),
                    f"<@{who['user']}>" if who.get("user") else "")
        return self.default_channel, "", ""

    def _binding(self, thread_id: str):
        """``(channel, root_ts)`` of the Slack thread this conversation lives in."""
        try:
            bound = cs.get_slack_thread(thread_id)
        except Exception:  # noqa: BLE001
            bound = None
        if bound and bound.get("root_ts"):
            return str(bound["channel"]), str(bound["root_ts"])
        return None

    # ── 👀 / ✅ on the message itself ──────────────────────────────────────────
    # The fastest sign of life Slack has: one call, made the moment a message
    # arrives — before attachments are fetched, before the turn starts, before
    # the canvas is asked for. "working on it" follows once the turn is up.
    ACK_SEEN = "eyes"
    ACK_DONE = "white_check_mark"

    def react(self, channel: str, ts: str, name: str, remove: bool = False) -> None:
        if not self.client or not channel or not ts:
            return
        try:
            if remove:
                self.client.reactions_remove(channel=channel, timestamp=ts, name=name)
            else:
                self.client.reactions_add(channel=channel, timestamp=ts, name=name)
        except Exception as exc:  # noqa: BLE001 — a missing scope or an old message
            logger.debug("slack: reaction %s %s failed: %s", name, ts, exc)

    def acknowledge(self, channel: str, ts: str) -> None:
        """Seen. Synchronous on purpose: it is the whole point to be first."""
        self.react(channel, ts, self.ACK_SEEN)

    def bind_ack(self, request_id: str, channel: str, ts: str) -> None:
        """The message *ts* started (or fed) turn *request_id*: tick it when that ends."""
        if request_id and ts:
            with self._lock:
                self.acks[request_id] = (channel, ts)

    def finish_ack(self, request_id: str) -> None:
        """👀 → ✅ on every message that turn answered. Runs on the worker."""
        with self._lock:
            hits = [v for k, v in self.acks.items() if k == request_id]
            self.acks.pop(request_id, None)
        for channel, ts in hits:
            self.react(channel, ts, self.ACK_SEEN, remove=True)
            self.react(channel, ts, self.ACK_DONE)

    def drop_ack(self, channel: str, ts: str) -> None:
        """Seen, and nothing will come of it here (busy, refused, ignored)."""
        self.react(channel, ts, self.ACK_SEEN, remove=True)

    # ── the agent handing something over on purpose ───────────────────────────
    def send_files(self, paths: list, message: str = "", thread_id: str = "") -> dict:
        """Put files in the DM deliberately, rather than because a run made them.

        The mirror uploads what a run *produced*. This is for everything else the
        agent decides is worth having in your hand: the JSON it just wrote, one
        chosen frame out of sixty, a script, a log it wants you to look at.

        Into the Slack thread of the conversation *thread_id* — the one the request
        came from — like everything else that conversation says. Posted at the top
        of the DM, a screenshot asked for in a thread opened a thread of its own.

        Every file is checked here rather than in the worker, so the agent is told
        what it actually sent while it can still say so in the same breath — a
        report that arrives after the turn is a report nobody reads.
        """
        if not self.client:
            return {"error": "the Slack bridge is not connected."}
        # A conversation asked for in a channel answers in that channel's thread,
        # never in the DM: files meant for a team do not go to one person.
        channel, root, _mention = self._where(thread_id) if thread_id else (self.default_channel, "", "")
        if not channel:
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
        if thread_id and not root and (sent or message.strip()):
            try:
                root = self.conversation_root(thread_id)
            except Exception:  # noqa: BLE001 — the DM's top level beats not sending
                logger.exception("slack: no thread for %s; sending to the DM", thread_id)
        if message.strip():
            for piece in split_message(message, 3000):
                self._call(self.post, channel, to_mrkdwn(piece), thread_ts=root)
        for p in sent:
            self._call(self._do_upload, channel, str(p), "", thread_ts=root)
        return {"sent": [str(p) for p in sent], "missing": missing,
                "too_large": too_large}

    # ── inbound ───────────────────────────────────────────────────────────────
    def route(self, user: str, text: str, files: list | None = None,
              thread_ts: str = "", *, channel: str = "", channel_type: str = "im",
              mentioned: bool = False, msg_ts: str = "", user_name: str = "") -> dict:
        """What this message means, and what was done about it.

        Returns ``{"action": …}`` — ``answer``/``interject``/``turn``/``busy``/
        ``denied``/``ignored``, and for channels ``queued``/``amended``/
        ``command``/``approval`` — which is what the tests assert on and what the
        log records.

        *thread_ts* is the Slack thread the message was posted in, and it is what
        picks the conversation: inside a thread, this continues the conversation
        that thread IS; at the top level, it starts a new one.

        In a channel (*channel_type* ``channel``/``group``/``mpim``) the message
        must be an @mention (*mentioned*), the channel must be one of
        ``slack.channels``, and the conversation's thread is the requester's own
        message (*msg_ts*), so every request has its thread.
        """
        text = self.strip_mention(text)
        files = list(files or [])
        if not text and not files:
            return {"action": "ignored", "why": "empty"}
        if user and self.bot_user_id and user == self.bot_user_id:
            return {"action": "ignored", "why": "own message"}
        in_channel = channel_type in CHANNEL_TYPES
        if in_channel:
            if channel not in self.allowed_channels:
                return {"action": "ignored", "why": "channel not in slack.channels"}
            # An approver may answer for a request without being allowed to make
            # one; the check that stops them starting a turn comes after the
            # approval and answer handling below.
            if not self.allowed and not self.members_may_ask and user not in self.approvers:
                logger.warning("slack: a channel message arrived but SLACK_ALLOWED_USERS "
                               "is empty - refusing to act on it")
                return {"action": "denied", "why": "no allow-list configured"}
            if (user not in self.allowed and not self.members_may_ask
                    and user not in self.approvers):
                return {"action": "denied", "why": "not in SLACK_ALLOWED_USERS"}
        else:
            if not self.allowed:
                logger.warning("slack: a message arrived but SLACK_ALLOWED_USERS is "
                               "empty — refusing to act on it")
                return {"action": "denied", "why": "no allow-list configured"}
            if user not in self.allowed:
                return {"action": "denied", "why": "not in SLACK_ALLOWED_USERS"}

        where = channel if in_channel else self.default_channel
        target = self.conversation_for(thread_ts, channel=where)

        # An approval the agent is waiting on in this thread (a tool that needs a
        # yes). Only a bare yes/no counts; anyone else's reply is not an approval.
        appr = self._pending_approval(target) if target else None
        if appr is not None and text:
            if not self.may_steer(user, target):
                return {"action": "denied", "thread_id": target,
                        "why": "only the requester or an approver may answer this"}
            verdict = parse_decision(text)
            if verdict is None:
                return {"action": "approval_unclear", "thread_id": target}
            appr.allowed = verdict
            appr.by = user
            appr.event.set()
            return {"action": "approval", "allowed": verdict, "thread_id": target}

        if in_channel:
            if not mentioned and not (target and self._pending_ask(target)):
                return {"action": "ignored", "why": "not an @mention"}
            if (user not in self.allowed and not self.members_may_ask
                    and not (target and self._pending_ask(target))):
                return {"action": "denied", "why": "not in SLACK_ALLOWED_USERS"}
            cmd = self._command(user, text, channel, user_name)
            if cmd is not None:
                return {"action": "command", "reply": cmd}

        # A question the agent is holding a turn open for — but only the one in
        # THIS conversation. Answering with a message meant for another chat is
        # worse than not answering at all.
        pending = self._pending_ask(target)
        if pending is not None and text:
            rid, st = pending
            if not self.may_steer(user, target):
                return {"action": "denied", "thread_id": target,
                        "why": "only the requester or an approver may answer this"}
            if self._answer and self._answer(rid, text):
                st.ask_request_id = ""
                return {"action": "answer", "request_id": rid,
                        "thread_id": target}

        if self.queue is not None and (in_channel or self.queue_dm):
            return self._route_queued(user, text, files, target, thread_ts, channel,
                                      msg_ts, in_channel, user_name)

        live = self._live_turn()
        if live is not None and text:
            rid, running_thread = live
            # Steering the running turn is for a message written INTO it — a
            # reply in its own thread. Anything else is meant for a different
            # conversation (a top-level message starts a new one), and there is
            # one pipeline: interjecting would put the message in a chat it was
            # not written for, and starting a turn would run two at once.
            if not (target and running_thread and running_thread == target):
                return {"action": "busy",
                        "why": ("a turn is running in another conversation"
                                if target else
                                "a turn is running; a new conversation has to wait"),
                        "running_thread": running_thread}
            if self._interject and self._interject(rid, text):
                return {"action": "interject", "request_id": rid,
                        "thread_id": running_thread}

        if self._start_turn is None:
            return {"action": "ignored", "why": "no turn starter wired"}
        rid = self._start_turn(text, files, target)
        return {"action": "turn", "request_id": rid, "thread_id": target}

    def _route_queued(self, user, text, files, target, thread_ts, channel, msg_ts,
                      in_channel, user_name) -> dict:
        """The queue path: steer a running turn of THIS conversation, add to a
        waiting one, or take a place in line. Several conversations may run at the
        same time (up to ``server.max_running_turns``); the rest wait."""
        if self._start_turn is None:
            return {"action": "ignored", "why": "no turn starter wired"}
        rid_live = self._turn_in(target)
        if rid_live and text:
            if not self.may_steer(user, target):
                return {"action": "denied", "thread_id": target,
                        "why": "only the requester or an approver may steer this turn"}
            if self._interject and self._interject(rid_live, text):
                return {"action": "interject", "request_id": rid_live, "thread_id": target}
        notify_channel = channel if in_channel else self.default_channel
        root = (thread_ts or msg_ts) if in_channel else ""
        key = (notify_channel, root)
        waiting = self.queue.waiting_in(target) if target else self._waiting_roots.get(key)
        if waiting is not None and text:
            if waiting.user and waiting.user != user and not self.may_steer(user, target):
                return {"action": "denied", "why": "that request belongs to someone else"}
            waiting.extra["amended"] = (waiting.extra.get("amended", "") + "\n\n" + text).strip()
            return {"action": "amended", "position": self.queue.position(waiting),
                    "thread_id": waiting.thread_id}

        extra: dict = {}
        if in_channel:
            extra["requester"] = {"user": user, "channel": channel, "project": channel,
                                  "name": user_name}
            extra["slack_root"] = (channel, root)
        holder: dict = {}

        def _starter():
            ticket = holder.get("ticket")
            body = text
            if ticket is not None and ticket.extra.get("amended"):
                body = (text + "\n\n" + ticket.extra["amended"]).strip()
            return self._start_turn(body, files, target, **extra)

        try:
            ticket = self.queue.submit(
                _starter, text=text, user=user, thread_id=target, label="slack",
                extra={"notify": (notify_channel, root or thread_ts or msg_ts), "key": key})
        except turn_queue.QueueFull:
            return {"action": "busy", "why": "the queue is full - try again in a few minutes"}
        holder["ticket"] = ticket
        if ticket.state == turn_queue.QUEUED and not target:
            self._waiting_roots[key] = ticket
        return {"action": "turn" if ticket.state != turn_queue.QUEUED else "queued",
                "request_id": ticket.request_id, "thread_id": ticket.thread_id or target,
                "position": self.queue.position(ticket)}

    def conversation_for(self, thread_ts: str, channel: str = "") -> str:
        """Which agentY conversation a message in *thread_ts* belongs to.

        "" means a new one: a message at the top level of the DM starts a fresh
        conversation, the way a new chat does in the panel. Replying inside a
        thread is how you go back to one.
        """
        ts = str(thread_ts or "").strip()
        if not ts:
            return ""
        try:
            return cs.thread_for_slack(channel or self.default_channel, ts) or ""
        except Exception:  # noqa: BLE001
            logger.exception("slack: could not resolve the thread %s", ts)
            return ""

    # ── who may steer, who may approve ────────────────────────────────────────
    def may_steer(self, user: str, thread_id: str) -> bool:
        """May *user* answer, approve or interject in this conversation?

        The person who asked, and anyone configured as an approver (settings
        ``slack.approvers`` / ``SLACK_APPROVERS``, or a user with the ``approver``
        role in config/users.json). A conversation nobody is recorded against - a
        DM, a panel turn - answers to everyone on the allow-list, as it always did.
        """
        who = team_store.requester(thread_id) if thread_id else {}
        if not who.get("user"):
            return True
        if user == who["user"] or user in self.approvers:
            return True
        try:
            from src.utils import user_auth
            return "approver" in user_auth.slack_roles().get(user, [])
        except Exception:  # noqa: BLE001
            return False

    def strip_mention(self, text: str) -> str:
        """*text* without the bot's own ``<@U…>``, which is how a request starts."""
        text = str(text or "")
        if self.bot_user_id:
            text = text.replace(f"<@{self.bot_user_id}>", " ")
        return text.strip()

    def seen(self, channel: str, ts: str) -> bool:
        """True the second time an event is offered: Slack redelivers, and an
        @mention can arrive as both ``app_mention`` and ``message``."""
        key = (channel, ts)
        now = time.time()
        with self._lock:
            if key in self._seen:
                return True
            self._seen[key] = now
            if len(self._seen) > 500:
                for k in [k for k, t in self._seen.items() if now - t > 600]:
                    self._seen.pop(k, None)
        return False

    def display_name(self, user: str) -> str:
        if user in self._names:
            return self._names[user]
        name = ""
        if self.client is not None and hasattr(self.client, "users_info"):
            try:
                prof = ((self.client.users_info(user=user) or {}).get("user") or {})
                name = str(prof.get("real_name") or prof.get("name") or "")
            except Exception:  # noqa: BLE001 - the id is name enough
                logger.debug("slack: users_info(%s) failed", user, exc_info=True)
        self._names[user] = name
        return name

    def wants_message(self, event: dict) -> bool:
        """Is this message one the bridge will act on (so it is worth a 👀)?

        A DM always is. A channel message is, if it is an @mention, or a reply in
        a thread of ours where something is waiting for an answer.
        """
        if str(event.get("channel_type") or "") not in CHANNEL_TYPES:
            return True
        if event.get("_mentioned"):
            return True
        channel = str(event.get("channel") or "")
        thread_ts = str(event.get("thread_ts") or "")
        target = self.conversation_for(thread_ts, channel=channel) if thread_ts else ""
        return bool(target and (self._pending_approval(target) or self._pending_ask(target)))

    # ── commands that need no agent ───────────────────────────────────────────
    def _command(self, user: str, text: str, channel: str, user_name: str = ""):
        """A few words answered by the bridge itself, or None if this is a request.

        Written with a colon (``remember: …``) so an ordinary request that merely
        starts with the word is not swallowed.
        """
        low = text.strip().lower()
        project = channel
        if low.startswith("remember for me:") or low.startswith("remember:"):
            body = text.split(":", 1)[1].strip()
            team_store.add_note("users", user, body)
            return f"Noted for you: _{body}_"
        if low.startswith("remember for project:") or low.startswith("remember for channel:"):
            body = text.split(":", 1)[1].strip()
            team_store.add_note("projects", project, body)
            return f"Noted for this channel: _{body}_"
        if low in ("notes", "my notes"):
            mine = team_store.read_notes("users", user) or "_none_"
            proj = team_store.read_notes("projects", project) or "_none_"
            return f"*Your notes*\n{mine}\n\n*This channel's notes*\n{proj}"
        if low in ("forget my notes", "forget notes"):
            team_store.clear_notes("users", user)
            return "Your notes are cleared."
        if low in ("forget channel notes", "forget project notes"):
            if user not in self.approvers and user not in self.allowed:
                return "Only an approver can clear the channel's notes."
            team_store.clear_notes("projects", project)
            return "This channel's notes are cleared."
        if low in ("queue", "status"):
            return self.queue_text()
        if low in ("usage", "costs", "cost"):
            if user not in self.approvers and user not in self.allowed:
                return "Usage is shown to approvers."
            rows = team_store.usage_summary("user", since=time.time() - 30 * 86400)
            if not rows:
                return "No usage recorded in the last 30 days."
            return "*Last 30 days*\n" + "\n".join(
                f"• <@{r['user']}>: {r['turns']} turns, "
                f"{('$%.2f' % r['cost']) if r['cost'] else 'cost unknown'}"
                for r in rows[:15])
        return None

    def queue_text(self) -> str:
        if self.queue is None:
            return "No queue is configured."
        snap = self.queue.snapshot()
        lines = [f"*Running {len(snap['running'])}/{snap['limit']}*"]
        for t in snap["running"]:
            lines.append(f"• ▶️ <@{t['user']}> {t['text']}" if t["user"] else f"• ▶️ {t['text']}")
        for t in snap["queued"]:
            lines.append(f"• {t['position']}. ⏳ <@{t['user']}> {t['text']}" if t["user"]
                         else f"• {t['position']}. ⏳ {t['text']}")
        if len(lines) == 1:
            lines.append("Nothing is waiting.")
        return "\n".join(lines)

    # ── a place in line, said in the thread ───────────────────────────────────
    def on_queue_change(self, ticket) -> None:
        """Queue listener: keep one message in the request's thread current."""
        if self.client is None or "notify" not in ticket.extra:
            return
        self._call(self._queue_note, ticket)

    def _queue_note(self, ticket) -> None:
        channel, thread_ts = ticket.extra.get("notify") or ("", "")
        if not channel:
            return
        note = ticket.extra.get("note_ts", "")
        if ticket.state == turn_queue.QUEUED:
            pos = self.queue.position(ticket)
            if pos == ticket.extra.get("note_pos"):
                return
            ticket.extra["note_pos"] = pos
            snap = self.queue.snapshot()
            text = (f"⏳ Queued, number {pos} in line "
                    f"({len(snap['running'])} running, up to {snap['limit']} at a time). "
                    "I will start as soon as a place is free.")
            if note:
                self._do_update(channel, note, text)
            else:
                ticket.extra["note_ts"] = self.post(channel, text, thread_ts=thread_ts)
        else:
            if note:
                self._do_delete(channel, note)
                ticket.extra["note_ts"] = ""
            self._waiting_roots.pop(ticket.extra.get("key"), None)

    # ── approvals asked in the thread ─────────────────────────────────────────
    def _pending_approval(self, thread_id: str):
        with self._lock:
            for ap in self._approvals.values():
                if ap.thread_id == str(thread_id) and not ap.event.is_set():
                    return ap
        return None

    def ask_approval(self, thread_id: str, tool: str, summary: str, timeout: float):
        """Ask for a yes in the conversation's Slack thread; block for the answer.

        Registered with :mod:`tool_permissions` as the remote asker. Returns
        ``{"allowed", "note"}``, or None when this conversation is not one Slack
        can ask in (then the panel, or the unattended policy, decides as before).

        A conversation asked for in a channel is ALWAYS asked here, even with a
        panel open on the host: the person at the host's screen is not the person
        who asked.
        """
        binding = self._binding(thread_id) if thread_id else None
        if not binding or self.client is None:
            return None
        who = team_store.requester(thread_id)
        if not who.get("user"):
            try:
                from src.utils import tool_permissions
                if tool_permissions.has_listener():
                    return None          # a panel is open: it asks, as before
            except Exception:  # noqa: BLE001
                pass
        channel, root = binding
        mention = f"<@{who['user']}> " if who.get("user") else ""
        ap = _Approval(thread_id)
        with self._lock:
            self._approvals[ap.id] = ap
        try:
            self.post(channel,
                      f"🔐 {mention}*Approval needed* — `{tool}`\n"
                      f"{to_mrkdwn(clip(summary, 1500))}\n"
                      "_Reply *yes* or *no* in this thread"
                      + (" (you or an approver)" if who.get("user") else "") + "._",
                      thread_ts=root)
            if not ap.event.wait(max(1.0, float(timeout or 0))):
                self.post(channel, f"_No answer in {int(timeout)}s, so `{tool}` was not run._",
                          thread_ts=root)
                return {"allowed": False,
                        "note": f"nobody approved this in Slack within {int(timeout)}s."}
            if ap.allowed:
                return {"allowed": True, "note": f"approved by {ap.by}"}
            return {"allowed": False, "note": f"declined by {ap.by} in Slack."}
        finally:
            with self._lock:
                self._approvals.pop(ap.id, None)

    def _turn_in(self, thread_id: str) -> str:
        """The request id of the running turn of *thread_id*, or ""."""
        if not thread_id:
            return ""
        for t in turn_bus.active():
            if str(t.thread_id) == str(thread_id):
                return t.request_id
        return ""

    def _pending_ask(self, thread_id: str = ""):
        """The unanswered question in *thread_id*, or in any conversation when
        the message came in at the top level and names none."""
        with self._lock:
            for _rid, st in self.turns.items():
                if not st.ask_request_id:
                    continue
                if thread_id and str(st.turn.thread_id) != str(thread_id):
                    continue
                return st.ask_request_id, st
        return None

    def _live_turn(self):
        """``(request_id, thread_id)`` of the turn in flight, or None."""
        running = {t.request_id: t.thread_id for t in turn_bus.active()}
        with self._lock:
            for rid, st in self.turns.items():
                if rid in running and not st.ended:
                    return rid, str(running[rid] or "")
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
        if self.queue is not None:
            self.queue.remove_listener(self.on_queue_change)
        try:
            team_store.stop_recording()
            from src.utils import tool_permissions
            tool_permissions.set_remote_asker(None)
        except Exception:  # noqa: BLE001
            pass
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
    if str((event or {}).get("type")) == "app_mention":
        # An @mention in a channel. Whether the channel and the person are allowed
        # is the bridge's call (route), not the parser's.
        return bool(str(event.get("user") or "")) and not event.get("bot_id")
    if str((event or {}).get("type")) != "message":
        return False
    subtype = str(event.get("subtype") or "")
    if subtype and subtype not in _ACTIONABLE_SUBTYPES:
        return False
    if event.get("bot_id"):
        return False
    if str(event.get("channel_type") or "") not in ({"im"} | CHANNEL_TYPES):
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

    # One list: SLACK_ALLOWED_USERS (Settings > Slack). The old settings key
    # slack.allowed_users is still honoured when the variable is empty, so no setup
    # breaks, but it is off the settings page: two lists where only one counts is how
    # someone gets removed from the wrong one and stays allowed.
    allowed = _env_list("SLACK_ALLOWED_USERS")
    legacy = list(_setting("allowed_users", []) or [])
    if legacy:
        logger.warning("slack: settings key slack.allowed_users is deprecated%s - put the "
                       "member ids in SLACK_ALLOWED_USERS (Settings > Slack) and delete it",
                       "" if not allowed else " and IGNORED (SLACK_ALLOWED_USERS is set)")
        if not allowed:
            allowed = legacy
    if not allowed and not (_setting("channel_members_may_ask", False)
                            and _setting_list("channels", "SLACK_ALLOWED_CHANNELS")):
        _complain("Slack bridge: SLACK_ALLOWED_USERS is empty — it will connect "
                  "but refuse every message. Anyone able to DM the bot could "
                  "otherwise run generations and tools on this machine.")

    web = WebClient(token=os.environ["SLACK_BOT_TOKEN"])
    bridge = SlackBridge(client=web, start_turn=start_turn, answer=answer,
                         interject=interject, allowed_users=allowed)
    # The line turns wait in, and who is told where they stand. One queue for the
    # host (the turn API waits in it too), so one limit covers everyone.
    bridge.queue = turn_queue.shared()
    bridge.queue.add_listener(bridge.on_queue_change)
    # Who asked and what it cost, written as turns end (only turns somebody is
    # attributed to), and approvals asked in the thread instead of on a panel.
    team_store.start_recording()
    try:
        from src.utils import tool_permissions
        tool_permissions.set_remote_asker(
            lambda thread_id, tool, summary, timeout:
            bridge.ask_approval(thread_id, tool, summary,
                                float(_setting("approval_timeout_seconds", 600) or 600)))
    except Exception:  # noqa: BLE001
        logger.exception("slack: could not register the approval asker")
    if bridge.allowed_channels:
        logger.info("slack: answering @mentions in %d channel(s)", len(bridge.allowed_channels))
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
    channel_type = str(event.get("channel_type")
                       or ("channel" if event.get("type") == "app_mention" else "im"))
    in_channel = channel_type in CHANNEL_TYPES
    msg_ts = str(event.get("ts") or "")
    thread_ts = str(event.get("thread_ts") or "")
    if thread_ts == msg_ts:
        thread_ts = ""          # a thread's own root message is not a reply
    if in_channel:
        # An @mention arrives as an `app_mention` and, with message.channels
        # subscribed, again as a `message`; Slack also redelivers. Act once.
        if bridge.seen(channel, msg_ts):
            return
        event["_mentioned"] = (str(event.get("type")) == "app_mention"
                               or bool(bridge.bot_user_id
                                       and f"<@{bridge.bot_user_id}>" in text))
        if not bridge.wants_message(event):
            return
    else:
        # The DM the message came from is where the answer belongs, even if the
        # mirror was pointed somewhere else at startup.
        if channel and not bridge.default_channel:
            bridge.default_channel = channel
    # A channel request is answered in a thread of its own: the requester's
    # message is the root, or the thread it was written in.
    reply_ts = (thread_ts or msg_ts) if in_channel else thread_ts
    permitted = (user in bridge.allowed
                 or (in_channel and bridge.members_may_ask
                     and channel in bridge.allowed_channels))
    # Seen — first, before anything that takes time. Only for people the bridge
    # answers: a stranger's message gets the refusal, not a reaction.
    acked = bool(user and permitted and user != bridge.bot_user_id
                 and (text.strip() or event.get("files")))
    if acked:
        bridge.acknowledge(channel, msg_ts)
    files, skipped = ([], [])
    if in_channel and event.get("_mentioned") and not event.get("files"):
        event["files"] = _fetch_files(web, event)
    if event.get("files"):
        # Downloaded BEFORE the allow-list is consulted only in the sense that
        # route() is next: an unauthorised sender's files are fetched and then
        # their message is refused. Cheap to reorder if that ever matters.
        here = Path(dest)
        if in_channel:
            # One folder per request: files from two people's threads never mix.
            here = here / _safe_name(f"{channel}_{reply_ts}")
        files, skipped = download_files(web, event, here)
    name = bridge.display_name(user) if in_channel and user else ""
    result = bridge.route(user, text, files, thread_ts=thread_ts, channel=channel,
                          channel_type=channel_type,
                          mentioned=bool(event.get("_mentioned")),
                          msg_ts=msg_ts, user_name=name)
    action = result.get("action")
    if acked:
        if action in ("turn", "answer", "interject") and result.get("request_id"):
            bridge.bind_ack(str(result["request_id"]), channel, msg_ts)
        else:
            bridge.drop_ack(channel, msg_ts)
    broadcast = bool(thread_ts) and not in_channel
    if action == "busy":
        if in_channel or str(result.get("why", "")).startswith("the queue"):
            bridge.post(channel, f"_{result.get('why', 'Busy.')}_", thread_ts=reply_ts)
        else:
            bridge.post(channel, "_Busy with another conversation right now — send "
                                 "this again when it finishes, or reply in that "
                                 "thread to steer the run that is going._",
                        thread_ts=thread_ts or "", broadcast=broadcast)
    elif action == "denied":
        if in_channel:
            if event.get("_mentioned") or result.get("thread_id"):
                why = (result.get("why") if result.get("thread_id") else
                       "Not authorised. Ask the owner of this agentY host to add your "
                       "Slack member id to `SLACK_ALLOWED_USERS`.")
                bridge.post(channel, f"_{why}_", thread_ts=reply_ts)
        else:
            bridge.post(channel, "_Not authorised. Ask the owner of this agentY host "
                                 "to add your Slack member id to `SLACK_ALLOWED_USERS`._")
    elif action == "interject":
        bridge.post(channel, "_Passed to the turn already running._",
                    thread_ts=reply_ts or "", broadcast=broadcast)
    elif action == "command":
        bridge.post(channel, result.get("reply", ""), thread_ts=reply_ts)
    elif action == "amended":
        bridge.post(channel, "_Added to your request - still number "
                    f"{result.get('position') or 1} in line._", thread_ts=reply_ts)
    elif action == "approval_unclear":
        bridge.post(channel, "_An approval is waiting here: reply *yes* or *no*._",
                    thread_ts=reply_ts)
    elif action == "approval":
        bridge.react(channel, msg_ts, "white_check_mark" if result.get("allowed") else "x")
    if skipped and action not in ("denied", "ignored"):
        # Said in Slack, not just logged: from a phone there is no canvas to look
        # at and no terminal to check, so an attachment that quietly did not
        # arrive looks exactly like an agent that ignored it.
        bridge.post(channel, "_Could not take: " + "; ".join(skipped) + "._",
                    thread_ts=reply_ts or "", broadcast=broadcast)
    logger.info("slack: message from %s → %s (%s) in %s, %d file(s), %d skipped",
                user, action, result.get("why", ""),
                result.get("thread_id") or "a new conversation",
                len(files), len(skipped))


def _fetch_files(web, event: dict) -> list:
    """The files on a mentioned message whose event did not carry them.

    An ``app_mention`` is a trimmed copy of the message and may leave ``files``
    out. One lookup of that message (top-level requests only) gets them back.
    """
    try:
        ts = str(event.get("ts") or "")
        if event.get("thread_ts") and event.get("thread_ts") != ts:
            return []
        resp = web.conversations_history(channel=event.get("channel"), latest=ts,
                                         inclusive=True, limit=1)
        msgs = (resp or {}).get("messages") or []
        if msgs and str(msgs[0].get("ts")) == ts:
            return list(msgs[0].get("files") or [])
    except Exception:  # noqa: BLE001 - a missing scope just means no attachment
        logger.debug("slack: could not look up the mention's files", exc_info=True)
    return []
