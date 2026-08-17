"""Turning one turn's event stream into Slack messages.

The sidebar and Slack show the same turn, and the split between them is the same
split the panel already makes on screen — it just uses a different device for it:

======================  ==========================  =========================
what                    panel                       Slack
======================  ==========================  =========================
the agent's answer      the message bubble          one message, edited as it
                                                    streams
generated media         dropped on the canvas       uploaded to the channel
a question to answer    a highlighted ask row       its own message (it pings)
tool calls, reasoning,  collapsible blocks          replies in that message's
plans, canvas edits     inline in the log           thread
progress / QA / exec    a transient status line     one thread reply, edited
                        under the composer          in place and cleared at the
                                                    end
======================  ==========================  =========================

A Slack thread is what a collapsible block is for: present, ignorable, and out of
the way of the answer. Without that split a single hook run posts dozens of
messages and the thing you actually wanted to read is somewhere in the middle.

Everything here is pure — events in, :class:`Post` list out, no network, no SDK.
The bridge owns the Slack ids and does the talking; this owns what gets said, so
what Slack shows can be tested without a workspace.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

# Slack renders a message body up to 40k characters, but anything past a few
# thousand is a scroll nobody reads and an edit that costs a full round trip.
_ANSWER_MAX = 3500
_DETAIL_MAX = 1200
_THINK_MAX = 1500
_TOOL_ARG_MAX = 300


@dataclass(frozen=True)
class Post:
    """One thing to say, and where.

    ``where``:
      * ``answer``  — the turn's own message, edited in place as text streams.
      * ``detail``  — a reply in that message's thread.
      * ``channel`` — a new top-level message: things that must be *seen*
        (a question, a finished file, an error) rather than found.

    ``key`` names a message that gets **rewritten** rather than repeated: the
    transient status line, and each tool call as its result comes back. Without
    it a turn is a wall of near-identical messages, which is the difference
    between a channel you keep open and one you mute.

    ``kind`` is ``text``, ``file`` (upload ``path``), or ``clear`` (delete the
    keyed message — how the transient status line goes away at the end).
    """
    where: str
    text: str = ""
    path: str = ""
    kind: str = "text"
    key: str = ""


# ── markdown → Slack mrkdwn ───────────────────────────────────────────────────
# Slack is not markdown: bold is *one* star, italics are underscores, and there
# are no headings. Left alone, every **emphasis** the agent writes arrives as
# literal asterisks and every heading as a stray hash.

_FENCE = re.compile(r"```.*?```", re.S)


def to_mrkdwn(text: str) -> str:
    """Rewrite the agent's markdown as Slack's. Code fences are left untouched."""
    parts = []
    last = 0
    for m in _FENCE.finditer(text or ""):
        parts.append(_convert(text[last:m.start()]))
        parts.append(m.group(0))
        last = m.end()
    parts.append(_convert((text or "")[last:]))
    return "".join(parts)


def _convert(chunk: str) -> str:
    # Headings first: "### Title" has no Slack equivalent, and bold is the
    # closest thing that still reads as a heading in a wall of text.
    chunk = re.sub(r"^\s{0,3}#{1,6}\s+(.+?)\s*$", r"*\1*", chunk, flags=re.M)
    # **bold** → *bold*  (before the single-star case can see it)
    chunk = re.sub(r"\*\*(?!\s)(.+?)(?<!\s)\*\*", r"*\1*", chunk, flags=re.S)
    chunk = re.sub(r"__(?!\s)(.+?)(?<!\s)__", r"_\1_", chunk, flags=re.S)
    # [text](url) → <url|text>
    chunk = re.sub(r"\[([^\]]+)\]\((https?://[^)\s]+)\)", r"<\2|\1>", chunk)
    # "- item" → "• item": Slack does not render markdown bullets.
    chunk = re.sub(r"^(\s*)[-*]\s+", r"\1• ", chunk, flags=re.M)
    return chunk


def clip(text: str, limit: int) -> str:
    """Cut to *limit*, saying how much was cut rather than trailing off."""
    text = str(text or "")
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + f"\n… (+{len(text) - limit:,} more characters)"


# ── the per-turn renderer ─────────────────────────────────────────────────────

# Nothing to show: panel bookkeeping, or a duplicate of something already said.
_IGNORED = {"thread", "request", "turn_start", "plan_step"}


class TurnRender:
    """One turn's events → what Slack should say, in order.

    Stateful because two of the streams are: the answer accumulates across many
    ``text`` deltas, and reasoning accumulates until the step that produced it
    ends (a `think` event is a handful of characters — posting each one would be
    thousands of Slack messages for one turn).
    """

    def __init__(self, *, origin: str = "panel", started_by: str = "",
                 show_thinking: bool = True, show_tools: bool = True):
        self.origin = origin
        self.started_by = started_by
        self.show_thinking = show_thinking
        self.show_tools = show_tools
        self.answer = ""
        self._think = ""
        self._files = 0
        self._done = False
        self._tools: set = set()   # tool calls still waiting on a result

    # The message that opens the turn, before there is any answer to show.
    def opening(self) -> str:
        asked = clip(" ".join((self.started_by or "").split()), 300)
        if self.origin == "slack":
            return f"_working on it…_\n\n>{asked}" if asked else "_working on it…_"
        head = self._header() + "_working on it…_"
        return f"{head}\n\n>{asked}" if asked else head

    def _header(self) -> str:
        """Whose turn this is, on the answer itself.

        A turn you did not start, arriving in your DM with no explanation, reads
        as the bot talking to itself. The opening message says so — but when text
        starts streaming immediately there IS no opening message, only an answer,
        so the attribution has to live on the answer or it is lost exactly when
        the turn is fast.
        """
        return "" if self.origin == "slack" else "_from the ComfyUI panel_\n\n"

    def feed(self, event: dict) -> list:
        """The Slack posts this event produces, in the order they should go out."""
        kind = str((event or {}).get("type") or "")
        if kind in _IGNORED or self._done:
            return []
        fn = getattr(self, "_on_" + kind, None)
        return fn(event) if fn else []

    # ── the answer ────────────────────────────────────────────────────────────
    def _on_text(self, ev) -> list:
        self.answer += str(ev.get("data") or "")
        return [Post("answer", self.body())]

    def body(self) -> str:
        """The answer message as it currently stands."""
        if not self.answer.strip():
            return self.opening()
        text = self._header() + to_mrkdwn(clip(self.answer, _ANSWER_MAX))
        return text if self._done else text + "\n\n_…_"

    def _on_done(self, ev) -> list:
        self._done = True
        out = self._flush_think()
        out.append(Post("detail", kind="clear", key="status"))
        if self.answer.strip():
            out.append(Post("answer", self.body()))
        elif not self._files:
            # Nothing said and nothing made. Silence would read as a crash.
            out.append(Post("answer", "_Finished with nothing to report._"))
        else:
            out.append(Post("answer", "_Finished._"))
        return out

    def _on_error(self, ev) -> list:
        # Top-level, not buried in the thread: an error is the one thing you must
        # not have to go looking for.
        return [Post("channel", "❌ " + to_mrkdwn(clip(ev.get("message"), _DETAIL_MAX)))]

    # ── what must be seen ─────────────────────────────────────────────────────
    def _on_output(self, ev) -> list:
        self._files += 1
        role = str(ev.get("role") or "").strip()
        name = str(ev.get("name") or "output")
        caption = f"{name} — _{role}_" if role else name
        return [Post("channel", caption, path=str(ev.get("path") or ""), kind="file")]

    def _on_ask(self, ev) -> list:
        return [Post("channel", "⏸️ *The agent is waiting on you*\n"
                     + to_mrkdwn(clip(ev.get("prompt"), _DETAIL_MAX))
                     + "\n\n_Reply here to answer._")]

    def _on_notify(self, ev) -> list:
        """A background completion — a Magnific render that finished minutes
        after the turn that queued it. The panel drops it on the canvas; here it
        is a message with the file attached, which is the whole point of having a
        second channel at all."""
        toast = ev.get("toast") or {}
        title = str(toast.get("title") or "").strip()
        body = str(toast.get("body") or "").strip()
        text = "🔔 " + " — ".join(p for p in (title, body) if p) if (title or body) else "🔔"
        out = [Post("channel", to_mrkdwn(clip(text, _DETAIL_MAX)))]
        path = str((ev.get("output") or {}).get("path") or "")
        if path:
            self._files += 1
            out.append(Post("channel", str((ev.get("output") or {}).get("name") or "output"),
                            path=path, kind="file"))
        return out

    # ── the transient status line ─────────────────────────────────────────────
    def _on_progress(self, ev) -> list:
        line = clip(str(ev.get("data") or "").strip(), 300)
        return [Post("detail", line, key="status")] if line else []

    _on_qa = _on_progress
    _on_console = _on_progress

    def _on_exec(self, ev) -> list:
        if str(ev.get("state") or "") == "start":
            return [Post("detail", "⚙️ ComfyUI running…", key="status")]
        return [Post("detail", kind="clear", key="status")]

    # ── the thread ────────────────────────────────────────────────────────────
    def _on_think(self, ev) -> list:
        if not self.show_thinking:
            return []
        self._think += str(ev.get("data") or "")
        return []

    def _flush_think(self) -> list:
        if not self._think.strip():
            self._think = ""
            return []
        text = clip(self._think.strip(), _THINK_MAX)
        self._think = ""
        return [Post("detail", "💭 _thinking_\n" + _quote(text))]

    def _on_step_start(self, ev) -> list:
        out = self._flush_think()
        out.append(Post("detail", "▶️ *" + clip(str(ev.get("name") or "step"), 120) + "*"))
        return out

    def _on_step_end(self, ev) -> list:
        return self._flush_think()

    def _on_tool(self, ev) -> list:
        """A call and its result are one event each, and one message between them.

        Keyed by the tool-use id so the result *rewrites* the call rather than
        following it — the same pairing the panel does with a collapsible block,
        and the difference between a readable thread and two lines per tool.
        """
        if not self.show_tools:
            return []
        name = str(ev.get("name") or "tool")
        key = "tool:" + str(ev.get("id") or name)
        if str(ev.get("phase") or "") == "result":
            result = str(ev.get("result") or "").strip()
            failed = result.lower().startswith("error")
            line = ("⚠️ `" if failed else "✅ `") + name + "`"
            if result:
                line += "\n" + _quote(clip(result, _TOOL_ARG_MAX))
            self._tools.discard(key)
            return [Post("detail", line, key=key)]
        self._tools.add(key)
        line = "🔧 `" + name + "`"
        detail = str(ev.get("input") or "").strip()
        if detail and detail not in ("{}", "None"):
            line += "\n" + _quote(clip(detail, _TOOL_ARG_MAX))
        return [Post("detail", line, key=key)]

    def _on_plan(self, ev) -> list:
        steps = [str(s) for s in (ev.get("steps") or [])]
        if not steps:
            return []
        body = "\n".join(f"{i + 1}. {s}" for i, s in enumerate(steps))
        return [Post("detail", "🗂️ *Plan*\n" + to_mrkdwn(clip(body, _DETAIL_MAX)))]

    def _on_system(self, ev) -> list:
        text = str(ev.get("data") or "").strip()
        return [Post("detail", to_mrkdwn(clip(text, _DETAIL_MAX)))] if text else []

    _on_status_line = _on_system

    def _on_canvas_patch(self, ev) -> list:
        # What landed on the canvas, said in words — the one part of a turn a
        # phone genuinely cannot show.
        op = str(ev.get("op") or "")
        said = {
            "place_text": "📝 Placed a text node on the canvas.",
            "review_collector": "🗳️ Collected the outputs into a review node on the canvas.",
            "review_released": "▶️ Review released — the chain continues.",
            "delete_nodes": "🗑️ Removed node(s) from the canvas.",
        }.get(op, "✏️ Updated a node on the canvas.")
        return [Post("detail", said)]

    def _on_interject_undelivered(self, ev) -> list:
        return [Post("detail", "_The agent had already finished its last step, so "
                               "that message goes out with the next turn._")]


def _quote(text: str) -> str:
    """Slack quotes per line; a bare `>` only quotes the first one."""
    return "\n".join("> " + ln for ln in str(text).splitlines())
