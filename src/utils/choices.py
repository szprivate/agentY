"""Choices the agent puts to the user, as cards in the panel and numbers in Slack.

The agent ends a turn that needs the user to pick with ONE fenced block::

    ```choices
    Which workflow should I use?
    1. Local - LTX 2.5 | runs on this GPU, needs 25 GB of models
    2. API - Kling 3 | partner nodes, costs credits
    ```

The first lines that are not options are the question; every ``<n>. <label>``
line is an option, and the text after ``|`` its detail. The panel draws it as
cards (a click sends the option as the user's answer), Slack and anything else
that only shows text get it as a numbered list (:func:`to_plain`), and a reply
that is only the number (``2``, ``2.``, ``option 2``, ``b``) is spelled out to
the option it names before the next turn starts (:func:`resolve_reply`), so the
agent never has to guess what "2" meant.

The same arithmetic is in agentY-comfyuiConnect's ``web/agent_choices.js``.
"""
from __future__ import annotations

import re

MAX_OPTIONS = 6

BLOCK_RE = re.compile(r"```[ \t]*choices[ \t]*\r?\n(?P<body>.*?)```", re.S | re.I)
_OPTION_RE = re.compile(r"^\s*(?P<n>\d{1,2})[.)]\s+(?P<text>\S.*?)\s*$")
_REPLY_RE = re.compile(r"^\s*(?:option|choice|nr\.?|number|no\.?|#)?\s*(?P<n>\d{1,2}|[a-f])\s*[.)]?\s*$",
                       re.I)


def parse_block(body: str) -> dict | None:
    """``{"question", "options": [{"n", "label", "detail"}]}`` of one block's body,
    or None when it holds fewer than two options."""
    question, options = [], []
    for line in str(body or "").splitlines():
        m = _OPTION_RE.match(line)
        if m:
            text = m.group("text")
            label, _, detail = text.partition("|")
            options.append({"n": len(options) + 1, "label": label.strip(), "detail": detail.strip()})
            continue
        if line.strip() and not options:
            question.append(line.strip())
    options = [o for o in options if o["label"]][:MAX_OPTIONS]
    if len(options) < 2:
        return None
    return {"question": " ".join(question), "options": options}


def find(text: str) -> dict | None:
    """The LAST choices block of an answer (the one the user is asked), or None."""
    found = None
    for m in BLOCK_RE.finditer(str(text or "")):
        found = parse_block(m.group("body")) or found
    return found


def option_text(opt: dict) -> str:
    return opt["label"] + (f" ({opt['detail']})" if opt.get("detail") else "")


def answer_for(choice: dict, opt: dict) -> str:
    """What a picked option says as the user's message - the panel's card sends
    the same words."""
    return f"{opt['n']}. {option_text(opt)}"


def resolve_reply(previous_answer: str, reply: str) -> str:
    """*reply* with a bare option number spelled out, else *reply* unchanged."""
    choice = find(previous_answer)
    if not choice or not reply:
        return reply
    m = _REPLY_RE.match(str(reply))
    if not m:
        return reply
    key = m.group("n").lower()
    n = int(key) if key.isdigit() else "abcdef".index(key) + 1
    opts = choice["options"]
    if not 1 <= n <= len(opts):
        return reply
    return f"{answer_for(choice, opts[n - 1])} - I pick option {n} of: {choice['question'] or 'your question'}"


def to_plain(text: str, hint: str = "Reply with the number, or in your own words.") -> str:
    """*text* with every choices block written as a numbered list, for a reader
    that shows text only (Slack, a log)."""
    def _one(m):
        c = parse_block(m.group("body"))
        if not c:
            return m.group("body").strip()
        lines = [c["question"]] if c["question"] else []
        lines += [f"{o['n']}. {o['label']}" + (f" - {o['detail']}" if o["detail"] else "")
                  for o in c["options"]]
        if hint:
            lines.append(f"_{hint}_")
        return "\n".join(lines)
    return BLOCK_RE.sub(_one, str(text or ""))
