"""Code the agent makes, sent to the bEpic viewer's "Agent code" panel.

Same idea as :mod:`src.utils.media_sink` for media, one seam each, no per-tool
patching:

* :class:`CodeSinkHookProvider` - a ``BeforeToolCallEvent`` hook on every agent. It
  reads the tool's *arguments* (so the code shows up even when the tool then
  fails): ``run_script`` commands, ``run_python_node`` snippets, files written with
  ``write_text_file`` that have a code extension (which is also how the coder
  agent's ``create_custom_node`` pack arrives), and any MCP tool argument that is
  clearly source code (a Remotion composition, ``remotion__update_composition``).
* :func:`observe_turn` - a :mod:`src.utils.turn_bus` observer that watches the
  reply's text and sends each fenced code block once it is closed.
* :func:`send` - ``POST /bepic/code`` on ComfyUI (ComfyUI-ImageViewer), in the
  background, once per identical snippet.

The snippet keeps ``source`` (the tool), ``conversation`` and ``turn`` so a later
"edit it and send it back to the agent" has everything it needs to find its way
home. Nothing here changes what a tool does; a failure to deliver is logged and
forgotten.
"""
from __future__ import annotations

import hashlib
import logging
import os
import re
import threading
import time

logger = logging.getLogger("agentY.code_sink")

_LOCK = threading.Lock()
_sent: set = set()               # hashes already sent (or in flight)
_off_until = 0.0                 # a viewer without /bepic/code: not asked again for a while
_MAX_SET = 2000
_MAX_CODE = 400_000
_texts: dict = {}                # request_id -> {"buf": str, "n": int}: reply text not yet fenced
_installed = {"on": False}

# What counts as code when written to a file, and what language it is.
CODE_EXT = {".py": "python", ".pyw": "python", ".js": "javascript", ".mjs": "javascript",
            ".cjs": "javascript", ".jsx": "jsx", ".ts": "typescript", ".tsx": "tsx",
            ".json": "json", ".sh": "shell", ".bash": "shell", ".ps1": "shell", ".bat": "shell",
            ".html": "html", ".css": "css", ".yaml": "yaml", ".yml": "yaml", ".toml": "toml"}

_FENCE_LANG = {"py": "python", "python3": "python", "js": "javascript", "node": "javascript",
               "ts": "typescript", "sh": "shell", "bash": "shell", "zsh": "shell",
               "powershell": "shell", "ps1": "shell", "console": "shell", "cmd": "shell",
               "yml": "yaml", "md": "markdown", "htm": "html", "jsonc": "json"}

# MCP argument names that usually carry source; the value must still look like code.
_CODE_KEYS = {"code", "source", "script", "content", "composition", "component", "tsx", "jsx",
              "src", "body", "text", "snippet", "program", "module"}
_CODE_HINT = re.compile(
    r"(^|\n)\s*(import\s|export\s|const\s|let\s|function\s|def\s|class\s|from\s+\S+\s+import|"
    r"return\s|if\s*\(|for\s*\(|#!/|@\w+\n)|=>|</?[A-Za-z][\w.]*[\s/>]|;\s*(\n|$)")
_PY_C = re.compile(r"""\bpython[\d.]*(?:\.exe)?["']?\s+-c\s+(?P<q>["'])(?P<code>.*)(?P=q)\s*$""", re.S)


# -- recognising code ----------------------------------------------------------------

def looks_like_code(text) -> bool:
    """A string with several lines (or a long line) and code-shaped content."""
    if not isinstance(text, str):
        return False
    t = text.strip()
    if len(t) < 40 or len(t) > _MAX_CODE:
        return False
    if "\n" not in t and len(t) < 120:
        return False
    return bool(_CODE_HINT.search(t))


def guess_language(code: str, hint: str = "") -> str:
    """Language of untagged code: the hint (an extension or tag) first."""
    h = str(hint or "").strip().lower()
    if h.startswith("."):
        h = CODE_EXT.get(h, "")
    h = _FENCE_LANG.get(h, h)
    if h and h not in ("text", "txt", "plain", "plaintext"):
        return h
    t = code.strip()
    if t[:1] in "{[" and re.search(r'"\s*:', t):
        return "json"
    if re.search(r"from\s+['\"](react|remotion)|</?[A-Z]\w*[\s/>]|React\.", t):
        return "tsx"
    if re.search(r"^\s*(def |class |import \w|from \w+ import|async def |@\w+\n)", t, re.M):
        return "python"
    if re.search(r"^\s*(#!/bin/(ba)?sh|\$ |npm |pip |git |cd |echo )", t, re.M):
        return "shell"
    if re.search(r"\b(const|let|function|export|=>)\b", t):
        return "javascript"
    return "text"


def extract_fences(text: str) -> list:
    """``[(language, code), ...]`` for each closed fenced block in *text*."""
    out = []
    for m in re.finditer(r"(?m)^[ \t]*(```|~~~)[ \t]*([\w+.-]*)[^\n]*\n(.*?)^[ \t]*\1[ \t]*$",
                         text or "", re.S):
        code = m.group(3).rstrip("\n")
        if code.strip():
            out.append((guess_language(code, m.group(2)), code))
    return out


def snippets_for_tool(name: str, args) -> list:
    """The code in one tool call's arguments: ``[{title, language, code, source}]``."""
    name = str(name or "")
    if not isinstance(args, dict):
        return []
    out = []

    def add(title, code, language, source=None):
        if isinstance(code, str) and code.strip() and len(code) <= _MAX_CODE * 2:
            out.append({"title": title, "language": language, "code": code,
                        "source": source or name})

    if name == "run_python_node":
        add(str(args.get("title") or "python node"), args.get("code"), "python")
    elif name == "run_script":
        if str(args.get("code") or "").strip():
            add("run_script (code)", args.get("code"), "python")
        cmd = str(args.get("command") or "")
        m = _PY_C.search(cmd.strip())
        if m:
            add("run_script (python -c)", m.group("code").replace('\\"', '"'), "python")
        elif "\n" in cmd.strip() or len(cmd) > 160:
            add("run_script", cmd, "shell")
    elif name == "write_text_file":
        path = str(args.get("path") or "")
        ext = os.path.splitext(path)[1].lower()
        if ext in CODE_EXT:
            add(os.path.basename(path) or "file", args.get("content"), CODE_EXT[ext])
    elif "__" in name:
        for key, val in args.items():
            if str(key).lower() in _CODE_KEYS or re.search(r"composition|code|script", name, re.I):
                if looks_like_code(val):
                    title = name.split("__", 1)[1].replace("_", " ")
                    path = str(args.get("path") or args.get("file") or args.get("name") or "")
                    ext = os.path.splitext(path)[1].lower()
                    add(f"{title} ({os.path.basename(path)})" if ext in CODE_EXT else title,
                        val, guess_language(val, ext if ext in CODE_EXT else ""))
    return out


# -- delivery ------------------------------------------------------------------------

def _enabled() -> bool:
    env = os.environ.get("AGENTY_VIEWER_CODE", "").strip().lower()
    if env in ("0", "false", "no", "off"):
        return False
    try:
        from src.utils import media_sink
        return media_sink._viewer_enabled()
    except Exception:  # noqa: BLE001
        return True


def _context() -> dict:
    """Which conversation and turn the code belongs to (best effort)."""
    try:
        from agenty_core.utils import turn_scope
        scope = turn_scope.current()
        return {"conversation": str(scope.thread_id or ""), "turn": str(scope.request_id or "")[:8]}
    except Exception:  # noqa: BLE001
        return {"conversation": "", "turn": ""}


def send(snippet: dict, *, wait: bool = False, **context) -> bool:
    """Send one snippet to the viewer, once. Never raises; True when handed on."""
    global _off_until
    code = snippet.get("code") if isinstance(snippet, dict) else None
    if not isinstance(code, str) or not code.strip() or not _enabled() or time.time() < _off_until:
        return False
    key = hashlib.sha1((str(snippet.get("language", "")) + "\0" + code).encode("utf-8", "replace")).hexdigest()
    with _LOCK:
        if key in _sent:
            return False
        if len(_sent) > _MAX_SET:
            _sent.clear()
        _sent.add(key)
    body = dict(snippet)
    body["code"] = code[:_MAX_CODE]
    body.setdefault("ts", time.time())
    for k, v in {**(context or _context())}.items():
        body.setdefault(k, v)

    def _post() -> bool:
        global _off_until
        try:
            from agenty_core.utils.comfyui_client import get_client
            res = get_client().post("/bepic/code", json_data=body)
            return bool(isinstance(res, dict) and res.get("ok"))
        except Exception as exc:  # noqa: BLE001
            if "404" in str(exc):          # a viewer from before /bepic/code
                _off_until = time.time() + 600
            with _LOCK:
                _sent.discard(key)         # not delivered: may be tried again
            logger.info("code not sent to the viewer (%s): %s", body.get("title"), exc)
            return False

    if wait:
        return _post()
    threading.Thread(target=_post, name="agentY-viewer-code", daemon=True).start()
    return True


def observe_tool(name: str, args) -> int:
    """Send the code in a tool call's arguments. Returns how many snippets."""
    n = 0
    ctx = _context()
    for snip in snippets_for_tool(name, args):
        n += 1 if send(snip, **ctx) else 0
    return n


class CodeSinkHookProvider:
    """``BeforeToolCallEvent`` on every agent: a tool's code arguments are read here."""

    def register_hooks(self, registry, **kwargs) -> None:  # noqa: ARG002
        from strands.hooks.events import BeforeToolCallEvent
        registry.add_callback(BeforeToolCallEvent, self._before)

    def _before(self, event, **kwargs) -> None:  # noqa: ARG002
        try:
            tu = getattr(event, "tool_use", None) or {}
            observe_tool(tu.get("name", ""), tu.get("input"))
        except Exception as exc:  # noqa: BLE001 - reading arguments must never fail the tool
            logger.warning("code sink: %s", exc, exc_info=True)


# -- the reply's fenced blocks -------------------------------------------------------

def observe_turn(event: dict, turn) -> None:
    """:mod:`src.utils.turn_bus` observer: fenced blocks of the reply, as they close."""
    try:
        kind = event.get("type")
        rid = str(getattr(turn, "request_id", ""))
        if kind == "text":
            chunk = event.get("data")
            if not isinstance(chunk, str):
                return
            with _LOCK:
                st = _texts.setdefault(rid, {"buf": "", "n": 0})
                st["buf"] += chunk
                buf = st["buf"]
            if "```" not in buf and "~~~" not in buf:
                return
            blocks = extract_fences(buf)
            with _LOCK:
                st = _texts.get(rid)
                if st is None:
                    return
                fresh, st["n"] = blocks[st["n"]:], max(st["n"], len(blocks))
            _send_fences(fresh, turn)
        elif kind == "done":
            with _LOCK:
                st = _texts.pop(rid, None)
            if st and (st["buf"].count("```") % 2 == 1):      # a block the reply never closed
                tail = st["buf"] + chr(10) + "```" + chr(10)
                _send_fences(extract_fences(tail)[st["n"]:], turn)
    except Exception as exc:  # noqa: BLE001 - a watcher must never break the turn
        logger.warning("code sink (reply): %s", exc, exc_info=True)


def _send_fences(blocks, turn) -> None:
    ctx = {"conversation": str(getattr(turn, "thread_id", "") or ""),
           "turn": str(getattr(turn, "request_id", "") or "")[:8]}
    for lang, code in blocks:
        first = next((ln.strip() for ln in code.splitlines() if ln.strip()), "")
        send({"title": ("Reply: " + first[:50]) if first else "Reply code",
              "language": lang, "code": code, "source": "reply"}, **ctx)


def install() -> None:
    """Start listening to replies (once). The tool hook is added by the agent factory."""
    if _installed["on"]:
        return
    from src.utils import turn_bus
    turn_bus.observe(observe_turn)
    _installed["on"] = True
