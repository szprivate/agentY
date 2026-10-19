"""One place for media a tool makes outside a ComfyUI save node.

A workflow's save node reaches the canvas (the executor's outputs) and the bEpic
viewer (``bEpicSendToViewer``). Everything else a turn produces - an MCP server's
render, a script's output, a sub-conversation's picture, an edit render, a
downloaded reference - used to reach neither, unless something happened to put the
path in the turn's output list, and even then a turn nobody streams (a shot) had no
page to drop it on.

Three small pieces, no per-tool patching:

* :class:`MediaSinkHookProvider` - the ONE seam where tool results are seen (an
  ``AfterToolCallEvent`` hook on every agent). For an MCP tool (``server__tool``) or
  ``run_script`` it reads the result with :mod:`agenty_core.utils.media_refs`:
  existing media files written during the call, plus downloadable media URLs from
  MCP tools, which are fetched into ``output_images/<server>/``. Each is handed to
  the turn's output sink (the same call a downloaded reference takes), so it is
  shown in the panel, staged into ComfyUI's input dir and dropped on the canvas.
* :func:`note` / :func:`take` - the group the file belongs under in the viewer (the
  MCP server, the shot, the hook title), remembered from the moment the file is
  registered until the host stages it and sends it on.
* :func:`push_to_viewer` / :func:`claim_drop` - the viewer side (``POST /bepic/push``
  on ComfyUI, from ComfyUI-ImageViewer) and the de-duplication for both.

The viewer only opens files inside ComfyUI's own folders, so what is pushed is the
staged copy in the input dir, not the original.
"""
from __future__ import annotations

import logging
import os
import threading
import time
from pathlib import Path

logger = logging.getLogger("agentY.media_sink")

_LOCK = threading.Lock()
_intent: dict = {}        # norm path -> {"group", "label"}: not yet sent on
_seen: set = set()        # (norm path, mtime): read from a tool result already
_viewer_sent: set = set() # (norm path, mtime): already pushed to the viewer
_dropped: set = set()     # (norm path, mtime): already announced for a drop
_viewer_off_until = 0.0   # a viewer without /bepic/push: not asked again for a while
_MAX_SET = 5000


def _norm(path) -> str:
    return os.path.normcase(os.path.abspath(str(path)))


def _key(path) -> tuple:
    try:
        return (_norm(path), int(os.path.getmtime(path)))
    except OSError:
        return (_norm(path), 0)


def _once(bucket: set, path) -> bool:
    """True the first time *path* (as it is now) is offered to *bucket*."""
    k = _key(path)
    with _LOCK:
        if k in bucket:
            return False
        if len(bucket) > _MAX_SET:
            bucket.clear()
        bucket.add(k)
    return True


# -- what a file is called in the viewer --------------------------------------

def default_group() -> str:
    """The tool the current turn is running - the MCP server for ``server__tool`` -
    for a file nobody named a group for."""
    try:
        from agenty_core.utils import turn_scope
        from src.utils import turn_bus
        step = turn_bus.current_tool(turn_scope.current().request_id)
        name = str((step or {}).get("name") or "")
    except Exception:  # noqa: BLE001
        return ""
    return name.split("__", 1)[0] if "__" in name else name


def note(path, group: str = "", label: str = "") -> None:
    """Remember where *path* belongs in the viewer. The first note wins, so the
    hook (which knows the MCP server) is not overruled by the generic sink."""
    if not path:
        return
    with _LOCK:
        if len(_intent) > _MAX_SET:
            _intent.clear()
        _intent.setdefault(_norm(path), {"group": " ".join(str(group or "").split())[:60],
                                         "label": " ".join(str(label or "").split())[:60]})


def take(path):
    """What :func:`note` recorded for *path* (once), or None for a file that came
    through a save node and is in the viewer already."""
    with _LOCK:
        return _intent.pop(_norm(path), None)


# -- the viewer ----------------------------------------------------------------

def _viewer_enabled() -> bool:
    env = os.environ.get("AGENTY_VIEWER_PUSH", "").strip().lower()
    if env in ("0", "false", "no", "off"):
        return False
    if env in ("1", "true", "yes", "on"):
        return True
    try:
        from src.agent import _load_settings
        return bool(((_load_settings() or {}).get("media_nodes") or {}).get("viewer_push", True))
    except Exception:  # noqa: BLE001
        return True


def push_to_viewer(path, group: str = "", label: str = "", *, wait: bool = False) -> bool:
    """Show *path* in the viewer's agentY tab, once. *path* must lie inside
    ComfyUI's folders (pass the staged copy). Never raises; True when sent."""
    global _viewer_off_until
    if not path or not _viewer_enabled() or time.time() < _viewer_off_until:
        return False
    if not _once(_viewer_sent, path):
        return False
    try:
        from src.utils.viewer_group import DEFAULT_TAB, tab_name
        tab = tab_name() or DEFAULT_TAB
    except Exception:  # noqa: BLE001
        tab = "agentY"
    body = {"paths": [str(path)], "tab_name": tab, "group": group, "label": label}

    def _send() -> bool:
        global _viewer_off_until
        try:
            from agenty_core.utils.comfyui_client import get_client
            res = get_client().post("/bepic/push", json_data=body)
            return bool(isinstance(res, dict) and res.get("shown"))
        except Exception as exc:  # noqa: BLE001
            if "404" in str(exc):          # a viewer from before /bepic/push
                _viewer_off_until = time.time() + 600
            logger.info("viewer push skipped for %s: %s", path, exc)
            return False

    if wait:
        return _send()
    threading.Thread(target=_send, name="agentY-viewer-push", daemon=True).start()
    return True


def claim_drop(path) -> bool:
    """True the first time a background turn announces *path* for the canvas."""
    return _once(_dropped, path)


# -- tool results ---------------------------------------------------------------

def watches(tool_name: str) -> str:
    """The group a tool's files belong under, or "" when this tool is not watched:
    MCP tools (``server__tool``, except Magnific, which has its own watcher) and
    ``run_script``. Other agentY tools publish their own files."""
    name = str(tool_name or "")
    if "__" in name and not name.startswith("magnific__"):
        return name.split("__", 1)[0]
    if name == "run_script":
        return "script"
    return ""


def _output_dir(group: str) -> Path:
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in group)[:40] or "tools"
    return Path(__file__).resolve().parents[2] / "output_images" / safe


def _publish(path: str) -> None:
    from agenty_core.tools import image_io
    publish = getattr(image_io, "publish_output", None) or getattr(image_io, "_publish")
    publish(path)


def observe(tool_name: str, result, started=None) -> list:
    """Publish the media files *result* names. Returns the paths handed on."""
    group = watches(tool_name)
    if not group:
        return []
    from agenty_core.utils import media_refs
    label = str(tool_name).split("__", 1)[-1]
    since = (started if started is not None else time.time()) - 2.0
    paths = media_refs.find_local(result, min_mtime=since)
    if "__" in str(tool_name):
        for url in media_refs.find_urls(result):
            got = media_refs.download(url, _output_dir(group), stem=f"{group}_{label}")
            if got:
                paths.append(got)
    published = []
    for p in paths:
        if not _once(_seen, p):
            continue
        note(p, group, label)
        _publish(p)
        published.append(p)
    return published


class MediaSinkHookProvider:
    """``AfterToolCallEvent`` on every agent: tool results are looked at here."""

    def __init__(self) -> None:
        self._started: dict = {}

    def register_hooks(self, registry, **kwargs) -> None:  # noqa: ARG002
        from strands.hooks.events import AfterToolCallEvent, BeforeToolCallEvent
        registry.add_callback(BeforeToolCallEvent, self._before)
        registry.add_callback(AfterToolCallEvent, self._after)

    def _before(self, event, **kwargs) -> None:  # noqa: ARG002
        try:
            tu = getattr(event, "tool_use", None) or {}
            if watches(tu.get("name", "")):
                self._started[str(tu.get("toolUseId", ""))] = time.time()
        except Exception:  # noqa: BLE001
            pass

    def _after(self, event, **kwargs) -> None:  # noqa: ARG002
        try:
            tu = getattr(event, "tool_use", None) or {}
            started = self._started.pop(str(tu.get("toolUseId", "")), None)
            if getattr(event, "exception", None) is not None:
                return
            got = observe(tu.get("name", ""), getattr(event, "result", None), started)
            if got:
                logger.info("%s: %d media file(s) published", tu.get("name"), len(got))
        except Exception as exc:  # noqa: BLE001 - reading a result must never fail the tool
            logger.warning("media sink: %s", exc, exc_info=True)
