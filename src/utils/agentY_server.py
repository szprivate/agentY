"""
agentY bridge + chat host — runs on localhost:5000.

This is the backend for the **ComfyUI-native agentY chat UI** (the sidebar
custom-node panel in the separate ``agentY-comfyuiConnect`` repo). It replaces the
Chainlit GUI: the pipeline runs here, conversations persist to a local SQLite
store (:mod:`src.utils.conversation_store`), and — crucially — generated media is
**not** streamed back as inline images. Instead the executor's output files are
staged into ComfyUI's input directory and announced to the frontend, which drops
an image / video loader node onto the open ComfyUI graph — a VHS ``(Path)``
loader pointed at the original file where that node pack is installed, otherwise
a ``LoadImage`` naming the staged copy. Only the agent's *text* flows into the
chat.

Endpoints
---------
Chat UI
    GET  /agentY/health
    GET  /agentY/commands                     slash-command list (for the popup)
    GET  /agentY/threads                       list saved conversations
    POST /agentY/threads                       create a thread            -> {id}
    GET  /agentY/threads/<id>                   thread messages + gallery
    DELETE /agentY/threads/<id>                 delete a thread
    POST /agentY/threads/clear                  delete all (keep ?current)
    POST /agentY/upload                         multipart image attachment -> {path}
    POST /agentY/chat        (SSE)              stream a turn; body {thread_id,message,image_paths}
    POST /agentY/reply                          answer an interactive ask  {request_id,text}
    POST /agentY/interject                      speak into a RUNNING turn  {request_id,text,urgent}

Viewers (self-contained HTML pages served here so they fetch same-origin)
    GET  /agentY/log_viewer                     message-history log viewer page
    GET  /agentY/message_history                raw message-history log feed
    POST /agentY/message_history/clear          purge the entire history log
    GET  /agentY/project_memory_viewer          project-memory editor page
    GET  /agentY/project_memory                 list project facts -> {entries}
    POST /agentY/project_memory/delete          forget selected      {names}
    GET  /agentY/memory_viewer                  long-term-memory viewer page
    GET  /agentY/memory                          list stored long-term memories -> {memories}
    POST /agentY/memory/update                   edit one memory      {id,text}
    POST /agentY/memory/delete                   delete selected      {ids}
    POST /agentY/memory/clear                    purge all memory

Legacy ComfyUI -> agent image-review bridge (kept)
    GET  /agentY/pending_previews
    POST /agentY/review
    GET  /agentY/node_responses
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import queue
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path

from src.utils import conversation_store as cs
from src.utils import status_bus
from src.utils import notify_bus
from src.utils import interject_bus
from src.utils import turn_bus
from src.utils import turn_watchdog as _wd
from src.utils.media_loaders import CANDIDATES as _LOADER_CANDIDATES
from src.utils.models import AgentSession

logger = logging.getLogger("agentY.server")

# ── In-memory state ───────────────────────────────────────────────────────────

_lock = threading.Lock()
_pending_previews: dict[str, dict] = {}
_node_responses: dict[str, str] = {}       # node_id (str) -> accumulated agent text
_agent_ref = None                          # the pipeline singleton

# Identifies THIS host process, handed out by /agentY/health. The sidebar remembers
# the last one it saw, so a host that was restarted is recognised as a *different*
# process even when the panel never observed the gap — a backgrounded ComfyUI tab
# throttles its heartbeat to about one tick a minute, which is easily long enough
# to miss a whole restart.
_BOOT_ID = uuid.uuid4().hex[:12]
_BOOT_TIME = time.time()

# Live brain-history cache keyed by thread, so switching threads inside a running
# app restores the exact Brain messages (which may contain image bytes that don't
# JSON-serialise). Durable state (agent_session, summaries, gallery) lives in the
# SQLite store for cross-restart resume.
_thread_brain_cache: dict[str, list] = {}

# Pending interactive asks: request_id -> (event_loop, asyncio.Queue). The SSE
# generator's async task awaits the queue; POST /agentY/reply feeds it thread-safely.
_reply_lock = threading.Lock()
_reply_registry: dict[str, tuple] = {}
# Active pipeline runs: request_id -> {"loop", "task"}. POST /agentY/stop cancels
# the task (halting the agent loop) and interrupts any running ComfyUI job.
_run_registry: dict[str, dict] = {}


# ── The canvas, for a turn that has no browser behind it ──────────────────────
#
# The graph reaches the agent because the PANEL captures it and posts it with the
# message. A turn asked for from Slack has no browser round-trip, so it arrived
# with no graph, no hooks and no selection — and every canvas tool answered "no
# on-canvas graph is loaded this turn", which reads as the agent refusing to look
# at a workflow that is plainly open.
#
# So the host asks for one. A flag rides out on /agentY/health (polled every 5s
# by the panel's heartbeat), the panel posts the same payload it would send with
# a message, and the snapshot is cached here. Kept with the time it was taken:
# a graph presented as current when it is four minutes old is the kind of wrong
# that costs a run rather than a sentence.
_canvas_lock = threading.Lock()
_canvas_cache: dict = {}          # {prompt, hooks, selection, ts}
_canvas_wanted_until = 0.0        # a request is outstanding until this time
_CANVAS_REQUEST_TTL = 30.0        # stop asking if nobody answers
_CANVAS_WAIT = 8.0                # how long a Slack turn waits for a fresh one
_CANVAS_FRESH = 20.0              # newer than this and it is worth using as-is
_CANVAS_STALE = 180.0             # older than this and it is not worth trusting


def canvas_wanted() -> bool:
    """Whether the panel should post its graph on the next heartbeat."""
    with _canvas_lock:
        return time.time() < _canvas_wanted_until


def remember_canvas(prompt, hooks, selection) -> None:
    """Cache what the panel says is on screen right now."""
    global _canvas_wanted_until
    with _canvas_lock:
        _canvas_cache.update({
            "prompt": prompt if isinstance(prompt, dict) else None,
            "hooks": [h for h in (hooks or []) if isinstance(h, dict)],
            "selection": [n for n in (selection or []) if isinstance(n, dict)],
            "ts": time.time(),
        })
        _canvas_wanted_until = 0.0


def request_canvas(wait: float = _CANVAS_WAIT) -> dict:
    """Ask the panel for the live graph and wait briefly for it.

    Returns the snapshot (possibly a stale one, possibly empty). Never raises and
    never blocks for long: a turn that cannot see the canvas is worth running
    anyway — it just has to be told what it is looking at.
    """
    global _canvas_wanted_until
    with _canvas_lock:
        before = _canvas_cache.get("ts", 0.0)
        if before and time.time() - before < _CANVAS_FRESH:
            return dict(_canvas_cache)     # someone just sent one; do not wait
        _canvas_wanted_until = time.time() + _CANVAS_REQUEST_TTL
    deadline = time.time() + max(0.0, wait)
    while time.time() < deadline:
        time.sleep(0.25)
        with _canvas_lock:
            if _canvas_cache.get("ts", 0.0) > before:
                return dict(_canvas_cache)
    with _canvas_lock:
        stale = dict(_canvas_cache)
    age = time.time() - stale.get("ts", 0.0) if stale.get("ts") else None
    if age is None or age > _CANVAS_STALE:
        # Nobody answered and what we have is old. A graph handed over as
        # current when it is minutes out of date is worse than none: the agent
        # edits nodes that have moved, or reports on a workflow you closed. With
        # nothing, the canvas tools say so and the agent can ask.
        logger.info("slack: no usable canvas snapshot (age %s)",
                    "none" if age is None else f"{age:.0f}s")
        return {}
    return stale


# ── Slash commands (mirrors the frontend popup list) ──────────────────────────

# The /help command points here (GitHub renders the guide with its images inline).
DOCS_URL = "https://github.com/szprivate/agentY/blob/main/docs/using-agentY.md"

SLASH_COMMANDS = [
    {"name": "/help",            "description": "Open the agentY usage guide in a new browser tab"},
    {"name": "/restart",         "description": "Restart the agent pipeline"},
    {"name": "/stop",            "description": "Stop and shut down the agent"},
    {"name": "/unload",          "description": "Unload Ollama models from VRAM"},
    {"name": "/clear_vram",      "description": "Clear ComfyUI GPU VRAM"},
    {"name": "/images",          "description": "List images generated in this thread (reference them by number)"},
    {"name": "/project_memory", "description": "Inspect and forget what is remembered for THIS project"},
    {"name": "/clearhistory",    "description": "Delete all conversation history (keeps the current thread)"},
    {"name": "/switch_model",    "description": "Switch an agent's LLM — /switch_model <agent|all> <provider,model> (use 'all' for every agent)"},
    {"name": "/add_workflow",    "description": "Add a ComfyUI workflow — /add_workflow <path/to/workflow.json> OR /add_workflow canvas <name> for the graph open in the canvas"},
    {"name": "/resend",          "description": "Resend the first user message of the current thread"},
    {"name": "/remove_workflow", "description": "Remove a workflow by name — /remove_workflow <template_name>"},
]


# ── Media helpers ─────────────────────────────────────────────────────────────

_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp"}
# Same list as canvas_hooks.VID_EXTS and the collector node's own, so a file
# is the same kind of thing everywhere it is asked about.
_VIDEO_SUFFIXES = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v", ".mpg", ".mpeg"}

# Node classes the frontend will try (first one registered in ComfyUI wins) when
# it drops a loader onto the graph for a generated output. Which node, and which
# of `path`/`filename` belongs in it, are one decision — hence media_loaders. The
# staging copy still happens either way: it is the fallback when VHS is not
# installed, and other lookups lean on it.
_NODE_CANDIDATES = _LOADER_CANDIDATES


def _is_image_path(path: str) -> bool:
    return Path(path).suffix.lower() in _IMAGE_SUFFIXES


def _is_video_path(path: str) -> bool:
    return Path(path).suffix.lower() in _VIDEO_SUFFIXES


# The canvas selection feeds the agent's inputs ONLY when the user asks for it
# (or on the very first turn). This keeps a selection that drifts over the course
# of a conversation from silently rebinding — or losing — the image input(s).
_CANVAS_INPUT_INTENT = re.compile(
    r"\b("
    r"select(ed|ion)?|highlight(ed)?|marked|"
    r"these\s+(nodes?|images?|photos?|pictures?|frames?|clips?|videos?|inputs?)|"
    r"this\s+(image|photo|picture|node|selection|frame|clip|video|input)|"
    r"the\s+selected|"
    r"(on|from|in)\s+(the\s+)?(canvas|graph)|"
    r"canvas\s+selection|"
    r"the\s+nodes?\s+i\b"
    r")\b",
    re.IGNORECASE,
)


def _message_wants_canvas_inputs(message: str) -> bool:
    """True when the user's message references the canvas selection as input —
    e.g. "the selected images", "these nodes", "from the canvas". Gates whether
    the CURRENT selection feeds the agent's inputs on this turn."""
    return bool(_CANVAS_INPUT_INTENT.search(message or ""))


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent.parent


_COMFY_INPUT_DIR: Path | None = None
_COMFY_INPUT_RESOLVED = False


def _comfy_input_dir() -> Path | None:
    """Resolve ComfyUI's --input-directory (cached), for staging outputs as node inputs.

    Order: parse ``--input-directory`` / ``--user-directory`` from ComfyUI's argv
    (``/system_stats``), else derive ``<base>/input`` from ``comfyui_user_dir`` in
    settings.json (``<base>/user/default`` → ``<base>/input``).
    """
    global _COMFY_INPUT_DIR, _COMFY_INPUT_RESOLVED
    if _COMFY_INPUT_RESOLVED:
        return _COMFY_INPUT_DIR
    _COMFY_INPUT_RESOLVED = True
    # 1. From ComfyUI argv.
    try:
        from agenty_core.utils.comfyui_client import get_client, parse_argv_dir_flag
        stats = get_client().get("/system_stats")
        argv = stats.get("system", {}).get("argv", []) if isinstance(stats, dict) else []
        d = parse_argv_dir_flag(argv, "--input-directory")
        if d:
            _COMFY_INPUT_DIR = Path(d).resolve()
            return _COMFY_INPUT_DIR
        u = parse_argv_dir_flag(argv, "--user-directory")
        if u:
            base = Path(u).resolve().parent.parent  # <base>/user/default -> <base>
            cand = base / "input"
            if cand.exists():
                _COMFY_INPUT_DIR = cand
                return _COMFY_INPUT_DIR
    except Exception as exc:
        logger.debug("input-dir resolve via argv failed: %s", exc)
    # 2. Derive from settings comfyui_user_dir.
    try:
        from src.utils.settings import load_settings
        ud = load_settings().get("comfyui_user_dir")
        if ud:
            base = Path(ud).parent.parent
            cand = base / "input"
            if cand.exists():
                _COMFY_INPUT_DIR = cand
    except Exception as exc:
        logger.debug("input-dir resolve via settings failed: %s", exc)
    return _COMFY_INPUT_DIR


def _effective_comfyui_user_dir() -> str | None:
    """The live ComfyUI --user-directory (from /system_stats argv), or None.

    Used so the settings UI shows the directory ComfyUI was actually launched
    with, rather than the static fallback in settings.json.
    """
    try:
        from agenty_core.utils.comfyui_client import get_client, parse_argv_dir_flag
        stats = get_client().get("/system_stats")
        argv = stats.get("system", {}).get("argv", []) if isinstance(stats, dict) else []
        d = parse_argv_dir_flag(argv, "--user-directory")
        if d:
            return str(Path(d).resolve())
    except Exception as exc:
        logger.debug("effective user-dir resolve failed: %s", exc)
    return None


def _drop_outputs_into_canvas() -> bool:
    """Should a finished image/video be dropped onto the canvas as a loader node?

    Decided here rather than in the panel so one answer covers every route a
    result can arrive by — a turn's own stream, a background Magnific completion,
    a Slack-driven run watched from the sidebar. A browser-side toggle would have
    to be repeated in each of them, and would disagree with itself the moment one
    was missed.
    """
    env = os.environ.get("AGENTY_CANVAS_DROP", "").strip().lower()
    if env in ("1", "true", "yes", "on"):
        return True
    if env in ("0", "false", "no", "off"):
        return False
    try:
        from src.agent import _load_settings
        return bool((_load_settings() or {}).get("drop_outputs_into_canvas", True))
    except Exception:  # noqa: BLE001
        return True


def _configure_shell_sandbox() -> None:
    """Tell agenty_core which folders and programs run_script may use here.

    agenty_core defaults to the checkout and a temp directory, because it is
    framework-agnostic and resolving ComfyUI's directories means an HTTP call it
    has no business making on every command. agentY knows them, so it says so
    once, at startup.

    Best-effort throughout: a ComfyUI that is not up yet costs the media folders,
    not the ability to start.
    """
    roots: list[str] = []
    try:
        from src.agent import _load_settings
        settings = _load_settings() or {}
        sec = settings.get("security") or {}
        extra_cmds = list(sec.get("shell_allowed_commands") or [])
        roots.extend(str(r) for r in (sec.get("shell_extra_roots") or []))
        for key in ("comfyui_models_dir", "output_dir", "output_workflows_dir"):
            value = str(settings.get(key) or "").strip()
            if value:
                roots.append(value)
    except Exception:  # noqa: BLE001
        extra_cmds = []

    try:
        from agenty_core.tools.comfyui import get_comfyui_dirs
        dirs = json.loads(get_comfyui_dirs())
        for key in ("input_dir", "output_dir", "user_dir", "base_dir", "comfyui_dir"):
            value = str(dirs.get(key) or "").strip()
            if value and value.lower() != "unknown":
                roots.append(value)
    except Exception as exc:  # noqa: BLE001
        logger.debug("ComfyUI directories not added to the shell sandbox: %s", exc)

    try:
        from agenty_core import sandbox
        sandbox.configure(executables=extra_cmds, roots=roots)
        logger.info("run_script sandbox: %d root(s), %d extra program(s)",
                    len(sandbox.allowed_roots()), len(extra_cmds))
    except Exception as exc:  # noqa: BLE001
        logger.warning("could not configure the run_script sandbox: %s", exc)


def _output_role(path: str) -> tuple[str, bool]:
    """(what this output is FOR, whether the user named it themselves).

    Resolving it also freezes the role for this path and drops the sidecar beside
    it — see :mod:`src.utils.output_tags`. The second value gates the tag node the
    panel attaches: a role the *user* wrote in the hook's prompt is worth putting
    on their canvas; one derived from a directive is worth putting in a title.
    """
    try:
        from src.utils import output_tags
        return output_tags.role_for(path), bool(output_tags.meta_for(path).get("declared"))
    except Exception:  # noqa: BLE001
        return "", False


def _copy_sidecar(src: str, staged_name: str) -> None:
    """Give the staged copy its own record, so a canvas node can be traced back.

    A loader holds an input-relative filename and nothing else. Without a sidecar
    next to *that* file, the next turn looking at the node has no way from the
    name to what the thing is.
    """
    try:
        from src.utils.output_tags import read_sidecar, write_sidecar
        in_dir = _comfy_input_dir()
        rec = read_sidecar(src)
        if in_dir is None or not rec.get("role"):
            return
        write_sidecar(in_dir / staged_name, rec.pop("role"), **rec)
    except Exception as exc:  # noqa: BLE001
        logger.debug("sidecar copy failed for %s: %s", staged_name, exc)


def _stage_into_comfy_input(path: str) -> str | None:
    """Copy *path* into ComfyUI's input dir so a loader node can reference it.

    Returns the input-relative filename (basename) the node's widget should use,
    or None when the input dir can't be resolved (frontend then falls back to the
    absolute path for path-based loaders).
    """
    in_dir = _comfy_input_dir()
    if in_dir is None:
        return None
    try:
        in_dir.mkdir(parents=True, exist_ok=True)
        src = Path(path)
        dest = in_dir / src.name
        if dest.resolve() == src.resolve():
            return src.name
        if dest.exists():
            dest = in_dir / f"{src.stem}_{uuid.uuid4().hex[:6]}{src.suffix}"
        shutil.copy2(src, dest)
        return dest.name
    except Exception as exc:
        logger.warning("could not stage %s into ComfyUI input: %s", path, exc)
        return None


# ── Magnific background auto-drop (async creation → canvas) ────────────────────

_CT_EXT = {
    "image/png": ".png", "image/jpeg": ".jpg", "image/webp": ".webp",
    "image/gif": ".gif", "image/avif": ".avif", "image/bmp": ".bmp",
    "image/tiff": ".tiff", "video/mp4": ".mp4", "video/webm": ".webm",
    "video/quicktime": ".mov", "video/x-matroska": ".mkv",
}


def _asset_ext(url: str, content_type: str, kind_hint: str = "") -> str:
    """Pick a file extension from the URL path, then Content-Type, then kind."""
    from urllib.parse import urlparse
    suf = Path(urlparse(url).path).suffix.lower()
    if suf in _IMAGE_SUFFIXES or suf in _VIDEO_SUFFIXES:
        return suf
    ct = (content_type or "").split(";", 1)[0].strip().lower()
    if ct in _CT_EXT:
        return _CT_EXT[ct]
    return ".mp4" if kind_hint == "video" else ".png"


def _download_url_to(url: str, dest_dir: Path, stem: str, kind_hint: str = "") -> Path | None:
    """Stream a remote asset to *dest_dir* (extension inferred). Returns the saved
    path, or None on failure."""
    import requests  # noqa: PLC0415
    try:
        dest_dir.mkdir(parents=True, exist_ok=True)
        with requests.get(url, stream=True, timeout=180) as r:
            r.raise_for_status()
            ext = _asset_ext(url, r.headers.get("content-type", ""), kind_hint)
            safe = re.sub(r"[^A-Za-z0-9._-]", "_", stem)[:40] or "magnific"
            dest = dest_dir / f"{safe}_{uuid.uuid4().hex[:8]}{ext}"
            with open(dest, "wb") as f:
                for chunk in r.iter_content(chunk_size=1 << 16):
                    if chunk:
                        f.write(chunk)
        return dest
    except Exception as exc:  # noqa: BLE001
        logger.warning("magnific: download failed (%s): %s", url, exc)
        return None


def _handle_magnific_complete(identifier: str, asset_url: str, kind: str,
                              meta: dict) -> dict | None:
    """Download a finished Magnific asset, stage it into ComfyUI's input dir, and
    return the drop payload the sidebar's ``injectNode`` consumes. Runs in the
    watcher's background thread (see :mod:`src.utils.magnific_watch`)."""
    dest_dir = _project_root() / "output_images" / "magnific"
    path = _download_url_to(asset_url, dest_dir, f"magnific_{identifier}", kind)
    if path is None:
        return None
    p = str(path)
    real_kind = "video" if _is_video_path(p) else "image" if _is_image_path(p) else (kind or "image")
    staged = _stage_into_comfy_input(p) if real_kind in ("image", "video") else None
    tid = (meta or {}).get("thread_id") or ""
    if tid and real_kind in ("image", "video"):
        try:
            cs.add_gallery_image(tid, p, "")
        except Exception:  # noqa: BLE001
            pass
    return {
        "kind": real_kind,
        "path": p,
        "filename": staged,
        "name": path.name,
        "node_candidates": _NODE_CANDIDATES.get(real_kind, []),
        "web_url": (meta or {}).get("web_url", ""),
        # A background completion is still the agent putting a result on the
        # canvas, so it answers to the same setting. Asked here rather than
        # trusted from the turn that started it: this lands minutes later, and the
        # setting may have been changed in between.
        "drop": _drop_outputs_into_canvas(),
    }


def _resolve_media_ref(value: str, kind: str = "") -> str | None:
    """Resolve a ComfyUI loader-node file reference to an absolute path.

    Handles the three shapes a Load Image / Load Video widget value can take:
      • an absolute path (e.g. a ``VHS_LoadVideoPath`` "video" widget),
      • an input-dir-relative filename, optionally ComfyUI-annotated with the
        source dir — ``"clip.png [input]"`` / ``"out.png [output]"`` / ``[temp]``
        (and possibly a ``subfolder/name`` prefix),
      • a plain filename already sitting in ComfyUI's input dir.

    Returns the resolved absolute path (as a string) or None when it can't be
    found on disk.
    """
    v = (value or "").strip().strip('"')
    if not v:
        return None
    m = re.match(r"^(?P<name>.*?)(?:\s*\[(?P<t>input|output|temp)\])?\s*$", v)
    name = (m.group("name").strip() if m else v) or v
    ann = (m.group("t") if m else None) or "input"

    p = Path(name)
    if p.is_absolute() and p.exists():
        return str(p.resolve())

    in_dir = _comfy_input_dir()
    bases: list[Path] = []
    if in_dir is not None:
        if ann == "output":
            bases.append(in_dir.parent / "output")
        elif ann == "temp":
            bases.append(in_dir.parent / "temp")
        else:
            bases.append(in_dir)
        if in_dir not in bases:
            bases.append(in_dir)  # always fall back to the input dir
    for base in bases:
        cand = base / name
        if cand.exists():
            return str(cand.resolve())
    # Last resort: interpret as cwd-relative.
    if p.exists():
        return str(p.resolve())
    return None


# agentY collector nodes (AgentYImageCollector) wired as hook anchors carry
# on-disk media paths in their ``files`` widget — plain node data, so the agent can
# see them with NO pre-run (unlike a runtime IMAGE batch, which exists only after a
# run). At most this many are embedded as vision blocks per turn to bound token
# cost; describe_hooks still lists every path so the agent can bind them all.
# Override with AGENTY_MAX_HOOK_IMAGES.
try:
    _MAX_HOOK_IMAGES = max(0, int(os.environ.get("AGENTY_MAX_HOOK_IMAGES", "6") or "6"))
except ValueError:
    _MAX_HOOK_IMAGES = 6


def _collector_hook_images(canvas_hooks: list) -> list[str]:
    """Resolved absolute paths of images held by ``AgentYImageCollector`` nodes
    wired as hook anchors (deduped, existing on disk). Read straight from widget
    data — no graph execution needed."""
    out: list[str] = []
    seen: set = set()
    for h in (canvas_hooks or []):
        if not isinstance(h, dict):
            continue
        for a in (h.get("anchors") or []):
            if not isinstance(a, dict) or str(a.get("type") or "") != "AgentYImageCollector":
                continue
            files = (a.get("widgets") or {}).get("files")
            if not isinstance(files, str):
                continue
            for line in files.splitlines():
                p = line.strip().strip('"')
                # The collector holds video as well since the two were merged, and
                # a .mp4 is not something to embed as a vision block. The agent
                # reads those with analyze_video from the path, which describe_hooks
                # lists either way.
                if not p or p in seen or not _is_image_path(p):
                    continue
                seen.add(p)
                resolved = _resolve_media_ref(p, "image")
                if resolved:
                    out.append(resolved)
    return out


def _tap_hook_tensors(canvas_prompt: dict, canvas_hooks: list,
                      image_paths: list[str]) -> list[str]:
    """Materialise hook anchors that carry a runtime tensor, before the turn runs.

    A loader wired into a hook names its file in its own widgets, so the agent can
    already see it. An anchor fed by anything else — a ``VAEDecode``, an upscaler,
    a mask op — has no file at all: its value exists only inside a run. This
    renders those wires to disk (see :mod:`src.utils.canvas_tap`), annotating the
    hook dicts in place so ``describe_hooks`` lists the paths, and returns the
    turn's image paths with the new images prepended for a vision-capable
    orchestrator. Videos are left out of the vision blocks — the agent reads them
    with ``analyze_video`` from the path in the hook block.
    """
    try:
        from src.utils.canvas_hooks import splice_hook_nodes
        from src.utils.canvas_tap import materialize_hook_tensors

        base, _removed = splice_hook_nodes(canvas_prompt, canvas_hooks)
        paths = materialize_hook_tensors(canvas_hooks, base, resolver=_resolve_media_ref,
                                         on_progress=status_bus.notify)
    except Exception as exc:  # noqa: BLE001 — a tap must never cost the user a turn
        logger.warning("hook tap: skipped (%s)", exc)
        return image_paths
    if not paths:
        return image_paths
    images = [p for p in paths if _is_image_path(p)]
    if not images or not _orchestrator_supports_vision():
        logger.info("hook tap: %d file(s) rendered; paths listed in the [CANVAS HOOKS] "
                    "block (not embedded as vision).", len(paths))
        return image_paths
    existing = set(image_paths)
    return [p for p in images[:_MAX_HOOK_IMAGES] if p not in existing] + image_paths


def _resolve_qa_briefing(canvas_hooks: list | None, thread_id: str):
    """The QA briefing in force for this turn, or None.

    Announces itself on the status bus when one is found: QA that switches on
    silently — spending a strong model's tokens and possibly re-rendering — is
    the kind of thing a user should never have to discover from the bill.
    """
    try:
        from src.utils.qa import resolve_briefing
        briefing = resolve_briefing(hooks=canvas_hooks or [], thread_id=thread_id,
                                    resolver=_resolve_media_ref)
    except Exception as exc:  # noqa: BLE001 — QA must never cost the user a turn
        logger.warning("qa: could not resolve a briefing (%s) — running without QA", exc)
        return None
    if briefing:
        status_bus.notify(f"🔍 QA briefing active — {briefing.describe()}")
    return briefing


def _orchestrator_supports_vision() -> bool:
    """True when the configured orchestrator LLM can accept image content blocks.

    Collector images are embedded into the orchestrator's message as vision blocks
    only when it can actually process them — a text-only model (e.g.
    ``dashscope,qwen3.6-flash``) rejects image content with DashScope's
    "Unexpected item type in content." We can't probe the endpoint, so decide from
    the configured provider/model name, erring toward False (skip embedding — the
    paths are still listed for the agent, and the vision/video agents can read
    them on demand) unless the model is confidently multimodal.

    Resolved through ``role_model`` rather than by reading
    ``llm.pipeline.orchestrator`` directly: that key is only the first of three
    sources (env var, then the per-role pin, then the tier), so a model set the
    normal way — ``llm.tiers.orchestrator`` — left this reading an empty string
    and answering about a model nobody was running.
    """
    try:
        from src.agent import role_model
        # Same default as the factory in agent.py, so an unconfigured install is
        # judged on the model it will actually run, not on an empty string.
        raw = str(role_model("orchestrator", default="claude,claude-haiku-4-5",
                             env_var="ORCHESTRATOR_LLM") or "")
    except Exception:  # noqa: BLE001
        return False
    provider, _, model = raw.lower().partition(",")
    provider = provider.strip()
    model = (model or provider).strip()
    # Operator overrides first, in both directions. Model families move faster
    # than any list kept here, and both mistakes are costly now that this gates
    # what gets sent: a missed multimodal model silently stops images reaching
    # something that could read them, and a wrongly-assumed one breaks every
    # turn of a conversation. Whoever runs the model knows; let them say.
    try:
        from src.utils.settings import load_settings
        _llm = load_settings().get("llm") or {}
        for pattern in (_llm.get("text_only_models") or []):
            if str(pattern).lower().strip() and str(pattern).lower().strip() in model:
                return False
        for pattern in (_llm.get("vision_models") or []):
            if str(pattern).lower().strip() and str(pattern).lower().strip() in model:
                return True
    except Exception:  # noqa: BLE001 — settings must never break the check
        pass
    # Providers whose current models are multimodal across the board.
    if provider in ("claude", "anthropic", "bedrock", "google", "gemini"):
        return True
    # Otherwise require an explicit vision marker in the model id.
    markers = ("-vl", "vl-", "vl:", "vision", "omni", "4o", "gpt-4.1", "o4-",
               "llava", "minicpm-v", "gemma3", "gemma-3", "pixtral", "internvl", "moondream")
    return any(m in model for m in markers)


# ── Content builder (text + attached images/videos -> Strands content blocks) ─

def _build_content(message: str, media_paths: list[str],
                   embed_images: bool | None = None) -> list | str:
    """Build a Strands-compatible content list from text + input media paths.

    Image paths are embedded as vision blocks (downsized to satisfy Claude's
    5 MB / 1568 px constraints) AND listed as file paths. Video paths are not
    embedded (they can't be sent inline) but ARE listed as file paths so the
    agent can wire them into a loader node — same effect as attaching them.

    Embedding is skipped when the orchestrator cannot read images. A text-only
    model rejects the whole request — DashScope answers "Unexpected item type in
    content." — and because the block stays in the message history, every later
    turn in that conversation is rejected too, which is why the symptom looks
    intermittent and why only a fresh conversation appeared to fix it. The paths
    are still listed either way, so the agent can wire them into a loader or hand
    them to the vision agent; it just cannot look at them itself.
    """
    if not media_paths:
        return message or "(no message)"
    if embed_images is None:
        embed_images = _orchestrator_supports_vision()

    from src.tools.image_handling import _downsize, _detect_format, _MAX_IMAGE_BYTES

    blocks: list = []
    img_valid: list[str] = []
    vid_valid: list[str] = []
    for path in media_paths:
        if _is_video_path(path):
            if os.path.exists(path):
                vid_valid.append(path)
            else:
                logger.warning("Input video not found: %s", path)
            continue
        if not embed_images:
            # Still an input the agent must use — just listed, not looked at.
            if os.path.exists(path):
                img_valid.append(path)
            else:
                logger.warning("Input image not found: %s", path)
            continue
        try:
            raw = Path(path).read_bytes()
            img_fmt = _detect_format(path) or "png"
            image_bytes, img_fmt = _downsize(raw, img_fmt)
            if len(image_bytes) > _MAX_IMAGE_BYTES:
                raise ValueError(f"Image still {len(image_bytes):,} bytes after downsize — skipping")
            blocks.append({"image": {"format": img_fmt, "source": {"bytes": image_bytes}}})
            img_valid.append(path)
        except Exception as exc:
            logger.warning("Could not load image %s: %s", path, exc)

    # img_valid counts even with nothing embedded: when the orchestrator is
    # text-only the images survive only as the path list below, and returning the
    # bare message here would silently drop the user's inputs altogether.
    if not blocks and not vid_valid and not img_valid:
        return message or "(no message)"

    path_lines = [f"  - {p}  [image, use this path for ComfyUI input]"
                  for p in img_valid if os.path.exists(p)]
    path_lines += [f"  - {p}  [video, use this path for ComfyUI input]" for p in vid_valid]
    paths_info = ("\n\nAttached input file paths (use these for ComfyUI):\n" + "\n".join(path_lines)
                  if path_lines else "")
    intro = message if message else "The user sent media for processing."
    blocks.insert(0, {"text": intro + paths_info})
    return blocks


# ── Pipeline state save / restore (per thread) ────────────────────────────────

def _memory_agent(pipeline):
    """Return the agent whose message history is this thread's durable memory.

    The free-agent orchestrator owns the whole turn and holds the multi-turn
    conversation, so it IS the per-thread memory that must be cleared/restored on
    every thread switch. (The old ``_free_agent`` gate + ``_assemble_workflow``
    fallback are dead — free-agent is the only path, ``_free_agent`` is never set,
    so this used to return ``None`` and the orchestrator's history was never scoped
    per conversation → it accumulated across every thread = cross-conversation
    bleed. Return the orchestrator directly so reset/restore/save actually target
    the live conversation.)
    """
    return getattr(pipeline, "_orchestrator_agent", None)


def _reset_pipeline_state(pipeline) -> None:
    """Wipe per-conversation state from the shared pipeline singleton."""
    agent = _memory_agent(pipeline)
    if agent is not None and hasattr(agent, "messages"):
        agent.messages.clear()
    existing = getattr(pipeline, "_session", None)
    sid = getattr(existing, "session_id", "default") if existing else "default"
    pipeline._session = AgentSession(session_id=sid)
    pipeline._last_brainbriefing_json = None
    pipeline._last_prior_summary = None


# Words that mean a command even without the slash. Convenient, and for most of
# them harmless — but two of them are also what a person says to a *paused run*.
_BARE_COMMANDS = {"restart", "stop", "unload", "clearhistory", "images", "resend"}
# The answers to a review halt. `stop` there means "end this run at the hook";
# `stop` as a command means "shut the agent host down". They are not remotely the
# same request, and one of them was answering for the other: the panel tells the
# user in those very words to "say continue — or stop to end the run here", and
# the action-bar button sends exactly that.
_HALT_ANSWERS = {"stop", "continue"}


def _is_command(message: str, thread_id: str) -> bool:
    """Whether this message is a slash command rather than something to run.

    A typed `/command` always is. A bare word only is when it cannot be mistaken
    for an answer the user was just asked to give.
    """
    text = (message or "").strip()
    if text.startswith("/"):
        return True
    word = text.lower()
    if word not in _BARE_COMMANDS:
        return False
    if word in _HALT_ANSWERS and _halt_pending(thread_id):
        return False
    return True


def _halt_pending(thread_id: str) -> bool:
    """Is this thread stopped at a review hook, waiting to be told what to do?

    Read from the store rather than from the live pipeline, because the question
    is asked in the HTTP route — before the turn starts and before the pipeline
    has been restored to this thread. Any failure answers "no", which is the
    reading that leaves every command working as it always did.
    """
    try:
        st = cs.load_state(thread_id) or {}
        halt = ((st.get("agent_session") or {}).get("review_halt") or {})
        return bool(halt.get("hook_node_id"))
    except Exception:  # noqa: BLE001
        return False


def _restore_state(pipeline, thread_id: str) -> None:
    """Restore pipeline state for *thread_id* from the memory cache + SQLite store."""
    _reset_pipeline_state(pipeline)
    st = cs.load_state(thread_id)
    if st:
        if st.get("agent_session"):
            try:
                pipeline._session = AgentSession(**st["agent_session"])
            except Exception:
                pass
        pipeline._last_brainbriefing_json = st.get("last_brainbriefing")
        pipeline._last_prior_summary = st.get("last_prior_summary")
    # Scope memory to THIS conversation: the pipeline's session id becomes the
    # thread id, so the auto request-log is written and recalled per-conversation
    # (no bleed between threads). Curated learnings + explicit notes stay in the
    # global namespace and are still shared across conversations.
    try:
        pipeline._session.session_id = thread_id
    except Exception:
        pass
    agent = _memory_agent(pipeline)
    cached = _thread_brain_cache.get(thread_id)
    if agent is not None and hasattr(agent, "messages") and cached is not None:
        agent.messages[:] = cached
    # Rebuild the generated-image gallery into the session so /images and
    # "use image 2" references work after a restart.
    sess = getattr(pipeline, "_session", None)
    if sess is not None:
        try:
            from src.utils.models import GeneratedImage
            gal = cs.get_gallery(thread_id)
            imgs = [
                GeneratedImage(index=i + 1, path=g["path"], caption=g.get("caption", "") or "", turn=0)
                for i, g in enumerate(gal) if os.path.isfile(g["path"])
            ]
            if imgs:
                sess.generated_images = imgs
        except Exception as exc:
            logger.debug("gallery rebuild skipped: %s", exc)


def _save_state(pipeline, thread_id: str) -> None:
    """Snapshot pipeline state for *thread_id* (memory cache + durable SQLite)."""
    agent = _memory_agent(pipeline)
    if agent is not None and hasattr(agent, "messages"):
        _thread_brain_cache[thread_id] = list(agent.messages)
    session = getattr(pipeline, "_session", None)
    try:
        cs.save_state(
            thread_id,
            agent_session=session.model_dump() if session is not None else None,
            last_brainbriefing=getattr(pipeline, "_last_brainbriefing_json", None),
            last_prior_summary=getattr(pipeline, "_last_prior_summary", None),
        )
    except Exception as exc:
        logger.warning("save_state failed for %s: %s", thread_id, exc)


# ── <think>…</think> chunk parser (peels reasoning out of main text) ──────────

def _parse_think_chunk(chunk: str, state: dict) -> tuple[str, str]:
    OPEN, CLOSE = "<think>", "</think>"
    combined = state["buf"] + chunk
    state["buf"] = ""
    normal: list[str] = []
    think: list[str] = []
    while combined:
        if not state["in_think"]:
            idx = combined.find(OPEN)
            if idx == -1:
                for cut in range(min(len(OPEN) - 1, len(combined)), 0, -1):
                    if OPEN[:cut] == combined[-cut:]:
                        normal.append(combined[:-cut]); state["buf"] = combined[-cut:]; combined = ""; break
                else:
                    normal.append(combined); combined = ""
            else:
                normal.append(combined[:idx]); combined = combined[idx + len(OPEN):]; state["in_think"] = True
        else:
            idx = combined.find(CLOSE)
            if idx == -1:
                for cut in range(min(len(CLOSE) - 1, len(combined)), 0, -1):
                    if CLOSE[:cut] == combined[-cut:]:
                        think.append(combined[:-cut]); state["buf"] = combined[-cut:]; combined = ""; break
                else:
                    think.append(combined); combined = ""
            else:
                think.append(combined[:idx]); combined = combined[idx + len(CLOSE):]; state["in_think"] = False
    return "".join(normal), "".join(think)


# ── Conversation auto-title (short 2-3 word summary, like Claude chat) ────────

_TITLE_SYSTEM = (
    "You write ultra-short chat titles. Given the user's first message, reply "
    "with a 2-4 word title in Title Case that summarises the topic. No quotes, "
    "no punctuation, no trailing period, no explanation — output ONLY the title."
)


def _clean_title(raw: str) -> str:
    """Normalise an LLM title reply to a clean 2-4 word label (or '' if unusable)."""
    if not raw:
        return ""
    # Strip any <think> reasoning a Qwen/thinking model may prepend.
    t = re.sub(r"<think>.*?</think>", "", raw, flags=re.S).strip()
    # First non-empty line only.
    t = next((ln.strip() for ln in t.splitlines() if ln.strip()), "")
    t = t.strip().strip('"').strip("'").strip("`*").strip()
    t = t.rstrip(".!,;:").strip()
    words = t.split()
    if len(words) > 6:
        t = " ".join(words[:6])
    return t[:48]


def _generate_and_set_title(thread_id: str, user_text: str) -> None:
    """Background worker: ask the cheap ``llm_functions`` model for a short title
    and rename the thread. Best-effort — leaves the first-message title on any
    failure (e.g. no API key / provider unreachable)."""
    try:
        from src.utils.llm_functions import LLMFunctions
        llm = LLMFunctions.from_settings()
        messages = [
            {"role": "system", "content": _TITLE_SYSTEM},
            {"role": "user", "content": user_text[:2000]},
        ]
        loop = asyncio.new_event_loop()
        try:
            raw = loop.run_until_complete(llm.chat(messages))
        finally:
            _close_loop(loop)
        title = _clean_title(raw)
        if title:
            cs.rename_thread(thread_id, title)
    except Exception as exc:  # noqa: BLE001
        logger.debug("auto-title failed for %s: %s", thread_id, exc)


# ── SSE pipeline runner ───────────────────────────────────────────────────────

def _close_loop(loop) -> None:
    """Finalize pending async generators, then close *loop*.

    Every per-run loop drives async generators (the pipeline stream, the executor,
    the ComfyUI ws-progress stream). On a Stop/cancel or an error these are torn
    down asynchronously — their aclose()/athrow() finalizers get scheduled on the
    loop — so closing it out from under them raises "Task was destroyed but it is
    pending! … async_generator_athrow". Draining shutdown_asyncgens first drives
    those finalizers to completion.
    """
    try:
        loop.run_until_complete(loop.shutdown_asyncgens())
    except Exception:
        pass
    try:
        loop.close()
    except Exception:
        pass


def _run_pipeline_stream(thread_id: str, message: str, image_paths: list[str],
                         out_q: "queue.Queue", req_id: str,
                         canvas_prompt: dict | None = None,
                         canvas_hooks: list | None = None,
                         canvas_selection: list | None = None,
                         open_workflows: list | None = None,
                         dry_run: bool = False, origin: str = "panel") -> None:
    """Run one turn, guaranteeing the SSE queue is always terminated.

    The queue's ``None`` sentinel is what ends the stream, and ``done`` is what
    releases the panel from its streaming state. If the turn runner dies without
    emitting them — an exception anywhere in turn *setup*, which used to run
    outside any handler — the panel keeps a connection open on keep-alives
    forever: it never re-enables sending, queues everything typed after, and
    only a host restart clears it. That failure is indistinguishable from "the
    agent went quiet after a turn", so termination is enforced here rather than
    trusted to every path inside.
    """
    _wd.begin(req_id, thread_id)
    # Everything this turn puts on the queue reaches the panel exactly as before,
    # and is offered to whoever else is watching (the Slack bridge). One wrapper
    # at the one place a turn starts, so nothing inside the turn has to know.
    out_q = turn_bus.tee(out_q, request_id=req_id, thread_id=thread_id,
                         origin=origin, text=message)
    finished = {"emitted": False}
    try:
        _run_pipeline_turn(thread_id, message, image_paths, out_q, req_id, finished,
                           canvas_prompt=canvas_prompt, canvas_hooks=canvas_hooks,
                           canvas_selection=canvas_selection,
                           open_workflows=open_workflows, dry_run=dry_run)
    except BaseException as exc:  # noqa: BLE001 — the stream must close on ANY failure
        logger.error("turn %s died before completing: %s", req_id, exc, exc_info=True)
        # Also into the turn log with a full traceback: the terminal scrollback is
        # usually gone by the time anyone investigates, and this exception is the
        # one thing that names WHICH call in the turn failed.
        import traceback as _tb
        _wd.note(req_id, f"runner raised: {type(exc).__name__}: {exc}\n"
                         + "".join(_tb.format_exception(type(exc), exc, exc.__traceback__)))
        if not finished["emitted"]:
            out_q.put({"type": "error", "message": f"The turn failed to start: {exc}"})
    finally:
        # Registrations are keyed by queue and both unregisters are no-ops when
        # already removed, so repeating them here is safe and stops a dead queue
        # from lingering on either bus.
        try:
            status_bus.unregister_live(out_q)
            notify_bus.unregister_live(out_q)
        except Exception:  # noqa: BLE001
            pass
        with _reply_lock:
            _reply_registry.pop(req_id, None)
            _run_registry.pop(req_id, None)
        if not finished["emitted"]:
            finished["emitted"] = True
            out_q.put({"type": "done"})
            out_q.put(None)
        _wd.end(req_id, "runner exited")


def _run_pipeline_turn(thread_id: str, message: str, image_paths: list[str],
                       out_q: "queue.Queue", req_id: str, finished: dict,
                       canvas_prompt: dict | None = None,
                       canvas_hooks: list | None = None,
                       canvas_selection: list | None = None,
                       open_workflows: list | None = None,
                       dry_run: bool = False) -> None:
    """Drive the pipeline for one turn on a private event loop, pushing SSE dicts
    to *out_q*. Interactive asks register on ``_reply_registry`` so POST
    /agentY/reply can feed the answer thread-safely. Terminates *out_q* with None
    and sets ``finished["emitted"]`` once it has; the caller enforces both.
    """
    pipeline = _agent_ref
    if pipeline is None:
        out_q.put({"type": "error", "message": "pipeline not initialised"})
        finished["emitted"] = True
        out_q.put(None)
        return

    # Surface CLI-side status notices (e.g. the FAISS memory layer initialising)
    # in the panel too: for the life of this turn, status_bus fans notices out
    # onto out_q as live ``status_line`` events (unregistered in finally).
    status_bus.register_live(out_q)
    # Same for structured notifications: a Magnific creation that finishes mid-turn
    # streams its auto-drop live (between-turn ones go via the panel's idle poll).
    notify_bus.register_live(out_q)

    # Hook anchors fed by a mid-graph node carry a tensor, not a file — render them
    # to disk now so the agent has something to look at. Done here rather than in
    # the /chat route so the "rendering…" line streams while it happens.
    if canvas_prompt and canvas_hooks:
        image_paths = _tap_hook_tensors(canvas_prompt, canvas_hooks, image_paths)

    # The QA briefing for this turn. Resolved AFTER the tap so a mood image wired
    # into a qa hook from mid-graph is already a file by the time we read it.
    qa_briefing = _resolve_qa_briefing(canvas_hooks, thread_id)

    _restore_state(pipeline, thread_id)
    session = getattr(pipeline, "_session", None)
    if image_paths and session is not None:
        session.last_user_input_images = image_paths
    content = _build_content(message, image_paths)

    # First assistant turn of a thread → generate a short 2-3 word title in the
    # background (like Claude chat), joined before `done` (see finally). The
    # user message is already persisted, so 0 assistant messages ⇒ first turn.
    title_thread = None
    try:
        _existing = (cs.get_thread(thread_id) or {}).get("messages", [])
        _first_turn = not any(m.get("role") == "assistant" for m in _existing)
    except Exception:
        _first_turn = False
    if _first_turn and message and message.strip():
        title_thread = threading.Thread(
            target=_generate_and_set_title, args=(thread_id, message.strip()),
            name="agentY-autotitle", daemon=True,
        )
        title_thread.start()

    sent_paths: set[str] = set(getattr(session, "current_output_paths", []) or [])
    assistant_parts: list[str] = []
    think_state = {"in_think": False, "buf": ""}
    cur_step = {"name": None}

    def _check_outputs() -> None:
        current = list(getattr(session, "current_output_paths", []) or [])
        for p in current:
            if p in sent_paths or not os.path.isfile(p):
                continue
            sent_paths.add(p)
            kind = "image" if _is_image_path(p) else "video" if _is_video_path(p) else "file"
            # What this file is FOR, recorded by whoever started the run. Resolving
            # it here also writes the sidecar beside the original, and the copy in
            # the input dir gets its own — that one is what a canvas node names.
            role, declared = _output_role(p)
            staged = _stage_into_comfy_input(p) if kind in ("image", "video") else None
            if staged and role:
                _copy_sidecar(p, staged)
            if kind in ("image", "video"):
                try:
                    cs.add_gallery_image(thread_id, p, role)
                except Exception:
                    pass
            out_q.put({
                "type": "output", "kind": kind, "path": p,
                "filename": staged, "name": os.path.basename(p),
                "role": role, "role_declared": declared,
                "node_candidates": _NODE_CANDIDATES.get(kind, []),
                "drop": _drop_outputs_into_canvas(),
            })

    def _emit_paths(paths: list[str], caption: str = "") -> None:
        for p in paths:
            if p in sent_paths or not os.path.isfile(p):
                continue
            sent_paths.add(p)
            kind = "image" if _is_image_path(p) else "video" if _is_video_path(p) else "file"
            staged = _stage_into_comfy_input(p) if kind in ("image", "video") else None
            out_q.put({
                "type": "output", "kind": kind, "path": p, "filename": staged,
                "name": os.path.basename(p), "caption": caption,
                "node_candidates": _NODE_CANDIDATES.get(kind, []),
                "drop": _drop_outputs_into_canvas(),
            })

    def _translate(event: dict) -> None:
        # Interactive asks ------------------------------------------------------
        if event.get("brain_assembly_fail_ask"):
            out_q.put({"type": "ask", "request_id": req_id, "kind": "brain_fail",
                       "prompt": "The Brain failed to assemble a workflow. Describe what it should fix "
                                 "(or send blank to abort).",
                       "latest_workflow_path": event.get("latest_workflow_path", "")})
            return
        if event.get("qa_fail_ask"):
            _emit_paths(event.get("image_paths", []))
            out_q.put({"type": "ask", "request_id": req_id, "kind": "qa_fail",
                       "prompt": "QA check failed. Reply 'yes' to retry this step or 'no' to skip.",
                       "details": [{"path": d.get("path"), "verdict": d.get("verdict")}
                                   for d in event.get("fail_details", [])]})
            return
        if event.get("approval_ask"):
            _check_outputs()
            label = event.get("description") or event.get("label") or "this step"
            out_q.put({"type": "ask", "request_id": req_id, "kind": "approval",
                       "prompt": f"Approval needed — {label}. Reply 'yes' to continue, 'no' to abort, "
                                 "or type a revision note."})
            return
        if event.get("_references_ready"):
            _emit_paths(event.get("paths", []), caption=event.get("caption") or "Web references")
            return

        # Sub-agent step brackets ----------------------------------------------
        for key, name in (("_researcher_start", "Query Templates"), ("_query_templates_start", "Query Templates"),
                          ("_brain_start", "Assemble Workflow"), ("_assemble_workflow_start", "Assemble Workflow"),
                          ("_planner_start", "Planner")):
            if event.get(key):
                cur_step["name"] = name
                out_q.put({"type": "step_start", "name": name})
                return
        if event.get("_story_start"):
            cur_step["name"] = event.get("name") or "Story"
            out_q.put({"type": "step_start", "name": cur_step["name"]})
            return
        for key in ("_researcher_done", "_query_templates_done", "_brain_done",
                    "_assemble_workflow_done", "_planner_done", "_story_done"):
            if event.get(key):
                out_q.put({"type": "step_end", "name": cur_step.get("name"), "raw": event.get("raw", "")})
                cur_step["name"] = None
                return

        # Planner task list -----------------------------------------------------
        if event.get("_plan_ready"):
            out_q.put({"type": "plan", "steps": [s.get("description") for s in event.get("steps", [])]})
            return
        if event.get("_step_start"):
            out_q.put({"type": "plan_step", "idx": event.get("idx"), "state": "running",
                       "description": event.get("description")})
            return
        if event.get("_step_done"):
            out_q.put({"type": "plan_step", "idx": event.get("idx"),
                       "state": "failed" if event.get("failed") else "done"})
            return
        if event.get("_executor_start"):
            out_q.put({"type": "exec", "state": "start"})
            return
        if event.get("_executor_done"):
            _check_outputs()
            out_q.put({"type": "exec", "state": "end"})
            return

        # Tool use --------------------------------------------------------------
        ta = event.get("tool_activity")
        if ta is not None:
            # Show which agent ran the tool: "[orchestrator] upload_image". The
            # panel renders ta["name"] verbatim, so fold the agent label in here.
            agent = ta.get("agent")
            if agent and not str(ta.get("name", "")).startswith("["):
                ta = {**ta, "name": f"[{agent}] {ta.get('name', 'tool')}"}
            out_q.put({"type": "tool", **ta})
            return

        # Canvas node edit — apply to the live graph in the panel -----------------
        cp = event.get("canvas_patch")
        if cp is not None:
            out_q.put({"type": "canvas_patch", **cp})
            return

        # Reasoning -------------------------------------------------------------
        rt = event.get("reasoningText")
        if rt:
            out_q.put({"type": "think", "data": rt})
            return
        if event.get("reasoning_signature") is not None:
            return

        # Text chunks -----------------------------------------------------------
        chunk = event.get("data", "") or ""
        if not chunk:
            return
        # `lstrip` because a drained progress line arrives with the newline that
        # keeps it off the end of the previous one (src.utils.progress_lines).
        # Matching on the raw prefix would drop every download bar into the
        # transcript as text, one line per frame.
        if chunk.lstrip().startswith("⬇️ ") or "🎨 [" in chunk:
            out_q.put({"type": "progress", "data": chunk.strip()})
            return
        # A line relayed from ComfyUI's own terminal. Its own channel, and
        # deliberately NOT appended to assistant_parts: the panel collects it
        # into a collapsible log, and it is ComfyUI talking, not the assistant —
        # persisting it as the reply would put a model-loading dump in the
        # transcript under the agent's name.
        if chunk.lstrip().startswith("🖥"):
            out_q.put({"type": "console", "data": chunk.lstrip()[1:].strip()})
            return
        if "🔍 QA" in chunk or "Vision QA" in chunk:
            out_q.put({"type": "qa", "data": chunk})
            return
        if cur_step.get("name"):
            if cur_step["name"].startswith("Assemble"):
                assistant_parts.append(chunk)
            out_q.put({"type": "step_text", "name": cur_step["name"], "data": chunk})
            _check_outputs()
            return
        normal, think = _parse_think_chunk(chunk, think_state)
        if think:
            out_q.put({"type": "think", "data": think})
        # A drained progress line carries a leading newline so it never lands on
        # the end of the previous one. When it is the FIRST thing said this turn
        # there is nothing to be kept off, and the newline would open the reply
        # with a blank line instead.
        if normal and not assistant_parts:
            normal = normal.lstrip("\n")
        if normal:
            assistant_parts.append(normal)
            out_q.put({"type": "text", "data": normal})
        _check_outputs()

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    qa_queue: asyncio.Queue = asyncio.Queue()

    # Flush the tool-activity / canvas-edit buffers to the SSE stream. The
    # pipeline only drains these right after each Strands event, so when the
    # model goes quiet mid-turn (executing a tool, generating a long tool call)
    # the panel stalls and then burst-updates seconds behind the CLI. A tiny
    # background pump drains them on a short timer instead, keeping the panel in
    # lock-step. The buffers drain atomically (deque under a lock), so the pump
    # and the pipeline's own drain never double-emit an event.
    from src.utils.tool_activity import drain as _drain_tool_activity
    from src.utils.canvas_patch import drain as _drain_canvas_activity
    from src.utils.progress_lines import drain_chunks as _drain_progress_chunks

    def _flush_activity() -> None:
        # Executor progress emitted from inside a tool call (e.g. run_workflow_now,
        # which drives chained hook stages) only reaches the CLI unless drained
        # here — the pipeline's own loop is blocked awaiting the tool. Draining the
        # progress buffer on the pump's short timer streams it to the panel live.
        # drain() is atomic, so this never double-emits with the pipeline's drain.
        for _chunk in _drain_progress_chunks():
            _translate({"data": _chunk})
        for _ta in _drain_tool_activity():
            _translate({"tool_activity": _ta})
        for _cp in _drain_canvas_activity():
            _translate({"canvas_patch": _cp})

    async def _pump() -> None:
        while True:
            await asyncio.sleep(0.2)
            _flush_activity()

    async def _run() -> None:
        pump = asyncio.ensure_future(_pump())
        try:
            async for event in pipeline.stream_async(
                content, qa_reply_queue=qa_queue,
                canvas_prompt=canvas_prompt, canvas_hooks=canvas_hooks,
                canvas_selection=canvas_selection, open_workflows=open_workflows,
                qa_briefing=qa_briefing,
                dry_run=dry_run,
            ):
                if isinstance(event, dict):
                    _translate(event)
        finally:
            pump.cancel()
            try:
                await pump
            except asyncio.CancelledError:
                pass

    # Run the pipeline as a cancellable task so POST /agentY/stop can halt it.
    task = loop.create_task(_run())
    with _reply_lock:
        _reply_registry[req_id] = (loop, qa_queue)
        _run_registry[req_id] = {"loop": loop, "task": task, "thread_id": thread_id}
    # From here until the turn ends, POST /agentY/interject can speak into it.
    interject_bus.open_run(req_id, thread_id)

    stopped = False
    try:
        _wd.phase(req_id, "stream")
        loop.run_until_complete(task)
        _wd.phase(req_id, "post:check_outputs")
        _check_outputs()
    except asyncio.CancelledError:
        # User pressed Stop → the task was cancelled from /agentY/stop.
        stopped = True
        logger.info("pipeline run %s stopped by user", req_id)
    except Exception as exc:
        logger.error("pipeline stream error: %s", exc, exc_info=True)
        out_q.put({"type": "error", "message": str(exc)})
    finally:
        # Everything from here to `done` is local bookkeeping that should take
        # milliseconds. It is also the stretch where the panel has the answer on
        # screen but no `done` yet — so a hang here looks exactly like "the agent
        # went quiet". Each step is breadcrumbed so a stall names its own phase.
        _wd.phase(req_id, "post:unregister")
        status_bus.unregister_live(out_q)
        notify_bus.unregister_live(out_q)
        with _reply_lock:
            _reply_registry.pop(req_id, None)
            _run_registry.pop(req_id, None)
        # An interjection sent after the agent's last tool call has nowhere left
        # to land. Hand it back rather than swallow it: the panel re-queues it as
        # an ordinary message, which is what would have happened without the
        # "send now" click.
        _undelivered = interject_bus.close_run(req_id)
        if _undelivered:
            out_q.put({"type": "interject_undelivered", "texts": _undelivered})
        _wd.phase(req_id, "post:flush_activity")
        _flush_activity()  # emit any tool/canvas activity left after the last event
        _wd.phase(req_id, "post:persist_message")
        text = "".join(assistant_parts).strip()
        if text:
            try:
                cs.add_message(thread_id, "assistant", text)
            except Exception:
                pass
        _wd.phase(req_id, "post:save_state")
        _save_state(pipeline, thread_id)
        _wd.phase(req_id, "post:compression")
        try:
            loop.run_until_complete(pipeline._await_pending_compression())  # type: ignore[attr-defined]
        except Exception:
            pass
        # Let the background auto-title finish (first turn only) so the thread
        # list shows the short summary when the panel refreshes on `done`.
        _wd.phase(req_id, "post:title_join")
        if title_thread is not None:
            title_thread.join(timeout=6.0)
        _wd.phase(req_id, "post:emit_done")
        if stopped:
            out_q.put({"type": "system", "data": "⏹ Stopped."})
        # Terminate the stream BEFORE the loop teardown below: shutdown_asyncgens
        # drives async-generator finalizers (the ComfyUI ws-progress stream among
        # them) and one that refuses to finish would otherwise hold `done` hostage.
        # The panel has everything it needs by this point.
        finished["emitted"] = True
        out_q.put({"type": "done"})
        out_q.put(None)
        _wd.phase(req_id, "post:close_loop")
        _close_loop(loop)


# ── Stop / interrupt helpers ──────────────────────────────────────────────────

def _interrupt_comfy() -> dict:
    """Stop the agent's work in ComfyUI: the running job AND everything it queued.

    Interrupting alone was not stopping. ``POST /interrupt`` ends the job that is
    running and ComfyUI immediately starts the next one, so a run several prompts
    deep — a batch member per variant, a repaired graph queued behind the original
    — carried on until somebody pressed Stop once per remaining item.

    Clearing the whole queue would be the wrong cure: the user queues their own
    work in the same ComfyUI. So only the prompts agentY submitted are removed
    (:mod:`agenty_core.queue_ledger` records each one at submission), and anything
    else in the queue is left exactly where it is.

    The running job is interrupted unless the queue says it is one of the user's —
    a stop meant for the agent should not end somebody else's render. When the
    queue cannot be read at all we interrupt anyway: the person pressed Stop, and
    a stop that does nothing is the worse failure.
    """
    report: dict = {}
    try:
        from agenty_core import queue_ledger
        report = queue_ledger.cancel_ours()
    except Exception as exc:  # noqa: BLE001
        logger.debug("could not clear the agent's ComfyUI queue: %s", exc)

    interrupt = True
    if report.get("ok") and report.get("running") and not report.get("running_is_ours"):
        interrupt = False
        logger.info("stop: leaving ComfyUI's running job alone — it is not ours (%s)",
                    ", ".join(report.get("running") or []))
    if interrupt:
        try:
            from agenty_core.utils.comfyui_client import get_client
            get_client().post("/interrupt", json_data={})
        except Exception as exc:  # noqa: BLE001
            logger.debug("ComfyUI interrupt failed: %s", exc)

    deleted = report.get("deleted") or []
    if deleted:
        logger.info("stop: removed %d queued agentY prompt(s); left %d of the user's",
                    len(deleted), report.get("kept", 0))
    report["interrupted_running"] = interrupt
    return report


def _cancel_run(req_id: str) -> bool:
    """Cancel the active pipeline run *req_id*. Returns True if one matched.

    The agent loop only. Stopping ComfyUI is :func:`_interrupt_comfy`, called once
    by the route rather than from in here — it now reads and edits the queue, and
    doing that twice per Stop meant the second pass reported nothing to remove
    because the first had already removed it.
    """
    with _reply_lock:
        entry = _run_registry.get(req_id)
    if not entry:
        return False
    loop, task = entry.get("loop"), entry.get("task")
    if loop is None or task is None:
        return False
    try:
        loop.call_soon_threadsafe(task.cancel)
    except Exception as exc:  # noqa: BLE001
        logger.warning("could not cancel run %s: %s", req_id, exc)
        return False
    return True


def _cancel_run_by_thread(thread_id: str) -> bool:
    """Cancel the active run for *thread_id* (fallback when no request_id is known)."""
    with _reply_lock:
        rid = next((k for k, v in _run_registry.items() if v.get("thread_id") == thread_id), None)
    return _cancel_run(rid) if rid else False


# ── Slash-command handlers (return a list of SSE event dicts) ─────────────────

def _sys(text: str) -> dict:
    return {"type": "system", "data": text}


def _handle_qa_command(thread_id: str, parts: list[str]) -> list[dict]:
    """``/qa`` — show, set, or clear this conversation's QA briefing.

    Forms:
      ``/qa``                     show what is active (and what is available)
      ``/qa off``                 clear the thread briefing
      ``/qa <name>``              use the named briefing from the briefing dir
      ``/qa <free text>``         use that text as the criteria

    The chat surface exists for turns with no canvas graph; a ``qa`` hook on the
    canvas takes precedence when there is one, since it is the more specific and
    more visible statement.
    """
    from src.utils.qa import (briefing_dir, list_named_briefings, load_named_briefing,
                              qa_settings, resolve_briefing)

    arg = (parts[1] if len(parts) > 1 else "").strip()
    rest = " ".join(parts[1:]).strip()
    available = list_named_briefings()
    avail_txt = (" Available named briefings: "
                 + ", ".join(f"`{n}`" for n in available)) if available else (
        f" No named briefings yet — add one as `{briefing_dir()}/<name>.md`.")

    if not rest:
        active = resolve_briefing(hooks=[], thread_id=thread_id)
        cfg = qa_settings()
        if not cfg["enabled"]:
            return [_sys("🔍 QA is switched off in Settings ▸ qa ▸ enabled.")]
        if not active:
            return [_sys("🔍 No QA briefing set for this conversation.\n\n"
                         "`/qa <name>` to use a named one, `/qa <your criteria>` to type "
                         "one, `/qa off` to clear." + avail_txt)]
        retry = (f"failed outputs are re-generated up to {cfg['max_retries']}×"
                 if cfg["max_retries"] else "failures are reported, not re-generated")
        return [_sys(f"🔍 QA briefing active — {active.describe()}; {retry}.\n\n```\n"
                     f"{active.criteria.strip()[:1200]}\n```")]

    if rest.lower() in ("off", "none", "clear", "stop", "disable"):
        cs.set_qa_briefing(thread_id, None)
        return [_sys("🔍 QA briefing cleared for this conversation. A `qa` hook on the "
                     "canvas still applies to turns that run that graph.")]

    # A bare word that names a briefing on disk loads it; anything else is criteria.
    if " " not in rest:
        named = load_named_briefing(arg)
        if named is not None:
            cs.set_qa_briefing(thread_id, {"criteria": named.criteria,
                                           "reference_paths": list(named.reference_paths),
                                           "name": arg})
            return [_sys(f"🔍 QA briefing set from `{arg}` — {named.describe()}.")]
        if available:
            return [_sys(f"❌ No briefing named `{arg}`.{avail_txt}\n\nTo use that word as "
                         "the criteria itself, write it as a full sentence.")]

    cs.set_qa_briefing(thread_id, {"criteria": rest, "reference_paths": []})
    return [_sys("🔍 QA briefing set for this conversation:\n\n```\n" + rest[:1200] + "\n```\n\n"
                 "Reference images: wire them into a `qa` hook on the canvas, or put them "
                 "in a named briefing's `.refs/` folder. `/qa off` to clear.")]


def _handle_command(thread_id: str, text: str, canvas_prompt: dict | None = None) -> list[dict]:
    low = text.strip().lower()
    parts = text.strip().split(None, 2)
    cmd = parts[0].lower()

    # /help is normally intercepted client-side (agent_chat.js opens the guide in a
    # new tab). This backend branch is the fallback for any client that forwards it —
    # it surfaces the link (and keeps `/help` from ever reaching the LLM).
    if cmd in ("/help", "help", "/docs", "docs", "/guide", "guide"):
        return [_sys(f"📖 agentY usage guide: [{DOCS_URL}]({DOCS_URL})\n\n"
                     "Type `/help` in the chat panel to open it in a new browser tab.")]

    if cmd in ("/restart", "restart"):
        def _do():
            time.sleep(1.0)
            subprocess.Popen([sys.executable] + sys.argv, cwd=str(_project_root()))
            os._exit(0)
        threading.Thread(target=_do, daemon=True).start()
        return [_sys("🔄 Restarting the agent process… the panel will reconnect shortly.")]

    if cmd in ("/stop", "stop", "!stop", "/shutdown", "shutdown"):
        def _do():
            time.sleep(0.5)
            os.kill(os.getpid(), signal.SIGTERM)
        threading.Thread(target=_do, daemon=True).start()
        return [_sys("🛑 Stopping the agent…")]

    if cmd in ("/unload", "unload"):
        try:
            from src.tools.agent_control import unload_ollama_models
            unloaded = unload_ollama_models()
            if unloaded:
                return [_sys("✅ Unloaded: " + ", ".join(f"`{m}`" for m in unloaded))]
            return [_sys("⚠️ No models were unloaded (Ollama unreachable or none loaded).")]
        except Exception as exc:
            return [_sys(f"❌ Unload failed: {exc}")]

    if cmd in ("/clear_vram", "/clearvram", "clear_vram", "clearvram"):
        try:
            from src.tools.comfyui import free_memory
            result = json.loads(free_memory())
            if "error" not in result:
                return [_sys("✅ VRAM cleared — models unloaded and GPU cache freed.")]
            return [_sys(f"❌ Clear VRAM failed: {result.get('error')}")]
        except Exception as exc:
            return [_sys(f"❌ Clear VRAM failed: {exc}")]

    if cmd in ("/qa", "qa"):
        return _handle_qa_command(thread_id, parts)

    if cmd in ("/images", "images", "/gallery", "gallery"):
        gal = [g for g in cs.get_gallery(thread_id) if os.path.isfile(g["path"])]
        if not gal:
            return [_sys("🖼️ No images have been generated in this thread yet.")]
        lines = [f"**{g['idx']}.** `{os.path.basename(g['path'])}`"
                 + (f" — {g['caption']}" if g.get("caption") else "") for g in gal]
        return [_sys(f"🖼️ **{len(gal)} image(s) in this thread** — reference by number "
                     "(\"use image 2\"), recency (\"the last image\"), or description:\n\n" + "\n".join(lines))]

    if cmd in ("/clearhistory", "clearhistory"):
        try:
            n = cs.delete_all_threads(except_id=thread_id)
            return [_sys(f"🗑️ Cleared {n} thread(s) from history (current thread kept).")]
        except Exception as exc:
            return [_sys(f"❌ Failed to clear history: {exc}")]

    if cmd in ("/add_workflow", "add_workflow"):
        if len(parts) < 2:
            return [_sys("⚠️ Usage: `/add_workflow <path_to_workflow.json>` — or "
                         "`/add_workflow canvas <name>` to add the graph open in the canvas.")]
        # `/add_workflow canvas <name>` adds the workflow currently open in the
        # ComfyUI canvas (captured this turn) instead of a JSON file on disk.
        if parts[1].strip().lower() == "canvas":
            name = parts[2].strip() if len(parts) > 2 else ""
            if not name:
                return [_sys("⚠️ Usage: `/add_workflow canvas <name>` — a name for the template is required.")]
            return _add_canvas_workflow(canvas_prompt, name)
        return _add_workflow(parts[1].strip())

    if cmd in ("/remove_workflow", "remove_workflow"):
        if len(parts) < 2:
            return [_sys("⚠️ Usage: `/remove_workflow <template_name>`")]
        return _remove_workflow(parts[1].strip())

    if cmd in ("/switch_model", "switch_model"):
        return _switch_model(parts[1:] )

    if cmd in ("/resend", "resend"):
        return None  # sentinel: handled in the chat route (re-runs first user msg)

    return [_sys(f"❓ Unknown command `{cmd}`.")]


def _add_workflow(path_str: str) -> list[dict]:
    wf_path = Path(path_str)
    if not wf_path.exists():
        return [_sys(f"❌ File not found: `{wf_path}`")]
    try:
        from src.utils.workflow_admin import register_workflow, format_recipe_counts
        wf_data = json.loads(wf_path.read_text(encoding="utf-8"))
        res = register_workflow(wf_data, wf_path.stem, source_path=wf_path)
        return [_sys(f"✅ Workflow `{res['name']}` added — {format_recipe_counts(res['recipes'])}."
                     + (f"\n\n**Description:**\n{res['description']}" if res['description'] else ""))]
    except Exception as exc:
        return [_sys(f"❌ Failed to add workflow: {exc}")]


def _add_canvas_workflow(canvas_prompt: dict | None, name: str) -> list[dict]:
    """Add the workflow currently open in the ComfyUI canvas as a custom template."""
    if not isinstance(canvas_prompt, dict) or not canvas_prompt:
        return [_sys("❌ No workflow is open in the canvas (nothing was captured). "
                     "Open a graph in ComfyUI and try again.")]
    try:
        from src.utils.canvas_hooks import splice_hook_nodes
        from src.utils.workflow_admin import register_workflow, format_recipe_counts
        clean, _removed = splice_hook_nodes(canvas_prompt)  # never persist hook nodes
        res = register_workflow(clean, name)
        return [_sys(f"✅ Canvas workflow added as `{res['name']}` — {format_recipe_counts(res['recipes'])}."
                     + (f"\n\n**Description:**\n{res['description']}" if res['description'] else ""))]
    except Exception as exc:
        return [_sys(f"❌ Failed to add canvas workflow: {exc}")]


def _remove_workflow(name: str) -> list[dict]:
    try:
        from src.utils.workflow_admin import remove_workflow, format_recipe_counts
        res = remove_workflow(name)
        note = "" if res["removed_file"] else " (no template file was on disk)"
        return [_sys(f"✅ Workflow `{res['name']}` removed{note} — {format_recipe_counts(res['recipes'])}.")]
    except Exception as exc:
        return [_sys(f"❌ Failed to remove workflow: {exc}")]


# Agents the pipeline holds LIVE, so a switch takes effect this turn instead of at
# the next start. `orchestrator` is special-cased in _rebuild_agent (it needs its
# delegation tools re-wired), the rest are a plain factory + attribute swap. Every
# OTHER role is still switchable — the setting is written and picked up on restart;
# see _switch_targets, which is the authority on what a target may be.
_LIVE_AGENTS: dict[str, tuple[str, str]] = {
    "query_templates": ("create_query_templates_agent", "_researcher"),
    "info": ("create_info_agent", "_info_agent"),
    "planner": ("create_planner_agent", "_planner_agent"),
}


def _switch_targets() -> tuple[list[str], list[str]]:
    """``(tiers, roles)`` a switch may target.

    Tiers are the normal thing to switch — they are what the settings UI presents,
    and one of them covers every role. Roles remain targetable for the exceptions,
    minus the two whose tier covers exactly one role (`orchestrator`, `coder`):
    for those, tier and role would mean the same thing written to two different
    places, which is a distinction with no difference and a good way to end up
    with an override quietly shadowing a tier.
    """
    from src.agent import _ROLE_TIERS, TIER_LABELS

    tiers = list(TIER_LABELS)
    one_to_one = {t for t in tiers if sum(1 for v in _ROLE_TIERS.values() if v == t) == 1}
    roles = [r for r in _ROLE_TIERS if r not in one_to_one]
    return tiers, roles


def _rebuild_agent(agent_name: str, provider: str, model: str, llm_spec: str) -> str | None:
    """Rebuild one pipeline agent with the given provider/model and swap it into
    the live pipeline. Returns None on success, or an error string."""
    from src.agent import (
        _DASHSCOPE_PROVIDERS, _OPENAI_PROVIDERS, _GEMINI_PROVIDERS,
        _settings as get_settings,
        create_orchestrator_agent,
        create_query_templates_agent, create_info_agent, create_planner_agent,
    )
    if _agent_ref is None:
        return "pipeline not initialised"

    # OpenAI-compatible providers (DashScope/OpenAI/Gemini) read their model from
    # settings, which the caller has already written before getting here.
    _OPENAI_COMPAT = _DASHSCOPE_PROVIDERS | _OPENAI_PROVIDERS | _GEMINI_PROVIDERS

    # The orchestrator is rebuilt specially: its tool list must include the
    # pipeline's delegation tools, and it must be re-wired (skills plugin + live
    # context) via set_orchestrator rather than a plain setattr.
    if agent_name == "orchestrator":
        kwargs = {"llm": provider, "extra_tools": getattr(_agent_ref, "_delegation_tools", None)}
        if provider not in _OPENAI_COMPAT and model:
            kwargs["ollama_model" if provider == "ollama" else "anthropic_model"] = model
        try:
            _agent_ref.set_orchestrator(create_orchestrator_agent(**kwargs))
            return None
        except Exception as exc:  # noqa: BLE001
            return str(exc)

    entry = _LIVE_AGENTS.get(agent_name)
    if entry is None:
        return None  # not held live — the written setting applies at the next start
    factory_name, attr = entry
    factory = {
        "create_query_templates_agent": create_query_templates_agent,
        "create_info_agent": create_info_agent,
        "create_planner_agent": create_planner_agent,
    }[factory_name]
    kwargs = {"llm": provider}
    if provider not in _OPENAI_COMPAT and model:
        kwargs["ollama_model" if provider == "ollama" else "anthropic_model"] = model
    try:
        setattr(_agent_ref, attr, factory(**kwargs))
        return None
    except Exception as exc:  # noqa: BLE001
        return str(exc)


def _switch_model(args: list[str]) -> list[dict]:
    """``/switch_model <target> <provider,model>`` — set the model for a tier, a
    single role, or everything.

    Writes to ``settings.local.json`` so the choice survives a restart: this picker
    lives in the composer bar and is how most model changes actually get made, so
    silently reverting it at the next start would be the bigger surprise. Agents the
    pipeline holds live are rebuilt immediately; the rest apply at the next start,
    and the reply says which is which rather than implying everything took effect.
    """
    from src.agent import _ROLE_TIERS
    from src.utils.settings import set_local

    tiers, roles = _switch_targets()
    if len(args) < 2:
        return [_sys(
            "⚠️ Usage: `/switch_model <target> <provider,model>`\n\n"
            f"**Tiers** (the usual thing to switch): `{'`, `'.join(tiers)}`\n\n"
            f"**Single roles** — writes a per-role override that beats its tier: "
            f"`{'`, `'.join(roles)}`\n\n"
            "`all` sets every tier at once — e.g. `/switch_model all claude,claude-haiku-4-5`."
        )]
    target = args[0].lower().strip()
    llm_spec = args[1].strip()
    provider, _, model = llm_spec.partition(",")
    provider = provider.strip().lower()
    model = model.strip()

    from src.agent import _DASHSCOPE_PROVIDERS, _OPENAI_PROVIDERS, _GEMINI_PROVIDERS
    if provider not in ({"claude", "ollama"} | _DASHSCOPE_PROVIDERS | _OPENAI_PROVIDERS | _GEMINI_PROVIDERS):
        return [_sys(f"❌ Unknown provider `{provider}`. Use `claude`, `ollama`, "
                     "`dashscope`, `openai`, or `google`.")]

    # Resolve the target into the setting it writes and the roles it affects.
    if target == "all":
        overrides = {"llm": {"tiers": {t: llm_spec for t in tiers}}}
        affected = set(_ROLE_TIERS)
        what = f"All {len(tiers)} tiers"
    elif target in tiers:
        overrides = {"llm": {"tiers": {target: llm_spec}}}
        affected = {r for r, t in _ROLE_TIERS.items() if t == target}
        what = f"Tier `{target}`"
    elif target in roles:
        overrides = {"llm": {"pipeline": {target: llm_spec}}}
        affected = {target}
        what = f"Role `{target}`"
    else:
        return [_sys(f"❌ Unknown target `{target}`.\n\n"
                     f"Tiers: `{'`, `'.join(tiers)}`\n\n"
                     f"Roles: `{'`, `'.join(roles)}`\n\nOr `all`.")]

    try:
        set_local(overrides)
    except Exception as exc:  # noqa: BLE001
        return [_sys(f"❌ Could not save the setting: {exc}")]
    # The merged settings are cached in-process; refresh so the rebuilds below (and
    # the rest of this turn) read what we just wrote.
    try:
        from src.utils.settings import load_settings
        load_settings(refresh=True)
        from src.agent import _settings as get_settings
        _llm = get_settings().setdefault("llm", {})
        _llm.setdefault("tiers", {}).update(overrides["llm"].get("tiers", {}))
        _llm.setdefault("pipeline", {}).update(overrides["llm"].get("pipeline", {}))
    except Exception as exc:  # noqa: BLE001
        logger.debug("switch_model: settings cache refresh skipped: %s", exc)

    # Rebuild whatever the pipeline holds live so the change lands this turn. With
    # no pipeline running there is nothing to swap and the setting simply applies at
    # start — that is not a failure worth warning about.
    live = (sorted(affected & ({"orchestrator"} | set(_LIVE_AGENTS)))
            if _agent_ref is not None else [])
    failures = [f"`{r}`: {err}" for r in live
                if (err := _rebuild_agent(r, provider, model, llm_spec))]
    ok_live = [r for r in live if not any(f.startswith(f"`{r}`") for f in failures)]
    deferred = sorted(affected - set(ok_live))

    lines = [f"✅ {what} → `{llm_spec}` (saved to `settings.local.json`)."]
    if ok_live:
        lines.append("Live now: " + ", ".join(f"`{r}`" for r in ok_live) + ".")
    if deferred:
        lines.append("Applies on the next agent start: "
                     + ", ".join(f"`{r}`" for r in deferred) + ".")
    if target in roles:
        lines.append(f"_`{target}` now ignores its `{_ROLE_TIERS.get(target)}` tier "
                     "until you clear the override in Settings._")
    if failures:
        lines.append("⚠️ Some live rebuilds failed:\n" + "\n".join(failures))
    return [_sys("\n\n".join(lines))]


# ── Legacy ComfyUI → agent image-review bridge (kept) ─────────────────────────

def add_preview_job(job_id: str, label: str, origin_pos: list | None = None) -> None:
    with _lock:
        _pending_previews[job_id] = {"job_id": job_id, "label": label, "origin_pos": origin_pos or [100, 100]}


def clear_preview_job(job_id: str) -> None:
    with _lock:
        _pending_previews[job_id] = {"job_id": job_id, "clear": True}


def _dispatch_to_agent(message: str, image_paths: list[str], node_id: str | None) -> None:
    if _agent_ref is None:
        logger.error("No agent registered")
        return

    def _run():
        content = _build_content(message, image_paths)

        async def _stream():
            acc = []
            async for event in _agent_ref.stream_async(content):
                if isinstance(event, dict) and event.get("data"):
                    acc.append(event["data"])
            if node_id and acc:
                with _lock:
                    _node_responses[str(node_id)] = "".join(acc)

        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(_stream())
        finally:
            _close_loop(loop)

    threading.Thread(target=_run, name="agentY-review-dispatch", daemon=True).start()


# ── Model discovery (for the panel's quick-switch dropdown) ───────────────────
#
# The cloud vendors are enumerated **live** from their own ``/models`` endpoints
# so the dropdown never drifts as new models ship (Anthropic's list, and whatever
# the configured DashScope endpoint actually serves — which can include DeepSeek,
# GLM, Kimi alongside Qwen). Results are curated (snapshot/translation/OCR noise
# dropped), cached briefly, and fall back to the static catalogs below when an
# endpoint is unreachable. Each entry is [ "<provider>,<model>", "Display name" ]
# — the provider,model string is exactly what /switch_model expects.

# Fallbacks — used only when a provider's /models endpoint can't be reached.
_ANTHROPIC_FALLBACK = [
    ["claude,claude-opus-4-8", "Claude Opus 4.8"],
    ["claude,claude-sonnet-5", "Claude Sonnet 5"],
    ["claude,claude-haiku-4-5", "Claude Haiku 4.5"],
    ["claude,claude-sonnet-4-5", "Claude Sonnet 4.5"],
]
_DASHSCOPE_FALLBACK = [
    ["dashscope,qwen3.6-flash", "Qwen3.6 Flash"],
    ["dashscope,qwen3.7-plus", "Qwen3.7 Plus"],
    ["dashscope,qwen3-max", "Qwen3 Max"],
    ["dashscope,qwen3-coder-plus", "Qwen3 Coder Plus"],
    ["dashscope,deepseek-v4-pro", "DeepSeek V4 Pro"],
]
_OPENAI_FALLBACK = [
    ["openai,gpt-4o", "gpt-4o"],
    ["openai,gpt-4o-mini", "gpt-4o-mini"],
    ["openai,o3", "o3"],
]
_GEMINI_FALLBACK = [
    ["google,gemini-2.5-pro", "gemini-2.5-pro"],
    ["google,gemini-2.5-flash", "gemini-2.5-flash"],
]

# Live-list cache (avoid hitting the endpoints on every panel load).
_MODEL_CACHE: dict = {}
_MODEL_CACHE_TTL = 300  # seconds — full cache life once every vendor enumerated cleanly
_MODEL_CACHE_TTL_RETRY = 20  # seconds — short life when the live-only vendor (Ollama) was
# missing, so a startup race or a transient /api/tags timeout self-heals fast instead of
# hiding Ollama for a full TTL (it has no static fallback, unlike the cloud vendors).

# DashScope curation: drop dated snapshots and the translation / OCR variants —
# they bloat the dropdown without adding a distinct chat/coding/vision model.
_DS_SNAPSHOT_RE = re.compile(r"-\d{4}-\d{2}-\d{2}$")

# Cosmetic label prettifiers.
_MODEL_BRAND = {"qwen": "Qwen", "deepseek": "DeepSeek", "glm": "GLM",
                "kimi": "Kimi", "claude": "Claude"}
_MODEL_ACRONYM = {"vl": "VL", "ocr": "OCR", "moe": "MoE", "mt": "MT",
                  "a2b": "A2B", "a3b": "A3B", "a10b": "A10B", "a17b": "A17B",
                  "a22b": "A22B", "a35b": "A35B"}


def _dashscope_keep(mid: str) -> bool:
    """True if *mid* is a real, distinct model worth listing (not noise)."""
    if _DS_SNAPSHOT_RE.search(mid):
        return False
    if mid.startswith("qwen-mt") or "-mt-" in mid:
        return False
    if "ocr" in mid:
        return False
    return True


def _prettify_model_id(mid: str) -> str:
    """Turn a raw model id into a readable label (e.g. deepseek-v4-pro → DeepSeek V4 Pro)."""
    words: list[str] = []
    for tok in mid.split("-"):
        low = tok.lower()
        if low in _MODEL_ACRONYM:
            words.append(_MODEL_ACRONYM[low])
            continue
        m = re.match(r"^([a-z]+)(.*)$", tok)
        if m and m.group(1) in _MODEL_BRAND:  # brand prefix: qwen3.6 → Qwen3.6
            words.append(_MODEL_BRAND[m.group(1)] + m.group(2))
        elif tok:
            words.append(tok[:1].upper() + tok[1:])
    return " ".join(words) or mid


def _fetch_anthropic_models(key: str) -> list[list[str]]:
    """Live Anthropic model list (newest first) via GET /v1/models."""
    import requests  # noqa: PLC0415
    rows: list[tuple[str, list[str]]] = []
    after: str | None = None
    for _ in range(10):  # pagination guard
        params: dict = {"limit": 1000}
        if after:
            params["after_id"] = after
        resp = requests.get(
            "https://api.anthropic.com/v1/models",
            headers={"x-api-key": key, "anthropic-version": "2023-06-01"},
            params=params, timeout=6,
        )
        resp.raise_for_status()
        j = resp.json()
        for m in j.get("data", []):
            mid = m.get("id")
            if mid:
                rows.append((m.get("created_at", ""),
                             [f"claude,{mid}", m.get("display_name") or mid]))
        if not j.get("has_more"):
            break
        after = j.get("last_id")
        if not after:
            break
    rows.sort(key=lambda t: t[0], reverse=True)
    return [pair for _, pair in rows]


def _fetch_dashscope_models(key: str, base_url: str) -> list[list[str]]:
    """Live DashScope model list via the OpenAI-compatible GET {base}/models.

    Curated (snapshot/translation/OCR variants dropped) and sorted by id so
    families group together.
    """
    import requests  # noqa: PLC0415
    if not base_url:
        raise ValueError("no DashScope base_url configured")
    resp = requests.get(base_url.rstrip("/") + "/models",
                        headers={"Authorization": f"Bearer {key}"}, timeout=6)
    resp.raise_for_status()
    ids = sorted({m.get("id", "") for m in resp.json().get("data", []) if m.get("id")})
    return [[f"dashscope,{mid}", _prettify_model_id(mid)]
            for mid in ids if _dashscope_keep(mid)]


# OpenAI curation: keep chat / reasoning models, drop the non-conversational ones
# (embeddings, audio, image, moderation, …) that the /models endpoint also lists.
_OPENAI_KEEP_RE = re.compile(r"^(gpt-|chatgpt|o[1-9])")
_OPENAI_DROP = ("embedding", "whisper", "tts", "audio", "realtime", "transcribe",
                "image", "dall-e", "moderation", "search", "instruct")
_GEMINI_DROP = ("embedding", "aqa", "imagen", "-vision")


def _openai_keep(mid: str) -> bool:
    if not _OPENAI_KEEP_RE.match(mid):
        return False
    return not any(s in mid for s in _OPENAI_DROP)


def _gemini_keep(mid: str) -> bool:
    if "gemini" not in mid:
        return False
    return not any(s in mid for s in _GEMINI_DROP)


def _fetch_openai_models(key: str, base_url: str) -> list[list[str]]:
    """Live OpenAI model list (chat/reasoning only), newest first, via GET /models."""
    import requests  # noqa: PLC0415
    resp = requests.get(base_url.rstrip("/") + "/models",
                        headers={"Authorization": f"Bearer {key}"}, timeout=6)
    resp.raise_for_status()
    rows: list[tuple[int, list[str]]] = []
    for m in resp.json().get("data", []):
        mid = m.get("id", "")
        if mid and _openai_keep(mid):
            rows.append((int(m.get("created", 0) or 0), [f"openai,{mid}", mid]))
    rows.sort(key=lambda t: t[0], reverse=True)
    return [pair for _, pair in rows]


def _fetch_gemini_models(key: str, base_url: str) -> list[list[str]]:
    """Live Gemini model list via the OpenAI-compatible GET {base}/models.

    Ids arrive namespaced (``models/gemini-…``); the ``models/`` prefix is stripped
    so the spec matches what the pipeline expects (``google,gemini-…``).
    """
    import requests  # noqa: PLC0415
    resp = requests.get(base_url.rstrip("/") + "/models",
                        headers={"Authorization": f"Bearer {key}"}, timeout=6)
    resp.raise_for_status()
    ids = {(m.get("id", "") or "").split("/")[-1] for m in resp.json().get("data", [])}
    keep = sorted(mid for mid in ids if mid and _gemini_keep(mid))
    return [[f"google,{mid}", mid] for mid in keep]


def _available_models() -> dict:
    """Return {vendor: [[spec, label], …]} for every vendor currently usable.

    Cloud vendors are enumerated live from their ``/models`` endpoints (cached for
    ``_MODEL_CACHE_TTL`` seconds); if an endpoint is unreachable the static
    ``*_FALLBACK`` catalog is used instead. Ollama's installed models are listed
    live via ``GET {host}/api/tags``. A vendor appears only when reachable /
    configured.
    """
    now = time.time()
    cached = _MODEL_CACHE.get("groups")
    ttl = _MODEL_CACHE.get("ttl", _MODEL_CACHE_TTL)
    if cached is not None and (now - _MODEL_CACHE.get("ts", 0)) < ttl:
        return cached

    groups: dict[str, list] = {}

    anthropic_key = os.environ.get("ANTHROPIC_API_KEY")
    if anthropic_key:
        try:
            models = _fetch_anthropic_models(anthropic_key)
            groups["Anthropic"] = models or list(_ANTHROPIC_FALLBACK)
        except Exception as exc:  # noqa: BLE001
            logger.debug("Anthropic live model list failed (%s); using fallback", exc)
            groups["Anthropic"] = list(_ANTHROPIC_FALLBACK)

    dashscope_key = os.environ.get("DASHSCOPE_API_KEY") or os.environ.get("ALIBABA_API_KEY")
    if dashscope_key:
        try:
            from src.agent import _cfg  # noqa: PLC0415
            base = str(_cfg("DASHSCOPE_BASE_URL", "dashscope", "base_url", default="")) \
                or os.environ.get("DASHSCOPE_BASE_URL", "")
            models = _fetch_dashscope_models(dashscope_key, base)
            groups["Alibaba (DashScope)"] = models or list(_DASHSCOPE_FALLBACK)
        except Exception as exc:  # noqa: BLE001
            logger.debug("DashScope live model list failed (%s); using fallback", exc)
            groups["Alibaba (DashScope)"] = list(_DASHSCOPE_FALLBACK)

    # OpenAI and Google Gemini appear only when their key is configured — same
    # gate as the vendors above ("don't show if not set up").
    openai_key = os.environ.get("OPENAI_API_KEY")
    if openai_key:
        try:
            from src.agent import _cfg  # noqa: PLC0415
            base = str(_cfg("OPENAI_BASE_URL", "openai", "base_url",
                            default="https://api.openai.com/v1")) or "https://api.openai.com/v1"
            models = _fetch_openai_models(openai_key, base)
            groups["OpenAI"] = models or list(_OPENAI_FALLBACK)
        except Exception as exc:  # noqa: BLE001
            logger.debug("OpenAI live model list failed (%s); using fallback", exc)
            groups["OpenAI"] = list(_OPENAI_FALLBACK)

    gemini_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if gemini_key:
        try:
            from src.agent import _cfg  # noqa: PLC0415
            base = str(_cfg("GEMINI_BASE_URL", "google", "base_url",
                            default="https://generativelanguage.googleapis.com/v1beta/openai/")) \
                or "https://generativelanguage.googleapis.com/v1beta/openai/"
            models = _fetch_gemini_models(gemini_key, base)
            groups["Google (Gemini)"] = models or list(_GEMINI_FALLBACK)
        except Exception as exc:  # noqa: BLE001
            logger.debug("Gemini live model list failed (%s); using fallback", exc)
            groups["Google (Gemini)"] = list(_GEMINI_FALLBACK)

    # Ollama has no static fallback — its models are known only from a live query,
    # so any miss (daemon not up yet, a slow /api/tags under GPU load) drops the
    # whole vendor. Track whether it was enumerated so a miss is cached only briefly
    # and retried, rather than hiding Ollama for the full TTL.
    ollama_seen = False
    try:
        import requests  # noqa: PLC0415
        from src.agent import _cfg  # noqa: PLC0415
        host = str(_cfg("OLLAMA_HOST", "ollama", "host", default="http://localhost:11434"))
        resp = requests.get(f"{host}/api/tags", timeout=5)
        resp.raise_for_status()
        names = sorted({m.get("name", "") for m in resp.json().get("models", []) if m.get("name")})
        if names:
            groups["Ollama"] = [[f"ollama,{n}", n] for n in names]
            ollama_seen = True
    except Exception as exc:  # noqa: BLE001 — Ollama not running ⇒ hide the vendor
        logger.debug("Ollama model list unavailable: %s", exc)

    _MODEL_CACHE["groups"] = groups
    _MODEL_CACHE["ts"] = now
    _MODEL_CACHE["ttl"] = _MODEL_CACHE_TTL if ollama_seen else _MODEL_CACHE_TTL_RETRY
    return groups


# ── Application settings (.env auth keys + config/settings.json) ──────────────

# Auth / host keys surfaced in the settings modal even when absent from .env, so
# the user can fill them in. Any additional keys already present in .env are
# merged in on read.
# Auth/connection keys surfaced in the settings UI's ".env" section. Host/port
# (agent_server_url) and the conversation DB live in the settings files now, not here.
# The DashScope endpoint sits right beneath its API key; it is pre-seeded from the
# merged settings (llm.dashscope.base_url) in the GET so the field is never blank.
_KNOWN_ENV_KEYS = [
    "HF_TOKEN", "ANTHROPIC_API_KEY", "COMFYUI_API_KEY",
    "DASHSCOPE_API_KEY", "DASHSCOPE_BASE_URL",
    "OPENAI_API_KEY", "GEMINI_API_KEY",
    # Slack bridge. Listed here so the settings UI offers the three fields the
    # workspace side cannot be set up without, rather than sending someone to a
    # text editor for the one integration whose whole point is not being at the
    # machine. SLACK_ALLOWED_USERS is not a secret and is deliberately not named
    # like one — it renders as a plain, readable field.
    "SLACK_BOT_TOKEN", "SLACK_APP_TOKEN", "SLACK_ALLOWED_USERS",
]


def _env_path() -> Path:
    return _project_root() / ".env"


# ── Request admission ────────────────────────────────────────────────────────
# The rules live in src.utils.api_guard (pure, and tested there); this is the
# Flask end of them: read the settings, read the request, ask.
from src.utils import api_guard as _api_guard  # noqa: E402
from src.utils import key_age as _key_age  # noqa: E402

# The port actually bound, recorded at startup. Needed to tell an Origin on our
# own port (a viewer page we served) from one that merely claims to be local.
_bound_port: int = 0


def set_bound_port(port: int) -> None:
    """Record the port actually bound, for the Origin check's port rule."""
    global _bound_port
    _bound_port = int(port or 0)


def _security_settings() -> dict:
    try:
        from src.agent import _load_settings
        cfg = (_load_settings() or {}).get("security")
        return cfg if isinstance(cfg, dict) else {}
    except Exception:  # noqa: BLE001
        return {}


# Refusals seen recently: {(path, reason) -> [first_logged_at, count]}. A panel
# that cannot authenticate does not fail once — it polls, so the same refusal
# arrives several times a second. Logging each one buries the startup banner, the
# key-age warning and every real error under a wall of identical lines, which is
# how a useful message becomes noise nobody reads.
_refusals: dict[tuple[str, str], list] = {}
_REFUSAL_QUIET_SECONDS = 60.0


def _log_refusal(method: str, path: str, why: str) -> None:
    """Say it once, then say how often it kept happening — not every time."""
    key = (str(path), str(why)[:60])
    now = time.time()
    seen = _refusals.get(key)
    if seen is None:
        # Count starts at 0, not 1: it counts what has been SUPPRESSED since the
        # last line, and this one is being printed. Starting at 1 made the summary
        # report the occurrence it had already reported.
        _refusals[key] = [now, 0]
        logger.warning("refused %s %s: %s", method, path, why)
        return
    seen[1] += 1
    if now - seen[0] < _REFUSAL_QUIET_SECONDS:
        return
    logger.warning("refused %s %s (%d more times in the last %ds): %s",
                   method, path, seen[1], int(now - seen[0]), why)
    _refusals[key] = [now, 0]


def _guard_verdict(request) -> tuple[bool, str]:
    """Admit this request? Memoised per request — before_request and after_request
    both need the answer, and settings reads are not free."""
    try:
        from flask import g
        cached = getattr(g, "_agentY_verdict", None)
        if cached is not None:
            return cached
    except Exception:  # noqa: BLE001
        g = None  # type: ignore[assignment]

    sec = _security_settings()
    try:
        from src.agent import _load_settings
        comfy = str((_load_settings() or {}).get("comfyui_url", "http://127.0.0.1:8188"))
    except Exception:  # noqa: BLE001
        comfy = "http://127.0.0.1:8188"

    result = _api_guard.verdict(
        method=request.method,
        path=request.path,
        host_header=request.headers.get("Host", ""),
        origin=request.headers.get("Origin", ""),
        token=request.headers.get(_api_guard.TOKEN_HEADER, ""),
        expected_token=_api_guard.session_token(_project_root()),
        agent_port=_bound_port,
        comfyui_url=comfy,
        allowed_hosts=tuple(sec.get("allowed_hosts") or ()),
        allowed_origins=tuple(sec.get("allowed_origins") or ()),
        check_origin=bool(sec.get("check_origin", True)),
        require_token=bool(sec.get("require_token", True)),
    )
    try:
        if g is not None:
            g._agentY_verdict = result
    except Exception:  # noqa: BLE001
        pass
    return result


def _pricing_config_path() -> Path:
    return _project_root() / "config" / "pricing.json"


def _load_pricing_config() -> dict:
    """Return config/pricing.json (user-editable model prices, USD per Mtok)."""
    try:
        return json.loads(_pricing_config_path().read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return {"models": {}, "provider_defaults": {}}


def _save_pricing_config(data: dict) -> None:
    """Persist the whole config/pricing.json (replaces the file)."""
    _pricing_config_path().write_text(
        json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


# ── Token-usage overview ──────────────────────────────────────────────────────
# Parse the token-usage log (written by src.agent.TokenUsageHookProvider) into
# per-model aggregates for the "Token Usage" panel. Each line looks like:
#   2026-07-10 13:20:22 [orchestrator] tool=patch_workflow
#     delta=+623303in/+247out/+0cache_read/+0cache_write
#     total=…  cost=$53.16/tokens=… model=claude/claude-haiku-4-5
# The *delta* fields (per-call increments) are what we sum — the *total* column
# is a running accumulation and must not be summed. ``model=`` is recent; lines
# that predate it fall back to their role (keyed ``role:<role>``).
_RE_TOKENS_HEAD = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\s+\[([^\]]+)\]")
_RE_TOKENS_DELTA = re.compile(
    r"delta=\+?(-?\d+)in/\+?(-?\d+)out/\+?(-?\d+)cache_read/\+?(-?\d+)cache_write"
)
_RE_TOKENS_MODEL = re.compile(r"\bmodel=(\S+)")
_RE_TOKENS_TOTAL = re.compile(r"total=(\d+)in/")


def _parse_token_usage(from_ts: float | None, to_ts: float | None) -> dict:
    """Aggregate the token-usage log by model within [from_ts, to_ts] (epoch secs).

    A thin wrapper over :func:`_aggregate_token_usage` with a simple range check.
    """
    def _accept(ts: float) -> bool:
        if from_ts is not None and ts < from_ts:
            return False
        if to_ts is not None and ts > to_ts:
            return False
        return True
    return _aggregate_token_usage(_accept)


def _aggregate_token_usage(accept) -> dict:
    """Aggregate the token-usage log by model, keeping lines *accept(ts)* passes.

    ``accept`` is ``Callable[[float], bool]`` over each line's epoch timestamp, so
    callers can scope by a simple ``[from, to]`` range (:func:`_parse_token_usage`)
    or by a set of per-turn windows (:func:`_parse_token_usage_thread`).

    Returns ``all_models`` (every model key ever seen — for the filter dropdown,
    independent of the scope), ``rows`` (per-model input/output/cache/cost/calls
    inside the scope), the summed ``total``, and the log's overall time range.
    """
    from src.agent import _load_settings
    from src.utils.costs import compute_cost_from_usage

    class _Meta:
        """Minimal stand-in carrying model metadata for cost lookup."""
        __slots__ = ("_cost_meta",)

        def __init__(self, provider: str, model_id: str) -> None:
            self._cost_meta = {
                "provider": provider,
                "model_id": model_id,
                "is_ollama": provider == "ollama",
            }

    rel = (_load_settings() or {}).get("tokens_usage_log", "./.logs/tokens_usage.log")
    path = _project_root() / rel

    all_models: set[str] = set()
    agg: dict[str, dict] = {}
    log_min: float | None = None
    log_max: float | None = None

    def _bucket(key: str) -> dict:
        b = agg.get(key)
        if b is None:
            b = {"model": key, "input": 0, "output": 0,
                 "cache_read": 0, "cache_write": 0, "cost": 0.0, "calls": 0}
            agg[key] = b
        return b

    if path.exists():
        with path.open("r", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                head = _RE_TOKENS_HEAD.match(line)
                if not head:
                    continue
                delta = _RE_TOKENS_DELTA.search(line)
                if not delta:
                    continue
                try:
                    ts = time.mktime(time.strptime(head.group(1), "%Y-%m-%d %H:%M:%S"))
                except (ValueError, OverflowError):
                    continue

                mm = _RE_TOKENS_MODEL.search(line)
                model = mm.group(1) if mm else ""
                key = model or f"role:{head.group(2)}"

                # Dropdown + log range span the whole file, not just the window.
                all_models.add(key)
                if log_min is None or ts < log_min:
                    log_min = ts
                if log_max is None or ts > log_max:
                    log_max = ts

                if not accept(ts):
                    continue

                # Clamp negatives: a fresh agent resets its accumulator, so the
                # first post-reset delta is already correct and never negative.
                d_in = max(0, int(delta.group(1)))
                d_out = max(0, int(delta.group(2)))
                d_cr = max(0, int(delta.group(3)))
                d_cw = max(0, int(delta.group(4)))

                b = _bucket(key)
                b["input"] += d_in
                b["output"] += d_out
                b["cache_read"] += d_cr
                b["cache_write"] += d_cw
                b["calls"] += 1

                # Recompute cost from *this delta* (the logged cost column is
                # cumulative and cannot be summed). Only possible when the model
                # is known; role-only historical lines contribute 0 cost.
                if model:
                    provider, _, model_id = model.partition("/")
                    try:
                        cost, _ = compute_cost_from_usage(
                            {"inputTokens": d_in, "outputTokens": d_out,
                             "cacheReadInputTokens": d_cr, "cacheWriteInputTokens": d_cw},
                            _Meta(provider, model_id),
                        )
                        b["cost"] += cost
                    except Exception:
                        pass

    rows = sorted(agg.values(), key=lambda r: r["input"] + r["output"], reverse=True)
    total = {"input": 0, "output": 0, "cache_read": 0,
             "cache_write": 0, "cost": 0.0, "calls": 0}
    for r in rows:
        for k in total:
            total[k] += r[k]
        r["cost"] = round(r["cost"], 4)
    total["cost"] = round(total["cost"], 4)

    return {
        "ok": True,
        "all_models": sorted(all_models),
        "rows": rows,
        "total": total,
        "log_range": {"min": log_min, "max": log_max},
    }


# The token log is second-resolution and its timestamps are floored to the whole
# second, while conversation message timestamps are sub-second ``time.time()``
# floats. Pad each per-turn window by this many seconds so a tool call logged a
# hair before/after the bracketing message still falls inside the window. A few
# seconds is negligible in a single-user host where turns never interleave.
_THREAD_WINDOW_PAD_SECS = 3.0


def _thread_windows(thread_id: str) -> list[tuple[float, float]]:
    """Per-turn ``[start, end]`` epoch windows bracketing *thread_id*'s token use.

    A turn is a user message (saved on receipt) followed by the agent's work and
    then the assistant reply (saved on completion); every token-log line for the
    turn falls between the two. So each user message opens a window that closes at
    the next message's timestamp (its reply), or ``now`` for an in-flight turn.

    Using tight per-turn windows — rather than one ``[created_at, updated_at]``
    span — keeps a resumed conversation from also claiming the token usage of
    other conversations that ran in between. Falls back to the thread's own
    created/updated span if it somehow has no user messages.
    """
    try:
        from src.utils import conversation_store as cs
        thread = cs.get_thread(thread_id)
    except Exception:
        thread = None
    if not thread:
        return []
    msgs = thread.get("messages") or []
    now = time.time()
    windows: list[tuple[float, float]] = []
    for i, m in enumerate(msgs):
        if (m.get("role") or "") != "user":
            continue
        start = float(m.get("created_at") or 0)
        end = float(msgs[i + 1].get("created_at") or 0) if i + 1 < len(msgs) else now
        if end < start:
            end = now
        windows.append((start - _THREAD_WINDOW_PAD_SECS, end + _THREAD_WINDOW_PAD_SECS))
    if not windows:
        c0 = float(thread.get("created_at") or 0)
        c1 = float(thread.get("updated_at") or c0)
        if c1 >= c0 > 0:
            windows.append((c0 - _THREAD_WINDOW_PAD_SECS, c1 + _THREAD_WINDOW_PAD_SECS))
    return windows


def _parse_token_usage_thread(thread_id: str) -> dict:
    """Aggregate token usage for the conversation *thread_id* (the "Current run"
    scope in the Token Usage overview), summing over its per-turn windows."""
    windows = _thread_windows(thread_id)

    def _accept(ts: float) -> bool:
        return any(f <= ts <= t for (f, t) in windows)

    return _aggregate_token_usage(_accept)


# A gap larger than this between consecutive token-log lines starts a new "run"
# (one user turn is a contiguous burst of tool calls; the next turn follows a pause).
_RUN_GAP_SECS = 180.0


def _last_run_from_ts() -> float | None:
    """Epoch-secs start of the most recent run in the token log.

    A "run" is one orchestrator turn. The orchestrator's cumulative accumulator
    resets each turn, so a turn-start is an ``[orchestrator]`` line whose ``total``
    input equals its own ``delta`` input (a fresh accumulator) — the last such line
    starts the last run. Falls back to a time-gap heuristic (consecutive lines
    within ``_RUN_GAP_SECS`` are one run) when the log has no orchestrator lines
    (e.g. a legacy/non-free-agent run). Returns ``None`` on an empty/absent log.
    """
    from src.agent import _load_settings  # local import — matches _parse_token_usage
    rel = (_load_settings() or {}).get("tokens_usage_log", "./.logs/tokens_usage.log")
    path = _project_root() / rel
    if not path.exists():
        return None
    ts_all: list[float] = []
    orch_starts: list[float] = []
    with path.open("r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            head = _RE_TOKENS_HEAD.match(line)
            if not head:
                continue
            try:
                ts = time.mktime(time.strptime(head.group(1), "%Y-%m-%d %H:%M:%S"))
            except (ValueError, OverflowError):
                continue
            ts_all.append(ts)
            if head.group(2).strip() == "orchestrator":
                dm = _RE_TOKENS_DELTA.search(line)
                tm = _RE_TOKENS_TOTAL.search(line)
                if dm and tm and int(tm.group(1)) == int(dm.group(1)):
                    orch_starts.append(ts)
    if orch_starts:
        return orch_starts[-1]
    if not ts_all:
        return None
    ts_all.sort()
    j = len(ts_all) - 1
    while j > 0 and (ts_all[j] - ts_all[j - 1]) <= _RUN_GAP_SECS:
        j -= 1
    return ts_all[j]


def _read_env_file() -> dict:
    """Parse the .env file into {KEY: value} (ignores comments / blank lines)."""
    out: dict[str, str] = {}
    path = _env_path()
    if not path.exists():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s or s.startswith("#") or "=" not in s:
            continue
        key, _, val = s.partition("=")
        out[key.strip()] = val.strip()
    return out


def _update_env_file(updates: dict) -> None:
    """Write KEY=value updates into .env, preserving comments and key order.

    Existing keys are replaced in place; new keys are appended. ``os.environ`` is
    updated too so freshly-built agents pick the values up without a restart."""
    path = _env_path()
    lines = path.read_text(encoding="utf-8").splitlines(keepends=True) if path.exists() else []
    seen: set[str] = set()
    for i, raw in enumerate(lines):
        s = raw.strip()
        if not s or s.startswith("#") or "=" not in s:
            continue
        key = s.split("=", 1)[0].strip()
        if key in updates:
            nl = "\n" if raw.endswith("\n") else ""
            lines[i] = f"{key}={updates[key]}{nl}"
            seen.add(key)
    missing = [k for k in updates if k not in seen]
    if missing:
        if lines and not lines[-1].endswith("\n"):
            lines[-1] = lines[-1] + "\n"
        for k in missing:
            lines.append(f"{k}={updates[k]}\n")
    path.write_text("".join(lines), encoding="utf-8")
    for k, v in updates.items():
        os.environ[str(k)] = str(v)
    # A key that just changed should stop being "old" immediately, not at the next
    # restart — otherwise the one action the warning asks for appears not to work.
    _note_key_ages()


# ── API-key age ──────────────────────────────────────────────────────────────

# What the settings API sends in place of a secret it will not disclose.
#
# A fixed string, carrying nothing of the value — not even the last few
# characters. The field name (HF_TOKEN) already says which credential it is, so a
# partial reveal would buy nothing and cost the one thing masking is for. Its
# other job is to be recognisable on the way back in: the panel only sends fields
# the user edited, but a save that did include an untouched field must not write
# the mask over the key.
_SECRET_MASK = "\u2022" * 8


def _masked_env(env: dict) -> dict:
    """The .env as the settings API is allowed to describe it.

    Secrets become the mask; everything else (endpoints, the Slack allow-list)
    keeps its real value, because those are settings the UI has to round-trip and
    none of them is a credential.
    """
    out: dict[str, str] = {}
    for key, value in (env or {}).items():
        text = str(value or "")
        out[key] = (_SECRET_MASK if (text and _key_age.is_secret_key(key)) else text)
    return out


def _drop_masked(updates: dict) -> dict:
    """Discard any incoming value that is just the mask we sent out.

    Belt and braces: today's panel only submits fields whose value changed, so
    this should never fire. It exists because the failure it prevents is
    unrecoverable and silent — the mask written into .env, the real key gone, and
    nothing to say so until the next API call fails with a 401.
    """
    return {k: v for k, v in (updates or {}).items()
            if not (isinstance(v, str) and v.strip("\u2022 ") == "" and v.strip())}


def _key_age_path() -> Path:
    """Beside settings.local.json: machine state, gitignored, not a secret itself."""
    return _project_root() / "config" / "key_ages.json"


def _max_key_age_days() -> float:
    raw = _security_settings().get("api_key_max_age_days",
                                   _key_age.DEFAULT_MAX_AGE_DAYS)
    try:
        return max(0.0, float(raw))
    except (TypeError, ValueError):
        return float(_key_age.DEFAULT_MAX_AGE_DAYS)


def _note_key_ages() -> list[dict]:
    """Fold the current .env into the age ledger and return the per-key report.

    Called at startup and after every settings save, so the ledger only ever
    describes keys that are really there.
    """
    try:
        env = _read_env_file()
        path = _key_age_path()
        ledger = _key_age.record(env, _key_age.load(path), env_path=_env_path())
        _key_age.save(path, ledger)
        return _key_age.report(env, ledger, _max_key_age_days())
    except Exception as exc:  # noqa: BLE001
        logger.debug("key-age bookkeeping skipped: %s", exc)
        return []


def _diff_leaves(old, new, prefix: tuple = ()) -> list:
    """Yield (path_tuple, value) for every scalar/list leaf in *new* that differs
    from *old* (recursing into dicts). Used to write only the changed leaves into
    settings.local.json, so the local override file stays a minimal delta."""
    changes: list = []
    if isinstance(new, dict):
        base = old if isinstance(old, dict) else {}
        for k, v in new.items():
            changes.extend(_diff_leaves(base.get(k), v, prefix + (k,)))
    else:
        if old != new:
            changes.append((prefix, new))
    return changes


def _update_settings_file(new_settings: dict) -> list:
    """Persist changed leaves from *new_settings* as machine overrides in
    config/settings.local.json (deep-merged over the committed TOML defaults),
    leaving the defaults untouched. Returns the applied dotted paths.

    Only leaves that differ from the current *effective* settings are written, so
    the local override file stays a minimal delta rather than a full copy.
    """
    from src.utils.settings import load_settings, set_local
    current = load_settings()
    updates = _diff_leaves(current, new_settings)
    if not updates:
        return []
    override: dict = {}
    applied: list = []
    for pathv, value in updates:
        node = override
        for k in pathv[:-1]:
            node = node.setdefault(k, {})
        node[pathv[-1]] = value
        applied.append(".".join(str(p) for p in pathv))
    set_local(override)
    # set_local dropped the central loader's cache (the single shared settings
    # object), so agents rebuilt after this POST read the new values.
    return applied


# ── Flask application ─────────────────────────────────────────────────────────

def _sse(obj: dict) -> str:
    """One SSE frame. Never raises — a frame it cannot render is still a frame.

    This runs inside the streaming generator, where an exception does not merely
    drop one event: it tears the generator down, so the `done` sitting behind it
    in the queue is never sent. The panel listens for exactly that, so it stays
    in its streaming state and ignores everything typed afterwards. The turn
    itself finished perfectly, which is why it reads as "the agent went quiet"
    rather than as a crash.

    So anything json cannot render is rendered with `str` instead. Losing the
    exact shape of one event is a far smaller thing than losing the end of the
    turn.
    """
    try:
        body = json.dumps(obj, ensure_ascii=False)
    except (TypeError, ValueError):
        try:
            body = json.dumps(obj, ensure_ascii=False, default=str)
        except Exception:  # noqa: BLE001 — a frame must come out regardless
            body = json.dumps({"type": "error",
                               "message": "an event could not be encoded"})
    return f"data: {body}\n\n"


def _stream_turn(q, rid: str, thread_id: str, poll: float = 15.0):
    """Yield one turn's SSE frames, ending — always — with `done`.

    Lifted out of the route so it can be driven directly. The failure this
    guards against is invisible from outside: the turn completes, the runner
    logs `post:emit_done`, and the panel still never hears about it.
    """
    # Breadcrumbed separately from the runner thread: if the runner logs
    # `post:emit_done` but the panel still hangs, the loss is on the wire
    # (or in the browser); if the runner never gets there, the turn is
    # parked and the keep-alive count below shows how long we waited.
    yield _sse({"type": "thread", "id": thread_id})
    yield _sse({"type": "request", "request_id": rid})
    idle = 0
    sent_done = False

    def _release():
        """The frame that lets the panel out of its streaming state.

        It listens for exactly one thing, so a turn that ends any other way
        leaves it deaf to everything typed afterwards until the tab is reloaded.
        """
        _wd.note(rid, "sse closing without a done — releasing the panel")
        return _sse({"type": "done"})

    try:
        while True:
            try:
                item = q.get(timeout=poll)
            except queue.Empty:
                idle += 1
                # Liveness check, not a timeout: a turn may legitimately be
                # silent for a long time (a video render). But if the runner
                # is no longer tracked, it exited without terminating this
                # queue — keep-aliving on would leave the panel streaming
                # forever, which is the "agent went quiet" failure. Close it
                # so the panel unblocks and can send again.
                if not _wd.is_in_flight(rid):
                    _wd.note(rid, "sse runner gone without done — closing stream")
                    yield _sse({"type": "error", "message":
                                "The turn ended without completing. "
                                "You can send another message."})
                    yield _sse({"type": "done"})
                    sent_done = True
                    break
                # Every ~2min of silence, mark it: a healthy turn is either
                # streaming events or finished, not quiet for minutes.
                if idle % 8 == 0:
                    _wd.note(rid, f"sse idle — {idle * poll:.0f}s with no event, still keep-alive")
                yield ": keep-alive\n\n"  # keep the stream warm / defeat idle buffering
                continue
            if item is None:
                break
            if isinstance(item, dict) and item.get("type") == "done":
                _wd.note(rid, "sse yielding done")
                sent_done = True
            yield _sse(item)
        if not sent_done:
            yield _release()
    except GeneratorExit:
        # The client went away (tab closed, Stop, network). Recorded because
        # a disconnect mid-turn leaves the runner writing into a queue no
        # one drains — worth seeing in the trace next to the runner's phases.
        # Nothing to release: there is no one left to tell.
        _wd.note(rid, "sse client disconnected before done")
        raise
    except Exception as exc:  # noqa: BLE001
        # Something else killed the loop with the turn's `done` still in the
        # queue behind it. Name what broke — that is the one fact this trace
        # could not give before, and it is the difference between "the agent
        # went quiet again" and something to go and fix.
        _wd.note(rid, f"sse stream failed: {type(exc).__name__}: {exc}")
        if not sent_done:
            yield _release()
    finally:
        _wd.note(rid, "sse generator closed")


def _build_app():
    from flask import Flask, jsonify, request, Response, stream_with_context

    app = Flask("agentY_bridge")
    app.logger.disabled = True
    # Keep dict order as authored. The settings form is GENERATED from the settings
    # file, so that file's ordering is the only thing deciding what the user reads
    # first — and Flask alphabetises JSON keys by default, which was quietly
    # reordering it (putting "per-role overrides" above the tiers they inherit
    # from, for instance).
    try:
        app.json.sort_keys = False
    except AttributeError:  # Flask < 2.3
        app.config["JSON_SORT_KEYS"] = False

    # Magnific background auto-drop: when an async creation finishes, the watcher
    # downloads + stages the asset via this handler, then raises a notify_bus
    # event the panel drains (idle poll / live SSE) to drop it onto the canvas.
    try:
        from src.utils import magnific_watch
        magnific_watch.set_completion_handler(_handle_magnific_complete)
    except Exception as exc:  # noqa: BLE001
        logger.warning("magnific_watch wiring skipped: %s", exc)

    # ── Who may call this host ─────────────────────────────────────────────
    # This used to be `Access-Control-Allow-Origin: *` and nothing else, which
    # made every response readable by any page in any tab — including
    # GET /agentY/settings, which returns .env. See src.utils.api_guard for what
    # replaced it and why each check is there.
    @app.before_request
    def _guard():
        ok, why = _guard_verdict(request)
        if ok:
            return None
        _log_refusal(request.method, request.path, why)
        # 403 rather than 401: there is no credential the caller could supply that
        # would make a cross-site request acceptable, and a WWW-Authenticate
        # challenge would invite a browser to prompt for one.
        return jsonify({"ok": False, "error": why}), 403

    @app.after_request
    def _cors(resp):
        # Reflect the one origin we accepted, never "*". A reflected origin is
        # what lets the panel read its own responses while leaving every other
        # page with a CORS error — the browser enforces this on our behalf, which
        # is the only reason it works at all.
        origin = request.headers.get("Origin", "")
        if origin and _guard_verdict(request)[0]:
            resp.headers["Access-Control-Allow-Origin"] = origin
            # Caches sit between us and the panel (and the viewers are plain
            # pages). Without this a response allowed for one origin can be
            # replayed to another out of the browser cache.
            resp.headers["Vary"] = "Origin"
        resp.headers["Access-Control-Allow-Headers"] = f"Content-Type, {_api_guard.TOKEN_HEADER}"
        resp.headers["Access-Control-Allow-Methods"] = "GET, POST, DELETE, OPTIONS"
        # A preflight the browser re-asks for on every request is a visible stall
        # on a panel that polls; ten minutes is short enough that a settings
        # change is not stuck behind it.
        resp.headers["Access-Control-Max-Age"] = "600"
        return resp

    def _sse_response(generator):
        """Wrap an SSE generator in a Response with buffering defeated.

        ``direct_passthrough`` stops Werkzeug from re-buffering the body, and the
        no-cache / no-transform / X-Accel-Buffering headers stop any intermediary
        (or the browser) from coalescing frames — so events reach the panel as
        soon as they're yielded rather than in a batch.

        Because ``direct_passthrough`` bypasses Werkzeug's usual str→bytes
        encoding, the WSGI server asserts every chunk is ``bytes``. Our
        generators yield ``str`` (``_sse`` frames, keep-alive comments), so
        encode each chunk to UTF-8 here — the single boundary every SSE stream
        passes through."""
        def _encoded():
            for chunk in generator:
                yield chunk.encode("utf-8") if isinstance(chunk, str) else chunk
        resp = Response(stream_with_context(_encoded()), mimetype="text/event-stream")
        resp.headers["Cache-Control"] = "no-cache, no-transform"
        resp.headers["X-Accel-Buffering"] = "no"
        resp.headers["Connection"] = "keep-alive"
        resp.direct_passthrough = True
        return resp

    # ── Health / commands ──────────────────────────────────────────────────
    @app.route("/agentY/health", methods=["GET"])
    def health():
        # project_root lets the panel tell the ComfyUI extension where this host
        # lives (browser-mediated self-registration), so the "Start server" button
        # can relaunch it later with no env var or manual config.
        return jsonify({"status": "ok", "pipeline": _agent_ref is not None,
                        "project_root": str(_project_root()),
                        # ...and which script restarts it. The panel forwards this
                        # to the extension, which would otherwise have to guess an
                        # operating system it cannot see.
                        "launcher": _launcher_name(),
                        # "post me the graph on your next tick" — how a turn with
                        # no browser behind it (Slack) gets to see the canvas.
                        "want_canvas": canvas_wanted(),
                        "boot_id": _BOOT_ID, "uptime": round(time.time() - _BOOT_TIME, 1)})

    @app.route("/agentY/canvas", methods=["POST", "OPTIONS"])
    def canvas_snapshot():
        """The panel answering `want_canvas` — the same payload it sends with a
        message, minus the message."""
        if request.method == "OPTIONS":
            return "", 204
        body = request.get_json(silent=True) or {}
        remember_canvas(body.get("canvas_prompt"), body.get("canvas_hooks"),
                        body.get("canvas_selection"))
        return jsonify({"ok": True})

    @app.route("/agentY/commands", methods=["GET"])
    def commands():
        return jsonify(SLASH_COMMANDS)

    @app.route("/agentY/diag", methods=["GET"])
    def diag():
        """In-flight turns and every thread's stack, live.

        For the "panel went quiet after a turn" failure: hit this while it is
        stuck and the answer is in the response — which phase the turn is parked
        in, and the exact frame its thread is blocked on. ``?stacks=0`` returns
        just the phase summary. Read-only.
        """
        want = request.args.get("stacks", "1").lower() not in ("0", "false", "no")
        return jsonify(_wd.snapshot(include_stacks=want))

    # ── In-flight runs ──────────────────────────────────────────────────────
    # The panel has one DOM and one stream, so it can lose track of a turn: a
    # reload drops the SSE connection outright, and switching conversations puts
    # the running one off-screen. Without somewhere to ask, it cannot tell "still
    # working" from "finished while you were away" — the difference between a
    # spinner worth waiting on and one that never clears.
    @app.route("/agentY/runs", methods=["GET"])
    def active_runs():
        with _reply_lock:
            runs = [{"request_id": rid, "thread_id": v.get("thread_id")}
                    for rid, v in _run_registry.items()]
            awaiting = set(_reply_registry)
        for r in runs:
            r["awaiting_reply"] = r["request_id"] in awaiting
        return jsonify({"runs": runs})

    # ── CLI-side status notices (memory init, model pulls, …) ───────────────
    # The panel drains this on connect (so startup lines that predate it still
    # show) and after each turn; live lines during a turn arrive as SSE
    # ``status_line`` events. ``since`` is the highest seq already shown, so this
    # never re-returns a line already delivered live.
    @app.route("/agentY/status", methods=["GET"])
    def status_feed():
        try:
            since = int(request.args.get("since", "0") or 0)
        except (TypeError, ValueError):
            since = 0
        return jsonify(status_bus.snapshot(since))

    # ── Structured background notifications (Magnific auto-drop, …) ─────────
    # The panel polls this on an idle timer (there's no live SSE between turns)
    # and also on connect / after each turn; ``since`` is the highest seq it has
    # already handled so a completion delivered live during a turn isn't
    # re-dropped. See :mod:`src.utils.notify_bus`.
    @app.route("/agentY/notifications", methods=["GET"])
    def notifications_feed():
        try:
            since = int(request.args.get("since", "0") or 0)
        except (TypeError, ValueError):
            since = 0
        snap = notify_bus.snapshot(since)
        # `pending` = background generations still being watched. The panel uses
        # it to stop the idle poll once nothing is in flight (re-arming on the
        # next turn), so an idle tab isn't polling this endpoint forever.
        try:
            from src.utils import magnific_watch
            snap["pending"] = magnific_watch.active_count()
        except Exception:  # noqa: BLE001
            snap["pending"] = 0
        return jsonify(snap)

    # ── Canvas probes: ask the open page something and wait for its answer ──
    # The one place the host asks the PAGE a question. Deliberately not on the
    # SSE stream: that drains inside the orchestrator's event loop, so a tool
    # blocked on an answer would be holding the channel meant to deliver it.
    # ── Tool permission prompts ────────────────────────────────────────────
    # Long-polled on its own connection, for the reason canvas_probe is: the
    # agent thread is BLOCKED inside the tool while this question is outstanding,
    # so nothing that rides on the turn's own stream could deliver it.
    @app.route("/agentY/permission", methods=["GET", "OPTIONS"])
    def permission_take():
        if request.method == "OPTIONS":
            return "", 204
        from src.utils import tool_permissions as tp
        # ?wait=N holds the connection open until there is something to say. The
        # panel used to ask every second or so instead, which cost two requests a
        # second through the access log — the token header makes even a GET a
        # non-simple request, so each poll is a preflight and then the poll.
        try:
            wait = min(30.0, max(0.0, float(request.args.get("wait", 0))))
        except (TypeError, ValueError):
            wait = 0.0
        return jsonify({"request": tp.take(wait), "granted": tp.granted_for_session()})

    @app.route("/agentY/permission/reply", methods=["POST", "OPTIONS"])
    def permission_reply():
        if request.method == "OPTIONS":
            return "", 204
        from src.utils import tool_permissions as tp
        body = request.get_json(silent=True) or {}
        ok = tp.answer(str(body.get("permission_id") or ""),
                       bool(body.get("allowed")),
                       remember=bool(body.get("remember")),
                       note=str(body.get("note") or ""))
        # False means the waiter gave up first (the timeout ran out, or the turn
        # was stopped). Reported rather than treated as an error: the panel should
        # take the prompt down either way.
        return jsonify({"ok": ok, "expired": not ok})

    @app.route("/agentY/canvas_probe", methods=["GET"])
    def canvas_probe_poll():
        """Long-poll: hand the panel the next probe, or nothing after a while.

        Held open rather than answered empty immediately — a screenshot should
        appear when the agent asks for it, not up to one poll interval later.
        Returns promptly when a probe arrives, and always within `wait`.
        """
        from src.utils import canvas_probe
        try:
            wait = float(request.args.get("wait", "25") or 25)
        except (TypeError, ValueError):
            wait = 25.0
        wait = max(0.0, min(wait, 55.0))    # under any sane proxy read timeout
        deadline = time.time() + wait
        while True:
            probe = canvas_probe.take()
            if probe is not None:
                return jsonify({"ok": True, "probe": probe})
            if time.time() >= deadline:
                return jsonify({"ok": True, "probe": None})
            time.sleep(0.15)

    @app.route("/agentY/canvas_probe/reply", methods=["POST", "OPTIONS"])
    def canvas_probe_reply():
        if request.method == "OPTIONS":
            return "", 204
        from src.utils import canvas_probe
        body = request.get_json(silent=True) or {}
        pid = str(body.get("probe_id") or "")
        if not pid:
            return jsonify({"ok": False, "error": "probe_id is required"}), 400
        # False = nobody is waiting any more (the tool timed out first). Not an
        # error: the answer simply arrived too late to be of use.
        delivered = canvas_probe.reply(pid, body.get("data") or {})
        return jsonify({"ok": True, "delivered": bool(delivered)})

    # ── Available models (per vendor) for the quick-switch dropdown ─────────
    @app.route("/agentY/models", methods=["GET"])
    def models():
        return jsonify(_available_models())

    # ── Token-usage overview (input/output per model, filterable by time) ───
    @app.route("/agentY/token_usage", methods=["GET", "OPTIONS"])
    def token_usage():
        if request.method == "OPTIONS":
            return "", 204

        def _num(name):
            v = request.args.get(name)
            if v in (None, "", "null"):
                return None
            try:
                return float(v)
            except (TypeError, ValueError):
                return None

        try:
            scope = request.args.get("scope")
            if scope == "last_run":
                return jsonify(_parse_token_usage(_last_run_from_ts(), None))
            if scope == "thread":
                tid = (request.args.get("thread_id") or request.args.get("thread") or "").strip()
                return jsonify(_parse_token_usage_thread(tid))
            return jsonify(_parse_token_usage(_num("from"), _num("to")))
        except Exception as exc:  # noqa: BLE001
            logger.error("token_usage failed: %s", exc, exc_info=True)
            return jsonify({"ok": False, "error": str(exc)}), 500

    # ── Purge the token-usage log (irreversible) ────────────────────────────
    @app.route("/agentY/token_usage/clear", methods=["POST", "OPTIONS"])
    def token_usage_clear():
        if request.method == "OPTIONS":
            return "", 204
        try:
            from src.agent import _load_settings
            rel = (_load_settings() or {}).get("tokens_usage_log", "./.logs/tokens_usage.log")
            path = _project_root() / rel
            cleared = 0
            if path.exists():
                # Count entries for feedback, then truncate in place. The token
                # hook opens the log per-write in append mode (no long-held
                # handle), so truncating here can't race with a live write.
                try:
                    with path.open("r", encoding="utf-8", errors="replace") as fh:
                        cleared = sum(1 for _ in fh)
                except Exception:  # noqa: BLE001
                    cleared = 0
                path.open("w", encoding="utf-8").close()
            return jsonify({"ok": True, "cleared_lines": cleared})
        except Exception as exc:  # noqa: BLE001
            logger.error("token_usage clear failed: %s", exc, exc_info=True)
            return jsonify({"ok": False, "error": str(exc)}), 500

    # ── Message-history log viewer (self-contained HTML + raw log feed) ─────
    # The viewer page (scripts/log_viewer.html) is served here so it can fetch
    # the log same-origin; opened from the ComfyUI panel via web/agent_log_viewer.js.
    @app.route("/agentY/log_viewer", methods=["GET"])
    def log_viewer():
        page = _project_root() / "scripts" / "log_viewer.html"
        if not page.exists():
            return "log_viewer.html not found", 404
        html = page.read_text(encoding="utf-8", errors="replace")
        # These pages are opened as a navigation, which cannot carry a header, so
        # the token is handed to them in the document itself. The Origin check is
        # what stops another page from fetching this HTML to read it out.
        html = _api_guard.inject_token(html, _api_guard.session_token(_project_root()))
        return Response(html, mimetype="text/html; charset=utf-8")

    @app.route("/agentY/message_history", methods=["GET", "OPTIONS"])
    def message_history():
        if request.method == "OPTIONS":
            return "", 204
        from src.agent import _load_settings
        rel = (_load_settings() or {}).get("message_history_log", "./.logs/message_history.log")
        path = _project_root() / rel
        if not path.exists():
            return Response("", mimetype="text/plain; charset=utf-8", status=404)
        # The log grows without bound (tens of MB); shipping the whole thing makes
        # the in-browser viewer hang on load. Default to the last ~2 MB of history,
        # trimmed to a clean record boundary; ?full=1 returns everything on demand.
        full = str(request.args.get("full", "")).strip().lower() in ("1", "true", "yes")
        try:
            size = path.stat().st_size
        except OSError:
            size = 0
        tail_bytes = 2 * 1024 * 1024
        if full or size <= tail_bytes:
            text = path.read_text(encoding="utf-8", errors="replace")
        else:
            with path.open("rb") as fh:
                fh.seek(size - tail_bytes)
                chunk = fh.read()
            text = chunk.decode("utf-8", errors="replace")
            # Start on a record separator (a line of '=') so the viewer's parser
            # doesn't choke on a half record at the cut point.
            cut = text.find("\n====")
            if cut > 0:
                text = text[cut + 1:]
            text = ("[showing the last ~2 MB of history — append ?full=1 to the URL "
                    "for the complete log]\n\n") + text
        return Response(text, mimetype="text/plain; charset=utf-8")

    @app.route("/agentY/message_history/clear", methods=["POST", "OPTIONS"])
    def message_history_clear():
        if request.method == "OPTIONS":
            return "", 204
        try:
            from src.utils.chat_summary import purge_message_history
            res = purge_message_history()
            return jsonify({"ok": True, **res})
        except Exception as exc:  # noqa: BLE001
            logger.error("message history clear failed: %s", exc, exc_info=True)
            return jsonify({"ok": False, "error": str(exc)}), 500

    # ── Long-term-memory viewer (self-contained HTML + JSON CRUD feed) ──────
    # ── project memory (per-project facts, in ComfyUI's user dir) ─────────────
    # Inspect and delete only. Writing is deliberately not offered: an entry is
    # established by the agent (project_memory_write) or by the `remember` switch
    # on an `agentY add tag` node, and a second way to author one by hand would be
    # a second source of truth for the same file. Removing, though, has to be a
    # human gesture — turning a tag's switch off never deletes, so this is where
    # something stops being true of the project.
    @app.route("/agentY/project_memory_viewer", methods=["GET"])
    def project_memory_viewer():
        page = _project_root() / "scripts" / "project_memory_viewer.html"
        if not page.exists():
            return "project_memory_viewer.html not found", 404
        html = page.read_text(encoding="utf-8", errors="replace")
        # These pages are opened as a navigation, which cannot carry a header, so
        # the token is handed to them in the document itself. The Origin check is
        # what stops another page from fetching this HTML to read it out.
        html = _api_guard.inject_token(html, _api_guard.session_token(_project_root()))
        return Response(html, mimetype="text/html; charset=utf-8")

    @app.route("/agentY/project_memory", methods=["GET", "OPTIONS"])
    def project_memory_list():
        if request.method == "OPTIONS":
            return "", 204
        try:
            from agenty_core.utils.project_memory import list_entries, store_dir
            entries = [
                {"name": e.name, "type": e.type, "summary": e.summary,
                 "body": e.body, "path": str(e.path)}
                for e in list_entries()
            ]
            d = store_dir()
            return jsonify({"ok": True, "count": len(entries), "entries": entries,
                            "store": str(d) if d else ""})
        except Exception as exc:  # noqa: BLE001
            logger.error("project memory list failed: %s", exc, exc_info=True)
            return jsonify({"ok": False, "error": str(exc)}), 500

    @app.route("/agentY/project_memory/delete", methods=["POST", "OPTIONS"])
    def project_memory_delete():
        if request.method == "OPTIONS":
            return "", 204
        body = request.get_json(silent=True) or {}
        names = body.get("names")
        if names is None and body.get("name"):
            names = [body.get("name")]
        names = [str(n).strip() for n in (names or []) if str(n).strip()]
        if not names:
            return jsonify({"ok": False, "error": "names are required"}), 400
        try:
            from agenty_core.utils.project_memory import delete_entry
            gone = [n for n in names if delete_entry(n)]
            missing = [n for n in names if n not in gone]
            return jsonify({"ok": True, "deleted": gone, "not_found": missing})
        except Exception as exc:  # noqa: BLE001
            logger.error("project memory delete failed: %s", exc, exc_info=True)
            return jsonify({"ok": False, "error": str(exc)}), 500

    # The page (scripts/memory_viewer.html) is served here so it fetches the
    # memory feed same-origin; opened from the ComfyUI panel via
    # web/agent_memory_viewer.js. Listing reads the FAISS docstore directly (no
    # Ollama needed); edit/delete/purge go through the mem0 client.
    @app.route("/agentY/memory_viewer", methods=["GET"])
    def memory_viewer():
        page = _project_root() / "scripts" / "memory_viewer.html"
        if not page.exists():
            return "memory_viewer.html not found", 404
        html = page.read_text(encoding="utf-8", errors="replace")
        # These pages are opened as a navigation, which cannot carry a header, so
        # the token is handed to them in the document itself. The Origin check is
        # what stops another page from fetching this HTML to read it out.
        html = _api_guard.inject_token(html, _api_guard.session_token(_project_root()))
        return Response(html, mimetype="text/html; charset=utf-8")

    @app.route("/agentY/memory", methods=["GET", "OPTIONS"])
    def memory_list():
        if request.method == "OPTIONS":
            return "", 204
        try:
            from src.utils.memory import memory_list_raw
            items = memory_list_raw()
            return jsonify({"ok": True, "count": len(items), "memories": items})
        except Exception as exc:  # noqa: BLE001
            logger.error("memory list failed: %s", exc, exc_info=True)
            return jsonify({"ok": False, "error": str(exc)}), 500

    @app.route("/agentY/memory/update", methods=["POST", "OPTIONS"])
    def memory_update_route():
        if request.method == "OPTIONS":
            return "", 204
        body = request.get_json(silent=True) or {}
        mid = str(body.get("id") or "").strip()
        text = body.get("text")
        if not mid or not isinstance(text, str) or not text.strip():
            return jsonify({"ok": False, "error": "id and non-empty text are required"}), 400
        try:
            from src.utils.memory import memory_update
            memory_update(mid, text.strip())
            return jsonify({"ok": True, "id": mid})
        except Exception as exc:  # noqa: BLE001
            logger.error("memory update failed: %s", exc, exc_info=True)
            return jsonify({"ok": False, "error": str(exc)}), 500

    @app.route("/agentY/memory/delete", methods=["POST", "OPTIONS"])
    def memory_delete_route():
        if request.method == "OPTIONS":
            return "", 204
        body = request.get_json(silent=True) or {}
        ids = body.get("ids")
        if ids is None and body.get("id"):
            ids = [body.get("id")]
        ids = [str(i) for i in (ids or []) if str(i).strip()]
        if not ids:
            return jsonify({"ok": False, "error": "no memory ids given"}), 400
        try:
            from src.utils.memory import memory_delete_ids
            res = memory_delete_ids(ids)
            return jsonify({"ok": True, **res})
        except Exception as exc:  # noqa: BLE001
            logger.error("memory delete failed: %s", exc, exc_info=True)
            return jsonify({"ok": False, "error": str(exc)}), 500

    @app.route("/agentY/memory/clear", methods=["POST", "OPTIONS"])
    def memory_clear_route():
        if request.method == "OPTIONS":
            return "", 204
        try:
            from src.utils.memory import memory_purge
            res = memory_purge()
            return jsonify({"ok": True, **res})
        except Exception as exc:  # noqa: BLE001
            logger.error("memory clear failed: %s", exc, exc_info=True)
            return jsonify({"ok": False, "error": str(exc)}), 500

    # ── Application settings (.env auth keys + config/settings.json) ────────
    @app.route("/agentY/settings", methods=["GET", "POST", "OPTIONS"])
    def settings_route():
        if request.method == "OPTIONS":
            return "", 204
        if request.method == "GET":
            import copy
            from src.agent import _load_settings
            # Deep-copy: _load_settings() returns the shared cached merge, and we
            # mutate `settings` below (live user dir) — must not poison the cache.
            settings = copy.deepcopy(_load_settings())
            env = {k: "" for k in _KNOWN_ENV_KEYS}
            env.update(_read_env_file())
            # Show the DashScope endpoint (beneath its API key) even when it's not
            # overridden in .env — seed it from the merged settings so it's never blank.
            if not env.get("DASHSCOPE_BASE_URL"):
                ds_url = ((settings.get("llm") or {}).get("dashscope") or {}).get("base_url", "")
                if ds_url:
                    env["DASHSCOPE_BASE_URL"] = ds_url
            # Show ComfyUI's live --user-directory rather than the static fallback.
            live_user_dir = _effective_comfyui_user_dir()
            if live_user_dir:
                settings["comfyui_user_dir"] = live_user_dir
            # Friendly names for the model tiers, so the mapping lives once in
            # src/agent.py instead of being duplicated in the settings JS.
            try:
                from src.agent import TIER_LABELS as _tier_labels
            except Exception:  # noqa: BLE001
                _tier_labels = {}
            return jsonify({
                # Masked, never the real values. This response used to carry every
                # API key in plaintext to anyone who asked, which — with the old
                # `Allow-Origin: *` — meant any website the user had open.
                "env": _masked_env(env),
                "env_keys": list(dict.fromkeys(_KNOWN_ENV_KEYS + list(env.keys()))),
                "env_mask": _SECRET_MASK,
                # How long each key has been in place, so the panel can show the
                # same rotation warning the host prints at startup.
                "key_ages": _note_key_ages(),
                "key_age_limit": _max_key_age_days(),
                "settings": settings,
                "tier_labels": _tier_labels,
                "model_groups": _available_models(),
                "pricing": _load_pricing_config(),
            })
        # POST — persist env and/or settings changes (settings → settings.local.json).
        body = request.get_json(silent=True) or {}
        result: dict = {"ok": True}
        try:
            env_updates = _drop_masked(body.get("env"))
            if isinstance(env_updates, dict) and env_updates:
                _update_env_file({str(k): "" if v is None else str(v)
                                  for k, v in env_updates.items()})
                result["env_updated"] = sorted(env_updates.keys())
            settings_updates = body.get("settings")
            if isinstance(settings_updates, dict) and settings_updates:
                result["settings_updated"] = _update_settings_file(settings_updates)
            pricing_updates = body.get("pricing")
            if isinstance(pricing_updates, dict):
                _save_pricing_config(pricing_updates)
                result["pricing_updated"] = True
        except Exception as exc:  # noqa: BLE001
            logger.error("settings save failed: %s", exc, exc_info=True)
            return jsonify({"ok": False, "error": str(exc)}), 500
        return jsonify(result)

    # ── Auto-graph toggle (autoload_workflows_into_canvas) ──────────────────
    # Backs the side-panel top-bar toggle. GET reports the current state; POST
    # flips it. Persisted via _update_settings_file → settings.local.json, whose
    # save drops the settings cache, so the executor's next run reads the new
    # value with no restart. The AGENTY_CANVAS_AUTOLOAD env var, when set, wins in
    # the executor — reported as env_locked so the UI can show the toggle as fixed.
    @app.route("/agentY/autograph", methods=["GET", "POST", "OPTIONS"])
    def autograph_route():
        if request.method == "OPTIONS":
            return "", 204
        from src.utils.settings import load_settings
        env_locked = os.environ.get("AGENTY_CANVAS_AUTOLOAD") is not None
        if request.method == "GET":
            try:
                enabled = bool(load_settings().get("autoload_workflows_into_canvas", False))
                return jsonify({"ok": True, "enabled": enabled, "env_locked": env_locked})
            except Exception as exc:  # noqa: BLE001
                return jsonify({"ok": False, "error": str(exc)}), 500
        body = request.get_json(silent=True) or {}
        enabled = bool(body.get("enabled"))
        try:
            _update_settings_file({"autoload_workflows_into_canvas": enabled})
            return jsonify({"ok": True, "enabled": enabled, "env_locked": env_locked})
        except Exception as exc:  # noqa: BLE001
            logger.error("autograph toggle failed: %s", exc, exc_info=True)
            return jsonify({"ok": False, "error": str(exc)}), 500

    # ── MCP servers (config/mcp.json + one-time OAuth authorize) ────────────
    @app.route("/agentY/mcp", methods=["GET", "POST", "OPTIONS"])
    def mcp_config_route():
        if request.method == "OPTIONS":
            return "", 204
        from src.tools.mcp_tools import load_mcp_config, save_mcp_config, mcp_status
        if request.method == "GET":
            try:
                return jsonify({"ok": True, "config": load_mcp_config(), "status": mcp_status()})
            except Exception as exc:  # noqa: BLE001
                return jsonify({"ok": False, "error": str(exc)}), 500
        body = request.get_json(silent=True) or {}
        cfg = body.get("config") if isinstance(body.get("config"), dict) else body
        try:
            save_mcp_config(cfg)
            return jsonify({"ok": True})
        except Exception as exc:  # noqa: BLE001
            logger.error("mcp config save failed: %s", exc, exc_info=True)
            return jsonify({"ok": False, "error": str(exc)}), 500

    @app.route("/agentY/mcp/authorize", methods=["POST", "OPTIONS"])
    def mcp_authorize_route():
        # Runs the interactive OAuth flow (opens a browser, waits for the redirect);
        # a deliberate one-time action, so blocking this request is fine.
        if request.method == "OPTIONS":
            return "", 204
        body = request.get_json(silent=True) or {}
        name = str(body.get("name") or "").strip()
        if not name:
            return jsonify({"ok": False, "error": "missing server name"}), 400
        from src.tools.mcp_tools import authorize_server
        try:
            res = authorize_server(name)
            return jsonify(res), (200 if res.get("ok") else 400)
        except Exception as exc:  # noqa: BLE001
            logger.error("mcp authorize failed: %s", exc, exc_info=True)
            return jsonify({"ok": False, "error": str(exc)}), 500

    # ── Threads ────────────────────────────────────────────────────────────
    @app.route("/agentY/threads", methods=["GET", "POST", "OPTIONS"])
    def threads():
        if request.method == "OPTIONS":
            return "", 204
        if request.method == "POST":
            body = request.get_json(silent=True) or {}
            tid = cs.create_thread(title=body.get("title") or "New chat")
            return jsonify({"id": tid})
        return jsonify(cs.list_threads())

    @app.route("/agentY/threads/<tid>", methods=["GET", "DELETE", "OPTIONS"])
    def thread_detail(tid):
        if request.method == "OPTIONS":
            return "", 204
        if request.method == "DELETE":
            cs.delete_thread(tid)
            _thread_brain_cache.pop(tid, None)
            return jsonify({"ok": True})
        t = cs.get_thread(tid)
        if t is None:
            return jsonify({"error": "not found"}), 404
        # The rendered panel HTML (collapsible think/step blocks) restores the
        # exact UI on reopen; the message list is the text-only fallback.
        t["panel_html"] = cs.get_panel(tid)
        return jsonify(t)

    @app.route("/agentY/threads/<tid>/panel", methods=["POST", "OPTIONS"])
    def thread_panel(tid):
        if request.method == "OPTIONS":
            return "", 204
        body = request.get_json(silent=True) or {}
        cs.save_panel(tid, body.get("html", ""))
        return jsonify({"ok": True})

    @app.route("/agentY/threads/clear", methods=["POST", "OPTIONS"])
    def threads_clear():
        if request.method == "OPTIONS":
            return "", 204
        body = request.get_json(silent=True) or {}
        n = cs.delete_all_threads(except_id=body.get("current"))
        return jsonify({"deleted": n})

    # ── Attachment upload ──────────────────────────────────────────────────
    @app.route("/agentY/upload", methods=["POST", "OPTIONS"])
    def upload():
        if request.method == "OPTIONS":
            return "", 204
        f = request.files.get("file")
        if f is None:
            return jsonify({"error": "no file"}), 400
        dest_dir = _project_root() / "output_images" / ".uploads"
        dest_dir.mkdir(parents=True, exist_ok=True)
        name = f"{uuid.uuid4().hex[:8]}_{Path(f.filename or 'upload').name}"
        dest = dest_dir / name
        f.save(str(dest))
        return jsonify({"path": str(dest), "name": name})

    # ── Interactive reply ──────────────────────────────────────────────────
    @app.route("/agentY/reply", methods=["POST", "OPTIONS"])
    def reply():
        if request.method == "OPTIONS":
            return "", 204
        body = request.get_json(silent=True) or {}
        req_id = body.get("request_id")
        text = body.get("text", "")
        with _reply_lock:
            entry = _reply_registry.get(req_id)
        if not entry:
            return jsonify({"ok": False, "error": "no pending request"}), 404
        loop, q = entry
        loop.call_soon_threadsafe(q.put_nowait, text)
        return jsonify({"ok": True})

    # ── Mid-run interjection ───────────────────────────────────────────────
    @app.route("/agentY/interject", methods=["POST", "OPTIONS"])
    def interject():
        """Hand a message to the turn that is *currently running*.

        Distinct from /agentY/reply, which answers a question the agent asked and
        feeds the QA queue — routing an interjection through that queue would let
        it be swallowed as the answer to a pending "retry?" prompt. This goes to
        its own mailbox, and the orchestrator's hook picks it up at the next tool
        boundary. ``urgent`` cancels the pending tool call so the agent reads the
        message instead of taking that step.

        ok=false means there was nothing to interject into (the turn ended, or the
        request id is stale) — the caller should queue the text as a normal message.
        """
        if request.method == "OPTIONS":
            return "", 204
        body = request.get_json(silent=True) or {}
        req_id = str(body.get("request_id") or "")
        text = (body.get("text") or "").strip()
        urgent = bool(body.get("urgent"))
        if not text:
            return jsonify({"ok": False, "error": "empty message"}), 400
        if not interject_bus.post(req_id, text, urgent=urgent):
            return jsonify({"ok": False, "error": "no running turn to interject into"}), 409
        # Not persisted here: the delivering hook writes it into the thread at the
        # moment the model actually reads it, so the stored conversation keeps that
        # order and a message that misses the turn isn't stored twice.
        return jsonify({"ok": True, "urgent": urgent, "pending": interject_bus.pending_count()})

    # ── What a model switch may target (drives the composer's scope picker) ──
    @app.route("/agentY/switch_targets", methods=["GET", "OPTIONS"])
    def switch_targets_route():
        """Tiers + roles the picker offers, so the panel never drifts from the
        settings UI — both read the same tier map out of src/agent.py."""
        if request.method == "OPTIONS":
            return "", 204
        try:
            from src.agent import TIER_LABELS, _ROLE_TIERS
            tiers, roles = _switch_targets()
        except Exception as exc:  # noqa: BLE001
            logger.warning("switch_targets failed: %s", exc)
            return jsonify({"tiers": [], "roles": []})
        return jsonify({
            "tiers": [{"value": t, "label": TIER_LABELS.get(t, t)} for t in tiers],
            "roles": [{"value": r, "label": r, "tier": _ROLE_TIERS.get(r, "")} for r in roles],
        })

    # ── Switch an agent's model (same as the /switch_model command) ─────────
    @app.route("/agentY/switch_model", methods=["POST", "OPTIONS"])
    def switch_model_route():
        if request.method == "OPTIONS":
            return "", 204
        body = request.get_json(silent=True) or {}
        target = (body.get("target") or "all").strip()
        spec = (body.get("spec") or "").strip()
        if not spec:
            return jsonify({"ok": False, "error": "no model spec"}), 400
        try:
            result = _switch_model([target, spec])
        except Exception as exc:  # noqa: BLE001
            logger.error("switch_model failed: %s", exc, exc_info=True)
            return jsonify({"ok": False, "error": str(exc)}), 500
        messages = [r.get("data", "") for r in result if isinstance(r, dict)]
        ok = not any("❌" in m or "⚠️" in m for m in messages)
        return jsonify({"ok": ok, "messages": messages})

    # ── Stop the current run ───────────────────────────────────────────────
    @app.route("/agentY/stop", methods=["POST", "OPTIONS"])
    def stop_run():
        if request.method == "OPTIONS":
            return "", 204
        body = request.get_json(silent=True) or {}
        req_id = body.get("request_id")
        thread_id = body.get("thread_id")
        found = _cancel_run(req_id) if req_id else False
        # Fallback: if the request_id was unknown (e.g. Stop pressed before it
        # reached the client), cancel by thread.
        if not found and thread_id:
            found = _cancel_run_by_thread(thread_id)
        # Unconditionally, and after the cancel: stopping the agent's loop does
        # nothing about the prompts it has ALREADY put in ComfyUI's queue, and
        # those are most of what "stop" means to somebody watching a batch run.
        report = _interrupt_comfy()
        return jsonify({"ok": True, "cancelled": found,
                        "queue_removed": len(report.get("deleted") or []),
                        "queue_kept": report.get("kept", 0)})

    # ── Chat (SSE) ─────────────────────────────────────────────────────────
    @app.route("/agentY/chat", methods=["POST", "OPTIONS"])
    def chat():
        if request.method == "OPTIONS":
            return "", 204
        body = request.get_json(silent=True) or {}
        message = (body.get("message") or "").strip()
        image_paths = [p for p in (body.get("image_paths") or []) if isinstance(p, str)]
        # Load Image / Load Video nodes selected on the ComfyUI canvas become
        # inputs too, in selection order — same as chat attachments. Each entry is
        # {value, kind}; resolve its widget value to an absolute path on disk.
        canvas_paths: list[str] = []
        for ci in (body.get("canvas_inputs") or []):
            if not isinstance(ci, dict):
                continue
            resolved = _resolve_media_ref(ci.get("value", ""), ci.get("kind", ""))
            if resolved:
                canvas_paths.append(resolved)
            else:
                logger.warning("Unresolved canvas input: %r", ci)
        # NOTE: canvas_paths (the CURRENT selection's Load-Image/Video nodes) are
        # merged into image_paths further down — but ONLY on the first turn or when
        # the message references the selection (see the gate after thread_id is
        # resolved). This stops a selection that drifts over the conversation from
        # silently rebinding or dropping the image input(s). Chat attachments
        # (body.image_paths) are always kept — they're explicit per-message uploads.
        # Canvas-hook mode: the captured API-format prompt of the user's on-canvas
        # graph + the hook directives attached to it. Present only when the graph
        # has AgentYHook nodes; drives the "run my canvas graph" execution path.
        canvas_prompt = body.get("canvas_prompt")
        if not isinstance(canvas_prompt, dict):
            canvas_prompt = None
        canvas_hooks = [h for h in (body.get("canvas_hooks") or []) if isinstance(h, dict)]
        # Arbitrary selected nodes (any type) with their widget values, so the
        # agent can read/alter their parameters and write the change back live.
        canvas_selection = [n for n in (body.get("canvas_selection") or []) if isinstance(n, dict)]
        # ComfyUI's open workflow tabs. The graph above is the ACTIVE one and
        # cannot say whether it was one of several — so without this, "the other
        # workflow" gets answered about this one, confidently and wrongly.
        open_workflows = [w for w in (body.get("open_workflows") or []) if isinstance(w, dict)]
        # Dry run: the panel's "Run agentY hooks ▾ → Dry run". Build every graph,
        # submit none of them (src/utils/dry_run.py).
        dry_run = bool(body.get("dry_run"))
        # agentY image-collector nodes wired as hook anchors carry on-disk image
        # paths (widget data) — surface them to the agent as vision with NO
        # pre-run. ONLY when the orchestrator is a vision model: a text-only
        # orchestrator (e.g. dashscope,qwen3.6-flash) rejects image content, so we
        # skip embedding there and rely on the path list in the [CANVAS HOOKS]
        # block (which every orchestrator receives). Cap how many are embedded to
        # bound token cost.
        if canvas_hooks and _orchestrator_supports_vision():
            _coll_imgs = _collector_hook_images(canvas_hooks)
            if _coll_imgs:
                _existing = set(image_paths)
                _add = [p for p in _coll_imgs[:_MAX_HOOK_IMAGES] if p not in _existing]
                if _add:
                    image_paths = _add + image_paths
                if len(_coll_imgs) > _MAX_HOOK_IMAGES:
                    logger.info(
                        "collector hook: embedding %d of %d images as vision "
                        "(all paths listed in the [CANVAS HOOKS] block)",
                        _MAX_HOOK_IMAGES, len(_coll_imgs))
        elif canvas_hooks and _collector_hook_images(canvas_hooks):
            logger.info(
                "collector hook: orchestrator is text-only — not embedding images "
                "as vision; paths are listed in the [CANVAS HOOKS] block.")
        thread_id = body.get("thread_id")
        if not thread_id or cs.get_thread(thread_id) is None:
            thread_id = cs.create_thread(thread_id=thread_id)

        # Gate the canvas selection as inputs: use the CURRENT selection only on
        # the first turn of the thread, or when the message explicitly references
        # the selection ("the selected images", "these nodes", "from the canvas").
        # Otherwise a mid-conversation selection change would silently rebind the
        # inputs — which is not what the user intends.
        if canvas_paths:
            _existing_msgs = (cs.get_thread(thread_id) or {}).get("messages", [])
            _first_turn = not any(m.get("role") == "assistant" for m in _existing_msgs)
            if _first_turn or _message_wants_canvas_inputs(message):
                image_paths = canvas_paths + image_paths
            else:
                logger.info(
                    "Canvas has %d selected input(s) but the message didn't reference "
                    "the selection (and it's not the first turn) — ignoring them as inputs.",
                    len(canvas_paths),
                )

        # The panel just told us what is on its canvas. Keep it: a Slack turn a
        # moment later has no browser of its own to ask, and this one is free.
        remember_canvas(canvas_prompt, canvas_hooks, canvas_selection)

        # Persist the user's message (raw text).
        if message:
            cs.add_message(thread_id, "user", message)

        # Slash command? Handle synchronously, stream the result lines.
        is_slash = _is_command(message, thread_id)
        if is_slash:
            result = _handle_command(thread_id, message, canvas_prompt=canvas_prompt)
            if result is None:  # /resend → replay first user message as a fresh turn
                t = cs.get_thread(thread_id) or {}
                first = next((m for m in t.get("messages", []) if m["role"] == "user"
                              and not m["content"].strip().lower().startswith("/resend")), None)
                if first is None:
                    result = [_sys("❌ Nothing to resend — no earlier user message in this thread.")]
                else:
                    resend_text = first["content"]
                    def gen_resend():
                        yield _sse({"type": "thread", "id": thread_id})
                        yield _sse(_sys(f"🔁 Resending: {resend_text}"))
                        q: queue.Queue = queue.Queue()
                        rid = uuid.uuid4().hex
                        yield _sse({"type": "request", "request_id": rid})
                        threading.Thread(target=_run_pipeline_stream,
                                         args=(thread_id, resend_text, [], q, rid), daemon=True).start()
                        while True:
                            try:
                                item = q.get(timeout=15)
                            except queue.Empty:
                                yield ": keep-alive\n\n"
                                continue
                            if item is None:
                                break
                            yield _sse(item)
                    return _sse_response(gen_resend())

            def gen_cmd():
                yield _sse({"type": "thread", "id": thread_id})
                for ev in result:
                    yield _sse(ev)
                yield _sse({"type": "done"})
            return _sse_response(gen_cmd())

        # Normal turn → run the pipeline and stream SSE.
        q: queue.Queue = queue.Queue()
        rid = uuid.uuid4().hex
        threading.Thread(target=_run_pipeline_stream,
                         args=(thread_id, message, image_paths, q, rid),
                         kwargs={"canvas_prompt": canvas_prompt, "canvas_hooks": canvas_hooks,
                                 "canvas_selection": canvas_selection,
                                 "open_workflows": open_workflows, "dry_run": dry_run},
                         daemon=True).start()

        return _sse_response(_stream_turn(q, rid, thread_id))

    # ── Legacy bridge endpoints ────────────────────────────────────────────
    @app.route("/agentY/pending_previews", methods=["GET", "OPTIONS"])
    def pending_previews():
        if request.method == "OPTIONS":
            return "", 204
        with _lock:
            jobs = list(_pending_previews.values())
            for j in [x["job_id"] for x in jobs if x.get("clear")]:
                _pending_previews.pop(j, None)
        return jsonify(jobs)

    @app.route("/agentY/review", methods=["POST", "OPTIONS"])
    def review():
        if request.method == "OPTIONS":
            return "", 204
        payload = request.get_json(silent=True) or {}
        node_id = payload.get("node_id", "?")
        _dispatch_to_agent(payload.get("message", ""), payload.get("image_paths", []),
                           str(node_id) if node_id not in ("?", None) else None)
        return jsonify({"status": "dispatched"})

    @app.route("/agentY/node_responses", methods=["GET", "OPTIONS"])
    def node_responses():
        if request.method == "OPTIONS":
            return "", 204
        with _lock:
            data = dict(_node_responses)
            _node_responses.clear()
        return jsonify(data)

    return app


# ── Server startup ─────────────────────────────────────────────────────────────

_server_thread: threading.Thread | None = None

# The sidebar polls a few endpoints on a timer — /agentY/health every 5s whenever
# the panel is open (that heartbeat is what notices a dead host), plus the
# notification/status drains while a generation is in flight. Werkzeug logs one
# access line per request, so an idle host scrolls its own console away and any
# real message drowns in "GET /agentY/health 200". The polling still happens; only
# its *successful* log lines are dropped. A poll that 404s or 500s still prints —
# that is exactly the thing worth seeing. Set AGENTY_LOG_POLLS=1 to log them all.
_QUIET_POLL_PATHS = ("/agentY/health", "/agentY/notifications", "/agentY/status",
                     # Long polls. They are quiet per minute rather than per
                     # second, but each one is now TWO lines — the token header
                     # makes them non-simple requests, so the browser sends a CORS
                     # preflight first and OPTIONS is logged like anything else.
                     "/agentY/permission", "/agentY/canvas_probe")
_ACCESS_LINE_RE = re.compile(r'"[A-Z]+ (?P<path>[^" ]+) HTTP/[\d.]+" (?P<code>\d{3})')
# Werkzeug wraps the request line in ANSI colour for anything that isn't a plain
# 200, which would otherwise stop the pattern above from matching.
_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


class _QuietPollFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:  # True = keep the line
        try:
            m = _ACCESS_LINE_RE.search(_ANSI_RE.sub("", record.getMessage()))
        except Exception:  # noqa: BLE001 — a log filter must never break logging
            return True
        if not m or not m.group("code").startswith(("2", "3")):
            return True
        return not m.group("path").split("?", 1)[0].endswith(_QUIET_POLL_PATHS)


def _quiet_poll_logging() -> None:
    if os.environ.get("AGENTY_LOG_POLLS", "").strip().lower() in ("1", "true", "yes"):
        return
    wz = logging.getLogger("werkzeug")
    if not any(isinstance(f, _QuietPollFilter) for f in wz.filters):
        wz.addFilter(_QuietPollFilter())


def _launcher_name(platform: str = "") -> str:
    """The script that starts this host on *platform* (default: this one).

    Named by the HOST's platform, not the extension's: the button runs the script
    on the machine the host lives on, and that is the machine running this code.
    """
    return "run_agent.ps1" if (platform or sys.platform) == "win32" else "run_agent.sh"


def _register_with_comfyui(port: int = 0) -> None:
    """Best-effort: tell the ComfyUI sidebar extension where this host lives, so its
    "Start server" button can relaunch the launcher when the host is down, and so
    the panel knows which port to call. Fire-and-forget; never blocks or fails
    startup.

    The launcher is named by THIS process's platform, not the extension's, which is
    the correct way round: the button runs the script on the machine the host lives
    on, and that is the machine running this code.

    *port* is the port actually bound, which is why it is passed in rather than
    re-read from settings. It is the only value that is true no matter how it was
    chosen - a settings file, AGENTY_UI_PORT, or `--port` on the launcher, which no
    file records at all. The panel used to assume 5000 and had no way to learn
    otherwise; on a Mac, where AirPlay holds 5000 and answers, that assumption
    reports a healthy host as down. Sent as a port rather than a whole URL so the
    panel keeps building the address from its own location.hostname and a sidebar
    reaching ComfyUI across the network still works.
    """
    def _do() -> None:
        try:
            from src.utils.settings import load_settings
            import urllib.request
            base = str(load_settings().get("comfyui_url", "http://127.0.0.1:8188")).rstrip("/")
            payload = {"project_root": str(_project_root()),
                       "run_script": _launcher_name()}
            if port:
                payload["agent_server_port"] = int(port)
            body = json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(base + "/agent/register_host", data=body,
                                         headers={"Content-Type": "application/json"}, method="POST")
            urllib.request.urlopen(req, timeout=5).read()
            logger.info("registered agentY host location with the ComfyUI extension")
        except Exception as exc:  # noqa: BLE001
            logger.debug("host self-registration skipped: %s", exc)
    threading.Thread(target=_do, name="agentY-register-host", daemon=True).start()


def start_agentY_server(agent, host: str = "127.0.0.1", port: int | None = None) -> bool:
    """Start the agentY bridge + chat host in a background daemon thread.

    *port* None means this platform's default - 5000, or 5001 on macOS, where
    AirPlay holds 5000. Resolved here rather than in the signature so the answer
    is this machine's, not whichever machine imported the module.
    """
    global _server_thread, _agent_ref
    if port is None:
        from src.utils.settings import default_agent_port
        port = default_agent_port()
    _agent_ref = agent
    cs.init_db()

    if _server_thread is not None and _server_thread.is_alive():
        return True
    try:
        from flask import Flask  # noqa: F401
    except ImportError:
        logger.error("Flask is not installed. Run: pip install flask")
        return False

    # Before the app: the guard reads it on the very first request, and the token
    # file must exist before the panel asks ComfyUI for it.
    set_bound_port(port)
    _api_guard.session_token(_project_root())
    _note_key_ages()
    _configure_shell_sandbox()
    # A restart is a new session: "allow for this session" must not survive the
    # process whose session it was.
    try:
        from src.utils.tool_permissions import reset_session
        reset_session()
    except Exception:  # noqa: BLE001
        pass

    app = _build_app()
    _quiet_poll_logging()

    def _run():
        try:
            from werkzeug.serving import make_server
            srv = make_server(host, port, app, threaded=True)
            logger.info("agentY chat host ready on http://%s:%d", host, port)
            srv.serve_forever()
        except Exception as exc:
            logger.error("agentY server crashed: %s", exc, exc_info=True)

    _server_thread = threading.Thread(target=_run, name="agentY-bridge-server", daemon=True)
    _server_thread.start()
    logger.info("agentY chat host started on http://%s:%d", host, port)
    # The port too: it is what the panel calls, and only this process knows it.
    _register_with_comfyui(port)
    _start_slack_bridge()
    return True


# ── Slack: a second line in, and a second place to watch ──────────────────────

def _slack_start_turn(text: str, image_paths: list, thread_id: str = "") -> str:
    """Run a turn asked for from Slack, in the conversation it belongs to.

    *thread_id* comes from the Slack thread the message was posted in. Empty
    means a message at the top level of the DM, which starts a NEW conversation
    — the same gesture as opening a new chat in the panel, and the reason a
    thread is worth having: without one, every message would land in whatever
    chat happened to be current.

    Its events reach the panel too; the turn bus does not care who asked.
    """
    thread_id = str(thread_id or "")
    if not thread_id or cs.get_thread(thread_id) is None:
        thread_id = cs.create_thread(title="New chat")
    if text:
        cs.add_message(thread_id, "user", text)
    # Ask the browser what is on the canvas. Without this a Slack turn has no
    # graph at all and every canvas tool answers "no on-canvas graph is loaded",
    # which reads as the agent refusing to look at a workflow that is open in
    # front of you.
    snap = request_canvas()
    q: queue.Queue = queue.Queue()
    rid = uuid.uuid4().hex
    threading.Thread(target=_run_pipeline_stream,
                     args=(thread_id, text, list(image_paths or []), q, rid),
                     kwargs={"origin": "slack",
                             "canvas_prompt": snap.get("prompt"),
                             "canvas_hooks": snap.get("hooks") or [],
                             "canvas_selection": snap.get("selection") or []},
                     name="agentY-slack-turn", daemon=True).start()
    # Nothing reads this queue: Slack is fed by the bus, not by the stream. Drain
    # it anyway, or the turn blocks on a queue that fills and never empties.
    threading.Thread(target=_drain_queue, args=(q,), daemon=True).start()
    return rid


def _drain_queue(q: "queue.Queue") -> None:
    while True:
        try:
            if q.get(timeout=3600) is None:
                return
        except queue.Empty:
            return


def _slack_answer(request_id: str, text: str) -> bool:
    """Feed a Slack reply to an agent question that is holding a turn open."""
    with _reply_lock:
        entry = _reply_registry.get(request_id)
    if not entry:
        return False
    loop, q = entry
    loop.call_soon_threadsafe(q.put_nowait, text)
    return True


def _slack_interject(request_id: str, text: str) -> bool:
    """Hand a Slack message to the turn already running (see /agentY/interject)."""
    return bool(interject_bus.post(request_id, text, urgent=False))


def _start_slack_bridge() -> None:
    try:
        from src.utils import slack_bridge
    except Exception:  # noqa: BLE001
        return
    if not slack_bridge.enabled():
        return
    try:
        ok = slack_bridge.start(start_turn=_slack_start_turn, answer=_slack_answer,
                                interject=_slack_interject,
                                downloads_dir=_project_root() / "output" / "slack_uploads")
        if ok:
            status_bus.notify("💬 Slack bridge connected — turns mirror to your DM.")
    except Exception:  # noqa: BLE001 — Slack must never stop the host from starting
        logger.exception("slack: bridge failed to start")
