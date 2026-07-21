"""
agentY bridge + chat host — runs on localhost:5000.

This is the backend for the **ComfyUI-native agentY chat UI** (the sidebar
custom-node panel in the separate ``agentY-comfyuiConnect`` repo). It replaces the
Chainlit GUI: the pipeline runs here, conversations persist to a local SQLite
store (:mod:`src.utils.conversation_store`), and — crucially — generated media is
**not** streamed back as inline images. Instead the executor's output files are
staged into ComfyUI's input directory and announced to the frontend, which drops
a ``LoadImage`` / video-loader node onto the open ComfyUI graph. Only the agent's
*text* flows into the chat.

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

Viewers (self-contained HTML pages served here so they fetch same-origin)
    GET  /agentY/log_viewer                     message-history log viewer page
    GET  /agentY/message_history                raw message-history log feed
    POST /agentY/message_history/clear          purge the entire history log
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
from src.utils.models import AgentSession

logger = logging.getLogger("agentY.server")

# ── In-memory state ───────────────────────────────────────────────────────────

_lock = threading.Lock()
_pending_previews: dict[str, dict] = {}
_node_responses: dict[str, str] = {}       # node_id (str) -> accumulated agent text
_agent_ref = None                          # the pipeline singleton

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


# ── Slash commands (mirrors the frontend popup list) ──────────────────────────

SLASH_COMMANDS = [
    {"name": "/restart",         "description": "Restart the agent pipeline"},
    {"name": "/stop",            "description": "Stop and shut down the agent"},
    {"name": "/unload",          "description": "Unload Ollama models from VRAM"},
    {"name": "/clear_vram",      "description": "Clear ComfyUI GPU VRAM"},
    {"name": "/images",          "description": "List images generated in this thread (reference them by number)"},
    {"name": "/clearhistory",    "description": "Delete all conversation history (keeps the current thread)"},
    {"name": "/switch_model",    "description": "Switch an agent's LLM — /switch_model <agent|all> <provider,model> (use 'all' for every agent)"},
    {"name": "/add_workflow",    "description": "Add a ComfyUI workflow — /add_workflow <path/to/workflow.json> OR /add_workflow canvas <name> for the graph open in the canvas"},
    {"name": "/resend",          "description": "Resend the first user message of the current thread"},
    {"name": "/remove_workflow", "description": "Remove a workflow by name — /remove_workflow <template_name>"},
]


# ── Media helpers ─────────────────────────────────────────────────────────────

_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp"}
_VIDEO_SUFFIXES = {".mp4", ".mov", ".avi", ".mkv", ".webm"}

# Node classes the frontend will try (first one registered in ComfyUI wins) when
# it drops a loader onto the graph for a generated output.
_NODE_CANDIDATES = {
    "image": ["LoadImage"],
    "video": ["VHS_LoadVideo", "LoadVideo", "VHS_LoadVideoPath"],
}


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


# ── Content builder (text + attached images/videos -> Strands content blocks) ─

def _build_content(message: str, media_paths: list[str]) -> list | str:
    """Build a Strands-compatible content list from text + input media paths.

    Image paths are embedded as vision blocks (downsized to satisfy Claude's
    5 MB / 1568 px constraints) AND listed as file paths. Video paths are not
    embedded (they can't be sent inline) but ARE listed as file paths so the
    agent can wire them into a loader node — same effect as attaching them.
    """
    if not media_paths:
        return message or "(no message)"

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

    if not blocks and not vid_valid:
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

    In free-agent mode that's the orchestrator (it owns the whole turn and keeps
    the multi-turn conversation); otherwise it's the legacy Brain/assembler.
    """
    orch = getattr(pipeline, "_orchestrator_agent", None)
    if orch is not None and getattr(pipeline, "_free_agent", False):
        return orch
    return getattr(pipeline, "_assemble_workflow", None)


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
                         canvas_selection: list | None = None) -> None:
    """Drive the pipeline for one turn on a private event loop, pushing SSE dicts
    to *out_q*. Interactive asks register on ``_reply_registry`` so POST
    /agentY/reply can feed the answer thread-safely. Terminates *out_q* with None.
    """
    pipeline = _agent_ref
    if pipeline is None:
        out_q.put({"type": "error", "message": "pipeline not initialised"})
        out_q.put(None)
        return

    # Surface CLI-side status notices (e.g. the FAISS memory layer initialising)
    # in the panel too: for the life of this turn, status_bus fans notices out
    # onto out_q as live ``status_line`` events (unregistered in finally).
    status_bus.register_live(out_q)

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
            staged = _stage_into_comfy_input(p) if kind in ("image", "video") else None
            if kind in ("image", "video"):
                try:
                    cs.add_gallery_image(thread_id, p, "")
                except Exception:
                    pass
            out_q.put({
                "type": "output", "kind": kind, "path": p,
                "filename": staged, "name": os.path.basename(p),
                "node_candidates": _NODE_CANDIDATES.get(kind, []),
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
        if chunk.startswith("⬇️ ") or "🎨 [" in chunk:
            out_q.put({"type": "progress", "data": chunk.strip()})
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
    from agenty_core.utils.progress_signal import drain as _drain_progress_lines

    def _flush_activity() -> None:
        # Executor progress emitted from inside a tool call (e.g. run_workflow_now,
        # which drives chained hook stages) only reaches the CLI unless drained
        # here — the pipeline's own loop is blocked awaiting the tool. Draining the
        # progress buffer on the pump's short timer streams it to the panel live.
        # drain() is atomic, so this never double-emits with the pipeline's drain.
        for _line in _drain_progress_lines():
            _translate({"data": _line})
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
                canvas_selection=canvas_selection,
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

    stopped = False
    try:
        loop.run_until_complete(task)
        _check_outputs()
    except asyncio.CancelledError:
        # User pressed Stop → the task was cancelled from /agentY/stop.
        stopped = True
        logger.info("pipeline run %s stopped by user", req_id)
    except Exception as exc:
        logger.error("pipeline stream error: %s", exc, exc_info=True)
        out_q.put({"type": "error", "message": str(exc)})
    finally:
        status_bus.unregister_live(out_q)
        with _reply_lock:
            _reply_registry.pop(req_id, None)
            _run_registry.pop(req_id, None)
        _flush_activity()  # emit any tool/canvas activity left after the last event
        text = "".join(assistant_parts).strip()
        if text:
            try:
                cs.add_message(thread_id, "assistant", text)
            except Exception:
                pass
        _save_state(pipeline, thread_id)
        try:
            loop.run_until_complete(pipeline._await_pending_compression())  # type: ignore[attr-defined]
        except Exception:
            pass
        # Let the background auto-title finish (first turn only) so the thread
        # list shows the short summary when the panel refreshes on `done`.
        if title_thread is not None:
            title_thread.join(timeout=6.0)
        if stopped:
            out_q.put({"type": "system", "data": "⏹ Stopped."})
        out_q.put({"type": "done"})
        out_q.put(None)
        _close_loop(loop)


# ── Stop / interrupt helpers ──────────────────────────────────────────────────

def _interrupt_comfy() -> None:
    """Best-effort: tell ComfyUI to interrupt any running job (POST /interrupt)."""
    try:
        from agenty_core.utils.comfyui_client import get_client
        get_client().post("/interrupt", json_data={})
    except Exception as exc:  # noqa: BLE001
        logger.debug("ComfyUI interrupt failed: %s", exc)


def _cancel_run(req_id: str) -> bool:
    """Cancel the active pipeline run *req_id* and interrupt ComfyUI.

    Returns True when a matching run was found and its cancellation scheduled.
    Always interrupts ComfyUI (harmless when nothing is running) so a Stop during
    a GPU generation halts the job too, not just the agent loop.
    """
    _interrupt_comfy()
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


def _handle_command(thread_id: str, text: str, canvas_prompt: dict | None = None) -> list[dict]:
    low = text.strip().lower()
    parts = text.strip().split(None, 2)
    cmd = parts[0].lower()

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


# Pipeline agents that can be swapped live, and the utility settings keys that
# are read from settings.json on demand rather than held as a live agent.
_SWITCHABLE_AGENTS = ("orchestrator", "query_templates", "info", "planner")
_SWITCH_UTILITY_KEYS = ("build_skill", "llm_functions", "executor_vision_model")


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
    # settings; update it so the rebuilt agent picks up the requested model.
    _OPENAI_COMPAT = _DASHSCOPE_PROVIDERS | _OPENAI_PROVIDERS | _GEMINI_PROVIDERS
    if provider in _OPENAI_COMPAT:
        get_settings().setdefault("llm", {}).setdefault("pipeline", {})[agent_name] = llm_spec

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

    factory = {
        "query_templates": create_query_templates_agent,
        "info": create_info_agent, "planner": create_planner_agent,
    }[agent_name]
    attr = {
        "query_templates": "_researcher",
        "info": "_info_agent", "planner": "_planner_agent",
    }[agent_name]
    kwargs = {"llm": provider}
    if provider not in _OPENAI_COMPAT and model:
        kwargs["ollama_model" if provider == "ollama" else "anthropic_model"] = model
    try:
        setattr(_agent_ref, attr, factory(**kwargs))
        return None
    except Exception as exc:  # noqa: BLE001
        return str(exc)


def _switch_model(args: list[str]) -> list[dict]:
    AGENTS = set(_SWITCHABLE_AGENTS)
    SETTINGS_KEYS = set(_SWITCH_UTILITY_KEYS)
    ALL = AGENTS | SETTINGS_KEYS
    if len(args) < 2:
        return [_sys("⚠️ Usage: `/switch_model <agent|all> <provider,model>`\n\n"
                     f"Agents: `{', '.join(sorted(AGENTS))}`\n"
                     f"Utilities: `{', '.join(sorted(SETTINGS_KEYS))}`\n"
                     "Use `all` to switch every agent at once — e.g. "
                     "`/switch_model all claude,claude-haiku-4-5`.")]
    agent_name = args[0].lower()
    llm_spec = args[1].strip()
    provider, _, model = llm_spec.partition(",")
    provider = provider.strip().lower()
    model = model.strip()
    from src.agent import _DASHSCOPE_PROVIDERS, _OPENAI_PROVIDERS, _GEMINI_PROVIDERS
    if provider not in ({"claude", "ollama"} | _DASHSCOPE_PROVIDERS | _OPENAI_PROVIDERS | _GEMINI_PROVIDERS):
        return [_sys(f"❌ Unknown provider `{provider}`. Use `claude`, `ollama`, "
                     "`dashscope`, `openai`, or `google`.")]

    # ── all: switch every pipeline agent in one go ──────────────────────────
    if agent_name == "all":
        if _agent_ref is None:
            return [_sys("⚠️ Pipeline not initialised.")]
        failures = [f"`{a}`: {err}"
                    for a in sorted(AGENTS)
                    if (err := _rebuild_agent(a, provider, model, llm_spec))]
        if failures:
            return [_sys(f"⚠️ Switched agents to `{llm_spec}`, but some failed:\n" + "\n".join(failures))]
        return [_sys(f"✅ All {len(AGENTS)} agents now using `{llm_spec}`.\n\n"
                     "_(The vision/utility keys `llm_functions` and `executor_vision_model` "
                     "were left unchanged — switch those individually if needed.)_")]

    if agent_name not in ALL:
        return [_sys(f"❌ Unknown agent/utility `{agent_name}`. Valid: `{', '.join(sorted(ALL))}`, or `all`.")]

    # ── utility settings keys (read from settings.json on demand) ───────────
    if agent_name in SETTINGS_KEYS:
        from src.agent import _settings as get_settings
        get_settings().setdefault("llm", {}).setdefault("pipeline", {})[agent_name] = llm_spec
        return [_sys(f"✅ `{agent_name}` now using `{llm_spec}`.")]

    # ── single pipeline agent ───────────────────────────────────────────────
    err = _rebuild_agent(agent_name, provider, model, llm_spec)
    if err:
        return [_sys(f"❌ Failed to switch model: {err}")]
    return [_sys(f"✅ `{agent_name}` now using `{llm_spec}`.")]


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
_MODEL_CACHE_TTL = 300  # seconds

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
    if cached is not None and (now - _MODEL_CACHE.get("ts", 0)) < _MODEL_CACHE_TTL:
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

    try:
        import requests  # noqa: PLC0415
        from src.agent import _cfg  # noqa: PLC0415
        host = str(_cfg("OLLAMA_HOST", "ollama", "host", default="http://localhost:11434"))
        resp = requests.get(f"{host}/api/tags", timeout=3)
        resp.raise_for_status()
        names = sorted({m.get("name", "") for m in resp.json().get("models", []) if m.get("name")})
        if names:
            groups["Ollama"] = [[f"ollama,{n}", n] for n in names]
    except Exception as exc:  # noqa: BLE001 — Ollama not running ⇒ hide the vendor
        logger.debug("Ollama model list unavailable: %s", exc)

    _MODEL_CACHE["groups"] = groups
    _MODEL_CACHE["ts"] = now
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
]


def _env_path() -> Path:
    return _project_root() / ".env"


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
    return f"data: {json.dumps(obj, ensure_ascii=False)}\n\n"


def _build_app():
    from flask import Flask, jsonify, request, Response, stream_with_context

    app = Flask("agentY_bridge")
    app.logger.disabled = True

    @app.after_request
    def _cors(resp):
        resp.headers["Access-Control-Allow-Origin"] = "*"
        resp.headers["Access-Control-Allow-Headers"] = "Content-Type"
        resp.headers["Access-Control-Allow-Methods"] = "GET, POST, DELETE, OPTIONS"
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
                        "project_root": str(_project_root())})

    @app.route("/agentY/commands", methods=["GET"])
    def commands():
        return jsonify(SLASH_COMMANDS)

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
            return jsonify({
                "env": env,
                "env_keys": list(dict.fromkeys(_KNOWN_ENV_KEYS + list(env.keys()))),
                "settings": settings,
                "model_groups": _available_models(),
                "pricing": _load_pricing_config(),
            })
        # POST — persist env and/or settings changes (settings → settings.local.json).
        body = request.get_json(silent=True) or {}
        result: dict = {"ok": True}
        try:
            env_updates = body.get("env")
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
        # reached the client), cancel by thread. _cancel_run already interrupts
        # ComfyUI; ensure we do so even when nothing matched.
        if not found and thread_id:
            found = _cancel_run_by_thread(thread_id)
        if not found:
            _interrupt_comfy()
        return jsonify({"ok": True, "cancelled": found})

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

        # Persist the user's message (raw text).
        if message:
            cs.add_message(thread_id, "user", message)

        # Slash command? Handle synchronously, stream the result lines.
        is_slash = message.startswith("/") or message.lower() in {
            "restart", "stop", "unload", "clearhistory", "images", "resend"
        }
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
                                 "canvas_selection": canvas_selection},
                         daemon=True).start()

        def gen():
            yield _sse({"type": "thread", "id": thread_id})
            yield _sse({"type": "request", "request_id": rid})
            while True:
                try:
                    item = q.get(timeout=15)
                except queue.Empty:
                    yield ": keep-alive\n\n"  # keep the stream warm / defeat idle buffering
                    continue
                if item is None:
                    break
                yield _sse(item)
        return _sse_response(gen())

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


def _register_with_comfyui() -> None:
    """Best-effort: tell the ComfyUI sidebar extension where this host lives, so its
    "Start server" button can relaunch run_agent.ps1 when the host is down. Fire-
    and-forget; never blocks or fails startup."""
    def _do() -> None:
        try:
            from src.utils.settings import load_settings
            import urllib.request
            base = str(load_settings().get("comfyui_url", "http://127.0.0.1:8188")).rstrip("/")
            body = json.dumps({"project_root": str(_project_root()),
                               "run_script": "run_agent.ps1"}).encode("utf-8")
            req = urllib.request.Request(base + "/agent/register_host", data=body,
                                         headers={"Content-Type": "application/json"}, method="POST")
            urllib.request.urlopen(req, timeout=5).read()
            logger.info("registered agentY host location with the ComfyUI extension")
        except Exception as exc:  # noqa: BLE001
            logger.debug("host self-registration skipped: %s", exc)
    threading.Thread(target=_do, name="agentY-register-host", daemon=True).start()


def start_agentY_server(agent, host: str = "127.0.0.1", port: int = 5000) -> bool:
    """Start the agentY bridge + chat host in a background daemon thread."""
    global _server_thread, _agent_ref
    _agent_ref = agent
    cs.init_db()

    if _server_thread is not None and _server_thread.is_alive():
        return True
    try:
        from flask import Flask  # noqa: F401
    except ImportError:
        logger.error("Flask is not installed. Run: pip install flask")
        return False

    app = _build_app()

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
    _register_with_comfyui()  # so the sidebar's "Start server" button knows where we live
    return True
