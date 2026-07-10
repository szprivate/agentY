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
        from src.utils.comfyui_client import get_client, parse_argv_dir_flag
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
    # 2. Derive from settings.json comfyui_user_dir.
    try:
        cfg_path = _project_root() / "config" / "settings.json"
        if cfg_path.exists():
            cfg = json.loads(
                "".join(ln for ln in cfg_path.read_text(encoding="utf-8").splitlines(keepends=True)
                        if not ln.lstrip().startswith("//"))
            )
            ud = cfg.get("comfyui_user_dir")
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
        from src.utils.comfyui_client import get_client, parse_argv_dir_flag
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
            loop.close()
        title = _clean_title(raw)
        if title:
            cs.rename_thread(thread_id, title)
    except Exception as exc:  # noqa: BLE001
        logger.debug("auto-title failed for %s: %s", thread_id, exc)


# ── SSE pipeline runner ───────────────────────────────────────────────────────

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

    def _flush_activity() -> None:
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
        try:
            loop.close()
        except Exception:
            pass


# ── Stop / interrupt helpers ──────────────────────────────────────────────────

def _interrupt_comfy() -> None:
    """Best-effort: tell ComfyUI to interrupt any running job (POST /interrupt)."""
    try:
        from src.utils.comfyui_client import get_client
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
_SWITCHABLE_AGENTS = ("orchestrator", "query_templates", "assemble_workflow", "info",
                      "story", "planner", "error_checker", "dop", "detect_user_intent")
_SWITCH_UTILITY_KEYS = ("build_skill", "llm_functions", "executor_vision_model")


def _rebuild_agent(agent_name: str, provider: str, model: str, llm_spec: str) -> str | None:
    """Rebuild one pipeline agent with the given provider/model and swap it into
    the live pipeline. Returns None on success, or an error string."""
    from src.agent import (
        _DASHSCOPE_PROVIDERS, _settings as get_settings,
        create_orchestrator_agent,
        create_query_templates_agent, create_assemble_workflow_agent, create_info_agent,
        create_story_agent, create_detect_user_intent_agent, create_planner_agent,
        create_error_checker_agent, create_dop_agent,
    )
    if _agent_ref is None:
        return "pipeline not initialised"

    # DashScope factories read their model from settings; update it so the rebuilt
    # agent picks up the requested Qwen model.
    if provider in _DASHSCOPE_PROVIDERS:
        get_settings().setdefault("llm", {}).setdefault("pipeline", {})[agent_name] = llm_spec

    # The orchestrator is rebuilt specially: its tool list must include the
    # pipeline's delegation tools, and it must be re-wired (skills plugin + live
    # context) via set_orchestrator rather than a plain setattr.
    if agent_name == "orchestrator":
        kwargs = {"llm": provider, "extra_tools": getattr(_agent_ref, "_delegation_tools", None)}
        if provider not in _DASHSCOPE_PROVIDERS and model:
            kwargs["ollama_model" if provider == "ollama" else "anthropic_model"] = model
        try:
            _agent_ref.set_orchestrator(create_orchestrator_agent(**kwargs))
            return None
        except Exception as exc:  # noqa: BLE001
            return str(exc)

    factory = {
        "query_templates": create_query_templates_agent, "assemble_workflow": create_assemble_workflow_agent,
        "info": create_info_agent, "story": create_story_agent,
        "detect_user_intent": create_detect_user_intent_agent, "planner": create_planner_agent,
        "error_checker": create_error_checker_agent, "dop": create_dop_agent,
    }[agent_name]
    attr = {
        "query_templates": "_researcher", "assemble_workflow": "_assemble_workflow",
        "info": "_info_agent", "story": "_story_agent", "detect_user_intent": "_triage_agent",
        "planner": "_planner_agent", "error_checker": "_error_checker_agent", "dop": "_dop_agent",
    }[agent_name]
    kwargs = {"llm": provider}
    if provider not in _DASHSCOPE_PROVIDERS and model:
        kwargs["ollama_model" if provider == "ollama" else "anthropic_model"] = model
    try:
        setattr(_agent_ref, attr, factory(**kwargs))
        # Keep the legacy _brain alias in sync when the assembler is swapped.
        if agent_name == "assemble_workflow":
            _agent_ref._brain = _agent_ref._assemble_workflow
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
    from src.agent import _DASHSCOPE_PROVIDERS
    if provider not in ({"claude", "ollama"} | _DASHSCOPE_PROVIDERS):
        return [_sys(f"❌ Unknown provider `{provider}`. Use `claude`, `ollama`, or `dashscope`.")]

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
            loop.close()

    threading.Thread(target=_run, name="agentY-review-dispatch", daemon=True).start()


# ── Model discovery (for the panel's quick-switch dropdown) ───────────────────

# Curated catalogs for the cloud vendors. Ollama is enumerated live from the
# running server. Each entry is [ "<provider>,<model>", "Display name" ] — the
# provider,model string is exactly what /switch_model expects.
_ANTHROPIC_MODELS = [
    ["claude,claude-haiku-4-5", "Claude Haiku 4.5"],
    ["claude,claude-sonnet-4-5", "Claude Sonnet 4.5"],
]
_DASHSCOPE_MODELS = [
    ["dashscope,qwen3.6-flash", "Qwen3.6 Flash"],
    ["dashscope,qwen-plus", "Qwen Plus"],
    ["dashscope,qwen3.7-plus", "Qwen3.7 Plus"],
    ["dashscope,qwen-max", "Qwen Max"],
]


def _available_models() -> dict:
    """Return {vendor: [[spec, label], …]} for every vendor currently usable.

    A vendor is included only when it can actually be reached: Anthropic /
    DashScope when their API key is set, Ollama when its server answers (its
    installed models are listed live via ``GET {host}/api/tags``).
    """
    groups: dict[str, list] = {}
    if os.environ.get("ANTHROPIC_API_KEY"):
        groups["Anthropic"] = list(_ANTHROPIC_MODELS)
    if os.environ.get("DASHSCOPE_API_KEY") or os.environ.get("ALIBABA_API_KEY"):
        groups["Alibaba (DashScope)"] = list(_DASHSCOPE_MODELS)
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
    return groups


# ── Application settings (.env auth keys + config/settings.json) ──────────────

# Auth / host keys surfaced in the settings modal even when absent from .env, so
# the user can fill them in. Any additional keys already present in .env are
# merged in on read.
# Auth/connection keys surfaced in the settings UI's ".env" section. Host/port
# (agent_server_url) and the conversation DB live in settings.json now, not here. The
# DashScope endpoint sits right beneath its API key; it is pre-seeded from
# settings.json (llm.dashscope.base_url) in the GET so the field is never blank.
_KNOWN_ENV_KEYS = [
    "HF_TOKEN", "ANTHROPIC_API_KEY", "COMFYUI_API_KEY",
    "DASHSCOPE_API_KEY", "DASHSCOPE_BASE_URL",
]


def _env_path() -> Path:
    return _project_root() / ".env"


def _settings_path() -> Path:
    return _project_root() / "config" / "settings.json"


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
    from *old* (recursing into dicts). Used to apply only the changed values to
    settings.json, so untouched lines — and their comments — stay byte-identical."""
    changes: list = []
    if isinstance(new, dict):
        base = old if isinstance(old, dict) else {}
        for k, v in new.items():
            changes.extend(_diff_leaves(base.get(k), v, prefix + (k,)))
    else:
        if old != new:
            changes.append((prefix, new))
    return changes


def _find_leaf_line(lines: list, path: tuple) -> int:
    """Index of the line defining leaf *path* in a standard 2-space-indented JSON
    file (one key per line; objects open with a trailing '{' and close on their
    own line). Returns -1 if not found. Full-line ``//`` comments are ignored."""
    stack: list[str] = []
    for i, raw in enumerate(lines):
        s = raw.strip()
        if not s or s.startswith("//"):
            continue
        if s[0] in "}]":
            if stack:
                stack.pop()
            continue
        m = re.match(r'"((?:[^"\\]|\\.)*)"\s*:\s*(.*)$', s)
        if not m:
            continue
        key = m.group(1)
        rest = m.group(2).rstrip().rstrip(",")
        cur = tuple(stack) + (key,)
        if rest.endswith("{") or rest.endswith("["):
            stack.append(key)
        elif cur == tuple(path):
            return i
    return -1


def _update_settings_file(new_settings: dict) -> list:
    """Apply changed leaves from *new_settings* onto config/settings.json in
    place, preserving comments/formatting. Returns the list of applied paths."""
    path = _settings_path()
    text = path.read_text(encoding="utf-8")
    from src.agent import _load_settings  # comment-stripping parser
    current = _load_settings()
    updates = _diff_leaves(current, new_settings)
    if not updates:
        return []
    lines = text.splitlines(keepends=True)
    bare = [l.rstrip("\n") for l in lines]
    applied: list = []
    for pathv, value in updates:
        idx = _find_leaf_line(bare, pathv)
        if idx < 0:
            continue  # only existing leaves are edited
        raw = lines[idx]
        nl = "\n" if raw.endswith("\n") else ""
        body = raw[:-len(nl)] if nl else raw
        m = re.match(r'^(\s*"(?:[^"\\]|\\.)*"\s*:\s*)(.*)$', body)
        if not m:
            continue
        trailing_comma = "," if m.group(2).rstrip().endswith(",") else ""
        lines[idx] = f"{m.group(1)}{json.dumps(value, ensure_ascii=False)}{trailing_comma}{nl}"
        applied.append(".".join(str(p) for p in pathv))
    path.write_text("".join(lines), encoding="utf-8")
    # Invalidate the cached settings so _cfg() / future agent rebuilds see them.
    try:
        import src.agent as _agentmod
        _agentmod._SETTINGS = {}
    except Exception:  # noqa: BLE001
        pass
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
        return jsonify({"status": "ok", "pipeline": _agent_ref is not None})

    @app.route("/agentY/commands", methods=["GET"])
    def commands():
        return jsonify(SLASH_COMMANDS)

    # ── Available models (per vendor) for the quick-switch dropdown ─────────
    @app.route("/agentY/models", methods=["GET"])
    def models():
        return jsonify(_available_models())

    # ── Application settings (.env auth keys + config/settings.json) ────────
    @app.route("/agentY/settings", methods=["GET", "POST", "OPTIONS"])
    def settings_route():
        if request.method == "OPTIONS":
            return "", 204
        if request.method == "GET":
            from src.agent import _load_settings
            settings = _load_settings()
            env = {k: "" for k in _KNOWN_ENV_KEYS}
            env.update(_read_env_file())
            # Show the DashScope endpoint (beneath its API key) even when it's not
            # overridden in .env — seed it from settings.json so it's never blank.
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
            })
        # POST — persist env and/or settings.json changes.
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
        # Canvas-selected inputs lead (they carry the user's chosen order), then
        # any chat attachments.
        image_paths = canvas_paths + image_paths
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
    return True
