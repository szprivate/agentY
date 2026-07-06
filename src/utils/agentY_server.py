"""
agentY bridge + chat host — runs on localhost:5000.

This is the backend for the **ComfyUI-native agentY chat UI** (the sidebar
custom-node panel in ``comfyui_extension/agentY-comfyuiConnect``). It replaces the
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


# ── Slash commands (mirrors the frontend popup list) ──────────────────────────

SLASH_COMMANDS = [
    {"name": "/restart",         "description": "Restart the agent pipeline"},
    {"name": "/stop",            "description": "Stop and shut down the agent"},
    {"name": "/unload",          "description": "Unload Ollama models from VRAM"},
    {"name": "/clear_vram",      "description": "Clear ComfyUI GPU VRAM"},
    {"name": "/images",          "description": "List images generated in this thread (reference them by number)"},
    {"name": "/clearhistory",    "description": "Delete all conversation history (keeps the current thread)"},
    {"name": "/switch_model",    "description": "Switch an agent's LLM — /switch_model <agent|all> <provider,model> (use 'all' for every agent)"},
    {"name": "/add_workflow",    "description": "Add a ComfyUI workflow — /add_workflow <path/to/workflow.json>"},
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


# ── Content builder (text + attached images -> Strands content blocks) ────────

def _build_content(message: str, image_paths: list[str]) -> list | str:
    """Build a Strands-compatible content list from text + image file paths.

    Images are downsized to satisfy Claude's 5 MB / 1568 px constraints.
    """
    if not image_paths:
        return message or "(no message)"

    from src.tools.image_handling import _downsize, _detect_format, _MAX_IMAGE_BYTES

    blocks: list = []
    valid: list[str] = []
    for path in image_paths:
        try:
            raw = Path(path).read_bytes()
            img_fmt = _detect_format(path) or "png"
            image_bytes, img_fmt = _downsize(raw, img_fmt)
            if len(image_bytes) > _MAX_IMAGE_BYTES:
                raise ValueError(f"Image still {len(image_bytes):,} bytes after downsize — skipping")
            blocks.append({"image": {"format": img_fmt, "source": {"bytes": image_bytes}}})
            valid.append(path)
        except Exception as exc:
            logger.warning("Could not load image %s: %s", path, exc)

    if not blocks:
        return message or "(no message)"

    path_lines = "\n".join(
        f"  - {p}  [image, use this path for ComfyUI input]" for p in valid if os.path.exists(p)
    )
    paths_info = f"\n\nAttached image file paths (use these for ComfyUI):\n{path_lines}" if path_lines else ""
    intro = message if message else "The user sent an image for processing."
    blocks.insert(0, {"text": intro + paths_info})
    return blocks


# ── Pipeline state save / restore (per thread) ────────────────────────────────

def _reset_pipeline_state(pipeline) -> None:
    """Wipe per-conversation state from the shared pipeline singleton."""
    brain = getattr(pipeline, "_assemble_workflow", None)
    if brain is not None and hasattr(brain, "messages"):
        brain.messages.clear()
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
    brain = getattr(pipeline, "_assemble_workflow", None)
    cached = _thread_brain_cache.get(thread_id)
    if brain is not None and hasattr(brain, "messages") and cached is not None:
        brain.messages[:] = cached
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
    brain = getattr(pipeline, "_assemble_workflow", None)
    if brain is not None and hasattr(brain, "messages"):
        _thread_brain_cache[thread_id] = list(brain.messages)
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


# ── SSE pipeline runner ───────────────────────────────────────────────────────

def _run_pipeline_stream(thread_id: str, message: str, image_paths: list[str],
                         out_q: "queue.Queue", req_id: str) -> None:
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
    with _reply_lock:
        _reply_registry[req_id] = (loop, qa_queue)

    async def _run() -> None:
        async for event in pipeline.stream_async(content, qa_reply_queue=qa_queue):
            if isinstance(event, dict):
                _translate(event)

    try:
        loop.run_until_complete(_run())
        _check_outputs()
    except Exception as exc:
        logger.error("pipeline stream error: %s", exc, exc_info=True)
        out_q.put({"type": "error", "message": str(exc)})
    finally:
        with _reply_lock:
            _reply_registry.pop(req_id, None)
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
        out_q.put({"type": "done"})
        out_q.put(None)
        try:
            loop.close()
        except Exception:
            pass


# ── Slash-command handlers (return a list of SSE event dicts) ─────────────────

def _sys(text: str) -> dict:
    return {"type": "system", "data": text}


def _handle_command(thread_id: str, text: str) -> list[dict]:
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
            return [_sys("⚠️ Usage: `/add_workflow <path_to_workflow.json>`")]
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
    from agenty_core.paths import corpus_root
    wf_path = Path(path_str)
    if not wf_path.exists():
        return [_sys(f"❌ File not found: `{wf_path}`")]
    try:
        from src.utils.workflow_parser import parse_workflow, _custom_index_path
        wf_data = json.loads(wf_path.read_text(encoding="utf-8"))
        stem = wf_path.stem
        parse_workflow(wf_data, name=stem, update_index=True)
        description = ""
        try:
            import importlib.util as ilu
            bs_path = str(_project_root() / "scripts" / "build_skill.py")
            mod = sys.modules.get("_agenty_build_skill")
            if mod is None:
                spec = ilu.spec_from_file_location("_agenty_build_skill", bs_path)
                mod = ilu.module_from_spec(spec)
                sys.modules["_agenty_build_skill"] = mod
                spec.loader.exec_module(mod)
            description = mod._generate_workflow_template_description(wf_data, stem)
        except Exception as exc:
            description = ""
            logger.warning("description generation failed: %s", exc)
        tpl_path = corpus_root() / "config" / "workflow_templates.json"
        tpl = json.loads(tpl_path.read_text(encoding="utf-8")) if tpl_path.exists() else {}
        if stem not in tpl:
            tpl[stem] = description
            tpl_path.write_text(json.dumps(tpl, indent=4, ensure_ascii=False) + "\n", encoding="utf-8")
        return [_sys(f"✅ Workflow `{stem}` registered in `{_custom_index_path()}`."
                     + (f"\n\n**Description:**\n{description}" if description else ""))]
    except Exception as exc:
        return [_sys(f"❌ Failed to add workflow: {exc}")]


def _remove_workflow(name: str) -> list[dict]:
    from agenty_core.paths import corpus_root
    try:
        from src.utils.workflow_parser import workflow_remove, _custom_index_path
        idx = workflow_remove(name)
        tpl_path = corpus_root() / "config" / "workflow_templates.json"
        if tpl_path.exists():
            tpl = json.loads(tpl_path.read_text(encoding="utf-8"))
            if name in tpl:
                del tpl[name]
                tpl_path.write_text(json.dumps(tpl, indent=4, ensure_ascii=False) + "\n", encoding="utf-8")
        kebab = name.lower().replace("_", "-")
        skill_dir = _project_root() / "skills" / kebab
        if skill_dir.exists():
            shutil.rmtree(skill_dir)
        return [_sys(f"✅ Workflow `{name}` removed from `{idx}`.")]
    except Exception as exc:
        return [_sys(f"❌ Failed to remove workflow: {exc}")]


# Pipeline agents that can be swapped live, and the utility settings keys that
# are read from settings.json on demand rather than held as a live agent.
_SWITCHABLE_AGENTS = ("query_templates", "assemble_workflow", "info", "story",
                      "detect_user_intent", "planner", "error_checker", "dop")
_SWITCH_UTILITY_KEYS = ("build_skill", "llm_functions", "executor_vision_model")


def _rebuild_agent(agent_name: str, provider: str, model: str, llm_spec: str) -> str | None:
    """Rebuild one pipeline agent with the given provider/model and swap it into
    the live pipeline. Returns None on success, or an error string."""
    from src.agent import (
        _DASHSCOPE_PROVIDERS, _settings as get_settings,
        create_query_templates_agent, create_assemble_workflow_agent, create_info_agent,
        create_story_agent, create_detect_user_intent_agent, create_planner_agent,
        create_error_checker_agent, create_dop_agent,
    )
    if _agent_ref is None:
        return "pipeline not initialised"
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
    if provider in _DASHSCOPE_PROVIDERS:
        # DashScope factories read their model from settings; update it so the
        # rebuilt agent picks up the requested Qwen model.
        get_settings().setdefault("llm", {}).setdefault("pipeline", {})[agent_name] = llm_spec
    elif model:
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

    # ── Health / commands ──────────────────────────────────────────────────
    @app.route("/agentY/health", methods=["GET"])
    def health():
        return jsonify({"status": "ok", "pipeline": _agent_ref is not None})

    @app.route("/agentY/commands", methods=["GET"])
    def commands():
        return jsonify(SLASH_COMMANDS)

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
        return jsonify(t)

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

    # ── Chat (SSE) ─────────────────────────────────────────────────────────
    @app.route("/agentY/chat", methods=["POST", "OPTIONS"])
    def chat():
        if request.method == "OPTIONS":
            return "", 204
        body = request.get_json(silent=True) or {}
        message = (body.get("message") or "").strip()
        image_paths = [p for p in (body.get("image_paths") or []) if isinstance(p, str)]
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
            result = _handle_command(thread_id, message)
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
                            item = q.get()
                            if item is None:
                                break
                            yield _sse(item)
                    return Response(stream_with_context(gen_resend()), mimetype="text/event-stream")

            def gen_cmd():
                yield _sse({"type": "thread", "id": thread_id})
                for ev in result:
                    yield _sse(ev)
                yield _sse({"type": "done"})
            return Response(stream_with_context(gen_cmd()), mimetype="text/event-stream")

        # Normal turn → run the pipeline and stream SSE.
        q: queue.Queue = queue.Queue()
        rid = uuid.uuid4().hex
        threading.Thread(target=_run_pipeline_stream,
                         args=(thread_id, message, image_paths, q, rid), daemon=True).start()

        def gen():
            yield _sse({"type": "thread", "id": thread_id})
            yield _sse({"type": "request", "request_id": rid})
            while True:
                item = q.get()
                if item is None:
                    break
                yield _sse(item)
        return Response(stream_with_context(gen()), mimetype="text/event-stream")

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
