"""
Image handling tools – upload, download, resolution, and visual analysis.

Consolidates all image-related @tool functions:
  • upload_image: push images to ComfyUI's input folder
  • view_image: download images from ComfyUI's output
  • get_image_resolution: read local image dimensions
  • analyze_image: forward an image to the model for visual inspection
"""

import json
import os
import re
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Optional

import requests
from PIL import Image
from strands import Agent, tool

from agenty_core.utils.comfyui_client import get_client


# ─────────────────────────────────────────────────────────────────────────────
# Vision agent registry – injected by the pipeline at startup
# ─────────────────────────────────────────────────────────────────────────────

_vision_agent: Optional[Agent] = None


class AgentPool:
    """A bounded pool of interchangeable, stateless single-shot agents.

    The orchestrator routinely asks about several images in ONE assistant turn.
    Strands runs those tool calls concurrently (``ConcurrentToolExecutor``) and
    each sync ``@tool`` lands in its own thread (``asyncio.to_thread``), so they
    all reach the vision agent at the same moment. A Strands ``Agent`` refuses
    that: ``stream_async`` takes ``_invocation_lock`` non-blocking and raises
    ``ConcurrencyException`` for every caller but the first. Sharing one instance
    therefore meant N images in, exactly one description out — the rest fell
    through to ``mode='full'`` and came back as "the image itself is not shown",
    which the model then papered over by inventing the missing descriptions.

    One agent cannot be re-entered, so parallelism needs *more agents*, not a
    cleverer lock. Callers borrow an instance for the duration of a call and hand
    it back. Instances are created lazily — a lone `analyze_image` never pays for
    a second model handshake — and capped at ``size``, which is what keeps a
    local Ollama on one GPU from being asked to run four generations at once.
    ``size=1`` reproduces strict serialisation.
    """

    def __init__(self, primary: Agent, factory=None, size: int = 1):
        self.primary = primary
        self._factory = factory
        # A factory is what allows growth; without one the pool is just `primary`.
        self.size = max(1, int(size)) if factory else 1
        self._sem = threading.Semaphore(self.size)
        self._lock = threading.Lock()
        self._free: list[Agent] = [primary]
        # Every instance ever handed out, in-flight or not. The pipeline folds
        # each one's token delta into the turn cost, so an agent the pool grew
        # must not be able to spend tokens invisibly.
        self._all: list[Agent] = [primary]

    def instances(self) -> list[Agent]:
        """Every agent this pool has created, for usage/cost accounting."""
        with self._lock:
            return list(self._all)

    def _take(self) -> Agent:
        with self._lock:
            if self._free:
                return self._free.pop()
            if len(self._all) < self.size and self._factory:
                try:
                    agent = self._factory()
                    self._all.append(agent)
                    return agent
                except Exception as exc:  # noqa: BLE001
                    print(f"[AgentPool] could not add an instance ({exc}); "
                          "waiting for a free one instead.")
        # The semaphore admitted us, so an instance is in flight and will be
        # returned; block for it rather than failing the call.
        while True:
            with self._lock:
                if self._free:
                    return self._free.pop()
            time.sleep(0.01)

    @contextmanager
    def borrow(self):
        """Yield an agent nobody else is using; return it on the way out."""
        self._sem.acquire()
        agent = None
        try:
            agent = self._take()
            yield agent
        finally:
            if agent is not None:
                with self._lock:
                    self._free.append(agent)
            self._sem.release()


_vision_pool: Optional[AgentPool] = None


def set_vision_agent(agent: Agent, factory=None, max_parallel: int = 1) -> None:
    """Register the Vision :class:`~strands.Agent` used by :func:`analyze_image`.

    Call this once during pipeline initialisation before any ``analyze_image``
    invocations that use ``mode='describe'``.

    Args:
        agent:        The first (and, without a factory, only) vision agent.
        factory:      Zero-arg callable building another equivalent agent. Supply
                      it to allow concurrent describes; the pipeline passes
                      ``create_vision_agent``.
        max_parallel: Cap on simultaneous describes. Keep at 1 for a local model
                      on one GPU; raise it for a hosted one, where the calls are
                      network-bound and genuinely overlap.
    """
    global _vision_agent, _vision_pool
    _vision_agent = agent
    _vision_pool = AgentPool(agent, factory=factory, size=max_parallel)


def _ensure_vision_pool() -> Optional[AgentPool]:
    """The pool wrapping the registered agent, built on demand.

    ``_vision_agent`` is assigned directly in a few places (tests, older callers
    that predate the pool). Rather than have those explode on a missing pool,
    wrap whatever agent is registered in a size-1 pool — the previous behaviour.
    """
    global _vision_pool
    if _vision_agent is None:
        return None
    if _vision_pool is None or _vision_pool.primary is not _vision_agent:
        _vision_pool = AgentPool(_vision_agent)
    return _vision_pool


def vision_agents() -> list[Agent]:
    """Every live vision agent, so the pipeline can price all of their tokens."""
    pool = _ensure_vision_pool()
    return pool.instances() if pool else []


# ── moved to agenty_core ─────────────────────────────────────────────────────
# These are the parts agentY and agentY-mcp were maintaining twice: byte
# wrangling (98-100% identical), staging a local file into ComfyUI's input dir,
# fetching a web image, and the presigned PUT. They now live in the shared layer
# and are re-exported here under the names the rest of this repo already imports
# (src/utils/qa.py, agentY_server.py, video_handling.py, annotate.py), so nothing
# downstream had to change.
#
# What stays local is what genuinely differs per host: analyze_image and
# view_image, because this app delegates to a Vision Agent to keep pixels out of
# the orchestrator's context while an MCP host looks at the image itself.
from agenty_core.tools.image_io import (  # noqa: F401
    comfy_input_dir as _comfy_input_dir,
    download_image,
    resolve_local_image as _resolve_local_image,
    stage_image as _stage_image,
    upload_file_to_url,
)
from agenty_core.utils.image_bytes import (  # noqa: F401
    MAX_IMAGE_BYTES as _MAX_IMAGE_BYTES,
    OPTIMAL_LONG_EDGE as _OPTIMAL_LONG_EDGE,
    detect_format as _detect_format,
    downsize as _downsize,
    input_long_edge as _input_long_edge,
)

# ═══════════════════════════════════════════════════════════════════════════════
# Internal helpers
# ═══════════════════════════════════════════════════════════════════════════════









# ═══════════════════════════════════════════════════════════════════════════════
# Tools
# ═══════════════════════════════════════════════════════════════════════════════

def _upload_one(
    file_path: str,
    subfolder: str = "",
    image_type: str = "input",
    overwrite: bool = False,
) -> dict:
    """Stage one image into ComfyUI's input dir, with agentY's dry-run behaviour.

    The staging itself is ``agenty_core.tools.image_io.stage_image`` — resolving a
    bare canvas filename, the idempotency skip, the API upload. What is agentY's
    is the line above it: a dry-run stand-in has no bytes to stage, so answer with
    the name a real upload would have returned, letting the caller bind it to the
    next stage's loader and keeping the chain under test moving.
    """
    try:
        from src.utils import dry_run as _dry
        if _dry.active() and _dry.is_stand_in(file_path):
            return {"name": os.path.basename(str(file_path)), "subfolder": "",
                    "type": image_type or "input",
                    "note": _dry.stand_in_notice(file_path)}
    except Exception:  # noqa: BLE001
        pass
    return _stage_image(file_path, subfolder=subfolder, image_type=image_type,
                        overwrite=overwrite)


@tool
def upload_image(
    file_path: str,
    subfolder: str = "",
    image_type: str = "input",
    overwrite: bool = False,
) -> dict:
    """Upload an image file to the ComfyUI input directory for use in workflows.

    To stage several images at once, prefer ``upload_image_multiple`` — one tool
    call instead of N.

    Args:
        file_path: Local path to the image file.
        subfolder: Subfolder inside the target directory. Defaults to the input
                   root (``""``): LoadImage on some ComfyUI builds cannot load
                   files from input subdirectories, so agent inputs are staged
                   flat and referenced by bare filename.
        image_type: 'input', 'output', or 'temp' (default 'input').
        overwrite: Overwrite existing file with the same name.
    """
    return json.dumps(_upload_one(file_path, subfolder, image_type, overwrite))


@tool
def upload_image_multiple(
    file_paths: list,
    subfolder: str = "",
    image_type: str = "input",
    overwrite: bool = False,
) -> str:
    """Upload several image files to ComfyUI's input directory in one call.

    Thin batch wrapper over ``upload_image`` — use it when staging multiple input
    images for the same turn (a multi-image edit, or the source + reference of a
    Mode C batch) so you make one tool call instead of N. Each file goes through
    the same idempotent logic as ``upload_image`` (a file already in the input dir
    is a no-op) and is reported individually, so a single bad path does not lose
    the others.

    Args:
        file_paths: List of local paths to the image files to stage. A JSON array
                    string or a comma/newline-separated string is also accepted.
        subfolder: Subfolder inside the target directory. Defaults to the input
                   root (``""``) — LoadImage on some ComfyUI builds cannot read
                   input subdirectories, so agent inputs are staged flat and
                   referenced by bare filename.
        image_type: 'input', 'output', or 'temp' (default 'input').
        overwrite: Overwrite existing files with the same name.

    Returns:
        JSON ``{"results": [...], "uploaded": N, "skipped": M, "failed": K}``.
        Each ``results`` entry mirrors ``upload_image``'s response with the source
        ``file_path`` added. Reference a staged image in a ``LoadImage`` node by
        its returned ``name``.
    """
    # Be forgiving about how the list arrives — models sometimes pass a stringified
    # array or a delimited string instead of a real list.
    if isinstance(file_paths, str):
        try:
            parsed = json.loads(file_paths)
            file_paths = parsed if isinstance(parsed, list) else [file_paths]
        except Exception:
            file_paths = [p.strip() for p in re.split(r"[,\n]", file_paths) if p.strip()]
    if not isinstance(file_paths, (list, tuple)):
        return json.dumps({"error": f"file_paths must be a list, got {type(file_paths).__name__}"})

    results = []
    uploaded = skipped = failed = 0
    for fp in file_paths:
        r = _upload_one(str(fp), subfolder, image_type, overwrite)
        entry = dict(r) if isinstance(r, dict) else {"raw": r}
        entry["file_path"] = str(fp)
        if entry.get("error"):
            failed += 1
        elif "already staged" in str(entry.get("note", "")):
            skipped += 1
        else:
            uploaded += 1
        results.append(entry)

    return json.dumps({
        "results": results,
        "uploaded": uploaded,
        "skipped": skipped,
        "failed": failed,
    })




# Extension → Content-Type for the PUT header when the caller doesn't pass one.
_MIME_BY_EXT = {
    ".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
    ".webp": "image/webp", ".gif": "image/gif", ".bmp": "image/bmp",
    ".tif": "image/tiff", ".tiff": "image/tiff", ".mp4": "video/mp4",
    ".webm": "video/webm", ".mov": "video/quicktime", ".pdf": "application/pdf",
}




@tool
def view_image(
    filename: str,
    save_to: str,
    subfolder: str = "",
    image_type: str = "output",
) -> str:
    """Download an image from the ComfyUI output directory and save it to a local path.

    After saving, use analyze_image(file_path=save_to) to inspect the image
    contents.

    Args:
        filename: Image filename on the server e.g. 'ComfyUI_00001_.png'.
        save_to: Local file path to save the image. Required.
        subfolder: Optional subfolder where the image is located.
        image_type: Directory type: 'output', 'input', or 'temp'.
    """
    try:
        params: dict = {"filename": filename, "type": image_type}
        if subfolder:
            params["subfolder"] = subfolder

        resp = get_client().get("/view", params=params, raw=True)
        content_type = resp.headers.get("content-type", "image/png")
        image_bytes = resp.content

        os.makedirs(os.path.dirname(save_to) or ".", exist_ok=True)
        with open(save_to, "wb") as f:
            f.write(image_bytes)
        result = {
            "saved_to": save_to,
            "content_type": content_type,
            "size_bytes": len(image_bytes),
        }
        if len(image_bytes) > 5 * 1024 * 1024:
            result["warning"] = (
                f"Image is {len(image_bytes) / 1024 / 1024:.1f} MB — exceeds 5 MB limit. "
                "Activate the 'image-downsize' skill to produce a smaller copy."
            )
        return json.dumps(result)
    except Exception as e:
        return json.dumps({"error": str(e)})


@tool
def get_image_resolution(image_path: str) -> str:
    """Return the resolution (width and height in pixels) of a local image file.

    Args:
        image_path: Absolute or relative path to the image file on disk.
    """
    resolved = _resolve_local_image(image_path)
    if resolved is None:
        return json.dumps({"error": f"File not found: {image_path}"})
    try:
        with Image.open(resolved) as img:
            width, height = img.size
        return json.dumps({"width": width, "height": height, "image_path": resolved})
    except Exception as e:
        return json.dumps({"error": str(e)})


@tool
def analyze_image(
    file_path: str = "",
    image_url: str = "",
    question: str = "",
    mode: Literal["describe", "full"] = "describe",
) -> dict:
    """Load an image from a local file or URL and analyse it.

    Supported formats: PNG, JPEG/JPG, GIF, WEBP.
    Images are automatically downsized to satisfy Claude's 5 MB / 1568 px constraints.

    Two analysis modes are available:

    * ``mode='describe'`` **(default, recommended)**: The image is sent to an
      isolated Vision Agent call that returns a plain-text description.  No image
      bytes enter the Query Templates' context window (~10K tokens vs ~600K).
    * ``mode='full'``: Legacy behaviour – returns the raw image bytes so the
      Query Templates can reason over pixels directly (~600K tokens).  Use only when
      precise spatial or identity comparison is needed.

    Args:
        file_path: Absolute or relative path to a local image file.
                   Provide either this or ``image_url`` – not both.
        image_url: Public http/https URL of an image to download.
        question:  Specific question or aspect to focus on when analysing the image.
        mode:      ``'describe'`` (default) or ``'full'``.

    Use ``mode='describe'`` unless you need pixel-level reasoning (e.g. comparing
    two images for identical content, precise spatial positioning, or explicit
    user request for raw pixel analysis).
    """
    # A dry run's "generations" are paths and nothing else. Answering "file not
    # found" would read as a failure and send the agent healing something that was
    # never broken — so say what it actually is.
    try:
        from src.utils import dry_run as _dry
        if _dry.active() and _dry.is_stand_in(file_path):
            return {"status": "ok", "content": [{"text": _dry.stand_in_notice(file_path)}]}
    except Exception:  # noqa: BLE001
        pass
    # Test/speed mode: skip the vision model entirely and return a canned
    # description. Used by the query_templates-only reliability sweep so image-input
    # recipes exercise briefing generation without loading a vision model.
    if os.environ.get("AGENTY_STUB_VISION"):
        label = os.path.basename(file_path or image_url) or "input image"
        return {"status": "ok", "content": [{"text": (
            f"Image analysis for {label} (stubbed): a standard test image — a "
            "centered subject on a plain background, natural lighting, ordinary "
            "colours, no text. Suitable as an edit / upscale / animation source."
        )}]}
    data: Optional[bytes] = None
    source_name = ""
    detected_mime = ""

    if file_path:
        resolved = _resolve_local_image(file_path)
        if resolved is None:
            return {"status": "error", "content": [{"text": f"File not found: {file_path}"}]}
        p = Path(resolved)
        source_name = str(p)
        try:
            data = p.read_bytes()
        except Exception as exc:
            return {"status": "error", "content": [{"text": f"Could not read file: {exc}"}]}

    elif image_url:
        source_name = image_url
        try:
            resp = requests.get(image_url, timeout=30)
            resp.raise_for_status()
            detected_mime = resp.headers.get("content-type", "")
            data = resp.content
        except Exception as exc:
            return {"status": "error", "content": [{"text": f"Could not download image: {exc}"}]}

    else:
        return {"status": "error", "content": [{"text": "Provide either file_path or image_url."}]}

    # Detect format
    img_fmt = _detect_format(source_name, detected_mime)
    if img_fmt is None:
        if data[:4] == b"\x89PNG":
            img_fmt = "png"
        elif data[:3] == b"\xff\xd8\xff":
            img_fmt = "jpeg"
        elif data[:6] in (b"GIF87a", b"GIF89a"):
            img_fmt = "gif"
        elif data[:4] == b"RIFF" and data[8:12] == b"WEBP":
            img_fmt = "webp"
        else:
            return {"status": "error", "content": [{"text": f"Unsupported or undetectable image format for: {source_name}"}]}

    # Downsize if needed
    original_size = len(data)
    _safe_limit = _MAX_IMAGE_BYTES - 64 * 1024  # matches _downsize's _SAFE_IMAGE_BYTES
    try:
        data, img_fmt = _downsize(data, img_fmt)
    except Exception as exc:
        return {"status": "error", "content": [{"text": (
            f"Could not process image from {source_name}: {exc}"
        )}]}
    downsized = len(data) < original_size

    # Hard guard: reject if still over the safe limit (belt-and-suspenders)
    if len(data) > _safe_limit:
        return {"status": "error", "content": [{"text": (
            f"Image from {source_name} could not be reduced to under {_safe_limit:,} bytes "
            f"(final size: {len(data):,} bytes). Try a smaller or simpler image."
        )}]}

    # ── describe mode: isolated Vision Agent call (token-efficient) ─────────
    _describe_error = ""
    if mode == "describe":
        print(
            f"[analyze_image] mode=describe  src={source_name}  "
            f"size={len(data):,}B  est_tokens=~{len(data)//400:,} (describe) "
            f"vs ~{len(data)*4//100:,} (full)"
        )
        if _vision_agent is None:
            # No vision agent registered – fall back to full mode with a warning.
            print(
                "[analyze_image] WARNING: no VisionAgent registered; "
                "falling back to mode='full'. Call set_vision_agent() during pipeline init."
            )
        else:
            try:
                # Strands-native ContentBlock format for multimodal input.
                # OllamaModel expects {"image": {"format": ..., "source": {"bytes": <raw bytes>}}}.
                user_message = [
                    {
                        "image": {
                            "format": img_fmt,
                            "source": {"bytes": data},
                        }
                    },
                    {"text": question or "Describe this image in detail."},
                ]
                # Borrow an instance nobody else is mid-call on: one agent is not
                # re-entrant, so concurrent describes need one agent each (see
                # AgentPool). Beyond the pool's size, callers queue here.
                with _ensure_vision_pool().borrow() as _agent:
                    # Wipe history so every invocation is fully independent.
                    _agent.messages.clear()
                    vision_result = str(_agent(user_message))
                print(f"[analyze_image] describe result length: {len(vision_result):,} chars")
                label = source_name if source_name else "provided image"
                return {
                    "status": "success",
                    "content": [
                        {
                            "text": (
                                f"Image analysis for {label}:\n\n{vision_result}"
                            )
                        }
                    ],
                }
            except Exception as exc:
                print(
                    f"[analyze_image] WARNING: VisionAgent call failed ({exc}); "
                    "falling back to mode='full'."
                )
                _describe_error = f"{type(exc).__name__}: {exc}"
                # Fall through to full mode below.

    # ── full mode (or fallback): return bytes in context ─────────────────────
    print(
        f"[analyze_image] mode=full  src={source_name}  "
        f"size={len(data):,}B  est_tokens=~{len(data)*4//100:,}"
    )
    info_parts = [
        f"Image loaded from: {source_name}",
        f"Format: {img_fmt.upper()}, Size: {len(data):,} bytes",
    ]
    if downsized:
        info_parts.append(f"(downsized from {original_size:,} bytes to fit API limits)")
    if question:
        info_parts.append(f"\nUser question: {question}")

    # Handing raw bytes back is only useful if the agent that called this can
    # look at them. When it cannot, the image does not merely go unread: it
    # lands in that agent's history as an image block and every later turn of
    # the conversation is rejected by the API ("Unexpected item type in
    # content."), which reads as the whole conversation breaking for no reason.
    # Return the metadata and say where the description comes from instead.
    _embed = True
    try:
        from src.utils.agentY_server import _orchestrator_supports_vision
        _embed = _orchestrator_supports_vision()
    except Exception:  # noqa: BLE001 — never let this check break the tool
        pass
    if not _embed:
        if _describe_error:
            # The caller DID ask for a description and the vision agent failed —
            # telling it to "call mode='describe'" here is the advice it just
            # followed, and reporting success invites it to invent a description
            # for an image nobody looked at. Say what broke, and say it failed.
            return {"status": "error", "content": [{"text": (
                f"Could not analyse {source_name}: the vision agent call failed "
                f"({_describe_error}).\n\nNo description was produced and this "
                "agent's model cannot read the image itself — do NOT guess or "
                "infer its content. Retry analyze_image for this file, and if it "
                "keeps failing, say so and ask the user how to proceed."
            )}]}
        info_parts.append(
            "\n[The image itself is not shown: this agent's model cannot read "
            "images. Call analyze_image(mode='describe') to have the vision "
            "agent look at it and describe it back in text.]"
        )
        return {"status": "success", "content": [{"text": "\n".join(info_parts)}]}

    return {
        "status": "success",
        "content": [
            {"text": "\n".join(info_parts)},
            {
                "image": {
                    "format": img_fmt,
                    "source": {"bytes": data},
                }
            },
        ],
    }
