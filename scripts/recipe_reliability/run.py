"""Recipe reliability harness (Phase 1).

Drives local workflow-recipe intents through the real researcher->brain->execute
pipeline headless, captures the ComfyUI execution outcome, and classifies each as
pass / agent build failure / ComfyUI-or-model error / missing model / timeout.
Writes an incremental report so progress survives a hang or crash.

Run:
    python -m scripts.recipe_reliability.run --task "Image Edit" --limit 1
    python -m scripts.recipe_reliability.run --only image_edit__qwen_image
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
import urllib.request

from dotenv import load_dotenv

_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _root not in sys.path:
    sys.path.insert(0, _root)
load_dotenv(os.path.join(_root, ".env"))

# The pipeline's verbose logging prints unicode (e.g. "->" arrows); make stdout
# UTF-8 so a cp1252 console does not crash the run with UnicodeEncodeError.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    except Exception:
        pass

from src.pipeline import create_pipeline               # noqa: E402
from agenty_core.tools.comfyui import clear_tool_caches  # noqa: E402

_DB = os.path.join(_root, "config", "workflow_recipes.json")
_REPORT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "report.json")
_COMFY = "http://127.0.0.1:8188"

# ComfyUI's input directory (LoadImage resolves filenames relative to here).
_INPUT_DIR = "W:/0193_Never_Stop_Dreaming_Spec/02_build/comfy/sebastian.zilius/input"

_IMAGE_INPUT_TASKS = {
    "image_edit", "controlnet", "inpaint", "outpaint", "upscale",
    "style_transfer", "image_to_video", "video_edit", "first_last_frame_to_video",
}

_TEST_COLORS = [(200, 60, 60), (60, 160, 80), (70, 90, 200),
                (210, 170, 40), (150, 70, 170), (60, 180, 190)]

# Keep test inputs small so edit/i2v workflows that derive their working
# resolution from the input size do not blow up GPU VRAM. This is for
# reliability testing only - real users supply full-res images. Reduced to 384
# after repeated machine power-loss crashes under heavy GPU load (RTX 5090):
# smaller inputs shorten the high-draw window for edit/i2v/upscale workflows.
_TEST_IMG_SIZE = 384


def _ensure_test_images(n: int, size: int = _TEST_IMG_SIZE) -> list[str]:
    """Synthesize n visually-distinct, low-resolution test images in ComfyUI's
    input dir and return their paths. Controllable PIL images (distinct colour +
    shape + label); always regenerated so a changed size takes effect."""
    from PIL import Image, ImageDraw  # noqa: PLC0415
    os.makedirs(_INPUT_DIR, exist_ok=True)
    s = size
    for i in range(max(1, n)):
        p = os.path.join(_INPUT_DIR, f"recipe_test_input_{i + 1}.png")
        base = _TEST_COLORS[i % len(_TEST_COLORS)]
        img = Image.new("RGB", (s, s), base)
        dr = ImageDraw.Draw(img)
        dr.ellipse([s * 0.23, s * 0.23, s * 0.77, s * 0.77],
                   fill=tuple(min(255, c + 70) for c in base))
        dr.rectangle([s * 0.16 + i * 20, s * 0.16, s * 0.84, s * 0.84],
                     outline=(255, 255, 255), width=max(4, s // 76))
        dr.text((s * 0.09, s * 0.08), f"TEST {i + 1}", fill=(255, 255, 255))
        img.save(p)
    return [os.path.join(_INPUT_DIR, f"recipe_test_input_{i + 1}.png").replace("\\", "/")
            for i in range(max(1, n))]


def _image_count(recipe: dict) -> int:
    """How many IMAGE inputs this recipe's type exposes (>=1 for image tasks)."""
    imgs = [p for p in recipe["boundary_ports"].get("inputs", []) if p.get("data_type") == "IMAGE"]
    return max(1, len(imgs))


def _load_local_recipes(task: str | None, only: str | None, limit: int | None,
                        exclude: set[str] | None = None, include: set[str] | None = None):
    db = json.load(open(_DB, encoding="utf-8"))
    exclude = exclude or set()
    include = include or set()
    out = []
    for t in db["tasks"]:
        if task and t["task"] != task:
            continue
        for m in t["models"]:
            if m["execution"] != "local":
                continue
            if only and m["id"] != only:
                continue
            if include and m["id"] not in include:
                continue
            if m["id"] in exclude:
                continue
            out.append(m)
    return out[:limit] if limit else out


def _build_intent(recipe: dict, pool: list[str]) -> tuple[str, int]:
    """Return (intent_text, n_images). For image-input recipes, embed as many
    distinct input image paths as the recipe's type exposes (capped to the pool)."""
    ui = recipe["user_intent"]
    task, model = ui.get("task"), recipe["model"]
    needs_image = (task in _IMAGE_INPUT_TASKS) or any(
        p.get("data_type") in ("IMAGE", "VIDEO") for p in recipe["boundary_ports"].get("inputs", [])
    )
    n = min(_image_count(recipe), len(pool)) if needs_image else 0
    images = pool[:n]
    # Task-appropriate phrasing. Video tasks MUST NOT get image-edit "blend"
    # phrasing or the Researcher mis-classifies them as image edit.
    phrasings = {
        "text_to_image": f"Generate an image of a serene mountain lake at sunrise using {model}.",
        "image_edit": f"Edit this image: turn it into a vibrant watercolor painting. Use {model}.",
        "image_edit_with_controlnet": f"Generate an image guided by this control image using {model}.",
        "controlnet": f"Generate an image guided by this control image using {model}.",
        "inpaint_outpaint": f"Inpaint this image, filling the masked area naturally, using {model}.",
        "inpaint": f"Inpaint this image, filling the masked area naturally, using {model}.",
        "upscale": f"Upscale this image using {model}.",
        "text_to_video": f"Generate a short video of a serene mountain lake at sunrise using {model}.",
        "image_to_video": f"Generate a short video by animating this image using {model}.",
        "first_last_frame_to_video": (
            f"Generate a short video that starts on the first frame and ends on the "
            f"last frame, interpolating between them, using {model}."),
        "video_to_video": f"Transform this video using {model}.",
        "video_inpaint": f"Inpaint the masked region of this video using {model}.",
        "character": f"Generate a short video of this character using {model}.",
    }
    phrasing = phrasings.get(task)
    if phrasing is None:
        if n >= 2:
            phrasing = (f"Edit using these {n} reference images: blend them into a single "
                        f"cohesive composition. Use {model}.")
        else:
            phrasing = ui.get("example_requests", [f"Build a workflow using {model}."])[0]
    # Reliability testing on an RTX 5090 that hard-crashed twice under sustained
    # GPU load: nudge a modest output size and few sampling steps to shorten the
    # high-draw window (heavy models like Qwen-Image are the main offenders).
    phrasing += (" Keep this test light on the GPU: use a small, sub-HD output "
                 "resolution (512x512 for images, 480p for video) and few sampling steps.")
    if images:
        label = "Input image:" if n == 1 else "Input images:"
        phrasing += " " + label + " " + " ".join(f'"{p}"' for p in images)
    return phrasing, n


def _comfy_history_ids() -> set:
    try:
        h = json.load(urllib.request.urlopen(f"{_COMFY}/history?max_items=50", timeout=5))
        return set(h.keys())
    except Exception:
        return set()


def _comfy_outcome(new_ids: set) -> dict:
    """Inspect ComfyUI history for the prompts run during this recipe."""
    try:
        h = json.load(urllib.request.urlopen(f"{_COMFY}/history?max_items=50", timeout=5))
    except Exception as e:  # noqa: BLE001
        return {"submitted": False, "error": f"history fetch failed: {e}"}
    statuses, errors = [], []
    for pid in new_ids:
        entry = h.get(pid, {})
        st = entry.get("status", {})
        statuses.append(st.get("status_str"))
        for mtype, mdata in st.get("messages", []):
            if "error" in mtype:
                errors.append({
                    "node_type": mdata.get("node_type"),
                    "exception_type": mdata.get("exception_type"),
                    "exception_message": (mdata.get("exception_message") or "")[:300],
                })
    return {"submitted": bool(new_ids), "statuses": statuses, "errors": errors}


async def _drive(pipeline, user_input: str, timeout: float):
    qa_q: asyncio.Queue = asyncio.Queue()
    seen: list[str] = []
    parts: list[str] = []
    in_researcher = False

    async def _consume():
        nonlocal in_researcher
        async for event in pipeline.stream_async(user_input, qa_reply_queue=qa_q):
            if not isinstance(event, dict):
                continue
            if event.get("brain_assembly_fail_ask"):
                seen.append("brain_assembly_fail"); await qa_q.put(""); continue
            if event.get("qa_fail_ask"):
                seen.append("qa_fail"); await qa_q.put("n"); continue
            if event.get("approval_ask"):
                seen.append("approval"); await qa_q.put("y"); continue
            if event.get("_researcher_start"):
                in_researcher = True; continue
            if event.get("_researcher_done"):
                in_researcher = False; continue
            data = event.get("data")
            if data and not in_researcher:
                parts.append(data)
        await pipeline._await_pending_compression()

    await asyncio.wait_for(_consume(), timeout=timeout)
    return seen, "".join(parts)


def _classify(seen, response, comfy):
    text = (response or "").lower()
    if comfy.get("submitted") and comfy.get("statuses"):
        if any(s == "success" for s in comfy["statuses"]) and not comfy.get("errors"):
            return "pass"
        errs = comfy.get("errors") or []
        # Distinguish environment/resource failures from agent-build failures:
        # if the graph reached execution and only failed on VRAM or a missing
        # model file, that is not the agent's fault.
        blob = " ".join(
            f"{e.get('exception_type', '')} {e.get('exception_message', '')}".lower()
            for e in errs
        )
        if any(p in blob for p in (
            "outofmemory", "out of memory", "allocation on device",
            "not enough memory", "defaultcpuallocator", "alloc_cpu", "cannot allocate",
        )):
            return "resource_oom"
        if ("not in list" in blob or "not found" in blob or "no such file" in blob
                or "does not exist" in blob or "value not in" in blob):
            return "missing_model"
        if errs:
            return "comfyui_exec_error"
    # Missing-model blockers (incl. download-disabled) are environment, not agent.
    if any(p in text for p in (
        "not installed", "not available", "download is disabled", "missing model",
        "unavailable", "empty model list",
    )) and "template" not in text:
        return "missing_model"
    if "not found" in text and ("model" in text or "check_model" in text):
        return "missing_model"
    if "brain_assembly_fail" in seen:
        return "agent_build_fail"
    if "template" in text and "not found" in text:
        return "agent_build_fail"
    return "no_execution"  # built nothing / unclear - inspect logs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default=None, help="canonical task to filter (e.g. 'Image Edit')")
    ap.add_argument("--only", default=None, help="single recipe id")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--timeout", type=float, default=900.0, help="per-recipe seconds")
    ap.add_argument("--exclude", default="", help="comma-separated recipe ids to skip")
    ap.add_argument("--include", default="", help="comma-separated recipe ids allowlist")
    ap.add_argument("--no-downloads", action="store_true",
                    help="disable HF model downloads (missing models fail fast); "
                         "default lets the agent download missing models")
    ap.add_argument("--build", action="store_true",
                    help="build-from-scratch mode: disable templates so the agent "
                         "assembles each workflow node-by-node from the recipe")
    ap.add_argument("--max-dim", type=int, default=768,
                    help="hard cap (px) on generated latent width/height, applied in "
                         "apply_brainbriefing (sub-HD by default to protect a "
                         "power-limited GPU)")
    args = ap.parse_args()

    # By default the researcher may download missing models via its HF tools.
    # --no-downloads fails fast to a missing-model blocker instead.
    if args.no_downloads:
        os.environ["AGENTY_DISABLE_DOWNLOADS"] = "1"
    # Build-from-scratch: gate template loading so get_workflow_catalog is empty
    # and get_workflow_template returns an empty canvas (see agenty_core comfyui).
    if args.build:
        os.environ["AGENTY_FORCE_BUILD"] = "1"
    # Hard-cap generated latent dimensions sub-HD for every recipe (apply_brainbriefing
    # clamps width/height to this). Guards a power-limited GPU from HD+ latents that
    # the researcher or a template default would otherwise request. Set explicitly
    # (not setdefault) so the CLI value always wins; also export it as a shell prefix
    # at launch so subprocess-executed tools inherit it at spawn time.
    os.environ["AGENTY_MAX_DIM"] = str(args.max_dim)

    exclude = {s.strip() for s in args.exclude.split(",") if s.strip()}
    include = {s.strip() for s in args.include.split(",") if s.strip()}
    recipes = _load_local_recipes(args.task, args.only, args.limit, exclude, include)
    print(f"[harness] {len(recipes)} local recipe(s) to test")
    # Provision a pool of distinct test images sized to the largest recipe need.
    max_imgs = max((_image_count(r) for r in recipes), default=1)
    pool = _ensure_test_images(max_imgs)
    print(f"[harness] test image pool ({len(pool)}): {[os.path.basename(p) for p in pool]}")
    results = []
    for i, recipe in enumerate(recipes, 1):
        rid = recipe["id"]
        intent, n_images = _build_intent(recipe, pool)
        if args.build:
            intent += (" Build this workflow from scratch by assembling the nodes "
                       "yourself to the recipe standard; do not reuse an existing "
                       "workflow template.")
        print(f"\n{'='*70}\n[harness] ({i}/{len(recipes)}) {rid}\n  intent: {intent}\n{'='*70}")
        # Full isolation: a fresh pipeline + unique session per recipe so no
        # session state (input images, chat history, memory) leaks into the next
        # recipe and confuses triage/researcher.
        pipeline = create_pipeline(session_id=f"recipe-reliability-{rid}", verbose=True)
        clear_tool_caches()
        before = _comfy_history_ids()
        t0 = time.time()
        timed_out = False
        try:
            seen, response = asyncio.run(_drive(pipeline, intent, args.timeout))
        except asyncio.TimeoutError:
            seen, response, timed_out = ["timeout"], "", True
        except Exception as e:  # noqa: BLE001
            seen, response = [f"exception:{e}"], str(e)
        dur = round(time.time() - t0, 1)
        new_ids = _comfy_history_ids() - before
        comfy = _comfy_outcome(new_ids)
        # Trust ComfyUI's outcome even on timeout: with a slow local LLM the
        # workflow can build + execute successfully and only the post-execution
        # steps (vision QA / memory) overrun the clock. Only fall back to
        # "timeout" when ComfyUI gave no conclusive result.
        outcome = _classify(seen, response, comfy)
        if timed_out and outcome == "no_execution":
            outcome = "timeout"
        rec = {
            "id": rid, "intent": intent, "n_images": n_images,
            "duration_s": dur, "events": seen, "comfyui": comfy,
            "outcome": outcome, "response_tail": (response or "")[-600:],
        }
        results.append(rec)
        print(f"[harness] -> {rid}: {outcome}  ({dur}s)  comfyui={comfy.get('statuses')} errors={len(comfy.get('errors', []))}")
        json.dump({"results": results}, open(_REPORT, "w", encoding="utf-8"), indent=2)

    print(f"\n[harness] done. report -> {_REPORT}")
    from collections import Counter
    print("[harness] outcomes:", dict(Counter(r["outcome"] for r in results)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
