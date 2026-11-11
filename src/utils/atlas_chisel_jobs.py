"""Atlas Chisel pictures the viewer asked for: one background job each, polled by id.

The ComfyUI Image Viewer's "Chisel this view" dialog (3D space) posts the camera it is looking through, a
prompt and, optionally, a picture of that view with a painted mask and a style picture. The viewer runs
inside ComfyUI's Python (no ``agenty_core``, no World Labs key), so ComfyUI forwards the request to this
host. The picture is made by the core function the agent's ``atlas_chisel`` tool runs, with the world's
nearest views as context, and handed to ``deliver`` (the host's out-of-turn delivery, as for Atlas renders)
in a turn scope of its own, so it cannot land in whichever conversation is running. No agent turn.

The pictures arrive as absolute paths the viewer's route already checked (inside the folders the viewer
serves); this side checks again that each one is a file.
"""
from __future__ import annotations

import json
import logging
import os
import threading
import time
import uuid
from typing import Callable, Optional

logger = logging.getLogger(__name__)

MAX_JOBS = 40


def _clean(body) -> dict:
    if not isinstance(body, dict):
        raise ValueError("bad request")
    world = str(body.get("world") or "").strip()
    if not world:
        raise ValueError("name the world (its folder or world.glb)")
    camera = body.get("camera")
    if not isinstance(camera, dict) or not isinstance(camera.get("viewer"), dict):
        raise ValueError("send the viewer's camera")
    prompt = str(body.get("prompt") or "").strip()[:2000]
    if not prompt:
        raise ValueError("say what Atlas should paint or change")
    paths = {}
    for k in ("image_path", "mask_path", "style_image"):
        p = str(body.get(k) or "").strip()
        if p and not os.path.isfile(p):
            raise ValueError(f"no such picture: {os.path.basename(p)}")
        paths[k] = p
    if paths["mask_path"] and not paths["image_path"]:
        raise ValueError("a mask needs the picture it was painted on")
    try:
        seed = int(body.get("seed") if body.get("seed") not in (None, "") else -1)
    except (TypeError, ValueError):
        raise ValueError("seed must be a whole number")
    return {"world": world, "camera": {"viewer": dict(camera["viewer"])}, "prompt": prompt, **paths,
            "context": None if body.get("context", True) is not False else [],
            "seed": seed if seed >= 0 else -1, "enhance_prompt": body.get("enhance_prompt", True) is not False}


class AtlasChiselJobs:
    def __init__(self, deliver: Optional[Callable[[str], None]] = None, run: Optional[Callable[..., str]] = None):
        self._deliver = deliver
        self._run = run
        self._jobs: dict[str, dict] = {}
        self._lock = threading.Lock()

    def _chisel(self, **kw) -> str:
        if self._run is not None:
            return self._run(**kw)
        from agenty_core.tools.atlas_chisel import atlas_chisel
        return atlas_chisel(**kw)

    def start(self, body) -> dict:
        args = _clean(body)
        mode = "edit" if args["mask_path"] else ("restyle" if args["image_path"] else "paint")
        phase = {"edit": "Atlas is changing the masked part", "restyle": "Atlas is repainting the picture",
                 "paint": "Atlas is painting the view"}[mode]
        job = {"id": uuid.uuid4().hex[:12], "state": "running", "phase": phase, "mode": mode,
               "world": args["world"], "created": time.time(), "message": "", "path": None, "result": None}
        with self._lock:
            self._jobs[job["id"]] = job
            while len(self._jobs) > MAX_JOBS:
                self._jobs.pop(next(iter(self._jobs)))
        threading.Thread(target=self._work, args=(job, args), name=f"atlas-chisel-{job['id']}", daemon=True).start()
        return self.public(job)

    def _work(self, job: dict, args: dict) -> None:
        from agenty_core.utils import turn_scope
        scope = turn_scope.Scope()
        scope.set("image_io.output_sink", self._deliver if self._deliver is not None else (lambda *_a, **_k: None))
        token = turn_scope.enter(scope)
        try:
            raw = self._chisel(**args)
            try:
                res = json.loads(raw)
            except (TypeError, ValueError):
                res = {"error": str(raw)[:300]}
            job["result"] = res
            if res.get("error"):
                job.update(state="error", message=str(res["error"]), error_kind=res.get("error_kind"),
                           operation_id=res.get("operation_id"))
            else:
                job.update(state="done", path=res.get("path"))
        except Exception as exc:  # noqa: BLE001
            logger.exception("atlas chisel job failed")
            job.update(state="error", message=f"{type(exc).__name__}: {exc}")
        finally:
            turn_scope.leave(token)

    def get(self, job_id: str) -> Optional[dict]:
        with self._lock:
            return self._jobs.get(job_id)

    def public(self, job: dict) -> dict:
        out = {k: job.get(k) for k in ("id", "state", "phase", "mode", "world", "message", "path",
                                       "error_kind", "operation_id")}
        out["elapsed"] = round(time.time() - job["created"], 1)
        res = job.get("result") or {}
        if res.get("prompt_used"):
            out["prompt_used"] = res["prompt_used"]
        return out
