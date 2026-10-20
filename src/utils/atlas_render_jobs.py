"""Atlas renders the viewer asked for: one background job each, polled by id.

The bEpic viewer's "Render with Atlas" button posts the camera it is looking through; the viewer runs
inside ComfyUI's Python (no ``agenty_core``, no World Labs key), so ComfyUI forwards the request to this
host, which has both. The render itself is the shared core function
(:func:`agenty_core.tools.atlas_render.render_view`), the same one the agent's ``atlas_render_view``
tool runs. A render takes tens of seconds to minutes, so the request answers at once with an ``id`` and
the page polls ``GET /agentY/atlas_render?id=``.

The finished picture is handed to ``deliver`` (the host's out-of-turn delivery: a loader node on the
canvas, a notification, the viewer's agentY tab). It runs in its own turn scope so it cannot land in
whichever conversation happens to be running.
"""
from __future__ import annotations

import json
import logging
import threading
import time
import uuid
from typing import Callable, Optional

logger = logging.getLogger(__name__)

MAX_JOBS = 40
_UPSCALES = (0, 2, 4)


def _clean(body) -> dict:
    if not isinstance(body, dict):
        raise ValueError("bad request")
    world = str(body.get("world") or "").strip()
    camera = body.get("camera")
    if not world:
        raise ValueError("name the world (its folder or world.glb)")
    if not isinstance(camera, dict) or not camera:
        raise ValueError("send the camera")
    try:
        upscale = int(body.get("upscale") or 0)
    except (TypeError, ValueError):
        raise ValueError("upscale is 0, 2 or 4")
    if upscale not in _UPSCALES:
        raise ValueError("upscale is 0, 2 or 4")
    return {"world": world, "camera": camera, "upscale": upscale,
            "add_to_world": bool(body.get("add_to_world")), "depth": bool(body.get("depth")),
            "prompt": str(body.get("prompt") or ""), "upscale_model": str(body.get("upscale_model") or "")}


class AtlasRenderJobs:
    def __init__(self, deliver: Optional[Callable[[str], None]] = None, run: Optional[Callable[..., str]] = None):
        self._deliver = deliver
        self._run = run
        self._jobs: dict[str, dict] = {}
        self._lock = threading.Lock()

    def _render(self, **kw) -> str:
        if self._run is not None:
            return self._run(**kw)
        from agenty_core.tools.atlas_render import render_view
        return render_view(**kw)

    def start(self, body) -> dict:
        args = _clean(body)
        job = {"id": uuid.uuid4().hex[:12], "state": "running", "phase": "Atlas is painting the view",
               "world": args["world"], "created": time.time(), "message": "", "path": None, "original": None,
               "result": None}
        with self._lock:
            self._jobs[job["id"]] = job
            while len(self._jobs) > MAX_JOBS:
                self._jobs.pop(next(iter(self._jobs)))
        threading.Thread(target=self._work, args=(job, args), name=f"atlas-render-{job['id']}", daemon=True).start()
        return self.public(job)

    def _work(self, job: dict, args: dict) -> None:
        from agenty_core.utils import turn_scope
        scope = turn_scope.Scope()
        if self._deliver is not None:
            scope.set("image_io.output_sink", self._deliver)
        token = turn_scope.enter(scope)
        try:
            raw = self._render(**args)
            try:
                res = json.loads(raw)
            except (TypeError, ValueError):
                res = {"error": str(raw)[:300]}
            job["result"] = res
            if res.get("error"):
                job.update(state="error", message=str(res["error"]), error_kind=res.get("error_kind"),
                           operation_id=res.get("operation_id"))
            else:
                job.update(state="done", path=res.get("path"), original=res.get("original"),
                           message=res.get("upscale_error") or "")
        except Exception as exc:  # noqa: BLE001
            logger.exception("atlas render job failed")
            job.update(state="error", message=f"{type(exc).__name__}: {exc}")
        finally:
            turn_scope.leave(token)

    def get(self, job_id: str) -> Optional[dict]:
        with self._lock:
            return self._jobs.get(job_id)

    def public(self, job: dict) -> dict:
        out = {k: job.get(k) for k in ("id", "state", "phase", "world", "message", "path", "original",
                                       "error_kind", "operation_id")}
        out["elapsed"] = round(time.time() - job["created"], 1)
        res = job.get("result") or {}
        for k in ("camera_mesh_frame", "fov_deg", "size", "upscale", "added_to_world"):
            if k in res:
                out[k] = res[k]
        return out
