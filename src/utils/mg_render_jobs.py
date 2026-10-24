"""Motion-graphics renders the bEpic viewer's Motion panel asked for: one background job each, polled by id.

The panel's Render button posts a scene file; the viewer runs inside ComfyUI's Python (no ``agenty_core``, no
Node), so ComfyUI forwards the request to this host, which runs the shared core tool
(:func:`agenty_core.tools.motion_graphics.mg_render`, the same one the agent calls). A render takes seconds to
minutes, so the request answers at once with an ``id`` and the page polls ``GET /agentY/mg_render?id=``.

The finished video is handed to ``deliver`` (the host's out-of-turn delivery: notification, the viewer's agentY tab).
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
_FORMATS = ("mp4", "webm", "mov", "gif")


def _clean(body) -> dict:
    if not isinstance(body, dict):
        raise ValueError("bad request")
    path = str(body.get("scene_path") or "").strip()
    if not path.lower().endswith(".mgscene.json"):
        raise ValueError("scene_path must be a *.mgscene.json file")
    fmt = str(body.get("format") or "mp4").lower()
    if fmt not in _FORMATS:
        raise ValueError("format is mp4, webm, mov or gif")
    try:
        scale = min(1.0, max(0.1, float(body.get("scale") or 1.0)))
    except (TypeError, ValueError):
        raise ValueError("scale must be a number")
    return {"scene_path": path, "format": fmt, "scale": scale}


class MgRenderJobs:
    def __init__(self, deliver: Optional[Callable[[str], None]] = None, run: Optional[Callable[..., str]] = None):
        self._deliver = deliver
        self._run = run
        self._jobs: dict[str, dict] = {}
        self._lock = threading.Lock()

    def _render(self, **kw) -> str:
        if self._run is not None:
            return self._run(**kw)
        from agenty_core.tools.motion_graphics import mg_render
        return mg_render(**kw)

    def start(self, body) -> dict:
        args = _clean(body)
        job = {"id": uuid.uuid4().hex[:12], "state": "running", "scene": args["scene_path"], "created": time.time(),
               "message": "", "path": None, "result": None}
        with self._lock:
            self._jobs[job["id"]] = job
            while len(self._jobs) > MAX_JOBS:
                self._jobs.pop(next(iter(self._jobs)))
        threading.Thread(target=self._work, args=(job, args), name=f"mg-render-{job['id']}", daemon=True).start()
        return self.public(job)

    def _work(self, job: dict, args: dict) -> None:
        from agenty_core.utils import turn_scope
        token = turn_scope.enter(turn_scope.Scope())
        try:
            raw = self._render(**args)
            try:
                res = json.loads(raw)
            except (TypeError, ValueError):
                res = {"error": str(raw)[:300]}
            job["result"] = res
            if res.get("error"):
                job.update(state="error", message=str(res["error"]))
            else:
                job.update(state="done", path=res.get("path"))
                if self._deliver is not None and res.get("path"):
                    try:
                        self._deliver(res["path"])
                    except Exception:  # noqa: BLE001 - delivery never costs the render
                        logger.warning("could not deliver the motion render", exc_info=True)
        except Exception as exc:  # noqa: BLE001
            logger.exception("motion render job failed")
            job.update(state="error", message=f"{type(exc).__name__}: {exc}")
        finally:
            turn_scope.leave(token)

    def get(self, job_id: str) -> Optional[dict]:
        with self._lock:
            return self._jobs.get(job_id)

    def public(self, job: dict) -> dict:
        out = {k: job.get(k) for k in ("id", "state", "scene", "message", "path")}
        out["error"] = job.get("message") if job.get("state") == "error" else None
        out["elapsed"] = round(time.time() - job["created"], 1)
        res = job.get("result") or {}
        for k in ("frames", "width", "height", "fps", "bytes", "seconds"):
            if k in res:
                out[k] = res[k]
        return out
