"""Atlas camera moves the viewer asked for: one background job each, polled by id.

The bEpic viewer's "Render camera with Atlas" button posts the previz camera's animation (sampled from the
timeline). The viewer runs inside ComfyUI's Python (no ``agenty_core``, no World Labs key), so ComfyUI relays
the request here. The work is the shared core function
(:func:`agenty_core.tools.atlas_camera_move.render_move`), the one the agent's ``atlas_render_camera_move``
tool runs. A move takes minutes (several Atlas tasks, then the viewer renders the frames), so the request
answers at once with an ``id`` and the page polls ``GET /agentY/atlas_camera_move?id=`` for ``tasks_done`` /
``tasks_total``.

``dry_run`` is answered in the request (it only plans and prices; nothing is sent to World Labs). The real run
needs ``confirm_cost`` - the viewer sends it only after the user pressed the button on the price it showed.
The viewer itself draws the finished clip (a new tab, and a loader node on the canvas): nothing is delivered
from here.
"""
from __future__ import annotations

import json
import logging
import threading
import time
import uuid
from typing import Callable, Optional

logger = logging.getLogger(__name__)

MAX_JOBS = 20
MAX_KEYS = 800
QUALITIES = ("draft", "standard", "high")


def _nums(v, n):
    if not isinstance(v, (list, tuple)) or len(v) != n:
        return None
    try:
        return [float(x) for x in v]
    except (TypeError, ValueError):
        return None


def _clean(body) -> dict:
    if not isinstance(body, dict):
        raise ValueError("bad request")
    world = str(body.get("world") or "").strip()
    if not world:
        raise ValueError("name the world (its folder or world.glb)")
    raw = body.get("keys")
    if not isinstance(raw, list) or len(raw) < 2:
        raise ValueError("send the camera keys (at least two)")
    if len(raw) > MAX_KEYS:
        raise ValueError(f"at most {MAX_KEYS} camera keys")
    mw = body.get("model_world")
    if mw is not None and _nums(mw, 16) is None:
        raise ValueError("model_world must be 16 numbers")
    keys = []
    for k in raw:
        k = k if isinstance(k, dict) else {}
        cw = _nums(k.get("camera_world"), 16)
        if cw is None or k.get("t") is None:
            raise ValueError("each key needs t and camera_world (16 numbers)")
        v = {"camera_world": cw, "gated": bool(k.get("gated", True))}
        if mw is not None:
            v["model_world"] = [float(x) for x in mw]
        for a in ("fov", "aspect"):
            if k.get(a) is not None:
                v[a] = float(k[a])
        keys.append({"t": float(k["t"]), "viewer": v})
    quality = str(body.get("quality") or "standard")
    if quality not in QUALITIES:
        raise ValueError("quality is draft, standard or high")
    out = {"name": str(body.get("name") or "move"), "world": world, "keys": keys, "quality": quality,
           "fps": float(body.get("fps") or 24.0), "prompt": str(body.get("prompt") or ""),
           "ease": str(body.get("ease") or "linear"), "confirm_cost": bool(body.get("confirm_cost")),
           "dry_run": bool(body.get("dry_run")), "add_to_canvas": body.get("add_to_canvas", True) is not False,
           "grow_world": bool(body.get("grow_world"))}
    if body.get("seconds"):
        out["seconds"] = float(body["seconds"])
    return out


class AtlasMoveJobs:
    def __init__(self, run: Optional[Callable[..., str]] = None):
        self._run = run
        self._jobs: dict[str, dict] = {}
        self._lock = threading.Lock()

    def _render(self, **kw) -> str:
        if self._run is not None:
            return self._run(**kw)
        from agenty_core.tools.atlas_camera_move import render_move
        return render_move(**kw)

    def start(self, body) -> dict:
        args = _clean(body)
        if args["dry_run"]:
            try:
                res = json.loads(self._render(**args))
            except (TypeError, ValueError):
                res = {"error": "no answer"}
            return {"dry_run": True, **res}
        job = {"id": uuid.uuid4().hex[:12], "state": "running", "phase": "Planning", "world": args["world"],
               "created": time.time(), "message": "", "path": None, "result": None, "tasks_done": 0, "tasks_total": 0}
        with self._lock:
            self._jobs[job["id"]] = job
            while len(self._jobs) > MAX_JOBS:
                self._jobs.pop(next(iter(self._jobs)))
        threading.Thread(target=self._work, args=(job, args), name=f"atlas-move-{job['id']}", daemon=True).start()
        return self.public(job)

    def _work(self, job: dict, args: dict) -> None:
        def progress(p: dict) -> None:
            job.update(phase=p.get("phase", job["phase"]), tasks_done=p.get("tasks_done", job["tasks_done"]),
                       tasks_total=p.get("tasks_total", job["tasks_total"]))
        try:
            raw = self._render(**args, on_progress=progress)
            try:
                res = json.loads(raw)
            except (TypeError, ValueError):
                res = {"error": str(raw)[:300]}
            job["result"] = res
            if res.get("error"):
                job.update(state="error", message=str(res["error"]), error_kind=res.get("error_kind"),
                           operation_id=res.get("operation_id"))
            else:
                job.update(state="done", path=res.get("path"), tasks_done=res.get("tasks_used", job["tasks_done"]),
                           message="; ".join(res.get("render_warnings") or []))
        except Exception as exc:  # noqa: BLE001
            logger.exception("atlas move job failed")
            job.update(state="error", message=f"{type(exc).__name__}: {exc}")

    def get(self, job_id: str) -> Optional[dict]:
        with self._lock:
            return self._jobs.get(job_id)

    def public(self, job: dict) -> dict:
        out = {k: job.get(k) for k in ("id", "state", "phase", "world", "message", "path", "error_kind",
                                       "operation_id", "tasks_done", "tasks_total")}
        out["elapsed"] = round(time.time() - job["created"], 1)
        res = job.get("result") or {}
        for k in ("frames", "fps", "seconds", "tasks_used", "estimate", "world_dir", "operations"):
            if k in res:
                out[k] = res[k]
        return out
