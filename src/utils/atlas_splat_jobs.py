"""Gaussian splats of Atlas worlds that the viewer asked for: one background job each, polled by id.

The ComfyUI Image Viewer's Atlas menu has "Export Gaussian splat…": World Labs builds a splat from the open
world's posed views (``posedImages2Splats``). The viewer runs inside ComfyUI's Python (no ``agenty_core``, no
World Labs key), so ComfyUI relays the request here. The work is the core function the agent's
``atlas_views_to_splat`` tool runs.

``dry_run`` is answered in the request (the views it would send, the consistency check and the price; nothing
is sent to World Labs). The real run needs ``confirm_cost`` - the viewer sends it only after the user pressed
the button on the price it showed. It takes 10-20 minutes, so it answers at once with an ``id``. The splat is
NOT opened in the viewer from here: the viewer puts the PLY into its History and stays in the 3D space.
"""
from __future__ import annotations

import json
import logging
import os
import re
import threading
import time
import uuid
from typing import Callable, Optional

logger = logging.getLogger(__name__)

MAX_JOBS = 20


def _clean(body) -> dict:
    if not isinstance(body, dict):
        raise ValueError("bad request")
    world = str(body.get("world") or "").strip()
    if not world:
        raise ValueError("name the world (its folder or world.glb)")
    if os.path.isabs(world) and not os.path.exists(world):
        raise ValueError("no such Atlas world")
    name = re.sub(r"[^A-Za-z0-9_\-]+", "_", str(body.get("name") or "")).strip("_")[:60]
    return {"world": world, "name": name, "refine": body.get("refine", True) is not False,
            "check_consistency": body.get("check_consistency", True) is not False,
            "dry_run": bool(body.get("dry_run")), "confirm_cost": bool(body.get("confirm_cost"))}


class AtlasSplatJobs:
    def __init__(self, run: Optional[Callable[..., str]] = None):
        self._run_fn = run
        self._jobs: dict[str, dict] = {}
        self._lock = threading.Lock()

    def _run(self, **kw) -> str:
        if self._run_fn is not None:
            return self._run_fn(**kw)
        from agenty_core.tools.atlas_splat import atlas_views_to_splat
        return atlas_views_to_splat(**kw)

    def _call(self, args: dict, **extra) -> dict:
        raw = self._run(world=args["world"], name=args["name"], refine=args["refine"],
                        check_consistency=args["check_consistency"], open_in_viewer=False, **extra)
        try:
            out = json.loads(raw)
        except (TypeError, ValueError):
            out = {"error": str(raw)[:300]}
        return out if isinstance(out, dict) else {"error": str(out)[:300]}

    def start(self, body) -> dict:
        args = _clean(body)
        if args["dry_run"]:
            out = self._call(args, dry_run=True)
            if out.get("error"):
                raise ValueError(str(out["error"]))
            return {"state": "dry_run", "plan": out.get("plan"), "note": out.get("note", "")}
        if not args["confirm_cost"]:
            raise ValueError("a splat costs World Labs credits: ask for the price first (dry_run), then send "
                             "confirm_cost")
        with self._lock:
            if any(j["state"] == "running" and j["world"] == args["world"] for j in self._jobs.values()):
                raise ValueError("a splat of this world is already being made")
        job = {"id": uuid.uuid4().hex[:12], "state": "running", "phase": "World Labs is building the splat",
               "world": args["world"], "created": time.time(), "message": "", "result": None}
        with self._lock:
            self._jobs[job["id"]] = job
            while len(self._jobs) > MAX_JOBS:
                self._jobs.pop(next(iter(self._jobs)))
        threading.Thread(target=self._work, args=(job, args), name=f"atlas-splat-{job['id']}", daemon=True).start()
        return self.public(job)

    def _work(self, job: dict, args: dict) -> None:
        from agenty_core.utils import turn_scope
        scope = turn_scope.Scope()
        # The splat's files must not land in whichever conversation is running; the viewer takes the PLY itself.
        scope.set("image_io.output_sink", lambda *_a, **_k: None)
        token = turn_scope.enter(scope)
        try:
            res = self._call(args, confirm_cost=True, timeout_s=2400)
            job["result"] = res
            if res.get("error"):
                job.update(state="error", message=str(res["error"]), error_kind=res.get("error_kind"),
                           operation_id=res.get("operation_id"))
                return
            if res.get("ply_error"):
                job["message"] = f"the SPZ is made, but no PLY for the viewer: {res['ply_error']}"
            job["state"] = "done"
        except Exception as exc:  # noqa: BLE001
            logger.exception("atlas splat job failed")
            job.update(state="error", message=f"{type(exc).__name__}: {exc}")
        finally:
            turn_scope.leave(token)

    def get(self, job_id: str) -> Optional[dict]:
        with self._lock:
            return self._jobs.get(job_id)

    def public(self, job: dict) -> dict:
        out = {k: job.get(k) for k in ("id", "state", "phase", "message", "error_kind", "operation_id")}
        out["elapsed"] = round(time.time() - job["created"], 1)
        res = job.get("result") or {}
        for k in ("spz", "ply", "ply_lite", "splats", "splat_level", "frames_used", "excluded"):
            if res.get(k) is not None:
                out[k] = res[k]
        return out
