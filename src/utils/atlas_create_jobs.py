"""New worlds the viewer asked for (World Labs Atlas, or SpAItial Echo): one background job each, polled by id.

The ComfyUI Image Viewer's "New Atlas world" dialog (3D space) posts a name, a prompt and / or pictures;
the viewer runs inside ComfyUI's Python (no ``agenty_core``, no World Labs key), so ComfyUI forwards the
request to this host. The world is made by the same core function the agent's ``atlas_create_world`` tool
runs, then opened in the viewer's 3D space (``atlas_open_in_viewer``'s function), so the user can fly it and
press the Previz panel's Atlas button. No agent turn is involved.

The pictures arrive as absolute paths the viewer's route already checked (inside the folders the viewer
serves); this side checks again that each one is a file.

``engine`` picks the generator: "atlas" (default) or "echo" - SpAItial Echo, the second world generator
(``agenty_core.tools.spaitial_echo``): one picture, a 360 panorama or a prompt -> a Gaussian-splat world,
opened in the viewer's 3D space by the core function itself.
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
MAX_IMAGES = 8
_ASPECTS = ("16:9", "9:16", "4:3", "3:4", "1:1")
ENGINES = ("atlas", "echo")
ECHO_MODELS = ("echo-2", "echo-hq")
ECHO_MESH = ("", "mesh", "mesh-simplified")


def _clean(body) -> dict:
    if not isinstance(body, dict):
        raise ValueError("bad request")
    name = re.sub(r"[^A-Za-z0-9_\- ]+", "_", str(body.get("name") or "")).strip()[:60]
    if not name:
        raise ValueError("give the world a name")
    prompt = str(body.get("prompt") or "").strip()[:2000]
    imgs = [str(p) for p in (body.get("image_paths") or []) if str(p or "").strip()]
    if len(imgs) > MAX_IMAGES:
        raise ValueError(f"at most {MAX_IMAGES} pictures")
    for p in imgs:
        if not os.path.isfile(p):
            raise ValueError(f"no such picture: {os.path.basename(p)}")
    if not imgs and not prompt:
        raise ValueError("describe the place, or add a picture of it")
    engine = str(body.get("engine") or "atlas").lower()
    if engine not in ENGINES:
        raise ValueError("engine is atlas or echo")
    if engine == "echo":
        if len(imgs) > 1:
            raise ValueError("Echo makes a world from ONE picture (or a panorama, or words)")
        model = str(body.get("model") or "echo-2").lower()
        if model not in ECHO_MODELS:
            raise ValueError("model is " + " or ".join(ECHO_MODELS))
        pano = str(body.get("is_pano") or "auto").lower()
        if pano not in ("auto", "true", "false"):
            raise ValueError("is_pano is auto, true or false")
        mesh = str(body.get("mesh") or "").lower()
        if mesh not in ECHO_MESH:
            raise ValueError("mesh is '', mesh or mesh-simplified")
        return {"engine": "echo", "name": name, "prompt": prompt, "image_paths": imgs, "model": model,
                "is_pano": pano, "mesh": mesh, "open_in_viewer": body.get("open_in_viewer", True) is not False}
    aspect = str(body.get("aspect_ratio") or "16:9")
    if aspect not in _ASPECTS:
        raise ValueError("aspect_ratio is one of " + ", ".join(_ASPECTS))
    try:
        detail = int(body.get("mesh_detail") or 4)
    except (TypeError, ValueError):
        raise ValueError("mesh_detail must be a number")
    if detail not in (2, 4, 8):
        raise ValueError("mesh_detail is 2, 4 or 8")
    return {"engine": "atlas", "name": name, "prompt": prompt, "image_paths": imgs, "aspect_ratio": aspect,
            "mesh_detail": detail, "open_in_viewer": body.get("open_in_viewer", True) is not False}


class AtlasCreateJobs:
    def __init__(self, create: Optional[Callable[..., str]] = None, open_world: Optional[Callable[..., str]] = None,
                 create_echo: Optional[Callable[..., str]] = None):
        self._create_fn = create
        self._open_fn = open_world
        self._echo_fn = create_echo
        self._jobs: dict[str, dict] = {}
        self._lock = threading.Lock()

    def _create(self, **kw) -> str:
        if self._create_fn is not None:
            return self._create_fn(**kw)
        from agenty_core.tools.worldlabs_atlas import atlas_create_world
        return atlas_create_world(**kw)

    def _create_echo(self, **kw) -> str:
        if self._echo_fn is not None:
            return self._echo_fn(**kw)
        from agenty_core.tools.spaitial_echo import echo_create_world
        return echo_create_world(**kw)

    def _open(self, world: str) -> str:
        if self._open_fn is not None:
            return self._open_fn(world)
        from agenty_core.tools.atlas_render import _open_in_viewer
        return _open_in_viewer(world, True, "")

    def start(self, body) -> dict:
        args = _clean(body)
        with self._lock:
            if any(j["state"] == "running" and j["name"].lower() == args["name"].lower() for j in self._jobs.values()):
                raise ValueError(f"a world named '{args['name']}' is already being made")
        if args["engine"] == "echo":
            phase = (f"Echo is building the world (about {'an hour' if args['model'] == 'echo-hq' else '8 minutes'})")
        else:
            phase = "Atlas is placing the pictures in 3D" if args["image_paths"] else "Atlas is painting the place"
        job = {"id": uuid.uuid4().hex[:12], "state": "running", "phase": phase, "name": args["name"],
               "engine": args["engine"],
               "created": time.time(), "message": "", "result": None}
        with self._lock:
            self._jobs[job["id"]] = job
            while len(self._jobs) > MAX_JOBS:
                self._jobs.pop(next(iter(self._jobs)))
        threading.Thread(target=self._work, args=(job, args), name=f"atlas-create-{job['id']}", daemon=True).start()
        return self.public(job)

    def _work(self, job: dict, args: dict) -> None:
        from agenty_core.utils import turn_scope
        scope = turn_scope.Scope()
        # The world's files must not land in whichever conversation is running: the world is opened in the
        # viewer below instead (its own sink, which takes nothing).
        scope.set("image_io.output_sink", lambda *_a, **_k: None)
        token = turn_scope.enter(scope)
        try:
            if args["engine"] == "echo":
                self._work_echo(job, args)
                return
            raw = self._create(name=args["name"], prompt=args["prompt"], image_paths=args["image_paths"] or None,
                               aspect_ratio=args["aspect_ratio"], mesh_detail=args["mesh_detail"])
            try:
                res = json.loads(raw)
            except (TypeError, ValueError):
                res = {"error": str(raw)[:300]}
            job["result"] = res
            if res.get("error"):
                job.update(state="error", message=str(res["error"]), error_kind=res.get("error_kind"),
                           operation_id=res.get("operation_id"))
                return
            job.update(world_dir=res.get("world_dir"), glb=res.get("glb"), phase="Opening the world in the viewer")
            if args["open_in_viewer"] and res.get("world_dir"):
                try:
                    opened = json.loads(self._open(res["world_dir"]))
                except Exception as exc:  # noqa: BLE001 - the world is made either way
                    opened = {"error": str(exc)}
                if opened.get("error"):
                    job["message"] = f"the world is made, but could not be opened in the viewer: {opened['error']}"
                else:
                    job["tab"] = opened.get("tab")
            job["state"] = "done"
        except Exception as exc:  # noqa: BLE001
            logger.exception("atlas create job failed")
            job.update(state="error", message=f"{type(exc).__name__}: {exc}")
        finally:
            turn_scope.leave(token)

    def _work_echo(self, job: dict, args: dict) -> None:
        """An Echo world: the core function makes it and opens the splat in the viewer's 3D space itself."""
        raw = self._create_echo(name=args["name"], image=(args["image_paths"] or [""])[0],
                                prompt="" if args["image_paths"] else args["prompt"], model=args["model"],
                                is_pano=args["is_pano"], mesh=args["mesh"], open_in_viewer=args["open_in_viewer"])
        try:
            res = json.loads(raw)
        except (TypeError, ValueError):
            res = {"error": str(raw)[:300]}
        job["result"] = res
        if res.get("error"):
            job.update(state="error", message=str(res["error"]), error_kind=res.get("error_kind"),
                       operation_id=res.get("request_id"))
            return
        viewer = res.get("viewer") or {}
        notes = [str(n) for n in (res.get("notes") or [])]
        if res.get("mesh_error"):
            notes.append(f"the world is made, but the mesh export failed: {res['mesh_error']}")
        if viewer.get("error"):
            notes.append(f"the world is made, but could not be opened in the viewer: {viewer['error']}")
        job.update(world_dir=res.get("world_dir"), splat=res.get("splat"), mesh=res.get("mesh"),
                   tab=viewer.get("tab"), message=" ".join(notes), state="done")

    def get(self, job_id: str) -> Optional[dict]:
        with self._lock:
            return self._jobs.get(job_id)

    def public(self, job: dict) -> dict:
        out = {k: job.get(k) for k in ("id", "state", "phase", "name", "engine", "message", "world_dir", "glb",
                                       "splat", "mesh", "tab", "error_kind", "operation_id")}
        out["elapsed"] = round(time.time() - job["created"], 1)
        res = job.get("result") or {}
        if res.get("frames") is not None:
            out["frames"] = len(res.get("frames") or [])
        if res.get("prompt_used"):
            out["prompt_used"] = res["prompt_used"]
        return out
