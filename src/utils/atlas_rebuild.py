"""Rebuild an Atlas world's mesh for the viewer (its Atlas menu's "Rebuild mesh").

The viewer runs inside ComfyUI's Python (no ``agenty_core``), so ComfyUI relays the request here. It is the
core's own ``build_world_mesh`` - the one every added view runs, and the agent's ``atlas_rebuild_world_mesh`` -
from the depth maps on disk: no World Labs call, no cost, a few seconds. A world made before the mesh cleaning
(spikes at silhouettes) gets it this way.
"""
from __future__ import annotations

import os
from typing import Callable, Optional


def world_dir(body) -> str:
    if not isinstance(body, dict):
        raise ValueError("bad request")
    p = os.path.abspath(os.path.expanduser(str(body.get("world") or "").strip().strip('"')))
    if os.path.isfile(p):
        p = os.path.dirname(p)
    if not os.path.isfile(os.path.join(p, "world.json")):
        raise ValueError("no such Atlas world (no world.json there)")
    return p


def rebuild(body, build: Optional[Callable[[str], dict]] = None) -> dict:
    """``{"ok", "glb", "mesh"}`` after rebuilding ``world.glb`` of the world ``body["world"]`` names."""
    wd = world_dir(body)
    if build is not None:
        return {"ok": True, **build(wd)}
    from pathlib import Path
    from agenty_core.tools import worldlabs_atlas as wa
    world = wa._load_world(Path(wd))
    stats = wa.build_world_mesh(Path(wd), world)
    wa._save_world(Path(wd), world)
    return {"ok": True, "glb": os.path.join(wd, world.get("glb") or "world.glb"), "mesh": stats}
