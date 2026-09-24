"""Walkable worlds from reference pictures — the tools of the World Builder.

The worlds themselves are made by the **ComfyUI-bEpicWorlds** pack (its
``/bepic_worlds/*`` routes) and shown in the bEpic Image Viewer, where the user
walks them and pins notes. These tools are the agent's side of that loop:

* thin wrappers over the routes — create, rebuild, describe, edit, revert,
  feedback, calibrate, open;
* two fat, deterministic pipelines that run ComfyUI workflows and put the
  results into a world in one call, so the model supplies only words
  ("car", "floor") and nothing mechanical:

  - ``world_add_objects``: SAM3 finds every instance of a label in the
    reference, the pack picks the cleanest one and crops it, Hunyuan3D 2.1
    turns that crop into a mesh, and one copy is placed wherever the picture
    shows the object — standing on the ground, as tall as it looks, textured
    from the picture.
  - ``world_make_material``: SAM3 finds a surface, the pack cuts the clearest
    near patch of it, which is upscaled and turned into PBR maps by Chord
    (albedo, normal, roughness) and laid onto a terrain layer.

Depth for a new world comes from the pack's *bEpic World Depth (16-bit)* node;
the usual depth nodes hand on 8 bits, which terraces the far end of a picture.

``BEPIC_WORLDS_URL`` points the route calls somewhere other than ComfyUI (a test
harness); workflows always run on ComfyUI itself.
"""

from __future__ import annotations

import json
import os
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any

import requests
from strands import tool

from agenty_core.utils.comfyui_client import get_client

try:  # the chat panel's progress line; absent outside the pipeline
    from agenty_core.utils.progress_signal import push as _progress
except Exception:  # noqa: BLE001
    def _progress(_msg: str) -> None:
        pass

# Models the pipelines use. Named here, once, so a swap is one line.
SAM3_CKPT = "sam3.1_multiplex_fp16.safetensors"
HUNYUAN3D_CKPT = "hunyuan_3d_v2.1.safetensors"
CHORD_CKPT = "CHORD\\chord_v1.safetensors"
UPSCALER = "4x-ClearRealityV1.pth"
DEPTH_CKPT = "depth_anything_v2_vitl.pth"


# ── plumbing ─────────────────────────────────────────────────────────────────

def _worlds_url() -> str:
    return (os.environ.get("BEPIC_WORLDS_URL") or get_client().base_url).rstrip("/")


def _call(method: str, path: str, body: dict | None = None, params: dict | None = None,
          timeout: float = 600) -> Any:
    """A /bepic_worlds route. Its own error message is raised as-is."""
    url = _worlds_url() + path
    try:
        r = requests.request(method, url, json=body, params=params, timeout=timeout)
    except requests.RequestException as exc:
        raise RuntimeError(f"ComfyUI is not reachable at {_worlds_url()} ({exc})") from None
    if r.status_code == 404 and path == "/bepic_worlds/info":
        raise RuntimeError("the bEpic Worlds pack isn't loaded in ComfyUI (install ComfyUI-bEpicWorlds "
                           "into custom_nodes and restart ComfyUI)")
    if r.status_code >= 400:
        try:
            msg = r.json().get("error")
        except ValueError:
            msg = None
        raise RuntimeError(msg or f"{path} answered {r.status_code}")
    return r.json()


def _post(path: str, body: dict, timeout: float = 600) -> Any:
    return _call("POST", path, body, timeout=timeout)


def _get(path: str, **params) -> Any:
    return _call("GET", path, params={k: v for k, v in params.items() if v is not None})


def _ok(**payload) -> str:
    return json.dumps({"status": "ok", **payload}, default=str)


def _fail(exc: Exception | str) -> str:
    return json.dumps({"status": "error", "error": str(exc)})


def _as_ref(value: Any) -> Any:
    """A file as ComfyUI names it. Accepts {filename, subfolder, type}, its JSON,
    a path relative to ComfyUI's input folder, or a path on this machine (which
    is uploaded to ComfyUI's input first)."""
    if isinstance(value, dict):
        return value
    text = str(value or "").strip()
    if not text:
        return None
    if text.startswith("{"):
        return json.loads(text)
    if os.path.isabs(text) and os.path.isfile(text):
        with open(text, "rb") as fh:
            up = get_client().post("/upload/image", data={"subfolder": "worlds_refs", "overwrite": "true"},
                                   files={"image": (os.path.basename(text), fh)})
        return {"filename": up["name"], "subfolder": up.get("subfolder", ""), "type": up.get("type", "input")}
    sub, _, name = text.replace("\\", "/").rpartition("/")
    return {"filename": name, "subfolder": sub, "type": "input"}


def _load_image_name(ref: dict) -> str:
    """How LoadImage names a file: 'sub/name.png', with ' [output]' for outputs."""
    name = "/".join(p for p in (ref.get("subfolder"), ref["filename"]) if p)
    t = ref.get("type") or "input"
    return name if t == "input" else f"{name} [{t}]"


def _run(prompt: dict, label: str, timeout: float = 1800) -> dict:
    """Queue an API-format workflow on ComfyUI and wait for its outputs."""
    client = get_client()
    payload: dict = {"prompt": prompt, "client_id": uuid.uuid4().hex}
    if client.api_key:
        payload["extra_data"] = {"api_key_comfy_org": client.api_key}
    res = client.post("/prompt", json_data=payload)
    if not isinstance(res, dict) or not res.get("prompt_id"):
        raise RuntimeError(f"{label}: ComfyUI did not queue the workflow ({res})")
    if res.get("node_errors"):
        raise RuntimeError(f"{label}: {json.dumps(res['node_errors'])[:600]}")
    pid = res["prompt_id"]
    try:
        from agenty_core.queue_ledger import remember
        remember(pid)
    except Exception:  # noqa: BLE001
        pass
    t0 = time.time()
    while time.time() - t0 < timeout:
        hist = client.get(f"/history/{pid}")
        if isinstance(hist, dict) and pid in hist:
            status = hist[pid].get("status") or {}
            if status.get("status_str") == "error":
                msgs = [m[1] for m in status.get("messages") or [] if m and m[0] == "execution_error"]
                detail = (msgs[0].get("exception_message") if msgs else "") or "execution failed"
                raise RuntimeError(f"{label}: {detail.strip()[:600]}")
            if status.get("completed", True):
                return hist[pid].get("outputs") or {}
        time.sleep(1.5)
    raise TimeoutError(f"{label}: no result after {int(timeout)} s (prompt {pid} is still queued or running)")


def _files(outputs: dict, node: str, key: str = "images") -> list[dict]:
    return list((outputs.get(node) or {}).get(key) or [])


def _slug(text: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in str(text).lower()).strip("_")[:40] or "x"


# ── workflows ────────────────────────────────────────────────────────────────

def _wf_depth(image: str, prefix: str) -> dict:
    return {
        "1": {"class_type": "LoadImage", "inputs": {"image": image}},
        "2": {"class_type": "bEpicWorldDepth", "inputs": {"image": ["1", 0], "ckpt_name": DEPTH_CKPT,
                                                          "input_size": 770, "filename_prefix": prefix}},
    }


def _wf_sam3(image: str, prompt: str, individual: bool, prefix: str, threshold: float = 0.4) -> dict:
    return {
        "1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": SAM3_CKPT}},
        "2": {"class_type": "CLIPTextEncode", "inputs": {"text": prompt, "clip": ["1", 1]}},
        "3": {"class_type": "LoadImage", "inputs": {"image": image}},
        "4": {"class_type": "SAM3_Detect", "inputs": {"model": ["1", 0], "image": ["3", 0], "conditioning": ["2", 0],
                                                      "threshold": threshold, "refine_iterations": 2,
                                                      "individual_masks": individual}},
        "5": {"class_type": "MaskToImage", "inputs": {"mask": ["4", 0]}},
        "6": {"class_type": "SaveImage", "inputs": {"images": ["5", 0], "filename_prefix": prefix}},
    }


def _wf_hunyuan3d(image: str, prefix: str, seed: int) -> dict:
    # The wiring of ComfyUI's own "Hunyuan3D 2.1" template, API format.
    return {
        "1": {"class_type": "ImageOnlyCheckpointLoader", "inputs": {"ckpt_name": HUNYUAN3D_CKPT}},
        "2": {"class_type": "LoadImage", "inputs": {"image": image}},
        "3": {"class_type": "ModelSamplingAuraFlow", "inputs": {"model": ["1", 0], "shift": 1.0}},
        "4": {"class_type": "CLIPVisionEncode", "inputs": {"clip_vision": ["1", 1], "image": ["2", 0], "crop": "center"}},
        "5": {"class_type": "Hunyuan3Dv2Conditioning", "inputs": {"clip_vision_output": ["4", 0]}},
        "6": {"class_type": "EmptyLatentHunyuan3Dv2", "inputs": {"resolution": 4096, "batch_size": 1}},
        "7": {"class_type": "KSampler", "inputs": {"model": ["3", 0], "positive": ["5", 0], "negative": ["5", 1],
                                                   "latent_image": ["6", 0], "seed": int(seed), "steps": 30, "cfg": 5.0,
                                                   "sampler_name": "euler", "scheduler": "normal", "denoise": 1.0}},
        "8": {"class_type": "VAEDecodeHunyuan3D", "inputs": {"samples": ["7", 0], "vae": ["1", 2],
                                                             "num_chunks": 8000, "octree_resolution": 256}},
        "9": {"class_type": "VoxelToMesh", "inputs": {"voxel": ["8", 0], "algorithm": "surface net", "threshold": 0.6}},
        "10": {"class_type": "SaveGLB", "inputs": {"mesh": ["9", 0], "filename_prefix": prefix}},
    }


def _wf_chord(image: str, prefix: str) -> dict:
    return {
        "1": {"class_type": "LoadImage", "inputs": {"image": image}},
        "2": {"class_type": "UpscaleModelLoader", "inputs": {"model_name": UPSCALER}},
        "3": {"class_type": "ImageUpscaleWithModel", "inputs": {"upscale_model": ["2", 0], "image": ["1", 0]}},
        "4": {"class_type": "ImageScale", "inputs": {"image": ["3", 0], "upscale_method": "lanczos",
                                                     "width": 1024, "height": 1024, "crop": "disabled"}},
        "5": {"class_type": "ChordLoadModel", "inputs": {"ckpt_name": CHORD_CKPT}},
        "6": {"class_type": "ChordMaterialEstimation", "inputs": {"chord_model": ["5", 0], "image": ["4", 0]}},
        "7": {"class_type": "SaveImage", "inputs": {"images": ["6", 0], "filename_prefix": f"{prefix}_albedo"}},
        "8": {"class_type": "SaveImage", "inputs": {"images": ["6", 1], "filename_prefix": f"{prefix}_normal"}},
        "9": {"class_type": "SaveImage", "inputs": {"images": ["6", 2], "filename_prefix": f"{prefix}_rough"}},
    }


def _stage_reference(name: str) -> tuple[dict, str]:
    ref = _post("/bepic_worlds/stage_reference", {"name": name})["reference"]
    return ref, _load_image_name(ref)


def _sam3(name: str, image: str, prompt: str, individual: bool, what: str) -> list[dict]:
    _progress(f"🔍 SAM3: finding {what} …")
    out = _run(_wf_sam3(image, prompt, individual, f"worlds/{_slug(name)}/sam3_{_slug(what)}"), f"SAM3 ({what})")
    return _files(out, "6")


# ── tools: the world itself ──────────────────────────────────────────────────

@tool
def world_schema() -> str:
    """The world format: item kinds and ids, their fields and units, and every
    edit operation with its arguments. Read it before writing edit ops."""
    try:
        info = _get("/bepic_worlds/info")
        return _ok(**info)
    except Exception as exc:  # noqa: BLE001
        return _fail(exc)


@tool
def world_create(reference: str, name: str, spec: str = "", fov: float = 50.0,
                 depth: str = "auto", world_size: float = 0.0, seed: int = 0) -> str:
    """Build a new walkable world from a reference picture; it opens in the viewer.

    Args:
        reference: The picture: a filename in ComfyUI's input folder (as staged,
            e.g. "garage.jpg" or "sub/garage.jpg"), a ComfyUI ref
            {"filename", "subfolder", "type"} as JSON, or a local path.
        name: The world's name (letters, digits, _ -). An existing name gets a
            numbered suffix; use world_rebuild to remake an existing world.
        spec: Plain words for what the picture can't say: biome, time of day,
            density, what grows ("misty pine forest at sunset", "underground car
            park, fluorescent light"). Empty = read everything from the picture.
        fov: The picture's VERTICAL field of view in degrees. Phones and most
            photos ~50; wide interior shots 60-70.
        depth: "auto" (default) = estimate a 16-bit depth map, which gives the
            picture itself as a 3D hero view and tells interiors from outdoors;
            "none" = no depth; or a file/ref of your own depth map.
        world_size: Metres across (0 = the pack's default, 240).
        seed: Random seed for everything generated (0 = the pack's choice).
    """
    try:
        ref = _as_ref(reference)
        body: dict = {"reference": ref, "name": name, "spec": spec or None, "fov": float(fov),
                      "world_size": float(world_size) or None, "seed": int(seed) or None, "open_in_viewer": True}
        if str(depth).strip().lower() == "auto":
            _progress("🗺️ Estimating depth (16-bit) …")
            out = _run(_wf_depth(_load_image_name(ref), f"{_slug(name)}_depth"), "depth")
            files = _files(out, "2")
            if not files:
                raise RuntimeError("the depth node wrote no file")
            body["depth"] = files[0]
        elif str(depth).strip().lower() not in ("", "none", "no", "false"):
            body["depth"] = _as_ref(depth)
        _progress("🏗️ Building the world …")
        res = _post("/bepic_worlds/create", {k: v for k, v in body.items() if v is not None})
        return _ok(**res)
    except Exception as exc:  # noqa: BLE001
        return _fail(exc)


@tool
def world_rebuild(name: str, pitch: float | None = None, fov: float | None = None, spec: str | None = None,
                  note: str = "") -> str:
    """Make a world again from the picture and depth it was made from, with some
    settings changed — its next version. Objects and materials added since are
    NOT carried over (rebuild first, then add them).

    Args:
        name: The world.
        pitch: Camera tilt in degrees (from world_add_objects' camera fit).
        fov: Vertical field of view in degrees.
        spec: A new spec.
        note: What changed, for the history.
    """
    try:
        body = {"name": name, "pitch": pitch, "fov": fov, "spec": spec, "note": note or None}
        return _ok(**_post("/bepic_worlds/rebuild", {k: v for k, v in body.items() if v is not None}))
    except Exception as exc:  # noqa: BLE001
        return _fail(exc)


@tool
def world_list() -> str:
    """All worlds, with their current version and open feedback count."""
    try:
        return _ok(**_get("/bepic_worlds/list"))
    except Exception as exc:  # noqa: BLE001
        return _fail(exc)


@tool
def world_describe(name: str) -> str:
    """What a world is now: its items (ids, kinds, key settings), version and
    history notes, what the picture analysis found. Read before editing."""
    try:
        return _ok(**_get("/bepic_worlds/world", name=name, summary=1))
    except Exception as exc:  # noqa: BLE001
        return _fail(exc)


@tool
def world_edit(name: str, ops: str, note: str) -> str:
    """Apply edit operations to a world as one new version (the viewer refreshes).

    Args:
        name: The world.
        ops: A JSON list of operations, e.g.
            [{"op": "set", "id": "env", "path": "fog.density", "value": 0.01},
             {"op": "scale_scatter", "factor": 0.5}]. world_schema lists them all.
        note: What this version changes and why, in a sentence.
    """
    try:
        parsed = json.loads(ops) if isinstance(ops, str) else ops
        return _ok(**_post("/bepic_worlds/edit", {"name": name, "ops": parsed, "note": note}))
    except Exception as exc:  # noqa: BLE001
        return _fail(exc)


@tool
def world_revert(name: str, version: int, note: str = "") -> str:
    """Go back to an earlier version (saved as a new version; nothing is lost)."""
    try:
        return _ok(**_post("/bepic_worlds/revert", {"name": name, "version": int(version), "note": note}))
    except Exception as exc:  # noqa: BLE001
        return _fail(exc)


@tool
def world_feedback(name: str, status: str = "open") -> str:
    """The notes the user pinned in the world while walking it. Each carries the
    text, the spot (x, y, z), the view, and a snapshot of what they saw — saved
    locally as `snapshot_file`; look at it with analyze_image before acting.

    Args:
        name: The world.
        status: "open" (default), "resolved" or "all".
    """
    try:
        entries = _get("/bepic_worlds/feedback", name=name, status=status or "open").get("feedback") or []
        folder = Path(tempfile.gettempdir()) / "agenty_worlds" / _slug(name)
        for e in entries:
            ref = e.pop("snapshot_view", None)
            e.pop("snapshot", None)
            if not ref:
                continue
            try:
                resp = get_client().get("/view", params=ref, raw=True)
                folder.mkdir(parents=True, exist_ok=True)
                path = folder / f"{e.get('id', 'fb')}.jpg"
                path.write_bytes(resp.content)
                e["snapshot_file"] = str(path)
            except Exception:  # noqa: BLE001
                pass
        return _ok(name=name, feedback=entries)
    except Exception as exc:  # noqa: BLE001
        return _fail(exc)


@tool
def world_resolve_feedback(name: str, ids: list, reply: str) -> str:
    """Mark notes as dealt with, with a reply the user sees. Resolve only notes a
    saved version actually addresses.

    Args:
        name: The world.
        ids: Feedback ids (["fb_1a2b3c4d", …]).
        reply: What was done, in a sentence.
    """
    try:
        return _ok(**_post("/bepic_worlds/feedback/resolve", {"name": name, "ids": list(ids or []), "reply": reply}))
    except Exception as exc:  # noqa: BLE001
        return _fail(exc)


@tool
def world_calibrate(name: str, wait_seconds: float = 120.0) -> str:
    """Match the world's look (exposure, fill, sun, fog, how much photographed
    light each surface keeps) to its reference picture by measurement. Runs in
    the viewer — the world is opened there — and saves a new version whose note
    gives the error before and after. Do this last, after objects and materials.
    """
    try:
        req = _post("/bepic_worlds/calibrate", {"name": name})
        before = int(req.get("version_before") or 0)
        deadline = time.time() + max(0.0, float(wait_seconds))
        _progress("🎚️ Matching the look to the picture (in the viewer) …")
        while time.time() < deadline:
            time.sleep(2.0)
            d = _get("/bepic_worlds/world", name=name, summary=1)
            if int(d.get("version") or 0) > before:
                return _ok(matched=True, version=d["version"], note=((d.get("history") or [{}])[-1]).get("note"))
        return _ok(matched=False, note="no new version yet — the match runs in the viewer, which must be open "
                                       "in a browser tab that is visible (a background tab doesn't render)")
    except Exception as exc:  # noqa: BLE001
        return _fail(exc)


@tool
def world_open(name: str, version: int = 0) -> str:
    """Show a world (or one of its versions) in the user's viewer."""
    try:
        return _ok(**_post("/bepic_worlds/open", {"name": name, "version": int(version) or None}))
    except Exception as exc:  # noqa: BLE001
        return _fail(exc)


# ── tools: real things from the picture ──────────────────────────────────────

@tool
def world_add_objects(name: str, label: str, max_count: int = 12, known_height_m: float = 0.0,
                      fit_camera: bool = False, seed: int = 7) -> str:
    """Put real 3D copies of an object in the picture into the world, where the
    picture shows them: SAM3 finds every instance, the cleanest one becomes a
    mesh (Hunyuan3D 2.1, ~1-2 min), and a copy of it stands at each instance's
    spot, as tall as it looks and textured from the picture.

    One call per kind of object; things that stand on the ground (cars, pillars,
    benches, trees, lamp posts, crates). Not for the ground, walls or sky.

    Args:
        name: The world.
        label: What to find, as a short noun: "car", "pillar", "bench".
        max_count: At most this many instances (1-32).
        known_height_m: The real height of this kind of object in metres, if it
            is standard (car 1.5, person 1.75, door 2.0, parking pillar ~2.5).
            Needed for fit_camera.
        fit_camera: Measure the camera's tilt from these objects' known height
            and, if it differs from the world's, REBUILD the world with it first
            — which drops objects and materials added before. So use it on the
            first kind of object you add, and only when known_height_m is sure.
            Worth it when objects come out too small or too big.
        seed: Seed for the mesh generation (another seed = another variant).
    """
    try:
        label = _slug(label)
        n = max(1, min(32, int(max_count)))
        _ref, image = _stage_reference(name)
        masks = _sam3(name, image, f"{label}:{n}", True, label)
        if not masks:
            return _ok(added=0, note=f"SAM3 found no '{label}' in the picture")
        crops = _post("/bepic_worlds/object_crops", {"name": name, "label": label, "masks": masks, "limit": n, "crops": 1})
        objs = crops.get("objects") or []
        camera = None
        if fit_camera and known_height_m > 0:
            clean = [o for o in objs if not o.get("clipped") and o.get("solidity", 0) > 0.6]
            if clean:
                fit = _post("/bepic_worlds/fit_camera", {"name": name, "objects": [
                    {"bbox": o["bbox"], "height": float(known_height_m)} for o in clean]})
                camera = {"pitch": fit["pitch"], "pitch_before": fit.get("pitch_before"), "from": len(clean)}
                if fit.get("pitch_before") is None or abs(float(fit["pitch"]) - float(fit["pitch_before"])) > 0.5:
                    _progress(f"📐 Camera tilt {fit['pitch']:.1f}° (from {len(clean)} {label}s) — rebuilding …")
                    _post("/bepic_worlds/rebuild", {"name": name, "pitch": fit["pitch"],
                                                    "note": f"camera tilt {fit['pitch']:.2f}° measured from {len(clean)} {label}s "
                                                            f"of {known_height_m} m", "open_in_viewer": False})
                    camera["rebuilt"] = True
                    crops = _post("/bepic_worlds/object_crops", {"name": name, "label": label, "masks": masks,
                                                                 "limit": n, "crops": 1})
                    objs = crops.get("objects") or []
        placed = [o for o in objs if "error" not in (o.get("placement") or {}) and o.get("score", 0) > 0.02]
        if not placed or not objs[0].get("crop"):
            return _ok(added=0, found=len(objs), camera=camera,
                       note="found, but none could be placed on the ground (behind the horizon, or clipped by the frame)")
        best = objs[0]
        _progress(f"🧊 Hunyuan3D: a {label} from the picture …")
        out = _run(_wf_hunyuan3d(_load_image_name(best["crop"]), f"worlds/{_slug(name)}/{label}", seed), "Hunyuan3D")
        glbs = _files(out, "10", "3d") or _files(out, "10", "result")
        if not glbs:
            raise RuntimeError("Hunyuan3D wrote no mesh")
        res = _post("/bepic_worlds/edit", {"name": name, "ops": [{
            "op": "add_asset", "label": label, "glb": glbs[0], "texture": best["texture"],
            "bboxes": [o["bbox"] for o in placed]}],
            "note": f"{len(placed)} {label}{'s' if len(placed) != 1 else ''} from the picture (SAM3 + Hunyuan3D 2.1)"})
        spots = [{"height_m": o["placement"].get("height"), "distance_m": o["placement"].get("distance")} for o in placed]
        return _ok(added=len(placed), found=len(objs), label=label, version=res.get("version"),
                   ids=res.get("changed"), placements=spots, camera=camera, mesh=glbs[0])
    except Exception as exc:  # noqa: BLE001
        return _fail(exc)


@tool
def world_make_material(name: str, surface: str = "floor", layer: int = 0, terrain_id: str = "terrain",
                        tile_m: float = 5.0, normal_strength: float = 0.6) -> str:
    """Give a terrain layer a real PBR material made from a surface in the
    picture: SAM3 finds the surface, the clearest near patch is cut out,
    upscaled, and turned into albedo, normal and roughness maps by Chord.

    Args:
        name: The world.
        surface: What the surface is, as SAM3 should find it: "floor",
            "asphalt", "grass", "sand", "gravel path".
        layer: The terrain layer it replaces (0 = the main ground; see
            world_describe for what each layer is).
        terrain_id: The terrain item ("terrain"; interiors also have "ceiling").
        tile_m: Metres one repeat of the texture covers.
        normal_strength: Relief of the normal map (0-2).
    """
    try:
        _ref, image = _stage_reference(name)
        masks = _sam3(name, image, _slug(surface).replace("_", " "), False, surface)
        if not masks:
            return _ok(applied=False, note=f"SAM3 found no '{surface}' in the picture")
        crop = _post("/bepic_worlds/material_crop", {"name": name, "masks": masks, "label": _slug(surface)})
        _progress(f"🧱 Chord: PBR maps for the {surface} …")
        out = _run(_wf_chord(_load_image_name(crop["patch"]), f"worlds/{_slug(name)}/mat_{_slug(surface)}"), "Chord")
        maps = {k: (_files(out, node) or [None])[0] for k, node in (("albedo", "7"), ("normal", "8"), ("roughness", "9"))}
        if not maps["albedo"]:
            raise RuntimeError("Chord wrote no maps")
        op = {"op": "set_material", "id": terrain_id, "layer": int(layer), "tile": float(tile_m),
              "normalScale": float(normal_strength), **{k: v for k, v in maps.items() if v}}
        res = _post("/bepic_worlds/edit", {"name": name, "ops": [op],
                                           "note": f"{surface} material from the picture (Chord)"})
        return _ok(applied=True, version=res.get("version"), patch_box=crop.get("box"), maps=maps)
    except Exception as exc:  # noqa: BLE001
        return _fail(exc)


WORLD_TOOLS = [
    world_schema, world_create, world_rebuild, world_list, world_describe, world_edit, world_revert,
    world_feedback, world_resolve_feedback, world_calibrate, world_open,
    world_add_objects, world_make_material,
]
