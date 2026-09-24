"""Walkable worlds from reference pictures — the tools of the World Builder.

The worlds themselves are made by the **ComfyUI-bEpicWorlds** pack (its
``/bepic_worlds/*`` routes) and shown in the bEpic Image Viewer, where the user
walks them and pins notes. These tools are the agent's side of that loop:

* thin wrappers over the routes — create, rebuild, describe, edit, revert,
  feedback, calibrate, open;
* fat, deterministic pipelines that run ComfyUI workflows and put the results
  into a world in one call, so the model supplies words ("car", "floor",
  "a weathered wooden bench") and nothing mechanical.

**Every generative step is a slot filled by a ComfyUI workflow**, never by
code here: depth, segment, image_to_3d, texture_refine, texture_generate,
material, sky, object_image. The pack ships a workflow (or several) per slot
and a default; ``world_slots`` lists them and ``world_choose_slot`` swaps one
for another — one of the pack's, or any template from the agent's own library
(agenty_core's custom and official templates). Workflows are bound by their
node titles (``IN:image``, ``IN:prompt.text``, ``OUT:mesh`` …, see the pack's
``bepic_worlds/slots.py``); a library template without such titles is bound
by its node types (its LoadImage nodes, its positive prompt, its save nodes).

``BEPIC_WORLDS_URL`` points the route calls somewhere other than ComfyUI (a test
harness); workflows always run on ComfyUI itself.
"""

from __future__ import annotations

import copy
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

# API nodes whose meshes come textured (their materials are kept, not replaced
# by the picture) — for library templates the pack's manifest doesn't describe.
_TEXTURED_3D_NODES = ("Meshy", "Tripo", "Rodin")


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


def _slug(text: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in str(text).lower()).strip("_")[:40] or "x"


def _download(ref: dict, folder: str = "") -> str:
    """A ComfyUI file, saved locally (for the vision agent to look at)."""
    resp = get_client().get("/view", params=ref, raw=True)
    out = Path(tempfile.gettempdir()) / "agenty_worlds" / (folder or "files")
    out.mkdir(parents=True, exist_ok=True)
    path = out / ref["filename"]
    path.write_bytes(resp.content)
    return str(path)


# ── slots: which workflow does each generative step ──────────────────────────

def _slot_info() -> dict:
    return _get("/bepic_worlds/slots")


def _library_template(name: str) -> dict | None:
    """A template from the agent's own library, in API format."""
    try:
        from agenty_core.tools import comfyui as core
    except Exception:  # noqa: BLE001
        return None
    wf = core._fetch_template(name)
    if wf is None:
        return None
    wf = copy.deepcopy(wf)
    if core._is_graph_format(wf):
        wf = core._convert_graph_to_api(wf)
    return wf


def _template_for(slot: str, override: str = "") -> tuple[str, dict, dict]:
    """(name, API workflow, what the manifest says about it) for a slot."""
    info = _slot_info()
    spec = (info.get("slots") or {}).get(slot)
    if spec is None:
        raise ValueError(f"no slot '{slot}'")
    name = override or (info.get("choices") or {}).get(slot) or spec.get("default")
    meta = (spec.get("templates") or {}).get(name)
    if meta is not None:
        return name, _get("/bepic_worlds/slot_template", name=name), meta
    wf = _library_template(name)
    if wf is None:
        raise ValueError(f"slot '{slot}': no workflow '{name}' in the pack or the template library")
    textured = any(str(n.get("class_type", "")).startswith(_TEXTURED_3D_NODES) for n in wf.values())
    return name, wf, {"about": "from the template library", "textured": textured, "library": True}


def _title(node: dict) -> str:
    return str((node.get("_meta") or {}).get("title") or "")


def _bind(wf: dict, inputs: dict, prefix: str) -> tuple[dict, dict]:
    """Fill a workflow's inputs; returns (workflow, {output node id: name})."""
    wf = copy.deepcopy(wf)
    outputs: dict[str, str] = {}
    titled = any(_title(n).startswith(("IN:", "OUT:")) for n in wf.values())
    if titled:
        for nid, node in wf.items():
            t = _title(node)
            if t.startswith("IN:"):
                for spec in t[3:].split("|"):
                    key, _, field = spec.strip().partition(".")
                    if key not in inputs or inputs[key] is None:
                        continue
                    val = inputs[key]
                    if not field:
                        field = "image"
                        val = _load_image_name(val) if isinstance(val, dict) else val
                    node["inputs"][field] = val
            elif t.startswith("OUT:"):
                outputs[nid] = t[4:].strip()
    else:
        # A library template: its LoadImage nodes take the images in order,
        # its first prompt box the prompt, its save nodes are the outputs.
        images = [v for k, v in inputs.items() if isinstance(v, dict) and "filename" in v]
        loaders = sorted((nid for nid, n in wf.items() if n.get("class_type") == "LoadImage"), key=lambda x: int(x) if x.isdigit() else 0)
        for nid, ref in zip(loaders, images):
            wf[nid]["inputs"]["image"] = _load_image_name(ref)
        if inputs.get("prompt"):
            boxes = [nid for nid, n in wf.items() if isinstance(n["inputs"].get("text"), str)
                     and "negative" not in _title(n).lower()]
            boxes += [nid for nid, n in wf.items() if isinstance(n["inputs"].get("prompt"), str)]
            if boxes:
                node = wf[boxes[0]]["inputs"]
                node["text" if "text" in node else "prompt"] = inputs["prompt"]
        for nid, n in wf.items():
            if str(n.get("class_type", "")).startswith(("Save", "Preview")):
                outputs[nid] = "mesh" if "GLB" in n["class_type"] or "3D" in n["class_type"] else f"out_{nid}"
    for nid in outputs:
        if "filename_prefix" in wf[nid]["inputs"]:
            wf[nid]["inputs"]["filename_prefix"] = f"{prefix}_{outputs[nid]}"
    return wf, outputs


def _run_slot(slot: str, inputs: dict, prefix: str, override: str = "",
              timeout: float = 1800) -> tuple[dict, str, dict]:
    """Run a slot's workflow. Returns ({output name: [ComfyUI refs]}, template name, meta)."""
    name, wf, meta = _template_for(slot, override)
    wf, outs = _bind(wf, inputs, prefix)
    _progress(f"⚙️ {slot}: {name} …")
    res = _run(wf, f"{slot} ({name})", timeout=timeout)
    files: dict[str, list] = {}
    for nid, oname in outs.items():
        for key, vals in (res.get(nid) or {}).items():
            if isinstance(vals, list):
                files.setdefault(oname, []).extend(v for v in vals if isinstance(v, dict) and v.get("filename"))
    if not files:
        raise RuntimeError(f"{slot} ({name}) produced no files")
    return files, name, meta


def _stage_reference(name: str) -> dict:
    return _post("/bepic_worlds/stage_reference", {"name": name})["reference"]


def _describe(ref: dict, question: str, folder: str) -> str:
    """What the vision agent sees in a picture, or '' when it can't look."""
    try:
        from src.tools.image_handling import analyze_image
        out = analyze_image(file_path=_download(ref, folder), question=question)
        if isinstance(out, dict) and out.get("status") == "ok":
            return " ".join(c.get("text", "") for c in out.get("content") or []).strip()
    except Exception:  # noqa: BLE001
        pass
    return ""


def _texture_prompt(description: str) -> str:
    return (f"seamless tileable texture of {description}. Orthographic top-down view, flat even diffuse "
            "lighting, no shadows, no perspective, no objects, sharp fine surface detail, photorealistic")


# ── tools: slots ─────────────────────────────────────────────────────────────

@tool
def world_slots() -> str:
    """The generative steps of world building ("slots": depth, segment,
    image_to_3d, texture_refine, texture_generate, material, sky, object_image),
    the workflows that can fill each (with what they cost and need), and the
    one chosen for each now."""
    try:
        info = _slot_info()
        slots = {k: {"about": v.get("about"), "chosen": info["choices"].get(k), "default": v.get("default"),
                     "workflows": v.get("templates")} for k, v in info["slots"].items()}
        return _ok(slots=slots)
    except Exception as exc:  # noqa: BLE001
        return _fail(exc)


@tool
def world_choose_slot(slot: str, template: str = "") -> str:
    """Choose which workflow fills a slot from now on (for every world).

    Args:
        slot: The slot (see world_slots), e.g. "image_to_3d".
        template: One of the slot's workflows (e.g. "i23d_meshy"), or the name
            of any template in the template library whose inputs and outputs
            fit the slot. Empty = back to the slot's default.
    """
    try:
        if template:
            _template_for(slot, template)            # fails early when it can't be found
        return _ok(**_post("/bepic_worlds/slots", {"slot": slot, "template": template}))
    except Exception as exc:  # noqa: BLE001
        return _fail(exc)


# ── tools: the world itself ──────────────────────────────────────────────────

@tool
def world_schema() -> str:
    """The world format: item kinds and ids, their fields and units, and every
    edit operation with its arguments. Read it before writing edit ops."""
    try:
        return _ok(**_get("/bepic_worlds/info"))
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
        depth: "auto" (default) = the depth slot's chosen workflow; a depth
            workflow's name ("depth_da2_16bit", "depth_sharp_metric") for this
            once; "none" = no depth; or a file/ref of a depth map (bright = near).
        world_size: Metres across (0 = the pack's default, 240).
        seed: Random seed for everything generated (0 = the pack's choice).
    """
    try:
        ref = _as_ref(reference)
        body: dict = {"reference": ref, "name": name, "spec": spec or None, "fov": float(fov),
                      "world_size": float(world_size) or None, "seed": int(seed) or None, "open_in_viewer": True}
        d = str(depth or "").strip()
        if d.lower() in ("", "none", "no", "false"):
            pass
        elif d.lower() == "auto" or d.startswith("depth_") or not ("." in d or d.startswith("{")):
            files, used, _meta = _run_slot("depth", {"image": ref, "prefix": f"{_slug(name)}_depth"},
                                           f"worlds/{_slug(name)}/depth", "" if d.lower() == "auto" else d)
            body["depth"] = files["depth"][0]
            body["note"] = f"created (depth: {used})"
        else:
            body["depth"] = _as_ref(d)
        _progress("🏗️ Building the world …")
        res = _post("/bepic_worlds/create", {k: v for k, v in body.items() if v is not None})
        return _ok(**res)
    except Exception as exc:  # noqa: BLE001
        return _fail(exc)


@tool
def world_rebuild(name: str, pitch: float | None = None, fov: float | None = None, spec: str | None = None,
                  note: str = "") -> str:
    """Make a world again from the picture and depth it was made from, with some
    settings changed — its next version. Objects, materials and skies added
    since are NOT carried over (rebuild first, then add them).

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
        for e in entries:
            ref = e.pop("snapshot_view", None)
            e.pop("snapshot", None)
            if ref:
                try:
                    e["snapshot_file"] = _download(ref, _slug(name))
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


# ── tools: real things from the picture, and made-up ones ────────────────────

def _make_mesh(name: str, label: str, crop: dict, seed: int) -> tuple[dict, str, bool]:
    """image_to_3d on one object crop → (GLB ref, workflow name, textured?)."""
    files, used, meta = _run_slot("image_to_3d", {"image": crop, "seed": int(seed)},
                                  f"worlds/{_slug(name)}/{label}", timeout=2400)
    glbs = files.get("mesh") or next(iter(files.values()))
    return glbs[0], used, bool(meta.get("textured"))


@tool
def world_add_objects(name: str, label: str, max_count: int = 12, known_height_m: float = 0.0,
                      fit_camera: bool = False, seed: int = 7) -> str:
    """Put real 3D copies of an object in the picture into the world, where the
    picture shows them: the segment slot (SAM3) finds every instance, the
    cleanest one becomes a mesh (the image_to_3d slot: Hunyuan3D locally, or
    Meshy with its own PBR textures), and a copy stands at each instance's
    spot, as tall as it looks.

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
        ref = _stage_reference(name)
        seg, _used, _m = _run_slot("segment", {"image": ref, "prompt": f"{label}:{n}", "individual": True},
                                   f"worlds/{_slug(name)}/seg_{label}")
        masks = seg.get("masks") or next(iter(seg.values()))
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
        if not placed or not objs or not objs[0].get("crop"):
            return _ok(added=0, found=len(objs), camera=camera,
                       note="found, but none could be placed on the ground (behind the horizon, or clipped by the frame)")
        best = objs[0]
        glb, used, textured = _make_mesh(name, label, best["crop"], seed)
        op = {"op": "add_asset", "label": label, "glb": glb, "bboxes": [o["bbox"] for o in placed]}
        if textured:
            op["textured"] = True
        else:
            op["texture"] = best["texture"]
        res = _post("/bepic_worlds/edit", {"name": name, "ops": [op],
                                           "note": f"{len(placed)} {label}{'s' if len(placed) != 1 else ''} "
                                                   f"from the picture ({used})"})
        spots = [{"height_m": o["placement"].get("height"), "distance_m": o["placement"].get("distance")} for o in placed]
        return _ok(added=len(placed), found=len(objs), label=label, version=res.get("version"),
                   ids=res.get("changed"), placements=spots, camera=camera, mesh=glb, workflow=used,
                   textured_by="the model" if textured else "the picture")
    except Exception as exc:  # noqa: BLE001
        return _fail(exc)


@tool
def world_add_props(name: str, description: str, height_m: float, label: str = "",
                    positions: list | None = None, count: int = 0, center: list | None = None,
                    radius_m: float = 10.0, seed: int = 7) -> str:
    """Put made-up objects into a world — things the picture doesn't show: the
    object_image slot pictures it from your words, the image_to_3d slot makes
    the mesh, and copies stand on the ground where you say.

    Args:
        name: The world.
        description: The object in a few concrete words ("a weathered wooden
            park bench with iron legs", "a red fire extinguisher cabinet").
        height_m: Its real height in metres.
        label: A short id stem ("bench"); defaults to a word of the description.
        positions: Where, as [[x, z], …] in metres (world_describe gives the
            walk spawn and bounds; feedback notes carry points).
        count: Or scatter this many copies…
        center: …around [x, z] (default: where the walk starts)…
        radius_m: …within this radius, no closer than their height.
        seed: Variant of the generated object.
    """
    try:
        label = _slug(label or description.split()[-1])
        prompt = (f"{description}. A single object, the whole of it in view, three-quarter front view, "
                  "centred, plain white background, soft even studio light, photorealistic")
        pic, _used, _m = _run_slot("object_image", {"prompt": prompt, "seed": int(seed)},
                                   f"worlds/{_slug(name)}/prop_{label}")
        image = (pic.get("image") or [None])[0]
        mask = (pic.get("mask") or [None])[0]
        if not image or not mask:
            raise RuntimeError("the object_image workflow must give an 'image' and a 'mask'")
        cut = _post("/bepic_worlds/cutout", {"name": name, "image": image, "mask": mask, "label": label})
        glb, used, textured = _make_mesh(name, label, cut["crop"], seed)
        op: dict = {"op": "add_asset", "label": label, "glb": glb, "height": float(height_m), "seed": int(seed)}
        if textured:
            op["textured"] = True
        else:
            op["texture"] = cut["texture"]
        if positions:
            op["positions"] = positions
        else:
            if not center:
                spawn = ((_get("/bepic_worlds/world", name=name, summary=1).get("walk") or {}).get("spawn")) or [0, 0, 0]
                center = [spawn[0], spawn[2]]
            op["scatter"] = {"count": max(1, int(count or 1)), "center": center, "radius": float(radius_m),
                             "spacing": float(height_m), "seed": int(seed)}
        res = _post("/bepic_worlds/edit", {"name": name, "ops": [op], "note": f"{label}: {description} ({used})"})
        return _ok(added=len(res.get("changed") or []), ids=res.get("changed"), version=res.get("version"),
                   picture=image, mesh=glb, workflow=used)
    except Exception as exc:  # noqa: BLE001
        return _fail(exc)


@tool
def world_make_material(name: str, surface: str = "floor", source: str = "picture", description: str = "",
                        refine: bool = True, denoise: float = 0.55, layer: int = 0, terrain_id: str = "terrain",
                        tile_m: float = 5.0, normal_strength: float = 0.6, seed: int = 7) -> str:
    """Give a terrain layer a real, TILEABLE PBR material.

    From the picture (default): the segment slot finds the surface, the pack
    cuts the clearest near patch; the patch is described (what the surface is
    made of), refined by the texture_refine slot (img2img diffusion guided by
    that description: sharper, cleaner, even light, seamless), and turned into
    albedo, normal and roughness maps by the material slot (Chord). From words
    (source="prompt"): the texture_generate slot makes the texture instead.

    Args:
        name: The world.
        surface: What the surface is, as segmentation should find it: "floor",
            "asphalt", "grass", "sand", "gravel path".
        source: "picture" (cut from the reference) or "prompt" (from description).
        description: What the surface is made of, concretely ("worn grey
            concrete with tyre marks and faint oil stains"). Empty = the vision
            agent describes the patch. Required for source="prompt".
        refine: Run the diffusion refinement on a picture patch (default). Off
            keeps the photo's own pixels (made tileable only by the maps' step).
        denoise: How far refinement may move from the photo (0.3 close – 0.6 free).
        layer: The terrain layer it replaces (0 = the main ground).
        terrain_id: The terrain item ("terrain"; interiors also have "ceiling").
        tile_m: Metres one repeat of the texture covers.
        normal_strength: Relief of the normal map (0-2).
        seed: Variant.
    """
    try:
        folder = f"worlds/{_slug(name)}/mat_{_slug(surface)}"
        patch = None
        if source == "prompt":
            if not description:
                raise ValueError("source='prompt' needs a description of the surface")
            tex, _u, _m = _run_slot("texture_generate", {"prompt": _texture_prompt(description), "seed": int(seed)}, folder)
            texture = tex["texture"][0]
        else:
            ref = _stage_reference(name)
            seg, _u, _m = _run_slot("segment", {"image": ref, "prompt": _slug(surface).replace("_", " "),
                                                "individual": False}, f"{folder}_seg")
            masks = seg.get("masks") or next(iter(seg.values()))
            crop = _post("/bepic_worlds/material_crop", {"name": name, "masks": masks, "label": _slug(surface)})
            patch = crop["patch"]
            if not description:
                description = _describe(patch, (
                    f"This is a patch of the {surface} in a photo. In one sentence, say only what the surface is made "
                    "of and how it looks (material, colour, wear, pattern, marks) — words for a texture prompt."),
                    _slug(name)) or surface
            texture = patch
            if refine:
                tex, _u, _m = _run_slot("texture_refine", {"image": patch, "prompt": _texture_prompt(description),
                                                           "denoise": float(denoise), "seed": int(seed)}, folder)
                texture = tex["texture"][0]
        maps, used, _m = _run_slot("material", {"image": texture}, folder)
        pick = {k: (maps.get(k) or [None])[0] for k in ("albedo", "normal", "roughness")}
        if not pick["albedo"]:
            raise RuntimeError("the material workflow gave no albedo")
        op = {"op": "set_material", "id": terrain_id, "layer": int(layer), "tile": float(tile_m),
              "normalScale": float(normal_strength), "description": description,
              **{k: v for k, v in pick.items() if v}}
        res = _post("/bepic_worlds/edit", {"name": name, "ops": [op],
                                           "note": f"{surface} material: {description} ({source}, {used})"})
        return _ok(applied=True, version=res.get("version"), description=description, patch=patch,
                   texture=texture, maps=pick)
    except Exception as exc:  # noqa: BLE001
        return _fail(exc)


@tool
def world_make_sky(name: str, description: str = "", seed: int = 7) -> str:
    """Give an outdoor world a generated 360° sky panorama (the sky slot),
    replacing the gradient sky. Interiors don't show their sky — skip them.

    Args:
        name: The world.
        description: The sky in words ("late afternoon, scattered cumulus,
            warm low sun, hazy horizon"). Empty = the vision agent describes the
            reference picture's sky.
        seed: Variant.
    """
    try:
        if not description:
            description = _describe(_stage_reference(name), (
                "Describe only the sky in this photo in one sentence for an image prompt: time of day, clouds, "
                "colours, haze, where the light comes from."), _slug(name)) or "a clear sky"
        prompt = (f"equirectangular 360 image, 360 panorama of the open sky: {description}. The horizon runs "
                  "straight across the middle, open land below it, no buildings, no text")
        pano, used, _m = _run_slot("sky", {"prompt": prompt, "seed": int(seed)}, f"worlds/{_slug(name)}/sky")
        res = _post("/bepic_worlds/edit", {"name": name, "ops": [{"op": "set_sky", "panorama": pano["panorama"][0]}],
                                           "note": f"sky: {description} ({used})"})
        return _ok(version=res.get("version"), description=description, panorama=pano["panorama"][0], workflow=used)
    except Exception as exc:  # noqa: BLE001
        return _fail(exc)


WORLD_TOOLS = [
    world_schema, world_create, world_rebuild, world_list, world_describe, world_edit, world_revert,
    world_feedback, world_resolve_feedback, world_calibrate, world_open,
    world_add_objects, world_add_props, world_make_material, world_make_sky,
    world_slots, world_choose_slot,
]
