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

**Not only its own slots.** The slots are the tried route, not a fence:
``world_find_templates`` searches the agent's whole template library (its own
templates and every official ComfyUI one), ``world_run_template`` runs any of
them on the world's picture or a file and hands back ComfyUI refs that the
edit ops take (a mesh for ``add_asset``, a panorama for ``set_sky`` …), and
``request_workflow`` (added by the pipeline, see src/pipeline.py) asks the
workflow researcher to pick and assemble one from a plain request.

**It says what it is doing.** A world takes minutes, most of them inside a
tool, so every tool pushes a plain-words line to the chat panel at each stage
(`_progress`: what it is finding, making, placing, and what came of it), and
`_run` reports a ComfyUI job while it waits: its place in the queue, that it
started, a heartbeat while it runs, and how long it took. The specialist's own
remarks between steps reach the panel too (NarrationHookProvider, src/agent.py).
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
    watch = _JobWatch(client, pid, label, t0)
    while time.time() - t0 < timeout:
        hist = client.get(f"/history/{pid}")
        if isinstance(hist, dict) and pid in hist:
            status = hist[pid].get("status") or {}
            if status.get("status_str") == "error":
                msgs = [m[1] for m in status.get("messages") or [] if m and m[0] == "execution_error"]
                detail = (msgs[0].get("exception_message") if msgs else "") or "execution failed"
                _progress(f"❌ {label} failed after {_took(t0)}")
                raise RuntimeError(f"{label}: {detail.strip()[:600]}")
            if status.get("completed", True):
                _progress(f"✅ {label} — done in {_took(t0)}")
                return hist[pid].get("outputs") or {}
        watch.tick()
        time.sleep(1.5)
    raise TimeoutError(f"{label}: no result after {int(timeout)} s (prompt {pid} is still queued or running)")


def _took(t0: float) -> str:
    """Seconds since t0, the way a person says it: 42 s, 3 min 5 s."""
    sec = int(time.time() - t0)
    return f"{sec} s" if sec < 60 else f"{sec // 60} min {sec % 60:02d} s"


class _JobWatch:
    """What a ComfyUI job is doing while `_run` waits on it, told to the panel.

    Waiting behind other jobs says how many are ahead (once, and again when that
    changes); starting says so; running says so again every half minute, so a
    three-minute mesh doesn't look like a hang. Reads /queue at most every 5 s.
    """

    HEARTBEAT = 30.0

    def __init__(self, client, pid: str, label: str, t0: float):
        self.client, self.pid, self.label, self.t0 = client, pid, label, t0
        self.state, self.ahead = None, None
        self.last_look = 0.0
        self.last_said = t0

    def tick(self) -> None:
        now = time.time()
        if now - self.last_look < 5.0:
            return
        self.last_look = now
        try:
            q = self.client.get("/queue") or {}
        except Exception:  # noqa: BLE001
            return
        running = [e[1] for e in q.get("queue_running") or [] if len(e) > 1]
        pending = [e[1] for e in q.get("queue_pending") or [] if len(e) > 1]
        if self.pid in running:
            if self.state != "running":
                self.state = "running"
                self.last_said = now
                _progress(f"▶️ {self.label} — running")
            elif now - self.last_said >= self.HEARTBEAT:
                self.last_said = now
                _progress(f"⏳ {self.label} — still running ({_took(self.t0)})")
        elif self.pid in pending:
            ahead = len(running) + pending.index(self.pid)
            if self.state != "queued" or ahead != self.ahead:
                self.state, self.ahead = "queued", ahead
                self.last_said = now
                _progress(f"🕒 {self.label} — waiting in ComfyUI's queue ({ahead} ahead)")


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


# What each slot's job is called in the panel ("Depth map (depth_da2_16bit)").
_SLOT_WORDS = {
    "depth": "Depth map", "segment": "Segmentation", "image_to_3d": "3D model",
    "texture_refine": "Texture refinement", "texture_generate": "Texture", "material": "PBR maps",
    "sky": "Sky panorama", "object_image": "Object picture", "motion": "Motion loop",
}


def _run_slot(slot: str, inputs: dict, prefix: str, override: str = "",
              timeout: float = 1800, say: str = "") -> tuple[dict, str, dict]:
    """Run a slot's workflow. Returns ({output name: [ComfyUI refs]}, template name, meta).

    `say` is the line the panel shows as it starts ("Finding every car in the
    picture"); the job's own lines (queue, running, done) follow from `_run`."""
    name, wf, meta = _template_for(slot, override)
    wf, outs = _bind(wf, inputs, prefix)
    what = _SLOT_WORDS.get(slot, slot)
    _progress(f"⚙️ {say or what} — {name}")
    res = _run(wf, f"{what} ({name})", timeout=timeout)
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


def _describe(ref: dict | str, question: str, folder: str) -> str:
    """What the vision agent sees in a picture (a ComfyUI ref, or a file on
    this machine), or '' when it can't look."""
    try:
        from src.tools.image_handling import analyze_image
        path = ref if isinstance(ref, str) else _download(ref, folder)
        out = analyze_image(file_path=path, question=question)
        if isinstance(out, dict) and out.get("status") == "ok":
            return " ".join(c.get("text", "") for c in out.get("content") or []).strip()
    except Exception:  # noqa: BLE001
        pass
    return ""


def _texture_prompt(description: str) -> str:
    return (f"seamless tileable texture of {description}. Orthographic top-down view, flat even diffuse "
            "lighting, no shadows, no perspective, no objects, sharp fine surface detail, photorealistic")


def comfy_ref_for_path(path: str) -> dict | None:
    """A file on disk as ComfyUI names it: under its output folder it is an
    output ref; anywhere else it is uploaded into the input folder first."""
    try:
        from agenty_core.utils.comfyui_client import comfyui_output_dir
        out = comfyui_output_dir()
        rel = Path(path).resolve().relative_to(out.resolve()) if out else None
    except (ValueError, OSError, Exception):  # noqa: BLE001
        rel = None
    if rel is not None:
        parts = rel.as_posix().rpartition("/")
        return {"filename": parts[2], "subfolder": parts[0], "type": "output"}
    return _as_ref(str(path)) if os.path.isfile(str(path)) else None


# ── tools: the template library ──────────────────────────────────────────────

@tool
def world_find_templates(query: str, limit: int = 12) -> str:
    """Search the WHOLE workflow template library — the agent's own templates
    and every official ComfyUI template — for one that does a job the world
    tools don't: Meshy/Tripo/Rodin/Hunyuan image-to-3D of a whole picture,
    HDR or 360° skies, upscalers, relighting, other depth or segmentation
    models, video. Run the one you pick with world_run_template.

    Args:
        query: What it should do, in a few words ("image to 3d meshy",
            "360 panorama hdr", "upscale 4x").
        limit: At most this many matches.
    """
    try:
        from agenty_core.tools.comfyui import get_workflow_catalog
        catalog = json.loads(get_workflow_catalog()) or {}
        words = [w for w in str(query).lower().replace("-", " ").split() if len(w) > 1]
        scored = []
        for name, desc in catalog.items():
            hay = f"{name} {desc}".lower().replace("_", " ").replace("-", " ")
            score = sum(3 if w in name.lower() else 1 for w in words if w in hay)
            if score:
                scored.append((score, name, desc))
        scored.sort(key=lambda t: (-t[0], t[1]))
        hits = [{"template": n, "about": (d or "")[:240]} for _, n, d in scored[:max(1, int(limit))]]
        _progress(f"🔎 Templates for “{query}”: {len(hits)} match{'es' if len(hits) != 1 else ''}"
                  + (f" — {', '.join(h['template'] for h in hits[:3])}" if hits else ""))
        return _ok(query=query, matches=hits, total=len(catalog))
    except Exception as exc:  # noqa: BLE001
        return _fail(exc)


@tool
def world_run_template(template: str, name: str = "", image: str = "", prompt: str = "",
                       seed: int = 0, timeout_s: float = 2400) -> str:
    """Run ANY template from the library (world_find_templates) — or one of the
    pack's slot workflows by name — and get its files back as ComfyUI refs.

    Those refs go straight into world edits: a mesh as `glb` of an `add_asset`
    op, a panorama as `panorama` of `set_sky`, an image as input to the next
    template. Use it for what the world tools don't do — e.g. Meshy on the
    whole reference picture for one connected mesh of the entire scene.

    The template's LoadImage node(s) take `image`, its first prompt box takes
    `prompt`, its seed takes `seed`; its save nodes are the outputs. A template
    that needs more (two images, a mask, special settings) is a job for
    request_workflow instead.

    Args:
        template: The template's exact name.
        name: The world, when `image` should default to its reference picture.
        image: The input picture: "reference" (the world's picture, needs
            `name`), a ComfyUI ref as JSON, a filename in ComfyUI's input
            folder, or a local path. Empty = none (text-to-X templates).
        prompt: Text for its prompt box, if it has one.
        seed: Seed (0 = the template's own).
        timeout_s: How long to wait for it.
    """
    try:
        wf = _library_template(template)
        if wf is None:
            try:
                wf = _get("/bepic_worlds/slot_template", name=template)
            except Exception:  # noqa: BLE001
                wf = None
        if not wf:
            raise ValueError(f"no template '{template}' in the library or the pack (see world_find_templates)")
        inputs: dict = {}
        img = str(image or "").strip()
        if img.lower() == "reference" or (not img and name and any(
                n.get("class_type") == "LoadImage" for n in wf.values())):
            if not name:
                raise ValueError("image='reference' needs the world's name")
            inputs["image"] = _stage_reference(name)
        elif img:
            inputs["image"] = _as_ref(img)
        if prompt:
            inputs["prompt"] = prompt
        prefix = f"worlds/{_slug(name or 'library')}/{_slug(template)}"
        wf, outs = _bind(wf, inputs, prefix)
        if seed:
            for n in wf.values():
                for k in ("seed", "noise_seed"):
                    if isinstance(n.get("inputs", {}).get(k), int):
                        n["inputs"][k] = int(seed)
        _progress(f"⚙️ Running template {template}" + (f" on {name}'s picture" if inputs.get("image") and img.lower() in ("", "reference") else ""))
        res = _run(wf, f"Template {template}", timeout=float(timeout_s))
        files: dict[str, list] = {}
        for nid, node_out in res.items():
            oname = outs.get(nid, f"out_{nid}")
            for key, vals in (node_out or {}).items():
                if isinstance(vals, list):
                    got = [v for v in vals if isinstance(v, dict) and v.get("filename")]
                    if got:
                        files.setdefault(oname if len(node_out) == 1 else f"{oname}_{key}", []).extend(got)
        if not files:
            raise RuntimeError(f"template {template} ran but saved no files")
        _progress(f"📦 {sum(len(v) for v in files.values())} file(s) from {template}: "
                  + ", ".join(v[0]["filename"] for v in files.values())[:200])
        return _ok(template=template, files=files)
    except Exception as exc:  # noqa: BLE001
        return _fail(exc)


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
        _progress(f"🌍 New world '{name}' from {ref.get('filename') if isinstance(ref, dict) else reference}"
                  + (f" — {spec}" if spec else ""))
        d = str(depth or "").strip()
        if d.lower() in ("", "none", "no", "false"):
            _progress("• No depth map: the picture stands flat")
        elif d.lower() == "auto" or d.startswith("depth_") or not ("." in d or d.startswith("{")):
            files, used, _meta = _run_slot("depth", {"image": ref, "prefix": f"{_slug(name)}_depth"},
                                           f"worlds/{_slug(name)}/depth", "" if d.lower() == "auto" else d,
                                           say="Measuring how far away everything in the picture is")
            body["depth"] = files["depth"][0]
            body["note"] = f"created (depth: {used})"
        else:
            body["depth"] = _as_ref(d)
            _progress("• Using the depth map given")
        _progress("🏗️ Building the world: reading the picture, laying out ground, sky and light …")
        t0 = time.time()
        res = _post("/bepic_worlds/create", {k: v for k, v in body.items() if v is not None})
        _progress(f"✅ World '{res.get('name', name)}' built in {_took(t0)}"
                  + (f" (version {res['version']})" if res.get("version") else "") + " — open in the viewer")
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
        changed = ", ".join(f"{k} {v}" for k, v in (("tilt", pitch), ("fov", fov), ("spec", spec)) if v is not None)
        _progress(f"🏗️ Rebuilding '{name}'" + (f" with {changed}" if changed else "") + " …")
        res = _post("/bepic_worlds/rebuild", {k: v for k, v in body.items() if v is not None})
        _progress(f"✅ Rebuilt '{name}'" + (f" — version {res['version']}" if res.get("version") else ""))
        return _ok(**res)
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
        _progress(f"✏️ Editing '{name}': {note}")
        res = _post("/bepic_worlds/edit", {"name": name, "ops": parsed, "note": note})
        _progress(f"✅ Saved as version {res.get('version')}" if res.get("version") else "✅ Saved")
        return _ok(**res)
    except Exception as exc:  # noqa: BLE001
        return _fail(exc)


@tool
def world_revert(name: str, version: int, note: str = "") -> str:
    """Go back to an earlier version (saved as a new version; nothing is lost)."""
    try:
        _progress(f"↩️ Taking '{name}' back to version {int(version)}")
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
        _progress(f"📌 {len(entries)} {status or 'open'} note{'s' if len(entries) != 1 else ''} on '{name}'")
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
        _progress(f"☑️ Marking {len(ids or [])} note{'s' if len(ids or []) != 1 else ''} done: {reply}")
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
                note = ((d.get("history") or [{}])[-1]).get("note")
                _progress(f"✅ Look matched — {note}" if note else "✅ Look matched")
                return _ok(matched=True, version=d["version"], note=note)
        return _ok(matched=False, note="no new version yet — the match runs in the viewer, which must be open "
                                       "in a browser tab that is visible (a background tab doesn't render)")
    except Exception as exc:  # noqa: BLE001
        return _fail(exc)


@tool
def world_open(name: str, version: int = 0) -> str:
    """Show a world (or one of its versions) in the user's viewer."""
    try:
        _progress(f"👁️ Opening '{name}'" + (f" version {int(version)}" if version else "") + " in the viewer")
        return _ok(**_post("/bepic_worlds/open", {"name": name, "version": int(version) or None}))
    except Exception as exc:  # noqa: BLE001
        return _fail(exc)


# ── tools: real things from the picture, and made-up ones ────────────────────

def _make_mesh(name: str, label: str, crop: dict, seed: int) -> tuple[dict, str, bool]:
    """image_to_3d on one object crop → (GLB ref, workflow name, textured?)."""
    files, used, meta = _run_slot("image_to_3d", {"image": crop, "seed": int(seed)},
                                  f"worlds/{_slug(name)}/{label}", timeout=2400,
                                  say=f"Turning the {label} into a 3D model (takes a few minutes)")
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
                                   f"worlds/{_slug(name)}/seg_{label}",
                                   say=f"Finding every {label} in the picture (up to {n})")
        masks = seg.get("masks") or next(iter(seg.values()))
        crops = _post("/bepic_worlds/object_crops", {"name": name, "label": label, "masks": masks, "limit": n, "crops": 1})
        objs = crops.get("objects") or []
        _progress(f"🔍 Found {len(objs)} {label}{'s' if len(objs) != 1 else ''}"
                  + (f"; the clearest one becomes the model" if objs else ""))
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
        if len(placed) < len(objs):
            _progress(f"• {len(objs) - len(placed)} of them can't stand on the ground (clipped, or past the horizon) — skipped")
        glb, used, textured = _make_mesh(name, label, best["crop"], seed)
        _progress(f"📍 Standing {len(placed)} {label}{'s' if len(placed) != 1 else ''} where the picture shows them, "
                  f"textured by {'the model' if textured else 'the picture'}")
        op = {"op": "add_asset", "label": label, "glb": glb, "bboxes": [o["bbox"] for o in placed]}
        if textured:
            op["textured"] = True
        else:
            op["texture"] = best["texture"]
        res = _post("/bepic_worlds/edit", {"name": name, "ops": [op],
                                           "note": f"{len(placed)} {label}{'s' if len(placed) != 1 else ''} "
                                                   f"from the picture ({used})"})
        spots = [{"height_m": o["placement"].get("height"), "distance_m": o["placement"].get("distance")} for o in placed]
        _progress(f"✅ {len(placed)} {label}{'s' if len(placed) != 1 else ''} added — version {res.get('version')}")
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
                                   f"worlds/{_slug(name)}/prop_{label}", say=f"Picturing {description}")
        image = (pic.get("image") or [None])[0]
        mask = (pic.get("mask") or [None])[0]
        if not image or not mask:
            raise RuntimeError("the object_image workflow must give an 'image' and a 'mask'")
        _progress("✂️ Cutting it out of its background")
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
        where = f"at {len(positions)} spot{'s' if len(positions) != 1 else ''}" if positions \
            else f"scattered {op['scatter']['count']}× within {radius_m:g} m"
        _progress(f"📍 Placing it {where}, {height_m:g} m tall")
        res = _post("/bepic_worlds/edit", {"name": name, "ops": [op], "note": f"{label}: {description} ({used})"})
        _progress(f"✅ {label} added — version {res.get('version')}")
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
            tex, _u, _m = _run_slot("texture_generate", {"prompt": _texture_prompt(description), "seed": int(seed)},
                                    folder, say=f"Generating a seamless texture: {description}")
            texture = tex["texture"][0]
        else:
            ref = _stage_reference(name)
            seg, _u, _m = _run_slot("segment", {"image": ref, "prompt": _slug(surface).replace("_", " "),
                                                "individual": False}, f"{folder}_seg",
                                    say=f"Finding the {surface} in the picture")
            masks = seg.get("masks") or next(iter(seg.values()))
            crop = _post("/bepic_worlds/material_crop", {"name": name, "masks": masks, "label": _slug(surface)})
            patch = crop["patch"]
            _progress(f"✂️ Cut the clearest near patch of {surface}")
            if not description:
                _progress("👀 Looking at what the surface is made of …")
                description = _describe(patch, (
                    f"This is a patch of the {surface} in a photo. In one sentence, say only what the surface is made "
                    "of and how it looks (material, colour, wear, pattern, marks) — words for a texture prompt."),
                    _slug(name)) or surface
                _progress(f"📝 It is: {description}")
            texture = patch
            if refine:
                tex, _u, _m = _run_slot("texture_refine", {"image": patch, "prompt": _texture_prompt(description),
                                                           "denoise": float(denoise), "seed": int(seed)}, folder,
                                        say="Cleaning it into a sharp, seamless texture")
                texture = tex["texture"][0]
        maps, used, _m = _run_slot("material", {"image": texture}, folder,
                                   say="Making albedo, normal and roughness maps")
        pick = {k: (maps.get(k) or [None])[0] for k in ("albedo", "normal", "roughness")}
        if not pick["albedo"]:
            raise RuntimeError("the material workflow gave no albedo")
        op = {"op": "set_material", "id": terrain_id, "layer": int(layer), "tile": float(tile_m),
              "normalScale": float(normal_strength), "description": description,
              **{k: v for k, v in pick.items() if v}}
        res = _post("/bepic_worlds/edit", {"name": name, "ops": [op],
                                           "note": f"{surface} material: {description} ({source}, {used})"})
        _progress(f"✅ {surface.capitalize()} material on, {tile_m:g} m per repeat — version {res.get('version')}")
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
            _progress("👀 Looking at the picture's sky …")
            description = _describe(_stage_reference(name), (
                "Describe only the sky in this photo in one sentence for an image prompt: time of day, clouds, "
                "colours, haze, where the light comes from."), _slug(name)) or "a clear sky"
            _progress(f"📝 Sky: {description}")
        prompt = (f"equirectangular 360 image, 360 panorama of the open sky: {description}. The horizon runs "
                  "straight across the middle, open land below it, no buildings, no text")
        pano, used, _m = _run_slot("sky", {"prompt": prompt, "seed": int(seed)}, f"worlds/{_slug(name)}/sky",
                                   say="Painting a 360° sky")
        res = _post("/bepic_worlds/edit", {"name": name, "ops": [{"op": "set_sky", "panorama": pano["panorama"][0]}],
                                           "note": f"sky: {description} ({used})"})
        _progress(f"✅ Sky set — version {res.get('version')}")
        return _ok(version=res.get("version"), description=description, panorama=pano["panorama"][0], workflow=used)
    except Exception as exc:  # noqa: BLE001
        return _fail(exc)


@tool
def world_add_motion(name: str, what: str, motion: str = "", seconds: float = 5.0, seed: int = 7) -> str:
    """Bring part of the picture to life: a short seamless loop (the motion
    slot: Wan 2.2, starting and ending on the picture itself) played on the
    world's hero view, only where `what` is (the segment slot finds it) —
    water rippling, leaves stirring, a flag, flickering light, drifting clouds.
    Seen best from the reference camera, where the hero view is. Takes minutes.

    Args:
        name: The world.
        what: What moves, as segmentation should find it ("water", "trees",
            "flag", "clouds"). "all" = the whole picture may move.
        motion: How it moves, in words ("small ripples drifting left, glints of
            light"). Empty = gentle natural motion of `what`.
        seconds: Loop length, 2-5 s.
        seed: Variant.
    """
    try:
        staged = _post("/bepic_worlds/stage_reference", {"name": name})
        ref, (w0, h0) = staged["reference"], staged.get("size") or (1280, 720)
        scale = 832 / max(w0, h0)
        width, height = max(16, int(w0 * scale) // 16 * 16), max(16, int(h0 * scale) // 16 * 16)
        length = max(17, min(81, int(round(float(seconds) * 16 / 4)) * 4 + 1))
        mask = None
        if what.strip().lower() not in ("all", "everything", ""):
            seg, _u, _m = _run_slot("segment", {"image": ref, "prompt": what, "individual": False},
                                    f"worlds/{_slug(name)}/motion_seg", say=f"Finding the {what} in the picture")
            mask = (seg.get("masks") or next(iter(seg.values())))[0]
        prompt = (f"{motion or f'gentle, natural motion of the {what}'}. Locked-off tripod shot, the camera does "
                  f"not move at all. Only the {what} moves; everything else stays perfectly still. A seamless loop.")
        vid, used, _m = _run_slot("motion", {"image": ref, "prompt": prompt, "width": width, "height": height,
                                             "length": length, "seed": int(seed)},
                                  f"worlds/{_slug(name)}/motion", timeout=3600,
                                  say=f"Animating the {what}: a {length}-frame seamless loop (takes several minutes)")
        video = (vid.get("video") or next(iter(vid.values())))[0]
        op = {"op": "set_motion", "video": video}
        if mask:
            op["mask"] = mask
        res = _post("/bepic_worlds/edit", {"name": name, "ops": [op], "note": f"motion: {what} ({used})"})
        _progress(f"✅ The {what} moves now — version {res.get('version')}")
        return _ok(version=res.get("version"), video=video, mask=mask, size=[width, height], frames=length,
                   workflow=used)
    except Exception as exc:  # noqa: BLE001
        return _fail(exc)


# ── tools: the whole picture as one model, and a real sky ─────────────────────

# Library templates that make a mesh from one picture, beyond the pack's own
# scene_model workflows. Name patterns, then what rules one out.
_I23D_NAME = ("image_to_model", "image_to_3d", "perspective_to_mesh")
_I23D_NOT = ("text_to", "multiview", "multi_image", "multi_views", "mv_to", "model2uv", "retopo", "part",
             "smart_topology", "gaussian_splat")
# Pack workflows already cover these templates (same engine, set up for a whole picture).
_I23D_COVERED = ("3d_pixal3d_trellis2_image_to_model", "3d_moge_perspective_to_mesh", "api_meshy_image_to_model",
                 "api_tripo3_1_image_to_model", "3d_hunyuan3d_image_to_model", "3d_hunyuan3d-v2.1")


def _scene_engines() -> dict:
    """{name: {about, frame, textured, cost, source}} — every way to make one
    model of the whole picture: the pack's scene_model workflows, then the
    image-to-3D templates of the library they don't already cover."""
    engines: dict = {}
    slot = (_slot_info().get("slots") or {}).get("scene_model") or {}
    for name, meta in (slot.get("templates") or {}).items():
        engines[name] = {**meta, "source": "pack", "default": name == slot.get("default")}
    try:
        from agenty_core.tools.comfyui import get_workflow_catalog
        catalog = json.loads(get_workflow_catalog()) or {}
    except Exception:  # noqa: BLE001
        catalog = {}
    for name, desc in sorted(catalog.items()):
        low = name.lower()
        if not any(p in low for p in _I23D_NAME) or any(p in low for p in _I23D_NOT) or name in _I23D_COVERED:
            continue
        engines[name] = {"about": (desc or "")[:200], "frame": "object", "textured": None,
                         "cost": "api" if low.startswith("api_") else "local", "source": "library"}
    return engines


@tool
def world_scene_model_options() -> str:
    """The ways to build a world around ONE 3D model of the whole picture —
    the engines to choose from (TRELLIS.2, Pixal3D, Hunyuan3D, SHARP, MoGe,
    Meshy, Tripo, and every other image-to-3D template in the library), each
    with what it gives, whether it costs API credits, and how it is placed.

    Let the USER choose: when they haven't named one, list these (briefly:
    name, what it gives, local or paid) and ask. "object" models are fitted
    to the picture (size from its buildings, turned and placed by measure);
    "camera" models (SHARP, MoGe) are built in the picture's own camera, so
    they line up with it exactly but hold only what the picture shows."""
    try:
        eng = _scene_engines()
        _progress(f"🧭 {len(eng)} ways to model the whole picture: " + ", ".join(list(eng)[:8]))
        return _ok(engines=eng, note="Ask the user which one unless they said. API engines cost credits.")
    except Exception as exc:  # noqa: BLE001
        return _fail(exc)


def _world_camera(name: str) -> dict:
    """The reference camera's vertical fov, picture size, and derived lens."""
    import math
    scene = _get("/bepic_worlds/world", name=name)
    cam = next((i for i in scene.get("items", []) if i.get("id") == "refcam"), {}) or {}
    w, h = cam.get("resolution") or [1920, 1080]
    vfov = float(cam.get("fov") or 50.0)
    hfov = math.degrees(2 * math.atan(math.tan(math.radians(vfov) / 2) * w / max(h, 1)))
    return {"vfov": vfov, "hfov": hfov, "focal_mm": 18.0 / math.tan(math.radians(hfov) / 2), "size": [w, h]}


@tool
def world_scene_model(name: str, engine: str, height_m: float = 0.0, yaw: float = -1.0,
                      remove_background: bool = True, seed: int = 7) -> str:
    """Make ONE 3D model of the whole reference picture and build the world
    around it: the model stands where the picture is (item `scene`), the
    terrain is flattened to meet its ground, the flat picture it replaces is
    hidden, and nothing grows inside it. Takes minutes.

    Args:
        name: The world (world_create first).
        engine: One of world_scene_model_options — "s3d_trellis2",
            "s3d_pixal3d", "s3d_hunyuan21", "s3d_sharp", "s3d_moge", "s3d_meshy",
            "s3d_tripo", or a library template's name. API engines (Meshy,
            Tripo, Rodin, api_*) cost credits: only when the user chose them.
        height_m: The height of the scene's tallest parts in metres (its
            buildings, trees) when you know it better than the picture's depth
            does — e.g. three-storey houses ≈ 10-12. 0 = measured from the picture.
        yaw: A fixed turn in degrees for an object model; -1 = found by fitting.
        remove_background: TRELLIS.2 / Pixal3D only — cut the background first
            (keep for one building; turn off for a whole street or landscape).
        seed: Variant.
    """
    try:
        eng = _scene_engines()
        meta = eng.get(engine)
        if meta is None:
            raise ValueError(f"no engine '{engine}' — see world_scene_model_options: {', '.join(eng)}")
        ref = _stage_reference(name)
        cam = _world_camera(name)
        folder = f"worlds/{_slug(name)}/scene"
        if meta.get("source") == "pack":
            files, used, _m = _run_slot(
                "scene_model",
                {"image": ref, "seed": int(seed), "remove_background": bool(remove_background),
                 "focal_length_mm": round(cam["focal_mm"], 2), "fov_x": round(cam["hfov"], 2)},
                folder, override=engine, timeout=3600,
                say=f"Modelling the whole picture in 3D with {engine} (takes minutes)")
            mesh = (files.get("mesh") or next(iter(files.values())))[0]
            frame = meta.get("frame") or "object"
        else:
            _progress(f"⚙️ Modelling the whole picture in 3D with the template {engine} (takes minutes)")
            wf = _library_template(engine)
            if wf is None:
                raise ValueError(f"no template '{engine}'")
            wf, outs = _bind(wf, {"image": ref}, folder)
            res = _run(wf, f"3D model ({engine})", timeout=3600)
            refs = [v for node in res.values() for vals in (node or {}).values() if isinstance(vals, list)
                    for v in vals if isinstance(v, dict) and str(v.get("filename", "")).lower().endswith((".glb", ".gltf", ".obj", ".ply"))]
            if not refs:
                raise RuntimeError(f"{engine} ran but saved no 3D model")
            mesh, used, frame = refs[0], engine, "object"
        op: dict = {"op": "set_scene_model", "glb": mesh, "frame": frame, "engine": used}
        if height_m and height_m > 0:
            op["height_m"] = float(height_m)
        if yaw is not None and yaw >= 0:
            op["yaw"] = float(yaw)
        _progress("📐 Placing it where the picture is — " + (
            "the camera it was built in puts it there" if frame != "object"
            else "sizing it from the picture's buildings, then turning and moving it to fit"))
        res = _post("/bepic_worlds/edit", {"name": name, "ops": [op], "note": f"scene model: {used}"}, timeout=900)
        scene = _get("/bepic_worlds/world", name=name)
        item = next((i for i in scene.get("items", []) if i.get("id") == "scene"), {}) or {}
        fit = item.get("scene_model") or {}
        size = fit.get("size_m")
        said = f"✅ Scene model in — version {res.get('version')}"
        if size:
            said += f"; {size[0]:.0f} × {size[1]:.0f} × {size[2]:.0f} m"
        if fit.get("error_m") is not None:
            said += f", fits the picture to {fit['error_m']:.2f} m on {fit.get('coverage', 0) * 100:.0f}% of it"
        _progress(said)
        return _ok(version=res.get("version"), engine=used, frame=frame, mesh=mesh, fit=fit,
                   check=("Is the size right? size_m is width × height × depth in metres. A street of houses "
                          "should be ~8-15 m tall; if not, call again with height_m. A poor fit (error_m over "
                          "~1.5 m or coverage under ~0.4) means the engine reshaped the scene; say so, or try "
                          "another engine."))
    except Exception as exc:  # noqa: BLE001
        return _fail(exc)


def _contact_sheet(reference: str, thumbs: list, out: str) -> str:
    """The reference on top, numbered panoramas below — one picture for the
    vision agent to compare."""
    from PIL import Image, ImageDraw
    ref = Image.open(reference).convert("RGB")
    ref.thumbnail((1024, 420))
    cells = []
    for i, path in enumerate(thumbs, 1):
        im = Image.open(path).convert("RGB").resize((500, 250))
        d = ImageDraw.Draw(im)
        d.rectangle([0, 0, 44, 36], fill=(0, 0, 0))
        d.text((12, 8), str(i), fill=(255, 255, 255))
        cells.append(im)
    cols = 2
    rows = (len(cells) + cols - 1) // cols
    sheet = Image.new("RGB", (1024, ref.height + 12 + rows * 262), (255, 255, 255))
    sheet.paste(ref, ((1024 - ref.width) // 2, 0))
    for i, im in enumerate(cells):
        sheet.paste(im, (6 + (i % cols) * 512, ref.height + 12 + (i // cols) * 262))
    sheet.save(out)
    return out


def _read_json(text: str) -> dict:
    import re
    m = re.search(r"\{.*\}", text or "", re.S)
    try:
        return json.loads(m.group(0)) if m else {}
    except ValueError:
        return {}


@tool
def world_environment(name: str, description: str = "", resolution: str = "4k", generate: bool = True,
                      indoor_ok: bool = False) -> str:
    """Give a world a REAL environment: a photographed HDRI from the web (Poly
    Haven, CC0) that matches the picture — its sky, its light and its
    reflections, with the world's sun turned to where the picture's is. Only
    when none fits is a high-resolution 360° sky generated instead (the sky
    slot, 8192×4096). Prefer this to world_make_sky.

    Args:
        name: The world.
        description: The sky and light in words ("overcast grey noon, soft
            light, old town street"). Empty = the vision agent reads the picture.
        resolution: HDRI size: "2k", "4k" (default), "8k" (heavy in a browser).
        generate: Make a sky when no HDRI fits (default). False = report instead.
        indoor_ok: Interiors don't show a sky; set true to light one with an
            indoor HDRI anyway.
    """
    try:
        import tempfile
        from pathlib import Path
        ref = _stage_reference(name)
        want: dict = {}
        if not description:
            _progress("👀 Reading the picture's sky and light …")
            want = _read_json(_describe(ref, (
                "Look at this photo's sky and light. Answer ONLY with JSON: {\"indoor\": true|false, "
                "\"time_of_day\": \"sunrise\"|\"midday\"|\"sunset\"|\"night\"|\"morning-afternoon\", "
                "\"weather\": \"clear\"|\"partly_cloudy\"|\"overcast\"|\"foggy\", \"urban\": true|false, "
                "\"words\": \"a few words on the sky, light and place\"}"), _slug(name)))
            description = want.get("words") or ""
            if want:
                seen = ["indoors" if want.get("indoor") else "outdoors",
                        str(want.get("time_of_day") or "").replace("_", " "),
                        str(want.get("weather") or "").replace("_", " "),
                        {True: "town", False: "nature"}.get(want.get("urban"), "")]
                _progress("📝 " + ", ".join(w for w in seen if w) + (f" — {description}" if description else ""))
        if want.get("indoor") and not indoor_ok:
            return _ok(applied=False, note="an interior: its sky isn't seen, so no environment was set "
                                           "(indoor_ok=true lights it with an indoor HDRI)")
        params = {"query": description, "limit": 6,
                  "time_of_day": want.get("time_of_day") or None, "weather": want.get("weather") or None,
                  "environment": "indoor" if want.get("indoor") else "outdoor"}
        if not want.get("indoor"):
            # The world brings its own buildings and land; from the panorama it
            # wants sky and distance, not another town's walls around it.
            params["open_sky"] = "true"
        _progress("🔎 Looking for a matching HDRI on Poly Haven …")
        found = _get("/bepic_worlds/hdri_search", **params).get("matches") or []
        pick = None
        if found:
            tmp = Path(tempfile.gettempdir()) / "agenty_worlds" / _slug(name) / "hdri"
            tmp.mkdir(parents=True, exist_ok=True)
            thumbs = []
            for m in found:
                try:
                    r = requests.get(m["thumbnail"], timeout=30)
                    r.raise_for_status()
                    p = tmp / f"{m['id']}.png"
                    p.write_bytes(r.content)
                    thumbs.append((m, str(p)))
                except Exception:  # noqa: BLE001
                    continue
            if thumbs:
                sheet = _contact_sheet(_download(ref, _slug(name)), [p for _, p in thumbs], str(tmp / "choices.jpg"))
                answer = _describe(sheet, (
                    "The photo at the top is a scene. Below are numbered 360° HDR panoramas. The scene keeps its own "
                    "buildings and ground; the panorama gives it its sky, light and far distance. Which ONE fits "
                    "most believably — same time of day, weather, cloud and sun hardness, and nothing close by that "
                    "would tower over the scene? Answer with just its number, or 0 if none is a reasonable match."),
                    _slug(name))
                import re
                num = re.search(r"\d+", answer or "")
                if num is None:
                    pick = thumbs[0][0]                      # no eyes: the best-scored one
                elif 1 <= int(num.group(0)) <= len(thumbs):
                    pick = thumbs[int(num.group(0)) - 1][0]
        if pick:
            _progress(f"🌅 {pick['name']} fits — downloading it ({resolution}) and lighting the world with it")
            res = _post("/bepic_worlds/hdri", {"name": name, "id": pick["id"], "resolution": resolution}, timeout=900)
            sun = res.get("sun") or {}
            _progress(f"✅ Environment: {pick['name']} — version {res.get('version')}"
                      + (f", sun at {sun.get('azimuth')}° / {sun.get('elevation')}° up" if sun.get("shadows") else ", no direct sun"))
            return _ok(applied=True, source="Poly Haven (CC0)", hdri=pick["id"], title=pick["name"],
                       page=pick["page"], version=res.get("version"), sun=sun)
        if not generate:
            return _ok(applied=False, candidates=[m["id"] for m in found],
                       note="no HDRI on Poly Haven fits the picture")
        _progress("• No photographed HDRI fits — generating a high-resolution sky instead")
        made = json.loads(world_make_sky(name=name, description=description))
        return _ok(applied=made.get("status") == "ok", source="generated", sky=made)
    except Exception as exc:  # noqa: BLE001
        return _fail(exc)


WORLD_TOOLS = [
    world_schema, world_create, world_rebuild, world_list, world_describe, world_edit, world_revert,
    world_feedback, world_resolve_feedback, world_calibrate, world_open,
    world_scene_model_options, world_scene_model, world_environment,
    world_add_objects, world_add_props, world_make_material, world_make_sky, world_add_motion,
    world_slots, world_choose_slot, world_find_templates, world_run_template,
]
