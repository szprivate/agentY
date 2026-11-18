"""agentY farm helper: the part of a rented farm machine that agentY talks to besides ComfyUI itself.

* ``POST /farm/fetch`` ``{"items": [{"folder": "diffusion_models", "name": "x.safetensors", "url": "https://..."}]}``
  downloads model files into ComfyUI's model folders in the background (``.part`` first, renamed when complete);
  ``GET /farm/fetch`` lists them with ``state`` (queued, downloading, done, failed), ``bytes``, ``total``, ``error``.
  A file that is already there is reported done at once.
* When the machine was started with ``COMFY_TOKEN`` (agentY's farm sets one per machine), every request must carry
  it: ``Authorization: Bearer <token>``, or once ``?token=<token>`` in a browser (kept as a cookie). Without the
  token the machine's port would be open to anyone who finds it.

It registers no nodes. Only https URLs are fetched; HF_TOKEN, when set on the machine, is sent to huggingface.co.
"""
from __future__ import annotations

import hmac
import logging
import os
import threading
import time
import urllib.request
from pathlib import Path

from aiohttp import web

import folder_paths
from server import PromptServer

NODE_CLASS_MAPPINGS: dict = {}
NODE_DISPLAY_NAME_MAPPINGS: dict = {}

log = logging.getLogger("agenty_farm_helper")
TOKEN = os.environ.get("COMFY_TOKEN", "").strip()
COOKIE = "agenty_farm"
CHUNK = 8 << 20
_items: dict = {}
_lock = threading.Lock()
_queue: list = []
_worker = None


# ── token ────────────────────────────────────────────────────────────────────
def _ok(request) -> bool:
    if not TOKEN:
        return True
    auth = request.headers.get("Authorization", "")
    for got in (auth[7:] if auth.startswith("Bearer ") else "", request.cookies.get(COOKIE, ""),
                request.query.get("token", "")):
        if got and hmac.compare_digest(got, TOKEN):
            return True
    return False


@web.middleware
async def _token_gate(request, handler):
    if not _ok(request):
        return web.Response(status=401, text="This farm machine needs its token.")
    resp = await handler(request)
    if TOKEN and request.query.get("token") and not request.cookies.get(COOKIE):
        try:
            resp.set_cookie(COOKIE, TOKEN, httponly=True, samesite="Strict")
        except Exception:  # noqa: BLE001 - websocket and streamed responses
            pass
    return resp


if TOKEN:
    try:
        PromptServer.instance.app.middlewares.insert(0, _token_gate)
        log.info("agentY farm helper: requests need the machine's token")
    except Exception as exc:  # noqa: BLE001
        log.error("agentY farm helper: could not install the token check: %s", exc)


# ── model downloads ──────────────────────────────────────────────────────────
def _target(folder: str, name: str) -> Path:
    if folder not in folder_paths.folder_names_and_paths:
        raise ValueError(f"unknown model folder {folder!r}")
    rel = Path(str(name).replace("\\", "/"))
    if rel.is_absolute() or ".." in rel.parts or not rel.name:
        raise ValueError(f"bad file name {name!r}")
    for base in folder_paths.get_folder_paths(folder):
        if (Path(base) / rel).is_file():
            return Path(base) / rel
    return Path(folder_paths.get_folder_paths(folder)[0]) / rel


def _download(it: dict) -> None:
    dest = Path(it["path"])
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    headers = {"User-Agent": "agentY-farm-helper/1"}
    hf = os.environ.get("HF_TOKEN", "").strip()
    if hf and urllib.request.urlparse(it["url"]).hostname in ("huggingface.co", "hf.co"):
        headers["Authorization"] = f"Bearer {hf}"
    req = urllib.request.Request(it["url"], headers=headers)
    with urllib.request.urlopen(req, timeout=60) as r, open(part, "wb") as fh:
        it["total"] = int(r.headers.get("Content-Length") or 0)
        while True:
            chunk = r.read(CHUNK)
            if not chunk:
                break
            fh.write(chunk)
            it["bytes"] += len(chunk)
    if it["total"] and it["bytes"] != it["total"]:
        raise IOError(f"got {it['bytes']} of {it['total']} bytes")
    os.replace(part, dest)


def _work() -> None:
    while True:
        with _lock:
            it = _queue.pop(0) if _queue else None
        if it is None:
            time.sleep(0.5)
            continue
        it["state"] = "downloading"
        t0 = time.time()
        try:
            _download(it)
            it["state"] = "done"
            log.info("agentY farm helper: %s/%s in %.0fs", it["folder"], it["name"], time.time() - t0)
        except Exception as exc:  # noqa: BLE001
            it.update(state="failed", error=str(exc)[:300])
            log.error("agentY farm helper: %s/%s failed: %s", it["folder"], it["name"], exc)


def _public(it: dict) -> dict:
    return {k: it[k] for k in ("folder", "name", "state", "bytes", "total", "error")}


routes = PromptServer.instance.routes


@routes.post("/farm/fetch")
async def farm_fetch(request):
    global _worker
    body = await request.json()
    out = []
    for raw in (body or {}).get("items") or []:
        folder, name, url = str(raw.get("folder") or ""), str(raw.get("name") or ""), str(raw.get("url") or "")
        key = f"{folder}/{name}"
        try:
            if not url.startswith("https://"):
                raise ValueError("only https:// addresses are fetched")
            path = _target(folder, name)
        except ValueError as exc:
            out.append({"folder": folder, "name": name, "state": "failed", "bytes": 0, "total": 0, "error": str(exc)})
            continue
        with _lock:
            cur = _items.get(key)
            if path.is_file():
                size = path.stat().st_size
                _items[key] = cur = {"folder": folder, "name": name, "url": url, "path": str(path), "state": "done",
                                     "bytes": size, "total": size, "error": ""}
            elif not cur or cur["state"] == "failed":
                _items[key] = cur = {"folder": folder, "name": name, "url": url, "path": str(path), "state": "queued",
                                     "bytes": 0, "total": 0, "error": ""}
                _queue.append(cur)
            if _worker is None or not _worker.is_alive():
                _worker = threading.Thread(target=_work, name="agenty-farm-fetch", daemon=True)
                _worker.start()
        out.append(_public(cur))
    return web.json_response({"ok": True, "items": out})


@routes.get("/farm/fetch")
async def farm_fetch_status(request):
    with _lock:
        return web.json_response({"items": [_public(it) for it in _items.values()]})


@routes.get("/farm/helper")
async def farm_helper(request):
    return web.json_response({"ok": True, "version": 1, "token": bool(TOKEN)})
