"""The rating page: collect the user's picks between sibling renders.

The fitness weights are learned from *choices* — this one, not those — and the
only place those were ever recorded is a review halt, which this user had never
answered. So there were no labels, and the "fit to your taste" step had nothing
to fit.

This page asks for them directly, in the one form the fit is built for: a slate.
It shows a few renders from the SAME folder (siblings of one shot — the
comparison the fitness score is actually used for) and the user clicks the best.
That is one chosen against the rest, written to ``preference_log`` exactly like a
review, with ``source: "rating"``.

Files are handed to the page by an opaque id, never by path: the page can only
ask for images this module chose to show it. They are served downscaled — the
originals live on a network share and are several megabytes each.
"""

from __future__ import annotations

import io
import logging
import os
import random
import secrets
import threading
import time
from pathlib import Path

logger = logging.getLogger("agentY.rating")

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp"}
SLATE_SIZE = 4
MIN_SIBLINGS = 3          # a folder with fewer is not a choice between siblings
PREVIEW_EDGE = 1280       # long edge of the served preview, in pixels
_SKIP_DIRS = {"temp", "_thumbs", "thumbnails", ".cache", "__pycache__", "agent"}
_MAX_FOLDERS = 5000       # stop the walk somewhere on a very large share

_lock = threading.Lock()
_state = {"folders": [], "scanned": False, "scanning": False, "error": "",
          "roots": [], "started": 0.0}
_ids: dict = {}           # id -> absolute path, only for files we have shown
_rated: set = set()       # paths already judged, this run or in the log


def roots() -> list[Path]:
    """Where renders live: ComfyUI's output folder and agentY's own images."""
    out: list[Path] = []
    try:
        from src.executor import _get_comfyui_output_dir
        comfy = _get_comfyui_output_dir()
        if comfy is not None:
            out.append(Path(comfy))
    except Exception as exc:  # noqa: BLE001
        logger.debug("rating: no ComfyUI output dir (%s)", exc)
    own = Path(__file__).resolve().parent.parent.parent / "output" / "agent" / "images"
    if own.exists():
        out.append(own)
    return out


def scan_folders(root_paths) -> list[dict]:
    """Every folder under *root_paths* holding at least ``MIN_SIBLINGS`` images."""
    found: list[dict] = []
    for root in root_paths:
        stack = [Path(root)]
        while stack and len(found) < _MAX_FOLDERS:
            here = stack.pop()
            images: list[str] = []
            try:
                with os.scandir(here) as it:
                    for de in it:
                        try:
                            if de.is_dir(follow_symlinks=False):
                                if de.name.lower() not in _SKIP_DIRS and not de.name.startswith("."):
                                    stack.append(Path(de.path))
                            elif Path(de.name).suffix.lower() in IMAGE_SUFFIXES:
                                images.append(de.path)
                        except OSError:
                            continue
            except OSError:
                continue
            if len(images) >= MIN_SIBLINGS:
                found.append({"folder": str(here), "images": sorted(images)})
    return found


def _already_rated() -> set:
    """Paths that already appear in a logged rating or review."""
    try:
        from src.utils.preference_log import read_events
        seen = set()
        for ev in read_events():
            for side in ("chosen", "rejected"):
                for row in ev.get(side) or []:
                    if row.get("path"):
                        seen.add(str(row["path"]))
        return seen
    except Exception:  # noqa: BLE001
        return set()


def start_scan(force: bool = False) -> None:
    """Scan in the background; the page polls ``status`` until it is done."""
    with _lock:
        if _state["scanning"] or (_state["scanned"] and not force):
            return
        _state.update(scanning=True, error="", started=time.time())

    def work():
        try:
            rs = roots()
            folders = scan_folders(rs)
            rated = _already_rated()
            with _lock:
                _state.update(folders=folders, roots=[str(r) for r in rs], scanned=True)
                _rated.update(rated)
        except Exception as exc:  # noqa: BLE001
            logger.warning("rating: scan failed — %s", exc)
            with _lock:
                _state["error"] = str(exc)
        finally:
            with _lock:
                _state["scanning"] = False

    threading.Thread(target=work, name="agentY-rating-scan", daemon=True).start()


def status() -> dict:
    """How far the scan is, and how much there is to rate."""
    with _lock:
        folders = list(_state["folders"])
        out = {"scanned": _state["scanned"], "scanning": _state["scanning"],
               "error": _state["error"], "roots": list(_state["roots"]),
               "folders": len(folders),
               "images": sum(len(f["images"]) for f in folders),
               "unrated": sum(1 for f in folders for p in f["images"] if p not in _rated)}
    try:
        from src.utils.preference_log import summary
        out["log"] = summary()
    except Exception:  # noqa: BLE001
        out["log"] = ""
    return out


def next_slate(rng: random.Random | None = None) -> dict | None:
    """A few unrated siblings from one folder, each under a fresh opaque id."""
    rng = rng or random
    with _lock:
        pools = [(f["folder"], [p for p in f["images"] if p not in _rated])
                 for f in _state["folders"]]
        pools = [(folder, imgs) for folder, imgs in pools if len(imgs) >= 2]
        if not pools:
            return None
        # Weighted by what is left, so a big folder is not exhausted last.
        folder, imgs = rng.choices(pools, weights=[len(i) for _, i in pools])[0]
        pick = rng.sample(imgs, min(SLATE_SIZE, len(imgs)))
        items = []
        for path in pick:
            ident = secrets.token_urlsafe(9)
            _ids[ident] = path
            items.append({"id": ident, "name": Path(path).name})
    return {"folder": folder, "label": _label(folder), "items": items}


def _label(folder: str) -> str:
    """The last few path parts — enough to recognise the shot."""
    parts = Path(folder).parts
    return "/".join(parts[-3:])


def path_of(ident: str) -> str | None:
    with _lock:
        return _ids.get(str(ident))


def preview(ident: str) -> bytes | None:
    """A downscaled JPEG of a file we showed, or None for an unknown id."""
    path = path_of(ident)
    if not path:
        return None
    from PIL import Image
    with Image.open(path) as im:
        im = im.convert("RGB")
        im.thumbnail((PREVIEW_EDGE, PREVIEW_EDGE))
        buf = io.BytesIO()
        im.save(buf, "JPEG", quality=88)
    return buf.getvalue()


def record_pick(chosen: str, shown: list, folder: str = "") -> dict:
    """The user picked *chosen* out of *shown*: log it as one slate.

    Measuring files on a network share takes a moment, so the write happens off
    the request; the page moves on at once.
    """
    keep = path_of(chosen)
    rest = [p for p in (path_of(i) for i in shown or []) if p and p != keep]
    if not keep or not rest:
        return {"ok": False, "error": "unknown images — reload the page"}
    with _lock:
        _rated.update([keep, *rest])

    def work():
        from src.utils.preference_log import record_review
        record_review([keep], rest, source="rating", request=_label(folder or
                      str(Path(keep).parent)))
    threading.Thread(target=work, name="agentY-rating-log", daemon=True).start()
    return {"ok": True, "pairs": len(rest)}


def skip(shown: list) -> None:
    """Can't decide: don't show these again this session, log nothing."""
    with _lock:
        _rated.update(p for p in (_ids.get(str(i)) for i in shown or []) if p)
