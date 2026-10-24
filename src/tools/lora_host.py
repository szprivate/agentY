"""Host side of the LoRA tools (agenty_core.tools.lora_training): the dataset captioner and the training watcher.

* Captioner: caption writing is a vision call, so the host lends the tool layer its Vision / Video agent
  (``analyze_image`` / ``analyze_video``). Without it (agentY-mcp) the tools return ``needs_captions`` and the
  calling model writes them.
* Watcher: a trainer runs in a detached process (it survives this host). A daemon thread reads the runs' new
  ``events.jsonl`` lines and raises them on :mod:`src.utils.notify_bus` (toast / browser notification) - start,
  each sample, checkpoint, finish, failure. Runs that were already finished when the host started are not replayed.
"""
from __future__ import annotations

import logging
import threading
import time

logger = logging.getLogger("agentY.lora_host")

_started = False
_lock = threading.Lock()
_seen: dict[str, int] = {}
POLL_S = 10.0


def _text(res) -> str:
    if isinstance(res, dict):
        c = res.get("content") or []
        if c and isinstance(c[0], dict):
            return str(c[0].get("text", ""))
        return ""
    return str(res or "")


def caption_with_vision(path: str, instruction: str) -> str:
    """The Vision Agent's caption for one picture (the Video Agent for a clip)."""
    if str(path).lower().endswith((".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v")):
        from src.tools.video_handling import analyze_video
        res = analyze_video(file_path=path, question=instruction, max_frames=8)
    else:
        from src.tools.image_handling import analyze_image
        res = analyze_image(file_path=path, question=instruction, mode="describe")
    if isinstance(res, dict) and res.get("status") == "error":
        raise RuntimeError(_text(res)[:200])
    return _text(res)


def _runs_root():
    from agenty_core.tools.lora_training import formats as F
    return F.runs_root()


def _scan(first: bool) -> None:
    from agenty_core.tools.lora_training import formats as F
    from src.utils import notify_bus
    root = _runs_root()
    for d in root.iterdir():
        if not (d / "progress.json").exists():
            continue
        evs = F.read_events(d)
        last = evs[-1]["id"] if evs else 0
        if d.name not in _seen:
            _seen[d.name] = last if first else 0       # runs from before this host started are not replayed
        for e in F.read_events(d, _seen[d.name]):
            level = "error" if e["kind"] == "failed" else "info"
            try:
                notify_bus.emit({"kind": "toast", "toast": {"title": f"LoRA training - {d.name}", "body": e["text"][:300],
                                                            "level": level}})
            except Exception:  # noqa: BLE001
                logger.debug("notify failed", exc_info=True)
            _seen[d.name] = e["id"]


def _loop() -> None:
    first = True
    while True:
        try:
            _scan(first)
            first = False
        except Exception as exc:  # noqa: BLE001  (ComfyUI offline: the output folder is unknown - try again)
            logger.debug("lora watch: %s", exc)
        time.sleep(POLL_S)


def install() -> None:
    """Lend the captioner and start the watcher (idempotent)."""
    global _started
    with _lock:
        if _started:
            return
        _started = True
    try:
        from agenty_core.tools.lora_training import set_captioner
        set_captioner(caption_with_vision)
    except Exception:  # noqa: BLE001
        logger.warning("could not register the LoRA captioner", exc_info=True)
    threading.Thread(target=_loop, name="lora-watch", daemon=True).start()
