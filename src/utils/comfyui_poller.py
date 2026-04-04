"""
comfyui_poller – Zero-token async polling for ComfyUI job completion.

Extracted from pipeline.py so both the Pipeline (interrupt-resume flow) and
the Executor (post-Brain execution flow) can share the same polling logic
without a circular import.
"""

from __future__ import annotations

import asyncio
import logging

logger = logging.getLogger("agentY.comfyui_poller")


def _load_poll_settings() -> tuple[float, float]:
    """Return ``(poll_interval_s, poll_timeout_s)`` from settings.json."""
    import json
    from pathlib import Path

    path = Path(__file__).parent.parent.parent / "config" / "settings.json"
    cfg: dict = {}
    if path.exists():
        try:
            cfg = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            pass
    interval = float(cfg.get("comfyui_poll_interval_s", 3))
    timeout = float(cfg.get("comfyui_poll_timeout_s", 30 * 60))
    return interval, timeout


async def poll_comfyui_job(prompt_id: str) -> dict:
    """Poll ``GET /history/{prompt_id}`` until the job completes.

    Sleeps between checks using ``asyncio.sleep`` so no LLM tokens are burned
    while waiting.  Returns the stripped history dict on success, or an error
    dict if the timeout is exceeded or the job reports an error status.

    Args:
        prompt_id: The ComfyUI prompt ID returned by ``submit_prompt``.

    Returns:
        A dict with the job result on success, or ``{"error": "..."}`` on
        timeout / ComfyUI error.
    """
    from src.utils.comfyui_client import get_client
    from src.tools.comfyui import _strip_history

    interval, timeout = _load_poll_settings()
    client = get_client()
    waited: float = 0.0

    logger.info(
        "poller: start polling prompt_id=%s (interval=%.1fs, timeout=%.0fs)",
        prompt_id, interval, timeout,
    )

    while waited < timeout:
        await asyncio.sleep(interval)
        waited += interval

        try:
            raw = client.get(f"/history/{prompt_id}")
        except Exception as exc:
            logger.warning("poller: HTTP error after %.0fs — %s", waited, exc)
            continue

        if not isinstance(raw, dict) or prompt_id not in raw:
            logger.debug(
                "poller: prompt_id=%s not in history yet (%.0fs elapsed)",
                prompt_id, waited,
            )
            continue

        entry = raw[prompt_id]
        status_info = entry.get("status", {})
        status_str = status_info.get("status_str", "")
        completed: bool = status_info.get("completed", False)

        if completed:
            logger.info("poller: prompt_id=%s completed after %.0fs", prompt_id, waited)
            return _strip_history(raw)

        if status_str == "error":
            logger.error(
                "poller: prompt_id=%s reported error after %.0fs", prompt_id, waited
            )
            return {"error": "ComfyUI job failed", "details": _strip_history(raw)}

        logger.debug(
            "poller: prompt_id=%s status=%s (%.0fs elapsed)",
            prompt_id, status_str, waited,
        )

    logger.error("poller: prompt_id=%s timed out after %.0fs", prompt_id, timeout)
    return {"error": f"ComfyUI job {prompt_id} timed out after {timeout:.0f}s"}
