"""Keep the graph that made each output beside it, so a shot can be recreated later.

When a run finishes (``executor._process_completed_job``) every output gets its
provenance (``agenty_core.utils.provenance``, format v1): the API graph ComfyUI ran,
taken from the run's own ``/history`` entry - the exact prompt, never its
``extra_data`` - plus its prompt id and the output node / index the file came from.

The output folder is often a network share (W:), where writing two small files per
output can take seconds, so the writes run on a background thread per run
(``provenance.capture``). A caller about to move or read those files (recreating a
shot, staging a copy) calls :func:`wait` first.
"""
from __future__ import annotations

import logging

logger = logging.getLogger("agentY.provenance")


def capture(paths, history: dict, prompt_id: str, workflow_path: str = "", *, background: bool = True):
    """Write the provenance of *paths* (the run's outputs) from its *history* entry.

    Returns the background thread (``background=True``), or how many were written."""
    try:
        from agenty_core.utils import provenance as pv
        return pv.capture(paths, history, prompt_id, workflow_path, source="agentY", background=background)
    except Exception as exc:  # noqa: BLE001 - provenance is never worth failing a run
        logger.debug("provenance not captured: %s", exc)
        return None if background else 0


def wait(timeout: float = 60.0) -> None:
    """Wait for the provenance writes started so far."""
    try:
        from agenty_core.utils import provenance as pv
        pv.wait_pending(timeout)
    except Exception:  # noqa: BLE001
        pass
