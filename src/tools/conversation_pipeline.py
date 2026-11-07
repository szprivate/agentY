"""``conversation_as_pipeline`` - a conversation that worked, as the steps of a hook pipeline.

The record comes from the host (src/utils/conversation_pipeline.py): the
transcript, the outputs and the provenance of each output. The agent plans the
pipeline from it with the hook-pipeline skill and draws it with
``edit_canvas_graph``.
"""
from __future__ import annotations

import json

from strands import tool


@tool
def conversation_as_pipeline(conversation: str = "", max_steps: int = 40) -> str:
    """What a conversation did, step by step, to rebuild it as a hook pipeline.

    Use it when the user asks to turn a conversation that worked into a pipeline
    they can re-run ("make what we just did a pipeline", "rebuild the robot
    conversation as hooks"). Your own memory of a conversation is compacted; this
    is the record: each user message with the outputs made before the next one,
    the workflow file behind each output (``workflow``), its prompt / model /
    seed, the earlier outputs it read (``reads``), whether a later step built on
    it, and the user's next message (``reaction``).

    Args:
        conversation: "" for this conversation; else a conversation's id or words
            from its title.
        max_steps: How many of the newest steps to return (default 40).

    Returns JSON ``{status, conversation, title, steps, workflows, how_to}``.
    """
    from src.utils import conversation_pipeline as cp
    tid, title = cp.resolve(conversation)
    if not tid:
        return json.dumps({"status": "error", "error": title})
    try:
        got = cp.steps(tid, max_steps=max(1, min(int(max_steps or cp.MAX_STEPS), 200)))
    except Exception as exc:  # noqa: BLE001
        return json.dumps({"status": "error", "error": f"could not read the conversation: {exc}"})
    return json.dumps({"status": "ok", "conversation": tid, "title": title, **got, "how_to": cp.HOW_TO},
                      ensure_ascii=False, default=str)
