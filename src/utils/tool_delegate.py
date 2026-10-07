"""Which agent a tool call is really answered by.

``analyze_image`` is called by the orchestrator and answered by the vision agent,
on the vision tier's model. The panel showed only the first half — "[orchestrator]
analyze_image" — so it looked as if the orchestrator read every image itself and
the vision model in Settings did nothing.

:func:`delegate_for` names the agent and model behind such a call, for the tool
card to show. It answers None whenever the call is NOT handed on, so the card
never credits an agent that did not look.
"""
from __future__ import annotations


def _vision():
    from src.tools import image_handling
    return image_handling._vision_agent


def _video():
    from src.tools import video_handling
    return video_handling._video_agent


def delegate_for(name: str, tool_input) -> dict | None:
    """``{"agent": "vision", "model": "qwen3-vl-flash"}`` for a handed-on call."""
    try:
        args = tool_input if isinstance(tool_input, dict) else {}
        if name == "analyze_image":
            # mode="full" hands the pixels to the caller; nobody else looks.
            if str(args.get("mode") or "describe").lower() == "full":
                return None
            role, agent = "vision", _vision()
        elif name == "analyze_video":
            role, agent = "video", _video()
        else:
            return None
        if agent is None:
            return None
        from src.utils.vision_capability import model_name
        return {"agent": role, "model": model_name(agent)}
    except Exception:  # noqa: BLE001 — a label must never cost a tool call
        return None
