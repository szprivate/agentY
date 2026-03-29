"""Queue management tools for ComfyUI."""

import json

from strands import tool

from src.comfyui_client import get_client


@tool
def get_queue() -> str:
    """Get the current ComfyUI queue (running and pending items)."""
    try:
        return json.dumps(get_client().get("/queue"))
    except Exception as e:
        return json.dumps({"error": str(e)})


@tool
def manage_queue(action: str) -> str:
    """Clear the ComfyUI execution queue.

    Args:
        action: 'clear' (pending) or 'clear_running' (running items).
    """
    try:
        if action == "clear":
            payload = {"clear": True}
        elif action == "clear_running":
            payload = {"clear_running": True}
        else:
            return json.dumps({"error": f"Unknown action '{action}'. Use 'clear' or 'clear_running'"})
        return json.dumps(get_client().post("/queue", json_data=payload))
    except Exception as e:
        return json.dumps({"error": str(e)})
