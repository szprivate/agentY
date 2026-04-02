"""Queue management tools for ComfyUI."""

import json

from strands import tool

from src.comfyui_client import get_client


@tool
def queue(action: str = "status") -> str:
    """Get or manage the ComfyUI execution queue.

    Args:
        action: 'status' (view queue), 'clear' (clear pending), or 'clear_running' (stop running items).
    """
    try:
        if action == "status":
            return json.dumps(get_client().get("/queue"))
        elif action in ("clear", "clear_running"):
            payload = {"clear": True} if action == "clear" else {"clear_running": True}
            return json.dumps(get_client().post("/queue", json_data=payload))
        else:
            return json.dumps({"error": f"Unknown action '{action}'. Use 'status', 'clear', or 'clear_running'"})
    except Exception as e:
        return json.dumps({"error": str(e)})
