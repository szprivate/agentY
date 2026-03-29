"""Queue management tools for ComfyUI."""

import json

from strands import tool

from src.comfyui_client import get_client


@tool
def get_queue() -> str:
    """Retrieve the current state of the ComfyUI execution queue.

    Shows both currently running and pending items.

    Returns:
        A dictionary with 'queue_running' and 'queue_pending' lists.
    """
    try:
        return json.dumps(get_client().get("/queue"))
    except Exception as e:
        return json.dumps({"error": str(e)})


@tool
def manage_queue(action: str) -> str:
    """Manage the ComfyUI execution queue by clearing pending or running items.

    Args:
        action: The queue action to perform. Must be one of:
                - 'clear': Clear all pending items from the queue.
                - 'clear_running': Clear all currently running items.

    Returns:
        A confirmation dictionary or error details.
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
