"""Queue management tools for ComfyUI."""

from strands import tool

from src.comfyui_client import get_client


@tool
def get_queue() -> dict:
    """Retrieve the current state of the ComfyUI execution queue.

    Shows both currently running and pending items.

    Returns:
        A dictionary with 'queue_running' and 'queue_pending' lists.
    """
    try:
        return get_client().get("/queue")
    except Exception as e:
        return {"error": str(e)}


@tool
def manage_queue(action: str) -> dict:
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
            return {"error": f"Unknown action '{action}'. Use 'clear' or 'clear_running'."}
        return get_client().post("/queue", json_data=payload)
    except Exception as e:
        return {"error": str(e)}
