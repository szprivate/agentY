"""Execution history tools for ComfyUI."""

from strands import tool

from src.comfyui_client import get_client


@tool
def get_history(max_items: int = 0) -> dict:
    """Retrieve the execution history from ComfyUI.

    Args:
        max_items: Maximum number of history entries to return. 0 means return all.

    Returns:
        A dictionary of prompt_id -> execution details.
    """
    try:
        params = {}
        if max_items > 0:
            params["max_items"] = max_items
        return get_client().get("/history", params=params or None)
    except Exception as e:
        return {"error": str(e)}


@tool
def get_prompt_history(prompt_id: str) -> dict:
    """Retrieve the execution history for a specific prompt by its ID.

    Args:
        prompt_id: The unique prompt identifier returned when a prompt was submitted.

    Returns:
        A dictionary with the execution details for that prompt.
    """
    try:
        return get_client().get(f"/history/{prompt_id}")
    except Exception as e:
        return {"error": str(e)}


@tool
def manage_history(action: str, prompt_id: str = "") -> dict:
    """Manage execution history by clearing all entries or deleting a specific item.

    Args:
        action: The history action. Must be one of:
                - 'clear': Delete all history entries.
                - 'delete': Delete a specific history entry (requires prompt_id).
        prompt_id: The prompt ID to delete. Required when action is 'delete'.

    Returns:
        A confirmation dictionary or error details.
    """
    try:
        if action == "clear":
            payload = {"clear": True}
        elif action == "delete":
            if not prompt_id:
                return {"error": "prompt_id is required when action is 'delete'."}
            payload = {"delete": [prompt_id]}
        else:
            return {"error": f"Unknown action '{action}'. Use 'clear' or 'delete'."}
        return get_client().post("/history", json_data=payload)
    except Exception as e:
        return {"error": str(e)}
