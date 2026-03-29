"""Execution history tools for ComfyUI."""

import json

from strands import tool

from src.comfyui_client import get_client


@tool
def get_history(max_items: int = 0) -> str:
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
        return json.dumps(get_client().get("/history", params=params or None))
    except Exception as e:
        return json.dumps({"error": str(e)})


@tool
def get_prompt_history(prompt_id: str) -> str:
    """Retrieve the execution history for a specific prompt by its ID.

    Args:
        prompt_id: The unique prompt identifier returned when a prompt was submitted.

    Returns:
        A dictionary with the execution details for that prompt.
    """
    try:
        return json.dumps(get_client().get(f"/history/{prompt_id}"))
    except Exception as e:
        return json.dumps({"error": str(e)})


@tool
def manage_history(action: str, prompt_id: str = "") -> str:
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
                return json.dumps({"error": "prompt_id is required when action is 'delete'"})
            payload = {"delete": [prompt_id]}
        else:
            return json.dumps({"error": f"Unknown action '{action}'. Use 'clear' or 'delete'"})
        return json.dumps(get_client().post("/history", json_data=payload))
    except Exception as e:
        return json.dumps({"error": str(e)})
