"""Execution history tools for ComfyUI."""

import json

from strands import tool

from src.comfyui_client import get_client


@tool
def get_history(max_items: int = 5) -> str:
    """Get recent ComfyUI execution history.

    Args:
        max_items: Max entries to return (default 5; 0 = all).
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
    """Get execution history for a specific prompt ID.

    Args:
        prompt_id: Prompt ID returned by submit_prompt.
    """
    try:
        return json.dumps(get_client().get(f"/history/{prompt_id}"))
    except Exception as e:
        return json.dumps({"error": str(e)})


@tool
def manage_history(action: str, prompt_id: str = "") -> str:
    """Clear or delete ComfyUI execution history entries.

    Args:
        action: 'clear' (all) or 'delete' (one entry).
        prompt_id: Required when action is 'delete'.
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
