"""Execution history tools for ComfyUI."""

import json

from strands import tool

from src.comfyui_client import get_client


def _strip_history(data: dict | list) -> dict | list:
    """Strip embedded workflow/prompt JSON from history entries to save tokens.

    Keeps only: status, timestamps, output filenames, and node errors.
    """
    if isinstance(data, list):
        return [_strip_history(item) for item in data]
    if not isinstance(data, dict):
        return data

    stripped: dict = {}
    for prompt_id, entry in data.items():
        if not isinstance(entry, dict):
            stripped[prompt_id] = entry
            continue
        slim: dict = {}
        # Keep status info
        if "status" in entry:
            slim["status"] = entry["status"]
        # Keep outputs (filenames only)
        if "outputs" in entry:
            outputs: dict = {}
            for node_id, node_out in entry.get("outputs", {}).items():
                if isinstance(node_out, dict):
                    # Keep image/video file references but strip large data
                    slim_out: dict = {}
                    for key, val in node_out.items():
                        if isinstance(val, list):
                            slim_out[key] = [
                                {k: v for k, v in item.items() if k != "abs_path"}
                                if isinstance(item, dict) else item
                                for item in val
                            ]
                        else:
                            slim_out[key] = val
                    outputs[node_id] = slim_out
            slim["outputs"] = outputs
        stripped[prompt_id] = slim
    return stripped


@tool
def get_history(max_items: int = 3) -> str:
    """Get recent ComfyUI execution history (status and output filenames only).

    Args:
        max_items: Max entries to return (default 3; 0 = all).
    """
    try:
        params = {}
        if max_items > 0:
            params["max_items"] = max_items
        raw = get_client().get("/history", params=params or None)
        return json.dumps(_strip_history(raw))
    except Exception as e:
        return json.dumps({"error": str(e)})


@tool
def get_prompt_status_by_id(prompt_id: str) -> str:
    """Check execution status for a specific prompt ID. Returns status and output filenames only.

    Args:
        prompt_id: Prompt ID returned by submit_prompt.
    """
    try:
        raw = get_client().get(f"/history/{prompt_id}")
        return json.dumps(_strip_history(raw))
    except Exception as e:
        return json.dumps({"error": str(e)})


@tool
def clear_history(prompt_id: str = "") -> str:
    """Clear ComfyUI execution history. If prompt_id given, deletes that entry only.

    Args:
        prompt_id: Optional specific prompt ID to delete. If empty, clears all history.
    """
    try:
        if prompt_id:
            payload = {"delete": [prompt_id]}
        else:
            payload = {"clear": True}
        return json.dumps(get_client().post("/history", json_data=payload))
    except Exception as e:
        return json.dumps({"error": str(e)})
