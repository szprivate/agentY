"""Prompt submission and status tools for ComfyUI."""

import json

from strands import tool

from src.comfyui_client import get_client


@tool
def get_prompt_status() -> str:
    """Retrieve the current queue status and execution information from ComfyUI.

    Returns the exec_info including the number of items in the queue.

    Returns:
        A dictionary with queue status information.
    """
    try:
        return json.dumps(get_client().get("/prompt"))
    except Exception as e:
        return json.dumps({"error": str(e)})


@tool
def submit_prompt(prompt_workflow: str, client_id: str = "") -> str:
    """Submit a prompt workflow to the ComfyUI execution queue.

    The workflow must be a valid ComfyUI API-format JSON object describing
    the node graph to execute.

    Args:
        prompt_workflow: A JSON string of the workflow to execute in ComfyUI API format.
                         This is the node graph with node IDs as keys and node configs as values.
        client_id: Optional client identifier for tracking. Leave empty for auto-generated.

    Returns:
        A dictionary with 'prompt_id' and 'number' (queue position) on success,
        or 'error' and 'node_errors' on validation failure.
    """
    try:
        workflow = json.loads(prompt_workflow) if isinstance(prompt_workflow, str) else prompt_workflow
        payload: dict = {"prompt": workflow}
        if client_id:
            payload["client_id"] = client_id
        return json.dumps(get_client().post("/prompt", json_data=payload))
    except json.JSONDecodeError as e:
        return json.dumps({"error": f"Invalid JSON in prompt_workflow: {e}"})
    except Exception as e:
        return json.dumps({"error": str(e)})
