"""Prompt submission and status tools for ComfyUI."""

import json
from pathlib import Path

from strands import tool

from src.comfyui_client import get_client


@tool
def get_prompt_status() -> str:
    """Get ComfyUI queue status and exec_info (items pending)."""
    try:
        return json.dumps(get_client().get("/prompt"))
    except Exception as e:
        return json.dumps({"error": str(e)})


@tool
def submit_prompt(workflow_path: str, client_id: str = "") -> str:
    """Submit a workflow to the ComfyUI execution queue. Returns prompt_id on success.

    Args:
        workflow_path: File path to the workflow JSON (from get_workflow_template or save_workflow).
        client_id: Optional client identifier for tracking.
    """
    try:
        p = Path(workflow_path)
        if p.exists() and p.suffix == ".json":
            workflow = json.loads(p.read_text(encoding="utf-8"))
        else:
            # Legacy fallback: accept inline JSON string
            workflow = json.loads(workflow_path)

        client = get_client()
        payload: dict = {"prompt": workflow}
        if client_id:
            payload["client_id"] = client_id
        # Forward the ComfyUI API key in extra_data so API/partner nodes receive
        # it via the AUTH_TOKEN_COMFY_ORG / API_KEY_COMFY_ORG hidden input.
        if client.api_key:
            payload["extra_data"] = {"api_key_comfy_org": client.api_key}
        return json.dumps(client.post("/prompt", json_data=payload))
    except json.JSONDecodeError as e:
        return json.dumps({"error": f"Invalid JSON in workflow: {e}"})
    except Exception as e:
        return json.dumps({"error": str(e)})
