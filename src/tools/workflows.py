"""Workflow template tools for ComfyUI."""

import json

from strands import tool

from src.comfyui_client import get_client


@tool
def get_workflow_templates() -> str:
    """Retrieve a map of custom node modules and their associated template workflows.

    Returns:
        A dictionary mapping module names to lists of workflow templates.
    """
    try:
        return json.dumps(get_client().get("/workflow_templates"))
    except Exception as e:
        return json.dumps({"error": str(e)})
