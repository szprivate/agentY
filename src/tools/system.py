"""System & server info tools for ComfyUI."""

import json

from strands import tool

from src.comfyui_client import get_client


@tool
def get_system_stats() -> str:
    """Get ComfyUI system stats: Python version, GPU devices, VRAM usage."""
    try:
        return json.dumps(get_client().get("/system_stats"))
    except Exception as e:
        return json.dumps({"error": str(e)})


@tool
def get_features() -> str:
    """List server features and capabilities of this ComfyUI instance."""
    try:
        return json.dumps(get_client().get("/features"))
    except Exception as e:
        return json.dumps({"error": str(e)})


@tool
def get_embeddings() -> str:
    """List available embedding model names installed in ComfyUI."""
    try:
        return json.dumps(get_client().get("/embeddings"))
    except Exception as e:
        return json.dumps({"error": str(e)})


@tool
def get_extensions() -> str:
    """List ComfyUI extensions that register a WEB_DIRECTORY."""
    try:
        return json.dumps(get_client().get("/extensions"))
    except Exception as e:
        return json.dumps({"error": str(e)})
