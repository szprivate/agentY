"""Model and node-type tools for ComfyUI."""

import json

from strands import tool

from src.comfyui_client import get_client


@tool
def get_model_types() -> str:
    """List available model folder types in ComfyUI (checkpoints, loras, unet, vae, clip, etc.)."""
    try:
        return json.dumps(get_client().get("/models"))
    except Exception as e:
        return json.dumps({"error": str(e)})


@tool
def get_models_in_folder(folder: str) -> str:
    """List model files in a ComfyUI model folder.

    Args:
        folder: Folder name e.g. 'checkpoints', 'loras', 'vae', 'clip', 'unet'.
    """
    try:
        return json.dumps(get_client().get(f"/models/{folder}"))
    except Exception as e:
        return json.dumps({"error": str(e)})


@tool
def get_object_info() -> str:
    """Return a compact list of all ComfyUI node class names and categories. Use get_node_schema(node_class) for full details on any specific node."""
    try:
        return json.dumps(get_client().get("/object_info"))
    except Exception as e:
        return json.dumps({"error": str(e)})


@tool
def get_node_info(node_class: str) -> str:
    """Get raw ComfyUI object_info for a single node class. Prefer get_node_schema() for a friendlier format.

    Args:
        node_class: Node class name e.g. 'KSampler', 'CLIPTextEncode'.
    """
    try:
        return json.dumps(get_client().get(f"/object_info/{node_class}"))
    except Exception as e:
        return json.dumps({"error": str(e)})


@tool
def get_view_metadata(folder: str) -> str:
    """Get metadata for models in a ComfyUI folder.

    Args:
        folder: Model folder name e.g. 'checkpoints'.
    """
    try:
        return json.dumps(get_client().get(f"/view_metadata/{folder}"))
    except Exception as e:
        return json.dumps({"error": str(e)})
