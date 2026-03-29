"""Model and node-type tools for ComfyUI."""

import json

from strands import tool

from src.comfyui_client import get_client


@tool
def get_model_types() -> str:
    """Retrieve the list of available model type categories (folders) in ComfyUI, such as checkpoints, loras, vae, etc.

    Returns:
        A list of model folder names.
    """
    try:
        return json.dumps(get_client().get("/models"))
    except Exception as e:
        return json.dumps({"error": str(e)})


@tool
def get_models_in_folder(folder: str) -> str:
    """Retrieve the list of model files inside a specific model folder.

    Args:
        folder: The model folder name (e.g. 'checkpoints', 'loras', 'vae', 'clip', 'unet').

    Returns:
        A list of model filenames in that folder.
    """
    try:
        return json.dumps(get_client().get(f"/models/{folder}"))
    except Exception as e:
        return json.dumps({"error": str(e)})


@tool
def get_object_info() -> str:
    """Retrieve detailed information about ALL node types available in this ComfyUI instance.

    This returns a large dictionary with every node class and its inputs, outputs, and configuration.

    Returns:
        A dictionary keyed by node class name with full node definitions.
    """
    try:
        return json.dumps(get_client().get("/object_info"))
    except Exception as e:
        return json.dumps({"error": str(e)})


@tool
def get_node_info(node_class: str) -> str:
    """Retrieve detailed information about a single node type by its class name.

    Args:
        node_class: The node class name (e.g. 'KSampler', 'CLIPTextEncode', 'CheckpointLoaderSimple').

    Returns:
        A dictionary with the node's inputs, outputs, display name, and description.
    """
    try:
        return json.dumps(get_client().get(f"/object_info/{node_class}"))
    except Exception as e:
        return json.dumps({"error": str(e)})


@tool
def get_view_metadata(folder: str) -> str:
    """Retrieve metadata for models in a specific folder.

    Args:
        folder: The model folder to read metadata from (e.g. 'checkpoints').

    Returns:
        A dictionary with model metadata.
    """
    try:
        return json.dumps(get_client().get(f"/view_metadata/{folder}"))
    except Exception as e:
        return json.dumps({"error": str(e)})
