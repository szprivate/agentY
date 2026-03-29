"""System & server info tools for ComfyUI."""

from strands import tool

from src.comfyui_client import get_client


@tool
def get_system_stats() -> dict:
    """Retrieve ComfyUI system statistics including Python version, GPU devices, VRAM usage, and other hardware information.

    Returns:
        A dictionary with system statistics (python version, devices, vram, etc.).
    """
    try:
        return get_client().get("/system_stats")
    except Exception as e:
        return {"error": str(e)}


@tool
def get_features() -> dict:
    """Retrieve the list of server features and capabilities supported by this ComfyUI instance.

    Returns:
        A dictionary describing available server features.
    """
    try:
        return get_client().get("/features")
    except Exception as e:
        return {"error": str(e)}


@tool
def get_embeddings() -> dict:
    """Retrieve the list of available embedding model names installed in ComfyUI.

    Returns:
        A list of embedding names.
    """
    try:
        return get_client().get("/embeddings")
    except Exception as e:
        return {"error": str(e)}


@tool
def get_extensions() -> dict:
    """Retrieve the list of ComfyUI extensions that register a WEB_DIRECTORY.

    Returns:
        A list of extension paths/names.
    """
    try:
        return get_client().get("/extensions")
    except Exception as e:
        return {"error": str(e)}
