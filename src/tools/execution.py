"""Execution control tools – interrupt and free memory."""

from strands import tool

from src.comfyui_client import get_client


@tool
def interrupt_execution() -> dict:
    """Immediately interrupt / stop the currently running workflow execution in ComfyUI.

    Returns:
        A confirmation or error dictionary.
    """
    try:
        return get_client().post("/interrupt", json_data={})
    except Exception as e:
        return {"error": str(e)}


@tool
def free_memory(unload_models: bool = True, free_memory_flag: bool = True) -> dict:
    """Free GPU/system memory by unloading models and/or clearing caches in ComfyUI.

    Args:
        unload_models: If True, unload all loaded models from memory.
        free_memory_flag: If True, free cached memory.

    Returns:
        A confirmation or error dictionary.
    """
    try:
        payload = {
            "unload_models": unload_models,
            "free_memory": free_memory_flag,
        }
        return get_client().post("/free", json_data=payload)
    except Exception as e:
        return {"error": str(e)}
