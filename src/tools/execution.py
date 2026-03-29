"""Execution control tools – interrupt and free memory."""

import json

from strands import tool

from src.comfyui_client import get_client


@tool
def interrupt_execution() -> str:
    """Immediately stop the currently running ComfyUI workflow execution."""
    try:
        return json.dumps(get_client().post("/interrupt", json_data={}))
    except Exception as e:
        return json.dumps({"error": str(e)})


@tool
def free_memory(unload_models: bool = True, free_memory_flag: bool = True) -> str:
    """Free GPU/system memory in ComfyUI by unloading models and clearing caches.

    Args:
        unload_models: Unload all loaded models from VRAM (default True).
        free_memory_flag: Free cached memory (default True).
    """
    try:
        payload = {
            "unload_models": unload_models,
            "free_memory": free_memory_flag,
        }
        return json.dumps(get_client().post("/free", json_data=payload))
    except Exception as e:
        return json.dumps({"error": str(e)})
