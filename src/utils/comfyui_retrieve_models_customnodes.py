"""Compatibility shim — implementation moved to ``agenty_core.utils.comfyui_retrieve_models_customnodes``.

The real code now lives in the shared **agenty_core** package; this module
remains so existing ``from src.utils.comfyui_retrieve_models_customnodes import ...``
imports keep working.
"""
import sys as _sys
from agenty_core.utils import comfyui_retrieve_models_customnodes as _mod
_sys.modules[__name__] = _mod
