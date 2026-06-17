"""Compatibility shim — implementation moved to ``agenty_core.utils.comfyui_client``.

The real code now lives in the shared **agenty_core** package; this module
remains so existing ``from src.utils.comfyui_client import ...``
imports keep working.
"""
import sys as _sys
from agenty_core.utils import comfyui_client as _mod
_sys.modules[__name__] = _mod
