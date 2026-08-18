"""Compatibility shim — implementation moved to ``agenty_core.utils.project_memory``.

Project memory is a fact about the CURRENT project, stored in ComfyUI's own user
directory. Nothing about that is specific to this app, and a Claude Desktop
session reaching the same ComfyUI should read and write the same memory the panel
does — which is why it moved to the shared layer. This module remains so existing
``from src.utils.project_memory import ...`` imports keep working.
"""
import sys as _sys
from agenty_core.utils import project_memory as _mod
_sys.modules[__name__] = _mod
