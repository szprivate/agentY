"""Compatibility shim — implementation moved to ``agenty_core.tools.project_memory``.

Project memory is a fact about the CURRENT project, stored in ComfyUI's own user
directory. Nothing about that is specific to this app, and a Claude Desktop
session reaching the same ComfyUI should read and write the same memory the panel
does — which is why it moved to the shared layer. This module remains so existing
``from src.tools.project_memory import ...`` imports keep working.

It also hands the shared tools this app's write pause (``/agentY/memory/writes``),
so a benchmark or test run that pauses long-term memory leaves project memory
alone too — a test once stored a wrong "Wan 3.0 has no resolution control" there.
"""
import sys as _sys
from agenty_core.tools import project_memory as _mod
from src.utils.memory import writes_paused as _writes_paused

if hasattr(_mod, "set_write_pause"):
    _mod.set_write_pause(_writes_paused)
_sys.modules[__name__] = _mod
