"""Compatibility shim — implementation moved to ``agenty_core.tools.file_tools``.

The real code now lives in the shared **agenty_core** package; this module
remains so existing ``from src.tools.file_tools import ...``
imports keep working.
"""
import sys as _sys
from agenty_core.tools import file_tools as _mod
_sys.modules[__name__] = _mod
