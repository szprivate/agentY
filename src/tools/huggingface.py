"""Compatibility shim — implementation moved to ``agenty_core.tools.huggingface``.

The real code now lives in the shared **agenty_core** package; this module
remains so existing ``from src.tools.huggingface import ...``
imports keep working.
"""
import sys as _sys
from agenty_core.tools import huggingface as _mod
_sys.modules[__name__] = _mod
