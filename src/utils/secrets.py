"""Compatibility shim — implementation moved to ``agenty_core.utils.secrets``.

The real code now lives in the shared **agenty_core** package; this module
remains so existing ``from src.utils.secrets import ...``
imports keep working.
"""
import sys as _sys
from agenty_core.utils import secrets as _mod
_sys.modules[__name__] = _mod
