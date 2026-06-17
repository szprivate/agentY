"""Compatibility shim — implementation moved to ``agenty_core.utils.workflow_parser``.

The real code now lives in the shared **agenty_core** package; this module
remains so existing ``from src.utils.workflow_parser import ...``
imports keep working.
"""
import sys as _sys
from agenty_core.utils import workflow_parser as _mod
_sys.modules[__name__] = _mod
