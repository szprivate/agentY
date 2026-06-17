"""Compatibility shim — implementation moved to ``agenty_core.utils.model_node_mapping``.

The real code now lives in the shared **agenty_core** package; this module
remains so existing ``from src.utils.model_node_mapping import ...``
imports keep working.
"""
import sys as _sys
from agenty_core.utils import model_node_mapping as _mod
_sys.modules[__name__] = _mod
