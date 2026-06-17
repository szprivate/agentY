"""Compatibility shim — implementation moved to ``agenty_core.utils.progress_signal``.

The real code now lives in the shared **agenty_core** package; this module
remains so existing ``from src.utils.progress_signal import ...``
imports keep working.
"""
import sys as _sys
from agenty_core.utils import progress_signal as _mod
_sys.modules[__name__] = _mod
