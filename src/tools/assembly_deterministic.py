"""Compatibility shim — implementation lives in
``agenty_core.tools.assembly_deterministic``.

Re-exports the deterministic (no-LLM) workflow-assembly tool and helpers so
``from src.tools.assembly_deterministic import ...`` keeps working.
"""
import sys as _sys
from agenty_core.tools import assembly_deterministic as _mod
_sys.modules[__name__] = _mod
