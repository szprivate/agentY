"""Makes this directory a package, and keeps it importable from inside itself.

Two things are needed for `python -m unittest discover -s tests` to work here, and
both are accidents of history rather than design:

* This file has to exist at all. A dependency (literalai) ships a top-level
  `tests` package into site-packages, and a REGULAR package there beats a
  namespace one wherever it sits on sys.path — so without this, discovery finds
  literalai's tests and reports this directory as "not importable".
* Sixteen test modules import their shared harness as `from pipeline_stub import
  ...`, which only resolves when this directory is itself on sys.path. That is
  true when a file is run directly and false under discovery, so the line below
  makes it true either way rather than rewriting sixteen imports to say
  `tests.pipeline_stub`.
"""
import os as _os
import sys as _sys

_here = _os.path.dirname(_os.path.abspath(__file__))
if _here not in _sys.path:
    _sys.path.insert(0, _here)
