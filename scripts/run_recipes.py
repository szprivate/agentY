#!/usr/bin/env python
"""Launcher for the workflow_recipes CLI.

Convenience wrapper for `python -m agenty_core.workflow_recipes.cli ...`.
`agenty_core` is installed as a package, so this runs from any working
directory:

    python scripts/run_recipes.py --no-fetch
"""

from agenty_core.workflow_recipes.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
