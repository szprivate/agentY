#!/usr/bin/env python
"""Launcher for the workflow_recipes tool.

Runnable from any working directory, because Python puts this script's own
directory (the repo root) on sys.path[0], which makes the workflow_recipes
package importable regardless of where you invoke it from:

    python D:\\AI\\agentY\\run_recipes.py --similarity-threshold 0.2

Equivalent to `python -m workflow_recipes.cli ...` run from the repo root.
"""

from workflow_recipes.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
