"""Thin wrapper that delegates all work to ``src.sub.producer.iteration``.

Maintains backwards compatibility with the original entrypoint.
"""
import sys
import os

# ensure the parent directory (workspace root) is on sys.path so that
# the "src" package can be imported even when the script is executed
# directly (e.g. "python src/main.py").
root = os.path.dirname(os.path.dirname(__file__))
if root not in sys.path:
    sys.path.insert(0, root)

from src.sub.producer import iteration


if __name__ == "__main__":
    iteration()
