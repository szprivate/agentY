"""Iterate tool – executes a Python call n times and collects results."""

from __future__ import annotations

import io
import json
import traceback
from contextlib import redirect_stdout, redirect_stderr

from strands import tool


@tool
def iterate(python_call: str, iter: int) -> str:
    """Execute a Python expression or statement repeatedly and return the results.

    Runs *python_call* exactly *iter* times in a shared execution context so
    that variables defined in earlier iterations are available in later ones.
    Each iteration's stdout, stderr, and (for expressions) return value are
    captured and returned as a JSON array.

    Args:
        python_call: A valid Python expression or statement to execute.
                     If it is an expression its value is recorded under
                     ``"result"``; statements record ``null``.
        iter: Number of times to execute *python_call* (must be >= 1).

    Returns:
        JSON string – an object with keys:
          - ``"iterations"``: list of per-iteration records, each containing
            ``"index"`` (1-based), ``"result"``, ``"stdout"``, ``"stderr"``,
            and ``"error"`` (null on success).
          - ``"total"``: total number of iterations requested.
          - ``"succeeded"``: number that completed without exception.
    """
    if iter < 1:
        return json.dumps({"error": "iter must be >= 1", "total": iter, "succeeded": 0, "iterations": []})

    shared_globals: dict = {}
    results = []
    succeeded = 0

    for i in range(1, iter + 1):
        stdout_buf = io.StringIO()
        stderr_buf = io.StringIO()
        result_value = None
        error = None

        try:
            with redirect_stdout(stdout_buf), redirect_stderr(stderr_buf):
                # Try expression first so we can capture its value
                try:
                    result_value = eval(python_call, shared_globals)  # noqa: S307
                except SyntaxError:
                    # Not an expression – execute as a statement
                    exec(python_call, shared_globals)  # noqa: S102
                    result_value = None
            succeeded += 1
        except Exception:  # noqa: BLE001
            error = traceback.format_exc()

        results.append({
            "index": i,
            "result": result_value if not isinstance(result_value, type(None)) else None,
            "stdout": stdout_buf.getvalue(),
            "stderr": stderr_buf.getvalue(),
            "error": error,
        })

    return json.dumps(
        {
            "total": iter,
            "succeeded": succeeded,
            "iterations": results,
        },
        default=str,
    )
