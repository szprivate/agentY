"""
Lightweight file I/O tools that always work with plain text.

`strands_tools.file_read` can return Bedrock-style ``document`` content
blocks, which the Ollama model handler does not support.  This module
provides simple alternatives that work with any Strands-compatible model.
"""

import json
from pathlib import Path

from strands import tool


@tool
def read_text_file(path: str) -> str:
    """Read a text file from disk and return its contents as a plain string.

    Use this tool to inspect configuration files, JSON templates, markdown
    documents, or any other UTF-8 text file.  Binary files are not supported.

    Args:
        path: Absolute or relative path to the file to read.

    Returns:
        The full text contents of the file, or an error message if the file
        cannot be opened.
    """
    try:
        return Path(path).read_text(encoding="utf-8", errors="replace")
    except FileNotFoundError:
        return f"[read_text_file] File not found: {path}"
    except PermissionError:
        return f"[read_text_file] Permission denied: {path}"
    except Exception as exc:  # noqa: BLE001
        return f"[read_text_file] Error reading {path}: {exc}"


@tool
def write_text_file(path: str, content: str) -> str:
    """Write *content* to a file on disk (UTF-8).  Parent directories are
    created automatically.  Any existing file at *path* is overwritten.

    Use this tool to persist JSON, markdown, or any other plain-text data.
    Prefer this over ``run_script`` for simple file-write operations — it
    always uses the correct workspace root regardless of process CWD.

    Args:
        path: Absolute path, OR a path relative to the agentY workspace root
              (e.g. ``output/_workflows/multiprompt.json``).
        content: The text to write.  Must already be a string; pass
                 ``json.dumps(data, indent=2)`` for JSON payloads.

    Returns:
        A JSON string ``{"ok": true, "path": "<absolute_path>", "bytes": N}``
        on success, or ``{"ok": false, "error": "<message>"}`` on failure.
    """
    try:
        # Resolve relative paths against the workspace root (two levels up from
        # this file: src/tools/file_tools.py → src/ → project root).
        _WORKSPACE_ROOT = Path(__file__).parent.parent.parent

        p = Path(path)
        if not p.is_absolute():
            p = _WORKSPACE_ROOT / p

        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
        return json.dumps({"ok": True, "path": str(p), "bytes": len(content.encode("utf-8"))})
    except PermissionError as exc:
        return json.dumps({"ok": False, "error": f"Permission denied: {exc}"})
    except Exception as exc:  # noqa: BLE001
        return json.dumps({"ok": False, "error": str(exc)})
