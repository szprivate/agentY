"""
Lightweight file-read tool that always returns plain text.

`strands_tools.file_read` can return Bedrock-style ``document`` content
blocks, which the Ollama model handler does not support.  This module
provides a simple alternative that reads a file and returns its contents
as a plain string so it works with any Strands-compatible model.
"""

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
