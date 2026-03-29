"""User-data file management tools for ComfyUI."""

import json

from strands import tool

from src.comfyui_client import get_client


@tool
def list_userdata(directory: str) -> str:
    """List user data files in a specified directory on the ComfyUI server.

    Args:
        directory: The directory path to list (relative to the userdata root).

    Returns:
        A list of filenames in the directory.
    """
    try:
        return json.dumps(get_client().get("/userdata", params={"dir": directory}))
    except Exception as e:
        return json.dumps({"error": str(e)})


@tool
def list_userdata_v2(directory: str) -> str:
    """List user data files and directories in a structured format (v2 endpoint).

    Returns both files and subdirectories with additional metadata.

    Args:
        directory: The directory path to list (relative to the userdata root).

    Returns:
        A structured dictionary with files and directories.
    """
    try:
        return json.dumps(get_client().get("/v2/userdata", params={"dir": directory}))
    except Exception as e:
        return json.dumps({"error": str(e)})


@tool
def get_userdata_file(file_path: str) -> str:
    """Retrieve the content of a specific user data file from ComfyUI.

    Args:
        file_path: The file path relative to the userdata root (e.g. 'workflows/my_flow.json').

    Returns:
        The file content (parsed as JSON if applicable, otherwise as text).
    """
    try:
        return json.dumps(get_client().get(f"/userdata/{file_path}"))
    except Exception as e:
        return json.dumps({"error": str(e)})


@tool
def save_userdata_file(file_path: str, content: str, overwrite: bool = True) -> str:
    """Upload or update a user data file on the ComfyUI server.

    Args:
        file_path: Destination path relative to the userdata root.
        content: The file content to save (text or JSON string).
        overwrite: If True (default), overwrite existing files.

    Returns:
        A confirmation or error dictionary.
    """
    try:
        params = {}
        if overwrite:
            params["overwrite"] = "true"
        resp = get_client().post(
            f"/userdata/{file_path}",
            data=content.encode("utf-8") if isinstance(content, str) else content,
        )
        return json.dumps(resp if isinstance(resp, dict) else {"status": "ok", "response": resp})
    except Exception as e:
        return json.dumps({"error": str(e)})


@tool
def delete_userdata_file(file_path: str) -> str:
    """Delete a specific user data file from the ComfyUI server.

    Args:
        file_path: The file path relative to the userdata root to delete.

    Returns:
        A confirmation or error dictionary.
    """
    try:
        resp = get_client().delete(f"/userdata/{file_path}")
        return json.dumps(resp if isinstance(resp, dict) else {"status": "ok", "response": resp})
    except Exception as e:
        return json.dumps({"error": str(e)})


@tool
def move_userdata_file(file_path: str, destination: str) -> str:
    """Move or rename a user data file on the ComfyUI server.

    Args:
        file_path: The current file path relative to the userdata root.
        destination: The new file path relative to the userdata root.

    Returns:
        A confirmation or error dictionary.
    """
    try:
        resp = get_client().post(f"/userdata/{file_path}/move/{destination}", json_data={})
        return json.dumps(resp if isinstance(resp, dict) else {"status": "ok", "response": resp})
    except Exception as e:
        return json.dumps({"error": str(e)})
