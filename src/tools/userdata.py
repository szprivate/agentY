"""User-data file management tools for ComfyUI."""

import json

from strands import tool

from src.comfyui_client import get_client


@tool
def list_userdata(directory: str) -> str:
    """List user data files in a directory on the ComfyUI server.

    Args:
        directory: Directory path relative to the userdata root.
    """
    try:
        return json.dumps(get_client().get("/userdata", params={"dir": directory}))
    except Exception as e:
        return json.dumps({"error": str(e)})


@tool
def list_userdata_v2(directory: str) -> str:
    """List user data files and directories with metadata (v2 endpoint).

    Args:
        directory: Directory path relative to the userdata root.
    """
    try:
        return json.dumps(get_client().get("/v2/userdata", params={"dir": directory}))
    except Exception as e:
        return json.dumps({"error": str(e)})


@tool
def get_userdata_file(file_path: str) -> str:
    """Get the content of a user data file from ComfyUI.

    Args:
        file_path: Path relative to userdata root e.g. 'workflows/my_flow.json'.
    """
    try:
        return json.dumps(get_client().get(f"/userdata/{file_path}"))
    except Exception as e:
        return json.dumps({"error": str(e)})


@tool
def save_userdata_file(file_path: str, content: str, overwrite: bool = True) -> str:
    """Upload or update a user data file on the ComfyUI server.

    Args:
        file_path: Destination path relative to userdata root.
        content: File content (text or JSON string).
        overwrite: Overwrite existing file (default True).
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
    """Delete a user data file from the ComfyUI server.

    Args:
        file_path: Path of the file to delete, relative to userdata root.
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
        file_path: Current path relative to userdata root.
        destination: New path relative to userdata root.
    """
    try:
        resp = get_client().post(f"/userdata/{file_path}/move/{destination}", json_data={})
        return json.dumps(resp if isinstance(resp, dict) else {"status": "ok", "response": resp})
    except Exception as e:
        return json.dumps({"error": str(e)})
