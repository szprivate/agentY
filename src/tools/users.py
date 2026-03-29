"""User account tools for ComfyUI (multi-user mode)."""

import json

from strands import tool

from src.comfyui_client import get_client


@tool
def get_users() -> str:
    """Retrieve user information from the ComfyUI server.

    Returns:
        A dictionary with user details. In single-user mode this returns default user info.
    """
    try:
        return json.dumps(get_client().get("/users"))
    except Exception as e:
        return json.dumps({"error": str(e)})


@tool
def create_user(username: str) -> str:
    """Create a new user on the ComfyUI server (multi-user mode only).

    Args:
        username: The username for the new user.

    Returns:
        A confirmation or error dictionary.
    """
    try:
        return json.dumps(get_client().post("/users", json_data={"username": username}))
    except Exception as e:
        return json.dumps({"error": str(e)})
