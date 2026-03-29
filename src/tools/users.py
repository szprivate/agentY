"""User account tools for ComfyUI (multi-user mode)."""

from strands import tool

from src.comfyui_client import get_client


@tool
def get_users() -> dict:
    """Retrieve user information from the ComfyUI server.

    Returns:
        A dictionary with user details. In single-user mode this returns default user info.
    """
    try:
        return get_client().get("/users")
    except Exception as e:
        return {"error": str(e)}


@tool
def create_user(username: str) -> dict:
    """Create a new user on the ComfyUI server (multi-user mode only).

    Args:
        username: The username for the new user.

    Returns:
        A confirmation or error dictionary.
    """
    try:
        return get_client().post("/users", json_data={"username": username})
    except Exception as e:
        return {"error": str(e)}
