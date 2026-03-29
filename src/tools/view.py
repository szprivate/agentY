"""Image viewing tool for ComfyUI."""

import base64
import json
import os

from strands import tool

from src.comfyui_client import get_client


@tool
def view_image(
    filename: str,
    subfolder: str = "",
    image_type: str = "output",
    save_to: str = "",
) -> str:
    """View or download an image from the ComfyUI server.

    Retrieves the image from the server. If save_to is provided the image is
    written to that local path; otherwise a base64-encoded representation is
    returned (useful for passing to other tools or models).

    Args:
        filename: The image filename on the server (e.g. 'ComfyUI_00001_.png').
        subfolder: Optional subfolder where the image is located.
        image_type: The directory type: 'output', 'input', or 'temp'.
        save_to: Optional local file path to save the downloaded image to.

    Returns:
        A dictionary with 'saved_to' path if saved, or 'base64' encoded image data,
        along with 'content_type' and 'size_bytes'.
    """
    try:
        params: dict = {"filename": filename, "type": image_type}
        if subfolder:
            params["subfolder"] = subfolder

        resp = get_client().get("/view", params=params, raw=True)
        content_type = resp.headers.get("content-type", "image/png")
        image_bytes = resp.content

        if save_to:
            os.makedirs(os.path.dirname(save_to) or ".", exist_ok=True)
            with open(save_to, "wb") as f:
                f.write(image_bytes)
            return json.dumps({
                "saved_to": save_to,
                "content_type": content_type,
                "size_bytes": len(image_bytes),
            })

        return json.dumps({
            "base64": base64.b64encode(image_bytes).decode("utf-8"),
            "content_type": content_type,
            "size_bytes": len(image_bytes),
        })
    except Exception as e:
        return json.dumps({"error": str(e)})
