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
    """Download an image from ComfyUI. Returns base64 data or saves to a local path.

    Args:
        filename: Image filename on the server e.g. 'ComfyUI_00001_.png'.
        subfolder: Optional subfolder where the image is located.
        image_type: Directory type: 'output', 'input', or 'temp'.
        save_to: Local file path to save the image; returns base64 if omitted.
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
