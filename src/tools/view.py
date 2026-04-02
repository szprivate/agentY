"""Image viewing tool for ComfyUI."""

import json
import os

from PIL import Image
from strands import tool

from src.comfyui_client import get_client


@tool
def view_image(
    filename: str,
    save_to: str,
    subfolder: str = "",
    image_type: str = "output",
) -> str:
    """Download an image from ComfyUI and save it to a local path.

    After saving, use analyze_image(file_path=save_to) if you need to
    inspect the image contents. Use slack_send_image(file_path=save_to) to
    post it to Slack.

    Args:
        filename: Image filename on the server e.g. 'ComfyUI_00001_.png'.
        save_to: Local file path to save the image (e.g. './output/image.png'). Required.
        subfolder: Optional subfolder where the image is located.
        image_type: Directory type: 'output', 'input', or 'temp'.
    """
    try:
        params: dict = {"filename": filename, "type": image_type}
        if subfolder:
            params["subfolder"] = subfolder

        resp = get_client().get("/view", params=params, raw=True)
        content_type = resp.headers.get("content-type", "image/png")
        image_bytes = resp.content

        os.makedirs(os.path.dirname(save_to) or ".", exist_ok=True)
        with open(save_to, "wb") as f:
            f.write(image_bytes)
        result = {
            "saved_to": save_to,
            "content_type": content_type,
            "size_bytes": len(image_bytes),
        }
        if len(image_bytes) > 5 * 1024 * 1024:
            result["warning"] = (
                f"Image is {len(image_bytes) / 1024 / 1024:.1f} MB — exceeds 5 MB limit. "
                "Activate the 'image-downsize' skill to produce a smaller copy."
            )
        return json.dumps(result)
    except Exception as e:
        return json.dumps({"error": str(e)})


@tool
def get_image_resolution(image_path: str) -> str:
    """Return the resolution (width and height in pixels) of a local image file.

    Args:
        image_path: Absolute or relative path to the image file on disk.
    """
    try:
        with Image.open(image_path) as img:
            width, height = img.size
        return json.dumps({"width": width, "height": height, "image_path": image_path})
    except FileNotFoundError:
        return json.dumps({"error": f"File not found: {image_path}"})
    except Exception as e:
        return json.dumps({"error": str(e)})
