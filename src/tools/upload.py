"""Upload tools for ComfyUI – images and masks."""

import json
import os

from strands import tool

from src.comfyui_client import get_client


@tool
def upload_image(
    file_path: str,
    subfolder: str = "",
    image_type: str = "input",
    overwrite: bool = False,
) -> dict:
    """Upload an image file to ComfyUI for use in workflows.

    Args:
        file_path: Local path to the image file.
        subfolder: Optional subfolder inside the target directory.
        image_type: 'input', 'output', or 'temp' (default 'input').
        overwrite: Overwrite existing file with the same name.
    """
    try:
        if not os.path.isfile(file_path):
            return json.dumps({"error": f"File not found: {file_path}"})

        filename = os.path.basename(file_path)
        with open(file_path, "rb") as f:
            files = {"image": (filename, f, "image/png")}
            data = {"type": image_type, "overwrite": str(overwrite).lower()}
            if subfolder:
                data["subfolder"] = subfolder
            return json.dumps(get_client().post("/upload/image", data=data, files=files))
    except Exception as e:
        return json.dumps({"error": str(e)})


@tool
def upload_mask(
    file_path: str,
    original_ref: str,
    subfolder: str = "",
    image_type: str = "input",
    overwrite: bool = False,
) -> str:
    """Upload a mask image to ComfyUI linked to an original reference image.

    Args:
        file_path: Local path to the mask image.
        original_ref: Filename of the original image this mask is associated with.
        subfolder: Optional subfolder.
        image_type: Typically 'input'.
        overwrite: Overwrite existing file.
    """
    try:
        if not os.path.isfile(file_path):
            return json.dumps({"error": f"File not found: {file_path}"})

        filename = os.path.basename(file_path)
        with open(file_path, "rb") as f:
            files = {"image": (filename, f, "image/png")}
            data = {
                "type": image_type,
                "overwrite": str(overwrite).lower(),
                "original_ref": original_ref,
            }
            if subfolder:
                data["subfolder"] = subfolder
            return json.dumps(get_client().post("/upload/mask", data=data, files=files))
    except Exception as e:
        return json.dumps({"error": str(e)})
