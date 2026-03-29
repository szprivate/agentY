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
    """Upload an image file to ComfyUI so it can be used in workflows.

    Args:
        file_path: Absolute or relative path to the image file on the local filesystem.
        subfolder: Optional subfolder inside the target directory.
        image_type: The image type/directory. Typically 'input', 'output', or 'temp'.
        overwrite: If True, overwrite an existing file with the same name.

    Returns:
        A dictionary with the upload result including the final filename.
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
    """Upload a mask image to ComfyUI, associated with an original reference image.

    Args:
        file_path: Absolute or relative path to the mask image file.
        original_ref: The filename of the original image this mask is associated with.
        subfolder: Optional subfolder inside the target directory.
        image_type: The image type/directory. Typically 'input'.
        overwrite: If True, overwrite an existing file with the same name.

    Returns:
        A dictionary with the upload result.
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
