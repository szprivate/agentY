"""
Image handling tools – upload, download, resolution, and visual analysis.

Consolidates all image-related @tool functions:
  • upload_image: push images to ComfyUI's input folder
  • view_image: download images from ComfyUI's output
  • get_image_resolution: read local image dimensions
  • analyze_image: forward an image to the model for visual inspection
"""

import io
import json
import os
from pathlib import Path
from typing import Optional

import requests
from PIL import Image
from strands import tool

from src.utils.comfyui_client import get_client


# ═══════════════════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════════════════

_MAX_IMAGE_BYTES = 5 * 1024 * 1024   # 5 MB hard limit (Claude API)
_OPTIMAL_LONG_EDGE = 1568            # Claude resizes beyond this anyway

_FORMAT_MAP: dict[str, str] = {
    "png":  "png",
    "jpg":  "jpeg",
    "jpeg": "jpeg",
    "gif":  "gif",
    "webp": "webp",
}


# ═══════════════════════════════════════════════════════════════════════════════
# Internal helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _detect_format(path_or_name: str, mime: str = "") -> Optional[str]:
    """Resolve the Strands image format string from a filename or MIME type."""
    ext = Path(path_or_name).suffix.lstrip(".").lower()
    fmt = _FORMAT_MAP.get(ext)
    if fmt:
        return fmt
    if mime.startswith("image/"):
        sub = mime.split("/")[-1].lower()
        return _FORMAT_MAP.get(sub)
    return None


def _downsize(data: bytes, img_fmt: str) -> bytes:
    """Downsize image in-memory to fit Claude API constraints.

    Caps long edge at 1568 px and enforces the 5 MB hard limit.
    """
    if len(data) <= _MAX_IMAGE_BYTES:
        img = Image.open(io.BytesIO(data))
        if max(img.width, img.height) <= _OPTIMAL_LONG_EDGE:
            return data

    img = Image.open(io.BytesIO(data))
    long_edge = max(img.width, img.height)

    if long_edge > _OPTIMAL_LONG_EDGE:
        ratio = _OPTIMAL_LONG_EDGE / long_edge
        new_w, new_h = int(img.width * ratio), int(img.height * ratio)
        img = img.resize((new_w, new_h), Image.LANCZOS)

    pil_fmt = "PNG" if img_fmt == "png" else "JPEG"
    if img.mode == "RGBA" and pil_fmt == "JPEG":
        img = img.convert("RGB")

    buf = io.BytesIO()
    quality = 90
    while quality >= 20:
        buf.seek(0)
        buf.truncate()
        if pil_fmt == "JPEG":
            img.save(buf, format=pil_fmt, quality=quality, optimize=True)
        else:
            img.save(buf, format=pil_fmt, optimize=True)
        if buf.tell() <= _MAX_IMAGE_BYTES:
            break
        if pil_fmt == "PNG":
            pil_fmt = "JPEG"
            if img.mode == "RGBA":
                img = img.convert("RGB")
            continue
        quality -= 10

    return buf.getvalue()


# ═══════════════════════════════════════════════════════════════════════════════
# Tools
# ═══════════════════════════════════════════════════════════════════════════════

@tool
def upload_image(
    file_path: str,
    subfolder: str = "",
    image_type: str = "input",
    overwrite: bool = False,
) -> dict:
    """Upload an image file to the ComfyUI input directory for use in workflows.

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
def view_image(
    filename: str,
    save_to: str,
    subfolder: str = "",
    image_type: str = "output",
) -> str:
    """Download an image from the ComfyUI output directory and save it to a local path.

    After saving, use analyze_image(file_path=save_to) to inspect the image
    contents.

    Args:
        filename: Image filename on the server e.g. 'ComfyUI_00001_.png'.
        save_to: Local file path to save the image. Required.
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


@tool
def analyze_image(
    file_path: str = "",
    image_url: str = "",
    question: str = "",
) -> dict:
    """Load an image from a local file or URL and forward it to the model for visual analysis.

    Supported formats: PNG, JPEG/JPG, GIF, WEBP.
    Images are automatically downsized to satisfy Claude's 5 MB / 1568 px constraints.

    Args:
        file_path: Absolute or relative path to a local image file.
                   Provide either this or ``image_url`` – not both.
        image_url: Public http/https URL of an image to download.
        question:  Optional specific question about the image.
    """
    data: Optional[bytes] = None
    source_name = ""
    detected_mime = ""

    if file_path:
        p = Path(file_path).expanduser()
        if not p.exists():
            p = Path(os.getcwd()) / file_path
        if not p.exists():
            return {"status": "error", "content": [{"text": f"File not found: {file_path}"}]}
        source_name = str(p)
        try:
            data = p.read_bytes()
        except Exception as exc:
            return {"status": "error", "content": [{"text": f"Could not read file: {exc}"}]}

    elif image_url:
        source_name = image_url
        try:
            resp = requests.get(image_url, timeout=30)
            resp.raise_for_status()
            detected_mime = resp.headers.get("content-type", "")
            data = resp.content
        except Exception as exc:
            return {"status": "error", "content": [{"text": f"Could not download image: {exc}"}]}

    else:
        return {"status": "error", "content": [{"text": "Provide either file_path or image_url."}]}

    # Detect format
    img_fmt = _detect_format(source_name, detected_mime)
    if img_fmt is None:
        if data[:4] == b"\x89PNG":
            img_fmt = "png"
        elif data[:3] == b"\xff\xd8\xff":
            img_fmt = "jpeg"
        elif data[:6] in (b"GIF87a", b"GIF89a"):
            img_fmt = "gif"
        elif data[:4] == b"RIFF" and data[8:12] == b"WEBP":
            img_fmt = "webp"
        else:
            return {"status": "error", "content": [{"text": f"Unsupported or undetectable image format for: {source_name}"}]}

    # Downsize if needed
    original_size = len(data)
    data = _downsize(data, img_fmt)
    downsized = len(data) < original_size

    # Build multimodal ToolResult
    info_parts = [
        f"Image loaded from: {source_name}",
        f"Format: {img_fmt.upper()}, Size: {len(data):,} bytes",
    ]
    if downsized:
        info_parts.append(f"(downsized from {original_size:,} bytes to fit API limits)")
    if question:
        info_parts.append(f"\nUser question: {question}")

    return {
        "status": "success",
        "content": [
            {"text": "\n".join(info_parts)},
            {
                "image": {
                    "format": img_fmt,
                    "source": {"bytes": data},
                }
            },
        ],
    }
