"""Vision tool – forward a local image file or URL to the model for analysis.

The model cannot read image bytes by itself from a file path; this tool
loads the image, applies the same size/resolution constraints used for
Slack uploads (5 MB hard limit, 1568 px long-edge cap), and returns the
image bytes as a Strands multimodal content block so the model can perform
visual analysis.

Supported sources
-----------------
* ``file_path`` – absolute or relative path to a local image file.
* ``image_url`` – publicly reachable URL (http/https) to fetch.

Supported formats: PNG, JPEG/JPG, GIF, WEBP.
"""

import io
import os
from pathlib import Path
from typing import Optional

import requests
from strands import tool

# ---------------------------------------------------------------------------
# Constants – kept in sync with slack_server.py
# ---------------------------------------------------------------------------

_MAX_IMAGE_BYTES = 5 * 1024 * 1024   # 5 MB hard limit (Claude API)
_OPTIMAL_LONG_EDGE = 1568            # Claude resizes beyond this anyway

# Map common extensions / MIME sub-types → Strands/Bedrock format string
_FORMAT_MAP: dict[str, str] = {
    "png":  "png",
    "jpg":  "jpeg",
    "jpeg": "jpeg",
    "gif":  "gif",
    "webp": "webp",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _detect_format(path_or_name: str, mime: str = "") -> Optional[str]:
    """Resolve the Strands image format string from a filename or MIME type."""
    ext = Path(path_or_name).suffix.lstrip(".").lower()
    fmt = _FORMAT_MAP.get(ext)
    if fmt:
        return fmt
    # Fallback: parse MIME type (e.g. "image/jpeg" → "jpeg")
    if mime.startswith("image/"):
        sub = mime.split("/")[-1].lower()
        return _FORMAT_MAP.get(sub)
    return None


def _downsize(data: bytes, img_fmt: str) -> bytes:
    """Downsize *data* in-memory to fit Claude API constraints.

    Caps long edge at 1568 px and enforces the 5 MB hard limit.
    Returns the (possibly recompressed) image bytes.
    """
    try:
        from PIL import Image
    except ImportError:
        # Pillow not available – return as-is; Claude will reject if too large
        return data

    if len(data) <= _MAX_IMAGE_BYTES:
        img = Image.open(io.BytesIO(data))
        if max(img.width, img.height) <= _OPTIMAL_LONG_EDGE:
            return data  # Nothing to do

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


# ---------------------------------------------------------------------------
# Tool
# ---------------------------------------------------------------------------

@tool
def analyze_image(
    file_path: str = "",
    image_url: str = "",
    question: str = "",
) -> dict:
    """Load an image from a local file path or URL and forward it to the model for visual analysis.

    Use this tool whenever the user provides an image path or URL and asks you
    to look at, describe, analyse, compare, or reason about its contents.
    The tool returns the image as a multimodal content block so you can see
    it directly in your context window.

    Images are automatically downsized to satisfy Claude's 5 MB / 1568 px
    constraints before being forwarded.

    Args:
        file_path: Absolute or relative path to a local image file
                   (PNG, JPEG, GIF, WEBP).  Provide either this or
                   ``image_url`` – not both.
        image_url: Public http/https URL of an image to download and
                   analyse.  Provide either this or ``file_path``.
        question:  Optional specific question to answer about the image.
                   If omitted the model will give a general description.

    Returns:
        A list of Strands content blocks (text + image) that the model
        can process directly.
    """
    # ------------------------------------------------------------------ #
    # 1. Load raw bytes                                                    #
    # ------------------------------------------------------------------ #
    data: Optional[bytes] = None
    source_name = ""
    detected_mime = ""

    if file_path:
        p = Path(file_path).expanduser()
        # Try the path as given, then relative to cwd
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

    # ------------------------------------------------------------------ #
    # 2. Detect format                                                     #
    # ------------------------------------------------------------------ #
    img_fmt = _detect_format(source_name, detected_mime)
    if img_fmt is None:
        # Last-resort: sniff magic bytes
        if data[:4] == b"\x89PNG":
            img_fmt = "png"
        elif data[:3] in (b"\xff\xd8\xff",):
            img_fmt = "jpeg"
        elif data[:6] in (b"GIF87a", b"GIF89a"):
            img_fmt = "gif"
        elif data[:4] == b"RIFF" and data[8:12] == b"WEBP":
            img_fmt = "webp"
        else:
            return {"status": "error", "content": [{"text": f"Unsupported or undetectable image format for: {source_name}"}]}

    # ------------------------------------------------------------------ #
    # 3. Downsize if needed                                                #
    # ------------------------------------------------------------------ #
    original_size = len(data)
    data = _downsize(data, img_fmt)
    downsized = len(data) < original_size

    # ------------------------------------------------------------------ #
    # 4. Build multimodal ToolResult                                       #
    # ------------------------------------------------------------------ #
    # IMPORTANT: returning {"status": "success", "content": [...]} is the
    # only way Strands skips JSON-serialisation and keeps the image block
    # intact.  The Anthropic model adapter then converts bytes → base64
    # automatically before sending to the API.
    intro_parts = [f"Image loaded from: {source_name}"]
    intro_parts.append(f"Format: {img_fmt.upper()}, Size: {len(data):,} bytes")
    if downsized:
        intro_parts.append(f"(downsized from {original_size:,} bytes to fit API limits)")
    if question:
        intro_parts.append(f"\nUser question: {question}")
    else:
        intro_parts.append("\nPlease describe and analyse the image.")

    return {
        "status": "success",
        "content": [
            {"text": "\n".join(intro_parts)},
            {
                "image": {
                    "format": img_fmt,
                    "source": {"bytes": data},
                }
            },
        ],
    }
