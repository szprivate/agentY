#!/usr/bin/env python3
"""Resize and/or downsize images.

Two modes:
  downsize  (default) — shrink to fit Claude API upload limits.
  resize              — scale to an exact resolution (--width / --height).
"""

import argparse
import json
import sys
from pathlib import Path
from PIL import Image

SUPPORTED = {".jpg", ".jpeg", ".png", ".webp", ".gif"}
DEFAULT_MAX_PX = 1568
MULTI_MAX_PX = 2000
DEFAULT_MAX_BYTES = 5 * 1024 * 1024  # 5MB


def _save(img: Image.Image, output: Path, quality: int, max_bytes: int) -> tuple[int, int]:
    """Save *img* to *output*, reducing quality if the file exceeds *max_bytes*.
    Returns the final quality value used."""
    fmt = output.suffix.lower()
    if fmt in (".jpg", ".jpeg") and img.mode == "RGBA":
        img = img.convert("RGB")
    q = quality
    while q >= 20:
        img.save(output, quality=q)
        if output.stat().st_size <= max_bytes:
            break
        q -= 5
    else:
        img.save(output, quality=20)
        q = 20
    return q


def downsize_image(
    path: Path,
    output: Path,
    max_pixels: int,
    max_bytes: int,
    quality: int,
) -> dict:
    img = Image.open(path)
    original_size = (img.width, img.height)
    original_bytes = path.stat().st_size
    resized = False

    # Resize if long edge exceeds limit
    long_edge = max(img.width, img.height)
    if long_edge > max_pixels:
        ratio = max_pixels / long_edge
        new_w = int(img.width * ratio)
        new_h = int(img.height * ratio)
        img = img.resize((new_w, new_h), Image.LANCZOS)
        resized = True

    q = _save(img, output, quality, max_bytes)

    return {
        "path": str(output),
        "original": {"size": list(original_size), "bytes": original_bytes},
        "result": {
            "size": [img.width, img.height],
            "bytes": output.stat().st_size,
            "quality": q,
        },
        "resized": resized,
    }


def resize_image(
    path: Path,
    output: Path,
    width: int | None,
    height: int | None,
    fit: str,
    quality: int,
    max_bytes: int,
) -> dict:
    """Resize to an exact resolution.

    fit modes:
      contain  — scale to fit inside (width x height), preserving aspect ratio (default).
      cover    — scale to cover (width x height), cropping the excess, preserving AR.
      exact    — stretch / squash to exactly (width x height); may distort.
    """
    img = Image.open(path)
    original_size = (img.width, img.height)
    original_bytes = path.stat().st_size

    # If only one dimension given, derive the other from the aspect ratio.
    ar = img.width / img.height
    if width and not height:
        height = max(1, round(width / ar))
    elif height and not width:
        width = max(1, round(height * ar))

    target_w, target_h = int(width), int(height)  # type: ignore[arg-type]

    if fit == "exact":
        img = img.resize((target_w, target_h), Image.LANCZOS)
    elif fit == "cover":
        # Scale so the image COVERS the target, then crop to exact size.
        scale = max(target_w / img.width, target_h / img.height)
        scaled_w = round(img.width * scale)
        scaled_h = round(img.height * scale)
        img = img.resize((scaled_w, scaled_h), Image.LANCZOS)
        left = (scaled_w - target_w) // 2
        top = (scaled_h - target_h) // 2
        img = img.crop((left, top, left + target_w, top + target_h))
    else:  # contain
        # Scale so the image FITS inside the target, no cropping.
        scale = min(target_w / img.width, target_h / img.height)
        new_w = max(1, round(img.width * scale))
        new_h = max(1, round(img.height * scale))
        img = img.resize((new_w, new_h), Image.LANCZOS)

    q = _save(img, output, quality, max_bytes)

    return {
        "path": str(output),
        "original": {"size": list(original_size), "bytes": original_bytes},
        "result": {
            "size": [img.width, img.height],
            "bytes": output.stat().st_size,
            "quality": q,
            "fit": fit,
        },
        "resized": True,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Resize images (exact resolution) or downsize for Claude API limits."
    )
    parser.add_argument("input", type=Path, help="Single image or directory of images")
    parser.add_argument("--output", type=Path, default=None, help="Output path (default: overwrite in place)")
    parser.add_argument("--quality", type=int, default=85, help="JPEG/WebP quality 1-100 (default: 85)")
    parser.add_argument("--max-bytes", type=int, default=DEFAULT_MAX_BYTES, help=f"File size limit in bytes (default: {DEFAULT_MAX_BYTES})")

    # Exact-resize mode
    resize_grp = parser.add_argument_group("Exact resize (--width / --height)")
    resize_grp.add_argument("--width", type=int, default=None, help="Target width in pixels")
    resize_grp.add_argument("--height", type=int, default=None, help="Target height in pixels")
    resize_grp.add_argument(
        "--fit",
        choices=["contain", "cover", "exact"],
        default="contain",
        help="contain: fit inside (default) | cover: fill & crop | exact: stretch to exact size",
    )

    # Downsize mode
    downsize_grp = parser.add_argument_group("Downsize (Claude API limits)")
    downsize_grp.add_argument("--max-pixels", type=int, default=DEFAULT_MAX_PX, help=f"Long edge limit (default: {DEFAULT_MAX_PX})")
    downsize_grp.add_argument("--multi", action="store_true", help=f"Shortcut for --max-pixels {MULTI_MAX_PX} (for 20+ image requests)")
    args = parser.parse_args()

    exact_resize_mode = args.width is not None or args.height is not None

    if not exact_resize_mode and args.multi:
        args.max_pixels = MULTI_MAX_PX

    inputs = []
    if args.input.is_dir():
        inputs = [f for f in args.input.iterdir() if f.suffix.lower() in SUPPORTED]
    elif args.input.suffix.lower() in SUPPORTED:
        inputs = [args.input]
    else:
        print(json.dumps({"error": f"Unsupported format: {args.input}"}))
        sys.exit(1)

    if not inputs:
        print(json.dumps({"error": "No supported images found", "path": str(args.input)}))
        sys.exit(1)

    results = []
    for img_path in sorted(inputs):
        out = args.output if args.output and not args.output.is_dir() else None
        if out is None:
            if args.output and args.output.is_dir():
                out = args.output / img_path.name
            else:
                out = img_path  # overwrite in place
        if exact_resize_mode:
            results.append(
                resize_image(img_path, out, args.width, args.height, args.fit, args.quality, args.max_bytes)
            )
        else:
            results.append(
                downsize_image(img_path, out, args.max_pixels, args.max_bytes, args.quality)
            )

    print(json.dumps({"processed": len(results), "images": results}, indent=2))


if __name__ == "__main__":
    main()
