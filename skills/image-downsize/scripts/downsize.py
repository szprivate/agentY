#!/usr/bin/env python3
"""Downsize images to fit Claude API upload constraints."""

import argparse
import json
import sys
from pathlib import Path
from PIL import Image

SUPPORTED = {".jpg", ".jpeg", ".png", ".webp", ".gif"}
DEFAULT_MAX_PX = 1568
MULTI_MAX_PX = 2000
DEFAULT_MAX_BYTES = 5 * 1024 * 1024  # 5MB


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

    # Convert RGBA to RGB for JPEG output
    fmt = path.suffix.lower()
    if fmt in (".jpg", ".jpeg") and img.mode == "RGBA":
        img = img.convert("RGB")

    # Save with quality, then check filesize — reduce quality if needed
    q = quality
    while q >= 20:
        img.save(output, quality=q)
        if output.stat().st_size <= max_bytes:
            break
        q -= 5
        resized = True
    else:
        # Last resort: save at minimum quality
        img.save(output, quality=20)

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


def main():
    parser = argparse.ArgumentParser(description="Downsize images for Claude API upload limits.")
    parser.add_argument("input", type=Path, help="Single image or directory of images")
    parser.add_argument("--output", type=Path, default=None, help="Output path (default: overwrite in place)")
    parser.add_argument("--max-pixels", type=int, default=DEFAULT_MAX_PX, help=f"Long edge limit (default: {DEFAULT_MAX_PX})")
    parser.add_argument("--max-bytes", type=int, default=DEFAULT_MAX_BYTES, help=f"File size limit in bytes (default: {DEFAULT_MAX_BYTES})")
    parser.add_argument("--multi", action="store_true", help=f"Shortcut for --max-pixels {MULTI_MAX_PX} (for 20+ image requests)")
    parser.add_argument("--quality", type=int, default=85, help="JPEG/WebP quality 1-100 (default: 85)")
    args = parser.parse_args()

    if args.multi:
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
        results.append(
            downsize_image(img_path, out, args.max_pixels, args.max_bytes, args.quality)
        )

    print(json.dumps({"processed": len(results), "images": results}, indent=2))


if __name__ == "__main__":
    main()
