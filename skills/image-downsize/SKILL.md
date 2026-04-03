---
name: image-downsize
description: Resize images to a specific resolution OR downsize to fit Claude API upload limits (max 1568px long edge, max 5MB). Use the exact-resize mode when the user requests a specific width/height; use downsize mode before sending images to Claude for analysis.
allowed-tools: run_script file_read
---

## When to use this skill

Only use this skill when the user explicitly asks to resize or downsize an image. Do NOT resize automatically.

Step-by-step:
1. Call `get_image_resolution` to get the current width and height.
2. If the user gave a relative size ("half the size", "2×", "75%", etc.), use `calculator` to derive the exact target pixels from the current dimensions.
3. Run `downsize.py` via `run_script` with `--width` / `--height` for the resolved resolution.

# Image Resize & Downsize

Two modes in one script:

| Mode | When to use | Key args |
|---|---|---|
| **Exact resize** | User asks for a specific resolution | `--width` and/or `--height` |
| **Downsize** | Preparing images for Claude API upload | `--max-pixels` / `--multi` |

## Exact resize mode

Resize an image to a specific resolution. When only one dimension is given, the other is derived from the aspect ratio.

```bash
python {skill_path}/scripts/downsize.py <input_path> --width 1920 --height 1080 [--fit contain|cover|exact] [--output <output_path>]
```

`--fit` modes (default: `contain`):
- `contain` — scale to fit **inside** the target box, preserving aspect ratio (no cropping)
- `cover` — scale to **fill** the target box, cropping the excess, preserving aspect ratio
- `exact` — stretch/squash to exactly the target size (may distort)

Examples:
```bash
# Resize to 1920x1080, letterboxed (no crop)
python {skill_path}/scripts/downsize.py photo.jpg --width 1920 --height 1080

# Resize to 1920x1080, cropping to fill
python {skill_path}/scripts/downsize.py photo.jpg --width 1920 --height 1080 --fit cover

# Resize width to 512px, height from aspect ratio
python {skill_path}/scripts/downsize.py photo.jpg --width 512

# Save to a new file
python {skill_path}/scripts/downsize.py photo.jpg --width 1024 --height 1024 --fit cover --output resized.jpg
```

## Downsize mode (Claude API limits)

```bash
python {skill_path}/scripts/downsize.py <input_path> [--output <output_path>] [--max-pixels 1568] [--max-bytes 5242880] [--multi]
```

- `--max-pixels` — long edge limit (default: 1568; the optimal quality threshold for Claude)
- `--max-bytes` — file size limit in bytes (default: 5 MB)
- `--multi` — shortcut for `--max-pixels 2000` (for batches of 20+ images)

## Common arguments

- `input_path` — single image or directory of images
- `--output` — output path (default: overwrites in place)
- `--quality` — JPEG/WebP quality 1-100 (default: 85)
- `--max-bytes` — enforced in both modes

## Supported formats

JPEG, PNG, WebP, GIF

## Output

```json
{
  "processed": 1,
  "images": [{
    "path": "/path/to/output.jpg",
    "original": {"size": [3000, 2000], "bytes": 4200000},
    "result":   {"size": [1920, 1080], "bytes": 980000, "quality": 85, "fit": "contain"},
    "resized": true
  }]
}
```
