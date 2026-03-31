---
name: image-downsize
description: Downsize images to fit Claude API upload limits (max 1568px long edge, max 5MB filesize, max 8000px absolute limit). Use before sending images to Claude for analysis.
allowed-tools: run_script file_read
---

# Image Downsizing for Claude API

When you need to prepare images for Claude API upload, run the downsize script.

## Limits
- **Single image**: max 8000x8000px, max 5MB (API)
- **Optimal for quality**: resize long edge to 1568px (beyond this Claude resizes server-side anyway, adding latency)
- **Multi-image requests (20+)**: max 2000x2000px per image
- **Formats**: JPEG, PNG, WebP, GIF

## Usage

Run the script with `run_script`:

```bash
python {skill_path}/scripts/downsize.py <input_path> [--output <output_path>] [--max-pixels 1568] [--max-bytes 5242880] [--multi]
```

Arguments:
- `input_path` — single image or directory of images
- `--output` — output path (default: overwrites in place)
- `--max-pixels` — long edge limit (default: 1568 for optimal; use 2000 for multi-image batches)
- `--max-bytes` — file size limit in bytes (default: 5MB = 5242880)
- `--multi` — shortcut for `--max-pixels 2000` (for 20+ image requests)
- `--quality` — JPEG/WebP quality 1-100 (default: 85)

The script outputs a JSON summary to stdout with paths and before/after dimensions.
