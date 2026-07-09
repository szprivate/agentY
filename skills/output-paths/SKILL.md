---
name: output-paths
description: Media-kind output_path mapping for brainbriefing output_nodes. Activate during step 6 (Identify output nodes) to set the correct output_path for each output node.
allowed-tools: get_workflow_template, get_agent_output_dirs
---

# Output Paths / Media-Kind Mapping

Every agent-generated file goes into a small, fixed set of folders under the
ComfyUI output directory, keyed by the **kind of media** the saver node produces
— **not** by task type. Set `output_path` on each entry in `output_nodes`
accordingly.

## Mapping (by output node class_type)

| Output node class_type                                   | Media kind | output_path     |
|----------------------------------------------------------|------------|-----------------|
| `SaveImage`, `PreviewImage`, `SaveAnimatedPNG/WEBP`      | image      | `agent/images`  |
| `VHS_VideoCombine`, `SaveVideo`, `SaveWEBM`              | video      | `agent/videos`  |
| `SaveAudio`, `VHS_SaveAudio`, `SaveAudioMP3/Opus`        | audio      | `agent/audio`   |
| `SaveGLB`, `SaveGLTF`, `Save3DModel`, `TripoSG_*`         | 3d model   | `agent/models`  |

`VHS_VideoCombine` in **audio-only** mode (no image/frame input) is audio →
`agent/audio`; otherwise it is video → `agent/videos`.

## Rules

- Every entry in `output_nodes` MUST have an `output_path` from the table above.
- Paths are **relative to the ComfyUI output directory**
  (`get_comfyui_dirs().output_dir`). Do not prepend the base dir yourself and do
  not invent custom folders.
- Multiple savers of the same media kind share the same `output_path`.
- **This routing is enforced automatically** by `apply_brainbriefing`: it derives
  the bucket from each saver's `class_type` and rewrites `filename_prefix` to
  `agent/<kind>/<name>`. Your `output_path` supplies the descriptive filename
  stem, so images always end up in `agent/images/` and videos in `agent/videos/`
  regardless — but set it correctly anyway so the briefing reads true.

## Producing media with a script instead of a workflow

If you generate an image/video by running a Python script (not a ComfyUI
workflow), call **`get_agent_output_dirs()`** and write into the **absolute**
`images` / `videos` folders it returns — the very same buckets. Never scatter
files in the ComfyUI output root or the working directory.
