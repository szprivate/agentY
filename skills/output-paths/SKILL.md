```skill
---
name: output-paths
description: Static task-type → output_path mapping for brainbriefing output_nodes. Activate during step 6 (Identify output nodes) to set the correct output_path for each output node.
allowed-tools: get_workflow_template
---

# Output Paths — Task-Type Mapping

Use this table to set `output_path` for each entry in `output_nodes`.

## Primary mapping (by task type)

| Task type          | output_path                    |
|--------------------|-------------------------------|
| `image_generation` | `./agentOut/image_generation` |
| `image_edit`       | `./agentOut/image_edit`       |
| `video_i2v`        | `./agentOut/video_i2v`        |
| `video_flf`        | `./agentOut/video_flf`        |
| `video_v2v`        | `./agentOut/video_v2v`        |
| `audio`            | `./agentOut/audio`            |
| `3d`               | `./agentOut/model`            |

---

## Inference rules for ambiguous cases

If the task type is not explicitly determinable from the user request, infer it from the output node's `class_type`:

| Output node class_type         | Inferred task type  | output_path                    |
|-------------------------------|---------------------|-------------------------------|
| `SaveImage`, `PreviewImage`    | `image_generation`  | `./agentOut/image_generation` |
| `VHS_VideoCombine`             | check input nodes — if has image input → `video_i2v`, else `video_v2v` |
| `SaveAudio`, `VHS_SaveAudio`   | `audio`             | `./agentOut/audio`            |
| `Save3DModel`, `TripoSG_*`     | `3d`                | `./agentOut/model`            |

**Image edit vs image generation**: if the workflow has any `LoadImage`, `VHS_LoadImagePath`, or similar input node AND the primary output is `SaveImage`, check the template name or user request for the word "edit" / "inpaint" / "modify". If found → `image_edit`. Otherwise → `image_generation`.

**VHS_VideoCombine ambiguity**:
- Input nodes include a LoadImage / VHS_LoadImagePath → `video_i2v` → `./agentOut/video_i2v`
- Input nodes are text-only → check if template name contains `flf` (first-last-frame) → `video_flf` → `./agentOut/video_flf`
- Otherwise → `video_v2v` → `./agentOut/video_v2v`

---

## Rules

- Every entry in `output_nodes` MUST have an `output_path`.
- If there are multiple output nodes of the same type (e.g. two SaveImage nodes), they share the same `output_path`.
- Never guess a custom path — use only the paths listed above.
```
