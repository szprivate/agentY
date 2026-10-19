---
name: dashscope-generate
description: Generate an image or video with an Alibaba DashScope (Model Studio) model - Wan, HappyHorse, Qwen-Image - billed to the user's own DashScope key instead of ComfyUI partner credits. Activate when the user names a DashScope / Alibaba / Wan / HappyHorse / Qwen-Image model, or when a partner node fails with "Payment Required" (Comfy credits exhausted) and a DashScope model can do the job.
allowed-tools: dashscope_generate, dashscope_fetch
---

# Generating with DashScope directly

`dashscope_generate(kind, model, prompt, image_paths=...)` calls the DashScope API itself and
returns saved files; in agentY they appear in the chat and on the canvas like any other output.
No workflow, no ComfyUI run, no Comfy credits.

## When
- The user asks for a DashScope / Alibaba model by name ("with Wan 2.7", "HappyHorse", "qwen-image").
- A partner-node run failed with **Payment Required**: say so, and offer the DashScope equivalent
  (do not switch silently: it is a different model and a different bill).
- Not for a workflow the user wants on their canvas to edit, local OSS models, or anything a
  ComfyUI template already does well on credits that are available.

## Picking the model (exact ids)
| Want | Model |
|---|---|
| image, text to image / edit | `qwen-image-2.0` (`-pro` for best quality); give `image_paths` to edit |
| video, text to video | `wan2.7-t2v`, `wan2.6-t2v`, `happyhorse-1.0-t2v` |
| video, image to video | `wan2.7-i2v`, `wan2.6-i2v-flash`, `happyhorse-1.0-i2v`; cheapest `wan2.2-i2v-flash` (480P, 5 s) |
| video, reference to video | `wan2.7-r2v`, `wan2.6-r2v-flash`, `happyhorse-1.0-r2v`: name subjects character1, character2... in the prompt |

Full list and parameters: `docs/dashscope_generation.md`. An unknown id comes back as
`error_kind: unknown_model`.

## Use
- Input images: pass local paths or URLs in `image_paths`; the tool encodes them (10 MB each at most).
- Videos take 1-5 minutes and the call waits. Prefer 720P for finals; do not make several at once
  without telling the user: every call is billed.
- The result's `usage` is what DashScope billed (seconds / image count / resolution). Tell the user.

## Errors
- `balance`: DashScope account out of money or quota. Tell the user; do not retry.
- `content_refusal`: reword the prompt (or pick another reference image). One reroll with a new
  seed is fair when the refusal came after generating, not when the input itself was refused.
- `timeout`: not lost. Call `dashscope_fetch(task_id)` later.
- `auth`: `DASHSCOPE_API_KEY` missing or wrong in `.env`.
