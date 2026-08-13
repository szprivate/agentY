---
name: seedance-reference
description: Seedance 2.0/2.5 node wiring (ByteDance2ReferenceNode, ByteDance2TextToVideoNode, ByteDance2FirstLastFrameNode). Activate in the Query Templates when a Seedance template is selected, and in the Assemble Workflow when patching one — it carries the model-version choice, the reference-slot binding order, and which widgets each task locks. Prompt wording lives in the `prompting` skill's seedance fragment, not here.
allowed-tools: update_workflow, get_workflow_template, check_node_types
---

# Seedance 2.0 / 2.5 — node and parameter skill

The `prompting` skill's `references/seedance.md` says what to *write*. This says where
it *goes* and what to set around it. Both are needed for a Seedance run; neither
covers the other.

## When to activate
- **Researcher**: the selected template drives any `ByteDance2*` node, or the request
  is a Seedance video (reference-to-video, first/last frame, edit, extension).
- **Assembly**: patching such a template.
- Not for Seedance **1.x** nodes (`ByteDanceTextToVideoNode`, `ByteDanceImageToVideoNode`,
  `ByteDanceFirstLastFrameNode`, `ByteDanceImageReferenceNode`) — different, older schema.

## The three nodes

| Node | Use for | Locked by the task |
|---|---|---|
| `ByteDance2TextToVideoNode` | text → video, no references | nothing |
| `ByteDance2ReferenceNode` | any reference job: subjects, style, motion, keyframes, storyboard, **and video editing** | `ratio: adaptive` when a reference video sets the frame |
| `ByteDance2FirstLastFrameNode` | first frame, or first + last | **no `ratio` widget at all** — the first frame's aspect ratio wins |

`model` is a **dynamic combo**: each option carries its own set of nested widgets, so
switching versions is not a one-string edit. Read the live schema
(`check_node_types` / `/object_info`) before patching, and prefer a template that
already pins the version you want.

## Version choice

| | 2.0 · Fast · Mini | **2.5** |
|---|---|---|
| `duration` | 4–15 s (default 7) | **4–30 s** (default 5) |
| `resolution` | 480p/720p/1080p/**4k** (2.0 only; Fast/Mini cap at 720p) | 480p / **720p** |
| reference slots | 9 image · 3 video · 3 audio · 9 asset | **30 · 10 · 10 · 30** |
| `video_editing` | — | present (boolean) |

Pick **2.5** whenever the job needs >15 s, >9 reference images, real timestamps, or
video editing. Pick **2.0** when the user wants 1080p/4K, which 2.5 does not offer.
Fast/Mini are cheap tiers — tests, not finals.

## Binding references — the part that goes wrong

The prompt cites `@Image1`, `@Video1`, `@Audio1`. Those are **slot positions, not
names**: `@Image1` is the `image_1` input of `reference_images`, `@Video2` is `video_2`,
and so on. The inputs are autogrow lists — wire them in the exact order the prompt
cites them, with no gaps. A mismatch here silently swaps which character is which.

- Re-order the *wires* to match the prompt, or renumber the *prompt* to match the
  wires. Never leave them disagreeing.
- `reference_assets` takes Seedance asset ids (STRING, `forceInput`) produced by
  `ByteDanceCreateImageAsset` / `ByteDanceCreateVideoAsset` — that is the real-person
  verification path, not a file input. Only use it when the user asked for identity
  verification or supplied a `group_id`.
- Total reference **video** duration ≤30 s, same for audio. Over that, drop assets
  rather than trimming the prompt's references.

## Widgets

- `duration` — user's intent, clamped to the version's range. It is **not** written
  into the prompt text.
- `ratio` — `adaptive` whenever a reference video or a locking frame decides the shape;
  otherwise the ratio the user asked for. Absent on the first/last-frame node.
- `resolution` — 720p default. 1080p/4K exist on 2.0 only.
- `generate_audio` — **on by default**; Seedance generates synced audio. Turn it off
  only when the user asks for a silent clip. `no BGM` / `no subtitles` still go in the
  prompt, not here.
- `output_format` — **`mp4` is the only option** on these nodes. ByteDance's own docs
  recommend `mov` for edit/extension continuity; it is not available here. Do not
  promise seamless colour/audio continuity on a long extension chain.
- `video_editing` (2.5, reference node) — **on** when the prompt edits a connected
  reference video (replace a subject, remove an object, change the audio). Off for
  every other reference job, including a blockout re-render, which is generation.
- `auto_downscale` (default on) / `auto_upscale` (default off) — reference videos
  outside the model's pixel budget. Leave as they are unless a run fails on it.
- `watermark` — off unless asked.
- `seed` — controls re-runs only; output is non-deterministic regardless.

## Task routing, in node terms

Decide **one** primary task; the full procedure and its edge cases are in the
`prompting` skill's `references/seedance-2.5-full.md`, section "Select One Primary Task".

- Target ratio or duration differs from the source video → **not** an edit. Treat it as
  generation with the video as a content reference, and leave `video_editing` off.
- "Continue / extend before or after" → extension: the source video's ratio is locked,
  duration stays configurable.
- Modify a specific object, region or sound while ratio and duration stay as they are
  → editing: `video_editing` on, `ratio: adaptive`.
- A first/last frame that must be exact → the FLF node with `first_frame` / `last_frame`.
  Passing the same images as ordinary references instead only *approximates* them.
  Give both frames the same aspect ratio or the last one is stretched.
