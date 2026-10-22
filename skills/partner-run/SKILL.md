---
name: partner-run
description: Run ONE partner (API) node directly - Seedance, Kling, GPT Image, Nano Banana / Gemini image, Ideogram, Meshy, Veo, Runway, Hailuo ... - with partner_run, no template search. Activate when the request names a partner model or node and the job is that node alone (text/image/first-frame/reference to image or video or 3D); for the settings spelling of dynamic combos and reference slots, and for N shots of one node.
allowed-tools: partner_run, run_variants, upload_image, analyze_image, get_node_schema, search_nodes
---

# partner_run - one partner node, built from its own schema

`partner_run(node_type, inputs, settings, save, name, run=True)` replaces: search the
catalog -> `prepare_workflow` -> validate -> `signal_workflow_ready`. It reads the node's
schema, loads each file into the slot you name, sets the widgets, adds the save node,
validates and **runs** it (partner credits - once per request, never to test). The outputs
come back like `run_workflow_now`'s and land on the canvas. Do not also signal the file.

## When

- The request names the model or node: "Seedance 2.5 first-frame", "GPT Image 2 edit",
  "Nano Banana", "Kling start-end", "Ideogram", "Meshy image to 3D".
- The job is that node alone. Several stages (generate, then upscale, then animate), a
  local model, or a pipeline of nodes: `prepare_workflow` / hooks as before.
- Unsure of the node's name? Give your best name ("Seedance 2.5 first last frame"); an
  ambiguous or unknown name builds nothing and returns candidates - pick one and call again.

## Arguments

- `node_type`: class name (`ByteDance2FirstLastFrameNode`) or display name.
- `inputs`: FILE slots only: `{"first_frame": "D:/shots/a.png"}`. A slot GROUP takes a list
  and fills `image_1, image_2 ...` in order: `{"model.reference_images": ["a.png", "b.png"],
  "model.reference_videos": "motion.mp4"}`. One slot: `model.reference_images.image_3`.
  Local files only - download a URL first (`download_image`). Images, videos and audio load.
- `settings`: the prompt and every widget. **Dynamic combos** (`model`) pick an option, and the
  option decides which other settings exist - they are addressed `model.resolution`,
  `model.duration`, `model.generate_audio`. You may write the bare name (`resolution`) when it
  is unambiguous. Examples: `{"model": "Seedance 2.5", "prompt": "...", "resolution": "480p",
  "duration": 5, "generate_audio": false}`.
- `save`: "image" | "video" | "auto" (by what the node outputs). A node that is its own
  output (Meshy) gets no save node. `name`: short name for the files.
- `run=False`: build + validate only; returns `workflow_path`.

An option or value the node does not offer is an ERROR that lists the options (nothing is
changed silently); a number out of range is clamped and reported in `warnings`. A required file
slot left empty names the slot.

## Patterns

- **First-frame video** (Seedance 2.5): `node_type="Seedance 2.5 First-Last-Frame to Video"`,
  `inputs={"first_frame": path}`, `settings={"model": "Seedance 2.5", "prompt": ..., ...}`.
  Reference mode (`ByteDance2ReferenceNodeV2`) is for subject/style/motion references, not for
  "animate this frame".
- **No audio / 480p / duration**: set `generate_audio=false`, `resolution="480p"`, `duration=N`.
  Check the node's own options (Seedance Draft is 480p only; 2.5 allows 4-30 s).
- **Image edit with references** (GPT Image / Nano Banana): `inputs={"image": path}`; several
  references go in the node's multi-image slot (`get_node_schema` names it).
- **One node, several shots** ("each starting frame", "4 prompts"): call `partner_run(run=False)`
  once for the first shot, then `run_variants(workflow_path, variants=[{"label": "shot 1",
  "set": {"<loader id>.image": "frame1.png", "1.model.prompt": "..."}}, ...])` - or, when the
  shots are few, N parallel `partner_run` calls in ONE message (they are independent).
- **Cut into one film**: that is a separate stage after the shots exist (concatenate the
  videos); do not look for a template that does both.

## Do not

- Search templates first, or call `prepare_workflow`, for a request that names a partner node.
- Call it again for the same request "to check" - it spends money. Read the result.
- Put a file path in `settings` or a plain value in `inputs`: the error says which is which.
- Use a template whose name merely sounds right (a storyboard template for a first-frame job).
