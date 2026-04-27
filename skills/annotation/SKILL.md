```skill
---
name: annotation
description: Handles user-provided annotated images. When the user submits a drawing or mark-up on top of an image alongside a message, this skill wires the last generated output as the primary edit target and the annotation as the second (control/reference) input, then selects an appropriate image-editing template.
allowed-tools: upload_image, get_workflow_template, get_workflow_catalog, get_image_resolution
---

# Annotation Skill

Activate this skill whenever the user attaches an **annotated image** to their message — i.e. an image the user drew on, circled, scribbled over, or otherwise marked up to indicate what should change.

Typical trigger signals (any of these is sufficient):
- The user's message contains words like: *annotation*, *annotated*, *I marked*, *I drew*, *I circled*, *here's my edit*, *see the marking*, *the scribble*, *my sketch*, *highlighted area*, *indicated area*
- The user's message explicitly states they have drawn on an image
- An image is attached and the context makes clear it is a user-made mark-up rather than a raw photo

---

## Researcher — Annotation handling

Follow these steps **instead of** the normal template-selection and input-image steps for the current turn:

### A. Identify the annotation image
- The user-attached image IS the annotation. Record it as the second input (`role: control_image`).
- Do **not** confuse the annotation with the primary subject image.

### B. Identify the primary image (first input)
Priority order:
1. **User-specified image**: If the user's message explicitly names or describes a different image as the edit target, use that image. Upload it via `upload_image` if it is a file path, and use the returned filename.
2. **Last output image**: If no specific image is named, inspect the `[CONVERSATION SUMMARY FROM PRIOR ROUND]` block for an `OUTPUT_PATHS` line. Use the **last** (most recent) path listed there.
   - Call `upload_image(file_path=<full path>)` and use the returned `name` as the filename.
3. **Fallback**: If no prior output exists and no image was specified, set a BLOCKER: *"Annotation workflow requires a base image. No prior output and no image specified by user."*

### C. Select the workflow template
Priority order:
1. **History match**: Look in the `[CONVERSATION SUMMARY FROM PRIOR ROUND]` for any `template_name` field (or equivalent template reference) from the previous turn(s). If a valid image-editing template is found there, reuse it exactly.
2. **Fallback**: If no prior template is available or the prior template is not an image-editing template, use `qwen2511_imageEdit`.

Call `get_workflow_template(<chosen_template_name>)` to load the template and retrieve its node metadata.

### D. Set brainbriefing fields
- `input_image_count`: **2**
- `input_images`: two entries — `[{ "filename": "<base_image>" }, { "filename": "<annotation_image>" }]`
- `input_nodes`: populate from `io.inputs` returned by `get_workflow_template`:
  - First input node → `role: master_image`, filename = base image
  - Second input node → `role: control_image`, filename = annotation image
- `task.type`: `image edit`
- `task.description`: summarise what the annotation instructs (e.g. "Edit the marked area of the base image according to the user's annotation")
- `template.name`: the selected template name
- `prompt.positive`: describe the desired change based on the annotation and user message. Be explicit about what area or element the annotation targets.

### E. Normal steps continue
After setting the above: continue with standard steps 6–9 of the researcher (output nodes, parameters, blockers). Do NOT re-run template selection or input-image steps — they are already resolved above.

---

## Brain — Annotation workflow assembly

When the brainbriefing contains an annotation-based task (`input_image_count == 2` with `role: control_image` on the second input), follow these additional constraints during `assemble-from-template`:

- Load the template via `get_workflow_template` as normal.
- Patch **both** input nodes:
  - Node with `role: master_image` → patch with the base image filename.
  - Node with `role: control_image` → patch with the annotation image filename.
- If the chosen template is a **Nano Banana / Nano Banana 2 / Nano Banana Pro** variant: activate the `nano-banana` skill. In the prompt, refer to the base image as `@img1` and the annotation as `@img2`. Add an instruction like: *"@img2 is a user annotation showing what to change — apply the indicated edit to @img1."*
- If the chosen template is `qwen2511_imageEdit`: the template already expects up to 3 images; place the base image in the first LoadImage node and the annotation in the second LoadImage node.
- Otherwise: proceed with standard `assemble-from-template` patching, ensuring both input nodes are populated.
```
