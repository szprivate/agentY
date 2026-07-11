# Other models — prompt fragments (holding pen)

Models not yet split into their own fragment file. Same compact schema — split any
of these into `references/<model>.md` and add an index row when it earns its own page.

```yaml
- model: SD 1.5
  kind: image
  negative: CRITICAL — quality tags + embeddings
  positive_ex: "(masterpiece:1.2), (best quality:1.2), subject, details, style"
  negative_ex: "worst quality, low quality, blurry, bad anatomy, bad hands, extra fingers, watermark, embedding:easynegative"
  style: tag-based (danbooru) works well; keep < 77 tokens or use BREAK
  embeddings: [embedding:easynegative, embedding:badhandv4]

- model: SDXL
  kind: image
  negative: moderate importance
  style: natural language preferred over tags; dual CLIP ~154 tokens native (no BREAK unless > ~154)
  advanced: CLIPTextEncodeSDXL for separate text_g (global concept) / text_l (local detail)
  turbo_lightning: minimal or empty negative
  embeddings: [embedding:negativeXL_D]

- model: SD3 / SD3.5
  kind: image
  encoders: triple CLIP (CLIP-L + CLIP-G + T5-XXL)
  style: long natural-language prompts
  negative: minimal ("low quality, blurry" is enough)

- model: Qwen Image Edit (2511 / fp8) / Flux Klein Image Edit
  kind: image-edit — INSTRUCTION model (not CLIP)
  encoding: dual — Qwen2.5-VL semantic + VAE appearance
  formula: an instruction, not a description
  patterns:
    - "Keep [X], change [Y] to [Z]"
    - "Replace the [material/object] with [reference], preserve [geometry/lighting]"
    - "Enhance [attribute], leave [other elements] unchanged"
    - "multi-image: Apply the leather texture from Figure 2 to the chair in Figure 1, keep the frame unchanged, match lighting"
  do: [explicit + short, one edit goal per instruction, specify what must stay unchanged, say "photograph" for realism]
  avoid: [stacked conflicting edits, tag-soup / quality keywords, "photorealistic"/"3D render", keyword-packed negatives]

- model: Nano Banana 2 / Nano Banana Pro (Gemini Image)
  kind: image (gen + edit) — up to 14 refs (10 objects + 4 characters)
  variants: { NB2: "editing, style transfer, iteration", NB_Pro: "layouts, infographics, text rendering, brand consistency" }
  style: natural language ONLY — no tag soup, no quality keywords; refer to inputs as @img1, @img2, ...
  formula: Subject + Action + Location/Context + Composition + Lighting/Atmosphere + Style + [optional text/constraint]
  text_render: enclose desired text in quotes; specify font ("bold white sans-serif" / "Century Gothic"); Pro is best for text-heavy
  editing: conversational follow-ups ("keep everything, change lighting to golden hour and jacket to leather") — auto semantic masking
  character_consistency: upload reference images + assign names in the prompt
  avoid: ["4k / trending on artstation / masterpiece spam", "re-describing a reference — name it and state the change"]
```
