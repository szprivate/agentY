# Flux (1 and 2) — prompt fragment

Both generations drop the negative prompt and want natural-language prose — but they
differ on guidance: **Flux 1 runs at cfg ~1, Flux 2 at cfg 3–5.** Pick the block for
the exact model.

```yaml
- family: Flux 1 (dev / schnell / Klein / Krea)
  kind: image (text-to-image)
  encoder: T5-XXL — natural language, not tags
  sampler: { cfg: 1.0 }              # cfg > 1 → artifacts
  negative: none                     # omit the negative node / text entirely
  formula: one flowing description (scene → subject → detail → light → style); long (200+ tokens) fine
  krea_note: Flux Krea is a photorealism finetune — same prompting; lean into photographic detail
  do: [full descriptive sentences, describe quality in prose ("sharp studio photograph, soft rim light")]
  avoid: [negative prompt, cfg > 1, quality tags (masterpiece/best quality — ignored), danbooru/tag-soup]

- family: Flux 2 (pro / dev / Klein)
  kind: image (text-to-image, image-edit)
  encoder: single Mistral VLM prompt — understands context; word ORDER matters (front-load the important stuff)
  sampler: { cfg: 3-5 }              # [dev]; never > ~4.5-5 without dynamic thresholding → oversaturated / black
  negative: none                     # distilled: guidance is a learned input, not CFG subtraction
  formula: Subject + Action + Style + Context — clear descriptive prose
  klein_note: Flux 2 Klein is the distilled sub-second variant — same Flux 2 prompting
  do: [natural prose, front-load key subject/action]
  avoid: [negative prompt, cfg > 5, tag-soup]
```
