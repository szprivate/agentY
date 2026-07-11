# Flux — prompt fragment

```yaml
model: Flux (dev / schnell / Klein)
kind: image (text-to-image)
encoder: T5-XXL — reads natural language, not tags
formula: one flowing natural-language description (scene → subject → detail → light → style)
length: long is fine (200+ tokens)
sampler: { cfg: 1.0 }          # cfg > 1 causes artifacts
negative: none                 # omit the negative node / text entirely
do:
  - full descriptive sentences
  - describe quality in prose ("sharp studio photograph, soft rim light")
avoid:
  - negative prompt (omit it)
  - cfg > 1
  - quality tags (masterpiece, best quality) — silently ignored
  - danbooru / tag-soup style
```
