# Ideogram — prompt fragment

Text-to-image model whose defining strength is **text rendering / typography** (~90–95%
accuracy, highest of the public models). Prompt it for that.

```yaml
model: Ideogram (3.x / 4.x)
kind: image (text-to-image), typography / design specialist
formula: complete sentences with punctuation (design-aware natural language)
text_render:
  - put the exact words in STRAIGHT quotes: "Grand Opening"
  - keep headline copy short — 1-4 words ~90% accurate; 5-12 words ~70%; >15 words drops/crams letters
  - name a REAL type style ("Bauhaus geometric sans", "1970s slab serif", "condensed display gothic") —
    interpreted far better than vague words like "modern" or "clean"
style_presets: [Design (logos/posters/graphic type), Realistic (photo/packaging/signage), 3D, Anime, General]
magic_prompt: turn OFF for precise wording — it rewrites your prompt and swaps specific terms for near-misses
negative: not needed — describe what you want
```
