# Z-Image / Z-Image-Turbo — prompt fragment

Alibaba Tongyi distilled 6B DiT. Turbo is few-step and runs **without** classifier-free
guidance, so it behaves very differently from SD/Flux.

```yaml
model: Z-Image / Z-Image-Turbo
kind: image (text-to-image, image-edit, controlnet, upscale)
sampler: { steps: 8-12, cfg/guidance: 0-1 }   # distilled; official ~9 steps (8 effective)
negative: NONE — no classifier-free guidance; negatives are ignored
                  # recipe zeroes them via ConditioningZeroOut — don't rely on a negative
strategy: "addition, not subtraction" — describe what TO draw; you cannot tell it what to avoid
formula: structured natural language — 3-5 key visual concepts (subject, setting, detail, light, style)
style: powerful text encoder — prose, not a tag salad
length: concise + structured; native resolution 1024x1024
multilingual: strong — English / Chinese, can mix; renders non-Latin scripts well
text_render: put target text in "quotes"; describe typography (font, weight) separately
edit: image_edit is instruction-style (TextEncodeZImageOmni) — 1 image + text; state the change to make
avoid: [negative prompts (no effect), tag-soup, more than ~5 competing concepts]
```
