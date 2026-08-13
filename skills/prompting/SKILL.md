---
name: prompting
description: ComfyUI prompt engineering — universal CLIP/weight syntax plus per-model prompt rules loaded on demand from references/. Activate whenever composing a generation or edit prompt.
allowed-tools:
---

# ComfyUI Prompt Engineering
# Universal rules based on artokun/comfyui-mcp — Copyright (c) 2024 Arthur R Longbottom, MIT
# Per-model fragments (references/) are agentY-authored.

**How to use this skill:**
1. The **universal rules** below apply to every prompt.
2. Then read the **one** `references/*.md` fragment for the model family you are
   building for (take the family from the recipe / brainbriefing `model`). It
   carries that model's prompt formula, length target, negative policy, sampler
   constraints, and pitfalls.
3. Do **not** load fragments you don't need — that's the whole point of the split.

## Model index
| Model family | Fragment |
|---|---|
| Flux 1 & 2 (dev / schnell / Klein / Krea) | `references/flux.md` |
| Z-Image / Z-Image-Turbo | `references/z-image.md` |
| LTX-2 / LTX-2.3 — video + audio | `references/ltx-2.md` |
| WAN 2.1 / 2.2 / 2.6 / VACE — video | `references/wan.md` |
| Kling 2.x / 3.0 — video, multishot | `references/kling.md` |
| Veo 3 — video + audio | `references/veo.md` |
| Seedance 2.0 / 2.5 — video, reference-to-video, edit, extend | `references/seedance.md` (deep cases: `references/seedance-2.5-full.md`) |
| Ideogram — text-to-image, typography / text rendering | `references/ideogram.md` |
| SD 1.5 · SDXL · SD3/3.5 · Qwen Image Edit · Nano Banana · Seedream | `references/other-models.md` |

_New model family? Add a `references/<model>.md` fragment in the compact schema
(see any existing fragment) and add a row here._

---

## Universal rules

### CLIP token basics
- 77-token limit per chunk; words = 1–3 tokens each. Tokens past the limit are **silently dropped**.
- `BREAK` forces a new 77-token chunk for long prompts.
- Flux / SD3 / Gemini / instruction-edit models do **not** use CLIP weighting — see their fragment.

### Weight syntax (CLIP models — SD 1.5 / SDXL)
| Syntax | Weight |
|--------|--------|
| `(word:N)` | explicit 0.0–2.0 ( >1.5 → artifacts ) |
| `(word)` / `((word))` / `(((word)))` | 1.1 / 1.21 / 1.331 |
| `[word]` / `[[word]]` | 0.909 / 0.826 |

Phrases work: `(red sports car:1.3)`. Nesting is multiplicative.

### Embeddings
`embedding:name` — file must exist in `models/embeddings/`. Use in **negatives** for SD 1.5 / SDXL (e.g. `embedding:easynegative`, `embedding:badhandv4`).

### Prompt order (SD 1.5 / SDXL)
Quality → Subject → Subject details → Action/pose → Environment → Composition → Lighting → Style → Technical quality.

### LoRA triggers
Place the LoRA's exact trigger word(s) naturally in the prompt (check the model page). Multiple LoRAs: one trigger each; keep node strength at default unless tuning.

### BREAK example
```
masterpiece, detailed Japanese garden, cherry blossoms, koi pond, morning mist
BREAK
8k uhd, photorealistic, volumetric lighting, depth of field, golden hour
```

## Universal mistakes
1. Weight > 1.5 → artifacts / color bleed.
2. Conflicting weighted terms → confuses the model.
3. Missing `BREAK` on 60+ word CLIP prompts → silent truncation.
4. Wrong LoRA trigger word → concept doesn't activate.
5. Applying CLIP tricks (weights, tag-soup) to a model whose fragment says prose / instruction only.
