---
name: prompt-craft
description: ComfyUI prompt engineering — CLIP syntax, weight modifiers, model-specific strategies. Activate whenever the Researcher composes the generation prompt (step 7).
allowed-tools:
---

# ComfyUI Prompt Engineering
# Based on artokun/comfyui-mcp
# Copyright (c) 2024 Arthur R Longbottom
# MIT License - https://github.com/artokun/comfyui-mcp/LICENSE


## CLIP Basics
- 77-token limit per chunk; words = 1–3 tokens each. Tokens past the limit are **silently dropped**.
- Use `BREAK` to force a new 77-token chunk for long prompts.

## Weight Syntax
| Syntax | Weight |
|--------|--------|
| `(word:N)` | Explicit (0.0–2.0; >1.5 causes artifacts) |
| `(word)` / `((word))` / `(((word)))` | 1.1 / 1.21 / 1.331 |
| `[word]` / `[[word]]` | 0.909 / 0.826 |

Phrases work: `(red sports car:1.3)`. Nesting is multiplicative.

## Embeddings
Syntax: `embedding:name` (file must exist in `models/embeddings/`).  
Use in **negatives** for SD 1.5 / SDXL: `embedding:easynegative`, `embedding:badhandv4`, `embedding:negativeXL_D`.

## Model Quick-Reference

### SD 1.5
- **Negative: critical.** Use quality tags + embeddings.
- Positive: `(masterpiece:1.2), (best quality:1.2), subject, details, style`
- Negative: `worst quality, low quality, blurry, bad anatomy, bad hands, extra fingers, watermark, embedding:easynegative`
- Tag-based (danbooru style) works well. Keep under 77 tokens or use `BREAK`.

### SDXL
- **Negative: moderate importance.**
- Natural language preferred over tags. Dual CLIP supports ~154 tokens natively.
- Use `CLIPTextEncodeSDXL` for separate `text_g` (global) / `text_l` (local) control.
- Turbo/Lightning: CFG 1.0–2.0, minimal or empty negative.

### Flux
- **No negative prompt. CFG must = 1.0** (higher → artifacts).
- T5-XXL encoder: write natural descriptive sentences, not tag lists.
- Long prompts (200+ tokens) work fine.
- Schnell: 4 steps. Dev: 20–50 steps, sgm_uniform scheduler.
- **Don't use** quality tags (`masterpiece`, `best quality`) — describe quality in prose.

### SD3 / SD3.5
- Triple CLIP (CLIP-L + CLIP-G + T5-XXL). Long natural language prompts.
- CFG 4–7. Minimal negatives (`low quality, blurry` is enough).

## Prompt Order (SD 1.5 / SDXL)
1. Quality modifiers → 2. Subject → 3. Subject details → 4. Action/pose → 5. Environment → 6. Composition → 7. Lighting → 8. Style → 9. Technical quality

## LoRA Trigger Words
Place the LoRA's exact trigger word(s) naturally in the prompt. Check the model page for triggers. Multiple LoRAs: one trigger each; keep node strength at default unless tuning.

## BREAK Example
```
masterpiece, detailed Japanese garden, cherry blossoms, koi pond, morning mist
BREAK
8k uhd, photorealistic, volumetric lighting, depth of field, golden hour
```

---

## Video & API Model Guides

### WAN 2.1 / 2.2 (T2V and I2V)
**Prompt formula:** `Subject (desc) + Scene (desc) + Motion (desc) + Camera language + Atmosphere + Styling`  
**Target length:** 80–120 words. Under-specify → model fills in random defaults; over-specify → details ignored.

**T2V:** Write full natural language. Describe who/what, setting, action, camera, and lighting explicitly.  
**I2V:** The image defines *what*. Focus the prompt entirely on *how things move* — camera behavior, subject motion, environmental effects, speed. Do **not** redescribe the image content.

**Camera verbs that work reliably (2.2 > 2.1):** dolly-in, dolly-out, pull back, pan left/right, tilt up/down, tracking shot, orbital shot, bird's-eye, low-angle. Use **one** camera move per generation. Whip pan is unreliable on both versions.

**Negative prompt (supported):** `worst quality, low quality, blurry, static, morphing, warping, flickering, deformed face, extra fingers, watermark, subtitle`

**WAN 2.2 vs 2.1:** 2.2 uses MoE architecture — sharper frames, better multi-object scenes, more reliable camera direction. The 5B hybrid model runs at 720p on a single 4090. Prompt approach is the same.

---

### Kling (2.x / 3.0)
**Prompt formula:** `Subject (specific details) + Action (precise movement) + Context (3–5 elements) + Style (camera, lighting, mood)`  
Write like a film director giving scene instructions, not like an image prompt.

**Key principles:**
- Always specify camera behavior explicitly — without it, the model guesses and output looks static or random.
- Anchor hands/limbs to objects to prevent floating (`"fingers grip the edge of the cup"` not `"she moves her hands"`).
- For I2V: describe motion only; never redescribe the image.
- Use motion endpoints to prevent hangs: `"spins, then settles back into place"`.
- Negative prompts are recommended: `smiling, cartoonish, smooth plastic skin, floating limbs, sliding feet, text morphing`

**Kling 3.0 additions:** Multi-shot up to 6 shots in one prompt — label each explicitly (`Shot 1: …, Shot 2: …`). Native audio/dialogue: name speakers explicitly. Character consistency via "Elements" feature (upload reference from multiple angles). Reference images addressable as `@image1`, `@image2`.

**Physics tip (walking):** Describe weight transfer explicitly — `"each step lands heel-first, rolls forward with visible weight transfer"` — to prevent the AI moonwalk.

---

### Qwen Image Edit (2511 / fp8)
This model takes an **instruction** rather than a descriptive prompt. It uses dual encoding (Qwen2.5-VL semantic + VAE appearance), so it understands both high-level meaning and low-level pixel appearance.

**Instruction patterns that work:**
- `"Keep [X], change [Y] to [Z]"`
- `"Replace the [material/object] with [reference], preserve [geometry/lighting]"`
- `"Enhance [attribute], leave [other elements] unchanged"`
- Multi-image: `"Apply the leather texture from Figure 2 to the chair in Figure 1, keep the frame unchanged, match lighting."`

**Do:**
- Be explicit and short. One clear edit goal per instruction.
- Specify what must stay unchanged — the model uses this to preserve identity and geometry.
- Use the Lightning LoRA for fast iteration; disable for maximum fidelity.

**Don't:**
- Stack multiple conflicting edits in one pass.
- Use tag-soup or quality keywords — this is an instruction model, not CLIP.
- Use `photorealistic`, `3D render` etc. — say `photograph` for realism.
- Pack the negative prompt with keywords; use natural language describing what you don't want.

---

### Nano Banana 2 / Nano Banana Pro (Gemini Image)
These are **API-only, cloud-side models** running via Google's Gemini API (ComfyUI Partner Nodes). No local weights.

| | Nano Banana 2 (gemini-3.1-flash-image) | Nano Banana Pro (gemini-3-pro-image) |
|---|---|---|
| Speed | Fast / Flash | Slower, reasoning-first |
| Best for | Editing, style transfer, iteration | Complex layouts, infographics, text rendering, brand consistency |
| Thinking | Optional (Minimal / Dynamic) | Deep Think default |
| Max refs | 14 (10 objects + 4 chars) | 14 |
| Resolution | Up to 1K (512px option) | 1K–4K native |

**Both models:** Natural language only — no tag soups, no quality keywords like `masterpiece`. They reason through prompts before generating.

**Prompt formula:** `Subject + Action + Location/Context + Composition + Lighting/Atmosphere + Style + [optional: text/constraint]`  
Example: `"A stoic robot barista with glowing blue optics preparing espresso in a rain-soaked Tokyo alley at night. Low-angle tracking shot. Neon reflections on wet pavement. Cinematic, desaturated teal-orange grade."`

**Text rendering (Pro especially):** Enclose desired text in quotes in the prompt. Specify font style: `"bold white sans-serif"` or `"Century Gothic"`. For complex text-heavy images, first describe the text concepts conversationally, then request the image.

**Editing (conversational):** If an output is 80% right, don't regenerate — issue a follow-up instruction: `"Keep everything, change the lighting to golden hour and make the jacket leather."` The model does semantic masking automatically.

**Character consistency:** Upload reference images and assign names in the prompt. Supports up to 14 references.

**Don't:** Use `4k, trending on artstation, masterpiece` spam. Don't re-describe a reference image; just name it and specify the change.

---

## Common Mistakes
1. Negative prompt or CFG > 1 with Flux → artifacts
2. Tag-style prompts for Flux / SD3 → use sentences
3. Missing `BREAK` on 60+ word prompts → silent truncation
4. Weight > 1.5 → artifacts / color bleed
5. Conflicting weighted terms → confuses model
6. Missing embedding file → node error
7. Wrong LoRA trigger word → concept doesn't activate
8. Quality tags (`masterpiece`) in Flux prompts → ignored
