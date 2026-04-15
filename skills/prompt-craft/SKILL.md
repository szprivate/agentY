```skill
---
name: prompt-craft
description: Model-family-specific prompt writing rules. Activate this skill whenever the Researcher is composing the generation prompt in step 7.
allowed-tools: analyze_image
---

# Prompt Craft — Model-Family Rules

Follow these rules **exactly** for the active model family. The model family is determined by the template selected in step 2 or the model shortname from `get_workflow_template`.

---

## Flux (flux1-dev, flux1-schnell, flux1-fill, flux1-canny, flux1-depth, flux1-kontext, flux2-*)

- Write **natural sentences**, not comma-separated tag lists.
- Be **specific**: include lighting conditions, material textures, camera angle/lens, mood, colour palette.
- Length: match complexity — 2–4 sentences for simple subjects, up to 8 for complex scenes.
- **Negative prompt → `null`** (Flux ignores negative prompts).

**Example:**  
`"A close-up portrait of a weathered lighthouse keeper, warm golden-hour sidelighting, coarse linen jacket, shallow depth of field with a bokeh harbour background, desaturated teal tones."`

---

## Flux Kontext (flux1-kontext)

Use the Kontext master/change format:

```
"master image — [description of what to keep]. change: [description of the edit]"
```

- Keep description: describe the subject and key elements to preserve.
- Change description: describe the edit precisely — avoid vague verbs like "make it better".
- **Negative prompt → `null`**.

**Example:**  
`"master image — a woman in a red dress standing in a park. change: replace the background with a snowy mountain landscape, maintain the subject's position and pose"`

---

## WAN (wan21-*, wan22-*)

- Describe **motion** first: what moves, how it moves, speed/rhythm.
- Include **camera movement** if relevant: pan, zoom, tracking shot, static.
- Describe **start → end states** for key elements if they change.
- Include ambient details: lighting changes, environmental motion (wind, water).
- **Negative prompt → `null`**.

**Example:**  
`"A lone tree sways gently in a summer breeze, its leaves shimmering in dappled afternoon light. Slow pan left, revealing a distant mountain range. The sky transitions from clear blue to a faint orange glow."`

---

## SD 1.5 / SDXL (cyberrealistic, juggernaut, photon, sdxl-base, epicrealism-xl)

- Tag-style prompts are acceptable: quality tags up front, subject, then modifiers.
- Quality tokens: `masterpiece, best quality, 8k uhd` — place these first.
- Negative prompt: **active** — include common artefact suppressors:  
  `ugly, blurry, low quality, deformed, extra limbs, watermark, text`
- SDXL: slightly longer positive prompts work better than SD 1.5.

**Example (positive):**  
`"masterpiece, best quality, photorealistic portrait of a middle-aged woman, professional studio lighting, sharp focus, elegant blouse"`

**Example (negative):**  
`"ugly, blurry, low quality, deformed, extra limbs, watermark, nsfw"`

---

## Nano Banana / Gemini (api_nano_banana_*, api_google_*, GeminiNanoBanana, IdeogramV3, api_bytedance_*)

- These models use a **single combined text input** — no separate negative prompt.
- Write as **imperative instructions** describing the desired output directly.
- Be concise and direct. Do not use artistic prose — these are API models, not diffusion models.
- **Negative prompt → `null`**.
- If multiple input images: refer to them as `@img1`, `@img2`, etc. in the prompt.

**Example:**  
`"Generate a photorealistic image of a golden retriever sitting in a sunlit garden. The dog should be looking directly at the camera with a happy expression."`

**Example (image edit):**  
`"Edit @img1: change the background to a snowy forest. Keep the subject unchanged."`

---

## General rules (all models)

- **No filler**: avoid phrases like "high quality", "stunning", "amazing", "beautiful" unless they describe a specific style.
- **Match length to complexity**: a simple portrait doesn't need 10 sentences.
- **Mirror user's style reference**: if the user specifies an artistic style ("oil painting", "cyberpunk", "Studio Ghibli"), incorporate it naturally.
- **Flag assumptions**: if you assumed a style or mood not explicitly requested, add a WARNING to `blockers`.
```
