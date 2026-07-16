---
name: story-scene
description: Expand a synopsis into a sequence of consistent, visual scene descriptions that serve as the starting point for start-frame image and video generation. Activate when the user provides (or you just produced) a synopsis and wants it turned into scenes.
allowed-tools:
---

# Scene Description

Turn a synopsis into an ordered set of **scene descriptions** that will later seed
start-frame images and videos. This is a text-writing task: load this skill and
write the blueprint yourself (you the orchestrator, or a spawned writer subagent).
You write the textual blueprint only — media is generated in a **separate** step.

Use the synopsis from the user's message, or — if this is a follow-up right after a
synopsis was written — the synopsis produced earlier in this conversation.

## Output structure

### 1. Story bible (consistency anchors)
Define these **once**, before the scenes, and reuse them **verbatim** wherever the element reappears:

- **Characters** — for each recurring character: a short stable **tag** (e.g. `MARA`) followed by a fixed visual description: approximate age, build, hair, skin, wardrobe, and one or two distinguishing features.
- **Locations / surroundings** — for each recurring place: a fixed description of its defining visual features, materials, and palette.
- **Key objects / props** — for each recurring object: a fixed description (shape, material, colour, condition).

### 2. Scenes
Number scenes in story order. For each scene give a tight block:

- **Scene N — beat:** one line on what happens.
- **Setting:** which bible location (reuse its description), time of day, weather, atmosphere/lighting.
- **Characters present:** reference each by their **tag** and reuse their bible description — do not re-invent appearance.
- **Framing (for the start frame):** subject placement in frame, shot type (wide / medium / close-up), camera angle, focal feel. Describe what a single still photograph of this moment would show.
- **Motion note (for the video step):** a brief line on what moves and any camera move. Keep it short — the video-generation step will expand it.

## Consistency rules (critical — this is the whole point of this skill)
- Recurring **characters, surroundings, and objects must be described with the SAME wording every time they appear.** Copy the bible description; never paraphrase it. Downstream image/video models anchor identity to the exact string, so paraphrasing breaks visual consistency.
- Keep each character's **tag** identical across all scenes.
- If a new recurring element appears mid-story, **add it to the bible first**, then reference it.
- Prefer concrete, camera-visible detail over interior states ("white-knuckled grip on the railing", not "she feels afraid").

## Style
- Present tense, concrete, visual. Prioritise what a camera would see.
- One scene per block; keep each scene a short paragraph.
- Default to 3–6 scenes unless the user asks for more or fewer.

## Scope
Stay textual — do not call image/video tools here, and do not claim to render media.
Your output is the description; generation happens in a separate step, driven by you
(the orchestrator) via the normal generation contract or a spawned generation subagent.
