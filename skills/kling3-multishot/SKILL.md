---
name: kling-multishot
description: Kling 3.0 multi-shot storyboard (template Kling3_multiShot). Activate in the Researcher when the selected template is Kling3_multiShot — overrides the standard prompt-craft step. Activate in the Brain during assembly of the Kling3_multiShot template to patch storyboard nodes correctly.
allowed-tools: update_workflow, get_workflow_template
---

# Kling 3.0 — Multi-Shot Storyboard Skill

## When to activate
- **Researcher** (step 2 + step 7): selected template is `Kling3_multiShot`, OR user asks for a multi-shot video, storyboard, scene sequence, or narrative clip with more than one camera cut
- **Brain** (step 1.1 assembly): loaded template is `Kling3_multiShot`
- Do **not** activate for single-shot I2V or T2V requests

---

## Researcher — Template selection and prompt composition

### Template selection (step 2)
Select the `Kling3_multiShot` template when any of these are true:
- User asks for a multi-shot video, storyboard, scene sequence, or narrative clip
- User describes a scene with implied cuts ("then show…", "close-up of…", "cut to…")
- User specifies more than one camera angle or shot type in a single request

Set `task.type` to `video i2v`.

### Prompt composition (replaces prompt-craft step 7)
Count the number of shots the user asked for (max 6, default 2). Generate that many **DISTINCT** shot prompts following the formula below. Do not repeat the same prompt across shots.

Store all shot prompts in the brainbriefing `prompt.positive` field as a **JSON array string**:
```
["<shot_1_prompt>", "<shot_2_prompt>", ...]
```

Set `prompt.negative` to:
```
blurry, deformed hands, morphing face, identity change, flickering, jerky motion, warped background, two people, extra limbs, watermark
```

Add a note in `blockers` (WARNING level) with:
- shot count (e.g. `"WARNING: 3-shot storyboard — Brain must set multi_shot to '3 storyboard' on node 12"`)
- per-shot duration if specified by user (e.g. `"WARNING: user requested 5s per shot"`)

---

## Brain — Template patching (step 1.1)

### Template node map (Kling3_multiShot)

| Node ID | Class | Role |
|---------|-------|------|
| `12` | `KlingVideoNode` | Main generator — holds all inputs: `multi_shot`, per-shot prompts, durations, model params |
| `14` | `LoadImage` | Start frame — patched by `assemble-from-template` via `input_nodes` |
| `22` | `VHS_VideoCombine` | Output — patched via `output_nodes` |

### Assembly steps

**1. Parse shot prompts**  
Parse the JSON array from `brainbriefing.prompt.positive`. Count the entries — this is the shot count (N, max 6).

**2. Set `multi_shot` on node 12**

| Shot count (N) | `multi_shot` patch value |
|---|---|
| 1 | `1 storyboard` |
| 2 | `2 storyboards` |
| 3 | `3 storyboards` |
| 4 | `4 storyboards` |
| 5 | `5 storyboards` |
| 6 | `6 storyboards` |

**3. Set per-shot prompts directly on node 12**  
Patch each shot prompt as a string value directly into the corresponding input on node 12:  
`multi_shot.storyboard_1_prompt`, `multi_shot.storyboard_2_prompt`, … up to N.  
Do **not** patch nodes 25–30 (`Text Multiline`) — those nodes are not used for prompt delivery.

**4. Set per-shot durations on node 12**  
Use the duration from user request; default `1` per shot if unspecified.  
Input names: `multi_shot.storyboard_1_duration`, `multi_shot.storyboard_2_duration`, … up to N.

**5. Call `update_workflow` in a single call**  
Build `patches` covering all of node 12: `multi_shot`, per-shot prompts, per-shot durations, `model.aspect_ratio`, `model.resolution`, `generate_audio`, `model`.

**Example `update_workflow` patches array (3-shot):**
```json
[
  { "node_id": "12", "input_name": "multi_shot", "value": "3 storyboards" },
  { "node_id": "12", "input_name": "multi_shot.storyboard_1_prompt", "value": "<shot_1_prompt>" },
  { "node_id": "12", "input_name": "multi_shot.storyboard_2_prompt", "value": "<shot_2_prompt>" },
  { "node_id": "12", "input_name": "multi_shot.storyboard_3_prompt", "value": "<shot_3_prompt>" },
  { "node_id": "12", "input_name": "multi_shot.storyboard_1_duration", "value": 1 },
  { "node_id": "12", "input_name": "multi_shot.storyboard_2_duration", "value": 1 },
  { "node_id": "12", "input_name": "multi_shot.storyboard_3_duration", "value": 1 },
  { "node_id": "12", "input_name": "model.aspect_ratio", "value": "16:9" },
  { "node_id": "12", "input_name": "model.resolution", "value": "720p" },
  { "node_id": "12", "input_name": "generate_audio", "value": false },
  { "node_id": "12", "input_name": "model", "value": "kling-v3" }
]
```

### Default parameters

| Parameter | Default | Override condition |
|---|---|---|
| `model.aspect_ratio` | `16:9` | **Always use `16:9` regardless of input image dimensions. Override ONLY if the user explicitly requests a different ratio.** |
| `model.resolution` | `720p` | **Always use `720p` regardless of input image resolution. Override ONLY if the user explicitly requests a different resolution.** |
| `generate_audio` | `false` | Set `true` + use `kling-v3-omni` only if user explicitly asks for audio |
| `model` | `kling-v3` | Use `kling-v3-omni` if audio required |

---

## Shot prompt formula

```
[CHARACTER LOCK] + [TRANSITION CUE] + [SUBJECT MOTION] + [CAMERA MOVE] + [END STATE] + [STYLE]
```

- One subject motion AND one camera move per shot — not both if duration is ≤ 5s
- Write subject motion and camera motion as **separate sentences**
- Be specific: `"she slowly raises her right hand"` not `"she moves"`

### Character lock
Paste the **exact same** character description string at the start of every shot prompt. Never paraphrase.  
Example: `a tall woman with short copper hair, steel-blue lab coat, early 40s, slim silver-framed glasses`

If the user provides a reference image, it will be loaded via `input_nodes` (node 14). Reference it in prompts as `@character` if the workflow supports Elements.

### Transition cues (shot 2+)
Open each shot after shot 1 with one of:
- `Continuous from previous shot:`
- `Immediately following:`
- `Moments later:`
- `Reverse angle:`

### Camera vocabulary
`slow push-in` · `pull-back reveal` · `static locked` · `orbit/arc shot` · `crane up` · `handheld drift` · `rack focus` · `dolly-in` · `low-angle` · `over-the-shoulder` · `bird's-eye`  
Use **one** camera move per shot. Whip pan is unreliable — avoid.

### End-frame handoff
End every shot prompt by describing the subject's final position/state.  
Open the next shot referencing that state — this creates continuity across cuts.

### Example — 4 shots

```
Shot 1: A tall woman with short copper hair, steel-blue lab coat, early 40s, slim silver-framed glasses stands at a lab bench. She remains still, looking down at a petri dish. Camera executes a slow pull-back reveal from tight on her hands to a full environment wide shot. Shot ends with her centered in frame, both hands on the bench, gaze down. Clinical, desaturated, 35mm grain.

Shot 2: A tall woman with short copper hair, steel-blue lab coat, early 40s, slim silver-framed glasses. Continuous from previous shot: she lifts her gaze. She slowly picks up a pipette with her right hand and brings it toward the dish in a deliberate arc. Camera slow push-in from waist to extreme close-up on hands and pipette tip. Shot ends with pipette tip held 2cm above the dish. Clinical, desaturated, 35mm grain.

Shot 3: A tall woman with short copper hair, steel-blue lab coat, early 40s, slim silver-framed glasses. Immediately following: she has released a drop. She slowly raises her head and looks directly into camera. Camera static locked, medium close-up on face. Shot ends with direct eye contact, neutral expression. Clinical, desaturated, 35mm grain.

Shot 4: A tall woman with short copper hair, steel-blue lab coat, early 40s, slim silver-framed glasses. Continuation: she turns away toward a large window. She walks slowly toward it. Camera cranes up and arcs right, ending on her silhouette against the bright window, back to camera. Shot ends with silhouette still and centered. Clinical, desaturated, slight lens flare, 35mm grain.
```

---

## Troubleshooting

| Problem | Fix |
|---|---|
| Face changes between shots | Verbatim character lock every shot |
| Camera drifts on locked shot | Add `camera does not move under any circumstances` |
| Lighting inconsistency | Copy-paste lighting string verbatim — never paraphrase |
| Motion doesn't complete | One action per shot; increase duration or simplify |
| Shots feel disconnected | Missing end-frame handoff — add final position description |
| Accessories disappear | Name them in character lock every shot |
| Shot prompts ignored | Check `multi_shot` value matches shot count exactly, and that prompts are on `multi_shot.storyboard_N_prompt` inputs of node 12 |

---

## Checklist

**Researcher:**
- [ ] Template set to `Kling3_multiShot`
- [ ] `task.type` set to `video i2v`
- [ ] Shot count confirmed (max 6)
- [ ] `prompt.positive` is a JSON array with N distinct shot prompts
- [ ] `prompt.negative` populated
- [ ] Shot count + duration noted in `blockers` as WARNING

**Brain:**
- [ ] Shot prompts parsed from `prompt.positive` JSON array
- [ ] `multi_shot` enum matches shot count exactly
- [ ] Shot prompts patched directly into `multi_shot.storyboard_N_prompt` inputs on node 12
- [ ] Durations set per shot on node 12 (default `1`)
- [ ] `generate_audio` set to `true` + model `kling-v3-omni` only if user requested audio
- [ ] Character lock phrase identical across all shots
- [ ] Each shot (except first) opens with transition cue
- [ ] Each shot (except last) ends with explicit end-frame state
