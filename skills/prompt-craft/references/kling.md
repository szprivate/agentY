# Kling 2.x / 3.0 — prompt fragment

Write like a film director giving scene instructions, not like an image prompt.

```yaml
model: Kling 2.x / 3.0
kind: video (i2v, t2v, multishot on 3.0)
formula: Subject (specific) + Action (precise movement) + Context (3-5 elements) + Style (camera, lighting, mood)
limits: { positive: "<=2500 chars", negative: "<=2500 chars" }   # API-enforced HARD reject, not silent truncation
negative: "smiling, cartoonish, smooth plastic skin, floating limbs, sliding feet, text morphing"
do:
  - always specify camera behavior explicitly (else output looks static / random)
  - anchor hands/limbs to objects ("fingers grip the cup edge") to stop floating
  - i2v: describe motion only, never redescribe the image
  - add motion endpoints to prevent hangs ("spins, then settles back into place")
  - walking: describe weight transfer ("each step lands heel-first, rolls forward with visible weight transfer") to stop the AI moonwalk
kling_3_adds: [up to 6 shots in one prompt, native audio/dialogue (name speakers), reference images addressable as @image1 @image2]
```

## Multishot (Kling 3.0 — up to 6 shots per generation)

```yaml
formula_per_shot: "[CHARACTER LOCK] + [ENVIRONMENT] + [TRANSITION CUE] + [SUBJECT MOTION] + [CAMERA MOVE] + [END STATE] + [STYLE]"
budget: whole multi-shot prompt <=2500 chars → ~400 chars/shot at 6 shots; keep each tight
character_lock: paste the EXACT same character description at the start of every shot — never paraphrase (Kling anchors identity to this string)
transition_cues: ["Continuous from previous shot:", "Immediately following:", "Moments later:", "Reverse angle:"]
motion: subject-motion and camera-motion as separate sentences; ONE gesture OR one camera move per 5s shot
camera: [slow push-in, pull-back reveal, static locked, orbit/arc, crane up, handheld drift, rack focus]
end_frame_handoff: describe the subject's final position at the end of each shot; open the next shot referencing that state → continuity across separate generations
negative_all_shots: "blurry, deformed hands, morphing face, identity change, flickering, jerky motion, warped background, two people"
failure_modes:
  face changes between shots: verbatim character lock + last-frame input image
  camera drifts on locked shot: add "camera does not move under any circumstances"
  lighting inconsistency: copy-paste the lighting string, don't paraphrase
  motion doesn't complete: one action per shot; extend to 8-10s
  shots feel disconnected: missing end-frame handoff
  accessories disappear: name them in the character lock every shot
```
