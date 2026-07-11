# LTX-2 / LTX-2.3 — prompt fragment

Lightricks video diffusion **with synchronized audio**. Rewards long, detailed,
cinematographer-style prompts; vague prompts give weak or uncertain motion.

```yaml
model: LTX-2 / LTX-2.3
kind: video (t2v, i2v, first-last-frame, v2v / controlnet: depth·canny·pose) + synced audio
formula: write a cinematographer's shot description — long and detailed
length: long — "long videos need long prompts"; under-specify → weak/uncertain motion
include: [subject, action, lighting, color palette, textures, atmosphere, camera move, audio]
motion:
  - use EXPLICIT motion verbs: rotate, pan, dolly, track, zoom, tilt
  - concrete move ("slow dolly-in") beats vague language → more stable
  - describe how subjects LOOK AFTER the movement → helps the model complete the motion
  - chronological order (what happens first, then next)
audio: model generates synced audio — describe ambient sound / SFX. Dialogue: break into
       short phrases with acting directions between lines; use physical cues, not emotional labels
i2v: the image defines the opening frame — prompt only the motion / camera / audio, don't redescribe it
negative: supported (CLIPTextEncode x2 pos/neg) — keep it to brief quality negatives
```
