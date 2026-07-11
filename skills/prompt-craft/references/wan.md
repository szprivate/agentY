# WAN 2.1 / 2.2 — prompt fragment

Prompt approach is identical for both versions; camera direction is more reliable on 2.2.

```yaml
model: WAN 2.1 / 2.2
kind: video (t2v, i2v)
formula: Subject + Scene + Motion + Camera language + Atmosphere + Styling
length: 80-120 words          # under → model fills random defaults; over → details ignored
t2v: full natural language — who/what, setting, action, camera, lighting all explicit
i2v: the image defines WHAT; prompt only HOW things move (camera, subject motion, speed). Do NOT redescribe image content.
camera: [dolly-in, dolly-out, pull back, pan left/right, tilt up/down, tracking, orbital, bird's-eye, low-angle]
camera_rule: one camera move per generation; whip-pan is unreliable on both versions
negative: "worst quality, low quality, blurry, static, morphing, warping, flickering, deformed face, extra fingers, watermark, subtitle"
```
