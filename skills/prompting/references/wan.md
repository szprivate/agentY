# WAN — prompt fragment

Covers WAN 2.1 / 2.2 / 2.6 and WAN VACE. Prompt approach is the same across versions;
camera direction is more reliable on 2.2+ .

```yaml
model: WAN 2.1 / 2.2 / 2.6 (+ VACE)
kind: video (t2v, i2v; VACE also v2v / video-inpaint)
formula: Subject + Scene + Motion + Camera language + Atmosphere + Styling
length: 80-120 words          # under → model fills random defaults; over → details ignored
t2v: full natural language — who/what, setting, action, camera, lighting all explicit
i2v: the image defines WHAT; prompt only HOW things move (camera, subject motion, speed). Do NOT redescribe image content.
camera: [dolly-in, dolly-out, pull back, pan left/right, tilt up/down, tracking, orbital, bird's-eye, low-angle]
camera_rule: one camera move per generation; whip-pan is unreliable on all versions
vace: control/edit variant (v2v, video inpaint). The control input (pose/depth/mask/reference) defines
      structure — prompt the motion / the change you want, not what the control already fixes.
version_note: 2.6 is the newest (API); prompting is unchanged from 2.2
negative: "worst quality, low quality, blurry, static, morphing, warping, flickering, deformed face, extra fingers, watermark, subtitle"
```
