# Veo 3 — prompt fragment

Google video model with native synchronized audio. Shares the cinematic + audio approach
of LTX-2 — **read `ltx-2.md` for the shared motion/camera principles**; below are Veo's
specifics.

```yaml
model: Veo 3 / 3.1
kind: video (t2v, i2v) + native audio & dialogue
template: "[Camera move + lens]: [Subject] [Action & physics], in [Setting + atmosphere], lit by [Light]. Style: [texture/finish]. Audio: [dialogue/SFX/ambience]."
camera: [dolly, tracking, crane, aerial, slow pan, POV]
detail: rich character/appearance detail helps ("a woman in her twenties with wavy brown hair and light freckles" > "a brown-haired woman")
audio: describe SFX + ambience explicitly
dialogue: use "quotes"; keep each line short — ~8 seconds of speech max
scene: describe thoroughly ("a smoky jazz club at night", not "a jazz club")
```
