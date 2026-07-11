# Seedance 2.0 — prompt fragment

ByteDance cinematic video with multi-shot, synced audio, and strong physics. Shares the
cinematic approach of LTX-2/Veo — **read `ltx-2.md` for shared motion/camera principles**;
below are Seedance's specifics. Its standout feature is reference-to-video.

```yaml
model: Seedance 2.0 (ByteDance)
kind: video (t2v, i2v, reference-to-video, v2v, first-last-frame) + synced audio, multi-shot
formula: Subject + Motion + Environment + Aesthetics + Camera + Audio
length: under ~200 words
order: scene description → camera move ("slow dolly in") → lighting/mood → subject details
camera: director-level — dolly zoom, rack focus, tracking, POV switch, handheld
reference:  # R2V — the distinctive feature
  - combine up to 9 images / 3 video clips / 3 audio files in one generation
  - cite each in the prompt with tokens: [Image1], [Video1], [Audio1], ...
  - 'e.g. "use the composition from [Image1]", "follow the action from [Video2]"'
  - model auto-extracts core features of the reference and fuses with your text
audio: synchronized — describe SFX / ambience; text overlays supported across modes
physics: strong — collisions, fabric, vehicle chases, fight motion render well
```
