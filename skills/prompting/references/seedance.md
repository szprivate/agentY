# Seedance 2.0 / 2.5 — prompt fragment

ByteDance cinematic video: multi-shot, synced audio, strong physics, and its standout
feature — reference-to-video. Shares the cinematic approach of LTX-2/Veo, so **read
`ltx-2.md` for shared motion/camera principles**; below is what is Seedance-specific.

Prompt craft here is condensed from ByteDance's own `sd25-pe` skill. **For a long or
multi-asset job — more than ~5 references, a storyboard grid, a blockout, video
editing, extension, or a hybrid — read `seedance-2.5-full.md`** (the vendored
original, with its task-routing gate, per-asset role procedure and final checklist).
Skip it for ordinary one-shot generations; this fragment covers those.

```yaml
model: Seedance 2.0 / 2.0 Fast / 2.0 Mini / 2.5 (ByteDance, ComfyUI ByteDance2* nodes)
kind: video (t2v, i2v, reference-to-video, first-last-frame, edit, extend) + synced audio
formula: Subject + action/event + scene → visual treatment → camera → sound
length: no hard limit; every line must earn its place (see "don't pad" below)
order: goal → asset roles → subjects/relationships → event script → consistency
```

## Pick the version deliberately

| | 2.0 / Fast / Mini | **2.5** |
|---|---|---|
| duration | 4–15 s | **4–30 s** |
| references | 9 img / 3 vid / 3 audio | **30 img / 10 vid / 10 audio** (50 assets total) |
| timestamps | **ignored** — responds to shot numbers only | **honoured**, integer seconds |
| multi-view subject images | not recommended | supported |

Anything needing >15 s, >9 images, or real timestamps **must** be 2.5. Mini/Fast are
the cheap tiers — use them for tests, not finals.

## The one rule that breaks most often: asset binding

References are bound **by upload order**, and the prompt must say so explicitly.
`@Image1` is the first image slot, `@Video1` the first video, `@Audio1` the first audio.

- Give every asset you use **one explicit role, and say what NOT to take from it**:
  `@Image1 is used for the carpenter's face, hair and blue apron; do not use its background.`
- List every available-but-unused asset under `【Unused Assets】` so nothing reactivates it.
- Never encode the mapping *inside* the picture (a name written on a character sheet).
  Name it in the text or the model confuses or duplicates characters.
- One single-character image must not define two on-screen characters — one image per
  character, one per group.
- When the asset is already accurate, just cite it. Don't re-describe what it shows.

## Structure

Simple, few references — four lines, drop any that isn't wanted:

```text
<subject> <does the main action> in <scene>.
The visuals present <style or mood>.
The camera uses <shot size, position, movement, cuts>.
The sound includes <dialogue, ambience, effects, music>.
```

Multi-asset or narrative — labelled blocks (the labels stay in the submitted prompt):

```text
【Generation Goal】 …          【Reference Asset Roles】 …    【Unused Assets】 …
【Subjects and Relationships】 …  【Event Script】 start / principal event / end state
【Maintain Consistency】 identities, counts, clothing, prop ownership, directions
```

## Time

- **2.5 only:** integer-second ranges, 1 s granularity, contiguous — `0-3s… 3-7s… 7-15s`.
  No gaps. Don't time high-frequency actions ("shakes head three times per second").
  Too little plot in a range → the model improvises; too much → cuts or dropped beats.
- **2.0:** use `Shot 1 / Shot 2 …` instead; timestamps are ignored.
- Relative and point-in-time control both work: *"at the 5-second mark, a quick left wipe"*.

## Sound and dialogue

Audio is generated with the video. **Put spoken lines in double quotes** — that is what
steers the dialogue. State the speaker, language and on/off-screen position per beat.
Negative control works for audio and subtitles only: `no subtitles`, `no BGM;
environmental and action sound only`, `no audio`.

## Camera

Write plain film language directly — shot size, dolly/pan/track/orbit/handheld, low
angle, one-take, dolly zoom, FPV, bullet time, speed ramp. For anything niche, give
term **plus** the observable result: *"rack focus: the foreground trees blur as the
background character comes clear."* For a transition, name the trigger **and** the method.

## Don't pad

- No unrequested quality/stability packs, no invented style, camera move or sound.
- Never write output parameters into the prompt — ratio, duration, resolution, fps and
  the audio toggle are node widgets. See the `seedance-reference` skill for those, and
  for which of them the task locks (edits, first/last frame, extension).
- Actions: describe generally ("several high-knee raises, then a somersault"); spell out
  only the one or two memorable beats. Expressions: describe, don't reach for idioms.

## Reference counts that actually hold up

1–8 subject images (1–5 for audio/video subject refs, 5–10 s clips); ≤15 storyboard
panels, line-art rather than rendered, no text baked in; ≤20 s of source video for an
edit, with 1–5 reference images. Past those it still runs, but expect retries.

## Storyboard vs keyframes — different things

A multi-panel storyboard grid is a **loose** plot reference: the model keeps autonomy
and won't follow it panel by panel. When the video must match the drawings, pass them
as **separate images in order** and open with *"Use Images 1 to N in order as
keyframes."* — that binds tightly.

---

_Prompt guidance condensed from ByteDance's `sd25-pe` skill (v0.1.1) — see
`seedance-2.5-full.md` for the vendored original and its provenance. Node-side
parameter rules verified against ComfyUI `/object_info`._
