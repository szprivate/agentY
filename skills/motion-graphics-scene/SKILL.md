---
name: motion-graphics-scene
description: Make EDITABLE motion graphics - titles, lower thirds, logo reveals, animated text / shapes / pictures - by writing a scene file (*.mgscene.json, data, not React), checking it, looking at stills, rendering it to video, and opening it in the viewer's Motion panel so the user can change layers and keyframes themselves. Use this for any motion-graphics request instead of writing Remotion code; use the `remotion` skill only for what a scene cannot say.
allowed-tools: write_text_file, read_text_file, mg_scene_validate, mg_scene_preview_frame, mg_render, show_in_viewer, view_image, get_agent_output_dirs
---

# Motion graphics as a scene file

You do NOT write React. You write a **scene**: one JSON file that describes layers and their
keyframes. One generic renderer (Remotion) turns any scene into a video, and the bEpic viewer's
**Motion** panel edits the same file (layers, timeline, keyframes, curves, gizmos), so you and the
user change one thing. What the panel previews is exactly what renders.

## Workflow

1. `get_agent_output_dirs` -> save the scene under `<videos>/motion/<project>/<name>.mgscene.json`
   (`write_text_file`). The file name MUST end in `.mgscene.json`. Local pictures / fonts go beside it
   (assets use paths relative to the scene file).
2. `mg_scene_validate(path)` - instant. Fix every `errors` entry (the `path` field says where);
   read the `warnings`.
3. `mg_scene_preview_frame(path, frames=[0, 15, 30, 60, 89])` -> one contact sheet; `view_image` it.
   Check the motion: is the text where it should be, does it arrive, hold, leave? Iterate on the file.
4. Open it for the user: `show_in_viewer(paths=[scene_path])` - the Motion panel opens on it. Tell
   them they can drag layers, move keyframes, change easing and press Render.
5. `mg_render(scene_path)` (mp4 next to the scene; `format="mov"` keeps transparency when
   `comp.background` is null; `webm`, `gif`). The first render after install takes minutes (npm
   install, browser download); later ones take seconds. Tell the user the path.
6. When the user edited the scene in the panel (it saves the same file), `read_text_file` it again
   before changing it - never overwrite their changes from memory.

Needs Node.js 18+ on the machine; if `mg_render` says it is missing, say so and stop.

## The format (`"mgscene": 1`)

```json
{ "mgscene": 1,
  "comp":   {"width":1920,"height":1080,"fps":30,"duration":90,"background":"#0a0a0a"},
  "assets": {"logo":{"type":"image","src":"logo.png","width":512,"height":512},
             "clip":{"type":"video","src":"shot.mp4"},
             "brand":{"type":"font","src":"Brand-Bold.ttf","family":"Brand","weight":700}},
  "layers": [ ... ] }
```

`duration` is in frames (seconds x fps). `background: null` = transparent (use `mov`/`webm`).
**Layers are listed bottom to top** (the last one is drawn on top). Every layer has a unique `id`.

### Layer

```json
{"id":"title","name":"Title","type":"text","parent":"grp","in":0,"out":90,"hidden":false,"locked":false,
 "blend":"normal", "anchor":[0,0],"position":[960,540],"scale":[100,100],"rotation":0,"opacity":100,
 "text":"THE WAY OUT","fontSize":120,"weight":800,"tracking":6,"align":"center","fill":"#ffffff",
 "tracks":{"opacity":[{"f":0,"v":0,"ease":"linear"},{"f":20,"v":100}]}}
```

- **types**: `text` (`text` with `\n` for lines, `font`, `weight`, `align` left|center|right, `lineHeight`),
  `shape` (`shape`: `rect` | `ellipse` | `path` + `d` SVG path data; `size`, `radius`), `image` / `video`
  (`asset`, `size`, `fit` contain|cover|fill, video: `startFrom`, `audio`), `group` (no drawing, a
  parent to move / scale / rotate children together), `component` (a named code component with `params`,
  e.g. `nodeTag` - preview shows a placeholder).
- **transform** (all animatable): `anchor` (the layer-space point that sits on `position`, and that
  rotation / scale turn about), `position` (px, comp space; y down), `scale` (percent, `[x,y]`),
  `rotation` (degrees, 720 = two turns), `opacity` (0-100). Shapes / images / videos are centred on
  their own origin; text sits on its origin with `align` choosing the side.
- **appearance** (animatable): `fill`, `stroke` (colours `#rgb`/`#rrggbb`/`#rrggbbaa`/`rgba()`; `null` =
  none), `strokeWidth`, `size` `[w,h]`, `radius`, `fontSize`, `tracking` (letter spacing px),
  `reveal` 0-1 (a wipe; `revealDir` right|left|down|up), `trimStart` / `trimEnd` 0-1 (draw-on of a
  stroke - the way to animate a ring or a line "drawing itself").
- `in` / `out` = frames the layer exists (out exclusive). `blend`: normal multiply screen overlay add
  darken lighten color-dodge difference.
- **parent**: a child follows its parent's transform (not its opacity). Its position is in the parent's
  space. Use a `group` to slide a whole lower third in and out with two keys.

### Keyframes

`tracks` maps a property to keys `{ "f": frame, "v": value, "ease": ... }`; `f` is a COMP frame
(not relative to `in`). A property with no track holds its static value. `ease` shapes the segment
AFTER the key: `smooth` (default, eases both ends), `linear`, `hold` (steps), `easeIn`, `easeOut`,
`easeInOut`, `backOut` (overshoots - good for pops), `expoOut` (fast then settle - good for arrivals), or
`[x1,y1,x2,y2]` (css cubic-bezier). Colours animate too. Optional `ti` / `to` are the curve editor's
hand-set tangents (value units per frame) - leave them to the user.

### Motion design rules that save retries

- Arrive with `expoOut` / `easeOut`, leave with `easeIn`; hold the readable state >= 1 s.
- Stagger: offset similar layers by 3-6 frames. Animate 2-3 properties together (position + opacity +
  reveal), not one.
- Keep text inside a 5 % safe margin; big type, high contrast. Use `\n` for line breaks (no auto-wrap).
- Do not key what does not move. Do not put keys past `comp.duration`.
- 30 fps unless asked; 1920x1080 (9:16 = 1080x1920).

## Examples (in this skill's `examples/`)

`lower_third.mgscene.json` (group slide-out, wipe reveals), `title_card.mgscene.json` (tracking
resolve, trimmed ring, rule), `logo_reveal.mgscene.json` (pop with `backOut`, ring draw-on, wordmark
wipe). Read one before writing yours - they are valid and render.

## When a scene cannot say it

Particles, 3D, per-letter animators, complex masks, audio sync: say so, build the part that fits in a
scene, and for the rest use the `remotion` skill (free React) - the user then loses the editable panel for
that part.
