---
name: model-from-reference
description: Model a building, object or set in a 3D app (Blender, Maya or Houdini) so that it matches a reference photo — the way a 3D artist does it - measure the photo, match the camera, block out grey shapes until the silhouette and proportions match, THEN details, THEN materials and light. Use for "model this house / object / room in Blender from this photo", "rebuild this in 3D", "make a 3D version of this picture", "match this reference in Blender".
allowed-tools: dcc_build dcc_silhouette_check measure_reference blender__execute_blender_code blender__render_viewport_to_path blender__get_objects_summary blender__get_screenshot_of_window_as_image analyze_image get_image_resolution run_script write_text_file read_text_file show_in_panel
---

# Model from a reference photo

Two tools do the work (in the `dcc` tool pack: `load_tools("dcc")` if they are not in your list):
`dcc_build` runs your one build script in the app, `dcc_silhouette_check` checks the proportions by
numbers. You may NOT replace either with live edits or with judging by eye.

| app | live (the app the user has open) | batch (a fresh headless process, saves a scene file) |
|---|---|---|
| Blender (Z up) | yes, through the Blender MCP add-on (port 9876) | `blender -b`, saves a .blend |
| Maya (Y up) | yes, if its commandPort is open (port 7001) | `mayapy`, saves a .ma |
| Houdini (Y up) | no: batch only | `hython`, saves a .hip |

`mode="live"` changes the scene the user is looking at; `mode="batch"` never touches it. Prefer live in Blender
when the user has it open, batch otherwise. Coordinates are always the app's own (Blender Z up, Maya and
Houdini Y up): the build script, the camera and the check points all use the same ones.

The order is the whole method. Never build geometry and materials in one go: when the
proportions are wrong, every material, every detail and every light has to be redone
with them. Work in stages, and do not leave a stage until its check passes.

| stage | what you do | leave it when |
|---|---|---|
| 1 Measure | read positions off the photo, turn them into metres | a dimension table exists |
| 2 Camera | match the photo's camera before any geometry | horizon and frame match |
| 3 Blockout | plain grey boxes and slabs, no detail, no colour | `dcc_silhouette_check` passes (the gate) |
| 4 Detail | windows, railings, bracing, trim, props | the check still passes |
| 5 Look | materials, sun, sky, final render | colours and light match |

## One build script, rebuilt every pass

Keep the whole model in ONE Python file, `output/scripts/<name>_build.py` (write it with
`write_text_file`). At its top, every dimension is a named constant with a comment saying
where it came from (`DECK_Z = 3.3  # deck top: 0.22 of image height above the sand line`).
The script first deletes its own collection, then builds everything from the constants.

Put a `# --- DETAIL ---` line in the script: everything above it is the block-out (constants, camera, masses),
everything below is detail and look. EVERY pass goes through `dcc_build` with that one file. Never run the geometry with
`execute_blender_code` or any other live call:

```
dcc_build(script_path="<abs path>/output/scripts/<name>_build.py", app="blender", mode="live",
          stage="blockout", render_to="<abs path>/output/<name>_blockout.png", render_engine="fast", resolution="800x600")
```

Set `stage` on every build: `blockout` (always allowed), `detail`, `look`. **detail and look are gated**: the build is
refused, nothing runs, unless the last `dcc_silhouette_check(script_path=<this script>)` passed AND the script above
the `# --- DETAIL ---` line is unchanged since (edits below it never need a re-check; with no marker any edit does).
The refusal lists the failing points and hints: fix the block-out constants, rebuild with `stage="blockout"`, check
again. `force_reason="<sentence>"` overrides the gate and is shown to the user in the result (`gate.forced`). Use it
only when a point is truly unmeasurable (hidden in the photo), and say which point and why. Never to get past a
failing check, and never because the remaining misses "look small".

It runs the whole file, returns the script's own error line if it fails (fix the constant or the line, run
again), the objects it made, and a fast render (Blender Workbench, Maya hardware, Houdini OpenGL). In batch
mode add `save_to` (or take the default) and keep the `saved` path it returns: the check and later renders open it.
`render_engine="final"` is Cycles / Arnold / Karma, for stage 5 only.

A correction is a changed constant and a re-run: the scene is rebuilt clean every time.
The Blender MCP tools (`execute_blender_code`, `get_objects_summary`, screenshots) are for INSPECTING only:
reading a bounding box, listing what is there. Never patch the scene object by object over many calls;
leftovers pile up (stray blocks, doubled parts) and you lose track of what is really there. If the scene is
wrong, change the script and call `dcc_build` again.

## 1 Measure the photo

1. `get_image_resolution` on the reference; note its aspect ratio.
2. Make a measuring copy with `run_script`: draw a labelled grid every 5 % of width and
   height over the photo (PIL), save it as `<name>_grid.jpg`.
3. Look at the grid copy yourself (`analyze_image(mode='full')`) and read positions as
   fractions of the image: ground contact line, deck/floor line, eaves, ridge, top of mast,
   left and right extents of each part, the horizon line.
4. Pick ONE scale anchor of known size, two points on it: a railing height (~1.0 m), a door
   (~2.0 m), a step (~0.18 m), a person (~1.75 m), a storey (~2.7 m).
5. Measure it in 3D with `measure_reference(image_path, points, pairs, anchor={"pair": [nameA, nameB], "meters": 1.0})`
   (MoGe, runs on the local ComfyUI, cached per image). Give it the points you read off the
   grid as `{name, px:[x_frac, y_from_top_frac]}`, the pairs you need as `[nameA, nameB]`
   (ground contact -> deck line, left end -> right end, deck -> ridge) and the anchor pair. It
   returns the field of view, the horizon, and, SCALED by the anchor, the camera height, every
   point in metres and each pair's distance and height difference. MoGe's raw metres run 10-40 %
   off at 20-30 m (1.3x too large for a house 29 m away): never use them without the anchor, and
   treat a `scale` outside 0.5-2.0 (warning) as a wrongly placed anchor.
6. For each facade edge write down its `yaw_deg` and `nearer` end from the pairs. Positive yaw:
   the right end is farther away (camera at the front-left); 0 is straight-on. A building shot
   from a corner has edges at +-15-30 deg: the camera must reproduce that, a straight-on camera
   is wrong.
7. Write the table into the script as constants, with where each came from, and the yaw of the
   receding edges.

## 2 Match the camera first

- Resolution: the photo's aspect (portrait stays portrait).
- Keep the camera level (no tilt, no roll) and move the frame with lens shift instead:
  a level camera puts the horizon at the frame centre; for a portrait frame
  `shift_y = horizon_from_top - 0.5`, for landscape multiply that by height/width.
  Positive `shift_y` moves the frame UP, so the subject moves DOWN in the picture.
- Camera height: whatever sits ON the horizon line in the photo is at eye height. If the
  horizon cuts the stilts 1.5 m above the sand, the camera is at 1.5 m.
- Lens: use the field of view `measure_reference` returned (vertical fov for a portrait frame). Distance: `d = real_width / (2 * tan(hfov / 2) * width_fraction)`.
- Aim with `(target - cam.location).to_track_quat('-Z', 'Y')`. Hand-written Euler angles
  are how a camera ends up rolled, or pointing backwards at an empty frame.
- The camera lives in the build script too (the check uses the scene camera, Houdini: the first camera in
  /obj, or the one you name). In Maya the lens shift is the camera's film offset, in Houdini its window
  offset; a Maya camera must be set renderable (or named in `camera=`).
- Which side of the subject is the camera on? A wall you can see means the camera is on
  that wall's side. The edge yaw from stage 1 decides it: place the camera (or rotate the model) until the
  receding edges have the photo's yaw. The check in stage 3 tests this.

## 3 Blockout: grey only

Simple boxes and slabs for every mass: legs, deck, each wing, each roof plane, mast.
ONE flat grey material, no detail. Render fast (Eevee or Workbench, ~800 px).

Gable roofs: build each roof plane from its four vertices (eave left, eave right, ridge
right, ridge left) so the ridge is the highest edge. A rotated slab overshoots its ridge
and turns the roof into an inverted V.

**The silhouette check, by numbers, not by eye. This stage is not passed until `dcc_silhouette_check`
passes.** Pick 6-12 key points of the model (ground contact, deck corner, eave, ridge ends, mast top, left
and right extents of each part) and write where each sits in the photo, as fractions read off your grid
(x from the left, y from the TOP):

```
dcc_silhouette_check(app="blender", mode="live", tolerance=0.03,
  script_path="<abs path>/output/scripts/<name>_build.py",
  points=[
    {"name": "deck front-left corner",  "world": [-3.0, 0.0, 3.3], "target": [0.21, 0.62]},
    {"name": "deck front-right corner", "world": [4.0, 2.0, 3.3],  "target": [0.77, 0.60]},
    {"name": "ridge left end",          "world": [-2.5, 1.0, 6.1], "target": [0.30, 0.35]},
    {"name": "mast top",                "world": [0.0, 0.0, 9.0],  "target": [0.52, 0.08]}],
  yaw_pairs=[{"pair": ["deck front-left corner", "deck front-right corner"], "target_yaw_deg": 21, "tolerance_deg": 8}])
```

ALWAYS pass `script_path`: it records the result (`<script>.check.json`) that the detail / look gate reads.
Include at least one receding edge as a `yaw_pair` with the yaw measured from the photo; it counts toward
`overall_pass` and fails with a hint ("facade yaw 2 deg vs photo 21 deg: move the camera left / rotate the model")
when the camera is on the wrong side or straight-on.

`world` is read from the build script's constants (the app's own coordinates). In batch mode also pass
`scene=` the `saved` path from `dcc_build`. The answer lists every point with dx, dy (projected minus photo),
pass or fail, and a hint, the worst first. Read `overall_hint` first: when every point is off the same way,
the camera is wrong (height, distance, aim, lens shift), not the model, so fix the camera constants before
touching any mass. Then fix the worst point's constant in the script, run `dcc_build`, run the check again.
Do not move on to stage 4 while `overall_pass` is false (the gate refuses it anyway), and do not argue a
failure away by eye: 6 of 12 passing is a fail. The numbers decide, the picture comparison below finds what
they missed. If a point cannot be checked (hidden in the
photo), leave it out instead of guessing its target.

Then compare pictures: `analyze_image(file_paths=[reference, render], mode='full')` when
your model can see images, otherwise mode `describe`. Ask open questions ("List the
differences in shape, proportions and camera, most important first"), never a checklist
to confirm. If a comparison says something is missing or extra, check it in the scene
(bounding boxes in camera view), not by object names.

## 4 Detail

Windows and doors where the photo has them (count them), railings, bracing, stairs,
trim, props. All of it goes in the same build script BELOW the `# --- DETAIL ---` line, built with
`dcc_build(stage="detail")`. Re-run `dcc_silhouette_check` (same points, same `script_path`); details must not
move the masses.

## 5 Look

- Materials last. Take colours from the photo and convert sRGB to linear before putting
  them in a Principled BSDF (`c / 12.92` below 0.04045, else `((c + 0.055) / 1.055) ** 2.4`).
- Use the "Standard" view transform for colours that match the photo. AgX and Filmic wash
  out saturated colours such as a blue sky or orange wood.
- Sun direction from the photo's shadows. The side facing the camera is usually lit.
- Final render: `dcc_build(..., stage="look", render_engine="final", render_to="<name>_final.png")` (Cycles with
  denoising in Blender, Arnold in Maya, Karma in Houdini). Camera, lens and aspect are those the check passed.

## Report

Show the final render next to the photo. List what still differs honestly; do not rate
your own match ("6/10"), and do not claim a measurement you did not check: quote the last
`dcc_silhouette_check` result (points, worst distance) as the measurement, and report any `gate.forced` build
with its reason.
