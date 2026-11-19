---
name: matchmove
description: Track a plate's camera in the ComfyUI Image Viewer's matchmove - the lens distortion from a grid, lines or a comp, masks over what moves, feature tracks, the camera solve and its error, cleaning up bad tracks, orienting and scaling the scene, a moving object's own track, and the solved camera with the undistorted plate built into the 3D space. Use for "track this shot", "matchmove the plate", "solve the camera of this clip", "camera track for CG", "get me the camera from this footage", "undistort the plate", "lens grid", "track the car in the shot", "put a CG object into this shot", "export the lens distortion to Comp", "why does the solve drift".
allowed-tools: matchmove get_viewer_state show_in_viewer edit_comp animate_scene
---

# Matchmove: a plate's camera, step by step

`matchmove(action, options)` does the steps of the viewer's Matchmove guide (the 3D space's Matchmove
menu) on the open page. Every answer carries `matchmove`: the plate, camera, lens, masks, tracks and solve
in numbers. **Read those numbers after every step**: they are how you know whether the track is good.
Plate coordinates are its own pixels, x right, y down from the top left; frames count from 0.

The order matters. A wrong lens or an unmasked moving car spoils every step after it, and no amount of
solving fixes it later.

## 1. Plate

`matchmove("open", {"plate": "<film, or one frame of a numbered sequence>"})`.

- Track the **original** plate: not a graded, retimed, stabilised, cropped or rescaled copy. The lens and
  the frame size have to be the camera's.
- Look at `result.facts` and `matchmove.camera`. When the file names a focal length and sensor, check them
  against what the user knows. The sensor size is the **recorded** area (the camera's format), not the
  whole chip. A wrong sensor gives a wrong field of view, and then a wrong depth for everything.
- If the user knows the lens and format, give them:
  `matchmove("camera", {"focal_mm": 35, "sensor": [24.9, 14.0], "known": true})`.
  A known focal length makes the solve faster and surer. When unsure, leave `known` false and let the
  solve find it.
- A zoom shot (focal length changing over the shot) is not supported. Say so instead of solving it.

## 2. Lens

Straight lines in the world must stay straight on the plate, or the solve bends the camera path to make
up for it. Pick one way:

- **Grid** (best): a checkerboard or line chart shot with the same lens at the same focus.
  `matchmove("lens", {"how": "grid", "path": "<grid plate>", "frames": [0, 20, 40]})`.
  Use several frames with the chart in different places. Then check:
  - `lens.fit_error_px` under about 0.3 is good.
  - `lens.coverage` under 0.6 means the corners are guessed. Say so, and ask for a chart that fills the frame.
- **Lines**: no grid, but the plate has architecture.
  `{"how": "lines", "lines": [[[x, y], ...], ...]}`. Use 6 or more lines with 3 or more points each, along
  door frames, building edges or horizons, out to the corners where distortion is strongest. Never use
  things that are curved in the world.
- **Comp**: the lens is already in a comp's Lens Distortion node. `{"how": "comp", "path": "..."}`.
- **Typed** (`{"how": "set", "k1": ..., "k2": ...}`) or **none** (CG, or an already undistorted plate).

Leave `k2` unset ("if the data allows") unless there is a reason. Forcing k2 onto weak data overfits.
`fit_center` only helps with a good grid.

To get the lens into compositing: `matchmove("lens_export", {"to": "comp", "mode": "undistort"})`, and
`"redistort"` for CG rendered through the solved camera. `{"to": "stmap"}` writes an ST map. The Comp's
LensDistortion node is the same model (k1, k2, centre), so the numbers carry over exactly.

## 3. Masks

Mask everything that **moves on its own or slides over surfaces**: people, cars, animals, water, waves,
smoke, trees in wind, the sky and clouds, reflections, highlights on glass or cars, screens, and lens flares.
Tracks there pull the camera towards their motion.

`matchmove("mask", {"name": "car", "keys": [{"frame": 0, "points": [[x, y], ...]}, {"frame": 48, "points": [...]}]})`.

- The shape moves in a straight line between keys and is held outside them. Key it where the motion turns.
- A mask needs the same number of points in every key.
- Make masks generous. Losing a few good features costs less than keeping one bad one.

## 4. Tracks

`matchmove("track", {"features": 200})`, then read `matchmove.tracks`:

- `fewest_on_a_frame` should be at least 8, and 20 or more is comfortable. `thin_frames` lists the runs
  below 8. Track again with `from` / `to` over those frames (more features, or a lower `min_ncc` such as
  0.65), or add tracks by hand with `track_add` on clear corners.
- Long tracks are worth more than many short ones. A healthy `median_length` is a good share of the shot.
  Drop very short ones with `matchmove("track_delete", {"shorter_than": 5})`.
- The camera needs **depth and spread**: features near and far, left and right, top and bottom. Tracks
  only on a distant wall, or only on the ground, solve badly. A shot with all tracks far away and no
  parallax is a nodal pan (see step 5).
- Motion blur, rolling shutter, heavy grain and fast whip pans make tracks jump. Expect more cleanup there.

## 5. Solve

`matchmove("solve", {"mode": "auto"})`, then read `matchmove.solve`:

- `rms_px` under 0.5 is very good and under 1 is usable. Over 1 means the track needs work before anything
  is built on it.
- `focal_px` / `focal_mm` should make sense for the camera. A long lens solved as a wide one (or the other
  way round) means bad tracks or a missed lens. Hold a known focal with `refine: "off"`.
- `frames_over_1px` lists frames where the camera is unsure. Look at the tracks there.
- `unsolved_frames` means too few tracks there: back to step 4.

**Clean up and solve again**, a few rounds and not more:

1. `matchmove("track_delete", {"worst_share": 0.05})` removes the worst 5 %. `{"error_over": 2}` removes
   everything over 2 px.
2. `matchmove("solve")` again.
3. Stop when rms stops falling, or when the tracks per frame get thin. Deleting half the tracks to get a
   pretty number gives a camera that fits a few points and nothing else.

A worst track far above the rest is usually a feature that moves: a reflection, an occlusion edge (two
objects at different depths that make a fake corner), or something on a moving thing. Mask the area
rather than only deleting the track, or the next track run finds it again.

**Tripod / nodal pan**: the camera only turns, so there is no depth. `mode: "auto"` finds that by itself
(the answer's `mode` says `tripod`); `mode: "tripod"` forces it. Points then have no true distance, so
put geometry at a distance that looks right rather than trusting their depth.

## 6. Objects (only when something moving has to be tracked)

1. `matchmove("object_add", {"name": "car", "keys": [...its mask...]})`
2. `matchmove("object_track", {"object": "car"})`
3. `matchmove("object_solve", {"object": "car"})`

The object is solved through the camera, so solve the camera first. Its mask also keeps its tracks out of
the camera solve. A rigid object needs 6 or more good tracks on it. Give `scale` when its size is known.

## 7. Into the scene

`matchmove("build")` puts into the 3D space:

- a "Matchmove" group,
- the camera, keyed every frame and looked through,
- the undistorted plate on it,
- the solved points as locators,
- the objects.

Then orient it. A solve's axes and size are arbitrary until you set them:

- **Ground**: `matchmove("orient", {"how": "ground", "points": [ids of 3+ tracks on the floor]})`.
- **Origin**: `{"how": "origin", "points": [id]}` on a meaningful spot, such as where CG will stand.
- **Axis**: `{"how": "x", "points": [a, b]}` along a wall or road.
- **Scale**: `{"how": "scale", "points": [a, b], "distance": 1.8}` uses two points whose real distance is
  known (a door's height, a car's length). The 3D space is in metres, Y up. A wrong scale makes CG look
  like a miniature.

Use track ids that have a 3D point (the solve's points: tracks deleted before the last solve have none).

**Check it**: `matchmove("geometry", {"kind": "locator" | "box" | "plane", "points": [...]})` places
geometry on solved points. Then render or playblast through the matchmove camera
(`animate_scene("playblast", {"camera": "Matchmove camera"})`) and look. Geometry must stick to the plate
without sliding, especially near the frame edges and at the start and end of the shot. Sliding at the
edges means the lens is off; sliding everywhere means bad tracks.

## Saving and reporting

The matchmove saves itself into its `.mm.json` after every change (`matchmove("save")` gives the path).
The camera lives in the 3D scene, and the scene saves as USD with the camera keyed on every frame.

Tell the user:

- the solve's rms in pixels and what it means (very good / usable / not yet),
- the focal length found,
- how many tracks were kept,
- what you masked and why,
- anything you could not fix (thin frames, guessed corners of the lens, a tripod shot without depth).
