---
name: matchmove
description: Track a plate's camera in the ComfyUI Image Viewer's matchmove - the lens distortion from a grid, lines or a comp, masks over what moves, feature tracks, the camera solve and its error, cleaning up bad tracks, orienting and scaling the scene, a moving object's own track, and the solved camera with the undistorted plate built into the 3D space. Use for "track this shot", "matchmove the plate", "solve the camera of this clip", "camera track for CG", "get me the camera from this footage", "undistort the plate", "lens grid", "track the car in the shot", "put a CG object into this shot", "export the lens distortion to Comp", "why does the solve drift", "track my iPhone clip", "the solve looks squashed", "use the iPhone's depth", "set the ground", "mask the people with AI".
allowed-tools: matchmove get_viewer_state show_in_viewer edit_comp animate_scene analyze_image view_image
---

# Matchmove: a plate's camera, step by step

`matchmove(action, options)` does the steps of the viewer's Matchmove guide (the 3D space's Matchmove
menu) on the open page. Every answer carries `matchmove`: the plate, camera, lens, masks, tracks and solve
in numbers. **Read those numbers after every step**: they are how you know whether the track is good.
Plate coordinates are its own pixels, x right, y down from the top left; frames count from 0.

The order matters. A wrong lens or an unmasked moving car spoils every step after it, and no amount of
solving fixes it later.

## 0. Is a plate loaded already?

The user often loads the plate themselves (the guide's Plate step, the Plate panel, the plate in the main
view). The turn's `[VIEWER]` lines then carry a `matchmove plate=...` line: its path, size, frame range,
the frame shown ("this frame"), the step, the tracks and the solve. `get_viewer_state` has the same under
`state.matchmove`. **Start with `matchmove("info")`** and go on with that plate; `open` only when the user
wants another one. Never ask the user for the plate's path when one is loaded.

## Looking at the plate

`matchmove("frame", {"frames": [0, 40, 80]})` (or `{"frame": n}`; no frame = the one shown) saves those
frames as pictures with the tracks drawn on them (green: solved under 1 px, yellow: not solved yet, red:
over 1 px or switched off; their ids beside them when there are few) and lists each frame's tracks with
x, y and error. Hand `result.frames[i].path` (or `result.contact_sheet`) to
`analyze_image(file_paths=[...])`. Use it to see what is in the shot (what moves and needs a mask, where
the floor is, straight lines for the lens), to check tracks sit on corners rather than on reflections or
edges, and before telling the user anything about how the plate looks. `"undistort": true` shows the plate
through the lens (no tracks drawn). At most 16 frames a call.

## 1. Plate

`matchmove("open", {"plate": "<film, or one frame of a numbered sequence>"})` - only for a plate that is
not loaded yet (step 0).

- Track the **original** plate: not a graded, retimed, stabilised, cropped or rescaled copy. The lens and
  the frame size have to be the camera's.
- Look at `result.facts` and `matchmove.camera`. When the file names a focal length and sensor, check them
  against what the user knows. The sensor size is the **recorded** area (the camera's format), not the
  whole chip. A wrong sensor gives a wrong field of view, and then a wrong depth for everything.
- If the user knows the lens and format, give them:
  `matchmove("camera", {"focal_mm": 35, "sensor": [24.9, 14.0], "known": true})`.
  A known focal length is where the solve starts. With `refine: "auto"` (the default) the solve still
  refines it, which is what you want: metadata that is a little off gives a squashed or stretched solve
  when it is held fixed. Hold it (`refine: "off"`) only for a lens you are sure of, measured on set.
  When unsure, leave `known` false and let the solve find it.
- A camera the user names but the file does not: `matchmove("camera", {"preset": "iphone14-wide"})`. The
  list has phones (`iphone12-wide` .. `iphone16-wide`, `iphone14pro-wide`, `iphone-ultra`,
  `iphone-tele77`, `iphone-tele120` ...) as 35 mm equivalents on the plate's own shape; an unknown id
  answers with the whole list.
- An anamorphic plate: `{"pixel_aspect": 2}` (or 1.33, 1.5 ...) before the solve. Films say it themselves
  when they carry it; check `matchmove.camera.pixel_aspect`.
- A zoom shot (focal length changing over the shot) is not supported. Say so instead of solving it.

### A plate from an iPhone

An iPhone's own film (.mov) or photo (.heic, .heif) is read in full: open the original file straight off
the phone (AirDrop or a cable, "Most Compatible" off), not a copy an app re-encoded, or the metadata is gone.

- `result.facts` and `matchmove.camera` give the model and the lens (wide, ultra wide, tele) with its focal
  length from the phone's 35 mm equivalent. `matchmove.phone` says which camera, how it was turned (a film
  shot upright is turned upright for you), the iOS version, and whether it carries **depth**.
- A phone stabilises and lens-corrects its films, which crops them a little: its focal length is a good
  start, never the answer. A fresh iPhone film therefore starts with `refine: "on"`; keep it.
- Only a **Cinematic** film carries depth (`matchmove.phone.depth` with `samples`); ordinary iPhone video
  has none, no gyro data and no LiDAR. A HEIC portrait photo may carry a depth picture. Use the depth in
  step 7 (`ground` how `phone`) and, when the user wants the maps, `matchmove("phone_export")` writes the
  depth track as a 16-bit picture sequence (bright = near) and answers the pattern.
- The phone's depth has no units: it sets the ground, not the scale. Measure for the scale (step 5).
- The phone's lens distortion is already corrected in its film: use lens `none` unless lines still bend.

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

**With AI** (one ComfyUI job, a matte on every frame of the stretch, no keys to set):

- `matchmove("ai_mask", {"engine": "objects", "prompt": "person"})` - SAM 3 finds what the words name
  ("person", "car", "water", "sky", "tree"). One run per kind of thing.
- `matchmove("ai_mask", {"engine": "background"})` - a background remover keeps the subject: for a shot
  that follows one person or object. Films go through the video model (at most 60 s).
- `step` (every Nth frame) and `scale` (0.25 / 0.5 runs the model on a smaller copy: faster, rougher
  edges) for long or large plates; `from` / `to` for part of the shot; `object` makes it an object's mask.
- A model that is not installed refuses with what is missing: tell the user what to download in the guide.
- Track again after a mask is added (`matchmove("track")`): tracks already inside it are switched off.

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
- **Hard shots** (things coming into view, a fast turn, a start where little is trackable): after the first
  pass, `matchmove("track", {"seeds": "thin"})` finds new points in the middle of every thin stretch and
  follows them both ways; `{"seeds": [frame, ...], "direction": "both"}` does it on frames you pick;
  `{"extend": true}` (or a list of track ids) carries tracks on past where they stopped. `direction`
  (`forward`, `backward`, `both`) and `start` (the frame to start from) also work on a plain track.

## 5. Solve

`matchmove("solve", {"mode": "auto"})`, then read `matchmove.solve`:

- `rms_px` under 0.5 is very good and under 1 is usable. Over 1 means the track needs work before anything
  is built on it.
- `focal_px` / `focal_mm` should make sense for the camera. A long lens solved as a wide one (or the other
  way round) means bad tracks or a missed lens. Hold a known focal with `refine: "off"`.
- `focal_check` (free solves) re-solves with the focal 10 % shorter and longer. `weak: true` means the
  shot cannot tell the focal length apart (little parallax): the proportions in depth are not to be
  trusted - give the real focal length, or measure (below). `better` names a direction that fitted better.
- `units` is `"m"` once a distance is measured, else the solve's own units.

### Measurements: when the solve looks squashed or stretched

A solve can fit the plate well (low rms) and still be wrong in depth: a room too shallow, a floor too
steep, a cube that comes out as a slab. Real measurements fix that. Pick tracks with a 3D point and add:

- `matchmove("measure", {"add": {"kind": "distance", "ids": ["t12", "t40"], "metres": 2.1}})` - two tracks
  and how far apart they are in the world. One sets the scale (1 unit = 1 m); two or more **in different
  directions** (across and into the picture) fix the proportions.
- `{"kind": "plane", "ids": [3+ tracks on one floor, wall or table]}`, `{"kind": "line", "ids": [3+ along
  one straight edge]}`, `{"kind": "right", "ids": [end, corner, end]}` (a corner of a room, a box, a door).

Then `matchmove("solve")` again. `matchmove.measurements` says for each whether it now holds. Measured
tracks are never deleted by the clean-up. `{"remove": id}` or `{"clear": true}` takes them off. Ask the
user for a real measurement (a door is about 2.0 m, a table top 0.75 m high, a parking bay 2.5 m wide)
when the shot needs one.
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

Then set it on the ground. `matchmove("ground", {...})` finds the floor and puts it level at height 0:

- `{"how": "auto"}` - the widest low flat surface among the solved points. Try this first.
- `{"how": "points", "points": [3+ track ids on the floor]}` - when auto picks a table or a wall.
- `{"how": "ai", "prompt": "floor"}` - SAM 3 finds the floor (or "road", "grass") on 5 frames; the tracks on
  it make the plane. Good when few tracks lie on the ground.
- `{"how": "depth"}` - a depth model on 5 frames gives dense points for the plane, scaled onto the solve's
  tracks. With the metric model (`metres`: true, the default) the world comes out in metres and
  `ground.camera_height` says how high the camera was: check it against what the user knows.
- `{"how": "phone"}` - an iPhone Cinematic film's own depth: no model and no ComfyUI job. Not in metres.

Read `ground.tilt` (degrees off level the cameras suggest; a few is fine) and `ground.below` (points under
the floor: many means the wrong surface). `lift` (scene units) raises or lowers it, `flip` turns it over
when the camera comes out below the floor.

Then orient the rest. A solve's axes and size are arbitrary until you set them:

- **Ground by hand**: `matchmove("orient", {"how": "ground", "points": [ids of 3+ tracks on the floor]})`
  (the same as ground `points`, without the search).
- **Origin**: `{"how": "origin", "points": [id]}` on a meaningful spot, such as where CG will stand.
- **Axis**: `{"how": "x", "points": [a, b]}` along a wall or road.
- **Scale**: `{"how": "scale", "points": [a, b], "distance": 1.8}` uses two points whose real distance is
  known (a door's height, a car's length). The 3D space is in metres, Y up. A wrong scale makes CG look
  like a miniature. Skip it when a distance was measured (step 5) or the metric depth set the scale.

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
- the focal length found (and whether the focal check called it weak),
- the scale: measured, from metric depth, or not set,
- how many tracks were kept,
- what you masked and why (by hand or with AI),
- for an iPhone plate: which camera, and whether its depth was used,
- anything you could not fix (thin frames, guessed corners of the lens, a tripod shot without depth).
