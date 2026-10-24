---
name: camera-move-control
description: Make a new shot with a chosen camera move from ONE picture - picture to 3D scene, animate a camera through it in the bEpic viewer, render the move, and use the render as the control video of a video model (Wan Fun Control, LTX IC-LoRA). Use for "fly the camera through this", "a low drone pass toward the houses", "same place, new camera move", "turn this picture into a 3D scene and film it".
allowed-tools: render_camera_move run_world_builder show_in_viewer
---

# Picture -> 3D scene -> camera move -> control video

The chain, in order. Say the plan in one line first, then run it.

1. **A 3D scene of the picture.** Stage the picture, then `run_world_builder` with the
   world's name and an engine that builds the model in the picture's own camera
   (`s3d_moge`: metric mesh, about 20 s; `s3d_sharp`: Gaussians meshed with colours). Camera
   models line up with the picture exactly, so the camera can start where the picture's
   camera stood. Ask the user for the engine only if they have a preference; `s3d_moge` is
   the default here.
   With a World Labs key (`WORLDLABS_API_KEY`) the user may ask for Atlas instead: follow the
   `worldlabs-atlas` skill, then pass its `glb` as `render_camera_move(model=...)`.
2. **Animate and render the camera.** `render_camera_move(name, world, moves)`. The viewer
   shows the scene in a "Previz: <name>" tab, plays the camera through it and writes
   `output/previz/<name>.mp4`; the tool returns the path. The moves are relative to the
   picture's camera: `forward` metres along the horizon, `up`, `right`, and `yaw` / `pitch`
   / `roll` degrees, at `t` seconds. Needs at least two. Examples:
   - low drone pass toward the horizon, slight rise: `[{"t":0},{"t":5,"forward":45,"up":1.2}]`
   - push-in with a drift: `[{"t":0},{"t":5,"forward":12,"right":1.5,"yaw":-4}]`
   - circling a person or object ("orbit", "circle around him", "arc shot"): do NOT key it - use
     `render_camera_move(name, world, orbit={...})` instead of `moves`. A pivot on the subject turns
     about the vertical axis while the camera, its child, moves straight back / up, so the arc is
     smooth and the horizon level. E.g. close-up to wide, 180 degrees behind him over 5 s:
     `orbit={"center": [x, eye_height, z], "degrees": 180, "seconds": 5, "distance": [0.5, 7],
     "height": [0, 1.4]}`, `ease: "linear"`. The centre is the subject's position at eye height
     (a close-up) or chest height; positive degrees turn left.
   - a product turntable, whip turns, holds on hero views (side / front / back): the same orbit
     rig with a speed curve, `orbit={"center": [x, y, z], "distance": [d, d], "height": [h, h],
     "keys": [{"t": 0, "degrees": 0}, {"t": 1.3, "degrees": 15}, {"t": 1.65, "degrees": 105,
     "ease": "smooth"}, {"t": 2.95, "degrees": 120}, ...]}` - each key is the angle turned so far,
     its ease how the turn arrives there; every step under 180 degrees. Never build a turntable or
     a whip from absolute `position` + `look_at` moves: the camera cuts across the circle (the
     subject jumps toward the lens) and the Euler angles flip mid-turn.
   - **Frames.** `forward`/`right`/`up`/`yaw`/`pitch`/`roll` are always relative to the picture's camera;
     absolute `position`+`look_at` are read in `frame` (`"picture"`: metres from the picture camera,
     `"world"`: the scene's own coordinates). Same for a mesh; for a trained Gaussian splat (the Atlas world's
     absolute frame, picture camera at y~1.01, not the origin) you must name it. The result lists the resolved
     `keys`, `picture_camera` and `scene_bounds`; a path under the floor / outside the scene / looking at nothing
     is refused with the numbers (`camera_outside_scene`) - fix the keys. Splat fly-through:
     `render_camera_move(name, model="<splat>.ply", frame="world", moves=[{"t":0,"position":[-0.1,0.71,-0.4],
     "look_at":[-0.2,0.71,-2.4]},{"t":4.5,"position":[0.45,0.79,-2.3],"look_at":[1.2,0.79,-4.15]}])`, or
     relative `moves=[{"t":0},{"t":4.5,"forward":2.2,"right":0.4}]`.
   Keep it believable: a single picture holds only what it shows, so a move far past the
   picture's depth shows holes. About the depth of the scene is the limit; say so when the
   user asks for more. `shading: "clay"` gives the grey layout look some models want.
3. **Control video -> new shot.** The rendered clip is the control signal. Run a workflow with
   a control-video input and the picture as the first frame / look reference
   (`prepare_workflow`, with the clip as an input): Wan 2.2 Fun Control (depth or the render
   itself as control), or an LTX IC-LoRA (LTX 2.5 Layout-to-Render takes the clay render as
   the layout and the picture as the first frame). For depth-controlled models, estimate depth
   from the render (Depth Anything) and feed that.
4. Show the result to the user (`show_in_viewer`) only when asked; the render and the new shot
   already reach the viewer.

5. **Check the render with the measurements, not just a description.** The render result carries `measured`
   (camera moving / static and which way, empty-fraction per frame, black / soft / frozen frames) and
   `render_warnings`; `video_frames` and `analyze_video` return the same for any clip. A vision model
   invents smear, voids and "locked off" cameras: when it disagrees with `measured`, trust `measured` and say
   so - never call a camera static when `camera.state` is "moving", nor a frame a void when its
   `empty_fraction` is under 0.6.

If `render_camera_move` says it timed out, the bEpic viewer is not open in a browser tab -
tell the user to open ComfyUI with the viewer and retry. Do not loop on it.
