---
name: worldlabs-atlas
description: Make a 3D scene from a picture or a text prompt with World Labs Atlas (a depth mesh in the picture's camera), grow it with new views, use it as the set for render_camera_move, and render single new views of it from any camera (consistent backgrounds). Use for "turn this image into a 3D world with Atlas / World Labs", "make a courtyard world from a prompt", "extend the scene sideways", "image to Atlas world to camera move to control video", "same room from another angle", "a new shot of this background", "render this view with Atlas", "a full 360 environment", "make a Gaussian splat of this place", "explorable 3D scene from this picture", "make a splat in Marble", "Marble world from these images", "export the splat as SPZ / PLY", "splat to mesh", "a camera move through the Atlas world rendered with Atlas", "render the camera animation with Atlas", "high quality camera move of this place".
allowed-tools: atlas_render_camera_move atlas_create_world atlas_expand_world atlas_rebuild_world_mesh atlas_splat_to_mesh atlas_fetch atlas_list atlas_render_view atlas_open_in_viewer atlas_capture_around train_splat splat_training_status marble_create_world marble_export render_camera_move show_in_viewer
---

# World Labs Atlas: picture or prompt -> 3D scene -> camera move

Atlas is not a one-call world generator. It poses pictures (camera + depth) and invents new views
at cameras you choose; the tools here turn the posed depth into one textured mesh (`world.glb`) in
the FIRST picture's camera frame (camera at the origin looking down -Z) - exactly what
`render_camera_move(model=...)` wants. It bills the user's World Labs credits (`WORLDLABS_API_KEY`),
so say what you are about to spend (one call = one task) and do not loop.

1. **Make the scene.** `atlas_create_world(name, image_paths=[...])` for a picture (several only if
   they show the SAME place with overlapping views), or `atlas_create_world(name, prompt=...)` to
   paint one first (wide eye-level view of one place: layout, materials, light; no people, text or
   close-ups). The result lists the `glb` and the saved views under
   `output/3d_scenes/atlas/<name>/`.
2. **Optional: go beyond the picture.** A camera move past the picture's depth shows holes.
   `atlas_expand_world(name, moves=[{"forward":2,"yaw":25},{"forward":4,"right":-1.5}])` has Atlas
   paint those new views (needs real camera translation, not just a turn; a few metres per step,
   at most 8 per call) and adds them to the mesh.
3. **Film it.** `render_camera_move(name="shot1", model="<glb from step 1>", moves=[...])` - the
   moves are relative to the picture's camera (`forward`, `up`, `right` metres; `yaw`, `pitch`,
   `roll` degrees), then follow the `camera-move-control` skill: the clip is the control video for
   LTX 2.5 Layout-to-Render or Wan Fun Control, with the picture as the first frame / look reference.
   `shading: "clay"` gives the layout look.
4. **A new picture of the same place.** `atlas_render_view(world, camera={...}, upscale=0|2|4)` has Atlas
   paint ONE new view (1280x720) from a camera you choose, optionally upscaled with a local ComfyUI
   upscale model. The camera is in the mesh frame (the first picture's camera = origin, looking down -Z):
   a relative move `{"forward":2,"right":-0.5,"yaw":20}`, or absolute `{"position":[x,y,z],"look_at":[x,y,z]}`,
   or a 4x4 `{"matrix": ...}`; add `"fov"` for a different lens. It lands in `renders/` under the world,
   on the canvas and in the viewer. Pass `add_to_world=true` to make the new view part of the mesh, so
   later renders of the same place stay consistent with it. One call = one billed task: do not loop.
   Use it when a scene must stay the same place across shots:
   - the first frame / reference for an image-to-video model from a different angle than the original
     picture (LTX 2.5 i2v, the Layout-To-Render first frame, MiniMax H3 r2v reference);
   - several establishing angles of one set, before the clips are made;
   - upscale 2 or 4 when the picture goes on as a first frame (1280x720 is the model's native size).
   Keep each camera a believable step (a few metres, under ~60 degrees of turn) from an existing view:
   Atlas invents what no view showed, and a far jump drifts. For a far camera, step there with
   `add_to_world=true` renders, or `atlas_expand_world`.
   Not for a camera MOVE (that is `render_camera_move`, a clip of the mesh), and not when the user only
   wants to look around: that is the viewer.
   **Frames - read this before any absolute camera key.** `forward` / `right` / `up` / `yaw` / `pitch` /
   `roll` are ALWAYS relative to the picture's camera. Absolute `position` + `look_at` keys are read in
   `frame`: `"picture"` = metres from the picture camera (it is the origin, looking down -Z, Y up),
   `"world"` = the scene's own coordinates. For a mesh (`world.glb`, MoGe, SHARP) the two are the same. A
   trained Gaussian splat lives in the Atlas world's absolute frame, where the picture camera stands at
   about y = 1.01, NOT at the origin: so for a splat name the `frame` (it is refused otherwise), or use
   relative moves. The result echoes the resolved `keys`, the `picture_camera` and the `scene_bounds`
   (floor, top, box): read them. A path under the floor, outside the room or looking at nothing is
   refused with the numbers (`camera_outside_scene`) instead of rendered - fix the keys, do not pass
   `check=False`. Splat fly-through, room, picture camera at y = 1.01: relative
   `render_camera_move(name, model="<splat>.ply", moves=[{"t":0},{"t":4.5,"forward":2.2,"right":0.4,"up":-0.3}])`
   or absolute in the splat's coordinates `frame="world"`, `moves=[{"t":0,"position":[-0.1,0.71,-0.4],
   "look_at":[-0.2,0.71,-2.4]},{"t":4.5,"position":[0.45,0.79,-2.3],"look_at":[1.2,0.79,-4.15]}]` (eye height
   ~0.7-0.8, the room's floor at 0). Look at the viewer's own coordinates before inventing keys.
5. **Let the user pick the camera.** `atlas_open_in_viewer(world)` opens the world in the viewer's 3D view,
   looking through a 16:9 camera at the first picture's viewpoint. The user navigates (Alt + mouse) and
   presses **Render with Atlas** in the Previz panel (upscale choice, "add to world"); the picture lands in
   the viewer and as a loader node on the canvas. Tell them that; do not render on their behalf.
6. `show_in_viewer` on the glb only when the user wants to look at it.
6b. **A moving shot of the world - the Atlas camera render.** `atlas_render_camera_move(name, world, moves=[...])`
   is the HIGH-QUALITY way to get a camera move through an Atlas world, and better than any splat trained from
   the same views: Atlas itself paints the pictures along the path. Atlas has NO video mode (the API takes a list
   of cameras and returns one still per camera), so the tool paints keyframes along the path (a painted view
   every 1 / 0.5 / 0.25 s for `quality` draft / standard / high, 4-8 per task, each task sees the views the
   earlier ones painted and they join the world) and then renders every frame through the grown world in the
   viewer (browser tab open). The clip is `output/previz/<world>_<name>.mp4` with `measured` facts and a sidecar.
   The moves are the `render_camera_move` format, relative to the picture camera (`frame="picture"`), or absolute
   in the Atlas world's own coordinates (`frame="world"`); the camera must TRANSLATE (a pure pan gives Atlas no
   depth and is refused: use `render_camera_move` for those). It works on a COPY (`<world>__<name>`) unless
   `grow_world=true`.
   **It costs money and the price is only an estimate**: World Labs publishes no per-task price for Atlas.
   ALWAYS run `dry_run=true` first, tell the user the plan (tasks, painted views, `estimate` low-high in US$,
   `estimate.basis`) and wait for a yes; only then run it with `confirm_cost=true`. A plan above the cap
   (`max_usd`, default 5) is refused; lower the quality or shorten the move. After a timeout or a failure run the
   SAME call again: the tasks already paid are remembered and not paid twice. The user can also press
   **Render camera with Atlas** (Previz panel, "Atlas move") on a keyframed camera: the viewer shows the price
   first and has its own confirm - tell them that, do not run it for them.
   Which to use for a moving shot of an Atlas world:
   - **free preview / layout / control video** -> `render_camera_move` through `world.glb` (or a splat): instant,
     free, stretchy where the mesh has holes;
   - **the shot itself** -> `atlas_render_camera_move`: painted by Atlas, consistent with the world, paid;
   - **a look-around where the camera can go anywhere** -> `atlas_capture_around` + `train_splat` (Brush, local, no
     credits) is a side option; a splat of ~40 painted views is softer than the painted move above.
7. **A full 360 environment / a Gaussian splat.** Two calls, in this order:
   - `atlas_capture_around(world, prompt=...)` has Atlas paint the place ALL ROUND (about 40 views: a ring
     outwards from the middle, then ceiling and floor; 5-6 billed tasks, a few minutes). It grows outwards
     from the picture, so the unseen sides stay consistent. People in the picture get repeated all round:
     for an empty place remove them from the picture FIRST (a local image edit, e.g. Qwen-Image-Edit 2511),
     make the world from the edited picture, and say "no people" in `prompt`. `center` / `radius` for a
     place that is not a room-sized space in front of the picture.
   - `train_splat(world)` trains a Gaussian splat from those views on the local GPU (Brush; downloaded once,
     no API cost): ~2 min for `steps=3000` (a quick look), 10-30 min for the default 30000. It frees ComfyUI's
     VRAM first, cleans floaters, saves a standard 3DGS `.ply` (+ a colour-only `_lite.ply`), shows a preview
     picture and opens the splat in the viewer's 3D view. It returns `"running"` when it takes longer than
     `wait_s`: tell the user, then `splat_training_status(name, wait_s=...)` finishes it. Do not start it twice.
   A splat from the first picture alone (or a few views) is a flat shell: capture first.

Other tools: `atlas_rebuild_world_mesh` (free, offline: finer/lighter mesh, cut the sky with
`max_depth`), `atlas_splat_to_mesh` (a Marble world name, a `.spz` or a `.ply` -> GLB), `atlas_list` (scenes here, or the
project's API jobs), `atlas_fetch(operation_id)` (it also finishes a timed-out `atlas_render_view`: download, upscale, publish).

Errors (`error_kind`): `timeout` -> call `atlas_fetch` with the `operation_id`, do not resubmit;
`balance` / `spend_limit` -> tell the user to add credits / raise the limit on the World Labs
platform; `not_found` -> the task is a beta entitlement the account lacks; `content_refusal` ->
change the prompt or picture, a retry is declined again; `auth` / `scope` -> the key is missing or
needs scopes tasks.create, operations.read, assets.create, assets.read; `rate_limit` -> wait.

Upscale errors do not lose the render: the 1280x720 picture is saved and the result carries `upscale_error`
(ComfyUI down, or no model in `models/upscale_models`); say so and offer to retry with `upscale_model=...`.

## Marble or Atlas or Brush? (a splat world)

World Labs has two services, with their own keys and credits. Pick by what the user needs:

- **Atlas** (everything above): exact camera views, painting the view you look through, growing a world
  view by view, a depth mesh for `render_camera_move`.
- **Marble** (`marble_create_world`, `marble_export`): a READY navigable Gaussian-splat world from one
  to four pictures, no training: SPZ (full ~2M splats, 500k, 100k), PLY, a collider mesh, a panorama.
  Use it for "give me a splat of this", an export for another app, or a splat to turn into a mesh.
  It invents what the pictures do not show on its own: you cannot aim cameras at it.
- **Brush** (`atlas_capture_around` + `train_splat`): for real photos or our own capture sets, when the
  splat must stay faithful to views we hold (local GPU, no credits).

Marble steps:
1. **Price it first, say the price, then run it.** One call = one world, ~5 minutes, billed to the World API
   balance (1,250 credits per US$): marble-1.1 1,600 credits (US$ 1.3), marble-1.1-plus (default, the newest;
   there is no "Marble 2" in this API) 1,600 to 3,100 (up to US$ 2.5), marble-1.0-draft 150-250 for a quick
   look. `marble_create_world(..., dry_run=true)` prints the request and the price and spends nothing.
   Do not loop and do not re-run to compare unless asked.
2. **From an Atlas world:** `marble_create_world(name="room_marble", atlas_world="room")` takes up to 4
   posed views spread round the compass (azimuths worked out from their cameras; 8 with
   `reconstruct_images=true`, which stays closer to the pictures). From pictures:
   `marble_create_world(name, images=[...])` (several only if they show the SAME place; `azimuths=[0,180]`
   if you know where each looks: 0 front, 90 right, 180 back, 270 left). A `prompt` is optional.
3. **Files** land in `output/3d_scenes/marble/<name>/` (`world.json`, `splat_full_res.spz`, thumbnail) and
   the splat opens in the viewer. `marble_export(name, format="ply"|"spz", resolution="full"|"500k"|"100k",
   collider_mesh=true)` fetches more of it (free; PLY comes from World Labs, else it is converted here;
   `metric=true` gives metres with the floor at y = 0). It also collects a creation that timed out.
4. **A mesh:** `atlas_splat_to_mesh(splat_path="<marble world name>")` (also a `.spz` or a `.ply` path) ->
   `mesh.glb` in the same folder. Billed to the Atlas credits.
5. **Axes:** files are raw OpenCV (x right, y DOWN, z forward) - the same Y-down frame as `train_splat`'s
   `.ply`; the viewer turns it upright. Scale is not metres (see `semantics`). For an OpenGL tool negate Y and Z.

Marble errors: `auth` -> the Marble API needs its OWN key, `WLT_MARBLE_API_KEY` (the Atlas key is refused
with 401); `balance` -> credits at platform.worldlabs.ai/billing; `content_refusal` -> change the
picture / prompt; `timeout` -> `marble_export(name)` collects it, do not resubmit.
