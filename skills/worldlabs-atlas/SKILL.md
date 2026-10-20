---
name: worldlabs-atlas
description: Make a 3D scene from a picture or a text prompt with World Labs Atlas (a depth mesh in the picture's camera), grow it with new views, use it as the set for render_camera_move, and render single new views of it from any camera (consistent backgrounds). Use for "turn this image into a 3D world with Atlas / World Labs", "make a courtyard world from a prompt", "extend the scene sideways", "image to Atlas world to camera move to control video", "same room from another angle", "a new shot of this background", "render this view with Atlas", "a full 360 environment", "make a Gaussian splat of this place", "explorable 3D scene from this picture".
allowed-tools: atlas_create_world atlas_expand_world atlas_rebuild_world_mesh atlas_splat_to_mesh atlas_fetch atlas_list atlas_render_view atlas_open_in_viewer atlas_capture_around train_splat splat_training_status render_camera_move show_in_viewer
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
5. **Let the user pick the camera.** `atlas_open_in_viewer(world)` opens the world in the viewer's 3D view,
   looking through a 16:9 camera at the first picture's viewpoint. The user navigates (Alt + mouse) and
   presses **Render with Atlas** in the Previz panel (upscale choice, "add to world"); the picture lands in
   the viewer and as a loader node on the canvas. Tell them that; do not render on their behalf.
6. `show_in_viewer` on the glb only when the user wants to look at it.
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
`max_depth`), `atlas_splat_to_mesh` (a Marble `.spz` -> GLB), `atlas_list` (scenes here, or the
project's API jobs), `atlas_fetch(operation_id)` (it also finishes a timed-out `atlas_render_view`: download, upscale, publish).

Errors (`error_kind`): `timeout` -> call `atlas_fetch` with the `operation_id`, do not resubmit;
`balance` / `spend_limit` -> tell the user to add credits / raise the limit on the World Labs
platform; `not_found` -> the task is a beta entitlement the account lacks; `content_refusal` ->
change the prompt or picture, a retry is declined again; `auth` / `scope` -> the key is missing or
needs scopes tasks.create, operations.read, assets.create, assets.read; `rate_limit` -> wait.

Upscale errors do not lose the render: the 1280x720 picture is saved and the result carries `upscale_error`
(ComfyUI down, or no model in `models/upscale_models`); say so and offer to retry with `upscale_model=...`.
