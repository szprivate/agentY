---
name: worldlabs-atlas
description: Make a 3D scene from a picture or a text prompt with World Labs Atlas (a depth mesh in the picture's camera), grow it with new views, and use it as the set for render_camera_move. Use for "turn this image into a 3D world with Atlas / World Labs", "make a courtyard world from a prompt", "extend the scene sideways", "image to Atlas world to camera move to control video".
allowed-tools: atlas_create_world atlas_expand_world atlas_rebuild_world_mesh atlas_splat_to_mesh atlas_fetch atlas_list render_camera_move show_in_viewer
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
4. `show_in_viewer` on the glb only when the user wants to look at it.

Other tools: `atlas_rebuild_world_mesh` (free, offline: finer/lighter mesh, cut the sky with
`max_depth`), `atlas_splat_to_mesh` (a Marble `.spz` -> GLB), `atlas_list` (scenes here, or the
project's API jobs), `atlas_fetch(operation_id)`.

Errors (`error_kind`): `timeout` -> call `atlas_fetch` with the `operation_id`, do not resubmit;
`balance` / `spend_limit` -> tell the user to add credits / raise the limit on the World Labs
platform; `not_found` -> the task is a beta entitlement the account lacks; `content_refusal` ->
change the prompt or picture, a retry is declined again; `auth` / `scope` -> the key is missing or
needs scopes tasks.create, operations.read, assets.create, assets.read; `rate_limit` -> wait.
