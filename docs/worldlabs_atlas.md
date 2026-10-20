# World Labs Atlas (Marble 2 beta Developer API)

Tools: `atlas_create_world`, `atlas_expand_world`, `atlas_rebuild_world_mesh`, `atlas_splat_to_mesh`,
`atlas_fetch`, `atlas_list` (`agenty_core/tools/worldlabs_atlas.py`, also in agentY-mcp; skill
`worldlabs-atlas`). Billed to the user's World Labs credits.

## Setup
- Key: `WORLDLABS_API_KEY` in `.env` (`WLT_API_KEY`, the docs' own name, is read too). Create it on the World Labs
  platform (API keys) with scopes `tasks.create`, `operations.read`, `assets.create`, `assets.read`. Sent as the
  `WLT-Api-Key` header; never logged.
- Endpoint: `https://api.atlas-beta.worldlabs.ai/api/v2` (override with `WLT_API_BASE_URL`).
- Atlas tasks are a beta entitlement: a key without access gets 404 on the task (`error_kind: not_found`).
- Depth maps are EXR: needs the `OpenEXR` package (agenty_core depends on it; `pip install OpenEXR` if missing).

## What the API is
Not a "prompt -> world" call. Every task is an async *operation* (`POST /tasks:<name>` -> `done` / `response` /
`error`; poll with `POST /operations/{id}:wait` (30 s slices) or `GET`; kept 30 days). Idempotency-Key on submit.

| Task | In | Out |
|---|---|---|
| `atlasTextToImage` | prompt, `aspectRatio`, `returnDepth` | image (1024x578 at 16:9), camera, depth EXR + confidence PNG |
| `images2PosedRGBD` | `frames[].imageAsset` (url / assetId / base64 < 10 MiB), `targetResolution` | image, estimated RUB camera, linear depth EXR, confidence; several frames are reconstructed together |
| `atlasGenerate` | posed RGBD `contextFrames`, `targetCameras` (exactly 1280x720), prompt, `returnDepth` | one new view per camera (+ depth), `promptUsed` |
| `splats2Mesh` | a `.spz` asset | GLB mesh |
| `atlasMasked`, `atlasChisel` | inpainting / depth-guided generation | not wrapped |

Also `/assets` (create from url/base64/upload URL, read URLs), `/operations` (list, cancel, trace), `/health`.
Output URLs live 24 h (`:createReadUrl` makes a new one from the `assetId`); the tools download everything at once.

## What agentY builds from it
`output/3d_scenes/atlas/<name>/`: `view_NNN.png`, `depth_NNN.exr`, `conf_NNN.png`, `world.json` (cameras, asset ids,
operations), `world.glb`, and `.agenty.json` sidecars. `world.glb` is a textured, unlit depth mesh (one mesh per
view, glTF `KHR_materials_unlit`), unprojected from the depth grid (stride `mesh_detail`, default 4; triangles
across depth jumps and low-confidence / sky pixels are cut), in the FIRST view's camera frame: camera at the
origin, looking down -Z, Y up. That is what `render_camera_move(model=<glb>)` assumes, so

    atlas_create_world(name, image_paths=[picture])      # or prompt=...
    atlas_expand_world(name, moves=[...])                # optional: views the picture never showed
    render_camera_move(name, model=<glb>, moves=[...], shading="clay")   # control video
    -> LTX 2.5 Layout-to-Render / Wan Fun Control (see the camera-move-control skill)

The viewer already loads glb/ply/splats, so no viewer change was needed. Atlas never returns a splat; a Marble
`.spz` goes through `atlas_splat_to_mesh`.

## Measured live (2026-10-20, one prompt world + one expansion + one picture world)
- Depth EXR: single channel `Y`, float, ZIP, row 0 = top (no flip); NaN where there is no depth (sky). The mesh
  matched between the first view and a generated view to ~0.05 units (floor at y -2.03 vs -2.08), so reading depth
  as z-depth and scaling the 1024x578 camera to 1280x720 for targets is right.
- `images2PosedRGBD` puts the camera at ~1.86 above a gravity-levelled origin; the base-frame transform absorbs it.
- A prompt world is about 8 MB of GLB at stride 4 (34k vertices per view); stride 8 is a quarter of that.

## Render a view (agent tool and viewer button)
`atlas_render_view(world, camera, upscale=0|2|4, add_to_world, depth, fov, prompt)` (`agenty_core/tools/atlas_render.py`)
asks `atlasGenerate` for ONE new picture at one target camera (1280 x 720 grid), with the 1 to 4 posed views nearest the
camera as context. Camera in the MESH frame (`world.glb`: first picture's camera at the origin, looking -Z): a relative
move (`forward/right/up/yaw/pitch/roll`), `position` + `look_at`, a 4x4 `matrix`, or the viewer's own camera
(`viewer`: raw three.js `matrixWorld.elements` of the camera and of the model's node + fov/aspect). The conversion to
Atlas's world frame is `mesh_pose_to_camera` (`R_w = R_0 R_m`, `t_w = R_0 t_m + t_0`, `R_0, t_0` = the first camera);
three.js, glTF and Atlas's "rub" cameras share axes (X right, Y up, camera looks -Z), so the viewer pose needs no axis
flip, only the model node's transform undone (`viewer_pose_to_mesh`). Files: `<world>/renders/render_NNN.png`
(+ `_x2` / `_x4`) with sidecars; recorded in `world.json` under `renders`; `add_to_world` also appends the view (with
depth) to `frames` and rebuilds `world.glb`. A timeout is resumable: `atlas_fetch(operation_id)` completes it
(download, upscale, publish).

Upscale: a small ComfyUI workflow through the core client (`LoadImage` -> `UpscaleModelLoader` ->
`ImageUpscaleWithModel` [-> `ImageScaleBy` lanczos when the model's native factor is not the one asked] ->
`SaveImage`). The model comes from `/models/upscale_models`: photographic names first (UltraSharp, NMKD, ESRGAN...),
illustration ones (anime...) last, a native factor equal to the request preferred; `upscale_model` forces one. A failed
upscale keeps the render and reports `upscale_error`.

The viewer: `atlas_open_in_viewer(world)` (or the Previz panel opened on any `world.glb` from an Atlas folder) shows
a **Render with Atlas** button. The viewer runs in ComfyUI's Python, which has neither `agenty_core` nor the key, so
`POST /bepic/atlas_render` (ComfyUI-ImageViewer) forwards to this host's `POST /agentY/atlas_render` (address and
session token from the connect extension's `/agent/host_info`), which runs the same core function in a background job
(`src/utils/atlas_render_jobs.py`); the page polls `GET /bepic/atlas_render?id=`. The finished picture is delivered
outside any turn: a loader node on the canvas, a toast, and the viewer's agentY tab.

## A 360 environment and a Gaussian splat

`atlas_capture_around(world, center, radius=1.0, step_deg=30, ceiling_floor=True, prompt)`
(`agenty_core/tools/atlas_capture.py`) adds about 40 Atlas views all round the middle of the place: a ring of
cameras standing `radius` behind the centre looking outwards (yaw 0, +-30, ... 180, an inner camera every other
step for parallax), then ceiling and floor. Batches of 8 (one `atlasGenerate` task each) go out in growing yaw,
each with the 4 existing views nearest to it as context, so the unseen sides join what is already there. The
default centre is half-way to what the first picture shows straight ahead.

`train_splat(source, steps=30000, max_splats=3000000, sh_degree=3)` (`agenty_core/tools/splat_train.py`) writes
the world as a Nerfstudio dataset (cameras, pictures, and the depth maps unprojected as starting points) and
trains a Gaussian splat with [Brush](https://github.com/ArthurBrussee/brush) (Apache-2.0) as a background
process. Brush `v0.3.0` is downloaded once into `<app>/tools/brush/` (checksum checked) unless `BRUSH_PATH`
names a binary. A copy of the first picture is held out for evaluation; its renders are the progress signal and
the preview. When Brush is done: huge splats and big splats in front of a camera are removed, the splat is
turned into 3DGS axes (Y down; positions, rotations and the SH coefficients turned together) and saved as
`output/3d_scenes/splats/<name>/<name>.ply` plus `<name>_lite.ply` (colour only), and opened in the viewer's 3D
view through the first training camera. `splat_training_status(name)` checks on / finishes a training that
outlived the call. A COLMAP or Nerfstudio folder works as `source` too.

Measured (2026-10-21, RTX 5090, 41 views at 1280x720): 3,000 steps in 1 min 42 s (2.3 M splats), 30,000 steps
about 30 min with ComfyUI's models still on the GPU (free them: the tool does). The restaurant from one picture:
people removed with Qwen-Image-Edit 2511, 41 views, splat holds up all round; mirrors and window glass smear.

## Cost and limits
Atlas reports no per-task cost: the operation carries no usage / credit field, so tools return `usage: null`. Credits
(1,000 per US$, no expiry) and a monthly spend limit are on the platform's Billing / Usage pages. Rate limits are
"to be announced"; 429 / 503 are retried a few times with `Retry-After`.

## Errors (`error_kind`)
`auth` (401 / no key), `scope` (403), `not_found` (404: task not enabled), `balance` (402 `CREDITS_UNAVAILABLE`),
`spend_limit` (402 `SPEND_LIMIT_EXCEEDED`), `rate_limit` (429 / 503), `args` (422 / bad input), `content_refusal`
(operation `CONTENT_POLICY`), `execution`, `execution_timeout`, `timeout` (still running: the error carries
`operation_id`; `atlas_fetch` collects it, the job record is kept under `<world>/pending/`), `dependency` (no EXR
reader), `api`.

## Not done / assumptions
`atlasMasked` / `atlasChisel` are not wrapped; `splats2Mesh` and the `.spz` upload path are only mock-tested; the
multi-image `images2PosedRGBD` reconstruction and `atlas_expand_world` with several moves were not run live beyond one
view; GLB loading in the bEpic viewer itself was not exercised (the GLB parses in trimesh / pygltflib).
