# World Labs Atlas (Marble 2 beta Developer API); Marble splat worlds below

Tools: `atlas_create_world`, `atlas_expand_world`, `atlas_rebuild_world_mesh`, `atlas_splat_to_mesh`,
`atlas_fetch`, `atlas_list` (`agenty_core/tools/worldlabs_atlas.py`), `atlas_views_to_splat`
(`agenty_core/tools/atlas_splat.py`), all also in agentY-mcp; skill `worldlabs-atlas`. Billed to the user's World Labs credits.

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
| `posedImages2Splats` | >= 2 posed RGBD `posedFrames` (image, camera, linear-depth EXR, optional alpha mask), `refine`, `prompt` | a refined Gaussian splat (SPZ), see below |
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

The viewer already loads glb/ply/splats, so no viewer change was needed. Atlas returns a splat only through
`posedImages2Splats` (`atlas_views_to_splat`, below); a splat (that one, or a Marble `.spz`) goes through
`atlas_splat_to_mesh` for a mesh.

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

## A camera move painted by Atlas (`atlas_render_camera_move`)

**Experimental, off by default.** It is paid and has not had a real run yet, so it is switched off: set `[atlas] camera_move_enabled = true` in `config/settings.default.toml` (or `settings.local.json`; `AGENTY_ATLAS_CAMERA_MOVE=1` overrides it, and is how agentY-mcp is switched on) and restart the host. Off, the agent has no `atlas_render_camera_move` tool (agentY and agentY-mcp), the `worldlabs-atlas` skill does not mention it, and `/agentY/atlas_camera_move` answers "disabled". The viewer has a matching switch (Settings -> bEpic Viewer -> Experimental: Atlas camera move) that shows the "Atlas move" button. Turn both on to try it; the skill step for the agent is not shipped while it is off (add a step 6b to `skills/worldlabs-atlas/SKILL.md` telling it to price with `dry_run=true` first and to run it only after a yes).

The Marble 2 Atlas beta API has **no video, trajectory or sequence mode**: `atlasGenerate` takes posed RGBD context
frames and a list of 1280 x 720 target cameras and returns one still per camera (no maximum is published; this layer
uses at most 8 per task). Video along a camera path exists only in World Labs' closed early-access "Atlas" omni model.
So the move is built here (`agenty_core/tools/atlas_camera_move.py`): keyframe cameras along the path are painted by
Atlas, a few per task, each task conditioned on the views the earlier ones painted (they are added, with depth, to a
copy of the world), then every frame is rendered through the grown depth mesh in the bEpic viewer along the exact same
path. At a keyframe the picture is Atlas's own; in between it is those pictures reprojected, which is stable in time
by construction. `quality`: draft = a painted view every 1 s (8 per task), standard = 0.5 s (4 per task), high = 0.25 s.

Price: World Labs publishes none for Atlas (billing docs: "check the rate card in the platform"; 1,000 credits per
US$; no balance API). The estimate is a range from `WLT_ATLAS_CREDITS_PER_TASK` (default 100, an assumption) and
`WLT_CREDITS_PER_USD` (1000): low = one charge per task, high = one per painted view. `max_usd` /
`WLT_ATLAS_MOVE_MAX_USD` (default 5) refuses bigger plans; at or above `WLT_ATLAS_CONFIRM_ABOVE_USD` (default 0.25)
`confirm_cost=true` is needed. `dry_run=true` prints the plan and spends nothing. A re-run with the same arguments
resumes from `<work world>/moves/<name>/plan.json` without paying a task twice.

The viewer: **Atlas move** (Previz panel, next to Atlas) samples the active keyframed camera frame by frame, asks
`POST /bepic/atlas_camera_move` (dry run) for the plan and price, and only the second press sends `confirm_cost`.
ComfyUI-ImageViewer relays to this host's `/agentY/atlas_camera_move` (`src/utils/atlas_move_jobs.py`); the page polls
for tasks done / total and shows the clip in an "Atlas move" tab and on the canvas.

## A 360 environment and a Gaussian splat

`atlas_capture_around(world, center, radius=1.0, step_deg=30, ceiling_floor=True, prompt)`
(`agenty_core/tools/atlas_capture.py`) adds about 40 Atlas views all round the middle of the place: a ring of
cameras standing `radius` behind the centre looking outwards (yaw 0, +-30, ... 180, an inner camera every other
step for parallax), then ceiling and floor. Batches of 8 (one `atlasGenerate` task each) go out in growing yaw,
each with the 4 existing views nearest to it as context, so the unseen sides join what is already there. The
default centre is half-way to what the first picture shows straight ahead.

**Preferred: World Labs builds the splat (`posedImages2Splats`).** `atlas_views_to_splat(world, frames="",
exclude="", check_consistency=True, refine=True, masks="", dry_run, confirm_cost)` (`agenty_core/tools/atlas_splat.py`)
sends the world's posed views to `POST /tasks:posedImages2Splats`, which reconstructs a draft splat from the views'
depth, then refines it (renders new views, cleans them up, trains on them). Researched 2026-10-30 from
[the task guide](https://atlas-beta.worldlabs.ai/docs/posed-images-to-splats),
[splat best practices](https://atlas-beta.worldlabs.ai/docs/splat-best-practices) and the OpenAPI
(`https://api.atlas-beta.worldlabs.ai/api/v2/openapi.json`):

- Request: `posedFrames` (>= 2, ordered; all ONE image size), each `{imageAsset, camera, depth: {depthAsset}}`
  (+ optional `maskAsset`: 8-bit PNG alpha, 1 keep / 0 leave out; + optional `depth.confidenceAsset`); `prompt`
  (optional, enhanced by default; `enhancePrompt: false` skips that), `refine` (default true), `targetSplatCounts`
  (only for a draft: one count, default 2,000,000), `webhookUrl`, `purgeContentOnOperationCompletion`. Assets are
  project asset ids, inline base64 (< 10 MiB) or public URLs. No documented maximum frame count or resolution
  limit; World Labs suggests 8-16 well-spread views with ~50 % overlap (>= 10 %) for one room, covering the floor
  and every side you will look at.
- Cameras: `PinholeCamera` = camera-to-world `position` + quaternion (x, y, z, w) with `coordinateSystem: "rub"`
  (x right, y up, the camera looks down -Z), intrinsics `fx fy cx cy width height` on the image grid. That is how
  `world.json` stores every frame, so world frames go in unchanged (the stored asset ids are reused; a frame
  without them has its PNG and EXR uploaded). A pose from the viewer or the mesh frame is converted first
  (`atlas_render.viewer_pose_to_mesh`, `worldlabs_atlas.mesh_pose_to_camera`: no axis flip, the viewer, glTF
  and Atlas all share RUB). Depth must be the linear-depth EXR (`depthAsset`), not a log-depth PNG.
- Output: `response.refinedSplats` {splat count: {assetId, url}} (refined) or `response.draftSplat` (draft).
  The tool keeps the largest level: `<world>/splats/<name>.spz` (as returned), plus `<name>.ply` and
  `<name>_lite.ply` turned 180 degrees about X (Y down, the 3DGS convention the bEpic viewer turns upright, like a
  `train_splat` PLY), records it under `splats` in `world.json`, opens it in the viewer. (`_lite.ply` only when the
  splat carries higher SH bands; World Labs' are colour only.)
- Measured live 2026-10-30 (the promo's interior2 cabin, 33 views at 1280x720, refined): 17.5 min, SPZ v2,
  2,000,000 splats, SH degree 0, 31 MB (112 MB as PLY), in the Atlas world frame (Y up, floor ~0, the first
  camera at y 1.01): rendered through the input cameras it lines up with the views. Versus the earlier local
  gsplat training on the same views: clearly cleaner from NEW viewpoints (no ghosted plant leaves, no floaters
  under the bench or on the rug, crisp floor), softer and slightly drifted on the input views themselves (PSNR
  18-21 dB against 22-34 dB for the overfitted local splat). What lies outside the windows and the open door
  comes out as a grey haze: put a background behind it, or mask the openings.
- Async: 10-20 minutes; submitted without `?wait`, polled with `:wait`. A timeout keeps the job in
  `<world>/pending/`; `atlas_fetch(operation_id)` finishes it.
- Cost: refined 5,000 credits (US$ 5), draft (`refine: false`) 500 (US$ 0.50); failed runs are free. The tool
  refuses without `confirm_cost=True` (or the turn's cost confirmation); `dry_run=True` lists the views it would
  send, the consistency check and the price, and spends nothing.
- View consistency (before anything is uploaded): every view is compared, in world order (the source picture
  first), with each view already kept that it shares at least 10 % of its sampled pixels with. Its depth is
  reprojected into the other view and back; a pixel conflicts where one view puts a surface more than 15 % (of
  the depth) in FRONT of what the other sees (a surface behind is occlusion, not conflict). A view whose worst
  conflict with a kept view is over 5 % of the shared pixels is left out (`max_conflict`). On the promo's
  interior2 world (57 views) this leaves out 29, 34, 38, 39, 40, 48, 54 and 56: 38-40 are the views that invented
  a coat on the wall, 34 a plant-moved view that had been left out by eye before. Colour is NOT compared
  (lighting and window content differ too much between generated views); a view that only changes what hangs on
  a wall passes - look at the views and `exclude` by hand.

**Local fallback: Brush.** `train_splat(source, steps=30000, max_splats=3000000, sh_degree=3)` (`agenty_core/tools/splat_train.py`) writes
the world as a Nerfstudio dataset (cameras, pictures, and the depth maps unprojected as starting points) and
trains a Gaussian splat with [Brush](https://github.com/ArthurBrussee/brush) (Apache-2.0) as a background
process. Brush `v0.3.0` is downloaded once into `<app>/tools/brush/` (checksum checked) unless `BRUSH_PATH`
names a binary. A copy of the first picture is held out for evaluation; its renders are the progress signal and
the preview. When Brush is done: huge splats and big splats in front of a camera are removed, the splat is
turned into 3DGS axes (Y down; positions, rotations and the SH coefficients turned together) and saved as
`output/3d_scenes/splats/<name>/<name>.ply` plus `<name>_lite.ply` (colour only), and opened in the viewer's 3D
view through the first training camera. `splat_training_status(name)` checks on / finishes a training that
outlived the call. A COLMAP or Nerfstudio folder works as `source` too. It uses every view it is given: leave
out the ones `atlas_views_to_splat(dry_run=True)` reports.

Measured (2026-10-21, RTX 5090, 41 views at 1280x720): 3,000 steps in 1 min 42 s (2.3 M splats), 30,000 steps
about 30 min with ComfyUI's models still on the GPU (free them: the tool does). The restaurant from one picture:
people removed with Qwen-Image-Edit 2511, 41 views, splat holds up all round; mirrors and window glass smear.

## Cost and limits
Atlas reports no per-task cost (`posedImages2Splats` has a documented flat price, above): the operation carries no usage / credit field, so tools return `usage: null`. Credits
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

## The hosted World Labs MCP server

World Labs runs its own MCP server at `https://api.atlas-beta.worldlabs.ai/mcp` (streamable HTTP; listed in the
[worldlabsai/skills](https://github.com/worldlabsai/skills) plugin's `.mcp.json`; checked 2026-10-30). Auth is
OAuth only, a World Labs account sign-in through Clerk (`https://clerk.worldlabs.ai`, discovery at
`/.well-known/oauth-protected-resource/mcp`); the API key is refused (401 with `WLT-Api-Key` or as a bearer).
Its tools (per World Labs' own skill): `list_projects`, `import_image`, `create_image_upload`,
`estimate_camera_depth`, `get_operation`, `get_asset` - **camera + depth from ONE image, nothing else**
("the connected tools cover only this task"; everything else goes through their CLI, `@worldlabsai/cli`).

Compared with agentY's tools: `estimate_camera_depth` is `images2PosedRGBD`, i.e. `atlas_create_world` without
the mesh, the world folder and the follow-on tools. It has no new views, no capture, no splat
(`posedImages2Splats` is not exposed), no Marble. Overlap only; nothing new except that it signs in with the
user's account instead of a project key. Use it from a client without agentY's tools; inside agentY use ours.

agentY carries it as the opt-in `worldlabs` preset (`config/mcp_presets.json`, not selected by default; OAuth
tokens in the gitignored `config/.mcp_tokens/`; startup never opens a browser). **Caveat:** Clerk has dynamic
client registration switched off (`POST /oauth/register` answers `feature_requires_dynamic_oauth_client_registration`)
and advertises client-ID metadata documents instead, so agentY's sign-in fails until the server entry carries
`oauth_client_metadata_url`: an https URL serving agentY's client metadata JSON (`client_id` = that URL,
`redirect_uris: ["http://localhost:<redirect_port>/callback"]`). `src/tools/mcp_tools.py` passes it to the MCP
SDK's OAuth provider. Claude Code and Claude Desktop connect to the URL directly.

## Marble (the World API): splat worlds, SPZ / PLY export

Tools: `marble_create_world`, `marble_export` (`agenty_core/tools/worldlabs_marble.py`; SPZ / PLY code in
`splat_formats.py`; agentY-mcp offers both). Researched 2026-10-25 from the public docs; the paid path has NOT been run.

**Findings (sources)**
- A separate service from Atlas: `https://api.worldlabs.ai/marble/v1`, same `WLT-Api-Key` header, async operations
  polled with `GET /operations/{operation_id}` (~5 min per world). [docs.worldlabs.ai/api](https://docs.worldlabs.ai/api/index.md),
  [OpenAPI](https://docs.worldlabs.ai/api/reference/openapi.md).
- **Our Atlas key is refused by it: a free `GET /credits` with the Atlas key answered 401 (2026-10-25)**, while the same
  key works on the Atlas API. The World API needs its own key from platform.worldlabs.ai, set as `WLT_MARBLE_API_KEY`
  (`WORLDLABS_MARBLE_API_KEY`); the Atlas key is only the fallback. Credits are separate too (API credits are not the web
  app's). [pricing](https://docs.worldlabs.ai/api/pricing.md)
- **Models** (`model`): `marble-1.1-plus` (newest, largest worlds), `marble-1.1`, `marble-1.0`, `marble-1.0-draft`.
  **There is no "Marble 2" in this API**: the models page and the release notes (latest entry 2026-04-02: Marble 1.1 /
  1.1 Plus) do not mention it; "Marble 2 beta" is World Labs' name for the Atlas developer API, which has no
  world-generation task (tasks: atlasTextToImage, images2PosedRGBD, atlasGenerate, atlasMasked, atlasChisel,
  splats2Mesh). [models](https://docs.worldlabs.ai/api/models.md), [release notes](https://docs.worldlabs.ai/marble/release-notes.md),
  [Atlas reference](https://atlas-beta.worldlabs.ai/docs/api-reference). Default of the tool: `marble-1.1-plus`.
- **Cost** (credits, 1,250 per US$, min purchase US$ 5, no expiry): draft 150 (pano) / 230 (text, image) / 250 (multi-image,
  video); `marble-1.0` and `marble-1.1` 1,500 / 1,580 / 1,600; `marble-1.1-plus` the same plus 0-1,500 for world size
  (so 1,600-3,100 for multi-image, up to US$ 2.5). PLY splat export: free. HQ mesh export (GLB): 3,500 credits (not wrapped).
  The finished operation carries `cost.total_credits`, which the tool reports. `GET /credits` gives the balance (free);
  `marble_create_world` reads it first and refuses if it is below the minimum.
- **Inputs** (`world_prompt.type`): `text`; `image` (`image_prompt`, optional `text_prompt`, `is_pano` auto/true/false);
  `multi-image` (`multi_image_prompt`: list of `{azimuth?, content}`, 4 images, 8 with `reconstruct_images`; the only pose is
  `azimuth` in degrees, 0 front / 90 right / 180 back / 270 left - no pitch, no full camera pose); `video`. A content is
  `{source: "uri"|"media_asset"|"data_base64", ...}`. Local files go through `POST /media-assets:prepare_upload` (jpg,
  png, webp) and a PUT to the signed URL; that is what the tool does. Other body fields: `display_name` (64), `seed`,
  `tags`, `permission` (private by default).
- **Assets** of a World: `assets.splats.spz_urls` {`100k`, `500k`, `full_res`}, `assets.splats.semantics_metadata`
  {`metric_scale_factor`, `ground_plane_offset`}, `assets.mesh.collider_mesh_url` (GLB, 100-200k triangles),
  `assets.imagery.pano_url` (2560x1280 equirectangular PNG), `assets.thumbnail_url`, `caption`. PLY: `POST /worlds/{id}:export`
  `{asset_type: "splats", format: "ply", resolution: full_res|500k|150k|100k}`, synchronous, the result `url` is the file.
  [export](https://docs.worldlabs.ai/api/reference/worlds/export.md), [file specs](https://docs.worldlabs.ai/marble/export/specs.md)
- **Axes:** SPZ assets are `marble_raw_opencv` (x right, y down, z forward); Marble's own viewer applies a 180 degree turn
  about X, which is also what the bEpic viewer (Spark) does to every splat, so a Marble splat and a `train_splat` `.ply`
  (Y down) load the same way and nothing is flipped here. For an OpenGL tool negate Y and Z. `semantics_metadata` is metric
  scale + ground only: centres x factor then minus `ground_plane_offset` on Y, log-scales + ln(factor) (`marble_export(metric=true)`
  does this). [rendering-spz](https://docs.worldlabs.ai/api/rendering-spz.md)

**What agentY does**
- `marble_create_world(name, images=[...] | atlas_world="room", prompt, model, ...)` -> `output/3d_scenes/marble/<name>/`
  with `world.json` (world id, operation, cost, scale metadata, inputs), `splat_full_res.spz` and sidecars, opens it in
  the viewer. With `atlas_world` it picks up to 4 posed views spread round the compass and works each view's azimuth out from
  its camera. `dry_run=true` returns the request and the price and spends nothing. A timeout leaves `world.json` as `pending`.
- `marble_export(name, format, resolution, collider_mesh, metric, refresh)`: SPZ at full / 500k / 100k, PLY (World Labs'
  own export, else converted from the SPZ with `splat_formats`), collider mesh; collects a pending creation.
- `splat_formats.py`: pure-numpy SPZ v2/v3 reader, v3 writer, PLY reader/writer (no new dependency; the PyPI `spz` package
  is a 0.0.1 placeholder). SPZ v4 (zstd streams) is refused with a pointer to the PLY export.
- `atlas_splat_to_mesh(splat_path)` takes a Marble world name (its SPZ, fetched if absent; `mesh.glb` lands in that world's
  folder), a `.spz`, or a `.ply` (converted to SPZ first, axes unchanged).

**Not verified live:** every call above is stub-tested only. Unconfirmed: that the Marble key works with the endpoints
as documented, the `data_base64` content shape (the tool uploads media assets instead), whether Marble SPZ files are
version 2/3 (the reader handles those) and that `splats2Mesh` reads a Marble SPZ in its raw axes.
