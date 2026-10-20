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
