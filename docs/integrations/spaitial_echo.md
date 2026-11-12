# SpAItial Echo: the second world generator

Beside World Labs (Atlas, Marble, see [worldlabs_atlas.md](worldlabs_atlas.md)) agentY can make a world with
**SpAItial Echo** (SpAItial, Munich): a navigable Gaussian-splat world from **one picture**, a **360
panorama** or a **text prompt**, with a mesh export if you want one.

## Set it up

1. Make an API key at [developers.spaitial.ai](https://developers.spaitial.ai) (it starts `spt_live_`).
2. agentY Settings > API keys > **SpAItial Echo**, paste it (or `SPAITIAL_API_KEY=` in `.env`).
3. Buy credits there. 1 credit = US$ 0.01.

## Use it

- **In the viewer** (no agent turn): the 3D space, Atlas menu > **New Echo world…** (or New Atlas world… and
  switch *Engine* to Echo). Type a description or drop ONE picture, pick the quality, press Create. The splat
  opens in the 3D space when it is done.
- **Ask agentY**: "make an Echo world of this picture", "try the same place with SpAItial". It prices it
  first (`echo_create_world(..., dry_run=true)`).

| Model | Time | Price |
|---|---|---|
| `echo-2` (Standard, default) | about 8 minutes | 160 credits, US$ 1.60 |
| `echo-hq` (High detail) | about an hour | 800 credits, US$ 8 |

SpAItial takes a picture OR words, never both: with a picture the words are left out.

## Files

`<ComfyUI output>/3d_scenes/echo/<name>/`: `world.json` (the job, the model, the request id), `splat.spz`
(or `.ply` / `.sog`), `thumbnail.*`, `panorama.*`, and `mesh.ply` / `mesh_simplified.ply` / `floorplan.png`
when asked for. `echo_export(name, what=...)` fetches more later and collects a world whose creation timed out.
SpAItial's docs give the mesh and floorplan exports no price.

## Notes

- The key is sent only to `api.spaitial.ai` (or `SPAITIAL_BASE_URL`), never to the signed storage links the
  downloads redirect to.
- The client is written from SpAItial's published REST reference (docs.spaitial.ai/api); no SDK, plugin or
  third-party node code is used. Some response fields are not in their schema yet, so the client reads them
  loosely; if SpAItial changes them, `world.json` keeps the raw request id so `echo_export` can collect it.
- The splat's axes are as SpAItial writes them; the viewer turns splats upright the way ComfyUI does. Not yet
  tried with a real key (2026-11-12).
