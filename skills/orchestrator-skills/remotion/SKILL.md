---
name: remotion
description: Make programmatic motion-graphics video (titles, promos, explainers, animated text/data, stills, GIFs) with Remotion — React compositions rendered to mp4. Activate when the user asks for a Remotion video, kinetic typography, a title card/lower third, or "install the Remotion MCP".
allowed-tools: add_mcp_server, use_mcp_server, list_mcp_servers, get_agent_output_dirs, read_web_page, write_text_file
---

# Remotion

Remotion renders a React component to video, frame by frame. Use it for designed
motion graphics (text, shapes, charts, titles, edits of existing clips) — not for
photoreal generation, which stays with ComfyUI.

## Needs Node — and the sandbox has none

`run_script` cannot run `npx`/`npm`/`node` (no package manager, by design), so do
not try the Remotion CLI through it. Execution goes through the MCP server below.

## One-time setup

1. `list_mcp_servers` — is a `remotion` server already connected? If yes, skip.
2. If not, and the user wants it: call
   `add_mcp_server(definition='{"command":"npx","args":["-y","remotion-mcp@1.0.5"]}', name="remotion")`.
   The user is asked to approve exactly what will run (it is not sandboxed: it reads
   and writes any path and runs `npm install` plus the TSX you write — say so).
   Then `use_mcp_server("remotion")`.

## Workflow

1. Pick a dedicated project folder under the agent output directory
   (`get_agent_output_dirs`), e.g. `<output>/remotion/<project>`; one project per
   video family. Never reuse a folder holding something else.
2. `init_remotion_project(directory)` once (runs npm install; takes a while).
3. Write the composition with `update_composition` / `create_composition_scaffold`
   (check `list_compositions`, `get_composition_code`). `query_remotion_docs` looks up
   the API instead of guessing it.
4. Look before the full render: `render_still` of a few key frames, `view_image`
   them, fix, then `render_video` (or `render_gif`). Renders are jobs — poll
   `get_render_status(jobId)` until `done`; the result has `outputPath`.
5. Tell the user the file path; the file is in the project's `out` folder, copy or
   reference it from there.

## Composition rules that save retries

- Everything animates from `useCurrentFrame()` + `useVideoConfig()`; use
  `interpolate`/`spring`. No CSS transitions or `setTimeout` — they do not render.
- State fps, width, height, duration in frames up front (30 fps, 1920x1080 default;
  9:16 = 1080x1920). Seconds * fps = frames.
- Compose scenes with `<Sequence from durationInFrames>`; put big type, high
  contrast; keep text inside safe margins.
- Local images/video: put them in the project's `public/` and use `staticFile()`;
  use `<Img>`/`<OffthreadVideo>`, not `<img>`/`<video>`.

## Deeper guidance

Remotion publishes official agent skills (best practices for markup, timing,
transitions, captions, maps, audio, rendering) at github.com/remotion-dev/skills
(`npx skills add remotion-dev/skills`). They are large and meant for a coding
agent with a shell; fetch the one page you need with `read_web_page` (e.g.
`.../skills/remotion-markup/transitions.md`) rather than loading them wholesale.
