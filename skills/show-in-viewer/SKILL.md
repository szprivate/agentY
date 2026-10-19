---
name: show-in-viewer
description: Load files into the bEpic image viewer when the user asks to see them there - "show me clip 3 in the viewer", "open these frames side by side", "open the edit", "put the two renders in a tab". Also opens an edit's cut in the viewer's Edit panel.
allowed-tools: show_in_viewer
---

# Showing files in the viewer

`show_in_viewer(paths=[...])` loads files into the bEpic viewer. By default it files them in
the `agentY` tab, brings the viewer up (even if it is closed or docked) and selects the first
item. Use it only when the user asks to look at something in the viewer; generated pictures
reach the viewer by themselves.

- `paths`: absolute paths - images, videos, a folder, an image sequence (`frame_%04d.png`,
  `frame_####.png`, `frame_*.png`) or an `.otio` (it stands for the film beside it).
  Files must lie in ComfyUI's input/output/temp folders or a folder the user allowed in the
  viewer; the answer lists what was `refused`.
- `tab_name`: a new name puts them in a tab of their own - use it for "side by side" or for a
  comparison ("clip 3 vs clip 4"). Leave empty for the agentY tab.
- `group`, `label`: a heading and a caption in the history strip, when it helps tell them apart.
- `focus`: default true. Pass false only to load quietly.
- `play`: start playing the first video.
- `panel="edit"`: open the viewer's Edit panel on the cut, so the user sees the edit shot by shot.
  The film needs an `.otio` beside it (an `agentY edit` render writes one). "Open the edit" means
  this: pass the edit's film (or its `.otio`) with `panel="edit"`.

An edit stage's render already opens itself this way when it finishes (Settings, `media_nodes.viewer_open_edit`),
so do not call the tool again right after a render unless the user asks.

If the answer has an `error` or `refused` paths, say so plainly; do not retry with other paths.
