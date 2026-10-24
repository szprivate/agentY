---
name: mcp-recipes
description: Install and set up an MCP server completely and deterministically - a catalog preset in one call, or a server that is not in the catalog through a recipe you draft from its README and the user approves. Activate when the user asks to "install / add / connect the X MCP", or to re-check one.
allowed-tools: list_mcp_presets, install_mcp_preset, setup_mcp_preset, draft_mcp_recipe, install_mcp_bundle, find_app_installs, list_mcp_servers, use_mcp_server, add_mcp_server, read_web_page, web_search
---

# MCP presets and recipes

A **preset** is an MCP server plus a **recipe**: a list of typed steps that set it up (unpack a bundle, install an
add-on into Blender / Houdini / Maya / Nuke / ..., edit an app's preferences, test the connection). Every step
looks first and skips what is done, and reports `done` / `skipped` / `needs_user` / `failed`. The same runner
serves the installer, the Settings buttons and you.

## A preset in the catalog

1. `list_mcp_presets` - is it there, installed, which `.env` keys does it need?
2. `install_mcp_preset(preset_id)` - the whole thing. Read `report_text`.
3. `needs_user` lists what only the person can do: start the app, press a button in it, add a key in Settings >
   MCP servers, confirm a licence. Say it in plain words; do not paste the report.
4. `failed` steps: read the message, fix what your tools can, else tell the user. Later `setup_mcp_preset(id)`
   re-checks everything (safe to repeat; `only_steps="addon,test"` runs part).
5. System-wide installs (uv, Node, git through winget) and starting an app are off. Pass
   `allow_system_installs=true` / `launch_app=true` only when the user said yes to that.
6. When it connected, the server's tools are in your list now (`<id>__<tool>`).

Unity needs `project=<folder>`; ask which project. A missing key comes back as `needs_user` naming the `.env`
key: the user enters it in Settings; you never write it.

## A server that is not in the catalog

1. Find it: the user gives a GitHub URL, or a name (search the web, read the page). Prefer a published, pinned release.
2. `draft_mcp_recipe(source)` returns a `draft`, its `validation` and the README. The draft covers the server entry
   (a PyPI package -> `uvx pkg==x.y.z`; npm -> `npx -y pkg@x.y.z`; a repo -> clone at a commit + `uv sync`).
3. Read the README for what the draft cannot know and add steps for it:
   - needs an app? `find_app` (app, `min_version`) first;
   - an add-on or plug-in for the app? `install_addon` (apps: blender, houdini, maya, nuke, cavalry, premiere-pro,
     after-effects, unity) or `edit_file` (`marker_block` / `json_merge` / `regex_set`, backed up, idempotent);
   - a download? `download` with `sha256` (compute it from the file you fetched); a repo? `git_clone` with the full commit;
   - a key? `env_keys: [{"key": "X_API_KEY", "label": "...", "help_url": "..."}]` and `${X_API_KEY}` in `server.env`;
   - a user-scope variable? `set_env`; a command? `run` (argument list, no shell string);
   - a human action (open a menu, sign in, a licence)? `manual` with the exact words;
   - only when nothing typed fits: `run_script` (interpreter + the whole script + a description). It is shown to the
     user in full and flagged.
   - end with `test`; add `needs` so a step waits for the ones it depends on.
4. Worked examples: `list_mcp_presets(include_recipe=true)` - blender (bundle + add-on + prefs), nuke (pinned
   clone + `~/.nuke` edit), premiere-pro (npm fetch + panel + registry flag), davinci-resolve (env vars, manual
   Studio check), cavalry (clone + bridge script + a manual menu click).
5. Check it: `install_mcp_preset(recipe=<json>, dry_run=true)` returns the validation and the step list.
6. Show the user the step list (each `run_script` in full, say what runs and where), then END YOUR TURN.
   Call `install_mcp_preset(recipe=<same json>, user_message=<their words>)` only after they said yes in their own
   words. A changed recipe is a new recipe and needs a new yes. Never approve for them.
7. An approved recipe is saved as a local preset (`config/mcp_presets.local.json`) and is from then on treated like
   a catalog preset: `setup_mcp_preset` re-checks it.

Rules: pin versions and commits; never put a key in the recipe; never `npm -g`; prefer the package's own installer
over rewriting an app's files by hand; say what you could not verify.

`find_app_installs(["blender", ...])` tells which apps are installed and which version is the newest.
