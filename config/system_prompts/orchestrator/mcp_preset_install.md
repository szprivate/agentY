The user added the MCP server preset "{name}" (id `{id}`) from Settings > MCP servers > Add from presets. The host has already run the preset's setup recipe deterministically; you are called only for what it could not finish.

What the host did: {done}

The setup report (done = did it, skipped = already in place, needs_user = only the person can, failed = it broke):

{report}

Setup steps for this preset, as text (for a preset without a recipe, do each one that applies to this machine; skip what is already in place):

{steps}

The last connection test said: {test}

Continue from the failed steps: read their message, fix what your tools can (`setup_mcp_preset("{id}")` re-runs the recipe and skips what is done; `find_app_installs` shows the installed apps), then test the server until it reports connected. A step marked needs_user is not yours: tell the user exactly what is left (the report line says it) and stop there. Never write a key into config/mcp.json: keys live in .env and the entry refers to them as ${VAR}. Finish with a short message to the user: connected, or what is still missing.
