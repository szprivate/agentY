The user added the MCP server preset "{name}" (id `{id}`) from Settings > MCP servers > Add from presets. Install it and finish its setup, then tell the user the result.

What the host already did: {done}

Setup steps for this preset (do each one that applies to this machine; skip what is already in place):

{steps}

The last connection test said: {test}

Then test the server (the MCP settings test, or the server's own tools) until it reports connected. If a step needs the user (a browser sign-in, an app only they can open, a key they have not given), say exactly what is left and stop there. Never write a key into config/mcp.json: keys live in .env and the entry refers to them as ${VAR}. Finish with a short message to the user: connected, or what is still missing.
