"""agentY MCP support — expose tools from configured MCP servers to the orchestrator.

Servers are declared in ``config/mcp.json`` (machine-local, gitignored)::

    {"servers": {
        "magnific": {"enabled": true, "transport": "http",
                     "url": "https://mcp.magnific.com", "auth": "oauth"},
        "example":  {"enabled": false, "transport": "http",
                     "url": "https://example.com/mcp", "auth": "header",
                     "headers": {"Authorization": "Bearer ${EXAMPLE_API_KEY}"}}
    }}

Each server picks a ``transport`` (``http`` streamable / ``sse`` / ``stdio``) and an
``auth`` mode:

* ``none``   — no auth.
* ``header`` — static request headers; ``${ENV_VAR}`` references are expanded from
  the environment (put the secret in ``.env``).
* ``oauth``  — OAuth 2.0 (e.g. Magnific). Tokens persist under ``config/.mcp_tokens/``
  and are reused silently on every start. The interactive browser sign-in is an
  explicit one-time step: ``authorize_server(name)`` (wired to POST
  /agentY/mcp/authorize). Startup NEVER opens a browser — a server that needs
  authorization is skipped and reported as ``needs_auth``.

Connected clients are cached module-level and kept alive for the process — the
orchestrator is built once per :5000 session (see create_orchestrator_agent), so
the tools stay usable across turns. Every failure is contained per-server so a
broken/unauthorized MCP server never blocks orchestrator creation.
"""
from __future__ import annotations

import json
import logging
import os
import re
import threading
import webbrowser
from pathlib import Path

logger = logging.getLogger("agentY.mcp")

_DEFAULT_REDIRECT_PORT = 8199

# name -> live MCPClient (kept alive for the process). name -> status string.
_CLIENTS: dict = {}
_STATUS: dict = {}


class _AuthRequired(Exception):
    """Raised (silently) when an OAuth server has no usable token yet — the user
    must run the interactive authorize flow first."""


# ── config ───────────────────────────────────────────────────────────────────

def _config_path() -> Path:
    return Path(__file__).resolve().parent.parent.parent / "config" / "mcp.json"


def _token_dir() -> Path:
    return Path(__file__).resolve().parent.parent.parent / "config" / ".mcp_tokens"


def load_mcp_config() -> dict:
    """Return the parsed ``config/mcp.json`` (``{"servers": {...}}``), or an empty
    scaffold when the file is missing/unreadable."""
    p = _config_path()
    if not p.exists():
        return {"servers": {}}
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        if not isinstance(data, dict) or not isinstance(data.get("servers"), dict):
            return {"servers": {}}
        return data
    except Exception as exc:  # noqa: BLE001
        logger.warning("mcp: could not read %s: %s", p, exc)
        return {"servers": {}}


def save_mcp_config(cfg: dict) -> None:
    """Persist the MCP config (``{"servers": {...}}``) to ``config/mcp.json``."""
    if not isinstance(cfg, dict):
        raise ValueError("mcp config must be an object")
    servers = cfg.get("servers")
    if not isinstance(servers, dict):
        raise ValueError('mcp config must have a "servers" object')
    p = _config_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"servers": servers}, indent=2), encoding="utf-8")


# ── env expansion for header auth ────────────────────────────────────────────

_ENV_REF = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


def _expand(value: str) -> str:
    """Expand ``${VAR}`` references in *value* from the environment (blank if unset)."""
    return _ENV_REF.sub(lambda m: os.environ.get(m.group(1), ""), str(value))


def _expand_map(d) -> dict:
    return {str(k): _expand(v) for k, v in (d or {}).items()} if isinstance(d, dict) else {}


# ── OAuth token storage (disk) ───────────────────────────────────────────────

def _make_token_storage(name: str):
    from mcp.client.auth import TokenStorage
    from mcp.shared.auth import OAuthClientInformationFull, OAuthToken

    tok_path = _token_dir() / f"{name}.token.json"
    cli_path = _token_dir() / f"{name}.client.json"

    class _DiskTokenStorage(TokenStorage):
        def has_tokens(self) -> bool:
            return tok_path.exists()

        async def get_tokens(self):
            if not tok_path.exists():
                return None
            try:
                return OAuthToken.model_validate_json(tok_path.read_text(encoding="utf-8"))
            except Exception:  # noqa: BLE001
                return None

        async def set_tokens(self, tokens) -> None:
            _token_dir().mkdir(parents=True, exist_ok=True)
            tok_path.write_text(tokens.model_dump_json(), encoding="utf-8")

        async def get_client_info(self):
            if not cli_path.exists():
                return None
            try:
                return OAuthClientInformationFull.model_validate_json(cli_path.read_text(encoding="utf-8"))
            except Exception:  # noqa: BLE001
                return None

        async def set_client_info(self, info) -> None:
            _token_dir().mkdir(parents=True, exist_ok=True)
            cli_path.write_text(info.model_dump_json(), encoding="utf-8")

    return _DiskTokenStorage()


# ── OAuth provider (silent vs interactive) ───────────────────────────────────

def _make_oauth_provider(name: str, url: str, interactive: bool, holder: dict | None,
                         redirect_port: int):
    from mcp.client.auth import OAuthClientProvider
    from mcp.shared.auth import OAuthClientMetadata

    storage = _make_token_storage(name)
    client_metadata = OAuthClientMetadata(
        client_name="agentY",
        redirect_uris=[f"http://localhost:{redirect_port}/callback"],
        grant_types=["authorization_code", "refresh_token"],
        response_types=["code"],
    )

    async def redirect_handler(authorization_url: str) -> None:
        if not interactive:
            raise _AuthRequired()
        logger.info("mcp[%s]: opening browser for OAuth sign-in", name)
        try:
            webbrowser.open(authorization_url)
        except Exception:  # noqa: BLE001
            pass
        print(f"[agentY:mcp] Authorize '{name}' in your browser:\n  {authorization_url}")

    async def callback_handler():
        if not interactive or holder is None:
            raise _AuthRequired()
        import asyncio
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, holder["event"].wait, 300)
        code = holder.get("code")
        if not code:
            raise RuntimeError("OAuth callback timed out or returned no code")
        return code, holder.get("state")

    return OAuthClientProvider(
        server_url=url,
        client_metadata=client_metadata,
        storage=storage,
        redirect_handler=redirect_handler,
        callback_handler=callback_handler,
    )


def _start_callback_server(port: int) -> dict:
    """Start a one-shot local HTTP server that captures the OAuth redirect."""
    from http.server import BaseHTTPRequestHandler, HTTPServer
    from urllib.parse import urlparse, parse_qs

    holder: dict = {"event": threading.Event()}

    class _Handler(BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802
            q = parse_qs(urlparse(self.path).query)
            holder["code"] = (q.get("code") or [None])[0]
            holder["state"] = (q.get("state") or [None])[0]
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            self.wfile.write(b"<h2>agentY: MCP server authorized.</h2>"
                             b"<p>You can close this tab and return to ComfyUI.</p>")
            holder["event"].set()

        def log_message(self, *args):  # silence
            return

    srv = HTTPServer(("127.0.0.1", port), _Handler)
    holder["server"] = srv
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return holder


# ── transport ────────────────────────────────────────────────────────────────

def _transport_callable(sc: dict, provider):
    transport = str(sc.get("transport", "http") or "http").lower()
    if transport in ("http", "streamable-http", "streamable_http"):
        from mcp.client.streamable_http import streamablehttp_client
        url = sc["url"]
        headers = _expand_map(sc.get("headers")) or None
        return lambda: streamablehttp_client(url, headers=headers, auth=provider)
    if transport == "sse":
        from mcp.client.sse import sse_client
        url = sc["url"]
        headers = _expand_map(sc.get("headers")) or None
        return lambda: sse_client(url, headers=headers, auth=provider)
    if transport == "stdio":
        from mcp import StdioServerParameters, stdio_client
        params = StdioServerParameters(
            command=sc["command"], args=list(sc.get("args") or []),
            env=_expand_map(sc.get("env")) or None,
        )
        return lambda: stdio_client(params)
    raise ValueError(f"unknown transport {transport!r}")


def _connect(name: str, sc: dict, interactive: bool):
    """Connect to one server and return ``(client, tools)``. Raises ``_AuthRequired``
    for an OAuth server with no usable token (silent mode)."""
    from strands.tools.mcp import MCPClient

    auth = str(sc.get("auth", "none") or "none").lower()
    provider = None
    holder = None
    port = int(sc.get("redirect_port") or _DEFAULT_REDIRECT_PORT)
    if auth == "oauth":
        if not interactive:
            # Never open a browser at startup: only connect if a token already exists
            # (the provider then refreshes it silently as needed).
            if not _make_token_storage(name).has_tokens():
                raise _AuthRequired()
        else:
            holder = _start_callback_server(port)
        provider = _make_oauth_provider(name, sc["url"], interactive, holder, port)

    try:
        client = MCPClient(_transport_callable(sc, provider), prefix=f"{name}_")
        client.start()
        tools = client.list_tools_sync()
        return client, tools
    finally:
        if holder is not None and holder.get("server") is not None:
            try:
                holder["server"].shutdown()
            except Exception:  # noqa: BLE001
                pass


# ── public API ───────────────────────────────────────────────────────────────

def load_mcp_tools() -> list:
    """Connect every enabled MCP server (silently) and return the combined tool list
    for the orchestrator. Contained per-server: a failing/unauthorized server is
    skipped and recorded in the status map, never raised."""
    tools: list = []
    for name, sc in (load_mcp_config().get("servers") or {}).items():
        if not isinstance(sc, dict) or not sc.get("enabled"):
            _STATUS[name] = "disabled"
            continue
        # Reuse a live client if we already connected this process.
        existing = _CLIENTS.get(name)
        if existing is not None:
            try:
                t = existing.list_tools_sync()
                tools.extend(t)
                _STATUS[name] = f"connected ({len(t)})"
                continue
            except Exception:  # noqa: BLE001
                _CLIENTS.pop(name, None)  # stale — reconnect below
        try:
            client, server_tools = _connect(name, sc, interactive=False)
            _CLIENTS[name] = client
            _STATUS[name] = f"connected ({len(server_tools)})"
            tools.extend(server_tools)
            logger.info("mcp[%s]: %d tool(s) loaded", name, len(server_tools))
        except _AuthRequired:
            _STATUS[name] = "needs_auth"
            logger.warning("mcp[%s]: needs authorization — use Authorize in agentY Settings", name)
        except Exception as exc:  # noqa: BLE001
            _STATUS[name] = f"error: {exc}"
            logger.warning("mcp[%s]: connect failed: %s", name, exc)
    return tools


def authorize_server(name: str) -> dict:
    """Run the interactive OAuth flow for *name* (opens a browser), persisting the
    token. Returns ``{ok, name, tools, message}``. The token is reused silently on
    the next start; the orchestrator picks up the tools when it is next built
    (restart the agent, or start a new one)."""
    servers = load_mcp_config().get("servers") or {}
    sc = servers.get(name)
    if not isinstance(sc, dict):
        return {"ok": False, "error": f"no MCP server named {name!r} in config/mcp.json"}
    if str(sc.get("auth", "none")).lower() != "oauth":
        return {"ok": False, "error": f"server {name!r} does not use oauth (auth={sc.get('auth')!r})"}
    try:
        client, tools = _connect(name, sc, interactive=True)
    except Exception as exc:  # noqa: BLE001
        logger.error("mcp[%s]: authorize failed: %s", name, exc, exc_info=True)
        return {"ok": False, "error": str(exc)}
    # Keep it live so the current process can use it too; the token is now on disk.
    _CLIENTS[name] = client
    _STATUS[name] = f"connected ({len(tools)})"
    return {"ok": True, "name": name, "tools": len(tools),
            "message": (f"Authorized '{name}' — {len(tools)} tool(s). They load into the "
                        "orchestrator when it is next built (restart the agent to use them now).")}


def is_server_connected(name: str) -> bool:
    """True when *name* has a live MCP client in this process."""
    return name in _CLIENTS


def call_mcp_tool(server: str, name: str, arguments: dict | None = None,
                  timeout_s: int = 300) -> dict:
    """Call a tool on a live MCP server from **non-agent** code (e.g. a background
    poller), bypassing the LLM. Returns a plain dict — never raises::

        {"ok": bool, "status": <str|None>, "text": <joined text blocks>,
         "json": <parsed dict/list or None>, "error": <str, on failure>}

    ``ok`` is False when the server isn't connected or the call errored. ``json``
    is the parsed first JSON payload when the response body is JSON.
    """
    import datetime
    import uuid as _uuid

    client = _CLIENTS.get(server)
    if client is None:
        return {"ok": False, "error": f"MCP server {server!r} not connected"}
    try:
        res = client.call_tool_sync(
            tool_use_id=f"bg-{_uuid.uuid4().hex[:8]}",
            name=name,
            arguments=arguments or {},
            read_timeout_seconds=datetime.timedelta(seconds=timeout_s),
        )
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": str(exc)}

    texts: list[str] = []
    for block in (getattr(res, "content", None) or []):
        t = block.get("text") if isinstance(block, dict) else getattr(block, "text", None)
        if t:
            texts.append(str(t))
        elif isinstance(block, dict) and "json" in block:
            texts.append(json.dumps(block["json"], ensure_ascii=False))
    text = "\n".join(texts)
    parsed = None
    stripped = text.strip()
    if stripped[:1] in ("{", "["):
        try:
            parsed = json.loads(stripped)
        except Exception:  # noqa: BLE001
            parsed = None
    status = getattr(res, "status", None)
    return {"ok": status != "error", "status": status, "text": text, "json": parsed}


def mcp_status() -> dict:
    """Per-server status for the settings UI: enabled, auth mode, and connection
    state (connected/needs_auth/disabled/error)."""
    out = {}
    for name, sc in (load_mcp_config().get("servers") or {}).items():
        sc = sc if isinstance(sc, dict) else {}
        state = _STATUS.get(name)
        if state is None:
            if not sc.get("enabled"):
                state = "disabled"
            elif str(sc.get("auth", "none")).lower() == "oauth":
                state = "connected" if name in _CLIENTS else (
                    "authorized (restart to load)" if _make_token_storage(name).has_tokens()
                    else "needs_auth")
            else:
                state = "not connected"
        out[name] = {
            "enabled": bool(sc.get("enabled")),
            "transport": sc.get("transport", "http"),
            "auth": sc.get("auth", "none"),
            "state": state,
        }
    return out
