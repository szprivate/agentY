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

  Two things have to be carried across process restarts for the silent path to
  actually work, because the MCP client only holds them in memory: the token's
  **expiry** (without it a stale token looks valid forever, so the refresh is
  never attempted) and the **discovered auth endpoints** (the refresh runs before
  discovery, so without them it POSTs to ``<mcp-url>/token`` — wrong whenever the
  authorization server is a separate host). Both are cached next to the token, so
  an expired access token is renewed silently instead of demanding a new sign-in.

Connected clients are cached module-level and kept alive for the process — the
orchestrator is built once per :5000 session (see create_orchestrator_agent), so
the tools stay usable across turns. Every failure is contained per-server so a
broken/unauthorized MCP server never blocks orchestrator creation.
"""
from __future__ import annotations

import contextlib
import json
import logging
import os
import re
import threading
import time
import warnings
import webbrowser
from pathlib import Path

logger = logging.getLogger("agentY.mcp")

_DEFAULT_REDIRECT_PORT = 8199
# Refresh a stored token this many seconds before it actually expires.
_EXPIRY_SKEW_S = 60

# name -> live MCPClient (kept alive for the process). name -> status string.
_CLIENTS: dict = {}
_STATUS: dict = {}

# strands' MCPClient.stop() schedules a coroutine onto the background loop even
# when start() never got one running, so a failed connect prints a RuntimeWarning
# about an un-awaited coroutine that has nothing to do with agentY.
warnings.filterwarnings(
    "ignore",
    message=r"coroutine 'MCPClient\.stop\.<locals>\._set_close_event' was never awaited",
    category=RuntimeWarning,
)


class _AuthRequired(Exception):
    """Raised (silently) when an OAuth server has no usable token yet — the user
    must run the interactive authorize flow first."""


def _contains_auth_required(exc: BaseException | None, _depth: int = 0) -> bool:
    """True when *exc* is, or wraps, an :class:`_AuthRequired`.

    It is raised inside the MCP client's httpx auth flow, which runs on a strands
    background thread inside an anyio task group — by the time it reaches the
    caller it is buried under an ExceptionGroup and a
    MCPClientInitializationError, so a plain ``except _AuthRequired`` never fires.
    """
    if exc is None or _depth > 12:
        return False
    if isinstance(exc, _AuthRequired):
        return True
    for sub in getattr(exc, "exceptions", None) or ():
        if isinstance(sub, BaseException) and _contains_auth_required(sub, _depth + 1):
            return True
    return (_contains_auth_required(exc.__cause__, _depth + 1)
            or _contains_auth_required(exc.__context__, _depth + 1))


@contextlib.contextmanager
def _quiet_auth_required():
    """Drop the multi-frame tracebacks the MCP/strands stacks log when a *silent*
    connect stops at ``_AuthRequired``. That is an expected outcome (the user has
    not authorized yet), not a crash; anything else those loggers emit is left
    alone."""

    class _Filter(logging.Filter):
        def filter(self, record):  # noqa: A003
            return not _contains_auth_required((record.exc_info or (None, None, None))[1])

    flt = _Filter()
    loggers = [logging.getLogger(n) for n in
               ("mcp.client.auth.oauth2", "strands.tools.mcp.mcp_client")]
    for lg in loggers:
        lg.addFilter(flt)
    try:
        yield
    finally:
        for lg in loggers:
            lg.removeFilter(flt)


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

        def _raw(self) -> dict:
            try:
                data = json.loads(tok_path.read_text(encoding="utf-8"))
                return data if isinstance(data, dict) else {}
            except Exception:  # noqa: BLE001
                return {}

        def expires_at(self) -> float | None:
            """Absolute unix time the stored access token expires, or None when
            unknown. ``expires_at`` is agentY's own key (OAuthToken ignores extra
            fields); tokens written before it existed fall back to the file's
            mtime — which is when the token was received — plus its lifetime."""
            raw = self._raw()
            exp = raw.get("expires_at")
            if isinstance(exp, (int, float)):
                return float(exp)
            ttl = raw.get("expires_in")
            if isinstance(ttl, (int, float)):
                try:
                    return tok_path.stat().st_mtime + float(ttl)
                except OSError:
                    return None
            return None

        async def get_tokens(self):
            if not tok_path.exists():
                return None
            try:
                return OAuthToken.model_validate_json(tok_path.read_text(encoding="utf-8"))
            except Exception:  # noqa: BLE001
                return None

        async def set_tokens(self, tokens) -> None:
            _token_dir().mkdir(parents=True, exist_ok=True)
            previous = self._raw()
            raw = json.loads(tokens.model_dump_json())
            # RFC 6749 §6: a refresh response may omit refresh_token, meaning
            # "keep the one you have" — don't let a refresh strand the next start.
            if not raw.get("refresh_token") and previous.get("refresh_token"):
                raw["refresh_token"] = previous["refresh_token"]
            if isinstance(raw.get("expires_in"), (int, float)):
                raw["expires_at"] = time.time() + float(raw["expires_in"])
            tok_path.write_text(json.dumps(raw), encoding="utf-8")

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


# ── OAuth discovery cache ────────────────────────────────────────────────────

def _discovery_path(name: str) -> Path:
    return _token_dir() / f"{name}.discovery.json"


def _seed_discovery(name: str, provider) -> None:
    """Restore the OAuth server metadata discovered on an earlier run.

    ``async_auth_flow`` attempts the token refresh *before* it discovers anything,
    so on a fresh process it POSTs to ``<mcp-server-url>/token`` — which for a
    server whose authorization lives on another host (Magnific: ``auth.magnific
    .com``) is a 404. The refresh then "fails" and the client falls through to a
    full browser re-authorization. Seeding the cache points that first refresh at
    the real token endpoint."""
    from mcp.shared.auth import OAuthMetadata, ProtectedResourceMetadata

    path = _discovery_path(name)
    if not path.exists():
        return
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        if raw.get("oauth_metadata"):
            provider.context.oauth_metadata = OAuthMetadata.model_validate(raw["oauth_metadata"])
        if raw.get("protected_resource_metadata"):
            provider.context.protected_resource_metadata = ProtectedResourceMetadata.model_validate(
                raw["protected_resource_metadata"])
        if raw.get("auth_server_url"):
            provider.context.auth_server_url = str(raw["auth_server_url"])
    except Exception as exc:  # noqa: BLE001
        logger.debug("mcp[%s]: ignoring unusable discovery cache: %s", name, exc)


def _save_discovery(name: str, provider) -> None:
    """Cache what the provider discovered — including after a *failed* connect. A
    run that stops at ``needs_auth`` still learned the endpoints on its way there,
    and that is exactly what lets the next start refresh silently."""
    ctx = getattr(provider, "context", None)
    if ctx is None or getattr(ctx, "oauth_metadata", None) is None:
        return
    try:
        payload = {
            "oauth_metadata": json.loads(ctx.oauth_metadata.model_dump_json(exclude_none=True)),
            "auth_server_url": ctx.auth_server_url,
        }
        if ctx.protected_resource_metadata is not None:
            payload["protected_resource_metadata"] = json.loads(
                ctx.protected_resource_metadata.model_dump_json(exclude_none=True))
        _token_dir().mkdir(parents=True, exist_ok=True)
        _discovery_path(name).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    except Exception as exc:  # noqa: BLE001
        logger.debug("mcp[%s]: could not cache OAuth discovery: %s", name, exc)


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

    provider = OAuthClientProvider(
        server_url=url,
        client_metadata=client_metadata,
        storage=storage,
        redirect_handler=redirect_handler,
        callback_handler=callback_handler,
    )

    _seed_discovery(name, provider)

    # The provider only learns a token's expiry when it *receives* one, so a token
    # read back from disk looks valid forever: the stale bearer goes out, the
    # server 401s, and the client jumps straight to a full (browser)
    # re-authorization — never touching the refresh token it has. Seed the expiry
    # from disk so an expired access token is refreshed silently instead.
    try:
        expires_at = storage.expires_at()
        if expires_at is not None:
            provider.context.token_expiry_time = expires_at - _EXPIRY_SKEW_S
    except Exception as exc:  # noqa: BLE001
        logger.debug("mcp[%s]: could not seed token expiry: %s", name, exc)
    return provider


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

    quiet = contextlib.nullcontext() if interactive else _quiet_auth_required()
    try:
        client = MCPClient(_transport_callable(sc, provider), prefix=f"{name}_")
        with quiet:
            client.start()
            tools = client.list_tools_sync()
        return client, tools
    except Exception as exc:  # noqa: BLE001
        # Surfaces as an ExceptionGroup from the strands background thread — turn
        # it back into the clean signal callers branch on.
        if _contains_auth_required(exc):
            raise _AuthRequired() from None
        raise
    finally:
        if provider is not None:
            _save_discovery(name, provider)
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
            stale = (str(sc.get("auth", "none")).lower() == "oauth"
                     and _make_token_storage(name).has_tokens())
            logger.warning(
                "mcp[%s]: %s — use Authorize in agentY Settings", name,
                "stored token expired and could not be refreshed" if stale
                else "needs authorization")
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
    previous = _CLIENTS.get(name)
    _CLIENTS[name] = client
    if previous is not None and previous is not client:
        try:
            previous.stop(None, None, None)  # drop the client this one replaces
        except Exception:  # noqa: BLE001
            pass
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

    # call_tool_sync returns a ToolResult **dict** ({"status", "content":[{"text"|
    # "json"}], "isError"}) in this Strands version; guard for an object too.
    def _field(obj, key):
        return obj.get(key) if isinstance(obj, dict) else getattr(obj, key, None)

    texts: list[str] = []
    for block in (_field(res, "content") or []):
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
    status = _field(res, "status")
    is_error = bool(_field(res, "isError"))
    return {"ok": status != "error" and not is_error, "status": status,
            "text": text, "json": parsed}


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
