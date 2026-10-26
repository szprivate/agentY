"""agentY MCP support — expose tools from configured MCP servers to the orchestrator.

Servers are declared in ``config/mcp.json`` (tracked, and shared between
machines — so no machine's paths belong in it)::

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

Every server also takes ``timeout_s`` (per-call limit, default 600), ``tool_timeouts``
(``{"render_*": 1800}``) and ``guards`` (rules that refuse a call; see mcp_resilience).

``${VAR}`` expands in a stdio server's ``command``, ``args``, ``cwd`` and ``env`` too.
When the environment doesn't set one, a few are worked out on this machine
(``_BUILTINS``): ``${PYTHON}`` (agentY's own interpreter), ``${AGENTY_ROOT}``,
``${COMFYUI_URL}`` and ``${BEPIC_WORLDS_DIR}`` (the bEpic Worlds pack, found in the
running ComfyUI's custom_nodes). A relative ``cwd`` is relative to agentY's folder,
and a bundle whose recorded ``cwd`` isn't on this machine runs from its own
``bundle.dir``. A server whose folder can't be found is skipped with a note saying
which variable to set — never a crash, and never another machine's ``D:/…``.
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
import shutil
import sys
import threading
import time
import warnings
import webbrowser
from pathlib import Path

from strands import tool

logger = logging.getLogger("agentY.mcp")

_DEFAULT_REDIRECT_PORT = 8199
# Refresh a stored token this many seconds before it actually expires.
_EXPIRY_SKEW_S = 60

# name -> live MCPClient (kept alive for the process). name -> status string.
_CLIENTS: dict = {}
_STATUS: dict = {}
# name -> (id of its client, its tool list): listing is a round trip to the
# server, and mcp_on_demand loads a server into an agent from this list.
_LISTS: dict = {}

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
    # A bundle whose server was removed takes its unpacked files with it.
    try:
        from src.utils.mcp_bundle import prune_unreferenced
        removed = prune_unreferenced(servers)
        if removed:
            logger.info("mcp: removed unused bundle(s): %s", ", ".join(removed))
    except Exception as exc:  # noqa: BLE001 — cleanup must never fail a save
        logger.warning("mcp: could not prune unused bundles: %s", exc)


# ── env expansion for header auth ────────────────────────────────────────────

_ENV_REF = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


_ROOT = Path(__file__).resolve().parent.parent.parent


def _comfy_custom_nodes() -> list[Path]:
    """ComfyUI's custom_nodes folders on this machine: $COMFYUI_PATH's, then
    what the running ComfyUI reports (it knows extra folders, and it is the only
    one that knows where it was installed)."""
    found: list[Path] = []
    for var in ("COMFYUI_PATH", "COMFYUI_DIR"):
        if os.environ.get(var):
            found.append(Path(os.environ[var]) / "custom_nodes")
    try:
        from agenty_core.utils.comfyui_client import get_client
        paths = (get_client().get("/internal/folder_paths") or {}).get("custom_nodes") or []
        found += [Path(p) for p in paths if isinstance(p, str)]
    except Exception:  # noqa: BLE001 — ComfyUI down or remote: nothing to add
        pass
    return [p for p in found if p.is_dir()]


def _find_custom_node(marker: str) -> str:
    """The custom node folder holding *marker* (a file inside it), whatever the
    folder itself is called on this machine; '' when none does."""
    for base in _comfy_custom_nodes():
        try:
            for child in sorted(base.iterdir()):
                if (child / marker).is_file():
                    return str(child)
        except OSError:
            continue
    return ""


def _comfy_url() -> str:
    try:
        from agenty_core.utils.comfyui_client import get_client
        return str(get_client().base_url)
    except Exception:  # noqa: BLE001
        return ""


# Variables worked out on this machine when the environment doesn't set them.
_BUILTINS = {
    "PYTHON": lambda: sys.executable,
    "AGENTY_ROOT": lambda: str(_ROOT),
    "COMFYUI_URL": _comfy_url,
    "BEPIC_WORLDS_DIR": lambda: _find_custom_node(os.path.join("bepic_worlds", "mcp_server.py")),
}
_BUILTIN_CACHE: dict[str, str] = {}


def _var(name: str) -> str:
    if os.environ.get(name):
        return os.environ[name]
    if name in _BUILTINS:
        if not _BUILTIN_CACHE.get(name):
            _BUILTIN_CACHE[name] = _BUILTINS[name]() or ""
        return _BUILTIN_CACHE[name]
    return ""


def _expand(value: str) -> str:
    """Expand ``${VAR}`` references in *value* from the environment, else from
    ``_BUILTINS`` (blank if neither knows it)."""
    return _ENV_REF.sub(lambda m: _var(m.group(1)), str(value))


def _stdio_cwd(sc: dict) -> str | None:
    """Where a stdio server runs, on THIS machine — or a clear error naming what
    to set. See the module docstring."""
    raw = str(sc.get("cwd") or "")
    if not raw:
        return None
    cwd = _expand(raw)
    if cwd and not os.path.isabs(cwd):
        cwd = str(_ROOT / cwd)
    if cwd and os.path.isdir(cwd):
        return cwd
    bundle_dir = (sc.get("bundle") or {}).get("dir")
    if bundle_dir and (_ROOT / bundle_dir).is_dir():
        return str(_ROOT / bundle_dir)
    unset = [m.group(1) for m in _ENV_REF.finditer(raw) if not _var(m.group(1))]
    if unset:
        hint = (" — the bEpic Worlds pack isn't in this machine's ComfyUI custom_nodes (or ComfyUI isn't "
                "running); set BEPIC_WORLDS_DIR to its folder" if "BEPIC_WORLDS_DIR" in unset else "")
        raise FileNotFoundError(f"its folder {raw} needs {', '.join(unset)}{hint}")
    raise FileNotFoundError(f"its folder {cwd} isn't on this machine")


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

def _stdio_env(env: dict) -> dict:
    from scripts import tool_hints
    return tool_hints.stdio_env(env)


def _which_tool(command: str):
    """``shutil.which`` over the PATH plus the usual node / uv folders (see scripts/tool_hints.py)."""
    from scripts import tool_hints
    return shutil.which(command, path=tool_hints.augmented_path())


def _transport_callable(sc: dict, provider, errlog=None):
    transport = str(sc.get("transport", "http") or "http").lower()
    if transport in ("http", "streamable-http", "streamable_http"):
        from mcp.client.streamable_http import streamablehttp_client
        url = _expand(sc["url"])   # ${AYON_SERVER_URL}/api/mcp and the like
        headers = _expand_map(sc.get("headers")) or None
        return lambda: streamablehttp_client(url, headers=headers, auth=provider)
    if transport == "sse":
        from mcp.client.sse import sse_client
        url = _expand(sc["url"])   # ${AYON_SERVER_URL}/api/mcp and the like
        headers = _expand_map(sc.get("headers")) or None
        return lambda: sse_client(url, headers=headers, auth=provider)
    if transport == "stdio":
        from mcp import StdioServerParameters, stdio_client
        # ${VAR} expands in the arguments too: an installed bundle can put a key
        # there (``--token=${MCP_X_TOKEN}``). ``cwd`` is the bundle's own folder.
        params = StdioServerParameters(
            command=_expand(sc["command"]), args=[_expand(a) for a in (sc.get("args") or [])],
            # A process started from the Finder / ComfyUI has no login-shell PATH on macOS: add Homebrew,
            # nvm, Volta and ~/.local/bin so npx / uvx are found.  (scripts/tool_hints.py)
            env=_stdio_env(_expand_map(sc.get("env"))),
            cwd=_stdio_cwd(sc),
        )
        if errlog is not None:   # the server's own stderr goes to its log, not the console
            return lambda: stdio_client(params, errlog=errlog)
        return lambda: stdio_client(params)
    raise ValueError(f"unknown transport {transport!r}")


def _harden_client(name: str, client) -> None:
    """Timeouts, reconnect-on-dead-session and guards for every call (mcp_resilience)."""
    from src.tools import mcp_resilience

    def get_config() -> dict:
        sc = (load_mcp_config().get("servers") or {}).get(name)
        return sc if isinstance(sc, dict) else {}

    mcp_resilience.harden(name, client, get_config,
                          lambda text: _STATUS.__setitem__(name, text))


# ── start-up failures: one friendly line, the detail in a log file ───────────

_CONNECT_LOCK = threading.RLock()
_ATTEMPT_OFFSET: dict[str, int] = {}
_REASONS: dict[str, str] = {}
_NO_ENV_CHECK = {"PYTHON", "AGENTY_ROOT", "COMFYUI_URL"}
_VAR_HINTS = {
    "BEPIC_WORLDS_DIR": "the bEpic Worlds pack isn't in this machine's ComfyUI custom_nodes "
                        "(or ComfyUI isn't running)",
}
_COMMAND_HINTS = {
    "uv": "install uv (https://docs.astral.sh/uv/)", "uvx": "install uv (https://docs.astral.sh/uv/)",
    "npx": "install Node.js (https://nodejs.org)", "npm": "install Node.js (https://nodejs.org)",
    "node": "install Node.js (https://nodejs.org)",
}


class McpUnavailable(Exception):
    """A server that cannot start on this machine, with the plain reason why."""


def mcp_log_path(name: str) -> Path:
    return _ROOT / ".logs" / f"mcp_{re.sub(r'[^A-Za-z0-9_.-]', '_', str(name))}.log"


def _log_rel(name: str) -> str:
    return f".logs/{mcp_log_path(name).name}"


def _write_log(name: str, text: str) -> None:
    try:
        path = mcp_log_path(name)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8", errors="replace") as f:
            f.write(text if text.endswith("\n") else text + "\n")
    except OSError:
        pass


def _open_errlog(name: str):
    """The file a stdio server's stderr is written to; marks where this attempt starts."""
    try:
        path = mcp_log_path(name)
        path.parent.mkdir(parents=True, exist_ok=True)
        f = open(path, "a", encoding="utf-8", errors="replace")
        f.write(f"\n=== {time.strftime('%Y-%m-%d %H:%M:%S')} starting MCP server {name} ===\n")
        f.flush()
        _ATTEMPT_OFFSET[name] = os.fstat(f.fileno()).st_size
        return f
    except OSError:
        return None


def _log_tail(name: str) -> str:
    """What the server wrote to stderr at the start of the latest attempt."""
    try:
        with open(mcp_log_path(name), "rb") as f:
            f.seek(_ATTEMPT_OFFSET.get(name, 0))
            return f.read(6000).decode("utf-8", "replace")
    except OSError:
        return ""


@contextlib.contextmanager
def _capture_client_logs(name: str):
    """strands logs a full traceback on a failed start-up; send it to the server's log."""
    lg = logging.getLogger("strands.tools.mcp.mcp_client")
    with _CONNECT_LOCK:
        try:
            fh = logging.FileHandler(mcp_log_path(name), encoding="utf-8")
        except OSError:
            fh = None
        saved = (lg.handlers[:], lg.propagate)
        if fh is not None:
            lg.handlers, lg.propagate = [fh], False
        try:
            yield
        finally:
            lg.handlers, lg.propagate = saved
            if fh is not None:
                fh.close()


def _preset_env_keys(name: str) -> list[tuple[str, bool]]:
    try:
        from scripts import mcp_presets as mp
        preset = mp.find(name)
    except Exception:  # noqa: BLE001
        return []
    return [(str(k["key"]), bool(k.get("optional"))) for k in (preset or {}).get("env_keys") or []]


def _missing_env(name: str, sc: dict) -> list[str]:
    """Keys the server needs that nothing on this machine sets: the preset's
    required ``env_keys`` plus every ``${VAR}`` in the entry (cwd excepted: its
    own check names the folder)."""
    keys = _preset_env_keys(name)
    optional = {k for k, opt in keys if opt}
    wanted = [k for k, opt in keys if not opt]
    env = sc.get("env") if isinstance(sc.get("env"), dict) else {}
    headers = sc.get("headers") if isinstance(sc.get("headers"), dict) else {}
    parts = [sc.get("command"), sc.get("url"), *(sc.get("args") or []), *env.values(), *headers.values()]
    for part in parts:
        wanted += [m.group(1) for m in _ENV_REF.finditer(str(part or ""))]
    out: list[str] = []
    for var in wanted:
        if var in out or var in optional or var in _NO_ENV_CHECK:
            continue
        if not _var(var):
            out.append(var)
    return out


def _check_startable(name: str, sc: dict) -> None:
    """Refuse to start a server that cannot work here, before spawning anything."""
    missing = _missing_env(name, sc)
    if missing:
        hints = "; ".join(_VAR_HINTS[v] for v in missing if v in _VAR_HINTS)
        raise McpUnavailable(f"needs {', '.join(missing)} in .env" + (f" ({hints})" if hints else ""))
    if str(sc.get("transport", "http") or "http").lower() == "stdio":
        command = _expand(sc.get("command") or "")
        if command and _which_tool(command) is None:
            base = os.path.splitext(os.path.basename(command))[0].lower()
            hint = _COMMAND_HINTS.get(base)
            raise McpUnavailable(f"command '{command}' not found" + (f" - {hint}" if hint else ""))


_VARNAME = r"[A-Z][A-Z0-9]*(?:_[A-Z0-9]+)+"
_ENV_LINE = re.compile(rf"\b({_VARNAME})\b[^\n]{{0,60}}?(?:is required|required|not set|is missing|missing)"
                       rf"|(?:missing|set|requires?)\s+(?:the\s+)?({_VARNAME})\b")


def _friendly_reason(name: str, sc: dict, exc: BaseException) -> str:
    """A short plain-language cause for a failed start-up (never a traceback)."""
    if isinstance(exc, McpUnavailable):
        return str(exc)
    sc = sc if isinstance(sc, dict) else {}
    leaf = _leaf_error(exc)
    tail = _log_tail(name)
    text = f"{leaf}\n{exc}\n{tail}"
    if isinstance(exc, FileNotFoundError) and "its folder" in str(exc):
        return " ".join(str(exc).split())[:200]
    m = _ENV_LINE.search(tail) or _ENV_LINE.search(text)
    if m:
        return f"needs {m.group(1) or m.group(2)} in .env"
    cmd = _expand(sc.get("command") or "")
    if isinstance(exc, FileNotFoundError) or re.search(r"WinError 2|No such file or directory|command not found", text):
        base = os.path.splitext(os.path.basename(cmd))[0].lower()
        hint = _COMMAND_HINTS.get(base)
        return f"command '{cmd or '?'}' not found" + (f" - {hint}" if hint else "")
    if re.search(r"did not start in|timed out|TimeoutError|Timeout", text):
        return "no answer in time (the first start can be slow while its package downloads - try again)"
    if re.search(r"ModuleNotFoundError|No module named|Cannot find module|ERR_MODULE_NOT_FOUND|E404|"
                 r"No solution found|Failed to (?:download|fetch|resolve)|not found in the package registry", text):
        return "its package isn't installed or couldn't be downloaded"
    if re.search(r"Connection closed|connection refused|ConnectError|All connection attempts failed", text, re.I):
        stdio = str(sc.get("transport", "http")).lower() == "stdio"
        return "the server exited right after starting" if stdio else "the server is not reachable"
    return " ".join(leaf.split())[:160]


def _connect(name: str, sc: dict, interactive: bool):
    """Connect to one server and return ``(client, tools)``. Raises ``_AuthRequired``
    for an OAuth server with no usable token (silent mode), ``McpUnavailable`` for one
    that cannot start here (missing key / command). Other failures are logged in full
    to ``.logs/mcp_<name>.log`` and re-raised for ``_friendly_reason`` to summarise."""
    from strands.tools.mcp import MCPClient

    _check_startable(name, sc)
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
    is_stdio = str(sc.get("transport", "http") or "http").lower() == "stdio"
    errlog = _open_errlog(name) if is_stdio else None
    try:
        with _capture_client_logs(name):
            client = MCPClient(_transport_callable(sc, provider, errlog), prefix=f"{name}_")
            with quiet:
                client.start()
                tools = client.list_tools_sync()
        _harden_client(name, client)
        return client, tools
    except Exception as exc:  # noqa: BLE001
        # Surfaces as an ExceptionGroup from the strands background thread — turn
        # it back into the clean signal callers branch on.
        if _contains_auth_required(exc):
            raise _AuthRequired() from None
        import traceback
        _write_log(name, "".join(traceback.format_exception(exc)))
        if errlog is not None:
            errlog.close()
            errlog = None
        raise
    finally:
        if errlog is not None:   # connected: the server process keeps writing to it
            try:
                errlog.flush()
            except OSError:
                pass
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
                _LISTS[name] = (id(existing), list(t))
                _STATUS[name] = f"connected ({len(t)})"
                continue
            except Exception:  # noqa: BLE001
                _CLIENTS.pop(name, None)  # stale — reconnect below
        try:
            client, server_tools = _connect(name, sc, interactive=False)
            _CLIENTS[name] = client
            _STATUS[name] = f"connected ({len(server_tools)})"
            _LISTS[name] = (id(client), list(server_tools))
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
            reason = _friendly_reason(name, sc, exc)
            if isinstance(exc, McpUnavailable):
                _write_log(name, f"{time.strftime('%Y-%m-%d %H:%M:%S')} not started: {reason}")
            _REASONS[name] = reason
            _STATUS[name] = f"error: {reason}"
            logger.warning("MCP %s: not connected — %s (details: %s)", name, reason, _log_rel(name))
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


def _tool_names(server: str, tools) -> list:
    """A server's own tool names. MCPClient(prefix="<name>_") hands them back as
    "<name>__<tool>" (strands adds its own separator), which is how they appear in
    an agent's tool list but not what the server calls them."""
    names = []
    for tool_obj in tools or ():
        name = str(getattr(tool_obj, "tool_name", "") or getattr(tool_obj, "name", ""))
        if name.startswith(server):
            name = name[len(server):].lstrip("_") or name
        names.append(name)
    return names


def _leaf_error(exc: BaseException, _depth: int = 0) -> str:
    """The message worth showing from an exception the MCP stack has wrapped in
    exception groups and initialization errors."""
    if _depth < 12:
        for sub in getattr(exc, "exceptions", None) or ():
            if isinstance(sub, BaseException):
                return _leaf_error(sub, _depth + 1)
        cause = exc.__cause__ or exc.__context__
        if cause is not None:
            return _leaf_error(cause, _depth + 1)
    return (str(exc).strip() or type(exc).__name__)[:300]


_WANTS_CREDENTIALS = re.compile(r"\b40[13]\b|unauthori[sz]ed|forbidden|invalid[_ ]token|"
                                r"authenticat", re.I)


def test_server(name: str, sc: dict, secrets: dict | None = None) -> dict:
    """Connect to one server as the settings form describes it (saved or not),
    list its tools, and disconnect.

    ``secrets`` are values typed into the form but not saved yet; they are in the
    environment only while the connection is made, so a ``${VAR}`` reference
    expands to them. The live clients the orchestrator uses are not touched. OAuth
    never opens a browser here: a server with no stored token answers
    ``needs_auth`` and the form points at Authorize. Returns ``{ok, tools, names}``
    or ``{ok: False, error, needs_auth}``.
    """
    if not isinstance(sc, dict):
        return {"ok": False, "error": "no server settings to test", "needs_auth": False}
    overlay = {str(k): str(v) for k, v in (secrets or {}).items() if v not in (None, "")}
    previous = {k: os.environ.get(k) for k in overlay}
    client = None
    try:
        os.environ.update(overlay)
        client, tools = _connect(name, sc, interactive=False)
        return {"ok": True, "tools": len(tools), "names": _tool_names(name, tools)[:50]}
    except _AuthRequired:
        return {"ok": False, "needs_auth": True,
                "error": "This server signs in through the browser."}
    except Exception as exc:  # noqa: BLE001
        message = _friendly_reason(name, sc, exc)
        return {"ok": False, "error": message,
                "needs_auth": bool(_WANTS_CREDENTIALS.search(message))}
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        if client is not None:
            try:
                client.stop(None, None, None)
            except Exception:  # noqa: BLE001
                pass


_STATE_WORDS = {
    "disabled": "switched off in agentY Settings ▸ MCP servers",
    "needs_auth": "needs a browser sign-in — agentY Settings ▸ MCP servers ▸ Authorize…",
    "not connected": "configured, but its tools are not loaded in this process "
                     "(restart the agent to load them)",
    "authorized (restart to load)": "signed in; its tools load when the agent is next started",
}


def mcp_server_report() -> str:
    """What the MCP servers on this machine are, and whether their tools are usable.

    The plain function behind :func:`list_mcp_servers`; the tool is the agent's way
    in, this is the testable one.
    """
    servers = load_mcp_config().get("servers") or {}
    if not servers:
        return ("No MCP servers are configured. They are added in agentY Settings ▸ "
                "MCP servers (paste a server's address, its start command or its JSON "
                "config, or install a .mcpb bundle).")
    try:
        status = mcp_status()
    except Exception:  # noqa: BLE001
        status = {}
    lines = []
    for name, sc in servers.items():
        sc = sc if isinstance(sc, dict) else {}
        info = status.get(name) or {}
        state = str(info.get("state") or "")
        transport = str(sc.get("transport") or info.get("transport") or "http").lower()
        how = "local command" if transport == "stdio" else transport
        auth = str(sc.get("auth") or info.get("auth") or "none").lower()
        sign_in = {"oauth": "browser sign-in", "header": "API key"}.get(auth, "no sign-in")
        said = _STATE_WORDS.get(state)
        if said is None and state.startswith("connected"):
            said = "connected"
            client = _CLIENTS.get(name)
            if client is not None:
                try:
                    names = _tool_names(name, client.list_tools_sync())
                    said = f"connected, {len(names)} tool(s): " + ", ".join(names[:25])
                except Exception as exc:  # noqa: BLE001
                    from src.tools.mcp_resilience import describe
                    said = f"NOT running — its session is closed ({describe(exc)}); the next call restarts it"
                    _STATUS[name] = f"error: session closed ({describe(exc)})"
        elif said is None:
            said = state.replace("error:", "failed to start:") or "unknown"
        bundle = sc.get("bundle") if isinstance(sc.get("bundle"), dict) else None
        origin = f" [installed bundle {bundle.get('name')} {bundle.get('version')}]" if bundle else ""
        lines.append(f"- {name} — {how}, {sign_in} — {said}{origin}")
    report = ("MCP servers configured on this machine (their tools are the agent's, "
              "prefixed with the server name):\n" + "\n".join(lines))
    try:
        from src.tools import mcp_on_demand
        if mcp_on_demand.enabled():
            report += ("\nTools load on demand: `use_mcp_server(<name>)` puts a server's "
                       "tools in your list for this conversation — and connects one that "
                       "is signed in but not loaded yet, without a restart.")
    except Exception:  # noqa: BLE001
        pass
    return report


@tool
def list_mcp_servers() -> str:
    """Which external MCP servers this system has, and whether their tools work now.

    Answers "what MCP servers are set up?", "can you talk to Blender/Magnific?",
    and anything about capabilities beyond ComfyUI itself. Ask it before telling
    someone a capability is missing: a server can be configured yet not connected —
    switched off, waiting for a browser sign-in, or failing to start — and its tools
    are then absent from the tool list even though the user installed it.

    Returns one line per server: its name, how it is reached (http/sse/local
    command), its sign-in mode, and its state — connected (with its tool names),
    switched off, needing a sign-in, or the error it failed with — plus the bundle
    it was installed from, when it was.
    """
    return mcp_server_report()


def is_server_connected(name: str) -> bool:
    """True when *name* has a live MCP client in this process."""
    return name in _CLIENTS


def _session_dead(name: str) -> bool:
    """True when *name* has a client whose session is no longer running (its stdio
    process died). The status then must not keep saying "connected"."""
    client = _CLIENTS.get(name)
    check = getattr(client, "_is_session_active", None)
    try:
        return client is not None and callable(check) and not check()
    except Exception:  # noqa: BLE001
        return False


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
        if state and state.startswith("connected") and _session_dead(name):
            state = _STATUS[name] = "error: session closed (the next call restarts it)"
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
        if state and state.startswith("error"):
            out[name]["reason"] = _REASONS.get(name) or state.partition(":")[2].strip()
            out[name]["log"] = _log_rel(name)
    return out
