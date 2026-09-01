"""Who is allowed to talk to the agent host, and how the panel proves it is the panel.

The host listens on localhost and used to answer everyone. That is not the same as
being unreachable: a browser is a confused deputy, and any page in any tab can
``fetch('http://127.0.0.1:5001/…')`` on the user's own machine. Combined with
``Access-Control-Allow-Origin: *`` — which made the *responses* readable — that put
``GET /agentY/settings``, which returns ``.env`` verbatim, one line of JavaScript
away from any website the user happened to visit. ``POST /agentY/chat`` was the
other half: the agent holds ``run_script``, so a page that can put words in its
mouth can run commands.

Three checks, because they stop three different callers:

**Host** — the ``Host`` header must name an IP literal or ``localhost``. This is
the DNS-rebinding defence and nothing else: rebinding works by pointing a *name*
the attacker controls at 127.0.0.1, so refusing names is what breaks it. A real
name for this machine (``studio.local``) can be added to ``security.allowed_hosts``.

**Origin** — a browser sets it, and cannot be talked out of setting it truthfully.
The legitimate panel is served by ComfyUI and builds the backend address from its
own ``location.hostname``, so in every honest case the Origin's hostname EQUALS the
Host's. That equality is the rule, which is why a LAN install works without any
configuration: both sides are whatever name the user typed. The port must be one we
expect — ComfyUI's, or our own for the pages we serve ourselves.

**Token** — a shared secret minted per host start, handed to the panel through
ComfyUI (see :func:`session_token`). Origin checking is enforced by the *browser's*
honesty about who it is; a curl, a script, or anything else on the machine or the
LAN simply omits the header. The token is what those have to get past, and they
cannot read it without already being able to read the user's files.

Ordering matters: OPTIONS is answered before any of it, because a preflight cannot
carry a token and refusing it would make the browser report a CORS failure for what
is really an auth decision — the most confusing possible way to say "no".
"""

from __future__ import annotations

import ipaddress
import os
import re
import secrets
import threading
from pathlib import Path
from urllib.parse import urlsplit

# The header the panel sends. Named, not guessed: a custom header is also what
# forces a preflight on every state-changing request, so a form-post from another
# origin cannot reach a handler even if the Origin check were ever bypassed.
TOKEN_HEADER = "X-AgentY-Token"
# Where the host leaves the token for the ComfyUI extension to read. Beside the
# checkout rather than passed over HTTP: registration is best-effort and races
# ComfyUI's own startup, and a panel that could not learn the token would be a
# panel that cannot talk to its host at all.
TOKEN_FILENAME = ".agenty_token"

# Reachable without a token, and only these.
#
# health   the liveness probe. The panel calls it to decide whether to show "host
#          is down", and it must be able to reach that conclusion before it has a
#          token — otherwise a token problem is indistinguishable from a dead host,
#          which is precisely the diagnosis we spent a week getting wrong before.
# viewers  top-level page loads. A navigation cannot carry a header; the token is
#          injected into the HTML instead (see inject_token), and the Origin check
#          is what stops another page from fetching them to read it out.
PUBLIC_PATHS = frozenset({
    "/agentY/health",
    "/agentY/log_viewer",
    "/agentY/memory_viewer",
    "/agentY/project_memory_viewer",
})

_lock = threading.Lock()
_token: str | None = None


# ── The token ────────────────────────────────────────────────────────────────

def token_path(root: Path) -> Path:
    return Path(root) / TOKEN_FILENAME


# What a token from the file has to look like before we trust it. Narrow on
# purpose: it is spliced into a <script> for the viewer pages, and a value that
# could contain a quote or a tag would be a way to run script in that page.
_TOKEN_RE = re.compile(r"^[A-Za-z0-9_-]{16,128}$")


def session_token(root: Path | None = None) -> str:
    """This host's token: the one beside the checkout, or a new one if there isn't.

    **Reused across restarts, deliberately.** It used to be minted per process, and
    that made the panel's most ordinary situation a broken one: a ComfyUI tab reads
    the token once, at page load, and the host is restarted far more often than the
    tab is reloaded. Every restart therefore silently invalidated every open panel
    — "the tab is always older than the token" — and the only cure on offer was
    asking people to reload a tab that gave them no reason to think they should.

    Nothing is bought by rotating it. The file is owner-only and sits beside
    ``.env``: anyone who can read one can read the other, and the other is every
    API key you own. A token that changes hourly protects nothing from an attacker
    who has already won, while costing every user a panel that stops working for
    reasons it cannot explain.

    ``AGENTY_SESSION_TOKEN`` overrides, for the one case the file cannot serve: a
    panel and a host on different machines, where somebody has to carry the secret
    across by hand.
    """
    global _token
    with _lock:
        if _token:
            return _token
        forced = (os.environ.get("AGENTY_SESSION_TOKEN") or "").strip()
        if forced:
            _token = forced
            return _token
        existing = _read_token(root) if root is not None else ""
        _token = existing or secrets.token_urlsafe(32)
        if root is not None and not existing:
            _write_token(Path(root), _token)
        return _token


def _read_token(root: Path) -> str:
    """The token already beside this checkout, or "" if there is nothing usable.

    A malformed file is treated as absent rather than repaired: the next line
    writes a good one over it, which is the only outcome anybody wants.
    """
    try:
        raw = token_path(Path(root)).read_text(encoding="utf-8").strip()
    except OSError:
        return ""
    return raw if _TOKEN_RE.match(raw) else ""


def _write_token(root: Path, token: str) -> bool:
    """Drop the token beside the checkout, owner-only. Best-effort by design.

    A read-only checkout is a bad place to fail startup over; the panel can still
    be given the token by hand through ``AGENTY_SESSION_TOKEN``, and the Origin
    check is unaffected either way.
    """
    path = token_path(root)
    try:
        path.write_text(token + "\n", encoding="utf-8")
    except OSError:
        return False
    try:
        os.chmod(path, 0o600)
    except OSError:
        pass
    return True


def clear_token_file(root: Path) -> None:
    """Delete the token, so the next start mints a fresh one.

    NOT called at shutdown any more. Deleting it there is what made every host
    restart lock out every open ComfyUI tab: the panel had read the old value at
    page load and there was no new one to be had until it reloaded. Kept for the
    case it was always right for — deliberately revoking a token you believe
    somebody else has seen.
    """
    try:
        token_path(Path(root)).unlink()
    except OSError:
        pass


# ── Deciding ─────────────────────────────────────────────────────────────────

def _hostname(value: str) -> str:
    """The host part of a ``Host`` header or an origin, without the port.

    Bracketed IPv6 (``[::1]:5001``) is why this is not a ``split(":")``.
    """
    raw = str(value or "").strip()
    if not raw:
        return ""
    if "://" in raw:
        raw = urlsplit(raw).netloc
    if raw.startswith("["):
        end = raw.find("]")
        return raw[1:end].lower() if end > 0 else raw.lower()
    # A bare IPv6 literal is all colons and no brackets, so splitting on the first
    # one returns the empty string — which then fails every check, including the
    # one that would have let it through. Two or more colons means the whole thing
    # is the address; a host:port pair has exactly one.
    if raw.count(":") > 1:
        return raw.lower()
    return raw.split(":", 1)[0].lower()


def _port(value: str, default: int = 0) -> int:
    raw = str(value or "").strip()
    if "://" in raw:
        try:
            return urlsplit(raw).port or default
        except ValueError:
            return default
    if raw.startswith("["):
        end = raw.find("]")
        raw = raw[end + 1:] if end > 0 else raw
    if ":" in raw:
        try:
            return int(raw.rsplit(":", 1)[1])
        except ValueError:
            return default
    return default


def is_local_hostname(name: str) -> bool:
    """Is this a name that cannot have been pointed here by an attacker's DNS?

    IP literals and ``localhost`` only. Everything else is a name somebody else
    may control, which is the whole mechanism of a rebinding attack.
    """
    host = _hostname(name)
    if not host:
        return False
    if host in ("localhost", "localhost.localdomain"):
        return True
    try:
        ipaddress.ip_address(host)
        return True
    except ValueError:
        return False


def allowed_ports(agent_port: int, comfyui_url: str) -> set[int]:
    """Ports an honest Origin can be on: ComfyUI's (the panel) and ours (viewers)."""
    ports = {int(agent_port)} if agent_port else set()
    comfy = _port(comfyui_url, 8188)
    if comfy:
        ports.add(comfy)
    return ports


def verdict(*, method: str, path: str, host_header: str, origin: str,
            token: str, expected_token: str, agent_port: int,
            comfyui_url: str = "http://127.0.0.1:8188",
            allowed_hosts: tuple[str, ...] = (),
            allowed_origins: tuple[str, ...] = (),
            check_origin: bool = True,
            require_token: bool = True) -> tuple[bool, str]:
    """Allow this request? Returns ``(ok, reason)``; *reason* is empty when ok.

    Pure, so the rules can be tested without a server and without a browser —
    which matters more here than anywhere else in the codebase, because the
    interesting cases are all the ones a manual test cannot produce.
    """
    if str(method or "").upper() == "OPTIONS":
        return True, ""

    host = _hostname(host_header)
    extra_hosts = {_hostname(h) for h in allowed_hosts if h}
    if check_origin and host and not is_local_hostname(host) and host not in extra_hosts:
        return False, (f"refusing a request addressed to '{host}'. This host answers "
                       "on an IP address or 'localhost'; a request naming anything "
                       "else is how DNS rebinding reaches a local server. Add the "
                       "name to security.allowed_hosts if it is really yours.")

    if origin and check_origin:
        allowed = {str(o).rstrip("/").lower() for o in allowed_origins if o}
        if str(origin).rstrip("/").lower() not in allowed:
            o_host = _hostname(origin)
            o_port = _port(origin)
            if o_host != host:
                return False, (f"refusing a cross-site request from {origin}. The agentY "
                               f"panel is served from the same host you reached this "
                               f"server on ({host or 'unknown'}); this request came from "
                               "somewhere else.")
            if o_port and o_port not in allowed_ports(agent_port, comfyui_url):
                return False, (f"refusing a request from {origin}: port {o_port} is not "
                               "where ComfyUI or this host is served. Add the origin to "
                               "security.allowed_origins if it is really yours.")

    if require_token and str(path or "") not in PUBLIC_PATHS:
        if not expected_token:
            return True, ""      # nothing to check against; origin rules still applied
        if not secrets.compare_digest(str(token or ""), str(expected_token)):
            return False, ("missing or stale session token. The panel reads it from "
                           "ComfyUI at page load: reload the ComfyUI tab. If it keeps "
                           "failing, the agentY extension is older than this host — "
                           "update it, or set security.require_token = false to fall "
                           "back to origin checks alone.")
    return True, ""


def inject_token(html: str, token: str) -> str:
    """Hand a page we serve ourselves the token, since a navigation carries no header.

    The shim goes in as well as the value. Every request these pages make is to
    us, so wrapping ``fetch`` once is exact — and it means the viewer HTML files
    stay untouched, which is the difference between a change that is done and one
    that has to be repeated in every page somebody adds later.

    Inserted right after ``<head>`` so it runs before any script on the page. A
    page with no ``<head>`` gets it prepended — still before everything.
    """
    if not token:
        return html
    # json.dumps-equivalent escaping is unnecessary: the token is url-safe base64
    # from secrets.token_urlsafe, so it contains nothing that can close a string
    # or a script tag. Asserted rather than assumed.
    safe = "".join(c for c in str(token) if c.isalnum() or c in "-_")
    tag = (
        "<script>\n"
        f'window.AGENTY_TOKEN = "{safe}";\n'
        "(function () {\n"
        "  var real = window.fetch.bind(window);\n"
        "  window.fetch = function (input, init) {\n"
        "    var opts = Object.assign({}, init);\n"
        "    var head = new Headers((init && init.headers) ||\n"
        "                           (input && input.headers) || undefined);\n"
        f'    head.set("{TOKEN_HEADER}", window.AGENTY_TOKEN);\n'
        "    opts.headers = head;\n"
        "    return real(input, opts);\n"
        "  };\n"
        "})();\n"
        "</script>"
    )
    lowered = html.lower()
    at = lowered.find("<head>")
    if at >= 0:
        cut = at + len("<head>")
        return html[:cut] + "\n" + tag + html[cut:]
    return tag + "\n" + html
