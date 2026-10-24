"""Per-user access tokens for the HTTP side of a server install.

The desktop install has ONE secret, the session token the panel reads from
ComfyUI. That is right for one person and wrong for a team: nobody can be told
apart, and revoking one person means changing the secret for everyone.

With ``security.user_tokens`` on, each person gets a token of their own. They are
stored as SHA-256 digests in ``config/users.json`` (gitignored), so the file does
not reveal a usable token; the token itself is shown once, when it is made, by
``scripts/agenty_users.py``. A token is 32 random bytes, so a plain digest is
enough - there is no password to brute-force.

This host is meant to sit behind a reverse proxy that terminates TLS (see
``docs/server.md``); a token over plain HTTP on an open network is a token on the
wire.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import os
import secrets
import tempfile
import threading
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent
_lock = threading.Lock()


def users_path() -> Path:
    return Path(os.environ.get("AGENTY_USERS_FILE") or (_ROOT / "config" / "users.json"))


def _digest(token: str) -> str:
    return hashlib.sha256(str(token).encode("utf-8")).hexdigest()


def load(path: Path | None = None) -> dict:
    """``{name: {token_sha256, slack_id, roles, created}}``. Missing or broken = {}."""
    try:
        data = json.loads((path or users_path()).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    users = data.get("users") if isinstance(data, dict) else None
    return {str(k): dict(v) for k, v in (users or {}).items() if isinstance(v, dict)}


def _save(users: dict, path: Path | None = None) -> None:
    path = path or users_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=".users-", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump({"users": users}, fh, indent=2, ensure_ascii=False)
            fh.write("\n")
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)
    try:
        os.chmod(path, 0o600)
    except OSError:
        pass


def add_user(name: str, *, slack_id: str = "", roles: list | None = None,
             path: Path | None = None) -> str:
    """Create (or re-issue the token of) *name*. Returns the token, shown once."""
    name = str(name).strip()
    if not name or not all(c.isalnum() or c in "._-@" for c in name):
        raise ValueError("a user name is letters, digits and . _ - @")
    token = secrets.token_urlsafe(32)
    with _lock:
        users = load(path)
        old = users.get(name, {})
        users[name] = {
            "token_sha256": _digest(token),
            "slack_id": str(slack_id or old.get("slack_id") or ""),
            "roles": sorted(set(roles if roles is not None else old.get("roles") or [])),
            "created": old.get("created") or int(time.time()),
            "token_issued": int(time.time()),
        }
        _save(users, path)
    return token


def remove_user(name: str, path: Path | None = None) -> bool:
    with _lock:
        users = load(path)
        if name not in users:
            return False
        del users[name]
        _save(users, path)
        return True


def match_token(token: str, users: dict | None = None) -> dict | None:
    """The user a token belongs to (``{"name", "slack_id", "roles"}``), or None.

    Every stored digest is compared, in constant time each, so how long a refusal
    takes says nothing about which digest was close.
    """
    token = str(token or "").strip()
    if not token:
        return None
    users = load() if users is None else users
    probe = _digest(token)
    found = None
    for name, rec in users.items():
        if hmac.compare_digest(probe, str(rec.get("token_sha256") or "")):
            found = {"name": name, "slack_id": rec.get("slack_id", ""),
                     "roles": list(rec.get("roles") or [])}
    return found


def bearer_or_header(headers) -> str:
    """The token a request carries: ``Authorization: Bearer`` or ``X-AgentY-Token``."""
    auth = str(headers.get("Authorization", "") or "")
    if auth.lower().startswith("bearer "):
        return auth[7:].strip()
    return str(headers.get("X-AgentY-Token", "") or "").strip()


def enabled() -> bool:
    if str(os.environ.get("AGENTY_USER_TOKENS", "")).strip().lower() in ("1", "true", "yes", "on"):
        return True
    try:
        from src.utils.settings import load_settings
        return bool((load_settings().get("security") or {}).get("user_tokens", False))
    except Exception:  # noqa: BLE001
        return False


def slack_roles() -> dict:
    """``{slack member id: roles}`` for the users who have a Slack id on file."""
    return {str(r["slack_id"]): list(r.get("roles") or [])
            for r in load().values() if r.get("slack_id")}
