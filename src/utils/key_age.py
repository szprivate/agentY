"""How long each API key in ``.env`` has been sitting there, and when to say so.

A key is a bearer credential with no expiry: it works until somebody revokes it,
which means a copy taken today is still good next year. The only defence that does
not depend on noticing a breach is rotating on a clock — and nobody rotates on a
clock without a reminder, because nothing about a working key looks any different
from a leaked one.

So this keeps a small ledger beside the settings: for each key, a fingerprint and
the date we first saw that exact value. Change the value and the clock restarts by
itself; that is the whole interaction, and it is why the ledger stores a
fingerprint rather than asking anyone to press "I rotated this".

**The value is never stored.** A SHA-256 prefix is enough to answer "is this still
the same secret" and is worthless to anyone who reads the file — which matters,
because a file that recorded the keys in order to nag about the keys would be the
thing it was warning about.

Seeding an install that predates this module
--------------------------------------------
There is no way to learn when a key already in ``.env`` was issued. The honest
lower bound is the file's own mtime: the last time anything in it was written. For
a key set long ago and never touched, that is *younger* than the truth, so the
first warning may come late — but it is never later than seeding from "now", which
is the only alternative, and it means a checkout whose ``.env`` was last edited two
years ago warns on the first run rather than in thirty days. Entries seeded this
way are marked ``estimated`` so the UI can say where the date came from instead of
implying a precision it does not have.
"""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

# What counts as a secret. Kept in step with the settings panel's own isSecret()
# test (see is_secret_key) — the two decide the same thing on either side of the
# wire, and a disagreement is a leak rather than a cosmetic difference.
_SECRET_WORDS = ("TOKEN", "KEY", "SECRET", "PASSWORD")
# Names that contain one of those words and are still not credentials —
# DASHSCOPE_BASE_URL and SLACK_ALLOWED_USERS live in .env beside the real ones. A
# rotation reminder about a URL trains people to ignore rotation reminders.
_NOT_SECRET_SUFFIXES = ("_URL", "_USERS", "_HOST", "_PORT", "_DIR", "_PATH")

DEFAULT_MAX_AGE_DAYS = 30


def is_secret_key(name: str) -> bool:
    """Is *name* a credential, as opposed to a setting that lives beside one?

    Substring, not suffix, and deliberately so: the settings panel decides which
    fields to render as password inputs with the same test in JavaScript
    (``/KEY|TOKEN|SECRET|PASSWORD/i``), and the user can add arbitrary keys to
    ``.env`` through it. If the two disagreed, a name like ``SECRET_VALUE`` would
    be shown as a password field and sent to the browser in plaintext — masked to
    the eye and not to the wire, which is the worst of both.

    Erring wide is the cheap direction. Masking something that turns out not to be
    a credential costs one field the user has to retype; failing to mask one costs
    the credential.
    """
    n = str(name or "").upper()
    if not n:
        return False
    if any(n.endswith(sfx) for sfx in _NOT_SECRET_SUFFIXES):
        return False
    return any(word in n for word in _SECRET_WORDS)


def fingerprint(value: str) -> str:
    """A stable, non-reversible id for a secret's *value*.

    Sixteen hex characters of SHA-256. Long enough that two different keys will
    not collide in a file that holds under a dozen of them; short enough that it
    reads as an opaque tag rather than as something worth trying to crack.
    """
    return hashlib.sha256(str(value or "").encode("utf-8")).hexdigest()[:16]


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _iso(when: datetime) -> str:
    return when.astimezone(timezone.utc).isoformat(timespec="seconds")


def _parse(stamp: str) -> datetime | None:
    try:
        parsed = datetime.fromisoformat(str(stamp))
    except (TypeError, ValueError):
        return None
    # A ledger hand-edited (or written by an older build) may carry a naive
    # timestamp. Treat it as UTC rather than throwing on the subtraction below.
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def age_days(first_seen: str, now: datetime | None = None) -> float | None:
    """Whole and fractional days since *first_seen*, or None if unparseable."""
    seen = _parse(first_seen)
    if seen is None:
        return None
    return max(0.0, ((now or _now()) - seen).total_seconds() / 86400.0)


def _seed_time(env_path: Path | None, now: datetime) -> tuple[str, bool]:
    """When to say a pre-existing key was first seen, and whether that is a guess.

    See the module docstring: ``.env``'s mtime is the best available lower bound
    and is never worse than "now".
    """
    try:
        if env_path is not None and env_path.exists():
            mtime = datetime.fromtimestamp(env_path.stat().st_mtime, tz=timezone.utc)
            if mtime < now:
                return _iso(mtime), True
    except OSError:
        pass
    return _iso(now), False


def load(path: Path) -> dict:
    """Read the ledger. A missing or corrupt file is an empty one, never an error.

    Losing the ledger costs a reset clock, so there is nothing here worth failing
    a startup over.
    """
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def save(path: Path, ledger: dict) -> bool:
    """Write the ledger, owner-readable only. Returns whether it landed."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(ledger, indent=2, sort_keys=True) + "\n",
                        encoding="utf-8")
    except OSError:
        return False
    # Not a secret store — but it names which credentials exist, and that is not
    # world-readable information either. Best-effort: a filesystem without POSIX
    # modes (a Windows share) is not a reason to fail.
    try:
        os.chmod(path, 0o600)
    except OSError:
        pass
    return True


def record(env: dict, ledger: dict, *, env_path: Path | None = None,
           now: datetime | None = None) -> dict:
    """Fold the current ``.env`` into *ledger* and return the updated copy.

    * a key we have not seen → recorded, seeded per :func:`_seed_time`
    * a key whose value changed → clock restarted, ``estimated`` cleared, because
      this one we watched happen
    * a key gone or blanked → dropped, so emptying a field also clears its warning
    * a key unchanged → left exactly as it was, including its original date
    """
    at = now or _now()
    out = {k: dict(v) for k, v in (ledger or {}).items() if isinstance(v, dict)}
    live = {k: v for k, v in (env or {}).items() if is_secret_key(k) and str(v or "").strip()}

    for key, value in live.items():
        fp = fingerprint(value)
        prev = out.get(key)
        if prev and prev.get("fp") == fp:
            continue
        if prev:
            # A value we saw change: the date is now, and it is not a guess.
            out[key] = {"fp": fp, "first_seen": _iso(at)}
        else:
            seeded, estimated = _seed_time(env_path, at)
            entry = {"fp": fp, "first_seen": seeded}
            if estimated:
                entry["estimated"] = True
            out[key] = entry

    for key in list(out):
        if key not in live:
            out.pop(key, None)
    return out


def report(env: dict, ledger: dict, max_age_days: float = DEFAULT_MAX_AGE_DAYS,
           now: datetime | None = None) -> list[dict]:
    """Per-key age, newest last, for the settings UI and the startup check.

    Each entry: ``{key, age_days, first_seen, estimated, stale}``. Only keys the
    ledger knows about appear — a key present in ``.env`` but missing here means
    :func:`record` has not run yet, and inventing an age for it would be a lie.
    """
    at = now or _now()
    limit = float(max_age_days or 0)
    out: list[dict] = []
    for key, entry in (ledger or {}).items():
        if not isinstance(entry, dict) or not str((env or {}).get(key, "")).strip():
            continue
        days = age_days(entry.get("first_seen", ""), at)
        if days is None:
            continue
        out.append({
            "key": key,
            "age_days": round(days, 1),
            "first_seen": entry.get("first_seen", ""),
            "estimated": bool(entry.get("estimated")),
            "stale": limit > 0 and days >= limit,
        })
    out.sort(key=lambda e: e["age_days"], reverse=True)
    return out


def stale(env: dict, ledger: dict, max_age_days: float = DEFAULT_MAX_AGE_DAYS,
          now: datetime | None = None) -> list[dict]:
    """Just the keys past their age limit. Empty when the limit is 0 (off)."""
    return [e for e in report(env, ledger, max_age_days, now) if e["stale"]]


def warning_lines(entries: list[dict], max_age_days: float = DEFAULT_MAX_AGE_DAYS) -> list[str]:
    """The startup warning, as lines. Empty when nothing is stale.

    Says the age and how to make it stop, because a warning that cannot be acted
    on gets turned off rather than fixed — and the way to make this one stop is
    the thing we actually want to happen.
    """
    if not entries:
        return []
    limit = int(max_age_days) if float(max_age_days).is_integer() else max_age_days
    lines = [f"{len(entries)} API key(s) older than {limit} days:"]
    for e in entries:
        days = int(e["age_days"])
        about = "at least " if e.get("estimated") else ""
        lines.append(f"  • {e['key']} — {about}{days} days old")
    lines.append("Rotate them at the provider, then paste the new value into the "
                 "settings panel (or .env). The clock restarts on its own.")
    lines.append("Change the interval with security.api_key_max_age_days "
                 "(0 turns this off).")
    return lines
