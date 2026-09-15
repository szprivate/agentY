"""Install MCP Bundles (.mcpb, formerly .dxt) as agentY MCP servers.

A bundle is a zip archive: a ``manifest.json`` plus a local MCP server with its
dependencies, for Node, Python, uv or a compiled binary. Claude Desktop installs
one with a click; this does the same for agentY, from Settings ▸ MCP servers.

Three calls, so the modal can show what it is about to do before doing it:

1. :func:`stage_upload` writes the uploaded file to a staging folder. A signed
   bundle carries a PKCS#7 block after the zip (``MCPB_SIG_V1`` + length +
   signature + ``MCPB_SIG_END``); it is cut off so the archive opens, and the
   bundle is reported as signed. The signature itself is NOT verified.
2. :func:`inspect` reads the manifest: what the bundle is, the command it will
   run, the settings it asks for (``user_config``), and whether this machine can
   run it: platform, and the runtime on PATH.
3. :func:`install` unpacks it into ``config/mcp_bundles/<name>/`` and resolves the
   manifest's ``mcp_config`` the way the MCPB reference implementation does
   (``getMcpConfigForManifest`` in modelcontextprotocol/mcpb): platform override,
   manifest defaults then the user's values, ``${__dirname}``, ``${HOME}`` and the
   other system folders, and ``${user_config.KEY}``, where a multi-value setting
   that is a whole argument expands into several arguments. A sensitive setting is
   never written into the entry: it becomes ``${MCP_<NAME>_<KEY>}`` and comes back
   in ``secrets`` for .env, like a key pasted into the Add form.

The result is an ordinary stdio server entry with the bundle folder as its working
directory and a ``bundle`` note. Unpacking refuses any path that leaves the bundle
folder, and a folder is only replaced or deleted when agentY's marker file is in
it: :func:`prune_unreferenced` removes installed bundles no saved server uses.
"""
from __future__ import annotations

import json
import os
import re
import secrets as _secrets
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path

from src.utils.mcp_import import _env_name, _slug

BUNDLES_ROOT = Path(__file__).resolve().parents[2] / "config" / "mcp_bundles"
MARKER = ".agenty-bundle.json"
MAX_UPLOAD_BYTES = 1 << 30
MAX_UNPACKED_BYTES = 2 << 30
MAX_FILES = 100_000
_STAGING_TTL_S = 24 * 3600
_SIGNATURES = ((b"MCPB_SIG_V1", b"MCPB_SIG_END"), (b"DXT_SIG_V1", b"DXT_SIG_END"))
_TOKEN = re.compile(r"^[0-9a-f]{24}$")
_NAME = re.compile(r"^[a-z0-9_]+$")
_USER_REF = re.compile(r"^\$\{user_config\.([^}]+)\}$")
_TYPES = ("string", "number", "boolean", "directory", "file")
_CLAUSE = re.compile(r"(>=|<=|>|<|==|=|\^|~)?\s*v?(\d+(?:\.\d+){0,2})")
_RUNTIME_OF = {"node": "node", "python": "python", "python3": "python", "py": "python"}
_INSTALL_HINTS = {"node": "Node.js (https://nodejs.org)", "uv": "uv (https://docs.astral.sh/uv/)",
                  "python": "Python", "python3": "Python", "py": "Python"}


class BundleError(ValueError):
    """The bundle cannot be staged, read or installed; the message says why."""


# ── staging ──────────────────────────────────────────────────────────────────

def _staging() -> Path:
    return BUNDLES_ROOT / ".staging"


def stage_upload(stream, filename: str = "", limit: int = MAX_UPLOAD_BYTES) -> str:
    """Write an uploaded bundle to the staging folder; return its token."""
    staging = _staging()
    staging.mkdir(parents=True, exist_ok=True)
    _sweep(staging)
    token = _secrets.token_hex(12)
    path = staging / f"{token}.mcpb"
    total = 0
    try:
        with open(path, "wb") as out:
            while True:
                chunk = stream.read(1 << 20)
                if not chunk:
                    break
                total += len(chunk)
                if total > limit:
                    raise BundleError(f"the bundle is larger than {limit // (1 << 20)} MB")
                out.write(chunk)
        if not total:
            raise BundleError("the upload was empty")
        signed = _cut_signature(path)
    except BaseException:
        path.unlink(missing_ok=True)
        raise
    (staging / f"{token}.json").write_text(json.dumps({
        "filename": os.path.basename(str(filename or "")), "signed": signed, "size": total}),
        encoding="utf-8")
    return token


def discard(token: str) -> None:
    """Forget a staged upload."""
    if _TOKEN.match(str(token or "")):
        for suffix in (".mcpb", ".json"):
            (_staging() / f"{token}{suffix}").unlink(missing_ok=True)


def _sweep(staging: Path) -> None:
    now = time.time()
    for item in staging.iterdir():
        try:
            if now - item.stat().st_mtime > _STAGING_TTL_S:
                item.unlink()
        except OSError:
            pass


def _cut_signature(path: Path) -> bool:
    """Cut a trailing signature block off the archive; True when there was one."""
    size = path.stat().st_size
    with open(path, "r+b") as f:
        f.seek(max(0, size - (1 << 20)))
        tail = f.read()
        for header, footer in _SIGNATURES:
            if tail.endswith(footer):
                start = tail.rfind(header)
                if start >= 0:
                    f.truncate(size - len(tail) + start)
                    return True
    return False


def _staged(token: str):
    if not _TOKEN.match(str(token or "")):
        raise BundleError("unknown upload; choose the bundle file again")
    path = _staging() / f"{token}.mcpb"
    if not path.is_file():
        raise BundleError("that upload has expired; choose the bundle file again")
    try:
        meta = json.loads((_staging() / f"{token}.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        meta = {}
    return path, meta


# ── reading ──────────────────────────────────────────────────────────────────

def _read_bundle(path: Path):
    """``(manifest, prefix, stats)``; *prefix* is the folder a bundle was zipped in."""
    try:
        zf = zipfile.ZipFile(path)
    except (zipfile.BadZipFile, OSError):
        raise BundleError("this is not an MCP bundle: the file is not a zip archive") from None
    with zf:
        names = [i.filename for i in zf.infolist()]
        prefix = _manifest_prefix(names)
        try:
            manifest = json.loads(zf.read(prefix + "manifest.json").decode("utf-8-sig"))
        except (KeyError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise BundleError(f"the bundle's manifest.json cannot be read: {exc}") from None
        infos = [i for i in zf.infolist() if i.filename.startswith(prefix) and not i.is_dir()]
        stats = {"files": len(infos), "unpacked": sum(i.file_size for i in infos)}
    if not isinstance(manifest, dict):
        raise BundleError("the bundle's manifest.json is not an object")
    for field in ("name", "version"):
        if not manifest.get(field):
            raise BundleError(f"the bundle's manifest has no {field}")
    return manifest, prefix, stats


def _manifest_prefix(names: list) -> str:
    if "manifest.json" in names:
        return ""
    tops = {n.split("/", 1)[0] for n in names if n.strip("/")}
    if len(tops) == 1:
        top = next(iter(tops))
        if f"{top}/manifest.json" in names:
            return f"{top}/"
    raise BundleError("this is not an MCP bundle: it has no manifest.json")


def _platform() -> str:
    if sys.platform.startswith("linux"):
        return "linux"
    return sys.platform


def _mcp_config(manifest: dict) -> dict:
    """The manifest's start command for this platform, before any substitution."""
    server = manifest.get("server")
    if not isinstance(server, dict):
        raise BundleError("the bundle's manifest has no server section")
    config = server.get("mcp_config")
    if not isinstance(config, dict) or not config.get("command"):
        config = _default_config(server)
    result = {"command": config.get("command"), "args": list(config.get("args") or []),
              "env": dict(config.get("env") or {})}
    override = (config.get("platform_overrides") or {}).get(_platform())
    if isinstance(override, dict):
        # As the reference does it: `override.command || base`, `override.args ||
        # base` (an empty list is a real override there), env merged over base.
        result["command"] = override.get("command") or result["command"]
        if override.get("args") is not None:
            result["args"] = list(override["args"])
        if override.get("env"):
            result["env"] = {**result["env"], **override["env"]}
    if not result["command"]:
        raise BundleError("the bundle's manifest does not say how to start the server")
    return result


def _default_config(server: dict) -> dict:
    kind = str(server.get("type") or "").lower()
    entry = str(server.get("entry_point") or "")
    if not entry:
        raise BundleError("the bundle's manifest has neither mcp_config nor an entry_point")
    if kind == "node":
        return {"command": "node", "args": ["${__dirname}/" + entry]}
    if kind == "python":
        return {"command": "python", "args": ["${__dirname}/" + entry]}
    if kind == "uv":
        return {"command": "uv", "args": ["run", "--directory", "${__dirname}", entry]}
    if kind == "binary":
        return {"command": "${__dirname}/" + entry, "args": []}
    raise BundleError(f"the bundle's server type {kind!r} is not one agentY knows")


def _variables(dirname: str) -> dict:
    home = Path.home()
    return {"__dirname": dirname, "pathSeparator": os.sep, "/": os.sep, "HOME": str(home),
            "DESKTOP": str(home / "Desktop"), "DOCUMENTS": str(home / "Documents"),
            "DOWNLOADS": str(home / "Downloads")}


def _replace(value, variables: dict):
    """``replaceVariables`` from the MCPB reference implementation, in Python."""
    if isinstance(value, str):
        result = value
        for key, replacement in variables.items():
            token = "${" + key + "}"
            if token in result and not isinstance(replacement, list):
                result = result.replace(token, replacement)
        return result
    if isinstance(value, list):
        out = []
        for item in value:
            ref = _USER_REF.match(item) if isinstance(item, str) else None
            if ref:
                replacement = variables.get("user_config." + ref.group(1))
                if isinstance(replacement, list):
                    out.extend(replacement)
                elif replacement:
                    out.append(replacement)
                else:
                    out.append(item)
            else:
                out.append(_replace(item, variables))
        return out
    if isinstance(value, dict):
        return {k: _replace(v, variables) for k, v in value.items()}
    return value


def _blank(value) -> bool:
    if isinstance(value, list):
        return not value or any(v is None or v == "" for v in value)
    return value is None or value == ""


def _user_values(manifest: dict, values: dict | None, name: str, system: dict):
    """The settings merged as the reference merges them (manifest defaults, then
    what was entered), checked, and turned into substitution variables. A sensitive
    value becomes a ``${MCP_<NAME>_<KEY>}`` reference and is returned for .env."""
    spec = manifest.get("user_config") if isinstance(manifest.get("user_config"), dict) else {}
    merged: dict = {}
    for key, opt in spec.items():
        if isinstance(opt, dict) and opt.get("default") is not None:
            merged[key] = opt["default"]
    for key, value in (values or {}).items():
        if key in spec and not _blank(value):
            merged[key] = value
    secrets: dict = {}
    variables: dict = {}
    for key, opt in spec.items():
        opt = opt if isinstance(opt, dict) else {}
        title = opt.get("title") or key
        value = _replace(merged.get(key), system)
        if _blank(value):
            if opt.get("required"):
                raise BundleError(f"{title} is required")
            continue
        kind = opt.get("type") if opt.get("type") in _TYPES else "string"
        if kind == "number":
            try:
                number = float(value)
            except (TypeError, ValueError):
                raise BundleError(f"{title} must be a number") from None
            if opt.get("min") is not None and number < float(opt["min"]):
                raise BundleError(f"{title} must be at least {opt['min']}")
            if opt.get("max") is not None and number > float(opt["max"]):
                raise BundleError(f"{title} must be at most {opt['max']}")
            value = int(number) if number.is_integer() else number
        elif kind == "boolean":
            value = value if isinstance(value, bool) else str(value).strip().lower() in ("true", "1", "yes", "on")
        elif kind in ("directory", "file") and opt.get("multiple"):
            items = value if isinstance(value, list) else str(value).splitlines()
            value = [str(v).strip() for v in items if str(v).strip()]
        if kind == "string" and opt.get("sensitive"):
            var = f"MCP_{_env_name(name)}_{_env_name(key)}"
            secrets[var] = str(value)
            value = "${" + var + "}"
        if isinstance(value, list):
            variables["user_config." + key] = [str(v) for v in value]
        elif isinstance(value, bool):
            variables["user_config." + key] = "true" if value else "false"
        else:
            variables["user_config." + key] = str(value)
    return variables, secrets


# ── can this machine run it ──────────────────────────────────────────────────

def _compatibility(manifest: dict, config: dict):
    """``(problems, warnings)``. A problem blocks the install (wrong platform); a
    warning does not (a runtime missing from PATH can be installed afterwards, and
    Test shows whether the server starts)."""
    problems, warnings = [], []
    compat = manifest.get("compatibility") if isinstance(manifest.get("compatibility"), dict) else {}
    platforms = compat.get("platforms")
    if isinstance(platforms, list) and platforms and _platform() not in platforms:
        problems.append(f"This bundle runs on {', '.join(map(str, platforms))}, "
                        f"not on this machine ({_platform()}).")
    command = str(config.get("command") or "")
    if command and "${" not in command and not os.path.isabs(command):
        found = shutil.which(command)
        stem = Path(command).stem.lower()
        if not found:
            hint = _INSTALL_HINTS.get(stem)
            warnings.append(f"`{command}` is not on this machine's PATH, so the server will not "
                            f"start until it is installed" + (f": {hint}." if hint else "."))
        else:
            runtime = _RUNTIME_OF.get(stem)
            wanted = (compat.get("runtimes") or {}).get(runtime) if runtime else None
            if wanted:
                have = _version_of(found)
                if have and _satisfies(have, str(wanted)) is False:
                    warnings.append(f"This bundle wants {runtime} {wanted}; this machine has {have}.")
    return problems, warnings


def _version_of(executable: str):
    try:
        proc = subprocess.run([executable, "--version"], capture_output=True, text=True, timeout=8)
    except (OSError, subprocess.SubprocessError):
        return None
    found = re.search(r"\d+\.\d+(?:\.\d+)?", (proc.stdout or "") + (proc.stderr or ""))
    return found.group(0) if found else None


def _version_parts(text: str):
    found = re.search(r"(\d+)(?:\.(\d+))?(?:\.(\d+))?", str(text or ""))
    return tuple(int(g or 0) for g in found.groups()) if found else None


def _satisfies(version: str, constraint: str):
    """Whether *version* meets a semver-style *constraint* (``>=3.10 <4``, ``^18``,
    ``>=3.8,<4.0``, ``a || b``); None when the constraint cannot be read."""
    have = _version_parts(version)
    if have is None:
        return None
    verdicts = []
    for alternative in str(constraint or "").split("||"):
        clauses = _CLAUSE.findall(alternative)
        if clauses:
            verdicts.append(all(_meets(have, op, ver) for op, ver in clauses))
    return any(verdicts) if verdicts else None


def _meets(have: tuple, op: str, ver: str) -> bool:
    given = [int(p) for p in ver.split(".")]
    want = tuple(given + [0] * (3 - len(given)))
    if op in ("", "=", "=="):
        return have[:len(given)] == tuple(given)
    if op == ">=":
        return have >= want
    if op == ">":
        return have > want
    if op == "<=":
        return have <= want
    if op == "<":
        return have < want
    if op == "^":
        upper = (want[0] + 1, 0, 0) if want[0] else (0, want[1] + 1, 0)
        return want <= have < upper
    upper = (want[0], want[1] + 1, 0) if len(given) > 1 else (want[0] + 1, 0, 0)   # "~"
    return want <= have < upper


# ── inspect ──────────────────────────────────────────────────────────────────

_PREVIEW_DIR = "<bundle folder>"


def _quote(arg: str) -> str:
    # The placeholder's own space is not a reason to quote the argument.
    return f'"{arg}"' if re.search(r"\s", arg.replace(_PREVIEW_DIR, "")) else arg


def _suggest_name(manifest: dict, servers: dict, taken) -> tuple:
    """A server name for the bundle; the installed one when this is a reinstall."""
    for existing, sc in (servers or {}).items():
        bundle = sc.get("bundle") if isinstance(sc, dict) else None
        if isinstance(bundle, dict) and bundle.get("name") == manifest.get("name"):
            return existing, existing
    used = set(servers or {}) | {str(t) for t in taken or ()}
    base = _slug(manifest.get("name"))
    name, n = base, 2
    while name in used or ((BUNDLES_ROOT / name).exists() and not (BUNDLES_ROOT / name / MARKER).is_file()):
        name, n = f"{base}_{n}", n + 1
    return name, ""


def inspect(token: str, servers: dict | None = None, taken=()) -> dict:
    """What the staged bundle is, what it will run, what it asks for, and whether
    this machine can run it."""
    path, meta = _staged(token)
    manifest, _prefix, stats = _read_bundle(path)
    config = _mcp_config(manifest)
    problems, warnings = _compatibility(manifest, config)
    preview = _replace(config, _variables(_PREVIEW_DIR))
    system = _variables("")
    system.pop("__dirname")
    fields = []
    spec = manifest.get("user_config") if isinstance(manifest.get("user_config"), dict) else {}
    for key, opt in spec.items():
        opt = opt if isinstance(opt, dict) else {}
        fields.append({
            "key": key, "type": opt.get("type") if opt.get("type") in _TYPES else "string",
            "title": opt.get("title") or key, "description": opt.get("description") or "",
            "required": bool(opt.get("required")), "sensitive": bool(opt.get("sensitive")),
            "multiple": bool(opt.get("multiple")), "min": opt.get("min"), "max": opt.get("max"),
            "default": _replace(opt.get("default"), system),
        })
    author = manifest.get("author")
    name, replaces = _suggest_name(manifest, servers or {}, taken)
    tools = [t.get("name") for t in manifest.get("tools") or [] if isinstance(t, dict) and t.get("name")]
    server = manifest.get("server") if isinstance(manifest.get("server"), dict) else {}
    return {
        "ok": True, "token": token, "name": name, "replaces": replaces,
        "bundle": {
            "name": manifest.get("name"), "display_name": manifest.get("display_name") or manifest.get("name"),
            "version": str(manifest.get("version")), "description": manifest.get("description") or "",
            "author": (author.get("name") if isinstance(author, dict) else author) or "",
            "homepage": manifest.get("homepage") or manifest.get("repository") or "",
            "server_type": server.get("type") or "", "signed": bool(meta.get("signed")),
            "size": meta.get("size"), "files": stats["files"], "unpacked": stats["unpacked"],
            "tools": tools,
            "command": " ".join(_quote(str(a)) for a in [preview["command"], *preview["args"]]),
        },
        "user_config": fields, "problems": problems, "warnings": warnings,
    }


# ── install ──────────────────────────────────────────────────────────────────

def _extract(zf: zipfile.ZipFile, prefix: str, dest: Path) -> int:
    """Unpack into *dest*, refusing any path that leaves it; returns links skipped."""
    dest.mkdir(parents=True, exist_ok=True)
    root = dest.resolve()
    skipped = written = files = 0
    for info in zf.infolist():
        name = info.filename.replace("\\", "/")
        if prefix:
            if not name.startswith(prefix):
                continue
            name = name[len(prefix):]
        if not name.strip("/"):
            continue
        parts = [p for p in name.split("/") if p not in ("", ".")]
        if name.startswith("/") or re.match(r"^[A-Za-z]:", name) or ".." in parts:
            raise BundleError(f"refusing to unpack {info.filename}: the path leaves the bundle folder")
        target = root.joinpath(*parts)
        if root != target.resolve() and root not in target.resolve().parents:
            raise BundleError(f"refusing to unpack {info.filename}: the path leaves the bundle folder")
        if info.is_dir():
            target.mkdir(parents=True, exist_ok=True)
            continue
        mode = info.external_attr >> 16
        if (mode & 0o170000) == 0o120000:
            skipped += 1
            continue
        files += 1
        if files > MAX_FILES:
            raise BundleError("the bundle has too many files to unpack")
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(info) as src, open(target, "wb") as out:
                while True:
                    chunk = src.read(1 << 20)
                    if not chunk:
                        break
                    written += len(chunk)
                    if written > MAX_UNPACKED_BYTES:
                        raise BundleError("the bundle unpacks to more than 2 GB")
                    out.write(chunk)
        except OSError as exc:
            raise BundleError(f"could not unpack {info.filename}: {exc}") from None
        if os.name != "nt" and mode & 0o111:
            os.chmod(target, (mode & 0o777) | 0o600)
    return skipped


def install(token: str, name: str, values: dict | None = None) -> dict:
    """Unpack the staged bundle and return its server entry and its secrets.

    Nothing is written to mcp.json or .env here: the settings modal adds the entry
    as an unsaved server, and Save writes both, like any other server.
    """
    path, meta = _staged(token)
    manifest, prefix, stats = _read_bundle(path)
    name = str(name or "").strip()
    if not _NAME.match(name):
        raise BundleError("the server name may use lowercase letters, digits and _")
    config = _mcp_config(manifest)
    problems, warnings = _compatibility(manifest, config)
    if problems:
        raise BundleError(" ".join(problems))
    if stats["files"] > MAX_FILES or stats["unpacked"] > MAX_UNPACKED_BYTES:
        raise BundleError("the bundle is too large to unpack")
    target = BUNDLES_ROOT / name
    # Checked before anything touches the disk: a bad value installs nothing.
    variables = _variables(str(target))
    user_vars, secrets = _user_values(manifest, values, name, variables)
    if target.exists() and not (target / MARKER).is_file():
        raise BundleError(f"config/mcp_bundles/{name} already exists and was not installed by "
                          "agentY; choose another name")
    BUNDLES_ROOT.mkdir(parents=True, exist_ok=True)
    work = BUNDLES_ROOT / f".{name}.installing-{token[:8]}"
    old = BUNDLES_ROOT / f".{name}.replaced-{token[:8]}"
    shutil.rmtree(work, ignore_errors=True)
    moved_old = False
    try:
        with zipfile.ZipFile(path) as zf:
            skipped = _extract(zf, prefix, work)
        (work / MARKER).write_text(json.dumps({
            "name": manifest.get("name"), "version": str(manifest.get("version")),
            "installed": time.strftime("%Y-%m-%dT%H:%M:%S")}), encoding="utf-8")
        try:
            if target.exists():
                target.rename(old)
                moved_old = True
            work.rename(target)
        except OSError as exc:
            raise BundleError(f"could not replace config/mcp_bundles/{name} ({exc}). If the "
                              "server is running from it, stop the agent and install again.") from None
    except BaseException:
        shutil.rmtree(work, ignore_errors=True)
        if moved_old and old.exists() and not target.exists():
            old.rename(target)
        raise
    if moved_old:
        shutil.rmtree(old, ignore_errors=True)

    variables.update(user_vars)
    resolved = _replace(config, variables)
    entry = {"enabled": True, "transport": "stdio", "command": str(resolved["command"]),
             "args": [str(a) for a in resolved["args"]], "auth": "none"}
    if resolved["env"]:
        entry["env"] = {str(k): str(v) for k, v in resolved["env"].items()}
    entry["cwd"] = str(target)
    entry["bundle"] = {"name": manifest.get("name"), "version": str(manifest.get("version")),
                       "dir": f"config/mcp_bundles/{name}", "signed": bool(meta.get("signed"))}
    discard(token)
    if skipped:
        warnings.append(f"Skipped {skipped} link(s) inside the bundle.")
    return {"ok": True, "name": name, "server": entry, "secrets": secrets, "warnings": warnings}


def prune_unreferenced(servers: dict) -> list:
    """Delete installed bundles no server uses any more; return their names.

    Only folders carrying agentY's marker are touched, so nothing a person put in
    config/mcp_bundles by hand is ever removed. A folder still in use by a running
    server may refuse to go on Windows; it is simply tried again on the next save.
    """
    if not BUNDLES_ROOT.is_dir():
        return []
    used = set()
    for sc in (servers or {}).values():
        bundle = sc.get("bundle") if isinstance(sc, dict) else None
        if isinstance(bundle, dict) and bundle.get("dir"):
            used.add(Path(str(bundle["dir"])).name)
    removed = []
    now = time.time()
    for child in sorted(BUNDLES_ROOT.iterdir()):
        if not child.is_dir():
            continue
        if child.name.startswith("."):
            if child.name != ".staging" and now - child.stat().st_mtime > _STAGING_TTL_S:
                shutil.rmtree(child, ignore_errors=True)
            continue
        if child.name in used or not (child / MARKER).is_file():
            continue
        shutil.rmtree(child, ignore_errors=True)
        if not child.exists():
            removed.append(child.name)
    return removed
