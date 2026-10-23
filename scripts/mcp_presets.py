"""The MCP preset catalog (config/mcp_presets.json) and what installing a preset does.

One place for the logic, so the installers (install_agent.ps1 / .sh), the host's
Settings route and the tests all do the same thing:

    python scripts/mcp_presets.py list [--json]        the catalog
    python scripts/mcp_presets.py keys <id>            its .env keys (| separated)
    python scripts/mcp_presets.py steps <id>           its "finish later" text
    python scripts/mcp_presets.py apply <id>...        merge into config/mcp.json
    python scripts/mcp_presets.py warmup <id>          fetch its package, failures only warn

``apply`` never drops a server that is already in config/mcp.json and skips a preset
whose id is already there. The ``${VAR}`` placeholders of a preset stay as they are:
a key goes to .env, never into mcp.json. The catalog is read as it stands, any number
of presets, nothing here names one.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CATALOG = ROOT / "config" / "mcp_presets.json"
CONFIG = ROOT / "config" / "mcp.json"


def load_catalog(path: Path | None = None) -> list[dict]:
    """The presets, each with its optional parts filled in. A missing or unreadable
    catalog is an empty list; an entry without an id or a server is left out."""
    try:
        data = json.loads(Path(path or CATALOG).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    out = []
    for p in (data.get("presets") if isinstance(data, dict) else None) or []:
        if not isinstance(p, dict) or not p.get("id") or not isinstance(p.get("server"), dict):
            continue
        out.append({
            "id": str(p["id"]), "name": str(p.get("name") or p["id"]),
            "category": str(p.get("category") or ""), "description": str(p.get("description") or ""),
            "server": p["server"],
            "env_keys": [k for k in (p.get("env_keys") or []) if isinstance(k, dict) and k.get("key")],
            "requires": [str(r) for r in (p.get("requires") or [])],
            "setup_steps": str(p.get("setup_steps") or ""),
            "default_selected": bool(p.get("default_selected")),
        })
    return out


def find(preset_id: str, path: Path | None = None) -> dict | None:
    return next((p for p in load_catalog(path) if p["id"] == preset_id), None)


def read_config(path: Path | None = None) -> dict:
    try:
        data = json.loads(Path(path or CONFIG).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {"servers": {}}
    if not isinstance(data, dict) or not isinstance(data.get("servers"), dict):
        return {"servers": {}}
    return data


def _fill_root(value, root: str):
    """``${AGENTY_ROOT}`` -> this checkout, in every string of *value*."""
    if isinstance(value, str):
        return value.replace("${AGENTY_ROOT}", root)
    if isinstance(value, list):
        return [_fill_root(v, root) for v in value]
    if isinstance(value, dict):
        return {k: _fill_root(v, root) for k, v in value.items()}
    return value


def server_entry(preset: dict, platform: str | None = None, root: Path | None = None) -> dict:
    """A copy of the preset's server entry for this machine. ``${AGENTY_ROOT}`` becomes
    the agentY checkout (forward slashes: they work on every system), and ``npx`` is
    written as ``npx.cmd`` on Windows, where a bare ``npx`` is not something that can be
    started. Other ``${VAR}`` placeholders (keys) stay as they are."""
    entry = _fill_root(json.loads(json.dumps(preset["server"])), str(Path(root or ROOT)).replace(chr(92), "/"))
    if (platform or sys.platform).startswith("win") and entry.get("command") == "npx":
        entry["command"] = "npx.cmd"
    return entry


def needs_bundle(entry: dict, root: Path | None = None) -> bool:
    """A bundle server (a .mcpb unpacked under config/mcp_bundles) whose folder is not
    there yet: writing it would leave a server that cannot start."""
    bundle = entry.get("bundle")
    if not isinstance(bundle, dict):
        return False
    return not (Path(root or ROOT) / str(bundle.get("dir") or "")).is_dir()


def merge(config: dict, presets: list[dict], platform: str | None = None,
          root: Path | None = None) -> tuple[dict, dict]:
    """``(new config, {id: "added" | "skipped" | "needs_bundle"})``. Existing servers are
    untouched and keep their order; a preset whose id is taken is skipped, not
    overwritten; a bundle preset whose bundle is not installed is not written."""
    servers = dict((config or {}).get("servers") or {})
    result: dict[str, str] = {}
    for p in presets:
        if p["id"] in servers:
            result[p["id"]] = "skipped"
            continue
        entry = server_entry(p, platform, root)
        if needs_bundle(entry, root):
            result[p["id"]] = "needs_bundle"
            continue
        servers[p["id"]] = entry
        result[p["id"]] = "added"
    return {**(config or {}), "servers": servers}, result


def apply(ids: list[str], catalog_path: Path | None = None, config_path: Path | None = None,
          root: Path | None = None) -> dict:
    """Merge the presets *ids* into the config file (created if missing). Returns
    ``{id: "added" | "skipped" | "needs_bundle" | "unknown"}``. Nothing is written when
    nothing is added."""
    catalog = {p["id"]: p for p in load_catalog(catalog_path)}
    result = {i: "unknown" for i in ids if i not in catalog}
    chosen = [catalog[i] for i in ids if i in catalog]
    cfg_path = Path(config_path or CONFIG)
    merged, done = merge(read_config(cfg_path), chosen, root=root)
    result.update(done)
    if "added" in done.values():
        cfg_path.parent.mkdir(parents=True, exist_ok=True)
        cfg_path.write_text(json.dumps({"servers": merged["servers"]}, indent=2), encoding="utf-8")
    return result


def warmup_command(preset: dict) -> list[str] | None:
    """What fetches the preset's package ahead of its first use, or None when there
    is nothing cheap and safe to run (a local bundle, a hosted server)."""
    sc = preset["server"]
    if sc.get("transport", "stdio") != "stdio":
        return None
    cmd = str(sc.get("command") or "")
    args = [str(a) for a in sc.get("args") or []]
    base = os.path.basename(cmd).lower()
    if base in ("uvx", "uvx.exe"):
        return [cmd, *args, "--help"]
    if base in ("npx", "npx.cmd", "npx.exe"):
        pkg = next((a for a in args if not a.startswith("-")), "")
        return [cmd, "-y", pkg, "--version"] if pkg else None
    return None


def warmup(preset: dict, timeout: int = 120) -> tuple[bool, str]:
    """Run :func:`warmup_command`. ``(ok, message)``; never raises. A package that
    starts a server instead of printing a version runs into the timeout, which is fine
    (it has been fetched by then) and is reported as such."""
    cmd = warmup_command(preset)
    if not cmd:
        return True, "nothing to fetch"
    exe = shutil.which(cmd[0]) or (shutil.which(cmd[0] + ".cmd") if sys.platform.startswith("win") else None)
    if exe is None:
        return False, f"{cmd[0]} is not installed"
    try:
        run = subprocess.run([exe, *cmd[1:]], capture_output=True, text=True, timeout=timeout,
                             stdin=subprocess.DEVNULL)
    except subprocess.TimeoutExpired:
        return True, "fetched (it started instead of printing and was stopped)"
    except OSError as exc:
        return False, str(exc)
    if run.returncode == 0:
        return True, "fetched"
    return False, ((run.stderr or run.stdout or f"exit {run.returncode}").strip().splitlines() or [""])[-1][:200]


SEP = "|"   # not a tab: bash's read collapses empty tab-separated fields


def _row(*fields: str) -> str:
    return SEP.join(str(x).replace(SEP, "/").replace("\t", " ").replace("\n", " ") for x in fields)


def _main(argv: list[str] | None = None) -> int:
    try:   # one line ending everywhere: a shell reading the output must not see a CR
        sys.stdout.reconfigure(newline="\n", encoding="utf-8")
    except (AttributeError, ValueError):
        pass
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--catalog", default=None)
    ap.add_argument("--config", default=None)
    sub = ap.add_subparsers(dest="cmd", required=True)
    ls = sub.add_parser("list")
    ls.add_argument("--json", action="store_true")
    for name in ("keys", "steps", "warmup"):
        sub.add_parser(name).add_argument("id")
    sub.add_parser("apply").add_argument("ids", nargs="+")
    a = ap.parse_args(argv)
    cat = Path(a.catalog) if a.catalog else None
    if a.cmd == "list":
        presets = load_catalog(cat)
        if a.json:
            print(json.dumps(presets, indent=2))
        else:
            for p in presets:
                print(_row(p["id"], p["name"], p["category"], "1" if p["default_selected"] else "0", p["description"]))
        return 0
    if a.cmd == "apply":
        res = apply(a.ids, cat, Path(a.config) if a.config else None)
        for i, state in res.items():
            print(_row(state, i))
        return 0
    p = find(a.id, cat)
    if p is None:
        print(f"unknown preset: {a.id}", file=sys.stderr)
        return 2
    if a.cmd == "keys":    # key, label, help_url, optional (0/1)
        for k in p["env_keys"]:
            print(_row(k["key"], k.get("label") or k["key"], k.get("help_url") or "", "1" if k.get("optional") else "0"))
    elif a.cmd == "steps":
        print(p["setup_steps"])
    elif a.cmd == "warmup":
        ok, msg = warmup(p)
        print(msg)
        return 0 if ok else 1
    return 0


if __name__ == "__main__":
    sys.exit(_main())
