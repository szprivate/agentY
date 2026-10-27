"""Draft a setup recipe for an MCP server that is not in the preset catalog.

``draft_mcp_recipe(source)`` (the agent tool) calls :func:`draft`. What happens here is mechanical and offline once
the facts are in: :func:`gather` reads the public sources (a GitHub repo's README, ``pyproject.toml`` /
``package.json``, the newest commit, PyPI / npm versions); :func:`build_draft` turns the facts into a preset in the
catalog's shape, made of the typed steps of ``scripts/mcp_setup.py``, pinned wherever the source allows it:

* a package on PyPI -> ``uvx pkg==x.y.z`` (steps: uv, entry, test);
* a package on npm  -> ``npx -y pkg@x.y.z`` (node, entry, test);
* a repo with a Python entry point and no PyPI release -> clone at the newest commit, ``uv sync``, ``uv run``;
* anything else -> a skeleton the agent has to fill in, and the validator says what is missing.

The agent then reads the README for what the draft cannot know (an add-on for a desktop app, a key, a port),
adds the steps for it (``install_addon``, ``edit_file``, ``manual``, and a ``run_script`` only when nothing typed
fits), and gets the user's approval through ``install_mcp_preset``. A draft is a proposal, never run as it is.
"""
from __future__ import annotations

import json
import re
import urllib.request
from typing import Callable

import sys  # noqa: E402
from pathlib import Path  # noqa: E402

# Run as a script, ``scripts`` must be THIS checkout's folder: another folder called ``scripts`` earlier
# on the path (a dependency's stray top-level folder, another checkout) would otherwise win the import.
_ROOT_DIR = Path(__file__).resolve().parent.parent
if not sys.path or sys.path[0] != str(_ROOT_DIR):
    sys.path.insert(0, str(_ROOT_DIR))
_pkg = sys.modules.get("scripts")
if _pkg is not None and str(_ROOT_DIR / "scripts") not in [str(Path(p)) for p in (getattr(_pkg, "__path__", None) or [])]:
    del sys.modules["scripts"]

from scripts import mcp_setup as ms

_UA = {"User-Agent": "agentY-recipe-draft"}
README_LIMIT = 14000


def _fetch(url: str, timeout: int = 20) -> str:
    req = urllib.request.Request(url, headers=_UA)
    with urllib.request.urlopen(req, timeout=timeout) as r:   # noqa: S310
        return r.read().decode("utf-8", "replace")


def parse_source(source: str) -> dict:
    """``https://github.com/o/r[/tree/branch/sub/dir]``, ``o/r``, ``npm:pkg``, ``pypi:pkg`` or a bare name."""
    s = str(source or "").strip()
    m = re.match(r"^(?:https?://)?(?:www\.)?github\.com/([^/\s]+)/([^/\s#?]+?)(?:\.git)?(?:/(?:tree|blob)/([^/]+)/?(.*))?$", s)
    if m:
        return {"kind": "github", "owner": m.group(1), "repo": m.group(2), "branch": m.group(3) or "", "subdir": (m.group(4) or "").strip("/")}
    m = re.match(r"^([A-Za-z0-9_.-]+)/([A-Za-z0-9_.-]+)$", s)
    if m:
        return {"kind": "github", "owner": m.group(1), "repo": m.group(2), "branch": "", "subdir": ""}
    if s.startswith("npm:") or s.startswith("@"):
        return {"kind": "npm", "package": s[4:] if s.startswith("npm:") else s}
    if s.startswith("pypi:"):
        return {"kind": "pypi", "package": s[5:]}
    return {"kind": "name", "name": s}


def gather(source: str, fetch: Callable[[str], str] | None = None) -> dict:
    """The public facts about *source*. Every lookup that fails is left out; ``errors`` says which."""
    fetch = fetch or _fetch
    src = parse_source(source)
    facts: dict = {"source": source, "parsed": src, "errors": []}

    def get(url: str) -> str:
        try:
            return fetch(url)
        except Exception as exc:  # noqa: BLE001
            facts["errors"].append(f"{url}: {exc}")
            return ""

    if src["kind"] == "github":
        o, r = src["owner"], src["repo"]
        meta = get(f"https://api.github.com/repos/{o}/{r}")
        try:
            branch = src["branch"] or json.loads(meta).get("default_branch") or "main"
        except ValueError:
            branch = src["branch"] or "main"
        commit = get(f"https://api.github.com/repos/{o}/{r}/commits/{branch}")
        try:
            facts["commit"] = json.loads(commit).get("sha", "")
        except ValueError:
            facts["commit"] = ""
        base = f"https://raw.githubusercontent.com/{o}/{r}/{facts['commit'] or branch}/" + (src["subdir"] + "/" if src["subdir"] else "")
        facts.update(repo_url=f"https://github.com/{o}/{r}.git", branch=branch, base=base)
        facts["readme"] = get(base + "README.md")[:README_LIMIT]
        facts["pyproject"] = get(base + "pyproject.toml")
        facts["package_json"] = get(base + "package.json")
    elif src["kind"] == "npm":
        facts["package_json"] = ""
        facts["npm_package"] = src["package"]
    elif src["kind"] == "pypi":
        facts["pypi_package"] = src["package"]
    else:
        facts["errors"].append("not a GitHub address, npm: or pypi: name; the agent has to search for the server first")
    # package names and versions from the registries
    names: list[tuple[str, str]] = []
    pp = facts.get("pyproject") or ""
    m = re.search(r'^\[project\][^\[]*?^name\s*=\s*"([^"]+)"', pp, re.M | re.S)
    if m:
        names.append(("pypi", m.group(1)))
        facts["pypi_package"] = m.group(1)
        sc = re.search(r"^\[project\.scripts\]\s*\n((?:[^\[\n].*\n?)*)", pp, re.M)
        facts["scripts"] = re.findall(r'^([A-Za-z0-9_.-]+)\s*=', sc.group(1), re.M) if sc else []
    pj = facts.get("package_json") or ""
    if pj:
        try:
            j = json.loads(pj)
            facts["npm_package"] = j.get("name", "")
            facts["npm_bin"] = list(j["bin"]) if isinstance(j.get("bin"), dict) else ([j.get("name")] if j.get("bin") else [])
            facts["npm_version_in_repo"] = j.get("version", "")
        except ValueError:
            pass
    if facts.get("pypi_package"):
        try:
            facts["pypi_version"] = json.loads(get(f"https://pypi.org/pypi/{facts['pypi_package']}/json"))["info"]["version"]
        except (ValueError, KeyError):
            facts["pypi_version"] = ""
    if facts.get("npm_package"):
        try:
            facts["npm_version"] = json.loads(get(f"https://registry.npmjs.org/{facts['npm_package']}/latest")).get("version", "")
        except ValueError:
            facts["npm_version"] = ""
    return facts


def _slug(text: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", str(text).lower()).strip("-")
    s = re.sub(r"^(mcp-server-|server-|mcp-)|(-mcp-server|-mcp|-server)$", "", s) or s
    return s or "server"


def build_draft(facts: dict) -> dict:
    """A preset (catalog shape) from the facts. Unknowns are left as ``TODO`` text the validator or the agent catches."""
    parsed = facts.get("parsed") or {}
    base_name = facts.get("pypi_package") or facts.get("npm_package") or parsed.get("repo") or parsed.get("name") or "server"
    pid = _slug(base_name)
    readme = facts.get("readme", "")
    desc = ""
    for line in readme.splitlines():
        t = line.strip()
        if t and not t.startswith(("#", "<!--", "[", "!", "|", "```")):
            desc = re.sub(r"\s+", " ", t)[:240]
            break
    preset = {"id": pid, "name": str(base_name), "category": "Community", "description": desc or f"MCP server {base_name}",
              "env_keys": [], "requires": [], "params": {}, "notes": "Drafted by the agent from the server's public README; approved by the user."}
    if facts.get("pypi_version") and facts.get("pypi_package"):
        pkg = f"{facts['pypi_package']}=={facts['pypi_version']}"
        script = (facts.get("scripts") or [facts["pypi_package"]])[0]
        args = [pkg] if script == facts["pypi_package"] else ["--from", pkg, script]
        preset["server"] = {"enabled": True, "transport": "stdio", "command": "uvx", "args": args, "auth": "none"}
        preset["requires"] = ["uv"]
        preset["setup"] = [{"id": "uv", "type": "ensure_tool", "tool": "uv"},
                           {"id": "entry", "type": "ensure_entry"},
                           {"id": "test", "type": "test", "needs": ["uv", "entry"]}]
    elif facts.get("npm_version") and facts.get("npm_package"):
        preset["server"] = {"enabled": True, "transport": "stdio", "command": "npx",
                            "args": ["-y", f"{facts['npm_package']}@{facts['npm_version']}"], "auth": "none"}
        preset["requires"] = ["node"]
        preset["setup"] = [{"id": "node", "type": "ensure_tool", "tool": "node"},
                           {"id": "entry", "type": "ensure_entry"},
                           {"id": "test", "type": "test", "needs": ["node", "entry"]}]
    elif facts.get("commit") and facts.get("repo_url") and facts.get("scripts"):
        dest = f"${{BUNDLES}}/{pid}"
        preset["server"] = {"enabled": True, "transport": "stdio", "command": "uv", "args": ["run", facts["scripts"][0]],
                            "auth": "none", "cwd": f"${{AGENTY_ROOT}}/config/mcp_bundles/{pid}"}
        preset["requires"] = ["uv", "git"]
        preset["setup"] = [{"id": "uv", "type": "ensure_tool", "tool": "uv"}, {"id": "git", "type": "ensure_tool", "tool": "git"},
                           {"id": "clone", "type": "git_clone", "url": facts["repo_url"], "commit": facts["commit"], "dest": dest, "needs": ["git"]},
                           {"id": "sync", "type": "run", "cmd": ["uv", "sync"], "cwd": dest, "skip_if_exists": dest + "/.venv", "needs": ["uv", "clone"]},
                           {"id": "entry", "type": "ensure_entry", "needs": ["clone"]},
                           {"id": "test", "type": "test", "needs": ["sync", "entry"]}]
    else:
        preset["server"] = {"enabled": True, "transport": "stdio", "command": "TODO", "args": [], "auth": "none"}
        preset["setup"] = [{"id": "entry", "type": "ensure_entry"}, {"id": "test", "type": "test", "needs": ["entry"]}]
    preset["setup_steps"] = "\n".join(ms.describe_recipe(preset))
    return preset


def draft(source: str, fetch: Callable[[str], str] | None = None, facts: dict | None = None) -> dict:
    """``{draft, validation, hash, facts, how_to_finish, step_types}``: what ``draft_mcp_recipe`` returns."""
    facts = facts or gather(source, fetch)
    d = build_draft(facts)
    errs = ms.validate_recipe(d)
    if d["server"].get("command") == "TODO":
        errs.append("server.command is a TODO: the facts did not say how to start this server; read the README")
    return {
        "draft": d, "validation": {"ok": not errs, "errors": errs}, "hash": ms.recipe_hash(d),
        "facts": {k: v for k, v in facts.items() if k not in ("pyproject", "package_json", "base")},
        "step_types": {k: {"required": sorted(v[0]), "optional": sorted(v[1])} for k, v in ms.STEP_SPECS.items()},
        "how_to_finish": (
            "This is a draft from public facts, not an approved recipe. Read the README (facts.readme) for what it cannot "
            "know: a key (put it in env_keys and as ${VAR} in server.env), a port, an add-on or plug-in for a desktop app "
            "(install_addon / edit_file / manual), a check for the app (find_app). Pin versions and commits. Use run_script "
            "only when no typed step fits, and describe it. Look at the catalog presets (list_mcp_presets, get a recipe with "
            "include_recipe=true) for worked examples. Validate your final recipe by passing it to install_mcp_preset "
            "(recipe=..., dry_run=true) - it returns the same validation - then show the user the step list and wait for a yes."),
    }
