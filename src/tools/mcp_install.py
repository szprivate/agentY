"""The agent adds an MCP server when the user asks for one.

Before this, "install the Remotion MCP" had nowhere to go: the shell sandbox has
no package manager (``npx``/``npm`` are not on its list, on purpose), the
self-extension rules forbid writing ``config/``, and ``mcp.json`` was only read
at startup. :func:`add_mcp_server` is the one sanctioned path, and it is built
from the pieces Settings ▸ "+ Add MCP server" already uses: the paste parser
(:mod:`src.utils.mcp_import`), the connection test (``mcp_tools.test_server``)
and the config / ``.env`` writers.

A new MCP server is code that runs on this machine, so the order is fixed:

1. parse what the user gave (address, start command, README JSON);
2. pin an unpinned ``npx`` / ``uvx`` package to its current version, so what is
   approved is what runs every time;
3. **ask the user** — always, whatever ``security.ask_before_tools`` says, and an
   "allow for this session" never covers it — showing the exact command, the
   arguments and the names of the env keys;
4. connect once and list the tools; nothing is written if that fails;
5. write ``config/mcp.json`` (secrets to ``.env``), connect the live client.

The server's tools are usable at once, without a restart: ``use_mcp_server``
connects any configured server it finds not yet connected.
"""
from __future__ import annotations

import json
import logging
import re
import subprocess
import sys
import urllib.request

from strands import tool
from strands.types.tools import ToolContext

logger = logging.getLogger("agentY.mcp")

TOOL_NAME = "add_mcp_server"
_NPM_RUNNERS = {"npx", "bunx", "pnpx"}
_UV_RUNNERS = {"uvx"}
_VERSIONED = re.compile(r"^(@?[^@\s]+)@([^@\s]+)$")


def _runner(command: str) -> str:
    word = re.split(r"[\\/]", str(command))[-1].lower()
    return re.sub(r"\.(exe|cmd|bat)$", "", word)


def _npm_latest(package: str) -> str:
    out = subprocess.run(["npm.cmd" if sys.platform == "win32" else "npm", "view", package, "version"],
                         capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=30)
    return out.stdout.strip().splitlines()[-1].strip() if out.returncode == 0 and out.stdout.strip() else ""


def _pypi_latest(package: str) -> str:
    with urllib.request.urlopen(f"https://pypi.org/pypi/{package}/json", timeout=15) as r:  # noqa: S310
        return str(json.load(r)["info"]["version"])


def pin_package(command: str, args: list, resolve_npm=None, resolve_pypi=None) -> tuple:
    """Pin the package of an ``npx`` / ``uvx`` command to its current version.

    Returns ``(command, args, notes)``. A package that already names a version is
    left alone; one whose version cannot be looked up stays unpinned, and the note
    says so, because the user is shown it before anything runs. On Windows ``npx``
    becomes ``npx.cmd``, which is what can actually be started without a shell.
    """
    resolve_npm = resolve_npm or _npm_latest
    resolve_pypi = resolve_pypi or _pypi_latest
    args = [str(a) for a in args or []]
    notes: list = []
    runner = _runner(command)
    if runner in _NPM_RUNNERS | _UV_RUNNERS:
        for i, arg in enumerate(args):
            if arg.startswith("-"):
                continue
            if re.match(r"^(https?|git\+|github:|file:|\.|/)", arg):
                notes.append(f"{arg} is not a registry package, so it cannot be pinned.")
                break
            npm = runner in _NPM_RUNNERS
            has_version = ("==" in arg) if not npm else bool(_VERSIONED.match(arg))
            if has_version:
                break
            try:
                version = (resolve_npm if npm else resolve_pypi)(arg)
            except Exception as exc:  # noqa: BLE001
                version = ""
                logger.info("mcp install: could not look up %s: %s", arg, exc)
            if version:
                args[i] = f"{arg}@{version}" if npm else f"{arg}=={version}"
                notes.append(f"Pinned {arg} to {version}.")
            else:
                notes.append(f"Could not look up a version for {arg}: it runs UNPINNED (whatever "
                             "is newest each time).")
            break
        if sys.platform == "win32" and runner in {"npx", "pnpx", "bunx"} \
                and not str(command).lower().endswith((".cmd", ".exe", ".bat")):
            command = f"{command}.cmd"
    return command, args, notes


def describe_entry(name: str, entry: dict, secrets: dict, notes: list) -> str:
    """What the approval prompt shows: exactly what will run, never a summary of it."""
    if entry.get("transport") == "stdio":
        what = "runs on this machine:\n  " + " ".join([entry["command"], *entry.get("args", [])])
    else:
        what = f"connects to {entry.get('url')} ({entry.get('transport')})"
    lines = [f"Add MCP server '{name}' — it {what}"]
    if entry.get("cwd"):
        lines.append(f"  in folder {entry['cwd']}")
    env_keys = sorted(entry.get("env") or {})
    if env_keys:
        lines.append("  env keys: " + ", ".join(env_keys))
    if secrets:
        lines.append("  keys saved to .env: " + ", ".join(sorted(secrets)))
    lines += ["  " + n for n in notes]
    lines.append("  MCP servers run code with your user's permissions (files, network, packages).")
    return "\n".join(lines)


def _failure(error: str, **extra) -> str:
    return json.dumps({"ok": False, "error": error, **extra}, ensure_ascii=False)


def install(definition: str, name: str = "", *, ask=None, loader=None) -> dict:
    """The whole flow without the agent around it. *ask(summary, payload)* returns a
    Decision-like object; *loader(name)* connects the saved server."""
    from src.tools import mcp_tools as mt
    from src.utils.mcp_import import McpImportError, parse_mcp_snippet

    cfg = mt.load_mcp_config()
    servers = cfg.get("servers") or {}
    try:
        parsed = parse_mcp_snippet(definition, existing=servers)
    except McpImportError as exc:
        return {"ok": False, "error": str(exc)}
    if len(parsed["servers"]) != 1:
        return {"ok": False, "error": "Give one server at a time; that was "
                f"{len(parsed['servers'])}: {', '.join(parsed['servers'])}."}
    (auto_name, entry), = parsed["servers"].items()
    secrets = dict(parsed["secrets"])
    if parsed["missing"]:
        return {"ok": False, "missing": parsed["missing"],
                "error": "It needs " + ", ".join(parsed["missing"]) + ". Ask the user for the "
                         "value(s) (or to enter them in Settings ▸ MCP servers), then call again "
                         "with the value in the definition's env."}
    name = re.sub(r"[^a-z0-9]+", "_", (name or "").lower()).strip("_") if name else ""
    if name:
        if name in servers:
            return {"ok": False, "error": f"A server named {name!r} already exists."}
    else:
        name = auto_name
    notes = list(parsed["notes"])
    if entry.get("transport") == "stdio":
        entry["command"], entry["args"], more = pin_package(entry["command"], entry.get("args", []))
        notes += more

    summary = describe_entry(name, entry, secrets, notes)
    decision = (ask or _ask)(summary, {"name": name, "server": entry})
    if not decision.allowed:
        return {"ok": False, "declined": True,
                "error": f"Not added — {decision.reason} Do not retry; ask the user what they "
                         "would like instead."}

    result = mt.test_server(name, entry, secrets=secrets)
    if not result.get("ok"):
        return {"ok": False, "tested": False, "error": "The server did not start, so nothing was "
                f"saved: {result.get('error')}", "needs_auth": result.get("needs_auth", False)}

    if secrets:
        from src.utils.agentY_server import _update_env_file
        _update_env_file(secrets)
    servers[name] = entry
    mt.save_mcp_config({"servers": servers})
    out = {"ok": True, "name": name, "tools": result["tools"], "tool_names": result["names"],
           "notes": notes}
    try:
        (loader or _connect_live)(name)
        out["connected"] = True
    except Exception as exc:  # noqa: BLE001
        out["connected"] = False
        out["restart_needed"] = f"Saved, but it could not be connected live ({exc}); restart the agent."
    return out


def _ask(summary: str, payload: dict):
    from src.utils import tool_permissions as tp
    return tp.request(TOOL_NAME, {"summary": summary}, unattended_allows=False, allow_session=False)


def _connect_live(name: str) -> None:
    from src.tools import mcp_on_demand as od
    od.server_tools(name)


@tool(name=TOOL_NAME, context=True)
def add_mcp_server(definition: str, tool_context: ToolContext, name: str = "") -> str:
    """Install a new external MCP server when the user asks for one.

    Use it for "install / add / connect the X MCP". Do NOT use run_script, npm or
    hand-edit config for this. It shows the user exactly what will run and waits
    for their approval, tests the server, saves it and connects it, so its tools
    work right away.

    Args:
        definition: What to add: the server's start command (`npx -y some-mcp`), its
            https address, or the JSON block from its README. Put a key the user gave
            you in its env / headers; ask for it if the server needs one and you have
            none. npx/uvx packages are pinned to their current version.
        name: Optional short name, e.g. "remotion". Defaults to one derived from the package.
    """
    try:
        out = install(definition, name)
    except Exception as exc:  # noqa: BLE001
        logger.warning("add_mcp_server failed: %s", exc, exc_info=True)
        return _failure(str(exc))
    if out.get("ok"):
        try:
            from src.tools import mcp_on_demand as od
            if od.enabled():
                out["next"] = (f"Call use_mcp_server('{out['name']}') now; its tools are then named "
                               f"{out['name']}__<tool>.")
            else:
                od.load(tool_context.agent, out["name"])
                out["next"] = "Its tools are in your list now."
        except Exception as exc:  # noqa: BLE001
            out["next"] = f"Saved and connected; call use_mcp_server('{out['name']}') ({exc})."
    return json.dumps(out, ensure_ascii=False)
