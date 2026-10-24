"""The agent installs an MCP preset the way the Settings Install button does.

"Install the Blender MCP" in the chat is one call: :func:`install_mcp_preset`. It is the same code as the
Settings route (``src.utils.mcp_preset_install.install_preset``): keys checked, the preset's recipe run
(``scripts/mcp_setup.py``: bundle, entry, app add-on, ...), the connection tested; the tools are then in the
agent's list in the same turn (:func:`src.tools.mcp_on_demand.load`).

Why these live in agentY and not in ``agenty_core``: they act on THIS host's ``config/mcp.json``, its live MCP
clients and the running agent's tool list. What is general is in core: ``find_app_installs``.

A server that is NOT in the catalog gets a recipe the agent drafts (:func:`draft_mcp_recipe`). Nothing of it runs
before the user has seen the step list and said yes in their own words (``approval_gate``, the same gate the LoRA
dataset review has). An approved recipe is saved to ``config/mcp_presets.local.json`` and from then on is a preset
like the others; a recipe edited after approval has a new hash and has to be approved again.
"""
from __future__ import annotations

import json
import logging
import os
import tempfile
import urllib.request
from pathlib import Path

from strands import tool
from strands.types.tools import ToolContext

from scripts import mcp_draft as md
from scripts import mcp_presets as mp
from scripts import mcp_setup as ms

logger = logging.getLogger("agentY.mcp")


def _host() -> dict:
    """The host's own functions, the ones the Settings route passes in."""
    from src.tools import mcp_tools as mt
    from src.utils.agentY_server import _update_env_file
    return dict(load_config=mt.load_mcp_config, save_config=mt.save_mcp_config, update_env=_update_env_file,
                test_server=mt.test_server, get_env=lambda k: os.environ.get(k, ""))


def _json(obj) -> str:
    return json.dumps(obj, ensure_ascii=False, default=str)


def preset_list(include_recipe: bool = False, *, configured: dict | None = None) -> list[dict]:
    if configured is None:
        from src.tools import mcp_tools as mt
        configured = mt.load_mcp_config().get("servers") or {}
    out = []
    for p in mp.load_catalog():
        row = {"id": p["id"], "name": p["name"], "category": p["category"], "description": p["description"],
               "installed": p["id"] in configured, "has_setup": bool(p.get("setup")), "local": bool(p.get("local")),
               "keys_needed": [k["key"] for k in p["env_keys"] if not k.get("optional")],
               "requires": p["requires"]}
        if p.get("local"):
            row["approved"] = ms.is_approved(p)
        if include_recipe:
            row["recipe"] = {"params": p.get("params"), "setup": p.get("setup"), "server": p["server"]}
        out.append(row)
    return out


def run_install(preset_id: str = "", *, recipe: str = "", user_message: str = "", run_setup: bool = True,
                dry_run: bool = False, recheck: bool = False, allow_system_installs: bool = False,
                launch_app: bool = False, project: str = "", host: dict | None = None, pending_store: Path | None = None,
                local_path: Path | None = None, config_path=None, root=None, overrides: dict | None = None,
                progress=None) -> dict:
    """The body of ``install_mcp_preset`` / ``setup_mcp_preset`` without the agent around it."""
    from src.utils import mcp_preset_install as mpi
    host = host or _host()
    preset = None
    if str(recipe or "").strip():
        try:
            draft = json.loads(recipe) if isinstance(recipe, str) else recipe
        except ValueError as exc:
            return {"ok": False, "status": "invalid", "errors": [f"the recipe is not valid JSON: {exc}"]}
        n = ms.normalise_draft(draft)
        if n.get("id") and any(p["id"] == n["id"] and not p.get("local") for p in mp.load_catalog()):
            return {"ok": False, "status": "invalid", "errors": [f"'{n['id']}' is a catalog preset; call install_mcp_preset(preset_id='{n['id']}')"]}
        errs = ms.validate_recipe(n)
        if errs:
            return {"ok": False, "status": "invalid", "errors": errs}
        if dry_run:
            return {"ok": True, "status": "dry_run", "valid": True, "hash": ms.recipe_hash(n), "summary": ms.describe_recipe(n),
                    "run_script_steps": [i + 1 for i, s in enumerate(n["setup"]) if s["type"] == "run_script"]}
        gate = ms.approval_gate(n, user_message, store=pending_store)
        if gate["status"] != "approved":
            return {"ok": True, **gate}
        ms.save_local_preset(n, gate["hash"], local_path)
        preset = mp.find(n["id"], local_path=local_path) if local_path else mp.find(n["id"])
        if preset is None:   # a catalog path was given but not the local one: build it directly
            preset = {**n, "local": True, "approved_hash": gate["hash"]}
        preset_id = n["id"]
        recheck = recheck or preset_id in (host["load_config"]().get("servers") or {})
    elif not preset_id:
        return {"ok": False, "error": "give preset_id (see list_mcp_presets) or a recipe"}
    ov = dict(overrides or {})
    if project:
        ov["params"] = {**(ov.get("params") or {}), "project": project}
    payload, status = mpi.install_preset(
        preset_id, {}, run_setup=run_setup, dry_run=dry_run, recheck=recheck, preset=preset, progress=progress,
        allow_system=allow_system_installs, launch=launch_app, overrides=ov or None, config_path=config_path, root=root,
        **host)
    payload.pop("_report", None)
    payload.pop("_given", None)
    payload["http_status"] = status
    if payload.get("report"):
        payload["report_text"] = ms.format_report(payload["report"])
    return payload


def _after(payload: dict, preset_id: str, tool_context) -> dict:
    """Put the new server's tools in the running agent's list when it connected."""
    if payload.get("ok") and (payload.get("test") or {}).get("ok") and tool_context is not None:
        try:
            from src.tools import mcp_on_demand as od
            names = od.load(tool_context.agent, preset_id)
            payload["tools_loaded"] = len(names)
            payload["next"] = (f"{len(names)} tool(s) of {preset_id} are in your list now, named {preset_id}__<tool>. "
                               "Call them directly.")
        except Exception as exc:  # noqa: BLE001
            payload["next"] = f"Connected; call use_mcp_server('{preset_id}') to load its tools ({exc})."
    if payload.get("needs_user"):
        payload["tell_the_user"] = payload["needs_user"]
    return payload


@tool(context=True)
def install_mcp_preset(preset_id: str = "", tool_context: ToolContext = None, recipe: str = "", user_message: str = "",
                       run_setup: bool = True, dry_run: bool = False, allow_system_installs: bool = False,
                       launch_app: bool = False, project: str = "") -> str:
    """Install an MCP server for the user and set it up completely: what the Settings "Install" button does.

    Use it when the user asks to install or add an MCP ("install the Blender MCP"). Look at `list_mcp_presets`
    first; a catalog preset is one call. It writes the server into the config, runs the preset's setup (unpacks
    the bundle, installs the add-on into the app, edits its preferences), tests the connection, and puts the
    server's tools in your list. The result has `report_text` (each step: done / skipped / needs_user / failed)
    and `needs_user` (what only the user can do, say it plainly). A missing API key comes back as `needs_user`
    naming the .env key: the user enters it in Settings > MCP servers; never ask for it in chat to write it
    yourself. Steps that failed: read the report, fix what you can with your other tools, or tell the user.

    A server that is NOT in the catalog: call `draft_mcp_recipe(source)`, finish the draft, then pass it as
    `recipe` (a JSON string). First with dry_run=true (validation + the step list). Without dry_run the first
    call returns status `waiting_for_user` with the step list: show it to the user (every run_script in
    full), END YOUR TURN and wait. When they say yes in their own words, call again with the same recipe and
    `user_message` = their words. Never approve for them. A changed recipe needs a new yes.

    Args:
        preset_id: The catalog id, e.g. "blender". Leave empty when you pass `recipe`.
        recipe: A drafted preset (JSON) for a server that is not in the catalog.
        user_message: The user's own words approving the recipe (second call only).
        run_setup: False writes only the config entry and tests it.
        dry_run: Show what would happen; change nothing.
        allow_system_installs: Let the setup install missing uv / Node / git with winget. Only after the user agreed.
        launch_app: Let the setup start the app and wait for its port. Only if the user asked for it.
        project: For Unity: the project folder to add the package to.
    """
    try:
        out = run_install(preset_id, recipe=recipe, user_message=user_message, run_setup=run_setup, dry_run=dry_run,
                          allow_system_installs=allow_system_installs, launch_app=launch_app, project=project)
    except Exception as exc:  # noqa: BLE001
        logger.warning("install_mcp_preset failed: %s", exc, exc_info=True)
        return _json({"ok": False, "error": str(exc)})
    pid = preset_id or (json.loads(recipe).get("id") if recipe.strip().startswith("{") else "")
    return _json(_after(out, pid, tool_context))


@tool(context=True)
def setup_mcp_preset(preset_id: str, tool_context: ToolContext = None, dry_run: bool = False, only_steps: str = "",
                     allow_system_installs: bool = False, launch_app: bool = False, project: str = "") -> str:
    """Run (or re-check) the setup of an MCP preset that is already installed: the same runner as the
    Settings "Run setup / Re-check" button. Every step looks first and skips what is already done, so it is
    safe to repeat. Use it when a server's app is missing its add-on, after the user installed the app, or to
    see what is left (`needs_user`). For a new server use `install_mcp_preset`.

    Args:
        preset_id: The preset id (catalog or an approved local one).
        dry_run: Report what would be done without changing anything.
        only_steps: Comma-separated step ids to run (the rest are skipped), e.g. "addon,test".
        allow_system_installs: Let the setup install missing uv / Node / git with winget. Only after the user agreed.
        launch_app: Let the setup start the app and wait for its port. Only if the user asked for it.
        project: For Unity: the project folder to add the package to.
    """
    try:
        steps = [s.strip() for s in only_steps.split(",") if s.strip()]
        from src.utils import mcp_preset_install as mpi
        host = _host()
        ov = {"params": {"project": project}} if project else None
        payload, status = mpi.install_preset(preset_id, {}, recheck=True, dry_run=dry_run, allow_system=allow_system_installs,
                                             launch=launch_app, overrides=ov, only_steps=steps or None, **host)
        payload.pop("_report", None)
        payload.pop("_given", None)
        if payload.get("report"):
            payload["report_text"] = ms.format_report(payload["report"])
    except Exception as exc:  # noqa: BLE001
        logger.warning("setup_mcp_preset failed: %s", exc, exc_info=True)
        return _json({"ok": False, "error": str(exc)})
    return _json(_after(payload, preset_id, tool_context))


@tool
def list_mcp_presets(include_recipe: bool = False) -> str:
    """The MCP servers agentY can install and set up by itself (the preset catalog, plus recipes the user
    approved earlier): id, what it is, whether it is installed, whether it has a setup recipe, and the .env keys
    it needs. Look here first when the user asks to install an MCP; `list_mcp_servers` shows what is configured
    and connected. Not in the list: `draft_mcp_recipe`.

    Args:
        include_recipe: Also return each preset's recipe (worked examples for drafting a new one).
    """
    return _json({"presets": preset_list(include_recipe)})


@tool
def draft_mcp_recipe(source: str) -> str:
    """Draft the setup recipe of an MCP server that is not in the preset catalog.

    Give a GitHub URL, `owner/repo`, `npm:package` or `pypi:package` (search the web for the server first when the
    user only named it). It reads the public README, the package files and the newest commit and returns a `draft`
    (a preset made of typed, idempotent steps, pinned to a version or commit), the `validation` of it, and the
    README text. Finish the draft yourself from the README (an app add-on, a key, a port), then call
    `install_mcp_preset(recipe=..., dry_run=True)` to validate, and again without dry_run to get the step list for the user.
    Nothing runs from here.

    Args:
        source: GitHub URL, owner/repo, npm:package or pypi:package.
    """
    try:
        return _json({"ok": True, **md.draft(source)})
    except Exception as exc:  # noqa: BLE001
        logger.warning("draft_mcp_recipe failed: %s", exc, exc_info=True)
        return _json({"ok": False, "error": str(exc)})


def _ask(summary: str):
    from src.utils import tool_permissions as tp
    return tp.request("install_mcp_bundle", {"summary": summary}, unattended_allows=False, allow_session=False)


def run_bundle(url_or_path: str, *, name: str = "", ask=None, host: dict | None = None, bundles_root: Path | None = None) -> dict:
    """Unpack a .mcpb (a URL or a local path) as an agentY MCP server, add its entry, test it."""
    from src.utils import mcp_bundle as mb
    host = host or _host()
    src = str(url_or_path or "").strip().strip('"')
    if not src:
        return {"ok": False, "error": "give a .mcpb URL or a local path"}
    old_root = mb.BUNDLES_ROOT
    if bundles_root:
        mb.BUNDLES_ROOT = Path(bundles_root)
    token = None
    try:
        if src.lower().startswith(("http://", "https://")):
            tmp = Path(tempfile.mkdtemp(prefix="agenty-mcpb-")) / "bundle.mcpb"
            req = urllib.request.Request(src, headers={"User-Agent": "agentY-bundle"})
            with urllib.request.urlopen(req, timeout=60) as r, open(tmp, "wb") as f:   # noqa: S310
                f.write(r.read(mb.MAX_UPLOAD_BYTES + 1))
            path = tmp
        else:
            path = Path(src)
            if not path.is_file():
                return {"ok": False, "error": f"{path} is not a file"}
        with open(path, "rb") as f:
            token = mb.stage_upload(f, path.name)
        servers = host["load_config"]().get("servers") or {}
        info = mb.inspect(token, servers, ())
        if info["problems"]:
            return {"ok": False, "needs_user": info["problems"], "error": "this machine cannot run the bundle"}
        wanted = name or info["name"]
        required = [f for f in info["user_config"] if f["required"] and f.get("default") in (None, "", [])]
        if required:
            return {"ok": False, "status": "needs_user",
                    "needs_user": [f"The bundle asks for: {', '.join(f['title'] for f in required)}. Add it in Settings > MCP servers > Install bundle (.mcpb)."]}
        b = info["bundle"]
        summary = (f"Install MCP bundle '{b['display_name']}' {b['version']} by {b['author'] or 'unknown'} as server '{wanted}'.\n"
                   f"  It runs on this machine: {b['command']}\n  {'signed (not verified)' if b['signed'] else 'unsigned'}; "
                   "MCP servers run code with your user's permissions.")
        decision = (ask or _ask)(summary)
        if not decision.allowed:
            return {"ok": False, "declined": True, "error": f"Not installed - {decision.reason} Do not retry; ask the user."}
        res = mb.install(token, wanted, {})
        token = None
        cfg = host["load_config"]()
        servers = dict(cfg.get("servers") or {})
        servers[wanted] = res["server"]
        host["save_config"]({"servers": servers})
        test = host["test_server"](wanted, res["server"], {})
        return {"ok": bool(test.get("ok")), "name": wanted, "test": test, "warnings": res.get("warnings", []),
                "bundle": b["display_name"], "version": b["version"]}
    except mb.BundleError as exc:
        return {"ok": False, "error": str(exc)}
    finally:
        if token:
            mb.discard(token)
        mb.BUNDLES_ROOT = old_root


@tool(context=True)
def install_mcp_bundle(url_or_path: str, tool_context: ToolContext = None, name: str = "") -> str:
    """Install an MCP bundle (.mcpb, formerly .dxt) as a server: unpack it, add it to the config, test it.
    The same as Settings > MCP servers > Install bundle. It asks the user first (it is third-party code).
    For a server that has a catalog preset, use `install_mcp_preset` instead: it knows the bundle URL.

    Args:
        url_or_path: An https URL of a .mcpb file or a local path.
        name: Optional server name (lowercase letters, digits, _).
    """
    try:
        out = run_bundle(url_or_path, name=name)
    except Exception as exc:  # noqa: BLE001
        logger.warning("install_mcp_bundle failed: %s", exc, exc_info=True)
        return _json({"ok": False, "error": str(exc)})
    return _json(_after(out, out.get("name", ""), tool_context))
