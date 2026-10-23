"""Installing an MCP preset from Settings: the catalog entry in, a tested server out.

The host's routes (``GET /agentY/mcp/presets``, ``POST /agentY/mcp/presets/install``)
call into here. The catalog and the merge are ``scripts/mcp_presets.py``, the same code
the installers run. What a route adds is the live part: keys into .env, a connection
test, and, when something outside config/mcp.json is left to do (setup steps, a bundle,
a failed test that is not a missing sign-in), an agentY turn that finishes it.
"""
from __future__ import annotations

import logging
import re

from scripts import mcp_presets as mp

logger = logging.getLogger(__name__)

_ENV_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _partial() -> str:
    try:
        from src.pipeline import _orch_partial
        return _orch_partial("mcp_preset_install")
    except Exception:  # noqa: BLE001
        return ""


def catalog(configured: dict) -> list[dict]:
    """The presets with ``installed`` (the id is already a server) and ``needs_bundle``."""
    out = []
    for p in mp.load_catalog():
        out.append({**p, "installed": p["id"] in configured,
                    "needs_bundle": mp.needs_bundle(mp.server_entry(p))})
    return out


def turn_message(preset: dict, done: str, test: dict | None) -> str:
    """What the finishing turn is asked, from the prompt partial."""
    if test is None:
        said = "not run (the server is not in config/mcp.json yet)"
    elif test.get("ok"):
        said = f"connected, {test.get('tools', 0)} tools"
    else:
        said = str(test.get("error") or "failed")
    text = _partial() or ("Install and finish MCP preset {name} (id {id}). Done so far: {done}\n"
                          "Steps:\n{steps}\nLast test: {test}\nTest until connected and tell the user.")
    return (text.replace("{name}", preset["name"]).replace("{id}", preset["id"])
            .replace("{done}", done).replace("{steps}", preset["setup_steps"] or "(none)")
            .replace("{test}", said))


def install(preset_id: str, secrets: dict, *, load_config, save_config, update_env,
            test_server, start_turn, get_env=lambda k: "") -> tuple[dict, int]:
    """Do everything the Install button asks. The host passes its own functions in
    (config load/save, the .env writer, the connection test, the turn starter), so this
    is the same path the Settings save takes and a test can pass stand-ins.

    Returns ``(payload, http status)``; the payload is ``{ok, state, test, turn_id,
    conversation, ...}``.
    """
    preset = mp.find(preset_id)
    if preset is None:
        return {"ok": False, "error": f"unknown preset '{preset_id}'"}, 404
    cfg = load_config()
    if preset_id in (cfg.get("servers") or {}):
        return {"ok": False, "error": f"'{preset_id}' is already configured"}, 409
    allowed = {k["key"]: k for k in preset["env_keys"]}
    given = {}
    for k, v in (secrets or {}).items():
        if k not in allowed or not _ENV_NAME.match(str(k)):
            return {"ok": False, "error": f"'{k}' is not a key of this preset"}, 400
        if str(v or "").strip():
            given[str(k)] = str(v).strip()
    missing = [k for k, meta in allowed.items()
               if not meta.get("optional") and not given.get(k) and not get_env(k)]
    if missing:
        return {"ok": False, "error": "missing: " + ", ".join(missing), "missing": missing}, 400
    if given:
        update_env(given)
    merged, states = mp.merge(cfg, [preset])
    state = states[preset_id]
    entry = merged["servers"].get(preset_id)
    test = None
    if state == "added":
        save_config({"servers": merged["servers"]})
        try:
            test = test_server(preset_id, entry, given)
        except Exception as exc:  # noqa: BLE001
            logger.warning("mcp preset test failed to run: %s", exc)
            test = {"ok": False, "error": str(exc), "needs_auth": False}
        done = ("keys saved to .env, entry written to config/mcp.json"
                if given else "entry written to config/mcp.json")
    else:   # needs_bundle: nothing written, the agent installs the bundle first
        done = "nothing yet: the server's .mcpb bundle is not installed (config/mcp_bundles)"
    # A turn only where something outside the config is left: setup steps, a missing
    # bundle, a failure that is not "sign in / give a key" (those are the user's).
    wants_turn = bool(preset["setup_steps"]) or state == "needs_bundle" or (
        test is not None and not test.get("ok") and not test.get("needs_auth"))
    out = {"ok": True, "state": state, "test": test, "turn_id": None, "conversation": None}
    if wants_turn:
        try:
            rid, tid = start_turn(turn_message(preset, done, test), [], "",
                                  f"MCP preset: {preset['name']}")
            out.update(turn_id=rid, conversation=tid)
        except ValueError as exc:
            out["turn_error"] = str(exc)
    return out, 200
