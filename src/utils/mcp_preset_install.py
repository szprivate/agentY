"""Installing an MCP preset from Settings: the catalog entry in, a tested server out.

The host's routes (``GET /agentY/mcp/presets``, ``POST /agentY/mcp/presets/install``)
call into here. The catalog and the merge are ``scripts/mcp_presets.py``, the same code
the installers run. What a route adds is the live part: keys into .env, a connection
test, and, when something outside config/mcp.json is left to do (setup steps, a bundle,
a failed test that is not a missing sign-in), an agentY turn that finishes it.
"""
from __future__ import annotations

import logging
import threading
import time
import uuid
import re

from scripts import mcp_presets as mp
from scripts import mcp_setup as ms

logger = logging.getLogger(__name__)

_ENV_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _partial() -> str:
    try:
        from src.pipeline import _orch_partial
        return _orch_partial("mcp_preset_install")
    except Exception:  # noqa: BLE001
        return ""


def catalog(configured: dict) -> list[dict]:
    """The presets with ``installed`` (the id is already a server), ``needs_bundle`` and ``has_setup``
    (a recipe the host can run: Settings shows Run setup / Re-check on an installed one)."""
    out = []
    for p in mp.load_catalog():
        out.append({**p, "installed": p["id"] in configured,
                    "needs_bundle": mp.needs_bundle(mp.server_entry(p)),
                    "has_setup": bool(p.get("setup")), "approved": ms.is_approved(p)})
    return out


def turn_message(preset: dict, done: str, test: dict | None, report: dict | None = None) -> str:
    """What the finishing turn is asked, from the prompt partial."""
    if test is None:
        said = "not run (the server is not in config/mcp.json yet)"
    elif test.get("ok"):
        said = f"connected, {test.get('tools', 0)} tools"
    else:
        said = str(test.get("error") or "failed")
    text = _partial() or ("Install and finish MCP preset {name} (id {id}). Done so far: {done}\n"
                          "Steps:\n{steps}\nSetup report:\n{report}\nLast test: {test}\n"
                          "Test until connected and tell the user.")
    return (text.replace("{name}", preset["name"]).replace("{id}", preset["id"])
            .replace("{done}", done).replace("{steps}", preset["setup_steps"] or "(none)")
            .replace("{report}", ms.format_report(ms.public(report)) if report else "(no recipe ran)")
            .replace("{test}", said))


def _test_of(report: dict) -> dict | None:
    for st in report.get("steps", []):
        if st["type"] == "test" and st["status"] in (ms.DONE, ms.FAILED, ms.NEEDS_USER):
            d = st.get("detail") or {}
            if st["status"] == ms.DONE:
                return {"ok": True, "tools": d.get("tools", 0), "names": d.get("names", [])}
            return {"ok": False, "error": st["message"], "needs_auth": st["status"] == ms.NEEDS_USER}
    return None


def install_preset(preset_id: str, secrets: dict | None = None, *, run_setup: bool = True, dry_run: bool = False,
                   recheck: bool = False, preset: dict | None = None, load_config, save_config, update_env,
                   test_server, get_env=lambda k: "", progress=None, interactive: bool = False,
                   allow_system: bool = False, launch: bool = False, overrides: dict | None = None,
                   config_path=None, root=None, only_steps=None, **ctx_extra) -> tuple[dict, int]:
    """Everything the Install button, Re-check and the agent's ``install_mcp_preset`` do - one code path.

    keys to .env -> the recipe (entry, bundle, add-on, ...) -> the connection test. A key that is missing is
    ``needs_user`` naming the .env key (nothing is written for it). ``recheck`` runs the recipe of a preset that is
    already installed. ``dry_run`` writes nothing at all (not even keys). Returns ``(payload, http status)``;
    the payload has ``{ok, status, state, report, needs_user, test, ...}``.
    """
    preset = preset or mp.find(preset_id)
    if preset is None:
        return {"ok": False, "error": f"unknown preset '{preset_id}'"}, 404
    if preset.get("local") and not ms.is_approved(preset):
        return {"ok": False, "status": "failed", "error": "this local preset's recipe was not approved, or it was changed "
                "after it was approved; it has to be shown to the user and approved again"}, 403
    cfg = load_config()
    installed = preset_id in (cfg.get("servers") or {})
    if installed and not recheck:
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
        where = "see the server's page"
        return {"ok": False, "status": "needs_user", "error": "missing: " + ", ".join(missing), "missing": missing,
                "needs_user": [f"The .env key {k} ({allowed[k].get('label') or k}) is not set. The user adds it in "
                               f"agentY Settings (MCP servers) or in .env; {allowed[k].get('help_url') or where} says where to get one."
                               for k in missing]}, 400
    if given and not dry_run:
        update_env(given)
    has_recipe = bool(preset.get("setup"))
    out: dict = {"ok": True, "state": "installed" if installed else "pending", "test": None, "turn_id": None,
                 "conversation": None, "has_recipe": has_recipe}
    if has_recipe and run_setup:
        report = ms.run_setup(preset_id, preset=preset, dry_run=dry_run, interactive=interactive,
                              allow_system=allow_system, launch=launch, overrides=overrides,
                              load_config=load_config, save_config=save_config, test_server=test_server,
                              progress=progress, config_path=config_path, root=root, only_steps=only_steps, **ctx_extra)
        pub = ms.public(report)
        servers = (load_config().get("servers") or {})
        out.update(report=pub, needs_user=pub["needs_user"], test=_test_of(pub),
                   state="added" if (preset_id in servers and not installed) else ("installed" if installed else "pending"),
                   status=("dry_run" if dry_run else "failed" if not pub["ok"] else "needs_user" if pub["needs_user"] else "done"),
                   ok=pub["ok"])
        out["_report"] = report
        return out, 200
    # No recipe (or the caller only wants the entry): merge it, test it once.
    merged, states = mp.merge(cfg, [preset])
    state = states[preset_id]
    entry = merged["servers"].get(preset_id)
    test = None
    if state == "added" and not dry_run:
        save_config({"servers": merged["servers"]})
        try:
            test = test_server(preset_id, entry, given)
        except Exception as exc:  # noqa: BLE001
            logger.warning("mcp preset test failed to run: %s", exc)
            test = {"ok": False, "error": str(exc), "needs_auth": False}
    status = ("dry_run" if dry_run else "needs_user" if state == "needs_bundle"
              else "done" if (test is None or test.get("ok")) else "failed")
    out.update(state=state, test=test, report=None, needs_user=[], status=status)
    out["_given"] = bool(given)
    return out, 200


def install(preset_id: str, secrets: dict, *, load_config, save_config, update_env,
            test_server, start_turn, get_env=lambda k: "", progress=None, **more) -> tuple[dict, int]:
    """What the Settings Install button asks: :func:`install_preset`, then an agentY turn only for what the
    deterministic setup could not finish (a step that failed, a preset without a recipe that has setup text,
    a bundle that is not installed, a failure that is not "give a key / sign in")."""
    payload, status = install_preset(preset_id, secrets, load_config=load_config, save_config=save_config,
                                     update_env=update_env, test_server=test_server, get_env=get_env,
                                     progress=progress, **more)
    report = payload.pop("_report", None)
    given = payload.pop("_given", False)
    if status != 200 or payload.get("status") == "dry_run":
        return payload, status
    preset = mp.find(preset_id)
    test = payload.get("test")
    state = payload.get("state")
    if payload.get("has_recipe") and report is not None:
        wants_turn = bool(payload["report"]["failed"])
        done = "the host ran the setup recipe (report below)"
    else:
        done = ("keys saved to .env, entry written to config/mcp.json" if given else "entry written to config/mcp.json") \
            if state == "added" else "nothing yet: the server's .mcpb bundle is not installed (config/mcp_bundles)"
        wants_turn = bool(preset["setup_steps"]) or state == "needs_bundle" or (
            test is not None and not test.get("ok") and not test.get("needs_auth"))
    if wants_turn:
        try:
            rid, tid = start_turn(turn_message(preset, done, test, payload.get("report")), [], "",
                                  f"MCP preset: {preset['name']}")
            payload.update(turn_id=rid, conversation=tid)
        except ValueError as exc:
            payload["turn_error"] = str(exc)
    return payload, 200


# ── jobs: the setup of a preset can take minutes (downloads, uv sync, an app starting) ─────────────────────────

_JOBS: dict[str, dict] = {}
_JOBS_LOCK = threading.Lock()


def start_job(run, *, label: str = "") -> str:
    """Run ``run(progress)`` on a thread and return a job id. *run* returns ``(payload, status)``;
    ``progress(step_row)`` is called after each step. Poll with :func:`job`."""
    jid = uuid.uuid4().hex[:12]
    rec = {"id": jid, "label": label, "running": True, "steps": [], "result": None, "status": None, "started": time.time()}
    with _JOBS_LOCK:
        for old in sorted(_JOBS.values(), key=lambda j: j["started"])[:-19]:
            _JOBS.pop(old["id"], None)
        _JOBS[jid] = rec

    def work():
        try:
            payload, status = run(lambda row: rec["steps"].append(row))
            rec.update(result=payload, status=status)
        except Exception as exc:  # noqa: BLE001
            logger.warning("mcp preset job failed: %s", exc, exc_info=True)
            rec.update(result={"ok": False, "error": str(exc)}, status=500)
        finally:
            rec["running"] = False

    threading.Thread(target=work, name=f"mcp-preset-{jid}", daemon=True).start()
    return jid


def job(jid: str) -> dict | None:
    rec = _JOBS.get(jid)
    if rec is None:
        return None
    return {"ok": True, "id": rec["id"], "label": rec["label"], "running": rec["running"],
            "steps": list(rec["steps"]), "result": rec["result"], "status": rec["status"]}
