"""The orchestrator can change its own model, and a rule can tell it when to ask.

Two pieces, and neither makes a model call of its own:

* ``list_models`` / ``switch_model`` (src/tools/model_switching.py) are tools the
  orchestrator calls when it sees fit. The logic is here so it can be tested
  without an agent.
* A deterministic rule check (:class:`ModelRoutingHook`). Settings
  ``llm.routing.rules`` say "when skill X is loaded / tool pack Y is loaded / tool
  Z is called, model M is the better seat". The hook reads the tool call that just
  finished (its NAME and input), compares it to the rules, and appends one line to
  that tool's result. That is a dictionary lookup on a string the host already
  has: a per-message classifier call was built and reverted (agentY 1e05c1f,
  edea633) because it added 4-8 s to every turn.

**The seat swap.** Only ``Agent.model`` is replaced; Strands reads it per model
call, so the conversation, tools, hooks, skills and MCP tools all stay. The cost
metadata moves with it, or the new model's tokens would bill at the old rate.

**The vision trap.** The panel decides whether to embed image bytes BEFORE a turn,
from the orchestrator's seat, and a text-only model that has an image block in its
history rejects every later turn ("Unexpected item type in content"). So a
conversation that carries images is never moved to a text-only model, and a
conversation override is what ``_orchestrator_supports_vision`` consults.

**Scope.** ``conversation`` is kept per thread id and re-applied each time the
pooled pipeline loads that conversation (``sync_thread``), so another
conversation that lands on the same pipeline is not dragged along. It lasts until
the host restarts or ``model="default"``. ``settings`` writes
``settings.local.json`` through the same call as ``/switch_model``.
"""

from __future__ import annotations

import difflib
import json
import logging
import threading
import time
from typing import Any, Callable

logger = logging.getLogger("agentY.routing")

_LOCK = threading.RLock()
# thread id -> the 'provider,model' that conversation was moved to.
_OVERRIDES: dict[str, str] = {}
# (thread id, rule key): a rule says its piece once per conversation, never again.
_NOTIFIED: set = set()
# What was switched, newest last; kept for tests and for the panel's curiosity.
RECENT: list[dict] = []

# Roles whose job is reading images: a text-only model is never a candidate.
_VISION_ROLES = frozenset({"executor_vision_model", "vision_agent", "video_agent", "qa_checker"})
DEFAULT = "default"


# -- settings ------------------------------------------------------------------

def _settings() -> dict:
    try:
        from src.utils.settings import load_settings
        return (load_settings().get("llm") or {}).get("routing") or {}
    except Exception:  # noqa: BLE001
        return {}


def auto_switch() -> bool:
    return bool(_settings().get("auto_switch", False))


def rules() -> list[dict]:
    """The routing rules, normalised: ``{key, kind, name, role, suggest[], why}``.

    A malformed entry is skipped, not an error: a settings typo must not cost a
    turn. ``suggest`` may be one ``"provider,model"`` or a list in preference order.
    """
    out: list[dict] = []
    raw = _settings().get("rules") or []
    for i, rule in enumerate(raw if isinstance(raw, list) else []):
        if not isinstance(rule, dict):
            continue
        when = rule.get("when") or {}
        trigger = next(((k, str(when[k]).strip()) for k in ("skill", "tool_pack", "tool")
                        if isinstance(when, dict) and str(when.get(k) or "").strip()), None)
        suggest = rule.get("suggest")
        suggest = [suggest] if isinstance(suggest, str) else list(suggest or [])
        suggest = [str(s).strip() for s in suggest if "," in str(s)]
        if not trigger or not suggest:
            continue
        out.append({"key": f"{i}:{trigger[0]}={trigger[1]}", "kind": trigger[0], "name": trigger[1],
                    "role": str(rule.get("role") or "orchestrator").strip() or "orchestrator",
                    "suggest": suggest, "why": str(rule.get("why") or "").strip()})
    return out


# -- what exists ---------------------------------------------------------------

def _catalog() -> dict:
    """``{vendor: [[spec, label], ...]}``: the models of every configured provider.

    Isolated so tests can stand in for the network. A provider appears here only
    when it has a key or a URL (src/utils/agentY_server.py ``_available_models``).
    """
    from src.utils.agentY_server import _available_models
    return _available_models()


def _price(spec: str) -> dict | None:
    """USD per million tokens from config/pricing.json or the built-in tables."""
    try:
        from src.utils import costs
        _provider, _, model_id = spec.partition(",")
        key = model_id.lower().strip()
        over = costs._load_pricing_overrides()["models"]
        pair = over.get(key)
        if pair is None:
            pair = next((pv for mk, pv in over.items() if key.startswith(mk) or mk.startswith(key)), None)
        if pair is None:
            pair = costs._lookup_prices(model_id)
        if not pair:
            return None
        return {"in": round(pair[0] * 1e6, 4), "out": round(pair[1] * 1e6, 4)}
    except Exception:  # noqa: BLE001
        return None


def available_models() -> list[dict]:
    """Every usable model: ``{id, label, vendor, vision, price_per_mtok}``."""
    from src.utils.vision_capability import supports_vision
    seen: set = set()
    out: list[dict] = []
    for vendor, entries in (_catalog() or {}).items():
        for entry in entries or []:
            spec, label = (entry[0], entry[1]) if isinstance(entry, (list, tuple)) and len(entry) > 1 \
                else (str(entry), str(entry))
            if "," not in str(spec) or spec in seen:
                continue
            seen.add(spec)
            out.append({"id": spec, "label": label, "vendor": vendor,
                        "vision": supports_vision(spec), "price_per_mtok": _price(spec)})
    return out


def find(spec: str, models: list[dict] | None = None) -> dict | None:
    spec = str(spec or "").strip().lower()
    for m in models if models is not None else available_models():
        if m["id"].lower() == spec:
            return m
    return None


def closest(spec: str, models: list[dict] | None = None, n: int = 5) -> list[str]:
    """The available ids nearest to a misspelt or invented one."""
    ids = [m["id"] for m in (models if models is not None else available_models())]
    spec = str(spec or "").strip().lower()
    near = difflib.get_close_matches(spec, [i.lower() for i in ids], n=n, cutoff=0.4)
    by_lower = {i.lower(): i for i in ids}
    named = [i for i in ids if spec.partition(",")[2] and spec.partition(",")[2] in i.lower()]
    return list(dict.fromkeys([by_lower[x] for x in near] + named))[:n]


def current_models() -> dict:
    """What every tier and role resolves to right now."""
    from src.agent import _ROLE_TIERS, TIER_LABELS, role_model
    tiers = {t: str(role_model_tier(t) or "") for t in TIER_LABELS}
    roles = {r: str(role_model(r) or "") for r in _ROLE_TIERS}
    return {"tiers": tiers, "roles": roles}


def role_model_tier(tier: str) -> str:
    from src.agent import _cfg
    return str(_cfg("", "tiers", tier, default="") or "")


# -- the images a conversation carries -----------------------------------------

def carries_images(messages) -> bool:
    """Whether any message holds an image block, also inside a tool result."""
    def _walk(node) -> bool:
        if isinstance(node, dict):
            if "image" in node and isinstance(node["image"], (dict, bytes)):
                return True
            return any(_walk(v) for k, v in node.items() if k != "image")
        if isinstance(node, (list, tuple)):
            return any(_walk(v) for v in node)
        return False
    return any(_walk(m.get("content") if isinstance(m, dict) else None) for m in messages or ())


def _any_conversation_has_images(agent) -> bool:
    if carries_images(getattr(agent, "messages", None)):
        return True
    try:
        from src.utils import agentY_server
        for pipeline in agentY_server.pipelines():
            other = getattr(pipeline, "_orchestrator_agent", None)
            if other is not None and carries_images(getattr(other, "messages", None)):
                return True
    except Exception:  # noqa: BLE001
        pass
    return False


# -- the seat ------------------------------------------------------------------

def thread_id() -> str:
    try:
        from agenty_core.utils import turn_scope
        return str(turn_scope.current().thread_id or "")
    except Exception:  # noqa: BLE001
        return ""


def conversation_spec(tid: str | None = None) -> str:
    """The model this conversation was moved to, or ''."""
    with _LOCK:
        return _OVERRIDES.get(thread_id() if tid is None else tid, "")


def running_spec(agent) -> str:
    """The ``'provider,model'`` *agent* runs, read off its cost metadata (the one
    place every provider records it in the same shape)."""
    meta = getattr(agent, "_cost_meta", None)
    if not isinstance(meta, dict):
        return ""
    provider, model = str(meta.get("provider") or "").strip(), str(meta.get("model_id") or "").strip()
    return f"{provider},{model}" if provider and model else ""


def _retarget(agent, spec: str) -> str:
    """Point *agent* at *spec* (``Agent.model`` + cost meta). Returns the spec."""
    from src import agent as A
    provider, model_name = A._parse_llm_setting(spec)
    provider = (provider or "claude").strip().lower()
    if not model_name:
        raise ValueError(f"{spec!r} is not a 'provider,model' spec")
    built, model_id = A._build_model(
        role="orchestrator", llm=provider, system_prompt=str(getattr(agent, "system_prompt", "") or ""),
        ollama_model=model_name,
        anthropic_model=model_name if provider != "ollama" else None,
        dashscope_model=model_name)
    old = getattr(agent, "model", None)
    if getattr(agent, "_routing_base", None) is None:
        agent._routing_base = (old, dict(getattr(agent, "_cost_meta", None) or {}) or None,
                               getattr(agent, "_is_claude", None))
    agent.model = built
    # The cost log and the `model=` field of tokens_usage.log read this: left behind,
    # the new model's tokens would be billed at the old model's rate.
    try:
        agent._cost_meta = {"provider": provider, "model_id": model_id, "is_ollama": provider == "ollama"}
        agent._is_claude = provider != "ollama"
    except Exception:  # noqa: BLE001
        pass
    _strip_foreign_reasoning(agent, old, built)
    return f"{provider},{model_id}"


def _strip_foreign_reasoning(agent, old, new) -> None:
    # Reasoning from one provider means nothing to another, and Anthropic refuses a
    # thinking block it did not sign. The answers themselves stay.
    if type(old) is type(new):
        return
    for msg in getattr(agent, "messages", None) or []:
        content = msg.get("content") if isinstance(msg, dict) else None
        if isinstance(content, list) and any("reasoningContent" in c for c in content if isinstance(c, dict)):
            msg["content"] = [c for c in content
                              if not (isinstance(c, dict) and "reasoningContent" in c)] or [{"text": ""}]


def _restore_base(agent) -> bool:
    base = getattr(agent, "_routing_base", None)
    if base is None:
        return False
    model, meta, is_claude = base
    if model is not None:
        _strip_foreign_reasoning(agent, getattr(agent, "model", None), model)
        agent.model = model
    if meta:
        agent._cost_meta = meta
    if is_claude is not None:
        agent._is_claude = is_claude
    agent._routing_base = None
    return True


def sync_thread(agent, tid: str) -> None:
    """Run when a pooled pipeline loads conversation *tid*: its override applies,
    and one left by a conversation that ran there before is taken off."""
    if agent is None:
        return
    with _LOCK:
        spec = _OVERRIDES.get(tid or "", "")
    try:
        if spec:
            if running_spec(agent) != spec:
                _retarget(agent, spec)
        else:
            _restore_base(agent)
    except Exception as exc:  # noqa: BLE001 - a failed swap keeps the working model
        logger.warning("routing: could not apply the seat for %s: %s", tid, exc)


def _note(kind: str, **fields) -> None:
    entry = {"ts": time.strftime("%Y-%m-%d %H:%M:%S"), "kind": kind, **fields}
    RECENT.append(entry)
    del RECENT[:-50]
    logger.info("model routing: %s", json.dumps(entry, ensure_ascii=False))
    try:
        from src.utils import status_bus
        if kind == "switch":
            status_bus.notify(f"🔀 Model switched ({fields.get('scope')}): {fields.get('role')} "
                              f"{fields.get('from') or '?'} → {fields.get('to')} — {fields.get('reason')}")
    except Exception:  # noqa: BLE001
        pass


# -- list_models / switch_model ------------------------------------------------

def list_models(agent=None, role: str = "") -> dict:
    models = available_models()
    role = str(role or "").strip().lower()
    cur = current_models()
    out: dict[str, Any] = {"ok": True, "models": models, "current": cur,
                           "auto_switch": auto_switch()}
    if agent is not None:
        out["running"] = running_spec(agent)
        out["conversation_override"] = conversation_spec()
        out["conversation_has_images"] = carries_images(getattr(agent, "messages", None))
    if role:
        affected = _affected_roles(role)
        needs_vision = bool(affected & _VISION_ROLES) or (
            "orchestrator" in affected and agent is not None and _any_conversation_has_images(agent))
        out["role"] = role
        out["current_for_role"] = cur["tiers"].get(role) or cur["roles"].get(role) or ""
        out["needs_vision"] = needs_vision
        out["candidates"] = [m["id"] for m in models if m["vision"] or not needs_vision]
    return out


def _affected_roles(target: str) -> set:
    from src.agent import _ROLE_TIERS, TIER_LABELS
    if target in TIER_LABELS:
        return {r for r, t in _ROLE_TIERS.items() if t == target} | ({target} if target in _ROLE_TIERS else set())
    return {target}


def switch_model(agent, role: str, model: str, reason: str, scope: str = "conversation",
                 confirm: bool = False) -> dict:
    """The body of the ``switch_model`` tool. Returns a plain dict.

    *confirm* is the model saying the user said yes. What makes that true is the
    approval gate on the tool call itself (``DEFAULT_ASK_TOOLS`` /
    ``permission_gate``) and ``llm.routing.auto_switch``; this function only
    refuses what is never right (unknown model, the vision trap).
    """
    role = str(role or "orchestrator").strip().lower() or "orchestrator"
    scope = str(scope or "conversation").strip().lower()
    spec = str(model or "").strip()
    reason = str(reason or "").strip()
    if scope not in ("conversation", "settings"):
        return {"ok": False, "error": "scope must be 'conversation' or 'settings'."}
    tid = thread_id()

    if spec.lower() == DEFAULT:
        if scope != "conversation":
            return {"ok": False, "error": "model='default' only undoes a conversation switch."}
        with _LOCK:
            had = _OVERRIDES.pop(tid, "")
        before = running_spec(agent)
        _restore_base(agent)
        _note("switch", scope="conversation", role="orchestrator", **{"from": before},
              to=running_spec(agent) or "default", reason=reason or "back to the configured model")
        return {"ok": True, "applied": True, "reverted": bool(had),
                "running": running_spec(agent), "takes_effect": "at your next model call"}

    models = available_models()
    entry = find(spec, models)
    if entry is None:
        near = closest(spec, models)
        return {"ok": False, "error": f"{spec!r} is not an available model. Use list_models; never name one "
                f"from memory.", "closest": near}
    spec = entry["id"]

    from src.agent import TIER_LABELS, _ROLE_TIERS
    if role == "all":
        return {"ok": False, "error": "Switch one tier or role at a time, not 'all'."}
    if role not in TIER_LABELS and role not in _ROLE_TIERS:
        return {"ok": False, "error": f"Unknown role {role!r}.",
                "tiers": list(TIER_LABELS), "roles": list(_ROLE_TIERS)}
    if scope == "conversation" and role != "orchestrator":
        return {"ok": False, "error": "scope='conversation' moves the orchestrator only. Other roles "
                "change with scope='settings'."}

    affected = _affected_roles(role)
    # The vision trap. A role that reads images needs a model that can, always; the
    # orchestrator needs one when this conversation (or, for a settings change, any
    # running one) already holds an image block, which a text-only model rejects on
    # every later turn.
    if not entry["vision"]:
        if affected & _VISION_ROLES:
            return {"ok": False, "error": f"{spec} cannot read images and {role} reads images.",
                    "closest": [m["id"] for m in models if m["vision"]][:8]}
        if "orchestrator" in affected and (
                carries_images(getattr(agent, "messages", None)) if scope == "conversation"
                else _any_conversation_has_images(agent)):
            return {"ok": False,
                    "error": f"{spec} is text-only and a conversation here already carries images; "
                    "the provider would reject every following turn. Pick a vision-capable model.",
                    "closest": [m["id"] for m in models if m["vision"]][:8]}

    before = running_spec(agent) if role == "orchestrator" else (
        current_models()["tiers"].get(role) or current_models()["roles"].get(role) or "")
    proposal = {"role": role, "from": before, "to": spec, "scope": scope, "reason": reason}
    if before == spec:
        return {"ok": True, "applied": False, "note": f"{role} is already on {spec}."}
    if not confirm:
        return {"ok": True, "applied": False, "proposal": proposal,
                "next": "Nothing changed. Ask the user whether to switch; once they say yes, call "
                        "switch_model again with the same arguments and confirm=true."}
    if scope == "conversation":
        try:
            landed = _retarget(agent, spec)
        except Exception as exc:  # noqa: BLE001 - the old model still works
            return {"ok": False, "error": f"Could not build {spec}: {exc}. Staying on {before}."}
        with _LOCK:
            _OVERRIDES[tid] = spec
        _note("switch", **proposal)
        return {"ok": True, "applied": True, "scope": scope, "running": landed,
                "takes_effect": "at your next model call (this tool result is the last one answered "
                                "by the old model)",
                "undo": "switch_model(role='orchestrator', model='default', reason=...)"}
    # settings: the same write as /switch_model, then the existing live reload
    # (deferred to the end of this turn while it runs).
    try:
        from src.utils import agentY_server
        events = agentY_server._switch_model([role, spec])
        text = " ".join(str(e.get("data", "")) for e in events if isinstance(e, dict))
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": f"Could not save the setting: {exc}"}
    if "❌" in text:
        return {"ok": False, "error": text}
    if "orchestrator" in affected:
        with _LOCK:
            _OVERRIDES.pop(tid, None)
    _note("switch", **proposal)
    return {"ok": True, "applied": True, "scope": scope, "message": text,
            "takes_effect": "when this turn ends" if "turn" in text.lower() else "now"}


# -- the rule check ------------------------------------------------------------

def triggers_of(tool_name: str, tool_input: dict) -> list[tuple[str, str]]:
    """The ``(kind, name)`` pairs one tool call stands for."""
    data = tool_input if isinstance(tool_input, dict) else {}
    out: list[tuple[str, str]] = []
    if tool_name == "skills":
        out.append(("skill", str(data.get("skill_name") or "").strip()))
    elif tool_name == "load_tools":
        out.append(("tool_pack", str(data.get("pack") or "").strip().lower()))
    else:
        out.append(("tool", tool_name))
        try:
            from src.tools.tool_packs import _PACK_OF
            if tool_name in _PACK_OF:                 # a held tool called straight away loads its pack
                out.append(("tool_pack", _PACK_OF[tool_name]))
        except Exception:  # noqa: BLE001
            pass
    return [t for t in out if t[1]]


def check_rules(agent, triggers: list[tuple[str, str]], *,
                models: Callable[[], list[dict]] | None = None) -> list[str]:
    """Notes to add to a tool result for the rules these *triggers* fire.

    No model call: a comparison of two short lists. Each rule speaks once per
    conversation, whether it suggests, switches, or finds nothing to suggest.
    """
    wanted = rules()
    if not wanted or not triggers:
        return []
    tid = thread_id() or f"agent-{id(agent)}"
    notes: list[str] = []
    avail: list[dict] | None = None
    for rule in wanted:
        if (rule["kind"], rule["name"]) not in triggers:
            continue
        mark = (tid, rule["key"])
        with _LOCK:
            if mark in _NOTIFIED:
                continue
            _NOTIFIED.add(mark)
        if avail is None:
            avail = (models or available_models)()
        has_images = carries_images(getattr(agent, "messages", None))
        pick = next((find(s, avail) for s in rule["suggest"]
                     if find(s, avail) and (find(s, avail)["vision"] or not has_images)), None)
        if pick is None:
            continue                                  # never invent a model
        if rule["role"] == "orchestrator":
            now = running_spec(agent)
        else:
            cur = current_models()
            now = cur["tiers"].get(rule["role"]) or cur["roles"].get(rule["role"]) or ""
        if now.lower() == pick["id"].lower():
            continue
        why = rule["why"] or "the rule for this kind of task"
        if auto_switch() and rule["role"] == "orchestrator":
            res = switch_model(agent, "orchestrator", pick["id"], f"routing rule: {why}",
                               scope="conversation", confirm=True)
            if res.get("applied"):
                notes.append(f"Routing: switched this conversation to {pick['id']} ({why}); "
                             f"auto_switch is on. It answers from your next model call. "
                             f"switch_model(role='orchestrator', model='default', reason=...) undoes it.")
                continue
        notes.append(f"Routing: for this task, {pick['id']} is recommended ({why}). "
                     f"Ask the user whether to switch (switch_model).")
    return notes


def reset_notices(tid: str | None = None) -> None:
    with _LOCK:
        for mark in [m for m in _NOTIFIED if tid is None or m[0] == tid]:
            _NOTIFIED.discard(mark)


try:  # the hook needs strands; the rest of this module must import without it
    from strands.hooks import AfterToolCallEvent, HookProvider, HookRegistry

    class ModelRoutingHook(HookProvider):
        """After a tool call: if a routing rule matches it, say so in the result."""

        def register_hooks(self, registry: HookRegistry, **kwargs) -> None:  # noqa: ARG002
            registry.add_callback(AfterToolCallEvent, self._on_after)

        def _on_after(self, event: AfterToolCallEvent, **kwargs) -> None:  # noqa: ARG002
            try:
                if not _settings().get("rules"):
                    return
                use = getattr(event, "tool_use", None) or {}
                notes = check_rules(event.agent, triggers_of(str(use.get("name") or ""), use.get("input") or {}))
                result = getattr(event, "result", None)
                if not notes or not isinstance(result, dict):
                    return
                content = list(result.get("content") or [])
                content.extend({"text": n} for n in notes)
                event.result = {**result, "content": content}
            except Exception as exc:  # noqa: BLE001 - never cost a turn
                logger.warning("routing hook failed: %s", exc)
except Exception:  # noqa: BLE001
    ModelRoutingHook = None  # type: ignore[assignment]
