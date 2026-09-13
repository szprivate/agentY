"""Switch a model without restarting the host.

Most of agentY's agents are built once, when the host starts, and kept: the
orchestrator, the template researcher, the info / planner / web specialists, the
vision and video readers. Their model was fixed at construction, so changing a
tier in Settings did nothing until the next restart — and ``/switch_model``
rebuilt only four of them, telling you the rest would "apply on the next agent
start".

What changed is knowable: every role resolves its model through
:func:`src.agent.role_model`, so a :func:`fingerprint` taken before a settings
change, compared with one taken after, names exactly the agents that are now
wrong. :func:`reload_live_agents` rebuilds those in place. Agents built per call
(QA, learnings, coder, the chat-title helper) need nothing: they read settings
every time.

The orchestrator is the agent that carries a conversation, so its messages move
across to the rebuilt one — switching model mid-conversation must not make the
agent forget what it was doing.
"""

from __future__ import annotations

import hashlib
import json

# Live agent -> the roles its model resolves from. A role with no explicit model
# falls back to another's (search_web to info; the repair and build specialists
# to assemble_workflow; the local fallbacks to llm_functions; vision to
# executor_vision_model), so a change to the fallback moves them too.
LIVE_AGENTS: dict[str, tuple[str, ...]] = {
    "orchestrator": ("orchestrator",),
    "query_templates": ("query_templates", "llm_functions"),
    "info": ("info", "llm_functions"),
    "planner": ("planner", "llm_functions"),
    "search_web": ("search_web", "info", "llm_functions"),
    "vision_agent": ("vision_agent", "executor_vision_model"),
    "video_agent": ("video_agent",),
    "fix_workflow_assembly": ("fix_workflow_assembly", "assemble_workflow"),
    "generate_new_workflow": ("generate_new_workflow", "assemble_workflow"),
}

# The key in a fingerprint that stands for everything in `llm` that is not a model
# name — endpoints, provider options, the history window. Every agent reads it.
PROVIDERS = "__providers__"

# An environment variable a model client reads when it is BUILT. A new API key does
# not reach an agent already holding the old one, and no fingerprint of model
# names can see it changed, so any of these changing rebuilds everything.
_CLIENT_ENV_MARKERS = ("ANTHROPIC", "DASHSCOPE", "OPENAI", "GEMINI", "GOOGLE", "OLLAMA")


def env_affects_clients(keys) -> bool:
    """Whether changing these environment keys changes what a model client holds."""
    return any(marker in str(key).upper() for key in (keys or []) for marker in _CLIENT_ENV_MARKERS)


def fingerprint() -> dict[str, str]:
    """What every role resolves to right now, plus the provider settings they share."""
    from src.agent import _ROLE_TIERS, _settings, role_model

    roles = set(_ROLE_TIERS)
    for deps in LIVE_AGENTS.values():
        roles.update(deps)
    fp: dict[str, str] = {}
    for role in sorted(roles):
        try:
            fp[role] = str(role_model(role) or "")
        except Exception as exc:  # noqa: BLE001
            fp[role] = f"<unresolvable: {exc}>"
    shared = dict((_settings() or {}).get("llm") or {})
    shared.pop("tiers", None)
    shared.pop("pipeline", None)
    encoded = json.dumps(shared, sort_keys=True, default=str).encode("utf-8")
    fp[PROVIDERS] = hashlib.sha1(encoded).hexdigest()
    return fp


def changed_agents(before: dict, after: dict) -> list[str]:
    """The live agents whose model is not what it was when *before* was taken."""
    before, after = before or {}, after or {}
    if before.get(PROVIDERS) != after.get(PROVIDERS):
        return list(LIVE_AGENTS)
    return [name for name, deps in LIVE_AGENTS.items()
            if any(before.get(dep) != after.get(dep) for dep in deps)]


def reload_live_agents(pipeline, names) -> tuple[list[str], dict[str, str]]:
    """Rebuild *names* on *pipeline* from the current settings.

    Returns ``(rebuilt, failures)``. An agent that fails to build is left exactly
    as it was — the old model keeps working — and the reason is reported, rather
    than leaving nothing, or half of something, in its place.
    """
    import src.agent as agents

    def orchestrator():
        old = getattr(pipeline, "_orchestrator_agent", None)
        new = agents.create_orchestrator_agent(
            extra_tools=getattr(pipeline, "_delegation_tools", None))
        if old is not None and hasattr(new, "messages"):
            new.messages[:] = list(getattr(old, "messages", None) or [])
        pipeline.set_orchestrator(new)

    def swap(attribute: str, factory: str):
        return lambda: setattr(pipeline, attribute, getattr(agents, factory)())

    builders = {
        "orchestrator": orchestrator,
        "query_templates": swap("_researcher", "create_query_templates_agent"),
        "info": swap("_info_agent", "create_info_agent"),
        "planner": swap("_planner_agent", "create_planner_agent"),
        "search_web": swap("_search_web_agent", "create_search_web_agent"),
        "vision_agent": lambda: pipeline._init_vision_agent(strict=True),
        "video_agent": lambda: pipeline._init_video_agent(strict=True),
        # Built lazily on first use: dropping the cached one IS the rebuild.
        "fix_workflow_assembly": lambda: setattr(pipeline, "_fix_agent", None),
        "generate_new_workflow": lambda: setattr(pipeline, "_generate_agent", None),
    }
    rebuilt: list[str] = []
    failures: dict[str, str] = {}
    for name in names:
        build = builders.get(name)
        if build is None:
            continue
        try:
            build()
            rebuilt.append(name)
        except Exception as exc:  # noqa: BLE001
            failures[name] = str(exc)
    return rebuilt, failures
