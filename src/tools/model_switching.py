"""The orchestrator's two model tools: see what exists, and move its own seat.

Host-specific on purpose: switching changes THIS host's running agents, so it does
not belong in agenty_core (which is for host-independent tools and is also served
by agentY-mcp). The logic and its reasons are in src/utils/model_routing.py.
"""
from __future__ import annotations

import json

from strands import tool
from strands.types.tools import ToolContext

from src.utils import model_routing


@tool(context=True)
def list_models(tool_context: ToolContext, role: str = "") -> str:
    """List the models you can switch to, and what every tier and role runs now.

    Only models of providers that are set up are listed. Each has its id
    ("provider,model"), whether it can read images, its price per million tokens
    when known, and `current` says what each tier/role uses. Name only models from
    this list.

    Args:
        role: Optional tier or role (e.g. "orchestrator", "vision") to also get the
            candidates that fit it.
    """
    try:
        return json.dumps(model_routing.list_models(tool_context.agent, role), ensure_ascii=False)
    except Exception as exc:  # noqa: BLE001
        return json.dumps({"ok": False, "error": str(exc)})


@tool(context=True)
def switch_model(role: str, model: str, reason: str, tool_context: ToolContext,
                 scope: str = "conversation", confirm: bool = False) -> str:
    """Propose or make a model switch for the orchestrator or another role.

    Without confirm=true this only returns a proposal: ask the user, and call again
    with confirm=true after they say yes (setting llm.routing.auto_switch lets a
    rule switch without asking). Never changes anything you did not first list.

    Args:
        role: "orchestrator" (the model that is talking to the user) or, with
            scope="settings", any tier or role from list_models.
        model: "provider,model" from list_models, or "default" to undo a
            conversation switch.
        reason: One sentence on why this task wants that model; it is logged.
        scope: "conversation" (this conversation only, from your next model call
            until you undo it) or "settings" (saved, for every conversation).
        confirm: true only after the user agreed to this exact switch.
    """
    try:
        return json.dumps(model_routing.switch_model(tool_context.agent, role, model, reason,
                                                      scope=scope, confirm=bool(confirm)),
                          ensure_ascii=False)
    except Exception as exc:  # noqa: BLE001
        return json.dumps({"ok": False, "error": str(exc)})
