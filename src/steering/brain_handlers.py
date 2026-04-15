"""Brain steering handlers for agentY.

Two handlers enforce Brain-specific guardrails just-in-time:

Handler 1 — ForbiddenToolHandler (rule-based, no LLM)
    Blocks save_workflow calls that are not for new-workflow builds.
    Note: submit_prompt, view_image, and analyze_image are NOT registered in
    BRAIN_TOOLS, so those are already blocked at the tool-registration layer.
    This handler guards save_workflow, which IS registered but should only
    be used when the Brain is building a workflow from scratch, not patching.

Handler 2 — ModelSamplingFluxHandler (LLM-based)
    Before an update_workflow call, checks that any ModelSamplingFlux node in
    the patches includes all four required inputs (max_shift, base_shift,
    width, height). Guides the agent to add missing inputs if needed.
    Falls back gracefully — update_workflow's own validation will also catch
    this error, so if the steering LLM misreads the patches JSON, the worst
    outcome is a validation error from update_workflow.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from strands import Agent
from strands.types.content import Message
from strands.types.tools import ToolUse
from strands.vended_plugins.steering import (
    Guide,
    LLMSteeringHandler,
    Proceed,
    SteeringHandler,
    ToolSteeringAction,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Handler 1: Forbidden tool guard (rule-based)
# ---------------------------------------------------------------------------

_FORBIDDEN_TOOLS = frozenset(["save_workflow"])
_FORBIDDEN_REASON = (
    "save_workflow is reserved for building entirely new workflows from scratch. "
    "You are patching an existing template — use update_workflow() instead. "
    "Call update_workflow(workflow_path, patches, add_nodes, remove_nodes)."
)


class BrainForbiddenToolHandler(SteeringHandler):
    """Blocks save_workflow when the Brain is patching a template.

    submit_prompt, view_image, and analyze_image are not registered in
    BRAIN_TOOLS, so this handler only needs to guard save_workflow.
    """

    name: str = "brain_forbidden_tool"

    async def steer_before_tool(
        self, *, agent: Agent, tool_use: ToolUse, **kwargs: Any
    ) -> ToolSteeringAction:
        tool_name = tool_use.get("name", "")
        if tool_name in _FORBIDDEN_TOOLS:
            logger.debug("brain_forbidden_tool: blocking %s", tool_name)
            return Guide(reason=_FORBIDDEN_REASON)
        return Proceed(reason="tool allowed")


# ---------------------------------------------------------------------------
# Handler 2: ModelSamplingFlux patch validator (LLM-based)
# ---------------------------------------------------------------------------

_FLUX_SAMPLING_SYSTEM_PROMPT = """\
You are a guardrail monitor for a ComfyUI workflow patching agent.

Your only job: when the agent calls update_workflow, inspect the patches array.
If ANY patch targets a node whose inputs suggest it is a ModelSamplingFlux node
(inputs like max_shift, base_shift, or the node is identified by type), verify
that ALL FOUR of these inputs are present somewhere in the patches OR in the
existing tool session history:
  - max_shift  (recommended value: 1.15)
  - base_shift (recommended value: 0.5)
  - width      (from brainbriefing.resolution)
  - height     (from brainbriefing.resolution)

Decision rules:
- If update_workflow is being called and there is a ModelSamplingFlux node in
  the patches but one or more of the four required inputs are missing:
  → decision: "guide"
  → reason: explain exactly which inputs are missing and that all four are
    required for ModelSamplingFlux (max_shift=1.15, base_shift=0.5, width, height).
- In all other cases (no ModelSamplingFlux involved, or all four inputs present):
  → decision: "proceed"

Only assess what is present in the steering context. Do not speculate about nodes
not mentioned in the patches.
"""


class ModelSamplingFluxHandler(LLMSteeringHandler):
    """LLM-based guard that ensures ModelSamplingFlux patches include all four required inputs.

    Uses LLM evaluation against the patches JSON in the tool call.
    If the steering LLM misreads the patches, update_workflow's own validation
    will still catch the error on the next call — this handler is a best-effort
    early warning.
    """

    name: str = "model_sampling_flux_guard"

    async def steer_before_tool(
        self, *, agent: Agent, tool_use: ToolUse, **kwargs: Any
    ) -> ToolSteeringAction:
        tool_name = tool_use.get("name", "")
        if tool_name != "update_workflow":
            return Proceed(reason="not an update_workflow call")
        # Delegate to LLM evaluation only for update_workflow calls.
        return await super().steer_before_tool(agent=agent, tool_use=tool_use, **kwargs)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def get_brain_steering_handlers(model=None) -> list:
    """Return the list of steering handler plugins for the Brain agent.

    Args:
        model: Optional Strands Model instance for the LLM-based handlers.
               Defaults to None, which reuses the Brain agent's own model
               (claude-haiku-4-5 by default — cheapest Anthropic option).

    Returns:
        List of SteeringHandler instances ready to pass as ``plugins=``.
    """
    return [
        BrainForbiddenToolHandler(),
        ModelSamplingFluxHandler(
            system_prompt=_FLUX_SAMPLING_SYSTEM_PROMPT,
            model=model,  # None → reuse agent's model (haiku-4-5)
        ),
    ]
