"""Brain steering handlers for agentY.

Two handlers enforce Brain-specific guardrails just-in-time:

Handler 1 — ForbiddenToolHandler (rule-based, no LLM)
    Blocks save_workflow calls that are not for new-workflow builds.
    Note: submit_prompt, view_image, and analyze_image are NOT registered in
    BRAIN_TOOLS, so those are already blocked at the tool-registration layer.
    This handler guards save_workflow, which IS registered but should only
    be used when the Brain is building a workflow from scratch, not patching.

Handler 2 — ModelSamplingFluxHandler (rule-based, no LLM)
    Before an update_workflow call, parses the patches / add_nodes JSON and
    checks whether any ModelSamplingFlux node is being patched or added.  If
    so, verifies that all four required inputs (max_shift, base_shift, width,
    height) are present.  Guides the agent to add missing inputs if needed.
    Falls back gracefully — update_workflow's own validation will also catch
    this error, so if the rule misreads the JSON the worst outcome is a
    validation error from update_workflow.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from strands import Agent
from strands.types.tools import ToolUse
from strands.vended_plugins.steering import (
    Guide,
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
# Handler 2: ModelSamplingFlux patch validator (rule-based, no LLM)
# ---------------------------------------------------------------------------

# The four inputs that ModelSamplingFlux always requires.
_FLUX_REQUIRED_INPUTS = frozenset({"max_shift", "base_shift", "width", "height"})

# Heuristic markers — if a patch touches any of these input names it is
# likely targeting a ModelSamplingFlux node.
_FLUX_MARKER_INPUTS = frozenset({"max_shift", "base_shift"})


def _parse_json_arg(val: Any) -> list:
    """Parse a tool argument that may be a JSON string or already a list."""
    if isinstance(val, str):
        try:
            parsed = json.loads(val)
            return parsed if isinstance(parsed, list) else []
        except (json.JSONDecodeError, TypeError):
            return []
    return val if isinstance(val, list) else []


def _collect_flux_inputs(patches: list, add_nodes: list) -> tuple[bool, set[str]]:
    """Scan *patches* and *add_nodes* for ModelSamplingFlux references.

    Returns ``(is_flux_involved, found_inputs)`` where *found_inputs* is the
    set of flux-relevant input names present across all matching entries.
    """
    found: set[str] = set()
    is_flux = False

    # -- patches: [{"node_id": ..., "input_name": ..., "value": ...}, ...]
    for patch in patches:
        if not isinstance(patch, dict):
            continue
        input_name = patch.get("input_name", "")
        class_type = str(patch.get("class_type", "")).lower()
        if input_name in _FLUX_REQUIRED_INPUTS or "modelsamplingflux" in class_type:
            is_flux = True
            if input_name in _FLUX_REQUIRED_INPUTS:
                found.add(input_name)

    # -- add_nodes: [{"class_type": ..., "inputs": {...}}, ...]
    for spec in add_nodes:
        if not isinstance(spec, dict):
            continue
        class_type = str(spec.get("class_type", "")).lower()
        if "modelsamplingflux" in class_type:
            is_flux = True
            inputs = spec.get("inputs", {})
            if isinstance(inputs, str):
                try:
                    inputs = json.loads(inputs)
                except (json.JSONDecodeError, TypeError):
                    inputs = {}
            if isinstance(inputs, dict):
                found.update(k for k in inputs if k in _FLUX_REQUIRED_INPUTS)

    return is_flux, found


class ModelSamplingFluxHandler(SteeringHandler):
    """Rule-based guard that ensures ModelSamplingFlux patches include all four required inputs.

    Parses the ``patches`` and ``add_nodes`` JSON arguments of update_workflow
    calls and checks for ModelSamplingFlux references.  No LLM call is made —
    this is a fast, deterministic string/key check.

    If the rule misreads the JSON, update_workflow's own validation will still
    catch the error on the next call — this handler is a best-effort early
    warning.
    """

    name: str = "model_sampling_flux_guard"

    async def steer_before_tool(
        self, *, agent: Agent, tool_use: ToolUse, **kwargs: Any
    ) -> ToolSteeringAction:
        tool_name = tool_use.get("name", "")
        if tool_name != "update_workflow":
            return Proceed(reason="not an update_workflow call")

        tool_input = tool_use.get("input") or {}
        patches = _parse_json_arg(tool_input.get("patches", "[]"))
        add_nodes = _parse_json_arg(tool_input.get("add_nodes", "[]"))

        is_flux, found = _collect_flux_inputs(patches, add_nodes)

        if not is_flux:
            return Proceed(reason="no ModelSamplingFlux nodes in patches")

        missing = _FLUX_REQUIRED_INPUTS - found
        if not missing:
            return Proceed(reason="all ModelSamplingFlux inputs present")

        logger.debug("model_sampling_flux_guard: missing inputs %s", missing)
        return Guide(
            reason=(
                f"ModelSamplingFlux patch is missing required inputs: {', '.join(sorted(missing))}. "
                "All four inputs are required: max_shift (recommended: 1.15), "
                "base_shift (recommended: 0.5), width, height (from brainbriefing resolution). "
                "Add the missing inputs to your update_workflow patches."
            )
        )


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def get_brain_steering_handlers(model=None) -> list:
    """Return the list of steering handler plugins for the Brain agent.

    Args:
        model: Unused — retained for API compatibility.  Both handlers are
               now rule-based and make zero LLM calls.

    Returns:
        List of SteeringHandler instances ready to pass as ``plugins=``.
    """
    return [
        BrainForbiddenToolHandler(),
        ModelSamplingFluxHandler(),
    ]
