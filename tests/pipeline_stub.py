"""One Pipeline stand-in for the tests that drive the orchestrator's own tools.

``Pipeline._build_delegation_tools`` builds closures over ``self``, and the tools
only touch a handful of attributes — so a SimpleNamespace carrying those, with
the real methods bound to it, exercises exactly what the agent calls without an
agent, a model, or a ComfyUI.

The methods are **bound, not reimplemented**: a fake that drifts from the real
one tests nothing. This lives in its own module because three test files were
each maintaining their own copy of the list, and each new guard in front of a
tool (the hard-limit check, the plan gate, the output tagging) broke whichever
copies had not been updated.

Not named ``test_*``, so unittest discovery imports it only when asked.
"""

import inspect
from types import SimpleNamespace

from src.pipeline import Pipeline

# Real Pipeline methods the delegation tools reach for. Add to this when a tool
# starts calling a new one — that is the whole point of having it in one place.
_BOUND = (
    "_run_canvas_batch",
    "_batch_limit_refusal",
    "_canvas_limit_refusal",
    "_count_handback",
    "_plan_gate_refusal",
    "_tag_run_outputs",
    "_hook_output_role",
    "_hook_for_targets",
    "_caption_from_brief",
    "_name_variants",
    "_variant_report",
    "_variant_label",
    "_collector_refusal",
    "_collector_text_refusal",
    "_policy_rejection",
    "_retry_after_refusal",
    "_reroll_seeds",
)


def pipeline_stub(**over):
    """A Pipeline stand-in. Keyword arguments override any base attribute."""
    base = dict(
        _hook_run_stopped=None,
        _canvas_keeplive_run=False,
        _canvas_base_prompt={"1": {"class_type": "KSampler", "inputs": {"seed": 1}}},
        _canvas_hooks=[],
        _verbose=False,
        _session=SimpleNamespace(current_output_paths=[]),
        _last_brainbriefing_json="{}",
        _chain_output_paths=[],
        _qa_briefing=None,
        _qa_retry=None,
        _heal_exec_failure=lambda *a, **k: None,
        _limit_handbacks={},
        _plan_approval=None,
        _plan_gate_open=False,
        _plan_gate_fired=False,
        _policy_retries={},
    )
    base.update(over)
    ns = SimpleNamespace(**base)
    for name in _BOUND:
        fn = getattr(Pipeline, name)
        # A staticmethod takes no self; binding one would feed it the namespace.
        static = isinstance(inspect.getattr_static(Pipeline, name), staticmethod)
        setattr(ns, name, fn if static else fn.__get__(ns))
    return ns


def tools(pipe):
    """Name -> callable for the orchestrator's delegation tools."""
    return {t.tool_name: t for t in Pipeline._build_delegation_tools(pipe)}
