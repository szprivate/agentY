"""agentY-only pipeline handoff tool.

``signal_workflow_ready`` is specific to the Strands multi-agent pipeline: the
Assemble Workflow calls it as its final step instead of executing the workflow itself, and
the Executor stage (submission, polling, Ollama Vision-QA, output saving) takes
over.  It is therefore NOT part of the shared ``agenty_core`` tool layer — it
lives here, in the agentY repo only.  The agentY-mcp server has no equivalent
(its host model calls ``execute_workflow`` directly).
"""

import json
from pathlib import Path

from strands import tool


@tool
def signal_workflow_ready(workflow_path: str) -> str:
    """Signal that the workflow is fully assembled and validated, ready for execution.

    Call this as your **final step** once ``update_workflow()`` returns ``status: "ok"``.
    The pipeline will automatically handle ComfyUI submission, completion
    polling, Vision QA (via Ollama), and saving outputs to ``./output``.

    For **batch runs** (``count_iter > 1``): call this tool once for every
    workflow file produced by ``duplicate_workflow()``.  Each call appends to
    the execution queue; the pipeline submits them to ComfyUI in order.

    Do NOT call ``submit_prompt`` — this tool replaces it.

    Args:
        workflow_path: File path to the validated workflow JSON
                       (the same path returned by ``get_workflow_template`` or
                       ``save_workflow`` and used in ``update_workflow``).
    """
    from src.utils.workflow_signal import append_workflow_path

    p = Path(workflow_path)
    if not p.exists():
        return json.dumps({"error": f"Workflow file not found: {workflow_path}"})

    resolved = str(p.resolve())
    append_workflow_path(resolved)
    return json.dumps({
        "status": "ready",
        "workflow_path": resolved,
        "message": (
            "Workflow has been added to the execution queue. "
            "The pipeline will submit it to ComfyUI, run Vision QA, "
            "and save outputs to ./output automatically. "
            "For batch runs, call signal_workflow_ready for each duplicate workflow; "
            "otherwise your work here is done — no further tool calls are needed."
        ),
    })
