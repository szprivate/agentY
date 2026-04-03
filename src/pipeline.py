"""
agentY – Two-agent pipeline: Researcher → Brain.

The pipeline exposes a single callable that accepts a raw user request,
runs it through the Researcher to produce a brainbriefing JSON, then
hands that JSON to the Brain for workflow assembly, execution, and QA.

Usage
-----
>>> from src.pipeline import create_pipeline
>>> pipeline = create_pipeline()
>>> response = pipeline("Generate a cinematic wide-shot of Tokyo at night.")
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import textwrap
from typing import Any, List, Optional

from pydantic import BaseModel, Field, ValidationError
from strands import Agent

from src.agent import create_brain_agent, create_researcher_agent
from src.comfyui_interrupt_hook import INTERRUPT_NAME


# ---------------------------------------------------------------------------
# Brainbriefing schema (Pydantic) — mirrors config/brainbrief_example.json
# ---------------------------------------------------------------------------

class InputImage(BaseModel):
    """A single input image/video asset referenced in the task."""
    node_id: str = Field(description="Node ID in the workflow JSON (from io.inputs[].nodeId)")
    filename: str = Field(description="Filename of the asset")
    role: str = Field(description="Role: master_image | reference_image | mask | depth_map | control_image")
    node: str = Field(description="ComfyUI loader node class name")
    slot: str = Field(description="Input slot name on the node")
    path: str = Field(description="Full path to the asset on disk")


class Task(BaseModel):
    """High-level description of what is being generated."""
    type: str = Field(description="Task type: image edit | image generation | video flf | video i2v | video v2v | audio")
    description: str = Field(description="One sentence summary of the task")


class BriefTemplate(BaseModel):
    """Selected ComfyUI workflow template."""
    name: Optional[str] = Field(default=None, description="Template name, or null if not resolved")


class BriefPrompt(BaseModel):
    """Generation prompts."""
    positive: str = Field(description="Positive generation prompt")
    negative: Optional[str] = Field(default=None, description="Negative prompt, or null")


class OutputNode(BaseModel):
    """A single output node in the ComfyUI workflow that saves generated assets."""
    node_id: str = Field(description="Node ID in the workflow JSON")
    node: str = Field(description="ComfyUI output node class name (e.g. SaveImage, VHS_VideoCombine)")
    output_path: str = Field(description="Full directory path where the node will save its output")


class BrainBriefing(BaseModel):
    """Structured handoff document from the Researcher to the Brain."""
    status: str = Field(description="'ready' or 'blocked'")
    blockers: List[str] = Field(default_factory=list, description="List of blocker descriptions")
    task: Task
    template: BriefTemplate
    input_images: List[InputImage] = Field(default_factory=list)
    input_image_count: int = Field(default=0, description="Must equal len(input_images)")
    output_nodes: List[OutputNode] = Field(default_factory=list, description="Output nodes from the workflow with their save paths")
    resolution_width: Optional[Any] = Field(default=None, description="Image width in pixels")
    resolution_height: Optional[Any] = Field(default=None, description="Image height in pixels")
    prompt: BriefPrompt
    # notes_for_executor: Optional[str] = Field(default=None, description="Additional notes for the Brain")


# ---------------------------------------------------------------------------
# JSON extraction helper
# ---------------------------------------------------------------------------

def _extract_json(text: str) -> str | None:
    """Pull the first JSON object out of *text*, even if wrapped in a code fence."""
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fenced:
        return fenced.group(1).strip()
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    for i, ch in enumerate(text[start:], start):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1].strip()
    return None


# ---------------------------------------------------------------------------
# ComfyUI async poller (zero-token polling between interrupt and resume)
# ---------------------------------------------------------------------------

# How many seconds to wait between /history checks.
_COMFYUI_POLL_INTERVAL: float = float(os.environ.get("COMFYUI_POLL_INTERVAL_S", "3"))
# Hard ceiling so we never wait forever (seconds).
_COMFYUI_POLL_TIMEOUT: float = float(os.environ.get("COMFYUI_POLL_TIMEOUT_S", str(30 * 60)))


async def _poll_comfyui_job(prompt_id: str) -> dict:
    """Poll ``GET /history/{prompt_id}`` until the job is complete.

    This runs entirely in Python (asyncio.sleep) — no LLM calls, no tokens
    burned.  Returns the stripped history dict on success, or an error dict
    if the timeout is exceeded or the job reports an error status.

    Args:
        prompt_id: The ComfyUI prompt ID returned by ``submit_prompt``.

    Returns:
        A dict suitable for passing as the interrupt response to the Brain.
    """
    import logging
    from src.comfyui_client import get_client
    from src.tools.history import _strip_history

    logger = logging.getLogger("agentY.pipeline.poller")
    client = get_client()
    waited: float = 0.0

    logger.info("poller: start polling prompt_id=%s (interval=%.1fs, timeout=%.0fs)",
                prompt_id, _COMFYUI_POLL_INTERVAL, _COMFYUI_POLL_TIMEOUT)

    while waited < _COMFYUI_POLL_TIMEOUT:
        await asyncio.sleep(_COMFYUI_POLL_INTERVAL)
        waited += _COMFYUI_POLL_INTERVAL

        try:
            raw = client.get(f"/history/{prompt_id}")
        except Exception as exc:
            logger.warning("poller: HTTP error after %.0fs — %s", waited, exc)
            continue

        if not isinstance(raw, dict) or prompt_id not in raw:
            # Job not in history yet — still queued
            logger.debug("poller: prompt_id=%s not in history yet (%.0fs elapsed)", prompt_id, waited)
            continue

        entry = raw[prompt_id]
        status_info = entry.get("status", {})
        status_str = status_info.get("status_str", "")
        completed: bool = status_info.get("completed", False)

        if completed:
            logger.info("poller: prompt_id=%s completed after %.0fs", prompt_id, waited)
            return _strip_history(raw)

        if status_str == "error":
            logger.error("poller: prompt_id=%s reported error after %.0fs", prompt_id, waited)
            return {"error": "ComfyUI job failed", "details": _strip_history(raw)}

        logger.debug("poller: prompt_id=%s status=%s (%.0fs elapsed)", prompt_id, status_str, waited)

    logger.error("poller: prompt_id=%s timed out after %.0fs", prompt_id, waited)
    return {"error": f"ComfyUI job {prompt_id} timed out after {_COMFYUI_POLL_TIMEOUT:.0f}s"}


# ---------------------------------------------------------------------------
# Pipeline callable
# ---------------------------------------------------------------------------

class Pipeline:
    """Chains Researcher → Brain with logging and JSON validation.

    Call ``pipeline(user_input)`` just like a Strands Agent.
    The Researcher runs once per call (stateless); the Brain keeps
    a sliding-window conversation so multi-turn interactions work.

    ``stream_async`` is also supported so the Slack server can update
    its placeholder message in real-time from the Brain stage.
    ``event_loop_metrics`` delegates to the Brain agent so token-usage
    reporting in Slack continues to work.
    """

    def __init__(self, researcher: Agent, brain: Agent, *, verbose: bool = True, skip_brain: bool = False) -> None:
        self._researcher = researcher
        self._brain = brain
        self._verbose = verbose
        self._skip_brain = skip_brain

    def _should_skip_brain(self) -> bool:
        if self._skip_brain:
            return True
        return str(os.environ.get("PIPELINE_SKIP_BRAIN", "false")).strip().lower() in (
            "1", "true", "yes", "on"
        )

    # The Slack server and main.py both do:  response = agent(user_input)
    def __call__(self, user_input: str, **kwargs: Any) -> str:
        return self.run(user_input, **kwargs)

    # Delegate metric access to the Brain so the Slack server can read
    # token usage summaries just as it would from a plain Strands Agent.
    @property
    def event_loop_metrics(self):  # noqa: ANN201
        return self._brain.event_loop_metrics

    def run(self, user_input: str, **_: Any) -> str:
        """Run the full pipeline for *user_input* and return the Brain's response."""
        raw_json, error, researcher_output = self._run_researcher(user_input)
        if error:
            return error

        if self._should_skip_brain():
            if self._verbose:
                print("[pipeline] Skipping Brain stage; returning Researcher output.")
            return researcher_output

        brain_prompt = self._build_brain_prompt(raw_json)
        brain_response = str(self._brain(brain_prompt))
        if self._verbose:
            print("[pipeline] Brain finished.")
        return brain_response

    async def stream_async(self, user_input: str):  # noqa: ANN201
        """Async generator compatible with the Slack server's streaming loop.

        Runs the Researcher synchronously (it's a single-turn spec dump),
        then transparently streams the Brain's token output so Slack can
        update its placeholder message in real time.

        When the Brain is interrupted by ``ComfyUIInterruptHook`` (i.e. a
        ``submit_prompt`` call was just made), this method:
        1. Detects the ``"interrupt"`` stop reason in the event stream.
        2. Extracts the ``prompt_id`` from the interrupt's ``reason`` field.
        3. Polls ``GET /history/{prompt_id}`` in a cheap asyncio.sleep loop
           (zero LLM tokens burned during the wait).
        4. Resumes the Brain with an ``interruptResponse`` carrying the
           completed ComfyUI history, then continues streaming QA output.

        Yields the same event dicts that a Strands Agent.stream_async would.
        """
        # Stage 1 – Researcher (synchronous; fast pattern-matching turn)
        if self._verbose:
            print("[pipeline:stream] Stage 1 – Researcher resolving spec …")
        raw_json, error, researcher_output = self._run_researcher(user_input)
        if error:
            yield {"data": error}
            return

        if self._should_skip_brain():
            if self._verbose:
                print("[pipeline:stream] Skipping Brain stage; returning Researcher output.")
            yield {"data": researcher_output}
            return

        # Stage 2 – Brain (streamed, with ComfyUI interrupt handling)
        if self._verbose:
            print("[pipeline:stream] Stage 2 – Brain streaming …")
        brain_prompt = self._build_brain_prompt(raw_json)

        # Brain input for the first invocation is the brain_prompt string;
        # subsequent invocations (after a ComfyUI interrupt) supply the
        # interruptResponse list.
        current_input: Any = brain_prompt

        while True:
            interrupt_result = None

            async for event in self._brain.stream_async(current_input):
                yield event  # forward all events to Slack verbatim

                # Detect agent interrupt: the final event before the loop stops
                # is AgentResultEvent = {"result": AgentResult(...)}
                if "result" in event:
                    agent_result = event["result"]
                    if getattr(agent_result, "stop_reason", None) == "interrupt":
                        interrupts = getattr(agent_result, "interrupts", [])
                        for intr in interrupts:
                            if getattr(intr, "name", None) == INTERRUPT_NAME:
                                interrupt_result = intr
                                break

            if interrupt_result is None:
                # Normal completion — no ComfyUI interrupt pending.
                if self._verbose:
                    print("[pipeline:stream] Brain finished (no interrupt).")
                break

            # ── ComfyUI interrupt: poll cheaply, then resume ───────── #
            prompt_id: str = interrupt_result.reason
            if self._verbose:
                print(f"[pipeline:stream] ComfyUI interrupt — polling prompt_id={prompt_id}")

            yield {"data": f"\n\n_⏳ ComfyUI job queued (`{prompt_id}`). Waiting for completion…_"}

            history_result = await _poll_comfyui_job(prompt_id)

            yield {"data": "\n_✅ ComfyUI job finished — resuming…_"}
            if self._verbose:
                print(f"[pipeline:stream] ComfyUI job {prompt_id} finished. Resuming Brain.")

            # Resume the agent: supply the polled history as the interrupt response.
            current_input = [
                {
                    "interruptResponse": {
                        "interruptId": interrupt_result.id,
                        "response": json.dumps(history_result),
                    }
                }
            ]

    # ── Internal helpers ─────────────────────────────────────────────── #

    _MAX_RESEARCHER_RETRIES = 2  # up to 2 correction rounds after the first attempt

    def _run_researcher(self, user_input: str) -> tuple[str | None, str | None, str]:
        """Run the Researcher and return ``(raw_json, error_message, researcher_output)``.

        Calls the Researcher as a normal text agent (preserving its tool use),
        extracts the JSON from the response, then validates it with the
        ``BrainBriefing`` Pydantic model.  On success the model is re-serialised
        so the Brain always receives canonically formatted JSON.

        If validation fails, the error is fed back to the Researcher for up to
        ``_MAX_RESEARCHER_RETRIES`` correction rounds before giving up.

        Returns ``(json_str, None, researcher_output)`` on success, ``(None, error_str, last_output)`` on failure.
        """
        researcher_prompt = textwrap.dedent(f"""
            User request:
            {user_input}

            Resolve all fields and output the brainbriefing JSON.
        """).strip()

        last_error: str | None = None

        for attempt in range(1 + self._MAX_RESEARCHER_RETRIES):
            if attempt == 0:
                prompt = researcher_prompt
            else:
                # Feed the validation error back so the model can self-correct
                if self._verbose:
                    print(
                        f"[pipeline] Researcher retry {attempt}/{self._MAX_RESEARCHER_RETRIES} …"
                    )
                prompt = textwrap.dedent(f"""
                    Your previous brainbriefing output failed JSON/schema validation:
                    {last_error}

                    Please output ONLY the corrected brainbriefing JSON with all
                    required fields correctly typed. No prose, no markdown fences.
                """).strip()

            last_response = str(self._researcher(prompt))
            label = "initial" if attempt == 0 else f"retry {attempt}"
            if self._verbose:
                print(f"[pipeline] Researcher finished ({label}). Extracting brainbriefing …")

            raw_json = _extract_json(last_response)
            if raw_json is None:
                last_error = "No JSON object found in the output."
                continue

            try:
                data = json.loads(raw_json)
                briefing = BrainBriefing.model_validate(data)
            except (json.JSONDecodeError, ValidationError) as exc:
                last_error = str(exc)
                if self._verbose:
                    print(f"[pipeline] Researcher ({label}) validation failed: {last_error}")
                continue

            # Success — re-serialise from validated model so Brain always gets clean JSON
            raw_json = briefing.model_dump_json(indent=2)
            if self._verbose:
                if attempt > 0:
                    print(f"[pipeline] Brainbriefing recovered after {attempt} retry(ies).")
                print(
                    f"[pipeline] Brainbriefing OK ({label}) — "
                    f"status={briefing.status}, task={briefing.task.description!r}, "
                    f"template={briefing.template.name!r}"
                )
            return raw_json, None, raw_json

        # All attempts exhausted
        return None, (
            f"Brainbriefing validation failed after {1 + self._MAX_RESEARCHER_RETRIES} attempts: "
            f"{last_error}"
        ), ""

    def _build_brain_prompt(self, raw_json: str) -> str:
        """Format the Brain's input prompt from the resolved brainbriefing JSON."""
        task_description = "unknown"
        try:
            task_description = json.loads(raw_json).get("task", {}).get("description", "unknown")
        except Exception:
            pass
        return textwrap.dedent(f"""
            Brainbriefing from Researcher (task: {task_description}):

            ```json
            {raw_json}
            ```

            Assemble the ComfyUI workflow from this spec, execute it, run QA,
            and send the result to Slack.
        """).strip()


# ---------------------------------------------------------------------------
# Public factory
# ---------------------------------------------------------------------------

def create_pipeline(
    *,
    researcher_llm: str | None = None,
    researcher_ollama_model: str | None = None,
    researcher_anthropic_model: str | None = None,
    brain_llm: str | None = None,
    brain_anthropic_model: str | None = None,
    brain_ollama_model: str | None = None,
    verbose: bool = True,
    skip_brain: bool = False,
) -> Pipeline:
    """Create and return a ready-to-use two-agent Pipeline.

    All arguments are optional; each falls back to environment variables,
    then to hard-coded defaults.

    Researcher defaults:
        RESEARCHER_LLM          = ollama
        RESEARCHER_OLLAMA_MODEL = qwen3-coder:32b
        RESEARCHER_ANTHROPIC_MODEL (if llm=claude)

    Brain defaults:
        BRAIN_LLM               = claude
        BRAIN_ANTHROPIC_MODEL   = (ANTHROPIC_MODEL env, then claude-haiku-4-5)
        BRAIN_OLLAMA_MODEL (if llm=ollama)

    Args:
        researcher_llm: LLM backend for the Researcher (``'ollama'`` or ``'claude'``).
        researcher_ollama_model: Ollama model override for the Researcher.
        researcher_anthropic_model: Anthropic model override for the Researcher.
        brain_llm: LLM backend for the Brain (``'claude'`` or ``'ollama'``).
        brain_anthropic_model: Anthropic model override for the Brain.
        brain_ollama_model: Ollama model override for the Brain.
        verbose: Print stage-transition log lines (default True).
    """
    researcher = create_researcher_agent(
        llm=researcher_llm,
        ollama_model=researcher_ollama_model,
        anthropic_model=researcher_anthropic_model,
    )
    brain = create_brain_agent(
        llm=brain_llm,
        anthropic_model=brain_anthropic_model,
        ollama_model=brain_ollama_model,
    )
    return Pipeline(researcher, brain, verbose=verbose, skip_brain=skip_brain)
