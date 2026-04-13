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
import shutil
import textwrap
from pathlib import Path
from typing import Any, List, Optional

from pydantic import BaseModel, Field, ValidationError
from strands import Agent

from src.agent import create_brain_agent, create_info_agent, create_planner_agent, create_researcher_agent, create_triage_agent, _settings
from src.utils.chat_summary import summarize_conversation
from src.utils.comfyui_interrupt_hook import INTERRUPT_NAME
from src.utils.comfyui_poller import poll_comfyui_job as _poll_comfyui_job
from src.utils.costs import compute_cost_from_usage
from src.utils.models import AgentSession, ChatSummary, MessageIntent, TriageResult
from src.utils.triage import triage as _triage, route as _route
from src.utils.workflow_signal import clear_and_get as _get_workflow_signal
from src.executor import execute_workflow as _execute_workflow, execute_workflows_batch as _execute_workflows_batch


# ---------------------------------------------------------------------------
# Brainbriefing schema (Pydantic) — mirrors config/brainbrief_example.json
# ---------------------------------------------------------------------------

class BriefInputImage(BaseModel):
    """Lightweight reference to an input image (filename only)."""
    filename: str = Field(description="Filename of the asset")


class InputImage(BaseModel):
    """A single input image/video asset with full ComfyUI node binding."""
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
    input_images: List[BriefInputImage] = Field(default_factory=list, description="Lightweight list of input image assets (filename + path)")
    input_nodes: List[InputImage] = Field(default_factory=list, description="Full ComfyUI node bindings for each input image")
    input_image_count: int = Field(default=0, description="Must equal len(input_images)")
    output_nodes: List[OutputNode] = Field(default_factory=list, description="Output nodes from the workflow with their save paths")
    resolution_width: Optional[Any] = Field(default=None, description="Image width in pixels")
    resolution_height: Optional[Any] = Field(default=None, description="Image height in pixels")
    prompt: BriefPrompt
    count_iter: int = Field(default=1, description="Number of batch iterations to generate (1 = single run, N > 1 = batch)")
    variations: bool = Field(default=False, description="True when each iteration should use a distinct prompt from multiprompt.json")
    positive_prompt_node_id: Optional[str] = Field(default=None, description="ComfyUI node ID of the positive prompt text node (used to splice per-variation prompts into workflow copies)")
    # notes_for_executor: Optional[str] = Field(default=None, description="Additional notes for the Brain")


# ---------------------------------------------------------------------------
# Multiprompt variations helper
# ---------------------------------------------------------------------------

# Canonical path where the image-batch skill writes variation prompts.
_MULTIPROMPT_PATH = Path("output/_workflows/multiprompt.json")


def _apply_multiprompt_variations(
    base_workflow_path: str,
    positive_prompt_node_id: str,
    *,
    verbose: bool = True,
) -> list[str]:
    """Expand one base workflow into N per-variation copies using multiprompt.json.

    When ``count_iter > 1`` **and** ``variations == True``, the image-batch
    skill writes ``output/_workflows/multiprompt.json`` with one key per
    prompt (``prompt1`` … ``promptN``).  This helper:

    1. Reads that file.
    2. Patches the base workflow in-place with ``prompt1``.
    3. Creates a copy of the base for each remaining prompt and patches it.
    4. Returns the ordered list of all workflow paths (base first).

    If ``multiprompt.json`` is absent or contains fewer than 2 entries the
    base workflow path is returned unchanged (single-workflow passthrough).

    Args:
        base_workflow_path:     Absolute path to the validated base workflow.
        positive_prompt_node_id: Node ID whose ``inputs.text`` field receives
                                 the per-variation prompt text.
        verbose:                 Log progress to stdout when True.
    """
    mp_file = _MULTIPROMPT_PATH
    if not mp_file.exists():
        if verbose:
            print(f"[pipeline] multiprompt.json not found at {mp_file} — skipping variation expansion.")
        return [base_workflow_path]

    try:
        prompts_data: dict = json.loads(mp_file.read_text(encoding="utf-8"))
    except Exception as exc:
        if verbose:
            print(f"[pipeline] WARNING: could not parse multiprompt.json — {exc}")
        return [base_workflow_path]

    # Support both formats:
    #   {"prompts": ["p1", "p2", ...]}   ← Brain/image-batch skill output
    #   {"prompt1": "p1", "prompt2": "p2", ...}  ← legacy flat format
    if "prompts" in prompts_data and isinstance(prompts_data["prompts"], list):
        prompts: list[str] = [p for p in prompts_data["prompts"] if isinstance(p, str)]
    else:
        prompts = [v for v in prompts_data.values() if isinstance(v, str)]

    if len(prompts) < 2:
        if verbose:
            print("[pipeline] multiprompt.json has < 2 entries — skipping variation expansion.")
        return [base_workflow_path]

    if verbose:
        print(f"[pipeline] Expanding {len(prompts)} variation prompts onto workflows …")

    base = Path(base_workflow_path)
    all_paths: list[str] = []

    for idx, prompt_text in enumerate(prompts, 1):
        if idx == 1:
            target = base  # patch the original in-place
        else:
            stem_clean = re.sub(r"_var_\d+$", "", base.stem)
            dest = base.parent / f"{stem_clean}_var_{idx:03d}.json"
            shutil.copy2(base, dest)
            target = dest

        try:
            data: dict = json.loads(target.read_text(encoding="utf-8"))
            node = data.get(str(positive_prompt_node_id))
            if node is None:
                if verbose:
                    print(f"[pipeline] WARNING: prompt node '{positive_prompt_node_id}' not found "
                          f"in {target.name} — skipping prompt patch for variation {idx}.")
            else:
                node.setdefault("inputs", {})["text"] = prompt_text
                target.write_text(json.dumps(data, indent=2), encoding="utf-8")
                if verbose:
                    print(f"[pipeline] variation {idx}/{len(prompts)} → {target.name}")
        except Exception as exc:
            if verbose:
                print(f"[pipeline] WARNING: could not patch variation {idx} — {exc}")

        all_paths.append(str(target))

    # Clean up the multiprompt.json so it doesn't bleed into the next pipeline run.
    try:
        mp_file.unlink()
    except Exception:
        pass

    return all_paths


# ---------------------------------------------------------------------------
# JSON extraction helper
# ---------------------------------------------------------------------------

def _extract_json(text: str) -> str | None:
    """Pull the first JSON object out of *text*, even if wrapped in a code fence.

    Strips ``<think>…</think>`` reasoning blocks emitted by Ollama models
    (e.g. qwen3) before scanning for JSON, so that JSON examples that appear
    inside the thinking block are never mistaken for the brainbriefing payload.
    """
    # Remove <think>...</think> blocks (qwen3 / DeepSeek reasoning traces)
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

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

    def __init__(
        self,
        researcher: Agent,
        brain: Agent,
        *,
        info_agent: Agent | None = None,
        triage_agent: Agent | None = None,
        planner_agent: Agent | None = None,
        verbose: bool = True,
        skip_brain: bool = False,
        info_context: dict | None = None,
        session_id: str = "default",
    ) -> None:
        self._researcher = researcher
        self._brain = brain
        self._info_agent: Agent = info_agent or create_info_agent()
        self._triage_agent: Agent = triage_agent or create_triage_agent()
        self._planner_agent: Agent = planner_agent or create_planner_agent()
        self._verbose = verbose
        self._skip_brain = skip_brain
        self._info_context: dict = info_context or {}
        self._session: AgentSession = AgentSession(session_id=session_id)
        # Brainbriefing JSON from the most recent Researcher run; used by the
        # Executor for Vision QA comparison in follow-up / feedback-loop rounds.
        self._last_brainbriefing_json: str | None = None

    def _should_skip_brain(self) -> bool:
        if self._skip_brain:
            return True
        return str(os.environ.get("PIPELINE_SKIP_BRAIN", "false")).strip().lower() in (
            "1", "true", "yes", "on"
        )

    # The Slack server and main.py both do:  response = agent(user_input)
    def __call__(self, user_input, **kwargs: Any) -> str:
        return self.run(user_input, **kwargs)

    # Delegate metric access to the Brain so the Slack server can read
    # token usage summaries just as it would from a plain Strands Agent.
    @property
    def event_loop_metrics(self):  # noqa: ANN201
        return self._brain.event_loop_metrics

    def run(self, user_input, **_: Any) -> str:
        """Run the full pipeline for *user_input* and return the Brain's response.

        Triage runs first to classify intent, then routes to:
        - ``researcher`` (new_request / low-confidence): full Researcher → Brain flow
        - ``brain`` (param_tweak / chain / feedback): direct Brain follow-up
        - ``answer`` (info_query): return the triage answer directly
        - ``log_warning`` (low-confidence fallback): treat as new_request
        """
        user_text = self._extract_text(user_input)
        user_text = self._annotate_attachments(user_input, user_text)
        triage_result = asyncio.run(_triage(user_text, self._session, self._info_context, self._triage_agent))
        handler = _route(triage_result)

        if self._verbose:
            print(f"[pipeline] Triage → intent={triage_result.intent.value},"
                  f" confidence={triage_result.confidence:.2f}, handler={handler}")

        if handler == "answer":
            if self._verbose:
                print("[pipeline] info_query → Info agent")
            return str(self._info_agent(user_text))

        if handler == "needs_image":
            if self._verbose:
                print("[pipeline] needs_image → handoff to user (missing image)")
            self._record_chat_summary(user_text, triage_result, status="needs_image")
            return triage_result.response or (
                "It looks like your request requires an input image, but I don't see one attached. "
                "Please share the image you'd like me to work with and I'll get started!"
            )

        if handler == "brain":
            # Follow-up: skip Researcher, send directly to Brain
            self._ensure_clean_history()
            brain_prompt = self._build_followup_prompt(user_text, triage_result)
            brain_response = str(self._brain(brain_prompt))
            self._session.follow_up_count += 1
            # Executor handoff: Brain signals a (re-)assembled workflow is ready
            workflow_paths = _get_workflow_signal()
            workflow_paths = self._expand_variations(workflow_paths, self._last_brainbriefing_json or "")
            executor_paths: list[str] = []
            if workflow_paths:
                count = len(workflow_paths)
                if self._verbose:
                    tag = f"{count} workflows (batch)" if count > 1 else workflow_paths[0]
                    print(f"[pipeline] Brain (follow-up) signaled {tag} ready.")
                executor_lines, executor_paths = asyncio.run(
                    self._drain_executor_batch(
                        workflow_paths,
                        self._last_brainbriefing_json or "",
                        user_message=user_text,
                    )
                )
                if executor_paths:
                    self._session.current_output_paths[:] = executor_paths
                if executor_lines:
                    brain_response = brain_response + "\n\n" + "\n".join(executor_lines)
            self._record_chat_summary(user_text, triage_result, status="completed")
            asyncio.run(self._compress_brain_history(extra_output_paths=executor_paths))
            if self._verbose:
                print("[pipeline] Brain (follow-up) finished.")
            return brain_response

        if handler == "planner":
            return self._run_planned_request(user_text, triage_result)

        # handler == "researcher" or "log_warning" → full Researcher → Brain flow
        raw_json, error, researcher_output = self._run_researcher(user_input)
        if error:
            self._record_chat_summary(user_text, triage_result, status="error")
            return error

        if self._should_skip_brain():
            if self._verbose:
                print("[pipeline] Skipping Brain stage; returning Researcher output.")
            return researcher_output

        self._last_brainbriefing_json = raw_json
        brain_prompt = self._build_brain_prompt(raw_json)
        self._ensure_clean_history()
        brain_response = str(self._brain(brain_prompt))
        # Executor handoff: Brain signals the assembled workflow(s) are ready
        workflow_paths_r = _get_workflow_signal()
        # Expand variation prompts from multiprompt.json if applicable
        workflow_paths_r = self._expand_variations(workflow_paths_r, raw_json)
        executor_paths_r: list[str] = []
        if workflow_paths_r:
            count = len(workflow_paths_r)
            if self._verbose:
                tag = f"{count} workflows (batch)" if count > 1 else workflow_paths_r[0]
                print(f"[pipeline] Brain signaled {tag} ready.")
            executor_lines_r, executor_paths_r = asyncio.run(
                self._drain_executor_batch(
                    workflow_paths_r,
                    raw_json,
                    user_message=user_text,
                )
            )
            if executor_paths_r:
                self._session.current_output_paths[:] = executor_paths_r
            if executor_lines_r:
                brain_response = brain_response + "\n\n" + "\n".join(executor_lines_r)
        self._record_chat_summary(user_text, triage_result, status="completed", raw_json=raw_json)
        asyncio.run(self._compress_brain_history(extra_output_paths=executor_paths_r))
        if self._verbose:
            print("[pipeline] Brain finished.")
        return brain_response

    async def stream_async(self, user_input):  # noqa: ANN201
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
        # Stage 0 – Triage (classify intent before any agent is called)
        user_text = self._extract_text(user_input)
        user_text = self._annotate_attachments(user_input, user_text)
        triage_result = await _triage(user_text, self._session, self._info_context, self._triage_agent)
        handler = _route(triage_result)

        if self._verbose:
            print(f"[pipeline:stream] Triage → intent={triage_result.intent.value},"
                  f" confidence={triage_result.confidence:.2f}, handler={handler}")

        if handler == "answer":
            if self._verbose:
                print("[pipeline:stream] info_query → Info agent (streamed)")
            async for event in self._info_agent.stream_async(user_text):
                yield event
            return

        if handler == "needs_image":
            if self._verbose:
                print("[pipeline:stream] needs_image → handoff to user (missing image)")
            self._record_chat_summary(user_text, triage_result, status="needs_image")
            message = triage_result.response or (
                "It looks like your request requires an input image, but I don't see one attached. "
                "Please share the image you'd like me to work with and I'll get started!"
            )
            yield {"data": message}
            return

        if handler == "brain":
            # Follow-up: skip Researcher, send directly to Brain (streamed)
            self._ensure_clean_history()
            brain_prompt = self._build_followup_prompt(user_text, triage_result)
            self._session.follow_up_count += 1
            async for event in self._brain.stream_async(brain_prompt):
                yield event
            # Executor handoff: stream execution events back to Slack
            workflow_paths_fu = _get_workflow_signal()
            workflow_paths_fu = self._expand_variations(workflow_paths_fu, self._last_brainbriefing_json or "")
            executor_paths_fu: list[str] = []
            if workflow_paths_fu:
                count = len(workflow_paths_fu)
                if self._verbose:
                    tag = f"{count} workflows (batch)" if count > 1 else workflow_paths_fu[0]
                    print(f"[pipeline:stream] Brain (follow-up) signaled {tag} ready.")
                hdr = f"batch of {count} workflows" if count > 1 else "workflow"
                yield {"data": f"\n\n_⚙️ Handing off to executor ({hdr})…_"}
                async for line in _execute_workflows_batch(
                    workflow_paths_fu,
                    self._last_brainbriefing_json or "",
                    user_message=user_text,
                    verbose=self._verbose,
                    collected_paths=executor_paths_fu,
                ):
                    yield {"data": f"\n{line}"}
            if executor_paths_fu:
                self._session.current_output_paths[:] = executor_paths_fu
            self._record_chat_summary(user_text, triage_result, status="completed")
            await self._compress_brain_history(extra_output_paths=executor_paths_fu)
            return

        if handler == "planner":
            async for event in self._stream_planned_request(user_text, triage_result):
                yield event
            return

        # handler == "researcher" or "log_warning" → full Researcher → Brain flow
        # Stage 1 – Researcher (synchronous; fast pattern-matching turn)
        if self._verbose:
            print("[pipeline:stream] Stage 1 – Researcher resolving spec …")
        raw_json, error, researcher_output = self._run_researcher(user_input)
        if error:
            self._record_chat_summary(user_text, triage_result, status="error")
            yield {"data": error}
            return

        self._last_brainbriefing_json = raw_json

        if self._should_skip_brain():
            if self._verbose:
                print("[pipeline:stream] Skipping Brain stage; returning Researcher output.")
            yield {"data": researcher_output}
            return

        # Stage 2 – Brain (streamed, with optional ComfyUI interrupt handling)
        if self._verbose:
            print("[pipeline:stream] Stage 2 – Brain streaming …")
        self._ensure_clean_history()
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
                # Normal Brain completion — no ComfyUI interrupt pending.
                # Stage 3 – Executor: submit, poll, QA, Slack, save
                workflow_paths_s = _get_workflow_signal()
                workflow_paths_s = self._expand_variations(workflow_paths_s, raw_json)
                executor_paths_s: list[str] = []
                if workflow_paths_s:
                    count = len(workflow_paths_s)
                    if self._verbose:
                        tag = f"{count} workflows (batch)" if count > 1 else workflow_paths_s[0]
                        print(f"[pipeline:stream] Brain signaled {tag} ready.")
                    hdr = f"batch of {count} workflows" if count > 1 else "workflow"
                    yield {"data": f"\n\n_⚙️ Handing off to executor ({hdr})…_"}
                    async for line in _execute_workflows_batch(
                        workflow_paths_s,
                        raw_json,
                        user_message=user_text,
                        verbose=self._verbose,
                        collected_paths=executor_paths_s,
                    ):
                        yield {"data": f"\n{line}"}
                if executor_paths_s:
                    self._session.current_output_paths[:] = executor_paths_s
                self._record_chat_summary(user_text, triage_result, status="completed", raw_json=raw_json)
                await self._compress_brain_history(extra_output_paths=executor_paths_s)
                if self._verbose:
                    print("[pipeline:stream] Brain finished.")
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

    # ── Planner helpers ──────────────────────────────────────────────── #

    def _run_planner(self, user_text: str) -> list[dict[str, str]]:
        """Call the Planner agent to decompose *user_text* into ordered steps.

        Returns a list of ``{"request": str, "description": str}`` dicts on
        success, or an empty list when parsing fails (the caller falls back to
        treating the request as a plain ``new_request``).
        """
        raw: str = str(self._planner_agent(user_text))
        # Reset single-turn history immediately.
        try:
            self._planner_agent.conversation_manager.messages.clear()
        except AttributeError:
            try:
                self._planner_agent.conversation_manager._messages.clear()  # noqa: SLF001
            except AttributeError:
                pass

        json_str = _extract_json(raw) or raw
        try:
            parsed = json.loads(json_str)
            steps = parsed.get("steps", [])
            if not isinstance(steps, list) or len(steps) < 2:
                if self._verbose:
                    print(f"[planner] WARNING: plan has {len(steps)} step(s) — need ≥ 2. "
                          "Falling back to researcher.")
                return []
            # Validate each step has at least a 'request' field.
            validated: list[dict[str, str]] = []
            for s in steps:
                if isinstance(s, dict) and "request" in s:
                    validated.append({
                        "request": str(s["request"]),
                        "description": str(s.get("description", f"Step {len(validated) + 1}")),
                    })
            if len(validated) < 2:
                if self._verbose:
                    print("[planner] WARNING: could not validate ≥ 2 steps. Falling back.")
                return []
            return validated
        except (json.JSONDecodeError, KeyError, TypeError) as exc:
            if self._verbose:
                print(f"[planner] WARNING: plan parse failed ({exc}). Falling back to researcher.")
            return []

    def _inject_context_into_step(self, step_request: str, step_index: int) -> str:
        """Prepend previous-step output paths into *step_request* when available.

        Steps 2+ that reference "the previous step" will have the actual
        output file paths injected so the Researcher can resolve them directly.
        """
        if step_index == 0 or not self._session.current_output_paths:
            return step_request
        paths_hint = ", ".join(self._session.current_output_paths)
        return (
            f"{step_request}\n"
            f"[Previous step output(s): {paths_hint}]"
        )

    def _run_planned_request(self, user_text: str, triage_result: TriageResult) -> str:
        """Execute a multi-step plan synchronously; return a combined summary string."""
        if self._verbose:
            print("[pipeline] Planner — decomposing multi-step request …")
        steps = self._run_planner(user_text)

        # Fallback: treat as a plain new_request when planning fails.
        if not steps:
            if self._verbose:
                print("[pipeline] Planner fallback → researcher path")
            raw_json, error, researcher_output = self._run_researcher(user_text)
            if error:
                self._record_chat_summary(user_text, triage_result, status="error")
                return error
            self._last_brainbriefing_json = raw_json
            if self._should_skip_brain():
                return researcher_output
            self._ensure_clean_history()
            brain_response = str(self._brain(self._build_brain_prompt(raw_json)))
            wf = _get_workflow_signal()
            wf = self._expand_variations(wf, raw_json)
            ep: list[str] = []
            if wf:
                _, ep = asyncio.run(
                    self._drain_executor_batch(wf, raw_json, user_message=user_text)
                )
                if ep:
                    self._session.current_output_paths[:] = ep
            self._record_chat_summary(user_text, triage_result, status="completed", raw_json=raw_json)
            asyncio.run(self._compress_brain_history(extra_output_paths=ep))
            return brain_response

        total = len(steps)
        if self._verbose:
            print(f"[pipeline] Plan has {total} step(s):")
            for i, s in enumerate(steps, 1):
                print(f"  {i}. {s['description']}")

        step_results: list[str] = []

        for idx, step in enumerate(steps):
            description = step["description"]
            step_req = self._inject_context_into_step(step["request"], idx)

            if self._verbose:
                print(f"\n[pipeline] ── Plan step {idx + 1}/{total}: {description} ──")

            raw_json, error, researcher_output = self._run_researcher(step_req)
            if error:
                msg = f"Step {idx + 1} ({description}) failed: {error}"
                if self._verbose:
                    print(f"[pipeline] {msg}")
                step_results.append(msg)
                # Abort remaining steps when the researcher fails.
                break

            self._last_brainbriefing_json = raw_json

            if self._should_skip_brain():
                step_results.append(f"Step {idx + 1}: {researcher_output}")
                continue

            self._ensure_clean_history()
            brain_response = str(self._brain(self._build_brain_prompt(raw_json)))
            wf_paths = _get_workflow_signal()
            wf_paths = self._expand_variations(wf_paths, raw_json)
            exec_paths: list[str] = []
            if wf_paths:
                count = len(wf_paths)
                if self._verbose:
                    tag = f"{count} workflows (batch)" if count > 1 else wf_paths[0]
                    print(f"[pipeline] Step {idx + 1} Brain signaled {tag} ready.")
                exec_lines, exec_paths = asyncio.run(
                    self._drain_executor_batch(
                        wf_paths, raw_json, user_message=step_req
                    )
                )
                if exec_paths:
                    self._session.current_output_paths[:] = exec_paths
                if exec_lines:
                    brain_response = brain_response + "\n\n" + "\n".join(exec_lines)

            self._record_chat_summary(step_req, triage_result, status="completed", raw_json=raw_json)
            asyncio.run(self._compress_brain_history(extra_output_paths=exec_paths))

            step_results.append(f"**Step {idx + 1} — {description}**\n{brain_response}")

            if self._verbose:
                print(f"[pipeline] Step {idx + 1}/{total} finished.")

        combined = "\n\n---\n\n".join(step_results)
        if self._verbose:
            print(f"[pipeline] Planned execution complete ({total} step(s)).")
        return combined

    async def _stream_planned_request(
        self,
        user_text: str,
        triage_result: TriageResult,
    ):
        """Stream a multi-step plan; yields Strands-compatible event dicts."""
        if self._verbose:
            print("[pipeline:stream] Planner — decomposing multi-step request …")
        steps = self._run_planner(user_text)

        # Fallback: treat as a plain researcher path when planning fails.
        if not steps:
            if self._verbose:
                print("[pipeline:stream] Planner fallback → researcher path")
            raw_json, error, researcher_output = self._run_researcher(user_text)
            if error:
                self._record_chat_summary(user_text, triage_result, status="error")
                yield {"data": error}
                return
            self._last_brainbriefing_json = raw_json
            if self._should_skip_brain():
                yield {"data": researcher_output}
                return
            self._ensure_clean_history()
            async for event in self._brain.stream_async(self._build_brain_prompt(raw_json)):
                yield event
            wf = _get_workflow_signal()
            wf = self._expand_variations(wf, raw_json)
            ep: list[str] = []
            if wf:
                yield {"data": "\n\n_⚙️ Handing off to executor…_"}
                async for line in _execute_workflows_batch(
                    wf, raw_json,
                    user_message=user_text,
                    verbose=self._verbose,
                    collected_paths=ep,
                ):
                    yield {"data": f"\n{line}"}
                if ep:
                    self._session.current_output_paths[:] = ep
            self._record_chat_summary(user_text, triage_result, status="completed", raw_json=raw_json)
            await self._compress_brain_history(extra_output_paths=ep)
            return

        total = len(steps)
        yield {"data": f"_🗂️ Plan ready — {total} step(s) to execute._\n"}
        if self._verbose:
            print(f"[pipeline:stream] Plan has {total} step(s):")
            for i, s in enumerate(steps, 1):
                print(f"  {i}. {s['description']}")

        for idx, step in enumerate(steps):
            description = step["description"]
            step_req = self._inject_context_into_step(step["request"], idx)

            yield {"data": f"\n\n**Step {idx + 1}/{total} — {description}**\n"}
            if self._verbose:
                print(f"\n[pipeline:stream] ── Plan step {idx + 1}/{total}: {description} ──")

            raw_json, error, researcher_output = self._run_researcher(step_req)
            if error:
                yield {"data": f"\n❌ Step {idx + 1} failed: {error}"}
                if self._verbose:
                    print(f"[pipeline:stream] Step {idx + 1} researcher error: {error}")
                break

            self._last_brainbriefing_json = raw_json

            if self._should_skip_brain():
                yield {"data": researcher_output}
                continue

            self._ensure_clean_history()
            async for event in self._brain.stream_async(self._build_brain_prompt(raw_json)):
                yield event

            wf_paths = _get_workflow_signal()
            wf_paths = self._expand_variations(wf_paths, raw_json)
            exec_paths: list[str] = []
            qa_step_failed = False
            if wf_paths:
                count = len(wf_paths)
                hdr = f"batch of {count} workflows" if count > 1 else "workflow"
                yield {"data": f"\n\n_⚙️ Handing off to executor ({hdr})…_"}
                async for line in _execute_workflows_batch(
                    wf_paths, raw_json,
                    user_message=step_req,
                    verbose=self._verbose,
                    collected_paths=exec_paths,
                ):
                    if isinstance(line, dict) and line.get("qa_fail"):
                        # Forward the QA failure event up to the UI layer.
                        yield line
                        qa_step_failed = True
                        break
                    yield {"data": f"\n{line}"}
                if exec_paths:
                    self._session.current_output_paths[:] = exec_paths

            if qa_step_failed:
                self._record_chat_summary(step_req, triage_result, status="qa_failed", raw_json=raw_json)
                if self._verbose:
                    print(f"[pipeline:stream] Step {idx + 1}/{total} QA failed — aborting plan.")
                break  # stop processing further steps

            self._record_chat_summary(step_req, triage_result, status="completed", raw_json=raw_json)
            await self._compress_brain_history(extra_output_paths=exec_paths)

            if self._verbose:
                print(f"[pipeline:stream] Step {idx + 1}/{total} finished.")

        if self._verbose:
            print(f"[pipeline:stream] Planned execution complete ({total} step(s)).")

    # ── Triage helpers ───────────────────────────────────────────────── #

    @staticmethod
    def _extract_text(user_input: Any) -> str:
        """Extract a plain-text string from a str or multimodal content-block list."""
        if isinstance(user_input, list):
            return "\n".join(block["text"] for block in user_input if "text" in block)
        return str(user_input)

    @staticmethod
    def _annotate_attachments(user_input: Any, user_text: str) -> str:
        """Append an attachment hint to *user_text* so triage knows images are present.

        Triage only receives the plain-text portion of the request, so without
        this hint it would classify image-edit requests as ``needs_image`` even
        when the caller already attached image content blocks or embedded a file
        path directly in their CLI message.
        """
        _IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".tiff", ".tif"}
        _VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}

        if not isinstance(user_input, list):
            # CLI / plain-text mode: scan for image/video file paths in the message.
            # Extract tokens that could be paths (quoted or unquoted).
            tokens = re.findall(r'"([^"]+)"|\'([^\']+)\'|(\S+)', user_text)
            flat = [t for group in tokens for t in group if t]
            img_paths = [t for t in flat if Path(t).suffix.lower() in _IMAGE_EXTS and os.path.isfile(t)]
            vid_paths = [t for t in flat if Path(t).suffix.lower() in _VIDEO_EXTS and os.path.isfile(t)]
            parts: list[str] = []
            if img_paths:
                parts.append(f"{len(img_paths)} image{'s' if len(img_paths) > 1 else ''}")
            if vid_paths:
                parts.append(f"{len(vid_paths)} video{'s' if len(vid_paths) > 1 else ''}")
            if parts:
                return user_text + f"\n[Attached: {', '.join(parts)}]"
            return user_text

        img_count = sum(1 for b in user_input if "image" in b)
        vid_count = sum(1 for b in user_input if "video" in b)
        parts = []
        if img_count:
            parts.append(f"{img_count} image{'s' if img_count > 1 else ''}")
        if vid_count:
            parts.append(f"{vid_count} video{'s' if vid_count > 1 else ''}")
        if parts:
            return user_text + f"\n[Attached: {', '.join(parts)}]"
        return user_text

    def _build_followup_prompt(self, user_text: str, triage_result: TriageResult) -> str:
        """Prompt for Brain when handling a follow-up (no Researcher pass needed).

        For ``feedback`` intent the raw user message is returned verbatim — the
        Brain receives it exactly as the user wrote it, with no wrapper.
        For all other follow-up intents a compact context block is prepended.
        """
        if triage_result.intent == MessageIntent.feedback:
            # The user's feedback message IS the new prompt for the Brain.
            return user_text

        context_lines: list[str] = []
        if self._session.chat_summaries:
            last = self._session.chat_summaries[-1]
            context_lines.append(f"Last workflow: {last.workflow_name}")
            context_lines.append(f"Last status: {last.status}")
        if self._session.current_output_paths:
            context_lines.append(
                f"Current outputs: {', '.join(self._session.current_output_paths)}"
            )
        context_block = ("\n".join(context_lines) + "\n\n") if context_lines else ""
        return textwrap.dedent(f"""\
            Follow-up request (intent: {triage_result.intent.value}):

            {context_block}{user_text}

            Apply the requested change directly, reusing the current session context.
        """).strip()

    def _ensure_clean_history(self) -> None:
        """Sanitize the Brain's message list before an invocation.

        If a previous call crashed mid-tool-execution, the Brain's
        messages may contain orphaned ``toolResult`` or ``toolUse``
        blocks.  This guard cleans them before the next API call so
        the Anthropic API doesn't reject the request.
        """
        msgs = self._brain.messages
        if not msgs:
            return
        cleaned = self._sanitize_messages(list(msgs))
        if len(cleaned) != len(msgs):
            if self._verbose:
                removed = len(msgs) - len(cleaned)
                print(f"[pipeline] Sanitized Brain history: removed {removed} "
                      f"orphaned tool message(s).")
            self._brain.messages[:] = cleaned

    @staticmethod
    def _sanitize_messages(messages: list[dict]) -> list[dict]:
        """Ensure *messages* don't start with orphaned ``toolResult`` blocks.

        The Anthropic API requires every ``tool_result`` content block to have
        a corresponding ``tool_use`` block in the immediately preceding
        assistant message.  When we slice message history (e.g. fallback
        ``messages[-4:]``), the slice may begin with a ``user`` message
        carrying ``toolResult`` blocks whose matching ``toolUse`` was in an
        earlier (now-discarded) assistant message.

        This helper trims leading messages until the first message no longer
        contains orphaned ``toolResult`` (or ``toolUse`` without a following
        ``toolResult``) blocks.
        """
        while messages:
            first = messages[0]
            content = first.get("content", [])
            if isinstance(content, list):
                has_tool_result = any(
                    isinstance(b, dict) and "toolResult" in b for b in content
                )
                if has_tool_result:
                    messages = messages[1:]
                    continue

                # An assistant message with toolUse but no following
                # toolResult message is also invalid.
                has_tool_use = any(
                    isinstance(b, dict) and "toolUse" in b for b in content
                )
                if has_tool_use:
                    if len(messages) < 2 or not any(
                        isinstance(b, dict) and "toolResult" in b
                        for b in (messages[1].get("content", []) if isinstance(messages[1].get("content", []), list) else [])
                    ):
                        messages = messages[1:]
                        continue
            break
        return messages

    async def _compress_brain_history(self, extra_output_paths: list[str] | None = None) -> None:
        """Summarise the Brain's conversation and replace its messages.

        After every Brain invocation the full message history is compressed
        into a single compact summary via ``summarize_conversation``.  The
        Brain's ``messages`` list is then replaced with a single assistant
        message containing that summary.  This ensures that the next agent
        call carries only the summary — not the full, token-heavy history.

        ``extra_output_paths`` are executor-produced file paths that don't
        appear in the Brain's message history (the executor runs outside the
        Brain loop).  They are injected into the summary so the next round
        can reference the generated outputs.
        """
        messages = self._brain.messages
        if not messages:
            return

        if self._verbose:
            msg_count = len(messages)
            print(f"[pipeline] Compressing Brain history ({msg_count} messages) …")

        try:
            summary = await summarize_conversation(messages, extra_output_paths=extra_output_paths)
        except Exception as exc:
            if self._verbose:
                print(f"[pipeline] WARNING: conversation summarisation failed ({exc}); "
                      "keeping last 4 messages as fallback.")
            # Fallback: keep only the last few messages to cap token growth.
            # Sanitize to avoid orphaned toolResult blocks at the start.
            self._brain.messages[:] = self._sanitize_messages(messages[-4:])
            return

        if not summary:
            if self._verbose:
                print("[pipeline] Empty summary returned; clearing history.")
            self._brain.messages.clear()
            return

        # Append token-usage and cost lines to the summary
        try:
            usage = self._brain.event_loop_metrics.accumulated_usage
            in_tok = usage.get("inputTokens", 0)
            out_tok = usage.get("outputTokens", 0)
            cache_read = usage.get("cacheReadInputTokens", 0)
            cache_write = usage.get("cacheWriteInputTokens", 0)
            token_parts = [f"{in_tok:,} in", f"{out_tok:,} out"]
            if cache_read:
                token_parts.append(f"{cache_read:,} cache hit")
            if cache_write:
                token_parts.append(f"{cache_write:,} cache write")
            cost_val, total_tokens = compute_cost_from_usage(usage, self._brain)
            cost_lines = (
                f"TOKENS: {' / '.join(token_parts)} (total: {total_tokens:,})\n"
                f"COST: ${cost_val:.2f}"
            )
        except Exception:
            cost_lines = ""

        if cost_lines:
            summary = summary.rstrip() + "\n" + cost_lines

        print(f"[pipeline] Chat summary:\n{summary}\n")

        # Replace the entire history with a single summary message.
        # Using an "assistant" message so the agent treats it as its own
        # prior context rather than a new user instruction.
        self._brain.messages[:] = [
            {
                "role": "user",
                "content": [
                    {
                        "text": (
                            "[CONVERSATION SUMMARY FROM PRIOR ROUND]\n\n"
                            f"{summary}\n\n"
                            "[END OF SUMMARY — use this context for follow-up requests]"
                        ),
                    }
                ],
            },
            {
                "role": "assistant",
                "content": [
                    {
                        "text": "Understood. I have the context from the prior round and am ready for the next request.",
                    }
                ],
            },
        ]

        if self._verbose:
            print(f"[pipeline] Brain history compressed → {len(summary)} chars summary.")

    def _record_chat_summary(
        self,
        user_text: str,
        triage_result: TriageResult,
        *,
        status: str,
        raw_json: str | None = None,
    ) -> None:
        """Append a ChatSummary to the session after each pipeline invocation."""
        workflow_name = "unknown"
        if raw_json:
            try:
                workflow_name = (
                    json.loads(raw_json).get("template", {}).get("name") or "unknown"
                )
            except Exception:
                pass
        self._session.chat_summaries.append(
            ChatSummary(
                workflow_name=workflow_name,
                output_paths=list(self._session.current_output_paths),
                user_intent=triage_result.intent.value,
                status=status,
            )
        )

    _MAX_RESEARCHER_RETRIES = 2  # up to 2 correction rounds after the first attempt

    def _run_researcher(self, user_input) -> tuple[str | None, str | None, str]:
        """Run the Researcher and return ``(raw_json, error_message, researcher_output)``.

        Calls the Researcher as a normal text agent (preserving its tool use),
        extracts the JSON from the response, then validates it with the
        ``BrainBriefing`` Pydantic model.  On success the model is re-serialised
        so the Brain always receives canonically formatted JSON.

        If validation fails, the error is fed back to the Researcher for up to
        ``_MAX_RESEARCHER_RETRIES`` correction rounds before giving up.

        ``user_input`` may be a plain string *or* a list of Strands content
        blocks (the multimodal format produced by the Slack server when the
        user attaches images/videos).  When it is a list, the image/video
        blocks are forwarded to the Researcher intact so it can visually
        inspect the attachments via its ``analyze_image`` tool.

        Returns ``(json_str, None, researcher_output)`` on success, ``(None, error_str, last_output)`` on failure.
        """
        # Build a text-only version of the user request for the preamble and
        # for retry prompts where we can only send plain text.
        if isinstance(user_input, list):
            text_parts = [block["text"] for block in user_input if "text" in block]
            user_text = "\n".join(text_parts)
        else:
            user_text = user_input

        researcher_prompt_text = textwrap.dedent(f"""
            User request:
            {user_text}

            Resolve all fields and output the brainbriefing JSON.
        """).strip()

        # Always pass only the text prompt to the Researcher.
        # When user_input is a multimodal list, the text block already contains
        # the on-disk paths of any attached files (added by slack_server._build_content_blocks).
        # The Researcher can call analyze_image(file_path=...) if it needs to inspect
        # an image — much cheaper than embedding raw bytes in every LLM call.
        first_attempt_input: Any = researcher_prompt_text

        last_error: str | None = None

        for attempt in range(1 + self._MAX_RESEARCHER_RETRIES):
            if attempt == 0:
                prompt = first_attempt_input
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

            Assemble and validate the ComfyUI workflow from this spec, then call
            `signal_workflow_ready(workflow_path)` as your final step.
            The pipeline will handle ComfyUI submission, completion polling,
            Vision QA (via Ollama), saving outputs to ./output, and Slack.
        """).strip()

    async def _drain_executor(
        self,
        workflow_path: str,
        brainbriefing_json: str,
        slack_channel_id: str = "",
        slack_thread_ts: str = "",
    ) -> tuple[list[str], list[str]]:
        """Drain the executor for a single workflow; return ``(status_lines, output_paths)``."""
        lines: list[str] = []
        output_paths: list[str] = []
        async for line in _execute_workflow(
            workflow_path,
            brainbriefing_json,
            slack_channel_id=slack_channel_id,
            slack_thread_ts=slack_thread_ts,
            verbose=self._verbose,
            collected_paths=output_paths,
        ):
            lines.append(line)
            if self._verbose:
                print(f"[executor] {line}")
        return lines, output_paths

    async def _drain_executor_batch(
        self,
        workflow_paths: list[str],
        brainbriefing_json: str,
        user_message: str = "",
    ) -> tuple[list[str], list[str]]:
        """Submit all workflows then poll + process each; return ``(status_lines, output_paths)``.

        Uses ``execute_workflows_batch`` (submit-all-then-poll) for any number
        of workflows, including a single one.

        ``user_message`` is forwarded to the Vision QA agent as the ground-truth
        reference.  It never enters any agent's context window or history.
        """
        lines: list[str] = []
        output_paths: list[str] = []
        async for line in _execute_workflows_batch(
            workflow_paths,
            brainbriefing_json,
            user_message=user_message,
            verbose=self._verbose,
            collected_paths=output_paths,
        ):
            lines.append(line)
            if self._verbose:
                print(f"[executor] {line}")
        return lines, output_paths

    def _expand_variations(
        self,
        workflow_paths: list[str],
        brainbriefing_json: str,
    ) -> list[str]:
        """Replace the base workflow list with per-variation copies when applicable.

        Conditions to activate:
        - Exactly **one** base workflow was signalled (the Brain's normal output
          in variations mode).
        - The brainbriefing has ``variations: true`` and ``count_iter > 1``.
        - ``positive_prompt_node_id`` is set so the pipeline knows which node
          to patch with each variation prompt.
        - ``output/_workflows/multiprompt.json`` exists (written by image-batch skill).

        When all conditions are met the single base workflow path is expanded to
        N paths (one per prompt in multiprompt.json).  If any condition fails the
        original list is returned unchanged so the executor still runs normally.
        """
        if not workflow_paths:
            return workflow_paths

        try:
            briefing: dict = json.loads(brainbriefing_json) if brainbriefing_json else {}
        except Exception:
            briefing = {}

        if not (briefing.get("variations") and briefing.get("count_iter", 1) > 1):
            return workflow_paths

        node_id = briefing.get("positive_prompt_node_id")
        if not node_id:
            if self._verbose:
                print("[pipeline] variations=true but positive_prompt_node_id is missing — "
                      "skipping multiprompt expansion.")
            return workflow_paths

        if not _MULTIPROMPT_PATH.exists():
            if self._verbose:
                print("[pipeline] variations=true but multiprompt.json not found — "
                      "running base workflow as-is.")
            return workflow_paths

        # Use only the first base workflow (Brain should signal exactly one)
        base_path = workflow_paths[0]
        expanded = _apply_multiprompt_variations(
            base_path,
            node_id,
            verbose=self._verbose,
        )
        if self._verbose and len(expanded) > 1:
            print(f"[pipeline] Variation expansion: 1 base → {len(expanded)} workflows.")
        return expanded

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
    triage_llm: str | None = None,
    triage_ollama_model: str | None = None,
    triage_anthropic_model: str | None = None,
    planner_llm: str | None = None,
    planner_ollama_model: str | None = None,
    planner_anthropic_model: str | None = None,
    verbose: bool = True,
    skip_brain: bool = False,
    info_context: dict | None = None,
    session_id: str = "default",
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

    Triage defaults:
        TRIAGE_LLM              = ollama  (reads llm.pipeline.triage from settings.json)
        TRIAGE_OLLAMA_MODEL     = (model from settings, then llm.pipeline.llm_functions)
        TRIAGE_ANTHROPIC_MODEL  (if llm=claude)

    Planner defaults:
        PLANNER_LLM             = (inherits from triage settings)
        PLANNER_OLLAMA_MODEL    = (model from settings, then llm.pipeline.llm_functions)
        PLANNER_ANTHROPIC_MODEL (if llm=claude)

    Args:
        researcher_llm: LLM backend for the Researcher (``'ollama'`` or ``'claude'``).
        researcher_ollama_model: Ollama model override for the Researcher.
        researcher_anthropic_model: Anthropic model override for the Researcher.
        brain_llm: LLM backend for the Brain (``'claude'`` or ``'ollama'``).
        brain_anthropic_model: Anthropic model override for the Brain.
        brain_ollama_model: Ollama model override for the Brain.
        triage_llm: LLM backend for the Triage agent (``'ollama'`` or ``'claude'``).
        triage_ollama_model: Ollama model override for the Triage agent.
        triage_anthropic_model: Anthropic model override for the Triage agent.
        planner_llm: LLM backend for the Planner agent (``'ollama'`` or ``'claude'``).
        planner_ollama_model: Ollama model override for the Planner agent.
        planner_anthropic_model: Anthropic model override for the Planner agent.
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
    info_agent = create_info_agent()
    triage_agent = create_triage_agent(
        llm=triage_llm,
        ollama_model=triage_ollama_model,
        anthropic_model=triage_anthropic_model,
    )
    planner_agent = create_planner_agent(
        llm=planner_llm,
        ollama_model=planner_ollama_model,
        anthropic_model=planner_anthropic_model,
    )
    return Pipeline(
        researcher,
        brain,
        info_agent=info_agent,
        triage_agent=triage_agent,
        planner_agent=planner_agent,
        verbose=verbose,
        skip_brain=skip_brain,
        info_context=info_context,
        session_id=session_id,
    )
