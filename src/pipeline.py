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

from src.agent import create_brain_agent, create_error_checker_agent, create_info_agent, create_planner_agent, create_researcher_agent, create_triage_agent, create_vision_agent, _settings
from src.tools.image_handling import set_vision_agent as _set_vision_agent
from src.utils.chat_summary import summarize_conversation, log_agent_messages
from src.utils.comfyui_interrupt_hook import INTERRUPT_NAME
from src.utils.comfyui_progress import stream_comfyui_job as _stream_comfyui_job
from src.utils.costs import compute_cost_from_usage
from src.utils.models import AgentSession, ChatSummary, MessageIntent, TriageResult
from src.utils.triage import triage as _triage, route as _route
from src.utils.workflow_signal import clear_and_get as _get_workflow_signal
from src.executor import execute_workflow as _execute_workflow, execute_workflows_batch as _execute_workflows_batch
from src.utils.memory import format_memories, memory_add, memory_search
from src.tools.memory_tools import set_session_id as _set_memory_session_id
from src.utils.learnings import count_tool_calls, maybe_run_learnings


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
_MULTIPROMPT_PATH = Path("output_workflows/multiprompt.json")
_OUTPUT_WORKFLOWS_DIR = Path("output_workflows")


def _latest_output_workflow() -> str | None:
    """Return the path of the most recently modified workflow JSON in output_workflows/."""
    try:
        jsons = sorted(
            (f for f in _OUTPUT_WORKFLOWS_DIR.glob("*.json") if f.stem != "multiprompt"),
            key=lambda f: f.stat().st_mtime,
            reverse=True,
        )
        return str(jsons[0]) if jsons else None
    except Exception:
        return None


def _apply_multiprompt_variations(
    base_workflow_path: str,
    positive_prompt_node_id: str,
    *,
    verbose: bool = True,
) -> list[str]:
    """Expand one base workflow into N per-variation copies using multiprompt.json.

    When ``count_iter > 1`` **and** ``variations == True``, the image-batch
    skill writes ``output_workflows/multiprompt.json`` with one key per
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
            print(f"pipeline: multiprompt.json not found at {mp_file} — skipping variation expansion.")
        return [base_workflow_path]

    try:
        prompts_data: dict = json.loads(mp_file.read_text(encoding="utf-8"))
    except Exception as exc:
        if verbose:
            print(f"pipeline: WARNING: could not parse multiprompt.json — {exc}")
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
            print("pipeline: multiprompt.json has < 2 entries — skipping variation expansion.")
        return [base_workflow_path]

    if verbose:
        print(f"pipeline: Expanding {len(prompts)} variation prompts onto workflows …")

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
                    print(f"pipeline: WARNING: prompt node '{positive_prompt_node_id}' not found "
                          f"in {target.name} — skipping prompt patch for variation {idx}.")
            else:
                node.setdefault("inputs", {})["text"] = prompt_text
                target.write_text(json.dumps(data, indent=2), encoding="utf-8")
                if verbose:
                    print(f"pipeline: variation {idx}/{len(prompts)} → {target.name}")
        except Exception as exc:
            if verbose:
                print(f"pipeline: WARNING: could not patch variation {idx} — {exc}")

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
# Per-turn aggregated metrics helper
# ---------------------------------------------------------------------------

class _TurnMetrics:
    """Aggregates token-usage dicts from all agents that ran in a single turn.

    Exposes ``accumulated_usage`` so that callers that do
    ``pipeline.event_loop_metrics.accumulated_usage`` receive a combined
    picture instead of only the Brain's tokens.
    """

    def __init__(self, usages: list) -> None:
        aggregated: dict[str, int] = {
            "inputTokens": 0,
            "outputTokens": 0,
            "cacheReadInputTokens": 0,
            "cacheWriteInputTokens": 0,
        }
        for usage, _ in usages:
            for k in aggregated:
                aggregated[k] += int(usage.get(k, 0) or 0)
        self.accumulated_usage: dict[str, int] = aggregated


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _is_affirmative(text: str) -> bool:
    """Return True when *text* looks like a yes/retry answer."""
    return text.strip().lower() in {"y", "yes", "retry", "repeat", "yep", "yup", "sure", "ok"}


# ---------------------------------------------------------------------------
# Pipeline callable
# ---------------------------------------------------------------------------

class Pipeline:
    """Chains Researcher → Brain with logging and JSON validation.

    Call ``pipeline(user_input)`` just like a Strands Agent.
    The Researcher runs once per call (stateless); the Brain keeps
    a sliding-window conversation so multi-turn interactions work.

    ``stream_async`` is also supported so Chainlit can update
    its message in real-time from the Brain stage.
    ``event_loop_metrics`` delegates to the Brain agent so token-usage
    reporting in Chainlit continues to work.
    """

    def __init__(
        self,
        researcher: Agent,
        brain: Agent,
        *,
        info_agent: Agent | None = None,
        triage_agent: Agent | None = None,
        planner_agent: Agent | None = None,
        error_checker_agent: Agent | None = None,
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
        self._error_checker_agent: Agent = error_checker_agent or create_error_checker_agent()
        self._verbose = verbose
        self._skip_brain = skip_brain
        self._info_context: dict = info_context or {}
        self._session: AgentSession = AgentSession(session_id=session_id)
        # Brainbriefing JSON from the most recent Researcher run; used by the
        # Executor for Vision QA comparison in follow-up / feedback-loop rounds.
        self._last_brainbriefing_json: str | None = None
        # Compressed summary text from the previous turn, cached so it can be
        # injected into the Researcher on the NEXT turn regardless of triage intent.
        # This is the authoritative source for OUTPUT_PATHS / INPUT_PATHS_USER_MESSAGE
        # that bridges chained sessions even when triage says "new_request".
        self._last_prior_summary: str | None = None
        # Whether the current turn requested an explicit Vision QA pass.
        self._run_qa: bool = False
        # Bind the memory tools module-level session so memory_read / memory_write
        # always operate on the correct per-session namespace.
        _set_memory_session_id(session_id)
        # Initialise Vision Agent so analyze_image(mode='describe') works for
        # all agents (Researcher, Info, etc.) in this pipeline.
        try:
            _set_vision_agent(create_vision_agent())
        except Exception as _va_exc:
            print(f"[agentY] WARNING: could not initialise VisionAgent ({_va_exc}). "
                  "analyze_image will fall back to mode='full'.")
        # Per-turn usage tracking: list of (delta_usage_dict, agent_obj) for every
        # agent that contributed tokens this turn. Reset at the start of each turn.
        self._last_turn_usages: list = []
        # Brain-history compression is moved off the critical path: the executor
        # finishes, the final stream events flow to the UI, and only then this
        # background task summarises the conversation.  The next user turn
        # awaits it so the brain never sees uncompressed history.
        self._pending_compression: asyncio.Task | None = None

    async def _await_pending_compression(self) -> None:
        """Block until any deferred brain-history compression has finished."""
        task = self._pending_compression
        if task is None:
            return
        self._pending_compression = None
        try:
            await task
        except Exception as exc:  # noqa: BLE001
            if self._verbose:
                print(f"pipeline: deferred compression failed — {exc}")

    def _schedule_compression(self, extra_output_paths: list[str] | None = None) -> None:
        """Replace the previous deferred compression (if any) with a new one.

        Snapshots ``extra_output_paths`` because the live session list may be
        mutated by the next turn before the task runs.
        """
        snapshot = list(extra_output_paths) if extra_output_paths else None
        prev = self._pending_compression
        if prev is not None and not prev.done():
            # Previous task hasn't started yet — safe to drop; new content
            # supersedes it.  If it has run, _await_pending_compression already
            # awaited it.
            prev.cancel()
        self._pending_compression = asyncio.create_task(
            self._compress_brain_history(extra_output_paths=snapshot)
        )

    def _should_skip_brain(self) -> bool:
        if self._skip_brain:
            return True
        return str(os.environ.get("PIPELINE_SKIP_BRAIN", "false")).strip().lower() in (
            "1", "true", "yes", "on"
        )

    # Chainlit and main.py both do:  response = agent(user_input)
    def __call__(self, user_input, **kwargs: Any) -> str:
        return self.run(user_input, **kwargs)

    # Aggregate token usage from ALL agents that contributed to the last turn.
    # Callers that do ``pipeline.event_loop_metrics.accumulated_usage`` see
    # the combined picture (triage + researcher + brain + info, etc.) instead
    # of only the Brain.
    @property
    def event_loop_metrics(self):  # noqa: ANN201
        return _TurnMetrics(self._last_turn_usages)

    # ── Per-turn usage tracking helpers ─────────────────────────────── #

    def _usage_snapshot(self, agent) -> dict:
        """Return a copy of *agent*'s current accumulated usage, or {} on error."""
        try:
            return dict(agent.event_loop_metrics.accumulated_usage)
        except Exception:  # noqa: BLE001
            return {}

    def _record_agent_usage(self, agent, before: dict) -> None:
        """Compute the token delta for *agent* since *before* and store it.

        Only appends an entry when the delta contains at least one positive
        value (i.e. the agent actually issued LLM calls).
        """
        try:
            after = dict(agent.event_loop_metrics.accumulated_usage)
            delta = {
                "inputTokens": int(after.get("inputTokens", 0) or 0) - int(before.get("inputTokens", 0) or 0),
                "outputTokens": int(after.get("outputTokens", 0) or 0) - int(before.get("outputTokens", 0) or 0),
                "cacheReadInputTokens": (
                    int(after.get("cacheReadInputTokens", 0) or 0)
                    - int(before.get("cacheReadInputTokens", 0) or 0)
                ),
                "cacheWriteInputTokens": (
                    int(after.get("cacheWriteInputTokens", 0) or 0)
                    - int(before.get("cacheWriteInputTokens", 0) or 0)
                ),
            }
            if any(v > 0 for v in delta.values()):
                self._last_turn_usages.append((delta, agent))
        except Exception:  # noqa: BLE001
            pass

    def compute_turn_cost(self) -> tuple:
        """Return ``(total_cost_usd, total_tokens)`` for the current turn.

        Unlike ``compute_cost_from_usage(usage, pipeline)``, this method prices
        each agent's delta with *that agent's* model rates, so e.g. Researcher
        tokens billed at claude-haiku prices while Brain tokens at a different
        rate, and Ollama agents contribute 0 cost regardless of token count.
        """
        total_cost = 0.0
        total_tokens = 0
        for usage, agent in self._last_turn_usages:
            cost, tokens = compute_cost_from_usage(usage, agent)
            total_cost += cost
            total_tokens += tokens
        return total_cost, total_tokens

    def run(self, user_input, **_: Any) -> str:
        """Run the full pipeline for *user_input* and return the Brain's response.

        Triage runs first to classify intent, then routes to:
        - ``researcher`` (new_request / chain / low-confidence): full Researcher → Brain flow
        - ``brain`` (param_tweak / feedback): direct Brain follow-up
        - ``answer`` (info_query): return the triage answer directly
        - ``log_warning`` (low-confidence fallback): treat as new_request
        """
        # Block until any deferred brain-history compression from the prior
        # turn has finished, otherwise this turn would see uncompressed history.
        if self._pending_compression is not None:
            asyncio.run(self._await_pending_compression())
        self._last_turn_usages = []
        user_text = self._extract_text(user_input)
        user_text = self._annotate_attachments(user_input, user_text)
        _triage_snap = self._usage_snapshot(self._triage_agent)
        _triage_input = (
            user_input
            if (
                isinstance(user_input, list)
                and any("image" in b for b in user_input)
                and getattr(self._triage_agent, "_is_claude", False)
            )
            else user_text
        )
        triage_result = asyncio.run(_triage(_triage_input, self._session, self._info_context, self._triage_agent))
        self._record_agent_usage(self._triage_agent, _triage_snap)
        self._run_qa = triage_result.run_qa
        handler = _route(triage_result)

        if self._verbose:
            print(f"pipeline: Triage → intent={triage_result.intent.value},"
                  f" confidence={triage_result.confidence:.2f}, handler={handler}")

        # Context-dependent routing: researcher was previously blocked waiting for user input.
        # Re-run the researcher with the original request + user's clarification before
        # normal triage dispatch — regardless of how triage classified this message.
        if self._session.last_agent == "researcher" and self._session.last_researcher_request:
            if self._verbose:
                print("pipeline: Researcher was blocked — re-running with user clarification")
            _bls = self._session.last_researcher_blockers
            _blockers_ctx = (
                "\n\nYou previously identified these blockers:\n" + "\n".join(f"- {b}" for b in _bls)
                if _bls else ""
            )
            _enriched = (
                f"{self._session.last_researcher_request}"
                f"{_blockers_ctx}\n\n"
                f"The user provided this clarification: {user_text}"
            )
            self._session.last_researcher_request = None
            self._session.last_researcher_blockers = []
            _r_json, _r_err, _ = self._run_researcher(_enriched)
            if _r_err:
                self._record_chat_summary(user_text, triage_result, status="error")
                return _r_err
            _question = self._researcher_blocked_question(_r_json)
            if _question:
                # Still blocked — store enriched context so the next reply continues the chain.
                self._session.last_researcher_request = _enriched
                try:
                    self._session.last_researcher_blockers = json.loads(_r_json).get("blockers", [])
                except Exception:
                    pass
                self._session.last_agent = "researcher"
                self._record_chat_summary(user_text, triage_result, status="blocked")
                return _question
            # Researcher now ready — hand off to Brain.
            self._last_brainbriefing_json = _r_json
            self._brain.messages.clear()
            self._ensure_clean_history()
            _snap_b = self._usage_snapshot(self._brain)
            _b_resp = str(self._brain(self._build_brain_prompt(_r_json)))
            self._record_agent_usage(self._brain, _snap_b)
            _wf = _get_workflow_signal()
            _wf = self._expand_variations(_wf, _r_json)
            _ex_paths: list[str] = []
            if _wf:
                _ex_lines, _ex_paths, _ = asyncio.run(
                    self._drain_executor_batch(_wf, _r_json, user_message=user_text)
                )
                if _ex_paths:
                    self._session.current_output_paths[:] = _ex_paths
                if _ex_lines:
                    _b_resp += "\n\n" + "\n".join(_ex_lines)
            self._record_chat_summary(user_text, triage_result, status="completed", raw_json=_r_json)
            asyncio.run(self._compress_brain_history(extra_output_paths=_ex_paths))
            self._session.last_agent = "brain"
            return _b_resp

        if handler == "answer":
            if self._verbose:
                print("pipeline: info_query → Info agent")
            _info_snap = self._usage_snapshot(self._info_agent)
            response = str(self._info_agent(user_text))
            self._record_agent_usage(self._info_agent, _info_snap)
            self._session.last_agent = "info"
            self._session.last_info_response = response
            self._record_chat_summary(user_text, triage_result, status="completed")
            return response

        if handler == "needs_image":
            if self._verbose:
                print("pipeline: needs_image → handoff to user (missing image)")
            self._record_chat_summary(user_text, triage_result, status="needs_image")
            return triage_result.response or (
                "It looks like your request requires an input image, but I don't see one attached. "
                "Please share the image you'd like me to work with and I'll get started!"
            )

        if handler == "brain":
            # Context-dependent feedback routing: if the previous turn was handled by
            # the Info agent (e.g. it created/refined a prompt), route feedback back to
            # Info instead of the Brain, which has no knowledge of the prior prompt.
            if triage_result.intent == MessageIntent.feedback and self._session.last_agent == "info":
                if self._verbose:
                    print("pipeline: feedback on Info-agent output → routing back to Info agent")
                _info_snap = self._usage_snapshot(self._info_agent)
                response = str(self._info_agent(user_text))
                self._record_agent_usage(self._info_agent, _info_snap)
                self._session.last_agent = "info"
                self._record_chat_summary(user_text, triage_result, status="completed")
                return response
            # Follow-up: skip Researcher, send directly to Brain
            self._ensure_clean_history()
            brain_prompt = self._build_followup_prompt(user_text, triage_result)
            _brain_snap = self._usage_snapshot(self._brain)
            brain_response = str(self._brain(brain_prompt))
            self._record_agent_usage(self._brain, _brain_snap)
            self._session.follow_up_count += 1
            # Executor handoff: Brain signals a (re-)assembled workflow is ready
            workflow_paths = _get_workflow_signal()
            workflow_paths = self._expand_variations(workflow_paths, self._last_brainbriefing_json or "")
            executor_paths: list[str] = []
            if workflow_paths:
                count = len(workflow_paths)
                if self._verbose:
                    tag = f"{count} workflows (batch)" if count > 1 else workflow_paths[0]
                    print(f"pipeline: Brain (follow-up) signaled {tag} ready.")
                executor_lines, executor_paths, _ = asyncio.run(
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
            self._session.last_agent = "brain"
            if self._verbose:
                print("pipeline: Brain (follow-up) finished.")
            return brain_response

        if handler == "planner":
            return self._run_planned_request(user_text, triage_result)

        # handler == "researcher" or "log_warning" → full Researcher → Brain flow
        raw_json, error, researcher_output = self._run_researcher(user_input)
        if error:
            self._record_chat_summary(user_text, triage_result, status="error")
            return error

        # Check if the researcher needs user clarification before it can proceed.
        question = self._researcher_blocked_question(raw_json)
        if question:
            self._session.last_researcher_request = user_text
            try:
                self._session.last_researcher_blockers = json.loads(raw_json).get("blockers", [])
            except Exception:
                self._session.last_researcher_blockers = []
            self._session.last_agent = "researcher"
            self._record_chat_summary(user_text, triage_result, status="blocked")
            return question

        if self._should_skip_brain():
            if self._verbose:
                print("pipeline: Skipping Brain stage; returning Researcher output.")
            return researcher_output

        self._last_brainbriefing_json = raw_json
        brain_prompt = self._build_brain_prompt(raw_json)
        # New request → discard prior history entirely to save tokens.
        self._brain.messages.clear()
        self._ensure_clean_history()
        _brain_snap = self._usage_snapshot(self._brain)
        brain_response = str(self._brain(brain_prompt))
        self._record_agent_usage(self._brain, _brain_snap)
        # Executor handoff: Brain signals the assembled workflow(s) are ready
        workflow_paths_r = _get_workflow_signal()
        # Expand variation prompts from multiprompt.json if applicable
        workflow_paths_r = self._expand_variations(workflow_paths_r, raw_json)
        executor_paths_r: list[str] = []
        if workflow_paths_r:
            count = len(workflow_paths_r)
            if self._verbose:
                tag = f"{count} workflows (batch)" if count > 1 else workflow_paths_r[0]
                print(f"pipeline: Brain signaled {tag} ready.")
            executor_lines_r, executor_paths_r, _ = asyncio.run(
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
        self._session.last_agent = "brain"
        if self._verbose:
            print("pipeline: Brain finished.")
        return brain_response

    async def stream_async(self, user_input, *, qa_reply_queue: asyncio.Queue | None = None):  # noqa: ANN201
        """Async generator compatible with Chainlit's streaming loop.

        Runs the Researcher synchronously (it's a single-turn spec dump),
        then transparently streams the Brain's token output so Chainlit can
        update its message in real time.

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
        # Block until any deferred brain-history compression from the prior
        # turn has finished, otherwise this turn would see uncompressed history.
        await self._await_pending_compression()
        # Stage 0 – Triage (classify intent before any agent is called)
        self._last_turn_usages = []
        user_text = self._extract_text(user_input)
        user_text = self._annotate_attachments(user_input, user_text)
        _triage_snap = self._usage_snapshot(self._triage_agent)
        _triage_input = (
            user_input
            if (
                isinstance(user_input, list)
                and any("image" in b for b in user_input)
                and getattr(self._triage_agent, "_is_claude", False)
            )
            else user_text
        )
        triage_result = await _triage(_triage_input, self._session, self._info_context, self._triage_agent)
        self._record_agent_usage(self._triage_agent, _triage_snap)
        self._run_qa = triage_result.run_qa
        handler = _route(triage_result)

        if self._verbose:
            _msg = (f"pipeline: Triage → intent={triage_result.intent.value},"
                    f" confidence={triage_result.confidence:.2f}, handler={handler}")
            print(_msg)
            yield {"data": f"\n_{_msg}_"}

        # Context-dependent routing: researcher was previously blocked waiting for user input.
        if self._session.last_agent == "researcher" and self._session.last_researcher_request:
            if self._verbose:
                print("pipeline: Researcher was blocked — re-running with user clarification")
                yield {"data": "\n_pipeline: Researcher was blocked — re-running with user clarification_"}
            _bls_s = self._session.last_researcher_blockers
            _blockers_ctx_s = (
                "\n\nYou previously identified these blockers:\n" + "\n".join(f"- {b}" for b in _bls_s)
                if _bls_s else ""
            )
            _enriched_s = (
                f"{self._session.last_researcher_request}"
                f"{_blockers_ctx_s}\n\n"
                f"The user provided this clarification: {user_text}"
            )
            self._session.last_researcher_request = None
            self._session.last_researcher_blockers = []
            _r_json_s: str | None = None
            _r_err_s: str | None = None
            yield {"_researcher_start": True}
            async for _ev in self._arun_researcher(_enriched_s):
                if isinstance(_ev, dict) and "_researcher_done" in _ev:
                    _r_json_s = _ev["raw_json"]
                    _r_err_s = _ev["error"]
                    yield {"_researcher_done": True}
                else:
                    yield _ev
            if _r_err_s:
                self._record_chat_summary(user_text, triage_result, status="error")
                yield {"data": _r_err_s}
                return
            _question_r = self._researcher_blocked_question(_r_json_s)
            if _question_r:
                self._session.last_researcher_request = _enriched_s
                try:
                    self._session.last_researcher_blockers = json.loads(_r_json_s).get("blockers", [])
                except Exception:
                    pass
                self._session.last_agent = "researcher"
                self._record_chat_summary(user_text, triage_result, status="blocked")
                yield {"data": _question_r}
                return
            # Researcher now ready — stream Brain stage.
            if self._verbose:
                print("pipeline: Researcher (retry) resolved — handing off to Brain …")
                yield {"data": "\n_pipeline: Researcher (retry) resolved — handing off to Brain …_"}
            async for event in self._astream_brain_stage(
                _r_json_s, user_text, triage_result, qa_reply_queue=qa_reply_queue
            ):
                yield event
            return

        if handler == "answer":
            if self._verbose:
                print("pipeline: info_query → Info agent (streamed)")
                yield {"data": "\n_pipeline: info_query → Info agent (streamed)_"}
            _info_snap = self._usage_snapshot(self._info_agent)
            _info_chunks: list[str] = []
            async for event in self._info_agent.stream_async(user_text):
                if isinstance(event, dict):
                    _chunk = event.get("data", "")
                    if _chunk:
                        _info_chunks.append(_chunk)
                yield event
            self._record_agent_usage(self._info_agent, _info_snap)
            self._session.last_agent = "info"
            self._session.last_info_response = "".join(_info_chunks) or None
            self._record_chat_summary(user_text, triage_result, status="completed")
            return

        if handler == "needs_image":
            if self._verbose:
                print("pipeline: needs_image → handoff to user (missing image)")
                yield {"data": "\n_pipeline: needs_image → handoff to user (missing image)_"}
            self._record_chat_summary(user_text, triage_result, status="needs_image")
            message = triage_result.response or (
                "It looks like your request requires an input image, but I don't see one attached. "
                "Please share the image you'd like me to work with and I'll get started!"
            )
            yield {"data": message}
            return

        if handler == "brain":
            # Context-dependent feedback routing: if the previous turn was handled by
            # the Info agent (e.g. it created/refined a prompt), route feedback back to
            # Info instead of the Brain, which has no knowledge of the prior prompt.
            if triage_result.intent == MessageIntent.feedback and self._session.last_agent == "info":
                if self._verbose:
                    print("pipeline: feedback on Info-agent output → routing back to Info agent")
                    yield {"data": "\n_pipeline: feedback on Info-agent output → routing back to Info agent_"}
                _info_snap = self._usage_snapshot(self._info_agent)
                async for event in self._info_agent.stream_async(user_text):
                    yield event
                self._record_agent_usage(self._info_agent, _info_snap)
                self._session.last_agent = "info"
                self._record_chat_summary(user_text, triage_result, status="completed")
                return
            # Follow-up: skip Researcher, send directly to Brain (streamed)
            self._ensure_clean_history()
            brain_prompt = self._build_followup_prompt(user_text, triage_result)
            self._session.follow_up_count += 1
            _brain_snap_fu = self._usage_snapshot(self._brain)
            yield {"_brain_start": True}
            async for event in self._brain.stream_async(brain_prompt):
                yield event
            yield {"_brain_done": True}
            self._record_agent_usage(self._brain, _brain_snap_fu)
            # Executor handoff: stream execution events back to Chainlit
            workflow_paths_fu = _get_workflow_signal()
            workflow_paths_fu = self._expand_variations(workflow_paths_fu, self._last_brainbriefing_json or "")
            # Pass the session list directly so chainlit's mid-stream flush sees
            # each output the moment the executor resolves it, instead of all at
            # the end.
            self._session.current_output_paths.clear()
            executor_paths_fu = self._session.current_output_paths
            if workflow_paths_fu:
                count = len(workflow_paths_fu)
                if self._verbose:
                    tag = f"{count} workflows (batch)" if count > 1 else workflow_paths_fu[0]
                    print(f"pipeline: Brain (follow-up) signaled {tag} ready.")
                    yield {"data": f"\n_pipeline: Brain (follow-up) signaled {tag} ready._"}
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
            self._record_chat_summary(user_text, triage_result, status="completed")
            self._schedule_compression(extra_output_paths=executor_paths_fu)
            self._session.last_agent = "brain"
            return

        if handler == "planner":
            async for event in self._stream_planned_request(
                user_text, triage_result, qa_reply_queue=qa_reply_queue
            ):
                yield event
            return

        # handler == "researcher" or "log_warning" → full Researcher → Brain flow
        # Stage 1 – Researcher (streamed)
        if self._verbose:
            print("pipeline: Stage 1 – Researcher resolving spec …")
            yield {"data": "\n_pipeline: Stage 1 – Researcher resolving spec …_"}
        raw_json: str | None = None
        error: str | None = None
        researcher_output: str = ""
        yield {"_researcher_start": True}
        async for _r_ev in self._arun_researcher(user_input):
            if isinstance(_r_ev, dict) and "_researcher_done" in _r_ev:
                raw_json = _r_ev["raw_json"]
                error = _r_ev["error"]
                researcher_output = _r_ev["researcher_output"]
                yield {"_researcher_done": True}
            else:
                yield _r_ev
        if error:
            self._record_chat_summary(user_text, triage_result, status="error")
            yield {"data": error}
            return

        # Check if the researcher needs user clarification before it can proceed.
        _question_first = self._researcher_blocked_question(raw_json)
        if _question_first:
            self._session.last_researcher_request = user_text
            try:
                self._session.last_researcher_blockers = json.loads(raw_json).get("blockers", [])
            except Exception:
                self._session.last_researcher_blockers = []
            self._session.last_agent = "researcher"
            self._record_chat_summary(user_text, triage_result, status="blocked")
            yield {"data": _question_first}
            return

        self._last_brainbriefing_json = raw_json

        if self._should_skip_brain():
            if self._verbose:
                print("pipeline: Skipping Brain stage; returning Researcher output.")
                yield {"data": "\n_pipeline: Skipping Brain stage; returning Researcher output._"}
            yield {"data": researcher_output}
            return

        # Stage 2 – Brain (streamed, with optional ComfyUI interrupt handling)
        if self._verbose:
            print("pipeline: Stage 2 – Brain streaming …")
            yield {"data": "\n_pipeline: Stage 2 – Brain streaming …_"}
        async for event in self._astream_brain_stage(
            raw_json, user_text, triage_result, qa_reply_queue=qa_reply_queue
        ):
            yield event

    # ── Internal helpers ─────────────────────────────────────────────── #

    # ── Planner helpers ──────────────────────────────────────────────── #

    def _run_planner(self, user_text: str) -> tuple[list[dict[str, str]], str]:
        """Call the Planner agent to decompose *user_text* into ordered steps.

        Returns a tuple of:
          - list of ``{"request": str, "description": str}`` dicts (empty on failure)
          - the raw agent response string (for display in the UI thinking block)
        """
        raw: str
        _planner_snap = self._usage_snapshot(self._planner_agent)
        raw = str(self._planner_agent(user_text))
        self._record_agent_usage(self._planner_agent, _planner_snap)
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
                return [], raw
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
                return [], raw
            return validated, raw
        except (json.JSONDecodeError, KeyError, TypeError) as exc:
            if self._verbose:
                print(f"[planner] WARNING: plan parse failed ({exc}). Falling back to researcher.")
            return [], raw

    def _inject_context_into_step(
        self,
        step_request: str,
        step_index: int,
        prev_brainbriefing: dict | None = None,
    ) -> str:
        """Prepend previous-step context into *step_request* when available.

        Steps 2+ receive:
          • The actual output file paths from the previous step, and
          • A compact brainbriefing snippet (template, task type, resolution,
            abbreviated positive prompt) so the Researcher doesn't start
            completely blind when making decisions that must be compatible
            with the prior step's output.

        This avoids passing full message histories (agents-as-tools style)
        which would accumulate tokens across every step.
        """
        if step_index == 0:
            return step_request
        hints: list[str] = []
        if self._session.current_output_paths:
            paths = ", ".join(self._session.current_output_paths)
            hints.append(f"Previous step output file(s): {paths}")
        if prev_brainbriefing:
            ctx: dict = {}
            tmpl = (prev_brainbriefing.get("template") or {}).get("name")
            if tmpl:
                ctx["template"] = tmpl
            task_type = (prev_brainbriefing.get("task") or {}).get("type")
            if task_type:
                ctx["task_type"] = task_type
            w = prev_brainbriefing.get("resolution_width")
            h = prev_brainbriefing.get("resolution_height")
            if w and h:
                ctx["resolution"] = f"{w}x{h}"
            pos = (prev_brainbriefing.get("prompt") or {}).get("positive", "")
            if pos:
                ctx["prompt_positive"] = pos[:200]
            if ctx:
                hints.append(f"Previous step brainbriefing context: {json.dumps(ctx)}")
        if not hints:
            return step_request
        return step_request + "\n" + "\n".join(f"[{h}]" for h in hints)

    def _clear_error_checker_history(self) -> None:
        """Reset the error-checker agent's single-turn conversation history."""
        try:
            self._error_checker_agent.conversation_manager.messages.clear()
        except AttributeError:
            try:
                self._error_checker_agent.conversation_manager._messages.clear()  # noqa: SLF001
            except AttributeError:
                pass

    def _build_fix_prompt(self, fix_plan: str, attempt: int) -> str:
        """Build a Brain prompt asking it to apply a fix from error-checker output."""
        return (
            f"[Error-checker] The ComfyUI workflow just failed (fix attempt {attempt}/3).\n\n"
            f"Fix plan:\n{fix_plan}\n\n"
            "Apply the fix to the current workflow. Re-validate every change, then call "
            "`signal_workflow_ready(workflow_path)` with the corrected workflow once it is ready."
        )

    def _run_error_check(self, task_description: str) -> dict:
        """Invoke the error-checker agent and return a parsed verdict dict.

        The agent reads ComfyUI logs and returns JSON with keys:
        ``status`` (ok | error_fixable | error_unfixable), ``errors``,
        ``fix_plan``, ``user_message``.

        On any failure the method returns ``{"status": "ok", ...}``
        (fail-open) so a transient error doesn't abort the plan.
        """
        _snap = self._usage_snapshot(self._error_checker_agent)
        raw = ""
        try:
            raw = str(self._error_checker_agent(task_description))
            self._record_agent_usage(self._error_checker_agent, _snap)
        except Exception as exc:
            if self._verbose:
                print(f"[error_checker] WARNING: agent call failed — {exc}")
            return {"status": "ok", "errors": [], "fix_plan": "", "user_message": ""}
        finally:
            self._clear_error_checker_history()
        json_str = _extract_json(raw) or raw
        try:
            verdict = json.loads(json_str)
            if not isinstance(verdict, dict):
                return {"status": "ok", "errors": [], "fix_plan": "", "user_message": ""}
            verdict.setdefault("status", "ok")
            verdict.setdefault("errors", [])
            verdict.setdefault("fix_plan", "")
            verdict.setdefault("user_message", "")
            return verdict
        except (json.JSONDecodeError, TypeError):
            if self._verbose:
                print("[error_checker] WARNING: could not parse verdict JSON — treating as ok.")
            return {"status": "ok", "errors": [], "fix_plan": "", "user_message": ""}

    def _run_planned_request(self, user_text: str, triage_result: TriageResult) -> str:
        """Execute a multi-step plan synchronously; return a combined summary string."""
        if self._verbose:
            print("pipeline: Planner — decomposing multi-step request …")
        steps, _planner_raw = self._run_planner(user_text)

        # Fallback: treat as a plain new_request when planning fails.
        if not steps:
            if self._verbose:
                print("pipeline: Planner fallback → researcher path")
            raw_json, error, researcher_output = self._run_researcher(user_text)
            if error:
                self._record_chat_summary(user_text, triage_result, status="error")
                return error
            self._last_brainbriefing_json = raw_json
            if self._should_skip_brain():
                return researcher_output
            self._ensure_clean_history()
            _brain_snap_pfb = self._usage_snapshot(self._brain)
            brain_response = str(self._brain(self._build_brain_prompt(raw_json)))
            self._record_agent_usage(self._brain, _brain_snap_pfb)
            wf = _get_workflow_signal()
            wf = self._expand_variations(wf, raw_json)
            ep: list[str] = []
            if wf:
                _, ep, _ = asyncio.run(
                    self._drain_executor_batch(wf, raw_json, user_message=user_text)
                )
                if ep:
                    self._session.current_output_paths[:] = ep
            self._record_chat_summary(user_text, triage_result, status="completed", raw_json=raw_json)
            asyncio.run(self._compress_brain_history(extra_output_paths=ep))
            return brain_response

        total = len(steps)
        if self._verbose:
            print(f"pipeline: Plan has {total} step(s):")
            for i, s in enumerate(steps, 1):
                print(f"  {i}. {s['description']}")

        step_results: list[str] = []
        prev_brainbriefing: dict | None = None  # compact context forwarded step-to-step

        for idx, step in enumerate(steps):
            description = step["description"]
            step_req = self._inject_context_into_step(step["request"], idx, prev_brainbriefing)

            if self._verbose:
                print(f"\npipeline: ── Plan step {idx + 1}/{total}: {description} ──")

            raw_json, error, researcher_output = self._run_researcher(step_req)
            if error:
                msg = f"Step {idx + 1} ({description}) failed: {error}"
                if self._verbose:
                    print(f"pipeline: {msg}")
                step_results.append(msg)
                # Abort remaining steps when the researcher fails.
                break

            # Check for a soft blocker (researcher status=blocked) — stop the plan.
            blocked_question = self._researcher_blocked_question(raw_json)
            if blocked_question:
                msg = (
                    f"🚫 Plan stopped at step {idx + 1}/{total} — **{description}** — "
                    f"the researcher needs more information before it can continue:\n\n"
                    f"{blocked_question}"
                )
                if self._verbose:
                    print(f"pipeline: Researcher blocked at step {idx + 1}: {blocked_question}")
                step_results.append(msg)
                break

            self._last_brainbriefing_json = raw_json
            # Stash brainbriefing for the next step's context injection.
            try:
                prev_brainbriefing = json.loads(raw_json) if raw_json else None
            except (json.JSONDecodeError, TypeError):
                prev_brainbriefing = None

            if self._should_skip_brain():
                step_results.append(f"Step {idx + 1}: {researcher_output}")
                continue

            qa_step_failed = False
            while True:  # ── QA retry loop ──────────────────────────────── #
                self._ensure_clean_history()
                _brain_snap_ps = self._usage_snapshot(self._brain)
                brain_response = str(self._brain(self._build_brain_prompt(raw_json)))
                self._record_agent_usage(self._brain, _brain_snap_ps)
                wf_paths = _get_workflow_signal()
                wf_paths = self._expand_variations(wf_paths, raw_json)
                exec_paths: list[str] = []
                step_error_abort = False
                verdict: dict = {"status": "ok", "errors": [], "fix_plan": "", "user_message": ""}
                _qa_fail_event: dict | None = None
                if wf_paths:
                    count = len(wf_paths)
                    if self._verbose:
                        tag = f"{count} workflows (batch)" if count > 1 else wf_paths[0]
                        print(f"pipeline: Step {idx + 1} Brain signaled {tag} ready.")
                    exec_lines, exec_paths, _qa_fail_event = asyncio.run(
                        self._drain_executor_batch(
                            wf_paths, raw_json, user_message=step_req
                        )
                    )
                    if exec_paths:
                        self._session.current_output_paths[:] = exec_paths
                    if exec_lines:
                        brain_response = brain_response + "\n\n" + "\n".join(exec_lines)

                    if not _qa_fail_event:
                        # ── Error check + fix-retry loop ─────────────────── #
                        _MAX_FIX_ATTEMPTS = 3
                        for _fix_attempt in range(1, _MAX_FIX_ATTEMPTS + 1):
                            verdict = self._run_error_check(description)
                            if verdict["status"] == "ok":
                                break
                            if self._verbose:
                                print(
                                    f"[error_checker] Step {idx + 1} attempt {_fix_attempt}: "
                                    f"{verdict['status']} — {verdict.get('errors', [])[:1]}"
                                )
                            if verdict["status"] == "error_unfixable" or _fix_attempt == _MAX_FIX_ATTEMPTS:
                                step_error_abort = True
                                if self._verbose:
                                    print(
                                        f"pipeline: Step {idx + 1} aborted after "
                                        f"{_fix_attempt} error-check attempt(s)."
                                    )
                                break
                            # Fixable error — ask Brain to fix and rerun.
                            _brain_snap_fix = self._usage_snapshot(self._brain)
                            fix_response = str(self._brain(self._build_fix_prompt(verdict["fix_plan"], _fix_attempt)))
                            self._record_agent_usage(self._brain, _brain_snap_fix)
                            wf_paths_fix = _get_workflow_signal()
                            wf_paths_fix = self._expand_variations(wf_paths_fix, raw_json)
                            exec_paths = []
                            if wf_paths_fix:
                                exec_lines_fix, exec_paths, _qa_fail_event = asyncio.run(
                                    self._drain_executor_batch(wf_paths_fix, raw_json, user_message=step_req)
                                )
                                if exec_paths:
                                    self._session.current_output_paths[:] = exec_paths
                                if exec_lines_fix:
                                    brain_response = fix_response + "\n\n" + "\n".join(exec_lines_fix)
                                if _qa_fail_event:
                                    break
                            else:
                                # Brain could not produce a fixed workflow.
                                step_error_abort = True
                                break
                        # ── end error-check retry loop ───────────────────── #

                if _qa_fail_event:
                    # Vision QA failed — show result and ask whether to retry.
                    _image_paths = _qa_fail_event.get("image_paths", [])
                    _fail_details = _qa_fail_event.get("fail_details", [])
                    print(f"\n⚠️  QA check failed for step {idx + 1}/{total} — {description}")
                    for p in _image_paths:
                        print(f"   📎 Output: {p}")
                    for d in _fail_details:
                        print(f"   • {Path(d['path']).name}: {d['verdict']}")
                    try:
                        _answer = input("\n🔁 Retry this step? [y/n]: ").strip()
                    except (EOFError, KeyboardInterrupt):
                        _answer = "n"
                    if _is_affirmative(_answer):
                        print("  ↩️  Retrying…\n")
                        continue  # re-run Brain + executor
                    qa_step_failed = True

                break  # ── end QA retry loop ──────────────────────────────── #

            if qa_step_failed:
                step_results.append(f"**Step {idx + 1} — {description}**: ⚠️ QA failed, step skipped.")
                break  # abort remaining steps

            if step_error_abort:
                err_msg = (
                    verdict.get("user_message")
                    or f"Step {idx + 1} failed after {_MAX_FIX_ATTEMPTS} retry attempt(s)."
                )
                if self._verbose:
                    print(f"pipeline: Step {idx + 1} permanently failed: {err_msg}")
                step_results.append(f"**Step {idx + 1} — {description}**: ❌ {err_msg}")
                break  # abort remaining steps

            self._record_chat_summary(step_req, triage_result, status="completed", raw_json=raw_json)
            asyncio.run(self._compress_brain_history(extra_output_paths=exec_paths))

            step_results.append(f"**Step {idx + 1} — {description}**\n{brain_response}")

            if self._verbose:
                print(f"pipeline: Step {idx + 1}/{total} finished.")

        combined = "\n\n---\n\n".join(step_results)
        if self._verbose:
            print(f"pipeline: Planned execution complete ({total} step(s)).")
        return combined

    async def _stream_planned_request(
        self,
        user_text: str,
        triage_result: TriageResult,
        *,
        qa_reply_queue: asyncio.Queue | None = None,
    ):
        """Stream a multi-step plan; yields Strands-compatible event dicts."""
        if self._verbose:
            print("pipeline: Planner — decomposing multi-step request …")
            yield {"data": "\n_pipeline: Planner — decomposing multi-step request …_"}
        yield {"_planner_start": True}
        steps, _planner_raw = self._run_planner(user_text)
        yield {"_planner_done": True, "raw": _planner_raw}

        # Fallback: treat as a plain researcher path when planning fails.
        if not steps:
            if self._verbose:
                print("pipeline: Planner fallback → researcher path")
                yield {"data": "\n_pipeline: Planner fallback → researcher path_"}
            yield {"_researcher_start": True}
            raw_json = error = researcher_output = None
            async for _r_ev in self._arun_researcher(user_text):
                if isinstance(_r_ev, dict) and "_researcher_done" in _r_ev:
                    raw_json = _r_ev["raw_json"]
                    error = _r_ev["error"]
                    researcher_output = _r_ev["researcher_output"]
                    yield {"_researcher_done": True}
                else:
                    yield _r_ev
            if error:
                self._record_chat_summary(user_text, triage_result, status="error")
                yield {"data": error}
                return
            self._last_brainbriefing_json = raw_json
            if self._should_skip_brain():
                yield {"data": researcher_output}
                return
            self._ensure_clean_history()
            _brain_snap_pfb = self._usage_snapshot(self._brain)
            async for event in self._brain.stream_async(self._build_brain_prompt(raw_json)):
                yield event
            self._record_agent_usage(self._brain, _brain_snap_pfb)
            wf = _get_workflow_signal()
            wf = self._expand_variations(wf, raw_json)
            self._session.current_output_paths.clear()
            ep = self._session.current_output_paths
            if wf:
                yield {"data": "\n\n_⚙️ Handing off to executor…_"}
                async for line in _execute_workflows_batch(
                    wf, raw_json,
                    user_message=user_text,
                    verbose=self._verbose,
                    collected_paths=ep,
                ):
                    yield {"data": f"\n{line}"}
            self._record_chat_summary(user_text, triage_result, status="completed", raw_json=raw_json)
            self._schedule_compression(extra_output_paths=ep)
            return

        total = len(steps)
        yield {"_plan_ready": True, "steps": [{"description": s["description"]} for s in steps]}
        yield {"data": f"_🗂️ Plan ready — {total} step(s) to execute._\n"}
        if self._verbose:
            print(f"pipeline: Plan has {total} step(s):")
            for i, s in enumerate(steps, 1):
                print(f"  {i}. {s['description']}")
            _steps_list = "\n".join(f"  {i}. {s['description']}" for i, s in enumerate(steps, 1))
            yield {"data": f"\n_pipeline: Plan has {total} step(s):_\n{_steps_list}\n"}

        prev_brainbriefing: dict | None = None  # compact context forwarded step-to-step

        for idx, step in enumerate(steps):
            description = step["description"]
            step_req = self._inject_context_into_step(step["request"], idx, prev_brainbriefing)

            yield {"_step_start": True, "idx": idx, "total": total, "description": description}
            yield {"data": f"\n\n**Step {idx + 1}/{total} — {description}**\n"}
            if self._verbose:
                print(f"\npipeline: ── Plan step {idx + 1}/{total}: {description} ──")
                yield {"data": f"\n_pipeline: ── Plan step {idx + 1}/{total}: {description} ──_"}

            yield {"_researcher_start": True}
            raw_json = error = researcher_output = None
            async for _r_ev in self._arun_researcher(step_req):
                if isinstance(_r_ev, dict) and "_researcher_done" in _r_ev:
                    raw_json = _r_ev["raw_json"]
                    error = _r_ev["error"]
                    researcher_output = _r_ev["researcher_output"]
                    yield {"_researcher_done": True}
                else:
                    yield _r_ev
            if error:
                yield {"data": f"\n❌ Step {idx + 1} failed: {error}"}
                if self._verbose:
                    print(f"pipeline: Step {idx + 1} researcher error: {error}")
                break

            # Check for a soft blocker (researcher status=blocked) — stop the plan.
            blocked_question = self._researcher_blocked_question(raw_json)
            if blocked_question:
                block_msg = (
                    f"\n\n🚫 **Plan stopped at step {idx + 1}/{total} — {description}**\n\n"
                    f"The researcher needs more information before it can continue:\n\n"
                    f"{blocked_question}"
                )
                yield {"data": block_msg}
                if self._verbose:
                    print(f"pipeline: Researcher blocked at step {idx + 1}: {blocked_question}")
                break

            self._last_brainbriefing_json = raw_json
            # Stash brainbriefing for the next step's context injection.
            try:
                prev_brainbriefing = json.loads(raw_json) if raw_json else None
            except (json.JSONDecodeError, TypeError):
                prev_brainbriefing = None

            if self._should_skip_brain():
                yield {"data": researcher_output}
                continue

            self._ensure_clean_history()
            qa_step_failed = False
            _step_brain_prompt_override: str | None = None
            while True:  # ── QA retry loop ──────────────────────────────── #
                _brain_prompt_for_step = _step_brain_prompt_override or self._build_brain_prompt(raw_json)
                _step_brain_prompt_override = None  # consume once
                _brain_snap_ps = self._usage_snapshot(self._brain)
                self._ensure_clean_history()
                async for event in self._brain.stream_async(_brain_prompt_for_step):
                    yield event
                self._record_agent_usage(self._brain, _brain_snap_ps)

                wf_paths = _get_workflow_signal()
                wf_paths = self._expand_variations(wf_paths, raw_json)
                self._session.current_output_paths.clear()
                exec_paths = self._session.current_output_paths
                step_error_abort = False
                verdict: dict = {"status": "ok", "errors": [], "fix_plan": "", "user_message": ""}
                _qa_fail_event: dict | None = None
                if wf_paths:
                    count = len(wf_paths)
                    hdr = f"batch of {count} workflows" if count > 1 else "workflow"
                    yield {"data": f"\n\n_⚙️ Handing off to executor ({hdr})…_"}
                    async for line in _execute_workflows_batch(
                        wf_paths, raw_json,
                        user_message=step_req,
                        verbose=self._verbose,
                        collected_paths=exec_paths,
                        run_qa=self._run_qa,
                    ):
                        if isinstance(line, dict) and line.get("qa_fail"):
                            _qa_fail_event = line
                            break
                        yield {"data": f"\n{line}"}

                    # ── Error check + fix-retry loop ─────────────────────── #
                    if not _qa_fail_event:
                        _MAX_FIX_ATTEMPTS = 3
                        for _fix_attempt in range(1, _MAX_FIX_ATTEMPTS + 1):
                            verdict = self._run_error_check(description)
                            if verdict["status"] == "ok":
                                break
                            if self._verbose:
                                print(
                                    f"[error_checker] Step {idx + 1} attempt {_fix_attempt}: "
                                    f"{verdict['status']} — {verdict.get('errors', [])[:1]}"
                                )
                                yield {"data": (
                                    f"\n_[error_checker] Step {idx + 1} attempt {_fix_attempt}: "
                                    f"{verdict['status']}_"
                                )}
                            if verdict["status"] == "error_unfixable" or _fix_attempt == _MAX_FIX_ATTEMPTS:
                                step_error_abort = True
                                if self._verbose:
                                    print(
                                        f"pipeline: Step {idx + 1} aborted after "
                                        f"{_fix_attempt} error-check attempt(s)."
                                    )
                                    yield {"data": (
                                        f"\n_pipeline: Step {idx + 1} aborted — "
                                        f"{verdict.get('user_message', 'workflow error')}_"
                                    )}
                                break
                            # Fixable error — ask Brain to fix (streamed) and rerun.
                            yield {"data": f"\n_🔧 Fixing step {idx + 1} (attempt {_fix_attempt}/{_MAX_FIX_ATTEMPTS})…_"}
                            _brain_snap_fix = self._usage_snapshot(self._brain)
                            async for event in self._brain.stream_async(
                                self._build_fix_prompt(verdict["fix_plan"], _fix_attempt)
                            ):
                                yield event
                            self._record_agent_usage(self._brain, _brain_snap_fix)
                            wf_paths_fix = _get_workflow_signal()
                            wf_paths_fix = self._expand_variations(wf_paths_fix, raw_json)
                            # Reset for the fix run so the chainlit "already-sent"
                            # tracker treats fixed outputs as fresh.
                            self._session.current_output_paths.clear()
                            exec_paths = self._session.current_output_paths
                            if wf_paths_fix:
                                hdr_fix = f"batch of {len(wf_paths_fix)} workflows" if len(wf_paths_fix) > 1 else "workflow"
                                yield {"data": f"\n\n_⚙️ Re-submitting to executor ({hdr_fix})…_"}
                                async for line in _execute_workflows_batch(
                                    wf_paths_fix, raw_json,
                                    user_message=step_req,
                                    verbose=self._verbose,
                                    collected_paths=exec_paths,
                                    run_qa=self._run_qa,
                                ):
                                    if isinstance(line, dict) and line.get("qa_fail"):
                                        _qa_fail_event = line
                                        break
                                    yield {"data": f"\n{line}"}
                            else:
                                # Brain could not produce a fixed workflow.
                                step_error_abort = True
                                break
                            if _qa_fail_event:
                                break
                    # ── end error-check retry loop ───────────────────────── #

                if _qa_fail_event:
                    if qa_reply_queue is not None:
                        yield {"qa_fail_ask": True, **_qa_fail_event}
                        _answer = await qa_reply_queue.get()
                        if _is_affirmative(_answer):
                            _qa_step_feedback_prompt = self._build_qa_feedback_prompt(
                                self._build_brain_prompt(raw_json), step_req, _qa_fail_event
                            )
                            yield {"data": (
                                "\n\n---\n"
                                "🔍 **QA feedback sent to Brain:**\n\n"
                                + _qa_step_feedback_prompt
                                + "\n\n---\n"
                            )}
                            yield {"data": "\n_🔄 Retrying step with QA feedback…_"}
                            _step_brain_prompt_override = _qa_step_feedback_prompt
                            continue  # re-run Brain + executor with feedback
                    # No queue or user declined — mark failed and stop plan.
                    qa_step_failed = True

                break  # ── end QA retry loop ──────────────────────────────── #

            if qa_step_failed:
                yield {"_step_done": True, "idx": idx, "failed": True}
                self._record_chat_summary(step_req, triage_result, status="qa_failed", raw_json=raw_json)
                if self._verbose:
                    print(f"pipeline: Step {idx + 1}/{total} QA failed — aborting plan.")
                    yield {"data": f"\n_pipeline: Step {idx + 1}/{total} QA failed — aborting plan._"}
                break  # stop processing further steps

            if step_error_abort:
                err_msg = (
                    verdict.get("user_message")
                    or f"Step {idx + 1} failed after {_MAX_FIX_ATTEMPTS} retry attempt(s)."
                )
                yield {"_step_done": True, "idx": idx, "failed": True}
                self._record_chat_summary(step_req, triage_result, status="error", raw_json=raw_json)
                yield {"data": f"\n\n❌ **Step {idx + 1} — {description}**: {err_msg}"}
                if self._verbose:
                    print(f"pipeline: Step {idx + 1} permanently failed — aborting plan.")
                break  # abort remaining steps

            self._record_chat_summary(step_req, triage_result, status="completed", raw_json=raw_json)
            await self._compress_brain_history(extra_output_paths=exec_paths)

            yield {"_step_done": True, "idx": idx}
            if self._verbose:
                print(f"pipeline: Step {idx + 1}/{total} finished.")
                yield {"data": f"\n_pipeline: Step {idx + 1}/{total} finished._"}

        if self._verbose:
            print(f"pipeline: Planned execution complete ({total} step(s)).")
            yield {"data": f"\n_pipeline: Planned execution complete ({total} step(s))._"}
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
            # Include last brainbriefing so Brain knows which template/workflow to modify.
            context_parts: list[str] = []
            if self._last_brainbriefing_json:
                context_parts.append(
                    f"Previous brainbriefing (reuse this template, apply feedback below):\n"
                    f"```json\n{self._last_brainbriefing_json}\n```"
                )
            if self._session.current_output_paths:
                context_parts.append(
                    f"Current outputs: {', '.join(self._session.current_output_paths)}"
                )
            if context_parts:
                return (
                    "\n\n".join(context_parts)
                    + f"\n\nUser feedback: {user_text}\n\n"
                    "Apply the feedback to the previous brainbriefing, keeping everything else the same. "
                    "Assemble the updated workflow and call `signal_workflow_ready(workflow_path)`."
                )
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
                print(f"pipeline: Sanitized Brain history: removed {removed} "
                      f"orphaned tool message(s).")
            self._brain.messages[:] = cleaned

    @staticmethod
    def _sanitize_messages(messages: list[dict]) -> list[dict]:
        """Ensure *messages* don't contain orphaned ``toolResult`` / ``toolUse`` blocks.

        The Anthropic API requires:
        - Every ``tool_result`` content block to have a corresponding ``tool_use``
          block in the immediately preceding assistant message.
        - Every ``tool_use`` block in an assistant message to be followed by a
          user message containing the matching ``tool_result`` blocks.

        This helper trims messages from both ends:
        - **Leading**: removes user messages whose first content block is a
          ``toolResult`` with no preceding ``toolUse``, and assistant messages
          whose ``toolUse`` has no following ``toolResult``.
        - **Trailing**: removes assistant messages that end with unresolved
          ``toolUse`` blocks (i.e. no following user message with ``toolResult``).
          This is the main cause of HTTP 400 errors when a session is interrupted
          mid-tool-call and the same brain agent is reused for the next session.
        """
        # ── Trim leading orphaned toolResult / unresolved toolUse ────────────
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

        # ── Trim trailing unresolved toolUse (causes HTTP 400 on next call) ──
        # If the last message is an assistant message that contains toolUse
        # blocks, Anthropic expects a following user message with toolResult.
        # When the session was interrupted before that result arrived, the next
        # call will be rejected.  Remove such trailing assistant messages so the
        # new user prompt can be appended cleanly.
        while messages:
            last = messages[-1]
            last_content = last.get("content", [])
            if last.get("role") == "assistant" and isinstance(last_content, list):
                has_tool_use = any(
                    isinstance(b, dict) and "toolUse" in b for b in last_content
                )
                if has_tool_use:
                    messages = messages[:-1]
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

        Before compressing, the message history is checked for repeated tool
        calls.  If the Brain used more than 5 tool calls in this session the
        learnings agent is started in a background thread to extract and
        persist any actionable learnings.
        """
        messages = self._brain.messages
        if not messages:
            return

        # ── Self-learning check (fire-and-forget background thread) ─────────
        tool_call_count = count_tool_calls(messages)
        if self._verbose:
            print(f"pipeline: Brain used {tool_call_count} tool call(s) in this session.")
        maybe_run_learnings(messages, session_id=self._session.session_id)

        if self._verbose:
            msg_count = len(messages)
            print(f"pipeline: Compressing Brain history ({msg_count} messages) …")

        try:
            summary = await summarize_conversation(
                messages,
                extra_output_paths=extra_output_paths,
                user_message_image_paths=self._session.last_user_input_images or None,
            )
        except Exception as exc:
            if self._verbose:
                print(f"pipeline: WARNING: conversation summarisation failed ({exc}); "
                      "keeping last 4 messages as fallback.")
            # Fallback: keep only the last few messages to cap token growth.
            # Sanitize to avoid orphaned toolResult blocks at the start.
            self._brain.messages[:] = self._sanitize_messages(messages[-4:])
            return

        if not summary:
            if self._verbose:
                print("pipeline: Empty summary returned; clearing history.")
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

        print(f"pipeline: Chat summary:\n{summary}\n")

        # ── Log compressed summary to file ──────────────────────────────────
        try:
            import datetime
            _log_dir = Path(".logs")
            _log_dir.mkdir(parents=True, exist_ok=True)
            _log_path = _log_dir / "message_history_compressed.log"
            _timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            with open(_log_path, "a", encoding="utf-8") as _lf:
                _lf.write(f"\n{'='*80}\n[{_timestamp}] SESSION: {self._session.session_id}\n{'='*80}\n")
                _lf.write(summary)
                _lf.write("\n")
        except Exception as _log_exc:
            if self._verbose:
                print(f"pipeline: WARNING: failed to write compressed log ({_log_exc})")

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
            print(f"pipeline: Brain history compressed → {len(summary)} chars summary.")
        # Cache the summary so the Researcher can reference it next turn.
        self._last_prior_summary = summary

        # Inject the output paths into the Researcher's history so that on the
        # next chain/follow-up request the Researcher knows which files were
        # produced (executor runs outside the Researcher loop, so they never
        # appear in its messages otherwise).
        _effective_out_paths = extra_output_paths or list(self._session.current_output_paths)
        if _effective_out_paths:
            _out_str = ", ".join(_effective_out_paths)
            self._researcher.messages.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "text": (
                                f"[Session output update] The executor produced the following "
                                f"output file(s) in the last step: {_out_str}. "
                                "If the user refers to 'last output', 'the result', or asks to "
                                "chain/continue with the generated file, use these paths as input."
                            )
                        }
                    ],
                }
            )
            self._researcher.messages.append(
                {
                    "role": "assistant",
                    "content": [{"text": f"Understood. I will use {_out_str} as input when the user refers to the previous output."}],
                }
            )

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
        # Auto-persist a memory when a request completed with a known workflow so
        # future sessions can recall template/model preferences.
        if status == "completed" and raw_json:
            self._auto_save_memory(user_text, raw_json)

    # ── Memory helpers ───────────────────────────────────────────────── #

    def _get_memory_context(self, user_text: str) -> str:
        """Return a formatted memory block for *user_text*, or an empty string.

        Searches the local FAISS store for facts relevant to the user\'s
        current request and formats them as a Markdown section that can be
        prepended to any agent prompt.
        """
        try:
            results = memory_search(user_text, session_id=self._session.session_id, limit=5)
            return format_memories(results)
        except Exception as exc:
            if self._verbose:
                print(f"[memory] context retrieval error: {exc}")
            return ""

    def _auto_save_memory(self, user_text: str, raw_json: str) -> None:
        """Distil a compact memory from a completed researcher→brain run.

        Builds a brief, self-contained sentence from the task description and
        selected template name, then calls ``memory_add`` so future sessions
        can recall the user\'s workflow preferences.  Runs synchronously but
        is entirely best-effort — any error is swallowed.
        """
        try:
            data = json.loads(raw_json)
            task_desc = data.get("task", {}).get("description", "")
            template_name = data.get("template", {}).get("name") or ""
            positive_prompt = data.get("prompt", {}).get("positive", "")
            width = data.get("resolution_width")
            height = data.get("resolution_height")

            parts: list[str] = []
            if task_desc:
                parts.append(task_desc)
            if template_name:
                parts.append(f"using template '{template_name}'")
            if width and height:
                parts.append(f"at {width}x{height}")
            if positive_prompt:
                short_prompt = positive_prompt[:120].rstrip()
                if len(positive_prompt) > 120:
                    short_prompt += "…"
                parts.append(f"| prompt: {short_prompt}")

            if not parts:
                return

            memory_text = "User requested: " + ", ".join(parts) + "."
            memory_add(memory_text, session_id=self._session.session_id)
            if self._verbose:
                print(f"[memory] Saved: {memory_text[:100]}")
        except Exception as exc:
            if self._verbose:
                print(f"[memory] auto-save error: {exc}")

    _MAX_RESEARCHER_RETRIES = 2  # up to 2 correction rounds after the first attempt

    async def _arun_researcher(self, user_input):
        """Async-generator variant of _run_researcher.

        Streams the Researcher's token output (including tool-use events) so
        Chainlit can display it in real time.  Yields standard Strands event
        dicts followed by a single sentinel::

            {"_researcher_done": True, "raw_json": str|None,
             "error": str|None, "researcher_output": str}

        Callers must consume the stream, watch for the sentinel, then act on
        its ``raw_json`` / ``error`` fields.
        """
        if isinstance(user_input, list):
            text_parts = [block["text"] for block in user_input if "text" in block]
            user_text = "\n".join(text_parts)
        else:
            user_text = str(user_input)

        researcher_prompt_text = textwrap.dedent(f"""
            User request:
            {user_text}

            Resolve all fields and output the brainbriefing JSON.
        """).strip()

        memory_ctx = self._get_memory_context(user_text)
        if memory_ctx:
            researcher_prompt_text = memory_ctx + "\n\n" + researcher_prompt_text

        # Inject paths from earlier uploads when the current message has no attachments.
        current_has_images = isinstance(user_input, list) and any("image" in b for b in user_input)
        if not current_has_images and self._session.last_user_input_images:
            _paths_hint = "\n".join(
                f"  - {p}  [image uploaded earlier in this thread, use as input]"
                for p in self._session.last_user_input_images
            )
            researcher_prompt_text += f"\n\nInput image(s) from earlier in this thread:\n{_paths_hint}"

        # Inject the compressed summary from the previous turn so the Researcher
        # can resolve OUTPUT_PATHS (prior generated files) as inputs for this turn.
        # This bridges chained sessions regardless of how triage classified the intent.
        if self._last_prior_summary:
            researcher_prompt_text += (
                f"\n\n[CONVERSATION SUMMARY FROM PRIOR ROUND]\n\n"
                f"{self._last_prior_summary}\n\n"
                f"[END OF SUMMARY — if this request refers to previously generated outputs, "
                f"upload the file(s) from OUTPUT_PATHS via upload_image() and use the "
                f"returned filename as the workflow input.]"
            )

        # If the previous turn was handled by the Info agent (e.g. it crafted a prompt),
        # pass only the key output as a compact hint so the Researcher can reuse it.
        if self._session.last_agent == "info" and self._session.last_info_response:
            _trimmed = self._session.last_info_response[:2000]  # hard cap — keeps tokens low
            researcher_prompt_text += (
                f"\n\nThe Info agent produced the following output in the previous turn "
                f"(use any prompt text or details from it):\n{_trimmed}"
            )
            self._session.last_info_response = None  # consume once

        last_error: str | None = None
        _researcher_snap = self._usage_snapshot(self._researcher)

        for attempt in range(1 + self._MAX_RESEARCHER_RETRIES):
            if attempt == 0:
                prompt = researcher_prompt_text
            else:
                if self._verbose:
                    print(f"pipeline: Researcher retry {attempt}/{self._MAX_RESEARCHER_RETRIES} …")
                    yield {"data": f"\n_pipeline: Researcher retry {attempt}/{self._MAX_RESEARCHER_RETRIES} …_"}
                prompt = textwrap.dedent(f"""
                    Your previous brainbriefing output failed JSON/schema validation:
                    {last_error}

                    Please output ONLY the corrected brainbriefing JSON with all
                    required fields correctly typed. No prose, no markdown fences.
                """).strip()

            chunks: list[str] = []
            async for event in self._researcher.stream_async(prompt):
                if isinstance(event, dict):
                    chunk = event.get("data", "")
                    if chunk:
                        chunks.append(chunk)
                yield event

            last_response = "".join(chunks)
            label = "initial" if attempt == 0 else f"retry {attempt}"
            if self._verbose:
                print(f"pipeline: Researcher finished ({label}). Extracting brainbriefing …")

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
                    print(f"pipeline: Researcher ({label}) validation failed: {last_error}")
                continue

            raw_json = briefing.model_dump_json(indent=2)
            if self._verbose:
                if attempt > 0:
                    print(f"pipeline: Brainbriefing recovered after {attempt} retry(ies).")
                print(
                    f"pipeline: Brainbriefing OK ({label}) — "
                    f"status={briefing.status}, task={briefing.task.description!r}, "
                    f"template={briefing.template.name!r}"
                )
            log_agent_messages("RESEARCHER", list(self._researcher.messages))
            self._record_agent_usage(self._researcher, _researcher_snap)
            yield {"_researcher_done": True, "raw_json": raw_json, "error": None, "researcher_output": raw_json}
            return

        log_agent_messages("RESEARCHER", list(self._researcher.messages))
        self._record_agent_usage(self._researcher, _researcher_snap)
        yield {
            "_researcher_done": True,
            "raw_json": None,
            "error": (
                f"Brainbriefing validation failed after {1 + self._MAX_RESEARCHER_RETRIES} attempts: "
                f"{last_error}"
            ),
            "researcher_output": "",
        }

    def _run_researcher(self, user_input) -> tuple[str | None, str | None, str]:
        """Run the Researcher and return ``(raw_json, error_message, researcher_output)``.

        Calls the Researcher as a normal text agent (preserving its tool use),
        extracts the JSON from the response, then validates it with the
        ``BrainBriefing`` Pydantic model.  On success the model is re-serialised
        so the Brain always receives canonically formatted JSON.

        If validation fails, the error is fed back to the Researcher for up to
        ``_MAX_RESEARCHER_RETRIES`` correction rounds before giving up.

        ``user_input`` may be a plain string *or* a list of Strands content
        blocks (the multimodal format produced by Chainlit when the
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

        # Prepend relevant long-term memories so the Researcher can apply
        # the user's past style preferences and template choices.
        memory_ctx = self._get_memory_context(user_text)
        if memory_ctx:
            researcher_prompt_text = memory_ctx + "\n\n" + researcher_prompt_text

        # If the current message has no image attachments but the user uploaded
        # images earlier in this thread, surface those paths so the Researcher
        # knows what input image(s) to use (e.g. "make a video from it").
        current_has_images = isinstance(user_input, list) and any("image" in b for b in user_input)
        if not current_has_images and self._session.last_user_input_images:
            _paths_hint = "\n".join(
                f"  - {p}  [image uploaded earlier in this thread, use as input]"
                for p in self._session.last_user_input_images
            )
            researcher_prompt_text += f"\n\nInput image(s) from earlier in this thread:\n{_paths_hint}"

        # Inject the compressed summary from the previous turn so the Researcher
        # can resolve OUTPUT_PATHS (prior generated files) as inputs for this turn.
        # This bridges chained sessions regardless of how triage classified the intent.
        if self._last_prior_summary:
            researcher_prompt_text += (
                f"\n\n[CONVERSATION SUMMARY FROM PRIOR ROUND]\n\n"
                f"{self._last_prior_summary}\n\n"
                f"[END OF SUMMARY — if this request refers to previously generated outputs, "
                f"upload the file(s) from OUTPUT_PATHS via upload_image() and use the "
                f"returned filename as the workflow input.]"
            )

        # Always pass only the text prompt to the Researcher.
        # When user_input is a multimodal list, the text block already contains
        # the on-disk paths of any attached files (added by _build_content in chainlit_app.py).
        # The Researcher can call analyze_image(file_path=...) if it needs to inspect
        # an image — much cheaper than embedding raw bytes in every LLM call.
        first_attempt_input: Any = researcher_prompt_text

        last_error: str | None = None
        _researcher_snap = self._usage_snapshot(self._researcher)

        for attempt in range(1 + self._MAX_RESEARCHER_RETRIES):
            if attempt == 0:
                prompt = first_attempt_input
            else:
                # Feed the validation error back so the model can self-correct
                if self._verbose:
                    print(
                        f"pipeline: Researcher retry {attempt}/{self._MAX_RESEARCHER_RETRIES} …"
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
                print(f"pipeline: Researcher finished ({label}). Extracting brainbriefing …")

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
                    print(f"pipeline: Researcher ({label}) validation failed: {last_error}")
                continue

            # Success — re-serialise from validated model so Brain always gets clean JSON
            raw_json = briefing.model_dump_json(indent=2)
            if self._verbose:
                if attempt > 0:
                    print(f"pipeline: Brainbriefing recovered after {attempt} retry(ies).")
                print(
                    f"pipeline: Brainbriefing OK ({label}) — "
                    f"status={briefing.status}, task={briefing.task.description!r}, "
                    f"template={briefing.template.name!r}"
                )
            log_agent_messages("RESEARCHER", list(self._researcher.messages))
            self._record_agent_usage(self._researcher, _researcher_snap)
            return raw_json, None, raw_json

        # All attempts exhausted
        log_agent_messages("RESEARCHER", list(self._researcher.messages))
        self._record_agent_usage(self._researcher, _researcher_snap)
        return None, (
            f"Brainbriefing validation failed after {1 + self._MAX_RESEARCHER_RETRIES} attempts: "
            f"{last_error}"
        ), ""

    def _researcher_blocked_question(self, raw_json: str | None) -> str | None:
        """Return a user-facing question string if the brainbriefing status is 'blocked', else None."""
        if not raw_json:
            return None
        try:
            data = json.loads(raw_json)
            if data.get("status") == "blocked":
                blockers = data.get("blockers") or []
                if blockers:
                    items = "\n".join(f"- {b}" for b in blockers)
                    return f"I need a bit more information before I can proceed:\n\n{items}"
                return "I need more information before I can continue."
        except Exception:
            pass
        return None

    async def _astream_brain_stage(
        self,
        raw_json: str,
        user_text: str,
        triage_result: TriageResult,
        *,
        _is_error_retry: bool = False,
        _override_brain_prompt: str | None = None,
        qa_reply_queue: asyncio.Queue | None = None,
    ):
        """Async generator: stream the full Brain stage (assembly → executor) for a given brainbriefing.

        Clears brain history, builds the brain prompt, streams token output and
        handles ComfyUI interrupts transparently.  Shared by the normal
        Researcher→Brain flow and the blocked-researcher resume path.

        After every executor run the Error Checker agent scans ComfyUI logs.
        On ``error_fixable`` (and when this is not already a retry) the Brain is
        re-invoked once with the error details and the fix plan embedded in the
        prompt.  On ``error_unfixable`` the user-facing error message is yielded
        and the stage terminates.
        """
        self._brain.messages.clear()
        self._ensure_clean_history()
        brain_prompt = _override_brain_prompt or self._build_brain_prompt(raw_json)
        current_input: Any = brain_prompt
        _brain_snap = self._usage_snapshot(self._brain)

        # Track whether a brain-assembly failure was resolved via user advice.
        _assembly_fail_error: str | None = None
        _assembly_fail_advice: str | None = None

        while True:
            interrupt_result = None

            yield {"_brain_start": True}
            async for event in self._brain.stream_async(current_input):
                yield event
                if "result" in event:
                    agent_result = event["result"]
                    if getattr(agent_result, "stop_reason", None) == "interrupt":
                        for intr in getattr(agent_result, "interrupts", []):
                            if getattr(intr, "name", None) == INTERRUPT_NAME:
                                interrupt_result = intr
                                break
            yield {"_brain_done": True}

            if interrupt_result is None:
                # Normal completion — Stage 3: Executor
                workflow_paths_b = _get_workflow_signal()
                workflow_paths_b = self._expand_variations(workflow_paths_b, raw_json)
                self._session.current_output_paths.clear()
                executor_paths_b = self._session.current_output_paths
                _qa_fail_event_b: dict | None = None
                if workflow_paths_b:
                    count = len(workflow_paths_b)
                    if self._verbose:
                        tag = f"{count} workflows (batch)" if count > 1 else workflow_paths_b[0]
                        print(f"pipeline: Brain signaled {tag} ready.")
                        yield {"data": f"\n_pipeline: Brain signaled {tag} ready._"}
                    hdr = f"batch of {count} workflows" if count > 1 else "workflow"
                    yield {"data": f"\n\n_⚙️ Handing off to executor ({hdr})…_"}
                    async for line in _execute_workflows_batch(
                        workflow_paths_b,
                        raw_json,
                        user_message=user_text,
                        verbose=self._verbose,
                        collected_paths=executor_paths_b,
                        run_qa=self._run_qa,
                    ):
                        if isinstance(line, dict) and line.get("qa_fail"):
                            _qa_fail_event_b = line
                            break
                        yield {"data": f"\n{line}"}

                if not workflow_paths_b:
                    # Brain finished without signalling a workflow — assembly failed.
                    latest_wf = _latest_output_workflow()
                    if self._verbose:
                        print(f"pipeline: Brain did not signal any workflow. Latest JSON: {latest_wf}")
                    yield {
                        "brain_assembly_fail_ask": True,
                        "latest_workflow_path": latest_wf or "",
                    }
                    if qa_reply_queue is not None:
                        _advice = await qa_reply_queue.get()
                        if _advice and _advice.strip():
                            yield {"data": "\n_🔄 Retrying with your advice…_"}
                            _assembly_fail_error = (
                                "Brain did not call signal_workflow_ready"
                                + (f" (latest JSON: {latest_wf})" if latest_wf else "")
                            )
                            _assembly_fail_advice = _advice.strip()
                            current_input = (
                                f"The previous workflow assembly attempt failed — "
                                f"`signal_workflow_ready` was never called.\n"
                                f"The user reviewed the latest workflow JSON"
                                + (f" ({latest_wf})" if latest_wf else "")
                                + f" and provided this advice:\n\n{_advice}\n\n"
                                f"Please fix the issue and try again. "
                                f"Call `signal_workflow_ready(workflow_path)` when the workflow is ready.\n\n"
                                f"Original brainbriefing:\n\n{brain_prompt}"
                            )
                            continue
                    # No queue or empty advice — abort gracefully.
                    self._record_chat_summary(user_text, triage_result, status="error", raw_json=raw_json)
                    self._record_agent_usage(self._brain, _brain_snap)
                    return

                if _qa_fail_event_b:
                    if qa_reply_queue is not None:
                        yield {"qa_fail_ask": True, **_qa_fail_event_b}
                        _answer_b = await qa_reply_queue.get()
                        if _is_affirmative(_answer_b):
                            _qa_feedback_prompt = self._build_qa_feedback_prompt(
                                brain_prompt, user_text, _qa_fail_event_b
                            )
                            yield {"data": (
                                "\n\n---\n"
                                "🔍 **QA feedback sent to Brain:**\n\n"
                                + _qa_feedback_prompt
                                + "\n\n---\n"
                            )}
                            yield {"data": "\n_🔄 Retrying with QA feedback…_"}
                            current_input = _qa_feedback_prompt
                            continue  # restart the while True loop
                    # No queue or user declined — abort.
                    self._record_chat_summary(user_text, triage_result, status="qa_failed", raw_json=raw_json)
                    self._record_agent_usage(self._brain, _brain_snap)
                    return

                self._record_chat_summary(user_text, triage_result, status="completed", raw_json=raw_json)
                self._schedule_compression(extra_output_paths=executor_paths_b)
                self._record_agent_usage(self._brain, _brain_snap)
                self._session.last_agent = "brain"
                # If this run succeeded after a user-advice retry, record the learning.
                if _assembly_fail_error and _assembly_fail_advice:
                    from src.utils.learnings import record_user_advice_learning
                    record_user_advice_learning(
                        error_context=_assembly_fail_error,
                        user_advice=_assembly_fail_advice,
                        session_id=self._session.session_id,
                    )
                if self._verbose:
                    print("pipeline: Brain finished.")
                    yield {"data": "\n_pipeline: Brain finished._"}
                break

            # ── ComfyUI interrupt: stream progress, then resume ────── #
            # Reason is JSON-encoded {"prompt_id": ..., "client_id": ...}.
            # Older callers may still send a bare prompt_id string, so handle both.
            raw_reason = interrupt_result.reason or ""
            prompt_id_b: str
            client_id_b: str = ""
            try:
                _r = json.loads(raw_reason)
                if isinstance(_r, dict):
                    prompt_id_b = str(_r.get("prompt_id", ""))
                    client_id_b = str(_r.get("client_id", "") or "")
                else:
                    prompt_id_b = str(_r)
            except Exception:
                prompt_id_b = raw_reason

            if self._verbose:
                print(f"pipeline: ComfyUI interrupt — streaming prompt_id={prompt_id_b}")
                yield {"data": f"\n_pipeline: ComfyUI interrupt — streaming prompt_id={prompt_id_b}_"}
            yield {"data": f"\n\n_⏳ ComfyUI job queued (`{prompt_id_b}`). Streaming progress…_"}

            history_result_b: dict = {}
            async for ev in _stream_comfyui_job(prompt_id_b, client_id_b):
                if isinstance(ev, dict):
                    if "history" in ev:
                        history_result_b = ev["history"]
                    else:
                        history_result_b = ev  # error payload
                    break
                yield {"data": f"\n_{ev}_"}

            yield {"data": "\n_✅ ComfyUI job finished — resuming…_"}
            if self._verbose:
                print(f"pipeline: ComfyUI job {prompt_id_b} finished. Resuming Brain.")
                yield {"data": f"\n_pipeline: ComfyUI job {prompt_id_b} finished. Resuming Brain._"}
            current_input = [
                {
                    "interruptResponse": {
                        "interruptId": interrupt_result.id,
                        "response": json.dumps(history_result_b),
                    }
                }
            ]

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
            Vision QA (via Ollama) and saving outputs to ./output_images.
        """).strip()

    def _build_qa_feedback_prompt(
        self,
        original_brain_prompt: str,
        user_text: str,
        qa_fail_event: dict,
    ) -> str:
        """Build a Brain retry prompt that incorporates Vision QA failure verdicts as feedback.

        Instructs the Brain to revise the workflow or prompt parameters to better
        satisfy the user's original request, using the QA agent's verdict as guidance.
        """
        fail_details: list[dict] = qa_fail_event.get("fail_details", [])
        verdict_lines = "\n".join(
            f"  - {Path(d['path']).name}: {d['verdict']}" for d in fail_details
        )
        return (
            f"The Vision QA agent reviewed the previous output and determined it did NOT "
            f"meet the user's original request.\n\n"
            f"**User's original request:** {user_text}\n\n"
            f"**QA verdicts:**\n{verdict_lines}\n\n"
            f"Please revise the workflow or the prompt parameters to address these issues "
            f"and better satisfy the user's original request. "
            f"Keep the same workflow template and input images unless the QA verdict clearly "
            f"indicates a fundamentally different approach is needed. "
            f"Call `signal_workflow_ready(workflow_path)` when the revised workflow is ready.\n\n"
            f"--- Original Brainbriefing ---\n\n{original_brain_prompt}"
        )

    async def _drain_executor_batch(
        self,
        workflow_paths: list[str],
        brainbriefing_json: str,
        user_message: str = "",
    ) -> tuple[list[str], list[str], dict | None]:
        """Submit all workflows then poll + process each; return ``(status_lines, output_paths, qa_fail_event)``.

        Uses ``execute_workflows_batch`` (submit-all-then-poll) for any number
        of workflows, including a single one.

        ``user_message`` is forwarded to the Vision QA agent as the ground-truth
        reference.  It never enters any agent's context window or history.

        When QA fails the generator yields a ``{"qa_fail": True, ...}`` dict.
        That event is returned as the third element instead of being appended to
        ``lines`` (which is always strings).
        """
        lines: list[str] = []
        output_paths: list[str] = []
        qa_fail_event: dict | None = None
        async for line in _execute_workflows_batch(
            workflow_paths,
            brainbriefing_json,
            user_message=user_message,
            verbose=self._verbose,
            collected_paths=output_paths,
            run_qa=self._run_qa,
        ):
            if isinstance(line, dict) and line.get("qa_fail"):
                qa_fail_event = line
                break
            lines.append(line if isinstance(line, str) else str(line))
            if self._verbose:
                print(f"[executor] {line}")
        return lines, output_paths, qa_fail_event

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
        - ``output_workflows/multiprompt.json`` exists (written by image-batch skill).

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
                print("pipeline: variations=true but positive_prompt_node_id is missing — "
                      "skipping multiprompt expansion.")
            return workflow_paths

        if not _MULTIPROMPT_PATH.exists():
            if self._verbose:
                print("pipeline: variations=true but multiprompt.json not found — "
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
            print(f"pipeline: Variation expansion: 1 base → {len(expanded)} workflows.")
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
