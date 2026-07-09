"""
agentY – Two-agent pipeline: Query Templates → Assemble Workflow.

The pipeline exposes a single callable that accepts a raw user request,
runs it through the Query Templates to produce a brainbriefing JSON, then
hands that JSON to the Assemble Workflow for workflow assembly, execution, and QA.

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
from strands.types.exceptions import MaxTokensReachedException

from src.agent import create_assemble_workflow_agent, create_dop_agent, create_error_checker_agent, create_info_agent, create_orchestrator_agent, create_planner_agent, create_query_templates_agent, create_search_web_agent, create_story_agent, create_detect_user_intent_agent, create_vision_agent, _settings
from src.tools.image_handling import set_vision_agent as _set_vision_agent
from src.utils.chat_summary import summarize_conversation, log_agent_messages, log_agent_exchange
from src.utils.comfyui_interrupt_hook import INTERRUPT_NAME
from src.utils.comfyui_progress import stream_comfyui_job as _stream_comfyui_job
from src.utils.progress_signal import drain as _drain_progress
from src.utils.tool_activity import drain as _drain_tools, clear as _clear_tools
from src.utils.canvas_patch import drain as _drain_canvas_patch, clear as _clear_canvas_patch
from src.utils.costs import compute_cost_from_usage
from src.utils.models import AgentSession, ChatSummary, GeneratedImage, MessageIntent, TriageResult
from src.utils.triage import detect_user_intent as _triage, route as _route
from src.utils.workflow_signal import clear_and_get as _get_workflow_signal
from src.executor import execute_workflow as _execute_workflow, execute_workflows_batch as _execute_workflows_batch
from src.utils.memory import format_memories, memory_add, memory_search
from src.tools.memory_tools import set_session_id as _set_memory_session_id
from src.tools.comfyui import clear_tool_caches as _clear_tool_caches
# Deterministic brain happy-path: the mechanical assembly (load template ->
# apply briefing -> validate) lives in the shared assembly_deterministic module;
# the pipeline calls it, then signals. signal_workflow_ready is a Strands
# DecoratedFunctionTool, so use __wrapped__ to call it directly.
from agenty_core.tools.assembly_deterministic import assemble_workflow_deterministic as _assemble_workflow_deterministic
from src.tools.workflow_handoff import signal_workflow_ready as _det_signal_tool
_det_signal_ready = _det_signal_tool.__wrapped__
# Deterministic download+rerun: resolve a named missing model on HF and fetch it
# into ComfyUI's extra model path, then retry the query_templates.
from agenty_core.tools.huggingface import find_hf_file as _find_hf_file
from agenty_core.tools.huggingface import download_hf_model as _download_hf_model
from src.utils.learnings import count_tool_calls, maybe_run_learnings
from src.utils.debug_log import trace as _trace


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
    """Structured handoff document from the Query Templates to the Assemble Workflow."""
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

# Image file extensions registered in the per-thread generated-image gallery.
_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp"}


def _is_image_file(path: str) -> bool:
    """Return True when *path* points to an image file (by extension)."""
    return Path(path).suffix.lower() in _IMAGE_SUFFIXES


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
# Hard-constraint extraction (guided orchestrator)
# ---------------------------------------------------------------------------
# Generic task/plumbing words that appear in template names but carry no
# "brand" identity. Everything left after stripping these is a distinctive
# model/template name (flux, wan, qwen, kling, nano, banana, vace, krea, …) —
# exactly what a user means when they say "use Nano Banana".
_ORCH_STOP_TOKENS: frozenset[str] = frozenset({
    "image", "images", "video", "videos", "edit", "editing", "editor", "workflow",
    "workflows", "basic", "simple", "advanced", "standard", "default", "api",
    "t2i", "i2v", "v2v", "t2v", "flf", "txt2img", "img2img", "text", "to", "and",
    "the", "with", "from", "for", "gen", "generation", "generate", "model", "models",
    "upscale", "upscaler", "upscaling", "portrait", "lighting", "camera", "motion",
    "style", "background", "remove", "removal", "swap", "face", "inpaint",
    "inpainting", "outpaint", "outpainting", "control", "controlnet", "lora",
    "sampler", "sampling", "latent", "vae", "clip", "encode", "decode", "load",
    "save", "node", "nodes", "example", "examples", "template", "templates",
    "comfy", "comfyui", "ref", "reference", "start", "frame", "multishot", "multi",
    "shot", "shots", "sequence", "clip", "clips", "audio", "sound", "speech",
    "first", "last", "single", "dual", "batch", "run", "versions", "variations",
    "new", "old", "pro", "plus", "mini", "small", "large", "base", "full", "high",
    "low", "res", "quality", "fast", "turbo", "lite", "light", "photo", "picture",
})


def _split_identifier(name: str) -> str:
    """Insert spaces at camelCase and letter/digit boundaries.

    Template names are camelCase / snake blends — ``imageEdit_nano_banana2`` and
    ``NanoBanana2_outpaintUpscale`` must both yield the words *nano* and *banana*
    (not blobs like ``nanobanana2``) so a plain "use Nano Banana" matches.
    """
    s = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", name)          # camelCase → camel Case
    s = re.sub(r"(?<=[A-Za-z])(?=[0-9])", " ", s)          # banana2 → banana 2
    s = re.sub(r"(?<=[0-9])(?=[A-Za-z])", " ", s)          # 3x → 3 x
    return s


def _brand_tokens(name: str) -> list[str]:
    """Return the distinctive (brand/model) tokens of a template name.

    Splits camelCase/snake/digit boundaries, drops the generic task words in
    ``_ORCH_STOP_TOKENS``, pure digits, and 1-2 char fragments — leaving the
    identifying tokens (e.g. ``imageEdit_nano_banana2`` → ``['nano', 'banana']``).
    """
    toks = re.findall(r"[a-z0-9]+", _split_identifier(name).lower())
    return [t for t in toks if t not in _ORCH_STOP_TOKENS and len(t) >= 3 and not t.isdigit()]


# ---------------------------------------------------------------------------
# Pipeline callable
# ---------------------------------------------------------------------------

class Pipeline:
    """Chains Query Templates → Assemble Workflow with logging and JSON validation.

    Call ``pipeline(user_input)`` just like a Strands Agent.
    The Query Templates runs once per call (stateless); the Assemble Workflow keeps
    a sliding-window conversation so multi-turn interactions work.

    ``stream_async`` is also supported so Chainlit can update
    its message in real-time from the Assemble Workflow stage.
    ``event_loop_metrics`` delegates to the Assemble Workflow agent so token-usage
    reporting in Chainlit continues to work.
    """

    def __init__(
        self,
        query_templates: Agent,
        assemble_workflow: Agent,
        *,
        info_agent: Agent | None = None,
        story_agent: Agent | None = None,
        triage_agent: Agent | None = None,
        planner_agent: Agent | None = None,
        error_checker_agent: Agent | None = None,
        scout_agent: Agent | None = None,
        dop_agent: Agent | None = None,
        orchestrator_agent: Agent | None = None,
        free_agent: bool = True,
        verbose: bool = True,
        skip_brain: bool = False,
        info_context: dict | None = None,
        session_id: str = "default",
    ) -> None:
        self._researcher = query_templates
        self._assemble_workflow = assemble_workflow
        # Legacy alias used by a few older code paths (_ensure_clean_history,
        # _compress_brain_history). Kept in sync with _assemble_workflow.
        self._brain = assemble_workflow
        self._info_agent: Agent = info_agent or create_info_agent()
        self._story_agent: Agent = story_agent or create_story_agent()
        # Free-agent mode routes every turn through the orchestrator, so the
        # triage classifier is not built (and not needed). It is only constructed
        # for the legacy pipeline path.
        self._free_agent = free_agent
        self._triage_agent: Agent | None = (
            triage_agent if free_agent else (triage_agent or create_detect_user_intent_agent())
        )
        self._planner_agent: Agent = planner_agent or create_planner_agent()
        self._error_checker_agent: Agent = error_checker_agent or create_error_checker_agent()
        self._search_web_agent: Agent = scout_agent or create_search_web_agent()
        self._dop_agent: Agent = dop_agent or create_dop_agent()
        # Orchestrator (the free-agent entry point) + its delegation tools. The
        # delegation tools are closures over this Pipeline so they always hit the
        # current specialist instances (surviving /switch_model rebuilds).
        self._orchestrator_agent: Agent | None = None
        # Canvas-hook mode (set per-turn): the spliced base API prompt of the
        # user's on-canvas graph + the hook directives attached to it.
        self._canvas_base_prompt: dict | None = None
        self._canvas_hooks: list = []
        # Outputs produced mid-turn by run_workflow_now (chained hook stages).
        # Tracked so they survive the end-of-turn current_output_paths reset and
        # still get staged onto the canvas. Empty on every non-chain turn.
        self._chain_output_paths: list = []
        # Snapshot of the nodes the user has selected on the canvas this turn
        # (id/type/title/widgets), so the orchestrator can read — and, via
        # set_canvas_node_params, write back — arbitrary node parameters.
        self._canvas_selection: list = []
        self._delegation_tools: list = self._build_delegation_tools()
        if orchestrator_agent is not None:
            self.set_orchestrator(orchestrator_agent)
        self._verbose = verbose
        self._skip_brain = skip_brain
        # Storyboard director: max Vision-QA attempts per visual step before the
        # user is asked whether to keep retrying (settings.json director.auto_retries).
        self._storyboard_max_qa = self._resolve_director_retries()
        # Default for the per-step approval gate (True = ask). Overridable per-run
        # by the user's message (settings.json director.user_approval_step).
        self._approval_default = self._resolve_director_approval()
        # Default for the DoP cinematography pass (True = run it). Overridable
        # per-run by the user's message (settings.json director.apply_cinematography).
        self._cinematography_default = self._resolve_director_cinematography()
        self._info_context: dict = info_context or {}
        self._session: AgentSession = AgentSession(session_id=session_id)
        # Brainbriefing JSON from the most recent Query Templates run; used by the
        # Executor for Vision QA comparison in follow-up / feedback-loop rounds.
        self._last_brainbriefing_json: str | None = None
        # Compressed summary text from the previous turn, cached so it can be
        # injected into the Query Templates on the NEXT turn regardless of triage intent.
        # This is the authoritative source for OUTPUT_PATHS / INPUT_PATHS_USER_MESSAGE
        # that bridges chained sessions even when triage says "new_request".
        self._last_prior_summary: str | None = None
        # Whether the current turn requested an explicit Vision QA pass.
        self._run_qa: bool = False
        # Bind the memory tools module-level session so memory_read / memory_write
        # always operate on the correct per-session namespace.
        _set_memory_session_id(session_id)
        # Reset session-level tool-response caches so every new pipeline session
        # fetches fresh data from ComfyUI instead of reusing stale results from
        # a previous session in the same process.
        _clear_tool_caches()
        # Initialise Vision Agent so analyze_image(mode='describe') works for
        # all agents (Query Templates, Info, etc.) in this pipeline. Keep a reference so
        # its per-turn token usage can be folded into the cost accounting (it runs
        # outside the per-agent snapshot brackets, via the analyze_image tool).
        self._vision_agent: Agent | None = None
        try:
            self._vision_agent = create_vision_agent()
            _set_vision_agent(self._vision_agent)
        except Exception as _va_exc:
            print(f"[agentY] WARNING: could not initialise VisionAgent ({_va_exc}). "
                  "analyze_image will fall back to mode='full'.")
        # Per-turn usage tracking: list of (delta_usage_dict, agent_obj) for every
        # agent that contributed tokens this turn. Reset at the start of each turn.
        self._last_turn_usages: list = []
        # Turn-start snapshot of the Vision agent's accumulated usage, so its
        # delta for the turn can be recorded at cost-finalisation time.
        self._vision_usage_snap: dict = {}
        # Brain-history compression is moved off the critical path: the executor
        # finishes, the final stream events flow to the UI, and only then this
        # background task summarises the conversation.  The next user turn
        # awaits it so the brain never sees uncompressed history.
        self._pending_compression: asyncio.Task | None = None

    # Max seconds the next turn will wait for the previous turn's deferred
    # history compression. Compression is a background optimisation (a cheap
    # Ollama summary) and must never hold the next user turn hostage — e.g. if
    # Ollama is slow swapping the summariser model into VRAM that a prior
    # generation just filled. On timeout the task is cancelled and the turn
    # proceeds with the brain's recent (sanitised) history instead.
    _COMPRESSION_WAIT_TIMEOUT = 30.0

    async def _await_pending_compression(self) -> None:
        """Block until any deferred brain-history compression has finished.

        Bounded by ``_COMPRESSION_WAIT_TIMEOUT`` so a stuck or slow background
        summary can never hang the next turn indefinitely.
        """
        task = self._pending_compression
        if task is None:
            return
        self._pending_compression = None
        try:
            await asyncio.wait_for(task, timeout=self._COMPRESSION_WAIT_TIMEOUT)
        except asyncio.TimeoutError:
            # wait_for has already cancelled the task; cancelling at the LLM
            # await point means brain.messages was not yet replaced, so the
            # history is simply left uncompressed for this turn (the brain
            # paths sanitise it before use). Better a bigger prompt than a hang.
            print(
                f"pipeline: deferred compression exceeded {self._COMPRESSION_WAIT_TIMEOUT:.0f}s — "
                "cancelled; continuing with recent history."
            )
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
    # the combined picture (triage + query_templates + assemble_workflow + info, etc.) instead
    # of only the Assemble Workflow.
    @property
    def event_loop_metrics(self):  # noqa: ANN201
        # Fold in the Vision agent's per-turn delta before reporting, so the
        # combined token picture includes analyze_image usage.
        self._record_vision_usage()
        return _TurnMetrics(self._last_turn_usages)

    # ── Per-turn usage tracking helpers ─────────────────────────────── #

    def _record_vision_usage(self) -> None:
        """Fold the shared Vision agent's per-turn token delta into the turn usage.

        The Vision agent (used by the ``analyze_image`` tool across Query Templates /
        Info / etc.) runs outside the per-agent ``_usage_snapshot`` brackets, so
        its tokens are captured here from the turn-start snapshot taken in
        :meth:`stream_async`. Idempotent within a turn: after recording, the
        snapshot is advanced so a repeat call contributes nothing (and a later
        call only adds vision tokens accrued since). Priced at the Vision model's
        own rate via ``_cost_meta`` — so an Ollama vision model contributes 0 cost
        (but its tokens still count toward the displayed total).
        """
        agent = self._vision_agent
        if agent is None:
            return
        self._record_agent_usage(agent, self._vision_usage_snap)
        self._vision_usage_snap = self._usage_snapshot(agent)

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
        # Ensure the Vision agent's per-turn usage is included before pricing.
        self._record_vision_usage()
        total_cost = 0.0
        total_tokens = 0
        for usage, agent in self._last_turn_usages:
            cost, tokens = compute_cost_from_usage(usage, agent)
            total_cost += cost
            total_tokens += tokens
        return total_cost, total_tokens

    def run(self, user_input, **_: Any) -> str:
        """Run the full pipeline for *user_input* and return the assembled response.

        Thin synchronous wrapper over :meth:`stream_async` — the single source of
        truth for triage, routing, and execution.  It drives the async event
        stream to completion on a private event loop, collecting the same
        user-facing text Chainlit shows in its main message (everything except
        the Researcher's internal token stream), and bridges the interactive
        QA / brain-assembly prompts to the console via ``input()``.

        Used by the CLI entry point (``src/main.py``).
        """
        async def _consume() -> str:
            parts: list[str] = []
            in_researcher = False
            qa_q: asyncio.Queue = asyncio.Queue()
            async for event in self.stream_async(user_input, qa_reply_queue=qa_q):
                if not isinstance(event, dict):
                    continue

                # Interactive prompts: stream_async yields the request, then
                # blocks on the queue for the answer.  Bridge them to the console.
                if event.get("brain_assembly_fail_ask"):
                    _latest = event.get("latest_workflow_path", "")
                    if _latest:
                        print(f"\n⚠️  Brain failed to assemble a workflow. Latest JSON: {_latest}")
                    try:
                        _advice = input("Advice for the Brain (blank to abort): ").strip()
                    except (EOFError, KeyboardInterrupt):
                        _advice = ""
                    await qa_q.put(_advice)
                    continue
                if event.get("qa_fail_ask"):
                    print("\n⚠️  QA check failed.")
                    for _d in event.get("fail_details", []):
                        print(f"   • {Path(_d['path']).name}: {_d['verdict']}")
                    try:
                        _answer = input("🔁 Retry this step? [y/n]: ").strip()
                    except (EOFError, KeyboardInterrupt):
                        _answer = "n"
                    await qa_q.put(_answer)
                    continue
                if event.get("_references_ready"):
                    _rp = event.get("paths", [])
                    if _rp:
                        print(f"\n🌐 {event.get('caption') or 'Web references found'}:")
                        for _p in _rp:
                            print(f"   • {_p}")
                    continue
                if event.get("approval_ask"):
                    _label = event.get("description") or event.get("label") or "this step"
                    print(f"\n⏸️  Approval needed — {_label}")
                    for _p in event.get("image_paths", []):
                        print(f"   • {_p}")
                    try:
                        _answer = input(
                            "✅ Approve and continue? [y = approve / n = abort / or type a revision note]: "
                        ).strip()
                    except (EOFError, KeyboardInterrupt):
                        _answer = "y"
                    await qa_q.put(_answer)
                    continue

                # The Researcher's internal stream (its brainbriefing JSON) is
                # bracketed by these markers and excluded from the response,
                # mirroring how Chainlit hides it inside a collapsible step.
                if event.get("_researcher_start"):
                    in_researcher = True
                    continue
                if event.get("_researcher_done"):
                    in_researcher = False
                    continue

                _data = event.get("data")
                if _data and not in_researcher:
                    parts.append(_data)

            # The CLI has no persistent event loop to await deferred compression
            # on the next turn, so finish it here before this loop is torn down.
            await self._await_pending_compression()
            return "".join(parts)

        return asyncio.run(_consume())

    # ── Free-agent orchestrator ──────────────────────────────────────────── #

    def set_orchestrator(self, agent: Agent) -> None:
        """Install *agent* as the live orchestrator and wire its meta-tools.

        Grabs the agent's ``AgentSkills`` plugin (attached by
        ``create_orchestrator_agent``) and registers it with the orchestration
        module so ``create_skill`` re-scans the correct, live plugin instance.
        """
        self._orchestrator_agent = agent
        plugin = getattr(agent, "_agentskills_plugin", None)
        try:
            from src.tools.orchestration import set_orchestrator_context
            set_orchestrator_context(agent=agent, skills_plugin=plugin)
        except Exception as exc:  # noqa: BLE001
            if getattr(self, "_verbose", False):
                print(f"pipeline: WARNING: could not wire orchestrator context ({exc}).")

    def _build_delegation_tools(self) -> list:
        """Build the specialist-as-tool closures the orchestrator can delegate to.

        Each is an async ``@tool`` bound to this Pipeline, so it always invokes the
        *current* specialist instance (surviving ``/switch_model`` rebuilds) and
        folds the specialist's per-turn token usage into the turn cost.
        """
        from strands import tool as _tool

        async def _run_specialist(agent, label: str, text: str) -> str:
            snap = self._usage_snapshot(agent)
            try:
                out = str(await agent.invoke_async(text))
            finally:
                self._record_agent_usage(agent, snap)
                try:
                    agent.messages.clear()
                except Exception:  # noqa: BLE001
                    pass
            log_agent_exchange(label, text, out)
            return out

        @_tool
        async def run_research(request: str) -> str:
            """Resolve a request into a validated ComfyUI **brainbriefing** JSON.

            Returns the brainbriefing (template + models + prompts + input/output
            node bindings). Feed the result to ``apply_brainbriefing`` to assemble
            the workflow, then ``signal_workflow_ready``.

            Args:
                request: A natural-language description of what to generate/edit.
            """
            raw_json = None
            error = None
            researcher_output = ""
            async for _ev in self._arun_researcher(request):
                if isinstance(_ev, dict) and "_researcher_done" in _ev:
                    raw_json = _ev.get("raw_json")
                    error = _ev.get("error")
                    researcher_output = _ev.get("researcher_output", "")
            if error:
                return json.dumps({"error": error})
            if raw_json:
                self._last_brainbriefing_json = raw_json
                return raw_json
            return researcher_output or json.dumps({"error": "researcher produced no briefing"})

        @_tool
        async def run_info(question: str) -> str:
            """Answer a read-only question about installed models, workflows, or capabilities.

            Args:
                question: The user's question about what agentY/ComfyUI can do.
            """
            return await _run_specialist(self._info_agent, "INFO", self._prepend_gallery(question))

        @_tool
        async def run_story(request: str) -> str:
            """Write a short synopsis or scene descriptions for a visual story.

            Args:
                request: What to write (e.g. "a 3-scene synopsis about …").
            """
            return await _run_specialist(self._story_agent, "STORY", request)

        @_tool
        async def run_dop(text: str) -> str:
            """Rewrite a prompt/storyboard with concrete cinematography (light/camera/colour).

            Args:
                text: The prompt or storyboard to enrich.
            """
            return await _run_specialist(self._dop_agent, "DOP", text)

        @_tool
        async def run_web_search(request: str) -> str:
            """Search the web and stage reference image(s); returns a JSON manifest.

            Args:
                request: What reference to find (e.g. "a 1950s American diner interior").
            """
            return await _run_specialist(self._search_web_agent, "WEB", request)

        @_tool
        async def run_planner(request: str) -> str:
            """Decompose a complex, multi-stage request into ordered steps (JSON).

            Use only for genuinely multi-step projects; simple requests need no plan.

            Args:
                request: The multi-part request to break down.
            """
            return await _run_specialist(self._planner_agent, "PLANNER", request)

        @_tool
        async def apply_canvas_hooks(resolutions: list) -> str:
            """Run the user's ON-CANVAS graph, expanded per the canvas hooks.

            Use this ONLY when a ``[CANVAS HOOKS]`` block is present. It runs the
            graph the user has open (already captured this turn) — do NOT assemble
            a template or call ``run_research``. Each resolution mutates ONE input
            of one anchor node across a set of values; the batch is the Cartesian
            product of all resolutions (capped), and each variant is queued for
            execution automatically.

            Each resolution is an object::

                {"target_node_id": "12", "param": "seed",
                 "mode": "sweep_seed", "count": 6}
                {"target_node_id": "4", "param": "text", "mode": "value_list",
                 "values": ["a cat, cinematic", "a dog, cinematic"]}
                {"target_node_id": "9", "param": "image", "mode": "folder",
                 "folder": "C:/inputs", "extensions": ["png", "jpg"]}

            ``param`` is the input/widget name on the anchor node (see its inputs
            in the ``[CANVAS HOOKS]`` block). Modes: ``sweep_seed`` (needs
            ``count``, optional integer ``start``), ``value_list`` (needs
            ``values``), ``folder`` (needs ``folder``, optional ``extensions`` and
            ``use_full_path``). Call this ONCE with all resolutions.

            Args:
                resolutions: list of per-node mutation specs (see above).
            """
            import os as _os
            import tempfile as _tempfile
            from src.utils.canvas_hooks import build_batch as _build_batch
            from src.utils.workflow_signal import append_workflow_path as _append

            base = getattr(self, "_canvas_base_prompt", None)
            if not base:
                return json.dumps({
                    "error": "no on-canvas graph is loaded for this turn — "
                             "apply_canvas_hooks is only valid with a [CANVAS HOOKS] block."
                })
            try:
                cap = int(_os.environ.get("AGENTY_MAX_CANVAS_BATCH", "25") or "25")
            except ValueError:
                cap = 25
            prompts, notes = _build_batch(base, list(resolutions or []), cap=cap)
            if not prompts:
                return json.dumps({"error": "no batch was produced", "notes": notes})
            out_dir = Path(_tempfile.mkdtemp(prefix="agenty_canvas_"))
            paths: list[str] = []
            for i, p in enumerate(prompts):
                fp = out_dir / f"canvas_{i:03d}.json"
                fp.write_text(json.dumps(p), encoding="utf-8")
                _append(str(fp))
                paths.append(str(fp))
            if self._verbose:
                print(f"pipeline: apply_canvas_hooks queued {len(paths)} canvas variant(s).")
            return json.dumps({
                "status": "queued",
                "count": len(paths),
                "notes": notes,
                "message": (
                    f"{len(paths)} canvas graph variant(s) queued for execution — "
                    "your work here is done; do NOT call signal_workflow_ready."
                ),
            })

        @_tool
        async def run_workflow_now(workflow_path: str) -> str:
            """Run a validated workflow NOW (synchronously) and return its output paths.

            Use this ONLY to CHAIN stages — when you need one workflow's OUTPUT as
            the INPUT to the next (a canvas hook chain, or any
            generate-then-transform pipeline like upscale→animate). Unlike
            ``signal_workflow_ready`` (which defers execution to the end of the
            turn and can't feed a later stage), this submits the workflow now,
            waits for ComfyUI to finish, stages the results onto the canvas, and
            returns the absolute output file paths — so you can ``upload_image`` one
            and bind it to the next stage's loader, then run that stage.

            For a single, terminal generation use ``signal_workflow_ready`` instead;
            do NOT also signal a workflow you already ran here.

            Args:
                workflow_path: Absolute path to a validated workflow JSON file
                    (assemble + validate it first, exactly as for signalling).
            """
            from src.executor import execute_workflow as _execute_workflow

            base = self._session.current_output_paths
            before = len(base)
            brief = self._last_brainbriefing_json or "{}"
            try:
                async for _line in _execute_workflow(
                    workflow_path, brief, user_message="", verbose=self._verbose,
                    collected_paths=base, run_qa=False,
                ):
                    if self._verbose:
                        print(f"[run_workflow_now] {_line}")
            except Exception as exc:  # noqa: BLE001
                return json.dumps({"error": f"execution failed: {exc}"})
            new = list(base[before:])
            # Preserve these past the end-of-turn current_output_paths reset so
            # they're still staged onto the canvas.
            self._chain_output_paths.extend(new)
            if self._verbose:
                print(f"pipeline: run_workflow_now produced {len(new)} output(s).")
            return json.dumps({
                "status": "done",
                "outputs": new,
                "message": (
                    f"{len(new)} output(s) produced and staged onto the canvas. To "
                    "chain: upload_image one of these paths and bind it to the next "
                    "stage's loader, then run that stage."
                ) if new else "Workflow ran but produced no output files.",
            })

        @_tool
        async def add_canvas_workflow(name: str, description: str = "") -> str:
            """Save the workflow currently open in the ComfyUI canvas as a custom template.

            Use when the user asks to add / save the graph they have open in the
            canvas (e.g. "add the workflow open in the canvas", "save this graph
            as a template"). The on-canvas graph is captured automatically each
            turn; this registers it in the custom-template corpus exactly as if
            the user had added a JSON file, and regenerates the recipe database so
            the new template is immediately usable. Hook nodes are stripped out.

            Args:
                name: A short template name (filename-safe stem, no spaces/slashes).
                description: Optional one-line description; auto-generated if omitted.
            """
            base = getattr(self, "_canvas_base_prompt", None)
            if not base:
                return json.dumps({
                    "error": "no workflow is open in the canvas this turn — ask the user to "
                             "open a graph in ComfyUI, then try again."
                })
            try:
                from src.utils.workflow_admin import register_workflow, format_recipe_counts
                res = await asyncio.to_thread(register_workflow, dict(base), name)
                return json.dumps({
                    "status": "added",
                    "name": res["name"],
                    "template_file": res["template_file"],
                    "description": res["description"],
                    "recipes": res["recipes"],
                    "message": (f"Canvas workflow saved as '{res['name']}'. "
                                f"{format_recipe_counts(res['recipes'])}."),
                })
            except Exception as exc:  # noqa: BLE001
                return json.dumps({"error": str(exc)})

        @_tool
        async def set_canvas_node_params(node_id: str, params: dict) -> str:
            """Write parameter values back onto a node selected on the ComfyUI canvas.

            Use when the user asks you to change a value on a node they have
            selected — e.g. "rewrite this prompt", "set steps to 30", "bump the
            CFG". The nodes the user selected are listed in the ``[CANVAS
            SELECTION]`` block with their current widget values; read the value
            there, then call this to apply your change. The edit lands on the live
            graph instantly (no browser refresh, no re-queue). It does NOT run the
            graph — the user queues it themselves when ready.

            Args:
                node_id: The id of a node from the ``[CANVAS SELECTION]`` block.
                params: Mapping of widget name -> new value, e.g.
                    ``{"text": "a rainy neon street"}`` or ``{"steps": 30, "cfg": 6.5}``.
                    Only include the widgets you are changing.
            """
            sel = getattr(self, "_canvas_selection", []) or []
            node = next((n for n in sel if str(n.get("id")) == str(node_id)), None)
            if node is None:
                ids = ", ".join(str(n.get("id")) for n in sel) or "(none selected)"
                return json.dumps({
                    "error": f"node '{node_id}' is not in the current canvas selection. "
                             f"Selected node ids: {ids}."
                })
            if not isinstance(params, dict) or not params:
                return json.dumps({"error": "params must be a non-empty mapping of widget -> value."})
            widgets = node.get("widgets", {}) or {}
            unknown = [k for k in params if k not in widgets]
            from src.utils.canvas_patch import push as _push_patch
            _push_patch({
                "node_id": str(node_id),
                "params": params,
                "node_title": node.get("title") or node.get("type") or "",
            })
            result = {
                "status": "applied",
                "node_id": str(node_id),
                "node": node.get("title") or node.get("type"),
                "changed": params,
            }
            if unknown:
                # Not fatal — the frontend adds the widget if it can — but flag it.
                result["warning"] = (f"widget(s) {unknown} were not in the node's known "
                                     "widgets; applied anyway if the node accepts them.")
            return json.dumps(result)

        @_tool
        async def classify_intent(message: str) -> str:
            """Classify the user's message intent (advisory — you still decide).

            Consult this when a request is ambiguous and you're unsure how to route
            it (e.g. is it a fresh generation, a follow-up/chain on prior output, a
            plain question, creative writing, or a full storyboard?). Returns a JSON
            object ``{"intent": ..., "confidence": ..., "run_qa": ...}``. Treat it as
            a hint, not an order.

            Args:
                message: The user's message to classify.
            """
            if self._triage_agent is None:
                return json.dumps({"error": "intent classifier not available"})
            try:
                res = await _triage(message, self._session, self._info_context, self._triage_agent)
                return json.dumps({
                    "intent": res.intent.value,
                    "confidence": res.confidence,
                    "run_qa": res.run_qa,
                })
            except Exception as exc:  # noqa: BLE001
                return json.dumps({"error": str(exc)})

        return [run_research, run_info, run_story, run_dop, run_web_search,
                run_planner, classify_intent, apply_canvas_hooks, run_workflow_now,
                add_canvas_workflow, set_canvas_node_params]

    def _ensure_orch_clean_history(self) -> None:
        """Sanitize the orchestrator's message list (drop orphaned tool blocks)."""
        agent = self._orchestrator_agent
        if agent is None:
            return
        msgs = getattr(agent, "messages", None)
        if not msgs:
            return
        cleaned = self._sanitize_messages(list(msgs))
        if len(cleaned) != len(msgs):
            if self._verbose:
                print(f"pipeline: Sanitized orchestrator history: removed "
                      f"{len(msgs) - len(cleaned)} orphaned tool message(s).")
            agent.messages[:] = cleaned

    def _template_brand_index(self) -> dict[str, set[str]]:
        """Return {template_name: {brand_tokens}} for every catalog template.

        Built once per pipeline from the local workflow catalog (no ComfyUI
        server call). Used to detect when the user explicitly names a template.
        """
        idx = getattr(self, "_brand_index_cache", None)
        if idx is not None:
            return idx
        idx = {}
        try:
            from agenty_core.tools.comfyui import get_workflow_catalog as _cat
            catalog = json.loads(_cat() or "{}")
            if isinstance(catalog, dict):
                for name in catalog:
                    toks = _brand_tokens(str(name))
                    if toks:
                        idx[name] = set(toks)
        except Exception as exc:  # noqa: BLE001
            if getattr(self, "_verbose", False):
                print(f"pipeline: brand-index build failed ({exc}); template pinning off.")
        self._brand_index_cache = idx
        return idx

    def _match_named_templates(self, user_text: str) -> tuple[str, list[str]] | None:
        """Detect an explicitly-named template in *user_text*.

        Returns ``(phrase, [template_names])`` when the user's words contain all
        brand tokens of one or more templates (highest-specificity match wins), or
        None. A single-token match must be ≥4 chars to avoid incidental hits.
        """
        index = self._template_brand_index()
        if not index:
            return None
        msg = set(re.findall(r"[a-z0-9]+", user_text.lower()))
        if not msg:
            return None
        best: list[tuple[str, set[str]]] = []
        best_score = 0
        for name, toks in index.items():
            if not toks or not toks <= msg:
                continue
            score = len(toks)
            if score == 1 and max(len(t) for t in toks) < 4:
                continue
            if score > best_score:
                best_score, best = score, [(name, toks)]
            elif score == best_score:
                best.append((name, toks))
        if not best:
            return None
        names = sorted({n for n, _ in best})
        phrase = " ".join(sorted(best[0][1]))
        return phrase, names

    def _extract_hard_constraints(self, user_text: str) -> list[str]:
        """Extract MUST-HONOR directives the orchestrator may not silently drop.

        Covers the two things an unconstrained agent most often ignores: an
        explicitly-named template, and a provided input image.
        """
        lines: list[str] = []
        matched = self._match_named_templates(user_text)
        if matched:
            phrase, names = matched
            if len(names) == 1:
                lines.append(
                    f'The user explicitly asked for the "{names[0]}" template — you '
                    "MUST use that exact template and MUST NOT substitute a different one."
                )
            else:
                lines.append(
                    f'The user explicitly named "{phrase}" — you MUST use one of these '
                    f'matching templates (never an unrelated one): {", ".join(names)}.'
                )
        imgs = list(self._session.last_user_input_images or [])
        if imgs:
            names_i = ", ".join(os.path.basename(p) for p in imgs)
            lines.append(
                f"The user provided input file(s): {names_i}. You MUST use them as the "
                "workflow input(s) (stage images with upload_image and bind to the loader "
                "node; use video paths directly in the video loader); do NOT fall back to a "
                "template's default input."
            )
        return lines

    def _describe_canvas_selection(self) -> str:
        """Render the selected canvas nodes + their params as a context block.

        The orchestrator reads a node's current values here ("read this prompt")
        and edits them by calling ``set_canvas_node_params(node_id, params)``.
        Returns "" when nothing informative is selected (loader-only selections
        are already handled as inputs, so they're skipped to avoid noise).
        """
        sel = getattr(self, "_canvas_selection", []) or []
        if not sel:
            return ""
        lines: list[str] = []
        for n in sel:
            widgets = n.get("widgets") or {}
            if not isinstance(widgets, dict) or not widgets:
                continue
            nid = n.get("id")
            ntype = n.get("type") or "?"
            title = n.get("title") or ntype
            head = f"- node #{nid} [{ntype}]" + (f' "{title}"' if title and title != ntype else "")
            lines.append(head)
            for wname, wval in widgets.items():
                sval = str(wval)
                if len(sval) > 400:
                    sval = sval[:400] + "…"
                lines.append(f"    • {wname} = {sval!r}")
        if not lines:
            return ""
        return (
            "[CANVAS SELECTION — nodes the user has selected on the ComfyUI graph, "
            "with their current parameter values. To change a value, call "
            "set_canvas_node_params(node_id, {widget: new_value}); the edit lands on "
            "the live canvas. Do not run the graph unless asked.]\n"
            + "\n".join(lines)
        )

    def _build_orchestrator_input(self, user_input, user_text: str):
        """Assemble the orchestrator's input: hard constraints + gallery + message.

        Any explicit, non-negotiable constraints (a named template, provided input
        images) are pinned as a MUST-HONOR block at the very top so the free
        orchestrator cannot silently drop them. For multimodal input the block +
        gallery are prepended as text blocks and the image blocks are preserved.
        """
        constraints = self._extract_hard_constraints(user_text)
        pin = ""
        if constraints:
            pin = (
                "[HARD CONSTRAINTS — the user was explicit; honor these exactly and do "
                "NOT substitute or omit them:]\n"
                + "\n".join(f"- {c}" for c in constraints)
                + "\n\n"
            )
            if self._verbose:
                print("pipeline: pinned hard constraints:\n" + "\n".join(f"  - {c}" for c in constraints))

        # Canvas-hook mode: prepend the directive block so the orchestrator runs
        # the on-canvas graph via apply_canvas_hooks (above the hard constraints).
        if self._canvas_base_prompt is not None and self._canvas_hooks:
            from src.utils.canvas_hooks import describe_hooks
            hooks_block = describe_hooks(self._canvas_hooks, self._canvas_base_prompt)
            if hooks_block:
                pin = hooks_block + "\n" + pin

        # Canvas selection: the nodes the user has selected on the graph, with
        # their current parameter values. Lets the orchestrator read a node ("read
        # this prompt") and write it back via set_canvas_node_params.
        sel_block = self._describe_canvas_selection()
        if sel_block:
            pin = pin + sel_block + "\n\n"

        if isinstance(user_input, list):
            gallery = self._format_image_gallery()
            blocks = list(user_input)
            prefix = pin + (gallery + "\n\n" if gallery else "")
            if prefix:
                blocks.insert(0, {"text": prefix})
            return blocks
        return pin + self._prepend_gallery(self._annotate_attachments(user_input, user_text))

    async def _astream_orchestrator(self, user_input, *, qa_reply_queue: asyncio.Queue | None = None,
                                    canvas_prompt: dict | None = None, canvas_hooks: list | None = None,
                                    canvas_selection: list | None = None):
        """Stream the orchestrator for one turn, then run any signalled workflow.

        This replaces the triage → route → handler block: the orchestrator owns
        the turn end-to-end. After it finishes (no ComfyUI interrupt pending), the
        workflow-signal mailbox is drained and the Executor runs exactly as in the
        legacy Brain stage — so ComfyUI submission / Vision-QA / output-staging is
        unchanged. ComfyUI interrupts are handled identically to the Brain stage.
        """
        self._last_turn_usages = []
        self._vision_usage_snap = self._usage_snapshot(self._vision_agent) if self._vision_agent else {}
        self._run_qa = False
        user_text = self._extract_text(user_input)
        # Register image paths embedded in a plain-text message so assembly/LoadImage
        # wiring receives real input paths (Chainlit-style callers set this already).
        if not isinstance(user_input, list):
            _imgs, _ = Pipeline._scan_media_paths(user_text)
            if _imgs:
                self._session.last_user_input_images = _imgs
        synth = TriageResult(intent=MessageIntent.new_request, response=None,
                             confidence=1.0, run_qa=False)

        # Canvas-hook mode: the user annotated their on-canvas graph. Splice the
        # hook nodes out of the captured API prompt and stash the clean base for
        # apply_canvas_hooks; describe the hooks in the orchestrator input.
        self._canvas_base_prompt = None
        self._canvas_hooks = [h for h in (canvas_hooks or []) if isinstance(h, dict)]
        self._chain_output_paths = []
        # Arbitrary selected nodes (id/type/title/widgets) the orchestrator can
        # read and write back via set_canvas_node_params.
        self._canvas_selection = [n for n in (canvas_selection or []) if isinstance(n, dict)]
        if isinstance(canvas_prompt, dict) and canvas_prompt:
            try:
                from src.utils.canvas_hooks import splice_hook_nodes
                cleaned, removed = splice_hook_nodes(canvas_prompt)
                self._canvas_base_prompt = cleaned
                if self._verbose:
                    print(f"pipeline: canvas-hook mode — {len(self._canvas_hooks)} hook(s), "
                          f"spliced {len(removed)} hook node(s); base graph has "
                          f"{len(cleaned)} node(s).")
            except Exception as exc:  # noqa: BLE001
                print(f"pipeline: canvas-hook splice failed ({exc}); ignoring canvas graph.")

        self._ensure_orch_clean_history()
        orch_input = self._build_orchestrator_input(user_input, user_text)
        current_input: Any = orch_input
        _snap = self._usage_snapshot(self._orchestrator_agent)
        # Drop any tool-activity / canvas-patch left over from a previous turn.
        _clear_tools()
        _clear_canvas_patch()

        while True:
            interrupt_result = None
            yield {"_orchestrator_start": True}
            async for event in self._orchestrator_agent.stream_async(current_input):
                yield event
                if "result" in event:
                    agent_result = event["result"]
                    if getattr(agent_result, "stop_reason", None) == "interrupt":
                        for intr in getattr(agent_result, "interrupts", []):
                            if getattr(intr, "name", None) == INTERRUPT_NAME:
                                interrupt_result = intr
                                break
                for _prog_line in _drain_progress():
                    yield {"data": _prog_line}
                # Surface the agent's tool calls + results inline in the chat.
                for _ta in _drain_tools():
                    yield {"tool_activity": _ta}
                # Push any node edits back to the live canvas.
                for _cp in _drain_canvas_patch():
                    yield {"canvas_patch": _cp}
            # Flush any tool activity / canvas edits emitted after the last event.
            for _ta in _drain_tools():
                yield {"tool_activity": _ta}
            for _cp in _drain_canvas_patch():
                yield {"canvas_patch": _cp}
            yield {"_orchestrator_done": True}

            if interrupt_result is None:
                # Executor handoff — drain the workflow-signal mailbox and run.
                workflow_paths = _get_workflow_signal()
                workflow_paths = self._expand_variations(workflow_paths, self._last_brainbriefing_json or "")
                # Reset this turn's outputs before the deferred batch, but KEEP any
                # produced mid-turn by run_workflow_now (chained stages) so they're
                # still staged. Non-chain turns have none, so this equals .clear().
                self._session.current_output_paths[:] = list(self._chain_output_paths)
                exec_paths = self._session.current_output_paths
                _qa_fail_event: dict | None = None
                if workflow_paths:
                    if self._verbose:
                        count = len(workflow_paths)
                        tag = f"{count} workflows (batch)" if count > 1 else workflow_paths[0]
                        print(f"pipeline: Orchestrator signaled {tag} ready.")
                    async for line in _execute_workflows_batch(
                        workflow_paths,
                        self._last_brainbriefing_json or "",
                        user_message=user_text,
                        verbose=self._verbose,
                        collected_paths=exec_paths,
                        run_qa=self._run_qa,
                    ):
                        if isinstance(line, dict) and line.get("qa_fail"):
                            _qa_fail_event = line
                            break
                        yield {"data": f"\n{line}"}
                        # Surface any tool calls the executor's agents make (QA,
                        # error-check) so they aren't stranded in the buffer.
                        for _ta in _drain_tools():
                            yield {"tool_activity": _ta}
                    # Flush executor-phase tool activity after the batch finishes.
                    for _ta in _drain_tools():
                        yield {"tool_activity": _ta}

                if _qa_fail_event:
                    if qa_reply_queue is not None:
                        yield {"qa_fail_ask": True, **_qa_fail_event}
                        _answer = await qa_reply_queue.get()
                        if _is_affirmative(_answer):
                            yield {"data": "\n\n_🔄 Retrying with QA feedback…_"}
                            current_input = self._build_qa_feedback_prompt(
                                user_text, user_text, _qa_fail_event
                            )
                            continue
                    self._record_chat_summary(user_text, synth, status="qa_failed",
                                              raw_json=self._last_brainbriefing_json)
                    self._record_agent_usage(self._orchestrator_agent, _snap)
                    self._session.last_agent = "orchestrator"
                    return

                self._record_chat_summary(user_text, synth, status="completed",
                                          raw_json=self._last_brainbriefing_json)
                self._record_agent_usage(self._orchestrator_agent, _snap)
                self._session.last_agent = "orchestrator"
                if self._verbose:
                    print("pipeline: Orchestrator finished.")
                return

            # ── ComfyUI interrupt: stream progress, then resume ────── #
            raw_reason = interrupt_result.reason or ""
            prompt_id_o = ""
            client_id_o = ""
            try:
                _r = json.loads(raw_reason)
                if isinstance(_r, dict):
                    prompt_id_o = str(_r.get("prompt_id", ""))
                    client_id_o = str(_r.get("client_id", "") or "")
                else:
                    prompt_id_o = str(_r)
            except Exception:
                prompt_id_o = raw_reason
            if self._verbose:
                print(f"pipeline: ComfyUI interrupt — streaming prompt_id={prompt_id_o}")
            yield {"data": f"\n\n_⏳ ComfyUI job queued (`{prompt_id_o}`). Streaming progress…_"}
            history_result_o: dict = {}
            async for ev in _stream_comfyui_job(prompt_id_o, client_id_o):
                if isinstance(ev, dict):
                    history_result_o = ev["history"] if "history" in ev else ev
                    break
                yield {"data": f"\n_{ev}_"}
            yield {"data": "\n_✅ ComfyUI job finished — resuming…_"}
            current_input = [
                {
                    "interruptResponse": {
                        "interruptId": interrupt_result.id,
                        "response": json.dumps(history_result_o),
                    }
                }
            ]

    async def stream_async(self, user_input, *, qa_reply_queue: asyncio.Queue | None = None,
                           canvas_prompt: dict | None = None, canvas_hooks: list | None = None,
                           canvas_selection: list | None = None):  # noqa: ANN201
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
        _trace("pipeline.stream_async: await pending compression")
        await self._await_pending_compression()

        # ── Free-agent mode: the orchestrator owns the whole turn. Skip triage
        # and the rigid router entirely. ─────────────────────────────────────
        if self._free_agent and self._orchestrator_agent is not None:
            _trace("pipeline.stream_async: orchestrator begin")
            async for event in self._astream_orchestrator(
                user_input, qa_reply_queue=qa_reply_queue,
                canvas_prompt=canvas_prompt, canvas_hooks=canvas_hooks,
                canvas_selection=canvas_selection,
            ):
                yield event
            _trace("pipeline.stream_async: orchestrator done")
            return

        # Stage 0 – Triage (classify intent before any agent is called)
        _trace("pipeline.stream_async: triage begin")
        self._last_turn_usages = []
        # Snapshot the Vision agent's usage so this turn's analyze_image tokens
        # can be attributed (and priced) at cost-finalisation time.
        self._vision_usage_snap = self._usage_snapshot(self._vision_agent) if self._vision_agent else {}
        user_text = self._extract_text(user_input)
        user_text = self._annotate_attachments(user_input, user_text)
        # Headless/CLI: register image paths embedded in the plain-text message so
        # downstream stages (brainbriefing, LoadImage wiring) receive real input
        # paths, mirroring Chainlit's attachment handling. Chainlit callers pass a
        # content-block list and set last_user_input_images themselves.
        if not isinstance(user_input, list):
            _cli_imgs, _ = Pipeline._scan_media_paths(user_text)
            if _cli_imgs:
                self._session.last_user_input_images = _cli_imgs
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
            print(f"pipeline: Triage → intent={triage_result.intent.value},"
                  f" confidence={triage_result.confidence:.2f}, handler={handler}")
        _trace(f"pipeline.stream_async: triage done → handler={handler}")

        # Context-dependent routing: query_templates was previously blocked waiting for user input.
        if self._session.last_agent == "query_templates" and self._session.last_researcher_request:
            if self._verbose:
                print("pipeline: Researcher was blocked — re-running with user clarification")
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
            async for event in self._astream_brain_stage(
                _r_json_s, user_text, triage_result, qa_reply_queue=qa_reply_queue
            ):
                yield event
            return

        if handler == "answer":
            if self._verbose:
                print("pipeline: info_query → Info agent (streamed)")
            _info_snap = self._usage_snapshot(self._info_agent)
            _info_chunks: list[str] = []
            _trace("pipeline.answer: info_agent.stream_async begin")
            _info_n = 0
            # Surface the thread's generated-image gallery so requests like
            # "analyse the second image" resolve to a real file the Info agent
            # can pass to analyze_image().
            _info_input = self._prepend_gallery(user_text)
            async for event in self._info_agent.stream_async(_info_input):
                _info_n += 1
                if isinstance(event, dict):
                    _chunk = event.get("data", "")
                    if _chunk:
                        _info_chunks.append(_chunk)
                yield event
            _trace(f"pipeline.answer: info stream done ({_info_n} events); bookkeeping")
            self._record_agent_usage(self._info_agent, _info_snap)
            _info_full_response = "".join(_info_chunks)
            log_agent_exchange("INFO", user_text, _info_full_response)
            self._session.last_agent = "info"
            self._session.last_info_response = _info_full_response or None
            self._record_chat_summary(user_text, triage_result, status="completed")
            _trace("pipeline.answer: return")
            return

        if handler == "story":
            if self._verbose:
                print("pipeline: story → Story agent (streamed)")
            async for event in self._stream_story(user_text, triage_result):
                yield event
            _trace("pipeline.story: return")
            return

        if handler == "storyboard":
            if self._verbose:
                print("pipeline: storyboard → Storyboard director (streamed)")
            async for event in self._stream_storyboard(
                user_text, triage_result, qa_reply_queue=qa_reply_queue
            ):
                yield event
            _trace("pipeline.storyboard: return")
            return

        if handler == "needs_image":
            if self._verbose:
                print("pipeline: needs_image → handoff to user (missing image)")
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
                _info_snap = self._usage_snapshot(self._info_agent)
                _info_fb_chunks: list[str] = []
                async for event in self._info_agent.stream_async(self._prepend_gallery(user_text)):
                    if isinstance(event, dict):
                        _chunk = event.get("data", "")
                        if _chunk:
                            _info_fb_chunks.append(_chunk)
                    yield event
                self._record_agent_usage(self._info_agent, _info_snap)
                log_agent_exchange("INFO", user_text, "".join(_info_fb_chunks))
                self._session.last_agent = "info"
                self._record_chat_summary(user_text, triage_result, status="completed")
                return
            # Context-dependent feedback routing: if the previous turn was the Story
            # agent (e.g. "make it darker", "shorter"), revise the story rather than
            # handing the feedback to the Brain, which has no knowledge of the story.
            if triage_result.intent == MessageIntent.feedback and self._session.last_agent == "story":
                if self._verbose:
                    print("pipeline: feedback on Story-agent output → routing back to Story agent")
                async for event in self._stream_story(user_text, triage_result):
                    yield event
                return
            # Follow-up: skip Researcher, send directly to Brain (streamed)
            self._ensure_clean_history()
            brain_prompt = self._build_followup_prompt(user_text, triage_result)
            # Surface the thread's generated-image gallery so feedback/param_tweak
            # follow-ups can resolve references like "image 2" / "the last image".
            _gallery_fu = self._format_image_gallery()
            if _gallery_fu:
                brain_prompt = f"{_gallery_fu}\n\n{brain_prompt}"
            self._session.follow_up_count += 1
            _brain_snap_fu = self._usage_snapshot(self._assemble_workflow)
            yield {"_brain_start": True}
            async for event in self._assemble_workflow.stream_async(brain_prompt):
                yield event
            yield {"_brain_done": True}
            self._record_agent_usage(self._assemble_workflow, _brain_snap_fu)
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
            self._session.last_agent = "assemble_workflow"
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
            yield {"data": researcher_output}
            return

        # Stage 2 – Brain (streamed, with optional ComfyUI interrupt handling)
        if self._verbose:
            print("pipeline: Stage 2 – Brain streaming …")
        async for event in self._astream_brain_stage(
            raw_json, user_text, triage_result, qa_reply_queue=qa_reply_queue
        ):
            yield event

    # ── Internal helpers ─────────────────────────────────────────────── #

    # ── Planner helpers ──────────────────────────────────────────────── #

    @staticmethod
    def _normalize_step_kind(raw: object) -> str:
        """Normalise a planner step's ``kind`` to analysis|writing|dop|generation.

        Unknown / missing values fall back to ``generation`` (the historical
        Researcher → Brain → Executor path), with light synonym handling so a
        slightly-off label from the planner still routes correctly.
        """
        k = str(raw or "generation").strip().lower()
        if k in {"analysis", "writing", "dop", "generation"}:
            return k
        if k.startswith("anal") or k in {"info", "describe", "description", "vision", "caption"}:
            return "analysis"
        if k in {"writing", "write", "story", "synopsis", "scene", "scenes",
                 "narrative", "screenplay", "text", "prose"}:
            return "writing"
        if k in {"dp", "cinematography", "cinematographer", "camera", "lighting",
                 "photography", "director of photography", "cinema"}:
            return "dop"
        return "generation"

    async def _run_planner(self, user_text: str) -> tuple[list[dict[str, str]], str]:
        """Call the Planner agent to decompose *user_text* into ordered steps.

        Returns a tuple of:
          - list of ``{"request": str, "description": str}`` dicts (empty on failure)
          - the raw agent response string (for display in the UI thinking block)

        Uses ``invoke_async`` (not the sync ``agent(...)``) so the persistent
        Planner agent runs on the caller's event loop instead of a throwaway
        per-call worker loop — see the triage note for why the sync path hangs
        on the second invocation.
        """
        raw: str
        _planner_snap = self._usage_snapshot(self._planner_agent)
        raw = str(await self._planner_agent.invoke_async(user_text))
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
                        "kind": self._normalize_step_kind(s.get("kind", "generation")),
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

    async def _run_error_check(self, task_description: str) -> dict:
        """Invoke the error-checker agent and return a parsed verdict dict.

        The agent reads ComfyUI logs and returns JSON with keys:
        ``status`` (ok | error_fixable | error_unfixable), ``errors``,
        ``fix_plan``, ``user_message``.

        On any failure the method returns ``{"status": "ok", ...}``
        (fail-open) so a transient error doesn't abort the plan.

        Uses ``invoke_async`` (not the sync ``agent(...)``) for the same reason
        as triage/planner: the persistent agent must run on the caller's event
        loop, not a throwaway per-call worker loop.
        """
        _snap = self._usage_snapshot(self._error_checker_agent)
        raw = ""
        try:
            raw = str(await self._error_checker_agent.invoke_async(task_description))
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

    async def _stream_plan_text_step(
        self,
        agent,
        agent_label: str,
        step_request: str,
        prev_text: str | None,
        *,
        with_gallery: bool,
    ):
        """Run one non-generation plan step (Info or Story agent) statelessly.

        Streams the agent's events, forwards the previous step's TEXT result into
        the prompt (so e.g. the synopsis step sees the image analysis, and the
        scene step sees the synopsis), optionally surfaces the image gallery
        (analysis steps only, so "image 4" resolves), and finally emits a
        ``{"_plan_step_text": <text>}`` sentinel that the planner loop captures —
        it is NOT forwarded to the UI — to hand the text to the next step.

        The agent's own history is cleared before and after so plan steps stay
        stateless (token-lean) and never bleed across runs.
        """
        try:
            agent.messages.clear()
        except Exception:  # noqa: BLE001
            pass

        parts: list[str] = []
        if with_gallery:
            _g = self._format_image_gallery()
            if _g:
                parts.append(_g)
        if prev_text:
            parts.append(
                "[Result from the previous step — use it as the input/source for this step:]\n"
                + prev_text[: self._STORY_CONTEXT_CHAR_CAP]
            )
        parts.append(step_request)
        _input = "\n\n".join(parts)

        _snap = self._usage_snapshot(agent)
        _chunks: list[str] = []
        async for event in agent.stream_async(_input):
            if isinstance(event, dict):
                _c = event.get("data", "")
                if _c:
                    _chunks.append(_c)
            yield event
        self._record_agent_usage(agent, _snap)
        _text = "".join(_chunks)
        log_agent_exchange(agent_label, step_request, _text)
        try:
            agent.messages.clear()
        except Exception:  # noqa: BLE001
            pass
        yield {"_plan_step_text": _text}

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
        yield {"_planner_start": True}
        steps, _planner_raw = await self._run_planner(user_text)
        yield {"_planner_done": True, "raw": _planner_raw}

        # Fallback: treat as a plain researcher path when planning fails.
        if not steps:
            if self._verbose:
                print("pipeline: Planner fallback → researcher path")
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
            _brain_snap_pfb = self._usage_snapshot(self._assemble_workflow)
            async for event in self._assemble_workflow.stream_async(self._build_brain_prompt(raw_json)):
                yield event
            self._record_agent_usage(self._assemble_workflow, _brain_snap_pfb)
            wf = _get_workflow_signal()
            wf = self._expand_variations(wf, raw_json)
            self._session.current_output_paths.clear()
            ep = self._session.current_output_paths
            if wf:
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
        if self._verbose:
            print(f"pipeline: Plan has {total} step(s):")
            for i, s in enumerate(steps, 1):
                print(f"  {i}. {s['description']}")

        prev_brainbriefing: dict | None = None  # compact context forwarded step-to-step
        prev_text_output: str | None = None     # TEXT result forwarded between steps

        for idx, step in enumerate(steps):
            description = step["description"]
            kind = self._normalize_step_kind(step.get("kind", "generation"))

            yield {"_step_start": True, "idx": idx, "total": total, "description": description}
            yield {"data": f"\n\n**Step {idx + 1}/{total} — {description}**\n"}
            if self._verbose:
                print(f"\npipeline: ── Plan step {idx + 1}/{total} ({kind}): {description} ──")

            # Non-generation steps (image analysis / creative writing / DoP
            # cinematography) must NOT go to the Researcher (which only emits ComfyUI
            # workflow briefs). Route them to the Info / Story / DoP agent and forward
            # their TEXT to the next step. The DoP agent reads the previous step's
            # storyboard/prompt and rewrites it with concrete camera/light/colour.
            if kind in ("analysis", "writing", "dop"):
                if kind == "analysis":
                    _agent, _label = self._info_agent, "INFO"
                elif kind == "writing":
                    _agent, _label = self._story_agent, "STORY"
                else:
                    _agent, _label = self._dop_agent, "DOP"
                _step_text = ""
                async for _tev in self._stream_plan_text_step(
                    _agent, _label, step["request"], prev_text_output,
                    with_gallery=(kind == "analysis"),
                ):
                    if isinstance(_tev, dict) and "_plan_step_text" in _tev:
                        _step_text = _tev["_plan_step_text"]
                    else:
                        yield _tev
                prev_text_output = _step_text or prev_text_output
                # Keep the per-agent caches coherent so a later non-plan follow-up
                # (e.g. "make it scarier") can build on the latest output.
                if kind == "analysis":
                    self._session.last_info_response = _step_text or None
                    self._session.last_agent = "info"
                else:
                    # writing + dop both produce text the next step / a follow-up
                    # can build on; treat them as the latest "story" text output.
                    self._session.last_story_response = _step_text or self._session.last_story_response
                    self._session.last_agent = "story"
                self._record_chat_summary(step["request"], triage_result, status="completed")
                yield {"_step_done": True, "idx": idx}
                if self._verbose:
                    print(f"pipeline: Step {idx + 1}/{total} ({kind}) finished.")
                continue

            # Generation step — Researcher → Brain → Executor.
            step_req = self._inject_context_into_step(step["request"], idx, prev_brainbriefing)
            if prev_text_output:
                step_req += (
                    "\n\n[Use the text produced by the previous step as the creative "
                    "source for this generation (e.g. the scene / shot descriptions):]\n"
                    f"{prev_text_output[: self._STORY_CONTEXT_CHAR_CAP]}"
                )

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
                _brain_snap_ps = self._usage_snapshot(self._assemble_workflow)
                self._ensure_clean_history()
                async for event in self._assemble_workflow.stream_async(_brain_prompt_for_step):
                    yield event
                self._record_agent_usage(self._assemble_workflow, _brain_snap_ps)

                wf_paths = _get_workflow_signal()
                wf_paths = self._expand_variations(wf_paths, raw_json)
                self._session.current_output_paths.clear()
                exec_paths = self._session.current_output_paths
                step_error_abort = False
                verdict: dict = {"status": "ok", "errors": [], "fix_plan": "", "user_message": ""}
                _qa_fail_event: dict | None = None
                if wf_paths:
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
                            verdict = await self._run_error_check(description)
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
                            # Fixable error — ask Brain to fix (streamed) and rerun.
                            yield {"data": f"\n_🔧 Fixing step {idx + 1} (attempt {_fix_attempt}/{_MAX_FIX_ATTEMPTS})…_"}
                            _brain_snap_fix = self._usage_snapshot(self._assemble_workflow)
                            async for event in self._brain.stream_async(
                                self._build_fix_prompt(verdict["fix_plan"], _fix_attempt)
                            ):
                                yield event
                            self._record_agent_usage(self._assemble_workflow, _brain_snap_fix)
                            wf_paths_fix = _get_workflow_signal()
                            wf_paths_fix = self._expand_variations(wf_paths_fix, raw_json)
                            # Reset for the fix run so the chainlit "already-sent"
                            # tracker treats fixed outputs as fresh.
                            self._session.current_output_paths.clear()
                            exec_paths = self._session.current_output_paths
                            if wf_paths_fix:
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
                            yield {"data": "\n\n_🔄 Retrying step with QA feedback…_"}
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

        if self._verbose:
            print(f"pipeline: Planned execution complete ({total} step(s)).")

    # ── Storyboard director (Option B — dynamic short-film orchestrator) ── #

    # Default bound on Vision-QA attempts per visual step before asking the user.
    # Overridable via settings.json ``director.auto_retries`` or the
    # ``DIRECTOR_AUTO_RETRIES`` env var; resolved per-instance into
    # ``self._storyboard_max_qa`` at construction time.
    _STORYBOARD_MAX_QA = 4
    # Free-text replies accepted at an approval gate as "approve" / "abort".
    _STORYBOARD_AFFIRM = {
        "approve", "approved", "ok", "okay", "proceed", "continue", "next",
        "yes", "y", "good", "looks good", "lgtm", "go", "accept", "fine",
    }
    _STORYBOARD_ABORT = {"abort", "stop", "cancel", "quit", "no", "nah", "halt", "end"}

    # Scope guards prepended to every per-step Researcher request. Without these,
    # the persistent style guidelines (which may mention "short film"/"trailer")
    # tempt the Researcher to treat a single image/video step as a multi-stage
    # project — wasting tokens deliberating, emitting spurious WARNING blockers,
    # or (worse) blocking for clarification and aborting the step.
    _SB_SCOPE_IMG = (
        "IMPORTANT — SCOPE: This is a SINGLE image-generation task. Produce ONLY {asset}. "
        "The wider short film is produced by an orchestrator across separate steps; do NOT "
        "treat the storyline/film as part of THIS task, do NOT plan extra steps, do NOT "
        "select a video template, and do NOT block for clarification.\n\n"
    )
    _SB_SCOPE_VID = (
        "IMPORTANT — SCOPE: This is a SINGLE video-generation task. Produce ONLY this one "
        "multi-shot sequence. Other sequences are produced separately by an orchestrator; "
        "do NOT plan extra steps and do NOT block for clarification.\n\n"
    )

    # The character sheet is a production REFERENCE asset, not film footage — so it
    # is generated and QA'd with a clean look, never the film's grade (grain / VHS /
    # found-footage / colour grading). This honours instructions like "the grungy
    # look doesn't need to apply to the character sheet" by default; the footage
    # ``style_guidelines`` are applied only to the start frames and videos.
    _SB_CHARSHEET_GUIDELINES = (
        "Clean character reference sheet: even, neutral studio lighting; plain, uncluttered "
        "background; sharp focus; natural colours true to the reference. This is a production "
        "reference asset — do NOT apply any film grade, film grain, VHS / found-footage "
        "artifacts, or stylisation; the film's visual style applies only to the footage, "
        "not to this sheet."
    )

    @classmethod
    def _resolve_director_retries(cls) -> int:
        """Resolve the storyboard QA auto-retry bound.

        Priority: ``DIRECTOR_AUTO_RETRIES`` env var > settings.json
        ``director.auto_retries`` > the ``_STORYBOARD_MAX_QA`` default. Always
        returns at least 1 (one attempt) so a misconfigured 0/negative value
        can't disable generation entirely.
        """
        env = os.environ.get("DIRECTOR_AUTO_RETRIES")
        if env is not None:
            try:
                return max(1, int(env))
            except ValueError:
                pass
        try:
            val = _settings().get("director", {}).get("auto_retries", cls._STORYBOARD_MAX_QA)
            return max(1, int(val))
        except Exception:  # noqa: BLE001
            return cls._STORYBOARD_MAX_QA

    @staticmethod
    def _coerce_bool(value: object) -> bool | None:
        """Coerce a settings/env value to bool, or None if not recognisable."""
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return bool(value)
        if isinstance(value, str):
            s = value.strip().lower()
            if s in {"1", "true", "yes", "on"}:
                return True
            if s in {"0", "false", "no", "off"}:
                return False
        return None

    @classmethod
    def _resolve_director_approval(cls) -> bool:
        """Resolve the DEFAULT for the storyboard per-step approval gate.

        Returns True when the director should pause for user approval after each
        step, False to run autonomously. Priority: ``DIRECTOR_USER_APPROVAL_STEP``
        env var > settings.json ``director.user_approval_step`` > default True.
        The user's message can still override this per-run (see
        :meth:`_stream_storyboard`).
        """
        env = os.environ.get("DIRECTOR_USER_APPROVAL_STEP")
        if env is not None:
            c = cls._coerce_bool(env)
            if c is not None:
                return c
        try:
            c = cls._coerce_bool(_settings().get("director", {}).get("user_approval_step", True))
            if c is not None:
                return c
        except Exception:  # noqa: BLE001
            pass
        return True

    @classmethod
    def _resolve_director_cinematography(cls) -> bool:
        """Resolve the DEFAULT for the storyboard DoP cinematography pass.

        Returns True when the director should run the DoP agent over the finished
        storyboard (rewriting every start frame + shot with concrete lighting /
        composition / camera / colour) before generation. Priority:
        ``DIRECTOR_APPLY_CINEMATOGRAPHY`` env var > settings.json
        ``director.apply_cinematography`` > default True. The user's message can
        still override this per-run (see :meth:`_stream_storyboard`).
        """
        env = os.environ.get("DIRECTOR_APPLY_CINEMATOGRAPHY")
        if env is not None:
            c = cls._coerce_bool(env)
            if c is not None:
                return c
        try:
            c = cls._coerce_bool(_settings().get("director", {}).get("apply_cinematography", True))
            if c is not None:
                return c
        except Exception:  # noqa: BLE001
            pass
        return True

    @staticmethod
    def _storyboard_skip_cinematography(user_text: str) -> bool:
        """Return True when the user explicitly opts out of the DoP pass."""
        t = user_text.lower()
        triggers = (
            "skip cinematography", "no cinematography", "without cinematography",
            "skip the dop", "no dop", "without the dop", "skip camera rules",
            "don't apply cinematography", "dont apply cinematography",
            "do not apply cinematography", "skip the camera pass",
            "no camera pass", "don't apply the camera", "skip dop",
        )
        return any(tok in t for tok in triggers)

    @staticmethod
    def _storyboard_wants_approval(user_text: str) -> bool:
        """Return True when the user explicitly asks to approve each step.

        Overrides a ``user_approval_step: false`` default so a user who asks for
        approvals gets them even when the director defaults to autonomous.
        """
        t = user_text.lower()
        triggers = (
            "ask for approval", "ask me for approval", "ask for my approval",
            "with approval", "with my approval", "approve each", "approval after each",
            "let me approve", "let me review each", "review each step",
            "pause for approval", "wait for my approval", "ask me before",
            "ask before each", "check with me", "i want to approve", "approval step",
        )
        return any(tok in t for tok in triggers)

    @staticmethod
    def _storyboard_auto_approve(user_text: str) -> bool:
        """Return True when the user opted out of per-step approval gating."""
        t = user_text.lower()
        triggers = (
            "without asking", "don't ask", "dont ask", "do not ask", "no approval",
            "without approval", "auto approve", "auto-approve", "automatically",
            "fully automatic", "no need to approve", "don't stop for approval",
            "without my approval", "no approvals", "skip approval",
        )
        return any(tok in t for tok in triggers)

    @staticmethod
    def _storyboard_wants_references(user_text: str) -> bool:
        """Return True when the user explicitly asks the agent to look up references.

        Gates the web Reference Search Web — we only search/download when the user asks
        for it (e.g. "find a reference for a 1950s diner", "look up what X looks
        like"), never proactively.
        """
        t = user_text.lower()
        triggers = (
            "find a reference", "find references", "look for a reference",
            "look for references", "search the web", "search for a reference",
            "reference image", "reference images", "look up", "find an image of",
            "find images of", "search for images", "use a reference from the web",
            "find a real", "look up what", "research what", "find pictures of",
            "web reference", "find a picture of", "get a reference",
        )
        return any(tok in t for tok in triggers)

    @staticmethod
    def _parse_storyboard_spec(text: str) -> dict | None:
        """Extract and lightly validate the storyboard JSON spec from Story output.

        Expects a trailing fenced JSON block with a ``characters`` list, a
        ``guidelines`` string and a non-empty ``sequences`` list (each with
        ``shots`` and ``character_tags``). Back-compatible with the old single
        ``character`` field. Returns the normalised dict, or ``None`` when no
        usable spec can be recovered.

        Normalised shape:
          - ``characters``: list of ``{tag, description, shot_count}``
          - ``sequences``: list of ``{index, summary, start_frame_prompt,
            character_tags, shots:[{prompt, duration}]}``
        """
        raw = _extract_json(text or "")
        if not raw:
            return None
        try:
            spec = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            return None
        if not isinstance(spec, dict):
            return None
        seqs = spec.get("sequences")
        if not isinstance(seqs, list) or not seqs:
            return None

        # ── Characters: prefer the new list; fall back to the old singular field ──
        chars_raw = spec.get("characters")
        norm_chars: list[dict] = []
        if isinstance(chars_raw, list):
            for c in chars_raw:
                if isinstance(c, dict) and c.get("tag") and c.get("description"):
                    try:
                        sc = int(c.get("shot_count", 0) or 0)
                    except (TypeError, ValueError):
                        sc = 0
                    norm_chars.append({
                        "tag": str(c["tag"]).strip(),
                        "description": str(c["description"]).strip(),
                        "shot_count": sc,
                    })
        elif isinstance(spec.get("character"), dict) and spec["character"].get("present"):
            c = spec["character"]
            norm_chars.append({
                "tag": str(c.get("tag") or "MAIN").strip(),
                "description": str(c.get("description") or "").strip(),
                "shot_count": 0,
            })

        valid_tags = {c["tag"] for c in norm_chars}

        # ── Sequences ────────────────────────────────────────────────────────
        norm_seqs: list[dict] = []
        for i, s in enumerate(seqs, 1):
            if not isinstance(s, dict):
                continue
            shots = s.get("shots") if isinstance(s.get("shots"), list) else []
            clean_shots = []
            for sh in shots:
                if isinstance(sh, dict) and sh.get("prompt"):
                    try:
                        dur = int(sh.get("duration", 5) or 5)
                    except (TypeError, ValueError):
                        dur = 5
                    clean_shots.append({"prompt": str(sh["prompt"]), "duration": max(1, dur)})
            if not clean_shots:
                continue
            tags_raw = s.get("character_tags")
            if isinstance(tags_raw, list):
                seq_tags = [str(t).strip() for t in tags_raw if str(t).strip() in valid_tags]
            else:
                seq_tags = list(valid_tags)  # default: assume all characters present
            norm_seqs.append({
                "index": int(s.get("index", i) or i),
                "summary": str(s.get("summary", f"Sequence {i}")),
                "start_frame_prompt": str(s.get("start_frame_prompt", "")).strip(),
                "character_tags": seq_tags,
                "shots": clean_shots,
            })
        if not norm_seqs:
            return None

        # ── Derive missing shot_counts from sequence appearances ─────────────
        # When the Story agent didn't supply shot_count, approximate it as the
        # number of shots in the sequences that list the character.
        for ch in norm_chars:
            if ch["shot_count"] <= 0:
                ch["shot_count"] = sum(
                    len(sq["shots"]) for sq in norm_seqs if ch["tag"] in sq["character_tags"]
                )

        spec["characters"] = norm_chars
        spec["sequences"] = norm_seqs
        return spec

    async def _storyboard_story(self, user_text: str, ref_desc: str, *, reminder: str = ""):
        """Stream the Story agent to produce the bible + ≤10s sequence breakdown.

        Drives the Story agent in **Mode C** (the ``story-storyboard`` skill),
        which holds the authoring rules and the trailing JSON contract this
        director parses.  Yields the agent's events plus a final
        ``{"_sb_story": <full_text>}`` sentinel.

        ``reminder`` is appended on retry to push the agent to emit the JSON block.
        """
        self._clear_story_history()
        instruction = textwrap.dedent(f"""
            Produce a SHORT-FILM storyboard breakdown. Activate the `story-storyboard`
            skill (Mode C) and follow it exactly: write the story bible and prose
            breakdown, splitting the WHOLE story (start to finish) into Kling multi-shot
            sequences of <=10s each, then end your reply with the SINGLE trailing
            ```json block defined by that skill — and nothing after it.

            If the brief states an explicit total video length and/or total shot
            count, treat them as hard targets: derive per-shot duration from them
            (e.g. 10s / 5 shots = five 2s shots) and pack the shots into the FEWEST
            sequences allowed (<=6 shots AND <=10s each) — do not pad with extra
            sequences or inflate the total to fit a default 5s/shot.

            User brief / storyline / quality guidelines:
            {user_text}
        """).strip()
        if ref_desc:
            instruction += (
                f"\n\nReference character/style description (from the user's reference "
                f"image(s) — keep the character consistent with this):\n{ref_desc}"
            )
        if reminder:
            instruction += f"\n\n{reminder}"

        _snap = self._usage_snapshot(self._story_agent)
        chunks: list[str] = []
        yield {"_story_start": True, "name": "📝 Storyboard breakdown"}
        async for event in self._story_agent.stream_async(instruction):
            if isinstance(event, dict):
                c = event.get("data", "")
                if c:
                    chunks.append(c)
            yield event
        yield {"_story_done": True}
        self._record_agent_usage(self._story_agent, _snap)
        text = "".join(chunks)
        log_agent_exchange("STORY", "[storyboard breakdown]", text)
        self._clear_story_history()
        yield {"_sb_story": text}

    async def _storyboard_apply_dop(self, spec: dict):
        """Run the DoP agent over the finished storyboard to enrich cinematography.

        Sends the whole finalized breakdown (guidelines, character locks, every
        start frame and shot) to the DoP agent, which applies concrete lighting /
        composition / camera-movement / colour decisions ACROSS THE WHOLE FILM (so
        the colour arc is coherent) and returns the SAME JSON schema with each
        ``start_frame_prompt`` and shot ``prompt`` rewritten.  Merges the rewrites
        back into *spec* in place (preserving indices, character_tags, durations
        and the verbatim character locks).  Yields the agent's events plus a final
        ``{"_sb_dop_applied": <bool>}`` sentinel.  Any parse failure is non-fatal:
        the spec is left unchanged and ``False`` is reported.
        """
        payload = {
            "guidelines": spec.get("guidelines", ""),
            "characters": [
                {"tag": c.get("tag", ""), "description": c.get("description", "")}
                for c in spec.get("characters", [])
            ],
            "sequences": [
                {
                    "index": s.get("index"),
                    "summary": s.get("summary", ""),
                    "character_tags": s.get("character_tags", []),
                    "start_frame_prompt": s.get("start_frame_prompt", ""),
                    "shots": [
                        {"prompt": sh.get("prompt", ""), "duration": sh.get("duration", 5)}
                        for sh in s.get("shots", [])
                    ],
                }
                for s in spec.get("sequences", [])
            ],
        }
        instruction = (
            "Apply your cinematography rules (Storyboard mode) to this FINISHED "
            "storyboard. Return ONE ```json block with the SAME schema — keep every "
            "character lock, tag, sequence index, summary, character_tags and shot "
            "duration UNCHANGED; rewrite only `guidelines`, each `start_frame_prompt` "
            "and each shot `prompt`, weaving in concrete lighting, composition, camera "
            "movement and colour, coherent across the whole film. Keep each shot prompt "
            "<=480 characters. Output the JSON only.\n\nStoryboard JSON:\n```json\n"
            + json.dumps(payload, ensure_ascii=False, indent=2)
            + "\n```"
        )
        try:
            self._dop_agent.messages.clear()
        except Exception:  # noqa: BLE001
            pass
        _snap = self._usage_snapshot(self._dop_agent)
        chunks: list[str] = []
        yield {"_story_start": True, "name": "🎥 Cinematography (DoP)"}
        async for event in self._dop_agent.stream_async(instruction):
            if isinstance(event, dict):
                c = event.get("data", "")
                if c:
                    chunks.append(c)
            yield event
        yield {"_story_done": True}
        self._record_agent_usage(self._dop_agent, _snap)
        raw = "".join(chunks)
        log_agent_exchange("DOP", "[storyboard cinematography]", raw)
        try:
            self._dop_agent.messages.clear()
        except Exception:  # noqa: BLE001
            pass
        yield {"_sb_dop_applied": self._merge_dop_spec(spec, raw)}

    @staticmethod
    def _merge_dop_spec(spec: dict, raw: str) -> bool:
        """Merge DoP-rewritten prompts from *raw* JSON back into *spec* in place.

        Only ``guidelines``, each sequence's ``start_frame_prompt`` and each shot's
        ``prompt`` are taken from the DoP output; sequences are matched by ``index``
        and shots by position, so indices, tags, durations and the verbatim
        character locks are never disturbed.  Returns ``True`` when at least one
        field was updated, ``False`` when nothing usable parsed (spec unchanged).
        """
        parsed_raw = _extract_json(raw or "")
        if not parsed_raw:
            return False
        try:
            enriched = json.loads(parsed_raw)
        except (json.JSONDecodeError, TypeError):
            return False
        if not isinstance(enriched, dict):
            return False
        e_seqs = enriched.get("sequences")
        if not isinstance(e_seqs, list):
            return False

        updated = False
        # Enriched guidelines (base palette + film look) — adopt when non-empty.
        e_guidelines = enriched.get("guidelines")
        if isinstance(e_guidelines, str) and e_guidelines.strip():
            spec["guidelines"] = e_guidelines.strip()
            updated = True

        by_index = {
            es.get("index"): es
            for es in e_seqs
            if isinstance(es, dict) and es.get("index") is not None
        }
        for s in spec.get("sequences", []):
            es = by_index.get(s.get("index"))
            if not isinstance(es, dict):
                continue
            e_sf = es.get("start_frame_prompt")
            if isinstance(e_sf, str) and e_sf.strip():
                s["start_frame_prompt"] = e_sf.strip()
                updated = True
            e_shots = es.get("shots")
            if isinstance(e_shots, list):
                for orig_shot, e_shot in zip(s.get("shots", []), e_shots):
                    if isinstance(e_shot, dict):
                        e_p = e_shot.get("prompt")
                        if isinstance(e_p, str) and e_p.strip():
                            orig_shot["prompt"] = e_p.strip()
                            updated = True
        return updated

    async def _stream_search_web(self, user_text: str):
        """Run the Reference Search Web to find + stage web references for *user_text*.

        Streams the scout's events inside a UI bracket and yields a final
        ``{"_scout_manifest": <dict>}`` sentinel with the parsed manifest
        (``{"references": [{query, mode, path?, name?, subfolder?, description}]}``).
        """
        try:
            self._search_web_agent.messages.clear()
        except Exception:  # noqa: BLE001
            pass
        prompt = (
            "Find the visual reference(s) the user asks for below, stage the best "
            "candidate(s) as files with download_image, and return ONLY the JSON "
            "manifest exactly as specified in your instructions.\n\n"
            f"User request:\n{user_text}"
        )
        _snap = self._usage_snapshot(self._search_web_agent)
        chunks: list[str] = []
        yield {"_story_start": True, "name": "🌐 Reference scout"}
        async for event in self._search_web_agent.stream_async(prompt):
            if isinstance(event, dict):
                c = event.get("data", "")
                if c:
                    chunks.append(c)
            yield event
        yield {"_story_done": True}
        self._record_agent_usage(self._search_web_agent, _snap)
        raw = "".join(chunks)
        log_agent_exchange("SCOUT", user_text, raw)
        manifest: dict = {}
        try:
            parsed = json.loads(_extract_json(raw) or raw)
            if isinstance(parsed, dict):
                manifest = parsed
        except Exception:  # noqa: BLE001
            manifest = {}
        try:
            self._search_web_agent.messages.clear()
        except Exception:  # noqa: BLE001
            pass
        yield {"_scout_manifest": manifest}

    async def _storyboard_gen_step(
        self,
        *,
        request: str,
        qa_brief: str,
        guidelines: str,
        reference_paths: list[str],
        qa_reply_queue: asyncio.Queue | None,
    ):
        """Run one visual generation sub-step (Researcher→Brain→Executor) with
        guideline/reference-grounded Vision QA and bounded auto-retry.

        ``request`` is the full task-specific instruction sent to the Researcher.
        ``qa_brief`` is a concise statement of what THIS step must produce — it is
        the ground-truth reference the Vision QA judges the output against (so the
        character-sheet step is judged as a character sheet, not against the
        original video brief).  ``guidelines`` is the persistent style/quality bar
        that applies to every visual step.

        Yields the underlying Strands events plus a final
        ``{"_sb_result": {"success", "output_paths", "raw_json"}}`` sentinel.
        Vision QA failures are auto-retried up to ``self._storyboard_max_qa`` times
        (feeding the QA verdicts back to the Researcher); once exhausted the user
        is asked (via ``qa_reply_queue``) whether to keep trying.
        """
        ref_path_objs = [Path(p) for p in reference_paths if p]
        attempt = 0
        qa_feedback: str | None = None
        success = False
        output_paths: list[str] = []
        raw_json: str | None = None

        while True:
            attempt += 1
            req = request
            if qa_feedback:
                req = (
                    f"{request}\n\n[The previous attempt FAILED Vision QA. Fix exactly these "
                    f"issues while keeping the character and style consistent:]\n{qa_feedback}"
                )

            # 1. Researcher → brainbriefing
            yield {"_researcher_start": True}
            raw_json = None
            r_error: str | None = None
            async for ev in self._arun_researcher(req):
                if isinstance(ev, dict) and "_researcher_done" in ev:
                    raw_json = ev["raw_json"]
                    r_error = ev["error"]
                    yield {"_researcher_done": True}
                else:
                    yield ev
            if r_error:
                yield {"data": f"\n❌ {r_error}"}
                break
            blocked = self._researcher_blocked_question(raw_json)
            if blocked:
                yield {"data": f"\n🚫 {blocked}"}
                break
            self._last_brainbriefing_json = raw_json

            # 2. Brain → workflow assembly
            self._ensure_clean_history()
            self._assemble_workflow.messages.clear()
            _bsnap = self._usage_snapshot(self._assemble_workflow)
            yield {"_brain_start": True}
            async for ev in self._assemble_workflow.stream_async(self._build_brain_prompt(raw_json)):
                yield ev
            yield {"_brain_done": True}
            self._record_agent_usage(self._assemble_workflow, _bsnap)

            # 3. Executor + grounded Vision QA
            wf_paths = _get_workflow_signal()
            wf_paths = self._expand_variations(wf_paths, raw_json)
            self._session.current_output_paths.clear()
            exec_paths = self._session.current_output_paths
            if not wf_paths:
                qa_feedback = "The Brain did not assemble a workflow (signal_workflow_ready was never called)."
                if attempt < self._storyboard_max_qa:
                    yield {"data": f"\n\n_🔁 No workflow produced (attempt {attempt}/{self._storyboard_max_qa}). Retrying…_"}
                    continue
                break

            qa_fail_event: dict | None = None
            yield {"_executor_start": True}
            async for line in _execute_workflows_batch(
                wf_paths,
                raw_json,
                user_message=qa_brief,
                verbose=self._verbose,
                collected_paths=exec_paths,
                run_qa=True,
                guidelines=guidelines,
                reference_image_paths=ref_path_objs or None,
            ):
                if isinstance(line, dict) and line.get("qa_fail"):
                    qa_fail_event = line
                    break
                yield {"data": f"\n{line}"}
            yield {"_executor_done": True}
            output_paths = list(exec_paths)

            # Keep brain token usage bounded between sub-steps.
            await self._compress_brain_history(extra_output_paths=output_paths)

            if not qa_fail_event:
                success = True
                break

            # ── Vision QA failed ─────────────────────────────────────────── #
            details = qa_fail_event.get("fail_details", [])
            qa_feedback = "\n".join(
                f"- {Path(d['path']).name}: {d['verdict']}" for d in details
            ) or "Vision QA failed."
            if attempt < self._storyboard_max_qa:
                yield {"data": f"\n\n_🔁 Vision QA failed (attempt {attempt}/{self._storyboard_max_qa}). Auto-retrying with feedback…_"}
                continue

            # Exhausted auto-retries — ask the user whether to keep trying.
            if qa_reply_queue is not None:
                yield {"qa_fail_ask": True, **qa_fail_event}
                ans = (await qa_reply_queue.get() or "").strip()
                if _is_affirmative(ans) or ans.lower() in self._STORYBOARD_AFFIRM:
                    attempt = 0  # grant a fresh round of auto-retries
                    yield {"data": "\n\n_🔄 Retrying this step…_"}
                    continue
            success = False
            break

        if success:
            self._register_generated_images(raw_json)
        yield {"_sb_result": {"success": success, "output_paths": output_paths, "raw_json": raw_json}}

    async def _storyboard_step(
        self,
        *,
        idx: int,
        total: int,
        label: str,
        request: str,
        qa_brief: str,
        guidelines: str,
        reference_paths: list[str],
        auto_approve: bool,
        qa_reply_queue: asyncio.Queue | None,
    ):
        """Run one storyboard step (generation + QA) behind an approval gate.

        Emits ``_step_start`` / ``_step_done`` so the UI task list updates, runs
        the generation sub-step, then — unless ``auto_approve`` — pauses for user
        approval.  On a non-approve / non-abort reply the user's text is treated
        as a revision note and the step is re-run.  Yields events plus a final
        ``{"_sb_done": {"success", "output_paths", "raw_json", "aborted"}}``
        sentinel.
        """
        yield {"_step_start": True, "idx": idx, "total": total, "description": label}
        yield {"data": f"\n\n**Step {idx + 1}/{total} — {label}**\n"}

        feedback: str | None = None
        while True:
            req = request if not feedback else f"{request}\n\n[User revision note for this step:]\n{feedback}"
            result: dict | None = None
            async for ev in self._storyboard_gen_step(
                request=req,
                qa_brief=qa_brief,
                guidelines=guidelines,
                reference_paths=reference_paths,
                qa_reply_queue=qa_reply_queue,
            ):
                if isinstance(ev, dict) and "_sb_result" in ev:
                    result = ev["_sb_result"]
                else:
                    yield ev

            if not result or not result["success"]:
                yield {"_step_done": True, "idx": idx, "failed": True}
                yield {"_sb_done": {**(result or {"output_paths": [], "raw_json": None}), "success": False, "aborted": False}}
                return

            if auto_approve or qa_reply_queue is None:
                yield {"_step_done": True, "idx": idx}
                yield {"_sb_done": {**result, "aborted": False}}
                return

            # ── Approval gate ────────────────────────────────────────────── #
            yield {
                "approval_ask": True,
                "label": label,
                "description": f"Step {idx + 1}/{total} — {label}",
                "image_paths": result["output_paths"],
            }
            decision = (await qa_reply_queue.get() or "").strip()
            low = decision.lower()
            if not decision or _is_affirmative(low) or low in self._STORYBOARD_AFFIRM:
                yield {"_step_done": True, "idx": idx}
                yield {"_sb_done": {**result, "aborted": False}}
                return
            if low in self._STORYBOARD_ABORT:
                yield {"_step_done": True, "idx": idx, "failed": True}
                yield {"_sb_done": {**result, "success": False, "aborted": True}}
                return
            # Otherwise: treat the reply as a revision note and re-run the step.
            feedback = decision
            yield {"data": f"\n\n_🔄 Revising **{label}** per your note…_"}

    @staticmethod
    def _first_image(paths: list[str]) -> str | None:
        """Return the first image file path in *paths*, or None."""
        for p in paths:
            if _is_image_file(p):
                return p
        return None

    async def _stream_storyboard(
        self,
        user_text: str,
        triage_result: TriageResult,
        *,
        qa_reply_queue: asyncio.Queue | None = None,
    ):
        """Option B orchestrator: storyline → character sheet → per-sequence
        start-frame + Kling multi-shot video, each Vision-QA'd and approval-gated.

        The story is split into Kling multi-shot sequences of <=10s that together
        cover the whole storyline (preserving character consistency within each
        sequence and via the shared character sheet across sequences).
        """
        if self._verbose:
            print("pipeline: Storyboard director — starting short-film production …")
        # Resolve the approval gate: an explicit request in the user's message
        # wins; otherwise fall back to the settings default (director.user_approval_step).
        if self._storyboard_wants_approval(user_text):
            auto_approve = False          # user explicitly asked to approve each step
        elif self._storyboard_auto_approve(user_text):
            auto_approve = True           # user explicitly asked to run autonomously
        else:
            auto_approve = not self._approval_default  # settings default
        if self._verbose:
            print(f"pipeline: Storyboard approval gate {'OFF (autonomous)' if auto_approve else 'ON'}.")
        user_refs = list(dict.fromkeys(
            [p for p in (self._session.last_user_input_images or []) if p]
        ))

        yield {"data": (
            "\n🎬 **Storyboard mode** — I'll design the character(s), break the story into "
            "≤10s Kling multi-shot sequences covering the whole story, then generate a "
            "start frame and video for each."
            + ("" if auto_approve else " I'll pause for your approval after each step.")
            + "\n"
        )}

        # ── Pre-step: web reference scout (only when the user asks for it) ──
        ref_desc = ""
        scout_image_refs: list[str] = []
        if self._storyboard_wants_references(user_text):
            manifest: dict = {}
            async for ev in self._stream_scout(user_text):
                if isinstance(ev, dict) and "_scout_manifest" in ev:
                    manifest = ev["_scout_manifest"]
                else:
                    yield ev
            refs = manifest.get("references", []) if isinstance(manifest, dict) else []
            ref_lines: list[str] = []
            staged: list[str] = []
            for r in refs:
                if not isinstance(r, dict):
                    continue
                d = (r.get("description") or "").strip()
                q = (r.get("query") or "reference").strip()
                if d:
                    ref_lines.append(f"- {q}: {d}")
                if r.get("mode") == "image":
                    p = (r.get("path") or "").strip()
                    if p and os.path.isfile(p):
                        staged.append(p)
            scout_image_refs = list(dict.fromkeys(staged))
            if ref_lines:
                ref_desc += "Web references found:\n" + "\n".join(ref_lines) + "\n"
            if scout_image_refs:
                yield {"_references_ready": True, "paths": scout_image_refs, "caption": "Web references found"}
            elif refs:
                yield {"data": "\n_🌐 Found reference description(s); folded into the brief._\n"}
            else:
                yield {"data": "\n_🌐 No usable web references found; continuing from your brief._\n"}

        # ── Pre-step: analyse user-provided reference images → description ──
        if user_refs:
            yield {"_story_start": True, "name": "🔎 Reference analysis"}
            _ref_request = (
                "Analyse the following reference image(s) and describe the main character's "
                "appearance (age, build, hair, skin, wardrobe, distinguishing features) and the "
                "overall visual style, concisely, for use as a character/style lock. "
                "Call analyze_image on each path.\nReference image file(s):\n"
                + "\n".join(f"- {p}" for p in user_refs)
            )
            _ud = ""
            async for ev in self._stream_plan_text_step(
                self._info_agent, "INFO", _ref_request, None, with_gallery=False
            ):
                if isinstance(ev, dict) and "_plan_step_text" in ev:
                    _ud = ev["_plan_step_text"] or ""
                else:
                    yield ev
            yield {"_story_done": True}
            if _ud:
                ref_desc = (ref_desc + "\n" + _ud).strip() if ref_desc else _ud

        # ── Story step: bible + ≤10s sequence breakdown covering the story ──
        spec: dict | None = None
        for _story_try in range(2):
            _reminder = (
                "Your previous reply did not include the required trailing ```json block "
                "from the story-storyboard skill. Output the full breakdown again and END "
                "with that single JSON block exactly per the skill's schema."
                if _story_try > 0 else ""
            )
            story_text = ""
            async for ev in self._storyboard_story(user_text, ref_desc, reminder=_reminder):
                if isinstance(ev, dict) and "_sb_story" in ev:
                    story_text = ev["_sb_story"]
                else:
                    yield ev
            spec = self._parse_storyboard_spec(story_text)
            if spec:
                break
            if self._verbose:
                print("pipeline: storyboard spec JSON missing/invalid — re-asking Story agent.")

        if not spec:
            self._record_chat_summary(user_text, triage_result, status="error")
            yield {"data": (
                "\n\n❌ I couldn't derive a structured sequence breakdown from the story. "
                "Please restate the storyline (and any character/style guidelines) and I'll retry."
            )}
            return

        characters = spec.get("characters", [])
        sequences = spec["sequences"]
        n_seq = len(sequences)

        # Story-breakdown approval gate.
        if not auto_approve and qa_reply_queue is not None:
            n_char = len(characters)
            yield {
                "approval_ask": True,
                "label": "Story breakdown",
                "description": f"Approve the {n_seq}-sequence / {n_char}-character breakdown?",
                "image_paths": [],
            }
            _d = (await qa_reply_queue.get() or "").strip()
            if _d and not (_is_affirmative(_d) or _d.lower() in self._STORYBOARD_AFFIRM):
                if _d.lower() in self._STORYBOARD_ABORT:
                    yield {"data": "\n\n🛑 Storyboard cancelled."}
                    self._record_chat_summary(user_text, triage_result, status="aborted")
                    return
                # Re-run the story once with the user's note.
                async for ev in self._storyboard_story(user_text + f"\n\n[Revision note:]\n{_d}", ref_desc):
                    if isinstance(ev, dict) and "_sb_story" in ev:
                        spec = self._parse_storyboard_spec(ev["_sb_story"]) or spec
                    else:
                        yield ev
                characters = spec.get("characters", [])
                sequences = spec["sequences"]
                n_seq = len(sequences)

        # ── DoP pass: enrich the finished breakdown with concrete cinematography ──
        # On by default (settings director.apply_cinematography); the user can opt
        # out per-run ("skip cinematography"). Rewrites every start frame + shot with
        # lighting / composition / camera movement / colour, coherent across the
        # whole film. Runs after the breakdown is approved so the approved structure
        # is what gets photographed; a parse failure is non-fatal (prompts unchanged).
        if self._cinematography_default and not self._storyboard_skip_cinematography(user_text):
            async for ev in self._storyboard_apply_dop(spec):
                if isinstance(ev, dict) and "_sb_dop_applied" in ev:
                    if ev["_sb_dop_applied"]:
                        sequences = spec["sequences"]
                        if self._verbose:
                            print("pipeline: DoP cinematography pass applied to storyboard.")
                    elif self._verbose:
                        print("pipeline: DoP pass returned no usable rewrite — keeping original prompts.")
                else:
                    yield ev

        # Persistent FOOTAGE style/quality bar (echoed by the Story agent) — applies
        # to the start frames and videos. Character references use the dedicated
        # clean guidelines instead (a reference asset, never the film grade).
        style_guidelines = (spec.get("guidelines") or "").strip()
        char_descs: dict[str, str] = {c["tag"]: c["description"] for c in characters}

        # Map known references to characters. We can only auto-map the user's image
        # when there is exactly one character; otherwise every character gets a
        # generated single-frame reference (and user/scout images stay global refs).
        char_ref_map: dict[str, str] = {}
        if len(characters) == 1 and user_refs:
            char_ref_map[characters[0]["tag"]] = user_refs[0]

        # ── Plan the character phase: per character, a single-frame reference
        # (when none is provided) and, when it appears in >1 shot, a 3x3 chart. ──
        char_plan: list[dict] = []
        step_labels: list[str] = []
        for c in characters:
            base_ref = char_ref_map.get(c["tag"])
            need_single = base_ref is None
            need_chart = int(c.get("shot_count", 0)) > 1
            char_plan.append({
                "tag": c["tag"], "desc": c["description"],
                "base_ref": base_ref, "need_single": need_single, "need_chart": need_chart,
            })
            if need_single:
                step_labels.append(f"Character ref — {c['tag']}")
            if need_chart:
                step_labels.append(f"Character chart — {c['tag']}")
        for s in sequences:
            step_labels.append(f"Sequence {s['index']} — start frame")
            step_labels.append(f"Sequence {s['index']} — multi-shot video")
        total = len(step_labels)
        yield {"_plan_ready": True, "steps": [{"description": d} for d in step_labels]}

        async def _run_one(idx: int, label: str, request: str, qa_brief: str,
                           guidelines: str, reference_paths: list[str]):
            """Drive one storyboard step and return its ``_sb_done`` result dict."""
            _res: dict | None = None
            async for _ev in self._storyboard_step(
                idx=idx, total=total, label=label, request=request, qa_brief=qa_brief,
                guidelines=guidelines, reference_paths=reference_paths,
                auto_approve=auto_approve, qa_reply_queue=qa_reply_queue,
            ):
                if isinstance(_ev, dict) and "_sb_done" in _ev:
                    _res = _ev["_sb_done"]
                else:
                    yield _ev
            yield {"_one": _res}

        step_idx = 0
        references: dict[str, str] = {}  # tag → best consistency reference path

        # ── Character phase: per-character single-frame reference + chart ──
        for cp in char_plan:
            tag, desc = cp["tag"], cp["desc"]
            base_ref = cp["base_ref"]

            if cp["need_single"]:
                sref_request = (
                    self._SB_SCOPE_IMG.format(asset=f"a single clean reference portrait of {tag}")
                    + f"Generate ONE clean character reference portrait of {tag} — a single image, "
                    "full-body or 3/4 view, neutral standing pose, plain uncluttered background — to "
                    "serve as the identity reference for this character in a short film. Use a "
                    "text-to-image or Nano-Banana template. "
                    f"Character: {desc}. "
                    f"Sheet style: {self._SB_CHARSHEET_GUIDELINES}"
                )
                sref_qa = (
                    f"A single clean reference portrait of one character ({tag}) on a plain "
                    f"background, natural colours, no film grain / VHS / grading. Character: {desc}."
                )
                _res = None
                async for ev in _run_one(step_idx, step_labels[step_idx], sref_request, sref_qa,
                                         self._SB_CHARSHEET_GUIDELINES, scout_image_refs):
                    if isinstance(ev, dict) and "_one" in ev:
                        _res = ev["_one"]
                    else:
                        yield ev
                step_idx += 1
                if not _res or not _res.get("success"):
                    self._record_chat_summary(user_text, triage_result, status="aborted" if (_res or {}).get("aborted") else "qa_failed")
                    yield {"data": f"\n\n🛑 Storyboard stopped at the character reference for {tag}."}
                    return
                base_ref = self._first_image(_res.get("output_paths", [])) or base_ref

            references[tag] = base_ref or ""

            if cp["need_chart"] and base_ref:
                chart_request = (
                    self._SB_SCOPE_IMG.format(asset=f"a single character chart image for {tag}")
                    + f"Create a CHARACTER SHEET (a 3x3 turnaround) for {tag} using the "
                    "NanoBananaPro_3x3CharacterSheet template. "
                    f"Use this reference image as the character: {base_ref}. "
                    f"Character (keep identity consistent): {desc}. "
                    "Do NOT apply the film's grungy / VHS / found-footage look to this sheet. "
                    f"Sheet style: {self._SB_CHARSHEET_GUIDELINES}"
                )
                chart_qa = (
                    f"A clean character turnaround sheet for one character ({tag}) showing them from "
                    "multiple distinct angles/poses on a clean background, identity matching the "
                    f"reference, natural colours, no film grain / VHS / grading. Character: {desc}."
                )
                _res = None
                async for ev in _run_one(step_idx, step_labels[step_idx], chart_request, chart_qa,
                                         self._SB_CHARSHEET_GUIDELINES, [base_ref]):
                    if isinstance(ev, dict) and "_one" in ev:
                        _res = ev["_one"]
                    else:
                        yield ev
                step_idx += 1
                if not _res or not _res.get("success"):
                    self._record_chat_summary(user_text, triage_result, status="aborted" if (_res or {}).get("aborted") else "qa_failed")
                    yield {"data": f"\n\n🛑 Storyboard stopped at the character chart for {tag}."}
                    return
                chart_path = self._first_image(_res.get("output_paths", []))
                if chart_path:
                    references[tag] = chart_path  # prefer the multi-angle chart for shots

        # ── Per-sequence loop: start frame → Kling multi-shot video ──
        produced_videos: list[str] = []
        for s in sequences:
            seq_i = s["index"]
            seq_tags = s.get("character_tags", [])
            seq_char_refs = list(dict.fromkeys(references[t] for t in seq_tags if references.get(t)))
            seq_char_desc = "; ".join(
                f"{t}: {char_descs[t]}" for t in seq_tags if char_descs.get(t)
            ) or ref_desc
            # Reference pool for QA + identity: this sequence's character refs first,
            # then global web/user references.
            seq_ref_pool = list(dict.fromkeys(
                p for p in (seq_char_refs + scout_image_refs + user_refs) if p
            ))

            # 1. Start frame
            sf_ref_hint = (
                "Use these character reference image(s) for identity consistency: "
                + ", ".join(seq_char_refs) + ". "
                if seq_char_refs else ""
            )
            sf_request = (
                self._SB_SCOPE_IMG.format(asset="a single start-frame still image")
                + f"Generate a single START FRAME still image (16:9) for sequence {seq_i} of a short film, "
                f"to be used as the first frame of a Kling video. {sf_ref_hint}"
                "Use an image generation/editing template that accepts the character reference(s) (e.g. a "
                "Nano-Banana image-edit template) so the character(s) stay identical. "
                f"Start frame description: {s['start_frame_prompt']}\n"
                + (f"Characters present (keep identical): {seq_char_desc}\n" if seq_char_desc else "")
                + (f"Style/quality guidelines: {style_guidelines}" if style_guidelines else "")
            )
            sf_qa_brief = (
                f"A single still image — the opening START FRAME (16:9) of sequence {seq_i} — depicting: "
                f"{s['start_frame_prompt']} The character(s) must match their reference image(s). "
                + (f"Characters: {seq_char_desc}." if seq_char_desc else "")
            )
            sf_result = None
            async for ev in _run_one(step_idx, step_labels[step_idx], sf_request, sf_qa_brief,
                                     style_guidelines, seq_ref_pool):
                if isinstance(ev, dict) and "_one" in ev:
                    sf_result = ev["_one"]
                else:
                    yield ev
            step_idx += 1
            if not sf_result or not sf_result.get("success"):
                self._record_chat_summary(user_text, triage_result, status="aborted" if (sf_result or {}).get("aborted") else "qa_failed")
                yield {"data": f"\n\n🛑 Storyboard stopped at sequence {seq_i} (start frame)."}
                return
            start_frame_path = self._first_image(sf_result.get("output_paths", []))

            # 2. Kling multi-shot video for this sequence
            shots_json = json.dumps([{"prompt": sh["prompt"], "duration": sh["duration"]} for sh in s["shots"]])
            total_secs = sum(sh["duration"] for sh in s["shots"])
            vid_request = (
                self._SB_SCOPE_VID
                + f"Generate a Kling multi-shot video for sequence {seq_i} using the Kling3_multiShot template. "
                f"Start frame (LoadImage node 14): {start_frame_path}. "
                f"Use EXACTLY these {len(s['shots'])} shot prompts as the storyboard array, in order, each with "
                f"its duration (total {total_secs}s, must be <=10s):\n{shots_json}\n"
                + (f"Character lock(s) (keep verbatim in every shot): {seq_char_desc}\n" if seq_char_desc else "")
                + "Set multi_shot to match the shot count. Aspect ratio 16:9, resolution 720p. "
                + (f"Style/quality guidelines: {style_guidelines}" if style_guidelines else "")
            )
            seq_summary = s.get("summary") or f"sequence {seq_i}"
            shot_lines = "; ".join(sh["prompt"][:90] for sh in s["shots"])
            vid_qa_brief = (
                f"A {total_secs}s multi-shot video (sequence {seq_i}, {len(s['shots'])} shots) depicting: "
                f"{seq_summary}. Shots in order: {shot_lines}. The character(s) must stay visually "
                "consistent with their reference(s) and the start frame."
                + (f" Characters: {seq_char_desc}." if seq_char_desc else "")
            )
            vid_ref_pool = list(dict.fromkeys(
                p for p in ([start_frame_path] + seq_char_refs + scout_image_refs + user_refs) if p
            ))
            vid_result = None
            async for ev in _run_one(step_idx, step_labels[step_idx], vid_request, vid_qa_brief,
                                     style_guidelines, vid_ref_pool):
                if isinstance(ev, dict) and "_one" in ev:
                    vid_result = ev["_one"]
                else:
                    yield ev
            step_idx += 1
            if not vid_result or not vid_result.get("success"):
                self._record_chat_summary(user_text, triage_result, status="aborted" if (vid_result or {}).get("aborted") else "qa_failed")
                yield {"data": f"\n\n🛑 Storyboard stopped at sequence {seq_i} (video)."}
                return
            produced_videos.extend(vid_result.get("output_paths", []))

        self._record_chat_summary(user_text, triage_result, status="completed")
        self._session.last_agent = "assemble_workflow"
        vids = ", ".join(f"`{Path(p).name}`" for p in produced_videos if p)
        n_char = len(characters)
        yield {"data": (
            f"\n\n✅ **Storyboard complete** — {n_char} character(s), {n_seq} sequence(s) rendered."
            + (f"\nVideos: {vids}" if vids else "")
        )}
        if self._verbose:
            print(f"pipeline: Storyboard director finished — {n_char} char(s), {n_seq} sequence(s).")

    # ── Triage helpers ───────────────────────────────────────────────── #

    @staticmethod
    def _extract_text(user_input: Any) -> str:
        """Extract a plain-text string from a str or multimodal content-block list."""
        if isinstance(user_input, list):
            return "\n".join(block["text"] for block in user_input if "text" in block)
        return str(user_input)

    @staticmethod
    def _scan_media_paths(user_text: str) -> tuple[list[str], list[str]]:
        """Return (image_paths, video_paths) for existing files referenced by
        path in a plain-text message. Tokens may be quoted or unquoted."""
        _IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".tiff", ".tif"}
        _VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
        tokens = re.findall(r'"([^"]+)"|\'([^\']+)\'|(\S+)', user_text)
        flat = [t for group in tokens for t in group if t]
        imgs = [t for t in flat if Path(t).suffix.lower() in _IMAGE_EXTS and os.path.isfile(t)]
        vids = [t for t in flat if Path(t).suffix.lower() in _VIDEO_EXTS and os.path.isfile(t)]
        return imgs, vids

    @staticmethod
    def _annotate_attachments(user_input: Any, user_text: str) -> str:
        """Append an attachment hint to *user_text* so triage knows images are present.

        Triage only receives the plain-text portion of the request, so without
        this hint it would classify image-edit requests as ``needs_image`` even
        when the caller already attached image content blocks or embedded a file
        path directly in their CLI message.
        """
        if not isinstance(user_input, list):
            # CLI / plain-text mode: scan for image/video file paths in the message.
            img_paths, vid_paths = Pipeline._scan_media_paths(user_text)
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
        msgs = self._assemble_workflow.messages
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
        messages = self._assemble_workflow.messages
        if not messages:
            return

        # ── Self-learning check — started AFTER summarisation to avoid ────────
        # competing for the same Ollama model concurrently.
        tool_call_count = count_tool_calls(messages)
        if self._verbose:
            print(f"pipeline: Brain used {tool_call_count} tool call(s) in this session.")

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

        # ── Fire learnings now that summarisation is done ────────────────────
        maybe_run_learnings(messages, session_id=self._session.session_id)

        if not summary:
            if self._verbose:
                print("pipeline: Empty summary returned; clearing history.")
            self._assemble_workflow.messages.clear()
            return

        # Append token-usage and cost lines to the summary
        try:
            usage = self._assemble_workflow.event_loop_metrics.accumulated_usage
            in_tok = usage.get("inputTokens", 0)
            out_tok = usage.get("outputTokens", 0)
            cache_read = usage.get("cacheReadInputTokens", 0)
            cache_write = usage.get("cacheWriteInputTokens", 0)
            token_parts = [f"{in_tok:,} in", f"{out_tok:,} out"]
            if cache_read:
                token_parts.append(f"{cache_read:,} cache hit")
            if cache_write:
                token_parts.append(f"{cache_write:,} cache write")
            cost_val, total_tokens = compute_cost_from_usage(usage, self._assemble_workflow)
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

    # ── Generated-image gallery helpers ──────────────────────────────── #

    @staticmethod
    def _caption_from_brief(raw_json: str | None) -> str:
        """Derive a short human caption from a brainbriefing JSON string.

        Prefers the positive prompt (trimmed), then the task description, then
        the template name.  Returns an empty string when nothing usable is found.
        """
        if not raw_json:
            return ""
        try:
            brief = json.loads(raw_json)
        except Exception:
            return ""
        pos = (brief.get("prompt") or {}).get("positive") or ""
        if pos:
            pos = " ".join(pos.split())
            return pos[:120] + ("…" if len(pos) > 120 else "")
        desc = (brief.get("task") or {}).get("description") or ""
        if desc:
            return desc[:120]
        return (brief.get("template") or {}).get("name") or ""

    def _register_generated_images(self, raw_json: str | None) -> None:
        """Append newly produced images to the thread gallery (dedup by path).

        Called once per completed turn while ``current_output_paths`` still holds
        that turn's outputs.  Only image files are registered; each gets a
        1-based index, a caption derived from the brainbriefing, and the turn
        number so the user can later reference it ("image 2", "the last image").
        """
        new_paths = [p for p in self._session.current_output_paths if _is_image_file(p)]
        if not new_paths:
            return
        caption = self._caption_from_brief(raw_json or self._last_brainbriefing_json)
        existing = {gi.path for gi in self._session.generated_images}
        turn = len(self._session.chat_summaries)  # this turn's summary is already appended
        for p in new_paths:
            if p in existing:
                continue
            self._session.generated_images.append(
                GeneratedImage(
                    index=len(self._session.generated_images) + 1,
                    path=p,
                    caption=caption,
                    turn=turn,
                )
            )
            existing.add(p)

    def _format_image_gallery(self) -> str:
        """Render the thread's generated-image gallery as a compact prompt block.

        Returns an empty string when no images have been generated yet.  The
        block is injected into agent prompts so the model can resolve a user
        reference ("image 2", "the last one", a description) to the real path.
        """
        gallery = self._session.generated_images
        if not gallery:
            return ""
        lines = [
            f"  {gi.index}. {gi.path}" + (f"  — {gi.caption}" if gi.caption else "")
            for gi in gallery
        ]
        return (
            "[GENERATED IMAGES IN THIS THREAD] — the user may reference these by "
            "number (\"image 2\"), recency (\"the last image\"), or description "
            "(\"the lighthouse one\"). Image numbers are 1-based and ordered oldest→newest:\n"
            + "\n".join(lines)
            + "\n[When the user refers to one of these generated images, use the matching "
            "path above as the file to act on — to analyse/describe it call "
            "analyze_image(path); to use it as a workflow input upload it via "
            "upload_image(path). These are real files; never claim no image is available.]"
        )

    def _prepend_gallery(self, text: str) -> str:
        """Prefix *text* with the generated-image gallery block when non-empty."""
        gallery = self._format_image_gallery()
        return f"{gallery}\n\n{text}" if gallery else text

    # ── Story-agent helpers (stateless, explicit-continuity) ─────────────── #

    # Hard cap on the previously-produced story text injected into the next turn.
    # A synopsis is tiny; scene descriptions are larger but bounded — this is a
    # safety valve against pathological growth, not an expected truncation point.
    _STORY_CONTEXT_CHAR_CAP = 6000

    def _clear_story_history(self) -> None:
        """Reset the Story agent's conversation history.

        The Story agent is driven *statelessly* across turns: continuity (the
        Mode A synopsis → Mode B scenes hand-off, and later refinements) is
        provided explicitly via ``AgentSession.last_story_response`` injection.
        Clearing its history each turn means we don't carry the prior turn's
        skill-tool transcript (the full activated SKILL.md, ~hundreds of tokens)
        forward — keeping token usage low — and also prevents any cross-thread
        bleed from the shared agent singleton.
        """
        try:
            self._story_agent.messages.clear()
        except Exception:  # noqa: BLE001
            pass

    def _build_story_input(self, user_text: str) -> str:
        """Prepend the Story agent's most recent output so it can be expanded.

        This is what lets "now turn that into scenes" (Mode B) or "make it
        darker" (refinement) see the prior synopsis/scenes without relying on
        conversation history. Returns *user_text* unchanged when there is no
        cached story output yet.
        """
        prev = (self._session.last_story_response or "")[: self._STORY_CONTEXT_CHAR_CAP]
        if not prev:
            return user_text
        return (
            "[Earlier in this conversation you produced the story text below. When the "
            "user asks to turn it into scene descriptions, expand it, or refine it, use "
            "this as the source/synopsis. If this is an unrelated new request, ignore it.]\n"
            f"{prev}\n\n---\n\nUser request: {user_text}"
        )

    async def _stream_story(self, user_text: str, triage_result: TriageResult):
        """Run the Story agent statelessly and stream its events.

        Clears the agent's history, injects the cached prior story output, streams
        the response, then caches the new output and records bookkeeping. Shared
        by the ``story`` handler and the feedback-on-story follow-up path.
        """
        self._clear_story_history()
        _story_input = self._build_story_input(user_text)
        _story_snap = self._usage_snapshot(self._story_agent)
        _story_chunks: list[str] = []
        _trace("pipeline.story: story_agent.stream_async begin")
        async for event in self._story_agent.stream_async(_story_input):
            if isinstance(event, dict):
                _chunk = event.get("data", "")
                if _chunk:
                    _story_chunks.append(_chunk)
            yield event
        _trace("pipeline.story: story stream done; bookkeeping")
        self._record_agent_usage(self._story_agent, _story_snap)
        _story_full = "".join(_story_chunks)
        log_agent_exchange("STORY", user_text, _story_full)
        # Cache the new output so the *next* story turn can build on it, then drop
        # the agent's own (skill-heavy) history.
        self._session.last_story_response = _story_full or None
        self._clear_story_history()
        self._session.last_agent = "story"
        self._record_chat_summary(user_text, triage_result, status="completed")

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
        # Register freshly generated images into the thread gallery so the user
        # can browse and reference them in later turns. Only successful turns
        # contribute referenceable outputs.
        if status == "completed":
            self._register_generated_images(raw_json)
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

    # Up to N correction rounds after the first attempt. Set high enough to absorb
    # local reasoning models (qwen3.6) that spiral to the output cap on a large
    # fraction of briefing attempts: at ~40% stochastic runaway, 4 retries leaves
    # ~1% residual failure (reliably-runaway recipes still need a prompt/model fix).
    _MAX_RESEARCHER_RETRIES = 4
    # Wall-clock cap per researcher attempt. Local models can loop on tool calls
    # indefinitely (no exception, no output) and would otherwise hang until the
    # whole-recipe timeout; bounding each attempt converts a hang into a retry.
    _RESEARCHER_ATTEMPT_TIMEOUT = 150.0
    # How many times to reject a content-free 'blocked' briefing before accepting
    # it (avoids exhausting all retries into a fail when a model is truly missing).
    _MAX_EMPTY_BLOCKER_RETRIES = 2
    # Auto self-correction rounds when the brain ends without calling
    # signal_workflow_ready (common with local models that stop early).
    _MAX_BRAIN_AUTORETRIES = 2

    def _build_researcher_prompt(self, user_input) -> tuple[str, str]:
        """Build the Researcher's first-attempt prompt and the extracted user text.

        Injects all the context the Researcher needs: long-term memory, image
        paths uploaded earlier in the thread, the prior-round conversation
        summary, and any Info-agent output from the previous turn.

        Returns ``(prompt_text, user_text)``.
        """
        if isinstance(user_input, list):
            user_text = "\n".join(b["text"] for b in user_input if "text" in b)
        else:
            user_text = str(user_input)

        prompt = textwrap.dedent(f"""
            User request:
            {user_text}

            Resolve all fields and output the brainbriefing JSON.
        """).strip()

        # Prepend relevant long-term memories (past style/template preferences).
        memory_ctx = self._get_memory_context(user_text)
        if memory_ctx:
            prompt = memory_ctx + "\n\n" + prompt

        # Surface images uploaded earlier in the thread when the current message
        # carries no attachments (e.g. "now make a video from it").
        current_has_images = isinstance(user_input, list) and any("image" in b for b in user_input)
        if not current_has_images and self._session.last_user_input_images:
            _paths_hint = "\n".join(
                f"  - {p}  [image uploaded earlier in this thread, use as input]"
                for p in self._session.last_user_input_images
            )
            prompt += f"\n\nInput image(s) from earlier in this thread:\n{_paths_hint}"

        # Surface the gallery of images generated earlier in this thread so the
        # Researcher can resolve references like "image 2" / "the last image".
        _gallery = self._format_image_gallery()
        if _gallery:
            prompt += f"\n\n{_gallery}"

        # Bridge chained sessions: inject the prior-round summary so the
        # Researcher can resolve OUTPUT_PATHS (prior generated files) as inputs.
        if self._last_prior_summary:
            prompt += (
                f"\n\n[CONVERSATION SUMMARY FROM PRIOR ROUND]\n\n"
                f"{self._last_prior_summary}\n\n"
                f"[END OF SUMMARY — if this request refers to previously generated outputs, "
                f"upload the file(s) from OUTPUT_PATHS via upload_image() and use the "
                f"returned filename as the workflow input.]"
            )

        # Reuse Info-agent output from the previous turn (e.g. a crafted prompt).
        if self._session.last_agent == "info" and self._session.last_info_response:
            _trimmed = self._session.last_info_response[:2000]  # hard cap — keeps tokens low
            prompt += (
                f"\n\nThe Info agent produced the following output in the previous turn "
                f"(use any prompt text or details from it):\n{_trimmed}"
            )
            self._session.last_info_response = None  # consume once

        return prompt, user_text

    @staticmethod
    def _flatten_researcher_context(messages: list) -> str:
        """Flatten a Strands message list (user request, assistant text, tool calls
        and tool results) into a compact text transcript for the constrained call."""
        parts: list[str] = []
        for m in list(messages)[-40:]:
            if not isinstance(m, dict):
                continue
            role = m.get("role", "")
            for block in (m.get("content") or []):
                if not isinstance(block, dict):
                    continue
                if "text" in block and block["text"]:
                    parts.append(f"{role}: {block['text'][:1200]}")
                elif "toolUse" in block:
                    tu = block["toolUse"] or {}
                    parts.append(f"{role} called {tu.get('name')}"
                                 f"({json.dumps(tu.get('input', {}))[:200]})")
                elif "toolResult" in block:
                    for c in ((block["toolResult"] or {}).get("content") or []):
                        if isinstance(c, dict):
                            if "text" in c:
                                parts.append(f"tool result: {c['text'][:800]}")
                            elif "json" in c:
                                parts.append(f"tool result: {json.dumps(c['json'])[:800]}")
        return "\n".join(parts)[:14000]

    def _constrain_briefing(self, user_request: str, messages: list) -> str | None:
        """Force a schema-valid BrainBriefing JSON via Ollama structured outputs
        (format=schema, no tools) — which cannot spiral and must terminate. Uses the
        researcher's own gathered context (chosen template, installed models, image
        analysis) so the output is a REAL briefing for the request, not a stub.
        Only applies to an Ollama researcher."""
        st = _settings()
        spec = str(((st.get("llm") or {}).get("pipeline") or {}).get("researcher", "")).strip()
        if "ollama" not in spec.lower():
            return None
        model_id = spec.split(",")[-1].strip()
        host = str(((st.get("llm") or {}).get("ollama") or {}).get("host", "http://localhost:11434"))
        ctx = self._flatten_researcher_context(messages)
        try:
            import ollama  # noqa: PLC0415
            schema = BrainBriefing.model_json_schema()
            r = ollama.Client(host).chat(
                model=model_id,
                messages=[
                    {"role": "system", "content":
                        "You finalise the researcher's work into ONE brainbriefing JSON "
                        "conforming to the schema. Use the USER REQUEST and RESEARCH "
                        "CONTEXT (tool results: the chosen template name, installed "
                        "model files, any image analysis). status MUST be 'ready' "
                        "(use 'blocked' only if a required model/template is missing). "
                        "Use the real template name and the actual prompt text from the "
                        "request. Never describe this instruction — output only the JSON "
                        "briefing for the request. No prose, no markdown."},
                    {"role": "user", "content":
                        f"USER REQUEST:\n{user_request[:2000]}\n\n"
                        f"RESEARCH CONTEXT (researcher's tool calls + results):\n{ctx}\n\n"
                        "Output the brainbriefing JSON for the user request now."},
                ],
                format=schema,
                options={"num_ctx": 24576, "num_predict": 6144, "temperature": 0.1},
            )
            return r.message.content
        except Exception as exc:  # noqa: BLE001
            if self._verbose:
                print(f"pipeline: schema-constrained briefing failed ({exc}).")
            return None

    def _attempt_model_downloads(self, blockers: list) -> bool:
        """Resolve each model filename named in a blocked briefing's blockers on
        HuggingFace (find_hf_file) and download it into ComfyUI's extra model path
        (download_hf_model). Returns True if at least one model was newly downloaded
        (caller should retry the researcher). No-op when downloads are disabled."""
        import re as _re  # noqa: PLC0415
        files: set[str] = set()
        for b in (blockers or []):
            if isinstance(b, str):
                for m in _re.findall(r"[\w./\\-]+\.(?:safetensors|pth|ckpt|gguf|bin|onnx|sft)", b):
                    files.add(m.replace("\\", "/").rsplit("/", 1)[-1])
        if not files:
            return False
        got = False
        for fn in files:
            try:
                res = json.loads(_find_hf_file(fn))
                exact = [m for m in res.get("matches", []) if m.get("exact")]
                if not exact:
                    if self._verbose:
                        print(f"pipeline: no HF match for missing model {fn}")
                    continue
                m = exact[0]
                hf_sub = m.get("subfolder", "") or ""
                # The HF subfolder leaf names the local model category (e.g.
                # 'diffusion_models', 'checkpoints'); download_hf_model maps it to
                # ComfyUI's extra model path.
                local_cat = hf_sub.replace("\\", "/").rstrip("/").rsplit("/", 1)[-1]
                dl = json.loads(_download_hf_model(
                    m["repo_id"], m["filename"], destination_folder=local_cat, subfolder=hf_sub))
                if dl.get("ok") and not dl.get("skipped"):
                    got = True
                    if self._verbose:
                        print(f"pipeline: downloaded missing model {fn} from "
                              f"{m['repo_id']} -> {dl.get('path')}")
                    # Record the download (filename, repo, full path) to a manifest.
                    _mlog = os.environ.get("AGENTY_DOWNLOAD_LOG")
                    if _mlog:
                        try:
                            with open(_mlog, "a", encoding="utf-8") as _mf:
                                _mf.write(json.dumps({
                                    "filename": fn, "repo_id": m["repo_id"],
                                    "path": dl.get("path"), "size_mb": dl.get("size_mb"),
                                    "subfolder": hf_sub,
                                }) + "\n")
                        except Exception:  # noqa: BLE001
                            pass
                elif dl.get("skipped") and self._verbose:
                    print(f"pipeline: downloads disabled — cannot fetch {fn}")
            except Exception as exc:  # noqa: BLE001
                if self._verbose:
                    print(f"pipeline: download of {fn} failed ({exc})")
        if got:
            try:
                _clear_tool_caches()  # so check_model / object_info see the new files
            except Exception:  # noqa: BLE001
                pass
        return got

    async def _arun_researcher(self, user_input):
        """Run the Researcher, streaming its token output as Strands events.

        Streams the Researcher's token output (including tool-use events) so
        Chainlit can display it in real time, validating and retrying the
        brainbriefing JSON up to ``_MAX_RESEARCHER_RETRIES`` times.  Yields
        standard Strands event dicts followed by a single sentinel::

            {"_researcher_done": True, "raw_json": str|None,
             "error": str|None, "researcher_output": str}

        Callers must consume the stream, watch for the sentinel, then act on
        its ``raw_json`` / ``error`` fields.
        """
        researcher_prompt_text, _ = self._build_researcher_prompt(user_input)

        last_error: str | None = None
        # A runaway/hang forces messages.clear(), which also wipes the original
        # request from context — the next attempt must then re-send it in full
        # rather than a terse correction that assumes prior context.
        _context_reset = False
        _eb_retries = 0  # empty-blocker rejections (bounded, then accept the block)
        _downloaded = False  # attempted a named-missing-model download once
        _researcher_snap = self._usage_snapshot(self._researcher)

        for attempt in range(1 + self._MAX_RESEARCHER_RETRIES):
            if attempt == 0:
                prompt = researcher_prompt_text
            elif _context_reset:
                _context_reset = False
                if self._verbose:
                    print(f"pipeline: Researcher retry {attempt}/{self._MAX_RESEARCHER_RETRIES} "
                          f"(context was reset — re-sending full request) …")
                prompt = (researcher_prompt_text
                          + "\n\nIMPORTANT: Output ONLY the brainbriefing JSON. Be concise "
                          "and use as few tool calls as possible.")
            else:
                if self._verbose:
                    print(f"pipeline: Researcher retry {attempt}/{self._MAX_RESEARCHER_RETRIES} …")
                prompt = textwrap.dedent(f"""
                    Your previous brainbriefing output failed JSON/schema validation:
                    {last_error}

                    Please output ONLY the corrected brainbriefing JSON with all
                    required fields correctly typed. No prose, no markdown fences.
                """).strip()

            chunks: list[str] = []
            try:
                async with asyncio.timeout(self._RESEARCHER_ATTEMPT_TIMEOUT):
                    async for event in self._researcher.stream_async(prompt):
                        if isinstance(event, dict):
                            chunk = event.get("data", "")
                            if chunk:
                                chunks.append(chunk)
                        yield event
                        # Drain any progress lines pushed by sync tools (e.g. download_hf_model)
                        # and surface them as plain data events so Chainlit can display them.
                        for _prog_line in _drain_progress():
                            yield {"data": _prog_line}
            except (TimeoutError, asyncio.TimeoutError):
                # The attempt looped/stalled past the per-attempt cap. Reset the
                # (now cancelled) message history and retry with a terser demand.
                last_error = ("The previous attempt did not finish in time — it looped "
                              "or stalled. Output ONLY the concise brainbriefing JSON "
                              "now, with as few tool calls as possible.")
                if self._verbose:
                    print(f"pipeline: Researcher attempt {attempt} timed out "
                          f"({self._RESEARCHER_ATTEMPT_TIMEOUT:.0f}s); resetting and retrying.")
                try:
                    self._researcher.messages.clear()
                except Exception:  # noqa: BLE001
                    pass
                _context_reset = True
                continue
            except MaxTokensReachedException:
                # Local reasoning models (qwen3.6) intermittently spiral to the
                # output cap, raising this otherwise-unrecoverable exception
                # mid-briefing. The runaway is stochastic, so reset the corrupted
                # message history and let the retry loop try again with a clean
                # context and a terser instruction.
                last_error = ("The previous attempt exceeded the model output limit "
                              "(runaway generation). Output ONLY concise, valid "
                              "brainbriefing JSON — no long explanations or repetition.")
                if self._verbose:
                    print(f"pipeline: Researcher hit max_tokens (attempt {attempt}); "
                          f"resetting context and retrying.")
                try:
                    self._researcher.messages.clear()
                except Exception:  # noqa: BLE001
                    pass
                _context_reset = True
                continue
            except Exception as exc:  # noqa: BLE001
                # Transient model/server errors (e.g. Ollama 5xx / truncated or
                # malformed response, connection resets — common right after a
                # model (re)load) — reset and retry after a short backoff rather
                # than aborting the whole recipe.
                last_error = f"researcher stream error: {exc}"
                if self._verbose:
                    print(f"pipeline: Researcher attempt {attempt} errored ({exc}); "
                          f"backing off and retrying.")
                try:
                    self._researcher.messages.clear()
                except Exception:  # noqa: BLE001
                    pass
                _context_reset = True
                await asyncio.sleep(3)
                continue

            last_response = "".join(chunks)
            label = "initial" if attempt == 0 else f"retry {attempt}"
            if self._verbose:
                print(f"pipeline: Researcher finished ({label}). Extracting brainbriefing …")

            raw_json = _extract_json(last_response)
            briefing = None
            if raw_json is not None:
                try:
                    briefing = BrainBriefing.model_validate(json.loads(raw_json))
                except (json.JSONDecodeError, ValidationError) as exc:
                    last_error = str(exc)
                    if self._verbose:
                        print(f"pipeline: Researcher ({label}) validation failed: {last_error}")
                    briefing = None
            else:
                last_error = "No JSON object found in the output."

            # #1 schema-constrained emission: if the free-text output didn't yield a
            # valid briefing, force one from the draft via Ollama structured outputs
            # (format=schema, no tools, no thinking) — which cannot run away.
            if briefing is None:
                constrained = self._constrain_briefing(user_input, list(self._researcher.messages))
                if constrained:
                    try:
                        cand = BrainBriefing.model_validate(json.loads(constrained))
                        tname = (cand.template.name or "").lower()
                        # Reject a meta/stub briefing (model describing the instruction
                        # rather than producing a briefing for the request).
                        if cand.status not in ("ready", "blocked") or \
                                "draft" in tname or "researcher" in tname or not tname:
                            last_error = "schema-constrained emission produced a stub/meta briefing"
                            briefing = None
                        else:
                            briefing = cand
                            if self._verbose:
                                print("pipeline: briefing recovered via schema-constrained emission.")
                    except (json.JSONDecodeError, ValidationError) as exc:
                        last_error = f"schema-constrained emission still invalid: {exc}"
                        briefing = None

            if briefing is None:
                continue

            # A content-free block (status='blocked' with no concrete blocker) is
            # usually an agent error — the model is often installed. Retry a bounded
            # number of times, nudging it to name the blocker or proceed ready; if it
            # still insists, accept the block (a clean 'blocked', not a fail).
            if briefing.status == "blocked" and not any(
                    isinstance(b, str) and b.strip() for b in (briefing.blockers or [])):
                if _eb_retries < self._MAX_EMPTY_BLOCKER_RETRIES:
                    _eb_retries += 1
                    last_error = (
                        "You set status='blocked' but listed no concrete blocker. If a "
                        "specific model file or template is genuinely missing, name it "
                        "exactly in 'blockers'. Otherwise the requirements ARE met — set "
                        "status='ready' and produce the full brainbriefing."
                    )
                    if self._verbose:
                        print(f"pipeline: rejected empty-blocker briefing "
                              f"({_eb_retries}/{self._MAX_EMPTY_BLOCKER_RETRIES}); retrying.")
                    _context_reset = True
                    continue
                if self._verbose:
                    print("pipeline: empty-blocker persisted — accepting the block.")

            # Deterministic download+rerun: if blocked on a named missing model,
            # resolve it on HF and fetch it into ComfyUI's extra model path, then
            # retry so the researcher can proceed. Once per request; a no-op when
            # downloads are disabled (download_hf_model fails fast).
            if briefing.status == "blocked" and not _downloaded and any(
                    isinstance(b, str) and b.strip() for b in (briefing.blockers or [])):
                _downloaded = True
                if self._attempt_model_downloads(briefing.blockers or []):
                    _context_reset = True
                    last_error = ("The previously-missing model(s) have now been "
                                  "downloaded and are available. Reassess and produce "
                                  "the full brainbriefing with status='ready'.")
                    if self._verbose:
                        print("pipeline: downloaded missing model(s); retrying researcher.")
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

    def _try_deterministic_brain(self, raw_json: str) -> str | None:
        """Mechanical brain happy-path with no LLM.

        A ``ready`` briefing that names a standard template needs no reasoning:
        load the template, apply the briefing (the same deterministic patcher the
        LLM would call), and signal. Returns the signalled workflow path on
        success, or ``None`` to fall back to the LLM brain — for a missing/build
        template, batch/variations, an annotation (2-image control) job, a
        non-standard node, or an ``apply_brainbriefing`` error the model should
        fix. Any exception also falls back.
        """
        def _bail(reason: str) -> None:
            if self._verbose:
                print(f"pipeline: deterministic brain declined — {reason}")
            return None

        try:
            bb = json.loads(raw_json)
        except Exception:  # noqa: BLE001
            return _bail("raw_json not parseable")
        if not isinstance(bb, dict) or bb.get("status") != "ready":
            return _bail(f"status={bb.get('status') if isinstance(bb, dict) else '?'}")
        tmpl = bb.get("template") or {}
        name = tmpl.get("name") if isinstance(tmpl, dict) else None
        if not name or name in ("build_new", "Kling3_multiShot"):
            return _bail(f"template={name!r}")
        # Genuine variation/batch jobs need the brain's image-batch skill.
        if bb.get("variations") or bb.get("batch_request"):
            return _bail("variations/batch")
        # NOTE: a bare count_iter>1 WITHOUT variations is a spurious researcher
        # over-count — the multiprompt expansion (_expand_variations) only fires
        # when variations is also true, so the LLM batch path just signals N
        # workflows that all fail. Don't bail on it: the deterministic path
        # renders one workflow, far more reliable than the failing batch.
        # input_image_count is likewise checked AFTER assembly against the
        # template's actual image loaders (the briefing's count is often inflated
        # because recipe ports aggregate several member templates).
        try:
            res = json.loads(_assemble_workflow_deterministic(raw_json))
            # Aux models the template references but that aren't installed (and
            # the researcher never named as blockers, e.g. a VAE/LoRA) surface in
            # missing_models. Download them and re-assemble once so they don't
            # force an LLM round-trip or a 400 at submission.
            _mm = res.get("missing_models") or []
            if res.get("status") != "ready" and _mm and not os.environ.get("AGENTY_DISABLE_DOWNLOADS"):
                if self._verbose:
                    print(f"pipeline: deterministic assembly needs {len(_mm)} model(s): "
                          f"{', '.join(m.rsplit(chr(92),1)[-1] for m in _mm[:4])} — downloading …")
                if self._attempt_model_downloads(_mm):
                    res = json.loads(_assemble_workflow_deterministic(raw_json))
            if res.get("status") != "ready":
                return _bail(f"assembly status={res.get('status')} "
                             f"missing_models={res.get('missing_models')} "
                             f"problems={str(res.get('problems'))[:400]}")
            wf_path = res.get("workflow_path")
            if not wf_path:
                return _bail("assembly returned no workflow_path")
            try:
                wf = json.load(open(wf_path, encoding="utf-8"))
                _nodes = [n for n in wf.values() if isinstance(n, dict)]
                # BatchImagesNode needs the brain's replace_node step (1.2.1); defer.
                if any(n.get("class_type") == "BatchImagesNode" for n in _nodes):
                    return None
                # A genuine 2-image annotation / control-image job — the template
                # has two image loaders whose roles the brain must assign — is
                # deferred to the LLM. A spurious input_image_count on a
                # single-loader template is ignored (assembly already bound it).
                if bb.get("input_image_count") == 2 and sum(
                    1 for n in _nodes if n.get("class_type")
                    in ("LoadImage", "LoadImageMask", "LoadImageOutput")
                ) >= 2:
                    return _bail("2 image loaders — annotation/control needs LLM")
            except Exception:  # noqa: BLE001
                pass
            _det_signal_ready(wf_path)
            return wf_path
        except Exception as exc:  # noqa: BLE001
            if self._verbose:
                print(f"pipeline: deterministic brain path errored ({exc}); LLM fallback.")
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
        self._assemble_workflow.messages.clear()
        self._ensure_clean_history()
        brain_prompt = _override_brain_prompt or self._build_brain_prompt(raw_json)
        current_input: Any = brain_prompt
        _brain_snap = self._usage_snapshot(self._assemble_workflow)

        # Deterministic happy-path DISABLED. The LLM workflow-assembly agent now
        # owns assembly end to end: it calls apply_brainbriefing and then fixes any
        # resulting errors (get_node_schema / update_workflow / replace_node) before
        # signalling. The mechanical _try_deterministic_brain path skipped the model
        # but silently mis-assembled some templates (e.g. Nano Banana — snapping a
        # staged input image to the template's default), so it is no longer used.
        # (_try_deterministic_brain is retained, uncalled, for easy revert.)
        _skip_brain_llm = False

        # Track whether a brain-assembly failure was resolved via user advice.
        _assembly_fail_error: str | None = None
        _assembly_fail_advice: str | None = None
        # Local models (qwen3.6) sometimes end the brain turn without calling
        # signal_workflow_ready; auto self-correct a bounded number of times
        # before falling back to asking a user (headless callers have no user).
        _brain_autoretry = 0

        while True:
            interrupt_result = None

            yield {"_brain_start": True}
            if _skip_brain_llm:
                # Deterministic path already assembled + signalled; skip the model
                # for this first pass and go straight to the executor below.
                _skip_brain_llm = False
            else:
                async for event in self._assemble_workflow.stream_async(current_input):
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
                    # Auto self-correction: re-prompt the brain to finish the exact
                    # tool sequence before asking a user (headless callers reply with
                    # empty advice, which would otherwise abort immediately).
                    if _brain_autoretry < self._MAX_BRAIN_AUTORETRIES:
                        _brain_autoretry += 1
                        if self._verbose:
                            print(f"pipeline: Brain auto-retry {_brain_autoretry}/"
                                  f"{self._MAX_BRAIN_AUTORETRIES} — re-prompting to signal.")
                        # Keep this re-prompt MINIMAL — do NOT re-inject the full
                        # briefing. The brain still has its own conversation history;
                        # re-injecting bloats context and sends local models into a
                        # tool-call loop that can hang until the recipe timeout.
                        current_input = (
                            "You did NOT call signal_workflow_ready, so nothing was "
                            "submitted. Do ONLY this, no explanation, no other tools:\n"
                            + (f"- Your assembled workflow is saved at: {latest_wf}\n"
                               if latest_wf else "")
                            + "- If you have not applied the brainbriefing yet, call "
                            "apply_brainbriefing once.\n"
                            "- Then call signal_workflow_ready(workflow_path) as your "
                            "final action."
                        )
                        continue
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
                    self._record_agent_usage(self._assemble_workflow, _brain_snap)
                    return

                if _qa_fail_event_b:
                    if qa_reply_queue is not None:
                        yield {"qa_fail_ask": True, **_qa_fail_event_b}
                        _answer_b = await qa_reply_queue.get()
                        if _is_affirmative(_answer_b):
                            _qa_feedback_prompt = self._build_qa_feedback_prompt(
                                brain_prompt, user_text, _qa_fail_event_b
                            )
                            yield {"data": "\n\n_🔄 Retrying with QA feedback…_"}
                            current_input = _qa_feedback_prompt
                            continue  # restart the while True loop
                    # No queue or user declined — abort.
                    self._record_chat_summary(user_text, triage_result, status="qa_failed", raw_json=raw_json)
                    self._record_agent_usage(self._assemble_workflow, _brain_snap)
                    return

                self._record_chat_summary(user_text, triage_result, status="completed", raw_json=raw_json)
                self._schedule_compression(extra_output_paths=executor_paths_b)
                self._record_agent_usage(self._assemble_workflow, _brain_snap)
                self._session.last_agent = "assemble_workflow"
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
    orchestrator_llm: str | None = None,
    free_agent: bool = True,
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
    researcher = create_query_templates_agent(
        llm=researcher_llm,
        ollama_model=researcher_ollama_model,
        anthropic_model=researcher_anthropic_model,
    )
    brain = create_assemble_workflow_agent(
        llm=brain_llm,
        anthropic_model=brain_anthropic_model,
        ollama_model=brain_ollama_model,
    )
    info_agent = create_info_agent()
    story_agent = create_story_agent()
    # The intent classifier is still built: in free-agent mode it's no longer a
    # gate, but the orchestrator can consult it on demand via the classify_intent
    # tool. (It also drives the legacy pipeline path when free_agent=False.)
    triage_agent = create_detect_user_intent_agent(
        llm=triage_llm,
        ollama_model=triage_ollama_model,
        anthropic_model=triage_anthropic_model,
    )
    planner_agent = create_planner_agent(
        llm=planner_llm,
        ollama_model=planner_ollama_model,
        anthropic_model=planner_anthropic_model,
    )
    scout_agent = create_search_web_agent()
    dop_agent = create_dop_agent()
    pipeline = Pipeline(
        researcher,
        brain,
        info_agent=info_agent,
        story_agent=story_agent,
        triage_agent=triage_agent,
        planner_agent=planner_agent,
        scout_agent=scout_agent,
        dop_agent=dop_agent,
        free_agent=free_agent,
        verbose=verbose,
        skip_brain=skip_brain,
        info_context=info_context,
        session_id=session_id,
    )
    # Build the orchestrator with the pipeline's delegation tools appended, then
    # wire it (and its live AgentSkills plugin) into the pipeline.
    if free_agent:
        orchestrator = create_orchestrator_agent(
            llm=orchestrator_llm,
            extra_tools=pipeline._delegation_tools,
        )
        pipeline.set_orchestrator(orchestrator)
    return pipeline
