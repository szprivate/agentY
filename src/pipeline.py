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

from src.agent import create_fix_workflow_assembly_agent, create_generate_new_workflow_agent, create_info_agent, create_orchestrator_agent, create_planner_agent, create_query_templates_agent, create_search_web_agent, create_vision_agent, create_video_agent, video_parallelism, vision_parallelism, _settings
from src.tools.image_handling import set_vision_agent as _set_vision_agent, vision_agents as _vision_agents
from src.tools.video_handling import set_video_agent as _set_video_agent, video_agents as _video_agents
from src.tools.annotate import set_output_sink as _set_output_sink
from src.utils.chat_summary import summarize_conversation, log_agent_messages, log_agent_exchange, set_log_thread
from src.utils.comfyui_interrupt_hook import INTERRUPT_NAME
from agenty_core.utils.comfyui_progress import stream_comfyui_job as _stream_comfyui_job
from agenty_core.utils.progress_signal import drain as _drain_progress, push as _push_progress
from src.utils.tool_activity import drain as _drain_tools, clear as _clear_tools
from src.utils.canvas_patch import drain as _drain_canvas_patch, clear as _clear_canvas_patch
from src.utils.costs import compute_cost_from_usage
from src.utils.models import AgentSession, ChatSummary, GeneratedImage, MessageIntent, TriageResult
from src.utils.workflow_signal import (
    clear_and_get as _get_workflow_signal,
    peek as _peek_workflow_signal,
)
from src.executor import (
    execute_workflow as _execute_workflow,
    execute_workflows_batch as _execute_workflows_batch,
    clear_exec_errors as _clear_exec_errors,
    get_and_clear_exec_errors as _get_exec_errors,
)
from src.utils.memory import MEMORY_NAMESPACE, format_memories, memory_add, memory_search
from src.tools.memory_tools import set_session_id as _set_memory_session_id
from src.tools.comfyui import clear_tool_caches as _clear_tool_caches
# Deterministic download+rerun: resolve a named missing model on HF and fetch it
# into ComfyUI's extra model path, then retry the query_templates.
from agenty_core.tools.huggingface import find_hf_file as _find_hf_file
from agenty_core.tools.huggingface import download_hf_model as _download_hf_model
from src.utils.learnings import count_tool_calls, maybe_run_learnings
from src.utils.debug_log import trace as _trace


# ---------------------------------------------------------------------------
# Orchestrator prompt partials — guidance for contexts that only occur on SOME
# turns (canvas hooks, input images, selected nodes) lives in separate .md files
# under config/system_prompts/orchestrator/ and is appended to the per-turn input
# ONLY when that context is present, instead of bloating the always-sent system
# prompt. Loaded from disk (never hardcoded here) and cached after first read.
# ---------------------------------------------------------------------------
_ORCH_PARTIALS_DIR = Path(__file__).parent.parent / "config" / "system_prompts" / "orchestrator"
_orch_partial_cache: dict[str, str] = {}


def _orch_partial(name: str) -> str:
    """Return the orchestrator guidance partial ``<name>.md`` (cached), or '' if it
    is missing/unreadable — a missing partial degrades to no guidance, never a crash."""
    if name not in _orch_partial_cache:
        try:
            _orch_partial_cache[name] = (_ORCH_PARTIALS_DIR / f"{name}.md").read_text(encoding="utf-8").strip()
        except Exception:  # noqa: BLE001
            _orch_partial_cache[name] = ""
    return _orch_partial_cache[name]


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


class PromptNode(BaseModel):
    """A prompt-text injection target, traced deterministically from the template graph.

    Crucially carries the node's **real** input slot name so the prompt lands in
    the correct field — ``text`` for CLIPTextEncode-style nodes but ``prompt`` for
    API / partner nodes (OpenAIGPTImageNodeV2, Gemini*, Ideogram*, …). This is the
    one piece of slot knowledge the scaffold resolves; it MUST survive to
    ``apply_brainbriefing`` (its exact-slot path) or every downstream consumer
    falls back to guessing ``text`` and mis-binds API-node prompts.
    """
    node_id: str = Field(description="Node ID in the workflow JSON")
    role: str = Field(description="'positive' or 'negative'")
    slot: str = Field(description="Real input slot for the prompt text (e.g. 'text' for CLIPTextEncode, 'prompt' for API/partner nodes)")
    node: str = Field(default="", description="ComfyUI node class name (informational)")
    max_chars: int | None = Field(default=None, description="Hard character cap this model enforces on this input (e.g. 2500 for Kling 3.0 Omni). When set, the prompt written for this node MUST fit inside it — the model refuses the call otherwise, and no repair can shorten a prompt without deciding what it was for.")


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
    prompt_nodes: List[PromptNode] = Field(default_factory=list, description="Deterministic per-node prompt-injection targets (node_id + role + real input slot) traced from the template graph. Used by apply_brainbriefing's exact-slot path so API-node prompts land in 'prompt', not 'text'.")
    positive_prompt_node_id: Optional[str] = Field(default=None, description="ComfyUI node ID of the positive prompt text node (used to splice per-variation prompts into workflow copies)")
    # notes_for_executor: Optional[str] = Field(default=None, description="Additional notes for the Brain")


class ResearcherDecision(BaseModel):
    """The Researcher's THIN output contract (Option B).

    The Researcher only makes the two judgment calls it is good at — pick a
    template and author the prompt — plus a little request metadata. The pipeline
    then assembles the full :class:`BrainBriefing` by merging this decision with
    the deterministic scaffold (``build_briefing_scaffold``), which owns every
    mechanical field (input/output/prompt node bindings, paths, model checks).
    All mechanical fields are therefore ABSENT here by design.
    """
    status: str = Field(default="ready", description="'ready' or 'blocked'")
    blockers: List[str] = Field(default_factory=list, description="Request-level blockers only (e.g. no template fits, request too unclear). Missing-model blockers are detected downstream.")
    task: Task
    template: BriefTemplate
    prompt: BriefPrompt
    count_iter: int = Field(default=1, description="Batch iterations (1 = single run, N > 1 = batch)")
    variations: bool = Field(default=False, description="True when each iteration should use a distinct prompt")
    resolution_width: Optional[Any] = Field(default=None, description="Requested width in px, or null to let the template/scaffold decide")
    resolution_height: Optional[Any] = Field(default=None, description="Requested height in px, or null")


# ---------------------------------------------------------------------------
# Multiprompt variations helper
# ---------------------------------------------------------------------------

# Canonical path where the image-batch skill writes variation prompts. This one
# stays repo-relative on purpose: it is scratch coordination between a skill and
# this module (the skill writes it with `write_text_file` at exactly this path —
# see skills/image-batch), not a workflow the user ever opens. Generated
# workflows themselves now live in ComfyUI's user dir; see _output_workflows_dir.
_MULTIPROMPT_PATH = Path("output_workflows/multiprompt.json")


def _output_workflows_dir() -> Path:
    """Directory holding generated workflow JSON (ComfyUI's user dir by default).

    Resolved through agenty_core so it tracks the same setting the assembler
    writes with, instead of assuming the old in-repo folder.
    """
    try:
        from agenty_core.tools.comfyui import _workflows_dir
        return _workflows_dir()
    except Exception:  # noqa: BLE001
        return Path("output_workflows")

# Media extensions registered in the per-thread gallery, which the user
# references by number ("image 2", "the second video").
_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp"}
_VIDEO_SUFFIXES = {".mp4", ".mov", ".webm", ".mkv", ".avi", ".m4v", ".gifv"}


def _is_image_file(path: str) -> bool:
    """Return True when *path* points to an image file (by extension)."""
    return Path(path).suffix.lower() in _IMAGE_SUFFIXES


def _is_video_file(path: str) -> bool:
    """Return True when *path* points to a video file (by extension)."""
    return Path(path).suffix.lower() in _VIDEO_SUFFIXES


def _latest_output_workflow() -> str | None:
    """Return the path of the most recently modified generated workflow JSON."""
    try:
        jsons = sorted(
            (f for f in _output_workflows_dir().glob("*.json") if f.stem != "multiprompt"),
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
    slot: str = "text",
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
        positive_prompt_node_id: Node ID whose prompt-text input receives the
                                 per-variation prompt text.
        slot:                    Real input slot name on that node ("text" for
                                 CLIPTextEncode, "prompt" for API/partner nodes).
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
                node.setdefault("inputs", {})[slot] = prompt_text
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
        *,
        info_agent: Agent | None = None,
        planner_agent: Agent | None = None,
        scout_agent: Agent | None = None,
        orchestrator_agent: Agent | None = None,
        verbose: bool = True,
        info_context: dict | None = None,
        session_id: str = "default",
    ) -> None:
        self._researcher = query_templates
        self._info_agent: Agent = info_agent or create_info_agent()
        self._planner_agent: Agent = planner_agent or create_planner_agent()
        # Consolidated workflow-repair specialist (assembly errors + execution
        # errors). Built lazily on first use to keep pipeline construction light.
        self._fix_agent: Agent | None = None
        # Build-from-scratch specialist (template.name == "build_new"). Lazy.
        self._generate_agent: Agent | None = None
        self._search_web_agent: Agent = scout_agent or create_search_web_agent()
        # Orchestrator (the free-agent entry point) + its delegation tools. The
        # delegation tools are closures over this Pipeline so they always hit the
        # current specialist instances (surviving /switch_model rebuilds).
        self._orchestrator_agent: Agent | None = None
        # Canvas-hook mode (set per-turn): the spliced base API prompt of the
        # user's on-canvas graph + the hook directives attached to it.
        self._canvas_base_prompt: dict | None = None
        self._canvas_hooks: list = []
        # Set when a keep-live producer hook injects a value into the base graph but
        # queues no batch — the base graph is then run once at turn end (below).
        self._canvas_keeplive_run: bool = False
        # Set by stop_hook_run when a hook's own directive says to abort (e.g. "if
        # any reference failed, STOP and ask"). Nothing is executed at turn end
        # while this is set. Per-turn, like the fields above.
        self._hook_run_stopped: dict | None = None
        # Set for a turn the user launched as a dry run: build everything, submit
        # nothing (see src/utils/dry_run.py). Per-turn, like the fields above;
        # _dry_graphed holds the built graphs already filed for inspection.
        self._dry_run: bool = False
        self._dry_graphed: list = []
        # Who asked to approve this turn's plan before anything runs (None = nobody,
        # the normal case), and whether they have since answered. Both per-turn; the
        # answer itself rides across turns on the session's plan_awaiting_reply.
        self._plan_approval = None
        self._plan_gate_open: bool = False
        self._plan_gate_fired: bool = False
        # workflow path -> re-runs spent on a provider content refusal (per turn).
        self._policy_retries: dict = {}
        # Outputs produced mid-turn by run_workflow_now (chained hook stages).
        # Tracked so they survive the end-of-turn current_output_paths reset and
        # still get staged onto the canvas. Empty on every non-chain turn.
        self._chain_output_paths: list = []
        # Interactive iterative-refine loop state (iterate_step). Unlike the canvas
        # fields above, this PERSISTS across turns — the loop spans many turns — so it
        # is init-ed here only, never in the per-turn reset. `_iterate_history` is the
        # numbered generation stack ([{gen, prompt, from, output_path, input_ref}], gen
        # 0 = the original image); `_iterate_targets` fingerprints the loop's hook/nodes
        # so a fresh loop (or a different graph) auto-resets it.
        self._iterate_history: list = []
        self._iterate_targets: dict | None = None
        # Snapshot of the nodes the user has selected on the canvas this turn
        # (id/type/title/widgets), so the orchestrator can read — and, via
        # set_canvas_node_params, write back — arbitrary node parameters.
        self._canvas_selection: list = []
        self._delegation_tools: list = self._build_delegation_tools()
        if orchestrator_agent is not None:
            self.set_orchestrator(orchestrator_agent)
        self._verbose = verbose
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
        # The QA briefing in force this turn (src.utils.qa.QaBriefing) — set by the
        # server from the canvas qa hook / thread /qa briefing. None = no QA.
        self._qa_briefing = None
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
            # A turn commonly asks about a whole folder of references at once, and
            # one agent can only serve one call at a time. Hand the tool a factory
            # so it can grow a small pool and actually run those in parallel; the
            # cap follows the backend (1 for a local Ollama sharing one GPU).
            _n_vision = vision_parallelism()
            _set_vision_agent(self._vision_agent,
                              factory=create_vision_agent if _n_vision > 1 else None,
                              max_parallel=_n_vision)
            if _n_vision > 1:
                print(f"[agentY] Vision agent pool: up to {_n_vision} concurrent describes.")
        except Exception as _va_exc:
            print(f"[agentY] WARNING: could not initialise VisionAgent ({_va_exc}). "
                  "analyze_image will fall back to mode='full'.")
        # Initialise the Video Agent so analyze_video works (samples frames -> a
        # vision-language model, default Qwen2.5-VL on DashScope). Same lifecycle as
        # the Vision agent: shared, stateless, folded into the turn's cost below.
        self._video_agent: Agent | None = None
        try:
            self._video_agent = create_video_agent()
            _n_video = video_parallelism()
            _set_video_agent(self._video_agent,
                             factory=create_video_agent if _n_video > 1 else None,
                             max_parallel=_n_video)
        except Exception as _vd_exc:
            print(f"[agentY] WARNING: could not initialise VideoAgent ({_vd_exc}). "
                  "analyze_video will return an error until it is configured.")
        # annotate_image writes a finished PNG mid-turn, outside the executor, so
        # it has no other way onto the canvas. Registering the session's output
        # list as its sink is what makes a marked-up image show up in the panel
        # and get staged into ComfyUI's input dir like any generated output.
        _set_output_sink(self._register_output_path)
        # Per-turn usage tracking: list of (delta_usage_dict, agent_obj) for every
        # agent that contributed tokens this turn. Reset at the start of each turn.
        self._last_turn_usages: list = []
        # Turn-start snapshots of the Vision / Video agents' accumulated usage, so
        # their per-turn deltas can be recorded at cost-finalisation time.
        self._vision_usage_snap: dict = {}
        self._video_usage_snap: dict = {}

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
        self._record_video_usage()
        return _TurnMetrics(self._last_turn_usages)

    # ── Per-turn usage tracking helpers ─────────────────────────────── #

    def _register_output_path(self, path: str) -> None:
        """Publish a file produced mid-turn by a tool as one of this turn's outputs.

        The server polls ``current_output_paths`` while streaming, so appending
        here is what pushes the file to the chat panel, adds it to the gallery and
        stages it into ComfyUI's input dir. It is also recorded as a chain output
        so it survives the end-of-turn reset and still reaches the canvas.
        """
        if not path:
            return
        try:
            paths = self._session.current_output_paths
            if path not in paths:
                paths.append(path)
            if path not in self._chain_output_paths:
                self._chain_output_paths.append(path)
        except Exception as exc:  # noqa: BLE001 — delivery must never break a tool
            print(f"[agentY] could not register output {path}: {exc}")

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

        Covers **every** agent in the vision pool, not just the first: concurrent
        describes run on grown instances, whose tokens are just as real.
        """
        if self._vision_agent is None:
            return
        for agent in _vision_agents():
            key = id(agent)
            self._record_agent_usage(agent, self._vision_usage_snap.get(key, {}))
            self._vision_usage_snap[key] = self._usage_snapshot(agent)

    def _record_video_usage(self) -> None:
        """Fold the shared Video agent's per-turn token delta into the turn usage.

        Same contract as :meth:`_record_vision_usage`: the Video agent (used by the
        ``analyze_video`` tool) runs outside the per-agent snapshot brackets, so its
        delta since the turn-start snapshot is recorded here and priced at the video
        model's own rate. Idempotent within a turn. Covers the whole pool.
        """
        if self._video_agent is None:
            return
        for agent in _video_agents():
            key = id(agent)
            self._record_agent_usage(agent, self._video_usage_snap.get(key, {}))
            self._video_usage_snap[key] = self._usage_snapshot(agent)

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
        # Ensure the Vision + Video agents' per-turn usage is included before pricing.
        self._record_vision_usage()
        self._record_video_usage()
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
        async def prepare_workflow(request: str, staged_inputs: list | None = None) -> str:
            """Research + assemble a generation request into a READY ComfyUI workflow.

            This is the one call to set up a generation. It selects the template,
            writes the prompt, and assembles the workflow deterministically — then
            you simply call ``signal_workflow_ready(workflow_path)``. Do NOT load the
            template, apply the briefing, or inspect nodes yourself; that is all done
            here.

            Args:
                request: A natural-language description of what to generate/edit.
                staged_inputs: The input image(s) you already staged, an ordered list
                    of ``{"filename": "<name in ComfyUI input dir>", "role":
                    "master_image|reference_image|mask|control_image|depth_map"}``.
                    Pass ``[]`` for pure text-to-image/video (no inputs).

            Returns JSON with a ``status`` field:
              * ``ready``     → ``workflow_path`` is assembled & validated; call
                                ``signal_workflow_ready(workflow_path)`` next.
              * ``blocked``   → ``blockers``: ask the user for the missing detail.
              * ``needs_fix`` → ``workflow_path`` + ``problems``: repair with the
                                assembly tools, then ``signal_workflow_ready``.
              * ``limit_exceeded`` → the model refuses an input outright (a prompt
                                over its character cap, more reference images than
                                it takes). ``violations`` says which input, by how
                                much; ``guidance`` says what to do. This one is
                                YOURS: shorten the prompt or drop images, patch it
                                in with ``update_workflow``, then
                                ``signal_workflow_ready``. Do NOT call
                                ``prepare_workflow`` again — the workflow is
                                otherwise assembled and valid — and do not hand it
                                to a repair agent, which cannot rewrite your prompt
                                for you.
              * ``build_new`` → ``briefing``: no template fit — build from scratch,
                                then ``signal_workflow_ready``.
              * ``error``     → ``error``: report it.
            """
            _push_progress("🔎 Researching template & prompt …")
            raw_json = None
            error = None
            async for _ev in self._arun_researcher(request, staged_inputs):
                if isinstance(_ev, dict) and "_researcher_done" in _ev:
                    raw_json = _ev.get("raw_json")
                    error = _ev.get("error")
            if error:
                _push_progress(f"⚠️ Research failed: {error}")
                return json.dumps({"status": "error", "error": error})
            if not raw_json:
                return json.dumps({"status": "error", "error": "researcher produced no briefing"})
            self._last_brainbriefing_json = raw_json
            try:
                briefing = BrainBriefing.model_validate(json.loads(raw_json))
            except (json.JSONDecodeError, ValidationError) as exc:
                return json.dumps({"status": "error", "error": f"invalid briefing: {exc}"})
            if briefing.status == "blocked":
                _push_progress("🚧 Blocked — need more information.")
                return json.dumps({"status": "blocked", "blockers": briefing.blockers})
            result = await self._assemble_deterministic(briefing)
            return json.dumps(result)

        @_tool
        async def run_info(question: str) -> str:
            """Answer a read-only question about installed models, workflows, or capabilities.

            Args:
                question: The user's question about what agentY/ComfyUI can do.
            """
            return await _run_specialist(self._info_agent, "INFO", self._prepend_gallery(question))

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
            from src.utils.plan_gate import plan_note
            steps = await _run_specialist(self._planner_agent, "PLANNER", request)
            # The instruction rides back with the plan itself: told only in a system
            # prompt several thousand tokens earlier, "say it before you start" is
            # the first thing to go when the model gets busy.
            return steps + "\n\n" + plan_note(getattr(self, "_plan_approval", None),
                                              bool(getattr(self, "_plan_gate_open", False)))

        @_tool
        async def apply_canvas_hooks(resolutions: list, run_now: bool = False) -> str:
            """Run the user's ON-CANVAS graph, expanded per the canvas hooks.

            Use this ONLY when a ``[CANVAS HOOKS]`` block is present. It runs the
            graph the user has open (already captured this turn) — do NOT assemble
            a template or call ``run_research``. Each resolution mutates ONE input
            of one anchor node across a set of values; by default the batch is the
            Cartesian product of all resolutions (capped), and each variant is queued
            for execution automatically.

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

            ZIP / PAIR (advance inputs together instead of crossing them). Give two+
            resolutions the same ``zip_group`` and they step in lockstep — run i takes
            the i-th of each — rather than cross-producting. Two ways to pair:

            * By position (default): the value lists are zipped by index (shortest
              wins). Use when you know both lists are already in the same order::

                {"target_node_id":"9","param":"image","values":[…imgs…],"zip_group":"pair"}
                {"target_node_id":"7","param":"video","values":[…vids…],"zip_group":"pair"}

            * By filename key (robust; order-independent): set ``match_by":"name"`` and
              a ``key_pattern`` regex — each value's basename is matched to a shot key
              (first capture group, else the whole match) and members are joined on
              equal keys (unmatched keys are dropped). Add a ``mode":"join_key"`` member
              to name each output by that shared key (e.g. a save node's
              ``filename_prefix``)::

                {"target_node_id":"9","param":"image","values":[…imgs…],
                 "zip_group":"shot","match_by":"name","key_pattern":"SEQ\\\\d+_SH\\\\d+"}
                {"target_node_id":"7","param":"video","values":[…vids…],
                 "zip_group":"shot","match_by":"name","key_pattern":"SEQ\\\\d+_SH\\\\d+"}
                {"target_node_id":"20","param":"filename_prefix",
                 "zip_group":"shot","mode":"join_key"}

            A ``zip_group`` behaves like one axis, so it still cross-products with any
            ungrouped resolutions (e.g. a seed sweep runs for every pair).

            RUN NOW vs QUEUE. By default the variants are queued and execute after
            your turn ends — you never see their results. Pass ``run_now=True`` when
            you need to KNOW how they turned out before deciding what to do next:
            it executes them immediately and returns, per variant, whether it
            succeeded and what it produced. Use it whenever a later hook's directive
            is conditional ("if ANY reference failed, STOP", "once all shots exist,
            …") — the RUN PLAN in the ``[CANVAS HOOKS]`` block names the hooks that
            need it. It costs the turn the generation time, so don't use it for a
            terminal batch nothing depends on.

            Args:
                resolutions: list of per-node mutation specs (see above).
                run_now: execute immediately and report per-variant results, instead
                    of queueing them for the end of the turn.
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
            if self._hook_run_stopped:
                return json.dumps({
                    "error": "this hook run was stopped ("
                             + str(self._hook_run_stopped.get("reason", "")) + ") — "
                             "nothing more runs this turn. Reply to the user instead.",
                })
            gate = self._plan_gate_refusal()
            if gate:
                return json.dumps(gate)
            try:
                cap = int(_os.environ.get("AGENTY_MAX_CANVAS_BATCH", "25") or "25")
            except ValueError:
                cap = 25
            # Whatever this batch produces belongs to the hook that asked for it.
            hook = self._hook_for_targets(
                str(r.get("target_node_id", r.get("node_id", "")) or "")
                for r in (resolutions or []) if isinstance(r, dict))
            self._tag_run_outputs(hook)
            labels: list = []
            if resolutions is None or (isinstance(resolutions, list) and not resolutions):
                # Deliberately empty: "run the graph exactly as it stands". The case
                # is a canvas whose every hook was answered from memory — there is
                # nothing left to sweep, but the run still has to happen.
                import copy as _copy
                prompts, notes = [_copy.deepcopy(base)], [
                    "no resolutions given — running the canvas as it stands"]
            else:
                # Which targets take a wire rather than a value — the hooks know,
                # the spliced graph no longer does.
                from src.utils.canvas_hooks import connection_targets as _conn
                prompts, notes = _build_batch(
                    base, list(resolutions), cap=cap, labels=labels,
                    connection_inputs=_conn(self._canvas_hooks))
            # A collector's batch aimed at a single numbered slot uses only its
            # first image, silently. Re-routed through the expander here rather
            # than left for the agent to notice: it is a mechanical rewrite with
            # one right answer, and the failure it prevents reports nothing.
            prompts, notes = self._expand_batches(prompts, notes)
            # Trim each BUILT variant — never the base it was built from. What a
            # resolution wires up is only visible after it has been written: the
            # agent selects one of the hook's wired images by NODE ID, so trimming
            # first deletes the very node the next line was about to connect.
            prompts, notes = self._trim_variants(prompts, hook, resolutions, notes)
            if not prompts:
                return json.dumps({"error": "no batch was produced", "notes": notes})
            # Every variant is a complete graph, so measure them rather than the
            # resolutions: this catches a swept prompt over the model's cap AND too
            # many images arriving at a limited input, before anything is queued.
            over = self._batch_limit_refusal(prompts)
            if over:
                return json.dumps(over)
            gone = self._collector_refusal(prompts)
            if gone:
                return json.dumps(gone)
            out_dir = Path(_tempfile.mkdtemp(prefix="agenty_canvas_"))
            paths: list[str] = []
            for i, p in enumerate(prompts):
                fp = out_dir / f"canvas_{i:03d}.json"
                fp.write_text(json.dumps(p), encoding="utf-8")
                # A dry run builds the same files and queues none of them: the
                # graph on disk is the thing being checked, and submitting it is
                # the one step being skipped.
                if not run_now and not self._dry_run:
                    _append(str(fp))
                paths.append(str(fp))
            # Name each variant by the value that makes it different, BEFORE any of
            # it runs — five reference frames are five different things, and one
            # role for the batch cannot say which is which. Applies to the queued
            # path too: same files, same names, whenever they get executed.
            self._name_variants(paths, labels, hook)
            if self._dry_run:
                return self._dry_run_report(paths, prompts, labels, notes, hook)
            if not run_now:
                if self._verbose:
                    print(f"pipeline: apply_canvas_hooks queued {len(paths)} canvas variant(s).")
                return json.dumps({
                    "status": "queued",
                    "count": len(paths),
                    "notes": notes,
                    "variants": self._variant_report(paths, labels),
                    "message": (
                        f"{len(paths)} canvas graph variant(s) queued for execution — "
                        "your work here is done; do NOT call signal_workflow_ready."
                    ),
                })
            return await self._run_canvas_batch(paths, notes, labels)

        @_tool
        async def stop_hook_run(reason: str, question: str = "",
                                keep_queued: bool = False) -> str:
            """STOP this canvas-hook run and hand back to the user.

            Use this when a hook's own directive tells you to stop — e.g. *"if ANY
            reference generation failed, STOP and ask the user for advice"*, "only
            continue if …", "abort if the script is missing X". It is the way to
            obey a conditional stop: the remaining hooks are left untouched and the
            turn ends with your explanation instead of a half-finished pipeline.

            By default it also DISCARDS work queued this turn but not yet run
            (``apply_canvas_hooks`` variants, a signalled workflow, a pending
            keep-live run) — stopping means stopping. If you meant "let what I
            already queued finish, just don't go further", pass
            ``keep_queued=True``. Anything ComfyUI already finished is kept and
            staged either way; a run from an earlier turn cannot be cancelled.

            Note the ordering trap: queued variants execute AFTER your turn, so you
            cannot check their results and then stop on them. To decide based on how
            a generation went, run it with ``apply_canvas_hooks(run_now=True)`` (or
            ``run_workflow_now``) and read the results it returns.

            After calling this, STOP calling tools: write the user a short account of
            what happened, what you did produce, and what you need from them.

            Args:
                reason: What made you stop, concretely — which hook, which step,
                    which failure. This is shown to the user.
                question: Optional. The decision you need from them, phrased as a
                    question ("Re-run the two that failed, or change the prompt?").
                keep_queued: Let already-queued workflows still run at turn end
                    instead of discarding them. Default False (discard).
            """
            reason = str(reason or "").strip()
            if not reason:
                return json.dumps({"error": "give a reason — the user is told why the run stopped."})
            if keep_queued:
                kept = len(_peek_workflow_signal())
                discarded = 0
            else:
                kept = 0
                discarded = len(_get_workflow_signal())   # queued but not yet executed
                self._canvas_keeplive_run = False
            self._hook_run_stopped = {"reason": reason,
                                      "question": str(question or "").strip(),
                                      "discarded": discarded, "kept": kept,
                                      "keep_queued": bool(keep_queued)}
            _push_progress("🛑 Hook run stopped — " + reason)
            if self._verbose:
                print(f"pipeline: stop_hook_run — {reason} (discarded {discarded}, "
                      f"kept {kept} queued workflow(s)).")
            return json.dumps({
                "status": "stopped",
                "reason": reason,
                "discarded_queued_workflows": discarded,
                "kept_queued_workflows": kept,
                "message": (
                    "Run stopped."
                    + (f" {kept} already-queued workflow(s) will still run at turn end."
                       if kept else
                       (f" {discarded} queued workflow(s) were discarded." if discarded
                        else " Nothing was queued."))
                    + " Do NOT call apply_canvas_hooks, run_workflow_now, "
                      "signal_workflow_ready or any other tool now — reply to the user: "
                      "say what stopped it, what you already produced, and ask "
                    + (f'"{question}"' if str(question or "").strip()
                       else "how they want to proceed.")
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

            if self._hook_run_stopped:
                return json.dumps({
                    "error": "this hook run was stopped ("
                             + str(self._hook_run_stopped.get("reason", "")) + ") — "
                             "nothing more runs this turn. Reply to the user instead.",
                })
            gate = self._plan_gate_refusal()
            if gate:
                return json.dumps(gate)
            # One stage of a chain: tag its outputs with whatever this turn is for,
            # so the file that feeds the next stage carries its own description.
            hooks = [h for h in (self._canvas_hooks or []) if isinstance(h, dict)]
            self._tag_run_outputs(hooks[0] if len(hooks) == 1 else None)
            if self._dry_run:
                return self._dry_run_one(workflow_path,
                                         hooks[0] if len(hooks) == 1 else None)
            base = self._session.current_output_paths
            before = len(base)
            brief = self._last_brainbriefing_json or "{}"
            try:
                async for _line in _execute_workflow(
                    workflow_path, brief, user_message="", verbose=self._verbose,
                    collected_paths=base, qa_briefing=self._qa_briefing,
                ):
                    # Surface each executor line in the chat panel too — this runs
                    # inside a tool call, so the pipeline's own event loop isn't
                    # draining meanwhile; the progress buffer (drained live by the
                    # server pump) is what carries it to the panel instead of the
                    # CLI only.
                    _push_progress(str(_line))
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
        async def place_canvas_text(hook_node_id: str, text: str) -> str:
            """Place a written answer onto the canvas as a wireable string node.

            Use this ONLY to fulfil a **TEXT canvas hook** (purpose='text') listed
            in the ``[CANVAS HOOKS]`` block. After you have written the answer,
            call this with the hook's node id and the final text. It drops an
            ``agentY text`` node on the live canvas carrying *text* and wires its
            STRING output wherever the hook's output was connected, so downstream
            nodes (or the next hook stage) consume the string on a normal run — no
            agent needed to reproduce it. The answer still streams into the chat as
            usual; do NOT generate media, call ``apply_canvas_hooks``, or run a
            workflow for a text hook.

            Args:
                hook_node_id: The id of the TEXT hook from the ``[CANVAS HOOKS]`` block.
                text: The final written answer to place (plain text / markdown).
            """
            if not str(text or "").strip():
                return json.dumps({"error": "text is empty — write the answer first, then place it."})
            from src.utils.canvas_patch import push as _push_patch
            from src.utils.canvas_hooks import inject_produced_value as _inject, _is_text

            # Resolve this hook's freeze toggle. keep-live (freeze OFF, the default)
            # leaves the hook wired and injects the value into the captured base
            # graph; freeze ON bakes the value into the target (legacy rewire). When
            # the frontend didn't send a `freeze` field at all (older extension),
            # preserve the legacy bake so existing graphs behave as before.
            hook = next((h for h in (self._canvas_hooks or [])
                         if str(h.get("hook_node_id")) == str(hook_node_id)), None)
            freeze = True
            if hook is not None and "freeze" in hook:
                freeze = bool(hook.get("freeze"))
            keep_live = not freeze

            # Refuse a value the model will refuse, before it is placed. This is the
            # one moment the agent can still fix it: it wrote the text, it is still
            # holding the turn, and the alternative is a queued run that dies inside
            # the node and reaches the user as an apology.
            over = self._canvas_limit_refusal(hook, str(text))
            if over:
                return json.dumps(over)
            gone = self._collector_text_refusal(hook, str(text))
            if gone:
                return json.dumps(gone)

            injected: list[str] = []
            if keep_live and hook is not None and isinstance(self._canvas_base_prompt, dict):
                injected = _inject(self._canvas_base_prompt, hook, str(text))
                # A PRODUCER (inline_parameter) hook whose output feeds a real node needs the
                # canvas run once so the injected value renders; a TEXT hook only
                # delivers a string for a later/other run, so it must not auto-generate.
                if injected and not _is_text(hook) and not self._hook_run_stopped:
                    self._canvas_keeplive_run = True

            # Remember it, if the hook asked to be remembered. Keyed on what fed the
            # hook, so this exact answer comes back for free until something changes.
            # Never on a dry run: a hook downstream of a stand-in writes a value
            # derived from a generation that did not happen, and a memorised one is
            # served silently to the next REAL run, where nothing says where it came
            # from. A dry run checks the logic; it does not establish facts.
            if hook is not None and hook.get("_cache_key") and not self._dry_run:
                try:
                    from src.utils.hook_cache import memorizing, write as _remember
                    if memorizing(hook):
                        _remember(hook["_cache_key"], str(text),
                                  hook=str(hook_node_id),
                                  role=self._hook_output_role(hook),
                                  directive=str(hook.get("directive") or "")[:200])
                except Exception as exc:  # noqa: BLE001
                    if self._verbose:
                        print(f"[hook-cache] could not store hook {hook_node_id}: {exc}")

            _push_patch({
                "op": "place_text",
                "hook_node_id": str(hook_node_id),
                "text": str(text),
                "keep_live": keep_live,
            })
            if keep_live:
                msg = ("Placed an 'agentY text' node on the canvas as a reference; left the "
                       "hook wired and injected your answer into the graph at run time"
                       + (f" (targets {', '.join(injected)})." if injected
                          else " (no wired real-node target — reference only)."))
            else:
                msg = ("Placed an 'agentY text' node on the canvas carrying your answer and "
                       "froze it into the input the hook's output fed.")
            return json.dumps({
                "status": "placed",
                "hook_node_id": str(hook_node_id),
                "chars": len(text),
                "keep_live": keep_live,
                "injected_targets": injected,
                "message": msg,
            })

        @_tool
        async def iterate_step(prompt: str, from_generation: str = "",
                               reset: bool = False) -> str:
            """Run ONE step of an interactive iterative-refine loop on the on-canvas graph.

            For an ``iterate`` canvas hook (see the ``[CANVAS HOOKS]`` block). In one
            deterministic call it: writes *prompt* into the hook's prompt-target node,
            feeds the chosen image into the wired LoadImage node, runs the on-canvas
            graph ONCE (synchronously), stages the result, records it in a numbered
            generation history, and updates that LoadImage in place so the next step
            continues from this result. Call it ONCE per user turn; between calls, ask
            the user for the next prompt (or a go-back) — see the ``iterative-refine``
            skill.

            Args:
                prompt: The exact prompt/instruction for THIS generation (from the user).
                from_generation: Which image to start THIS step from. ``""`` (default) =
                    the most recent generation (a normal forward step). ``"original"``
                    (or ``"0"``) = the image the loop started from. A number (``"3"``) =
                    the output of that generation. This is how you honour "go back to the
                    original / to generation N, then apply …".
                reset: Start a NEW loop, discarding the history and re-capturing the
                    LoadImage's current image as the original. Use when the user begins a
                    fresh iterative session (or points the loop at a different graph).
            """
            import copy as _copy
            import tempfile as _tempfile
            from src.utils.canvas_hooks import _is_iterate as _is_iter, _output_targets as _targets
            from src.tools.image_handling import _upload_one as _upload

            def _view(h):
                return [{"gen": e["gen"], "prompt": e["prompt"], "from": e["from"],
                         "output": e["output_path"]} for e in h]

            gate = self._plan_gate_refusal()
            if gate:
                return json.dumps(gate)
            if self._dry_run:
                # The one tool a stand-in cannot serve. Each step exists to be
                # LOOKED at, and its result is written back into the user's own
                # LoadImage node and kept as history across turns — feeding that
                # loop a path with no image behind it corrupts a real thing.
                return json.dumps({
                    "error": "this is a DRY RUN, and an iterate step is a refine loop on "
                             "real pixels — there is nothing to look at and nothing to "
                             "feed back. Tell the user the iterate hook was skipped, and "
                             "that it needs a full run.",
                })
            base = getattr(self, "_canvas_base_prompt", None)
            if not isinstance(base, dict) or not base:
                return json.dumps({"error": "no on-canvas graph is loaded this turn — open "
                                   "the graph with the iterate hook in ComfyUI, then retry."})
            hook = next((h for h in (self._canvas_hooks or []) if _is_iter(h)), None)
            if hook is None:
                return json.dumps({"error": "no `iterate` hook on the canvas — add an agentY "
                                   "hook with purpose 'iterate', wire its output into the "
                                   "prompt node and a LoadImage node into its anchor."})

            # Resolve the prompt-target node/input (the hook's OUTPUT destination) and
            # the feedback LoadImage node (a wired anchor, preferring a LoadImage-typed).
            targets = _targets(hook)
            if not targets:
                return json.dumps({"error": "the iterate hook's OUTPUT is unwired — wire it "
                                   "into the prompt node's text input so I know where the "
                                   "prompt goes."})
            prompt_node, _tt, prompt_input, _tit, _ttl = targets[0]
            prompt_input = prompt_input or "text"
            anchors = [a for a in (hook.get("anchors") or [])
                       if isinstance(a, dict) and a.get("node_id") is not None]
            if not anchors:
                return json.dumps({"error": "no feedback node wired — wire the LoadImage "
                                   "node's image output into the iterate hook's anchor."})
            fb = next((a for a in anchors if "loadimage" in str(a.get("type", "")).lower()),
                      anchors[0])
            feedback_node = str(fb["node_id"])
            feedback_input = "image"
            if prompt_node not in base:
                return json.dumps({"error": f"prompt node {prompt_node} is not in the graph."})
            if feedback_node not in base:
                return json.dumps({"error": f"LoadImage node {feedback_node} is not in the graph."})

            # (Re)initialize when asked, on the first call, or when the loop's target
            # nodes changed (a different graph). State 0 records the ORIGINAL image the
            # LoadImage node currently holds.
            tgt_key = {"prompt_node": prompt_node, "prompt_input": prompt_input,
                       "feedback_node": feedback_node, "hook": str(hook.get("hook_node_id"))}
            if reset or not self._iterate_history or self._iterate_targets != tgt_key:
                original = (base[feedback_node].get("inputs", {}) or {}).get(feedback_input, "")
                if isinstance(original, list):  # a wired link, not a filename widget
                    original = ""
                self._iterate_history = [{"gen": 0, "prompt": None, "from": None,
                                          "output_path": None, "input_ref": original}]
                self._iterate_targets = tgt_key
            hist = self._iterate_history

            # Pick the source image ref for THIS step.
            sel = str(from_generation or "").strip().lower()
            if sel in ("", "prev", "previous", "last", "next"):
                src = hist[-1]
            elif sel in ("original", "0", "orig", "source", "start"):
                src = hist[0]
            else:
                try:
                    g = int(sel)
                except ValueError:
                    return json.dumps({"error": f"from_generation '{from_generation}' is not "
                                       "'original' or a generation number.", "history": _view(hist)})
                src = next((e for e in hist if e["gen"] == g), None)
                if src is None:
                    return json.dumps({"error": f"no generation {g} in history.",
                                       "history": _view(hist)})
            source_ref = src.get("input_ref") or ""
            if not source_ref:
                return json.dumps({"error": "the starting image is unknown — set an image on "
                                   "the LoadImage node, or run one forward step first."})

            # Build the patched graph and run it synchronously (as run_workflow_now does).
            graph = _copy.deepcopy(base)
            graph[prompt_node].setdefault("inputs", {})[prompt_input] = str(prompt)
            graph[feedback_node].setdefault("inputs", {})[feedback_input] = source_ref
            run_dir = Path(_tempfile.mkdtemp(prefix="agenty_iterate_"))
            wf = run_dir / "iterate.json"
            wf.write_text(json.dumps(graph), encoding="utf-8")

            from src.executor import execute_workflow as _execute_workflow
            out_base = self._session.current_output_paths
            before = len(out_base)
            brief = self._last_brainbriefing_json or "{}"
            try:
                async for _line in _execute_workflow(
                    str(wf), brief, user_message="", verbose=self._verbose,
                    collected_paths=out_base, qa_briefing=self._qa_briefing,
                ):
                    _push_progress(str(_line))
                    if self._verbose:
                        print(f"[iterate_step] {_line}")
            except Exception as exc:  # noqa: BLE001
                return json.dumps({"error": f"generation failed: {exc}", "history": _view(hist)})
            produced = list(out_base[before:])
            self._chain_output_paths.extend(produced)
            if not produced:
                return json.dumps({
                    "error": "the run produced no fetchable output. If your saver is the "
                             "bEpic viewer node, turn its `save_to_output` ON — only then "
                             "are files written to ComfyUI history where the agent can fetch "
                             "them (temp-mode previews are not fetchable by design).",
                    "history": _view(hist),
                })

            result_path = produced[0]
            # Stage the result into the input dir and point the LoadImage node at it, so
            # the next step continues from here — and the user sees it update in place.
            up = _upload(result_path, image_type="input")
            input_ref = up.get("name") if isinstance(up, dict) else None
            if not input_ref:
                return json.dumps({"error": f"could not stage the result for feedback: {up}",
                                   "output": result_path, "history": _view(hist)})
            from src.utils.canvas_patch import push as _push_patch
            _push_patch({"node_id": feedback_node, "params": {feedback_input: input_ref},
                         "node_title": "LoadImage"})

            gen = hist[-1]["gen"] + 1
            hist.append({"gen": gen, "prompt": str(prompt), "from": src.get("gen"),
                         "output_path": result_path, "input_ref": input_ref})
            if self._verbose:
                print(f"pipeline: iterate_step produced generation {gen} -> {result_path}")
            origin = "the original" if src.get("gen") == 0 else f"generation {src.get('gen')}"
            return json.dumps({
                "status": "done",
                "generation": gen,
                "from_generation": src.get("gen"),
                "output": result_path,
                "history": _view(hist),
                "message": (f"Generation {gen} produced from {origin} and staged; the "
                            "LoadImage node now holds it. Show the user, then ask for the "
                            "next prompt or a go-back. Do NOT signal_workflow_ready."),
            })

        # NOTE: intent classification is the orchestrator's own job in free-agent
        # mode — it routes natively by choosing which specialist tool to call. The
        # former `classify_intent` advisory tool (a separate detect_user_intent LLM
        # round-trip) was never used in practice and only enlarged the tool surface,
        # so it is no longer exposed. The detect_user_intent agent survives only for
        # the legacy free_agent=False router path.
        return [prepare_workflow, run_info,
                run_web_search, run_planner, apply_canvas_hooks, stop_hook_run,
                run_workflow_now, add_canvas_workflow, set_canvas_node_params,
                place_canvas_text, iterate_step]

    async def _run_canvas_batch(self, paths: list[str], notes: list,
                                labels: list | None = None) -> str:
        """Execute canvas-hook variants NOW and report per-variant outcomes.

        The deferred path can't answer "did they work?" — it runs after the turn.
        This runs the same batch inline (same healing and QA retries as the
        end-of-turn executor, so "failed" means *couldn't be healed either*), then
        hands back which members survived. That is what makes a hook directive like
        "if ANY reference failed, STOP" answerable while there is still a decision
        to make. Outputs are staged as usual and kept past the end-of-turn reset.
        """
        collected = self._session.current_output_paths
        before = len(collected)
        _clear_exec_errors()
        qa_verdicts: dict = {}
        try:
            async for _line in _execute_workflows_batch(
                paths, self._last_brainbriefing_json or "",
                user_message="", verbose=self._verbose,
                collected_paths=collected, qa_briefing=self._qa_briefing,
                qa_retry_fn=self._qa_retry, repair_fn=self._heal_exec_failure,
                max_concurrent_repairs=3, qa_verdicts=qa_verdicts,
            ):
                # Inside a tool call the pipeline's own loop isn't draining, so the
                # progress buffer is what carries these to the panel.
                _push_progress(str(_line))
                if self._verbose:
                    print(f"[apply_canvas_hooks run_now] {_line}")
        except Exception as exc:  # noqa: BLE001
            return json.dumps({"status": "error", "error": f"execution failed: {exc}",
                               "notes": notes})
        new = list(collected[before:])
        self._chain_output_paths.extend(new)
        # Only members inline healing could NOT fix land here — a healed failure
        # is a success, and reporting it as a failure would stop a run that worked.
        errors = _get_exec_errors()
        by_path = {str(e.get("workflow_path") or ""): str(e.get("error") or "failed")
                   for e in errors}
        variants = self._variant_report(paths, labels or [], by_path)
        for v in variants:
            verdict = qa_verdicts.get(v.get("workflow"))
            if verdict:
                v["qa"] = {"passed": False, "missed": verdict.get("missed") or [],
                           "summary": verdict.get("summary") or "",
                           "retried": verdict.get("tries", 0)}
        failed = [v for v in variants if not v["ok"]]
        qa_missed = [v for v in variants if v.get("qa")]
        set_verdict = await self._qa_set_verdict(new)
        if self._verbose:
            print(f"pipeline: apply_canvas_hooks(run_now) ran {len(paths)} variant(s), "
                  f"{len(failed)} failed, {len(new)} output(s).")
        out = {
            "status": "ran",
            "count": len(paths),
            "failed_count": len(failed),
            "variants": variants,
            "outputs": new,
            "notes": notes,
            "message": (
                f"{len(paths)} variant(s) ran; {len(failed)} failed. "
                + ("Each variant's own file(s) are under its `outputs` — use THOSE to "
                   "say which result is which, never the position in the flat list "
                   "(a healed member is re-queued and finishes last). "
                   if any(v.get("outputs") for v in variants) else "")
                + ("Every variant succeeded — continue with the next hook."
                   if not failed else
                   f"{len(failed)} variant(s) could not be produced even after repair. "
                   "If a hook's directive says to stop when one fails, call "
                   "stop_hook_run now; otherwise continue with what did succeed.")
                + f" {len(new)} output file(s) staged onto the canvas."
            ),
        }
        if qa_missed:
            out["qa_failed_count"] = len(qa_missed)
            out["message"] += self._qa_instruction(len(qa_missed))
        if set_verdict:
            out["qa_set"] = set_verdict
            if not set_verdict.get("passed"):
                out["message"] += (
                    " The set was ALSO judged as a whole and missed: "
                    + "; ".join(set_verdict.get("missed") or []) + ".")
        return json.dumps(out)

    # How many built graphs a dry run files into the Workflows sidebar. One per
    # BUILD, not per variant: an 18-way sweep is one graph eighteen times with a
    # different prompt in it, while a four-stage chain is four different graphs —
    # and it is the second one you open ComfyUI to look at.
    _DRY_GRAPH_CAP = 8

    def _graph_dry_build(self, workflow_path: str, name: str = "") -> str:
        """File a built-but-unsubmitted workflow where the user can open it.

        A real run graphs what it submits (inside the executor, on the way to
        /prompt), which a dry run never reaches — so a dry run would build the
        thing worth looking at and then show it to nobody.

        It goes into the Workflows sidebar under ``agent/``. It is NOT pushed onto
        the open canvas unless the user has auto-graphing on, because that swaps
        out the graph they have open — which, during a dry run, is the hook graph
        being tested.
        """
        if len(getattr(self, "_dry_graphed", []) or []) >= self._DRY_GRAPH_CAP:
            return ""
        try:
            from agenty_core.tools.comfyui import open_workflow_in_canvas as _canvas
            from src.executor import _autoload_workflows_into_canvas as _autoload
            stem = name or Path(workflow_path).stem
            _canvas(workflow_path, name=f"dryrun_{stem}", push_to_canvas=_autoload())
        except Exception as exc:  # noqa: BLE001 — inspection is a courtesy, not the run
            if self._verbose:
                print(f"[dry-run] could not graph {workflow_path}: {exc}")
            return ""
        saved = f"agent/dryrun_{name or Path(workflow_path).stem}"
        if not hasattr(self, "_dry_graphed") or self._dry_graphed is None:
            self._dry_graphed = []
        self._dry_graphed.append(saved)
        return saved

    def _expand_batches(self, prompts: list, notes: list) -> tuple[list, list]:
        """Route a collector's batch through the expander, per built variant.

        Reported rather than done quietly: the graph the user opens afterwards has
        a node in it they did not place, and they are entitled to know why.
        """
        notes = list(notes or [])
        try:
            from src.utils.canvas_hooks import expand_image_batches
        except Exception:  # noqa: BLE001
            return prompts, notes
        out, said = [], set()
        for p in prompts:
            try:
                fixed, why = expand_image_batches(p)
            except Exception as exc:  # noqa: BLE001 — never cost the run
                if self._verbose:
                    print(f"[expand-batch] left a variant alone ({exc}).")
                fixed, why = p, []
            out.append(fixed)
            said.update(why)
        for line in sorted(said):
            notes.append(line)
            _push_progress(f"🖼️ {line}.")
        return out, notes

    def _trim_variants(self, prompts: list, hook: dict | None,
                       resolutions: list | None, notes: list) -> tuple[list, list]:
        """Cut each BUILT variant down to what it actually runs.

        Two cuts, both on the finished variant and never on the base graph it came
        from. Order is the whole lesson here: a resolution names one of the hook's
        wired images **by node id**, and ``as_connection`` resolves that id against
        the graph — so trimming the base first deleted the node the next line was
        about to connect, and the input was silently left empty. The reference
        workflows came out with no images in them.

        After the build, everything a variant uses is visibly wired:

        * **scope** — to the branch this hook's output drives, so one call builds
          one stage. Without it a five-reference sweep carries the video node five
          times and generates five videos nobody asked for.
        * **prune** — nodes that feed nothing and are not outputs. Their consumer
          was the hook, which is spliced out before the run; ComfyUI walks back
          from output nodes, so they were never going to execute.

        Guarded: every node a resolution targets must survive, and the result must
        still render something. Either failing leaves that variant exactly as it
        was — a scope that is too tight is worse than one that is too loose.
        """
        notes = list(notes or [])
        if not prompts:
            return prompts, notes
        try:
            from src.utils.canvas_hooks import prune_dead_nodes, scope_to_hook
        except Exception:  # noqa: BLE001 — never cost the run
            return prompts, notes
        wanted = {str(r.get("target_node_id", r.get("node_id", "")) or "")
                  for r in (resolutions or []) if isinstance(r, dict)}
        wanted.discard("")
        out, scoped_n, pruned_n = [], 0, 0
        for p in prompts:
            kept = p
            try:
                if hook is not None:
                    cand, dropped = scope_to_hook(kept, hook)
                    if dropped and not (wanted - set(cand)):
                        kept, scoped_n = cand, max(scoped_n, len(dropped))
                cand, dropped = prune_dead_nodes(kept)
                if dropped and not (wanted - set(cand)):
                    kept, pruned_n = cand, max(pruned_n, len(dropped))
            except Exception as exc:  # noqa: BLE001
                if self._verbose:
                    print(f"[hook-trim] left a variant whole ({exc}).")
                kept = p
            out.append(kept)
        if scoped_n:
            note = (f"scoped to hook {hook.get('hook_node_id')}'s own stage — left out "
                    f"{scoped_n} node(s) belonging to the rest of the canvas")
            notes.append(note)
            _push_progress(f"🎯 {note}.")
        if pruned_n:
            notes.append(f"dropped {pruned_n} node(s) that fed nothing and rendered "
                         "nothing (hook context inputs, ref notes)")
        if self._verbose and (scoped_n or pruned_n):
            print(f"pipeline: trimmed variants — scoped out {scoped_n}, pruned {pruned_n}.")
        return out, notes

    def _dry_run_one(self, workflow_path: str, hook: dict | None = None) -> str:
        """A single assembled workflow, built and not submitted.

        The chaining tool (``run_workflow_now``) exists to feed one stage's output
        into the next, so a dry run has to answer it with paths or the chain it is
        meant to be testing stops at stage one.
        """
        from src.utils import dry_run as _dry
        prompt = {}
        try:
            prompt = json.loads(Path(workflow_path).read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            prompt = {}
        if isinstance(prompt, dict) and isinstance(prompt.get("prompt"), dict):
            prompt = prompt["prompt"]          # a full /prompt body, not a bare graph
        role = self._hook_output_role(hook) or self._caption_from_brief(
            self._last_brainbriefing_json)
        outs = _dry.stand_ins(prompt if isinstance(prompt, dict) else {}, workflow_path,
                              label=role, index=len(_dry.runs()) + 1)
        _dry.record(workflow_path, outs, label=role, what=role)
        graphed = self._graph_dry_build(workflow_path,
                                        name=_dry.slug(role, 30) or Path(workflow_path).stem)
        _push_progress(f"🧪 Dry run — {Path(workflow_path).name} built, not submitted."
                       + (f" Graph filed as {graphed}." if graphed else ""))
        return json.dumps({
            "status": "dry_run",
            "workflow": workflow_path,
            "graphed_as": graphed,
            "outputs": outs,
            "message": (
                "DRY RUN — this workflow was built but NOT submitted. The paths under "
                "`outputs` are stand-ins; no file exists at them. Chain them onward as "
                "if the stage had succeeded (upload_image accepts them and answers with "
                "a stand-in name), and say at the end what would have been produced."
            ),
        })

    def _dry_run_report(self, paths: list, prompts: list, labels: list,
                        notes: list, hook: dict | None = None) -> str:
        """Answer a batch that was built and deliberately not submitted.

        Shaped like the answer a real run gives — same per-variant report, same
        ``outputs`` field — because the point of a dry run is to exercise what
        the agent does NEXT, and an answer it has to read differently is an
        answer that tests a different chain. What differs is stated instead of
        implied: the status says dry_run, every output path is marked, and the
        message says plainly that nothing ran.
        """
        from src.utils import dry_run as _dry
        role = self._hook_output_role(hook)
        made = 0
        for i, p in enumerate(paths):
            label = self._variant_label(labels[i] if i < len(labels) else {})
            outs = _dry.stand_ins(prompts[i] if i < len(prompts) else {}, p,
                                  label=label or role, index=i + 1)
            _dry.record(p, outs, label=label, what=role)
            made += len(outs)
            # The same join a real run makes as files land, so _variant_report
            # pairs each stand-in with the values that would have produced it.
            try:
                from src.utils.output_tags import note_source
                for o in outs:
                    note_source(o, p)
            except Exception:  # noqa: BLE001
                pass
        # EVERY variant into the Workflows sidebar, each under its own name. Filing
        # one representative was wrong: the variants of a reference sweep are five
        # different characters, not one graph five times, and "did it make them
        # all?" is the first thing anyone checks. Bounded by _DRY_GRAPH_CAP so a
        # runaway sweep still cannot bury the sidebar.
        graphed: list = []
        for i, p in enumerate(paths):
            # The VARIANT's own label first, the hook's role only as a fallback.
            # The other way round, the role — which is the directive when the user
            # named no role, and directives are long — ate the whole 40-character
            # budget, and six different reference frames were filed as
            # "dryrun_0N_take-the-character-and-place-prompts-anc". Numbered, so
            # not literally identical, and useless: the one thing that tells them
            # apart is the one thing that got truncated away.
            what = self._variant_label(labels[i] if i < len(labels) else {})
            stem = _dry.slug(what, 40) or _dry.slug(role, 40) or Path(p).stem
            got = self._graph_dry_build(
                p, name=f"{i + 1:02d}_{stem}" if len(paths) > 1 else stem)
            if got:
                graphed.append(got)
        _push_progress(f"🧪 Dry run — built {len(paths)} graph(s), submitted none; "
                       f"{made} stand-in output(s)."
                       + (f" {len(graphed)} filed for inspection." if graphed else ""))
        if self._verbose:
            print(f"pipeline: dry run — built {len(paths)} canvas variant(s), "
                  f"{made} stand-in output(s), nothing queued.")
        variants = self._variant_report(paths, labels or [])
        return json.dumps({
            "status": "dry_run",
            "count": len(paths),
            "graphed_as": graphed,
            "variants": variants,
            "outputs": [o for v in variants for o in v.get("outputs", [])],
            "notes": notes,
            "message": (
                f"DRY RUN — all {len(paths)} graph(s) were BUILT (the JSON files under "
                f"`workflow` are real and can be opened; {len(graphed)} are filed in "
                "ComfyUI's Workflows sidebar under `agent/`) and NONE were submitted to "
                "ComfyUI. The paths under `outputs` are stand-ins: no file exists at "
                "them. Continue exactly as if every variant had succeeded — hand these "
                "paths to the next hook, keep going through the chain, and report at "
                "the end what the run WOULD have produced. Do not try to open, analyse, "
                "re-run or repair a stand-in, and do not call stop_hook_run over one."
            ),
        })

    async def _qa_set_verdict(self, paths: list) -> dict | None:
        """Judge the run's outputs AS A SET, for the criteria only a set can answer.

        Per-file QA is told to mark "all of them must be consistent" as `n/a` —
        it is shown one image and cannot honestly answer otherwise. That promise
        is only worth making because the question is answered here instead.
        """
        if not self._qa_briefing:
            return None
        if len(paths or []) < 2:
            # One output cannot be inconsistent with itself, so a set criterion is
            # satisfied by default here — but the per-file judge marked it `n/a`
            # and this would otherwise say nothing at all, leaving "was that
            # checked or skipped?" unanswerable. Say which.
            if paths:
                _push_progress("🔍 QA (set) — not applicable: a single output has "
                               "nothing to be consistent with.")
            return None
        try:
            from src.utils.qa import check_set
            res = await asyncio.to_thread(check_set, list(paths), self._qa_briefing)
        except Exception as exc:  # noqa: BLE001
            if self._verbose:
                print(f"[qa] set check unavailable: {exc}")
            return None
        if res.error:
            return None
        _push_progress(f"🔍 QA (set of {len(paths)}) — "
                       + ("✅ consistent" if res.passed else f"❌ {res.summary}"))
        return {"passed": res.passed, "summary": res.summary,
                "missed": res.failed_criteria()}

    def _qa_instruction(self, n: int) -> str:
        """What to do about outputs that ran fine and missed the briefing."""
        hook = str(getattr(self._qa_briefing, "retry_hook", "") or "")
        if hook:
            return (f" {n} output(s) RAN but missed the QA briefing, and the briefing "
                    f"says to re-run hook {hook} for those — see `variants[].qa.missed`. "
                    f"Produce fresh value(s) for hook {hook} addressing exactly what was "
                    f"missed, for those variants only, and run them again. Do not re-run "
                    f"the ones that passed.")
        return (f" {n} output(s) RAN but missed the QA briefing (`variants[].qa.missed`) "
                f"and their automatic retries are spent. Decide: adjust the value(s) for "
                f"those variants and run just those again, or keep them and say what "
                f"missed. Do not re-run the ones that passed.")

    def _pending_execution_paths(self) -> list[str]:
        """The workflows to execute at turn end — none when a hook stop is in force.

        The mailbox is drained either way, so paths abandoned by a stop can't leak
        into the next turn. A plain stop drops everything — including anything
        queued *after* it, which the agent was told not to do, because a stop the
        user can silently lose is worse than no stop at all — and clears the
        keep-live flag so a producer value injected before the stop doesn't run the
        canvas by the back door. ``keep_queued`` is the deliberate exception: "let
        what I already queued finish, just go no further".
        """
        paths = _get_workflow_signal()
        # A plan still waiting to be approved runs nothing, including the keep-live
        # canvas run — that one is queued by a producer's injection rather than by a
        # tool call, so the refusals never see it and this is where it stops.
        if self._plan_gate_refusal(announce=False) is not None:
            if self._canvas_keeplive_run or paths:
                self._plan_gate_fired = True
                _push_progress("✋ Holding the run until the plan is approved.")
            self._canvas_keeplive_run = False
            return []
        stop = self._hook_run_stopped
        if not stop:
            return paths
        if stop.get("keep_queued"):
            if self._verbose:
                print(f"pipeline: hook run stopped, keeping {len(paths)} "
                      "already-queued workflow(s).")
            return paths
        if paths and self._verbose:
            print(f"pipeline: hook run stopped — dropping {len(paths)} "
                  "workflow(s) queued after the stop.")
        self._canvas_keeplive_run = False
        return []

    @staticmethod
    def _strip_unreadable_images(messages: list[dict]) -> int:
        """Drop image blocks the orchestrator cannot read. Returns how many went.

        A text-only model rejects the entire request when the history contains an
        image block ("Unexpected item type in content." from DashScope), and the
        block does not go away by itself — so one image attached under a
        text-only orchestrator breaks every subsequent turn of that conversation,
        and the only apparent escape is starting a new one. New images are no
        longer embedded for such a model, but a conversation poisoned before that
        fix — or before the user switched models mid-thread — still has to heal.

        Only the image block is removed; the accompanying text (which lists the
        file paths) stays, so the agent keeps the inputs it needs.
        """
        try:
            from src.utils.agentY_server import _orchestrator_supports_vision
            if _orchestrator_supports_vision():
                return 0
        except Exception:  # noqa: BLE001 — never let this check cost a turn
            return 0
        dropped = 0
        for msg in messages:
            content = msg.get("content")
            if not isinstance(content, list):
                continue
            kept = [b for b in content
                    if not (isinstance(b, dict) and "image" in b)]
            if len(kept) != len(content):
                dropped += len(content) - len(kept)
                # A message emptied of everything would itself be invalid.
                msg["content"] = kept or [{"text": "(image omitted)"}]
        return dropped

    def _ensure_orch_clean_history(self) -> None:
        """Sanitize the orchestrator's message list (drop orphaned tool blocks,
        and images the configured model cannot accept)."""
        agent = self._orchestrator_agent
        if agent is None:
            return
        msgs = getattr(agent, "messages", None)
        if not msgs:
            return
        dropped = self._strip_unreadable_images(msgs)
        if dropped and self._verbose:
            print(f"pipeline: dropped {dropped} image block(s) the orchestrator "
                  f"cannot read (text-only model).")
        cleaned = self._sanitize_messages(list(msgs))
        if len(cleaned) != len(msgs):
            if self._verbose:
                print(f"pipeline: Sanitized orchestrator history: removed "
                      f"{len(msgs) - len(cleaned)} orphaned tool message(s).")
            agent.messages[:] = cleaned

    def _learn_from_orchestrator_turn(self, msg_start: int) -> None:
        """Fire the learnings agent on this turn's orchestrator activity.

        Passes only the messages produced since the turn started (not the whole
        sliding window) so the analyser focuses on the failure→fix arc that just
        played out and stays cheap. ``maybe_run_learnings`` self-gates on the
        tool-call count, so trivial turns are skipped; substantial ones (a
        repeated error the agent finally resolved, or a user correction it acted
        on) get a concise ``problem | solution`` line appended to the learnings
        skill + FAISS. Best-effort — never raises into the turn.
        """
        agent = self._orchestrator_agent
        if agent is None:
            return
        try:
            msgs = list(getattr(agent, "messages", []) or [])
            # Normal case: this turn's messages are the tail from msg_start. If the
            # sliding window trimmed the front mid-turn (msg_start now past the end),
            # fall back to a bounded recent tail so we still learn from what happened.
            turn_messages = msgs[msg_start:] if 0 <= msg_start < len(msgs) else msgs[-50:]
            if turn_messages:
                maybe_run_learnings(turn_messages, session_id=self._session.session_id)
        except Exception as exc:  # noqa: BLE001
            if self._verbose:
                print(f"pipeline: orchestrator learnings trigger failed ({exc}).")

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
        and edits them by calling ``set_canvas_node_params(node_id, params)``, and
        is prompted to open every turn with a summary of ALL selected nodes — so
        every selected node is listed, including ones with no editable widgets
        (shown as "(no editable parameters)"). Returns "" only when the selection
        is empty.
        """
        sel = getattr(self, "_canvas_selection", []) or []
        if not sel:
            return ""
        # Inline the full widget value so a long prompt in a selected node reaches
        # the agent intact — the old flat 400-char cap silently truncated exactly
        # the "read this prompt" case, and there is no read-back tool to recover the
        # rest. A generous per-widget cap plus a total budget still guards a
        # pathological many-large-nodes selection. Override via
        # AGENTY_CANVAS_SEL_WIDGET_CHARS / AGENTY_CANVAS_SEL_TOTAL_CHARS.
        def _int_env(name: str, default: int) -> int:
            try:
                return max(1, int(os.environ.get(name, str(default))))
            except (TypeError, ValueError):
                return default
        per_cap = _int_env("AGENTY_CANVAS_SEL_WIDGET_CHARS", 8000)
        total_cap = max(per_cap, _int_env("AGENTY_CANVAS_SEL_TOTAL_CHARS", 24000))

        lines: list[str] = []
        used = 0
        budget_hit = False
        for n in sel:
            widgets = n.get("widgets")
            widgets = widgets if isinstance(widgets, dict) else {}
            nid = n.get("id")
            ntype = n.get("type") or "?"
            title = n.get("title") or ntype
            head = f"- node #{nid} [{ntype}]" + (f' "{title}"' if title and title != ntype else "")
            lines.append(head)
            # List every selected node, even ones with no readable/editable widgets
            # (a Reroute, Note, …), so the agent's summary covers ALL of them.
            if not widgets:
                lines.append("    • (no editable parameters)")
                continue
            for wname, wval in widgets.items():
                sval = str(wval)
                if len(sval) > per_cap:
                    sval = sval[:per_cap] + "…[truncated]"
                remaining = total_cap - used
                if remaining <= 0:
                    lines.append("    • …[further selected-node params omitted to bound context]")
                    budget_hit = True
                    break
                if len(sval) > remaining:
                    sval = sval[:remaining] + "…[truncated]"
                used += len(sval)
                lines.append(f"    • {wname} = {sval!r}")
            if budget_hit:
                break
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
                # Attach the how-to-run-hooks guidance only now that hooks exist
                # (it's absent from the base system prompt to keep every non-hook
                # turn lean). Guidance first, then the concrete hook block.
                guide = _orch_partial("canvas_hooks")
                block = (guide + "\n\n" if guide else "") + hooks_block
                pin = block + "\n" + pin

        # Canvas selection: the nodes the user has selected on the graph, with
        # their current parameter values. Lets the orchestrator read a node ("read
        # this prompt") and write it back via set_canvas_node_params. The read/edit
        # guidance rides along only when a selection is actually present.
        sel_block = self._describe_canvas_selection()
        if sel_block:
            guide = _orch_partial("selected_nodes")
            pin = pin + (guide + "\n\n" if guide else "") + sel_block + "\n\n"

        # Input-image handling guidance — attached only when the user has input
        # images to stage or generated images to reference (else it's dead weight).
        has_input_images = bool(getattr(self._session, "last_user_input_images", None)) \
            or bool(getattr(self._session, "generated_images", None))
        if not has_input_images and isinstance(user_input, list):
            has_input_images = any(isinstance(b, dict) and "image" in b for b in user_input)
        if has_input_images:
            guide = _orch_partial("input_images")
            if guide:
                pin = pin + guide + "\n\n"

        # When canvas auto-graphing is off, the user opted out of having every
        # generated workflow loaded onto their canvas — so tell the orchestrator
        # to offer it instead of doing it silently.
        try:
            from src.executor import _autoload_workflows_into_canvas as _autoload
            if not _autoload():
                pin = pin + (
                    "[CANVAS DISPLAY] Auto-graphing of generated workflows onto the "
                    "ComfyUI canvas is OFF. Build and run workflows as usual, but do NOT "
                    "load them onto the canvas automatically. After you produce a result, "
                    "offer once, in your reply: \"Want me to graph the generated "
                    "workflows — just say the word and I'll load them for you to "
                    "inspect.\" If the user agrees (now or in a later turn), call "
                    "open_workflow_in_canvas(workflow_path) for each workflow you built.\n\n"
                )
        except Exception:  # noqa: BLE001
            pass

        # Recall relevant long-term memory (past preferences + failure→fix lessons)
        # and surface it up front, so the orchestrator ALWAYS sees it before building
        # even if it never calls memory_read itself. Guaranteed recall is the point of
        # the store — leaving it to a discretionary tool call means a model that skips
        # the call never learns from the 60+ lessons on record. memory_read stays
        # available for pulling more detail on demand. Best-effort (returns "" on any
        # error or when memory is disabled); labelled so it reads as recalled context,
        # not a fresh instruction.
        memory_ctx = self._get_memory_context(user_text)
        if memory_ctx:
            pin = pin + (
                "[RECALLED FROM LONG-TERM MEMORY — honor any stated preference below "
                "and avoid repeating a failure noted here; call memory_read for more.]\n"
                + memory_ctx + "\n\n"
            )

        # Per-project memory: what this production has already established
        # (characters, style, locked references, delivery specs). Read from the
        # store beside the project — ComfyUI's user directory, which the pipeline
        # switches when it switches project — so it needs no session state here.
        # Guidance rides along only when there is something to apply it to.
        project_ctx = self._get_project_memory_context()
        if project_ctx:
            guide = _orch_partial("project_memory")
            pin = pin + (guide + "\n\n" if guide else "") + project_ctx + "\n\n"

        # Plan approval — only where someone actually asked to be asked. Resolved
        # here, once, from the three places such a standing rule can live (the
        # user's message, a hook node's directive, the project's memory), so the
        # tools that would run work all consult the same verdict. Last of the
        # blocks: it decides whether anything above it happens this turn.
        self._plan_approval = self._detect_plan_approval(user_text, project_ctx)
        if self._plan_approval is not None:
            from src.utils.plan_gate import approval_state
            guide = _orch_partial("plan_approval")
            pin = pin + (guide + "\n\n" if guide else "") + approval_state(
                self._plan_approval, bool(getattr(self, "_plan_gate_open", False))) + "\n\n"

        # Ahead of every other block, including the hooks: it does not add a rule,
        # it changes what all of them mean this turn.
        if getattr(self, "_dry_run", False):
            guide = _orch_partial("dry_run")
            if guide:
                pin = guide + "\n\n" + pin

        if isinstance(user_input, list):
            gallery = self._format_image_gallery()
            blocks = list(user_input)
            # Token control: by default the orchestrator receives image bytes (so a
            # vision-capable seat can route on pixels). Set AGENTY_ORCH_IMAGES=0 to
            # send only the attached file paths (already listed in the text block) —
            # the orchestrator delegates visual understanding to run_research, so
            # dropping the bytes removes every input image from each of its tool-call
            # round-trips and slashes context for small/expensive orchestrator models.
            if os.environ.get("AGENTY_ORCH_IMAGES", "1") == "0":
                blocks = [b for b in blocks
                          if not (isinstance(b, dict) and "image" in b)]
            prefix = pin + (gallery + "\n\n" if gallery else "")
            if prefix:
                blocks.insert(0, {"text": prefix})
            return blocks
        return pin + self._prepend_gallery(self._annotate_attachments(user_input, user_text))

    def _log_orchestrator(self) -> None:
        """Write the orchestrator's message history to message_history.log — once
        per turn. Called from the terminal branches AND from a ``finally`` in
        ``stream_async``, so a turn that never reaches a terminal branch (user
        ``/stop``, an exception, or a hang) is still captured. Idempotent via the
        per-turn ``_orch_turn_logged`` flag (reset at the start of each turn).
        """
        if getattr(self, "_orch_turn_logged", False):
            return
        self._orch_turn_logged = True
        try:
            log_agent_messages("ORCHESTRATOR", list(self._orchestrator_agent.messages))
        except Exception:  # noqa: BLE001
            pass

    async def _astream_orchestrator(self, user_input, *, qa_reply_queue: asyncio.Queue | None = None,
                                    canvas_prompt: dict | None = None, canvas_hooks: list | None = None,
                                    canvas_selection: list | None = None, qa_briefing=None,
                                    dry_run: bool = False):
        """Stream the orchestrator for one turn, then run any signalled workflow.

        This replaces the triage → route → handler block: the orchestrator owns
        the turn end-to-end. After it finishes (no ComfyUI interrupt pending), the
        workflow-signal mailbox is drained and the Executor runs exactly as in the
        legacy Brain stage — so ComfyUI submission / Vision-QA / output-staging is
        unchanged. ComfyUI interrupts are handled identically to the Brain stage.
        """
        self._last_turn_usages = []
        # Keyed by id(agent): the vision/video pools may hold several instances.
        self._vision_usage_snap = {id(a): self._usage_snapshot(a) for a in _vision_agents()}
        self._video_usage_snap = {id(a): self._usage_snapshot(a) for a in _video_agents()}
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
        self._canvas_keeplive_run = False
        self._hook_run_stopped = None
        # Dry run: everything up to the submission happens — the hooks are read, the
        # values written, the variants built to disk — and each graph is answered
        # with stand-in paths instead of being handed to ComfyUI. Armed on the
        # module too, because the tools that must recognise a stand-in (analysis,
        # upload) are module-level and never see `self`.
        self._dry_run = bool(dry_run)
        self._dry_graphed = []
        try:
            from src.utils import dry_run as _dry_mod
            _dry_mod.arm(self._dry_run)
        except Exception:  # noqa: BLE001
            pass
        # How many times each (node, input) has been handed back this turn for
        # breaking a hard model limit. Told once, an agent shortens; told the same
        # thing three times it is stuck, and needs different advice, not the same
        # sentence again.
        self._limit_handbacks = {}
        # Plan approval. The gate is shut for a turn whose plan someone asked to
        # approve, and stands open for exactly the one turn that follows the user
        # answering — which is what their last message was, if the flag survived
        # from the previous turn. Resolved into _plan_approval by the input build.
        self._plan_gate_open = bool(getattr(self._session, "plan_awaiting_reply", False))
        self._session.plan_awaiting_reply = False
        self._plan_approval = None
        self._plan_gate_fired = False
        # How many times each workflow has been re-run after a provider refused it
        # on content grounds. Per-turn: a refusal that ran out of retries is not
        # held against the next request.
        self._policy_retries = {}
        self._chain_output_paths = []
        # The QA briefing in force this turn, already resolved by the caller
        # (a canvas qa hook wins over the thread's /qa briefing). None = no QA.
        self._qa_briefing = qa_briefing
        # Arbitrary selected nodes (id/type/title/widgets) the orchestrator can
        # read and write back via set_canvas_node_params.
        self._canvas_selection = [n for n in (canvas_selection or []) if isinstance(n, dict)]
        if isinstance(canvas_prompt, dict) and canvas_prompt:
            try:
                from src.utils.canvas_hooks import (splice_hook_nodes, prune_to_hooks,
                                                    hook_scoped_graph)
                # Scope the canvas to what the executed hooks actually reach, so a
                # hook on one branch of a big graph doesn't drag every unrelated
                # output chain into the run (and into every workflow written for
                # it). Only when hooks actually drive this turn: the canvas is sent
                # on every turn, and a disabled hook left on it must not silently
                # trim a plain request. Off → the whole canvas runs, as before.
                scoped, dropped = canvas_prompt, []
                if self._canvas_hooks and hook_scoped_graph():
                    scoped, dropped = prune_to_hooks(
                        canvas_prompt,
                        [h.get("hook_node_id") for h in self._canvas_hooks])
                # Hooks carry the declared type of each input they feed, which is
                # what keeps a STRING target from being rewired to an IMAGE anchor.
                cleaned, removed = splice_hook_nodes(scoped, self._canvas_hooks)
                self._canvas_base_prompt = cleaned
                if dropped:
                    _push_progress(
                        f"🎯 Hook scope: {len(cleaned)} node(s) connected to the hook(s); "
                        f"left out {len(dropped)} unrelated node(s).")
                if self._verbose:
                    print(f"pipeline: canvas-hook mode — {len(self._canvas_hooks)} hook(s), "
                          f"spliced {len(removed)} hook node(s); base graph has "
                          f"{len(cleaned)} node(s)"
                          + (f" (scoped out {len(dropped)})" if dropped else "") + ".")
            except Exception as exc:  # noqa: BLE001
                print(f"pipeline: canvas-hook splice failed ({exc}); ignoring canvas graph.")

        # What this turn produces is tagged with what it was for; last turn's tags
        # are not this turn's. (The sidecars on disk are the record that lasts.)
        try:
            from src.utils.output_tags import clear as _clear_tags
            _clear_tags()
        except Exception:  # noqa: BLE001
            pass
        # Hooks that remember: put last time's answer back before anyone is asked
        # to produce it again.
        self._apply_hook_cache()

        self._ensure_orch_clean_history()
        # Mark where this turn's messages begin so the learnings pass (fired on
        # completion) analyses only what just happened, not the whole window.
        _orch_msg_start = len(getattr(self._orchestrator_agent, "messages", []) or [])
        orch_input = self._build_orchestrator_input(user_input, user_text)
        # signal_workflow_ready is a module-level tool (the subagents share it), so
        # its gate has to travel on the mailbox rather than through `self`.
        try:
            from src.utils.workflow_signal import set_execution_hold
            set_execution_hold(self._plan_gate_refusal(announce=False))
        except Exception:  # noqa: BLE001
            pass
        current_input: Any = orch_input
        _snap = self._usage_snapshot(self._orchestrator_agent)
        # Drop any tool-activity / canvas-patch left over from a previous turn.
        _clear_tools()
        _clear_canvas_patch()
        _clear_exec_errors()

        # ComfyUI run failures are healed inline by the executor (repair_fn below):
        # each failed member is repaired concurrently and re-queued on the fly,
        # bounded per-member, so there is no orchestrator re-drive loop here.
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
                workflow_paths = self._pending_execution_paths()
                # Keep-live producers injected their value(s) into the captured base
                # graph but queued no batch (no sweep / signal) — run the canvas once
                # so those values render. Skipped when anything else was already
                # queued (a sweep's build_batch deep-copies the injected base graph,
                # so the values ride along there).
                _keeplive_run = False
                if (not workflow_paths and getattr(self, "_canvas_keeplive_run", False)
                        and isinstance(self._canvas_base_prompt, dict) and self._canvas_base_prompt):
                    try:
                        import tempfile as _tf
                        _kd = Path(_tf.mkdtemp(prefix="agenty_keeplive_"))
                        _kp = _kd / "canvas_keeplive.json"
                        _kp.write_text(json.dumps(self._canvas_base_prompt), encoding="utf-8")
                        workflow_paths = [str(_kp)]
                        _keeplive_run = True
                        if self._verbose:
                            print("pipeline: keep-live canvas run — queued base graph with "
                                  "injected producer value(s).")
                    except Exception as _exc:  # noqa: BLE001
                        print(f"pipeline: keep-live canvas run could not be queued ({_exc}).")
                workflow_paths = self._expand_variations(workflow_paths, self._last_brainbriefing_json or "")
                # The last gate before ComfyUI. apply_canvas_hooks queues nothing in a
                # dry run, but a signalled workflow and the keep-live canvas run both
                # arrive here without passing a tool that could have stopped them —
                # so "nothing is submitted" is enforced at the submission itself.
                if self._dry_run and workflow_paths:
                    _hh = [h for h in (self._canvas_hooks or []) if isinstance(h, dict)]
                    for _wp in workflow_paths:
                        self._dry_run_one(_wp, _hh[0] if len(_hh) == 1 else None)
                    workflow_paths = []
                # Reset this turn's outputs before the deferred batch, but KEEP any
                # produced mid-turn by run_workflow_now (chained stages) so they're
                # still staged. Non-chain turns have none, so this equals .clear().
                self._session.current_output_paths[:] = list(self._chain_output_paths)
                exec_paths = self._session.current_output_paths
                _outputs_before = len(exec_paths)  # chain outputs already staged
                _qa_fail_event: dict | None = None
                if workflow_paths:
                    # Name what is about to come out: the hook that drove this run if
                    # there was one, else the briefing the workflow was built from.
                    _hooks = [h for h in (self._canvas_hooks or []) if isinstance(h, dict)]
                    self._tag_run_outputs(_hooks[0] if len(_hooks) == 1 else None)
                    if self._verbose:
                        count = len(workflow_paths)
                        tag = f"{count} workflows (batch)" if count > 1 else workflow_paths[0]
                        print(f"pipeline: Orchestrator signaled {tag} ready.")
                    # Fresh error mailbox for this run; the executor records only
                    # members it could NOT heal (healed failures never land here).
                    _clear_exec_errors()
                    async for line in _execute_workflows_batch(
                        workflow_paths,
                        self._last_brainbriefing_json or "",
                        user_message=user_text,
                        verbose=self._verbose,
                        collected_paths=exec_paths,
                        qa_briefing=self._qa_briefing,
                        # A member that RAN but missed the user's QA criteria is
                        # re-generated against exactly the criteria it missed,
                        # bounded by qa.max_retries. Never for the keep-live run:
                        # that is the user's own canvas graph, not ours to rewrite.
                        qa_retry_fn=None if _keeplive_run else self._qa_retry,
                        # Heal failed members in place, concurrently, on the fly:
                        # the executor re-queues each healed workflow immediately
                        # while the survivors keep running (≤3 repairs at once).
                        # EXCEPT the keep-live run — that is the user's ON-CANVAS
                        # graph; never rebuild it (a real problem, e.g. a missing
                        # output node, is surfaced for the user to fix instead of
                        # looping the fixer over their graph).
                        repair_fn=None if _keeplive_run else self._heal_exec_failure,
                        max_concurrent_repairs=3,
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

                    # The set verdict: the criteria only a set can answer, on the
                    # path that produces most sets. Per-file QA is told to mark
                    # those n/a because they are checked here — a promise that has
                    # to hold on the queued batch too, not just on run_now.
                    _set = await self._qa_set_verdict(list(exec_paths[_outputs_before:]))
                    if _set and not _set.get("passed"):
                        yield {"data": ("\n\n🔍 QA — the outputs pass individually, but "
                                        "as a SET they miss: "
                                        + "; ".join(_set.get("missed") or [])
                                        + ". They are all delivered; say the word to "
                                          "re-run the ones that break it.")}

                # ── Surface members inline-healing couldn't fix ──────────────── #
                # The executor already healed failed members on the fly (repair_fn
                # above) and re-queued each one immediately, concurrently, while the
                # survivors kept running. Anything still in the error mailbox is a
                # member that exhausted its heal budget — surface it (keeping every
                # partial success) instead of re-repairing or re-driving the loop.
                _exec_errors = _get_exec_errors() if not _qa_fail_event else []
                if _exec_errors:
                    _partial = len(exec_paths) > _outputs_before
                    _failed: dict[str, dict] = {}
                    for _e in _exec_errors:
                        _wp = _e.get("workflow_path") or (workflow_paths[0] if workflow_paths else "")
                        if _wp:
                            _failed.setdefault(_wp, _e)
                    _err0 = next(iter(_failed.values()), {})
                    _det = _err0.get("details") or {}
                    _nt = _det.get("node_type", "?")
                    _why = _det.get("exception_message") or _err0.get("error") or "unknown error"
                    # A provider refusing the content is not a broken run, and
                    # telling the user it "could not be auto-healed" sends them
                    # looking for a defect that isn't there.
                    _refused = _det if _det.get("kind") == "content_policy" else None
                    if _refused and not _partial:
                        yield {"data": (
                            f"\n\n🚫 {_refused.get('provider', 'The provider')} refused this "
                            f"generation on content grounds, and it was still refused after "
                            f"re-running it. Nothing is wrong with the workflow.\n\n"
                            f"> {_refused.get('what_it_said', _why)}\n\n"
                            f"{_refused.get('what_to_do', '')}")}
                        self._record_chat_summary(user_text, synth, status="refused",
                                                  raw_json=self._last_brainbriefing_json)
                        self._record_agent_usage(self._orchestrator_agent, _snap)
                        self._session.last_agent = "orchestrator"
                        self._log_orchestrator()
                        return
                    if _partial:
                        # Some members succeeded — keep and report them; just flag
                        # the ones that couldn't be produced instead of hard-failing.
                        _lead = (f"were refused by {_refused.get('provider', 'the provider')} "
                                 f"on content grounds" if _refused else
                                 f"could not be healed (e.g. `{_nt}`: {_why})")
                        yield {"data": (f"\n\n⚠️ {len(_failed)} of the batch {_lead}; keeping "
                                        f"the {len(exec_paths) - _outputs_before} that "
                                        f"succeeded.")}
                        # fall through to the normal completion path below.
                    elif _keeplive_run:
                        # The user's own on-canvas graph errored — don't heal it;
                        # point them at the likely cause (often no output node).
                        yield {"data": (f"\n\n⚠️ Your canvas graph couldn't run — ComfyUI reported: "
                                        f"{_why}. The hook value was placed on the canvas; the graph "
                                        f"itself needs a fix (commonly a missing output node like "
                                        f"SaveImage/PreviewImage wired to the generator). I left your "
                                        f"graph untouched.")}
                        self._record_chat_summary(user_text, synth, status="failed",
                                                  raw_json=self._last_brainbriefing_json)
                        self._record_agent_usage(self._orchestrator_agent, _snap)
                        self._session.last_agent = "orchestrator"
                        self._log_orchestrator()
                        return
                    else:
                        yield {"data": (f"\n\n❌ ComfyUI run failed and could not be auto-healed "
                                        f"(error in `{_nt}`: {_why}). Stopping so you can take a look.")}
                        self._record_chat_summary(user_text, synth, status="failed",
                                                  raw_json=self._last_brainbriefing_json)
                        self._record_agent_usage(self._orchestrator_agent, _snap)
                        self._session.last_agent = "orchestrator"
                        self._log_orchestrator()
                        return

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
                    self._log_orchestrator()
                    return

                # A dry run's whole product is the account of what it would have
                # done, so it is stated by the runtime rather than left to the
                # agent's summary — which graphs were built, where they are, and
                # what each would have produced.
                if self._dry_run:
                    try:
                        from src.utils import dry_run as _dry_mod
                        _summary = _dry_mod.summary()
                    except Exception:  # noqa: BLE001
                        _summary = ""
                    if _summary and self._dry_graphed:
                        _summary += ("\n   Open in ComfyUI ▸ Workflows: "
                                     + ", ".join(self._dry_graphed))
                    yield {"data": "\n\n" + (_summary or (
                        "🧪 DRY RUN — nothing was submitted to ComfyUI, and nothing was "
                        "built either: no run was reached this turn."))}
                self._record_chat_summary(user_text, synth, status="completed",
                                          raw_json=self._last_brainbriefing_json)
                self._record_agent_usage(self._orchestrator_agent, _snap)
                self._session.last_agent = "orchestrator"
                self._log_orchestrator()
                self._learn_from_orchestrator_turn(_orch_msg_start)
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

    @staticmethod
    def _user_asked_for_subagent(text: str) -> bool:
        """True only when the user's message explicitly asks to use/spawn a subagent.

        Gates ``spawn_subagent`` so the orchestrator can't reach for it on routine
        turns. Matches an explicit ``subagent`` mention, or a spawn/use/delegate
        verb near the word ``agent``.
        """
        import re as _re
        t = (text or "").lower()
        if "subagent" in t or "sub-agent" in t or "sub agent" in t:
            return True
        return bool(_re.search(
            r"\b(spawn|spin\s*up|launch|create|use|delegate\s+to)\b[^.?!]{0,30}\bagents?\b", t))

    async def stream_async(self, user_input, *, qa_reply_queue: asyncio.Queue | None = None,
                           canvas_prompt: dict | None = None, canvas_hooks: list | None = None,
                           canvas_selection: list | None = None, qa_briefing=None,
                           dry_run: bool = False):  # noqa: ANN201
        """Async generator for one turn: the orchestrator owns the whole turn
        and streams its events (and those of the specialists it delegates to).

        Yields the same event dicts that a Strands Agent.stream_async would.
        """
        # Stamp this conversation onto every message-history record for this turn
        # so the log viewer can group records by conversation. session_id is the
        # thread id in the side panel (see _restore_state); "default"/CLI otherwise.
        try:
            set_log_thread(getattr(self._session, "session_id", "") or "")
        except Exception:  # noqa: BLE001
            pass

        # The orchestrator owns the whole turn — it routes natively by choosing
        # which specialist tool to call (no triage, no rigid router).
        _trace("pipeline.stream_async: orchestrator begin")
        # Gate spawn_subagent to an explicit user request only — a small
        # orchestrator model otherwise spins up subagents for routine work the
        # direct path already handles. Re-armed every turn from THIS message.
        try:
            from src.tools.orchestration import set_subagent_allowed as _set_sa
            _set_sa(self._user_asked_for_subagent(self._extract_text(user_input)))
        except Exception:  # noqa: BLE001
            pass
        self._orch_turn_logged = False  # reset per turn; _log_orchestrator sets it
        try:
            async for event in self._astream_orchestrator(
                user_input, qa_reply_queue=qa_reply_queue,
                canvas_prompt=canvas_prompt, canvas_hooks=canvas_hooks,
                canvas_selection=canvas_selection, qa_briefing=qa_briefing,
                dry_run=dry_run,
            ):
                yield event
        finally:
            # Log on ANY exit: normal completion, exception, or cancellation
            # (user /stop or a hang) — so every request lands in the log.
            self._log_orchestrator()
            # Disarm unconditionally: a dry run that died mid-turn must not leave
            # the next one refusing to submit anything.
            try:
                from src.utils import dry_run as _dry_mod
                _dry_mod.reset()
            except Exception:  # noqa: BLE001
                pass
            # Same reason: a gated turn that died mid-way still handed the ball to
            # the user, and must not leave the queue held shut for the next one.
            self._arm_plan_gate()
        _trace("pipeline.stream_async: orchestrator done")

    # ── Internal helpers ─────────────────────────────────────────────── #

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
        """Append newly produced media to the thread gallery (dedup by path).

        Called once per completed turn while ``current_output_paths`` still holds
        that turn's outputs. Each entry gets a 1-based index, a caption, and the
        turn number, so the user can later reference it ("image 2", "the last
        one"). Videos count: they are produced by the same runs and referred to
        the same way, and leaving them out meant "the second video" resolved to
        nothing at all.

        The caption prefers what the run was recorded as being FOR — the role
        stated in the hook's prompt, or the directive that produced it — over the
        brainbriefing, because a canvas-hook turn has no briefing and used to
        register every one of its outputs with an empty caption.
        """
        new_paths = [p for p in self._session.current_output_paths
                     if _is_image_file(p) or _is_video_file(p)]
        if not new_paths:
            return
        caption = self._caption_from_brief(raw_json or self._last_brainbriefing_json)
        existing = {gi.path for gi in self._session.generated_images}
        turn = len(self._session.chat_summaries)  # this turn's summary is already appended
        try:
            from src.utils.output_tags import role_for
        except Exception:  # noqa: BLE001
            role_for = None  # noqa: N806
        for p in new_paths:
            if p in existing:
                continue
            self._session.generated_images.append(
                GeneratedImage(
                    index=len(self._session.generated_images) + 1,
                    path=p,
                    caption=(role_for(p) if role_for else "") or caption,
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
            "[GENERATED IN THIS THREAD] — the user may reference these by number "
            "(\"image 2\"), recency (\"the last one\"), or description (\"the lighthouse "
            "one\"). Numbers are 1-based and ordered oldest→newest; the text after the "
            "dash is what each one was made FOR, so prefer it over looking again:\n"
            + "\n".join(lines)
            + "\n[When the user refers to one of these, use the matching path above as "
            "the file to act on — to analyse/describe it call analyze_image(path) (or "
            "analyze_video for a video); to use it as a workflow input upload it via "
            "upload_image(path). These are real files; never claim none is available.]"
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
            # A relevance floor keeps this always-on inject clean: only genuinely
            # related preferences/lessons surface, so trivial turns add no noise.
            #
            # Two scopes are merged:
            #   • GLOBAL (MEMORY_NAMESPACE): curated learnings + explicit notes —
            #     the shared long-term memory, recalled in every conversation.
            #   • THIS CONVERSATION (session id = thread): the auto request-log of
            #     what the agent did here — recalled only within this thread.
            # Legacy request-log lines that predate per-conversation scoping were
            # written to the global namespace; drop them (prefix "Generated:") so
            # old activity can't bleed across threads either.
            deliberate = [
                m for m in memory_search(user_text, session_id=MEMORY_NAMESPACE,
                                         limit=5, min_score=0.5)
                if not str(m.get("memory", "")).startswith("Generated:")
            ]
            own: list[dict] = []
            sid = getattr(self._session, "session_id", None)
            if sid and sid != MEMORY_NAMESPACE:
                own = memory_search(user_text, session_id=sid, limit=3, min_score=0.5)
            seen: set = set()
            merged: list[dict] = []
            for m in sorted(deliberate + own, key=lambda x: x.get("score", 0), reverse=True):
                mid = m.get("id")
                if mid in seen:
                    continue
                seen.add(mid)
                merged.append(m)
                if len(merged) >= 5:
                    break
            return format_memories(merged)
        except Exception as exc:
            if self._verbose:
                print(f"[memory] context retrieval error: {exc}")
            return ""

    def _get_project_memory_context(self) -> str:
        """Return the per-project memory block, or '' when there is nothing to say.

        Unrelated to *user_text*: this is state, not recall, so it is not searched
        or ranked — everything the project has established is either in force
        (technical settings, in full) or listed by name. The store resolves itself
        from the running ComfyUI's user directory on every call, which is what
        makes a project switch mid-session land without any bookkeeping here.
        """
        try:
            from src.utils.project_memory import render_context
            return render_context()
        except Exception as exc:  # noqa: BLE001
            if self._verbose:
                print(f"[project-memory] context error: {exc}")
            return ""

    def _apply_hook_cache(self) -> None:
        """Put back what the memorizing hooks answered last time, and release the rest.

        Runs once per turn, before the orchestrator sees anything: a hook whose
        ``memorize`` toggle is on and whose inputs are unchanged has its stored
        value injected straight into the graph, exactly as if the agent had just
        produced it — no vision call, no turn spent re-describing a picture that
        did not move. A hook with the toggle OFF drops whatever was stored under
        its current key, which is what makes the toggle the forget gesture.
        """
        hooks = [h for h in (self._canvas_hooks or []) if isinstance(h, dict)]
        if not hooks:
            return
        try:
            from src.utils.canvas_hooks import inject_produced_value
            from src.utils.hook_cache import fingerprint, forget, memorizing, read
        except Exception as exc:  # noqa: BLE001
            if self._verbose:
                print(f"[hook-cache] unavailable ({exc}).")
            return
        for h in hooks:
            try:
                key = fingerprint(h, self._canvas_base_prompt)
            except Exception as exc:  # noqa: BLE001
                if self._verbose:
                    print(f"[hook-cache] could not key hook {h.get('hook_node_id')}: {exc}")
                continue
            h["_cache_key"] = key
            if not memorizing(h):
                forget(key)
                continue
            value = str((read(key) or {}).get("value") or "")
            if not value:
                continue
            targets = (inject_produced_value(self._canvas_base_prompt, h, value)
                       if isinstance(self._canvas_base_prompt, dict) else [])
            h["_cached"] = {"value": value, "targets": targets,
                            "when": str((read(key) or {}).get("when") or "")}
            _push_progress(f"♻️ Hook {h.get('hook_node_id')} — reused the remembered value.")
            if self._verbose:
                print(f"[hook-cache] hook {h.get('hook_node_id')} hit {key} "
                      f"→ {len(targets)} target(s).")

    def _hook_output_role(self, hook: dict | None) -> str:
        """What outputs produced for *hook* should be recorded as.

        The user's own words when they stated a role in the hook's prompt; failing
        that the directive itself, trimmed — still better than a filename, and it
        is what the next turn reads off the node.
        """
        if not isinstance(hook, dict):
            return ""
        try:
            from src.utils.canvas_hooks import declared_output_role
            role = declared_output_role(hook)
        except Exception:  # noqa: BLE001
            role = ""
        if role:
            return role
        return " ".join(str(hook.get("directive") or "").split())[:80]

    def _tag_run_outputs(self, hook: dict | None = None, role: str = "") -> None:
        """Declare what the run about to start is for, before its files appear.

        Outputs are emitted the moment they land — by the server's pump, on
        another thread — so there is no later point at which "this batch was the
        shot start frames" is still known. ``declared`` records that the user
        named the role themselves in the hook's prompt, which is what earns an
        ``agentY ref note`` on the node dropped for it: decorating someone's
        canvas is a thing you do when asked, not by default.
        """
        try:
            from src.utils.canvas_hooks import declared_output_role
            from src.utils.output_tags import set_run_role
        except Exception:  # noqa: BLE001
            return
        declared = declared_output_role(hook) if hook else ""
        final = role or declared or self._hook_output_role(hook) \
            or self._caption_from_brief(self._last_brainbriefing_json)
        set_run_role(final,
                     declared=bool(declared or role),
                     hook=str((hook or {}).get("hook_node_id") or ""))

    @staticmethod
    def _variant_label(label: dict) -> str:
        """The one value that makes a variant itself — usually the prompt.

        A seed is what makes two variants of the SAME thing different; a prompt is
        what makes them different things. Names the variant after the latter, so
        "reference frame 3" reads as "Ben, grey suit, late 40s" wherever it turns
        up later — the node's title, its sidecar, the anchor line in the next turn.
        """
        best = ""
        for slot, val in (label or {}).items():
            param = str(slot).rsplit(".", 1)[-1].lower()
            if "seed" in param or not isinstance(val, str) or not val.strip():
                continue
            text = " ".join(val.split())
            if any(w in param for w in ("prompt", "text", "description", "caption")):
                return text[:70]
            best = best or text[:70]
        return best

    def _name_variants(self, paths: list, labels: list, hook: dict | None) -> None:
        """Give every member of a batch its own name, before any of it runs."""
        if not labels:
            return
        try:
            from src.utils.canvas_hooks import declared_output_role
            from src.utils.output_tags import set_workflow_role
        except Exception:  # noqa: BLE001
            return
        declared = declared_output_role(hook) if hook else ""
        for i, path in enumerate(paths):
            what = self._variant_label(labels[i] if i < len(labels) else {})
            if not what:
                continue
            role = f"{declared}: {what}" if declared else what
            set_workflow_role(path, role, hook=str((hook or {}).get("hook_node_id") or ""),
                              variant=i + 1, declared=bool(declared))

    @staticmethod
    def _variant_report(paths: list, labels: list, errors: dict | None = None) -> list:
        """Per-variant: what it was made from, and what came out of it.

        The pairing is the point. Without it the agent gets a flat list of files
        and has to assume they came back in the order they went in — which holds
        right up until one member fails, is healed, and is re-queued behind the
        others. A batch of character references that quietly transposes two of
        them produces a video that looks fine and stars the wrong people.
        """
        try:
            from src.utils.output_tags import outputs_of
        except Exception:  # noqa: BLE001
            def outputs_of(_p):  # noqa: ANN001, ANN202
                return []
        out = []
        for i, p in enumerate(paths):
            v = {"variant": i + 1, "workflow": p, "ok": p not in (errors or {})}
            if errors and p in errors:
                v["error"] = errors[p]
            if i < len(labels) and labels[i]:
                v["made_from"] = {k: (" ".join(str(val).split())[:90] if isinstance(val, str)
                                      else val) for k, val in labels[i].items()}
            produced = outputs_of(p)
            if produced:
                v["outputs"] = produced
            out.append(v)
        return out

    def _hook_for_targets(self, node_ids) -> dict | None:
        """The hook whose output feeds any of *node_ids* — who asked for this run."""
        wanted = {str(n) for n in node_ids if n is not None}
        for h in (self._canvas_hooks or []):
            if not isinstance(h, dict):
                continue
            for t in (h.get("targets") or []):
                if isinstance(t, dict) and str(t.get("node_id")) in wanted:
                    return h
        return None

    def _detect_plan_approval(self, user_text: str, project_ctx: str = ""):
        """Who, if anyone, asked to approve the plan before anything runs.

        Three places a standing "show me first" can live, in the order they beat
        each other: the user's own message this turn, the directives on the hook
        nodes they wired, and the project's memory. The user's message can also
        *waive* the rule — "just do it" — because a rule they set is theirs to
        suspend, and the alternative is a user arguing with their own hook node.
        """
        from src.utils.plan_gate import find_approval_request, waived
        if waived(user_text):
            return None
        sources = [("the user's message", user_text)]
        for h in (self._canvas_hooks or []):
            if isinstance(h, dict):
                sources.append((f"hook {h.get('hook_node_id')}'s directive",
                                str(h.get("directive") or "")))
        if project_ctx:
            sources.append(("the project's memory", project_ctx))
        return find_approval_request(sources)

    def _plan_gate_refusal(self, announce: bool = True) -> dict | None:
        """The refusal a run tool returns while the plan is still awaiting a yes.

        Open by default: this is only ever closed when :meth:`_detect_plan_approval`
        found someone asking to be asked, and it re-opens for one turn as soon as
        the user has spoken again (see :meth:`_arm_plan_gate`). Pass
        ``announce=False`` to build the payload without saying anything — the hold
        is seeded at the start of every gated turn, whether or not it ever fires.
        """
        req = getattr(self, "_plan_approval", None)
        if req is None or getattr(self, "_plan_gate_open", False):
            return None
        from src.utils.plan_gate import execution_refusal
        if announce:
            self._plan_gate_fired = True
            _push_progress("✋ The plan was asked to be approved first — holding.")
        return execution_refusal(req)

    def _arm_plan_gate(self) -> None:
        """At the end of a turn, decide whether the next one may execute.

        Only a gate that actually *stopped* something arms the next turn: work was
        wanted, the user was handed the plan instead, and their answer — in
        whatever words — is what releases it. A gated turn that never tried to run
        anything (a question, a chat about the graph) has put no plan to anyone and
        leaves the gate shut. Neither does a turn that ran with the gate open: the
        next request is new work, and the standing "ask me first" covers it too.

        The clean path doesn't need this at all — a reply that is just "yes" or
        "go ahead" reads as the approval it is (:func:`plan_gate.waived`). This is
        the backstop for the ones that don't, so an unusual "sounds perfect, ship
        it" can never leave the user re-approving the same plan forever.
        """
        try:
            from src.utils.workflow_signal import hold_fired, set_execution_hold
            fired = bool(getattr(self, "_plan_gate_fired", False)) or hold_fired()
            set_execution_hold(None)
            if fired and getattr(self, "_plan_approval", None) is not None \
                    and not getattr(self, "_plan_gate_open", False):
                self._session.plan_awaiting_reply = True
        except Exception:  # noqa: BLE001
            pass

    def _auto_save_memory(self, user_text: str, raw_json: str) -> None:
        """Append one trimmed, verbatim request-log line to long-term memory.

        Records *which template + resolution* a completed run used, with a short
        intent snippet, so the agent can recall what it did earlier in THIS
        conversation. Deliberately compact — the full prompt is NOT stored (it was
        the noisiest part of the old telemetry). Written under the current
        conversation's session id (NOT the global namespace) and tagged
        ``source=request_log`` so it is recalled only within this conversation and
        never bleeds into other threads — unlike curated learnings and explicit
        notes, which are the shared long-term memory. Best-effort.
        """
        try:
            data = json.loads(raw_json)
            task_desc = data.get("task", {}).get("description", "")
            template_name = data.get("template", {}).get("name") or ""
            width = data.get("resolution_width")
            height = data.get("resolution_height")

            parts: list[str] = []
            if template_name:
                parts.append(f"template '{template_name}'")
            if width and height:
                parts.append(f"{width}x{height}")
            if task_desc:
                short = task_desc[:80].rstrip()
                if len(task_desc) > 80:
                    short += "…"
                parts.append(short)

            if not parts:
                return

            memory_text = "Generated: " + ", ".join(parts) + "."
            # Per-conversation: scope to this thread's session id so it doesn't
            # bleed into other conversations. Falls back to the global namespace
            # only for the CLI (no thread), where there's a single session anyway.
            _sid = getattr(self._session, "session_id", None) or MEMORY_NAMESPACE
            memory_add(memory_text, session_id=_sid, metadata={"source": "request_log"})
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
    # How many times to reject a hallucinated (non-catalog) template name before
    # falling back to build_new. Common after a history-clear + constrain fallback.
    _MAX_BAD_TEMPLATE_RETRIES = 2
    # Wall-clock cap for one fix_workflow_assembly repair turn.
    _FIX_ASSEMBLY_TIMEOUT = 120.0
    # Wall-clock cap for one generate_new_workflow (build-from-scratch) turn.
    _GENERATE_WORKFLOW_TIMEOUT = 240.0
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

            Pick the template and write the prompt; output the decision JSON.
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

        # Inject the template list (hybrid: bare names + a short note for cryptic
        # ones) as the FRONT (stable) prefix so the Researcher picks an EXACT real
        # name without a tool call and cannot hallucinate one. Front placement keeps
        # it a cacheable prefix (the variable memory/request follow it).
        # Staged/earlier images are part of the scope key: "now make a video from
        # it" names no input, but an image on hand still rules the media guess.
        _staged = "image" if (current_has_images or self._session.last_user_input_images) else ""
        catalog_block = self._format_template_catalog(user_text, _staged)
        if catalog_block:
            prompt = catalog_block + "\n\n" + prompt

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

    @staticmethod
    def _openai_compat_endpoint(provider: str) -> tuple[str, str] | None:
        """Resolve ``(base_url, api_key)`` for an OpenAI-compatible provider token,
        mirroring ``agent._make_agent``. Returns ``None`` for backends that are not
        OpenAI-compatible (e.g. ``ollama``, ``anthropic``)."""
        llm = (_settings().get("llm") or {})
        p = provider.strip().lower()
        if p in {"dashscope", "modelstudio", "qwen", "alibaba"}:
            base_url = os.environ.get("DASHSCOPE_BASE_URL") or str(
                (llm.get("dashscope") or {}).get(
                    "base_url", "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"))
            api_key = os.environ.get("DASHSCOPE_API_KEY") or os.environ.get("ALIBABA_API_KEY") or ""
            return base_url, api_key
        if p in {"openai", "gpt"}:
            base_url = os.environ.get("OPENAI_BASE_URL") or str(
                (llm.get("openai") or {}).get("base_url", "https://api.openai.com/v1"))
            return base_url, (os.environ.get("OPENAI_API_KEY") or "")
        if p in {"google", "gemini"}:
            base_url = os.environ.get("GEMINI_BASE_URL") or str(
                (llm.get("google") or {}).get(
                    "base_url", "https://generativelanguage.googleapis.com/v1beta/openai/"))
            api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY") or ""
            return base_url, api_key
        return None

    def _constrain_briefing(self, user_request: str, messages: list) -> str | None:
        """Force a schema-valid ResearcherDecision JSON via a constrained, tool-free
        model call that cannot spiral. Uses the researcher's own gathered context
        (chosen template, image descriptions) so the output is a REAL decision for
        the request, not a stub. The pipeline assembles the full briefing from it.

        Drives whichever backend the researcher (``pipeline.query_templates``) uses:
        an OpenAI-compatible endpoint (DashScope/Qwen, OpenAI, Gemini) via
        ``response_format={"type": "json_object"}`` with the schema embedded in the
        system prompt. Returns the raw JSON string, or ``None`` when the backend is
        unsupported or the call fails."""
        pipe = ((_settings().get("llm") or {}).get("pipeline") or {})
        spec = str(pipe.get("query_templates") or pipe.get("researcher") or "").strip()
        provider, _, model_id = spec.partition(",")
        provider, model_id = provider.strip().lower(), model_id.strip()
        if not model_id:
            return None
        endpoint = self._openai_compat_endpoint(provider)
        if endpoint is None:
            return None  # non-OpenAI-compatible backend — skip the net
        base_url, api_key = endpoint
        if not api_key:
            return None
        ctx = self._flatten_researcher_context(messages)
        schema = ResearcherDecision.model_json_schema()
        # Inject the template catalog so the constrained call can ONLY pick a real
        # template name (history clears can wipe the catalog the researcher fetched,
        # otherwise inviting a hallucinated name that the scaffold then blocks on).
        _cat_block = self._format_template_catalog(user_request)
        catalog_hint = ("\n\n" + _cat_block) if _cat_block else ""
        sys_msg = (
            "You finalise the researcher's work into ONE decision JSON conforming "
            "to the schema: the chosen template name, the authored prompt, and task "
            "metadata ONLY — no node bindings or paths (those are added later). Use "
            "the USER REQUEST and RESEARCH CONTEXT (tool results: the chosen template "
            "name, any image description). status MUST be 'ready' (use 'blocked' only "
            "if no template fits or the request is truly unclear). Use the real "
            "template name and the actual prompt text from the request. Never "
            "describe this instruction — output only the JSON decision for the "
            "request. No prose, no markdown.\n\nThe JSON MUST validate against this "
            "JSON Schema:\n" + json.dumps(schema)
        )
        user_msg = (
            f"USER REQUEST:\n{user_request[:2000]}\n\n"
            f"RESEARCH CONTEXT (researcher's tool calls + results):\n{ctx}"
            f"{catalog_hint}\n\n"
            "Output the decision JSON for the user request now."
        )
        try:
            from openai import OpenAI  # noqa: PLC0415
            client = OpenAI(base_url=base_url, api_key=api_key)
            r = client.chat.completions.create(
                model=model_id,
                messages=[
                    {"role": "system", "content": sys_msg},
                    {"role": "user", "content": user_msg},
                ],
                response_format={"type": "json_object"},
                temperature=0.1,
                max_tokens=6144,
            )
            return r.choices[0].message.content
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

    @staticmethod
    def _catalog_hint(desc) -> str:
        """A short one-line task hint for a template (first sentence, tag-stripped)."""
        d = re.sub(r"^\[[^\]]*\]\s*", "", str(desc)).split(". ")[0].strip()
        return d[:60]

    def _format_template_catalog(self, request: str = "", staged: str = "") -> str:
        """Render the template catalog injected into the Researcher's prompt so it
        selects an EXACT real name with no tool call (and cannot hallucinate one).

        Prefers a recipe-derived catalog organised ``task → model → template`` (the
        two axes users actually specify), falling back to a flat ``name: hint`` list
        if the recipe DB is unavailable.

        Given a *request*, the tree is narrowed to the scope its
        (execution, media, model) key resolves to — the whole corpus is ~17k tokens
        and a keyed scope is a few hundred. Cached per resolved scope, so a repeated
        scope within a session stays a stable, cacheable prompt prefix."""
        scope_key, tasks, note = "", None, ""
        if request:
            execution, media, model = self.resolve_catalog_scope(request, staged)
            named_tasks = self._resolve_tasks(request, media, staged)
            scope_key, tasks, note = self._scope_recipes(
                execution, media, model, named_tasks)
        cache = getattr(self, "_catalog_block_cache", None)
        if not isinstance(cache, dict):
            cache = self._catalog_block_cache = {}
        if scope_key not in cache:
            if tasks is None and scope_key:
                block = self._format_catalog_index()   # nothing resolved: compact index
            else:
                block = self._format_recipe_catalog(tasks, scope_note=note)
            if not block:
                block = self._format_flat_catalog()    # fallback: flat name: hint list
            cache[scope_key] = block
        return cache[scope_key]

    # --------------------------------------------------------------------- #
    # Catalog scoping
    #
    # The recipe DB already carries the three axes a request pins down —
    # execution (partner-API vs local), media, and model — so the scope can be
    # resolved deterministically, with no extra model call and no drill-down
    # round trips. Each axis is a widening fallback, never a hard gate: a miss
    # costs tokens, not reachability, and get_workflow_catalog remains available
    # for the full inventory.
    # --------------------------------------------------------------------- #
    _MEDIA_RE = (
        ("video", re.compile(r"video|clip|animation|animate|footage|motion|i2v|t2v|v2v|flf")),
        ("3d", re.compile(r"\b3 d\b|mesh|glb|splat|gaussian")),
        ("audio", re.compile(r"audio|music|song|sound|voice|speech|tts")),
        ("image", re.compile(r"image|picture|photo|still|portrait|poster|upscale|thumbnail")),
    )
    # Partner-API models are the default (documented in the catalog header); only
    # an explicit ask for local/offline flips it.
    _LOCAL_RE = re.compile(r"local|offline|on my machine|without api|no api|self hosted")

    @staticmethod
    def _norm_request(text: str) -> str:
        """Space-normalised view for keyword matching, with letter/digit runs split
        so "wan2.2", "WAN 2.2" and "wan-2-2" all read as "wan 2 2".

        Deliberately NOT a separator-stripped view: stripping fuses word
        boundaries, so "an image" would match the model "Anima"."""
        s = str(text).lower()
        s = re.sub(r"([a-z])(\d)", r"\1 \2", s)
        s = re.sub(r"(\d)([a-z])", r"\1 \2", s)
        return " " + " ".join(re.sub(r"[^a-z0-9]+", " ", s).split()) + " "

    def _catalog_models(self) -> list:
        """Model names the recipe DB knows, longest-first so "WAN 2.2" is matched
        before "WAN". ``Generic`` is excluded — it is the catch-all the grouper
        falls back to when a template's text names no family, so it identifies
        nothing and would swallow any request."""
        if getattr(self, "_catalog_models_cache", None) is None:
            names = {str(m.get("model") or "") for t in self._load_recipe_tasks()
                     for m in (t.get("models") or [])}
            names.discard("")
            names.discard("Generic")
            self._catalog_models_cache = sorted(
                names, key=lambda n: (-len(self._norm_request(n).split()), -len(n), n))
        return self._catalog_models_cache

    def resolve_catalog_scope(self, request: str, staged: str = "") -> tuple:
        """Resolve *request* to ``(execution, media, model)``. Any part may be None."""
        q = self._norm_request(f"{request} {staged}")
        execution = "local" if self._LOCAL_RE.search(q) else "api"
        media = next((m for m, rx in self._MEDIA_RE if rx.search(q)), None)
        model = next((m for m in self._catalog_models()
                      if f" {self._norm_request(m).strip()} " in q), None)
        return execution, media, model

    # Request wording → the task-name token it implies. Only *distinctive* verbs
    # go here: matching on a task's own generic words is what made a naive
    # matcher resolve "make a video" to "Video to Video" (both name tokens are
    # "video") instead of "Text to Video".
    # Single words are matched as *stems* against word starts, so "upscale"
    # catches upscaling/upscaled; multi-word entries are matched as phrases.
    # Deliberately excludes vague verbs (change, modify, make, do): they would
    # pull half the corpus into a scope the user never asked for, and the media
    # bucket is a better answer than a confident wrong task.
    _TASK_KEYWORDS: dict = {
        "upscale": ("upscal", "upres", "up res", "enlarg", "higher resolution",
                    "more resolution", "4 k", "8 k"),
        "inpaint": ("inpaint", "in paint", "remove object", "erase", "patch out",
                    "clean plate"),
        "outpaint": ("outpaint", "out paint", "extend the frame", "extend the canvas",
                     "expand the frame", "widen the shot"),
        "edit": ("edit", "restyl", "relight", "retouch", "recolour", "recolor"),
        "character": ("character sheet", "turnaround", "consistent character",
                      "same character", "character consistency"),
        "controlnet": ("controlnet", "control net", "canny", "openpose", "scribble"),
        "preprocessors": ("depth map", "pose map", "segmentation", "estimate depth",
                          "preprocess"),
        "tools": ("crop", "stitch", "concat", "contact sheet"),
        "audio": ("audio", "music", "song", "voice", "speech"),
        # "First / Last Frame to Video" is not spelled "<in> to <out>", so the
        # direction matcher below can never reach it — without these it was only
        # reachable by naming a model, and a plain "video from a start and an end
        # frame" resolved to Text to Video, hiding every flf2v template there is.
        "frame": ("first frame", "start frame", "end frame", "first and last",
                  "start and end", "first last", "flf"),
    }
    # Generic words in task names that must never carry a match on their own.
    _TASK_STOPWORDS: frozenset = frozenset({
        "api", "partner", "nodes", "to", "and", "with", "from", "the", "of",
        "image", "video", "text", "audio", "3", "d",
    })

    @staticmethod
    def _task_base(task_name: str) -> str:
        """Task name without the 'API / Partner Nodes - ' prefix, so the API and
        local variants of one capability match together."""
        return task_name.split(" - ", 1)[1] if task_name.startswith("API / ") else task_name

    def _tasks_naming_templates(self, request: str) -> list:
        """Tasks owning any template the request names outright.

        The orchestrator routinely pins a template by name ("use
        api_seedance2_0_flf2v…"), which is the strongest signal there is — and
        the one both matchers below miss, because a template name is not a task
        name. Without this the researcher could be handed a scope that excludes
        the very template it was just told to use, and would report that
        template as not existing.
        """
        q = (request or "").lower()
        if not q:
            return []

        def _names(f: str) -> bool:
            """Is *f* in the request as a whole name, not inside a longer one?

            Plain substring matching is not enough: ``api_bytedance_seed`` sits
            inside ``api_bytedance_seedance1_5_flf2v``, so asking for the second
            would drag in the first one's task as well. A following letter,
            digit or underscore means we matched a prefix of some other
            template; trailing punctuation (a full stop, a comma) is a real
            boundary and must still count.
            """
            fl = str(f).lower()
            # The length floor keeps a short, generic template name from
            # matching ordinary prose.
            if len(fl) < 8:
                return False
            at = q.find(fl)
            while at >= 0:
                before = q[at - 1] if at else " "
                after = q[at + len(fl)] if at + len(fl) < len(q) else " "
                if not (before.isalnum() or before == "_") and \
                   not (after.isalnum() or after == "_"):
                    return True
                at = q.find(fl, at + 1)
            return False

        named = []
        for t in self._load_recipe_tasks():
            for m in (t.get("models") or []):
                if any(_names(f) for f in (m.get("member_files") or [])):
                    named.append(t.get("task") or "")
                    break
        return named

    def _resolve_tasks(self, request: str, media, staged: str = "") -> list:
        """Task names the request plausibly asks for — [] when it pins none.

        Three deterministic signals, strongest first:

        * **A template named outright.** Nothing outranks the request naming a
          template that exists — see ``_tasks_naming_templates``.

        * **Direction.** A task named "<in> to <out>" is a match when the media
          the user *has* and the media they *want* line up. What they have comes
          from the staged inputs, so "make a video" with nothing staged reads as
          Text to Video, while the same words with an image on the canvas read as
          Image to Video. Direction alone was the naive matcher's blind spot.
        * **Distinctive wording.** ``_TASK_KEYWORDS`` maps verbs users actually
          type onto a task-name token, so "which image upscaling templates do I
          have" reaches Upscale without naming it.

        A task must win on one of the three; anything vaguer falls through to the
        media bucket rather than guessing a task and hiding the rest.
        """
        by_name = self._tasks_naming_templates(request)
        if by_name:
            return by_name
        q = self._norm_request(f"{request} {staged}")
        words = q.split()
        in_media = "image" if "image" in self._norm_request(staged) else "text"

        def _mentions(term: str) -> bool:
            if " " in term:                       # phrase: match as written
                return f" {term} " in q
            return any(w.startswith(term) for w in words)   # stem

        by_keyword, by_direction = [], []
        for t in self._load_recipe_tasks():
            name = t.get("task") or ""
            base = self._norm_request(self._task_base(name)).strip()
            tokens = {w for w in base.split() if w not in self._TASK_STOPWORDS}
            if any(_mentions(term) for tok in tokens
                   for term in self._TASK_KEYWORDS.get(tok, ())):
                by_keyword.append(name)
            elif media and base == f"{in_media} to {media}":
                by_direction.append(name)
        # Naming a capability beats inferring one from media direction: "which
        # image upscaling templates do I have" is about Upscale, and reading it
        # as Text to Image (nothing staged, image wanted) would hide the answer.
        return by_keyword or by_direction

    def _scope_recipes(self, execution, media, model, named_tasks=()) -> tuple:
        """Narrowest useful scope for the resolved key: ``(cache_key, tasks, note)``.

        ``tasks`` is the recipe tree filtered to that scope, or None when nothing
        resolved (the caller then falls back to the compact index)."""
        def _pick(keep) -> list:
            out = []
            for t in self._load_recipe_tasks():
                models = [m for m in (t.get("models") or []) if keep(m)]
                if models:
                    out.append({**t, "models": models})
            return out

        if model:
            # A named model outranks the API-first default and the media guess:
            # "build me a wan 2.2 workflow" must not resolve to the one partner-API
            # WAN 2.2 recipe while hiding the six local ones, and "relight this shot
            # with magnific" reads as video though Magnific is image-only.
            tasks = _pick(lambda m: str(m.get("model") or "") == model)
            if tasks:
                return f"model:{model}", tasks, f"model “{model}”"
        if named_tasks:
            # Between the model key and the media bucket: the request named a
            # capability but no model. Both the partner-API and local variants of
            # the task are kept — which to prefer is the researcher's call, and
            # the API ones already sort first.
            wanted = set(named_tasks)
            tasks = [t for t in self._load_recipe_tasks() if (t.get("task") or "") in wanted]
            if tasks:
                labels = sorted({self._task_base(t.get("task") or "") for t in tasks})
                key = "task:" + "+".join(sorted(wanted))
                return key, tasks, "the " + " / ".join(labels) + " workflows"
        if media:
            want_api = execution == "api"
            tasks = _pick(lambda m: bool(m.get("uses_api_nodes")) == want_api
                          and ((m.get("user_intent") or {}).get("media")) == media)
            if tasks:
                label = "partner-API" if want_api else "local"
                return f"{execution}:{media}", tasks, f"{label} {media} workflows"
        return "index", None, ""

    def _format_catalog_index(self) -> str:
        """Compact ``task → model names`` index, no template leaves.

        The floor for a request that pins nothing down: ~900 tokens for the whole
        corpus, where rendering every leaf would be ~17k."""
        tasks = self._load_recipe_tasks()
        if not tasks:
            return ""
        lines = []
        for t in tasks:
            models = sorted((m for m in (t.get("models") or [])),
                            key=lambda m: (not m.get("uses_api_nodes"), str(m.get("model") or "")))
            names = ", ".join(("[API] " if m.get("uses_api_nodes") else "")
                              + str(m.get("model") or "?") for m in models)
            if names:
                lines.append(f"- {t.get('task')}: {names}")
        if not lines:
            return ""
        return ("AVAILABLE WORKFLOWS — grouped task → model. This is an index, not "
                "the template list: it names no template. Call get_workflow_catalog "
                "for the full inventory, or ask the user which task and model they "
                "mean.\n" + "\n".join(lines))

    def _load_recipe_tasks(self) -> list:
        """The recipe DB's ``tasks`` list, loaded from the corpus (cached per pipeline)."""
        if getattr(self, "_recipe_tasks_cache", None) is None:
            tasks: list = []
            try:
                from agenty_core.paths import corpus_root  # noqa: PLC0415
                p = corpus_root() / "config" / "workflow_recipes.json"
                db = json.loads(p.read_text(encoding="utf-8"))
                if isinstance(db, dict):
                    tasks = db.get("tasks") or []
            except Exception:  # noqa: BLE001
                tasks = []
            self._recipe_tasks_cache = tasks
        return self._recipe_tasks_cache

    # Trailing numeric version token: ``_2511`` (build date) or ``_2_3`` (semver-ish).
    _VERSION_TOKEN_RE = re.compile(r"_(\d{3,4}|\d+_\d+)$")

    # Templates hidden from the researcher's catalog: each is redundant with a
    # concrete sibling in the same task, so suppressing it tightens retrieval
    # without losing any capability. They stay on disk and remain resolvable if
    # referenced by exact name (e.g. the user explicitly asks for one).
    _SUPPRESSED_TEMPLATES = frozenset({
        # Abstract "… blueprint" skeletons — a concrete model template always exists:
        "image_edit",                 # → image_edit_qwen_2511, image_edit_flux_2_dev, …
        "image_to_video",             # → image_to_video_wan_2_2, …
        "image_to_depth_map_lotus",   # → image_depth_estimation_lotus_depth
        "video_inpaint_wan2_1_vace",  # → video_inpainting_wan2_1_vace
        # Exact duplicates filed under a different filename:
        "text_to_image",              # generic Z-Image-Turbo t2i → text_to_image_z_image_turbo
        "image_z_image_turbo",        # dup of text_to_image_z_image_turbo
    })

    @classmethod
    def _capability_key(cls, stem: str) -> tuple[str, tuple]:
        """Split a template stem into ``(capability_key, version_tuple)`` by stripping
        a trailing numeric version token (``_2511``, ``_2_3``). Members sharing a
        capability_key are version-variants of the SAME workflow, and the highest
        version_tuple is the latest. Word suffixes like ``_base``/``_turbo``/``_4b``
        are deliberately NOT treated as versions — they denote distinct models or
        capabilities and must remain separately selectable."""
        m = cls._VERSION_TOKEN_RE.search(stem)
        if not m:
            return stem, ()
        ver = tuple(int(x) for x in re.findall(r"\d+", m.group(1)))
        return stem[: m.start()], ver

    def _format_recipe_catalog(self, tasks: list | None = None, scope_note: str = "") -> str:
        """Render the catalog from the recipe DB, grouped ``task → model → template``.

        Collapsing rule (agreed design): within a ``(task, model)`` group, pure
        version-variants of one workflow collapse to the LATEST; genuinely distinct
        capabilities (media splits, control types, separate tools) stay as separate
        leaves. Members not present in the live catalog (stale recipe entries) are
        dropped. Models that run via API/partner nodes are flagged ``[API]`` and
        listed first, so a request that names no model defaults to API-first.

        *tasks* defaults to the whole recipe tree; pass a filtered one (with
        *scope_note* naming the filter) to render only a resolved scope. Scoped
        renders drop the "Other" catch-all — it exists to keep every template
        reachable in a full render, and re-listing what the scope just excluded
        would defeat the scoping entirely.

        Returns ``""`` when the recipe DB or live catalog is unavailable, letting the
        caller fall back to the flat catalog."""
        tasks = self._load_recipe_tasks() if tasks is None else tasks
        if not tasks:
            return ""
        # Live catalog: authoritative {name: description}. Drives the stale-member
        # intersection and supplies each leaf's one-line hint.
        try:
            from src.tools import get_workflow_catalog as _gwc  # noqa: PLC0415
            live = json.loads(getattr(_gwc, "func", _gwc)())
            if not isinstance(live, dict) or not live:
                return ""
        except Exception:  # noqa: BLE001
            return ""

        def _stem(f: str) -> str:
            return f[:-5] if f.endswith(".json") else f

        emitted: set[str] = set()   # leaves shown (latest version of each capability)
        hidden: set[str] = set()    # older versions deliberately collapsed away
        blocks: list[str] = []
        for task in tasks:
            tname = task.get("task") or ""
            # API/partner models first (API-first default for an unnamed model).
            models = sorted(
                task.get("models") or [],
                key=lambda m: (not m.get("uses_api_nodes"), str(m.get("model") or "")),
            )
            model_lines: list[str] = []
            for m in models:
                members = [_stem(f) for f in (m.get("member_files") or [])]
                members = [s for s in members if s in live]  # drop stale
                if not members:
                    continue
                # Collapse version-variants: keep the latest per capability key.
                best: dict[str, tuple[tuple, str]] = {}
                for s in members:
                    key, ver = self._capability_key(s)
                    if key not in best or ver > best[key][0]:
                        best[key] = (ver, s)
                latest = {v[1] for v in best.values()}
                hidden.update(s for s in members if s not in latest)  # collapsed older versions
                # Drop suppressed templates (blueprints / exact dups).
                leaves = sorted(s for s in latest if s not in self._SUPPRESSED_TEMPLATES)
                if not leaves:
                    continue
                emitted.update(leaves)
                tag = " [API]" if m.get("uses_api_nodes") else ""
                leaf_txt = "\n".join(
                    f"    - {s}: {self._catalog_hint(live.get(s, ''))}" for s in leaves
                )
                model_lines.append(f"  {m.get('model') or '?'}{tag}:\n{leaf_txt}")
            if model_lines:
                blocks.append(f"## {tname}\n" + "\n".join(model_lines))
        if not blocks:
            return ""
        # Lossless safety net: any live template that no recipe references (recipe/
        # catalog drift) would otherwise be unreachable. List it so it stays
        # selectable — but never resurrect a version we deliberately collapsed.
        # Skipped for a scoped render: there the exclusion is the point, and
        # get_workflow_catalog is still there when the scope turns out too narrow.
        if not scope_note:
            uncovered = sorted(set(live) - emitted - hidden - self._SUPPRESSED_TEMPLATES)
            if uncovered:
                other = "\n".join(
                    f"    - {s}: {self._catalog_hint(live.get(s, ''))}" for s in uncovered
                )
                blocks.append("## Other\n  (uncategorised):\n" + other)
        header = (
            "AVAILABLE TEMPLATES — set template.name to EXACTLY one of the template "
            "names below (the leaf after each model), or \"build_new\" if none fit. "
            "They are grouped by task → model. Match the model the user named; if the "
            "user names no model (and does not ask for a local/offline workflow), "
            "prefer the API/partner option — the '[API]' models and the "
            "'API / Partner Nodes - …' task groups. Only the latest version of each "
            "workflow is listed; if the user explicitly names an older version, you "
            "may still use that exact template name.\n"
        )
        if scope_note:
            header += (
                f"This list is filtered to {scope_note} because the request named it. "
                "If none of these fit, call get_workflow_catalog for the full "
                "inventory before falling back to \"build_new\".\n"
            )
        return header + "\n".join(blocks)

    def _format_flat_catalog(self) -> str:
        """Fallback catalog: a flat ``- name: <first-sentence hint>`` list from the
        live catalog. Used only when the recipe DB is unavailable."""
        block = ""
        try:
            from src.tools import get_workflow_catalog as _gwc  # noqa: PLC0415
            cat = json.loads(getattr(_gwc, "func", _gwc)())
            if isinstance(cat, dict) and cat:
                lines = [f"- {name}: {self._catalog_hint(cat[name])}"
                         for name in sorted(cat)
                         if name not in self._SUPPRESSED_TEMPLATES]
                block = ("AVAILABLE TEMPLATES — set template.name to EXACTLY one of "
                         "these names (or \"build_new\" if none fit); each line is "
                         "name: what it does:\n" + "\n".join(lines))
        except Exception:  # noqa: BLE001
            block = ""
        return block

    def _catalog_names(self) -> list[str]:
        """Sorted template names from get_workflow_catalog (cached per pipeline)."""
        if getattr(self, "_catalog_names_cache", None) is None:
            names: list[str] = []
            try:
                from src.tools import get_workflow_catalog as _gwc  # noqa: PLC0415
                cat = json.loads(getattr(_gwc, "func", _gwc)())
                if isinstance(cat, dict):
                    names = sorted(cat.keys())
            except Exception:  # noqa: BLE001
                names = []
            self._catalog_names_cache = names
        return self._catalog_names_cache

    @staticmethod
    def _name_tokens(name: str) -> set:
        _common = {"api", "image", "video", "text", "to", "the", "of", "and",
                   "workflow", "dev", "template", "gen", "generation", "using",
                   "model", "new", "base", "simple", "basic", "default"}
        return set(re.findall(r"[a-z0-9]+", name.lower())) - _common

    def _match_template(self, name: str) -> str | None:
        """Resolve *name* to a real catalog template (exact, case-insensitive, then
        token-overlap fuzzy ≥ 0.6). Returns ``None`` when nothing plausibly matches
        (i.e. the name is hallucinated). Trusts the name if the catalog is empty."""
        names = self._catalog_names()
        if not names:
            return name
        if name in names:
            return name
        low = {n.lower(): n for n in names}
        if name.lower() in low:
            return low[name.lower()]
        q = self._name_tokens(name)
        if not q:
            return None
        best, bscore = None, 0.0
        for n in names:
            t = self._name_tokens(n)
            if not t:
                continue
            sc = len(q & t) / min(len(q), len(t))
            if sc > bscore:
                best, bscore = n, sc
        return best if bscore >= 0.6 else None

    def _closest_catalog_names(self, name: str, k: int = 8) -> list[str]:
        """The k catalog names sharing the most tokens with *name* (for a nudge)."""
        q = self._name_tokens(name)
        scored = sorted(self._catalog_names(),
                        key=lambda n: len(q & self._name_tokens(n)), reverse=True)
        return scored[:k]

    def _assemble_briefing(self, decision: "ResearcherDecision",
                           staged_inputs: list | None) -> "BrainBriefing":
        """Merge the Researcher's thin decision with the deterministic scaffold
        into a full :class:`BrainBriefing` (Option B).

        The scaffold owns every mechanical field (input/output/prompt node
        bindings, paths, model checks); the decision owns the authored prompt,
        task, template, and batch metadata. Missing-model blockers surface here
        (via the scaffold's ``check_model``). ``build_new`` and any scaffold
        failure fall back to a briefing built from the decision alone.
        """
        tname = (decision.template.name or "").strip()
        res = ({"width": decision.resolution_width, "height": decision.resolution_height}
               if decision.resolution_width and decision.resolution_height else None)
        sc: dict | None = None
        if tname and tname.lower() not in ("build_new", "none"):
            try:
                from src.tools.briefing_scaffold import build_briefing_scaffold  # noqa: PLC0415
                sc = build_briefing_scaffold(
                    tname, staged_inputs=staged_inputs, task_type=decision.task.type,
                    task_description=decision.task.description, count_iter=decision.count_iter,
                    variations=decision.variations, resolution=res)
            except Exception as exc:  # noqa: BLE001
                if self._verbose:
                    print(f"pipeline: scaffold assembly failed ({exc}); "
                          f"building briefing from decision alone.")
                sc = None

        if sc is None:
            # build_new / scaffold unavailable: minimal briefing from the decision.
            sc = {
                "input_images": [], "input_nodes": [], "input_image_count": 0,
                "output_nodes": [], "resolution_width": (res or {}).get("width"),
                "resolution_height": (res or {}).get("height"), "prompt_nodes": [],
                "positive_prompt_node_id": None, "blockers": [],
            }
        sc.pop("_scaffold_meta", None)

        # Merge the LLM's authored fields over the scaffold skeleton.
        sc["template"] = {"name": (tname or "build_new")}
        sc["task"] = {"type": decision.task.type, "description": decision.task.description}
        sc["prompt"] = {"positive": decision.prompt.positive, "negative": decision.prompt.negative}
        sc["count_iter"] = decision.count_iter
        sc["variations"] = decision.variations
        # positive_prompt_node_id only matters for per-variation splicing.
        if not (decision.variations or decision.count_iter > 1):
            sc["positive_prompt_node_id"] = None

        # Blockers: request-level (from the LLM) ∪ real model blockers (from the
        # scaffold). WARNING-prefixed scaffold notes stay non-fatal.
        scaffold_blockers = [b for b in sc.get("blockers", [])
                             if b and not str(b).startswith("WARNING")]
        warnings = [b for b in sc.get("blockers", []) if str(b).startswith("WARNING")]
        request_blockers = list(decision.blockers or [])
        real_blockers = request_blockers + [b for b in scaffold_blockers if b not in request_blockers]
        sc["blockers"] = real_blockers + warnings
        sc["status"] = "blocked" if (real_blockers or decision.status == "blocked") else "ready"

        briefing = BrainBriefing.model_validate(sc)
        if self._verbose:
            print(f"pipeline: briefing assembled (template={briefing.template.name}, "
                  f"in={len(briefing.input_nodes)} out={len(briefing.output_nodes)} "
                  f"pos_node={briefing.positive_prompt_node_id} status={briefing.status}).")
        return briefing

    async def _assemble_deterministic(self, briefing: "BrainBriefing") -> dict:
        """Assemble the workflow from a briefing WITHOUT an LLM on the happy path.

        `apply_brainbriefing` patches the template deterministically and validates
        it server-side, so a template briefing needs no agent to become a ready
        workflow. Returns one of:
          * ``{"status": "ready", "workflow_path": ...}``          — done, signal it
          * ``{"status": "needs_fix", "workflow_path", "problems", "server_errors"}``
                                                                    — repair required
          * ``{"status": "build_new", "briefing": <json>}``        — build from scratch
          * ``{"status": "error", "error": ...}``                  — template load failed
        (In Phase 1 the caller/orchestrator handles needs_fix/build_new; Phase 2/3
        route them to the fix_workflow_assembly / generate_new_workflow agents.)
        """
        from src.tools import get_workflow_template as _gwt, apply_brainbriefing as _abb  # noqa: PLC0415
        _gwt = getattr(_gwt, "func", _gwt)
        _abb = getattr(_abb, "func", _abb)

        tname = (briefing.template.name or "").strip()
        if not tname or tname.lower() in ("build_new", "none"):
            _push_progress("🆕 No template fits — building a new workflow …")
            return await self._run_generate_new_workflow(briefing)

        _push_progress(f"🧩 Assembling workflow from template '{tname}' …")
        try:
            tpl = json.loads(_gwt(tname))
        except Exception as exc:  # noqa: BLE001
            return {"status": "error", "error": f"template load failed: {exc}"}
        wf = tpl.get("workflow_path")
        if not wf or tpl.get("error"):
            return {"status": "error", "error": tpl.get("error") or "template load failed"}

        try:
            res = json.loads(_abb(wf, briefing.model_dump_json()))
        except Exception as exc:  # noqa: BLE001
            return {"status": "error", "error": f"apply_brainbriefing failed: {exc}"}

        if res.get("status") == "ok":
            path = res.get("workflow_path", wf)
            # A hard API limit (Kling's 2,500-character prompt, its seven reference
            # images) passes every validation ComfyUI can do and then fails inside
            # the node, at cost. Catch it here, where the orchestrator is still
            # holding the turn and can rewrite what it wrote.
            over = self._limit_violations(path)
            if over:
                return over
            _push_progress("✅ Workflow assembled and validated.")
            if self._verbose:
                print(f"pipeline: deterministic assembly OK — {wf}")
            return {"status": "ready", "workflow_path": path}

        # Distinguish a real workflow defect from ComfyUI being unreachable. The
        # patch itself is deterministic and already done; if there are no concrete
        # `problems` and the only server_error is a connection failure, the workflow
        # IS assembled — pre-validation is just deferred to execution. The fixer
        # can't repair a down server, so treat it as ready (unvalidated).
        problems = res.get("problems") or []
        if not problems and self._server_unreachable(res.get("server_errors")):
            _push_progress("✅ Workflow assembled (ComfyUI offline — validation deferred).")
            if self._verbose:
                print(f"pipeline: deterministic assembly patched OK; server offline — {wf}")
            return {"status": "ready", "workflow_path": wf, "unvalidated": True}

        _push_progress("🩹 Workflow has issues — running the repair specialist …")
        if self._verbose:
            print(f"pipeline: apply_brainbriefing errors — problems={problems} "
                  f"server_errors={res.get('server_errors')}")
        return await self._run_fix_workflow_assembly(
            wf, problems=problems, server_errors=res.get("server_errors", {}))

    def _count_handback(self, violation) -> int:
        """Which attempt this is at the same input, this turn (1 for the first)."""
        counts = getattr(self, "_limit_handbacks", None)
        if counts is None:
            counts = self._limit_handbacks = {}
        key = (violation.node_id, violation.field)
        counts[key] = counts.get(key, 0) + 1
        return counts[key]

    def _collector_refusal(self, prompts: list) -> dict | None:
        """Refuse a collector list whose paths don't exist — before it runs.

        The collector skips a line it cannot find, so a bad path does not fail: it
        renumbers. ``@image4`` then names the picture that used to be ``@image5``,
        the video comes back starring the wrong character, and nothing anywhere
        reported an error. Cheaper to catch here, while the agent still has the
        paths the run that produced them handed back.
        """
        try:
            from src.utils.canvas_hooks import missing_collector_files
        except Exception:  # noqa: BLE001
            return None
        for graph in (prompts or []):
            bad = missing_collector_files(graph)
            if not bad:
                continue
            first = bad[0]
            _push_progress("📁 Collector paths don't exist — sent back to be fixed.")
            return {
                "error": "the collector was given paths that do not exist — nothing "
                         "was queued",
                "what_to_fix": (
                    f"{len(first['missing'])} of the {first['lines']} line(s) you put in "
                    f"node {first['node_id']} ({first['class_type']})'s `files` name no "
                    f"file on disk: " + "; ".join(first["missing"]) + ". Use the ABSOLUTE "
                    "path of each image — the ones a generation handed back in its "
                    "`outputs`, not the filename you had in mind for it."),
                "why_it_matters": (
                    "The collector silently drops a line it cannot find, so this would "
                    "not fail — it would renumber. Every reference after the missing one "
                    "shifts up, and a prompt that says @image4 then names a different "
                    "picture than the table you wrote."),
                "do_not": "Do not report this to the user as a failure: fix the paths and "
                          "call this tool again.",
            }
        return None

    def _collector_text_refusal(self, hook: dict | None, text: str) -> dict | None:
        """The same check for a value written straight into a collector's list.

        A hook feeding a collector's ``files`` produces ONE value — every
        reference path, one per line — so it arrives through place_canvas_text
        rather than through a batch, and would otherwise skip the check entirely.
        """
        base = getattr(self, "_canvas_base_prompt", None)
        if not isinstance(hook, dict) or not isinstance(base, dict):
            return None
        try:
            from src.utils.canvas_hooks import _COLLECTOR_TYPES, _output_targets
        except Exception:  # noqa: BLE001
            return None
        for tid, _ttype, tin, _tintype, _ttitle in _output_targets(hook):
            node = base.get(str(tid))
            if (isinstance(node, dict) and tin == "files"
                    and node.get("class_type") in _COLLECTOR_TYPES):
                probe = {str(tid): {"class_type": node["class_type"],
                                    "inputs": {"files": text}}}
                return self._collector_refusal([probe])
        return None

    def _canvas_limit_refusal(self, hook: dict | None, text: str) -> dict | None:
        """Refuse a hook value the target node's model would reject, or None.

        Checked against the inputs the hook's output actually feeds, so the cap
        applied is the one belonging to the node that will receive it — Kling's
        2,500 for a prompt, 512 for a storyboard slot.
        """
        if hook is None or not isinstance(self._canvas_base_prompt, dict):
            return None
        try:
            from src.utils.canvas_hooks import _output_targets
            from src.utils.model_limits import canvas_refusal, check_value
            found = []
            for tid, _ttype, to_input, _tin_type, _title in _output_targets(hook):
                v = check_value(self._canvas_base_prompt, tid, to_input, text)
                if v is not None:
                    found.append(v)
        except Exception as exc:  # noqa: BLE001
            if self._verbose:
                print(f"[limits] canvas check failed: {exc}")
            return None
        if not found:
            return None
        _push_progress("📏 Value exceeds the model's hard limit — sent back to be rewritten.")
        found.sort(key=lambda v: v.actual - v.limit, reverse=True)
        return canvas_refusal(found, self._count_handback(found[0]))

    def _batch_limit_refusal(self, prompts: list) -> dict | None:
        """Refuse a canvas batch whose variants break a hard model limit, or None.

        One report per distinct (node, input) rather than per variant: twenty-five
        variants of the same over-long prompt is one mistake, and listing it
        twenty-five times buries what to do about it.
        """
        try:
            from src.utils.model_limits import canvas_refusal, check_workflow
            seen: dict = {}
            for p in (prompts or []):
                for v in check_workflow(p):
                    seen.setdefault((v.node_id, v.field), v)
        except Exception as exc:  # noqa: BLE001
            if self._verbose:
                print(f"[limits] batch check failed: {exc}")
            return None
        if not seen:
            return None
        _push_progress("📏 Batch exceeds the model's hard limit — sent back to be rewritten.")
        found = sorted(seen.values(), key=lambda v: v.actual - v.limit, reverse=True)
        return canvas_refusal(found, self._count_handback(found[0]))

    def _limit_violations(self, workflow_path: str, exec_error: dict | None = None) -> dict | None:
        """``{"status": "limit_exceeded", …}`` if the workflow breaks a hard model
        limit, else None.

        Two ways to know: the workflow itself measured against the limits table, and
        the model's own complaint when it already ran. Either way the answer is the
        same and it is not a repair — see :mod:`src.utils.model_limits`.
        """
        try:
            from src.utils.model_limits import (check_workflow_file, guidance,
                                                runtime_limit_error, summary)
            violations = check_workflow_file(workflow_path)
            runtime = runtime_limit_error(exec_error)
        except Exception as exc:  # noqa: BLE001
            if self._verbose:
                print(f"[limits] check failed: {exc}")
            return None
        if not violations and not runtime:
            return None
        _push_progress("📏 Input exceeds the model's hard limit — handing it back.")
        if self._verbose:
            print(f"pipeline: model limit exceeded — {[v.describe() for v in violations]}"
                  f"{' | runtime: ' + runtime if runtime else ''}")
        return {
            "status": "limit_exceeded",
            "workflow_path": workflow_path,
            # The batch executor heals members mid-run and reports `error` when a
            # heal fails; there is no orchestrator listening at that point, so the
            # line it prints has to carry the reason itself.
            "error": summary(violations, runtime),
            "violations": [
                {"node_id": v.node_id, "class_type": v.class_type, "field": v.field,
                 "kind": v.kind, "limit": v.limit, "actual": v.actual}
                for v in violations
            ],
            "guidance": guidance(violations, runtime),
        }

    def _ensure_fix_agent(self) -> "Agent":
        if self._fix_agent is None:
            self._fix_agent = create_fix_workflow_assembly_agent()
        return self._fix_agent

    async def _run_fix_workflow_assembly(
        self, workflow_path: str, *, problems: list | None = None,
        server_errors: dict | None = None, exec_error: dict | None = None,
        agent: "Agent | None" = None,
    ) -> dict:
        """Run the consolidated repair specialist on a broken workflow.

        Handles both assembly errors (``problems``/``server_errors``) and
        execution errors (``exec_error``). Returns
        ``{"status": "ready"|"needs_fix"|"failed", "workflow_path": ...}``.
        The agent patches the workflow file in place; it does not signal or submit.

        Pass ``agent`` to run on a caller-owned instance instead of the shared
        cached one — required when several repairs run concurrently (inline batch
        healing), since one agent's message history can't be reused in parallel.
        """
        # Before spending a repair turn: is this a hard model limit rather than a
        # defect? The specialist has no move here — it cannot shorten a prompt
        # without deciding what the prompt was for — so it would patch around the
        # symptom and hand back a workflow that fails the same way.
        over = self._limit_violations(workflow_path, exec_error)
        if over:
            return over

        agent = agent or self._ensure_fix_agent()
        try:
            agent.messages.clear()
        except Exception:  # noqa: BLE001
            pass

        if exec_error:
            det = exec_error.get("details") or {}
            tb = det.get("traceback") or []
            tb_tail = "\n".join(("".join(str(x) for x in tb) if isinstance(tb, list)
                                 else str(tb)).splitlines()[-15:]).strip()
            prompt = (
                f"EXECUTION ERROR — ComfyUI failed to run the workflow.\n\n"
                f"workflow_path: {workflow_path}\n"
                f"Failing node: {det.get('node_type', '?')} (id {det.get('node_id', '?')})\n"
                f"Exception: {det.get('exception_type', '')}: "
                f"{det.get('exception_message', '') or exec_error.get('error', '')}\n"
                + (f"\nTraceback (tail):\n{tb_tail}\n" if tb_tail else "")
                + "\nDiagnose the failing node, apply the minimal fix, and re-validate. "
                "Do NOT signal or submit."
            )
        else:
            prompt = (
                f"ASSEMBLY ERROR — apply_brainbriefing reported validation problems.\n\n"
                f"workflow_path: {workflow_path}\n"
                f"problems: {json.dumps(problems or [])[:2000]}\n"
                f"server_errors: {json.dumps(server_errors or {})[:1500]}\n\n"
                "Fix each problem with the smallest change, then re-validate. "
                "Do NOT signal or submit."
            )

        _push_progress("🩹 Repairing the workflow …")
        try:
            async with asyncio.timeout(self._FIX_ASSEMBLY_TIMEOUT):
                await agent.invoke_async(prompt)
        except (TimeoutError, asyncio.TimeoutError):
            _push_progress("⚠️ Repair timed out.")
            return {"status": "failed", "workflow_path": workflow_path,
                    "error": "repair agent timed out"}
        except Exception as exc:  # noqa: BLE001
            return {"status": "failed", "workflow_path": workflow_path, "error": str(exc)}

        # Re-validate: a clean validate (or an unreachable server) means ready.
        try:
            from src.tools import validate_workflow as _vw  # noqa: PLC0415
            vres = json.loads(getattr(_vw, "func", _vw)(workflow_path))
        except Exception:  # noqa: BLE001
            vres = {}
        vstatus = vres.get("status")
        if vstatus == "ok" or self._server_unreachable(vres.get("server_errors")) \
                or (not vres.get("problems") and not vres.get("server_errors")):
            _push_progress("✅ Workflow repaired.")
            return {"status": "ready", "workflow_path": workflow_path}
        _push_progress("⚠️ Workflow still has issues after repair.")
        return {"status": "needs_fix", "workflow_path": workflow_path,
                "problems": vres.get("problems", []),
                "server_errors": vres.get("server_errors", {})}

    async def _heal_exec_failure(self, workflow_path: str, exec_error: dict) -> dict:
        """Inline heal callback for ``execute_workflows_batch``.

        Called by the executor the instant a batch member fails at ComfyUI run
        time. Runs the consolidated repair specialist on that member's file in
        place and returns its ``{"status": "ready"|"needs_fix"|"failed", ...}``
        dict; the executor re-queues the workflow when status is ``"ready"``.
        Repairs run concurrently (bounded by the executor's semaphore), so each
        gets its OWN fix agent (the shared cached one can't carry parallel message
        histories) and only edits the given file.

        A provider's content refusal never reaches that agent: there is nothing in
        the graph for it to fix, and running it anyway spends a repair budget
        rewriting a workflow that was already correct. Those re-run instead.
        """
        rejection = self._policy_rejection(exec_error)
        if rejection is not None:
            return self._retry_after_refusal(workflow_path, rejection)
        agent = create_fix_workflow_assembly_agent()
        try:
            return await self._run_fix_workflow_assembly(
                workflow_path, exec_error=exec_error, agent=agent)
        finally:
            try:
                agent.messages.clear()
            except Exception:  # noqa: BLE001
                pass

    @staticmethod
    def _reroll_seeds(graph: dict) -> int:
        """Give every seed in *graph* a fresh value. Returns how many changed.

        Without this a re-run reproduces the previous result exactly, and the
        retry is pure waste. Not every node has one — several hosted APIs expose a
        seed that their backend ignores — but those are non-deterministic anyway,
        so a re-run is a fresh roll either way; this just makes it explicit where
        the model does honour it.
        """
        import random as _random
        rerolled = 0
        for node in (graph or {}).values():
            if not isinstance(node, dict):
                continue
            for slot in ("seed", "noise_seed"):
                if isinstance((node.get("inputs") or {}).get(slot), (int, float)):
                    node["inputs"][slot] = _random.randint(0, 2**31 - 1)
                    rerolled += 1
        return rerolled

    @staticmethod
    def _policy_rejection(exec_error: dict | None):
        """Read a run failure as a provider content refusal, or None."""
        try:
            from src.utils.content_policy import classify
            return classify(exec_error or {})
        except Exception:  # noqa: BLE001
            return None

    def _retry_after_refusal(self, workflow_path: str, rejection) -> dict:
        """Run a refused generation again with a fresh seed, a bounded number of times.

        Returns the same shape the repair specialist does, so the executor's own
        re-queue path carries it: ``ready`` means "try this file again". The graph
        is edited IN PLACE because that is the file the executor re-submits.
        """
        from src.utils.content_policy import exhausted, retry_note
        key = str(workflow_path)
        used = self._policy_retries.get(key, 0)
        allowed = rejection.retries()
        if used >= allowed:
            _push_progress(f"🚫 {rejection.who()} refused it on content grounds "
                           f"after {used} retry(s) — not a workflow defect.")
            return exhausted(rejection, used)
        self._policy_retries[key] = used + 1
        try:
            path = Path(workflow_path)
            graph = json.loads(path.read_text(encoding="utf-8"))
            rerolled = self._reroll_seeds(graph)
            path.write_text(json.dumps(graph, indent=2), encoding="utf-8")
        except Exception as exc:  # noqa: BLE001
            return {"status": "failed",
                    "error": f"could not re-roll the refused workflow: {exc}"}
        _push_progress(retry_note(rejection, used + 1, allowed, rerolled))
        if self._verbose:
            print(f"pipeline: content refusal ({rejection.stage}) on "
                  f"{path.name} — retry {used + 1}/{allowed}, {rerolled} seed(s) rerolled.")
        return {"status": "ready", "workflow_path": str(path),
                "retried_after": "content_policy"}

    async def _qa_retry(self, workflow_path: str, qa_fail: dict) -> dict:
        """Inline QA-retry callback for ``execute_workflows_batch``.

        Called when a member RAN cleanly but its output missed the user's QA
        briefing. Deliberately small: reroll the seeds and rewrite the positive
        prompt to address exactly the criteria that failed. It does not re-plan or
        re-assemble — the graph is proven to run and the user chose it; what was
        wrong is the picture. Rebuilding the graph would also invalidate the very
        verdict that asked for the retry.

        Writes a SIBLING file rather than editing in place, so the rejected
        workflow (and the output it made) remains exactly as it was for comparison.
        Returns ``{"status": "ready"|"failed", "workflow_path": …}``.
        """
        import random as _random
        from src.utils.qa import load_qa_prompts as _qa_prompts

        failures = [f for d in (qa_fail.get("fail_details") or [])
                    for f in (d.get("failed") or [])]
        try:
            src_path = Path(workflow_path)
            graph = json.loads(src_path.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001
            return {"status": "failed", "error": f"could not read the workflow: {exc}"}

        # 1. Reroll every seed. Without this a re-run reproduces the rejected image
        #    byte-for-byte and the retry is pure waste — this alone fixes a good
        #    share of sampling defects (hands, duplicated limbs).
        rerolled = self._reroll_seeds(graph)

        # 2. Rewrite the positive prompt against the named failures. The briefing's
        #    deterministic trace says which node and which slot carries it.
        rewrote = False
        try:
            briefing = json.loads(self._last_brainbriefing_json or "{}")
        except Exception:  # noqa: BLE001
            briefing = {}
        node_id = str(briefing.get("positive_prompt_node_id") or "")
        slot = next((pn.get("slot", "text") for pn in (briefing.get("prompt_nodes") or [])
                     if str(pn.get("node_id")) == node_id and pn.get("role") == "positive"), "text")
        current = ((graph.get(node_id) or {}).get("inputs") or {}).get(slot)
        if failures and isinstance(current, str) and current.strip():
            prompts = _qa_prompts()
            user = (prompts.get("retry_user", "")
                    .replace("{{PROMPT}}", current)
                    .replace("{{FAILURES}}", "\n".join(f"- {f}" for f in failures)))
            try:
                from src.utils.llm_functions import LLMFunctions
                reply = await LLMFunctions.from_settings().chat([
                    {"role": "system", "content": prompts.get("retry_system", "")},
                    {"role": "user", "content": user},
                ])
                new_prompt = (reply or "").strip().strip('"')
                if new_prompt and new_prompt != current:
                    graph[node_id]["inputs"][slot] = new_prompt
                    rewrote = True
            except Exception as exc:  # noqa: BLE001
                print(f"pipeline: QA retry could not rewrite the prompt ({exc}) — "
                      "re-running with a fresh seed only.")

        if not rerolled and not rewrote:
            # Nothing would differ, so a re-run would reproduce the same output.
            return {"status": "failed",
                    "error": "no seed or positive prompt to change in this workflow"}

        out_path = src_path.with_name(f"{src_path.stem}.qa{_random.randint(1000, 9999)}.json")
        try:
            out_path.write_text(json.dumps(graph, indent=2), encoding="utf-8")
        except Exception as exc:  # noqa: BLE001
            return {"status": "failed", "error": f"could not write the retry workflow: {exc}"}
        if self._verbose:
            what = "prompt + seed" if rewrote else "seed"
            print(f"pipeline: QA retry — adjusted {what} → {out_path.name}")
        return {"status": "ready", "workflow_path": str(out_path)}

    def _ensure_generate_agent(self) -> "Agent":
        if self._generate_agent is None:
            self._generate_agent = create_generate_new_workflow_agent()
        return self._generate_agent

    async def _run_generate_new_workflow(self, briefing: "BrainBriefing") -> dict:
        """Build a workflow from scratch for a build_new briefing via the
        generate_new_workflow specialist. Returns
        ``{"status": "ready"|"failed", "workflow_path": ...}``.
        """
        agent = self._ensure_generate_agent()
        try:
            agent.messages.clear()
        except Exception:  # noqa: BLE001
            pass
        prompt = (
            "Build a ComfyUI workflow from scratch for this build_new briefing. "
            "Follow the assemble-new-workflow skill, apply the briefing's input/"
            "prompt/output bindings, validate the graph, and end your reply with a "
            "line `workflow_path: <path>`.\n\nbrainbriefing:\n"
            + briefing.model_dump_json(indent=2)[:6000]
        )
        _push_progress("🆕 Building a new workflow from scratch …")
        out = ""
        try:
            async with asyncio.timeout(self._GENERATE_WORKFLOW_TIMEOUT):
                out = str(await agent.invoke_async(prompt))
        except (TimeoutError, asyncio.TimeoutError):
            _push_progress("⚠️ Build-from-scratch timed out.")
            return {"status": "failed", "error": "generate agent timed out"}
        except Exception as exc:  # noqa: BLE001
            return {"status": "failed", "error": str(exc)}

        m = re.search(r"workflow_path[\"']?\s*[:=]\s*[\"']?([^\s\"'\n]+)", out)
        wf = (m.group(1).strip() if m else "") or (_latest_output_workflow() or "")
        if wf and os.path.exists(wf):
            _push_progress("✅ New workflow built.")
            if self._verbose:
                print(f"pipeline: generate_new_workflow produced — {wf}")
            return {"status": "ready", "workflow_path": wf}
        return {"status": "failed",
                "error": "generate agent did not produce a workflow file",
                "output": out[:400]}

    @staticmethod
    def _server_unreachable(server_errors) -> bool:
        """True when server_errors is only a ComfyUI connection failure (env issue,
        not a workflow defect)."""
        if not isinstance(server_errors, dict):
            return False
        blob = json.dumps(server_errors).lower()
        return any(s in blob for s in (
            "connection refused", "max retries", "actively refused",
            "failed to establish a new connection", "connectionerror",
            "newconnectionerror", "10061", "timed out", "connection aborted"))

    async def _arun_researcher(self, user_input, staged_inputs: list | None = None):
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
        _bad_tmpl_retries = 0  # hallucinated-template rejections (bounded → build_new)
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
            decision = None
            if raw_json is not None:
                try:
                    decision = ResearcherDecision.model_validate(json.loads(raw_json))
                except (json.JSONDecodeError, ValidationError) as exc:
                    last_error = str(exc)
                    if self._verbose:
                        print(f"pipeline: Researcher ({label}) validation failed: {last_error}")
                    decision = None
            else:
                last_error = "No JSON object found in the output."

            # #1 schema-constrained emission: if the free-text output didn't yield a
            # valid decision, force one from the draft via a tool-free JSON-mode call
            # (response_format=json_object, schema in the prompt) — which cannot run away.
            if decision is None:
                constrained = self._constrain_briefing(user_input, list(self._researcher.messages))
                if constrained:
                    try:
                        cand = ResearcherDecision.model_validate(json.loads(constrained))
                        tname = (cand.template.name or "").lower()
                        # Reject a meta/stub decision (model describing the instruction
                        # rather than deciding for the request).
                        if cand.status not in ("ready", "blocked") or \
                                "draft" in tname or "researcher" in tname:
                            last_error = "schema-constrained emission produced a stub/meta decision"
                            decision = None
                        else:
                            decision = cand
                            if self._verbose:
                                print("pipeline: decision recovered via schema-constrained emission.")
                    except (json.JSONDecodeError, ValidationError) as exc:
                        last_error = f"schema-constrained emission still invalid: {exc}"
                        decision = None

            if decision is None:
                continue

            # A content-free block (status='blocked' with no concrete blocker) is
            # usually an agent error. Retry a bounded number of times, nudging it to
            # name the blocker or proceed ready; if it still insists, accept the block.
            if decision.status == "blocked" and not any(
                    isinstance(b, str) and b.strip() for b in (decision.blockers or [])):
                if _eb_retries < self._MAX_EMPTY_BLOCKER_RETRIES:
                    _eb_retries += 1
                    last_error = (
                        "You set status='blocked' but listed no concrete blocker. If no "
                        "template genuinely fits or the request is truly unclear, name "
                        "that in 'blockers'. Otherwise set status='ready' and return your "
                        "template + prompt decision."
                    )
                    if self._verbose:
                        print(f"pipeline: rejected empty-blocker decision "
                              f"({_eb_retries}/{self._MAX_EMPTY_BLOCKER_RETRIES}); retrying.")
                    _context_reset = True
                    continue
                if self._verbose:
                    print("pipeline: empty-blocker persisted — accepting the block.")

            # Validate the template pick against the catalog. A hallucinated name
            # (common after a history-clear → constrain fallback, e.g. 'img2img')
            # would otherwise make the scaffold block. Snap near-misses to the real
            # name; retry a bounded number of times with suggestions; then build_new.
            _tname = (decision.template.name or "").strip()
            if _tname and _tname.lower() not in ("build_new", "none"):
                _match = self._match_template(_tname)
                if _match is None:
                    if _bad_tmpl_retries < self._MAX_BAD_TEMPLATE_RETRIES:
                        _bad_tmpl_retries += 1
                        _suggest = ", ".join(self._closest_catalog_names(_tname))
                        last_error = (
                            f"Template '{_tname}' does not exist. Choose an EXACT name "
                            f"from get_workflow_catalog. Closest available: {_suggest}. "
                            f"If none genuinely fit, set template.name to \"build_new\".")
                        if self._verbose:
                            print(f"pipeline: rejected hallucinated template '{_tname}' "
                                  f"({_bad_tmpl_retries}/{self._MAX_BAD_TEMPLATE_RETRIES}); retrying.")
                        _context_reset = True
                        continue
                    if self._verbose:
                        print(f"pipeline: template '{_tname}' unresolved after retries — "
                              f"falling back to build_new.")
                    decision.template.name = "build_new"
                elif _match != _tname:
                    if self._verbose:
                        print(f"pipeline: snapped template '{_tname}' → '{_match}'.")
                    decision.template.name = _match

            # Assemble the full briefing: merge the decision with the deterministic
            # scaffold (which owns node bindings + runs check_model). Missing-model
            # blockers surface here.
            briefing = self._assemble_briefing(decision, staged_inputs)

            # Deterministic download+rerun: if the scaffold flagged a named missing
            # model, resolve it on HF and fetch it into ComfyUI's extra model path,
            # then retry. Once per request; a no-op when downloads are disabled.
            if briefing.status == "blocked" and not _downloaded and any(
                    isinstance(b, str) and b.strip() for b in (briefing.blockers or [])):
                _downloaded = True
                if self._attempt_model_downloads(briefing.blockers or []):
                    _context_reset = True
                    last_error = ("The previously-missing model(s) have now been "
                                  "downloaded and are available. Reassess and return your "
                                  "template + prompt decision with status='ready'.")
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

        # Real prompt slot for this node ("prompt" for API/partner nodes, else
        # "text"), taken from the briefing's deterministic prompt_nodes trace so a
        # batched API workflow's per-variation prompts don't get spliced into a
        # non-existent "text" field.
        pos_slot = next(
            (pn.get("slot", "text") for pn in (briefing.get("prompt_nodes") or [])
             if str(pn.get("node_id")) == str(node_id) and pn.get("role") == "positive"),
            "text",
        )

        # Use only the first base workflow (Brain should signal exactly one)
        base_path = workflow_paths[0]
        expanded = _apply_multiprompt_variations(
            base_path,
            node_id,
            slot=pos_slot,
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
    planner_llm: str | None = None,
    planner_ollama_model: str | None = None,
    planner_anthropic_model: str | None = None,
    orchestrator_llm: str | None = None,
    verbose: bool = True,
    info_context: dict | None = None,
    session_id: str = "default",
) -> Pipeline:
    """Create and return a ready-to-use free-agent Pipeline.

    All arguments are optional; each falls back to environment variables,
    then to hard-coded defaults.

    Researcher defaults:
        RESEARCHER_LLM          = ollama
        RESEARCHER_OLLAMA_MODEL = qwen3-coder:32b
        RESEARCHER_ANTHROPIC_MODEL (if llm=claude)

    Planner defaults:
        PLANNER_LLM             = (reads llm.pipeline.planner from settings.json)
        PLANNER_OLLAMA_MODEL    = (model from settings, then llm.pipeline.llm_functions)
        PLANNER_ANTHROPIC_MODEL (if llm=claude)

    Args:
        researcher_llm: LLM backend for the Researcher (``'ollama'`` or ``'claude'``).
        researcher_ollama_model: Ollama model override for the Researcher.
        researcher_anthropic_model: Anthropic model override for the Researcher.
        planner_llm: LLM backend for the Planner agent (``'ollama'`` or ``'claude'``).
        planner_ollama_model: Ollama model override for the Planner agent.
        planner_anthropic_model: Anthropic model override for the Planner agent.
        orchestrator_llm: LLM backend for the Orchestrator.
        verbose: Print stage-transition log lines (default True).
    """
    researcher = create_query_templates_agent(
        llm=researcher_llm,
        ollama_model=researcher_ollama_model,
        anthropic_model=researcher_anthropic_model,
    )
    info_agent = create_info_agent()
    planner_agent = create_planner_agent(
        llm=planner_llm,
        ollama_model=planner_ollama_model,
        anthropic_model=planner_anthropic_model,
    )
    scout_agent = create_search_web_agent()
    pipeline = Pipeline(
        researcher,
        info_agent=info_agent,
        planner_agent=planner_agent,
        scout_agent=scout_agent,
        verbose=verbose,
        info_context=info_context,
        session_id=session_id,
    )
    # Build the orchestrator with the pipeline's delegation tools appended, then
    # wire it (and its live AgentSkills plugin) into the pipeline.
    orchestrator = create_orchestrator_agent(
        llm=orchestrator_llm,
        extra_tools=pipeline._delegation_tools,
    )
    pipeline.set_orchestrator(orchestrator)
    return pipeline
