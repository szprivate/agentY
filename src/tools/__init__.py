"""
ComfyUI tools for the Strands agent.

Exports tool lists for the Query Templates, Assemble Workflow, Info, Detect User Intent, Planner,
Learnings, ErrorChecker, and VisionAgent agents.
"""

from src.tools.comfyui import (  # noqa: F401
    # Execution control
    interrupt_execution,
    free_memory,
    # Queue
    queue,
    # History
    get_history,
    get_prompt_status_by_id,
    clear_history,
    # Diagnostics
    get_logs,
    get_system_stats,
    get_comfyui_dirs,
    get_agent_output_dirs,
    # Prompt submission
    submit_prompt,
    # Batch: create iteration copies of a validated workflow
    duplicate_workflow,
    # Node inspection
    get_node_schema,
    get_workflow_node_info,
    search_nodes,
    # Custom-node install + auto-heal (find the pack that provides a missing node)
    find_custom_node_for,
    install_custom_node,
    # Workflow templates
    get_workflow_catalog,
    get_workflow_template,
    # Workflow recipes (task -> model -> node clusters knowledge base)
    list_workflow_recipes,
    get_workflow_recipe,
    # Workflow modification
    save_workflow,
    open_workflow_in_canvas,
    patch_workflow,
    add_workflow_node,
    remove_workflow_node,
    update_workflow,
    replace_node,
    apply_brainbriefing,
    # Workflow validation
    validate_workflow,
    # Public helpers
    reset_patch_workflow_guard,
    # Session cache management
    clear_tool_caches,
)
from src.tools.image_handling import (  # noqa: F401
    upload_image,
    upload_image_multiple,
    view_image,
    get_image_resolution,
    analyze_image,
    download_image,
)
from src.tools.comfyui import check_model  # noqa: F401
from src.tools.huggingface import (  # noqa: F401
    search_huggingface_models,
    get_model_info,
    find_hf_file,
    download_hf_model,
)
from src.tools.file_tools import read_text_file, write_text_file  # noqa: F401
from src.tools.iterate import iterate  # noqa: F401
# NOTE: the deterministic-assembly tool (agenty_core.tools.assembly_deterministic)
# is intentionally NOT re-exported here. The free-agent orchestrator assembles via
# apply_brainbriefing + LLM-supervised patch/validate; the old headless
# _try_deterministic_brain fast-path that used it was removed (it silently
# mis-assembled some templates).
from src.tools.shell import run_script  # noqa: F401
from src.tools.memory_tools import memory_read, memory_write  # noqa: F401
from src.tools.web_search import web_search, web_search_images  # noqa: F401
# agentY-only pipeline handoff (not part of the shared agenty_core layer)
from src.tools.workflow_handoff import signal_workflow_ready  # noqa: F401
from src.tools.bake import bake_hooks_to_canvas  # noqa: F401
# Orchestrator meta-tools: live skill authoring + ad-hoc subagents (agentY-only)
from src.tools.orchestration import (  # noqa: F401
    create_skill,
    list_skills,
    remove_skill,
    spawn_subagent,
    # custom-node-creator: build a ComfyUI custom node from a model's GitHub repo
    create_custom_node,
    list_generated_nodes,
)
# Headless batch jobs — shared with agentY-mcp via agenty_core
from src.tools.batch import (  # noqa: F401
    start_batch_job,
    get_batch_status,
    stop_batch_job,
    list_batch_jobs,
)
from strands_tools import file_read  # noqa: F401
from strands_tools import calculator  # noqa: F401
from strands_tools import stop  # noqa: F401

# ---------------------------------------------------------------------------
# Strands tool wrapping for the shared agenty_core tools.
#
# The shared tools in agenty_core are framework-agnostic plain functions (their
# ``@tool`` decorator is a no-op).  Strands needs each agent-callable tool to be
# a DecoratedFunctionTool, so we wrap the shared ones here.  agentY-local tools
# (image_handling, memory_tools, iterate, agent_control, signal_workflow_ready)
# are already ``@tool``-decorated in their own modules and are left untouched.
# ---------------------------------------------------------------------------
from strands import tool as _strands_tool

_SHARED_CORE_TOOLS = [
    # comfyui
    "interrupt_execution", "free_memory", "queue", "get_history",
    "get_prompt_status_by_id", "clear_history", "get_logs", "get_system_stats",
    "get_comfyui_dirs", "get_agent_output_dirs", "submit_prompt", "duplicate_workflow", "get_node_schema",
    "get_workflow_node_info", "search_nodes", "find_custom_node_for", "install_custom_node",
    "get_workflow_catalog",
    "get_workflow_template", "list_workflow_recipes", "get_workflow_recipe",
    "save_workflow", "open_workflow_in_canvas", "patch_workflow", "add_workflow_node",
    "remove_workflow_node", "update_workflow", "replace_node", "apply_brainbriefing",
    "validate_workflow", "check_model",
    # huggingface
    "search_huggingface_models", "get_model_info", "find_hf_file", "download_hf_model",
    # file / shell / web
    "read_text_file", "write_text_file", "run_script", "web_search", "web_search_images",
    # batch
    "start_batch_job", "get_batch_status", "stop_batch_job", "list_batch_jobs",
]
for _n in _SHARED_CORE_TOOLS:
    globals()[_n] = _strands_tool(globals()[_n])
del _n

# ---------------------------------------------------------------------------
# Info-agent tools – read-only; answers questions about capabilities/models/workflows.
# ---------------------------------------------------------------------------
INFO_TOOLS: list = [
    memory_read,
    get_workflow_catalog,
    get_workflow_template,
    check_model,
    get_node_schema,
    search_nodes,
    read_text_file,
    file_read,
    stop,
    analyze_image,
    get_image_resolution,
    # Web search
    web_search,
    web_search_images,
    download_image,   # fetch a found reference image to disk
]

# ---------------------------------------------------------------------------
# Story-agent tools – pure text generation; no tools needed.
# The story agent writes small storylines and calls no ComfyUI tools.
# ---------------------------------------------------------------------------
STORY_TOOLS: list = []

# ---------------------------------------------------------------------------
# Search Web-agent tools – web reference search + staging. Shares the same web/image
# tools as the Info agent, but is a focused subagent the Storyboard director uses
# to find references and return a structured JSON manifest.
# ---------------------------------------------------------------------------
SEARCH_WEB_TOOLS: list = [
    web_search,
    web_search_images,
    download_image,      # stage a found image into ComfyUI's input dir
    analyze_image,       # verify a candidate matches the need
    get_image_resolution,
    stop,
]

# ---------------------------------------------------------------------------
# DoP-agent tools – pure text transformation; no tools needed. The Director of
# Photography agent reads a finished storyboard/prompt and rewrites it with
# concrete lighting/composition/camera/colour decisions. It calls no tools.
# ---------------------------------------------------------------------------
DOP_TOOLS: list = []

# ---------------------------------------------------------------------------
# Query Templates tools – template lookup, model resolution, prompting.
#
# The researcher is deliberately scoped to template retrieval + prompt writing.
# Option B (thin decision contract): the Researcher only picks a template and
# writes the prompt. The deterministic scaffold (build_briefing_scaffold) owns
# EVERY mechanical field — node bindings, paths, model checks, resolution — so
# the researcher needs none of those tools. Trimming the set is what actually
# caps the call count: a tool that isn't here cannot be called.
#   - get_workflow_catalog: browse templates to pick one (its only discovery need)
#   - stop: clean termination
# Deliberately excluded: get_workflow_template, get_comfyui_dirs, check_model,
# get_image_resolution, read_text_file, run_script, iterate, memory_*, web_search*,
# all HF tools — the scaffold or orchestrator handles each of these downstream,
# and the model writes prompts from its own knowledge. A tool absent here cannot
# be called, which is what actually caps the researcher's call count.
# ---------------------------------------------------------------------------
QUERY_TEMPLATES_TOOLS: list = [
    get_workflow_catalog,
    stop,
]

# ---------------------------------------------------------------------------
# Assemble Workflow tools – workflow assembly only (steps 1-5 + handoff).
# Execution, polling, and Vision QA are handled by the Executor.
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Triage tools – stateless intent classifier; no tools needed.
# ---------------------------------------------------------------------------
TRIAGE_TOOLS: list = []

# ---------------------------------------------------------------------------
# Planner tools – stateless multi-step decomposer; no tools needed.
# ---------------------------------------------------------------------------
PLANNER_TOOLS: list = []

# ---------------------------------------------------------------------------
# Vision Agent tools – stateless, single-shot vision analyser.
# Makes direct Ollama API calls; no Strands tools required.
# ---------------------------------------------------------------------------
VISION_AGENT_TOOLS: list = []

# ---------------------------------------------------------------------------
# Learnings tools – stateless pattern-analyser; no tools needed.
# ---------------------------------------------------------------------------
LEARNINGS_TOOLS: list = []

# ---------------------------------------------------------------------------
# Error-checker tools – diagnostics only; no workflow modification.
# ---------------------------------------------------------------------------
ERROR_CHECKER_TOOLS: list = [
    get_logs,
    get_system_stats,
]

# ---------------------------------------------------------------------------
# custom-node-creator tools – read a cloned model repo, author a ComfyUI custom
# node pack into the output folder. Code-authoring only: file I/O, repo
# inspection (run_script), a web-search fallback for thin READMEs, and memory.
# It does NOT get execution/assembly tools — it writes a pack, it doesn't run one.
# ---------------------------------------------------------------------------
CUSTOM_NODE_TOOLS: list = [
    read_text_file,
    write_text_file,
    file_read,
    run_script,        # inspect the cloned repo (list files, grep for entry points)
    get_agent_output_dirs,
    web_search,        # fallback when the repo's own docs are thin
    memory_read,
    memory_write,
    stop,
]

ASSEMBLE_WORKFLOW_TOOLS: list = [
    # Node inspection (schema lookup only – no model checking)
    get_node_schema,
    get_workflow_node_info,
    # Server directories (resolve authoritative output path)
    get_comfyui_dirs,
    get_agent_output_dirs,  # canonical agent image/video/scripts folders
    # Upload input images
    upload_image,
    upload_image_multiple,
    get_image_resolution,
    # Workflow assembly, modification & validation
    get_workflow_template,
    # Recipe knowledge base (task -> model -> node clusters) for build_new
    list_workflow_recipes,
    get_workflow_recipe,
    apply_brainbriefing,
    update_workflow,
    replace_node,
    save_workflow,
    search_nodes,
    check_model,
    # Handoff to executor (replaces submit_prompt)
    signal_workflow_ready,
    # Load the assembled workflow into the ComfyUI canvas for the user to inspect
    open_workflow_in_canvas,
    # Batch: duplicate workflow for each iteration
    duplicate_workflow,
    # Headless batch jobs over a folder of inputs (detached worker)
    start_batch_job,
    get_batch_status,
    stop_batch_job,
    list_batch_jobs,
    # Script execution (for skills, e.g. image-downsize)
    run_script,
    # Iteration utility
    iterate,
    # File operations (strands built-in + project)
    file_read,
    read_text_file,
    write_text_file,
    # Long-term memory (local FAISS + nomic-embed-text)
    memory_read,
    memory_write,
    stop,
]

# ---------------------------------------------------------------------------
# fix_workflow_assembly tools – the CONSOLIDATED workflow-repair specialist.
# Fires on two triggers, with the same toolset:
#   * assembly-time: apply_brainbriefing returned status:error with `problems`
#   * execution-time: ComfyUI failed to run the workflow (node/model error)
# It diagnoses the failing node, patches the graph, and re-validates. It does NOT
# select templates or write prompts (that is prepare_workflow's job) and does NOT
# submit for execution (the pipeline re-runs it). Includes node install + model
# download so it can heal missing-node / missing-model execution failures.
# ---------------------------------------------------------------------------
FIX_WORKFLOW_ASSEMBLY_TOOLS: list = [
    # Diagnose
    get_node_schema,
    get_workflow_node_info,
    search_nodes,
    # Repair
    update_workflow,
    replace_node,
    save_workflow,
    validate_workflow,
    # Heal missing node types (execution failures)
    find_custom_node_for,
    install_custom_node,
    # Heal missing model files
    check_model,
    search_huggingface_models,
    get_model_info,
    find_hf_file,
    download_hf_model,
    # Server dirs (resolve paths when patching)
    get_comfyui_dirs,
    stop,
]

# ---------------------------------------------------------------------------
# Orchestrator tools – the union of every real capability the specialists have
# (execution, node inspection, template/recipe lookup, assembly, image handling,
# HuggingFace, web search, files, memory, batch, iteration) PLUS the meta-tools
# that let the orchestrator extend itself (author skills, spawn subagents).
#
# Delegation tools (run_research / run_info / …) are bound to the live Pipeline
# instance and appended at pipeline-build time (see Pipeline._build_delegation_tools),
# NOT here — they need the running specialist agents.
# ---------------------------------------------------------------------------
ORCHESTRATOR_TOOLS: list = [
    # Execution / queue / history / diagnostics
    interrupt_execution,
    free_memory,
    queue,
    get_history,
    get_prompt_status_by_id,
    clear_history,
    get_logs,
    get_system_stats,
    get_comfyui_dirs,
    get_agent_output_dirs,
    submit_prompt,
    # Node inspection
    get_node_schema,
    get_workflow_node_info,
    search_nodes,
    # Custom-node install + auto-heal a missing node type
    find_custom_node_for,
    install_custom_node,
    # Templates + recipes.
    # NOTE: the two *browse* tools — get_workflow_catalog and
    # list_workflow_recipes — are intentionally NOT given to the orchestrator.
    # Template/recipe *selection* is delegated to run_research (the
    # query_templates specialist); without a browse menu the orchestrator cannot
    # keyword-match its way into the wrong template/upscale/relight workflow.
    # The by-name loaders stay so the orchestrator can LOAD what run_research
    # already chose (or a [HARD CONSTRAINTS]-pinned name) for assembly:
    #   get_workflow_template(name)         — load the selected template
    #   get_workflow_recipe(task, model)    — fetch the recipe for a build_new
    #                                         briefing run_research produced
    get_workflow_template,
    get_workflow_recipe,
    # Workflow assembly / modification / validation
    duplicate_workflow,
    save_workflow,
    open_workflow_in_canvas,
    patch_workflow,
    add_workflow_node,
    remove_workflow_node,
    update_workflow,
    replace_node,
    apply_brainbriefing,
    validate_workflow,
    check_model,
    # Handoff to the executor (assemble → signal, never submit_prompt directly)
    signal_workflow_ready,
    # Bake a chain of standin workflows into canvas subgraphs (bake_to_canvas hook)
    bake_hooks_to_canvas,
    # Image handling (the orchestrator owns input prep: stage + analyze)
    upload_image,
    upload_image_multiple,  # stage several inputs in one call
    view_image,
    get_image_resolution,
    analyze_image,
    download_image,
    # HuggingFace – discover + download models
    search_huggingface_models,
    get_model_info,
    find_hf_file,
    download_hf_model,
    # Files / shell / web
    read_text_file,
    write_text_file,
    file_read,
    run_script,
    web_search,
    web_search_images,
    # Long-term memory
    memory_read,
    memory_write,
    # Headless batch jobs
    start_batch_job,
    get_batch_status,
    stop_batch_job,
    list_batch_jobs,
    # Iteration + math
    iterate,
    calculator,
    # Self-extension meta-tools
    create_skill,
    list_skills,
    remove_skill,
    spawn_subagent,
    # custom-node-creator: turn a model's GitHub repo into a ComfyUI custom node
    create_custom_node,
    list_generated_nodes,
    stop,
]

# ---------------------------------------------------------------------------
# Rename-compat aliases.
#
# src/agent.py imports the compact spellings of these tool lists, while the
# canonical definitions above use the underscored spellings (and TRIAGE_TOOLS
# predates the triage -> detect_user_intent rename). Alias them so both
# spellings resolve to the same list object.
# ---------------------------------------------------------------------------
QUERYTEMPLATES_TOOLS = QUERY_TEMPLATES_TOOLS
ASSEMBLEWORKFLOW_TOOLS = ASSEMBLE_WORKFLOW_TOOLS
SEARCHWEB_TOOLS = SEARCH_WEB_TOOLS
DETECTUSERINTENT_TOOLS = TRIAGE_TOOLS
