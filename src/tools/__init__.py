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
    upload_file_to_url,
)
from src.tools.video_handling import (  # noqa: F401
    analyze_video,
    set_video_agent,
    # agenty_core owns the implementation (agentY-mcp offers the same tool);
    # this is the thin agentY wrapper that adds dry-run stand-in handling.
    split_video_into_shots,
)
# Mark up an image (circle / box / arrow a subject) without regenerating it.
from src.tools.annotate import annotate_image, set_output_sink  # noqa: F401
from src.tools.comfyui import check_model  # noqa: F401
from src.tools.huggingface import (  # noqa: F401
    search_huggingface_models,
    get_model_info,
    find_hf_file,
    download_hf_model,
    wait_for_download,
)
from src.tools.file_tools import read_text_file, write_text_file  # noqa: F401
from src.tools.iterate import iterate  # noqa: F401
from src.tools.model_switching import list_models, switch_model  # noqa: F401
# NOTE: the deterministic-assembly tool (agenty_core.tools.assembly_deterministic)
# is intentionally NOT re-exported here. The free-agent orchestrator assembles via
# apply_brainbriefing + LLM-supervised patch/validate; the old headless
# _try_deterministic_brain fast-path that used it was removed (it silently
# mis-assembled some templates).
from src.tools.shell import run_script  # noqa: F401
from src.tools.memory_tools import memory_read, memory_write  # noqa: F401
# agentY-local: it reads THIS host's config/mcp.json and live MCP clients, so it
# stays here rather than in the shared core layer.
from src.tools.mcp_tools import list_mcp_servers  # noqa: F401
from src.tools.mcp_install import add_mcp_server  # noqa: F401
from src.tools.project_memory import (project_memory_forget,  # noqa: F401
                                      project_memory_read, project_memory_write)
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
    # coder (custom-node-from-github skill): build a ComfyUI custom node from a repo
    create_custom_node,
    list_generated_nodes,
)
# A lead conversation starting and steering shot conversations (src/utils/shots.py)
from src.tools.shots import (  # noqa: F401
    start_shot,
    message_shot,
    shot_status,
    stop_shot,
    close_shot,
    set_sequence_notes,
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

# Finding models on Hugging Face, on this machine and in workflow files — what
# agents used to piece together with run_script (agenty_core.tools.model_research).
from agenty_core.tools.model_research import (  # noqa: F401,E402
    hf_search, hf_repo, hf_file, find_local_models, inspect_workflow_file,
)
# Looking around — node source, folders, workflow files, media, frames, web pages
# (agenty_core.tools.workspace_tools): the rest of what run_script was used for.
from agenty_core.tools.workspace_tools import (  # noqa: F401,E402
    get_node_source, list_files, find_workflows, media_info, video_frames, read_web_page,
)

# Find other photos of the same thing from one picture (agenty_core.tools.reverse_search); drives the
# system Chrome through Playwright, an optional dependency.
from agenty_core.tools.reverse_search import reverse_image_search  # noqa: F401,E402

# Image / video generation on the user's own DashScope key, not Comfy credits
# (agenty_core.tools.dashscope_generate).
from agenty_core.tools.dashscope_generate import dashscope_generate, dashscope_fetch  # noqa: F401,E402
# Load files into the bEpic viewer on request (agenty_core.tools.show_in_viewer);
# the default tab follows Settings > media_nodes.viewer_tab.
from agenty_core.tools.show_in_viewer import show_in_viewer, set_default_tab_provider as _set_viewer_tab  # noqa: F401,E402
# Fly a camera through a 3D scene in the viewer and render the move (agenty_core.tools.previz_shot).
from agenty_core.tools.previz_shot import render_camera_move  # noqa: F401,E402
# A 3D scene from a picture or a prompt with World Labs Atlas (agenty_core.tools.worldlabs_atlas).
from agenty_core.tools.worldlabs_atlas import (  # noqa: F401,E402
    atlas_create_world, atlas_expand_world, atlas_rebuild_world_mesh, atlas_splat_to_mesh, atlas_fetch, atlas_list,
)
# Render one new view of an Atlas world at a camera, optionally upscaled; open a world in the viewer.
from agenty_core.tools.atlas_render import atlas_render_view, atlas_open_in_viewer  # noqa: F401,E402
# Capture an Atlas world all round; train a Gaussian splat from posed views locally (Brush).
from agenty_core.tools.atlas_capture import atlas_capture_around  # noqa: F401,E402
from agenty_core.tools.splat_train import train_splat, splat_training_status  # noqa: F401,E402
# Rebuild a model in Blender / Maya / Houdini from ONE script file, and check its proportions against a
# photo by projecting key points through the scene camera (agenty_core.tools.dcc).
from agenty_core.tools.dcc import dcc_build, dcc_silhouette_check  # noqa: F401,E402
# Measure a reference photo with MoGe (fov, camera height, metric points) via the local ComfyUI.
from agenty_core.tools.reference_measure import measure_reference  # noqa: F401,E402
# Pipeline actions for a ComfyUI session launched through Ayon (agenty_core.tools.ayon; docs/ayon.md). The host
# lends it the canvas the panel posted and the panel's graph edits (src/tools/ayon_host.py).
from agenty_core.tools.ayon import ayon_context, ayon_load, ayon_mark_publish, ayon_publish, ayon_open_tool  # noqa: F401,E402
from src.tools import ayon_host as _ayon_host  # noqa: E402
_ayon_host.install()
# Send ComfyUI workflows to the on-prem Deadline render farm and follow them (agenty_core.tools.deadline_farm).
from agenty_core.tools.deadline_farm import submit_to_deadline, deadline_job_status, deadline_farm_info  # noqa: F401,E402
# Build and run one partner (API) node directly from its schema (agenty_core.tools.partner_run).
# The orchestrator's own copy is the pipeline closure of the same name (it runs the graph
# through run_workflow_now); this one serves every agent that holds the plain core tool.
from agenty_core.tools.partner_run import partner_run  # noqa: F401,E402
# N independent variants of one workflow duplicated, patched and queued together
# (agenty_core.tools.variants); the orchestrator uses the pipeline closure of the same name.
from agenty_core.tools.variants import run_variants  # noqa: F401,E402
try:
    from src.utils.viewer_group import tab_name as _viewer_tab_name
    _set_viewer_tab(_viewer_tab_name)
except Exception:  # noqa: BLE001
    pass

_SHARED_CORE_TOOLS = [
    # comfyui
    "interrupt_execution", "free_memory", "get_history",
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
    "wait_for_download",
    # model research
    "hf_search", "hf_repo", "hf_file", "find_local_models", "inspect_workflow_file",
    # looking around
    "get_node_source", "list_files", "find_workflows", "media_info", "video_frames",
    "read_web_page",
    # image i/o — moved to the shared layer, so they arrive here as the plain
    # callables every other core tool does and need the same wrapping. They used
    # to be agentY-local @tool functions; missing them here hands Strands a bare
    # function where it expects a DecoratedFunctionTool.
    "download_image", "upload_file_to_url",
    # generation on the user's own DashScope key
    "dashscope_generate", "dashscope_fetch",
    # show files in the bEpic viewer, open an edit in its Edit panel
    "show_in_viewer",
    # animate a camera through a 3D scene in the viewer and render the move
    "render_camera_move",
    # a 3D scene from a picture / prompt with World Labs Atlas
    "atlas_create_world", "atlas_expand_world", "atlas_rebuild_world_mesh", "atlas_splat_to_mesh",
    "atlas_fetch", "atlas_list", "atlas_render_view", "atlas_open_in_viewer",
    # all-round Atlas capture, and a Gaussian splat trained from it (local, Brush)
    "atlas_capture_around", "train_splat", "splat_training_status",
    # modelling from a reference photo in Blender / Maya / Houdini: one build script, a numeric silhouette check
    "dcc_build", "dcc_silhouette_check", "measure_reference",
    # the studio's Ayon pipeline: session context, load a published product, mark an output for publish, open the Publisher
    "ayon_context", "ayon_load", "ayon_mark_publish", "ayon_publish", "ayon_open_tool",
    # the Deadline render farm: submit a workflow as a job, follow it, look at the pools
    "submit_to_deadline", "deadline_job_status", "deadline_farm_info",
    # one partner (API) node built from its schema and run directly
    "partner_run", "run_variants",
    # per-project memory — moved to the shared layer (a Claude Desktop session on
    # the same ComfyUI now reads and writes the same facts the panel does), so it
    # arrives as a plain callable and needs wrapping like every other core tool.
    "project_memory_read", "project_memory_write", "project_memory_forget",
    # file / shell / web
    "read_text_file", "write_text_file", "run_script", "web_search", "web_search_images",
    "reverse_image_search",
    # batch
    "start_batch_job", "get_batch_status", "stop_batch_job", "list_batch_jobs",
]
for _n in _SHARED_CORE_TOOLS:
    globals()[_n] = _strands_tool(globals()[_n])
# agentY's own: ComfyUI's queue plus the workflows signalled this turn, which
# reach ComfyUI only when it ends (workflow_handoff.queue).
from src.tools.workflow_handoff import queue, withdraw_workflow  # noqa: E402,F401
del _n

# ---------------------------------------------------------------------------
# Info-agent tools – read-only; answers questions about capabilities/models/workflows.
# ---------------------------------------------------------------------------
INFO_TOOLS: list = [
    memory_read,
    get_workflow_catalog,
    get_workflow_template,
    check_model,
    find_local_models,
    hf_search,
    hf_repo,
    hf_file,
    inspect_workflow_file,
    get_node_source,
    find_workflows,
    list_files,
    media_info,
    read_web_page,
    get_node_schema,
    search_nodes,
    # "What can this system do?" includes the MCP servers the user has set up, and
    # whether each one's tools are actually loaded right now.
    list_mcp_servers,
    read_text_file,
    file_read,
    stop,
    analyze_image,
    get_image_resolution,
    # Web search
    web_search,
    web_search_images,
    reverse_image_search,   # other photos of the same thing, from one picture
    download_image,   # fetch a found reference image to disk
]

# ---------------------------------------------------------------------------
# Search Web-agent tools – web reference search + staging. Shares the same web/image
# tools as the Info agent; a focused subagent that finds references and returns a
# structured JSON manifest.
# ---------------------------------------------------------------------------
SEARCH_WEB_TOOLS: list = [
    web_search,
    web_search_images,
    reverse_image_search,   # other photos / angles of the same thing from one picture
    download_image,      # stage a found image into ComfyUI's input dir
    analyze_image,       # verify a candidate matches the need
    get_image_resolution,
    stop,
]

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
# Planner tools – stateless multi-step decomposer; no tools needed.
# ---------------------------------------------------------------------------
PLANNER_TOOLS: list = []

# ---------------------------------------------------------------------------
# Vision Agent tools – stateless, single-shot vision analyser.
# Makes direct Ollama API calls; no Strands tools required.
# ---------------------------------------------------------------------------
VISION_AGENT_TOOLS: list = []

# ---------------------------------------------------------------------------
# Video Agent tools – stateless, single-shot video understanding (frames in).
# ---------------------------------------------------------------------------
VIDEO_AGENT_TOOLS: list = []

# ---------------------------------------------------------------------------
# Learnings tools – stateless pattern-analyser; no tools needed.
# ---------------------------------------------------------------------------
LEARNINGS_TOOLS: list = []

# ---------------------------------------------------------------------------
# coder tools – general code-authoring toolset for the `coder` agent (e.g. the
# custom-node-from-github skill: read a cloned model repo, write a custom-node pack).
# Code-authoring only: file I/O, repo inspection (run_script), a web-search
# fallback for thin docs, and memory. It does NOT get execution/assembly tools —
# it writes code, it doesn't run a workflow.
# ---------------------------------------------------------------------------
CODER_TOOLS: list = [
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
    # How a node works, and how other graphs used it
    get_node_source,
    find_workflows,
    inspect_workflow_file,
    # Heal missing model files
    check_model,
    find_local_models,
    search_huggingface_models,
    get_model_info,
    find_hf_file,
    hf_file,
    download_hf_model,
    wait_for_download,
    # Server dirs (resolve paths when patching)
    get_comfyui_dirs,
    stop,
]

# ---------------------------------------------------------------------------
# generate_new_workflow tools – build a ComfyUI workflow from scratch when no
# template fits (template.name == "build_new"). Follows the assemble-new-workflow
# skill: fetch the recipe, load the closest member template as a scaffold, and
# conform it to the recipe (nodes, wiring, boundary ports). Does NOT select a
# template for a normal request (that's prepare_workflow) and does NOT submit.
# ---------------------------------------------------------------------------
GENERATE_NEW_WORKFLOW_TOOLS: list = [
    # Recipe knowledge base (task -> model -> node clusters)
    get_workflow_recipe,
    list_workflow_recipes,
    # Scaffold from a recipe member template
    get_workflow_template,
    # Node inspection
    get_node_schema,
    get_workflow_node_info,
    search_nodes,
    # Build / modify
    add_workflow_node,
    remove_workflow_node,
    update_workflow,
    replace_node,
    save_workflow,
    validate_workflow,
    # How a node works, and how other graphs used it
    get_node_source,
    find_workflows,
    inspect_workflow_file,
    # Models
    check_model,
    find_local_models,
    search_huggingface_models,
    get_model_info,
    find_hf_file,
    hf_file,
    download_hf_model,
    wait_for_download,
    # Server dirs
    get_comfyui_dirs,
    get_agent_output_dirs,
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
    # The MCP servers' own tools arrive unlabelled and only when their server
    # connected; this is how the orchestrator can tell a capability that is missing
    # from one that is merely switched off or waiting to be authorized.
    list_mcp_servers,
    # "Install the X MCP": asks the user, tests, saves, connects.
    add_mcp_server,
    # Models: is it installed, where is it on HuggingFace, download it into the
    # folder ComfyUI loads that kind from. On the orchestrator and not only on the
    # repair specialists, because "download the missing models" is something users
    # ask for directly — and without these it improvised downloads in run_script.
    check_model,
    find_hf_file,
    download_hf_model,
    wait_for_download,
    # Model research: one call each for what run_script used to do in dozens —
    # repos with the file you want, a repo's files and access, one file's size /
    # access / mirrors / destination, what is installed, what a workflow loads.
    hf_search,
    hf_repo,
    hf_file,
    find_local_models,
    inspect_workflow_file,
    # Looking around: what run_script was mostly used for, one call each.
    get_node_source,
    list_files,
    find_workflows,
    media_info,
    video_frames,
    read_web_page,
    # Workflow SET-UP is delegated to prepare_workflow (research + deterministic
    # assembly). Workflow REPAIR (assembly + execution errors) and BUILD-FROM-
    # SCRATCH are the fix_workflow_assembly / generate_new_workflow specialists,
    # invoked inside prepare_workflow / the pipeline. So the orchestrator no longer
    # carries template-load, node-inspection, apply/patch/validate or node-install
    # tools — that whole surface moved off its critical path.
    # It keeps only:
    #   duplicate_workflow + update_workflow — for the batch-handoff skill
    #     (duplicate the assembled base per iteration and swap the input); these
    #     move to a deterministic pipeline loop in a later phase.
    duplicate_workflow,
    update_workflow,
    # ...and for adding a stage the request asked for that the template lacks
    # (FreeU, a LoRA, a second pass): a template covers the core job, and
    # "no template has FreeU" was ending turns with nothing built. The schema
    # is how the added node gets wired right.
    get_node_schema,
    open_workflow_in_canvas,
    # Handoff to the executor (prepare_workflow → signal, never submit directly)
    signal_workflow_ready,
    # ...and taking a signal back before the turn ends and it is submitted
    withdraw_workflow,
    # Bake a chain of standin workflows into canvas subgraphs (a hook's `bake` switch)
    bake_hooks_to_canvas,
    # Image handling (the orchestrator owns input prep: stage + analyze)
    upload_image,
    upload_image_multiple,  # stage several inputs in one call
    view_image,
    get_image_resolution,
    analyze_image,
    analyze_video,       # understand a video input (samples frames -> Video Agent)
    split_video_into_shots,  # find the cuts in a video, write one file per shot
    annotate_image,      # circle/box/arrow things in an image (overlay, not a re-render)
    download_image,
    upload_file_to_url,  # PUT a local file to a presigned URL (MCP upload flows)
    dashscope_generate,  # image/video on the user's own DashScope key (not Comfy credits)
    dashscope_fetch,     # collect one that timed out
    show_in_viewer,      # "show me clip 3 in the viewer" / "open the edit"
    render_camera_move,  # fly a camera through a 3D scene in the viewer, render the move (control video)
    atlas_create_world,  # World Labs Atlas: picture / prompt -> posed depth mesh (GLB)
    atlas_expand_world,  # Atlas invents more views at new cameras and grows the mesh
    atlas_rebuild_world_mesh,
    atlas_splat_to_mesh,
    atlas_fetch,         # collect an Atlas job that timed out
    atlas_list,
    atlas_render_view,   # one new view of an Atlas world at a camera (+ optional upscale): consistent backgrounds
    atlas_open_in_viewer,  # open an Atlas world in the viewer's 3D view for the user to pick a camera
    atlas_capture_around,  # Atlas views all round a place (for a 360 environment / splat)
    train_splat,           # Gaussian splat from posed views (an Atlas world), trained locally with Brush
    splat_training_status, # check on / finish a splat training
    dcc_build,             # run the one build script in Blender / Maya / Houdini (live or batch), save, render a still
    dcc_silhouette_check,  # project key points through the scene camera, compare with the photo by numbers
    measure_reference,     # MoGe on the reference photo: fov, camera height, horizon, metric points
    ayon_context,          # the Ayon project / folder / task of the session ComfyUI was launched in
    ayon_load,             # load a published Ayon product into the canvas (Load node + container)
    ayon_mark_publish,     # make an output a publish instance (Save node + registered instance)
    ayon_publish,          # open the Ayon Publisher on the registered instances (the click stays the user's)
    ayon_open_tool,        # open an Ayon tool (loader, creator, publisher, workfiles, inventory)
    submit_to_deadline,    # send a workflow to the Deadline farm (dedicated ComfyUI pool)
    deadline_job_status,   # follow a farm job, fetch its outputs
    deadline_farm_info,    # pools / workers / the ComfyUI pool
    # Files / shell / web
    read_text_file,
    write_text_file,
    file_read,
    run_script,
    web_search,
    web_search_images,
    reverse_image_search,  # other photos of the same thing from one picture (Yandex / Bing visual search)
    # Long-term memory (across projects: how the user works, what we've learned)
    memory_read,
    memory_write,
    # Per-project memory (this project only: characters, style, locked refs,
    # delivery specs). Lives in ComfyUI's user dir, so it switches with the project.
    project_memory_read,
    project_memory_write,
    project_memory_forget,
    # Headless batch jobs
    start_batch_job,
    get_batch_status,
    stop_batch_job,
    list_batch_jobs,
    # Iteration + math
    iterate,
    list_models,           # which models exist, and what each role runs
    switch_model,          # propose / make a model switch (asks the user first)
    calculator,
    # Self-extension meta-tools
    create_skill,
    list_skills,
    remove_skill,
    spawn_subagent,
    # Sequences: this conversation as the lead of shot conversations
    start_shot,
    message_shot,
    shot_status,
    stop_shot,
    close_shot,
    set_sequence_notes,
    # coder (custom-node-from-github skill): turn a repo into a ComfyUI custom node
    create_custom_node,
    list_generated_nodes,
    stop,
]

# ---------------------------------------------------------------------------
# Rename-compat aliases.
#
# src/agent.py imports the compact spellings of these tool lists, while the
# canonical definitions above use the underscored spellings. Alias them so both
# spellings resolve to the same list object.
# ---------------------------------------------------------------------------
QUERYTEMPLATES_TOOLS = QUERY_TEMPLATES_TOOLS
ASSEMBLEWORKFLOW_TOOLS = ASSEMBLE_WORKFLOW_TOOLS
SEARCHWEB_TOOLS = SEARCH_WEB_TOOLS
