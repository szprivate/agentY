"""
ComfyUI tools for the Strands agent.

Exports tool lists for the Researcher and Brain agents.
"""

from src.tools.comfyui import (  # noqa: F401
    # Models
    get_model_types,
    get_models_in_folder,
    # Execution control
    interrupt_execution,
    free_memory,
    # Queue
    queue,
    # History
    get_history,
    get_prompt_status_by_id,
    clear_history,
    # Prompt submission
    submit_prompt,
    # Workflow handoff (replaces submit_prompt for the Brain)
    signal_workflow_ready,
    # Batch: create iteration copies of a validated workflow
    duplicate_workflow,
    # Node inspection
    get_node_schema,
    get_workflow_node_info,
    search_nodes,
    # Workflow templates
    get_workflow_catalog,
    get_workflow_template,
    # Workflow modification
    save_workflow,
    patch_workflow,
    add_workflow_node,
    remove_workflow_node,
    # Workflow validation
    validate_workflow,
    # Public helpers
    reset_patch_workflow_guard,
)
from src.tools.image_handling import (  # noqa: F401
    upload_image,
    view_image,
    get_image_resolution,
    analyze_image,
)
from src.tools.slack_tools import (  # noqa: F401
    slack_send_dm,
    slack_send_image,
    slack_send_video,
    slack_send_json,
)
from src.tools.huggingface import (  # noqa: F401
    search_huggingface_models,
    get_model_info,
    check_local_model,
    download_hf_model,
)
from src.tools.file_tools import read_text_file, write_text_file  # noqa: F401
from src.tools.iterate import iterate  # noqa: F401
from src.tools.shell import run_script  # noqa: F401
from strands_tools import file_read  # noqa: F401
from strands_tools import calculator  # noqa: F401
from strands_tools import stop  # noqa: F401

# ---------------------------------------------------------------------------
# Info-agent tools – read-only; answers questions about capabilities/models/workflows.
# ---------------------------------------------------------------------------
INFO_TOOLS: list = [
    get_workflow_catalog,
    get_workflow_template,
    get_model_types,
    get_models_in_folder,
    get_node_schema,
    search_nodes,
    read_text_file,
    file_read,
    stop,
    analyze_image,
    get_image_resolution,
]

# ---------------------------------------------------------------------------
# Researcher tools – read-only resolution (template lookup, model listing).
# No workflow execution or I/O side-effects.
# ---------------------------------------------------------------------------
RESEARCHER_TOOLS: list = [
    get_workflow_catalog,
    get_workflow_template,
    get_model_types,
    read_text_file,
    get_image_resolution,
    analyze_image,
    run_script,  # needed for skills (e.g. image-downsize)
    iterate,
    calculator,
    stop,
]

# ---------------------------------------------------------------------------
# Brain tools – workflow assembly only (steps 1-5 + handoff).
# Execution, polling, Vision QA, and Slack are handled by the Executor.
# ---------------------------------------------------------------------------
BRAIN_TOOLS: list = [
    # Models / nodes
    get_model_types,
    get_models_in_folder,
    get_node_schema,
    get_workflow_node_info,
    # Upload input images
    upload_image,
    get_image_resolution,
    # Workflow assembly & validation
    get_workflow_template,
    patch_workflow,
    add_workflow_node,
    remove_workflow_node,
    save_workflow,
    search_nodes,
    validate_workflow,
    # Handoff to executor (replaces submit_prompt)
    signal_workflow_ready,
    # Batch: duplicate workflow for each iteration
    duplicate_workflow,
    # Script execution (for skills, e.g. image-downsize)
    run_script,
    # Iteration utility
    iterate,
    # Slack – DM only, for reporting assembly errors / blockers
    slack_send_dm,
    slack_send_json,
    # Hugging Face model management
    search_huggingface_models,
    get_model_info,
    check_local_model,
    download_hf_model,
    # File operations (strands built-in + project)
    file_read,
    read_text_file,
    write_text_file,
    stop,
]
