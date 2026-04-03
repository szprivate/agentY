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
from src.tools.file_tools import read_text_file  # noqa: F401
from src.tools.shell import run_script  # noqa: F401
from strands_tools import file_read  # noqa: F401
from strands_tools import calculator  # noqa: F401

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
    calculator,
]

# ---------------------------------------------------------------------------
# Brain tools – full execution suite: assembly, validation, run, QA, Slack.
# ---------------------------------------------------------------------------
BRAIN_TOOLS: list = [
    # Models / nodes
    get_model_types,
    get_models_in_folder,
    get_node_schema,
    get_workflow_node_info,
    # Prompt execution
    submit_prompt,
    interrupt_execution,
    free_memory,
    # Queue
    queue,
    # History / status tracking
    get_history,
    get_prompt_status_by_id,
    clear_history,
    # Upload / view
    upload_image,
    view_image,
    get_image_resolution,
    # Vision / image analysis
    analyze_image,
    # Script execution (for skills)
    run_script,
    # Workflows & building
    get_workflow_template,
    patch_workflow,
    add_workflow_node,
    remove_workflow_node,
    save_workflow,
    search_nodes,
    validate_workflow,
    # Slack
    slack_send_dm,
    slack_send_image,
    slack_send_video,
    slack_send_json,
    # Hugging Face model management
    search_huggingface_models,
    get_model_info,
    check_local_model,
    download_hf_model,
    # File operations (strands built-in)
    file_read,
]
