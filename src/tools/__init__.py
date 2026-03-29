"""
ComfyUI tools for the Strands agent.

Each sub-module exposes one or more @tool-decorated functions that map
directly to ComfyUI server REST endpoints.
"""

from src.tools.execution import free_memory, interrupt_execution  # noqa: F401
from src.tools.history import (  # noqa: F401
    get_history,
    get_prompt_history,
    manage_history,
)
from src.tools.models import (  # noqa: F401
    get_model_types,
    get_models_in_folder,
    get_node_info,
    get_object_info,
    get_view_metadata,
)
from src.tools.prompt import get_prompt_status, submit_prompt  # noqa: F401
from src.tools.queue import get_queue, manage_queue  # noqa: F401
from src.tools.system import (  # noqa: F401
    get_embeddings,
    get_extensions,
    get_features,
    get_system_stats,
)
from src.tools.upload import upload_image, upload_mask  # noqa: F401
from src.tools.userdata import (  # noqa: F401
    delete_userdata_file,
    get_userdata_file,
    list_userdata,
    list_userdata_v2,
    move_userdata_file,
    save_userdata_file,
)
from src.tools.users import create_user, get_users  # noqa: F401
from src.tools.view import view_image  # noqa: F401
from src.tools.workflow_builder import (  # noqa: F401
    get_node_schema,
    get_workflow_template,
    list_workflow_templates,
    parse_workflow_connections,
    search_nodes,
    search_workflow_templates,
    validate_workflow,
)
from src.tools.workflows import get_workflow_templates as get_server_workflow_templates  # noqa: F401
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
from strands_tools import file_read, file_write, editor  # noqa: F401

ALL_TOOLS: list = [
    # System / info
    get_system_stats,
    # Models / nodes
    get_model_types,
    get_models_in_folder,
    get_node_info,
    # Prompt execution
    submit_prompt,
    interrupt_execution,
    free_memory,
    # Queue
    get_queue,
    manage_queue,
    # History
    get_history,
    get_prompt_history,
    manage_history,
    # Upload / view
    upload_image,
    view_image,
    # Workflows & building
    list_workflow_templates,
    search_workflow_templates,
    get_workflow_template,
    get_node_schema,
    search_nodes,
    validate_workflow,
    parse_workflow_connections,
    # User data
    list_userdata,
    get_userdata_file,
    save_userdata_file,
    delete_userdata_file,
    move_userdata_file,
    # Slack DM
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
    file_write,
    editor,
]
