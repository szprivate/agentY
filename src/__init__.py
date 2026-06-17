# agentY - ComfyUI Agent powered by Strands Agents SDK

# The shared tool layer now lives in the sibling ``agenty_core`` package, which
# can't resolve this repo's config/ and output dirs from its own __file__.
# Anchor it on this repo's root before any tool/util module is imported.
from pathlib import Path as _Path

from agenty_core import set_project_root as _set_project_root

_set_project_root(_Path(__file__).resolve().parent.parent)
