# agentY - ComfyUI Agent powered by Strands Agents SDK

from pathlib import Path as _Path

_PROJECT_ROOT = _Path(__file__).resolve().parent.parent


def _ensure_agenty_core_importable() -> None:
    """Put the sibling ``agenty_core`` checkout on sys.path when the venv can't see it.

    ``agenty_core`` is installed *editable*, so ONE .pth file in site-packages is
    the entire mechanism by which it reaches the interpreter. Since 3.11,
    ``site.addpackage()`` silently skips a .pth carrying the "hidden" file flag —
    and on macOS, iCloud Drive (Desktop & Documents sync) sets that flag on
    everything inside a dot-named directory, which includes ``.venv``.

    The launchers clear the flag before starting. That is a race they usually
    lose: measured on this failure, iCloud puts the flag back **0.75 s** after
    it is cleared, while the seconds between the launcher's ``chflags`` and the
    first import are spent on the update check, the dependency install and the
    model-cache refresh. Sometimes the daemon is busy and the start works;
    sometimes it isn't and the start dies with::

        ModuleNotFoundError: No module named 'agenty_core'

    …from an install that is complete, correct and plainly visible on disk. That
    is the "only sometimes" in the bug report, and no amount of clearing the flag
    can close it, because the flag is not ours to keep clear.

    So stop depending on the .pth. The editable finder maps ``agenty_core`` to
    ``<parent>/agenty_core/agenty_core``, an ordinary package directory — adding
    its parent to ``sys.path`` resolves to exactly the same code by exactly the
    same rules, and needs no file flag to survive.

    Only when the import cannot be resolved otherwise: a working install keeps
    priority, and this can never shadow one.
    """
    import importlib.util
    import os
    import sys

    try:
        if importlib.util.find_spec("agenty_core") is not None:
            return
    except (ImportError, ValueError):
        pass  # a broken finder is the same problem — fall through and fix it

    candidates = []
    env = os.environ.get("AGENTY_CORE_DIR", "").strip()
    if env:
        candidates.append(_Path(env).expanduser())
    candidates.append(_PROJECT_ROOT.parent / "agenty_core")
    for cand in candidates:
        # The checkout, not the package inside it: requirements.txt installs
        # `-e ../agenty_core`, so the importable package is one level down.
        if (cand / "agenty_core" / "__init__.py").is_file():
            sys.path.insert(0, str(cand))
            return


_ensure_agenty_core_importable()

# The shared tool layer lives in the sibling ``agenty_core`` package, which
# can't resolve this repo's config/ and output dirs from its own __file__.
# Anchor it on this repo's root before any tool/util module is imported.
from agenty_core import set_project_root as _set_project_root  # noqa: E402

_set_project_root(_PROJECT_ROOT)
