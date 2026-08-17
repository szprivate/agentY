"""Which node a finished file goes into on the canvas, and what to write in it.

Two shapes of image/video loader exist on a ComfyUI graph, and they do not take
the same thing:

* **name loaders** — core ``LoadImage``/``LoadVideo``, and the Video Helper
  Suite's *upload* loaders. The widget is a list of what sits in ComfyUI's
  **input directory**, so the file has to be copied in there first and the widget
  holds the copy's name.
* **path loaders** — VHS's ``(Path)`` nodes (``VHS_LoadImagePath``,
  ``VHS_LoadVideoPath``, …). The widget is free text holding an **absolute
  path**, so the node reads the original file where it was written.

A path loader is preferred wherever the pack is installed: nothing is copied, and
the node points at the file the run actually produced rather than at a duplicate
that ages separately from it, is not what any log or sidecar refers to, and
quietly doubles what a session costs on disk.

Both halves of that decision live here together, because getting them out of step
is silent: a path loader handed an input-relative name, or a name loader handed an
absolute path, is a node that looks entirely normal on the canvas and fails only
when it runs.
"""
from __future__ import annotations

# Tried in order; the frontend takes the first one this ComfyUI actually has
# registered, so an install without VHS keeps the behaviour it always had.
CANDIDATES: dict[str, list[str]] = {
    "image": ["VHS_LoadImagePath", "LoadImage"],
    "video": ["VHS_LoadVideoPath", "VHS_LoadVideo", "LoadVideo"],
}


def candidates(kind: str) -> list[str]:
    """Loader classes to try for ``"image"`` / ``"video"``, best first."""
    return list(CANDIDATES.get(str(kind or ""), []))


def takes_absolute_path(class_type: str) -> bool:
    """True when this loader's file widget holds an absolute path, not a name.

    VHS names every path variant of a loader ``…Path`` — ``VHS_LoadVideoPath``,
    ``VHS_LoadImagePath``, ``VHS_LoadVideoFFmpegPath``, ``VHS_LoadImagesPath`` —
    and that suffix is the whole distinction between the two shapes, so it is
    what is matched. A node pack that ships a path loader under some other name
    is read as a name loader, which is the safe way to be wrong: the file is
    staged into the input directory either way, so the value written is one that
    at least exists.
    """
    return str(class_type or "").strip().endswith("Path")
