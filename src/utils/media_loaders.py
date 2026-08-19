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


def installed(kind: str) -> str:
    """The best loader class for *kind* that this ComfyUI actually has, or ``""``.

    The frontend answers this from LiteGraph's registry when it drops a node for
    a generated output. Anything building a node SERVER-side — splicing a hook,
    resolving a `#tag` into a wire — has no registry to consult and used to just
    write ``LoadImage``, which is the one loader that cannot take a path.

    ``""`` means "could not be asked", not "nothing available".
    """
    try:
        from agenty_core.tools.comfyui import registered_node_classes
        have = registered_node_classes()
    except Exception:  # noqa: BLE001
        return ""
    # An empty set is "could not ask", and falls out of the loop as "" on its
    # own — nothing is ever in it. Which is the answer wanted either way: the
    # caller must not be told core is available when that was never established.
    for cls in candidates(kind):
        if cls in have:
            return cls
    return ""


def _bare_name(value: str) -> bool:
    """A filename with no directory in it — already what a name loader wants."""
    return "/" not in value and "\\" not in value


def value_for(class_type: str, path: str) -> str | None:
    """What to write into *class_type*'s file widget so it loads *path*.

    ``None`` when this loader cannot be made to load that file at all, which the
    caller must treat as "do not build this node" — a loader pointing at nothing
    looks entirely normal on the canvas and fails only when it runs.

    A **path loader** takes the path as given. A **name loader** takes a name in
    ComfyUI's input directory, so anything that is not already a bare name has to
    be COPIED in there first and the copy's name written instead. Staging is what
    makes the fallback real: without it, "this ComfyUI has no path loader" would
    mean "this reference cannot be used", which is not true — it only means the
    file has to be somewhere else first.
    """
    text = str(path or "").strip().strip('"')
    if not text:
        return None
    if takes_absolute_path(class_type):
        return text
    if _bare_name(text):
        return text                     # already names something in the input dir
    try:
        from agenty_core.tools.image_io import stage_image
        staged = stage_image(text) or {}
    except Exception:  # noqa: BLE001
        return None
    name = str(staged.get("name") or "").strip()
    return name or None


def image_loader_node(path: str) -> dict | None:
    """An API-prompt node that will actually load the image at *path*, or None.

    ``{"class_type": …, "inputs": {…}}``, ready to drop into a prompt dict.

    Images only. Every image loader worth choosing between keeps its file under
    the same widget name, ``image``, which the video loaders do not — and a
    generic version of this would have to guess at a widget it has never been
    run against.
    """
    text = str(path or "").strip().strip('"')
    if not text:
        return None
    # A bare filename already names something in the input directory, which is
    # exactly what core's loader takes. No pack to look for, nothing to copy, and
    # no reason to need ComfyUI reachable to work that out.
    if _bare_name(text):
        return {"class_type": "LoadImage",
                "inputs": {"image": text, "upload": "image"}}
    cls = installed("image")
    if not cls:
        return None                     # cannot ask, and cannot stage either
    value = value_for(cls, text)
    if value is None:
        return None
    # Start from the node's OWN required widgets. ComfyUI rejects a prompt with a
    # required input missing rather than falling back to the declared default, so
    # a node built with only its file set never runs: `VHS_LoadImagePath` needs
    # `custom_width` and `custom_height` as much as it needs the path.
    try:
        from agenty_core.tools.comfyui import node_default_inputs
        inputs = node_default_inputs(cls)
    except Exception:  # noqa: BLE001
        inputs = {}
    inputs["image"] = value
    if not takes_absolute_path(cls):
        inputs["upload"] = "image"      # the widget core LoadImage draws its picker from
    return {"class_type": cls, "inputs": inputs}


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
