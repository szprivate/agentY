"""Add / remove custom workflow templates *and* keep the recipe DB in sync.

This is the single place that mutates the canonical custom-template corpus (in
``agenty_core``) so every entry point behaves identically — the ``/add_workflow``
and ``/remove_workflow`` slash commands, and the orchestrator's
``add_canvas_workflow`` chat tool.

Adding a workflow performs the full set of steps the recipe pipeline needs, so
the template is immediately usable by the researcher/brain:

1. write the workflow JSON into ``comfyui_workflow_templates_custom/templates/``
   (the folder the recipe generator walks) — this was the missing step that left
   ``/add_workflow`` registering metadata for a file the corpus never contained;
2. register the ``name -> {models, io, description}`` entry in that folder's
   ``index.json`` (the sole catalog — the flat ``workflow_templates.json`` is
   retired), generating a best-effort one-line description;
3. **regenerate the recipe database** so the new workflow appears as a recipe.

Removing reverses 1–2 (plus the derived skill dir) and regenerates the DB.

The recipe DB is a pure, deterministic function of the corpus (grouping is
``(task, model)`` with no similarity threshold), so "keep recipes in sync" is
simply "re-run the generator" — implemented by :func:`regenerate_recipes`, which
drives ``agenty_core.workflow_recipes.cli`` offline (cache-only, so it never
needs a live ComfyUI).
"""
from __future__ import annotations

import importlib.util
import json
import logging
import re
import shutil
import sys
from pathlib import Path
from typing import Any

from agenty_core.paths import project_root

from agenty_core.utils.workflow_parser import (
    _custom_index_path,
    parse_workflow,
    workflow_remove,
)

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
def _templates_dir() -> Path:
    """Folder holding the custom workflow JSON files (next to their index.json)."""
    return _custom_index_path().parent


def sanitize_name(name: str) -> str:
    """Reduce a user-supplied name to a safe template stem (filename-safe)."""
    stem = Path(str(name or "").strip()).stem  # drop any dir / .json the user typed
    stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", stem).strip("._-")
    return stem


# --------------------------------------------------------------------------- #
# Description generation (best-effort; reuses scripts/build_skill.py)
# --------------------------------------------------------------------------- #
def _generate_description(wf_data: dict, name: str) -> str:
    """Generate a one-line template description, or "" if generation fails.

    ``scripts/build_skill.py`` is not an importable package module, so it is
    loaded by path (cached in ``sys.modules`` under a private key)."""
    try:
        mod = sys.modules.get("_agenty_build_skill")
        if mod is None:
            bs_path = str(project_root() / "scripts" / "build_skill.py")
            spec = importlib.util.spec_from_file_location("_agenty_build_skill", bs_path)
            mod = importlib.util.module_from_spec(spec)
            sys.modules["_agenty_build_skill"] = mod
            spec.loader.exec_module(mod)
        return mod._generate_workflow_template_description(wf_data, name) or ""
    except Exception as exc:  # noqa: BLE001 - description is a nice-to-have
        logger.warning("workflow description generation failed for %r: %s", name, exc)
        return ""


# --------------------------------------------------------------------------- #
# Recipe DB regeneration
# --------------------------------------------------------------------------- #
def regenerate_recipes() -> dict:
    """Rebuild the canonical recipe DB (+ report + node knowledge) from the
    corpus, offline. Returns ``{workflow_count, task_count, recipe_count}`` or
    ``{}`` if nothing was parsed / regeneration failed."""
    try:
        from agenty_core.workflow_recipes.cli import build_arg_parser, run

        args = build_arg_parser().parse_args(["--no-fetch"])  # cache-only, deterministic
        result = run(args)
        db = result.get("database") if isinstance(result, dict) else None
        if db is None:
            return {}
        return {
            "workflow_count": db.workflow_count,
            "task_count": len(db.tasks),
            "recipe_count": db.recipe_count,
        }
    except Exception as exc:  # noqa: BLE001 - never let a regen failure mask the add/remove
        logger.exception("recipe regeneration failed: %s", exc)
        return {"error": str(exc)}


# --------------------------------------------------------------------------- #
# Add
# --------------------------------------------------------------------------- #
def register_workflow(wf_data: dict, name: str, *, source_path: Path | None = None,
                      regenerate: bool = True) -> dict:
    """Register ``wf_data`` as custom template ``name`` and regenerate recipes.

    Both a workflow loaded from a JSON file and the graph captured from the
    ComfyUI canvas flow through here, so they land in the corpus identically.

    Set ``regenerate=False`` to skip the recipe rebuild (for bulk callers that
    register many workflows and regenerate once at the end).

    Returns a summary dict: ``{name, template_file, index_path, description,
    recipes}``. Raises ``ValueError`` for an empty/invalid name or a non-dict
    workflow.
    """
    if not isinstance(wf_data, dict) or not wf_data:
        raise ValueError("workflow is empty or not a JSON object")
    stem = sanitize_name(name)
    if not stem:
        raise ValueError(f"invalid template name: {name!r}")

    # 1. Write the workflow JSON into the corpus templates folder (skip if the
    #    source file already *is* that canonical file, to avoid reformatting it).
    templates_dir = _templates_dir()
    templates_dir.mkdir(parents=True, exist_ok=True)
    target = templates_dir / f"{stem}.json"
    already_there = source_path is not None and source_path.resolve() == target.resolve()
    if not already_there:
        target.write_text(json.dumps(wf_data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    # 2. Best-effort one-line description, then register the index.json entry
    #    (name, models, io, description). index.json is the sole catalog now —
    #    re-registration preserves an existing description if this one is blank.
    description = _generate_description(wf_data, stem)
    parse_workflow(wf_data, name=stem, description=description, update_index=True)

    # 3. Regenerate the recipe DB so the new workflow is a recipe.
    recipes = regenerate_recipes() if regenerate else {}

    return {
        "name": stem,
        "template_file": str(target),
        "index_path": str(_custom_index_path()),
        "description": description,
        "recipes": recipes,
    }


# --------------------------------------------------------------------------- #
# Remove
# --------------------------------------------------------------------------- #
def remove_workflow(name: str) -> dict:
    """Remove custom template ``name`` from the corpus and regenerate recipes.

    Reverses :func:`register_workflow`: drops the index entry (and with it the
    catalog description), the template JSON file, and the derived skill
    directory. Returns ``{name, index_path, removed_file, recipes}``.
    """
    stem = sanitize_name(name)
    if not stem:
        raise ValueError(f"invalid template name: {name!r}")

    # 1. Remove from the custom index.json (also drops its catalog description).
    idx = workflow_remove(stem)

    # 2. Delete the template JSON file from the corpus templates folder.
    target = _templates_dir() / f"{stem}.json"
    removed_file = False
    if target.exists():
        target.unlink()
        removed_file = True

    # 3. Remove the derived skill directory (kebab-case of the name).
    kebab = stem.lower().replace("_", "-")
    skill_dir = project_root() / "skills" / kebab
    if skill_dir.exists():
        shutil.rmtree(skill_dir, ignore_errors=True)

    # 4. Regenerate the recipe DB so the removed workflow drops out.
    recipes = regenerate_recipes()

    return {
        "name": stem,
        "index_path": str(idx),
        "removed_file": removed_file,
        "recipes": recipes,
    }


def reindex_all(regenerate: bool = True) -> dict:
    """Rebuild the index.json entries for every workflow JSON already present in
    the corpus templates folder, then regenerate the recipe DB once.

    Used by bulk maintenance (``update_all_workflows.ps1``) after the templates
    themselves were dropped into the folder directly. Returns
    ``{registered, failed, recipes}``.
    """
    templates_dir = _templates_dir()
    registered: list[str] = []
    failed: list[str] = []
    for path in sorted(templates_dir.glob("*.json")):
        if path.name.lower().startswith("index"):
            continue
        try:
            wf = json.loads(path.read_text(encoding="utf-8"))
            register_workflow(wf, path.stem, source_path=path, regenerate=False)
            registered.append(path.stem)
        except Exception as exc:  # noqa: BLE001 - one bad file must not stop the batch
            logger.warning("reindex skipped %s: %s", path.name, exc)
            failed.append(path.stem)
    recipes = regenerate_recipes() if regenerate else {}
    return {"registered": registered, "failed": failed, "recipes": recipes}


def format_recipe_counts(recipes: dict) -> str:
    """Render the recipe-count summary for a user-facing status line."""
    if not recipes:
        return "recipes unchanged"
    if "error" in recipes:
        return f"⚠️ recipe regeneration failed: {recipes['error']}"
    return (f"recipes rebuilt -> {recipes['workflow_count']} workflows, "
            f"{recipes['task_count']} tasks, {recipes['recipe_count']} recipes")
