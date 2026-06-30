"""Register every official ComfyUI workflow template into the template library.

Replicates the registration step of the `/add_workflow` slash command (see
`src/chainlit_app.py` ~L835-846): loads `config/workflow_templates.json`,
adds `{stem: description}` for any official template not already present, and
writes the file back with `json.dumps(..., indent=4, ensure_ascii=False) + "\n"`.

Descriptions come from the authoritative `index.json` blueprint entries; only
templates missing an index description fall back to the LLM generator in
`scripts/build_skill.py`.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OFFICIAL_DIR = PROJECT_ROOT / "comfyui_workflow_templates_official"
TEMPLATES_PATH = PROJECT_ROOT / "config" / "workflow_templates.json"
EXCLUDE = {"index", "index.schema"}


def _sanitize(text: str) -> str:
    """ASCII-safe punctuation: en/em dashes -> '-', arrow -> '->'."""
    if not text:
        return text
    return (
        text.replace("→", "->")  # → arrow
        .replace("—", "-")  # — em dash
        .replace("–", "-")  # – en dash
    )


def _load_index_descriptions() -> dict[str, str]:
    """Map template name -> description across all categories' blueprints."""
    with open(OFFICIAL_DIR / "index.json", encoding="utf-8") as f:
        categories = json.load(f)
    descriptions: dict[str, str] = {}
    for cat in categories:
        for bp in cat.get("blueprints", []):
            name = bp.get("name")
            if name:
                descriptions[name] = bp.get("description", "") or ""
    return descriptions


def _load_build_skill():
    """Import scripts/build_skill.py via importlib (as chainlit_app does)."""
    import importlib.util as ilu

    bs_path = str(PROJECT_ROOT / "scripts" / "build_skill.py")
    mod = sys.modules.get("_agenty_build_skill")
    if mod is None:
        spec = ilu.spec_from_file_location("_agenty_build_skill", bs_path)
        mod = ilu.module_from_spec(spec)
        sys.modules["_agenty_build_skill"] = mod
        spec.loader.exec_module(mod)
    return mod


def main() -> int:
    # 1. Load existing library (or {}).
    if TEMPLATES_PATH.exists():
        library = json.loads(TEMPLATES_PATH.read_text(encoding="utf-8"))
    else:
        library = {}

    index_descriptions = _load_index_descriptions()

    # 2. Enumerate official templates (exclude index files).
    template_files = sorted(
        p for p in OFFICIAL_DIR.glob("*.json") if p.stem not in EXCLUDE
    )

    added: list[str] = []
    skipped: list[str] = []
    fallback: list[str] = []
    build_skill_mod = None

    for path in template_files:
        stem = path.stem
        if stem in library:
            skipped.append(stem)
            continue

        description = index_descriptions.get(stem, "")
        if not description:
            # Fallback: generate via build_skill LLM helper.
            try:
                if build_skill_mod is None:
                    build_skill_mod = _load_build_skill()
                with open(path, encoding="utf-8") as f:
                    wf_data = json.load(f)
                description = build_skill_mod._generate_workflow_template_description(
                    wf_data, stem
                )
                fallback.append(stem)
            except Exception as exc:  # noqa: BLE001
                print(
                    f"  ! Fallback FAILED for '{stem}': {exc}", file=sys.stderr
                )
                fallback.append(f"{stem} (FAILED)")
                description = ""

        library[stem] = _sanitize(description)
        added.append(stem)

    # 5. Write back exactly as the chainlit handler does.
    TEMPLATES_PATH.write_text(
        json.dumps(library, indent=4, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    # Report.
    print(f"Added:    {len(added)}")
    print(f"Skipped:  {len(skipped)} (already present)")
    print(f"Fallback: {len(fallback)} (LLM-generated){' -> ' + ', '.join(fallback) if fallback else ''}")
    print(f"Total keys in library: {len(library)}")

    official_names = {p.stem for p in template_files}
    missing = official_names - set(library)
    print(f"Official templates present: {len(official_names - missing)}/{len(official_names)}")
    if missing:
        print(f"  ! MISSING: {sorted(missing)}")

    # Spot-check 3 added entries.
    print("\nSpot-check (3 added entries):")
    for stem in added[:3]:
        print(f"  - {stem}: {library[stem]}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
