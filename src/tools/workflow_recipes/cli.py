"""Thin command line entry point: parse the corpus, build the recipe database,
write the three outputs.

Run from the repo root:

    python -m src.tools.workflow_recipes.cli            # fetch object_info if cache missing
    python -m src.tools.workflow_recipes.cli --no-fetch # offline, cache only

Outputs (default, under ``config/``):
  workflow_recipes.json                  the consumable recipe database
  workflow_recipes_node_knowledge.json   per node-class signatures
  workflow_recipes_report.md             human-readable hierarchical report
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict

from .parser import Corpus
from .recipe import RecipeBuilder, node_knowledge_json_dict

_PKG_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _PKG_DIR.parents[2]   # workflow_recipes -> tools -> src -> <repo root>

# Defaults (anchored on the repo root so the tool runs from any cwd).
_DEFAULT_CUSTOM = str(_REPO_ROOT / "comfyui_workflow_templates_custom")
_DEFAULT_OFFICIAL = str(_REPO_ROOT / "comfyui_workflow_templates_official")
_DEFAULT_CACHE = str(_PKG_DIR / "object_info_cache.json")
_DEFAULT_TEMPLATE_DESCS = str(_REPO_ROOT / "config" / "workflow_templates.json")
_DEFAULT_OUT = str(_REPO_ROOT / "config" / "workflow_recipes.json")


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="workflow_recipes",
        description="Discover ComfyUI workflow types and emit a recipe database "
                    "(task -> model -> node clusters).",
    )
    p.add_argument("--custom-folder", default=_DEFAULT_CUSTOM,
                   help="folder of custom workflow templates")
    p.add_argument("--official-folder", default=_DEFAULT_OFFICIAL,
                   help="folder of official workflow templates")
    p.add_argument("--out", default=_DEFAULT_OUT,
                   help="output path for the recipe database JSON; the node-"
                        "knowledge JSON and markdown report are written alongside it")
    p.add_argument("--object-info-cache", default=_DEFAULT_CACHE,
                   help="path to cached /object_info JSON (read, or written on fetch)")
    p.add_argument("--templates-descriptions", default=_DEFAULT_TEMPLATE_DESCS,
                   help="flat name->description JSON used to enrich workflows the "
                        "index.json files do not describe (typically custom ones)")
    p.add_argument("--host", default="127.0.0.1", help="ComfyUI host for object_info")
    p.add_argument("--port", type=int, default=8188, help="ComfyUI port for object_info")
    p.add_argument("--no-fetch", action="store_true",
                   help="never contact ComfyUI; use cache only (offline)")
    return p


def _sibling_paths(out: str) -> Dict[str, str]:
    """Derive the node-knowledge and report paths from the database path:
    config/workflow_recipes.json -> config/workflow_recipes_node_knowledge.json
                                  -> config/workflow_recipes_report.md."""
    base = out[:-5] if out.lower().endswith(".json") else out
    return {
        "db": out,
        "node_knowledge": base + "_node_knowledge.json",
        "report": base + "_report.md",
    }


def _write_json(path: str, payload) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def _write_text(path: str, text: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def run(args) -> Dict:
    folders = {"custom": args.custom_folder, "official": args.official_folder}
    corpus = Corpus.load(
        folders,
        object_info_cache=args.object_info_cache,
        host=args.host,
        port=args.port,
        allow_fetch=not args.no_fetch,
        templates_descriptions=args.templates_descriptions,
    )
    graphs = corpus.graphs
    if not graphs:
        print("[error] no workflows parsed; nothing to do")
        return {}

    builder = RecipeBuilder()
    database = builder.build(graphs)
    node_knowledge = builder.node_knowledge(graphs, database.leaves, corpus.object_info)

    paths = _sibling_paths(args.out)
    _write_json(paths["db"], database.to_json_dict())
    _write_text(paths["report"], database.to_report_markdown())
    _write_json(paths["node_knowledge"],
                node_knowledge_json_dict(node_knowledge, len(graphs)))

    print(f"[done] {len(graphs)} workflows -> {len(database.tasks)} tasks / "
          f"{len(database.leaves)} task+model recipes; {len(node_knowledge)} node classes")
    for pth in (paths["db"], paths["report"], paths["node_knowledge"]):
        print(f"[done] wrote {pth}")
    return {"graphs": graphs, "database": database, "node_knowledge": node_knowledge}


def main(argv=None) -> int:
    args = build_arg_parser().parse_args(argv)
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
