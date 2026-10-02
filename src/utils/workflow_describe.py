"""Write a template's catalog description: the line retrieval runs on.

The researcher picks a template from a list where each one is its name and the
first sentence of its description, cut to 60 characters; the same text helps file
the template under a task and a model. A description that opens with filler
("Local generation via ComfyUI Model", "Example workflow X shipped with pack Y")
therefore makes a template unfindable however good the workflow is.

The old writer (scripts/build_skill.py) knew two providers and read the raw
per-role setting, so on a machine configured by tier it called a local model that
was not installed and stored its generic fallback instead — worse than storing
nothing, because a blank description is filled in from the graph at read time.

This one gives the utility model (``llm_functions``, resolved through the tiers)
the facts read off the graph — operation, the distinctive nodes and whose they
are, model files, what goes in and out, the author's own notes — and asks for one
line in the catalog's shape. No answer, or one that fails the checks, is ``""``:
never filler.

    python -m src.utils.workflow_describe --generic        # rewrite filler descriptions
    python -m src.utils.workflow_describe --missing        # write the ones that are blank
    python -m src.utils.workflow_describe NAME [NAME ...]  # rewrite these custom templates
    python -m src.utils.workflow_describe --node-packs     # the node pack examples
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

logger = logging.getLogger(__name__)

# What the old writer stored when its model call failed.
GENERIC_MARKERS = ("Processes and generates content using ComfyUI workflows",
                   "via ComfyUI Model.")

_SYSTEM = (
    "You write one-line catalog descriptions of ComfyUI workflow templates. An agent "
    "reads a list of templates to pick the right one for a user's request, and sees "
    "ONLY the first sentence of your line, cut to 60 characters — so that sentence "
    "must say what the workflow does and with which model or nodes.\n\n"
    "Format, exactly one line, no quotes:\n"
    "[Local|API] <operation> via <model or distinctive nodes>. <inputs> -> <outputs>. "
    "<one sentence: what it is for, or when to pick it over a similar one>.\n\n"
    "Rules:\n"
    "- Start with [API] if it runs on partner/API nodes, else [Local].\n"
    "- <operation> in plain task words a user would use: text-to-image, image edit, "
    "image-to-video, video-to-video, first/last-frame-to-video, upscale, segmentation, "
    "background removal, inpainting, outpainting, face detailing, depth estimation, "
    "relighting, lip sync, text-to-3D, image-to-3D, text-to-audio, colour conversion …\n"
    "- Name the model family or the node that does the work (e.g. Wan 2.2, Flux Kontext, "
    "SAM3, FaceDetailer), not plumbing nodes. Never write 'ComfyUI Model' or 'workflow'.\n"
    "- The first sentence, after the bracket, is at most 60 characters.\n"
    "- Use only what the facts say. What a pack says it is, and its nodes' own "
    "descriptions, outrank what a node's name suggests. If nothing says what something "
    "is for, describe what the nodes do; do not invent capabilities, quality claims, "
    "techniques or model names, and do not describe the output's quality.\n"
    "- At most 260 characters in all. No line breaks, no markdown."
)


def _prompt(facts: dict, pack: str, workflow: str) -> str:
    lines = [f"Template name: {facts.get('name')}"]
    if pack:
        lines.append(f"It is the example workflow \"{workflow}\" shipped with the custom node pack {pack}.")
    lines.append("Facts read from the graph:")
    lines.append(json.dumps({
        "classifier_guess_task": facts.get("task") or None,
        "output_media": facts.get("media") or None,
        "model_families": facts.get("model_families") or [],
        "model_files": (facts.get("models") or [])[:12],
        "api_nodes": (facts.get("api_nodes") or [])[:10],
        "api_models_selected": facts.get("api_models") or [],
        "custom_pack_nodes": {k: v[:10] for k, v in (facts.get("pack_nodes") or {}).items()},
        "what_each_pack_says_it_is": facts.get("pack_about") or {},
        "node_descriptions": facts.get("node_descriptions") or {},
        "other_node_classes": (facts.get("classes") or [])[:30],
        "inputs": facts.get("inputs") or {},
        "outputs": facts.get("outputs") or {},
        "author_notes_in_graph": facts.get("notes") or None,
    }, ensure_ascii=False, indent=1))
    lines.append("The classifier's task guess is often wrong for unusual graphs: trust the "
                 "nodes and the name over it. Write the line.")
    return "\n".join(lines)


def clean(text) -> str:
    """The model's answer as a description, or ``""`` when it is not one."""
    line = " ".join(str(text or "").split()).strip().strip("\"'`").strip()
    if not (25 <= len(line) <= 420):
        return ""
    if not re.match(r"^\[(Local|API)\]\s+\S", line):
        return ""
    if any(m in line for m in GENERIC_MARKERS):
        return ""
    return line


def _ask(prompt: str) -> str:
    from src.utils.llm_functions import LLMFunctions
    llm = LLMFunctions.from_settings()
    messages = [{"role": "system", "content": _SYSTEM}, {"role": "user", "content": prompt}]
    return asyncio.run(llm.chat(messages))


def describe(wf_data: dict, name: str, *, pack: str = "", workflow: str = "", ask=None) -> str:
    """A model-written description of *wf_data*, or ``""`` — never filler."""
    try:
        from agenty_core.utils.workflow_facts import workflow_facts
        facts = workflow_facts(wf_data, name)
        return clean((ask or _ask)(_prompt(facts, pack, workflow or name)))
    except Exception as exc:  # noqa: BLE001 — a description is never worth failing an add for
        logger.warning("could not write a description for %r: %s", name, exc)
        return ""


# ── node pack examples ─────────────────────────────────────────────────────────

def _node_pack_dir(root=None) -> Path:
    from agenty_core.paths import corpus_root
    from agenty_core.templates_sync import NODE_PACK_DIR
    return (Path(root) if root else corpus_root()) / NODE_PACK_DIR


def describe_node_pack_examples(root=None, *, describe_one=describe, workers: int = 4,
                                limit: int | None = None) -> dict:
    """Give each mirrored node pack example a model-written description, once.

    The sync writes a description read off the graph; this replaces it and records
    the file's sha beside it, so a later start skips what is already written and a
    changed example is described again. Entries the model fails on keep the graph
    description and are tried again next start. Returns ``{written, failed,
    pending}`` (names, names, count left because of *limit*).
    """
    from agenty_core.templates_sync import blob_sha
    folder = _node_pack_dir(root)
    index_path = folder / "index.json"
    try:
        index = json.loads(index_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {"written": [], "failed": [], "pending": 0}
    todo = []
    for group in index:
        pack = str(group.get("moduleName") or "")
        for tpl in group.get("templates") or []:
            path = folder / f"{tpl.get('name')}.json"
            try:
                raw = path.read_bytes()
            except OSError:
                continue
            sha = blob_sha(raw)
            if tpl.get("description_source") == "llm" and tpl.get("description_sha") == sha:
                continue
            todo.append((tpl, pack, raw, sha))
    pending = max(0, len(todo) - limit) if limit is not None else 0
    todo = todo[:limit] if limit is not None else todo
    if not todo:
        return {"written": [], "failed": [], "pending": pending}

    def _one(item):
        tpl, pack, raw, _sha = item
        try:
            data = json.loads(raw.decode("utf-8-sig"))
        except ValueError:
            return ""
        return describe_one(data, tpl["name"], pack=pack, workflow=str(tpl.get("title") or ""))

    written, failed = [], []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for (tpl, _pack, _raw, sha), text in zip(todo, pool.map(_one, todo)):
            if text:
                tpl.update({"description": text, "description_source": "llm", "description_sha": sha})
                written.append(tpl["name"])
            else:
                failed.append(tpl["name"])
    if written:
        tmp = index_path.with_name("index.json.tmp")
        tmp.write_text(json.dumps(index, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        os.replace(tmp, index_path)
    return {"written": written, "failed": failed, "pending": pending}


# ── the user's own custom templates ────────────────────────────────────────────

def redescribe_custom(names=None, *, generic: bool = False, missing: bool = False,
                      describe_one=describe) -> dict:
    """Rewrite descriptions in the custom ``index.json``: the named templates, the
    ones holding the old writer's filler (*generic*), the blank ones (*missing*).
    An entry the model fails on is left exactly as it was. Returns
    ``{rewritten: {name: text}, failed: [names]}``."""
    from agenty_core.utils.workflow_parser import _custom_index_path
    index_path = _custom_index_path()
    index = json.loads(index_path.read_text(encoding="utf-8"))
    wanted = set(names or [])
    rewritten, failed = {}, []
    for group in index:
        for tpl in group.get("templates") or []:
            name, old = tpl.get("name"), (tpl.get("description") or "").strip()
            pick = (name in wanted
                    or (generic and any(m in old for m in GENERIC_MARKERS))
                    or (missing and not old))
            if not pick:
                continue
            path = index_path.parent / f"{name}.json"
            try:
                data = json.loads(path.read_text(encoding="utf-8-sig"))
            except (OSError, ValueError):
                failed.append(name)
                continue
            text = describe_one(data, name)
            if text:
                tpl["description"] = text
                rewritten[name] = text
            else:
                failed.append(name)
    if rewritten:
        tmp = index_path.with_name("index.json.tmp")
        tmp.write_text(json.dumps(index, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        os.replace(tmp, index_path)
    return {"rewritten": rewritten, "failed": failed}


def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser(prog="workflow_describe", description=__doc__.split("\n")[0])
    ap.add_argument("names", nargs="*", help="custom templates to rewrite")
    ap.add_argument("--generic", action="store_true", help="rewrite the old writer's filler")
    ap.add_argument("--missing", action="store_true", help="write the blank ones")
    ap.add_argument("--node-packs", action="store_true", help="describe the node pack examples")
    ap.add_argument("--no-recipes", action="store_true", help="do not rebuild the recipe database")
    args = ap.parse_args(argv)
    try:
        from dotenv import load_dotenv
        load_dotenv(Path(__file__).resolve().parents[2] / ".env")
    except Exception:  # noqa: BLE001
        pass
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass
    changed = False
    if args.names or args.generic or args.missing:
        res = redescribe_custom(args.names, generic=args.generic, missing=args.missing)
        for name, text in res["rewritten"].items():
            print(f"{name}\n    {text}")
        print(f"rewritten {len(res['rewritten'])}, failed {len(res['failed'])}: {res['failed']}")
        changed = changed or bool(res["rewritten"])
    if args.node_packs:
        res = describe_node_pack_examples()
        print(f"node pack examples: written {len(res['written'])}, failed {len(res['failed'])}")
        changed = changed or bool(res["written"])
    if changed and not args.no_recipes:
        from src.utils.workflow_admin import format_recipe_counts, regenerate_recipes
        print(format_recipe_counts(regenerate_recipes()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
