"""Thin command line entry point wiring the phases together.

Phases 1-3 (parse -> fingerprint -> cluster) run here and emit:
  - clustering_debug.json    fingerprints + pairwise similarities (tune threshold)
  - clustering_report.md     human-readable groupings with "why" they grouped

Phase 4 (recipe synthesis) is added once the clustering is sanity-checked.

Run:
  python -m workflow_recipes.cli --similarity-threshold 0.6
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List

from . import cluster as cluster_mod
from . import fingerprint as fp_mod
from . import parser as parser_mod
from . import recipe_builder as recipe_mod

# Default input folders (relative to repo root / current working directory).
_DEFAULT_CUSTOM = "comfyui_workflow_templates_custom"
_DEFAULT_OFFICIAL = "comfyui_workflow_templates_official"
_DEFAULT_OUTDIR = os.path.join("workflow_recipes", "output")
_DEFAULT_CACHE = os.path.join("workflow_recipes", "object_info_cache.json")
_DEFAULT_TEMPLATE_DESCS = os.path.join("config", "workflow_templates.json")


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="workflow_recipes",
        description="Discover ComfyUI workflow types and emit a recipe database.",
    )
    p.add_argument("--custom-folder", default=_DEFAULT_CUSTOM,
                   help="folder of custom workflow templates")
    p.add_argument("--official-folder", default=_DEFAULT_OFFICIAL,
                   help="folder of official workflow templates")
    p.add_argument("--out-dir", default=_DEFAULT_OUTDIR,
                   help="directory for generated reports / databases")
    p.add_argument("--similarity-threshold", type=float, default=0.55,
                   help="merge clusters while average similarity >= this value")
    p.add_argument("--object-info-cache", default=_DEFAULT_CACHE,
                   help="path to cached /object_info JSON (read, or written on fetch)")
    p.add_argument("--templates-descriptions", default=_DEFAULT_TEMPLATE_DESCS,
                   help="flat name->description JSON used to enrich workflows the "
                        "index.json files do not describe (typically custom ones)")
    p.add_argument("--host", default="127.0.0.1", help="ComfyUI host for object_info")
    p.add_argument("--port", type=int, default=8188, help="ComfyUI port for object_info")
    p.add_argument("--no-fetch", action="store_true",
                   help="never contact ComfyUI; use cache only (offline)")
    # Fingerprint signal weights (relative; any may be 0 to drop the signal).
    p.add_argument("--weight-classes", type=float, default=fp_mod.DEFAULT_WEIGHTS["classes"])
    p.add_argument("--weight-connections", type=float, default=fp_mod.DEFAULT_WEIGHTS["connections"])
    p.add_argument("--weight-clusters", type=float, default=fp_mod.DEFAULT_WEIGHTS["clusters"])
    p.add_argument("--weight-spine", type=float, default=fp_mod.DEFAULT_WEIGHTS["spine"])
    p.add_argument("--weight-category", type=float, default=fp_mod.DEFAULT_WEIGHTS["category"],
                   help="weight of the catalog-category signal (0 = off, structural only; "
                        "neutral for workflows without a catalog category)")
    return p


def _weights_from_args(args) -> Dict[str, float]:
    return {
        "classes": args.weight_classes,
        "connections": args.weight_connections,
        "clusters": args.weight_clusters,
        "spine": args.weight_spine,
        "category": args.weight_category,
    }


def run(args) -> Dict:
    os.makedirs(args.out_dir, exist_ok=True)
    weights = _weights_from_args(args)

    object_info = parser_mod.load_object_info(
        args.object_info_cache, host=args.host, port=args.port,
        allow_fetch=not args.no_fetch,
    )

    folders = {"custom": args.custom_folder, "official": args.official_folder}
    descriptions = parser_mod.load_descriptions(folders, args.templates_descriptions)
    graphs = parser_mod.load_corpus(folders, object_info, descriptions)
    if not graphs:
        print("[error] no workflows parsed; nothing to do")
        return {}

    fps = [fp_mod.fingerprint(g) for g in graphs]
    matrix = cluster_mod.pairwise_matrix(fps, weights)
    clusters = cluster_mod.agglomerate(fps, matrix, args.similarity_threshold)

    debug_path = os.path.join(args.out_dir, "clustering_debug.json")
    _write_debug(debug_path, fps, matrix, clusters, weights, args.similarity_threshold)

    meta_by_name = {
        g.name: {"category": g.category, "title": g.index_title} for g in graphs
    }
    report_path = os.path.join(args.out_dir, "clustering_report.md")
    _write_report(report_path, fps, clusters, args.similarity_threshold, weights, meta_by_name)

    # Phase 4 - synthesize the recipe database.
    recipes = recipe_mod.build_recipes(
        graphs, clusters, object_info_available=bool(object_info)
    )
    types_path = os.path.join(args.out_dir, "workflow_types.json")
    _write_types_json(types_path, recipes, weights, args.similarity_threshold, len(graphs))
    types_report_path = os.path.join(args.out_dir, "workflow_types_report.md")
    _write_types_report(types_report_path, recipes, args.similarity_threshold)

    # Node knowledge - signatures + usage for the wiring brain.
    node_knowledge = recipe_mod.build_node_knowledge(graphs, recipes, object_info)
    nodes_path = os.path.join(args.out_dir, "node_knowledge.json")
    _write_node_knowledge(nodes_path, node_knowledge, len(graphs))

    print(f"[done] {len(graphs)} workflows -> {len(clusters)} types "
          f"(threshold {args.similarity_threshold}); {len(node_knowledge)} node classes")
    for pth in (debug_path, report_path, types_path, types_report_path, nodes_path):
        print(f"[done] wrote {pth}")
    return {"graphs": graphs, "fingerprints": fps, "clusters": clusters,
            "matrix": matrix, "recipes": recipes, "node_knowledge": node_knowledge}


def _write_debug(path, fps, matrix, clusters, weights, threshold) -> None:
    payload = {
        "config": {"weights": weights, "similarity_threshold": threshold},
        "fingerprints": [
            {
                "name": f.name,
                "source": f.source,
                "node_count": f.node_count,
                "classes": sorted(f.class_set),
                "class_multiset": dict(sorted(f.class_multiset.items())),
                "connection_patterns": sorted([list(c) for c in f.connection_set]),
                "spine": sorted(f.spine_set),
                "category": sorted(f.category_set),
                "cluster_signatures": sorted(f.cluster_set),
            }
            for f in fps
        ],
        "pairwise_similarity": [
            {
                "a": fps[i].name,
                "b": fps[j].name,
                "similarity": round(score, 4),
                "per_signal": {k: round(v, 4) for k, v in per.items()},
            }
            for (i, j), (score, per) in sorted(matrix.items())
        ],
        "clusters": [
            {"members": [fps[m].name for m in c.members], "cohesion": round(c.cohesion, 4)}
            for c in clusters
        ],
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def _cluster_categories(meta_by_name, names) -> Dict[str, int]:
    """Distribution of official catalog categories across a cluster's members,
    ordered by count then name (deterministic)."""
    from collections import Counter
    cats = Counter(
        meta_by_name.get(n, {}).get("category")
        for n in names
        if meta_by_name.get(n, {}).get("category")
    )
    return dict(sorted(cats.items(), key=lambda kv: (-kv[1], kv[0])))


def _write_report(path, fps, clusters, threshold, weights, meta_by_name) -> None:
    singletons = [c for c in clusters if len(c.members) == 1]
    grouped = [c for c in clusters if len(c.members) > 1]
    lines: List[str] = []
    lines.append("# Clustering report (Phase 3)")
    lines.append("")
    lines.append(f"- Workflows: {len(fps)}")
    lines.append(f"- Types (clusters): {len(clusters)}  "
                 f"({len(grouped)} multi-member, {len(singletons)} singletons)")
    lines.append(f"- Similarity threshold: {threshold}")
    lines.append(f"- Signal weights: {weights}")
    lines.append("")
    lines.append("Sanity-check the groupings below, then tell me a threshold to "
                 "lock in before I build Phase 4 (recipe synthesis).")
    lines.append("")

    for idx, c in enumerate(clusters, 1):
        names = [fps[m].name for m in c.members]
        sources = sorted({fps[m].source for m in c.members})
        src_label = sources[0] if len(sources) == 1 else "mixed"
        shared = cluster_mod.shared_signals(fps, c.members)
        cats = _cluster_categories(meta_by_name, [fps[m].name for m in c.members])
        lines.append(f"## Type {idx}  -  {len(c.members)} member(s)  -  source: {src_label}")
        lines.append(f"- cohesion (mean intra-similarity): {round(c.cohesion, 3)}")
        if cats:
            purity = "pure" if len(cats) == 1 else "MIXED categories"
            cat_str = ", ".join(f"{k} ({v})" for k, v in cats.items())
            lines.append(f"- official categories: {cat_str}  [{purity}]")
        lines.append("- members:")
        for m in c.members:
            title = meta_by_name.get(fps[m].name, {}).get("title")
            suffix = f' - "{title}"' if title else ""
            lines.append(f"    - {fps[m].name}  ({fps[m].source}, {fps[m].node_count} nodes){suffix}")
        if len(c.members) > 1:
            lines.append(f"- shared node classes ({len(shared['shared_classes'])}): "
                         + (", ".join(shared["shared_classes"]) or "(none)"))
            conns = shared["shared_connections"]
            lines.append(f"- shared connection patterns ({len(conns)}):")
            for a, b, t in conns[:25]:
                lines.append(f"    - {a} -> {b}  [{t}]")
            if len(conns) > 25:
                lines.append(f"    - ... and {len(conns) - 25} more")
        lines.append("")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def _write_types_json(path, recipes, weights, threshold, n_workflows) -> None:
    payload = {
        "config": {"weights": weights, "similarity_threshold": threshold},
        "generated_from": {"workflow_count": n_workflows, "type_count": len(recipes)},
        "types": recipes,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def _write_types_report(path, recipes, threshold) -> None:
    from collections import Counter
    media_dist = Counter((r["user_intent"].get("media") or "?") for r in recipes)
    lines: List[str] = []
    lines.append("# Workflow types - recipe database")
    lines.append("")
    lines.append(f"- Types: {len(recipes)} (similarity threshold {threshold})")
    lines.append("- Self-contained: every type has a description + user_intent; no "
                 "human annotation step required.")
    lines.append("- Media distribution: "
                 + ", ".join(f"{k} ({v})" for k, v in sorted(media_dist.items())))
    lines.append("- Sorted by member_count descending.")
    lines.append("")

    for r in recipes:
        title = r.get("suggested_title")
        head = f"## `{r['id']}`"
        if title:
            head += f'  -  "{title}"'
        head += f"  -  {r['member_count']} member(s)  -  source: {r['source']}"
        lines.append(head)
        ui = r["user_intent"]
        fams = ", ".join(ui.get("model_families") or []) or "n/a"
        lines.append(f"- user intent: media={ui.get('media')} | task={ui.get('task')} "
                     f"| model families: {fams}")
        lines.append(f"- when to use: {ui.get('when_to_use')}")
        lines.append(f"- example requests: "
                     + "; ".join(f'"{e}"' for e in ui.get("example_requests", [])))
        lines.append(f"- description ({r['description_source']}): {r['description']}")
        cat = r.get("category") or {}
        if cat.get("primary"):
            dist = ", ".join(f"{k} ({v})" for k, v in cat.get("distribution", {}).items())
            if cat.get("spans_multiple"):
                label = "spans multiple catalog categories"
            elif cat.get("uncategorized"):
                label = f"single category (+{cat['uncategorized']} uncategorized)"
            else:
                label = "pure"
            lines.append(f"- official category: {cat['primary']}  [{label}: {dist}]")
        lines.append("")
        descs = r.get("member_descriptions") or []
        lines.append("- member files:")
        desc_by_name = {d["name"]: d.get("description") for d in descs}
        for mf in r["member_files"]:
            d = desc_by_name.get(mf)
            lines.append(f"    - {mf}" + (f" - {d}" if d else ""))
        lines.append("")
        lines.append("- REQUIRED node roles (structural invariants):")
        functional = [e for e in r["required_node_roles"] if not e.get("utility")]
        utility = [e for e in r["required_node_roles"] if e.get("utility")]
        if functional:
            for e in functional:
                tag = ""
                if e.get("paired_or_multiple"):
                    tag = f"  **[PAIRED: {e['min_instances']}x required]**"
                lines.append(f"    - {e['node_class']}  ({e['role']}) - {e['frequency']}{tag}")
                for di in e.get("distinct_instances", []):
                    lines.append(f"        - instance feeds {di['feeds_into']} / fed by "
                                 f"{di['fed_by']}  (x{di['occurrences']})")
        else:
            lines.append("    - (none)")
        if utility:
            util_str = ", ".join(
                f"{e['node_class']}({e['max_instances']}x)" if e["max_instances"] > 1
                else e["node_class"] for e in utility
            )
            lines.append(f"    - utility/plumbing (always present): {util_str}")
        lines.append("")
        lines.append("- OPTIONAL node roles (variant, only in some members):")
        opt_functional = [e for e in r["optional_node_roles"] if not e.get("utility")]
        opt_utility = [e for e in r["optional_node_roles"] if e.get("utility")]
        if opt_functional:
            for e in opt_functional:
                lines.append(f"    - {e['node_class']}  ({e['role']}) - {e['frequency']}")
        else:
            lines.append("    - (none)")
        if opt_utility:
            lines.append(f"    - utility/plumbing (some members): "
                         + ", ".join(e["node_class"] for e in opt_utility))
        lines.append("")
        lines.append("- connection patterns (role level):")
        for p in r["connection_patterns"][:30]:
            inv = "invariant" if p["invariant"] else p["frequency"]
            lines.append(f"    - {p['from_role']} -> {p['to_role']}  [{p['data_type']}]  ({inv})")
        if len(r["connection_patterns"]) > 30:
            lines.append(f"    - ... and {len(r['connection_patterns']) - 30} more")
        lines.append("")
        bp = r["boundary_ports"]
        lines.append("- boundary ports:")
        lines.append("    - inputs:  " + (", ".join(
            f"{p['data_type']}({p['role']})" for p in bp["inputs"]) or "(none)"))
        lines.append("    - outputs: " + (", ".join(
            f"{p['data_type']}({p['role']})" for p in bp["outputs"]) or "(none)"))
        lines.append("")
        lines.append(f"- param variability: {r['param_variability']}")
        if r["unresolved_nodes"]:
            lines.append(f"- unresolved nodes (not in object_info): {', '.join(r['unresolved_nodes'])}")
        if r["custom_nodes"]:
            lines.append(f"- custom nodes: {', '.join(r['custom_nodes'])}")
        lines.append("")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def _write_node_knowledge(path, node_knowledge, n_workflows) -> None:
    custom = sum(1 for n in node_knowledge if n["is_custom"])
    unresolved = sum(1 for n in node_knowledge if not n["resolved"])
    payload = {
        "generated_from": {
            "workflow_count": n_workflows,
            "node_class_count": len(node_knowledge),
            "custom_classes": custom,
            "unresolved_classes": unresolved,
        },
        "nodes": node_knowledge,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def main(argv=None) -> int:
    args = build_arg_parser().parse_args(argv)
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
