"""Parse and normalize ComfyUI workflow JSON into a directed graph.

Two on-disk formats are auto-detected per file:

  UI  format: a dict with "nodes" (list) and "links" (list) arrays. Nodes carry
              "type" (class name) and "widgets_values". Links are arrays shaped
              [link_id, src_node, src_slot, dst_node, dst_slot, data_type].
              Newer exports collapse whole graphs into a single node whose
              "type" is a UUID referencing definitions.subgraphs[]; those are
              expanded recursively here so the real structure is recovered.

  API format: a dict of node_id -> {"class_type", "inputs", "_meta"}. An input
              value that is a 2-list [node_id, slot] is a connection; anything
              else is a literal widget value. API edges do not carry a data
              type, so it is inferred from the source node's /object_info
              output signature when available (else "UNKNOWN").

The output is a WorkflowGraph: nodes keyed by a namespaced id, typed edges, and
explicit boundary ports recovered from the outermost subgraph definition.

The ``Corpus`` class owns the end-to-end load: object_info (cache or live fetch),
the catalog descriptions, and the parsed + enriched graphs.

Robustness: malformed JSON is skipped (logged), never fatal to the batch.
Determinism: nodes/edges are emitted in a stable, sorted order.
"""

from __future__ import annotations

import json
import os
import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from .model import Edge, Node, WorkflowGraph

# A type whose name looks like a UUID is a subgraph reference, not a real class.
_UUID_RE = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-", re.IGNORECASE)

# Guard against a subgraph that (directly or transitively) references itself.
_MAX_SUBGRAPH_DEPTH = 16

# ComfyUI subgraph boundary node ids (negative, fixed by the format).
_DEFAULT_INPUT_NODE_ID = -10
_DEFAULT_OUTPUT_NODE_ID = -20


# --------------------------------------------------------------------------- #
# object_info loading / caching
# --------------------------------------------------------------------------- #
def load_object_info(
    cache_path: Optional[str],
    host: str = "127.0.0.1",
    port: int = 8188,
    allow_fetch: bool = True,
    log=print,
) -> Dict[str, Any]:
    """Return the /object_info mapping (class_type -> signature).

    Prefers an on-disk cache so reruns work offline. If the cache is missing and
    a live ComfyUI instance is reachable, fetches and writes the cache. On any
    failure returns an empty dict - parsing still works from UI edge types, and
    unresolved classes are flagged downstream.
    """
    if cache_path and os.path.exists(cache_path):
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            log(f"[object_info] loaded cache: {len(data)} classes from {cache_path}")
            return data
        except Exception as exc:  # noqa: BLE001 - cache is best-effort
            log(f"[object_info] cache unreadable ({exc}); ignoring")

    if not allow_fetch:
        log("[object_info] no cache and fetch disabled; running without signatures")
        return {}

    url = f"http://{host}:{port}/object_info"
    try:
        import urllib.request

        with urllib.request.urlopen(url, timeout=5) as resp:  # noqa: S310 - local tool
            data = json.load(resp)
        log(f"[object_info] fetched {len(data)} classes from {url}")
        if cache_path:
            try:
                os.makedirs(os.path.dirname(os.path.abspath(cache_path)), exist_ok=True)
                with open(cache_path, "w", encoding="utf-8") as f:
                    json.dump(data, f)
                log(f"[object_info] cached to {cache_path}")
            except Exception as exc:  # noqa: BLE001
                log(f"[object_info] could not write cache ({exc})")
        return data
    except Exception as exc:  # noqa: BLE001 - offline is a supported mode
        log(f"[object_info] not reachable at {url} ({exc}); running offline")
        return {}


def _output_types_from_info(info: Dict[str, Any]) -> List[str]:
    """Normalize the 'output' field of an object_info entry to a list of type names."""
    out = info.get("output", [])
    types: List[str] = []
    for t in out if isinstance(out, list) else []:
        if isinstance(t, list):          # COMBO declared as a list of choices
            types.append("COMBO")
        else:
            types.append(str(t))
    return types


def _input_types_from_info(info: Dict[str, Any]) -> Dict[str, str]:
    """Map input name -> declared type for required + optional inputs."""
    result: Dict[str, str] = {}
    spec = info.get("input", {})
    for section in ("required", "optional"):
        for name, decl in (spec.get(section, {}) or {}).items():
            t = decl[0] if isinstance(decl, list) and decl else decl
            result[name] = "COMBO" if isinstance(t, list) else str(t)
    return result


def _is_custom_node(info: Dict[str, Any]) -> bool:
    """Decide whether an object_info entry describes a third-party custom node."""
    module = str(info.get("python_module", ""))
    # Core nodes live in "nodes" or "comfy_extras.*"; everything else is custom.
    return module.startswith("custom_nodes")


def _is_api_node(info: Dict[str, Any]) -> bool:
    """True for ComfyUI API / partner nodes (the cloud-service integrations,
    e.g. Kling, Veo, ByteDance/Seedream, BFL Flux Pro). These live in the
    ``comfy_api_nodes`` python module."""
    return str(info.get("python_module", "")).startswith("comfy_api_nodes")


# --------------------------------------------------------------------------- #
# Template catalog (index.json) descriptions
# --------------------------------------------------------------------------- #
def _merge_index_file(path: str, source: str, out: Dict[str, Dict], log) -> None:
    """Merge one index.json into the {name: metadata} map.

    Two known shapes are handled:
      official - list of {moduleName, title, blueprints:[{name, title,
                 description, mediaType}]}; the entry title is the category.
      custom   - list of {templates:[{name, models, io}]} with no descriptions;
                 names are recorded with empty metadata (gracefully no-op).
    Unknown shapes are ignored, never fatal.
    """
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        log(f"[index] unreadable {path} ({exc}); ignoring")
        return
    if not isinstance(data, list):
        return
    for entry in data:
        if not isinstance(entry, dict):
            continue
        category = entry.get("title")
        for bp in entry.get("blueprints", []) or []:
            name = bp.get("name")
            if not name:
                continue
            out[name] = {
                "category": category,
                "title": bp.get("title"),
                "description": bp.get("description"),
                "media_type": bp.get("mediaType"),
                "source": source,
            }
        for tpl in entry.get("templates", []) or []:   # custom shape, no desc
            name = tpl.get("name")
            if name and name not in out:
                out[name] = {
                    "category": None, "title": tpl.get("title"),
                    "description": tpl.get("description"),
                    "media_type": None, "source": source,
                }


def load_descriptions(
    folders: Dict[str, str],
    templates_descriptions_path: Optional[str] = None,
    log=print,
) -> Dict[str, Dict]:
    """Return {workflow_name: metadata} parsed from every index.json found under
    each {source_label: folder}, then enriched with a flat name->description map
    (e.g. config/workflow_templates.json) for workflows the indexes do not
    describe. Folders are walked, so a catalog in a nested 'templates'
    subdirectory is picked up too.

    Precedence for the description text: an index.json description wins; the flat
    map only fills entries that have no description yet (typically custom
    workflows, whose index carries names but no prose)."""
    out: Dict[str, Dict] = {}
    for source, folder in folders.items():
        if not os.path.isdir(folder):
            continue
        for root, _dirs, files in os.walk(folder):
            for fn in files:
                if fn.lower() == "index.json":
                    _merge_index_file(os.path.join(root, fn), source, out, log)

    if templates_descriptions_path and os.path.exists(templates_descriptions_path):
        try:
            with open(templates_descriptions_path, "r", encoding="utf-8") as f:
                flat = json.load(f)
        except (OSError, json.JSONDecodeError) as exc:
            log(f"[index] unreadable {templates_descriptions_path} ({exc}); ignoring")
            flat = {}
        added = 0
        for name, desc in (flat.items() if isinstance(flat, dict) else []):
            if not isinstance(desc, str):
                continue
            entry = out.setdefault(
                name,
                {"category": None, "title": None, "description": None,
                 "media_type": None, "source": "custom"},
            )
            if not entry.get("description"):
                entry["description"] = desc
                added += 1
        log(f"[index] filled {added} descriptions from {templates_descriptions_path}")

    described = sum(1 for m in out.values() if m.get("description"))
    log(f"[index] catalog metadata for {len(out)} workflows "
        f"({described} with descriptions)")
    return out


# --------------------------------------------------------------------------- #
# Format detection
# --------------------------------------------------------------------------- #
def detect_format(data: Any) -> Optional[str]:
    """Return "ui", "api", or None for an unrecognized top-level structure."""
    if not isinstance(data, dict):
        return None
    if isinstance(data.get("nodes"), list):
        return "ui"
    values = list(data.values())
    if values and all(
        isinstance(v, dict) and "class_type" in v for v in values
    ):
        return "api"
    return None


# --------------------------------------------------------------------------- #
# UI format: recursive subgraph expansion
# --------------------------------------------------------------------------- #
@dataclass
class _Flat:
    """Intermediate result of flattening one container (workflow or subgraph)."""

    nodes: Dict[str, Node]
    edges: List[Edge]
    # boundary input slot -> list of internal (real_id, slot, data_type) it feeds
    input_map: Dict[int, List[Tuple[str, int, str]]]
    # boundary output slot -> ("real", real_id, slot, type) or ("pass", in_slot, type)
    output_map: Dict[int, Tuple]


def _normalize_links(container: Dict[str, Any]) -> List[Tuple[int, int, int, int, str]]:
    """Yield links as (src_id, src_slot, dst_id, dst_slot, data_type) tuples.

    Top-level links are arrays; subgraph-internal links are dicts. Both shapes
    are normalized here. Malformed entries are skipped.
    """
    out: List[Tuple[int, int, int, int, str]] = []
    for link in container.get("links", []) or []:
        try:
            if isinstance(link, dict):
                out.append(
                    (
                        int(link["origin_id"]),
                        int(link["origin_slot"]),
                        int(link["target_id"]),
                        int(link["target_slot"]),
                        str(link.get("type", "UNKNOWN")),
                    )
                )
            elif isinstance(link, (list, tuple)) and len(link) >= 6:
                out.append(
                    (int(link[1]), int(link[2]), int(link[3]), int(link[4]), str(link[5]))
                )
        except (TypeError, ValueError, KeyError):
            continue
    return out


def _flatten(
    container: Dict[str, Any],
    prefix: str,
    registry: Dict[str, Dict[str, Any]],
    depth: int,
) -> _Flat:
    """Recursively flatten a container into real nodes and real edges.

    Subgraph instances are inlined with namespaced ids; their boundary ports are
    rewired so a connection that crosses a subgraph boundary becomes a direct
    real-node-to-real-node edge in the flattened graph.
    """
    in_id = (container.get("inputNode") or {}).get("id", _DEFAULT_INPUT_NODE_ID)
    out_id = (container.get("outputNode") or {}).get("id", _DEFAULT_OUTPUT_NODE_ID)

    nodes: Dict[str, Node] = {}
    edges: List[Edge] = []
    input_map: Dict[int, List[Tuple[str, int, str]]] = defaultdict(list)
    output_map: Dict[int, Tuple] = {}

    # Boundary maps of child subgraph instances, keyed by their local node id.
    inst_inputs: Dict[int, Dict[int, List[Tuple[str, int, str]]]] = {}
    inst_outputs: Dict[int, Dict[int, Tuple]] = {}

    for node in container.get("nodes", []) or []:
        nid = node.get("id")
        ntype = node.get("type")
        if (
            isinstance(ntype, str)
            and ntype in registry
            and depth < _MAX_SUBGRAPH_DEPTH
        ):
            sub = registry[ntype]
            res = _flatten(sub, f"{prefix}{nid}/", registry, depth + 1)
            nodes.update(res.nodes)
            edges.extend(res.edges)
            inst_inputs[nid] = res.input_map
            inst_outputs[nid] = res.output_map
        else:
            real_id = f"{prefix}{nid}"
            nodes[real_id] = Node(
                id=real_id,
                class_type=str(ntype),
                widgets_values=node.get("widgets_values"),
                title=node.get("title"),
            )

    def resolve_source(sid: int, slot: int, typ: str):
        if sid == in_id:
            return [("bin", slot, typ)]
        if sid in inst_outputs:
            entry = inst_outputs[sid].get(slot)
            if not entry or entry[0] != "real":
                return []          # nothing real produces it (or pass-through)
            return [("real", entry[1], entry[2], typ)]
        return [("real", f"{prefix}{sid}", slot, typ)]

    def resolve_dests(did: int, slot: int, typ: str):
        if did == out_id:
            return [("bout", slot, typ)]
        if did in inst_inputs:
            return [("real", rid, s, typ) for (rid, s, _t) in inst_inputs[did].get(slot, [])]
        return [("real", f"{prefix}{did}", slot, typ)]

    for sid, ss, did, ds, typ in _normalize_links(container):
        for s in resolve_source(sid, ss, typ):
            for d in resolve_dests(did, ds, typ):
                if s[0] == "real" and d[0] == "real":
                    edges.append(Edge(s[1], s[2], d[1], d[2], typ))
                elif s[0] == "real" and d[0] == "bout":
                    output_map[d[1]] = ("real", s[1], s[2], typ)
                elif s[0] == "bin" and d[0] == "real":
                    input_map[s[1]].append((d[1], d[2], typ))
                elif s[0] == "bin" and d[0] == "bout":
                    output_map[d[1]] = ("pass", s[1], typ)

    return _Flat(nodes, edges, dict(input_map), output_map)


def _subgraph_registry(data: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    defs = data.get("definitions", {})
    subgraphs = defs.get("subgraphs", []) if isinstance(defs, dict) else []
    return {sg["id"]: sg for sg in subgraphs if isinstance(sg, dict) and "id" in sg}


def _boundary_ports(data: Dict[str, Any], registry: Dict[str, Dict[str, Any]]):
    """Recover explicit boundary ports from top-level subgraph instances.

    The outermost subgraph definitions enumerate the ports the workflow author
    exposed (e.g. "prompt": STRING, "IMAGE": IMAGE), which makes a curated
    boundary description. Returns (inputs, outputs) as lists of {name, data_type}.
    """
    inputs: List[Dict[str, str]] = []
    outputs: List[Dict[str, str]] = []
    for node in data.get("nodes", []) or []:
        ntype = node.get("type")
        if isinstance(ntype, str) and ntype in registry:
            sub = registry[ntype]
            for port in sub.get("inputs", []) or []:
                inputs.append(
                    {"name": str(port.get("name", "")), "data_type": str(port.get("type", "UNKNOWN"))}
                )
            for port in sub.get("outputs", []) or []:
                outputs.append(
                    {"name": str(port.get("name", "")), "data_type": str(port.get("type", "UNKNOWN"))}
                )
    return inputs, outputs


def parse_ui(data: Dict[str, Any], name: str, path: str, source: str) -> WorkflowGraph:
    registry = _subgraph_registry(data)
    flat = _flatten(data, "", registry, depth=0)
    graph = WorkflowGraph(name=name, path=path, source=source, fmt="ui")
    graph.nodes = flat.nodes
    # Stable edge order for determinism.
    graph.edges = sorted(
        flat.edges,
        key=lambda e: (e.src_id, e.src_slot, e.dst_id, e.dst_slot, e.data_type),
    )
    graph.boundary_inputs, graph.boundary_outputs = _boundary_ports(data, registry)
    return graph


# --------------------------------------------------------------------------- #
# API format
# --------------------------------------------------------------------------- #
def _is_api_link(value: Any, all_ids: set) -> bool:
    """An API input is a connection iff it is [node_id, slot] referencing a node."""
    return (
        isinstance(value, list)
        and len(value) == 2
        and isinstance(value[1], int)
        and str(value[0]) in all_ids
    )


def parse_api(data: Dict[str, Any], name: str, path: str, source: str) -> WorkflowGraph:
    graph = WorkflowGraph(name=name, path=path, source=source, fmt="api")
    all_ids = set(data.keys())

    for nid, node in data.items():
        widgets = {
            k: v
            for k, v in (node.get("inputs", {}) or {}).items()
            if not _is_api_link(v, all_ids)
        }
        graph.nodes[str(nid)] = Node(
            id=str(nid),
            class_type=str(node.get("class_type")),
            widgets_values=widgets,
            title=(node.get("_meta", {}) or {}).get("title"),
        )

    edges: List[Edge] = []
    for nid, node in data.items():
        for slot_index, (in_name, val) in enumerate(
            sorted((node.get("inputs", {}) or {}).items())
        ):
            if _is_api_link(val, all_ids):
                edges.append(
                    Edge(
                        src_id=str(val[0]),
                        src_slot=int(val[1]),
                        dst_id=str(nid),
                        dst_slot=slot_index,
                        data_type="UNKNOWN",   # filled later from object_info
                        dst_input_name=in_name,
                    )
                )
    graph.edges = sorted(
        edges, key=lambda e: (e.dst_id, e.dst_input_name or "", e.src_id, e.src_slot)
    )
    return graph


# --------------------------------------------------------------------------- #
# Enrichment from object_info
# --------------------------------------------------------------------------- #
def enrich(graph: WorkflowGraph, object_info: Dict[str, Any]) -> WorkflowGraph:
    """Attach type signatures from object_info and infer missing edge types.

    - Resolved nodes get input/output types and a custom-vs-core flag.
    - API edges (typeless on disk) get their data type from the source node's
      output signature when the source class is resolved.
    - Classes absent from object_info are recorded as unresolved (flagged).
    """
    unresolved: set = set()
    for node in graph.nodes.values():
        info = object_info.get(node.class_type)
        if info:
            node.resolved = True
            node.is_custom = _is_custom_node(info)
            node.is_api = _is_api_node(info)
            node.input_types = _input_types_from_info(info)
            node.output_types = _output_types_from_info(info)
        else:
            node.resolved = False
            if node.class_type and node.class_type != "None":
                unresolved.add(node.class_type)

    for edge in graph.edges:
        if edge.data_type in ("", "UNKNOWN", "*"):
            src = graph.nodes.get(edge.src_id)
            if src and 0 <= edge.src_slot < len(src.output_types):
                edge.data_type = src.output_types[edge.src_slot]

    graph.unresolved_classes = sorted(unresolved)
    return graph


# --------------------------------------------------------------------------- #
# File-level parsing
# --------------------------------------------------------------------------- #
# Files that are catalogs, not workflows.
_SKIP_BASENAMES = {"index.json", "index.schema.json"}


def parse_file(
    path: str, source: str, object_info: Optional[Dict[str, Any]] = None, log=print
) -> Optional[WorkflowGraph]:
    """Parse a single workflow file. Returns None (logged) on any failure."""
    base = os.path.basename(path)
    if base in _SKIP_BASENAMES:
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        log(f"[skip] {base}: unreadable JSON ({exc})")
        return None

    fmt = detect_format(data)
    name = os.path.splitext(base)[0]
    try:
        if fmt == "ui":
            graph = parse_ui(data, name, path, source)
        elif fmt == "api":
            graph = parse_api(data, name, path, source)
        else:
            log(f"[skip] {base}: unrecognized workflow format")
            return None
    except Exception as exc:  # noqa: BLE001 - never let one file break the batch
        log(f"[skip] {base}: parse error ({exc})")
        return None

    if not graph.nodes:
        log(f"[skip] {base}: no nodes after parsing")
        return None

    enrich(graph, object_info or {})
    return graph


# --------------------------------------------------------------------------- #
# Corpus - the end-to-end loader
# --------------------------------------------------------------------------- #
class Corpus:
    """Owns the workflow corpus: object_info signatures, catalog descriptions,
    and the parsed + enriched ``WorkflowGraph`` list.

    Build one with :meth:`load`, then read ``.graphs`` (the parsed workflows) and
    ``.object_info`` (kept for node-knowledge generation downstream)."""

    def __init__(
        self,
        graphs: List[WorkflowGraph],
        object_info: Dict[str, Any],
        descriptions: Dict[str, Dict],
    ) -> None:
        self.graphs = graphs
        self.object_info = object_info
        self.descriptions = descriptions

    def __iter__(self):
        return iter(self.graphs)

    def __len__(self) -> int:
        return len(self.graphs)

    @classmethod
    def load(
        cls,
        folders: Dict[str, str],
        object_info_cache: Optional[str] = None,
        host: str = "127.0.0.1",
        port: int = 8188,
        allow_fetch: bool = True,
        templates_descriptions: Optional[str] = None,
        log=print,
    ) -> "Corpus":
        """Load object_info (cache or live), catalog descriptions, then parse
        every workflow under each {source_label: folder} mapping."""
        object_info = load_object_info(
            object_info_cache, host=host, port=port, allow_fetch=allow_fetch, log=log
        )
        descriptions = load_descriptions(folders, templates_descriptions, log=log)
        graphs = cls._parse_folders(folders, object_info, descriptions, log=log)
        return cls(graphs, object_info, descriptions)

    @staticmethod
    def _parse_folders(
        folders: Dict[str, str],
        object_info: Optional[Dict[str, Any]] = None,
        descriptions: Optional[Dict[str, Dict]] = None,
        log=print,
    ) -> List[WorkflowGraph]:
        """Parse every workflow under each {source_label: folder} mapping.

        Folders are walked recursively. Each graph is annotated with its catalog
        category/title/description when present. Results are returned sorted by
        (source, name) for deterministic ordering."""
        descriptions = descriptions or {}
        graphs: List[WorkflowGraph] = []
        for source, folder in folders.items():
            if not os.path.isdir(folder):
                log(f"[warn] input folder not found: {folder}")
                continue
            paths = []
            for root, _dirs, files in os.walk(folder):
                for fn in files:
                    if fn.lower().endswith(".json"):
                        paths.append(os.path.join(root, fn))
            for path in sorted(paths):
                graph = parse_file(path, source, object_info, log=log)
                if graph is not None:
                    meta = descriptions.get(graph.name)
                    if meta:
                        graph.category = meta.get("category")
                        graph.index_title = meta.get("title")
                        graph.index_description = meta.get("description")
                        graph.media_type = meta.get("media_type")
                    graphs.append(graph)
        graphs.sort(key=lambda g: (g.source, g.name))
        log(f"[corpus] parsed {len(graphs)} workflows")
        return graphs
