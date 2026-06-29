"""Phase 4 - synthesize one recipe record per discovered workflow type.

A recipe describes a *pattern* (node roles and relationships), not a literal
copied subgraph. The two highest-value behaviors here:

  Invariant detection - a node class present in ALL members of a type is a
  structural invariant (mandatory); one present in only some is optional/variant.

  Paired-node preservation - when a type consistently uses *multiple* instances
  of the same class (e.g. two model loaders for a high-noise/low-noise pair),
  both are surfaced as required. These are never collapsed into one role, and
  where the instances play structurally distinct roles that is recorded too.

Roles are expressed functionally (classify_role); when a node has no known role
its class name is used so custom/unknown nodes stay distinguishable.

The database is self-contained: every type carries a populated ``description``
(authoritative catalog text when available, else synthesized from intent and
structure) and a ``user_intent`` block (media / task / model families /
when_to_use / example_requests) so the researcher can match a request like
"build a video workflow using WAN 2.2" without any human annotation step.
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Tuple

from . import intent as intent_mod
from . import taxonomy as taxonomy_mod
from .cluster import Cluster
from .fingerprint import Fingerprint, classify_role, role_description, SPINE_ROLES
from .parser import WorkflowGraph


# Honor the hyphen-only rule: normalize en/em dashes and arrows in any embedded
# or synthesized text (catalog descriptions contain "->" written as an arrow).
_SANITIZE = {
    "→": "->", "←": "<-", "–": "-", "—": "-",
    "‒": "-", "―": "-", "‘": "'", "’": "'",
    "“": '"', "”": '"',
}


def sanitize_text(text: str) -> str:
    if not text:
        return text
    for bad, good in _SANITIZE.items():
        text = text.replace(bad, good)
    return text

# Data types that represent workflow content crossing the boundary (as opposed
# to internal config like MODEL/CLIP/VAE weights).
_CONTENT_TYPES = {"IMAGE", "VIDEO", "AUDIO", "MASK", "LATENT", "MESH", "STRING"}

_OUTPUT_HINT = {
    "IMAGE": "image", "MASK": "image", "VIDEO": "video", "LATENT": "latent",
    "AUDIO": "audio", "MESH": "model", "STRING": "text",
}

# Short tokens for spine roles, used only to build structural slugs.
_ROLE_ABBR = {
    "model_loader": "ld", "sampler": "sample", "vae_decode": "dec",
    "vae_encode": "enc", "text_encode": "txt", "latent_source": "lat",
    "api_node": "api",
}


def role_or_class(class_type: str) -> str:
    """Functional role when known, else the raw class name (keeps custom nodes
    distinguishable instead of bucketing them all as 'other')."""
    role = classify_role(class_type)
    return role if role != "other" else class_type


# Plumbing nodes that are frequently present in multiples but carry no
# functional intent (primitives, math, switches, reroutes, string/json ops).
# Used only to keep the "paired/multiple required" emphasis on meaningful nodes;
# it does NOT affect fingerprinting or clustering.
_UTILITY_SUBSTRINGS = (
    "primitive", "reroute", "comfymathexpression", "comfynumberconvert",
    "comfyswitchnode", "mathint", "mathfloat", "getimagesize",
    "stringreplace", "stringconcatenate", "jsonextractstring", "previewany",
)


def is_utility(class_type: str) -> bool:
    name = (class_type or "").lower()
    return any(s in name for s in _UTILITY_SUBSTRINGS)


# --------------------------------------------------------------------------- #
# Per-member structural facts
# --------------------------------------------------------------------------- #
def _class_counts(graph: WorkflowGraph) -> Counter:
    return Counter(n.class_type for n in graph.nodes.values())


def _member_connection_roles(graph: WorkflowGraph) -> set:
    """Distinct (from_role, to_role, data_type) tuples present in one member."""
    out = set()
    for e in graph.edges:
        out.add(
            (
                role_or_class(graph.class_of(e.src_id)),
                role_or_class(graph.class_of(e.dst_id)),
                e.data_type or "UNKNOWN",
            )
        )
    return out


def _instance_contexts(graph: WorkflowGraph, class_type: str) -> List[Tuple[Tuple, Tuple]]:
    """For each node of class_type, its (downstream roles, upstream roles)
    context - used to tell apart multiple instances that play distinct roles."""
    out_adj = graph.out_adjacency()
    in_adj = graph.in_adjacency()
    contexts = []
    for nid, node in graph.nodes.items():
        if node.class_type != class_type:
            continue
        downstream = tuple(sorted(role_or_class(graph.class_of(e.dst_id)) for e in out_adj.get(nid, [])))
        upstream = tuple(sorted(role_or_class(graph.class_of(e.src_id)) for e in in_adj.get(nid, [])))
        contexts.append((downstream, upstream))
    return contexts


# --------------------------------------------------------------------------- #
# Node roles: required (invariant) vs optional, with paired-node detection
# --------------------------------------------------------------------------- #
def _node_roles(members: List[WorkflowGraph]) -> Tuple[List[Dict], List[Dict]]:
    n = len(members)
    counts_per_member: Dict[str, List[int]] = defaultdict(lambda: [0] * n)
    for idx, g in enumerate(members):
        for cls, c in _class_counts(g).items():
            counts_per_member[cls][idx] = c

    required: List[Dict] = []
    optional: List[Dict] = []
    for cls in sorted(counts_per_member):
        counts = counts_per_member[cls]
        present_in = sum(1 for c in counts if c > 0)
        present_counts = [c for c in counts if c > 0]
        min_inst = min(present_counts) if present_counts else 0
        max_inst = max(present_counts) if present_counts else 0
        role = classify_role(cls)
        utility = is_utility(cls)
        entry = {
            "role": role_description(role),
            "role_key": role,
            "node_class": cls,
            "utility": utility,
            "frequency": f"all members ({present_in}/{n})" if present_in == n
            else f"{present_in}/{n} members",
            "min_instances": min_inst if present_in == n else 0,
            "max_instances": max_inst,
        }
        if present_in == n:
            # Invariant. A meaningful (non-utility) class consistently present
            # 2+ times is a required paired/multiple node - surface every
            # instance. Utility nodes can repeat too, but are not emphasized.
            if min_inst >= 2 and not utility:
                entry["paired_or_multiple"] = True
                entry["frequency"] += f", {min_inst} required instances"
                distinct = _distinct_instances(members, cls)
                if distinct:
                    entry["distinct_instances"] = distinct
            required.append(entry)
        else:
            optional.append(entry)

    # Required: meaningful paired nodes first, then other functional nodes, then
    # utility plumbing last (so the valuable invariants are not buried).
    required.sort(key=lambda e: (
        not e.get("paired_or_multiple"), e.get("utility", False),
        -e["max_instances"], e["node_class"],
    ))
    optional.sort(key=lambda e: (e.get("utility", False), -e["max_instances"], e["node_class"]))
    return required, optional


def _distinct_instances(members: List[WorkflowGraph], class_type: str) -> List[Dict]:
    """Describe structurally distinct instances of a multiply-required class by
    the recurring downstream/upstream context they appear in across members."""
    ctx_counter: Counter = Counter()
    for g in members:
        for downstream, upstream in _instance_contexts(g, class_type):
            ctx_counter[(downstream, upstream)] += 1
    if len(ctx_counter) < 2:
        return []   # instances are not structurally distinguishable
    distinct = []
    for (downstream, upstream), freq in sorted(ctx_counter.items(), key=lambda kv: (-kv[1], kv[0])):
        distinct.append(
            {
                "feeds_into": list(downstream),
                "fed_by": list(upstream),
                "occurrences": freq,
            }
        )
    return distinct


# --------------------------------------------------------------------------- #
# Connection patterns (role level), with per-type frequency
# --------------------------------------------------------------------------- #
def _connection_patterns(members: List[WorkflowGraph]) -> List[Dict]:
    n = len(members)
    pattern_members: Counter = Counter()
    for g in members:
        for pat in _member_connection_roles(g):
            pattern_members[pat] += 1
    threshold = 1 if n == 1 else max(2, (n + 1) // 2)   # >= ~half the members
    out = []
    for (frm, to, dtype), cnt in pattern_members.items():
        if cnt < threshold:
            continue
        out.append(
            {
                "from_role": frm,
                "to_role": to,
                "data_type": dtype,
                "frequency": f"all members ({cnt}/{n})" if cnt == n else f"{cnt}/{n} members",
                "invariant": cnt == n,
            }
        )
    out.sort(key=lambda p: (not p["invariant"], p["from_role"], p["to_role"], p["data_type"]))
    return out


# --------------------------------------------------------------------------- #
# Boundary ports
# --------------------------------------------------------------------------- #
def _boundary_ports(members: List[WorkflowGraph]) -> Dict[str, List[Dict]]:
    """Prefer explicit subgraph boundary ports; fall back to a structural
    derivation (content leaving source nodes / entering sink nodes)."""
    n = len(members)
    explicit_in: Counter = Counter()
    explicit_out: Counter = Counter()
    have_explicit = False
    for g in members:
        if g.boundary_inputs or g.boundary_outputs:
            have_explicit = True
        for p in g.boundary_inputs:
            explicit_in[(p["data_type"], p.get("name", ""))] += 1
        for p in g.boundary_outputs:
            explicit_out[(p["data_type"], p.get("name", ""))] += 1

    if have_explicit:
        keep = lambda cnt: cnt >= max(1, (n + 1) // 2)
        return {
            "inputs": [
                {"data_type": dt, "role": name or dt}
                for (dt, name), c in sorted(explicit_in.items()) if keep(c)
            ],
            "outputs": [
                {"data_type": dt, "role": name or dt}
                for (dt, name), c in sorted(explicit_out.items()) if keep(c)
            ],
        }

    # Structural fallback (API / flat graphs without explicit ports).
    struct_in: Counter = Counter()
    struct_out: Counter = Counter()
    for g in members:
        in_adj = g.in_adjacency()
        out_adj = g.out_adjacency()
        for nid, node in g.nodes.items():
            has_in = bool(in_adj.get(nid))
            has_out = bool(out_adj.get(nid))
            if not has_in and has_out:           # source node
                for e in out_adj[nid]:
                    if e.data_type in _CONTENT_TYPES:
                        struct_in[(e.data_type, role_or_class(node.class_type))] += 1
            if has_in and not has_out:            # sink node
                for e in in_adj[nid]:
                    if e.data_type in _CONTENT_TYPES:
                        struct_out[(e.data_type, role_or_class(node.class_type))] += 1
    keep = lambda cnt: cnt >= max(1, (n + 1) // 2)
    return {
        "inputs": [
            {"data_type": dt, "role": role} for (dt, role), c in sorted(struct_in.items()) if keep(c)
        ],
        "outputs": [
            {"data_type": dt, "role": role} for (dt, role), c in sorted(struct_out.items()) if keep(c)
        ],
    }


# --------------------------------------------------------------------------- #
# Parameter variability
# --------------------------------------------------------------------------- #
def _param_variability(members: List[WorkflowGraph], required: List[Dict]) -> str:
    """Summarize which invariant single-instance node classes keep constant
    widget values across members versus those that vary."""
    if len(members) == 1:
        return "single member - no cross-member variability to report"
    constant, varying = [], []
    for entry in required:
        cls = entry["node_class"]
        if entry.get("min_instances", 0) != 1:
            continue   # skip multi-instance classes (ambiguous to align)
        value_repr = set()
        for g in members:
            vals = [n.widgets_values for n in g.nodes.values() if n.class_type == cls]
            value_repr.add(repr(vals[0]) if vals else "None")
        (constant if len(value_repr) == 1 else varying).append(cls)
    parts = []
    if constant:
        parts.append("constant across members: " + ", ".join(sorted(constant)))
    if varying:
        parts.append("varies across members: " + ", ".join(sorted(varying)))
    return "; ".join(parts) if parts else "no single-instance invariant params to compare"


# --------------------------------------------------------------------------- #
# Custom / unresolved nodes and annotation policy
# --------------------------------------------------------------------------- #
def _custom_and_unresolved(members: List[WorkflowGraph]) -> Tuple[List[str], List[str]]:
    unresolved: set = set()
    custom: set = set()
    for g in members:
        unresolved.update(g.unresolved_classes)
        for node in g.nodes.values():
            if node.is_custom:
                custom.add(node.class_type)
    return sorted(unresolved), sorted(custom)


def _category_info(members: List[WorkflowGraph]) -> Dict:
    """Aggregate the authoritative catalog category across members.

    Returns the dominant category, the full distribution, and whether the type
    is "pure" (every member shares one category) - low purity is a useful hint
    that a structural cluster spans more than one catalog category."""
    cats = [g.category for g in members if g.category]
    if not cats:
        return {"primary": None, "distribution": {}, "pure": False, "coverage": 0}
    dist = Counter(cats)
    # Deterministic dominant: highest count, then alphabetical.
    primary = sorted(dist.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]
    # "pure" means no category conflict among the *categorized* members. A type
    # with one category plus some uncategorized (custom) members is still pure;
    # only >=2 distinct catalog categories is a genuine span / possible over-merge.
    return {
        "primary": primary,
        "distribution": dict(sorted(dist.items())),
        "pure": len(dist) == 1,
        "spans_multiple": len(dist) > 1,
        "coverage": len(cats),               # members carrying a catalog category
        "uncategorized": len(members) - len(cats),
    }


def _suggested_title(members: List[WorkflowGraph]) -> Optional[str]:
    """A human title hint from the catalog: the most common member title (ties
    broken alphabetically), or None if no member carries a catalog title."""
    titles = [g.index_title for g in members if g.index_title]
    if not titles:
        return None
    return sorted(Counter(titles).items(), key=lambda kv: (-kv[1], kv[0]))[0][0]


def _member_descriptions(members: List[WorkflowGraph]) -> List[Dict]:
    """Authoritative per-member catalog descriptions (only where present)."""
    out = []
    for g in sorted(members, key=lambda g: g.name):
        if g.index_description or g.index_title:
            out.append(
                {"name": g.name, "title": g.index_title,
                 "description": sanitize_text(g.index_description) if g.index_description else None}
            )
    return out


def _user_intent(members: List[WorkflowGraph]) -> Dict:
    """Aggregate per-member intent into a type-level matching surface."""
    intents = [intent_mod.classify_intent(g) for g in members]
    media = _dominant([i.media for i in intents if i.media])
    task = intent_mod.dominant_task([i.task for i in intents if i.task])
    fam_counter: Counter = Counter()
    for i in intents:
        for fam in i.model_families:
            fam_counter[fam] += 1
    # Order families by frequency then name (deterministic).
    families = [f for f, _c in sorted(fam_counter.items(), key=lambda kv: (-kv[1], kv[0]))]
    return {
        "media": media,
        "task": task,
        "model_families": families,
        "when_to_use": intent_mod.when_to_use(media, task, families),
        "example_requests": intent_mod.example_requests(media, task, families),
    }


def _dominant(values: List) -> Optional[str]:
    if not values:
        return None
    return sorted(Counter(values).items(), key=lambda kv: (-kv[1], kv[0]))[0][0]


def _synthesized_description(required: List[Dict], boundary: Dict, user_intent: Dict) -> str:
    """A factual description for types lacking authoritative catalog text.

    Built from the derived intent plus the structural spine, so it is concrete
    and self-contained (no placeholder / draft language)."""
    lead = intent_mod.task_phrase(
        user_intent.get("task"), user_intent.get("media")
    ).capitalize()
    fam = user_intent.get("model_families") or []
    fam_clause = f" using {', '.join(fam)}" if fam else ""
    roles = [e["role_key"] for e in required]
    bits = []
    if "model_loader" in roles:
        bits.append("loads a diffusion model")
    if "vae_loader" in roles or "vae_decode" in roles:
        bits.append("uses a VAE")
    if "text_encode" in roles:
        bits.append("encodes a text prompt")
    if "latent_source" in roles:
        bits.append("starts from an empty latent")
    if "sampler" in roles:
        bits.append("runs a diffusion sampler")
    if "vae_decode" in roles:
        bits.append("decodes the latent to pixels")
    in_types = ", ".join(sorted({p["data_type"] for p in boundary.get("inputs", [])})) or "none"
    out_types = ", ".join(sorted({p["data_type"] for p in boundary.get("outputs", [])})) or "none"
    body = "; ".join(bits) if bits else "applies a sequence of node operations"
    return sanitize_text(
        f"{lead}{fam_clause}. Structurally it {body}. "
        f"Boundary inputs: {in_types}; outputs: {out_types}."
    )


def _description(members: List[WorkflowGraph], required: List[Dict],
                 boundary: Dict, user_intent: Dict) -> Tuple[str, str]:
    """Return an always-populated (description, description_source).

    Authoritative catalog descriptions are preferred (and used even for
    custom-node types). When only some members are described, the catalog text
    is kept and the source notes the mix; when none are, a factual description is
    synthesized from intent + structure."""
    described = [(g.name, g.index_description) for g in members if g.index_description]
    if not described:
        return _synthesized_description(required, boundary, user_intent), "synthesized"

    unique = sorted({sanitize_text(d) for _n, d in described})
    if len(unique) == 1:
        text = unique[0]
    else:
        text = " | ".join(unique)
    source = "catalog" if len(described) == len(members) else "catalog+synthesized"
    return text, source


# --------------------------------------------------------------------------- #
# Slugs (deterministic, structural - not semantic intent)
# --------------------------------------------------------------------------- #
# Priority for choosing the slug's content hint when a type has several output
# data types (e.g. a video graph emits both IMAGE frames and AUDIO).
_OUTPUT_PRIORITY = ["MESH", "VIDEO", "IMAGE", "LATENT", "MASK", "AUDIO", "STRING"]


def _slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def _slug(members: List[WorkflowGraph], boundary: Dict, user_intent: Dict,
          canonical_category: Optional[str] = None) -> str:
    """A readable, deterministic id. Prefers the canonical category (e.g.
    "image_to_video"); falls back to intent (task + model family), then to a
    structural signature."""
    if canonical_category:
        return _slugify(canonical_category)
    task = user_intent.get("task")
    media = user_intent.get("media")
    families = user_intent.get("model_families") or []
    parts: List[str] = []
    if task:
        parts.append(task)
    elif media:
        parts.append(media)
    if families:
        parts.append(_slugify(families[0]))
    base = "_".join(p for p in parts if p)
    if base:
        return base

    # Structural fallback (no task/media/family resolved).
    out_types = {p["data_type"] for p in boundary.get("outputs", [])}
    primary = next((t for t in _OUTPUT_PRIORITY if t in out_types), None)
    hint = _OUTPUT_HINT.get(primary, "") if primary else ""
    spine = sorted({
        classify_role(c)
        for g in members for c in {n.class_type for n in g.nodes.values()}
        if classify_role(c) in SPINE_ROLES
    })
    tokens = [hint] + [_ROLE_ABBR.get(r, r) for r in spine[:3]]
    return "_".join(t for t in tokens if t) or "type"


def _assign_ids(recipes: List[Dict]) -> None:
    """Assign deterministic, de-duplicated ids to recipes (already ordered)."""
    seen: Counter = Counter()
    for r in recipes:
        base = r.pop("_slug")
        seen[base] += 1
        r_id = base if seen[base] == 1 else f"{base}_{seen[base]}"
        # Re-key so 'id' is first in the dict for readability.
        new = {"id": r_id}
        new.update(r)
        r.clear()
        r.update(new)


# --------------------------------------------------------------------------- #
# Node clusters - group the required roles into functional units
# --------------------------------------------------------------------------- #
# Ordered functional groups; each role_key (or class) is placed in the first
# group it matches. This is the "node clusters" view at a recipe leaf.
_CLUSTER_GROUPS = [
    ("inputs", {"image_loader", "video_loader"}),
    ("model loading", {"model_loader", "lora_loader", "clip_loader", "vae_loader"}),
    ("conditioning", {"text_encode", "conditioning_op", "controlnet", "guidance"}),
    ("latent / canvas", {"latent_source", "vae_encode"}),
    ("sampling", {"sampler"}),
    ("decoding", {"vae_decode"}),
    ("output", {"save_output"}),
]


def _node_clusters(required: List[Dict]) -> List[Dict]:
    """Group the required (invariant) node roles into functional clusters - the
    structural building blocks of the recipe."""
    buckets: Dict[str, List[str]] = {name: [] for name, _ in _CLUSTER_GROUPS}
    other: List[str] = []
    for entry in required:
        role = entry.get("role_key", "other")
        label = entry["node_class"]
        if entry.get("min_instances", 1) and entry.get("min_instances", 1) >= 2:
            label += f" (x{entry['min_instances']})"
        placed = False
        for name, roles in _CLUSTER_GROUPS:
            if role in roles:
                buckets[name].append(label)
                placed = True
                break
        if not placed:
            other.append(label)
    clusters = [{"cluster": name, "nodes": sorted(nodes)}
                for name, _ in _CLUSTER_GROUPS for nodes in [buckets[name]] if nodes]
    if other:
        clusters.append({"cluster": "other operations", "nodes": sorted(other)})
    return clusters


# --------------------------------------------------------------------------- #
# Recipe body (shared by flat and hierarchical builders)
# --------------------------------------------------------------------------- #
def _recipe_body(members: List[WorkflowGraph]) -> Dict:
    required, optional = _node_roles(members)
    boundary = _boundary_ports(members)
    unresolved, custom = _custom_and_unresolved(members)
    sources = sorted({g.source for g in members})
    user_intent = _user_intent(members)
    description, description_source = _description(members, required, boundary, user_intent)
    return {
        "source": sources[0] if len(sources) == 1 else "mixed",
        "member_files": sorted(g.name for g in members),
        "member_descriptions": _member_descriptions(members),
        "member_count": len(members),
        "user_intent": user_intent,
        "description": description,
        "description_source": description_source,
        "node_clusters": _node_clusters(required),
        "required_node_roles": required,
        "optional_node_roles": optional,
        "connection_patterns": _connection_patterns(members),
        "boundary_ports": boundary,
        "param_variability": _param_variability(members, required),
        "catalog_category": _category_info(members),
        "unresolved_nodes": unresolved,
        "custom_nodes": custom,
    }


def _primary_model(graph: WorkflowGraph) -> str:
    """The single model family a workflow is filed under (its first detected
    family), or "Generic" when no model family is detected (e.g. GLSL filters)."""
    fams = intent_mod.classify_intent(graph).model_families
    return fams[0] if fams else "Generic"


# --------------------------------------------------------------------------- #
# Top-level builders
# --------------------------------------------------------------------------- #
def build_recipes(
    graphs: List[WorkflowGraph],
    clusters: List[Cluster],
    object_info_available: bool = True,
) -> List[Dict]:
    """Flat recipe list (one per cluster) - used by the description / structural
    grouping modes."""
    recipes: List[Dict] = []
    for c in clusters:
        members = [graphs[m] for m in c.members]
        body = _recipe_body(members)
        canonical_category = _dominant([taxonomy_mod.classify(g) for g in members])
        recipe = {
            "_slug": _slug(members, body["boundary_ports"], body["user_intent"],
                           canonical_category),
            "canonical_category": canonical_category,
            "suggested_title": _suggested_title(members),
            "cohesion": round(c.cohesion, 4),
        }
        recipe.update(body)
        recipes.append(recipe)
    recipes.sort(key=lambda r: (-r["member_count"], r["_slug"], r["member_files"]))
    _assign_ids(recipes)
    return recipes


def build_database(graphs: List[WorkflowGraph]) -> Tuple[List[Dict], List[Dict]]:
    """Hierarchical database: task -> model -> node clusters.

    Level 1 is the canonical task category (taxonomy.classify); level 2 is the
    model family the workflow is filed under; the leaf is a tight recipe (node
    clusters + roles + connections) over the workflows sharing that task+model.

    Returns (tasks_tree, flat_leaves). The flat leaves are reused for
    node-knowledge usage links."""
    groups: Dict[Tuple[str, str], List[int]] = defaultdict(list)
    for i, g in enumerate(graphs):
        groups[(taxonomy_mod.classify(g), _primary_model(g))].append(i)

    by_task: Dict[str, List[Dict]] = defaultdict(list)
    flat_leaves: List[Dict] = []
    for (task, model), idxs in groups.items():
        members = [graphs[i] for i in sorted(idxs)]
        leaf = {"id": f"{_slugify(task)}__{_slugify(model)}", "model": model}
        leaf.update(_recipe_body(members))
        by_task[task].append(leaf)
        flat_leaves.append(leaf)

    tasks: List[Dict] = []
    for task, leaves in by_task.items():
        leaves.sort(key=lambda r: (-r["member_count"], r["model"]))
        tasks.append({
            "task": task,
            "id": _slugify(task),
            "member_count": sum(l["member_count"] for l in leaves),
            "model_count": len(leaves),
            "models": leaves,
        })
    tasks.sort(key=lambda t: (-t["member_count"], t["task"]))
    return tasks, flat_leaves


# --------------------------------------------------------------------------- #
# Node knowledge - per node class used in the corpus (for the wiring "brain")
# --------------------------------------------------------------------------- #
def build_node_knowledge(
    graphs: List[WorkflowGraph],
    recipes: List[Dict],
    object_info: Optional[Dict] = None,
) -> List[Dict]:
    """Describe every node class used in the corpus so the brain can wire nodes
    to standard: role, custom flag, I/O signature (from object_info), and which
    types use it. Bounded to classes actually present in the corpus."""
    object_info = object_info or {}
    name_to_type = {
        mf: r["id"] for r in recipes for mf in r["member_files"]
    }

    occ: Counter = Counter()
    types_by_class: Dict[str, set] = defaultdict(set)
    custom_flag: Dict[str, bool] = {}
    resolved_flag: Dict[str, bool] = {}
    for g in graphs:
        tid = name_to_type.get(g.name)
        for node in g.nodes.values():
            cls = node.class_type
            occ[cls] += 1
            if tid:
                types_by_class[cls].add(tid)
            custom_flag[cls] = custom_flag.get(cls, False) or node.is_custom
            resolved_flag[cls] = resolved_flag.get(cls, False) or node.resolved

    out: List[Dict] = []
    for cls in sorted(occ):
        info = object_info.get(cls, {})
        spec = info.get("input", {}) if isinstance(info, dict) else {}
        required_in = sorted((spec.get("required", {}) or {}).keys())
        optional_in = sorted((spec.get("optional", {}) or {}).keys())
        input_types = {}
        for section in ("required", "optional"):
            for nm, decl in (spec.get(section, {}) or {}).items():
                t = decl[0] if isinstance(decl, list) and decl else decl
                input_types[nm] = "COMBO" if isinstance(t, list) else str(t)
        outputs = []
        for t in (info.get("output", []) if isinstance(info, dict) else []):
            outputs.append("COMBO" if isinstance(t, list) else str(t))
        role = classify_role(cls)
        out.append({
            "class": cls,
            "role": role,
            "role_description": role_description(role),
            "resolved": resolved_flag.get(cls, False),
            "is_custom": custom_flag.get(cls, False),
            "inputs": {"required": required_in, "optional": optional_in, "types": input_types},
            "outputs": outputs,
            "used_in_type_ids": sorted(types_by_class.get(cls, set())),
            "occurrences": occ[cls],
        })
    return out
