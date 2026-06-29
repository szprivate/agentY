"""Phase 2 - structural fingerprinting.

A fingerprint captures a workflow's node-setup *shape* while ignoring parameter
values and incidental node-count noise. Several independent signals are
computed so the clustering phase can weight them (or drop one entirely):

  class_set         - the set of node classes present (presence, not count)
  class_multiset    - class -> count (used for paired-node analysis later)
  connection_set    - typed connection patterns as (src_role, dst_role, type)
  cluster_set       - radius-1 neighborhood signatures (recurring local units)
  spine_set         - which functional "spine" roles are present

Connection patterns are expressed in terms of functional *roles* (sampler,
vae_decode, ...) rather than raw class names, so two workflows that use
different-but-equivalent classes for the same job still match. The raw-class
multiset is kept separately for the invariant/paired-node analysis in Phase 4.

The role vocabulary lives here because role classification is itself a
structural signal; recipe_builder imports classify_role to describe roles.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, FrozenSet, List, Tuple

from .parser import WorkflowGraph

# --------------------------------------------------------------------------- #
# Role classification
# --------------------------------------------------------------------------- #
# Ordered (role, predicate) rules; first match wins. Predicates run on the
# lowercased class name. Kept rule-based and transparent so it is easy to tune.
def _has(*subs: str):
    return lambda c: any(s in c for s in subs)


# Each rule: (role, human description, predicate over lowercased class name).
_ROLE_RULES: List[Tuple[str, str, object]] = [
    ("sampler", "diffusion sampler / denoiser", _has("ksampler", "samplercustom", "sampler")),
    ("model_loader", "diffusion model / UNET loader", _has("unetloader", "checkpointloader", "diffusionmodel", "wanmodel", "modelloader", "load diffusion")),
    ("lora_loader", "LoRA / model patch loader", _has("loraloader", "modelpatch", "lora")),
    ("clip_loader", "text encoder / CLIP loader", _has("cliploader", "dualcliploader", "tripleclip")),
    ("text_encode", "prompt text encoding", _has("textencode", "cliptextencode", "encodeprompt")),
    ("vae_loader", "VAE loader", _has("vaeloader",)),
    ("vae_decode", "latent -> pixel decode", _has("vaedecode",)),
    ("vae_encode", "pixel -> latent encode", _has("vaeencode",)),
    ("latent_source", "empty latent / canvas", _has("emptylatent", "emptysd3", "emptyimage", "emptylatentvideo")),
    ("controlnet", "controlnet / guidance conditioning", _has("controlnet", "control_net")),
    ("upscale", "upscale / resize", _has("upscale", "scaleimage", "imagescale")),
    ("image_loader", "image input / load", _has("loadimage", "loadimagemask")),
    ("video_loader", "video input / load", _has("loadvideo", "vhs_loadvideo")),
    ("conditioning_op", "conditioning combine / edit", _has("conditioning",)),
    ("guidance", "guider / sigma / scheduler", _has("guider", "basicscheduler", "sigmas", "fluxguidance")),
    ("save_output", "save / preview / combine output", _has("saveimage", "previewimage", "savevideo", "vhs_videocombine", "saveaudio", "savelatent")),
    ("api_node", "external API generation node", _has("klingo", "veo", "magnific", "topaz", "meshy", "ideogram", "seedream", "nanobanana", "gemini")),
]

# Roles that act as the structural "spine" of a generation graph.
SPINE_ROLES: FrozenSet[str] = frozenset(
    {"sampler", "model_loader", "vae_decode", "vae_encode", "text_encode",
     "latent_source", "api_node"}
)


def classify_role(class_type: str) -> str:
    """Map a node class to a coarse functional role. Returns "other" if no rule
    matches (custom/unknown nodes still get grouped by class downstream)."""
    name = (class_type or "").lower()
    for role, _desc, pred in _ROLE_RULES:
        if pred(name):
            return role
    return "other"


def role_description(role: str) -> str:
    for r, desc, _pred in _ROLE_RULES:
        if r == role:
            return desc
    return "unclassified node role"


# --------------------------------------------------------------------------- #
# Fingerprint
# --------------------------------------------------------------------------- #
@dataclass
class Fingerprint:
    name: str
    source: str
    class_set: FrozenSet[str]
    class_multiset: Counter
    connection_set: FrozenSet[Tuple[str, str, str]]
    cluster_set: FrozenSet[str]
    spine_set: FrozenSet[str]
    node_count: int = 0
    # Authoritative catalog category as a one-element set (empty if unknown).
    # An optional, off-by-default clustering signal; see DEFAULT_WEIGHTS.
    category_set: FrozenSet[str] = frozenset()

    def signal_sets(self) -> Dict[str, FrozenSet]:
        """Expose the comparable sets keyed by signal name (used by clustering)."""
        return {
            "classes": self.class_set,
            "connections": self.connection_set,
            "clusters": self.cluster_set,
            "spine": self.spine_set,
            "category": self.category_set,
        }


def _neighborhood_signatures(graph: WorkflowGraph) -> FrozenSet[str]:
    """Radius-1 neighborhood signature per node: the node's role together with
    the sorted multiset of roles feeding it and the roles it feeds. Recurring
    local clusters (functional units) collapse to identical signatures across
    workflows, so the set of signatures captures shared local structure."""
    in_adj = graph.in_adjacency()
    out_adj = graph.out_adjacency()
    sigs = set()
    for node_id, node in graph.nodes.items():
        role = classify_role(node.class_type)
        preds = sorted(classify_role(graph.class_of(e.src_id)) for e in in_adj.get(node_id, []))
        succs = sorted(classify_role(graph.class_of(e.dst_id)) for e in out_adj.get(node_id, []))
        sigs.add(f"{role}|<-{','.join(preds)}|->{','.join(succs)}")
    return frozenset(sigs)


def _connection_patterns(graph: WorkflowGraph) -> FrozenSet[Tuple[str, str, str]]:
    """Typed connection patterns expressed as (src_role, dst_role, data_type)."""
    patterns = set()
    for e in graph.edges:
        patterns.add(
            (
                classify_role(graph.class_of(e.src_id)),
                classify_role(graph.class_of(e.dst_id)),
                e.data_type or "UNKNOWN",
            )
        )
    return frozenset(patterns)


def fingerprint(graph: WorkflowGraph) -> Fingerprint:
    """Compute the structural fingerprint of a normalized workflow graph."""
    class_multiset = Counter(n.class_type for n in graph.nodes.values())
    class_set = frozenset(class_multiset)
    spine = frozenset(
        r for r in (classify_role(c) for c in class_set) if r in SPINE_ROLES
    )
    return Fingerprint(
        name=graph.name,
        source=graph.source,
        class_set=class_set,
        class_multiset=class_multiset,
        connection_set=_connection_patterns(graph),
        cluster_set=_neighborhood_signatures(graph),
        spine_set=spine,
        node_count=len(graph.nodes),
        category_set=frozenset({graph.category}) if graph.category else frozenset(),
    )


# Default weighting of the signals. Sums are normalized in clustering, so these
# are relative importances; any can be set to 0 to drop the signal.
#
# "category" is the authoritative catalog category (index.json). It is OFF by
# default (0.0) so clustering stays purely structural; raise it to let catalog
# category nudge the grouping. It is neutral for any pair where either side has
# no category (custom workflows), so it never pulls uncategorized graphs together.
DEFAULT_WEIGHTS: Dict[str, float] = {
    "classes": 0.40,
    "connections": 0.35,
    "clusters": 0.20,
    "spine": 0.05,
    "category": 0.0,
}
