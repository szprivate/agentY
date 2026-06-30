"""Data model - the normalized graph and the derived intent.

These are plain data holders shared across the package: the parser produces
``WorkflowGraph`` objects, the intent classifier produces ``Intent`` objects,
and everything downstream reads them. Keeping them in one dependency-free module
avoids import cycles between parser / intent / taxonomy / recipe.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class Node:
    """A single resolved node in the normalized graph."""

    id: str                       # namespaced unique id within the workflow
    class_type: str               # e.g. "KSampler"; UUID kept only if unexpandable
    widgets_values: Any = None    # list (UI) or dict (API) of parameter values
    title: Optional[str] = None
    resolved: bool = False        # True if class_type was found in object_info
    is_custom: bool = False       # True if object_info says it is a custom node
    is_api: bool = False          # True if a ComfyUI API / partner node
    input_types: Dict[str, str] = field(default_factory=dict)   # input name -> type
    output_types: List[str] = field(default_factory=list)       # output slot types


@dataclass
class Edge:
    """A typed directed connection: src output slot -> dst input slot."""

    src_id: str
    src_slot: int
    dst_id: str
    dst_slot: int
    data_type: str = "UNKNOWN"
    dst_input_name: Optional[str] = None


@dataclass
class WorkflowGraph:
    """A fully normalized workflow."""

    name: str                     # file basename without extension
    path: str
    source: str                   # "custom" or "official"
    fmt: str                      # "ui" or "api"
    nodes: Dict[str, Node] = field(default_factory=dict)
    edges: List[Edge] = field(default_factory=list)
    boundary_inputs: List[Dict[str, str]] = field(default_factory=list)   # {name, data_type}
    boundary_outputs: List[Dict[str, str]] = field(default_factory=list)  # {name, data_type}
    unresolved_classes: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)

    # Authoritative metadata from the template catalog (index.json), when present.
    category: Optional[str] = None        # human category, e.g. "Image Tools"
    index_title: Optional[str] = None     # human title, e.g. "Brightness and Contrast"
    index_description: Optional[str] = None
    media_type: Optional[str] = None      # e.g. "image" / "video" / "audio"

    # -- convenience accessors used by later phases -----------------------
    def class_of(self, node_id: str) -> str:
        node = self.nodes.get(node_id)
        return node.class_type if node else "UNKNOWN"

    def out_adjacency(self) -> Dict[str, List[Edge]]:
        adj: Dict[str, List[Edge]] = defaultdict(list)
        for e in self.edges:
            adj[e.src_id].append(e)
        return adj

    def in_adjacency(self) -> Dict[str, List[Edge]]:
        adj: Dict[str, List[Edge]] = defaultdict(list)
        for e in self.edges:
            adj[e.dst_id].append(e)
        return adj


@dataclass
class Intent:
    """A workflow's derived user intent: what it makes, how, and with which model."""

    media: Optional[str] = None
    task: Optional[str] = None
    model_families: List[str] = field(default_factory=list)
    keywords: List[str] = field(default_factory=list)
