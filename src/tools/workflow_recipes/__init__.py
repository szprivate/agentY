"""workflow_recipes - discover ComfyUI workflow *types* from a corpus of
workflow JSON files and emit a high-level "recipe" database grouped by a
canonical taxonomy: task -> model -> node clusters.

The package is split into cohesive modules:

  model     - the data classes (Node, Edge, WorkflowGraph, Intent).
  parser    - the ``Corpus`` loader: normalize workflow JSON (UI and API
              formats) into directed graphs, expanding ComfyUI subgraphs
              recursively and enriching node signatures from /object_info.
  roles     - functional role classification for node classes.
  intent    - ``IntentClassifier``: derive {media, task, model families} and
              the natural-language matching phrasing.
  taxonomy  - ``TaxonomyClassifier``: classify a workflow into one canonical
              task category.
  recipe    - ``RecipeBuilder`` / ``RecipeDatabase``: synthesize the task->model
              tree, node clusters, and per-node-class knowledge.
  cli       - thin command line entry point that wires the phases together.

This tool only *discovers* workflow types and writes the recipe database. It
does not build workflows, select recipes, or wire nodes - those are downstream
components that consume the database this tool produces.
"""

from .intent import IntentClassifier
from .model import Edge, Intent, Node, WorkflowGraph
from .parser import Corpus
from .recipe import RecipeBuilder, RecipeDatabase
from .taxonomy import TaxonomyClassifier

__all__ = [
    "Corpus",
    "IntentClassifier",
    "TaxonomyClassifier",
    "RecipeBuilder",
    "RecipeDatabase",
    "Node",
    "Edge",
    "WorkflowGraph",
    "Intent",
]
