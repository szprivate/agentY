"""workflow_recipes - discover ComfyUI workflow *types* from a corpus of
workflow JSON files and emit a high-level "recipe" database.

The package is split into independently testable modules:

  parser         - load + normalize workflow JSON (UI and API formats) into a
                   directed graph, expanding ComfyUI subgraphs recursively and
                   enriching node signatures from a cached /object_info response.
  fingerprint    - turn a normalized graph into a structural fingerprint
                   (node-class multiset, typed connection patterns, local
                   neighborhood clusters, spine roles).
  cluster        - threshold-based agglomerative clustering over fingerprint
                   similarity (no pre-specified cluster count).
  recipe_builder - synthesize one recipe record per discovered type (Phase 4).
  cli            - thin command line entry point that wires the phases together.

This tool only *discovers* workflow types and writes the recipe database. It
does not build workflows, select recipes, or wire nodes - those are downstream
components that consume the database this tool produces.
"""

__all__ = ["parser", "fingerprint", "cluster"]
