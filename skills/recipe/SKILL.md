---
name: recipe
description: Recipe-based ComfyUI workflow building — the (task -> model -> node clusters) knowledge base. Front door for building a workflow from a recipe; the full build procedure lives in the assemble-new-workflow skill.
allowed-tools: list_workflow_recipes, get_workflow_recipe, get_workflow_catalog, get_workflow_template
---

# Recipe-Based Workflow Building

The **recipe database** is the standard for building ComfyUI workflows: it maps
`task -> model -> node clusters`, telling you which node classes are required,
how they connect, the boundary ports, and which existing templates already
implement a given task+model.

- **`list_workflow_recipes()`** — see the available `(task, model)` recipes.
- **`get_workflow_recipe(task, model)`** — fetch the recipe: `execution` mode,
  `member_workflows` (scaffolds), `required_nodes`, `node_clusters`,
  `connection_patterns`, `boundary_ports`.

**To build:** activate the **`assemble-new-workflow`** skill — it is the full,
step-by-step recipe build procedure (fetch recipe → load a member scaffold →
conform it to the recipe → patch → validate). This `recipe` skill is a shortcut
entry point; `assemble-new-workflow` carries the complete instructions.
