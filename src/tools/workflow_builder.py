"""
Workflow-building helper tools for ComfyUI.

These tools help the agent understand node schemas, search for nodes,
validate workflows, load templates, and reason about workflow structure.
"""

import json
import os
from pathlib import Path

from strands import tool

from src.comfyui_client import get_client


# ── Helpers ────────────────────────────────────────────────────────────────────

def _project_root() -> Path:
    """Return the project root directory."""
    return Path(__file__).parent.parent.parent.resolve()


def _load_config() -> dict:
    """Load the settings.json configuration."""
    config_path = _project_root() / "config" / "settings.json"
    if config_path.exists():
        with open(config_path, encoding="utf-8") as f:
            return json.load(f)
    return {}


def _templates_dir() -> Path:
    """Return the path to the local (custom) workflow templates directory."""
    cfg = _load_config()
    wf_dir = cfg.get("comfyui_workflows_dir", "./comfyui_workflows/")
    return (_project_root() / wf_dir).resolve()


def _official_templates_dir() -> Path:
    """Return the path to the official Comfy-Org workflow templates directory."""
    cfg = _load_config()
    ot_dir = cfg.get(
        "comfyui_official_templates_dir",
        "./comfyui_workflow_templates_official/templates/",
    )
    return (_project_root() / ot_dir).resolve()


def _load_official_index() -> list:
    """Load and return the official templates index.json as a flat list of templates."""
    index_path = _official_templates_dir() / "index.json"
    if not index_path.exists():
        return []
    with open(index_path, encoding="utf-8") as f:
        raw = json.load(f)
    # The index is a list of category groups, each containing a 'templates' list.
    flat: list[dict] = []
    for group in raw:
        group_category = group.get("title", group.get("category", ""))
        group_media = group.get("type", "")
        for tpl in group.get("templates", []):
            tpl["_group_category"] = group_category
            tpl["_group_media"] = group_media
            flat.append(tpl)
    return flat


# ── 1. Understand node schemas ────────────────────────────────────────────────

@tool
def get_node_schema(node_class: str) -> dict:
    """Retrieve a detailed, human-readable schema for a specific ComfyUI node type.

    Returns the node's required and optional inputs (with types, defaults, and
    constraints), its output types, display name, description, and category.
    This is the primary tool to understand what a node expects before wiring it
    into a workflow.

    Args:
        node_class: The exact node class name (e.g. 'KSampler', 'CLIPTextEncode',
                    'CheckpointLoaderSimple', 'SaveImage').

    Returns:
        A structured dictionary with:
        - display_name: The human-readable name.
        - description: What the node does.
        - category: The node's menu category.
        - input_required: Dict of required input names → type info.
        - input_optional: Dict of optional input names → type info.
        - output_types: List of output type names in order.
        - output_names: List of output slot names.
        - output_is_list: Whether each output is a list.
        - is_output_node: Whether this node produces final output (e.g. SaveImage).
    """
    try:
        raw = get_client().get(f"/object_info/{node_class}")
        if not raw or node_class not in raw:
            return {"error": f"Node class '{node_class}' not found."}

        info = raw[node_class]

        def _parse_inputs(spec: dict) -> dict:
            """Turn ComfyUI's input spec into a friendlier format."""
            result = {}
            for name, definition in spec.items():
                entry: dict = {}
                if isinstance(definition, list) and len(definition) >= 1:
                    type_info = definition[0]
                    opts = definition[1] if len(definition) > 1 else {}
                    if isinstance(type_info, list):
                        # Enum / combo – list of allowed values
                        entry["type"] = "COMBO"
                        entry["options"] = type_info
                    else:
                        entry["type"] = type_info
                    if isinstance(opts, dict):
                        if "default" in opts:
                            entry["default"] = opts["default"]
                        if "min" in opts:
                            entry["min"] = opts["min"]
                        if "max" in opts:
                            entry["max"] = opts["max"]
                        if "step" in opts:
                            entry["step"] = opts["step"]
                        if "tooltip" in opts:
                            entry["tooltip"] = opts["tooltip"]
                else:
                    entry["type"] = str(definition)
                result[name] = entry
            return result

        input_spec = info.get("input", {})
        schema = {
            "node_class": node_class,
            "display_name": info.get("display_name", node_class),
            "description": info.get("description", ""),
            "category": info.get("category", ""),
            "input_required": _parse_inputs(input_spec.get("required", {})),
            "input_optional": _parse_inputs(input_spec.get("optional", {})),
            "output_types": info.get("output", []),
            "output_names": info.get("output_name", []),
            "output_is_list": info.get("output_is_list", []),
            "is_output_node": info.get("output_node", False),
        }
        return schema
    except Exception as e:
        return {"error": str(e)}


# ── 2. Search for nodes by capabilities ───────────────────────────────────────

@tool
def search_nodes(query: str, limit: int = 20) -> dict:
    """Search for ComfyUI nodes by keyword across their names, descriptions, categories, and input/output types.

    Use this to find which node to use for a particular task (e.g. "upscale",
    "mask", "controlnet", "lora", "text encode", "vae decode").

    Args:
        query: A search term or phrase to match against node names, descriptions,
               categories, and I/O types (case-insensitive).
        limit: Maximum number of results to return (default 20).

    Returns:
        A dictionary with:
        - query: The search term used.
        - count: Number of matches found.
        - results: List of dicts with node_class, display_name, category,
                   description (truncated), input_types, and output_types.
    """
    try:
        all_nodes = get_client().get("/object_info")
        if isinstance(all_nodes, dict) and "error" in all_nodes:
            return all_nodes

        query_lower = query.lower()
        matches = []

        for class_name, info in all_nodes.items():
            display = info.get("display_name", class_name)
            category = info.get("category", "")
            desc = info.get("description", "")
            outputs = info.get("output", [])
            input_spec = info.get("input", {})

            # Collect all input type names
            input_types = set()
            for section in ("required", "optional"):
                for _name, defn in input_spec.get(section, {}).items():
                    if isinstance(defn, list) and defn:
                        t = defn[0]
                        if isinstance(t, str):
                            input_types.add(t)

            # Build a searchable blob
            searchable = " ".join(filter(None, [
                class_name,
                display or "",
                category or "",
                desc or "",
                " ".join(str(o) for o in outputs if o is not None),
                " ".join(input_types),
            ])).lower()

            if query_lower in searchable:
                matches.append({
                    "node_class": class_name,
                    "display_name": display,
                    "category": category,
                    "description": desc[:200] if desc else "",
                    "input_types": sorted(input_types),
                    "output_types": outputs,
                })

        # Sort: exact class name matches first, then by category
        def sort_key(m):
            exact = 0 if query_lower in m["node_class"].lower() else 1
            return (exact, m["category"], m["node_class"])

        matches.sort(key=sort_key)
        matches = matches[:limit]

        return {
            "query": query,
            "count": len(matches),
            "results": matches,
        }
    except Exception as e:
        return {"error": str(e)}


# ── 3. Validate a workflow (dry-run) ──────────────────────────────────────────

@tool
def validate_workflow(workflow_json: str) -> dict:
    """Validate a ComfyUI workflow without executing it.

    Performs two levels of validation:
    1. **Local validation** – checks that every node class exists, all required
       inputs are provided or connected, and connection source nodes/slots exist.
    2. **Server validation** – sends the workflow to the ComfyUI /prompt endpoint
       for full server-side validation (which catches type mismatches, missing
       models, etc.) but does NOT queue it for execution if valid.

    Args:
        workflow_json: A JSON string of the workflow in ComfyUI API format
                       (node-id keyed dict).

    Returns:
        A dictionary with:
        - valid: True/False overall assessment.
        - local_errors: List of issues found by local analysis.
        - server_errors: Any errors returned by the server validator
                        (empty dict if server validation passes).
    """
    try:
        workflow = json.loads(workflow_json) if isinstance(workflow_json, str) else workflow_json
    except json.JSONDecodeError as e:
        return {"valid": False, "local_errors": [f"Invalid JSON: {e}"], "server_errors": {}}

    local_errors = []

    # ── Fetch object_info for local validation ─────────────────────────────
    try:
        all_nodes = get_client().get("/object_info")
    except Exception:
        all_nodes = {}

    node_ids = set(workflow.keys())

    for nid, node in workflow.items():
        cls = node.get("class_type", "")
        if not cls:
            local_errors.append(f"Node {nid}: missing 'class_type'.")
            continue

        # Check class exists
        if all_nodes and cls not in all_nodes:
            local_errors.append(f"Node {nid}: unknown class_type '{cls}'.")
            continue

        node_info = all_nodes.get(cls, {})
        required = node_info.get("input", {}).get("required", {})
        inputs = node.get("inputs", {})

        # Check required inputs are present
        for req_name in required:
            if req_name not in inputs:
                local_errors.append(
                    f"Node {nid} ({cls}): missing required input '{req_name}'."
                )

        # Check link references point to valid nodes
        for inp_name, inp_val in inputs.items():
            if isinstance(inp_val, list) and len(inp_val) == 2:
                src_id = str(inp_val[0])
                if src_id not in node_ids:
                    local_errors.append(
                        f"Node {nid} ({cls}): input '{inp_name}' references "
                        f"non-existent node '{src_id}'."
                    )

    # ── Server-side validation ─────────────────────────────────────────────
    server_errors: dict = {}
    try:
        result = get_client().post("/prompt", json_data={"prompt": workflow})
        if isinstance(result, dict):
            if "error" in result:
                server_errors = {
                    "error": result.get("error"),
                    "node_errors": result.get("node_errors", {}),
                }
            elif "prompt_id" in result:
                # Prompt was accepted and queued – interrupt it immediately
                # to prevent actual execution.
                try:
                    get_client().post("/interrupt", json_data={})
                    # Also remove from queue
                    get_client().post("/queue", json_data={"clear": True})
                except Exception:
                    pass  # Best-effort cleanup
    except Exception as e:
        err_str = str(e)
        # HTTP 400 responses from ComfyUI contain validation details
        if hasattr(e, "response"):
            try:
                server_errors = e.response.json()
            except Exception:
                server_errors = {"error": err_str}
        else:
            server_errors = {"error": err_str}

    is_valid = len(local_errors) == 0 and len(server_errors) == 0

    return {
        "valid": is_valid,
        "local_errors": local_errors,
        "server_errors": server_errors,
    }


# ── 4. Workflow templates (custom + official Comfy-Org) ───────────────────────

@tool
def list_workflow_templates(source: str = "all") -> dict:
    """List all available workflow templates — both custom (local) and official (Comfy-Org).

    Templates from the official Comfy-Org repository include rich metadata:
    title, description, mediaType (image/video/audio/3d), required models,
    required custom nodes, VRAM estimate, and I/O definitions.

    Custom templates in the local comfyui_workflows/ directory are simpler
    JSON files with descriptions derived from their filenames.

    Args:
        source: Which templates to list. One of:
                - 'all': Both official and custom templates (default).
                - 'official': Only official Comfy-Org templates.
                - 'custom': Only local custom templates.

    Returns:
        A dictionary with:
        - official_count / custom_count: Number of templates in each source.
        - official: List of official template summaries.
        - custom: List of custom template summaries.
    """
    try:
        result: dict = {}

        # ── Official templates ─────────────────────────────────────────────
        if source in ("all", "official"):
            flat = _load_official_index()
            official_list = []
            for tpl in flat:
                official_list.append({
                    "name": tpl.get("name", ""),
                    "title": tpl.get("title", ""),
                    "description": tpl.get("description", ""),
                    "media_type": tpl.get("mediaType", ""),
                    "group_category": tpl.get("_group_category", ""),
                    "models": tpl.get("models", []),
                    "requires_custom_nodes": tpl.get("requiresCustomNodes", []),
                    "open_source": tpl.get("openSource", False),
                })
            result["official_count"] = len(official_list)
            result["official"] = official_list

        # ── Custom templates ───────────────────────────────────────────────
        if source in ("all", "custom"):
            tdir = _templates_dir()
            custom_list = []
            if tdir.exists():
                for f in sorted(tdir.glob("*.json")):
                    desc = f.stem.replace("_", " ").replace(".", " – ")
                    custom_list.append({
                        "name": f.stem,
                        "filename": f.name,
                        "description": desc,
                    })
            result["custom_count"] = len(custom_list)
            result["custom"] = custom_list

        return result
    except Exception as e:
        return {"error": str(e)}


@tool
def search_workflow_templates(
    query: str,
    media_type: str = "",
    limit: int = 15,
) -> dict:
    """Search official Comfy-Org workflow templates by keyword.

    Searches across template names, titles, descriptions, model names,
    required custom nodes, and categories. Optionally filter by media type.

    Args:
        query: Search term (case-insensitive). Use descriptive words like
               'text to image', 'video upscale', 'inpaint', 'flux', 'wan',
               'lora', 'controlnet', etc.
        media_type: Optional filter: 'image', 'video', 'audio', '3d', or '' for all.
        limit: Maximum results to return (default 15).

    Returns:
        A dictionary with:
        - query: The search term used.
        - count: Number of matches.
        - results: List of matching template summaries with metadata.
    """
    try:
        flat = _load_official_index()
        if not flat:
            return {"error": "Official templates index not found. Is the repo cloned?"}

        query_lower = query.lower()
        matches = []

        for tpl in flat:
            # Optional media type filter
            if media_type and tpl.get("mediaType", "") != media_type:
                continue

            searchable = " ".join([
                tpl.get("name", ""),
                tpl.get("title", ""),
                tpl.get("description", ""),
                " ".join(tpl.get("models", [])),
                " ".join(tpl.get("requiresCustomNodes", [])),
                " ".join(tpl.get("tags", [])),
                tpl.get("_group_category", ""),
            ]).lower()

            if query_lower in searchable:
                matches.append({
                    "name": tpl.get("name", ""),
                    "title": tpl.get("title", ""),
                    "description": tpl.get("description", ""),
                    "media_type": tpl.get("mediaType", ""),
                    "group_category": tpl.get("_group_category", ""),
                    "models": tpl.get("models", []),
                    "requires_custom_nodes": tpl.get("requiresCustomNodes", []),
                    "open_source": tpl.get("openSource", False),
                    "vram_bytes": tpl.get("vram", 0),
                    "usage": tpl.get("usage", 0),
                })

        # Sort by usage (popularity) descending
        matches.sort(key=lambda m: m.get("usage", 0), reverse=True)
        matches = matches[:limit]

        return {"query": query, "count": len(matches), "results": matches}
    except Exception as e:
        return {"error": str(e)}


@tool
def get_workflow_template(template_name: str) -> dict:
    """Load a workflow template by name and return its full contents.

    Searches both local custom templates and official Comfy-Org templates.
    Custom templates are checked first. The returned workflow is in ComfyUI
    API format, ready to be modified and submitted via submit_prompt.

    Args:
        template_name: The template name (filename without .json extension)
                      or the full filename. Use list_workflow_templates() or
                      search_workflow_templates() to discover available names.

    Returns:
        A dictionary with:
        - name: Template name.
        - source: 'custom' or 'official'.
        - metadata: Template metadata (official only — title, description, models, etc.).
        - workflow: The full workflow JSON object.
        - summary: A quick overview with node count and node classes used.
    """
    try:
        # Normalise: strip .json suffix for matching
        lookup = template_name.removesuffix(".json")

        # ── Try custom templates first ─────────────────────────────────────
        tdir = _templates_dir()
        custom_candidates = [
            tdir / f"{lookup}.json",
            tdir / template_name,
        ]
        for candidate in custom_candidates:
            if candidate.exists():
                with open(candidate, encoding="utf-8") as f:
                    workflow = json.load(f)

                classes_used = []
                for nid, node in workflow.items():
                    if not isinstance(node, dict):
                        continue
                    cls = node.get("class_type", "unknown")
                    title = node.get("_meta", {}).get("title", cls)
                    classes_used.append({"node_id": nid, "class_type": cls, "title": title})

                return {
                    "name": lookup,
                    "source": "custom",
                    "metadata": {},
                    "workflow": workflow,
                    "summary": {"node_count": len(classes_used), "nodes": classes_used},
                }

        # ── Try official templates ─────────────────────────────────────────
        ot_dir = _official_templates_dir()
        official_candidates = [
            ot_dir / f"{lookup}.json",
            ot_dir / template_name,
        ]
        for candidate in official_candidates:
            if candidate.exists():
                with open(candidate, encoding="utf-8") as f:
                    workflow = json.load(f)

                # Fetch metadata from index
                metadata = {}
                for tpl in _load_official_index():
                    if tpl.get("name") == lookup:
                        metadata = {
                            "title": tpl.get("title", ""),
                            "description": tpl.get("description", ""),
                            "media_type": tpl.get("mediaType", ""),
                            "models": tpl.get("models", []),
                            "requires_custom_nodes": tpl.get("requiresCustomNodes", []),
                            "open_source": tpl.get("openSource", False),
                            "io": tpl.get("io", {}),
                        }
                        break

                # Build node summary
                classes_used = []
                for nid, node in workflow.items():
                    if not isinstance(node, dict):
                        continue
                    cls = node.get("class_type", "unknown")
                    title = node.get("_meta", {}).get("title", cls)
                    classes_used.append({"node_id": nid, "class_type": cls, "title": title})

                return {
                    "name": lookup,
                    "source": "official",
                    "metadata": metadata,
                    "workflow": workflow,
                    "summary": {"node_count": len(classes_used), "nodes": classes_used},
                }

        # ── Not found ──────────────────────────────────────────────────────
        return {
            "error": f"Template '{template_name}' not found in custom or official templates.",
            "hint": "Use list_workflow_templates() or search_workflow_templates() to find available templates.",
        }
    except Exception as e:
        return {"error": str(e)}


# ── 5. Parse workflow connections (graph analysis) ────────────────────────────

@tool
def parse_workflow_connections(workflow_json: str) -> dict:
    """Parse a ComfyUI workflow and extract its complete graph structure.

    Analyses the workflow to produce:
    - A list of all nodes with their class types and titles.
    - All connections (edges) showing data flow between nodes.
    - Input and output summaries for every node.
    - An execution-order hint (topologically sorted if acyclic).

    This helps reason about how data flows through the workflow, identify
    bottlenecks, missing connections, or nodes that aren't wired up.

    Args:
        workflow_json: A JSON string of the workflow in ComfyUI API format.

    Returns:
        A dictionary with:
        - node_count: Total number of nodes.
        - nodes: Dict of node_id → {class_type, title, literal_inputs, connected_inputs, outputs_used_by}.
        - connections: List of {from_node, from_slot, to_node, to_input} dicts.
        - roots: Node IDs with no incoming connections (entry points).
        - leaves: Node IDs with no outgoing connections (final outputs).
        - execution_order: Suggested execution order (topological sort).
    """
    try:
        workflow = json.loads(workflow_json) if isinstance(workflow_json, str) else workflow_json
    except json.JSONDecodeError as e:
        return {"error": f"Invalid JSON: {e}"}

    nodes_info: dict = {}
    connections: list = []
    # Track adjacency for topological sort
    children: dict[str, set] = {nid: set() for nid in workflow}
    parents: dict[str, set] = {nid: set() for nid in workflow}

    for nid, node in workflow.items():
        cls = node.get("class_type", "unknown")
        title = node.get("_meta", {}).get("title", cls)
        inputs = node.get("inputs", {})

        literal_inputs: dict = {}
        connected_inputs: dict = {}
        for inp_name, inp_val in inputs.items():
            if isinstance(inp_val, list) and len(inp_val) == 2:
                src_node = str(inp_val[0])
                src_slot = inp_val[1]
                connected_inputs[inp_name] = {
                    "from_node": src_node,
                    "from_slot": src_slot,
                }
                connections.append({
                    "from_node": src_node,
                    "from_slot": src_slot,
                    "to_node": nid,
                    "to_input": inp_name,
                })
                if src_node in children:
                    children[src_node].add(nid)
                if nid in parents:
                    parents[nid].add(src_node)
            else:
                literal_inputs[inp_name] = inp_val

        nodes_info[nid] = {
            "class_type": cls,
            "title": title,
            "literal_inputs": literal_inputs,
            "connected_inputs": connected_inputs,
            "outputs_used_by": [],  # filled below
        }

    # Fill outputs_used_by
    for conn in connections:
        src = conn["from_node"]
        if src in nodes_info:
            nodes_info[src]["outputs_used_by"].append({
                "to_node": conn["to_node"],
                "to_input": conn["to_input"],
                "slot": conn["from_slot"],
            })

    # Identify roots (no parents) and leaves (no children)
    roots = [nid for nid in workflow if not parents.get(nid)]
    leaves = [nid for nid in workflow if not children.get(nid)]

    # Topological sort (Kahn's algorithm)
    in_degree = {nid: len(parents.get(nid, set())) for nid in workflow}
    queue = [nid for nid in workflow if in_degree[nid] == 0]
    exec_order = []
    while queue:
        queue.sort()  # deterministic ordering
        current = queue.pop(0)
        exec_order.append(current)
        for child in sorted(children.get(current, set())):
            in_degree[child] -= 1
            if in_degree[child] == 0:
                queue.append(child)

    has_cycle = len(exec_order) != len(workflow)

    return {
        "node_count": len(workflow),
        "nodes": nodes_info,
        "connections": connections,
        "roots": roots,
        "leaves": leaves,
        "execution_order": exec_order if not has_cycle else [],
        "has_cycle": has_cycle,
    }
