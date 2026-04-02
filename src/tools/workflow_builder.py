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
def get_node_schema(node_class: str) -> str:
    """Get a structured schema for a ComfyUI node: required/optional inputs with types and defaults, output types, and description.

    Args:
        node_class: Exact node class name e.g. 'KSampler', 'CLIPTextEncode', 'SaveImage'.
    """
    try:
        raw = get_client().get(f"/object_info/{node_class}")
        if not raw or node_class not in raw:
            return json.dumps({"error": f"Node class '{node_class}' not found."})

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
        return json.dumps(schema)
    except Exception as e:
        return json.dumps({"error": str(e)})


# ── 2. Search for nodes by capabilities ───────────────────────────────────────

@tool
def search_nodes(query: str, limit: int = 20) -> str:
    """Search ComfyUI nodes by keyword across names, descriptions, categories, and I/O types.

    Args:
        query: Search term e.g. 'upscale', 'mask', 'lora', 'vae decode'.
        limit: Max results (default 20).
    """
    try:
        all_nodes = get_client().get("/object_info")
        if isinstance(all_nodes, dict) and "error" in all_nodes:
            return json.dumps(all_nodes)

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

        return json.dumps({
            "query": query,
            "count": len(matches),
            "results": matches,
        })
    except Exception as e:
        return json.dumps({"error": str(e)})


# ── 3. Validate a workflow (dry-run) ──────────────────────────────────────────

@tool
def validate_workflow(workflow_json: str) -> str:
    """Validate a ComfyUI workflow (local + server-side) without executing it.

    Returns valid=true/false, local_errors list, and server_errors dict.

    Args:
        workflow_json: Workflow JSON string in ComfyUI API format.
    """
    try:
        workflow = json.loads(workflow_json) if isinstance(workflow_json, str) else workflow_json
    except json.JSONDecodeError as e:
        return json.dumps({"valid": False, "local_errors": [f"Invalid JSON: {e}"], "server_errors": {}})

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

    return json.dumps({
        "valid": is_valid,
        "local_errors": local_errors,
        "server_errors": server_errors,
    })


# ── 4. Workflow templates (custom + official Comfy-Org) ───────────────────────

@tool
def list_workflow_templates(source: str = "all", verbose: bool = False) -> str:
    """List available workflow templates (custom local and/or official Comfy-Org).

    Returns a lean name+description list by default. Set verbose=True to also
    include models and requires_custom_nodes fields for model-compatibility checks.

    Args:
        source: 'all' (default), 'official', or 'custom'.
        verbose: When True, also include 'models' and 'requires_custom_nodes' per entry.
    """
    try:
        result: dict = {}

        # ── Official templates ─────────────────────────────────────────────
        if source in ("all", "official"):
            flat = _load_official_index()
            official_list = []
            for tpl in flat:
                entry: dict = {
                    "name": tpl.get("name", ""),
                    "description": tpl.get("description", ""),
                }
                if verbose:
                    entry["models"] = tpl.get("models", [])
                    entry["requires_custom_nodes"] = tpl.get("requiresCustomNodes", [])
                official_list.append(entry)
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
                        "description": desc,
                    })
            result["custom_count"] = len(custom_list)
            result["custom"] = custom_list

        return json.dumps(result)
    except Exception as e:
        return json.dumps({"error": str(e)})


@tool
def get_workflow_template(template_name: str) -> str:
    """Load a workflow template by name. Checks custom templates first, then official Comfy-Org.

    Returns the full workflow JSON plus metadata and a node summary.

    Args:
        template_name: Template name (without .json) from list_workflow_templates().
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

                return json.dumps({
                    "name": lookup,
                    "source": "custom",
                    "workflow": workflow,
                })

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

                return json.dumps({
                    "name": lookup,
                    "source": "official",
                    "workflow": workflow,
                    "models": metadata.get("models", []),
                    "io": metadata.get("io", {})
                })

        # ── Not found ──────────────────────────────────────────────────────
        return json.dumps({
            "error": f"Template '{template_name}' not found in custom or official templates.",
            "hint": "Use list_workflow_templates() to find available templates.",
        })
    except Exception as e:
        return json.dumps({"error": str(e)})


# ── 5. Parse workflow connections (graph analysis) ────────────────────────────

@tool
def parse_workflow_connections(workflow_json: str) -> str:
    """Parse a ComfyUI workflow and return its graph structure: nodes, connections, roots, leaves, and execution order.

    Args:
        workflow_json: Workflow JSON string in ComfyUI API format.
    """
    try:
        workflow = json.loads(workflow_json) if isinstance(workflow_json, str) else workflow_json
    except json.JSONDecodeError as e:
        return json.dumps({"error": f"Invalid JSON: {e}"})

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

    return json.dumps({
        "node_count": len(workflow),
        "nodes": nodes_info,
        "connections": connections,
        "roots": roots,
        "leaves": leaves,
        "execution_order": exec_order if not has_cycle else [],
        "has_cycle": has_cycle,
    })
