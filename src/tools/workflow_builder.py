"""
Workflow-building helper tools for ComfyUI.

These tools help the agent understand node schemas, search for nodes,
validate workflows, load templates, and reason about workflow structure.
"""

import json
import os
import uuid
from pathlib import Path

from strands import tool

from src.comfyui_client import get_client


# ── Workflow file buffer ───────────────────────────────────────────────────────
# Instead of passing full workflow JSON through the conversation (thousands of
# tokens), workflows are written to a temp directory and referenced by path.
# This avoids bloating the sliding-window context.

_WORKFLOW_DIR = Path(__file__).parent.parent.parent / "output" / "_workflows"

# ── patch_workflow failure guard ───────────────────────────────────────────────
# If patch_workflow fails (returns non-empty errors) more than this many times
# in a single brain session, the tool saves a snapshot and raises StopIteration
# to break the agent loop.

_PATCH_FAIL_LIMIT: int = 3
_patch_fail_count: int = 0
_patch_last_workflow_path: str | None = None


def reset_patch_workflow_guard() -> None:
    """Reset the patch_workflow failure counter.  Call once per brain session."""
    global _patch_fail_count, _patch_last_workflow_path
    _patch_fail_count = 0
    _patch_last_workflow_path = None


def _save_workflow(workflow: dict, name: str = "") -> str:
    """Save *workflow* dict to a JSON file and return the absolute path."""
    _WORKFLOW_DIR.mkdir(parents=True, exist_ok=True)
    stem = name or uuid.uuid4().hex[:8]
    path = _WORKFLOW_DIR / f"{stem}.json"
    path.write_text(json.dumps(workflow, indent=2), encoding="utf-8")
    return str(path.resolve())


def _load_workflow(path_or_json: str) -> dict:
    """Load a workflow from *path_or_json*.

    Accepts either:
    - An absolute file path to a previously saved workflow JSON
    - A raw JSON string (legacy callers)

    If the loaded JSON is in ComfyUI graph/export format (has a top-level
    ``"nodes"`` list), it is automatically converted to ComfyUI API format
    before being returned, so all downstream tools can treat the dict uniformly.
    """
    # Try as file path first
    p = Path(path_or_json)
    if p.exists() and p.suffix == ".json":
        data = json.loads(p.read_text(encoding="utf-8"))
    else:
        # Fallback: try parsing as inline JSON
        data = json.loads(path_or_json)

    # Auto-convert graph format → API format
    if _is_graph_format(data):
        data = _convert_graph_to_api(data)

    return data


def _is_graph_format(workflow: dict) -> bool:
    """Return True if *workflow* is in ComfyUI graph/export format.

    Graph format has a top-level ``"nodes"`` key whose value is a list.
    API format is a flat ``{node_id: node_dict}`` mapping.
    """
    return isinstance(workflow.get("nodes"), list)


# Input types that ComfyUI always wires via links (never appear as widget values)
_LINK_ONLY_TYPES: frozenset[str] = frozenset({
    "MODEL", "CLIP", "VAE", "LATENT", "IMAGE", "MASK", "CONDITIONING",
    "CONTROL_NET", "EMBEDS", "SAMPLER", "SIGMAS", "AUDIO", "VIDEO",
    "SEGS", "BBOX", "UPSCALE_MODEL", "CLIPREGION", "PHOTOMAKER",
    "GEMINI_INPUT_FILES",
})


def _schema_widget_names(schema: dict, linked_names: set[str]) -> list[str]:
    """Return the ordered list of widget input names for a node schema.

    Mirrors the logic ComfyUI's frontend uses to assign *widget_values* entries
    to named inputs: iterate ``required`` then ``optional`` inputs in order,
    skip any that are link-only types or are already linked in this instance.

    Also marks seed-type INT inputs so the caller can skip the frontend-injected
    ``seed_control_mode`` value that immediately follows them in ``widgets_values``
    but is absent from the schema.
    """
    names: list[str] = []
    for section in ("required", "optional"):
        for inp_name, inp_spec in schema.get(section, {}).items():
            if inp_name in linked_names:
                continue  # this slot has a live link in the graph
            inp_type = inp_spec[0] if (isinstance(inp_spec, (list, tuple)) and inp_spec) else ""
            # COMBO types have inp_type as a list of options — those are always widgets
            if isinstance(inp_type, str) and inp_type in _LINK_ONLY_TYPES:
                continue  # link-only type — never a widget
            names.append(inp_name)
    return names


# Values that ComfyUI's frontend injects after a seed widget (seed_control_mode).
# These are NOT part of the schema and must be skipped during widgets_values mapping.
_SEED_CONTROL_VALUES: frozenset[str] = frozenset({"fixed", "randomize", "increment", "decrement"})
# Input names that trigger the seed_control_mode injection
_SEED_INPUT_NAMES: frozenset[str] = frozenset({"seed", "noise_seed"})


def _convert_graph_to_api(workflow: dict) -> dict:
    """Convert a ComfyUI graph-format workflow dict to API format.

    Graph format (exported from the UI) structure::

        {
            "nodes": [{"id": 16, "type": "LoadImage", "inputs": [...],
                       "widgets_values": [...], ...}, ...],
            "links": [[link_id, src_node, src_slot, dst_node, dst_slot, type], ...],
            ...
        }

    API format (accepted by ComfyUI's ``/prompt`` endpoint) structure::

        {
            "16": {"class_type": "LoadImage",
                   "inputs": {"image": "photo.png", "upload": "image"},
                   "_meta": {"title": "Load Image"}},
            ...
        }

    Widget-value → input-name mapping requires the node schema from
    ``object_info``.  If the server is unreachable, widget values are stored
    under a special ``__widgets_values`` key so that ``patch_workflow``'s
    ``widget_values_index`` patching still works; they will be re-mapped on
    the next call once the server is available.
    """
    # ── Build link lookup: link_id → [str(src_node_id), src_slot] ─────────────
    link_table: dict[int, list] = {}
    for link in workflow.get("links", []):
        if isinstance(link, (list, tuple)) and len(link) >= 3:
            link_id, src_node, src_slot = int(link[0]), link[1], link[2]
            link_table[link_id] = [str(src_node), int(src_slot)]

    # ── Fetch node schemas (best-effort; may be unavailable offline) ───────────
    try:
        object_info = _get_object_info()
    except Exception:
        object_info = {}

    api_workflow: dict = {}

    for node in workflow.get("nodes", []):
        if not isinstance(node, dict) or "id" not in node:
            continue

        nid = str(node["id"])
        class_type: str = node.get("type", "unknown")
        api_inputs: dict = {}

        # ── Map linked inputs ──────────────────────────────────────────────────
        # In graph format, node["inputs"] is a list of connector descriptors;
        # each entry with a non-null "link" key refers to a live wired connection.
        linked_names: set[str] = set()
        for connector in node.get("inputs", []):
            if not isinstance(connector, dict):
                continue
            name = connector.get("name", "")
            link_id = connector.get("link")
            if name and link_id is not None and link_id in link_table:
                api_inputs[name] = link_table[link_id]
                linked_names.add(name)

        # ── Map widget values → named inputs ───────────────────────────────────
        widgets_values: list = node.get("widgets_values", node.get("widget_values", []))
        if isinstance(widgets_values, list) and widgets_values:
            schema = object_info.get(class_type, {}).get("input", {}) if object_info else {}
            if schema:
                widget_names = _schema_widget_names(schema, linked_names)
                wv_idx = 0  # index into widgets_values
                for name in widget_names:
                    if wv_idx >= len(widgets_values):
                        break
                    val = widgets_values[wv_idx]
                    api_inputs[name] = val
                    wv_idx += 1
                    # Skip the frontend-injected seed_control_mode value that
                    # immediately follows any seed-type widget in widgets_values
                    # but is absent from the schema.
                    if (name in _SEED_INPUT_NAMES
                            and wv_idx < len(widgets_values)
                            and widgets_values[wv_idx] in _SEED_CONTROL_VALUES):
                        wv_idx += 1  # skip "fixed"/"randomize"/… entry
                # Any remaining widgets_values entries are extra frontend values
                for extra_i, extra_val in enumerate(widgets_values[wv_idx:], start=wv_idx):
                    api_inputs[f"__extra_widget_{extra_i}"] = extra_val
            else:
                # Server unavailable — store as indexed list for widget_values_index patching
                api_inputs["__widgets_values"] = list(widgets_values)

        # ── Assemble API node ──────────────────────────────────────────────────
        api_node: dict = {"class_type": class_type, "inputs": api_inputs}
        title = node.get("title", "")
        if title:
            api_node["_meta"] = {"title": title}

        api_workflow[nid] = api_node

    return api_workflow


# ── /object_info cache ─────────────────────────────────────────────────────────
# The full node database from ComfyUI doesn't change during a session.
# Caching avoids re-downloading it on every search_nodes / validate call.

_object_info_cache: dict | None = None


def _get_object_info() -> dict:
    """Return the full /object_info dict, cached after first fetch."""
    global _object_info_cache
    if _object_info_cache is None:
        _object_info_cache = get_client().get("/object_info")
    return _object_info_cache


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


# ── 1b. Inspect a specific node inside a workflow ─────────────────────────────

@tool
def get_workflow_node_info(node_id: str, workflow_path: str) -> str:
    """Return full metadata for a single node inside a saved workflow.

    Combines the node's current state from the workflow (class_type, title,
    literal inputs, connected inputs, widget values) with the ComfyUI schema
    for its class (required/optional inputs with types and defaults, outputs).
    Use this to understand exactly what a node does, what values it holds, and
    what it is connected to before editing it.

    Args:
        node_id: The node's key inside the workflow JSON, e.g. "6" or "190".
        workflow_path: File path to the workflow JSON (from get_workflow_template
            or save_workflow / patch_workflow).
    """
    # ── Load workflow ──────────────────────────────────────────────────────────
    try:
        workflow = _load_workflow(workflow_path)
    except (json.JSONDecodeError, FileNotFoundError, OSError) as e:
        return json.dumps({"error": f"Cannot load workflow: {e}"})

    node_id = str(node_id)
    if node_id not in workflow:
        available = sorted(workflow.keys())
        return json.dumps({"error": f"Node '{node_id}' not found.", "available_node_ids": available})

    raw_node = workflow[node_id]
    cls = raw_node.get("class_type", "")
    title = raw_node.get("_meta", {}).get("title", cls)

    # ── Split inputs into literal values and node-link connections ─────────────
    inputs_raw = raw_node.get("inputs", {})
    literal_inputs: dict = {}
    connected_inputs: dict = {}
    for name, val in inputs_raw.items():
        if isinstance(val, list) and len(val) == 2 and isinstance(val[1], int):
            connected_inputs[name] = {"from_node": str(val[0]), "from_slot": val[1]}
        else:
            literal_inputs[name] = val

    # ── Fetch ComfyUI schema for this class ────────────────────────────────────
    schema: dict = {}
    if cls:
        try:
            all_nodes = _get_object_info()
            if cls in all_nodes:
                info = all_nodes[cls]

                def _parse_inputs(spec: dict) -> dict:
                    result = {}
                    for inp_name, definition in spec.items():
                        entry: dict = {}
                        if isinstance(definition, list) and len(definition) >= 1:
                            type_info = definition[0]
                            opts = definition[1] if len(definition) > 1 else {}
                            if isinstance(type_info, list):
                                entry["type"] = "COMBO"
                                entry["options"] = type_info
                            else:
                                entry["type"] = type_info
                            if isinstance(opts, dict):
                                for key in ("default", "min", "max", "step", "tooltip"):
                                    if key in opts:
                                        entry[key] = opts[key]
                        else:
                            entry["type"] = str(definition)
                        result[inp_name] = entry
                    return result

                input_spec = info.get("input", {})
                schema = {
                    "display_name": info.get("display_name", cls),
                    "description": info.get("description", ""),
                    "category": info.get("category", ""),
                    "input_required": _parse_inputs(input_spec.get("required", {})),
                    "input_optional": _parse_inputs(input_spec.get("optional", {})),
                    "output_types": info.get("output", []),
                    "output_names": info.get("output_name", []),
                    "is_output_node": info.get("output_node", False),
                }
            else:
                schema = {"warning": f"Class '{cls}' not found in ComfyUI object_info."}
        except Exception as e:
            schema = {"warning": f"Could not fetch schema: {e}"}

    result = {
        "node_id": node_id,
        "class_type": cls,
        "title": title,
        "literal_inputs": literal_inputs,
        "connected_inputs": connected_inputs,
        "widget_values": raw_node.get("widgets_values", raw_node.get("widget_values")),
        "schema": schema,
    }
    return json.dumps(result)


# ── 2. Search for nodes by capabilities ───────────────────────────────────────

@tool
def search_nodes(query: str, limit: int = 10) -> str:
    """Search ComfyUI nodes by keyword across names, descriptions, and categories.

    Args:
        query: Search term e.g. 'upscale', 'mask', 'lora', 'vae decode'.
        limit: Max results (default 10).
    """
    try:
        all_nodes = _get_object_info()
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
                    "description": desc[:120] if desc else "",
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
def validate_workflow(workflow_path: str) -> str:
    """Validate a ComfyUI workflow (local + server-side) without executing it.

    Returns valid=true/false, local_errors list, and server_errors dict.

    Args:
        workflow_path: File path to the workflow JSON (from get_workflow_template or save_workflow).
    """
    try:
        workflow = _load_workflow(workflow_path)
    except (json.JSONDecodeError, FileNotFoundError, OSError) as e:
        return json.dumps({"valid": False, "local_errors": [f"Cannot load workflow: {e}"], "server_errors": {}})

    local_errors = []

    # ── Fetch object_info for local validation ─────────────────────────────
    try:
        all_nodes = _get_object_info()
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
def get_workflow_catalog() -> str:
    """Return the workflow template catalog as a flat {name: description} dictionary.

    This is the cheapest way for the Researcher to discover available templates.
    The dictionary keys are the exact names to pass to get_workflow_template().
    """
    catalog_path = _project_root() / "config" / "workflow_templates.json"
    try:
        return catalog_path.read_text(encoding="utf-8")
    except Exception as exc:
        return json.dumps({"error": str(exc)})


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
    """Load a workflow template by name. Saves the full workflow to a file and returns a compact summary with the file path.

    The returned summary includes: node list (id, class, title, key literal inputs),
    model info, and io metadata. The full workflow JSON is at the returned
    ``workflow_path`` — pass that path to validate_workflow / submit_prompt.

    Args:
        template_name: Template name (without .json) from the workflow-templates skill or list_workflow_templates().
    """
    try:
        # Normalise: strip .json suffix for matching
        lookup = template_name.removesuffix(".json")
        workflow = None
        source = ""
        metadata: dict = {}

        # ── Try custom templates first ─────────────────────────────────────
        tdir = _templates_dir()
        for candidate in [tdir / f"{lookup}.json", tdir / template_name]:
            if candidate.exists():
                with open(candidate, encoding="utf-8") as f:
                    workflow = json.load(f)
                source = "custom"
                break

        # ── Try official templates ─────────────────────────────────────────
        if workflow is None:
            ot_dir = _official_templates_dir()
            for candidate in [ot_dir / f"{lookup}.json", ot_dir / template_name]:
                if candidate.exists():
                    with open(candidate, encoding="utf-8") as f:
                        workflow = json.load(f)
                    source = "official"
                    # Fetch metadata from index
                    for tpl in _load_official_index():
                        if tpl.get("name") == lookup:
                            metadata = {
                                "models": tpl.get("models", []),
                                "io": tpl.get("io", {}),
                            }
                            break
                    break

        if workflow is None:
            return json.dumps({
                "error": f"Template '{template_name}' not found.",
                "hint": "Use list_workflow_templates() or the workflow-templates skill.",
            })

        # ── Normalise to API format ────────────────────────────────────────
        # Official Comfy-Org templates are stored in graph/export format.
        # All downstream tools (patch_workflow, validate_workflow, submit_prompt)
        # expect API format, so convert once here and save the API-format copy.
        converted = False
        if _is_graph_format(workflow):
            workflow = _convert_graph_to_api(workflow)
            converted = True

        # ── Save full workflow to file ─────────────────────────────────────
        workflow_path = _save_workflow(workflow, name=lookup)

        # ── Build compact node summary ─────────────────────────────────────
        node_summary = []
        for nid, node in workflow.items():
            if not isinstance(node, dict):
                continue
            cls = node.get("class_type", "unknown")
            title = node.get("_meta", {}).get("title", cls)
            # Include key literal inputs (skip links which are lists)
            inputs = node.get("inputs", {})
            key_inputs = {
                k: v for k, v in inputs.items()
                if not isinstance(v, list) and v is not None and v != ""
            }
            entry: dict = {"id": nid, "class": cls, "title": title}
            if key_inputs:
                entry["inputs"] = key_inputs
            node_summary.append(entry)

        result: dict = {
            "name": lookup,
            "source": source,
            "workflow_path": workflow_path,
            "node_count": len(node_summary),
            "nodes": node_summary,
        }
        if converted:
            result["converted_from_graph_format"] = True
        if metadata.get("models"):
            result["models"] = metadata["models"]
        if metadata.get("io"):
            result["io"] = metadata["io"]

        return json.dumps(result)
    except Exception as e:
        return json.dumps({"error": str(e)})


@tool
def save_workflow(workflow_json: str, name: str = "") -> str:
    """Save a modified workflow JSON to a file and return the file path.

    Use this after patching a workflow template. Pass the returned path to
    validate_workflow() and submit_prompt().

    Args:
        workflow_json: The complete workflow JSON string in ComfyUI API format.
        name: Optional name for the file (default: auto-generated).
    """
    try:
        workflow = json.loads(workflow_json) if isinstance(workflow_json, str) else workflow_json
        path = _save_workflow(workflow, name=name)
        return json.dumps({"workflow_path": path, "node_count": len([k for k in workflow if isinstance(workflow.get(k), dict)])})
    except json.JSONDecodeError as e:
        return json.dumps({"error": f"Invalid JSON: {e}"})
    except Exception as e:
        return json.dumps({"error": str(e)})


# ── 4b. Patch workflow in-place (token-efficient) ────────────────────────────

@tool
def patch_workflow(workflow_path: str, patches: str) -> str:
    """Apply targeted edits to a saved workflow without re-outputting the full JSON.

    Much more token-efficient than save_workflow for modifying templates.
    Each patch targets a specific node input or widget value.

    Args:
        workflow_path: File path to the workflow JSON (from get_workflow_template).
        patches: JSON string — a list of patch objects. Each patch object has:
            - node_id (str): The node ID to modify (e.g. "6", "190").
            - input_name (str): The input field name to set (e.g. "text", "image", "filename").
            - value: The new value (string, number, bool, or list for links like [node_id, slot]).
            Optional fields:
            - widget_values_index (int): If set, patch widget_values[index] inside the node instead of inputs.
            - class_type (str): If set, change the node's class_type.
            Example: [{"node_id": "6", "input_name": "text", "value": "a photo of a chimp"},
                      {"node_id": "190", "input_name": "image", "value": "image.png"}]
    """
    try:
        workflow = _load_workflow(workflow_path)
    except (json.JSONDecodeError, FileNotFoundError, OSError) as e:
        return json.dumps({"error": f"Cannot load workflow: {e}"})

    try:
        patch_list = json.loads(patches) if isinstance(patches, str) else patches
    except json.JSONDecodeError as e:
        return json.dumps({"error": f"Invalid patches JSON: {e}"})

    if not isinstance(patch_list, list):
        return json.dumps({"error": "patches must be a JSON array of patch objects."})

    applied: list[str] = []
    errors: list[str] = []

    for i, patch in enumerate(patch_list):
        nid = str(patch.get("node_id", ""))
        if nid not in workflow:
            errors.append(f"Patch {i}: node '{nid}' not found in workflow.")
            continue

        node = workflow[nid]

        # Optional: change class_type
        if "class_type" in patch:
            node["class_type"] = patch["class_type"]
            applied.append(f"Node {nid}: class_type → {patch['class_type']}")

        # Patch widget_values by index
        if "widget_values_index" in patch:
            idx = int(patch["widget_values_index"])
            wv = node.get("widget_values", node.get("widgets_values"))
            wv_key = "widget_values" if "widget_values" in node else "widgets_values"
            if isinstance(wv, list) and 0 <= idx < len(wv):
                wv[idx] = patch["value"]
                node[wv_key] = wv
                applied.append(f"Node {nid}: {wv_key}[{idx}] → {patch['value']!r}")
            else:
                errors.append(f"Patch {i}: node '{nid}' has no {wv_key}[{idx}].")
            continue

        # Patch inputs
        inp_name = patch.get("input_name")
        if inp_name:
            if "inputs" not in node:
                node["inputs"] = {}
            node["inputs"][inp_name] = patch["value"]
            val_repr = repr(patch["value"])
            if len(val_repr) > 80:
                val_repr = val_repr[:77] + "..."
            applied.append(f"Node {nid}.inputs.{inp_name} → {val_repr}")

    # Save back
    path = _save_workflow(workflow, name=Path(workflow_path).stem)

    # ── failure guard ──────────────────────────────────────────────────────────
    global _patch_fail_count, _patch_last_workflow_path
    _patch_last_workflow_path = path

    if errors:
        _patch_fail_count += 1
        print(
            f"[patch_workflow] Failure {_patch_fail_count}/{_PATCH_FAIL_LIMIT}: "
            f"{len(errors)} patch error(s)."
        )

        if _patch_fail_count >= _PATCH_FAIL_LIMIT:
            # Save a human-readable debug snapshot with a distinct name.
            debug_name = f"{Path(workflow_path).stem}_patch_debug"
            debug_path = _save_workflow(workflow, name=debug_name)
            print(
                f"[patch_workflow] LIMIT REACHED — debug snapshot saved to: {debug_path}"
            )
            return json.dumps({
                "workflow_path": path,
                "applied": applied,
                "errors": errors,
                "patch_count": len(applied),
                "patch_failure_limit_reached": True,
                "debug_workflow_path": debug_path,
                "message": (
                    f"patch_workflow has failed {_PATCH_FAIL_LIMIT} times. "
                    f"Current workflow snapshot saved to: {debug_path}. "
                    "STOP — do not call patch_workflow again. "
                    "Report the debug_workflow_path to the user and ask for guidance."
                ),
            })
    else:
        # Reset counter on a fully-successful patch (no errors).
        _patch_fail_count = 0

    return json.dumps({
        "workflow_path": path,
        "applied": applied,
        "errors": errors,
        "patch_count": len(applied),
    })


@tool
def add_workflow_node(workflow_path: str, node_id: str, class_type: str, inputs: str = "{}", meta_title: str = "") -> str:
    """Add a new node to an existing workflow file.

    Args:
        workflow_path: File path to the workflow JSON.
        node_id: The node ID string (e.g. "200"). Must not already exist.
        class_type: The ComfyUI node class (e.g. "LoadImage", "CLIPTextEncode").
        inputs: JSON string of the node's inputs dict.
        meta_title: Optional display title for the node.
    """
    try:
        workflow = _load_workflow(workflow_path)
    except (json.JSONDecodeError, FileNotFoundError, OSError) as e:
        return json.dumps({"error": f"Cannot load workflow: {e}"})

    if node_id in workflow:
        return json.dumps({"error": f"Node '{node_id}' already exists."})

    try:
        inputs_dict = json.loads(inputs) if isinstance(inputs, str) else inputs
    except json.JSONDecodeError as e:
        return json.dumps({"error": f"Invalid inputs JSON: {e}"})

    node: dict = {"class_type": class_type, "inputs": inputs_dict}
    if meta_title:
        node["_meta"] = {"title": meta_title}

    workflow[node_id] = node
    path = _save_workflow(workflow, name=Path(workflow_path).stem)

    return json.dumps({
        "workflow_path": path,
        "added_node": node_id,
        "class_type": class_type,
        "node_count": len([k for k in workflow if isinstance(workflow.get(k), dict)]),
    })


@tool
def remove_workflow_node(workflow_path: str, node_id: str) -> str:
    """Remove a node from an existing workflow and clean up any links pointing to it.

    Args:
        workflow_path: File path to the workflow JSON.
        node_id: The node ID to remove.
    """
    try:
        workflow = _load_workflow(workflow_path)
    except (json.JSONDecodeError, FileNotFoundError, OSError) as e:
        return json.dumps({"error": f"Cannot load workflow: {e}"})

    if node_id not in workflow:
        return json.dumps({"error": f"Node '{node_id}' not found."})

    del workflow[node_id]

    # Remove dangling links to the deleted node
    cleaned = 0
    for nid, node in workflow.items():
        if not isinstance(node, dict):
            continue
        inputs = node.get("inputs", {})
        for inp_name, inp_val in list(inputs.items()):
            if isinstance(inp_val, list) and len(inp_val) == 2 and str(inp_val[0]) == node_id:
                del inputs[inp_name]
                cleaned += 1

    path = _save_workflow(workflow, name=Path(workflow_path).stem)

    return json.dumps({
        "workflow_path": path,
        "removed_node": node_id,
        "cleaned_links": cleaned,
        "node_count": len([k for k in workflow if isinstance(workflow.get(k), dict)]),
    })


# ── 5. Parse workflow connections (graph analysis) ────────────────────────────

@tool
def parse_workflow_connections(workflow_path: str) -> str:
    """Parse a ComfyUI workflow and return its graph structure: roots, leaves, connections, and execution order.

    Args:
        workflow_path: File path to the workflow JSON (from get_workflow_template or save_workflow).
    """
    try:
        workflow = _load_workflow(workflow_path)
    except (json.JSONDecodeError, FileNotFoundError, OSError) as e:
        return json.dumps({"error": f"Cannot load workflow: {e}"})

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
