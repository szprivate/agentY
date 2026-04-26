"""
ComfyUI tools – server communication, workflow management, and node inspection.

Consolidates all ComfyUI-related @tool functions into a single module:
  • Server: models, execution control, queue, history, prompt submission
  • Workflows: template loading, patching, validation
  • Nodes: schema inspection, keyword search
"""

import json
import os
import uuid
from pathlib import Path

from strands import tool

from src.utils.comfyui_client import get_client


# ═══════════════════════════════════════════════════════════════════════════════
# Module-level state
# ═══════════════════════════════════════════════════════════════════════════════

# Workflow files are saved to disk and referenced by path to avoid bloating
# the LLM's sliding-window context with full JSON.
# patch_workflow failure guard
_PATCH_FAIL_LIMIT: int = 3
_patch_fail_count: int = 0
_patch_last_workflow_path: str | None = None

# /object_info cache – the full node database doesn't change during a session.
_object_info_cache: dict | None = None

# In-memory caches for template data (reset on process restart).
_index_cache: list | None = None
_template_cache: dict[str, dict] = {}


# ═══════════════════════════════════════════════════════════════════════════════
# Public helpers (non-tool, used by other modules)
# ═══════════════════════════════════════════════════════════════════════════════

def reset_patch_workflow_guard() -> None:
    """Reset the patch_workflow failure counter.  Call once per brain session."""
    global _patch_fail_count, _patch_last_workflow_path
    _patch_fail_count = 0
    _patch_last_workflow_path = None


# ═══════════════════════════════════════════════════════════════════════════════
# Internal helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _get_object_info() -> dict:
    """Return the full /object_info dict, cached after first fetch."""
    global _object_info_cache
    if _object_info_cache is None:
        _object_info_cache = get_client().get("/object_info")
    return _object_info_cache


def _project_root() -> Path:
    """Return the project root directory."""
    return Path(__file__).parent.parent.parent.resolve()


def _load_config() -> dict:
    """Load the settings.json configuration."""
    config_path = _project_root() / "config" / "settings.json"
    if config_path.exists():
        with open(config_path, encoding="utf-8") as f:
            return json.loads("".join(ln for ln in f if not ln.lstrip().startswith("//")))
    return {}


# NOTE: `_user_workflows_dir` intentionally removed. Use `_custom_templates_dir()`
# instead — user ad-hoc workflows are stored with the custom templates.


def _custom_templates_dir() -> Path:
    """Return the path to the custom workflow templates directory (has its own index.json)."""
    cfg = _load_config()
    ct_dir = cfg.get(
        "comfyui_custom_templates_dir",
        "./comfyui_workflows_templates_custom/",
    )
    return (_project_root() / ct_dir).resolve()


def _workflows_dir() -> Path:
    """Return the directory where generated/patched workflow JSON files are saved."""
    cfg = _load_config()
    wd = cfg.get("output_workflows_dir", "./output_workflows/")
    return (_project_root() / wd).resolve()


def _load_index() -> list:
    """Return the templates index as a flat list from the custom templates directory.

    Result is cached for the lifetime of the process.
    """
    global _index_cache
    if _index_cache is not None:
        return _index_cache

    flat: list[dict] = []
    index_path = _custom_templates_dir() / "index.json"
    if index_path.exists():
        try:
            with open(index_path, encoding="utf-8") as f:
                raw = json.load(f)
            for group in (raw or []):
                group_category = group.get("title", group.get("category", ""))
                group_media = group.get("type", "")
                for tpl in group.get("templates", []):
                    tpl["_group_category"] = group_category
                    tpl["_group_media"] = group_media
                    flat.append(tpl)
        except Exception:
            pass

    _index_cache = flat
    return flat


def _fetch_template(name: str) -> dict | None:
    """Load a single template JSON by name from the custom templates directory.

    Returns the parsed dict, or None if not found.
    Results are cached.
    """
    if name in _template_cache:
        return _template_cache[name]

    data: dict | None = None
    ct_dir = _custom_templates_dir()
    for candidate in [ct_dir / f"{name}.json", ct_dir / name]:
        if candidate.exists():
            with open(candidate, encoding="utf-8") as f:
                data = json.load(f)
            break

    if data is not None:
        _template_cache[name] = data
    return data


def _save_workflow(workflow: dict, name: str = "") -> str:
    """Save *workflow* dict to a JSON file and return the absolute path."""
    wd = _workflows_dir()
    wd.mkdir(parents=True, exist_ok=True)
    stem = name or uuid.uuid4().hex[:8]
    path = wd / f"{stem}.json"
    path.write_text(json.dumps(workflow, indent=2), encoding="utf-8")
    return str(path.resolve())


def _load_workflow(path_or_json: str) -> dict:
    """Load a workflow from a file path or raw JSON string.

    Auto-converts graph-format workflows to API format.
    """
    p = Path(path_or_json)
    if p.exists() and p.suffix == ".json":
        data = json.loads(p.read_text(encoding="utf-8"))
    else:
        data = json.loads(path_or_json)
    if _is_graph_format(data):
        data = _convert_graph_to_api(data)
    return data


def _is_graph_format(workflow: dict) -> bool:
    """Return True if *workflow* is in ComfyUI graph/export format."""
    return isinstance(workflow.get("nodes"), list)


# Input types that ComfyUI always wires via links (never appear as widget values)
_LINK_ONLY_TYPES: frozenset[str] = frozenset({
    "MODEL", "CLIP", "VAE", "LATENT", "IMAGE", "MASK", "CONDITIONING",
    "CONTROL_NET", "EMBEDS", "SAMPLER", "SIGMAS", "AUDIO", "VIDEO",
    "SEGS", "BBOX", "UPSCALE_MODEL", "CLIPREGION", "PHOTOMAKER",
    "GEMINI_INPUT_FILES",
})

_SEED_CONTROL_VALUES: frozenset[str] = frozenset({"fixed", "randomize", "increment", "decrement"})
_SEED_INPUT_NAMES: frozenset[str] = frozenset({"seed", "noise_seed"})


def _schema_widget_names(schema: dict, linked_names: set[str]) -> list[str]:
    """Return the ordered list of widget input names for a node schema.

    Mirrors the logic ComfyUI's frontend uses to assign widget_values entries
    to named inputs.
    """
    names: list[str] = []
    for section in ("required", "optional"):
        for inp_name, inp_spec in schema.get(section, {}).items():
            if inp_name in linked_names:
                continue
            inp_type = inp_spec[0] if (isinstance(inp_spec, (list, tuple)) and inp_spec) else ""
            if isinstance(inp_type, str) and inp_type in _LINK_ONLY_TYPES:
                continue
            names.append(inp_name)
    return names


def _convert_graph_to_api(workflow: dict) -> dict:
    """Convert a ComfyUI graph-format workflow dict to API format."""
    # Build link lookup: link_id → [str(src_node_id), src_slot]
    link_table: dict[int, list] = {}
    for link in workflow.get("links", []):
        if isinstance(link, (list, tuple)) and len(link) >= 3:
            link_id, src_node, src_slot = int(link[0]), link[1], link[2]
            link_table[link_id] = [str(src_node), int(src_slot)]

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

        # Map linked inputs
        linked_names: set[str] = set()
        for connector in node.get("inputs", []):
            if not isinstance(connector, dict):
                continue
            name = connector.get("name", "")
            link_id = connector.get("link")
            if name and link_id is not None and link_id in link_table:
                api_inputs[name] = link_table[link_id]
                linked_names.add(name)

        # Map widget values → named inputs
        widgets_values: list = node.get("widgets_values", node.get("widget_values", []))
        if isinstance(widgets_values, list) and widgets_values:
            schema = object_info.get(class_type, {}).get("input", {}) if object_info else {}
            if schema:
                widget_names = _schema_widget_names(schema, linked_names)
                wv_idx = 0
                for name in widget_names:
                    if wv_idx >= len(widgets_values):
                        break
                    val = widgets_values[wv_idx]
                    api_inputs[name] = val
                    wv_idx += 1
                    if (name in _SEED_INPUT_NAMES
                            and wv_idx < len(widgets_values)
                            and widgets_values[wv_idx] in _SEED_CONTROL_VALUES):
                        wv_idx += 1
                for extra_i, extra_val in enumerate(widgets_values[wv_idx:], start=wv_idx):
                    api_inputs[f"__extra_widget_{extra_i}"] = extra_val
            else:
                api_inputs["__widgets_values"] = list(widgets_values)

        api_node: dict = {"class_type": class_type, "inputs": api_inputs}
        title = node.get("title", "")
        if title:
            api_node["_meta"] = {"title": title}
        api_workflow[nid] = api_node

    return api_workflow


def _strip_history(data: dict | list) -> dict | list:
    """Strip embedded workflow/prompt JSON from history entries to save tokens."""
    if isinstance(data, list):
        return [_strip_history(item) for item in data]
    if not isinstance(data, dict):
        return data

    stripped: dict = {}
    for prompt_id, entry in data.items():
        if not isinstance(entry, dict):
            stripped[prompt_id] = entry
            continue
        slim: dict = {}
        if "status" in entry:
            slim["status"] = entry["status"]
        if "outputs" in entry:
            outputs: dict = {}
            for node_id, node_out in entry.get("outputs", {}).items():
                if isinstance(node_out, dict):
                    slim_out: dict = {}
                    for key, val in node_out.items():
                        if isinstance(val, list):
                            slim_out[key] = [
                                {k: v for k, v in item.items() if k != "abs_path"}
                                if isinstance(item, dict) else item
                                for item in val
                            ]
                        else:
                            slim_out[key] = val
                    outputs[node_id] = slim_out
            slim["outputs"] = outputs
        stripped[prompt_id] = slim
    return stripped


def _parse_inputs_schema(spec: dict) -> dict:
    """Turn ComfyUI's input spec into a friendlier format."""
    result = {}
    for name, definition in spec.items():
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
        result[name] = entry
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# Tools: Models
# ═══════════════════════════════════════════════════════════════════════════════

@tool
def get_model_types() -> str:
    """List available model folder types in ComfyUI (checkpoints, loras, unet, vae, clip, etc.)."""
    try:
        return json.dumps(get_client().get("/models"))
    except Exception as e:
        return json.dumps({"error": str(e)})


@tool
def get_models_in_folder(folder: str) -> str:
    """List model files in a ComfyUI model folder.

    Args:
        folder: Folder name e.g. 'checkpoints', 'loras', 'vae', 'clip', 'unet'.
    """
    try:
        return json.dumps(get_client().get(f"/models/{folder}"))
    except Exception as e:
        return json.dumps({"error": str(e)})


# ═══════════════════════════════════════════════════════════════════════════════
# Tools: Execution control
# ═══════════════════════════════════════════════════════════════════════════════

@tool
def interrupt_execution() -> str:
    """Immediately stop the currently running ComfyUI workflow execution."""
    try:
        return json.dumps(get_client().post("/interrupt", json_data={}))
    except Exception as e:
        return json.dumps({"error": str(e)})


@tool
def free_memory(unload_models: bool = True, free_memory_flag: bool = True) -> str:
    """Free GPU/system memory in ComfyUI by unloading models and clearing caches.

    Args:
        unload_models: Unload all loaded models from VRAM (default True).
        free_memory_flag: Free cached memory (default True).
    """
    try:
        payload = {
            "unload_models": unload_models,
            "free_memory": free_memory_flag,
        }
        return json.dumps(get_client().post("/free", json_data=payload))
    except Exception as e:
        return json.dumps({"error": str(e)})


# ═══════════════════════════════════════════════════════════════════════════════
# Tools: Queue
# ═══════════════════════════════════════════════════════════════════════════════

@tool
def queue(action: str = "status") -> str:
    """Get or manage the ComfyUI execution queue.

    Args:
        action: 'status' (view queue), 'clear' (clear pending), or 'clear_running' (stop running items).
    """
    try:
        if action == "status":
            return json.dumps(get_client().get("/queue"))
        elif action in ("clear", "clear_running"):
            payload = {"clear": True} if action == "clear" else {"clear_running": True}
            return json.dumps(get_client().post("/queue", json_data=payload))
        else:
            return json.dumps({"error": f"Unknown action '{action}'. Use 'status', 'clear', or 'clear_running'"})
    except Exception as e:
        return json.dumps({"error": str(e)})


# ═══════════════════════════════════════════════════════════════════════════════
# Tools: History
# ═══════════════════════════════════════════════════════════════════════════════

@tool
def get_history(max_items: int = 3) -> str:
    """Get recent ComfyUI execution history (status and output filenames only).

    Args:
        max_items: Max entries to return (default 3; 0 = all).
    """
    try:
        params = {}
        if max_items > 0:
            params["max_items"] = max_items
        raw = get_client().get("/history", params=params or None)
        return json.dumps(_strip_history(raw))
    except Exception as e:
        return json.dumps({"error": str(e)})


@tool
def get_prompt_status_by_id(prompt_id: str) -> str:
    """Check execution status for a specific prompt ID. Returns status and output filenames only.

    Args:
        prompt_id: Prompt ID returned by submit_prompt.
    """
    try:
        raw = get_client().get(f"/history/{prompt_id}")
        return json.dumps(_strip_history(raw))
    except Exception as e:
        return json.dumps({"error": str(e)})


@tool
def clear_history(prompt_id: str = "") -> str:
    """Clear ComfyUI execution history. If prompt_id given, deletes that entry only.

    Args:
        prompt_id: Optional specific prompt ID to delete. If empty, clears all history.
    """
    try:
        if prompt_id:
            payload = {"delete": [prompt_id]}
        else:
            payload = {"clear": True}
        return json.dumps(get_client().post("/history", json_data=payload))
    except Exception as e:
        return json.dumps({"error": str(e)})


# ═══════════════════════════════════════════════════════════════════════════════
# Tools: Diagnostics
# ═══════════════════════════════════════════════════════════════════════════════

@tool
def get_system_stats() -> str:
    """Return ComfyUI system info: GPU, VRAM, Python version, PyTorch version, OS."""
    try:
        return json.dumps(get_client().get("/system_stats"))
    except Exception as e:
        return json.dumps({"error": str(e)})


@tool
def get_comfyui_dirs() -> str:
    """Return the authoritative ComfyUI server directory paths.

    Queries ``/system_stats`` and extracts the ``--input-directory``,
    ``--output-directory``, and ``--user-directory`` flags from the server's
    argv.  Falls back to ComfyUI's compiled-in defaults when a flag is absent.

    Use this tool to resolve:
    - Where uploaded images land after ``upload_image()`` (input_dir).
    - Where ComfyUI will save generated outputs (output_dir) — use this path
      when populating ``output_nodes[].output_path`` in the brainbriefing.
    - Where workflow JSON files are stored (user_dir/workflows/).

    Returns a JSON object with keys ``input_dir``, ``output_dir``, ``user_dir``,
    and ``source`` ("argv" when resolved from server flags, "default" otherwise).
    """
    try:
        stats = get_client().get("/system_stats")
        if not isinstance(stats, dict):
            return json.dumps({"error": "Unexpected /system_stats response format"})

        argv: list = stats.get("system", {}).get("argv", [])
        result: dict[str, str] = {"source": "argv"}

        for arg in argv:
            if not isinstance(arg, str):
                continue
            if arg.startswith("--input-directory="):
                result["input_dir"] = arg.split("=", 1)[1]
            elif arg.startswith("--output-directory="):
                result["output_dir"] = arg.split("=", 1)[1]
            elif arg.startswith("--user-directory="):
                result["user_dir"] = arg.split("=", 1)[1]

        # Fill in missing dirs with ComfyUI's conventional defaults
        # (relative to where the server was launched from, typically the ComfyUI root).
        if "input_dir" not in result or "output_dir" not in result or "user_dir" not in result:
            result["source"] = "partial_argv_with_defaults"
            # Try to infer the ComfyUI root from the argv[0] path
            comfy_root: str | None = None
            if argv:
                import os as _os
                comfy_root = str(Path(argv[0]).parent.resolve()) if argv[0] else None

            if "input_dir" not in result:
                result["input_dir"] = str(Path(comfy_root) / "input") if comfy_root else "unknown"
            if "output_dir" not in result:
                result["output_dir"] = str(Path(comfy_root) / "output") if comfy_root else "unknown"
            if "user_dir" not in result:
                result["user_dir"] = str(Path(comfy_root) / "user") if comfy_root else "unknown"

        return json.dumps(result)
    except Exception as exc:
        return json.dumps({"error": str(exc)})


@tool
def get_logs(keyword: str = "", max_lines: int = 100) -> str:
    """Get ComfyUI server runtime logs. Useful for debugging errors, missing nodes, and tracebacks.

    Args:
        keyword: Filter log lines containing this string (case-insensitive). E.g. 'error', 'warning'.
        max_lines: Max lines to return from the end (default 100).
    """
    try:
        raw = get_client().get("/logs")
        # Endpoint may return list of strings or a dict with a 'logs' key
        if isinstance(raw, dict):
            lines: list = raw.get("logs", raw.get("lines", []))
        elif isinstance(raw, list):
            lines = raw
        else:
            lines = str(raw).splitlines()

        if keyword:
            kw = keyword.lower()
            lines = [l for l in lines if kw in str(l).lower()]

        if len(lines) > max_lines:
            lines = lines[-max_lines:]

        # Strip ANSI codes for readability
        import re as _re
        clean = [_re.sub(r"\x1b\[[0-9;]*m", "", str(l)) for l in lines]
        return json.dumps({"lines": clean, "count": len(clean)})
    except Exception as e:
        return json.dumps({"error": str(e)})


# ═══════════════════════════════════════════════════════════════════════════════
# Tools: Prompt submission
# ═══════════════════════════════════════════════════════════════════════════════

@tool
def submit_prompt(workflow_path: str, client_id: str = "") -> str:
    """Submit a workflow to the ComfyUI execution queue. Returns prompt_id on success.

    Args:
        workflow_path: File path to the workflow JSON (from get_workflow_template or save_workflow).
        client_id: Optional client identifier for tracking.
    """
    try:
        p = Path(workflow_path)
        if p.exists() and p.suffix == ".json":
            workflow = json.loads(p.read_text(encoding="utf-8"))
        else:
            # Legacy fallback: accept inline JSON string
            workflow = json.loads(workflow_path)

        client = get_client()
        payload: dict = {"prompt": workflow}
        if client_id:
            payload["client_id"] = client_id
        # Forward the ComfyUI API key so API/partner nodes receive it.
        if client.api_key:
            payload["extra_data"] = {"api_key_comfy_org": client.api_key}
        return json.dumps(client.post("/prompt", json_data=payload))
    except json.JSONDecodeError as e:
        return json.dumps({"error": f"Invalid JSON in workflow: {e}"})
    except Exception as e:
        return json.dumps({"error": str(e)})


@tool
def signal_workflow_ready(workflow_path: str) -> str:
    """Signal that the workflow is fully assembled and validated, ready for execution.

    Call this as your **final step** once ``update_workflow()`` returns ``status: "ok"``.
    The pipeline will automatically handle ComfyUI submission, completion
    polling, Vision QA (via Ollama), and saving outputs to ``./output``.

    For **batch runs** (``count_iter > 1``): call this tool once for every
    workflow file produced by ``duplicate_workflow()``.  Each call appends to
    the execution queue; the pipeline submits them to ComfyUI in order.

    Do NOT call ``submit_prompt`` — this tool replaces it.

    Args:
        workflow_path: File path to the validated workflow JSON
                       (the same path returned by ``get_workflow_template`` or
                       ``save_workflow`` and used in ``update_workflow``).
    """
    from src.utils.workflow_signal import append_workflow_path

    p = Path(workflow_path)
    if not p.exists():
        return json.dumps({"error": f"Workflow file not found: {workflow_path}"})

    resolved = str(p.resolve())
    append_workflow_path(resolved)
    return json.dumps({
        "status": "ready",
        "workflow_path": resolved,
        "message": (
            "Workflow has been added to the execution queue. "
            "The pipeline will submit it to ComfyUI, run Vision QA, "
            "and save outputs to ./output automatically. "
            "For batch runs, call signal_workflow_ready for each duplicate workflow; "
            "otherwise your work here is done — no further tool calls are needed."
        ),
    })


@tool
def duplicate_workflow(source_path: str) -> str:
    """Create a copy of an existing workflow JSON with a fresh random seed.

    Use this during **batch runs** (``count_iter > 1``) to produce iteration
    copies from the base validated workflow.  Each duplicate gets a new random
    seed injected into every KSampler / RandomNoise / similar seed field so
    outputs are genuinely varied.

    Typical batch flow:
    1. Assemble + validate the base workflow normally.
    2. Call ``signal_workflow_ready(base_path)`` for iteration 1.
    3. For iterations 2..N: call ``duplicate_workflow(base_path)`` → validate
       the returned path → call ``signal_workflow_ready(new_path)``.

    Args:
        source_path: Path to the already-validated base workflow JSON.

    Returns:
        JSON with ``new_path`` (the path of the newly created workflow copy)
        or ``error`` on failure.
    """
    import random
    import shutil

    p = Path(source_path)
    if not p.exists():
        return json.dumps({"error": f"Source workflow not found: {source_path}"})

    # Build a unique destination path: <stem>_iter_<N>.json in the same directory.
    parent = p.parent
    stem = p.stem
    # Strip any existing _iter_<N> suffix so duplicating a duplicate stays tidy.
    import re as _re
    stem = _re.sub(r"_iter_\d+$", "", stem)
    idx = 1
    while True:
        candidate = parent / f"{stem}_iter_{idx:03d}.json"
        if not candidate.exists():
            break
        idx += 1

    shutil.copy2(p, candidate)

    # Randomise seed fields in the copy so each iteration produces a unique result.
    try:
        data = json.loads(candidate.read_text(encoding="utf-8"))
        _SEED_KEYS = {"seed", "noise_seed"}
        changed = 0
        for node in data.values():
            if not isinstance(node, dict):
                continue
            inputs = node.get("inputs", {})
            for key in _SEED_KEYS:
                if key in inputs and isinstance(inputs[key], (int, float)):
                    inputs[key] = random.randint(0, 2**32 - 1)
                    changed += 1
        candidate.write_text(json.dumps(data, indent=2), encoding="utf-8")
    except Exception as exc:
        # If seed patching fails, the copy is still usable (same seed → same image,
        # but the pipeline will still queue it as requested).
        return json.dumps({
            "new_path": str(candidate),
            "warning": f"Seed randomisation failed — copy uses original seed: {exc}",
        })

    return json.dumps({
        "new_path": str(candidate),
        "seeds_randomised": changed,
        "message": f"Workflow duplicated to {candidate.name} with {changed} seed(s) randomised.",
    })


# ═══════════════════════════════════════════════════════════════════════════════
# Tools: Node inspection
# ═══════════════════════════════════════════════════════════════════════════════

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
        input_spec = info.get("input", {})
        schema = {
            "node_class": node_class,
            "display_name": info.get("display_name", node_class),
            "description": info.get("description", ""),
            "category": info.get("category", ""),
            "input_required": _parse_inputs_schema(input_spec.get("required", {})),
            "input_optional": _parse_inputs_schema(input_spec.get("optional", {})),
            "output_types": info.get("output", []),
            "output_names": info.get("output_name", []),
            "output_is_list": info.get("output_is_list", []),
            "is_output_node": info.get("output_node", False),
        }
        return json.dumps(schema)
    except Exception as e:
        return json.dumps({"error": str(e)})


@tool
def get_workflow_node_info(node_id: str, workflow_path: str) -> str:
    """Return full metadata for a single node inside a saved workflow.

    Combines the node's current state (class_type, title, literal inputs,
    connected inputs, widget values) with the ComfyUI schema for its class.

    Args:
        node_id: The node's key inside the workflow JSON, e.g. "6" or "190".
        workflow_path: File path to the workflow JSON.
    """
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

    inputs_raw = raw_node.get("inputs", {})
    literal_inputs: dict = {}
    connected_inputs: dict = {}
    for name, val in inputs_raw.items():
        if isinstance(val, list) and len(val) == 2 and isinstance(val[1], int):
            connected_inputs[name] = {"from_node": str(val[0]), "from_slot": val[1]}
        else:
            literal_inputs[name] = val

    schema: dict = {}
    if cls:
        try:
            all_nodes = _get_object_info()
            if cls in all_nodes:
                info = all_nodes[cls]
                input_spec = info.get("input", {})
                schema = {
                    "display_name": info.get("display_name", cls),
                    "description": info.get("description", ""),
                    "category": info.get("category", ""),
                    "input_required": _parse_inputs_schema(input_spec.get("required", {})),
                    "input_optional": _parse_inputs_schema(input_spec.get("optional", {})),
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

            input_types = set()
            for section in ("required", "optional"):
                for _name, defn in input_spec.get(section, {}).items():
                    if isinstance(defn, list) and defn:
                        t = defn[0]
                        if isinstance(t, str):
                            input_types.add(t)

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


# ═══════════════════════════════════════════════════════════════════════════════
# Tools: Workflow templates
# ═══════════════════════════════════════════════════════════════════════════════

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
def get_workflow_template(template_name: str) -> str:
    """Load a workflow template by name. Saves the full workflow to a file and returns a compact summary with the file path.

    The returned summary includes: node list (id, class, title, key literal inputs),
    model info, and io metadata. The full workflow JSON is at the returned
    ``workflow_path`` — pass that path to validate_workflow / submit_prompt.

    Args:
        template_name: Template name (without .json) from get_workflow_catalog().
    """
    try:
        lookup = template_name.removesuffix(".json")
        workflow = None
        source = ""
        metadata: dict = {}

        # Try user ad-hoc workflows first (stored alongside custom templates)
        tdir = _custom_templates_dir()
        for candidate in [tdir / f"{lookup}.json", tdir / template_name]:
            if candidate.exists():
                with open(candidate, encoding="utf-8") as f:
                    workflow = json.load(f)
                source = "user"
                break

        # Try custom + official templates (custom dir, then GitHub, then local fallback)
        if workflow is None:
            data = _fetch_template(lookup)
            if data is None and lookup != template_name:
                data = _fetch_template(template_name)
            if data is not None:
                workflow = data
                source = "templates"
                for tpl in _load_index():
                    if tpl.get("name") == lookup:
                        metadata = {
                            "models": tpl.get("models", []),
                            "io": tpl.get("io", {}),
                        }
                        break

        if workflow is None:
            return json.dumps({
                "error": f"Template '{template_name}' not found.",
                "hint": "Use get_workflow_catalog() to see available templates.",
            })

        # Normalise to API format
        converted = False
        if _is_graph_format(workflow):
            workflow = _convert_graph_to_api(workflow)
            converted = True

        workflow_path = _save_workflow(workflow, name=lookup)

        # Build compact node summary
        node_summary = []
        for nid, node in workflow.items():
            if not isinstance(node, dict):
                continue
            cls = node.get("class_type", "unknown")
            title = node.get("_meta", {}).get("title", cls)
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


# ═══════════════════════════════════════════════════════════════════════════════
# Tools: Workflow modification
# ═══════════════════════════════════════════════════════════════════════════════

@tool
def save_workflow(workflow_json: str, name: str = "") -> str:
    """Save a complete workflow JSON to a file and return the file path.

    Only use for building entirely new workflows from scratch.
    For editing existing workflows, use patch_workflow() instead.

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
            - widget_values_index (int): If set, patch widget_values[index] instead of inputs.
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

    # Failure guard
    global _patch_fail_count, _patch_last_workflow_path
    _patch_last_workflow_path = path

    if errors:
        _patch_fail_count += 1
        print(
            f"[patch_workflow] Failure {_patch_fail_count}/{_PATCH_FAIL_LIMIT}: "
            f"{len(errors)} patch error(s)."
        )

        if _patch_fail_count >= _PATCH_FAIL_LIMIT:
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
        _patch_fail_count = 0

    return json.dumps({
        "workflow_path": path,
        "applied": applied,
        "errors": errors,
        "patch_count": len(applied),
    })


@tool
def update_workflow(
    workflow_path: str,
    patches: str = "[]",
    add_nodes: str = "[]",
    remove_nodes: str = "[]",
) -> str:
    """Add/remove nodes, patch inputs, and validate a workflow in one atomic step.

    Use this instead of calling add_workflow_node / remove_workflow_node,
    patch_workflow, and validate_workflow separately.  Returns status "ok" when
    the workflow is valid and ready to hand off, or "error" with a full list of
    problems to fix.

    Execution order: remove nodes → add nodes → apply patches → validate.

    Args:
        workflow_path: File path to the workflow JSON (from get_workflow_template).
        patches: JSON array of patch objects (same format as patch_workflow).
            Each: {"node_id": "6", "input_name": "text", "value": "..."}
            Optional per-patch fields: widget_values_index, class_type.
            Pass "[]" (default) to skip patching.
        add_nodes: JSON array of node-add specs.
            Each: {"node_id": "200", "class_type": "LoadImage",
                   "inputs": {"image": "file.png"}, "meta_title": "My Node"}
            Pass "[]" (default) to add nothing.
        remove_nodes: JSON array of node ID strings to remove.
            Example: ["42", "55"]
            Pass "[]" (default) to remove nothing.
    """
    # ── Load ──────────────────────────────────────────────────────────────────
    try:
        workflow = _load_workflow(workflow_path)
    except (json.JSONDecodeError, FileNotFoundError, OSError) as e:
        return json.dumps({"status": "error", "error": f"Cannot load workflow: {e}"})

    try:
        remove_list: list = json.loads(remove_nodes) if isinstance(remove_nodes, str) else remove_nodes
        add_list: list = json.loads(add_nodes) if isinstance(add_nodes, str) else add_nodes
        patch_list: list = json.loads(patches) if isinstance(patches, str) else patches
    except json.JSONDecodeError as e:
        return json.dumps({"status": "error", "error": f"Invalid JSON argument: {e}"})

    removed: list[str] = []
    added: list[str] = []
    applied: list[str] = []
    node_errors: list[str] = []
    cleaned_links: int = 0

    # ── Remove nodes ──────────────────────────────────────────────────────────
    for nid in (str(x) for x in remove_list):
        if nid not in workflow:
            node_errors.append(f"remove: node '{nid}' not found.")
            continue
        del workflow[nid]
        removed.append(nid)

    # Clean dangling links pointing to removed nodes
    removed_set = set(removed)
    for _nid, node in workflow.items():
        if not isinstance(node, dict):
            continue
        inputs_dict = node.get("inputs", {})
        for inp_name, inp_val in list(inputs_dict.items()):
            if isinstance(inp_val, list) and len(inp_val) == 2 and str(inp_val[0]) in removed_set:
                del inputs_dict[inp_name]
                cleaned_links += 1

    # ── Add nodes ─────────────────────────────────────────────────────────────
    for spec in add_list:
        nid = str(spec.get("node_id", ""))
        cls = spec.get("class_type", "")
        if not nid or not cls:
            node_errors.append(f"add: spec missing node_id or class_type — {spec}")
            continue
        if nid in workflow:
            node_errors.append(f"add: node '{nid}' already exists.")
            continue
        raw_inputs = spec.get("inputs", {})
        node_inputs = json.loads(raw_inputs) if isinstance(raw_inputs, str) else raw_inputs
        new_node: dict = {"class_type": cls, "inputs": node_inputs}
        if spec.get("meta_title"):
            new_node["_meta"] = {"title": spec["meta_title"]}
        workflow[nid] = new_node
        added.append(nid)

    # ── Patch ─────────────────────────────────────────────────────────────────
    if not isinstance(patch_list, list):
        return json.dumps({"status": "error", "error": "patches must be a JSON array of patch objects."})

    for i, patch in enumerate(patch_list):
        nid = str(patch.get("node_id", ""))
        if nid not in workflow:
            node_errors.append(f"patch {i}: node '{nid}' not found in workflow.")
            continue

        node = workflow[nid]

        if "class_type" in patch:
            node["class_type"] = patch["class_type"]
            applied.append(f"Node {nid}: class_type → {patch['class_type']}")

        if "widget_values_index" in patch:
            idx = int(patch["widget_values_index"])
            wv = node.get("widget_values", node.get("widgets_values"))
            wv_key = "widget_values" if "widget_values" in node else "widgets_values"
            if isinstance(wv, list) and 0 <= idx < len(wv):
                wv[idx] = patch["value"]
                node[wv_key] = wv
                applied.append(f"Node {nid}: {wv_key}[{idx}] → {patch['value']!r}")
            else:
                node_errors.append(f"patch {i}: node '{nid}' has no {wv_key}[{idx}].")
            continue

        inp_name = patch.get("input_name")
        if inp_name:
            if "inputs" not in node:
                node["inputs"] = {}
            node["inputs"][inp_name] = patch["value"]
            val_repr = repr(patch["value"])
            if len(val_repr) > 80:
                val_repr = val_repr[:77] + "..."
            applied.append(f"Node {nid}.inputs.{inp_name} → {val_repr}")

    # ── Save ──────────────────────────────────────────────────────────────────
    path = _save_workflow(workflow, name=Path(workflow_path).stem)

    # Update patch failure guard
    global _patch_fail_count, _patch_last_workflow_path
    _patch_last_workflow_path = path

    if node_errors:
        _patch_fail_count += 1
        print(
            f"[update_workflow] Failure {_patch_fail_count}/{_PATCH_FAIL_LIMIT}: "
            f"{len(node_errors)} error(s)."
        )
        if _patch_fail_count >= _PATCH_FAIL_LIMIT:
            debug_name = f"{Path(workflow_path).stem}_update_debug"
            debug_path = _save_workflow(workflow, name=debug_name)
            print(f"[update_workflow] LIMIT REACHED — debug snapshot: {debug_path}")
            return json.dumps({
                "status": "error",
                "workflow_path": path,
                "removed_nodes": removed,
                "added_nodes": added,
                "applied_patches": applied,
                "node_errors": node_errors,
                "valid": False,
                "local_errors": [],
                "server_errors": {},
                "patch_failure_limit_reached": True,
                "debug_workflow_path": debug_path,
                "message": (
                    f"update_workflow has failed {_PATCH_FAIL_LIMIT} times. "
                    f"Debug snapshot saved to: {debug_path}. "
                    "STOP — do not call update_workflow again. "
                    "Report the debug_workflow_path to the user and ask for guidance."
                ),
            })
    else:
        _patch_fail_count = 0

    # ── Validate ──────────────────────────────────────────────────────────────
    local_errors: list[str] = []
    server_errors: dict = {}

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
        if all_nodes and cls not in all_nodes:
            local_errors.append(f"Node {nid}: unknown class_type '{cls}'.")
            continue
        node_info = all_nodes.get(cls, {})
        required = node_info.get("input", {}).get("required", {})
        node_inputs = node.get("inputs", {})
        for req_name in required:
            if req_name not in node_inputs:
                local_errors.append(f"Node {nid} ({cls}): missing required input '{req_name}'.")
        for inp_name, inp_val in node_inputs.items():
            if isinstance(inp_val, list) and len(inp_val) == 2:
                src_id = str(inp_val[0])
                if src_id not in node_ids:
                    local_errors.append(
                        f"Node {nid} ({cls}): input '{inp_name}' references "
                        f"non-existent node '{src_id}'."
                    )

    try:
        result = get_client().post("/prompt", json_data={"prompt": workflow})
        if isinstance(result, dict):
            if "error" in result:
                server_errors = {
                    "error": result.get("error"),
                    "node_errors": result.get("node_errors", {}),
                }
            elif "prompt_id" in result:
                try:
                    get_client().post("/interrupt", json_data={})
                    get_client().post("/queue", json_data={"clear": True})
                except Exception:
                    pass
    except Exception as e:
        err_str = str(e)
        if hasattr(e, "response"):
            try:
                server_errors = e.response.json()
            except Exception:
                server_errors = {"error": err_str}
        else:
            server_errors = {"error": err_str}

    is_valid = len(local_errors) == 0 and len(server_errors) == 0
    all_errors = node_errors + local_errors

    return json.dumps({
        "status": "ok" if (is_valid and not node_errors) else "error",
        "workflow_path": path,
        "removed_nodes": removed,
        "cleaned_links": cleaned_links,
        "added_nodes": added,
        "applied_patches": applied,
        "node_errors": node_errors,
        "valid": is_valid,
        "local_errors": local_errors,
        "server_errors": server_errors,
    })


@tool
def replace_node(workflow_path: str, old_node_id: str, new_class_type: str, new_node_id: str = "", meta_title: str = "") -> str:
    """Replace a node with a different class_type while preserving all connections.

    Copies every input entry (including link arrays) from the old node to the new
    node, rewrites every other node's inputs that referenced the old node ID to
    point at the new node ID instead, then removes the old node.

    Args:
        workflow_path: File path to the workflow JSON.
        old_node_id: ID of the node to remove.
        new_class_type: class_type for the replacement node.
        new_node_id: ID for the new node.  Defaults to old_node_id (in-place swap).
        meta_title: Optional display title for the new node.
    """
    try:
        workflow = _load_workflow(workflow_path)
    except (json.JSONDecodeError, FileNotFoundError, OSError) as e:
        return json.dumps({"status": "error", "error": f"Cannot load workflow: {e}"})

    old_id = str(old_node_id)
    if old_id not in workflow:
        return json.dumps({"status": "error", "error": f"Node '{old_id}' not found in workflow."})

    new_id = str(new_node_id) if new_node_id else old_id

    if new_id != old_id and new_id in workflow:
        return json.dumps({"status": "error", "error": f"Node ID '{new_id}' already exists."})

    old_node = workflow[old_id]

    # Build the replacement node, inheriting all inputs from the old node.
    import copy
    new_node: dict = {
        "class_type": new_class_type,
        "inputs": copy.deepcopy(old_node.get("inputs", {})),
    }
    if meta_title:
        new_node["_meta"] = {"title": meta_title}
    elif "_meta" in old_node:
        new_node["_meta"] = copy.deepcopy(old_node["_meta"])

    # Insert the new node (if new_id == old_id this temporarily overwrites it).
    if new_id != old_id:
        del workflow[old_id]
    workflow[new_id] = new_node

    # Rewrite outgoing links: any node whose input is [old_id, slot] → [new_id, slot].
    rewritten: list[str] = []
    if new_id != old_id:
        for nid, node in workflow.items():
            if not isinstance(node, dict):
                continue
            for inp_name, inp_val in node.get("inputs", {}).items():
                if isinstance(inp_val, list) and len(inp_val) == 2 and str(inp_val[0]) == old_id:
                    inp_val[0] = new_id
                    rewritten.append(f"{nid}.inputs.{inp_name}")

    path = _save_workflow(workflow, name=Path(workflow_path).stem)
    return json.dumps({
        "status": "ok",
        "workflow_path": path,
        "old_node_id": old_id,
        "new_node_id": new_id,
        "new_class_type": new_class_type,
        "inherited_inputs": list(new_node["inputs"].keys()),
        "rewired_downstream": rewritten,
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


# ═══════════════════════════════════════════════════════════════════════════════
# Tools: Workflow validation
# ═══════════════════════════════════════════════════════════════════════════════

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

        if all_nodes and cls not in all_nodes:
            local_errors.append(f"Node {nid}: unknown class_type '{cls}'.")
            continue

        node_info = all_nodes.get(cls, {})
        required = node_info.get("input", {}).get("required", {})
        inputs = node.get("inputs", {})

        for req_name in required:
            if req_name not in inputs:
                local_errors.append(
                    f"Node {nid} ({cls}): missing required input '{req_name}'."
                )

        for inp_name, inp_val in inputs.items():
            if isinstance(inp_val, list) and len(inp_val) == 2:
                src_id = str(inp_val[0])
                if src_id not in node_ids:
                    local_errors.append(
                        f"Node {nid} ({cls}): input '{inp_name}' references "
                        f"non-existent node '{src_id}'."
                    )

    # Server-side validation
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
                # Accepted and queued – interrupt immediately to prevent execution.
                try:
                    get_client().post("/interrupt", json_data={})
                    get_client().post("/queue", json_data={"clear": True})
                except Exception:
                    pass
    except Exception as e:
        err_str = str(e)
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
