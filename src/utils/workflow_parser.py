"""
Utility for parsing raw ComfyUI workflow JSON files into the index.json template format.

Usage
-----
    from src.utils.workflow_parser import parse_workflow

    with open("comfyui_workflows_templates_custom/qwen2511_imageEdit.json") as f:
        raw = json.load(f)

    entry = parse_workflow(raw, name="qwen2511_imageEdit")
    # entry is a dict matching one element of the index.json list
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Node-class taxonomy
# ---------------------------------------------------------------------------

#: ComfyUI class_type values that represent *inputs* (things loaded from disk /
#: provided by the caller), grouped by the media type they carry.
INPUT_NODE_TYPES: dict[str, str] = {
    # images
    "LoadImage": "image",
    "Image Load": "image",
    "ETN_LoadImageBase64": "image",
    # video
    "LoadVideo": "video",
    "VHS_LoadVideo": "video",
    "LoadVideoPath": "video",
    # audio
    "LoadAudio": "audio",
    "VHS_LoadAudio": "audio",
    # 3-D
    "Load3D": "3d",
    "Load3DAnimation": "3d",
}

#: class_type values that represent *outputs* (results saved / returned to the
#: caller), grouped by the media type they produce.
OUTPUT_NODE_TYPES: dict[str, str] = {
    # images
    "SaveImage": "image",
    "PreviewImage": "image",
    "ETN_SendImageWebSocket": "image",
    # video
    "VHS_VideoCombine": "video",
    "SaveVideo": "video",
    # audio
    "SaveAudio": "audio",
    "VHS_SaveAudio": "audio",
    # 3-D
    "Save3D": "3d",
}

#: Node classes whose inputs carry useful model names.
MODEL_LOADER_TYPES: set[str] = {
    "UNETLoader",
    "CheckpointLoaderSimple",
    "CheckpointLoader",
    "CLIPLoader",
    "VAELoader",
    "LoraLoader",
    "LoraLoaderModelOnly",
    "DiffusersLoader",
    "unCLIPCheckpointLoader",
    "IPAdapterModelLoader",
    "ControlNetLoader",
}

#: Fields inside a loader node's inputs that hold the model file name.
MODEL_NAME_FIELDS: tuple[str, ...] = (
    "unet_name",
    "ckpt_name",
    "clip_name",
    "vae_name",
    "lora_name",
    "model_name",
    "diffusers_model_path",
    "control_net_name",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _stem(path_str: str) -> str:
    """Return the file stem of a path string, stripping any directory prefix."""
    return Path(path_str).stem


def _extract_models(workflow: dict[str, Any]) -> list[str]:
    """Return a deduplicated list of human-readable model names found in *workflow*."""
    models: list[str] = []
    for node in workflow.values():
        if not isinstance(node, dict):
            continue
        if node.get("class_type") not in MODEL_LOADER_TYPES:
            continue
        inputs = node.get("inputs", {})
        for field in MODEL_NAME_FIELDS:
            value = inputs.get(field)
            if isinstance(value, str) and value.strip():
                stem = _stem(value)
                if stem not in models:
                    models.append(stem)
    return models


def _project_root() -> Path:
    """Return the project root (three levels above this file: src/utils → src → root)."""
    return Path(__file__).parent.parent.parent.resolve()


def _load_config() -> dict:
    """Load config/settings.json from the project root."""
    config_path = _project_root() / "config" / "settings.json"
    if config_path.exists():
        with open(config_path, encoding="utf-8") as f:
            return json.load(f)
    return {}


def _custom_index_path() -> Path:
    """Return the resolved path to the custom-templates index.json from settings."""
    cfg = _load_config()
    ct_dir = cfg.get("comfyui_custom_templates_dir", "./comfyui_workflows_templates_custom/")
    return (_project_root() / ct_dir / "index.json").resolve()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def append_to_index(
    entry: dict[str, Any],
    index_path: Path | str | None = None,
) -> Path:
    """Append (or update) *entry* in the custom-workflows ``index.json``.

    If a template with the same ``name`` already exists anywhere in the
    index, its parent entry is updated in-place rather than duplicated.

    Parameters
    ----------
    entry:
        A parsed workflow entry as returned by :func:`parse_workflow`.
    index_path:
        Path to ``index.json``.  Defaults to
        ``comfyui_workflows_templates_custom/index.json`` relative to the
        repository root.

    Returns
    -------
    Path
        The resolved path that was written.
    """
    target = Path(index_path) if index_path is not None else _custom_index_path()
    target.parent.mkdir(parents=True, exist_ok=True)

    if target.exists():
        with open(target, encoding="utf-8") as f:
            raw = f.read().strip()
        index: list[dict[str, Any]] = json.loads(raw) if raw else []
    else:
        index = []

    new_names: set[str] = {
        t["name"] for t in entry.get("templates", []) if "name" in t
    }

    # Remove any existing entries whose templates overlap with the new names
    # so we avoid duplicates.
    index = [
        e
        for e in index
        if not any(
            t.get("name") in new_names for t in e.get("templates", [])
        )
    ]

    index.append(entry)

    with open(target, "w", encoding="utf-8") as f:
        json.dump(index, f, indent=2, ensure_ascii=False)
        f.write("\n")  # trailing newline

    return target


def parse_workflow(
    workflow: dict[str, Any],
    *,
    name: str = "",
    update_index: bool = True,
    index_path: Path | str | None = None,
) -> dict[str, Any]:
    """Parse a raw ComfyUI workflow dict and return an *index.json*-compatible entry.

    Only the keys used by the agent are emitted: ``name``, ``models``, and
    ``io`` (with ``inputs`` / ``outputs`` sub-lists).

    Parameters
    ----------
    workflow:
        The parsed JSON content of a ComfyUI workflow file (``dict[node_id, node]``).
    name:
        The template slug / file name (without extension).  Falls back to
        ``"workflow"`` when omitted.
    update_index:
        When ``True`` (default), the entry is appended to / updated in
        ``comfyui_workflows_templates_custom/index.json``.
    index_path:
        Override the path to ``index.json``.  Only used when
        *update_index* is ``True``.

    Returns
    -------
    dict
        A single entry matching the minimal structure of one element in
        ``index.json`` (``{"templates": [{"name", "models", "io"}]}``).
    """
    if not isinstance(workflow, dict):
        raise TypeError(f"workflow must be a dict, got {type(workflow).__name__}")

    # ------------------------------------------------------------------ io
    io_inputs: list[dict[str, Any]] = []
    io_outputs: list[dict[str, Any]] = []

    for node_id, node in workflow.items():
        if not isinstance(node, dict):
            continue
        class_type: str = node.get("class_type", "")

        if class_type in INPUT_NODE_TYPES:
            io_inputs.append(
                {
                    "nodeId": int(node_id) if str(node_id).isdigit() else node_id,
                    "nodeType": class_type,
                    "file": "",
                    "mediaType": INPUT_NODE_TYPES[class_type],
                }
            )
        elif class_type in OUTPUT_NODE_TYPES:
            io_outputs.append(
                {
                    "nodeId": int(node_id) if str(node_id).isdigit() else node_id,
                    "nodeType": class_type,
                    "file": "",
                    "mediaType": OUTPUT_NODE_TYPES[class_type],
                }
            )

    # Sort by node id so the order is deterministic
    io_inputs.sort(key=lambda x: x["nodeId"])
    io_outputs.sort(key=lambda x: x["nodeId"])

    # ------------------------------------------------------------------ assemble
    template: dict[str, Any] = {
        "name": name or "workflow",
        "models": _extract_models(workflow),
        "io": {
            "inputs": io_inputs,
            "outputs": io_outputs,
        },
    }

    entry: dict[str, Any] = {
        "templates": [template],
    }

    if update_index:
        append_to_index(entry, index_path)

    return entry


# ---------------------------------------------------------------------------
# CLI convenience
# ---------------------------------------------------------------------------


def _main() -> None:
    import argparse
    import sys

    parser = argparse.ArgumentParser(
        description="Parse a ComfyUI workflow JSON and print the index.json entry."
    )
    parser.add_argument("workflow_file", help="Path to the ComfyUI workflow .json file")
    # The CLI always uses the input file stem as the template name;
    # do not allow overriding via CLI to avoid accidental mismatches.
    parser.add_argument(
        "--index-path",
        default="",
        help="Override path to index.json (default: comfyui_workflow_templates_custom/index.json)",
    )
    args = parser.parse_args()

    path = Path(args.workflow_file)
    if not path.exists():
        print(f"File not found: {path}", file=sys.stderr)
        sys.exit(1)

    with open(path, encoding="utf-8") as f:
        workflow = json.load(f)

    name = path.stem

    # Pre-check: ensure the template name doesn't already exist in the index.
    idx_path = Path(args.index_path) if args.index_path else _custom_index_path()
    if idx_path.exists():
        with open(idx_path, encoding="utf-8") as f:
            raw = f.read().strip()
        try:
            existing_index = json.loads(raw) if raw else []
        except Exception:
            existing_index = []

        for group in (existing_index or []):
            for tpl in group.get("templates", []):
                if tpl.get("name") == name:
                    print(
                        json.dumps({"error": f"Template '{name}' already exists in {idx_path}"}),
                        file=sys.stderr,
                    )
                    sys.exit(2)

    entry = parse_workflow(
        workflow,
        name=name,
        update_index=True,
        index_path=args.index_path or None,
    )
    print(json.dumps(entry, indent=2))
    idx = Path(args.index_path) if args.index_path else _custom_index_path()
    print(f"\nIndex updated: {idx}", file=sys.stderr)


if __name__ == "__main__":
    _main()
