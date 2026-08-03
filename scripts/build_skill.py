#!/usr/bin/env python3
"""
build_skill.py — Generate a standards-compliant SKILL.md from a ComfyUI API-format workflow JSON.

Usage
-----
    python scripts/build_skill.py path/to/workflow.json [--name skill-name] [--dry-run]

On success prints:
    ✓  skills/<name>/SKILL.md
       Patchable : N nodes
       Locked    : N nodes
       Models    : N (check_model required)
       LLM call  : 1 (prose only)
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from textwrap import dedent
from typing import Any

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

from agenty_core.utils.secrets import get_secret
from agenty_core.utils.workflow_parser import INPUT_NODE_TYPES, OUTPUT_NODE_TYPES


# ---------------------------------------------------------------------------
# Settings helpers (mirrors agent.py _cfg / _parse_llm_setting)
# ---------------------------------------------------------------------------

_SETTINGS_CACHE: dict = {}


def _load_project_settings() -> dict:
    """Merged settings: committed TOML defaults ⊕ gitignored local JSON overrides.

    Self-contained (this is a standalone script) — mirrors src.utils.settings so the
    same precedence applies: settings.local.json is deep-merged over settings.default.toml.
    """
    global _SETTINGS_CACHE
    if _SETTINGS_CACHE:
        return _SETTINGS_CACHE
    import tomllib

    def _merge(base: dict, over: dict) -> dict:
        out = dict(base)
        for k, v in (over or {}).items():
            out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
        return out

    defaults: dict = {}
    local: dict = {}
    try:
        with open(PROJECT_ROOT / "config" / "settings.default.toml", "rb") as fh:
            defaults = tomllib.load(fh)
    except Exception:
        pass
    try:
        local = json.loads((PROJECT_ROOT / "config" / "settings.local.json").read_text(encoding="utf-8"))
    except Exception:
        pass
    _SETTINGS_CACHE = _merge(defaults, local)
    return _SETTINGS_CACHE


def _settings_get(*path: str, default: str = "") -> str:
    """Walk the merged settings[path] with a default."""
    node: Any = _load_project_settings()
    for key in path:
        if not isinstance(node, dict):
            return default
        node = node.get(key, {})
    return str(node) if node and not isinstance(node, dict) else default


def _parse_provider_model(value: str) -> tuple[str, str]:
    """Split 'provider,model' into (provider, model). Same logic as agent.py."""
    provider, _, model = value.partition(",")
    return provider.strip().lower(), model.strip()


# ---------------------------------------------------------------------------
# workflow_templates.json lookup
# ---------------------------------------------------------------------------

_WORKFLOW_TEMPLATES_CACHE: dict = {}


def _load_workflow_templates() -> dict:
    global _WORKFLOW_TEMPLATES_CACHE
    if _WORKFLOW_TEMPLATES_CACHE:
        return _WORKFLOW_TEMPLATES_CACHE
    wt_path = PROJECT_ROOT / "config" / "workflow_templates.json"
    if wt_path.exists():
        with open(wt_path, encoding="utf-8") as fh:
            _WORKFLOW_TEMPLATES_CACHE = json.load(fh)
    return _WORKFLOW_TEMPLATES_CACHE


def _lookup_template(workflow_path: Path) -> tuple[str, str]:
    """Return (template_name, known_description) for *workflow_path*.

    *template_name* is the exact key in workflow_templates.json (which equals
    the filename stem).  Returns the stem as-is and an empty description when
    the file is not registered in workflow_templates.json.
    """
    stem = workflow_path.stem  # exact filename stem, no extension
    templates = _load_workflow_templates()
    # Case-sensitive lookup first, then case-insensitive fallback
    if stem in templates:
        return stem, templates[stem]
    stem_lower = stem.lower()
    for key, desc in templates.items():
        if key.lower() == stem_lower:
            return key, desc
    return stem, ""


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SKILLS_DIR = PROJECT_ROOT / "skills"

_PATCHABLE_TITLE_KEYWORDS = re.compile(
    r"WIDTH|HEIGHT|LENGTH|FRAMES|FPS|STEPS|CFG|GUIDANCE|STRENGTH|SCALE|DENOISE",
    re.IGNORECASE,
)

_SAMPLER_REASON = "Sampler recipe — do not change."
_MATH_REASON = "Math passthrough — reads from upstream primitive."
_PREPROCESS_REASON = "Auto-driven by resolution primitives."
_WIRING_REASON = "Pure wiring — do not patch."
_CONDITIONING_REASON = "Conditioning node — do not patch."
_STRUCTURAL_REASON = "Structural node — do not patch."
_CFG1_REASON = "CFG=1 — distilled model, never raise."
_SIGMA_REASON = "Checkpoint-matched sigma schedule — do not change."
_FLUX_REASON = "max_shift=1.15, base_shift=0.5 are baked for Flux — do not change."

_SECTION_HEADINGS = [
    "## Workflow shape",
    "## Patchable nodes",
    "## DO NOT TOUCH",
    "## Validation checklist",
    "## Common patch bundle",
    "## Troubleshooting",
]


# ---------------------------------------------------------------------------
# Node classification
# ---------------------------------------------------------------------------

def _meta_title(node: dict) -> str:
    return (node.get("_meta") or {}).get("title", "")


def _classify_nodes(workflow: dict) -> tuple[list[dict], list[dict], list[dict]]:
    """Return (bucket_a_patchable, bucket_b_model_loaders, bucket_c_locked)."""
    bucket_a: list[dict] = []
    bucket_b: list[dict] = []
    bucket_c: list[dict] = []

    for node_id, node in workflow.items():
        if not isinstance(node, dict) or "class_type" not in node:
            continue
        ct: str = node.get("class_type", "")
        inputs: dict = node.get("inputs", {})
        title: str = _meta_title(node)

        # ── Bucket B — model loaders ────────────────────────────────────────
        loader_inputs = {"ckpt_name", "model_name", "text_encoder", "vae_name", "unet_name"}
        if ct.endswith("Loader") and loader_inputs.intersection(inputs.keys()):
            bucket_b.append({"id": node_id, "class_type": ct, "inputs": inputs, "title": title})
            continue

        # ── Bucket A — patchable ────────────────────────────────────────────
        placed_in_a = False

        if ct in ("LoadImage", "LoadImageMask", "VHS_LoadVideo", "VHS_LoadImages"):
            patchable_keys = [k for k in ("image", "video", "filename") if k in inputs]
            if patchable_keys:
                bucket_a.append({
                    "id": node_id, "class_type": ct, "title": title,
                    "group": "Input images / video",
                    "patchable": patchable_keys,
                    "inputs": inputs,
                })
                placed_in_a = True

        if not placed_in_a and "TextEncode" in ct:
            if "text" in inputs:
                bucket_a.append({
                    "id": node_id, "class_type": ct, "title": title,
                    "group": "Prompts",
                    "patchable": ["text"],
                    "inputs": inputs,
                })
                placed_in_a = True

        if not placed_in_a and ct in ("KSampler", "KSamplerAdvanced"):
            patch_keys = [k for k in ("seed", "steps", "cfg", "sampler_name", "scheduler", "denoise") if k in inputs]
            if patch_keys:
                bucket_a.append({
                    "id": node_id, "class_type": ct, "title": title,
                    "group": "Sampler params",
                    "patchable": patch_keys,
                    "inputs": inputs,
                })
                placed_in_a = True

        if not placed_in_a and ct == "RandomNoise":
            if "noise_seed" in inputs:
                bucket_a.append({
                    "id": node_id, "class_type": ct, "title": title,
                    "group": "Seed",
                    "patchable": ["noise_seed"],
                    "inputs": inputs,
                })
                placed_in_a = True

        if not placed_in_a and ct in ("PrimitiveInt", "PrimitiveFloat", "PrimitiveString"):
            if _PATCHABLE_TITLE_KEYWORDS.search(title) and "value" in inputs:
                group = _group_from_title(title)
                bucket_a.append({
                    "id": node_id, "class_type": ct, "title": title,
                    "group": group,
                    "patchable": ["value"],
                    "inputs": inputs,
                })
                placed_in_a = True

        if not placed_in_a and ct in ("SaveImage", "VHS_VideoCombine", "CreateVideo"):
            patch_keys = [k for k in ("filename_prefix", "format", "crf") if k in inputs]
            if patch_keys:
                bucket_a.append({
                    "id": node_id, "class_type": ct, "title": title,
                    "group": "Output",
                    "patchable": patch_keys,
                    "inputs": inputs,
                })
                placed_in_a = True

        if not placed_in_a and ct in ("LoraLoader", "LoraLoaderModelOnly"):
            patch_keys = [k for k in ("lora_name", "strength_model", "strength_clip") if k in inputs]
            if patch_keys:
                bucket_a.append({
                    "id": node_id, "class_type": ct, "title": title,
                    "group": "LoRA",
                    "patchable": patch_keys,
                    "inputs": inputs,
                })
                placed_in_a = True

        if not placed_in_a and ct.startswith("IPAdapter"):
            patch_keys = [k for k in ("weight", "weight_type") if k in inputs]
            if patch_keys:
                bucket_a.append({
                    "id": node_id, "class_type": ct, "title": title,
                    "group": "IP-Adapter",
                    "patchable": patch_keys,
                    "inputs": inputs,
                })
                placed_in_a = True

        if not placed_in_a:
            reason = _locked_reason(ct)
            bucket_c.append({
                "id": node_id, "class_type": ct, "title": title,
                "reason": reason,
                "inputs": inputs,
            })

    return bucket_a, bucket_b, bucket_c


def _group_from_title(title: str) -> str:
    title_up = title.upper()
    if "WIDTH" in title_up or "HEIGHT" in title_up:
        return "Resolution"
    if "LENGTH" in title_up or "FRAMES" in title_up:
        return "Length / FPS"
    if "FPS" in title_up:
        return "Length / FPS"
    if "STEPS" in title_up:
        return "Sampler params"
    if "CFG" in title_up or "GUIDANCE" in title_up:
        return "Sampler params"
    if "STRENGTH" in title_up or "DENOISE" in title_up or "SCALE" in title_up:
        return "Sampler params"
    return "Parameters"


def _locked_reason(ct: str) -> str:
    if re.search(r"Sampler|Guider|Sigma", ct):
        return _SAMPLER_REASON
    if re.search(r"MathExpression", ct):
        return _MATH_REASON
    if re.search(r"Preprocess|Resize|GetImageSize", ct):
        return _PREPROCESS_REASON
    if re.search(r"Concat|Separate|Crop|Guide|Plumbing", ct):
        return _WIRING_REASON
    if re.search(r"Conditioning|CLIPVision", ct):
        return _CONDITIONING_REASON
    return _STRUCTURAL_REASON


# ---------------------------------------------------------------------------
# Reverse-dependency map
# ---------------------------------------------------------------------------

def _build_reverse_deps(workflow: dict) -> dict[str, list[str]]:
    """Map node_id → [list of node_ids that reference it via [node_id, slot] links]."""
    rev: dict[str, list[str]] = {k: [] for k in workflow}
    for node_id, node in workflow.items():
        if not isinstance(node, dict):
            continue
        for val in (node.get("inputs") or {}).values():
            if isinstance(val, list) and len(val) == 2 and isinstance(val[0], str):
                upstream_id = val[0]
                if upstream_id in rev:
                    rev[upstream_id].append(node_id)
    return rev


def _apply_downstream_notes(
    bucket_a: list[dict],
    bucket_c: list[dict],
    rev_deps: dict[str, list[str]],
) -> dict[str, dict[str, str]]:
    """Return a mapping of {primitive_node_id: {patchable_input: extra_note}}."""
    c_ids = {n["id"] for n in bucket_c}
    notes: dict[str, dict[str, str]] = {}
    for node in bucket_a:
        node_id = node["id"]
        downstream_c = [d for d in rev_deps.get(node_id, []) if d in c_ids]
        if downstream_c:
            for inp in node["patchable"]:
                existing = notes.setdefault(node_id, {})
                ids_str = ", ".join(f"`{d}`" for d in downstream_c)
                existing[inp] = f"Do not touch node {ids_str} — reads from this automatically."
    return notes


# ---------------------------------------------------------------------------
# Constraint inference
# ---------------------------------------------------------------------------

def _infer_constraints(
    workflow: dict,
    bucket_a: list[dict],
    bucket_b: list[dict],
    bucket_c: list[dict],
) -> dict:
    """Return a dict of derived constraint info.

    Keys:
      ltxv: bool
      manual_sigmas: list of node ids
      flux_sampling: list of node ids
      has_audio: bool
      cfg_is_one_nodes: list of node ids (KSampler with cfg==1)
      lora_files: list of str
    """
    all_node_ids = {n["id"] for n in bucket_a + bucket_b + bucket_c}
    all_class_types = {n["id"]: n["class_type"] for n in bucket_a + bucket_b + bucket_c}

    ltxv = any("LTXVAddGuide" in n["class_type"] for n in bucket_c)
    manual_sigmas = [n["id"] for n in bucket_c if "ManualSigmas" in n["class_type"]]
    flux_sampling = [n["id"] for n in bucket_c if "ModelSamplingFlux" in n["class_type"]]
    has_audio = any(
        re.search(r"Audio", n["class_type"], re.IGNORECASE)
        for n in bucket_a + bucket_b + bucket_c
    )
    cfg_is_one = []
    for n in bucket_a:
        if n["class_type"] in ("KSampler", "KSamplerAdvanced"):
            cfg_val = n["inputs"].get("cfg")
            if cfg_val == 1 or cfg_val == 1.0:
                cfg_is_one.append(n["id"])
    # Also check CFGGuider in bucket_c
    for n in bucket_c:
        if "CFGGuider" in n["class_type"]:
            cfg_val = n["inputs"].get("cfg")
            if cfg_val == 1 or cfg_val == 1.0:
                cfg_is_one.append(n["id"])

    lora_files: list[str] = []
    for n in bucket_a:
        if n["class_type"] in ("LoraLoader", "LoraLoaderModelOnly"):
            lname = n["inputs"].get("lora_name")
            if isinstance(lname, str):
                lora_files.append(lname)

    return {
        "ltxv": ltxv,
        "manual_sigmas": manual_sigmas,
        "flux_sampling": flux_sampling,
        "has_audio": has_audio,
        "cfg_is_one": cfg_is_one,
        "lora_files": lora_files,
    }


# ---------------------------------------------------------------------------
# Analysis summary builder
# ---------------------------------------------------------------------------

def _build_analysis(
    workflow: dict,
    bucket_a: list[dict],
    bucket_b: list[dict],
    bucket_c: list[dict],
    constraints: dict,
    rev_deps: dict[str, list[str]],
    downstream_notes: dict[str, dict[str, str]],
) -> dict:
    """Build the analysis_summary dict passed to the LLM."""

    # What media inputs are present?
    media_inputs = []
    for n in bucket_a:
        if n["class_type"] in ("LoadImage", "LoadImageMask"):
            media_inputs.append("image")
        elif n["class_type"] in ("VHS_LoadVideo", "VHS_LoadImages"):
            media_inputs.append("video")
    media_inputs = sorted(set(media_inputs)) or ["none"]

    # Output nodes
    output_nodes = [n for n in bucket_a if n["class_type"] in ("SaveImage", "VHS_VideoCombine", "CreateVideo")]
    if output_nodes:
        out_ct = output_nodes[0]["class_type"]
        if out_ct == "SaveImage":
            output_type = "image (PNG/WebP)"
        elif out_ct == "VHS_VideoCombine":
            output_type = "video (VHS container)"
        else:
            output_type = "video"
    else:
        output_type = "unknown"

    # Model info from bucket_b
    model_files: list[str] = []
    for n in bucket_b:
        for key in ("ckpt_name", "model_name", "unet_name"):
            val = n["inputs"].get(key)
            if isinstance(val, str) and val:
                model_files.append(val)

    # Sampler info
    sampler_info = {}
    for n in bucket_a:
        if n["class_type"] in ("KSampler", "KSamplerAdvanced"):
            sampler_info = {
                "sampler_name": n["inputs"].get("sampler_name", "euler"),
                "steps": n["inputs"].get("steps", "?"),
                "cfg": n["inputs"].get("cfg", "?"),
                "scheduler": n["inputs"].get("scheduler", "?"),
                "denoise": n["inputs"].get("denoise", "?"),
            }
            break

    # Primitive groups summary
    groups: dict[str, list[str]] = {}
    for n in bucket_a:
        g = n.get("group", "Other")
        groups.setdefault(g, []).append(n["id"])

    # Build patchable summary for LLM
    patchable_summary = []
    for n in bucket_a:
        patchable_summary.append({
            "id": n["id"],
            "class_type": n["class_type"],
            "title": n.get("title", ""),
            "group": n.get("group", ""),
            "patchable_inputs": n["patchable"],
        })

    # Build locked summary
    locked_summary = []
    for n in bucket_c:
        locked_summary.append({
            "id": n["id"],
            "class_type": n["class_type"],
            "reason": n["reason"],
        })

    # Model loaders
    model_loaders = [
        {"id": n["id"], "class_type": n["class_type"], "inputs": {
            k: v for k, v in n["inputs"].items()
            if k in ("ckpt_name", "model_name", "text_encoder", "vae_name", "unet_name")
            and isinstance(v, str)
        }}
        for n in bucket_b
    ]

    return {
        "media_inputs": media_inputs,
        "output_type": output_type,
        "model_files": model_files,
        "sampler_info": sampler_info,
        "constraints": {
            "ltxv_guide": constraints["ltxv"],
            "manual_sigmas": constraints["manual_sigmas"],
            "flux_sampling": constraints["flux_sampling"],
            "has_audio": constraints["has_audio"],
            "cfg_is_one_nodes": constraints["cfg_is_one"],
            "lora_files": constraints["lora_files"],
        },
        "patchable_count": len(bucket_a),
        "locked_count": len(bucket_c),
        "model_loader_count": len(bucket_b),
        "patchable_nodes": patchable_summary,
        "locked_nodes": locked_summary,
        "model_loaders": model_loaders,
        "patchable_groups": {g: ids for g, ids in groups.items()},
        # Populated later by build_skill_from_workflow / main()
        "template_name": "",
        "known_description": "",
    }


# ---------------------------------------------------------------------------
# LLM call — prose only
# ---------------------------------------------------------------------------

def _call_llm_prose(analysis_summary: dict, skill_name: str) -> dict:
    """Make ONE LLM call for description, troubleshooting, notes_overrides.

    Provider and model are read from ``llm.pipeline.build_skill`` in
    settings.json (format: ``'provider,model'``, e.g. ``'ollama,gemma4:26b'``
    or ``'claude,claude-haiku-4-5'``).  Falls back to deterministic template
    prose if the call fails or the response cannot be parsed.
    """
    template_name = analysis_summary.get("template_name") or skill_name
    known_desc = analysis_summary.get("known_description", "")
    known_desc_note = (
        f"\nKNOWN DESCRIPTION (from workflow_templates.json — use as authoritative context):\n{known_desc}\n"
        if known_desc else ""
    )
    prompt = (
        "You are writing documentation for a ComfyUI workflow automation agent.\n"
        "Given the workflow analysis below, return exactly this JSON structure and nothing else.\n"
        "No preamble, no markdown fences.\n\n"
        f"ANALYSIS:\n{json.dumps(analysis_summary, indent=2)}\n"
        f"{known_desc_note}\n"
        "Return:\n"
        "{\n"
        '  "description": "<single paragraph>",\n'
        '  "troubleshooting": ["<symptom> → <fix>", ...],\n'
        '  "notes_overrides": {"<node_id>": "<extra note>"}\n'
        "}\n\n"
        "description rules:\n"
        f'- Start: "Patch and validate [{_workflow_purpose(analysis_summary)}] workflows."\n'
        f'- Must include: "Activate when brainbriefing template name contains \'{template_name}\'."\n'
        f"  (Use that exact string — \'{template_name}\' — not a kebab or lowercased variant.)\n"
        "- End with comma-joined summary of patchable and locked categories.\n"
        "- One paragraph, no line breaks.\n\n"
        "troubleshooting rules:\n"
        "- 4–6 entries as single strings: \"**<symptom>** → <fix referencing node IDs where possible>\"\n"
        "- Cover: model missing, OOM, constraint violations found in analysis, bad output quality.\n\n"
        "notes_overrides: only emit if there are edge cases not covered by the classification rules.\n"
        "Omit the key entirely if empty."
    )

    # Resolve provider + model from settings.json (same pattern as agent.py)
    raw_setting = _settings_get("llm", "pipeline", "build_skill", default="ollama,gemma4:26b")
    provider, model_id = _parse_provider_model(raw_setting)

    raw_text: str | None = None

    if provider == "claude":
        _model = model_id or _settings_get("llm", "anthropic", "model", default="claude-haiku-4-5")
        api_key = get_secret("ANTHROPIC_API_KEY")
        if not api_key:
            print("[build_skill] ANTHROPIC_API_KEY not set — falling back to Ollama.", file=sys.stderr)
        else:
            try:
                import anthropic as _anthropic
                client = _anthropic.Anthropic(api_key=api_key)
                msg = client.messages.create(
                    model=_model,
                    max_tokens=600,
                    messages=[{"role": "user", "content": prompt}],
                )
                raw_text = msg.content[0].text
            except Exception as exc:
                print(f"[build_skill] Claude call failed: {exc}. Falling back to Ollama.", file=sys.stderr)

    # Ollama (primary when provider=='ollama', or fallback from Claude)
    if raw_text is None:
        _model = model_id if provider == "ollama" else _settings_get("llm", "pipeline", "llm_functions", default="gemma4:26b")
        _host = _settings_get("llm", "ollama", "host", default="http://localhost:11434")
        try:
            raw_text = _ollama_chat_sync(prompt, model=_model, host=_host)
        except Exception as exc:
            print(f"[build_skill] Ollama call failed: {exc}. Using template fallback.", file=sys.stderr)

    if raw_text:
        try:
            # Strip markdown fences if present
            cleaned = re.sub(r"^```[a-z]*\n?", "", raw_text.strip())
            cleaned = re.sub(r"\n?```$", "", cleaned)
            return json.loads(cleaned)
        except (json.JSONDecodeError, ValueError) as exc:
            print(f"[build_skill] LLM JSON parse failed: {exc}. Using template fallback.", file=sys.stderr)

    # Deterministic fallback
    purpose = _workflow_purpose(analysis_summary)
    template_name = analysis_summary.get("template_name") or skill_name
    return {
        "description": (
            f"Patch and validate {purpose} workflows. "
            f"Activate when brainbriefing template name contains '{template_name}'. "
            f"Patchable groups: {', '.join(analysis_summary.get('patchable_groups', {}).keys()) or 'none'}. "
            f"Locked nodes: {analysis_summary['locked_count']}. "
            f"Model loaders require check_model before patching."
        ),
        "troubleshooting": [
            "**Missing model file** → run `check_model` on the checkpoint name and download via HuggingFace if absent.",
            "**OOM / out of memory** → reduce resolution primitives (width/height nodes) and retry.",
            "**Black / empty output** → verify all image inputs were uploaded and patched before running.",
            "**Workflow validation fails** → re-check all required node patches are applied and `signal_workflow_ready` was called.",
        ],
        "notes_overrides": {},
    }


# ---------------------------------------------------------------------------
# Workflow template description generation
# ---------------------------------------------------------------------------


def _extract_model_hints_from_filename(workflow_name: str) -> dict:
    """Extract model/service hints from the template filename.
    
    Returns a dict with keys like 'provider', 'model_name', 'api_type', etc.
    """
    name_lower = workflow_name.lower()
    hints = {}
    
    # Detect provider from filename
    if "api_" in name_lower:
        hints["provider"] = "API"
        if "kling" in name_lower:
            hints["service"] = "Kling"
        elif "ideogram" in name_lower:
            hints["service"] = "Ideogram"
        elif "seedream" in name_lower:
            hints["service"] = "ByteDance Seedream"
        elif "magnific" in name_lower:
            hints["service"] = "Magnific"
        elif "meshy" in name_lower:
            hints["service"] = "Meshy"
        elif "topaz" in name_lower:
            hints["service"] = "Topaz"
        elif "veo" in name_lower:
            hints["service"] = "Google Veo"
        elif "ltx" in name_lower:
            hints["service"] = "Lightricks LTX"
        elif "wan" in name_lower:
            hints["service"] = "Wan"
        elif "nano" in name_lower and "banana" in name_lower:
            hints["service"] = "Nano-Banana"
    elif "image_" in name_lower or "video_" in name_lower:
        hints["provider"] = "Local"
        if "flux" in name_lower:
            hints["model_name"] = "Flux"
        elif "ltx" in name_lower:
            hints["model_name"] = "LTX-Video"
        elif "wan" in name_lower:
            hints["model_name"] = "Wan"
        elif "kling" in name_lower:
            hints["model_name"] = "Kling"
        elif "qwen" in name_lower:
            hints["model_name"] = "Qwen"
        elif "zimage" in name_lower or "z_image_turbo" in name_lower:
            hints["model_name"] = "Z-Image-Turbo"
    else:
        # Try to extract from any part of the name
        if "flux" in name_lower:
            hints["provider"] = "Local"
            hints["model_name"] = "Flux"
        elif "ltx" in name_lower or "ltxv" in name_lower:
            hints["provider"] = "Local"
            hints["model_name"] = "LTX-Video"
        elif "wan" in name_lower:
            hints["provider"] = "Local"
            hints["model_name"] = "Wan"
        elif "nano" in name_lower and "banana" in name_lower:
            hints["provider"] = "API"
            hints["service"] = "Nano-Banana"
    
    return hints


def _try_websearch_workflow_description(workflow_name: str) -> str | None:
    """Try to use Strands webSearch tool to find workflow description.
    
    Returns the search result or None if unavailable.
    """
    try:
        # Check if webSearch is available from Strands
        from strands_tools import webSearch
        
        # Format the search query
        clean_name = workflow_name.replace("_", " ").replace("-", " ")
        query = f"ComfyUI workflow {clean_name}"
        
        # Call webSearch (this is synchronous for now)
        result = webSearch(query)
        if result and isinstance(result, str) and len(result) > 20:
            return result
    except ImportError:
        # webSearch not available in this version of Strands
        pass
    except Exception as exc:
        # Silently fail and fall back to other methods
        print(f"[generate_description] webSearch failed: {exc}", file=sys.stderr)
    
    return None


def _detect_api_or_local(workflow: dict) -> tuple[str, str]:
    """Detect if workflow is API-based or local. Return (type, service_name).
    
    Returns
    -------
    tuple[str, str]
        ("API", "Service Name") or ("Local", "Model Family")
    """
    # Collect all class types
    class_types = []
    for node in workflow.values():
        if isinstance(node, dict) and "class_type" in node:
            class_types.append(node["class_type"])
    
    all_types = " ".join(class_types).lower()
    
    # Detect API services
    if "api_" in all_types or "klingapinode" in all_types:
        if "kling" in all_types:
            return "API", "Kling"
        if "ideogram" in all_types:
            return "API", "Ideogram"
        if "seedream" in all_types:
            return "API", "ByteDance Seedream"
        if "magnific" in all_types:
            return "API", "Magnific"
        if "meshy" in all_types:
            return "API", "Meshy"
        if "topaz" in all_types:
            return "API", "Topaz"
        if "ltx" in all_types and "api" in all_types:
            return "API", "Lightricks LTX"
        if "veo" in all_types:
            return "API", "Google Veo"
        if "wan" in all_types and "api" in all_types:
            return "API", "Wan"
        if "nano_banana" in all_types or "nanobanana" in all_types:
            return "API", "Nano-Banana"
        return "API", "API Service"
    
    # Detect local models
    if "flux" in all_types:
        return "Local", "Flux"
    if "ltx" in all_types or "ltxv" in all_types:
        return "Local", "LTX-Video"
    if "wan" in all_types:
        return "Local", "Wan"
    if "kling" in all_types:
        return "Local", "Kling"
    if "qwen" in all_types:
        return "Local", "Qwen"
    if "zimage" in all_types or "z_image" in all_types:
        return "Local", "Z-Image-Turbo"
    
    # Default to local
    return "Local", "ComfyUI Model"


def _extract_count_info(workflow: dict) -> tuple[int, int]:
    """Count distinct input and output nodes. Return (input_count, output_count).
    
    Uses the same INPUT_NODE_TYPES and OUTPUT_NODE_TYPES definitions as workflow_parser.
    """
    input_count = 0
    output_count = 0
    
    for node in workflow.values():
        if isinstance(node, dict):
            ct = node.get("class_type", "")
            if ct in INPUT_NODE_TYPES:
                input_count += 1
            elif ct in OUTPUT_NODE_TYPES:
                output_count += 1
    
    return max(1, input_count), max(1, output_count)


def _generate_workflow_template_description(workflow: dict, workflow_name: str) -> str:
    """Generate a one-line description for workflow_templates.json.
    
    Strategy (in order of priority):
    1. Try webSearch to find what the workflow does
    2. Extract hints from the workflow filename (model, API/Local detection)
    3. Analyze the workflow structure (inputs/outputs, node types)
    4. Use LLM to generate a polished description
    5. Fall back to deterministic template if all else fails
    
    Parameters
    ----------
    workflow:
        The workflow dict
    workflow_name:
        The template name (filename stem)
    
    Returns
    -------
    str
        Description in the format: "[API|Local] <operation> via <service>. <inputs> → <outputs>. <purpose>"
    """
    # ─ Step 1: Try webSearch first ─────────────────────────────────────────
    websearch_result = _try_websearch_workflow_description(workflow_name)
    if websearch_result:
        print(f"[generate_description] webSearch found info for '{workflow_name}'", file=sys.stderr)
        # Use webSearch result to inform the LLM call
    
    # ─ Step 2: Extract hints from filename ──────────────────────────────────
    filename_hints = _extract_model_hints_from_filename(workflow_name)
    
    # ─ Step 3: Extract basic workflow info ─────────────────────────────────
    wf_type, service = _detect_api_or_local(workflow)
    input_count, output_count = _extract_count_info(workflow)
    
    # Classify nodes for analysis
    bucket_a, bucket_b, bucket_c = _classify_nodes(workflow)
    rev_deps = _build_reverse_deps(workflow)
    downstream_notes = _apply_downstream_notes(bucket_a, bucket_c, rev_deps)
    constraints = _infer_constraints(workflow, bucket_a, bucket_b, bucket_c)
    
    # Build minimal analysis
    analysis = {
        "workflow_type": wf_type,
        "service": service,
        "input_count": input_count,
        "output_count": output_count,
        "patchable_count": len(bucket_a),
        "locked_count": len(bucket_c),
        "model_loader_count": len(bucket_b),
        "template_name": workflow_name,
        "filename_hints": filename_hints,
        "websearch_info": websearch_result,
    }
    
    # Extract media info
    media_inputs = []
    for n in bucket_a:
        if n["class_type"] in ("LoadImage", "LoadImageMask"):
            media_inputs.append("image")
        elif n["class_type"] in ("VHS_LoadVideo", "VHS_LoadImages"):
            media_inputs.append("video")
    media_inputs = sorted(set(media_inputs)) or ["text"]
    analysis["media_types"] = media_inputs
    
    # Detect output type
    output_nodes = [n for n in bucket_a if n["class_type"] in ("SaveImage", "VHS_VideoCombine", "CreateVideo")]
    if output_nodes:
        out_ct = output_nodes[0]["class_type"]
        if out_ct == "SaveImage":
            analysis["output_type"] = "image"
        else:
            analysis["output_type"] = "video"
    else:
        analysis["output_type"] = "unknown"
    
    # Detect operation type
    if "video" in analysis["output_type"]:
        if "image" in analysis["media_types"]:
            operation = "image-to-video"
        else:
            operation = "text-to-video"
    elif "image" in analysis["media_types"]:
        operation = "image editing"
    else:
        operation = "generation"
    
    analysis["operation"] = operation
    
    # Apply filename hints to analysis if not already detected
    if filename_hints.get("provider") and not wf_type:
        wf_type = filename_hints["provider"]
    if filename_hints.get("service") and service == "API Service":
        service = filename_hints["service"]
    if filename_hints.get("model_name") and service == "ComfyUI Model":
        service = filename_hints["model_name"]
    
    # ─ Step 4: Build LLM prompt for template description ──────────────────
    prompt = (
        "You are writing a concise, one-line description for a ComfyUI workflow template.\n"
        "These descriptions go into config/workflow_templates.json and help users understand what the workflow does.\n\n"
        f"WORKFLOW ANALYSIS:\n{json.dumps(analysis, indent=2)}\n\n"
        "Return ONLY a single line description following this exact format:\n"
        f'"[{wf_type}] [operation] via [{service}]. [inputs_summary] → [outputs_summary]. [brief_purpose]."\n\n'
        "Examples:\n"
        '"API image editing via ByteDance Seedream 5.0 Lite. 2 image inputs + text prompt → 1 image output. Modifies a source image based on prompt instructions, using a second image as additional reference."\n'
        '"Local text-to-image via Flux 2 Klein distilled. Text prompt only → 2 image outputs. BFL distilled model delivering outstanding quality at sub-second speed, ideal for real-time generation."\n'
        '"API first-last-frame-to-video via Kling O3 (Kling 3.0). Up to 4 reference/keyframe images → 1 video output. Generates videos with precise semantic control, longer duration, and improved narrative coherence."\n\n'
        "Rules:\n"
        "1. Start with [API|Local] exactly as shown.\n"
        "2. Include operation type (text-to-image, image-to-video, editing, upscaling, etc).\n"
        f'3. Include "via [{service}]" with the exact service/model name.\n'
        f"4. Summarize inputs based on analysis: {input_count} input(s) of type {analysis['media_types']}.\n"
        f"5. Summarize outputs: {output_count} output(s) of type {analysis['output_type']}.\n"
        "6. End with a single sentence describing purpose/capabilities.\n"
        "7. Start first letter after numbers/brackets as uppercase.\n"
        "8. Keep entire description under 200 characters if possible.\n"
        "Return ONLY the description string, no quotes, no preamble."
    )
    
    # Resolve provider + model
    raw_setting = _settings_get("llm", "pipeline", "build_skill", default="ollama,gemma4:26b")
    provider, model_id = _parse_provider_model(raw_setting)
    
    raw_text: str | None = None
    
    # Try Claude first if configured
    if provider == "claude":
        _model = model_id or _settings_get("llm", "anthropic", "model", default="claude-haiku-4-5")
        api_key = get_secret("ANTHROPIC_API_KEY")
        if api_key:
            try:
                import anthropic as _anthropic
                client = _anthropic.Anthropic(api_key=api_key)
                msg = client.messages.create(
                    model=_model,
                    max_tokens=300,
                    messages=[{"role": "user", "content": prompt}],
                )
                raw_text = msg.content[0].text.strip()
            except Exception as exc:
                print(f"[generate_description] Claude call failed: {exc}. Falling back to Ollama.", file=sys.stderr)
    
    # Try Ollama (primary when provider=='ollama', or fallback from Claude)
    if raw_text is None:
        _model = model_id if provider == "ollama" else _settings_get("llm", "pipeline", "llm_functions", default="gemma4:26b")
        _host = _settings_get("llm", "ollama", "host", default="http://localhost:11434")
        try:
            raw_text = _ollama_chat_sync(prompt, model=_model, host=_host)
        except Exception as exc:
            print(f"[generate_description] Ollama call failed: {exc}. Using template fallback.", file=sys.stderr)
            raw_text = None
    
    # Use LLM response if available
    if raw_text:
        desc = raw_text.strip()
        # Clean up any stray quotes
        desc = desc.strip('"\'')
        if desc and len(desc) > 10:  # Sanity check
            return desc
    
    # ─ Step 5: Fallback: deterministic template description ───────────────
    input_desc = f"{input_count} " + ("image" if "image" in analysis["media_types"] else "") + ("video" if "video" in analysis["media_types"] else "") + " input" + ("s" if input_count > 1 else "")
    if "image" not in analysis["media_types"] and "video" not in analysis["media_types"]:
        input_desc = "text input"
    
    output_desc = f"{output_count} " + ("image" if analysis["output_type"] == "image" else "video") + " output" + ("s" if output_count > 1 else "")
    
    return f"{wf_type} {operation} via {service}. {input_desc} → {output_desc}. Processes and generates content using ComfyUI workflows."


def _workflow_purpose(analysis: dict) -> str:
    inputs = analysis.get("media_inputs", [])
    output = analysis.get("output_type", "")
    if "video" in output.lower():
        if "image" in inputs:
            return "image-to-video"
        return "text-to-video"
    if "image" in inputs:
        return "image editing"
    return "text-to-image"


def _ollama_chat_sync(prompt: str, model: str, host: str) -> str:
    """Synchronous Ollama /api/chat call via httpx (works inside and outside async loops)."""
    import httpx
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
    }
    with httpx.Client(timeout=120.0) as client:
        resp = client.post(f"{host}/api/chat", json=payload)
    resp.raise_for_status()
    return resp.json()["message"]["content"]


# ---------------------------------------------------------------------------
# SKILL.md assembly
# ---------------------------------------------------------------------------

_SKILL_TEMPLATE = """\
---
name: {name}
description: {description}
allowed-tools: patch_workflow check_model
---

# {name_title} workflow assembly

{one_line_summary}

---

## Workflow shape

- **Inputs:** {inputs_str}
- **Output:** {output_str}
- **Model:** {model_str}
- **Sampler:** {sampler_str}
- **CFG:** {cfg_str}

---

## Patchable nodes — the ONLY things you may change

All patches go through `patch_workflow({{node_id, input_name, value}})`.

{patchable_section}

---

## DO NOT TOUCH — these nodes are fixed

### Model loaders

| Node | Reason |
|---|---|
{model_loader_rows}

Use `check_model` to verify each file is present locally and resolve its full path
before patching the node. Missing models are handled upstream by the Researcher
(HuggingFace download) — the Brain's job is path resolution, not availability checking.

### Locked structural nodes

| Node | Reason |
|---|---|
{locked_rows}

---

## Validation checklist

{checklist}

If all pass → `signal_workflow_ready(workflow_path)`.

---

## Common patch bundle

```json
{patch_bundle}
```

Send all patches in a single `patch_workflow` call.

---

## Troubleshooting

{troubleshooting_lines}
"""


def _build_skill_md(
    skill_name: str,
    workflow: dict,
    bucket_a: list[dict],
    bucket_b: list[dict],
    bucket_c: list[dict],
    constraints: dict,
    downstream_notes: dict[str, dict[str, str]],
    analysis: dict,
    llm_prose: dict,
) -> str:
    name_title = skill_name.replace("-", " ").title()

    # ── inputs / output / model / sampler strings ────────────────────────────
    inputs_str = ", ".join(analysis["media_inputs"]) if analysis["media_inputs"] != ["none"] else "text only"
    output_str = analysis["output_type"]
    model_str = ", ".join(f"`{m}`" for m in analysis["model_files"]) if analysis["model_files"] else "N/A"

    si = analysis.get("sampler_info", {})
    if si:
        sampler_str = f"{si.get('sampler_name', '?')} {si.get('steps', '?')} steps, {si.get('scheduler', '?')} scheduler"
    else:
        sampler_str = "N/A"

    cfg_val = si.get("cfg", "?") if si else "?"
    cfg_note = " (distilled — never raise)" if analysis["constraints"]["cfg_is_one_nodes"] else ""
    cfg_str = f"{cfg_val}{cfg_note}"

    # ── Patchable section ───────────────────────────────────────────────────
    patchable_section = _build_patchable_section(
        bucket_a, constraints, downstream_notes, analysis, llm_prose
    )

    # ── Model loader rows ───────────────────────────────────────────────────
    if bucket_b:
        ml_rows = []
        for n in bucket_b:
            files_str = ", ".join(
                f"`{v}`"
                for k, v in n["inputs"].items()
                if k in ("ckpt_name", "model_name", "unet_name", "text_encoder", "vae_name")
                and isinstance(v, str)
            )
            reason = f"Loads {files_str} — swap not safe without full reconfig."
            if constraints["cfg_is_one"] and "checkpoint" in n["class_type"].lower():
                reason += " " + _CFG1_REASON
            ml_rows.append(f"| `{n['id']}` `{n['class_type']}` ({n['title'] or 'no title'}) | {reason} |")
        model_loader_rows = "\n".join(ml_rows)
    else:
        model_loader_rows = "| — | No model loaders detected. |"

    # ── Locked rows ─────────────────────────────────────────────────────────
    # Apply constraint overrides to reasons
    c_id_to_reason: dict[str, str] = {}
    for n in bucket_c:
        r = n["reason"]
        if n["id"] in constraints["manual_sigmas"]:
            r = _SIGMA_REASON
        elif n["id"] in constraints["flux_sampling"]:
            r = _FLUX_REASON
        c_id_to_reason[n["id"]] = r

    if bucket_c:
        locked_rows = "\n".join(
            f"| `{n['id']}` `{n['class_type']}` ({n['title'] or 'no title'}) | {c_id_to_reason[n['id']]} |"
            for n in bucket_c
        )
    else:
        locked_rows = "| — | No locked nodes detected. |"

    # ── Validation checklist ─────────────────────────────────────────────────
    checklist = _build_checklist(bucket_a, bucket_b, constraints)

    # ── Patch bundle ─────────────────────────────────────────────────────────
    patch_bundle = _build_patch_bundle(bucket_a, constraints)

    # ── Troubleshooting ─────────────────────────────────────────────────────
    ts_items = llm_prose.get("troubleshooting") or []
    if constraints["has_audio"] and not any("audio" in t.lower() or "music" in t.lower() for t in ts_items):
        ts_items.append("**No audio / silent output** → check positive prompt contains a `Music: <style>` line.")
    troubleshooting_lines = "\n".join(f"- {t}" for t in ts_items) if ts_items else "- No known issues."

    # ── one-line summary ─────────────────────────────────────────────────────
    purpose = _workflow_purpose(analysis)
    model_fam = _infer_model_family(analysis["model_files"])
    one_line_summary = f"Assembles and validates {purpose} workflows using {model_fam}."

    # ── description (escape YAML special chars) ──────────────────────────────
    desc = llm_prose.get("description", "")
    # Wrap in double quotes, escape inner quotes
    desc_yaml = '"' + desc.replace('"', '\\"') + '"'

    return _SKILL_TEMPLATE.format_map({
        "name": skill_name,
        "name_title": name_title,
        "description": desc_yaml,
        "one_line_summary": one_line_summary,
        "inputs_str": inputs_str,
        "output_str": output_str,
        "model_str": model_str,
        "sampler_str": sampler_str,
        "cfg_str": cfg_str,
        "patchable_section": patchable_section,
        "model_loader_rows": model_loader_rows,
        "locked_rows": locked_rows,
        "checklist": checklist,
        "patch_bundle": patch_bundle,
        "troubleshooting_lines": troubleshooting_lines,
    })


def _infer_model_family(model_files: list[str]) -> str:
    if not model_files:
        return "the configured model"
    combined = " ".join(model_files).lower()
    if "flux" in combined:
        return "Flux"
    if "ltxv" in combined or "ltx" in combined:
        return "LTX-Video"
    if "wan" in combined:
        return "Wan"
    if "kling" in combined:
        return "Kling"
    if "qwen" in combined:
        return "Qwen"
    if "sdxl" in combined:
        return "SDXL"
    if "sd" in combined:
        return "Stable Diffusion"
    # Use first filename stem
    return Path(model_files[0]).stem


def _build_patchable_section(
    bucket_a: list[dict],
    constraints: dict,
    downstream_notes: dict[str, dict[str, str]],
    analysis: dict,
    llm_prose: dict,
) -> str:
    from collections import defaultdict
    groups: dict[str, list[dict]] = defaultdict(list)
    for n in bucket_a:
        groups[n.get("group", "Other")].append(n)

    notes_overrides: dict[str, str] = llm_prose.get("notes_overrides") or {}

    # Do any nodes need upload instructions?
    has_upload = any(
        n["class_type"] in ("LoadImage", "LoadImageMask", "VHS_LoadVideo", "VHS_LoadImages")
        for n in bucket_a
    )

    parts = []
    for group_name, nodes in groups.items():
        parts.append(f"### {group_name}")
        parts.append("")
        parts.append("| Node | Type | Input | Notes |")
        parts.append("|---|---|---|---|")
        for n in nodes:
            node_id = n["id"]
            ct = n["class_type"]
            title_str = n.get("title") or ""
            title_part = f" ({title_str})" if title_str else ""
            for inp in n["patchable"]:
                notes_list = []
                # Constraint-based notes
                if constraints["ltxv"] and inp == "value" and group_name == "Length / FPS":
                    notes_list.append("Must satisfy length = 8n + 1 (e.g. 49, 73, 97, 121, 161).")
                if constraints["has_audio"] and ct in ("CLIPTextEncode",) and "text" == inp:
                    group = n.get("group", "")
                    if "positive" in title_str.lower() or "prompt" in title_str.lower():
                        notes_list.append("Must include a `Music: <style>` line — parsed by the audio decoder.")
                if node_id in downstream_notes and inp in downstream_notes[node_id]:
                    notes_list.append(downstream_notes[node_id][inp])
                if node_id in notes_overrides:
                    notes_list.append(notes_overrides[node_id])
                notes_str = " ".join(notes_list) if notes_list else ""
                parts.append(f"| `{node_id}` | `{ct}`{title_part} | `{inp}` | {notes_str} |")
        parts.append("")

    if has_upload:
        parts.append("Always upload input files via `upload_image` before patching.")
        parts.append("")

    # LTXV semi-patchable guide nodes
    if constraints["ltxv"]:
        parts.append("### LTXV Guide nodes (semi-patchable)")
        parts.append("")
        parts.append("| Input | Valid range | Note |")
        parts.append("|---|---|---|")
        parts.append("| `strength` | 0.5–0.9 | Asymmetric values cause instability. |")
        parts.append("")

    return "\n".join(parts)


def _build_checklist(
    bucket_a: list[dict],
    bucket_b: list[dict],
    constraints: dict,
) -> str:
    items = []
    idx = 1

    # Image inputs
    img_nodes = [n for n in bucket_a if n["class_type"] in ("LoadImage", "LoadImageMask")]
    if img_nodes:
        ids_str = ", ".join(f"`{n['id']}`" for n in img_nodes)
        items.append(f"{idx}. Image inputs uploaded and patched into nodes {ids_str}.")
        idx += 1

    # Video inputs
    vid_nodes = [n for n in bucket_a if n["class_type"] in ("VHS_LoadVideo", "VHS_LoadImages")]
    if vid_nodes:
        ids_str = ", ".join(f"`{n['id']}`" for n in vid_nodes)
        items.append(f"{idx}. Video inputs uploaded and patched into nodes {ids_str}.")
        idx += 1

    # Prompt nodes
    prompt_nodes = [n for n in bucket_a if "TextEncode" in n["class_type"]]
    if prompt_nodes:
        items.append(f"{idx}. Positive and negative prompt nodes patched with user brief text.")
        idx += 1

    # LTXV
    if constraints["ltxv"]:
        items.append(f"{idx}. Length value satisfies 8n + 1 constraint (e.g. 49, 73, 97, 121, 161).")
        idx += 1

    # Audio
    if constraints["has_audio"]:
        items.append(f"{idx}. Positive prompt contains a `Music: <style>` line.")
        idx += 1

    # CFG distilled
    if constraints["cfg_is_one"]:
        items.append(f"{idx}. CFG input not modified (distilled model requires CFG=1).")
        idx += 1

    # LoRA
    if constraints["lora_files"]:
        items.append(f"{idx}. LoRA files verified via `check_model`: {', '.join(f'`{f}`' for f in constraints['lora_files'])}.")
        idx += 1

    # Model loaders
    if bucket_b:
        ml_ids = ", ".join(f"`{n['id']}`" for n in bucket_b)
        items.append(f"{idx}. Model paths resolved via `check_model` and patched into loader nodes {ml_ids}.")
        idx += 1
    else:
        items.append(f"{idx}. Model paths resolved via `check_model` and patched into loader nodes.")
        idx += 1

    return "\n".join(items)


def _build_patch_bundle(bucket_a: list[dict], constraints: dict) -> str:
    bundle = []
    for n in bucket_a:
        for inp in n["patchable"]:
            val = n["inputs"].get(inp)
            # Use placeholder if value is a link (list) or None
            if val is None or isinstance(val, list):
                val_json = '"<value>"'
            elif isinstance(val, str):
                val_json = json.dumps(val)
            else:
                val_json = json.dumps(val)
            bundle.append(f'  {{"node_id": "{n["id"]}", "input_name": "{inp}", "value": {val_json}}}')

    if not bundle:
        return '[]'
    return "[\n" + ",\n".join(bundle) + "\n]"


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_skill(content: str, skill_name: str) -> list[str]:
    """Run structural checks on the generated SKILL.md. Returns list of error strings."""
    errors: list[str] = []

    # 1. Split frontmatter
    if not content.startswith("---"):
        errors.append("Frontmatter missing: content must start with '---'.")
        return errors

    parts = content.split("---", 2)
    if len(parts) < 3:
        errors.append("Frontmatter block not properly closed with second '---'.")
        return errors

    fm_raw = parts[1].strip()
    body = parts[2]

    # 2. Frontmatter parses as YAML
    try:
        import yaml  # PyYAML is available (pyyaml is in requirements)
        fm: dict = yaml.safe_load(fm_raw) or {}
    except Exception as exc:
        errors.append(f"Frontmatter YAML parse error: {exc}")
        fm = {}

    # 3. name == skill_name
    if fm.get("name") != skill_name:
        errors.append(f"name mismatch: frontmatter has '{fm.get('name')}', expected '{skill_name}'.")

    # 4. description present and non-empty
    desc = fm.get("description", "")
    if not desc or not str(desc).strip():
        errors.append("description is missing or empty in frontmatter.")

    # 5. allowed-tools has no commas
    at = str(fm.get("allowed-tools", ""))
    if "," in at:
        errors.append("allowed-tools must be space-delimited (no commas).")

    # 6. name matches regex
    name_val = str(fm.get("name", ""))
    if not re.match(r'^[a-z0-9][a-z0-9-]{0,62}[a-z0-9]$', name_val):
        errors.append(f"name '{name_val}' does not match ^[a-z0-9][a-z0-9-]{{0,62}}[a-z0-9]$.")

    # 7. All 6 section headings present
    for heading in _SECTION_HEADINGS:
        if heading not in body:
            errors.append(f"Required section heading missing: '{heading}'.")

    # 8. At least one markdown table in patchable section
    patchable_idx = body.find("## Patchable nodes")
    do_not_touch_idx = body.find("## DO NOT TOUCH")
    if patchable_idx != -1 and do_not_touch_idx != -1:
        patchable_body = body[patchable_idx:do_not_touch_idx]
        if "|" not in patchable_body:
            errors.append("No markdown table found in the Patchable nodes section.")
    else:
        errors.append("Cannot locate Patchable nodes section boundaries for table check.")

    # 9. JSON code block present in patch bundle section
    bundle_idx = body.find("## Common patch bundle")
    ts_idx = body.find("## Troubleshooting")
    if bundle_idx != -1 and ts_idx != -1:
        bundle_body = body[bundle_idx:ts_idx]
        if "```json" not in bundle_body:
            errors.append("No ```json code block found in the Common patch bundle section.")
    else:
        errors.append("Cannot locate Common patch bundle section boundaries for JSON check.")

    return errors


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_skill_from_workflow(
    workflow_path: str,
    skill_name: str | None = None,
) -> str:
    """Parse *workflow_path* and write a SKILL.md to skills/<skill_name>/SKILL.md.

    Parameters
    ----------
    workflow_path:
        Path to a ComfyUI API-format workflow JSON file.
    skill_name:
        Kebab-case name for the skill directory.
        If omitted, derived from the workflow filename stem.

    Returns
    -------
    str
        Absolute path to the written SKILL.md.

    Raises
    ------
    ValueError
        If validate_skill() finds errors (file is NOT written).
    FileNotFoundError
        If the workflow JSON does not exist.
    """
    wf_path = Path(workflow_path)
    if not wf_path.exists():
        raise FileNotFoundError(f"Workflow not found: {wf_path}")

    # Derive skill name from filename if not given
    if not skill_name:
        stem = wf_path.stem.lower().replace("_", "-")
        skill_name = stem[:64]

    # Load workflow
    with open(wf_path, encoding="utf-8") as fh:
        workflow: dict = json.load(fh)

    # Look up exact template name + known description from workflow_templates.json
    template_name, known_description = _lookup_template(wf_path)

    # Classify nodes
    bucket_a, bucket_b, bucket_c = _classify_nodes(workflow)

    # Build reverse-dep map and downstream notes
    rev_deps = _build_reverse_deps(workflow)
    downstream_notes = _apply_downstream_notes(bucket_a, bucket_c, rev_deps)

    # Infer constraints
    constraints = _infer_constraints(workflow, bucket_a, bucket_b, bucket_c)

    # Build analysis summary
    analysis = _build_analysis(workflow, bucket_a, bucket_b, bucket_c, constraints, rev_deps, downstream_notes)
    analysis["template_name"] = template_name
    analysis["known_description"] = known_description

    # LLM call (one-shot prose)
    llm_prose = _call_llm_prose(analysis, skill_name)

    # Assemble SKILL.md
    content = _build_skill_md(
        skill_name=skill_name,
        workflow=workflow,
        bucket_a=bucket_a,
        bucket_b=bucket_b,
        bucket_c=bucket_c,
        constraints=constraints,
        downstream_notes=downstream_notes,
        analysis=analysis,
        llm_prose=llm_prose,
    )

    # Validate before writing
    errors = validate_skill(content, skill_name)
    if errors:
        raise ValueError("SKILL.md validation failed:\n" + "\n".join(f"  • {e}" for e in errors))

    # Write to disk
    skill_dir = SKILLS_DIR / skill_name
    skill_dir.mkdir(parents=True, exist_ok=True)
    skill_md = skill_dir / "SKILL.md"
    skill_md.write_text(content, encoding="utf-8")

    return str(skill_md)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a SKILL.md from a ComfyUI API-format workflow JSON.",
    )
    parser.add_argument("workflow", help="Path to the ComfyUI API workflow JSON.")
    parser.add_argument("--name", help="Skill name (kebab-case). Derived from filename if omitted.")
    parser.add_argument("--dry-run", action="store_true", help="Parse and validate but do not write to disk.")
    args = parser.parse_args()

    wf_path = Path(args.workflow)
    if not wf_path.exists():
        print(f"Error: workflow not found: {wf_path}", file=sys.stderr)
        sys.exit(1)

    # Derive skill name
    skill_name = args.name
    if not skill_name:
        skill_name = wf_path.stem.lower().replace("_", "-")[:64]

    print(f"[build_skill] Parsing workflow: {wf_path}")
    print(f"[build_skill] Skill name     : {skill_name}")

    with open(wf_path, encoding="utf-8") as fh:
        workflow: dict = json.load(fh)

    template_name, known_description = _lookup_template(wf_path)
    bucket_a, bucket_b, bucket_c = _classify_nodes(workflow)
    rev_deps = _build_reverse_deps(workflow)
    downstream_notes = _apply_downstream_notes(bucket_a, bucket_c, rev_deps)
    constraints = _infer_constraints(workflow, bucket_a, bucket_b, bucket_c)
    analysis = _build_analysis(workflow, bucket_a, bucket_b, bucket_c, constraints, rev_deps, downstream_notes)
    analysis["template_name"] = template_name
    analysis["known_description"] = known_description

    if known_description:
        print(f"[build_skill] Template found in workflow_templates.json: '{template_name}'")
    else:
        print(f"[build_skill] Template not in workflow_templates.json — using filename stem: '{template_name}'")

    _llm_setting = _settings_get("llm", "pipeline", "build_skill", default="ollama,gemma4:26b")
    print(f"[build_skill] Calling LLM for prose (one-shot) — {_llm_setting}")
    llm_prose = _call_llm_prose(analysis, skill_name)

    content = _build_skill_md(
        skill_name=skill_name,
        workflow=workflow,
        bucket_a=bucket_a,
        bucket_b=bucket_b,
        bucket_c=bucket_c,
        constraints=constraints,
        downstream_notes=downstream_notes,
        analysis=analysis,
        llm_prose=llm_prose,
    )

    # Validate
    errors = validate_skill(content, skill_name)
    if errors:
        print("\nValidation errors:", file=sys.stderr)
        for e in errors:
            print(f"  • {e}", file=sys.stderr)
        sys.exit(1)

    if args.dry_run:
        print("[build_skill] --dry-run: validation passed, not writing to disk.")
        # Use stdout.buffer to avoid codec errors on Windows terminals.
        sys.stdout.buffer.write(content.encode("utf-8"))
        sys.stdout.buffer.write(b"\n")
        return

    skill_dir = SKILLS_DIR / skill_name
    skill_dir.mkdir(parents=True, exist_ok=True)
    skill_md = skill_dir / "SKILL.md"
    skill_md.write_text(content, encoding="utf-8")

    rel_path = skill_md.relative_to(PROJECT_ROOT)
    _llm_label = _settings_get("llm", "pipeline", "build_skill", default="ollama,gemma4:26b")
    sys.stdout.buffer.write(f"\n\u2713  {rel_path}\n".encode("utf-8"))
    sys.stdout.buffer.write(f"   Patchable : {len(bucket_a)} nodes\n".encode("utf-8"))
    sys.stdout.buffer.write(f"   Locked    : {len(bucket_c)} nodes\n".encode("utf-8"))
    sys.stdout.buffer.write(f"   Models    : {len(bucket_b)} (check_model required)\n".encode("utf-8"))
    sys.stdout.buffer.write(f"   LLM call  : 1 ({_llm_label}, prose only)\n".encode("utf-8"))


if __name__ == "__main__":
    main()
