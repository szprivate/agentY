"""
Hugging Face integration tools for agentY.

Provides @tool-decorated functions for discovering and downloading models
from the Hugging Face Hub via its HTTP API.

Environment variables:
    HF_TOKEN            – Hugging Face access token (required for gated models)
    COMFYUI_MODELS_DIR  – Base directory where ComfyUI stores models
                          (falls back to config/settings.json → comfyui_models_dir,
                           then to the sensible default D:/AI/ComfyUI/models)
"""

import json
import logging
import os
from pathlib import Path
from typing import Optional

import requests
from strands import tool

logger = logging.getLogger(__name__)

HF_API_BASE = "https://huggingface.co/api/models"

# Subfolders to scan when checking for local models
LOCAL_MODEL_FOLDERS = [
    "FLUX1",
    "FLUX2",
    "WAN21",
    "WAN22",
    "MISC",
    "SD15",
    "SDXL",
    "LoRA",
    "QWEN",
    "ICLight",
    "Flux-Dev",
    "WAN",
]


def _hf_headers() -> dict:
    """Return request headers including HF auth token if available."""
    headers = {"Accept": "application/json"}
    token = os.environ.get("HF_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def _models_base_dir() -> Path:
    """Resolve the ComfyUI models base directory.

    Priority:
    1. COMFYUI_MODELS_DIR env var
    2. comfyui_models_dir key in config/settings.json
    3. Default: D:/AI/ComfyUI/models
    """
    env_dir = os.environ.get("COMFYUI_MODELS_DIR")
    if env_dir:
        return Path(env_dir)

    config_path = Path(__file__).parent.parent.parent / "config" / "settings.json"
    if config_path.exists():
        try:
            with open(config_path, encoding="utf-8") as f:
                config = json.load(f)
            d = config.get("comfyui_models_dir")
            if d:
                return Path(d)
        except Exception:
            pass

    return Path("D:/AI/ComfyUI/models")


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------


@tool
def search_huggingface_models(
    query: str,
    filter_tag: str = "",
    limit: int = 10,
) -> str:
    """Search the Hugging Face Hub for models matching a text query.

    Args:
        query:      Free-text search string (e.g. "flux lora", "wan2.1 video").
        filter_tag: (Optional) Pipeline or library tag to filter by
                    (e.g. "diffusers", "flux", "wan", "text-to-image").
        limit:      Maximum number of results to return (default 10, max 50).

    Returns:
        JSON array of matching models, each containing model_id, downloads,
        likes, tags, last_modified, and pipeline_tag.
    """
    try:
        params: dict = {
            "search": query,
            "limit": min(limit, 50),
            "sort": "downloads",
            "direction": "-1",
        }
        if filter_tag:
            params["filter"] = filter_tag

        resp = requests.get(
            HF_API_BASE,
            headers=_hf_headers(),
            params=params,
            timeout=30,
        )
        resp.raise_for_status()
        models = resp.json()

        results = []
        for m in models:
            results.append({
                "model_id": m.get("modelId") or m.get("id", ""),
                "downloads": m.get("downloads", 0),
                "likes": m.get("likes", 0),
                "pipeline_tag": m.get("pipeline_tag", ""),
                "tags": m.get("tags", []),
                "last_modified": m.get("lastModified", ""),
            })

        return json.dumps({"ok": True, "count": len(results), "models": results})
    except requests.HTTPError as exc:
        logger.error("HF API HTTP error in search: %s", exc)
        return json.dumps({"ok": False, "error": f"HTTP {exc.response.status_code}: {exc.response.text[:300]}"})
    except Exception as exc:
        logger.error("Error in search_huggingface_models: %s", exc, exc_info=True)
        return json.dumps({"ok": False, "error": str(exc)})


@tool
def get_model_info(model_id: str) -> str:
    """Fetch full metadata for a specific Hugging Face model.

    Use this to inspect a model's file list (siblings), gated status, license,
    and tags before deciding which file to download.

    Args:
        model_id: The Hugging Face model identifier (e.g. "black-forest-labs/FLUX.1-dev").

    Returns:
        JSON object with model metadata including id, tags, license, gated status,
        pipeline_tag, card_data summary, and a files array listing every file in
        the repo with name and size.
    """
    try:
        url = f"{HF_API_BASE}/{model_id}"
        resp = requests.get(url, headers=_hf_headers(), timeout=30)
        resp.raise_for_status()
        data = resp.json()

        # Extract the file listing from siblings
        files = []
        for s in data.get("siblings", []):
            files.append({
                "filename": s.get("rfilename", ""),
                "size": s.get("size"),
            })

        result = {
            "ok": True,
            "model_id": data.get("modelId") or data.get("id", model_id),
            "pipeline_tag": data.get("pipeline_tag", ""),
            "tags": data.get("tags", []),
            "license": data.get("cardData", {}).get("license", "unknown") if isinstance(data.get("cardData"), dict) else "unknown",
            "gated": data.get("gated", False),
            "downloads": data.get("downloads", 0),
            "likes": data.get("likes", 0),
            "last_modified": data.get("lastModified", ""),
            "files": files,
        }
        return json.dumps(result)
    except requests.HTTPError as exc:
        logger.error("HF API HTTP error in get_model_info: %s", exc)
        return json.dumps({"ok": False, "error": f"HTTP {exc.response.status_code}: {exc.response.text[:300]}"})
    except Exception as exc:
        logger.error("Error in get_model_info: %s", exc, exc_info=True)
        return json.dumps({"ok": False, "error": str(exc)})


@tool
def check_local_model(filename: str) -> str:
    """Check whether a model file already exists in the local model folders.

    Scans the known ComfyUI model subdirectories (FLUX1, FLUX2, WAN21, WAN22,
    MISC, SD15, SDXL, LoRA, QWEN, ICLight, Flux-Dev, WAN) under the models
    base directory.

    ALWAYS call this before attempting any download.

    Args:
        filename: The filename to look for (e.g. "flux1-dev-fp8.safetensors").
                  Can also be a relative path like "FLUX1/flux1-dev-fp8.safetensors".

    Returns:
        JSON object with found=True and full path, or found=False.
    """
    try:
        base = _models_base_dir()
        target = Path(filename)

        # If the filename includes a subfolder prefix, check as a direct path
        direct = base / target
        if direct.exists():
            return json.dumps({
                "ok": True,
                "found": True,
                "path": str(direct),
                "size_mb": round(direct.stat().st_size / (1024 * 1024), 2),
            })

        # Otherwise scan each known subfolder for the bare filename
        bare_name = target.name
        for folder_name in LOCAL_MODEL_FOLDERS:
            folder = base / folder_name
            if not folder.is_dir():
                continue
            candidate = folder / bare_name
            if candidate.exists():
                return json.dumps({
                    "ok": True,
                    "found": True,
                    "path": str(candidate),
                    "size_mb": round(candidate.stat().st_size / (1024 * 1024), 2),
                })

        # Also do a recursive search as a last resort
        if base.is_dir():
            for match in base.rglob(bare_name):
                if match.is_file():
                    return json.dumps({
                        "ok": True,
                        "found": True,
                        "path": str(match),
                        "size_mb": round(match.stat().st_size / (1024 * 1024), 2),
                    })

        return json.dumps({"ok": True, "found": False, "searched_base": str(base)})
    except Exception as exc:
        logger.error("Error in check_local_model: %s", exc, exc_info=True)
        return json.dumps({"ok": False, "error": str(exc)})


@tool
def download_hf_model(
    model_id: str,
    filename: str,
    destination_folder: str,
    subfolder: str = "",
) -> str:
    """Download a specific file from a Hugging Face model repository.

    Streams the download in chunks and reports progress.  Only call this
    AFTER check_local_model has confirmed the file does not exist locally.

    Args:
        model_id:           HF model identifier (e.g. "black-forest-labs/FLUX.1-dev").
        filename:           Name of the file to download (e.g. "flux1-dev.safetensors").
        destination_folder: Local folder name under the models base dir to save to
                            (e.g. "FLUX1", "WAN21", "MISC").
        subfolder:          (Optional) Subfolder within the HF repo where the file
                            lives (e.g. "transformer", "vae").  If empty, the file
                            is assumed to be at the repo root.

    Returns:
        JSON object with ok=True and the full local path, or an error.
    """
    try:
        base = _models_base_dir()
        dest_dir = base / destination_folder
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest_path = dest_dir / filename

        # Safety: don't re-download if already present
        if dest_path.exists():
            return json.dumps({
                "ok": True,
                "path": str(dest_path),
                "message": "File already exists — skipping download.",
                "size_mb": round(dest_path.stat().st_size / (1024 * 1024), 2),
            })

        # Build the download URL
        if subfolder:
            url = f"https://huggingface.co/{model_id}/resolve/main/{subfolder}/{filename}"
        else:
            url = f"https://huggingface.co/{model_id}/resolve/main/{filename}"

        headers = {}
        token = os.environ.get("HF_TOKEN")
        if token:
            headers["Authorization"] = f"Bearer {token}"

        logger.info("Downloading %s from %s …", filename, model_id)

        resp = requests.get(url, headers=headers, stream=True, timeout=60)
        resp.raise_for_status()

        total_size = int(resp.headers.get("content-length", 0))
        downloaded = 0
        chunk_size = 8 * 1024 * 1024  # 8 MB chunks

        # Write to a temp file first, rename on completion
        tmp_path = dest_path.with_suffix(dest_path.suffix + ".downloading")
        try:
            with open(tmp_path, "wb") as f:
                for chunk in resp.iter_content(chunk_size=chunk_size):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total_size > 0:
                            pct = (downloaded / total_size) * 100
                            if downloaded % (64 * 1024 * 1024) < chunk_size:
                                logger.info(
                                    "  %.1f%% (%d / %d MB)",
                                    pct,
                                    downloaded // (1024 * 1024),
                                    total_size // (1024 * 1024),
                                )

            # Rename temp → final
            tmp_path.rename(dest_path)
        except Exception:
            # Clean up partial file on error
            if tmp_path.exists():
                tmp_path.unlink()
            raise

        size_mb = round(dest_path.stat().st_size / (1024 * 1024), 2)
        return json.dumps({
            "ok": True,
            "path": str(dest_path),
            "size_mb": size_mb,
            "message": f"Downloaded {filename} ({size_mb} MB) to {destination_folder}/",
        })
    except requests.HTTPError as exc:
        status = exc.response.status_code
        body = exc.response.text[:400]
        logger.error("HF download HTTP error: %s %s", status, body)
        if status == 401:
            hint = " — Is HF_TOKEN set and authorised for this gated model?"
        elif status == 403:
            hint = " — Access denied. You may need to accept the model's license on HF."
        elif status == 404:
            hint = " — File not found. Check model_id, subfolder, and filename."
        else:
            hint = ""
        return json.dumps({"ok": False, "error": f"HTTP {status}{hint}: {body}"})
    except Exception as exc:
        logger.error("Error in download_hf_model: %s", exc, exc_info=True)
        return json.dumps({"ok": False, "error": str(exc)})
