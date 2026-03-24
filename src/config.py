"""Application configuration for AgentY.

Loads settings from ``config/settings.json``, resolves file paths
relative to the project root, and exposes typed properties for every
setting the rest of the application needs.

Typical usage::

    config = AppConfig()
    print(config.ollama_model_id)
    images = config.collect_mood_images()
"""

import json
import os
from typing import Any

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Absolute path to the project root (one level above ``src/``).
BASE_DIR: str = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

#: Image file extensions recognised by the application.
IMAGE_EXTENSIONS: tuple[str, ...] = (".jpg", ".jpeg", ".png", ".webp")


# ---------------------------------------------------------------------------
# AppConfig
# ---------------------------------------------------------------------------


class AppConfig:
    """Central, read-only configuration object.

    All relative paths found in ``settings.json`` are resolved against
    :data:`BASE_DIR` so that callers never have to worry about working
    directories.

    Args:
        config_path: Absolute path to the JSON configuration file.
                     Defaults to ``<project_root>/config/settings.json``.
    """

    def __init__(self, config_path: str | None = None) -> None:
        path = config_path or os.path.join(BASE_DIR, "config", "settings.json")
        with open(path, "r", encoding="utf-8") as fh:
            self._settings: dict[str, Any] = json.load(fh)

    # -- raw access ---------------------------------------------------------

    def get(self, key: str, default: Any = None) -> Any:
        """Return a raw setting value by its top-level key."""
        return self._settings.get(key, default)

    # -- path helpers -------------------------------------------------------

    @staticmethod
    def resolve_path(path: str | None) -> str | None:
        """Turn a relative path into an absolute one based on the project root.

        * ``None`` / empty → ``None``
        * Already absolute   → returned as-is
        * Relative            → joined with :data:`BASE_DIR` and normalised
        """
        if not path:
            return None
        if os.path.isabs(path):
            return path
        return os.path.normpath(os.path.join(BASE_DIR, path))

    @staticmethod
    def read_text_file(path: str | None) -> str:
        """Read a UTF-8 text file and return its stripped contents.

        Resolves relative paths via :meth:`resolve_path`.
        Returns an empty string when the path is falsy or the file is missing.
        """
        resolved = AppConfig.resolve_path(path)
        if not resolved or not os.path.exists(resolved):
            return ""
        with open(resolved, "r", encoding="utf-8") as fh:
            return fh.read().strip()

    # -- typed properties ---------------------------------------------------

    @property
    def ollama_model_id(self) -> str:
        """Ollama model identifier (e.g. ``qwen3-vl:30b``).

        Raises:
            RuntimeError: If the key is missing from settings.
        """
        model = self._settings.get("ollama-model")
        if not model:
            raise RuntimeError("Missing 'ollama-model' in config/settings.json.")
        return model

    @property
    def ollama_api_url(self) -> str:
        """OpenAI-compatible API base URL, guaranteed to end in ``/v1``."""
        url = (
            self._settings.get("ollama_api_url") or "http://localhost:11434/v1"
        ).rstrip("/")
        return url if url.endswith("/v1") else f"{url}/v1"

    @property
    def ollama_native_url(self) -> str:
        """Base URL for Ollama's native REST API (without ``/v1``)."""
        url = self.ollama_api_url
        return url[:-3] if url.endswith("/v1") else url

    @property
    def comfyui_url(self) -> str:
        """ComfyUI web-interface base URL (no trailing slash)."""
        return (
            self._settings.get("comfyui_url") or "http://127.0.0.1:8188/"
        ).rstrip("/")

    @property
    def comfyui_mcp_url(self) -> str:
        """URL for the ComfyUI MCP JSON-RPC endpoint."""
        return (
            self._settings.get("comfyui_mcp_url") or "http://127.0.0.1:9000/mcp"
        ).rstrip("/")

    @property
    def max_iterations(self) -> int:
        """Maximum prompt→generate→review iterations before giving up."""
        return int(self._settings.get("max_iter", 3))

    @property
    def ollama_timeout(self) -> int:
        """Timeout (seconds) for Ollama API requests."""
        return int(self._settings.get("ollama_timeout", 120))

    @property
    def briefing_text(self) -> str:
        """Full text of the briefing prompt file (may be empty)."""
        path = self._settings.get("prompts", {}).get("briefing")
        return self.read_text_file(path)

    # -- resource discovery -------------------------------------------------

    def collect_mood_images(self) -> list[str]:
        """Return sorted absolute paths to every image in the mood-images dir."""
        mood_dir = self.resolve_path(
            self._settings.get("mood_images_dir", "./mood_images/")
        )
        if not mood_dir or not os.path.isdir(mood_dir):
            return []
        return [
            os.path.join(mood_dir, name)
            for name in sorted(os.listdir(mood_dir))
            if name.lower().endswith(IMAGE_EXTENSIONS)
        ]

    def select_workflow_file(self) -> str:
        """Choose the ComfyUI workflow JSON file to use.

        Resolution order:

        1. The explicit ``comfyui_workflow`` path from settings (if it exists).
        2. The first ``.json`` file (alphabetically) in the workflows directory.

        Raises:
            RuntimeError: If no usable workflow file can be located.
        """
        # 1. Explicitly configured workflow
        explicit = self.resolve_path(self._settings.get("comfyui_workflow"))
        if explicit and os.path.exists(explicit):
            return explicit

        # 2. First JSON in the workflows directory
        workflows_dir = self.resolve_path(
            self._settings.get("comfyui_workflows_dir", "./comfyui_workflows/")
        )
        if not workflows_dir or not os.path.isdir(workflows_dir):
            raise RuntimeError("Configured workflow directory does not exist.")

        files = sorted(f for f in os.listdir(workflows_dir) if f.endswith(".json"))
        if not files:
            raise RuntimeError("No ComfyUI workflow JSON files found.")

        return os.path.join(workflows_dir, files[0])
