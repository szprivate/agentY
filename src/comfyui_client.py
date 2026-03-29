"""
HTTP client wrapper for communicating with the ComfyUI server.

Provides a singleton client that handles authentication and base URL configuration.
The ComfyUI API key is read from the API_KEY_COMFY_ORG environment variable.
"""

import json
import os
from pathlib import Path

import requests


class ComfyUIClient:
    """HTTP client for the ComfyUI REST API."""

    def __init__(self, base_url: str | None = None, api_key: str | None = None):
        self.base_url = (base_url or self._load_base_url()).rstrip("/")
        self.api_key = api_key or os.environ.get("API_KEY_COMFY_ORG", "")

    @staticmethod
    def _load_base_url() -> str:
        config_path = Path(__file__).parent.parent / "config" / "settings.json"
        if config_path.exists():
            with open(config_path, encoding="utf-8") as f:
                config = json.load(f)
            return config.get("comfyui_url", "http://127.0.0.1:8188")
        return "http://127.0.0.1:8188"

    def _headers(self) -> dict:
        headers = {"Accept": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def get(
        self,
        path: str,
        params: dict | None = None,
        stream: bool = False,
        raw: bool = False,
    ) -> requests.Response | dict | list | str:
        """Send a GET request. Returns parsed JSON unless raw=True."""
        url = f"{self.base_url}{path}"
        resp = requests.get(
            url, headers=self._headers(), params=params, stream=stream, timeout=120
        )
        resp.raise_for_status()
        if raw or stream:
            return resp
        try:
            return resp.json()
        except ValueError:
            return resp.text

    def post(
        self,
        path: str,
        json_data: dict | None = None,
        data: dict | None = None,
        files: dict | None = None,
    ) -> dict | str:
        """Send a POST request. Returns parsed JSON when possible."""
        url = f"{self.base_url}{path}"
        headers = self._headers()
        if files:
            # Let requests set content-type with boundary for multipart
            headers.pop("Accept", None)
        resp = requests.post(
            url, headers=headers, json=json_data, data=data, files=files, timeout=120
        )
        resp.raise_for_status()
        try:
            return resp.json()
        except ValueError:
            return resp.text

    def delete(self, path: str) -> dict | str:
        """Send a DELETE request."""
        url = f"{self.base_url}{path}"
        resp = requests.delete(url, headers=self._headers(), timeout=120)
        resp.raise_for_status()
        try:
            return resp.json()
        except ValueError:
            return resp.text


# ── Singleton ──────────────────────────────────────────────────────────────────

_client: ComfyUIClient | None = None


def get_client() -> ComfyUIClient:
    """Return (and lazily create) the singleton ComfyUI client."""
    global _client  # noqa: PLW0603
    if _client is None:
        _client = ComfyUIClient()
    return _client
