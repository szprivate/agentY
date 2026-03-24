"""ComfyUI integration via the MCP (Model Context Protocol) endpoint.

Provides :class:`ComfyUIClient` which encapsulates:

* Loading and patching ComfyUI workflow JSON templates.
* Communicating with ComfyUI through its MCP JSON-RPC interface.
* Extracting output-image URLs from workflow results.

Typical usage::

    client = ComfyUIClient(config)
    workflow = client.prepare_workflow("wf.json", prompt, images)
    result   = client.run_workflow(workflow, workflow_id="my_wf")
    image    = client.extract_output_image(result)
"""

import json
import os
from typing import Any

import requests

from config import AppConfig, IMAGE_EXTENSIONS

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Default HTTP headers for MCP JSON-RPC requests.
_MCP_HEADERS: dict[str, str] = {
    "Content-Type": "application/json",
    "Accept": "application/json, text/event-stream",
}

#: Timeout (seconds) for MCP requests to ComfyUI.
_MCP_TIMEOUT: int = 300


# ---------------------------------------------------------------------------
# ComfyUIClient
# ---------------------------------------------------------------------------


class ComfyUIClient:
    """High-level client for ComfyUI's MCP endpoint.

    Bundles workflow template manipulation and MCP communication into a
    single, testable object.

    Args:
        config: The application configuration (supplies URLs, paths, etc.).
    """

    def __init__(self, config: AppConfig) -> None:
        self._config = config

    # ── Workflow template helpers ──────────────────────────────────────────

    @staticmethod
    def load_template(path: str) -> dict[str, Any]:
        """Deserialise a ComfyUI workflow JSON file."""
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)

    @staticmethod
    def _inject_prompts(obj: Any, prompt: str) -> None:
        """Walk *obj* recursively, replacing every ``"prompt"`` value."""
        if isinstance(obj, dict):
            for key in obj:
                if key == "prompt":
                    obj[key] = prompt
                else:
                    ComfyUIClient._inject_prompts(obj[key], prompt)
        elif isinstance(obj, list):
            for item in obj:
                ComfyUIClient._inject_prompts(item, prompt)

    @staticmethod
    def _inject_images(obj: Any, images: list[str]) -> None:
        """Walk *obj* recursively, replacing ``"image_path"`` fields.

        Images are consumed from the front of *images* in the order the
        fields are encountered; the list is **mutated** in place.
        """
        if isinstance(obj, dict):
            for key in obj:
                if key == "image_path" and images:
                    obj[key] = images.pop(0)
                else:
                    ComfyUIClient._inject_images(obj[key], images)
        elif isinstance(obj, list):
            for item in obj:
                ComfyUIClient._inject_images(item, images)

    def prepare_workflow(
        self,
        workflow_file: str,
        prompt: str,
        input_images: list[str],
    ) -> dict[str, Any]:
        """Load a workflow template, inject the prompt and images, return it.

        Args:
            workflow_file: Path to the ``.json`` workflow template.
            prompt: Creative prompt to write into every ``"prompt"`` field.
            input_images: Image paths written into ``"image_path"`` fields.

        Returns:
            A fully-patched workflow dict ready for :meth:`run_workflow`.
        """
        workflow = self.load_template(workflow_file)
        self._inject_prompts(workflow, prompt)
        self._inject_images(workflow, list(input_images))  # copy to avoid mutation
        return workflow

    # ── MCP communication ─────────────────────────────────────────────────

    @staticmethod
    def _parse_sse(text: str) -> dict[str, Any]:
        """Extract the first valid JSON payload from an SSE text stream."""
        for line in text.replace("\r\n", "\n").split("\n"):
            stripped = line.strip()
            if stripped.startswith("data: "):
                try:
                    return json.loads(stripped[6:])
                except json.JSONDecodeError:
                    continue
        raise ValueError("No valid JSON data found in MCP SSE response.")

    def _rpc(
        self,
        method: str,
        params: dict[str, Any],
        request_id: int = 1,
    ) -> dict[str, Any]:
        """Send a JSON-RPC 2.0 request to the MCP endpoint.

        Transparently handles both ``application/json`` and
        ``text/event-stream`` response content types.
        """
        response = requests.post(
            self._config.comfyui_mcp_url,
            json={
                "jsonrpc": "2.0",
                "id": request_id,
                "method": method,
                "params": params,
            },
            headers=_MCP_HEADERS,
            timeout=_MCP_TIMEOUT,
        )
        response.raise_for_status()

        if "text/event-stream" in response.headers.get("content-type", ""):
            return self._parse_sse(response.text)
        return response.json()

    def _call_tool(
        self,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> dict[str, Any]:
        """Invoke an MCP tool and unwrap its result payload.

        Raises:
            RuntimeError: On an MCP-level error or an unrecognised format.
        """
        resp = self._rpc(
            "tools/call",
            {"name": tool_name, "arguments": arguments},
            request_id=2,
        )

        # Surface MCP errors immediately
        if "error" in resp:
            raise RuntimeError(json.dumps(resp["error"], ensure_ascii=False))

        result = resp.get("result", {})
        content = result.get("content")

        # Standard MCP content envelope: [{type, text}, ...]
        if isinstance(content, list) and content:
            first = content[0]
            if isinstance(first, dict) and "text" in first:
                text_value = first["text"]
                if isinstance(text_value, dict):
                    return text_value
                try:
                    return json.loads(text_value)
                except (json.JSONDecodeError, TypeError):
                    return {"message": text_value}

        if isinstance(result, dict):
            return result

        raise RuntimeError("Unexpected MCP response format.")

    # ── Workflow execution ────────────────────────────────────────────────

    def run_workflow(
        self,
        workflow: dict[str, Any],
        workflow_id: str,
    ) -> dict[str, Any]:
        """Submit a prepared workflow to ComfyUI and return the raw result."""
        return self._call_tool(
            "run_raw_workflow",
            {
                "workflow": workflow,
                "workflow_id": workflow_id,
                "return_inline_preview": False,
            },
        )

    # ── Output extraction ─────────────────────────────────────────────────

    def extract_output_image(self, result: dict[str, Any]) -> str | None:
        """Locate the output image URL/path inside a ComfyUI result.

        Three strategies are tried, in order:

        1. **Top-level URL** — ``asset_url`` or ``image_url`` field.
        2. **Outputs map** — walks ``result.outputs.<node>.images`` and
           constructs a ComfyUI ``/view`` URL from the filename metadata.
        3. **Deep string scan** — traverses the entire result tree looking
           for any string that ends with a known image extension.

        Returns:
            The image URL/path, or ``None`` if nothing was found.
        """
        # Strategy 1: explicit URL fields
        asset_url = result.get("asset_url") or result.get("image_url")
        if isinstance(asset_url, str) and asset_url:
            return asset_url

        # Strategy 2: outputs → images → filename
        outputs = result.get("outputs") if isinstance(result, dict) else None
        if isinstance(outputs, dict):
            for node_output in outputs.values():
                if not isinstance(node_output, dict):
                    continue
                for image in node_output.get("images", []):
                    if not isinstance(image, dict):
                        continue
                    filename = image.get("filename")
                    if not filename:
                        continue
                    subfolder = image.get("subfolder", "")
                    folder_type = image.get("type", "output")
                    base = self._config.comfyui_url
                    return (
                        f"{base}/view?filename={filename}"
                        f"&subfolder={subfolder}&type={folder_type}"
                    )

        # Strategy 3: depth-first search for image file paths
        stack: list[Any] = [result]
        while stack:
            current = stack.pop()
            if isinstance(current, dict):
                stack.extend(current.values())
            elif isinstance(current, list):
                stack.extend(current)
            elif (
                isinstance(current, str)
                and current.lower().endswith(IMAGE_EXTENSIONS)
            ):
                return current

        return None
