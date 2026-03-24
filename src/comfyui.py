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
import re
import time
from typing import Any
from urllib.parse import parse_qs, urlparse

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
_URL_RE = re.compile(r"https?://[^\s)\]]+")
_IMAGE_NAME_RE = re.compile(r"[A-Za-z0-9._-]+\.(?:png|jpg|jpeg|webp)", re.IGNORECASE)


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
        """Walk *obj* recursively, replacing prompt-bearing fields.

        Supports both the simplified API-style workflow format used in the
        local workspace and exported ComfyUI graph JSON files.
        """
        if isinstance(obj, dict):
            class_type = obj.get("class_type")
            inputs = obj.get("inputs")
            if isinstance(class_type, str) and isinstance(inputs, dict):
                text_value = inputs.get("text")
                node_title = str((obj.get("_meta") or {}).get("title") or "").lower()
                if (
                    class_type in {"CLIPTextEncode", "Text Multiline"}
                    and isinstance(text_value, str)
                    and "negative" not in node_title
                ):
                    inputs["text"] = prompt

            node_inputs = obj.get("inputs")
            widgets_values = obj.get("widgets_values")
            if isinstance(node_inputs, list) and isinstance(widgets_values, list):
                for index, node_input in enumerate(node_inputs):
                    if not isinstance(node_input, dict):
                        continue
                    label = str(node_input.get("label") or "").lower()
                    name = str(node_input.get("name") or "").lower()
                    widget = node_input.get("widget") or {}
                    widget_name = str(widget.get("name") or "").lower()
                    input_type = str(node_input.get("type") or "").upper()
                    if input_type != "STRING":
                        continue
                    if (
                        label == "prompt"
                        or name in {"prompt", "text"}
                        or widget_name in {"prompt", "text"}
                    ) and index < len(widgets_values):
                        widgets_values[index] = prompt

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

    @staticmethod
    def _iter_text_values(value: Any) -> list[str]:
        """Collect all string values nested inside an arbitrary object."""
        collected: list[str] = []
        stack: list[Any] = [value]
        while stack:
            current = stack.pop()
            if isinstance(current, dict):
                stack.extend(current.values())
            elif isinstance(current, list):
                stack.extend(current)
            elif isinstance(current, str):
                collected.append(current)
        return collected

    def _extract_output_references_from_text(self, value: Any) -> list[str]:
        """Extract output image references from free-form result text."""
        references: list[str] = []
        seen: set[str] = set()

        for text in self._iter_text_values(value):
            for url in _URL_RE.findall(text):
                parsed = urlparse(url)
                params = parse_qs(parsed.query)
                filename = (params.get("filename") or [""])[0]
                subfolder = (params.get("subfolder") or [""])[0]

                if filename and filename.lower().endswith(IMAGE_EXTENSIONS):
                    resolved = self._config.resolve_comfyui_output_file(
                        filename,
                        subfolder,
                    )
                    if resolved not in seen:
                        seen.add(resolved)
                        references.append(resolved)
                    continue

                if url.lower().endswith(IMAGE_EXTENSIONS) and url not in seen:
                    seen.add(url)
                    references.append(url)

            for filename in _IMAGE_NAME_RE.findall(text):
                resolved = self._config.resolve_comfyui_output_file(filename)
                if resolved not in seen:
                    seen.add(resolved)
                    references.append(resolved)

        return references

    def list_output_files_on_disk(self) -> list[str]:
        """Return all image files currently present in configured output dirs."""
        discovered: list[str] = []
        for output_dir in self._config.list_comfyui_output_dirs():
            if not os.path.isdir(output_dir):
                continue
            for root, _, file_names in os.walk(output_dir):
                for file_name in file_names:
                    if not file_name.lower().endswith(IMAGE_EXTENSIONS):
                        continue
                    discovered.append(os.path.normpath(os.path.join(root, file_name)))
        return sorted(discovered)

    def find_recent_output_files(
        self,
        started_at: float,
        known_files: set[str] | None = None,
    ) -> list[str]:
        """Return newly created or recently modified output files on disk.

        This is used as a robust fallback when the MCP response does not expose
        enough metadata to reconstruct the generated output path.
        """
        recent_files: list[tuple[float, str]] = []
        tolerance_seconds = 2.0
        known_files = known_files or set()

        for file_path in self.list_output_files_on_disk():
            if file_path in known_files:
                continue
            try:
                modified_at = os.path.getmtime(file_path)
            except OSError:
                continue
            if modified_at >= started_at - tolerance_seconds:
                recent_files.append((modified_at, file_path))

        recent_files.sort(key=lambda item: (item[0], item[1]))
        return [file_path for _, file_path in recent_files]

    # ── Output extraction ─────────────────────────────────────────────────

    def extract_output_files(self, result: dict[str, Any]) -> list[str]:
        """Locate generated image file paths inside a ComfyUI result."""
        files: list[str] = []

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
                    files.append(
                        self._config.resolve_comfyui_output_file(filename, subfolder)
                    )

        files.extend(self._extract_output_references_from_text(result))

        deduplicated: list[str] = []
        seen: set[str] = set()
        for file_path in files:
            if file_path not in seen:
                seen.add(file_path)
                deduplicated.append(file_path)
        return deduplicated

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

        text_references = self._extract_output_references_from_text(result)
        if text_references:
            return text_references[0]

        # Strategy 4: fall back to the first resolved generated output file.
        output_files = self.extract_output_files(result)
        if output_files:
            return output_files[0]

        return None
