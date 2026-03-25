"""ComfyUI integration via the shawnrushefsky/comfyui-mcp MCP server.

Provides :class:`ComfyUIClient` which encapsulates:

* Managing a comfyui-mcp server subprocess (STDIO MCP transport).
* Calling MCP tools: search_templates, run_workflow, validate_workflow, etc.
* Loading and patching local ComfyUI workflow JSON templates.
* Extracting output-image references from workflow results.

The MCP server is started as a child process (Node.js or Docker) and
communicates over stdin/stdout using the Model Context Protocol.

Typical usage::

    client = ComfyUIClient(config)
    client.start()
    try:
        result = client.run_workflow(workflow_json, sync=True)
        image  = client.extract_output_image(result)
    finally:
        client.stop()
"""

import asyncio
import json
import os
import re
import threading
import time
from typing import Any
from urllib.parse import parse_qs, urlparse

from config import AppConfig, IMAGE_EXTENSIONS

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Timeout (seconds) for individual MCP tool calls.
_MCP_CALL_TIMEOUT: int = 600

#: Timeout (seconds) for the initial MCP server connection.
_MCP_CONNECT_TIMEOUT: int = 60

_URL_RE = re.compile(r"https?://[^\s)\]]+")
_IMAGE_NAME_RE = re.compile(
    r"[A-Za-z0-9._-]+\.(?:png|jpg|jpeg|webp)", re.IGNORECASE
)


# ---------------------------------------------------------------------------
# ComfyUIClient
# ---------------------------------------------------------------------------


class ComfyUIClient:
    """MCP client for the comfyui-mcp server.

    Manages the MCP server subprocess lifecycle and provides synchronous
    wrappers for MCP tool calls.  Also handles local workflow template
    manipulation (loading, prompt injection, parameter overrides).

    Args:
        config: The application configuration (supplies URLs, paths, etc.).
    """

    def __init__(self, config: AppConfig) -> None:
        self._config = config
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._session: Any = None  # mcp.ClientSession
        self._stdio_cm: Any = None  # stdio_client context manager
        self._session_cm: Any = None  # ClientSession context manager
        self._connected: bool = False

    # ── MCP lifecycle ─────────────────────────────────────────────────────

    def start(self) -> None:
        """Start the MCP server subprocess and establish a session."""
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(
            target=self._loop.run_forever,
            daemon=True,
            name="comfyui-mcp-loop",
        )
        self._thread.start()
        self._run_async(self._async_connect(), timeout=_MCP_CONNECT_TIMEOUT)
        self._connected = True

    def stop(self) -> None:
        """Disconnect and terminate the MCP server subprocess."""
        if self._connected:
            try:
                self._run_async(self._async_disconnect(), timeout=30)
            except Exception:
                pass
            self._connected = False
        if self._loop and self._loop.is_running():
            self._loop.call_soon_threadsafe(self._loop.stop)
        if self._thread:
            self._thread.join(timeout=10)
            self._thread = None
        if self._loop:
            self._loop.close()
            self._loop = None

    @property
    def is_connected(self) -> bool:
        return self._connected

    def __enter__(self) -> "ComfyUIClient":
        self.start()
        return self

    def __exit__(self, *exc: Any) -> None:
        self.stop()

    # ── Async internals ───────────────────────────────────────────────────

    def _run_async(self, coro: Any, timeout: int = _MCP_CALL_TIMEOUT) -> Any:
        """Submit a coroutine to the background event loop and block."""
        if not self._loop or not self._loop.is_running():
            raise RuntimeError("MCP event loop is not running.")
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return future.result(timeout=timeout)

    async def _async_connect(self) -> None:
        """Establish the MCP session over STDIO (async)."""
        from mcp import ClientSession
        from mcp.client.stdio import stdio_client

        try:
            from mcp import StdioServerParameters
        except ImportError:
            from mcp.client.stdio import StdioServerParameters

        env = dict(os.environ)
        env["COMFYUI_URL"] = self._config.comfyui_url

        params = StdioServerParameters(
            command=self._config.comfyui_mcp_command,
            args=self._config.comfyui_mcp_args,
            env=env,
        )
        self._stdio_cm = stdio_client(params)
        read, write = await self._stdio_cm.__aenter__()
        self._session_cm = ClientSession(read, write)
        self._session = await self._session_cm.__aenter__()
        await self._session.initialize()

    async def _async_disconnect(self) -> None:
        """Tear down the MCP session (async)."""
        if self._session_cm:
            await self._session_cm.__aexit__(None, None, None)
            self._session_cm = None
            self._session = None
        if self._stdio_cm:
            await self._stdio_cm.__aexit__(None, None, None)
            self._stdio_cm = None

    # ── MCP tool calling ──────────────────────────────────────────────────

    def _call_tool(
        self,
        name: str,
        arguments: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Call an MCP tool synchronously, returning a parsed dict.

        Raises:
            RuntimeError: If the session is not connected or the tool errors.
        """
        if not self._session:
            raise RuntimeError(
                "MCP session not connected. Call start() first."
            )
        result = self._run_async(
            self._session.call_tool(name, arguments or {})
        )
        parsed = self._parse_mcp_result(result)

        # Surface server-side errors
        if isinstance(parsed.get("error"), str):
            raise RuntimeError(
                f"MCP tool '{name}' error: {parsed['error']}"
            )
        return parsed

    @staticmethod
    def _parse_mcp_result(result: Any) -> dict[str, Any]:
        """Convert an MCP ``CallToolResult`` into a plain dict."""
        if hasattr(result, "content") and result.content:
            texts: list[str] = []
            for item in result.content:
                if hasattr(item, "text") and item.text:
                    text = item.text
                    if isinstance(text, dict):
                        return text
                    try:
                        parsed = json.loads(text)
                        if isinstance(parsed, dict):
                            return parsed
                        return {"data": parsed}
                    except (json.JSONDecodeError, TypeError):
                        texts.append(text)
            if texts:
                combined = "\n".join(texts)
                try:
                    return json.loads(combined)
                except (json.JSONDecodeError, TypeError):
                    return {"message": combined}
        if isinstance(result, dict):
            return result
        return {"raw": str(result)}

    # ── MCP tool wrappers: Status & Discovery ─────────────────────────────

    def get_status(self) -> dict[str, Any]:
        """Check ComfyUI connection and installation status."""
        return self._call_tool("get_status")

    def get_capabilities(self) -> dict[str, Any]:
        """Detect capabilities of the connected ComfyUI instance."""
        return self._call_tool("get_capabilities")

    def list_models(self, model_type: str = "all") -> dict[str, Any]:
        """List available models in ComfyUI."""
        return self._call_tool("list_models", {"type": model_type})

    # ── MCP tool wrappers: Templates & Workflows ─────────────────────────

    def search_templates(self, **kwargs: Any) -> dict[str, Any]:
        """Search workflow templates (built-in, examples, and custom).

        Keyword Args:
            modelType: ``"sd15"`` | ``"sdxl"`` | ``"sd3"`` | ``"flux"`` | ``"any"``
            taskType: ``"txt2img"`` | ``"img2img"`` | ``"inpaint"`` | …
            query: Free-text search string.
            category: Filter by category.
        """
        return self._call_tool("search_templates", kwargs)

    def get_template(
        self,
        template_id: str,
        parameters: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Build a workflow from a named template with parameters.

        Args:
            template_id: Template ID returned by :meth:`search_templates`.
            parameters: Values to inject (prompt, model, dimensions, …).
        """
        args: dict[str, Any] = {"templateId": template_id}
        if parameters:
            args["parameters"] = parameters
        return self._call_tool("get_template", args)

    def recommend_workflow(
        self,
        model_name: str,
        task_type: str = "txt2img",
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Get the recommended workflow and settings for a model."""
        args: dict[str, Any] = {
            "modelName": model_name,
            "taskType": task_type,
        }
        args.update(kwargs)
        return self._call_tool("recommend_workflow", args)

    def get_prompting_guide(self, model_type: str = "all") -> dict[str, Any]:
        """Get prompting best practices for a model architecture."""
        return self._call_tool("get_prompting_guide", {"modelType": model_type})

    def save_template(self, **kwargs: Any) -> dict[str, Any]:
        """Save a workflow as a reusable custom template."""
        return self._call_tool("save_template", kwargs)

    def list_examples(self, category: str | None = None) -> dict[str, Any]:
        """List official ComfyUI example workflows."""
        args: dict[str, Any] = {}
        if category:
            args["category"] = category
        return self._call_tool("list_examples", args)

    def get_example_workflow(
        self, name: str, variant: int = 0,
    ) -> dict[str, Any]:
        """Fetch an example workflow by name."""
        return self._call_tool(
            "get_example_workflow", {"name": name, "variant": variant}
        )

    # ── MCP tool wrappers: Generation ─────────────────────────────────────

    def validate_workflow(self, workflow: dict[str, Any]) -> dict[str, Any]:
        """Validate a workflow before running it.

        Returns a dict with ``valid``, ``errors``, ``warnings``, ``info``.
        """
        return self._call_tool("validate_workflow", {"workflow": workflow})

    def run_workflow(
        self,
        workflow: dict[str, Any],
        *,
        sync: bool = True,
        output_mode: str = "file",
        name: str | None = None,
        image_format: str = "png",
    ) -> dict[str, Any]:
        """Submit a workflow for execution.

        Args:
            workflow: ComfyUI workflow JSON (API format).
            sync: Block until completion (default ``True``).
            output_mode: ``"file"`` | ``"base64"`` | ``"auto"``.
            name: Optional descriptive name for later retrieval.
            image_format: ``"png"`` | ``"jpeg"`` | ``"webp"``.

        Returns:
            With *sync=True*: the complete result including outputs.
            With *sync=False*: a dict with ``taskId`` for polling.
        """
        args: dict[str, Any] = {
            "workflow": workflow,
            "sync": sync,
            "outputMode": output_mode,
            "imageFormat": image_format,
        }
        if name:
            args["name"] = name
        return self._call_tool("run_workflow", args)

    def get_image(
        self,
        filename: str,
        subfolder: str = "",
        image_type: str = "output",
        image_format: str = "png",
    ) -> dict[str, Any]:
        """Retrieve a generated image by filename."""
        args: dict[str, Any] = {
            "filename": filename,
            "type": image_type,
            "imageFormat": image_format,
        }
        if subfolder:
            args["subfolder"] = subfolder
        return self._call_tool("get_image", args)

    # ── MCP tool wrappers: Task & Queue ───────────────────────────────────

    def get_task(self, task_id: str) -> dict[str, Any]:
        """Get the status of an async generation task."""
        return self._call_tool("get_task", {"taskId": task_id})

    def get_task_result(self, task_id: str) -> dict[str, Any]:
        """Get the result of a completed generation task."""
        return self._call_tool("get_task_result", {"taskId": task_id})

    def get_queue(self) -> dict[str, Any]:
        """Get the current ComfyUI queue status."""
        return self._call_tool("get_queue")

    def cancel_task(self, task_id: str) -> dict[str, Any]:
        """Cancel an async generation task."""
        return self._call_tool("cancel_task", {"taskId": task_id})

    def interrupt(self) -> dict[str, Any]:
        """Interrupt the currently running job."""
        return self._call_tool("interrupt")

    def get_history(self, limit: int = 10) -> dict[str, Any]:
        """Get generation history."""
        return self._call_tool("get_history", {"limit": limit})

    # ── MCP tool wrappers: Workflow Composition ───────────────────────────

    def build_node(
        self,
        node_type: str,
        node_id: str,
        inputs: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Generate valid node JSON with proper defaults."""
        args: dict[str, Any] = {"nodeType": node_type, "nodeId": node_id}
        if inputs:
            args["inputs"] = inputs
        return self._call_tool("build_node", args)

    def get_node_info(self, node: str) -> dict[str, Any]:
        """Get detailed info about a ComfyUI node type."""
        return self._call_tool("get_node_info", {"node": node})

    def find_nodes_by_type(
        self,
        input_type: str | None = None,
        output_type: str | None = None,
    ) -> dict[str, Any]:
        """Find nodes by their input or output types."""
        args: dict[str, Any] = {}
        if input_type:
            args["inputType"] = input_type
        if output_type:
            args["outputType"] = output_type
        return self._call_tool("find_nodes_by_type", args)

    def list_nodes(
        self,
        category: str | None = None,
        search: str | None = None,
    ) -> dict[str, Any]:
        """List available ComfyUI nodes."""
        args: dict[str, Any] = {}
        if category:
            args["category"] = category
        if search:
            args["search"] = search
        return self._call_tool("list_nodes", args)

    # ── Local workflow template helpers ────────────────────────────────────
    # Kept for backward-compatibility with local workflow JSON files in
    # the comfyui_workflows/ directory.

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
                node_title = str(
                    (obj.get("_meta") or {}).get("title") or ""
                ).lower()
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

    @staticmethod
    def _is_editable_scalar(value: Any) -> bool:
        """Return whether *value* is a scalar that can be overridden safely."""
        return isinstance(value, (str, int, float, bool))

    @staticmethod
    def _iter_workflow_nodes(
        workflow: Any,
    ) -> list[tuple[str | None, dict[str, Any]]]:
        """Return all detectable workflow nodes with optional node ids."""
        nodes: list[tuple[str | None, dict[str, Any]]] = []
        seen: set[int] = set()

        def visit(value: Any, node_id: str | None = None) -> None:
            if isinstance(value, dict):
                identity = id(value)
                if identity in seen:
                    return
                seen.add(identity)

                if isinstance(value.get("class_type"), str):
                    detected_id = (
                        node_id
                        or str(value.get("id") or "").strip()
                        or None
                    )
                    nodes.append((detected_id, value))

                for key, child in value.items():
                    next_node_id = None
                    if isinstance(child, dict) and isinstance(
                        child.get("class_type"), str
                    ):
                        next_node_id = str(key)
                    visit(child, next_node_id)
            elif isinstance(value, list):
                for item in value:
                    visit(item, None)

        visit(workflow)
        return nodes

    @staticmethod
    def _coerce_override_value(value: Any, current_value: Any) -> Any:
        """Convert *value* to the type already used by the workflow."""
        if isinstance(current_value, bool):
            if isinstance(value, str):
                return value.strip().lower() in {"1", "true", "yes", "on"}
            return bool(value)
        if isinstance(current_value, int) and not isinstance(
            current_value, bool
        ):
            return int(value)
        if isinstance(current_value, float):
            return float(value)
        if isinstance(current_value, str):
            return str(value)
        return value

    def describe_workflow_parameters(
        self,
        workflow_file: str,
        max_parameters: int = 80,
    ) -> list[dict[str, Any]]:
        """List editable scalar workflow parameters for agent-side selection."""
        workflow = self.load_template(workflow_file)
        parameters: list[dict[str, Any]] = []
        excluded_names = {"prompt", "text", "image_path"}

        for node_id, node in self._iter_workflow_nodes(workflow):
            class_type = str(node.get("class_type") or "")
            node_title = str(
                (node.get("_meta") or {}).get("title") or ""
            )

            inputs = node.get("inputs")
            if isinstance(inputs, dict):
                for input_name, current_value in inputs.items():
                    if str(input_name).lower() in excluded_names:
                        continue
                    if not self._is_editable_scalar(current_value):
                        continue
                    parameters.append(
                        {
                            "node_id": node_id,
                            "node_title": node_title,
                            "class_type": class_type,
                            "input_name": input_name,
                            "current_value": current_value,
                            "value_type": type(current_value).__name__,
                            "source": "inputs",
                        }
                    )

            if isinstance(inputs, list) and isinstance(
                node.get("widgets_values"), list
            ):
                widgets_values = node["widgets_values"]
                for index, node_input in enumerate(inputs):
                    if (
                        not isinstance(node_input, dict)
                        or index >= len(widgets_values)
                    ):
                        continue
                    current_value = widgets_values[index]
                    if not self._is_editable_scalar(current_value):
                        continue
                    input_name = (
                        node_input.get("name")
                        or node_input.get("label")
                        or (
                            (node_input.get("widget") or {}).get("name")
                        )
                    )
                    if (
                        not input_name
                        or str(input_name).lower() in excluded_names
                    ):
                        continue
                    parameters.append(
                        {
                            "node_id": node_id,
                            "node_title": node_title,
                            "class_type": class_type,
                            "input_name": input_name,
                            "current_value": current_value,
                            "value_type": type(current_value).__name__,
                            "source": "widgets_values",
                        }
                    )

        return parameters[:max_parameters]

    def apply_parameter_overrides(
        self,
        workflow: dict[str, Any],
        parameter_overrides: list[dict[str, Any]] | None,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """Apply agent-selected overrides to a prepared workflow."""
        if not parameter_overrides:
            return [], []

        applied: list[dict[str, Any]] = []
        skipped: list[dict[str, Any]] = []
        nodes = self._iter_workflow_nodes(workflow)

        for override in parameter_overrides:
            requested_input_name = str(
                override.get("input_name") or ""
            ).strip()
            if not requested_input_name:
                skipped.append(
                    {**override, "reason": "Missing input_name."}
                )
                continue

            requested_node_id = str(
                override.get("node_id") or ""
            ).strip()
            requested_node_title = (
                str(override.get("node_title") or "").strip().lower()
            )
            requested_class_type = (
                str(override.get("class_type") or "").strip().lower()
            )
            requested_value = override.get("value")

            matched = False
            for node_id, node in nodes:
                class_type = str(node.get("class_type") or "")
                node_title = str(
                    (node.get("_meta") or {}).get("title") or ""
                )

                if requested_node_id and requested_node_id != str(
                    node_id or ""
                ):
                    continue
                if (
                    requested_node_title
                    and requested_node_title != node_title.lower()
                ):
                    continue
                if (
                    requested_class_type
                    and requested_class_type != class_type.lower()
                ):
                    continue

                inputs = node.get("inputs")
                if isinstance(inputs, dict):
                    for actual_name, current_value in inputs.items():
                        if (
                            str(actual_name).lower()
                            != requested_input_name.lower()
                        ):
                            continue
                        if not self._is_editable_scalar(current_value):
                            continue
                        coerced_value = self._coerce_override_value(
                            requested_value, current_value
                        )
                        inputs[actual_name] = coerced_value
                        applied.append(
                            {
                                "node_id": node_id,
                                "node_title": node_title,
                                "class_type": class_type,
                                "input_name": actual_name,
                                "value": coerced_value,
                                "previous_value": current_value,
                                "source": "inputs",
                                "rationale": override.get("rationale", ""),
                            }
                        )
                        matched = True
                        break
                    if matched:
                        break

                if isinstance(inputs, list) and isinstance(
                    node.get("widgets_values"), list
                ):
                    widgets_values = node["widgets_values"]
                    for index, node_input in enumerate(inputs):
                        if (
                            not isinstance(node_input, dict)
                            or index >= len(widgets_values)
                        ):
                            continue
                        input_name = (
                            node_input.get("name")
                            or node_input.get("label")
                            or (
                                (node_input.get("widget") or {}).get("name")
                            )
                        )
                        if (
                            str(input_name or "").lower()
                            != requested_input_name.lower()
                        ):
                            continue
                        current_value = widgets_values[index]
                        if not self._is_editable_scalar(current_value):
                            continue
                        coerced_value = self._coerce_override_value(
                            requested_value, current_value
                        )
                        widgets_values[index] = coerced_value
                        applied.append(
                            {
                                "node_id": node_id,
                                "node_title": node_title,
                                "class_type": class_type,
                                "input_name": input_name,
                                "value": coerced_value,
                                "previous_value": current_value,
                                "source": "widgets_values",
                                "rationale": override.get("rationale", ""),
                            }
                        )
                        matched = True
                        break
                    if matched:
                        break

            if not matched:
                skipped.append(
                    {
                        **override,
                        "reason": (
                            "No matching editable workflow parameter "
                            "was found."
                        ),
                    }
                )

        return applied, skipped

    def prepare_workflow(
        self,
        workflow_file: str,
        prompt: str,
        input_images: list[str],
        parameter_overrides: list[dict[str, Any]] | None = None,
    ) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
        """Load a local workflow template, inject prompt and images.

        Args:
            workflow_file: Path to the ``.json`` workflow template.
            prompt: Creative prompt for every ``"prompt"`` field.
            input_images: Image paths for ``"image_path"`` fields.
            parameter_overrides: Optional scalar workflow input changes.

        Returns:
            (patched_workflow, applied_overrides, skipped_overrides)
        """
        workflow = self.load_template(workflow_file)
        self._inject_prompts(workflow, prompt)
        self._inject_images(workflow, list(input_images))  # copy
        applied_overrides, skipped_overrides = (
            self.apply_parameter_overrides(workflow, parameter_overrides)
        )
        return workflow, applied_overrides, skipped_overrides

    # ── Output extraction ─────────────────────────────────────────────────

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

    def _extract_output_references_from_text(
        self, value: Any,
    ) -> list[str]:
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
                        filename, subfolder,
                    )
                    if resolved not in seen:
                        seen.add(resolved)
                        references.append(resolved)
                    continue

                if (
                    url.lower().endswith(IMAGE_EXTENSIONS)
                    and url not in seen
                ):
                    seen.add(url)
                    references.append(url)

            for filename in _IMAGE_NAME_RE.findall(text):
                resolved = self._config.resolve_comfyui_output_file(
                    filename
                )
                if resolved not in seen:
                    seen.add(resolved)
                    references.append(resolved)

        return references

    def list_output_files_on_disk(self) -> list[str]:
        """Return all image files in configured output directories."""
        discovered: list[str] = []
        for output_dir in self._config.list_comfyui_output_dirs():
            if not os.path.isdir(output_dir):
                continue
            for root, _, file_names in os.walk(output_dir):
                for file_name in file_names:
                    if not file_name.lower().endswith(IMAGE_EXTENSIONS):
                        continue
                    discovered.append(
                        os.path.normpath(os.path.join(root, file_name))
                    )
        return sorted(discovered)

    def find_recent_output_files(
        self,
        started_at: float,
        known_files: set[str] | None = None,
    ) -> list[str]:
        """Return newly created or recently modified output files."""
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

    def wait_for_output_files(
        self,
        started_at: float,
        known_files: set[str] | None = None,
        expected_files: list[str] | None = None,
        timeout_seconds: float | None = None,
        poll_interval: float | None = None,
    ) -> list[str]:
        """Poll disk until generated output files appear or time out.

        Used as a fallback when the MCP result does not directly expose
        file paths for multi-step chaining.
        """
        known_files = known_files or set()
        expected_files = [
            os.path.normpath(path)
            for path in (expected_files or [])
            if isinstance(path, str) and path
        ]
        deadline = time.time() + (
            timeout_seconds
            if timeout_seconds is not None
            else self._config.comfyui_output_wait_timeout
        )
        interval = (
            poll_interval
            if poll_interval is not None
            else self._config.comfyui_output_poll_interval
        )

        while time.time() <= deadline:
            matched_expected = [
                path for path in expected_files if os.path.exists(path)
            ]
            if matched_expected:
                return matched_expected

            recent_files = self.find_recent_output_files(
                started_at=started_at,
                known_files=known_files,
            )
            if recent_files:
                return recent_files

            time.sleep(max(interval, 0.1))

        matched_expected = [
            path for path in expected_files if os.path.exists(path)
        ]
        if matched_expected:
            return matched_expected
        return self.find_recent_output_files(
            started_at=started_at, known_files=known_files,
        )

    # ── Output extraction helpers ─────────────────────────────────────────

    def extract_output_files(self, result: dict[str, Any]) -> list[str]:
        """Locate generated image file paths inside a workflow result."""
        files: list[str] = []

        # MCP server result: top-level images array
        for image in result.get("images", []):
            if isinstance(image, dict):
                fn = (
                    image.get("filename")
                    or image.get("path")
                    or image.get("url")
                )
                if fn:
                    # Resolve to local path when possible
                    if not os.path.isabs(fn) and not fn.startswith("http"):
                        subfolder = image.get("subfolder", "")
                        fn = self._config.resolve_comfyui_output_file(
                            fn, subfolder
                        )
                    files.append(fn)
            elif isinstance(image, str) and image:
                files.append(image)

        # ComfyUI outputs map (returned by some MCP sync results)
        outputs = result.get("outputs")
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
                        self._config.resolve_comfyui_output_file(
                            filename, subfolder
                        )
                    )

        # Text scan fallback
        files.extend(self._extract_output_references_from_text(result))

        # Deduplicate
        deduplicated: list[str] = []
        seen: set[str] = set()
        for file_path in files:
            if file_path not in seen:
                seen.add(file_path)
                deduplicated.append(file_path)
        return deduplicated

    def extract_output_image(self, result: dict[str, Any]) -> str | None:
        """Locate the primary output image path/URL in a result.

        Strategies tried in order:

        1. **images array** from MCP server sync result.
        2. **Explicit URL fields** (``asset_url``, ``image_url``).
        3. **Outputs map** (``outputs.<node>.images``).
        4. **Deep text scan** for image filenames/URLs.
        """
        # Strategy 1: MCP server images array
        images = result.get("images", [])
        if images:
            first = images[0]
            if isinstance(first, dict):
                fn = (
                    first.get("filename")
                    or first.get("url")
                    or first.get("path")
                )
                if fn:
                    if not os.path.isabs(fn) and not fn.startswith("http"):
                        subfolder = first.get("subfolder", "")
                        fn = self._config.resolve_comfyui_output_file(
                            fn, subfolder
                        )
                    return fn
            if isinstance(first, str) and first:
                return first

        # Strategy 2: explicit URL fields
        asset_url = result.get("asset_url") or result.get("image_url")
        if isinstance(asset_url, str) and asset_url:
            return asset_url

        # Strategy 3: outputs map → resolved path
        outputs = result.get("outputs")
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
                    return self._config.resolve_comfyui_output_file(
                        filename, subfolder
                    )

        # Strategy 4: deep text scan
        text_references = self._extract_output_references_from_text(result)
        if text_references:
            return text_references[0]

        # Strategy 5: all output files
        output_files = self.extract_output_files(result)
        if output_files:
            return output_files[0]

        return None
