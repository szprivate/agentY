"""Multi-agent creative pipeline for image generation.

Defines :class:`CreativePipeline`, which wires together four specialised
AI agents (prompter, creator, supervisor, orchestrator) and exposes a
single :meth:`~CreativePipeline.run` method that executes the full
brief → prompt → image → review cycle.

Agent overview
--------------
* **Prompter**      – converts a brief + mood references into a detailed prompt.
* **Creator**       – fills a ComfyUI workflow template and submits it.
* **Supervisor**    – reviews whether the output satisfies the brief.
* **Orchestrator**  – chains the three agents above in the correct order.

Architecture note
-----------------
The ``strands`` framework requires tool functions to be plain callables
decorated with ``@tool``.  To give those callables access to pipeline
state (ComfyUI client, config, etc.) without module-level globals, they
are defined as **closures** inside :meth:`CreativePipeline._build_agents`.
"""

import json
import os
from urllib.parse import parse_qs, urlparse
from typing import Any
import time

import requests
from strands import Agent, tool
from strands.models.openai import OpenAIModel

from comfyui import ComfyUIClient
from config import AppConfig
from models import (
    ExecutionPlanOutput,
    FinalResult,
    OrchestrationTrace,
    PromptOutput,
    PromptDetails,
    StepTrace,
    SupervisionDetails,
    SupervisionOutput,
    WorkflowSelectionDetails,
    WorkflowSelectionOutput,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def summarize_for_llm(value: Any, limit: int = 4000) -> str:
    """Serialise *value* to indented JSON, truncating beyond *limit* chars."""
    text = json.dumps(value, indent=2, ensure_ascii=False, default=str)
    if len(text) <= limit:
        return text
    return f"{text[:limit]}\n... [truncated]"


def describe_local_workflow(file_name: str) -> str:
    """Create a short description for a local workflow file name."""
    stem = os.path.splitext(file_name)[0].replace("_", " ").replace(".", " ")
    lowered = stem.lower()

    if "upscale" in lowered:
        capability = "Local workflow for image upscaling or enhancement"
    elif "edit" in lowered:
        capability = "Local workflow for image editing using reference images"
    elif "generation" in lowered or "prompt" in lowered:
        capability = "Local workflow for prompt-driven image generation"
    else:
        capability = "Local workflow available in this workspace"

    return f"{capability}. Filename hint: {stem}."


# ---------------------------------------------------------------------------
# CreativePipeline
# ---------------------------------------------------------------------------


class CreativePipeline:
    """End-to-end pipeline: brief → prompt → image → supervision.

    Instantiating this class creates the LLM model, the ComfyUI client,
    and all four agents.  Call :meth:`run` to execute a full cycle.

    Args:
        config: Application configuration (URLs, paths, model id, …).
    """

    def __init__(self, config: AppConfig) -> None:
        self._config = config
        self._comfyui = ComfyUIClient(config)
        self._model = self._create_llm()
        self._progress_enabled = True

        # Mutable dict shared with the creator tool closure so that the
        # orchestrator can retrieve the ComfyUI result after the tool runs.
        self._creator_state: dict[str, Any] = {}

        # Agents are built last because they reference _model / _comfyui.
        self._build_agents()

    # ── LLM lifecycle ─────────────────────────────────────────────────────

    def _create_llm(self) -> OpenAIModel:
        """Instantiate the OpenAI-compatible model backed by Ollama."""
        api_key = (
            os.getenv("OLLAMA_API_KEY")
            or os.getenv("OPENAI_API_KEY")
            or "ollama"
        )
        return OpenAIModel(
            model_id=self._config.ollama_model_id,
            client_args={
                "api_key": api_key,
                "base_url": self._config.ollama_api_url,
            },
        )

    def unload_llm(self) -> None:
        """Ask Ollama to evict the model from GPU memory (best-effort).

        Sends a zero-length ``keep_alive`` request to the native Ollama
        API.  Failures are silently ignored so that the caller is never
        blocked by an unreachable Ollama instance.
        """
        base = self._config.ollama_native_url
        model = self._config.ollama_model_id
        if not base or not model:
            return
        try:
            requests.post(
                f"{base}/api/generate",
                json={"model": model, "prompt": "", "keep_alive": 0},
                timeout=30,
            )
        except requests.RequestException:
            pass

    def _emit_progress(self, title: str, details: str | None = None) -> None:
        """Print a short progress line before an agent starts a task."""
        if not self._progress_enabled:
            return
        print(f"[agent] {title}")
        if details:
            print(f"        {details}")

    @staticmethod
    def _brief_preview(text: str, limit: int = 140) -> str:
        """Return a one-line preview of a longer brief or instruction."""
        compact = " ".join(text.split())
        if len(compact) <= limit:
            return compact
        return f"{compact[:limit - 3]}..."

    def _resolve_output_reference(self, output_reference: str | None) -> str | None:
        """Convert an output reference into a local file path when possible.

        ComfyUI sometimes returns a preview URL rather than a filesystem path.
        For multi-step chaining, later steps need a usable local input path, so
        this method resolves ``/view?filename=...`` URLs back to the configured
        output directory.
        """
        if not output_reference:
            return None
        if os.path.exists(output_reference):
            return output_reference
        if output_reference.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
            return output_reference

        parsed = urlparse(output_reference)
        if not parsed.query:
            return None

        params = parse_qs(parsed.query)
        filename = (params.get("filename") or [""])[0]
        subfolder = (params.get("subfolder") or [""])[0]
        if not filename:
            return None
        return self._config.resolve_comfyui_output_file(filename, subfolder)

    def _derive_next_step_inputs(self, creator_payload: dict[str, Any]) -> list[str]:
        """Return the concrete files that should feed the next planned step."""
        output_files = list(creator_payload.get("output_files", []))
        if output_files:
            return output_files

        resolved_output = self._resolve_output_reference(
            creator_payload.get("output_image")
        )
        if resolved_output:
            return [resolved_output]

        return []

    # ── Agent construction ────────────────────────────────────────────────

    def _build_agents(self) -> None:
        """Create the four pipeline agents and their tool closures.

        Tool functions are defined as inner functions (closures) that
        capture ``self`` and sibling references.  This keeps them as
        plain ``@tool``-decorated callables — which ``strands`` requires —
        while avoiding module-level mutable state.
        """
        # Local aliases used by the closures below
        comfyui = self._comfyui
        creator_state = self._creator_state
        pipeline = self  # for unload_llm inside submit_workflow
        retrieval_state: dict[str, dict[str, Any]] = {}

        # ── Creator's tool ────────────────────────────────────────────────

        @tool
        def submit_workflow(
            prompt: str,
            workflow_file: str,
            input_images: list[str],
            parameter_overrides: list[dict[str, Any]] | None = None,
            template_id: str | None = None,
        ) -> dict[str, Any]:
            """Fill a ComfyUI workflow template and submit it for rendering.

            For local workflows, loads the JSON template and injects the
            prompt and images.  For MCP templates (when *template_id* is
            provided), builds the workflow via the MCP server's
            ``get_template`` tool instead.

            The generated image URL (if found) and a truncated result
            summary are returned to the calling agent.
            """
            applied_overrides: list[dict[str, Any]] = []
            skipped_overrides: list[dict[str, Any]] = []

            if template_id:
                # Build workflow from MCP server template
                template_params: dict[str, Any] = {"prompt": prompt}
                template_result = comfyui.get_template(
                    template_id, parameters=template_params,
                )
                workflow = template_result.get(
                    "workflow", template_result
                )
            else:
                # Local workflow JSON file
                workflow, applied_overrides, skipped_overrides = (
                    comfyui.prepare_workflow(
                        workflow_file,
                        prompt,
                        input_images,
                        parameter_overrides,
                    )
                )

            # Validate the workflow before running
            try:
                validation = comfyui.validate_workflow(workflow)
                if validation.get("errors"):
                    pipeline._emit_progress(
                        "Workflow validation warnings",
                        summarize_for_llm(
                            validation["errors"], limit=500
                        ),
                    )
            except Exception:
                pass  # validation is best-effort

            workflow_name = (
                template_id
                or os.path.splitext(
                    os.path.basename(workflow_file or "unnamed")
                )[0]
            )
            started_at = time.time()
            known_output_files = set(comfyui.list_output_files_on_disk())

            # Free GPU VRAM before ComfyUI needs it
            pipeline.unload_llm()

            # Run via MCP server (synchronous mode: blocks until done)
            result = comfyui.run_workflow(
                workflow, sync=True, name=workflow_name,
            )

            output_files = comfyui.extract_output_files(result)
            output_image = comfyui.extract_output_image(result)

            # Fallback: scan disk for recently created files
            if not output_files:
                recent_output_files = comfyui.find_recent_output_files(
                    started_at=started_at,
                    known_files=known_output_files,
                )
                if not recent_output_files:
                    pipeline._emit_progress(
                        "Creator: waiting for generated output files",
                        workflow_name,
                    )
                    recent_output_files = comfyui.wait_for_output_files(
                        started_at=started_at,
                        known_files=known_output_files,
                        expected_files=output_files,
                    )
                if recent_output_files:
                    output_files = recent_output_files

            if not output_image and output_files:
                output_image = output_files[0]

            # Stash full payload for the orchestrator
            creator_state["last_payload"] = {
                "result": result,
                "output_image": output_image,
                "output_files": output_files,
                "applied_parameter_overrides": applied_overrides,
                "skipped_parameter_overrides": skipped_overrides,
            }
            return {
                "output_image": output_image,
                "output_files": output_files,
                "applied_parameter_overrides": applied_overrides,
                "skipped_parameter_overrides": skipped_overrides,
                "result_summary": summarize_for_llm(result, limit=2000),
            }

        @tool
        def retrieve_workflow_candidates(
            brief: str,
            input_image_count: int = 0,
            limit: int = 10,
        ) -> dict[str, Any]:
            """Retrieve executable local and MCP server workflow candidates.

            Combines local workspace workflows with templates discovered
            via the comfyui-mcp server's ``search_templates`` tool.
            """
            local_candidates: list[dict[str, Any]] = []
            for workflow_path in self._config.list_workflow_files():
                workflow_name = os.path.basename(workflow_path)
                local_candidates.append(
                    {
                        "workflow_id": workflow_name,
                        "source": "local",
                        "title": os.path.splitext(workflow_name)[0],
                        "description": describe_local_workflow(workflow_name),
                        "workflow_file": workflow_path,
                        "template_id": None,
                        "compatibility": "native",
                    }
                )

            # Query the MCP server for matching templates
            mcp_candidates: list[dict[str, Any]] = []
            try:
                # Determine search parameters based on the brief
                search_kwargs: dict[str, Any] = {}
                brief_lower = brief.lower()
                if "upscal" in brief_lower:
                    search_kwargs["taskType"] = "upscale"
                elif "inpaint" in brief_lower:
                    search_kwargs["taskType"] = "inpaint"
                elif input_image_count > 0:
                    search_kwargs["taskType"] = "img2img"
                else:
                    search_kwargs["taskType"] = "txt2img"
                search_kwargs["query"] = brief[:200]

                search_result = comfyui.search_templates(**search_kwargs)
                templates = search_result.get(
                    "templates", search_result.get("results", [])
                )
                if isinstance(templates, list):
                    for tmpl in templates[: max(3, limit // 2)]:
                        if not isinstance(tmpl, dict):
                            continue
                        tmpl_id = (
                            tmpl.get("id")
                            or tmpl.get("templateId")
                            or tmpl.get("name")
                            or ""
                        )
                        if not tmpl_id:
                            continue
                        mcp_candidates.append(
                            {
                                "workflow_id": tmpl_id,
                                "source": "mcp",
                                "title": tmpl.get(
                                    "name", tmpl.get("title", tmpl_id)
                                ),
                                "description": tmpl.get(
                                    "description", ""
                                ),
                                "workflow_file": None,
                                "template_id": tmpl_id,
                                "compatibility": "mcp_template",
                                "model_type": tmpl.get("modelType", ""),
                                "task_type": tmpl.get("taskType", ""),
                            }
                        )
            except Exception as exc:
                pipeline._emit_progress(
                    "MCP template search failed (using local only)",
                    str(exc)[:200],
                )

            ordered_candidates = [
                *local_candidates, *mcp_candidates
            ][:limit]
            retrieval_state.clear()
            retrieval_state.update(
                {
                    candidate["workflow_id"]: candidate
                    for candidate in ordered_candidates
                }
            )
            return {
                "brief": brief,
                "input_image_count": input_image_count,
                "selection_notes": [
                    "Choose exactly one workflow_id from the candidate list.",
                    "Use descriptions as the primary evidence for task fit.",
                    "Local candidates are immediately executable in this workspace.",
                    "MCP template candidates are built on-the-fly by the MCP server and support diverse model architectures.",
                ],
                "candidates": ordered_candidates,
            }

        # ── Individual agents ─────────────────────────────────────────────

        self._prompter = Agent(
            model=self._model,
            name="prompter",
            description="Turns a creative brief into a production-ready prompt.",
            system_prompt=(
                "You are a creative image prompt engineer. "
                "Turn the user's brief, mood references, and instructions "
                "into one strong image-generation prompt. Be specific about "
                "composition, camera angle, lighting, and style."
            ),
        )

        self._orchestrator = Agent(
            model=self._model,
            name="orchestrator",
            description="Plans single-step or multi-step execution from the brief.",
            system_prompt=(
                "You are the orchestration agent. Analyze the user's brief and "
                "convert it into an ordered execution plan. Each step must be a "
                "single image-processing action that can be handled by one "
                "workflow-selection, prompt-generation, creation, and supervision "
                "cycle. If the brief contains multiple sequential actions such as "
                "'first', 'then', or 'finally', return multiple steps in order. "
                "Ignore setup-only actions like collecting files from a folder, "
                "because the application already provides available input images. "
                "Later steps may assume they receive the previous step's output as "
                "their input. Keep the number of steps minimal but complete."
            ),
        )

        self._workflow_selector = Agent(
            model=self._model,
            name="workflow_selector",
            description="Chooses the most suitable ComfyUI workflow for a brief.",
            tools=[retrieve_workflow_candidates],
            system_prompt=(
                "You are selecting the best ComfyUI workflow for an image task. "
                "Always retrieve workflow candidates first, then choose exactly "
                "one workflow_id from the returned candidates. Use candidate "
                "descriptions as the primary evidence for task fit, and prefer "
                "the most specific executable workflow for the requested step."
            ),
        )

        self._creator = Agent(
            model=self._model,
            name="creator",
            description="Submits a prepared prompt and workflow to ComfyUI.",
            tools=[submit_workflow],
            system_prompt=(
                "You are the creator agent. Always call `submit_workflow` "
                "exactly once using the values provided in the user's "
                "message. If a template_id is provided, pass it to "
                "`submit_workflow` so the MCP server builds the workflow. "
                "If the current step explicitly requests measurable "
                "workflow settings such as width, height, batch size, steps, "
                "denoise, seed, CFG, or similar scalar parameters, include a "
                "`parameter_overrides` list. Only use parameters that appear "
                "in the provided editable parameter list, and copy the exact "
                "`node_title`, `class_type`, and `input_name` fields from that "
                "list when creating overrides. Do not invent parameters. After "
                "the tool succeeds, reply with a short factual summary."
            ),
        )

        self._supervisor = Agent(
            model=self._model,
            name="supervisor",
            description="Reviews whether the generated output satisfies the brief.",
            system_prompt=(
                "You are an expert creative supervisor. Assess whether the "
                "output matches the brief, mood references, and input "
                "images. Be strict but concise."
            ),
        )

        # ── Orchestrator tools (delegate to the agents above) ─────────────

        prompter = self._prompter
        creator = self._creator
        supervisor = self._supervisor
        workflow_selector = self._workflow_selector

        @tool
        def select_workflow(brief: str, input_image_count: int = 0) -> dict[str, Any]:
            """Choose the most suitable workflow for the supplied brief.

            The selection is based on the semantic intent of the brief and the
            available local workflows plus retrieved remote template metadata.
            """
            def resolve_selected_candidate(
                workflow_id: str,
            ) -> dict[str, Any] | None:
                """Resolve exact or extension-less workflow identifiers."""
                candidate = retrieval_state.get(workflow_id)
                if candidate is not None:
                    return candidate

                candidate = retrieval_state.get(f"{workflow_id}.json")
                if candidate is not None:
                    return candidate

                for candidate in retrieval_state.values():
                    candidate_id = str(candidate.get("workflow_id", ""))
                    if os.path.splitext(candidate_id)[0] == workflow_id:
                        return candidate
                return None

            retrieval_state.clear()

            result = workflow_selector(
                json.dumps(
                    {
                        "brief": brief,
                        "input_image_count": input_image_count,
                        "selection_rules": [
                            "Call `retrieve_workflow_candidates` exactly once before making a decision.",
                            "Pick exactly one workflow_id from the retrieved candidates.",
                            "Use descriptions as the primary basis for the selection.",
                            "Prefer the most specific executable workflow for the task.",
                        ],
                    },
                    indent=2,
                ),
                structured_output_model=WorkflowSelectionOutput,
            )
            if result.structured_output is None:
                raise RuntimeError(
                    "Workflow selector agent did not return structured output."
                )

            selection = result.structured_output
            selected_name = selection.workflow_name
            selected_candidate = resolve_selected_candidate(selected_name)
            if selected_candidate is None:
                retrieve_workflow_candidates(
                    brief,
                    input_image_count=input_image_count,
                )
                selected_candidate = resolve_selected_candidate(selected_name)

            if selected_candidate is None:
                raise RuntimeError(
                    "Workflow selector chose an unknown workflow: "
                    f"{selected_name}"
                )

            selected_path = selected_candidate.get("workflow_file")
            selected_template_id = selected_candidate.get("template_id")

            # MCP templates don't have a local file — they are built
            # on-the-fly by the server when get_template is called.
            if not selected_path and not selected_template_id:
                raise RuntimeError(
                    "Could not resolve the selected workflow to a local "
                    f"file or MCP template: {selected_name}"
                )

            return {
                "workflow_file": selected_path,
                "workflow_name": selected_name,
                "template_id": selected_template_id,
                "rationale": selection.rationale,
            }

        @tool
        def run_prompter(
            brief: str,
            mood_images: list[str],
            instructions: str,
        ) -> dict[str, Any]:
            """Generate a polished image prompt from the user's brief."""
            mood_list = (
                ", ".join(os.path.basename(p) for p in mood_images) or "None"
            )
            result = prompter(
                f"Brief:\n{brief}\n\n"
                f"Mood images:\n{mood_list}\n\n"
                f"Instructions:\n{instructions or 'None'}",
                structured_output_model=PromptOutput,
            )
            if result.structured_output is None:
                raise RuntimeError("Prompter agent did not return structured output.")
            return result.structured_output.model_dump()

        @tool
        def run_creator(
            brief: str,
            prompt: str,
            workflow_file: str,
            input_images: list[str],
            template_id: str | None = None,
        ) -> dict[str, Any]:
            """Create an image by submitting the workflow to ComfyUI.

            For local workflows, loads the JSON template and injects the
            prompt / images.  For MCP server templates, passes the
            *template_id* to the creator so the workflow is built via
            ``get_template``.
            """
            creator_state.clear()

            # For local workflows, describe editable parameters so the
            # agent can set overrides.  MCP templates handle parameters
            # internally via get_template.
            editable_parameters: list[dict[str, Any]] = []
            if workflow_file and not template_id:
                editable_parameters = comfyui.describe_workflow_parameters(
                    workflow_file
                )

            creator(
                json.dumps(
                    {
                        "brief": brief,
                        "prompt": prompt,
                        "workflow_file": workflow_file or "",
                        "template_id": template_id or "",
                        "input_images": input_images,
                        "editable_workflow_parameters": editable_parameters,
                        "override_rules": [
                            "Set parameter_overrides only when the brief explicitly asks for a measurable workflow setting.",
                            "Use the editable_workflow_parameters list as the only allowed source of override targets.",
                            "Prefer exact node_title plus input_name matches.",
                            "Do not override prompt text or input image paths through parameter_overrides.",
                            "When template_id is provided, pass it to submit_workflow.",
                        ],
                    },
                    indent=2,
                ),
            )
            payload = creator_state.get("last_payload")
            if payload is None:
                raise RuntimeError("Creator agent did not submit a workflow.")
            return payload

        @tool
        def run_supervisor(
            primary_brief: str,
            supporting_brief: str,
            prompt: str,
            mood_images: list[str],
            input_images: list[str],
            creator_payload: dict[str, Any],
        ) -> dict[str, Any]:
            """Review the generated output and decide whether it's acceptable."""
            mood_list = (
                ", ".join(os.path.basename(p) for p in mood_images) or "None"
            )
            input_list = (
                ", ".join(os.path.basename(p) for p in input_images) or "None"
            )
            output_files = creator_payload.get("output_files", [])
            existing_output_files = [
                file_path for file_path in output_files if os.path.exists(file_path)
            ]
            output_file_list = (
                "\n".join(f"- {file_path}" for file_path in output_files)
                if output_files
                else "None"
            )
            existing_output_file_list = (
                "\n".join(f"- {file_path}" for file_path in existing_output_files)
                if existing_output_files
                else "None"
            )
            result = supervisor(
                f"Primary evaluation target:\n{primary_brief}\n\n"
                f"Supporting context:\n{supporting_brief}\n\n"
                f"Generated prompt (implementation detail, not the success criteria):\n{prompt}\n\n"
                f"Mood images:\n{mood_list}\n\n"
                f"Input images:\n{input_list}\n\n"
                f"Output image reference:\n{creator_payload.get('output_image') or 'None'}\n\n"
                f"Output files:\n{output_file_list}\n\n"
                f"Existing output files on disk:\n{existing_output_file_list}\n\n"
                f"ComfyUI result summary:\n"
                f"{summarize_for_llm(creator_payload.get('result', {}), limit=2500)}\n\n"
                "Judge success primarily against the primary evaluation target. "
                "Use the supporting context only to understand the broader goal. "
                "Treat the generated prompt as implementation context rather than "
                "the source of truth for success. "
                "If output files are listed under existing output files on disk, "
                "treat that as stronger evidence than a stale job status such as "
                "'running'. A real file on disk means the step produced an image. "
                "If an output image reference or output files are listed above, "
                "treat that as evidence that an image was produced. Decide whether "
                "the result should be accepted.",
                structured_output_model=SupervisionOutput,
            )
            if result.structured_output is None:
                raise RuntimeError(
                    "Supervisor agent did not return structured output."
                )
            return result.structured_output.model_dump()

        # Store the tool callables so the Python execution engine can reuse the
        # same logic after the orchestrator has produced a multi-step plan.
        self._select_workflow_tool = select_workflow
        self._run_prompter_tool = run_prompter
        self._run_creator_tool = run_creator
        self._run_supervisor_tool = run_supervisor

    # ── Public API ────────────────────────────────────────────────────────

    def run(
        self,
        brief: str,
        input_images: list[str] | None = None,
    ) -> dict[str, Any]:
        """Execute the full generation pipeline.

          1. Start the MCP server connection to ComfyUI.
          2. Collect mood images and available workflow names.
          3. Ask the orchestration agent to turn the brief into one or more
              sequential execution steps.
          4. Execute each planned step with workflow selection, prompt
              generation, image creation, and supervision.
          5. Return a structured :class:`FinalResult` including a per-step trace.

        Args:
            brief: The creative brief describing the desired output.
            input_images: Optional paths to reference / input images.

        Returns:
            Dict with the final result plus verbose execution details for each
            planned step.

        Raises:
            RuntimeError: If the orchestrator fails to produce output.
        """
        # Start the MCP server connection
        self._emit_progress("Starting ComfyUI MCP server connection")
        self._comfyui.start()
        try:
            return self._run_pipeline(brief, input_images)
        finally:
            self._comfyui.stop()

    def _run_pipeline(
        self,
        brief: str,
        input_images: list[str] | None = None,
    ) -> dict[str, Any]:
        """Inner pipeline execution (assumes MCP server is connected)."""
        # Optional: check ComfyUI status
        try:
            status = self._comfyui.get_status()
            self._emit_progress(
                "ComfyUI status",
                summarize_for_llm(status, limit=200),
            )
        except Exception:
            pass  # status check is informational

        mood_images = self._config.collect_mood_images()
        instructions = self._config.briefing_text
        available_workflows = [
            os.path.basename(path)
            for path in self._config.list_workflow_files()
        ]

        self._emit_progress(
            "Orchestrator: building execution plan",
            self._brief_preview(brief),
        )
        planning_result = self._orchestrator(
            json.dumps(
                {
                    "brief": brief,
                    "available_workflows": available_workflows,
                    "input_images": input_images or [],
                    "mood_images": mood_images,
                    "instructions": instructions,
                    "planning_rules": [
                        "Return a single step when the brief is one coherent task.",
                        "Return multiple steps when the brief clearly describes a sequence.",
                        "Ignore setup-only actions such as collecting files from disk.",
                        "Later steps can build on previous step outputs.",
                    ],
                },
                indent=2,
            ),
            structured_output_model=ExecutionPlanOutput,
        )

        if planning_result.structured_output is None:
            raise RuntimeError(
                "Orchestrator agent did not return a valid execution plan."
            )

        plan = planning_result.structured_output
        steps = sorted(plan.steps, key=lambda step: step.step_number)
        if not steps:
            raise RuntimeError("Orchestrator returned an empty execution plan.")

        current_input_images = list(input_images or mood_images)
        step_traces: list[StepTrace] = []

        for step in steps:
            step_input_images = list(current_input_images)

            self._emit_progress(
                f"Step {step.step_number}/{len(steps)}: preparing execution plan item",
                self._brief_preview(step.title or step.brief),
            )

            self._emit_progress(
                "Workflow selector: choosing workflow",
                self._brief_preview(step.brief),
            )
            workflow_selection_payload = self._select_workflow_tool(
                step.brief,
                input_image_count=len(step_input_images),
            )
            self._emit_progress(
                "Selected workflow",
                workflow_selection_payload["workflow_name"],
            )

            self._emit_progress(
                "Prompter: generating image prompt",
                self._brief_preview(step.brief),
            )
            prompt_payload = self._run_prompter_tool(
                step.brief,
                mood_images,
                instructions,
            )

            self._emit_progress(
                "Creator: running selected workflow",
                (
                    f"workflow={workflow_selection_payload['workflow_name']}; "
                    f"inputs={len(step_input_images)}"
                ),
            )
            creator_payload = self._run_creator_tool(
                step.brief,
                prompt_payload["prompt"],
                workflow_selection_payload.get("workflow_file") or "",
                step_input_images,
                template_id=workflow_selection_payload.get("template_id"),
            )

            self._emit_progress(
                "Supervisor: reviewing generated result",
                f"workflow={workflow_selection_payload['workflow_name']}",
            )
            supervision_payload = self._run_supervisor_tool(
                step.brief,
                brief,
                prompt_payload["prompt"],
                mood_images,
                step_input_images,
                creator_payload,
            )

            step_trace = StepTrace(
                step_number=step.step_number,
                title=step.title,
                brief=step.brief,
                input_images=step_input_images,
                output_image=creator_payload.get("output_image"),
                output_files=creator_payload.get("output_files", []),
                parameter_overrides=creator_payload.get(
                    "applied_parameter_overrides",
                    [],
                ),
                skipped_parameter_overrides=creator_payload.get(
                    "skipped_parameter_overrides",
                    [],
                ),
                workflow_selection=WorkflowSelectionDetails(
                    workflow_name=workflow_selection_payload["workflow_name"],
                    workflow_file=(
                        workflow_selection_payload.get("workflow_file")
                        or workflow_selection_payload.get("template_id")
                        or ""
                    ),
                    rationale=workflow_selection_payload.get("rationale", ""),
                ),
                prompt_generation=PromptDetails(
                    prompt=prompt_payload["prompt"],
                ),
                supervision=SupervisionDetails(
                    accepted=supervision_payload["accepted"],
                    verdict=supervision_payload["supervision"],
                ),
            )
            step_traces.append(step_trace)

            next_input_images = self._derive_next_step_inputs(creator_payload)
            if next_input_images:
                current_input_images = list(next_input_images)
                self._emit_progress(
                    "Chaining step output into next input",
                    f"files={len(next_input_images)}",
                )
            elif step.step_number < len(steps):
                raise RuntimeError(
                    "A multi-step plan requires the previous step to produce an "
                    "input image for the next step, but no reusable output file "
                    f"was found after step {step.step_number}."
                )

        final_step = step_traces[-1]
        final_supervision_payload = self._run_supervisor_tool(
            brief,
            final_step.brief,
            final_step.prompt_generation.prompt,
            mood_images,
            final_step.input_images,
            {
                "output_image": final_step.output_image,
                "output_files": final_step.output_files,
                "result": {
                    "step_count": len(step_traces),
                    "step_summaries": [
                        {
                            "step_number": step.step_number,
                            "title": step.title,
                            "accepted": step.supervision.accepted,
                            "verdict": step.supervision.verdict,
                            "output_image": step.output_image,
                            "output_files": step.output_files,
                        }
                        for step in step_traces
                    ],
                },
            },
        )
        final_supervision = SupervisionDetails(
            accepted=final_supervision_payload["accepted"],
            verdict=final_supervision_payload["supervision"],
        )

        overall_accepted = final_supervision.accepted
        step_supervision_summary = "\n".join(
            (
                f"Step {step.step_number} ({step.title}): "
                f"{'accepted' if step.supervision.accepted else 'rejected'} - "
                f"{step.supervision.verdict}"
            )
            for step in step_traces
        )
        overall_supervision = (
            f"{step_supervision_summary}\n"
            f"Final review against original brief: "
            f"{'accepted' if final_supervision.accepted else 'rejected'} - "
            f"{final_supervision.verdict}"
        )
        orchestration_summary = (
            f"Planned {len(step_traces)} step(s). {plan.summary} "
            f"Overall result: {'accepted' if overall_accepted else 'rejected'}."
        )

        final_result = FinalResult(
            accepted=overall_accepted,
            supervision=overall_supervision,
            output_image=final_step.output_image,
            output_files=final_step.output_files,
            prompt=final_step.prompt_generation.prompt,
            brief=brief,
            workflow_name=final_step.workflow_selection.workflow_name,
            workflow_file=final_step.workflow_selection.workflow_file,
            workflow_rationale=final_step.workflow_selection.rationale,
            plan_summary=plan.summary,
            step_count=len(step_traces),
            orchestration_summary=orchestration_summary,
            trace=OrchestrationTrace(
                summary=orchestration_summary,
                plan_summary=plan.summary,
                workflow_selection=final_step.workflow_selection,
                prompt_generation=final_step.prompt_generation,
                supervision=final_supervision,
                steps=step_traces,
            ),
        )
        return final_result.model_dump()
