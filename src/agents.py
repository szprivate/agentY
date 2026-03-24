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
from typing import Any

import requests
from strands import Agent, tool
from strands.models.openai import OpenAIModel

from comfyui import ComfyUIClient
from config import AppConfig
from models import FinalResult, PromptOutput, SupervisionOutput


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def summarize_for_llm(value: Any, limit: int = 4000) -> str:
    """Serialise *value* to indented JSON, truncating beyond *limit* chars."""
    text = json.dumps(value, indent=2, ensure_ascii=False, default=str)
    if len(text) <= limit:
        return text
    return f"{text[:limit]}\n... [truncated]"


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

        # ── Creator's tool ────────────────────────────────────────────────

        @tool
        def submit_workflow(
            prompt: str,
            workflow_file: str,
            input_images: list[str],
        ) -> dict[str, Any]:
            """Fill a ComfyUI workflow template and submit it for rendering.

            The generated image URL (if found) and a truncated result
            summary are returned to the calling agent.
            """
            workflow = comfyui.prepare_workflow(workflow_file, prompt, input_images)
            workflow_id = os.path.splitext(os.path.basename(workflow_file))[0]

            # Free GPU VRAM before ComfyUI needs it
            pipeline.unload_llm()

            result = comfyui.run_workflow(workflow, workflow_id=workflow_id)
            output_image = comfyui.extract_output_image(result)

            # Stash full payload so the orchestrator can forward it later
            creator_state["last_payload"] = {
                "result": result,
                "output_image": output_image,
            }
            return {
                "output_image": output_image,
                "result_summary": summarize_for_llm(result, limit=2000),
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

        self._creator = Agent(
            model=self._model,
            name="creator",
            description="Submits a prepared prompt and workflow to ComfyUI.",
            tools=[submit_workflow],
            system_prompt=(
                "You are the creator agent. Always call `submit_workflow` "
                "exactly once using the values provided in the user's "
                "message. After the tool succeeds, reply with a short "
                "factual summary."
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
            prompt: str,
            workflow_file: str,
            input_images: list[str],
        ) -> dict[str, Any]:
            """Create an image by submitting the workflow to ComfyUI."""
            creator_state.clear()
            creator(
                json.dumps(
                    {
                        "prompt": prompt,
                        "workflow_file": workflow_file,
                        "input_images": input_images,
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
            brief: str,
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
            result = supervisor(
                f"Brief:\n{brief}\n\n"
                f"Prompt:\n{prompt}\n\n"
                f"Mood images:\n{mood_list}\n\n"
                f"Input images:\n{input_list}\n\n"
                f"Output image:\n{creator_payload.get('output_image')}\n\n"
                f"ComfyUI result summary:\n"
                f"{summarize_for_llm(creator_payload.get('result', {}), limit=2500)}\n\n"
                "Decide whether the result should be accepted.",
                structured_output_model=SupervisionOutput,
            )
            if result.structured_output is None:
                raise RuntimeError(
                    "Supervisor agent did not return structured output."
                )
            return result.structured_output.model_dump()

        # ── Orchestrator agent ────────────────────────────────────────────

        self._orchestrator = Agent(
            model=self._model,
            name="orchestrator",
            description="Coordinates prompt creation, image generation, and review.",
            tools=[run_prompter, run_creator, run_supervisor],
            system_prompt=(
                "You are the orchestration agent. Use the tools in this "
                "exact order: `run_prompter`, then `run_creator`, then "
                "`run_supervisor`. Do not skip any step. When you finish, "
                "return the final accepted flag, supervision summary, "
                "output image path, prompt, and brief."
            ),
        )

    # ── Public API ────────────────────────────────────────────────────────

    def run(
        self,
        brief: str,
        input_images: list[str] | None = None,
    ) -> dict[str, Any]:
        """Execute the full generation pipeline.

        1. Resolve the workflow file and collect mood images.
        2. Invoke the orchestrator (which chains prompter → creator →
           supervisor internally).
        3. Return the structured :class:`FinalResult` as a plain dict.

        Args:
            brief: The creative brief describing the desired output.
            input_images: Optional paths to reference / input images.

        Returns:
            Dict with keys ``accepted``, ``supervision``,
            ``output_image``, ``prompt``, and ``brief``.

        Raises:
            RuntimeError: If the orchestrator fails to produce output.
        """
        workflow_file = self._config.select_workflow_file()
        mood_images = self._config.collect_mood_images()
        instructions = self._config.briefing_text

        result = self._orchestrator(
            json.dumps(
                {
                    "brief": brief,
                    "workflow_file": workflow_file,
                    "input_images": input_images or [],
                    "mood_images": mood_images,
                    "instructions": instructions,
                },
                indent=2,
            ),
            structured_output_model=FinalResult,
        )

        if result.structured_output is None:
            raise RuntimeError(
                "Orchestrator agent did not return structured output."
            )
        return result.structured_output.model_dump()
