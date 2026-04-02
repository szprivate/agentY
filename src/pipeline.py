"""
agentY – Two-agent pipeline: Researcher → Brain.

The pipeline exposes a single callable that accepts a raw user request,
runs it through the Researcher to produce a brainbriefing JSON, then
hands that JSON to the Brain for workflow assembly, execution, and QA.

Usage
-----
>>> from src.pipeline import create_pipeline
>>> pipeline = create_pipeline()
>>> response = pipeline("Generate a cinematic wide-shot of Tokyo at night.")
"""

from __future__ import annotations

import json
import os
import re
import textwrap
from typing import Any

from strands import Agent

from src.agent import create_brain_agent, create_researcher_agent


# ---------------------------------------------------------------------------
# JSON extraction helpers
# ---------------------------------------------------------------------------

def _extract_json(text: str) -> str | None:
    """Pull the first JSON object out of *text*, even if wrapped in a code fence.

    Returns the raw JSON string, or None if nothing parseable is found.
    """
    # 1. Strip optional ```json … ``` fences
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fenced:
        return fenced.group(1).strip()

    # 2. Grab the outermost { … } block
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    for i, ch in enumerate(text[start:], start):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1].strip()
    return None


def _validate_brainbriefing(raw: str) -> dict:
    """Parse *raw* JSON and check minimum required keys.

    Raises ``ValueError`` with a descriptive message on failure.
    """
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Researcher did not return valid JSON: {exc}") from exc

    required_top = {"brief", "workflow", "models", "sampler_config", "prompt"}
    missing = required_top - data.keys()
    if missing:
        raise ValueError(
            f"Brainbriefing is missing required top-level keys: {sorted(missing)}"
        )

    for key in ("brief", "workflow"):
        if not isinstance(data[key], dict):
            raise ValueError(
                f"Brainbriefing key '{key}' must be a JSON object, "
                f"got {type(data[key]).__name__}"
            )

    return data


# ---------------------------------------------------------------------------
# Pipeline callable
# ---------------------------------------------------------------------------

class Pipeline:
    """Chains Researcher → Brain with logging and JSON validation.

    Call ``pipeline(user_input)`` just like a Strands Agent.
    The Researcher runs once per call (stateless); the Brain keeps
    a sliding-window conversation so multi-turn interactions work.

    ``stream_async`` is also supported so the Slack server can update
    its placeholder message in real-time from the Brain stage.
    ``event_loop_metrics`` delegates to the Brain agent so token-usage
    reporting in Slack continues to work.
    """

    def __init__(self, researcher: Agent, brain: Agent, *, verbose: bool = True, skip_brain: bool = False) -> None:
        self._researcher = researcher
        self._brain = brain
        self._verbose = verbose
        self._skip_brain = skip_brain

    def _should_skip_brain(self) -> bool:
        if self._skip_brain:
            return True
        return str(os.environ.get("PIPELINE_SKIP_BRAIN", "false")).strip().lower() in (
            "1", "true", "yes", "on"
        )

    # The Slack server and main.py both do:  response = agent(user_input)
    def __call__(self, user_input: str, **kwargs: Any) -> str:
        return self.run(user_input, **kwargs)

    # Delegate metric access to the Brain so the Slack server can read
    # token usage summaries just as it would from a plain Strands Agent.
    @property
    def event_loop_metrics(self):  # noqa: ANN201
        return self._brain.event_loop_metrics

    def run(self, user_input: str, **_: Any) -> str:
        """Run the full pipeline for *user_input* and return the Brain's response."""
        raw_json, error, researcher_output = self._run_researcher(user_input)
        if error:
            return error

        if self._should_skip_brain():
            if self._verbose:
                print("[pipeline] Skipping Brain stage; returning Researcher output.")
            return researcher_output

        brain_prompt = self._build_brain_prompt(raw_json)
        brain_response = str(self._brain(brain_prompt))
        if self._verbose:
            print("[pipeline] Brain finished.")
        return brain_response

    async def stream_async(self, user_input: str):  # noqa: ANN201
        """Async generator compatible with the Slack server's streaming loop.

        Runs the Researcher synchronously (it's a single-turn spec dump),
        then transparently streams the Brain's token output so Slack can
        update its placeholder message in real time.

        Yields the same event dicts that a Strands Agent.stream_async would.
        """
        # Stage 1 – Researcher (synchronous; fast pattern-matching turn)
        if self._verbose:
            print("[pipeline:stream] Stage 1 – Researcher resolving spec …")
        raw_json, error, researcher_output = self._run_researcher(user_input)
        if error:
            yield {"data": error}
            return

        if self._should_skip_brain():
            if self._verbose:
                print("[pipeline:stream] Skipping Brain stage; returning Researcher output.")
            yield {"data": researcher_output}
            return

        # Stage 2 – Brain (streamed)
        if self._verbose:
            print("[pipeline:stream] Stage 2 – Brain streaming …")
        brain_prompt = self._build_brain_prompt(raw_json)
        async for event in self._brain.stream_async(brain_prompt):
            yield event

    # ── Internal helpers ─────────────────────────────────────────────── #

    _MAX_RESEARCHER_RETRIES = 2  # up to 2 correction rounds after the first attempt

    def _run_researcher(self, user_input: str) -> tuple[str | None, str | None, str]:
        """Run the Researcher and return ``(raw_json, error_message, researcher_output)``.

        If the first attempt fails validation, the error is fed back to the
        Researcher for up to ``_MAX_RESEARCHER_RETRIES`` correction rounds
        before giving up.

        Returns ``(json_str, None, researcher_output)`` on success, ``(None, error_str, last_output)`` on failure.
        """
        researcher_prompt = textwrap.dedent(f"""
            User request:
            {user_input}

            Resolve all fields and output the brainbriefing JSON.
        """).strip()

        last_error: str | None = None
        last_response: str = ""

        for attempt in range(1 + self._MAX_RESEARCHER_RETRIES):
            if attempt == 0:
                prompt = researcher_prompt
            else:
                # Feed the validation error back so the model can self-correct
                if self._verbose:
                    print(
                        f"[pipeline] Researcher retry {attempt}/{self._MAX_RESEARCHER_RETRIES} …"
                    )
                prompt = textwrap.dedent(f"""
                    Your previous output failed validation with this error:
                    {last_error}

                    Please output ONLY the corrected brainbriefing JSON object with
                    all required top-level keys: brief, workflow, models,
                    sampler_config, prompt.  No commentary, no markdown fences — just
                    the raw JSON.
                """).strip()

            last_response = str(self._researcher(prompt))
            if self._verbose:
                label = "initial" if attempt == 0 else f"retry {attempt}"
                print(f"[pipeline] Researcher finished ({label}). Extracting brainbriefing …")

            raw_json = _extract_json(last_response)
            if raw_json is None:
                last_error = "No JSON object found in the output."
                continue

            try:
                briefing = _validate_brainbriefing(raw_json)
            except ValueError as exc:
                last_error = str(exc)
                continue

            # Success
            task_id = briefing.get("task_id", "unknown")
            intent = briefing.get("brief", {}).get("intent", "unknown")
            template = briefing.get("workflow", {}).get("template_name", "unknown")
            if self._verbose:
                if attempt > 0:
                    print(f"[pipeline] Brainbriefing recovered after {attempt} retry(ies).")
                print(
                    f"[pipeline] Brainbriefing OK — task_id={task_id}, "
                    f"intent={intent}, template={template}"
                )
            return raw_json, None, last_response

        # All attempts exhausted
        return None, (
            f"Brainbriefing validation failed after {1 + self._MAX_RESEARCHER_RETRIES} attempts: "
            f"{last_error}\n\nLast researcher output:\n{last_response}"
        ), last_response

    def _build_brain_prompt(self, raw_json: str) -> str:
        """Format the Brain's input prompt from the resolved brainbriefing JSON."""
        task_id = "unknown"
        try:
            task_id = json.loads(raw_json).get("task_id", "unknown")
        except Exception:
            pass
        return textwrap.dedent(f"""
            Brainbriefing from Researcher (task_id={task_id}):

            ```json
            {raw_json}
            ```

            Assemble the ComfyUI workflow from this spec, execute it, run QA,
            and send the result to Slack.
        """).strip()


# ---------------------------------------------------------------------------
# Public factory
# ---------------------------------------------------------------------------

def create_pipeline(
    *,
    researcher_llm: str | None = None,
    researcher_ollama_model: str | None = None,
    researcher_anthropic_model: str | None = None,
    brain_llm: str | None = None,
    brain_anthropic_model: str | None = None,
    brain_ollama_model: str | None = None,
    verbose: bool = True,
    skip_brain: bool = False,
) -> Pipeline:
    """Create and return a ready-to-use two-agent Pipeline.

    All arguments are optional; each falls back to environment variables,
    then to hard-coded defaults.

    Researcher defaults:
        RESEARCHER_LLM          = ollama
        RESEARCHER_OLLAMA_MODEL = qwen3-coder:32b
        RESEARCHER_ANTHROPIC_MODEL (if llm=claude)

    Brain defaults:
        BRAIN_LLM               = claude
        BRAIN_ANTHROPIC_MODEL   = (ANTHROPIC_MODEL env, then claude-haiku-4-5)
        BRAIN_OLLAMA_MODEL (if llm=ollama)

    Args:
        researcher_llm: LLM backend for the Researcher (``'ollama'`` or ``'claude'``).
        researcher_ollama_model: Ollama model override for the Researcher.
        researcher_anthropic_model: Anthropic model override for the Researcher.
        brain_llm: LLM backend for the Brain (``'claude'`` or ``'ollama'``).
        brain_anthropic_model: Anthropic model override for the Brain.
        brain_ollama_model: Ollama model override for the Brain.
        verbose: Print stage-transition log lines (default True).
    """
    researcher = create_researcher_agent(
        llm=researcher_llm,
        ollama_model=researcher_ollama_model,
        anthropic_model=researcher_anthropic_model,
    )
    brain = create_brain_agent(
        llm=brain_llm,
        anthropic_model=brain_anthropic_model,
        ollama_model=brain_ollama_model,
    )
    return Pipeline(researcher, brain, verbose=verbose, skip_brain=skip_brain)
