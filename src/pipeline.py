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

import asyncio
import json
import os
import re
import textwrap
from typing import Any, List, Optional

from pydantic import BaseModel, Field, ValidationError
from strands import Agent

from src.agent import create_brain_agent, create_info_agent, create_researcher_agent, _settings
from src.utils.chat_summary import summarize_conversation
from src.utils.comfyui_interrupt_hook import INTERRUPT_NAME
from src.utils.comfyui_poller import poll_comfyui_job as _poll_comfyui_job
from src.utils.models import AgentSession, ChatSummary, MessageIntent, TriageResult
from src.utils.triage import triage as _triage, route as _route
from src.utils.workflow_signal import clear_and_get as _get_workflow_signal
from src.executor import execute_workflow as _execute_workflow
from src.tools.restart import restart_agent as _restart_agent


# ---------------------------------------------------------------------------
# Brainbriefing schema (Pydantic) — mirrors config/brainbrief_example.json
# ---------------------------------------------------------------------------

class InputImage(BaseModel):
    """A single input image/video asset referenced in the task."""
    node_id: str = Field(description="Node ID in the workflow JSON (from io.inputs[].nodeId)")
    filename: str = Field(description="Filename of the asset")
    role: str = Field(description="Role: master_image | reference_image | mask | depth_map | control_image")
    node: str = Field(description="ComfyUI loader node class name")
    slot: str = Field(description="Input slot name on the node")
    path: str = Field(description="Full path to the asset on disk")


class Task(BaseModel):
    """High-level description of what is being generated."""
    type: str = Field(description="Task type: image edit | image generation | video flf | video i2v | video v2v | audio")
    description: str = Field(description="One sentence summary of the task")


class BriefTemplate(BaseModel):
    """Selected ComfyUI workflow template."""
    name: Optional[str] = Field(default=None, description="Template name, or null if not resolved")


class BriefPrompt(BaseModel):
    """Generation prompts."""
    positive: str = Field(description="Positive generation prompt")
    negative: Optional[str] = Field(default=None, description="Negative prompt, or null")


class OutputNode(BaseModel):
    """A single output node in the ComfyUI workflow that saves generated assets."""
    node_id: str = Field(description="Node ID in the workflow JSON")
    node: str = Field(description="ComfyUI output node class name (e.g. SaveImage, VHS_VideoCombine)")
    output_path: str = Field(description="Full directory path where the node will save its output")


class BrainBriefing(BaseModel):
    """Structured handoff document from the Researcher to the Brain."""
    status: str = Field(description="'ready' or 'blocked'")
    blockers: List[str] = Field(default_factory=list, description="List of blocker descriptions")
    task: Task
    template: BriefTemplate
    input_images: List[InputImage] = Field(default_factory=list)
    input_image_count: int = Field(default=0, description="Must equal len(input_images)")
    output_nodes: List[OutputNode] = Field(default_factory=list, description="Output nodes from the workflow with their save paths")
    resolution_width: Optional[Any] = Field(default=None, description="Image width in pixels")
    resolution_height: Optional[Any] = Field(default=None, description="Image height in pixels")
    prompt: BriefPrompt
    # notes_for_executor: Optional[str] = Field(default=None, description="Additional notes for the Brain")


# ---------------------------------------------------------------------------
# JSON extraction helper
# ---------------------------------------------------------------------------

def _extract_json(text: str) -> str | None:
    """Pull the first JSON object out of *text*, even if wrapped in a code fence."""
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fenced:
        return fenced.group(1).strip()
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

    def __init__(
        self,
        researcher: Agent,
        brain: Agent,
        *,
        info_agent: Agent | None = None,
        verbose: bool = True,
        skip_brain: bool = False,
        info_context: dict | None = None,
        session_id: str = "default",
    ) -> None:
        self._researcher = researcher
        self._brain = brain
        self._info_agent: Agent = info_agent or create_info_agent()
        self._verbose = verbose
        self._skip_brain = skip_brain
        self._info_context: dict = info_context or {}
        self._session: AgentSession = AgentSession(session_id=session_id)
        # Brainbriefing JSON from the most recent Researcher run; used by the
        # Executor for Vision QA comparison in follow-up / feedback-loop rounds.
        self._last_brainbriefing_json: str | None = None

    def _should_skip_brain(self) -> bool:
        if self._skip_brain:
            return True
        return str(os.environ.get("PIPELINE_SKIP_BRAIN", "false")).strip().lower() in (
            "1", "true", "yes", "on"
        )

    # The Slack server and main.py both do:  response = agent(user_input)
    def __call__(self, user_input, **kwargs: Any) -> str:
        return self.run(user_input, **kwargs)

    # Delegate metric access to the Brain so the Slack server can read
    # token usage summaries just as it would from a plain Strands Agent.
    @property
    def event_loop_metrics(self):  # noqa: ANN201
        return self._brain.event_loop_metrics

    def run(self, user_input, **_: Any) -> str:
        """Run the full pipeline for *user_input* and return the Brain's response.

        Triage runs first to classify intent, then routes to:
        - ``researcher`` (new_request / low-confidence): full Researcher → Brain flow
        - ``brain`` (param_tweak / chain / feedback): direct Brain follow-up
        - ``answer`` (info_query): return the triage answer directly
        - ``log_warning`` (low-confidence fallback): treat as new_request
        """
        user_text = self._extract_text(user_input)
        triage_result = asyncio.run(_triage(user_text, self._session, self._info_context))
        handler = _route(triage_result)

        if self._verbose:
            print(f"[pipeline] Triage → intent={triage_result.intent.value},"
                  f" confidence={triage_result.confidence:.2f}, handler={handler}")

        if handler == "answer":
            if self._verbose:
                print("[pipeline] info_query → Info agent")
            return str(self._info_agent(user_text))

        if handler == "restart":
            if self._verbose:
                print("[pipeline] restart → restarting agent process")
            _restart_agent()
            return "♻️ Restarting agent…"  # unreachable; os.execv replaces the process

        if handler == "brain":
            # Follow-up: skip Researcher, send directly to Brain
            self._ensure_clean_history()
            brain_prompt = self._build_followup_prompt(user_text, triage_result)
            brain_response = str(self._brain(brain_prompt))
            self._session.follow_up_count += 1
            # Executor handoff: Brain signals a (re-)assembled workflow is ready
            workflow_path = _get_workflow_signal()
            executor_paths: list[str] = []
            if workflow_path:
                if self._verbose:
                    print(f"[pipeline] Brain (follow-up) signaled workflow ready: {workflow_path}")
                executor_lines, executor_paths = asyncio.run(
                    self._drain_executor(
                        workflow_path,
                        self._last_brainbriefing_json or "",
                    )
                )
                if executor_paths:
                    self._session.current_output_paths[:] = executor_paths
                if executor_lines:
                    brain_response = brain_response + "\n\n" + "\n".join(executor_lines)
            self._record_chat_summary(user_text, triage_result, status="completed")
            asyncio.run(self._compress_brain_history(extra_output_paths=executor_paths))
            if self._verbose:
                print("[pipeline] Brain (follow-up) finished.")
            return brain_response

        # handler == "researcher" or "log_warning" → full Researcher → Brain flow
        raw_json, error, researcher_output = self._run_researcher(user_input)
        if error:
            self._record_chat_summary(user_text, triage_result, status="error")
            return error

        if self._should_skip_brain():
            if self._verbose:
                print("[pipeline] Skipping Brain stage; returning Researcher output.")
            return researcher_output

        self._last_brainbriefing_json = raw_json
        brain_prompt = self._build_brain_prompt(raw_json)
        self._ensure_clean_history()
        brain_response = str(self._brain(brain_prompt))
        # Executor handoff: Brain signals the assembled workflow is ready
        workflow_path = _get_workflow_signal()
        executor_paths_r: list[str] = []
        if workflow_path:
            if self._verbose:
                print(f"[pipeline] Brain signaled workflow ready: {workflow_path}")
            executor_lines, executor_paths_r = asyncio.run(
                self._drain_executor(workflow_path, raw_json)
            )
            if executor_paths_r:
                self._session.current_output_paths[:] = executor_paths_r
            if executor_lines:
                brain_response = brain_response + "\n\n" + "\n".join(executor_lines)
        self._record_chat_summary(user_text, triage_result, status="completed", raw_json=raw_json)
        asyncio.run(self._compress_brain_history(extra_output_paths=executor_paths_r))
        if self._verbose:
            print("[pipeline] Brain finished.")
        return brain_response

    async def stream_async(self, user_input):  # noqa: ANN201
        """Async generator compatible with the Slack server's streaming loop.

        Runs the Researcher synchronously (it's a single-turn spec dump),
        then transparently streams the Brain's token output so Slack can
        update its placeholder message in real time.

        When the Brain is interrupted by ``ComfyUIInterruptHook`` (i.e. a
        ``submit_prompt`` call was just made), this method:
        1. Detects the ``"interrupt"`` stop reason in the event stream.
        2. Extracts the ``prompt_id`` from the interrupt's ``reason`` field.
        3. Polls ``GET /history/{prompt_id}`` in a cheap asyncio.sleep loop
           (zero LLM tokens burned during the wait).
        4. Resumes the Brain with an ``interruptResponse`` carrying the
           completed ComfyUI history, then continues streaming QA output.

        Yields the same event dicts that a Strands Agent.stream_async would.
        """
        # Stage 0 – Triage (classify intent before any agent is called)
        user_text = self._extract_text(user_input)
        triage_result = await _triage(user_text, self._session, self._info_context)
        handler = _route(triage_result)

        if self._verbose:
            print(f"[pipeline:stream] Triage → intent={triage_result.intent.value},"
                  f" confidence={triage_result.confidence:.2f}, handler={handler}")

        if handler == "answer":
            if self._verbose:
                print("[pipeline:stream] info_query → Info agent (streamed)")
            async for event in self._info_agent.stream_async(user_text):
                yield event
            return

        if handler == "restart":
            if self._verbose:
                print("[pipeline:stream] restart → restarting agent process")
            yield {"data": "♻️ Restarting agent…"}
            _restart_agent()
            return

        if handler == "brain":
            # Follow-up: skip Researcher, send directly to Brain (streamed)
            self._ensure_clean_history()
            brain_prompt = self._build_followup_prompt(user_text, triage_result)
            self._session.follow_up_count += 1
            async for event in self._brain.stream_async(brain_prompt):
                yield event
            # Executor handoff: stream execution events back to Slack
            workflow_path = _get_workflow_signal()
            executor_paths_fu: list[str] = []
            if workflow_path:
                if self._verbose:
                    print(f"[pipeline:stream] Brain (follow-up) signaled workflow ready: {workflow_path}")
                yield {"data": "\n\n_⚙️ Handing off to executor…_"}
                slack_channel_id, slack_thread_ts = self._get_slack_context()
                async for line in _execute_workflow(
                    workflow_path,
                    self._last_brainbriefing_json or "",
                    slack_channel_id=slack_channel_id,
                    slack_thread_ts=slack_thread_ts,
                    verbose=self._verbose,
                    collected_paths=executor_paths_fu,
                ):
                    yield {"data": f"\n{line}"}
            if executor_paths_fu:
                self._session.current_output_paths[:] = executor_paths_fu
            self._record_chat_summary(user_text, triage_result, status="completed")
            await self._compress_brain_history(extra_output_paths=executor_paths_fu)
            return

        # handler == "researcher" or "log_warning" → full Researcher → Brain flow
        # Stage 1 – Researcher (synchronous; fast pattern-matching turn)
        if self._verbose:
            print("[pipeline:stream] Stage 1 – Researcher resolving spec …")
        raw_json, error, researcher_output = self._run_researcher(user_input)
        if error:
            self._record_chat_summary(user_text, triage_result, status="error")
            yield {"data": error}
            return

        self._last_brainbriefing_json = raw_json

        if self._should_skip_brain():
            if self._verbose:
                print("[pipeline:stream] Skipping Brain stage; returning Researcher output.")
            yield {"data": researcher_output}
            return

        # Stage 2 – Brain (streamed, with optional ComfyUI interrupt handling)
        if self._verbose:
            print("[pipeline:stream] Stage 2 – Brain streaming …")
        self._ensure_clean_history()
        brain_prompt = self._build_brain_prompt(raw_json)

        # Brain input for the first invocation is the brain_prompt string;
        # subsequent invocations (after a ComfyUI interrupt) supply the
        # interruptResponse list.
        current_input: Any = brain_prompt

        while True:
            interrupt_result = None

            async for event in self._brain.stream_async(current_input):
                yield event  # forward all events to Slack verbatim

                # Detect agent interrupt: the final event before the loop stops
                # is AgentResultEvent = {"result": AgentResult(...)}
                if "result" in event:
                    agent_result = event["result"]
                    if getattr(agent_result, "stop_reason", None) == "interrupt":
                        interrupts = getattr(agent_result, "interrupts", [])
                        for intr in interrupts:
                            if getattr(intr, "name", None) == INTERRUPT_NAME:
                                interrupt_result = intr
                                break

            if interrupt_result is None:
                # Normal Brain completion — no ComfyUI interrupt pending.
                # Stage 3 – Executor: submit, poll, QA, Slack, save
                workflow_path = _get_workflow_signal()
                executor_paths_s: list[str] = []
                if workflow_path:
                    if self._verbose:
                        print(f"[pipeline:stream] Brain signaled workflow ready: {workflow_path}")
                    yield {"data": "\n\n_⚙️ Handing off to executor…_"}
                    slack_channel_id, slack_thread_ts = self._get_slack_context()
                    async for line in _execute_workflow(
                        workflow_path,
                        raw_json,
                        slack_channel_id=slack_channel_id,
                        slack_thread_ts=slack_thread_ts,
                        verbose=self._verbose,
                        collected_paths=executor_paths_s,
                    ):
                        yield {"data": f"\n{line}"}
                if executor_paths_s:
                    self._session.current_output_paths[:] = executor_paths_s
                self._record_chat_summary(user_text, triage_result, status="completed", raw_json=raw_json)
                await self._compress_brain_history(extra_output_paths=executor_paths_s)
                if self._verbose:
                    print("[pipeline:stream] Brain finished.")
                break

            # ── ComfyUI interrupt: poll cheaply, then resume ───────── #
            prompt_id: str = interrupt_result.reason
            if self._verbose:
                print(f"[pipeline:stream] ComfyUI interrupt — polling prompt_id={prompt_id}")

            yield {"data": f"\n\n_⏳ ComfyUI job queued (`{prompt_id}`). Waiting for completion…_"}

            history_result = await _poll_comfyui_job(prompt_id)

            yield {"data": "\n_✅ ComfyUI job finished — resuming…_"}
            if self._verbose:
                print(f"[pipeline:stream] ComfyUI job {prompt_id} finished. Resuming Brain.")

            # Resume the agent: supply the polled history as the interrupt response.
            current_input = [
                {
                    "interruptResponse": {
                        "interruptId": interrupt_result.id,
                        "response": json.dumps(history_result),
                    }
                }
            ]

    # ── Internal helpers ─────────────────────────────────────────────── #

    # ── Triage helpers ───────────────────────────────────────────────── #

    @staticmethod
    def _extract_text(user_input: Any) -> str:
        """Extract a plain-text string from a str or multimodal content-block list."""
        if isinstance(user_input, list):
            return "\n".join(block["text"] for block in user_input if "text" in block)
        return str(user_input)

    def _build_followup_prompt(self, user_text: str, triage_result: TriageResult) -> str:
        """Prompt for Brain when handling a follow-up (no Researcher pass needed).

        For ``feedback`` intent the raw user message is returned verbatim — the
        Brain receives it exactly as the user wrote it, with no wrapper.
        For all other follow-up intents a compact context block is prepended.
        """
        if triage_result.intent == MessageIntent.feedback:
            # The user's feedback message IS the new prompt for the Brain.
            return user_text

        context_lines: list[str] = []
        if self._session.chat_summaries:
            last = self._session.chat_summaries[-1]
            context_lines.append(f"Last workflow: {last.workflow_name}")
            context_lines.append(f"Last status: {last.status}")
        if self._session.current_output_paths:
            context_lines.append(
                f"Current outputs: {', '.join(self._session.current_output_paths)}"
            )
        context_block = ("\n".join(context_lines) + "\n\n") if context_lines else ""
        return textwrap.dedent(f"""\
            Follow-up request (intent: {triage_result.intent.value}):

            {context_block}{user_text}

            Apply the requested change directly, reusing the current session context.
        """).strip()

    def _ensure_clean_history(self) -> None:
        """Sanitize the Brain's message list before an invocation.

        If a previous call crashed mid-tool-execution, the Brain's
        messages may contain orphaned ``toolResult`` or ``toolUse``
        blocks.  This guard cleans them before the next API call so
        the Anthropic API doesn't reject the request.
        """
        msgs = self._brain.messages
        if not msgs:
            return
        cleaned = self._sanitize_messages(list(msgs))
        if len(cleaned) != len(msgs):
            if self._verbose:
                removed = len(msgs) - len(cleaned)
                print(f"[pipeline] Sanitized Brain history: removed {removed} "
                      f"orphaned tool message(s).")
            self._brain.messages[:] = cleaned

    @staticmethod
    def _sanitize_messages(messages: list[dict]) -> list[dict]:
        """Ensure *messages* don't start with orphaned ``toolResult`` blocks.

        The Anthropic API requires every ``tool_result`` content block to have
        a corresponding ``tool_use`` block in the immediately preceding
        assistant message.  When we slice message history (e.g. fallback
        ``messages[-4:]``), the slice may begin with a ``user`` message
        carrying ``toolResult`` blocks whose matching ``toolUse`` was in an
        earlier (now-discarded) assistant message.

        This helper trims leading messages until the first message no longer
        contains orphaned ``toolResult`` (or ``toolUse`` without a following
        ``toolResult``) blocks.
        """
        while messages:
            first = messages[0]
            content = first.get("content", [])
            if isinstance(content, list):
                has_tool_result = any(
                    isinstance(b, dict) and "toolResult" in b for b in content
                )
                if has_tool_result:
                    messages = messages[1:]
                    continue

                # An assistant message with toolUse but no following
                # toolResult message is also invalid.
                has_tool_use = any(
                    isinstance(b, dict) and "toolUse" in b for b in content
                )
                if has_tool_use:
                    if len(messages) < 2 or not any(
                        isinstance(b, dict) and "toolResult" in b
                        for b in (messages[1].get("content", []) if isinstance(messages[1].get("content", []), list) else [])
                    ):
                        messages = messages[1:]
                        continue
            break
        return messages

    async def _compress_brain_history(self, extra_output_paths: list[str] | None = None) -> None:
        """Summarise the Brain's conversation and replace its messages.

        After every Brain invocation the full message history is compressed
        into a single compact summary via ``summarize_conversation``.  The
        Brain's ``messages`` list is then replaced with a single assistant
        message containing that summary.  This ensures that the next agent
        call carries only the summary — not the full, token-heavy history.

        ``extra_output_paths`` are executor-produced file paths that don't
        appear in the Brain's message history (the executor runs outside the
        Brain loop).  They are injected into the summary so the next round
        can reference the generated outputs.
        """
        messages = self._brain.messages
        if not messages:
            return

        if self._verbose:
            msg_count = len(messages)
            print(f"[pipeline] Compressing Brain history ({msg_count} messages) …")

        try:
            summary = await summarize_conversation(messages, extra_output_paths=extra_output_paths)
        except Exception as exc:
            if self._verbose:
                print(f"[pipeline] WARNING: conversation summarisation failed ({exc}); "
                      "keeping last 4 messages as fallback.")
            # Fallback: keep only the last few messages to cap token growth.
            # Sanitize to avoid orphaned toolResult blocks at the start.
            self._brain.messages[:] = self._sanitize_messages(messages[-4:])
            return

        if not summary:
            if self._verbose:
                print("[pipeline] Empty summary returned; clearing history.")
            self._brain.messages.clear()
            return

        if self._verbose:
            print(f"[pipeline] Chat summary:\n{summary}\n")

        # Replace the entire history with a single summary message.
        # Using an "assistant" message so the agent treats it as its own
        # prior context rather than a new user instruction.
        self._brain.messages[:] = [
            {
                "role": "user",
                "content": [
                    {
                        "text": (
                            "[CONVERSATION SUMMARY FROM PRIOR ROUND]\n\n"
                            f"{summary}\n\n"
                            "[END OF SUMMARY — use this context for follow-up requests]"
                        ),
                    }
                ],
            },
            {
                "role": "assistant",
                "content": [
                    {
                        "text": "Understood. I have the context from the prior round and am ready for the next request.",
                    }
                ],
            },
        ]

        if self._verbose:
            print(f"[pipeline] Brain history compressed → {len(summary)} chars summary.")

    def _record_chat_summary(
        self,
        user_text: str,
        triage_result: TriageResult,
        *,
        status: str,
        raw_json: str | None = None,
    ) -> None:
        """Append a ChatSummary to the session after each pipeline invocation."""
        workflow_name = "unknown"
        if raw_json:
            try:
                workflow_name = (
                    json.loads(raw_json).get("template", {}).get("name") or "unknown"
                )
            except Exception:
                pass
        self._session.chat_summaries.append(
            ChatSummary(
                workflow_name=workflow_name,
                output_paths=list(self._session.current_output_paths),
                user_intent=triage_result.intent.value,
                status=status,
            )
        )

    _MAX_RESEARCHER_RETRIES = 2  # up to 2 correction rounds after the first attempt

    def _run_researcher(self, user_input) -> tuple[str | None, str | None, str]:
        """Run the Researcher and return ``(raw_json, error_message, researcher_output)``.

        Calls the Researcher as a normal text agent (preserving its tool use),
        extracts the JSON from the response, then validates it with the
        ``BrainBriefing`` Pydantic model.  On success the model is re-serialised
        so the Brain always receives canonically formatted JSON.

        If validation fails, the error is fed back to the Researcher for up to
        ``_MAX_RESEARCHER_RETRIES`` correction rounds before giving up.

        ``user_input`` may be a plain string *or* a list of Strands content
        blocks (the multimodal format produced by the Slack server when the
        user attaches images/videos).  When it is a list, the image/video
        blocks are forwarded to the Researcher intact so it can visually
        inspect the attachments via its ``analyze_image`` tool.

        Returns ``(json_str, None, researcher_output)`` on success, ``(None, error_str, last_output)`` on failure.
        """
        # Build a text-only version of the user request for the preamble and
        # for retry prompts where we can only send plain text.
        if isinstance(user_input, list):
            text_parts = [block["text"] for block in user_input if "text" in block]
            user_text = "\n".join(text_parts)
        else:
            user_text = user_input

        researcher_prompt_text = textwrap.dedent(f"""
            User request:
            {user_text}

            Resolve all fields and output the brainbriefing JSON.
        """).strip()

        # Always pass only the text prompt to the Researcher.
        # When user_input is a multimodal list, the text block already contains
        # the on-disk paths of any attached files (added by slack_server._build_content_blocks).
        # The Researcher can call analyze_image(file_path=...) if it needs to inspect
        # an image — much cheaper than embedding raw bytes in every LLM call.
        first_attempt_input: Any = researcher_prompt_text

        last_error: str | None = None

        for attempt in range(1 + self._MAX_RESEARCHER_RETRIES):
            if attempt == 0:
                prompt = first_attempt_input
            else:
                # Feed the validation error back so the model can self-correct
                if self._verbose:
                    print(
                        f"[pipeline] Researcher retry {attempt}/{self._MAX_RESEARCHER_RETRIES} …"
                    )
                prompt = textwrap.dedent(f"""
                    Your previous brainbriefing output failed JSON/schema validation:
                    {last_error}

                    Please output ONLY the corrected brainbriefing JSON with all
                    required fields correctly typed. No prose, no markdown fences.
                """).strip()

            last_response = str(self._researcher(prompt))
            label = "initial" if attempt == 0 else f"retry {attempt}"
            if self._verbose:
                print(f"[pipeline] Researcher finished ({label}). Extracting brainbriefing …")

            raw_json = _extract_json(last_response)
            if raw_json is None:
                last_error = "No JSON object found in the output."
                continue

            try:
                data = json.loads(raw_json)
                briefing = BrainBriefing.model_validate(data)
            except (json.JSONDecodeError, ValidationError) as exc:
                last_error = str(exc)
                if self._verbose:
                    print(f"[pipeline] Researcher ({label}) validation failed: {last_error}")
                continue

            # Success — re-serialise from validated model so Brain always gets clean JSON
            raw_json = briefing.model_dump_json(indent=2)
            if self._verbose:
                if attempt > 0:
                    print(f"[pipeline] Brainbriefing recovered after {attempt} retry(ies).")
                print(
                    f"[pipeline] Brainbriefing OK ({label}) — "
                    f"status={briefing.status}, task={briefing.task.description!r}, "
                    f"template={briefing.template.name!r}"
                )
            return raw_json, None, raw_json

        # All attempts exhausted
        return None, (
            f"Brainbriefing validation failed after {1 + self._MAX_RESEARCHER_RETRIES} attempts: "
            f"{last_error}"
        ), ""

    def _build_brain_prompt(self, raw_json: str) -> str:
        """Format the Brain's input prompt from the resolved brainbriefing JSON."""
        task_description = "unknown"
        try:
            task_description = json.loads(raw_json).get("task", {}).get("description", "unknown")
        except Exception:
            pass
        return textwrap.dedent(f"""
            Brainbriefing from Researcher (task: {task_description}):

            ```json
            {raw_json}
            ```

            Assemble and validate the ComfyUI workflow from this spec, then call
            `signal_workflow_ready(workflow_path)` as your final step.
            The pipeline will handle ComfyUI submission, completion polling,
            Vision QA (via Ollama), saving outputs to ./output, and Slack.
        """).strip()

    async def _drain_executor(
        self,
        workflow_path: str,
        brainbriefing_json: str,
        slack_channel_id: str = "",
        slack_thread_ts: str = "",
    ) -> tuple[list[str], list[str]]:
        """Drain the executor; return ``(status_lines, output_paths)``."""
        lines: list[str] = []
        output_paths: list[str] = []
        async for line in _execute_workflow(
            workflow_path,
            brainbriefing_json,
            slack_channel_id=slack_channel_id,
            slack_thread_ts=slack_thread_ts,
            verbose=self._verbose,
            collected_paths=output_paths,
        ):
            lines.append(line)
            if self._verbose:
                print(f"[executor] {line}")
        return lines, output_paths

    @staticmethod
    def _get_slack_context() -> tuple[str, str]:
        """Return ``(channel_id, thread_ts)`` from the active Slack contextvar."""
        try:
            from src.tools.slack_tools import _ctx_channel_id, _ctx_thread_ts
            return _ctx_channel_id.get() or "", _ctx_thread_ts.get() or ""
        except Exception:
            return "", ""


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
    info_context: dict | None = None,
    session_id: str = "default",
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
    info_agent = create_info_agent()
    return Pipeline(
        researcher,
        brain,
        info_agent=info_agent,
        verbose=verbose,
        skip_brain=skip_brain,
        info_context=info_context,
        session_id=session_id,
    )
