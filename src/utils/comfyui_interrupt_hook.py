"""
Human-in-the-loop interrupt hook for ComfyUI job submission.

Flow
----
1. The Brain agent calls ``submit_prompt(workflow_path)`` → gets prompt_id.
2. ``ComfyUIInterruptHook.after_submit`` (AfterToolCallEvent) stores prompt_id.
3. Before the agent's *next* tool call, ``before_any_tool`` fires and calls
   ``event.interrupt("wait_for_comfyui", reason=prompt_id)``, which:
   a. Halts the agent event loop (no more LLM calls / token burn).
   b. Surfaces ``AgentResult(stop_reason="interrupt", interrupts=[...])``.
4. The Pipeline's ``stream_async`` catches the interrupt, polls
   ``GET /history/{prompt_id}`` in a cheap asyncio.sleep loop, then resumes:
      agent.stream_async([{
          "interruptResponse": {
              "interruptId": interrupt.id,
              "response": result_json,   # ComfyUI history dict
          }
      }])
5. On resume the pending tool call re-executes (returning a fresh, now-complete
   status) and the agent finishes its QA pass.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from strands.hooks import AfterToolCallEvent, BeforeToolCallEvent, HookProvider, HookRegistry
from strands.hooks.events import AfterInvocationEvent

logger = logging.getLogger("agentY.comfyui_interrupt_hook")

# Sentinel name used to identify our interrupt so the pipeline knows which
# interrupt to handle.
INTERRUPT_NAME = "wait_for_comfyui"


class ComfyUIInterruptHook(HookProvider):
    """Pauses the Brain after ``submit_prompt`` and parks the polling work in
    the orchestrator, eliminating repeated agent turns during job execution."""

    def __init__(self) -> None:
        # prompt_id / client_id that are pending an interrupt (set after submit_prompt)
        self._pending_prompt_id: str | None = None
        self._pending_client_id: str | None = None

    # ------------------------------------------------------------------ #
    # HookProvider interface                                               #
    # ------------------------------------------------------------------ #

    def register_hooks(self, registry: HookRegistry, **kwargs: Any) -> None:
        registry.add_callback(AfterToolCallEvent, self._after_submit)
        registry.add_callback(BeforeToolCallEvent, self._before_any_tool)
        registry.add_callback(AfterInvocationEvent, self._after_invocation)

    # ------------------------------------------------------------------ #
    # Callbacks                                                            #
    # ------------------------------------------------------------------ #

    def _after_submit(self, event: AfterToolCallEvent) -> None:
        """Detect a successful ``submit_prompt`` call and cache the prompt_id."""
        if event.tool_use["name"] != "submit_prompt":
            return

        # result is a ToolResult dict: {"toolUseId": ..., "status": ..., "content": [{"text": "..."}]}
        try:
            content = event.result.get("content", [])
            if not content:
                return
            text = content[0].get("text", "")
            data = json.loads(text)
            prompt_id: str | None = data.get("prompt_id")
            client_id: str | None = data.get("client_id")
            if prompt_id:
                self._pending_prompt_id = prompt_id
                self._pending_client_id = client_id
                logger.info(
                    "ComfyUIInterruptHook: captured prompt_id=%s client_id=%s",
                    prompt_id, client_id,
                )
        except Exception as exc:
            logger.debug("ComfyUIInterruptHook: could not parse submit_prompt result: %s", exc)

    def _before_any_tool(self, event: BeforeToolCallEvent) -> None:
        """If a prompt_id is pending, raise an interrupt before the next tool
        call so the agent loop stops and hands control back to the orchestrator.

        On *resume* (when the orchestrator feeds back the ComfyUI result), this
        callback fires again.  ``event.interrupt()`` returns the stored response
        instead of raising, so we simply log and let the tool run normally —
        by that point the job is complete and the tool will get a fresh status.
        """
        if not self._pending_prompt_id:
            return

        prompt_id = self._pending_prompt_id
        client_id = self._pending_client_id or ""
        self._pending_prompt_id = None  # consume so we don't loop
        self._pending_client_id = None

        logger.info(
            "ComfyUIInterruptHook: interrupting before tool '%s'; waiting for prompt_id=%s",
            event.tool_use.get("name"),
            prompt_id,
        )

        # Encode both ids in the reason so the orchestrator can stream WS progress.
        reason = json.dumps({"prompt_id": prompt_id, "client_id": client_id})

        # event.interrupt() either:
        #   – raises InterruptException (first call → stops the agent loop), or
        #   – returns the resume response (subsequent call after orchestrator fed the result)
        response = event.interrupt(INTERRUPT_NAME, reason=reason)

        # If we reach here, the orchestrator has already polled and passed back
        # the ComfyUI result as the interrupt response.  The pending tool call
        # will execute normally and get the completed history.
        logger.info(
            "ComfyUIInterruptHook: resumed for prompt_id=%s; "
            "response length=%d chars",
            prompt_id,
            len(str(response)),
        )

    def _after_invocation(self, event: AfterInvocationEvent) -> None:  # noqa: ARG002
        """Safety cleanup: if a prompt_id was stored but the agent ended its
        turn without calling another tool (so ``_before_any_tool`` never fired),
        discard the stale prompt_id to avoid a spurious interrupt on the next
        message.
        """
        if self._pending_prompt_id:
            logger.warning(
                "ComfyUIInterruptHook: prompt_id=%s was stored but "
                "no BeforeToolCallEvent fired — discarding to avoid stale interrupt.",
                self._pending_prompt_id,
            )
            self._pending_prompt_id = None
            self._pending_client_id = None
