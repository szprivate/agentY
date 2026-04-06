"""
agentY – Shared domain models.

Pydantic models used across the agent pipeline.  Centralised here to avoid
circular imports between modules that both define and consume these types.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class MessageIntent(str, Enum):
    param_tweak  = "param_tweak"   # adjust a param of the last run
    chain        = "chain"          # pipe last output into a new workflow
    feedback     = "feedback"       # qualitative feedback / correction on the generated output
    new_request  = "new_request"    # fresh generation request
    info_query   = "info_query"     # question about capabilities / workflows / models
    needs_image  = "needs_image"    # request requires an image input that the user forgot to attach


class ChatSummary(BaseModel):
    workflow_name: str
    output_paths: list[str]
    user_intent: str
    status: str


class AgentSession(BaseModel):
    session_id: str
    chat_summaries: list[ChatSummary] = Field(default_factory=list)
    current_output_paths: list[str] = Field(default_factory=list)
    follow_up_count: int = 0


class TriageResult(BaseModel):
    intent: MessageIntent
    response: str | None = None   # populated only for info_query
    confidence: float
