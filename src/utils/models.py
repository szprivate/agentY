"""
agentY – Shared domain models.

Pydantic models used across the agent pipeline.  Centralised here to avoid
circular imports between modules that both define and consume these types.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class MessageIntent(str, Enum):
    param_tweak         = "param_tweak"          # adjust a param of the last run
    chain               = "chain"                # pipe last output into a new workflow
    feedback            = "feedback"             # qualitative feedback / correction on the generated output
    new_request         = "new_request"          # fresh generation request
    new_planned_request = "new_planned_request"  # multi-step generation plan (several consecutive tasks)
    info_query          = "info_query"           # question about capabilities / workflows / models
    needs_image         = "needs_image"          # request requires an image input that the user forgot to attach


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
    last_agent: str = "brain"  # "brain" | "info" | "researcher" — tracks which agent handled the most recent turn
    last_researcher_request: str | None = None  # original user text stored when researcher returned status=blocked
    last_researcher_blockers: list[str] = Field(default_factory=list)  # blocker strings from the last blocked brainbriefing
    last_user_input_images: list[str] = Field(default_factory=list)  # paths of images uploaded by the user, persisted across turns
    last_info_response: str | None = None  # last response from the Info agent (e.g. a crafted prompt), injected into the Researcher when relevant


class TriageResult(BaseModel):
    intent: MessageIntent
    response: str | None = None   # populated only for info_query
    confidence: float
    run_qa: bool = False          # True only when user explicitly requests a QA pass
