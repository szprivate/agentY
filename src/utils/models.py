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
    batch_request       = "batch_request"        # same workflow N times with varied parameters
    info_query          = "info_query"           # question about capabilities / workflows / models
    needs_image         = "needs_image"          # request requires an image input that the user forgot to attach
    chat                = "chat"                 # casual/conversational message with no generation intent
    story               = "story"                # request to write a small storyline / narrative
    storyboard          = "storyboard"           # end-to-end short-film: character sheet → shots → Kling multishot video sequences


class ChatSummary(BaseModel):
    workflow_name: str
    output_paths: list[str]
    user_intent: str
    status: str


class GeneratedImage(BaseModel):
    """A single image generated earlier in the current thread.

    Forms an ordered, user-referenceable gallery so the user can say
    "image 2", "the last image", or describe it ("the lighthouse one") and
    the agent can resolve the reference back to the real file on disk.
    """
    index: int                 # 1-based position in the thread gallery
    path: str                  # resolved path of the generated file on disk
    caption: str = ""          # short description (positive prompt / template)
    turn: int = 0              # turn number (len(chat_summaries)+1) when produced


class AgentSession(BaseModel):
    session_id: str
    chat_summaries: list[ChatSummary] = Field(default_factory=list)
    current_output_paths: list[str] = Field(default_factory=list)
    follow_up_count: int = 0
    last_agent: str = "assemble_workflow"  # "assemble_workflow" | "info" | "query_templates" — tracks which agent handled the most recent turn
    last_researcher_request: str | None = None  # original user text stored when query_templates returned status=blocked
    last_researcher_blockers: list[str] = Field(default_factory=list)  # blocker strings from the last blocked brainbriefing
    last_user_input_images: list[str] = Field(default_factory=list)  # paths of images uploaded by the user, persisted across turns
    last_info_response: str | None = None  # last response from the Info agent (e.g. a crafted prompt), injected into the Query Templates when relevant
    last_story_response: str | None = None  # last text the Story agent produced (synopsis or scenes), injected into the next story turn for Mode A→B handoff and refinements
    generated_images: list[GeneratedImage] = Field(default_factory=list)  # ordered gallery of images generated in this thread, referenceable by index/description
    plan_awaiting_reply: bool = False  # a plan was put to the user for approval and they haven't answered yet; their next message opens the execution gate for one turn
    # A hook chain stopped at a `review` hook so the user can choose what goes on
    # to the next stage. Only the FLAG lives here — which hook, which collector
    # node, and what it held at the halt (for the message, and the log). The
    # answer is whatever that collector holds when they reply, read off the canvas
    # at resume: they are expected to edit it while this is up, and a cached list
    # is a list that can be wrong by then.
    review_halt: dict | None = None


class TriageResult(BaseModel):
    intent: MessageIntent
    response: str | None = None   # populated only for info_query
    confidence: float
    run_qa: bool = False          # True only when user explicitly requests a QA pass
