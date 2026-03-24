"""Pydantic data models for the AgentY creative pipeline.

Defines the structured output schemas used by the AI agents to ensure
type-safe, validated responses at each stage of the pipeline.
"""

from pydantic import BaseModel, Field


class PromptOutput(BaseModel):
    """Structured output from the prompter agent.

    Contains the final, production-ready image-generation prompt
    synthesised from the user's brief, mood references, and instructions.
    """

    prompt: str = Field(..., description="A production-ready creative prompt.")


class SupervisionOutput(BaseModel):
    """Structured output from the supervisor agent's quality review.

    Captures the accept/reject decision and a brief justification
    explaining whether the generated image satisfies the original brief.
    """

    accepted: bool = Field(
        ..., description="Whether the result should be accepted."
    )
    supervision: str = Field(
        ..., description="A short review explaining the decision."
    )


class FinalResult(BaseModel):
    """Aggregate result returned by the orchestrator at the end of the pipeline.

    Combines the supervisor's verdict, the generated image path,
    the prompt that was used, and the original brief text.
    """

    accepted: bool
    supervision: str
    output_image: str | None = None
    prompt: str
    brief: str
