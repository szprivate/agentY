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


class WorkflowSelectionOutput(BaseModel):
    """Structured output from the workflow-selection step.

    The selected workflow is expressed as a workflow identifier rather than a
    full path. Local workflows use their file name, while remote ComfyUI
    templates use the upstream template name from ``templates/index.json``.
    """

    workflow_name: str = Field(
        ...,
        description="The exact workflow identifier chosen for this brief.",
    )
    rationale: str = Field(
        ...,
        description="Short explanation of why this workflow best matches the brief.",
    )


class WorkflowSelectionDetails(BaseModel):
    """Verbose record of the workflow-selection step."""

    workflow_name: str
    workflow_file: str
    rationale: str


class ExecutionPlanStep(BaseModel):
    """One executable step in the orchestration plan."""

    step_number: int = Field(..., description="1-based sequence number.")
    title: str = Field(..., description="Short human-readable step title.")
    brief: str = Field(
        ...,
        description=(
            "Self-contained creative instruction for this step only. "
            "Later steps may assume they receive previous step outputs as input."
        ),
    )


class ExecutionPlanOutput(BaseModel):
    """Structured orchestration plan derived from the original brief."""

    summary: str = Field(
        ...,
        description="Short explanation of the overall multi-step plan.",
    )
    steps: list[ExecutionPlanStep] = Field(
        ...,
        description="Ordered list of executable steps.",
    )


class PromptDetails(BaseModel):
    """Verbose record of the prompt-generation step."""

    prompt: str


class SupervisionDetails(BaseModel):
    """Verbose record of the supervisor review step."""

    accepted: bool
    verdict: str


class StepTrace(BaseModel):
    """Verbose execution record for one planned step."""

    step_number: int
    title: str
    brief: str
    input_images: list[str]
    output_image: str | None = None
    output_files: list[str] = Field(default_factory=list)
    workflow_selection: WorkflowSelectionDetails
    prompt_generation: PromptDetails
    supervision: SupervisionDetails


class OrchestrationTrace(BaseModel):
    """Human-readable trace of the orchestrator's choices.

    This captures user-visible decisions rather than hidden reasoning.
    """

    summary: str
    plan_summary: str
    workflow_selection: WorkflowSelectionDetails
    prompt_generation: PromptDetails
    supervision: SupervisionDetails
    steps: list[StepTrace] = Field(default_factory=list)


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
    output_files: list[str] = Field(default_factory=list)
    prompt: str
    brief: str
    workflow_name: str
    workflow_file: str
    workflow_rationale: str
    plan_summary: str
    step_count: int
    orchestration_summary: str
    trace: OrchestrationTrace
