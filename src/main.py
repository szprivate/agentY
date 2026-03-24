"""AgentY - AI-driven creative image generation pipeline.

Entry point for the application. Loads configuration, builds the
multi-agent pipeline, and runs a single generation cycle based on
the briefing text defined in ``config/settings.json``.

Usage::

    cd src
    python main.py
"""

import json

from agents import CreativePipeline
from config import AppConfig


def print_verbose_result(result: dict) -> None:
    """Print a human-friendly summary of the pipeline output."""
    trace = result.get("trace", {})
    steps = trace.get("steps", [])

    print("=" * 80)
    print("ORCHESTRATION SUMMARY")
    print("=" * 80)
    print(result.get("orchestration_summary", ""))
    print()

    print("PLAN")
    print("-" * 80)
    print(f"Steps:   {result.get('step_count', len(steps))}")
    print(f"Summary: {result.get('plan_summary', trace.get('plan_summary', ''))}")
    print()

    for step in steps:
        workflow = step.get("workflow_selection", {})
        prompt_data = step.get("prompt_generation", {})
        supervision = step.get("supervision", {})

        print(f"STEP {step.get('step_number', '?')}: {step.get('title', '')}")
        print("-" * 80)
        print("Brief:")
        print(step.get("brief", ""))
        print()
        print("Inputs:")
        input_images = step.get("input_images", [])
        if input_images:
            for image in input_images:
                print(f"- {image}")
        else:
            print("- None")
        print()
        print("Workflow selection:")
        print(f"- Workflow: {workflow.get('workflow_name', '')}")
        print(f"- File:     {workflow.get('workflow_file', '')}")
        print(f"- Reason:   {workflow.get('rationale', '')}")
        print()
        print("Generated prompt:")
        print(prompt_data.get("prompt", ""))
        print()
        print("Supervisor verdict:")
        print(f"- Accepted: {supervision.get('accepted', False)}")
        print(f"- Verdict:  {supervision.get('verdict', '')}")
        print()
        print("Outputs:")
        print(f"- Preview:  {step.get('output_image', '')}")
        output_files = step.get("output_files", [])
        if output_files:
            for output_file in output_files:
                print(f"- File:     {output_file}")
        print()

    print("FINAL OUTPUT IMAGE")
    print("-" * 80)
    print(result.get("output_image", ""))
    print()

    print("FINAL SUPERVISION")
    print("-" * 80)
    print(f"Accepted: {result.get('accepted', False)}")
    print(result.get("supervision", ""))
    print()


def main() -> None:
    """Load the brief and execute the creative generation pipeline."""
    config = AppConfig()
    pipeline = CreativePipeline(config)

    brief = config.briefing_text
    if not brief:
        print(
            "No briefing text found. "
            "Check the prompts.briefing path in settings.json."
        )
        return

    try:
        result = pipeline.run(brief=brief)
        print_verbose_result(result)
    finally:
        # Always release GPU memory, even if the pipeline crashes
        pipeline.unload_llm()


if __name__ == "__main__":
    main()
