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
        print(json.dumps(result, indent=2, ensure_ascii=False))
    finally:
        # Always release GPU memory, even if the pipeline crashes
        pipeline.unload_llm()


if __name__ == "__main__":
    main()
