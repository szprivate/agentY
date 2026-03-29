#!/usr/bin/env python3
"""
agentY – main entry point.

Run with:
    python -m src.main

Or:
    python src/main.py
"""

import os
import sys

from dotenv import load_dotenv

# Ensure project root is on sys.path when run as a script
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

# Load .env from project root
load_dotenv(os.path.join(_project_root, ".env"))

from src.agent import create_agent  # noqa: E402


def main() -> None:
    """Launch the interactive agent loop."""
    api_key = os.environ.get("API_KEY_COMFY_ORG", "")
    if api_key:
        print("[agentY] ComfyUI API key loaded from API_KEY_COMFY_ORG.")
    else:
        print("[agentY] No API_KEY_COMFY_ORG set – using unauthenticated access.")

    hf_token = os.environ.get("HF_TOKEN", "")
    if hf_token:
        print("[agentY] Hugging Face token loaded from HF_TOKEN.")
    else:
        print("[agentY] No HF_TOKEN set – gated model downloads will fail.")

    slack_token = os.environ.get("SLACK_BOT_TOKEN", "")
    slack_member = os.environ.get("SLACK_MEMBER_ID", "")
    if slack_token and slack_member:
        print("[agentY] Slack integration enabled (SLACK_BOT_TOKEN + SLACK_MEMBER_ID loaded).")
    else:
        print("[agentY] Slack env vars missing – Slack tools will be unavailable.")

    agent = create_agent()

    print("\n=== agentY – ComfyUI Agent ===")
    print("Type your message (or 'quit' / 'exit' to stop).\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break

        response = agent(user_input)
        print(f"\nagentY: {response}\n")


if __name__ == "__main__":
    main()
