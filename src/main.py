#!/usr/bin/env python3
"""
agentY – main entry point.

Run with:
    python -m src.main
    python -m src.main --llm claude
    python -m src.main --llm ollama
"""

import argparse
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
from src.slack_server import start_slack_server  # noqa: E402


def main() -> None:
    """Launch the interactive agent loop."""
    parser = argparse.ArgumentParser(description="agentY – ComfyUI AI agent")
    parser.add_argument(
        "--llm",
        choices=["claude", "ollama"],
        default=None,
        help="LLM backend to use: 'claude' (default) or 'ollama'. "
             "Overrides the AGENT_LLM env var.",
    )
    args = parser.parse_args()

    api_key = os.environ.get("API_KEY_COMFY_ORG", "")
    if api_key:
        print("[agentY] ComfyUI API key loaded from API_KEY_COMFY_ORG.")
    else:
        print("[agentY] No API_KEY_COMFY_ORG set - using unauthenticated access.")

    hf_token = os.environ.get("HF_TOKEN", "")
    if hf_token:
        print("[agentY] Hugging Face token loaded from HF_TOKEN.")
    else:
        print("[agentY] No HF_TOKEN set - gated model downloads will fail.")

    slack_token = os.environ.get("SLACK_BOT_TOKEN", "")
    slack_member = os.environ.get("SLACK_MEMBER_ID", "")
    if slack_token and slack_member:
        print("[agentY] Slack integration enabled (SLACK_BOT_TOKEN + SLACK_MEMBER_ID loaded).")
    else:
        print("[agentY] Slack env vars missing - Slack tools will be unavailable.")

    agent = create_agent(llm=args.llm)

    # -- Start Slack Events API server + ngrok tunnel ------------------- #
    if slack_token and slack_member:
        events_url = start_slack_server(agent)
        if events_url:
            print(f"[agentY] Slack event listener active at {events_url}")
        else:
            print("[agentY] WARNING: Slack event server failed to start.")
            print("[agentY]   Ensure ngrok is installed and NGROK_AUTH_TOKEN is set in .env")
    else:
        print("[agentY] Skipping Slack event server (missing env vars).")

    print("\n=== agentY - ComfyUI Agent ===")
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
