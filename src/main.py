#!/usr/bin/env python3
"""
agentY – main entry point.

Pipeline mode (default) — Researcher resolves the spec, Brain assembles + runs it:
    python -m src.main
    python -m src.main --researcher-llm ollama --researcher-ollama-model qwen3-coder:32b
    python -m src.main --brain-llm claude --brain-anthropic-model claude-sonnet-4-5

Single-agent mode (legacy) — one model does everything:
    python -m src.main --mode single
    python -m src.main --mode single --llm claude
    python -m src.main --mode single --llm ollama --ollama-model llama3.2

Environment variable equivalents (all optional):
    AGENT_MODE                  pipeline | single          (default: single)
    RESEARCHER_LLM              ollama | claude            (default: ollama)
    RESEARCHER_OLLAMA_MODEL     model id                   (default: qwen3-coder:32b)
    RESEARCHER_ANTHROPIC_MODEL  model id
    BRAIN_LLM                   claude | ollama            (default: claude)
    BRAIN_ANTHROPIC_MODEL       model id
    BRAIN_OLLAMA_MODEL          model id
    AGENT_LLM                   claude | ollama            (single-agent mode)
    OLLAMA_MODEL                model id                   (single-agent mode)
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

from src.agent import create_agent, _cfg  # noqa: E402
from src.pipeline import create_pipeline  # noqa: E402
from src.slack_server import start_slack_server  # noqa: E402


def main() -> None:
    """Launch the interactive agent loop."""
    parser = argparse.ArgumentParser(
        description="agentY – ComfyUI AI agent",
        formatter_class=argparse.RawTextHelpFormatter,
    )

    # ── Mode selection ─────────────────────────────────────────────────── #
    parser.add_argument(
        "--mode",
        choices=["pipeline", "single"],
        default=None,
        help=(
            "Execution mode:\n"
            "  pipeline  Researcher (Ollama) → Brain (Claude) [default]\n"
            "  single    Legacy single-agent mode"
        ),
    )

    # ── Pipeline: Researcher overrides ────────────────────────────────── #
    pipeline_group = parser.add_argument_group("Pipeline – Researcher agent")
    pipeline_group.add_argument(
        "--researcher-llm",
        choices=["ollama", "claude"],
        default=None,
        metavar="BACKEND",
        help="LLM backend for the Researcher (default: ollama / RESEARCHER_LLM env).",
    )
    pipeline_group.add_argument(
        "--researcher-ollama-model",
        default=None,
        metavar="MODEL",
        help="Ollama model for the Researcher (default: qwen3-coder:32b).",
    )
    pipeline_group.add_argument(
        "--researcher-anthropic-model",
        default=None,
        metavar="MODEL",
        help="Anthropic model for the Researcher when --researcher-llm=claude.",
    )

    # ── Pipeline: Brain overrides ──────────────────────────────────────── #
    brain_group = parser.add_argument_group("Pipeline – Brain agent")
    brain_group.add_argument(
        "--brain-llm",
        choices=["claude", "ollama"],
        default=None,
        metavar="BACKEND",
        help="LLM backend for the Brain (default: claude / BRAIN_LLM env).",
    )
    brain_group.add_argument(
        "--brain-anthropic-model",
        default=None,
        metavar="MODEL",
        help="Anthropic model for the Brain (e.g. claude-sonnet-4-5).",
    )
    brain_group.add_argument(
        "--brain-ollama-model",
        default=None,
        metavar="MODEL",
        help="Ollama model for the Brain when --brain-llm=ollama.",
    )

    # ── Single-agent (legacy) overrides ───────────────────────────────── #
    single_group = parser.add_argument_group("Single-agent mode (--mode single)")
    single_group.add_argument(
        "--llm",
        choices=["claude", "ollama"],
        default=None,
        help="LLM backend for single-agent mode.",
    )
    single_group.add_argument(
        "--ollama-model",
        default=None,
        metavar="MODEL",
        help="Ollama model for single-agent mode.",
    )

    args = parser.parse_args()

    # Infer mode: if any single-agent flag is set, default to single.
    mode = args.mode
    if mode is None:
        if args.llm or args.ollama_model:
            mode = "single"
        else:
            mode = str(_cfg("AGENT_MODE", "agent_mode", default="single"))

    # ── Environment checks ─────────────────────────────────────────────── #
    api_key = os.environ.get("API_KEY_COMFY_ORG", "")
    print(
        "[agentY] ComfyUI API key loaded." if api_key
        else "[agentY] No API_KEY_COMFY_ORG set - using unauthenticated access."
    )

    hf_token = os.environ.get("HF_TOKEN", "")
    if not hf_token:
        print("[agentY] No HF_TOKEN set - gated model downloads will fail.")

    slack_token = os.environ.get("SLACK_BOT_TOKEN", "")
    slack_member = os.environ.get("SLACK_MEMBER_ID", "")
    if slack_token and slack_member:
        print("[agentY] Slack integration enabled.")
    else:
        print("[agentY] Slack env vars missing - Slack tools will be unavailable.")

    # ── Build callable agent / pipeline ───────────────────────────────── #
    if mode == "single":
        # Legacy: --ollama-model implies ollama backend
        if args.ollama_model and args.llm is None:
            args.llm = "ollama"
        agent = create_agent(llm=args.llm, ollama_model=args.ollama_model)
        print("[agentY] Mode: single-agent")
    else:
        agent = create_pipeline(
            researcher_llm=args.researcher_llm,
            researcher_ollama_model=args.researcher_ollama_model,
            researcher_anthropic_model=args.researcher_anthropic_model,
            brain_llm=args.brain_llm,
            brain_anthropic_model=args.brain_anthropic_model,
            brain_ollama_model=args.brain_ollama_model,
        )
        print("[agentY] Mode: pipeline (Researcher → Brain)")

    # ── Start Slack Events API server + ngrok tunnel ───────────────────── #
    if slack_token and slack_member:
        events_url = start_slack_server(agent)
        if events_url:
            print(f"[agentY] Slack event listener active at {events_url}")
            print("[agentY] Slack server logs → output/slack_server.log")
        else:
            print("[agentY] WARNING: Slack event server failed to start.")
            print("[agentY]   Ensure ngrok is installed and NGROK_AUTH_TOKEN is set.")
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

        # Display token usage in the shell
        try:
            usage = agent.event_loop_metrics.accumulated_usage
            in_tok = usage.get("inputTokens", 0)
            out_tok = usage.get("outputTokens", 0)
            cache_read = usage.get("cacheReadInputTokens", 0)
            cache_write = usage.get("cacheWriteInputTokens", 0)
            parts = [f"{in_tok:,} in", f"{out_tok:,} out"]
            if cache_read:
                parts.append(f"{cache_read:,} cache hit")
            if cache_write:
                parts.append(f"{cache_write:,} cache write")
            print(f"🪙 Tokens: {' / '.join(parts)}\n")
        except Exception:
            pass


if __name__ == "__main__":
    main()
