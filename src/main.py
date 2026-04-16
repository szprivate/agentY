#!/usr/bin/env python3
"""
agentY – main entry point.

Pipeline mode — Researcher resolves the spec, Brain assembles + runs it:
    python -m src.main
    python -m src.main --researcher-llm ollama --researcher-ollama-model qwen3-coder:32b
    python -m src.main --brain-llm claude --brain-anthropic-model claude-sonnet-4-5

Environment variable equivalents (all optional):
    RESEARCHER_LLM              ollama | claude            (default: ollama)
    RESEARCHER_OLLAMA_MODEL     model id                   (default: qwen3-coder:32b)
    RESEARCHER_ANTHROPIC_MODEL  model id
    BRAIN_LLM                   claude | ollama            (default: claude)
    BRAIN_ANTHROPIC_MODEL       model id
    BRAIN_OLLAMA_MODEL          model id
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

from src.pipeline import create_pipeline  # noqa: E402
from src.utils.secrets import get_secret  # noqa: E402
from src.tools.agent_control import is_restart_command, restart_process, is_unload_command, unload_ollama_models  # noqa: E402
from src.utils.costs import compute_cost_from_usage  # noqa: E402


def main() -> None:
    """Launch the interactive agent loop."""
    parser = argparse.ArgumentParser(
        description="agentY – ComfyUI AI agent",
        formatter_class=argparse.RawTextHelpFormatter,
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
    brain_group.add_argument(
        "--skip-brain",
        action="store_true",
        default=False,
        help="Pipeline mode only: return Researcher output directly and skip Brain stage.",
    )


    args = parser.parse_args()

    # ── Environment checks ─────────────────────────────────────────────── #
    api_key = get_secret("COMFYUI_API_KEY")
    print(
        "[agentY] ComfyUI API key loaded." if api_key
        else "[agentY] No COMFYUI_API_KEY set - using unauthenticated access."
    )

    hf_token = os.environ.get("HF_TOKEN", "")
    if not hf_token:
        print("[agentY] No HF_TOKEN set - gated model downloads will fail.")


    # ── Build callable agent / pipeline ───────────────────────────────── #
    agent = create_pipeline(
        researcher_llm=args.researcher_llm,
        researcher_ollama_model=args.researcher_ollama_model,
        researcher_anthropic_model=args.researcher_anthropic_model,
        brain_llm=args.brain_llm,
        brain_anthropic_model=args.brain_anthropic_model,
        brain_ollama_model=args.brain_ollama_model,
        skip_brain=args.skip_brain,
    )
    print("[agentY] Mode: pipeline (Researcher → Brain)")
    if args.skip_brain:
        print("[agentY] SkipBrain is activated: Brain stage will be bypassed and Researcher output will be returned.")


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
        if is_restart_command(user_input):
            print("[agentY] Restarting...")
            restart_process()
            break  # restart_process replaces the process; break is a safety net
        if is_unload_command(user_input):
            print("[agentY] Unloading Ollama models from VRAM...")
            unloaded = unload_ollama_models()
            if unloaded:
                print(f"[agentY] Unloaded: {', '.join(unloaded)}")
            else:
                print("[agentY] No models were unloaded (Ollama unreachable or none loaded).")
            continue

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
            try:
                if hasattr(agent, "compute_turn_cost"):
                    cost_val, total_tokens = agent.compute_turn_cost()
                else:
                    cost_val, total_tokens = compute_cost_from_usage(usage, agent)
                print(f"💵 Cost: ${cost_val:.2f} (total {total_tokens:,} tokens)\n")
            except Exception:
                pass
        except Exception:
            pass


if __name__ == "__main__":
    main()
