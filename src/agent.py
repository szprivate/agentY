"""
agentY – A ComfyUI agent built on the Strands Agents SDK.

This module configures and exposes the Strands Agent instance with all
ComfyUI tools registered.
"""

import os

from strands import Agent
from strands.models.anthropic import AnthropicModel as _BaseAnthropicModel
from strands.models.ollama import OllamaModel
from strands.agent.conversation_manager import SlidingWindowConversationManager

from src.tools import ALL_TOOLS


class AnthropicModel(_BaseAnthropicModel):
    """AnthropicModel with cache_control injected on the last tool.

    This causes Anthropic to cache the entire tools block on every request,
    reducing cached-token cost to 10 % of the normal input price after the
    first call (which pays the 1.25× cache-write surcharge).
    """

    def format_request(self, messages, tool_specs=None, system_prompt=None, tool_choice=None):  # type: ignore[override]
        req = super().format_request(messages, tool_specs, system_prompt, tool_choice)
        if req.get("tools"):
            *head, last = req["tools"]
            req["tools"] = head + [{**last, "cache_control": {"type": "ephemeral"}}]
        return req

SYSTEM_PROMPT = """\
You are agentY, a ComfyUI workflow agent. Construct and execute ComfyUI workflows
via the available tools. Follow the standards below unless told otherwise.

## Models
Use paths below directly. Only call get_model_types() or get_models_in_folder()
for models not listed here. Never guess a path.

UNETs: flux1-dev-fp8 → FLUX1/flux1-dev-fp8.safetensors | flux1-kontext →
FLUX1/flux1-dev-kontext_fp8_scaled.safetensors | wan21-i2v-720p →
WAN21/Wan2_1-I2V-14B-720P_fp8_e4m3fn.safetensors | wan22-i2v-high →
WAN22/wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors | wan22-i2v-low →
WAN22/wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors
VAE: flux → FLUX1/ae.safetensors | wan21 → WAN21/Wan2_1_VAE_bf16.safetensors |
wan22 → WAN22/wan2.2_vae.safetensors
CLIP (Flux): Flux-Dev/t5xxl_fp16.safetensors + Flux-Dev/clip_l.safetensors, type=flux

Full model list in settings.json — load it when a model above is not sufficient.

## Workflow standards
- ALWAYS ask for SequenceName and ShotName if not provided before doing anything.
- ALWAYS create bepicSetPath (path_id="claude_01234") with SequenceName/ShotName.
- ALWAYS load images via VHS_LoadImagePath, videos via VHS_LoadVideoPath.
- ALWAYS call upload_image() with base64 + filename BEFORE building workflow.
- ALWAYS save images with SaveImage (PNG), videos with VHS_VideoCombine (mp4).
- ALWAYS connect bEpicGetPath (with path_id="claude_01234", path_key=pathImages or pathVideo,
  suffix=descriptive name) to every SaveImage / VHS_VideoCombine filename_prefix.
- API format only. Search templates first, scaffold with get_workflow_template(),
  modify minimally. Validate before queuing. Track and report results.

## Node defaults
- GeminiNanoBananaPro: resolution="1K", thinking_level="MINIMAL",
  model="gemini-3-pro-image-preview", response_modalities="IMAGE", aspect_ratio="16:9"
- GeminiNanoBanana2: resolution="1K", thinking_level="MINIMAL",
  model="Nano Banana 2 (Gemini 3.1 Flash Image)", response_modalities="IMAGE", aspect_ratio="16:9"
- ModelSamplingFlux: max_shift=1.15, base_shift=0.5, explicit width+height required.

## Hugging Face
1. Identify exact file via search_huggingface_models() or get_model_info().
2. check_local_model(filename) — if found, use it and stop.
3. Only if not found: download_hf_model() to correct folder.

## Slack
You are ALWAYS running inside a Slack DM. Every response is displayed in Slack.
Slack CANNOT render local file paths or base64 data URIs — they appear as broken
text. You MUST upload every generated image/video via the tools below.

After every generation, WITHOUT asking the user, immediately:
1. Call view_image(filename=..., save_to="./output/<filename>") to download the
   file to disk. NEVER omit save_to.
2. Call slack_send_image(file_path="./output/<filename>") to upload it.

NEVER write markdown image syntax ![...](...)  — it does not work in Slack.
NEVER include base64 or data URIs in your replies.
NEVER ask "would you like me to send it to Slack?" — just send it.

## Test mode
When user says "test mode": skip all slack_send_*, skip bEpicSendToViewer,
use 512×512.

Be concise. Ask when ambiguous. Report errors clearly.
"""


def _build_model(llm: str, ollama_model: str | None = None):
    """Instantiate the requested LLM backend.

    Args:
        llm: ``'claude'`` (default) or ``'ollama'``.
        ollama_model: Optional Ollama model name override. Takes precedence
                      over the ``OLLAMA_MODEL`` env var when provided.
    """
    llm = llm.strip().lower()
    if llm == "ollama":
        model_id = ollama_model or os.environ.get("OLLAMA_MODEL", "qwen3-vl:30b")
        return OllamaModel(
            host=os.environ.get("OLLAMA_HOST", "http://localhost:11434"),
            model_id=model_id,
        )
    # Default: claude
    # Pass system_prompt as a structured content block so Anthropic's
    # prompt-caching kicks in (cache_control="ephemeral"). params is
    # expanded last in AnthropicModel.format_request, so it overrides
    # the plain-string "system" key that Strands sets from system_prompt.
    return AnthropicModel(
        model_id=os.environ.get("ANTHROPIC_MODEL", "claude-haiku-4-5"),
        max_tokens=int(os.environ.get("ANTHROPIC_MAX_TOKENS", "4096")),
        params={
            "system": [
                {
                    "type": "text",
                    "text": SYSTEM_PROMPT,
                    "cache_control": {"type": "ephemeral"},
                }
            ]
        },
    )


def create_agent(llm: str | None = None, ollama_model: str | None = None, **kwargs) -> Agent:
    """Create and return the agentY Strands Agent with all ComfyUI tools.

    Args:
        llm: Which LLM backend to use: ``'claude'`` (default) or ``'ollama'``.
             Falls back to the ``AGENT_LLM`` env var, then to ``'claude'``.
        ollama_model: Ollama model name to use when ``llm='ollama'``. Overrides
                      the ``OLLAMA_MODEL`` env var when provided.
        **kwargs: Extra keyword arguments forwarded to the Strands Agent
                  constructor (e.g. to override the model or system prompt).
    """
    resolved_llm = llm or os.environ.get("AGENT_LLM", "claude")
    model = _build_model(resolved_llm, ollama_model=ollama_model)
    print(f"[agentY] Using LLM backend: {resolved_llm} ({model.__class__.__name__})")
    # Limit conversation history to the last 40 messages (≈20 turns).
    # This prevents costs from compounding as history grows across long sessions.
    window_size = int(os.environ.get("AGENT_HISTORY_WINDOW", "40"))
    agent_kwargs = {
        "model": model,
        "system_prompt": SYSTEM_PROMPT,
        "tools": ALL_TOOLS,
        "conversation_manager": SlidingWindowConversationManager(window_size=window_size),
    }
    agent_kwargs.update(kwargs)
    return Agent(**agent_kwargs)
