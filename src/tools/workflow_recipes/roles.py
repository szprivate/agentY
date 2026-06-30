"""Functional role classification for node classes.

A node's *role* is a coarse functional label (sampler, vae_decode, ...) derived
from its class name. Roles let connection patterns and recipe structure match
across different-but-equivalent classes (two samplers count the same), while a
node with no known role keeps its raw class name so custom nodes stay
distinguishable.

The rules are intentionally transparent and rule-based: an ordered list of
(role, description, predicate); first match wins. ``is_utility`` is a separate,
orthogonal flag for plumbing nodes (primitives, math, switches) that may repeat
but carry no functional intent - it does not affect role classification.
"""

from __future__ import annotations

from typing import FrozenSet, List, Tuple


def _has(*subs: str):
    return lambda c: any(s in c for s in subs)


# Each rule: (role, human description, predicate over lowercased class name).
_ROLE_RULES: List[Tuple[str, str, object]] = [
    ("sampler", "diffusion sampler / denoiser", _has("ksampler", "samplercustom", "sampler")),
    ("model_loader", "diffusion model / UNET loader", _has("unetloader", "checkpointloader", "diffusionmodel", "wanmodel", "modelloader", "load diffusion")),
    ("lora_loader", "LoRA / model patch loader", _has("loraloader", "modelpatch", "lora")),
    ("clip_loader", "text encoder / CLIP loader", _has("cliploader", "dualcliploader", "tripleclip")),
    ("text_encode", "prompt text encoding", _has("textencode", "cliptextencode", "encodeprompt")),
    ("vae_loader", "VAE loader", _has("vaeloader",)),
    ("vae_decode", "latent -> pixel decode", _has("vaedecode",)),
    ("vae_encode", "pixel -> latent encode", _has("vaeencode",)),
    ("latent_source", "empty latent / canvas", _has("emptylatent", "emptysd3", "emptyimage", "emptylatentvideo")),
    ("controlnet", "controlnet / guidance conditioning", _has("controlnet", "control_net")),
    ("upscale", "upscale / resize", _has("upscale", "scaleimage", "imagescale")),
    ("image_loader", "image input / load", _has("loadimage", "loadimagemask")),
    ("video_loader", "video input / load", _has("loadvideo", "vhs_loadvideo")),
    ("conditioning_op", "conditioning combine / edit", _has("conditioning",)),
    ("guidance", "guider / sigma / scheduler", _has("guider", "basicscheduler", "sigmas", "fluxguidance")),
    ("save_output", "save / preview / combine output", _has("saveimage", "previewimage", "savevideo", "vhs_videocombine", "saveaudio", "savelatent")),
    ("api_node", "external API generation node", _has("klingo", "veo", "magnific", "topaz", "meshy", "ideogram", "seedream", "nanobanana", "gemini")),
]

# Roles that act as the structural "spine" of a generation graph.
SPINE_ROLES: FrozenSet[str] = frozenset(
    {"sampler", "model_loader", "vae_decode", "vae_encode", "text_encode",
     "latent_source", "api_node"}
)

# Plumbing nodes that are frequently present in multiples but carry no functional
# intent (primitives, math, switches, reroutes, string/json ops). Used only to
# keep the "paired/multiple required" emphasis on meaningful nodes.
_UTILITY_SUBSTRINGS = (
    "primitive", "reroute", "comfymathexpression", "comfynumberconvert",
    "comfyswitchnode", "mathint", "mathfloat", "getimagesize",
    "stringreplace", "stringconcatenate", "jsonextractstring", "previewany",
)


def classify_role(class_type: str) -> str:
    """Map a node class to a coarse functional role. Returns "other" if no rule
    matches (custom/unknown nodes still get grouped by class downstream)."""
    name = (class_type or "").lower()
    for role, _desc, pred in _ROLE_RULES:
        if pred(name):
            return role
    return "other"


def role_description(role: str) -> str:
    for r, desc, _pred in _ROLE_RULES:
        if r == role:
            return desc
    return "unclassified node role"


def role_or_class(class_type: str) -> str:
    """Functional role when known, else the raw class name (keeps custom nodes
    distinguishable instead of bucketing them all as 'other')."""
    role = classify_role(class_type)
    return role if role != "other" else class_type


def is_utility(class_type: str) -> bool:
    name = (class_type or "").lower()
    return any(s in name for s in _UTILITY_SUBSTRINGS)
