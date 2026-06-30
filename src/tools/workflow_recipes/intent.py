"""User-intent extraction - map a workflow to {media, task, model families}.

The downstream researcher needs to turn a request like "build a video workflow
using WAN 2.2" into a recipe. To make that possible, every workflow (and then
every type) carries a derived *intent*: what media it produces, what task it
performs, and which model family it uses.

Extraction is deterministic and rule-based from transparent, tunable vocab
tables (kept as module-level constants), using several signals per workflow:
  - the filename tokens (e.g. "video_wan2_2_14B_flf2v"),
  - the catalog title/description (index.json + config/workflow_templates.json),
  - the node roles present (reusing roles.classify_role),
  - model-loader widget filenames (e.g. "wan2.2_i2v_high_noise.safetensors").

``IntentClassifier`` wraps the rules; nothing here calls an LLM, it only reads
the corpus metadata.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import List, Optional

from .model import Intent, WorkflowGraph
from .roles import classify_role

# --------------------------------------------------------------------------- #
# Vocab tables (ordered, specific -> general; first match wins for task)
# --------------------------------------------------------------------------- #
# (label, [substrings]) - a label matches if any substring is in the search text.
_MODEL_FAMILIES: List[tuple] = [
    ("WAN VACE", ["vace"]),
    ("WAN 2.2", ["wan2_2", "wan 2.2", "wan_2_2", "wan22", "wan2.2"]),
    ("WAN 2.6", ["wan2_6", "wan 2.6", "wan_2_6", "wan26", "wan2.6"]),
    ("WAN", ["wan"]),
    ("LTX-2", ["ltx_2", "ltx2", "ltx-2", "ltxv", "ltx 2"]),
    ("LTX", ["ltx"]),
    ("Flux 2 Klein", ["flux2_klein", "flux 2 klein", "klein"]),
    ("Flux 2", ["flux_2", "flux 2", "flux2"]),
    ("Flux Krea", ["krea"]),
    ("Flux", ["flux"]),
    ("Qwen Image", ["qwen"]),
    ("Z-Image", ["z_image", "z-image", "zimage"]),
    ("Kling", ["kling"]),
    ("Nano-Banana", ["nano_banana", "nanobanana", "nano-banana"]),
    ("Veo", ["veo"]),
    ("Hunyuan3D", ["hunyuan3d", "hunyuan"]),
    ("SAM3", ["sam3", "sam_3"]),
    ("MoGe", ["moge"]),
    ("Ideogram", ["ideogram"]),
    ("Seedream", ["seedream"]),
    ("Topaz", ["topaz"]),
    ("Magnific", ["magnific"]),
    ("Meshy", ["meshy"]),
    ("ACE-Step", ["ace_step", "ace-step", "ace step"]),
    ("Stable Audio", ["stable_audio", "stable audio"]),
    ("Depth Anything", ["depth_anything", "depth anything"]),
    ("Lotus", ["lotus"]),
    ("Bernini", ["bernini"]),
    ("SCAIL", ["scail"]),
    ("ERNIE", ["ernie"]),
    ("Anima", ["anima"]),
    ("Lumina", ["lumina", "netayume"]),
    ("FireRed", ["firered"]),
    ("LongCat", ["longcat"]),
    ("Gemini", ["gemini"]),
    ("TripoSplat", ["triposplat"]),
    ("BiRefNet", ["birefnet"]),
    ("MediaPipe", ["mediapipe"]),
    ("SDPose", ["sdpose"]),
]

# When a specific variant matches, drop the generic family it implies.
_FAMILY_SUPPRESS = {
    "WAN VACE": ["WAN"], "WAN 2.2": ["WAN"], "WAN 2.6": ["WAN"],
    "LTX-2": ["LTX"],
    "Flux 2 Klein": ["Flux 2", "Flux"], "Flux 2": ["Flux"], "Flux Krea": ["Flux"],
}

_TASKS: List[tuple] = [
    ("first_last_frame_to_video", ["flf2v", "first_last_frame", "first last frame"]),
    ("image_to_video", ["image_to_video", "image-to-video", "i2v", "img2vid"]),
    ("text_to_video", ["text_to_video", "text-to-video", "t2v"]),
    ("video_edit", ["video_edit", "video editing", "vid2vid", "v2v", "video_to_video"]),
    ("video_inpaint", ["video_inpaint", "video inpaint", "video_inpainting"]),
    ("video_captioning", ["video_captioning"]),
    ("captioning", ["caption"]),
    ("prompt_enhance", ["prompt_enhance", "prompt enhance"]),
    ("motion_prompt", ["motionprompt", "motion prompt", "motion_prompt"]),
    ("geometry_estimation", ["geometry_estimation"]),
    ("depth_estimation", ["depth_estimation", "depth_anything", "to_depth", "depth_map", "to_depth_map"]),
    ("pose_estimation", ["pose_map", "to_pose", "sdpose", "pose_estimation"]),
    ("segmentation", ["segmentation", "sam3"]),
    ("remove_background", ["remove_background", "birefnet", "rmbg"]),
    ("3d_generation", ["to_model", "to_3d", "image_to_model", "text_to_model",
                        "gaussian_splat", "triposplat", "hunyuan3d", "to_gaussian_splat"]),
    ("frame_interpolation", ["frame_interpolation", "interpolat"]),
    ("relight", ["relight"]),
    ("character_sheet", ["charactersheet", "character_sheet"]),
    ("character_replacement", ["character_replacement"]),
    ("style_transfer", ["styletransfer", "style_transfer", "style transfer"]),
    ("outpaint", ["outpaint"]),
    ("inpaint", ["inpaint"]),
    ("upscale", ["upscale", "enhance", "super_resolution", "superres"]),
    ("controlnet", ["controlnet", "canny", "pose_to", "depth_to", "control"]),
    ("image_edit", ["image_edit", "image editing", "imageedit", "img2img",
                     "image-to-image", "image_to_image", "edit"]),
    ("text_to_image", ["text_to_image", "text-to-image", "t2i"]),
    ("audio_generation", ["audio_generation", "text_to_audio", "stable_audio", "ace_step", "audio"]),
    ("video_generation", ["video"]),
    ("image_generation", ["image"]),
]

# Task -> the media it produces (overrides structural guesses).
_TASK_MEDIA = {
    "captioning": "text", "video_captioning": "text", "prompt_enhance": "text",
    "motion_prompt": "text",
    "geometry_estimation": "3d", "3d_generation": "3d",
    "audio_generation": "audio",
    "first_last_frame_to_video": "video", "image_to_video": "video",
    "text_to_video": "video", "video_edit": "video", "video_inpaint": "video",
    "video_generation": "video", "frame_interpolation": "video",
}

# Phrases may use "{m}" (the media noun, e.g. "image"/"video") and "{am}"
# (article + noun, e.g. "a video") so media-neutral tasks read consistently with
# the workflow's actual output media.
_TASK_PHRASE = {
    "text_to_image": "generate an image from a text prompt",
    "image_generation": "generate an image",
    "image_edit": "edit an existing {m}",
    "inpaint": "inpaint masked regions of {am}",
    "outpaint": "outpaint / extend {am} beyond its borders",
    "upscale": "upscale / enhance {am}",
    "controlnet": "generate {am} guided by a control map (canny/depth/pose)",
    "style_transfer": "transfer a style onto {am}",
    "relight": "relight {am}",
    "character_sheet": "generate a multi-pose character sheet",
    "character_replacement": "replace a character in {am}",
    "image_to_video": "generate a video from an input image",
    "text_to_video": "generate a video from a text prompt",
    "first_last_frame_to_video": "generate a video interpolating between a first and last frame",
    "video_edit": "edit an existing video",
    "video_inpaint": "inpaint regions of a video",
    "video_generation": "generate a video",
    "frame_interpolation": "increase a video's frame rate via interpolation",
    "captioning": "caption an image as text",
    "video_captioning": "caption a video as text",
    "prompt_enhance": "expand a short prompt into a detailed one",
    "motion_prompt": "describe the motion in a video as text",
    "depth_estimation": "estimate a depth map",
    "geometry_estimation": "estimate 3D scene geometry",
    "pose_estimation": "estimate a pose map",
    "segmentation": "segment {am}",
    "remove_background": "remove the background from {am}",
    "3d_generation": "generate a 3D model",
    "audio_generation": "generate audio from a text prompt",
}

# When several members carry different (equally frequent) tasks, prefer the core
# generation task over a conditioning/modifier task for the type-level label.
_TASK_PRIORITY = [
    "image_to_video", "first_last_frame_to_video", "text_to_video",
    "video_generation", "text_to_image", "image_generation", "3d_generation",
    "audio_generation", "geometry_estimation",
    "image_edit", "video_edit", "video_inpaint", "inpaint", "outpaint",
    "upscale", "controlnet", "style_transfer", "relight",
    "character_replacement", "character_sheet", "segmentation",
    "remove_background", "depth_estimation", "pose_estimation",
    "frame_interpolation", "captioning", "video_captioning",
    "prompt_enhance", "motion_prompt",
]

_TOKEN_SPLIT = re.compile(r"[^a-z0-9.]+")


class IntentClassifier:
    """Derive a workflow's ``Intent`` (media / task / model families) and the
    natural-language phrasing used in the type-level matching surface.

    Stateless: the rules live in the module-level vocab tables, so one shared
    instance is enough and every call is deterministic."""

    # --------------------------------------------------------------------- #
    # Per-workflow classification
    # --------------------------------------------------------------------- #
    def classify(self, graph: WorkflowGraph) -> Intent:
        text = self._search_text(graph)
        task = self._task(text)
        media = self._media(graph, task)
        families = self._model_families(text)
        keywords = [t for t in _TOKEN_SPLIT.split(graph.name.lower()) if len(t) > 2]
        return Intent(media=media, task=task, model_families=families, keywords=keywords)

    def description_text(self, graph: WorkflowGraph) -> str:
        """The text used for description-based comparison: the authoritative
        catalog description (+ title + humanized filename). For the few workflows
        the catalog does not describe, a fallback intent phrase keeps them
        comparable instead of empty."""
        parts: List[str] = []
        if graph.index_description:
            parts.append(graph.index_description)
        if graph.index_title:
            parts.append(graph.index_title)
        parts.append(graph.name.replace("_", " "))
        if not graph.index_description:
            it = self.classify(graph)
            parts.append(self.when_to_use(it.media, it.task, it.model_families))
            parts.extend(it.model_families)
        return " ".join(parts)

    # --------------------------------------------------------------------- #
    # Signals
    # --------------------------------------------------------------------- #
    def _search_text(self, graph: WorkflowGraph) -> str:
        """Lowercased text to match vocab against: filename + catalog title/desc
        + model-loader widget filenames."""
        parts = [graph.name, graph.index_title or "", graph.index_description or ""]
        for node in graph.nodes.values():
            if classify_role(node.class_type) in ("model_loader", "lora_loader"):
                wv = node.widgets_values
                values = wv if isinstance(wv, list) else (wv.values() if isinstance(wv, dict) else [])
                parts.extend(str(v) for v in values if isinstance(v, str))
        # Match against two views so tokens hit regardless of separator style:
        #   joined - raw, keeps "_" (matches "text_to_image", "wan2_2", "wan2.2")
        #   norm   - "_"/"-" -> space (matches "wan 2.2")
        # A fully separator-stripped view is deliberately avoided: it would fuse
        # word boundaries (e.g. "an image" -> "animage", spuriously matching
        # "anima").
        joined = " ".join(parts).lower()
        norm = joined.replace("_", " ").replace("-", " ")
        return f"{norm} {joined}"

    @staticmethod
    def _model_families(text: str) -> List[str]:
        matched = []
        for label, subs in _MODEL_FAMILIES:
            if any(s in text for s in subs):
                matched.append(label)
        suppressed = set()
        for label in matched:
            suppressed.update(_FAMILY_SUPPRESS.get(label, []))
        # Preserve table order, drop suppressed generics and duplicates.
        out: List[str] = []
        for label, _subs in _MODEL_FAMILIES:
            if label in matched and label not in suppressed and label not in out:
                out.append(label)
        return out

    @staticmethod
    def _task(text: str) -> Optional[str]:
        for label, subs in _TASKS:
            if any(s in text for s in subs):
                return label
        return None

    @staticmethod
    def _media(graph: WorkflowGraph, task: Optional[str]) -> Optional[str]:
        if task and task in _TASK_MEDIA:
            return _TASK_MEDIA[task]
        classes = {n.class_type.lower() for n in graph.nodes.values()}
        if any("saveglb" in c or "splat" in c or "mesh" in c for c in classes):
            return "3d"
        if any("saveaudio" in c or c.startswith("audio") for c in classes):
            return "audio"
        if any("videocombine" in c or "createvideo" in c or "savevideo" in c for c in classes):
            return "video"
        if graph.media_type in ("image", "video", "audio"):
            return graph.media_type
        return "image"

    # --------------------------------------------------------------------- #
    # Type-level aggregation + phrasing
    # --------------------------------------------------------------------- #
    @staticmethod
    def dominant_task(tasks: List[str]) -> Optional[str]:
        """Most frequent task; ties broken by _TASK_PRIORITY (generation > modifier)."""
        if not tasks:
            return None
        counts = Counter(tasks)
        rank = {t: i for i, t in enumerate(_TASK_PRIORITY)}
        return sorted(counts.items(), key=lambda kv: (-kv[1], rank.get(kv[0], 999)))[0][0]

    @staticmethod
    def _article(word: str) -> str:
        return "an" if word[:1] in "aeiou" else "a"

    def task_phrase(self, task: Optional[str], media: Optional[str] = None) -> str:
        """Natural-language phrase for a task. Media-neutral tasks adapt to the
        actual output media so a video controlnet does not read as 'an image'."""
        phrase = _TASK_PHRASE.get(task or "", "run a node graph")
        if "{m}" in phrase or "{am}" in phrase:
            noun = media or "image"
            phrase = phrase.replace("{am}", f"{self._article(noun)} {noun}").replace("{m}", noun)
        return phrase

    def when_to_use(self, media: Optional[str], task: Optional[str], families: List[str]) -> str:
        phrase = self.task_phrase(task, media)
        fam = f" using {', '.join(families)}" if families else ""
        return f"Use to {phrase}{fam}."

    def example_requests(self, media: Optional[str], task: Optional[str], families: List[str]) -> List[str]:
        media = media or "image"
        out: List[str] = []
        if families:
            for fam in families:
                out.append(f"build {self._article(media)} {media} workflow using {fam}")
        else:
            out.append(f"build {self._article(media)} {media} workflow")
        tp = self.task_phrase(task, media)
        out.append(tp + (f" using {families[0]}" if families else ""))
        # Deterministic de-duplication preserving order.
        seen, deduped = set(), []
        for r in out:
            if r not in seen:
                seen.add(r)
                deduped.append(r)
        return deduped
