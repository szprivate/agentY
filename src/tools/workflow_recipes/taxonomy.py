"""Canonical workflow taxonomy - classify each workflow into the task category a
user actually thinks in, matching ComfyUI's own template taxonomy
(https://comfy.org/workflows): Text to Image, Image Edit, Image to Video, Video
to Video, Partner/API Nodes, and so on.

Deterministic and rule-based (no clustering, no LLM). Classification anchors on
the most reliable signals available, in this order:

  1. text utilities (captioning / prompt) - even when they call a partner LLM;
  2. API / partner nodes (authoritative: node.is_api, the comfy_api_nodes module);
  3. the authoritative catalog category (index.json) for the coarse buckets
     (Image Tools, Video Tools, Audio, 3D, Preprocessors, Image Editing);
  4. for the two broad generation buckets (and for custom workflows with no
     catalog category) refine by output media + input modality + name, where
     input/output modality comes from the workflow's boundary ports (or load /
     save nodes for flat graphs) - NOT from a LoadImage node, since collapsed
     subgraphs feed images through a boundary port instead.

``TaxonomyClassifier.classify`` returns the canonical category each workflow is
grouped under, so each "type" in the database is one canonical category.
"""

from __future__ import annotations

from typing import List, Set

from .model import WorkflowGraph
from .roles import classify_role

CATEGORIES: List[str] = [
    "Text to Image", "Image Edit", "Image Edit with ControlNet",
    "Inpaint / Outpaint", "Upscale", "Character", "Image Tools",
    "Text to Video", "Image to Video", "Video to Video",
    "First / Last Frame to Video", "Video Inpaint", "Video Tools",
    "3D", "Audio", "Preprocessors / Estimation", "Text Tools",
    "API / Partner Nodes",
]

# Authoritative catalog (index.json) category -> canonical category, for the
# buckets that need no refinement. The two generation buckets are absent on
# purpose so they fall through to media-based refinement.
_INDEX_MAP = {
    "Image Tools": "Image Tools",
    "Video Tools": "Video Tools",
    "Audio": "Audio",
    "3D": "3D",
    "Conditioning & Preprocessors": "Preprocessors / Estimation",
    "Text Tools": "Text Tools",
    "Image Editing": "Image Edit",
}


def _has(classes, *needles) -> bool:
    """True if any class name in the set contains any needle."""
    return any(any(n in c for n in needles) for c in classes)


def _name_has(name: str, *needles) -> bool:
    """True if the (string) name contains any needle."""
    return any(n in name for n in needles)


class TaxonomyClassifier:
    """Classify a ``WorkflowGraph`` into one canonical task category.

    Stateless and deterministic; relies on functional role classification
    (``roles.classify_role``), the authoritative catalog category, boundary-port
    modality, and the filename. One shared instance is sufficient."""

    def classify(self, graph: WorkflowGraph) -> str:
        classes = {n.class_type.lower() for n in graph.nodes.values()}
        name = graph.name.lower()
        cat = graph.category

        # 1. Text utilities - captioning / prompt / motion-description tools (even
        #    when they call a partner LLM such as Gemini).
        if cat == "Text Tools" or _name_has(name, "caption", "prompt_enhance",
                                       "motionprompt", "motion_prompt", "prompt_generation"):
            return "Text Tools"

        # 2. Partner / API nodes - sub-split by task so each bucket stays coherent.
        #    The partner node IS the generator, so pass gen=True.
        if any(n.is_api for n in graph.nodes.values()) or name.startswith("api_"):
            return "API / Partner Nodes - " + self._task_category(graph, gen=True)

        # 3. Authoritative coarse buckets from the catalog.
        if cat in _INDEX_MAP:
            return _INDEX_MAP[cat]

        # 4. Refine the generation buckets (and custom workflows) by media + IO.
        diffusion = any(classify_role(c) in ("sampler", "model_loader") for c in classes)
        return self._task_category(graph, gen=diffusion)

    @staticmethod
    def _input_modalities(graph: WorkflowGraph) -> Set[str]:
        """Content modalities entering the workflow, from boundary ports
        (collapsed subgraphs) plus load nodes (flat graphs)."""
        types = {p["data_type"] for p in graph.boundary_inputs}
        classes = {n.class_type.lower() for n in graph.nodes.values()}
        if _has(classes, "loadimage"):
            types.add("IMAGE")
        if _has(classes, "loadvideo", "vhs_loadvideo", "loadvideopath"):
            types.add("VIDEO")
        return types

    @staticmethod
    def _output_media(graph: WorkflowGraph) -> str:
        """The media the workflow produces: 3d | audio | video | image."""
        out = {p["data_type"] for p in graph.boundary_outputs}
        classes = {n.class_type.lower() for n in graph.nodes.values()}
        name = graph.name.lower()
        if "MESH" in out or _has(classes, "saveglb", "splat", "gaussian") \
                or _name_has(name, "to_model", "gaussian_splat", "hunyuan3d", "triposplat", "_3d"):
            return "3d"
        if _name_has(name, "audio") or _has(classes, "saveaudio", "stableaudio", "ace_step"):
            return "audio"
        if "VIDEO" in out or _has(classes, "videocombine", "createvideo", "savevideo") \
                or _name_has(name, "video", "i2v", "t2v", "flf", "v2v"):
            return "video"
        return "image"

    def _task_category(self, graph: WorkflowGraph, gen: bool) -> str:
        """The task category from output media + input modality + name. ``gen`` is
        whether the workflow can *generate* content (a local diffusion sampler, or
        a partner generation node) - used so a non-diffusion partner video graph is
        not mistaken for a video manipulation tool."""
        name = graph.name.lower()
        classes = {n.class_type.lower() for n in graph.nodes.values()}
        uses_api = (any(getattr(n, "is_api", False) for n in graph.nodes.values())
                    or name.startswith("api_"))
        in_types = self._input_modalities(graph)
        has_img_in = "IMAGE" in in_types
        has_vid_in = "VIDEO" in in_types
        media = self._output_media(graph)

        if media == "3d":
            return "3D"
        if media == "audio":
            return "Audio"

        if media == "video":
            if _name_has(name, "flf", "first_last", "first-last", "firstlast") \
                    or _has(classes, "firstlast"):
                return "First / Last Frame to Video"
            if "inpaint" in name:
                return "Video Inpaint"
            if _name_has(name, "upscale", "enhance"):
                return "Upscale"
            if not gen:
                return "Video Tools"
            # Explicit task in the name or the node class wins over inferred input
            # modality. Partner nodes encode the task in their class name, e.g.
            # WanTextToVideoApi / KlingOmniProImageToVideoNode.
            if _name_has(name, "t2v", "text_to_video") or _has(classes, "texttovideo", "text2video"):
                return "Text to Video"
            if _name_has(name, "i2v", "image_to_video") or _has(classes, "imagetovideo", "image2video"):
                return "Image to Video"
            if _name_has(name, "v2v", "vid2vid", "video_to_video", "vace") or has_vid_in:
                return "Video to Video"
            # A bare image input implies image-to-video for LOCAL graphs. Partner
            # generation nodes (e.g. Veo) often expose an OPTIONAL image input, so
            # for API workflows a generic node with no i2v signal stays text-to-video.
            if has_img_in and not uses_api:
                return "Image to Video"
            return "Text to Video"

        # media == image. Note direction: "image_to_depth" / "depth_map" is depth
        # *estimation* (a preprocessor), whereas "depth_to_image" is controlnet
        # *generation* guided by a depth map.
        if _name_has(name, "estimation", "to_depth", "to_pose", "depth_map",
                     "pose_map", "geometry", "segmentation", "detection",
                     "remove_background"):
            return "Preprocessors / Estimation"
        if _name_has(name, "inpaint", "outpaint"):
            return "Inpaint / Outpaint"
        if _name_has(name, "canny", "depth_to", "pose_to", "controlnet", "_control") \
                or _has(classes, "controlnet"):
            return "Image Edit with ControlNet"
        if _name_has(name, "upscale", "enhance"):
            return "Upscale"
        if "character" in name:
            return "Character"
        if not gen and (has_img_in or _name_has(name, "crop", "grid", "tile",
                                                "channel", "blur", "color")):
            return "Image Tools"
        if has_img_in or _name_has(name, "edit", "img2img", "image_to_image", "relight"):
            return "Image Edit"
        return "Text to Image"
