from pathlib import Path
from typing import List, Optional, Dict, Any
import logging

from .tools import PrompterBase


class Writer(PrompterBase):
    """Encapsulates the logic formerly present in ``prompter.create_image_prompt``
    from the monolithic main script.  The heavy lifting is performed in the
    inherited base class; this subclass only provides the single public method
    used by the producer workflow.
    """

    def create_image_prompt(
        self,
        briefing: str,
        selected_workflow_path: Path,
        mood_image_paths: Optional[List[Path]] = None,
        prompt_type: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Create prompt for image generation based on briefing and provided images.

        ``prompt_type`` is an optional string that corresponds to one of the
        predefined prompt guides (the part of the filename after ``guide.``).
        When supplied, the contents of that guide will be appended to the
        system prompt to influence how the writer composes the final prompts.
        """
        # Determine which guideline to use.  Prompt type (suggested by producer)
        # has the highest priority; if a corresponding guide file exists we'll
        # load it and skip the workflow-specific/default logic entirely.  This
        # addresses the case where the producer wants a style such as
        # "concise" regardless of which workflow is chosen.
        keyframes_guidelines = ""
        if prompt_type:
            pt_path = self.guide_keyframes_path.parent / f"guide.{prompt_type}.md"
            if pt_path.is_file():
                logging.info(f"Prompt type '{prompt_type}' yields guideline file: {pt_path}")
                keyframes_guidelines = self._load_llm_prompt(pt_path)
            else:
                logging.warning(
                    f"Prompt type guide not found: {pt_path}; falling back to workflow/default"
                )

        # Any remaining prompt_type guidance is redundant because we've already
        # consumed it; keep variable for compatibility but it will be empty.
        prompt_type_guidelines = ""
        # note: we intentionally do not load it again to avoid duplicate
        # content in the system prompt.  The writer prompt template is already
        # written to handle this case without issue.

        raw_system_prompt = self._load_llm_prompt(self.path_prompt_writer)
        # provide both guideline sets to the format string; the writer prompt
        # template must be updated accordingly.
        system_prompt = raw_system_prompt.format(
            guidelines=keyframes_guidelines,
            prompt_type_guidelines=prompt_type_guidelines,
        )

        # Initiate LLM prompt with briefing text
        llm_prompt: Dict[str, Any] = {"role": "user", "content": f"BRIEFING: {briefing}"}

        # encode any mood images that were supplied
        images_data = self._encode_existing_images(mood_image_paths)
        if images_data:
            llm_prompt["images"] = images_data

        payload = {
            "model": self.ollamamodel,
            "messages": [
                {"role": "system", "content": system_prompt},
                llm_prompt,
            ],
            "format": "json",
            "stream": False,
            "options": {
                "temperature": 0.5,
                "num_predict": 4096,
            },
            "keep_alive": "5s",
        }

        # dispatch to shared helper that handles the chat call and JSON parsing
        return self._chat(payload, normalize=True)
