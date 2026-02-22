from pathlib import Path
from typing import List, Optional, Dict, Any
import logging

from .tools import PrompterBase


class Supervisor(PrompterBase):
    """Wraps the supervision functionality that was previously part of
    ``prompter`` in the original script.  This class is only responsible for
    the one public method ``supervise``.
    """

    def _build_supervision_llm_prompt(
        self, briefing: str, 
        mood_image_paths: Optional[List[Path]], 
        original_prompt: Optional[str] = None
    ) -> str:
        """Compose the text portion of a supervision request.

        The returned string includes the briefing and, when mood images are
        provided, a textual list of their filenames.  The generated-image
        section is always appended by the caller.  If an `original_prompt` is
        supplied it is included as an additional section so the supervisor can
        comment on how to improve it.

        To help the LLM distinguish which of the transmitted pictures are
        merely inspiration versus the actual output under review, the text
        explicitly notes that the mood images come first and are not to be
        judged as the generated image.  The generated image is always listed
        last and clearly labelled "for review".
        """
        llm_prompt = f"BRIEFING: {briefing}"
        if mood_image_paths:
            llm_prompt += (
                "\n\nMOOD IMAGES (for reference only; do NOT evaluate these as the\n"
                "generated result – they are provided to convey style/tone):"
            )
            for path in mood_image_paths:
                llm_prompt += f"\n- {path.name}"
        if original_prompt:
            llm_prompt += f"\n\nORIGINAL POSITIVE PROMPT: {original_prompt}"
        llm_prompt += "\n\nGENERATED IMAGE (for review):"
        return llm_prompt

    def supervise(
        self,
        briefing: str,
        generated_image_path: Path,
        mood_image_paths: Optional[List[Path]] = None,
        original_prompt: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Supervises a generated image against the original brief, mood images,
        and optionally the positive prompt that was used to generate it.
        """
        system_prompt = self._load_llm_prompt(self.path_prompt_supe)

        # encode mood images (warnings handled by helper)
        all_images_data = self._encode_existing_images(mood_image_paths)

        # ensure the generated image is available and append its resized version
        if generated_image_path.is_file():
            all_images_data.append(
                self._encode_and_resize_image(generated_image_path, scale_factor=0.75)
            )
        else:
            logging.error(f"Generated image not found: {generated_image_path}")
            return {"error": "Generated image not found", "raw_content": ""}

        # build the text portion of the user prompt
        llm_text = self._build_supervision_llm_prompt(
            briefing, mood_image_paths, original_prompt
        )
        llm_prompt = {"role": "user", "content": llm_text, "images": all_images_data}

        payload = {
            "model": self.ollamamodel,
            "messages": [{"role": "system", "content": system_prompt}, llm_prompt],
            "format": "json",
            "stream": False,
            "options": {"temperature": 0.2},
            "keep_alive": "5s",
        }

        # use the shared chat/parse helper; it already logs appropriately
        logging.info(f"Supervising image: {generated_image_path}")
        return self._chat(payload)
