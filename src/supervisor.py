from pathlib import Path
from typing import List, Optional, Dict, Any
import logging

from .tools import PrompterBase


class Supervisor(PrompterBase):
    """Wraps the supervision functionality that was previously part of
    ``prompter`` in the original script.  This class is only responsible for
    the one public method ``supervise``.
    """

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
