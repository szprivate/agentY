from pathlib import Path
from typing import List, Dict, Any, Optional
import re

from .tools import SYS_CONFIG, PrompterBase


class Collector(PrompterBase):
    """Analyzes a briefing and breaks it into discrete tasks.

    The collector returns a list of task dictionaries containing:
    ``type`` - a string identifying the kind of work to perform.  Examples
        include ``data_collection``, ``prompt_creation``, ``image_generation``,
        ``supervision``, ``upscale`` and ``creative_upscale`` (additional
        types may be defined by the persona).
    ``summary`` - a short text description of the task (usually a sentence
        extracted from the briefing)
    ``paths`` - any file paths relevant to the task (e.g. images/videos
        gathered from folders mentioned in the briefing or the default mood
        directory).

    The order of tasks in the list matches the order they appear in the
    briefing so that the producer can execute them sequentially.
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        api_base_url: Optional[str] = None,
        prompt_persona_collector: Optional[Path] = None,
    ):
        # call base constructor to initialise the LLM client; other personas
        # will ignore their respective persona paths but that's harmless.
        super().__init__(
            model_name=model_name,
            api_base_url=api_base_url,
            prompt_persona_supe=None,
            prompt_persona_writer=None,
            prompt_persona_producer=None,
        )
        # collector persona prompt path is configured separately
        self.path_prompt_collector = (
            prompt_persona_collector
            or Path(SYS_CONFIG["prompts"].get("persona_collector", "./prompts/persona.collector.md"))
        )

    def analyze(self, briefing: str) -> Dict[str, Any]:
        """Parse the briefing and return metadata about extracted tasks.

        The returned dictionary contains two keys:
        ``tasks`` - a list of task dictionaries (same format as before).
        ``summary`` - a human-readable summary describing what the collector
            decided, useful for logging or debugging.

        The collector prefers to ask the LLM for the decomposition; if that
        attempt returns a usable list of tasks we return it immediately.  Any
        failure (exception, missing keys, empty list) causes the method to
        fall back to the original keyword-based logic that follows.
        """
        # try LLM first
        try:
            system_prompt = self._load_llm_prompt(self.path_prompt_collector)
            llm_text = f"BRIEFING: {briefing}"
            payload = {
                "model": self.ollamamodel,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": llm_text},
                ],
                "format": "json",
                "stream": False,
                "options": {"temperature": 0.0},
            }
            result = self._chat(payload)
            llm_tasks = result.get("tasks")
            if isinstance(llm_tasks, list):
                validated: List[Dict[str, Any]] = []
                for t in llm_tasks:
                    if not isinstance(t, dict):
                        continue
                    ttype = t.get("type")
                    summary = t.get("summary")
                    if not ttype or not summary:
                        continue
                    paths = t.get("paths") or []
                    normalized: List[Path] = []
                    for p in paths:
                        try:
                            normalized.append(Path(p))
                        except Exception:
                            pass
                    validated.append({"type": ttype, "summary": summary, "paths": normalized})
                if validated:
                    text = "; ".join([f"{t['type']}: {t['summary']}" for t in validated])
                    return {"tasks": validated, "summary": text}
        except Exception:
            # ignore and fall through to rule-based parser
            pass

        tasks: List[Dict[str, Any]] = []

        # try to detect any folder paths mentioned in the briefing
        folder_paths: List[Path] = []
        for token in briefing.split():
            candidate = Path(token.strip(".,\"'"))
            if candidate.is_dir():
                folder_paths.append(candidate)

        # fallback to configured mood_images dir if no explicit folder was found
        if not folder_paths:
            mood_dir = Path(SYS_CONFIG.get("mood_images_dir", "./mood_images/"))
            if mood_dir.is_dir():
                folder_paths.append(mood_dir)

        # collect all supported files from the discovered folders
        collected_paths: List[Path] = []
        for folder in folder_paths:
            for pattern in ("*.png", "*.jpg", "*.jpeg", "*.gif", "*.mp4", "*.mov", "*.webp"):
                collected_paths.extend(folder.glob(pattern))

        # split briefing into sentences so we can create tasks in order
        sentences = re.split(r'(?<=[.!?])\s+', briefing.strip())
        for sentence in sentences:
            if not sentence:
                continue
            lower = sentence.lower()
            task_type: Optional[str] = None
            if "collect" in lower and "data" in lower:
                task_type = "data_collection"
            elif "prompt" in lower and ("write" in lower or "create" in lower):
                task_type = "prompt_creation"
            elif "supervise" in lower or "review" in lower or "approve" in lower:
                task_type = "supervision"
            elif "upscale" in lower:
                task_type = "upscale"
            elif "generate" in lower or "image" in lower:
                task_type = "image_generation"

            if task_type:
                tasks.append({
                    "type": task_type,
                    "summary": sentence.strip(),
                    "paths": collected_paths.copy(),
                })

        # if the collector couldn't identify any explicit tasks, fall back to a
        # single image_generation task using the whole briefing and any paths we
        # gathered.
        if not tasks:
            tasks.append({
                "type": "image_generation",
                "summary": briefing,
                "paths": collected_paths,
            })

        summary_text = "; ".join([f"{t['type']}: {t['summary']}" for t in tasks])
        return {"tasks": tasks, "summary": summary_text}


# helper to mirror the pattern of other modules

def _init_collector() -> Collector:
    """Instantiate a ``Collector``.

    This is used by the top-level iteration helper in ``producer.py`` but is
    defined here to keep the implementation encapsulated.
    """
    return Collector()
