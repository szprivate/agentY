from pathlib import Path
from typing import List, Optional, Dict, Any
import logging
import time
import ollama

from .tools import PrompterBase, Generator, SYS_CONFIG
from .writer import Writer
from .supervisor import Supervisor


class Producer(PrompterBase):
    """Responsible for choosing a workflow and providing a directive to the
    writer. 
    """

    def select_workflow(
        self,
        briefing: str,
        input_summary: str,
        workflow_dir: Path,
    ) -> Optional[Dict[str, Any]]:
        """Acts as a "producer" to select the best ComfyUI workflow and provide a
        directive.  Now also decides on a ``prompt_type`` based on available
        prompt guides.
        Returns a dictionary with 'workflow_path', 'reason', and optionally
        'prompt_type'.
        """
        system_prompt = self._load_llm_prompt(self.path_prompt_producer)
        if not workflow_dir.is_dir():
            logging.error(f"Workflow directory not found: {workflow_dir}")
            return None

        # Get a list of workflow file paths
        workflow_files = [str(p.resolve()) for p in workflow_dir.glob('*.json')]
        if not workflow_files:
            logging.error(f"No workflow files (.json) found in {workflow_dir}")
            return None

        workflow_list_str = "\n".join(workflow_files)

        # gather available prompt guides (files prefixed with 'guide.' in prompts directory)
        prompts_dir = Path(self.guide_keyframes_path).parent
        prompt_guides = sorted(prompts_dir.glob("guide.*.md"))
        # Exclude the keyframes guideline itself since that's used elsewhere
        prompt_guides = [p for p in prompt_guides if not p.name.startswith("guide.keyframes")]
        # we pass the filenames so the LLM can choose one; producers should strip the 'guide.' prefix
        prompt_guides_str = "\n".join([p.name for p in prompt_guides])

        llm_text = (
            f"BRIEFING: {briefing}\n\n"
            f"INPUT SUMMARY: {input_summary}\n\n"
            f"AVAILABLE WORKFLOWS:\n{workflow_list_str}\n\n"
            f"AVAILABLE PROMPT GUIDES:\n{prompt_guides_str}"
        )
        payload = {
            "model": self.ollamamodel,
            "messages": [{"role": "system", "content": system_prompt}, {"role": "user", "content": llm_text}],
            "format": "json",
            "stream": False,
            "options": {"temperature": 0.0},
        }
        result = self._chat(payload)

        workflow_path_str = result.get("workflow_path")
        reason = result.get("reason", "No reason provided.")
        prompt_type = result.get("prompt_type")

        logging.info(f"Producer raw result: {result}")

        if not workflow_path_str:
            return None
        output = {"workflow_path": Path(workflow_path_str), "reason": reason}
        if prompt_type:
            # normalize value by stripping potential file name parts
            # if user returned a filename like 'guide.concise.md', keep 'concise'
            if prompt_type.startswith("guide."):
                prompt_type = (
                    Path(prompt_type).stem.split('.', 1)[1]
                    if '.' in Path(prompt_type).stem
                    else Path(prompt_type).stem
                )
            output["prompt_type"] = prompt_type
        return output



# utility helpers used by the module-level ``iteration`` entrypoint.
# they make the main loop much easier to follow and test.

def _load_mood_images() -> List[Path]:
    """Return a list of existing mood image paths from configuration."""
    mood_images_dir = Path(SYS_CONFIG.get("mood_images_dir", "./mood_images/"))
    if mood_images_dir.is_dir():
        return list(mood_images_dir.glob("*"))
    return []


def _load_briefing() -> str:
    """Read the briefing text file specified in the config.

    Raises FileNotFoundError if the file does not exist (caller should handle).
    """
    briefing_path = Path(SYS_CONFIG["prompts"]["briefing"])
    text = briefing_path.read_text(encoding="utf-8").strip()
    if not text:
        logging.warning(f"Briefing file was empty: {briefing_path}")
    return text


def _init_producer() -> Producer:
    """Instantiate the ``Producer`` and propagate initialization errors."""
    return Producer()


def _choose_workflow(
    producer: Producer, briefing: str, input_summary: str
) -> Dict[str, Any]:
    """Ask the producer to select a workflow and return its choice dict.

    Raises RuntimeError if the producer fails to pick a valid path.
    """
    workflows_dir = Path(SYS_CONFIG.get("comfyui_workflows_dir"))
    choice = producer.select_workflow(briefing, input_summary, workflows_dir)
    if not choice or not choice.get("workflow_path"):
        raise RuntimeError("Producer failed to select a valid workflow")
    return choice


def _init_writer() -> Writer:
    """Return an initialized ``Writer`` or propagate errors."""
    return Writer()


def _init_generator(selected_workflow: Path) -> Generator:
    """Return a configured ``Generator`` instance.

    May raise exceptions if initialization fails; caller should handle.
    """
    override = SYS_CONFIG.get("comfyui_output_dir_override")
    return Generator(
        workflow_path=selected_workflow,
        comfyui_output_dir=Path(override) if override else None,
    )


def _supervision_loop(
    generator: Generator,
    positive_prompt: str,
    negative_prompt: str,
    mood_images: List[Path],
    briefing: str,
) -> None:
    """Perform the generate/review loop until the image is approved or we bail."""

    max_iter = SYS_CONFIG.get("max_iter", 3)
    logging.info(f"Maximum iterations set to {max_iter}")

    current_prompt = positive_prompt
    generated_image: Optional[Path] = None
    supervisor: Optional[Supervisor] = None
    approved = False

    for attempt in range(1, max_iter + 1):
        logging.info(f"--- Iteration {attempt}/{max_iter} ---")
        if not current_prompt:
            logging.error("No prompt available for generation; stopping iterations.")
            break

        logging.info(f"Queuing generation with prompt: {current_prompt}")
        prompt_id, output_path = generator.generate(current_prompt, negative_prompt, mood_images)
        if prompt_id:
            generated_image = generator.wait_for_generation(prompt_id, output_path)
        else:
            logging.error("Failed to queue generation; aborting iterations.")
            break

        if not generated_image:
            logging.error("Generation did not produce an image; aborting iterations.")
            break

        if supervisor is None:
            try:
                logging.info("Re-initializing prompter to supervise the generated image...")
                supervisor = Supervisor()
            except (FileNotFoundError, ollama.ResponseError):
                logging.error("Failed to initialize supervisor for supervision. Stopping further iterations.")
                break

        logging.info("--- Starting supervisor step ---")
        verdict = supervisor.supervise(briefing, generated_image, mood_images, current_prompt)
        if "error" in verdict:
            logging.error(f"Supervisor step failed: {verdict.get('error')}")
            break

        approved = verdict.get("approved", False)
        logging.info(f"Supervisor verdict: {'APPROVED' if approved else 'REJECTED'}")
        logging.info(f"Reason: {verdict.get('reason')}")
        logging.info(f"ToDo: {verdict.get('todo')}")
        suggestion = verdict.get("prompt_suggestion")
        if suggestion:
            logging.info(f"Prompt suggestion: {suggestion}")

        if approved:
            logging.info("Image approved by supervisor; ending iterations.")
            break
        if suggestion:
            logging.info("Supervisor rejected image; using suggestion for next generation.")
            current_prompt = suggestion
            continue
        logging.info("Supervisor rejected image but provided no suggestion; stopping iterations.")
        break

    if not approved:
        if attempt >= max_iter:
            logging.warning("Reached maximum iterations without approval.")
        else:
            logging.warning("Stopped before receiving approval.")


# entry‑point helper so that ``python -m src.sub.producer`` or running this file
# directly behaves like the old monolithic script.

def iteration():
    try:
        mood_images = _load_mood_images()
        brief = _load_briefing()
    except FileNotFoundError as e:
        logging.error(e)
        exit(1)

    try:
        producer = _init_producer()
    except (FileNotFoundError, ollama.ResponseError) as e:
        logging.error(e)
        exit(1)

    input_summary = f"Available inputs: {len(mood_images)} mood image(s), 1 text prompt."

    logging.info("--- Running Producer to select workflow ---")
    try:
        producer_choice = _choose_workflow(producer, brief, input_summary)
    except RuntimeError as e:
        logging.error(e)
        exit(1)

    selected_workflow = producer_choice["workflow_path"]
    logging.info(f"Producer selected workflow: {selected_workflow}")

    logging.info("--- Running Writer to generate prompts ---")
    prompt_type = producer_choice.get("prompt_type")
    if prompt_type:
        logging.info(f"Producer suggested prompt type: {prompt_type}")

    try:
        writer = _init_writer()
    except (FileNotFoundError, ollama.ResponseError) as e:
        logging.error(e)
        exit(1)

    prompt = writer.create_image_prompt(brief, selected_workflow, mood_images, prompt_type=prompt_type)
    if "error" in prompt:
        logging.error("Writer failed to generate prompt. Exiting.")
        exit(1)

    logging.info("Writer successfully generated prompts.")
    positive_prompt = prompt.get("positive_prompt", "")
    negative_prompt = prompt.get("negative_prompt", "")
    logging.info(f"Positive Prompt: {positive_prompt}")
    logging.info(f"Negative Prompt: {negative_prompt}")

    if not positive_prompt:
        logging.error("No positive prompt was generated, cannot create image.")
        exit(1)

    logging.info("Initializing generator for image creation...")
    try:
        generator = _init_generator(selected_workflow)
    except Exception as e:
        logging.error(f"Generator initialization failed: {e}")
        exit(1)

    if positive_prompt:
        logging.info("Releasing LLM from VRAM to free up resources for image generation...")
        del writer
        time.sleep(5)
        logging.info("LLM resources should now be free.")

    _supervision_loop(generator, positive_prompt, negative_prompt, mood_images, brief)


if __name__ == "__main__":
    iteration()
