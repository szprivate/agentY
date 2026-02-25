from pathlib import Path
from typing import List, Optional, Dict, Any
import logging
import time
import ollama

from .tools import PrompterBase, Generator, SYS_CONFIG
from .writer import Writer
from .supervisor import Supervisor
from .collector import _init_collector, Collector


class Producer(PrompterBase):
    """Responsible for choosing a workflow and providing a directive to the
    writer. 
    """

    def select_workflow(
        self,
        briefing: str,
        input_summary: str,
        workflow_dir: Path,
        task_type: Optional[str] = None,
        paths: Optional[List[Path]] = None,
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
        # include any task-specific metadata so the model can choose workflows
        if task_type:
            llm_text += f"\n\nTASK TYPE: {task_type}"
        if paths:
            # note: only include filenames to keep the prompt concise
            path_list_str = "\n" + "\n".join(str(p) for p in paths)
            llm_text += f"\n\nTASK PATHS:{path_list_str}"
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
    producer: Producer, briefing: str, input_summary: str, task_type: Optional[str] = None, paths: Optional[List[Path]] = None
) -> Dict[str, Any]:
    """Ask the producer to select a workflow and return its choice dict.

    Raises RuntimeError if the producer fails to pick a valid path.
    """
    workflows_dir = Path(SYS_CONFIG.get("comfyui_workflows_dir"))
    choice = producer.select_workflow(briefing, input_summary, workflows_dir, task_type=task_type, paths=paths)
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



def _retry_writer_prompt(
    writer: Writer,
    briefing: str,
    selected_workflow: Path,
    mood_images: List[Path],
    prompt_type: Optional[str],
    retries: int = 3,
) -> Dict[str, Any]:
    """Attempt to create an image prompt multiple times if the response is incomplete.

    The writer sometimes returns no prompt or an error; if so we log a warning
    and retry.  After ``retries`` attempts the last result is returned and the
    caller can decide how to handle it.
    """
    last = {}
    for attempt in range(1, retries + 1):
        logging.info(f"Writer attempt {attempt}/{retries}")
        result = writer.create_image_prompt(
            briefing, selected_workflow, mood_images, prompt_type=prompt_type
        )
        last = result
        if "error" in result:
            logging.warning(f"Writer returned error: {result.get('error')}")
            continue
        if not result.get("positive_prompt"):
            logging.warning("Writer did not provide a positive prompt; retrying.")
            continue
        # if we got here, the result looks usable
        return result
    return last


def _supervise_with_retries(
    supervisor: Supervisor,
    briefing: str,
    generated_image: Path,
    mood_images: List[Path],
    original_prompt: str,
    retries: int = 3,
) -> Dict[str, Any]:
    """Call ``supervisor.supervise`` and retry if response lacks reason/todo.

    The supervisor might return an empty verdict or omit the reason/todo keys.
    We retry up to ``retries`` times before returning whatever we last got.
    """
    last = {}
    for attempt in range(1, retries + 1):
        logging.info(f"Supervisor attempt {attempt}/{retries}")
        verdict = supervisor.supervise(briefing, generated_image, mood_images, original_prompt)
        last = verdict
        if "error" in verdict:
            logging.error(f"Supervisor returned error: {verdict.get('error')}")
            break
        # if either of the informative fields is present we consider it valid
        if verdict.get("reason") or verdict.get("todo") or verdict.get("prompt_suggestion"):
            return verdict
        logging.warning("Supervisor result missing reason/todo/suggestion; retrying.")
    return last


def _supervision_loop(
    generator: Generator,
    positive_prompt: str,
    negative_prompt: str,
    mood_images: List[Path],
    briefing: str,
) -> Optional[Path]:
    """Perform the generate/review loop until the image is approved or we bail.

    Returns the last generated image path (regardless of approval) so that
    callers can remember it for subsequent tasks.
    """

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
        verdict = _supervise_with_retries(
            supervisor, briefing, generated_image, mood_images, current_prompt
        )
        if "error" in verdict:
            # if we got an error even after retries bail out
            logging.error(f"Supervisor step failed ultimately: {verdict.get('error')}")
            break

        approved = verdict.get("approved", False)
        logging.info(f"Supervisor verdict: {'APPROVED' if approved else 'REJECTED'}")
        logging.info(f"Reason: {verdict.get('reason')}" )
        logging.info(f"ToDo: {verdict.get('todo')}")
        suggestion = verdict.get("prompt_suggestion")
        if suggestion:
            logging.info(f"Prompt suggestion: {suggestion}")

        if approved:
            logging.info("Image approved by supervisor; ending iterations.")
            break
        # replay current prompt if no suggestion (do not stop the loop)
        if suggestion:
            logging.info("Supervisor rejected image; using suggestion for next generation.")
            current_prompt = suggestion
            continue
        logging.warning("Supervisor provided no suggestion; will retry next iteration with same prompt.")
        # loop will naturally proceed to next attempt with unchanged prompt

    if not approved:
        if attempt >= max_iter:
            logging.warning("Reached maximum iterations without approval.")
        else:
            logging.warning("Stopped before receiving approval.")
    return generated_image


# entry‑point helper so that ``python -m src.sub.producer`` or running this file
# directly behaves like the old monolithic script.

def iteration():
    try:
        mood_images = _load_mood_images()
        brief = _load_briefing()
    except FileNotFoundError as e:
        logging.error(e)
        exit(1)

    # run the collector to split the briefing into discrete tasks
    try:
        collector = _init_collector()
        collector_result = collector.analyze(brief)
        tasks = collector_result.get("tasks", [])
        task_summary = collector_result.get("summary")
        if task_summary:
            logging.info(f"Collector summary: {task_summary}")
    except Exception as e:
        logging.error(f"Collector failed to initialise or analyse briefing: {e}")
        exit(1)

    try:
        producer = _init_producer()
    except (FileNotFoundError, ollama.ResponseError) as e:
        logging.error(e)
        exit(1)

    # iterate through tasks sequentially, keeping state for prompts and images
    last_positive: str = ""
    last_negative: str = ""
    last_generated_image: Optional[Path] = None

    for idx, task in enumerate(tasks, start=1):
        ttype = task.get("type")
        summary = task.get("summary", "")
        paths = task.get("paths", [])
        logging.info(f"--- Task {idx}/{len(tasks)}: {summary} ({ttype}) ---")
        input_summary = f"Task type: {ttype}. Available paths: {len(paths)}"

        logging.info("--- Running Producer to select workflow ---")
        try:
            producer_choice = _choose_workflow(
                producer,
                summary,
                input_summary,
                task_type=ttype,
                paths=paths,
            )
        except RuntimeError as e:
            logging.error(e)
            exit(1)

        selected_workflow = producer_choice["workflow_path"]
        logging.info(f"Producer selected workflow: {selected_workflow}")

        prompt_type = producer_choice.get("prompt_type")
        if prompt_type:
            logging.info(f"Producer suggested prompt type: {prompt_type}")

        # handle simple pass-through tasks
        if ttype == "data_collection":
            logging.info("Data collection task - skipping further steps.")
            continue

        if ttype == "prompt_creation":
            try:
                writer = _init_writer()
            except (FileNotFoundError, ollama.ResponseError) as e:
                logging.error(e)
                exit(1)
            mood_for_writer = [
                p for p in paths
                if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".gif")
            ]
            prompt = _retry_writer_prompt(writer, summary, selected_workflow, mood_for_writer, prompt_type)
            if "error" in prompt:
                logging.error("Writer failed to generate prompt after retries. Exiting.")
                exit(1)
            last_positive = prompt.get("positive_prompt", "")
            last_negative = prompt.get("negative_prompt", "")
            logging.info("Stored prompts for subsequent tasks.")
            continue

        if ttype == "supervision":
            if not last_generated_image:
                logging.error("No generated image available for supervision task.")
                continue
            try:
                supervisor = Supervisor()
            except (FileNotFoundError, ollama.ResponseError) as e:
                logging.error(f"Failed to initialize supervisor: {e}")
                continue
            verdict = _supervise_with_retries(
                supervisor,
                summary,
                last_generated_image,
                [p for p in paths if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".gif")],
                last_positive,
            )
            logging.info(f"Supervision verdict: {verdict}")
            continue

        # for generation-type tasks we need prompts
        if not last_positive:
            try:
                writer = _init_writer()
            except (FileNotFoundError, ollama.ResponseError) as e:
                logging.error(e)
                exit(1)
            mood_for_writer = [
                p for p in paths
                if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".gif")
            ]
            prompt = _retry_writer_prompt(writer, summary, selected_workflow, mood_for_writer, prompt_type)
            if "error" in prompt:
                logging.error("Writer failed to generate prompt after retries. Exiting.")
                exit(1)
            last_positive = prompt.get("positive_prompt", "")
            last_negative = prompt.get("negative_prompt", "")

        if not last_positive:
            logging.error("No positive prompt available, cannot proceed with generation.")
            exit(1)

        try:
            generator = _init_generator(selected_workflow)
        except Exception as e:
            logging.error(f"Generator initialization failed: {e}")
            exit(1)

        gen_moods = [
            p for p in paths
            if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".gif")
        ]
        if ttype in ("upscale", "creative_upscale") and last_generated_image:
            gen_moods.append(last_generated_image)

        last_generated_image = _supervision_loop(
            generator,
            last_positive,
            last_negative,
            gen_moods,
            summary,
        )


if __name__ == "__main__":
    iteration()
