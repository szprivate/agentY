import base64
import json
import logging
import copy
from pathlib import Path
import random
import uuid
import requests
import ollama
from PIL import Image
import io
import time
from typing import List, Dict, Any, Optional


def load_settings(config_file: Path = Path("./config/settings.json")) -> Dict[str, Any]:
    """READ SETTINGS
    """
    try:
        text = config_file.read_text(encoding="utf-8")
        return json.loads(text)
    except FileNotFoundError:
        logging.error(f"Path configuration file not found: {config_file}")
        raise
    except json.JSONDecodeError as exc:
        logging.error(f"Invalid JSON in path configuration file {config_file}: {exc}")
        raise


# load once, reuse globally
SYS_CONFIG: Dict[str, Any] = load_settings()


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class prompter:
    """PROMPTER CLASS FOR CREATING IMAGE PROMPTS, USES MOOD IMAGES AND SINPLE BRIEFING PROMPTS
    """

    def __init__(self,
                 model_name: Optional[str] = None,
                 api_base_url: Optional[str] = None,
                 prompt_guide_default: Optional[Path] = None,
                 prompt_persona_supe: Optional[Path] = None,
                 prompt_persona_writer: Optional[Path] = None,
                 prompt_persona_producer: Optional[Path] = None):

        # load defaults from configuration if the caller didn't supply them
        api_base_url = api_base_url or SYS_CONFIG.get("ollama_api_url")
        prompt_guide_default = prompt_guide_default or Path(SYS_CONFIG["prompts"]["guide_keyframes"])
        prompt_persona_supe = prompt_persona_supe or Path(SYS_CONFIG["prompts"]["persona_supervisor"])
        prompt_persona_writer = prompt_persona_writer or Path(SYS_CONFIG["prompts"]["persona_writer"])
        prompt_persona_producer = prompt_persona_producer or Path(SYS_CONFIG["prompts"]["persona_producer"])

        self.api_base_url = api_base_url.replace("/v1", "")  # Use the native Ollama API endpoint
        self.ollamaclient = ollama.Client(host=self.api_base_url, timeout=60)
        self.ollamamodel = model_name or SYS_CONFIG.get("ollama-model")
        self.guide_keyframes_path = prompt_guide_default
        self.path_prompt_producer = prompt_persona_producer
        self.path_prompt_writer = prompt_persona_writer
        self.path_prompt_supe = prompt_persona_supe
        self._ensure_model_is_loaded()

    def _ensure_model_is_loaded(self):
        """Ensures the specified model is loaded in Ollama."""
        logging.info(f"Checking if model '{self.ollamamodel}' is loaded...")
        try:
            # call model to check whether it's active
            self.ollamaclient.show(self.ollamamodel)
            logging.info(f"Model '{self.ollamamodel}' is loaded and ready.")
        except ollama.ResponseError as e:
            logging.error(f"Failed to preload model '{self.ollamamodel}': {e}")
            raise

    def _encode_image(self, 
                      image_path: Path) -> str:
        """Converts an image file to a base64 string."""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    def _load_llm_prompt(self,
                         prompt_file: Path) -> str:
        """Load LLM prompt from a file."""
        try:
            logging.info(f"Loading LLM prompt from file: {prompt_file}")
            return prompt_file.read_text(encoding='utf-8')
        except FileNotFoundError:
            raise FileNotFoundError(f"Prompt file not found: {prompt_file}")

    def _encode_and_resize_image(self, 
                                 image_path: Path, 
                                 scale_factor: float = 0.5) -> str:
        """Resizes an image and encodes it to a base64 string."""
        try:
            with Image.open(image_path) as img:
                if img.mode == 'P': # Convert palette-based images to RGB
                    img = img.convert('RGB')
                new_size = (int(img.width * scale_factor), int(img.height * scale_factor))
                resized_img = img.resize(new_size, Image.Resampling.LANCZOS)

                buffer = io.BytesIO()
                resized_img.save(buffer, format="JPEG") # Save as JPEG for efficiency
                return base64.b64encode(buffer.getvalue()).decode('utf-8')
        except Exception as e:
            logging.error(f"Failed to resize and encode image {image_path}: {e}")
            return self._encode_image(image_path) # Fallback to original encoding

    def _encode_existing_images(self, 
                                image_paths: Optional[List[Path]]) -> List[str]:
        """Convert a list of image paths to base64 strings, skipping missing files.
        """
        encoded: List[str] = []
        if not image_paths:
            return encoded

        for path in image_paths:
            if path.is_file():
                encoded.append(self._encode_image(path))
            else:
                logging.warning(f"Image file not found, skipping: {path}")
        return encoded

    def _build_supervision_llm_prompt(self, 
                                   briefing: str, 
                                   mood_image_paths: Optional[List[Path]],
                                   original_prompt: Optional[str] = None) -> str:
        """Compose the text portion of a supervision request.

        The returned string includes the briefing and, when mood images are
        provided, a textual list of their filenames.  The generated-image
        section is always appended by the caller.  If an `original_prompt` is
        supplied it is included as an additional section so the supervisor can
        comment on how to improve it.
        """
        llm_prompt = f"BRIEFING: {briefing}"
        if mood_image_paths:
            llm_prompt += "\n\nMOOD IMAGES:"
            for path in mood_image_paths:
                # only filenames are included - the LLM receives the actual
                # data separately via the ``images`` field.
                llm_prompt += f"\n- {path.name}"
        if original_prompt:
            llm_prompt += f"\n\nORIGINAL POSITIVE PROMPT: {original_prompt}"
        llm_prompt += "\n\nGENERATED IMAGE (for review):"
        return llm_prompt

    def _chat(self, payload: Dict[str, Any], normalize: bool = False) -> Dict[str, Any]:
        """Send a chat payload to Ollama and parse the resulting JSON.

        ``normalize`` enables the special behaviour that ``create_prompt``
        previously implemented (renaming ``prompt`` to ``positive_prompt`` and
        warning if the key is missing).
        """
        try:
            result = self.ollamaclient.chat(**payload)
            if result.get("done_reason") == "length":
                logging.warning("LLM response may be truncated (finish_reason: length)")
            message_content = result.get("message", {}).get("content", "{}")
            parsed = json.loads(message_content)
            if normalize:
                if "prompt" in parsed and "positive_prompt" not in parsed:
                    logging.warning("LLM returned key 'prompt' instead of 'positive_prompt'; normalizing")
                    parsed["positive_prompt"] = parsed.pop("prompt")
                if "positive_prompt" not in parsed:
                    logging.warning("Parsed LLM response did not contain a positive prompt")
            return parsed
        except ollama.ResponseError as e:
            logging.error(f"Failed to get response from LLM: {e}")
            return {"error": "Failed to get response from LLM", "raw_content": str(e)}
        except json.JSONDecodeError as e:
            raw_content = result.get("message", {}).get("content", "") if 'result' in locals() else ""
            logging.error(f"Failed to decode LLM response as JSON. Raw response:\n{raw_content}")
            return {"error": "Failed to parse JSON response", "raw_content": raw_content}

    def create_image_prompt(self, 
                      briefing: str,
                      selected_workflow_path: Path,
                      mood_image_paths: Optional[List[Path]] = None,
                      prompt_type: Optional[str] = None) -> Dict[str, Any]:
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
                logging.warning(f"Prompt type guide not found: {pt_path}; falling back to workflow/default")

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
            prompt_type_guidelines=prompt_type_guidelines
        )
        
        # Initiate LLM prompt with briefing text
        llm_prompt: Dict[str, Any] = {
            "role": "user",
            "content": f"BRIEFING: {briefing}"
        }

        # encode any mood images that were supplied
        images_data = self._encode_existing_images(mood_image_paths)
        if images_data:
            llm_prompt["images"] = images_data

        payload = {
            "model": self.ollamamodel,
            "messages": [
                {"role": "system", "content": system_prompt},
                llm_prompt
            ],
            "format": "json",
            "stream": False,
            "options": {
                "temperature": 0.5,
                "num_predict": 4096,
            },
            "keep_alive": "5s"
        }

        # dispatch to shared helper that handles the chat call and JSON parsing
        return self._chat(payload, normalize=True)

    def supervise(self,
                  briefing: str,
                  generated_image_path: Path,
                  mood_image_paths: Optional[List[Path]] = None,
                  original_prompt: Optional[str] = None) -> Dict[str, Any]:
        """Supervises a generated image against the original brief, mood images,
        and optionally the positive prompt that was used to generate it.
        """
        system_prompt = self._load_llm_prompt(self.path_prompt_supe)

        # encode mood images (warnings handled by helper)
        all_images_data = self._encode_existing_images(mood_image_paths)

        # ensure the generated image is available and append its resized version
        if generated_image_path.is_file():
            all_images_data.append(self._encode_and_resize_image(generated_image_path, scale_factor=0.75))
        else:
            logging.error(f"Generated image not found: {generated_image_path}")
            return {"error": "Generated image not found", "raw_content": ""}

        # build the text portion of the user prompt
        llm_text = self._build_supervision_llm_prompt(briefing, mood_image_paths, original_prompt)
        llm_prompt = {
            "role": "user",
            "content": llm_text,
            "images": all_images_data
        }

        payload = {
            "model": self.ollamamodel,
            "messages": [{"role": "system", "content": system_prompt}, llm_prompt],
            "format": "json",
            "stream": False,
            "options": {"temperature": 0.2},
            "keep_alive": "5s"
        }

        # use the shared chat/parse helper; it already logs appropriately
        logging.info(f"Supervising image: {generated_image_path}")
        return self._chat(payload)

    def select_workflow(self,
                        briefing: str,
                        input_summary: str,
                        workflow_dir: Path) -> Optional[Dict[str, Any]]:
        """
        Acts as a "producer" to select the best ComfyUI workflow and provide a directive.
        Now also decides on a ``prompt_type`` based on available prompt guides.
        Returns a dictionary with 'workflow_path', 'reason', and optionally 'prompt_type'.
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
                prompt_type = Path(prompt_type).stem.split('.', 1)[1] if '.' in Path(prompt_type).stem else Path(prompt_type).stem
            output["prompt_type"] = prompt_type
        return output

class generator:
    """GENERATOR CLASS FOR GENERATING IMAGES USING COMFYUI TEMPLATE
    - look for Image Load node (WAS Suite), and replace the images with the mood images
    - replace prompts in the workflow
    - queue the prompt in ComfyUI
    """

    def __init__(self,
                 comfyui_url: Optional[str] = None,
                 workflow_path: Optional[Path] = None,
                 comfyui_output_dir: Optional[Path] = None):
        # resolve configuration defaults
        comfyui_url = comfyui_url or SYS_CONFIG.get("comfyui_url")
        workflow_path = workflow_path or Path(SYS_CONFIG.get("comfyui_workflow"))
        default_out = SYS_CONFIG.get("comfyui_output_dir_default")
        comfyui_output_dir = comfyui_output_dir or (Path(SYS_CONFIG.get("comfyui_output_dir_override")) if SYS_CONFIG.get("comfyui_output_dir_override") else Path(default_out))

        self.comfyui_url = comfyui_url
        self.workflow_path = workflow_path
        self.client_id = str(uuid.uuid4())
        self.session = requests.Session()
        self.workflow = self._load_workflow()

        if comfyui_output_dir and comfyui_output_dir.is_dir():
            self.comfyui_output_dir = comfyui_output_dir
        else:
            # Otherwise, try to fetch it from the ComfyUI API
            self.comfyui_output_dir = self._get_comfyui_output_path(comfyui_url)

        if not self.comfyui_output_dir:
            raise FileNotFoundError(
                "Could not determine ComfyUI output directory. "
                "Please ensure ComfyUI is running or specify the path manually "
                "via the `comfyui_output_dir` parameter."
            )

    def _get_comfyui_output_path(self, 
                                 comfyui_url: str) -> Optional[Path]:
        """Fetches the output directory path from the running ComfyUI instance."""
        logging.info("Attempting to fetch ComfyUI output directory from API...")
        try:
            # The 'SaveImage' node info contains the default output directory.
            response = self.session.get(f"{comfyui_url}object_info/SaveImage", timeout=5)
            response.raise_for_status()
            info = response.json()
            # The key 'output_dir' holds the path set by --output-directory
            if 'output_dir' in info:
                path = Path(info['output_dir'])
                logging.info(f"Successfully fetched ComfyUI output directory: {path}")
                return path
            logging.warning("Could not find 'output_dir' in SaveImage node info.")
        except requests.exceptions.RequestException as e:
            logging.warning(f"Could not connect to ComfyUI to get output path: {e}")
        return None

    def _load_workflow(self) -> Dict[str, Any]:
        """Loads the ComfyUI workflow from a JSON file."""
        try:
            with self.workflow_path.open('r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            logging.error(f"Workflow file not found: {self.workflow_path}")
            raise
        except json.JSONDecodeError:
            logging.error(f"Failed to decode workflow JSON from {self.workflow_path}")
            raise

    def _find_node_id_by_title(self, 
                               title: str) -> Optional[str]:
        """Finds a node's ID in the workflow by its title."""
        for node_id, node in self.workflow.items():
            # Ensure we are only checking dictionary-like node objects with a _meta key
            if isinstance(node, dict) and "_meta" in node and node["_meta"].get("title") == title:
                    return node_id
        return None
    
    def _find_node_ids_by_class_type(self, 
                                     class_type: str) -> List[str]:
        """Finds node IDs in the workflow by their class type."""
        return [
            node_id
            for node_id, node in self.workflow.items()
            if isinstance(node, dict) and node.get("class_type") == class_type
        ]

    def generate(self, 
                 positive_prompt: str, 
                 negative_prompt: str, 
                 mood_images: Optional[List[Path]] = None) -> Optional[tuple[str, Path]]:
        """Generates an image by queueing a prompt in ComfyUI."""
        positive_node_id = self._find_node_id_by_title("positive_prompt")
        negative_node_id = self._find_node_id_by_title("negative_prompt")
        load_image_node_ids = self._find_node_ids_by_class_type("Image Load")
        save_image_node_ids = self._find_node_ids_by_class_type("SaveImage")
        # Find KSampler nodes to ensure seed is randomized for each run
        ksampler_node_ids = self._find_node_ids_by_class_type("KSampler")
        
        if not positive_node_id:
            logging.error("Could not find node with title 'positive_prompt' in the workflow.")
            return None
        if not negative_node_id:
            logging.error("Could not find node with title 'negative_prompt' in the workflow.")
            return None

        # Create a copy of the workflow
        workflow_copy = copy.deepcopy(self.workflow)

        # Update the prompts in the copied workflow
        workflow_copy[positive_node_id]["inputs"]["prompt"] = positive_prompt
        workflow_copy[negative_node_id]["inputs"]["prompt"] = negative_prompt

        # Set KSampler seed and steps.
        if ksampler_node_ids:
            for node_id in ksampler_node_ids:
                # Set a new random seed for each execution.
                workflow_copy[node_id]["inputs"]["seed"] = random.randint(0, 2**32 - 1)
                workflow_copy[node_id]["inputs"]["steps"] = 8

        # Update image paths in "Image Load" nodes and handle surplus nodes
        if load_image_node_ids:
            num_mood_images = len(mood_images) if mood_images else 0
            num_load_nodes = len(load_image_node_ids)

            # Assign images to available load nodes
            for i, node_id in enumerate(load_image_node_ids):
                if i < num_mood_images:
                    workflow_copy[node_id]["inputs"]["image_path"] = str(mood_images[i].resolve())
                else:
                    # This is a surplus "Image Load" node. Remove it and its references.
                    logging.info(f"Removing surplus 'Image Load' node {node_id} and its references.")
                    # Remove the input from the positive prompt node (e.g., "image2", "image3")
                    # This assumes inputs are named image1, image2, ...
                    image_input_key = f"image{i+1}"
                    if image_input_key in workflow_copy[positive_node_id]["inputs"]:
                        del workflow_copy[positive_node_id]["inputs"][image_input_key]

                    # Remove the node itself from the workflow
                    del workflow_copy[node_id]
        
        # Determine the output path from the SaveImage node
        output_path_prefix = None
        if save_image_node_ids:
            save_node_id = save_image_node_ids[0] # Assume the first SaveImage node is the target
            prefix = workflow_copy[save_node_id]["inputs"].get("filename_prefix")
            # The prefix in ComfyUI is relative to its output directory.
            # We only need the directory part of the prefix.
            output_path_prefix = self.comfyui_output_dir / Path(prefix).parent
            logging.info(f"Resolved image output directory to: {output_path_prefix}")
        if not output_path_prefix:
            logging.error("Could not determine output path from SaveImage node in workflow.")

        # Save the modified workflow for debugging purposes
        try:
            debug_workflow_path = self.workflow_path.with_name(f"{self.workflow_path.stem}_debug.json")
            with debug_workflow_path.open('w', encoding='utf-8') as f:
                json.dump(workflow_copy, f, indent=2)
            logging.info(f"Saved modified workflow for debugging to: {debug_workflow_path}")
        except Exception as e:
            logging.error(f"Could not save debug workflow file: {e}")

        # Filter out non-node entries and the '_meta' key from the workflow before sending.
        # The ComfyUI API backend does not recognize the '_meta' field.
        prompt_data = {}
        for node_id, node_data in workflow_copy.items():
            if isinstance(node_data, dict) and 'class_type' in node_data:
                prompt_data[node_id] = {k: v for k, v in node_data.items() if k != '_meta'}

        # Run the prompt (= workflow) in ComfyUI
        payload = {"prompt": prompt_data, "client_id": self.client_id}
        try:
            response = self.session.post(f"{self.comfyui_url}prompt", json=payload)
            response.raise_for_status()
            result = response.json()
            return result.get('prompt_id'), output_path_prefix
        except requests.exceptions.RequestException as e:
            logging.error(f"Failed to queue prompt in ComfyUI: {e}")
            return None, None

    def wait_for_generation(self, 
                            prompt_id: str, 
                            output_dir: Path) -> Optional[Path]:
        """
        Waits for the generation to complete and returns the path to the first output image.
        Note: This assumes the ComfyUI output directory is accessible.
        """
        if not prompt_id or not output_dir:
            return None

        logging.info(f"Successfully queued prompt. Prompt ID: {prompt_id}. Waiting for completion...")
        # You would typically implement a WebSocket connection here to get real-time status.
        # For simplicity, we will poll the history endpoint.
        while True:
            try:
                history_url = f"{self.comfyui_url}history/{prompt_id}"
                response = self.session.get(history_url)
                response.raise_for_status()
                history = response.json()

                if prompt_id in history and history[prompt_id].get("outputs"):
                    logging.info(f"Generation complete for prompt ID: {prompt_id}")
                    outputs = history[prompt_id]["outputs"]
                    # Find the first output node that has images
                    for node_output in outputs.values():
                        if "images" in node_output:
                            image_info = node_output["images"][0]
                            filename = image_info["filename"]
                            # Construct the full path using the directory derived from the workflow's SaveImage node
                            image_path = output_dir / filename
                            if image_path.exists():
                                logging.info(f"Found generated image: {image_path}")
                                return image_path
                            else:
                                logging.error(f"Could not find generated image at expected path: {image_path}")
                            return None # Return after finding the first image
                    return None # No image outputs found
            except requests.exceptions.RequestException as e:
                logging.error(f"Error checking history for prompt {prompt_id}: {e}")
                break
            time.sleep(1) # Add a delay to prevent overwhelming the server and causing socket errors


if __name__ == "__main__":
    
    # Get mood images directory from configuration
    mood_images_dir = Path(SYS_CONFIG.get("mood_images_dir", "./mood_images/"))    
    mood_images = []
    if mood_images_dir.is_dir():
        mood_images = list(mood_images_dir.glob('*'))

    # Load the briefing from the markdown file defined in configuration
    briefing_path = Path(SYS_CONFIG["prompts"]["briefing"])
    try:
        brief = briefing_path.read_text(encoding='utf-8').strip()
    except FileNotFoundError:
        logging.error(f"Briefing file not found at: {briefing_path}")
        exit(1)
    num_generations = 1

    # Initialize the prompter class
    try:
        promptr = prompter()
    except (FileNotFoundError, ollama.ResponseError):
        exit(1) # Exit if model preloading fails

    # Create a summary of available inputs for the producer
    num_mood_images = len(mood_images)
    input_summary = f"Available inputs: {num_mood_images} mood image(s), 1 text prompt."

    # Have the producer select the workflow ---
    logging.info("--- Running Producer to select workflow ---")
    workflows_dir = Path(SYS_CONFIG.get("comfyui_workflows_dir"))
    producer_choice = promptr.select_workflow(brief, input_summary, workflows_dir)

    if not producer_choice or not producer_choice.get("workflow_path"):
        logging.error("Producer failed to select a valid workflow. Exiting.")
        exit(1)

    selected_workflow = producer_choice["workflow_path"]
    logging.info(f"Producer selected workflow: {selected_workflow}")

    # Generate the prompt using the LLM, informed by the producer's directive ---
    logging.info("--- Running Writer to generate prompts ---")
    prompt_type = producer_choice.get("prompt_type")
    if prompt_type:
        logging.info(f"Producer suggested prompt type: {prompt_type}")
    prompt = promptr.create_image_prompt(brief, selected_workflow, mood_images, prompt_type=prompt_type)

    if "error" in prompt:
        logging.error("Writer failed to generate prompt. Exiting.")
        exit(1)

    logging.info("Writer successfully generated prompts.")
    positive_prompt = prompt.get("positive_prompt", "")
    negative_prompt = prompt.get("negative_prompt", "")
    logging.info(f"Positive Prompt: {positive_prompt}")
    logging.info(f"Negative Prompt: {negative_prompt}")

    # Generate the image using ComfyUI ---
    logging.info("Initializing generator for image creation...")
    try:
        override = SYS_CONFIG.get("comfyui_output_dir_override")
        vfxguy = generator(
            workflow_path=selected_workflow,
            comfyui_output_dir=Path(override) if override else None)
    
    except Exception as e:
       logging.error(f"Generator initialization failed: {e}")
       exit(1)

    if positive_prompt:
        # --- Release LLM from memory before loading diffusion models ---
        logging.info("Releasing LLM from VRAM to free up resources for image generation...")
        del promptr
        # A small delay can help ensure resources are fully released.
        time.sleep(5)
        logging.info("LLM resources should now be free.")


        logging.info(f"--- Queuing {num_generations} generation jobs in ComfyUI ---")
        
        for i in range(num_generations):
            logging.info(f"Queuing job {i+1}/{num_generations}...")
            prompt_id, output_path = vfxguy.generate(positive_prompt, negative_prompt, mood_images)
            if prompt_id:
                logging.info(f"--- Waiting for job {i+1}/{num_generations} (Prompt ID: {prompt_id}) ---")
                generated_image = vfxguy.wait_for_generation(prompt_id, output_path)
    else:
        logging.error("No positive prompt was generated, cannot create image.")

    # Supervise the generated image ---

    if generated_image:
        # Reload the LLM for the supervision task ---
        logging.info("Re-initializing prompter to supervise the generated image...")
        try:
            promptr = prompter()
        except (FileNotFoundError, ollama.ResponseError):
            logging.error("Failed to re-initialize prompter for supervision. Skipping.")

        logging.info("--- Starting supervisor step ---")
        verdict = promptr.supervise(brief, generated_image, mood_images, positive_prompt)
        if "error" not in verdict:
            logging.info(f"Supervisor verdict: {'APPROVED' if verdict.get('approved') else 'REJECTED'}")
            logging.info(f"Reason: {verdict.get('reason')}")
            logging.info(f"ToDo: {verdict.get('todo')}")
            if verdict.get('prompt_suggestion'):
                logging.info(f"Prompt suggestion: {verdict.get('prompt_suggestion')}")
        else:
            logging.error(f"Supervisor step failed: {verdict.get('error')}")
    else:
        logging.warning("Could not retrieve generated image for supervision.")