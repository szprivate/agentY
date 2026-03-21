# import statements
from collections import deque
from dataclasses import dataclass
import json
import os
import time
from typing import Any
import uuid


BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def resolve_path(path: str | None) -> str | None:
	if not path:
		return path
	if os.path.isabs(path):
		return path
	return os.path.normpath(os.path.join(BASE_DIR, path))


@dataclass(slots=True)
class Message:
	sender: str
	recipient: str
	data: dict[str, Any]


class Agent:
	def __init__(self, name: str):
		self.name = name
		self._runtime = None

	def attach_runtime(self, runtime: "AgentRuntime"):
		self._runtime = runtime

	def send(self, recipient: str, data: dict[str, Any]):
		if self._runtime is None:
			raise RuntimeError(f"Agent '{self.name}' is not attached to a runtime.")
		self._runtime.enqueue(Message(sender=self.name, recipient=recipient, data=data))

	def on_message(self, message: Message):
		raise NotImplementedError


class AgentRuntime:
	def __init__(self, agents: list[Agent]):
		self.agents = {agent.name: agent for agent in agents}
		self.queue: deque[Message] = deque()
		self.external_messages: list[Message] = []
		for agent in agents:
			agent.attach_runtime(self)

	def enqueue(self, message: Message):
		if message.recipient in self.agents:
			self.queue.append(message)
		else:
			self.external_messages.append(message)

	def run(self, initial_messages: list[Message] | None = None, max_steps: int = 100):
		for message in initial_messages or []:
			self.enqueue(message)

		steps = 0
		while self.queue:
			if steps >= max_steps:
				raise RuntimeError(f"Agent runtime exceeded max_steps={max_steps}.")
			message = self.queue.popleft()
			self.agents[message.recipient].on_message(message)
			steps += 1

		return self.external_messages


def run_agents(agents: list[Agent], initial_messages: list[Message] | None = None, max_steps: int | None = None):
	runtime = AgentRuntime(agents)
	if not initial_messages:
		print(f"Initialized {len(agents)} agents. No initial messages queued.")
		return []
	return runtime.run(initial_messages=initial_messages, max_steps=max_steps or 100)


def load_briefing_text() -> str:
	briefing_path = resolve_path(config.get("prompts", {}).get("briefing"))
	if not briefing_path or not os.path.exists(briefing_path):
		return ""
	with open(briefing_path, "r", encoding="utf-8") as f:
		return f.read().strip()


def print_external_messages(messages: list[Message]):
	if not messages:
		print("Run completed, but no user-facing messages were produced.")
		return

	print(f"Run completed with {len(messages)} message(s):")
	for index, message in enumerate(messages, start=1):
		print(f"\n[{index}] {message.sender} -> {message.recipient}")
		print(json.dumps(message.data, indent=2, ensure_ascii=False))


def update_workflow_prompt(workflow: dict[str, Any], prompt: str) -> bool:
	replaced = False

	for node in workflow.values():
		if not isinstance(node, dict):
			continue
		inputs = node.get("inputs")
		if not isinstance(inputs, dict):
			continue

		title = str(node.get("_meta", {}).get("title", "")).lower()
		class_type = str(node.get("class_type", "")).lower()

		if "prompt" in inputs:
			inputs["prompt"] = prompt
			replaced = True
		elif class_type == "cliptextencode" and "text" in inputs and "negative" not in title:
			inputs["text"] = prompt
			replaced = True

	return replaced


def extract_comfyui_output_images(result: dict[str, Any]) -> list[str]:
	comfyui_url = config.get("comfyui_url", "").rstrip("/")
	image_urls: list[str] = []
	outputs = result.get("outputs", {}) if isinstance(result, dict) else {}

	for node_output in outputs.values():
		if not isinstance(node_output, dict):
			continue
		for image in node_output.get("images", []):
			if not isinstance(image, dict):
				continue
			filename = image.get("filename")
			if not filename:
				continue
			subfolder = image.get("subfolder", "")
			folder_type = image.get("type", "output")
			image_urls.append(
				f"{comfyui_url}/view?filename={filename}&subfolder={subfolder}&type={folder_type}"
			)

	return image_urls


def run_comfyui_workflow(workflow: dict[str, Any]) -> dict[str, Any]:
	import requests

	comfyui_url = config.get("comfyui_url", "").rstrip("/")
	client_id = str(uuid.uuid4())
	queue_response = requests.post(
		comfyui_url + "/prompt",
		json={"prompt": workflow, "client_id": client_id},
		timeout=120,
	)
	queue_response.raise_for_status()
	queue_data = queue_response.json()
	prompt_id = queue_data.get("prompt_id")
	if not prompt_id:
		return queue_data

	max_polls = int(config.get("comfyui_poll_attempts", 120))
	poll_interval = float(config.get("comfyui_poll_interval", 2))
	history_url = comfyui_url + f"/history/{prompt_id}"

	for _ in range(max_polls):
		history_response = requests.get(history_url, timeout=30)
		history_response.raise_for_status()
		history = history_response.json()
		result = history.get(str(prompt_id)) or history.get(prompt_id)
		if isinstance(result, dict) and result.get("outputs"):
			return result
		time.sleep(poll_interval)

	return {
		"prompt_id": prompt_id,
		"status": "timeout",
		"message": f"Workflow queued but did not finish after {max_polls} polls.",
	}

# Agent definitions
class OrchestratorAgent(Agent):
	def on_message(self, message: Message):
		if "brief" in message.data:
			self._start_generation(message)
			return
		if "prompt" in message.data:
			self._handle_prompt(message)
			return
		if "supervision" in message.data or "accepted" in message.data:
			self._handle_supervision(message)
			return
		if "error" in message.data:
			self._handle_error(message)

	def _start_generation(self, message: Message):
		# Expecting message.data = {"brief": str, "input_images": list}
		brief = message.data.get("brief") or load_briefing_text()
		input_images = message.data.get("input_images", [])
		if not brief:
			self.send(message.sender, {"error": "No briefing text available. Set prompts.briefing in settings.json."})
			return

		# 1. Collect mood images
		mood_dir = resolve_path(config.get("mood_images_dir", "./mood_images/"))
		mood_images = [os.path.join(mood_dir, f) for f in os.listdir(mood_dir) if f.lower().endswith(('.jpg', '.png'))]

		# 2. Select workflow (simple: pick first workflow in comfyui_workflows_dir)
		configured_workflow = resolve_path(config.get("comfyui_workflow"))
		workflows_dir = resolve_path(config.get("comfyui_workflows_dir", "./comfyui_workflows/"))
		workflow_files = [f for f in os.listdir(workflows_dir) if f.endswith('.json')]
		if configured_workflow and os.path.exists(configured_workflow):
			selected_workflow = configured_workflow
		elif workflow_files:
			selected_workflow = os.path.join(workflows_dir, workflow_files[0])
		else:
			self.send(message.sender, {"error": "No workflows found."})
			return

		# 3. Ask prompter to generate a prompt
		prompter_msg = {
			"brief": brief,
			"mood_images": mood_images
		}
		self.send("prompter", prompter_msg)

		# 4. Store state for next step (simulate FSM)
		self._pending = {
			"input_images": input_images,
			"selected_workflow": selected_workflow,
			"brief": brief,
			"mood_images": mood_images,
			"original_sender": message.sender,
			"iterations": 0
		}

	def _handle_prompt(self, message: Message):
		if not hasattr(self, '_pending'):
			return
		prompt = message.data.get("prompt")
		if not prompt:
			self.send(self._pending["original_sender"], {"error": "No prompt generated."})
			return
		self._pending["prompt"] = prompt
		# Forward to creator
		creator_msg = {
			"workflow_file": self._pending["selected_workflow"],
			"prompt": prompt,
			"input_images": self._pending["input_images"],
			"brief": self._pending["brief"],
			"mood_images": self._pending["mood_images"]
		}
		self.send("creator", creator_msg)

	def _handle_supervision(self, message: Message):
		if not hasattr(self, '_pending'):
			return
		final_message = {
			"accepted": message.data.get("accepted", False),
			"supervision": message.data.get("supervision"),
			"output_image": message.data.get("output_image"),
			"prompt": self._pending.get("prompt"),
			"brief": self._pending.get("brief")
		}
		self.send(self._pending["original_sender"], final_message)
		del self._pending

	def _handle_error(self, message: Message):
		if hasattr(self, '_pending'):
			self.send(self._pending["original_sender"], message.data)
			del self._pending

class PrompterAgent(Agent):
	def on_message(self, message: Message):
		# message.data: {"brief": str, "mood_images": list}
		brief = message.data.get("brief", "")
		mood_images = message.data.get("mood_images", [])

		# Compose prompt for LLM
		# Optionally, read briefing.md for extra context
		briefing_path = resolve_path(config["prompts"].get("briefing"))
		briefing_text = ""
		if briefing_path and os.path.exists(briefing_path):
			with open(briefing_path, "r", encoding="utf-8") as f:
				briefing_text = f.read()

		# Prepare LLM input
		llm_input = f"Brief: {brief}\nMood images: {', '.join(os.path.basename(m) for m in mood_images)}\nInstructions: {briefing_text}"

		# Call Ollama LLM server
		import requests
		ollama_url = config.get("ollama_api_url")
		ollama_model = config.get("ollama-model")
		try:
			response = requests.post(
				ollama_url.rstrip("/") + "/chat/completions",
				json={
					"model": ollama_model,
					"messages": [
						{"role": "system", "content": "You are a creative image prompt engineer."},
						{"role": "user", "content": llm_input}
					]
				},
				timeout=config.get("ollama_timeout", 60)
			)
			response.raise_for_status()
			data = response.json()
			prompt = data["choices"][0]["message"]["content"]
		except Exception as e:
			self.send(message.sender, {"error": f"LLM error: {e}"})
			return

		# Reply to orchestrator
		self.send("orchestrator", {"prompt": prompt})

class CreatorAgent(Agent):
	def on_message(self, message: Message):
		# message.data: {"workflow_file", "prompt", "input_images", "brief", "mood_images"}
		workflow_file = message.data.get("workflow_file")
		prompt = message.data.get("prompt")
		input_images = message.data.get("input_images", [])
		# Load workflow JSON
		try:
			with open(workflow_file, "r", encoding="utf-8") as f:
				workflow = json.load(f)
		except Exception as e:
			self.send(message.sender, {"error": f"Workflow load error: {e}"})
			return

		if prompt:
			update_workflow_prompt(workflow, prompt)

		# Replace reference images in workflow (if any, adjust as needed)
		def replace_images(obj, images):
			if isinstance(obj, dict):
				for k, v in obj.items():
					if k == "image_path" and images:
						obj[k] = images.pop(0)
					else:
						replace_images(v, images)
			elif isinstance(obj, list):
				for item in obj:
					replace_images(item, images)
		replace_images(workflow, input_images.copy())

		# Send workflow to ComfyUI server
		try:
			result = run_comfyui_workflow(workflow)
		except Exception as e:
			self.send(message.sender, {"error": f"ComfyUI error: {e}"})
			return

		# Send result to supervisor
		supervisor_msg = {
			"result": result,
			"brief": message.data.get("brief"),
			"mood_images": message.data.get("mood_images"),
			"input_images": input_images,
			"prompt": prompt
		}
		self.send("supervisor", supervisor_msg)

class SupervisorAgent(Agent):
	def on_message(self, message: Message):
		# message.data: {"result", "brief", "mood_images", "input_images", "prompt"}
		import requests
		result = message.data.get("result")
		brief = message.data.get("brief")
		mood_images = message.data.get("mood_images", [])
		input_images = message.data.get("input_images", [])
		prompt = message.data.get("prompt")

		# Extract output image path from result
		output_image = None
		if isinstance(result, dict):
			output_images = extract_comfyui_output_images(result)
			if output_images:
				output_image = output_images[0]
		# Compose evaluation prompt for LLM
		eval_prompt = f"Brief: {brief}\nPrompt: {prompt}\nMood images: {', '.join(os.path.basename(m) for m in mood_images)}\nInput images: {', '.join(os.path.basename(i) for i in input_images)}\nOutput image: {output_image}\nAnalyze if the output image matches the brief and mood. Reply with 'ACCEPT' or 'REJECT' and a short reason."

		# Call Ollama LLM for evaluation
		ollama_url = config.get("ollama_api_url")
		ollama_model = config.get("ollama-model")
		try:
			response = requests.post(
				ollama_url.rstrip("/") + "/chat/completions",
				json={
					"model": ollama_model,
					"messages": [
						{"role": "system", "content": "You are an expert creative supervisor. Evaluate the result strictly."},
						{"role": "user", "content": eval_prompt}
					]
				},
				timeout=config.get("ollama_timeout", 60)
			)
			response.raise_for_status()
			data = response.json()
			verdict = data["choices"][0]["message"]["content"]
		except Exception as e:
			self.send("orchestrator", {"error": f"LLM evaluation error: {e}"})
			return

		# Parse verdict
		accept = verdict.strip().upper().startswith("ACCEPT")
		# Notify orchestrator
		self.send("orchestrator", {
			"supervision": verdict,
			"accepted": accept,
			"output_image": output_image
		})

# Load config
with open(resolve_path('./config/settings.json'), 'r', encoding='utf-8') as f:
	config = json.load(f)

# Instantiate agents
orchestrator = OrchestratorAgent('orchestrator')
prompter = PrompterAgent('prompter')
creator = CreatorAgent('creator')
supervisor = SupervisorAgent('supervisor')

# Register agents
agents = [orchestrator, prompter, creator, supervisor]

if __name__ == '__main__':
	brief = load_briefing_text()
	initial_messages = []
	if brief:
		print("Loaded briefing from settings.json and queued it for the orchestrator.")
		initial_messages.append(
			Message(
				sender="user",
				recipient="orchestrator",
				data={"brief": brief, "input_images": []},
			)
		)
	else:
		print("No briefing text found. Check the prompts.briefing path in settings.json.")

	results = run_agents(agents, initial_messages=initial_messages)
	print_external_messages(results)
