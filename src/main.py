import json
import os
from typing import Any

import requests
from pydantic import BaseModel, Field
from strands import Agent, tool
from strands.models.openai import OpenAIModel


BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def resolve_path(path: str | None) -> str | None:
	if not path:
		return path
	if os.path.isabs(path):
		return path
	return os.path.normpath(os.path.join(BASE_DIR, path))


def load_config() -> dict[str, Any]:
	config_path = resolve_path("./config/settings.json")
	if not config_path:
		raise RuntimeError("Missing config path.")
	with open(config_path, "r", encoding="utf-8") as file:
		return json.load(file)


config = load_config()
MCP_REQUEST_HEADERS = {
	"Content-Type": "application/json",
	"Accept": "application/json, text/event-stream",
}


class PromptOutput(BaseModel):
	prompt: str = Field(..., description="A production-ready creative prompt.")


class SupervisionOutput(BaseModel):
	accepted: bool = Field(..., description="Whether the result should be accepted.")
	supervision: str = Field(..., description="A short review explaining the decision.")


class FinalResult(BaseModel):
	accepted: bool
	supervision: str
	output_image: str | None = None
	prompt: str
	brief: str


def ensure_openai_base_url(url: str | None) -> str:
	base_url = (url or "http://localhost:11434/v1").rstrip("/")
	if base_url.endswith("/v1"):
		return base_url
	return f"{base_url}/v1"


def ensure_mcp_url(url: str | None) -> str:
	return (url or "http://127.0.0.1:9000/mcp").rstrip("/")


def create_model() -> OpenAIModel:
	model_id = config.get("ollama-model")
	if not model_id:
		raise RuntimeError("Missing 'ollama-model' in config/settings.json.")

	api_key = (
		os.getenv("OLLAMA_API_KEY")
		or os.getenv("OPENAI_API_KEY")
		or "ollama"
	)

	return OpenAIModel(
		model_id=model_id,
		client_args={
			"api_key": api_key,
			"base_url": ensure_openai_base_url(config.get("ollama_api_url")),
		},
	)


def read_text_file(path: str | None) -> str:
	if not path or not os.path.exists(path):
		return ""
	with open(path, "r", encoding="utf-8") as file:
		return file.read().strip()


def collect_mood_images() -> list[str]:
	mood_dir = resolve_path(config.get("mood_images_dir", "./mood_images/"))
	if not mood_dir or not os.path.isdir(mood_dir):
		return []
	return [
		os.path.join(mood_dir, file_name)
		for file_name in os.listdir(mood_dir)
		if file_name.lower().endswith((".jpg", ".jpeg", ".png", ".webp"))
	]


def select_workflow_file() -> str:
	configured_workflow = resolve_path(config.get("comfyui_workflow"))
	if configured_workflow and os.path.exists(configured_workflow):
		return configured_workflow

	workflows_dir = resolve_path(config.get("comfyui_workflows_dir", "./comfyui_workflows/"))
	if not workflows_dir or not os.path.isdir(workflows_dir):
		raise RuntimeError("Configured workflow directory does not exist.")

	workflow_files = sorted(
		file_name for file_name in os.listdir(workflows_dir) if file_name.endswith(".json")
	)
	if not workflow_files:
		raise RuntimeError("No ComfyUI workflow JSON files were found.")

	return os.path.join(workflows_dir, workflow_files[0])


def load_workflow_template(workflow_file: str) -> dict[str, Any]:
	with open(workflow_file, "r", encoding="utf-8") as file:
		return json.load(file)


def replace_prompt_fields(obj: Any, prompt: str):
	if isinstance(obj, dict):
		for key, value in obj.items():
			if key == "prompt":
				obj[key] = prompt
			else:
				replace_prompt_fields(value, prompt)
	elif isinstance(obj, list):
		for item in obj:
			replace_prompt_fields(item, prompt)


def replace_image_fields(obj: Any, images: list[str]):
	if isinstance(obj, dict):
		for key, value in obj.items():
			if key == "image_path" and images:
				obj[key] = images.pop(0)
			else:
				replace_image_fields(value, images)
	elif isinstance(obj, list):
		for item in obj:
			replace_image_fields(item, images)


def prepare_workflow_payload(workflow_file: str, prompt: str, input_images: list[str]) -> dict[str, Any]:
	workflow = load_workflow_template(workflow_file)
	replace_prompt_fields(workflow, prompt)
	replace_image_fields(workflow, input_images.copy())
	return workflow


def parse_sse_response(response_text: str) -> dict[str, Any]:
	for line in response_text.replace("\r\n", "\n").split("\n"):
		line = line.strip()
		if line.startswith("data: "):
			payload = line[6:]
			try:
				return json.loads(payload)
			except json.JSONDecodeError:
				continue
	raise ValueError("No valid JSON data found in MCP SSE response.")


def mcp_request(method: str, params: dict[str, Any], request_id: int = 1) -> dict[str, Any]:
	mcp_url = ensure_mcp_url(config.get("comfyui_mcp_url"))
	response = requests.post(
		mcp_url,
		json={
			"jsonrpc": "2.0",
			"id": request_id,
			"method": method,
			"params": params,
		},
		headers=MCP_REQUEST_HEADERS,
		timeout=300,
	)
	response.raise_for_status()
	content_type = response.headers.get("content-type", "")
	if "text/event-stream" in content_type:
		return parse_sse_response(response.text)
	return response.json()


def mcp_call_tool(tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
	response = mcp_request(
		"tools/call",
		{"name": tool_name, "arguments": arguments},
		request_id=2,
	)
	if "error" in response:
		raise RuntimeError(json.dumps(response["error"], ensure_ascii=False))

	result = response.get("result", {})
	content = result.get("content")
	if isinstance(content, list) and content:
		first_item = content[0]
		if isinstance(first_item, dict) and "text" in first_item:
			text_value = first_item["text"]
			if isinstance(text_value, dict):
				return text_value
			try:
				return json.loads(text_value)
			except (json.JSONDecodeError, TypeError):
				return {"message": text_value}
	if isinstance(result, dict):
		return result
	raise RuntimeError("Unexpected MCP response format.")


def run_comfyui_workflow(workflow: dict[str, Any], workflow_id: str) -> dict[str, Any]:
	return mcp_call_tool(
		"run_raw_workflow",
		{
			"workflow": workflow,
			"workflow_id": workflow_id,
			"return_inline_preview": False,
		},
	)


def extract_output_image(result: dict[str, Any]) -> str | None:
	asset_url = result.get("asset_url") or result.get("image_url")
	if isinstance(asset_url, str) and asset_url:
		return asset_url

	outputs = result.get("outputs") if isinstance(result, dict) else None
	if isinstance(outputs, dict):
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
				return (
					f"{(config.get('comfyui_url') or '').rstrip('/')}"
					f"/view?filename={filename}&subfolder={subfolder}&type={folder_type}"
				)

	stack: list[Any] = [result]
	while stack:
		current = stack.pop()
		if isinstance(current, dict):
			stack.extend(current.values())
		elif isinstance(current, list):
			stack.extend(current)
		elif isinstance(current, str) and current.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
			return current
	return None


def summarize_for_llm(value: Any, limit: int = 4000) -> str:
	text = json.dumps(value, indent=2, ensure_ascii=False, default=str)
	if len(text) <= limit:
		return text
	return f"{text[:limit]}\n... [truncated]"


def unload_ollama_model():
	ollama_url = (config.get("ollama_api_url") or "").rstrip("/")
	ollama_model = config.get("ollama-model")
	if not ollama_url or not ollama_model:
		return

	api_base = ollama_url[:-3] if ollama_url.endswith("/v1") else ollama_url
	try:
		response = requests.post(
			f"{api_base}/api/generate",
			json={
				"model": ollama_model,
				"prompt": "",
				"keep_alive": 0,
			},
			timeout=30,
		)
		response.raise_for_status()
	except requests.RequestException:
		pass


model = create_model()
_creator_state: dict[str, Any] = {}


prompter_agent = Agent(
	model=model,
	name="prompter",
	description="Turns a creative brief into a production-ready prompt.",
	system_prompt=(
		"You are a creative image prompt engineer. "
		"Turn the user's brief, mood references, and instructions into one strong image-generation prompt. "
		"Be specific about composition, camera angle, lighting, and style."
	),
)


@tool
def submit_workflow(prompt: str, workflow_file: str, input_images: list[str]) -> dict[str, Any]:
	"""Render an image by filling a ComfyUI workflow and submitting it to the MCP endpoint."""
	workflow = prepare_workflow_payload(workflow_file, prompt, input_images)
	workflow_id = os.path.splitext(os.path.basename(workflow_file))[0]
	unload_ollama_model()
	result = run_comfyui_workflow(workflow, workflow_id=workflow_id)
	output_image = extract_output_image(result)
	payload = {
		"result": result,
		"output_image": output_image,
	}
	_creator_state["last_payload"] = payload
	return {
		"output_image": output_image,
		"result_summary": summarize_for_llm(result, limit=2000),
	}


creator_agent = Agent(
	model=model,
	name="creator",
	description="Submits a prepared prompt and workflow to ComfyUI.",
	tools=[submit_workflow],
	system_prompt=(
		"You are the creator agent. "
		"Always call `submit_workflow` exactly once using the values provided in the user's message. "
		"After the tool succeeds, reply with a short factual summary."
	),
)


supervisor_agent = Agent(
	model=model,
	name="supervisor",
	description="Reviews whether the generated output satisfies the brief.",
	system_prompt=(
		"You are an expert creative supervisor. "
		"Assess whether the output matches the brief, mood references, and input images. "
		"Be strict but concise."
	),
)


def run_prompter_agent(brief: str, mood_images: list[str], instructions: str) -> PromptOutput:
	mood_list = ", ".join(os.path.basename(path) for path in mood_images) or "None"
	result = prompter_agent(
		(
			f"Brief:\n{brief}\n\n"
			f"Mood images:\n{mood_list}\n\n"
			f"Instructions:\n{instructions or 'None'}"
		),
		structured_output_model=PromptOutput,
	)
	if result.structured_output is None:
		raise RuntimeError("Prompter agent did not return structured output.")
	return result.structured_output


def run_creator_agent(prompt: str, workflow_file: str, input_images: list[str]) -> dict[str, Any]:
	_creator_state.clear()
	creator_agent(
		json.dumps(
			{
				"prompt": prompt,
				"workflow_file": workflow_file,
				"input_images": input_images,
			},
			indent=2,
		),
	)
	payload = _creator_state.get("last_payload")
	if payload is None:
		raise RuntimeError("Creator agent did not submit a workflow.")
	return payload


def run_supervisor_agent(
	brief: str,
	prompt: str,
	mood_images: list[str],
	input_images: list[str],
	creator_payload: dict[str, Any],
) -> SupervisionOutput:
	mood_list = ", ".join(os.path.basename(path) for path in mood_images) or "None"
	input_list = ", ".join(os.path.basename(path) for path in input_images) or "None"
	result = supervisor_agent(
		(
			f"Brief:\n{brief}\n\n"
			f"Prompt:\n{prompt}\n\n"
			f"Mood images:\n{mood_list}\n\n"
			f"Input images:\n{input_list}\n\n"
			f"Output image:\n{creator_payload.get('output_image')}\n\n"
			f"ComfyUI result summary:\n{summarize_for_llm(creator_payload.get('result', {}), limit=2500)}\n\n"
			"Decide whether the result should be accepted."
		),
		structured_output_model=SupervisionOutput,
	)
	if result.structured_output is None:
		raise RuntimeError("Supervisor agent did not return structured output.")
	return result.structured_output


@tool
def run_prompter(brief: str, mood_images: list[str], instructions: str) -> dict[str, Any]:
	"""Generate a polished image prompt from the user's brief and references."""
	return run_prompter_agent(brief, mood_images, instructions).model_dump()


@tool
def run_creator(prompt: str, workflow_file: str, input_images: list[str]) -> dict[str, Any]:
	"""Create an image by submitting the selected workflow to ComfyUI."""
	return run_creator_agent(prompt, workflow_file, input_images)


@tool
def run_supervisor(
	brief: str,
	prompt: str,
	mood_images: list[str],
	input_images: list[str],
	creator_payload: dict[str, Any],
) -> dict[str, Any]:
	"""Review the generated output and decide whether it satisfies the brief."""
	return run_supervisor_agent(brief, prompt, mood_images, input_images, creator_payload).model_dump()


orchestrator_agent = Agent(
	model=model,
	name="orchestrator",
	description="Coordinates prompt creation, image generation, and review.",
	tools=[run_prompter, run_creator, run_supervisor],
	system_prompt=(
		"You are the orchestration agent. "
		"Use the tools in this exact order: `run_prompter`, then `run_creator`, then `run_supervisor`. "
		"Do not skip any step. "
		"When you finish, return the final accepted flag, supervision summary, output image path, prompt, and brief."
	),
)


def run_generation_pipeline(brief: str, input_images: list[str] | None = None) -> dict[str, Any]:
	workflow_file = select_workflow_file()
	mood_images = collect_mood_images()
	instructions = read_text_file(resolve_path(config.get("prompts", {}).get("briefing")))
	orchestrator_result = orchestrator_agent(
		json.dumps(
			{
				"brief": brief,
				"workflow_file": workflow_file,
				"input_images": input_images or [],
				"mood_images": mood_images,
				"instructions": instructions,
			},
			indent=2,
		),
		structured_output_model=FinalResult,
	)
	if orchestrator_result.structured_output is None:
		raise RuntimeError("Orchestrator agent did not return structured output.")
	return orchestrator_result.structured_output.model_dump()


def main():
	brief = read_text_file(resolve_path(config.get("prompts", {}).get("briefing")))
	if not brief:
		print("No briefing text found. Check the prompts.briefing path in settings.json.")
		return

	try:
		result = run_generation_pipeline(brief=brief)
		print(json.dumps(result, indent=2, ensure_ascii=False))
	finally:
		unload_ollama_model()


if __name__ == "__main__":
	main()
