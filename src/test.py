from ollama import generate
response = generate('qwen3-vl:latest', 'Why is the sky blue?')
print(response['response'])