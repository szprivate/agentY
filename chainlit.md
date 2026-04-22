**agentY** is a ComfyUI AI agent powered by a multi-stage pipeline:

- **Triage** - classifies your request and routes it to the right handler
- **Researcher** - resolves your request into a structured workflow specification
- **Brain** - assembles, executes, and quality-checks the ComfyUI workflow

---

## What you can do

- **Generate images** - "Generate a cinematic wide-shot of Tokyo neon streets at night"
- **Edit images** - attach a photo and describe the change you want
- **Style transfer** - "Apply a Studio Ghibli style to my image"
- **Batch generation** - "Create 5 variations of this portrait in different lighting"
- **Planned batched workflows** - "Create an image, upscale it, then animate it with Kling 3.0"
- **Video generation** - "Animate this image as a slow zoom-in"

## Attaching images

Click the attachment button to upload images directly into the chat.
agentY will automatically detect them and wire them into the correct ComfyUI nodes.

## Tips

- Be descriptive - more detail produces better results
- Mention aspect ratio, lighting mood, or style references when relevant
- Follow up naturally: "Make it warmer", "Try a higher contrast version"

---

*Configuration is read from `config/settings.json` and `.env` in the project root.*
