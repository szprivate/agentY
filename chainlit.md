# Welcome to agentY 🎨

**agentY** is a ComfyUI AI agent powered by a two-stage pipeline:

- **Researcher** — resolves your request into a structured workflow specification
- **Brain** — assembles, executes, and quality-checks the ComfyUI workflow

---

## What you can do

- **Generate images** — _"Generate a cinematic wide-shot of Tokyo neon streets at night"_
- **Edit images** — attach a photo and describe the change you want
- **Style transfer** — _"Apply a Studio Ghibli style to my image"_
- **Batch generation** — _"Create 5 variations of this portrait in different lighting"_
- **Video generation** — _"Animate this image as a slow zoom-in"_

## Attaching images

Click the **📎 attachment** button to upload images directly into the chat.  
agentY will automatically detect them and wire them into the correct ComfyUI nodes.

## Tips

- Be descriptive — more detail produces better results
- Mention aspect ratio, lighting mood, or style references when relevant
- Follow up naturally: _"Make it warmer"_, _"Try a higher contrast version"_

---

*Configuration is read from `config/settings.json` and `.env` in the project root.*
