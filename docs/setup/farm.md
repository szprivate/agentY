# Farm: rented GPUs for ComfyUI renders

The viewer's Farm space renders a workflow on GPUs you rent from vast.ai or RunPod, or on any ComfyUI you can reach
by address. agentY does the work; the viewer only shows it.

## Setup

1. **Key.** Settings > Add API key > *vast.ai (Farm GPUs)* or *RunPod (Farm GPUs)*. The server URL is already filled
   in. Saved keys work at once.
2. **Machine image.** Farm > Setup > Machine image defaults to `ghcr.io/szprivate/agenty-farm:latest`, built from
   `deploy/farm-image` by the *Farm image* GitHub workflow on every change there. GitHub publishes a new package as
   private: set it to public once (GitHub > your profile > Packages > agenty-farm > Package settings > Change
   visibility), or the providers cannot pull it.
3. **Limits.** Daily cap, most per hour, idle stop. They are enforced while the agentY host runs; a machine you
   leave running with agentY closed keeps billing.

## What happens on a render

- **Inputs**: every file a `Load…` node names is uploaded to the machine (once per machine, again when it changes).
- **Models**: the machine starts with none. Before a render the farm lists the model files the workflow needs that
  the machine lacks and asks it to download them (the image's `agenty_farm_helper` node). The task shows
  *getting models* with progress. A machine keeps what it downloaded for its next renders.
- **Outputs**: saved images, videos, audio and 3D files are downloaded to `output_images/farm/<job>/`.

### Where a model is downloaded from

In this order:

1. `config/farm_models.json` (yours):

   ```json
   {"models": {"loras/my_style.safetensors": "https://huggingface.co/me/loras/resolve/main/my_style.safetensors"},
    "repos": ["me/my-models"]}
   ```

   `models` maps `folder/name` to an address; `repos` are extra Hugging Face repos searched by file name.
2. The addresses the workflow carries: ComfyUI's templates list one per loader node.
3. The Comfy-Org repos on Hugging Face (`Comfy-Org/MOSS-Image`, `Comfy-Org/Z-Image-Turbo`, ...), whose folders match
   ComfyUI's model folders. Their file lists are read once a day.

A model none of these knows fails the render with its name, so you can add it to `config/farm_models.json`.
Gated Hugging Face repos need `HF_TOKEN` on the machine.

## The machine's token

Every rented machine gets its own token, and the farm image answers only requests that carry it. *Open ComfyUI* on
a machine card opens it with the token. A machine added by address uses the token you give it, if any.

## Not covered yet

Custom nodes the image does not have are not installed; *Check workflow* on a machine card lists what is missing.
