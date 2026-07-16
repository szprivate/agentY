---
name: custom-node-from-github
description: Turn a model's GitHub repository (already cloned locally) into a self-contained, importable ComfyUI custom-node pack — __init__.py, nodes.py, requirements.txt, README.md, pyproject.toml. Use when handed a repo_dir to read and an empty pack_dir to fill.
allowed-tools: read_text_file, file_read, write_text_file, run_script, web_search
---

# Custom node from a GitHub repo

Turn a **model's source repository** into a **self-contained ComfyUI custom-node
pack**. You are given:

- `repo_url` — the GitHub URL the pack is built from.
- `repo_dir` — a local clone of that repo you can read (already on disk).
- `pack_dir` — the **empty output folder** you must fill. Everything the node
  needs lives here; the folder is meant to become its own GitHub repo later, so
  it must be complete and stand alone.
- `node_name` and optional `notes`.

Your job: **read the repo, understand how the model is loaded and run, and write
a working ComfyUI custom node that exposes it.** You do not execute the model —
you author the code.

## Workflow

1. **Read the repo.** Start with `README*`, `docs/`, `examples/`, then the actual
   inference entry points (`inference.py`, `pipeline*.py`, `app.py`,
   `demo/`, `*.py` under the package). Use `read_text_file` and `run_script`
   (e.g. list files with a short `os.walk`). You must find, concretely: the
   **load** call (weights path, config, dtype/device) and the **inference** call
   (its inputs, their shapes/dtypes, and its outputs). If the README is thin,
   `web_search` the model name for a usage snippet — but prefer the repo's code.
2. **Design the node(s).** Map the model's inputs/outputs onto ComfyUI types
   (table below). Split into a **Loader node** (loads weights, outputs a MODEL
   handle) and a **Run node** (takes that handle + inputs, produces IMAGE/etc.)
   when loading is expensive or reused; use a single node when it's trivial.
3. **Write the pack** (all files below) with `write_text_file`, using paths under
   `pack_dir`.
4. **Return a summary** (final message): the pack path, the files you wrote, the
   node keys/display names, and a bullet list of every **Unresolved / TODO** item
   you could not determine from the docs. Be honest about gaps.

## ComfyUI node anatomy (get this exactly right)

A node is a Python class. ComfyUI discovers it through two module-level dicts in
`__init__.py`:

```python
NODE_CLASS_MAPPINGS = {"MyModelLoader": MyModelLoader, "MyModelSampler": MyModelSampler}
NODE_DISPLAY_NAME_MAPPINGS = {"MyModelLoader": "MyModel Loader", "MyModelSampler": "MyModel Sampler"}
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
```

Each node class:

```python
class MyModelSampler:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MYMODEL",),                       # handle from the loader node
                "image": ("IMAGE",),                          # ComfyUI image tensor
                "prompt": ("STRING", {"multiline": True, "default": ""}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 200}),
                "cfg": ("FLOAT", {"default": 7.0, "min": 0.0, "max": 30.0, "step": 0.1}),
                "mode": (["fast", "quality"], {"default": "fast"}),  # dropdown
            },
            "optional": {"mask": ("MASK",)},
        }
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)     # optional but nice
    FUNCTION = "run"
    CATEGORY = "MyModel"
    # OUTPUT_NODE = True          # only for terminal/save/preview nodes

    def run(self, model, image, prompt, steps, cfg, mode, mask=None):
        ...
        return (out_image,)       # ALWAYS a tuple matching RETURN_TYPES
```

Rules that trip people up:
- `INPUT_TYPES` is a **classmethod** returning the `required`/`optional`/`hidden`
  dict. Widget options: `default, min, max, step, multiline, tooltip`.
- The `FUNCTION` method **returns a tuple** (even for one output: `(x,)`).
- A dropdown is a list of choices as the type: `(["a","b"], {"default":"a"})`.
- Set `OUTPUT_NODE = True` only for nodes that terminate a graph (save/preview);
  they may return `{"ui": {...}}` instead of / alongside the tuple.

## ComfyUI data types at the node boundary

- **IMAGE**: `torch.Tensor`, shape `[B, H, W, C]`, `float32` in `[0, 1]`, RGB.
  - to PIL: `Image.fromarray((t[0].cpu().numpy() * 255).astype(np.uint8))`
  - from PIL: `torch.from_numpy(np.array(pil).astype(np.float32) / 255.0)[None,]`
  - Batch several outputs with `torch.cat(list_of_BHWC, dim=0)`.
- **MASK**: `torch.Tensor`, shape `[B, H, W]`, float `[0, 1]`.
- **LATENT**: a dict `{"samples": tensor}`.
- **STRING / INT / FLOAT / BOOLEAN**: plain Python scalars.
- **MODEL / CLIP / VAE / CONDITIONING**: opaque handles — for a brand-new model,
  invent your own handle type string (e.g. `"MYMODEL"`) and pass your own object
  (a small dataclass/dict holding the loaded model + metadata) between your nodes.

## Weights, device, and safe importing

- Resolve model files through `folder_paths`, never hard-coded paths:
  ```python
  import folder_paths, os
  base = os.path.join(folder_paths.models_dir, "mymodel")   # ComfyUI/models/mymodel
  os.makedirs(base, exist_ok=True)
  # dropdown of available files:  folder_paths.get_filename_list("checkpoints")
  ```
  Expose available weight files as a dropdown in `INPUT_TYPES` when you can.
- Device / VRAM: `import comfy.model_management as mm; device = mm.get_torch_device()`;
  free with `mm.soft_empty_cache()`. Load to `device`, offload big models when done.
- **Keep `__init__.py` import cheap and side-effect-free.** Do NOT import the
  model's heavy libraries (torch pipelines, diffusers, the repo's own package) or
  download anything at module import — ComfyUI imports every pack at startup and a
  failing/slow import breaks the whole server. Do heavy `import`s and any download
  **inside the node's FUNCTION**, and wrap optional deps in a clear
  `try/except ImportError` that tells the user what to `pip install`.

## Files you MUST write into `pack_dir`

1. `__init__.py` — the two mapping dicts (+ `WEB_DIRECTORY` only if you ship JS).
2. `nodes.py` — the node class(es) and all implementation. (Split into more
   modules if large; `__init__.py` imports from them.)
3. `requirements.txt` — the model's runtime deps (copy from the repo; pin loosely).
4. `README.md` — what the node does, the node names, **install** (clone into
   `ComfyUI/custom_nodes/`, `pip install -r requirements.txt`, where weights go),
   inputs/outputs, an example, and an **"Unresolved / TODO"** section listing
   anything you stubbed.
5. `pyproject.toml` — Comfy Registry metadata:
   ```toml
   [project]
   name = "<slug>"
   version = "0.1.0"
   description = "<one line>"
   dependencies = [ ... ]           # mirror requirements.txt
   [tool.comfy]
   PublisherId = ""                 # user fills in before publishing
   DisplayName = "<Node Pack Name>"
   ```

## Honesty rule

Implement the documented behaviour faithfully. Where a detail genuinely cannot be
determined from the repo (an undocumented arg, an unclear output shape), insert a
clearly marked `# TODO(custom-node-from-github): <what's missing and where to look>`
stub with a reasonable placeholder — **never invent an API you did not see.** List
every such stub in the README's "Unresolved / TODO" section so the user knows
exactly what to finish. A runnable node with two honest TODOs beats a
confident-looking node built on guesses.

Do not ask clarifying questions — make reasonable, documented assumptions and note
them. When every file is written, stop and return your summary.
