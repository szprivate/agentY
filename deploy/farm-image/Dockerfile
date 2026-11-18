# The machine image for agentY's farm (the viewer's Farm space): ComfyUI with CUDA 13 (RTX 50xx, PRO 6000, H100)
# from yanwk/comfyui-boot, plus the agentY farm helper node, which downloads the models a render needs and
# checks the per-machine token. Built and published by .github/workflows/farm-image.yml as
# ghcr.io/szprivate/agenty-farm:latest. No models are baked in: the farm downloads them on first use.
FROM yanwk/comfyui-boot:cu130-slim-v2

# The entrypoint copies this bundle to /root/ComfyUI on a fresh machine, custom nodes included.
COPY agenty_farm_helper /default-comfyui-bundle/ComfyUI/custom_nodes/agenty_farm_helper

# Previews for the Farm tab's live view.
ENV CLI_ARGS="--preview-method auto"
