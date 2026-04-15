```skill
---
name: model-reference
description: Pre-validated model reference table. Activate during step 8 (Resolve parameters) to look up model shortnames and paths without calling list_models. Also includes the resolution procedure for unknown models.
allowed-tools: get_model_types, get_models_in_folder
---

# Model Reference — Known Pre-Validated Models

Model paths are relative to the external model directory configured on the ComfyUI server.

> **Keep in sync**: this table mirrors `config/models.json`. If models are added to that file, update this skill too.

---

## Resolution procedure

1. **Check this table** → if the shortname is listed, use the path directly. No tool call needed.
2. **Not in table** → call `get_models_in_folder` or `get_model_types` to verify the path on the server.
3. **Still not found** → do NOT fabricate a path. Note the model as unverified in the brainbriefing. Flag as BLOCKER if no fallback exists.

---

## UNETs

| shortname          | path                                                        |
|--------------------|-------------------------------------------------------------|
| flux1-dev-fp8      | FLUX1/flux1-dev-fp8.safetensors                             |
| flux1-dev          | FLUX1/flux1-dev.safetensors                                 |
| flux1-schnell      | FLUX1/flux1-schnell.safetensors                             |
| flux1-fill         | FLUX1/flux1-fill-dev.safetensors                            |
| flux1-canny        | FLUX1/flux1-canny-dev.safetensors                           |
| flux1-depth        | FLUX1/flux1-depth-dev.safetensors                           |
| flux1-kontext      | FLUX1/flux1-dev-kontext_fp8_scaled.safetensors              |
| flux2-klein        | FLUX2/flux-2-klein-9b.safetensors                           |
| flux2-dev          | FLUX2/flux2-dev.safetensors                                 |
| qwen-edit          | QWEN/qwen_image_edit_2511_fp8_e4m3fn.safetensors            |
| iclight-fc         | ICLight/iclight_sd15_fc.safetensors                         |
| iclight-fbc        | ICLight/iclight_sd15_fbc.safetensors                        |
| wan21-t2v          | WAN21/Wan2_1-T2V-14B_fp8_e4m3fn.safetensors                 |
| wan21-i2v-480p     | WAN21/Wan2_1-I2V-14B-480P_fp8_e4m3fn.safetensors            |
| wan21-i2v-720p     | WAN21/Wan2_1-I2V-14B-720P_fp8_e4m3fn.safetensors            |
| wan22-t2v-high     | WAN22/Wan2.2-T2V-A14B_high.safetensors                      |
| wan22-t2v-low      | WAN22/Wan2.2-T2V-A14B_low.safetensors                       |
| wan22-i2v-high     | WAN22/wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors      |
| wan22-i2v-low      | WAN22/wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors       |

---

## Checkpoints

| shortname       | path                                                     |
|-----------------|----------------------------------------------------------|
| cyberrealistic  | SD15/cyberrealistic_v80.safetensors                      |
| juggernaut      | SD15/juggernaut_reborn.safetensors                       |
| photon          | SD15/photon_v1.safetensors                               |
| sdxl-base       | SDXL/sd_xl_base_1.0.safetensors                          |
| epicrealism-xl  | SDXL/epicrealismXL_vxviLastfameDMD2.safetensors          |

---

## VAE

| shortname    | path                                                        |
|--------------|-------------------------------------------------------------|
| flux-vae     | FLUX1/ae.safetensors                                        |
| sd15-vae     | SD15/vae-ft-mse-840000-ema-pruned.safetensors               |
| sdxl-vae     | SDXL/sdxl_vae.safetensors                                   |
| wan21-vae    | WAN21/Wan2_1_VAE_bf16.safetensors                           |
| wan22-vae    | WAN22/wan2.2_vae.safetensors                                |

---

## CLIP

| shortname    | path                                                                      |
|--------------|---------------------------------------------------------------------------|
| flux-t5      | Flux-Dev/t5xxl_fp16.safetensors                                           |
| flux-clip_l  | Flux-Dev/clip_l.safetensors                                               |
| wan-clip     | WAN/open-clip-xlm-roberta-large-vit-huge-14_fp16.safetensors              |

---

## ControlNets

| shortname               | path                                                                |
|-------------------------|---------------------------------------------------------------------|
| flux-union-pro          | Flux-Dev/Flux.1-dev-ControlNet-Union-Pro.safetensors                |
| flux-inpainting-beta    | Flux-Dev/FLUX.1-dev-Controlnet-Inpainting-Beta.safetensors          |

---

## LoRAs

| shortname                    | path                                                                          |
|------------------------------|-------------------------------------------------------------------------------|
| flux-canny-lora              | MISC/flux1-canny-dev-lora.safetensors                                         |
| flux-depth-lora              | MISC/flux1-depth-dev-lora.safetensors                                         |
| wan21-causvid                | WAN21/Wan21_CausVid_14B_T2V_lora_rank32_v2.safetensors                        |
| wan21-lightx2v               | WAN21/Wan21_T2V_14B_lightx2v_cfg_step_distill_lora_rank32.safetensors         |
| wan21-orbit                  | WAN21/Wan21_360_Orbit.safetensors                                             |
| wan21-tile                   | WAN21/wan2.1-1.3b-control-lora-tile-v1.0_comfy.safetensors                    |
| wan22-relight                | WAN22/WanAnimate_relight_lora_fp16.safetensors                                |
| wan22-lightx2v-256           | WAN22/lightx2v_T2V_14B_cfg_step_distill_v2_lora_rank256_bf16.safetensors      |
| wan22-lightx2v-64            | WAN22/lightx2v_T2V_14B_cfg_step_distill_v2_lora_rank64_bf16.safetensors       |
| wan22-i2v-lightx2v-high      | WAN22/wan2.2_i2v_lightx2v_4steps_lora_v1_high_noise.safetensors               |
| wan22-i2v-lightx2v-low       | WAN22/wan2.2_i2v_lightx2v_4steps_lora_v1_low_noise.safetensors                |
```
