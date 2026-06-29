# Workflow types - recipe database

- Types: 35 (similarity threshold 0.3)
- Self-contained: every type has a description + user_intent; no human annotation step required.
- Media distribution: 3d (4), audio (1), image (21), text (2), video (7)
- Sorted by member_count descending.

## `text_to_image_z_image`  -  "Canny to Image (Z-Image-Turbo)"  -  15 member(s)  -  source: mixed
- user intent: media=image | task=text_to_image | model families: Z-Image, Anima, Flux, Qwen Image, ACE-Step, Flux Krea, Lotus
- when to use: Use to generate an image from a text prompt using Z-Image, Anima, Flux, Qwen Image, ACE-Step, Flux Krea, Lotus.
- example requests: "build an image workflow using Z-Image"; "build an image workflow using Anima"; "build an image workflow using Flux"; "build an image workflow using Qwen Image"; "build an image workflow using ACE-Step"; "build an image workflow using Flux Krea"; "build an image workflow using Lotus"; "generate an image from a text prompt using Z-Image"
- description (catalog): FLUX.1 Krea [dev] (Black Forest Labs × Krea): open-weight 12B rectified-flow text-to-image drop-in alongside FLUX.1 [dev], tuned away from overcooked saturation toward more natural diversity in people, realism, and style while keeping ecosystem compatibility. | Generates an image from a Canny edge map using Z-Image-Turbo, with text conditioning. | Generates an image from a depth map using Z-Image-Turbo with text conditioning. | Generates an image from pose keypoints using Z-Image-Turbo with text conditioning. | Generates audio/music from text prompts using ACE-Step 1.5, a diffusion-based audio generation model. | Generates images from a text prompt and ControlNet conditioning (e.g. depth, canny) using Z-Image-Turbo. | Generates images from prompts using FLUX.1 [dev]: a 12B rectified-flow MMDiT with dual CLIP plus T5-XXL text encoders and guidance-distilled sampling for sharp prompt following versus classic DDPM diffusion. | Generates images from text prompts using Z-Image base weights with Qwen3 text encoder and bundled VAE. | Generates images from text prompts using Z-Image-Turbo defaults with Qwen3 text encoder and VAE. | Generates images from text prompts using Z-Image-Turbo, Alibaba's distilled 6B DiT model. | Inpaints masked image regions using Flux.1 fill [dev], Black Forest Labs' inpainting/outpainting model. | This subgraph converts text prompts into non-photorealistic illustrations using a 2-billion-parameter model optimized for anime and artistic styles. It is ideal for generating concept art, character designs, or stylized illustrations where photorealism is not required. The model excels with anime and artistic content but performs poorly on realistic subjects. | This subgraph generates non-photorealistic illustrations from text prompts using a 2-billion-parameter model optimized for anime concepts, characters, and styles. It is ideal for creating artistic images, concept art, or stylized illustrations where photorealism is not required. The model excels with anime and artistic content but performs poorly on realistic subjects. | [Local] image editing via Z-Image-Turbo. 1 image input -> 1 image output. Uses ControlNet for precise and controlled image editing. | [Local] text-to-image via Z-Image-Turbo. 1 text input -> 1 image output. High-speed image generation from text prompts.
- official category: Image generation and editing  [spans multiple catalog categories: Audio (1), Image generation and editing (12)]

- member files:
    - canny_to_image_z_image_turbo - Generates an image from a Canny edge map using Z-Image-Turbo, with text conditioning.
    - controlnet_z_image_turbo - Generates images from a text prompt and ControlNet conditioning (e.g. depth, canny) using Z-Image-Turbo.
    - depth_to_image_z_image_turbo - Generates an image from a depth map using Z-Image-Turbo with text conditioning.
    - image_inpainting_flux_1_fill_dev - Inpaints masked image regions using Flux.1 fill [dev], Black Forest Labs' inpainting/outpainting model.
    - image_z_image_turbo - [Local] text-to-image via Z-Image-Turbo. 1 text input -> 1 image output. High-speed image generation from text prompts.
    - image_z_image_turbo_fun_union_controlnet - [Local] image editing via Z-Image-Turbo. 1 image input -> 1 image output. Uses ControlNet for precise and controlled image editing.
    - pose_to_image_z_image_turbo - Generates an image from pose keypoints using Z-Image-Turbo with text conditioning.
    - text_to_audio_ace_step_1_5 - Generates audio/music from text prompts using ACE-Step 1.5, a diffusion-based audio generation model.
    - text_to_image - Generates images from text prompts using Z-Image-Turbo defaults with Qwen3 text encoder and VAE.
    - text_to_image_anima - This subgraph converts text prompts into non-photorealistic illustrations using a 2-billion-parameter model optimized for anime and artistic styles. It is ideal for generating concept art, character designs, or stylized illustrations where photorealism is not required. The model excels with anime and artistic content but performs poorly on realistic subjects.
    - text_to_image_anima_base_1_0 - This subgraph generates non-photorealistic illustrations from text prompts using a 2-billion-parameter model optimized for anime concepts, characters, and styles. It is ideal for creating artistic images, concept art, or stylized illustrations where photorealism is not required. The model excels with anime and artistic content but performs poorly on realistic subjects.
    - text_to_image_flux_1_dev - Generates images from prompts using FLUX.1 [dev]: a 12B rectified-flow MMDiT with dual CLIP plus T5-XXL text encoders and guidance-distilled sampling for sharp prompt following versus classic DDPM diffusion.
    - text_to_image_flux_1_krea_dev - FLUX.1 Krea [dev] (Black Forest Labs × Krea): open-weight 12B rectified-flow text-to-image drop-in alongside FLUX.1 [dev], tuned away from overcooked saturation toward more natural diversity in people, realism, and style while keeping ecosystem compatibility.
    - text_to_image_z_image_base - Generates images from text prompts using Z-Image base weights with Qwen3 text encoder and bundled VAE.
    - text_to_image_z_image_turbo - Generates images from text prompts using Z-Image-Turbo, Alibaba's distilled 6B DiT model.

- REQUIRED node roles (structural invariants):
    - UNETLoader  (diffusion model / UNET loader) - all members (15/15)
    - VAELoader  (VAE loader) - all members (15/15)
    - KSampler  (diffusion sampler / denoiser) - all members (15/15)

- OPTIONAL node roles (variant, only in some members):
    - CLIPTextEncode  (prompt text encoding) - 14/15 members
    - VAEDecode  (latent -> pixel decode) - 14/15 members
    - BasicGuider  (guider / sigma / scheduler) - 1/15 members
    - BasicScheduler  (guider / sigma / scheduler) - 1/15 members
    - CLIPLoader  (text encoder / CLIP loader) - 11/15 members
    - Canny  (unclassified node role) - 2/15 members
    - ConditioningZeroOut  (conditioning combine / edit) - 12/15 members
    - DifferentialDiffusion  (unclassified node role) - 1/15 members
    - DisableNoise  (unclassified node role) - 1/15 members
    - DualCLIPLoader  (text encoder / CLIP loader) - 4/15 members
    - EmptyAceStep1.5LatentAudio  (unclassified node role) - 1/15 members
    - EmptyLatentImage  (empty latent / canvas) - 2/15 members
    - EmptySD3LatentImage  (empty latent / canvas) - 11/15 members
    - FluxGuidance  (guider / sigma / scheduler) - 1/15 members
    - ImageInvert  (unclassified node role) - 1/15 members
    - ImageScaleToTotalPixels  (upscale / resize) - 2/15 members
    - InpaintModelConditioning  (conditioning combine / edit) - 1/15 members
    - KSamplerSelect  (diffusion sampler / denoiser) - 1/15 members
    - LoadImage  (image input / load) - 1/15 members
    - LotusConditioning  (conditioning combine / edit) - 1/15 members
    - MarkdownNote  (unclassified node role) - 1/15 members
    - ModelPatchLoader  (LoRA / model patch loader) - 5/15 members
    - ModelSamplingAuraFlow  (unclassified node role) - 10/15 members
    - PreviewImage  (save / preview / combine output) - 2/15 members
    - QwenImageDiffsynthControlnet  (controlnet / guidance conditioning) - 5/15 members
    - SamplerCustomAdvanced  (diffusion sampler / denoiser) - 1/15 members
    - SaveImage  (save / preview / combine output) - 2/15 members
    - SetFirstSigma  (unclassified node role) - 1/15 members
    - TextEncodeAceStepAudio1.5  (prompt text encoding) - 1/15 members
    - VAEDecodeAudio  (latent -> pixel decode) - 1/15 members
    - VAEEncode  (pixel -> latent encode) - 1/15 members
    - utility/plumbing (some members): GetImageSize, PrimitiveFloat, PrimitiveNode

- connection patterns (role level):
    - clip_loader -> text_encode  [CLIP]  (invariant)
    - sampler -> vae_decode  [LATENT]  (invariant)
    - vae_loader -> vae_decode  [VAE]  (invariant)
    - ModelSamplingAuraFlow -> sampler  [MODEL]  (10/15 members)
    - conditioning_op -> sampler  [CONDITIONING]  (12/15 members)
    - latent_source -> sampler  [LATENT]  (13/15 members)
    - text_encode -> conditioning_op  [CONDITIONING]  (12/15 members)
    - text_encode -> sampler  [CONDITIONING]  (14/15 members)

- boundary ports:
    - inputs:  COMBO(clip_name), COMBO(unet_name), COMBO(vae_name), INT(seed), STRING(text)
    - outputs: IMAGE(IMAGE)

- param variability: varies across members: KSampler, UNETLoader, VAELoader
- unresolved nodes (not in object_info): MarkdownNote, PrimitiveNode

## `text_to_video_wan_2_2`  -  "Character Replacement (SCAIL-2 Base)"  -  15 member(s)  -  source: mixed
- user intent: media=video | task=text_to_video | model families: WAN 2.2, WAN VACE, SAM3, Anima, SCAIL, WAN
- when to use: Use to generate a video from a text prompt using WAN 2.2, WAN VACE, SAM3, Anima, SCAIL, WAN.
- example requests: "build a video workflow using WAN 2.2"; "build a video workflow using WAN VACE"; "build a video workflow using SAM3"; "build a video workflow using Anima"; "build a video workflow using SCAIL"; "build a video workflow using WAN"; "generate a video from a text prompt using WAN 2.2"
- description (catalog+synthesized): Generates video from text prompts using Wan2.2, Alibaba's diffusion video model. | Image to Video blueprint | Image-to-video with Wan 2.2 using a start image plus text prompt to extend motion from the still frame. | Removes objects from video by inpainting masked regions using Wan 2.1 VACE, with SAM3 text-guided segmentation and optional Lightning LoRA turbo mode. | Replaces a character in a video with a reference image using the SCAIL-2 model for end-to-end controlled animation without intermediate pose maps. Key inputs include a source video, a reference character image, and optional text prompts for style or context. Suitable for animated or live-action footage, multi-character scenes, and creative video editing where direct pose-free animation is needed; works best with moderate-length videos. | Video Inpaint(Wan2.1 VACE) blueprint | [Local] image editing via Wan. 3 image inputs -> 1 image output. Performs advanced image-to-image editing and transformations.
- official category: Video generation and editing  [single category (+8 uncategorized): Video generation and editing (7)]

- member files:
    - Wan22Vace_VID2VID - [Local] image editing via Wan. 3 image inputs -> 1 image output. Performs advanced image-to-image editing and transformations.
    - character_replacement_scail_2_base - Replaces a character in a video with a reference image using the SCAIL-2 model for end-to-end controlled animation without intermediate pose maps. Key inputs include a source video, a reference character image, and optional text prompts for style or context. Suitable for animated or live-action footage, multi-character scenes, and creative video editing where direct pose-free animation is needed; works best with moderate-length videos.
    - character_replacement_scail_2_extend - Replaces a character in a video with a reference image using the SCAIL-2 model for end-to-end controlled animation without intermediate pose maps. Key inputs include a source video, a reference character image, and optional text prompts for style or context. Suitable for animated or live-action footage, multi-character scenes, and creative video editing where direct pose-free animation is needed; works best with moderate-length videos.
    - image_to_video - Image to Video blueprint
    - image_to_video_wan_2_2 - Image-to-video with Wan 2.2 using a start image plus text prompt to extend motion from the still frame.
    - text_to_video_wan_2_2 - Generates video from text prompts using Wan2.2, Alibaba's diffusion video model.
    - video_inpaint_wan2_1_vace - Video Inpaint(Wan2.1 VACE) blueprint
    - video_inpainting_wan2_1_vace - Removes objects from video by inpainting masked regions using Wan 2.1 VACE, with SAM3 text-guided segmentation and optional Lightning LoRA turbo mode.
    - video_wan2_2_14B_flf2v
    - video_wan2_2_14B_fun_camera
    - video_wan2_2_14B_fun_control
    - video_wan_vace_14B_ref2v
    - video_wan_vace_14B_v2v
    - video_wan_vace_flf2v
    - video_wan_vace_outpainting

- REQUIRED node roles (structural invariants):
    - CLIPTextEncode  (prompt text encoding) - all members (15/15), 2 required instances  **[PAIRED: 2x required]**
        - instance feeds ['WanVaceToVideo'] / fed by ['clip_loader']  (x7)
        - instance feeds ['WanVaceToVideo'] / fed by ['lora_loader']  (x7)
        - instance feeds ['SAM3_VideoTrack'] / fed by ['model_loader']  (x4)
        - instance feeds ['WanImageToVideo'] / fed by ['clip_loader']  (x4)
        - instance feeds ['WanSCAILToVideo'] / fed by ['clip_loader']  (x4)
        - instance feeds ['Wan22FunControlToVideo'] / fed by ['clip_loader']  (x2)
        - instance feeds ['WanCameraImageToVideo'] / fed by ['clip_loader']  (x2)
        - instance feeds ['WanFirstLastFrameToVideo'] / fed by ['clip_loader']  (x2)
        - instance feeds ['sampler', 'sampler'] / fed by ['clip_loader']  (x2)
        - instance feeds ['SAM3_Detect'] / fed by ['model_loader']  (x1)
    - ModelSamplingSD3  (unclassified node role) - all members (15/15)
    - CLIPLoader  (text encoder / CLIP loader) - all members (15/15)
    - VAEDecode  (latent -> pixel decode) - all members (15/15)
    - VAELoader  (VAE loader) - all members (15/15)

- OPTIONAL node roles (variant, only in some members):
    - ImageBatch  (unclassified node role) - 1/15 members
    - ImageFromBatch  (unclassified node role) - 4/15 members
    - MarkdownNote  (unclassified node role) - 3/15 members
    - DiffusionModelLoaderKJ  (diffusion model / UNET loader) - 1/15 members
    - DiffusionModelSelector  (diffusion model / UNET loader) - 1/15 members
    - GetVideoComponents  (unclassified node role) - 12/15 members
    - KSamplerAdvanced  (diffusion sampler / denoiser) - 7/15 members
    - LoadImage  (image input / load) - 7/15 members
    - LoraLoaderModelOnly  (LoRA / model patch loader) - 8/15 members
    - MaskToImage  (unclassified node role) - 4/15 members
    - Note  (unclassified node role) - 2/15 members
    - PreviewImage  (save / preview / combine output) - 6/15 members
    - SAM3_VideoTrack  (unclassified node role) - 2/15 members
    - SolidMask  (unclassified node role) - 1/15 members
    - UNETLoader  (diffusion model / UNET loader) - 14/15 members
    - BasicScheduler  (guider / sigma / scheduler) - 2/15 members
    - BatchImagesNode  (unclassified node role) - 1/15 members
    - CLIPVisionEncode  (unclassified node role) - 2/15 members
    - CLIPVisionLoader  (unclassified node role) - 2/15 members
    - CheckpointLoaderSimple  (diffusion model / UNET loader) - 3/15 members
    - ColorTransfer  (unclassified node role) - 1/15 members
    - CreateVideo  (unclassified node role) - 13/15 members
    - EmptyHunyuanLatentVideo  (unclassified node role) - 1/15 members
    - GrowMask  (unclassified node role) - 1/15 members
    - ImageCompositeMasked  (unclassified node role) - 2/15 members
    - ImagePadForOutpaint  (unclassified node role) - 1/15 members
    - ImageStitch  (unclassified node role) - 1/15 members
    - ImageToMask  (unclassified node role) - 3/15 members
    - Int  (unclassified node role) - 1/15 members
    - InvertMask  (unclassified node role) - 2/15 members
    - KSampler  (diffusion sampler / denoiser) - 6/15 members
    - KSamplerSelect  (diffusion sampler / denoiser) - 2/15 members
    - LoadVideo  (video input / load) - 4/15 members
    - LoraLoader  (LoRA / model patch loader) - 4/15 members
    - MaskPreview  (unclassified node role) - 1/15 members
    - RebatchImages  (unclassified node role) - 1/15 members
    - RepeatImageBatch  (unclassified node role) - 3/15 members
    - ResizeImageMaskNode  (unclassified node role) - 3/15 members
    - SAM3_Detect  (unclassified node role) - 1/15 members
    - SCAIL2ColoredMask  (unclassified node role) - 2/15 members
    - SamplerCustom  (diffusion sampler / denoiser) - 2/15 members
    - SaveVideo  (save / preview / combine output) - 1/15 members
    - TrimVideoLatent  (unclassified node role) - 7/15 members
    - VHS_VideoCombine  (save / preview / combine output) - 7/15 members
    - Wan22FunControlToVideo  (unclassified node role) - 1/15 members
    - WanCameraEmbedding  (unclassified node role) - 1/15 members
    - WanCameraImageToVideo  (unclassified node role) - 1/15 members
    - WanFirstLastFrameToVideo  (unclassified node role) - 1/15 members
    - WanImageToVideo  (unclassified node role) - 2/15 members
    - WanSCAILToVideo  (unclassified node role) - 2/15 members
    - WanVaceToVideo  (unclassified node role) - 7/15 members
    - bEpicReformat  (unclassified node role) - 1/15 members
    - bepicVaceKeyframeReplacer  (unclassified node role) - 1/15 members
    - utility/plumbing (some members): PrimitiveInt, ComfyMathExpression, ComfySwitchNode, PrimitiveBoolean, PrimitiveFloat, GetImageSize, easy mathInt

- connection patterns (role level):
    - ModelSamplingSD3 -> sampler  [MODEL]  (invariant)
    - vae_loader -> vae_decode  [VAE]  (invariant)
    - clip_loader -> text_encode  [CLIP]  (12/15 members)
    - lora_loader -> ModelSamplingSD3  [MODEL]  (9/15 members)
    - model_loader -> lora_loader  [MODEL]  (12/15 members)
    - sampler -> vae_decode  [LATENT]  (8/15 members)
    - vae_decode -> CreateVideo  [IMAGE]  (13/15 members)

- boundary ports:
    - inputs:  (none)
    - outputs: (none)

- param variability: varies across members: CLIPLoader, ModelSamplingSD3, VAEDecode, VAELoader
- unresolved nodes (not in object_info): MarkdownNote, Note, SCAIL2ColoredMask
- custom nodes: DiffusionModelLoaderKJ, DiffusionModelSelector, Int, VHS_VideoCombine, bEpicReformat, bepicVaceKeyframeReplacer, easy mathInt

## `image_generation`  -  "Brightness and Contrast"  -  12 member(s)  -  source: official
- user intent: media=image | task=image_generation | model families: n/a
- when to use: Use to generate an image.
- example requests: "build an image workflow"; "generate an image"
- description (catalog): Adds a glow/bloom effect around bright image areas via GPU fragment shader. | Adds lens-style chromatic aberration (color fringing) using a real-time GPU fragment shader. | Adds procedural film grain texture for a cinematic look via GPU fragment shader. | Adjusts black point, white point, and gamma for tonal range control via GPU shader. | Adjusts hue, saturation, and lightness of an image using a real-time GPU fragment shader. | Adjusts image brightness and contrast using a real-time GPU fragment shader. | Adjusts saturation, temperature, tint, and vibrance using a real-time GPU fragment shader. | Applies Gaussian, Box, or Radial blur to soften images and create stylized depth or motion effects. | Applies bilateral (edge-preserving) blur to soften images while retaining detail. | Balances colors across shadows, midtones, and highlights using a real-time GPU fragment shader. | Enhances edge contrast via unsharp masking for a sharper image appearance. | Sharpens image details using a GPU fragment shader for enhanced clarity.
- official category: Image Tools  [pure: Image Tools (12)]

- member files:
    - brightness_and_contrast - Adjusts image brightness and contrast using a real-time GPU fragment shader.
    - chromatic_aberration - Adds lens-style chromatic aberration (color fringing) using a real-time GPU fragment shader.
    - color_adjustment - Adjusts saturation, temperature, tint, and vibrance using a real-time GPU fragment shader.
    - color_balance - Balances colors across shadows, midtones, and highlights using a real-time GPU fragment shader.
    - edge_preserving_blur - Applies bilateral (edge-preserving) blur to soften images while retaining detail.
    - film_grain - Adds procedural film grain texture for a cinematic look via GPU fragment shader.
    - glow - Adds a glow/bloom effect around bright image areas via GPU fragment shader.
    - hue_and_saturation - Adjusts hue, saturation, and lightness of an image using a real-time GPU fragment shader.
    - image_blur - Applies Gaussian, Box, or Radial blur to soften images and create stylized depth or motion effects.
    - image_levels - Adjusts black point, white point, and gamma for tonal range control via GPU shader.
    - sharpen - Sharpens image details using a GPU fragment shader for enhanced clarity.
    - unsharp_mask - Enhances edge contrast via unsharp masking for a sharper image appearance.

- REQUIRED node roles (structural invariants):
    - GLSLShader  (unclassified node role) - all members (12/12)
    - utility/plumbing (always present): PrimitiveFloat(9x)

- OPTIONAL node roles (variant, only in some members):
    - CustomCombo  (unclassified node role) - 6/12 members
    - ColorToRGBInt  (unclassified node role) - 1/12 members
    - utility/plumbing (some members): PrimitiveBoolean, PrimitiveInt

- connection patterns (role level):
    - PrimitiveFloat -> GLSLShader  [FLOAT]  (invariant)
    - CustomCombo -> GLSLShader  [INT]  (6/12 members)

- boundary ports:
    - inputs:  IMAGE(images.image0)
    - outputs: IMAGE(IMAGE0)

- param variability: varies across members: GLSLShader, PrimitiveFloat

## `first_last_frame_to_video_ltx_2`  -  "Canny to Video (LTX 2.0)"  -  9 member(s)  -  source: mixed
- user intent: media=video | task=first_last_frame_to_video | model families: LTX-2, Lotus
- when to use: Use to generate a video interpolating between a first and last frame using LTX-2, Lotus.
- example requests: "build a video workflow using LTX-2"; "build a video workflow using Lotus"; "generate a video interpolating between a first and last frame using LTX-2"
- description (catalog+synthesized): Generates a video interpolating between first and last keyframes using LTX-2.3. | Generates a video that interpolates between the first and last keyframes using LTX-2.3, including optional audio. | Generates depth-controlled video with LTX-2: motion and structure follow a depth-reference video alongside text prompting, optional first-frame image conditioning, with optional synchronized audio. | Generates video from Canny edge maps using LTX-2, with optional synchronized audio. | Generates video from a single input image using LTX-2.3. | Generates video from pose reference frames using LTX-2, with optional synchronized audio. | Generates video from text prompts using LTX-2.3, Lightricks' video diffusion model.
- official category: Video generation and editing  [single category (+2 uncategorized): Video generation and editing (7)]

- member files:
    - canny_to_video_ltx_2_0 - Generates video from Canny edge maps using LTX-2, with optional synchronized audio.
    - depth_to_video_ltx_2_0 - Generates depth-controlled video with LTX-2: motion and structure follow a depth-reference video alongside text prompting, optional first-frame image conditioning, with optional synchronized audio.
    - first_last_frame_to_video - Generates a video that interpolates between the first and last keyframes using LTX-2.3, including optional audio.
    - first_last_frame_to_video_ltx_2_3 - Generates a video interpolating between first and last keyframes using LTX-2.3.
    - image_to_video_ltx_2_3 - Generates video from a single input image using LTX-2.3.
    - pose_to_video_ltx_2_0 - Generates video from pose reference frames using LTX-2, with optional synchronized audio.
    - text_to_video_ltx_2_3 - Generates video from text prompts using LTX-2.3, Lightricks' video diffusion model.
    - video_ltx2_3_flf2v
    - video_ltx2_3_i2v

- REQUIRED node roles (structural invariants):
    - CLIPTextEncode  (prompt text encoding) - all members (9/9), 2 required instances  **[PAIRED: 2x required]**
        - instance feeds ['conditioning_op'] / fed by ['text_encode']  (x14)
        - instance feeds ['conditioning_op'] / fed by ['PrimitiveStringMultiline', 'text_encode']  (x2)
        - instance feeds ['conditioning_op'] / fed by ['TextGenerateLTX2Prompt', 'text_encode']  (x1)
        - instance feeds ['conditioning_op'] / fed by ['lora_loader']  (x1)
    - SamplerCustomAdvanced  (diffusion sampler / denoiser) - all members (9/9)
    - CFGGuider  (guider / sigma / scheduler) - all members (9/9)
    - LTXVConcatAVLatent  (unclassified node role) - all members (9/9)
    - LTXVSeparateAVLatent  (unclassified node role) - all members (9/9)
    - ManualSigmas  (guider / sigma / scheduler) - all members (9/9)
    - RandomNoise  (unclassified node role) - all members (9/9)
    - CheckpointLoaderSimple  (diffusion model / UNET loader) - all members (9/9)
    - CreateVideo  (unclassified node role) - all members (9/9)
    - EmptyLTXVLatentVideo  (unclassified node role) - all members (9/9)
    - LTXAVTextEncoderLoader  (prompt text encoding) - all members (9/9)
    - LTXVAudioVAEDecode  (latent -> pixel decode) - all members (9/9)
    - LTXVAudioVAELoader  (VAE loader) - all members (9/9)
    - LTXVConditioning  (conditioning combine / edit) - all members (9/9)
    - LTXVCropGuides  (unclassified node role) - all members (9/9)
    - LTXVEmptyLatentAudio  (empty latent / canvas) - all members (9/9)
    - VAEDecodeTiled  (latent -> pixel decode) - all members (9/9)
    - utility/plumbing (always present): PrimitiveInt(4x)

- OPTIONAL node roles (variant, only in some members):
    - KSamplerSelect  (diffusion sampler / denoiser) - 6/9 members
    - LTXVAddGuide  (unclassified node role) - 6/9 members
    - LTXVImgToVideoInplace  (unclassified node role) - 6/9 members
    - LTXVPreprocess  (unclassified node role) - 6/9 members
    - LoadImage  (image input / load) - 3/9 members
    - LoraLoaderModelOnly  (LoRA / model patch loader) - 6/9 members
    - ResizeImageMaskNode  (unclassified node role) - 7/9 members
    - VAEDecode  (latent -> pixel decode) - 3/9 members
    - BasicGuider  (guider / sigma / scheduler) - 1/9 members
    - BasicScheduler  (guider / sigma / scheduler) - 1/9 members
    - DisableNoise  (unclassified node role) - 1/9 members
    - GetVideoComponents  (unclassified node role) - 3/9 members
    - ImageFromBatch  (unclassified node role) - 1/9 members
    - ImageInvert  (unclassified node role) - 1/9 members
    - ImageScaleBy  (upscale / resize) - 2/9 members
    - LTXVLatentUpsampler  (diffusion sampler / denoiser) - 6/9 members
    - LTXVScheduler  (unclassified node role) - 3/9 members
    - LatentUpscaleModelLoader  (diffusion model / UNET loader) - 6/9 members
    - LoraLoader  (LoRA / model patch loader) - 1/9 members
    - LotusConditioning  (conditioning combine / edit) - 1/9 members
    - MarkdownNote  (unclassified node role) - 3/9 members
    - ResizeImagesByLongerEdge  (unclassified node role) - 3/9 members
    - SamplerEulerAncestral  (diffusion sampler / denoiser) - 3/9 members
    - SetFirstSigma  (unclassified node role) - 1/9 members
    - TextGenerateLTX2Prompt  (unclassified node role) - 1/9 members
    - UNETLoader  (diffusion model / UNET loader) - 1/9 members
    - VAEEncode  (pixel -> latent encode) - 1/9 members
    - VAELoader  (VAE loader) - 1/9 members
    - VHS_VideoCombine  (save / preview / combine output) - 2/9 members
    - utility/plumbing (some members): ComfyMathExpression, GetImageSize, PreviewAny, PrimitiveBoolean, PrimitiveFloat, PrimitiveStringMultiline, Reroute

- connection patterns (role level):
    - LTXVConcatAVLatent -> sampler  [LATENT]  (invariant)
    - LTXVSeparateAVLatent -> LTXVCropGuides  [LATENT]  (invariant)
    - LTXVSeparateAVLatent -> vae_decode  [LATENT]  (invariant)
    - RandomNoise -> sampler  [NOISE]  (invariant)
    - guidance -> sampler  [GUIDER]  (invariant)
    - guidance -> sampler  [SIGMAS]  (invariant)
    - latent_source -> LTXVConcatAVLatent  [LATENT]  (invariant)
    - sampler -> LTXVSeparateAVLatent  [LATENT]  (invariant)
    - sampler -> sampler  [SAMPLER]  (invariant)
    - text_encode -> conditioning_op  [CONDITIONING]  (invariant)
    - text_encode -> text_encode  [CLIP]  (invariant)
    - vae_decode -> CreateVideo  [AUDIO]  (invariant)
    - vae_decode -> CreateVideo  [IMAGE]  (invariant)
    - vae_loader -> latent_source  [VAE]  (invariant)
    - vae_loader -> vae_decode  [VAE]  (invariant)
    - ComfyMathExpression -> CreateVideo  [FLOAT]  (6/9 members)
    - ComfyMathExpression -> EmptyLTXVLatentVideo  [INT]  (5/9 members)
    - ComfyMathExpression -> conditioning_op  [FLOAT]  (6/9 members)
    - ComfyMathExpression -> latent_source  [INT]  (5/9 members)
    - EmptyLTXVLatentVideo -> LTXVImgToVideoInplace  [LATENT]  (6/9 members)
    - GetImageSize -> EmptyLTXVLatentVideo  [INT]  (6/9 members)
    - LTXVAddGuide -> LTXVConcatAVLatent  [LATENT]  (6/9 members)
    - LTXVAddGuide -> LTXVCropGuides  [CONDITIONING]  (6/9 members)
    - LTXVAddGuide -> guidance  [CONDITIONING]  (6/9 members)
    - LTXVCropGuides -> guidance  [CONDITIONING]  (6/9 members)
    - LTXVImgToVideoInplace -> LTXVConcatAVLatent  [LATENT]  (6/9 members)
    - LTXVSeparateAVLatent -> LTXVConcatAVLatent  [LATENT]  (6/9 members)
    - PrimitiveInt -> ComfyMathExpression  [INT]  (6/9 members)
    - PrimitiveInt -> ResizeImageMaskNode  [INT]  (6/9 members)
    - PrimitiveInt -> latent_source  [INT]  (7/9 members)
    - ... and 12 more

- boundary ports:
    - inputs:  COMBO(ckpt_name), COMBO(lora_name), COMBO(model_name), COMBO(text_encoder), STRING(text)
    - outputs: VIDEO(VIDEO)

- param variability: varies across members: CFGGuider, CheckpointLoaderSimple, CreateVideo, EmptyLTXVLatentVideo, LTXAVTextEncoderLoader, LTXVAudioVAEDecode, LTXVAudioVAELoader, LTXVConcatAVLatent, LTXVConditioning, LTXVCropGuides, LTXVEmptyLatentAudio, LTXVSeparateAVLatent, ManualSigmas, PrimitiveInt, RandomNoise, SamplerCustomAdvanced, VAEDecodeTiled
- unresolved nodes (not in object_info): MarkdownNote, Reroute
- custom nodes: VHS_VideoCombine

## `image_edit_qwen_image`  -  "Image Edit"  -  9 member(s)  -  source: official
- user intent: media=image | task=image_edit | model families: Qwen Image, FireRed, LongCat, Z-Image
- when to use: Use to edit an existing image using Qwen Image, FireRed, LongCat, Z-Image.
- example requests: "build an image workflow using Qwen Image"; "build an image workflow using FireRed"; "build an image workflow using LongCat"; "build an image workflow using Z-Image"; "edit an existing image using Qwen Image"
- description (catalog): Decomposes an image into variable-resolution RGBA layers for independent editing using Qwen-Image-Layered. | Edits images from text instructions using Qwen-Image-Edit-2509 with optional Lightning LoRA for few-step sampling. | Edits images via text instructions using FireRed Image Edit 1.1, a diffusion-based instruction-following editing model. | Edits images via text instructions using LongCat Image Edit, an instruction-following image editing diffusion model. | Edits images via text instructions using Qwen-Image-Edit-2511 with improved character consistency and integrated LoRA. | Generates images from text prompts using Qwen-Image, Alibaba's 20B MMDiT model with excellent multilingual text rendering. | Generates images from text prompts using Qwen-Image-2512, with enhanced human realism and finer natural detail over the base version. | Image Edit blueprint | Upscales images to higher resolution using Z-Image-Turbo.
- official category: Image generation and editing  [spans multiple catalog categories: Image Editing (1), Image generation and editing (8)]

- member files:
    - image_edit - Image Edit blueprint
    - image_edit_firered_image_edit_1_1 - Edits images via text instructions using FireRed Image Edit 1.1, a diffusion-based instruction-following editing model.
    - image_edit_longcat_image_edit - Edits images via text instructions using LongCat Image Edit, an instruction-following image editing diffusion model.
    - image_edit_qwen_2509 - Edits images from text instructions using Qwen-Image-Edit-2509 with optional Lightning LoRA for few-step sampling.
    - image_edit_qwen_2511 - Edits images via text instructions using Qwen-Image-Edit-2511 with improved character consistency and integrated LoRA.
    - image_to_layers_qwen_image_layered - Decomposes an image into variable-resolution RGBA layers for independent editing using Qwen-Image-Layered.
    - image_upscale_z_image_turbo - Upscales images to higher resolution using Z-Image-Turbo.
    - text_to_image_qwen_image - Generates images from text prompts using Qwen-Image, Alibaba's 20B MMDiT model with excellent multilingual text rendering.
    - text_to_image_qwen_image_2512 - Generates images from text prompts using Qwen-Image-2512, with enhanced human realism and finer natural detail over the base version.

- REQUIRED node roles (structural invariants):
    - CLIPLoader  (text encoder / CLIP loader) - all members (9/9)
    - KSampler  (diffusion sampler / denoiser) - all members (9/9)
    - UNETLoader  (diffusion model / UNET loader) - all members (9/9)
    - VAEDecode  (latent -> pixel decode) - all members (9/9)
    - VAELoader  (VAE loader) - all members (9/9)

- OPTIONAL node roles (variant, only in some members):
    - CLIPTextEncode  (prompt text encoding) - 4/9 members
    - FluxGuidance  (guider / sigma / scheduler) - 1/9 members
    - FluxKontextMultiReferenceLatentMethod  (unclassified node role) - 3/9 members
    - ReferenceLatent  (unclassified node role) - 1/9 members
    - TextEncodeQwenImageEdit  (prompt text encoding) - 1/9 members
    - TextEncodeQwenImageEditPlus  (prompt text encoding) - 4/9 members
    - CFGNorm  (unclassified node role) - 4/9 members
    - EmptyQwenImageLayeredLatentImage  (unclassified node role) - 1/9 members
    - EmptySD3LatentImage  (empty latent / canvas) - 2/9 members
    - FluxKontextImageScale  (upscale / resize) - 3/9 members
    - ImageScaleBy  (upscale / resize) - 1/9 members
    - ImageScaleToTotalPixels  (upscale / resize) - 2/9 members
    - ImageUpscaleWithModel  (upscale / resize) - 1/9 members
    - LatentCutToBatch  (unclassified node role) - 1/9 members
    - LoraLoaderModelOnly  (LoRA / model patch loader) - 5/9 members
    - MarkdownNote  (unclassified node role) - 4/9 members
    - ModelSamplingAuraFlow  (unclassified node role) - 8/9 members
    - Note  (unclassified node role) - 2/9 members
    - ResizeImageMaskNode  (unclassified node role) - 1/9 members
    - UpscaleModelLoader  (diffusion model / UNET loader) - 1/9 members
    - VAEEncode  (pixel -> latent encode) - 7/9 members
    - utility/plumbing (some members): ComfySwitchNode, PrimitiveFloat, PrimitiveInt, GetImageSize, PrimitiveBoolean

- connection patterns (role level):
    - clip_loader -> text_encode  [CLIP]  (invariant)
    - vae_loader -> vae_decode  [VAE]  (invariant)
    - model_loader -> lora_loader  [MODEL]  (5/9 members)
    - sampler -> vae_decode  [LATENT]  (8/9 members)
    - text_encode -> sampler  [CONDITIONING]  (5/9 members)
    - upscale -> vae_encode  [IMAGE]  (5/9 members)
    - vae_encode -> sampler  [LATENT]  (6/9 members)
    - vae_loader -> text_encode  [VAE]  (5/9 members)
    - vae_loader -> vae_encode  [VAE]  (7/9 members)

- boundary ports:
    - inputs:  COMBO(clip_name), COMBO(unet_name), COMBO(vae_name), IMAGE(image), INT(seed), STRING(prompt)
    - outputs: IMAGE(IMAGE)

- param variability: varies across members: CLIPLoader, KSampler, UNETLoader, VAEDecode, VAELoader
- unresolved nodes (not in object_info): MarkdownNote, Note

## `image_to_video_kling`  -  8 member(s)  -  source: custom
- user intent: media=video | task=image_to_video | model families: Kling, LTX-2, WAN 2.6
- when to use: Use to generate a video from an input image using Kling, LTX-2, WAN 2.6.
- example requests: "build a video workflow using Kling"; "build a video workflow using LTX-2"; "build a video workflow using WAN 2.6"; "generate a video from an input image using Kling"
- description (catalog+synthesized): API first-last-frame-to-video via Kling O3 (Kling 3.0). Up to 4 reference/keyframe images -> 1 video output. Generates videos with precise semantic control, longer duration, and improved narrative coherence. | API image-to-video via Kling O3 (Kling 3.0). 1 reference image (+ optional audio/text prompt) -> 1 video output. Generates character-consistent video with native audio output and precise storyboard control. | API image-to-video via Wan 2.6. 1 image -> 1 video output. Generates 1080P video with enhanced image quality, smoother motion, and natural movement. | API multi-shot storyboard video via Kling 3.0 (kling-v3). 1 input image (start frame, LoadImage node) -> 1 video output (VHS_VideoCombine). Generates 1-6 sequential shots in a single generation: each shot has its own text prompt (max 512 chars) and duration set directly on the KlingVideoNode. Use for storyboards, scene sequences, and narrative clips with multiple camera cuts. Prompts go into multi_shot.storyboard_N_prompt inputs; multi_shot must match shot count exactly (e.g. '3 storyboards'). Aspect ratio defaults to 16:9, resolution to 720p - override only on explicit user request. | API text-to-video via Wan 2.6. Text prompt only -> 1 video output. Generates 1080P video with enhanced quality, smoother motion, and improved prompt understanding. | API video editing via Kling O3. 1 video + 1 reference image -> 1 edited video output. Enables precise subject editing and scene composition with native audio-visual synchronization.

- member files:
    - Kling3_multiShot - API multi-shot storyboard video via Kling 3.0 (kling-v3). 1 input image (start frame, LoadImage node) -> 1 video output (VHS_VideoCombine). Generates 1-6 sequential shots in a single generation: each shot has its own text prompt (max 512 chars) and duration set directly on the KlingVideoNode. Use for storyboards, scene sequences, and narrative clips with multiple camera cuts. Prompts go into multi_shot.storyboard_N_prompt inputs; multi_shot must match shot count exactly (e.g. '3 storyboards'). Aspect ratio defaults to 16:9, resolution to 720p - override only on explicit user request.
    - api_kling_o3_flf2v - API first-last-frame-to-video via Kling O3 (Kling 3.0). Up to 4 reference/keyframe images -> 1 video output. Generates videos with precise semantic control, longer duration, and improved narrative coherence.
    - api_kling_o3_i2v - API image-to-video via Kling O3 (Kling 3.0). 1 reference image (+ optional audio/text prompt) -> 1 video output. Generates character-consistent video with native audio output and precise storyboard control.
    - api_kling_o3_video_edit - API video editing via Kling O3. 1 video + 1 reference image -> 1 edited video output. Enables precise subject editing and scene composition with native audio-visual synchronization.
    - api_ltxv_image_to_video
    - api_ltxv_text_to_video
    - api_wan2_6_i2v - API image-to-video via Wan 2.6. 1 image -> 1 video output. Generates 1080P video with enhanced image quality, smoother motion, and natural movement.
    - api_wan2_6_t2v - API text-to-video via Wan 2.6. Text prompt only -> 1 video output. Generates 1080P video with enhanced quality, smoother motion, and improved prompt understanding.

- REQUIRED node roles (structural invariants):
    - GetVideoComponents  (unclassified node role) - all members (8/8)
    - VHS_VideoCombine  (save / preview / combine output) - all members (8/8)

- OPTIONAL node roles (variant, only in some members):
    - LoadImage  (image input / load) - 6/8 members
    - ImageBatchMulti  (unclassified node role) - 2/8 members
    - KlingOmniProEditVideoNode  (external API generation node) - 1/8 members
    - KlingOmniProFirstLastFrameNode  (external API generation node) - 1/8 members
    - KlingOmniProImageToVideoNode  (external API generation node) - 1/8 members
    - KlingVideoNode  (unclassified node role) - 1/8 members
    - LoadVideo  (video input / load) - 1/8 members
    - LtxvApiImageToVideo  (unclassified node role) - 1/8 members
    - LtxvApiTextToVideo  (unclassified node role) - 1/8 members
    - WanImageToVideoApi  (unclassified node role) - 1/8 members
    - WanTextToVideoApi  (unclassified node role) - 1/8 members

- connection patterns (role level):
    - GetVideoComponents -> save_output  [AUDIO]  (invariant)
    - GetVideoComponents -> save_output  [FLOAT]  (invariant)
    - GetVideoComponents -> save_output  [IMAGE]  (invariant)

- boundary ports:
    - inputs:  IMAGE(image_loader)
    - outputs: AUDIO(save_output), IMAGE(save_output)

- param variability: constant across members: GetVideoComponents, VHS_VideoCombine
- custom nodes: ImageBatchMulti, VHS_VideoCombine

## `character_sheet_nano_banana`  -  6 member(s)  -  source: custom
- user intent: media=image | task=character_sheet | model families: Nano-Banana, Magnific, Topaz, Veo, Z-Image
- when to use: Use to generate a multi-pose character sheet using Nano-Banana, Magnific, Topaz, Veo, Z-Image.
- example requests: "build an image workflow using Nano-Banana"; "build an image workflow using Magnific"; "build an image workflow using Topaz"; "build an image workflow using Veo"; "build an image workflow using Z-Image"; "generate a multi-pose character sheet using Nano-Banana"
- description (catalog+synthesized): API character sheet generation FOR FACE CLOSEUPS via Nano-Banana Pro. 1 character image -> 1 image output (3x3 sheet). Uses an LLM call to generate a prompt from the reference, then renders 9 character views with varying facial expressions in a single sheet. | API character sheet generation via Nano-Banana Pro. 1 character image -> 1 image output (3x3 sheet). Uses an LLM call to generate a prompt from the reference, then renders 9 character views with varying body pose in a single sheet. | API image enhancement/upscaling via Topaz Reimagine. 1 image -> 1 enhanced image output. Applies face enhancement and detail restoration for professional results. | API image relighting via Magnific. 1 source image + 1 lighting reference image -> 1 relit image output. Applies the lighting conditions from the reference onto the source image. | API upscale and outpaint via Nano-Banana 2. 1 image -> 1 image output. Upscales the input image while also generating new content around the edges to expand the overall dimensions, guided by the original image's style and content.

- member files:
    - NanoBanana2_outpaintUpscale - API upscale and outpaint via Nano-Banana 2. 1 image -> 1 image output. Upscales the input image while also generating new content around the edges to expand the overall dimensions, guided by the original image's style and content.
    - NanoBananaPro_3x3CharacterSheet - API character sheet generation via Nano-Banana Pro. 1 character image -> 1 image output (3x3 sheet). Uses an LLM call to generate a prompt from the reference, then renders 9 character views with varying body pose in a single sheet.
    - NanoBananaPro_3x3CharacterSheet_closeups - API character sheet generation FOR FACE CLOSEUPS via Nano-Banana Pro. 1 character image -> 1 image output (3x3 sheet). Uses an LLM call to generate a prompt from the reference, then renders 9 character views with varying facial expressions in a single sheet.
    - api_magnific_image_relight - API image relighting via Magnific. 1 source image + 1 lighting reference image -> 1 relit image output. Applies the lighting conditions from the reference onto the source image.
    - api_topaz_image_enhance - API image enhancement/upscaling via Topaz Reimagine. 1 image -> 1 enhanced image output. Applies face enhancement and detail restoration for professional results.
    - api_veo3

- REQUIRED node roles (structural invariants):
    - LoadImage  (image input / load) - all members (6/6)

- OPTIONAL node roles (variant, only in some members):
    - GeminiImage2Node  (external API generation node) - 2/6 members
    - GeminiNanoBanana2  (external API generation node) - 1/6 members
    - GeminiNode  (external API generation node) - 2/6 members
    - MagnificImageRelightNode  (external API generation node) - 1/6 members
    - SaveImage  (save / preview / combine output) - 5/6 members
    - SaveVideo  (save / preview / combine output) - 1/6 members
    - TopazImageEnhance  (external API generation node) - 1/6 members
    - Veo3VideoGenerationNode  (external API generation node) - 1/6 members
    - utility/plumbing (some members): PrimitiveStringMultiline

- connection patterns (role level):
    - image_loader -> api_node  [IMAGE]  (invariant)
    - api_node -> save_output  [IMAGE]  (5/6 members)

- boundary ports:
    - inputs:  IMAGE(image_loader)
    - outputs: IMAGE(save_output)

- param variability: varies across members: LoadImage

## `image_edit_flux_2_klein`  -  "Image Edit (Flux.2 Dev)"  -  6 member(s)  -  source: mixed
- user intent: media=image | task=image_edit | model families: Flux 2 Klein, Flux 2, Ideogram
- when to use: Use to edit an existing image using Flux 2 Klein, Flux 2, Ideogram.
- example requests: "build an image workflow using Flux 2 Klein"; "build an image workflow using Flux 2"; "build an image workflow using Ideogram"; "edit an existing image using Flux 2 Klein"
- description (catalog+synthesized): Edits an image from text instructions using Flux.2 [dev], with guidance, schedulers, and optional Turbo LoRAs. | Edits an input image via text instructions using FLUX.2 [klein] 4B. | Generates images from prompts using FLUX.2 [dev]: a newer 32B rectified-flow stack with distilled guidance plus a stronger long-context multimodal encoder for complex scenes, sharper typography/UI text, anatomy, lighting, and high-resolution latent decoding. | This subgraph generates images using Ideogram v4, accepting plain text or structured JSON prompts for precise layout and style control. It suits detailed illustrations, concept art, or marketing visuals needing predictable composition and color palettes. The model uses flow-matching with asymmetric guidance, so no negative prompt is needed, but JSON prompts yield the best results. | [Local] image editing via Flux. 1 image input -> 1 image output. Performs image editing using the Flux 2 Klein distilled model.
- official category: Image generation and editing  [single category (+2 uncategorized): Image generation and editing (4)]

- member files:
    - image_edit_flux_2_dev - Edits an image from text instructions using Flux.2 [dev], with guidance, schedulers, and optional Turbo LoRAs.
    - image_edit_flux_2_klein_4b - Edits an input image via text instructions using FLUX.2 [klein] 4B.
    - image_flux2_klein_image_edit_9b_distilled - [Local] image editing via Flux. 1 image input -> 1 image output. Performs image editing using the Flux 2 Klein distilled model.
    - image_flux2_klein_text_to_image
    - text_to_image_flux_2_dev - Generates images from prompts using FLUX.2 [dev]: a newer 32B rectified-flow stack with distilled guidance plus a stronger long-context multimodal encoder for complex scenes, sharper typography/UI text, anatomy, lighting, and high-resolution latent decoding.
    - text_to_image_ideogram_v4 - This subgraph generates images using Ideogram v4, accepting plain text or structured JSON prompts for precise layout and style control. It suits detailed illustrations, concept art, or marketing visuals needing predictable composition and color palettes. The model uses flow-matching with asymmetric guidance, so no negative prompt is needed, but JSON prompts yield the best results.

- REQUIRED node roles (structural invariants):
    - CLIPTextEncode  (prompt text encoding) - all members (6/6)
    - UNETLoader  (diffusion model / UNET loader) - all members (6/6)
    - CLIPLoader  (text encoder / CLIP loader) - all members (6/6)
    - EmptyFlux2LatentImage  (unclassified node role) - all members (6/6)
    - KSamplerSelect  (diffusion sampler / denoiser) - all members (6/6)
    - RandomNoise  (unclassified node role) - all members (6/6)
    - SamplerCustomAdvanced  (diffusion sampler / denoiser) - all members (6/6)
    - VAEDecode  (latent -> pixel decode) - all members (6/6)
    - VAELoader  (VAE loader) - all members (6/6)

- OPTIONAL node roles (variant, only in some members):
    - ReferenceLatent  (unclassified node role) - 3/6 members
    - BasicGuider  (guider / sigma / scheduler) - 2/6 members
    - CFGGuider  (guider / sigma / scheduler) - 3/6 members
    - CFGOverride  (unclassified node role) - 1/6 members
    - ConditioningZeroOut  (conditioning combine / edit) - 2/6 members
    - CustomCombo  (unclassified node role) - 1/6 members
    - DualModelGuider  (guider / sigma / scheduler) - 1/6 members
    - Flux2Scheduler  (unclassified node role) - 5/6 members
    - FluxGuidance  (guider / sigma / scheduler) - 2/6 members
    - Ideogram4Scheduler  (external API generation node) - 1/6 members
    - ImageScaleToTotalPixels  (upscale / resize) - 2/6 members
    - LoadImage  (image input / load) - 1/6 members
    - LoraLoaderModelOnly  (LoRA / model patch loader) - 2/6 members
    - SaveImage  (save / preview / combine output) - 2/6 members
    - VAEEncode  (pixel -> latent encode) - 3/6 members
    - utility/plumbing (some members): JsonExtractString, ComfyNumberConvert, ComfyMathExpression, ComfySwitchNode, PrimitiveInt, GetImageSize, PrimitiveBoolean, PrimitiveStringMultiline, StringReplace

- connection patterns (role level):
    - EmptyFlux2LatentImage -> sampler  [LATENT]  (invariant)
    - RandomNoise -> sampler  [NOISE]  (invariant)
    - clip_loader -> text_encode  [CLIP]  (invariant)
    - guidance -> sampler  [GUIDER]  (invariant)
    - sampler -> sampler  [SAMPLER]  (invariant)
    - sampler -> vae_decode  [LATENT]  (invariant)
    - vae_loader -> vae_decode  [VAE]  (invariant)
    - Flux2Scheduler -> sampler  [SIGMAS]  (5/6 members)
    - GetImageSize -> EmptyFlux2LatentImage  [INT]  (3/6 members)
    - GetImageSize -> Flux2Scheduler  [INT]  (3/6 members)
    - ReferenceLatent -> guidance  [CONDITIONING]  (3/6 members)
    - model_loader -> guidance  [MODEL]  (4/6 members)
    - text_encode -> guidance  [CONDITIONING]  (4/6 members)
    - vae_encode -> ReferenceLatent  [LATENT]  (3/6 members)

- boundary ports:
    - inputs:  COMBO(clip_name), COMBO(unet_name), COMBO(vae_name), STRING(text)
    - outputs: IMAGE(IMAGE)

- param variability: varies across members: CLIPLoader, CLIPTextEncode, EmptyFlux2LatentImage, KSamplerSelect, RandomNoise, SamplerCustomAdvanced, UNETLoader, VAEDecode, VAELoader

## `image_edit_nano_banana`  -  4 member(s)  -  source: custom
- user intent: media=image | task=image_edit | model families: Nano-Banana, Gemini, Kling, Seedream
- when to use: Use to edit an existing image using Nano-Banana, Gemini, Kling, Seedream.
- example requests: "build an image workflow using Nano-Banana"; "build an image workflow using Gemini"; "build an image workflow using Kling"; "build an image workflow using Seedream"; "edit an existing image using Nano-Banana"
- description (catalog+synthesized): API image editing/generation via Nano-Banana 2. Up to 6 reference images + text prompt -> 1 image output. Edits images while maintaining subject consistency, or uses references as style guides for new image generation. | API image editing/generation via Nano-Banana Pro (Gemini 3.0 Pro). 2 image inputs -> 1 image output. Studio-quality 4K generation and editing with enhanced text rendering and character consistency.

- member files:
    - api_bytedance_seedream_5_0_lite_image_edit
    - api_kling_o3_image
    - imageEdit_nano_banana2 - API image editing/generation via Nano-Banana 2. Up to 6 reference images + text prompt -> 1 image output. Edits images while maintaining subject consistency, or uses references as style guides for new image generation.
    - imageEdit_nano_banana_pro - API image editing/generation via Nano-Banana Pro (Gemini 3.0 Pro). 2 image inputs -> 1 image output. Studio-quality 4K generation and editing with enhanced text rendering and character consistency.

- REQUIRED node roles (structural invariants):
    - LoadImage  (image input / load) - all members (4/4), 2 required instances  **[PAIRED: 2x required]**
        - instance feeds ['AILab_ImageToList'] / fed by []  (x4)
        - instance feeds ['ImageBatchMulti'] / fed by []  (x4)
    - SaveImage  (save / preview / combine output) - all members (4/4)

- OPTIONAL node roles (variant, only in some members):
    - AILab_ImageToList  (unclassified node role) - 2/4 members
    - ByteDanceSeedreamNode  (external API generation node) - 1/4 members
    - GeminiImage2Node  (external API generation node) - 1/4 members
    - GeminiNanoBanana2  (external API generation node) - 1/4 members
    - ImageBatchMulti  (unclassified node role) - 2/4 members
    - ImageListToImageBatch  (unclassified node role) - 2/4 members
    - KlingOmniProImageNode  (external API generation node) - 1/4 members

- connection patterns (role level):
    - api_node -> save_output  [IMAGE]  (invariant)
    - AILab_ImageToList -> ImageListToImageBatch  [IMAGE]  (2/4 members)
    - ImageBatchMulti -> api_node  [IMAGE]  (2/4 members)
    - ImageListToImageBatch -> api_node  [IMAGE]  (2/4 members)
    - image_loader -> AILab_ImageToList  [IMAGE]  (2/4 members)
    - image_loader -> ImageBatchMulti  [IMAGE]  (2/4 members)

- boundary ports:
    - inputs:  IMAGE(image_loader)
    - outputs: IMAGE(save_output)

- param variability: varies across members: SaveImage
- custom nodes: AILab_ImageToList, ImageBatchMulti, ImageListToImageBatch

## `image_generation_2`  -  "Crop Images 2x2"  -  4 member(s)  -  source: official
- user intent: media=image | task=image_generation | model families: n/a
- when to use: Use to generate an image.
- example requests: "build an image workflow"; "generate an image"
- description (catalog): Extracts one image frame from a video at a chosen index, with optional trim and FPS control. | Splits an image into a 2×2 grid of four equal tiles. | Splits an image into a 3×3 grid of nine equal tiles. | Splits an image into a configurable columns×rows grid of equal tiles for tiled generation or processing.
- official category: Image Tools  [spans multiple catalog categories: Image Tools (3), Video Tools (1)]

- member files:
    - crop_images_2x2 - Splits an image into a 2×2 grid of four equal tiles.
    - crop_images_3x3 - Splits an image into a 3×3 grid of nine equal tiles.
    - get_any_video_frame - Extracts one image frame from a video at a chosen index, with optional trim and FPS control.
    - split_image_grid_to_tiles - Splits an image into a configurable columns×rows grid of equal tiles for tiled generation or processing.

- REQUIRED node roles (structural invariants):
    - (none)
    - utility/plumbing (always present): ComfyMathExpression(6x), PrimitiveInt(2x), GetImageSize

- OPTIONAL node roles (variant, only in some members):
    - ImageCropV2  (unclassified node role) - 2/4 members
    - BatchImagesNode  (unclassified node role) - 2/4 members
    - GetVideoComponents  (unclassified node role) - 1/4 members
    - ImageFromBatch  (unclassified node role) - 1/4 members
    - SplitImageToTileList  (unclassified node role) - 1/4 members
    - utility/plumbing (some members): PrimitiveBoundingBox

- connection patterns (role level):
    - GetImageSize -> ComfyMathExpression  [INT]  (invariant)
    - PrimitiveInt -> ComfyMathExpression  [INT]  (invariant)
    - ComfyMathExpression -> ComfyMathExpression  [INT]  (2/4 members)
    - ComfyMathExpression -> PrimitiveBoundingBox  [INT]  (2/4 members)
    - ImageCropV2 -> BatchImagesNode  [IMAGE]  (2/4 members)
    - PrimitiveBoundingBox -> ImageCropV2  [BOUNDING_BOX]  (2/4 members)

- boundary ports:
    - inputs:  IMAGE(image)
    - outputs: IMAGE(IMAGE), IMAGE(IMAGE_1), IMAGE(IMAGE_2), IMAGE(IMAGE_3), IMAGE(IMAGE_4)

- param variability: constant across members: GetImageSize; varies across members: ComfyMathExpression, PrimitiveInt

## `image_generation_gemini`  -  "Image Captioning(Gemini)"  -  4 member(s)  -  source: official
- user intent: media=text | task=image_generation | model families: Gemini
- when to use: Use to generate an image using Gemini.
- example requests: "build a text workflow using Gemini"; "generate an image using Gemini"
- description (catalog): Expands short text prompts into detailed descriptions using a text generation model for better generation quality. | Generates descriptive captions for images using Google's Gemini multimodal LLM. | Generates descriptive captions for video input using Google's Gemini multimodal LLM. | Manipulates individual RGBA channels for masking, compositing, and channel effects.
- official category: Image Tools  [spans multiple catalog categories: Image Tools (2), Text Tools (1), Video Tools (1)]

- member files:
    - image_captioning_gemini - Generates descriptive captions for images using Google's Gemini multimodal LLM.
    - image_channels - Manipulates individual RGBA channels for masking, compositing, and channel effects.
    - prompt_enhance - Expands short text prompts into detailed descriptions using a text generation model for better generation quality.
    - video_captioning_gemini - Generates descriptive captions for video input using Google's Gemini multimodal LLM.

- REQUIRED node roles (structural invariants):
    - (none)

- OPTIONAL node roles (variant, only in some members):
    - GLSLShader  (unclassified node role) - 1/4 members
    - GeminiNode  (external API generation node) - 3/4 members

- connection patterns (role level):

- boundary ports:
    - inputs:  COMBO(model), IMAGE(images), STRING(prompt)
    - outputs: STRING(STRING)

- param variability: no single-instance invariant params to compare

## `text_to_image_ernie`  -  "Audio Generation (Stable Audio 3 Medium Base)"  -  4 member(s)  -  source: official
- user intent: media=audio | task=text_to_image | model families: ERNIE, Qwen Image, Stable Audio
- when to use: Use to generate an image from a text prompt using ERNIE, Qwen Image, Stable Audio.
- example requests: "build an audio workflow using ERNIE"; "build an audio workflow using Qwen Image"; "build an audio workflow using Stable Audio"; "generate an image from a text prompt using ERNIE"
- description (catalog): Faster ERNIE Image Turbo variant (~8B DiT, distilled for fewer sampling steps): same strengths in Chinese/English on-image text and layout-heavy graphics as the base ERNIE Image lineup, with bundled encoders and VAE. | Generates images from text prompts using Baidu's open ERNIE Image (~8B DiT): bilingual in-image typography and layouts (posters, infographics, multi-panel compositions) alongside general scenes, with bundled encoders and VAE. | Generates music, instrument loops, sound effects, and one-shots from text using Stable Audio 3 Medium, with optional Qwen 3.5 category-based prompt expansion (Music, Instrument, SFX, One-shot). | Generates music, instrument loops, sound effects, and one-shots from text using the Stable Audio 3 Medium base checkpoint, with optional Qwen 3.5 category-based prompt expansion (Music, Instrument, SFX, One-shot).
- official category: Audio  [spans multiple catalog categories: Audio (2), Image generation and editing (2)]

- member files:
    - audio_generation_stable_audio_3_medium - Generates music, instrument loops, sound effects, and one-shots from text using Stable Audio 3 Medium, with optional Qwen 3.5 category-based prompt expansion (Music, Instrument, SFX, One-shot).
    - audio_generation_stable_audio_3_medium_base - Generates music, instrument loops, sound effects, and one-shots from text using the Stable Audio 3 Medium base checkpoint, with optional Qwen 3.5 category-based prompt expansion (Music, Instrument, SFX, One-shot).
    - text_to_image_ernie_image - Generates images from text prompts using Baidu's open ERNIE Image (~8B DiT): bilingual in-image typography and layouts (posters, infographics, multi-panel compositions) alongside general scenes, with bundled encoders and VAE.
    - text_to_image_ernie_image_turbo - Faster ERNIE Image Turbo variant (~8B DiT, distilled for fewer sampling steps): same strengths in Chinese/English on-image text and layout-heavy graphics as the base ERNIE Image lineup, with bundled encoders and VAE.

- REQUIRED node roles (structural invariants):
    - CLIPLoader  (text encoder / CLIP loader) - all members (4/4), 2 required instances  **[PAIRED: 2x required]**
        - instance feeds ['TextGenerate'] / fed by []  (x4)
        - instance feeds ['text_encode', 'text_encode'] / fed by []  (x3)
        - instance feeds ['text_encode'] / fed by []  (x1)
    - CLIPTextEncode  (prompt text encoding) - all members (4/4)
    - KSampler  (diffusion sampler / denoiser) - all members (4/4)
    - TextGenerate  (unclassified node role) - all members (4/4)
    - utility/plumbing (always present): PreviewAny(3x), StringReplace(3x), ComfySwitchNode, PrimitiveBoolean, PrimitiveStringMultiline

- OPTIONAL node roles (variant, only in some members):
    - CheckpointLoaderSimple  (diffusion model / UNET loader) - 2/4 members
    - ConditioningZeroOut  (conditioning combine / edit) - 1/4 members
    - CustomCombo  (unclassified node role) - 2/4 members
    - EmptyFlux2LatentImage  (unclassified node role) - 2/4 members
    - EmptyLatentAudio  (empty latent / canvas) - 2/4 members
    - UNETLoader  (diffusion model / UNET loader) - 2/4 members
    - VAEDecode  (latent -> pixel decode) - 2/4 members
    - VAEDecodeAudio  (latent -> pixel decode) - 2/4 members
    - VAELoader  (VAE loader) - 2/4 members
    - utility/plumbing (some members): ComfyMathExpression, JsonExtractString, PrimitiveFloat

- connection patterns (role level):
    - ComfySwitchNode -> text_encode  [STRING]  (invariant)
    - PreviewAny -> StringReplace  [STRING]  (invariant)
    - PrimitiveBoolean -> ComfySwitchNode  [BOOLEAN]  (invariant)
    - PrimitiveStringMultiline -> ComfySwitchNode  [STRING]  (invariant)
    - PrimitiveStringMultiline -> StringReplace  [STRING]  (invariant)
    - StringReplace -> StringReplace  [STRING]  (invariant)
    - StringReplace -> TextGenerate  [STRING]  (invariant)
    - TextGenerate -> ComfySwitchNode  [STRING]  (invariant)
    - clip_loader -> TextGenerate  [CLIP]  (invariant)
    - clip_loader -> text_encode  [CLIP]  (invariant)
    - model_loader -> sampler  [MODEL]  (invariant)
    - sampler -> vae_decode  [LATENT]  (invariant)
    - text_encode -> sampler  [CONDITIONING]  (invariant)
    - ComfyMathExpression -> PreviewAny  [INT]  (2/4 members)
    - ComfySwitchNode -> PreviewAny  [STRING]  (2/4 members)
    - CustomCombo -> JsonExtractString  [STRING]  (2/4 members)
    - EmptyFlux2LatentImage -> sampler  [LATENT]  (2/4 members)
    - JsonExtractString -> StringReplace  [STRING]  (2/4 members)
    - PrimitiveFloat -> ComfyMathExpression  [FLOAT]  (2/4 members)
    - PrimitiveFloat -> latent_source  [FLOAT]  (2/4 members)
    - TextGenerate -> PreviewAny  [STRING]  (2/4 members)
    - latent_source -> sampler  [LATENT]  (2/4 members)
    - model_loader -> vae_decode  [VAE]  (2/4 members)
    - vae_loader -> vae_decode  [VAE]  (2/4 members)

- boundary ports:
    - inputs:  BOOLEAN(use_reprompt), BOOLEAN(value_1), COMBO(category), COMBO(ckpt_name), COMBO(clip_name), COMBO(clip_name_1), COMBO(qwen_clip), COMBO(sa_clip), COMBO(unet_name), COMBO(vae_name), FLOAT(duration), INT(height), INT(seed), INT(width), STRING(user_input), STRING(value)
    - outputs: AUDIO(AUDIO), IMAGE(IMAGE)

- param variability: constant across members: CLIPTextEncode, ComfySwitchNode, PrimitiveBoolean, PrimitiveStringMultiline; varies across members: KSampler, TextGenerate

## `upscale_magnific`  -  4 member(s)  -  source: custom
- user intent: media=image | task=upscale | model families: Magnific, Flux
- when to use: Use to upscale / enhance an image using Magnific, Flux.
- example requests: "build an image workflow using Magnific"; "build an image workflow using Flux"; "upscale / enhance an image using Magnific"
- description (catalog): API creative image upscaling via Magnific. 1 image -> 1 upscaled image output. Supports up to 16x enlargement with creative detail enhancement. | API precise image upscaling via Magnific. 1 image -> 1 high-resolution image output. Upscales with strict detail preservation and enhanced sharpness. | Local image upscaling using UltimateSD upscale node (this uses a diffusion model for the upscale process, allowing a creative upscale that invents details). Setup with Flux-1 dev fp8. 1 image -> 1 upscaled image output. | Local, simple image upscaling via specified ESRGAN model. 1 image -> 1 upscaled image output. Supports various models.

- member files:
    - api_magnific_image_upscale_creative - API creative image upscaling via Magnific. 1 image -> 1 upscaled image output. Supports up to 16x enlargement with creative detail enhancement.
    - api_magnific_image_upscale_precise - API precise image upscaling via Magnific. 1 image -> 1 high-resolution image output. Upscales with strict detail preservation and enhanced sharpness.
    - upscale_ultimateSD - Local image upscaling using UltimateSD upscale node (this uses a diffusion model for the upscale process, allowing a creative upscale that invents details). Setup with Flux-1 dev fp8. 1 image -> 1 upscaled image output.
    - upscale_using_model - Local, simple image upscaling via specified ESRGAN model. 1 image -> 1 upscaled image output. Supports various models.

- REQUIRED node roles (structural invariants):
    - LoadImage  (image input / load) - all members (4/4)
    - SaveImage  (save / preview / combine output) - all members (4/4)

- OPTIONAL node roles (variant, only in some members):
    - CLIPTextEncode  (prompt text encoding) - 1/4 members
    - CheckpointLoaderSimple  (diffusion model / UNET loader) - 1/4 members
    - ImageUpscaleWithModel  (upscale / resize) - 1/4 members
    - MagnificImageUpscalerCreativeNode  (upscale / resize) - 1/4 members
    - MagnificImageUpscalerPreciseV2Node  (upscale / resize) - 1/4 members
    - UltimateSDUpscale  (upscale / resize) - 1/4 members
    - UpscaleModelLoader  (diffusion model / UNET loader) - 2/4 members

- connection patterns (role level):
    - image_loader -> upscale  [IMAGE]  (invariant)
    - upscale -> save_output  [IMAGE]  (invariant)
    - model_loader -> upscale  [UPSCALE_MODEL]  (2/4 members)

- boundary ports:
    - inputs:  IMAGE(image_loader)
    - outputs: IMAGE(save_output)

- param variability: constant across members: SaveImage; varies across members: LoadImage
- custom nodes: UltimateSDUpscale

## `3d_generation_meshy`  -  3 member(s)  -  source: custom
- user intent: media=3d | task=3d_generation | model families: Meshy
- when to use: Use to generate a 3D model using Meshy.
- example requests: "build a 3d workflow using Meshy"; "generate a 3D model using Meshy"
- description (catalog): API image-to-3D via Meshy 6. 1 image -> 1 3D model output. Generates characters, objects, or mechanical parts with production-quality geometry and clean topology. | API multi-image-to-3D via Meshy 6. 3+ images -> 1 3D model output. More input views yield better detail capture, accurate proportions, and cleaner mesh structure. | API text-to-3D via Meshy 6. Text prompt only -> 1 3D model output. Creates characters, mechanical objects, or game-ready low-poly assets with refined geometry.

- member files:
    - api_meshy_image_to_model - API image-to-3D via Meshy 6. 1 image -> 1 3D model output. Generates characters, objects, or mechanical parts with production-quality geometry and clean topology.
    - api_meshy_multi_image_to_model - API multi-image-to-3D via Meshy 6. 3+ images -> 1 3D model output. More input views yield better detail capture, accurate proportions, and cleaner mesh structure.
    - api_meshy_text_to_model - API text-to-3D via Meshy 6. Text prompt only -> 1 3D model output. Creates characters, mechanical objects, or game-ready low-poly assets with refined geometry.

- REQUIRED node roles (structural invariants):
    - SaveGLB  (unclassified node role) - all members (3/3), 2 required instances  **[PAIRED: 2x required]**

- OPTIONAL node roles (variant, only in some members):
    - LoadImage  (image input / load) - 2/3 members
    - MeshyImageToModelNode  (external API generation node) - 1/3 members
    - MeshyMultiImageToModelNode  (external API generation node) - 1/3 members
    - MeshyTextToModelNode  (external API generation node) - 1/3 members

- connection patterns (role level):
    - api_node -> SaveGLB  [FILE_3D_FBX]  (invariant)
    - api_node -> SaveGLB  [FILE_3D_GLB]  (invariant)
    - image_loader -> api_node  [IMAGE]  (2/3 members)

- boundary ports:
    - inputs:  IMAGE(image_loader)
    - outputs: (none)

- param variability: no single-instance invariant params to compare

## `depth_estimation_moge`  -  "Geometry Estimation (MoGe)"  -  3 member(s)  -  source: official
- user intent: media=3d | task=depth_estimation | model families: MoGe
- when to use: Use to estimate a depth map using MoGe.
- example requests: "build a 3d workflow using MoGe"; "estimate a depth map using MoGe"
- description (catalog): Estimates 3D scene geometry from an input image using MoGe, outputting a mesh plus OpenGL and DirectX normal maps. | Estimates monocular depth from an input image using MoGe, outputting both raw and colorized depth maps plus a mask. | Estimates monocular depth from an input video using MoGe, outputting both raw and colorized depth maps plus a mask.
- official category: Conditioning & Preprocessors  [spans multiple catalog categories: 3D (1), Conditioning & Preprocessors (2)]

- member files:
    - geometry_estimation_moge - Estimates 3D scene geometry from an input image using MoGe, outputting a mesh plus OpenGL and DirectX normal maps.
    - image_depth_estimation_moge - Estimates monocular depth from an input image using MoGe, outputting both raw and colorized depth maps plus a mask.
    - video_depth_estimation_moge - Estimates monocular depth from an input video using MoGe, outputting both raw and colorized depth maps plus a mask.

- REQUIRED node roles (structural invariants):
    - MoGeRender  (unclassified node role) - all members (3/3), 2 required instances  **[PAIRED: 2x required]**
        - instance feeds [] / fed by ['MoGeInference']  (x6)
        - instance feeds ['ImageToMask'] / fed by ['MoGeInference']  (x2)
    - LoadMoGeModel  (unclassified node role) - all members (3/3)
    - MoGeInference  (unclassified node role) - all members (3/3)
    - ResizeImagesByLongerEdge  (unclassified node role) - all members (3/3)
    - utility/plumbing (always present): ComfySwitchNode(2x), ComfyMathExpression, GetImageSize

- OPTIONAL node roles (variant, only in some members):
    - GetVideoComponents  (unclassified node role) - 1/3 members
    - ImageToMask  (unclassified node role) - 2/3 members
    - MoGePointMapToMesh  (unclassified node role) - 1/3 members

- connection patterns (role level):
    - ComfyMathExpression -> ComfySwitchNode  [BOOLEAN]  (invariant)
    - ComfySwitchNode -> ComfySwitchNode  [IMAGE]  (invariant)
    - ComfySwitchNode -> MoGeInference  [IMAGE]  (invariant)
    - GetImageSize -> ComfyMathExpression  [INT]  (invariant)
    - LoadMoGeModel -> MoGeInference  [MOGE_MODEL]  (invariant)
    - MoGeInference -> MoGeRender  [MOGE_GEOMETRY]  (invariant)
    - ResizeImagesByLongerEdge -> ComfySwitchNode  [IMAGE]  (invariant)
    - MoGeRender -> ImageToMask  [IMAGE]  (2/3 members)

- boundary ports:
    - inputs:  BOOLEAN(switch), COMBO(moge_model), IMAGE(source_image), INT(inference_batch_size), INT(inference_resolution)
    - outputs: IMAGE(depth), IMAGE(depth_colored), MASK(MASK)

- param variability: constant across members: ComfyMathExpression, GetImageSize, LoadMoGeModel, ResizeImagesByLongerEdge; varies across members: MoGeInference

## `pose_estimation_sdpose`  -  "Image to Pose Map (SDPose Multi-Person)"  -  3 member(s)  -  source: official
- user intent: media=image | task=pose_estimation | model families: SDPose
- when to use: Use to estimate a pose map using SDPose.
- example requests: "build an image workflow using SDPose"; "estimate a pose map using SDPose"
- description (catalog): Detects multiple people in an image and outputs per-person pose keypoints, skeleton renders, and bounding boxes using SDPose. | Extracts human pose keypoints and stick-figure visuals from an image using SDPose-OOD, with optional bounding-box input per subject. | Extracts multi-person pose keypoints and skeleton frame sequences from video using SDPose with built-in person detection.
- official category: Conditioning & Preprocessors  [pure: Conditioning & Preprocessors (3)]

- member files:
    - image_to_pose_map_sdpose_multi_person - Detects multiple people in an image and outputs per-person pose keypoints, skeleton renders, and bounding boxes using SDPose.
    - image_to_pose_map_sdpose_ood - Extracts human pose keypoints and stick-figure visuals from an image using SDPose-OOD, with optional bounding-box input per subject.
    - video_to_pose_map_sdpose_multi_person - Extracts multi-person pose keypoints and skeleton frame sequences from video using SDPose with built-in person detection.

- REQUIRED node roles (structural invariants):
    - CheckpointLoaderSimple  (diffusion model / UNET loader) - all members (3/3)
    - ResizeImageMaskNode  (unclassified node role) - all members (3/3)
    - SDPoseDrawKeypoints  (unclassified node role) - all members (3/3)
    - SDPoseKeypointExtractor  (unclassified node role) - all members (3/3)

- OPTIONAL node roles (variant, only in some members):
    - GetVideoComponents  (unclassified node role) - 1/3 members
    - RTDETR_detect  (unclassified node role) - 2/3 members
    - UNETLoader  (diffusion model / UNET loader) - 2/3 members

- connection patterns (role level):
    - ResizeImageMaskNode -> SDPoseKeypointExtractor  [IMAGE]  (invariant)
    - SDPoseKeypointExtractor -> SDPoseDrawKeypoints  [POSE_KEYPOINT]  (invariant)
    - model_loader -> SDPoseKeypointExtractor  [MODEL]  (invariant)
    - model_loader -> SDPoseKeypointExtractor  [VAE]  (invariant)
    - RTDETR_detect -> SDPoseKeypointExtractor  [BOUNDING_BOX]  (2/3 members)
    - ResizeImageMaskNode -> RTDETR_detect  [IMAGE]  (2/3 members)
    - model_loader -> RTDETR_detect  [MODEL]  (2/3 members)

- boundary ports:
    - inputs:  BOOLEAN(draw_body), BOOLEAN(draw_face), BOOLEAN(draw_feet), BOOLEAN(draw_hands), COMBO(ckpt_name), COMBO(class_name), COMBO(scale_method), COMBO(unet_name), FLOAT(score_threshold), FLOAT(threshold), IMAGE,MASK(input), INT(face_point_size), INT(max_detections), INT(resize_type.longer_size), INT(stick_width)
    - outputs: BOUNDING_BOX(bboxes), IMAGE(IMAGE), POSE_KEYPOINT(keypoints)

- param variability: constant across members: CheckpointLoaderSimple, SDPoseDrawKeypoints, SDPoseKeypointExtractor; varies across members: ResizeImageMaskNode

## `text_to_video_bernini`  -  "Image Edit (Bernini-R)"  -  3 member(s)  -  source: official
- user intent: media=video | task=text_to_video | model families: Bernini, WAN 2.2, Depth Anything, SAM3
- when to use: Use to generate a video from a text prompt using Bernini, WAN 2.2, Depth Anything, SAM3.
- example requests: "build a video workflow using Bernini"; "build a video workflow using WAN 2.2"; "build a video workflow using Depth Anything"; "build a video workflow using SAM3"; "generate a video from a text prompt using Bernini"
- description (catalog): Edits a single image using a text prompt, leveraging Bernini-R's latent semantic planning for changes like object addition, removal, or style transfer. Ideal for creative edits requiring precise semantic understanding, such as adding a snowman to a scene or altering an object's appearance. | Removes objects from video by inpainting masked regions using VOID (CogVideoX), with SAM3 text-guided segmentation and optional two-pass optical-flow refinement. | This subgraph uses Depth Anything 3 to predict spatially consistent geometry from any number of images or video frames, with or without known camera poses. It outputs depth maps, camera poses, and optionally 3D Gaussian parameters for novel view synthesis.
- official category: Video generation and editing  [spans multiple catalog categories: Image generation and editing (1), Video generation and editing (2)]

- member files:
    - image_edit_bernini_r - Edits a single image using a text prompt, leveraging Bernini-R's latent semantic planning for changes like object addition, removal, or style transfer. Ideal for creative edits requiring precise semantic understanding, such as adding a snowman to a scene or altering an object's appearance.
    - video_edit_bernini_r - This subgraph uses Depth Anything 3 to predict spatially consistent geometry from any number of images or video frames, with or without known camera poses. It outputs depth maps, camera poses, and optionally 3D Gaussian parameters for novel view synthesis.
    - video_inpaint_void - Removes objects from video by inpainting masked regions using VOID (CogVideoX), with SAM3 text-guided segmentation and optional two-pass optical-flow refinement.

- REQUIRED node roles (structural invariants):
    - CLIPTextEncode  (prompt text encoding) - all members (3/3), 2 required instances  **[PAIRED: 2x required]**
        - instance feeds ['conditioning_op'] / fed by ['clip_loader']  (x4)
        - instance feeds ['conditioning_op'] / fed by ['StringConcatenate', 'clip_loader']  (x2)
        - instance feeds ['SAM3_Detect'] / fed by ['model_loader']  (x1)
    - UNETLoader  (diffusion model / UNET loader) - all members (3/3), 2 required instances  **[PAIRED: 2x required]**
        - instance feeds ['ComfySwitchNode', 'guidance', 'lora_loader'] / fed by []  (x2)
        - instance feeds ['ComfySwitchNode', 'lora_loader'] / fed by []  (x2)
        - instance feeds ['guidance', 'guidance'] / fed by []  (x2)
    - BasicScheduler  (guider / sigma / scheduler) - all members (3/3)
    - VAEDecode  (latent -> pixel decode) - all members (3/3)
    - CLIPLoader  (text encoder / CLIP loader) - all members (3/3)
    - VAELoader  (VAE loader) - all members (3/3)
    - utility/plumbing (always present): ComfySwitchNode(5x), PrimitiveInt(5x), PrimitiveBoolean

- OPTIONAL node roles (variant, only in some members):
    - CFGGuider  (guider / sigma / scheduler) - 1/3 members
    - CreateVideo  (unclassified node role) - 2/3 members
    - LoraLoaderModelOnly  (LoRA / model patch loader) - 2/3 members
    - SamplerCustom  (diffusion sampler / denoiser) - 2/3 members
    - SamplerCustomAdvanced  (diffusion sampler / denoiser) - 1/3 members
    - VOIDSampler  (diffusion sampler / denoiser) - 1/3 members
    - BerniniConditioning  (conditioning combine / edit) - 2/3 members
    - CheckpointLoaderSimple  (diffusion model / UNET loader) - 1/3 members
    - CustomCombo  (unclassified node role) - 2/3 members
    - GetVideoComponents  (unclassified node role) - 2/3 members
    - ImageFromBatch  (unclassified node role) - 1/3 members
    - KSamplerSelect  (diffusion sampler / denoiser) - 2/3 members
    - MarkdownNote  (unclassified node role) - 2/3 members
    - MaskPreview  (unclassified node role) - 1/3 members
    - OpticalFlowLoader  (unclassified node role) - 1/3 members
    - RandomNoise  (unclassified node role) - 1/3 members
    - RegexExtract  (unclassified node role) - 2/3 members
    - SAM3_Detect  (unclassified node role) - 1/3 members
    - SplitSigmas  (guider / sigma / scheduler) - 2/3 members
    - TrimAudioDuration  (unclassified node role) - 1/3 members
    - VOIDInpaintConditioning  (conditioning combine / edit) - 1/3 members
    - VOIDWarpedNoise  (unclassified node role) - 1/3 members
    - VOIDWarpedNoiseSource  (unclassified node role) - 1/3 members
    - utility/plumbing (some members): ComfyMathExpression, PrimitiveFloat, GetImageSize, PreviewAny, PrimitiveStringMultiline, StringConcatenate, StringReplace

- connection patterns (role level):
    - PrimitiveBoolean -> ComfySwitchNode  [BOOLEAN]  (invariant)
    - clip_loader -> text_encode  [CLIP]  (invariant)
    - conditioning_op -> sampler  [LATENT]  (invariant)
    - guidance -> sampler  [SIGMAS]  (invariant)
    - model_loader -> guidance  [MODEL]  (invariant)
    - sampler -> sampler  [SAMPLER]  (invariant)
    - sampler -> vae_decode  [LATENT]  (invariant)
    - text_encode -> conditioning_op  [CONDITIONING]  (invariant)
    - vae_loader -> conditioning_op  [VAE]  (invariant)
    - vae_loader -> vae_decode  [VAE]  (invariant)
    - ComfySwitchNode -> guidance  [INT]  (2/3 members)
    - ComfySwitchNode -> sampler  [FLOAT]  (2/3 members)
    - ComfySwitchNode -> sampler  [MODEL]  (2/3 members)
    - CustomCombo -> PrimitiveInt  [INT]  (2/3 members)
    - GetVideoComponents -> CreateVideo  [FLOAT]  (2/3 members)
    - PreviewAny -> StringReplace  [STRING]  (2/3 members)
    - PrimitiveFloat -> ComfySwitchNode  [FLOAT]  (2/3 members)
    - PrimitiveInt -> ComfySwitchNode  [INT]  (2/3 members)
    - PrimitiveInt -> PreviewAny  [INT]  (2/3 members)
    - PrimitiveStringMultiline -> StringConcatenate  [STRING]  (2/3 members)
    - RegexExtract -> StringConcatenate  [STRING]  (2/3 members)
    - StringConcatenate -> text_encode  [STRING]  (2/3 members)
    - StringReplace -> RegexExtract  [STRING]  (2/3 members)
    - conditioning_op -> sampler  [CONDITIONING]  (2/3 members)
    - guidance -> guidance  [SIGMAS]  (2/3 members)
    - lora_loader -> ComfySwitchNode  [MODEL]  (2/3 members)
    - model_loader -> ComfySwitchNode  [MODEL]  (2/3 members)
    - model_loader -> lora_loader  [MODEL]  (2/3 members)
    - sampler -> sampler  [LATENT]  (2/3 members)
    - vae_decode -> CreateVideo  [IMAGE]  (2/3 members)

- boundary ports:
    - inputs:  COMBO(choice), COMBO(clip_name), COMBO(lora_name), COMBO(unet_name), COMBO(unet_name_1), COMBO(vae_name), IMAGE(reference_images.reference_image_0), IMAGE(reference_video), INT(height), INT(length), INT(noise_seed), INT(ref_max_size), INT(width)
    - outputs: (none)

- param variability: constant across members: ComfySwitchNode, VAEDecode; varies across members: BasicScheduler, CLIPLoader, PrimitiveBoolean, VAELoader
- unresolved nodes (not in object_info): BerniniConditioning, MarkdownNote

## `video_generation`  -  "Frame Interpolation"  -  3 member(s)  -  source: official
- user intent: media=video | task=video_generation | model families: n/a
- when to use: Use to generate a video.
- example requests: "build a video workflow"; "generate a video"
- description (catalog): Concatenates two videos end-to-end with optional resize, letterbox padding, and audio merge or drop. | Increases video frame rate by synthesizing intermediate frames with a frame interpolation model. | Stitches multiple video clips into a single sequential video file.
- official category: Video Tools  [pure: Video Tools (3)]

- member files:
    - frame_interpolation - Increases video frame rate by synthesizing intermediate frames with a frame interpolation model.
    - merge_videos - Concatenates two videos end-to-end with optional resize, letterbox padding, and audio merge or drop.
    - video_stitch - Stitches multiple video clips into a single sequential video file.

- REQUIRED node roles (structural invariants):
    - GetVideoComponents  (unclassified node role) - all members (3/3)
    - CreateVideo  (unclassified node role) - all members (3/3)

- OPTIONAL node roles (variant, only in some members):
    - AudioMerge  (unclassified node role) - 1/3 members
    - BatchImagesNode  (unclassified node role) - 1/3 members
    - EmptyAudio  (unclassified node role) - 1/3 members
    - FrameInterpolate  (unclassified node role) - 1/3 members
    - FrameInterpolationModelLoader  (diffusion model / UNET loader) - 1/3 members
    - ImageStitch  (unclassified node role) - 1/3 members
    - ResizeAndPadImage  (unclassified node role) - 1/3 members
    - ResizeImageMaskNode  (unclassified node role) - 1/3 members
    - utility/plumbing (some members): ComfyMathExpression, ComfySwitchNode, PrimitiveBoolean, GetImageSize, PrimitiveInt

- connection patterns (role level):
    - GetVideoComponents -> CreateVideo  [AUDIO]  (2/3 members)
    - GetVideoComponents -> CreateVideo  [FLOAT]  (2/3 members)
    - PrimitiveBoolean -> ComfySwitchNode  [BOOLEAN]  (2/3 members)

- boundary ports:
    - inputs:  VIDEO(video)
    - outputs: (none)

- param variability: constant across members: CreateVideo, GetVideoComponents

## `depth_estimation_depth_anything`  -  "Image Depth Estimation (Depth Anything 3)"  -  2 member(s)  -  source: official
- user intent: media=image | task=depth_estimation | model families: Depth Anything
- when to use: Use to estimate a depth map using Depth Anything.
- example requests: "build an image workflow using Depth Anything"; "estimate a depth map using Depth Anything"
- description (catalog): This subgraph processes a video input through Depth Anything 3 to produce temporally consistent depth maps for each frame, outputting a depth video. It is ideal for video content requiring spatial geometry estimation, such as 3D reconstruction, SLAM, or novel view synthesis from moving cameras. The model uses a plain transformer backbone trained with a depth-ray representation, supporting any number of views without requiring known camera poses. | This subgraph takes an input image and produces a depth map using the Depth Anything 3 model, which recovers spatially consistent geometry from any number of views. It is ideal for single or multi-view images, videos, and 3D scenes where accurate depth estimation is needed for tasks like SLAM, novel view synthesis, or spatial perception. The model uses a plain transformer backbone and supports both monocular and multi-view inputs without.
- official category: Conditioning & Preprocessors  [pure: Conditioning & Preprocessors (2)]

- member files:
    - image_depth_estimation_depth_anything_3 - This subgraph takes an input image and produces a depth map using the Depth Anything 3 model, which recovers spatially consistent geometry from any number of views. It is ideal for single or multi-view images, videos, and 3D scenes where accurate depth estimation is needed for tasks like SLAM, novel view synthesis, or spatial perception. The model uses a plain transformer backbone and supports both monocular and multi-view inputs without.
    - video_depth_estimation_depth_anything_3 - This subgraph processes a video input through Depth Anything 3 to produce temporally consistent depth maps for each frame, outputting a depth video. It is ideal for video content requiring spatial geometry estimation, such as 3D reconstruction, SLAM, or novel view synthesis from moving cameras. The model uses a plain transformer backbone trained with a depth-ray representation, supporting any number of views without requiring known camera poses.

- REQUIRED node roles (structural invariants):
    - DA3Inference  (unclassified node role) - all members (2/2)
    - DA3Render  (unclassified node role) - all members (2/2)
    - LoadDA3Model  (unclassified node role) - all members (2/2)

- OPTIONAL node roles (variant, only in some members):
    - GetVideoComponents  (unclassified node role) - 1/2 members
    - Video Slice  (unclassified node role) - 1/2 members

- connection patterns (role level):
    - DA3Inference -> DA3Render  [DA3_GEOMETRY]  (invariant)
    - LoadDA3Model -> DA3Inference  [DA3_MODEL]  (invariant)

- boundary ports:
    - inputs:  BOOLEAN(output.apply_sky_clip), COMBO(model_name), COMBO(output.normalization), COMBO(resize_method), COMFY_DYNAMICCOMBO_V3(output), FLOAT(duration), FLOAT(start_time), IMAGE(image), INT(resolution), VIDEO(video)
    - outputs: AUDIO(audio), FLOAT(fps), IMAGE(IMAGE)

- param variability: constant across members: DA3Render, LoadDA3Model; varies across members: DA3Inference
- unresolved nodes (not in object_info): DA3Inference, DA3Render, LoadDA3Model

## `depth_estimation_lotus`  -  "Image Depth Estimation (Lotus Depth)"  -  2 member(s)  -  source: official
- user intent: media=image | task=depth_estimation | model families: Lotus
- when to use: Use to estimate a depth map using Lotus.
- example requests: "build an image workflow using Lotus"; "estimate a depth map using Lotus"
- description (catalog): Estimates a monocular depth map from an input image using the Lotus depth estimation model. | Image to Depth Map (Lotus) blueprint
- official category: Conditioning & Preprocessors  [spans multiple catalog categories: Conditioning & Preprocessors (1), Image generation and editing (1)]

- member files:
    - image_depth_estimation_lotus_depth - Estimates a monocular depth map from an input image using the Lotus depth estimation model.
    - image_to_depth_map_lotus - Image to Depth Map (Lotus) blueprint

- REQUIRED node roles (structural invariants):
    - BasicGuider  (guider / sigma / scheduler) - all members (2/2)
    - BasicScheduler  (guider / sigma / scheduler) - all members (2/2)
    - DisableNoise  (unclassified node role) - all members (2/2)
    - ImageInvert  (unclassified node role) - all members (2/2)
    - KSamplerSelect  (diffusion sampler / denoiser) - all members (2/2)
    - LotusConditioning  (conditioning combine / edit) - all members (2/2)
    - SamplerCustomAdvanced  (diffusion sampler / denoiser) - all members (2/2)
    - SetFirstSigma  (unclassified node role) - all members (2/2)
    - UNETLoader  (diffusion model / UNET loader) - all members (2/2)
    - VAEDecode  (latent -> pixel decode) - all members (2/2)
    - VAEEncode  (pixel -> latent encode) - all members (2/2)
    - VAELoader  (VAE loader) - all members (2/2)

- OPTIONAL node roles (variant, only in some members):
    - (none)

- connection patterns (role level):
    - DisableNoise -> sampler  [NOISE]  (invariant)
    - SetFirstSigma -> sampler  [SIGMAS]  (invariant)
    - conditioning_op -> guidance  [CONDITIONING]  (invariant)
    - guidance -> SetFirstSigma  [SIGMAS]  (invariant)
    - guidance -> sampler  [GUIDER]  (invariant)
    - model_loader -> guidance  [MODEL]  (invariant)
    - sampler -> sampler  [SAMPLER]  (invariant)
    - sampler -> vae_decode  [LATENT]  (invariant)
    - vae_decode -> ImageInvert  [IMAGE]  (invariant)
    - vae_encode -> sampler  [LATENT]  (invariant)
    - vae_loader -> vae_decode  [VAE]  (invariant)
    - vae_loader -> vae_encode  [VAE]  (invariant)

- boundary ports:
    - inputs:  COMBO(unet_name), COMBO(vae_name), FLOAT(sigma), IMAGE(pixels)
    - outputs: IMAGE(IMAGE)

- param variability: constant across members: BasicScheduler, KSamplerSelect, SetFirstSigma, UNETLoader, VAELoader; varies across members: BasicGuider, DisableNoise, ImageInvert, LotusConditioning, SamplerCustomAdvanced, VAEDecode, VAEEncode

## `inpaint_qwen_image`  -  "Image Inpainting (Qwen-image)"  -  2 member(s)  -  source: official
- user intent: media=image | task=inpaint | model families: Qwen Image
- when to use: Use to inpaint masked regions of an image using Qwen Image.
- example requests: "build an image workflow using Qwen Image"; "inpaint masked regions of an image using Qwen Image"
- description (catalog): Inpaints masked regions using Qwen-Image, extending its multilingual text rendering to inpainting tasks. | Outpaints beyond image boundaries using Qwen-Image's outpainting capabilities.
- official category: Image generation and editing  [pure: Image generation and editing (2)]

- member files:
    - image_inpainting_qwen_image - Inpaints masked regions using Qwen-Image, extending its multilingual text rendering to inpainting tasks.
    - image_outpainting_qwen_image - Outpaints beyond image boundaries using Qwen-Image's outpainting capabilities.

- REQUIRED node roles (structural invariants):
    - CLIPTextEncode  (prompt text encoding) - all members (2/2), 2 required instances  **[PAIRED: 2x required]**
    - ImageToMask  (unclassified node role) - all members (2/2)
    - MaskToImage  (unclassified node role) - all members (2/2)
    - CLIPLoader  (text encoder / CLIP loader) - all members (2/2)
    - ControlNetInpaintingAliMamaApply  (controlnet / guidance conditioning) - all members (2/2)
    - ControlNetLoader  (controlnet / guidance conditioning) - all members (2/2)
    - GrowMask  (unclassified node role) - all members (2/2)
    - ImageBlur  (unclassified node role) - all members (2/2)
    - KSampler  (diffusion sampler / denoiser) - all members (2/2)
    - LoraLoaderModelOnly  (LoRA / model patch loader) - all members (2/2)
    - MaskPreview  (unclassified node role) - all members (2/2)
    - ModelSamplingAuraFlow  (unclassified node role) - all members (2/2)
    - UNETLoader  (diffusion model / UNET loader) - all members (2/2)
    - VAEDecode  (latent -> pixel decode) - all members (2/2)
    - VAEEncode  (pixel -> latent encode) - all members (2/2)
    - VAELoader  (VAE loader) - all members (2/2)

- OPTIONAL node roles (variant, only in some members):
    - ImageScaleToMaxDimension  (upscale / resize) - 1/2 members
    - FluxKontextImageScale  (upscale / resize) - 1/2 members
    - ImageCompositeMasked  (unclassified node role) - 1/2 members
    - ImagePadForOutpaint  (unclassified node role) - 1/2 members
    - MarkdownNote  (unclassified node role) - 1/2 members
    - Note  (unclassified node role) - 1/2 members
    - PreviewImage  (save / preview / combine output) - 1/2 members
    - SetLatentNoiseMask  (unclassified node role) - 1/2 members
    - utility/plumbing (some members): PrimitiveInt

- connection patterns (role level):
    - GrowMask -> MaskToImage  [MASK]  (invariant)
    - ImageBlur -> ImageToMask  [IMAGE]  (invariant)
    - ImageToMask -> MaskPreview  [MASK]  (invariant)
    - ImageToMask -> controlnet  [MASK]  (invariant)
    - MaskToImage -> ImageBlur  [IMAGE]  (invariant)
    - ModelSamplingAuraFlow -> sampler  [MODEL]  (invariant)
    - clip_loader -> text_encode  [CLIP]  (invariant)
    - controlnet -> controlnet  [CONTROL_NET]  (invariant)
    - controlnet -> sampler  [CONDITIONING]  (invariant)
    - lora_loader -> ModelSamplingAuraFlow  [MODEL]  (invariant)
    - model_loader -> lora_loader  [MODEL]  (invariant)
    - sampler -> vae_decode  [LATENT]  (invariant)
    - text_encode -> controlnet  [CONDITIONING]  (invariant)
    - upscale -> controlnet  [IMAGE]  (invariant)
    - upscale -> vae_encode  [IMAGE]  (invariant)
    - vae_loader -> controlnet  [VAE]  (invariant)
    - vae_loader -> vae_decode  [VAE]  (invariant)
    - vae_loader -> vae_encode  [VAE]  (invariant)

- boundary ports:
    - inputs:  COMBO(clip_name), COMBO(control_net_name), COMBO(lora_name), COMBO(unet_name), COMBO(vae_name), IMAGE(image), INT(bottom), INT(feathering), INT(left), INT(right), INT(top), MASK(mask), STRING(text)
    - outputs: IMAGE(IMAGE)

- param variability: constant across members: CLIPLoader, ControlNetInpaintingAliMamaApply, ControlNetLoader, ImageToMask, LoraLoaderModelOnly, MaskPreview, MaskToImage, ModelSamplingAuraFlow, UNETLoader, VAEDecode, VAEEncode, VAELoader; varies across members: GrowMask, ImageBlur, KSampler
- unresolved nodes (not in object_info): MarkdownNote, Note

## `segmentation_sam3`  -  "Image Segmentation (SAM3)"  -  2 member(s)  -  source: official
- user intent: media=image | task=segmentation | model families: SAM3
- when to use: Use to segment an image using SAM3.
- example requests: "build an image workflow using SAM3"; "segment an image using SAM3"
- description (catalog): Segments images into masks using Meta SAM3 from text prompts, points, or boxes. | Segments video into temporally consistent masks using Meta SAM3 from text or interactive prompts.
- official category: Conditioning & Preprocessors  [pure: Conditioning & Preprocessors (2)]

- member files:
    - image_segmentation_sam3 - Segments images into masks using Meta SAM3 from text prompts, points, or boxes.
    - video_segmentation_sam3 - Segments video into temporally consistent masks using Meta SAM3 from text or interactive prompts.

- REQUIRED node roles (structural invariants):
    - CLIPTextEncode  (prompt text encoding) - all members (2/2)
    - CheckpointLoaderSimple  (diffusion model / UNET loader) - all members (2/2)
    - SAM3_Detect  (unclassified node role) - all members (2/2)

- OPTIONAL node roles (variant, only in some members):
    - GetVideoComponents  (unclassified node role) - 1/2 members
    - Note  (unclassified node role) - 1/2 members

- connection patterns (role level):
    - model_loader -> SAM3_Detect  [MODEL]  (invariant)
    - model_loader -> text_encode  [CLIP]  (invariant)
    - text_encode -> SAM3_Detect  [CONDITIONING]  (invariant)

- boundary ports:
    - inputs:  BOOLEAN(individual_masks), BOUNDING_BOX(bboxes), COMBO(ckpt_name), FLOAT(threshold), IMAGE(image), INT(refine_iterations), STRING(negative_coords), STRING(positive_coords), STRING(text), VIDEO(video)
    - outputs: AUDIO(audio), BOUNDING_BOX(bboxes), FLOAT(fps), MASK(masks)

- param variability: constant across members: CLIPTextEncode, CheckpointLoaderSimple, SAM3_Detect
- unresolved nodes (not in object_info): Note

## `video_generation_mediapipe`  -  "Image Face Detection (Mediapipe)"  -  2 member(s)  -  source: official
- user intent: media=image | task=video_generation | model families: MediaPipe
- when to use: Use to generate a video using MediaPipe.
- example requests: "build an image workflow using MediaPipe"; "generate a video using MediaPipe"
- description (catalog): Detects facial landmarks from a video using MediaPipe, outputting landmark data, face bounding boxes, and an optional face-region mask. | Detects facial landmarks from an image using MediaPipe, outputting landmark data, face bounding boxes, and an optional face-region mask.
- official category: Conditioning & Preprocessors  [pure: Conditioning & Preprocessors (2)]

- member files:
    - image_face_detection_mediapipe - Detects facial landmarks from an image using MediaPipe, outputting landmark data, face bounding boxes, and an optional face-region mask.
    - video_face_detection_mediapipe - Detects facial landmarks from a video using MediaPipe, outputting landmark data, face bounding boxes, and an optional face-region mask.

- REQUIRED node roles (structural invariants):
    - LoadMediaPipeFaceLandmarker  (unclassified node role) - all members (2/2)
    - MediaPipeFaceLandmarker  (unclassified node role) - all members (2/2)
    - MediaPipeFaceMask  (unclassified node role) - all members (2/2)

- OPTIONAL node roles (variant, only in some members):
    - GetVideoComponents  (unclassified node role) - 1/2 members
    - Video Slice  (unclassified node role) - 1/2 members
    - utility/plumbing (some members): ComfySwitchNode

- connection patterns (role level):
    - LoadMediaPipeFaceLandmarker -> MediaPipeFaceLandmarker  [FACE_DETECTION_MODEL]  (invariant)
    - MediaPipeFaceLandmarker -> MediaPipeFaceMask  [FACE_LANDMARKS]  (invariant)

- boundary ports:
    - inputs:  BOOLEAN(regions.face_oval), BOOLEAN(regions.irises), BOOLEAN(regions.irises_1), BOOLEAN(regions.left_eye), BOOLEAN(regions.lips), BOOLEAN(regions.right_eye), BOOLEAN(regions.right_eye_1), BOOLEAN(switch), COMBO(detector_variant), COMBO(detector_variant_1), COMBO(model_name), FACE_LANDMARKER(face_landmarker), FACE_LANDMARKER(face_landmarker_1), FLOAT(duration), FLOAT(start_time), IMAGE(image), INT(num_faces), INT(num_faces_1), VIDEO(video)
    - outputs: BOUNDING_BOX(bboxes), BOUNDING_BOX(bboxes_1), FACE_LANDMARKS(face_landmarks), MASK(MASK_1)

- param variability: constant across members: LoadMediaPipeFaceLandmarker, MediaPipeFaceLandmarker, MediaPipeFaceMask

## `3d_generation_hunyuan3d`  -  "Image to 3D Model (Hunyuan3d 2.1)"  -  1 member(s)  -  source: official
- user intent: media=3d | task=3d_generation | model families: Hunyuan3D
- when to use: Use to generate a 3D model using Hunyuan3D.
- example requests: "build a 3d workflow using Hunyuan3D"; "generate a 3D model using Hunyuan3D"
- description (catalog): Generates 3D mesh models from a single input image using Hunyuan3D 2.0/2.1.
- official category: 3D  [pure: 3D (1)]

- member files:
    - image_to_model_hunyuan3d_2_1 - Generates 3D mesh models from a single input image using Hunyuan3D 2.0/2.1.

- REQUIRED node roles (structural invariants):
    - CLIPVisionEncode  (unclassified node role) - all members (1/1)
    - EmptyLatentHunyuan3Dv2  (empty latent / canvas) - all members (1/1)
    - Hunyuan3Dv2Conditioning  (conditioning combine / edit) - all members (1/1)
    - ImageOnlyCheckpointLoader  (diffusion model / UNET loader) - all members (1/1)
    - KSampler  (diffusion sampler / denoiser) - all members (1/1)
    - ModelSamplingAuraFlow  (unclassified node role) - all members (1/1)
    - VAEDecodeHunyuan3D  (latent -> pixel decode) - all members (1/1)
    - VoxelToMesh  (unclassified node role) - all members (1/1)

- OPTIONAL node roles (variant, only in some members):
    - (none)

- connection patterns (role level):
    - CLIPVisionEncode -> conditioning_op  [CLIP_VISION_OUTPUT]  (invariant)
    - ModelSamplingAuraFlow -> sampler  [MODEL]  (invariant)
    - conditioning_op -> sampler  [CONDITIONING]  (invariant)
    - latent_source -> sampler  [LATENT]  (invariant)
    - model_loader -> CLIPVisionEncode  [CLIP_VISION]  (invariant)
    - model_loader -> ModelSamplingAuraFlow  [MODEL]  (invariant)
    - model_loader -> vae_decode  [VAE]  (invariant)
    - sampler -> vae_decode  [LATENT]  (invariant)
    - vae_decode -> VoxelToMesh  [VOXEL]  (invariant)

- boundary ports:
    - inputs:  COMBO(ckpt_name), IMAGE(image)
    - outputs: MESH(MESH)

- param variability: single member - no cross-member variability to report

## `3d_generation_triposplat`  -  "Image to Gaussian Splat (TripoSplat)"  -  1 member(s)  -  source: official
- user intent: media=3d | task=3d_generation | model families: TripoSplat
- when to use: Use to generate a 3D model using TripoSplat.
- example requests: "build a 3d workflow using TripoSplat"; "generate a 3D model using TripoSplat"
- description (catalog): This subgraph takes a single 2D image as input and generates a variable number of 3D Gaussians (up to 262,144) as output, enabling high-quality 3D reconstruction. It is ideal for asset creation, AR/VR, game development, and simulation environments, handling diverse image styles from photos to illustrations.
- official category: 3D  [pure: 3D (1)]

- member files:
    - image_to_gaussian_splat_triposplat - This subgraph takes a single 2D image as input and generates a variable number of 3D Gaussians (up to 262,144) as output, enabling high-quality 3D reconstruction. It is ideal for asset creation, AR/VR, game development, and simulation environments, handling diverse image styles from photos to illustrations.

- REQUIRED node roles (structural invariants):
    - InvertMask  (unclassified node role) - all members (1/1), 2 required instances  **[PAIRED: 2x required]**
        - instance feeds ['ComfySwitchNode'] / fed by []  (x1)
        - instance feeds ['JoinImageWithAlpha'] / fed by ['RemoveBackground']  (x1)
    - VAELoader  (VAE loader) - all members (1/1), 2 required instances  **[PAIRED: 2x required]**
        - instance feeds ['TripoSplatSamplingPreview', 'vae_decode'] / fed by []  (x1)
        - instance feeds ['conditioning_op'] / fed by []  (x1)
    - CLIPVisionLoader  (unclassified node role) - all members (1/1)
    - JoinImageWithAlpha  (unclassified node role) - all members (1/1)
    - KSampler  (diffusion sampler / denoiser) - all members (1/1)
    - LoadBackgroundRemovalModel  (unclassified node role) - all members (1/1)
    - PreviewImage  (save / preview / combine output) - all members (1/1)
    - RemoveBackground  (unclassified node role) - all members (1/1)
    - TripoSplatConditioning  (conditioning combine / edit) - all members (1/1)
    - TripoSplatPreprocessImage  (unclassified node role) - all members (1/1)
    - TripoSplatSamplingPreview  (unclassified node role) - all members (1/1)
    - UNETLoader  (diffusion model / UNET loader) - all members (1/1)
    - VAEDecodeTripoSplat  (latent -> pixel decode) - all members (1/1)
    - utility/plumbing (always present): ComfySwitchNode(2x)

- OPTIONAL node roles (variant, only in some members):
    - (none)

- connection patterns (role level):
    - CLIPVisionLoader -> conditioning_op  [CLIP_VISION]  (invariant)
    - ComfySwitchNode -> TripoSplatPreprocessImage  [MASK]  (invariant)
    - ComfySwitchNode -> sampler  [MODEL]  (invariant)
    - InvertMask -> ComfySwitchNode  [MASK]  (invariant)
    - InvertMask -> JoinImageWithAlpha  [MASK]  (invariant)
    - LoadBackgroundRemovalModel -> RemoveBackground  [BACKGROUND_REMOVAL]  (invariant)
    - RemoveBackground -> ComfySwitchNode  [MASK]  (invariant)
    - RemoveBackground -> InvertMask  [MASK]  (invariant)
    - TripoSplatPreprocessImage -> conditioning_op  [IMAGE]  (invariant)
    - TripoSplatPreprocessImage -> save_output  [IMAGE]  (invariant)
    - TripoSplatSamplingPreview -> ComfySwitchNode  [MODEL]  (invariant)
    - conditioning_op -> sampler  [CONDITIONING]  (invariant)
    - conditioning_op -> sampler  [LATENT]  (invariant)
    - model_loader -> ComfySwitchNode  [MODEL]  (invariant)
    - model_loader -> TripoSplatSamplingPreview  [MODEL]  (invariant)
    - sampler -> vae_decode  [LATENT]  (invariant)
    - vae_loader -> TripoSplatSamplingPreview  [VAE]  (invariant)
    - vae_loader -> conditioning_op  [VAE]  (invariant)
    - vae_loader -> vae_decode  [VAE]  (invariant)

- boundary ports:
    - inputs:  BOOLEAN(switch), BOOLEAN(switch_1), COMBO(bg_removal_name), COMBO(clip_name), COMBO(unet_name), COMBO(vae_name), COMBO(vae_name_1), IMAGE(image), INT(num_gaussians_1), INT(seed), MASK(on_false)
    - outputs: SPLAT(splat)

- param variability: single member - no cross-member variability to report

## `controlnet_qwen_image`  -  1 member(s)  -  source: custom
- user intent: media=image | task=controlnet | model families: Qwen Image
- when to use: Use to generate an image guided by a control map (canny/depth/pose) using Qwen Image.
- example requests: "build an image workflow using Qwen Image"; "generate an image guided by a control map (canny/depth/pose) using Qwen Image"
- description (catalog): Local image editing via QWEN-Image-Edit-2511-Lightning. Up to 3 images (including optional depth/canny control inputs) -> 1 edited image output. Supports text-guided edits with optional structural control.

- member files:
    - qwen2511_imageEdit - Local image editing via QWEN-Image-Edit-2511-Lightning. Up to 3 images (including optional depth/canny control inputs) -> 1 edited image output. Supports text-guided edits with optional structural control.

- REQUIRED node roles (structural invariants):
    - CFGNorm  (unclassified node role) - all members (1/1), 2 required instances  **[PAIRED: 2x required]**
        - instance feeds ['ModelSamplingAuraFlow'] / fed by ['ModelSamplingAuraFlow']  (x1)
        - instance feeds ['sampler'] / fed by ['ModelSamplingAuraFlow']  (x1)
    - LoadImage  (image input / load) - all members (1/1), 2 required instances  **[PAIRED: 2x required]**
    - ModelSamplingAuraFlow  (unclassified node role) - all members (1/1), 2 required instances  **[PAIRED: 2x required]**
        - instance feeds ['CFGNorm'] / fed by ['CFGNorm']  (x1)
        - instance feeds ['CFGNorm'] / fed by ['lora_loader']  (x1)
    - TextEncodeQwenImageEditPlus  (prompt text encoding) - all members (1/1), 2 required instances  **[PAIRED: 2x required]**
        - instance feeds ['sampler'] / fed by ['image_loader', 'image_loader', 'lora_loader', 'vae_loader']  (x1)
        - instance feeds ['sampler'] / fed by ['lora_loader', 'vae_loader']  (x1)
    - CLIPLoader  (text encoder / CLIP loader) - all members (1/1)
    - EmptyLatentImage  (empty latent / canvas) - all members (1/1)
    - Image Load  (unclassified node role) - all members (1/1)
    - KSampler  (diffusion sampler / denoiser) - all members (1/1)
    - LoraLoader  (LoRA / model patch loader) - all members (1/1)
    - SaveImage  (save / preview / combine output) - all members (1/1)
    - UNETLoader  (diffusion model / UNET loader) - all members (1/1)
    - VAEDecode  (latent -> pixel decode) - all members (1/1)
    - VAELoader  (VAE loader) - all members (1/1)

- OPTIONAL node roles (variant, only in some members):
    - (none)

- connection patterns (role level):
    - CFGNorm -> ModelSamplingAuraFlow  [MODEL]  (invariant)
    - CFGNorm -> sampler  [MODEL]  (invariant)
    - ModelSamplingAuraFlow -> CFGNorm  [MODEL]  (invariant)
    - clip_loader -> lora_loader  [CLIP]  (invariant)
    - image_loader -> text_encode  [IMAGE]  (invariant)
    - latent_source -> sampler  [LATENT]  (invariant)
    - lora_loader -> ModelSamplingAuraFlow  [MODEL]  (invariant)
    - lora_loader -> text_encode  [CLIP]  (invariant)
    - model_loader -> lora_loader  [MODEL]  (invariant)
    - sampler -> vae_decode  [LATENT]  (invariant)
    - text_encode -> sampler  [CONDITIONING]  (invariant)
    - vae_decode -> save_output  [IMAGE]  (invariant)
    - vae_loader -> text_encode  [VAE]  (invariant)
    - vae_loader -> vae_decode  [VAE]  (invariant)

- boundary ports:
    - inputs:  IMAGE(image_loader), LATENT(latent_source)
    - outputs: IMAGE(save_output)

- param variability: single member - no cross-member variability to report
- custom nodes: Image Load

## `frame_interpolation_topaz`  -  1 member(s)  -  source: custom
- user intent: media=video | task=frame_interpolation | model families: Topaz
- when to use: Use to increase a video's frame rate via interpolation using Topaz.
- example requests: "build a video workflow using Topaz"; "increase a video's frame rate via interpolation using Topaz"
- description (catalog): API video upscaling via Topaz AI. 1 video -> 1 enhanced video output. Supports resolution upscaling (Starlight/Astra Fast model) and frame interpolation (apo-8 model).

- member files:
    - api_topaz_video_enhance - API video upscaling via Topaz AI. 1 video -> 1 enhanced video output. Supports resolution upscaling (Starlight/Astra Fast model) and frame interpolation (apo-8 model).

- REQUIRED node roles (structural invariants):
    - LoadVideo  (video input / load) - all members (1/1)
    - SaveVideo  (save / preview / combine output) - all members (1/1)
    - TopazVideoEnhance  (external API generation node) - all members (1/1)

- OPTIONAL node roles (variant, only in some members):
    - (none)

- connection patterns (role level):
    - api_node -> save_output  [VIDEO]  (invariant)
    - video_loader -> api_node  [VIDEO]  (invariant)

- boundary ports:
    - inputs:  VIDEO(video_loader)
    - outputs: VIDEO(save_output)

- param variability: single member - no cross-member variability to report

## `image`  -  "Color Curves"  -  1 member(s)  -  source: official
- user intent: media=image | task=None | model families: n/a
- when to use: Use to run a node graph.
- example requests: "build an image workflow"; "run a node graph"
- description (catalog): Fine-tunes tone and color with per-channel curve adjustments using a real-time GPU fragment shader.
- official category: Image Tools  [pure: Image Tools (1)]

- member files:
    - color_curves - Fine-tunes tone and color with per-channel curve adjustments using a real-time GPU fragment shader.

- REQUIRED node roles (structural invariants):
    - CurveEditor  (unclassified node role) - all members (1/1), 4 required instances  **[PAIRED: 4x required]**
    - GLSLShader  (unclassified node role) - all members (1/1)
    - ImageHistogram  (unclassified node role) - all members (1/1)

- OPTIONAL node roles (variant, only in some members):
    - (none)

- connection patterns (role level):
    - CurveEditor -> GLSLShader  [CURVE]  (invariant)
    - ImageHistogram -> CurveEditor  [HISTOGRAM]  (invariant)

- boundary ports:
    - inputs:  IMAGE(images.image0)
    - outputs: IMAGE(IMAGE0)

- param variability: single member - no cross-member variability to report

## `image_2`  -  "Select Per-Line Text by Index"  -  1 member(s)  -  source: official
- user intent: media=image | task=None | model families: n/a
- when to use: Use to run a node graph.
- example requests: "build an image workflow"; "run a node graph"
- description (catalog): Selects one line from multiline text by zero-based index for batch or list-driven prompt workflows.
- official category: Text Tools  [pure: Text Tools (1)]

- member files:
    - select_per_line_text_by_index - Selects one line from multiline text by zero-based index for batch or list-driven prompt workflows.

- REQUIRED node roles (structural invariants):
    - RegexExtract  (unclassified node role) - all members (1/1)
    - utility/plumbing (always present): PreviewAny, PrimitiveInt, StringReplace

- OPTIONAL node roles (variant, only in some members):
    - (none)

- connection patterns (role level):
    - PreviewAny -> StringReplace  [STRING]  (invariant)
    - PrimitiveInt -> PreviewAny  [INT]  (invariant)
    - StringReplace -> RegexExtract  [STRING]  (invariant)

- boundary ports:
    - inputs:  INT(index), STRING(text_per_line)
    - outputs: STRING(selected_line)

- param variability: single member - no cross-member variability to report

## `motion_prompt_gemini`  -  1 member(s)  -  source: custom
- user intent: media=text | task=motion_prompt | model families: Gemini
- when to use: Use to describe the motion in a video as text using Gemini.
- example requests: "build a text workflow using Gemini"; "describe the motion in a video as text using Gemini"
- description (catalog): [API] motion prompt generation via Gemini, analyses a video and output a desscription of the motion in it. 1 video input -> 1 output. Generates descriptive motion prompts for video generation.

- member files:
    - video_gemini_motionPromptGeneration - [API] motion prompt generation via Gemini, analyses a video and output a desscription of the motion in it. 1 video input -> 1 output. Generates descriptive motion prompts for video generation.

- REQUIRED node roles (structural invariants):
    - CreateVideo  (unclassified node role) - all members (1/1)
    - GeminiNode  (external API generation node) - all members (1/1)
    - GetVideoComponents  (unclassified node role) - all members (1/1)
    - ImageResizeKJv2  (unclassified node role) - all members (1/1)
    - LoadVideo  (video input / load) - all members (1/1)
    - easy saveText  (unclassified node role) - all members (1/1)

- OPTIONAL node roles (variant, only in some members):
    - (none)

- connection patterns (role level):
    - CreateVideo -> api_node  [VIDEO]  (invariant)
    - GetVideoComponents -> CreateVideo  [FLOAT]  (invariant)
    - GetVideoComponents -> ImageResizeKJv2  [IMAGE]  (invariant)
    - ImageResizeKJv2 -> CreateVideo  [IMAGE]  (invariant)
    - api_node -> easy saveText  [STRING]  (invariant)
    - video_loader -> GetVideoComponents  [VIDEO]  (invariant)

- boundary ports:
    - inputs:  VIDEO(video_loader)
    - outputs: STRING(easy saveText)

- param variability: single member - no cross-member variability to report
- custom nodes: ImageResizeKJv2, easy saveText

## `segmentation_birefnet`  -  "Remove Background (BiRefNet)"  -  1 member(s)  -  source: official
- user intent: media=image | task=segmentation | model families: BiRefNet
- when to use: Use to segment an image using BiRefNet.
- example requests: "build an image workflow using BiRefNet"; "segment an image using BiRefNet"
- description (catalog): Removes or replaces image backgrounds using BiRefNet segmentation and alpha compositing.
- official category: Image Tools  [pure: Image Tools (1)]

- member files:
    - remove_background_birefnet - Removes or replaces image backgrounds using BiRefNet segmentation and alpha compositing.

- REQUIRED node roles (structural invariants):
    - InvertMask  (unclassified node role) - all members (1/1)
    - JoinImageWithAlpha  (unclassified node role) - all members (1/1)
    - LoadBackgroundRemovalModel  (unclassified node role) - all members (1/1)
    - RemoveBackground  (unclassified node role) - all members (1/1)

- OPTIONAL node roles (variant, only in some members):
    - (none)

- connection patterns (role level):
    - InvertMask -> JoinImageWithAlpha  [MASK]  (invariant)
    - LoadBackgroundRemovalModel -> RemoveBackground  [BACKGROUND_REMOVAL]  (invariant)
    - RemoveBackground -> InvertMask  [MASK]  (invariant)

- boundary ports:
    - inputs:  COMBO(bg_removal_name), IMAGE(image)
    - outputs: IMAGE(IMAGE), MASK(mask)

- param variability: single member - no cross-member variability to report

## `style_transfer_gemini`  -  1 member(s)  -  source: custom
- user intent: media=image | task=style_transfer | model families: Gemini, Nano-Banana
- when to use: Use to transfer a style onto an image using Gemini, Nano-Banana.
- example requests: "build an image workflow using Gemini"; "build an image workflow using Nano-Banana"; "transfer a style onto an image using Gemini"
- description (catalog): Local style transfer FOR FULL BODY SHOTS via Nano-Banana Pro (Gemini). 1 video (layout reference) + 7 images (style + hero elements) -> 2 image outputs. Transfers the style reference onto the first video frame while integrating the look of hero element references.

- member files:
    - styletransfer_NanoBananaPro - Local style transfer FOR FULL BODY SHOTS via Nano-Banana Pro (Gemini). 1 video (layout reference) + 7 images (style + hero elements) -> 2 image outputs. Transfers the style reference onto the first video frame while integrating the look of hero element references.

- REQUIRED node roles (structural invariants):
    - VHS_LoadImagePath  (image input / load) - all members (1/1), 7 required instances  **[PAIRED: 7x required]**
    - BatchImagesNode  (unclassified node role) - all members (1/1), 2 required instances  **[PAIRED: 2x required]**
        - instance feeds ['BatchImagesNode'] / fed by ['image_loader', 'image_loader', 'image_loader', 'image_loader', 'image_loader']  (x1)
        - instance feeds ['api_node', 'bEpicReformat'] / fed by ['BatchImagesNode', 'VHS_SelectImages', 'image_loader', 'image_loader']  (x1)
    - GeminiImage2Node  (external API generation node) - all members (1/1)
    - GeminiNode  (external API generation node) - all members (1/1)
    - PreviewImage  (save / preview / combine output) - all members (1/1)
    - SaveImage  (save / preview / combine output) - all members (1/1)
    - VHS_LoadVideoPath  (video input / load) - all members (1/1)
    - VHS_SelectImages  (unclassified node role) - all members (1/1)
    - bEpicReformat  (unclassified node role) - all members (1/1)
    - utility/plumbing (always present): PreviewAny, PrimitiveStringMultiline

- OPTIONAL node roles (variant, only in some members):
    - (none)

- connection patterns (role level):
    - BatchImagesNode -> BatchImagesNode  [IMAGE]  (invariant)
    - BatchImagesNode -> api_node  [IMAGE]  (invariant)
    - BatchImagesNode -> bEpicReformat  [IMAGE]  (invariant)
    - PrimitiveStringMultiline -> api_node  [STRING]  (invariant)
    - VHS_SelectImages -> BatchImagesNode  [IMAGE]  (invariant)
    - api_node -> PreviewAny  [STRING]  (invariant)
    - api_node -> api_node  [STRING]  (invariant)
    - api_node -> save_output  [IMAGE]  (invariant)
    - bEpicReformat -> api_node  [IMAGE]  (invariant)
    - bEpicReformat -> save_output  [IMAGE]  (invariant)
    - image_loader -> BatchImagesNode  [IMAGE]  (invariant)
    - video_loader -> VHS_SelectImages  [IMAGE]  (invariant)

- boundary ports:
    - inputs:  IMAGE(image_loader), IMAGE(video_loader), STRING(PrimitiveStringMultiline)
    - outputs: IMAGE(save_output), STRING(PreviewAny)

- param variability: single member - no cross-member variability to report
- custom nodes: VHS_LoadImagePath, VHS_LoadVideoPath, VHS_SelectImages, bEpicReformat

## `text_to_image_ideogram`  -  1 member(s)  -  source: custom
- user intent: media=image | task=text_to_image | model families: Ideogram
- when to use: Use to generate an image from a text prompt using Ideogram.
- example requests: "build an image workflow using Ideogram"; "generate an image from a text prompt using Ideogram"
- description (synthesized): Generate an image from a text prompt using Ideogram. Structurally it applies a sequence of node operations. Boundary inputs: IMAGE; outputs: IMAGE.

- member files:
    - api_ideogram_v3_t2i

- REQUIRED node roles (structural invariants):
    - IdeogramV3  (external API generation node) - all members (1/1)
    - SaveImage  (save / preview / combine output) - all members (1/1)

- OPTIONAL node roles (variant, only in some members):
    - (none)

- connection patterns (role level):
    - api_node -> save_output  [IMAGE]  (invariant)

- boundary ports:
    - inputs:  IMAGE(api_node)
    - outputs: IMAGE(save_output)

- param variability: single member - no cross-member variability to report

## `text_to_image_lumina`  -  "Text to Image (NetaYume Lumina)"  -  1 member(s)  -  source: official
- user intent: media=image | task=text_to_image | model families: Lumina
- when to use: Use to generate an image from a text prompt using Lumina.
- example requests: "build an image workflow using Lumina"; "generate an image from a text prompt using Lumina"
- description (catalog): Generates images from text prompts using NetaYume Lumina, fine-tuned from Neta Lumina for anime-style and illustration generation.
- official category: Image generation and editing  [pure: Image generation and editing (1)]

- member files:
    - text_to_image_netayume_lumina - Generates images from text prompts using NetaYume Lumina, fine-tuned from Neta Lumina for anime-style and illustration generation.

- REQUIRED node roles (structural invariants):
    - CLIPTextEncode  (prompt text encoding) - all members (1/1), 2 required instances  **[PAIRED: 2x required]**
    - CheckpointLoaderSimple  (diffusion model / UNET loader) - all members (1/1)
    - EmptySD3LatentImage  (empty latent / canvas) - all members (1/1)
    - KSampler  (diffusion sampler / denoiser) - all members (1/1)
    - MarkdownNote  (unclassified node role) - all members (1/1)
    - ModelSamplingAuraFlow  (unclassified node role) - all members (1/1)
    - VAEDecode  (latent -> pixel decode) - all members (1/1)
    - utility/plumbing (always present): PrimitiveStringMultiline(4x), StringConcatenate(2x)

- OPTIONAL node roles (variant, only in some members):
    - (none)

- connection patterns (role level):
    - ModelSamplingAuraFlow -> sampler  [MODEL]  (invariant)
    - PrimitiveStringMultiline -> StringConcatenate  [STRING]  (invariant)
    - StringConcatenate -> text_encode  [STRING]  (invariant)
    - latent_source -> sampler  [LATENT]  (invariant)
    - model_loader -> ModelSamplingAuraFlow  [MODEL]  (invariant)
    - model_loader -> text_encode  [CLIP]  (invariant)
    - model_loader -> vae_decode  [VAE]  (invariant)
    - sampler -> vae_decode  [LATENT]  (invariant)
    - text_encode -> sampler  [CONDITIONING]  (invariant)

- boundary ports:
    - inputs:  COMBO(ckpt_name), INT(height), INT(seed), INT(width), STRING(value)
    - outputs: IMAGE(IMAGE)

- param variability: single member - no cross-member variability to report
- unresolved nodes (not in object_info): MarkdownNote

## `upscale`  -  "Video Upscale(GAN x4)"  -  1 member(s)  -  source: official
- user intent: media=video | task=upscale | model families: n/a
- when to use: Use to upscale / enhance a video.
- example requests: "build a video workflow"; "upscale / enhance a video"
- description (catalog): Upscales video to 4× resolution using a GAN-based upscaling model.
- official category: Video generation and editing  [pure: Video generation and editing (1)]

- member files:
    - video_upscale_gan_x4 - Upscales video to 4× resolution using a GAN-based upscaling model.

- REQUIRED node roles (structural invariants):
    - CreateVideo  (unclassified node role) - all members (1/1)
    - GetVideoComponents  (unclassified node role) - all members (1/1)
    - ImageUpscaleWithModel  (upscale / resize) - all members (1/1)
    - UpscaleModelLoader  (diffusion model / UNET loader) - all members (1/1)

- OPTIONAL node roles (variant, only in some members):
    - (none)

- connection patterns (role level):
    - GetVideoComponents -> CreateVideo  [AUDIO]  (invariant)
    - GetVideoComponents -> CreateVideo  [FLOAT]  (invariant)
    - GetVideoComponents -> upscale  [IMAGE]  (invariant)
    - model_loader -> upscale  [UPSCALE_MODEL]  (invariant)
    - upscale -> CreateVideo  [IMAGE]  (invariant)

- boundary ports:
    - inputs:  COMBO(model_name), VIDEO(video)
    - outputs: VIDEO(VIDEO)

- param variability: single member - no cross-member variability to report
