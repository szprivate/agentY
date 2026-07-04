# Workflow recipe database  (task -> model -> node clusters)

- Tasks: 27 | task+model recipes: 72
- Self-contained: every recipe has user_intent + description + node clusters. No human annotation step.

# Image Tools  (`image_tools`)  -  18 workflow(s), 2 model(s)

## Image Tools / Generic  (`image_tools__generic`)  -  17 workflow(s)  -  source: official
- execution: local
- when to use: Use to generate an image.
- example request: "build an image workflow"
- description: Adds a glow/bloom effect around bright image areas via GPU fragment shader. | Adds lens-style chromatic aberration (color fringing) using a real-time GPU fragment shader. | Adds procedural film grain texture for a cinematic look via GPU fragment shader. | Adjusts black point, white point, and gamma for tonal range control via GPU shader. | Adjusts hue, saturation, and lightness of an image using a real-time GPU fragment shader. | Adjusts image brightness and contrast using a real-time GPU fragment shader. | Adjusts saturation, temperature, tint, and vibrance using a real-time GPU fragment shader. | Applies Gaussian, Box, or Radial blur to soften images and create stylized depth or motion effects. | Applies bilateral (edge-preserving) blur to soften images while retaining detail. | Balances colors across shadows, midtones, and highlights using a real-time GPU fragment shader. | Enhances edge contrast via unsharp masking for a sharper image appearance. | Fine-tunes tone and color with per-channel curve adjustments using a real-time GPU fragment shader. | Manipulates individual RGBA channels for masking, compositing, and channel effects. | Sharpens image details using a GPU fragment shader for enhanced clarity. | Splits an image into a 2×2 grid of four equal tiles. | Splits an image into a 3×3 grid of nine equal tiles. | Splits an image into a configurable columns×rows grid of equal tiles for tiled generation or processing.
- member workflows:
    - brightness_and_contrast
    - chromatic_aberration
    - color_adjustment
    - color_balance
    - color_curves
    - crop_images_2x2
    - crop_images_3x3
    - edge_preserving_blur
    - film_grain
    - glow
    - hue_and_saturation
    - image_blur
    - image_channels
    - image_levels
    - sharpen
    - split_image_grid_to_tiles
    - unsharp_mask
- node clusters (required structure):
    - (none resolved)
- optional roles: ImageCropV2, CurveEditor, CustomCombo, BatchImagesNode, ColorToRGBInt, GLSLShader, ImageHistogram, SplitImageToTileList

## Image Tools / BiRefNet  (`image_tools__birefnet`)  -  1 workflow(s)  -  source: official
- execution: local
- when to use: Use to remove the background from an image using BiRefNet.
- example request: "build an image workflow using BiRefNet"
- description: Removes or replaces image backgrounds using BiRefNet segmentation and alpha compositing.
- member workflows:
    - remove_background_birefnet
- node clusters (required structure):
    - other operations: InvertMask, JoinImageWithAlpha, LoadBackgroundRemovalModel, RemoveBackground


# Text to Image  (`text_to_image`)  -  16 workflow(s), 10 model(s)

## Text to Image / Qwen Image  (`text_to_image__qwen_image`)  -  4 workflow(s)  -  source: official
- execution: local
- when to use: Use to generate an image from a text prompt using Qwen Image, Z-Image.
- example request: "build an image workflow using Qwen Image"
- description: Generates images from text prompts using Qwen-Image, Alibaba's 20B MMDiT model with excellent multilingual text rendering. | Generates images from text prompts using Qwen-Image-2512, with enhanced human realism and finer natural detail over the base version. | Generates images from text prompts using Z-Image base weights with Qwen3 text encoder and bundled VAE. | Generates images from text prompts using Z-Image-Turbo defaults with Qwen3 text encoder and VAE.
- member workflows:
    - text_to_image
    - text_to_image_qwen_image
    - text_to_image_qwen_image_2512
    - text_to_image_z_image_base
- node clusters (required structure):
    - model loading: CLIPLoader, UNETLoader, VAELoader
    - conditioning: CLIPTextEncode
    - latent / canvas: EmptySD3LatentImage
    - sampling: KSampler
    - decoding: VAEDecode
    - other operations: ModelSamplingAuraFlow
- optional roles: ConditioningZeroOut, LoraLoaderModelOnly, MarkdownNote
- unresolved nodes: MarkdownNote

## Text to Image / Anima  (`text_to_image__anima`)  -  2 workflow(s)  -  source: official
- execution: local
- when to use: Use to generate an image from a text prompt using Anima.
- example request: "build an image workflow using Anima"
- description: This subgraph converts text prompts into non-photorealistic illustrations using a 2-billion-parameter model optimized for anime and artistic styles. It is ideal for generating concept art, character designs, or stylized illustrations where photorealism is not required. The model excels with anime and artistic content but performs poorly on realistic subjects. | This subgraph generates non-photorealistic illustrations from text prompts using a 2-billion-parameter model optimized for anime concepts, characters, and styles. It is ideal for creating artistic images, concept art, or stylized illustrations where photorealism is not required. The model excels with anime and artistic content but performs poorly on realistic subjects.
- member workflows:
    - text_to_image_anima
    - text_to_image_anima_base_1_0
- node clusters (required structure):
    - model loading: CLIPLoader, UNETLoader, VAELoader
    - conditioning: CLIPTextEncode (x2)
    - latent / canvas: EmptyLatentImage
    - sampling: KSampler
    - decoding: VAEDecode
- paired/multiple required: CLIPTextEncode x2

## Text to Image / ERNIE  (`text_to_image__ernie`)  -  2 workflow(s)  -  source: official
- execution: local
- when to use: Use to generate an image from a text prompt using ERNIE.
- example request: "build an image workflow using ERNIE"
- description: Faster ERNIE Image Turbo variant (~8B DiT, distilled for fewer sampling steps): same strengths in Chinese/English on-image text and layout-heavy graphics as the base ERNIE Image lineup, with bundled encoders and VAE. | Generates images from text prompts using Baidu's open ERNIE Image (~8B DiT): bilingual in-image typography and layouts (posters, infographics, multi-panel compositions) alongside general scenes, with bundled encoders and VAE.
- member workflows:
    - text_to_image_ernie_image
    - text_to_image_ernie_image_turbo
- node clusters (required structure):
    - model loading: CLIPLoader (x2), UNETLoader, VAELoader
    - conditioning: CLIPTextEncode
    - sampling: KSampler
    - decoding: VAEDecode
    - other operations: ComfySwitchNode, EmptyFlux2LatentImage, PreviewAny (x3), PrimitiveBoolean, PrimitiveStringMultiline, StringReplace (x3), TextGenerate
- paired/multiple required: CLIPLoader x2
- optional roles: ConditioningZeroOut

## Text to Image / Z-Image  (`text_to_image__z_image`)  -  2 workflow(s)  -  source: mixed
- execution: local
- when to use: Use to generate an image from a text prompt using Z-Image.
- example request: "build an image workflow using Z-Image"
- description: Generates images from text prompts using Z-Image-Turbo, Alibaba's distilled 6B DiT model. | [Local] text-to-image via Z-Image-Turbo. 1 text input -> 1 image output. High-speed image generation from text prompts.
- member workflows:
    - image_z_image_turbo
    - text_to_image_z_image_turbo
- node clusters (required structure):
    - model loading: CLIPLoader, UNETLoader, VAELoader
    - conditioning: CLIPTextEncode, ConditioningZeroOut
    - latent / canvas: EmptySD3LatentImage
    - sampling: KSampler
    - decoding: VAEDecode
    - other operations: ModelSamplingAuraFlow
- optional roles: SaveImage

## Text to Image / Flux  (`text_to_image__flux`)  -  1 workflow(s)  -  source: official
- execution: local
- when to use: Use to generate an image from a text prompt using Flux.
- example request: "build an image workflow using Flux"
- description: Generates images from prompts using FLUX.1 [dev]: a 12B rectified-flow MMDiT with dual CLIP plus T5-XXL text encoders and guidance-distilled sampling for sharp prompt following versus classic DDPM diffusion.
- member workflows:
    - text_to_image_flux_1_dev
- node clusters (required structure):
    - model loading: DualCLIPLoader, UNETLoader, VAELoader
    - conditioning: CLIPTextEncode, ConditioningZeroOut
    - latent / canvas: EmptySD3LatentImage
    - sampling: KSampler
    - decoding: VAEDecode

## Text to Image / Flux 2  (`text_to_image__flux_2`)  -  1 workflow(s)  -  source: official
- execution: local
- when to use: Use to generate an image from a text prompt using Flux 2.
- example request: "build an image workflow using Flux 2"
- description: Generates images from prompts using FLUX.2 [dev]: a newer 32B rectified-flow stack with distilled guidance plus a stronger long-context multimodal encoder for complex scenes, sharper typography/UI text, anatomy, lighting, and high-resolution latent decoding.
- member workflows:
    - text_to_image_flux_2_dev
- node clusters (required structure):
    - model loading: CLIPLoader, LoraLoaderModelOnly, UNETLoader, VAELoader
    - conditioning: BasicGuider, CLIPTextEncode, FluxGuidance
    - sampling: KSamplerSelect, SamplerCustomAdvanced
    - decoding: VAEDecode
    - other operations: ComfySwitchNode (x2), EmptyFlux2LatentImage, Flux2Scheduler, PrimitiveBoolean, PrimitiveInt (x2), RandomNoise

## Text to Image / Flux 2 Klein  (`text_to_image__flux_2_klein`)  -  1 workflow(s)  -  source: custom
- execution: local
- when to use: Use to generate an image from a text prompt using Flux 2 Klein.
- example request: "build an image workflow using Flux 2 Klein"
- description: Generate an image from a text prompt using Flux 2 Klein. Structurally it loads a diffusion model; uses a VAE; encodes a text prompt; runs a diffusion sampler; decodes the latent to pixels. Boundary inputs: STRING; outputs: IMAGE.
- member workflows:
    - image_flux2_klein_text_to_image
- node clusters (required structure):
    - model loading: CLIPLoader, UNETLoader, VAELoader
    - conditioning: CFGGuider, CLIPTextEncode (x2)
    - sampling: KSamplerSelect, SamplerCustomAdvanced
    - decoding: VAEDecode
    - output: SaveImage
    - other operations: EmptyFlux2LatentImage, Flux2Scheduler, PrimitiveInt (x2), PrimitiveStringMultiline, RandomNoise
- paired/multiple required: CLIPTextEncode x2

## Text to Image / Flux Krea  (`text_to_image__flux_krea`)  -  1 workflow(s)  -  source: official
- execution: local
- when to use: Use to generate an image from a text prompt using Flux Krea.
- example request: "build an image workflow using Flux Krea"
- description: FLUX.1 Krea [dev] (Black Forest Labs × Krea): open-weight 12B rectified-flow text-to-image drop-in alongside FLUX.1 [dev], tuned away from overcooked saturation toward more natural diversity in people, realism, and style while keeping ecosystem compatibility.
- member workflows:
    - text_to_image_flux_1_krea_dev
- node clusters (required structure):
    - model loading: DualCLIPLoader, UNETLoader, VAELoader
    - conditioning: CLIPTextEncode, ConditioningZeroOut
    - latent / canvas: EmptySD3LatentImage
    - sampling: KSampler
    - decoding: VAEDecode

## Text to Image / Ideogram  (`text_to_image__ideogram`)  -  1 workflow(s)  -  source: official
- execution: local
- when to use: Use to generate an image from a text prompt using Ideogram.
- example request: "build an image workflow using Ideogram"
- description: This subgraph generates images using Ideogram v4, accepting plain text or structured JSON prompts for precise layout and style control. It suits detailed illustrations, concept art, or marketing visuals needing predictable composition and color palettes. The model uses flow-matching with asymmetric guidance, so no negative prompt is needed, but JSON prompts yield the best results.
- member workflows:
    - text_to_image_ideogram_v4
- node clusters (required structure):
    - model loading: CLIPLoader, UNETLoader (x2), VAELoader
    - conditioning: CLIPTextEncode, ConditioningZeroOut, DualModelGuider
    - sampling: KSamplerSelect, SamplerCustomAdvanced
    - decoding: VAEDecode
    - other operations: CFGOverride, ComfyMathExpression (x2), ComfyNumberConvert (x3), CustomCombo, EmptyFlux2LatentImage, Ideogram4Scheduler, JsonExtractString (x4), PrimitiveInt (x2), RandomNoise, StringReplace
- paired/multiple required: UNETLoader x2

## Text to Image / Lumina  (`text_to_image__lumina`)  -  1 workflow(s)  -  source: official
- execution: local
- when to use: Use to generate an image from a text prompt using Lumina.
- example request: "build an image workflow using Lumina"
- description: Generates images from text prompts using NetaYume Lumina, fine-tuned from Neta Lumina for anime-style and illustration generation.
- member workflows:
    - text_to_image_netayume_lumina
- node clusters (required structure):
    - model loading: CheckpointLoaderSimple
    - conditioning: CLIPTextEncode (x2)
    - latent / canvas: EmptySD3LatentImage
    - sampling: KSampler
    - decoding: VAEDecode
    - other operations: MarkdownNote, ModelSamplingAuraFlow, PrimitiveStringMultiline (x4), StringConcatenate (x2)
- paired/multiple required: CLIPTextEncode x2
- unresolved nodes: MarkdownNote


# Preprocessors / Estimation  (`preprocessors_estimation`)  -  13 workflow(s), 6 model(s)

## Preprocessors / Estimation / SDPose  (`preprocessors_estimation__sdpose`)  -  3 workflow(s)  -  source: official
- execution: local
- when to use: Use to estimate a pose map using SDPose.
- example request: "build an image workflow using SDPose"
- description: Detects multiple people in an image and outputs per-person pose keypoints, skeleton renders, and bounding boxes using SDPose. | Extracts human pose keypoints and stick-figure visuals from an image using SDPose-OOD, with optional bounding-box input per subject. | Extracts multi-person pose keypoints and skeleton frame sequences from video using SDPose with built-in person detection.
- member workflows:
    - image_to_pose_map_sdpose_multi_person
    - image_to_pose_map_sdpose_ood
    - video_to_pose_map_sdpose_multi_person
- node clusters (required structure):
    - model loading: CheckpointLoaderSimple
    - other operations: ResizeImageMaskNode, SDPoseDrawKeypoints, SDPoseKeypointExtractor
- optional roles: GetVideoComponents, RTDETR_detect, UNETLoader

## Preprocessors / Estimation / Depth Anything  (`preprocessors_estimation__depth_anything`)  -  2 workflow(s)  -  source: official
- execution: local
- when to use: Use to estimate a depth map using Depth Anything.
- example request: "build an image workflow using Depth Anything"
- description: This subgraph processes a video input through Depth Anything 3 to produce temporally consistent depth maps for each frame, outputting a depth video. It is ideal for video content requiring spatial geometry estimation, such as 3D reconstruction, SLAM, or novel view synthesis from moving cameras. The model uses a plain transformer backbone trained with a depth-ray representation, supporting any number of views without requiring known camera poses. | This subgraph takes an input image and produces a depth map using the Depth Anything 3 model, which recovers spatially consistent geometry from any number of views. It is ideal for single or multi-view images, videos, and 3D scenes where accurate depth estimation is needed for tasks like SLAM, novel view synthesis, or spatial perception. The model uses a plain transformer backbone and supports both monocular and multi-view inputs without.
- member workflows:
    - image_depth_estimation_depth_anything_3
    - video_depth_estimation_depth_anything_3
- node clusters (required structure):
    - other operations: DA3Inference, DA3Render, LoadDA3Model
- optional roles: GetVideoComponents, Video Slice
- unresolved nodes: DA3Inference, DA3Render, LoadDA3Model

## Preprocessors / Estimation / Lotus  (`preprocessors_estimation__lotus`)  -  2 workflow(s)  -  source: official
- execution: local
- when to use: Use to estimate a depth map using Lotus.
- example request: "build an image workflow using Lotus"
- description: Estimates a monocular depth map from an input image using the Lotus depth estimation model. | Image to Depth Map (Lotus) blueprint
- member workflows:
    - image_depth_estimation_lotus_depth
    - image_to_depth_map_lotus
- node clusters (required structure):
    - model loading: UNETLoader, VAELoader
    - conditioning: BasicGuider, BasicScheduler, LotusConditioning
    - latent / canvas: VAEEncode
    - sampling: KSamplerSelect, SamplerCustomAdvanced
    - decoding: VAEDecode
    - other operations: DisableNoise, ImageInvert, SetFirstSigma

## Preprocessors / Estimation / MediaPipe  (`preprocessors_estimation__mediapipe`)  -  2 workflow(s)  -  source: official
- execution: local
- when to use: Use to detect facial landmarks using MediaPipe.
- example request: "build an image workflow using MediaPipe"
- description: Detects facial landmarks from a video using MediaPipe, outputting landmark data, face bounding boxes, and an optional face-region mask. | Detects facial landmarks from an image using MediaPipe, outputting landmark data, face bounding boxes, and an optional face-region mask.
- member workflows:
    - image_face_detection_mediapipe
    - video_face_detection_mediapipe
- node clusters (required structure):
    - other operations: LoadMediaPipeFaceLandmarker, MediaPipeFaceLandmarker, MediaPipeFaceMask
- optional roles: GetVideoComponents, Video Slice

## Preprocessors / Estimation / MoGe  (`preprocessors_estimation__moge`)  -  2 workflow(s)  -  source: official
- execution: local
- when to use: Use to estimate a depth map using MoGe.
- example request: "build an image workflow using MoGe"
- description: Estimates monocular depth from an input image using MoGe, outputting both raw and colorized depth maps plus a mask. | Estimates monocular depth from an input video using MoGe, outputting both raw and colorized depth maps plus a mask.
- member workflows:
    - image_depth_estimation_moge
    - video_depth_estimation_moge
- node clusters (required structure):
    - other operations: ComfyMathExpression, ComfySwitchNode (x2), GetImageSize, ImageToMask, LoadMoGeModel, MoGeInference, MoGeRender (x3), ResizeImagesByLongerEdge
- paired/multiple required: MoGeRender x3
- optional roles: GetVideoComponents

## Preprocessors / Estimation / SAM3  (`preprocessors_estimation__sam3`)  -  2 workflow(s)  -  source: official
- execution: local
- when to use: Use to segment an image using SAM3.
- example request: "build an image workflow using SAM3"
- description: Segments images into masks using Meta SAM3 from text prompts, points, or boxes. | Segments video into temporally consistent masks using Meta SAM3 from text or interactive prompts.
- member workflows:
    - image_segmentation_sam3
    - video_segmentation_sam3
- node clusters (required structure):
    - model loading: CheckpointLoaderSimple
    - conditioning: CLIPTextEncode
    - other operations: SAM3_Detect
- optional roles: GetVideoComponents, Note
- unresolved nodes: Note


# Image Edit  (`image_edit`)  -  11 workflow(s), 6 model(s)

## Image Edit / Qwen Image  (`image_edit__qwen_image`)  -  5 workflow(s)  -  source: mixed
- execution: local
- when to use: Use to edit an existing image using Qwen Image.
- example request: "build an image workflow using Qwen Image"
- description: Decomposes an image into variable-resolution RGBA layers for independent editing using Qwen-Image-Layered. | Edits images from text instructions using Qwen-Image-Edit-2509 with optional Lightning LoRA for few-step sampling. | Edits images via text instructions using Qwen-Image-Edit-2511 with improved character consistency and integrated LoRA. | Image Edit blueprint | Local image editing via QWEN-Image-Edit-2511-Lightning. Up to 3 images (including optional depth/canny control inputs) -> 1 edited image output. Supports text-guided edits with optional structural control.
- member workflows:
    - image_edit
    - image_edit_qwen_2509
    - image_edit_qwen_2511
    - image_to_layers_qwen_image_layered
    - qwen2511_imageEdit
- node clusters (required structure):
    - model loading: CLIPLoader, UNETLoader, VAELoader
    - sampling: KSampler
    - decoding: VAEDecode
    - other operations: ModelSamplingAuraFlow
- optional roles: CFGNorm, CLIPTextEncode, FluxKontextMultiReferenceLatentMethod, LoadImage, ReferenceLatent, TextEncodeQwenImageEditPlus, EmptyLatentImage, EmptyQwenImageLayeredLatentImage, FluxKontextImageScale, Image Load, LatentCutToBatch, LoraLoader
- unresolved nodes: MarkdownNote, Note

## Image Edit / Flux 2 Klein  (`image_edit__flux_2_klein`)  -  2 workflow(s)  -  source: mixed
- execution: local
- when to use: Use to edit an existing image using Flux 2 Klein.
- example request: "build an image workflow using Flux 2 Klein"
- description: Edits an input image via text instructions using FLUX.2 [klein] 4B. | [Local] image editing via Flux. 1 image input -> 1 image output. Performs image editing using the Flux 2 Klein distilled model.
- member workflows:
    - image_edit_flux_2_klein_4b
    - image_flux2_klein_image_edit_9b_distilled
- node clusters (required structure):
    - model loading: CLIPLoader, UNETLoader, VAELoader
    - conditioning: CFGGuider, CLIPTextEncode
    - latent / canvas: VAEEncode
    - sampling: KSamplerSelect, SamplerCustomAdvanced
    - decoding: VAEDecode
    - other operations: EmptyFlux2LatentImage, Flux2Scheduler, GetImageSize, ImageScaleToTotalPixels, RandomNoise, ReferenceLatent (x2)
- paired/multiple required: ReferenceLatent x2
- optional roles: ConditioningZeroOut, LoadImage, SaveImage

## Image Edit / Bernini  (`image_edit__bernini`)  -  1 workflow(s)  -  source: official
- execution: local
- when to use: Use to edit an existing image using Bernini.
- example request: "build an image workflow using Bernini"
- description: Edits a single image using a text prompt, leveraging Bernini-R's latent semantic planning for changes like object addition, removal, or style transfer. Ideal for creative edits requiring precise semantic understanding, such as adding a snowman to a scene or altering an object's appearance.
- member workflows:
    - image_edit_bernini_r
- node clusters (required structure):
    - model loading: CLIPLoader, LoraLoaderModelOnly (x2), UNETLoader (x2), VAELoader
    - conditioning: BasicScheduler, BerniniConditioning, CLIPTextEncode (x2), SplitSigmas
    - sampling: KSamplerSelect, SamplerCustom (x2)
    - decoding: VAEDecode
    - other operations: ComfySwitchNode (x5), CustomCombo, MarkdownNote, PreviewAny, PrimitiveBoolean, PrimitiveFloat (x2), PrimitiveInt (x5), PrimitiveStringMultiline, RegexExtract, StringConcatenate, StringReplace
- paired/multiple required: CLIPTextEncode x2, LoraLoaderModelOnly x2, SamplerCustom x2, UNETLoader x2
- unresolved nodes: BerniniConditioning, MarkdownNote

## Image Edit / FireRed  (`image_edit__firered`)  -  1 workflow(s)  -  source: official
- execution: local
- when to use: Use to edit an existing image using FireRed.
- example request: "build an image workflow using FireRed"
- description: Edits images via text instructions using FireRed Image Edit 1.1, a diffusion-based instruction-following editing model.
- member workflows:
    - image_edit_firered_image_edit_1_1
- node clusters (required structure):
    - model loading: CLIPLoader, LoraLoaderModelOnly, UNETLoader, VAELoader
    - conditioning: TextEncodeQwenImageEditPlus (x2)
    - latent / canvas: VAEEncode
    - sampling: KSampler
    - decoding: VAEDecode
    - other operations: CFGNorm, ComfySwitchNode (x3), ModelSamplingAuraFlow, PrimitiveBoolean, PrimitiveFloat (x2), PrimitiveInt (x2), ResizeImageMaskNode
- paired/multiple required: TextEncodeQwenImageEditPlus x2

## Image Edit / Flux 2  (`image_edit__flux_2`)  -  1 workflow(s)  -  source: official
- execution: local
- when to use: Use to edit an existing image using Flux 2.
- example request: "build an image workflow using Flux 2"
- description: Edits an image from text instructions using Flux.2 [dev], with guidance, schedulers, and optional Turbo LoRAs.
- member workflows:
    - image_edit_flux_2_dev
- node clusters (required structure):
    - model loading: CLIPLoader, LoraLoaderModelOnly, UNETLoader, VAELoader
    - conditioning: BasicGuider, CLIPTextEncode, FluxGuidance
    - latent / canvas: VAEEncode
    - sampling: KSamplerSelect, SamplerCustomAdvanced
    - decoding: VAEDecode
    - other operations: ComfySwitchNode (x2), EmptyFlux2LatentImage, Flux2Scheduler, GetImageSize, PrimitiveBoolean, PrimitiveInt (x2), RandomNoise, ReferenceLatent

## Image Edit / LongCat  (`image_edit__longcat`)  -  1 workflow(s)  -  source: official
- execution: local
- when to use: Use to edit an existing image using LongCat.
- example request: "build an image workflow using LongCat"
- description: Edits images via text instructions using LongCat Image Edit, an instruction-following image editing diffusion model.
- member workflows:
    - image_edit_longcat_image_edit
- node clusters (required structure):
    - model loading: CLIPLoader, UNETLoader, VAELoader
    - conditioning: FluxGuidance (x2), TextEncodeQwenImageEdit (x2)
    - latent / canvas: VAEEncode
    - sampling: KSampler
    - decoding: VAEDecode
    - other operations: FluxKontextMultiReferenceLatentMethod (x2), ImageScaleToTotalPixels
- paired/multiple required: FluxGuidance x2, FluxKontextMultiReferenceLatentMethod x2, TextEncodeQwenImageEdit x2


# Image to Video  (`image_to_video`)  -  7 workflow(s), 2 model(s)

## Image to Video / LTX-2  (`image_to_video__ltx_2`)  -  4 workflow(s)  -  source: mixed
- execution: local
- when to use: Use to generate a video from an input image using LTX-2.
- example request: "build a video workflow using LTX-2"
- description: Generates video from Canny edge maps using LTX-2, with optional synchronized audio. | Generates video from a single input image using LTX-2.3. | Generates video from pose reference frames using LTX-2, with optional synchronized audio.
- member workflows:
    - canny_to_video_ltx_2_0
    - image_to_video_ltx_2_3
    - pose_to_video_ltx_2_0
    - video_ltx2_3_i2v
- node clusters (required structure):
    - model loading: CheckpointLoaderSimple, LTXVAudioVAELoader, LatentUpscaleModelLoader, LoraLoaderModelOnly
    - conditioning: CFGGuider (x2), CLIPTextEncode (x2), LTXAVTextEncoderLoader, LTXVConditioning, ManualSigmas
    - latent / canvas: LTXVEmptyLatentAudio
    - sampling: KSamplerSelect (x2), LTXVLatentUpsampler, SamplerCustomAdvanced (x2)
    - decoding: LTXVAudioVAEDecode, VAEDecodeTiled
    - other operations: CreateVideo, EmptyLTXVLatentVideo, LTXVConcatAVLatent (x2), LTXVCropGuides, LTXVImgToVideoInplace (x2), LTXVSeparateAVLatent (x2), PrimitiveInt, RandomNoise (x2)
- paired/multiple required: CFGGuider x2, CLIPTextEncode x2, KSamplerSelect x2, LTXVConcatAVLatent x2, LTXVImgToVideoInplace x2, LTXVSeparateAVLatent x2, RandomNoise x2, SamplerCustomAdvanced x2
- optional roles: GetVideoComponents, ImageScaleBy, LTXVAddGuide, LTXVPreprocess, LTXVScheduler, LoadImage, LoraLoader, MarkdownNote, ResizeImageMaskNode, ResizeImagesByLongerEdge, TextGenerateLTX2Prompt, VAEDecode
- unresolved nodes: MarkdownNote, Reroute

## Image to Video / WAN 2.2  (`image_to_video__wan_2_2`)  -  3 workflow(s)  -  source: mixed
- execution: local
- when to use: Use to generate a video from an input image using WAN 2.2.
- example request: "build a video workflow using WAN 2.2"
- description: Image to Video blueprint | Image-to-video with Wan 2.2 using a start image plus text prompt to extend motion from the still frame.
- member workflows:
    - image_to_video
    - image_to_video_wan_2_2
    - video_wan2_2_14B_fun_camera
- node clusters (required structure):
    - model loading: CLIPLoader, UNETLoader (x2), VAELoader
    - conditioning: CLIPTextEncode (x2)
    - sampling: KSamplerAdvanced (x2)
    - decoding: VAEDecode
    - other operations: CreateVideo, ModelSamplingSD3 (x2)
- paired/multiple required: CLIPTextEncode x2, KSamplerAdvanced x2, ModelSamplingSD3 x2, UNETLoader x2
- optional roles: LoraLoaderModelOnly, MarkdownNote, Note, GetVideoComponents, LoadImage, VHS_VideoCombine, WanCameraEmbedding, WanCameraImageToVideo, WanImageToVideo
- unresolved nodes: MarkdownNote, Note


# Video to Video  (`video_to_video`)  -  7 workflow(s), 4 model(s)

## Video to Video / WAN VACE  (`video_to_video__wan_vace`)  -  4 workflow(s)  -  source: custom
- execution: local
- when to use: Use to edit an existing video using WAN VACE, WAN 2.2.
- example request: "build a video workflow using WAN VACE"
- description: [Local] image editing via Wan. 3 image inputs -> 1 image output. Performs advanced image-to-image editing and transformations.
- member workflows:
    - Wan22Vace_VID2VID
    - video_wan_vace_14B_ref2v
    - video_wan_vace_14B_v2v
    - video_wan_vace_outpainting
- node clusters (required structure):
    - model loading: CLIPLoader, VAELoader
    - conditioning: CLIPTextEncode (x2)
    - decoding: VAEDecode
    - other operations: CreateVideo, GetVideoComponents, ModelSamplingSD3, TrimVideoLatent, WanVaceToVideo
- paired/multiple required: CLIPTextEncode x2
- optional roles: DiffusionModelLoaderKJ, DiffusionModelSelector, KSamplerAdvanced, LoadImage, LoraLoaderModelOnly, PreviewImage, BatchImagesNode, ImagePadForOutpaint, ImageStitch, ImageToMask, Int, KSampler

## Video to Video / Depth Anything  (`video_to_video__depth_anything`)  -  1 workflow(s)  -  source: official
- execution: local
- when to use: Use to edit an existing video using Bernini, Depth Anything.
- example request: "build a video workflow using Bernini"
- description: This subgraph uses Depth Anything 3 to predict spatially consistent geometry from any number of images or video frames, with or without known camera poses. It outputs depth maps, camera poses, and optionally 3D Gaussian parameters for novel view synthesis.
- member workflows:
    - video_edit_bernini_r
- node clusters (required structure):
    - model loading: CLIPLoader, LoraLoaderModelOnly (x2), UNETLoader (x2), VAELoader
    - conditioning: BasicScheduler, BerniniConditioning, CLIPTextEncode (x2), SplitSigmas
    - sampling: KSamplerSelect, SamplerCustom (x2)
    - decoding: VAEDecode
    - other operations: ComfySwitchNode (x5), CreateVideo, CustomCombo, GetVideoComponents, MarkdownNote, PreviewAny, PrimitiveBoolean, PrimitiveFloat (x2), PrimitiveInt (x5), PrimitiveStringMultiline, RegexExtract, StringConcatenate, StringReplace
- paired/multiple required: CLIPTextEncode x2, LoraLoaderModelOnly x2, SamplerCustom x2, UNETLoader x2
- unresolved nodes: BerniniConditioning, MarkdownNote

## Video to Video / LTX-2  (`video_to_video__ltx_2`)  -  1 workflow(s)  -  source: official
- execution: local
- when to use: Use to generate a video guided by a control map (canny/depth/pose) using LTX-2.
- example request: "build a video workflow using LTX-2"
- description: Generates depth-controlled video with LTX-2: motion and structure follow a depth-reference video alongside text prompting, optional first-frame image conditioning, with optional synchronized audio.
- member workflows:
    - depth_to_video_ltx_2_0
- node clusters (required structure):
    - model loading: CheckpointLoaderSimple, LTXVAudioVAELoader, LatentUpscaleModelLoader, LoraLoaderModelOnly (x2), UNETLoader, VAELoader
    - conditioning: BasicGuider, BasicScheduler, CFGGuider (x2), CLIPTextEncode (x2), LTXAVTextEncoderLoader, LTXVConditioning, LotusConditioning, ManualSigmas
    - latent / canvas: LTXVEmptyLatentAudio, VAEEncode
    - sampling: KSamplerSelect (x3), LTXVLatentUpsampler, SamplerCustomAdvanced (x3)
    - decoding: LTXVAudioVAEDecode, VAEDecode (x2), VAEDecodeTiled
    - other operations: CreateVideo, DisableNoise, EmptyLTXVLatentVideo, GetImageSize, GetVideoComponents, ImageFromBatch, ImageInvert, ImageScaleBy, LTXVAddGuide, LTXVConcatAVLatent (x2), LTXVCropGuides, LTXVImgToVideoInplace (x2), LTXVScheduler, LTXVSeparateAVLatent (x2), MarkdownNote, PrimitiveFloat, PrimitiveInt, RandomNoise (x2), Reroute, ResizeImageMaskNode, SetFirstSigma
- paired/multiple required: KSamplerSelect x3, SamplerCustomAdvanced x3, CFGGuider x2, CLIPTextEncode x2, LTXVConcatAVLatent x2, LTXVImgToVideoInplace x2, LTXVSeparateAVLatent x2, LoraLoaderModelOnly x2, RandomNoise x2, VAEDecode x2
- unresolved nodes: MarkdownNote, Reroute

## Video to Video / WAN 2.2  (`video_to_video__wan_2_2`)  -  1 workflow(s)  -  source: custom
- execution: local
- when to use: Use to generate a video guided by a control map (canny/depth/pose) using WAN 2.2.
- example request: "build a video workflow using WAN 2.2"
- description: Generate a video guided by a control map (canny/depth/pose) using WAN 2.2. Structurally it loads a diffusion model; uses a VAE; encodes a text prompt; runs a diffusion sampler; decodes the latent to pixels. Boundary inputs: IMAGE, VIDEO; outputs: AUDIO, IMAGE.
- member workflows:
    - video_wan2_2_14B_fun_control
- node clusters (required structure):
    - inputs: LoadImage, LoadVideo
    - model loading: CLIPLoader, UNETLoader (x2), VAELoader
    - conditioning: CLIPTextEncode (x2)
    - sampling: KSamplerAdvanced (x2)
    - decoding: VAEDecode
    - output: VHS_VideoCombine
    - other operations: CreateVideo, GetVideoComponents (x2), ModelSamplingSD3 (x2), Wan22FunControlToVideo
- paired/multiple required: CLIPTextEncode x2, GetVideoComponents x2, KSamplerAdvanced x2, ModelSamplingSD3 x2, UNETLoader x2


# API / Partner Nodes - Image Edit  (`api_partner_nodes_image_edit`)  -  6 workflow(s), 4 model(s)

## API / Partner Nodes - Image Edit / Nano-Banana  (`api_partner_nodes_image_edit__nano_banana`)  -  3 workflow(s)  -  source: custom
- execution: api (API nodes: GeminiImage2Node, GeminiNanoBanana2, GeminiNode)
- when to use: Use to edit an existing image using Nano-Banana, Gemini.
- example request: "build an image workflow using Nano-Banana"
- description: API image editing/generation via Nano-Banana 2. Up to 6 reference images + text prompt -> 1 image output. Edits images while maintaining subject consistency, or uses references as style guides for new image generation. | API image editing/generation via Nano-Banana Pro (Gemini 3.0 Pro). 2 image inputs -> 1 image output. Studio-quality 4K generation and editing with enhanced text rendering and character consistency. | Local style transfer FOR FULL BODY SHOTS via Nano-Banana Pro (Gemini). 1 video (layout reference) + 7 images (style + hero elements) -> 2 image outputs. Transfers the style reference onto the first video frame while integrating the look of hero element references.
- member workflows:
    - imageEdit_nano_banana2
    - imageEdit_nano_banana_pro
    - styletransfer_NanoBananaPro
- node clusters (required structure):
    - output: SaveImage
- optional roles: VHS_LoadImagePath, BatchImagesNode, LoadImage, AILab_ImageToList, GeminiImage2Node, GeminiNanoBanana2, GeminiNode, ImageListToImageBatch, PreviewImage, VHS_LoadVideoPath, VHS_SelectImages, bEpicReformat

## API / Partner Nodes - Image Edit / Kling  (`api_partner_nodes_image_edit__kling`)  -  1 workflow(s)  -  source: custom
- execution: api (API nodes: KlingOmniProImageNode)
- when to use: Use to generate an image using Kling.
- example request: "build an image workflow using Kling"
- description: Generate an image using Kling. Structurally it applies a sequence of node operations. Boundary inputs: IMAGE; outputs: IMAGE.
- member workflows:
    - api_kling_o3_image
- node clusters (required structure):
    - inputs: LoadImage (x2)
    - output: SaveImage
    - other operations: ImageBatchMulti, KlingOmniProImageNode
- paired/multiple required: LoadImage x2

## API / Partner Nodes - Image Edit / Magnific  (`api_partner_nodes_image_edit__magnific`)  -  1 workflow(s)  -  source: custom
- execution: api (API nodes: MagnificImageRelightNode)
- when to use: Use to relight an image using Magnific.
- example request: "build an image workflow using Magnific"
- description: API image relighting via Magnific. 1 source image + 1 lighting reference image -> 1 relit image output. Applies the lighting conditions from the reference onto the source image.
- member workflows:
    - api_magnific_image_relight
- node clusters (required structure):
    - inputs: LoadImage (x2)
    - output: SaveImage
    - other operations: MagnificImageRelightNode
- paired/multiple required: LoadImage x2

## API / Partner Nodes - Image Edit / Seedream  (`api_partner_nodes_image_edit__seedream`)  -  1 workflow(s)  -  source: custom
- execution: api (API nodes: ByteDanceSeedreamNode)
- when to use: Use to edit an existing image using Seedream.
- example request: "build an image workflow using Seedream"
- description: Edit an existing image using Seedream. Structurally it applies a sequence of node operations. Boundary inputs: IMAGE; outputs: IMAGE.
- member workflows:
    - api_bytedance_seedream_5_0_lite_image_edit
- node clusters (required structure):
    - inputs: LoadImage (x2)
    - output: SaveImage
    - other operations: ByteDanceSeedreamNode, ImageBatchMulti
- paired/multiple required: LoadImage x2


# First / Last Frame to Video  (`first_last_frame_to_video`)  -  5 workflow(s), 3 model(s)

## First / Last Frame to Video / LTX-2  (`first_last_frame_to_video__ltx_2`)  -  3 workflow(s)  -  source: mixed
- execution: local
- when to use: Use to generate a video interpolating between a first and last frame using LTX-2.
- example request: "build a video workflow using LTX-2"
- description: Generates a video interpolating between first and last keyframes using LTX-2.3. | Generates a video that interpolates between the first and last keyframes using LTX-2.3, including optional audio.
- member workflows:
    - first_last_frame_to_video
    - first_last_frame_to_video_ltx_2_3
    - video_ltx2_3_flf2v
- node clusters (required structure):
    - model loading: CheckpointLoaderSimple, LTXVAudioVAELoader
    - conditioning: CFGGuider, CLIPTextEncode (x2), LTXAVTextEncoderLoader, LTXVConditioning, ManualSigmas
    - latent / canvas: LTXVEmptyLatentAudio
    - sampling: SamplerCustomAdvanced, SamplerEulerAncestral
    - decoding: LTXVAudioVAEDecode, VAEDecodeTiled
    - other operations: ComfyMathExpression, CreateVideo, EmptyLTXVLatentVideo, GetImageSize, LTXVAddGuide (x2), LTXVConcatAVLatent, LTXVCropGuides, LTXVPreprocess (x2), LTXVSeparateAVLatent, PrimitiveInt (x4), RandomNoise, ResizeImageMaskNode (x2)
- paired/multiple required: CLIPTextEncode x2, LTXVAddGuide x2, LTXVPreprocess x2, ResizeImageMaskNode x2
- optional roles: LoadImage, GetVideoComponents, VHS_VideoCombine

## First / Last Frame to Video / WAN 2.2  (`first_last_frame_to_video__wan_2_2`)  -  1 workflow(s)  -  source: custom
- execution: local
- when to use: Use to generate a video interpolating between a first and last frame using WAN 2.2.
- example request: "build a video workflow using WAN 2.2"
- description: Generate a video interpolating between a first and last frame using WAN 2.2. Structurally it loads a diffusion model; uses a VAE; encodes a text prompt; runs a diffusion sampler; decodes the latent to pixels. Boundary inputs: IMAGE; outputs: AUDIO, IMAGE.
- member workflows:
    - video_wan2_2_14B_flf2v
- node clusters (required structure):
    - inputs: LoadImage (x2)
    - model loading: CLIPLoader, UNETLoader (x2), VAELoader
    - conditioning: CLIPTextEncode (x2)
    - sampling: KSamplerAdvanced (x2)
    - decoding: VAEDecode
    - output: VHS_VideoCombine
    - other operations: CreateVideo, GetVideoComponents, ModelSamplingSD3 (x2), WanFirstLastFrameToVideo
- paired/multiple required: CLIPTextEncode x2, KSamplerAdvanced x2, LoadImage x2, ModelSamplingSD3 x2, UNETLoader x2

## First / Last Frame to Video / WAN VACE  (`first_last_frame_to_video__wan_vace`)  -  1 workflow(s)  -  source: custom
- execution: local
- when to use: Use to generate a video interpolating between a first and last frame using WAN VACE.
- example request: "build a video workflow using WAN VACE"
- description: Generate a video interpolating between a first and last frame using WAN VACE. Structurally it loads a diffusion model; uses a VAE; encodes a text prompt; runs a diffusion sampler; decodes the latent to pixels. Boundary inputs: IMAGE, INT; outputs: IMAGE, MASK.
- member workflows:
    - video_wan_vace_flf2v
- node clusters (required structure):
    - inputs: LoadImage (x2)
    - model loading: CLIPLoader, LoraLoader, UNETLoader, VAELoader
    - conditioning: CLIPTextEncode (x2)
    - sampling: KSampler
    - decoding: VAEDecode
    - output: PreviewImage (x2), VHS_VideoCombine
    - other operations: CreateVideo, GetVideoComponents, ImageBatch (x4), ImageToMask, MaskToImage (x2), ModelSamplingSD3, PrimitiveInt (x4), RepeatImageBatch, SolidMask (x2), TrimVideoLatent, WanVaceToVideo
- paired/multiple required: ImageBatch x4, CLIPTextEncode x2, LoadImage x2, MaskToImage x2, PreviewImage x2, SolidMask x2


# Image Edit with ControlNet  (`image_edit_with_controlnet`)  -  5 workflow(s), 1 model(s)

## Image Edit with ControlNet / Z-Image  (`image_edit_with_controlnet__z_image`)  -  5 workflow(s)  -  source: mixed
- execution: local
- when to use: Use to generate an image guided by a control map (canny/depth/pose) using Z-Image.
- example request: "build an image workflow using Z-Image"
- description: Generates an image from a Canny edge map using Z-Image-Turbo, with text conditioning. | Generates an image from a depth map using Z-Image-Turbo with text conditioning. | Generates an image from pose keypoints using Z-Image-Turbo with text conditioning. | Generates images from a text prompt and ControlNet conditioning (e.g. depth, canny) using Z-Image-Turbo. | [Local] image editing via Z-Image-Turbo. 1 image input -> 1 image output. Uses ControlNet for precise and controlled image editing.
- member workflows:
    - canny_to_image_z_image_turbo
    - controlnet_z_image_turbo
    - depth_to_image_z_image_turbo
    - image_z_image_turbo_fun_union_controlnet
    - pose_to_image_z_image_turbo
- node clusters (required structure):
    - model loading: CLIPLoader, ModelPatchLoader, UNETLoader, VAELoader
    - conditioning: CLIPTextEncode, ConditioningZeroOut, QwenImageDiffsynthControlnet
    - latent / canvas: EmptySD3LatentImage
    - sampling: KSampler
    - decoding: VAEDecode
    - other operations: GetImageSize, ModelSamplingAuraFlow
- optional roles: BasicGuider, BasicScheduler, Canny, DisableNoise, ImageInvert, ImageScaleToTotalPixels, KSamplerSelect, LoadImage, LotusConditioning, PreviewImage, SamplerCustomAdvanced, SaveImage


# Text Tools  (`text_tools`)  -  5 workflow(s), 2 model(s)

## Text Tools / Gemini  (`text_tools__gemini`)  -  3 workflow(s)  -  source: mixed
- execution: hybrid (API nodes: GeminiNode)
- when to use: Use to caption an image as text using Gemini.
- example request: "build a text workflow using Gemini"
- description: Generates descriptive captions for images using Google's Gemini multimodal LLM. | Generates descriptive captions for video input using Google's Gemini multimodal LLM. | [API] motion prompt generation via Gemini, analyses a video and output a desscription of the motion in it. 1 video input -> 1 output. Generates descriptive motion prompts for video generation.
- member workflows:
    - image_captioning_gemini
    - video_captioning_gemini
    - video_gemini_motionPromptGeneration
- node clusters (required structure):
    - other operations: GeminiNode
- optional roles: CreateVideo, GetVideoComponents, ImageResizeKJv2, LoadVideo, easy saveText

## Text Tools / Generic  (`text_tools__generic`)  -  2 workflow(s)  -  source: official
- execution: hybrid (API nodes: GeminiNode)
- when to use: Use to expand a short prompt into a detailed one.
- example request: "build an image workflow"
- description: Expands short text prompts into detailed descriptions using a text generation model for better generation quality. | Selects one line from multiline text by zero-based index for batch or list-driven prompt workflows.
- member workflows:
    - prompt_enhance
    - select_per_line_text_by_index
- node clusters (required structure):
    - (none resolved)
- optional roles: GeminiNode, RegexExtract


# API / Partner Nodes - Text to Video  (`api_partner_nodes_text_to_video`)  -  4 workflow(s), 4 model(s)

## API / Partner Nodes - Text to Video / Kling  (`api_partner_nodes_text_to_video__kling`)  -  1 workflow(s)  -  source: custom
- execution: api (API nodes: KlingVideoNode)
- when to use: Use to generate a video using Kling.
- example request: "build a video workflow using Kling"
- description: API multi-shot storyboard video via Kling 3.0 (kling-v3). 1 input image (start frame, LoadImage node) -> 1 video output (VHS_VideoCombine). Generates 1-6 sequential shots in a single generation: each shot has its own text prompt (max 512 chars) and duration set directly on the KlingVideoNode. Use for storyboards, scene sequences, and narrative clips with multiple camera cuts. Prompts go into multi_shot.storyboard_N_prompt inputs; multi_shot must match shot count exactly (e.g. '3 storyboards'). Aspect ratio defaults to 16:9, resolution to 720p - override only on explicit user request.
- member workflows:
    - Kling3_multiShot
- node clusters (required structure):
    - inputs: LoadImage
    - output: VHS_VideoCombine
    - other operations: GetVideoComponents, KlingVideoNode

## API / Partner Nodes - Text to Video / LTX-2  (`api_partner_nodes_text_to_video__ltx_2`)  -  1 workflow(s)  -  source: custom
- execution: api (API nodes: LtxvApiTextToVideo)
- when to use: Use to generate a video from a text prompt using LTX-2.
- example request: "build a video workflow using LTX-2"
- description: Generate a video from a text prompt using LTX-2. Structurally it applies a sequence of node operations. Boundary inputs: VIDEO; outputs: AUDIO, IMAGE.
- member workflows:
    - api_ltxv_text_to_video
- node clusters (required structure):
    - output: VHS_VideoCombine
    - other operations: GetVideoComponents, LtxvApiTextToVideo

## API / Partner Nodes - Text to Video / Veo  (`api_partner_nodes_text_to_video__veo`)  -  1 workflow(s)  -  source: custom
- execution: api (API nodes: Veo3VideoGenerationNode)
- when to use: Use to run a node graph using Veo.
- example request: "build a video workflow using Veo"
- description: Run a node graph using Veo. Structurally it applies a sequence of node operations. Boundary inputs: IMAGE; outputs: VIDEO.
- member workflows:
    - api_veo3
- node clusters (required structure):
    - inputs: LoadImage
    - output: SaveVideo
    - other operations: Veo3VideoGenerationNode

## API / Partner Nodes - Text to Video / WAN 2.6  (`api_partner_nodes_text_to_video__wan_2_6`)  -  1 workflow(s)  -  source: custom
- execution: api (API nodes: WanTextToVideoApi)
- when to use: Use to generate a video from a text prompt using WAN 2.6.
- example request: "build a video workflow using WAN 2.6"
- description: API text-to-video via Wan 2.6. Text prompt only -> 1 video output. Generates 1080P video with enhanced quality, smoother motion, and improved prompt understanding.
- member workflows:
    - api_wan2_6_t2v
- node clusters (required structure):
    - output: VHS_VideoCombine
    - other operations: GetVideoComponents, WanTextToVideoApi


# API / Partner Nodes - Upscale  (`api_partner_nodes_upscale`)  -  4 workflow(s), 3 model(s)

## API / Partner Nodes - Upscale / Magnific  (`api_partner_nodes_upscale__magnific`)  -  2 workflow(s)  -  source: custom
- execution: api (API nodes: MagnificImageUpscalerCreativeNode, MagnificImageUpscalerPreciseV2Node)
- when to use: Use to upscale / enhance an image using Magnific.
- example request: "build an image workflow using Magnific"
- description: API creative image upscaling via Magnific. 1 image -> 1 upscaled image output. Supports up to 16x enlargement with creative detail enhancement. | API precise image upscaling via Magnific. 1 image -> 1 high-resolution image output. Upscales with strict detail preservation and enhanced sharpness.
- member workflows:
    - api_magnific_image_upscale_creative
    - api_magnific_image_upscale_precise
- node clusters (required structure):
    - inputs: LoadImage
    - output: SaveImage
- optional roles: MagnificImageUpscalerCreativeNode, MagnificImageUpscalerPreciseV2Node

## API / Partner Nodes - Upscale / Topaz  (`api_partner_nodes_upscale__topaz`)  -  1 workflow(s)  -  source: custom
- execution: api (API nodes: TopazVideoEnhance)
- when to use: Use to upscale / enhance a video using Topaz.
- example request: "build a video workflow using Topaz"
- description: API video upscaling via Topaz AI. 1 video -> 1 enhanced video output. Supports resolution upscaling (Starlight/Astra Fast model) and frame interpolation (apo-8 model).
- member workflows:
    - api_topaz_video_enhance
- node clusters (required structure):
    - inputs: LoadVideo
    - output: SaveVideo
    - other operations: TopazVideoEnhance

## API / Partner Nodes - Upscale / Z-Image  (`api_partner_nodes_upscale__z_image`)  -  1 workflow(s)  -  source: custom
- execution: api (API nodes: TopazImageEnhance)
- when to use: Use to upscale / enhance an image using Topaz, Z-Image.
- example request: "build an image workflow using Topaz"
- description: API image enhancement/upscaling via Topaz Reimagine. 1 image -> 1 enhanced image output. Applies face enhancement and detail restoration for professional results.
- member workflows:
    - api_topaz_image_enhance
- node clusters (required structure):
    - inputs: LoadImage
    - output: SaveImage
    - other operations: TopazImageEnhance


# Upscale  (`upscale`)  -  4 workflow(s), 3 model(s)

## Upscale / Generic  (`upscale__generic`)  -  2 workflow(s)  -  source: mixed
- execution: local
- when to use: Use to upscale / enhance an image.
- example request: "build an image workflow"
- description: Local, simple image upscaling via specified ESRGAN model. 1 image -> 1 upscaled image output. Supports various models. | Upscales video to 4× resolution using a GAN-based upscaling model.
- member workflows:
    - upscale_using_model
    - video_upscale_gan_x4
- node clusters (required structure):
    - model loading: UpscaleModelLoader
    - other operations: ImageUpscaleWithModel
- optional roles: CreateVideo, GetVideoComponents, LoadImage, SaveImage

## Upscale / Flux  (`upscale__flux`)  -  1 workflow(s)  -  source: custom
- execution: local
- when to use: Use to upscale / enhance an image using Flux.
- example request: "build an image workflow using Flux"
- description: Local image upscaling using UltimateSD upscale node (this uses a diffusion model for the upscale process, allowing a creative upscale that invents details). Setup with Flux-1 dev fp8. 1 image -> 1 upscaled image output.
- member workflows:
    - upscale_ultimateSD
- node clusters (required structure):
    - inputs: LoadImage
    - model loading: CheckpointLoaderSimple, UpscaleModelLoader
    - conditioning: CLIPTextEncode (x2)
    - output: SaveImage
    - other operations: UltimateSDUpscale
- paired/multiple required: CLIPTextEncode x2

## Upscale / Z-Image  (`upscale__z_image`)  -  1 workflow(s)  -  source: official
- execution: local
- when to use: Use to upscale / enhance an image using Z-Image.
- example request: "build an image workflow using Z-Image"
- description: Upscales images to higher resolution using Z-Image-Turbo.
- member workflows:
    - image_upscale_z_image_turbo
- node clusters (required structure):
    - model loading: CLIPLoader, UNETLoader, UpscaleModelLoader, VAELoader
    - conditioning: CLIPTextEncode (x2)
    - latent / canvas: VAEEncode
    - sampling: KSampler
    - decoding: VAEDecode
    - other operations: ImageScaleBy, ImageScaleToTotalPixels, ImageUpscaleWithModel, ModelSamplingAuraFlow
- paired/multiple required: CLIPTextEncode x2


# Video Tools  (`video_tools`)  -  4 workflow(s), 1 model(s)

## Video Tools / Generic  (`video_tools__generic`)  -  4 workflow(s)  -  source: official
- execution: local
- when to use: Use to generate a video.
- example request: "build a video workflow"
- description: Concatenates two videos end-to-end with optional resize, letterbox padding, and audio merge or drop. | Extracts one image frame from a video at a chosen index, with optional trim and FPS control. | Increases video frame rate by synthesizing intermediate frames with a frame interpolation model. | Stitches multiple video clips into a single sequential video file.
- member workflows:
    - frame_interpolation
    - get_any_video_frame
    - merge_videos
    - video_stitch
- node clusters (required structure):
    - other operations: GetVideoComponents
- optional roles: AudioMerge, BatchImagesNode, CreateVideo, EmptyAudio, FrameInterpolate, FrameInterpolationModelLoader, ImageFromBatch, ImageStitch, ResizeAndPadImage, ResizeImageMaskNode


# 3D  (`3d`)  -  3 workflow(s), 3 model(s)

## 3D / Hunyuan3D  (`3d__hunyuan3d`)  -  1 workflow(s)  -  source: official
- execution: local
- when to use: Use to generate a 3D model using Hunyuan3D.
- example request: "build a 3d workflow using Hunyuan3D"
- description: Generates 3D mesh models from a single input image using Hunyuan3D 2.0/2.1.
- member workflows:
    - image_to_model_hunyuan3d_2_1
- node clusters (required structure):
    - model loading: ImageOnlyCheckpointLoader
    - conditioning: Hunyuan3Dv2Conditioning
    - latent / canvas: EmptyLatentHunyuan3Dv2
    - sampling: KSampler
    - decoding: VAEDecodeHunyuan3D
    - other operations: CLIPVisionEncode, ModelSamplingAuraFlow, VoxelToMesh

## 3D / MoGe  (`3d__moge`)  -  1 workflow(s)  -  source: official
- execution: local
- when to use: Use to estimate 3D scene geometry using MoGe.
- example request: "build a 3d workflow using MoGe"
- description: Estimates 3D scene geometry from an input image using MoGe, outputting a mesh plus OpenGL and DirectX normal maps.
- member workflows:
    - geometry_estimation_moge
- node clusters (required structure):
    - other operations: ComfyMathExpression, ComfySwitchNode (x2), GetImageSize, LoadMoGeModel, MoGeInference, MoGePointMapToMesh, MoGeRender (x2), ResizeImagesByLongerEdge
- paired/multiple required: MoGeRender x2

## 3D / TripoSplat  (`3d__triposplat`)  -  1 workflow(s)  -  source: official
- execution: local
- when to use: Use to generate a 3D model using TripoSplat.
- example request: "build a 3d workflow using TripoSplat"
- description: This subgraph takes a single 2D image as input and generates a variable number of 3D Gaussians (up to 262,144) as output, enabling high-quality 3D reconstruction. It is ideal for asset creation, AR/VR, game development, and simulation environments, handling diverse image styles from photos to illustrations.
- member workflows:
    - image_to_gaussian_splat_triposplat
- node clusters (required structure):
    - model loading: UNETLoader, VAELoader (x2)
    - conditioning: TripoSplatConditioning
    - sampling: KSampler
    - decoding: VAEDecodeTripoSplat
    - output: PreviewImage
    - other operations: CLIPVisionLoader, ComfySwitchNode (x2), InvertMask (x2), JoinImageWithAlpha, LoadBackgroundRemovalModel, RemoveBackground, TripoSplatPreprocessImage, TripoSplatSamplingPreview
- paired/multiple required: InvertMask x2, VAELoader x2


# API / Partner Nodes - 3D  (`api_partner_nodes_3d`)  -  3 workflow(s), 1 model(s)

## API / Partner Nodes - 3D / Meshy  (`api_partner_nodes_3d__meshy`)  -  3 workflow(s)  -  source: custom
- execution: api (API nodes: MeshyImageToModelNode, MeshyMultiImageToModelNode, MeshyTextToModelNode)
- when to use: Use to generate a 3D model using Meshy.
- example request: "build a 3d workflow using Meshy"
- description: API image-to-3D via Meshy 6. 1 image -> 1 3D model output. Generates characters, objects, or mechanical parts with production-quality geometry and clean topology. | API multi-image-to-3D via Meshy 6. 3+ images -> 1 3D model output. More input views yield better detail capture, accurate proportions, and cleaner mesh structure. | API text-to-3D via Meshy 6. Text prompt only -> 1 3D model output. Creates characters, mechanical objects, or game-ready low-poly assets with refined geometry.
- member workflows:
    - api_meshy_image_to_model
    - api_meshy_multi_image_to_model
    - api_meshy_text_to_model
- node clusters (required structure):
    - other operations: SaveGLB (x2)
- paired/multiple required: SaveGLB x2
- optional roles: LoadImage, MeshyImageToModelNode, MeshyMultiImageToModelNode, MeshyTextToModelNode


# API / Partner Nodes - Image to Video  (`api_partner_nodes_image_to_video`)  -  3 workflow(s), 3 model(s)

## API / Partner Nodes - Image to Video / Kling  (`api_partner_nodes_image_to_video__kling`)  -  1 workflow(s)  -  source: custom
- execution: api (API nodes: KlingOmniProImageToVideoNode)
- when to use: Use to generate a video from an input image using Kling.
- example request: "build a video workflow using Kling"
- description: API image-to-video via Kling O3 (Kling 3.0). 1 reference image (+ optional audio/text prompt) -> 1 video output. Generates character-consistent video with native audio output and precise storyboard control.
- member workflows:
    - api_kling_o3_i2v
- node clusters (required structure):
    - inputs: LoadImage
    - output: VHS_VideoCombine
    - other operations: GetVideoComponents, ImageBatchMulti, KlingOmniProImageToVideoNode

## API / Partner Nodes - Image to Video / LTX-2  (`api_partner_nodes_image_to_video__ltx_2`)  -  1 workflow(s)  -  source: custom
- execution: api (API nodes: LtxvApiImageToVideo)
- when to use: Use to generate a video from an input image using LTX-2.
- example request: "build a video workflow using LTX-2"
- description: Generate a video from an input image using LTX-2. Structurally it applies a sequence of node operations. Boundary inputs: IMAGE; outputs: AUDIO, IMAGE.
- member workflows:
    - api_ltxv_image_to_video
- node clusters (required structure):
    - inputs: LoadImage
    - output: VHS_VideoCombine
    - other operations: GetVideoComponents, LtxvApiImageToVideo

## API / Partner Nodes - Image to Video / WAN 2.6  (`api_partner_nodes_image_to_video__wan_2_6`)  -  1 workflow(s)  -  source: custom
- execution: api (API nodes: WanImageToVideoApi)
- when to use: Use to generate a video from an input image using WAN 2.6.
- example request: "build a video workflow using WAN 2.6"
- description: API image-to-video via Wan 2.6. 1 image -> 1 video output. Generates 1080P video with enhanced image quality, smoother motion, and natural movement.
- member workflows:
    - api_wan2_6_i2v
- node clusters (required structure):
    - inputs: LoadImage
    - output: VHS_VideoCombine
    - other operations: GetVideoComponents, WanImageToVideoApi


# Audio  (`audio`)  -  3 workflow(s), 2 model(s)

## Audio / Qwen Image  (`audio__qwen_image`)  -  2 workflow(s)  -  source: official
- execution: local
- when to use: Use to generate audio from a text prompt using Qwen Image, Stable Audio.
- example request: "build an audio workflow using Qwen Image"
- description: Generates music, instrument loops, sound effects, and one-shots from text using Stable Audio 3 Medium, with optional Qwen 3.5 category-based prompt expansion (Music, Instrument, SFX, One-shot). | Generates music, instrument loops, sound effects, and one-shots from text using the Stable Audio 3 Medium base checkpoint, with optional Qwen 3.5 category-based prompt expansion (Music, Instrument, SFX, One-shot).
- member workflows:
    - audio_generation_stable_audio_3_medium
    - audio_generation_stable_audio_3_medium_base
- node clusters (required structure):
    - model loading: CLIPLoader (x2), CheckpointLoaderSimple
    - conditioning: CLIPTextEncode (x2)
    - latent / canvas: EmptyLatentAudio
    - sampling: KSampler
    - decoding: VAEDecodeAudio
    - other operations: ComfyMathExpression, ComfySwitchNode, CustomCombo, JsonExtractString, PreviewAny (x2), PrimitiveBoolean, PrimitiveFloat, PrimitiveStringMultiline, StringReplace (x3), TextGenerate
- paired/multiple required: CLIPLoader x2, CLIPTextEncode x2

## Audio / ACE-Step  (`audio__ace_step`)  -  1 workflow(s)  -  source: official
- execution: local
- when to use: Use to generate audio from a text prompt using ACE-Step.
- example request: "build an audio workflow using ACE-Step"
- description: Generates audio/music from text prompts using ACE-Step 1.5, a diffusion-based audio generation model.
- member workflows:
    - text_to_audio_ace_step_1_5
- node clusters (required structure):
    - model loading: DualCLIPLoader, UNETLoader, VAELoader
    - conditioning: ConditioningZeroOut, TextEncodeAceStepAudio1.5
    - sampling: KSampler
    - decoding: VAEDecodeAudio
    - other operations: EmptyAceStep1.5LatentAudio, ModelSamplingAuraFlow, PrimitiveFloat, PrimitiveNode
- unresolved nodes: PrimitiveNode


# Inpaint / Outpaint  (`inpaint_outpaint`)  -  3 workflow(s), 2 model(s)

## Inpaint / Outpaint / Qwen Image  (`inpaint_outpaint__qwen_image`)  -  2 workflow(s)  -  source: official
- execution: local
- when to use: Use to inpaint masked regions of an image using Qwen Image.
- example request: "build an image workflow using Qwen Image"
- description: Inpaints masked regions using Qwen-Image, extending its multilingual text rendering to inpainting tasks. | Outpaints beyond image boundaries using Qwen-Image's outpainting capabilities.
- member workflows:
    - image_inpainting_qwen_image
    - image_outpainting_qwen_image
- node clusters (required structure):
    - model loading: CLIPLoader, LoraLoaderModelOnly, UNETLoader, VAELoader
    - conditioning: CLIPTextEncode (x2), ControlNetInpaintingAliMamaApply, ControlNetLoader
    - latent / canvas: VAEEncode
    - sampling: KSampler
    - decoding: VAEDecode
    - other operations: GrowMask, ImageBlur, ImageToMask, MaskPreview, MaskToImage, ModelSamplingAuraFlow
- paired/multiple required: CLIPTextEncode x2
- optional roles: ImageScaleToMaxDimension, FluxKontextImageScale, ImageCompositeMasked, ImagePadForOutpaint, MarkdownNote, Note, PreviewImage, SetLatentNoiseMask
- unresolved nodes: MarkdownNote, Note

## Inpaint / Outpaint / Flux  (`inpaint_outpaint__flux`)  -  1 workflow(s)  -  source: official
- execution: local
- when to use: Use to inpaint masked regions of an image using Flux.
- example request: "build an image workflow using Flux"
- description: Inpaints masked image regions using Flux.1 fill [dev], Black Forest Labs' inpainting/outpainting model.
- member workflows:
    - image_inpainting_flux_1_fill_dev
- node clusters (required structure):
    - model loading: DualCLIPLoader, UNETLoader, VAELoader
    - conditioning: CLIPTextEncode, ConditioningZeroOut, FluxGuidance, InpaintModelConditioning
    - sampling: KSampler
    - decoding: VAEDecode
    - other operations: DifferentialDiffusion


# Video Inpaint  (`video_inpaint`)  -  3 workflow(s), 2 model(s)

## Video Inpaint / WAN VACE  (`video_inpaint__wan_vace`)  -  2 workflow(s)  -  source: official
- execution: local
- when to use: Use to inpaint regions of a video using WAN VACE, SAM3.
- example request: "build a video workflow using WAN VACE"
- description: Removes objects from video by inpainting masked regions using Wan 2.1 VACE, with SAM3 text-guided segmentation and optional Lightning LoRA turbo mode. | Video Inpaint(Wan2.1 VACE) blueprint
- member workflows:
    - video_inpaint_wan2_1_vace
    - video_inpainting_wan2_1_vace
- node clusters (required structure):
    - model loading: CLIPLoader, LoraLoaderModelOnly, UNETLoader, VAELoader
    - conditioning: CLIPTextEncode (x2)
    - sampling: KSampler
    - decoding: VAEDecode
    - output: PreviewImage
    - other operations: CreateVideo, GetImageSize, GetVideoComponents, ImageCompositeMasked, ImageFromBatch, InvertMask, MaskToImage, ModelSamplingSD3, TrimVideoLatent, WanVaceToVideo
- paired/multiple required: CLIPTextEncode x2
- optional roles: MarkdownNote, CheckpointLoaderSimple, GrowMask, ImageToMask, MaskPreview, RebatchImages, RepeatImageBatch, ResizeImageMaskNode, SAM3_Detect
- unresolved nodes: MarkdownNote

## Video Inpaint / SAM3  (`video_inpaint__sam3`)  -  1 workflow(s)  -  source: official
- execution: local
- when to use: Use to inpaint regions of a video using SAM3.
- example request: "build a video workflow using SAM3"
- description: Removes objects from video by inpainting masked regions using VOID (CogVideoX), with SAM3 text-guided segmentation and optional two-pass optical-flow refinement.
- member workflows:
    - video_inpaint_void
- node clusters (required structure):
    - model loading: CLIPLoader, CheckpointLoaderSimple, UNETLoader (x2), VAELoader
    - conditioning: BasicScheduler (x2), CFGGuider (x2), CLIPTextEncode (x3), VOIDInpaintConditioning
    - sampling: SamplerCustomAdvanced (x2), VOIDSampler (x2)
    - decoding: VAEDecode (x2)
    - other operations: ComfyMathExpression (x2), ComfySwitchNode, CreateVideo (x2), GetImageSize, GetVideoComponents, ImageFromBatch, MaskPreview, OpticalFlowLoader, PrimitiveBoolean, PrimitiveInt (x4), RandomNoise, SAM3_Detect, TrimAudioDuration, VOIDWarpedNoise, VOIDWarpedNoiseSource
- paired/multiple required: CLIPTextEncode x3, BasicScheduler x2, CFGGuider x2, CreateVideo x2, SamplerCustomAdvanced x2, UNETLoader x2, VAEDecode x2, VOIDSampler x2


# API / Partner Nodes - Character  (`api_partner_nodes_character`)  -  2 workflow(s), 1 model(s)

## API / Partner Nodes - Character / Nano-Banana  (`api_partner_nodes_character__nano_banana`)  -  2 workflow(s)  -  source: custom
- execution: api (API nodes: GeminiImage2Node, GeminiNode)
- when to use: Use to generate a multi-pose character sheet using Nano-Banana.
- example request: "build an image workflow using Nano-Banana"
- description: API character sheet generation FOR FACE CLOSEUPS via Nano-Banana Pro. 1 character image -> 1 image output (3x3 sheet). Uses an LLM call to generate a prompt from the reference, then renders 9 character views with varying facial expressions in a single sheet. | API character sheet generation via Nano-Banana Pro. 1 character image -> 1 image output (3x3 sheet). Uses an LLM call to generate a prompt from the reference, then renders 9 character views with varying body pose in a single sheet.
- member workflows:
    - NanoBananaPro_3x3CharacterSheet
    - NanoBananaPro_3x3CharacterSheet_closeups
- node clusters (required structure):
    - inputs: LoadImage
    - output: SaveImage
    - other operations: GeminiImage2Node, GeminiNode, PrimitiveStringMultiline


# Character  (`character`)  -  2 workflow(s), 1 model(s)

## Character / SCAIL  (`character__scail`)  -  2 workflow(s)  -  source: official
- execution: local
- when to use: Use to replace a character in a video using Anima, SCAIL.
- example request: "build a video workflow using Anima"
- description: Replaces a character in a video with a reference image using the SCAIL-2 model for end-to-end controlled animation without intermediate pose maps. Key inputs include a source video, a reference character image, and optional text prompts for style or context. Suitable for animated or live-action footage, multi-character scenes, and creative video editing where direct pose-free animation is needed; works best with moderate-length videos.
- member workflows:
    - character_replacement_scail_2_base
    - character_replacement_scail_2_extend
- node clusters (required structure):
    - model loading: CLIPLoader, CheckpointLoaderSimple, LoraLoaderModelOnly (x2), UNETLoader, VAELoader
    - conditioning: BasicScheduler, CLIPTextEncode (x4)
    - sampling: KSamplerSelect, SamplerCustom
    - decoding: VAEDecode
    - output: PreviewImage (x2)
    - other operations: CLIPVisionEncode, CLIPVisionLoader, ComfyMathExpression (x3), ComfySwitchNode (x3), GetImageSize, GetVideoComponents, ImageFromBatch, ModelSamplingSD3, PrimitiveBoolean (x2), PrimitiveFloat (x2), PrimitiveInt (x5), ResizeImageMaskNode, SAM3_VideoTrack (x2), SCAIL2ColoredMask, WanSCAILToVideo
- paired/multiple required: CLIPTextEncode x4, LoraLoaderModelOnly x2, PreviewImage x2, SAM3_VideoTrack x2
- optional roles: ColorTransfer
- unresolved nodes: SCAIL2ColoredMask


# Text to Video  (`text_to_video`)  -  2 workflow(s), 2 model(s)

## Text to Video / LTX-2  (`text_to_video__ltx_2`)  -  1 workflow(s)  -  source: official
- execution: local
- when to use: Use to generate a video from a text prompt using LTX-2.
- example request: "build a video workflow using LTX-2"
- description: Generates video from text prompts using LTX-2.3, Lightricks' video diffusion model.
- member workflows:
    - text_to_video_ltx_2_3
- node clusters (required structure):
    - inputs: LoadImage
    - model loading: CheckpointLoaderSimple, LTXVAudioVAELoader, LatentUpscaleModelLoader, LoraLoaderModelOnly
    - conditioning: CFGGuider (x2), CLIPTextEncode (x2), LTXAVTextEncoderLoader, LTXVConditioning, ManualSigmas (x2)
    - latent / canvas: LTXVEmptyLatentAudio
    - sampling: KSamplerSelect (x2), LTXVLatentUpsampler, SamplerCustomAdvanced (x2)
    - decoding: LTXVAudioVAEDecode, VAEDecodeTiled
    - other operations: ComfyMathExpression (x4), CreateVideo, EmptyLTXVLatentVideo, LTXVConcatAVLatent (x2), LTXVCropGuides, LTXVImgToVideoInplace (x2), LTXVPreprocess, LTXVSeparateAVLatent (x2), PrimitiveBoolean, PrimitiveInt (x4), PrimitiveStringMultiline, RandomNoise (x2), Reroute, ResizeImageMaskNode, ResizeImagesByLongerEdge
- paired/multiple required: CFGGuider x2, CLIPTextEncode x2, KSamplerSelect x2, LTXVConcatAVLatent x2, LTXVImgToVideoInplace x2, LTXVSeparateAVLatent x2, ManualSigmas x2, RandomNoise x2, SamplerCustomAdvanced x2
- unresolved nodes: Reroute

## Text to Video / WAN 2.2  (`text_to_video__wan_2_2`)  -  1 workflow(s)  -  source: official
- execution: local
- when to use: Use to generate a video from a text prompt using WAN 2.2.
- example request: "build a video workflow using WAN 2.2"
- description: Generates video from text prompts using Wan2.2, Alibaba's diffusion video model.
- member workflows:
    - text_to_video_wan_2_2
- node clusters (required structure):
    - model loading: CLIPLoader, LoraLoaderModelOnly (x2), UNETLoader (x2), VAELoader
    - conditioning: CLIPTextEncode (x2)
    - sampling: KSamplerAdvanced (x2)
    - decoding: VAEDecode
    - other operations: CreateVideo, EmptyHunyuanLatentVideo, MarkdownNote, ModelSamplingSD3 (x2), Note
- paired/multiple required: CLIPTextEncode x2, KSamplerAdvanced x2, LoraLoaderModelOnly x2, ModelSamplingSD3 x2, UNETLoader x2
- unresolved nodes: MarkdownNote, Note


# API / Partner Nodes - First / Last Frame to Video  (`api_partner_nodes_first_last_frame_to_video`)  -  1 workflow(s), 1 model(s)

## API / Partner Nodes - First / Last Frame to Video / Kling  (`api_partner_nodes_first_last_frame_to_video__kling`)  -  1 workflow(s)  -  source: custom
- execution: api (API nodes: KlingOmniProFirstLastFrameNode)
- when to use: Use to generate a video interpolating between a first and last frame using Kling.
- example request: "build a video workflow using Kling"
- description: API first-last-frame-to-video via Kling O3 (Kling 3.0). Up to 4 reference/keyframe images -> 1 video output. Generates videos with precise semantic control, longer duration, and improved narrative coherence.
- member workflows:
    - api_kling_o3_flf2v
- node clusters (required structure):
    - inputs: LoadImage (x3)
    - output: VHS_VideoCombine
    - other operations: GetVideoComponents, ImageBatchMulti, KlingOmniProFirstLastFrameNode
- paired/multiple required: LoadImage x3


# API / Partner Nodes - Inpaint / Outpaint  (`api_partner_nodes_inpaint_outpaint`)  -  1 workflow(s), 1 model(s)

## API / Partner Nodes - Inpaint / Outpaint / Nano-Banana  (`api_partner_nodes_inpaint_outpaint__nano_banana`)  -  1 workflow(s)  -  source: custom
- execution: api (API nodes: GeminiNanoBanana2)
- when to use: Use to outpaint / extend an image beyond its borders using Nano-Banana.
- example request: "build an image workflow using Nano-Banana"
- description: API upscale and outpaint via Nano-Banana 2. 1 image -> 1 image output. Upscales the input image while also generating new content around the edges to expand the overall dimensions, guided by the original image's style and content.
- member workflows:
    - NanoBanana2_outpaintUpscale
- node clusters (required structure):
    - inputs: LoadImage
    - output: SaveImage
    - other operations: GeminiNanoBanana2


# API / Partner Nodes - Text to Image  (`api_partner_nodes_text_to_image`)  -  1 workflow(s), 1 model(s)

## API / Partner Nodes - Text to Image / Ideogram  (`api_partner_nodes_text_to_image__ideogram`)  -  1 workflow(s)  -  source: custom
- execution: api (API nodes: IdeogramV3)
- when to use: Use to generate an image from a text prompt using Ideogram.
- example request: "build an image workflow using Ideogram"
- description: Generate an image from a text prompt using Ideogram. Structurally it applies a sequence of node operations. Boundary inputs: IMAGE; outputs: IMAGE.
- member workflows:
    - api_ideogram_v3_t2i
- node clusters (required structure):
    - output: SaveImage
    - other operations: IdeogramV3


# API / Partner Nodes - Video to Video  (`api_partner_nodes_video_to_video`)  -  1 workflow(s), 1 model(s)

## API / Partner Nodes - Video to Video / Kling  (`api_partner_nodes_video_to_video__kling`)  -  1 workflow(s)  -  source: custom
- execution: api (API nodes: KlingOmniProEditVideoNode)
- when to use: Use to edit an existing video using Kling.
- example request: "build a video workflow using Kling"
- description: API video editing via Kling O3. 1 video + 1 reference image -> 1 edited video output. Enables precise subject editing and scene composition with native audio-visual synchronization.
- member workflows:
    - api_kling_o3_video_edit
- node clusters (required structure):
    - inputs: LoadImage, LoadVideo
    - output: VHS_VideoCombine
    - other operations: GetVideoComponents, KlingOmniProEditVideoNode

