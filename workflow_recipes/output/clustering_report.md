# Clustering report (Phase 3)

- Workflows: 137
- Types (clusters): 35  (23 multi-member, 12 singletons)
- Similarity threshold: 0.3
- Signal weights: {'classes': 0.4, 'connections': 0.35, 'clusters': 0.2, 'spine': 0.05, 'category': 0.0}

Sanity-check the groupings below, then tell me a threshold to lock in before I build Phase 4 (recipe synthesis).

## Type 1  -  15 member(s)  -  source: mixed
- cohesion (mean intra-similarity): 0.422
- official categories: Video generation and editing (7)  [pure]
- members:
    - Wan22Vace_VID2VID  (custom, 29 nodes)
    - video_wan2_2_14B_flf2v  (custom, 17 nodes)
    - video_wan2_2_14B_fun_camera  (custom, 17 nodes)
    - video_wan2_2_14B_fun_control  (custom, 18 nodes)
    - video_wan_vace_14B_ref2v  (custom, 15 nodes)
    - video_wan_vace_14B_v2v  (custom, 17 nodes)
    - video_wan_vace_flf2v  (custom, 32 nodes)
    - video_wan_vace_outpainting  (custom, 25 nodes)
    - character_replacement_scail_2_base  (official, 42 nodes) - "Character Replacement (SCAIL-2 Base)"
    - character_replacement_scail_2_extend  (official, 45 nodes) - "Character Replacement (SCAIL-2 Extend)"
    - image_to_video  (official, 15 nodes) - "Image to Video"
    - image_to_video_wan_2_2  (official, 19 nodes) - "Image to Video (Wan 2.2)"
    - text_to_video_wan_2_2  (official, 17 nodes) - "Text to Video (Wan 2.2)"
    - video_inpaint_wan2_1_vace  (official, 26 nodes) - "Video Inpaint(Wan2.1 VACE)"
    - video_inpainting_wan2_1_vace  (official, 38 nodes) - "Video Inpainting (Wan2.1 VACE)"
- shared node classes (5): CLIPLoader, CLIPTextEncode, ModelSamplingSD3, VAEDecode, VAELoader
- shared connection patterns (3):
    - other -> sampler  [LATENT]
    - other -> sampler  [MODEL]
    - vae_loader -> vae_decode  [VAE]

## Type 2  -  15 member(s)  -  source: mixed
- cohesion (mean intra-similarity): 0.483
- official categories: Image generation and editing (12), Audio (1)  [MIXED categories]
- members:
    - image_z_image_turbo  (custom, 10 nodes)
    - image_z_image_turbo_fun_union_controlnet  (custom, 15 nodes)
    - canny_to_image_z_image_turbo  (official, 15 nodes) - "Canny to Image (Z-Image-Turbo)"
    - controlnet_z_image_turbo  (official, 12 nodes) - "ControlNet (Z-Image-Turbo)"
    - depth_to_image_z_image_turbo  (official, 26 nodes) - "Depth to Image (Z-Image-Turbo)"
    - image_inpainting_flux_1_fill_dev  (official, 10 nodes) - "Image Inpainting (Flux.1 Fill Dev)"
    - pose_to_image_z_image_turbo  (official, 12 nodes) - "Pose to Image (Z-Image-Turbo)"
    - text_to_audio_ace_step_1_5  (official, 11 nodes) - "Text to Audio (ACE-Step 1.5)"
    - text_to_image  (official, 9 nodes) - "Text to Image"
    - text_to_image_anima  (official, 8 nodes) - "Text to Image (Anima)"
    - text_to_image_anima_base_1_0  (official, 8 nodes) - "Text to Image (Anima Base 1.0)"
    - text_to_image_flux_1_dev  (official, 8 nodes) - "Text to Image (Flux.1 Dev)"
    - text_to_image_flux_1_krea_dev  (official, 8 nodes) - "Text to Image (Flux.1 Krea Dev)"
    - text_to_image_z_image_base  (official, 10 nodes) - "Text to Image (Z-Image-Base)"
    - text_to_image_z_image_turbo  (official, 9 nodes) - "Text to Image (Z-Image-Turbo)"
- shared node classes (3): KSampler, UNETLoader, VAELoader
- shared connection patterns (3):
    - clip_loader -> text_encode  [CLIP]
    - sampler -> vae_decode  [LATENT]
    - vae_loader -> vae_decode  [VAE]

## Type 3  -  12 member(s)  -  source: official
- cohesion (mean intra-similarity): 0.649
- official categories: Image Tools (12)  [pure]
- members:
    - brightness_and_contrast  (official, 3 nodes) - "Brightness and Contrast"
    - chromatic_aberration  (official, 3 nodes) - "Chromatic Aberration"
    - color_adjustment  (official, 5 nodes) - "Color Adjustment"
    - color_balance  (official, 11 nodes) - "Color Balance"
    - edge_preserving_blur  (official, 4 nodes) - "Edge-Preserving Blur"
    - film_grain  (official, 6 nodes) - "Film Grain"
    - glow  (official, 6 nodes) - "Glow"
    - hue_and_saturation  (official, 7 nodes) - "Hue and Saturation"
    - image_blur  (official, 3 nodes) - "Image Blur"
    - image_levels  (official, 7 nodes) - "Image Levels"
    - sharpen  (official, 2 nodes) - "Sharpen"
    - unsharp_mask  (official, 4 nodes) - "Unsharp Mask"
- shared node classes (2): GLSLShader, PrimitiveFloat
- shared connection patterns (1):
    - other -> other  [FLOAT]

## Type 4  -  9 member(s)  -  source: official
- cohesion (mean intra-similarity): 0.422
- official categories: Image generation and editing (8), Image Editing (1)  [MIXED categories]
- members:
    - image_edit  (official, 16 nodes) - "Image Edit"
    - image_edit_firered_image_edit_1_1  (official, 20 nodes) - "Image Edit (FireRed Image Edit 1.1)"
    - image_edit_longcat_image_edit  (official, 13 nodes) - "Image Edit (LongCat Image Edit)"
    - image_edit_qwen_2509  (official, 21 nodes) - "Image Edit (Qwen 2509)"
    - image_edit_qwen_2511  (official, 15 nodes) - "Image Edit (Qwen 2511)"
    - image_to_layers_qwen_image_layered  (official, 14 nodes) - "Image to Layers (Qwen-Image-Layered)"
    - image_upscale_z_image_turbo  (official, 13 nodes) - "Image Upscale (Z-image-Turbo)"
    - text_to_image_qwen_image  (official, 19 nodes) - "Text to Image (Qwen-Image)"
    - text_to_image_qwen_image_2512  (official, 18 nodes) - "Text to Image (Qwen-Image 2512)"
- shared node classes (5): CLIPLoader, KSampler, UNETLoader, VAEDecode, VAELoader
- shared connection patterns (2):
    - clip_loader -> text_encode  [CLIP]
    - vae_loader -> vae_decode  [VAE]

## Type 5  -  9 member(s)  -  source: mixed
- cohesion (mean intra-similarity): 0.622
- official categories: Video generation and editing (7)  [pure]
- members:
    - video_ltx2_3_flf2v  (custom, 35 nodes)
    - video_ltx2_3_i2v  (custom, 49 nodes)
    - canny_to_video_ltx_2_0  (official, 39 nodes) - "Canny to Video (LTX 2.0)"
    - depth_to_video_ltx_2_0  (official, 55 nodes) - "Depth to Video (LTX 2.0)"
    - first_last_frame_to_video  (official, 32 nodes) - "First-Last-Frame to Video"
    - first_last_frame_to_video_ltx_2_3  (official, 32 nodes) - "First-Last-Frame to Video (LTX-2.3)"
    - image_to_video_ltx_2_3  (official, 45 nodes) - "Image to Video (LTX-2.3)"
    - pose_to_video_ltx_2_0  (official, 40 nodes) - "Pose to Video (LTX 2.0)"
    - text_to_video_ltx_2_3  (official, 46 nodes) - "Text to Video (LTX-2.3)"
- shared node classes (18): CFGGuider, CLIPTextEncode, CheckpointLoaderSimple, CreateVideo, EmptyLTXVLatentVideo, LTXAVTextEncoderLoader, LTXVAudioVAEDecode, LTXVAudioVAELoader, LTXVConcatAVLatent, LTXVConditioning, LTXVCropGuides, LTXVEmptyLatentAudio, LTXVSeparateAVLatent, ManualSigmas, PrimitiveInt, RandomNoise, SamplerCustomAdvanced, VAEDecodeTiled
- shared connection patterns (22):
    - conditioning_op -> other  [CONDITIONING]
    - guidance -> sampler  [GUIDER]
    - guidance -> sampler  [SIGMAS]
    - latent_source -> other  [LATENT]
    - model_loader -> other  [VAE]
    - other -> conditioning_op  [FLOAT]
    - other -> guidance  [CONDITIONING]
    - other -> latent_source  [INT]
    - other -> other  [FLOAT]
    - other -> other  [INT]
    - other -> other  [LATENT]
    - other -> sampler  [LATENT]
    - other -> sampler  [NOISE]
    - other -> vae_decode  [LATENT]
    - sampler -> other  [LATENT]
    - sampler -> sampler  [SAMPLER]
    - text_encode -> conditioning_op  [CONDITIONING]
    - text_encode -> text_encode  [CLIP]
    - vae_decode -> other  [AUDIO]
    - vae_decode -> other  [IMAGE]
    - vae_loader -> latent_source  [VAE]
    - vae_loader -> vae_decode  [VAE]

## Type 6  -  8 member(s)  -  source: custom
- cohesion (mean intra-similarity): 0.499
- members:
    - Kling3_multiShot  (custom, 4 nodes)
    - api_kling_o3_flf2v  (custom, 7 nodes)
    - api_kling_o3_i2v  (custom, 5 nodes)
    - api_kling_o3_video_edit  (custom, 5 nodes)
    - api_ltxv_image_to_video  (custom, 4 nodes)
    - api_ltxv_text_to_video  (custom, 3 nodes)
    - api_wan2_6_i2v  (custom, 4 nodes)
    - api_wan2_6_t2v  (custom, 3 nodes)
- shared node classes (2): GetVideoComponents, VHS_VideoCombine
- shared connection patterns (3):
    - other -> save_output  [AUDIO]
    - other -> save_output  [FLOAT]
    - other -> save_output  [IMAGE]

## Type 7  -  6 member(s)  -  source: custom
- cohesion (mean intra-similarity): 0.478
- members:
    - NanoBanana2_outpaintUpscale  (custom, 3 nodes)
    - NanoBananaPro_3x3CharacterSheet  (custom, 5 nodes)
    - NanoBananaPro_3x3CharacterSheet_closeups  (custom, 5 nodes)
    - api_magnific_image_relight  (custom, 4 nodes)
    - api_topaz_image_enhance  (custom, 3 nodes)
    - api_veo3  (custom, 3 nodes)
- shared node classes (1): LoadImage
- shared connection patterns (1):
    - image_loader -> api_node  [IMAGE]

## Type 8  -  6 member(s)  -  source: mixed
- cohesion (mean intra-similarity): 0.474
- official categories: Image generation and editing (4)  [pure]
- members:
    - image_flux2_klein_image_edit_9b_distilled  (custom, 19 nodes)
    - image_flux2_klein_text_to_image  (custom, 16 nodes)
    - image_edit_flux_2_dev  (official, 21 nodes) - "Image Edit (Flux.2 Dev)"
    - image_edit_flux_2_klein_4b  (official, 17 nodes) - "Image Edit (Flux.2 Klein 4B)"
    - text_to_image_flux_2_dev  (official, 18 nodes) - "Text to Image (Flux.2 Dev)"
    - text_to_image_ideogram_v4  (official, 27 nodes) - "Text to Image (Ideogram v4)"
- shared node classes (9): CLIPLoader, CLIPTextEncode, EmptyFlux2LatentImage, KSamplerSelect, RandomNoise, SamplerCustomAdvanced, UNETLoader, VAEDecode, VAELoader
- shared connection patterns (8):
    - clip_loader -> text_encode  [CLIP]
    - guidance -> sampler  [GUIDER]
    - other -> other  [INT]
    - other -> sampler  [LATENT]
    - other -> sampler  [NOISE]
    - sampler -> sampler  [SAMPLER]
    - sampler -> vae_decode  [LATENT]
    - vae_loader -> vae_decode  [VAE]

## Type 9  -  4 member(s)  -  source: custom
- cohesion (mean intra-similarity): 0.636
- members:
    - api_bytedance_seedream_5_0_lite_image_edit  (custom, 5 nodes)
    - api_kling_o3_image  (custom, 5 nodes)
    - imageEdit_nano_banana2  (custom, 6 nodes)
    - imageEdit_nano_banana_pro  (custom, 6 nodes)
- shared node classes (2): LoadImage, SaveImage
- shared connection patterns (3):
    - api_node -> save_output  [IMAGE]
    - image_loader -> other  [IMAGE]
    - other -> api_node  [IMAGE]

## Type 10  -  4 member(s)  -  source: custom
- cohesion (mean intra-similarity): 0.454
- members:
    - api_magnific_image_upscale_creative  (custom, 3 nodes)
    - api_magnific_image_upscale_precise  (custom, 3 nodes)
    - upscale_ultimateSD  (custom, 7 nodes)
    - upscale_using_model  (custom, 4 nodes)
- shared node classes (2): LoadImage, SaveImage
- shared connection patterns (2):
    - image_loader -> upscale  [IMAGE]
    - upscale -> save_output  [IMAGE]

## Type 11  -  4 member(s)  -  source: official
- cohesion (mean intra-similarity): 0.608
- official categories: Audio (2), Image generation and editing (2)  [MIXED categories]
- members:
    - audio_generation_stable_audio_3_medium  (official, 21 nodes) - "Audio Generation (Stable Audio 3 Medium)"
    - audio_generation_stable_audio_3_medium_base  (official, 21 nodes) - "Audio Generation (Stable Audio 3 Medium Base)"
    - text_to_image_ernie_image  (official, 19 nodes) - "Text to Image (Ernie Image)"
    - text_to_image_ernie_image_turbo  (official, 19 nodes) - "Text to Image (Ernie Image Turbo)"
- shared node classes (9): CLIPLoader, CLIPTextEncode, ComfySwitchNode, KSampler, PreviewAny, PrimitiveBoolean, PrimitiveStringMultiline, StringReplace, TextGenerate
- shared connection patterns (8):
    - clip_loader -> other  [CLIP]
    - clip_loader -> text_encode  [CLIP]
    - model_loader -> sampler  [MODEL]
    - other -> other  [BOOLEAN]
    - other -> other  [STRING]
    - other -> text_encode  [STRING]
    - sampler -> vae_decode  [LATENT]
    - text_encode -> sampler  [CONDITIONING]

## Type 12  -  4 member(s)  -  source: official
- cohesion (mean intra-similarity): 0.532
- official categories: Image Tools (3), Video Tools (1)  [MIXED categories]
- members:
    - crop_images_2x2  (official, 15 nodes) - "Crop Images 2x2"
    - crop_images_3x3  (official, 27 nodes) - "Crop Images 3x3"
    - get_any_video_frame  (official, 5 nodes) - "Get Any Video Frame"
    - split_image_grid_to_tiles  (official, 6 nodes) - "Split Image Grid to Tiles"
- shared node classes (3): ComfyMathExpression, GetImageSize, PrimitiveInt
- shared connection patterns (1):
    - other -> other  [INT]

## Type 13  -  4 member(s)  -  source: official
- cohesion (mean intra-similarity): 0.675
- official categories: Image Tools (2), Text Tools (1), Video Tools (1)  [MIXED categories]
- members:
    - image_captioning_gemini  (official, 1 nodes) - "Image Captioning(Gemini)"
    - image_channels  (official, 1 nodes) - "Image Channels"
    - prompt_enhance  (official, 1 nodes) - "Prompt Enhance"
    - video_captioning_gemini  (official, 1 nodes) - "Video Captioning(Gemini)"
- shared node classes (0): (none)
- shared connection patterns (0):

## Type 14  -  3 member(s)  -  source: custom
- cohesion (mean intra-similarity): 0.522
- members:
    - api_meshy_image_to_model  (custom, 4 nodes)
    - api_meshy_multi_image_to_model  (custom, 6 nodes)
    - api_meshy_text_to_model  (custom, 3 nodes)
- shared node classes (1): SaveGLB
- shared connection patterns (2):
    - api_node -> other  [FILE_3D_FBX]
    - api_node -> other  [FILE_3D_GLB]

## Type 15  -  3 member(s)  -  source: official
- cohesion (mean intra-similarity): 0.483
- official categories: Video Tools (3)  [pure]
- members:
    - frame_interpolation  (official, 8 nodes) - "Frame Interpolation"
    - merge_videos  (official, 12 nodes) - "Merge Videos"
    - video_stitch  (official, 8 nodes) - "Video Stitch"
- shared node classes (2): CreateVideo, GetVideoComponents
- shared connection patterns (4):
    - other -> other  [AUDIO]
    - other -> other  [FLOAT]
    - other -> other  [IMAGE]
    - other -> other  [INT]

## Type 16  -  3 member(s)  -  source: official
- cohesion (mean intra-similarity): 0.877
- official categories: Conditioning & Preprocessors (2), 3D (1)  [MIXED categories]
- members:
    - geometry_estimation_moge  (official, 10 nodes) - "Geometry Estimation (MoGe)"
    - image_depth_estimation_moge  (official, 11 nodes) - "Image Depth Estimation (MoGe)"
    - video_depth_estimation_moge  (official, 12 nodes) - "Video Depth Estimation (MoGe)"
- shared node classes (7): ComfyMathExpression, ComfySwitchNode, GetImageSize, LoadMoGeModel, MoGeInference, MoGeRender, ResizeImagesByLongerEdge
- shared connection patterns (5):
    - other -> other  [BOOLEAN]
    - other -> other  [IMAGE]
    - other -> other  [INT]
    - other -> other  [MOGE_GEOMETRY]
    - other -> other  [MOGE_MODEL]

## Type 17  -  3 member(s)  -  source: official
- cohesion (mean intra-similarity): 0.498
- official categories: Video generation and editing (2), Image generation and editing (1)  [MIXED categories]
- members:
    - image_edit_bernini_r  (official, 35 nodes) - "Image Edit (Bernini-R)"
    - video_edit_bernini_r  (official, 37 nodes) - "Video Edit (Bernini-R)"
    - video_inpaint_void  (official, 39 nodes) - "Video Inpaint (VOID)"
- shared node classes (9): BasicScheduler, CLIPLoader, CLIPTextEncode, ComfySwitchNode, PrimitiveBoolean, PrimitiveInt, UNETLoader, VAEDecode, VAELoader
- shared connection patterns (13):
    - clip_loader -> text_encode  [CLIP]
    - conditioning_op -> sampler  [LATENT]
    - guidance -> sampler  [SIGMAS]
    - model_loader -> guidance  [MODEL]
    - model_loader -> other  [MODEL]
    - other -> other  [BOOLEAN]
    - other -> other  [FLOAT]
    - other -> other  [INT]
    - sampler -> sampler  [SAMPLER]
    - sampler -> vae_decode  [LATENT]
    - text_encode -> conditioning_op  [CONDITIONING]
    - vae_loader -> conditioning_op  [VAE]
    - vae_loader -> vae_decode  [VAE]

## Type 18  -  3 member(s)  -  source: official
- cohesion (mean intra-similarity): 0.716
- official categories: Conditioning & Preprocessors (3)  [pure]
- members:
    - image_to_pose_map_sdpose_multi_person  (official, 6 nodes) - "Image to Pose Map (SDPose Multi-Person)"
    - image_to_pose_map_sdpose_ood  (official, 4 nodes) - "Image to Pose Map (SDPose-OOD)"
    - video_to_pose_map_sdpose_multi_person  (official, 7 nodes) - "Video to Pose Map (SDPose Multi-Person)"
- shared node classes (4): CheckpointLoaderSimple, ResizeImageMaskNode, SDPoseDrawKeypoints, SDPoseKeypointExtractor
- shared connection patterns (4):
    - model_loader -> other  [MODEL]
    - model_loader -> other  [VAE]
    - other -> other  [IMAGE]
    - other -> other  [POSE_KEYPOINT]

## Type 19  -  2 member(s)  -  source: official
- cohesion (mean intra-similarity): 0.615
- official categories: Conditioning & Preprocessors (2)  [pure]
- members:
    - image_depth_estimation_depth_anything_3  (official, 3 nodes) - "Image Depth Estimation (Depth Anything 3)"
    - video_depth_estimation_depth_anything_3  (official, 5 nodes) - "Video Depth Estimation (Depth Anything 3)"
- shared node classes (3): DA3Inference, DA3Render, LoadDA3Model
- shared connection patterns (2):
    - other -> other  [DA3_GEOMETRY]
    - other -> other  [DA3_MODEL]

## Type 20  -  2 member(s)  -  source: official
- cohesion (mean intra-similarity): 1.0
- official categories: Conditioning & Preprocessors (1), Image generation and editing (1)  [MIXED categories]
- members:
    - image_depth_estimation_lotus_depth  (official, 12 nodes) - "Image Depth Estimation (Lotus Depth)"
    - image_to_depth_map_lotus  (official, 12 nodes) - "Image to Depth Map (Lotus)"
- shared node classes (12): BasicGuider, BasicScheduler, DisableNoise, ImageInvert, KSamplerSelect, LotusConditioning, SamplerCustomAdvanced, SetFirstSigma, UNETLoader, VAEDecode, VAEEncode, VAELoader
- shared connection patterns (12):
    - conditioning_op -> guidance  [CONDITIONING]
    - guidance -> other  [SIGMAS]
    - guidance -> sampler  [GUIDER]
    - model_loader -> guidance  [MODEL]
    - other -> sampler  [NOISE]
    - other -> sampler  [SIGMAS]
    - sampler -> sampler  [SAMPLER]
    - sampler -> vae_decode  [LATENT]
    - vae_decode -> other  [IMAGE]
    - vae_encode -> sampler  [LATENT]
    - vae_loader -> vae_decode  [VAE]
    - vae_loader -> vae_encode  [VAE]

## Type 21  -  2 member(s)  -  source: official
- cohesion (mean intra-similarity): 0.575
- official categories: Conditioning & Preprocessors (2)  [pure]
- members:
    - image_face_detection_mediapipe  (official, 3 nodes) - "Image Face Detection (Mediapipe)"
    - video_face_detection_mediapipe  (official, 6 nodes) - "Video Face Detection (Mediapipe)"
- shared node classes (3): LoadMediaPipeFaceLandmarker, MediaPipeFaceLandmarker, MediaPipeFaceMask
- shared connection patterns (2):
    - other -> other  [FACE_DETECTION_MODEL]
    - other -> other  [FACE_LANDMARKS]

## Type 22  -  2 member(s)  -  source: official
- cohesion (mean intra-similarity): 0.615
- official categories: Image generation and editing (2)  [pure]
- members:
    - image_inpainting_qwen_image  (official, 21 nodes) - "Image Inpainting (Qwen-image)"
    - image_outpainting_qwen_image  (official, 25 nodes) - "Image Outpainting (Qwen-Image)"
- shared node classes (16): CLIPLoader, CLIPTextEncode, ControlNetInpaintingAliMamaApply, ControlNetLoader, GrowMask, ImageBlur, ImageToMask, KSampler, LoraLoaderModelOnly, MaskPreview, MaskToImage, ModelSamplingAuraFlow, UNETLoader, VAEDecode, VAEEncode, VAELoader
- shared connection patterns (16):
    - clip_loader -> text_encode  [CLIP]
    - controlnet -> controlnet  [CONTROL_NET]
    - controlnet -> sampler  [CONDITIONING]
    - lora_loader -> other  [MODEL]
    - model_loader -> lora_loader  [MODEL]
    - other -> controlnet  [MASK]
    - other -> other  [IMAGE]
    - other -> other  [MASK]
    - other -> sampler  [MODEL]
    - sampler -> vae_decode  [LATENT]
    - text_encode -> controlnet  [CONDITIONING]
    - upscale -> controlnet  [IMAGE]
    - upscale -> vae_encode  [IMAGE]
    - vae_loader -> controlnet  [VAE]
    - vae_loader -> vae_decode  [VAE]
    - vae_loader -> vae_encode  [VAE]

## Type 23  -  2 member(s)  -  source: official
- cohesion (mean intra-similarity): 0.619
- official categories: Conditioning & Preprocessors (2)  [pure]
- members:
    - image_segmentation_sam3  (official, 3 nodes) - "Image Segmentation (SAM3)"
    - video_segmentation_sam3  (official, 5 nodes) - "Video Segmentation (SAM3)"
- shared node classes (3): CLIPTextEncode, CheckpointLoaderSimple, SAM3_Detect
- shared connection patterns (3):
    - model_loader -> other  [MODEL]
    - model_loader -> text_encode  [CLIP]
    - text_encode -> other  [CONDITIONING]

## Type 24  -  1 member(s)  -  source: custom
- cohesion (mean intra-similarity): 1.0
- members:
    - api_ideogram_v3_t2i  (custom, 2 nodes)

## Type 25  -  1 member(s)  -  source: custom
- cohesion (mean intra-similarity): 1.0
- members:
    - api_topaz_video_enhance  (custom, 3 nodes)

## Type 26  -  1 member(s)  -  source: official
- cohesion (mean intra-similarity): 1.0
- official categories: Image Tools (1)  [pure]
- members:
    - color_curves  (official, 6 nodes) - "Color Curves"

## Type 27  -  1 member(s)  -  source: official
- cohesion (mean intra-similarity): 1.0
- official categories: 3D (1)  [pure]
- members:
    - image_to_gaussian_splat_triposplat  (official, 17 nodes) - "Image to Gaussian Splat (TripoSplat)"

## Type 28  -  1 member(s)  -  source: official
- cohesion (mean intra-similarity): 1.0
- official categories: 3D (1)  [pure]
- members:
    - image_to_model_hunyuan3d_2_1  (official, 8 nodes) - "Image to 3D Model (Hunyuan3d 2.1)"

## Type 29  -  1 member(s)  -  source: custom
- cohesion (mean intra-similarity): 1.0
- members:
    - qwen2511_imageEdit  (custom, 17 nodes)

## Type 30  -  1 member(s)  -  source: official
- cohesion (mean intra-similarity): 1.0
- official categories: Image Tools (1)  [pure]
- members:
    - remove_background_birefnet  (official, 4 nodes) - "Remove Background (BiRefNet)"

## Type 31  -  1 member(s)  -  source: official
- cohesion (mean intra-similarity): 1.0
- official categories: Text Tools (1)  [pure]
- members:
    - select_per_line_text_by_index  (official, 4 nodes) - "Select Per-Line Text by Index"

## Type 32  -  1 member(s)  -  source: custom
- cohesion (mean intra-similarity): 1.0
- members:
    - styletransfer_NanoBananaPro  (custom, 18 nodes)

## Type 33  -  1 member(s)  -  source: official
- cohesion (mean intra-similarity): 1.0
- official categories: Image generation and editing (1)  [pure]
- members:
    - text_to_image_netayume_lumina  (official, 14 nodes) - "Text to Image (NetaYume Lumina)"

## Type 34  -  1 member(s)  -  source: custom
- cohesion (mean intra-similarity): 1.0
- members:
    - video_gemini_motionPromptGeneration  (custom, 6 nodes)

## Type 35  -  1 member(s)  -  source: official
- cohesion (mean intra-similarity): 1.0
- official categories: Video generation and editing (1)  [pure]
- members:
    - video_upscale_gan_x4  (official, 4 nodes) - "Video Upscale(GAN x4)"
