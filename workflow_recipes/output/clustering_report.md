# Clustering report

- Clustering basis: **structural** (node-graph fingerprints)
- Workflows: 137
- Types (clusters): 68  (36 multi-member, 32 singletons)
- Similarity threshold: 0.55
- Signal weights: {'classes': 0.4, 'connections': 0.35, 'clusters': 0.2, 'spine': 0.05, 'category': 0.0}

## Type 1  -  11 member(s)  -  source: official
- cohesion (mean intra-similarity): 0.684
- official categories: Image Tools (11)  [pure]
- members:
    - brightness_and_contrast  (official, 3 nodes) - "Brightness and Contrast"
    - chromatic_aberration  (official, 3 nodes) - "Chromatic Aberration"
    - color_adjustment  (official, 5 nodes) - "Color Adjustment"
    - edge_preserving_blur  (official, 4 nodes) - "Edge-Preserving Blur"
    - film_grain  (official, 6 nodes) - "Film Grain"
    - glow  (official, 6 nodes) - "Glow"
    - hue_and_saturation  (official, 7 nodes) - "Hue and Saturation"
    - image_blur  (official, 3 nodes) - "Image Blur"
    - image_levels  (official, 7 nodes) - "Image Levels"
    - sharpen  (official, 2 nodes) - "Sharpen"
    - unsharp_mask  (official, 4 nodes) - "Unsharp Mask"
- shared node classes (2): GLSLShader, PrimitiveFloat

## Type 2  -  6 member(s)  -  source: mixed
- cohesion (mean intra-similarity): 0.716
- official categories: Image generation and editing (5)  [pure]
- members:
    - image_z_image_turbo  (custom, 10 nodes)
    - text_to_image  (official, 9 nodes) - "Text to Image"
    - text_to_image_flux_1_dev  (official, 8 nodes) - "Text to Image (Flux.1 Dev)"
    - text_to_image_flux_1_krea_dev  (official, 8 nodes) - "Text to Image (Flux.1 Krea Dev)"
    - text_to_image_z_image_base  (official, 10 nodes) - "Text to Image (Z-Image-Base)"
    - text_to_image_z_image_turbo  (official, 9 nodes) - "Text to Image (Z-Image-Turbo)"
- shared node classes (6): CLIPTextEncode, EmptySD3LatentImage, KSampler, UNETLoader, VAEDecode, VAELoader

## Type 3  -  6 member(s)  -  source: mixed
- cohesion (mean intra-similarity): 0.678
- official categories: Video generation and editing (4)  [pure]
- members:
    - video_ltx2_3_flf2v  (custom, 35 nodes)
    - video_ltx2_3_i2v  (custom, 49 nodes)
    - first_last_frame_to_video  (official, 32 nodes) - "First-Last-Frame to Video"
    - first_last_frame_to_video_ltx_2_3  (official, 32 nodes) - "First-Last-Frame to Video (LTX-2.3)"
    - image_to_video_ltx_2_3  (official, 45 nodes) - "Image to Video (LTX-2.3)"
    - text_to_video_ltx_2_3  (official, 46 nodes) - "Text to Video (LTX-2.3)"
- shared node classes (21): CFGGuider, CLIPTextEncode, CheckpointLoaderSimple, ComfyMathExpression, CreateVideo, EmptyLTXVLatentVideo, LTXAVTextEncoderLoader, LTXVAudioVAEDecode, LTXVAudioVAELoader, LTXVConcatAVLatent, LTXVConditioning, LTXVCropGuides, LTXVEmptyLatentAudio, LTXVPreprocess, LTXVSeparateAVLatent, ManualSigmas, PrimitiveInt, RandomNoise, ResizeImageMaskNode, SamplerCustomAdvanced, VAEDecodeTiled

## Type 4  -  5 member(s)  -  source: custom
- cohesion (mean intra-similarity): 0.674
- members:
    - Kling3_multiShot  (custom, 4 nodes)
    - api_ltxv_image_to_video  (custom, 4 nodes)
    - api_ltxv_text_to_video  (custom, 3 nodes)
    - api_wan2_6_i2v  (custom, 4 nodes)
    - api_wan2_6_t2v  (custom, 3 nodes)
- shared node classes (2): GetVideoComponents, VHS_VideoCombine

## Type 5  -  4 member(s)  -  source: official
- cohesion (mean intra-similarity): 0.701
- official categories: Image generation and editing (4)  [pure]
- members:
    - image_edit_firered_image_edit_1_1  (official, 20 nodes) - "Image Edit (FireRed Image Edit 1.1)"
    - image_edit_qwen_2509  (official, 21 nodes) - "Image Edit (Qwen 2509)"
    - text_to_image_qwen_image  (official, 19 nodes) - "Text to Image (Qwen-Image)"
    - text_to_image_qwen_image_2512  (official, 18 nodes) - "Text to Image (Qwen-Image 2512)"
- shared node classes (11): CLIPLoader, ComfySwitchNode, KSampler, LoraLoaderModelOnly, ModelSamplingAuraFlow, PrimitiveBoolean, PrimitiveFloat, PrimitiveInt, UNETLoader, VAEDecode, VAELoader

## Type 6  -  4 member(s)  -  source: mixed
- cohesion (mean intra-similarity): 0.789
- official categories: Image generation and editing (3)  [pure]
- members:
    - image_z_image_turbo_fun_union_controlnet  (custom, 15 nodes)
    - canny_to_image_z_image_turbo  (official, 15 nodes) - "Canny to Image (Z-Image-Turbo)"
    - controlnet_z_image_turbo  (official, 12 nodes) - "ControlNet (Z-Image-Turbo)"
    - pose_to_image_z_image_turbo  (official, 12 nodes) - "Pose to Image (Z-Image-Turbo)"
- shared node classes (12): CLIPLoader, CLIPTextEncode, ConditioningZeroOut, EmptySD3LatentImage, GetImageSize, KSampler, ModelPatchLoader, ModelSamplingAuraFlow, QwenImageDiffsynthControlnet, UNETLoader, VAEDecode, VAELoader

## Type 7  -  4 member(s)  -  source: custom
- cohesion (mean intra-similarity): 0.745
- members:
    - video_wan_vace_14B_ref2v  (custom, 15 nodes)
    - video_wan_vace_14B_v2v  (custom, 17 nodes)
    - video_wan_vace_flf2v  (custom, 32 nodes)
    - video_wan_vace_outpainting  (custom, 25 nodes)
- shared node classes (13): CLIPLoader, CLIPTextEncode, CreateVideo, GetVideoComponents, KSampler, LoraLoader, ModelSamplingSD3, TrimVideoLatent, UNETLoader, VAEDecode, VAELoader, VHS_VideoCombine, WanVaceToVideo

## Type 8  -  3 member(s)  -  source: custom
- cohesion (mean intra-similarity): 0.733
- members:
    - NanoBanana2_outpaintUpscale  (custom, 3 nodes)
    - api_magnific_image_relight  (custom, 4 nodes)
    - api_topaz_image_enhance  (custom, 3 nodes)
- shared node classes (2): LoadImage, SaveImage

## Type 9  -  3 member(s)  -  source: official
- cohesion (mean intra-similarity): 0.798
- official categories: Video generation and editing (3)  [pure]
- members:
    - canny_to_video_ltx_2_0  (official, 39 nodes) - "Canny to Video (LTX 2.0)"
    - depth_to_video_ltx_2_0  (official, 55 nodes) - "Depth to Video (LTX 2.0)"
    - pose_to_video_ltx_2_0  (official, 40 nodes) - "Pose to Video (LTX 2.0)"
- shared node classes (30): CFGGuider, CLIPTextEncode, CheckpointLoaderSimple, CreateVideo, EmptyLTXVLatentVideo, GetImageSize, KSamplerSelect, LTXAVTextEncoderLoader, LTXVAddGuide, LTXVAudioVAEDecode, LTXVAudioVAELoader, LTXVConcatAVLatent, LTXVConditioning, LTXVCropGuides, LTXVEmptyLatentAudio, LTXVImgToVideoInplace, LTXVLatentUpsampler, LTXVScheduler, LTXVSeparateAVLatent, LatentUpscaleModelLoader, LoraLoaderModelOnly, ManualSigmas, MarkdownNote, PrimitiveFloat, PrimitiveInt, RandomNoise, Reroute, SamplerCustomAdvanced, VAEDecode, VAEDecodeTiled

## Type 10  -  3 member(s)  -  source: official
- cohesion (mean intra-similarity): 0.877
- official categories: Conditioning & Preprocessors (2), 3D (1)  [MIXED categories]
- members:
    - geometry_estimation_moge  (official, 10 nodes) - "Geometry Estimation (MoGe)"
    - image_depth_estimation_moge  (official, 11 nodes) - "Image Depth Estimation (MoGe)"
    - video_depth_estimation_moge  (official, 12 nodes) - "Video Depth Estimation (MoGe)"
- shared node classes (7): ComfyMathExpression, ComfySwitchNode, GetImageSize, LoadMoGeModel, MoGeInference, MoGeRender, ResizeImagesByLongerEdge

## Type 11  -  3 member(s)  -  source: official
- cohesion (mean intra-similarity): 1.0
- official categories: Image Tools (1), Text Tools (1), Video Tools (1)  [MIXED categories]
- members:
    - image_captioning_gemini  (official, 1 nodes) - "Image Captioning(Gemini)"
    - prompt_enhance  (official, 1 nodes) - "Prompt Enhance"
    - video_captioning_gemini  (official, 1 nodes) - "Video Captioning(Gemini)"
- shared node classes (1): GeminiNode

## Type 12  -  3 member(s)  -  source: official
- cohesion (mean intra-similarity): 0.716
- official categories: Conditioning & Preprocessors (3)  [pure]
- members:
    - image_to_pose_map_sdpose_multi_person  (official, 6 nodes) - "Image to Pose Map (SDPose Multi-Person)"
    - image_to_pose_map_sdpose_ood  (official, 4 nodes) - "Image to Pose Map (SDPose-OOD)"
    - video_to_pose_map_sdpose_multi_person  (official, 7 nodes) - "Video to Pose Map (SDPose Multi-Person)"
- shared node classes (4): CheckpointLoaderSimple, ResizeImageMaskNode, SDPoseDrawKeypoints, SDPoseKeypointExtractor

## Type 13  -  3 member(s)  -  source: official
- cohesion (mean intra-similarity): 0.757
- official categories: Video generation and editing (3)  [pure]
- members:
    - image_to_video  (official, 15 nodes) - "Image to Video"
    - image_to_video_wan_2_2  (official, 19 nodes) - "Image to Video (Wan 2.2)"
    - text_to_video_wan_2_2  (official, 17 nodes) - "Text to Video (Wan 2.2)"
- shared node classes (9): CLIPLoader, CLIPTextEncode, CreateVideo, KSamplerAdvanced, LoraLoaderModelOnly, ModelSamplingSD3, UNETLoader, VAEDecode, VAELoader

## Type 14  -  3 member(s)  -  source: custom
- cohesion (mean intra-similarity): 0.808
- members:
    - video_wan2_2_14B_flf2v  (custom, 17 nodes)
    - video_wan2_2_14B_fun_camera  (custom, 17 nodes)
    - video_wan2_2_14B_fun_control  (custom, 18 nodes)
- shared node classes (11): CLIPLoader, CLIPTextEncode, CreateVideo, GetVideoComponents, KSamplerAdvanced, LoadImage, ModelSamplingSD3, UNETLoader, VAEDecode, VAELoader, VHS_VideoCombine

## Type 15  -  2 member(s)  -  source: custom
- cohesion (mean intra-similarity): 1.0
- members:
    - NanoBananaPro_3x3CharacterSheet  (custom, 5 nodes)
    - NanoBananaPro_3x3CharacterSheet_closeups  (custom, 5 nodes)
- shared node classes (5): GeminiImage2Node, GeminiNode, LoadImage, PrimitiveStringMultiline, SaveImage

## Type 16  -  2 member(s)  -  source: custom
- cohesion (mean intra-similarity): 0.84
- members:
    - api_bytedance_seedream_5_0_lite_image_edit  (custom, 5 nodes)
    - api_kling_o3_image  (custom, 5 nodes)
- shared node classes (3): ImageBatchMulti, LoadImage, SaveImage

## Type 17  -  2 member(s)  -  source: custom
- cohesion (mean intra-similarity): 0.731
- members:
    - api_kling_o3_flf2v  (custom, 7 nodes)
    - api_kling_o3_i2v  (custom, 5 nodes)
- shared node classes (4): GetVideoComponents, ImageBatchMulti, LoadImage, VHS_VideoCombine

## Type 18  -  2 member(s)  -  source: custom
- cohesion (mean intra-similarity): 0.8
- members:
    - api_magnific_image_upscale_creative  (custom, 3 nodes)
    - api_magnific_image_upscale_precise  (custom, 3 nodes)
- shared node classes (2): LoadImage, SaveImage

## Type 19  -  2 member(s)  -  source: custom
- cohesion (mean intra-similarity): 0.7
- members:
    - api_meshy_image_to_model  (custom, 4 nodes)
    - api_meshy_multi_image_to_model  (custom, 6 nodes)
- shared node classes (2): LoadImage, SaveGLB

## Type 20  -  2 member(s)  -  source: official
- cohesion (mean intra-similarity): 1.0
- official categories: Audio (2)  [pure]
- members:
    - audio_generation_stable_audio_3_medium  (official, 21 nodes) - "Audio Generation (Stable Audio 3 Medium)"
    - audio_generation_stable_audio_3_medium_base  (official, 21 nodes) - "Audio Generation (Stable Audio 3 Medium Base)"
- shared node classes (16): CLIPLoader, CLIPTextEncode, CheckpointLoaderSimple, ComfyMathExpression, ComfySwitchNode, CustomCombo, EmptyLatentAudio, JsonExtractString, KSampler, PreviewAny, PrimitiveBoolean, PrimitiveFloat, PrimitiveStringMultiline, StringReplace, TextGenerate, VAEDecodeAudio

## Type 21  -  2 member(s)  -  source: official
- cohesion (mean intra-similarity): 0.947
- official categories: Video generation and editing (2)  [pure]
- members:
    - character_replacement_scail_2_base  (official, 42 nodes) - "Character Replacement (SCAIL-2 Base)"
    - character_replacement_scail_2_extend  (official, 45 nodes) - "Character Replacement (SCAIL-2 Extend)"
- shared node classes (26): BasicScheduler, CLIPLoader, CLIPTextEncode, CLIPVisionEncode, CLIPVisionLoader, CheckpointLoaderSimple, ComfyMathExpression, ComfySwitchNode, GetImageSize, GetVideoComponents, ImageFromBatch, KSamplerSelect, LoraLoaderModelOnly, ModelSamplingSD3, PreviewImage, PrimitiveBoolean, PrimitiveFloat, PrimitiveInt, ResizeImageMaskNode, SAM3_VideoTrack, SCAIL2ColoredMask, SamplerCustom, UNETLoader, VAEDecode, VAELoader, WanSCAILToVideo

## Type 22  -  2 member(s)  -  source: official
- cohesion (mean intra-similarity): 0.892
- official categories: Image Tools (2)  [pure]
- members:
    - crop_images_2x2  (official, 15 nodes) - "Crop Images 2x2"
    - crop_images_3x3  (official, 27 nodes) - "Crop Images 3x3"
- shared node classes (6): BatchImagesNode, ComfyMathExpression, GetImageSize, ImageCropV2, PrimitiveBoundingBox, PrimitiveInt

## Type 23  -  2 member(s)  -  source: official
- cohesion (mean intra-similarity): 0.585
- official categories: Image Tools (1), Video Tools (1)  [MIXED categories]
- members:
    - get_any_video_frame  (official, 5 nodes) - "Get Any Video Frame"
    - split_image_grid_to_tiles  (official, 6 nodes) - "Split Image Grid to Tiles"
- shared node classes (3): ComfyMathExpression, GetImageSize, PrimitiveInt

## Type 24  -  2 member(s)  -  source: custom
- cohesion (mean intra-similarity): 0.867
- members:
    - imageEdit_nano_banana2  (custom, 6 nodes)
    - imageEdit_nano_banana_pro  (custom, 6 nodes)
- shared node classes (4): AILab_ImageToList, ImageListToImageBatch, LoadImage, SaveImage

## Type 25  -  2 member(s)  -  source: official
- cohesion (mean intra-similarity): 0.615
- official categories: Conditioning & Preprocessors (2)  [pure]
- members:
    - image_depth_estimation_depth_anything_3  (official, 3 nodes) - "Image Depth Estimation (Depth Anything 3)"
    - video_depth_estimation_depth_anything_3  (official, 5 nodes) - "Video Depth Estimation (Depth Anything 3)"
- shared node classes (3): DA3Inference, DA3Render, LoadDA3Model

## Type 26  -  2 member(s)  -  source: official
- cohesion (mean intra-similarity): 1.0
- official categories: Conditioning & Preprocessors (1), Image generation and editing (1)  [MIXED categories]
- members:
    - image_depth_estimation_lotus_depth  (official, 12 nodes) - "Image Depth Estimation (Lotus Depth)"
    - image_to_depth_map_lotus  (official, 12 nodes) - "Image to Depth Map (Lotus)"
- shared node classes (12): BasicGuider, BasicScheduler, DisableNoise, ImageInvert, KSamplerSelect, LotusConditioning, SamplerCustomAdvanced, SetFirstSigma, UNETLoader, VAEDecode, VAEEncode, VAELoader

## Type 27  -  2 member(s)  -  source: official
- cohesion (mean intra-similarity): 0.835
- official categories: Image Editing (1), Image generation and editing (1)  [MIXED categories]
- members:
    - image_edit  (official, 16 nodes) - "Image Edit"
    - image_edit_qwen_2511  (official, 15 nodes) - "Image Edit (Qwen 2511)"
- shared node classes (13): CFGNorm, CLIPLoader, FluxKontextImageScale, FluxKontextMultiReferenceLatentMethod, KSampler, MarkdownNote, ModelSamplingAuraFlow, Note, TextEncodeQwenImageEditPlus, UNETLoader, VAEDecode, VAEEncode, VAELoader

## Type 28  -  2 member(s)  -  source: official
- cohesion (mean intra-similarity): 0.88
- official categories: Image generation and editing (1), Video generation and editing (1)  [MIXED categories]
- members:
    - image_edit_bernini_r  (official, 35 nodes) - "Image Edit (Bernini-R)"
    - video_edit_bernini_r  (official, 37 nodes) - "Video Edit (Bernini-R)"
- shared node classes (22): BasicScheduler, BerniniConditioning, CLIPLoader, CLIPTextEncode, ComfySwitchNode, CustomCombo, KSamplerSelect, LoraLoaderModelOnly, MarkdownNote, PreviewAny, PrimitiveBoolean, PrimitiveFloat, PrimitiveInt, PrimitiveStringMultiline, RegexExtract, SamplerCustom, SplitSigmas, StringConcatenate, StringReplace, UNETLoader, VAEDecode, VAELoader

## Type 29  -  2 member(s)  -  source: official
- cohesion (mean intra-similarity): 0.739
- official categories: Image generation and editing (2)  [pure]
- members:
    - image_edit_flux_2_dev  (official, 21 nodes) - "Image Edit (Flux.2 Dev)"
    - text_to_image_flux_2_dev  (official, 18 nodes) - "Text to Image (Flux.2 Dev)"
- shared node classes (16): BasicGuider, CLIPLoader, CLIPTextEncode, ComfySwitchNode, EmptyFlux2LatentImage, Flux2Scheduler, FluxGuidance, KSamplerSelect, LoraLoaderModelOnly, PrimitiveBoolean, PrimitiveInt, RandomNoise, SamplerCustomAdvanced, UNETLoader, VAEDecode, VAELoader

## Type 30  -  2 member(s)  -  source: official
- cohesion (mean intra-similarity): 0.575
- official categories: Conditioning & Preprocessors (2)  [pure]
- members:
    - image_face_detection_mediapipe  (official, 3 nodes) - "Image Face Detection (Mediapipe)"
    - video_face_detection_mediapipe  (official, 6 nodes) - "Video Face Detection (Mediapipe)"
- shared node classes (3): LoadMediaPipeFaceLandmarker, MediaPipeFaceLandmarker, MediaPipeFaceMask

## Type 31  -  2 member(s)  -  source: mixed
- cohesion (mean intra-similarity): 0.681
- official categories: Image generation and editing (1)  [pure]
- members:
    - image_flux2_klein_image_edit_9b_distilled  (custom, 19 nodes)
    - image_edit_flux_2_klein_4b  (official, 17 nodes) - "Image Edit (Flux.2 Klein 4B)"
- shared node classes (15): CFGGuider, CLIPLoader, CLIPTextEncode, EmptyFlux2LatentImage, Flux2Scheduler, GetImageSize, ImageScaleToTotalPixels, KSamplerSelect, RandomNoise, ReferenceLatent, SamplerCustomAdvanced, UNETLoader, VAEDecode, VAEEncode, VAELoader

## Type 32  -  2 member(s)  -  source: official
- cohesion (mean intra-similarity): 0.615
- official categories: Image generation and editing (2)  [pure]
- members:
    - image_inpainting_qwen_image  (official, 21 nodes) - "Image Inpainting (Qwen-image)"
    - image_outpainting_qwen_image  (official, 25 nodes) - "Image Outpainting (Qwen-Image)"
- shared node classes (16): CLIPLoader, CLIPTextEncode, ControlNetInpaintingAliMamaApply, ControlNetLoader, GrowMask, ImageBlur, ImageToMask, KSampler, LoraLoaderModelOnly, MaskPreview, MaskToImage, ModelSamplingAuraFlow, UNETLoader, VAEDecode, VAEEncode, VAELoader

## Type 33  -  2 member(s)  -  source: official
- cohesion (mean intra-similarity): 0.619
- official categories: Conditioning & Preprocessors (2)  [pure]
- members:
    - image_segmentation_sam3  (official, 3 nodes) - "Image Segmentation (SAM3)"
    - video_segmentation_sam3  (official, 5 nodes) - "Video Segmentation (SAM3)"
- shared node classes (3): CLIPTextEncode, CheckpointLoaderSimple, SAM3_Detect

## Type 34  -  2 member(s)  -  source: official
- cohesion (mean intra-similarity): 1.0
- official categories: Image generation and editing (2)  [pure]
- members:
    - text_to_image_anima  (official, 8 nodes) - "Text to Image (Anima)"
    - text_to_image_anima_base_1_0  (official, 8 nodes) - "Text to Image (Anima Base 1.0)"
- shared node classes (7): CLIPLoader, CLIPTextEncode, EmptyLatentImage, KSampler, UNETLoader, VAEDecode, VAELoader

## Type 35  -  2 member(s)  -  source: official
- cohesion (mean intra-similarity): 0.833
- official categories: Image generation and editing (2)  [pure]
- members:
    - text_to_image_ernie_image  (official, 19 nodes) - "Text to Image (Ernie Image)"
    - text_to_image_ernie_image_turbo  (official, 19 nodes) - "Text to Image (Ernie Image Turbo)"
- shared node classes (13): CLIPLoader, CLIPTextEncode, ComfySwitchNode, EmptyFlux2LatentImage, KSampler, PreviewAny, PrimitiveBoolean, PrimitiveStringMultiline, StringReplace, TextGenerate, UNETLoader, VAEDecode, VAELoader

## Type 36  -  2 member(s)  -  source: official
- cohesion (mean intra-similarity): 0.61
- official categories: Video generation and editing (2)  [pure]
- members:
    - video_inpaint_wan2_1_vace  (official, 26 nodes) - "Video Inpaint(Wan2.1 VACE)"
    - video_inpainting_wan2_1_vace  (official, 38 nodes) - "Video Inpainting (Wan2.1 VACE)"
- shared node classes (18): CLIPLoader, CLIPTextEncode, CreateVideo, GetImageSize, GetVideoComponents, ImageCompositeMasked, ImageFromBatch, InvertMask, KSampler, LoraLoaderModelOnly, MaskToImage, ModelSamplingSD3, PreviewImage, TrimVideoLatent, UNETLoader, VAEDecode, VAELoader, WanVaceToVideo

## Type 37  -  1 member(s)  -  source: custom
- cohesion (mean intra-similarity): 1.0
- members:
    - Wan22Vace_VID2VID  (custom, 29 nodes)

## Type 38  -  1 member(s)  -  source: custom
- cohesion (mean intra-similarity): 1.0
- members:
    - api_ideogram_v3_t2i  (custom, 2 nodes)

## Type 39  -  1 member(s)  -  source: custom
- cohesion (mean intra-similarity): 1.0
- members:
    - api_kling_o3_video_edit  (custom, 5 nodes)

## Type 40  -  1 member(s)  -  source: custom
- cohesion (mean intra-similarity): 1.0
- members:
    - api_meshy_text_to_model  (custom, 3 nodes)

## Type 41  -  1 member(s)  -  source: custom
- cohesion (mean intra-similarity): 1.0
- members:
    - api_topaz_video_enhance  (custom, 3 nodes)

## Type 42  -  1 member(s)  -  source: custom
- cohesion (mean intra-similarity): 1.0
- members:
    - api_veo3  (custom, 3 nodes)

## Type 43  -  1 member(s)  -  source: official
- cohesion (mean intra-similarity): 1.0
- official categories: Image Tools (1)  [pure]
- members:
    - color_balance  (official, 11 nodes) - "Color Balance"

## Type 44  -  1 member(s)  -  source: official
- cohesion (mean intra-similarity): 1.0
- official categories: Image Tools (1)  [pure]
- members:
    - color_curves  (official, 6 nodes) - "Color Curves"

## Type 45  -  1 member(s)  -  source: official
- cohesion (mean intra-similarity): 1.0
- official categories: Image generation and editing (1)  [pure]
- members:
    - depth_to_image_z_image_turbo  (official, 26 nodes) - "Depth to Image (Z-Image-Turbo)"

## Type 46  -  1 member(s)  -  source: official
- cohesion (mean intra-similarity): 1.0
- official categories: Video Tools (1)  [pure]
- members:
    - frame_interpolation  (official, 8 nodes) - "Frame Interpolation"

## Type 47  -  1 member(s)  -  source: official
- cohesion (mean intra-similarity): 1.0
- official categories: Image Tools (1)  [pure]
- members:
    - image_channels  (official, 1 nodes) - "Image Channels"

## Type 48  -  1 member(s)  -  source: official
- cohesion (mean intra-similarity): 1.0
- official categories: Image generation and editing (1)  [pure]
- members:
    - image_edit_longcat_image_edit  (official, 13 nodes) - "Image Edit (LongCat Image Edit)"

## Type 49  -  1 member(s)  -  source: custom
- cohesion (mean intra-similarity): 1.0
- members:
    - image_flux2_klein_text_to_image  (custom, 16 nodes)

## Type 50  -  1 member(s)  -  source: official
- cohesion (mean intra-similarity): 1.0
- official categories: Image generation and editing (1)  [pure]
- members:
    - image_inpainting_flux_1_fill_dev  (official, 10 nodes) - "Image Inpainting (Flux.1 Fill Dev)"

## Type 51  -  1 member(s)  -  source: official
- cohesion (mean intra-similarity): 1.0
- official categories: 3D (1)  [pure]
- members:
    - image_to_gaussian_splat_triposplat  (official, 17 nodes) - "Image to Gaussian Splat (TripoSplat)"

## Type 52  -  1 member(s)  -  source: official
- cohesion (mean intra-similarity): 1.0
- official categories: Image generation and editing (1)  [pure]
- members:
    - image_to_layers_qwen_image_layered  (official, 14 nodes) - "Image to Layers (Qwen-Image-Layered)"

## Type 53  -  1 member(s)  -  source: official
- cohesion (mean intra-similarity): 1.0
- official categories: 3D (1)  [pure]
- members:
    - image_to_model_hunyuan3d_2_1  (official, 8 nodes) - "Image to 3D Model (Hunyuan3d 2.1)"

## Type 54  -  1 member(s)  -  source: official
- cohesion (mean intra-similarity): 1.0
- official categories: Image generation and editing (1)  [pure]
- members:
    - image_upscale_z_image_turbo  (official, 13 nodes) - "Image Upscale (Z-image-Turbo)"

## Type 55  -  1 member(s)  -  source: official
- cohesion (mean intra-similarity): 1.0
- official categories: Video Tools (1)  [pure]
- members:
    - merge_videos  (official, 12 nodes) - "Merge Videos"

## Type 56  -  1 member(s)  -  source: custom
- cohesion (mean intra-similarity): 1.0
- members:
    - qwen2511_imageEdit  (custom, 17 nodes)

## Type 57  -  1 member(s)  -  source: official
- cohesion (mean intra-similarity): 1.0
- official categories: Image Tools (1)  [pure]
- members:
    - remove_background_birefnet  (official, 4 nodes) - "Remove Background (BiRefNet)"

## Type 58  -  1 member(s)  -  source: official
- cohesion (mean intra-similarity): 1.0
- official categories: Text Tools (1)  [pure]
- members:
    - select_per_line_text_by_index  (official, 4 nodes) - "Select Per-Line Text by Index"

## Type 59  -  1 member(s)  -  source: custom
- cohesion (mean intra-similarity): 1.0
- members:
    - styletransfer_NanoBananaPro  (custom, 18 nodes)

## Type 60  -  1 member(s)  -  source: official
- cohesion (mean intra-similarity): 1.0
- official categories: Audio (1)  [pure]
- members:
    - text_to_audio_ace_step_1_5  (official, 11 nodes) - "Text to Audio (ACE-Step 1.5)"

## Type 61  -  1 member(s)  -  source: official
- cohesion (mean intra-similarity): 1.0
- official categories: Image generation and editing (1)  [pure]
- members:
    - text_to_image_ideogram_v4  (official, 27 nodes) - "Text to Image (Ideogram v4)"

## Type 62  -  1 member(s)  -  source: official
- cohesion (mean intra-similarity): 1.0
- official categories: Image generation and editing (1)  [pure]
- members:
    - text_to_image_netayume_lumina  (official, 14 nodes) - "Text to Image (NetaYume Lumina)"

## Type 63  -  1 member(s)  -  source: custom
- cohesion (mean intra-similarity): 1.0
- members:
    - upscale_ultimateSD  (custom, 7 nodes)

## Type 64  -  1 member(s)  -  source: custom
- cohesion (mean intra-similarity): 1.0
- members:
    - upscale_using_model  (custom, 4 nodes)

## Type 65  -  1 member(s)  -  source: custom
- cohesion (mean intra-similarity): 1.0
- members:
    - video_gemini_motionPromptGeneration  (custom, 6 nodes)

## Type 66  -  1 member(s)  -  source: official
- cohesion (mean intra-similarity): 1.0
- official categories: Video generation and editing (1)  [pure]
- members:
    - video_inpaint_void  (official, 39 nodes) - "Video Inpaint (VOID)"

## Type 67  -  1 member(s)  -  source: official
- cohesion (mean intra-similarity): 1.0
- official categories: Video Tools (1)  [pure]
- members:
    - video_stitch  (official, 8 nodes) - "Video Stitch"

## Type 68  -  1 member(s)  -  source: official
- cohesion (mean intra-similarity): 1.0
- official categories: Video generation and editing (1)  [pure]
- members:
    - video_upscale_gan_x4  (official, 4 nodes) - "Video Upscale(GAN x4)"
