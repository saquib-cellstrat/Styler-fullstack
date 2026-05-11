from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "HairSwap-Backend"
    api_prefix: str = "/api/v1"
    model_weights_dir: str = "models/weights"
    api_key_header: str = Field(default="X-Api-Key")
    api_key: str = Field(default="change-me")
    # Local in-process service by default - flip to True to enforce X-Api-Key.
    api_key_required: bool = False

    # ONNX models (place files under model_weights_dir)
    modnet_onnx_filename: str = "modnet.onnx"
    retinaface_onnx_filename: str = "retinaface.onnx"
    face_parser_onnx_filename: str = "face_parser.onnx"

    # BiSeNet face parser (CelebAMask-HQ) used to isolate the hair region.
    face_parser_input_size: int = 512
    face_parser_mean_rgb: tuple[float, float, float] = (0.485, 0.456, 0.406)
    face_parser_std_rgb: tuple[float, float, float] = (0.229, 0.224, 0.225)
    # CelebAMask-HQ class indices: 17 = hair, 18 = hat.
    hair_class_index: int = 17
    hat_class_index: int = 18
    include_hat_in_hair: bool = False
    # Use MODNet (when available) as a binary-ish gate that zeros out hair
    # predicted outside the person silhouette. Off by default - the BiSeNet
    # softmax for hair is already a high-quality alpha matte and the gate
    # only helps on noisy backgrounds.
    refine_hair_with_portrait_matte: bool = False
    # Threshold below which the portrait alpha is treated as background;
    # values above ramp linearly to 1 to avoid sharp cuts on hair tips.
    portrait_gate_low: float = 0.05
    portrait_gate_high: float = 0.20
    # Multiplicative gain applied to the hair softmax so confident hair
    # pixels reach full opacity. Small values preserve soft tip edges.
    hair_alpha_gain: float = 1.15

    # Run face parsing on a crop around the detected face (much higher
    # effective resolution than letterboxing the full image) and paste
    # the resulting hair alpha back into the original canvas.
    face_crop_for_parsing: bool = True
    # Padding multipliers relative to the detected face box.
    face_crop_pad_top: float = 1.6
    face_crop_pad_bottom: float = 0.4
    face_crop_pad_horizontal: float = 0.7

    # Output trimming: crop the returned RGBA to the hair bounding box and
    # zero out RGB where alpha is fully transparent. Both reduce PNG size.
    crop_output_to_hair_bbox: bool = True
    output_bbox_padding_px: int = 24
    zero_rgb_where_transparent: bool = True
    alpha_visibility_threshold: float = 0.02

    # PNG encoding: zlib level 6 without exhaustive optimisation gives a
    # 6x faster encode for ~5% larger file - the zero-RGB transparency
    # change does most of the size reduction.
    png_compress_level: int = 6
    png_optimize: bool = False

    # ONNX Runtime (CoreML first on Apple Silicon; override on Linux CI)
    ort_provider_priority: list[str] = Field(
        default_factory=lambda: [
            "CoreMLExecutionProvider",
            "CPUExecutionProvider",
        ]
    )

    # MODNet preprocessing (ImageNet normalization; input NCHW)
    modnet_input_height: int = 512
    modnet_input_width: int = 512
    modnet_input_name: str = "input"
    # Output tensor holding alpha matte (1x1xHxW or 1xHxW); resolved at load if empty
    modnet_output_name: str = ""

    # Original MODNet normalization: (x - 0.5) / 0.5 -> [-1, 1] on each channel.
    modnet_mean_rgb: tuple[float, float, float] = (0.5, 0.5, 0.5)
    modnet_std_rgb: tuple[float, float, float] = (0.5, 0.5, 0.5)

    # RetinaFace (letterbox to square; standard MobileNet0.25 640 export)
    retinaface_input_size: int = 640
    retinaface_conf_threshold: float = 0.5
    retinaface_nms_threshold: float = 0.4
    retinaface_min_sizes: list[list[int]] = Field(
        default_factory=lambda: [[16, 32], [64, 128], [256, 512]]
    )
    retinaface_steps: list[int] = Field(default_factory=lambda: [8, 16, 32])
    retinaface_variances: tuple[float, float] = (0.1, 0.2)
    # Variance for landmarks (same as bbox in PyTorch_Retinaface)
    retinaface_land_variances: tuple[float, float] = (0.1, 0.2)

    # Quality gate: reject if |estimated_yaw_deg| exceeds threshold
    max_abs_yaw_deg: float = 35.0
    # Minimum inter-eye distance (pixels) on original image for a usable face
    min_inter_eye_fraction: float = 0.03

    # Pipeline implementation defaults (swappable via registry + API overrides)
    pipeline_alignment_impl: str = "retinaface"
    pipeline_extraction_impl: str = "modnet"
    pipeline_warp_impl: str = "tps"
    pipeline_harmonization_impl: str = "lab_transfer"
    pipeline_blending_impl: str = "laplacian"

    # Warp and blending quality controls
    tps_regularization: float = 1e-3
    tps_grid_size: int = 12
    laplacian_pyramid_levels: int = 4
    contact_shadow_opacity: float = 0.15
    contact_shadow_blur_sigma: float = 5.0
    hair_alpha_close_kernel: int = 5
    hair_alpha_core_min_opacity: float = 1.0
    hair_alpha_edge_gamma: float = 0.9
    debug_export_enabled: bool = False
    debug_export_dir: str = "debug/exports"
    export_landmarks_json: bool = False
    dense_landmark_backend: str = "heuristic68"
    dense_landmark_onnx_filename: str = "dense_landmarks.onnx"
    depth_onnx_filename: str = "depth_small.onnx"
    enable_experimental_geometry: bool = False
    enable_region_aware_tps: bool = False
    enable_depth_occlusion: bool = False

    model_config = SettingsConfigDict(env_file=".env", env_prefix="HAIRSWAP_")


@lru_cache
def get_settings() -> Settings:
    return Settings()
