from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "HairSwap-Backend"
    api_prefix: str = "/api/v1"
    model_weights_dir: str = "models/weights"
    api_key_header: str = Field(default="X-Api-Key")
    api_key: str = Field(default="change-me")

    # ONNX models (place files under model_weights_dir)
    modnet_onnx_filename: str = "modnet.onnx"
    retinaface_onnx_filename: str = "retinaface.onnx"

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

    model_config = SettingsConfigDict(env_file=".env", env_prefix="HAIRSWAP_")


@lru_cache
def get_settings() -> Settings:
    return Settings()
