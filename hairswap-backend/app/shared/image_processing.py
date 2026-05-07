from collections.abc import Iterable
from dataclasses import dataclass
from io import BytesIO

import cv2
import numpy as np
from numpy.typing import NDArray
from PIL import Image, ImageOps


RgbImage = NDArray[np.uint8]
RgbaImage = NDArray[np.uint8]
AlphaMatte = NDArray[np.float32]


def normalize_pixels(pixel_values: Iterable[float]) -> list[float]:
    """Legacy float-list helper kept for older tests."""
    values = list(pixel_values)
    if not values:
        return values
    return [min(1.0, max(0.0, value / 255.0)) for value in values]


@dataclass(frozen=True)
class LetterboxResult:
    """Padded image plus metadata to inverse-map results to the original frame."""

    image: NDArray[np.float32]
    scale: float
    pad_left: int
    pad_top: int
    target_size: tuple[int, int]
    original_size: tuple[int, int]


class ImageProcessor:
    """High-fidelity image utilities for the extraction pipeline.

    Preserves aspect ratio via letterbox padding and reverses the mapping
    when projecting alpha mattes back onto the original image so fine
    hair strands remain aligned with the source pixels.
    """

    def decode(self, raw_bytes: bytes) -> RgbImage:
        """Decode bytes (any common format) to an RGB uint8 array.

        Honors EXIF rotation so portrait phone shots aren't sideways.
        """
        if not raw_bytes:
            raise ValueError("Empty image payload")
        with Image.open(BytesIO(raw_bytes)) as pil_image:
            corrected = ImageOps.exif_transpose(pil_image).convert("RGB")
            return np.asarray(corrected, dtype=np.uint8)

    def letterbox_for_modnet(
        self,
        rgb: RgbImage,
        target_size: tuple[int, int],
        mean: tuple[float, float, float],
        std: tuple[float, float, float],
    ) -> LetterboxResult:
        """Resize preserving aspect ratio + ImageNet-normalize for NCHW MODNet."""
        target_h, target_w = target_size
        original_h, original_w = rgb.shape[:2]
        scale = min(target_w / original_w, target_h / original_h)
        new_w = max(1, int(round(original_w * scale)))
        new_h = max(1, int(round(original_h * scale)))

        resized = cv2.resize(rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)
        canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
        pad_left = (target_w - new_w) // 2
        pad_top = (target_h - new_h) // 2
        canvas[pad_top : pad_top + new_h, pad_left : pad_left + new_w] = resized

        normalized = canvas.astype(np.float32) / 255.0
        normalized -= np.asarray(mean, dtype=np.float32)
        normalized /= np.asarray(std, dtype=np.float32)

        return LetterboxResult(
            image=normalized,
            scale=scale,
            pad_left=pad_left,
            pad_top=pad_top,
            target_size=(target_h, target_w),
            original_size=(original_h, original_w),
        )

    def to_nchw(self, image_hwc: NDArray[np.float32]) -> NDArray[np.float32]:
        """Convert HxWxC float image to 1xCxHxW contiguous batch."""
        chw = np.transpose(image_hwc, (2, 0, 1))
        return np.ascontiguousarray(chw[np.newaxis, ...], dtype=np.float32)

    def remap_alpha_to_original(
        self,
        alpha_padded: AlphaMatte,
        letterbox: LetterboxResult,
    ) -> AlphaMatte:
        """Crop the padded alpha back to the original image coordinates.

        Uses INTER_LINEAR for the upscale to preserve fine hair detail without
        introducing the ringing that bilinear interpolation can produce.
        """
        target_h, target_w = letterbox.target_size
        original_h, original_w = letterbox.original_size

        if alpha_padded.shape != (target_h, target_w):
            alpha_padded = cv2.resize(
                alpha_padded,
                (target_w, target_h),
                interpolation=cv2.INTER_LINEAR,
            )

        unpadded_h = max(1, target_h - 2 * letterbox.pad_top)
        unpadded_w = max(1, target_w - 2 * letterbox.pad_left)
        cropped = alpha_padded[
            letterbox.pad_top : letterbox.pad_top + unpadded_h,
            letterbox.pad_left : letterbox.pad_left + unpadded_w,
        ]
        upscaled = cv2.resize(
            cropped,
            (original_w, original_h),
            interpolation=cv2.INTER_LINEAR,
        )
        return np.clip(upscaled, 0.0, 1.0).astype(np.float32)

    def compose_rgba(
        self,
        rgb: RgbImage,
        alpha: AlphaMatte,
        zero_rgb_where_transparent: bool = True,
    ) -> RgbaImage:
        """Stack RGB + alpha (uint8) into a 4-channel RGBA image.

        Zeroing the RGB channels where alpha is 0 dramatically improves PNG
        compression (background becomes a solid-color region) without
        affecting the rendered output - those pixels are invisible anyway.
        """
        if rgb.shape[:2] != alpha.shape[:2]:
            raise ValueError("RGB/alpha shape mismatch")
        alpha_u8 = np.clip(alpha * 255.0, 0, 255).astype(np.uint8)
        rgb_out = rgb
        if zero_rgb_where_transparent:
            mask = alpha_u8 == 0
            if mask.any():
                rgb_out = rgb.copy()
                rgb_out[mask] = 0
        return np.dstack([rgb_out, alpha_u8])

    def crop_to_alpha_bbox(
        self,
        rgba: RgbaImage,
        alpha: AlphaMatte,
        threshold: float,
        padding_px: int,
    ) -> tuple[RgbaImage, AlphaMatte, tuple[int, int, int, int] | None]:
        """Crop an RGBA image and its alpha to the bounding box of visible alpha."""
        mask = alpha > threshold
        if not mask.any():
            return rgba, alpha, None
        ys, xs = np.where(mask)
        h, w = alpha.shape[:2]
        y1 = max(0, int(ys.min()) - padding_px)
        y2 = min(h, int(ys.max()) + 1 + padding_px)
        x1 = max(0, int(xs.min()) - padding_px)
        x2 = min(w, int(xs.max()) + 1 + padding_px)
        cropped_rgba = rgba[y1:y2, x1:x2]
        cropped_alpha = alpha[y1:y2, x1:x2]
        return cropped_rgba, cropped_alpha, (x1, y1, x2, y2)

    def encode_png(
        self,
        rgba: RgbaImage,
        compress_level: int = 6,
        optimize: bool = False,
    ) -> bytes:
        """Encode an RGBA uint8 image as a PNG byte string."""
        if rgba.ndim != 3 or rgba.shape[2] != 4:
            raise ValueError("Expected an RGBA image")
        buffer = BytesIO()
        Image.fromarray(rgba, mode="RGBA").save(
            buffer,
            format="PNG",
            optimize=optimize,
            compress_level=compress_level,
        )
        return buffer.getvalue()
