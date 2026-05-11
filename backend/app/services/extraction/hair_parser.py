"""BiSeNet face-parsing wrapper that produces a soft hair-only alpha matte.

The model is the standard CelebAMask-HQ BiSeNet (19 classes); class 17 is
``hair`` and class 18 is ``hat``. The softmax probability for those classes
is naturally a soft alpha mask, so we use it directly instead of an argmax
threshold (which would lose fine hair strands at the boundary).

When a face detection is available we crop around the face before parsing
so the donor's head fills the BiSeNet 512x512 receptive field instead of
shrinking to a small region of a wide canvas - this dramatically improves
hair-edge quality on full-body or wide photos.
"""

import numpy as np
from numpy.typing import NDArray

from app.core.model_registry import LoadedOnnxModel
from app.core.settings import Settings
from app.shared.image_processing import AlphaMatte, ImageProcessor


class HairParser:
    def __init__(self, model: LoadedOnnxModel, settings: Settings) -> None:
        self.model = model
        self.settings = settings
        self.image_processor = ImageProcessor()

    def predict_hair_alpha(
        self,
        rgb: NDArray[np.uint8],
        face_box: tuple[float, float, float, float] | None = None,
    ) -> AlphaMatte:
        """Return a soft hair alpha matte at the original image resolution.

        If ``face_box`` is provided and ``face_crop_for_parsing`` is on,
        we run BiSeNet on a tight crop around the face and paste the result
        back into a full-image alpha canvas. Otherwise we fall back to
        letterboxing the whole image.
        """
        if face_box is not None and self.settings.face_crop_for_parsing:
            return self._predict_via_face_crop(rgb, face_box)
        return self._predict_full_image(rgb)

    def _predict_full_image(self, rgb: NDArray[np.uint8]) -> AlphaMatte:
        size = self.settings.face_parser_input_size
        letterbox = self.image_processor.letterbox_for_modnet(
            rgb,
            target_size=(size, size),
            mean=self.settings.face_parser_mean_rgb,
            std=self.settings.face_parser_std_rgb,
        )
        nchw = self.image_processor.to_nchw(letterbox.image)
        hair_512 = self._run_and_take_hair(nchw)
        return self.image_processor.remap_alpha_to_original(hair_512, letterbox)

    def _predict_via_face_crop(
        self,
        rgb: NDArray[np.uint8],
        face_box: tuple[float, float, float, float],
    ) -> AlphaMatte:
        h, w = rgb.shape[:2]
        x1, y1, x2, y2 = face_box
        fw = max(1.0, x2 - x1)
        fh = max(1.0, y2 - y1)
        s = self.settings
        cx1 = max(0, int(round(x1 - fw * s.face_crop_pad_horizontal)))
        cy1 = max(0, int(round(y1 - fh * s.face_crop_pad_top)))
        cx2 = min(w, int(round(x2 + fw * s.face_crop_pad_horizontal)))
        cy2 = min(h, int(round(y2 + fh * s.face_crop_pad_bottom)))
        if cx2 <= cx1 or cy2 <= cy1:
            return self._predict_full_image(rgb)

        crop = rgb[cy1:cy2, cx1:cx2]
        crop_alpha = self._predict_full_image(crop)

        full = np.zeros((h, w), dtype=np.float32)
        full[cy1:cy2, cx1:cx2] = crop_alpha
        return full

    def _run_and_take_hair(self, nchw: NDArray[np.float32]) -> NDArray[np.float32]:
        input_name = self.model.input_names[0]
        # Use only the main head (first output); subsequent outputs are
        # auxiliary supervision heads that would just slow inference.
        outputs = self.model.session.run(
            [self.model.output_names[0]], {input_name: nchw}
        )
        logits = np.asarray(outputs[0], dtype=np.float32)[0]  # (C, H, W)
        probs = self._softmax(logits, axis=0)
        hair = probs[self.settings.hair_class_index]
        if self.settings.include_hat_in_hair:
            hair = np.clip(hair + probs[self.settings.hat_class_index], 0.0, 1.0)
        return hair.astype(np.float32)

    @staticmethod
    def _softmax(x: NDArray[np.float32], axis: int) -> NDArray[np.float32]:
        x = x - np.max(x, axis=axis, keepdims=True)
        exp = np.exp(x)
        return (exp / np.sum(exp, axis=axis, keepdims=True)).astype(np.float32)
