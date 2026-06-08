import cv2
import numpy as np
from numpy.typing import NDArray

from app.core.model_registry import InferenceRegistry
from app.core.settings import Settings
from app.pipeline.context import ProcessingContext
from app.pipeline.stages.base import AbstractPipelineStage

# Mid-face skin sample (nose bridge, tip, inner cheeks). These vertices stay
# on bare skin for both a bald base and a donor whose hair covers the brow,
# so they give a clean read of each photo's lighting and white balance.
_SKIN_REGION = (1, 2, 4, 5, 6, 19, 94, 50, 280, 101, 330, 205, 425)


class LabColorTransferStage(AbstractPipelineStage):
    name = "harmonization"

    def __init__(self, models: InferenceRegistry, settings: Settings) -> None:
        _ = models
        self._settings = settings

    def process(self, context: ProcessingContext) -> ProcessingContext:
        if (
            context.base_rgb is None
            or context.warped_hair_rgb is None
            or context.warped_hair_alpha is None
        ):
            raise ValueError("Warp stage must run before harmonization")

        strength = float(self._settings.harmonization_strength)
        if strength <= 0.0:
            context.harmonized_hair_rgb = context.warped_hair_rgb
            return context

        harmonized = self._relight_to_base(
            hair_rgb=context.warped_hair_rgb,
            base_rgb=context.base_rgb,
            donor_rgb=context.donor_rgb,
            base_mesh=context.base_mesh_points,
            donor_mesh=context.donor_mesh_points,
            strength=strength,
            chroma_ratio=float(self._settings.harmonization_chroma_ratio),
        )
        context.harmonized_hair_rgb = harmonized if harmonized is not None else context.warped_hair_rgb
        return context

    def _relight_to_base(
        self,
        hair_rgb: NDArray[np.uint8],
        base_rgb: NDArray[np.uint8],
        donor_rgb: NDArray[np.uint8] | None,
        base_mesh: NDArray[np.float32] | None,
        donor_mesh: NDArray[np.float32] | None,
        strength: float,
        chroma_ratio: float,
    ) -> NDArray[np.uint8] | None:
        """Shift the donor hair by the base-vs-donor skin illumination delta.

        The mean LAB of each photo's mid-face skin is a stand-in for its
        lighting and white balance. Applying the difference (base - donor) to
        the hair re-lights it into the base scene without touching the hair's
        intrinsic darkness or hue variation. Luminance moves at `strength`;
        colour (a, b) moves at `strength * chroma_ratio` so the chosen style
        colour is preserved.
        """
        if donor_rgb is None or base_mesh is None or donor_mesh is None:
            return None
        base_skin = self._skin_lab_mean(base_rgb, base_mesh)
        donor_skin = self._skin_lab_mean(donor_rgb, donor_mesh)
        if base_skin is None or donor_skin is None:
            return None

        delta = base_skin - donor_skin
        # Clamp so a bad skin read (shadowed cheek, colour cast) can't blow the
        # hair out; these caps are generous relative to normal lighting gaps.
        delta = np.clip(delta, [-40.0, -25.0, -25.0], [40.0, 25.0, 25.0])
        shift = delta * np.array(
            [strength, strength * chroma_ratio, strength * chroma_ratio], dtype=np.float32
        )

        hair_lab = cv2.cvtColor(hair_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
        hair_lab += shift
        out = np.clip(hair_lab, 0, 255).astype(np.uint8)
        return cv2.cvtColor(out, cv2.COLOR_LAB2RGB)

    @staticmethod
    def _skin_lab_mean(
        rgb: NDArray[np.uint8], mesh_points: NDArray[np.float32]
    ) -> NDArray[np.float32] | None:
        h, w = rgb.shape[:2]
        idx = [i for i in _SKIN_REGION if i < mesh_points.shape[0]]
        if len(idx) < 3:
            return None
        pts = mesh_points[idx].astype(np.int32)
        pts[:, 0] = np.clip(pts[:, 0], 0, w - 1)
        pts[:, 1] = np.clip(pts[:, 1], 0, h - 1)
        mask = np.zeros((h, w), dtype=np.uint8)
        hull = cv2.convexHull(pts)
        cv2.fillConvexPoly(mask, hull, 255)
        mask = cv2.erode(mask, np.ones((5, 5), dtype=np.uint8), iterations=1)
        sel = mask > 0
        if int(sel.sum()) < 30:
            return None
        lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
        return lab[sel].mean(axis=0)
