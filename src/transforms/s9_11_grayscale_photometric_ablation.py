"""S9.11 OFAT strong-photometric augmentation support.

Scientific scope
----------------
This transform is used ONLY for the prespecified S9.11 secondary mechanistic
ablation study.

Fixed anchor:
    method         = SSL
    architecture   = Faster R-CNN R50-FPN
    labeled budget = 10%
    seeds          = all 10 locked training seeds

Prespecified augmentation variants
----------------------------------
A0:
    no brightness augmentation
    no contrast augmentation

A1:
    brightness only
    brightness factor ~ Uniform(0.90, 1.10)

A2:
    contrast only
    contrast factor ~ Uniform(0.90, 1.10)

A3 = frozen Main:
    brightness + contrast
    each factor independently sampled from Uniform(0.90, 1.10)

All variants preserve:
    grayscale R = G = B
    image geometry
    bounding boxes
    homography metadata

OFAT invariant
--------------
Only the presence/absence of the two prespecified strong photometric
operations changes.

Confidence thresholds, regression pseudo-label filtering, EMA,
labeled/unlabeled ratio, loss weights, optimizer, learning-rate schedule,
update budget, validation evaluator, and all other scientific settings
remain identical to frozen Main.

Thesis interpretation
---------------------
For each locked training seed:

    effect(Ak, seed)
        = metric(Ak, seed)
        - metric(Main/A3, same seed)

Non-Main A0/A1/A2 are validation-only and are never eligible for final test.
"""

from __future__ import annotations

import mmcv

from mmdet.registry import TRANSFORMS

from src.transforms.grayscale_photometric import (
    GrayscaleBrightnessContrast,
)


@TRANSFORMS.register_module()
class S911GrayscalePhotometricAblation(
    GrayscaleBrightnessContrast
):
    """Locked S9.11 A0/A1/A2/A3 strong-photometric OFAT transform."""

    VALID_VARIANTS = ("A0", "A1", "A2", "A3")

    def __init__(
        self,
        ablation_augmentation_variant: str = "A3",
        brightness_range=(0.90, 1.10),
        contrast_range=(0.90, 1.10),
    ) -> None:
        super().__init__(
            brightness_range=brightness_range,
            contrast_range=contrast_range,
        )

        variant = str(
            ablation_augmentation_variant
        ).strip().upper()

        if variant not in self.VALID_VARIANTS:
            raise ValueError(
                "S9.11 augmentation variant must be one of "
                f"{self.VALID_VARIANTS}; observed={variant!r}"
            )

        self.ablation_augmentation_variant = variant

    def transform(self, results: dict) -> dict:
        """Apply exactly the photometric operations specified by A0-A3."""

        if "img" not in results:
            raise KeyError(
                "'img' is required by "
                "S911GrayscalePhotometricAblation."
            )

        img = results["img"]

        if not self._is_three_channel_grayscale(img):
            raise ValueError(
                "S9.11 strong photometric ablation requires "
                "three identical grayscale channels (R=G=B)."
            )

        original_shape = img.shape

        # Identity values are recorded for disabled operations.
        # Importantly, disabled operations are NOT executed.
        brightness_factor = 1.0
        contrast_factor = 1.0

        if self.ablation_augmentation_variant in ("A1", "A3"):
            brightness_factor = (
                self._sample_brightness_factor()
            )
            img = mmcv.adjust_brightness(
                img,
                brightness_factor,
                backend="cv2",
            )

        if self.ablation_augmentation_variant in ("A2", "A3"):
            contrast_factor = (
                self._sample_contrast_factor()
            )
            img = mmcv.adjust_contrast(
                img,
                contrast_factor,
                backend="cv2",
            )

        if img.shape != original_shape:
            raise RuntimeError(
                "S9.11 photometric ablation changed image geometry."
            )

        if not self._is_three_channel_grayscale(img):
            raise RuntimeError(
                "S9.11 photometric ablation violated R=G=B."
            )

        results["img"] = img

        # Preserve the same runtime evidence fields used by Main.
        results["strong_brightness_factor"] = float(
            brightness_factor
        )
        results["strong_contrast_factor"] = float(
            contrast_factor
        )

        # Additional ablation provenance; this does not alter supervision.
        results["s9_11_augmentation_variant"] = (
            self.ablation_augmentation_variant
        )

        return results

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"variant={self.ablation_augmentation_variant!r}, "
            f"brightness_range={self.brightness_range}, "
            f"contrast_range={self.contrast_range})"
        )