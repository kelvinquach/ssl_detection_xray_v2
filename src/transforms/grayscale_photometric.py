"""S4.07 continuous grayscale brightness/contrast augmentation."""

from __future__ import annotations

from typing import Sequence, Tuple

import mmcv
import numpy as np
from mmcv.transforms import BaseTransform
from mmcv.transforms.utils import cache_randomness

from mmdet.registry import TRANSFORMS


@TRANSFORMS.register_module()
class GrayscaleBrightnessContrast(BaseTransform):
    """Apply protocol-locked continuous grayscale photometric augmentation.

    The two factors are sampled independently:

        brightness ~ Uniform(brightness_range[0], brightness_range[1])
        contrast   ~ Uniform(contrast_range[0], contrast_range[1])

    This transform changes image intensities only. It never modifies geometry,
    bounding boxes, masks, or homography metadata.

    The input and output must satisfy the chest-X-ray grayscale constraint
    R == G == B.
    """

    def __init__(
        self,
        brightness_range: Sequence[float] = (0.90, 1.10),
        contrast_range: Sequence[float] = (0.90, 1.10),
    ) -> None:
        self.brightness_range = self._validate_range(
            brightness_range,
            "brightness_range",
        )
        self.contrast_range = self._validate_range(
            contrast_range,
            "contrast_range",
        )

    @staticmethod
    def _validate_range(
        value: Sequence[float],
        name: str,
    ) -> Tuple[float, float]:
        if len(value) != 2:
            raise ValueError(f"{name} must contain exactly two values.")

        low = float(value[0])
        high = float(value[1])

        if not (0.0 < low <= high):
            raise ValueError(
                f"{name} must satisfy 0 < low <= high, got {value}."
            )

        return low, high

    @staticmethod
    def _is_three_channel_grayscale(img: np.ndarray) -> bool:
        return bool(
            img.ndim == 3
            and img.shape[2] == 3
            and np.array_equal(img[..., 0], img[..., 1])
            and np.array_equal(img[..., 1], img[..., 2])
        )

    @cache_randomness
    def _sample_brightness_factor(self) -> float:
        low, high = self.brightness_range
        return float(np.random.uniform(low, high))

    @cache_randomness
    def _sample_contrast_factor(self) -> float:
        low, high = self.contrast_range
        return float(np.random.uniform(low, high))

    def transform(self, results: dict) -> dict:
        if "img" not in results:
            raise KeyError("'img' is required by GrayscaleBrightnessContrast.")

        img = results["img"]

        if not self._is_three_channel_grayscale(img):
            raise ValueError(
                "S4.07 strong augmentation requires three identical "
                "grayscale channels (R=G=B) before augmentation."
            )

        original_shape = img.shape
        brightness_factor = self._sample_brightness_factor()
        contrast_factor = self._sample_contrast_factor()

        img = mmcv.adjust_brightness(
            img,
            brightness_factor,
            backend="cv2",
        )
        img = mmcv.adjust_contrast(
            img,
            contrast_factor,
            backend="cv2",
        )

        if img.shape != original_shape:
            raise RuntimeError(
                "S4.07 photometric transform changed image geometry."
            )

        if not self._is_three_channel_grayscale(img):
            raise RuntimeError(
                "S4.07 photometric transform violated R=G=B."
            )

        results["img"] = img

        # Runtime evidence only; these values do not alter supervision.
        results["strong_brightness_factor"] = brightness_factor
        results["strong_contrast_factor"] = contrast_factor

        return results

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"brightness_range={self.brightness_range}, "
            f"contrast_range={self.contrast_range})"
        )