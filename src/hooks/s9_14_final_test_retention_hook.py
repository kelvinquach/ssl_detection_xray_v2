"""S9.14 one-pass final-test prediction retention hook.

The hook observes the exact outputs already consumed by the frozen evaluators.
It does not run inference, NMS, score filtering, matching, or metric logic.
"""

from __future__ import annotations

from typing import Any

from mmengine.hooks import Hook
from mmengine.registry import HOOKS


def _value(obj: Any, name: str) -> Any:
    if isinstance(obj, dict):
        return obj[name]
    return getattr(obj, name)


def _to_list(value: Any) -> list:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "tolist"):
        return value.tolist()
    return list(value)


@HOOKS.register_module()
class FinalTestPredictionRetentionHook(Hook):
    """Retain detector-native test outputs from the single MMEngine test pass."""

    priority = "VERY_LOW"

    def __init__(self, max_dets_per_image: int = 100) -> None:
        self.max_dets_per_image = int(max_dets_per_image)
        if self.max_dets_per_image != 100:
            raise ValueError(
                "S9.14 retention hook requires detector-native max_per_img=100."
            )
        self.records: list[dict[str, Any]] = []
        self._seen_img_ids: set[int] = set()

    def before_test(self, runner) -> None:
        self.records.clear()
        self._seen_img_ids.clear()

    def after_test_iter(
        self,
        runner,
        batch_idx: int,
        data_batch=None,
        outputs=None,
    ) -> None:
        if outputs is None:
            raise RuntimeError("S9.14 final-test retention received no outputs.")

        for sample in outputs:
            img_id = int(_value(sample, "img_id"))
            if img_id in self._seen_img_ids:
                raise RuntimeError(
                    f"Duplicate final-test image id observed: {img_id}"
                )

            pred = _value(sample, "pred_instances")
            bboxes = _to_list(_value(pred, "bboxes"))
            scores = _to_list(_value(pred, "scores"))
            labels = _to_list(_value(pred, "labels"))

            if not (len(bboxes) == len(scores) == len(labels)):
                raise RuntimeError(
                    f"Prediction field length mismatch for image {img_id}."
                )
            if len(bboxes) > self.max_dets_per_image:
                raise RuntimeError(
                    "Detector-native final-test output exceeds max_per_img=100 "
                    f"for image {img_id}."
                )

            self.records.append(
                {
                    "img_id": img_id,
                    "bboxes": bboxes,
                    "scores": scores,
                    "labels": labels,
                }
            )
            self._seen_img_ids.add(img_id)

    def after_test(self, runner) -> None:
        dataset_size = len(runner.test_loop.dataloader.dataset)
        if len(self.records) != dataset_size:
            raise RuntimeError(
                "S9.14 retained prediction count mismatch: "
                f"records={len(self.records)} dataset={dataset_size}."
            )

    def export_native_records(self) -> list[dict[str, Any]]:
        """Return a copy of retained detector-native predictions."""
        return [
            {
                "img_id": int(record["img_id"]),
                "bboxes": [list(box) for box in record["bboxes"]],
                "scores": list(record["scores"]),
                "labels": list(record["labels"]),
            }
            for record in self.records
        ]
