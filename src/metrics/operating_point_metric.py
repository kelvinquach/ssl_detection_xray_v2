"""Deterministic operating-point metric for S5 evaluation.

S5.03 implements the locked Recall metric at tau_eval=0.50 using
detector-native predictions after NMS/max100 and deterministic
score-descending greedy, category-aware, one-to-one matching.
"""

from typing import Dict, List

import numpy as np
from mmengine.evaluator import BaseMetric
from mmdet.registry import METRICS
from pycocotools.coco import COCO


@METRICS.register_module()
class OperatingPointMetric(BaseMetric):
    """Locked operating-point Recall evaluator for the SSOD protocol."""

    REQUIRED_TAU_EVAL = 0.50
    REQUIRED_IOU_MATCH_THRESHOLD = 0.50
    REQUIRED_MAX_DETS = 100
    REQUIRED_CATEGORY_IDS = tuple(range(1, 15))

    def __init__(
        self,
        ann_file: str,
        tau_eval: float = REQUIRED_TAU_EVAL,
        iou_match_threshold: float = REQUIRED_IOU_MATCH_THRESHOLD,
        max_dets_per_image: int = REQUIRED_MAX_DETS,
        collect_device: str = "cpu",
        prefix: str = "operating",
    ):
        super().__init__(collect_device=collect_device, prefix=prefix)

        self.tau_eval = float(tau_eval)
        self.iou_match_threshold = float(iou_match_threshold)
        self.max_dets_per_image = int(max_dets_per_image)

        if self.tau_eval != self.REQUIRED_TAU_EVAL:
            raise ValueError(
                "OperatingPointMetric requires tau_eval=0.50; "
                f"got {self.tau_eval!r}"
            )
        if self.iou_match_threshold != self.REQUIRED_IOU_MATCH_THRESHOLD:
            raise ValueError(
                "OperatingPointMetric requires iou_match_threshold=0.50; "
                f"got {self.iou_match_threshold!r}"
            )
        if self.max_dets_per_image != self.REQUIRED_MAX_DETS:
            raise ValueError(
                "OperatingPointMetric requires max_dets_per_image=100; "
                f"got {self.max_dets_per_image!r}"
            )

        self.ann_file = ann_file
        self._coco_api = COCO(ann_file)
        self.cat_ids = tuple(sorted(int(x) for x in self._coco_api.getCatIds()))
        if self.cat_ids != self.REQUIRED_CATEGORY_IDS:
            raise ValueError(
                "OperatingPointMetric requires contiguous category IDs 1..14; "
                f"got {self.cat_ids!r}"
            )

    @staticmethod
    def _bbox_iou_xyxy(box_a: np.ndarray, box_b: np.ndarray) -> float:
        """Return IoU for two xyxy boxes using continuous coordinates."""
        x1 = max(float(box_a[0]), float(box_b[0]))
        y1 = max(float(box_a[1]), float(box_b[1]))
        x2 = min(float(box_a[2]), float(box_b[2]))
        y2 = min(float(box_a[3]), float(box_b[3]))

        inter_w = max(0.0, x2 - x1)
        inter_h = max(0.0, y2 - y1)
        inter = inter_w * inter_h

        area_a = max(0.0, float(box_a[2]) - float(box_a[0])) * max(
            0.0, float(box_a[3]) - float(box_a[1])
        )
        area_b = max(0.0, float(box_b[2]) - float(box_b[0])) * max(
            0.0, float(box_b[3]) - float(box_b[1])
        )
        union = area_a + area_b - inter
        if union <= 0.0:
            return 0.0
        return inter / union

    @staticmethod
    def _xywh_to_xyxy(bbox) -> np.ndarray:
        x, y, w, h = (float(v) for v in bbox)
        return np.asarray([x, y, x + w, y + h], dtype=np.float64)

    def _ground_truth_for_image(self, img_id: int) -> List[dict]:
        """Return GT in fixed COCO annotation order for one image."""
        return list(self._coco_api.imgToAnns.get(int(img_id), []))

    def match_image(
        self,
        img_id: int,
        bboxes: np.ndarray,
        scores: np.ndarray,
        labels: np.ndarray,
    ) -> Dict[str, int]:
        """Apply the locked deterministic one-to-one matching algorithm."""
        bboxes = np.asarray(bboxes, dtype=np.float64)
        scores = np.asarray(scores, dtype=np.float64)
        labels = np.asarray(labels, dtype=np.int64)

        if bboxes.ndim != 2 or bboxes.shape[1] != 4:
            raise ValueError(f"Expected bboxes shape (N, 4); got {bboxes.shape!r}")
        if len(scores) != len(bboxes) or len(labels) != len(bboxes):
            raise ValueError("bboxes, scores and labels must have identical length")
        if len(bboxes) > self.max_dets_per_image:
            raise ValueError(
                "Received more than 100 detector-native predictions for one image; "
                "OperatingPointMetric must run after native max_per_img=100"
            )

        gt_anns = self._ground_truth_for_image(int(img_id))
        gt_matched = np.zeros(len(gt_anns), dtype=bool)

        retained = np.flatnonzero(scores >= self.tau_eval)
        if retained.size:
            local_order = np.argsort(-scores[retained], kind="stable")
            ordered_pred_indices = retained[local_order]
        else:
            ordered_pred_indices = np.asarray([], dtype=np.int64)

        tp = 0
        fp = 0

        for pred_idx in ordered_pred_indices:
            label = int(labels[pred_idx])
            if label < 0 or label >= len(self.cat_ids):
                raise ValueError(f"Prediction label out of range: {label}")
            pred_cat_id = self.cat_ids[label]

            candidate_indices = [
                gt_idx
                for gt_idx, ann in enumerate(gt_anns)
                if (not gt_matched[gt_idx])
                and int(ann["category_id"]) == pred_cat_id
            ]

            if not candidate_indices:
                fp += 1
                continue

            pred_box = bboxes[pred_idx]
            candidate_ious = [
                self._bbox_iou_xyxy(
                    pred_box, self._xywh_to_xyxy(gt_anns[gt_idx]["bbox"])
                )
                for gt_idx in candidate_indices
            ]

            best_local_index = int(np.argmax(candidate_ious))
            best_gt_index = candidate_indices[best_local_index]
            best_iou = float(candidate_ious[best_local_index])

            if best_iou >= self.iou_match_threshold:
                gt_matched[best_gt_index] = True
                tp += 1
            else:
                fp += 1

        fn = int((~gt_matched).sum())
        return {
            "TP": int(tp),
            "FP": int(fp),
            "FN": int(fn),
            "GT": int(len(gt_anns)),
            "retained_detections": int(len(ordered_pred_indices)),
        }

    def process(self, data_batch: dict, data_samples) -> None:
        """Collect detector-native predictions without re-running NMS."""
        for data_sample in data_samples:
            pred = data_sample["pred_instances"]
            bboxes = pred["bboxes"].detach().cpu().numpy()
            scores = pred["scores"].detach().cpu().numpy()
            labels = pred["labels"].detach().cpu().numpy()

            if len(bboxes) > self.max_dets_per_image:
                raise ValueError(
                    "Detector-native output exceeds max_per_img=100 before "
                    "operating-point evaluation"
                )

            self.results.append(
                {
                    "img_id": int(data_sample["img_id"]),
                    "bboxes": bboxes,
                    "scores": scores,
                    "labels": labels,
                }
            )

    def compute_metrics(self, results) -> Dict[str, float]:
        """Compute Recall and FP/image at the locked operating point."""
        tp_total = 0
        fp_total = 0
        fn_total = 0
        evaluated_image_count = len(results)

        for result in results:
            matched = self.match_image(
                result["img_id"],
                result["bboxes"],
                result["scores"],
                result["labels"],
            )
            tp_total += matched["TP"]
            fp_total += matched["FP"]
            fn_total += matched["FN"]

        denominator = tp_total + fn_total
        recall = float("nan") if denominator == 0 else tp_total / denominator
        fp_per_image = (
            float("nan")
            if evaluated_image_count == 0
            else fp_total / evaluated_image_count
        )

        return {
            "Recall_tau_eval": float(recall),
            "FP_per_image": float(fp_per_image),
        }
