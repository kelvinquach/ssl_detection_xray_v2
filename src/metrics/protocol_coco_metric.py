"""Protocol-compatible COCO bbox metric for S5 evaluation.

This adapter preserves the locked COCO scientific semantics while normalizing
MMDetection 3.3.0 runtime behavior needed by S5.02:
- IoU thresholds are converted to a NumPy array so COCOeval AP50/AP75 lookup works.
- COCO AR@[0.50:0.95], maxDets=100 is exposed as AR_50_95_max100.
- Per-class AP@[0.50:0.95], AP50 and AP75 are exposed as machine-readable keys.
"""

import os.path as osp
import tempfile
from typing import Dict, Sequence

import numpy as np
from mmdet.evaluation.metrics import CocoMetric
from mmdet.registry import METRICS
from pycocotools.cocoeval import COCOeval


@METRICS.register_module()
class ProtocolCocoMetric(CocoMetric):
    """COCO bbox evaluator normalized to the locked project metric contract."""

    REQUIRED_PROPOSAL_NUMS = (1, 10, 100)

    def __init__(
        self,
        *args,
        classwise: bool = True,
        proposal_nums: Sequence[int] = REQUIRED_PROPOSAL_NUMS,
        iou_thrs=None,
        metric_items=None,
        **kwargs,
    ):
        proposal_nums = tuple(int(x) for x in proposal_nums)
        if proposal_nums != self.REQUIRED_PROPOSAL_NUMS:
            raise ValueError(
                "ProtocolCocoMetric requires proposal_nums=(1, 10, 100); "
                f"got {proposal_nums!r}"
            )

        if iou_thrs is None:
            iou_thrs = np.linspace(0.50, 0.95, 10, endpoint=True)
        else:
            iou_thrs = np.asarray(iou_thrs, dtype=float)

        expected_iou = np.linspace(0.50, 0.95, 10, endpoint=True)
        if iou_thrs.shape != expected_iou.shape or not np.allclose(
            iou_thrs, expected_iou, rtol=0.0, atol=1e-12
        ):
            raise ValueError(
                "ProtocolCocoMetric requires IoU thresholds 0.50:0.05:0.95"
            )

        if metric_items is None:
            metric_items = ["mAP", "mAP_50", "mAP_75", "AR@1000"]

        self.protocol_classwise = bool(classwise)

        # Base CocoMetric's classwise path only returns aggregate per-class AP
        # machine-readably; AP50/AP75 are only printed. We therefore disable
        # that path and expose all required per-class metrics below.
        super().__init__(
            *args,
            classwise=False,
            proposal_nums=proposal_nums,
            iou_thrs=iou_thrs,
            metric_items=metric_items,
            **kwargs,
        )

    @staticmethod
    def _mean_valid(values) -> float:
        values = np.asarray(values)
        values = values[values > -1]
        if values.size == 0:
            return float("nan")
        return float(np.mean(values))

    def _classwise_metrics(self, results) -> Dict[str, float]:
        """Compute machine-readable AP@[.50:.95], AP50 and AP75 per class."""
        _, preds = zip(*results)

        with tempfile.TemporaryDirectory() as tmp_dir:
            outfile_prefix = osp.join(tmp_dir, "results")
            result_files = self.results2json(preds, outfile_prefix)
            coco_dt = self._coco_api.loadRes(result_files["bbox"])

            coco_eval = COCOeval(self._coco_api, coco_dt, "bbox")
            coco_eval.params.catIds = self.cat_ids
            coco_eval.params.imgIds = self.img_ids
            coco_eval.params.maxDets = list(self.proposal_nums)
            coco_eval.params.iouThrs = self.iou_thrs
            coco_eval.evaluate()
            coco_eval.accumulate()

            precisions = coco_eval.eval["precision"]
            max_det_index = list(self.proposal_nums).index(100)
            iou50_index = int(np.where(np.isclose(self.iou_thrs, 0.50))[0][0])
            iou75_index = int(np.where(np.isclose(self.iou_thrs, 0.75))[0][0])

            output = {}
            for class_index, cat_id in enumerate(self.cat_ids):
                class_name = self._coco_api.loadCats(cat_id)[0]["name"]

                ap_50_95 = self._mean_valid(
                    precisions[:, :, class_index, 0, max_det_index]
                )
                ap50 = self._mean_valid(
                    precisions[iou50_index, :, class_index, 0, max_det_index]
                )
                ap75 = self._mean_valid(
                    precisions[iou75_index, :, class_index, 0, max_det_index]
                )

                output[f"classwise/{class_name}/AP_50_95"] = round(ap_50_95, 3)
                output[f"classwise/{class_name}/AP50"] = round(ap50, 3)
                output[f"classwise/{class_name}/AP75"] = round(ap75, 3)

            return output

    def compute_metrics(self, results) -> Dict[str, float]:
        """Compute aggregate COCO metrics and normalized protocol outputs."""
        eval_results = super().compute_metrics(results)

        # MMDetection 3.3.0 names stats[8] as AR@1000 regardless of the
        # configured proposal_nums. Under the locked (1, 10, 100) setting,
        # stats[8] is semantically AR@[0.50:0.95], maxDets=100.
        raw_ar_key = "bbox_AR@1000"
        if raw_ar_key not in eval_results:
            raise KeyError(
                f"Missing {raw_ar_key!r}; cannot derive AR_50_95_max100"
            )
        eval_results["AR_50_95_max100"] = eval_results.pop(raw_ar_key)

        if self.protocol_classwise:
            eval_results.update(self._classwise_metrics(results))

        return eval_results
