"""S5.03 — Operating-point Recall on fixed validation."""

_base_ = [
    "./s5_01_coco_bbox_evaluator.py",
]

custom_imports = dict(
    imports=[
        "src.metrics.operating_point_metric",
    ],
    allow_failed_imports=False,
)

# Replace the S5.01 COCO evaluator; retain its fixed-validation dataloader
## and ValLoop. Test remains deliberately unattached.
val_evaluator = dict(
    _delete_=True,
    type="OperatingPointMetric",
    ann_file="/workspace/ssod/data/coco/instances_val.json",
    tau_eval=0.50,
    iou_match_threshold=0.50,
    max_dets_per_image=100,
    prefix="operating",
)

# S5.03 scientific invariants:
# - evaluation is restricted to the fixed validation split.
# - detector-native predictions are consumed after native NMS/max_per_img=100.
# - no second NMS or evaluator-side top-100 truncation is applied.
# - detections are retained when score >= tau_eval=0.50.
# - retained detections are processed by descending confidence score.
# - equal-score detections preserve detector output order.
# - matching is category-aware, one-to-one, and uses unmatched same-class GT only.
# - the unmatched same-class GT with maximum IoU is selected.
# - equal maximum IoU uses earliest fixed COCO annotation order.
# - a match is accepted when IoU >= 0.50.
# - S5.03 reports Recall = TP / (TP + FN).
# - test remains unattached and inaccessible during development/preflight.
