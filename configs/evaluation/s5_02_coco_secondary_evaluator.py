"""S5.02 — COCO secondary metrics on fixed validation."""

_base_ = [
    "./s5_01_coco_bbox_evaluator.py",
]

custom_imports = dict(
    imports=[
        "src.metrics.protocol_coco_metric",
    ],
    allow_failed_imports=False,
)

# Replace, do not merge with S5.01 val_evaluator.
# In particular, S5.02 intentionally omits the tuple-valued iou_thrs from
# S5.01 so ProtocolCocoMetric constructs the exact 0.50:0.05:0.95 NumPy grid
# required by pycocotools for AP50/AP75 lookup.
val_evaluator = dict(
    _delete_=True,
    type="ProtocolCocoMetric",
    ann_file="/workspace/ssod/data/coco/instances_val.json",
    metric="bbox",
    classwise=True,
    proposal_nums=(1, 10, 100),
    metric_items=["mAP", "mAP_50", "mAP_75", "AR@1000"],
    format_only=False,
    prefix="coco",
)

# S5.02 scientific invariants:
# - bbox mAP@[0.50:0.95] remains the primary metric.
# - AP50 and AP75 are secondary metrics.
# - class-wise AP@[0.50:0.95], AP50 and AP75 are emitted machine-readably.
# - AR@[0.50:0.95], maxDets=100, area=all is normalized to
#   coco/AR_50_95_max100.
# - useCats=1 is preserved by bbox evaluation.
# - no extra fixed score threshold is applied before COCO evaluation.
# - AP_S/AP_M/AP_L are not official study endpoints.
# - test remains unattached.