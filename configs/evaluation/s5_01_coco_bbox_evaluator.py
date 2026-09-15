"""S5.01 — Shared COCO bbox evaluator for fixed validation."""

_base_ = [
    "../dataset/s2_coco_dataset.py",
]

COCO_IOU_THRS = (
    0.50,
    0.55,
    0.60,
    0.65,
    0.70,
    0.75,
    0.80,
    0.85,
    0.90,
    0.95,
)

# Fixed validation only. Test remains closed until the final evaluation campaign.
val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type="DefaultSampler", shuffle=False),
    dataset=_base_.DATASETS["val"],
)

val_evaluator = dict(
    type="CocoMetric",
    ann_file="/workspace/ssod/data/coco/instances_val.json",
    metric="bbox",
    classwise=False,
    proposal_nums=(1, 10, 100),
    iou_thrs=COCO_IOU_THRS,
    metric_items=["mAP"],
    format_only=False,
    prefix="coco",
)

val_cfg = dict(type="ValLoop")

# S5.01 scientific invariants:
# - bbox mAP@[0.50:0.95] is the primary detection metric.
# - category-aware COCO evaluation keeps useCats=1.
# - AP uses maxDets=100 through proposal_nums=(1, 10, 100).
# - no extra fixed score threshold is applied before COCO AP.
# - validation is the only dataset attached here; test is intentionally absent.
