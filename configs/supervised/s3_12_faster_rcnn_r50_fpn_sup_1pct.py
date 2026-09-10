"""S3.12 — supervised scientific BEST/LAST checkpoint logic."""

_base_ = [
    "./s3_11_faster_rcnn_r50_fpn_sup_1pct.py",
]

custom_imports = dict(
    imports=[
        "src.utils.resume_aware_loop",
        "src.utils.resume_checkpoint_hook",
        "src.utils.actual_update_budget_hook",
        "src.utils.actual_update_validation_loop",
        "src.utils.sup_best_last_checkpoint_hook",
    ],
    allow_failed_imports=False,
)

custom_hooks = [
    dict(type="ProtocolResumeCheckpointHook"),
    dict(
        type="ActualUpdateBudgetSchedulerHook",
        target_updates=1032,
    ),
    dict(
        type="SupervisedBestLastCheckpointHook",
        metric_key="coco/bbox_mAP",
    ),
]