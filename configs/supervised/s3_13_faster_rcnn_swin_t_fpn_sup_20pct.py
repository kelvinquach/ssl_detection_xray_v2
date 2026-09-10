"""S3.13 — supervised operational LATEST_RESUME checkpoint path."""

_base_ = [
    "./s3_12_faster_rcnn_swin_t_fpn_sup_20pct.py",
]

custom_imports = dict(
    imports=[
        "src.utils.resume_aware_loop",
        "src.utils.resume_checkpoint_hook",
        "src.utils.actual_update_budget_hook",
        "src.utils.actual_update_validation_loop",
        "src.utils.sup_best_last_checkpoint_hook",
        "src.utils.sup_latest_resume_checkpoint_hook",
    ],
    allow_failed_imports=False,
)

custom_hooks = [
    dict(type="ProtocolResumeCheckpointHook"),
    dict(
        type="ActualUpdateBudgetSchedulerHook",
        target_updates=2064,
    ),
    dict(
        type="SupervisedBestLastCheckpointHook",
        metric_key="coco/bbox_mAP",
    ),
    dict(
        type="SupervisedLatestResumeCheckpointHook",
        refresh_interval=172,
    ),
]