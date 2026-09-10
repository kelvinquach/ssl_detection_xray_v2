"""S3.09 — R50 supervised 20% standard 2064 actual-update schedule."""

_base_ = [
    "./s3_07_faster_rcnn_r50_fpn_sup_batch4.py",
]

custom_imports = dict(
    imports=[
        "src.utils.resume_aware_loop",
        "src.utils.resume_checkpoint_hook",
        "src.utils.actual_update_budget_hook",
    ],
    allow_failed_imports=False,
)

train_dataloader = dict(
    dataset=_base_.DATASETS["L_20pct"],
)

train_cfg = dict(
    type="ResumeAwareIterBasedTrainLoop",
    max_iters=2064,
    val_interval=2064,
)

param_scheduler = [
    dict(
        type="MultiStepLR",
        by_epoch=False,
        end=2065,
        milestones=[1376, 1892],
        gamma=0.1,
    ),
]

default_hooks = dict(
    param_scheduler=None,
    checkpoint=None,
)

custom_hooks = [
    dict(type="ProtocolResumeCheckpointHook"),
    dict(
        type="ActualUpdateBudgetSchedulerHook",
        target_updates=2064,
    ),
]