"""S3.08 — R50 supervised 1% actual-update budget and scheduler."""

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

train_cfg = dict(
    type="ResumeAwareIterBasedTrainLoop",
    max_iters=1032,
    val_interval=1032,
)

param_scheduler = [
    dict(
        type="MultiStepLR",
        by_epoch=False,
        end=1033,
        milestones=[688, 946],
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
        target_updates=1032,
    ),
]
