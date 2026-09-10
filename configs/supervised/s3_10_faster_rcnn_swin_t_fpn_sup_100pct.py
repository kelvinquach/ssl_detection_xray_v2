"""S3.10 — Swin-T supervised 100%-SUP reference, 10284 actual-update schedule."""

_base_ = [
    "./s3_07_faster_rcnn_swin_t_fpn_sup_batch4.py",
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
    dataset=_base_.DATASETS["train"],
)

train_cfg = dict(
    type="ResumeAwareIterBasedTrainLoop",
    max_iters=10284,
    val_interval=10284,
)

param_scheduler = [
    dict(
        type="MultiStepLR",
        by_epoch=False,
        end=10285,
        milestones=[6856, 9427],
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
        target_updates=10284,
    ),
]