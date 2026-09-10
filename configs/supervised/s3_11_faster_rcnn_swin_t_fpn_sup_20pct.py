"""S3.11 — supervised validation every 172 actual optimizer updates."""

_base_ = [
    "./s3_09_faster_rcnn_swin_t_fpn_sup_20pct.py",
]

custom_imports = dict(
    imports=[
        "src.utils.resume_aware_loop",
        "src.utils.resume_checkpoint_hook",
        "src.utils.actual_update_budget_hook",
        "src.utils.actual_update_validation_loop",
    ],
    allow_failed_imports=False,
)

train_cfg = dict(
    type="ActualUpdateValidationIterBasedTrainLoop",
    val_interval=172,
)