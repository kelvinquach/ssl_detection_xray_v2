"""S3.04 — Swin-T supervised AdamW optimizer recipe."""

_base_ = [
    "./s3_02_faster_rcnn_swin_t_fpn_sup.py",
]

optim_wrapper = dict(
    type="OptimWrapper",
    paramwise_cfg=dict(
        custom_keys={
            "absolute_pos_embed": dict(decay_mult=0.0),
            "relative_position_bias_table": dict(decay_mult=0.0),
            "norm": dict(decay_mult=0.0),
        },
    ),
    optimizer=dict(
        type="AdamW",
        lr=0.000025,
        betas=(0.9, 0.999),
        weight_decay=0.05,
    ),
)

# S3.04 scope only:
# - optimizer = AdamW
# - lr = 0.000025
# - betas = (0.9, 0.999)
# - weight_decay = 0.05
# - native Swin no-decay rules retained
#
# AMP, gradient clipping, scheduler, effective batch, update budgets,
# validation, checkpointing and resume are handled by later S3 tasks.