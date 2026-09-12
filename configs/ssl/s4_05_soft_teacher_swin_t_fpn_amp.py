"""S4.05 — AMP-enabled SSL path for Swin-T-FPN."""

_base_ = [
    "./s4_02_soft_teacher_swin_t_fpn.py",
]

optim_wrapper = dict(
    type="AmpOptimWrapper",
    loss_scale="dynamic",
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

# S4.05 scope:
# - AMP = ON through MMEngine AmpOptimWrapper.
# - Dynamic loss scaling follows the already-validated S3.05 runtime choice.
# - Gradient clipping remains OFF.
# - The closed Swin-T AdamW recipe and native no-decay rules are preserved.
# - Actual optimizer-step accounting remains authoritative for EMA timing.
# - An AMP-skipped underlying optimizer step must therefore produce no EMA.