"""S4.05 — AMP-enabled SSL path for R50-FPN."""

_base_ = [
    "./s4_02_soft_teacher_r50_fpn.py",
]

optim_wrapper = dict(
    type="AmpOptimWrapper",
    loss_scale="dynamic",
    optimizer=dict(
        type="SGD",
        lr=0.005,
        momentum=0.9,
        weight_decay=0.0001,
    ),
)

# S4.05 scope:
# - AMP = ON through MMEngine AmpOptimWrapper.
# - Dynamic loss scaling follows the already-validated S3.05 runtime choice.
# - Gradient clipping remains OFF.
# - The closed R50 SGD recipe is preserved exactly.
# - Actual optimizer-step accounting remains authoritative for EMA timing.
# - An AMP-skipped underlying optimizer step must therefore produce no EMA.