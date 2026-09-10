"""S3.05 — Swin-T supervised AMP numerics."""

_base_ = [
    "./s3_04_faster_rcnn_swin_t_fpn_sup_optimizer.py",
]

optim_wrapper = dict(
    type="AmpOptimWrapper",
    loss_scale="dynamic",
)

# S3.05 numerical contract:
# - AMP = ON
# - gradient clipping = OFF
#
# "dynamic" loss scaling is the canonical MMEngine 0.10.7 AMP behavior.
# It is an implementation choice because the scientific protocol does not
# prescribe a static/dynamic loss-scale hyperparameter.
#
# clip_grad is intentionally omitted, so MMEngine resolves clip_grad=None.
# The closed S3.04 AdamW optimizer recipe and native Swin no-decay rules
# are inherited unchanged.