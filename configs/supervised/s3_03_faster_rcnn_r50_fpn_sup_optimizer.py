"""S3.03 — R50 supervised SGD optimizer recipe."""

_base_ = [
    "./s3_01_faster_rcnn_r50_fpn_sup.py",
]

optim_wrapper = dict(
    type="OptimWrapper",
    optimizer=dict(
        type="SGD",
        lr=0.005,
        momentum=0.9,
        weight_decay=0.0001,
    ),
)

# S3.03 scope only:
# - optimizer = SGD
# - lr = 0.005
# - momentum = 0.9
# - weight_decay = 0.0001
#
# AMP, gradient clipping, scheduler, effective batch, update budgets,
# validation, checkpointing and resume are handled by later S3 tasks.
