"""S3.07 — R50 supervised effective labeled batch = 4."""

_base_ = [
    "./s3_05_faster_rcnn_r50_fpn_sup_amp.py",
    "../dataset/s2_coco_dataset.py",
]

train_dataloader = dict(
    batch_size=4,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type="DefaultSampler", shuffle=True),
    batch_sampler=dict(type="AspectRatioBatchSampler"),
    dataset=_base_.DATASETS["L_1pct"],
)

optim_wrapper = dict(
    accumulative_counts=1,
)

# Scientific invariant:
#   B_L_eff = 4
#
# Current validated single-process execution:
#   batch_size_per_process = 4
#   world_size = 1
#   accumulative_counts = 1
#   => B_L_eff = 4
#
# The specific execution decomposition is implementation/runtime detail.
# Official launch must fail-fast if the effective labeled batch is not 4.
