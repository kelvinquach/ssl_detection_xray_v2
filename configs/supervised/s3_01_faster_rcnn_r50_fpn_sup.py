"""S3.01 — Faster R-CNN ResNet-50 FPN supervised model path."""

_base_ = [
    "mmdet::_base_/models/faster-rcnn_r50_fpn.py",
]

model = dict(
    backbone=dict(
        frozen_stages=-1,
        norm_eval=True,
        init_cfg=dict(
            type="Pretrained",
            checkpoint="torchvision://resnet50",
        ),
    ),
    roi_head=dict(
        bbox_head=dict(
            num_classes=14,
        ),
    ),
)

# S3.01 scope only:
# - Faster R-CNN + ResNet-50 + FPN
# - ImageNet-1K backbone initialization
# - no detector-level COCO initialization
# - all backbone stages trainable
# - 14 detection classes
#
# Dataset/preprocessing, optimizer, AMP, batch, scheduler, validation,
# checkpoint and resume configuration are intentionally handled by their
# dedicated S2/S3 tasks and are not redefined here.
