"""S3.02 — Faster R-CNN Swin-T FPN supervised model path."""

_base_ = [
    "mmdet::_base_/models/faster-rcnn_r50_fpn.py",
]

SWIN_T_CHECKPOINT = "/workspace/ssod/cache/pretrained/swin_tiny_patch4_window7_224.pth"

model = dict(
    backbone=dict(
        _delete_=True,
        type="SwinTransformer",
        embed_dims=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=7,
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.2,
        patch_norm=True,
        out_indices=(0, 1, 2, 3),
        with_cp=False,
        convert_weights=True,
        frozen_stages=-1,
        init_cfg=dict(
            type="Pretrained",
            checkpoint=SWIN_T_CHECKPOINT,
        ),
    ),
    neck=dict(
        in_channels=[96, 192, 384, 768],
    ),
    roi_head=dict(
        bbox_head=dict(
            num_classes=14,
        ),
    ),
)

load_from = None

# S3.02 scope only:
# - Faster R-CNN + Swin-T + FPN
# - ImageNet-1K Swin-T backbone initialization
# - verified backbone checkpoint loaded from Vast-local cache
# - no detector-level COCO initialization
# - all backbone stages trainable
# - 14 detection classes
#
# Dataset/preprocessing, optimizer, AMP, batch, scheduler, validation,
# checkpoint and resume configuration are intentionally handled by their
# dedicated S2/S3 tasks and are not redefined here.
