"""S5.06 controlled repair — fixed-validation evaluator for SSL scientific checkpoint selection."""

SSL_VAL_CLASSES = (
    "Aortic enlargement",
    "Atelectasis",
    "Calcification",
    "Cardiomegaly",
    "Consolidation",
    "ILD",
    "Infiltration",
    "Lung Opacity",
    "Nodule/Mass",
    "Other lesion",
    "Pleural effusion",
    "Pleural thickening",
    "Pneumothorax",
    "Pulmonary fibrosis",
)

val_pipeline = [
    dict(type="LoadImageFromFile", color_type="color"),
    dict(type="LoadAnnotations", with_bbox=True),
    dict(type="Resize", scale=(1333, 800), keep_ratio=True),
    dict(type="Pad", size_divisor=32),
    dict(type="PackDetInputs"),
]

val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type="DefaultSampler", shuffle=False),
    dataset=dict(
        type="CocoDataset",
        ann_file="/workspace/ssod/data/coco/instances_val.json",
        data_prefix=dict(img="/workspace/ssod/data/mmdet_images"),
        metainfo=dict(classes=SSL_VAL_CLASSES),
        filter_cfg=dict(filter_empty_gt=False),
        test_mode=True,
        pipeline=val_pipeline,
    ),
)

val_evaluator = dict(
    type="ProtocolCocoMetric",
    ann_file="/workspace/ssod/data/coco/instances_val.json",
    metric="bbox",
    classwise=True,
    proposal_nums=(1, 10, 100),
    metric_items=["mAP", "mAP_50", "mAP_75", "AR@1000"],
    format_only=False,
    prefix="coco",
)

val_cfg = dict(type="ValLoop")

# Fixed validation only. Test and hidden-U GT remain unattached.
