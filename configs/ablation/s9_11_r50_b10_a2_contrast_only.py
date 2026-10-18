"""S9.11 OFAT ablation — strong-photometric augmentation variant A2.

Scientific identity
-------------------
Method        : SSL
Architecture  : Faster R-CNN R50-FPN
Label budget  : 10%
Seeds         : all 10 locked training seeds
Family        : STRONG_PHOTOMETRIC_AUGMENTATION
Variant       : A2

Scientific question
-------------------
How does the presence or absence of brightness and contrast augmentation in
the Student strong view affect pseudo-label quality and detector performance?

Frozen Main reference — A3
--------------------------
Strong Student view:
    Resize
    brightness ~ Uniform(0.90, 1.10)
    contrast   ~ Uniform(0.90, 1.10)
    Pad
    PackDetInputs

A3 = brightness + contrast = frozen Main.

A2 scientific meaning
-------------------------------
CONTRAST_ONLY

OFAT invariant
--------------
ONLY the presence/absence of brightness and contrast in the Student strong
view changes.

The following remain identical to frozen Main:
    Teacher weak view
    image geometry
    labeled dataset
    unlabeled dataset identity
    confidence thresholds
    bbox-regression filtering
    EMA
    labeled/unlabeled ratio = 1:1
    effective labeled batch = 4
    effective unlabeled batch = 4
    loss weights
    optimizer
    LR schedule
    update budget = 2064
    validation interval = 172
    validation evaluator
    BEST/LAST EMA Teacher checkpoint semantics

No geometric augmentation, random erasing, crop, rotation, flip or mosaic is
introduced.

Thesis interpretation
---------------------
A2 is compared against A3/Main OFF_SSL_R50_B10_Sxx using the SAME
locked training seed.

For each seed:

    delta_metric
        = metric(A2, seed)
        - metric(Main/A3, same seed)

Primary ablation metric:
    validation bbox mAP@[0.50:0.95]

Required secondary mechanistic metric:
    Q_pseudo = PL-mAP@[0.50:0.95]

Evaluation policy
-----------------
Non-Main ablation variant.
Validation only.
Not eligible for final test.
"""

_base_ = "../ssl/s5_06_soft_teacher_r50_fpn_scientific_eval_10pct.py"

# -------------------------------------------------------------------------
# S9.11 OFAT STUDENT STRONG VIEW
#
# Main/A3:
#   Resize -> brightness + contrast -> Pad -> PackDetInputs
#
# A2:
#   The S911GrayscalePhotometricAblation transform executes exactly the
#   prespecified operations for this variant.
#
# Geometry remains identical to Main.
# -------------------------------------------------------------------------

branch_field = ["sup", "unsup_teacher", "unsup_student"]

weak_pipeline = [
    dict(
        type="Resize",
        scale=(1333, 800),
        keep_ratio=True,
    ),
    dict(
        type="Pad",
        size_divisor=32,
    ),
    dict(
        type="PackDetInputs",
        meta_keys=(
            "img_id",
            "img_path",
            "ori_shape",
            "img_shape",
            "scale_factor",
            "flip",
            "flip_direction",
            "homography_matrix",
        ),
    ),
]

strong_pipeline = [
    dict(
        type="Resize",
        scale=(1333, 800),
        keep_ratio=True,
    ),
    dict(
        type="S911GrayscalePhotometricAblation",
        ablation_augmentation_variant="A2",
        brightness_range=(0.90, 1.10),
        contrast_range=(0.90, 1.10),
    ),
    dict(
        type="Pad",
        size_divisor=32,
    ),
    dict(
        type="PackDetInputs",
        meta_keys=(
            "img_id",
            "img_path",
            "ori_shape",
            "img_shape",
            "scale_factor",
            "flip",
            "flip_direction",
            "homography_matrix",
        ),
    ),
]

unlabeled_pipeline = [
    dict(
        type="LoadImageFromFile",
        color_type="color",
    ),
    dict(
        type="LoadEmptyAnnotations",
    ),
    dict(
        type="MultiBranch",
        branch_field=branch_field,
        unsup_teacher=weak_pipeline,
        unsup_student=strong_pipeline,
    ),
]

unlabeled_dataset = dict(
    type="CocoDataset",
    ann_file="/workspace/ssod/data/coco/instances_unlabeled_10pct.json",
    data_prefix=dict(
        img="/workspace/ssod/data/mmdet_images",
    ),
    metainfo=dict(
        classes=(
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
        ),
    ),
    filter_cfg=dict(
        filter_empty_gt=False,
    ),
    test_mode=False,
    pipeline=unlabeled_pipeline,
)

labeled_dataset = dict(
    type="CocoDataset",
    ann_file="/workspace/ssod/data/coco/instances_labeled_10pct.json",
    data_prefix=dict(
        img="/workspace/ssod/data/mmdet_images",
    ),
    metainfo=dict(
        classes=(
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
        ),
    ),
    filter_cfg=dict(
        filter_empty_gt=False,
    ),
    test_mode=False,
    pipeline=[
        dict(
            type="LoadImageFromFile",
            color_type="color",
        ),
        dict(
            type="LoadAnnotations",
            with_bbox=True,
        ),
        dict(
            type="Resize",
            scale=(1333, 800),
            keep_ratio=True,
        ),
        dict(
            type="Pad",
            size_divisor=32,
        ),
        dict(
            type="MultiBranch",
            branch_field=branch_field,
            sup=dict(
                type="PackDetInputs",
            ),
        ),
    ],
)

# Replace only the training dataloader definition with a semantically exact
# copy of frozen R50-10% Main whose Student strong-view photometric component
# is explicitly parameterized by the S9.11 A-variant above.
train_dataloader = dict(
    _delete_=True,
    batch_size=8,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(
        type="MultiSourceSampler",
        batch_size=8,
        source_ratio=[1, 1],
        shuffle=True,
    ),
    batch_sampler=None,
    dataset=dict(
        type="ConcatDataset",
        datasets=[
            labeled_dataset,
            unlabeled_dataset,
        ],
    ),
)

# Preserve every custom import required by frozen Main and add only the
# S9.11 augmentation transform implementation.
custom_imports = dict(
    imports=[
        "src.hooks.teacher_initialization_hook",
        "src.hooks.actual_update_mean_teacher_hook",
        "src.hooks.ssl_training_summary_hook",
        "src.hooks.ssl_latest_resume_checkpoint_hook",
        "src.hooks.ssl_best_last_checkpoint_hook",
        "src.utils.resume_checkpoint_hook",
        "src.utils.actual_update_validation_loop",
        "src.utils.actual_update_budget_hook",
        "src.transforms.grayscale_photometric",
        "src.transforms.s9_11_grayscale_photometric_ablation",
        "src.detectors.empty_pseudo_safe_soft_teacher",
        "src.metrics.protocol_coco_metric",
    ],
    allow_failed_imports=False,
)