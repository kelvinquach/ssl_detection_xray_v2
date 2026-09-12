"""S4.13 — SSL paired batch composition and labeled-exposure budget."""

_base_ = './s4_12_soft_teacher_r50_fpn_loss_weight.py'

classes = (
    'Aortic enlargement',
    'Atelectasis',
    'Calcification',
    'Cardiomegaly',
    'Consolidation',
    'ILD',
    'Infiltration',
    'Lung Opacity',
    'Nodule/Mass',
    'Other lesion',
    'Pleural effusion',
    'Pleural thickening',
    'Pneumothorax',
    'Pulmonary fibrosis',
)

branch_field = ['sup', 'unsup_teacher', 'unsup_student']

sup_pipeline = [
    dict(type='LoadImageFromFile', color_type='color'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', scale=(1333, 800), keep_ratio=True),
    dict(type='Pad', size_divisor=32),
    dict(
        type='MultiBranch',
        branch_field=branch_field,
        sup=dict(type='PackDetInputs'),
    ),
]

weak_pipeline = [
    dict(type='Resize', scale=(1333, 800), keep_ratio=True),
    dict(type='Pad', size_divisor=32),
    dict(
        type='PackDetInputs',
        meta_keys=(
            'img_id',
            'img_path',
            'ori_shape',
            'img_shape',
            'scale_factor',
            'flip',
            'flip_direction',
            'homography_matrix',
        ),
    ),
]

strong_pipeline = [
    dict(type='Resize', scale=(1333, 800), keep_ratio=True),
    dict(
        type='GrayscaleBrightnessContrast',
        brightness_range=(0.9, 1.1),
        contrast_range=(0.9, 1.1),
    ),
    dict(type='Pad', size_divisor=32),
    dict(
        type='PackDetInputs',
        meta_keys=(
            'img_id',
            'img_path',
            'ori_shape',
            'img_shape',
            'scale_factor',
            'flip',
            'flip_direction',
            'homography_matrix',
        ),
    ),
]

unlabeled_pipeline = [
    dict(type='LoadImageFromFile', color_type='color'),
    dict(type='LoadEmptyAnnotations'),
    dict(
        type='MultiBranch',
        branch_field=branch_field,
        unsup_teacher=weak_pipeline,
        unsup_student=strong_pipeline,
    ),
]

labeled_dataset = dict(
    type='CocoDataset',
    ann_file='/workspace/ssod/data/coco/instances_labeled_1pct.json',
    data_prefix=dict(img='/workspace/ssod/data/mmdet_images'),
    metainfo=dict(classes=classes),
    filter_cfg=dict(filter_empty_gt=False),
    test_mode=False,
    pipeline=sup_pipeline,
)

unlabeled_dataset = dict(
    type='CocoDataset',
    ann_file='/workspace/ssod/data/coco/instances_unlabeled_1pct.json',
    data_prefix=dict(img='/workspace/ssod/data/mmdet_images'),
    metainfo=dict(classes=classes),
    filter_cfg=dict(filter_empty_gt=False),
    test_mode=False,
    pipeline=unlabeled_pipeline,
)

train_dataloader = dict(
    batch_size=8,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(
        type='MultiSourceSampler',
        batch_size=8,
        source_ratio=[1, 1],
        shuffle=True,
    ),
    batch_sampler=None,
    dataset=dict(
        type='ConcatDataset',
        datasets=[labeled_dataset, unlabeled_dataset],
    ),
)

optim_wrapper = dict(
    accumulative_counts=1,
)

train_cfg = dict(
    type='ActualUpdateValidationIterBasedTrainLoop',
    max_iters=1032,
    val_interval=172,
)

param_scheduler = [
    dict(
        type='MultiStepLR',
        by_epoch=False,
        end=1033,
        milestones=[688, 946],
        gamma=0.1,
    ),
]

custom_imports = dict(
    imports=[
        'src.hooks.teacher_initialization_hook',
        'src.hooks.actual_update_mean_teacher_hook',
        'src.hooks.ssl_training_summary_hook',
        'src.utils.resume_checkpoint_hook',
        'src.utils.actual_update_validation_loop',
        'src.utils.actual_update_budget_hook',
        'src.transforms.grayscale_photometric',
    ],
    allow_failed_imports=False,
)

custom_hooks = [
    dict(
        type='TeacherInitializationHook',
        priority='VERY_HIGH',
    ),
    dict(type='ProtocolResumeCheckpointHook'),
    dict(
        type='ActualUpdateMeanTeacherHook',
        momentum=0.001,
        skip_buffers=True,
        priority='HIGH',
    ),
    dict(
        type='ActualUpdateBudgetSchedulerHook',
        target_updates=1032,
    ),
    dict(
        type='SSLTrainingSummaryHook',
        expected_optimizer_updates=1032,
        effective_labeled_batch=4,
        effective_unlabeled_batch=4,
        priority='LOW',
    ),
]