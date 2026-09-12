"""S4.06 — unlabeled weak view routed to EMA Teacher for R50-FPN."""

_base_ = [
    "./s4_05_soft_teacher_r50_fpn_amp.py",
]

_detector_data_preprocessor = {{_base_.model.detector.data_preprocessor}}

branch_field = ["sup", "unsup_teacher", "unsup_student"]

weak_pipeline = [
    dict(type="Resize", scale=(1333, 800), keep_ratio=True),
    dict(type="Pad", size_divisor=32),
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

unlabeled_weak_pipeline = [
    dict(type="LoadImageFromFile", color_type="color"),
    dict(type="LoadEmptyAnnotations"),
    dict(
        type="MultiBranch",
        branch_field=branch_field,
        unsup_teacher=weak_pipeline,
    ),
]

model = dict(
    data_preprocessor=dict(
        type="MultiBranchDataPreprocessor",
        data_preprocessor=_detector_data_preprocessor,
    ),
)

# S4.06 scope only:
# - same unlabeled image produces the weak Teacher branch.
# - weak branch key is exactly "unsup_teacher", consumed by SoftTeacher Teacher.
# - weak image transforms are Resize + Pad only.
# - standard normalization/channel conversion remains in DetDataPreprocessor.
# - no flip, crop, rotation, or photometric augmentation.
# - homography_matrix is retained for later pseudo-box projection.
# - unsup_student/strong augmentation belongs to the next S4 task.