"""S4.07 — unlabeled strong view routed to Student for Swin-T-FPN."""

_base_ = [
    "./s4_06_soft_teacher_swin_t_fpn_weak.py",
]

branch_field = {{_base_.branch_field}}
weak_pipeline = {{_base_.weak_pipeline}}

custom_imports = dict(
    imports=[
        "src.hooks.teacher_initialization_hook",
        "src.hooks.actual_update_mean_teacher_hook",
        "src.utils.resume_checkpoint_hook",
        "src.transforms.grayscale_photometric",
    ],
    allow_failed_imports=False,
)

strong_pipeline = [
    dict(type="Resize", scale=(1333, 800), keep_ratio=True),
    dict(
        type="GrayscaleBrightnessContrast",
        brightness_range=(0.90, 1.10),
        contrast_range=(0.90, 1.10),
    ),
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

unlabeled_pipeline = [
    dict(type="LoadImageFromFile", color_type="color"),
    dict(type="LoadEmptyAnnotations"),
    dict(
        type="MultiBranch",
        branch_field=branch_field,
        unsup_teacher=weak_pipeline,
        unsup_student=strong_pipeline,
    ),
]

# S4.07 scope:
# - the same unlabeled image now produces both weak and strong views.
# - weak view remains unchanged and routes to "unsup_teacher".
# - strong view routes to "unsup_student", consumed by SoftTeacher Student.
# - strong photometric factors are independently sampled from continuous
#   Uniform(0.90, 1.10).
# - R=G=B is preserved.
# - no additional geometric augmentation, random erasing, or mosaic.
# - Resize geometry is identical between weak and strong branches.