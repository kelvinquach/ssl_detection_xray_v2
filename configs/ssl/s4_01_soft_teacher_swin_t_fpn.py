"""S4.01 — SoftTeacher Student/Teacher initialization path for Swin-T-FPN."""

_base_ = [
    "../supervised/s3_02_faster_rcnn_swin_t_fpn_sup.py",
]

default_scope = "mmdet"

_supervised_detector = {{_base_.model}}
custom_imports = dict(
    imports=["src.hooks.teacher_initialization_hook"],
    allow_failed_imports=False,
)

custom_hooks = [
    dict(
        type="TeacherInitializationHook",
        priority="VERY_HIGH",
    ),
]

model = dict(
    _delete_=True,
    _scope_="mmdet",
    type="SoftTeacher",
    detector=_supervised_detector,
    semi_train_cfg=dict(
        freeze_teacher=True,
    ),
    semi_test_cfg=dict(
        predict_on="teacher",
        forward_on="teacher",
        extract_feat_on="teacher",
    ),
)

# S4.01 scope only:
# - Student and Teacher share the exact same detector architecture config.
# - Teacher is frozen.
# - Fresh-run Teacher(t0)=Student(t0) is established by the initialization
#   synchronization performed before the first training iteration.
# - No supervised burn-in.
# - EMA cadence/timing, AMP-skip handling, pseudo-label filtering,
#   augmentation, loss weighting, and checkpoint policy belong to later S4 tasks.
