"""S4.02 — actual-optimizer-update Mean Teacher EMA for R50-FPN."""
_base_ = [
    "./s4_01_soft_teacher_r50_fpn.py",
]

custom_imports = dict(
    imports=[
        "src.hooks.teacher_initialization_hook",
        "src.hooks.actual_update_mean_teacher_hook",
        "src.utils.resume_checkpoint_hook",
    ],
    allow_failed_imports=False,
)

custom_hooks = [
    dict(
        type="TeacherInitializationHook",
        priority="VERY_HIGH",
    ),
    dict(type="ProtocolResumeCheckpointHook"),
    dict(
        type="ActualUpdateMeanTeacherHook",
        momentum=0.001,
        skip_buffers=True,
        priority="HIGH",
    ),
]
