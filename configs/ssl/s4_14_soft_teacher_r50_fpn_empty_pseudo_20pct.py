_base_ = './s4_13_soft_teacher_r50_fpn_batch_20pct.py'

custom_imports = dict(
    imports=[
        'src.hooks.teacher_initialization_hook',
        'src.hooks.actual_update_mean_teacher_hook',
        'src.hooks.ssl_training_summary_hook',
        'src.hooks.ssl_latest_resume_checkpoint_hook',
        'src.utils.resume_checkpoint_hook',
        'src.utils.actual_update_validation_loop',
        'src.utils.actual_update_budget_hook',
        'src.transforms.grayscale_photometric',
        'src.detectors.empty_pseudo_safe_soft_teacher',
    ],
    allow_failed_imports=False,
)

model = dict(
    type='EmptyPseudoSafeSoftTeacher',
)
