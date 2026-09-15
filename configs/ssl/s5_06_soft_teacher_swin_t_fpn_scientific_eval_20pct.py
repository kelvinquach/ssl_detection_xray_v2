"""S5.06 controlled repair — SSL scientific BEST/LAST EMA Teacher path."""

_base_ = [
    "./s4_14_soft_teacher_swin_t_fpn_empty_pseudo_20pct.py",
    "./s5_06_ssl_scientific_validation_common.py",
]

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
        "src.detectors.empty_pseudo_safe_soft_teacher",
        "src.metrics.protocol_coco_metric",
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
    dict(
        type="ActualUpdateBudgetSchedulerHook",
        target_updates=2064,
    ),
    dict(
        type="SSLBestLastCheckpointHook",
        metric_key="coco/bbox_mAP",
    ),
    dict(
        type="SSLLatestResumeCheckpointHook",
        refresh_interval=172,
    ),
    dict(
        type="SSLTrainingSummaryHook",
        expected_optimizer_updates=2064,
        effective_labeled_batch=4,
        effective_unlabeled_batch=4,
        priority="LOW",
    ),
]
