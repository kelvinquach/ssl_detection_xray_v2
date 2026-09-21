"""S7.P01 — NON-OFFICIAL tiny end-to-end SSL R50 pilot."""

_base_ = ["../ssl/s5_06_soft_teacher_r50_fpn_scientific_eval_1pct.py"]

train_cfg = dict(type="ActualUpdateValidationIterBasedTrainLoop", max_iters=1, val_interval=1)

custom_hooks = [
    dict(type="TeacherInitializationHook", priority="VERY_HIGH"),
    dict(type="ProtocolResumeCheckpointHook"),
    dict(type="ActualUpdateMeanTeacherHook", momentum=0.001, skip_buffers=True, priority="HIGH"),
    dict(type="ActualUpdateBudgetSchedulerHook", target_updates=1),
    dict(type="SSLBestLastCheckpointHook", metric_key="coco/bbox_mAP"),
    dict(type="SSLLatestResumeCheckpointHook", refresh_interval=1),
    dict(type="SSLTrainingSummaryHook", expected_optimizer_updates=1, effective_labeled_batch=4, effective_unlabeled_batch=4, priority="LOW"),
]
