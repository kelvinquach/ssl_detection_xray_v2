_base_ = './s4_09_soft_teacher_swin_t_fpn_pseudo_thresholds.py'

model = dict(
    semi_train_cfg=dict(
        jitter_times=10,
        jitter_scale=0.06,
        reg_pseudo_thr=0.02,
        min_pseudo_bbox_wh=(0.01, 0.01),
    ),
)
