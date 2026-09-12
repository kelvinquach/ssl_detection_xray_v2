_base_ = './s4_07_soft_teacher_r50_fpn_strong.py'

model = dict(
    detector=dict(
        test_cfg=dict(
            rcnn=dict(
                score_thr=0.05,
            ),
        ),
    ),
    semi_train_cfg=dict(
        freeze_teacher=True,
        pseudo_label_initial_score_thr=0.50,
        rpn_pseudo_thr=0.90,
        cls_pseudo_thr=0.90,
    ),
)