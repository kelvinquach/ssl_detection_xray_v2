_base_ = './s4_11_soft_teacher_r50_fpn_regression_reliability.py'

model = dict(
    semi_train_cfg=dict(
        sup_weight=1.0,
        unsup_weight=4.0,
    ),
)
