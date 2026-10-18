"""S9.11 OFAT ablation — confidence-threshold variant C0.

Scientific identity
-------------------
Method        : SSL
Architecture  : Faster R-CNN R50-FPN
Label budget  : 10%
Seeds         : all 10 locked training seeds
Family        : CONFIDENCE_THRESHOLD
Variant       : C0

Scientific question
-------------------
How does changing the RCNN classification pseudo-label confidence threshold
affect pseudo-label quality and detector performance?

Frozen Main reference
---------------------
cls_pseudo_thr                 = 0.90
strong photometric augmentation= brightness + contrast
bbox-regression filtering      = reg_uncertainty < 0.02

OFAT change
-----------
cls_pseudo_thr: 0.90 -> 0.50

Everything else remains identical to frozen Main, including:
    initial pseudo score threshold
    RPN pseudo threshold
    strong augmentation
    regression filtering
    EMA
    labeled/unlabeled ratio
    loss weights
    optimizer
    learning-rate schedule
    update budget = 2064
    validation evaluator
    BEST/LAST checkpoint semantics

Thesis interpretation
---------------------
C0 is compared against OFF_SSL_R50_B10_Sxx using the SAME
training seed. The paired effect is:

    metric(C0, seed) - metric(Main, same seed)

Evaluation policy
-----------------
Non-Main ablation variant.
Validation only.
Not eligible for final test.
"""

_base_ = "../ssl/s5_06_soft_teacher_r50_fpn_scientific_eval_10pct.py"

# S9.11 OFAT:
# Change ONLY the RCNN classification pseudo-label confidence threshold.
model = dict(
    semi_train_cfg=dict(
        cls_pseudo_thr=0.50,
    ),
)