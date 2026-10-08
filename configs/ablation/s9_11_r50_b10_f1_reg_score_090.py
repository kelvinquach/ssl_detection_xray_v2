"""S9.11 OFAT ablation — regression-filter variant F1.

Scientific identity
-------------------
Method        : SSL
Architecture  : Faster R-CNN R50-FPN
Label budget  : 10%
Seeds         : all 10 locked training seeds
Family        : REGRESSION_PSEUDO_LABEL_FILTERING
Variant       : F1

Scientific question
-------------------
How does the criterion used to select pseudo-labels for RCNN bbox regression
affect pseudo-label quality and detector performance?

Frozen Main reference — F2
--------------------------
initial pseudo score > 0.50
AND bbox-regression pseudo-label reliability uses:
    reg_uncertainty < 0.02

F1 scientific meaning
-------------------------------
initial pseudo score > 0.50
then bbox-regression pseudo-labels require confidence score > 0.90

OFAT invariant
--------------
Only the pseudo-label selection rule for the RCNN bbox-regression branch
changes.

The following remain identical to frozen Main:
    RCNN classification threshold = 0.90
    RPN pseudo threshold = 0.90
    initial pseudo score threshold = 0.50
    brightness + contrast strong augmentation
    EMA
    labeled/unlabeled ratio
    loss weights
    optimizer
    learning-rate schedule
    update budget = 2064
    validation evaluator
    BEST/LAST checkpoint semantics

Important
---------
F1 is confidence-based bbox-regression filtering.
It MUST NOT be interpreted as reg_pseudo_thr = 0.90 because reg_pseudo_thr
represents regression uncertainty, not classification confidence.

Thesis interpretation
---------------------
F1 is compared against F2/Main OFF_SSL_R50_B10_Sxx using the
SAME training seed.

Evaluation policy
-----------------
Non-Main ablation variant.
Validation only.
Not eligible for final test.
"""

_base_ = "../ssl/s5_06_soft_teacher_r50_fpn_scientific_eval_10pct.py"

# The inherited frozen Main configuration remains intact.
# Only the detector class and the locked S9.11 regression-filter mode change.
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
        "src.detectors.s9_11_ablation_regression_filter_soft_teacher",
        "src.metrics.protocol_coco_metric",
    ],
    allow_failed_imports=False,
)

model = dict(
    type="S911AblationRegressionFilterSoftTeacher",
    ablation_regression_variant="F1",
)