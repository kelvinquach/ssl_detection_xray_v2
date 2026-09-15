"""S5.06 — Q_pseudo accepted RCNN classification pseudo-label contract.

This config is intentionally budget-independent. It records the locked
pseudo-label acceptance semantics used when evaluating Q_pseudo on fixed
validation with the LAST EMA Teacher.
"""

checkpoint_role = "LAST_EMA_TEACHER"
model_role = "EMA_TEACHER"
serialization_scope = "EMA_TEACHER_ONLY"

dataset = "fixed_validation"
validation_ann_file = "/workspace/ssod/data/coco/instances_val.json"
pseudo_set = "accepted_rcnn_classification"

detector_score_floor = 0.05
pseudo_label_initial_score_thr = 0.50
rpn_pseudo_thr = 0.90
cls_pseudo_thr = 0.90
acceptance_comparator = "STRICT_GREATER"

rcnn_nms_iou = 0.50
rcnn_max_per_img = 100

evaluator_config = "configs/evaluation/s5_02_coco_secondary_evaluator.py"
primary_metric = "PL_mAP_50_95"
hidden_u_gt_used = False
test_used = False
