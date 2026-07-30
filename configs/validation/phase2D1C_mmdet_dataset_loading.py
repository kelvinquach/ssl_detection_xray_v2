# Phase 2D.1C - MMDetection dataset-loading & empty-image retention validation.
#
# This is a REFERENCE / DOCUMENTATION config for the two datasets that the
# validation script (scripts/02D1C_validate_mmdet_dataset_loading.py) builds
# through the official MMDetection registry. It is a plain MMEngine-style config
# so it can be `Config.fromfile`-loaded for cross-checking, but the script does
# not depend on it: the script builds datasets directly and measures behaviour.
#
# It is NOT a training config: no model, no optimizer, no schedule, no
# pretrained weights, no train/val/test split, no evaluator that could drop
# zero-GT images. filter_empty_gt=False is mandatory for the retention dataset.

# --- Paths (override at runtime via the script CLI) ------------------------- #
ann_file = "data/processed/coco/coco_master_jpg.json"
data_root_img = "data/processed/images_jpg"

# --- 14 abnormal detection classes, ordered by ascending COCO category id ---- #
# The contiguous training label is the POSITION in this tuple (cat2label), not
# the raw COCO category_id. No "No Finding" / background class is added.
classes = (
    "Aortic enlargement",   # cat_id 1  -> label 0
    "Atelectasis",          # cat_id 2  -> label 1
    "Calcification",        # cat_id 3  -> label 2
    "Cardiomegaly",         # cat_id 4  -> label 3
    "Consolidation",        # cat_id 5  -> label 4
    "ILD",                  # cat_id 6  -> label 5
    "Infiltration",         # cat_id 7  -> label 6
    "Lung Opacity",         # cat_id 8  -> label 7
    "Nodule/Mass",          # cat_id 9  -> label 8
    "Other lesion",         # cat_id 10 -> label 9
    "Pleural effusion",     # cat_id 11 -> label 10
    "Pleural thickening",   # cat_id 12 -> label 11
    "Pneumothorax",         # cat_id 13 -> label 12
    "Pulmonary fibrosis",   # cat_id 14 -> label 13
)
metainfo = dict(classes=classes)

# --- Minimal, non-augmenting validation pipeline ---------------------------- #
# Loading / retention only; no RandomFlip / Resize / Crop / photometric aug.
validation_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="LoadAnnotations", with_bbox=True),
    dict(type="PackDetInputs",
         meta_keys=("img_id", "img_path", "ori_shape", "img_shape")),
]

# --- A. Retention dataset (the phase's subject) ----------------------------- #
retention_dataset = dict(
    type="CocoDataset",
    ann_file=ann_file,
    data_prefix=dict(img=data_root_img),
    metainfo=metainfo,
    filter_cfg=dict(filter_empty_gt=False),  # MANDATORY: keep all 4,894 images
    test_mode=False,
    pipeline=validation_pipeline,
)

# --- B. Controlled comparison dataset --------------------------------------- #
# Identical to A except filter_empty_gt=True, to MEASURE (not assume) the
# filtering behaviour and compare removed image IDs against the zero-GT set.
controlled_comparison_dataset = dict(
    type="CocoDataset",
    ann_file=ann_file,
    data_prefix=dict(img=data_root_img),
    metainfo=metainfo,
    filter_cfg=dict(filter_empty_gt=True),
    test_mode=False,
    pipeline=[],
)

# --- Dataloader guidance ---------------------------------------------------- #
# batch_size=1 for the primary validation, or pseudo_collate for batch_size>1
# so different-sized un-resized images batch as a list of tensors. num_workers=0
# is the deterministic default.
dataloader_notes = dict(
    primary_batch_size=1,
    num_workers=0,
    collate="pseudo_collate",
    sampler="deterministic (Subset with fixed, unshuffled indices)",
)

# --- Locked expectations (measured, never hard-coded to PASS) ---------------- #
expected = dict(
    images=4894,
    annotations=36096,
    categories=14,
    empty_images=500,
    filter_empty_gt_false_length=4894,
    filter_empty_gt_true_length=4394,
    removed_equals_zero_gt=True,
)

# This phase never authorizes training.
training_authorized = False
