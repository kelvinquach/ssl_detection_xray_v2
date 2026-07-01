# Phase 1D — Kappa Feasibility / Limitation-aware Analysis

_Generated 2026-07-01T15:09:11.999071+00:00._

## 1. Objective

Assess whether inter-rater agreement (Cohen's / Fleiss' Kappa) is computable on the controlled scope and, where computable, report it as data-quality evidence. Agreement is NEVER a model metric or a decision criterion for split/model/threshold.

## 2. Inputs and Scope

- Input: `data\interim\vinbigdata_phase1C_scope_annotations.csv`
- Total images: 4894; total rows: 37596.
- Metadata only: no image, DICOM, header, or dimension was read.

## 3. rad_id Availability

- rad_id available: True (column: rad_id).
- rad_id missing/null/empty count: 0.
- Radiologists total (distinct): 17.

## 4. Radiologists per Image

| radiologists_per_image | image_count |
|---|---|
| 3 | 4894 |

Each image has a uniform number of 3 radiologist ratings. Across the dataset, there are 17 distinct radiologists. Therefore, the panel size is fixed per image, but the exact radiologist identities may vary across images (same_rater_identity_panel_across_images=False).

## 5. Image-Class-Radiologist Binary Matrix Feasibility

- binary_matrix_feasible: **True**.
- Each image has a uniform number of radiologist ratings; 'No finding' rows carry rad_id, so a rater's read-coverage is known. A rater who read an image but did not mark class C is a VALID negative. The complete image x rater matrix per class is therefore constructible.
- Each image has a uniform number of 3 radiologist ratings. Across the dataset, there are 17 distinct radiologists. Therefore, the panel size is fixed per image, but the exact radiologist identities may vary across images (same_rater_identity_panel_across_images=False).

## 6. Cohen's Kappa Feasibility

- cohen_kappa_feasible: **False**.
- Panel has 3 raters per image, not 2; Cohen's Kappa (a two-rater statistic) is not the natural choice. Fleiss' Kappa is used instead. Pairwise Cohen could be computed per rater-pair if specifically required.

## 7. Fleiss' Kappa Feasibility

- fleiss_kappa_feasible: **True**.
- Uniform per-image rater count (3) with inferable negatives satisfies Fleiss' assumptions; per-class binary Fleiss' Kappa is computed. Rater identities may differ across images, which Fleiss' Kappa permits (it does not require the same raters per item).
- Mean Fleiss' Kappa across abnormal classes: **0.4879**.

## 8. Class-wise Image-level Agreement Feasibility

- Summary: 14 abnormal classes assessed; 14 with feasible Fleiss' Kappa; mean kappa=0.4879.
- Full per-class Fleiss kappa and coverage in `phase1D_classwise_agreement_feasibility.csv`.

| class_id | class_name | positive_images | fleiss_kappa |
|---|---|---|---|
| 0 | Aortic enlargement | 3067 | 0.6393 |
| 1 | Atelectasis | 186 | 0.3568 |
| 2 | Calcification | 452 | 0.396 |
| 3 | Cardiomegaly | 2300 | 0.7065 |
| 4 | Consolidation | 353 | 0.3397 |
| 5 | ILD | 386 | 0.4604 |
| 6 | Infiltration | 613 | 0.4119 |
| 7 | Lung Opacity | 1322 | 0.3414 |
| 8 | Nodule/Mass | 826 | 0.4946 |
| 9 | Other lesion | 1134 | 0.3024 |
| 10 | Pleural effusion | 1032 | 0.6711 |
| 11 | Pleural thickening | 1981 | 0.3583 |
| 12 | Pneumothorax | 96 | 0.7402 |
| 13 | Pulmonary fibrosis | 1617 | 0.6126 |

## 9. Rare Class Instability Risk for Kappa

- Summary: 5/14 classes carry kappa_instability_risk (severe=2, moderate=3, low=9); risk is prevalence/rarity-driven, not measured instability.
- This is a **prevalence/rarity-driven RISK**, not a measurement of computed instability. Risk tiers: severe (positive_images<100 or prevalence<0.05) and moderate (positive_images<500 or prevalence<0.10). A class may carry multiple flags.
- Per-class risk flags in `phase1D_rare_class_kappa_instability.csv`.

## 10. Label-level Agreement vs BBox-level Consistency

- label_level_agreement_status: evaluable_fleiss_computed.
- bbox_level_consistency_status: evaluated_descriptive_only.
- These are kept strictly separate. Label-level agreement uses present/absent decisions; bbox proximity is descriptive only. Near-duplicate boxes are retained (not fused) and are not treated as confirmed annotation errors.

### Note on Phase 1B vs Phase 1D near-duplicate counts

The near-duplicate count in Phase 1B and the bbox-pair count in Phase 1D use different counting units and serve different purposes. Phase 1B reports near-duplicate candidate bbox records for annotation-quality review, where each bbox involved in at least one IoU ≥ 0.95 pair may be listed as a candidate record. Phase 1D reports the number of bbox pairs with IoU ≥ 0.95 as descriptive inter-rater spatial consistency evidence.

Therefore, the Phase 1B count of 147 candidate records and the Phase 1D count of 78 near-duplicate bbox pairs are not contradictory. If 78 pairs were completely disjoint, they could involve up to 156 bbox records. The observed Phase 1B count of 147 candidate records is lower than 156 because some bbox records may participate in multiple near-duplicate pairs and are counted once as candidate records in Phase 1B.

Phase 1D additionally focuses on inter-rater spatial consistency by excluding same-rater bbox pairs. This keeps bbox-level consistency separate from label-level agreement. These counts are descriptive only; no bbox is deleted, fused, or treated as a confirmed annotation error.


## 11. Limitations

- Negatives are inferred from read-coverage (rater read image but did not mark class C). This assumes 'No finding' / absence of a positive row faithfully encodes a negative decision, which is the VinBigData labelling convention.
- Panel size is 3; Fleiss' Kappa is appropriate, while Cohen's Kappa applies only to two-rater strata.
- Kappa instability RISK: 2 class(es) at severe risk and 3 at moderate risk from low positive count / prevalence imbalance. This is a prevalence-driven risk, not measured instability; such classes can yield deflated Kappa (the well-known kappa paradox under class imbalance).
- 0.95 IoU near-duplicate bbox candidates are retained as multi-reader evidence; they are not fused or deleted, and are not used to conclude annotation errors.

## 12. Decision

Kappa/agreement analysis in Phase 1D is used ONLY as data-quality evidence and limitation evidence. It is NOT used as a model metric, NOT used to select split/model/threshold, NOT used to modify annotations, and NOT used to evaluate SSL performance. Because every image has a uniform number of radiologist ratings (identities may vary across images) and 'No finding' rows encode read-coverage, negatives are validly inferable and per-class Fleiss' Kappa IS computed and reported as agreement evidence.

## 13. Definition of Done

- dod_status: **PASS_agreement_computed_and_documented**.
- DoD is met when agreement feasibility, computed Kappa (where valid), and limitations are documented and exported. No agreement value is used for modelling. Send outputs to review before ticking the checklist.
