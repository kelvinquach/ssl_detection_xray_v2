# Phase 3A - Dataset Diagnostics Before Training

| field | value |
| --- | --- |
| phase | 3A |
| protocol_version | 1.0.0 |
| protocol_status | `RESEARCHER_APPROVED_LOCKED` |
| protocol_sha256 | `b03474cce0be773796f9d458e6273b8fd2b955c1b961bcccc4d955eb3bb0dcf5` |
| phase_status | `OPEN_REVIEW_REQUIRED` |
| generated_at_utc | `2026-08-19T10:45:36Z` |

This report is descriptive pre-training dataset diagnostics. It is not training, not evaluation, not hyperparameter search, not pseudo-label generation and not an ablation study. `phase_status` is `OPEN_REVIEW_REQUIRED`; the script cannot and does not declare the phase PASS or CLOSED.

Throughout the report **METHOD** states a locked rule, definition or formula, and **RESULT** states an observed statistic.

## 1. Scope and leakage policy

**METHOD.** `primary_diagnostic_scope = FIXED_TRAIN`. Detailed diagnostics are computed only on `instances_train.json` and on the labeled subsets L_1pct, L_5pct, L_10pct, L_20pct. The canonical masters are used for integrity reference only. Validation and test are read only through a structural-summary function that returns image count, annotation count, negative count, category ids and a checksum; the parsed validation/test documents are discarded inside that function. Test is reserved for final evaluation.

**RESULT.** Detailed scopes actually opened during this run: `labeled_10pct, labeled_1pct, labeled_20pct, labeled_5pct, train`. Structural-only scopes opened: `canonical, test, val`.

## 2. Dataset lineage and integrity

**METHOD.** Every locked input is hashed with SHA-256 before the diagnostics and re-hashed afterwards. A mismatch against the locked value, or any change during the run, is a hard failure.

**RESULT.**

| input | sha256 |
| --- | --- |
| coco_master | 36f09d1b1477ea4a63153a04d775c938752e224c26079a0d44881c14b9bb4d75 |
| coco_master_jpg | f587152278f713460ff1e727a2912248a47052f6abc48de8f7bad6e8a63b94c0 |
| labeled_10pct | 6814b5b9ba8dfcd4af483d97c1a6b7e7ab26914bdd02bbb923502b43ecf2fdf6 |
| labeled_1pct | 1e267c547a8f535f6a69088da4735bd12c0188fae1b49e9d785b3e1e6883df98 |
| labeled_20pct | 6a8b5bf9c59baea41eefb0611e15a387257255b6fc59a3de607bfc2e80801a14 |
| labeled_5pct | ebb98f32cad612fe48adbbd6b5da8aa5db10bec478d1c7fd9288a9a93bbc3763 |
| phase2F1_protocol | ba5b1a1adce67c3f1cf9dd46657e3db89c9d29b85cc37a744462c55a617d3234 |
| phase2F1_seed_manifest | fdc9db80fcdbc575f93307edb9c992a7bb778770d7e9ff98e32796cd54f3a40b |
| phase2F1_seed_state_manifest | a19bea7d5b48b52b128d541588b255d06092e94036f9f104b54421ff652bb81e |
| phase2F_class_distribution_csv | c94566ea06607ddaa9d8d25b42b57234e7cf217f01d114c41bc278c206567263 |
| phase2F_deterministic_reconstruction_check | 674abe36bd582e6836a054a7d7949d8d77ad2de5fe0fb2b8e233988f15fed8a5 |
| phase2F_leakage_check | c3d26982c4d63dca6bc817f01d589b5d096ff10cbb2dcaaf8557172d1e0ac9cc |
| phase2F_lock_manifest | d8659d6fe40f9a32de0d32833c22dccc804da15009e3131d2640f7b9fb473c88 |
| phase2F_negative_distribution_csv | 6924854e30fb52ae54079b395192517c58b74ed7d797c4f5bc146142d163b69f |
| phase2F_nested_split_check | 41a6d9f36e44dbeb826b1d95696f9223c3a773957ff1350cbd0c8188ec89640c |
| phase2F_protocol | fb75e389fa1ab6b065206a529d274cbc809ad047ce6c0d519504c672ada49e10 |
| phase2F_validation_report | 7b541eb09c60321b56d06ef4918a686c14a7c62eaa9965161ff1144f9a7b4467 |
| split_lock_manifest | b696ec0382c7520cfd8c7871211018b8ee0ffd98e62c72b1060bc4f7d646012b |
| test | e1a73110e92af2656276d6c532035afe474b6ac6f2b2f03849834c036e1c00a4 |
| train | 0f3c37a6f1b5bcc6971b01fd4c69c7a2a3a1e8bee145c11488c7a658dcbbebe3 |
| val | 33064f47ba690be13e9d418d8ec5d4a6a2bec8482c24ea3eadf83fb33d01762a |

Preflight hard checks: 45 executed, 0 failed, 0 warnings.

## 3. Fixed training-set overview

**METHOD.** The fixed training set is the Phase 2E split `instances_train.json`, locked with `partition_seed = 42`. It is not regenerated or modified here.

**RESULT.** 3426 images, 25260 bounding-box annotations, 350 zero-GT negative images, 3076 positive images, 14 detection categories.

## 4. Class distribution

**METHOD.** For class c, `N_c_img` is the number of unique training images with at least one annotation of class c (an image is counted once per class regardless of how many boxes it carries) and `P_c_img = N_c_img / 3426`. `N_c_bbox` is the bounding-box annotation count; `bbox_annotation_share = N_c_bbox / total_train_annotations`; `bbox_per_positive_image = N_c_bbox / N_c_img`. Class order is image support descending, then category_id ascending.

**RESULT.**

| category_id | class_name | image_count | image_prevalence | bbox_annotation_count | bbox_annotation_share | bbox_per_positive_image | rare_flag |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | Aortic enlargement | 2144 | 0.6258026853473438 | 5021 | 0.19877276326207444 | 2.341884328358209 | false |
| 4 | Cardiomegaly | 1607 | 0.46906012842965555 | 3827 | 0.15150435471100554 | 2.3814561294337273 | false |
| 12 | Pleural thickening | 1384 | 0.40396964389959134 | 3415 | 0.13519398258115597 | 2.467485549132948 | false |
| 14 | Pulmonary fibrosis | 1130 | 0.3298307063631057 | 3271 | 0.12949326999208235 | 2.8946902654867257 | false |
| 8 | Lung Opacity | 923 | 0.26941039112667836 | 1728 | 0.0684085510688836 | 1.8721560130010835 | false |
| 10 | Other lesion | 791 | 0.23088149445417397 | 1501 | 0.05942201108471892 | 1.897597977243995 | false |
| 11 | Pleural effusion | 720 | 0.21015761821366025 | 1779 | 0.07042755344418052 | 2.470833333333333 | false |
| 9 | Nodule/Mass | 576 | 0.1681260945709282 | 1790 | 0.07086302454473475 | 3.107638888888889 | false |
| 7 | Infiltration | 428 | 0.12492702860478692 | 879 | 0.03479809976247031 | 2.053738317757009 | false |
| 3 | Calcification | 314 | 0.09165207238762405 | 665 | 0.02632620744259699 | 2.1178343949044587 | false |
| 6 | ILD | 268 | 0.07822533566841798 | 658 | 0.026049089469517023 | 2.455223880597015 | false |
| 5 | Consolidation | 246 | 0.07180385288966724 | 390 | 0.015439429928741092 | 1.5853658536585367 | false |
| 2 | Atelectasis | 130 | 0.03794512551079977 | 194 | 0.007680126682501979 | 1.4923076923076923 | true |
| 13 | Pneumothorax | 66 | 0.01926444833625219 | 142 | 0.0056215360253365 | 2.1515151515151514 | true |

Figure: `plots/dataset/class_distribution.png` (Panel A image support, Panel B bounding-box annotation count, identical class order).

## 5. Class-imbalance assessment

**METHOD.** `R_max_min = max(N_c_img) / min(N_c_img)` over the 14 detection classes. `CV = sample_SD(N_c_img) / mean(N_c_img)` with `ddof = 1` is reported as a secondary descriptive statistic. Strong imbalance is a scientific finding, not a Phase 3A failure.

**RESULT.**

| metric | value | unit |
| --- | --- | --- |
| max_image_support | 2144 | count |
| min_image_support | 66 | count |
| median_image_support | 648.0 | count |
| mean_image_support | 766.2142857142857 | count |
| max_image_prevalence | 0.6258026853473438 | fraction |
| min_image_prevalence | 0.01926444833625219 | fraction |
| imbalance_ratio_max_over_min | 32.484848484848484 | ratio |
| coefficient_of_variation_image_support | 0.8025330318219815 | ratio (secondary) |
| rare_class_count | 2 | count |
| rare_class_list | Atelectasis; Pneumothorax | class names |
| rare_threshold_used | 0.05 | locked primary |
| total_train_images | 3426 | count |

## 6. Rare-class analysis

**METHOD.** `rare_definition_type = PRE_SPECIFIED_OPERATIONAL_DEFINITION`. `Rare(c)` holds when `P_c_img < 0.05` (strictly less than). With a fixed training set of 3426 images this is `N_c_img <= 171` rare and `N_c_img >= 172` non-rare. Rarity is never defined by bounding-box counts and the threshold was not chosen from any histogram.

**RESULT.** 2 of 14 classes are rare under the locked definition.

| category_id | class_name | image_count | image_prevalence |
| --- | --- | --- | --- |
| 2 | Atelectasis | 130 | 0.03794512551079977 |
| 13 | Pneumothorax | 66 | 0.01926444833625219 |

**Interpretation.** In this report a rare class is a dataset-level low-support operational category defined on the fixed training set. It is not a claim that the abnormality is clinically rare; no separate clinical evidence is used here.

## 7. Bounding-box annotation distribution

**METHOD.** `N_c_bbox` counts COCO annotation records of class c. The quantity is a bounding-box annotation count; bounding-box annotation records must not be interpreted as a count of independently resolved clinical lesions.

**RESULT.** Total bounding-box annotations on the fixed training set: 25260. Per-class counts and shares are in section 4 and in `reports/03A_class_distribution.csv`.

## 8. Bounding-box count per image

**METHOD.** `B_i` is the number of bounding-box annotation records of training image i. Statistics are reported for all training images and for positive training images only, with sample SD (`ddof = 1`) and numpy `linear` percentiles.

**RESULT.**

| population | n | mean | sd_ddof1 | median | p95 | max |
| --- | --- | --- | --- | --- | --- | --- |
| all_train_images | 3426 | 7.373029772329247 | 5.252660582729625 | 6.0 | 17.0 | 48.0 |
| positive_train_images_only | 3076 | 8.211963589076722 | 4.882548972439222 | 7.0 | 18.0 | 48.0 |

## 9. Bounding-box geometry

**METHOD.** COCO boxes are `[x, y, w, h]` in absolute pixels with image size `W x H`. `w_n = w/W`, `h_n = h/H`, `a_n = (w*h)/(W*H)`, `AR = w/h`. The primary scale descriptor is `normalized_area`. Every training annotation must satisfy `w > 0`, `h > 0`, `x >= 0`, `y >= 0`, `x + w <= W`, `y + h <= H` and `0 < a_n <= 1`; a violation is a hard failure and no box is clamped, repaired, deleted, moved or resized.

**RESULT.**

| variable | role | n | mean | sd | min | p05 | p25 | median | p75 | p95 | max |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| normalized_area | primary | 25260 | 0.030439067292574274 | 0.03977297736072103 | 2.288818359375e-05 | 0.0008757952870094797 | 0.005535241167860546 | 0.014892746614044168 | 0.044680372084384336 | 0.10033503078233698 | 0.9384266703859281 |
| normalized_width | primary | 25260 | 0.17715881319220736 | 0.13425012425835253 | 0.00390625 | 0.02978444805582061 | 0.08441742330608844 | 0.12868820139981332 | 0.2340637450199203 | 0.4596466707104926 | 0.9572115384615385 |
| normalized_height | primary | 25260 | 0.13665024588574368 | 0.11256443191840465 | 0.0009894459102902375 | 0.023111979166666668 | 0.06631944444444444 | 0.11284722222222222 | 0.16028323141293851 | 0.3692973839518783 | 0.9803754266211604 |
| aspect_ratio | primary | 25260 | 1.4098767703617139 | 1.0852498664829426 | 0.08723540731237973 | 0.4446883976157665 | 0.7464454976303317 | 0.9853479853479854 | 1.7900748374432585 | 3.542250573290138 | 45.0 |
| raw_area | secondary | 25260 | 217463.1753365004 | 286965.39659662114 | 180.0 | 6276.1500000000015 | 38801.5 | 106090.5 | 310106.5 | 724923.3999999998 | 4575318.0 |
| w | secondary | 25260 | 441.42917656373714 | 333.4988080019325 | 11.0 | 74.0 | 208.0 | 323.0 | 593.0 | 1143.0 | 2938.0 |
| h | secondary | 25260 | 389.4964370546318 | 325.68858619170965 | 3.0 | 66.0 | 189.0 | 319.0 | 453.0 | 1075.0 | 2777.0 |

## 10. Normalized-area Small/Medium/Large analysis

**METHOD.** `bbox_size_definition_type = PRE_SPECIFIED_OPERATIONAL_DEFINITION`. Small is `a_n < 0.01`, Medium is `0.01 <= a_n < 0.1`, Large is `a_n >= 0.1`. Boundaries are exact: `a_n == 0.01` is Medium and `a_n == 0.1` is Large.

**RESULT.**

| size_category | count | percentage_of_annotations |
| --- | --- | --- |
| small | 9372 | 37.1021377672209 |
| medium | 14612 | 57.84639746634996 |
| large | 1276 | 5.051464766429137 |

**Interpretation.** Small, Medium and Large are normalized-area-based operational categories defined by the locked thresholds 0.01 and 0.10 on normalized bounding-box area. They are not the standard COCO pixel-area categories.

Figure: `plots/dataset/bbox_distribution.png`.

## 11. Bounding-box location analysis

**METHOD.** `center_x_n = (x + w/2)/W` and `center_y_n = (y + h/2)/H` with a top-left origin, x from left to right and y from top to bottom. The density map uses a locked 50 x 50 grid over `[0,1] x [0,1]` and is annotation-weighted.

**RESULT.** Figure `plots/dataset/bbox_location_heatmap.png`; the per-annotation centres are in `reports/03A_bbox_distribution.csv`.

**Interpretation.** The heatmap describes bounding-box annotations. It is not by itself a density of unique clinical lesions.

## 12. Negative / No Finding analysis

**METHOD.** No Finding is a zero-ground-truth negative image, not a detection category and never a fifteenth class. On the fixed training set the positive count, negative count and negative prevalence are reported. Validation and test contribute structural counts only.

**RESULT.**

| scope | diagnostic_role | image_count | positive_image_count | negative_image_count | negative_prevalence |
| --- | --- | --- | --- | --- | --- |
| train | PRIMARY_DETAILED | 3426 | 3076 | 350 | 0.10215995329830706 |
| val | STRUCTURAL_ONLY | 734 | 659 | 75 | 0.10217983651226158 |
| test | STRUCTURAL_ONLY | 734 | 659 | 75 | 0.10217983651226158 |
| labeled_1pct | LABELED_SUBSET_DETAILED | 34 | 31 | 3 | 0.08823529411764706 |
| labeled_5pct | LABELED_SUBSET_DETAILED | 171 | 154 | 17 | 0.09941520467836257 |
| labeled_10pct | LABELED_SUBSET_DETAILED | 343 | 308 | 35 | 0.10204081632653061 |
| labeled_20pct | LABELED_SUBSET_DETAILED | 685 | 615 | 70 | 0.10218978102189781 |

Figure: `plots/dataset/negative_image_distribution.png`.

## 13. Labeled-budget diagnostics

**METHOD.** For each budget b and class c, `P_b,c_img` is the image-level prevalence inside L_b and `D_b,c = |P_b,c_img - P_train,c_img|`. The rare flag is always the fixed-train flag; rarity is never recomputed inside a budget. All figures are recomputed independently from the labeled COCO files; `reports/02F_class_distribution.csv` is a non-gating cross-check. The hidden ground truth of the unlabeled complements is not used. This analysis is descriptive and can never trigger a rebuild of L_b, whose membership is locked.

**RESULT.**

| budget | labeled_image_count | negative_image_count | negative_prevalence | class_coverage_out_of_14 | max_absolute_deviation | mean_absolute_deviation |
| --- | --- | --- | --- | --- | --- | --- |
| 1pct | 34 | 3 | 0.08823529411764706 | 14 | 0.012980323477902539 | 0.006991763430415951 |
| 5pct | 171 | 17 | 0.09941520467836257 | 14 | 0.0028574062125541547 | 0.0015424580131004736 |
| 10pct | 343 | 35 | 0.10204081632653061 | 14 | 0.0012781695114874037 | 0.0007462472461732155 |
| 20pct | 685 | 70 | 0.10218978102189781 | 14 | 0.0006626015740515689 | 0.00033811855241795496 |

Per-class detail is in `reports/03A_labeled_budget_coverage.csv`.

## 14. Label-cardinality analysis

**METHOD.** `LC_i` is the number of unique detection classes present in training image i; a negative image has `LC_i = 0`. Bounding-box counts are never substituted for unique class counts.

**RESULT.**

| cardinality | image_count | percentage |
| --- | --- | --- |
| 0 | 350 | 10.215995329830706 |
| 1 | 225 | 6.567425569176883 |
| 2 | 839 | 24.489200233508466 |
| 3 | 702 | 20.490367775831874 |
| 4 | 530 | 15.469935785172213 |
| 5 | 372 | 10.85814360770578 |
| 6 | 209 | 6.100408639813193 |
| 7 | 130 | 3.794512551079977 |
| 8 | 51 | 1.4886164623467601 |
| 9 | 14 | 0.40863981319322823 |
| 10 | 4 | 0.11675423234092236 |

| statistic | value |
| --- | --- |
| mean | 3.1310566258026853 |
| sd_ddof1 | 1.9319938322591697 |
| median | 3.0 |
| p95 | 7.0 |
| max | 10.0 |

## 15. Class co-occurrence

**METHOD.** Role is SECONDARY / NON-GATING. Co-occurrence uses unique image-level class presence, never bounding-box counts. For classes c and d, `N_cd` is the number of training images containing both and `J(c,d) = N_cd / (N_c + N_d - N_cd)`. All 91 unordered pairs are emitted, sorted by `category_id_a` then `category_id_b`. Co-occurrence never changes the training design.

**RESULT.** 91 unordered pairs written to `reports/03A_class_cooccurrence.csv`. The ten highest Jaccard coefficients:

| class_name_a | class_name_b | n_a_images | n_b_images | n_both_images | jaccard |
| --- | --- | --- | --- | --- | --- |
| Aortic enlargement | Cardiomegaly | 2144 | 1607 | 1373 | 0.5773759461732548 |
| Pleural effusion | Pleural thickening | 720 | 1384 | 581 | 0.381483913328956 |
| Pleural thickening | Pulmonary fibrosis | 1384 | 1130 | 665 | 0.35965386695511087 |
| Aortic enlargement | Pleural thickening | 2144 | 1384 | 927 | 0.356401384083045 |
| Lung Opacity | Pleural effusion | 923 | 720 | 393 | 0.3144 |
| Lung Opacity | Pulmonary fibrosis | 923 | 1130 | 481 | 0.3059796437659033 |
| Lung Opacity | Pleural thickening | 923 | 1384 | 502 | 0.27811634349030473 |
| Aortic enlargement | Pulmonary fibrosis | 2144 | 1130 | 644 | 0.24486692015209124 |
| Other lesion | Pleural thickening | 791 | 1384 | 427 | 0.244279176201373 |
| Cardiomegaly | Pleural thickening | 1607 | 1384 | 587 | 0.24417637271214643 |

## 16. Threshold sensitivity analysis

**METHOD.** Role is SECONDARY / NON-GATING. The rare threshold is varied over [0.01, 0.05, 0.1] with primary 0.05; the small boundary over [0.005, 0.01, 0.02] with the large boundary fixed at 0.1; the large boundary over [0.05, 0.1, 0.2] with the small boundary fixed at 0.01. The variation is one factor at a time; no 3 x 3 Cartesian grid is run.

**RESULT.**

| analysis_type | parameter | threshold | is_primary | category | count | percentage | primary_percentage | delta_percentage_points |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| rare_threshold_summary | rare_image_prevalence_threshold | 0.01 | false | rare_class_count | 0 | 0.0 | 14.285714285714286 | -14.285714285714286 |
| rare_threshold_summary | rare_image_prevalence_threshold | 0.05 | true | rare_class_count | 2 | 14.285714285714286 | 14.285714285714286 | 0.0 |
| rare_threshold_summary | rare_image_prevalence_threshold | 0.1 | false | rare_class_count | 5 | 35.714285714285715 | 14.285714285714286 | 21.428571428571427 |
| bbox_small_boundary_sensitivity | bbox_small_threshold | 0.005 | false | small | 5941 | 23.519398258115597 | 37.1021377672209 | -13.582739509105306 |
| bbox_small_boundary_sensitivity | bbox_small_threshold | 0.005 | false | medium | 18043 | 71.42913697545526 | 57.84639746634996 | 13.582739509105302 |
| bbox_small_boundary_sensitivity | bbox_small_threshold | 0.005 | false | large | 1276 | 5.051464766429137 | 5.051464766429137 | 0.0 |
| bbox_small_boundary_sensitivity | bbox_small_threshold | 0.01 | true | small | 9372 | 37.1021377672209 | 37.1021377672209 | 0.0 |
| bbox_small_boundary_sensitivity | bbox_small_threshold | 0.01 | true | medium | 14612 | 57.84639746634996 | 57.84639746634996 | 0.0 |
| bbox_small_boundary_sensitivity | bbox_small_threshold | 0.01 | true | large | 1276 | 5.051464766429137 | 5.051464766429137 | 0.0 |
| bbox_small_boundary_sensitivity | bbox_small_threshold | 0.02 | false | small | 14879 | 58.9034045922407 | 37.1021377672209 | 21.801266825019795 |
| bbox_small_boundary_sensitivity | bbox_small_threshold | 0.02 | false | medium | 9105 | 36.04513064133017 | 57.84639746634996 | -21.801266825019795 |
| bbox_small_boundary_sensitivity | bbox_small_threshold | 0.02 | false | large | 1276 | 5.051464766429137 | 5.051464766429137 | 0.0 |
| bbox_large_boundary_sensitivity | bbox_large_threshold | 0.05 | false | small | 9372 | 37.1021377672209 | 37.1021377672209 | 0.0 |
| bbox_large_boundary_sensitivity | bbox_large_threshold | 0.05 | false | medium | 10386 | 41.11638954869359 | 57.84639746634996 | -16.730007917656373 |
| bbox_large_boundary_sensitivity | bbox_large_threshold | 0.05 | false | large | 5502 | 21.78147268408551 | 5.051464766429137 | 16.730007917656373 |
| bbox_large_boundary_sensitivity | bbox_large_threshold | 0.1 | true | small | 9372 | 37.1021377672209 | 37.1021377672209 | 0.0 |
| bbox_large_boundary_sensitivity | bbox_large_threshold | 0.1 | true | medium | 14612 | 57.84639746634996 | 57.84639746634996 | 0.0 |
| bbox_large_boundary_sensitivity | bbox_large_threshold | 0.1 | true | large | 1276 | 5.051464766429137 | 5.051464766429137 | 0.0 |
| bbox_large_boundary_sensitivity | bbox_large_threshold | 0.2 | false | small | 9372 | 37.1021377672209 | 37.1021377672209 | 0.0 |
| bbox_large_boundary_sensitivity | bbox_large_threshold | 0.2 | false | medium | 15647 | 61.94378463974663 | 57.84639746634996 | 4.097387173396669 |
| bbox_large_boundary_sensitivity | bbox_large_threshold | 0.2 | false | large | 241 | 0.9540775930324624 | 5.051464766429137 | -4.097387173396675 |

Classes whose rare status is not identical across [0.01, 0.05, 0.1]: Atelectasis, Calcification, Consolidation, ILD, Pneumothorax

`primary_threshold_changed_after_diagnostics = false`; `sensitivity_used_for_threshold_selection = false`. Threshold sensitivity analysis was used only to assess robustness of descriptive conclusions and was not used to reselect primary thresholds.

## 17. Fixed split structural summary

**METHOD.** Structural fields only: image count, image percentage, annotation count, negative count, negative percentage, the official JSON SHA-256 and, where available, the locked image-membership SHA-256. No class distribution, bbox size, label cardinality, co-occurrence or heatmap is computed for validation or test.

**RESULT.**

| split | image_count | image_percentage | annotation_count | negative_count | negative_percentage | official_json_sha256 | image_membership_sha256 | diagnostic_usage |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| train | 3426 | 70.00408663669799 | 25260 | 350 | 10.215995329830706 | 0f3c37a6f1b5bcc6971b01fd4c69c7a2a3a1e8bee145c11488c7a658dcbbebe3 | 628b9bb8ba25129a928abe994b101b4c4efd5588d389feb60da6de2a371fa11a | PRIMARY_DETAILED |
| val | 734 | 14.997956681651 | 5399 | 75 | 10.217983651226158 | 33064f47ba690be13e9d418d8ec5d4a6a2bec8482c24ea3eadf83fb33d01762a | 87c23ebed4d1e6965731fc0b31245859f49e777119813c6152efde3531ba58c6 | STRUCTURAL_INTEGRITY_ONLY |
| test | 734 | 14.997956681651 | 5437 | 75 | 10.217983651226158 | e1a73110e92af2656276d6c532035afe474b6ac6f2b2f03849834c036e1c00a4 | 1f7903e069e872bf2e5fe13bb4d0fa257dc4a1c2c8290a621d3f7286ada66b37 | STRUCTURAL_INTEGRITY_ONLY |

## 18. Pre-training dataset risks

**METHOD.** Risks are stated as descriptive dataset properties observed before training. None of them is a prediction about model behaviour: no claim is made about average precision, detection failure, pseudo-label quality, semi-supervised superiority, convergence or training stability.

**RESULT.**

- Class support is unbalanced; the observed `imbalance_ratio_max_over_min` is reported in section 5.
- 2 of 14 classes fall below the locked rare threshold and therefore have low support in the fixed training set.
- The Small share of bounding boxes under the locked normalized-area definition is reported in section 10.
- Zero-GT negative images are present at the prevalence reported in section 12 and carry no boxes by construction.
- Rare-class support inside the smallest labeled budget is reported per class in `reports/03A_labeled_budget_coverage.csv`.

The descriptive statistics reported above characterize potential pre-training dataset risks. Their magnitude is reported directly rather than classified using additional post-hoc labels, and none of these observations constitutes a Phase 3A protocol failure.

## 19. Limitations

- Phase 3A is descriptive. It contains no model, no training run and no evaluation, so no statement about achievable detection performance can be derived from it.
- Rare, Small, Medium and Large are pre-specified operational categories of this study, not external standards.
- Bounding-box counts are annotation records. Multiple radiologist annotations of the same finding are not resolved here, so a count is not a count of unique clinical lesions.
- Validation and test were summarised structurally only, so no statement about their content distribution is made.
- The unlabeled complements U_b were not inspected, so nothing is claimed about their label distribution.
- Sensitivity analysis covers only the ranges listed in section 16 and one factor at a time.

## 20. Leakage statement

Detailed diagnostics were restricted to the fixed training set and legitimately labeled subsets. Validation and test were not used for diagnostic threshold selection or methodological tuning. The hidden original ground truth of the unlabeled subsets was not used in Phase 3A diagnostics.

Primary diagnostic definitions were specified before Phase 3A execution and were not selected or modified based on the observed diagnostic distributions.

Threshold sensitivity analysis was used only to assess robustness of descriptive conclusions and was not used to reselect primary thresholds.

## 21. Conclusions before training

**METHOD.** Conclusions are restricted to the claim boundaries of the protocol: class rarity under the locked definition, imbalance magnitude, size-category percentages, spatial concentration, negative prevalence, labeled-budget coverage and descriptive robustness under sensitivity.

**RESULT.** The fixed training set, the four labeled budgets and the locked split all match their recorded identities and counts; the descriptive properties above are documented for use as pre-training context. `phase_status = OPEN_REVIEW_REQUIRED`. This report does not declare Phase 3A PASS or CLOSED; the researcher and the GPT review decide.

