# SSOD + VLM — METRIC / OUTPUT CONTRACT v1.0 LOCKED

**Tên file:** `SSOD_VLM_METRIC_OUTPUT_CONTRACT_V1_0_LOCKED_VI.md`  
**Ngày khóa:** 2026-11-17  
**Trạng thái:** **LOCKED — METRIC / FIELD / OUTPUT CONTRACT**  
**Phạm vi:** B0, B1, B2, B3; validation; training provenance; candidate-level evidence; VLM evidence; U/H reliability; falsification; sham controls; comparison tables; publication-ready outputs.  
**Training B2/B3:** **NOT AUTHORIZED by this contract alone.**  
**Final-test mới:** **NOT AUTHORIZED by this contract alone.**

---

## 0. Mục tiêu

Contract này khóa chính xác:

1. Metric nào phải tồn tại cho B0/B1/B2/B3.
2. Tên field chính thức.
3. Ý nghĩa và đơn vị từng metric.
4. File output tối thiểu bắt buộc.
5. Provenance/hashes cần lưu để audit.
6. Cách giữ khả năng so sánh B0/B1/B2/B3.
7. Metric riêng của B2.
8. Metric riêng của B3.
9. Schema so sánh giữa các phương pháp.
10. Quy tắc không được đổi tên/đổi semantics sau khi bắt đầu thực nghiệm.

Contract này **không thay đổi thuật toán B2/B3** và **không tạo kết quả thực nghiệm**.

---

# 1. Quy tắc thuật ngữ

Trong tài liệu và implementation mới:

- Không dùng các thuật ngữ mơ hồ bị cấm theo yêu cầu nghiên cứu.
- Phải nói rõ:
  - metric của detector sau huấn luyện;
  - metric pseudo-label;
  - metric candidate-level;
  - metric VLM;
  - metric false-positive;
  - metric so sánh B3−B2;
  - metric trên validation;
  - metric trên reused historical test hoặc independent external evaluation.

Tên trạng thái cho bước cân nhắc huấn luyện Student:

```text
GO_TO_STUDENT_TRAINING_CONSIDERATION
```

Tên này chỉ nghĩa:

> đủ điều kiện để xem xét huấn luyện Student ở bước sau.

Nó **không tự động cấp quyền training**.

---

# 2. Quy tắc scale và đơn vị — LOCKED

Các metric AP/mAP/AR/Recall/FAR/PPV/CR/FSR được lưu ở **raw scale [0,1]**.

Ví dụ:

```text
0.054
```

nghĩa là:

```text
5.4 AP points
```

Khi viết bài:

\[
AP\ points = 100 \times raw\ AP.
\]

Không gọi chênh lệch AP là “percentage points” nếu đại lượng gốc là AP.

Ví dụ:

```text
raw delta = -0.0111
report = -1.11 AP points
```

Counts như `FP_total`, `accepted_pseudo_count`, `n_reviewed` là số nguyên không âm.

Rates như `FP_per_image`, `FP_per_negative` có thể >1.

---

# 3. Metric detector chung B0/B1/B2/B3 — LOCKED

B0/B1/B2/B3 phải dùng **cùng metric definition và evaluator semantics**.

## 3.1 Primary detector endpoint

```text
bbox mAP@[0.50:0.95]
```

Operational field:

```text
bbox_mAP_50_95
```

hoặc trong evaluation master:

```text
test_mAP_50_95
```

COCO IoU grid:

```text
0.50, 0.55, ..., 0.95
```

Evaluator:

```text
useCats = 1
maxDets = 100
```

Không áp fixed score threshold trước COCO AP.

---

## 3.2 AP50

```text
AP50 = AP at IoU 0.50
```

Fields:

```text
AP50
test_AP50
```

Role:

> secondary detector metric.

---

## 3.3 AP75

```text
AP75 = AP at IoU 0.75
```

Fields:

```text
AP75
test_AP75
```

Role:

> secondary detector metric.

---

## 3.4 Average Recall

```text
AR@[0.50:0.95], maxDets=100, area=all
```

Fields:

```text
AR_50_95_max100
test_AR_50_95_max100
```

Role:

> secondary/descriptive detector metric.

---

# 4. Operating-point metrics — LOCKED

Fixed operating point:

```text
tau_eval = 0.50
IoU match threshold = 0.50
```

Matching:

```text
category-aware
one-to-one
deterministic score-descending greedy
```

Applied after:

```text
detector-native NMS
max 100 detections/image
```

`tau_eval=0.50` chỉ áp cho operating-point metrics, không áp trước COCO AP.

---

## 4.1 Recall

\[
Recall_{\tau=0.50}
=
\frac{TP}{TP+FN}.
\]

Field:

```text
test_Recall_tau_eval
```

Raw scale:

```text
[0,1]
```

---

## 4.2 FP total

Field:

```text
test_FP_total
```

Definition:

> tổng false-positive detections tại fixed operating point.

---

## 4.3 FP per image

\[
FP/image
=
\frac{\sum_i FP_i}{N}.
\]

Field:

```text
test_FP_per_image
```

---

# 5. Class-wise AP — LOCKED

Cho đủ 14 detection classes:

```text
Aortic enlargement
Atelectasis
Calcification
Cardiomegaly
Consolidation
ILD
Infiltration
Lung Opacity
Nodule/Mass
Other lesion
Pleural effusion
Pleural thickening
Pneumothorax
Pulmonary fibrosis
```

Metric:

\[
AP_c = AP_c@[0.50:0.95].
\]

Canonical master fields:

```text
AP_Aortic_enlargement
AP_Atelectasis
AP_Calcification
AP_Cardiomegaly
AP_Consolidation
AP_ILD
AP_Infiltration
AP_Lung_Opacity
AP_Nodule_Mass
AP_Other_lesion
AP_Pleural_effusion
AP_Pleural_thickening
AP_Pneumothorax
AP_Pulmonary_fibrosis
```

Không được:

- bỏ lớp vì AP thấp;
- thay threshold riêng theo lớp;
- chọn checkpoint riêng theo lớp.

Rare-class reporting giữ:

```text
Atelectasis
Pneumothorax
```

Metric:

```text
AP@[0.50:0.95]
```

Không tạo `rare-mAP` như primary metric.

---

# 6. No Finding / negative-image metrics — LOCKED

`No Finding`:

> không phải detection class.

Nó là:

> negative evaluation stratum với zero GT boxes.

## 6.1 FP total negative

Field:

```text
FP_total_negative
```

Definition:

> tổng detections trên negative images tại fixed operating point.

---

## 6.2 FP per negative

\[
FP/negative
=
\frac{
\sum_{i\in\mathcal N}FP_i
}{
|\mathcal N|
}.
\]

Field:

```text
FP_per_negative
```

Interpretation:

> average false detections per negative image.

---

## 6.3 Negative FAR

\[
FAR_{neg}
=
\frac{
\#\{i\in\mathcal N:FP_i\ge1\}
}{
|\mathcal N|
}.
\]

Field:

```text
negative_FAR
```

Raw scale:

```text
[0,1]
```

`FP_per_negative` và `negative_FAR` không được dùng thay thế nhau.

---

# 7. Validation history schema — EXACT LOCK

File:

```text
metrics/validation/validation_history.csv
```

Exact header đã audit:

```text
optimizer_update,
bbox_mAP_50_95,
AP50,
AP75,
AR_50_95_max100,
model_role,
checkpoint_candidate,
best_so_far,
checkpoint_path
```

Mỗi validation event phải lưu một row.

Không đổi field names nếu không có controlled schema amendment.

---

# 8. Class-wise validation AP schema — EXACT LOCK

File:

```text
metrics/validation/classwise_validation_ap.csv
```

Exact header:

```text
category_id,
class_name,
AP_50_95,
AP50,
AP75
```

Phải có 14 rows cho 14 detection classes khi evaluator hoàn tất hợp lệ.

---

# 9. Training summary schema — EXACT LOCK

File:

```text
metrics/validation/training_summary.json
```

Các keys đã audit:

```text
amp_skipped_step_count
effective_labeled_batch
effective_unlabeled_batch
ema_update_count
ema_updates_match_optimizer_updates
expected_optimizer_updates
expected_update_budget_reached
l_u_ratio
labeled_exposure_matches_update_count
labeled_images_consumed_total
observed_optimizer_updates
unlabeled_exposure_matches_update_count
unlabeled_images_consumed_total
validation_events_observed
```

B2/B3 phải giữ các keys này nếu semantics áp dụng.

Nếu field không áp dụng cho B0:

> dùng explicit `NOT_APPLICABLE`/null theo schema producer, không bịa giá trị.

---

# 10. Validation BEST checkpoint schema — EXACT LOCK

File:

```text
metrics/validation/validation_best_teacher.json
```

Keys đã audit:

```text
checkpoint_path
checkpoint_sha256
rule
selected_metric_value
selected_update
selection_metric
selection_model
selection_split
tie_break
```

---

# 11. Checkpoint-role contract — LOCKED

## B0

Official detector checkpoint:

> BEST validation checkpoint selected by highest validation bbox mAP@[0.50:0.95].

## B1

Official detector checkpoint:

> BEST EMA Teacher selected by highest validation bbox mAP@[0.50:0.95].

Q_pseudo source:

> LAST EMA Teacher.

## B2

Official detector checkpoint:

> BEST EMA Teacher selected by the **same rule as B1**.

Pseudo-label mechanism evaluation:

> LAST EMA Teacher where contract requires pseudo-label quality evaluation.

## B3

Official detector checkpoint:

> BEST EMA Teacher selected by the **same rule as B1/B2**.

Offline VLM/candidate reliability protocol:

> LAST EMA Teacher according to G4.5 fixed snapshot policy.

Không được so sánh:

```text
B3 BEST vs B2 LAST
```

hoặc bất kỳ asymmetric checkpoint choice nào.

---

# 12. Historical evaluation master schema — EXACT LOCK

Canonical historical file:

```text
artifacts/final_test/official_final_test_results.csv
```

Exact header đã audit:

```text
run_id
condition_role
method
architecture
budget
seed_index
training_seed
best_checkpoint_sha256
best_validation_mAP_50_95
test_mAP_50_95
test_AP50
test_AP75
test_AR_50_95_max100
test_Recall_tau_eval
test_FP_total
test_FP_per_image
AP_Aortic_enlargement
AP_Atelectasis
AP_Calcification
AP_Cardiomegaly
AP_Consolidation
AP_ILD
AP_Infiltration
AP_Lung_Opacity
AP_Nodule_Mass
AP_Other_lesion
AP_Pleural_effusion
AP_Pleural_thickening
AP_Pneumothorax
AP_Pulmonary_fibrosis
FP_total_negative
FP_per_negative
negative_FAR
Q_pseudo
Q_pseudo_status
final_evaluation_status
```

B0/B1 historical values được bảo toàn.

B2/B3 mới **không được tự động ghi vào historical final-test master** nếu chưa có evaluation authorization rõ ràng.

---

# 13. Historical B0/B1 paired-effects schema — EXACT LOCK

Existing file:

```text
artifacts/results/paired_effects.csv
```

Exact header đã audit:

```text
pair_id
architecture
budget
seed_index
training_seed
mAP_test_SUP
mAP_test_SSL
delta_mAP_test
FP_per_negative_test_SUP
FP_per_negative_test_SSL
delta_FP_per_negative_test
AP_test_Atelectasis_SUP
AP_test_Atelectasis_SSL
delta_AP_test_Atelectasis
AP_test_Pneumothorax_SUP
AP_test_Pneumothorax_SSL
delta_AP_test_Pneumothorax
Q_pseudo_validation_SSL
```

File này là **legacy B0/B1 evidence**.

Không sửa tên field để nhét B2/B3 vào.

---

# 14. Generic method-comparison schema — NEW LOCK

File mới cho nghiên cứu B2/B3:

```text
artifacts/results/method_pair_effects_v1.csv
```

Long-format schema:

```text
schema_version
comparison_id
architecture
budget
seed_index
training_seed

reference_method
comparison_method

reference_run_id
comparison_run_id

metric_name
metric_unit
reference_value
comparison_value
delta_comparison_minus_reference

evaluator_config_sha256
checkpoint_rule_match
training_seed_match
architecture_match
budget_match
pair_valid

evaluation_role
evaluation_set_id
evaluation_set_sha256
```

Required comparisons when data legally/technically exist:

```text
B1 - B0
B2 - B1
B3 - B2   [PRIMARY METHOD COMPARISON]
B3 - B1
B3 - B0
```

Delta convention is always:

\[
\Delta
=
comparison-reference.
\]

Không đổi dấu giữa tables.

---

# 15. Q_pseudo compatibility contract — LOCKED

Historical definition:

\[
Q_{pseudo}
=
PL\text{-}mAP@[0.50:0.95].
\]

Historical B1 setup:

```text
fixed validation set
LAST EMA Teacher
accepted RCNN classification pseudo-labels
```

Operational B1 file:

```text
pseudo/qpseudo_validation.json
```

Historical keys include:

```text
PL_AP50
PL_AP75
PL_mAP_50_95
acceptance
accepted_pseudo_count
checkpoint_role
checkpoint_sha256
dataset
evaluator_config_sha256
hidden_u_gt_used
pseudo_acceptance_config_sha256
pseudo_set
schema_version
test_used
validation_image_count
validation_split_sha256
```

## B0

```text
Q_pseudo = NA
Q_pseudo_status = NOT_APPLICABLE
```

## B1

Use historical definition unchanged.

## B2

Compatibility `Q_pseudo` is computed from the pseudo-label set that contributes non-zero positive RCNN classification supervision under B2 at the LAST EMA Teacher evaluation snapshot.

Because B2 mined positives use reduced classification weight, `Q_pseudo` remains a **set-quality metric**, not a weighted-loss metric.

## B3

Compatibility `Q_pseudo` is computed from the pseudo-label set that contributes non-zero positive RCNN classification supervision under B3 at the LAST EMA Teacher evaluation snapshot.

H2-suppressed high-confidence candidates have positive classification weight zero and therefore are excluded from this compatibility pseudo-label set, but their candidate records must remain preserved in B3-specific artifacts.

Important:

> `Q_pseudo` alone is insufficient to characterize B3.

B3 additionally requires U/H metrics below.

---


## 15A. Candidate identity semantics — PRE-EXECUTION CLARIFICATION — LOCKED

Clarification class:

    PRE-EXECUTION CLARIFICATION
    schema_version = "1.0.0"
    B2/B3 scientific artifacts existing before clarification = 0

This clarification does not add, remove, or rename any schema field.
It fixes previously unspecified semantics of already-required candidate
identity fields before any B2/B3 scientific execution.

The authoritative stable candidate identity is the typed tuple:

    (
        run_id: str,
        logical_batch_id: int,
        image_id: int,
        native_candidate_index: int
    )

Locked field semantics:

- run_id:
  exact logical-run identifier from the authoritative
  "run_root_manifest.json", field "run_id".
  During online training, it is retrieved from
  Path(runner.work_dir).parent / "run_root_manifest.json".
  During offline replay/evaluation, the same run_id MUST be carried from
  the corresponding source logical-run manifest; a new run_id MUST NOT be
  invented from the offline execution context.

- logical_batch_id:
  zero-based raw training-loop batch index for the currently consumed
  logical batch; specifically MMEngine before_train_iter(batch_idx) /
  current IterBasedTrainLoop self._iter.
  It MUST NOT use ActualOptimizerUpdateCounter.

- image_id:
  integer DetDataSample.img_id of the image.

- native_candidate_index:
  zero-based position in that image's native Teacher ROI prediction result,
  attached immediately after Teacher ROI prediction and BEFORE strict
  initial score > 0.50 filtering and before B2 adaptive filtering/mining.

candidate_id is the canonical compact JSON serialization of exactly the
four-element typed stable identity tuple, with no insignificant whitespace.

Example:

    ["OFF_SSL_R50_B10_S01",17,42,3]

Canonical construction:

    json.dumps(
        [run_id, logical_batch_id, image_id, native_candidate_index],
        ensure_ascii=False,
        separators=(",", ":"),
    )

candidate_id is an artifact/provenance representation. It is NOT the
ordering primitive for scientific selection.

Whenever deterministic candidate-ID ascending tie-breaking is required,
the authoritative comparison MUST use lexicographic ascending order over
the typed tuple:

    run_id ASC as string
    logical_batch_id ASC as integer
    image_id ASC as integer
    native_candidate_index ASC as integer

Implementations MUST NOT use cryptographic hash order, Python object hash,
bbox floating-point coordinates, or lexical sorting of the serialized
candidate_id string as a substitute for typed tuple ordering.

For candidates from the same run and logical batch, the effective
tie-break reduces to:

    image_id ASC
    native_candidate_index ASC

Resume/retry MUST reconstruct the same identity from the authoritative
run manifest, resumed raw loop iteration, image identifier, and native
Teacher ROI candidate position.

---

# 16. B2 reference candidate artifact — LOCKED

File:

```text
b2_reference_selection.parquet
```

Minimum candidate-level fields:

```text
protocol_version
run_id
architecture
budget
training_seed
checkpoint_role
checkpoint_sha256
validation_role
logical_batch_id
image_id
native_candidate_index
candidate_id

bbox_original_xyxy
bbox_teacher_xyxy
class_id
class_name
teacher_score
reg_unc

initial_keep
adaptive_tau
branch_H_or_U

K_c
q_c

b2_selected
b2_slot_id
selection_source
is_mined
classification_weight_multiplier

image_cap_state
global_cap_state

fifo_hash_before
fifo_hash_after
```

`selection_source` enum:

```text
H
M2
Z2
NOT_SELECTED
```

`classification_weight_multiplier`:

```text
H retained positive = 1.0
M2/Z2 mined positive = 0.5
not selected = NULL
```

---

# 17. B2 batch-level audit outputs — LOCKED

Mỗi logical batch phải có đủ counters để tái dựng:

```text
n_initial_candidates
n_H
n_U
K_c per class
q_c per class
n_M2_pre_global_cap
n_M2_final
n_Z2
n_active_classification_pseudo_labels
n_mined_positive_pseudo_labels

image_cap_hits
global_cap_hits
class_cap_hits

fifo_size_before per class
fifo_size_after per class
fifo_hash_before
fifo_hash_after

amp_step_skipped
optimizer_update_committed
```

Loss/audit fields:

```text
n_sampled_rois
n_positive_rois
n_negative_rois
n_positive_rois_assigned_to_mined
reference_weight_sum
weighted_positive_weight_sum
F_ref
classification_loss_before_method_weighting
classification_loss_after_method_weighting
```

Các field này dùng audit implementation, không tự động là publication endpoints.

---

# 18. B3 required artifact set — LOCKED

B3 phải materialize tối thiểu:

```text
offline_verification_manifest.json
teacher_candidate_pool.parquet
b2_reference_selection.parquet
vlm_request_manifest.jsonl
vlm_response_cache.jsonl
b3_variant_selection.parquet
falsification_controls.csv
matched_burden_controls.csv
gt_matching.parquet
offline_summary_by_class.csv
offline_audit_report.json
```

Có thể bổ sung file nhưng không được bỏ file bắt buộc.

---

# 19. Teacher candidate pool schema — LOCKED

File:

```text
teacher_candidate_pool.parquet
```

Fields:

```text
protocol_version
run_id
architecture
budget
training_seed
checkpoint_role
checkpoint_sha256
validation_role
logical_batch_id
image_id
native_candidate_index
candidate_id

bbox_original_xyxy
bbox_teacher_xyxy
class_id
class_name
teacher_score
reg_unc

initial_keep
adaptive_tau
branch_H_or_U
K_c
q_c

b2_selected
b2_slot_id

image_cap_state
global_cap_state

fifo_hash_before
fifo_hash_after
```

Stable identity:

```text
(run_id, logical_batch_id, image_id, native_candidate_index)
```

Không dùng floating-point bbox coordinates làm identity.

---

# 20. VLM request schema — LOCKED

File:

```text
vlm_request_manifest.jsonl
```

Fields:

```text
request_id
candidate_id
analysis_role
control_id
review_branch
review_order

model_id
model_revision
model_hash
processor_hash
prompt_hash
renderer_hash
decoding_config_hash
rendered_image_hash
vlm_cache_key

class_term

payload_audit_pass
contains_teacher_score
contains_teacher_rank
contains_b2_selected
contains_gt
contains_annotation
contains_clinical_metadata
```

Required:

```text
contains_teacher_score = false
contains_teacher_rank = false
contains_b2_selected = false
contains_gt = false
contains_annotation = false
contains_clinical_metadata = false
```

Any violation:

> technical protocol failure.

---

# 21. VLM response schema — LOCKED

File:

```text
vlm_response_cache.jsonl
```

Fields:

```text
request_id
candidate_id
vlm_cache_key

response_hash
raw_response_text
parse_status

decision
region_relevant
abstain_reason

latency_ms
analysis_role
control_id
```

`decision` enum:

```text
support
reject
uncertain
```

`region_relevant` enum:

```text
yes
no
uncertain
```

Malformed/timeout/parser failure:

> abstain/neutral.

Không suy diễn free text thành support/reject nếu parser contract không chấp nhận.

---

# 22. B3 variant-selection schema — LOCKED

File:

```text
b3_variant_selection.parquet
```

Fields:

```text
run_id
validation_role
logical_batch_id
image_id
candidate_id
class_id
class_name

variant_id
branch_H_or_U

K_c
q_c

b2_selected
b3_selected
b2_slot_id

selection_status
replacement_candidate_id

h_reviewed
h_valid_reject
h_region_valid_reject
h_suppressed

gate_U_state
gate_H_state

vlm_cache_key
fallback_reason

image_cap_state
global_cap_state
```

`variant_id` enum:

```text
B2
B3-U
B3-H
B3-UH
B3-ALL-OFF
```

`selection_status` enum:

```text
SHARED
REMOVED
GAINED
UNCHANGED
NOT_APPLICABLE
```

Required parity:

\[
B3_{\text{ALL-OFF}}=B2.
\]

---

# 23. Candidate ↔ annotation matching schema — LOCKED

File:

```text
gt_matching.parquet
```

Created only in annotation-relative scoring phase.

Fields:

```text
candidate_id
image_id
class_id
class_name

matching_rule_id
iou_threshold
matched_gt_annotation_id
matched_iou

match_status
Y_i
```

Locked:

```text
matching_rule_id = G45_CLASSWISE_GREEDY_ONE_TO_ONE_V1
iou_threshold = 0.50
```

`match_status` enum:

```text
MATCHED
UNMATCHED
EVAL_IGNORED
```

Mapping:

```text
MATCHED      -> Y_i = 1
UNMATCHED    -> Y_i = 0
EVAL_IGNORED -> Y_i = NULL
```

`Y_i=0` means unmatched relative to available annotations, not clinical disease absence.

---

# 24. B3 U-branch metrics — LOCKED

## 24.1 U support reliability

\[
PPV_c^U
=
\frac{
\sum_{i\in S_c}Y_i
}{
|S_c|
}
\]

where:

```text
S_c = U candidates with decision=support and region_relevant=yes
```

Required outputs:

```text
PPV_U
PPV_U_LCB
PPV_U_UCB
```

---

## 24.2 U fixed-slot utility

\[
Q_c=\sum_t q_{c,t}
\]

\[
U\_utility_c
=
\frac{
\sum_{i\in M^{(3)}_c}Y_i
-
\sum_{i\in M^{(2)}_c}Y_i
}{
Q_c
}.
\]

Required:

```text
U_utility
U_utility_LCB
U_utility_UCB
```

If denominator undefined/zero:

> `INSUFFICIENT`, not zero.

---

## 24.3 U coverage

\[
Coverage_U
=
\frac{
\# actionable\ U
}{
\# reviewed\ U
}.
\]

Required:

```text
U_coverage
```

---

## 24.4 U abstention

\[
Abstain_U
=
\frac{
\# abstain\ U
}{
\# reviewed\ U
}.
\]

Required:

```text
U_abstention
```

---

# 25. B3 H-branch metrics — LOCKED

## 25.1 Correct reject

\[
CR_c^H
=
\frac{
\sum_{i\in J_c}(1-Y_i)
}{
|J_c|
}.
\]

Required:

```text
CR_H
CR_H_LCB
CR_H_UCB
```

Không gọi `CR_H` là clinical specificity hoặc NPV.

---

## 25.2 False suppression rate

\[
FSR_c
=
\frac{
\sum_{i\in H_{suppress,c}}Y_i
}{
|H_{suppress,c}|
}.
\]

Required:

```text
FSR_H
FSR_H_LCB
FSR_H_UCB
```

---

## 25.3 H suppression count

Required:

```text
n_H_suppressions
n_H_false_suppressions
```

---

## 25.4 H coverage

\[
Coverage_H
=
\frac{
|H_{suppress}|
}{
|H_{reviewed}|
}.
\]

Field:

```text
H_coverage
```

---

## 25.5 H abstention

\[
Abstain_H
=
\frac{
\#H\ abstain
}{
\#H\ reviewed
}.
\]

Field:

```text
H_abstention
```

---

# 26. B3 per-class summary schema — LOCKED

File:

```text
offline_summary_by_class.csv
```

One row:

```text
class × action branch
```

Required columns:

```text
schema_version

class_id
class_name
action_branch

n_images
n_reviewed
n_actionable

n_support
n_replacements
q_total

n_reject
n_suppressions
n_false_suppressions

PPV_U
PPV_U_LCB
PPV_U_UCB

U_utility
U_utility_LCB
U_utility_UCB

U_coverage
U_abstention

CR_H
CR_H_LCB
CR_H_UCB

FSR_H
FSR_H_LCB
FSR_H_UCB

H_coverage
H_abstention

harm_real_minus_sham
harm_LCB
harm_UCB

bootstrap_reps
cluster_unit
family_size
alpha_adjusted

support_status
gate_status
reason_code
```

If metric not applicable to action branch:

> NULL, not zero.

`gate_status` enum:

```text
PASS
FAIL
INSUFFICIENT
```

---

# 27. U/H gate thresholds — LOCKED

## U eligibility

Need all:

```text
n_images >= 30
n_reviewed >= 50
n_support >= 20
n_replacements >= 20
```

Else:

```text
INSUFFICIENT
```

U PASS only if all:

\[
LCB(PPV_U)\ge0.50
\]

\[
LCB(U\_utility)>0
\]

\[
U\_utility\ge0.05.
\]

---

## H eligibility

Need all:

```text
n_images >= 30
n_reviewed >= 50
n_reject >= 20
n_suppressions >= 20
```

Else:

```text
INSUFFICIENT
```

H PASS only if all:

\[
LCB(CR_H)\ge0.90
\]

\[
UCB(FSR_H)\le0.10
\]

\[
UCB(harm_{real-sham})\le0.05
\]

and:

\[
harm_{real-sham}\le0.
\]

---

# 28. Statistical resampling — LOCKED

Primary cluster:

```text
validation image
```

Bootstrap:

```text
10,000 replicates
```

Seed:

```text
45111726
```

If >10% replicates undefined:

```text
INSUFFICIENT
```

Multiplicity:

```text
Bonferroni
```

separately for:

```text
U class family
H class family
```

Family-wise alpha:

```text
0.05
```

---

# 29. Falsification-control schema — LOCKED

File:

```text
falsification_controls.csv
```

Required columns:

```text
control_id
control_name
action_branch
class_id
class_name
analysis_role

n_images
n_pairs
numerator
denominator

metric_name
point_estimate
CI_low
CI_high

threshold
threshold_comparator
verdict

control_seed
control_manifest_hash
```

Must cover F1–F11.

`verdict` enum:

```text
PASS
FAIL
INSUFFICIENT
CONTROL_UNAVAILABLE
NOT_APPLICABLE
```

---

# 30. Critical falsification metrics — LOCKED

## Null-image persistence

Required:

```text
persist_null
persist_null_LCB
persist_null_UCB
```

## Region-shift persistence

Required:

```text
persist_shift
persist_shift_LCB
persist_shift_UCB
```

Critical branch requirement:

```text
>=30 distinct images
>=50 evaluable pairs
```

PASS if:

\[
UCB(persist_{null})<0.80
\]

and:

\[
UCB(persist_{shift})<0.80.
\]

---

## Unsupported-class canary

Required:

```text
canary_support_rate
canary_support_LCB
canary_support_UCB
```

PASS if:

\[
UCB(canary\_support\_rate)<0.20.
\]

---

# 31. Matched-burden sham schema — LOCKED

File:

```text
matched_burden_controls.csv
```

Columns:

```text
action_branch
class_id
class_name

real_intervention_count
sham_intervention_count

real_metric
sham_metric
real_minus_sham

difference_LCB
difference_UCB

sham_seed
sham_manifest_hash
verdict
```

U evidence-sham requirement:

\[
LCB(U\_utility_{real}-U\_utility_{sham})>0.
\]

H harm requirement:

\[
UCB(FSR_{real}-FSR_{sham})\le0.05.
\]

---

# 32. Intervention opportunity coverage — LOCKED

\[
InterventionCoverage
=
\frac{
\#GATE\ logical\ batches\ with\ge1\ actionable\ intervention
}{
\#GATE\ logical\ batches
}.
\]

Required fields:

```text
intervention_coverage
intervention_coverage_LCB
intervention_coverage_UCB
```

Condition for consideration of Student training:

\[
point\ estimate\ge0.05
\]

and:

\[
LCB>0.02.
\]

---

# 33. B3 offline decision artifact — LOCKED

File:

```text
offline_audit_report.json
```

Required keys:

```text
protocol_version

candidate_pool_hash
b2_reference_hash

phase_A_invariants_pass
all_off_parity_pass
confidence_blind_pass

historical_test_used_for_design
hidden_unlabeled_train_gt_used

U_gate_pass_classes
U_gate_fail_classes
U_gate_insufficient_classes

H_gate_pass_classes
H_gate_fail_classes
H_gate_insufficient_classes

critical_falsification_status
F11_status

intervention_coverage
intervention_coverage_LCB
intervention_coverage_UCB

global_status
global_reason

student_training_authorized
```

`global_status` enum:

```text
GO_TO_STUDENT_TRAINING_CONSIDERATION
HOLD
STOP_B3
TECHNICAL_FAIL
```

Until separate authorization:

```text
student_training_authorized = false
```

---

# 34. B3 model/request provenance — LOCKED

Must preserve:

```text
model_id
model_revision
model_hash
processor_hash
prompt_hash
renderer_hash
decoding_config_hash
rendered_image_hash
vlm_cache_key
response_hash
latency_ms
parse_status
analysis_role
control_id
```

No primary VLM result is valid without immutable provenance.

---

# 35. Validation DEV/GATE identity fields — LOCKED

All B2/B3 offline artifacts must contain:

```text
validation_role
validation_manifest_sha256
validation_image_id
```

Roles:

```text
VAL_DEV
VAL_GATE
```

No result may silently mix roles.

---

# 36. Fixed snapshot provenance — LOCKED

Offline B3 records must include:

```text
architecture
budget
training_seed
checkpoint_role
checkpoint_sha256
```

Primary checkpoint role:

```text
LAST_EMA_TEACHER
```

DEV:

```text
S01 = 204886845
```

GATE:

```text
S02 = 1480646854
S03 = 1798418854
```

No snapshot selection based on favorable VLM result.

---

# 37. B3 variant roles — LOCKED

```text
B3-U
```

Tests U re-ranking/replacement only.

```text
B3-H
```

Tests H classification-positive suppression only.

```text
B3-UH
```

Full B3: U + H.

```text
B3-ALL-OFF
```

Implementation parity/control:

\[
B3_{\text{ALL-OFF}}=B2.
\]

`B3-ALL-OFF` is not treated as an independent scientific method arm.

---

# 38. Detector metric compatibility for trained B2/B3 — LOCKED

If Student training is later authorized, every valid B2/B3 run must produce:

```text
metrics/validation/validation_history.csv
metrics/validation/classwise_validation_ap.csv
metrics/validation/training_summary.json
metrics/validation/validation_best_teacher.json
checkpoint_index.json
run_manifest.json
resolved_config.yaml
```

and the same detector metric semantics as B0/B1.

Evaluation results must be traceable to:

```text
run_id
architecture
budget
seed_index
training_seed
checkpoint_sha256
evaluator_config_sha256
evaluation_set_sha256
```

---

# 39. Primary method comparison B3−B2 — LOCKED

If Student training is later separately authorized:

For architecture \(a\), budget \(b\), seed \(s\):

\[
d_{a,b,s}
=
mAP^{B3}_{a,b,s}
-
mAP^{B2}_{a,b,s}.
\]

Metric:

```text
bbox mAP@[0.50:0.95]
```

Seed aggregate:

\[
D_s
=
\frac{1}{8}
\sum_{a=1}^{2}
\sum_{b=1}^{4}
d_{a,b,s}.
\]

Overall estimand:

\[
\mu_D
=
\frac{1}{10}
\sum_{s=1}^{10}D_s.
\]

All 8 architecture×budget cells have equal weight.

Primary exact test:

```text
one-sided exact sign-flip test
```

over 10 seed-level \(D_s\).

Total sign assignments:

```text
1024
```

Report additionally:

```text
mean
sample SD
two-sided 95% t CI
all 10 seed-level values
```

Sensitivity:

```text
10,000 seed bootstrap
leave-one-seed-out
```

---

# 40. Required secondary method comparisons — LOCKED

When valid measurements exist:

```text
B1 - B0
B2 - B1
B3 - B2
B3 - B1
B3 - B0
```

B3−B2 is primary for the new method.

B2−B1 must be reported because B3 improvement relative to B2 alone does not establish improvement over original SSL.

B3−B0 and B3−B1 contextualize whether the final method exceeds supervised and original SSL references.

---

# 41. Missing/NO-OP result rules — LOCKED

Technically valid run with no effective VLM intervention:

> retain in analysis.

Do not delete because effect is zero.

Technical failure:

> do not impute B2 as B3.

Pair missing after governed retry:

> primary paired matrix = incomplete.

No silent complete-case claim as confirmatory result.

---

# 42. Historical test firewall — LOCKED

The historical test set was already exposed in the earlier project.

Therefore:

- do not use it for model selection;
- do not use it for prompt selection;
- do not use it for threshold selection;
- do not use it for gate selection;
- do not use it for algorithm amendment;
- do not call it a fresh blind test.

If later explicitly evaluated:

> label as reused historical test / secondary evaluation.

A new claim of independent blind generalization requires a genuinely independent evaluation set.

---

# 43. Required manifest-level provenance — LOCKED

Every scientific output must be traceable to:

```text
schema_version
protocol_version

git_commit
run_id
attempt_id

method
architecture
budget
seed_index
training_seed

dataset_manifest_sha256
train_split_sha256
validation_split_sha256

labeled_manifest_sha256
unlabeled_manifest_sha256

resolved_config_sha256
evaluator_config_sha256

checkpoint_role
checkpoint_sha256

software_environment_manifest_sha256
```

For B3 additionally:

```text
model_hash
processor_hash
prompt_hash
renderer_hash
decoding_config_hash
```

---

# 44. Schema-versioning rule — LOCKED

Current:

```text
schema_version = "1.0.0"
```

After first scientific run:

- no field deletion;
- no semantic redefinition under same field name;
- no unit change;
- no sign-convention change;
- no NULL→0 coercion for undefined metric.

Allowed additions require:

```text
controlled amendment
schema version increment
documented migration rule
```

---

# 45. Required NULL / NA semantics — LOCKED

Use:

```text
NULL/NA
```

when metric is undefined/not applicable.

Do not convert to zero.

Examples:

- B0 Q_pseudo → `NA`;
- U metric in H-only row → `NULL`;
- H metric in U-only row → `NULL`;
- no denominator → `NULL` + `INSUFFICIENT`;
- ignored annotation → `Y_i=NULL`.

---

# 46. Publication-ready mandatory outputs — LOCKED

Before manuscript result writing, materialization must support at least:

1. overall detector-performance table;
2. AP50/AP75/AR table;
3. 14-class AP table;
4. operating-point Recall/FP table;
5. No-Finding FP/negative + FAR table;
6. B2 selection/threshold audit table;
7. B3 U gate table by class;
8. B3 H gate table by class;
9. B3 falsification table;
10. B3 matched-burden sham table;
11. intervention opportunity coverage;
12. B3 variant comparison table;
13. B3−B2 paired seed effects if Student training exists;
14. B2−B1, B3−B1, B3−B0 comparison tables if valid measurements exist;
15. compute/runtime/provenance table;
16. FAIL/INSUFFICIENT classes;
17. test-history limitation statement.

Do not report only favorable aggregate metrics.

---

# 47. Minimum file tree — LOCKED

```text
artifacts/
  results/
    method_pair_effects_v1.csv

runs/<RUN_ID>/<ATTEMPT>/
  run_manifest.json
  resolved_config.yaml
  checkpoints/
    checkpoint_index.json

  metrics/
    validation/
      validation_history.csv
      classwise_validation_ap.csv
      training_summary.json
      validation_best_teacher.json

  pseudo/
    qpseudo_validation.json

  method_evidence/
    teacher_candidate_pool.parquet
    b2_reference_selection.parquet

  vlm/
    vlm_request_manifest.jsonl
    vlm_response_cache.jsonl
    b3_variant_selection.parquet
    falsification_controls.csv
    matched_burden_controls.csv
    gt_matching.parquet
    offline_summary_by_class.csv
    offline_audit_report.json
```

Exact directory placement may be adapted during implementation if required by repository architecture, but:

> canonical filenames, schema semantics and provenance requirements remain mandatory unless controlled amendment is approved.

---

# 48. B0/B1/B2/B3 metric matrix — LOCKED

| Metric / artifact | B0 | B1 | B2 | B3 |
|---|---:|---:|---:|---:|
| bbox mAP@[.50:.95] | REQUIRED | REQUIRED | REQUIRED | REQUIRED |
| AP50 | REQUIRED | REQUIRED | REQUIRED | REQUIRED |
| AP75 | REQUIRED | REQUIRED | REQUIRED | REQUIRED |
| AR@[.50:.95], max100 | REQUIRED | REQUIRED | REQUIRED | REQUIRED |
| Recall at score=.50 | REQUIRED | REQUIRED | REQUIRED | REQUIRED |
| FP total | REQUIRED | REQUIRED | REQUIRED | REQUIRED |
| FP/image | REQUIRED | REQUIRED | REQUIRED | REQUIRED |
| 14 class AP@[.50:.95] | REQUIRED | REQUIRED | REQUIRED | REQUIRED |
| FP total negative | REQUIRED | REQUIRED | REQUIRED | REQUIRED |
| FP/negative | REQUIRED | REQUIRED | REQUIRED | REQUIRED |
| negative FAR | REQUIRED | REQUIRED | REQUIRED | REQUIRED |
| validation history | REQUIRED | REQUIRED | REQUIRED | REQUIRED |
| classwise validation AP | REQUIRED | REQUIRED | REQUIRED | REQUIRED |
| training summary | REQUIRED | REQUIRED | REQUIRED | REQUIRED |
| BEST checkpoint provenance | REQUIRED | REQUIRED | REQUIRED | REQUIRED |
| Q_pseudo | N/A | REQUIRED | REQUIRED | REQUIRED compatibility metric |
| adaptive threshold / FIFO metrics | N/A | N/A | REQUIRED | inherited B2 reference |
| H/U/K/q selection trace | N/A | N/A | REQUIRED | REQUIRED |
| PPV_U | N/A | N/A | N/A | REQUIRED |
| U utility | N/A | N/A | N/A | REQUIRED |
| U coverage / abstention | N/A | N/A | N/A | REQUIRED |
| CR_H | N/A | N/A | N/A | REQUIRED |
| FSR_H | N/A | N/A | N/A | REQUIRED |
| H coverage / abstention | N/A | N/A | N/A | REQUIRED |
| null-image persistence | N/A | N/A | N/A | REQUIRED |
| region-shift persistence | N/A | N/A | N/A | REQUIRED |
| unsupported-class canary | N/A | N/A | N/A | REQUIRED |
| evidence-sham comparison | N/A | N/A | N/A | REQUIRED |
| matched-burden control | N/A | N/A | N/A | REQUIRED |
| intervention opportunity coverage | N/A | N/A | N/A | REQUIRED |
| B3-U / B3-H / B3-UH / ALL-OFF trace | N/A | N/A | N/A | REQUIRED |
| VLM provenance hashes | N/A | N/A | N/A | REQUIRED |

---

# 49. Final contract verdict

```text
METRIC_OUTPUT_CONTRACT = LOCKED
SCHEMA_VERSION = 1.0.0

B0/B1_COMPATIBILITY = PASS
B2_REQUIRED_OUTPUTS = LOCKED
B3_REQUIRED_OUTPUTS = LOCKED
GENERIC_METHOD_COMPARISON_SCHEMA = LOCKED

B2_TRAINING_AUTHORIZED = FALSE
B3_TRAINING_AUTHORIZED = FALSE
HISTORICAL_TEST_FOR_DESIGN = PROHIBITED
```

Any implementation that cannot produce the mandatory fields/artifacts above:

> **FAIL / HOLD before scientific training or evaluation.**

No missing required metric may be discovered only after training and then silently reconstructed by changing definitions.
