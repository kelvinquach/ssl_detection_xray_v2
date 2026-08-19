# PHASE HANDOFF — `ssl_detection_xray_v2`

Ngày cập nhật: 2026-08-19

Dự án: **Nghiên cứu học bán giám sát cho dò tìm bất thường trên X-quang phổi**

Bài toán: **Semi-supervised object detection trên VinBigData Chest X-ray**

---

## Cập nhật closure hiện hành — Phase 3A: Dataset Diagnostics Before Training

Status: **CLOSED / PASS**

Date closed: 2026-08-19

### Mục tiêu và phạm vi

Phase 3A thực hiện **descriptive pre-training dataset diagnostics** trên fixed
training set đã khóa và các legitimately labeled subsets của Phase 2F. Mục tiêu
là mô tả cấu trúc dữ liệu trước training, xác định các rủi ro dữ liệu có thể
quan sát được và tạo evidence machine-readable để đối chiếu downstream.

Phase 3A không:

```text
Khởi tạo hoặc huấn luyện mô hình
Chạy supervised hoặc SSL training
Sinh pseudo-label
Chọn checkpoint
Tính AP/mAP
Tune confidence threshold
Tune augmentation, loss, backbone hoặc model architecture
Thực hiện hyperparameter search
Thực hiện Ablation Study
Dùng validation/test để chọn diagnostic threshold hoặc thay đổi methodology
Dùng hidden ground truth của unlabeled subsets
Thay đổi fixed train/validation/test membership
Thay đổi labeled/unlabeled membership
Thay đổi partition_seed hoặc training-seed protocol
```

Sensitivity analysis trong Phase 3A chỉ là:

```text
SECONDARY / NON-GATING robustness analysis
```

cho các **diagnostic operational thresholds**; sensitivity này không phải
Ablation Study và không được dùng để chọn lại primary thresholds.

### Protocol identity và scientific locks

```text
phase: 3A
protocol_version: 1.0.0
protocol_status: RESEARCHER_APPROVED_LOCKED
protocol_sha256:
  b03474cce0be773796f9d458e6273b8fd2b955c1b961bcccc4d955eb3bb0dcf5

primary_diagnostic_scope: FIXED_TRAIN
training_seed_used: false
training_started: false
pseudo_label_created: false
checkpoint_created: false
```

Rare-class primary operational definition:

```text
primary measure:
  image-level class support / prevalence

P_c_img = N_c_img / 3426

rare_primary_threshold = 0.05

N_c_img <= 171  => Rare
N_c_img >= 172  => Non-rare

definition_type:
  PRE_SPECIFIED_OPERATIONAL_DEFINITION
```

`Rare` trong Phase 3A chỉ có nghĩa là low-support/rare trong fixed training
dataset theo operational definition đã khóa; không được diễn giải thành
clinically rare disease.

Bounding-box scale primary definition:

```text
normalized_width  = w / W
normalized_height = h / H
normalized_area   = (w*h) / (W*H)
aspect_ratio      = w / h
```

Normalized-area-based bbox size categories:

```text
Small:  normalized_area < 0.01
Medium: 0.01 <= normalized_area < 0.10
Large:  normalized_area >= 0.10

definition_type:
  PRE_SPECIFIED_OPERATIONAL_DEFINITION
```

Các nhóm trên không phải standard COCO pixel-area Small/Medium/Large.

### Threshold sensitivity đã khóa

Rare-class sensitivity:

```text
thresholds = [0.01, 0.05, 0.10]
primary = 0.05
```

Small-boundary sensitivity:

```text
small_thresholds = [0.005, 0.01, 0.02]
large_threshold_fixed = 0.10
primary_small_threshold = 0.01
```

Large-boundary sensitivity:

```text
large_thresholds = [0.05, 0.10, 0.20]
small_threshold_fixed = 0.01
primary_large_threshold = 0.10
```

BBox sensitivity policy:

```text
one_factor_at_a_time = true
cartesian_grid_allowed = false
sensitivity_role = SECONDARY_NON_GATING
primary_threshold_changed_after_diagnostics = false
sensitivity_used_for_threshold_selection = false
```

### Data-use firewall và leakage policy

Detailed diagnostics thực tế chỉ được mở trên:

```text
train
labeled_1pct
labeled_5pct
labeled_10pct
labeled_20pct
```

Structural-only scopes:

```text
canonical
validation
test
```

Các guardrail quan trọng:

```text
Validation design usage: false
Test design usage: false
Detailed test content diagnostics: false
Hidden unlabeled GT diagnostics: false
Reverse lookup từ U_b vào train GT cho detailed diagnostics: prohibited
Test reserved for final evaluation
```

Canonical master chỉ được dùng cho lineage/integrity reference. Validation và
test chỉ được đọc ở mức structural integrity; không tính class distribution,
bbox-size distribution, label cardinality, class co-occurrence hoặc location
heatmap để phục vụ methodology/tuning.

### Guardrail tests và full execution

Guardrail test cuối:

```text
pytest:
  53 passed
  failures: 0
  errors: 0
```

Full Phase 3A execution:

```text
hard_error_count: 0
warning_count: 0
dod_candidate: true

HF01-HF35: PASS
Input identity/checksum: PASS
Canonical counts/category mapping: PASS
No Finding zero-GT policy: PASS
Train bbox validity: PASS
Phase 2E split identity: PASS
Phase 2F labeled membership/nesting: PASS
Validation/test firewall: PASS
Hidden-U GT firewall: PASS
Locked input immutability: PASS
Training artifact check: PASS
Checkpoint check: PASS
Pseudo-label check: PASS
Mandatory output contract: PASS
Machine-readable structural audit: PASS
```

Script-generated evidence tiếp tục giữ:

```text
phase_status = OPEN_REVIEW_REQUIRED
```

đúng theo policy: script không được tự ghi `PASS`, `CLOSED` hoặc
`CLOSED_PASS`. Sau full execution, researcher + GPT đã review evidence và đưa
ra closure decision bên ngoài generated artifacts:

```text
PHASE_3A_STATUS: CLOSED
PHASE_3A_GATE: PASS
```

### Fixed training-set diagnostic findings

Fixed train được giữ nguyên từ Phase 2E:

```text
images: 3426
annotations: 25260
No Finding / zero-GT negatives: 350
positive images: 3076
detection categories: 14
```

Class support theo image-level prevalence cho thấy:

```text
max_image_support: 2144
min_image_support: 66
median_image_support: 648
imbalance_ratio_max_over_min: 32.484848484848484
coefficient_of_variation_image_support: 0.8025330318219815
```

Theo primary rare threshold `< 5%`, có đúng 2/14 rare classes:

```text
Atelectasis:
  image_count = 130
  image_prevalence = 0.03794512551079977
  bbox_annotation_count = 194

Pneumothorax:
  image_count = 66
  image_prevalence = 0.01926444833625219
  bbox_annotation_count = 142
```

Không được dùng bbox count để định nghĩa rarity.

### Bounding-box geometry và size findings

Trên 25,260 train bbox annotations:

```text
normalized_area:
  mean   = 0.030439067292574274
  median = 0.014892746614044168
  p95    = 0.10033503078233698
  max    = 0.9384266703859281

aspect_ratio:
  mean   = 1.4098767703617139
  median = 0.9853479853479854
  p95    = 3.542250573290138
  max    = 45.0
```

Primary normalized-area categories:

```text
Small:
  9372 / 25260 = 37.1021377672209%

Medium:
  14612 / 25260 = 57.84639746634996%

Large:
  1276 / 25260 = 5.051464766429137%
```

BBox count per image:

```text
all train images:
  mean   = 7.373029772329247
  median = 6
  p95    = 17
  max    = 48

positive train images only:
  mean   = 8.211963589076722
  median = 7
  p95    = 18
  max    = 48
```

Các số trên là bounding-box annotation records; không được diễn giải mặc định
thành số unique clinical lesions.

### BBox location diagnostics

Bounding-box centers được chuẩn hóa:

```text
center_x_n = (x + w/2) / W
center_y_n = (y + h/2) / H
```

Heatmap policy:

```text
grid: 50 x 50
origin: top-left
x: left -> right
y: top -> bottom
weighting: annotation-weighted
```

Heatmap mô tả spatial distribution của bbox annotations, không tự động chứng
minh spatial distribution của unique clinical lesions hoặc quan hệ giải phẫu
nhân quả.

### Negative / No Finding findings

Fixed train:

```text
Positive / abnormal:
  3076 / 3426 = 89.784%

No Finding / negative:
  350 / 3426 = 10.215995329830706%
```

Validation/test chỉ dùng structural counts:

```text
val:
  images = 734
  No Finding = 75

test:
  images = 734
  No Finding = 75
```

No Finding vẫn là zero-GT negative image, không phải detection class thứ 15.

### Labeled-budget diagnostics

Tất cả bốn labeled budgets giữ:

```text
class_coverage_out_of_14 = 14
```

Representativeness deviation so với fixed train:

```text
1pct:
  labeled_images = 34
  negative_images = 3
  max_absolute_deviation = 0.012980323477902539
  mean_absolute_deviation = 0.006991763430415951

5pct:
  labeled_images = 171
  negative_images = 17
  max_absolute_deviation = 0.0028574062125541547
  mean_absolute_deviation = 0.0015424580131004736

10pct:
  labeled_images = 343
  negative_images = 35
  max_absolute_deviation = 0.0012781695114874037
  mean_absolute_deviation = 0.0007462472461732155

20pct:
  labeled_images = 685
  negative_images = 70
  max_absolute_deviation = 0.0006626015740515689
  mean_absolute_deviation = 0.00033811855241795496
```

Rare status luôn kế thừa từ fixed train và không được định nghĩa lại riêng trong
từng budget. Hidden GT của unlabeled complement không được dùng để bổ sung
diagnostics.

### Multi-label diagnostics

Label cardinality trên fixed train:

```text
mean: 3.1310566258026853
sample_sd: 1.9319938322591697
median: 3
p95: 7
max: 10
```

Class co-occurrence:

```text
91 unordered class pairs
role: SECONDARY / NON-GATING
unit: unique image-level class presence
```

Co-occurrence không được dùng để tự động thay đổi training design.

### Sensitivity findings và giới hạn diễn giải

Rare threshold sensitivity:

```text
1%  -> 0 rare classes
5%  -> 2 rare classes  [PRIMARY]
10% -> 5 rare classes
```

Các class có trạng thái rare thay đổi trong sensitivity range:

```text
Atelectasis
Calcification
Consolidation
ILD
Pneumothorax
```

Small-boundary sensitivity, large threshold giữ 10%:

```text
small < 0.5%  -> 23.519398258115597%
small < 1.0%  -> 37.1021377672209%   [PRIMARY]
small < 2.0%  -> 58.9034045922407%
```

Large-boundary sensitivity, small threshold giữ 1%:

```text
large >= 5%   -> 21.78147268408551%
large >= 10%  -> 5.051464766429137%  [PRIMARY]
large >= 20%  -> 0.9540775930324624%
```

Sensitivity cho thấy các tỷ lệ operational categories có thể thay đổi theo
boundary; đây là robustness finding, **không phải lý do đổi primary threshold**.

### Claim boundaries và pre-training risk interpretation

Phase 3A được phép kết luận về:

```text
dataset-level rare status theo locked definition
class imbalance magnitude
normalized-area bbox size percentages
annotation-weighted spatial distribution
negative prevalence
labeled-budget coverage/representativeness
label cardinality
class co-occurrence
threshold-sensitivity robustness
```

Phase 3A không được dùng để kết luận:

```text
Rare class chắc chắn có AP thấp
Small bbox chắc chắn gây detector failure
SSL chắc chắn sửa được imbalance
EMA chắc chắn cải thiện kết quả
Một augmentation/confidence threshold/backbone nào là tối ưu
Pseudo-label quality
SSL superiority
Training convergence hoặc stability
```

Báo cáo cuối chỉ mô tả các statistics như potential pre-training dataset risks;
không gắn thêm các nhãn post-hoc không được operationally định nghĩa.

### Official source và artifacts

Source/config/tests:

```text
configs/protocol/phase3A_dataset_diagnostics.yaml
scripts/03A_dataset_diagnostics.py
tests/test_phase3A_dataset_diagnostics_guardrails.py
```

Mandatory report/plots:

```text
reports/dataset_analysis_report.md
plots/dataset/class_distribution.png
plots/dataset/bbox_distribution.png
plots/dataset/bbox_location_heatmap.png
plots/dataset/negative_image_distribution.png
```

Machine-readable evidence:

```text
reports/03A_dataset_diagnostics_validation.json
reports/03A_artifact_manifest.json
reports/03A_guardrails_junit.xml

reports/03A_class_distribution.csv
reports/03A_class_imbalance.csv
reports/03A_bbox_distribution.csv
reports/03A_bbox_summary.csv
reports/03A_bbox_count_per_image.csv
reports/03A_negative_distribution.csv
reports/03A_split_distribution.csv
reports/03A_labeled_budget_coverage.csv
reports/03A_label_cardinality.csv
reports/03A_class_cooccurrence.csv
reports/03A_threshold_sensitivity.csv
```

Final report SHA-256:

```text
reports/dataset_analysis_report.md:
  27e6ddfbd92ac2ba2a028e923f3be33f531a3a2a38f40468ccc5f38ef3823a2f
```

Final validation JSON SHA-256:

```text
reports/03A_dataset_diagnostics_validation.json:
  c24eff6907d92a23bbe9df39dfeeca9925b940fabbc087a5abd2e74c8732b892
```

Artifact manifest records:

```text
artifact_count: 17
self_hash_excluded: true
validation_json_hashed_after_final_audit: true
write_passes: 2
final mandatory-output audit: 18/18 artifacts present and structurally valid
```

### Closure decision và downstream authorization

Final researcher + GPT review:

```text
PHASE_3A_STATUS: CLOSED
PHASE_3A_GATE: PASS
PHASE3A_GUARDRAIL_TESTS: 53/53 PASS
PHASE3A_FULL_EXECUTION: PASS
HARD_ERRORS: 0
WARNINGS: 0
DoD_CANDIDATE: TRUE

DATASET_DIAGNOSTICS_COMPLETED: true
TRAINING_STARTED: false
TRAINING_AUTHORIZED: false
PSEUDO_LABEL_GENERATED: false
ABLATION_STUDY_PERFORMED_IN_PHASE_3A: false
```

Đóng Phase 3A không tự động cấp quyền training. Training chỉ được mở khi phase
downstream về training/Colab environment, model configuration, runtime seed
integration, checkpoint/evaluation policy và experiment-specific guardrails
được xác định, review và cho phép.

### Next checkpoint

```text
Next checkpoint: Phase 4 — Supervised Baseline
Phase 4 status: NEXT / NOT STARTED
Required action:
  define, review and lock the Phase 4 supervised-baseline protocol and
  training authorization gates before any official training run
```

---

## Handoff hiện hành — Phase 4: Supervised Baseline

Roadmap sau Phase 3A đã được xác định:

```text
Previous completed phase:
Phase 3A — Dataset Diagnostics Before Training: CLOSED / PASS

Next implementation phase:
Phase 4 — Supervised Baseline

Phase 4 status:
NEXT / NOT STARTED
```

Phase 4 có nhiệm vụ xây dựng **supervised baseline có kiểm soát** làm mốc so
sánh cho Phase 5 — SSL Detection.

Việc Phase 4 là phase kế tiếp **không đồng nghĩa training đã được authorize**.

Current authorization state:

```text
dataset_training_ready: true
phase3A_closed_pass: true
phase4_identified_as_next: true
phase4_started: false
training_started: false
training_authorized: false
```

Trước official supervised training, Phase 4 phải định nghĩa, review và khóa tối
thiểu:

```text
Google Colab / GPU environment
MMDetection supervised training configuration
runtime integration của ordered 10 training seeds
labeled-budget usage
augmentation policy
optimizer / learning-rate scheduler / epoch policy
checkpoint-selection policy
validation-only model selection
logging / artifact / retry policy
per-seed result schema
test-set leakage guardrails
training authorization gate
```

Test set tiếp tục chỉ dùng cho final evaluation; không dùng để chọn checkpoint,
model, hyperparameter, augmentation hoặc threshold.

---


## Cập nhật closure hiện hành — Phase 2F.1: Seed Protocol

Status: **CLOSED / PASS**

Date closed: 2026-08-18

### Mục tiêu và phạm vi

Phase 2F.1 xác định, kiểm định và khóa seed contract để các thí nghiệm
supervised và semi-supervised ở Phase 4–5 kế thừa. Đây không phải là một lần
chia dữ liệu mới và không thực hiện training.

Phase này không:

```text
Tạo lại train/validation/test split
Tạo lại labeled/unlabeled membership
Thay đổi partition_seed hoặc membership checksum
Khởi tạo hoặc huấn luyện mô hình
Chạy DataLoader hoặc augmentation runtime
Sinh pseudo-label
Tạo checkpoint hoặc kết quả mô hình
Chọn hyperparameter, threshold hoặc checkpoint
Sử dụng validation/test để chọn seed
Kiểm chứng GPU bitwise determinism
Capture hoặc restore RNG state
```

### Phân biệt partition seed và training seed

```text
partition_seed = 42
partition_seed_policy = PRE_SPECIFIED_LOCKED_NO_SEED_SEARCH
training_seed_count = 10
deterministic_policy = CONTROLLED_BEST_EFFORT
```

`partition_seed=42` đã khóa train/validation/test và labeled/unlabeled
membership từ Phase 2E/2F. Phase 2F.1 không thay đổi các membership này.

Thuật ngữ chính thức downstream là `partition_seed`. `split_seed` chỉ được
hiểu là legacy alias của `partition_seed`; không được gộp `partition_seed` và
`training_seed` thành một khái niệm.

`training_seed` kiểm soát các nguồn ngẫu nhiên của quá trình huấn luyện. Việc
seed Python, NumPy, PyTorch CPU/CUDA, DataLoader workers, sampler và
augmentation sẽ được tích hợp và kiểm chứng tại Phase 4–5, không phải tại
Phase 2F.1.

### Ordered training-seed list đã khóa

| Index | training_seed |
|---:|---:|
| 1 | 204886845 |
| 2 | 1480646854 |
| 3 | 1798418854 |
| 4 | 2045683682 |
| 5 | 1814859839 |
| 6 | 1603952859 |
| 7 | 1878351743 |
| 8 | 875651179 |
| 9 | 477581743 |
| 10 | 869675675 |

Danh sách được sinh trước training bằng quy tắc công khai, không gọi RNG:

```text
namespace = ssl_detection_xray_v2|phase2F.1|training_seed
payload_i = namespace + |index=i
digest_i = SHA256(UTF-8(payload_i))
seed_i = 1 + (integer(first 8 hexadecimal characters of digest_i)
              mod (2^31 - 1))
i = 1,...,10
```

Phép dẫn xuất SHA-256 đã được builder, pytest guardrails và GPT review tính lại
độc lập; kết quả `MATCH` cho 10/10 seed.

### Quy tắc áp dụng downstream

Mỗi tổ hợp thí nghiệm chính thức:

```text
budget × method × configuration
```

phải sử dụng đủ cùng ordered list gồm 10 training seed. Phase 2F.1 không khóa
số lượng method hoặc configuration của Phase 4–5; một cấu hình chỉ bị ràng
buộc bởi protocol này sau khi được xác định là cấu hình thí nghiệm chính thức.

So sánh supervised và SSL phải paired-by-seed:

```text
supervised(training_seed_index=k) ↔ SSL(training_seed_index=k)
```

Điều này không có nghĩa dùng một seed duy nhất cho mọi run. Mỗi cấu hình có 10
lần chạy độc lập tương ứng 10 seed khác nhau; supervised và SSL dùng cùng seed
tại cùng index để hỗ trợ so sánh ghép cặp.

### Retry và reporting policy

```text
retry_allowed_reason: TECHNICAL_FAILURE_ONLY
retry_changes_training_seed: false
silent_replacement_allowed: false
failed_run_must_be_retained: true
result_based_retry_allowed: false
```

Run chỉ được retry khi có lỗi kỹ thuật được ghi nhận. Retry phải giữ nguyên
`training_seed` và `training_seed_index`, liên kết run cũ qua `retry_of`, lưu
`technical_failure_reason` và giữ lại lịch sử run lỗi. Không được retry, đổi
seed hoặc loại run chỉ vì kết quả kém.

Khi có kết quả ở Phase 4–5, phải báo cáo:

```text
Kết quả từng seed
Mean across seeds
Sample standard deviation với ddof=1
Số run completed, technical failure và retry
```

Cấm chỉ báo cáo best seed, seed dropping hoặc outlier removal dựa trên kết quả.

### Metadata bắt buộc cho future run

```text
run_id
run_status
method
configuration_id
budget
partition_seed
training_seed
training_seed_index
rng_state_id
membership_checksum
config_hash
code_revision
environment
deterministic_runtime_settings
checkpoint
results
retry_of
technical_failure_reason
```

Đây là schema cho future run. Phase 2F.1 không tạo run, RNG state, checkpoint
hoặc result thực tế.

### Rationale và giới hạn diễn giải

Mười training seed được đăng ký trước khi training nhằm định lượng biến thiên
giữa các lần chạy, giảm phụ thuộc của kết luận vào một run và hỗ trợ so sánh
supervised–SSL theo cùng seed. Không tuyên bố 10 là số seed tối ưu thống kê,
không thực hiện power analysis và không tuyên bố variance/stability đã được đo
trong Phase 2F.1.

`CONTROLLED_BEST_EFFORT` yêu cầu Phase 4–5 seed mọi nguồn ngẫu nhiên có thể
kiểm soát và ghi runtime settings thực tế. Protocol không tuyên bố kết quả số
học giống hệt giữa các GPU, CUDA, cuDNN, driver, phần cứng hoặc phiên bản phần
mềm khác nhau. GPU/runtime determinism chỉ được kiểm chứng khi training thực
sự được triển khai.

### Bất biến kế thừa từ Phase 2F

```text
protocol_identity: 2F-C0-R11
protocol_version: 2.0.0
train_size: 3426
labeled_budgets: [1pct, 5pct, 10pct, 20pct]
labeled_sizes: [34, 171, 343, 685]
unlabeled_sizes: [3392, 3255, 3083, 2741]
```

Labeled membership SHA-256 giữ nguyên:

```text
1pct:  c54e7d61e84b7cfce68c04a795783b5c5d01331d2aa9c754fb1ae1dbae4ba071
5pct:  c4db3b5f7a5b0f391ad883ef665c3d341af3ef6afb3fc553e71f1c4ef17ee50b
10pct: fc008d31505227544087ba474613075a8cd077587df9d6b13146b378f5a3a7d6
20pct: 6f4aaba6be147983d56007c49ada234dfcabe7ec2f936f28f8cd240999b8417e
```

### Source và artifact chính thức

```text
configs/protocol/phase2F1_seed_protocol.yaml
scripts/02F1_build_seed_protocol.py
tests/test_phase2F1_seed_protocol_guardrails.py

data/manifests/seed_manifest.json
data/manifests/seed_state_manifest.json
reports/seed_protocol.md
reports/02F1_seed_protocol_validation_report.json
reports/02F1_guardrails_junit.xml
```

Source protocol SHA-256:

```text
ba5b1a1adce67c3f1cf9dd46657e3db89c9d29b85cc37a744462c55a617d3234
```

### Machine-readable validation evidence

```text
Builder execute checks: 22/22 PASS
Builder execute exit code: 0
Pytest guardrails: 20/20 PASS
JUnit failures: 0
JUnit errors: 0
JUnit skipped: 0
Pytest exit code: 0
Validate-existing checks: 23/23 PASS
Validate-existing exit code: 0
Validate-existing write behavior: READ-ONLY
Training-seed derivation: MATCH 10/10
Phase 2F inherited membership checksums: MATCH 4/4
```

GPT review đã đọc toàn bộ YAML, builder, independent guardrail tests, JSON
manifests, Markdown report, validation JSON, JUnit và console evidence. Builder
chỉ ghi đúng bốn artifact Phase 2F.1 ở `--execute`; `--validate-existing` không
ghi file. Không phát hiện truy cập dataset/test set, training, pseudo-label,
checkpoint hoặc thay đổi Phase 2E/2F artifact.

### Local environment provenance snapshot

Môi trường local chỉ là provenance cho các bước dữ liệu/protocol; không phải
training environment cho Google Colab.

```text
Platform: Windows-10-10.0.26200-SP0 / AMD64
Python: 3.10.20
PyYAML: 6.0.3
pytest: 9.1.0
NumPy runtime: 2.2.6
SciPy runtime: 1.15.2
scikit-learn: 1.7.0
iterative-stratification: 0.1.9
environment_role: LOCAL_DATA_AND_PROTOCOL_PROVENANCE
```

Snapshot artifacts:

```text
reports/02F1_local_conda_environment.yml
reports/02F1_local_conda_explicit.txt
reports/02F1_local_pip_freeze.txt
reports/02F1_local_runtime_environment.json
reports/02F1_local_environment_checksums.json
```

Snapshot SHA-256:

```text
02F1_local_conda_environment.yml:
  6ec68eb4bc0951f116559297085be35c5491ed712607e55f9dee7cd9b772043e
02F1_local_conda_explicit.txt:
  85ed214594e1e14aa328131e5554360f0686d85b2ce3a04f6dbe15b0e4d19d68
02F1_local_pip_freeze.txt:
  8afbc2590fa632409b0eab9725065a4441384c9ad9ae8141d336f4f918be3e62
02F1_local_runtime_environment.json:
  916c6aa998ad9f0192167c2f3a01419260c7d6902064f8e86e41b8e314b0418d
```

Environment có mixed Conda–pip provenance: Conda explicit giữ record NumPy
`1.24.3`, trong khi imported runtime và pip snapshot ghi NumPy `2.2.6`;
Conda YAML pip section ghi SciPy `1.15.3`, trong khi imported runtime,
Conda explicit và pip freeze phản ánh SciPy `1.15.2`. Không sửa ngược các
snapshot để che khác biệt. Phiên bản imported runtime được ưu tiên khi mô tả
execution thực tế. Khác biệt này không ảnh hưởng Seed Protocol vì Phase 2F.1
không dùng NumPy/SciPy để sinh seed hoặc thay đổi membership.

Phase 4–5 phải tạo và khóa một Google Colab training environment riêng sau
compatibility/smoke test và trước official training. Không dùng snapshot local
như Colab lock. Khi chuyển dữ liệu lên Colab, phải kiểm tra lại checksum và
không xây dựng lại split hoặc labeled/unlabeled membership.

### Closure decision và authorization

```text
PHASE_2F1_STATUS: CLOSED
PHASE_2F1_GATE: PASS
SEED_PROTOCOL: LOCKED
DATASET_MEMBERSHIP: UNCHANGED
TRAINING_STARTED: false
TRAINING_AUTHORIZED: false
```

Các generated artifacts giữ `PENDING_RESEARCHER_GPT_REVIEW` vì phản ánh trạng
thái tại thời điểm sinh trước review. Subsequent source/evidence review ngày
2026-08-18 là closure decision; không sửa ngược generated evidence lịch sử.

Đóng Phase 2F.1 không tự động cấp quyền training. Training chỉ được xem xét sau
khi training/Colab environment, model configuration, runtime seed integration,
checkpoint/evaluation policy và các guardrails tương ứng được định nghĩa,
kiểm định và cho phép trong phase downstream.

---

## Cập nhật closure hiện hành — Phase 2F: Labeled/Unlabeled Construction

Status: **CLOSED / PASS**

Date closed: 2026-08-13

### Protocol identity và phạm vi

```text
stage: 2F-C0-R11
protocol_version: 2.0.0
partition_seed: 42
seed_policy: PRE_SPECIFIED_LOCKED_NO_SEED_SEARCH
active_repair_policy: ONE_FOR_ONE_EXHAUSTIVE_DETERMINISTIC
local_optimum_neighborhood: one_for_one
global_optimum_claimed: false
```

Phase 2F chỉ sử dụng 3.426 ảnh thuộc fixed training split của Phase 2E để tạo
các cặp labeled/unlabeled. Validation và test không tham gia lựa chọn membership
và không được dùng làm nguồn pseudo-label.

Active construction thực hiện theo thứ tự:

```text
iterative multilabel stratification
→ exact-size repair
→ exact-No-Finding repair
→ minimum-class-coverage one-for-one repair
→ deterministic exhaustive one-for-one objective repair
→ one-for-one local optimum
```

Legacy two-for-two engine không được gọi trong active construction. Kết quả
objective repair chỉ chứng minh local optimum đối với admissible one-for-one
swap neighborhood; không chứng minh global optimum và không chứng minh tối ưu
đối với mọi neighborhood có thể có.

### Labeled/unlabeled membership đã khóa

| Budget | Labeled | Unlabeled | No Finding trong labeled | Repair moves |
|---|---:|---:|---:|---:|
| 1% | 34 | 3.392 | 3 | 3 |
| 5% | 171 | 3.255 | 17 | 7 |
| 10% | 343 | 3.083 | 35 | 10 |
| 20% | 685 | 2.741 | 70 | 26 |

Tổng số repair moves: `46`.

Mọi budget đạt:

```text
Exact labeled-size target: PASS
Exact No Finding target: PASS
Class coverage 14/14: PASS
One-for-one neighborhood exhausted: TRUE
One-for-one local optimum: TRUE
Global optimum claimed: FALSE
```

Quan hệ nested đã được kiểm định:

```text
1pct ⊆ 5pct ⊆ 10pct ⊆ 20pct
Nested labeled subsets: PASS
Nested No Finding subsets: PASS
```

Unlabeled set của mỗi budget là phần bù chính xác của labeled set trong fixed
train universe.

### Membership checksum

SHA-256 của canonical labeled `image_id` membership:

```text
1pct:  c54e7d61e84b7cfce68c04a795783b5c5d01331d2aa9c754fb1ae1dbae4ba071
5pct:  c4db3b5f7a5b0f391ad883ef665c3d341af3ef6afb3fc553e71f1c4ef17ee50b
10pct: fc008d31505227544087ba474613075a8cd077587df9d6b13146b378f5a3a7d6
20pct: 6f4aaba6be147983d56007c49ada234dfcabe7ec2f936f28f8cd240999b8417e
```

### Validation, leakage và readback

```text
Guardrails: 183 passed, 15 subtests passed
PREFLIGHT_GATE: PASS
PHASE_2F_GATE: PASS
Nested split check: PASS
Nested No Finding check: PASS
Labeled/unlabeled disjoint and complete: PASS
Validation/test isolation: PASS
Unlabeled GT exposure violations: []
Independent readback gates: PASS
```

Unlabeled COCO JSON không chứa annotations hoặc các trường dẫn xuất từ ground
truth thuộc forbidden-field policy. Kết luận leakage của Phase 2F giới hạn ở
fixed train/validation/test membership và các trường có thể kiểm tra từ artifact;
nó không mở rộng thành patient-level leakage claim.

### Deterministic reconstruction

Lệnh kiểm tra không promote official artifacts:

```text
python scripts\02F_build_labeled_unlabeled.py --reconstruct-check
```

Kết quả:

```text
1pct: MATCH
5pct: MATCH
10pct: MATCH
20pct: MATCH
RECONSTRUCT_CHECK_STATUS: MATCH
```

Reconstruct-check so sánh bảy nhóm trường ở từng budget:

```text
labeled_image_id_sha256
labeled_coco_json_sha256
unlabeled_coco_json_sha256
labeled_size
no_finding_size
repair_move_counts
integer_objective_final
```

### Runtime observability

```text
1pct: 7.063 seconds
5pct: 130.437 seconds
10pct: 253.485 seconds
20pct: 1255.344 seconds
total construction: 1646.329 seconds
timing_clock: time.monotonic
timing_role: OBSERVABILITY_ONLY_NOT_SELECTION_CRITERION
```

Timing không tham gia membership, seed, objective, acceptance criterion,
tie-break, checksum hoặc deterministic reconstruction comparison.

### Official artifacts

```text
scripts/02F_build_labeled_unlabeled.py
configs/protocol/phase2F_labeled_unlabeled.yaml
tests/test_phase2F_labeled_unlabeled_guardrails.py

data/processed/coco/labeled_splits/instances_labeled_1pct.json
data/processed/coco/labeled_splits/instances_labeled_5pct.json
data/processed/coco/labeled_splits/instances_labeled_10pct.json
data/processed/coco/labeled_splits/instances_labeled_20pct.json
data/processed/coco/unlabeled_splits/instances_unlabeled_1pct.json
data/processed/coco/unlabeled_splits/instances_unlabeled_5pct.json
data/processed/coco/unlabeled_splits/instances_unlabeled_10pct.json
data/processed/coco/unlabeled_splits/instances_unlabeled_20pct.json

data/manifests/audit/phase2F_unlabeled_gt_audit.csv
data/manifests/phase2F_partition_manifest.csv
data/manifests/phase2F_lock_manifest.json
data/manifests/phase2F_nested_split_check.json
data/manifests/phase2F_leakage_check.json
data/manifests/phase2F_seed_manifest.json

reports/02F_labeled_unlabeled_validation_report.json
reports/02F_labeled_unlabeled_log.json
reports/02F_deterministic_reconstruction_check.json
reports/02F_class_distribution.csv
reports/02F_negative_distribution.csv
reports/02F_repair_log.jsonl
reports/02F_errors.csv
```

Official materialization sử dụng staging, independent readback, refuse-overwrite
và transactional rollback. Tổng cộng 20 official artifacts được promote sau
khi validation PASS.

### Authorization và bất biến downstream

```text
training_authorized: false
training_started: false
pseudo_labels_generated: false
test_used: false
```

Phase 2F closure không cấp quyền training. Không được:

* tái chia train/validation/test;
* tái lấy mẫu labeled membership theo experiment;
* tìm kiếm hoặc thay đổi `partition_seed=42` để có kết quả thuận lợi;
* dùng validation/test để tạo labeled/unlabeled membership;
* dùng test để chọn checkpoint, model, threshold hoặc pseudo-label policy;
* diễn giải one-for-one local optimum thành global optimum;
* bắt đầu training khi `training_authorized=false`.

### Next phase

```text
Phase 2F.1 — Seed Protocol: CLOSED / PASS
```

Phase 2F.1 đã phân biệt và khóa `partition_seed` với `training_seed` mà không
thay đổi fixed split hoặc bất kỳ labeled/unlabeled membership nào của Phase 2F.

---

## 1. Vai trò làm việc

* Người nghiên cứu: người quyết định hướng nghiên cứu, protocol, phạm vi thí nghiệm.
* GPT: thiết kế quy trình, phản biện logic, review evidence, quyết định pass/fail DoD.
* Claude: viết code trong repo theo prompt được giao.
* Python: chạy script, kiểm tra dữ liệu, train/evaluate, tạo evidence.

Quy trình bắt buộc:

```text
script → output → DoD → GPT review → người nghiên cứu tick checklist
```

Không tick checklist nếu chưa có evidence.

---

## 2. Nguyên tắc nghiên cứu đã khóa

* Không nhảy phase.
* Không train khi data/split/COCO/No Finding/seed/checkpoint criterion chưa pass DoD.
* Không dùng test set để tune threshold.
* Không dùng test set để chọn checkpoint.
* Không dùng test set để chọn model/backbone.
* Không dùng test set để quyết định augmentation.
* `No Finding` là ảnh âm tính không có bbox, không phải detection class.
* Metric chính: `mAP@0.5:0.95`.
* Supervised và SSL phải dùng cùng labeled split, cùng `partition_seed`, cùng fixed test set và cùng ordered training-seed list.
* Stability phải dùng nhiều `training_seed`.
* Local environment snapshot chỉ dùng làm provenance; Google Colab training environment phải được khóa riêng trước official training.

---

## 3. Trạng thái hiện tại

```text
Current checkpoint: Phase 3A — Dataset Diagnostics Before Training: CLOSED / PASS
Next checkpoint: NOT OPENED — phải đọc roadmap/checklist hiện hành trước khi triển khai
Phase 0 core: PASS
Phase 0 local training framework: DEFERRED
Phase 1A — Dataset Overview: PASS
Phase 1B — Annotation Quality: PASS
Phase 1C — Dataset Scope Decision: PASS
Phase 1D — Label Reliability & Kappa Feasibility: PASS
Phase 2A — Data Standardization / Image-Boundary Validation: PASS
Phase 2B — Canonical Detection Annotation Schema: PASS
Phase 2C — Framework & Format Decision / COCO Conversion Planning: PASS
Phase 2D.1A — Image Representation Protocol Decision: CLOSED / PASS
Phase 2D.1B-Pilot — Representative DICOM-to-JPG Pilot: CLOSED / PASS
Phase 2D.1B-Full — Full DICOM-to-JPG Conversion: CLOSED / PASS
Phase 2D.1C — MMDetection Dataset Loading & Full Pipeline Audit: CLOSED / PASS
Phase 2D.1D — Evidence Consolidation, GPT Review & Closure: CLOSED / PASS
Phase 2D.1 overall: CLOSED / PASS
Phase 2E — Fixed Train/Validation/Test Split: CLOSED / PASS
Phase 2F — Labeled/Unlabeled Construction: CLOSED / PASS
Phase 2F.1 — Seed Protocol: CLOSED / PASS
Phase 3A — Dataset Diagnostics Before Training: CLOSED / PASS
Dataset training-ready: TRUE
Training authorized: FALSE
Phase 2D.1C implementation/evidence commit: 0bf30cb (pushed to origin/main)
Phase 2D.1C prompt/environment commit: 5ce88f6 (pushed to origin/main)
Locked reproducibility notebook commit: 267d4bc (pushed to origin/main)
```

Chính sách Git cho Phase 2D.1D closure package:

```text
Use one dedicated Phase 2D.1D closure commit.
Do not amend the locked Phase 2D.1C commits.
Read the resulting closure commit hash from Git history; do not write it back
into this same closure package.
```

Trạng thái chuyển tiếp:

```text
Phase 3A — Dataset Diagnostics Before Training: CLOSED / PASS
Next implementation phase: Phase 4 — Supervised Baseline
Phase 4 status: NEXT / NOT STARTED
training_started: false
training_authorized: false
```

Chưa được làm:

```text
Train supervised detector
Train SSL detector
Generate pseudo-label
Tune threshold
Use test set for tuning/model selection before final evaluation
Any training command before training_authorized is explicitly changed to true
```

Ghi chú:

```text
Phase 2A đã xác nhận 4,894 DICOM files tồn tại và đọc được metadata/header.
Image dimensions available cho toàn bộ 4,894 images.
Toàn bộ 36,096 abnormal bbox hợp lệ trong image boundary.
Phase 2B đã tạo portable canonical detection schema.
Canonical image table có 4,894 images.
Canonical bbox table có 36,096 abnormal bbox rows.
Canonical class mapping có đúng 14 abnormal detection classes.
No Finding vẫn là negative image không có bbox, không phải detection class.
No Finding không nằm trong canonical bbox table hoặc detection class mapping.
Path policy đã portable: downstream dùng VINBIGDATA_DICOM_ROOT + relative_dicom_path.
Không được tạo split, COCO master, train, pseudo-label hoặc tune threshold khi chưa mở đúng phase.
Phase 2D.1C đã xác nhận MMDetection có thể nạp và duyệt toàn bộ 4,894 ảnh.
Full pipeline audit đã duyệt 4,394 ảnh abnormal và 500 ảnh zero-GT, không có lỗi.
Regression test cho hành vi MMEngine `serialize_data=True` đã được bổ sung và toàn bộ 35 tests đã PASS.
Phase 2D.1C dataloader validation đã chạy với `num_workers=0`; multi-worker loading không được kiểm định trong phase này.
`dataset_training_ready=True` chỉ là kết luận kỹ thuật về dataset; không đồng nghĩa với quyền bắt đầu training.
`training_authorized=False` vẫn là gate có hiệu lực.
Phase 2E đã tạo và khóa fixed train/validation/test split ở cấp image_id với tỷ lệ mục tiêu 70/15/15.
Fixed split có 3,426/734/734 ảnh và 25,260/5,399/5,437 annotations tương ứng train/val/test.
No Finding/zero-GT được giữ trong cả ba split: 350/75/75 ảnh.
Image-level overlap giữa mọi cặp split bằng 0; union bao phủ đủ 4,894/4,894 ảnh.
Annotation ownership theo split bao phủ đủ 36,096/36,096 annotations.
Test set đã cố định và có 75 ảnh No Finding để hỗ trợ tính FP per negative image.
Patient-level leakage không thể kiểm định độc lập vì PatientID và các định danh nhóm liên quan không còn khả dụng sau de-identification của VinDr-CXR; không được tuyên bố patient-level leakage = 0.
Phase 2F đã xây dựng và khóa bốn labeled/unlabeled budgets lồng nhau chỉ từ fixed train split.
Labeled sizes 1%/5%/10%/20% lần lượt là 34/171/343/685 ảnh; No Finding trong labeled là 3/17/35/70 ảnh.
Mỗi labeled subset bao phủ 14/14 lớp; labeled/unlabeled disjoint và complete trong train universe; validation/test isolation PASS.
Active repair policy là ONE_FOR_ONE_EXHAUSTIVE_DETERMINISTIC và chỉ chứng minh one-for-one local optimum, không tuyên bố global optimum.
Phase 2F official construction, independent readback, nested/leakage checks đều PASS; deterministic reconstruction MATCH cho 4/4 budgets.
Phase 2F không thay đổi training authorization: training_authorized=False.
Phase 2F.1 đã khóa 10 training seed bằng quy tắc SHA-256 công khai; supervised và SSL phải dùng cùng ordered seed list và paired-by-seed.
Phase 2F.1 builder validation 22/22 PASS, pytest 20/20 PASS và validate-existing 23/23 PASS; mọi exit code bằng 0.
Phase 2F.1 không thay đổi dataset membership, không training và không cấp quyền training: training_authorized=False.
Phase 3A đã hoàn tất pre-training dataset diagnostics trên fixed train và bốn legitimately labeled budgets; validation/test chỉ structural-only.
Phase 3A guardrails cuối 53/53 PASS; full execution hard_error_count=0, warning_count=0, mandatory output contract PASS.
Phase 3A xác định 2 rare classes theo locked threshold <5%: Atelectasis và Pneumothorax; imbalance ratio max/min = 32.484848484848484.
BBox normalized-area categories trên train: Small 37.1021%, Medium 57.8464%, Large 5.0515%.
Phase 3A sensitivity analysis là SECONDARY/NON-GATING, không retune primary thresholds và không phải Ablation Study.
Phase 3A không dùng hidden GT của unlabeled subsets, không dùng detailed test diagnostics, không training, không pseudo-label.
Phase 3A closure không thay đổi training authorization: training_authorized=False.
```
---

## 4. Phase 0 — Kết quả bàn giao

### 4.1. Phase 0A — Repo structure

Trạng thái: **PASS**

Đã có các thư mục chính:

* `configs/protocol/`
* `data/raw/`
* `data/interim/`
* `data/processed/`
* `data/manifests/`
* `src/utils/`
* `scripts/`
* `experiments/`
* `reports/`
* `plots/`
* `models/`
* `logs/`
* `tests/`
* `draft/`

Đã có tài liệu:

* `README.md`
* `CLAUDE.md`
* `STRUCTURE.md`
* `RESEARCH_CHECKLIST.md`
* `repository_structure.md`
* `research_log.md`
* `PHASE_HANDOFF.md`

Đã có protocol:

* `configs/protocol/checkpoint_policy.yaml`

---

### 4.2. Phase 0B — Core environment

Trạng thái: **PASS core / DEFER training framework**

Evidence đã có:

* `reports/phase0_environment_check.json`
* `reports/phase0_pip_freeze.txt`
* `reports/reproducibility_settings.md`
* `data/manifests/seed_state_manifest.json`

Kết quả chính:

* Python: `3.10.20`
* Conda env: `sslxray`
* Platform: `Windows-10-10.0.26200-SP0`
* PyTorch: `2.3.1`
* torchvision: `0.18.1`
* numpy: `1.24.3`
* pandas: `2.3.3`
* OpenCV/cv2: `4.11.0`
* pydicom: `3.0.2`
* pycocotools: `2.0.11`
* `pip check`: `No broken requirements found.`
* `pytest tests/test_phase0.py -q`: `5 passed`

CUDA:

* `torch.cuda.is_available()`: `False`
* `torch.version.cuda`: `null`
* GPU device count: `0`

Detection framework:

* `mmengine`: not installed
* `mmcv`: not installed
* `mmdet`: not installed
* `framework_import_ok`: `false`

Các giá trị trên phản ánh evidence tại thời điểm Phase 0. Snapshot provenance
Phase 2F.1 ghi imported runtime hiện hành là NumPy `2.2.6`, SciPy `1.15.2`,
PyYAML `6.0.3` và pytest `9.1.0`. Conda record và imported runtime có mixed
Conda–pip provenance; xem closure Phase 2F.1 ở đầu tài liệu. Không sử dụng
environment local này như environment lock cho Google Colab training.

Quyết định:

```text
Local environment chỉ dùng cho repo/data/script/report validation.
MMDetection/GPU training environment sẽ setup riêng ở remote/GPU.
```

---

## 5. Seed và reproducibility

Seed mặc định Phase 0:

```text
seed = 2026
```

`seed=2026` là giá trị lịch sử của Phase 0 environment/reproducibility check;
không phải ordered training-seed protocol chính thức cho Phase 4–5. Downstream
official runs phải dùng 10 training seed đã khóa tại Phase 2F.1.

Đã ghi nhận trong:

* `data/manifests/seed_state_manifest.json`
* `reports/phase0_environment_check.json`
* `reports/reproducibility_settings.md`

Lưu ý lifecycle: đường dẫn `data/manifests/seed_state_manifest.json` hiện được
Phase 2F.1 quản lý và chứa template `TEMPLATE_LOCKED_NO_RUNS` với ordered list
10 training seed. Không được diễn giải current file này là Phase 0 seed-state
artifact hoặc tự thay lại bằng `seed=2026`.

Deterministic flags:

* `PYTHONHASHSEED = 2026`
* Python random seed: enabled
* NumPy seed: enabled
* PyTorch CPU seed: enabled
* PyTorch CUDA seed: not applied because CUDA unavailable
* `torch.use_deterministic_algorithms = true`
* `torch.backends.cudnn.deterministic = true`
* `torch.backends.cudnn.benchmark = false`
* `CUBLAS_WORKSPACE_CONFIG = :4096:8`

---

## 6. Checkpoint/evaluation protocol đã khóa

File:

```text
configs/protocol/checkpoint_policy.yaml
```

Protocol:

* Primary metric: `mAP@0.5:0.95`
* Checkpoint selection split: `val`
* Test usage: `final_evaluation_only`

Cấm:

* Dùng test set để tune threshold.
* Dùng test set để chọn checkpoint.
* Dùng test set để chọn model/backbone.
* Dùng test set để quyết định augmentation.

---

## 7. Checklist tick được sau Phase 0

Được tick:

* Phase 0A repo structure
* Phase 0B core environment
* pip dependency check
* PyTorch/torchvision import
* numpy/pandas/cv2/pydicom/pycocotools import
* seed manifest
* deterministic flags
* environment report
* reproducibility report
* pip freeze
* checkpoint policy
* Phase 0 pytest pass

Chưa được tick:

* MMDetection import OK
* `mmengine` import OK
* `mmcv` import OK
* `mmdet` import OK
* CUDA/GPU ready
* Local training-ready environment
* Full detection framework setup

---

## 8. Phase 1A — Dataset Overview

Ngày cập nhật: 2026-06-18
Status: PASS

Mục tiêu Phase 1A:

Đọc metadata/annotation CSV của VinBigData để tạo báo cáo tổng quan dataset.

Phase 1A chỉ được phép:

* đọc annotation CSV
* thống kê số dòng annotation
* thống kê số unique image_id
* thống kê class distribution
* kiểm tra sơ bộ bbox validity
* kiểm tra No Finding policy
* tạo evidence report

Phase 1A không được phép:

* tạo split
* convert COCO
* copy ảnh
* đọc DICOM/PNG
* train
* tạo pseudo-label
* tune threshold
* dùng test set

---

## 9. Câu hỏi Phase 1A phải trả lời

* Dataset có bao nhiêu annotation rows?
* Dataset có bao nhiêu unique `image_id`?
* Có bao nhiêu class?
* Có bao nhiêu abnormal class nếu loại `No Finding`?
* `No Finding` xuất hiện dưới dạng nào?
* `No Finding` có bbox không?
* `No Finding` có được xử lý như ảnh âm tính không?
* Có bao nhiêu No Finding images?
* Có bao nhiêu abnormal images?
* Có image nào vừa `No Finding` vừa abnormal label không?
* Có bbox thiếu coordinate không?
* Có bbox lỗi không?

  * `x_min >= x_max`
  * `y_min >= y_max`
  * `width <= 0`
  * `height <= 0`
* Phân bố bbox theo class như thế nào?
* Có class imbalance nghiêm trọng không?

---

## 10. Script cần tạo trong Phase 1A

Claude cần tạo:

```text
scripts/01A_dataset_overview.py
```

Script phải nhận tham số:

```text
--train-csv
--output-json
--class-csv
--image-summary-csv
--bbox-quality-csv
--report-md
```

Output mặc định:

```text
reports/phase1A_dataset_overview.json
reports/phase1A_class_distribution.csv
reports/phase1A_image_level_summary.csv
reports/phase1A_bbox_quality_summary.csv
reports/phase1A_dataset_overview.md
```

---

## 11. DoD Phase 1A

Phase 1A chỉ pass nếu có đủ:

* Script chạy được bằng Python.
* Có JSON report.
* Có Markdown report.
* Có class distribution CSV.
* Có image-level summary CSV.
* Có bbox quality summary CSV.
* Có thống kê total rows.
* Có thống kê unique images.
* Có thống kê No Finding images.
* Có thống kê abnormal images.
* Có danh sách abnormal classes excluding No Finding.
* Có kiểm tra bbox invalid/missing.
* Có cảnh báo nếu No Finding có bbox.
* Có cảnh báo nếu abnormal class thiếu bbox.
* Không tạo split.
* Không tạo COCO.
* Không train.
* Không dùng test set.
* GPT review pass.

Nếu No Finding có bbox thật, hoặc abnormal labels thiếu bbox hàng loạt, Phase 1A không pass cho đến khi làm rõ rule.

---

## 12. Prompt giao Claude cho Phase 1A

```text
Bạn đang ở repo D:\ssl_detection_xray_v2.

Phase hiện tại: Phase 1A — Dataset Overview.

Bối cảnh nghiên cứu:
- Đề tài: “Nghiên cứu học bán giám sát cho dò tìm bất thường trên X-quang phổi”.
- Dataset: VinBigData Chest X-ray.
- Bài toán: semi-supervised object detection.
- No Finding là ảnh âm tính không có bbox, không phải detection class.
- Metric chính sau này là mAP@0.5:0.95.
- Không được split, không convert COCO, không train trong Phase 1A.

Hãy tạo script:

scripts/01A_dataset_overview.py

Yêu cầu script:
1. Nhận tham số:
   --train-csv
   --output-json default reports/phase1A_dataset_overview.json
   --class-csv default reports/phase1A_class_distribution.csv
   --image-summary-csv default reports/phase1A_image_level_summary.csv
   --bbox-quality-csv default reports/phase1A_bbox_quality_summary.csv
   --report-md default reports/phase1A_dataset_overview.md

2. Đọc annotation CSV VinBigData.
   Script phải tự detect các cột phổ biến:
   - image_id
   - class_name
   - class_id
   - x_min, y_min, x_max, y_max
   Nếu thiếu cột bắt buộc thì báo lỗi rõ ràng.

3. Tính thống kê:
   - total_rows
   - unique_images
   - class_name list
   - class_id list nếu có
   - số row theo class_name
   - số image theo class_name
   - số bbox theo abnormal class
   - số No Finding rows
   - số No Finding images
   - số abnormal images
   - số image có cả No Finding và abnormal label nếu có
   - số bbox missing coordinate
   - số bbox có x_min >= x_max
   - số bbox có y_min >= y_max
   - số bbox width <= 0 hoặc height <= 0
   - min/mean/max width, height, area nếu có bbox hợp lệ

4. No Finding policy:
   - Treat No Finding / no finding / No finding as negative image label.
   - Không đưa No Finding vào detection class.
   - Nếu No Finding có bbox coordinates không null, ghi cảnh báo.
   - Nếu abnormal class thiếu bbox, ghi cảnh báo.

5. Output:
   - phase1A_dataset_overview.json
   - phase1A_class_distribution.csv
   - phase1A_image_level_summary.csv
   - phase1A_bbox_quality_summary.csv
   - phase1A_dataset_overview.md

6. Script phải in console summary:
   - total rows
   - unique images
   - abnormal images
   - No Finding images
   - number of abnormal classes excluding No Finding
   - bbox invalid count
   - warnings

7. Tuyệt đối không:
   - tạo split
   - tạo COCO json
   - copy ảnh
   - đọc DICOM/PNG
   - train
   - tune threshold
   - dùng test set

8. Viết code rõ ràng, có hàm main(), type hints cơ bản, error message dễ hiểu.

Sau khi tạo script, in ra lệnh chạy mẫu với đường dẫn:
python scripts/01A_dataset_overview.py --train-csv data/raw/vinbigdata/annotations/train.csv
```

---

## 13. Lệnh chạy Phase 1A

Nếu annotation nằm ở:

```text
data\raw\vinbigdata\annotations\train.csv
```

chạy:

```cmd
python scripts/01A_dataset_overview.py --train-csv data\raw\vinbigdata\annotations\train.csv
```

Nếu annotation nằm chỗ khác, dùng đúng path thực tế.

---

## 14. Output cần gửi GPT review sau Phase 1A

Gửi các output sau:

```cmd
type reports\phase1A_dataset_overview.md
```

```cmd
type reports\phase1A_dataset_overview.json
```

```cmd
type reports\phase1A_class_distribution.csv
```

```cmd
type reports\phase1A_bbox_quality_summary.csv
```

```cmd
type reports\phase1A_image_level_summary.csv
```

Nếu CSV quá dài, gửi 20 dòng đầu bằng PowerShell:

```cmd
powershell -Command "Get-Content reports\phase1A_class_distribution.csv -TotalCount 20"
```

```cmd
powershell -Command "Get-Content reports\phase1A_bbox_quality_summary.csv -TotalCount 20"
```

```cmd
powershell -Command "Get-Content reports\phase1A_image_level_summary.csv -TotalCount 20"
```

---

## 15. Gate sau Phase 1D

```text
Phase 0 core: PASS
Phase 0 training framework: DEFERRED
Phase 1A — Dataset Overview: PASS
Phase 1B — Annotation Quality: PASS
Phase 1C — Dataset Scope Decision: PASS
Phase 1D — Label Reliability & Kappa Feasibility: PASS

Controlled working scope: LOCKED
Controlled scope size: 4,894 images
Abnormal images retained: 4,394 / 4,394
No Finding images selected: 500 / 10,606
Selection unit: image_id
No Finding row-level sampling used: false

Label reliability:
rad_id available: true
rad_id missing count: 0
radiologists_total: 17
radiologists_per_image_distribution: {'3': 4894}
uniform_rater_count_per_image: true
same_rater_identity_panel_across_images: false
binary_matrix_feasible: true
cohen_kappa_feasible: false
fleiss_kappa_feasible: true
overall_fleiss_kappa_mean: 0.4879

Split train/val/test: LOCKED
COCO conversion: LOCKED
Training: LOCKED
Pseudo-labeling: LOCKED
Threshold tuning: LOCKED
Test-set usage: LOCKED

Next phase:
Phase 2A — Data Standardization / Image-Boundary Validation
```

---

### Phase 1B — Annotation Quality (kiểm tra chất lượng Annotation).

Status: PASS

Date: 2026-06-19

# Prompt giao Claude cho Phase 1B
Bạn đang ở repo:

D:\ssl_detection_xray_v2

Phase hiện tại: Phase 1B — Annotation Quality.

Bối cảnh nghiên cứu:

* Đề tài: “Nghiên cứu học bán giám sát cho dò tìm bất thường trên X-quang phổi”.
* Dataset: VinBigData Chest X-ray.
* Bài toán: semi-supervised object detection.
* Phase 0 core: PASS.
* Phase 1A — Dataset Overview: PASS.
* Phase 1A đã chạy trên full VinBigData train.csv source metadata:

  * 15,000 images.
  * 67,914 annotation rows.
  * 4,394 abnormal images.
  * 10,606 No Finding images.
  * 36,096 abnormal bbox rows.
  * 14 abnormal detection classes.
  * Invalid bbox count: 0.
  * No Finding rows with bbox: 0.
  * Images with both No Finding and abnormal labels: 0.
* Downstream controlled working scope đã khóa sau này là 4,894 images = 4,394 abnormal + 500 No Finding.
* Tuy nhiên Phase 1B vẫn chỉ chạy trên full source metadata train.csv, không tạo subset 4,894.

Nguyên tắc bắt buộc:

* Chỉ làm Phase 1B — Annotation Quality.
* Không split train/val/test.
* Không tạo subset 4,894.
* Không convert COCO.
* Không train.
* Không pseudo-label.
* Không tune threshold.
* Không dùng test set.
* Không đọc pixel ảnh.
* Không đọc DICOM/PNG.
* Chỉ đọc annotation-level metadata từ CSV.
* No Finding là ảnh âm tính không có bbox, không phải detection class.
* Không tự động xóa/sửa annotation, chỉ report lỗi/candidate.

Hãy tạo script:

scripts/01B_annotation_quality.py

Yêu cầu script:

1. Nhận tham số:
   --train-csv
   --output-json default reports/phase1B_annotation_quality.json
   --report-md default reports/phase1B_annotation_quality.md
   --annotation-sanity-md default reports/annotation_sanity_report.md
   --invalid-bbox-csv default reports/invalid_bbox_rows.csv
   --duplicate-csv default reports/duplicate_bbox_candidates.csv
   --class-mapping-csv default reports/phase1B_class_mapping.csv
   --bbox-quality-by-class-csv default reports/phase1B_bbox_quality_by_class.csv
   --image-label-consistency-csv default reports/phase1B_image_label_consistency.csv
   --near-duplicate-iou default 0.95

2. Đọc annotation CSV VinBigData.
   Script phải tự detect hoặc kiểm tra các cột:

   * image_id
   * class_name
   * class_id
   * x_min
   * y_min
   * x_max
   * y_max
   * rad_id nếu có

3. Kiểm tra bbox coordinate sanity:

   * missing coordinate trên abnormal rows
   * non-numeric coordinate
   * x_min < 0
   * y_min < 0
   * x_max < 0
   * y_max < 0
   * x_min >= x_max
   * y_min >= y_max
   * width <= 0
   * height <= 0
   * area <= 0

4. Kiểm tra bbox vượt biên ảnh:

   * Nếu CSV có cột image_width/image_height hoặc width/height đại diện kích thước ảnh, kiểm tra:

     * x_max > image_width
     * y_max > image_height
     * x_min > image_width
     * y_min > image_height
   * Nếu CSV không có image dimensions, không đọc ảnh.
   * Khi không có dimensions, report rõ:
     boundary_check_status = not_evaluable_without_image_dimensions
   * Không đọc DICOM/PNG để lấy shape trong Phase 1B.

5. Kiểm tra No Finding policy:

   * Treat No Finding / no finding / No finding as negative image label.
   * No Finding không phải detection class.
   * No Finding rows phải không có bbox coordinates.
   * Nếu No Finding có bbox, liệt kê vào invalid_bbox_rows.csv.
   * Kiểm tra image_id nào vừa có No Finding vừa có abnormal class.

6. Kiểm tra abnormal annotation consistency:

   * Abnormal class phải có bbox đầy đủ.
   * Nếu abnormal row thiếu bbox, liệt kê.
   * Tạo summary số lỗi theo class_name và class_id.

7. Kiểm tra duplicate / near-duplicate bbox:

   * Exact duplicate: cùng image_id, cùng class_id hoặc class_name, cùng x_min/y_min/x_max/y_max.
   * Near duplicate: cùng image_id, cùng class, IoU >= --near-duplicate-iou.
   * Nếu có rad_id, giữ rad_id trong output.
   * Không xóa duplicate.
   * Chỉ ghi duplicate_bbox_candidates.csv.
   * Trong report phải nói rõ duplicate candidates có thể là multi-radiologist annotations, chưa được xem là lỗi chắc chắn.

8. Kiểm tra class mapping:

   * Mỗi class_id map tới đúng một class_name.
   * Mỗi class_name map tới đúng một class_id.
   * No Finding không được đưa vào abnormal detection class.
   * Xuất phase1B_class_mapping.csv.

9. Output JSON phải có tối thiểu:

   * phase
   * train_csv
   * total_rows
   * unique_images
   * abnormal_rows
   * no_finding_rows
   * abnormal_images
   * no_finding_images
   * abnormal_detection_classes_excluding_no_finding
   * invalid_bbox_total
   * invalid_bbox_by_reason
   * no_finding_with_bbox_count
   * abnormal_missing_bbox_count
   * mixed_no_finding_abnormal_image_count
   * negative_coordinate_count
   * zero_or_negative_area_count
   * exact_duplicate_candidate_count
   * near_duplicate_candidate_count
   * class_mapping_issue_count
   * boundary_check_status
   * warnings
   * forbidden_actions_confirmed

10. Markdown report phải có:

* Executive summary.
* Scope: full source metadata train.csv only, not downstream 4,894 subset.
* Checks performed.
* Key findings.
* Invalid bbox summary.
* Duplicate / near-duplicate summary.
* Class mapping summary.
* No Finding policy summary.
* Boundary check status.
* Research risk interpretation.
* Recommended next action: send outputs to GPT review before ticking checklist.

11. Console summary phải in:

* total rows
* unique images
* invalid bbox total
* no_finding_with_bbox_count
* abnormal_missing_bbox_count
* exact duplicate candidates
* near duplicate candidates
* class mapping issues
* boundary_check_status
* warnings

12. Tuyệt đối không:

* tạo split
* tạo COCO json
* tạo subset 4,894
* copy ảnh
* đọc DICOM/PNG
* train
* pseudo-label
* tune threshold
* dùng test set
* tự động xóa/sửa annotation

13. Code cần có:

* main()
* argparse
* type hints cơ bản
* error message rõ ràng
* tạo thư mục reports nếu chưa có
* xử lý CSV lớn ổn định bằng pandas
* không phụ thuộc GPU/MMDetection

Sau khi tạo script, in ra lệnh chạy mẫu:
Scripts run:

```cmd
python scripts\01B_annotation_quality.py --train-csv data\raw\vinbigdata\annotations\train.csv
```

Outputs generated:

reports/phase1B_annotation_quality.json
reports/phase1B_annotation_quality.md
reports/annotation_sanity_report.md
reports/invalid_bbox_rows.csv
reports/duplicate_bbox_candidates.csv
reports/phase1B_class_mapping.csv
reports/phase1B_bbox_quality_by_class.csv
reports/phase1B_image_label_consistency.csv

DoD result:

Annotation-level bbox sanity: PASS
No Finding policy: PASS
Abnormal bbox completeness: PASS
Class mapping consistency: PASS
Duplicate/near-duplicate candidates reported: PASS
Boundary check: DEFERRED to image-level validation because train.csv has no image dimensions
Forbidden actions avoided: PASS

Key findings:

Total rows: 67,914
Unique images: 15,000
Abnormal rows: 36,096
No Finding rows: 31,818
Abnormal images: 4,394
No Finding images: 10,606
Abnormal detection classes excluding No Finding: 14
Invalid bbox total: 0
No Finding rows with bbox: 0
Abnormal rows missing bbox: 0
Mixed No Finding + abnormal images: 0
Class mapping issues: 0
Exact duplicate candidates: 0
Near-duplicate candidates IoU >= 0.95: 147
Boundary check status: not_evaluable_without_image_dimensions

Research decisions:

Do not delete or modify near-duplicate bbox candidates in Phase 1B.
Treat near-duplicate boxes as multi-radiologist annotation candidates requiring later fusion/handling decision.
Defer image-boundary validation to Phase 2A because Phase 1B is CSV-only.

Issues / risks:

Boundary validity cannot be concluded from CSV alone.
No Finding rows are repeated reader-level rows; future processing must operate at image level for negative images.

Next phase at that time:

Phase 1C — Dataset Scope Decision, now completed PASS.


---

## 16. Phase 1C — Dataset Scope Decision

Status: **PASS**

Date: 2026-07-01

### Mục tiêu

Chính thức hóa downstream controlled working scope ở mức image-level:

```text
4,894 images
= 4,394 abnormal images
+ 500 No Finding images
```

Phase 1C chỉ làm metadata/image-level scope decision.

Không tạo split, không convert COCO, không train, không pseudo-label, không tune threshold, không dùng test set, không đọc DICOM header, không đọc pixel và không đọc image dimensions.

---

### Scripts run

```cmd
python scripts\01C_dataset_scope_decision.py ^
  --train-csv D:\ssl_detection_xray_v2\data\raw\vinbigdata\annotations\train.csv ^
  --manifest-glob "D:\ssl_detection_xray\data\raw\vinbigdata\dicom_subset_chunks\dicom_package_manifest_part_*.csv" ^
  --dicom-root D:\ssl_detection_xray\data\raw\vinbigdata\dicom_subset\train ^
  --chunk-summary D:\ssl_detection_xray\data\raw\vinbigdata\dicom_subset_chunks\dicom_chunk_summary.csv
```

---

### Outputs generated

```text
reports/scope_decision.md
reports/phase1C_dataset_scope_decision.json
reports/phase1C_scope_class_distribution.csv
reports/phase1C_image_level_scope_summary.csv
reports/phase1C_no_finding_selection_audit.csv
data/manifests/phase1C_selected_images_manifest.csv
data/manifests/phase1C_downloaded_image_inventory.csv
data/manifests/phase1C_combined_package_manifest.csv
data/interim/vinbigdata_phase1C_scope_annotations.csv
```

---

### DoD result

```text
Dataset scope decision: PASS
Image-level manifest: PASS
DICOM filename inventory cross-check: PASS
Package manifest cross-check: PASS
train.csv metadata cross-check: PASS
Abnormal retention: PASS
No Finding image-level handling: PASS
Class distribution report: PASS
Forbidden actions avoided: PASS
```

---

### Key findings

```text
Source total rows: 67,914
Source unique images: 15,000
Source abnormal images: 4,394
Source No Finding images: 10,606

Manifest rows: 4,894
Manifest unique image_id: 4,894
DICOM files listed: 4,894
DICOM unique image_id: 4,894

Selected total images: 4,894
Selected abnormal images: 4,394
Selected No Finding images: 500
Selected mixed No Finding + abnormal images: 0
Lost abnormal image count: 0
Abnormal retention rate: 1.0

Selected subset rows: 37,596
Selected abnormal rows: 36,096
Selected No Finding rows: 1,500

Abnormal detection classes excluding No Finding: 14
No Finding is detection class: false
No Finding row-level sampling used: false
Image type / train.csv label mismatch count: 0
Unknown manifest image_id count: 0
Chunk summary match: true
```

---

### Research decisions

```text
Controlled working scope is officially locked to 4,894 image-level samples:
4,394 abnormal images + 500 No Finding images.

The 500 No Finding samples are selected and verified at image_id level, not row level.
The controlled scope is based on the already-downloaded DICOM package manifests and validated against train.csv and DICOM filename inventory.
No Finding remains a negative image label, not a detection class.
The metadata-only subset annotation CSV is created for selected image_id values only.
```

---

### Issues / risks

```text
The controlled scope uses 500 out of 10,606 No Finding images, not the full No Finding pool.
This is a deliberate controlled-scope design decision and must be stated as a limitation.
Boundary validity is not concluded in Phase 1C because image dimensions were not read.
147 near-duplicate bbox candidates from Phase 1B are retained, not deleted or fused.
Fusion/handling of multi-radiologist boxes is deferred to a later phase.
```

---

### Forbidden actions confirmed

```text
No train/val/test split created.
No COCO conversion created.
No training started.
No pseudo-label generated.
No threshold tuned.
No test set used.
No image copied.
No DICOM header read.
No pixel read.
No image dimension read.
No original annotation deleted or edited.
No near-duplicate bbox candidate deleted.
```

---

:::writing{variant="document" id="12370"}
### Next phase at that time

```text
Phase 1D — Kappa feasibility / limitation-aware analysis
Update:
Phase 1D has now been completed with PASS_agreement_computed_and_documented.
The current next phase is Phase 2A — Data Standardization / Image-Boundary Validation.
```
:::

---

## 17. Phase 1D — Label Reliability & Kappa Feasibility

Status: **PASS**

Date: 2026-07-01

### Mục tiêu

Kiểm tra tính khả thi của inter-radiologist agreement / Kappa analysis từ metadata trong controlled working scope đã khóa ở Phase 1C.

Phase 1D chỉ dùng metadata annotation. Không tạo split, không convert COCO, không train, không pseudo-label, không tune threshold, không dùng test set, không đọc pixel ảnh, không đọc DICOM/header/image dimensions và không sửa annotation gốc.

---

### Scripts run

```cmd
python scripts/01D_kappa_feasibility.py
```

---

### Outputs generated

```text
reports/phase1D_kappa_feasibility.md
reports/phase1D_kappa_feasibility.json
reports/phase1D_classwise_agreement_feasibility.csv
reports/phase1D_radiologist_per_image.csv
reports/phase1D_rare_class_kappa_instability.csv
```

---

### DoD result

```text
rad_id availability: PASS
radiologists per image: PASS
image-class-radiologist binary matrix feasibility: PASS
Cohen's Kappa feasibility: PASS
Fleiss' Kappa feasibility: PASS
class-wise image-level agreement: PASS
rare-class kappa instability risk: PASS
label-level agreement vs bbox-level consistency: PASS
forbidden actions avoided: PASS
```

---

### Key findings

```text
rad_id_available: true
rad_id_missing_count: 0
total_images: 4894
total_rows: 37596
radiologists_total: 17
radiologists_per_image_distribution: {'3': 4894}
uniform_rater_count_per_image: true
same_rater_identity_panel_across_images: false
binary_matrix_feasible: true
cohen_kappa_feasible: false
fleiss_kappa_feasible: true
overall_fleiss_kappa_mean: 0.4879
classwise_feasibility_summary: 14 abnormal classes assessed; 14 with feasible Fleiss' Kappa; mean kappa=0.4879
rare_class_instability_summary: 5/14 classes carry kappa_instability_risk (severe=2, moderate=3, low=9); risk is prevalence/rarity-driven, not measured instability
label_level_agreement_status: evaluable_fleiss_computed
bbox_level_consistency_status: evaluated_descriptive_only
```

---

### Research decisions

```text
Fleiss' Kappa is computed at image-level class agreement.
Cohen's Kappa is not used as the main agreement statistic because each image has 3 radiologist ratings.
Kappa/agreement is used only as data-quality evidence and limitation evidence.
Kappa is not a model metric.
Kappa is not used for split/model/threshold selection.
Kappa is not used for training or pseudo-labeling.
Kappa is not used to delete, fuse, or edit annotations.
BBox-level consistency is kept separate from label-level agreement and remains descriptive only.
```

---

### Issues / risks

```text
Negative class decisions are inferred from read-coverage according to VinBigData labelling convention.
Kappa can be affected by prevalence imbalance.
5/14 abnormal classes carry kappa_instability_risk.
BBox-level consistency is not a bbox fusion policy.
Near-duplicate bbox handling is still deferred to a later annotation standardization decision.
```

---

### Forbidden actions confirmed

```text
No train/val/test split created.
No COCO conversion created.
No training started.
No pseudo-label generated.
No threshold tuned.
No test set used.
No pixel read.
No DICOM/header read.
No image dimensions read.
No boundary validation performed.
No annotation deleted or edited.
No near-duplicate bbox deleted or fused.
No Kappa used as model metric.
No Kappa used for split/model/threshold.
```

---

### Next phase

```text
Phase 2A — Data Standardization / Image-Boundary Validation
```

:::writing{variant="document" id="93016"}
---

## 18. Phase 2A — Data Standardization / Image-Boundary Validation

Status: **PASS**

Date: 2026-07-08

### Mục tiêu

Kiểm tra DICOM availability, image dimensions và bbox boundary validity trong controlled working scope 4,894 images.

Phase 2A chỉ đọc DICOM metadata/header để lấy dimensions và validate bbox. Không tạo split, không convert COCO, không train, không pseudo-label, không tune threshold, không dùng test set, không sửa annotation và không tạo processed training images.

---

### Scripts run

```cmd
python scripts\02A_dicom_bbox_boundary_validation.py ^
  --annotations-csv data\interim\vinbigdata_phase1C_scope_annotations.csv ^
  --manifest-csv data\manifests\phase1C_selected_images_manifest.csv ^
  --dicom-root D:\ssl_detection_xray\data\raw\vinbigdata\dicom_subset\train
```

### Outputs generated

```text
reports/phase2A_dicom_bbox_validation.md
reports/phase2A_dicom_bbox_validation.json
reports/phase2A_image_metadata.csv
reports/phase2A_image_availability.csv
reports/phase2A_bbox_boundary_validation.csv
reports/phase2A_invalid_bbox_candidates.csv
reports/phase2A_dicom_read_errors.csv
```
---

  ### DoD result

```text
DICOM availability: PASS
DICOM metadata/header read: PASS
Image dimension extraction: PASS
BBox boundary validation: PASS
No Finding policy: PASS
Forbidden actions avoided: PASS
```
---

  ### Key findings

```text
dicom_files_indexed_under_root: 4894
total_annotation_rows: 37596
unique_annotation_images: 4894
manifest_rows: 4894
manifest_unique_images: 4894
selected_scope_expected_images: 4894
availability_checked_image_count: 4894
abnormal_images: 4394
no_finding_images: 500
abnormal_rows: 36096
no_finding_rows: 1500
dicom_available_count: 4894
dicom_missing_count: 0
dicom_read_success_count: 4894
dicom_read_error_count: 0
image_dimension_available_count: 4894
image_dimension_missing_count: 0
abnormal_bbox_rows_checked: 36096
bbox_boundary_valid_count: 36096
bbox_boundary_invalid_count: 0
no_finding_with_bbox_count: 0
abnormal_missing_bbox_count: 0
annotation_not_in_manifest_count: 0
manifest_not_in_annotation_count: 0
duplicate_manifest_image_id_count: 0
warnings: []
dod_pass_candidate: true
```
---

  ### Image dimension summary

```text
width_min: 1320
width_max: 3320
width_mean: 2491.66
height_min: 1416
height_max: 3408
height_mean: 2835.09
distinct_wh_pairs: 2186
```
---

  ### Research decisions

```text
All 4,894 controlled-scope DICOM files are available on local disk.
All 4,894 DICOM files are readable at metadata/header level.
Image dimensions are available for all controlled-scope images.
All 36,096 abnormal bbox rows are valid within original image boundaries.
BBox convention is treated as xyxy on original image coordinates.
No Finding remains a negative image label without bbox and is not a detection class.
No bbox was edited, clamped, deleted or fused.
No image was copied, converted, normalized or saved as processed training data.
```
---

  ### Issues / risks

```text
Pixel array decoding was not checked in the main run because pixel_array_checked=false.
This is acceptable for Phase 2A because the phase objective is metadata/header dimension and bbox boundary validation.
Canonical schema is not created yet.
COCO conversion is not created yet.
Train/val/test split is not created yet.
Framework dataloader / empty image loading is not checked yet.
Training is still locked.
Pseudo-labeling is still locked.
Threshold tuning is still locked.
```
---

  ### Forbidden actions confirmed

```text
No train/val/test split created.
No COCO conversion created.
No training started.
No pseudo-label generated.
No threshold tuned.
No test set used.
No annotation deleted or edited.
No bbox clamped or modified.
No near-duplicate bbox deleted or fused.
No processed training images created.
No image files copied.
No image files converted.
No PNG/JPG created.
```
---

  ### Next phase

```text
Phase 2B — Canonical Schema
```
---
:::

## Phase 2B — Canonical Detection Annotation Schema

Status: **PASS**

Date: 2026-07-08

### Mục tiêu

Tạo canonical detection annotation schema cho controlled working scope 4,894 images.

Phase 2B chỉ tạo schema canonical trung gian. Không convert COCO, không tạo split, không train, không pseudo-label, không tune threshold, không dùng test set, không sửa annotation gốc, không clamp bbox, không xóa bbox và không fuse near-duplicate bbox.

### Scripts run

```cmd
python scripts\02B_build_canonical_schema.py ^
  --annotations-csv data\interim\vinbigdata_phase1C_scope_annotations.csv ^
  --manifest-csv data\manifests\phase1C_selected_images_manifest.csv ^
  --image-metadata-csv reports\phase2A_image_metadata.csv ^
  --bbox-boundary-csv reports\phase2A_bbox_boundary_validation.csv ^
  --output-dir data\processed\canonical ^
  --report-md reports\phase2B_canonical_schema_report.md ^
  --validation-json reports\phase2B_canonical_schema_validation.json ^
  --no-finding-audit-csv reports\phase2B_no_finding_policy_audit.csv ^
  --schema-errors-csv reports\phase2B_schema_consistency_errors.csv
```

 ### Outputs generated

```text
data/processed/canonical/canonical_image_table.csv
data/processed/canonical/canonical_bbox_table.csv
data/processed/canonical/canonical_class_mapping.csv
reports/phase2B_canonical_schema_report.md
reports/phase2B_canonical_schema_validation.json
reports/phase2B_no_finding_policy_audit.csv
reports/phase2B_schema_consistency_errors.csv
```
---

 ### DoD result

```text
Canonical image table: PASS
Canonical bbox table: PASS
Canonical class mapping: PASS
No Finding policy audit: PASS
Schema consistency validation: PASS
Portable path policy: PASS
Forbidden actions avoided: PASS
```
---

 ### Key findings

```text
canonical_image_rows: 4894
canonical_image_unique_images: 4894
canonical_bbox_rows: 36096
canonical_class_count: 14
abnormal_images: 4394
no_finding_images: 500
no_finding_policy_pass: true
no_finding_in_detection_classes: false
bbox_without_image_count: 0
image_without_metadata_count: 0
bbox_missing_dimension_count: 0
bbox_invalid_count: 0
class_mapping_issue_count: 0
schema_error_count: 0
portable_path_policy_pass: true
relative_dicom_path_missing_count: 0
relative_dicom_path_absolute_count: 0
local_dicom_path_absolute_count: 4894
path_root_variable: VINBIGDATA_DICOM_ROOT
warnings: []
dod_pass_candidate: true
```
---

 ### Research decisions

```text
Canonical schema is accepted as the intermediate detection annotation schema.
Canonical image table keeps all 4,894 controlled-scope images.
Canonical bbox table keeps all 36,096 abnormal bbox rows.
Canonical class mapping contains exactly 14 abnormal detection classes.
No Finding remains a negative image-level sample with no bbox and is not a detection class.
No Finding is excluded from canonical bbox annotations and detection class mapping.
BBox format remains xyxy_original_image.
No bbox was edited, clamped, deleted, fused, or converted.
147 near-duplicate bbox candidates remain retained; fusion/handling remains deferred.
Portable path policy is adopted: downstream should resolve image files using VINBIGDATA_DICOM_ROOT + relative_dicom_path.
Phase 2B is not a COCO dataset and not a train/val/test split.
Dataset is still not training-ready until COCO/split/loading phases pass DoD.
```
---

 ### Issues / risks

```text
local_dicom_path stores absolute local evidence paths but must not be used as canonical downstream identifiers.
Remote/GPU environments must set VINBIGDATA_DICOM_ROOT or an equivalent data-root config.
source_row_id traces to the Phase 1C controlled-scope annotation file, not necessarily the original full VinBigData train.csv row index.
Framework dataloader validation has not been performed.
Empty image loading check has not been performed.
COCO conversion has not been performed.
Train/val/test split has not been created.
Near-duplicate bbox handling is still deferred.
```
---

 ### Forbidden actions confirmed

```text
No train/val/test split created.
No COCO conversion created.
No training started.
No pseudo-label generated.
No threshold tuned.
No test set used.
No pixel_array read.
No image copied.
No image converted.
No processed training images created.
No annotation deleted or edited.
No bbox clamped or modified.
No near-duplicate bbox deleted or fused.
No Finding was not added as a detection class.
```
---

 ### Next phase

```text
Phase 2C — Framework & Format Decision / COCO Conversion Planning
```
---

## 19. Phase 2D.1C — MMDetection Dataset Loading & Full Pipeline Audit

Status: **PASS**

Date: 2026-07-30

### Mục tiêu

Xác minh COCO master JPG của controlled working scope có thể được MMDetection nạp đúng, giữ đúng chính sách ảnh zero-GT và đi qua pipeline/dataloader mà không làm sai image identity, bbox hoặc class label.

Phase 2D.1C là validation gate ở mức dataset loading và full data pipeline. Kết quả PASS chỉ xác nhận dataset đã sẵn sàng kỹ thuật cho bước training. Phase này không cấp quyền chạy training, không tạo pseudo-label, không tune threshold và không sử dụng test set.

### Input và cấu hình

```text
Controlled working scope: 4,894 images
Abnormal images: 4,394
Zero-GT / No Finding images: 500
Detection classes: 14 abnormal classes
Annotation format: COCO Detection JSON
Image representation: high-quality JPG generated by the locked
DICOM metadata-aware, standard-aligned reference representation pipeline
MMDetection empty-GT retention policy: filter_empty_gt=False
Validation config: configs/validation/phase2D1C_mmdet_dataset_loading.py
```

### Scripts và tests

```text
scripts/02D1C_validate_mmdet_dataset_loading.py
tests/test_phase2D1C_mmdet_dataset_loading_guardrails.py
```

Validator lấy record theo public indexed API `dataset.get_data_info(index)`. Không dựa vào `dataset.data_list`, vì MMEngine có thể serialize dữ liệu và xóa danh sách này khi `serialize_data=True`.

### Lệnh validation chính

```bash
MPLBACKEND=Agg "$PYTHON" -m py_compile \
  scripts/02D1C_validate_mmdet_dataset_loading.py \
  tests/test_phase2D1C_mmdet_dataset_loading_guardrails.py

MPLBACKEND=Agg "$PYTHON" -m pytest \
  tests/test_phase2D1C_mmdet_dataset_loading_guardrails.py \
  -q -rs

MPLBACKEND=Agg "$PYTHON" \
  scripts/02D1C_validate_mmdet_dataset_loading.py \
  --config configs/validation/phase2D1C_mmdet_dataset_loading.py \
  --full-pipeline-audit
```

### Outputs generated

```text
reports/phase2D1C_mmdet_dataset_errors.csv
reports/phase2D1C_mmdet_dataset_image_audit.csv
reports/phase2D1C_mmdet_dataset_loading_report.json
reports/phase2D1C_mmdet_dataset_loading_report.md
```

### DoD result

```text
MMDetection dataset construction: PASS
COCO image/category mapping: PASS
filter_empty_gt=False retention: PASS
filter_empty_gt=True exclusion behavior: PASS
Standard dataloader batch: PASS
Forced empty-GT dataloader batch: PASS
Full pipeline audit: PASS
BBox/label validation: PASS
Serialized-data regression guard: PASS
Errors: 0
Dataset training-ready: TRUE
Training authorized: FALSE
```

### Key findings

```text
full_pipeline_audit: true
abnormal_audited: 4394/4394
empty_audited: 500/500
num_audited: 4894/4894
all_audited_valid: true
filter_empty_gt=false: retained 4894/4894 images
filter_empty_gt=true: excluded exactly 500 zero-GT images
regression_tests: 35 passed
errors: 0
dataset_training_ready: true
training_authorized: false
```

### Regression guard cho `serialize_data=True`

Regression test mô phỏng đúng hành vi cần khóa:

```text
len(dataset) > 0
dataset.data_list == []
dataset.get_data_info(index) trả record hợp lệ
dataset_image_ids_in_order(dataset) vẫn trả đủ ID theo đúng thứ tự
```

Kết quả:

```text
35 passed in 7.58s
PHASE 2D.1C REGRESSION RECHECK: PASS
```

### Evidence integrity

```text
0780595f5ff69c36329f05d69f7bb353fd095f32a0df3f76b16f039143a5f2cf  reports/phase2D1C_mmdet_dataset_errors.csv
00df8ed311e6de0ba863fa8e5a90551d34ef080b12cdd0063b6397fdfd76e474  reports/phase2D1C_mmdet_dataset_image_audit.csv
dabb3dbf27373c5271cdb3137406b583a9d3b7ee607ca2faabe18033ab772ca8  reports/phase2D1C_mmdet_dataset_loading_report.json
fb0170cadee8b7b66d81be4681af0b8955ba3c4e6b584fb5faf35d8e054b9ce9  reports/phase2D1C_mmdet_dataset_loading_report.md
```

Hash được kiểm tra lại sau khi bổ sung regression test và không thay đổi.

### Files thuộc Phase 2D.1C

```text
configs/validation/phase2D1C_mmdet_dataset_loading.py
reports/phase2D1C_mmdet_dataset_errors.csv
reports/phase2D1C_mmdet_dataset_image_audit.csv
reports/phase2D1C_mmdet_dataset_loading_report.json
reports/phase2D1C_mmdet_dataset_loading_report.md
scripts/02D1C_validate_mmdet_dataset_loading.py
tests/test_phase2D1C_mmdet_dataset_loading_guardrails.py
```

File backup tạm `scripts/02D1C_validate_mmdet_dataset_loading.py.bak_before_serialized_data_fix` đã được loại bỏ và không thuộc evidence/commit.

### Research decisions

```text
COCO master JPG đã vượt qua MMDetection dataset-loading gate trên toàn bộ controlled scope.
No Finding tiếp tục là ảnh âm tính zero-GT, không phải detection class.
filter_empty_gt=False là policy cần dùng khi phải giữ 500 ảnh âm tính trong dataset.
filter_empty_gt=True chỉ được kiểm tra như một guardrail và loại đúng 500 ảnh zero-GT.
Public API get_data_info(index) là cơ chế truy cập record dùng cho validation khi MMEngine serialize dataset.
Dataset được đánh dấu training-ready ở cấp kỹ thuật.
Training vẫn chưa được cho phép.
```

### Issues / risks

```text
dataset_training_ready=True không đồng nghĩa với toàn bộ experimental protocol đã training-ready.
training_authorized=False vẫn khóa mọi lệnh training.
Full pipeline audit xác minh data loading và annotation integrity; nó không thay thế split leakage audit, seed locking, training-config review hoặc experiment authorization.
Notebook chỉ tái lập sạch sau khi commit chứa script đã sửa và regression test thứ 35 được push, rồi cell clone/checkout được khóa vào commit hash đó.
Không được giữ các cell vá source, xóa backup hoặc git add như một phần của notebook validation cuối.
```

### Forbidden actions confirmed

```text
No supervised training started.
No SSL training started.
No pseudo-label generated.
No confidence threshold tuned.
No checkpoint selected.
No test set used.
No annotation deleted, clamped, fused, or edited.
No No Finding image converted into a detection class.
No near-duplicate bbox automatically removed.
```

### Trạng thái gate chính thức

```text
dataset_training_ready: True
training_authorized: False
```

Không được diễn giải hai cờ này thành “đã được phép training”.

### Resolution update sau Phase 2D.1C

Các bước commit và khóa notebook nêu trong kế hoạch tại thời điểm review
Phase 2D.1C đã hoàn tất:

```text
0bf30cb phase2D1C: validate MMDetection dataset loading
5ce88f6 docs: add Phase 2D1C prompt and environment snapshot
267d4bc docs: add locked Phase 2D1C reproducibility notebook
```

Notebook chính thức:

```text
notebooks/Phase_2D_1C_locked_0bf30cb.ipynb
```

Notebook khóa source tại commit `0bf30cb`, không giữ output cũ và không chứa
cell sửa code, `git add`, `git commit` hoặc `git push`.

Phase 2D.1D đã hoàn tất:

```text
Phase 2D.1D — Evidence Consolidation, GPT Review & Closure
Status: CLOSED / PASS
```

Phase 2E, Phase 2F và Phase 2F.1 sau đó đã hoàn tất (xem các mục tương ứng và
closure update ở đầu handoff). Trạng thái hiện tại:

```text
Phase 2F.1 — Seed Protocol: CLOSED / PASS
Phase 3A — Dataset Diagnostics Before Training: CLOSED / PASS
Next implementation phase: NOT OPENED
```

---

## 20. Phase 2D.1D — Evidence Consolidation, GPT Review & Closure

Status: **CLOSED / PASS**

Date opened: 2026-07-31
Date closed: 2026-07-31

### Mục tiêu và phạm vi

Phase 2D.1D tổng hợp và đối chiếu evidence của:

```text
Phase 2D.1A — Image Representation Protocol Decision
Phase 2D.1B-Pilot — Representative DICOM-to-JPG Pilot
Phase 2D.1B-Full — Full Controlled-Scope Conversion & Validation
Phase 2D.1C — MMDetection Dataset / Empty-Image Loading Validation
```

Phase này chỉ thực hiện evidence consolidation, consistency review,
documentation correction và closure decision. Phase này không:

```text
Tạo train/validation/test split
Tạo labeled/unlabeled split
Chạy supervised hoặc SSL training
Sinh pseudo-label
Tune threshold
Tính AP/mAP
Sử dụng test set
```

Split locking thuộc Phase 2E. Labeled/unlabeled split thuộc Phase 2F. Seed
protocol thuộc Phase 2F.1.

### Evidence inventory đã review

Đã đọc và đối chiếu trực tiếp:

```text
configs/protocol/phase2D1_jpg_representation.yaml
reports/phase2D1_image_representation_decision.json
reports/phase2D1_image_representation_decision.md
reports/phase2D1B_pilot_decision_template.json
reports/phase2D1B_pilot_validation.json
reports/phase2D1B_pilot_fidelity_metrics.csv
reports/phase2D1B_pilot_bbox_roi_metrics.csv
reports/phase2D1B_pilot_quality_summary.csv
reports/phase2D1B_pilot_quality_pairwise.csv
reports/phase2D1B_pilot_geometry_validation.csv
reports/phase2D1B_pilot_visual_audit_manifest.csv
reports/phase2D1B_full_preflight.json
reports/phase2D1B_full_validation.json
reports/phase2D1B_full_validation.md
reports/phase2D1B_full_promotion.json
reports/phase2D1B_full_cleanup_audit.json
reports/phase2D1B_full_metadata_audit.csv
reports/phase2D1B_full_bbox_audit.csv
reports/phase2D1B_full_no_finding_audit.csv
reports/phase2D1B_full_errors.csv
data/processed/coco/coco_master.json
data/processed/coco/coco_master_jpg.json
configs/validation/phase2D1C_mmdet_dataset_loading.py
scripts/02D1C_validate_mmdet_dataset_loading.py
tests/test_phase2D1C_mmdet_dataset_loading_guardrails.py
reports/phase2D1C_mmdet_dataset_errors.csv
reports/phase2D1C_mmdet_dataset_image_audit.csv
reports/phase2D1C_mmdet_dataset_loading_report.json
reports/phase2D1C_mmdet_dataset_loading_report.md
notebooks/Phase_2D_1C_locked_0bf30cb.ipynb
```

### Kết quả evidence review

```text
Controlled-scope images: 4,894
Abnormal images: 4,394
No Finding / zero-GT images: 500
Abnormal bbox annotations: 36,096
Abnormal detection classes: 14

Final JPEG quality: 95 / LOCKED
Full JPG conversion: PASS
Geometry and bbox invariance: PASS
Full-conversion errors: 0
COCO-JPG path-only derivative: PASS

MMDetection full-scope loading: PASS
filter_empty_gt=False retention: 4,894/4,894
filter_empty_gt=True control: removed exactly 500 zero-GT images
Full bbox/label pipeline audit: 4,894/4,894 PASS
Phase 2D.1C errors: 0
Guardrail test functions: 35

dataset_training_ready: true
training_authorized: false
```

So sánh trực tiếp `coco_master.json` và `coco_master_jpg.json` xác nhận:

```text
Image ID sets: identical
Image dimensions: identical
Annotations: exactly identical
Categories: exactly identical
Only image.file_name changed from DICOM path to train/<image_id>.jpg
```

Q100 có numerical fidelity cao hơn Q95 trên toàn bộ 64 whole-image comparisons
và 402 bbox-ROI comparisons. Q95 được chọn như một
fidelity–storage/I/O trade-off; không được diễn giải là có detector performance
tốt hơn Q100.

### Evidence-state interpretation

Các generated artifact sau là immutable pre-review/candidate records:

```text
reports/phase2D1B_pilot_validation.json
reports/phase2D1B_pilot_visual_audit_manifest.csv
reports/phase2D1B_full_validation.json
reports/phase2D1B_full_validation.md
reports/phase2D1B_full_promotion.json
reports/phase2D1B_full_cleanup_audit.json
```

Việc các file này giữ trạng thái `OPEN`, `PENDING_GPT` hoặc
`full_conversion_completed=false` phản ánh thời điểm chúng được sinh. Closure
subsequent được ghi trong decision record, research log và protocol lifecycle
state; không sửa ngược generated evidence lịch sử.

Protocol SHA-256 `1528da27758d35786847141c37d0ddb754dddb146aff116a8f3a9a7b07221229`
được Pilot và Full evidence tham chiếu nhất quán. Current YAML có fingerprint
khác vì các lifecycle/completion fields được cập nhật sau execution; locked
transformation policy không thay đổi. Khác biệt này không ảnh hưởng tính hợp
lệ kỹ thuật và không yêu cầu chạy lại conversion.

### Documentation correction tracking

```text
research_log.md: UPDATED; consistency check PASS
PHASE_HANDOFF.md: UPDATED; consistency check PASS
PROJECT_CONTEXT.md: UPDATED; consistency check PASS
README.md: UPDATED; consistency check PASS
CHECKLIST_TRIEN_KHAI_FULL.xlsx: UPDATED; consistency check PASS
configs/protocol/phase2D1_jpg_representation.yaml: UPDATED; consistency check PASS
```

Phase 2D.1C chỉ kiểm định dataloader với `num_workers=0`. Multi-worker loading
không được kiểm định và không được tuyên bố PASS.

### Training authorization

```text
dataset_training_ready: true
training_authorized: false
```

Đóng Phase 2D.1D không tự động chuyển `training_authorized` thành `true`.
Training chỉ được xem xét sau khi split, leakage, labeled/unlabeled membership,
seed protocol và training configuration đã được thực hiện, review và PASS
trong các phase tiếp theo.

### Trạng thái hiện tại

```text
Phase 2D.1 technical evidence inventory: COMPLETED
Phase 2D.1 documentation consistency review: COMPLETED
Phase 2D.1D final closure decision: PASS
Phase 2D.1D: CLOSED / PASS
Phase 2D.1 overall: CLOSED / PASS
Training authorized: FALSE
```

Không chạy lại Phase 2D.1A, 2D.1B-Pilot, 2D.1B-Full hoặc 2D.1C nếu không xuất
hiện lỗi kỹ thuật hoặc bằng chứng mâu thuẫn mới.

---

## Ghi chú bổ sung sau closure — Annotation QA & post-conversion supporting QA (2026-08-06)

> **NOTE ONLY — không thay đổi bất kỳ trạng thái phase/gate hiện có nào.**
> Ghi chú này chỉ tổng hợp và làm rõ evidence đã có/bổ sung sau closure; không
> mở lại Phase 1B, Phase 2D.1 hay Phase 2D.1D. Ghi chú được tạo trước closure
> Phase 2E; trạng thái phase hiện hành phải đọc theo mục 3 và mục Phase 2E bên
> dưới. `dataset_training_ready=True` và `training_authorized=False` vẫn giữ nguyên.

### Bounding-box validity và duplicate/near-duplicate annotation

Evidence Phase 1B xác nhận annotation-quality analysis đã bao gồm kiểm tra
bounding box và phát hiện duplicate/near-duplicate annotation. Phân tích được
thực hiện trên 67,914 annotation rows của 15,000 ảnh nguồn; trong đó có 36,096
abnormal bbox rows thuộc phạm vi annotation bất thường.

```text
Invalid abnormal bbox: 0
Exact duplicate candidates: 0
Near-duplicate threshold: IoU >= 0.95
Near-duplicate candidates: 147 bbox records
Near-duplicate groups: 71 (image, class) groups
```

Exact duplicate được xét trong cùng `(image_id, class)` với cùng tọa độ
`(x_min, y_min, x_max, y_max)`. Near-duplicate được xét pairwise trong cùng
`(image_id, class)` theo ngưỡng IoU >= 0.95. Con số 147 là số **bbox records
được flag là candidate**, không phải 147 cặp bbox.

Không phát hiện bounding box bất thường không hợp lệ nên không phát sinh trường
hợp cần loại bỏ vì lý do invalid geometry. Các near-duplicate candidates được
giữ lại có chủ đích thay vì xóa/fuse tự động, vì đây là dữ liệu
multi-radiologist và sự trùng/chồng lấp cao không tự động chứng minh annotation
là lỗi nhập liệu.

Evidence liên quan:

```text
scripts/01B_annotation_quality.py
reports/phase1B_annotation_quality.md
reports/duplicate_bbox_candidates.csv
```

### Post-conversion COCO/JPEG supporting QA

Sau chuyển đổi, exhaustive automated QA trên phạm vi thực nghiệm xác nhận tính
nhất quán kỹ thuật của 4,894/4,894 ảnh và 36,096/36,096 bounding boxes, bao gồm
kích thước ảnh, geometry, hệ tọa độ/reference và các ràng buộc bbox có thể kiểm
tra bằng code. Không phát hiện missing/ambiguous/unreadable JPEG hoặc lỗi
reference/category trong phạm vi audit.

Representative/stress visual QA được thực hiện trên 16 ảnh sau chuyển đổi,
bao phủ 14/14 lớp bất thường và gồm 4 zero-GT/No Finding samples. Kiểm tra
overlay không phát hiện sai lệch không gian rõ rệt giữa JPEG và annotation COCO
trên tập mẫu được chọn.

### JPEG Q95 fidelity và giới hạn diễn giải

Sai khác do mã hóa JPEG Q95 đã được đánh giá định lượng trong pilot bằng cách
so sánh decoded JPEG với lossless pre-JPEG reference trên 64 whole-image
comparisons và 402 bbox-ROI comparisons. Các metric fidelity của pilot gồm
MAE/RMSE/PSNR/SSIM; Q100 có numerical fidelity cao hơn Q95, còn Q95 được khóa
theo quyết định fidelity-storage/I/O trade-off đã ghi nhận trước đó.

Diễn giải khoa học được giới hạn như sau: tính hợp lệ của bounding box được
kiểm tra trên toàn bộ dữ liệu liên quan và không phát hiện bbox invalid cần loại
bỏ; sau chuyển đổi, kiểm tra tự động trên 4,894 ảnh và 36,096 bounding boxes xác
nhận tính nhất quán về kích thước, geometry và hệ tọa độ, còn representative
visual QA trên 16 mẫu không phát hiện sai lệch không gian rõ rệt. Sai khác do
mã hóa JPEG Q95 được định lượng so với lossless pre-JPEG reference trong pilot.

Các kết quả trên **không** được diễn giải thành tuyên bố rằng toàn bộ thông tin
gốc được bảo toàn tuyệt đối. Cụ thể, chúng không chứng minh toàn bộ 36,096
annotations đúng về mặt lâm sàng, không chứng minh JPEG Q95 là lossless hoặc
tương đương pixel-wise/diagnostic với DICOM gốc, và không chứng minh preprocessing
là optimal. Phạm vi kết luận được giới hạn ở annotation validity/duplicate QA,
technical geometry/coordinate consistency và JPEG fidelity đã được đánh giá.

---

## 21. Phase 2E — Fixed Train/Validation/Test Split

Status: **CLOSED / PASS**

Date closed: 2026-08-07

### Mục tiêu và nguyên tắc khóa

Phase 2E tạo một train/validation/test split cố định từ phạm vi thực nghiệm đã
khóa gồm 4,894 ảnh. Đơn vị phân hoạch là `image_id`. Candidate split đã qua gate
được tái tạo chính xác trước khi materialize; bước materialization không được
stratify lại, tìm seed mới hoặc tối ưu lại candidate.

Protocol sử dụng tỷ lệ mục tiêu 70/15/15 và giữ No Finding/zero-GT trong cả ba
split. Test set được khóa để chỉ dùng cho final evaluation; không dùng test để
tune threshold, chọn checkpoint, chọn model/backbone, augmentation hoặc các
quyết định ablation.

Tất cả experiment downstream phải sử dụng cùng fixed test set. Việc tuân thủ
quy tắc này phải được xác minh tiếp từ config/log của từng experiment; tại thời
điểm closure Phase 2E chưa được diễn giải thành “mọi experiment đã chạy trên
cùng test set”.

### Script và execution evidence

Script materialization chính thức:

```text
scripts/02E_build_fixed_split.py
```

Candidate implementation được kế thừa và khóa từ R2:

```text
scripts/02E_C0_R2_exact_constrained_candidate_split.py
```

Console evidence sau khi chạy trên máy dự án:

```text
02E_build_fixed_split_output.txt
```

Execution mode:

```text
STAGE_VALIDATE_THEN_PROMOTE
SEED_POLICY=INHERIT_R2_PRE_SPECIFIED_LOCKED_NO_SEED_SEARCH
R2_CANDIDATE_LOCK=PASS
FIXED_SPLIT_GATE=PASS
FILES_WRITTEN=9
FIXED_SPLIT_CREATED=True
FIXED_SPLIT_VALIDATED=True
```

Script ghi vào staging, đọc lại độc lập các artifact vừa tạo, chỉ promote khi
toàn bộ validation PASS, từ chối ghi đè artifact đích đã tồn tại và rollback
các file đã promote nếu quá trình promotion thất bại giữa chừng.

### Fixed split đã khóa

```text
Split            Train      Val       Test
Images           3,426      734       734
Annotations      25,260     5,399     5,437
No Finding       350        75        75
Categories       14         14        14
```

Tổng kiểm tra:

```text
Image union: 4,894 / 4,894
Annotation union: 36,096 / 36,096
Manifest rows: 4,894
Train–val image overlap: 0
Train–test image overlap: 0
Val–test image overlap: 0
```

Phân bố negative/No Finding:

```text
Train: 350 / 3,426 = 10.216%
Val:    75 /   734 = 10.218%
Test:   75 /   734 = 10.218%
```

Test set vì vậy chứa 75 ảnh No Finding/zero-GT và hỗ trợ tính chỉ số
`FP per negative image`. Việc có 75 ảnh âm tính cho phép tính metric này nhưng
không tự động đảm bảo độ bất định thống kê nhỏ; nếu dùng để so sánh phương pháp,
cần báo cáo uncertainty phù hợp trong protocol đánh giá/thống kê.

### Identity lock và checksum

SHA-256 của membership `image_id` tái tạo từ candidate R2:

```text
train: 628b9bb8ba25129a928abe994b101b4c4efd5588d389feb60da6de2a371fa11a
val:   87c23ebed4d1e6965731fc0b31245859f49e777119813c6152efde3531ba58c6
test:  1f7903e069e872bf2e5fe13bb4d0fa257dc4a1c2c8290a621d3f7286ada66b37
```

SHA-256 của ba COCO JSON sau independent readback validation:

```text
instances_train.json: 0f3c37a6f1b5bcc6971b01fd4c69c7a2a3a1e8bee145c11488c7a658dcbbebe3
instances_val.json:   33064f47ba690be13e9d418d8ec5d4a6a2bec8482c24ea3eadf83fb33d01762a
instances_test.json:  e1a73110e92af2656276d6c532035afe474b6ac6f2b2f03849834c036e1c00a4
```

### Artifact chính thức

Phase 2E materialization tạo đúng 9 artifact:

```text
data/processed/coco/instances_train.json
data/processed/coco/instances_val.json
data/processed/coco/instances_test.json
data/manifests/fixed_split_manifest.csv
data/manifests/split_lock_manifest.json
data/manifests/leakage_check_report.json
reports/split_negative_distribution.csv
reports/02E_build_fixed_split_validation_report.json
reports/02E_build_fixed_split_log.json
```

### Leakage interpretation

Kết quả phân hoạch chứng minh zero overlap ở cấp `image_id`. Annotation thuộc
về ảnh tương ứng và được phân bổ theo ownership của ảnh; toàn bộ 36,096
annotations được bao phủ trong ba split mà không trộn image membership giữa
train/validation/test.

Không được mở rộng kết luận này thành `patient-level leakage = 0`. Trong bản
VinDr-CXR đã de-identify dùng cho dự án, `PatientID` và các định danh DICOM có
thể dùng để liên kết ảnh theo bệnh nhân/study không còn khả dụng. Audit dữ liệu
trong phạm vi dự án không cung cấp khóa bệnh nhân/study để thực hiện grouped
split hoặc patient-level leakage check độc lập. Vì vậy kết luận chính xác là:

```text
Image-level leakage: 0 — PASS
Annotation ownership/leakage relative to image split: PASS
Patient-level leakage: NOT ASSESSABLE from the released de-identified data
```

Đây là giới hạn của dữ liệu công bố, không phải bằng chứng rằng patient-level
leakage tồn tại, và cũng không phải cơ sở để tuyên bố patient-level leakage bằng 0.

### DoD và trạng thái gate

```text
Fixed candidate reproduced and identity-hash matched: PASS
Fixed manifests written: PASS
COCO train/val/test materialized: PASS
Independent readback validation: PASS
All three splits contain 14 categories: PASS
No Finding retained in train/val/test: PASS
Pairwise image overlap = 0: PASS
Image union = 4,894: PASS
Annotation union = 36,096: PASS
Checksums recorded: PASS
Reproducibility/overwrite guardrails: PASS
Phase 2E — Fixed Train/Validation/Test Split: CLOSED / PASS
```

Phase 2E closure không thay đổi training authorization:

```text
dataset_training_ready=True
training_authorized=False
```

### Next phase

```text
Phase 2F — Labeled/Unlabeled Construction: CLOSED / PASS
Phase 2F.1 — Seed Protocol: CLOSED / PASS
Phase 3A — Dataset Diagnostics Before Training: CLOSED / PASS
Next implementation phase: Phase 4 — Supervised Baseline / NEXT / NOT STARTED
```

Phase 2F đã thực hiện labeled/unlabeled construction chỉ từ fixed training split
`instances_train.json`, theo các ngân sách nhãn đã khóa của nghiên cứu. Fixed
validation và test split không tham gia phân bổ labeled/unlabeled và test không
được dùng làm nguồn pseudo-label hoặc để tune. Membership Phase 2F hiện đã khóa;
không được lấy mẫu lại theo từng experiment.

---
