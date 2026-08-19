# ssl_detection_xray_v2

Semi-supervised object detection for anomaly detection on chest X-rays.

**Đề tài:** Nghiên cứu học bán giám sát cho dò tìm bất thường trên X-quang phổi.  
**Trọng tâm:** Semi-supervised object detection trên **VinBigData Chest X-ray**.  
**Framework chính:** [MMDetection](https://github.com/open-mmlab/mmdetection) (OpenMMLab). Detectron2 là *optional fallback*.

> **Trạng thái hiện tại: Phase 3A — Dataset Diagnostics Before Training: CLOSED / PASS**
>
> - Phase 3A — Dataset Diagnostics Before Training: **CLOSED / PASS**
> - Protocol: **1.0.0 / RESEARCHER_APPROVED_LOCKED**
> - Guardrail tests cuối: **53/53 PASS**
> - Full execution: **PASS**
> - Hard errors / warnings: **0 / 0**
> - DoD candidate: **TRUE**
> - Detailed diagnostics: **fixed train + labeled 1%/5%/10%/20%**
> - Validation/test usage: **STRUCTURAL_INTEGRITY_ONLY**
> - Hidden unlabeled GT diagnostics: **PROHIBITED / NOT USED**
> - Primary rare threshold: **image prevalence < 5% / LOCKED**
> - Rare classes theo definition đã khóa: **Atelectasis, Pneumothorax**
> - Image-support imbalance ratio max/min: **32.48**
> - BBox normalized-area Small/Medium/Large: **37.10% / 57.85% / 5.05%**
> - Phase 3A sensitivity analysis: **SECONDARY / NON-GATING**
> - Primary diagnostic thresholds retuned after diagnostics: **FALSE**
> - Phase 3A Ablation Study performed: **FALSE**
> - Phase 2F.1 — Seed Protocol: **CLOSED / PASS**
> - Partition seed: **42 / PRE_SPECIFIED_LOCKED_NO_SEED_SEARCH**
> - Training seed count: **10 / FIXED ORDERED LIST / LOCKED**
> - Deterministic policy: **CONTROLLED_BEST_EFFORT / LOCKED**
> - Supervised–SSL seed pairing: **SAME ORDERED LIST / PASS**
> - Phase 2F — Labeled/Unlabeled Construction: **CLOSED / PASS**
> - Labeled budgets: **1% / 5% / 10% / 20% = 34 / 171 / 343 / 685 ảnh**
> - Nested labeled/No Finding subsets: **PASS**
> - Deterministic reconstruction: **MATCH (4/4 budgets)**
> - Phase 2E — Fixed Train/Validation/Test Split: **CLOSED / PASS**
> - Split ratio: **70/15/15**
> - Train/validation/test images: **3,426 / 734 / 734**
> - Train/validation/test No Finding images: **350 / 75 / 75**
> - Image-level and annotation-level leakage: **0 / PASS**
> - Phase 2D.1 overall: **CLOSED / PASS**
> - Final JPEG quality: **95 / LOCKED**
> - Phase 2D — COCO Master Conversion & Validation: **CLOSED / PASS**
> - Phase 2C — Framework & Format Decision: **PASS**
> - Phase 2B — Canonical Detection Annotation Schema: **PASS**
> - Phase 2A — Data Standardization / Image-Boundary Validation: **PASS**
> - Phase 1D — Label Reliability & Kappa Feasibility: **PASS**
> - Phase 1C — Dataset Scope Decision: **PASS**
> - Phase 1B — Annotation Quality: **PASS**
> - Phase 1A — Dataset Overview: **PASS**
> - Phase 0 — Setup Environment: **CORE PASS**
> - Next implementation phase: **PHASE 4 — Supervised Baseline / NEXT / NOT STARTED**
>
> Phase 3A đã hoàn tất toàn bộ pre-training dataset diagnostics mà không mở
> hidden ground truth của unlabeled subsets và không dùng validation/test cho
> detailed design/tuning diagnostics. Dataset vẫn **training-ready về mặt kỹ thuật**,
> nhưng official training vẫn **chưa được phép** (`training_authorized=false`).



## Trạng thái Phase 3A

Phase 3A là **descriptive pre-training dataset diagnostics**, không phải model
experiment, hyperparameter search hoặc Ablation Study.

Protocol:

```text
phase: 3A
protocol_version: 1.0.0
protocol_status: RESEARCHER_APPROVED_LOCKED
protocol_sha256:
b03474cce0be773796f9d458e6273b8fd2b955c1b961bcccc4d955eb3bb0dcf5
```

Data-use firewall:

```text
Detailed diagnostics:
- train
- labeled_1pct
- labeled_5pct
- labeled_10pct
- labeled_20pct

Structural/integrity only:
- canonical
- validation
- test

Hidden unlabeled GT:
PROHIBITED

training_seed_used:
false
```

Primary rare-class definition:

```text
P_c_img = N_c_img / 3426
Rare nếu P_c_img < 0.05

N_c_img <= 171 => Rare
N_c_img >= 172 => Non-rare
```

`Rare` chỉ mang nghĩa dataset-level low support trong fixed train, không phải
clinical rarity.

Primary bbox-size definition theo normalized area:

```text
normalized_area = (w*h)/(W*H)

Small:  < 0.01
Medium: 0.01 <= area < 0.10
Large:  >= 0.10
```

Các nhóm này không phải standard COCO pixel-area Small/Medium/Large.

Kết quả execution cuối:

```text
Guardrail tests: 53/53 PASS
Full execution: PASS
hard_error_count: 0
warning_count: 0
HF01-HF35: PASS
dod_candidate: true
```

Kết quả fixed-train chính:

| Hạng mục | Kết quả |
|---|---:|
| Train images | 3,426 |
| Train bbox annotations | 25,260 |
| No Finding / zero-GT | 350 |
| Detection classes | 14 |
| Rare classes theo `<5%` | 2 / 14 |
| Rare classes | Atelectasis, Pneumothorax |
| Image-support imbalance ratio max/min | 32.48 |
| Small bbox share | 37.10% |
| Medium bbox share | 57.85% |
| Large bbox share | 5.05% |
| Mean label cardinality | 3.131 |
| BBox-center heatmap | 50 × 50 |

Labeled-budget diagnostics:

```text
1pct: 34 images; 3 No Finding; 14/14 class coverage
5pct: 171 images; 17 No Finding; 14/14 class coverage
10pct: 343 images; 35 No Finding; 14/14 class coverage
20pct: 685 images; 70 No Finding; 14/14 class coverage
```

Ở budget 1%, cả Atelectasis và Pneumothorax chỉ có 1 labeled image/class. Đây
là low-label support risk, không phải model-performance result.

Sensitivity analysis:

```text
Rare threshold:
1%  -> 0 rare classes
5%  -> 2 rare classes [PRIMARY]
10% -> 5 rare classes

Small boundary:
0.5% -> Small 23.52%
1.0% -> Small 37.10% [PRIMARY]
2.0% -> Small 58.90%

Large boundary:
5%  -> Large 21.78%
10% -> Large 5.05% [PRIMARY]
20% -> Large 0.95%
```

Sensitivity có vai trò:

```text
SECONDARY / NON-GATING
ONE_FACTOR_AT_A_TIME
primary_threshold_changed_after_diagnostics: false
sensitivity_used_for_threshold_selection: false
```

Source và evidence:

```text
configs/protocol/phase3A_dataset_diagnostics.yaml
scripts/03A_dataset_diagnostics.py
tests/test_phase3A_dataset_diagnostics_guardrails.py

reports/dataset_analysis_report.md
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

plots/dataset/class_distribution.png
plots/dataset/bbox_distribution.png
plots/dataset/bbox_location_heatmap.png
plots/dataset/negative_image_distribution.png
```

Final hashes:

```text
reports/dataset_analysis_report.md:
27e6ddfbd92ac2ba2a028e923f3be33f531a3a2a38f40468ccc5f38ef3823a2f

reports/03A_dataset_diagnostics_validation.json:
c24eff6907d92a23bbe9df39dfeeca9925b940fabbc087a5abd2e74c8732b892
```

Generated artifacts vẫn ghi `phase_status=OPEN_REVIEW_REQUIRED` vì script bị
cấm tự close phase. Researcher + GPT review sau execution đã đưa ra quyết định:

```text
Phase 3A: CLOSED / PASS
training_started: false
training_authorized: false
ablation_study_performed_in_phase3A: false
```

## Handoff sang Phase 4 — Supervised Baseline

Phase kế tiếp đã được xác định từ roadmap/checklist:

```text
Phase 4 — Supervised Baseline
Status: NEXT / NOT STARTED
```

Phase 4 sẽ xây dựng supervised baseline có kiểm soát để làm mốc so sánh cho
Phase 5 — SSL Detection.

Việc Phase 4 là phase kế tiếp **không đồng nghĩa training đã được authorize**.

Current gate:

```text
dataset_training_ready: true
phase3A_closed_pass: true

phase4_identified_as_next: true
phase4_started: false
training_started: false
training_authorized: false
```

Trước official supervised training, Phase 4 phải khóa và review tối thiểu:

```text
Google Colab / GPU training environment
MMDetection training configuration
runtime integration của ordered training-seed protocol
labeled-budget usage cho supervised baseline
augmentation policy
optimizer / scheduler / epoch policy
checkpoint-selection policy
validation-only model selection
logging / artifact / retry policy
per-seed result schema
guardrails chống test-set leakage
training authorization gate
```

Test set tiếp tục chỉ dành cho final evaluation theo protocol đã khóa; không
được dùng để chọn checkpoint, hyperparameter, threshold, augmentation hoặc
model configuration.

## Trạng thái Phase 2F.1

Phase 2F.1 không tạo split mới. Phase này tách rõ:

```text
partition_seed:
khóa train/validation/test và labeled/unlabeled membership.

training_seed:
kiểm soát model initialization, data-loader shuffling, augmentation,
sampler và các nguồn ngẫu nhiên của future training run.
```

Thuật ngữ chính thức là `partition_seed`; `split_seed` chỉ là legacy alias.
`partition_seed=42` không được thay đổi, tìm kiếm lại hoặc dùng chung khái niệm
với `training_seed`.

Ordered training-seed list đã khóa:

```text
1:  204886845
2:  1480646854
3:  1798418854
4:  2045683682
5:  1814859839
6:  1603952859
7:  1878351743
8:  875651179
9:  477581743
10: 869675675
```

Danh sách được sinh trước training, không gọi RNG và không dựa trên validation,
test hoặc kết quả thí nghiệm:

```text
namespace = ssl_detection_xray_v2|phase2F.1|training_seed
payload_i = namespace + |index=i
digest_i = SHA256(UTF-8(payload_i))
seed_i = 1 + (integer(first 8 hex characters of digest_i) mod (2^31 - 1))
```

Protocol chính:

```text
Supervised và SSL dùng cùng ordered seed list.
Pairing key: training_seed_index.
Retry chỉ dành cho technical failure và phải giữ nguyên seed.
Không silently replace run hoặc đổi seed vì kết quả kém.
Báo cáo per-seed, mean và sample SD với ddof=1.
Không bỏ seed, loại outlier hoặc chỉ báo cáo seed tốt nhất.
```

Kết quả kiểm định:

| Kiểm định | Kết quả |
|---|---:|
| Builder execute | 22/22 PASS; exit 0 |
| Independent pytest guardrails | 20/20 PASS; exit 0 |
| Read-only artifact validation | 23/23 PASS; exit 0 |
| SHA-256 seed derivation | MATCH 10/10 |
| Phase 2F membership checksums | MATCH 4/4 |

Source và evidence:

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

State tại closure:

```text
state: TEMPLATE_LOCKED_NO_RUNS
runs: []
training_started: false
training_authorized: false
```

Các generated artifact vẫn giữ
`phase_closure_status=PENDING_RESEARCHER_GPT_REVIEW` vì đó là snapshot trước
review. Quyết định `CLOSED / PASS` được xác lập sau đó bằng researcher/GPT
review; không sửa ngược artifact lịch sử. File
`data/manifests/seed_state_manifest.json` hiện là template của Phase 2F.1,
không phải schema seed-state lịch sử của Phase 0.

Snapshot môi trường local chỉ có vai trò
`LOCAL_DATA_AND_PROTOCOL_PROVENANCE`. Đây không phải environment lock cho
Google Colab. Môi trường Phase 4–5 phải được compatibility-test và snapshot
riêng trước official training.

Policy `CONTROLLED_BEST_EFFORT` không bảo đảm bitwise reproducibility giữa các
GPU, CUDA, cuDNN, driver, phần cứng hoặc phiên bản phần mềm khác nhau. Phase
2F.1 cũng không chứng minh 10 là số seed tối ưu, không thực hiện power analysis
và chưa đo variance hoặc training stability.

## Trạng thái Phase 2F

Phase 2F chỉ xây dựng labeled/unlabeled subsets từ 3.426 ảnh thuộc fixed train
split. Validation và test không tham gia phân bổ membership và không được dùng
làm nguồn pseudo-label. Tất cả budget sử dụng `partition_seed=42` theo chính
sách `PRE_SPECIFIED_LOCKED_NO_SEED_SEARCH`.

| Budget | Labeled | Unlabeled | No Finding trong labeled | Repair moves | Thời gian |
|---|---:|---:|---:|---:|---:|
| 1% | 34 | 3.392 | 3 | 3 | 7,063 giây |
| 5% | 171 | 3.255 | 17 | 7 | 130,437 giây |
| 10% | 343 | 3.083 | 35 | 10 | 253,485 giây |
| 20% | 685 | 2.741 | 70 | 26 | 1.255,344 giây |

```text
Protocol stage/version: 2F-C0-R11 / 2.0.0
Active repair policy: ONE_FOR_ONE_EXHAUSTIVE_DETERMINISTIC
Nested relation: 1pct subset_of 5pct subset_of 10pct subset_of 20pct
Nested labeled subsets: PASS
Nested No Finding subsets: PASS
Class coverage: 14/14 ở mọi budget
Labeled/unlabeled disjoint and complete: PASS
Validation/test isolation: PASS
Independent readback: PASS
Repair moves total: 46
Total construction elapsed: 1646.329 seconds
Timing role: OBSERVABILITY_ONLY_NOT_SELECTION_CRITERION
Phase 2F: CLOSED / PASS
training_authorized: false
```

Objective repair duyệt exhaustively admissible one-for-one swap neighborhood và
dừng tại **one-for-one local optimum** cho từng budget. Kết quả này không phải
bằng chứng về global optimum; report ghi rõ `global_optimum_claimed=false`.
Legacy two-for-two engine không được gọi trong active construction.

SHA-256 của labeled `image_id` membership đã khóa:

```text
1pct:  c54e7d61e84b7cfce68c04a795783b5c5d01331d2aa9c754fb1ae1dbae4ba071
5pct:  c4db3b5f7a5b0f391ad883ef665c3d341af3ef6afb3fc553e71f1c4ef17ee50b
10pct: fc008d31505227544087ba474613075a8cd077587df9d6b13146b378f5a3a7d6
20pct: 6f4aaba6be147983d56007c49ada234dfcabe7ec2f936f28f8cd240999b8417e
```

Deterministic reconstruction đã so sánh lại bảy nhóm trường cho từng budget:
labeled membership checksum, labeled/unlabeled COCO checksum, labeled size,
No Finding size, repair move count và integer objective cuối. Tất cả trường ở
cả bốn budget đều khớp với official lock manifest (`MATCH`). Runtime timing
không được so sánh vì chỉ phục vụ observability và tự nhiên có thể thay đổi giữa
các lần chạy.

Artifact chính thức của Phase 2F:

```text
scripts/02F_build_labeled_unlabeled.py
configs/protocol/phase2F_labeled_unlabeled.yaml
tests/test_phase2F_labeled_unlabeled_guardrails.py

data/processed/coco/labeled_splits/instances_labeled_{1pct,5pct,10pct,20pct}.json
data/processed/coco/unlabeled_splits/instances_unlabeled_{1pct,5pct,10pct,20pct}.json
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

## Trạng thái Phase 2E

| Hạng mục | Train | Validation | Test |
|---|---:|---:|---:|
| Ảnh | 3,426 | 734 | 734 |
| Annotation | 25,260 | 5,399 | 5,437 |
| No Finding / zero-GT | 350 | 75 | 75 |
| Tỷ lệ No Finding | 10.216% | 10.218% | 10.218% |
| Categories | 14 | 14 | 14 |

```text
Split unit: image_id
Split ratio: 70/15/15
Image union: 4,894/4,894
Annotation union: 36,096/36,096
Train-validation overlap: 0
Train-test overlap: 0
Validation-test overlap: 0
Fixed split created: true
Fixed split validated: true
Phase 2E: CLOSED / PASS
training_authorized: false
```

SHA-256 của danh sách `image_id` đã khóa:

```text
train: 628b9bb8ba25129a928abe994b101b4c4efd5588d389feb60da6de2a371fa11a
val:   87c23ebed4d1e6965731fc0b31245859f49e777119813c6152efde3531ba58c6
test:  1f7903e069e872bf2e5fe13bb4d0fa257dc4a1c2c8290a621d3f7286ada66b37
```

SHA-256 của ba COCO JSON chính thức:

```text
instances_train.json: 0f3c37a6f1b5bcc6971b01fd4c69c7a2a3a1e8bee145c11488c7a658dcbbebe3
instances_val.json:   33064f47ba690be13e9d418d8ec5d4a6a2bec8482c24ea3eadf83fb33d01762a
instances_test.json:  e1a73110e92af2656276d6c532035afe474b6ac6f2b2f03849834c036e1c00a4
```

Test set chứa 75 ảnh No Finding, cho phép tính
`FP per negative image = tổng FP trên ảnh No Finding / 75`. Test set được khóa
và chỉ dùng cho đánh giá cuối; không được dùng để chọn checkpoint, mô hình,
hyperparameter, threshold, augmentation hoặc pseudo-label filtering policy.

Kết quả leakage bằng 0 chỉ được khẳng định ở cấp `image_id` và annotation.
Patient-level leakage là **NOT ASSESSABLE** vì `PatientID` và các định danh DICOM
có thể dùng để liên kết ảnh theo bệnh nhân/study không còn khả dụng sau quy trình
de-identification của VinDr-CXR. Đây là giới hạn của dữ liệu công bố, không phải
bằng chứng rằng patient-level leakage bằng 0.

Artifact chính thức của Phase 2E:

```text
scripts/02E_build_fixed_split.py
02E_build_fixed_split_output.txt

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

## Trạng thái Phase 2D.1

| Hạng mục | Trạng thái |
|---|---|
| Phase 2D.1A — Image Representation Protocol Decision | CLOSED / PASS |
| Phase 2D.1B-Pilot — Representative DICOM-to-JPG Pilot | CLOSED / PASS |
| Final JPEG quality | 95 / LOCKED |
| Phase 2D.1B-Full — Full Controlled-Scope Conversion | CLOSED / PASS |
| Full conversion completed | TRUE |
| Full validation | PASS |
| Output promotion | PASS |
| Backup cleanup | PASS |
| Final output integrity | PASS |
| Phase 2D.1C — MMDetection Dataset / Empty-Image Loading Validation | CLOSED / PASS |
| Phase 2D.1D — Evidence Consolidation, GPT Review & Closure | CLOSED / PASS |
| Phase 2D.1 overall | CLOSED / PASS |
| JPG training representation ready | TRUE |
| COCO-JPG training annotation ready | TRUE |
| MMDetection dataset loading ready | TRUE |
| Empty-image retention ready | TRUE |
| Dataset training-ready | TRUE |
| Training authorized | FALSE |
| Phase 3A pre-training diagnostics | CLOSED / PASS |

## Giải thích ngắn gọn các phase

| Phase | Mục đích | Tại sao phải có | Ảnh hưởng đến phase nào |
|---|---|---|---|
| Phase 0 — Setup Environment | Chuẩn hóa repo, môi trường và quy tắc tái lập. | Tránh sai khác môi trường và thiếu dấu vết thực thi. | Tất cả phase sau. |
| Phase 1A — Dataset Overview | Thống kê quy mô, lớp, bbox và No Finding của dữ liệu gốc. | Cần biết dữ liệu thực tế trước khi xác định phạm vi nghiên cứu. | Phase 1B, 1C và các phase dữ liệu. |
| Phase 1B — Annotation Quality | Kiểm tra nhãn, bbox và các trường hợp bất thường trong annotation. | Nhãn lỗi sẽ làm sai schema, split, training và đánh giá. | Phase 1C, 1D, 2B và downstream experiments. |
| Phase 1C — Dataset Scope Decision | Khóa controlled scope 4,894 ảnh. | Bảo đảm mọi phase dùng cùng một tập ảnh có thể truy vết. | Phase 1D và toàn bộ Phase 2–7. |
| Phase 1D — Label Reliability & Kappa Feasibility | Đánh giá mức độ đồng thuận nhãn và giới hạn reliability. | Cần diễn giải đúng độ tin cậy của ground truth, không xem nhãn là tuyệt đối. | Phase 3, 4, 5, 6 và phần thảo luận Phase 7. |
| Phase 2A — Data Standardization / Image-Boundary Validation | Xác nhận DICOM khả dụng, kích thước ảnh và bbox nằm trong biên. | Ngăn lỗi geometry trước khi chuẩn hóa annotation và tạo representation. | Phase 2B, 2D và 2D.1. |
| Phase 2B — Canonical Detection Annotation Schema | Tạo bảng ảnh, bbox và class mapping chuẩn duy nhất. | Cần một nguồn annotation nhất quán cho COCO và mọi thí nghiệm. | Phase 2C, 2D, 2E và toàn bộ training/evaluation. |
| Phase 2C — Framework & Format Decision | Khóa MMDetection và COCO Detection JSON cùng kế hoạch chuyển đổi. | Framework và format quyết định cấu trúc dữ liệu, loader và config downstream. | Phase 2D, 2D.1C, 4 và 5. |
| Phase 2D — COCO Master Conversion & Validation | Tạo và kiểm định `coco_master.json`. | Chuyển canonical schema thành annotation chuẩn mà framework có thể sử dụng. | Phase 2D.1, 2E, 3, 4 và 5. |
| Phase 2D.1A — Image Representation Protocol Decision | Khóa quy tắc DICOM-to-JPG dựa trên metadata và bảo toàn geometry. | Ngăn preprocessing tùy ý làm thay đổi intensity, kích thước hoặc bbox. | Phase 2D.1B-Pilot và 2D.1B-Full. |
| Phase 2D.1B-Pilot — Representative Pilot | So sánh Q95/Q100 trên mẫu đại diện và khóa JPEG quality 95. | Cần bằng chứng fidelity trước khi chuyển đổi toàn bộ dữ liệu. | Phase 2D.1B-Full. |
| Phase 2D.1B-Full — Full Conversion & Validation | Chuyển đủ 4,894 DICOM sang JPG Q95 và tạo COCO-JPG derivative. | Cung cấp representation thống nhất, đã kiểm định geometry và traceability. | Phase 2D.1C, 2D.1D và các phase dùng ảnh JPG. |
| Phase 2D.1C — MMDetection Dataset Loading Validation | Kiểm tra MMDetection nạp đúng ảnh, bbox, label và 500 empty-GT images. | File hợp lệ chưa đủ; cần chứng minh pipeline dataset/dataloader hoạt động trong controlled scope. | Phase 2D.1D, 2E và technical training readiness. |
| Phase 2D.1D — Evidence Consolidation, GPT Review & Closure | Đối chiếu bằng chứng, sửa tài liệu và quyết định đóng Phase 2D.1. | Ngăn trạng thái mâu thuẫn hoặc kết luận vượt quá bằng chứng trước khi chuyển phase. | Phase 2E, 2F, 2F.1 và quy trình xem xét training authorization. |
| Phase 2E — Fixed Train/Validation/Test Split | Tạo split cố định, disjoint và kiểm tra leakage. | Nếu split không khóa, so sánh mô hình không công bằng và test có thể bị rò rỉ. | Phase 2F, 3, 4, 5 và 6. |
| Phase 2F — Labeled/Unlabeled Split for SSL | Khóa tập labeled/unlabeled và các labeled fractions lồng nhau. | SSL cần biết chính xác mẫu nào có nhãn được phép dùng ở từng mức. | Phase 3, 4, 5 và 6. |
| Phase 2F.1 — Seed Protocol | Tách `partition_seed` đã khóa khỏi ordered 10-`training_seed` protocol. | Giữ membership bất biến, cho phép đo biến thiên huấn luyện và paired comparison công bằng. | Phase 3, 4, 5 và 6. |
| Phase 3A — Dataset Diagnostics Before Training | Phân tích class support, imbalance, bbox geometry/size/location, negatives, labeled budgets, multilabel structure và sensitivity trước training. | Cần mô tả các pre-training dataset risks mà không dùng test/hidden-U GT để tune và không thay đổi protocol sau khi nhìn kết quả. | Phase 4, 5, 6 và phần Methodology/Results. |
| Phase 4 — Supervised Baseline | Xây dựng mốc supervised có kiểm soát. | Cần baseline để xác định SSL có cải thiện thực sự hay không. | Phase 5, 6 và 7. |
| Phase 5 — SSL Detection | Huấn luyện và đánh giá teacher–student pseudo-labeling. | Đây là thí nghiệm chính trả lời câu hỏi nghiên cứu bán giám sát. | Phase 6 và 7. |
| Phase 6 — Threshold Sweep & Error Analysis | Phân tích threshold, lỗi FP/FN và độ nhạy kết quả theo đúng split. | Metric tổng hợp không giải thích mô hình sai ở đâu hoặc nhạy với quyết định nào. | Phase 7. |
| Phase 7 — Thesis/Paper Synthesis | Tổng hợp phương pháp, kết quả, giới hạn và kết luận. | Chuyển evidence đã khóa thành báo cáo nghiên cứu có thể kiểm tra. | Luận văn, bài báo và báo cáo bảo vệ. |

> Quan hệ giữa các phase thể hiện dependency chính, không có nghĩa một phase
> đã tự động được phép chạy ngay khi phase trước PASS. Training chỉ được bắt đầu
> sau khi các gate bắt buộc và `training_authorized=true` được phê duyệt rõ ràng.

### Commit và notebook tái lập Phase 2D.1C

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

### Kết quả representative pilot

```text
Controlled DICOM paths resolved: 4,894/4,894
DICOM header inventory: 4,894/4,894

Pilot images selected: 64
No Finding pilot images: 16

Metadata/features coverage: 54/54
Abnormal class coverage: 14/14

Pixel decode success: 64/64
Pixel decode errors: 0

Geometry preservation: PASS
BBox invariance: PASS
Critical visual failure: false

Final JPEG quality: 95
Full conversion authorized: true
```

### Kết quả full controlled-scope conversion

```text
Full-scope images processed: 4,894
Output JPG files: 4,894
Conversion errors: 0

Native decoder images: 2,776
pylibjpeg JPEG 2000 decoder images: 2,118

VOI/windowing branch images: 4,536
Theoretical fallback branch images: 358
Presentation-polarity inversions: 1,562
Pixel-padding processing required: 0

COCO images: 4,894
COCO annotations: 36,096
COCO categories: 14

Abnormal images with bbox: 4,394
No Finding images: 500
No Finding annotations: 0

Geometry validation: PASS
BBox boundary validation: PASS
Category mapping validation: PASS
No Finding validation: PASS
Promotion: PASS
Cleanup: PASS
Final output integrity: PASS
Missing JPG referenced by COCO: 0
```

### Kết quả MMDetection dataset / empty-image loading validation

```text
Phase 2D.1C full pipeline audit: PASS
Images audited: 4,894/4,894
Abnormal images audited: 4,394/4,394
Zero-GT images audited: 500/500

BBox/label validation: PASS
Errors: 0
Regression/unit tests: 35 passed
Dataloader workers validated: num_workers=0

filter_empty_gt=False: retained 4,894/4,894 images
filter_empty_gt=True: excluded exactly 500 zero-GT images
Standard empty-GT dataloader batches: PASS
Forced empty-GT dataloader batches: PASS

Dataset training-ready: TRUE
Training authorized: FALSE
```

Kết quả này chứng minh COCO-JPG derivative có thể được MMDetection nạp đúng, toàn bộ
500 ảnh No Finding được giữ lại khi dùng cấu hình chính thức
`filter_empty_gt=False`, và empty-GT samples đi qua dataloader mà không gây lỗi.
Kết quả không cấp quyền bắt đầu training và không thay thế các gate còn lại của
experimental protocol.

Phase 2D.1C không kiểm định multi-worker loading. Không được diễn giải kết quả
`num_workers=0` thành bằng chứng cho cấu hình nhiều worker.

### Trạng thái Phase 2D.1D

Phase 2D.1D chỉ thực hiện evidence consolidation, consistency review,
documentation correction và closure decision. Phase này không tạo split và
không chạy training.

```text
Technical evidence inventory: COMPLETED
Documentation consistency review: COMPLETED
Final closure decision: PASS
Phase 2D.1D: CLOSED / PASS
Phase 2D.1 overall: CLOSED / PASS
dataset_training_ready: true
training_authorized: false
```

Evidence review đã xác nhận:

```text
Full conversion and audits: PASS
COCO-JPG path-only derivative: PASS
MMDetection full-scope loading: PASS
Empty-GT retention: PASS
Full bbox/label pipeline audit: 4,894/4,894 PASS
```

So sánh trực tiếp hai COCO JSON cho thấy image IDs và dimensions giống nhau,
annotations và categories giống hoàn toàn; chỉ `images[].file_name` chuyển từ
DICOM path sang `train/<image_id>.jpg`.

Q100 có numerical fidelity cao hơn Q95 trên toàn bộ 64 whole-image comparisons
và 402 bbox-ROI comparisons. Q95 được khóa như một
fidelity–storage/I/O trade-off, không phải bằng chứng detector superiority.

Trình tự phase downstream:

```text
Phase 2E: Fixed Train/Validation/Test Split — CLOSED / PASS
Phase 2F: Labeled/Unlabeled Split for SSL — CLOSED / PASS
Phase 2F.1: Seed Protocol — partition_seed versus training_seed — CLOSED / PASS
Phase 3A: Dataset Diagnostics Before Training — CLOSED / PASS
Phase 4: Supervised Baseline — NEXT / NOT STARTED
```

Việc đóng Phase 2D.1D không tự động cấp quyền training.

### Quyết định JPEG quality

JPEG quality 100 có numerical fidelity cao hơn quality 95 trên whole-image và bbox-ROI metrics.

JPEG quality 95 được chọn vì:

- vẫn giữ whole-image và bbox-ROI fidelity cao;
- geometry và bbox invariance đều PASS;
- không phát hiện critical visual failure trong representative pilot;
- giảm projected storage khoảng **48.79%** so với quality 100;
- projected storage cho 4,894 ảnh khoảng **7.38 GiB**, so với khoảng **14.41 GiB** ở quality 100.

Quyết định này là:

```text
fidelity–storage/I/O trade-off decision
```

Quyết định này không chứng minh:

```text
JPEG quality 95 có detector performance tốt hơn quality 100.
JPG tương đương lâm sàng với DICOM gốc.
Mọi đặc trưng chẩn đoán đều được bảo toàn tuyệt đối.
Pipeline đạt full DICOM-standard conformance.
Training đã được phép.
```

## Chú ý quan trọng

- DICOM tiếp tục là **immutable raw medical source**.
- Biểu diễn JPG được xây dựng bằng **DICOM metadata-aware, standard-aligned reference representation pipeline**.
- Đây là pipeline biểu diễn tham chiếu, không được mô tả là phương pháp mới hoặc thuật toán mới.
- JPG quality 95 là processed training representation đã được khóa và kiểm chứng cho toàn bộ controlled scope.
- Toàn bộ 4,894 ảnh đã được chuyển đổi thống nhất với `quality=95`.
- Không sử dụng mixed JPEG quality giữa các ảnh hoặc subset.
- Không resize, crop, rotate, flip hoặc transpose trong DICOM-to-JPG conversion.
- Geometry ảnh được bảo toàn; không thực hiện bbox scaling.
- Không tự động clamp, xóa, hợp nhất hoặc sửa canonical bbox.
- `coco_master.json` tiếp tục là annotation master chính thức.
- `coco_master_jpg.json` là path-only training derivative; không thay thế annotation master.
- `coco_master_jpg.json` đã được tạo, validate và promotion thành công.
- `data/processed/images_jpg/train/` đã có đủ 4,894 ảnh JPG.
- 36,096 bbox đều có width, height và area dương, nằm trong biên ảnh.
- 500 ảnh No Finding đều có `annotation_count=0`.
- Không có ảnh No Finding nào giao với tập ảnh có bbox.
- MMDetection loading và chính sách giữ empty-GT image đã được validate bằng full pipeline audit.
- `filter_empty_gt=False` là cấu hình bắt buộc để giữ đủ 500 ảnh No Finding.
- `filter_empty_gt=True` loại đúng 500 ảnh zero-GT và không được dùng cho protocol chính thức.
- `jpg_training_representation_ready=true` chỉ xác nhận output biểu diễn JPG.
- `coco_jpg_training_annotation_ready=true` chỉ xác nhận COCO JPG derivative đã được tạo và validate.
- `dataset_training_ready=true` còn bao gồm bằng chứng MMDetection loading và empty-image retention từ Phase 2D.1C.
- `dataset_training_ready=true` không đồng nghĩa `training_authorized=true`.
- Fixed train/validation/test split của Phase 2E đã được tạo, checksum-lock và
  independently validated; không được tái chia theo từng experiment.
- Mọi experiment downstream phải dùng cùng `instances_val.json` và
  `instances_test.json`; việc tuân thủ phải được xác minh từ config và log của
  từng lần chạy.
- Chỉ `instances_train.json` được dùng để xây dựng labeled/unlabeled subsets;
  validation và test không được tham gia phân bổ hoặc làm nguồn pseudo-label.
- Labeled membership của Phase 2F đã được checksum-lock và xác minh bằng
  deterministic reconstruction; không được lấy mẫu lại theo từng experiment.
- Bốn labeled subsets và các No Finding subsets tương ứng có quan hệ lồng nhau
  `1pct ⊆ 5pct ⊆ 10pct ⊆ 20pct`; unlabeled set ở mỗi budget là phần bù chính
  xác trong fixed train split.
- Unlabeled COCO JSON không chứa annotations hoặc trường dẫn xuất từ ground
  truth; audit không phát hiện GT exposure violation.
- Kết luận local optimum chỉ áp dụng cho admissible one-for-one neighborhood;
  không được diễn giải thành global optimum hoặc tối ưu trên mọi neighborhood.
- Timing Phase 2F chỉ dùng để quan sát runtime, không tham gia seed, membership,
  objective, acceptance, tie-break, checksum hay reconstruct comparison.
- Phase 2F.1 đã khóa seed protocol nhưng không triển khai runtime seeding cho
  Python/NumPy/PyTorch/CUDA/DataLoader/sampler/augmentation; việc tích hợp và
  kiểm chứng thực tế thuộc Phase 4–5.
- Supervised và SSL phải dùng cùng ordered 10-seed list, ghép theo
  `training_seed_index`; không được dùng một seed duy nhất, bỏ seed, chọn seed
  tốt nhất hoặc đổi seed vì kết quả kém.
- Mỗi experiment phải báo cáo kết quả từng seed, mean và sample SD với
  `ddof=1`; technical-failure retry phải giữ nguyên seed và lưu đầy đủ attempt.
- Snapshot môi trường local của Phase 2F.1 chỉ là provenance evidence; môi
  trường Google Colab cho Phase 4–5 chưa được khóa và phải được snapshot riêng.
- Phase 3A đã hoàn tất detailed diagnostics trên fixed train và legitimately
  labeled subsets; validation/test chỉ được dùng ở structural-integrity scope.
- Phase 3A không dùng hidden ground truth của unlabeled subsets và không được
  dùng để đổi primary diagnostic thresholds sau khi nhìn kết quả.
- Rare classes trong Phase 3A là dataset-level operational low-support classes,
  không được diễn giải thành clinically rare diseases.
- Các bbox size categories của Phase 3A dựa trên normalized area và không phải
  COCO standard Small/Medium/Large.
- Phase 3A sensitivity analysis là SECONDARY/NON-GATING và không phải Ablation Study.
- Phase 3A không tạo training artifact, checkpoint hoặc pseudo-label.
- Training tiếp tục bị khóa sau khi Phase 3A đóng và chỉ được xem xét sau khi
  training configuration, runtime seed integration, Colab environment và các
  authorization gate tiếp theo đã được review.
- Không chạy lại `--execute-full` nếu không có lý do kỹ thuật được ghi nhận và phê duyệt.
- Không commit 4,894 JPG files vào ordinary Git.
- Không thay đổi hoặc tái tạo fixed train/validation/test split đã khóa nếu
  không có change-control và lý do kỹ thuật được ghi nhận.
- Không tái tạo hoặc thay đổi labeled/unlabeled membership đã khóa nếu không có
  change-control và lý do kỹ thuật được ghi nhận.
- Không bắt đầu training khi `training_authorized=false`.

## Current project gate

```text
Current completed phase:
Phase 3A — Dataset Diagnostics Before Training: CLOSED / PASS

Dataset technical training readiness:
TRUE

Training started:
FALSE

Training authorized:
FALSE

Pseudo-label generated:
FALSE

Next implementation phase:
PHASE 4 — Supervised Baseline

Phase 4 status:
NEXT / NOT STARTED
```

Phase 3A closure does not authorize training. The next implementation phase is
**Phase 4 — Supervised Baseline**, but Phase 4 has not started yet. Before
official supervised training, the Phase 4 training/Colab environment, runtime
seed integration, model/configuration, checkpoint/evaluation policy and
experiment-specific authorization gates must be explicitly defined, reviewed
and approved.


## Vai trò trong dự án

| Vai trò | Trách nhiệm |
|---|---|
| Người nghiên cứu | Quyết định hướng nghiên cứu, protocol và phạm vi thí nghiệm |
| GPT | Thiết kế quy trình, phản biện logic và review evidence |
| Claude | Viết code trong repo theo prompt đã được kiểm soát |
| Python | Chạy script, kiểm tra dữ liệu và tạo evidence |

Quy trình bắt buộc:

```text
script → output → DoD → GPT review → người nghiên cứu tick checklist
```

## Cấu trúc repo

Xem:

- [`STRUCTURE.md`](STRUCTURE.md)
- [`repository_structure.md`](repository_structure.md)
- [`PROJECT_CONTEXT.md`](PROJECT_CONTEXT.md)
- [`PHASE_HANDOFF.md`](PHASE_HANDOFF.md)
- [`research_log.md`](research_log.md)

## Dữ liệu và annotation master

Controlled working scope:

```text
Images: 4,894
Abnormal images: 4,394
No Finding images: 500
Abnormal bbox annotations: 36,096
Abnormal detection classes: 14
```

COCO annotation master chính thức:

```text
data/processed/coco/coco_master.json
```

COCO master đã được validate:

```text
Images: 4,894
Annotations: 36,096
Categories: 14
No Finding annotations: 0
Invalid annotations: 0
pycocotools load: PASS
```

COCO JPG training derivative đã được tạo và validate:

```text
data/processed/coco/coco_master_jpg.json
```

Final processed image representation:

```text
data/processed/images_jpg/train/<image_id>.jpg
```

Final output integrity:

```text
JPG files: 4,894
COCO images: 4,894
COCO annotations: 36,096
COCO categories: 14
Missing JPG files referenced by COCO: 0
```

Artifact roles:

| Artifact | Vai trò |
|---|---|
| DICOM | Immutable raw medical source |
| JPG quality 95 | Processed training representation |
| `coco_master.json` | Official annotation master |
| `coco_master_jpg.json` | Path-only JPG training derivative |

## DICOM-to-JPG protocol

Protocol chính thức:

```text
configs/protocol/phase2D1_jpg_representation.yaml
```

Protocol version:

```text
1.0.0
```

Transformation order:

```text
DICOM decode
→ pixel-padding mask
→ modality transformation
→ VOI LUT/windowing
→ presentation-polarity normalization
→ deterministic uint8 conversion
→ JPEG encoding
```

Final encoding policy:

```text
JPEG quality: 95
Storage mode: grayscale L
Resize: false
Crop: false
Rotation: false
Flip: false
Transpose: false
BBox scaling: false
```

Full conversion status:

```text
Phase 2D.1B-Full: CLOSED / PASS
Full conversion completed: true
Validation passed: true
Promotion passed: true
Cleanup passed: true
Final output integrity passed: true
```

Primary implementation and evidence:

```text
scripts/02D1B_full_dicom_to_jpg.py
tests/test_phase2D1B_full_guardrails.py
reports/phase2D1B_full_preflight.json
reports/phase2D1B_full_validation.json
reports/phase2D1B_full_promotion.json
reports/phase2D1B_full_cleanup_audit.json
reports/phase2D1B_full_metadata_audit.csv
reports/phase2D1B_full_bbox_audit.csv
reports/phase2D1B_full_no_finding_audit.csv
reports/phase2D1B_full_errors.csv
```

## MMDetection dataset-loading validation

Validation config:

```text
configs/validation/phase2D1C_mmdet_dataset_loading.py
```

Implementation và regression tests:

```text
scripts/02D1C_validate_mmdet_dataset_loading.py
tests/test_phase2D1C_mmdet_dataset_loading_guardrails.py
```

Evidence:

```text
reports/phase2D1C_mmdet_dataset_errors.csv
reports/phase2D1C_mmdet_dataset_image_audit.csv
reports/phase2D1C_mmdet_dataset_loading_report.json
reports/phase2D1C_mmdet_dataset_loading_report.md
```

SHA-256 của evidence đã khóa:

```text
0780595f5ff69c36329f05d69f7bb353fd095f32a0df3f76b16f039143a5f2cf  reports/phase2D1C_mmdet_dataset_errors.csv
00df8ed311e6de0ba863fa8e5a90551d34ef080b12cdd0063b6397fdfd76e474  reports/phase2D1C_mmdet_dataset_image_audit.csv
dabb3dbf27373c5271cdb3137406b583a9d3b7ee607ca2faabe18033ab772ca8  reports/phase2D1C_mmdet_dataset_loading_report.json
fb0170cadee8b7b66d81be4681af0b8955ba3c4e6b584fb5faf35d8e054b9ce9  reports/phase2D1C_mmdet_dataset_loading_report.md
```

Locked loading policy:

```text
filter_empty_gt: false
dataset_training_ready: true
training_authorized: false
```

## Giao thức đánh giá

Xem:

```text
configs/protocol/checkpoint_policy.yaml
```

Tóm tắt:

```text
Primary metric: mAP@0.5:0.95
Checkpoint selection split: validation
Test usage: final evaluation only
```

Test set không được dùng để:

- tune threshold;
- chọn checkpoint;
- chọn model hoặc backbone;
- quyết định augmentation;
- lựa chọn pseudo-label filtering policy.

## Hướng dẫn cho Claude

Xem [`CLAUDE.md`](CLAUDE.md).

Claude không được tự thay đổi:

- protocol nghiên cứu;
- final JPEG quality;
- controlled dataset scope;
- No Finding policy;
- bbox semantics;
- category mapping;
- split policy;
- metric policy;
- training authorization.

Mọi thay đổi phải tuân theo:

```text
script → output → DoD → GPT review → researcher decision
```

## Ghi chú bổ sung — Annotation QA và kiểm chứng sau chuyển đổi

> **Supporting note only:** ghi chú này chỉ tổng hợp và làm rõ evidence đã có;
> không mở lại, phân loại lại hoặc thay đổi trạng thái của bất kỳ phase, gate,
> authorization hay quyết định đã khóa nào ở trên.

- Tính hợp lệ của bounding box đã được kiểm tra trên dữ liệu annotation, bao gồm
  điều kiện tọa độ hợp lệ và kiểm tra biên theo kích thước ảnh trong controlled
  scope. Không phát hiện bounding box không hợp lệ, vì vậy không phát sinh
  trường hợp cần loại bỏ vì lỗi validity/boundary.
- Phase 1B đã thực hiện phát hiện annotation trùng lặp trong cùng
  `(image_id, class)`: **exact duplicate = 0**; đồng thời phát hiện
  **147 near-duplicate bbox records** tại ngưỡng **IoU >= 0.95**, thuộc
  **71 `(image, class)` groups**. Đây là các *near-duplicate candidates*, không
  phải 147 cặp duplicate và không mặc nhiên được xem là annotation lỗi.
- Các near-duplicate candidates được giữ lại có chủ đích do đặc tính annotation
  multi-radiologist; không thực hiện xóa, hợp nhất hoặc sửa bbox một cách tự động.
- Sau chuyển đổi, kiểm tra tự động trên **4,894 ảnh** và **36,096 bounding box**
  xác nhận tính nhất quán về kích thước, geometry và hệ tọa độ annotation trong
  controlled scope.
- Kiểm tra trực quan sau chuyển đổi trên **16 mẫu representative/stress**
  (bao phủ **14/14 lớp** và **4 zero-GT samples**) không phát hiện sai lệch
  không gian rõ rệt của bounding box.
- Sai khác do mã hóa **JPEG Q95** đã được định lượng so với biểu diễn lossless
  trước JPEG trong representative pilot: **64 whole-image comparisons** và
  **402 bbox-ROI comparisons**. Evidence fidelity này thuộc phạm vi pilot,
  không phải full-dataset pixel-fidelity audit trên 4,894 ảnh.
- Các kiểm chứng trên hỗ trợ kết luận về **geometry/coordinate consistency** và
  fidelity của representation trong phạm vi đã đánh giá. Không được diễn giải
  chúng thành bằng chứng rằng JPEG Q95 giống hệt DICOM ở mức pixel, bảo toàn
  tuyệt đối mọi thông tin lâm sàng, hoặc chứng minh clinical equivalence với
  DICOM gốc.
