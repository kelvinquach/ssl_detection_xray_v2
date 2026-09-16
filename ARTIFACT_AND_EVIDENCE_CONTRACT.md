# ARTIFACT_AND_EVIDENCE_CONTRACT
## Semi-supervised Object Detection cho phát hiện bất thường trên X-quang ngực

**Artifact type:** Implementation / Evidence Governance Contract  
**Status:** `RESEARCHER_APPROVED / LOCKED`  
**Applies to:** Vast.ai pilot/preflight, official SUP, official SSL, ablation, final evaluation, statistical analysis  
**Scientific source of truth:** `sn-article.tex`  
**Implementation contract:** `IMPLEMENTATION_HANDOFF.md`
**Artifact contract version:** `1.1.0`
**Schema version:** `1.1`  
**Scientific source SHA-256 at lock:** `a5f63ff018ee28a37ad2daa3fff337daf6b30faf34d9f12c6ff5c0fa92fe5237`  
**Implementation handoff SHA-256 at lock:** `d70e556d45f4a54a5e0498842b4932deef7991de0d986095ec85e546dcc2d026`  
**Final cross-check:** `PASS — 88/88 audited criteria`  
**Controlled revision note (2026-09-11):** Researcher-approved scheduler clarification synchronizes the rule `LR drop 1 = 2/3 × U_opt` and `LR drop 2 = 11/12 × U_opt`; 100%-SUP (`U_opt=10284`) therefore uses milestones `[6856,9427]`. Existing `at lock` SHA-256 values below refer to the previous governance lock and must be recomputed when the updated canonical `sn-article.tex`, implementation handoff, and artifact contract are re-locked.
**Priority rule:** Nếu tài liệu này xung đột với Methodology trong `sn-article.tex`, `sn-article.tex` thắng. Nếu implementation không đáp ứng được scientific protocol, phải dừng và xử lý như controlled revision; không được âm thầm sửa thiết kế nghiên cứu.

---

# 1. MỤC ĐÍCH

Tài liệu này định nghĩa **artifact contract** và **evidence contract** cho toàn bộ pipeline huấn luyện và đánh giá SUP/SSL.

Mục tiêu là bảo đảm sau mỗi pilot hoặc official run có thể truy ngược đầy đủ:

```text
scientific protocol
→ frozen executable config
→ data membership
→ seed
→ runtime environment
→ training events
→ validation history
→ checkpoint selection
→ prediction/evaluation
→ per-run result
→ SUP–SSL pairing
→ RQ1–RQ7 analysis
→ thesis table/figure
```

Tài liệu này **không thay đổi bất kỳ giá trị khoa học nào** trong Methodology. Nó chỉ quy định:

- artifact nào phải sinh;
- artifact nằm ở đâu;
- schema tối thiểu;
- artifact nào là bắt buộc;
- artifact nào được phép xóa sau verification;
- cách phân biệt pilot / official / ablation;
- cách phân biệt validation / final test;
- cách giữ checkpoint;
- cách tạo provenance và checksum;
- cách ánh xạ output sang RQ1–RQ7 và luận văn.

---

# 2. NGUYÊN TẮC GOVERNANCE BẮT BUỘC

## 2.1. Source hierarchy

```text
sn-article.tex
    ↓ scientific source of truth

IMPLEMENTATION_HANDOFF.md
    ↓ implementation contract

ARTIFACT_AND_EVIDENCE_CONTRACT.md
    ↓ artifact/evidence operational contract

code/config/runtime
```

Nếu có xung đột:

```text
Methodology wins
→ implementation/artifact contract phải sửa
→ không sửa Methodology để hợp code
```

---

## 2.2. Evidence-before-claim

Không được ghi một scientific/runtime claim là `PASS` nếu chưa có artifact chứng minh.

Cấm:

```text
ASSUMED PASS
EXPECTED TO WORK
NOT TESTED BUT IMPLEMENTED
PASS WITH UNCHECKED PATH
```

Trạng thái hợp lệ cho từng check:

```text
PASS
FAIL
NOT_APPLICABLE
```

`NOT_APPLICABLE` chỉ được dùng khi scientific/implementation contract thật sự không yêu cầu path đó và phải có rationale.

---

## 2.3. Pilot không bao giờ trở thành official

Pilot/debug/smoke và official runs phải được **tách vật lý và logic**.

```text
runs/pilot/
runs/official/
runs/ablation/
```

Cấm:

```text
pilot run
→ rename
→ official run
```

Official run phải được khởi tạo mới **sau**:

```text
PILOT_IMPLEMENTATION_PREFLIGHT = CLOSED / PASS
RESEARCHER_AUTHORIZATION = YES
```

Pilot run IDs không được xuất hiện trong official master result table.

---

## 2.4. Hidden-U GT firewall áp dụng cả cho artifacts

Run directory của SSL **không được chứa** ground-truth annotation ẩn của `U_b`.

Được phép lưu:

```text
U_b manifest hash
U_b image IDs hash
unlabeled COCO path/hash
proof annotations are empty
```

Không được lưu vào official training/development artifact:

```text
hidden U GT boxes
hidden U class labels
hidden U derived GT statistics
hidden U evaluator result
```

Hidden-U GT chỉ có thể tồn tại trong các offline construction/audit artifacts đã được phép ở phase dữ liệu trước đó.

---

## 2.5. Validation và test phải tách

Validation được phép dùng theo Methodology cho:

- checkpoint selection;
- development metrics;
- Q_pseudo;
- prespecified ablation evaluation.

Test:

```text
FINAL ONLY
```

Do đó artifact phải tách:

```text
metrics/validation/
evaluation/final_test/
```

Không được ghi test metric vào validation logs.

---


## 2.6. Artifact immutability

Artifact scientific của từng giai đoạn trở thành immutable sau khi giai đoạn tương ứng đạt:

```text
TRAINING_STATUS = CLOSED_VERIFIED
```

hoặc, đối với đánh giá cuối cùng:

```text
FINAL_EVALUATION_STATUS = CLOSED_VERIFIED
```

Không sử dụng một `RUN_STATUS` tổng quát để làm mờ ranh giới giữa kết thúc huấn luyện và kết thúc đánh giá test.

Sau khi trạng thái tương ứng đã đóng, các artifact scientific chính phải immutable về nội dung.

Nếu cần technical retry:

- tạo `attempt_id` mới;
- giữ `run_id` logic;
- dùng cùng exact training seed;
- ghi retry/deviation record;
- không overwrite artifact của attempt cũ.

---

## 2.7. Canonical scientific-source identity

Tên canonical duy nhất dùng trong manifests, logs và frozen configuration là:

```text
sn-article.tex
```

Tên file upload, tên bản sao cục bộ hoặc tên transport như:

```text
sn-article(9).tex
sn-article(20260902-092314).tex
```

không được dùng làm scientific identity nếu researcher đã xác nhận canonical source là
`sn-article.tex`.

Mọi run phải ghi:

```text
scientific_source_name = sn-article.tex
scientific_source_sha256 = <SHA-256 của file canonical tại thời điểm freeze>
```

Nếu hash thay đổi sau freeze:

```text
RUN = BLOCKED
→ controlled revision / researcher review required
```

Exact 10 training-seed integers là operational values từ implementation/seed protocol
đã khóa; không được ghi sai provenance rằng 10 integer values này xuất phát trực tiếp
từ nội dung `sn-article.tex` nếu scientific source chỉ khóa policy dùng 10 seeds.

---

## 2.8. Condition-role firewall

Mỗi run phải có `condition_role` để ngăn reference/ablation lọt vào paired Main
analysis:

```text
MAIN_LOW_LABEL
SUP_REFERENCE
ABLATION
PILOT
```

Quy tắc:

```text
MAIN_LOW_LABEL
→ eligible cho paired SUP–SSL low-label analysis nếu pair_valid=true

SUP_REFERENCE
→ 100%-SUP reference only
→ không có SSL pair
→ không đi vào RQ2/RQ3/RQ4/RQ5/RQ6/RQ7 paired low-label matrix

ABLATION
→ validation-only cho non-Main variants
→ không được final-test

PILOT
→ never eligible for official results
```

---

# 3. ROOT DIRECTORY CONTRACT

Đề xuất root trong repository/persistent storage:

```text
artifacts/
├── governance/
├── preflight/
├── frozen/
├── runs/
│   ├── pilot/
│   ├── official/
│   └── ablation/
├── results/
├── final_test/
├── analysis/
├── thesis_artifacts/
└── archives/
```

Chi tiết:

```text
artifacts/
├── governance/
│   ├── ARTIFACT_AND_EVIDENCE_CONTRACT.md
│   ├── scientific_source_manifest.json
│   ├── implementation_contract_manifest.json
│   ├── artifact_contract_manifest.json
│   └── schema_version.json
│
├── preflight/
│   ├── environment/
│   ├── data/
│   ├── seed/
│   ├── sup/
│   ├── ssl/
│   ├── pairing/
│   ├── evaluator/
│   ├── statistics/
│   ├── geometry/
│   ├── resume/
│   ├── firewall/
│   └── GLOBAL_PREFLIGHT_REPORT.json
│
├── frozen/
│   ├── FROZEN_EXPERIMENTAL_CONFIGURATION.yaml
│   ├── frozen_config_manifest.json
│   ├── frozen_environment.json
│   ├── frozen_source_manifest.json
│   ├── frozen_data_manifest.json
│   ├── frozen_evaluator_manifest.json
│   ├── frozen_statistics_manifest.json
│   ├── TEST_FIREWALL_STATE.json
│   └── FREEZE_AUTHORIZATION.json
│
├── runs/
│   ├── pilot/
│   ├── official/
│   └── ablation/
│
├── results/
│   ├── official_run_inventory.csv
│   ├── official_training_summary.csv
│   ├── official_pair_inventory.csv
│   ├── paired_effects.csv
│   ├── completeness_report.json
│   ├── result_artifact_manifest.json
│   └── ablation/
│       ├── ablation_config_inventory.csv
│       ├── ablation_run_inventory.csv
│       ├── ablation_main_reference_map.csv
│       ├── ablation_pair_inventory.csv
│       ├── ablation_effects.csv
│       └── ablation_completeness_report.json
│
├── final_test/
│   ├── PRE_FINAL_TEST_GATE_REPORT.json
│   ├── FINAL_TEST_AUTHORIZATION.json
│   ├── eligible_model_inventory.csv
│   ├── final_test_campaign_manifest.json
│   ├── test_access_audit.json
│   └── official_final_test_results.csv
│
├── analysis/
│   ├── rq1/
│   ├── rq2/
│   ├── rq3/
│   ├── rq4/
│   ├── rq5/
│   ├── rq6/
│   ├── rq7/
│   ├── ablation/
│   ├── multiplicity/
│   └── robustness/
│
├── thesis_artifacts/
│   ├── tables/
│   ├── figures/
│   └── provenance/
│
└── archives/
    ├── completed_runs/
    └── evidence_packages/
```

`artifacts/final_test/` là campaign-level evidence. Per-run final-test artifacts chỉ
được materialize sau `FINAL_TEST_AUTHORIZATION = YES`.

# 4. RUN ID CONTRACT

## 4.1. Official low-label run

Canonical logical run ID:

```text
OFF_{METHOD}_{ARCH}_{BUDGET}_{SEED_INDEX}
```

Ví dụ:

```text
OFF_SUP_R50_B10_S03
OFF_SSL_SWIN_B05_S07
```

Trong đó:

```text
METHOD      = SUP | SSL
ARCH        = R50 | SWIN
BUDGET      = B01 | B05 | B10 | B20
SEED_INDEX  = S01 ... S10
```

`training_seed` vẫn phải ghi trong manifest; không suy diễn training seed chỉ từ `seed_index`.

---

## 4.2. 100%-SUP reference

```text
OFF_SUP_R50_B100_S03
OFF_SUP_SWIN_B100_S03
```

---

## 4.3. Pilot run

```text
PILOT_{METHOD}_{ARCH}_{PURPOSE}_{NNN}
```

Ví dụ:

```text
PILOT_SSL_R50_EMA_001
PILOT_SUP_SWIN_RESUME_002
```

Pilot ID **không dùng seed_index official như identity của scientific result**, dù pilot có thể dùng một seed kỹ thuật đã được phép cho preflight.

---

## 4.4. Ablation run

```text
ABL_R50_B10_{FACTOR}_{LEVEL}_S{SEED_INDEX}
```

Ví dụ:

```text
ABL_R50_B10_CLS_THR_070_S04
ABL_R50_B10_AUG_A1_S04
ABL_R50_B10_REG_F1_S04
```

---


## 4.5. Condition role by run type

```text
OFF_SUP_R50_B10_S03
→ condition_role = MAIN_LOW_LABEL

OFF_SSL_R50_B10_S03
→ condition_role = MAIN_LOW_LABEL

OFF_SUP_R50_B100_S03
→ condition_role = SUP_REFERENCE

ABL_R50_B10_...
→ condition_role = ABLATION

PILOT_...
→ condition_role = PILOT
```

`B100` không được suy diễn là một low-label Main budget.

---

## 4.6. Attempt ID

Mỗi logical run có thể có nhiều technical attempts:

```text
attempt_001
attempt_002
...
```

Ví dụ:

```text
runs/official/OFF_SSL_R50_B10_S03/attempt_001/
```

Nếu attempt 001 lỗi kỹ thuật và retry:

```text
runs/official/OFF_SSL_R50_B10_S03/attempt_002/
```

Same exact `training_seed` là bắt buộc.

---

# 5. ARTIFACT LIFECYCLE

Mỗi artifact hoặc run đi qua:

```text
STAGING
→ WRITTEN
→ VALIDATED
→ HASHED
→ CLOSED
```

Official scientific result chỉ được promote nếu:

```text
all mandatory artifacts present
AND schema validation PASS
AND artifact manifest complete
AND checkpoint provenance PASS
AND training_completion_report PASS
AND final_evaluation_report PASS when final evaluation is applicable
```

Không promote partial outputs.

---

# 6. PREFLIGHT EVIDENCE CONTRACT

Trước official training phải tạo tối thiểu:

```text
artifacts/preflight/
├── environment/
│   ├── vast_environment.json
│   ├── docker_image_identity.json
│   ├── storage_environment.json
│   └── statistics_environment.json
│
├── data/
│   ├── dataset_transfer_manifest.json
│   ├── dataset_integrity_report.json
│   ├── split_membership_audit.json
│   ├── labeled_unlabeled_membership_audit.json
│   ├── class_mapping_audit.json
│   └── dataset_loader_manifest.json
│
├── seed/
│   ├── training_seed_invariance_audit.json
│   ├── seed_propagation_audit.json
│   └── dataloader_worker_seed_audit.json
│
├── sup/
│   ├── sup_r50_preflight.json
│   ├── sup_swin_preflight.json
│   ├── schedule_1pct_preflight.json
│   ├── schedule_2064_preflight.json
│   ├── schedule_100pct_10284_preflight.json
│   ├── supervised_augmentation_manifest.json
│   └── supervised_entrypoint_manifest.json
│
├── ssl/
│   ├── ssl_r50_preflight.json
│   ├── ssl_swin_preflight.json
│   ├── teacher_initialization_test.json
│   ├── ema_timing_test.json
│   ├── amp_ema_skip_test.json
│   ├── empty_pseudo_batch_test.json
│   ├── zero_gt_test.json
│   ├── ssl_augmentation_manifest.json
│   └── ssl_entrypoint_manifest.json
│
├── pairing/
│   └── sup_ssl_pairing_config_diff_report.json
│
├── geometry/
│   ├── gt_bbox_alignment_report.json
│   ├── pseudo_bbox_alignment_report.json
│   └── visual_evidence_manifest.json
│
├── firewall/
│   ├── hidden_u_gt_firewall_report.json
│   └── test_firewall_preflight.json
│
├── evaluator/
│   ├── coco_evaluator_golden_test.json
│   ├── operating_point_golden_test.json
│   └── qpseudo_path_golden_test.json
│
├── resume/
│   └── resume_equivalence_test.json
│
├── statistics/
│   ├── statistical_fixture_report.json
│   ├── multiplicity_fixture_report.json
│   └── statistical_implementation_manifest.json
│
└── GLOBAL_PREFLIGHT_REPORT.json
```

### 6.1. 100%-SUP reference path

100%-SUP có `10284` **actual optimizer updates**. Scheduler phải dùng cùng
relative-position rule đã khóa trong scientific/implementation protocol:

```text
LR drop 1 = 2/3 × U_opt
LR drop 2 = 11/12 × U_opt
```

Do đó đối với 100%-SUP:

```text
expected_updates = 10284
scheduler milestones = [6856, 9427]
```

Nếu implementation sử dụng path/scheduler khác với low-label schedules thì path đó
phải được pilot trực tiếp. Nếu dùng cùng structural scheduler code, vẫn phải có
automated evidence:

```text
schedule_100pct_10284_preflight.json
```

chứng minh:

```text
expected_updates = 10284
actual-update accounting = PASS
scheduler_rule = RELATIVE_POSITIONS_2_3_AND_11_12
scheduler_milestones = [6856, 9427]
scheduler milestones/rule = MATCH FROZEN CONFIG
validation interval = 172 actual optimizer updates
checkpoint path = SAME VERIFIED IMPLEMENTATION FAMILY
```

Automated evidence cũng phải chứng minh scheduler tiến theo **actual optimizer
updates**; raw iteration/microbatch hoặc AMP-skipped optimizer step không được làm
dịch scheduler progress.

Không được suy diễn 100%-SUP là PASS chỉ vì 2064-update path đã PASS.

### 6.2. Pairing preflight

Trước official training phải có automated config diff cho SUP–SSL cùng
architecture/budget/seed template để chứng minh các trường phải matched là matched
và các trường được phép khác chỉ là SSL treatment.

### 6.3. Test firewall preflight

`test_firewall_preflight.json` là **MANDATORY**, không phải optional. Nó phải chứng minh:

```text
test evaluator not invoked in preflight
test dataloader not available to checkpoint-selection path
Q_pseudo uses validation only
non-Main ablation test access = BLOCKED
```


### 6.4. Canonical data invariants phải được evidence trên Vast.ai

Preflight không chỉ xác nhận hash tồn tại mà phải đối chiếu các invariant đã khóa:

```text
working_scope_images = 4894
detection_classes = 14
No Finding = zero-GT negative images, not class 15

train = 3426
validation = 734
test = 734

No Finding:
train = 350
validation = 75
test = 75
```

Low-label memberships:

```text
budget   L_b    U_b    No Finding in L_b
1%       34     3392   3
5%       171    3255   17
10%      343    3083   35
20%      685    2741   70
```

Relations:

```text
L_1% ⊂ L_5% ⊂ L_10% ⊂ L_20% ⊂ T
U_b = T \ L_b
L_b ∩ U_b = ∅
L_b ∪ U_b = T
```

Official 1% subset evidence phải xác nhận:

```text
size = 34
abnormality_class_coverage = 14/14
minimum_class_coverage_repair_move_count = 0
construction_identity = label-aware / coverage-constrained
```

Giá trị `minimum_class_coverage_repair_move_count = 0` phải được diễn giải đúng:
dedicated coverage-repair không cần swap trong official construction; không được ghi
rằng coverage repair đã tạo ra 14/14 coverage.

Các giá trị này phải xuất hiện trong:

```text
dataset_integrity_report.json
split_membership_audit.json
labeled_unlabeled_membership_audit.json
class_mapping_audit.json
```

Nếu count đúng nhưng membership hash sai:

```text
DATA PREFLIGHT = FAIL
```

Không rebuild split/L/U trên Vast để “sửa” mismatch.

---

### 6.5. Google Drive → Vast data-ingress contract

Google Drive chỉ là **transport/durable storage source**. Official/pilot DataLoader
không được train trực tiếp từ Google Drive.

Luồng bắt buộc:

```text
Google Drive
→ download/copy
→ Vast.ai local or persistent storage
→ integrity verification
→ DataLoader
```

`dataset_transfer_manifest.json` phải ghi tối thiểu:

```text
transport_source = Google Drive
expected_jpg_count = 4894
observed_jpg_count = 4894
missing_jpg_count = 0
unexpected_jpg_count = 0
destination_storage_type = local | persistent
destination_image_root
transfer timestamp
checksum/tree-hash evidence
```

`dataset_loader_manifest.json` phải ghi:

```text
training_image_root = <Vast local/persistent path>
direct_google_drive_io_for_training = false
COCO annotation roots/paths
image-root existence checks
sample path-resolution checks
```

Nếu DataLoader resolve ảnh qua Google Drive mount/remote path trong official training:

```text
DATA INGEST PREFLIGHT = FAIL
OFFICIAL TRAINING = BLOCKED
```

---

# 7. GLOBAL_PREFLIGHT_REPORT SCHEMA

`GLOBAL_PREFLIGHT_REPORT.json` phải ánh xạ trực tiếp các closure criteria trong
implementation handoff. Không thay đổi scientific gate; các evidence bổ sung dưới đây
làm mạnh hơn khả năng audit của cùng gate.

Tối thiểu:

```json
{
  "schema_version": "1.1",
  "status": "PASS",
  "researcher_authorization": false,
  "checks": [
    {
      "check_id": "PF01",
      "name": "data_membership_seed",
      "expected": "PASS",
      "observed": "PASS",
      "evidence_paths": [
        "artifacts/preflight/data/split_membership_audit.json"
      ],
      "status": "PASS"
    }
  ],
  "unresolved_protocol_relevant_warnings": 0,
  "all_required_checks_pass": true,
  "official_training_authorized": false
}
```

25 closure groups phải bao phủ:

```text
01 data/membership/seed
02 DataLoader/sampler/worker stochasticity
03 SUP R50
04 SUP Swin-T
05 SSL R50
06 SSL Swin-T
07 1% scheduler/update path
08 standard 2064-update path + explicit 100%-SUP/10284 path `[6856,9427]` reference-path evidence
09 image–GT bbox geometry
10 image–pseudo-box geometry
11 Teacher initialization
12 EMA timing
13 AMP skipped-step → EMA skip
14 empty pseudo-label batch
15 zero-GT handling
16 hidden-U GT firewall
17 BEST/LAST checkpoint logic
18 Q_pseudo LAST-Teacher/validation
19 COCO evaluator
20 operating-point metrics
21 resume path, if enabled
22 run manifest/seed/provenance + entrypoint/DataLoader/augmentation evidence
23 pilot-vs-official isolation + test-firewall preflight
24 frozen executable configs/hashability + SUP–SSL pairing config-diff evidence
25 no unresolved protocol-relevant warning
```

Nếu bất kỳ official execution path nào chưa được evidence dù không tạo một check ID
mới riêng, `status` của closure group liên quan phải là `FAIL`.

# 8. PER-RUN DIRECTORY CONTRACT

## 8.1. SUP official run — training stage

```text
runs/official/OFF_SUP_R50_B10_S03/
├── run_root_manifest.json
├── attempt_001/
│   ├── pre_run_assertions.json
│   ├── run_manifest.json
│   ├── resolved_config.yaml
│   ├── config.sha256
│   ├── data_provenance.json
│   ├── seed_runtime.json
│   ├── environment_reference.json
│   ├── train.jsonl
│   ├── runtime_events.jsonl
│   │
│   ├── metrics/
│   │   └── validation/
│   │       ├── validation_history.csv
│   │       ├── validation_best.json
│   │       ├── classwise_validation_ap.csv
│   │       └── training_summary.json
│   │
│   ├── checkpoints/
│   │   ├── checkpoint_index.json
│   │   ├── best_model.pth
│   │   ├── last_model.pth
│   │   └── latest_resume.pth
│   │
│   ├── retry_deviation.json
│   ├── training_completion_report.json
│   ├── artifact_manifest.json
│   └── backup_sync_manifest.json
│
└── logical_run_status.json
```

Không tạo `evaluation/final_test/` trong training stage.

---

## 8.2. SSL official run — training stage

```text
runs/official/OFF_SSL_R50_B10_S03/
├── run_root_manifest.json
├── attempt_001/
│   ├── pre_run_assertions.json
│   ├── run_manifest.json
│   ├── resolved_config.yaml
│   ├── config.sha256
│   ├── data_provenance.json
│   ├── seed_runtime.json
│   ├── environment_reference.json
│   ├── train.jsonl
│   ├── runtime_events.jsonl
│   │
│   ├── metrics/
│   │   └── validation/
│   │       ├── validation_history.csv
│   │       ├── validation_best_teacher.json
│   │       ├── classwise_validation_ap.csv
│   │       └── training_summary.json
│   │
│   ├── checkpoints/
│   │   ├── checkpoint_index.json
│   │   ├── best_ema_teacher.pth
│   │   ├── last_ema_teacher.pth
│   │   └── latest_resume.pth
│   │
│   ├── pseudo/
│   │   ├── qpseudo_validation.json
│   │   ├── qpseudo_classwise.csv
│   │   ├── val_last_teacher_predictions.json
│   │   └── pseudo_diagnostics.json
│   │
│   ├── retry_deviation.json
│   ├── training_completion_report.json
│   ├── artifact_manifest.json
│   └── backup_sync_manifest.json
│
└── logical_run_status.json
```

---

## 8.3. Final-test materialization sau authorization

Chỉ sau global `FINAL_TEST_AUTHORIZATION = YES`, eligible Main/reference run mới
được tạo:

```text
attempt_001/
└── evaluation/
    └── final_test/
        ├── coco_metrics.json
        ├── operating_point_metrics.json
        ├── classwise_ap.csv
        ├── negative_metrics.json
        ├── detections.json
        ├── evaluation_manifest.json
        └── final_evaluation_report.json
```

Non-Main ablation variant:

```text
evaluation/final_test/
= PROHIBITED
```

# 9. `run_manifest.json` — CANONICAL RUN IDENTITY

Bắt buộc cho mọi pilot, official và ablation run.

Schema tối thiểu:

```json
{
  "schema_version": "1.1",
  "run_id": "OFF_SSL_R50_B10_S03",
  "attempt_id": "attempt_001",
  "run_type": "OFFICIAL",
  "condition_role": "MAIN_LOW_LABEL",
  "method": "SSL",
  "architecture": "R50-FPN",
  "budget": "10pct",
  "seed_index": 3,
  "training_seed": 1798418854,
  "partition_seed": 42,

  "scientific_source_name": "sn-article.tex",
  "scientific_source_sha256": "...",
  "implementation_contract_sha256": "...",
  "artifact_contract_sha256": "...",

  "git_commit": "...",
  "docker_image": "...",
  "docker_digest": "...",
  "entrypoint_id": "...",

  "dataset_manifest_sha256": "...",
  "train_split_sha256": "...",
  "val_split_sha256": "...",
  "test_split_sha256": "...",
  "labeled_manifest_sha256": "...",
  "unlabeled_manifest_sha256": "...",

  "preprocessing_config_sha256": "...",
  "labeled_pipeline_sha256": "...",
  "optimizer_recipe_sha256": "...",
  "evaluator_config_sha256": "...",

  "optimizer": "...",
  "learning_rate": 0.0,
  "weight_decay": 0.0,
  "total_optimizer_updates": 0,
  "validation_interval_updates": 172,
  "effective_labeled_batch": 4,
  "effective_unlabeled_batch": 4,
  "amp": true,
  "gradient_clipping": false,

  "checkpoint_selection_metric": "bbox_mAP_50_95_validation",
  "test_usage": "FINAL_ONLY",
  "training_status": "RUNNING",
  "final_evaluation_status": "NOT_AUTHORIZED"
}
```

Đối với SUP:

```text
effective_unlabeled_batch = NOT_APPLICABLE
unlabeled_manifest_sha256 = NOT_APPLICABLE
```

Đối với `SUP_REFERENCE`:

```text
budget = 100pct
condition_role = SUP_REFERENCE
```

SSL thêm bắt buộc:

```json
{
  "ema_momentum": 0.001,
  "ema_skip_buffers": true,
  "ema_update_basis": "ACTUAL_OPTIMIZER_UPDATE",
  "labeled_unlabeled_ratio": "1:1",
  "unsupervised_loss_weight": 4.0,
  "weak_augmentation_id": "...",
  "strong_augmentation_id": "...",
  "detector_score_floor": 0.05,
  "initial_pseudo_score": 0.50,
  "rpn_pseudo_threshold": 0.90,
  "rcnn_classification_threshold": 0.90,
  "rpn_nms_iou": 0.70,
  "rpn_max": 1000,
  "rcnn_nms_iou": 0.50,
  "rcnn_max": 100,
  "jitter_times": 10,
  "jitter_scale": 0.06,
  "reg_uncertainty_threshold": 0.02,
  "min_pseudo_bbox_wh": [0.01, 0.01]
}
```

# 10. `data_provenance.json`

Mục tiêu: chứng minh run dùng đúng data membership và đúng preprocessing/data path.

Ví dụ low-label SSL:

```json
{
  "condition_role": "MAIN_LOW_LABEL",
  "train_image_count": 3426,
  "val_image_count": 734,
  "test_image_count": 734,
  "budget": "10pct",
  "labeled_image_count": 343,
  "unlabeled_image_count": 3083,
  "labeled_no_finding_count": 35,
  "category_count": 14,

  "train_membership_sha256": "...",
  "val_membership_sha256": "...",
  "test_membership_sha256": "...",
  "labeled_membership_sha256": "...",
  "unlabeled_membership_sha256": "...",

  "dataset_loader_config_sha256": "...",
  "preprocessing_config_sha256": "...",
  "image_root_identity": "...",

  "labeled_unlabeled_disjoint": true,
  "labeled_unlabeled_union_equals_train": true,
  "hidden_u_annotations_present_in_training_object": false
}
```

SUP low-label vẫn ghi labeled membership; `U_b` không được load vào supervised training path.

100%-SUP:

```text
condition_role = SUP_REFERENCE
labeled_image_count = 3426
unlabeled_image_count = NOT_APPLICABLE
```

Không copy hidden-U GT vào file này.

# 11. `seed_runtime.json`

Mục tiêu: chứng minh `training_seed` đã được propagate tới các stochastic sources
thực tế. Contract **không ép** mọi RNG stream phải nhận cùng một integer trực tiếp;
framework có thể derive worker/sampler seeds theo cơ chế hợp lệ.

Schema tối thiểu:

```json
{
  "base_training_seed": 1798418854,

  "python_seed_observed": 1798418854,
  "numpy_seed_observed": 1798418854,
  "torch_initial_seed_observed": 1798418854,

  "cuda_seed_policy": "...",
  "sampler_seed_observed": "...",
  "worker_seed_strategy": "...",
  "worker_seed_examples": [],
  "augmentation_rng_strategy": "...",

  "torch_deterministic_algorithms_observed": null,
  "cudnn_deterministic_observed": null,
  "cudnn_benchmark_observed": null,

  "membership_hash_before_seed_application": "...",
  "membership_hash_after_seed_application": "...",
  "membership_unchanged": true
}
```

Nếu một source sử dụng derived seed:

```text
record base seed
record derivation/policy
record representative observed worker/sampler seeds
```

Không sửa framework chỉ để mọi seed number giống hệt nhau.

Các deterministic flags phải ghi **observed runtime values**, không hard-code theo giả định.

# 12. TRAINING LOG CONTRACT

## 12.1. `train.jsonl`

Một record theo **actual optimizer-update event**.

Tối thiểu:

```json
{
  "optimizer_update": 344,
  "raw_iteration": 688,
  "learning_rate": 0.005,
  "loss_total": 0.0,
  "loss_supervised": 0.0,
  "amp_scale": 0.0,
  "optimizer_step_executed": true,
  "elapsed_seconds": 0.0,
  "gpu_memory_allocated_mb": 0.0,
  "labeled_microbatches_consumed_cumulative": 0,
  "labeled_images_consumed_cumulative": 0
}
```

SSL thêm:

```json
{
  "loss_unsupervised": 0.0,
  "pseudo_count": 0,
  "classification_pseudo_count": 0,
  "regression_pseudo_count": 0,
  "unlabeled_microbatches_consumed_cumulative": 0,
  "unlabeled_images_consumed_cumulative": 0,
  "ema_step_executed": true
}
```

Các cumulative exposure fields dùng để audit:

```text
same total optimizer-update budget
same effective labeled batch
same labeled-supervision exposure
```

giữa paired SUP–SSL.

---

## 12.2. `runtime_events.jsonl`

Chỉ dùng cho event-level evidence, ví dụ:

```text
RUN_START
SEED_APPLIED
DATALOADER_CREATED
OPTIMIZER_STEP_SKIPPED_BY_AMP
EMA_STEP_SKIPPED
VALIDATION_START
VALIDATION_END
BEST_CHECKPOINT_UPDATED
RESUME_SAVE
RESUME_RESTORE
FINAL_UPDATE_REACHED
RUN_END
```

Record mẫu:

```json
{
  "timestamp_utc": "...",
  "event": "OPTIMIZER_STEP_SKIPPED_BY_AMP",
  "optimizer_update_before": 125,
  "reason": "non_finite_gradient"
}
```

Nếu AMP skip:

```text
optimizer_step_executed = false
EMA_step_executed       = false
```

phải có evidence đồng bộ.

---

## 12.3. `training_summary.json`

Mỗi run phải tổng hợp observed values:

```text
expected_optimizer_updates
observed_optimizer_updates
effective_labeled_batch
effective_unlabeled_batch nếu SSL
labeled_images_consumed_total
unlabeled_images_consumed_total nếu SSL
validation_events_observed
AMP skipped-step count
EMA update count nếu SSL
```

Pair audit không được chỉ dựa trên config; phải đối chiếu observed labeled exposure
từ `training_summary.json`.

# 13. VALIDATION ARTIFACT CONTRACT

## 13.1. `validation_history.csv`

Bắt buộc:

```text
optimizer_update,
bbox_mAP_50_95,
AP50,
AP75,
AR_50_95_max100,
checkpoint_candidate,
best_so_far,
checkpoint_path
```

Không ghi test metric.

---

## 13.2. `validation_best.json`

SUP:

```json
{
  "selection_split": "validation",
  "selection_metric": "bbox_mAP_50_95",
  "selected_update": 0,
  "selected_metric_value": 0.0,
  "checkpoint_path": "checkpoints/best_model.pth",
  "checkpoint_sha256": "...",
  "rule": "ARGMAX_VALIDATION_BBOX_MAP_50_95"
}
```

SSL:

```json
{
  "selection_split": "validation",
  "selection_model": "EMA_TEACHER",
  "selection_metric": "bbox_mAP_50_95",
  "selected_update": 0,
  "selected_metric_value": 0.0,
  "checkpoint_path": "checkpoints/best_ema_teacher.pth",
  "checkpoint_sha256": "...",
  "rule": "ARGMAX_VALIDATION_TEACHER_BBOX_MAP_50_95"
}
```

---

# 14. CHECKPOINT RETENTION CONTRACT

## 14.1. SUP

Scientific checkpoints bắt buộc:

```text
BEST
LAST
```

Operational checkpoint:

```text
LATEST_RESUME
```

---

## 14.2. SSL

Scientific checkpoints bắt buộc:

```text
BEST EMA Teacher
LAST EMA Teacher
```

Operational checkpoint:

```text
LATEST_RESUME
```

`LATEST_RESUME` phải đủ để tiếp tục cùng training trajectory theo implementation contract, tối thiểu chứa hoặc tham chiếu tới:

```text
Student state
Teacher state
optimizer state
scheduler state
AMP scaler state
actual optimizer update counter
training seed identity
sampler/DataLoader resume state nếu implementation hỗ trợ
relevant RNG states nếu implementation hỗ trợ
```

Khi resume:

```text
restore Student
restore Teacher
DO NOT resync Teacher from Student
```

---

## 14.3. Intermediate checkpoint

Không bắt buộc giữ toàn bộ intermediate checkpoints.

Được phép dùng rolling checkpoint để tiết kiệm storage nếu:

```text
BEST không bị overwrite
LAST được materialize cuối run
resume checkpoint tồn tại khi run đang active
checkpoint_index.json ghi đầy đủ provenance
```

Chỉ xóa intermediate checkpoint sau khi:

```text
validation record đã persist
AND best/non-best decision đã ghi
AND artifact write verified
```

---

# 15. `checkpoint_index.json`

```json
{
  "checkpoint_policy": "LOCKED",
  "entries": [
    {
      "role": "BEST",
      "model_role": "EMA_TEACHER",
      "optimizer_update": 1548,
      "validation_bbox_mAP_50_95": 0.0,
      "path": "best_ema_teacher.pth",
      "sha256": "...",
      "retained": true
    },
    {
      "role": "LAST",
      "model_role": "EMA_TEACHER",
      "optimizer_update": 2064,
      "path": "last_ema_teacher.pth",
      "sha256": "...",
      "retained": true
    }
  ]
}
```

---

# 16. Q_PSEUDO ARTIFACT CONTRACT

Chỉ cho SSL.

Primary:

```text
Q_pseudo = PL-mAP@[0.50:0.95]
checkpoint = LAST EMA Teacher
dataset = fixed validation
pseudo set = accepted RCNN classification pseudo labels
```

Bắt buộc:

```text
pseudo/qpseudo_validation.json
pseudo/val_last_teacher_predictions.json
pseudo/qpseudo_classwise.csv
```

`qpseudo_validation.json` tối thiểu:

```json
{
  "checkpoint_role": "LAST_EMA_TEACHER",
  "checkpoint_sha256": "...",
  "dataset": "fixed_validation",
  "validation_split_sha256": "...",
  "pseudo_set": "accepted_rcnn_classification",
  "pseudo_acceptance_config_sha256": "...",
  "evaluator_config_sha256": "...",
  "hidden_u_gt_used": false,
  "test_used": false,
  "PL_mAP_50_95": 0.0,
  "PL_AP50": 0.0,
  "PL_AP75": 0.0
}
```

Diagnostics phụ có thể lưu:

```text
precision
recall
F1
matched IoU
pseudo count
retention
class-wise PL-AP
pseudo FP on validation negatives
regression uncertainty summary
```

Nhưng không thay thế Q_pseudo primary.

---

# 17. FINAL TEST ARTIFACT CONTRACT

Chỉ materialize sau **global final-test authorization**.

## 17.1. `coco_metrics.json`

```json
{
  "evaluation_model_role": "BEST",
  "evaluation_split": "test",
  "bbox_mAP_50_95": 0.0,
  "AP50": 0.0,
  "AP75": 0.0,
  "AR_50_95_max100": 0.0,
  "useCats": 1,
  "maxDets": 100,
  "extra_fixed_score_threshold_before_AP": false
}
```

SSL:

```text
evaluation_model_role = BEST_EMA_TEACHER
```

---

## 17.2. `operating_point_metrics.json`

Bắt buộc để không bỏ sót các secondary outcomes của Methodology:

```json
{
  "evaluation_split": "test",
  "tau_eval": 0.50,
  "iou_match_threshold": 0.50,
  "matching": "CATEGORY_AWARE_ONE_TO_ONE",
  "matching_algorithm": "DETERMINISTIC_SCORE_DESCENDING_GREEDY",
  "prediction_order": "DESCENDING_CONFIDENCE_STABLE_DETECTOR_ORDER",
  "gt_candidate_policy": "UNMATCHED_SAME_CLASS_ONLY",
  "gt_selection": "MAX_IOU",
  "equal_score_tie_break": "PRESERVE_DETECTOR_OUTPUT_ORDER",
  "equal_iou_tie_break": "EARLIEST_FIXED_COCO_ANNOTATION_ORDER",
  "after_detector_native_nms_max100": true,
  "evaluated_image_count": 734,
  "GT_total": 0,
  "TP_total": 0,
  "FN_total": 0,
  "Recall_tau_eval": 0.0,
  "FP_total": 0,
  "FP_per_image": 0.0
}
```

`tau_eval=0.50` chỉ dùng cho operating-point metrics; không được áp trước COCO AP.
`operating_point_golden_test.json` phải xác nhận đúng deterministic score-descending greedy, category-aware, one-to-one matching: detection được xử lý theo confidence giảm dần; score bằng nhau giữ thứ tự detector output; chỉ xét unmatched ground truth cùng lớp; chọn ground truth có IoU lớn nhất; IoU bằng nhau chọn ground truth xuất hiện sớm nhất theo fixed COCO annotation order; và điều kiện match sử dụng `IoU >= 0.50`.

---

## 17.3. `classwise_ap.csv`

```text
class_id,
class_name,
gt_support,
AP_50_95,
AP50,
AP75,
status
```

Nếu AP không xác định do không có GT support:

```text
status = UNDEFINED
metric = NA
```

Không gán `0`.

---

## 17.4. `negative_metrics.json`

```json
{
  "tau_eval": 0.50,
  "iou_match_threshold": 0.50,
  "matching": "CATEGORY_AWARE_ONE_TO_ONE",
  "negative_image_count": 75,
  "FP_total_on_negative": 0,
  "FP_per_negative": 0.0,
  "negative_image_FAR": 0.0
}
```

---

## 17.5. `detections.json`

Lưu detector-native detections đủ để tái tính evaluator nếu cần.

Không được thêm một threshold trước COCO AP.

---

## 17.6. `evaluation_manifest.json`

Phải ghi:

```text
run_id
checkpoint_role
checkpoint_sha256
evaluator_config_sha256
test_split_sha256
test_authorization_sha256
evaluation timestamp
```

---

## 17.7. Non-Main ablation firewall

```text
condition_role = ABLATION
AND variant != MAIN
→ FINAL TEST = PROHIBITED
```

Không có ngoại lệ hậu nghiệm dựa trên validation performance.

# 18. RETRY / FAILURE / DEVIATION CONTRACT

Mỗi attempt có:

```text
retry_deviation.json
```

Schema tối thiểu:

```json
{
  "run_id": "...",
  "attempt_id": "...",
  "training_seed": 0,
  "technical_failure": false,
  "failure_reason": null,
  "retry_required": false,
  "retry_reason": null,
  "same_seed_confirmed": true,
  "scientific_protocol_changed": false,
  "controlled_revision_id": null,
  "researcher_approval": null
}
```

Cấm retry vì:

```text
valid performance thấp
mAP xấu
seed không đẹp
SSL gain âm
```

---

# 19. COMPLETION STATUS CONTRACT

Training completion và final evaluation là **hai trạng thái khác nhau**.

## 19.1. `training_completion_report.json`

Được tạo khi official training đã hoàn tất và evidence training/validation hợp lệ.

```json
{
  "run_id": "...",
  "attempt_id": "...",
  "training_status": "CLOSED_VERIFIED",
  "expected_optimizer_updates": 2064,
  "observed_optimizer_updates": 2064,
  "validation_interval": 172,
  "best_checkpoint_present": true,
  "last_checkpoint_present": true,
  "artifact_manifest_valid": true,
  "scientific_protocol_deviation": false,
  "eligible_for_final_test": true,
  "final_evaluation_status": "NOT_AUTHORIZED"
}
```

Pilot luôn:

```text
eligible_for_final_test = false
```

Non-Main ablation:

```text
eligible_for_final_test = false
```

---

## 19.2. `final_evaluation_report.json`

Chỉ được tạo sau `FINAL_TEST_AUTHORIZATION = YES`.

```json
{
  "run_id": "...",
  "final_evaluation_status": "CLOSED_VERIFIED",
  "authorized": true,
  "eligible_model_inventory_match": true,
  "checkpoint_sha256_match": true,
  "test_split_sha256_match": true,
  "evaluation_manifest_valid": true,
  "eligible_for_official_analysis": true
}
```

Không dùng `TRAINING_STATUS = CLOSED` để suy diễn rằng test đã được mở.

# 20. `artifact_manifest.json`

Mỗi closed attempt phải có manifest của mọi artifact bắt buộc.

```json
{
  "schema_version": "1.1",
  "run_id": "...",
  "attempt_id": "...",
  "artifacts": [
    {
      "path": "metrics/validation/validation_history.csv",
      "role": "VALIDATION_HISTORY",
      "size_bytes": 0,
      "sha256": "...",
      "required": true
    }
  ],
  "mandatory_artifacts_complete": true
}
```

---

# 21. ENVIRONMENT EVIDENCE

Canonical Vast runtime evidence:

```text
artifacts/preflight/environment/vast_environment.json
```

Tối thiểu ghi observed values:

```text
OS
kernel
hostname/pod id nếu phù hợp
Python
PyTorch
MMDetection
MMCV
MMEngine
CUDA runtime
CUDA visible device
cuDNN
GPU model
GPU memory
NVIDIA driver
Docker image
Docker digest
```

Không claim bitwise reproducibility nếu evidence không chứng minh.

## 21.1. Statistical environment provenance

Phải có:

```text
artifacts/preflight/environment/statistics_environment.json
artifacts/preflight/statistics/statistical_implementation_manifest.json
```

Tối thiểu lưu:

```text
language/runtime versions
statistical package versions
script paths + SHA-256
implementation used for:
  - one-sample t tests
  - Greenhouse–Geisser correction
  - Holm adjustment
  - Kenward–Roger inference
  - exact sign-flip
  - LOSO
  - Friedman sensitivity
  - CR2 cluster-robust sensitivity
```

Không thay statistical implementation sau khi final test được mở nếu thay đổi đó có
thể làm đổi scientific result, trừ controlled revision theo protocol.

# 22. FROZEN CONFIGURATION CONTRACT

Trước official run phải tồn tại:

```text
artifacts/frozen/FROZEN_EXPERIMENTAL_CONFIGURATION.yaml
```

Mỗi parameter có:

```text
factor
level/value
scientific_role
source
change_permission
```

Ví dụ:

```yaml
partition_seed:
  value: 42
  source: sn-article.tex / approved methodology
  change_permission: PROHIBITED

training_seed_policy:
  value: 10_prespecified_locked_training_seeds
  source: sn-article.tex / approved methodology
  change_permission: PROHIBITED

training_seed_list:
  value:
    - 204886845
    - 1480646854
    - 1798418854
    - 2045683682
    - 1814859839
    - 1603952859
    - 1878351743
    - 875651179
    - 477581743
    - 869675675
  source: implementation_handoff / Phase_2F1_seed_protocol
  change_permission: PROHIBITED
```

Không ghi provenance của exact 10 integer values là `sn-article.tex` nếu canonical
scientific source chỉ khóa policy 10 seeds chứ không liệt kê chính các integer đó.

Scheduler fields trong frozen configuration phải ghi rõ:

```yaml
scheduler_relative_drop_positions:
  value: [2/3, 11/12]
  basis: ACTUAL_OPTIMIZER_UPDATE
  change_permission: PROHIBITED

scheduler_milestones:
  1pct: [688, 946]
  5pct: [1376, 1892]
  10pct: [1376, 1892]
  20pct: [1376, 1892]
  100pct_sup: [6856, 9427]
```

Mỗi frozen executable config phải có SHA-256.

# 23. PILOT-TO-OFFICIAL EQUIVALENCE EVIDENCE

Ngay trước mỗi official run, launcher tạo:

```text
pre_run_assertions.json
```

Phải assert ít nhất:

```text
scientific source name/hash
architecture
method
condition_role
budget
seed_index
training_seed
dataset/split manifest hash
L_b/U_b manifest hash khi applicable
detector family
preprocessing/input representation hash
labeled pipeline hash
optimizer recipe hash
learning rate
scheduler/update budget
effective labeled batch
effective unlabeled batch nếu SSL
AMP
augmentation
EMA nếu SSL
pseudo thresholds nếu SSL
loss weights nếu SSL
checkpoint rule
evaluation config
test-firewall state
```

Nếu mismatch ngoài allowed experimental factors:

```text
LAUNCH_STATUS = BLOCKED
```

`pre_run_assertions.json` phải được giữ trong official attempt directory.

## 23.1. SUP–SSL preflight pairing diff

Trước authorization phải có:

```text
artifacts/preflight/pairing/sup_ssl_pairing_config_diff_report.json
```

cho từng architecture/budget template. Report phải chứng minh các trường required-match:

```text
same train/val/test membership
same L_b
same architecture
same detector family
same preprocessing/input representation
same training seed mapping
same optimizer recipe within architecture
same total optimizer-update budget
same effective labeled batch
same validation/test evaluator config
same checkpoint-selection metric family
```

Khác biệt có chủ đích:

```text
SSL thêm U_b
Teacher/Student + EMA
weak/strong unlabeled streams
pseudo-labeling/filtering
unsupervised loss
```

Observed labeled-supervision exposure sẽ được kiểm lại sau run bằng
`training_summary.json`, không chỉ bằng config diff.

# 24. OFFICIAL MASTER INVENTORY

## 24.1. `official_run_inventory.csv`

Một dòng = một logical official training run.

Các cột tối thiểu:

```text
run_id
condition_role
method
architecture
budget
seed_index
training_seed
expected
started
training_completed
training_valid
eligible_for_final_test
final_evaluation_status
attempt_used
training_completion_report_path
artifact_manifest_path
```

Expected Main/reference training matrix:

```text
80 low-label SUP
20 100%-SUP reference
80 SSL
= 180 official runs trước ablation
```

Không giảm số seed vì compute/time nếu không controlled revision được researcher chấp thuận.

## 24.2. `official_training_summary.csv`

Một dòng = một official training run đã `TRAINING_STATUS=CLOSED_VERIFIED`.

Chứa training/validation provenance, không yêu cầu test metric:

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
last_checkpoint_sha256
Q_pseudo_if_SSL
training_status
eligible_for_final_test
```

File này có thể hoàn tất **trước khi test được mở**.

# 25. OFFICIAL FINAL-TEST RESULT MASTER TABLE

Canonical file sau final-test campaign:

```text
artifacts/final_test/official_final_test_results.csv
```

Một dòng = một eligible Main/reference official run đã final evaluation thành công.

Tối thiểu:

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
...
AP_14th_class

FP_total_negative
FP_per_negative
negative_FAR

Q_pseudo
Q_pseudo_status
final_evaluation_status
```

SUP:

```text
Q_pseudo = NA
Q_pseudo_status = NOT_APPLICABLE
```

`SUP_REFERENCE` được giữ để báo cáo reference nhưng không được đưa vào low-label
paired SUP–SSL analyses.

Không cross-seed prediction pooling.

Tên cũ `official_run_results.csv` không còn dùng làm canonical scientific result table
để tránh materialize test fields trước test authorization.

# 26. SUP–SSL PAIR INVENTORY

`official_pair_inventory.csv`

Một dòng = một paired low-label Main cell:

```text
architecture × budget × seed_index × training_seed
```

Chỉ:

```text
condition_role = MAIN_LOW_LABEL
budget ∈ {1%,5%,10%,20%}
```

Columns tối thiểu:

```text
pair_id
architecture
budget
seed_index
training_seed

sup_run_id
ssl_run_id

labeled_manifest_sha256_match
train_split_sha256_match
val_split_sha256_match
test_split_sha256_match
training_seed_match
architecture_match
detector_family_match
preprocessing_config_sha256_match
labeled_pipeline_sha256_match
optimizer_recipe_sha256_match
update_budget_match
effective_labeled_batch_match
evaluator_config_sha256_match
checkpoint_rule_match

observed_labeled_images_consumed_match
observed_optimizer_updates_match
labeled_supervision_exposure_match

pair_valid
```

`observed_*` phải lấy từ `training_summary.json`.

Chỉ `pair_valid=true` mới được đưa vào paired statistical analysis.

`SUP_REFERENCE` không được xuất hiện trong file này.

# 27. `paired_effects.csv`

Sinh từ:

```text
artifacts/final_test/official_final_test_results.csv
+
official_pair_inventory.csv
```

Các metric downstream trong file này là **final-test metrics**.

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

Không dùng validation mAP thay cho downstream test mAP trong RQ2/RQ3/RQ4/RQ7.

# 28. MAPPING ARTIFACT → RQ1–RQ7

## RQ1 — Supervised baseline theo labeled budget

Nguồn:

```text
artifacts/final_test/official_final_test_results.csv
filter condition_role=MAIN_LOW_LABEL
filter method=SUP
budget={1,5,10,20}
```

Output:

```text
analysis/rq1/rq1_supervised_budget_summary.csv
analysis/rq1/rq1_seed_level_values.csv
```

Báo cáo bắt buộc:

```text
mean
sample SD, ddof=1
```

Nếu trình bày two-sided 95% CI cho mean ở RQ1, phải ghi rõ đây là **descriptive
uncertainty**, không tạo thêm confirmatory hypothesis test hay endpoint.

RQ1 mô tả; không tạo hypothesis test riêng.

---

## RQ2 — Primary overall SSL gain

Nguồn:

```text
paired_effects.csv
```

Theo mỗi seed:

```text
G_s = mean của 8 ΔmAP_test architecture×budget cells
```

Output:

```text
analysis/rq2/rq2_gain_by_seed.csv
analysis/rq2/rq2_primary_test.json
analysis/rq2/rq2_cell_descriptives.csv
```

Primary:

```text
one-sample t-test
alternative=greater
n=10
df=9
two-sided 95% CI cho effect
one-sided p-value
```

Robustness:

```text
exact sign-flip
LOSO
```

---

## RQ3 — Budget-dependent SSL gain

Nguồn:

```text
paired_effects.csv
```

Theo budget, seed:

```text
B_b,s = mean ΔmAP_test của R50 và Swin-T
```

Output:

```text
analysis/rq3/rq3_budget_gain_by_seed.csv
analysis/rq3/rq3_gg_anova.json
analysis/rq3/rq3_pairwise_contrasts.csv
analysis/rq3/rq3_friedman_sensitivity.json
```

Primary implementation phải ghi rõ:

```text
one-way repeated-measures ANOVA
Greenhouse–Geisser correction = ALWAYS
Mauchly/sphericity = diagnostic only
```

Sáu paired contrasts prespecified duy nhất là:

```text
5%-1%
10%-1%
20%-1%
10%-5%
20%-5%
20%-10%
```

Với mỗi contrast c = b2 - b1 và training seed s:

```text
D_c,s = B_b2,s - B_b1,s
```

Pairwise inference được khóa như sau:

```text
two-sided one-sample t-test trên 10 paired differences D_c,s so với 0
equivalent to paired t-test giữa hai budget trên cùng training seeds
n = 10
df = 9
sample SD of paired differences: ddof = 1
effect = mean paired difference
CI = individual two-sided 95% CI
raw p-value = two-sided
```

Artifact analysis/rq3/rq3_pairwise_contrasts.csv phải lưu tối thiểu semantics của từng contrast gồm: contrast identity/direction, n, df, mean paired difference, sample SD với ddof=1, individual two-sided 95% CI và raw two-sided p-value.

S6.04 chỉ tạo và kiểm chứng six raw pairwise contrast results/raw p-values; không thực hiện Holm adjustment. Sáu raw p-values này thuộc Holm family F3. S6.10 thực hiện Holm step-down trên đúng sáu raw p-values, với m = 6 và FWER = 0.05. Individual 95% CI không phải Holm-adjusted simultaneous CI.

---

## RQ4 — Q_pseudo association

Nguồn:

```text
SSL qpseudo_validation.json
+
paired_effects.csv downstream test ΔmAP
```

Tối đa:

```text
2 architectures × 4 budgets × 10 seeds = 80 rows
```

Output:

```text
analysis/rq4/rq4_analysis_dataset.csv
analysis/rq4/rq4_primary_mixed_model.json
analysis/rq4/rq4_diagnostics.json
analysis/rq4/rq4_cr2_sensitivity.json
```

Dataset phải có:

```text
architecture
budget
architecture_budget_cell
seed_index
training_seed
Q_pseudo
Q_WC
X_per_1_pseudo_AP_point
delta_mAP_test
```

`rq4_primary_mixed_model.json` phải lưu tối thiểu:

```text
formula = Y ~ X + architecture_budget_cell + (1 | training_seed)
estimation = REML
fixed_effect_inference = Kenward–Roger
beta_Q_alternative = greater
effect_unit = downstream AP-point change per +1 pseudo AP point
effect_estimate
two-sided 95% CI
one-sided p-value
convergence status
singularity status
```

Không causal wording.

---

## RQ5 — Rare classes

Frozen rare classes:

```text
Atelectasis
Pneumothorax
```

Nguồn:

```text
paired_effects.csv
class-wise AP
```

Output:

```text
analysis/rq5/rq5_rare_class_gain_by_seed.csv
analysis/rq5/rq5_rare_class_summary.csv
```

Báo cáo:

```text
mean paired gain
sample SD
two-sided 95% CI
```

Không tạo `rare-mAP`.  
Không bắt buộc formal p-value.

---

## RQ6 — False positive trên No Finding

Nguồn:

```text
evaluation/final_test/negative_metrics.json
+
paired_effects.csv
```

Output:

```text
analysis/rq6/rq6_negative_fp_by_seed.csv
analysis/rq6/rq6_primary_test.json
analysis/rq6/rq6_far_summary.csv
```

Primary effect:

```text
H_s = mean 8 paired ΔFP_per_negative cells
```

Primary:

```text
two-sided one-sample t-test
n=10
df=9
```

FAR difference là estimation-focused secondary outcome.

---

## RQ7 — Architecture-dependent SSL effect

Nguồn:

```text
paired_effects.csv
```

Theo architecture và seed:

```text
A_a,s = mean ΔmAP_test qua 4 budgets
D_s = A_Swin,s - A_R50,s
```

Output:

```text
analysis/rq7/rq7_architecture_gain_by_seed.csv
analysis/rq7/rq7_primary_test.json
```

Primary:

```text
two-sided one-sample t-test
n=10
df=9
```

Không diễn giải thành absolute Swin superiority test.

---

# 28A. ABLATION EVIDENCE CONTRACT

Ablation là validation-only secondary mechanistic analysis theo frozen OFAT scope:

```text
architecture = R50
budget = 10%
seeds = all 10 locked seeds
Main SSL = frozen Main treatment
```

## 28A.1. Prespecified configuration inventory

Families:

```text
Confidence:
C0=0.5
C1=0.6
C2=0.7
C3=0.8
C4=0.9 = Main

Strong photometric augmentation:
A0=none
A1=brightness only
A2=contrast only
A3=brightness+contrast = Main

Regression filtering:
F0=no extra regression reliability filter
F1=confidence-based filtering >0.90
F2=initial score >0.50 AND reg_uncertainty <0.02 = Main
```

Vì:

```text
C4 = A3 = F2 = same frozen Main configuration
```

số unique configurations:

```text
5 + 4 + 3 - 2 duplicated Main identities = 10
```

Với 10 seeds:

```text
100 configuration–seed observations
```

Nếu 10 Main R50–10% official SSL runs đã tồn tại và hợp lệ:

```text
reuse Main observations
additional ablation trainings = 9 non-Main configs × 10 seeds = 90
```

Không train Main lại ba lần theo ba family.

---

## 28A.2. Ablation artifacts

```text
results/ablation/
├── ablation_config_inventory.csv
├── ablation_run_inventory.csv
├── ablation_main_reference_map.csv
├── ablation_pair_inventory.csv
├── ablation_effects.csv
└── ablation_completeness_report.json
```

`ablation_config_inventory.csv` tối thiểu:

```text
ablation_config_id
family
level
is_main
scientific_source_sha256
resolved_config_sha256
validation_only
final_test_eligible
```

Non-Main:

```text
validation_only = true
final_test_eligible = false
```

Main configuration được biểu diễn một lần trong `ablation_config_inventory.csv`.
Mapping theo từng seed phải nằm trong:

```text
ablation_main_reference_map.csv
```

với tối thiểu:

```text
seed_index
training_seed
main_run_id
main_checkpoint_sha256
main_validation_metrics_path
main_qpseudo_path
```

---

## 28A.3. Ablation pairing

Mỗi non-Main observation phải paired với frozen Main theo cùng:

```text
architecture = R50
budget = 10%
seed_index
training_seed
```

`ablation_pair_inventory.csv` phải có:

```text
ablation_run_id
main_run_id
family
level
seed_index
training_seed
config_diff_only_target_factor = true/false
pair_valid
```

OFAT pair chỉ hợp lệ khi:

```text
exactly one target scientific factor differs
all other frozen scientific settings match
```

---

## 28A.4. Ablation outputs

Ablation dùng **validation**, không dùng final test cho non-Main variants.

Tối thiểu tạo:

```text
analysis/ablation/
├── confidence_summary.csv
├── augmentation_summary.csv
├── regression_filter_summary.csv
├── ablation_variant_summary.csv
├── ablation_seed_level_values.csv
├── ablation_qpseudo_effects.csv
└── ablation_thesis_tables.tex
```

Mỗi variant phải truy ngược tới:

```text
run_id
seed_index
training_seed
validation checkpoint/evaluator
Main same-seed reference
paired validation metric
paired Q_pseudo khi applicable
```

Đối với từng variant và từng metric ablation được báo cáo, `ablation_variant_summary.csv`
phải chứa tối thiểu:

```text
variant_id
family
level
n_seeds = 10
variant_mean
variant_sample_SD_ddof1
main_mean
paired_mean_difference
paired_difference_two_sided_95CI_low
paired_difference_two_sided_95CI_high
formal_hypothesis_test = false
```

`ablation_seed_level_values.csv` phải giữ toàn bộ giá trị theo seed:

```text
variant_id
seed_index
training_seed
metric_name
variant_value
main_same_seed_value
paired_difference
```

Đối với ablation, `Q_pseudo` là **secondary mechanistic metric bắt buộc** theo
`sn-article.tex`, không phải metric tùy chọn.

```text
delta_Q_pseudo = Q_pseudo_variant - Q_pseudo_Main_same_seed
```

`ablation_qpseudo_effects.csv` phải lưu seed-level values, mean, sample SD,
Main mean, paired mean difference và two-sided 95% CI của paired difference.

Các detection metrics ablation tối thiểu gồm:

```text
validation bbox mAP@[0.50:0.95] = primary ablation metric
validation AP50
validation AP75
Q_pseudo = required secondary mechanistic metric
```

Không tạo formal hypothesis test riêng hoặc multiplicity family mới cho ablation.

`Q_pseudo` của ablation chỉ dùng fixed validation theo cùng Q_pseudo contract;
không dùng hidden-U GT và không dùng test.

---

## 28A.5. Hard test firewall

```text
condition_role = ABLATION
AND is_main = false
→ test_access_authorized = false
→ final_test directory must not exist
```

Không có “ablation winner → final test” sau khi xem validation.

---

# 29. MULTIPLICITY ARTIFACTS

```text
analysis/multiplicity/holm_F2.csv
analysis/multiplicity/holm_F3.csv
```

F2:

```text
RQ3 omnibus
RQ4 primary
RQ6 primary
RQ7 primary
m=4
Holm
```

F3:

```text
6 RQ3 pairwise contrasts
m=6
Holm
```

RQ2:

```text
family F1
m=1
no adjustment
```

---

# 30. ROBUSTNESS / SENSITIVITY ARTIFACTS

```text
analysis/robustness/
├── rq2_exact_sign_flip.json
├── rq2_loso.csv
├── rq6_exact_sign_flip.json
├── rq6_loso.csv
├── rq7_exact_sign_flip.json
├── rq7_loso.csv
├── rq3_friedman.json
├── rq4_cr2_sensitivity.json
└── rq4_seed_influence.csv
```

Exact sign-flip phải ghi rõ:

```text
2^10 = 1024 sign configurations

RQ2 = one-sided
RQ6 = two-sided
RQ7 = two-sided
```

Machine-readable exact sign-flip evidence cho mỗi RQ phải lưu tối thiểu:

```text
rq
effect
ordered_training_seeds
seed_effect_count = 10
test_statistic = arithmetic_mean_of_paired_seed_effects
observed_statistic
sign_configuration_count = 1024
observed_all_positive_configuration_included = true
alternative
extremeness_rule
equality_ties_included = true
plus_one_correction = false
zero_effect_seed_retained = true
duplicate_statistics_counted_by_configuration = true
extreme_configuration_count
exact_p_value
```

Semantics bắt buộc:

```text
RQ2:
  effect = G_s
  alternative = one-sided-greater
  extremeness_rule = T_flip >= T_obs
  exact_p_value = extreme_configuration_count / 1024

RQ6:
  effect = H_s
  alternative = two-sided
  extremeness_rule = abs(T_flip) >= abs(T_obs)
  exact_p_value = extreme_configuration_count / 1024

RQ7:
  effect = D_s
  alternative = two-sided
  extremeness_rule = abs(T_flip) >= abs(T_obs)
  exact_p_value = extreme_configuration_count / 1024
```

Nếu paired effect bằng đúng `0`, training seed đó vẫn phải xuất hiện trong
`ordered_training_seeds` và `seed_effect_count` vẫn bằng `10`. Không giảm số
configuration do các sign vectors tạo ra cùng một statistic.

Preflight implementation evidence của S6.11 phải được ghi machine-readable
trong:

```text
artifacts/preflight/statistics/statistical_fixture_report.json
```

và phải chứng minh implementation tuân thủ chính xác các semantics ở trên trước
khi S6.11 được CLOSED / PASS.

LOSO:

```text
exactly 10 analyses
```

Không dùng LOSO để loại seed.

RQ3:

```text
primary = GG repeated-measures ANOVA
sensitivity = Friedman
```

RQ4 sensitivity:

```text
linear model with within-cell Q predictor
CR2 covariance
cluster = training_seed
Satterthwaite df
```

Mỗi robustness/sensitivity artifact phải lưu script hash và statistical-environment
reference.

# 31. THESIS-READY ARTIFACTS

Không viết bảng luận văn trực tiếp từ console logs.

Pipeline:

```text
raw training evidence
→ official_training_summary.csv
→ final-test campaign
→ official_final_test_results.csv
→ paired/ablation/statistical analysis outputs
→ thesis-ready tables/figures
```

Suggested outputs:

```text
thesis_artifacts/
├── tables/
│   ├── supervised_baseline.tex
│   ├── ssl_vs_sup.tex
│   ├── rq2_primary_gain.tex
│   ├── rq3_budget_effect.tex
│   ├── rq4_qpseudo_association.tex
│   ├── rq5_rare_classes.tex
│   ├── rq6_negative_fp.tex
│   └── rq7_architecture_effect.tex
│
├── figures/
│   ├── supervised_vs_budget.pdf
│   ├── ssl_vs_sup_by_budget.pdf
│   ├── paired_ssl_gain.pdf
│   ├── qpseudo_vs_downstream_gain.pdf
│   ├── rare_class_gain.pdf
│   └── negative_fp.pdf
│
└── provenance/
    └── thesis_artifact_sources.csv
```

`thesis_artifact_sources.csv` phải ghi:

```text
thesis_artifact
source_analysis_file
source_result_file
generation_script
git_commit
sha256
```

---


## 31.1. Thesis statistical reporting format

Formatting của thesis-ready outputs phải bảo toàn reporting rules trong `sn-article.tex`:

```text
mAP/AP effects → AP points, 2 decimals
RQ4 slope      → AP-point outcome change per +1 pseudo AP point, 3 decimals
FP             → 3 decimals
FAR difference → percentage points, 2 decimals
p              → 3 decimals; nếu p < .001 thì ghi p < .001
```

Mọi official effect CI là **two-sided 95% CI** cho effect tương ứng; không được thay
bằng Holm-adjusted simultaneous CI.

Thứ tự diễn giải ưu tiên:

```text
effect estimate → two-sided 95% CI → p-value
```

Không sử dụng cụm `marginally significant`.

---

# 32. REPRODUCIBILITY → THESIS EVIDENCE MAP

| Nội dung cần viết trong luận văn | Artifact canonical |
|---|---|
| Canonical scientific source | `governance/scientific_source_manifest.json` |
| Vast/GPU/runtime environment | `preflight/environment/vast_environment.json` |
| Docker identity | `preflight/environment/docker_image_identity.json` |
| Dataset transfer integrity | `preflight/data/dataset_transfer_manifest.json` |
| Fixed split membership | `preflight/data/split_membership_audit.json` |
| L/U membership | `preflight/data/labeled_unlabeled_membership_audit.json` |
| Dataset/DataLoader path | `preflight/data/dataset_loader_manifest.json` |
| training seed không đổi membership | `preflight/seed/training_seed_invariance_audit.json` |
| Python/NumPy/PyTorch seed propagation | `seed_runtime.json` / preflight seed audit |
| DataLoader worker seeding | `preflight/seed/dataloader_worker_seed_audit.json` |
| deterministic runtime flags | `vast_environment.json`, `seed_runtime.json` |
| SUP/SSL entrypoint | `supervised_entrypoint_manifest.json`, `ssl_entrypoint_manifest.json` |
| augmentation pipelines | `supervised_augmentation_manifest.json`, `ssl_augmentation_manifest.json` |
| actual update accounting | `train.jsonl`, `training_summary.json`, `training_completion_report.json` |
| labeled-supervision exposure | `training_summary.json`, `official_pair_inventory.csv` |
| AMP skipped-step behavior | `runtime_events.jsonl`, `amp_ema_skip_test.json` |
| EMA timing | `ema_timing_test.json`, `runtime_events.jsonl` |
| image–GT bbox alignment | `gt_bbox_alignment_report.json` |
| image–pseudo-box alignment | `pseudo_bbox_alignment_report.json` |
| hidden-U GT firewall | `hidden_u_gt_firewall_report.json` |
| SUP–SSL config pairing | `sup_ssl_pairing_config_diff_report.json` |
| checkpoint rule | `checkpoint_index.json`, `validation_best*.json` |
| BEST/LAST provenance | `checkpoint_index.json` |
| Q_pseudo provenance | `qpseudo_validation.json` |
| COCO evaluator settings | `coco_evaluator_golden_test.json` |
| operating-point evaluator | `operating_point_golden_test.json` |
| final-test authorization | `final_test/FINAL_TEST_AUTHORIZATION.json` |
| test access audit | `final_test/test_access_audit.json` |
| official final-test results | `final_test/official_final_test_results.csv` |
| ablation provenance | `results/ablation/*`, `analysis/ablation/*` |
| statistical implementation versions | `statistics_environment.json`, `statistical_implementation_manifest.json` |
| retry/deviation | `retry_deviation.json` |
| official config identity | `run_manifest.json`, `resolved_config.yaml`, config hash |
| Git provenance | `run_manifest.json` |
| pilot/official separation | `GLOBAL_PREFLIGHT_REPORT.json`, `official_run_inventory.csv` |
| backup verification | per-attempt `backup_sync_manifest.json` |

# 33. STORAGE / RETENTION POLICY

## 33.1. Must retain permanently for scientific audit

```text
source/protocol hashes
frozen configs
environment evidence
data/membership hashes
run_manifest
resolved_config
seed_runtime
validation_history
training_summary
checkpoint_index
BEST checkpoint
LAST checkpoint
training_completion_report
final_evaluation_report khi applicable
artifact_manifest
official final-test metrics
operating-point metrics
class-wise AP
negative metrics
Q_pseudo artifact for SSL
official training/final-test master tables
ablation inventories/effects
statistical analysis outputs
thesis artifact provenance
retry/deviation record
```

---

## 33.2. Retain until run closure, then optional archive/delete

```text
temporary caches
download cache
intermediate non-BEST/non-LAST checkpoints
temporary visualization files
framework temporary work files
```

Chỉ xóa khi:

```text
training run CLOSED/VERIFIED
required artifacts verified
artifact_manifest hashed
BEST/LAST scientific checkpoints backed up
backup checksum verification PASS
```

---

## 33.3. Durable backup contract

Trong Vast.ai:

```text
active run
→ local/persistent storage
```

Sau run closure:

```text
scientific evidence + required checkpoints
→ second durable copy, e.g. Google Drive/project archive
```

Mỗi attempt đã đóng phải tạo:

```text
runs/<run_type>/<run_id>/<attempt_id>/backup_sync_manifest.json
```

tối thiểu:

```text
source path
destination identity
file count
bytes
SHA-256 per required scientific artifact hoặc manifest tree hash
copy timestamp
verification status
```

Không xóa local scientific artifact chỉ vì copy command thành công; phải verify hash.

Không coi ephemeral instance disk là nơi lưu duy nhất của official evidence.

# 34. TEST ACCESS AUDIT VÀ FINAL-TEST CAMPAIGN GATE

`TEST_FIREWALL_STATE.json` là **MANDATORY**.

Trước final-test authorization:

```json
{
  "test_access_authorized": false,
  "reason": "FINAL_ONLY"
}
```

Mọi training/pilot/ablation attempt phải có khả năng audit:

```text
test_accessed = false
```

---

## 34.1. Pre-final-test gate

Trước khi mở test phải tạo:

```text
artifacts/final_test/PRE_FINAL_TEST_GATE_REPORT.json
```

Gate chỉ PASS khi tối thiểu:

```text
[ ] 80/80 low-label SUP training runs CLOSED/VERIFIED
[ ] 20/20 100%-SUP reference training runs CLOSED/VERIFIED
[ ] 80/80 SSL training runs CLOSED/VERIFIED
[ ] all valid official seeds retained
[ ] no replacement seed
[ ] all BEST scientific checkpoints locked + hashed
[ ] all SSL LAST Teacher checkpoints needed for Q_pseudo locked + hashed
[ ] Q_pseudo validation artifacts complete
[ ] ablation validation campaign complete per prespecified OFAT scope
[ ] non-Main ablation final-test eligibility = false
[ ] official evaluator frozen + hashed
[ ] statistical implementation frozen + fixture-tested
[ ] no unresolved controlled revision
[ ] eligible model inventory complete
[ ] test firewall has zero unauthorized access events
```

---

## 34.2. Eligible model inventory

```text
artifacts/final_test/eligible_model_inventory.csv
```

Chỉ chứa:

```text
80 low-label SUP BEST checkpoints
20 100%-SUP BEST checkpoints
80 SSL BEST EMA Teacher checkpoints
```

Tổng:

```text
180 eligible Main/reference models
```

Không chứa non-Main ablation variants.

---

## 34.3. Final authorization

Chỉ sau pre-final gate PASS, researcher mới tạo/approve:

```text
FINAL_TEST_AUTHORIZATION.json
```

Tối thiểu ghi:

```text
authorization timestamp
scientific source hash
frozen protocol/config manifest hash
eligible model inventory hash
evaluator manifest hash
statistical implementation manifest hash
researcher authorization
```

---

## 34.4. Final-test campaign audit

```text
final_test_campaign_manifest.json
test_access_audit.json
```

Phải cho biết:

```text
exact eligible models evaluated
exact test split hash
checkpoint hashes
evaluator hash
evaluation completion state
any unauthorized access event
```

Nếu có unauthorized test access:

```text
FINAL TEST CAMPAIGN = FAIL / INVESTIGATE
```

# 35. OFFICIAL RUN FAIL-FAST REQUIREMENTS

Official training launcher phải block trước training nếu:

```text
membership mismatch
seed mismatch
config hash mismatch
scientific-source mismatch
condition_role mismatch
wrong class mapping
unexpected augmentation
wrong optimizer/update budget
wrong scheduler relative-position rule or milestones
wrong effective batch
hidden-U annotations detected
wrong EMA settings
wrong pseudo thresholds
wrong checkpoint policy
wrong evaluator config
pilot config not frozen/equivalent
test firewall not CLOSED
```

Final-test launcher phải block nếu:

```text
PRE_FINAL_TEST_GATE != PASS
FINAL_TEST_AUTHORIZATION != YES
run not in eligible_model_inventory
checkpoint hash mismatch
test split hash mismatch
evaluator hash mismatch
condition_role = ABLATION and variant != MAIN
```

Không chạy trước rồi audit sau.

# 36. COMPLETENESS CHECK TRƯỚC PHÂN TÍCH

## 36.1. Training completeness

Trước final-test gate:

```text
80/80 low-label SUP valid training runs
20/20 100%-SUP reference valid training runs
80/80 SSL valid training runs
```

Nếu technical failure:

```text
retry same exact training seed
new attempt_id
```

Nếu một run valid nhưng poor-performing:

```text
retain
```

Không replacement seed.

Phải tạo:

```text
results/completeness_report.json
```

---

## 36.2. Final-evaluation completeness

Trước RQ2–RQ7 downstream test analysis:

```text
180/180 eligible Main/reference final evaluations CLOSED/VERIFIED
```

Nếu chưa đủ 180/180 do lỗi kỹ thuật chưa được giải quyết:

```text
OFFICIAL ANALYSIS = BLOCKED
```

Phải hoàn tất technical retry bằng cùng exact training seed hoặc xử lý qua controlled
revision có researcher approval. Không silently drop unfavorable run, không replacement
seed và không tự tạo missingness rule hậu nghiệm.

Canonical final-test table:

```text
final_test/official_final_test_results.csv
```

---

## 36.3. Ablation completeness

Ablation phải có:

```text
10 unique configs
10 seeds/config
100 configuration–seed observations
90 additional trainings nếu Main 10 observations được reuse
```

Non-Main ablation:

```text
validation only
test access = 0
```

# 37. SCIENTIFIC CHECKPOINT ROLE SUMMARY

```text
SUP:
BEST → official detection evaluation
LAST → retained scientific endpoint / provenance
LATEST_RESUME → operational only

SSL:
BEST EMA Teacher → official detection evaluation
LAST EMA Teacher → Q_pseudo on fixed validation
LATEST_RESUME → operational only
```

Cấm:

```text
BEST Teacher → Q_pseudo
Student → official SSL result
test → checkpoint selection
```

---

# 38. REQUIRED SCHEMA VÀ RELATIONAL VALIDATION

Trước promotion, machine-readable artifacts phải được schema-validated.

Tối thiểu:

```text
JSON parse PASS
required fields present
enum fields valid
numeric ranges valid
hash fields valid
referenced paths exist
run identity fields consistent
seed fields consistent
condition_role consistent
```

CSV:

```text
expected columns present
unique key valid
no duplicate logical run
NA policy valid
```

Relational checks bắt buộc:

```text
run_manifest ↔ run_id directory match
seed_index ↔ locked training_seed mapping match
run ↔ data membership hashes match
SUP–SSL pair ↔ required-match hashes match
observed labeled exposure ↔ pair contract match
BEST checkpoint ↔ validation_best record match
LAST Teacher ↔ Q_pseudo checkpoint hash match
final-test run ↔ eligible_model_inventory match
final-test evaluator ↔ frozen evaluator hash match
ablation non-Main ↔ test access absent
thesis table/figure ↔ analysis source provenance match
```

# 39. VERSIONING

Tài liệu này đã được khóa với:

```text
artifact_contract_version = 1.1.0
schema_version = 1.1
status = RESEARCHER_APPROVED / LOCKED
```

Lock basis:

```text
scientific_source = sn-article.tex
implementation_contract = IMPLEMENTATION_HANDOFF.md
final_cross_check = PASS
audited_criteria = 88/88
```

Scientific protocol change:

```text
requires controlled revision
```

Pure schema/operational improvement không thay scientific meaning:

```text
may increment artifact contract version
must document backward compatibility
must not reinterpret existing scientific results
```

Sau official training start, mọi schema change phải giữ khả năng đọc/audit artifact
cũ hoặc có migration manifest rõ ràng.

# 40. MINIMUM DEFINITION OF DONE — PRE-OFFICIAL

Không được authorize official training nếu chưa có:

```text
[ ] Canonical scientific source = sn-article.tex + SHA-256
[ ] Scientific source manifest
[ ] Implementation contract manifest
[ ] Artifact contract manifest
[ ] Vast environment evidence
[ ] Dataset transfer integrity
[ ] Fixed split membership audit
[ ] L/U membership audit
[ ] dataset/DataLoader manifest
[ ] training-seed membership invariance
[ ] DataLoader/worker seed audit
[ ] SUP R50 preflight PASS
[ ] SUP Swin-T preflight PASS
[ ] SSL R50 preflight PASS
[ ] SSL Swin-T preflight PASS
[ ] 1% scheduler path PASS
[ ] 2064-update scheduler path PASS
[ ] 100%-SUP / 10284-update path evidence PASS
[ ] SUP/SSL entrypoint manifests
[ ] SUP/SSL augmentation manifests
[ ] SUP–SSL config pairing diff PASS
[ ] GT bbox geometry PASS
[ ] pseudo-box geometry PASS
[ ] Teacher initialization PASS
[ ] EMA timing PASS
[ ] AMP→EMA skip PASS
[ ] empty pseudo-label batch PASS
[ ] zero-GT PASS
[ ] hidden-U firewall PASS
[ ] BEST/LAST checkpoint PASS
[ ] Q_pseudo path PASS
[ ] COCO evaluator PASS
[ ] operating-point evaluator PASS
[ ] resume path PASS if official resume enabled
[ ] test-firewall preflight PASS
[ ] pilot/official isolation PASS
[ ] frozen executable configs + hashes
[ ] statistical synthetic/golden fixtures PASS
[ ] statistical environment + implementation manifest frozen
[ ] official fail-fast guardrails enabled
[ ] GLOBAL_PREFLIGHT_REPORT = PASS
[ ] Researcher authorization = YES
```

# 41. MINIMUM DEFINITION OF DONE — MỖI OFFICIAL TRAINING RUN

Một run chỉ được đưa vào `official_training_summary.csv` khi:

```text
[ ] correct official run_id
[ ] correct condition_role
[ ] correct training_seed
[ ] pre_run_assertions PASS
[ ] correct data hashes
[ ] correct config hash
[ ] expected optimizer updates reached
[ ] observed effective labeled batch correct
[ ] labeled-supervision exposure logged
[ ] no unresolved scientific deviation
[ ] validation history complete
[ ] BEST checkpoint present
[ ] LAST checkpoint present
[ ] checkpoint hashes present
[ ] training_completion_report PASS
[ ] artifact manifest complete
[ ] retry/deviation record present
[ ] test_accessed = false during training
[ ] SSL: Q_pseudo LAST-Teacher validation artifact present
[ ] durable backup verification PASS before deleting local scientific artifacts
```

Training completion **không** có nghĩa final test đã chạy.

Để đi vào `official_final_test_results.csv`, run còn phải:

```text
[ ] be listed in eligible_model_inventory.csv
[ ] FINAL_TEST_AUTHORIZATION = YES
[ ] final_evaluation_report PASS
```

# 42. MINIMUM DEFINITION OF DONE — THESIS ANALYSIS

```text
[ ] official training matrix completeness verified
[ ] final-test campaign gate PASS
[ ] 180 eligible Main/reference final evaluations verified
[ ] non-Main ablation test access = 0
[ ] paired SUP–SSL inventory valid
[ ] observed labeled-supervision exposure pairing valid
[ ] no cross-seed pooling
[ ] no excluded valid poor seed
[ ] no replacement seed
[ ] no imputed undefined AP
[ ] RQ1 output created
[ ] RQ2 output + one-sided exact sign-flip + LOSO created
[ ] RQ3 GG + contrasts + Friedman created
[ ] RQ4 primary + diagnostics + CR2 created
[ ] RQ5 rare-class summary created
[ ] RQ6 primary + FAR summary + two-sided sign-flip + LOSO created
[ ] RQ7 primary + two-sided sign-flip + LOSO created
[ ] Holm F2 created
[ ] Holm F3 created
[ ] ablation 10-config / 100-observation completeness verified
[ ] ablation validation-only outputs created
[ ] operating-point Recall + FP/image available
[ ] negative FP/FAR outputs available
[ ] thesis tables generated from analysis outputs
[ ] thesis figures generated from analysis outputs
[ ] thesis artifact provenance created
[ ] statistics environment/script hashes archived
```

# 43. FINAL DATA FLOW

```text
                 sn-article.tex
            APPROVED METHODOLOGY
                         │
                         ▼
              IMPLEMENTATION CONTRACT
                         │
                         ▼
              ARTIFACT/EVIDENCE CONTRACT
                         │
                         ▼
                 PRE-FLIGHT EVIDENCE
                         │
                  ALL CHECKS PASS
                         │
                         ▼
                 FROZEN EXECUTABLES
                         │
                         ▼
             RESEARCHER AUTHORIZATION
                         │
             ┌───────────┴───────────┐
             ▼                       ▼
       OFFICIAL SUP              OFFICIAL SSL
             │                       │
       training evidence         training evidence
             │                   + Q_pseudo
             └───────────┬───────────┘
                         ▼
              180 TRAINING RUNS CLOSED
                         │
                         ├──────────────► ABLATION
                         │                validation-only
                         │                10 configs / 100 observations
                         │                90 additional trainings if Main reused
                         │
                         ▼
               PRE-FINAL-TEST GATE
                         │
                  GATE PASS ONLY
                         │
                         ▼
               FINAL TEST AUTHORIZATION
                         │
                         ▼
            180 ELIGIBLE MODEL EVALUATIONS
                         │
                         ▼
          official_final_test_results.csv
                         │
                         ▼
                 SUP–SSL PAIR TABLE
                         │
                         ▼
              RQ1–RQ7 ANALYSIS FILES
                         │
                         ▼
         MULTIPLICITY / ROBUSTNESS / SENSITIVITY
                         │
                         ▼
               TABLES / FIGURES / CI
                         │
                         ▼
                     LUẬN VĂN
```

# 44. CONTRACT DECISION

Sau cross-check cuối cùng theo thứ tự ưu tiên:

```text
sn-article.tex
→ IMPLEMENTATION_HANDOFF.md
→ ARTIFACT_AND_EVIDENCE_CONTRACT.md
```

không còn unresolved scientific conflict, implementation-contract conflict hoặc
artifact/evidence gap thuộc phạm vi đã audit.

Trạng thái chính thức:

```text
ARTIFACT_AND_EVIDENCE_CONTRACT
= RESEARCHER_APPROVED / LOCKED

artifact_contract_version = 1.1.0
schema_version = 1.1
```

Lock basis:

```text
FINAL_CROSS_CHECK = PASS
AUDITED_CRITERIA = 88/88
SCIENTIFIC_SOURCE = sn-article.tex
SCIENTIFIC_SOURCE_SHA256 = a5f63ff018ee28a37ad2daa3fff337daf6b30faf34d9f12c6ff5c0fa92fe5237
IMPLEMENTATION_HANDOFF_SHA256 = d70e556d45f4a54a5e0498842b4932deef7991de0d986095ec85e546dcc2d026
```

Từ thời điểm này:

- code phải sinh đúng các artifact bắt buộc;
- official launcher không được bỏ qua evidence generation;
- test firewall và final-test campaign gate là bắt buộc;
- schema thay đổi phải có versioning;
- scientific value không được thay bằng sửa schema;
- mọi scientific change phải đi qua controlled revision và researcher approval;
- mọi official result phải truy ngược được tới source/config/data/seed/checkpoint/evaluator;
- mọi paired effect phải truy ngược được tới đúng SUP–SSL same-seed pair;
- mọi ablation result phải truy ngược được tới frozen Main same-seed reference;
- mọi bảng/figure luận văn phải truy ngược được tới official analysis artifacts.

**Mục tiêu khóa:** không có con số chính thức nào trong luận văn tồn tại mà không
thể truy ngược về canonical `sn-article.tex`, run, seed, config, data membership,
checkpoint, evaluator, statistical implementation và source artifact đã tạo ra nó.
