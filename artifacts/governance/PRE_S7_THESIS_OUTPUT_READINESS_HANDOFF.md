# PRE-S7 THESIS OUTPUT READINESS — OPERATIONAL HANDOFF

## 0. Mục đích

Tài liệu này là **operational handoff** cho các stage tiếp theo của project SSOD chest X-ray:

- S7 preflight/pilot;
- official supervised / semi-supervised training;
- final-test evaluation sau authorization;
- aggregation / statistical analysis;
- thesis artifact generation.

Tài liệu này **KHÔNG phải scientific authority mới**.

Thứ tự authority vẫn là:

1. `sn-article.tex`
2. `IMPLEMENTATION_HANDOFF.md`
3. `ARTIFACT_AND_EVIDENCE_CONTRACT.md`
4. management workbook chỉ dùng cho operational tracking, không phải technical gate.

Nếu handoff này mâu thuẫn với authority cao hơn, phải dừng và xử lý theo authority order; không tự chọn framework default.

---

# 1. Trạng thái project tại thời điểm handoff

```text
S0 = CLOSED / PASS
S1 = CLOSED / PASS
S2 = CLOSED / PASS
S3 = CLOSED / PASS
S4 = CLOSED / PASS
S5 = CLOSED / PASS
S6 = CLOSED / PASS
S7 = NOT STARTED
```

S6.01–S6.16 đã CLOSED/PASS.

Canonical main tại thời điểm mở PRE-S7 thesis-output readiness:

```text
main = origin/main
a438c8114e70f4341539ee28aba949f9296c1e10
```

Branch controlled hardening:

```text
governance/thesis-output-readiness
```

Boundary locks:

```text
GLOBAL_PREFLIGHT_CLOSED = False
OFFICIAL_TRAINING_AUTHORIZED = False
FINAL_TEST_AUTHORIZED = False
S7_P01_STARTED = False
```

---

# 2. Mục tiêu PRE-S7 THESIS OUTPUT READINESS

Mục tiêu là ngăn việc sau khi chạy official experiments mới phát hiện thiếu:

- raw prediction;
- seed-level result;
- checkpoint identity;
- image-level provenance;
- class-wise metrics;
- negative-image metrics;
- prediction artifacts cần dựng hình;
- generation-script provenance;
- dữ liệu cần tái dựng bảng/hình mà không retraining.

Nguyên tắc:

```text
PRE-S7 = khóa contract về những gì pipeline PHẢI sinh và PHẢI giữ
S7 = kiểm chứng pipeline có khả năng tuân thủ contract
OFFICIAL RUNS = sinh evidence thật
FINAL TEST = sinh final-test evidence thật sau authorization
THESIS GENERATOR = đọc evidence thật để tạo bảng/hình thật
```

Các file readiness hiện tại **không phải kết quả thí nghiệm**.

---

# 3. Current thesis-output readiness

Readiness matrix hiện đạt:

```text
MATRIX_ROWS = 14
MATRIX_PASS_COUNT = 14
MATRIX_GAP_COUNT = 0
```

Ý nghĩa:

```text
14/14 PASS
= contract/design readiness PASS
≠ runtime result artifacts đã tồn tại
≠ final thesis figures đã được tạo
```

Qualitative readiness:

```text
PASS_CONTRACT_COVERAGE_RUNTIME_DEFERRED
```

Overall PRE-S7 readiness vẫn **BLOCKED / IN_PROGRESS** cho tới khi governance re-lock, staged Git-blob identity verification, commit và independent post-commit verification hoàn tất.

---

# 4. Artifact contract controlled revision

`ARTIFACT_AND_EVIDENCE_CONTRACT.md` đang được controlled-revise theo phạm vi:

```text
controlled_revision = PRE_S7_THESIS_OUTPUT_READINESS
revision_scope = ARTIFACT_SCHEMA_RETENTION_PROVENANCE_ONLY
scientific_protocol_change = FALSE
backward_compatibility = ADDITIVE
```

Version:

```text
artifact_contract_version = 1.1.1
schema_version = 1.1
```

Không thay đổi:

- scientific estimand;
- model/training protocol;
- evaluation threshold;
- statistical hypothesis;
- final-test authorization semantics;
- hidden-U GT firewall;
- train/val/test membership;
- labeled/unlabeled membership.

---

# 5. Required future qualitative artifacts

Khi materialization gates tương ứng PASS, pipeline phải sinh:

```text
thesis_artifacts/qualitative/
├── qualitative_case_selection_manifest.csv
├── qualitative_figure_manifest.csv
└── qualitative_reconstruction_check.json
```

Không được tạo artifact giả trước khi có runtime evidence thật.

---

# 6. Deterministic qualitative case-selection rules

Case selection phải deterministic, pre-specified, prediction-independent và không cherry-pick.

## 6.1. FT_ABNORMAL_HASH_V1

```text
split = final_test
candidate_definition = images with >= 1 GT bbox
selection_count = 6
ordering_algorithm =
  ascending SHA256(ordering_salt + "|" + selection_rule_id + "|" + image_id)
ordering_salt = SSOD_THESIS_QUAL_V1
materialization_gate = FINAL_TEST_AUTHORIZED
prediction_independent = TRUE
```

Selection rule phải được khóa trước final-test authorization. Danh sách case thật chỉ được materialize sau authorization.

## 6.2. FT_NO_FINDING_HASH_V1

```text
split = final_test
candidate_definition = zero-GT images
selection_count = 3
ordering_algorithm =
  ascending SHA256(ordering_salt + "|" + selection_rule_id + "|" + image_id)
ordering_salt = SSOD_THESIS_QUAL_V1
materialization_gate = FINAL_TEST_AUTHORIZED
prediction_independent = TRUE
```

## 6.3. VAL_QPSEUDO_HASH_V1

```text
split = fixed_validation
candidate_definition = all images in fixed validation split
selection_count = 6
ordering_algorithm =
  ascending SHA256(ordering_salt + "|" + selection_rule_id + "|" + image_id)
ordering_salt = SSOD_THESIS_QUAL_V1
materialization_gate = SSL_RUN_CLOSED_VERIFIED
prediction_independent = TRUE
```

Không được dùng model predictions, prediction scores, error magnitude hoặc realized qualitative appearance để chọn case.

---

# 7. Permanent-retention requirements

## 7.1. detections.json

For every applicable official final-test evaluation:

```text
detections.json
```

phải được giữ vĩnh viễn cho scientific audit / evaluator recomputation / qualitative reconstruction.

Không được thêm threshold trước COCO AP.

## 7.2. pseudo/val_last_teacher_predictions.json

For official SSL runs:

```text
pseudo/val_last_teacher_predictions.json
```

phải được giữ vĩnh viễn sau run closure.

Mục đích:

- Q_pseudo audit;
- validation pseudo-label visualization;
- qualitative reconstruction;
- provenance back to LAST EMA Teacher.

---

# 8. S7 phải đọc gì trước khi implement

Trước mỗi task S7:

```text
1. sn-article.tex
2. IMPLEMENTATION_HANDOFF.md
3. ARTIFACT_AND_EVIDENCE_CONTRACT.md
4. PRE_S7_THESIS_OUTPUT_READINESS_HANDOFF.md
5. exact S7 tracker task
6. existing implementation / validator / evidence / reusable infrastructure
```

Handoff này chỉ là operational guide và không override authority 1–3.

---

# 9. S7 phải chứng minh gì

S7 không chỉ chứng minh model chạy được; phải chứng minh pipeline sinh đúng evidence và giữ đủ downstream thesis data.

Các nhóm kiểm tra chính:

- data/membership/seed invariance;
- DataLoader/sampler/worker determinism;
- SUP R50 / SUP Swin;
- SSL R50 / SSL Swin;
- scheduler semantics;
- GT geometry;
- pseudo geometry;
- teacher initialization;
- EMA timing;
- AMP skip ⇒ EMA skip;
- empty pseudo;
- zero-GT;
- hidden-U GT firewall;
- BEST/LAST;
- Q_pseudo;
- COCO evaluator;
- operating-point matching;
- resume;
- manifest/seed/provenance;
- pilot/official isolation;
- test firewall;
- frozen/hashable configs;
- no unresolved protocol warning.

Exact task order vẫn theo S7 tracker hiện hành.

---

# 10. Official training code phải sinh gì

Training-side pipeline phải materialize machine-readable runtime evidence, tối thiểu khi applicable:

```text
run_manifest.json
resolved_config.json
seed_runtime.json
runtime_events.jsonl

metrics/validation/
    validation_history.csv
    validation_best_teacher.json
    classwise_validation_ap.csv
    training_summary.json

checkpoints/
    checkpoint_index.json
    BEST checkpoint
    LAST checkpoint
    latest_resume.pth

training_completion_report.json
artifact_manifest.json
backup_sync_manifest.json
retry_deviation.json
```

Exact path/schema theo canonical artifact contract.

---

# 11. Official SSL phải sinh thêm gì

```text
pseudo/
├── qpseudo_validation.json
├── qpseudo_classwise.csv
├── val_last_teacher_predictions.json
└── pseudo_diagnostics.json
```

Locked semantics:

```text
Q_pseudo checkpoint = LAST EMA Teacher
Q_pseudo dataset = fixed validation
pseudo set = accepted RCNN classification pseudo labels
```

`val_last_teacher_predictions.json` là permanent scientific-retention evidence.

---

# 12. Final-test evaluation phải sinh gì

Không được materialize final-test artifacts trước authorization.

Sau authorization, applicable official evaluations phải sinh/giữ:

```text
coco_metrics.json
detections.json
classwise_ap.csv
negative_metrics.json
evaluation_manifest.json
```

`evaluation_manifest.json` phải truy được tối thiểu tới:

```text
run_id
checkpoint_role
checkpoint_sha256
evaluator_config_sha256
test_split_sha256
test_authorization_sha256
evaluation timestamp
```

Non-main ablation final-test vẫn bị cấm trừ khi canonical authority đổi.

---

# 13. Aggregation / statistical-analysis outputs

Downstream pipeline phải giữ canonical sources cho RQ1–RQ7 và robustness/sensitivity, ví dụ:

```text
official_training_summary.csv
official_final_test_results.csv
paired_effects.csv
classwise_ap.csv
negative_metrics.json
RQ1–RQ7 analysis artifacts
multiplicity artifacts
robustness artifacts
sensitivity artifacts
ablation inventories/effects
```

Statistical semantics vẫn theo `sn-article.tex` và S6. Handoff này không thêm test thống kê mới.

---

# 14. Thesis Output Matrix

14 output groups đang được cover:

```text
RQ1_QUANT
RQ2_QUANT
RQ3_QUANT
RQ4_QUANT
RQ5_QUANT
RQ6_QUANT
RQ7_QUANT
ABLATION_QUANT
QUAL_DETECTION
QUAL_NO_FINDING
QUAL_PSEUDO
QUAL_SELECTION
QUAL_PROVENANCE
QUAL_RECONSTRUCTION
```

Không ép mỗi RQ phải có đúng một bảng và một hình.

---

# 15. Qualitative figure-manifest schema

`qualitative_figure_manifest.csv` phải tối thiểu có:

```text
figure_id
figure_path
panel_id
panel_role
split
image_id
image_sha256
selection_rule_id
selection_rank
gt_source_file
gt_source_sha256
model_role
architecture
labeled_budget
run_id
seed
checkpoint_role
checkpoint_sha256
prediction_source_file
prediction_source_sha256
evaluator_or_acceptance_config_sha256
generation_script
generation_script_sha256
render_config_sha256
figure_sha256
```

Nếu panel không dùng model prediction, các field model/run/checkpoint/prediction phải để trống, không được suy đoán.

Panel dùng prediction phải trỏ tới retained raw prediction artifact đúng run/checkpoint.

---

# 16. Qualitative reconstruction requirements

`qualitative_reconstruction_check.json` phải chứng minh figure được regenerate không cần retraining.

Minimum schema:

```json
{
  "schema_version": "1.0",
  "status": "PASS|FAIL",
  "case_selection_manifest_sha256": "...",
  "figure_manifest_sha256": "...",
  "generation_script_sha256": "...",
  "render_config_sha256": "...",
  "all_required_source_files_present": true,
  "all_recorded_source_hashes_match": true,
  "retraining_required": false,
  "figure_regeneration_completed": true,
  "recorded_figure_sha256": "...",
  "regenerated_figure_sha256": "...",
  "exact_figure_hash_match": true
}
```

PASS chỉ khi source files đủ, hashes khớp, không retraining, regenerate thành công và figure hash khớp.

---

# 17. Thesis generator sẽ làm gì

Thesis generator không đọc console làm source of truth.

Conceptual flow:

```text
official training artifacts
        ↓
official final-test artifacts
        ↓
aggregation/statistical artifacts
        ↓
qualitative case-selection manifest
        ↓
raw retained predictions + images + GT
        ↓
THESIS GENERATOR
        ↓
tables
figures
real chest X-ray qualitative panels
provenance manifests
reconstruction checks
```

Ví dụ thesis-ready outputs đã được contract nhắc tới:

```text
supervised_baseline.tex
ssl_vs_sup.tex
rq2_primary_gain.tex
rq3_budget_effect.tex
rq4_qpseudo_association.tex
rq5_rare_classes.tex
rq6_negative_fp.tex
rq7_architecture_effect.tex

supervised_vs_budget.pdf
ssl_vs_sup_by_budget.pdf
paired_ssl_gain.pdf
qpseudo_vs_downstream_gain.pdf
rare_class_gain.pdf
negative_fp.pdf
```

---

# 18. Real chest X-ray qualitative outputs dự kiến

Khi dữ liệu thật đã tồn tại, có thể sinh:

```text
GT overlay trên X-quang thật

GT vs SUP vs SSL
trên cùng ảnh được chọn deterministic

No Finding behavior
với SUP/SSL predictions

validation pseudo-label visualization
từ LAST EMA Teacher
```

Đây là descriptive evidence, không tạo endpoint/hypothesis mới.

---

# 19. Reusable visual infrastructure hiện có

```text
scripts/02D1E_post_coco_visual_overlay_audit.py
reports/phase2D1E_post_coco_visual_overlay_manifest.csv
reports/phase2D1E_post_coco_visual_overlay_audit.json
reports/phase2D1B_pilot_reference_viewer_manifest.csv
reports/phase2D1B_pilot_visual_audit_manifest.csv
```

Có thể reuse cho:

- image identity;
- JPEG path resolution;
- GT overlay;
- COCO consistency;
- bbox geometry;
- image/annotation linkage;
- machine-readable visual audit.

Chưa có model prediction overlay đầy đủ; cần reuse + extend.

Không tự động lấy Phase 2D.1E selected cases làm result figures vì selection objective trước đó là geometry/data audit.

---

# 20. Current authoritative identities

Đã audit:

```text
sn-article.tex
Git-blob SHA256 =
b6881ce90da9194f88fd9664b51dc96bdf112145949e3a89705f16b06853c681

IMPLEMENTATION_HANDOFF.md
Git-blob SHA256 =
d274f98402e0492039a9da0bdfed538102fc6ca155f4f7870525c55c46aa0e1d
```

Current controlled-revision worktree SHA256 observed for:

```text
ARTIFACT_AND_EVIDENCE_CONTRACT.md
222270bfe62f64074ec1cc9c0436a68bd3b0766b92900a1457511ba4535d4ccc
```

Không được coi hash worktree trên là final committed Git-blob identity trước staged/commit verification.

---

# 21. Governance-manifest re-lock state

Ba governance manifests đã được refresh trong working tree với:

```text
hash_basis = GIT_BLOB_CONTENT_SHA256
controlled_revision = PRE_S7_THESIS_OUTPUT_READINESS
```

Expected changes:

```text
scientific_source_manifest.json
→ scientific_source_sha256
→ + hash_basis
→ + controlled_revision

implementation_contract_manifest.json
→ implementation_contract_sha256
→ scientific_source_sha256
→ + hash_basis
→ + controlled_revision

artifact_contract_manifest.json
→ artifact_contract_version = 1.1.1
→ artifact_contract_sha256
→ scientific_source_sha256
→ implementation_contract_sha256
→ + hash_basis
→ + controlled_revision
```

Không được silently remove field cũ.

---

# 22. Current PRE-S7 working state

Tại thời điểm handoff được tạo, expected working tree đang chứa:

```text
M  ARTIFACT_AND_EVIDENCE_CONTRACT.md
M  artifacts/governance/artifact_contract_manifest.json
M  artifacts/governance/implementation_contract_manifest.json
M  artifacts/governance/scientific_source_manifest.json
?? artifacts/governance/PRE_S7_THESIS_OUTPUT_READINESS_REPORT.json
?? artifacts/governance/THESIS_OUTPUT_READINESS_MATRIX.csv
```

Handoff này sẽ trở thành thêm một explicit path khi copy vào repo.

Không dùng:

```text
git add .
```

Chỉ stage các path đã audit.

---

# 23. PRE-S7 readiness artifacts

```text
artifacts/governance/THESIS_OUTPUT_READINESS_MATRIX.csv
artifacts/governance/PRE_S7_THESIS_OUTPUT_READINESS_REPORT.json
```

Matrix:

```text
rows = 14
PASS = 14
GAP = 0
```

Report tại thời điểm handoff:

```text
status = BLOCKED
```

BLOCKED là intentional cho tới khi:

1. governance identities khớp staged/committed bytes;
2. handoff được include;
3. exact path set được audit;
4. controlled revision commit được tạo;
5. independent post-commit verification PASS;
6. worktree clean;
7. readiness report được finalized nhất quán.

---

# 24. Những gì không được làm

Không:

- reopen S0–S6 nếu không có contradictory evidence;
- rebuild train/val/test;
- rebuild labeled/unlabeled membership;
- dùng hidden-U GT;
- truy cập final-test GT/predictions trước authorization;
- materialize final-test qualitative case IDs trước authorization;
- chọn qualitative cases theo model appearance;
- dùng console logs làm thesis source of truth;
- xóa permanent prediction artifacts;
- invent missing provenance;
- đổi scientific protocol để hợp default framework;
- dùng workbook làm technical gate;
- dùng `git add .`;
- push mid-stage nếu chưa được researcher yêu cầu;
- merge main bằng CLI nếu không được yêu cầu.

---

# 25. Operational workflow rules cho stage sau

```text
ONE operational step
ONE operational command
review full output
PASS / FAIL / BLOCKED
stop on FAIL/BLOCKED
```

Trước mỗi S7 task:

```text
Verification 0 = Git/repo state
Verification 1 = exact tracker task
Verification 2 = canonical contract / authority hierarchy
```

Trước khi tạo implementation:

```text
audit existing implementation
audit validator
audit evidence
audit schema
audit reusable infrastructure
```

Decision gap:

```text
STOP / BLOCKED
```

Không tự chọn framework default.

---

# 26. Khi vào S7

1. Confirm PRE-S7 THESIS OUTPUT READINESS = CLOSED/PASS.
2. Confirm governance branch đã merge vào canonical `main`.
3. Record **new canonical main commit** sau merge.
4. Create `implementation/s7` local từ new canonical main.
5. Không reuse old anchor `a438c811...` sau khi PRE-S7 merge.
6. Re-read authority order.
7. Re-read handoff này.
8. Verify exact S7 tracker order.
9. Start `S7.P01` only after identity checks PASS.

---

# 27. Khi vào official training

Trước official training:

- S7 CLOSED/PASS;
- GLOBAL_PREFLIGHT_REPORT PASS;
- frozen configs verified;
- training seed protocol verified;
- artifact-generation hooks enabled;
- permanent-retention paths configured;
- backup/checksum workflow verified;
- final test vẫn closed.

Training không đủ bằng chứng chỉ vì loss giảm hoặc process exit 0.

Machine-readable evidence phải tồn tại và validate.

---

# 28. Khi vào final-test evaluation

Trước final test:

- official training closure;
- pre-final-test gate PASS;
- explicit final-test authorization;
- authorization provenance recorded.

Chỉ sau đó mới:

- materialize final-test evaluation artifacts;
- materialize final-test qualitative case IDs theo rules đã khóa.

Selection vẫn phải prediction-independent.

---

# 29. Khi build thesis outputs

Thesis generation phải là reproducible downstream build.

Một table/figure chỉ được chấp nhận khi truy ngược được:

```text
canonical methodology
→ implementation contract
→ artifact contract
→ data membership
→ run
→ seed
→ config
→ checkpoint
→ evaluator/statistical script
→ source result artifact
→ thesis generation script
→ final artifact hash
```

Qualitative figure phải thêm:

```text
selection rule
→ selection rank
→ image_id
→ image SHA
→ GT source
→ prediction source
→ checkpoint
→ render config
→ figure SHA
```

---

# 30. Handoff acceptance checklist

```text
[ ] PRE-S7 contract diff verified
[ ] thesis-output matrix = 14/14 PASS
[ ] 6 qualitative gaps covered
[ ] qualitative rules locked
[ ] prediction retention locked
[ ] governance manifests semantically audited
[ ] exact staged path set verified
[ ] staged Git-blob SHA verified
[ ] artifact manifest matches staged/committed contract hash
[ ] readiness report finalized consistently
[ ] handoff added explicitly
[ ] controlled revision committed
[ ] independent post-commit verification PASS
[ ] worktree clean
[ ] branch not pushed unless authorized
```

Chỉ khi tất cả PASS:

```text
PRE_S7_THESIS_OUTPUT_READINESS = CLOSED / PASS
```

---

# 31. Core interpretation

```text
PRE-S7 defines what evidence must exist.
S7 proves the pipeline can produce and preserve it.
Official runs produce the real evidence.
Final-test evaluation produces authorized test evidence.
Thesis generator consumes the retained real evidence.
```

Không thesis table/figure nào được phụ thuộc vào dữ liệu mà official pipeline đã không lưu.

Không qualitative chest X-ray figure nào nên buộc phải retrain chỉ vì raw prediction/provenance đã bị bỏ.

---

# 32. Handoff status

```text
HANDOFF_TYPE = OPERATIONAL
AUTHORITY_LEVEL = NON_CANONICAL_GUIDE
SCIENTIFIC_PROTOCOL_CHANGE = FALSE
S7_P01_STARTED = FALSE
FINAL_TEST_ACCESSED = FALSE
PRE_S7_THESIS_OUTPUT_READINESS = CLOSED / PASS
```

Controlled PRE-S7 revision commit `fd0aed9fe8b46f0ac45ec01c7ba5a5bf29f93961` đã được independent post-commit verification PASS. Handoff này hiện mang trạng thái operational closure `CLOSED / PASS`.
