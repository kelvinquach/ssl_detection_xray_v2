# MASTER DECISION RECORD D1–D7
## Chuẩn bị triển khai thí nghiệm SSOD trên X-quang phổi

**Đề tài:** *Nghiên cứu học bán giám sát cho dò tìm bất thường trên X-quang phổi*  
**Mục đích:** Nguồn tham chiếu trung tâm trước khi implement/training, tổng hợp các quyết định nghiên cứu D1–D7 đã được researcher phê duyệt hoặc chủ động defer.  
**Ngôn ngữ:** Tiếng Việt  
**Ngày tổng hợp:** 26/08/2026
**Trạng thái tài liệu:** `FINAL IMPLEMENTATION CONTRACT`
**Ngày phát hành bản final contract:** 28/08/2026


---

# 0. NGUYÊN TẮC SỬ DỤNG FILE

File này là **Implementation Master Decision Record**.

Nó dùng để:

1. biết **quyết định nào đã khóa** và không được tự ý thay đổi;
2. ánh xạ quyết định nghiên cứu sang config/code/guardrail/artifact;
3. biết điều gì **chưa được phép execute**;
4. chuẩn bị các phase training theo đúng experimental design;
5. làm nguồn để sau này viết Methodology/Experimental Protocol;
6. phân biệt rõ:
   - `RESEARCHER_APPROVED / LOCKED`;
   - `OPEN / AWAITING PREFLIGHT`;
   - `DEFERRED`;
   - `PROHIBITED`;
   - `CONTROLLED REVISION REQUIRED`.

> Nếu code/config thực tế xung đột với file này, **không được tự sửa protocol theo code**. Phải xác định đây là implementation bug hay cần Controlled Revision.

---

# 1. MASTER STATUS D1–D7

| Gate | Nội dung | Trạng thái |
|---|---|---|
| D1 | Research Questions & Hypotheses | `RESEARCHER_APPROVED / LOCKED` |
| D2 | Research Variables | `RESEARCHER_APPROVED / LOCKED` |
| D3 | Experimental Matrix & Comparison Design | `RESEARCHER_APPROVED / LOCKED` |
| D4-A | Detector / Architecture Selection | `RESEARCHER_APPROVED / LOCKED` |
| D4-B | Base Training Configuration | `RESEARCHER_APPROVED / LOCKED` |
| D4-C | Main SSL / SoftTeacher Treatment Contract | Protocol `LOCKED`; implementation preflight `NOT YET EXECUTED` |
| D4-D | Ablation Study | Research design `D4-D.1–D4-D.8 = RESEARCHER_APPROVED / LOCKED`; `D4-D.9 = DEFERRED TO IMPLEMENTATION / TECHNICAL PREFLIGHT` |
| D5 | Evaluation Metrics / Metric Protocol | Protocol `LOCKED`; implementation preflight `NOT YET EXECUTED` |
| D6 | Statistical Analysis Protocol | Protocol `LOCKED`; implementation preflight `NOT YET EXECUTED` |
| D7 | Reproducibility Contract | **DEFERRED UNTIL AFTER TRAINING**; protocol chưa thiết kế/khóa |

Research-scientific source-of-truth alignment:

```text
vietluanvan.md =
CURRENT SCIENTIFIC SOURCE OF TRUTH

RESEARCH DESIGN D1–D6 =
COMPLETE / RESEARCHER APPROVED / LOCKED

D4-D.1–D4-D.8 =
RESEARCHER APPROVED / LOCKED

D4-D.9 =
DEFERRED TO IMPLEMENTATION / TECHNICAL PREFLIGHT
```

Implementation must inherit these scientific values exactly. Any mismatch requires either an implementation fix or an explicit Controlled Revision.

Current firewall state:

```text
D4-C_IMPLEMENTATION_PREFLIGHT = NOT YET EXECUTED
D4-C_PHASE_STATUS = OPEN / AWAITING PREFLIGHT
OFFICIAL_SSL_TRAINING_AUTHORIZED = FALSE

D5_IMPLEMENTATION_PREFLIGHT = NOT YET EXECUTED
D5_PHASE_STATUS = OPEN / AWAITING PREFLIGHT

D6_IMPLEMENTATION_PREFLIGHT = NOT YET EXECUTED
D6_PHASE_STATUS = OPEN / AWAITING PREFLIGHT

TEST_STATUS = CLOSED / NOT AUTHORIZED
```

---

# 2. EXECUTIVE IMPLEMENTATION VIEW

## 2.1. Những gì đã cố định

```text
Dataset:
- fixed train / validation / test
- train = 3426
- val = 734
- test = 734
- 14 abnormality detection classes
- No Finding = negative images, NOT a detection class

Labeled budgets:
- 1%
- 5%
- 10%
- 20%

Architectures:
- Faster R-CNN + ResNet-50 + FPN
- Faster R-CNN + Swin-T + FPN

Methods:
- SUP
- SSL teacher–student / SoftTeacher-style

Training seeds:
- exact ordered list of 10 locked seeds
- same ordered seed indices for SUP / SSL
- no seed search

Primary downstream metric:
- bbox mAP@[0.50:0.95]

Final test:
- final-only
- not authorized yet
```

## 2.2. Main official low-label experiment

\[
2\ methods
\times
4\ budgets
\times
2\ architectures
\times
10\ seeds
=
\boxed{160\ official\ low-label\ runs}
\]

100%-SUP reference:

\[
2\ architectures
\times
10\ seeds
=
20
\]

Total pre-ablation main/reference runs:

\[
\boxed{180}
\]

D4-D ablation:

- 10 configurations;
- R50 only;
- 10% labeled budget;
- all 10 locked seeds;
- 100 observations total;
- if 10 Main R50/10% SSL runs reused: 90 additional runs.

---

# 3. D1 — RESEARCH QUESTIONS & HYPOTHESES

**Status:** `RESEARCHER_APPROVED / LOCKED`

## 3.1. RQ1 — Supervised baseline performance

**Role:**

```text
DESCRIPTIVE
NO FORMAL HYPOTHESIS
```

Câu hỏi:

> Hiệu năng supervised baseline thay đổi như thế nào tại các labeled budgets 1%, 5%, 10%, 20%?

Mục tiêu:

- mô tả label-efficiency baseline;
- tạo paired reference cho SSL;
- không biến budget-specific baseline differences thành primary inferential hypothesis.

---

## 3.2. RQ2 — SSL vs SUP

**Role:**

```text
PRIMARY
DIRECTIONAL
ONE OVERALL HYPOTHESIS
```

Không phải 4 primary hypotheses theo từng budget.

Formal direction:

\[
H_0:\text{overall average SSL gain}\le 0
\]

\[
H_1:\text{overall average SSL gain}>0
\]

Primary metric:

\[
\boxed{mAP@[0.50:0.95]}
\]

Statistical implementation được khóa tại D6.2.

---

## 3.3. RQ3 — Label-budget effect on SSL gain

**Role:**

```text
SECONDARY
FORMAL
NON-DIRECTIONAL
```

Câu hỏi:

> SSL gain so với SUP có thay đổi theo label budget 1%, 5%, 10%, 20% hay không?

Không prespecify:

- linear trend;
- monotonic trend;
- “SSL chắc chắn có lợi hơn khi ít nhãn”.

Statistical implementation: D6.3.

---

## 3.4. RQ4 — Pseudo-label quality association

**Role:**

```text
SECONDARY
MECHANISTIC
DIRECTIONAL
POSITIVE ASSOCIATION
FORMALLY LOCKED
```

Pseudo-label quality:

\[
\boxed{
Q_{\text{pseudo}}
=
PL\text{-}mAP@[0.50:0.95]
}
\]

Scope:

```text
GT scope = fixed validation
checkpoint = LAST EMA Teacher
hidden U GT = PROHIBITED
```

Formal scientific statement:

> Pseudo-label quality cao hơn được hypothesize có **positive association** với downstream SSL gain.

Không được viết causal:

```text
Qpseudo causes downstream improvement
```

Statistical implementation: D6.5.

---

## 3.5. RQ5 — Rare classes

**Role:**

```text
EXPLORATORY
```

Frozen rare classes from Phase 3A:

```text
Atelectasis
Pneumothorax
```

Không confirmatory hypothesis.

Không rare-mAP official endpoint.

---

## 3.6. RQ6 — No Finding / false positives

**Role:**

```text
SECONDARY
FORMAL
NON-DIRECTIONAL
```

Câu hỏi:

> SSL có làm thay đổi false-positive burden trên các ảnh No Finding / zero-GT hay không?

Primary within-RQ6 metric được D5/D6 khóa là:

\[
\boxed{FP/\text{confirmed-negative image}}
\]

---

## 3.7. RQ7 — Architecture-dependent SSL effect

**Role:**

```text
SECONDARY
FORMAL
NON-DIRECTIONAL
```

Câu hỏi:

> Incremental SSL gain có phụ thuộc architecture conventional CNN (R50) so với hierarchical Vision Transformer (Swin-T) hay không?

Không phải hypothesis:

```text
Swin-T > R50
```

Không claim general architecture superiority.

---

# 4. D2 — RESEARCH VARIABLES

**Decision:** `D2-R1`  
**Status:** `RESEARCHER_APPROVED / LOCKED`

## 4.1. Independent variables

### X1 — Training paradigm

```text
SUP
SSL
```

Role:

\[
\boxed{PRIMARY\ FACTOR}
\]

---

### X2 — Label budget

```text
1%
5%
10%
20%
```

Role:

```text
SECONDARY
ORDERED CATEGORICAL DESIGN FACTOR
```

D6.3 primary implementation treats budget **categorically**, not as numeric regression.

---

### X3 — Architecture

```text
R50 conventional CNN
Swin-T hierarchical Vision Transformer
```

Role:

```text
SECONDARY / MODERATOR
```

---

## 4.2. Dependent variables

Bảy dependent variables đã xác định:

1. \(mAP@[0.50:0.95]\)
2. AP50
3. AP75
4. AP theo từng class
5. Recall
6. FP/image
7. FP/negative image

D5 xác định metric hierarchy/operating definitions.

---

## 4.3. Interactions

```text
Method × Budget
= SECONDARY / FORMAL
```

```text
Method × Architecture
= SECONDARY / FORMAL
```

```text
Budget × Architecture
= SECONDARY / EXPLORATORY
```

```text
Method × Budget × Architecture
= SECONDARY / EXPLORATORY
```

Same ordered training seeds và matched SUP–SSL là bắt buộc.

---

## 4.4. Control variables / matched conditions

Các so sánh SUP–SSL phải giữ matched/frozen ở mức phù hợp:

- same fixed dataset scope;
- same fixed train/val/test;
- same labeled budget;
- same architecture within paired comparison;
- same preprocessing/model input representation;
- same detector family;
- same evaluation protocol;
- same ordered seed index;
- same total optimizer-update budget;
- same labeled-supervision exposure;
- chỉ SSL được dùng thêm unlabeled stream.

---

# 5. D3 — EXPERIMENTAL MATRIX & COMPARISON DESIGN

**Status:** `RESEARCHER_APPROVED / LOCKED`

## 5.1. Main low-label factorial design

\[
2\ methods\times4\ budgets\times2\ architectures=16\ cells
\]

Mỗi cell:

\[
10\ seeds
\]

Total:

\[
\boxed{160\ official\ low-label\ runs}
\]

---

## 5.2. 100%-SUP reference

```text
R50 × 100%-SUP × 10 seeds
Swin-T × 100%-SUP × 10 seeds
```

Total:

\[
20
\]

Role:

```text
REFERENCE / CONTEXT
```

Không đưa 100%-SUP vào paired RQ2/RQ3/RQ4/RQ6/RQ7 estimands.

---

## 5.3. Phase execution mapping

```text
Phase 4A = R50 SUP
Phase 4B = Swin-T SUP

Phase 5A = R50 SSL
Phase 5B = Swin-T SSL
```

---

## 5.4. Pairing

SUP và SSL paired theo:

```text
architecture
budget
seed_index
training_seed
```

Pairing không được dựa vào row order hoặc performance.

---

## 5.5. Compute-budget fairness

Giữa matched SUP/SSL:

```text
same total optimizer-update budget
same labeled-supervision exposure
```

SSL additionally consumes U.

Mục tiêu:

> isolate effect của unlabeled-data/SSL treatment thay vì cho SSL đơn giản train lâu hơn.

---

## 5.6. Test firewall

```text
validation:
- checkpoint selection
- D4-D ablation
- Qpseudo
- pre-final development

test:
- FINAL ONLY
```

---

## 5.7. Hidden-U GT firewall

Hidden ground truth của \(U_b\):

```text
PROHIBITED FOR:
- training
- pseudo-label filtering
- threshold tuning
- model selection
```

---

# 6. D4-A — DETECTOR / ARCHITECTURE SELECTION

**Status:** `RESEARCHER_APPROVED / LOCKED`

Official architecture pair:

\[
\boxed{
A_{\text{baseline}}
=
\text{Faster R-CNN + ResNet-50 + FPN}
}
\]

\[
\boxed{
A_{\text{ViT/Attention}}
=
\text{Faster R-CNN + Swin-T + FPN}
}
\]

Terminology:

```text
ResNet-50-based conventional CNN architecture

Swin-T-based hierarchical Vision Transformer architecture
```

Same detection family:

```text
same Faster R-CNN family
same FPN role
same RPN/RoI family
```

Mục đích:

> giảm confounding detector-family khi so architecture/moderation của SSL.

---

## 6.1. Pretraining

```text
R50 backbone = ImageNet-1K pretrained
Swin-T backbone = ImageNet-1K pretrained

COCO detector-level initialization = PROHIBITED
```

---

## 6.2. Framework

Official framework:

```text
MMDetection
COCO detection JSON
```

Detectron2:

```text
FALLBACK_ONLY
```

YOLO:

```text
OUT OF SCOPE
```

Không model-shopping sau khi xem result.

---

## 6.3. Main software references

MMDetection v3.3.0:

https://github.com/open-mmlab/mmdetection/tree/v3.3.0

Base Faster R-CNN R50 FPN:

https://github.com/open-mmlab/mmdetection/blob/v3.3.0/configs/_base_/models/faster-rcnn_r50_fpn.py

Swin integration example:

https://github.com/open-mmlab/mmdetection/blob/v3.3.0/configs/swin/mask-rcnn_swin-t-p4-w7_fpn_1x_coco.py

Swin Transformer paper:

https://openaccess.thecvf.com/content/ICCV2021/html/Liu_Swin_Transformer_Hierarchical_Vision_Transformer_Using_Shifted_Windows_ICCV_2021_paper.html

---

# 7. D4-B — BASE TRAINING CONFIGURATION

**Status:** `RESEARCHER_APPROVED / LOCKED`

## 7.1. Input channel contract

Stored input:

```text
JPEG L / grayscale / 1 channel
```

Model input:

\[
\boxed{
1ch
\rightarrow
3\ identical\ channels
}
\]

\[
R=G=B.
\]

Không sử dụng color-information invention.

---

## 7.2. Image scale

Train/validation/test:

```text
scale = (1333, 800)
keep_ratio = True
pad size divisor = 32
```

---

## 7.3. Labeled augmentation

Conservative supervised stream:

```text
Load
→ Resize
→ Pack
```

Không default:

```text
flip
crop
rotation
mosaic
aggressive photometric augmentation
```

---

## 7.4. Effective labeled batch

\[
\boxed{
B^L_{\mathrm{eff}}=4
}
\]

Microbatch/gradient accumulation là execution detail để đạt đúng effective batch, không được âm thầm đổi scientific batch exposure.

---

## 7.5. R50 optimizer

```text
Optimizer = SGD
lr = 0.005
momentum = 0.9
weight_decay = 0.0001
```

---

## 7.6. Swin-T optimizer

```text
Optimizer = AdamW
lr = 0.000025
betas = (0.9, 0.999)
weight_decay = 0.05
native no-decay rules retained
```

---

## 7.7. Official optimizer-update budgets

```text
1% = 1032 updates

5%  = 2064
10% = 2064
20% = 2064

100%-SUP = 10284
```

---

## 7.8. Scheduler

For 2064-update schedule:

```text
drop 1 ≈ update 1376
drop 2 ≈ update 1892
```

1% schedule uses proportional locations.

---

## 7.9. Validation interval

```text
validation every 172 optimizer updates
```

1% schedule maintains comparable number of validation checkpoints.

---

## 7.10. Checkpoint selection

```text
BEST selected by validation bbox mAP@[0.50:0.95]

retain:
- BEST
- LAST
```

---

## 7.11. Numerical/training rules

```text
AMP = ON
gradient clipping = OFF unless Controlled Revision
all backbone stages = TRAINABLE
native normalization = retained
```

No Finding zero-GT images retained:

```text
filter_empty_gt = False
or exact equivalent
```

---

# 8. D4-C — MAIN SSL TREATMENT CONTRACT

**Method:** SoftTeacher-style end-to-end teacher–student pseudo-labeling  
**Status:** Protocol `RESEARCHER_APPROVED / LOCKED`; implementation preflight `NOT YET EXECUTED`

Official wording:

> Khung semi-supervised object detection teacher–student dựa trên pseudo-labeling, triển khai theo SoftTeacher.

Không claim:

```text
new algorithm
new SSOD method
best CXR SSL method
```

MMDetection semi-supervised guide:

https://mmdetection.readthedocs.io/en/v3.3.0/user_guides/semi_det.html

---

## 8.1. Student / Teacher

```text
Student = optimized by gradient descent
Teacher = EMA model
same architecture
```

Initialization:

```text
teacher(t=0) = student(t=0)
```

Không supervised checkpoint burn-in.

Resume:

```text
restore teacher state
DO NOT resync teacher from student
```

---

## 8.2. EMA convention

MMDetection-style locked convention:

```text
MeanTeacherHook momentum = 0.001
skip_buffers = True
```

Equivalent:

\[
\theta_T
=
0.999\,\theta_T^{old}
+
0.001\,\theta_S.
\]

EMA update:

```text
once per actual Student optimizer update
```

AMP skipped optimizer step:

```text
EMA SKIP too
```

---

## 8.3. Data streams

Labeled:

```text
L true GT -> Student
```

Unlabeled:

```text
same U image
weak view -> Teacher
strong view -> Student
```

Hidden U GT prohibited.

---

## 8.4. Weak U augmentation

```text
Resize only
+ standard preprocessing/padding
```

No:

```text
flip
crop
rotation
photometric augmentation
```

---

## 8.5. Strong U augmentation

```text
Resize
+
grayscale brightness U(0.90, 1.10)
+
grayscale contrast U(0.90, 1.10)
```

Constraint:

\[
R=G=B
\]

preserved.

No geometric transform/erasing/mosaic/etc.

---

## 8.6. Pseudo-label source

```text
ONLINE pseudo labels
Teacher RCNN detections
```

Core pseudo label:

```text
box
class
score
```

RPN proposals không được xem là final pseudo labels.

---

## 8.7. Thresholds

```text
detector score floor = 0.05

initial pseudo score threshold = 0.50

RPN pseudo threshold = 0.90

classification pseudo threshold = 0.90
```

---

## 8.8. NMS / max proposals

```text
RPN:
IoU = 0.70
max = 1000

RCNN:
IoU = 0.50
max = 100
```

---

## 8.9. Regression pseudo filtering

```text
jitter_times = 10
jitter_scale = 0.06

reg_pseudo_thr = 0.02

min_pseudo_bbox_wh = (0.01, 0.01)
```

Regression branch initial filter:

```text
initial_score > 0.50
AND
reg_uncertainty < 0.02
```

Important:

```text
regression branch does NOT require cls score > 0.90
```

---

## 8.10. Loss

Supervised:

```text
standard Faster R-CNN supervised loss
sup_weight = 1
```

Total:

\[
\boxed{
L_{total}
=
L_{sup}
+
4L_{unsup}
}
\]

```text
unsup_weight = 4
```

---

## 8.11. Burn-in / ramp-up

```text
burn_in = 0
lambda_unsup = 4 constant
```

No ramp-up.

---

## 8.12. L:U sampling ratio

\[
\boxed{L:U=1:1}
\]

Effective per optimizer update:

```text
4 labeled
+
4 unlabeled
```

---

## 8.13. SSL official evaluation model

```text
EMA Teacher
```

BEST:

```text
selected by validation Teacher mAP@[0.50:0.95]
```

Final test:

```text
EMA Teacher from locked BEST checkpoint
```

Student:

```text
SECONDARY
```

---

## 8.14. RQ4 Qpseudo checkpoint

Always:

```text
LAST EMA Teacher
fixed validation
```

Không BEST Teacher cho Qpseudo.

---

## 8.15. D4-C implementation preflight

Phải kiểm:

- empty pseudo-label case;
- zero-GT images;
- weak/strong geometry consistency;
- gradient accumulation vs optimizer update;
- EMA timing;
- AMP skipped-step behavior;
- resume restores Student+Teacher consistently;
- hidden-U GT firewall.

Current state:

```text
D4-C_IMPLEMENTATION_PREFLIGHT = NOT YET EXECUTED
OFFICIAL_SSL_TRAINING_AUTHORIZED = FALSE
```

---

# 9. D4-D — ABLATION STUDY

**Research status:** `D4-D.1–D4-D.8 = RESEARCHER_APPROVED / LOCKED`  
**Implementation status:** `D4-D.9 = DEFERRED TO IMPLEMENTATION / TECHNICAL PREFLIGHT`

Scientific source:

```text
vietluanvan.md
```

The implementation layer must not change the scientific decisions below.

---

## 9.1. D4-D.1 — Objective / Role / Governance

```text
SECONDARY
MECHANISTIC COMPONENT ANALYSIS
ONE-FACTOR-AT-A-TIME
NOT HYPERPARAMETER SEARCH
NOT MODEL SELECTION
NOT MAIN-PROTOCOL OPTIMIZATION
```

Governance:

```text
MAIN D4-C = FROZEN SCIENTIFIC ANCHOR

ABLATION RESULTS =
CANNOT RETROACTIVELY CHANGE MAIN D4-C

NEGATIVE RESULTS =
KEEP / REPORT

BEST-ABLATION SHOPPING =
PROHIBITED
```

Implementation consequence:
- never overwrite Main D4-C config from ablation results;
- never drop a valid variant because performance is poor;
- never mark one ablation variant as the new official Main without Controlled Revision.

---

## 9.2. D4-D.2 — Main Anchor / OFAT Invariants

Main anchor:

```text
Architecture = Faster R-CNN + ResNet-50 + FPN
Budget = 10%
Seeds = all 10 locked training seeds
Main SSL = locked D4-C configuration
```

OFAT contract:

```text
ONE TARGET FACTOR CHANGES
ALL OTHER SCIENTIFIC SETTINGS = IDENTICAL TO MAIN
```

Reuse:

```text
REUSE VALID MAIN R50/10% RUNS
DO NOT RETRAIN MAIN JUST TO CREATE A NEW ABLATION ANCHOR
```

unless prior Main runs are technically/protocol invalid.

Implementation guardrail:
- generated ablation configs should be diff-checked against Main;
- only allow fields belonging to the active ablation family to differ.

---

## 9.3. D4-D.3 — Scope

```text
Architecture = R50 only
Label budget = 10% only
Seeds = all 10 locked training seeds
```

No Swin-T ablation in the official D4-D scope.

---

## 9.4. D4-D.4 — Confidence-threshold ablation

\[
cls\_pseudo\_thr
\in
\{0.5,0.6,0.7,0.8,0.9\}
\]

```text
C0 = 0.5
C1 = 0.6
C2 = 0.7
C3 = 0.8
C4 = 0.9 = MAIN
```

Only `cls_pseudo_thr` changes.

All other D4-C scientific settings remain fixed.

---

## 9.5. D4-D.5 — Strong photometric augmentation ablation

```text
A0 = none
A1 = brightness only
A2 = contrast only
A3 = brightness + contrast = MAIN
```

Magnitude:

```text
brightness factor ∈ [0.90,1.10]
contrast factor   ∈ [0.90,1.10]
```

Constraint:

\[
R=G=B.
\]

No additional geometric augmentation, erasing, or mosaic.

---

## 9.6. D4-D.6 — Pseudo-label filtering ablation

```text
F0 =
initial score > 0.50
no extra bbox-regression reliability filtering

F1 =
confidence-based filtering
score > 0.90

F2 =
initial score > 0.50
AND reg_uncertainty < 0.02
= MAIN
```

Keep Main regression-reliability constants where applicable:

```text
jitter_times = 10
jitter_scale = 0.06
reg_pseudo_thr = 0.02
```

---

## 9.7. Excluded ablations

```text
EMA ablation = NOT INCLUDED
burn-in/ramp-up ablation = NOT INCLUDED
```

Do not add them during implementation without Controlled Revision.

---

## 9.8. Run count

Unique scientific configurations:

\[
\boxed{10}
\]

With 10 seeds:

\[
\boxed{100\ config\text{-}seed\ observations}
\]

If the 10 Main R50/10% runs are reused:

\[
\boxed{90\ additional\ training\ runs}
\]

Do not double-count Main C4/A3/F2 as separate physical Main runs.

---

## 9.9. D4-D.7 — Ablation Metrics / Qpseudo Diagnostics

### Primary ablation downstream outcome

\[
\boxed{
validation\ bbox\ mAP@[0.50:0.95]
}
\]

### Secondary downstream

```text
AP50
AP75
```

### Key mechanistic secondary outcome

\[
\boxed{
Q_{\mathrm{pseudo}}
=
PL\text{-}mAP@[0.50:0.95]
}
\]

Qpseudo contract:

```text
fixed validation
LAST EMA Teacher
accepted RCNN classification pseudo set
```

Additional mechanistic diagnostics may include:

```text
PL-AP50
PL-AP75
PL-Precision@IoU0.50
PL-Recall@IoU0.50
PL-F1@IoU0.50
mean matched IoU
pseudo-label count
retention/acceptance rate
pseudo FP / confirmed-negative validation image
regression uncertainty
```

Important:

```text
Qpseudo is NOT co-primary
study-wide primary endpoint remains unchanged
```

Implementation consequence:
- evaluator must produce primary validation mAP plus locked Qpseudo outputs for every valid ablation run.

---

## 9.10. D4-D.8 — Reporting / Statistical / Interpretation Contract

Ablation analysis is:

```text
ESTIMATION-FOCUSED
PAIRED-SEED ANALYSIS
NO FORMAL ABLATION P-VALUE
NO NEW MULTIPLICITY FAMILY
NO CHANGE TO D6 CONFIRMATORY FAMILIES
```

For metric \(M\):

\[
\Delta M_{v,s}
=
M_{v,s}
-
M_{\mathrm{Main},s}
\]

For Qpseudo:

\[
\Delta Q_{v,s}
=
Q_{v,s}
-
Q_{\mathrm{Main},s}
\]

Pair by:

```text
seed_index
training_seed
```

Required report per variant:

```text
variant mean
sample SD (ddof=1)
Main mean
paired mean difference
two-sided 95% CI of paired difference
seed-wise paired values/plot
direction and magnitude
```

No official ablation p-values.

Do not:
- create a new Holm family for ablation;
- call an ablation variant "statistically significant";
- modify D6 F1/F2/F3.

---

## 9.11. Validation / test use

```text
D4-D NON-MAIN =
VALIDATION ONLY

FINAL TEST =
PROHIBITED
```

Do not use test to choose:
- confidence threshold;
- augmentation;
- filtering;
- best ablation.

Main D4-C final-test treatment remains unchanged.

---

## 9.12. D4-D.9 — Implementation / Technical Preflight Gate

Research decision:

```text
DEFERRED TO IMPLEMENTATION
```

This gate should later validate at minimum:

### Universal

```text
correct R50 / 10% / locked seeds
Main scientific source loaded
only target factor differs
all non-target scientific settings identical
run IDs deterministic
artifact/provenance captured
```

### Confidence

```text
correct cls_pseudo_thr level
initial/RPN/reg thresholds unchanged
```

### Augmentation

```text
A0/A1/A2/A3 exact
weak branch unchanged
geometry unchanged
R=G=B preserved
```

### Filtering

```text
F0/F1/F2 exact
classification branch unchanged unless intended
NMS unchanged
F0 truly bypasses regression reliability filter
F2 uses locked reg_uncertainty path
```

Current:

```text
D4-D.9 IMPLEMENTATION PREFLIGHT =
NOT YET EXECUTED

OFFICIAL ABLATION EXECUTION =
NOT YET AUTHORIZED
```

Research design itself is already complete/locked; only implementation validation remains.

---

## 9.13. Final D4-D implementation handoff

```text
D4-D.1 = LOCKED
D4-D.2 = LOCKED
D4-D.3 = LOCKED
D4-D.4 = LOCKED
D4-D.5 = LOCKED
D4-D.6 = LOCKED
D4-D.7 = LOCKED
D4-D.8 = LOCKED

D4-D.9 =
IMPLEMENTATION-DEFERRED

D4-D RESEARCH DESIGN =
COMPLETE / RESEARCHER APPROVED / LOCKED
```

# 10. D5 — EVALUATION METRICS / METRIC PROTOCOL

**Status:** Protocol `RESEARCHER_APPROVED / LOCKED`  
Implementation preflight: `NOT YET EXECUTED`

---

## 10.1. Metric hierarchy

Study-wide primary:

\[
\boxed{bbox\ mAP@[0.50:0.95]}
\]

General secondary:

```text
AP50
AP75
Recall
FP/image
all 14 class AP
```

RQ4:

```text
Qpseudo = RQ4-specific primary mechanistic metric
but study-wide SECONDARY
```

RQ5:

```text
Atelectasis AP
Pneumothorax AP
```

RQ6:

```text
primary within RQ6 = FP/negative image
secondary = negative-image FAR
```

---

## 10.2. COCO evaluation

Official:

```text
bbox only
COCOeval / MMDetection-compatible
IoU = 0.50:0.05:0.95
recall thresholds = 0.00:0.01:1.00
useCats = 1
maxDets effective = 100
```

Class AP:

```text
all 14 classes
```

No official COCO APs/m/l because Phase 3A size bins use different normalized-area definitions.

No extra score threshold for AP.

---

## 10.3. Rare-class metric

Frozen rare:

```text
Atelectasis
Pneumothorax
```

Official:

\[
AP_c@[0.50:0.95]
\]

Paired:

\[
\Delta AP_c=SSL-SUP
\]

No rare-mAP.

No class-specific checkpoint selection.

---

## 10.4. Recall

General official secondary recall:

```text
COCO AR@[0.50:0.95]
maxDets = 100
area = all
```

Additional diagnostic:

```text
Recall@IoU=0.50
at fixed operating threshold
```

---

## 10.5. FP operating point

Global score threshold:

\[
\boxed{\tau_{\mathrm{eval}}=0.50}
\]

Matching:

```text
category-aware
one-to-one
same class
IoU >= 0.50
after detector-native NMS/max100
then apply score threshold 0.50
```

FP/image:

\[
FP/image
=
\frac{\sum_i FP_i}{N_{\mathrm{all\ eval\ images}}}
\]

Paired:

\[
\Delta FP=SSL-SUP
\]

Negative value = improvement.

---

## 10.6. No Finding

Fixed zero-GT subset:

```text
validation = 75
test = 75
```

No Finding is **not** a category.

Primary:

\[
FP/negative
=
\frac{\sum_{i\in\mathcal N}FP_i}{|\mathcal N|}
\]

Secondary:

\[
FAR_{neg}
=
\frac{\#\{i\in\mathcal N:FP_i\ge1\}}{|\mathcal N|}
\]

---

## 10.7. Qpseudo

\[
\boxed{
Q_{\text{pseudo}}
=
PL\text{-}mAP@[0.50:0.95]
}
\]

Scope:

```text
fixed validation
LAST EMA Teacher
accepted RCNN classification pseudo set
```

Prohibited:

```text
Qpseudo on test
hidden-U GT
```

Secondary Qpseudo diagnostics may include:

```text
PL-AP50
PL-AP75
Precision@IoU0.5
Recall@IoU0.5
F1@IoU0.5
mean matched IoU
pseudo count
retention
class PL-AP
PL-FP/val-negative
reg_uncertainty
```

80 main SSL rows.

---

## 10.8. Seed aggregation

Atomic observation:

```text
valid trained model / seed
```

Per official condition:

\[
n=10
\]

Report:

```text
mean
sample SD
```

\[
ddof=1
\]

No:

```text
cross-seed prediction pooling
performance outlier removal
seed replacement
metric imputation
```

---

## 10.9. Evaluation artifacts

Persistent prediction JSON:

```text
image_id
category_id
bbox = xywh
score
```

Store/hash:

```text
checkpoint
prediction
GT
config
category mapping
environment
code
metric output
```

Raw full precision metrics are source of truth.

---

## 10.10. Test firewall

Validation:

```text
BEST selection
development metrics
Qpseudo
D4-D
```

Test:

```text
FINAL ONLY
```

Main/reference scope:

\[
180\ official\ evaluations
\]

SSL final-test model:

```text
BEST EMA Teacher
```

D4-D non-main:

```text
NO TEST
```

---

## 10.11. D5 preflight

D5 cannot close until implementation checks PASS, including:

- toy evaluator;
- real structural evaluator smoke;
- both architectures;
- SUP and SSL Teacher predictions;
- zero-GT handling;
- Qpseudo smoke;
- exact frozen GT JSON;
- prediction schema;
- score threshold;
- category mapping;
- maxDets/IoU settings;
- artifact hashes.

Current:

```text
D5_IMPLEMENTATION_PREFLIGHT = NOT YET EXECUTED
TEST_STATUS = CLOSED
```

---

# 11. D6 — STATISTICAL ANALYSIS PROTOCOL

**Status:** D6.1–D6.13 `RESEARCHER_APPROVED / LOCKED`  
Implementation preflight: `NOT YET EXECUTED`

> Hồ sơ chi tiết riêng đã xuất:
>
> `D6_STATISTICAL_ANALYSIS_PROTOCOL_DECISION_EVIDENCE_RECORD_VI.md`
>
> File này chỉ giữ các quyết định cần trực tiếp cho implementation.

---

## 11.1. D6.1 — Statistical unit

Inferential replication unit:

\[
\boxed{training\ seed / trained\ run}
\]

\[
n=10
\]

Images/bboxes/detections không phải independent training-run replicates.

Primitive:

\[
\Delta M_{a,b,s}
=
M^{SSL}_{a,b,s}
-
M^{SUP}_{a,b,s}
\]

Effect scale:

```text
absolute metric difference
```

\[
\alpha=.05
\]

Estimation-first + 95% CI.

---

## 11.2. D6.2 — RQ2

Per seed:

\[
\boxed{
G_s
=
\frac18
\sum_{a,b}
\Delta mAP_{a,b,s}
}
\]

Equal weights:

```text
2 architectures × 4 budgets
each cell = 1/8
```

Formal:

\[
H_0:\mu_G\le0
\]

\[
H_1:\mu_G>0
\]

Primary:

```text
one-sample t-test
n=10
df=9
one-sided p
two-sided 95% CI
```

Cell effects reported descriptively.

---

## 11.3. D6.3 — RQ3

Architecture-balanced gain:

\[
B_{b,s}
=
\frac{
\Delta_{R50,b,s}
+
\Delta_{Swin,b,s}
}{2}
\]

Budget:

```text
categorical repeated factor
1%, 5%, 10%, 20%
```

Formal:

```text
one-way repeated-measures ANOVA
Greenhouse-Geisser ALWAYS
```

Mauchly:

```text
DIAGNOSTIC ONLY
```

Six prespecified contrasts:

```text
5-1
10-1
20-1
10-5
20-5
20-10
```

---

## 11.4. D6.4 — RQ7

Architecture gain:

\[
A_{a,s}
=
\frac14\sum_b\Delta_{a,b,s}
\]

Contrast:

\[
\boxed{
D_s
=
A_{Swin,s}
-
A_{R50,s}
}
\]

Formal:

```text
two-sided one-sample t-test
n=10
df=9
```

Not architecture-superiority test.

---

## 11.5. D6.5 — RQ4

80 rows:

\[
2\ arch\times4\ budget\times10\ seed
\]

Outcome:

\[
Y_{a,b,s}
=
mAP^{SSL,test}_{a,b,s}
-
mAP^{SUP,test}_{a,b,s}
\]

Predictor:

\[
Q_{\text{pseudo,val}}
\]

Within-cell centered:

\[
Q^{WC}_{a,b,s}
=
Q_{a,b,s}
-
\bar Q_{a,b}
\]

Reporting scale:

\[
X=Q^{WC}/0.01
\]

Primary model:

\[
\boxed{
\Delta mAP_{test}
\sim
Q^{WC}_{pseudo}
+
architecture\_budget\_cell
+
(1|training\_seed)
}
\]

Estimation/inference:

```text
REML
Kenward-Roger
```

Hypothesis:

\[
H_0:\beta_Q\le0
\]

\[
H_1:\beta_Q>0
\]

No causal wording.

---

## 11.6. D6.6 — RQ5

Frozen:

```text
Atelectasis
Pneumothorax
```

For each class:

\[
G_{c,s}
=
\frac18
\sum_{a,b}\Delta AP_{c,a,b,s}
\]

Official:

```text
mean paired gain
sample SD
two-sided 95% CI
NO required formal p
```

No rare-mAP.

---

## 11.7. D6.7 — RQ6

Primitive:

\[
\Delta F_{a,b,s}
=
FP^{SSL}_{neg}
-
FP^{SUP}_{neg}
\]

Per seed:

\[
H_s
=
\frac18\sum_{a,b}\Delta F_{a,b,s}
\]

Formal:

```text
two-sided one-sample t-test
n=10
df=9
```

Negative effect = fewer false positives.

FAR:

```text
estimation-focused
no official formal p
```

---

## 11.8. D6.8 — Multiplicity

### Family F1

```text
RQ2 only
m = 1
no adjustment
```

### Family F2

```text
RQ3
RQ4
RQ6
RQ7

m = 4
Holm step-down
FWER alpha = 0.05
```

Inputs:

```text
RQ3 = p_GG
RQ4 = one-sided KR p
RQ6 = two-sided p
RQ7 = two-sided p
```

### Family F3

Six RQ3 pairwise contrasts:

```text
Holm
m = 6
```

Formal RQ3 localization claim requires:

```text
RQ3 omnibus survives F2
AND
pair survives F3
```

No family shrinking for missing p-value.

---

## 11.9. D6.9 — Reporting

Principle:

\[
\boxed{
effect
\rightarrow
95\%CI
\rightarrow
p
}
\]

Official effect size:

```text
unstandardized absolute effect
```

mAP/AP:

```text
report as AP points
2 decimals
```

RQ4 slope:

```text
AP-point outcome change
per +1 pseudo-label AP point
3 decimals
```

FP:

```text
3 decimals
```

FAR:

```text
percentage-point difference
2 decimals
```

p:

```text
3 decimals
p < .001
never p = 0.000
```

CIs:

```text
two-sided 95%
not Holm-adjusted simultaneous CIs
```

---

## 11.10. D6.10 — Robustness

Primary methods frozen.

Shapiro-Wilk:

```text
DIAGNOSTIC ONLY
```

Q-Q:

```text
MANDATORY
```

No Shapiro-based test switching.

### RQ2/RQ6/RQ7

Exact sign-flip:

\[
2^{10}=1024
\]

```text
RQ2 = one-sided
RQ6 = two-sided
RQ7 = two-sided
```

LOSO mandatory.

Flags:

```text
SIGN_STABLE_ACROSS_LOSO
SEED_INFLUENCE_SENSITIVE
```

### RQ3

```text
primary = GG-RM ANOVA
robustness = Friedman
```

Friedman does not replace primary.

### RQ4

Diagnostics:

```text
convergence
gradient/Hessian
rank
singularity
residual-vs-fitted
Q-Q
seed influence
```

Singular:

```text
FLAG
NO automatic random-effect removal
```

Unresolved non-estimability:

```text
HARD FAIL
```

Robustness:

```text
linear model:
Y ~ Q_wc + Cell

CR2 cluster-robust covariance
cluster = training_seed
Satterthwaite df
```

CR2 p does **not** enter F2.

---

## 11.11. D6.11 — Missing / invalid runs

Core:

\[
\boxed{
run\ validity
\neq
performance
}
\]

States:

```text
VALID_OFFICIAL
TECHNICAL_FAILURE_RETRYABLE
PROTOCOL_INVALID
MISSING_OUTPUT / MISSING_METRIC
CRITICAL_PROTOCOL_VIOLATION
```

Valid-but-poor:

```text
KEEP
NO RETRY
```

Technical retry:

```text
SAME EXACT TRAINING SEED
```

Seed replacement:

```text
PROHIBITED
```

Repair hierarchy:

```text
metric bug -> reevaluate
prediction bug -> reinference + evaluate
training invalid -> same-seed retraining
```

No imputation.

No n=9 official analysis.

First authorized technically valid attempt = official.

---

## 11.12. D6.12 — Test firewall

One-way flow:

```text
LOCK
→ PREFLIGHT
→ FREEZE
→ AUTHORIZE TEST
→ FROZEN TEST CAMPAIGN
→ FREEZE TEST ARTIFACTS
→ LOCKED STATISTICS
→ REPORT
```

Before authorization:

```text
TEST PERFORMANCE = PROHIBITED
TEST STRUCTURAL ACCESS = ALLOWED
```

After unblinding:

```text
NO model change
NO checkpoint change
NO threshold change
NO metric change
NO hypothesis change
NO direction change
NO statistical model change
NO multiplicity change
NO rare-set redefinition
NO new confirmatory subgroup
```

Post-hoc allowed only when labeled:

```text
POST-HOC / EXPLORATORY / HYPOTHESIS-GENERATING
```

---

## 11.13. D6.13 — Statistical preflight

Protocol approved, but implementation still pending.

13 hard blocks:

```text
P1  Protocol integrity
P2  Statistical environment
P3  Official matrix / seeds
P4  Pairing / estimands
P5  RQ2
P6  RQ3
P7  RQ4
P8  RQ5/RQ6/RQ7
P9  Multiplicity
P10 Robustness
P11 Run-state handling
P12 Reporting / firewall
P13 Reproducibility / idempotence
```

Closure:

\[
\boxed{
P1\land\cdots\land P13=PASS
}
\]

Golden fixtures include:

```text
RQ2 known t/CI/p
exact sign-flip 1024
Holm known vector
GG forced path
KR path
sign-aware one-sided RQ4 p
LOSO exactly 10
CR2 seed clustering
firewall negative tests
two-run deterministic reproducibility
```

---

# 12. D7 — REPRODUCIBILITY CONTRACT

## 12.1. Researcher decision on timing

Current researcher decision:

```text
D7 = DEFERRED UNTIL AFTER TRAINING
```

Reason:

> Sau khi train xong sẽ có đầy đủ bằng chứng và cấu hình thực tế để xây Reproducibility Contract theo evidence thật.

Important:

```text
D7 scientific protocol = NOT YET DESIGNED
D7 protocol = NOT YET LOCKED
```

Không được ghi:

```text
D7 CLOSED / PASS
```

---

## 12.2. What should still be preserved during training

Dù D7 formalization được defer, **không xóa/ghi đè bằng chứng gốc**.

Nên giữ, nếu pipeline có sinh:

```text
config used
training log
training seed
run/attempt identifiers
BEST checkpoint
LAST checkpoint
Git commit
environment capture
dataset/L/U manifest
validation predictions
metrics
failure/retry evidence
```

Điều này không phải mở D7 sớm; chỉ là bảo toàn evidence để sau training có thể xây D7.

---

# 13. EXACT TRAINING SEED LIST

Ordered list đã khóa:

```text
seed_index 01 = 204886845
seed_index 02 = 1480646854
seed_index 03 = 1798418854
seed_index 04 = 2045683682
seed_index 05 = 1814859839
seed_index 06 = 1603952859
seed_index 07 = 1878351743
seed_index 08 = 875651179
seed_index 09 = 477581743
seed_index 10 = 869675675
```

Rules:

```text
same ordered list for SUP and SSL
no seed search
no replacement seed
partition_seed != training_seed
```

Data partition seed remains:

```text
partition_seed = 42
```

---

# 14. LABEL-BUDGET MANIFEST FACTS

Fixed train:

\[
|T|=3426
\]

Nested labeled/unlabeled:

| Budget | L | U | No Finding in L |
|---|---:|---:|---:|
| 1% | 34 | 3392 | 3 |
| 5% | 171 | 3255 | 17 |
| 10% | 343 | 3083 | 35 |
| 20% | 685 | 2741 | 70 |

Rules:

```text
L_b exact
U_b = complement
L budgets nested
No Finding subset nested
hidden U GT prohibited
```

---

# 14.1. SCIENTIFIC SOURCE-OF-TRUTH SYNC

Implementation master must remain synchronized with:

```text
vietluanvan.md
```

Locked research state:

\[
\boxed{
\text{RESEARCH DESIGN D1–D6}
=
\text{COMPLETE / RESEARCHER APPROVED / LOCKED}
}
\]

Implementation may add:

```text
config paths
script paths
assertions
unit/integration tests
artifact paths
hashes
runtime environment
preflight status
run-state metadata
```

Implementation may **not** independently modify scientific values.

Required equivalence examples:

```text
Scientific: cls_pseudo_thr = 0.90
Implementation: config field must equal 0.90

Scientific: Qpseudo = LAST EMA Teacher on fixed validation
Implementation: evaluator/checkpoint selector must enforce that exact rule

Scientific: D4-D analysis = no formal p-value
Implementation: ablation reporting code must not emit an official hypothesis-test p-value

Scientific: D4-D non-main = validation only
Implementation: test dispatcher must reject non-main ablation test evaluation
```

Any mismatch:

```text
IMPLEMENTATION BUG
or
CONTROLLED REVISION REQUIRED
```

---

# 15. IMPLEMENTATION READINESS CHECKLIST

## 15.1. Shared protocol layer

Before training code proliferation, create/verify one canonical source for:

```text
dataset paths/manifests
budget definitions
seed list
architecture IDs
method IDs
training update budgets
metric IDs
test firewall state
```

Recommended:

```text
configs/protocol/
  experimental_design.yaml
  phase2F1_seed_protocol.yaml
  d4_training_protocol.yaml
  d4_ssl_protocol.yaml
  d5_metric_protocol.yaml
  d6_statistical_protocol.yaml
```

Do not duplicate constants differently in multiple scripts.

---

## 15.2. Model configs

Prepare four core families:

```text
R50 SUP
Swin-T SUP
R50 SSL
Swin-T SSL
```

Budget/seed should be injected as controlled variables, not hand-edited copies.

Recommended naming:

```text
configs/models/r50_sup.py
configs/models/swint_sup.py
configs/models/r50_softteacher.py
configs/models/swint_softteacher.py
```

Budget-specific data config may be generated/read from locked manifests.

---

## 15.3. Run identity

Even though D7 is deferred, implementation should at minimum create deterministic run IDs:

```text
{ARCH}_{METHOD}_B{BUDGET}_S{SEED_INDEX}
```

Examples:

```text
R50_SUP_B01_S01
R50_SSL_B10_S06
SWINT_SUP_B20_S09
```

For 100%-SUP:

```text
R50_SUP_B100_S01
SWINT_SUP_B100_S01
```

This is recommended implementation organization, not a new scientific decision.

---

## 15.4. Supervised baseline preparation

Prepare:

```text
Phase 4A:
R50 × budgets 1/5/10/20 × 10 seeds
+ R50 100%-SUP × 10 seeds

Phase 4B:
Swin-T × budgets 1/5/10/20 × 10 seeds
+ Swin-T 100%-SUP × 10 seeds
```

Expected SUP count:

\[
4\times10\times2+20=100
\]

official SUP/reference training runs.

---

## 15.5. SSL preparation

Before official SSL:

```text
D4-C preflight MUST PASS
```

Then:

```text
Phase 5A:
R50 SSL × 4 budgets × 10 seeds
= 40

Phase 5B:
Swin-T SSL × 4 budgets × 10 seeds
= 40
```

Total main SSL:

\[
80
\]

---

## 15.6. Ablation preparation

Only after:

```text
Main R50/10% SSL anchor is available
AND
D4-D.9 technical preflight has been implemented and PASS
```

Scientific design D4-D.1–D4-D.8 is already locked.

Unique configurations:

```text
confidence: 0.5,0.6,0.7,0.8,0.9
augmentation: A0,A1,A2,A3
filter: F0,F1,F2
```

Do not double-count Main.

Non-main ablation:

```text
validation only
```

---

## 15.7. Evaluator implementation

Before final test:

```text
D5 preflight MUST PASS
```

Recommended modules:

```text
evaluation/
  coco_bbox_eval.py
  operating_point_eval.py
  negative_image_eval.py
  pseudo_label_eval.py
  prediction_export.py
```

Do not reconstruct GT at runtime from mutable sources; use exact frozen JSON/manifests.

---

## 15.8. Statistical implementation

Before test unblinding:

```text
D6 P1-P13 MUST PASS
```

Recommended split:

```text
statistics/
  build_seed_level_effects.py
  rq2_primary.py
  rq3_budget.R
  rq4_lmm_kr.R
  rq5_rare.py
  rq6_negative.py
  rq7_architecture.py
  multiplicity.py
  robustness.py
  reporting.py
```

Official RQ4 KR should remain R-based unless a verified equivalent is demonstrated.

---

# 16. ĐỀ XUẤT CÁCH IMPLEMENT TỐT HƠN: THEO EXECUTION GATES, KHÔNG THEO D-NUMBER

D1–D7 là **decision governance**, nhưng code nên triển khai theo execution dependency.

Tôi khuyến nghị sequence sau.

---

## GATE I — Freeze implementation inputs

Verify:

```text
fixed train/val/test manifests
L/U budget manifests
10-seed protocol
category mapping
input JPG representation
environment baseline
```

Không train nếu input hashes/membership chưa xác định.

---

## GATE II — Build shared SUP infrastructure

Implement chung:

```text
dataset loader
empty-GT handling
channel replication
resize/pad
optimizer-update accounting
validation scheduling
checkpoint selection
seed control
AMP
resume
```

Sau đó smoke test cả R50 và Swin-T.

---

## GATE III — Build and PASS D4-C SSL preflight

Trước khi chạy 80 SSL main runs, test nhỏ phải xác nhận:

```text
teacher initialization
EMA update timing
AMP skipped-step behavior
resume
weak/strong U views
pseudo-label thresholds
empty pseudo labels
zero-GT images
L:U ratio
unsup weight
hidden-U firewall
```

---

## GATE IV — Build evaluator and PASS D5 preflight

Không đợi tới cuối training mới viết evaluator.

Có thể build trước bằng:

```text
toy predictions
early validation checkpoints
structural smoke data
```

Nhưng:

```text
NO final test performance
```

---

## GATE V — Build statistical pipeline and PASS D6 preflight

D6 statistical code không cần real test metrics để test.

Dùng:

```text
golden synthetic fixtures
mock official matrix
structural real-project metadata
```

Khi P1–P13 PASS:

```text
D6 = CLOSED / PASS
```

---

## GATE VI — Official SUP training

Recommended order:

```text
R50 SUP first
then Swin-T SUP
```

Lý do operational:

- conventional baseline đơn giản hơn để xác nhận data/training infrastructure;
- dễ phát hiện lỗi chung trước Swin;
- không thay scientific design.

Đây là **implementation recommendation**, không phải locked research decision.

---

## GATE VII — Official SSL training

Chỉ sau D4-C preflight PASS.

Recommended:

```text
R50 SSL first
then Swin-T SSL
```

Sau đó D4-D R50/10% ablations **chỉ khi D4-D.9 implementation/technical preflight PASS**.

---

## GATE VIII — Freeze official model/checkpoint matrix

Trước final test, verify:

```text
all required official runs complete
all locked seeds present
no unresolved invalid runs
BEST/LAST mappings frozen
official checkpoints hashed
RQ4 validation Qpseudo table frozen
```

---

## GATE IX — Test authorization

Chỉ khi D4/D5/D6 closure prerequisites đều đạt.

Then:

```text
ONE FROZEN FINAL TEST CAMPAIGN
```

180 main/reference evaluations.

No D4-D non-main test.

---

## GATE X — Statistical analysis

Input:

```text
frozen full-precision metric artifacts
```

Not:

```text
manual copy/paste from terminal/table
```

Run locked D6 analyses.

Freeze outputs.

---

## GATE XI — D7 after training

Đúng theo researcher decision:

```text
collect actual evidence
then design/finalize D7 Reproducibility Contract
```

D7 lúc đó có thể document actual:

```text
Git commits
configs
environment
run manifests
attempt history
checkpoints
predictions
metrics
statistics
hashes
reproduction procedure
```

---

# 17. NHỮNG VIỆC TUYỆT ĐỐI KHÔNG ĐƯỢC TỰ Ý LÀM KHI IMPLEMENT

```text
PROHIBITED:
- đổi 10 training seeds
- seed search
- replacement seed
- đổi budget membership
- dùng hidden U GT
- đổi R50/Swin architecture pair
- đổi detector family
- đổi optimizer/schedule vì result thấp
- đổi update budget
- đổi SSL thresholds ngoài D4-D
- đổi lambda_unsup
- đổi L:U ratio
- thêm augmentation không được khóa
- dùng Student thay Teacher cho official SSL test
- dùng BEST Teacher thay LAST Teacher để tính Qpseudo
- tune tau_eval trên validation/test
- test D4-D non-main variants
- thay primary metric vì AP50 đẹp hơn
- chọn statistical test theo Shapiro/Mauchly result
- thay Holm bằng BH/Bonferroni sau khi xem result
- loại valid poor-performing seed
- rerun valid seed vì metric thấp
- impute missing official metric
- chạy official analysis với n=9
- đọc final test performance trước authorization
```

---

# 18. INTERPRETATION GUARDRAILS FROM FINAL PRE-IMPLEMENTATION SCIENTIFIC AUDIT

Các ghi chú trong mục này không thay đổi scientific protocol, experimental
matrix, training configuration, metric protocol hoặc statistical analysis đã
được xác định trước. Chúng quy định giới hạn diễn giải kết quả sau này và các
điểm cần được giữ nhất quán khi implementation, Results, Discussion và Threats
to Validity được hoàn thiện.

---

## 18.1. RQ3 — Label-budget effect không phải pure causal effect của label quantity

Các điều kiện ngân sách nhãn sử dụng optimizer-update budgets đã xác định trước:

```text
1%  = 1032 updates
5%  = 2064 updates
10% = 2064 updates
20% = 2064 updates
```

Do đó, đặc biệt đối với các so sánh liên quan đến mức 1%, các experimental
conditions khác nhau không chỉ ở số lượng ảnh có nhãn mà còn ở training
schedule đã xác định trước.

### Implementation rule

KHÔNG thay đổi update budget để làm các mức ngân sách giống nhau.

Matched SUP–SSL trong cùng một budget vẫn phải giữ:

```text
same optimizer-update budget
same labeled-supervision exposure
```

### Interpretation rule

RQ3 được diễn giải là:

> budget-dependent SSL gain under the prespecified budget-specific training conditions.

KHÔNG được diễn giải như:

> pure causal effect của riêng số lượng labeled images lên SSL gain.

Nếu RQ3 cho thấy sự khác biệt giữa các budget, kết luận chỉ áp dụng cho các
điều kiện ngân sách và lịch huấn luyện đã được xác định trước trong nghiên cứu.

---

## 18.2. Fixed partition — Training-seed uncertainty không bao gồm partition uncertainty

Nghiên cứu sử dụng:

```text
one fixed train/validation/test partition
one fixed nested L/U construction
10 training seeds per official condition
```

Do đó, biến thiên giữa 10 training seeds chủ yếu phản ánh stochasticity của
quá trình huấn luyện trên cùng một cấu trúc dữ liệu đã cố định.

Nó không lượng hóa đầy đủ uncertainty có thể phát sinh nếu:

```text
train/validation/test split khác
hoặc
labeled subset membership khác
```

### Implementation rule

KHÔNG xây dựng lại train/validation/test hoặc L/U membership theo từng
training seed.

```text
partition_seed != training_seed
```

phải tiếp tục được giữ.

### Interpretation rule

Mean, SD, 95% CI và paired-seed analyses được diễn giải conditional on the
fixed experimental partition.

KHÔNG được tuyên bố rằng 10 training seeds phản ánh toàn bộ uncertainty của
dataset partition hoặc labeled-subset construction.

---

## 18.3. Ablation — Validation evidence không phải unbiased final-test generalization estimate

D4-D non-main variants được đánh giá trên fixed validation set và không được
đưa sang final test.

Validation đồng thời tham gia checkpoint selection theo rule đã xác định
trước. Vì vậy, ablation validation performance có thể chịu ảnh hưởng của
checkpoint-selection optimism.

### Implementation rule

Tất cả Main và ablation variants phải sử dụng cùng checkpoint-selection rule
đã xác định trước.

Không được:

```text
dùng checkpoint rule khác cho variant
chọn ablation winner để thay Main
đưa non-main ablation sang test
đổi Main sau khi xem ablation
loại bỏ negative/null ablation results
```

### Interpretation rule

Ablation results được xem là:

> comparative validation evidence for secondary/mechanistic component analysis.

KHÔNG được diễn giải như:

> unbiased estimate of final-test generalization.

KHÔNG được dùng ablation để chứng minh một variant tổng quát hóa tốt hơn Main.

---

## 18.4. Patient-level leakage — Không thể loại trừ tuyệt đối

Nghiên cứu đã kiểm soát và xác nhận non-overlap trong phạm vi:

```text
image_id
available annotation structure
train/validation/test membership
L/U membership
```

Tuy nhiên, dữ liệu khả dụng không cung cấp patient/group identifier phù hợp để
xác định chắc chắn liệu nhiều ảnh khác nhau có thuộc cùng một bệnh nhân hay
không.

### Implementation rule

Tiếp tục kiểm tra image-level leakage bằng các định danh khả dụng.

Không được suy diễn:

```text
unique image_id
=> unique patient
```

### Interpretation rule

Được viết:

> Nghiên cứu xác nhận sự tách biệt ở cấp định danh ảnh và annotation khả dụng,
> nhưng không thể loại trừ tuyệt đối patient-level leakage do thiếu định danh
> bệnh nhân phù hợp.

KHÔNG được viết:

> patient-level leakage đã được loại trừ hoàn toàn.

Đây là known dataset limitation, không phải implementation failure.

---

## 18.5. Governance của các interpretation guardrails

Các guardrails trong mục này:

```text
DO NOT change:
- research questions
- hypotheses
- experimental matrix
- training budgets
- model configurations
- SSL thresholds
- metrics
- statistical models
```

Chúng chỉ quy định:

```text
HOW RESULTS MAY BE INTERPRETED
HOW LIMITATIONS MUST BE REPORTED
```

Nếu implementation sau này phát hiện một vấn đề mới có khả năng làm thay đổi
scientific interpretation, vấn đề đó phải được đánh giá riêng trước khi sửa
protocol hoặc thesis.

---

# 19. CONTROLLED REVISION TRIGGERS

Nếu implementation buộc phải thay một trong các điểm sau, dừng và lập Controlled Revision:

```text
architecture
detector family
optimizer
learning rate
training update budget
batch/exposure
SSL teacher/student semantics
EMA coefficient
SSL threshold
augmentation family/range
pseudo-label filtering
loss weight
burn-in/ramp
evaluation metric
score threshold
test use
seed list
multiplicity
primary statistical test
hypothesis direction
```

Technical correction chỉ được xem là implementation fix nếu nó đưa hệ thống trở lại **đúng locked protocol**.

---

# 20. MINIMUM ARTIFACTS NÊN SINH TRONG IMPLEMENTATION

D7 chưa formalize, nhưng để tránh mất evidence, mỗi official run **nên** giữ:

```text
run ID
method
architecture
budget
seed_index
training_seed
exact config
training log
validation log
BEST checkpoint
LAST checkpoint
exit status
retry/attempt info if any
```

Project-level evidence:

```text
protocol configs
environment export
Git commit
dataset manifests
category mapping
evaluation outputs
statistical outputs
```

Đây là **implementation evidence preservation recommendation**, không phải D7 protocol đã khóa.

---

# 21. READY / NOT READY MATRIX

| Component | Protocol | Implementation/preflight | Có thể dùng official? |
|---|---|---|---|
| D1 hypotheses | LOCKED | N/A | YES |
| D2 variables | LOCKED | N/A | YES |
| D3 design | LOCKED | N/A | YES |
| R50/Swin selection | LOCKED | config implementation needed | prepare |
| SUP training config | LOCKED | smoke/preflight needed | prepare |
| SSL treatment | LOCKED | **D4-C preflight pending** | **NO official SSL yet** |
| Ablation | LOCKED | depends Main R50/10% SSL | later |
| Evaluator | LOCKED | **D5 preflight pending** | not final-test ready |
| Statistics | LOCKED | **D6 preflight pending** | not final-test ready |
| Final test | LOCKED firewall | authorization pending | **NO** |
| D7 | DEFERRED | after training | later |

---

# 22. ĐỀ XUẤT CẤU TRÚC THƯ MỤC IMPLEMENTATION

Đây là recommendation để giảm lỗi config drift:

```text
configs/
  protocol/
    experimental_design.yaml
    phase2F1_seed_protocol.yaml
    d4_training_protocol.yaml
    d4_ssl_protocol.yaml
    d5_metric_protocol.yaml
    d6_statistical_protocol.yaml

  models/
    r50_sup.py
    swint_sup.py
    r50_softteacher.py
    swint_softteacher.py

scripts/
  train_official.py
  validate_official.py
  run_d4c_preflight.py
  run_d5_preflight.py
  run_d6_preflight.py

evaluation/
  prediction_export.py
  coco_bbox_eval.py
  operating_point_eval.py
  negative_image_eval.py
  pseudo_label_eval.py

statistics/
  build_effect_table.py
  rq2_primary.py
  rq3_budget.R
  rq4_lmm_kr.R
  rq5_rare.py
  rq6_negative.py
  rq7_architecture.py
  multiplicity.py
  robustness.py

reports/
  preflight/
  training/
  evaluation/
  statistics/

data/
  manifests/
```

Không bắt buộc đúng tên này nếu repository hiện có naming convention khác; mục tiêu là **single source of truth + separation of scientific config and execution code**.

---

# 23. IMPLEMENTATION ORDER — BẢN NGẮN

```text
1. Verify locked manifests + seeds

2. Implement shared R50/Swin SUP infrastructure

3. Smoke test empty-GT / input / updates / checkpoint / resume

4. Implement SoftTeacher contract

5. PASS D4-C preflight

6. Implement evaluator

7. PASS D5 preflight

8. Implement statistical pipeline on synthetic fixtures

9. PASS D6 P1-P13

10. Run official SUP:
    Phase 4A R50
    Phase 4B Swin-T

11. Run official SSL:
    Phase 5A R50
    Phase 5B Swin-T

12. Run D4-D R50/10% validation ablations

13. Freeze official run/checkpoint/Qpseudo matrix

14. Verify all test-authorization prerequisites

15. Run ONE frozen 180-model final-test campaign

16. Run locked D6 statistical analyses

17. Freeze results

18. Build D7 from actual post-training evidence
```

> Có thể triển khai evaluator/statistics **trước hoặc song song với SUP training**, miễn không đọc final-test performance. Điều quan trọng là D4-C phải PASS trước official SSL và D5/D6 phải PASS trước final-test authorization.

---

# 24. SOURCE-OF-TRUTH FILES ĐÃ CÓ

Các evidence/decision files liên quan đã được tạo/nhắc trong dự án:

```text
D1_GIA_THUYET_NGHIEN_CUU_D1_4_FORMALLY_LOCKED_VI.md
D2_R1_BIEN_NGHIEN_CUU_VIT_ATTENTION_LOCKED_VI.md

D4_D_ABLATION_STUDY_DECISION_EVIDENCE_RECORD_VI.md

D6_STATISTICAL_ANALYSIS_PROTOCOL_DECISION_EVIDENCE_RECORD_VI.md
```

Project governance/evidence sources thường dùng:

```text
PHASE_HANDOFF.md
PROJECT_CONTEXT.md
README.md
research_log.md
CHECKLIST_TRIEN_KHAI_FULL.xlsx
```

Các config/manifest evidence đã có từ phase trước:

```text
configs/protocol/phase2F_labeled_unlabeled.yaml
configs/protocol/phase2F1_seed_protocol.yaml

data/manifests/...
reports/02F_...
reports/02F1_...
reports/03A_...
```

Khi implement, không nên copy các quyết định bằng tay nếu có thể đọc trực tiếp từ locked protocol/manifests.

---

# 25. FINAL MASTER STATE

```text
D1 = LOCKED
D2 = LOCKED
D3 = LOCKED

D4-A = LOCKED
D4-B = LOCKED
D4-C PROTOCOL = LOCKED
D4-C PREFLIGHT = NOT EXECUTED
OFFICIAL SSL = NOT AUTHORIZED

D4-D = LOCKED

D5 PROTOCOL = LOCKED
D5 PREFLIGHT = NOT EXECUTED

D6 PROTOCOL = LOCKED
D6 PREFLIGHT = NOT EXECUTED

FINAL TEST = CLOSED / NOT AUTHORIZED

D7 = DEFERRED UNTIL AFTER TRAINING
D7 PROTOCOL = NOT YET DESIGNED / NOT LOCKED
```

Implementation phải giữ đúng nguyên tắc:

\[
\boxed{
\text{Scientific decisions are frozen;}
\quad
\text{implementation must prove compliance.}
}
\]

---

# 26. KẾT LUẬN

D1–D6 hiện đã cung cấp đủ **scientific decision layer** để bắt đầu chuẩn bị implementation:

- D1 xác định câu hỏi/hypothesis;
- D2 xác định variables/interactions;
- D3 xác định experimental matrix/pairing;
- D4 xác định models/training/SSL/ablation;
- D5 xác định evaluator/metrics/test usage;
- D6 xác định inferential/statistical protocol;
- D7 được researcher chủ động defer để xây từ actual post-training evidence.

Do đó, bước hợp lý nhất hiện tại **không phải tiếp tục thiết kế thêm research decisions**, mà chuyển sang:

\[
\boxed{
\text{IMPLEMENTATION PREPARATION}
}
\]

theo execution gates:

\[
\boxed{
\text{shared infrastructure}
\rightarrow
D4-C\ preflight
\rightarrow
D5\ preflight
\rightarrow
D6\ preflight
\rightarrow
official\ training
}
\]

trong khi final test vẫn giữ kín cho tới khi toàn bộ authorization gates đạt điều kiện.

---
