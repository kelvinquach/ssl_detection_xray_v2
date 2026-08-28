# VIETLUANVAN — CURRENT SCIENTIFIC DECISION & EVIDENCE RECORD D1–D6
## Nguồn chuẩn hiện hành để viết luận văn

**Đề tài:** *Nghiên cứu học bán giám sát cho dò tìm bất thường trên X-quang phổi*  
**Ngày hợp nhất current-state:** 26/08/2026  
**Ngôn ngữ:** Tiếng Việt  
**Mục đích:** Làm **nguồn nghiên cứu/decision/evidence hiện hành duy nhất** để viết luận văn, đối chiếu Methodology, kiểm tra terminology và bảo đảm nội dung luận văn nhất quán với scientific protocol sẽ được implement sau này.

---

# 0. PHẠM VI VÀ NGUYÊN TẮC QUẢN TRỊ

## 0.1. Vai trò của file này

`vietluanvan.md` là **CURRENT SCIENTIFIC STATE**, không phải historical archive.

Nó được xây dựng từ master verbatim D1–D6 nhưng:

- chỉ giữ **trạng thái cuối cùng có authority cao nhất**;
- loại bỏ historical contradictions;
- không giữ trạng thái cũ đã bị Controlled Revision thay thế;
- không giữ proposal/intermediate khi đã có bản LOCKED/LOCKED_COMPLETED;
- không biến implementation/preflight state thành scientific decision;
- không tạo quyết định khoa học mới.

Nguồn archive vẫn là:

```text
MASTER_D1_D6_FULL_VERBATIM_MERGED_RECORD_VI
```

Nguyên tắc authority của archive:

1. Controlled Revision đã khóa và mới hơn có authority cao hơn.
2. `LOCKED_COMPLETED` có authority cao hơn intermediate/historical.
3. Historical record chỉ dùng truy vết lý do/trao đổi, không dùng làm current state.

---

## 0.2. Quan hệ giữa file này và implementation

Scientific flow chính thức:

\[
\boxed{
\text{Scientific decisions trong }vietluanvan.md
\rightarrow
\text{Methodology trong luận văn}
}
\]

và sau này:

\[
\boxed{
\text{Scientific decisions trong }vietluanvan.md
\rightarrow
\text{Implementation master / config / code}
}
\]

Implementation không được tự thay đổi:

- RQ;
- hypothesis;
- architecture;
- budgets;
- seed policy;
- metric;
- threshold;
- statistical model;
- multiplicity;
- test firewall.

Nếu code/config sau này khác file này, phải xác định:

```text
IMPLEMENTATION BUG
```

hoặc:

```text
CONTROLLED REVISION REQUIRED
```

chứ không được sửa luận văn theo code một cách hậu nghiệm.

---

# 1. TỔNG QUAN CURRENT STATE D1–D6

```text
D1:
CURRENT = D1 base + D1-R1 + D1.4 Formalization
RQ1–RQ7
RQ4 formally locked
RQ7 formally added

D2:
CURRENT = D2 base + D2-R1
3 independent variables
7 dependent variables

D3:
RESEARCHER_APPROVED / LOCKED

D4-A:
RESEARCHER_APPROVED / LOCKED / COMPLETED

D4-B:
RESEARCHER_APPROVED / LOCKED / COMPLETED

D4-C:
SCIENTIFIC PROTOCOL = RESEARCHER_APPROVED / LOCKED

D4-D:
SCIENTIFIC RESEARCH DESIGN D4-D.1–D4-D.8 = RESEARCHER_APPROVED / LOCKED
D4-D.9 = DEFERRED TO IMPLEMENTATION / TECHNICAL PREFLIGHT
RESEARCH-SCOPE ABLATION DESIGN = COMPLETE

D5:
SCIENTIFIC METRIC PROTOCOL = RESEARCHER_APPROVED / LOCKED

D6:
SCIENTIFIC STATISTICAL PROTOCOL = RESEARCHER_APPROVED / LOCKED
```

---

# 2. LOGIC NGHIÊN CỨU TỪ D1 ĐẾN D6

\[
\boxed{\text{D1 — Ta muốn trả lời câu hỏi nào?}}
\]

\[
\Downarrow
\]

\[
\boxed{\text{D2 — Biến nào thay đổi, biến nào được đo, biến nào được kiểm soát?}}
\]

\[
\Downarrow
\]

\[
\boxed{\text{D3 — Thiết kế thực nghiệm nào cho phép trả lời các RQ?}}
\]

\[
\Downarrow
\]

\[
\boxed{\text{D4 — Model, training contract, SSL treatment và ablation được định nghĩa thế nào?}}
\]

\[
\Downarrow
\]

\[
\boxed{\text{D5 — Hiệu năng/chất lượng được đo bằng metric nào?}}
\]

\[
\Downarrow
\]

\[
\boxed{\text{D6 — Các hiệu ứng được suy luận thống kê như thế nào?}}
\]

Đây là trật tự phải được giữ khi viết luận văn.

---

# 3. D1 — RESEARCH QUESTIONS & HYPOTHESES

# 3.1. Current revision chain

Current D1 là kết quả hợp nhất:

\[
\boxed{
D1\ v1.0
+
D1\text{-}R1
+
D1.4\ Formalization
}
\]

Trong đó:

- D1 v1.0 khóa RQ1–RQ6;
- D1-R1 bổ sung RQ7 và Architecture;
- D1.4 Formalization khóa chính thức giả thuyết RQ4 sau khi \(Q_{\text{pseudo}}\) được operationalize.

Không sử dụng trạng thái historical:

```text
RQ4 = PROVISIONAL
```

trong luận văn hiện tại.

Current state:

```text
RQ4 = FORMALLY LOCKED
```

---

# 3.2. RQ1 — Supervised performance theo label budget

## Câu hỏi nghiên cứu

> Hiệu năng của supervised multi-class object detection thay đổi như thế nào khi ngân sách nhãn bounding box được giới hạn ở 1%, 5%, 10% và 20%?

## Vai trò

```text
DESCRIPTIVE / SUPPORTING
NO FORMAL RESEARCH HYPOTHESIS
```

## Quyết định

Không định nghĩa:

\[
H_{\mathrm{RQ1}}.
\]

Không pre-specify:

\[
M_{1\%}
<
M_{5\%}
<
M_{10\%}
<
M_{20\%}
\]

như một hypothesis bắt buộc.

## Tại sao chọn như vậy?

RQ1 hỏi “hiệu năng thay đổi như thế nào”, không hỏi liệu một quy luật đơn điệu định trước có tồn tại.

Giữ RQ1 descriptive giúp:

- tránh ép dữ liệu theo kỳ vọng;
- cho phép báo cáo pattern thực tế;
- tạo supervised reference cho RQ2/RQ3.

## Cách viết luận văn

Có thể viết:

> RQ1 được sử dụng để mô tả độ nhạy của supervised baseline đối với ngân sách nhãn và không được gắn với một giả thuyết khẳng định riêng.

Không viết:

> Hiệu năng chắc chắn tăng đơn điệu khi tăng tỷ lệ nhãn.

---

# 3.3. RQ2 — SSL so với SUP

## Câu hỏi nghiên cứu

> Trong các điều kiện matched về dữ liệu có nhãn, architecture, split, seed và evaluation protocol, teacher–student semi-supervised pseudo-labeling có cải thiện detection performance so với supervised baseline hay không?

## Vai trò

```text
PRIMARY
FORMAL
DIRECTIONAL
ONE OVERALL PRIMARY HYPOTHESIS
```

## Giả thuyết nghiên cứu

\[
\boxed{
H_{\mathrm{RQ2}}:
\text{SSL được kỳ vọng có positive downstream gain so với SUP}
}
\]

Primary endpoint:

\[
\boxed{
mAP@[0.50:0.95]
}
\]

## Tại sao directional?

Đề tài được thiết kế để kiểm tra liệu khai thác unlabeled data thông qua teacher–student pseudo-labeling có mang lại **lợi ích** so với supervised learning.

Directional không có nghĩa SSL chắc chắn tốt hơn.

Nếu kết quả không hỗ trợ:

\[
SSL>SUP,
\]

hypothesis đơn giản là không được hỗ trợ.

## Tại sao chỉ một primary hypothesis?

Không tạo:

```text
4 primary hypotheses theo 4 budgets
```

và không tạo:

```text
8 primary hypotheses theo 2 architectures × 4 budgets
```

Lý do:

- tránh multiplicity inflation;
- tránh selective reporting;
- giữ một scientific claim trung tâm.

## Statistical estimand ở D6

Với:

\[
\Delta_{a,b,s}
=
mAP^{SSL}_{a,b,s}
-
mAP^{SUP}_{a,b,s},
\]

định nghĩa:

\[
\boxed{
G_s
=
\frac{1}{8}
\sum_{a,b}
\Delta_{a,b,s}
}
\]

trên:

\[
2\ architectures
\times
4\ budgets.
\]

Formal statistical hypothesis:

\[
H_0:\mu_G\le0
\]

\[
H_1:\mu_G>0.
\]

## External evidence / literature context

### STAC

Paper:

https://arxiv.org/abs/2005.04757

GitHub:

https://github.com/google-research/ssl_detection

### Unbiased Teacher

Paper:

https://arxiv.org/abs/2102.09480

### Soft Teacher

Paper:

https://openaccess.thecvf.com/content/ICCV2021/html/Xu_End-to-End_Semi-Supervised_Object_Detection_With_Soft_Teacher_ICCV_2021_paper.html

GitHub:

https://github.com/microsoft/SoftTeacher

## Citation logic

Các paper trên hỗ trợ:

- SSOD teacher–student;
- pseudo-labeling;
- low-label object detection.

Nhưng việc:

```text
RQ2 = PRIMARY / DIRECTIONAL
```

là **quyết định của nghiên cứu**, không phải yêu cầu của các paper.

---

# 3.4. RQ3 — Label-budget effect on SSL gain

## Câu hỏi

> SSL gain so với SUP có thay đổi giữa 1%, 5%, 10% và 20% labeled budget hay không?

## Vai trò

```text
SECONDARY
FORMAL
NON-DIRECTIONAL
```

## Giả thuyết

\[
\boxed{
H_{\mathrm{RQ3}}:
\Delta_b
\text{ có thể khác nhau giữa các label budgets}
}
\]

## Không pre-specify

Không khóa:

\[
\Delta_{1\%}
>
\Delta_{5\%}
>
\Delta_{10\%}
>
\Delta_{20\%}.
\]

Không khóa linear trend.

Không khóa monotonic trend.

## Tại sao?

Không có cơ sở lý thuyết đủ mạnh để nói SSL gain chắc chắn lớn nhất ở budget thấp nhất.

Ngoài ra labeled subsets là nested, nên các budget conditions không thể được xem như bốn mẫu độc lập hoàn toàn.

## D6 estimand

Architecture-balanced:

\[
\boxed{
B_{b,s}
=
\frac{
\Delta_{R50,b,s}
+
\Delta_{Swin,b,s}
}{2}
}
\]

## D6 analysis

```text
one-way repeated-measures ANOVA
Greenhouse-Geisser correction ALWAYS
```

Six pre-specified contrasts:

```text
5%-1%
10%-1%
20%-1%
10%-5%
20%-5%
20%-10%
```

## Evidence

Greenhouse, S. W. & Geisser, S. (1959):

https://doi.org/10.1007/BF02289823

R/afex:

https://github.com/cran/afex

Documentation:

https://search.r-project.org/CRAN/refmans/afex/html/aov_car.html

statsmodels `AnovaRM`:

https://www.statsmodels.org/stable/generated/statsmodels.stats.anova.AnovaRM.html

Ghi chú:

`AnovaRM` không đủ một mình cho official GG workflow.

---

# 3.5. RQ4 — Pseudo-label quality association

## Current status

```text
SECONDARY
MECHANISTIC
DIRECTIONAL
POSITIVE ASSOCIATION
FORMALLY LOCKED
```

## Câu hỏi

> Chất lượng pseudo-label có mối liên hệ như thế nào với downstream SSL gain trong các architecture × label-budget conditions đã xác định trước?

## Operational definition

\[
\boxed{
Q_{\mathrm{pseudo}}
=
PL\text{-}mAP@[0.50:0.95]
}
\]

Audit scope:

```text
FIXED VALIDATION SET
```

Checkpoint:

```text
LAST EMA TEACHER
```

Observation unit:

\[
architecture
\times
budget
\times
training\ seed.
\]

Main SSL observations:

\[
2\times4\times10=80.
\]

## Research hypothesis

\[
\boxed{
H_{\mathrm{RQ4}}:
Q_{\mathrm{pseudo}}\uparrow
\text{ associated with }
\Delta mAP_{\mathrm{downstream}}\uparrow
}
\]

## Tại sao dùng PL-mAP@[.5:.95]?

Object-detection pseudo labels gồm cả:

- class;
- bounding box;
- confidence.

Metric chất lượng cần phản ánh cả classification và localization ở nhiều IoU thresholds.

Do đó PL-mAP@[.5:.95] phù hợp hơn việc chỉ dùng:

- mean confidence;
- pseudo count;
- retention rate.

## Secondary/mechanistic pseudo metrics

Có thể dùng:

- PL-AP50;
- PL-AP75;
- PL-Precision@IoU=.50;
- PL-Recall@IoU=.50;
- PL-F1@IoU=.50;
- mean matched IoU;
- class-wise PL-AP;
- pseudo FP trên validation negative images;
- regression uncertainty;
- pseudo-label count/retention.

Nhưng chúng không thay thế:

\[
Q_{\mathrm{pseudo}}.
\]

## Ground-truth firewall

Hidden GT của unlabeled training subsets:

```text
GT(U_b) -> pseudo-label quality evaluation = PROHIBITED
GT(U_b) -> threshold tuning = PROHIBITED
GT(U_b) -> filtering = PROHIBITED
GT(U_b) -> EMA tuning = PROHIBITED
GT(U_b) -> augmentation tuning = PROHIBITED
GT(U_b) -> checkpoint selection = PROHIBITED
```

## Validation vs test

Qpseudo:

```text
VALIDATION ONLY
LAST EMA Teacher
```

Formal downstream outcome ở D6:

\[
Y_{a,b,s}
=
\Delta mAP^{test}_{a,b,s}.
\]

Điều này không tạo test leakage vì test outcome chỉ được sử dụng sau khi protocol đã được khóa.

## D6 model

Within-cell centered predictor:

\[
Q^{WC}_{a,b,s}
=
Q_{a,b,s}
-
\bar Q_{a,b}.
\]

Scale:

\[
X
=
Q^{WC}/0.01.
\]

Formal model:

\[
\boxed{
\Delta mAP_{test}
\sim
Q^{WC}_{pseudo}
+
Cell
+
(1|training\_seed)
}
\]

Estimation:

```text
REML
Kenward-Roger
```

Formal direction:

\[
H_0:\beta_Q\le0
\]

\[
H_1:\beta_Q>0.
\]

## Tại sao association, không causal?

Pseudo-label quality không phải randomized intervention.

Do đó không được viết:

> Qpseudo cao gây ra hiệu năng tốt hơn.

Được viết:

> Qpseudo cao hơn có mối liên hệ dương với downstream SSL gain.

## Evidence

Enders & Tofighi — centering:

https://doi.org/10.1037/1082-989X.12.2.121

Kenward & Roger:

https://doi.org/10.2307/2533558

pbkrtest paper:

https://doi.org/10.18637/jss.v059.i09

lme4:

https://github.com/lme4/lme4

lmerTest:

https://github.com/cran/lmerTest

pbkrtest:

https://github.com/hojsgaard/pbkrtest

---

# 3.6. RQ5 — Rare classes

## Câu hỏi

> Các lớp có support thấp/lớp hiếm nhận được lợi ích như thế nào từ SSL?

## Vai trò

```text
EXPLORATORY
```

Frozen rare classes:

```text
Atelectasis
Pneumothorax
```

Nguồn:

Phase 3A, rare if train image prevalence <5%.

## Metric

\[
AP_c@[0.50:0.95].
\]

Paired effect:

\[
\Delta AP_c
=
AP^{SSL}_c
-
AP^{SUP}_c.
\]

## D6 estimand

\[
G_{c,s}
=
\frac18
\sum_{a,b}
\Delta AP_{c,a,b,s}.
\]

## Reporting

- mean paired gain;
- sample SD;
- two-sided 95% CI;
- no required formal p-value.

## Tại sao exploratory?

Rare-class support thấp làm uncertainty cao.

Nghiên cứu không được designed/powered như một class-specific confirmatory trial.

## Không làm

Không tạo:

```text
rare-mAP
```

như endpoint mới.

Không chuyển exploratory result thành confirmatory claim sau khi xem kết quả.

---

# 3.7. RQ6 — False positives trên No Finding

## Câu hỏi

> SSL có làm thay đổi false-positive burden trên các ảnh No Finding/zero-GT so với SUP hay không?

## Vai trò

```text
SECONDARY
FORMAL
NON-DIRECTIONAL
```

## Tại sao non-directional?

Không có cơ sở prespecified để khẳng định SSL chắc chắn:

\[
FP_{\mathrm{SSL}}<FP_{\mathrm{SUP}}
\]

hoặc:

\[
FP_{\mathrm{SSL}}>FP_{\mathrm{SUP}}.
\]

## Primary within-RQ6 metric

\[
\boxed{
FP/negative\ image
}
\]

## Secondary

\[
FAR_{neg}
=
P(FP_i\ge1\mid negative).
\]

## D6 effect

\[
\Delta F_{a,b,s}
=
FP^{SSL}_{neg}
-
FP^{SUP}_{neg}.
\]

\[
H_s
=
\frac18
\sum_{a,b}
\Delta F_{a,b,s}.
\]

Formal analysis:

```text
two-sided one-sample t-test
n=10
df=9
```

Negative effect means fewer false positives under SSL.

---

# 3.8. RQ7 — Architecture-dependent SSL effect

## Origin

RQ7 được bổ sung tại D1-R1 theo yêu cầu phát triển hướng ViT/Attention của giảng viên hướng dẫn.

## Câu hỏi

> Mức lợi ích của SSL so với supervised baseline có khác giữa conventional CNN architecture và hierarchical Vision Transformer architecture hay không?

## Vai trò

```text
SECONDARY
FORMAL
NON-DIRECTIONAL
```

## Research hypothesis

\[
\boxed{
H_{\mathrm{RQ7}}:
\text{SSL gain có thể khác giữa hai architecture}
}
\]

## Không phải hypothesis

\[
Swin-T>R50.
\]

Không có formal architecture-main-effect superiority claim.

## Conceptual target

\[
\boxed{
Method\times Architecture
}
\]

## D4 operationalization

Conventional:

\[
Faster\ R-CNN
+
ResNet\text{-}50
+
FPN.
\]

ViT/Attention:

\[
Faster\ R-CNN
+
Swin\text{-}T
+
FPN.
\]

## D6 estimand

\[
A_{a,s}
=
\frac14
\sum_b
\Delta_{a,b,s}.
\]

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
```

## Interpretation boundary

Không gọi đây là pure causal effect của backbone nếu architecture-specific optimizer/training recipe khác nhau.

Viết:

> architecture-dependent SSL effect under prespecified architecture-specific training recipes.

## Evidence

Swin Transformer:

https://openaccess.thecvf.com/content/ICCV2021/html/Liu_Swin_Transformer_Hierarchical_Vision_Transformer_Using_Shifted_Windows_ICCV_2021_paper.html

GitHub:

https://github.com/microsoft/Swin-Transformer

---

# 4. D2 — RESEARCH VARIABLES

# 4.1. Current state

Current D2:

\[
\boxed{
D2\ base + D2\text{-}R1
}
\]

Do đó nghiên cứu hiện có:

\[
\boxed{
3\ independent\ variables
+
7\ dependent\ variables
}
\]

Không tiếp tục viết “2 independent variables”.

---

# 4.2. X1 — Training paradigm

\[
X_1
\in
\{
SUP,
SSL
\}.
\]

Role:

```text
PRIMARY EXPERIMENTAL FACTOR
```

---

# 4.3. X2 — Labeled-data budget

\[
X_2
\in
\{
1\%,
5\%,
10\%,
20\%
\}.
\]

Role:

```text
SECONDARY EXPERIMENTAL FACTOR
ORDERED CATEGORICAL
```

D6 primary analysis treats budget categorically.

---

# 4.4. X3 — Architecture

\[
X_3
\in
\{
R50,
Swin-T
\}.
\]

Role:

```text
SECONDARY EXPERIMENTAL FACTOR / MODERATOR
```

---

# 4.5. Dependent variables

Primary:

1. bbox mAP@[0.50:0.95]

Secondary:

2. AP50
3. AP75
4. class-wise AP
5. Recall
6. FP/image
7. FP/negative image

Mechanistic predictor:

\[
Q_{\mathrm{pseudo}}.
\]

---

# 4.6. Interaction structure

```text
Method × Budget
= SECONDARY / FORMAL

Method × Architecture
= SECONDARY / FORMAL

Budget × Architecture
= SECONDARY / EXPLORATORY

Method × Budget × Architecture
= SECONDARY / EXPLORATORY
```

---

# 4.7. Blocking / pairing

`training_seed`:

```text
BLOCKING / PAIRING FACTOR
```

Không phải independent variable chính.

---

# 4.8. Controlled factors

Matched SUP–SSL giữ:

- fixed dataset scope;
- fixed train/validation/test;
- same \(L_b\);
- same architecture;
- same ordered training seed;
- same input representation;
- same detector family;
- same evaluation protocol;
- same total optimizer-update budget;
- same labeled-supervision exposure.

SSL additionally consumes:

\[
U_b.
\]

---

# 4.9. Supporting strata

Rare status:

```text
prespecified stratification / exploratory analysis factor
```

No Finding:

```text
zero-GT evaluation stratum
NOT detection class
```

---

# 5. D3 — CONTROLLED EXPERIMENTAL DESIGN

# 5.1. Main low-label factorial design

\[
2\ methods
\times
4\ budgets
\times
2\ architectures
=
16\ cells.
\]

Each cell:

\[
10\ training\ seeds.
\]

Therefore:

\[
\boxed{
160\ official\ low-label\ runs
}
\]

---

# 5.2. Full-label SUP references

\[
R50\times100\%\text{-SUP}\times10
\]

\[
Swin-T\times100\%\text{-SUP}\times10.
\]

Total:

\[
20.
\]

Role:

```text
REFERENCE / CONTEXT
```

Not included in main low-label paired estimands.

Total main/reference:

\[
\boxed{
180
}
\]

---

# 5.3. Phase mapping

```text
Phase 4A = R50 SUP
Phase 4B = Swin-T SUP
Phase 5A = R50 SSL
Phase 5B = Swin-T SSL
```

---

# 5.4. Pairing contract

Exact keys:

```text
architecture
budget
seed_index
training_seed
```

Pairing không dựa vào:

- row order;
- performance;
- best seed.

---

# 5.5. Training seed policy

Exact ordered seeds:

```text
01 = 204886845
02 = 1480646854
03 = 1798418854
04 = 2045683682
05 = 1814859839
06 = 1603952859
07 = 1878351743
08 = 875651179
09 = 477581743
10 = 869675675
```

Rules:

```text
same ordered list for SUP and SSL
no seed search
no replacement seed
retry must reuse same seed
```

Data construction seed:

```text
partition_seed = 42
```

Important:

\[
partition\_seed
\neq
training\_seed.
\]

---

# 5.6. Compute/exposure fairness

Matched SUP–SSL:

```text
same total optimizer-update budget
same labeled-supervision exposure
```

SSL additionally uses unlabeled data.

Tại sao?

Để giảm khả năng giải thích rằng SSL tốt hơn chỉ vì:

- train lâu hơn;
- nhận nhiều labeled updates hơn.

---

# 5.7. Validation role

Validation được phép dùng cho:

- checkpoint selection;
- predefined Qpseudo audit;
- predefined ablation;
- protocol-defined development.

Không được dùng để:

- seed search;
- architecture shopping sau freeze;
- tùy ý thay primary metric.

---

# 5.8. Test firewall

Test:

```text
FINAL ONLY
```

Không dùng test cho:

- checkpoint selection;
- detector selection;
- architecture selection;
- threshold tuning;
- augmentation selection;
- metric selection;
- statistical method selection.

---

# 5.9. Hidden-U firewall

Hidden GT của \(U_b\) bị cấm trong:

- training;
- pseudo-label filtering;
- threshold tuning;
- model selection;
- Qpseudo computation.

---

# 6. D4-A — DETECTOR / ARCHITECTURE SELECTION

# 6.1. Final selected detector family

\[
\boxed{
Faster\ R-CNN
}
\]

Architectures:

\[
\boxed{
Faster\ R-CNN
+
ResNet\text{-}50
+
FPN
}
\]

và:

\[
\boxed{
Faster\ R-CNN
+
Swin\text{-}T
+
FPN
}
\]

---

# 6.2. Official terminology

Use:

> **ResNet-50-based conventional CNN architecture**

and:

> **Swin-T-based hierarchical Vision Transformer architecture**

Swin-T:

> hierarchical Vision Transformer using shifted-window self-attention.

---

# 6.3. Tại sao giữ cùng Faster R-CNN/FPN?

Để giảm confounding từ detector family.

Shared concept:

```text
Faster R-CNN family
FPN
RPN
RoI Head family
```

Primary architecture-defining difference:

```text
ResNet-50
vs
Swin-T
```

---

# 6.4. Tại sao chọn R50?

ResNet-50:

- conventional CNN reference;
- mature detection backbone;
- broad MMDetection support;
- common Faster R-CNN baseline;
- computationally manageable for repeated-seed study.

## Evidence

ResNet paper:

https://openaccess.thecvf.com/content_cvpr_2016/html/He_Deep_Residual_Learning_CVPR_2016_paper.html

Faster R-CNN:

https://arxiv.org/abs/1506.01497

FPN:

https://openaccess.thecvf.com/content_cvpr_2017/html/Lin_Feature_Pyramid_Networks_CVPR_2017_paper.html

---

# 6.5. Tại sao chọn Swin-T?

Swin-T được chọn trước official training dựa trên:

- hierarchical feature representation;
- shifted-window attention;
- dense prediction suitability;
- object-detection integration;
- MMDetection support;
- compute phù hợp hơn các biến thể lớn cho repeated-seed design.

Không claim:

```text
Swin-T empirically better than PVT
```

vì study không chạy model-shopping.

## Evidence

Swin paper:

https://openaccess.thecvf.com/content/ICCV2021/html/Liu_Swin_Transformer_Hierarchical_Vision_Transformer_Using_Shifted_Windows_ICCV_2021_paper.html

Swin GitHub:

https://github.com/microsoft/Swin-Transformer

---

# 6.6. Framework

Official:

```text
MMDetection 3.3.0
COCO detection JSON
```

GitHub:

https://github.com/open-mmlab/mmdetection/tree/v3.3.0

R50 base config:

https://github.com/open-mmlab/mmdetection/blob/v3.3.0/configs/_base_/models/faster-rcnn_r50_fpn.py

Swin integration:

https://github.com/open-mmlab/mmdetection/blob/v3.3.0/configs/swin/mask-rcnn_swin-t-p4-w7_fpn_1x_coco.py

YOLO:

```text
OUT OF SCOPE
```

Detectron2:

```text
FALLBACK ONLY
```

---

# 6.7. Pretraining fairness

```text
R50 = ImageNet-1K backbone pretraining
Swin-T = ImageNet-1K backbone pretraining
COCO detector-level initialization = PROHIBITED
```

Tại sao?

Để external supervision tương đương về loại giữa hai architecture.

---

# 6.8. Model shopping prohibition

```text
ARCHITECTURE SEARCH = PROHIBITED
TEST-DRIVEN MODEL SELECTION = PROHIBITED
```

Không thay Swin-T bằng model khác vì kết quả tốt/xấu sau khi nhìn performance.

---

# 7. D4-B — BASE TRAINING CONFIGURATION

# 7.1. Input representation

Stored:

```text
JPEG L
grayscale
1 channel
```

Model input:

\[
1ch
\rightarrow
3\ identical\ channels
\]

với:

\[
R=G=B.
\]

Không tạo thông tin màu giả.

---

# 7.2. Image scale

Train/validation/test:

```text
scale = (1333, 800)
keep_ratio = True
pad_size_divisor = 32
```

---

# 7.3. Labeled augmentation

Main labeled stream:

```text
Load
→ Resize
→ Pack
```

No default:

- flip;
- crop;
- rotation;
- mosaic;
- aggressive photometric augmentation.

## Tại sao?

Giữ main supervised protocol conservative và tránh confound augmentation trước khi augmentation được phân tích riêng ở ablation.

---

# 7.4. Effective labeled batch

\[
\boxed{
B^L_{\mathrm{eff}}=4
}
\]

Microbatch/gradient accumulation chỉ là execution mechanism để đạt effective batch.

---

# 7.5. R50 optimizer

```text
SGD
lr = 0.005
momentum = 0.9
weight_decay = 0.0001
```

## Cơ sở

MMDetection common Faster R-CNN schedule sử dụng SGD và reference LR lớn hơn ở reference batch; study khóa architecture-specific recipe trước training.

MMDetection schedule reference:

https://github.com/open-mmlab/mmdetection/blob/v3.3.0/configs/_base_/schedules/schedule_1x.py

---

# 7.6. Swin-T optimizer

```text
AdamW
lr = 0.000025
betas = (0.9, 0.999)
weight_decay = 0.05
native no-decay rules retained
```

## Cơ sở

Swin detection recipes thường dùng AdamW và architecture-native no-decay handling.

Reference:

https://github.com/open-mmlab/mmdetection/blob/v3.3.0/configs/swin/mask-rcnn_swin-t-p4-w7_fpn_1x_coco.py

---

# 7.7. Official optimizer-update budgets

```text
1% = 1032

5% = 2064
10% = 2064
20% = 2064

100%-SUP = 10284
```

Đây là **study-specific prespecified decision**.

Không mô tả các số này như một giá trị paper bên ngoài bắt buộc.

---

# 7.8. Scheduler

For 2064-update schedule:

```text
drop 1 ≈ 1376
drop 2 ≈ 1892
```

1% uses proportional schedule positions.

---

# 7.9. Validation interval

```text
every 172 optimizer updates
```

1% schedule giữ số validation checkpoints có tính so sánh.

---

# 7.10. Checkpoint policy

BEST:

```text
selected by validation bbox mAP@[0.50:0.95]
```

Retain:

```text
BEST
LAST
```

---

# 7.11. Numerical/training rules

```text
AMP = ON
gradient clipping = OFF unless Controlled Revision
all backbone stages = trainable
architecture-native normalization retained
```

No Finding/zero-GT retained:

```text
filter_empty_gt = False
```

hoặc exact equivalent.

---

# 8. D4-C — MAIN SSL / SOFTTEACHER TREATMENT

# 8.1. Official method wording

> **Khung semi-supervised object detection teacher–student dựa trên pseudo-labeling, triển khai theo SoftTeacher.**

Không viết:

- thuật toán mới;
- phương pháp SSOD mới;
- best CXR SSL method.

---

# 8.2. Tại sao chọn SoftTeacher-style?

Soft Teacher cung cấp một teacher–student end-to-end SSOD framework với:

- online pseudo labels;
- EMA teacher;
- classification filtering;
- regression uncertainty filtering;
- strong/weak augmentation;
- Faster R-CNN reference implementation.

Nó phù hợp với mục tiêu nghiên cứu:

> đánh giá một phương pháp SSOD đại diện trong controlled CXR experiment, không đề xuất thuật toán mới.

## Evidence

Soft Teacher paper:

https://openaccess.thecvf.com/content/ICCV2021/html/Xu_End-to-End_Semi-Supervised_Object_Detection_With_Soft_Teacher_ICCV_2021_paper.html

Official repository:

https://github.com/microsoft/SoftTeacher

MMDetection guide:

https://mmdetection.readthedocs.io/en/v3.3.0/user_guides/semi_det.html

MMDetection source:

https://github.com/open-mmlab/mmdetection/blob/v3.3.0/mmdet/models/detectors/soft_teacher.py

---

# 8.3. Student / Teacher

Student:

```text
optimized by gradient descent
```

Teacher:

```text
EMA
same architecture as Student
```

Initialization:

```text
Teacher(t0) = Student(t0)
```

No supervised-checkpoint burn-in.

Resume:

```text
restore Teacher state
do not resync Teacher from Student
```

---

# 8.4. EMA rule

Locked convention:

```text
MeanTeacherHook momentum = 0.001
skip_buffers = True
```

Equivalent:

\[
\boxed{
\theta_T
=
0.999\theta_T^{old}
+
0.001\theta_S
}
\]

Teacher updated once per actual Student optimizer update.

---

# 8.5. Labeled / unlabeled data flow

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

# 8.6. Weak U augmentation

```text
Resize only
+ standard preprocessing/padding
```

No:
- flip;
- crop;
- rotation;
- photometric augmentation.

---

# 8.7. Strong U augmentation

```text
Resize
+
grayscale brightness U(0.90,1.10)
+
grayscale contrast U(0.90,1.10)
```

Constraint:

\[
R=G=B.
\]

No:
- geometric transforms;
- erasing;
- mosaic.

---

# 8.8. Pseudo-label source

```text
ONLINE
Teacher RCNN detections
```

Pseudo label consists of:

- box;
- class;
- score.

RPN proposals are not final pseudo labels.

---

# 8.9. Pseudo thresholds

```text
detector score floor = 0.05

pseudo_label_initial_score_thr = 0.50

rpn_pseudo_thr = 0.90

cls_pseudo_thr = 0.90
```

---

# 8.10. NMS / max proposals

```text
RPN:
IoU = 0.70
max = 1000

RCNN:
IoU = 0.50
max = 100
```

---

# 8.11. Regression pseudo-label reliability

```text
jitter_times = 10
jitter_scale = 0.06

reg_pseudo_thr = 0.02

min_pseudo_bbox_wh = (0.01,0.01)
```

Regression branch:

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

# 8.12. Loss

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

# 8.13. Burn-in / ramp-up

```text
burn_in = 0
lambda_unsup = 4 constant
```

No ramp-up.

---

# 8.14. L:U sampling ratio

\[
\boxed{
L:U=1:1
}
\]

Effective per optimizer update:

```text
4 labeled
4 unlabeled
```

---

# 8.15. Official SSL evaluation model

Official:

```text
EMA Teacher
```

BEST:

```text
selected by validation Teacher mAP@[0.50:0.95]
```

Final downstream model:

```text
BEST EMA Teacher
```

Student:

```text
secondary
```

---

# 8.16. Qpseudo checkpoint

Always:

```text
LAST EMA Teacher
fixed validation
```

Không sử dụng BEST Teacher để tính \(Q_{\text{pseudo}}\).

---

# 9. D4-D — ABLATION STUDY

# 9.1. Current scientific status — FINAL RESEARCH STATE

Ngày quyết định: **26/08/2026**

Researcher đã duyệt phương án:

\[
\boxed{\text{A-A-A-A}}
\]

tương ứng với:

```text
D4-D.1 = RESEARCHER_APPROVED / LOCKED
D4-D.2 = RESEARCHER_APPROVED / LOCKED
D4-D.3 = RESEARCHER_APPROVED / LOCKED
D4-D.4 = RESEARCHER_APPROVED / LOCKED
D4-D.5 = RESEARCHER_APPROVED / LOCKED
D4-D.6 = RESEARCHER_APPROVED / LOCKED
D4-D.7 = RESEARCHER_APPROVED / LOCKED
D4-D.8 = RESEARCHER_APPROVED / LOCKED

D4-D.9 =
DEFERRED TO IMPLEMENTATION / TECHNICAL PREFLIGHT

D4-D RESEARCH DESIGN =
COMPLETE / RESEARCHER_APPROVED / LOCKED

ABLATION SCIENTIFIC DESIGN =
CLOSED FOR RESEARCH-DECISION PURPOSES
```

D4-D.9 không còn được xem là một scientific/research decision gate. Nó là technical preflight/implementation closure gate và sẽ được xử lý ở giai đoạn code/implementation.

---

# 9.2. D4-D.1 — Objective / Role / Governance

## Quyết định đã khóa

**D4-D.1 = A**

Ablation Study có vai trò:

```text
SECONDARY
MECHANISTIC COMPONENT ANALYSIS
ONE-FACTOR-AT-A-TIME
NOT HYPERPARAMETER SEARCH
NOT MODEL SELECTION
NOT MAIN-PROTOCOL OPTIMIZATION
```

## Mục tiêu

Ablation được thiết kế để giải thích ảnh hưởng của các thành phần đã prespecify trong main SSL treatment lên:

1. downstream detection performance;
2. pseudo-label behavior / pseudo-label quality.

Ba component families được khảo sát:

```text
1. RCNN classification pseudo-label confidence threshold
2. strong photometric augmentation
3. bbox-regression pseudo-label reliability filtering
```

## Tại sao chọn vai trò này?

Mục tiêu ablation của luận văn là **giải thích cơ chế/thành phần**, không phải tìm cấu hình tốt nhất sau khi nhìn validation result.

Cách định vị này:

- phù hợp với mục tiêu giải thích “thành phần nào ảnh hưởng thế nào”;
- tránh biến ablation thành hyperparameter search;
- tránh dùng validation performance để sửa ngược Main D4-C;
- giữ primary scientific hypothesis RQ2 không bị thay thế;
- hạn chế multiplicity và selective reporting.

## Governance đã khóa

```text
MAIN D4-C TREATMENT =
FROZEN SCIENTIFIC ANCHOR

ABLATION RESULTS =
CANNOT RETROACTIVELY CHANGE MAIN D4-C

NEGATIVE RESULTS =
MUST BE RETAINED / REPORTED

BEST-ABLATION SHOPPING =
PROHIBITED
```

Không được:

- bỏ variant vì kết quả xấu;
- chỉ báo variant tốt hơn Main;
- thay Main threshold/augmentation/filtering sau khi xem ablation;
- nâng một finding hậu nghiệm thành formal primary claim.

---

# 9.3. D4-D.2 — Main Anchor / OFAT Invariants

## Quyết định đã khóa

**D4-D.2 = A**

Main anchor duy nhất:

```text
Detector/Architecture =
Faster R-CNN + ResNet-50 + FPN

Label budget =
10%

Seeds =
all 10 locked training seeds

Main SSL treatment =
locked D4-C configuration
```

## OFAT principle

\[
\boxed{
\text{One Factor At A Time}
}
\]

Mỗi ablation variant chỉ được thay **đúng factor thuộc family đang khảo sát**.

Tất cả scientific settings khác phải giữ bằng Main D4-C.

### Confidence family

Chỉ thay:

```text
cls_pseudo_thr
```

Giữ nguyên tối thiểu:

- architecture;
- 10% labeled membership;
- seed;
- optimizer;
- LR/schedule;
- update budget;
- EMA;
- L:U ratio;
- \(\lambda_u\);
- burn-in;
- weak augmentation;
- strong augmentation;
- regression uncertainty rule;
- NMS;
- Qpseudo definition;
- evaluator.

### Augmentation family

Chỉ thay:

```text
presence/absence of strong-view brightness and/or contrast
```

Giữ:

- weak branch;
- geometry;
- all thresholds;
- regression reliability;
- optimizer/schedule;
- all other D4-C scientific settings.

### Filtering family

Chỉ thay:

```text
bbox-regression pseudo-label reliability strategy
```

Giữ:

- classification pseudo threshold;
- augmentation;
- EMA;
- architecture;
- budget;
- seeds;
- all other D4-C scientific settings.

## Reuse main anchor

Nếu 10 Main R50/10% runs đã tồn tại và hợp lệ:

```text
REUSE = REQUIRED / PREFERRED
```

Không huấn luyện lại Main chỉ để tạo một anchor mới cho ablation, trừ khi implementation evidence chứng minh run cũ invalid.

## Tại sao chọn một anchor duy nhất?

- giảm confounding;
- cho phép paired seed-level comparison;
- tiết kiệm compute;
- giữ diễn giải component-specific;
- tránh full factorial explosion.

Full factorial:

\[
5\times4\times3=60
\]

configurations trước khi nhân seeds, nên **không được chọn** cho nghiên cứu chính.

---

# 9.4. D4-D.3 — Locked ablation scope

```text
Architecture = R50 only
Budget = 10% only
Seeds = all 10 locked seeds
```

## Tại sao R50?

- gần canonical SoftTeacher reference path;
- component analysis không nhằm trả lời architecture interaction;
- RQ7 đã xử lý architecture-dependent SSL effect;
- giảm compute;
- giữ ablation tập trung vào SSL components.

## Tại sao 10%?

- vẫn thuộc low-label regime;
- ít cực đoan hơn 1%;
- ổn định hơn cho mechanistic component analysis;
- giảm nguy cơ diễn giải từ support labeled quá thấp.

---

# 9.5. D4-D.4 — Confidence-threshold ablation

\[
cls\_pseudo\_thr
\in
\{
0.5,
0.6,
0.7,
0.8,
0.9
\}
\]

Main anchor:

\[
\boxed{0.9}
\]

Variants:

```text
C0 = 0.5
C1 = 0.6
C2 = 0.7
C3 = 0.8
C4 = 0.9 = Main
```

All other D4-C scientific settings fixed.

---

# 9.6. D4-D.5 — Strong photometric augmentation ablation

Variants:

```text
A0 = no brightness, no contrast
A1 = brightness only
A2 = contrast only
A3 = brightness + contrast = Main
```

Magnitude:

```text
brightness factor ∈ [0.90, 1.10]
contrast factor   ∈ [0.90, 1.10]
```

Grayscale constraint:

\[
R=G=B.
\]

Không thêm geometric augmentation, erasing hoặc mosaic trong D4-D family này.

---

# 9.7. D4-D.6 — Pseudo-label filtering ablation

Variants:

```text
F0:
initial score > 0.50
no additional bbox-regression reliability filtering

F1:
confidence-based filtering
score > 0.90

F2:
initial score > 0.50
AND reg_uncertainty < 0.02
= Main
```

Main:

\[
\boxed{F2}
\]

Các constants khác của main regression-reliability mechanism giữ nguyên, gồm:

```text
jitter_times = 10
jitter_scale = 0.06
reg_pseudo_thr = 0.02
```

---

# 9.8. Excluded ablations

Không bao gồm:

```text
EMA ablation = NOT INCLUDED
burn-in / ramp-up ablation = NOT INCLUDED
```

Lý do:

- không phải mọi ví dụ ablation đều cần đưa vào;
- tăng số run đáng kể;
- main objective chỉ cần ba component families đã prespecify;
- giữ ablation tractable và mechanistic.

---

# 9.9. Run-count design

Unique configurations:

\[
5+4+3-2=10.
\]

Trừ hai lần lặp Main vì:

- C4;
- A3;
- F2;

đều quy chiếu về cùng Main D4-C anchor.

Với 10 seeds:

\[
\boxed{
10\ configurations
\times
10\ seeds
=
100\ config\text{-}seed\ observations
}
\]

Nếu 10 Main R50/10% runs được reuse:

\[
\boxed{
90\ additional\ training\ runs
}
\]

---

# 9.10. D4-D.7 — Ablation Metrics / Qpseudo Diagnostics

## Quyết định đã khóa

**D4-D.7 = A**

Ablation sử dụng hierarchy hai tầng:

### Primary ablation downstream outcome

\[
\boxed{
validation\ bbox\ mAP@[0.50:0.95]
}
\]

### Secondary downstream metrics

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

Qpseudo giữ nguyên định nghĩa D1.4/D5:

```text
fixed validation set
LAST EMA Teacher
accepted RCNN classification pseudo set
```

### Additional mechanistic diagnostics

Có thể báo:

- PL-AP50;
- PL-AP75;
- PL-Precision@IoU=.50;
- PL-Recall@IoU=.50;
- PL-F1@IoU=.50;
- mean matched IoU;
- pseudo-label count;
- pseudo-label retention/acceptance rate;
- pseudo FP / confirmed-negative validation image;
- regression uncertainty diagnostics.

## Tại sao chọn hierarchy này?

Ablation cần trả lời đồng thời:

\[
\text{component}
\rightarrow
\text{pseudo-label behavior}
\rightarrow
\text{downstream detection performance}.
\]

Nếu chỉ dùng downstream mAP:

- biết performance thay đổi;
- nhưng khó giải thích mechanism.

Nếu chỉ dùng Qpseudo:

- biết pseudo labels thay đổi;
- nhưng chưa biết detector downstream thay đổi thế nào.

Do đó:

\[
\boxed{
mAP_{val}
=
primary\ ablation\ outcome
}
\]

và:

\[
\boxed{
Q_{\mathrm{pseudo}}
=
key\ mechanistic\ secondary\ outcome
}
\]

là hierarchy đã khóa.

## Không tạo co-primary endpoint

Qpseudo **không** là co-primary ablation endpoint.

Không thay study-wide primary endpoint của RQ2.

---

# 9.11. D4-D.8 — Reporting / Statistical / Interpretation Contract

## Quyết định đã khóa

**D4-D.8 = A**

Ablation analysis là:

```text
ESTIMATION-FOCUSED
PAIRED-SEED ANALYSIS
NO FORMAL ABLATION P-VALUE
NO NEW MULTIPLICITY FAMILY
NO CHANGE TO D6 CONFIRMATORY FAMILIES
```

## Seed-level paired effect

Với metric \(M\), variant \(v\), seed \(s\):

\[
\boxed{
\Delta M_{v,s}
=
M_{v,s}
-
M_{\mathrm{Main},s}
}
\]

Với Qpseudo:

\[
\boxed{
\Delta Q_{v,s}
=
Q_{v,s}
-
Q_{\mathrm{Main},s}
}
\]

Pairing dùng cùng:

```text
training_seed_index
training_seed
```

## Report cho mỗi variant

Ít nhất:

1. variant mean;
2. sample SD (`ddof=1`);
3. Main mean;
4. paired mean difference;
5. two-sided 95% CI của paired difference;
6. seed-wise paired visualization hoặc bảng paired values;
7. direction và magnitude của effect.

## Không formal p-value

Không thực hiện official hypothesis test cho từng ablation variant.

Do đó không cần:

- mở thêm D6 confirmatory family;
- thêm Holm family cho 9 ablation comparisons;
- định nghĩa “significant ablation winner”.

## Tại sao estimation-focused?

D4-D có vai trò:

```text
SECONDARY / MECHANISTIC
```

không phải confirmatory hypothesis family.

Estimation + 95% CI cung cấp:

- magnitude;
- direction;
- uncertainty;
- seed consistency;

mà không làm phình hệ thống multiplicity hiện có.

## Interpretation rule

Không viết:

> Variant X có ý nghĩa thống kê và vì vậy nên thay Main.

Được viết theo hướng:

> So với cấu hình Main, variant X cho chênh lệch trung bình ... AP points (95% CI ...), đồng thời Qpseudo thay đổi ..., cho thấy component này có liên hệ với thay đổi về pseudo-label behavior và downstream performance trong phạm vi ablation đã prespecify.

Negative/null results vẫn phải báo cáo.

---

# 9.12. Validation / Test contract cho D4-D

Non-main ablation variants:

```text
VALIDATION ONLY
NO FINAL TEST
```

Ablation không được dùng test để:

- chọn variant;
- chọn threshold;
- chọn augmentation;
- chọn filtering;
- chọn “best ablation”.

Main D4-C final-test treatment vẫn giữ nguyên.

---

# 9.13. D4-D.9 — Deferred to implementation

D4-D.9 là:

```text
TECHNICAL PREFLIGHT / IMPLEMENTATION CLOSURE GATE
```

Nó **không còn là research-decision gap**.

Sẽ xử lý sau trong implementation, gồm các kiểm tra như:

- config diff đúng factor;
- all non-target scientific settings identical;
- confidence variant đúng threshold;
- augmentation variant đúng A0–A3;
- grayscale equality preserved;
- filtering variant đúng F0–F2;
- seed mapping;
- artifact/provenance;
- runtime/guardrails.

Current research status:

```text
D4-D SCIENTIFIC DESIGN =
COMPLETE / LOCKED

D4-D IMPLEMENTATION PREFLIGHT =
DEFERRED
```

---

# 9.14. D4-D final research decision summary

```text
D4-D.1 =
SECONDARY / MECHANISTIC / OFAT
NOT HYPERPARAMETER SEARCH
LOCKED

D4-D.2 =
ONE MAIN D4-C R50/10% ANCHOR
ONE FACTOR CHANGES
ALL OTHERS FROZEN
LOCKED

D4-D.3 =
R50 / 10% / ALL 10 SEEDS
LOCKED

D4-D.4 =
CONFIDENCE {0.5,0.6,0.7,0.8,0.9}
LOCKED

D4-D.5 =
A0/A1/A2/A3 BRIGHTNESS-CONTRAST
LOCKED

D4-D.6 =
F0/F1/F2 FILTERING
LOCKED

D4-D.7 =
PRIMARY ABLATION OUTCOME = VAL mAP@[.50:.95]
KEY MECHANISTIC = Qpseudo
AP50/AP75 + mechanistic diagnostics secondary
LOCKED

D4-D.8 =
ESTIMATION-FOCUSED PAIRED ANALYSIS
MEAN / SD / PAIRED DIFFERENCE / 95% CI
NO FORMAL P
NO NEW MULTIPLICITY FAMILY
LOCKED

D4-D.9 =
DEFERRED TO IMPLEMENTATION
```

\[
\boxed{
D4\text{-}D\ RESEARCH\ DESIGN
=
COMPLETE / RESEARCHER\ APPROVED / LOCKED
}
\]

---

# 10. D5 — EVALUATION METRICS

# 10.1. Study-wide primary endpoint

\[
\boxed{
bbox\ mAP@[0.50:0.95]
}
\]

## Tại sao?

COCO-style mAP integrates detection quality across multiple IoU thresholds, phản ánh localization + classification rigorously hơn AP50 đơn lẻ.

---

# 10.2. General secondary metrics

```text
AP50
AP75
COCO AR@[.50:.95]
Recall@operating point
FP/image
all 14 class AP
```

---

# 10.3. COCO evaluation contract

```text
bbox only
IoU thresholds = .50:.05:.95
recall thresholds = .00:.01:1.00
useCats = 1
effective maxDet = 100
```

No extra score threshold for AP.

No official COCO APs/m/l because Phase 3A uses study-specific normalized-area definitions.

---

# 10.4. COCO evidence

COCO paper:

https://arxiv.org/abs/1405.0312

COCO API:

https://github.com/cocodataset/cocoapi

COCOeval source:

https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/cocoeval.py

MMDetection CocoMetric:

https://github.com/open-mmlab/mmdetection/blob/v3.3.0/mmdet/evaluation/metrics/coco_metric.py

---

# 10.5. Rare-class metrics

Frozen:

```text
Atelectasis
Pneumothorax
```

Official:

\[
AP_c@[0.50:0.95].
\]

No rare-mAP.

No class-specific checkpoint selection.

---

# 10.6. Recall

Official general secondary:

```text
COCO AR@[0.50:0.95]
maxDets=100
area=all
```

Additional diagnostic:

```text
Recall@IoU=.50
at fixed operating threshold
```

---

# 10.7. Fixed operating point

Global:

\[
\boxed{
\tau_{eval}=0.50
}
\]

Matching:

```text
category-aware
one-to-one
same class
IoU >= .50
after detector-native NMS/max100
then score >= .50
```

\[
FP/image
=
\frac{\sum_i FP_i}{N}.
\]

---

# 10.8. No Finding evaluation

No Finding:

```text
NOT a detection class
zero-GT evaluation stratum
```

Fixed:
- validation negative = 75;
- test negative = 75.

Primary within RQ6:

\[
\boxed{
FP/negative
=
\frac{\sum_{i\in\mathcal N}FP_i}{|\mathcal N|}
}
\]

Secondary:

\[
FAR_{neg}
=
\frac{
\#\{i:FP_i\ge1\}
}{
|\mathcal N|
}.
\]

---

# 10.9. Qpseudo metric protocol

\[
\boxed{
Q_{\mathrm{pseudo}}
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

Test Qpseudo:

```text
PROHIBITED
```

Hidden U GT:

```text
PROHIBITED
```

Secondary:
- PL-AP50;
- PL-AP75;
- Precision/Recall/F1;
- mean matched IoU;
- pseudo count;
- retention;
- class PL-AP;
- PL-FP/negative;
- regression uncertainty.

---

# 10.10. Seed aggregation

Atomic observation:

```text
valid trained model / training seed
```

Each main condition:

\[
n=10.
\]

Report:

- arithmetic mean;
- sample SD;
- `ddof=1`.

No cross-seed prediction pooling for statistical replication.

---

# 10.11. Prediction/evaluation provenance

Persistent prediction JSON:

```text
image_id
category_id
bbox xywh
score
```

Raw full precision metrics are source of truth.

Relevant hashes should include:
- checkpoint;
- predictions;
- GT;
- config;
- mapping;
- environment;
- code.

---

# 11. D6 — STATISTICAL ANALYSIS

# 11.1. Inferential unit

\[
\boxed{
training\ seed / trained\ run
}
\]

Images/bboxes/detections are not independent training-method replications.

Main n:

\[
10\ seeds.
\]

General alpha:

\[
\boxed{\alpha=.05}
\]

Reporting philosophy:

\[
\boxed{
effect
\rightarrow
95\%CI
\rightarrow
p
}
\]

---

# 11.2. Paired seed-level effects

Primitive:

\[
\Delta M_{a,b,s}
=
M^{SSL}_{a,b,s}
-
M^{SUP}_{a,b,s}.
\]

Main effect scale:

```text
absolute metric difference
```

Relative % change:

```text
supplementary descriptive only
```

---

# 11.3. RQ2 formal analysis

\[
G_s
=
\frac18
\sum_{a,b}
\Delta mAP_{a,b,s}.
\]

Formal:

\[
H_0:\mu_G\le0
\]

\[
H_1:\mu_G>0.
\]

Test:

```text
one-sample t-test
n=10
df=9
one-sided p
two-sided 95% CI
```

Eight cell effects:
- descriptive mean;
- SD;
- CI;
- no primary cell p-values.

---

# 11.4. RQ3 formal analysis

\[
B_{b,s}
=
\frac{
\Delta_{R50,b,s}
+
\Delta_{Swin,b,s}
}{2}.
\]

Formal null:

\[
H_0:
\mu_1=
\mu_5=
\mu_{10}=
\mu_{20}.
\]

Primary:

```text
one-way repeated-measures ANOVA
Greenhouse-Geisser always
```

Report:
- F;
- \(\epsilon_{GG}\);
- adjusted df;
- \(p_{GG}\).

Pairwise:
- six prespecified contrasts;
- Holm within F3.

No primary linear/monotonic trend.

---

# 11.5. RQ7 formal analysis

Architecture-specific gain:

\[
A_{a,s}
=
\frac14
\sum_b
\Delta_{a,b,s}.
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

\[
H_0:\mu_D=0
\]

\[
H_1:\mu_D\ne0.
\]

Two-sided one-sample t-test.

---

# 11.6. RQ4 formal analysis

Outcome:

\[
Y_{a,b,s}
=
\Delta mAP^{test}_{a,b,s}.
\]

Predictor:

\[
Q_{\mathrm{pseudo}}
=
PL\text{-}mAP@[.50:.95].
\]

Within-cell centered:

\[
Q^{WC}_{a,b,s}
=
Q_{a,b,s}
-
\bar Q_{a,b}.
\]

Scale:

\[
X=Q^{WC}/.01.
\]

Model:

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

Inference:

```text
REML
Kenward-Roger
```

Formal:

\[
H_0:\beta_Q\le0
\]

\[
H_1:\beta_Q>0.
\]

80 observations are **not 80 independent training seeds**.

---

# 11.7. RQ5 analysis

Per rare class:

\[
G_{c,s}
=
\frac18
\sum_{a,b}
\Delta AP_{c,a,b,s}.
\]

Report:
- mean;
- sample SD;
- two-sided 95% CI.

No required formal p.

Zero-GT class support:
- AP undefined;
- not zero.

---

# 11.8. RQ6 analysis

\[
\Delta F_{a,b,s}
=
FP^{SSL}_{neg}
-
FP^{SUP}_{neg}.
\]

\[
H_s
=
\frac18
\sum_{a,b}
\Delta F_{a,b,s}.
\]

Formal:

```text
two-sided one-sample t-test
n=10
df=9
```

Negative effect:

```text
fewer FP under SSL
```

FAR:
- estimation-focused;
- no official formal p.

---

# 11.9. Multiplicity structure

## Family F1

```text
RQ2 only
m=1
no adjustment
```

## Family F2

```text
RQ3
RQ4
RQ6
RQ7

m=4
Holm step-down
FWER alpha=.05
```

Inputs:
- RQ3 \(p_{GG}\);
- RQ4 one-sided KR p;
- RQ6 two-sided p;
- RQ7 two-sided p.

## Family F3

Six RQ3 budget contrasts:

```text
Holm
m=6
```

Formal RQ3 localization requires:
1. omnibus survives F2;
2. pair survives F3.

No family shrinking when one p is missing.

---

# 11.10. Holm evidence

Holm (1979):

https://www.jstor.org/stable/4615733

R:

https://stat.ethz.ch/R-manual/R-devel/library/stats/html/p.adjust.html

Python/statsmodels:

https://www.statsmodels.org/stable/generated/statsmodels.stats.multitest.multipletests.html

GitHub:

https://github.com/statsmodels/statsmodels/blob/main/statsmodels/stats/multitest.py

---

# 11.11. Reporting rules

Official effect size:

```text
unstandardized absolute effect
```

Not official:
- Cohen d;
- Hedges g;
- eta-squared.

mAP/AP:
- convert raw 0–1 to AP points;
- 2 decimals.

RQ4 slope:
- downstream AP-point change per +1 pseudo AP point;
- 3 decimals.

FP:
- 3 decimals.

FAR:
- percentage-point difference;
- 2 decimals.

p:
- 3 decimals;
- `p<.001`;
- never `p=0.000`.

95% CIs:
- two-sided;
- ordinary effect-specific intervals;
- not simultaneous Holm-adjusted intervals.

---

# 11.12. ASA reporting evidence

Wasserstein & Lazar (2016):

https://doi.org/10.1080/00031305.2016.1154108

Use:
- avoid p-only conclusions;
- effect/uncertainty first;
- no “marginal significance”.

---

# 11.13. Robustness / sensitivity

Primary tests remain frozen.

Diagnostics do not choose the primary method after seeing results.

## RQ2/RQ6/RQ7

Exact sign-flip:

\[
2^{10}=1024
\]

configurations.

LOSO:
- exactly 10 leave-one-seed-out analyses.

## RQ3

Primary:
- GG RM-ANOVA.

Friedman:
- sensitivity only.

## RQ4

Diagnostics:
- convergence;
- Hessian;
- design rank;
- singularity;
- residual-vs-fitted;
- Q-Q;
- seed influence.

Robustness model:

\[
\Delta mAP_{test}
\sim
Q^{WC}
+
Cell
\]

with:
- CR2;
- cluster=training seed;
- Satterthwaite df.

CR2 is sensitivity only and does not enter F2.

---

# 11.14. Robustness evidence

Pustejovsky & Tipton:

https://doi.org/10.1080/07350015.2016.1247004

clubSandwich:

https://github.com/jepusto/clubSandwich

SciPy permutation test:

https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.permutation_test.html

SciPy GitHub:

https://github.com/scipy/scipy

---

# 11.15. Run validity principle

\[
\boxed{
run\ validity
=
technical/protocol\ evidence
\neq
performance
}
\]

A valid but poor-performing run remains valid.

No:
- best-seed selection;
- performance-based rerun;
- replacement seed;
- imputation;
- n=9 official inference.

Technical failure:
- same seed retry.

---

# 11.16. Test firewall

Sequence:

\[
LOCK
\rightarrow
FREEZE
\rightarrow
TEST
\rightarrow
STATISTICS
\rightarrow
REPORT.
\]

Test is final-only.

After unblinding:
- no architecture change;
- no threshold change;
- no checkpoint-selection change;
- no metric change;
- no hypothesis change;
- no statistical-model change;
- no multiplicity change.

## Evidence

Reusable Holdout — Dwork et al.:

https://doi.org/10.1126/science.aaa9375

---

# 12. THESIS TERMINOLOGY — THUẬT NGỮ PHẢI GIỮ ĐỒNG BỘ

Use consistently:

```text
semi-supervised object detection (SSOD)

supervised baseline (SUP)

semi-supervised learning / SSL

teacher–student pseudo-labeling

SoftTeacher-style teacher–student framework

ResNet-50-based conventional CNN architecture

Swin-T-based hierarchical Vision Transformer architecture

labeled-data budget

fixed train/validation/test split

training seed

partition seed

No Finding / zero-GT negative image

pseudo-label quality

bbox mAP@[0.50:0.95]

FP/negative image

architecture-dependent SSL effect
```

Không đổi tùy đoạn giữa:

```text
backbone
architecture
detector
model
```

nếu chúng đang mang các nghĩa khác nhau.

---

# 13. CLAIM GUARDRAILS KHI VIẾT LUẬN VĂN

Không viết:

> SSL chắc chắn tốt hơn supervised learning.

Viết:

> Đây là primary research hypothesis cần được kiểm định.

Không viết:

> Swin-T tốt hơn ResNet-50.

Viết:

> Nghiên cứu khảo sát liệu SSL gain có phụ thuộc architecture hay không.

Không viết:

> Qpseudo tốt gây ra downstream improvement.

Viết:

> Qpseudo cao hơn được giả thuyết có association dương với downstream gain.

Không viết:

> Rare classes được cải thiện có ý nghĩa thống kê.

trừ khi sau này có analysis phù hợp; RQ5 current role là exploratory.

Không viết:

> No Finding là lớp thứ 15.

No Finding:

```text
zero-GT negative-image stratum
NOT detection category
```

---

# 14. STUDY-SPECIFIC DECISIONS — KHÔNG ĐƯỢC GÁN CHO PAPER

Các quyết định dưới đây là của đề tài:

- RQ2 = primary;
- RQ3/RQ4/RQ6/RQ7 = secondary;
- RQ5 = exploratory;
- 1/5/10/20%;
- 10 exact seeds;
- seed list;
- 160 main runs;
- 20 100%-SUP references;
- R50 vs Swin-T design;
- exact update budgets;
- exact validation interval;
- \(L:U=1:1\);
- \(\lambda_u=4\);
- cls threshold=.90;
- Qpseudo = LAST Teacher PL-mAP;
- tau_eval=.50;
- F2 membership;
- F3 membership;
- 1/8 equal weighting;
- D4-D R50/10% scope.

Paper/citation chỉ hỗ trợ:
- methodological principle;
- architecture definition;
- evaluator;
- statistical method;
- implementation capability.

---

# 15. MASTER EXTERNAL EVIDENCE / CITATION LIST

## Architecture / detection

### Faster R-CNN

Ren et al. (2015)

https://arxiv.org/abs/1506.01497

Suggested key:

```text
ren2015fasterrcnn
```

### ResNet

He et al. (2016)

https://openaccess.thecvf.com/content_cvpr_2016/html/He_Deep_Residual_Learning_CVPR_2016_paper.html

DOI:

https://doi.org/10.1109/CVPR.2016.90

Suggested key:

```text
he2016resnet
```

### FPN

Lin et al. (2017)

https://openaccess.thecvf.com/content_cvpr_2017/html/Lin_Feature_Pyramid_Networks_CVPR_2017_paper.html

DOI:

https://doi.org/10.1109/CVPR.2017.106

Suggested key:

```text
lin2017fpn
```

### Swin Transformer

Liu et al. (2021)

https://openaccess.thecvf.com/content/ICCV2021/html/Liu_Swin_Transformer_Hierarchical_Vision_Transformer_Using_Shifted_Windows_ICCV_2021_paper.html

GitHub:

https://github.com/microsoft/Swin-Transformer

Suggested key:

```text
liu2021swin
```

---

## SSOD

### STAC

https://arxiv.org/abs/2005.04757

https://github.com/google-research/ssl_detection

Suggested key:

```text
sohn2020simple
```

### Unbiased Teacher

https://arxiv.org/abs/2102.09480

Suggested key:

```text
liu2021unbiased
```

### Soft Teacher

https://openaccess.thecvf.com/content/ICCV2021/html/Xu_End-to-End_Semi-Supervised_Object_Detection_With_Soft_Teacher_ICCV_2021_paper.html

https://github.com/microsoft/SoftTeacher

Suggested key:

```text
xu2021end
```

---

## Evaluation

### COCO

Lin et al. (2014)

https://arxiv.org/abs/1405.0312

COCO API:

https://github.com/cocodataset/cocoapi

Suggested key:

```text
lin2014microsoft
```

---

## Statistics

### Greenhouse–Geisser

https://doi.org/10.1007/BF02289823

Suggested key:

```text
greenhouse1959profile
```

### Centering

Enders & Tofighi:

https://doi.org/10.1037/1082-989X.12.2.121

Suggested key:

```text
enders2007centering
```

### Kenward–Roger

https://doi.org/10.2307/2533558

Suggested key:

```text
kenward1997small
```

### pbkrtest

https://doi.org/10.18637/jss.v059.i09

Suggested key:

```text
halekoh2014pbkrtest
```

### Holm

https://www.jstor.org/stable/4615733

Suggested key:

```text
holm1979sequentially
```

### ASA p-value statement

https://doi.org/10.1080/00031305.2016.1154108

Suggested key:

```text
wasserstein2016asa
```

### CR2

https://doi.org/10.1080/07350015.2016.1247004

Suggested key:

```text
pustejovsky2018small
```

### Reusable holdout / adaptive data analysis

https://doi.org/10.1126/science.aaa9375

Suggested key:

```text
dwork2015reusable
```

---

# 16. MASTER GITHUB / OFFICIAL DOCUMENTATION LIST

MMDetection v3.3.0:

https://github.com/open-mmlab/mmdetection/tree/v3.3.0

Faster R-CNN R50-FPN base:

https://github.com/open-mmlab/mmdetection/blob/v3.3.0/configs/_base_/models/faster-rcnn_r50_fpn.py

Schedule:

https://github.com/open-mmlab/mmdetection/blob/v3.3.0/configs/_base_/schedules/schedule_1x.py

Swin integration:

https://github.com/open-mmlab/mmdetection/blob/v3.3.0/configs/swin/mask-rcnn_swin-t-p4-w7_fpn_1x_coco.py

MMDetection semi-supervised guide:

https://mmdetection.readthedocs.io/en/v3.3.0/user_guides/semi_det.html

MMDetection SoftTeacher source:

https://github.com/open-mmlab/mmdetection/blob/v3.3.0/mmdet/models/detectors/soft_teacher.py

Official SoftTeacher:

https://github.com/microsoft/SoftTeacher

Swin:

https://github.com/microsoft/Swin-Transformer

STAC:

https://github.com/google-research/ssl_detection

COCO API:

https://github.com/cocodataset/cocoapi

afex:

https://github.com/cran/afex

lme4:

https://github.com/lme4/lme4

lmerTest:

https://github.com/cran/lmerTest

pbkrtest:

https://github.com/hojsgaard/pbkrtest

clubSandwich:

https://github.com/jepusto/clubSandwich

statsmodels:

https://github.com/statsmodels/statsmodels

SciPy:

https://github.com/scipy/scipy

---

# 17. BẢN ĐỒ D1–D6 → CẤU TRÚC LUẬN VĂN

## Introduction / Research Questions

Dùng:
- RQ1–RQ7;
- scientific positioning;
- primary/secondary/exploratory distinction ở mức ngắn gọn.

Không đưa full statistical formulas.

---

## Methodology — Research Hypotheses

Dùng:
- RQ2 primary directional;
- RQ3 secondary non-directional;
- RQ4 secondary mechanistic positive association;
- RQ5 exploratory;
- RQ6/RQ7 secondary non-directional.

---

## Research Variables

D2:
- X1 Method;
- X2 Budget;
- X3 Architecture;
- 7 outcomes;
- controls;
- interactions.

---

## Controlled Experimental Design

D3:
- 2×4×2;
- 10 seeds;
- 160 low-label;
- 20 full-SUP references;
- pairing;
- same compute/exposure;
- validation/test roles.

---

## Architecture and Supervised Baseline

D4-A/B:
- Faster R-CNN;
- R50/FPN;
- Swin-T/FPN;
- pretraining;
- input;
- optimizer;
- training schedule;
- checkpoint policy.

---

## Semi-supervised Framework

D4-C:
- Student/Teacher;
- EMA;
- weak/strong;
- pseudo-label thresholds;
- filtering;
- loss;
- L:U;
- Teacher evaluation;
- Qpseudo.

---

## Ablation Study

D4-D:
- chỉ viết phần đã khóa;
- confidence;
- augmentation;
- filtering;
- R50/10%/10 seeds;
- OFAT;
- validation-only non-main.

Không viết rằng toàn D4-D đã closed/pass.

---

## Evaluation Metrics

D5:
- primary mAP;
- AP50/AP75/AR;
- rare classes;
- Qpseudo;
- FP;
- No Finding;
- operating point.

---

## Statistical Analysis

D6:
- seed as inferential unit;
- RQ2;
- RQ3;
- RQ7;
- RQ4;
- RQ5;
- RQ6;
- Holm;
- CI/reporting;
- robustness;
- test firewall.

---

# 18. TRẠNG THÁI NÀO KHÔNG ĐƯỢC VIẾT THÀNH KẾT QUẢ

File này là **Methodology/decision source**, không phải Results.

Không viết:

- SSL improved by X%;
- Swin-T better than R50;
- p-value actual;
- Holm-significant actual;
- pseudo-label association actual;
- ablation winner;
- test performance;
- actual final conclusions.

Những nội dung trên chỉ được viết sau execution/evaluation.

---

# 19. RANH GIỚI GIỮA PRESPECIFIED METHOD VÀ EXECUTED RESULT

Hiện có thể viết:

> Nghiên cứu quy định sử dụng...

> Giao thức thực nghiệm xác định...

> Giả thuyết nghiên cứu được xác định trước...

> Hiệu năng được đánh giá bằng...

> Phân tích thống kê được quy định sử dụng...

Không viết:

> Mô hình đã đạt...

> SSL đã cải thiện...

> Kiểm định cho thấy...

nếu chưa có result evidence.

---

# 20. CURRENT SCIENTIFIC CHECKLIST

```text
[✓] RQ1–RQ7 current
[✓] RQ4 formally locked
[✓] Architecture revision incorporated
[✓] 3 IV + 7 DV
[✓] 160 low-label design
[✓] 20 full-SUP references
[✓] R50 and Swin-T selected
[✓] D4-B current training contract
[✓] Main SoftTeacher treatment defined
[✓] Qpseudo defined
[✓] D5 metric hierarchy defined
[✓] D6 statistical estimands defined
[✓] Holm families defined
[✓] test firewall defined
[✓] historical contradictions removed from current-state narrative
[✓] D4-D.1 Objective/Role/Governance locked
[✓] D4-D.2 Main Anchor/OFAT invariants locked
[✓] D4-D.7 Ablation metrics/Qpseudo hierarchy locked
[✓] D4-D.8 Estimation-focused reporting/statistical contract locked
[✓] D4-D research design complete
[✓] D4-D.9 explicitly deferred to implementation
[✓] Research Design D1–D6 complete / locked
```

---

# 21. SOURCE PROVENANCE NOTE

File này được dựng từ:

```text
MASTER_D1_D6_FULL_VERBATIM_MERGED_RECORD_VI
```

nhưng **không sao chép các historical states đã superseded**.

Ví dụ đã loại khỏi current narrative:

```text
D1 = only RQ1–RQ6
RQ4 = PROVISIONAL
D2 = only 2 IV
D4-B = PROPOSED
D4-D = fully locked
```

Current-state thay thế tương ứng:

```text
D1 = RQ1–RQ7
RQ4 = FORMALLY LOCKED
D2 = 3 IV
D4-B = LOCKED / COMPLETED
D4-D.1–D4-D.8 = RESEARCHER_APPROVED / LOCKED
D4-D.9 = DEFERRED TO IMPLEMENTATION
D4-D RESEARCH DESIGN = COMPLETE / LOCKED
```

---

# 22. GOVERNANCE CHO CÁC REVISION SAU NÀY

Nếu có thay đổi scientific decision sau file này:

1. Không sửa âm thầm.
2. Tạo Controlled Revision.
3. Ghi:
   - target;
   - previous state;
   - new state;
   - reason;
   - downstream impact.
4. Cập nhật:
   - `vietluanvan.md`;
   - implementation master;
   - thesis Methodology nếu phần đó đã viết.

Mục tiêu:

\[
\boxed{
Scientific\ source
=
Thesis\ method
=
Implementation\ scientific\ values
}
\]

---


# 22.1. RESEARCH DESIGN CLOSURE — 26/08/2026

Researcher đã phê duyệt quyết định cuối cùng cho các research gates còn mở:

```text
D4-D.1 = A
D4-D.2 = A
D4-D.7 = A
D4-D.8 = A
```

Do đó:

```text
D1 = COMPLETE / LOCKED
D2 = COMPLETE / LOCKED
D3 = COMPLETE / LOCKED

D4-A = COMPLETE / LOCKED
D4-B = COMPLETE / LOCKED
D4-C SCIENTIFIC PROTOCOL = COMPLETE / LOCKED
D4-D SCIENTIFIC DESIGN D4-D.1–D4-D.8 = COMPLETE / LOCKED

D5 SCIENTIFIC PROTOCOL = COMPLETE / LOCKED
D6 SCIENTIFIC PROTOCOL = COMPLETE / LOCKED

D4-D.9 = DEFERRED TO IMPLEMENTATION
IMPLEMENTATION/PREFLIGHT = OUTSIDE CURRENT RESEARCH-CLOSURE SCOPE
```

Formal research state:

\[
\boxed{
\text{RESEARCH DESIGN D1–D6}
=
\text{COMPLETE / RESEARCHER APPROVED / LOCKED}
}
\]

Từ thời điểm này, bất kỳ thay đổi nào đối với scientific values trong D1–D6 phải đi qua **Controlled Revision**.

Không được thay đổi scientific protocol chỉ vì:

- implementation thuận tiện hơn;
- validation result không như kỳ vọng;
- một ablation variant tốt hơn Main;
- một seed cho kết quả xấu;
- test result gợi ý một cấu hình khác.

Scientific source-of-truth để viết luận văn:

```text
vietluanvan.md
```

Historical archive vẫn được giữ riêng để audit provenance.

---

# 23. KẾT LUẬN

`vietluanvan.md` là **nguồn current-state sạch để viết luận văn**.

Nó không thay historical archive.

Historical archive trả lời:

> “Quá trình ra quyết định đã diễn ra như thế nào?”

`vietluanvan.md` trả lời:

> “Quyết định khoa học cuối cùng hiện tại là gì, tại sao chọn, dùng bằng chứng/citation nào, và phải viết Methodology như thế nào?”

Scientific chain hiện tại:

\[
\boxed{
D1
\rightarrow
D2
\rightarrow
D3
\rightarrow
D4
\rightarrow
D5
\rightarrow
D6
}
\]

đã hoàn tất về mặt **research/scientific decision**.

\[
\boxed{
\text{RESEARCH DESIGN D1–D6}
=
\text{COMPLETE / RESEARCHER APPROVED / LOCKED}
}
\]

D4-D.9 và các preflight/runtime guardrails thuộc **implementation**, được để lại cho giai đoạn sau và không còn là research-decision gap.

Tất cả scientific decisions trong file này phải được giữ nhất quán khi chuyển sang cả:

- luận văn;
- implementation.

---

**END OF VIETLUANVAN CURRENT SCIENTIFIC RECORD**
