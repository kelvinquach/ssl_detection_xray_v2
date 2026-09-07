# BẢN BÀN GIAO TRIỂN KHAI CUỐI CÙNG
## Semi-supervised Object Detection cho phát hiện bất thường trên X-quang ngực

**Mục đích:** Bàn giao cho co-worker để bắt đầu implementation theo đúng Methodology đã được giảng viên chốt.  
**Scientific source of truth:** `sn-article.tex`.  
**Trạng thái:** `FINAL IMPLEMENTATION HANDOFF / PRE-IMPLEMENTATION`.  
**Nguyên tắc ưu tiên:** Nếu bất kỳ code/config nào xung đột với Methodology, phải sửa implementation để khớp Methodology. Không được tự thay đổi thiết kế nghiên cứu cho phù hợp với code. Mọi thay đổi khoa học sau bàn giao phải được xử lý như một controlled revision và báo lại researcher.

---

# 1. PHẠM VI BÀN GIAO

Tài liệu này chuyển các quyết định đã khóa trong Methodology thành đặc tả triển khai. Co-worker được phép quyết định các chi tiết kỹ thuật không làm thay đổi ý nghĩa khoa học, ví dụ cấu trúc thư mục, tên helper function, cách chia module, microbatch/gradient accumulation để đạt đúng effective batch, hoặc cơ chế logging.

Co-worker **không được tự thay đổi** các nội dung sau:

- train/validation/test membership;
- labeled/unlabeled membership;
- đơn vị labeled-data budget;
- partition seed và training-seed policy;
- architecture pair;
- SUP--SSL pairing;
- optimizer/update budget;
- SSL treatment và pseudo-label protocol;
- EMA rule;
- augmentation đã khóa;
- checkpoint-selection rule;
- validation/test firewall;
- hidden-unlabeled-GT firewall;
- metric hierarchy;
- statistical analysis protocol;
- ablation scope và OFAT rules.

---

# Runtime tái lập môi trường DOCKER - VAST: quachthainguyen/sslxray-runtime@sha256:fab1b1ea8850527e5728de526b3d53f063355e84cb3a0d8c75e3a7890411d989

# 1A. GHI CHÚ / YÊU CẦU CỦA GIẢNG VIÊN HƯỚNG DẪN — GOVERNANCE BẮT BUỘC

Phần này lưu các yêu cầu trực tiếp của giảng viên hướng dẫn để co-worker hiểu
không chỉ **giá trị cấu hình cần implement**, mà còn **lý do kiểm soát và các lỗi
implementation cần chủ động phòng tránh**.

## 1A.1. Data split và labeled-data budget

Giảng viên đồng ý giữ:

```text
fixed train/validation/test = 70% / 15% / 15%
nested labeled subsets = 1% ⊂ 5% ⊂ 10% ⊂ 20%
U_b = T \ L_b
```

Yêu cầu implementation:

- đơn vị sampling phải là **image**, tuyệt đối không vô tình chuyển thành
  bbox-level sampling;
- `1%`, `5%`, `10%`, `20%` phải luôn được hiểu là **image-level
  labeled-data budget**;
- số ảnh chính thức là `34 / 171 / 343 / 685`;
- một ảnh được chọn vào `L_b` thì toàn bộ ground-truth bounding boxes của ảnh đó
  được xem là available supervision.

## 1A.2. 1% subset và 14-class coverage

Giảng viên xem đây là điểm reviewer có thể hỏi trực tiếp.

Implementation phải phản ánh đúng rằng:

```text
1% = 34 images
final subset must cover all 14 abnormality classes
construction is NOT pure random sampling
```

Quy trình là:

```text
iterative multilabel stratification
→ exact-size repair
→ exact-No-Finding repair
→ minimum-class-coverage repair if needed
→ objective repair / one-for-one local search
```

Có sử dụng image-level class-presence information để xây subset. Vì vậy phải mô
tả/ghi log đây là **label-aware, coverage-constrained subset construction**.

`partition_seed = 42` được dùng cho phần data-partition / subset construction và
deterministic tie-break theo protocol. Không được seed-search.

Kết quả đã xác minh cho official 1% membership:

```text
final size = 34
coverage = 14/14 classes
minimum-class-coverage repair move count = 0
```

Điều cuối cùng có nghĩa là dedicated coverage-repair không cần swap trong run
chính thức; không được viết sai rằng coverage repair đã "tạo ra" 14/14 coverage.

## 1A.3. Partition seed khác training seed

Giảng viên yêu cầu verify bằng implementation, không chỉ ghi trên giấy.

Bắt buộc chứng minh:

```text
changing training_seed
DOES NOT change
train/validation/test membership
DOES NOT change
L_b/U_b membership
```

`training_seed` chỉ kiểm soát stochasticity của training, ví dụ:

```text
initialization
DataLoader order
sampler
worker RNG
stochastic augmentation
other framework RNG streams
```

Cả 10 training seeds phải được chạy đầy đủ trong official experiment. Không được
sau khi xem kết quả mới chọn seed "đẹp".

## 1A.4. SUP–SSL phải là paired experiments có thể audit

Tại một cell ví dụ:

```text
R50 – 10% – seed s
```

SUP và SSL phải có cùng:

```text
labeled subset
architecture
preprocessing
training seed
optimizer/update protocol
validation/test
labeled-supervision exposure
```

Khác biệt có chủ đích:

```text
SSL được phép khai thác thêm U_10%
```

Pairing phải được audit bằng **config/log/run manifest**, không chỉ mô tả trong
Methodology.

## 1A.5. Unlabeled firewall tuyệt đối

Ground truth ẩn của `U_b` không được tham gia vào:

```text
training
pseudo-label filtering
threshold tuning
EMA tuning
augmentation selection
checkpoint selection
Q_pseudo
model selection
```

Co-worker phải kiểm tra trực tiếp dataset loader, dataloader, hooks và evaluator
để chắc chắn annotation của `U_b` không bị đọc vô tình trong training/development
path.

## 1A.6. Toàn bộ SSL hyperparameters phải freeze trước final experiment

Giảng viên yêu cầu đặc biệt kiểm soát:

```text
confidence thresholds
pseudo-label filtering
regression reliability filtering
EMA momentum
weak/strong augmentation
unsupervised loss weight
learning rate
optimizer
scheduler
batch size
number of optimizer updates
warm-up / burn-in if any
AMP
gradient clipping
checkpoint frequency
evaluation frequency
```

Những tham số đã thuộc Main protocol không được thay đổi chỉ vì pilot/validation
performance thấp.

Ví dụ bị cấm:

```text
run thử
→ thấy threshold 0.7 tốt hơn
→ âm thầm đổi Main protocol
→ chạy final
```

Nếu có nhu cầu thay đổi scientific value, phải dừng và xử lý như
`CONTROLLED REVISION`.

## 1A.7. Weak/strong augmentation — geometry là hard implementation check

Giảng viên nhấn mạnh rằng mọi geometric transformation phải được áp dụng nhất
quán cho:

```text
image
ground-truth bbox
pseudo-box
```

Main hiện không bổ sung geometric transform giữa weak/strong ngoài resize theo
protocol. Nếu framework tự chèn hoặc yêu cầu thêm transform hình học, co-worker
không được tự quyết định; phải báo researcher trước.

Pilot phải có test trực quan hoặc assertion đủ mạnh để xác nhận image–bbox và
image–pseudo-box alignment.

## 1A.8. Checkpoint selection đã khóa

SUP:

```text
BEST = validation bbox mAP@[0.50:0.95]
```

SSL:

```text
BEST EMA Teacher = validation Teacher bbox mAP@[0.50:0.95]
LAST EMA Teacher = retained separately
```

Không được đổi BEST/LAST sau khi xem test result.

## 1A.9. Model selection khác pseudo-label quality analysis

Hai mục đích phải tách:

```text
official SSL detection model
= BEST EMA Teacher
```

```text
Q_pseudo
= LAST EMA Teacher on fixed validation
```

Không dùng BEST Teacher để tính `Q_pseudo` chỉ vì số đẹp hơn.

## 1A.10. 10 training seeds là thành phần của statistical design

10 seeds không chỉ để tạo error bar.

Chúng là replication unit của các phân tích RQ2/RQ3/RQ4/RQ6/RQ7 theo protocol.

Đặc biệt với RQ2:

```text
Δ = mAP_SSL - mAP_SUP
```

được tạo theo paired seed và tổng hợp trên 8 architecture×budget cells cho mỗi
seed trước khi thực hiện inferential test.

Official experiment không được rút xuống 1 seed.

## 1A.11. Pilot run KHÔNG phải final experiment

Pilot/debug/smoke chỉ dùng để kiểm tra:

```text
code path
image–annotation alignment
pseudo-box transform
EMA update
checkpoint save/select
seed propagation/reproducibility
data leakage
unlabeled firewall
```

Pilot:

```text
NOT part of official 10-seed results
NOT evidence for selecting a favorable seed
NOT hyperparameter shopping
```

Nếu pilot phát hiện **implementation bug**, được sửa bug để code tuân thủ
protocol. Nếu phát hiện cần thay **scientific value**, phải Controlled Revision.

## 1A.12. Frozen Experimental Configuration phải tồn tại trước official run

Trước khi official training được authorize, phải có một canonical frozen
configuration/table thể hiện tối thiểu:

```text
factor
levels
fixed value
source/rationale
change permission
```

và khóa:

```text
partition_seed = 42
exact 10 training seeds
budget = {1%,5%,10%,20%}
architecture = {R50-FPN,Swin-T-FPN}
paradigm = {SUP,SSL}
all Main hyperparameters
all pseudo-label protocol values
checkpoint/evaluation rules
```

Sau freeze, không được thay protocol dựa trên final test performance.

## 1A.13. Ba yêu cầu critical trước freeze — trạng thái hiện tại

| Critical item | Trạng thái | Handoff consequence |
|---|---|---|
| Budget là image-level hay bbox-level | `CLOSED / PASS` | Implement image-level 34/171/343/685; full bbox per selected image |
| Cơ chế 34 ảnh / 14 lớp | `CLOSED / PASS` | Implement label-aware iterative stratification + constrained repair; không gọi random sampling |
| Verify implementation: seed/DataLoader/augmentation/EMA/checkpoint/firewall | `OPEN / MUST PASS PREFLIGHT` | Không authorize official training trước khi evidence PASS |

## 1A.14. Quy tắc sau freeze

Từ thời điểm executable protocol/config được freeze:

```text
validation/test observation
MUST NOT silently trigger
model/config/threshold/checkpoint/statistical-protocol changes
```

Mọi thay đổi khoa học sau freeze phải được ghi nhận là protocol change /
Controlled Revision, có rationale và researcher approval.


---

# 2. DỮ LIỆU VÀ PHÂN HOẠCH CỐ ĐỊNH

## 2.1. Working scope

- Tổng số ảnh: **4.894**.
- 14 lớp bất thường detection.
- `No Finding` là ảnh âm tính / zero-GT, **không phải detection class thứ 15**.

## 2.2. Fixed train/validation/test

| Split | Số ảnh | No Finding |
|---|---:|---:|
| Train | 3.426 | 350 |
| Validation | 734 | 75 |
| Test | 734 | 75 |

Quy tắc:

- partition unit = `image_id`;
- train/validation/test đã khóa và không rebuild theo từng training seed;
- validation dùng cho development/checkpoint selection theo protocol;
- test chỉ dùng cho final evaluation sau khi pipeline/protocol đã freeze;
- không tune hyperparameter, threshold, checkpoint rule hoặc ablation bằng test.

---

# 3. LABELED-DATA BUDGET: IMAGE-LEVEL, KHÔNG PHẢI BBOX-LEVEL

`1%`, `5%`, `10%`, `20%` là **tỷ lệ số ảnh của fixed training set**, không phải tỷ lệ số bounding box.

Một ảnh thuộc `L_b` được xem là fully labeled image và giữ **toàn bộ ground-truth bounding boxes** của ảnh đó.

| Budget | Labeled `L_b` | Unlabeled `U_b` | No Finding trong `L_b` |
|---|---:|---:|---:|
| 1% | 34 | 3.392 | 3 |
| 5% | 171 | 3.255 | 17 |
| 10% | 343 | 3.083 | 35 |
| 20% | 685 | 2.741 | 70 |

Quan hệ bắt buộc:

```text
L_1% ⊂ L_5% ⊂ L_10% ⊂ L_20% ⊂ T
U_b = T \ L_b
L_b ∩ U_b = ∅
L_b ∪ U_b = T
```

---

# 4. CƠ CHẾ XÂY DỰNG LABELED SUBSET, ĐẶC BIỆT 1% = 34 ẢNH / 14 LỚP

Đây **không phải random sampling thuần túy**.

## 4.1. Biểu diễn dùng cho sampling

Mỗi ảnh được biểu diễn bằng:

- 14 chỉ báo binary class-presence ở cấp ảnh;
- 1 chỉ báo `zero-GT / No Finding`.

Số bounding box của một lớp trong ảnh không làm thay đổi chỉ báo class-presence: có ít nhất một bbox của lớp đó thì indicator = 1.

## 4.2. Initial candidate

Tại mỗi budget, tạo initial candidate bằng **iterative multilabel stratification** trên phần dữ liệu chưa thuộc nested prefix đã khóa.

`partition_seed = 42`:

- được truyền vào bộ chia multilabel stratified để kiểm soát nghiệm khởi đầu;
- được dùng trong deterministic tie-break ở repair khi cần;
- không seed search;
- không thay seed giữa budgets;
- không chọn seed vì cho distribution/performance đẹp.

## 4.3. Coverage constraint

Final labeled subset phải thỏa:

- exact image count;
- exact No Finding target;
- đủ **14/14 abnormality classes**;
- nested membership;
- distribution objective theo protocol.

Quy trình là **label-aware, coverage-constrained subset construction**.

## 4.4. Repair order bắt buộc

Thứ tự cố định:

1. `exact-size repair`;
2. `exact-No-Finding repair`;
3. `minimum-class-coverage repair`;
4. `objective repair` bằng one-for-one local search.

Coverage repair chỉ chấp nhận one-for-one swap làm giảm nghiêm ngặt số lớp còn thiếu và không phá:

- exact size;
- exact No Finding;
- nested prefix.

Nếu nhiều move hợp lệ, thứ tự ưu tiên là:

1. số lớp còn thiếu sau move;
2. integer distribution objective;
3. seeded deterministic tie-break;
4. canonical numeric move identity.

Không sử dụng greedy class-by-class image selection để chọn lần lượt 34 ảnh. Initial candidate được tạo đồng thời bằng iterative multilabel stratification; quyết định tuần tự chỉ xuất hiện trong repair.

`objective repair` dừng tại **one-for-one local optimum**, không được mô tả là global optimum.

---

# 5. PARTITION SEED VÀ TRAINING SEED

## 5.1. Partition seed

```text
partition_seed = 42
```

Chỉ dùng cho data partition / labeled-subset construction. Không được rebuild membership khi thay training seed.

## 5.2. Training seeds

Methodology khóa **một ordered list gồm 10 training seeds**, dùng giống nhau cho SUP và SSL. Danh sách operational hiện đã khóa trong seed protocol/implementation contract:

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

Quy tắc:

- same ordered seed list cho SUP và SSL;
- không seed search;
- không replacement seed;
- valid-but-low-performing run vẫn phải giữ;
- technical retry phải dùng **same exact training seed**;
- đổi training seed không được làm đổi train/val/test hoặc `L_b/U_b` membership.

Implementation phải truyền training seed tới các nguồn stochastic thực tế và ghi log tối thiểu cho:

- Python RNG;
- NumPy RNG;
- PyTorch CPU RNG;
- PyTorch CUDA RNG;
- DataLoader worker seeding;
- sampler;
- stochastic augmentation.

---

# 6. MA TRẬN THỰC NGHIỆM CHÍNH

## 6.1. Factors

```text
Method       = {SUP, SSL}
Budget       = {1%, 5%, 10%, 20%}
Architecture = {R50-FPN, Swin-T-FPN}
Seeds        = 10 locked seeds
```

Low-label matrix:

```text
2 methods × 4 budgets × 2 architectures = 16 cells
16 cells × 10 seeds = 160 official low-label runs
```

100%-SUP reference:

```text
2 architectures × 10 seeds = 20 runs
```

Main + reference trước ablation:

```text
180 runs
```

---

# 7. PAIRED SUP--SSL CONTRACT

Một paired cell được khóa bởi:

```text
architecture × budget × seed_index × training_seed
```

Trong mỗi cặp SUP--SSL phải giữ cùng:

- fixed train/validation/test;
- same `L_b`;
- same architecture;
- same preprocessing/input representation;
- same detector family;
- same training seed;
- same optimizer recipe trong cùng architecture;
- same total optimizer-update budget;
- same labeled-supervision exposure;
- same validation/test evaluation protocol.

Khác biệt có chủ đích:

```text
SUP: chỉ sử dụng L_b
SSL: sử dụng cùng L_b + U_b
```

Không được cho SSL nhiều supervised updates hơn rồi gán phần lợi ích đó cho unlabeled data.

---

# 8. ARCHITECTURE VÀ INPUT CONTRACT

Official architecture pair:

```text
Faster R-CNN + ResNet-50 + FPN
Faster R-CNN + Swin-T + FPN
```

Pretraining:

```text
R50 backbone   = ImageNet-1K pretrained
Swin-T backbone = ImageNet-1K pretrained
COCO detector-level initialization = PROHIBITED
```

Input:

```text
stored image = JPEG grayscale 1 channel
model input  = replicate to 3 identical channels
R = G = B
```

Resize/pad:

```text
scale = (1333, 800)
keep_ratio = True
pad_size_divisor = 32
```

No Finding / zero-GT images phải được giữ trong training; loader không được tự filter empty GT.

---

# 9. SUPERVISED BASELINE CONFIGURATION

## 9.1. Labeled augmentation

Main supervised stream:

```text
Load → Resize → Pack
```

Không mặc định dùng:

- flip;
- crop;
- rotation;
- mosaic;
- aggressive photometric augmentation.

## 9.2. Effective labeled batch

```text
B_L_eff = 4
```

Microbatch/gradient accumulation chỉ là execution detail để đạt đúng effective batch.

## 9.3. R50 optimizer

```text
optimizer = SGD
lr = 0.005
momentum = 0.9
weight_decay = 0.0001
```

## 9.4. Swin-T optimizer

```text
optimizer = AdamW
lr = 0.000025
betas = (0.9, 0.999)
weight_decay = 0.05
native Swin no-decay rules retained
```

## 9.5. Numerical rules

```text
AMP = ON
gradient clipping = OFF
all backbone stages = trainable
```

---

# 10. OPTIMIZER-UPDATE BUDGET, SCHEDULER VÀ VALIDATION

Official optimizer-update budgets:

| Budget | Updates |
|---|---:|
| 1% | 1.032 |
| 5% | 2.064 |
| 10% | 2.064 |
| 20% | 2.064 |
| 100%-SUP | 10.284 |

Đối với 2.064-update schedule:

```text
LR drop 1 ≈ update 1376
LR drop 2 ≈ update 1892
```

1% dùng proportional locations theo cùng scheduler structure.

Validation:

```text
validation interval = 172 optimizer updates
```

Update accounting phải tính **actual optimizer update**, không phải raw iteration/microbatch nếu có gradient accumulation.

---

# 11. SSL MAIN TREATMENT: SOFTTEACHER-STYLE

Scientific wording:

> teacher--student semi-supervised object detection dựa trên pseudo-labeling, triển khai theo SoftTeacher.

Không được claim là thuật toán SSOD mới.

## 11.1. Student / Teacher

```text
Student = optimized by gradient descent
Teacher = EMA model
Teacher architecture = Student architecture
Teacher(t0) = Student(t0)
no separate supervised burn-in
```

Resume:

- restore both Student and Teacher states;
- không resync Teacher từ Student khi resume.

## 11.2. EMA

```text
MeanTeacherHook momentum = 0.001
skip_buffers = True
Teacher update = once per actual Student optimizer update
```

Equivalent:

```text
Teacher_new = 0.999 * Teacher_old + 0.001 * Student_new
```

Nếu AMP làm optimizer step bị skip, EMA update của bước đó cũng phải skip.

---

# 12. SSL DATA STREAMS VÀ AUGMENTATION

Labeled stream:

```text
L_b + true GT → Student → supervised loss
```

Unlabeled stream:

```text
same U_b image
weak view   → Teacher → pseudo labels
strong view → Student → unsupervised loss
```

## 12.1. Weak view

```text
Resize
+ standard preprocessing
+ padding
```

Không:

- flip;
- crop;
- rotation;
- photometric augmentation.

## 12.2. Strong view

```text
Resize
+ grayscale brightness U(0.90, 1.10)
+ grayscale contrast   U(0.90, 1.10)
```

Constraint:

```text
R = G = B
```

Không geometric transform, random erasing, mosaic trong Main strong view.

Do Main không dùng geometric augmentation khác giữa weak/strong, geometry phải giữ nhất quán. Nếu implementation thêm bất kỳ transform hình học nào vì lý do framework, phải báo researcher trước; không tự thêm.

---

# 13. SSL LOSS VÀ L:U RATIO

Total loss:

```text
L_total = L_sup + 4 * L_unsup
sup_weight = 1
unsup_weight = 4
```

Không burn-in, không ramp-up:

```text
lambda_u(t) = 4 for all t
```

Sampling ratio:

```text
L:U = 1:1
B_L_eff = 4
B_U_eff = 4
```

---

# 14. PSEUDO-LABEL GENERATION VÀ FILTERING

Pseudo labels được sinh **online** từ EMA Teacher RCNN detections trên weak view.

Pseudo-label core:

```text
bbox
class
score
```

RPN proposals không được coi là final pseudo labels.

## 14.1. Thresholds

```text
detector score floor      = 0.05
initial pseudo score      = 0.50
RPN pseudo threshold      = 0.90
RCNN classification thr   = 0.90
```

Main values đã khóa; không tune hậu nghiệm.

## 14.2. NMS / max proposals

```text
RPN NMS IoU    = 0.70
RPN max        = 1000
RCNN NMS IoU   = 0.50
RCNN max       = 100
```

## 14.3. Regression reliability

```text
jitter_times = 10
jitter_scale = 0.06
reg_uncertainty_thr = 0.02
min_pseudo_bbox_wh = (0.01, 0.01)
```

Regression pseudo set:

```text
initial_score > 0.50
AND
reg_uncertainty < 0.02
```

Quan trọng:

```text
regression pseudo-label KHÔNG bắt buộc cls_score > 0.90
```

Không được gộp classification filtering và regression filtering thành một threshold duy nhất.

---

# 15. UNLABELED FIREWALL

Ground truth ẩn của `U_b` **không được phép** tham gia:

- training;
- supervised loss;
- pseudo-label filtering;
- confidence threshold tuning;
- regression-filter tuning;
- EMA tuning;
- augmentation selection;
- checkpoint selection;
- `Q_pseudo`;
- model selection.

Implementation phải bảo đảm unlabeled dataset object/dataloader không vô tình chuyển annotation của `U_b` vào model, hook hoặc evaluator.

Hidden U GT chỉ có thể tồn tại trong offline construction/audit artifacts cần thiết để tạo fixed membership; nó không được đi vào training/development path.

---

# 16. CHECKPOINT CONTRACT

## 16.1. SUP

```text
BEST = argmax validation bbox mAP@[0.50:0.95]
retain BEST + LAST
```

Official supervised evaluation model = BEST checkpoint.

## 16.2. SSL

Official SSL model = EMA Teacher.

```text
BEST EMA Teacher = selected by validation Teacher bbox mAP@[0.50:0.95]
LAST EMA Teacher = retained separately
```

Roles:

```text
BEST EMA Teacher → official detection evaluation
LAST EMA Teacher → Q_pseudo on fixed validation
```

Không được dùng BEST Teacher thay LAST Teacher cho `Q_pseudo` sau khi thấy kết quả.

---

# 17. Q_PSEUDO CONTRACT

Primary pseudo-label quality metric:

```text
Q_pseudo = PL-mAP@[0.50:0.95]
```

Scope:

```text
checkpoint = LAST EMA Teacher
dataset    = fixed validation
pseudo set = accepted RCNN classification pseudo labels
hidden U GT = prohibited
test Q_pseudo = prohibited
```

Các diagnostics phụ có thể gồm PL-AP50, PL-AP75, precision, recall, F1, matched IoU, pseudo count, retention, class-wise PL-AP, pseudo FP trên validation negatives và regression uncertainty, nhưng không thay thế `Q_pseudo`.

---

# 18. EVALUATION CONTRACT

Study-wide primary endpoint:

```text
bbox mAP@[0.50:0.95]
```

Secondary:

- AP50;
- AP75;
- class-wise AP cho 14 lớp;
- COCO AR@[0.50:0.95], maxDets=100;
- operating-point Recall;
- FP/image;
- FP/negative;
- negative-image FAR.

COCO AP:

```text
IoU = 0.50:0.05:0.95
useCats = 1
maxDets = 100
no extra fixed score threshold before AP
```

Operating-point metrics:

```text
tau_eval = 0.50
matching = category-aware, one-to-one
IoU match >= 0.50
after detector-native NMS/max100
```

No Finding evaluation:

- validation negatives = 75;
- test negatives = 75;
- mọi retained detection trên zero-GT image là FP.

Rare-class analyses dùng class-wise AP; không tạo `rare-mAP` mới.

---

# 19. VALIDATION / TEST FIREWALL

Validation được phép dùng cho:

- checkpoint selection;
- development metrics;
- `Q_pseudo`;
- prespecified ablation evaluation.

Test:

```text
FINAL ONLY
```

Không dùng test để:

- chọn checkpoint;
- đổi model;
- tune threshold;
- tune augmentation;
- tune EMA;
- tune filtering;
- chọn ablation winner;
- đổi metric/statistical model.

Non-main ablation variants không được đưa sang final test.

---

# 20. ABLATION SCOPE

Ablation là **secondary mechanistic component analysis**, không phải hyperparameter search và không được retroactively thay đổi Main SSL configuration.

Anchor:

```text
architecture = R50
budget = 10%
seeds = all 10 locked seeds
Main SSL = frozen Main treatment
```

OFAT:

```text
one target factor changes
all other scientific settings remain identical to Main
```

Prespecified families:

### Confidence threshold

```text
cls_pseudo_thr ∈ {0.5, 0.6, 0.7, 0.8, 0.9}
Main = 0.9
```

### Strong photometric augmentation

```text
A0 = none
A1 = brightness only
A2 = contrast only
A3 = brightness + contrast = Main
```

Magnitude giữ `[0.90, 1.10]`, grayscale `R=G=B`.

### Regression pseudo-label filtering

```text
F0 = initial score > 0.50; no extra regression reliability filter
F1 = confidence-based filtering score > 0.90
F2 = initial score > 0.50 AND reg_uncertainty < 0.02 = Main
```

Không tự bổ sung EMA ablation hoặc burn-in/ramp-up ablation.

---

# 21. STATISTICAL CONTRACT — KHÔNG THAY ĐỔI KHI IMPLEMENT

Scientific source: `sn-article.tex`, mục **Statistical Analysis**.

## 21.1. Đơn vị suy luận và tổng hợp seed

- Inferential replication unit = **training seed / trained run**.
- Image, bbox và detection **không** được xem là các replication độc lập.
- Mỗi official condition có `n = 10`.
- Mức ý nghĩa chung: `alpha = 0.05`.
- Báo cáo theo thứ tự: **effect estimate → two-sided 95% CI → p-value**.
- Mean theo 10 seed.
- Sample SD dùng `ddof = 1`.
- Không cross-seed prediction pooling.
- Không loại valid poor-performing seed.
- Không replacement seed.
- Không metric imputation.

Primitive paired effect:

```text
ΔM_{a,b,s} = M_SSL_{a,b,s} - M_SUP_{a,b,s}
```

Ghép cặp theo cùng:

```text
architecture × budget × seed_index × training_seed
```

## 21.2. RQ2 — Primary overall SSL gain

Với mỗi seed:

```text
G_s = (1/8) * Σ_{a,b} ΔmAP_{a,b,s}
```

trong đó có 2 architectures × 4 budgets và mỗi cell có trọng số bằng nhau `1/8`.

Primary inference:

```text
one-sample t-test
alternative = greater
n = 10
df = 9
p-value = one-sided
CI = two-sided 95%
```

Cell-specific effects chỉ báo cáo mô tả bằng mean, sample SD và 95% CI; không tạo thêm primary hypothesis tests cho từng cell.

## 21.3. RQ3 — Budget-dependent SSL gain

Trong mỗi budget và seed:

```text
B_{b,s} =
(ΔmAP_R50,b,s + ΔmAP_Swin-T,b,s) / 2
```

Budget là repeated categorical factor với 4 levels:

```text
1%, 5%, 10%, 20%
```

Primary analysis:

```text
one-way repeated-measures ANOVA
Greenhouse–Geisser correction = ALWAYS
Mauchly/sphericity checks = diagnostic only, không dùng để đổi primary method
```

Báo cáo tối thiểu:

```text
F
epsilon_GG
GG-corrected df
p_GG
```

Sáu prespecified paired contrasts:

```text
5%-1%
10%-1%
20%-1%
10%-5%
20%-5%
20%-10%
```

Các contrast này thuộc một multiplicity family riêng và dùng Holm.

## 21.4. RQ7 — Architecture-dependent SSL effect

Trong mỗi architecture và seed:

```text
A_{a,s} = (1/4) * Σ_b ΔmAP_{a,b,s}
D_s = A_Swin-T,s - A_R50,s
```

Inference:

```text
two-sided one-sample t-test
n = 10
df = 9
```

Không diễn giải đây là kiểm định ưu thế tổng quát của Swin-T so với R50.

## 21.5. RQ4 — Q_pseudo association với downstream SSL gain

Có tối đa:

```text
2 architectures × 4 budgets × 10 seeds = 80 rows
```

Outcome:

```text
Y_{a,b,s}
= ΔmAP_test
= mAP_SSL,test - mAP_SUP,test
```

Predictor:

```text
Q_{a,b,s} = PL-mAP@[0.50:0.95]
```

Q_pseudo được tính trên **fixed validation** bằng **LAST EMA Teacher**.

Within-cell centering:

```text
Q_WC_{a,b,s} = Q_{a,b,s} - mean_s(Q_{a,b,s})
X_{a,b,s} = Q_WC_{a,b,s} / 0.01
```

Primary mixed model:

```text
Y ~ X + architecture_budget_cell + (1 | training_seed)
```

Estimation/inference:

```text
REML
Kenward–Roger inference for the fixed effect
beta_Q test = one-sided, positive direction
```

`beta_Q` được diễn giải là thay đổi trung bình của downstream `ΔmAP_test`
(tính theo AP points) khi Q_pseudo tăng 1 pseudo AP point **trong cùng
architecture–budget cell**. Không causal wording.

## 21.6. RQ5 — Rare classes

Frozen rare classes:

```text
Atelectasis
Pneumothorax
```

Với mỗi class:

```text
ΔAP_{c,a,b,s} = AP_SSL - AP_SUP
G_{c,s} = (1/8) * Σ_{a,b} ΔAP_{c,a,b,s}
```

Báo cáo:

```text
mean paired gain
sample SD
two-sided 95% CI
NO required formal p-value
```

Không tạo `rare-mAP`.

Nếu AP không xác định do không có GT support trong phạm vi đánh giá, ghi
`undefined/NA` theo evaluator contract; **không gán bằng 0**.

## 21.7. RQ6 — False positives trên No Finding

Primitive:

```text
ΔF_{a,b,s} = FP_SSL,neg - FP_SUP,neg
H_s = (1/8) * Σ_{a,b} ΔF_{a,b,s}
```

Primary inference:

```text
two-sided one-sample t-test
n = 10
df = 9
```

Diễn giải:

```text
H_s < 0  => SSL ít FP/negative hơn
H_s > 0  => SSL nhiều FP/negative hơn
```

`FAR_neg` là estimation-focused secondary outcome, báo cáo effect + 95% CI,
không có official hypothesis-test p-value riêng.

## 21.8. Multiplicity

Family F1:

```text
RQ2 only
m = 1
no adjustment
```

Family F2:

```text
RQ3 omnibus p_GG
RQ4 one-sided Kenward–Roger p
RQ6 two-sided p
RQ7 two-sided p

m = 4
Holm step-down
FWER = 0.05
```

Family F3:

```text
6 prespecified RQ3 pairwise budget contrasts
Holm step-down
m = 6
```

Một RQ3 pairwise localization claim chỉ được xem là formal khi:

```text
RQ3 omnibus survives F2
AND
pair survives F3
```

Không shrink family sau khi thấy missing/unfavorable p-value.

## 21.9. Reporting rules

Official effect size = unstandardized absolute effect.

```text
mAP/AP effects -> AP points, 2 decimals
RQ4 slope       -> AP-point outcome change per +1 pseudo AP point, 3 decimals
FP              -> 3 decimals
FAR difference  -> percentage points, 2 decimals
p               -> 3 decimals; nếu < .001 thì ghi p < .001
```

Mọi official CI = two-sided 95% CI cho từng effect; không phải Holm-adjusted simultaneous CI.

Không dùng các cụm như `marginally significant`.

## 21.10. Robustness / sensitivity

Primary methods ở trên **không được đổi** sau khi xem diagnostics.

RQ2/RQ6/RQ7:

```text
exact sign-flip robustness
2^10 = 1024 sign configurations
RQ2 = one-sided
RQ6 = two-sided
RQ7 = two-sided
```

Đồng thời thực hiện:

```text
LOSO = leave-one-seed-out
exactly 10 analyses
```

LOSO không dùng để loại seed khỏi primary analysis.

RQ3:

```text
primary = GG repeated-measures ANOVA
sensitivity = Friedman test
```

RQ4 diagnostics tối thiểu:

```text
convergence
singularity
residual-vs-fitted
Q-Q plot
seed influence
```

RQ4 sensitivity:

```text
linear model: Y ~ Q_WC + architecture_budget_cell
CR2 cluster-robust covariance
cluster = training_seed
Satterthwaite df
```

CR2 là sensitivity analysis và không thay thế primary Kenward–Roger mixed model.

## 21.11. Test firewall cho statistical pipeline

Trình tự:

```text
COMPLETE PROTOCOL
→ FREEZE CONFIGURATION
→ FINAL TEST EVALUATION
→ LOCKED STATISTICAL ANALYSIS
→ REPORT
```

Sau khi test được mở:

```text
NO model change
NO checkpoint-rule change
NO threshold change
NO metric change
NO hypothesis change
NO hypothesis-direction change
NO statistical-model change
NO multiplicity change
NO rare-class redefinition
NO new confirmatory subgroup
```

Statistical implementation phải được kiểm tra trước bằng synthetic/golden fixtures,
không cần dùng real final-test performance để test code.

---

# 22. PILOT / IMPLEMENTATION PREFLIGHT TRƯỚC OFFICIAL TRAINING

Pilot/debug run **không phải official experiment** và không tính vào 10 training seeds chính thức.

Mục đích duy nhất là verify implementation compliance, không dùng để chọn seed/config có performance tốt.

## 22.0. Hard gate — PILOT PASS phải đủ điều kiện để chuyển nguyên trạng sang OFFICIAL

Phần preflight này là **hard implementation gate**, không phải checklist tối thiểu và
không phải smoke test mang tính tham khảo.

**TẤT CẢ** các kiểm tra được liệt kê trong Mục 22 phải có trạng thái rõ ràng và phải
`PASS` trước khi official training được authorize.

```text
PILOT / PREFLIGHT PASS
=
ALL REQUIRED CHECKS PASS
AND
NO UNRESOLVED WARNING THAT CAN CHANGE SCIENTIFIC BEHAVIOR
AND
NO UNVERIFIED IMPLEMENTATION PATH USED BY OFFICIAL RUNS
```

Không được có trạng thái:

```text
PARTIAL PASS
PASS WITH UNCHECKED OFFICIAL PATH
ASSUMED PASS
NOT TESTED BUT EXPECTED TO WORK
```

Nếu một thành phần sẽ được sử dụng trong official training thì thành phần đó phải
được đi qua preflight tương ứng **trước official run**.

Nguyên tắc chuyển tiếp bắt buộc:

```text
PILOT-PASS EXECUTABLE CONFIG
→ FREEZE
→ OFFICIAL RUNS USE THE SAME IMPLEMENTATION PATH
```

Official training không được tự ý thay đổi:

```text
dataset loader
sampler / DataLoader seed path
augmentation path
optimizer/update accounting
AMP behavior
Teacher/Student initialization
EMA update timing
pseudo-label generation/filtering
checkpoint selection
evaluator
firewall logic
run-state / resume logic
```

Nếu official configuration cần dùng một code path, branch, hook, transform, evaluator,
resume mode, batch composition hoặc architecture-specific behavior **chưa được pilot
kiểm tra**, thì:

```text
PILOT STATUS = NOT COMPLETE
OFFICIAL TRAINING = NOT AUTHORIZED
```

Pilot chỉ được xem là `CLOSED / PASS` khi các kiểm tra dưới đây đã bao phủ toàn bộ
implementation behavior mà official SUP/SSL runs sẽ sử dụng.

## 22.0.1. Coverage của preflight phải bao phủ official execution paths

Preflight không chỉ chạy một cấu hình đại diện rồi suy diễn cho mọi cấu hình còn lại.

Phải verify các execution path có thể khác nhau về implementation, tối thiểu theo:

```text
SUP R50
SUP Swin-T
SSL R50
SSL Swin-T
```

và phải xác nhận rằng các biến theo budget không làm phát sinh code path chưa kiểm tra:

```text
1%
5%
10%
20%
100%-SUP reference (nếu dùng path khác)
```

Không bắt buộc pilot phải train đủ thời lượng official ở mọi budget, nhưng mọi
**logic branch / configuration path / loader path / scheduler path / checkpoint path**
sẽ xuất hiện trong official runs phải được kiểm tra bằng test/pilot phù hợp.

Đặc biệt:

- schedule 1% phải được kiểm tra riêng vì update budget khác;
- schedule 5%/10%/20% có thể dùng cùng structural path nếu config equivalence được
  chứng minh bằng automated config diff/assertion;
- R50 và Swin-T phải đều được smoke/pilot vì optimizer/backbone implementation khác;
- SUP và SSL phải đều được kiểm tra vì data flow và model state khác;
- SSL phải kiểm tra cả normal pseudo-label batch và empty pseudo-label batch;
- resume path phải được kiểm tra nếu official training cho phép resume sau gián đoạn;
- AMP skipped-step path phải được kiểm tra nếu AMP được bật trong official runs.

## 22.1. Data/seed

- fixed split hashes/membership đúng;
- exact `L_b/U_b` membership đúng;
- đổi training seed không đổi membership;
- No Finding zero-GT được giữ;
- category mapping đúng 14 classes.

## 22.2. DataLoader / stochasticity

- training seed được propagate đúng;
- worker/sampler seeding có log;
- resume không đổi seed contract;
- reproducibility smoke test ở mức protocol.

## 22.3. SUP infrastructure

- grayscale → 3 identical channels;
- resize/pad đúng;
- optimizer recipe đúng theo architecture;
- actual optimizer-update counter đúng;
- validation interval đúng;
- BEST/LAST selection đúng;
- AMP hoạt động đúng.

## 22.4. SSL

- Teacher = Student tại t0;
- EMA update đúng timing;
- AMP skipped step → EMA also skipped;
- resume restores Teacher + Student;
- weak/strong view đúng;
- image–bbox alignment đúng sau mọi transform;
- image–pseudo-box alignment đúng giữa Teacher weak view và Student strong view;
- pseudo thresholds đúng;
- empty pseudo-label batch không crash/không tạo supervision giả;
- zero-GT handling đúng;
- L:U = 1:1;
- unsup weight = 4;
- hidden-U firewall PASS.

## 22.5. Evaluator

- COCO AP settings đúng;
- zero-GT metrics đúng;
- tau_eval=0.50 only for operating-point metrics;
- Q_pseudo dùng LAST EMA Teacher + validation;
- test performance không bị đọc trong preflight.

---

# 22.6. REPRODUCIBILITY IMPLEMENTATION CONTRACT

Methodology đã định nghĩa trước các nguyên tắc reproducibility; phần còn để trống
chỉ là **runtime evidence** cần điền sau implementation. Co-worker phải bảo toàn
và xuất bằng chứng để researcher hoàn thiện phần này, không được tự viết giá trị
môi trường/deterministic settings khi chưa đo thực tế.

## 22.6.1. Seed propagation phải được ghi nhận thực tế

Tối thiểu phải xác nhận và log cách `training_seed` được truyền tới:

```text
Python random
NumPy RNG
PyTorch CPU RNG
PyTorch CUDA RNG
DataLoader workers
sampler
stochastic augmentation
```

## 22.6.2. Data-membership evidence

Phải tạo bằng chứng thực tế cho:

```text
train/validation/test membership
labeled subset membership
unlabeled subset membership
architecture × budget × seed mapping
```

Khuyến nghị dùng manifest + checksum/hash để audit tự động.

## 22.6.3. Per-run configuration evidence

Mỗi run phải lưu tối thiểu:

```text
architecture
method = SUP/SSL
budget
seed_index
training_seed
optimizer
learning rate
weight decay
total optimizer updates
validation interval
batch composition
AMP state
config path/hash
run_id / attempt_id
```

Đối với SSL phải lưu thêm các giá trị thực tế của:

```text
Teacher/Student configuration
EMA momentum / skip_buffers / update timing
pseudo-label thresholds
weak augmentation
strong augmentation
regression uncertainty filtering
L:U ratio
unsupervised loss weight
```

## 22.6.4. Environment capture

Không giả định trước. Từ runtime phải ghi:

```text
OS
Python
PyTorch
MMDetection
MMCV
MMEngine
CUDA
cuDNN
GPU model
GPU memory
NVIDIA driver
```

## 22.6.5. Deterministic/numerical settings

Phải xác nhận từ implementation/runtime:

```text
PyTorch deterministic algorithms
cuDNN deterministic
cuDNN benchmark
worker seed initialization
AMP skipped-step behavior
EMA behavior khi optimizer step bị skip
```

Không được claim bitwise reproducibility nếu evidence không chứng minh điều đó.
Methodology phân biệt **protocol reproducibility** với **bitwise numerical reproducibility**.

## 22.6.6. Checkpoint/evaluation provenance

Phải lưu bằng chứng cho:

```text
SUP BEST checkpoint
SUP LAST checkpoint
SSL BEST EMA Teacher checkpoint
SSL LAST EMA Teacher checkpoint
validation metric used for selection
Q_pseudo evaluator using LAST EMA Teacher
official evaluator/config
```

## 22.6.7. Retry / deviation record

Nếu có lỗi kỹ thuật hoặc controlled revision, phải lưu:

```text
run_id
attempt_id
original training_seed
failure reason
retry reason
same-seed confirmation
whether scientific protocol changed
researcher approval if controlled revision
```

Valid-but-poor run không được retry chỉ vì performance thấp.

---

# 22.7. CLOSURE CRITERIA — Điều kiện duy nhất để PILOT / PREFLIGHT = PASS

Chỉ được ghi:

```text
PILOT_IMPLEMENTATION_PREFLIGHT = CLOSED / PASS
```

khi **đồng thời** thỏa tất cả điều kiện sau:

```text
1. Data / membership / seed checks = PASS
2. DataLoader / sampler / worker stochasticity checks = PASS
3. SUP R50 execution path = PASS
4. SUP Swin-T execution path = PASS
5. SSL R50 execution path = PASS
6. SSL Swin-T execution path = PASS
7. 1% scheduler/update path = PASS
8. standard 2064-update scheduler path = PASS
9. image–GT bbox geometry alignment = PASS
10. image–pseudo-box geometry alignment = PASS
11. Teacher initialization = PASS
12. EMA timing = PASS
13. AMP skipped-step → EMA skip behavior = PASS
14. empty pseudo-label behavior = PASS
15. zero-GT handling = PASS
16. hidden-U GT firewall = PASS
17. checkpoint BEST/LAST logic = PASS
18. Q_pseudo LAST-Teacher/validation path = PASS
19. evaluator / COCO metric configuration = PASS
20. operating-point metric path = PASS
21. resume path = PASS if resume is allowed in official runs
22. run manifest / seed logging / provenance = PASS
23. pilot-vs-official run isolation = PASS
24. frozen executable configs produced and hashable = PASS
25. no unresolved protocol-relevant warning = TRUE
```

Bất kỳ mục nào `FAIL`, `NOT TESTED`, `UNKNOWN`, `MANUAL ASSUMPTION` hoặc chưa có
evidence thì toàn bộ preflight **chưa PASS**.

## 22.7.1. Pilot-to-official equivalence assertion

Ngay trước khi launch từng official run, launcher phải thực hiện automated
pre-run assertions để xác nhận official config vẫn thuộc frozen implementation đã PASS.

Tối thiểu assert:

```text
scientific source / protocol version
architecture
method
budget
seed_index
training_seed
dataset/split manifest hash
L_b/U_b manifest hash
optimizer
learning rate
scheduler/update budget
effective batch
AMP
augmentation
EMA
pseudo thresholds
loss weights
checkpoint rule
evaluation config
```

Nếu khác frozen config ngoài các trường được phép thay đổi theo experimental matrix:

```text
RUN = BLOCKED
```

Không được "chạy trước rồi audit sau".

## 22.7.2. Official runtime guardrails phải tiếp tục kiểm tra

Pilot PASS không có nghĩa bỏ guardrail trong official runs. Official launcher/runtime
phải tiếp tục fail-fast nếu phát hiện:

```text
membership mismatch
seed mismatch
hidden-U annotation present
wrong class mapping
wrong effective batch/update count
unexpected augmentation
wrong EMA/update timing
checkpoint-rule mismatch
evaluator mismatch
```

Do đó:

```text
PILOT PASS
= evidence implementation đúng trước training chính thức

OFFICIAL GUARDRAILS
= bảo đảm implementation đó không bị lệch trong từng run chính thức
```

Hai lớp kiểm soát này đều bắt buộc.


---

# 23. BẰNG CHỨNG CO-WORKER PHẢI TRẢ LẠI SAU IMPLEMENTATION

Trước khi researcher cho phép official training, co-worker cần bàn giao tối thiểu:

1. canonical config(s) thực tế cho R50 SUP, Swin SUP, R50 SSL, Swin SSL;
2. script/command entry point;
3. dataset/DataLoader configuration;
4. exact seed propagation implementation;
5. augmentation pipeline;
6. EMA hook/config và timing evidence;
7. checkpoint selection code/config;
8. evaluator config;
9. unlabeled firewall test;
10. pilot/preflight report PASS/FAIL;
11. config diff chứng minh SUP--SSL pairing trong cùng architecture;
12. run manifest mẫu chứa architecture, budget, seed_index, training_seed, method;
13. environment capture;
14. Git commit/hash của implementation được audit;
15. failure/retry logging rule.
16. `FROZEN_EXPERIMENTAL_CONFIGURATION` canonical table/config gồm factor, levels, fixed value, source/rationale và change permission.
17. audit log chứng minh đổi `training_seed` không làm thay đổi train/val/test hoặc `L_b/U_b` membership.
18. geometry-alignment evidence cho image–GT bbox và image–pseudo-box.
19. proof/log rằng pilot/debug run IDs không bị trộn vào official 10-seed matrix.

Các evidence này sẽ dùng để điền các trường runtime/evidence còn để placeholder trong mục Reproducibility của Methodology hiện hành; các nguyên tắc reproducibility khoa học đã được xác định trước và không được co-worker tự thay đổi.

---

# 24. CÁC HÀNH VI BỊ CẤM KHI IMPLEMENT

```text
PROHIBITED
- đổi split hoặc budget membership
- đổi image-level budget thành bbox-level budget
- seed search
- replacement seed
- rebuild L/U theo training seed
- dùng hidden U GT trong training/development
- đổi architecture pair
- dùng COCO detector-level pretraining
- đổi optimizer/scheduler/update budget vì pilot performance thấp
- thêm augmentation ngoài Main mà không controlled revision
- đổi pseudo thresholds của Main dựa trên validation/test
- đổi EMA momentum/timing
- dùng Student thay EMA Teacher cho official SSL result
- dùng BEST Teacher thay LAST Teacher cho Q_pseudo
- test non-main ablation variants
- giảm 10 seeds xuống 1 seed cho final experiment
- rerun valid seed chỉ vì result xấu
- cherry-pick checkpoint ngoài rule BEST/LAST đã khóa
```

---

# 25. EXECUTION ORDER ĐỀ XUẤT CHO CO-WORKER

```text
1. Verify frozen inputs/manifests
2. Implement shared SUP data/training infrastructure
3. Smoke-test R50 SUP
4. Smoke-test Swin-T SUP
5. Implement SSL Teacher/Student + pseudo-label path
6. Run SSL implementation preflight
7. Implement/verify evaluator
8. Implement/verify statistical pipeline fixtures
9. Freeze executable configs + environment + Git commit
10. Researcher reviews preflight evidence
11. Only after authorization: official SUP runs
12. Only after SSL preflight PASS: official SSL runs
13. Ablation only under locked OFAT scope
14. Final test only after full protocol authorization
```

Đây là execution recommendation; nó không thay đổi scientific design.

---

# 26. GATE ĐỂ ĐƯỢC PHÉP CHẠY OFFICIAL EXPERIMENT

Official training chỉ được xem là authorized khi:

```text
PILOT_IMPLEMENTATION_PREFLIGHT = CLOSED / PASS
ALL SECTION 22 REQUIRED CHECKS = PASS
NO UNTESTED OFFICIAL EXECUTION PATH = TRUE
DATA INPUTS = VERIFIED
SUP R50 EXECUTION PATH = PASS
SUP SWIN-T EXECUTION PATH = PASS
SSL R50 EXECUTION PATH = PASS
SSL SWIN-T EXECUTION PATH = PASS
SEED / DATALOADER AUDIT = PASS
EMA / AMP TIMING = PASS
CHECKPOINT RULE = PASS
UNLABELED FIREWALL = PASS
EVALUATOR PREFLIGHT = PASS
FROZEN EXECUTABLE CONFIGS = CREATED
FROZEN EXPERIMENTAL CONFIGURATION TABLE = VERIFIED
PILOT RUNS SEPARATED FROM OFFICIAL RUNS = PASS
GEOMETRY / BBOX / PSEUDO-BOX ALIGNMENT = PASS
PILOT-TO-OFFICIAL CONFIG EQUIVALENCE CHECK = PASS
OFFICIAL FAIL-FAST RUNTIME GUARDRAILS = ENABLED
GIT/ENVIRONMENT EVIDENCE = CAPTURED
RESEARCHER AUTHORIZATION = YES
```

Trước gate này, mọi run chỉ là pilot/debug/smoke run và không được trộn vào official 10-seed results.

---

# 27. SOURCE-OF-TRUTH RULE CUỐI CÙNG

**Scientific source of truth:**

```text
sn-article.tex
```

Tài liệu bàn giao này chỉ chuyển Methodology sang dạng implementation contract dễ thi hành.

Nếu phát hiện khác biệt giữa tài liệu này và `.tex`:

```text
.tex APPROVED METHODOLOGY WINS
```

Nếu implementation framework không hỗ trợ trực tiếp một yêu cầu đã khóa:

```text
STOP
→ document the incompatibility
→ report to researcher
→ do not silently change protocol
```

---

# 27A. FINAL CROSS-CHECK — APPROVED METHODOLOGY VS HANDOFF

Nguồn ưu tiên tuyệt đối: `sn-article.tex`.

| Nhóm | Trạng thái sau cross-check | Ghi chú |
|---|---|---|
| Fixed train/validation/test | MATCH | 3426/734/734; No Finding 350/75/75 |
| Labeled-data budget | MATCH | image-level; 34/171/343/685; full bbox annotations per selected image |
| Nested L/U construction | MATCH | nested `L_b`, `U_b = T \ L_b` |
| 1% / 14-class mechanism | MATCH | label-aware iterative multilabel stratification + constrained repair |
| partition_seed | MATCH | 42; fixed; no seed search |
| 10 training seeds | MATCH + operational values included | same ordered list SUP/SSL |
| SUP–SSL pairing | MATCH | same arch/budget/seed/L_b/update exposure |
| R50/Swin architectures | MATCH | Faster R-CNN + FPN; ImageNet-1K backbone pretraining |
| SUP optimizer/training config | MATCH | architecture-specific recipe preserved |
| Optimizer-update budgets | MATCH | 1032/2064/2064/2064; 100%-SUP 10284 |
| Validation/checkpoint | MATCH | interval 172; BEST by val bbox mAP@[.50:.95]; retain BEST/LAST |
| SSL Teacher/Student | MATCH | same architecture; Teacher initialized from Student; no burn-in |
| EMA | MATCH | momentum=.001; skip_buffers=True; actual optimizer-step timing |
| Weak/strong augmentation | MATCH | weak conservative; strong brightness+contrast only; no geometry in Main |
| SSL loss and L:U | MATCH | `L_sup + 4 L_unsup`; 1:1; effective 4+4 |
| Pseudo-label thresholds/NMS | MATCH | det/init/RPN/cls + NMS/max fixed values |
| Regression reliability | MATCH | jitter 10, scale .06, uncertainty .02, min bbox .01/.01 |
| Hidden-U GT firewall | MATCH | prohibited from training/development/Q_pseudo |
| BEST/LAST Teacher roles | MATCH | BEST detection; LAST Q_pseudo |
| Evaluation metrics | MATCH | COCO bbox mAP primary; operating point tau=.50 |
| Validation/test firewall | MATCH | test final-only |
| Ablation OFAT | MATCH | R50, 10%, all 10 seeds; prespecified confidence/augmentation/filtering |
| Statistical analysis | MATCH | full RQ2/RQ3/RQ4/RQ5/RQ6/RQ7, Holm, robustness/sensitivity added |
| Reproducibility | MATCH | runtime evidence, environment, deterministic settings, provenance added |
| Source-of-truth filename | MATCH | `sn-article.tex` |
| Supervisor notes / governance | MATCH — ADDED | 12 supervisor items + 3 critical pre-freeze requirements explicitly mapped to implementation |
| Pilot/preflight closure | HARD GATE | all required checks must PASS; no untested official execution path; pilot-to-official equivalence + runtime guardrails required |
| Implementation status | CORRECT | not executed; official training requires full preflight closure + researcher authorization |

**Final handoff state:** `READY FOR CO-WORKER / PRE-IMPLEMENTATION`; supervisor requirements are incorporated, while implementation verification remains `OPEN / MUST PASS PREFLIGHT`.

---

# 28. TRẠNG THÁI BÀN GIAO

```text
METHODOLOGY = COMPLETED / SUPERVISOR-APPROVED
IMPLEMENTATION = NOT YET EXECUTED
FINAL IMPLEMENTATION HANDOFF = READY FOR CO-WORKER
OFFICIAL TRAINING = NOT YET AUTHORIZED UNTIL PREFLIGHT PASS
FINAL TEST = CLOSED UNTIL AUTHORIZATION
```
