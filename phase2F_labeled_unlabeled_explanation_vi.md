# Tài liệu giải thích Phase 2F — Xây dựng tập labeled/unlabeled

## 1. Mục đích của tài liệu

Tài liệu này giải thích bằng tiếng Việt cách Phase 2F hoạt động, ý nghĩa của
các hàm chính, các tệp đầu ra và cách đọc kết quả chạy. Đây là tài liệu hướng
dẫn đi kèm, **không phải protocol thực thi** và không thay thế ba tệp nguồn:

- `scripts/02F_build_labeled_unlabeled.py`;
- `configs/protocol/phase2F_labeled_unlabeled.yaml`;
- `tests/test_phase2F_labeled_unlabeled_guardrails.py`.

Tài liệu không làm thay đổi membership, seed, checksum hoặc bất kỳ official
artifact nào đã được khóa. Phiên bản được giải thích là `2F-C0-R11`, protocol
version `2.0.0`.

## 2. Phase 2F giải quyết việc gì?

Phase 2E đã cố định ba tập:

| Tập | Số ảnh |
|---|---:|
| Train | 3.426 |
| Validation | 734 |
| Test | 734 |

Phase 2F **chỉ thao tác trong tập Train gồm 3.426 ảnh**. Validation và Test
không được lấy mẫu lại, không được đưa vào tập labeled/unlabeled và không được
dùng để lựa chọn membership.

Từ Train, chương trình tạo bốn cặp tập:

| Budget | Labeled | No Finding trong labeled | Unlabeled |
|---|---:|---:|---:|
| 1% | 34 | 3 | 3.392 |
| 5% | 171 | 17 | 3.255 |
| 10% | 343 | 35 | 3.083 |
| 20% | 685 | 70 | 2.741 |

Với mỗi budget `b`:

```text
Unlabeled_b = Train − Labeled_b
```

Các tập labeled có tính lồng nhau:

```text
Labeled_1% ⊂ Labeled_5% ⊂ Labeled_10% ⊂ Labeled_20%
```

Điều này giúp so sánh các budget trên cùng một chuỗi dữ liệu: budget lớn giữ
toàn bộ ảnh của budget nhỏ rồi bổ sung thêm ảnh mới.

## 3. Luồng xử lý tổng quát

```text
Đọc YAML và Phase 2E artifacts
             ↓
Chạy preflight, xác minh input chưa thay đổi
             ↓
Đọc instances_train.json và tạo ma trận chỉ báo
             ↓
Tính target cho 1%, 5%, 10%, 20%
             ↓
Xây dựng tuần tự 1% → 5% → 10% → 20%
             ↓
Với mỗi budget:
  iterative multilabel stratification
  → sửa đúng kích thước
  → sửa đúng số No Finding
  → bảo đảm hiện diện đủ 14 lớp
  → cải thiện phân phối bằng one-for-one swaps
             ↓
Kiểm định độc lập trong staging
             ↓
Nếu tất cả PASS: promote đồng thời official artifacts
Nếu có lỗi: rollback, không promote kết quả dở dang
```

## 4. Các khái niệm cần hiểu trước

### 4.1 `image_id` và vị trí mảng

COCO nhận diện ảnh bằng `image_id`. Bên trong chương trình, mỗi ảnh còn được
gán một vị trí nguyên trong mảng NumPy để tính toán nhanh. Membership cuối cùng
được chuyển lại thành tập `image_id` trước khi ghi artifact.

### 4.2 Ma trận `labels14`

Mỗi hàng ứng với một ảnh, mỗi cột ứng với một trong 14 lớp bất thường:

- `1`: ảnh có ít nhất một bounding box của lớp đó;
- `0`: ảnh không có bounding box của lớp đó.

Một ảnh có thể có nhiều giá trị `1` vì đây là bài toán multilabel.

### 4.3 `zero_gt`

`zero_gt=1` chỉ ảnh No Finding, tức ảnh không có bounding box. No Finding là
một ràng buộc về thành phần dữ liệu, **không phải lớp detection thứ 15**.

### 4.4 Nested prefix

Khi xây dựng budget lớn hơn, toàn bộ membership của budget nhỏ hơn trở thành
`locked` prefix. Các bước repair chỉ được thay đổi phần mới bổ sung; chúng
không được loại ảnh đã khóa ở budget trước.

### 4.5 One-for-one swap

Một one-for-one swap loại đúng một ảnh khỏi candidate và thêm đúng một ảnh từ
pool. Vì số ảnh thêm và loại bằng nhau, kích thước tập không đổi. Những swap
phải giữ số No Finding cũng chỉ đổi hai ảnh có cùng trạng thái `zero_gt`.

### 4.6 Vì sao Phase 2F chọn deterministic one-for-one swap?

#### 4.6.1 One-for-one không phải toàn bộ phương pháp lấy mẫu

Cần phân biệt rõ: Phase 2F **không dùng one-for-one swap để tạo candidate ban
đầu**. Candidate ban đầu được tạo bằng iterative multilabel stratification.
One-for-one chỉ là neighborhood repair được dùng sau đó để đạt hoặc bảo toàn
các ràng buộc đã xác định trước.

Do đó, mô tả đầy đủ của phương pháp là:

```text
iterative multilabel stratification
→ deterministic constrained repair
→ exhaustive one-for-one objective local search
```

Không nên gọi toàn bộ Phase 2F là “thuật toán one-for-one swap”, vì cách gọi
đó bỏ qua bước stratification và các repair phase có mục tiêu khác nhau.

#### 4.6.2 Các phương án có thể thay thế

| Phương án | Nguyên lý | Điểm mạnh | Hạn chế đối với Phase 2F |
|---|---|---|---|
| Add/remove đơn lẻ | Thêm hoặc loại từng ảnh | Đơn giản, nhanh | Làm thay đổi kích thước; phù hợp với exact-size repair nhưng không phù hợp để cải thiện phân phối sau khi size đã khóa |
| One-for-one swap | Loại một ảnh, thêm một ảnh | Giữ nguyên kích thước; dễ giữ No Finding, coverage và nested prefix; có thể duyệt exhaustive | Chỉ bảo đảm local optimum trong neighborhood 1–1 |
| Two-for-two swap | Loại hai ảnh, thêm hai ảnh | Có thể thoát một số local optimum 1–1 | Không gian tìm kiếm tăng mạnh theo tổ hợp; benchmark thực tế của dự án không hoàn thành budget 1% trong 600 giây |
| General k-for-k swap | Loại và thêm cùng `k` ảnh | Neighborhood rộng hơn 1–1 và 2–2 | Chi phí tổ hợp tăng rất nhanh; khó duyệt exhaustive |
| Greedy constrained construction | Thêm tuần tự ảnh có lợi nhất | Nhanh, dễ triển khai | Phụ thuộc đường đi; không có điều kiện dừng theo neighborhood exhaustive sau khi tập đã đủ size |
| Beam search | Giữ nhiều candidate tốt ở mỗi bước | Khám phá rộng hơn greedy | Beam width là hyperparameter; vẫn cắt bỏ phần lớn không gian tìm kiếm |
| Simulated annealing | Có thể nhận tạm thời move xấu | Có khả năng thoát local optimum | Cần temperature schedule; tăng tính ngẫu nhiên và phức tạp audit |
| Tabu search | Cấm tạm thời một số move/state đã thăm | Hạn chế vòng lặp và có thể thoát local optimum | Cần tabu tenure và tiêu chí dừng; thêm hyperparameter chưa được khóa |
| Genetic algorithm | Tiến hóa một quần thể subset | Tìm kiếm trên phạm vi rộng | Nhiều hyperparameter, chi phí lớn và nhạy với seed |
| Random restart/multiple seeds | Khởi tạo nhiều lần rồi chọn nghiệm | Có thể tìm objective tốt hơn | Nếu chọn seed sau khi xem kết quả sẽ vi phạm chính sách `PRE_SPECIFIED_LOCKED_NO_SEED_SEARCH` |
| Mixed-Integer Programming | Mô hình hóa membership bằng biến nhị phân | Có thể tối ưu hoặc báo optimality gap nếu solver hoàn tất | Cần solver mới, tuyến tính hóa objective, cấu hình gap/time limit và bằng chứng solver status |
| Constraint Programming/CP-SAT | Biểu diễn ràng buộc logic và số nguyên | Mô hình hóa size, No Finding, coverage và nesting tốt | Chi phí và trạng thái tối ưu phụ thuộc solver; là thay đổi protocol lớn |
| Clustering/core-set selection | Chọn ảnh đại diện trong feature space | Có thể tăng độ đa dạng biểu diễn | Phụ thuộc embedding/model; không trực tiếp bảo đảm các target hiện tại và có thể đưa thêm thiên kiến mô hình |

Danh sách trên không có nghĩa tất cả phương án đã được triển khai và so sánh
thực nghiệm trong dự án. Bằng chứng thực nghiệm hiện có chỉ bao gồm protocol
one-for-one đã hoàn tất và các benchmark two-for-two R8/R9. Những phương án
còn lại là các lựa chọn phương pháp luận có thể có, không phải baseline đã
được đánh giá.

#### 4.6.3 Tiêu chí lựa chọn của Phase 2F

Phase 2F cần một cơ chế repair đáp ứng đồng thời các yêu cầu:

1. giữ đúng kích thước labeled của từng budget;
2. giữ hoặc điều chỉnh có kiểm soát số ảnh No Finding;
3. bảo đảm hiện diện đủ 14 lớp bất thường;
4. không thay đổi locked prefix của budget nhỏ hơn;
5. cải thiện objective phân phối bằng so sánh exact integer;
6. deterministic với seed `42` đã khóa trước;
7. có thể kiểm toán toàn bộ candidate neighborhood;
8. hoàn thành trong thời gian vận hành chấp nhận được trên dữ liệu thực tế;
9. không cần chọn seed hoặc hyperparameter sau khi quan sát kết quả.

One-for-one đáp ứng trực tiếp các tiêu chí trên:

- **Giữ kích thước theo cấu trúc:** loại một và thêm một nên tổng số ảnh không
  đổi.
- **Kiểm soát No Finding:** trong coverage/objective repair, hai ảnh swap có
  cùng trạng thái `zero_gt`, nên số No Finding không đổi. Riêng
  `exact_no_finding` swap hai trạng thái đối nghịch để thay đổi target đúng một
  đơn vị mà vẫn giữ size.
- **Giữ nesting:** ảnh thuộc locked prefix không nằm trong tập được phép loại.
- **Bảo đảm coverage:** coverage repair chỉ chấp nhận move làm giảm nghiêm
  ngặt số lớp còn thiếu.
- **Cải thiện phân phối:** objective repair chỉ nhận move cải thiện nghiêm ngặt
  `(E_max, E_mean, E_LC)`.
- **Deterministic:** khi objective hòa, seeded SHA-256 priority và canonical
  numeric fallback tạo thứ tự lựa chọn xác định.
- **Có điều kiện dừng rõ ràng:** toàn bộ admissible one-for-one neighborhood
  được duyệt; không còn move cải thiện nghĩa là đạt one-for-one local optimum.
- **Khả thi thực tế:** cả bốn budget đã hoàn thành trong official run.

#### 4.6.4 Bằng chứng thực nghiệm hỗ trợ quyết định

Các bằng chứng của chính Phase 2F R11 gồm:

| Bằng chứng | Kết quả |
|---|---|
| Guardrail suite | `183 passed, 15 subtests passed` |
| Official construction | Cả bốn budget `OUTCOME=OK` |
| Kích thước labeled | `34, 171, 343, 685` |
| No Finding trong labeled | `3, 17, 35, 70` |
| Tổng repair moves | `46` |
| Tổng construction | `1.646,329` giây, khoảng 27 phút 26 giây |
| Reconstruct-check | Cả bốn budget `MATCH=True` |
| Kết quả chung | `RECONSTRUCT_CHECK_STATUS=MATCH` |
| Seed policy | `42`, `PRE_SPECIFIED_LOCKED_NO_SEED_SEARCH` |

Chi phí theo budget là 7,063 giây (1%), 130,437 giây (5%), 253,485 giây
(10%) và 1.255,344 giây (20%). Điều này chứng minh khả năng vận hành trên cấu
hình nghiên cứu đã dùng; không nên suy rộng thành cam kết tốc độ cho mọi máy
hoặc mọi dataset.

Đối với neighborhood rộng hơn, benchmark two-for-two R8 và R9 đều trả
`COMPUTATIONAL_ABORT` ở ngay budget 1% sau khoảng 600 giây; các budget sau
không được chạy do phụ thuộc. Kết quả này không chứng minh rằng mọi triển khai
two-for-two đều bất khả thi, nhưng cho thấy implementation và protocol đã khóa
khi đó không đáp ứng yêu cầu vận hành của dự án.

#### 4.6.5 Kết luận lựa chọn

One-for-one được chọn không phải vì đã được chứng minh là phương pháp tối ưu
nhất trong mọi lựa chọn, mà vì đây là neighborhood nhỏ nhất đồng thời:

- bảo toàn các hard constraints;
- cho phép duyệt exhaustive trong thực tế;
- có thứ tự lựa chọn deterministic và audit được;
- không yêu cầu seed search hoặc hyperparameter tìm kiếm mới;
- đã hoàn thành và tái lập chính xác trên dữ liệu Phase 2F.

Đánh đổi là nghiệm cuối chỉ được khẳng định là **one-for-one local optimum**.
Có thể tồn tại một move 2–2 hoặc `k`–`k` cải thiện objective dù không còn move
1–1 nào cải thiện. Vì vậy, tài liệu và báo cáo không được gọi nghiệm là global
optimum, cũng không được suy luận rằng membership này chắc chắn làm tăng mAP.
Hiệu quả downstream chỉ có thể được xác định qua thí nghiệm huấn luyện và đánh
giá mô hình.

#### 4.6.6 One-for-one local optimum khác global optimum như thế nào?

Giả sử `S` là labeled subset hiện tại và `F` là tập tất cả labeled subset khả
thi của budget đang xét. Một subset thuộc `F` phải đồng thời thỏa các hard
constraints đã khóa, gồm đúng kích thước, đúng số No Finding, đủ 14 lớp và
không loại bất kỳ ảnh nào thuộc nested locked prefix.

Gọi `J(S)` là objective nguyên theo thứ tự từ điển:

```text
J(S) = (E_max, E_mean, E_LC)
```

và gọi `N_1-1(S)` là **one-for-one neighborhood** của `S`: tập các nghiệm khả
thi có thể tạo ra bằng cách loại đúng một ảnh được phép loại khỏi `S` và thêm
đúng một ảnh từ pool. Các swap không thỏa hard constraints không thuộc
neighborhood hợp lệ này.

Kết quả `S` là **one-for-one local optimum** khi:

```text
không tồn tại S' thuộc N_1-1(S) sao cho J(S') tốt hơn J(S).
```

Trong Phase 2F, kết luận này có cơ sở vì ở mỗi vòng
`repair_objective_local_search()` duyệt toàn bộ admissible one-for-one moves
và chỉ nhận một move nếu objective được cải thiện nghiêm ngặt. Khi một move
được nhận, neighborhood thay đổi và chương trình duyệt lại trên nghiệm mới.
Quá trình chỉ dừng khi không còn move 1–1 hợp lệ nào cải thiện objective. Vì
vậy, tuyên bố chính xác là: **nghiệm cuối tốt nhất đối với các thay đổi 1–1
được phép quanh chính nghiệm đó**.

**Global optimum** là yêu cầu mạnh hơn. Một nghiệm `S*` chỉ là global optimum
nếu:

```text
không tồn tại bất kỳ T thuộc F sao cho J(T) tốt hơn J(S*).
```

Điều kiện này phải xét toàn bộ không gian subset khả thi, không chỉ các nghiệm
cách nghiệm hiện tại một swap 1–1. Phase 2F không duyệt toàn bộ `F`, không dùng
solver kèm chứng nhận optimality và không thiết lập optimality gap. Do đó,
không có bằng chứng để khẳng định global optimum.

Ví dụ minh họa: giả sử nghiệm hiện tại có objective `(10, 60, 8)`. Mọi swap
1–1 hợp lệ đều cho objective bằng hoặc xấu hơn `(10, 60, 8)`, nên nghiệm này
là one-for-one local optimum. Tuy nhiên, có thể cần đồng thời loại hai ảnh và
thêm hai ảnh khác mới đạt `(9, 55, 7)`. Không ảnh nào trong bốn ảnh đó tạo ra
cải thiện khi đổi riêng lẻ; lợi ích chỉ xuất hiện khi thực hiện cả move 2–2.
Khi đó nghiệm 1–1 hiện tại vẫn là local optimum, nhưng rõ ràng không phải
global optimum.

Cũng có thể tồn tại một chuỗi nhiều swap mà bước đầu làm objective tạm thời
xấu đi hoặc giữ nguyên, sau đó mới dẫn tới nghiệm tốt hơn. Phase 2F không đi
theo chuỗi đó vì chỉ chấp nhận cải thiện nghiêm ngặt ở từng bước. Đây là đặc
điểm của local search đã khóa, không phải lỗi thực thi.

Ba tuyên bố cần được phân biệt:

| Tuyên bố | Phase 2F có chứng minh không? | Ý nghĩa |
|---|---|---|
| Không còn admissible swap 1–1 cải thiện | Có | One-for-one local optimum |
| Không còn move 2–2, `k`–`k` hoặc chuỗi move cải thiện | Không | Neighborhood rộng hơn chưa được giải quyết đầy đủ |
| Không có bất kỳ subset khả thi nào tốt hơn | Không | Global optimum chưa được chứng minh |

Vì vậy, cách viết khoa học phù hợp là:

> Sau khi deterministic constrained repair hội tụ, không còn admissible
> one-for-one swap nào cải thiện nghiêm ngặt objective đã xác định trước trong
> khi vẫn bảo toàn các hard constraints. Nghiệm cuối do đó đạt one-for-one
> local optimum; quy trình không cung cấp bảo đảm tối ưu toàn cục.

Không nên viết “thuật toán tìm được phân hoạch tối ưu” hoặc “membership tối
ưu”, vì hai cách diễn đạt này dễ bị hiểu là global optimum. Nếu cần viết ngắn,
có thể dùng “phân hoạch khả thi đạt one-for-one local optimum theo objective đã
khóa”.

#### 4.6.7 Cách trả lời ngắn gọn khi reviewer hỏi

> Candidate subset ban đầu được tạo bằng iterative multilabel
> stratification; one-for-one swap chỉ được dùng làm deterministic constrained
> repair. Neighborhood này giữ nguyên kích thước theo cấu trúc, cho phép kiểm
> soát chính xác thành phần No Finding, bảo vệ nested prefix và duy trì đủ 14
> lớp. Mọi admissible one-for-one move được đánh giá bằng objective nguyên đã
> xác định trước, với tie-break deterministic. Lựa chọn này đã hoàn thành cả
> bốn budget trong khoảng 27,4 phút và reconstruct-check tái tạo chính xác toàn
> bộ membership. Neighborhood two-for-two rộng hơn đã được khảo sát nhưng
> không hoàn thành ngay budget 1% trong giới hạn benchmark 600 giây. Do đó,
> one-for-one được chọn như một thỏa hiệp có kiểm soát giữa bảo toàn ràng buộc,
> khả năng kiểm toán, tính tái lập và chi phí tính toán; nghiên cứu chỉ tuyên
> bố one-for-one local optimum, không tuyên bố global optimum.

## 5. Giải thích các nhóm hàm trong script

### 5.1 Đọc cấu hình và I/O

- `load_config()`: đọc YAML và kiểm tra cấu trúc cấu hình cơ bản.
- `get_protocol_identity()`: lấy `stage` và `version`; lỗi nếu thiếu hoặc sai
  kiểu.
- `load_json()`, `write_json()`, `write_jsonl()`: đọc/ghi JSON và JSONL.
- `sha256_file()`: tính SHA-256 của nội dung một tệp.

### 5.2 Preflight

`run_preflight()` chạy trước construction. Hàm xác minh:

- ba COCO split Train/Validation/Test tồn tại;
- số ảnh, annotation, No Finding và category khớp bằng chứng Phase 2E;
- SHA-256 của các đầu vào không thay đổi;
- ba split không giao nhau và hợp lại đúng phạm vi 4.894 ảnh;
- category mapping giống nhau giữa ba split;
- phiên bản `iterative-stratification` đúng `0.1.9`;
- active repair policy trong YAML khớp code.

Preflight `PASS` chỉ chứng minh đầu vào và protocol nhất quán. Nó chưa tạo
labeled/unlabeled và không có nghĩa training đã được cho phép.

### 5.3 Tạo biểu diễn tính toán

`build_indicators()` chuyển COCO Train thành:

- danh sách `image_ids`;
- ma trận `labels14`;
- vector `zero_gt`;
- tên 14 lớp.

`full_train_integer_stats()` tính các thống kê phân phối trên toàn bộ Train để
làm mốc cho objective.

### 5.4 Tính target

`compute_locked_size_targets()` tính kích thước labeled và số No Finding của
từng budget theo round-half-up, sau đó đối chiếu với target đã khóa trong YAML.
Nếu tính lại không khớp, chương trình dừng thay vì âm thầm dùng con số cấu hình.

### 5.5 Candidate ban đầu

`stratified_initial_candidate()` sử dụng iterative multilabel stratification
trên 15 chỉ báo lấy mẫu: 14 lớp bất thường cộng một chỉ báo `zero_gt`.

Candidate này mới là điểm bắt đầu. Stratification có thể chưa đạt chính xác
kích thước, số No Finding hoặc coverage, nên phải qua các bước repair.

### 5.6 Repair đúng kích thước

`repair_exact_size()` thêm hoặc loại từng ảnh cho đến khi đạt đúng target của
budget. Với budget lớn hơn, locked prefix không bị thay đổi.

### 5.7 Repair đúng số No Finding

`repair_exact_no_finding()` dùng one-for-one swaps để đạt đúng target No
Finding trong khi giữ nguyên tổng kích thước labeled.

### 5.8 Bảo đảm đủ 14 lớp

`repair_min_class_coverage()` tìm one-for-one swap làm giảm số lớp còn thiếu.
Thứ tự ưu tiên là:

1. còn ít lớp thiếu hơn sau swap;
2. objective phân phối tốt hơn;
3. SHA-256 priority có seed để phá hòa một cách deterministic;
4. thứ tự số học của `image_id` nếu digest vẫn hòa.

Nếu đã duyệt hết neighborhood hợp lệ mà vẫn thiếu lớp, outcome là
`REPAIR_INFEASIBLE`; chương trình không đổi seed để tìm kết quả thuận lợi hơn.

### 5.9 Cải thiện phân phối

`repair_objective_local_search()` duyệt exhaustive toàn bộ one-for-one
neighborhood hợp lệ tại mỗi vòng và chỉ nhận move cải thiện **nghiêm ngặt**
objective nguyên theo thứ tự từ điển:

```text
(E_max, E_mean, E_LC)
```

Trong đó:

- `E_max`: sai lệch prevalence lớn nhất trong 14 lớp;
- `E_mean`: tổng sai lệch prevalence tuyệt đối của 14 lớp;
- `E_LC`: sai lệch mean label cardinality.

Tính toán dùng số nguyên nhân chéo, không dùng float để quyết định move. Khi
không còn one-for-one move cải thiện, kết quả là **one-for-one local optimum**.
Đây không phải bằng chứng về global optimum.

### 5.10 Xây dựng một budget

`build_budget_step()` là hàm điều phối toàn bộ một budget:

```text
stratification
→ exact_size
→ exact_no_finding
→ minimum_class_coverage
→ objective_repair
```

Hàm trả về membership, outcome và diagnostics. Ba outcome quan trọng:

- `OK`: budget hoàn tất hợp lệ;
- `REPAIR_INFEASIBLE`: ràng buộc không thể đạt trong neighborhood đã khóa;
- `COMPUTATIONAL_ABORT`: chạm global deadline/resource cutoff.

### 5.11 Xây dựng cả bốn budget

`compute_all_budgets()`:

- tạo đúng một global deadline nếu có `--max-seconds`;
- chạy `build_budget_step()` theo thứ tự `1% → 5% → 10% → 20%`;
- khóa membership budget trước cho budget sau;
- dừng ngay nếu một budget không trả `OK`;
- đo thời gian riêng từng budget bằng `time.monotonic()`;
- đo tổng thời gian construction.

Timing chỉ phục vụ quan sát. Nó không tham gia seed, objective, tie-break,
checksum, acceptance criterion hoặc membership identity.

### 5.12 Tạo COCO labeled và unlabeled

- `subset_labeled_coco()`: giữ ảnh labeled và toàn bộ annotation tương ứng.
- `subset_unlabeled_coco()`: giữ ảnh unlabeled nhưng đặt `annotations=[]` và
  loại các trường suy ra từ ground truth như `is_negative`, `scope_label`,
  `zero_gt`, `class_presence` hoặc thông tin bbox.

Việc loại các trường này ngăn rò rỉ nhãn vào nhánh unlabeled.

### 5.13 Kiểm định staging

`independently_validate_staging()` đọc lại chính các tệp tạm vừa ghi và kiểm
tra độc lập:

- cấu trúc COCO;
- ID trùng, ID ngoài phạm vi hoặc annotation treo;
- labeled/unlabeled rời nhau và hợp lại đúng Train;
- checksum membership và COCO;
- exact size, exact No Finding, đủ 14 lớp;
- quan hệ nested;
- unlabeled không chứa annotation hoặc trường GT bị cấm.

### 5.14 Ghi và promote artifact

`write_all_outputs_and_promote()` ghi mọi kết quả vào staging trước. Chỉ khi
tất cả validation đều `PASS`, `promote_with_rollback()` mới chuyển chúng thành
official artifacts. Nếu một bước lỗi, staging bị xóa và các tệp đã chuyển một
phần được rollback.

Chương trình cũng từ chối ghi đè official artifacts đã tồn tại. Vì vậy, sau
khi Phase 2F đã `PASS`, không nên chạy lại full mode để “kiểm tra”.

### 5.15 Reconstruct-check

`run_reconstruct_check()` xây dựng lại deterministic membership trong vùng
tạm rồi so sánh với lock manifest đã promote. Các trường so sánh gồm:

- membership SHA-256;
- labeled và unlabeled COCO SHA-256;
- kích thước labeled;
- số No Finding;
- số repair moves;
- objective nguyên cuối cùng.

Timing không được so sánh vì thời gian chạy tự nhiên có thể khác giữa hai lần.
Reconstruct-check không ghi đè official labeled/unlabeled artifacts.

## 6. Hai lệnh vận hành

### 6.1 Lệnh chính

```bat
python scripts\02F_build_labeled_unlabeled.py
```

Lệnh này xây dựng, kiểm định và promote official artifacts. Chỉ cần chạy một
lần khi chưa có official artifacts.

Có thể đặt một global deadline cho toàn bộ construction:

```bat
python scripts\02F_build_labeled_unlabeled.py --max-seconds 7200
```

Không truyền `--max-seconds` nghĩa là không có resource cutoff theo thời gian;
các lỗi dữ liệu, I/O, bộ nhớ hoặc exception khác vẫn có thể làm chương trình
dừng.

### 6.2 Lệnh xác minh tái lập

```bat
python scripts\02F_build_labeled_unlabeled.py --reconstruct-check
```

Lệnh này chỉ dùng sau lệnh chính. Kết quả mong đợi là:

```text
RECONSTRUCT_CHECK_STATUS= MATCH
```

## 7. Cách đọc console

Sau mỗi budget thực sự chạy:

```text
BUDGET=<budget> OUTCOME=<outcome> SIZE=<size> ELAPSED_SECONDS=<seconds>
```

Sau toàn bộ construction:

```text
TOTAL_CONSTRUCTION_ELAPSED_SECONDS=<seconds>
```

Budget không chạy vì budget trước thất bại sẽ không có elapsed time giả.

Kết quả official R11 đã ghi nhận:

| Budget | Outcome | Size | Thời gian |
|---|---|---:|---:|
| 1% | OK | 34 | 7,063 giây |
| 5% | OK | 171 | 130,437 giây |
| 10% | OK | 343 | 253,485 giây |
| 20% | OK | 685 | 1.255,344 giây |

Tổng construction là `1.646,329` giây, xấp xỉ 27 phút 26 giây. Budget 20%
chiếm phần lớn thời gian vì one-for-one neighborhood lớn hơn; điều này tự nó
không phải lỗi.

## 8. Ý nghĩa các tệp đầu ra

### 8.1 COCO dùng cho thí nghiệm

- `instances_labeled_1pct.json` đến `instances_labeled_20pct.json`: ảnh có
  nhãn và annotation thật tương ứng.
- `instances_unlabeled_1pct.json` đến `instances_unlabeled_20pct.json`: phần
  Train còn lại, không chứa annotation và trường GT bị cấm.

### 8.2 Manifest

- `phase2F_partition_manifest.csv`: với từng `image_id`, cho biết ảnh thuộc
  labeled budget nào.
- `phase2F_lock_manifest.json`: bằng chứng khóa membership, kích thước,
  checksum, objective, seed và timing.
- `phase2F_nested_split_check.json`: kiểm tra quan hệ lồng nhau.
- `phase2F_leakage_check.json`: kiểm tra labeled/unlabeled và GT leakage.
- `phase2F_seed_manifest.json`: ghi seed policy và phiên bản dependency.
- `phase2F_unlabeled_gt_audit.csv`: audit ground truth của ảnh được đưa vào
  unlabeled; đây là artifact kiểm toán, không phải đầu vào huấn luyện.

### 8.3 Báo cáo

- `02F_labeled_unlabeled_validation_report.json`: báo cáo kiểm định tổng hợp,
  phân phối, repair, checksum và timing.
- `02F_labeled_unlabeled_log.json`: nhật ký trạng thái materialization.
- `02F_class_distribution.csv`: phân phối 14 lớp theo budget.
- `02F_negative_distribution.csv`: số và tỷ lệ No Finding/abnormal.
- `02F_repair_log.jsonl`: từng repair move được chấp nhận.
- `02F_errors.csv`: danh sách lỗi; ở lần chạy thành công chỉ có header.
- `02F_deterministic_reconstruction_check.json`: kết quả so sánh lần dựng lại
  với official lock manifest.

### 8.4 Output File

OUTPUT_FILE= D:\ssl_detection_xray_v2\data\processed\coco\labeled_splits\instances_labeled_1pct.json
OUTPUT_FILE= D:\ssl_detection_xray_v2\data\processed\coco\labeled_splits\instances_labeled_5pct.json
OUTPUT_FILE= D:\ssl_detection_xray_v2\data\processed\coco\labeled_splits\instances_labeled_10pct.json
OUTPUT_FILE= D:\ssl_detection_xray_v2\data\processed\coco\labeled_splits\instances_labeled_20pct.json
OUTPUT_FILE= D:\ssl_detection_xray_v2\data\processed\coco\unlabeled_splits\instances_unlabeled_1pct.json
OUTPUT_FILE= D:\ssl_detection_xray_v2\data\processed\coco\unlabeled_splits\instances_unlabeled_5pct.json
OUTPUT_FILE= D:\ssl_detection_xray_v2\data\processed\coco\unlabeled_splits\instances_unlabeled_10pct.json
OUTPUT_FILE= D:\ssl_detection_xray_v2\data\processed\coco\unlabeled_splits\instances_unlabeled_20pct.json
OUTPUT_FILE= D:\ssl_detection_xray_v2\data\manifests\audit\phase2F_unlabeled_gt_audit.csv
OUTPUT_FILE= D:\ssl_detection_xray_v2\data\manifests\phase2F_partition_manifest.csv
OUTPUT_FILE= D:\ssl_detection_xray_v2\data\manifests\phase2F_lock_manifest.json
OUTPUT_FILE= D:\ssl_detection_xray_v2\data\manifests\phase2F_nested_split_check.json
OUTPUT_FILE= D:\ssl_detection_xray_v2\data\manifests\phase2F_leakage_check.json
OUTPUT_FILE= D:\ssl_detection_xray_v2\data\manifests\phase2F_seed_manifest.json
OUTPUT_FILE= D:\ssl_detection_xray_v2\reports\02F_labeled_unlabeled_validation_report.json
OUTPUT_FILE= D:\ssl_detection_xray_v2\reports\02F_labeled_unlabeled_log.json
OUTPUT_FILE= D:\ssl_detection_xray_v2\reports\02F_class_distribution.csv
OUTPUT_FILE= D:\ssl_detection_xray_v2\reports\02F_negative_distribution.csv
OUTPUT_FILE= D:\ssl_detection_xray_v2\reports\02F_repair_log.jsonl
OUTPUT_FILE= D:\ssl_detection_xray_v2\reports\02F_errors.csv


## 9. Vai trò của file test

`tests/test_phase2F_labeled_unlabeled_guardrails.py` bảo vệ các bất biến khoa
học và kỹ thuật, gồm:

- phép làm tròn target;
- checksum canonical;
- exact size và exact No Finding;
- minimum 14-class coverage;
- objective nguyên và thứ tự tie-break;
- nested prefix không bị thay đổi;
- không rò rỉ GT vào unlabeled;
- readback staging độc lập;
- rollback và không promote kết quả dở dang;
- deadline taxonomy;
- timing không thay đổi membership;
- reconstruct-check không so timing;
- active scientific guardrails không được skip/xfail.

Kết quả đã xác nhận trên môi trường nghiên cứu là:

```text
183 passed, 15 subtests passed
```

## 10. Trạng thái đã xác nhận của Phase 2F

Official run:

```text
PHASE_2F_GATE=PASS
```

Reconstruct-check:

```text
BUDGET=1pct MATCH=True
BUDGET=5pct MATCH=True
BUDGET=10pct MATCH=True
BUDGET=20pct MATCH=True
RECONSTRUCT_CHECK_STATUS=MATCH
```

Do đó, Phase 2F đã xây dựng và tái lập thành công các tập labeled/unlabeled.
Điều này chứng minh tính nhất quán với protocol và lock manifest, nhưng không
tự động chứng minh hiệu quả mô hình và không tự động cho phép training. Các
trạng thái vẫn được ghi rõ:

```text
training_authorized=false
training_started=false
pseudo_labels_generated=false
test_used=false
```

## 11. Những điều không nên hiểu sai

1. `PASS` không có nghĩa các subset là phân hoạch tối ưu toàn cục; objective
   repair chỉ bảo đảm one-for-one local optimum.
2. Seed `42` dùng để tái lập và phá hòa deterministic, không được lựa chọn sau
   khi xem kết quả.
3. No Finding không phải class detection thứ 15.
4. Unlabeled vẫn có ground truth trong dataset gốc để audit, nhưng file COCO
   đưa vào nhánh unlabeled không được chứa ground truth đó.
5. Reconstruct `MATCH` chứng minh tái lập artifact theo các trường đã khóa,
   không phải đánh giá chất lượng mô hình.
6. Validation và Test của Phase 2E không tham gia xây dựng labeled/unlabeled.
7. Timing chỉ mô tả chi phí thực thi; chạy nhanh hay chậm không quyết định ảnh
   nào được chọn.

## 12. Tóm tắt một câu

Phase 2F lấy duy nhất tập Train đã khóa của Phase 2E, tạo bốn labeled subset
lồng nhau bằng multilabel stratification và deterministic one-for-one repair,
tạo unlabeled bằng phép hiệu với Train, kiểm định độc lập và chỉ promote khi
toàn bộ guardrail đều đạt.
