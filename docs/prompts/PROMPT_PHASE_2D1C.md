Bạn là Senior Machine Learning Engineer kiêm Research Software Engineer, có kinh nghiệm với:

- MMDetection 3.3.0
- MMEngine 0.10.7
- MMCV 2.1.0
- COCO detection datasets
- PyTorch dataloader
- kiểm định dữ liệu cho object detection
- thiết kế pipeline nghiên cứu có khả năng tái lập và audit

Hãy làm việc trực tiếp trong repository:

/content/ssl_detection_xray_v2

NHIỆM VỤ

Triển khai:

Phase 2D.1C — MMDetection Dataset-Loading & Empty-Image Retention Validation

Đây là phase kiểm định khả năng nạp dữ liệu và giữ ảnh không có bounding box trong MMDetection. Đây KHÔNG phải phase huấn luyện mô hình.

==================================================
1. BỐI CẢNH ĐÃ KHÓA
==================================================

Các phase trước đã hoàn tất:

- Phase 2A: Data Standardization / Image-Boundary Validation — PASS
- Phase 2B: Canonical Detection Annotation Schema — PASS
- Phase 2C: Framework & Format Decision — PASS
- Phase 2D.1A: Image Representation Protocol — PASS
- Phase 2D.1B-Pilot: DICOM-to-JPG pilot — PASS
- Phase 2D.1B-Full: Full controlled-scope conversion — COMPLETED

Thuật ngữ chính thức:

“DICOM metadata-aware, standard-aligned reference representation pipeline”

Không được gọi đây là phương pháp mới, thuật toán mới, “optimal preprocessing” hay “clinical-grade renderer”.

Đầu vào đã được kiểm tra độc lập và PASS:

- COCO images: 4,894
- COCO annotations: 36,096
- COCO categories: 14
- Abnormal images: 4,394
- Zero-GT / No Finding images: 500
- Missing referenced JPG files: 0
- Invalid annotation image references: 0
- Invalid annotation category references: 0
- Unique resolved JPG paths: 4,894

Đường dẫn dự kiến:

- COCO JSON:
  data/processed/coco/coco_master_jpg.json

- Image root:
  data/processed/images_jpg/

- Các ảnh thực tế:
  data/processed/images_jpg/train/*.jpg

COCO JSON SHA-256 đã quan sát:

f587152278f713460ff1e727a2912248a47052f6abc48de8f7bad6e8a63b94c0

Môi trường đã được kiểm định:

- Python 3.10.16
- PyTorch 2.1.0+cu118
- TorchVision 0.16.0+cu118
- CUDA runtime 11.8
- MMCV 2.1.0
- MMEngine 0.10.7
- MMDetection 3.3.0
- NumPy 1.26.4
- MMCV CUDA ops: PASS
- MMDetection imports: PASS
- TRAINING ENVIRONMENT: PASS
- ENVIRONMENT VALIDATION: PASS
- MMDETECTION ENVIRONMENT REPRODUCTION: PASS

Python interpreter bắt buộc trên Colab:

/content/miniconda/envs/mmdet330/bin/python

==================================================
2. FILE PHẢI ĐỌC TRƯỚC KHI VIẾT CODE
==================================================

Trước khi chỉnh sửa hoặc tạo file, phải đọc đầy đủ các file đang tồn tại sau:

- configs/protocol/phase2D1_jpg_representation.yaml
- PHASE_HANDOFF.md
- PROJECT_CONTEXT.md
- research_log.md
- .gitignore
- scripts/setup_mmdet330_colab.sh
- scripts/validate_mmdet330_environment.sh
- scripts/02D1B_full_dicom_to_jpg.py
- tests/test_phase2D1B_full_guardrails.py

Nếu có:

- CHECKLIST_TRIEN_KHAI_FULL.xlsx
- data/processed/coco/coco_master_jpg.json

hãy kiểm tra chúng bằng công cụ phù hợp.

Phải kiểm tra cấu trúc thực tế của COCO JSON, đặc biệt:

- kiểu dữ liệu và miền giá trị của image IDs;
- category IDs;
- annotation IDs;
- nội dung file_name;
- width và height;
- bbox;
- area;
- iscrowd;
- số ảnh không có annotation.

Không được giả định file_name chỉ chứa tên file. Nó có thể có dạng:

- train/<image_id>.jpg; hoặc
- <image_id>.jpg.

Code phải xử lý theo cấu trúc thật được quan sát trong repository, không được âm thầm sửa COCO JSON để phù hợp với giả định của mình.

Nếu một file không tồn tại, hãy báo rõ file nào thiếu. Không được bịa nội dung file.

==================================================
3. MỤC TIÊU KHOA HỌC CỦA PHASE
==================================================

Phải chứng minh bằng MMDetection thật rằng:

1. `CocoDataset` của MMDetection 3.3.0 nạp thành công COCO master.

2. Khi sử dụng:

   filter_cfg=dict(filter_empty_gt=False)

   dataset giữ đúng toàn bộ 4,894 image records.

3. Dataset giữ đúng 500 ảnh zero-GT / No Finding.

4. Khi so sánh có kiểm soát với:

   filter_cfg=dict(filter_empty_gt=True)

   phải quan sát và báo cáo chính xác hành vi lọc thực tế.

Nếu dữ liệu đúng như trạng thái đã khóa, dự kiến:

- `filter_empty_gt=False`: 4,894 ảnh;
- `filter_empty_gt=True`: 4,394 ảnh;
- chênh lệch: 500 ảnh.

Tuy nhiên, không được hard-code kết quả PASS chỉ vì đây là giá trị dự kiến. Phải đo từ dataset MMDetection thật.

5. Một ảnh abnormal có annotation đi qua pipeline validation thành công.

6. Một ảnh zero-GT đi qua pipeline validation thành công và vẫn có:

- `gt_instances.bboxes` với shape `(0, 4)`;
- `gt_instances.labels` với shape `(0,)`.

Không được biến ảnh empty-GT thành annotation giả, dummy box, background box hoặc category “No Finding”.

7. Bounding box và label sau pipeline phải hợp lệ.

8. Dataloader phải tạo batch thành công.

9. Phải chứng minh một batch hoặc một lần lấy mẫu có chứa zero-GT sample thật sự.

Không được chỉ dựa vào xác suất shuffle để mong rằng zero-GT xuất hiện. Hãy dùng sampler, subset hoặc batch sampler xác định/deterministic để buộc kiểm tra một hoặc nhiều zero-GT samples.

10. Không được để evaluator hoặc API COCO vô tình loại zero-GT images khỏi dataset đang kiểm định.

==================================================
4. PHẠM VI ĐƯỢC PHÉP
==================================================

Được phép:

- tạo script validation;
- tạo unit/guardrail tests;
- tạo config validation riêng nếu thực sự cần;
- đọc COCO JSON và JPG;
- build `mmdet.datasets.CocoDataset` bằng registry chính thức;
- sử dụng MMEngine runner utilities khi phù hợp;
- xây dựng dataloader kiểm định;
- xuất JSON/CSV/Markdown evidence;
- thêm các đường dẫn report nhẹ vào Git;
- cập nhật tài liệu trạng thái chỉ sau khi validation thực sự PASS.

Không được:

- training bất kỳ detector nào;
- tải pretrained weights;
- chạy inference mô hình;
- tạo train/validation/test split;
- tạo labeled-percentage subsets;
- chọn supervised hoặc semi-supervised model;
- sửa nội dung COCO master;
- sửa JPG;
- đổi category mapping;
- crop hoặc resize dữ liệu nguồn;
- thêm dummy annotations cho No Finding;
- xóa zero-GT images;
- thay đổi quyết định JPEG quality 95;
- thay đổi protocol đã khóa;
- đánh dấu training-ready chỉ dựa trên input-integrity check;
- tự ý mở Phase 2D.1D;
- tự ý commit hoặc push Git;
- cài đặt lại package hoặc thay đổi môi trường đã khóa.

Nếu phát hiện lỗi dữ liệu, phải fail rõ ràng và tạo evidence. Không được tự động “sửa cho chạy”.

==================================================
5. FILE CẦN TẠO
==================================================

Ưu tiên tạo đúng các file sau:

1. Script chính:

scripts/02D1C_validate_mmdet_dataset_loading.py

2. Guardrail tests:

tests/test_phase2D1C_mmdet_dataset_loading_guardrails.py

3. Report JSON do script sinh ra:

reports/phase2D1C_mmdet_dataset_loading_report.json

4. Report Markdown do script sinh ra:

reports/phase2D1C_mmdet_dataset_loading_report.md

5. Per-image audit CSV:

reports/phase2D1C_mmdet_dataset_image_audit.csv

6. Error report CSV:

reports/phase2D1C_mmdet_dataset_errors.csv

Chỉ tạo thêm config riêng nếu có lý do kỹ thuật rõ ràng, ví dụ:

configs/validation/phase2D1C_mmdet_dataset_loading.py

Không tạo file thừa hoặc sao chép dữ liệu JPG.

==================================================
6. YÊU CẦU CLI
==================================================

Script chính phải có CLI rõ ràng, tối thiểu hỗ trợ:

--repo-root
--ann-file
--data-root
--batch-size
--num-workers
--seed
--report-json
--report-md
--image-audit-csv
--errors-csv
--expected-images
--expected-annotations
--expected-categories
--expected-empty-images
--strict

Giá trị mặc định phải phù hợp Colab repository:

- repo root:
  /content/ssl_detection_xray_v2

- ann file:
  data/processed/coco/coco_master_jpg.json

- data root:
  data/processed/images_jpg/

- expected images: 4894
- expected annotations: 36096
- expected categories: 14
- expected empty images: 500
- seed: một giá trị cố định, ví dụ 42.

Script phải có `--help` hoạt động.

Không được hard-code tuyệt đối `/content/...` ở mọi nơi. Đường dẫn phải được resolve từ `--repo-root`, trừ default CLI có thể dùng đường dẫn Colab nói trên.

==================================================
7. PREFLIGHT BẮT BUỘC
==================================================

Script phải fail-fast và kiểm tra:

- đúng Python environment;
- import được torch, mmcv, mmengine và mmdet;
- version đúng:
  - MMDetection 3.3.0
  - MMCV 2.1.0
  - MMEngine 0.10.7;
- COCO JSON tồn tại;
- image root tồn tại;
- SHA-256 của COCO JSON được ghi vào report;
- số images, annotations và categories trong JSON;
- image IDs không trùng;
- annotation IDs không trùng;
- category IDs không trùng;
- mọi annotation tham chiếu image/category hợp lệ;
- tất cả file_name resolve tới JPG duy nhất;
- không có missing referenced JPG;
- zero-GT count được tính từ JSON thật;
- output report paths an toàn;
- script không ghi đè dữ liệu nguồn.

Nếu SHA-256 khác hash đã quan sát, không được âm thầm PASS. Trong strict mode phải FAIL hoặc ít nhất tạo lỗi nghiêm trọng yêu cầu review, tùy bằng chứng trong protocol hiện hành cho thấy hash đã được chính thức khóa hay mới chỉ được ghi nhận. Hãy đọc tài liệu trước khi quyết định.

==================================================
8. XÂY DỰNG MMDETECTION DATASET
==================================================

Phải dùng registry/API chính thức của MMDetection 3.3.0, ví dụ:

from mmengine.registry import init_default_scope
from mmdet.registry import DATASETS

init_default_scope('mmdet')

Không viết một lớp dataset giả để thay thế `mmdet.datasets.CocoDataset`.

Hãy xây dựng ít nhất hai dataset độc lập:

A. Retention dataset:

- type='CocoDataset'
- ann_file trỏ đúng COCO JSON;
- data_prefix=dict(img=...);
- filter_cfg=dict(filter_empty_gt=False);
- test_mode=False;
- metainfo phù hợp với 14 categories theo thứ tự/category mapping thực tế;
- pipeline MMDetection hợp lệ.

B. Controlled comparison dataset:

- giống A nhưng:
  filter_cfg=dict(filter_empty_gt=True).

Phải báo cáo:

- raw COCO image count;
- retention dataset length;
- filtered dataset length;
- số image IDs bị loại khi bật filter;
- số image IDs zero-GT;
- xác nhận tập bị loại có khớp chính xác tập zero-GT hay không.

Không được chỉ so sánh length. Phải so sánh image IDs.

==================================================
9. PIPELINE VALIDATION
==================================================

Thiết kế pipeline validation tối thiểu, không thực hiện augmentation ngẫu nhiên.

Ưu tiên các transform chính thức tương thích MMDetection 3.3.0, ví dụ:

- LoadImageFromFile
- LoadAnnotations(with_bbox=True)
- PackDetInputs

Không thêm RandomFlip, resize ngẫu nhiên, crop hoặc photometric augmentation.

Mục đích phase này là kiểm tra loading/retention, không phải kiểm tra chiến lược augmentation.

Đối với từng sample được audit, phải ghi nhận tối thiểu:

- dataset index;
- image_id;
- file_name hoặc img_path;
- raw width/height từ COCO;
- loaded image shape;
- raw annotation count;
- post-pipeline GT box count;
- post-pipeline label count;
- is_empty_gt;
- bbox validity result;
- label validity result;
- pipeline load result;
- error message nếu có.

Bounding box post-pipeline phải được kiểm tra:

- tensor shape là `(N, 4)`;
- mọi giá trị hữu hạn;
- x2 > x1;
- y2 > y1;
- tọa độ nằm trong giới hạn ảnh với tolerance hợp lý;
- số labels bằng số boxes;
- labels nằm trong `[0, num_classes - 1]`.

Phải nhận thức rằng COCO category IDs không nhất thiết bằng contiguous training labels của MMDetection. Hãy kiểm tra mapping thật, không nhầm category ID với label index.

==================================================
10. EMPTY-GT RETENTION VALIDATION
==================================================

Phải tìm zero-GT images từ COCO JSON và xác nhận với MMDetection dataset.

Tối thiểu:

- kiểm tra toàn bộ 500 zero-GT image IDs có trong dataset khi `filter_empty_gt=False`;
- kiểm tra toàn bộ zero-GT image IDs bị loại trong controlled dataset khi `filter_empty_gt=True`, nếu đó là hành vi thực tế;
- chạy pipeline trực tiếp cho một tập xác định các zero-GT images;
- phải có ít nhất một zero-GT sample được kiểm tra end-to-end;
- không chấp nhận chỉ inspect `data_list` mà không chạy pipeline.

Đối với zero-GT sample sau `PackDetInputs`, phải xác nhận:

- inputs tồn tại và là tensor;
- data_samples tồn tại;
- gt_instances tồn tại;
- bboxes shape `(0, 4)`;
- labels shape `(0,)`;
- không phát sinh exception;
- image metadata còn truy vết được.

Nếu API thực tế biểu diễn empty bbox theo kiểu tensor/subclass khác, hãy kiểm tra semantic tương đương và ghi rõ trong report. Không nới điều kiện chỉ để PASS.

==================================================
11. DATALOADER VALIDATION
==================================================

Phải xây dựng dataloader sử dụng MMEngine/PyTorch theo cách tương thích với MMDetection data samples.

Tối thiểu kiểm tra:

A. Normal deterministic batch

- tạo batch từ các abnormal samples;
- batch load thành công;
- inputs và data_samples có cấu trúc mong đợi.

B. Forced empty-GT batch hoặc mixed batch

- dùng deterministic indices/sampler/subset để chắc chắn có zero-GT sample;
- không dựa vào random shuffle;
- batch load thành công;
- xác định được sample zero-GT trong batch;
- zero-GT sample vẫn có empty `gt_instances`.

Nếu batch size > 1 gặp vấn đề do ảnh có kích thước khác nhau và chưa resize/pad, không được lén thêm resize làm thay đổi mục tiêu kiểm định.

Hãy chọn một trong các cách khoa học:

- batch_size=1 cho validation chính; hoặc
- dùng pseudo_collate phù hợp để giữ danh sách tensor khác kích thước.

Phải giải thích lựa chọn trong report.

`num_workers=0` phải được hỗ trợ để giảm bất định trong validation. Có thể kiểm tra thêm `num_workers>0`, nhưng không được biến nó thành điều kiện bắt buộc nếu Colab gây bất ổn ngoài phạm vi.

==================================================
12. REPORT VÀ EVIDENCE
==================================================

JSON report phải có cấu trúc rõ ràng và machine-readable, tối thiểu gồm:

- phase;
- generated_at_utc;
- repository/root paths;
- environment versions;
- input paths;
- COCO JSON SHA-256;
- expected counts;
- observed raw counts;
- input-integrity results;
- MMDetection dataset configuration;
- retention dataset results;
- controlled filtering comparison;
- empty-GT image-ID comparison;
- pipeline validation summary;
- bbox/label validation summary;
- dataloader validation summary;
- sample evidence;
- errors;
- warnings;
- checks;
- overall_status;
- dataset_loading_validated;
- empty_image_retention_validated;
- dataset_training_ready;
- training_authorized.

Trạng thái chỉ được đặt:

- `dataset_loading_validated: true` khi toàn bộ loading checks PASS;
- `empty_image_retention_validated: true` khi toàn bộ retention checks PASS;
- `dataset_training_ready: true` chỉ khi Definition of Done của chính Phase 2D.1C đã PASS và không còn blocker thuộc phase này;
- `training_authorized: false` trong mọi trường hợp ở script này.

Lưu ý quan trọng:

`dataset_training_ready: true` không đồng nghĩa `training_authorized: true`.

Phase này có thể xác nhận dataset đạt kiểm định kỹ thuật, nhưng không được tự cấp quyền bắt đầu training hoặc mở phase tiếp theo.

Markdown report phải tóm tắt cùng dữ liệu với JSON, không chứa số liệu viết tay tách rời dễ sai lệch.

CSV audit phải bao gồm toàn bộ 4,894 image records nếu chi phí chấp nhận được. Nếu không audit pipeline toàn bộ vì chi phí, vẫn phải audit metadata/retention toàn bộ và phân biệt rõ:

- full dataset structural audit;
- targeted post-pipeline audit.

Errors CSV phải luôn được tạo, kể cả không có lỗi; khi không lỗi chỉ có header.

Việc ghi report nên dùng cơ chế an toàn:

- tạo parent directory nếu cần;
- ghi file tạm;
- flush/close;
- atomic replace khi phù hợp.

==================================================
13. DEFINITION OF DONE
==================================================

Phase 2D.1C chỉ PASS khi tất cả điều kiện sau đều đạt:

1. Environment/version checks PASS.
2. COCO JSON nạp được.
3. Raw counts đúng:
   - 4,894 images;
   - 36,096 annotations;
   - 14 categories.
4. Tất cả 4,894 referenced JPG tồn tại và resolve duy nhất.
5. Raw zero-GT count đúng 500.
6. MMDetection `CocoDataset` build thành công.
7. `filter_empty_gt=False` giữ đúng 4,894 images.
8. Dataset chứa đầy đủ đúng 500 zero-GT image IDs.
9. Controlled comparison với `filter_empty_gt=True` được đo và giải thích.
10. Tập image IDs bị lọc khớp với tập zero-GT nếu hành vi thực tế của MMDetection đúng như dự kiến.
11. Ít nhất một abnormal sample chạy hết pipeline.
12. Ít nhất một zero-GT sample chạy hết pipeline.
13. Empty sample có bboxes `(0,4)` và labels `(0,)`, hoặc semantic tương đương được chứng minh rõ.
14. Bounding boxes và labels hợp lệ.
15. Dataloader lấy normal batch thành công.
16. Dataloader lấy forced empty-GT hoặc mixed batch thành công.
17. Report JSON, Markdown và CSV được sinh đầy đủ.
18. Errors CSV không có lỗi blocker.
19. Guardrail tests PASS.
20. Không sửa dữ liệu nguồn.
21. Không thực hiện training.
22. `training_authorized` vẫn là false.

Nếu bất kỳ điều kiện bắt buộc nào fail:

- overall status phải là FAIL;
- process exit code phải khác 0;
- ghi rõ failed check;
- không đánh dấu dataset training-ready;
- không cập nhật phase thành completed.

==================================================
14. GUARDRAIL TESTS
==================================================

Tạo tests có thể chạy bằng:

/content/miniconda/envs/mmdet330/bin/python -m pytest \
  tests/test_phase2D1C_mmdet_dataset_loading_guardrails.py -q -rs

Tests cần bao phủ tối thiểu:

- CLI `--help`;
- path resolution;
- expected count validation;
- COCO hash calculation;
- duplicate IDs;
- invalid image/category references;
- missing JPG detection;
- zero-GT identification;
- category-to-label handling;
- `filter_empty_gt=False` bắt buộc;
- phát hiện cấu hình vô tình bật lọc empty GT;
- bbox validation;
- empty bbox shape;
- label validation;
- deterministic empty-sample selection;
- report schema;
- errors CSV header khi không có lỗi;
- failure exit code;
- không có training code;
- không sửa COCO/JPG.

Các pure helper functions nên được tách rõ để unit test không phải load 7.1 GB ảnh trong mọi test.

Ngoài unit tests, script validation thật vẫn phải chạy trên full dataset.

Không được mock `CocoDataset` trong bài kiểm định integration chính.

==================================================
15. QUY TẮC CẬP NHẬT FILE DỰ ÁN
==================================================

Ở lần triển khai đầu tiên:

- ưu tiên tạo script, tests và report;
- không tự ý thay đổi quyết định khoa học;
- không sửa protocol cũ ngoài việc thêm trạng thái Phase 2D.1C nếu cấu trúc file hiện tại thực sự có vị trí phù hợp;
- không đánh dấu phase PASS trước khi có output chạy thật.

Nếu script chưa được chạy thì trạng thái phải là:

- implementation_status: CREATED hoặc READY_FOR_EXECUTION;
- validation_status: NOT_EXECUTED;
- dataset_training_ready: false;
- training_authorized: false.

Chỉ sau khi người dùng chạy script trên Colab và có evidence PASS mới đề xuất patch cập nhật:

- PHASE_HANDOFF.md;
- PROJECT_CONTEXT.md;
- research_log.md;
- CHECKLIST_TRIEN_KHAI_FULL.xlsx;
- protocol/status YAML nếu phù hợp.

Không được giả vờ rằng validation đã chạy chỉ vì code được tạo.

==================================================
16. CHẤT LƯỢNG TRIỂN KHAI
==================================================

Yêu cầu:

- code tương thích Python 3.10;
- type hints hợp lý;
- docstrings;
- không dùng bare `except`;
- thông báo lỗi cụ thể;
- deterministic seed;
- không phụ thuộc notebook state;
- không dùng shell command bên trong Python nếu không cần thiết;
- không gọi pip/conda;
- không tải Internet;
- không yêu cầu GPU cho dataset-loading validation;
- không dùng deprecated API nếu MMDetection 3.3.0 có API chính thức;
- output ngắn gọn nhưng đủ audit;
- không che warning quan trọng;
- tránh O(N²) không cần thiết;
- không giữ toàn bộ ảnh đã decode trong RAM.

==================================================
17. QUY TRÌNH LÀM VIỆC BẮT BUỘC
==================================================

Thực hiện theo thứ tự:

Bước 1 — Inspect

- kiểm tra git status;
- đọc các file bối cảnh;
- inspect COCO JSON;
- inspect cấu trúc repository;
- xác nhận API MMDetection 3.3.0 đang cài;
- nêu các phát hiện và mọi bất nhất.

Bước 2 — Design

Trước khi sửa file, trình bày ngắn gọn:

- file nào sẽ tạo/sửa;
- dataset config sẽ dùng;
- cách chứng minh giữ zero-GT;
- cách tạo forced empty-GT batch;
- report schema;
- tiêu chí PASS/FAIL.

Bước 3 — Implement

- tạo script;
- tạo tests;
- tạo config chỉ khi cần;
- không sửa file ngoài phạm vi.

Bước 4 — Static/unit verification

Chạy tối thiểu:

/content/miniconda/envs/mmdet330/bin/python -m py_compile \
  scripts/02D1C_validate_mmdet_dataset_loading.py

/content/miniconda/envs/mmdet330/bin/python -m pytest \
  tests/test_phase2D1C_mmdet_dataset_loading_guardrails.py -q -rs

Bước 5 — Integration validation

Chạy script thật trên full dataset bằng Python của environment mmdet330.

Bước 6 — Inspect evidence

- đọc lại JSON report;
- kiểm tra CSV;
- xác nhận counts;
- xác nhận zero-GT identity comparison;
- xác nhận exit code;
- xác nhận không có source data bị thay đổi.

Bước 7 — Final response

Báo cáo:

- các file đã tạo/sửa;
- lệnh đã chạy;
- test results;
- observed counts;
- retention comparison;
- empty-GT pipeline result;
- dataloader result;
- report paths;
- git diff/status;
- các blocker hoặc warning;
- kết luận Phase 2D.1C là:
  - NOT EXECUTED;
  - FAIL;
  - hoặc PASS PENDING EXTERNAL REVIEW.

Không commit và không push.

==================================================
18. LỆNH CHẠY DỰ KIẾN
==================================================

Sau khi triển khai, cung cấp cho người dùng đúng một lệnh/cell chính để chạy, theo dạng tương tự:

cd /content/ssl_detection_xray_v2

/content/miniconda/envs/mmdet330/bin/python \
  scripts/02D1C_validate_mmdet_dataset_loading.py \
  --repo-root /content/ssl_detection_xray_v2 \
  --ann-file data/processed/coco/coco_master_jpg.json \
  --data-root data/processed/images_jpg \
  --batch-size 1 \
  --num-workers 0 \
  --seed 42 \
  --expected-images 4894 \
  --expected-annotations 36096 \
  --expected-categories 14 \
  --expected-empty-images 500 \
  --strict

Hãy điều chỉnh tên argument nếu thiết kế cuối cùng cần thiết, nhưng phải cung cấp lệnh chính xác khớp hoàn toàn với CLI đã viết.

==================================================
19. KẾT QUẢ CONSOLE MONG ĐỢI
==================================================

Console summary phải có dạng dễ audit, ví dụ:

PHASE 2D.1C — MMDetection Dataset-Loading Validation
Environment: PASS
Input integrity: PASS
Raw COCO images: 4894
Raw COCO annotations: 36096
Raw categories: 14
Raw zero-GT images: 500
Resolved JPG files: 4894
Missing JPG files: 0
CocoDataset filter_empty_gt=False: 4894
CocoDataset filter_empty_gt=True: 4394
Filtered image IDs: 500
Filtered IDs equal zero-GT IDs: PASS
Abnormal pipeline sample: PASS
Empty-GT pipeline sample: PASS
Empty bbox shape: (0, 4)
Empty label shape: (0,)
Normal dataloader batch: PASS
Forced empty-GT dataloader batch: PASS
Errors: 0
PHASE 2D.1C VALIDATION: PASS
TRAINING AUTHORIZED: FALSE

Đây chỉ là định dạng mong đợi. Không được in PASS bằng hard-code; mọi kết luận phải được suy ra từ kiểm tra thật.

BẮT ĐẦU BẰNG VIỆC INSPECT REPOSITORY. KHÔNG VIẾT CODE TRƯỚC KHI ĐỌC CÁC FILE BỐI CẢNH VÀ KIỂM TRA COCO JSON.