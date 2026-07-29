Bạn là Senior Python Engineer và ML Data Pipeline Engineer.

Tôi đang thực hiện đề tài:

“Nghiên cứu học bán giám sát cho dò tìm bất thường trên X-quang phổi”.

Repository local:

D:\ssl_detection_xray_v2

Phase hiện tại:

Phase 2D.1B-Full —
Full Controlled-Scope DICOM-to-JPG Conversion & Validation.

======================================================================
0. QUY TRÌNH VÀ PHÂN VAI BẮT BUỘC
======================================================================

Quy trình nghiên cứu đã khóa:

GPT thiết kế yêu cầu
→ Claude viết hoặc sửa script trong repository
→ Claude dừng
→ Tôi dùng Python chạy script/test
→ Python sinh output/evidence
→ GPT review code, output và Definition of Done
→ Tôi quyết định và tick checklist.

Vai trò:

- Tôi: người quyết định nghiên cứu và tick checklist.
- GPT: người thiết kế, phản biện và review logic/evidence.
- Claude: chỉ viết hoặc sửa code trong repository.
- Python: công cụ được tôi sử dụng để chạy test, conversion, validation và tạo evidence.

Trong nhiệm vụ này, Claude chỉ được:

1. đọc và kiểm tra repository;
2. xác định các artefact liên quan;
3. viết full conversion orchestrator;
4. viết full-specific guardrail tests;
5. bổ sung `.gitignore` ở mức tối thiểu;
6. kiểm tra tĩnh code vừa viết;
7. báo cáo chính xác file đã tạo/sửa;
8. cung cấp các lệnh để tôi tự chạy sau đó;
9. dừng lại.

Claude tuyệt đối không được:

- chạy pytest;
- chạy script bằng Python;
- chạy `--preflight-only`;
- chạy full conversion;
- chuyển đổi bất kỳ DICOM thật nào;
- sinh test output hoặc preflight evidence chính thức;
- tự sửa code dựa trên kết quả chạy chưa được người dùng cung cấp;
- tự đối chiếu và tuyên bố DoD đạt;
- kết luận implementation ready;
- kết luận preflight PASS;
- kết luận Phase 2D.1B-Full PASS;
- tick hoặc sửa checklist để đánh dấu hoàn thành;
- commit hoặc push Git.

Nếu Claude có cơ chế tự động chạy command sau khi sửa code, không được sử dụng cơ chế đó trong nhiệm vụ này.

======================================================================
I. TRẠNG THÁI VÀ QUYẾT ĐỊNH ĐÃ KHÓA
======================================================================

- Phase 2D.1A: CLOSED / PASS
- Phase 2D.1B-Pilot: CLOSED / PASS
- Pilot implementation: V6_FROZEN
- Pilot guardrail tests: 139/139 PASS trong môi trường chính thức
- JPEG final quality: 95 / LOCKED
- full_conversion_authorized: true
- Phase 2D.1B-Full: OPEN / NOT STARTED
- Phase 2D.1C: LOCKED
- dataset_training_ready: false
- training_authorized: false

Quality 95 được chọn vì đã đạt các tiêu chí kỹ thuật của pilot và tiết kiệm storage/I/O hơn quality 100.

Không được diễn giải rằng:

- quality 95 tốt hơn quality 100 về mAP;
- quality 95 đã được chứng minh tốt hơn cho detector;
- downstream detector ablation đã được thực hiện;
- quality 95 chắc chắn không ảnh hưởng tổn thương nhỏ hoặc lớp hiếm.

Controlled downstream q95-versus-q100 detector ablation:

- chưa được thực hiện;
- không thuộc nhiệm vụ này;
- chưa được xác định là bắt buộc;
- chỉ xem xét nếu được giảng viên xác nhận sau này.

======================================================================
II. ĐỌC REPOSITORY TRƯỚC KHI VIẾT CODE
======================================================================

Đọc và đối chiếu tối thiểu:

- PROJECT_CONTEXT.md
- PHASE_HANDOFF.md
- README.md
- research_log.md
- CHECKLIST_TRIEN_KHAI_FULL.xlsx
- .gitignore
- configs/protocol/phase2D1_jpg_representation.yaml
- reports/phase2D1B_pilot_decision_template.json
- reports/phase2D1B_pilot_decision_template.md
- các reports/phase2D1B_pilot_*
- scripts/02D1B_pilot_dicom_to_jpg.py
- src/utils/dicom_jpg_protocol.py
- tests/test_phase2D1B_pilot_guardrails.py
- reports/phase2D1B_pilot_unit_tests_output_v6.txt
- data/processed/coco/coco_master.json
- manifest controlled scope 4.894 ảnh
- Phase 2A metadata/bbox evidence
- Phase 2B/2D COCO validation evidence

Kiểm tra read-only:

- git status
- git log --oneline -5
- git diff --stat
- git diff --cached --name-only

Trước khi sửa, báo cáo ngắn:

1. artefact đã tìm thấy;
2. artefact không tìm thấy;
3. trạng thái working tree;
4. file modified/untracked đã tồn tại trước nhiệm vụ;
5. bất nhất hoặc blocker;
6. file dự kiến tạo/sửa.

Không xóa, ghi đè hoặc hoàn tác thay đổi hiện có của người dùng.

Nếu trạng thái thực tế khác prompt, ưu tiên evidence trong repository. Không tự thay đổi quyết định khoa học để thích ứng với code.

======================================================================
III. CANONICAL INPUT VÀ INVARIANTS
======================================================================

Canonical COCO master dự kiến:

data/processed/coco/coco_master.json

Trạng thái khóa cần xác minh bằng evidence:

- SHA-256:
  36f09d1b1477ea4a63153a04d775c938752e224c26079a0d44881c14b9bb4d75
- images: 4.894
- annotations: 36.096
- categories: 14
- unique image file names: 4.894
- No Finding images: 500
- No Finding không phải detection category
- source extension trong master: `.dicom`

Không hard-code mù quáng các con số trên như cách duy nhất để code hoạt động.

Implementation phải:

- đọc expected invariants từ canonical evidence thích hợp;
- kiểm tra chúng khớp trạng thái đã khóa;
- hard fail nếu canonical input drift;
- không tự sửa canonical input để vượt validation.

Protocol YAML thuộc Phase 2D.1A đã đóng băng. Nếu YAML vẫn chứa `final_quality: null` hoặc pilot pending thì không được tự sửa YAML.

Nguồn quyết định máy đọc về quality và full authorization là:

reports/phase2D1B_pilot_decision_template.json

Dù tên file chứa “template”, đây là decision artefact chính thức nếu nội dung và fingerprint khớp evidence.

Không dùng Markdown template pending làm nguồn quyết định máy đọc.

Full script phải:

- xác minh version/fingerprint của protocol;
- đọc quality=95 từ decision JSON;
- bắt buộc `full_conversion_authorized=true`;
- không sửa protocol YAML;
- không sửa decision evidence lịch sử.

======================================================================
IV. FILE ĐƯỢC PHÉP TẠO HOẶC SỬA
======================================================================

1. Sửa tối thiểu:

.gitignore

Bổ sung guardrail phù hợp với repository thực tế, ví dụ:

data/processed/images_jpg/
data/processed/images_jpg_staging/
data/processed/images_jpg_backup/
data/processed/images_jpg_failed/
*.part

Không thêm pattern quá rộng có thể che code, reports hoặc canonical metadata.

2. Tạo:

scripts/02D1B_full_dicom_to_jpg.py

3. Tạo:

tests/test_phase2D1B_full_guardrails.py

4. Chỉ khi thật sự cần thiết, tạo helper mới trong:

src/utils/

Ưu tiên tái sử dụng nguyên vẹn:

src/utils/dicom_jpg_protocol.py

Không sửa:

- pilot V6;
- pilot tests;
- protocol YAML;
- decision evidence;
- canonical COCO;
- canonical annotations;
- checklist;
- tài liệu trạng thái phase.

Nếu phát hiện bug blocker buộc phải sửa V6_FROZEN, phải dừng và báo cáo. Không tự sửa core đã đóng băng.

======================================================================
V. CÁC HÀNH ĐỘNG BỊ CẤM TRONG IMPLEMENTATION
======================================================================

Không được thiết kế hoặc thực hiện hành vi:

- sửa DICOM nguồn;
- mở DICOM bằng write/append;
- sửa hoặc ghi đè `coco_master.json`;
- sửa canonical annotation CSV;
- resize, crop, rotate, flip hoặc transpose;
- thay đổi orientation/geometry;
- scale hoặc ngầm clip bbox;
- tạo train/validation/test split;
- tạo labeled/unlabeled subsets;
- training;
- inference;
- pseudo-labeling;
- tính AP/mAP;
- gọi MMDetection để train/inference;
- đặt `dataset_training_ready=true`;
- đặt `training_authorized=true`;
- tự động chạy full conversion sau preflight;
- ghi trực tiếp partial outputs vào final directory;
- commit 4.894 JPG vào ordinary Git;
- tuyên bố q95 tốt hơn q100 về detector.

======================================================================
VI. THIẾT KẾ FULL ORCHESTRATOR
======================================================================

Tạo:

scripts/02D1B_full_dicom_to_jpg.py

Script phải được chia thành các hàm/stage có thể unit test độc lập.

1. `preflight_inputs()`

Kiểm tra:

- protocol và decision JSON tồn tại;
- protocol version/fingerprint đúng;
- decision quality chính xác bằng 95;
- `full_conversion_authorized` chính xác là true;
- COCO master tồn tại và hash đúng;
- đúng controlled scope;
- mapping image_id/file_name/DICOM là one-to-one;
- không duplicate image_id hoặc file_name;
- không thiếu hoặc thừa DICOM theo exact-inventory policy;
- không absolute path hoặc path traversal;
- canonical evidence bắt buộc tồn tại;
- image, annotation, category và No Finding counts đúng.

Mọi lỗi invariant phải hard fail trước pixel decode.

2. `preflight_environment()`

Kiểm tra:

- pydicom;
- NumPy;
- Pillow;
- JPEG encoder/decoder;
- backend decode explicit;
- không silent fallback;
- staging và final nằm cùng filesystem nếu dùng atomic rename;
- dung lượng đĩa cho staging, final, reports, safety margin và backup.

3. `preflight_output_safety()`

Yêu cầu:

- mặc định không overwrite;
- final directory tồn tại và không rỗng thì hard fail;
- conversion chỉ ghi vào staging;
- partial output không bao giờ được xem là valid dataset;
- chặn path traversal, absolute paths và collisions;
- không trộn q95/q100.

Không cần hỗ trợ resume nếu chưa thể bảo đảm an toàn. Hard fail rõ ràng được ưu tiên hơn resume không đáng tin cậy.

4. `convert_one_image()`

Phải:

- gọi lại V6 transformation core;
- dùng backend explicit;
- xử lý modality LUT/rescale;
- xử lý VOI LUT/window/fallback;
- xử lý pixel padding;
- xử lý polarity;
- chuyển uint8 đúng protocol;
- encode duy nhất JPEG quality 95;
- decode lại JPEG để validation;
- giữ nguyên width/height;
- không geometry transformation;
- ghi file tạm rồi atomic replace trong staging;
- trả structured conversion record;
- lưu transformation branches và warnings;
- không nuốt exception.

5. `build_full_mapping()`

Mỗi record tối thiểu:

- image_id;
- source relative path;
- output relative path;
- source DICOM SHA-256;
- pre-JPEG representation hash nếu V6 hỗ trợ;
- output JPEG SHA-256;
- width;
- height;
- JPEG quality;
- protocol version/hash;
- decision artefact hash;
- decoder/backend;
- modality branch;
- VOI/window branch;
- pixel-padding branch;
- polarity branch;
- warnings;
- status.

Mapping dự kiến:

reports/phase2D1B_full_mapping.csv

Có thể thêm JSONL nếu cần cho dữ liệu có cấu trúc.

6. `build_coco_jpg_derivative()`

Derivative dự kiến:

data/processed/coco/coco_master_jpg.json

Quy tắc:

- deep-copy master trong memory;
- chỉ thay `images[].file_name`;
- không sửa master trên đĩa;
- không thay IDs, dimensions, annotations, categories, bbox, area hoặc iscrowd;
- output paths phải relative và unique;
- derivative chỉ được tạo/promote sau full validation.

7. `validate_full_outputs()`

Kiểm tra:

- đủ 4.894 JPG;
- one-to-one với image_id;
- tất cả JPEG decode được;
- quality trong records là 95;
- dimensions khớp canonical;
- không geometry transformation;
- derivative chỉ khác master tại `images[].file_name`;
- annotations/categories/No Finding semantics bất biến;
- bbox và area bất biến;
- bbox nằm trong image boundary;
- traceability đầy đủ;
- canonical hashes không thay đổi;
- không còn temp/partial files.

8. `promote_outputs()`

Yêu cầu transaction-like:

- chỉ promote sau khi toàn bộ conversion và validation PASS;
- một ảnh lỗi thì không promote;
- không để partial dataset trong final;
- không phá valid previous output;
- promotion failure không được claim PASS;
- thiết kế phù hợp Windows;
- không giả định rename xuyên filesystem là atomic.

9. `write_reports()`

Thiết kế khả năng tạo:

- reports/phase2D1B_full_preflight.json
- reports/phase2D1B_full_preflight.md
- reports/phase2D1B_full_mapping.csv
- reports/phase2D1B_full_validation.json
- reports/phase2D1B_full_validation.md
- reports/phase2D1B_full_errors.csv
- reports/phase2D1B_full_metadata_audit.csv
- reports/phase2D1B_full_bbox_audit.csv
- reports/phase2D1B_full_no_finding_audit.csv

Nhưng trong nhiệm vụ viết code này, Claude không được chạy script để sinh các evidence trên.

Report schema phải duy trì:

- phase_status: OPEN hoặc NOT_STARTED;
- dataset_training_ready: false;
- training_authorized: false;
- full_conversion_completed: false;

cho đến khi có full execution và review chính thức.

======================================================================
VII. CLI BẮT BUỘC
======================================================================

Tối thiểu hỗ trợ:

python scripts/02D1B_full_dicom_to_jpg.py --preflight-only

Full execution phải cần explicit opt-in:

python scripts/02D1B_full_dicom_to_jpg.py ^
  --execute-full ^
  --acknowledge-full-scope 4894 ^
  --jpeg-quality 95

Quy tắc:

- không mode mặc định chạy full;
- preflight PASS không tự kích hoạt full;
- quality khác 95 phải hard fail;
- acknowledgement khác controlled scope phải hard fail;
- mode thiếu hoặc không hợp lệ chỉ hiển thị help và thoát;
- mọi failure trả exit code khác 0;
- preflight PASS trả exit code 0.

Claude chỉ viết CLI; không được chạy bất kỳ command nào trên.

======================================================================
VIII. FULL-SPECIFIC GUARDRAIL TESTS
======================================================================

Tạo:

tests/test_phase2D1B_full_guardrails.py

Chỉ dùng synthetic fixtures và temporary directories. Không phụ thuộc vào việc chuyển đổi 4.894 DICOM thật.

Kiểm tra tối thiểu:

A. Decision/protocol/canonical gates

- quality khác 95 → fail;
- authorization khác true → fail;
- thiếu decision JSON → fail;
- Markdown không được dùng thay JSON;
- protocol hash/version drift → fail;
- COCO hash/count drift → fail;
- duplicate image_id/file_name → fail;
- missing/extra DICOM → fail.

B. Source immutability

- source DICOM không mở write/append;
- COCO master không mở write/append;
- canonical annotation không bị sửa;
- hashes trước/sau không đổi;
- failure giữa chừng không sửa source.

C. Geometry và bbox

- width/height bất biến;
- bbox/area/IDs/category_id/iscrowd bất biến;
- derivative chỉ khác `images[].file_name`;
- out-of-bound bbox phải fail, không tự clip;
- phát hiện geometry transformation bị cấm.

D. JPEG/output controls

- encode chỉ q95;
- mixed quality → fail;
- collision, duplicate, missing hoặc corrupt JPEG → fail;
- absolute path/path traversal → fail;
- partial/temp file còn sót → fail.

E. Atomicity/failure handling

- decode/encode/validation failure → không promote;
- report-write failure → không claim PASS;
- promotion failure → không phá valid previous output;
- output có sẵn mà không có explicit policy → fail;
- final không chứa partial dataset.

F. Synthetic mini-scope end-to-end

Fixture phải có:

- vài synthetic DICOM;
- ít nhất một abnormal image có bbox;
- ít nhất một No Finding image không bbox;
- mini COCO master;
- decision fixture;
- protocol fixture và fingerprint phù hợp.

Kiểm tra:

- conversion vào staging;
- q95;
- geometry giữ nguyên;
- mapping đầy đủ;
- derivative chỉ đổi file_name;
- No Finding semantics giữ nguyên;
- validation và promotion hoạt động trong temporary directory.

G. Prohibited-action guardrails

Dùng AST và behavioral assertions phù hợp để bảo vệ khỏi:

- resize/crop/rotate/flip/transpose;
- training/inference/pseudo-labeling;
- split creation;
- AP/mAP computation;
- `dataset_training_ready=true`;
- `training_authorized=true`;
- tuyên bố q95 tốt hơn q100 về detector.

Không sửa hoặc làm yếu:

tests/test_phase2D1B_pilot_guardrails.py

======================================================================
IX. KHÔNG ĐƯỢC CHẠY TEST HOẶC SCRIPT
======================================================================

Sau khi viết code:

- không chạy pytest;
- không chạy import smoke test;
- không chạy compile command;
- không chạy preflight;
- không chạy synthetic end-to-end test;
- không chạy full conversion;
- không tạo output giả;
- không tuyên bố test PASS dựa trên inspection;
- không sửa checklist.

Chỉ cung cấp cho tôi các lệnh để tôi tự chạy:

1. Kiểm tra pilot và full tests:

python -m pytest tests/test_phase2D1B_pilot_guardrails.py tests/test_phase2D1B_full_guardrails.py -v

2. Lưu output test chính thức:

python -m pytest tests/test_phase2D1B_pilot_guardrails.py tests/test_phase2D1B_full_guardrails.py -v > reports/phase2D1B_full_unit_tests_output.txt 2>&1

3. Sau khi GPT review test output và tôi cho phép, chạy preflight:

python scripts/02D1B_full_dicom_to_jpg.py --preflight-only

Không cung cấp lệnh full execution như “bước tiếp theo cần chạy ngay”. Có thể ghi lại lệnh CLI để tham khảo, nhưng phải đánh dấu:

DO NOT RUN — REQUIRES GPT REVIEW AND USER AUTHORIZATION.

======================================================================
X. NGUYÊN TẮC CODE
======================================================================

- Thay đổi nhỏ, rõ ràng, dễ review.
- Không refactor ngoài phạm vi.
- Không xóa evidence lịch sử.
- Không ghi đè thay đổi của người dùng.
- Không broad exception rồi tiếp tục.
- Không silent fallback.
- Không hard-code đường dẫn tuyệt đối của máy.
- Dùng pathlib và tương thích Windows.
- Dùng relative paths trong artefacts.
- JSON serialization ổn định.
- Hash canonical inputs quan trọng.
- Structured logging.
- Mọi failure phải rõ nguyên nhân.
- Không giả định rename xuyên filesystem là atomic.
- Không thay đổi quyết định khoa học trong code.

======================================================================
XI. BÁO CÁO SAU KHI VIẾT CODE
======================================================================

Sau khi hoàn thành phần viết code, trả lời đúng cấu trúc:

1. Repository state before changes
   - branch;
   - HEAD;
   - modified/untracked files đã tồn tại;
   - bất nhất phát hiện.

2. Files changed
   - file tạo mới;
   - file sửa;
   - mục đích từng file;
   - xác nhận file nào cố ý không sửa.

3. Implementation summary
   - input gates;
   - environment/output preflight;
   - conversion;
   - validation;
   - staging/promotion;
   - source immutability;
   - COCO derivative semantics.

4. Test coverage written
   - nhóm test đã viết;
   - synthetic fixtures;
   - các failure paths đã bao phủ.

5. Commands for the user to run
   - pytest command;
   - command lưu pytest evidence;
   - preflight-only command;
   - full command phải ghi rõ DO NOT RUN.

6. Static limitations or blockers
   - artefact bị thiếu;
   - giả định cần GPT review;
   - điểm chưa thể xác minh nếu chưa chạy Python.

7. Prohibited-action confirmation
   - pytest executed: false;
   - Python script executed: false;
   - preflight executed: false;
   - full conversion executed: false;
   - full evidence generated: false;
   - source DICOM modified: false;
   - canonical COCO modified: false;
   - checklist modified: false;
   - commit/push performed: false.

8. Handoff status

Chỉ được kết luận:

CODE_WRITTEN — AWAITING USER PYTHON EXECUTION AND GPT REVIEW

Không được kết luận:

- TESTS_PASS;
- PREFLIGHT_PASS;
- READY_FOR_FULL_EXECUTION;
- DoD PASS;
- Phase 2D.1B-Full PASS.

======================================================================
XII. ĐIỀU KIỆN DỪNG
======================================================================

Dừng ngay và báo blocker nếu:

- cần sửa V6_FROZEN;
- canonical evidence mâu thuẫn với quyết định đã khóa;
- không xác định được canonical input;
- thay đổi cần thiết nằm ngoài file được phép sửa;
- repository có thay đổi người dùng chồng lấn với file cần sửa;
- implementation buộc phải thay đổi protocol khoa học;
- cần chạy dữ liệu để quyết định thiết kế nhưng chưa có GPT review.

BẮT ĐẦU BẰNG VIỆC ĐỌC REPOSITORY VÀ BÁO CÁO NGẮN TRẠNG THÁI TRƯỚC KHI SỬA.

Sau đó viết code theo yêu cầu và dừng ở trạng thái:

CODE_WRITTEN — AWAITING USER PYTHON EXECUTION AND GPT REVIEW