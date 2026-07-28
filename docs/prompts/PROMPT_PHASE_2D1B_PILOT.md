<!--
Updated: 2026-07-28
Status: Mentor-approved implementation and technical validation scope; critical technical corrections integrated
Official pipeline: DICOM metadata-aware, standard-aligned reference representation pipeline
-->

Tôi đang tiếp tục repo local:

```text
D:\ssl_detection_xray_v2
```

Đề tài đã khóa:

**“Nghiên cứu học bán giám sát cho dò tìm bất thường trên X-quang phổi”.**

Hãy viết code cho:

```text
Phase 2D.1B-Pilot — Representative DICOM-to-JPG Pilot
Environment: Local
Status: OPEN / CURRENT
```

# 0. Phase purpose, mentor approval, and locked scope

## 0.0 Critical technical corrections integrated

This prompt supersedes the prior Phase 2D.1B implementation prompt for
code-generation purposes.

The following corrections are now binding:

```text
actual nested YAML schema paths are mandatory
protocol version path = protocol_metadata.protocol_version
SOPClassUID and Modality must be inventoried
PresentationLUTSequence must be detected
metadata presentation conflicts must block for protocol review
patient-space orientation must not be overclaimed
exact pre-JPEG uint8 references must be persisted as lossless PNG
synthetic expected values must be independently defined
pending decision templates must be generated
CLI/environment root resolution behavior is explicit
```

These corrections do not authorize changing the locked protocol YAML,
incrementing protocol version, selecting a final JPEG quality, or starting
full conversion.


## 0.1 Phase 2D.1B-Pilot purpose

The purpose of this phase is to run a representative, failure-seeking
DICOM-to-JPG pilot before any full-dataset conversion.

This phase must answer the following technical questions:

```text
1. Can the locked DICOM metadata-aware, standard-aligned reference
   representation protocol be implemented deterministically?

2. Are the observed DICOM metadata branches handled explicitly and
   traceably, including pixel padding, modality transformation,
   VOI/windowing, and presentation/polarity normalization?

3. Are original image dimensions, orientation, geometry, and canonical
   bounding-box coordinates preserved exactly?

4. What measurable distortion is introduced by JPEG quality 95 and
   quality 100 relative to the same pre-JPEG uint8 reference image,
   at whole-image and bbox-ROI levels?

5. Are outputs reproducible in the same recorded software environment?

6. Do real or synthetic cases expose a structural implementation error,
   an unresolved protocol gap, or a need for expert visual review?
```

The pilot is representative and coverage-first, not a random statistical
sample and not a downstream model experiment.

The pilot must inspect headers for the locked 4,894-image controlled scope,
but pixel decoding is permitted only for the deterministically selected
pilot images.

The outputs of this phase are technical evidence for GPT/researcher review.
This phase does not:

```text
run full conversion
select the final JPEG quality automatically
create the final JPG training dataset
create coco_master_jpg.json
load MMDetection
train a detector
generate pseudo-labels
compute AP or mAP
claim downstream superiority
authorize training
```

## 0.2 Mentor approval status

The supervisor has approved implementation and technical validation of the:

```text
DICOM metadata-aware, standard-aligned reference representation pipeline
```

Record the following project state:

```text
mentor_approval_status:
approved

pipeline_implementation_authorized:
true

technical_validation_authorized:
true
```

The approved scope includes:

```text
metadata-based DICOM transformation decisions
stored-pixel padding handling
modality transformation
VOI LUT/windowing
presentation/polarity normalization
deterministic grayscale uint8 reference generation
no crop and no resize
geometry and bounding-box coordinate preservation
paired JPEG quality 95 and quality 100 generation
pre-JPEG versus decoded-JPEG fidelity comparison
technical validation
same-environment reproducibility validation
```

The supervisor approval does not automatically authorize:

```text
full conversion of all 4,894 images
automatic final JPEG-quality selection
creation of the final JPG training dataset
creation of coco_master_jpg.json
MMDetection dataset loading
training
downstream superiority claims
controlled downstream preprocessing ablation
```

Controlled downstream ablation remains pending separate mentor
confirmation. Do not infer that it is required, unnecessary, or authorized.

# 0.3 Official pipeline name and scientific positioning

Official pipeline name:

```text
DICOM metadata-aware, standard-aligned reference representation pipeline
```

Scientific positioning:

```text
This is a reference representation pipeline, not a novel preprocessing algorithm.

It is metadata-aware because transformation branches are selected from
relevant DICOM metadata.

It is standard-aligned because its sequence follows the DICOM grayscale
transformation model, but the project does not claim complete formal
DICOM Standard conformance.

Phase 2D.1B evaluates implementation correctness, metadata-branch
traceability, geometry and bbox-coordinate preservation,
same-environment reproducibility, and JPEG quality 95 versus quality 100
fidelity.

Phase 2D.1B does not evaluate downstream detection superiority,
AP/mAP improvement, semi-supervised learning performance, or
preprocessing-method superiority.

Controlled downstream ablation remains pending mentor confirmation.
It is not authorized and is not part of the Phase 2D.1B Definition of Done.
```

Scientific basis to record in the report:

```text
technical_basis:
DICOM grayscale transformation sequence

applied_research_precedent:
Cheng et al. (2024)

research_precedent_scope:
metadata_use_precedent_not_algorithm_replication
```

Authoritative technical reference:

```text
DICOM Standard — Grayscale Transformations:
https://dicom.nema.org/medical/dicom/current/output/chtml/part04/sect_n.2.html
```

Applied CXR object-detection precedent:

```text
Cheng et al. (2024), Diagnostics 14(23):2636:
https://www.mdpi.com/2075-4418/14/23/2636
```

Cheng et al. is used only as an applied research precedent showing that
DICOM metadata such as Window Center, Window Width, and BitsStored can
inform CXR image preparation. This pilot must not reproduce Cheng
logarithmic transformation or simplest color balance.

Khalili rib-guided projection-based augmentation is related work for
anatomical and cross-dataset augmentation. It is not the technical
foundation of this Phase 2D.1B representation pipeline.

Claude chỉ chịu trách nhiệm viết code và unit tests theo protocol đã khóa. Claude không được tự thay đổi quyết định nghiên cứu, không được tự kết luận pilot PASS, không được tự chọn final JPEG quality và không được mở full conversion.

Quy trình bắt buộc của dự án:

```text
script → output → DoD → GPT review → người nghiên cứu tick checklist
```

---

# 1. Trạng thái dự án

```text
Phase 0 core: PASS
Phase 1A: PASS
Phase 1B: PASS
Phase 1C: PASS
Phase 1D: PASS
Phase 2A: PASS
Phase 2B: PASS
Phase 2C: PASS
Phase 2D: CLOSED / PASS
Phase 2D.1A: CLOSED / PASS

Phase 2D.1B-Pilot: OPEN / CURRENT
Phase 2D.1B-Full: LOCKED
Phase 2D.1C: LOCKED
Phase 2D.1D: LOCKED
```

Các readiness flags phải tiếp tục giữ nguyên:

```text
jpg_training_representation_ready: false
coco_jpg_training_annotation_ready: false
mmdetection_dataset_loading_ready: false
empty_image_retention_ready: false
dataset_training_ready: false
training_authorized: false
```

Controlled working scope đã khóa:

```text
images: 4894
abnormal_images: 4394
no_finding_images: 500
annotations: 36096
categories: 14
no_finding_annotations: 0
```

`No Finding` là negative image không có bbox, không phải detection class.

---

# 2. Đọc các file điều phối trước khi code

Đọc theo thứ tự:

```text
PROJECT_CONTEXT.md
PHASE_HANDOFF.md
research_log.md
CHECKLIST_TRIEN_KHAI_FULL.xlsx
```

Các file này chỉ dùng để hiểu trạng thái và guardrails.

Không sửa các file điều phối trên trong nhiệm vụ viết code này.

Đọc các artifact Phase 2D.1A:

```text
configs/protocol/phase2D1_jpg_representation.yaml
reports/phase2D1_image_representation_decision.md
reports/phase2D1_image_representation_decision.json
scripts/02D1A_image_representation_protocol.py
tests/test_phase2D1A_protocol_guardrails.py
```

Protocol Phase 2D.1A:

```text
protocol_version: 1.0.0
protocol_sha256:
1528da27758d35786847141c37d0ddb754dddb146aff116a8f3a9a7b07221229
```

Không sửa các artifact Phase 2D.1A.

Không cập nhật `final_quality` trong YAML hoặc JSON Phase 2D.1A.

Trường `gpt_review_status` cũ trong artifact Phase 2D.1A không được dùng để block pilot. Trạng thái review đã được quản lý ở handoff và research log. Script pilot chỉ validate:

```text
protocol version
protocol fingerprint
locked transformation policies
quality candidates
forbidden actions
locked counts
```

---

# 3. Input chính thức

Sử dụng các file trong repo với đúng canonical path:

```text
configs/protocol/phase2D1_jpg_representation.yaml

data/processed/coco/coco_master.json

data/processed/canonical/canonical_bbox_table.csv
data/processed/canonical/canonical_class_mapping.csv

reports/phase2A_image_metadata.csv
reports/phase2A_dicom_bbox_validation.json
reports/phase2D_coco_master_validation.json
```

Không sử dụng tên file upload có hậu tố như `(1)` hoặc `(2)` trong code. Trong repo phải dùng tên canonical ở trên.

## DICOM root chính thức

DICOM thật nằm tại:

```text
D:\ssl_detection_xray\data\raw\vinbigdata\dicom_subset
```

Bắt buộc lấy DICOM root từ biến môi trường:

```text
VINBIGDATA_DICOM_ROOT
```

Giá trị đúng:

```text
D:\ssl_detection_xray\data\raw\vinbigdata\dicom_subset
```

`coco_master.json` chứa `file_name` dạng:

```text
train/<image_id>.dicom
```

Cách resolve bắt buộc:

```python
dicom_path = Path(os.environ["VINBIGDATA_DICOM_ROOT"]) / coco_file_name
```

Ví dụ:

```text
VINBIGDATA_DICOM_ROOT
= D:\ssl_detection_xray\data\raw\vinbigdata\dicom_subset

COCO file_name
= train/0005e8e3701dfb1dd93d53e2ff537b6e.dicom

resolved path
= D:\ssl_detection_xray\data\raw\vinbigdata\dicom_subset
  \train\0005e8e3701dfb1dd93d53e2ff537b6e.dicom
```

Không hard-code absolute DICOM path trong source code.

Không sử dụng cột `resolved_path` trong `phase2A_image_metadata.csv` làm nguồn path chính. Cột này chỉ dùng làm historical evidence và cross-check.

Có thể hỗ trợ CLI override:

```text
--dicom-root
```

Quy tắc resolution bắt buộc:

```text
CLI absent  + ENV present
→ use ENV

CLI present + ENV absent
→ use CLI

CLI present + ENV present + same resolved path
→ use that path

CLI present + ENV present + different resolved paths
→ hard fail

CLI absent  + ENV absent
→ hard fail
```

Path equivalence phải được đánh giá sau khi normalize và resolve theo hệ điều hành hiện tại.

Không được mô tả đơn giản rằng CLI luôn override ENV nếu cả hai cùng tồn tại. Nếu cả hai cùng tồn tại nhưng khác nhau, phải hard fail để tránh chạy nhầm dataset.

Không ghi absolute DICOM root vào portable CSV mapping.

---

# 4. Lưu ý về class ID

Không được giả định mọi class ID đều giống nhau.

Canonical mapping:

```text
canonical_class_id: 0..13
class_id_original: 0..13
```

COCO mapping:

```text
COCO category_id: 1..14
```

Trong `coco_master.json`, mỗi category có:

```text
id
name
canonical_class_id
class_id_original
```

Phải validate mapping theo các field trên.

Không dùng công thức `category_id == canonical_class_id`.

Quan hệ dự kiến hiện tại là:

```text
category_id = canonical_class_id + 1
```

nhưng script phải đối chiếu bằng metadata, không hard-code quan hệ mà không validate.

Class coverage pilot phải được tính theo:

```text
unique image_id × class presence
```

Không dùng annotation-row count làm trọng số coverage, vì một ảnh có thể có nhiều bbox từ nhiều radiologist.

---

# 5. File cần tạo

Tạo:

```text
scripts/02D1B_pilot_dicom_to_jpg.py
tests/test_phase2D1B_pilot_guardrails.py
```

Có thể tạo module utility riêng nếu thật sự cần, ví dụ:

```text
src/utils/dicom_jpg_protocol.py
```

nhưng không phân tán logic thành quá nhiều file không cần thiết.

Không tạo script full conversion.

Không tạo:

```text
scripts/02D1B_full_dicom_to_jpg.py
```

---

# 6. Hành động bị cấm

Code và tests phải bảo đảm không thực hiện:

```text
Full conversion 4894 DICOM pixels
Full JPG dataset creation
Creation of data/processed/images_jpg/train
Creation of coco_master_jpg.json
Modification of coco_master.json
Modification of canonical_bbox_table.csv
Modification of canonical_class_mapping.csv
Train/validation/test split
Labeled/unlabeled split
Training
Detector inference
Pseudo-label generation
Threshold tuning
AP/mAP computation
Test-set usage
Model/backbone selection
Checkpoint selection
MMDetection dataset loading
Changing final JPEG quality
Setting any readiness flag true
Setting training_authorized true
Automatic bbox scaling
Resize
Crop
Rotation
Flip
Transpose
Direct observed per-image min-max normalization
Automatic percentile clipping
RGB conversion or three-channel replication during master image conversion
Silent decoder fallback
Dumping complete DICOM metadata
Exporting PHI fields
```

Header-only inventory của 4.894 DICOM được phép.

Pixel decoding chỉ được phép trên các ảnh đã được deterministic pilot selection chọn.

## 6.1 Preexisting forbidden-artifact guardrail

Before execution, record a filesystem snapshot for all forbidden paths,
including at minimum:

```text
data/processed/images_jpg/train
data/processed/coco/coco_master_jpg.json
scripts/02D1B_full_dicom_to_jpg.py
```

If a forbidden full-conversion artifact already exists before the pilot:

```text
- do not delete it
- do not modify it
- report preexisting_forbidden_artifact = true
- record the path, type, size, modification time, and hash when applicable
- hard fail before any pixel decoding
```

The guardrail and evidence must distinguish:

```text
created_by_current_run
modified_by_current_run
preexisting_before_current_run
```

The pilot must never delete a preexisting forbidden artifact in order to
make the workspace appear compliant.

---

# 7. Script architecture

Script nên có các stage rõ ràng:

```text
1. protocol_preflight
2. input_crosscheck
3. decoder_preflight
4. header_inventory
5. metadata_strata_construction
6. coverage_universe_construction
7. deterministic_pilot_selection
8. selected_pixel_decoding
9. dicom_to_uint8_transformation
10. paired_jpeg_encoding
11. jpeg_fidelity_metrics
12. bbox_roi_metrics
13. geometry_validation
14. visual_evidence_generation
15. evidence_validation
16. atomic_output_promotion
```

Mỗi stage phải là function riêng, có thể unit test.

Không để toàn bộ logic trong một hàm `main()` dài.

Sử dụng:

```text
Python 3.10 compatible
pathlib
type hints
dataclasses khi phù hợp
logging
clear custom exceptions
stable sorting
strict JSON
atomic writes
UTF-8 CSV
```

---

# 8. Protocol preflight

Strict-load:

```text
configs/protocol/phase2D1_jpg_representation.yaml
```

## 8.0 Locked YAML schema paths

The protocol validator must use the actual nested YAML schema.

At minimum:

```text
protocol version must be read from:
protocol_metadata.protocol_version
```

Do not assume that protocol fields are top-level keys.

For every locked field, reuse the same canonical field-path resolution
used by Phase 2D.1A or define an explicit immutable field-path map after
inspecting the locked YAML.

The field-path map must be declared centrally, must be deterministic, and
must not silently search for similarly named fields.

If an expected nested path is missing:

```text
hard fail
protocol_schema_mismatch
do not search for a similarly named field elsewhere
do not infer or repair the YAML
do not continue to pilot selection or pixel decoding
```

The script must record the exact resolved YAML field path for each locked
value used during validation.

Validate:

```text
protocol_metadata.protocol_version == 1.0.0
quality_candidates == [95, 100]
final_quality is null
final_quality_status == pending_phase2D1B_pilot
resize == false
crop == false
rotation == false
flip == false
transpose == false
bbox_scaling_expected == false
direct observed per-image min-max == forbidden
automatic percentile clipping == forbidden
```

Tính canonical protocol SHA-256 bằng cùng quy tắc với Phase 2D.1A:

```python
canonical_json = json.dumps(
    protocol_dict,
    sort_keys=True,
    ensure_ascii=False,
    separators=(",", ":"),
)
protocol_sha256 = hashlib.sha256(
    canonical_json.encode("utf-8")
).hexdigest()
```

Fingerprint phải bằng:

```text
1528da27758d35786847141c37d0ddb754dddb146aff116a8f3a9a7b07221229
```

Nếu mismatch:

```text
hard fail
protocol_drift_detected
không tạo pilot JPG
```

## 8.1 Protocol-gap policy

The implementation must not invent transformation behavior for a case
that is not explicitly resolved by the locked protocol YAML or this
implementation specification.

If a real or synthetic input exposes an unresolved decision, report:

```text
protocol_gap_detected = true
protocol_review_required = true
phase_status = BLOCKED_PROTOCOL_REVIEW
structural_dod_candidate = false
```

Do not:

```text
silently choose a fallback
infer a policy from visual appearance
modify the YAML
modify Phase 2D.1A artifacts
increment the protocol version automatically
reinterpret the DICOM Standard to add an unapproved project policy
```

A protocol gap must be resolved by a separate research decision before
pixel conversion resumes.

---

# 9. Input cross-check

Cross-check các nguồn:

```text
coco_master.json
canonical_bbox_table.csv
canonical_class_mapping.csv
phase2A_image_metadata.csv
phase2A_dicom_bbox_validation.json
phase2D_coco_master_validation.json
```

Xác nhận:

```text
COCO images = 4894
COCO annotations = 36096
COCO categories = 14
abnormal images = 4394
No Finding images = 500
No Finding annotations = 0
canonical bbox rows = 36096
invalid bbox = 0
boundary violations = 0
```

Validate SHA-256 của `coco_master.json` với:

```text
reports/phase2D_coco_master_validation.json
```

Nếu hash khác evidence đã khóa:

```text
hard fail
coco_master_drift_detected
```

Validate:

```text
all image IDs unique
all original_image_id unique
all COCO file_name unique
all file_name relative
no absolute path
all file_name start with train/
all file_name end with .dicom
all COCO category IDs unique and contiguous 1..14
canonical class IDs unique and contiguous 0..13
No Finding absent from categories
background absent from categories
```

Không sửa dữ liệu khi phát hiện mismatch.

---

# 10. DICOM root validation

Lấy root từ CLI hoặc environment variable theo policy ở trên.

Resolve toàn bộ 4.894 paths bằng:

```python
resolved_path = dicom_root / Path(coco_file_name)
```

Validate:

```text
resolved path stays under dicom_root
4894/4894 files exist
no duplicate resolved path
file stem matches original_image_id
extension is .dicom
```

Không scan hoặc đưa thêm DICOM ngoài controlled scope vào pilot universe.

Không dựa vào tất cả file bất kỳ đang có trong thư mục. Controlled scope phải lấy từ `coco_master.json`.

Có thể scan inventory để phát hiện file thừa, nhưng file thừa chỉ được ghi là informational; không được đưa vào pilot.

---

# 11. Decoder preflight

Ghi environment:

```text
Python version
platform
numpy version
pydicom version
Pillow version
Pillow JPEG/libjpeg version
scikit-image version
PyYAML version
matplotlib version
pylibjpeg version nếu có
pylibjpeg-openjpeg version nếu có
gdcm version nếu có
available pydicom pixel handlers
```

Các Transfer Syntax đã biết có thể gồm:

```text
1.2.840.10008.1.2
1.2.840.10008.1.2.1
1.2.840.10008.1.2.4.90
```

Phải đọc inventory thật và không giới hạn bằng danh sách trên.

Đối với uncompressed transfer syntax:

```text
dùng pydicom native decoder
```

Đối với JPEG 2000 Lossless:

```text
phải chọn decoder backend rõ ràng
không silent fallback
```

Thiết kế CLI:

```text
--jpeg2000-decoder
```

Default:

```text
pylibjpeg
```

Có thể cho phép các lựa chọn rõ ràng nếu pydicom hỗ trợ:

```text
pylibjpeg
gdcm
pillow
```

Nhưng script chỉ dùng đúng backend đã chọn.

Nếu backend được yêu cầu không khả dụng:

```text
hard fail trước pixel decoding
```

Không tự động đổi sang backend khác.

Ghi decoder backend thực tế cho từng pilot image.

---

# 12. Header inventory toàn controlled scope

Được phép đọc header của toàn bộ 4.894 DICOM bằng:

```python
pydicom.dcmread(
    path,
    stop_before_pixels=True,
    force=False,
)
```

Không đọc `pixel_array` trong stage inventory.

Chỉ xuất allowlisted metadata:

```text
image_id
coco_image_id
canonical_image_id
dicom_relative_path
SOPClassUID
Modality
Rows
Columns
PhotometricInterpretation
TransferSyntaxUID
BitsAllocated
BitsStored
HighBit
PixelRepresentation
SamplesPerPixel
NumberOfFrames_raw
NumberOfFrames_effective
RescaleSlope
RescaleIntercept
rescale_slope_present
rescale_intercept_present
modality_lut_present
modality_lut_count
voi_lut_present
voi_lut_count
WindowCenter_all
WindowWidth_all
window_center_count
window_width_count
window_is_multivalued
VOILUTFunction
PresentationLUTShape
presentation_lut_sequence_present
presentation_lut_sequence_count
PixelPaddingValue
PixelPaddingRangeLimit
pixel_padding_value_present
pixel_padding_range_present
```

Không xuất các field nhận diện người bệnh hoặc cơ sở y tế, bao gồm nhưng không giới hạn:

```text
PatientName
PatientID
PatientBirthDate
PatientSex
InstitutionName
InstitutionAddress
AccessionNumber
StudyDescription
SeriesDescription
ReferringPhysicianName
full DICOM dump
private tags
```

Không cần các field này cho nghiên cứu representation.

Validate header:

```text
SamplesPerPixel == 1
PhotometricInterpretation in {MONOCHROME1, MONOCHROME2}
NumberOfFrames_effective == 1
Rows > 0
Columns > 0
BitsAllocated > 0
BitsStored > 0
BitsStored <= BitsAllocated
0 <= HighBit < BitsAllocated
PixelRepresentation in {0, 1}
Rows == COCO height
Columns == COCO width
Rows == phase2A image_height
Columns == phase2A image_width
```

`NumberOfFrames` absent được ghi:

```text
NumberOfFrames_raw = null
NumberOfFrames_effective = 1
```

Nếu `NumberOfFrames > 1`:

```text
unsupported_input
hard fail
```

Nếu `HighBit != BitsStored - 1`, không tự sửa metadata. Ghi thành một stratum riêng và warning để review, trừ khi metadata không thể giải mã hợp lệ.

Nếu `PresentationLUTSequence` tồn tại trong bất kỳ DICOM nào nhưng locked
protocol hiện chỉ định nghĩa `PresentationLUTShape`:

```text
protocol_gap_detected = true
protocol_review_required = true
phase_status = BLOCKED_PROTOCOL_REVIEW
structural_dod_candidate = false
```

Do not silently ignore `PresentationLUTSequence`.

Do not replace `PresentationLUTSequence` with `PresentationLUTShape`
behavior.

Do not decode pilot pixels after this unresolved case is identified.

---

# 13. Metadata strata

Xây dựng strata từ toàn bộ header inventory.

Không tạo một stratum riêng cho từng giá trị liên tục của:

```text
WindowCenter
WindowWidth
RescaleSlope
RescaleIntercept
```

Nếu làm vậy pilot có thể phình gần full dataset.

Exact categorical strata:

```text
SOPClassUID
Modality
PhotometricInterpretation
TransferSyntaxUID
BitsAllocated
BitsStored
HighBit
PixelRepresentation
SamplesPerPixel
NumberOfFrames_effective
PresentationLUTShape including absent
PresentationLUTSequence present/absent
VOILUTFunction including absent
```

Pattern strata:

```text
modality_lut_present
modality_lut_absent

rescale_absent
rescale_identity
rescale_non_identity
rescale_incomplete_invalid

voi_lut_present
voi_lut_absent

window_absent
window_single_valid
window_multi_valid
window_incomplete_or_invalid

padding_absent
padding_single_value
padding_range
```

Định nghĩa:

```text
rescale_absent:
slope absent and intercept absent

rescale_identity:
slope == 1 and intercept == 0

rescale_non_identity:
both present and not identity

rescale_incomplete_invalid:
only one of slope/intercept present

window_absent:
both center and width absent

window_single_valid:
one valid center-width pair

window_multi_valid:
more than one valid center-width pair

window_incomplete_or_invalid:
only one field exists, cardinality mismatch,
non-numeric value, or invalid width
```

Mọi unsupported/ambiguous metadata phải được ghi vào errors evidence.

Không được bỏ qua stratum hiếm vì số lượng nhỏ.

---

# 14. Coverage universe từ COCO và canonical tables

Pilot selection unit:

```text
image_id / original_image_id
```

Tạo image-level class presence từ canonical bbox table hoặc COCO annotations.

Mỗi ảnh có thể cover nhiều class.

Coverage bắt buộc:

```text
all 14 abnormal classes
No Finding images
all observed metadata strata
minimum width image
maximum width image
minimum height image
maximum height image
minimum pixel-count image
maximum pixel-count image
smallest absolute bbox
largest absolute bbox
smallest relative bbox
largest relative bbox
```

Relative bbox area:

```text
bbox_area / (image_width * image_height)
```

Lưu cả absolute bbox extrema và relative bbox extrema vì tổn thương nhỏ theo tỷ lệ ảnh có ý nghĩa hơn chỉ nhìn pixel area.

Không fuse, delete, round, clamp hoặc deduplicate bbox.

Near-duplicate bbox vẫn được giữ nguyên.

---

# 15. Deterministic coverage-first pilot selection

Policy khóa:

```text
minimum_images = 64
minimum_no_finding_images = 16
tie_break_seed = 2026
selection_unit = image_id
selection = deterministic_coverage_first
```

Thêm execution safety guardrail:

```text
max_pilot_images = 256
```

Đây chỉ là guardrail chống accidental near-full conversion, không phải scientific fidelity threshold và không thay đổi protocol.

Tie-break bắt buộc:

```python
tie_break_rank = hashlib.sha256(
    f"2026|{image_id}".encode("utf-8")
).hexdigest()
```

Không dùng:

```text
Python built-in hash()
unordered set iteration
filesystem order
random.choice trên unordered input
```

Thuật toán:

```text
1. Bắt đầu bằng union của mandatory extrema image IDs.
2. Tính toàn bộ feature chưa được cover.
3. Greedy chọn image cover nhiều uncovered features nhất.
4. Tie-break bằng deterministic SHA-256 rank.
5. Tiếp tục đến khi cover 14/14 abnormal classes.
6. Bảo đảm tối thiểu 16 No Finding images.
7. Khi chọn thêm No Finding, ưu tiên ảnh cover metadata strata chưa cover.
8. Nếu số ảnh vẫn dưới 64, fill theo stable SHA rank.
9. Nếu còn metadata strata chưa cover, tiếp tục mở rộng quá 64.
10. Validate lại toàn bộ coverage sau selection.
```

Nếu phải vượt 256 ảnh:

```text
phase_status = BLOCKED_PROTOCOL_REVIEW
hard_error = pilot_scope_explosion
```

Không decode pixel và không tự chuyển thành full conversion.

Chạy selection hai lần với cùng input phải cho:

```text
cùng image IDs
cùng selection order
cùng selected_for_features
cùng tie-break ranks
```

---

# 16. Pixel decoding

Chỉ gọi pixel decoder cho selected pilot image IDs.

Tạo một explicit set:

```text
selected_image_ids
```

Trước mỗi pixel decode phải assert:

```text
image_id in selected_image_ids
```

Theo dõi:

```text
pixel_decode_attempt_count
pixel_decode_success_count
pixel_decode_error_count
unique_pixel_decoded_image_count
```

Validate:

```text
unique_pixel_decoded_image_count == pilot_selected_image_count
unique_pixel_decoded_image_count < 4894
```

Nếu bằng 4.894:

```text
hard fail
accidental_full_conversion_detected
```

Không dùng:

```text
force=True
resize
crop
rotate
flip
transpose
```

---

# 17. DICOM-to-uint8 transformation

Thứ tự chính thức:

```text
DICOM decode
→ stored-pixel padding mask
→ modality transformation
→ VOI LUT/windowing
→ presentation-polarity normalization
→ deterministic uint8 conversion
→ reapply padding as 0
```

Không được đổi thứ tự.

## 17.1 Stored pixels

Validate decoded array:

```text
2-dimensional
shape == (Rows, Columns)
finite after transformation
single frame
single sample
```

Không sử dụng observed array min/max để xác định output mapping range.

Observed min/max có thể được ghi như descriptive diagnostics nếu cần, nhưng tuyệt đối không được dùng để đặt clipping bounds.

## 17.2 Pixel padding

Tạo padding mask trên stored pixel array trước modality transformation.

Nếu chỉ có `PixelPaddingValue`:

```text
mask = stored_pixels == PixelPaddingValue
```

Nếu có cả `PixelPaddingValue` và `PixelPaddingRangeLimit`:

```text
low = min(value, range_limit)
high = max(value, range_limit)
mask = (stored_pixels >= low) & (stored_pixels <= high)
```

Range là inclusive.

Nếu `PixelPaddingRangeLimit` tồn tại nhưng `PixelPaddingValue` không tồn tại:

```text
hard fail
ambiguous_padding_metadata
```

Ghi:

```text
padding_pixel_count
padding_fraction
padding_value
padding_range_limit
```

Padding không được ảnh hưởng tới observed statistics nếu statistics được sử dụng để audit.

Sau toàn bộ polarity và uint8 conversion:

```text
uint8_image[padding_mask] = 0
```

Không đặt padding thành 0 trước inversion rồi bỏ qua việc reapply, vì có thể khiến padding trở thành trắng.

## 17.3 Modality branch

Branch bắt buộc:

```text
if ModalityLUTSequence exists:
    apply Modality LUT only

elif both RescaleSlope and RescaleIntercept exist:
    apply rescale only

elif neither exists:
    identity

else:
    hard fail
```

Không áp dụng Modality LUT và rescale tuần tự.

Rationale phải được ghi rõ trong code comments, tests, và report:

```text
Modality LUT, Rescale Slope/Intercept, và Identity là các lựa chọn thay thế
để thực hiện cùng một modality-transformation stage. Chúng không phải là
ba bước cộng dồn.

Applying more than one branch would double-transform stored pixel values
and could corrupt the modality-domain intensity range before VOI processing.
Exactly one modality branch must be selected for each image.
```

Nếu Modality LUT cùng tồn tại với rescale metadata:

```text
apply Modality LUT
record rescale metadata as present_not_applied
do not apply rescale
```

Theoretical stored range:

```text
PixelRepresentation == 0:
0 .. 2**BitsStored - 1

PixelRepresentation == 1:
-2**(BitsStored - 1)
..
2**(BitsStored - 1) - 1
```

Đối với rescale:

```text
transform cả hai theoretical endpoints
sort lại low/high
```

Việc sort lại bắt buộc để hỗ trợ negative slope.

Đối với Modality LUT:

```text
derive output bounds từ LUT Descriptor/LUT data definition
không dùng transformed_array.min()/max()
```

Ghi:

```text
modality_branch
theoretical_stored_low
theoretical_stored_high
theoretical_modality_low
theoretical_modality_high
```

## 17.4 VOI branch

Branch bắt buộc:

```text
if VOILUTSequence exists:
    use VOI LUT index 0

elif valid WindowCenter and WindowWidth exist:
    use window index 0

else:
    use theoretical modality-domain fallback
```

Nếu VOI LUT tồn tại, không áp dụng thêm windowing.

Ghi tất cả WindowCenter và WindowWidth values trước khi chọn index 0.

Validate:

```text
numeric values
matching cardinality
WindowWidth valid
```

Respect:

```text
VOILUTFunction
```

Không tự chọn window khác sau khi quan sát hình ảnh.

Nếu index 0 tạo structural failure, ví dụ WindowWidth không hợp lệ,
cardinality mismatch, VOI LUT lỗi, output không tính được hoặc output range
degenerate, phải ghi `phase_status = BLOCKED` và
`protocol_review_required = true`.

Nếu index 0 chỉ visually suspicious hoặc không nhất quán với reference
viewer nhưng chưa xác lập lỗi cấu trúc, phải giữ:

```text
phase_status = OPEN_REVIEW_REQUIRED
protocol_review_required = true
visual_review_status = PENDING_EXPERT_REVIEW
critical_visual_failure = null
```

Script và GPT không được tự gắn nhãn `clinically implausible`, không được
thay thế đánh giá của chuyên gia X-quang và không được tự đổi sang window
khác.

VOI LUT output bounds phải được suy ra từ LUT Descriptor, không dùng observed output min/max.

Windowing output range phải được suy ra theo DICOM/pydicom behavior và theoretical modality bounds.

Fallback:

```text
clip/map bằng theoretical modality-domain range
```

Không dùng:

```text
arr.min()
arr.max()
np.percentile()
histogram percentile clipping
```

để quyết định mapping bounds.

Ghi:

```text
voi_branch
selected_window_index
all_window_centers
all_window_widths
voi_lut_index
voi_lut_function
theoretical_voi_low
theoretical_voi_high
```

## 17.5 Presentation polarity

Modality transformation and VOI transformation must remain earlier stages.
Presentation polarity handling occurs only after those stages.

The project action table is locked as follows:

```text
PhotometricInterpretation | PresentationLUTShape | project action
MONOCHROME1                | absent               | invert once
MONOCHROME2                | absent               | no inversion
MONOCHROME1                | INVERSE              | invert once
MONOCHROME2                | IDENTITY             | no inversion
MONOCHROME1                | IDENTITY             | metadata presentation conflict
MONOCHROME2                | INVERSE              | metadata presentation conflict
```

For the four supported combinations:

```text
MONOCHROME1 + absent
→ invert exactly once

MONOCHROME2 + absent
→ no inversion

MONOCHROME1 + INVERSE
→ invert exactly once

MONOCHROME2 + IDENTITY
→ no inversion
```

For the two metadata presentation conflicts:

```text
MONOCHROME1 + IDENTITY
MONOCHROME2 + INVERSE
```

the script must record:

```text
protocol_gap_detected = true
protocol_review_required = true
phase_status = BLOCKED_PROTOCOL_REVIEW
structural_dod_candidate = false
presentation_metadata_conflict = true
```

Do not infer that `PhotometricInterpretation` or `PresentationLUTShape`
must take precedence.

Do not choose a polarity from visual appearance.

Do not continue pilot pixel conversion for the conflicting image until
`SOPClassUID`, IOD context, and a separate protocol decision have been
reviewed.

If `PresentationLUTSequence` exists, apply the protocol-gap rule from
Section 12. Do not silently substitute `PresentationLUTShape`.

The synthetic transformation evidence must record:

```text
sop_class_uid
photometric_interpretation
presentation_lut_shape
presentation_lut_sequence_present
expected_project_action
expected_inversion_count
actual_inversion_count
metadata_presentation_conflict
protocol_gap_detected
test_status
```

For supported combinations:

```text
expected_inversion_count in {0, 1}
actual_inversion_count == expected_inversion_count
```

For conflict combinations:

```text
expected_inversion_count = null
actual_inversion_count = null
test_status = BLOCKED_PROTOCOL_REVIEW
```

Output target for supported combinations:

```text
MONOCHROME2-equivalent
low = dark
high = bright
```

Do not invert twice.

Ghi:

```text
SOPClassUID
PhotometricInterpretation
PresentationLUTShape
presentation_lut_sequence_present
presentation_branch
presentation_inversion_applied
presentation_inversion_count
presentation_metadata_conflict
```

Validate:

```text
presentation_inversion_count in {0, 1, null}
```

## 17.6 Deterministic uint8

Thứ tự:

```text
clip using theoretical output bounds
linear map to [0, 255]
numpy.rint
clip [0, 255]
cast np.uint8
reapply padding mask = 0
```

NaN hoặc Inf:

```text
hard fail
```

Nếu theoretical low == theoretical high:

```text
hard fail
degenerate_theoretical_range
```

Ghi:

```text
uint8_zero_fraction
uint8_255_fraction
pre_jpeg_width
pre_jpeg_height
pre_jpeg_dtype
pre_jpeg_mode
```

Saturation fractions chỉ là descriptive evidence. Không tự đặt numeric PASS threshold.

Pre-JPEG SHA-256:

```python
hashlib.sha256(
    np.ascontiguousarray(uint8_image).tobytes(order="C")
).hexdigest()
```

Rows, Columns và dtype phải được ghi riêng vì raw byte hash không tự mô tả shape.

## 17.7 Single-channel master representation policy

The pre-JPEG reference image and pilot JPEG files must remain a true
single-channel grayscale representation:

```text
array dimensions = 2
channel count = 1
Pillow mode = L
dtype = uint8
width = original DICOM Columns
height = original DICOM Rows
```

Scientific and technical rationale:

```text
The controlled DICOM scope is monochrome:
SamplesPerPixel == 1
PhotometricInterpretation in {MONOCHROME1, MONOCHROME2}

Replicating one grayscale channel into three identical channels does not add
new radiographic or pathological information. Therefore, the portable master
representation must remain one-channel grayscale.
```

Do not replicate the image into RGB during DICOM-to-JPG conversion.

Model-input channel adaptation is outside Phase 2D.1B. If a later pretrained
backbone requires three channels, the dataset/model pipeline may replicate
the same grayscale image into three identical channels or use an explicitly
approved one-channel backbone configuration.

That later decision must:

```text
be locked in the MMDetection dataset-loading or supervised-baseline phase
be applied consistently to labeled, unlabeled, validation, and test images
not modify the master JPG files
not be silently implemented in this pilot
```

Record:

```text
master_representation_channel_count:
1

master_representation_mode:
L

model_input_channel_adaptation_status:
deferred_to_dataset_loading_or_training_phase

model_input_channel_adaptation_authorized_in_phase2D1B:
false

patient_space_orientation_independently_validated:
false

pixel_matrix_order_unchanged:
true

presentation_metadata_conflict_detected:
false

presentation_lut_sequence_detected:
false
```

## 17.8 Exact pre-JPEG reference artifact

For every selected pilot image, persist the exact pre-JPEG uint8 reference
as a lossless PNG:

```text
data/processed/images_jpg_pilot/reference_uint8/train/<image_id>.png
```

Requirements:

```text
mode = L
dtype = uint8
lossless PNG
same width and height as DICOM
same pixel matrix ordering as the in-memory pre-JPEG uint8 array
not used as the final training representation
```

After writing each reference PNG:

```text
decode PNG
assert decoded mode == L
assert decoded dtype == uint8
assert decoded dimensions equal the DICOM and in-memory reference
assert decoded pixels exactly equal the in-memory pre-JPEG uint8 array
```

Record:

```text
reference_png_relative_path
reference_png_byte_sha256
reference_png_decoded_pixel_sha256
reference_png_exact_pixel_match
```

The reference PNG exists only to preserve auditable pilot evidence and to
allow GPT/researcher inspection without re-running DICOM conversion.

It must not be described as the final JPG training representation.

---

# 18. Paired JPEG encoding

Mỗi pilot image phải tạo hai candidate:

```text
quality 95
quality 100
```

Output pilot-only:

```text
data/processed/images_jpg_pilot/q95/train/<image_id>.jpg
data/processed/images_jpg_pilot/q100/train/<image_id>.jpg
```

Không ghi vào:

```text
data/processed/images_jpg/train
```

Pillow encoding:

```python
Image.fromarray(uint8_image, mode="L").save(
    output_path,
    format="JPEG",
    quality=quality,
    optimize=False,
    progressive=False,
)
```

Không thêm:

```text
EXIF
orientation transform
ICC profile
resize
thumbnail
```

Sau khi encode:

```text
reopen JPG bằng Pillow
load decoded pixels
validate format == JPEG
validate mode == L
validate size unchanged
validate dtype == uint8
validate EXIF orientation absent or 1
```

Không gọi JPEG quality 100 là lossless.

Tính:

```text
source DICOM SHA-256
pre-JPEG uint8 SHA-256
output JPG byte SHA-256
decoded JPG pixel SHA-256
```

Thực hiện same-environment determinism test trên một deterministic subset:

```text
encode cùng pre-JPEG array hai lần
cùng Pillow config
cùng environment
```

Byte hashes phải giống nhau.

Nếu khác:

```text
hard fail
non_deterministic_jpeg_encoding
```

---

# 19. Whole-image JPEG fidelity

JPEG fidelity reference:

```text
pre-JPEG uint8 image
```

Comparison target:

```text
decoded JPG uint8 image
```

Không mô tả khác biệt raw DICOM → JPG là JPEG compression error.

Tính metrics ở native resolution:

```text
MAE
RMSE
PSNR
SSIM
maximum absolute error
p95 absolute error
p99 absolute error
file size bytes
pre-JPEG uint8 bytes
compression ratio
```

Definitions:

```python
reference = pre_jpeg.astype(np.float64)
target = decoded_jpg.astype(np.float64)
error = target - reference
absolute_error = np.abs(error)
mae = np.mean(absolute_error)
rmse = np.sqrt(np.mean(error ** 2))
```

PSNR:

```text
data_range = 255
```

Nếu RMSE bằng 0:

```text
psnr_db = null
psnr_is_infinite = true
```

Không serialize `Infinity`, `NaN` hoặc `-Infinity` vào JSON.

SSIM:

```python
skimage.metrics.structural_similarity(
    pre_jpeg_uint8,
    decoded_jpg_uint8,
    data_range=255,
    channel_axis=None,
)
```

Ghi:

```text
skimage version
data_range
channel_axis
win_size nếu được chỉ định
gaussian_weights
use_sample_covariance
```

Percentile method phải được ghi rõ và dùng ổn định.

Compression ratio:

```text
pre_jpeg_uint8_bytes / jpg_file_size_bytes
```

Đồng thời ghi:

```text
jpg_bytes_per_pixel
```

Không tự đặt PSNR/SSIM/MAE threshold.

---

# 20. BBox ROI fidelity

Tính cho toàn bộ canonical bbox thuộc selected pilot images.

Canonical source:

```text
data/processed/canonical/canonical_bbox_table.csv
```

Cross-check với COCO annotation bằng:

```text
canonical_ann_id
image_id/original_image_id
canonical_class_id
coordinates
```

Không sửa COCO hoặc canonical bbox.

Canonical bbox là:

```text
xyxy_original_image
```

COCO bbox là:

```text
xywh_absolute
```

Validate hai representation khớp trong tolerance đã khóa.

Để extract NumPy ROI:

```python
x0 = floor(x_min)
y0 = floor(y_min)
x1 = ceil(x_max)
y1 = ceil(y_max)
```

Hoặc tương đương từ COCO:

```python
x0 = floor(x)
y0 = floor(y)
x1 = ceil(x + width)
y1 = ceil(y + height)
```

Lưu riêng:

```text
canonical floating coordinates
integer extraction coordinates
```

Việc tạo integer ROI slice không phải sửa canonical bbox.

Không ghi extraction coordinates ngược vào canonical/COCO files.

Mỗi:

```text
canonical annotation × JPEG quality
```

phải có:

```text
annotation_id
canonical_ann_id
source_row_id
rad_id
image_id
coco_image_id
category_id
canonical_class_id
class_id_original
class_name
canonical_x_min
canonical_y_min
canonical_x_max
canonical_y_max
bbox_width
bbox_height
bbox_area
relative_bbox_area
extraction_x0
extraction_y0
extraction_x1
extraction_y1
roi_width
roi_height
ROI_MAE
ROI_PSNR
ROI_PSNR_is_infinite
ROI_SSIM
ROI_SSIM_evaluable
ROI_SSIM_reason
ROI_SSIM_win_size
ROI_maximum_absolute_error
jpeg_quality
```

Nếu ROI quá nhỏ cho SSIM:

```text
chọn largest valid odd win_size
win_size <= min(roi_height, roi_width)
win_size >= 3
```

Nếu vẫn không evaluable:

```text
ROI_SSIM = null
ROI_SSIM_evaluable = false
ghi reason
```

Không ghi NaN.

Tạo các summary:

```text
annotation-level micro summary
image-macro summary
class-macro summary
per-class distributions
small-lesion summary
rare-class summary
worst-case ROI per quality
paired q100-minus-q95 metrics
```

Không chỉ báo cáo mean trên toàn bộ bbox, vì class phổ biến và ảnh có nhiều radiologist sẽ chi phối kết quả.

---

# 21. Định nghĩa small lesion và rare class trong pilot evidence

Không tự đặt clinical threshold mới.

`small lesion examples` được chọn bằng deterministic ranking:

```text
relative_bbox_area tăng dần
```

Tối thiểu lấy:

```text
smallest relative bbox overall
smallest relative bbox của từng class có trong pilot
```

`rare-class examples` phải được xác định từ canonical class mapping:

```text
image_count hoặc bbox_count tăng dần
```

Không tự đặt một ngưỡng arbitrary để gọi class là rare.

Báo cáo phải ghi rõ ranking basis.

---

# 22. Geometry validation

Mỗi selected image × JPEG quality phải xác nhận:

```text
DICOM Rows == COCO height
DICOM Columns == COCO width
DICOM Rows == canonical image height
DICOM Columns == canonical image width
pre-JPEG shape unchanged
reference PNG shape unchanged
decoded JPG shape unchanged
reference PNG mode == L
JPG mode == L
reference PNG dtype == uint8
decoded JPG dtype == uint8
reference PNG pixels exactly equal the in-memory pre-JPEG uint8 array
EXIF orientation absent or 1
all canonical bboxes remain valid
all canonical bboxes remain in bounds
bbox scaling required == false
pixel_matrix_order_unchanged == true
rotation_applied == false
flip_applied == false
transpose_applied == false
exif_orientation_transform_applied == false
```

If any dimension, pixel-matrix ordering, or transform check fails:

```text
hard fail
do not scale bbox
do not continue silently
```

Không được tự sửa bbox để làm geometry PASS.

The validation report must state exactly:

```text
Geometry preservation in Phase 2D.1B refers to unchanged pixel-matrix
dimensions and ordering, with no crop, resize, rotation, flip, transpose,
or EXIF orientation transform. It does not claim independent validation
of patient-space orientation.
```

Do not use the broader claim:

```text
patient-space orientation independently validated
```

# 23. Visual evidence

Tạo:

```text
plots/phase2D1B_pilot/full_image/
plots/phase2D1B_pilot/bbox_crops/
plots/phase2D1B_pilot/difference_heatmaps/
plots/phase2D1B_pilot/contact_sheets/
```

## Full-image panel

Mỗi panel gồm:

```text
pre-JPEG uint8 reference
quality 95 decoded
quality 100 decoded
quality 95 absolute difference
quality 100 absolute difference
metadata summary
modality branch
VOI branch
presentation branch
```

## BBox panel

Mỗi panel gồm:

```text
pre-JPEG reference crop
quality 95 crop
quality 100 crop
quality 95 difference
quality 100 difference
class name
canonical bbox
relative bbox area
ROI metrics
```

Difference heatmap q95 và q100 phải dùng cùng scale:

```text
0..255
```

Full image có thể resize chỉ để tạo contact sheet, nhưng caption phải ghi:

```text
display resized only; all metrics computed at native resolution
```

Không được resize ảnh dùng để tính metrics.

Visual subset deterministic phải bao gồm:

```text
all mandatory dimension/pixel/bbox extrema
smallest relative bbox examples
smallest relative bbox per class
rare-class examples có trong pilot
at least 4 No Finding images với metadata strata khác nhau
worst q95 whole-image cases
worst q95 ROI cases
padding unusual cases
saturation unusual cases
all warning cases
all error cases nếu có output an toàn
```

Không tự kết luận visual PASS.

Visual audit manifest phải để:

```text
review_status = PENDING_GPT
critical_visual_failure = null
review_notes = null
```

---

# 24. Output artifacts bắt buộc

Tạo:

```text
reports/phase2D1B_pilot_environment.json
reports/phase2D1B_pilot_synthetic_conformance.json
reports/phase2D1B_pilot_synthetic_conformance.md
reports/phase2D1B_pilot_multi_window_audit.csv
reports/phase2D1B_pilot_reference_renderer_concordance.csv
reports/phase2D1B_pilot_reference_viewer_manifest.csv
reports/phase2D1B_pilot_header_inventory.csv
reports/phase2D1B_pilot_metadata_strata.csv
reports/phase2D1B_pilot_selection.csv
reports/phase2D1B_pilot_selection_coverage.csv
reports/phase2D1B_pilot_fidelity_metrics.csv
reports/phase2D1B_pilot_bbox_roi_metrics.csv
reports/phase2D1B_pilot_quality_summary.csv
reports/phase2D1B_pilot_quality_pairwise.csv
reports/phase2D1B_pilot_geometry_validation.csv
reports/phase2D1B_pilot_visual_audit_manifest.csv
reports/phase2D1B_pilot_errors.csv
reports/phase2D1B_pilot_validation.json
reports/phase2D1B_pilot_validation.md
reports/phase2D1B_pilot_decision_template.json
reports/phase2D1B_pilot_decision_template.md
```

Mapping:

```text
data/processed/image_mapping/
phase2D1B_pilot_dicom_to_jpg_mapping.csv
```

Pilot images:

```text
data/processed/images_jpg_pilot/reference_uint8/train/<image_id>.png
data/processed/images_jpg_pilot/q95/train/<image_id>.jpg
data/processed/images_jpg_pilot/q100/train/<image_id>.jpg
```

Không tạo:

```text
data/processed/images_jpg/train/
data/processed/coco/coco_master_jpg.json
```

## 24.1 Pending decision evidence

The script must create decision templates only.

`reports/phase2D1B_pilot_decision_template.json` must contain:

```json
{
  "decision_status": "pending_gpt_and_researcher_review",
  "final_jpeg_quality": null,
  "selected_candidate": null,
  "full_conversion_authorized": false,
  "decision_rationale": null,
  "reviewed_evidence": [],
  "reviewer_notes": null
}
```

The Markdown decision template must communicate the same pending state.

The script must not:

```text
select q95
select q100
fill decision_rationale automatically
set full_conversion_authorized true
convert the template into a final decision artifact
```

After GPT review and explicit researcher confirmation, the template may be
updated or a separate final decision artifact may be created in a later
authorized step.

---

# 25. Mapping schema

Dùng long format:

```text
một row cho mỗi image × JPEG quality
```

Các field tối thiểu:

```text
original_image_id
canonical_image_id
coco_image_id
dicom_relative_path
pilot_jpg_relative_path
source_dicom_sha256
pre_jpeg_uint8_sha256
reference_png_relative_path
reference_png_byte_sha256
reference_png_decoded_pixel_sha256
reference_png_exact_pixel_match
output_jpg_sha256
decoded_jpg_uint8_sha256
protocol_version
protocol_sha256
jpeg_quality
decoder_backend
transfer_syntax_uid
sop_class_uid
modality
rows
columns
bits_allocated
bits_stored
high_bit
pixel_representation
samples_per_pixel
photometric_interpretation
number_of_frames_effective
modality_branch
voi_branch
presentation_lut_shape
presentation_lut_sequence_present
presentation_inversion_applied
presentation_inversion_count
presentation_metadata_conflict
padding_present
padding_pixel_count
pre_jpeg_channel_count
pre_jpeg_mode
model_input_channel_adaptation_applied
pixel_matrix_order_unchanged
rotation_applied
flip_applied
transpose_applied
exif_orientation_transform_applied
```

Không ghi absolute local DICOM path vào mapping.

---

# 26. Selection output schema

`reports/phase2D1B_pilot_selection.csv` tối thiểu có:

```text
selection_order
original_image_id
coco_image_id
canonical_image_id
scope_label
is_negative
class_ids_coco
canonical_class_ids
class_names
width
height
pixel_count
bbox_count
minimum_bbox_area
minimum_relative_bbox_area
selected_for_features
newly_covered_feature_count
tie_break_rank
```

`selected_for_features` phải giải thích tại sao ảnh được chọn.

Không chỉ ghi một boolean `selected=true`.

---

# 27. Metadata strata output

`reports/phase2D1B_pilot_metadata_strata.csv` tối thiểu có:

```text
stratum_type
stratum_name
stratum_value
scope_image_count
selected_image_count
covered
selected_image_ids
```

DoD yêu cầu:

```text
covered == true
```

cho tất cả observed supported strata.

Unsupported strata phải xuất ở errors thay vì bị bỏ qua.

---

# 28. JSON status rules

`reports/phase2D1B_pilot_validation.json` phải luôn dùng strict JSON:

```text
allow_nan = false
```

Trước GPT review, trạng thái tối đa:

```json
{
  "phase_id": "2D.1B-Pilot",
  "mentor_approval_status": "approved",
  "pipeline_implementation_authorized": true,
  "technical_validation_authorized": true,
  "pipeline_name": "dicom_metadata_aware_standard_aligned_reference_representation_pipeline",
  "pipeline_display_name": "DICOM metadata-aware, standard-aligned reference representation pipeline",
  "method_positioning": "reference_representation_pipeline",
  "technical_basis": "dicom_grayscale_transformation_sequence",
  "applied_research_precedent": "cheng_et_al_2024",
  "research_precedent_scope": "metadata_use_precedent_not_algorithm_replication",
  "novel_algorithm_claimed": false,
  "full_dicom_standard_conformance_claimed": false,
  "clinical_validation_claimed": false,
  "downstream_superiority_evaluated": false,
  "controlled_downstream_ablation_status": "pending_mentor_confirmation",
  "controlled_downstream_ablation_authorized": false,
  "controlled_downstream_ablation_required": null,
  "master_representation_channel_count": 1,
  "master_representation_mode": "L",
  "model_input_channel_adaptation_status": "deferred_to_dataset_loading_or_training_phase",
  "model_input_channel_adaptation_authorized_in_phase2D1B": false,
  "patient_space_orientation_independently_validated": false,
  "pixel_matrix_order_unchanged": true,
  "presentation_metadata_conflict_detected": false,
  "presentation_lut_sequence_detected": false,
  "protocol_gap_detected": false,
  "protocol_review_required": false,
  "visual_review_status": "PENDING_GPT",
  "critical_visual_failure": null,
  "phase_status": "OPEN_REVIEW_REQUIRED",
  "structural_dod_candidate": true,
  "gpt_review_status": "pending",
  "final_jpeg_quality": null,
  "final_quality_status": "pending_gpt_pilot_review",
  "full_conversion_authorized": false,
  "jpg_training_representation_ready": false,
  "coco_jpg_training_annotation_ready": false,
  "mmdetection_dataset_loading_ready": false,
  "empty_image_retention_ready": false,
  "dataset_training_ready": false,
  "training_authorized": false
}
```

Nếu structural failure:

```text
phase_status = BLOCKED
structural_dod_candidate = false
protocol_review_required = true
```

Nếu locked protocol không resolve được case:

```text
phase_status = BLOCKED_PROTOCOL_REVIEW
structural_dod_candidate = false
protocol_gap_detected = true
protocol_review_required = true
```

Nếu chỉ có visual/reference-viewer concern nhưng chưa xác lập lỗi cấu trúc:

```text
phase_status = OPEN_REVIEW_REQUIRED
protocol_review_required = true
visual_review_status = PENDING_EXPERT_REVIEW
critical_visual_failure = null
```

Không được tự động gọi case là clinically implausible.

Script không được ghi:

```text
phase_status = PASS
final_jpeg_quality = 95
final_jpeg_quality = 100
full_conversion_authorized = true
```

Script có thể tạo paired numeric evidence nhưng không tự chọn quality.

Không tạo automated recommendation dựa riêng trên mean PSNR/SSIM.

---

# 29. Definition of Done do script kiểm tra

Script phải tạo checklist machine-readable cho các điều kiện:

```text
mentor approval status recorded as approved
pipeline implementation authorized remains true
technical validation authorized remains true
approved scope recorded without expanding authorization
official pipeline name recorded exactly
scientific positioning recorded
technical basis recorded
Cheng et al. precedent scope recorded without algorithm-replication claim
novel algorithm claim remains false
full DICOM Standard conformance claim remains false
clinical validation claim remains false
downstream superiority evaluated remains false
controlled downstream ablation status is pending mentor confirmation
controlled downstream ablation authorized remains false
controlled downstream ablation required remains null
locked project protocol strict load PASS
protocol nested field-path map PASS
protocol_metadata.protocol_version PASS
protocol fingerprint PASS
protocol gap detected == false
final quality remains null
COCO master SHA PASS
locked counts PASS
canonical bbox count PASS
canonical class mapping PASS
class ID mapping PASS
DICOM root resolved through CLI/environment PASS
4894 controlled DICOM paths exist
header inventory 4894/4894
SOPClassUID and Modality inventory complete
PresentationLUTSequence presence audited
no PHI exported
decoder preflight PASS
synthetic transformation conformance PASS
synthetic tests explicitly disclaim formal complete DICOM conformance
supported polarity action table complete
metadata presentation conflicts block as BLOCKED_PROTOCOL_REVIEW
multi-window audit complete
independent renderer comparison status explicitly recorded
reference viewer manifest generated
preexisting forbidden artifacts absent before pixel decoding
pilot selection deterministic
pilot image count >= 64
No Finding count >= 16
14/14 abnormal classes covered
all supported metadata strata covered
dimension extrema covered
pixel-count extrema covered
bbox extrema covered
pixel decoding limited to selected pilot
lossless reference PNG generated for every pilot image
reference PNG exact-pixel equality PASS
q95 generated for every pilot image
q100 generated for every pilot image
paired outputs complete
JPG mode L
JPG dtype uint8
pre-JPEG reference channel count == 1
master representation remains single-channel grayscale
no RGB replication during conversion
model-input channel adaptation remains deferred and unapplied
geometry unchanged
pixel matrix order unchanged
no patient-space orientation claim
rotation/flip/transpose/EXIF transforms all false
all bbox valid
bbox scaling not required
traceability complete
whole-image metrics complete
ROI metrics complete
visual evidence generated
strict JSON PASS
no NaN/Infinity
atomic output promotion PASS
hard errors == 0
full JPG dataset not created
coco_master_jpg.json not created
split not created
training not started
decision templates generated with pending state
final quality remains null
all readiness flags remain false
```

Ngay cả khi tất cả điều kiện trên đạt, chỉ ghi:

```text
structural_dod_candidate = true
phase_status = OPEN_REVIEW_REQUIRED
```

Không ghi pilot PASS.

---

# 30. Atomic output policy

Tất cả output phải được tạo trong temporary staging directory.

Trước promotion phải validate:

```text
all expected files exist
CSV schemas đúng
row counts đúng
JSON strict parse
no NaN/Infinity
paired q95/q100 complete
hash fields non-empty
geometry PASS
forbidden artifacts absent
```

Chỉ sau khi tất cả pre-promotion checks PASS mới atomic-promote.

Nếu failure:

```text
không ghi đè valid prior output
xóa temporary files
giữ lại prior evidence
return non-zero exit code
```

Mặc định không overwrite evidence hiện có.

Hỗ trợ:

```text
--overwrite
```

nhưng chỉ overwrite sau khi staging validation PASS.

---

# 31. Guardrail tests

Tạo:

```text
tests/test_phase2D1B_pilot_guardrails.py
```

Tests phải dùng synthetic fixtures và temporary directories khi có thể. Không phụ thuộc toàn bộ 4.894 DICOM cho unit tests.

Tests tối thiểu:

```text
mentor approval status approved
pipeline implementation authorized true
technical validation authorized true
approved scope does not authorize full conversion, training, or ablation
official pipeline name exact match
method positioning is reference representation pipeline
novel algorithm claim false
full DICOM Standard conformance claim false
clinical validation claim false
downstream superiority evaluated false
controlled downstream ablation status pending mentor confirmation
controlled downstream ablation authorized false
controlled downstream ablation required null
protocol strict-load
actual nested YAML path protocol_metadata.protocol_version
missing nested path becomes protocol_schema_mismatch
similarly named field elsewhere is not used
protocol fingerprint
protocol-gap case blocks without invented fallback
final quality is null
quality candidates exactly [95, 100]
all readiness flags false
all forbidden actions false
DICOM root environment resolution
CLI/env path conflict hard fail
COCO relative path safe resolution
path traversal rejected
class mapping 0..13 versus category IDs 1..14
No Finding excluded from categories
locked counts validation
COCO hash drift detection
deterministic tie-break
deterministic selection
minimum 64 logic
minimum 16 No Finding logic
all-class coverage logic
metadata strata expansion logic
pilot scope cap
pixel decoding restricted to selected set
accidental 4894 pixel decode rejected
padding single-value mask
padding inclusive-range mask
range-limit without padding-value hard fail
unsigned theoretical range
signed theoretical range
negative rescale slope endpoint sorting
incomplete rescale metadata hard fail
Modality LUT priority
exactly one modality branch selected per image
Modality LUT and rescale not sequential
identity modality branch selected only when LUT and complete rescale are absent
VOI LUT priority over window
single-valued window parsing
multi-valued window parsing
window cardinality mismatch
theoretical fallback
observed array min/max not used for mapping
percentile clipping absent
MONOCHROME1 absent inversion
MONOCHROME2 absent no inversion
MONOCHROME1 INVERSE inversion
MONOCHROME2 IDENTITY no inversion
MONOCHROME1 IDENTITY becomes BLOCKED_PROTOCOL_REVIEW
MONOCHROME2 INVERSE becomes BLOCKED_PROTOCOL_REVIEW
PresentationLUTSequence presence becomes BLOCKED_PROTOCOL_REVIEW
supported presentation combinations use exactly zero or one inversion
padding reapplied to zero after inversion
numpy.rint uint8 behavior
NaN/Inf hard fail
lossless reference PNG output
reference PNG exact-pixel equality
paired q95/q100 outputs
pre-JPEG reference is a 2D single-channel uint8 array
JPEG mode L
pre-JPEG and decoded JPEG channel count == 1
RGB replication absent from conversion code
model-input channel adaptation status is deferred
model-input channel adaptation not applied in Phase 2D.1B
no EXIF orientation transform
pixel matrix order preserved
patient-space orientation is not claimed as independently validated
geometry preserved
bbox remains in bounds
bbox scaling never performed
whole-image metrics correctness
PSNR infinite strict-JSON handling
SSIM data_range 255
ROI extraction coordinates
small ROI SSIM handling
synthetic expected values independent from production helpers
self-referential synthetic tests rejected
strict JSON no NaN/Infinity
portable mapping excludes absolute path
PHI fields excluded
subjective visual concern does not automatically claim clinical failure
subjective visual concern remains OPEN_REVIEW_REQUIRED
structural VOI failure becomes BLOCKED
preexisting forbidden artifact detected before decoding
current run never deletes preexisting forbidden artifacts
reference renderer uncontrolled configuration marked NOT_COMPARABLE_CONFIGURATION_UNCONTROLLED
atomic failure preserves previous output
temporary files cleaned
pilot output paths only
coco_master_jpg.json never created
full images_jpg/train never created
no split/training/inference/pseudo-label/AP operations
phase status never PASS
decision template remains pending
final quality never selected
```

Có thể dùng AST/source guardrails, nhưng tránh false positive từ docstrings hoặc assertion strings.

AST guardrails phải kiểm tra hành động thực tế, không chỉ sự xuất hiện của từ bị cấm trong tài liệu.

---

# 32. Dependency policy

Không tự động cài package.

Code phải phát hiện và báo rõ dependency còn thiếu.

Các package dự kiến:

```text
numpy
pandas
pydicom
Pillow
PyYAML
scikit-image
matplotlib
pylibjpeg
pylibjpeg-openjpeg
```

`gdcm` chỉ optional nếu hỗ trợ explicit decoder selection.

Không import:

```text
MMDetection
mmcv
mmengine
torch
torchvision
Detectron2
```

Phase này không cần framework training.

---

# 33. Lệnh Claude được phép cung cấp

Cuối câu trả lời, Claude chỉ cung cấp:

## Unit tests

```cmd
python -m unittest discover -s tests -p "test_phase2D1B_pilot_guardrails.py" -v
```

## Kiểm tra biến môi trường trong Windows CMD

```cmd
echo %VINBIGDATA_DICOM_ROOT%
```

## Thiết lập biến môi trường cho cửa sổ CMD hiện tại

```cmd
set VINBIGDATA_DICOM_ROOT=D:\ssl_detection_xray\data\raw\vinbigdata\dicom_subset
```

## Chạy pilot

```cmd
python scripts\02D1B_pilot_dicom_to_jpg.py
```

Có thể thêm các option pilot hợp lệ như:

```cmd
python scripts\02D1B_pilot_dicom_to_jpg.py --jpeg2000-decoder pylibjpeg
```

Không cung cấp bất kỳ lệnh full conversion nào.

Không cung cấp lệnh tạo `coco_master_jpg.json`.

Không cung cấp lệnh train hoặc inference.

---

# 34. Kết quả Claude phải trả về

Sau khi viết code, trả lời:

```text
1. Danh sách file đã tạo hoặc sửa, bao gồm decision templates.
2. Tóm tắt architecture.
3. Danh sách dependencies được kiểm tra.
4. Mục đích Phase 2D.1B-Pilot và phạm vi mentor đã phê duyệt.
5. Các quyết định protocol được giữ nguyên.
6. Các guardrails đã triển khai.
7. Lệnh chạy unit tests.
8. Lệnh chạy pilot.
9. Khẳng định không có full-conversion command.
```

Không được:

```text
chạy full conversion
chọn quality 95
chọn quality 100
claim pilot PASS
sửa checklist
sửa PROJECT_CONTEXT.md
sửa PHASE_HANDOFF.md
sửa research_log.md
authorize full conversion
authorize training
```

# LOCKED SCIENTIFIC POSITIONING AND CORRECTNESS EVIDENCE

The Phase 2D.1B transformation remains exactly the locked protocol
version 1.0.0.

The supervisor has approved implementation and technical validation of
the DICOM metadata-aware, standard-aligned reference representation
pipeline within the explicitly approved pilot scope.

This approval does not authorize full conversion, final JPEG-quality
selection, training, downstream superiority claims, or controlled
downstream ablation.

Do not modify the transformation algorithm.
Do not modify the protocol YAML.
Do not modify Phase 2D.1A artifacts.
Do not increment the protocol version.
Do not invent behavior for unresolved protocol cases.

The official pipeline name is:

```text
DICOM metadata-aware, standard-aligned reference representation pipeline
```

This is a reference representation pipeline.

It must not be described as:

```text
a novel preprocessing algorithm
an optimal preprocessing method
the best image representation
a complete DICOM-conformant renderer
a clinically validated renderer
superior to CLAHE
superior to Histogram Equalization
superior to logarithmic transformation
superior to rib-guided augmentation
```

The pipeline is:

```text
metadata-aware because transformation branches use relevant DICOM
metadata;

standard-aligned because its transformation sequence is based on the
DICOM grayscale transformation model;

a reference representation because downstream detection superiority
has not been evaluated.
```

The technical basis is:

```text
DICOM grayscale transformation sequence
```

The applied research precedent is:

```text
Cheng et al. (2024)
```

Cheng et al. supports the use of DICOM metadata such as Window Center,
Window Width, and BitsStored during CXR image preparation.

This project does not reproduce Cheng logarithmic transformation or
simplest color balance.

Khalili rib-guided projection-based augmentation is related work for
anatomical and cross-dataset augmentation. It is not the technical
foundation of this Phase 2D.1B representation pipeline.

Phase 2D.1B evaluates only:

```text
locked project protocol conformity
implementation correctness
metadata-branch traceability
geometry preservation
bbox-coordinate preservation
same-environment reproducibility
JPEG quality 95 versus quality 100 fidelity
```

Phase 2D.1B does not evaluate:

```text
downstream detection superiority
AP or mAP improvement
semi-supervised learning performance
preprocessing-method superiority
```

Do not implement in this pilot:

```text
Cheng logarithmic transformation
simplest color balance
Histogram Equalization
CLAHE
anisotropic filtering
median filtering
rib segmentation
LightGlue
projection
cropping
resizing
bbox transformation
training-time augmentation
```

Add mandatory synthetic transformation conformance tests covering:

```text
unsigned stored pixels
signed stored pixels
identity modality branch
rescale slope/intercept
negative rescale slope
Modality LUT
Window LINEAR
Window LINEAR_EXACT
Window SIGMOID
VOI LUT
MONOCHROME1
MONOCHROME2
PresentationLUTShape absent
PresentationLUTShape IDENTITY
PresentationLUTShape INVERSE
all supported photometric/presentation combinations
both metadata presentation conflict combinations
PresentationLUTSequence presence
PixelPaddingValue
PixelPaddingRangeLimit
multi-valued windows
```

Synthetic expected outputs must be independently defined using:

```text
hand-computed arrays
explicit locked formulas
fixed expected constants
```

A synthetic test must not compute its expected output by calling the same
production helper, pydicom transformation wrapper, or branch function that
is being tested.

Self-referential tests are forbidden.

This requirement is especially important for:

```text
Modality LUT
Rescale Slope/Intercept
LINEAR
LINEAR_EXACT
SIGMOID
VOI LUT
polarity inversion
uint8 rounding
```

Multiple WindowCenter/WindowWidth pairs are alternative views. Only one
pair is applied at a time according to the locked selected index.

These tests validate implementation conformity to the locked project
protocol.

They do not constitute formal certification of complete DICOM Standard
conformance.

Create:

```text
reports/phase2D1B_pilot_synthetic_conformance.json
reports/phase2D1B_pilot_synthetic_conformance.md
reports/phase2D1B_pilot_multi_window_audit.csv
reports/phase2D1B_pilot_reference_renderer_concordance.csv
reports/phase2D1B_pilot_reference_viewer_manifest.csv
```

Synthetic transformation conformance is a mandatory structural DoD
gate.

Independent renderer comparison is optional and must be explicitly
reported as one of:

```text
PASS
FAIL
NOT_RUN_DEPENDENCY_UNAVAILABLE
NOT_COMPARABLE_CONFIGURATION_UNCONTROLLED
```

It must never be silently skipped.

Reference-renderer comparison is valid only when the following are
controlled:

```text
selected VOI index
Window Center and Window Width
VOI function
presentation polarity
padding treatment
output bit depth
```

For multi-valued WindowCenter/WindowWidth cases:

```text
retain selected_index = 0 according to the locked protocol
record all alternative values
generate alternative-view visual evidence
do not automatically change the selected index
```

If the metadata or transformation is structurally invalid:

```text
protocol_review_required = true
phase_status = BLOCKED
structural_dod_candidate = false
```

If index 0 is visually suspicious or inconsistent with a reference
viewer, but no structural error is established:

```text
protocol_review_required = true
phase_status = OPEN_REVIEW_REQUIRED
visual_review_status = PENDING_EXPERT_REVIEW
critical_visual_failure = null
```

Do not label an image clinically implausible automatically.

Do not automatically choose another window.

If the locked protocol does not resolve a transformation case:

```text
protocol_gap_detected = true
protocol_review_required = true
phase_status = BLOCKED_PROTOCOL_REVIEW
structural_dod_candidate = false
```

Do not invent a fallback.
Do not infer a policy from image appearance.

Add the following fields to the validation JSON:

```text
mentor_approval_status:
approved

pipeline_implementation_authorized:
true

technical_validation_authorized:
true

pipeline_name:
dicom_metadata_aware_standard_aligned_reference_representation_pipeline

pipeline_display_name:
DICOM metadata-aware, standard-aligned reference representation pipeline

method_positioning:
reference_representation_pipeline

technical_basis:
dicom_grayscale_transformation_sequence

applied_research_precedent:
cheng_et_al_2024

research_precedent_scope:
metadata_use_precedent_not_algorithm_replication

novel_algorithm_claimed:
false

full_dicom_standard_conformance_claimed:
false

clinical_validation_claimed:
false

downstream_superiority_evaluated:
false

controlled_downstream_ablation_status:
pending_mentor_confirmation

controlled_downstream_ablation_authorized:
false

controlled_downstream_ablation_required:
null

master_representation_channel_count:
1

master_representation_mode:
L

model_input_channel_adaptation_status:
deferred_to_dataset_loading_or_training_phase

model_input_channel_adaptation_authorized_in_phase2D1B:
false
```

The Markdown report must include:

```text
## Scientific Positioning and Scope
```

The section must state that Phase 2D.1B does not prove that this
representation produces the best downstream detection performance.

Claiming superiority over alternative preprocessing strategies would
require controlled downstream evidence.

Whether a controlled downstream ablation will be included remains
pending mentor confirmation and is not part of the Phase 2D.1B
Definition of Done.
