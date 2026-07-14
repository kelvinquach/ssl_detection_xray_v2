# Phase 2C — Framework & Format Decision / COCO Conversion Planning

_Generated 2026-07-14T08:15:47.966321+00:00._

## Executive summary

Primary framework: **MMDetection** (fallback: Detectron2_optional, only after GPT re-review). Primary annotation format: **COCO_detection_JSON**, sourced from `canonical_detection_schema`. Canonical schema re-confirmed: **4894** images, **36096** bboxes, **14** detection classes. COCO conversion is **NOT** performed here (actual_coco_conversion_done=False); it is planned for Phase 2D. DoD pass candidate: **True**.

## Phase scope

- Decision/protocol evidence ONLY. No COCO file, no split, no training.
- Canonical schema (Phase 2B) is read-only and unmodified.
- Dataset is NOT training-ready at the end of Phase 2C.

## Inputs used

- canonical_image_table: `data/processed/canonical/canonical_image_table.csv`
- canonical_bbox_table: `data/processed/canonical/canonical_bbox_table.csv`
- canonical_class_mapping: `data/processed/canonical/canonical_class_mapping.csv`
- phase2b_validation_json: `reports/phase2B_canonical_schema_validation.json`

## Framework/format comparison

### Table 1 — High-level format comparison

| Format | Strengths | Weaknesses | Fit for this thesis | Verdict |
|---|---|---|---|---|
| **COCO detection JSON** | Native fit for MMDetection and Detectron2; explicit `images` / `annotations` / `categories`; supports images with zero annotations (No Finding negatives); compatible with COCO-style mAP@0.5:0.95 and pycocotools; allows traceability fields such as `canonical_ann_id`, `source_row_id`, `original_image_id` | Verbose; needs a conversion step from the canonical schema | Best preserves the separation between image-level negatives and detection categories; keeps evaluation on the standard COCO path | **CHOSEN** |
| YOLO txt | Simple; popular in the YOLO ecosystem | Uses normalized center-based bbox, adding conversion risk from canonical `xyxy_original_image`; negative images are usually represented by empty txt files, which is fragile here; weaker fit for MMDetection and the COCO mAP pipeline | Poor: the empty-file negative representation is exactly the part this thesis cannot afford to get wrong | Rejected |
| Pascal VOC XML | Human-readable; supports bbox in absolute coordinates | One XML per image, cumbersome for 4,894 controlled-scope images; less suitable for COCO-style mAP@0.5:0.95 and modern MMDetection workflows; negative images represented less cleanly than in COCO | Poor: file sprawl plus a non-COCO AP definition works against the locked metric | Rejected |
| JSONL / custom | Flexible; can preserve arbitrary metadata | Non-standard; would require a custom dataset and evaluator; higher reproducibility risk; weak fit for MMDetection/SSOD without extra custom code | Poor: flexibility does not compensate for hand-rolled evaluation code on the primary metric | Rejected |

### Table 2 — Detailed format suitability matrix

| Criterion | COCO detection JSON | YOLO txt | Pascal VOC XML | JSONL / custom |
|---|---|---|---|---|
| MMDetection compatibility | native (CocoDataset, no custom dataset class needed) | weak (requires conversion or a custom dataset) | supported but legacy (VOCDataset); off the COCO path | none out of the box; requires a custom dataset class |
| COCO mAP@0.5:0.95 / pycocotools | native (pycocotools; standard mAP@0.5:0.95) | indirect (must convert back to COCO for pycocotools mAP) | poor: VOC-style AP differs from COCO mAP@0.5:0.95; conversion needed | requires a custom evaluator or a conversion to COCO |
| Negative / No Finding image support | first-class: an image may appear in `images` with zero entries in `annotations`; requires filter_empty_gt=False so the 500 No Finding negatives are retained | fragile: negatives are represented by an EMPTY .txt file (or a missing file); silent-drop risk for the 500 No Finding images | unclean: an object-less XML must still exist per negative image; less explicit than COCO's images-without-annotations | arbitrary (whatever we define) — but unvalidated |
| Multi-class object detection support | explicit `categories` list; 14 detection classes | yes, but classes are bare integer indices | yes (per-object <name> tags) | arbitrary |
| Category metadata support | explicit and extensible; can carry canonical_class_id and class_id_original alongside the contiguous COCO category_id | minimal: class names live in a separate side file; no room for canonical/original id metadata | implicit only; no central category table | arbitrary; fully flexible |
| BBox coordinate fidelity | absolute pixels; no normalization, no precision loss | reduced: normalization introduces float rounding and depends on exact image dimensions; conversion risk from the canonical absolute xyxy | good: absolute pixels, matches canonical xyxy | can be perfect, but the guarantee is ours to maintain |
| Traceability to canonical/source rows | annotation objects accept extra keys: canonical_ann_id, source_row_id, original_image_id, rad_id | none in-format; would need an external side-car mapping | would require non-standard custom XML tags | excellent in principle (any field can be carried) |
| SSOD teacher-student compatibility | strong: teacher-student SSOD implementations in MMDetection consume COCO; unlabeled/negative images fit the same schema | possible in YOLO-native SSOD, but off-stack for MMDetection | weak; modern SSOD tooling assumes COCO | requires custom teacher-student plumbing |
| Pseudo-label output compatibility | pseudo-labels can be emitted directly as COCO annotations, reusing the same categories and evaluation path | would need conversion back to COCO for evaluation, adding a second lossy hop | awkward: one XML file per pseudo-labelled image per iteration | custom; no standard evaluation path |
| Reproducibility / ecosystem support | widest ecosystem support; standard, well-specified | popular but tied to the YOLO ecosystem | dated; less tooling for COCO-style evaluation | weakest: bespoke format means results are harder for others to reproduce or compare |
| Implementation risk | low (single conversion step; standard tooling) | medium-high (normalization + empty-file negatives) | medium (file sprawl; eval mismatch) | high: custom dataset + custom evaluator are new surfaces for silent bugs in the very metric the thesis reports |
| **Verdict** | **CHOSEN** | **REJECTED** | **REJECTED** | **REJECTED** |

> COCO is selected not merely because it is common, but because it best preserves the required separation between image-level negatives and detection categories, supports standard COCO mAP evaluation, and minimizes custom evaluation code in the later MMDetection/SSOD pipeline.

## Framework selection rationale

### Table 3 — High-level framework comparison

| Framework | Strengths | Weaknesses / risks | Fit for this thesis | Verdict |
|---|---|---|---|---|
| **MMDetection** | Modular PyTorch-based object detection toolbox; supports COCO-format datasets and COCO-style evaluation; official semi-supervised object detection documentation/components including labeled/unlabeled dataset preparation, multi-branch pipeline, semi-supervised dataloader, and teacher-student / MeanTeacher-style training components; config-driven training suits reproducibility | Heavier learning curve; mmcv/mmengine version pinning can be fragile on some platforms | Better fit for the teacher-student SSOD thesis pipeline; COCO in, COCO mAP out, minimal custom code | **CHOSEN** |
| Detectron2 | Strong PyTorch detection framework; good COCO/custom dataset support; solid COCOEvaluator | No first-party SSOD components; the teacher-student pipeline would require more custom implementation in this project; historically awkward to build on Windows | Suitable fallback if MMDetection setup fails, but weaker as an SSOD platform out of the box | **Fallback only** |
| Ultralytics YOLO / YOLO-based | Easy to train and deploy; strong baseline ecosystem; fast iteration | Annotation/evaluation pipeline is YOLO-native and less aligned with the COCO master + MMDetection SSOD protocol; teacher-student SSOD and No Finding empty-image handling need project-specific adaptation | Diverges from the locked COCO/MMDetection protocol; negative-image handling is the exact risk this thesis cannot take | Rejected as primary framework |
| Custom PyTorch / torchvision | Maximum flexibility; no framework constraints | Requires custom dataset, dataloader, evaluator, trainer, pseudo-label loop, EMA teacher, COCO metric integration, logging and config protocol; high implementation risk and reproducibility risk | Engineering effort and silent-bug risk would dominate the research contribution | Rejected |

### Table 4 — Detailed framework suitability matrix

| Criterion | MMDetection | Detectron2 | YOLO-based | Custom PyTorch/torchvision |
|---|---|---|---|---|
| Native object detection support | native: modular PyTorch detection toolbox with a large model zoo (Faster R-CNN, RetinaNet, DINO, etc.) | native: strong PyTorch detection framework (FAIR) | native and fast; strong single-stage baselines | torchvision provides detection models, but the training/eval stack is ours to build |
| COCO dataset compatibility | native CocoDataset; no custom dataset class needed | strong: COCO and custom dataset registration | indirect: expects YOLO-native layout; COCO must be converted, which conflicts with the COCO-master source of truth | must be hand-written |
| COCO mAP@0.5:0.95 / pycocotools | native CocoMetric via pycocotools; mAP@0.5:0.95 out of the box | strong: COCOEvaluator via pycocotools | reports its own mAP; alignment with pycocotools mAP@0.5:0.95 requires care/conversion | must integrate pycocotools manually |
| Teacher-student / SSOD readiness | official semi-supervised components (e.g. SoftTeacher / MeanTeacher-style hooks, EMA teacher); teacher-student SSOD is a documented use case | no first-party SSOD components; teacher-student would rely on external repos (e.g. Unbiased Teacher) or bespoke implementation | SSOD exists in the YOLO ecosystem but is not aligned with the planned MMDetection SSOD protocol | none: EMA teacher, strong/weak augmentation branches, and the pseudo-label loop must all be implemented |
| Labeled/unlabeled pipeline support | official multi-branch pipeline and semi-supervised dataloader for labeled/unlabeled dataset preparation | no official labeled/unlabeled multi-branch dataloader; would need custom plumbing | not first-class for the planned labeled/unlabeled COCO protocol | must be implemented from scratch |
| Config-based reproducibility | config-driven training; the full experiment is captured in a versionable config file | good: LazyConfig / yacs configs | good within its own ecosystem | must design a config/experiment protocol ourselves |
| Empty / No Finding image handling risk | supported via filter_empty_gt=False; the 500 No Finding negatives are retained rather than silently dropped | supported, but requires care with filter_empty_annotations to keep negatives | risky: negatives are expressed as empty label files; silent-drop risk for the 500 No Finding images | fully under our control, but every guarantee is also ours to test |
| Custom DICOM loader extensibility | transform/pipeline registry allows a custom DICOM LoadImage transform without forking the framework | possible via a custom mapper | possible but off the framework's standard path | maximum flexibility (the one genuine advantage) |
| Pseudo-label workflow compatibility | pseudo-labels flow through the same COCO-shaped structures used for training and evaluation | would need a custom pseudo-label loop and EMA teacher | YOLO-native; would diverge from the COCO master | entirely custom |
| Class-wise AP / AP50 / AP75 readiness | CocoMetric reports AP50, AP75 and per-class AP without custom code | COCOEvaluator provides AP50/AP75/per-class AP | available, but under the framework's own metric implementation rather than pycocotools | custom evaluation wiring required |
| Implementation burden | low-medium: mostly configuration, little bespoke code | medium-high for SSOD: the semi-supervised layer must be built for this project | medium: easy to train, but the SSOD + COCO + negative-image protocol must be re-adapted | high: custom dataset, dataloader, evaluator, trainer, pseudo-label loop, EMA teacher, COCO metric integration, logging and config protocol |
| Research reproducibility | high: config + seed + published baselines make results comparable | good, but SSOD code would be project-specific | good, but tied to a different evaluation stack | low-medium: bespoke code is harder for others to reproduce or compare against published baselines |
| Fit for this thesis | direct match: COCO detection + COCO mAP + teacher-student SSOD + labeled/unlabeled handling + config reproducibility | adequate as a detection backbone, weaker as an SSOD platform without extra custom work | poor as primary: the annotation and evaluation pipeline is YOLO-native and diverges from the locked COCO/MMDetection protocol | poor: engineering effort and silent-bug risk would dominate the research contribution |
| **Verdict** | **CHOSEN** | **FALLBACK_ONLY** | **REJECTED** | **REJECTED** |

> MMDetection is selected as the primary framework not merely because it is popular, but because it most directly matches the thesis pipeline: COCO-based detection, COCO mAP evaluation, teacher-student semi-supervised object detection, labeled/unlabeled data handling, and config-driven reproducibility. Detectron2 remains a fallback because it is a strong detection framework, but it would require more custom SSOD plumbing for this project.

## Final framework decision

- primary_framework: **MMDetection**
- fallback_framework: **Detectron2_optional**
- Fallback is used ONLY if MMDetection remote/GPU setup fails AND the change is re-reviewed by GPT.
- local_training_framework_ready: False
- remote_gpu_training_required: True
- Phase 2C does NOT require a successful `mmdet` import; the import probe is defensive and a missing package does not fail this phase.

## Final annotation format decision

- primary_annotation_format: **COCO_detection_JSON**
- source_of_truth: `canonical_detection_schema`
- actual_coco_conversion_done: False
- actual_coco_conversion_phase: Phase 2D
- planned_coco_master_path: `data/processed/coco/coco_master.json` (NOT created here)

## COCO conversion plan for Phase 2D

- `images`: all **4894** controlled-scope images.
- `annotations`: only the **36096** abnormal bboxes.
- `categories`: only the **14** abnormal detection classes.
- No Finding images appear in `images` with ZERO annotations.
- No background class is created.
- Traceability: each COCO annotation keeps `canonical_ann_id` and `source_row_id`.

## No Finding / empty image policy

- no_finding_images: 500
- in COCO images: True
- in COCO annotations: False
- in COCO categories: False
- **MMDetection must set `filter_empty_gt=False`** (or equivalent) so the 500 negative images are NOT silently dropped.

## BBox conversion policy

- source: `xyxy_original_image` → target: `coco_xywh_absolute`
- width = x_max - x_min; height = y_max - y_min; area = width * height; iscrowd = 0.
- No clamping, no deletion, no fusion. 147 near-duplicate candidates retained.

## Category id policy

- COCO category_id is a contiguous integer 1..14 (No Finding excluded). canonical_class_id (0..13) and class_id_original are preserved in category metadata for traceability.
- The original/canonical class ids are retained in category metadata for traceability back to the canonical mapping.

## Path portability policy

- COCO file_name uses relative_dicom_path; resolved at load time by joining the VINBIGDATA_DICOM_ROOT root.
- image root env var: `VINBIGDATA_DICOM_ROOT`
- `local_dicom_path` is evidence only; never a downstream identifier.

## DICOM loader risk

- COCO annotations alone do NOT make the dataset training-ready.
- MMDetection's default `LoadImageFromFile` is NOT validated for DICOM (dicom_loader_validated=False).
- A later phase must provide a custom DICOM loader OR a processed-image conversion protocol before any training run.
- dataset_training_ready: False

## Metric readiness policy

Phase 2C does not compute AP metrics because no split, model training, inference, or prediction file exists yet. However, the selected COCO detection format is required to preserve downstream compatibility with COCO-style detection metrics, including the primary metric mAP@0.5:0.95 and secondary diagnostics such as AP50, AP75, class-wise AP, recall/sensitivity, FP/image, and FP per negative image. These metrics must only be computed in later evaluation phases after COCO conversion, fixed split creation, model training, and prediction generation. No test-set metric may be used for checkpoint selection, threshold tuning, model selection, or augmentation decisions.

## Forbidden actions avoided

- coco_master_json_created: False
- any_coco_json_created: False
- train_val_test_split_created: False
- labeled_unlabeled_split_created: False
- training_started: False
- inference_run: False
- pseudo_label_generated: False
- threshold_tuned: False
- test_set_used: False
- pixel_array_read: False
- image_copied_or_converted: False
- bbox_modified_clamped_deleted_or_fused: False
- source_annotation_modified: False
- phase2b_canonical_schema_modified: False

## Definition of Done status

- dod_pass_candidate: **True**
- Warnings:
  - MMDetection stack not importable locally; this is EXPECTED in Phase 2C (local training framework deferred; remote GPU required).

## Next phase

- **Phase 2D (actual COCO conversion) only after GPT review PASS** of this decision evidence. Do not proceed automatically.
