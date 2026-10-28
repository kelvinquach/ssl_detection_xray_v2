import csv
import json
from collections import Counter
from pathlib import Path

ROOT = Path(r"D:\ssl_detection_xray_v2")
STAGE = ROOT / ".fp_distribution_staging"

RUN_MAP = STAGE / "resolved_main_low_label_runs.csv"
DET_ROOT = STAGE / "detections"
TEST_JSON = ROOT / "data" / "processed" / "coco" / "instances_test.json"

OUT_DIR = ROOT / "analysis" / "rq6" / "fp_distribution"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_DETAIL = OUT_DIR / "fp_per_negative_image_campaign.csv"
OUT_RUN = OUT_DIR / "fp_distribution_run_summary.csv"
OUT_AUDIT = OUT_DIR / "fp_distribution_extraction_audit.json"

TAU = 0.50

EXPECTED_RUNS = 160
EXPECTED_NEGATIVE_IMAGES = 75
EXPECTED_DETAIL_ROWS = 12000


def percentile(values, p):
    values = sorted(values)
    n = len(values)

    if n == 0:
        return None

    if n == 1:
        return float(values[0])

    pos = (n - 1) * p
    lo = int(pos)
    hi = min(lo + 1, n - 1)
    frac = pos - lo

    return (
        values[lo] * (1 - frac)
        + values[hi] * frac
    )


# ============================================================
# 1. LOAD CANONICAL TEST SET
# ============================================================

if not TEST_JSON.exists():
    raise SystemExit(
        f"FAIL: missing test JSON: {TEST_JSON}"
    )

with TEST_JSON.open("r", encoding="utf-8") as f:
    gt = json.load(f)

images = gt["images"]
annotations = gt["annotations"]

image_ids = {int(x["id"]) for x in images}
annotated_ids = {int(a["image_id"]) for a in annotations}

zero_gt_ids = image_ids - annotated_ids

flag_negative_ids = {
    int(x["id"])
    for x in images
    if x.get("is_negative") is True
}

scope_negative_ids = {
    int(x["id"])
    for x in images
    if x.get("scope_label") == "no_finding"
}

if not (
    zero_gt_ids
    == flag_negative_ids
    == scope_negative_ids
):
    raise SystemExit(
        "FAIL: negative-image definitions disagree"
    )

negative_ids = sorted(zero_gt_ids)

if len(images) != 734:
    raise SystemExit(
        f"FAIL: expected 734 test images, got {len(images)}"
    )

if len(negative_ids) != EXPECTED_NEGATIVE_IMAGES:
    raise SystemExit(
        f"FAIL: expected 75 negative images, got {len(negative_ids)}"
    )

image_meta = {
    int(x["id"]): x
    for x in images
}

print("=== TEST SET GATE ===")
print(f"TEST_IMAGES={len(images)}")
print(f"TEST_ANNOTATIONS={len(annotations)}")
print(f"NEGATIVE_IMAGES={len(negative_ids)}")
print("NEGATIVE_DEFINITION_AGREEMENT=PASS")


# ============================================================
# 2. LOAD RESOLVED 160 RUNS
# ============================================================

with RUN_MAP.open("r", encoding="utf-8", newline="") as f:
    runs = list(csv.DictReader(f))

if len(runs) != EXPECTED_RUNS:
    raise SystemExit(
        f"FAIL: expected 160 runs, got {len(runs)}"
    )

methods = Counter(r["method"] for r in runs)

if methods != {"SUP": 80, "SSL": 80}:
    raise SystemExit(
        f"FAIL: method counts={dict(methods)}"
    )

print()
print("=== RUN MAP GATE ===")
print(f"RUNS={len(runs)}")
print(f"SUP={methods['SUP']}")
print(f"SSL={methods['SSL']}")
print("RUN_MAP=PASS")


# ============================================================
# 3. LOCAL EXTRACTION
# ============================================================

detail_rows = []
run_rows = []

for idx, r in enumerate(runs, start=1):

    run_id = r["run_id"]
    attempt_id = r["attempt_id"]

    det_path = (
        DET_ROOT
        / run_id
        / attempt_id
        / "evaluation"
        / "final_test"
        / "detections.json"
    )

    if not det_path.exists():
        raise SystemExit(
            f"FAIL: missing detections: {det_path}"
        )

    print(
        f"[{idx:03d}/{EXPECTED_RUNS}] "
        f"{run_id} {attempt_id}"
    )

    with det_path.open("r", encoding="utf-8") as f:
        detections = json.load(f)

    fp_counter = Counter()

    raw_negative_detection_count = 0
    accepted_negative_detection_count = 0

    max_negative_raw_score = None
    max_all_raw_score = None

    for d in detections:
        image_id = int(d["image_id"])
        score = float(d["score"])

        if (
            max_all_raw_score is None
            or score > max_all_raw_score
        ):
            max_all_raw_score = score

        if image_id in zero_gt_ids:

            raw_negative_detection_count += 1

            if (
                max_negative_raw_score is None
                or score > max_negative_raw_score
            ):
                max_negative_raw_score = score

            if score >= TAU:
                fp_counter[image_id] += 1
                accepted_negative_detection_count += 1

    counts = [
        fp_counter.get(image_id, 0)
        for image_id in negative_ids
    ]

    fp_total = sum(counts)

    any_fp = sum(
        1
        for x in counts
        if x > 0
    )

    fp_per_negative = (
        fp_total / EXPECTED_NEGATIVE_IMAGES
    )

    negative_far = (
        any_fp / EXPECTED_NEGATIVE_IMAGES
    )

    dist = Counter(counts)

    for image_id in negative_ids:

        im = image_meta[image_id]

        detail_rows.append({
            "run_id": run_id,
            "attempt_id": attempt_id,
            "method": r["method"],
            "architecture": r["architecture"],
            "budget": r["budget"],
            "seed_index": int(r["seed_index"]),
            "training_seed": int(r["training_seed"]),
            "image_id": image_id,
            "file_name": im.get("file_name"),
            "fp_count_tau_0_5":
                fp_counter.get(image_id, 0),
        })

    run_rows.append({
        "run_id": run_id,
        "attempt_id": attempt_id,
        "method": r["method"],
        "architecture": r["architecture"],
        "budget": r["budget"],
        "seed_index": int(r["seed_index"]),
        "training_seed": int(r["training_seed"]),

        "n_negative_images":
            EXPECTED_NEGATIVE_IMAGES,

        "tau_eval":
            TAU,

        "fp_total_negative_reconstructed":
            fp_total,

        "fp_per_negative_reconstructed":
            fp_per_negative,

        "negative_far_reconstructed":
            negative_far,

        "negative_images_with_any_fp":
            any_fp,

        "negative_images_zero_fp":
            dist.get(0, 0),

        "fp_count_min":
            min(counts),

        "fp_count_q1":
            percentile(counts, 0.25),

        "fp_count_median":
            percentile(counts, 0.50),

        "fp_count_q3":
            percentile(counts, 0.75),

        "fp_count_max":
            max(counts),

        "raw_negative_detection_count":
            raw_negative_detection_count,

        "accepted_negative_detection_count":
            accepted_negative_detection_count,

        "max_negative_raw_score":
            max_negative_raw_score,

        "max_all_raw_score":
            max_all_raw_score,

        "detections_local_path":
            str(det_path),
    })


# ============================================================
# 4. FINAL COUNT GATES
# ============================================================

if len(detail_rows) != EXPECTED_DETAIL_ROWS:
    raise SystemExit(
        f"FAIL: expected 12000 detail rows, "
        f"got {len(detail_rows)}"
    )

if len(run_rows) != EXPECTED_RUNS:
    raise SystemExit(
        f"FAIL: expected 160 run rows, "
        f"got {len(run_rows)}"
    )


# ============================================================
# 5. WRITE OUTPUT
# ============================================================

detail_fields = list(detail_rows[0].keys())

with OUT_DETAIL.open(
    "w",
    encoding="utf-8",
    newline=""
) as f:

    w = csv.DictWriter(
        f,
        fieldnames=detail_fields
    )

    w.writeheader()
    w.writerows(detail_rows)


run_fields = list(run_rows[0].keys())

with OUT_RUN.open(
    "w",
    encoding="utf-8",
    newline=""
) as f:

    w = csv.DictWriter(
        f,
        fieldnames=run_fields
    )

    w.writeheader()
    w.writerows(run_rows)


audit = {
    "artifact_role":
        "SUPPLEMENTARY_DESCRIPTIVE_FP_DISTRIBUTION",

    "source_predictions":
        "RETAINED_OFFICIAL_FINAL_TEST_DETECTIONS",

    "source_location":
        "LOCAL_STAGING_COPIED_FROM_CANONICAL_GDRIVE_BACKUP",

    "tau_eval":
        TAU,

    "new_model_evaluation_performed":
        False,

    "formal_hypothesis_test_added":
        False,

    "primary_estimand_changed":
        False,

    "inferential_unit_changed":
        False,

    "cross_seed_prediction_pooling_for_inference":
        False,

    "test_image_count":
        len(images),

    "test_annotation_count":
        len(annotations),

    "negative_image_count":
        len(negative_ids),

    "negative_definition":
        (
            "zero_GT == "
            "is_negative_true == "
            "scope_label_no_finding"
        ),

    "main_low_label_runs":
        len(runs),

    "method_counts":
        dict(methods),

    "detail_rows":
        len(detail_rows),

    "run_summary_rows":
        len(run_rows),

    "outputs": {
        "detail":
            str(OUT_DETAIL),

        "run_summary":
            str(OUT_RUN),
    },
}

with OUT_AUDIT.open(
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        audit,
        f,
        ensure_ascii=False,
        indent=2
    )


# ============================================================
# 6. FINAL GATE
# ============================================================

print()
print("======================================")
print("FINAL FP DISTRIBUTION EXTRACTION GATE")
print("======================================")
print(f"RUNS={len(runs)}")
print(f"SUP={methods['SUP']}")
print(f"SSL={methods['SSL']}")
print(f"NEGATIVE_IMAGES={len(negative_ids)}")
print(f"DETAIL_ROWS={len(detail_rows)}")
print(f"RUN_SUMMARY_ROWS={len(run_rows)}")
print(f"EXPECTED_DETAIL_ROWS={EXPECTED_DETAIL_ROWS}")
print(f"DETAIL_OUT={OUT_DETAIL}")
print(f"RUN_SUMMARY_OUT={OUT_RUN}")
print(f"AUDIT_OUT={OUT_AUDIT}")
print("FP_DISTRIBUTION_EXTRACTION=PASS")

