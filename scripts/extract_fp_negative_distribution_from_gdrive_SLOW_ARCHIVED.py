import csv
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

REMOTE_ROOT = "gdrive:SSOD_Backup/vast_54286944/artifacts/runs/official"

TEST_JSON = Path(
    r"D:\ssl_detection_xray_v2\data\processed\coco\instances_test.json"
)

OUT_DIR = Path(
    r"D:\ssl_detection_xray_v2\analysis\rq6\fp_distribution"
)

OUT_DETAIL = OUT_DIR / "fp_per_negative_image_campaign.csv"
OUT_RUN = OUT_DIR / "fp_distribution_run_summary.csv"
OUT_AUDIT = OUT_DIR / "fp_distribution_extraction_audit.json"

TAU = 0.50

EXPECTED_OFFICIAL_RUN_DIRS = 180
EXPECTED_MAIN_LOW_LABEL = 160
EXPECTED_SUP = 80
EXPECTED_SSL = 80
EXPECTED_NEGATIVE_IMAGES = 75
EXPECTED_DETAIL_ROWS = (
    EXPECTED_MAIN_LOW_LABEL * EXPECTED_NEGATIVE_IMAGES
)


def rclone_bytes(*args):
    p = subprocess.run(
        ["rclone", *args],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    if p.returncode != 0:
        raise RuntimeError(
            "rclone command failed\n"
            + "COMMAND: rclone "
            + " ".join(args)
            + "\nSTDERR:\n"
            + p.stderr.decode("utf-8", errors="replace")
        )

    return p.stdout


def rclone_text(*args):
    return rclone_bytes(*args).decode("utf-8")


def rclone_json(remote_path):
    return json.loads(
        rclone_bytes("cat", remote_path).decode("utf-8")
    )


# ============================================================
# 1. VERIFY LOCAL TEST SOURCE
# ============================================================

if not TEST_JSON.exists():
    raise FileNotFoundError(
        f"Canonical test JSON not found: {TEST_JSON}"
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
    raise RuntimeError(
        "Negative-image definitions disagree: "
        f"zero_gt={len(zero_gt_ids)}, "
        f"is_negative={len(flag_negative_ids)}, "
        f"scope_label={len(scope_negative_ids)}"
    )

negative_ids = sorted(zero_gt_ids)

if len(images) != 734:
    raise RuntimeError(
        f"Expected 734 final-test images, found {len(images)}"
    )

if len(negative_ids) != EXPECTED_NEGATIVE_IMAGES:
    raise RuntimeError(
        f"Expected 75 No Finding images, "
        f"found {len(negative_ids)}"
    )

image_meta = {
    int(x["id"]): x
    for x in images
}

print("=== TEST DATA GATE ===")
print(f"TEST_IMAGES={len(images)}")
print(f"TEST_ANNOTATIONS={len(annotations)}")
print(f"NEGATIVE_IMAGES={len(negative_ids)}")
print("NEGATIVE_DEFINITION_AGREEMENT=PASS")


# ============================================================
# 2. DISCOVER OFFICIAL RUN ROOTS FROM GOOGLE DRIVE
# ============================================================

raw = rclone_text(
    "lsf",
    REMOTE_ROOT,
    "--dirs-only"
)

run_ids = sorted(
    line.strip().rstrip("/")
    for line in raw.splitlines()
    if line.strip()
)

print()
print("=== GOOGLE DRIVE INVENTORY ===")
print(f"OFFICIAL_RUN_DIRS={len(run_ids)}")

if len(run_ids) != EXPECTED_OFFICIAL_RUN_DIRS:
    raise RuntimeError(
        f"Expected 180 official run directories, "
        f"found {len(run_ids)}"
    )


# ============================================================
# 3. RESOLVE SUCCESSFUL ATTEMPT + RUN IDENTITY
#    DO NOT GUESS attempt_NNN
# ============================================================

selected = []
discovery_errors = []

for idx, run_id in enumerate(run_ids, start=1):

    logical_root = f"{REMOTE_ROOT}/{run_id}"

    try:
        # ----------------------------------------------------
        # Logical-run identity
        # ----------------------------------------------------
        root_manifest_path = (
            f"{logical_root}/run_root_manifest.json"
        )

        root_manifest = rclone_json(root_manifest_path)

        if root_manifest.get("run_id") != run_id:
            raise RuntimeError(
                f"run_root_manifest run_id mismatch: "
                f"{root_manifest.get('run_id')} != {run_id}"
            )

        # ----------------------------------------------------
        # Authoritative successful attempt
        # ----------------------------------------------------
        logical_status_path = (
            f"{logical_root}/logical_run_status.json"
        )

        logical_status = rclone_json(logical_status_path)

        if logical_status.get("run_id") != run_id:
            raise RuntimeError(
                f"logical_run_status run_id mismatch: "
                f"{logical_status.get('run_id')} != {run_id}"
            )

        if logical_status.get("training_status") != "CLOSED_VERIFIED":
            raise RuntimeError(
                f"training_status is not CLOSED_VERIFIED: "
                f"{logical_status.get('training_status')}"
            )

        if logical_status.get("training_completed") is not True:
            raise RuntimeError(
                "training_completed != true"
            )

        if logical_status.get("training_valid") is not True:
            raise RuntimeError(
                "training_valid != true"
            )

        if logical_status.get("eligible_for_final_test") is not True:
            raise RuntimeError(
                "eligible_for_final_test != true"
            )

        attempt_id = logical_status.get("attempt_used")

        if (
            not attempt_id
            or not str(attempt_id).startswith("attempt_")
        ):
            raise RuntimeError(
                f"Invalid attempt_used={attempt_id}"
            )

        # ----------------------------------------------------
        # Attempt-level run manifest
        # ----------------------------------------------------
        run_manifest_path = (
            f"{logical_root}/{attempt_id}/run_manifest.json"
        )

        rm = rclone_json(run_manifest_path)

        if rm.get("run_id") != run_id:
            raise RuntimeError(
                f"attempt run_id mismatch: "
                f"{rm.get('run_id')} != {run_id}"
            )

        if rm.get("attempt_id") != attempt_id:
            raise RuntimeError(
                f"attempt_id mismatch: "
                f"{rm.get('attempt_id')} != {attempt_id}"
            )

        # MAIN LOW-LABEL ONLY
        if rm.get("condition_role") != "MAIN_LOW_LABEL":
            continue

        method = rm.get("method")

        if method not in {"SUP", "SSL"}:
            raise RuntimeError(
                f"Unexpected MAIN_LOW_LABEL method={method}"
            )

        budget = str(rm.get("budget"))

        if budget not in {
            "1pct",
            "5pct",
            "10pct",
            "20pct",
        }:
            raise RuntimeError(
                f"Unexpected MAIN_LOW_LABEL budget={budget}"
            )

        selected.append({
            "run_id": run_id,
            "attempt_id": attempt_id,
            "method": method,
            "architecture": rm["architecture"],
            "budget": budget,
            "seed_index": int(rm["seed_index"]),
            "training_seed": int(rm["training_seed"]),
            "logical_root": logical_root,
        })

    except Exception as e:
        discovery_errors.append({
            "run_id": run_id,
            "error": str(e),
        })


method_counts = Counter(
    x["method"]
    for x in selected
)

print()
print("=== MAIN LOW-LABEL GATE ===")
print(f"MAIN_LOW_LABEL={len(selected)}")
print(f"SUP={method_counts.get('SUP', 0)}")
print(f"SSL={method_counts.get('SSL', 0)}")
print(f"DISCOVERY_ERRORS={len(discovery_errors)}")

if discovery_errors:
    print(json.dumps(
        discovery_errors,
        indent=2,
        ensure_ascii=False
    ))
    raise RuntimeError(
        "Discovery errors detected. Stop."
    )

if len(selected) != EXPECTED_MAIN_LOW_LABEL:
    raise RuntimeError(
        f"Expected 160 MAIN_LOW_LABEL runs, "
        f"found {len(selected)}"
    )

if method_counts.get("SUP") != EXPECTED_SUP:
    raise RuntimeError(
        f"Expected SUP=80, "
        f"found {method_counts.get('SUP')}"
    )

if method_counts.get("SSL") != EXPECTED_SSL:
    raise RuntimeError(
        f"Expected SSL=80, "
        f"found {method_counts.get('SSL')}"
    )


# ============================================================
# 4. CHECK UNIQUE SCIENTIFIC CELLS
# ============================================================

cell_keys = [
    (
        x["method"],
        x["architecture"],
        x["budget"],
        x["seed_index"],
        x["training_seed"],
    )
    for x in selected
]

if len(cell_keys) != len(set(cell_keys)):
    raise RuntimeError(
        "Duplicate method×architecture×budget×seed cell"
    )

print("UNIQUE_CAMPAIGN_CELLS=PASS")


# ============================================================
# 5. EXTRACT FP COUNT PER NEGATIVE IMAGE
# ============================================================

selected = sorted(
    selected,
    key=lambda x: (
        x["architecture"],
        x["budget"],
        x["seed_index"],
        x["method"],
    )
)

detail_rows = []
run_rows = []

for idx, meta in enumerate(selected, start=1):

    det_path = (
        f'{meta["logical_root"]}/'
        f'{meta["attempt_id"]}/'
        f'evaluation/final_test/detections.json'
    )

    print(
        f"[{idx:03d}/160] "
        f'{meta["run_id"]} '
        f'{meta["attempt_id"]}'
    )

    detections = rclone_json(det_path)

    fp_counter = Counter()

    raw_negative_detections = 0
    accepted_negative_detections = 0

    max_negative_score = None
    max_all_score = None

    for d in detections:

        image_id = int(d["image_id"])
        score = float(d["score"])

        if (
            max_all_score is None
            or score > max_all_score
        ):
            max_all_score = score

        if image_id in zero_gt_ids:

            raw_negative_detections += 1

            if (
                max_negative_score is None
                or score > max_negative_score
            ):
                max_negative_score = score

            if score >= TAU:
                fp_counter[image_id] += 1
                accepted_negative_detections += 1

    counts = [
        fp_counter.get(image_id, 0)
        for image_id in negative_ids
    ]

    sorted_counts = sorted(counts)

    # Standard linear percentile-style descriptive quantiles
    def percentile(values, p):
        n = len(values)

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

    q1 = percentile(sorted_counts, 0.25)
    median = percentile(sorted_counts, 0.50)
    q3 = percentile(sorted_counts, 0.75)

    fp_total = sum(counts)

    any_fp_count = sum(
        1
        for c in counts
        if c > 0
    )

    fp_per_negative = (
        fp_total / EXPECTED_NEGATIVE_IMAGES
    )

    far_negative = (
        any_fp_count / EXPECTED_NEGATIVE_IMAGES
    )

    for image_id in negative_ids:

        im = image_meta[image_id]

        detail_rows.append({
            "run_id": meta["run_id"],
            "attempt_id": meta["attempt_id"],
            "method": meta["method"],
            "architecture": meta["architecture"],
            "budget": meta["budget"],
            "seed_index": meta["seed_index"],
            "training_seed": meta["training_seed"],
            "image_id": image_id,
            "file_name": im.get("file_name"),
            "fp_count_tau_0_5":
                fp_counter.get(image_id, 0),
        })

    distribution = Counter(counts)

    run_rows.append({
        "run_id": meta["run_id"],
        "attempt_id": meta["attempt_id"],
        "method": meta["method"],
        "architecture": meta["architecture"],
        "budget": meta["budget"],
        "seed_index": meta["seed_index"],
        "training_seed": meta["training_seed"],

        "n_negative_images":
            EXPECTED_NEGATIVE_IMAGES,

        "tau_eval": TAU,

        "fp_total_negative_reconstructed":
            fp_total,

        "fp_per_negative_reconstructed":
            fp_per_negative,

        "negative_far_reconstructed":
            far_negative,

        "negative_images_with_any_fp":
            any_fp_count,

        "negative_images_zero_fp":
            distribution.get(0, 0),

        "fp_count_min":
            min(counts),

        "fp_count_q1":
            q1,

        "fp_count_median":
            median,

        "fp_count_q3":
            q3,

        "fp_count_max":
            max(counts),

        "raw_negative_detection_count":
            raw_negative_detections,

        "accepted_negative_detection_count":
            accepted_negative_detections,

        "max_negative_raw_score":
            max_negative_score,

        "max_all_raw_score":
            max_all_score,

        "detections_remote_path":
            det_path,
    })


# ============================================================
# 6. FINAL ROW-COUNT GATE
# ============================================================

if len(detail_rows) != EXPECTED_DETAIL_ROWS:
    raise RuntimeError(
        f"Expected {EXPECTED_DETAIL_ROWS} "
        f"detail rows, got {len(detail_rows)}"
    )

if len(run_rows) != EXPECTED_MAIN_LOW_LABEL:
    raise RuntimeError(
        f"Expected 160 run-summary rows, "
        f"got {len(run_rows)}"
    )


# ============================================================
# 7. WRITE OUTPUTS
# ============================================================

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

detail_fields = [
    "run_id",
    "attempt_id",
    "method",
    "architecture",
    "budget",
    "seed_index",
    "training_seed",
    "image_id",
    "file_name",
    "fp_count_tau_0_5",
]

with OUT_DETAIL.open(
    "w",
    newline="",
    encoding="utf-8",
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=detail_fields,
    )

    writer.writeheader()
    writer.writerows(detail_rows)


run_fields = list(
    run_rows[0].keys()
)

with OUT_RUN.open(
    "w",
    newline="",
    encoding="utf-8",
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=run_fields,
    )

    writer.writeheader()
    writer.writerows(run_rows)


audit = {
    "artifact_role":
        "SUPPLEMENTARY_DESCRIPTIVE_FP_DISTRIBUTION",

    "source_predictions":
        "RETAINED_OFFICIAL_FINAL_TEST_DETECTIONS",

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

    "tau_eval":
        TAU,

    "remote_root":
        REMOTE_ROOT,

    "test_json":
        str(TEST_JSON),

    "test_image_count":
        len(images),

    "test_annotation_count":
        len(annotations),

    "negative_definition":
        (
            "zero_GT == "
            "is_negative_true == "
            "scope_label_no_finding"
        ),

    "negative_image_count":
        len(negative_ids),

    "official_run_directories":
        len(run_ids),

    "selected_main_low_label_runs":
        len(selected),

    "method_counts":
        dict(method_counts),

    "detail_rows":
        len(detail_rows),

    "run_summary_rows":
        len(run_rows),

    "expected_detail_rows":
        EXPECTED_DETAIL_ROWS,

    "discovery_errors":
        discovery_errors,

    "outputs": {
        "detail":
            str(OUT_DETAIL),

        "run_summary":
            str(OUT_RUN),
    },
}

with OUT_AUDIT.open(
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        audit,
        f,
        indent=2,
        ensure_ascii=False,
    )


# ============================================================
# 8. FINAL OUTPUT
# ============================================================

print()
print("======================================")
print("FINAL FP DISTRIBUTION EXTRACTION GATE")
print("======================================")

print(
    f"OFFICIAL_RUN_DIRS={len(run_ids)}"
)

print(
    f"MAIN_LOW_LABEL_RUNS={len(selected)}"
)

print(
    f"SUP_RUNS={method_counts['SUP']}"
)

print(
    f"SSL_RUNS={method_counts['SSL']}"
)

print(
    f"NEGATIVE_IMAGES={len(negative_ids)}"
)

print(
    f"DETAIL_ROWS={len(detail_rows)}"
)

print(
    f"RUN_SUMMARY_ROWS={len(run_rows)}"
)

print(
    f"EXPECTED_DETAIL_ROWS="
    f"{EXPECTED_DETAIL_ROWS}"
)

print(
    f"DETAIL_OUT={OUT_DETAIL}"
)

print(
    f"RUN_SUMMARY_OUT={OUT_RUN}"
)

print(
    f"AUDIT_OUT={OUT_AUDIT}"
)

print(
    "FP_DISTRIBUTION_EXTRACTION=PASS"
)
