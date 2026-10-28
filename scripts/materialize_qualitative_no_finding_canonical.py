import csv
import hashlib
import json
import shutil
from pathlib import Path

ROOT = Path(r"D:\ssl_detection_xray_v2")
Q = ROOT / "thesis_artifacts" / "qualitative"

CASE_MANIFEST = Q / "qualitative_case_selection_manifest.csv"
PRED_AUDIT = Q / "qualitative_selected_prediction_audit.json"

PREVIEW_PNG = Q / "qualitative_no_finding_preview.png"
PREVIEW_PDF = Q / "qualitative_no_finding_preview.pdf"

FINAL_PNG = Q / "qualitative_no_finding.png"
FINAL_PDF = Q / "qualitative_no_finding.pdf"

FIG_MANIFEST = Q / "qualitative_figure_manifest.csv"
RECON_CHECK = Q / "qualitative_reconstruction_check.json"

GT_PATH = ROOT / "data" / "processed" / "coco" / "instances_test.json"

SUP_DET = (
    ROOT / ".fp_distribution_staging" / "detections"
    / "OFF_SUP_R50_B01_S01" / "attempt_008"
    / "evaluation" / "final_test" / "detections.json"
)

SSL_DET = (
    ROOT / ".fp_distribution_staging" / "detections"
    / "OFF_SSL_R50_B01_S01" / "attempt_002"
    / "evaluation" / "final_test" / "detections.json"
)

TAU = 0.5
CASE_IDS = [202, 1272, 3015]

EXPECTED = {
    CASE_MANIFEST:
        "c42f2c22bdfb61cb7bda6e97f113152810a909a3e6f095896f82a8b4a9791a04",
    PRED_AUDIT:
        "102f17051cf020be29dc87d61a447ce20d5296b8d2eb9de8315b6b2f40f1ee5b",
    PREVIEW_PNG:
        "85e41a5896203a997c961886b26586530b0c8f96e2a5ec8f3120c7463e495e84",
    PREVIEW_PDF:
        "b9b782f11db640c24f7aba1c582b2a4176afad93d63776dbcc6161a82ab0f61f",
    GT_PATH:
        "e1a73110e92af2656276d6c532035afe474b6ac6f2b2f03849834c036e1c00a4",
    SUP_DET:
        "89ca0b618312c9d7eeae52f627af4f94a5678ae6e6a137395b13d5f6feb8fc81",
    SSL_DET:
        "7298b997962878c4ebd56f0424e5f5a06b880715782855701184d5026fb03af9",
}


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


# ============================================================
# 1. INPUT PROVENANCE GATE
# ============================================================

print("=== INPUT PROVENANCE GATE ===")

for path, expected_hash in EXPECTED.items():

    if not path.exists():
        raise SystemExit(f"FAIL: missing {path}")

    actual = sha256(path)

    ok = actual == expected_hash

    print(
        f"{'PASS' if ok else 'FAIL'} "
        f"{path.name} "
        f"{actual}"
    )

    if not ok:
        raise SystemExit(
            f"FAIL: SHA mismatch for {path}"
        )

print("INPUT_PROVENANCE_GATE=PASS")


# ============================================================
# 2. BYTE-COPY LOCKED PREVIEW → FINAL CANONICAL FIGURE
# ============================================================

shutil.copyfile(PREVIEW_PNG, FINAL_PNG)
shutil.copyfile(PREVIEW_PDF, FINAL_PDF)

png_final_sha = sha256(FINAL_PNG)
pdf_final_sha = sha256(FINAL_PDF)

if png_final_sha != EXPECTED[PREVIEW_PNG]:
    raise SystemExit(
        "FAIL: final PNG not byte-identical to locked preview"
    )

if pdf_final_sha != EXPECTED[PREVIEW_PDF]:
    raise SystemExit(
        "FAIL: final PDF not byte-identical to locked preview"
    )

print()
print("=== FINAL FIGURE BYTE IDENTITY ===")
print(f"PNG_SHA256={png_final_sha}")
print(f"PDF_SHA256={pdf_final_sha}")
print("FINAL_PNG_BYTE_IDENTICAL_TO_PREVIEW=PASS")
print("FINAL_PDF_BYTE_IDENTICAL_TO_PREVIEW=PASS")


# ============================================================
# 3. LOAD GT + DETECTIONS
# ============================================================

with GT_PATH.open("r", encoding="utf-8") as f:
    gt = json.load(f)

with SUP_DET.open("r", encoding="utf-8") as f:
    sup_det = json.load(f)

with SSL_DET.open("r", encoding="utf-8") as f:
    ssl_det = json.load(f)

categories = {
    int(c["id"]): c["name"]
    for c in gt.get("categories", [])
}

annotations_by_image = {}

for a in gt.get("annotations", []):
    annotations_by_image.setdefault(
        int(a["image_id"]), []
    ).append(a)

image_ids = {
    int(im["id"])
    for im in gt.get("images", [])
}

if len(gt.get("images", [])) != 734:
    raise SystemExit("FAIL: test image count != 734")

if len(gt.get("annotations", [])) != 5437:
    raise SystemExit("FAIL: test annotation count != 5437")


# ============================================================
# 4. INDEPENDENT RECONSTRUCTION
# ============================================================

def accepted_for_cases(detections):

    out = {image_id: [] for image_id in CASE_IDS}

    for d in detections:

        image_id = int(d["image_id"])

        if (
            image_id in out
            and float(d["score"]) >= TAU
        ):
            out[image_id].append({
                "category_id":
                    int(d["category_id"]),

                "category_name":
                    categories.get(
                        int(d["category_id"]),
                        None
                    ),

                "score":
                    float(d["score"]),

                "bbox":
                    d.get("bbox"),
            })

    for image_id in out:
        out[image_id].sort(
            key=lambda x: x["score"],
            reverse=True
        )

    return out


sup = accepted_for_cases(sup_det)
ssl = accepted_for_cases(ssl_det)

expected_sup_counts = {
    202: 2,
    1272: 0,
    3015: 1,
}

expected_ssl_counts = {
    202: 0,
    1272: 0,
    3015: 0,
}

case_results = []

for rank, image_id in enumerate(CASE_IDS, start=1):

    if image_id not in image_ids:
        raise SystemExit(
            f"FAIL: image_id {image_id} absent from test set"
        )

    gt_count = len(
        annotations_by_image.get(image_id, [])
    )

    if gt_count != 0:
        raise SystemExit(
            f"FAIL: image_id {image_id} is not zero-GT"
        )

    sup_count = len(sup[image_id])
    ssl_count = len(ssl[image_id])

    if sup_count != expected_sup_counts[image_id]:
        raise SystemExit(
            f"FAIL: SUP count mismatch for {image_id}: "
            f"{sup_count}"
        )

    if ssl_count != expected_ssl_counts[image_id]:
        raise SystemExit(
            f"FAIL: SSL count mismatch for {image_id}: "
            f"{ssl_count}"
        )

    case_results.append({
        "rank": rank,
        "image_id": image_id,
        "gt_annotation_count": gt_count,
        "no_finding_zero_gt": True,
        "sup_accepted_count": sup_count,
        "ssl_accepted_count": ssl_count,
        "sup_accepted_predictions": sup[image_id],
        "ssl_accepted_predictions": ssl[image_id],
    })


# ============================================================
# 5. WRITE RECONSTRUCTION CHECK
# ============================================================

recon = {
    "artifact_role":
        "QUALITATIVE_NO_FINDING_RECONSTRUCTION_CHECK",

    "scientific_role":
        "DESCRIPTIVE_QUALITATIVE_ILLUSTRATION_ONLY",

    "retraining_performed":
        False,

    "new_final_test_evaluation_performed":
        False,

    "case_reselection":
        False,

    "model_reselection":
        False,

    "threshold_tuning":
        False,

    "tau_eval":
        TAU,

    "selected_case_ids":
        CASE_IDS,

    "model_pair": {
        "architecture":
            "R50-FPN",

        "budget":
            "1pct",

        "seed_index":
            1,

        "training_seed":
            204886845,

        "sup_run_id":
            "OFF_SUP_R50_B01_S01",

        "sup_attempt_id":
            "attempt_008",

        "ssl_run_id":
            "OFF_SSL_R50_B01_S01",

        "ssl_attempt_id":
            "attempt_002",
    },

    "source_sha256": {
        "case_selection_manifest":
            EXPECTED[CASE_MANIFEST],

        "selected_prediction_audit":
            EXPECTED[PRED_AUDIT],

        "instances_test":
            EXPECTED[GT_PATH],

        "sup_detections":
            EXPECTED[SUP_DET],

        "ssl_detections":
            EXPECTED[SSL_DET],

        "locked_preview_png":
            EXPECTED[PREVIEW_PNG],

        "locked_preview_pdf":
            EXPECTED[PREVIEW_PDF],
    },

    "final_figure_sha256": {
        "png":
            png_final_sha,

        "pdf":
            pdf_final_sha,
    },

    "final_figure_byte_identical_to_locked_preview": {
        "png":
            True,

        "pdf":
            True,
    },

    "test_set": {
        "image_count":
            len(gt.get("images", [])),

        "annotation_count":
            len(gt.get("annotations", [])),
    },

    "cases":
        case_results,

    "expected_sup_counts":
        [2, 0, 1],

    "expected_ssl_counts":
        [0, 0, 0],

    "all_selected_cases_zero_gt":
        True,

    "accepted_prediction_counts_match_locked_preview":
        True,

    "qualitative_reconstruction_check":
        "PASS",
}

with RECON_CHECK.open(
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        recon,
        f,
        ensure_ascii=False,
        indent=2
    )


# ============================================================
# 6. WRITE FINAL FIGURE MANIFEST
# ============================================================

common = {
    "artifact_role":
        "FINAL_CANONICAL_QUALITATIVE_NO_FINDING_FIGURE",

    "scientific_role":
        "DESCRIPTIVE_QUALITATIVE_ILLUSTRATION_ONLY",

    "architecture":
        "R50-FPN",

    "budget":
        "1pct",

    "seed_index":
        1,

    "training_seed":
        204886845,

    "sup_run_id":
        "OFF_SUP_R50_B01_S01",

    "sup_attempt_id":
        "attempt_008",

    "ssl_run_id":
        "OFF_SSL_R50_B01_S01",

    "ssl_attempt_id":
        "attempt_002",

    "tau_eval":
        TAU,

    "selected_image_ids":
        "202;1272;3015",

    "sup_accepted_counts":
        "2;0;1",

    "ssl_accepted_counts":
        "0;0;0",

    "case_selection_manifest_sha256":
        EXPECTED[CASE_MANIFEST],

    "selected_prediction_audit_sha256":
        EXPECTED[PRED_AUDIT],

    "instances_test_sha256":
        EXPECTED[GT_PATH],

    "sup_detections_sha256":
        EXPECTED[SUP_DET],

    "ssl_detections_sha256":
        EXPECTED[SSL_DET],

    "case_reselection":
        False,

    "model_reselection":
        False,

    "threshold_tuning":
        False,

    "byte_identical_to_locked_preview":
        True,
}

rows = []

for fmt, final_path, preview_path in [
    ("PNG", FINAL_PNG, PREVIEW_PNG),
    ("PDF", FINAL_PDF, PREVIEW_PDF),
]:
    row = dict(common)

    row.update({
        "format":
            fmt,

        "final_path":
            str(final_path.relative_to(ROOT)),

        "source_preview_path":
            str(preview_path.relative_to(ROOT)),

        "source_preview_sha256":
            sha256(preview_path),

        "final_sha256":
            sha256(final_path),
    })

    rows.append(row)

with FIG_MANIFEST.open(
    "w",
    encoding="utf-8",
    newline=""
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=list(rows[0].keys())
    )

    writer.writeheader()
    writer.writerows(rows)


# ============================================================
# 7. FINAL GATE
# ============================================================

print()
print("======================================")
print("QUALITATIVE CANONICALIZATION GATE")
print("======================================")
print("CASE_IDS=202,1272,3015")
print("TAU_EVAL=0.5")
print("SUP_ACCEPTED_COUNTS=2,0,1")
print("SSL_ACCEPTED_COUNTS=0,0,0")
print("ALL_SELECTED_CASES_ZERO_GT=PASS")
print("CASE_RESELECTION=FALSE")
print("MODEL_RESELECTION=FALSE")
print("THRESHOLD_TUNING=FALSE")
print("FINAL_FIGURE_BYTE_IDENTITY=PASS")
print("QUALITATIVE_RECONSTRUCTION_CHECK=PASS")
print(f"FINAL_PNG={FINAL_PNG}")
print(f"FINAL_PDF={FINAL_PDF}")
print(f"FIGURE_MANIFEST={FIG_MANIFEST}")
print(f"RECONSTRUCTION_CHECK={RECON_CHECK}")
print("QUALITATIVE_CANONICALIZATION=PASS")
