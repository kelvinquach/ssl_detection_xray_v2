import csv
import json
import shutil
from pathlib import Path

ROOT = Path(r"D:\ssl_detection_xray_v2")

SRC_CSV = ROOT / "reports" / "03A_class_distribution.csv"
SRC_PNG = ROOT / "plots" / "dataset" / "class_distribution.png"

OUTDIR = ROOT / "thesis_artifacts" / "supplementary"
OUTDIR.mkdir(parents=True, exist_ok=True)

OUT_CSV = OUTDIR / "supp_class_distribution.csv"
OUT_TEX = OUTDIR / "supp_class_distribution.tex"
OUT_PNG = OUTDIR / "supp_class_distribution.png"
OUT_AUDIT = OUTDIR / "supp_class_distribution_audit.json"

EXPECTED_CSV_SHA = "59a0bfee13831412e3afa0ec5adb7b4bc1a6f0947be88562b9862087a112b7fa"
EXPECTED_PNG_SHA = "f12b146ae7a341983f6dea6f0bd3070d806ce3324c0acdc133901d3ac2d6acb0"

import hashlib

def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

# provenance gate
assert sha256(SRC_CSV) == EXPECTED_CSV_SHA
assert sha256(SRC_PNG) == EXPECTED_PNG_SHA

with SRC_CSV.open("r", encoding="utf-8", newline="") as f:
    rows = list(csv.DictReader(f))

assert len(rows) == 14

category_ids = sorted(int(r["category_id"]) for r in rows)
assert category_ids == list(range(1, 15))

# Sort only the in-memory presentation copy; canonical CSV remains untouched.
rows_sorted = sorted(rows, key=lambda r: int(r["category_id"]))

rare = sorted(
    r["class_name"]
    for r in rows
    if r["rare_flag"].lower() == "true"
)
assert rare == ["Atelectasis", "Pneumothorax"]

bbox_total = sum(int(r["bbox_annotation_count"]) for r in rows)
assert bbox_total == 25260

# byte-copy canonical sources into supplementary
shutil.copyfile(SRC_CSV, OUT_CSV)
shutil.copyfile(SRC_PNG, OUT_PNG)

assert sha256(OUT_CSV) == EXPECTED_CSV_SHA
assert sha256(OUT_PNG) == EXPECTED_PNG_SHA

# LaTeX table
tex = []
tex.append(r"\begin{table*}[htbp]")
tex.append(r"\centering")
tex.append(
    r"\caption{Full 14-class distribution in the fixed training set. "
    r"Image prevalence is computed over 3,426 training images. "
    r"Low-support status is defined operationally as image-level prevalence below 5\%; "
    r"this is not a definition of clinical rarity.}"
)
tex.append(r"\label{tab:supp_class_distribution}")
tex.append(r"\small")
tex.append(r"\setlength{\tabcolsep}{5pt}")
tex.append(r"\renewcommand{\arraystretch}{1.05}")
tex.append(r"\begin{tabular}{lrrrrr}")
tex.append(r"\toprule")
tex.append(
    r"\textbf{Class} & "
    r"\textbf{Images} & "
    r"\textbf{Prevalence (\%)} & "
    r"\textbf{BBox count} & "
    r"\textbf{BBox / positive image} & "
    r"\textbf{Low-support} \\"
)
tex.append(r"\midrule")

for r in rows_sorted:
    cls = r["class_name"].replace("&", r"\&")
    prevalence_pct = float(r["image_prevalence"]) * 100.0
    low = "Yes" if r["rare_flag"].lower() == "true" else "No"

    tex.append(
        f"{cls} & "
        f"{int(r['image_count']):d} & "
        f"{prevalence_pct:.2f} & "
        f"{int(r['bbox_annotation_count']):d} & "
        f"{float(r['bbox_per_positive_image']):.2f} & "
        f"{low} \\\\"
    )

tex.append(r"\bottomrule")
tex.append(r"\end{tabular}")
tex.append(r"\end{table*}")

OUT_TEX.write_text("\n".join(tex) + "\n", encoding="utf-8")

audit = {
    "source_csv": "reports/03A_class_distribution.csv",
    "source_csv_sha256": EXPECTED_CSV_SHA,
    "source_figure": "plots/dataset/class_distribution.png",
    "source_figure_sha256": EXPECTED_PNG_SHA,
    "row_count": 14,
    "category_ids_complete_1_to_14": True,
    "bbox_annotation_count_total": bbox_total,
    "low_support_definition": "image-level prevalence < 5% in fixed training set",
    "low_support_classes": rare,
    "clinical_rarity_claim": False,
    "statistics_recomputed": False,
    "source_csv_byte_reused": True,
    "source_figure_byte_reused": True,
    "status": "PASS",
}

with OUT_AUDIT.open("w", encoding="utf-8") as f:
    json.dump(audit, f, ensure_ascii=False, indent=2)

print("======================================")
print("SUPPLEMENTARY CLASS DISTRIBUTION GATE")
print("======================================")
print("ROWS=14")
print("CATEGORY_IDS=1..14")
print("BBOX_ANNOTATION_TOTAL=25260")
print("LOW_SUPPORT_CLASSES=Atelectasis,Pneumothorax")
print("CSV_BYTE_IDENTITY=PASS")
print("PNG_BYTE_IDENTITY=PASS")
print("STATISTICS_RECOMPUTED=FALSE")
print(f"CSV_OUT={OUT_CSV}")
print(f"LATEX_OUT={OUT_TEX}")
print(f"PNG_OUT={OUT_PNG}")
print(f"AUDIT_OUT={OUT_AUDIT}")
print("SUPPLEMENTARY_CLASS_DISTRIBUTION=PASS")


