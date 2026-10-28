import csv
import json
import shutil
import hashlib
from pathlib import Path

ROOT = Path(r"D:\ssl_detection_xray_v2")

SRC = ROOT / "reports" / "03A_bbox_count_per_image.csv"

OUTDIR = ROOT / "thesis_artifacts" / "supplementary"
OUTDIR.mkdir(parents=True, exist_ok=True)

OUT_CSV = OUTDIR / "supp_bbox_density.csv"
OUT_TEX = OUTDIR / "supp_bbox_density.tex"
OUT_AUDIT = OUTDIR / "supp_bbox_density_audit.json"

EXPECTED_SHA = "4f7773ffcff76721d5a74b840af0d37c880d779c58937d404e93910b07802e98"

def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

# ------------------------------------------------------------
# 1. Provenance
# ------------------------------------------------------------

assert SRC.exists()
assert sha256(SRC) == EXPECTED_SHA

with SRC.open("r", encoding="utf-8", newline="") as f:
    rows = list(csv.DictReader(f))

assert len(rows) == 2

expected_columns = [
    "population",
    "n",
    "mean",
    "sd_ddof1",
    "median",
    "p95",
    "max",
]

assert list(rows[0].keys()) == expected_columns

by_population = {r["population"]: r for r in rows}

assert set(by_population) == {
    "all_train_images",
    "positive_train_images_only",
}

all_row = by_population["all_train_images"]
pos_row = by_population["positive_train_images_only"]

assert int(all_row["n"]) == 3426
assert int(pos_row["n"]) == 3076

# Existing canonical values only; no recomputation of statistics.
assert float(all_row["mean"]) == 7.373029772329247
assert float(all_row["median"]) == 6.0
assert float(all_row["p95"]) == 17.0
assert float(all_row["max"]) == 48.0

assert float(pos_row["mean"]) == 8.211963589076722
assert float(pos_row["median"]) == 7.0
assert float(pos_row["p95"]) == 18.0
assert float(pos_row["max"]) == 48.0

# Structural consistency only.
assert abs(float(all_row["mean"]) * int(all_row["n"]) - 25260) < 1e-8
assert abs(float(pos_row["mean"]) * int(pos_row["n"]) - 25260) < 1e-8

# ------------------------------------------------------------
# 2. Byte reuse
# ------------------------------------------------------------

shutil.copyfile(SRC, OUT_CSV)
assert sha256(OUT_CSV) == EXPECTED_SHA

# ------------------------------------------------------------
# 3. LaTeX
# ------------------------------------------------------------

tex = []

tex.append(r"\begin{table}[htbp]")
tex.append(r"\centering")
tex.append(
    r"\caption{Bounding-box annotation density in the fixed training set. "
    r"$B_i$ denotes the number of bounding-box annotation records associated "
    r"with training image $i$. Sample standard deviation uses $ddof=1$ and "
    r"the 95th percentile follows the locked Phase~3A percentile definition.}"
)
tex.append(r"\label{tab:supp_bbox_density}")
tex.append(r"\small")
tex.append(r"\setlength{\tabcolsep}{5pt}")
tex.append(r"\renewcommand{\arraystretch}{1.05}")
tex.append(r"\begin{tabular}{lrrrrrr}")
tex.append(r"\toprule")
tex.append(
    r"\textbf{Population} & "
    r"\textbf{$n$} & "
    r"\textbf{Mean} & "
    r"\textbf{SD} & "
    r"\textbf{Median} & "
    r"\textbf{P95} & "
    r"\textbf{Max} \\"
)
tex.append(r"\midrule")

display = {
    "all_train_images": "All training images",
    "positive_train_images_only": "Positive training images only",
}

for key in ["all_train_images", "positive_train_images_only"]:
    r = by_population[key]

    tex.append(
        f"{display[key]} & "
        f"{int(r['n'])} & "
        f"{float(r['mean']):.2f} & "
        f"{float(r['sd_ddof1']):.2f} & "
        f"{float(r['median']):.0f} & "
        f"{float(r['p95']):.0f} & "
        f"{float(r['max']):.0f} \\\\"
    )

tex.append(r"\bottomrule")
tex.append(r"\end{tabular}")
tex.append(r"\end{table}")

OUT_TEX.write_text("\n".join(tex) + "\n", encoding="utf-8")

# ------------------------------------------------------------
# 4. Audit
# ------------------------------------------------------------

audit = {
    "artifact_role": "SUPPLEMENTARY_BOUNDING_BOX_DENSITY",
    "source_artifact": "reports/03A_bbox_count_per_image.csv",
    "source_sha256": EXPECTED_SHA,

    "row_count": 2,

    "populations": {
        "all_train_images": {
            "n": 3426,
            "mean": float(all_row["mean"]),
            "sd_ddof1": float(all_row["sd_ddof1"]),
            "median": float(all_row["median"]),
            "p95": float(all_row["p95"]),
            "max": float(all_row["max"]),
        },
        "positive_train_images_only": {
            "n": 3076,
            "mean": float(pos_row["mean"]),
            "sd_ddof1": float(pos_row["sd_ddof1"]),
            "median": float(pos_row["median"]),
            "p95": float(pos_row["p95"]),
            "max": float(pos_row["max"]),
        },
    },

    "bounding_box_annotation_total_consistency": 25260,
    "statistics_recomputed": False,
    "source_csv_byte_reused": True,

    "interpretation_scope":
        "descriptive pre-training dataset characterization",

    "independent_lesion_claim": False,

    "status": "PASS",
}

with OUT_AUDIT.open("w", encoding="utf-8") as f:
    json.dump(audit, f, indent=2)

print("======================================")
print("SUPPLEMENTARY BBOX DENSITY GATE")
print("======================================")
print("SOURCE_SHA256=PASS")
print("ROWS=2")
print("ALL_TRAIN_N=3426")
print("ALL_TRAIN_MEAN=7.373029772329247")
print("ALL_TRAIN_MEDIAN=6")
print("ALL_TRAIN_P95=17")
print("ALL_TRAIN_MAX=48")
print("POSITIVE_TRAIN_N=3076")
print("POSITIVE_TRAIN_MEAN=8.211963589076722")
print("POSITIVE_TRAIN_MEDIAN=7")
print("POSITIVE_TRAIN_P95=18")
print("POSITIVE_TRAIN_MAX=48")
print("BBOX_ANNOTATION_TOTAL_CONSISTENCY=25260")
print("CSV_BYTE_IDENTITY=PASS")
print("STATISTICS_RECOMPUTED=FALSE")
print(f"CSV_OUT={OUT_CSV}")
print(f"LATEX_OUT={OUT_TEX}")
print(f"AUDIT_OUT={OUT_AUDIT}")
print("SUPPLEMENTARY_BBOX_DENSITY=PASS")
