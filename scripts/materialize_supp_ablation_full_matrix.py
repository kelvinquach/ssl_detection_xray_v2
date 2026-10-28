import csv
import json
from pathlib import Path

ROOT = Path(r"D:\ssl_detection_xray_v2")
SRC = ROOT / "analysis" / "ablation" / "ablation_variant_summary.csv"

OUTDIR = ROOT / "thesis_artifacts" / "supplementary"
OUTDIR.mkdir(parents=True, exist_ok=True)

OUT_CSV = OUTDIR / "supp_ablation_full_effect_matrix.csv"
OUT_TEX = OUTDIR / "supp_ablation_full_effect_matrix.tex"
OUT_AUDIT = OUTDIR / "supp_ablation_full_effect_matrix_audit.json"

VARIANT_ORDER = ["C0","C1","C2","C3","A0","A1","A2","F0","F1"]

METRIC_ORDER = [
    "validation_bbox_mAP_50_95",
    "validation_AP50",
    "validation_AP75",
    "Q_pseudo",
]

METRIC_LABEL = {
    "validation_bbox_mAP_50_95": r"$\Delta$mAP",
    "validation_AP50": r"$\Delta$AP50",
    "validation_AP75": r"$\Delta$AP75",
    "Q_pseudo": r"$\Delta Q_{\mathrm{pseudo}}$",
}

FAMILY_LABEL = {
    "CONFIDENCE_THRESHOLD": "Confidence",
    "STRONG_PHOTOMETRIC_AUGMENTATION": "Augmentation",
    "REGRESSION_PSEUDO_LABEL_FILTERING": "Regression filtering",
}

with SRC.open("r", encoding="utf-8", newline="") as f:
    rows = list(csv.DictReader(f))

# ------------------------------------------------------------
# 1. Structural gates
# ------------------------------------------------------------

assert len(rows) == 36

keys = [(r["variant_id"], r["metric_name"]) for r in rows]
assert len(set(keys)) == 36

assert set(r["variant_id"] for r in rows) == set(VARIANT_ORDER)
assert set(r["metric_name"] for r in rows) == set(METRIC_ORDER)

assert set(r["n_seeds"] for r in rows) == {"10"}
assert set(r["ci_level"] for r in rows) == {"0.95"}
assert set(r["formal_hypothesis_test"].lower() for r in rows) == {"false"}
assert set(r["p_value_computed"].lower() for r in rows) == {"false"}
assert set(r["multiplicity_family_added"].lower() for r in rows) == {"false"}

lookup = {
    (r["variant_id"], r["metric_name"]): r
    for r in rows
}

# ------------------------------------------------------------
# 2. Presentation-only conversion: raw AP -> AP points
# ------------------------------------------------------------

out_rows = []

for variant in VARIANT_ORDER:
    for metric in METRIC_ORDER:
        r = lookup[(variant, metric)]

        mean = float(r["paired_mean_difference"]) * 100.0
        sd = float(r["paired_difference_sample_SD_ddof1"]) * 100.0
        lo = float(r["paired_difference_two_sided_95CI_low"]) * 100.0
        hi = float(r["paired_difference_two_sided_95CI_high"]) * 100.0

        out_rows.append({
            "variant_id": variant,
            "family": r["family"],
            "family_display": FAMILY_LABEL[r["family"]],
            "level": r["level"],
            "metric_name": metric,
            "metric_display": METRIC_LABEL[metric],
            "n_seeds": 10,
            "paired_mean_difference_AP_points": mean,
            "paired_difference_SD_AP_points": sd,
            "CI95_low_AP_points": lo,
            "CI95_high_AP_points": hi,
            "CI_contains_zero": lo <= 0 <= hi,
            "formal_hypothesis_test": False,
            "p_value_computed": False,
        })

# ------------------------------------------------------------
# 3. Scientific pattern gates
# ------------------------------------------------------------

mAP_rows = [
    r for r in out_rows
    if r["metric_name"] == "validation_bbox_mAP_50_95"
]

ap75_rows = [
    r for r in out_rows
    if r["metric_name"] == "validation_AP75"
]

q_rows = [
    r for r in out_rows
    if r["metric_name"] == "Q_pseudo"
]

assert all(r["CI_contains_zero"] for r in mAP_rows)

ap75_below_zero = {
    r["variant_id"]
    for r in ap75_rows
    if r["CI95_high_AP_points"] < 0
}

q_above_zero = {
    r["variant_id"]
    for r in q_rows
    if r["CI95_low_AP_points"] > 0
}

assert ap75_below_zero == {"C0", "A2", "F1"}
assert q_above_zero == {"C0", "C1"}

# ------------------------------------------------------------
# 4. CSV
# ------------------------------------------------------------

with OUT_CSV.open("w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=list(out_rows[0].keys())
    )
    writer.writeheader()
    writer.writerows(out_rows)

# ------------------------------------------------------------
# 5. LaTeX — long form to avoid width overflow
# ------------------------------------------------------------

tex = []

tex.append(r"\begin{table*}[htbp]")
tex.append(r"\centering")
tex.append(
    r"\caption{Full descriptive ablation effect matrix on the validation set. "
    r"Effects are paired differences (variant minus the same frozen Main configuration) "
    r"across 10 training seeds, expressed in AP points. "
    r"SD denotes the sample standard deviation of paired differences and CI denotes the "
    r"two-sided 95\% confidence interval. These intervals are descriptive; no formal "
    r"hypothesis test or multiplicity adjustment was performed for individual ablation variants.}"
)
tex.append(r"\label{tab:supp_ablation_full_matrix}")
tex.append(r"\small")
tex.append(r"\setlength{\tabcolsep}{4pt}")
tex.append(r"\renewcommand{\arraystretch}{1.06}")
tex.append(r"\begin{tabular}{lllrrrr}")
tex.append(r"\toprule")
tex.append(
    r"\textbf{Family} & \textbf{Variant} & \textbf{Metric} & "
    r"\textbf{Mean $\Delta$} & \textbf{SD} & "
    r"\textbf{95\% CI low} & \textbf{95\% CI high} \\"
)
tex.append(r"\midrule")

last_family = None

for r in out_rows:
    family = r["family_display"]

    if last_family is not None and family != last_family:
        tex.append(r"\addlinespace")

    tex.append(
        f"{family} & "
        f"{r['variant_id']} & "
        f"{r['metric_display']} & "
        f"{r['paired_mean_difference_AP_points']:+.3f} & "
        f"{r['paired_difference_SD_AP_points']:.3f} & "
        f"{r['CI95_low_AP_points']:+.3f} & "
        f"{r['CI95_high_AP_points']:+.3f} \\\\"
    )

    last_family = family

tex.append(r"\bottomrule")
tex.append(r"\end{tabular}")
tex.append(r"\end{table*}")

OUT_TEX.write_text(
    "\n".join(tex) + "\n",
    encoding="utf-8"
)

# ------------------------------------------------------------
# 6. Audit artifact
# ------------------------------------------------------------

audit = {
    "source_artifact":
        "analysis/ablation/ablation_variant_summary.csv",

    "source_rows":
        36,

    "variants":
        VARIANT_ORDER,

    "metrics":
        METRIC_ORDER,

    "variant_metric_groups":
        36,

    "n_seeds":
        10,

    "ci_level":
        0.95,

    "sample_sd_ddof":
        1,

    "presentation_unit":
        "AP_points",

    "presentation_conversion":
        "canonical raw AP values multiplied by 100",

    "statistics_recomputed":
        False,

    "formal_hypothesis_test_added":
        False,

    "p_value_added":
        False,

    "multiplicity_family_added":
        False,

    "scientific_pattern_checks": {
        "mAP_CI_contains_zero_9_of_9":
            all(r["CI_contains_zero"] for r in mAP_rows),

        "AP75_CI_entirely_below_zero":
            sorted(ap75_below_zero),

        "Qpseudo_CI_entirely_above_zero":
            sorted(q_above_zero),
    },

    "outputs": {
        "csv": str(OUT_CSV),
        "latex": str(OUT_TEX),
    },

    "status":
        "PASS",
}

with OUT_AUDIT.open("w", encoding="utf-8") as f:
    json.dump(audit, f, ensure_ascii=False, indent=2)

print("======================================")
print("SUPPLEMENTARY ABLATION MATRIX GATE")
print("======================================")
print(f"SOURCE_ROWS={len(rows)}")
print(f"OUTPUT_ROWS={len(out_rows)}")
print("VARIANTS=9")
print("METRICS=4")
print("VARIANT_METRIC_GROUPS=36")
print("N_SEEDS=10")
print("CI_LEVEL=0.95")
print("MAP_CI_CONTAINS_ZERO=9/9")
print("AP75_CI_BELOW_ZERO=C0,A2,F1")
print("QPSEUDO_CI_ABOVE_ZERO=C0,C1")
print("FORMAL_HYPOTHESIS_TEST_ADDED=FALSE")
print("P_VALUE_ADDED=FALSE")
print("MULTIPLICITY_FAMILY_ADDED=FALSE")
print(f"CSV_OUT={OUT_CSV}")
print(f"LATEX_OUT={OUT_TEX}")
print(f"AUDIT_OUT={OUT_AUDIT}")
print("SUPPLEMENTARY_ABLATION_FULL_MATRIX=PASS")
