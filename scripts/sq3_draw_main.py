import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =========================================================
# 1. PATHS
# =========================================================

BASE = Path(r"D:\ssl_detection_xray_v2")

RQ7_DIR = BASE / "analysis" / "rq7"
STATS_DIR = BASE / "analysis" / "multiplicity"

# Canonical SQ3 / old RQ7 artifacts
ARCH_SEED_CSV = RQ7_DIR / "rq7_architecture_gain_by_seed.csv"
PRIMARY_JSON = RQ7_DIR / "rq7_primary_test.json"

# Holm family F2
HOLM_F2_CSV = STATS_DIR / "holm_F2.csv"

# Output
OUT_PNG = RQ7_DIR / "sq3_architecture_effect_main.png"
OUT_PDF = RQ7_DIR / "sq3_architecture_effect_main.pdf"


# =========================================================
# 2. CHECK INPUT FILES
# =========================================================

required_files = [
    ARCH_SEED_CSV,
    PRIMARY_JSON,
    HOLM_F2_CSV,
]

for path in required_files:
    if not path.exists():
        raise FileNotFoundError(
            f"Không tìm thấy file:\n{path}"
        )


# =========================================================
# 3. LOAD DATA
# =========================================================

df = pd.read_csv(ARCH_SEED_CSV)

with open(PRIMARY_JSON, "r", encoding="utf-8") as f:
    primary = json.load(f)

holm = pd.read_csv(HOLM_F2_CSV)


# =========================================================
# 4. CHECK CSV SCHEMA
# =========================================================

required_cols = {
    "seed_index",
    "training_seed",
    "A_R50_s",
    "A_Swin-T_s",
    "D_s",
}

missing = required_cols - set(df.columns)

if missing:
    raise ValueError(
        "rq7_architecture_gain_by_seed.csv thiếu cột:\n"
        f"{sorted(missing)}"
    )


# =========================================================
# 5. CHECK NUMBER OF SEEDS
# =========================================================

df = df.sort_values("seed_index").reset_index(drop=True)

if len(df) != 10:
    raise ValueError(
        f"Kỳ vọng 10 training seeds, nhận được {len(df)}."
    )

if df["seed_index"].tolist() != list(range(1, 11)):
    raise ValueError(
        "seed_index không phải 1..10 theo thứ tự canonical."
    )


# =========================================================
# 6. VERIFY D_s
#
# D_s = A_Swin-T,s - A_R50,s
# =========================================================

recomputed_D = df["A_Swin-T_s"] - df["A_R50_s"]

if not np.allclose(
    recomputed_D.to_numpy(),
    df["D_s"].to_numpy(),
    atol=1e-12,
    rtol=0,
):
    raise ValueError(
        "D_s trong CSV không khớp "
        "A_Swin-T_s - A_R50_s."
    )

print("PASS: D_s = A_Swin-T_s - A_R50_s")


# =========================================================
# 7. LOAD PRIMARY RESULT
# =========================================================

if "primary_result" not in primary:
    raise ValueError(
        "rq7_primary_test.json không có primary_result."
    )

res = primary["primary_result"]

required_json_fields = [
    "effect_estimate",
    "sample_sd",
    "ci_low",
    "ci_high",
    "t_statistic",
    "df",
    "p_value",
    "n",
]

missing_json = [
    x for x in required_json_fields
    if x not in res
]

if missing_json:
    raise ValueError(
        f"Primary JSON thiếu: {missing_json}"
    )


# =========================================================
# 8. EXTRACT PRIMARY VALUES
# =========================================================

mean_D = float(res["effect_estimate"])
sd_D = float(res["sample_sd"])
ci_low = float(res["ci_low"])
ci_high = float(res["ci_high"])
t_stat = float(res["t_statistic"])
df_test = int(res["df"])
raw_p = float(res["p_value"])
n = int(res["n"])

if n != 10:
    raise ValueError(
        f"Primary JSON có n={n}, kỳ vọng n=10."
    )


# =========================================================
# 9. VERIFY PRIMARY MEAN AGAINST SEED CSV
# =========================================================

mean_D_csv = df["D_s"].mean()

if not np.isclose(
    mean_D_csv,
    mean_D,
    atol=1e-12,
    rtol=0,
):
    raise ValueError(
        "Mean D_s từ CSV không khớp primary JSON.\n"
        f"CSV  = {mean_D_csv}\n"
        f"JSON = {mean_D}"
    )

print("PASS: mean D_s khớp primary JSON")


# =========================================================
# 10. LOAD HOLM F2 RESULT FOR SQ3 / OLD RQ7
# =========================================================

required_holm_cols = {
    "family",
    "member",
    "raw_p_value",
    "adjusted_p_value",
}

missing_holm = required_holm_cols - set(holm.columns)

if missing_holm:
    raise ValueError(
        f"holm_F2.csv thiếu cột: {sorted(missing_holm)}"
    )

rq7_holm = holm[
    (holm["family"] == "F2")
    & (holm["member"] == "RQ7")
]

if len(rq7_holm) != 1:
    raise ValueError(
        "Không tìm thấy đúng 1 dòng RQ7 trong Holm F2."
    )

holm_p = float(
    rq7_holm.iloc[0]["adjusted_p_value"]
)

holm_raw_p = float(
    rq7_holm.iloc[0]["raw_p_value"]
)

if not np.isclose(
    holm_raw_p,
    raw_p,
    atol=1e-12,
    rtol=0,
):
    raise ValueError(
        "Raw p trong Holm F2 không khớp primary JSON."
    )

print("PASS: Holm F2 mapping khớp primary test")


# =========================================================
# 11. CONVERT TO AP POINTS
#
# Canonical files use mAP unit.
# Manuscript reports AP points:
# 0.001 = 0.1 AP point
# =========================================================

df["R50_AP"] = df["A_R50_s"] * 100.0
df["SWIN_AP"] = df["A_Swin-T_s"] * 100.0
df["D_AP"] = df["D_s"] * 100.0

mean_D_AP = mean_D * 100.0
ci_low_AP = ci_low * 100.0
ci_high_AP = ci_high * 100.0

mean_R50_AP = df["R50_AP"].mean()
mean_SWIN_AP = df["SWIN_AP"].mean()


# =========================================================
# 12. CANONICAL VALUE CHECKS
# =========================================================

expected_R50 = -1.1525
expected_SWIN = -1.0725
expected_D = 0.0800

if not np.isclose(
    mean_R50_AP,
    expected_R50,
    atol=1e-12,
):
    raise ValueError(
        f"R50 mean không khớp: {mean_R50_AP}"
    )

if not np.isclose(
    mean_SWIN_AP,
    expected_SWIN,
    atol=1e-12,
):
    raise ValueError(
        f"Swin mean không khớp: {mean_SWIN_AP}"
    )

if not np.isclose(
    mean_D_AP,
    expected_D,
    atol=1e-12,
):
    raise ValueError(
        f"D mean không khớp: {mean_D_AP}"
    )

print(
    "PASS: canonical means = "
    f"R50 {mean_R50_AP:.4f}, "
    f"Swin {mean_SWIN_AP:.4f}, "
    f"D {mean_D_AP:.4f} AP points"
)


# =========================================================
# 13. FIGURE LAYOUT
# =========================================================

fig, (ax1, ax2) = plt.subplots(
    1,
    2,
    figsize=(14, 6),
    gridspec_kw={
        "width_ratios": [1.35, 1.0]
    }
)

seed_x = df["seed_index"].to_numpy()


# =========================================================
# 14. PANEL A
# Architecture-specific SSL-SUP effects
# =========================================================

# paired connecting lines
for _, row in df.iterrows():
    x = row["seed_index"]

    ax1.plot(
        [x - 0.12, x + 0.12],
        [row["R50_AP"], row["SWIN_AP"]],
        linewidth=1.0,
        alpha=0.35,
        color="0.55",
        zorder=1,
    )

# R50
ax1.scatter(
    seed_x - 0.12,
    df["R50_AP"],
    marker="o",
    s=60,
    label=f"R50-FPN (mean {mean_R50_AP:.3f})",
    zorder=3,
)

# Swin
ax1.scatter(
    seed_x + 0.12,
    df["SWIN_AP"],
    marker="s",
    s=60,
    label=f"Swin-T-FPN (mean {mean_SWIN_AP:.3f})",
    zorder=3,
)

# null line
ax1.axhline(
    0,
    linewidth=1.0,
    color="0.25",
)

# architecture means
ax1.axhline(
    mean_R50_AP,
    linestyle="--",
    linewidth=1.2,
    alpha=0.9,
)

ax1.axhline(
    mean_SWIN_AP,
    linestyle="--",
    linewidth=1.2,
    alpha=0.9,
)

ax1.set_title(
    "(a) Hiệu ứng theo điều kiện kiến trúc",
    loc="left",
    fontsize=12,
)

ax1.set_xlabel(
    "Seed huấn luyện"
)

ax1.set_ylabel(
    "Hiệu ứng SSL−SUP (điểm AP)"
)

ax1.set_xticks(
    seed_x
)

ax1.set_xlim(
    0.4,
    10.6
)

# keep zero visible while retaining detail
panel_a_min = min(
    df["R50_AP"].min(),
    df["SWIN_AP"].min()
) - 0.12

ax1.set_ylim(
    panel_a_min,
    0.15
)

ax1.legend(
    loc="lower left",
    frameon=False,
    ncol=2,
)

ax1.spines["top"].set_visible(False)
ax1.spines["right"].set_visible(False)


# =========================================================
# 15. PANEL B
# Architecture-difference effect D_s
# =========================================================

# individual seed differences
ax2.bar(
    seed_x,
    df["D_AP"],
    width=0.60,
    alpha=0.75,
)

# zero line
ax2.axhline(
    0,
    linewidth=1.0,
    color="0.25",
)

# mean + individual two-sided 95% CI
summary_x = 11.4

lower_err = mean_D_AP - ci_low_AP
upper_err = ci_high_AP - mean_D_AP

ax2.errorbar(
    summary_x,
    mean_D_AP,
    yerr=np.array(
        [[lower_err], [upper_err]]
    ),
    fmt="D",
    markersize=7,
    capsize=6,
    linewidth=1.5,
    color="black",
    zorder=5,
)

# summary annotation
summary_text = (
    f"{mean_D_AP:+.3f}\n"
    f"[{ci_low_AP:+.3f}; {ci_high_AP:+.3f}]"
)

ax2.annotate(
    summary_text,
    xy=(summary_x, mean_D_AP),
    xytext=(0, 52),
    textcoords="offset points",
    ha="center",
    va="bottom",
    fontsize=10,
)

ax2.set_title(
    "(b) Chênh lệch hiệu ứng giữa hai kiến trúc",
    loc="left",
    fontsize=12,
)

ax2.set_xlabel(
    "Seed huấn luyện"
)

ax2.set_ylabel(
    "D = ΔSwin-T − ΔR50 (điểm AP)"
)

ax2.set_xticks(
    list(seed_x) + [summary_x]
)

ax2.set_xticklabels(
    [str(x) for x in seed_x]
    + ["Mean\n95% CI"]
)

ax2.set_xlim(
    0.3,
    12.0
)

panel_b_min = min(
    df["D_AP"].min(),
    ci_low_AP
) - 0.08

panel_b_max = max(
    df["D_AP"].max(),
    ci_high_AP
) + 0.08

ax2.set_ylim(
    panel_b_min,
    panel_b_max
)

ax2.spines["top"].set_visible(False)
ax2.spines["right"].set_visible(False)


# =========================================================
# 16. OPTIONAL STATISTICAL TEXT
#
# Giữ ngắn để hình không trở thành "bảng thống kê".
# =========================================================

stats_text = (
    f"Mean D = {mean_D_AP:+.3f} AP\n"
    f"95% CI [{ci_low_AP:+.3f}; {ci_high_AP:+.3f}]\n"
    f"t({df_test}) = {t_stat:.3f}, "
    f"p = {raw_p:.3f}; "
    f"Holm F2 p = {holm_p:.3f}"
)

ax2.text(
    0.03,
    0.97,
    stats_text,
    transform=ax2.transAxes,
    ha="left",
    va="top",
    fontsize=9,
    bbox={
        "boxstyle": "round,pad=0.35",
        "alpha": 0.10,
    },
)


# =========================================================
# 17. GLOBAL FORMATTING
# =========================================================

fig.suptitle(
    "SQ3 – Hiệu ứng SSL−SUP theo điều kiện kiến trúc",
    fontsize=14,
    y=1.02,
)

plt.tight_layout()


# =========================================================
# 18. SAVE
# =========================================================

RQ7_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

fig.savefig(
    OUT_PNG,
    dpi=300,
    bbox_inches="tight",
)

fig.savefig(
    OUT_PDF,
    bbox_inches="tight",
)

plt.close(fig)


# =========================================================
# 19. PRINT AUDIT SUMMARY
# =========================================================

print()
print("=== SQ3 PRIMARY FIGURE AUDIT ===")

print(
    f"R50-FPN mean effect  : "
    f"{mean_R50_AP:.4f} AP points"
)

print(
    f"Swin-T-FPN mean effect: "
    f"{mean_SWIN_AP:.4f} AP points"
)

print(
    f"Mean D_s             : "
    f"{mean_D_AP:+.4f} AP points"
)

print(
    f"95% CI               : "
    f"[{ci_low_AP:+.4f}, {ci_high_AP:+.4f}]"
)

print(
    f"t({df_test})           : "
    f"{t_stat:.6f}"
)

print(
    f"Raw p                 : "
    f"{raw_p:.12f}"
)

print(
    f"Holm F2 p             : "
    f"{holm_p:.12f}"
)

positive = int((df["D_s"] > 0).sum())
negative = int((df["D_s"] < 0).sum())
zero = int((df["D_s"] == 0).sum())

print(
    f"D_s direction        : "
    f"{positive} positive / "
    f"{negative} negative / "
    f"{zero} zero"
)

print(
    f"D_s observed range   : "
    f"[{df['D_AP'].min():+.3f}, "
    f"{df['D_AP'].max():+.3f}] AP points"
)

print()
print("=== OUTPUT ===")
print(OUT_PNG.resolve())
print(OUT_PDF.resolve())

print()
print("DONE.")
