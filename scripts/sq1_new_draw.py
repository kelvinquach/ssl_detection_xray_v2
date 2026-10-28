import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import t

# =========================================================
# 1) ĐƯỜNG DẪN
# =========================================================
BASE = Path(r"D:\ssl_detection_xray_v2")
RQ3_DIR = BASE / "analysis" / "rq3"

BUDGET_CSV = RQ3_DIR / "rq3_budget_gain_by_seed.csv"
ANOVA_JSON = RQ3_DIR / "rq3_gg_anova.json"
CONTRAST_CSV = RQ3_DIR / "rq3_pairwise_contrasts.csv"

OUT_PNG = RQ3_DIR / "sq1_budget_effect_main.png"
OUT_PDF = RQ3_DIR / "sq1_budget_effect_main.pdf"

# =========================================================
# 2) KIỂM TRA FILE ĐẦU VÀO
# =========================================================
for path in [BUDGET_CSV, ANOVA_JSON, CONTRAST_CSV]:
    if not path.exists():
        raise FileNotFoundError(f"Không tìm thấy file: {path}")

# =========================================================
# 3) ĐỌC DỮ LIỆU
# =========================================================
df = pd.read_csv(BUDGET_CSV)
contrasts = pd.read_csv(CONTRAST_CSV)

with open(ANOVA_JSON, "r", encoding="utf-8") as f:
    anova = json.load(f)

# =========================================================
# 4) CHUẨN HÓA DỮ LIỆU SQ1
# =========================================================
budget_order = ["1%", "5%", "10%", "20%"]

required_cols = {
    "training_seed",
    "seed_index",
    "budget",
    "B_b_s",
}

missing = required_cols - set(df.columns)

if missing:
    raise ValueError(
        "File rq3_budget_gain_by_seed.csv thiếu các cột bắt buộc: "
        f"{sorted(missing)}"
    )

plot_df = (
    df[["training_seed", "seed_index", "budget", "B_b_s"]]
    .copy()
    .rename(columns={"B_b_s": "effect"})
)

plot_df["budget"] = pd.Categorical(
    plot_df["budget"],
    categories=budget_order,
    ordered=True,
)

plot_df = (
    plot_df
    .sort_values(["seed_index", "budget"])
    .reset_index(drop=True)
)

# =========================================================
# 5) KIỂM TRA SỐ DÒNG / SEED / BUDGET
# =========================================================
if len(plot_df) != 40:
    raise ValueError(
        f"Kỳ vọng 40 budget-seed rows (10 seeds × 4 budgets), "
        f"nhưng nhận được {len(plot_df)} rows."
    )

seed_count = plot_df["training_seed"].nunique()

if seed_count != 10:
    raise ValueError(
        f"Kỳ vọng 10 training seeds, nhưng nhận được {seed_count}."
    )

budget_count = plot_df["budget"].nunique()

if budget_count != 4:
    raise ValueError(
        f"Kỳ vọng 4 budgets, nhưng nhận được {budget_count}."
    )

# Mỗi seed phải có đủ 4 budget
rows_per_seed = plot_df.groupby("training_seed").size()

if not (rows_per_seed == 4).all():
    raise ValueError(
        "Có training seed không có đủ 4 budget.\n"
        f"{rows_per_seed}"
    )

# =========================================================
# 6) ĐỔI SANG AP POINTS
#
# Ví dụ:
# -0.01210 mAP -> -1.210 AP points
# =========================================================
plot_df["effect_ap_points"] = plot_df["effect"] * 100.0

# =========================================================
# 7) CROSS-CHECK CSV VỚI CANONICAL ANOVA JSON
# =========================================================
if "result" not in anova:
    raise ValueError("JSON không có trường 'result'.")

if "budget_means" not in anova["result"]:
    raise ValueError("JSON không có result.budget_means.")

means_from_csv = (
    plot_df
    .groupby("budget", observed=False)["effect"]
    .mean()
    .reindex(budget_order)
    .to_numpy()
)

means_from_json = np.array(
    anova["result"]["budget_means"],
    dtype=float,
)

if len(means_from_json) != 4:
    raise ValueError(
        f"JSON budget_means phải có 4 giá trị, "
        f"nhưng nhận được {len(means_from_json)}."
    )

if not np.allclose(
    means_from_csv,
    means_from_json,
    atol=1e-12,
    rtol=0,
):
    raise ValueError(
        "Mean từ rq3_budget_gain_by_seed.csv KHÔNG khớp "
        "rq3_gg_anova.json.\n\n"
        f"CSV : {means_from_csv}\n"
        f"JSON: {means_from_json}"
    )

print("PASS: budget means trong CSV khớp canonical ANOVA JSON.")

# =========================================================
# 8) TÍNH MEAN + DESCRIPTIVE 95% CI
#
# CI này dùng để trực quan hóa variability trên 10 training seeds.
# Đây KHÔNG phải Holm-adjusted simultaneous CI.
# =========================================================
summary = (
    plot_df
    .groupby("budget", observed=False)["effect_ap_points"]
    .agg(["mean", "std", "count"])
    .reindex(budget_order)
)

summary["se"] = summary["std"] / np.sqrt(summary["count"])

df_ci = int(summary["count"].iloc[0] - 1)

tcrit = t.ppf(
    0.975,
    df=df_ci,
)

summary["ci_low"] = (
    summary["mean"] - tcrit * summary["se"]
)

summary["ci_high"] = (
    summary["mean"] + tcrit * summary["se"]
)

summary = summary.reset_index()

# =========================================================
# 9) ĐỌC THỐNG KÊ OMNIBUS TỪ JSON
# =========================================================
res = anova["result"]

required_json_fields = [
    "f_statistic",
    "df_gg_numerator",
    "df_gg_denominator",
    "epsilon_gg",
    "p_gg",
    "n_subjects",
]

missing_json = [
    field
    for field in required_json_fields
    if field not in res
]

if missing_json:
    raise ValueError(
        f"JSON thiếu các trường: {missing_json}"
    )

f_stat = float(res["f_statistic"])
df_num = float(res["df_gg_numerator"])
df_den = float(res["df_gg_denominator"])
eps_gg = float(res["epsilon_gg"])
p_gg = float(res["p_gg"])
n_subjects = int(res["n_subjects"])

if n_subjects != 10:
    raise ValueError(
        f"Canonical JSON ghi n_subjects={n_subjects}, kỳ vọng 10."
    )

# =========================================================
# 10) TẠO HÌNH
# =========================================================
fig, ax = plt.subplots(
    figsize=(9.0, 5.8)
)

# ---------------------------------------------------------
# 10.1 Seed-level trajectories
#
# Mỗi đường biểu diễn cùng một training seed qua 4 budgets.
# Các đường được làm mờ vì mục tiêu chính của hình là mean pattern.
# ---------------------------------------------------------
for training_seed, g in plot_df.groupby(
    "training_seed",
    sort=False,
):
    g = g.sort_values("budget")

    ax.plot(
        g["budget"],
        g["effect_ap_points"],
        marker="o",
        linewidth=1.0,
        markersize=4,
        alpha=0.22,
    )

# ---------------------------------------------------------
# 10.2 Mean + descriptive 95% CI
# ---------------------------------------------------------
yerr = np.vstack(
    [
        summary["mean"] - summary["ci_low"],
        summary["ci_high"] - summary["mean"],
    ]
)

ax.errorbar(
    x=summary["budget"],
    y=summary["mean"],
    yerr=yerr,
    marker="o",
    linewidth=2.5,
    markersize=7,
    capsize=5,
    capthick=1.5,
    zorder=5,
)

# ---------------------------------------------------------
# 10.3 Mốc null effect
# ---------------------------------------------------------
ax.axhline(
    y=0,
    linestyle="--",
    linewidth=1.2,
)

# ---------------------------------------------------------
# 10.4 Mean labels
# ---------------------------------------------------------
for _, row in summary.iterrows():
    ax.annotate(
        f"{row['mean']:.3f}",
        xy=(
            row["budget"],
            row["mean"],
        ),
        xytext=(0, 10),
        textcoords="offset points",
        ha="center",
        va="bottom",
        fontsize=10,
    )

# ---------------------------------------------------------
# 10.5 Axis labels
# ---------------------------------------------------------
ax.set_xlabel(
    "Tỷ lệ ảnh có bounding-box annotation",
    fontsize=11,
)

ax.set_ylabel(
    "SSL–SUP effect (AP points)",
    fontsize=11,
)

ax.set_title(
    "SQ1 – Sự thay đổi của SSL–SUP effect theo labeled-data budget",
    fontsize=12,
)

# ---------------------------------------------------------
# 10.6 Statistical annotation
#
# Chỉ sử dụng trực tiếp các giá trị từ canonical ANOVA JSON.
# Không hard-code Holm-adjusted p-value vào hình.
# ---------------------------------------------------------
stats_text = (
    "Greenhouse–Geisser RM-ANOVA\n"
    f"F({df_num:.3f}, {df_den:.3f}) = {f_stat:.3f}\n"
    f"ε_GG = {eps_gg:.3f}, p = {p_gg:.2e}"
)

ax.text(
    0.02,
    0.03,
    stats_text,
    transform=ax.transAxes,
    fontsize=9,
    ha="left",
    va="bottom",
    bbox={
        "boxstyle": "round,pad=0.35",
        "alpha": 0.12,
    },
)

# ---------------------------------------------------------
# 10.7 Grid nhẹ để dễ đọc
# ---------------------------------------------------------
ax.grid(
    axis="y",
    linestyle=":",
    linewidth=0.8,
    alpha=0.35,
)

# Khoảng trống phía trên để thấy mốc 0 rõ
current_bottom, current_top = ax.get_ylim()

ax.set_ylim(
    min(current_bottom, -2.0),
    max(current_top, 0.15),
)

plt.tight_layout()

# =========================================================
# 11) LƯU HÌNH
# =========================================================
RQ3_DIR.mkdir(
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

print()
print("=== OUTPUT ===")
print(f"PNG: {OUT_PNG.resolve()}")
print(f"PDF: {OUT_PDF.resolve()}")

# =========================================================
# 12) IN SUMMARY ĐỂ ĐỐI CHIẾU MANUSCRIPT
# =========================================================
print()
print("=== Budget-level effect summary ===")

display_summary = summary[
    [
        "budget",
        "mean",
        "std",
        "count",
        "ci_low",
        "ci_high",
    ]
].copy()

display_summary = display_summary.rename(
    columns={
        "budget": "Budget",
        "mean": "Mean_AP_points",
        "std": "SD_AP_points",
        "count": "n",
        "ci_low": "CI95_low",
        "ci_high": "CI95_high",
    }
)

print(
    display_summary.to_string(
        index=False,
        float_format=lambda x: f"{x:.6f}",
    )
)

# =========================================================
# 13) KIỂM TRA 4 MEAN QUAN TRỌNG
# =========================================================
expected_means_ap = np.array(
    [
        -1.210,
        -0.615,
        -1.045,
        -1.580,
    ]
)

observed_means_ap = (
    summary["mean"]
    .to_numpy(dtype=float)
)

if not np.allclose(
    observed_means_ap,
    expected_means_ap,
    atol=1e-9,
    rtol=0,
):
    raise ValueError(
        "Mean AP points không khớp các giá trị canonical "
        "đang báo cáo trong manuscript.\n\n"
        f"Observed: {observed_means_ap}\n"
        f"Expected: {expected_means_ap}"
    )

print()
print(
    "PASS: 4 budget-level means = "
    "-1.210, -0.615, -1.045, -1.580 AP points."
)

# =========================================================
# 14) ĐỌC VÀ KIỂM TRA PAIRWISE CONTRAST FILE
# =========================================================
required_contrast_cols = {
    "contrast",
    "effect_estimate",
    "ci_low",
    "ci_high",
    "raw_p_value",
    "multiplicity_family",
}

missing_contrast = (
    required_contrast_cols
    - set(contrasts.columns)
)

if missing_contrast:
    raise ValueError(
        "rq3_pairwise_contrasts.csv thiếu các cột: "
        f"{sorted(missing_contrast)}"
    )

contrast_view = contrasts[
    [
        "contrast",
        "effect_estimate",
        "ci_low",
        "ci_high",
        "raw_p_value",
        "multiplicity_family",
    ]
].copy()

# mAP -> AP points
for col in [
    "effect_estimate",
    "ci_low",
    "ci_high",
]:
    contrast_view[col] = (
        contrast_view[col] * 100.0
    )

contrast_view = contrast_view.rename(
    columns={
        "contrast": "Contrast",
        "effect_estimate": "Difference_AP_points",
        "ci_low": "CI95_low",
        "ci_high": "CI95_high",
        "raw_p_value": "Raw_p",
        "multiplicity_family": "Family",
    }
)

print()
print("=== Prespecified pairwise contrasts ===")

print(
    contrast_view.to_string(
        index=False,
        float_format=lambda x: f"{x:.6g}",
    )
)

# =========================================================
# 15) KẾT THÚC
# =========================================================
print()
print("DONE.")