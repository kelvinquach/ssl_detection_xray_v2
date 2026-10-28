from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import t


# =========================================================
# 1) PATHS
# =========================================================

BASE = Path(r"D:\ssl_detection_xray_v2")
RQ2_DIR = BASE / "analysis" / "rq2"

CELL_CSV = RQ2_DIR / "rq2_cell_descriptives.csv"
GAIN_CSV = RQ2_DIR / "rq2_gain_by_seed.csv"

OUT_PNG = RQ2_DIR / "mainrq_ssl_sup_effect.png"
OUT_PDF = RQ2_DIR / "mainrq_ssl_sup_effect.pdf"


# =========================================================
# 2) CHECK INPUT FILES
# =========================================================

for path in [CELL_CSV, GAIN_CSV]:
    if not path.exists():
        raise FileNotFoundError(
            f"Không tìm thấy file:\n{path}"
        )


# =========================================================
# 3) LOAD CANONICAL DATA
# =========================================================

cell_df = pd.read_csv(CELL_CSV)
gain_df = pd.read_csv(GAIN_CSV)

print("CELL columns:")
print(cell_df.columns.tolist())

print("\nGAIN columns:")
print(gain_df.columns.tolist())


# =========================================================
# 4) VALIDATE CELL-LEVEL FILE
# =========================================================

required_cell_cols = {
    "architecture",
    "budget",
    "mean",
    "mean_two_sided_95CI_low",
    "mean_two_sided_95CI_high",
}

missing = required_cell_cols - set(cell_df.columns)

if missing:
    raise ValueError(
        "rq2_cell_descriptives.csv thiếu cột:\n"
        f"{sorted(missing)}"
    )

# Nếu có nhiều metric, chỉ lấy primary bbox mAP
if "metric_name" in cell_df.columns:
    metric_values = cell_df["metric_name"].astype(str).unique()

    print("\nmetric_name values:")
    print(metric_values)

    # File canonical hiện tại dùng primary detection metric.
    # Tự tìm metric có mAP.
    candidates = [
        x for x in metric_values
        if "map" in x.lower()
    ]

    if len(candidates) == 1:
        primary_metric = candidates[0]
        cell_df = cell_df[
            cell_df["metric_name"] == primary_metric
        ].copy()

        print(
            f"\nUsing metric_name = {primary_metric}"
        )

# Kỳ vọng đúng 8 cells
if len(cell_df) != 8:
    raise ValueError(
        f"Kỳ vọng 8 architecture×budget cells, "
        f"nhưng nhận được {len(cell_df)}."
    )


# =========================================================
# 5) NORMALIZE ARCHITECTURE / BUDGET LABELS
# =========================================================

budget_map = {
    "1pct": "1%",
    "5pct": "5%",
    "10pct": "10%",
    "20pct": "20%",
    "1%": "1%",
    "5%": "5%",
    "10%": "10%",
    "20%": "20%",
}

architecture_map = {
    "R50-FPN": "R50-FPN",
    "R50": "R50-FPN",
    "Swin-T-FPN": "Swin-T-FPN",
    "Swin-T": "Swin-T-FPN",
}

cell_df["budget_label"] = (
    cell_df["budget"]
    .astype(str)
    .map(budget_map)
)

cell_df["architecture_label"] = (
    cell_df["architecture"]
    .astype(str)
    .map(architecture_map)
)

if cell_df["budget_label"].isna().any():
    raise ValueError(
        "Có budget chưa được map:\n"
        f"{cell_df[['budget']].drop_duplicates()}"
    )

if cell_df["architecture_label"].isna().any():
    raise ValueError(
        "Có architecture chưa được map:\n"
        f"{cell_df[['architecture']].drop_duplicates()}"
    )


# =========================================================
# 6) ORDER CELLS
# =========================================================

architecture_order = [
    "R50-FPN",
    "Swin-T-FPN",
]

budget_order = [
    "1%",
    "5%",
    "10%",
    "20%",
]

cell_order = [
    ("R50-FPN", "1%"),
    ("R50-FPN", "5%"),
    ("R50-FPN", "10%"),
    ("R50-FPN", "20%"),
    ("Swin-T-FPN", "1%"),
    ("Swin-T-FPN", "5%"),
    ("Swin-T-FPN", "10%"),
    ("Swin-T-FPN", "20%"),
]

cell_df["architecture_label"] = pd.Categorical(
    cell_df["architecture_label"],
    categories=architecture_order,
    ordered=True,
)

cell_df["budget_label"] = pd.Categorical(
    cell_df["budget_label"],
    categories=budget_order,
    ordered=True,
)

cell_df = (
    cell_df
    .sort_values(
        ["architecture_label", "budget_label"]
    )
    .reset_index(drop=True)
)

observed_order = list(
    zip(
        cell_df["architecture_label"].astype(str),
        cell_df["budget_label"].astype(str),
    )
)

if observed_order != cell_order:
    raise ValueError(
        "Thứ tự architecture×budget không đúng.\n"
        f"Observed: {observed_order}"
    )


# =========================================================
# 7) CONVERT CELL EFFECTS TO AP POINTS
#
# canonical file dùng mAP scale:
# -0.0147 -> -1.47 AP points
# =========================================================

cell_df["mean_ap"] = (
    cell_df["mean"] * 100.0
)

cell_df["ci_low_ap"] = (
    cell_df["mean_two_sided_95CI_low"] * 100.0
)

cell_df["ci_high_ap"] = (
    cell_df["mean_two_sided_95CI_high"] * 100.0
)


# =========================================================
# 8) EXTRACT G_s FROM rq2_gain_by_seed.csv
#
# Canonical contract:
# file chứa đúng 10 seed-level G_s values.
# =========================================================

if len(gain_df) != 10:
    raise ValueError(
        f"rq2_gain_by_seed.csv phải có đúng 10 rows, "
        f"nhưng nhận được {len(gain_df)}."
    )

# Ưu tiên cột tên G_s.
candidate_names = [
    "G_s",
    "g_s",
    "overall_gain",
    "overall_effect",
    "effect",
    "gain",
]

g_col = None

for col in candidate_names:
    if col in gain_df.columns:
        g_col = col
        break

# Fallback:
# tìm cột numeric không phải seed/index có tên chứa G/gain/effect.
if g_col is None:
    numeric_cols = gain_df.select_dtypes(
        include=[np.number]
    ).columns.tolist()

    excluded = {
        "seed",
        "seed_index",
        "training_seed",
        "index",
    }

    candidates = [
        c for c in numeric_cols
        if c not in excluded
        and (
            "gain" in c.lower()
            or "effect" in c.lower()
            or c.lower() in {"g", "g_s"}
        )
    ]

    if len(candidates) == 1:
        g_col = candidates[0]

if g_col is None:
    raise ValueError(
        "Không tự nhận diện được cột G_s trong "
        "rq2_gain_by_seed.csv.\n"
        f"Các cột hiện có: {gain_df.columns.tolist()}\n"
        "Hãy đặt tên cột seed-level overall effect là G_s "
        "hoặc bổ sung tên cột vào candidate_names."
    )

print(f"\nUsing G_s column: {g_col}")

G = gain_df[g_col].astype(float).to_numpy()


# =========================================================
# 9) COMPUTE OVERALL G_s MEAN + 95% CI FROM REAL SEED DATA
# =========================================================

n = len(G)

overall_mean = G.mean()
overall_sd = G.std(ddof=1)
overall_se = overall_sd / np.sqrt(n)

tcrit = t.ppf(
    0.975,
    df=n - 1,
)

overall_ci_low = (
    overall_mean
    - tcrit * overall_se
)

overall_ci_high = (
    overall_mean
    + tcrit * overall_se
)

# convert to AP points
overall_mean_ap = overall_mean * 100.0
overall_ci_low_ap = overall_ci_low * 100.0
overall_ci_high_ap = overall_ci_high * 100.0
overall_sd_ap = overall_sd * 100.0


# =========================================================
# 10) SCIENTIFIC CROSS-CHECKS
#
# Đây chỉ là guardrail.
# Hình vẫn được vẽ từ dữ liệu thật phía trên.
# =========================================================

if not np.isclose(
    overall_mean_ap,
    -1.11,
    atol=0.01,
):
    raise ValueError(
        "Overall mean không khớp manuscript.\n"
        f"Observed = {overall_mean_ap:.4f} AP points"
    )

if not np.isclose(
    overall_ci_low_ap,
    -1.18,
    atol=0.02,
):
    raise ValueError(
        "Overall CI low không khớp manuscript.\n"
        f"Observed = {overall_ci_low_ap:.4f}"
    )

if not np.isclose(
    overall_ci_high_ap,
    -1.05,
    atol=0.02,
):
    raise ValueError(
        "Overall CI high không khớp manuscript.\n"
        f"Observed = {overall_ci_high_ap:.4f}"
    )

if not (G < 0).all():
    raise ValueError(
        "Không phải 10/10 G_s đều âm."
    )

print("\nPASS: Overall G_s khớp manuscript.")
print(
    f"Mean = {overall_mean_ap:.4f} AP points"
)
print(
    f"SD   = {overall_sd_ap:.4f} AP points"
)
print(
    f"95% CI = "
    f"[{overall_ci_low_ap:.4f}, "
    f"{overall_ci_high_ap:.4f}]"
)
print(
    f"Negative seeds = {(G < 0).sum()}/{n}"
)


# =========================================================
# 11) PREPARE FIGURE DATA
# =========================================================

labels = [
    "R50-FPN — 1%",
    "R50-FPN — 5%",
    "R50-FPN — 10%",
    "R50-FPN — 20%",
    "Swin-T-FPN — 1%",
    "Swin-T-FPN — 5%",
    "Swin-T-FPN — 10%",
    "Swin-T-FPN — 20%",
    r"Overall $G_s$",
]

means = cell_df["mean_ap"].tolist()
ci_lows = cell_df["ci_low_ap"].tolist()
ci_highs = cell_df["ci_high_ap"].tolist()

means.append(overall_mean_ap)
ci_lows.append(overall_ci_low_ap)
ci_highs.append(overall_ci_high_ap)

means = np.array(means)
ci_lows = np.array(ci_lows)
ci_highs = np.array(ci_highs)

lower_errors = means - ci_lows
upper_errors = ci_highs - means


# =========================================================
# 12) DRAW FIGURE
# =========================================================

fig, ax = plt.subplots(
    figsize=(10.5, 7.4)
)

y = np.arange(len(labels))

# matplotlib default color cycle
colors = plt.rcParams[
    "axes.prop_cycle"
].by_key()["color"]

# 8 cell colors
for i in range(8):

    ax.errorbar(
        x=means[i],
        y=y[i],
        xerr=np.array(
            [
                [lower_errors[i]],
                [upper_errors[i]],
            ]
        ),
        fmt="o",
        markersize=8,
        linewidth=2,
        capsize=5,
        color=colors[i % len(colors)],
        markerfacecolor=colors[i % len(colors)],
        markeredgecolor=colors[i % len(colors)],
        zorder=3,
    )


# =========================================================
# 13) OVERALL G_s
# =========================================================

overall_i = 8

ax.errorbar(
    x=means[overall_i],
    y=y[overall_i],
    xerr=np.array(
        [
            [lower_errors[overall_i]],
            [upper_errors[overall_i]],
        ]
    ),
    fmt="D",
    markersize=10,
    linewidth=2.4,
    capsize=5,
    color=colors[8 % len(colors)],
    markerfacecolor=colors[8 % len(colors)],
    markeredgecolor=colors[8 % len(colors)],
    zorder=4,
)


# =========================================================
# 14) ZERO REFERENCE LINE
# =========================================================

ax.axvline(
    x=0,
    linewidth=1.4,
    color=colors[0],
)


# =========================================================
# 15) AXES / LABELS
# =========================================================

ax.set_yticks(y)
ax.set_yticklabels(
    labels,
    fontsize=14,
)

ax.invert_yaxis()

ax.set_xlabel(
    "Hiệu ứng SSL − SUP (AP points)",
    fontsize=14,
)

# Đã sửa nhãn RQ2 cũ thành Main RQ
ax.set_title(
    "Main RQ: Hiệu ứng SSL so với supervised baseline",
    fontsize=17,
    pad=12,
)

ax.grid(
    axis="x",
    linestyle="-",
    linewidth=0.8,
    alpha=0.18,
)

ax.grid(
    axis="y",
    visible=False,
)


# =========================================================
# 16) X LIMITS
# =========================================================

xmin = min(ci_lows.min(), -1.8)
xmax = max(0.05, ci_highs.max() + 0.05)

ax.set_xlim(
    xmin,
    xmax,
)


# =========================================================
# 17) SPINES / LAYOUT
# =========================================================

for spine in ax.spines.values():
    spine.set_linewidth(1.0)

plt.tight_layout()


# =========================================================
# 18) SAVE
# =========================================================

RQ2_DIR.mkdir(
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
# 19) AUDIT TABLE
# =========================================================

print("\n=== CELL-LEVEL EFFECTS ===")

audit = cell_df[
    [
        "architecture_label",
        "budget_label",
        "mean_ap",
        "ci_low_ap",
        "ci_high_ap",
    ]
].copy()

audit.columns = [
    "Architecture",
    "Budget",
    "Mean_AP_points",
    "CI95_low",
    "CI95_high",
]

print(
    audit.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}",
    )
)


# =========================================================
# 20) FINAL CHECKS
# =========================================================

# 8/8 cell means negative
if not (
    cell_df["mean_ap"] < 0
).all():
    raise ValueError(
        "Không phải 8/8 cell-level means đều âm."
    )

# 8/8 descriptive CIs below zero
if not (
    cell_df["ci_high_ap"] < 0
).all():
    raise ValueError(
        "Không phải 8/8 cell-level 95% CI đều dưới 0."
    )

print()
print(
    "PASS: 8/8 architecture×budget cell means âm."
)

print(
    "PASS: 8/8 descriptive cell-level 95% CI "
    "nằm hoàn toàn dưới 0."
)

print(
    "PASS: 10/10 overall seed-level G_s âm."
)


# =========================================================
# 21) OUTPUT
# =========================================================

print("\n=== OUTPUT ===")
print(OUT_PNG.resolve())
print(OUT_PDF.resolve())

print("\nDONE.")
