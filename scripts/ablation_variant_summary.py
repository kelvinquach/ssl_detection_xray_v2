from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =========================================================
# 1) PATHS
# =========================================================

BASE = Path(r"D:\ssl_detection_xray_v2")

ABLATION_DIR = BASE / "analysis" / "ablation"

SUMMARY_CSV = ABLATION_DIR / "ablation_variant_summary.csv"

OUT_PNG = ABLATION_DIR / "z_ablation_forest_publication.png"
OUT_PDF = ABLATION_DIR / "z_ablation_forest_publication.pdf"


# =========================================================
# 2) CHECK INPUT
# =========================================================

if not SUMMARY_CSV.exists():
    raise FileNotFoundError(
        f"Không tìm thấy file:\n{SUMMARY_CSV}"
    )


# =========================================================
# 3) LOAD DATA
# =========================================================

df = pd.read_csv(SUMMARY_CSV)

required_cols = {
    "variant_id",
    "family",
    "level",
    "metric_name",
    "n_seeds",
    "paired_mean_difference",
    "paired_difference_two_sided_95CI_low",
    "paired_difference_two_sided_95CI_high",
}

missing = required_cols - set(df.columns)

if missing:
    raise ValueError(
        f"Thiếu các cột bắt buộc: {sorted(missing)}"
    )


# =========================================================
# 4) EXPECTED DESIGN
# =========================================================

variant_order = [
    "F1",
    "F0",
    "A2",
    "A1",
    "A0",
    "C3",
    "C2",
    "C1",
    "C0",
]

variant_labels = {
    "F1": "F1  confidence >0.90",
    "F0": "F0  no extra filter",
    "A2": "A2  contrast only",
    "A1": "A1  brightness only",
    "A0": "A0  no brightness/contrast",
    "C3": "C3  confidence 0.80",
    "C2": "C2  confidence 0.70",
    "C1": "C1  confidence 0.60",
    "C0": "C0  confidence 0.50",
}

metrics = [
    "validation_bbox_mAP_50_95",
    "validation_AP50",
    "validation_AP75",
    "Q_pseudo",
]

metric_titles = {
    "validation_bbox_mAP_50_95":
        r"$\Delta$ validation bbox mAP@[0.50:0.95]",

    "validation_AP50":
        r"$\Delta$ validation AP50",

    "validation_AP75":
        r"$\Delta$ validation AP75",

    "Q_pseudo":
        r"$\Delta Q_{\mathrm{pseudo}}$",
}


# =========================================================
# 5) VALIDATE DATA
# =========================================================

observed_variants = set(df["variant_id"].unique())

if observed_variants != set(variant_order):
    raise ValueError(
        "Danh sách variant không đúng kỳ vọng.\n"
        f"Observed: {sorted(observed_variants)}\n"
        f"Expected: {sorted(variant_order)}"
    )

for metric in metrics:
    sub = df[df["metric_name"] == metric]

    if len(sub) != 9:
        raise ValueError(
            f"Metric {metric} phải có 9 variants, "
            f"nhưng nhận được {len(sub)}."
        )

if not (df["n_seeds"] == 10).all():
    raise ValueError(
        "Có dòng không sử dụng đúng 10 training seeds."
    )


# =========================================================
# 6) CONVERT TO AP POINTS
#
# CSV lưu ở mAP scale.
#
# Ví dụ:
# -0.0012 -> -0.12 AP points
# =========================================================

df["mean_ap"] = (
    df["paired_mean_difference"] * 100.0
)

df["ci_low_ap"] = (
    df["paired_difference_two_sided_95CI_low"] * 100.0
)

df["ci_high_ap"] = (
    df["paired_difference_two_sided_95CI_high"] * 100.0
)


# =========================================================
# 7) CROSS-CHECK IMPORTANT VALUES
# =========================================================

def get_row(variant, metric):
    row = df[
        (df["variant_id"] == variant)
        & (df["metric_name"] == metric)
    ]

    if len(row) != 1:
        raise ValueError(
            f"Không tìm thấy đúng 1 dòng cho "
            f"{variant}, {metric}"
        )

    return row.iloc[0]


# ---- Primary mAP
expected_map = {
    "C0": -0.12,
    "C1": -0.10,
    "C2": -0.07,
    "C3":  0.00,
    "A0": -0.01,
    "A1":  0.04,
    "A2": -0.13,
    "F0": -0.09,
    "F1": -0.13,
}

for variant, expected in expected_map.items():

    observed = get_row(
        variant,
        "validation_bbox_mAP_50_95"
    )["mean_ap"]

    if not np.isclose(
        observed,
        expected,
        atol=0.005,
    ):
        raise ValueError(
            f"{variant} ΔmAP không khớp.\n"
            f"Observed={observed}, Expected={expected}"
        )


# ---- Qpseudo C0/C1
for variant, expected in {
    "C0": 0.26,
    "C1": 0.09,
}.items():

    observed = get_row(
        variant,
        "Q_pseudo"
    )["mean_ap"]

    if not np.isclose(
        observed,
        expected,
        atol=0.005,
    ):
        raise ValueError(
            f"{variant} ΔQpseudo không khớp.\n"
            f"Observed={observed}, Expected={expected}"
        )


# ---- AP75 key variants
for variant, expected in {
    "C0": -0.23,
    "A2": -0.37,
    "F1": -0.27,
}.items():

    observed = get_row(
        variant,
        "validation_AP75"
    )["mean_ap"]

    if not np.isclose(
        observed,
        expected,
        atol=0.005,
    ):
        raise ValueError(
            f"{variant} ΔAP75 không khớp.\n"
            f"Observed={observed}, Expected={expected}"
        )


print("PASS: các giá trị canonical chính khớp.")


# =========================================================
# 8) PREPARE FIGURE
# =========================================================

fig, axes = plt.subplots(
    1,
    4,
    figsize=(20, 5.4),
    sharey=True,
)

y_positions = np.arange(len(variant_order))

# reversed visually so F1 is top
y_lookup = {
    variant: i
    for i, variant in enumerate(variant_order)
}


# =========================================================
# 9) FAMILY COLORS
#
# Không gán màu thủ công theo mã hex.
# Dùng default matplotlib color cycle.
# =========================================================

default_colors = plt.rcParams[
    "axes.prop_cycle"
].by_key()["color"]

family_color = {
    "REGRESSION_PSEUDO_LABEL_FILTERING":
        default_colors[0],

    "STRONG_PHOTOMETRIC_AUGMENTATION":
        default_colors[1],

    "CONFIDENCE_THRESHOLD":
        default_colors[2],
}


# =========================================================
# 10) DRAW EACH PANEL
# =========================================================

for ax, metric in zip(axes, metrics):

    sub = (
        df[df["metric_name"] == metric]
        .copy()
        .set_index("variant_id")
        .loc[variant_order]
        .reset_index()
    )

    for i, row in sub.iterrows():

        variant = row["variant_id"]

        mean = row["mean_ap"]
        low = row["ci_low_ap"]
        high = row["ci_high_ap"]

        lower_error = mean - low
        upper_error = high - mean

        # CI does not contain 0
        ci_excludes_zero = (
            (low > 0)
            or
            (high < 0)
        )

        # Filled diamond:
        # descriptive CI does not contain zero.
        # This is NOT a significance marker.
        marker = (
            "D"
            if ci_excludes_zero
            else "o"
        )

        marker_face = (
            family_color[row["family"]]
            if ci_excludes_zero
            else "white"
        )

        ax.errorbar(
            x=mean,
            y=i,
            xerr=np.array(
                [[lower_error], [upper_error]]
            ),
            fmt=marker,
            markersize=6,
            capsize=4,
            linewidth=1.4,
            color=family_color[row["family"]],
            markerfacecolor=marker_face,
            markeredgecolor=family_color[row["family"]],
            zorder=3,
        )

    # Zero reference line
    ax.axvline(
        0,
        linestyle="--",
        linewidth=1.1,
        color=default_colors[0],
        alpha=0.8,
    )

    ax.set_title(
        metric_titles[metric],
        fontsize=12,
    )

    ax.set_xlabel(
        "Paired mean difference, variant − Main (AP points)",
        fontsize=10,
    )

    ax.grid(
        axis="x",
        linestyle=":",
        linewidth=0.7,
        alpha=0.25,
    )

    # Horizontal separators between families:
    # F group: F1/F0
    # A group: A2/A1/A0
    # C group: C3/C2/C1/C0
    ax.axhline(
        1.5,
        linewidth=0.8,
        alpha=0.20,
    )

    ax.axhline(
        4.5,
        linewidth=0.8,
        alpha=0.20,
    )


# =========================================================
# 11) Y LABELS
# =========================================================

axes[0].set_yticks(y_positions)

axes[0].set_yticklabels(
    [
        variant_labels[v]
        for v in variant_order
    ],
    fontsize=10,
)

# F1 on top, C0 bottom
axes[0].invert_yaxis()

for ax in axes[1:]:
    ax.tick_params(
        axis="y",
        labelleft=False,
    )


# =========================================================
# 12) PANEL-SPECIFIC X LIMITS
#
# Dynamic limits based on actual CI ranges.
# =========================================================

for ax, metric in zip(axes, metrics):

    sub = df[
        df["metric_name"] == metric
    ]

    xmin = sub["ci_low_ap"].min()
    xmax = sub["ci_high_ap"].max()

    span = xmax - xmin

    pad = max(
        span * 0.08,
        0.03,
    )

    ax.set_xlim(
        xmin - pad,
        xmax + pad,
    )


# =========================================================
# 13) FINAL LAYOUT
# =========================================================

plt.tight_layout(
    w_pad=2.8
)


# =========================================================
# 14) SAVE
# =========================================================

ABLATION_DIR.mkdir(
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
# 15) AUDIT OUTPUT
# =========================================================

print()
print(
    "=== ABLATION FOREST PLOT AUDIT ==="
)

for metric in metrics:

    sub = df[
        df["metric_name"] == metric
    ]

    n_exclude = (
        (
            sub["ci_low_ap"] > 0
        )
        |
        (
            sub["ci_high_ap"] < 0
        )
    ).sum()

    print(
        f"{metric}: "
        f"{n_exclude}/9 descriptive CIs "
        f"do not contain 0"
    )


# =========================================================
# 16) EXPECTED SCIENTIFIC CHECKS
# =========================================================

# Primary mAP:
map_df = df[
    df["metric_name"]
    == "validation_bbox_mAP_50_95"
]

assert (
    (
        map_df["ci_low_ap"] <= 0
    )
    &
    (
        map_df["ci_high_ap"] >= 0
    )
).all()

print(
    "PASS: 9/9 ΔmAP descriptive CIs contain 0."
)


# AP50:
ap50_df = df[
    df["metric_name"]
    == "validation_AP50"
]

assert (
    (
        ap50_df["ci_low_ap"] <= 0
    )
    &
    (
        ap50_df["ci_high_ap"] >= 0
    )
).all()

print(
    "PASS: 9/9 ΔAP50 descriptive CIs contain 0."
)


# AP75:
ap75_df = df[
    df["metric_name"]
    == "validation_AP75"
]

assert (
    ap75_df["mean_ap"] < 0
).all()

ap75_excluding_zero = set(
    ap75_df.loc[
        ap75_df["ci_high_ap"] < 0,
        "variant_id",
    ]
)

if ap75_excluding_zero != {
    "C0",
    "A2",
    "F1",
}:
    raise ValueError(
        "Các variant có AP75 CI hoàn toàn dưới 0 "
        "không khớp C0/A2/F1.\n"
        f"Observed: {ap75_excluding_zero}"
    )

print(
    "PASS: AP75 mean âm ở 9/9 variants; "
    "C0/A2/F1 có CI hoàn toàn dưới 0."
)


# Qpseudo:
qp_df = df[
    df["metric_name"]
    == "Q_pseudo"
]

qp_above_zero = set(
    qp_df.loc[
        qp_df["ci_low_ap"] > 0,
        "variant_id",
    ]
)

if qp_above_zero != {
    "C0",
    "C1",
}:
    raise ValueError(
        "Qpseudo CI hoàn toàn trên 0 "
        "không khớp C0/C1.\n"
        f"Observed: {qp_above_zero}"
    )

print(
    "PASS: chỉ C0/C1 có ΔQpseudo CI "
    "hoàn toàn trên 0."
)


print()
print("=== OUTPUT ===")
print(OUT_PNG.resolve())
print(OUT_PDF.resolve())

print()
print("DONE.")
