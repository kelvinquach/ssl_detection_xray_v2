import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


# =========================================================
# 1) PATHS
# =========================================================
BASE = Path(r"D:\ssl_detection_xray_v2")

RQ2_DIR = BASE / "analysis" / "rq2"
RQ7_DIR = BASE / "analysis" / "rq7"

CELL_CSV = RQ2_DIR / "rq2_cell_descriptives.csv"
OUT_PNG = RQ7_DIR / "sq3_budget_architecture_descriptive.png"
OUT_PDF = RQ7_DIR / "sq3_budget_architecture_descriptive.pdf"


# =========================================================
# 2) CHECK INPUT
# =========================================================
if not CELL_CSV.exists():
    raise FileNotFoundError(f"Không tìm thấy file: {CELL_CSV}")


# =========================================================
# 3) LOAD DATA
# =========================================================
df = pd.read_csv(CELL_CSV)

required_cols = {
    "architecture",
    "budget",
    "mean",
    "mean_two_sided_95CI_low",
    "mean_two_sided_95CI_high",
}

missing = required_cols - set(df.columns)
if missing:
    raise ValueError(
        f"rq2_cell_descriptives.csv thiếu cột: {sorted(missing)}"
    )

# Chỉ giữ metric chính nếu file có nhiều metric
if "metric_name" in df.columns:
    metric_candidates = df["metric_name"].unique().tolist()
    # Ưu tiên bbox_mAP / mAP_test nếu có
    preferred = None
    for candidate in metric_candidates:
        c = str(candidate).lower()
        if "bbox" in c and "map" in c:
            preferred = candidate
            break
        if "map_test" in c or "mAP_test" in c:
            preferred = candidate
            break

    if preferred is not None:
        df = df[df["metric_name"] == preferred].copy()

# =========================================================
# 4) NORMALIZE LABELS
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

arch_map = {
    "R50-FPN": "R50-FPN",
    "R50": "R50-FPN",
    "Swin-T-FPN": "Swin-T-FPN",
    "Swin-T": "Swin-T-FPN",
}

df["budget_label"] = df["budget"].map(budget_map)
df["arch_label"] = df["architecture"].map(arch_map)

if df["budget_label"].isna().any():
    raise ValueError(
        "Có budget chưa được map. Kiểm tra lại cột budget."
    )

if df["arch_label"].isna().any():
    raise ValueError(
        "Có architecture chưa được map. Kiểm tra lại cột architecture."
    )

budget_order = ["1%", "5%", "10%", "20%"]
arch_order = ["R50-FPN", "Swin-T-FPN"]

df["budget_label"] = pd.Categorical(
    df["budget_label"],
    categories=budget_order,
    ordered=True
)

df["arch_label"] = pd.Categorical(
    df["arch_label"],
    categories=arch_order,
    ordered=True
)

df = df.sort_values(["arch_label", "budget_label"]).reset_index(drop=True)

# Kỳ vọng 8 hàng = 2 architectures × 4 budgets
if len(df) != 8:
    raise ValueError(
        f"Kỳ vọng 8 rows (2 architectures × 4 budgets), nhận được {len(df)} rows."
    )

# =========================================================
# 5) CONVERT TO AP POINTS
# =========================================================
for col in ["mean", "mean_two_sided_95CI_low", "mean_two_sided_95CI_high"]:
    df[col + "_ap"] = df[col] * 100.0

# =========================================================
# 6) SPLIT BY ARCHITECTURE
# =========================================================
r50 = (
    df[df["arch_label"] == "R50-FPN"]
    .sort_values("budget_label")
    .reset_index(drop=True)
)

swin = (
    df[df["arch_label"] == "Swin-T-FPN"]
    .sort_values("budget_label")
    .reset_index(drop=True)
)

if len(r50) != 4 or len(swin) != 4:
    raise ValueError("Không đủ 4 budgets cho mỗi kiến trúc.")

# Cross-check budget order
if list(r50["budget_label"]) != budget_order:
    raise ValueError("R50 budgets không đúng thứ tự 1%, 5%, 10%, 20%.")
if list(swin["budget_label"]) != budget_order:
    raise ValueError("Swin budgets không đúng thứ tự 1%, 5%, 10%, 20%.")

# =========================================================
# 7) EXTRACT VALUES
# =========================================================
x = np.arange(len(budget_order))

r50_mean = r50["mean_ap"].to_numpy()
r50_low = r50["mean_two_sided_95CI_low_ap"].to_numpy()
r50_high = r50["mean_two_sided_95CI_high_ap"].to_numpy()

swin_mean = swin["mean_ap"].to_numpy()
swin_low = swin["mean_two_sided_95CI_low_ap"].to_numpy()
swin_high = swin["mean_two_sided_95CI_high_ap"].to_numpy()

r50_yerr = np.vstack([r50_mean - r50_low, r50_high - r50_mean])
swin_yerr = np.vstack([swin_mean - swin_low, swin_high - swin_mean])

# Chênh lệch Swin - R50 theo budget
diff = swin_mean - r50_mean

# =========================================================
# 8) OPTIONAL CANONICAL CHECK
# =========================================================
expected_diff = np.array([0.52, -0.07, -0.11, -0.02])

if not np.allclose(diff, expected_diff, atol=0.02):
    print("WARNING: Chênh lệch theo budget không khớp xấp xỉ 0.52, -0.07, -0.11, -0.02")
    print("Observed:", diff)

# =========================================================
# 9) PLOT
# =========================================================
fig, ax = plt.subplots(figsize=(12, 6))

# R50
ax.errorbar(
    x - 0.10,
    r50_mean,
    yerr=r50_yerr,
    fmt="o",
    markersize=9,
    capsize=6,
    linewidth=2,
    label="R50-FPN",
)

# Swin-T
ax.errorbar(
    x + 0.10,
    swin_mean,
    yerr=swin_yerr,
    fmt="s",
    markersize=9,
    capsize=6,
    linewidth=2,
    label="Swin-T-FPN",
)

# nối hai đường mean
ax.plot(x - 0.10, r50_mean, linewidth=1.2, alpha=0.5)
ax.plot(x + 0.10, swin_mean, linewidth=1.2, alpha=0.5)

# line at 0
ax.axhline(0, linewidth=1.0, color="0.25")

# annotate differences
for i, d in enumerate(diff):
    txt = f"Chênh lệch\n{d:+.2f}".replace(".", ",")
    ax.text(
        x[i],
        -0.12,
        txt,
        ha="center",
        va="top",
        fontsize=12,
        color="0.25",
    )

# axes
ax.set_xticks(x)
ax.set_xticklabels(budget_order, fontsize=14)
ax.set_xlabel("Ngân sách nhãn (tỷ lệ ảnh có hộp giới hạn)", fontsize=15)
ax.set_ylabel("Hiệu ứng SSL−SUP (điểm AP)", fontsize=16)

# y-range gần giống hình mẫu
ymin = min(r50_low.min(), swin_low.min()) - 0.15
ymax = 0.10
ax.set_ylim(ymin, ymax)

# legend
ax.legend(loc="lower left", frameon=False, fontsize=16)

# clean frame
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.tight_layout()

# =========================================================
# 10) SAVE
# =========================================================
RQ7_DIR.mkdir(parents=True, exist_ok=True)

fig.savefig(OUT_PNG, dpi=300, bbox_inches="tight")
fig.savefig(OUT_PDF, bbox_inches="tight")
plt.close(fig)

# =========================================================
# 11) PRINT SUMMARY
# =========================================================
print("\n=== Budget-specific descriptive SQ3 figure ===")
for i, budget in enumerate(budget_order):
    print(
        f"{budget}: "
        f"R50={r50_mean[i]:.3f}, "
        f"Swin-T={swin_mean[i]:.3f}, "
        f"Diff={diff[i]:+.3f} AP points"
    )

print("\n=== OUTPUT ===")
print(OUT_PNG.resolve())
print(OUT_PDF.resolve())

print("\nDONE.")
