from __future__ import annotations

import csv
import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

CONFIDENCE = ROOT / "analysis/ablation/confidence_summary.csv"
AUGMENTATION = ROOT / "analysis/ablation/augmentation_summary.csv"
REGRESSION = ROOT / "analysis/ablation/regression_filter_summary.csv"

OUTPUT = ROOT / "analysis/ablation/ablation_thesis_tables.tex"

EXPECTED_SHA = {
    CONFIDENCE:
        "a7a5fc63dad652a4b373dcb6284bcafd1501742fd4b3547a1900312c6076b0a8",
    AUGMENTATION:
        "7221917b55935c0dbe832fece0363df2d3798fc752e367bfa11744d578bf6f1f",
    REGRESSION:
        "fdced4aa003a1bca0ba2a9e8b05fdea587ad6838c20d16f4c05c3b6487a43bd1",
}

METRIC_LABELS = {
    "validation_bbox_mAP_50_95":
        r"bbox mAP@[0.50:0.95]",
    "validation_AP50":
        r"AP50",
    "validation_AP75":
        r"AP75",
    "Q_pseudo":
        r"$Q_{\mathrm{pseudo}}$",
}

TABLES = [
    {
        "path": CONFIDENCE,
        "caption":
            "Ablation ngưỡng confidence của pseudo-label trên validation.",
        "label":
            "tab:ablation_confidence",
        "expected_rows": 20,
        "variants": ["C0", "C1", "C2", "C3", "C4"],
    },
    {
        "path": AUGMENTATION,
        "caption":
            "Ablation strong photometric augmentation trên validation.",
        "label":
            "tab:ablation_augmentation",
        "expected_rows": 16,
        "variants": ["A0", "A1", "A2", "A3"],
    },
    {
        "path": REGRESSION,
        "caption":
            "Ablation chiến lược lọc pseudo-label cho bbox regression trên validation.",
        "label":
            "tab:ablation_regression_filter",
        "expected_rows": 12,
        "variants": ["F0", "F1", "F2"],
    },
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def load_csv(path: Path) -> list[dict[str, str]]:
    require(path.is_file(), f"MISSING_FILE:{path}")

    with path.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as f:
        return list(csv.DictReader(f))


def tex_escape(value: str) -> str:
    value = str(value)

    replacements = [
        ("\\", r"\textbackslash{}"),
        ("&", r"\&"),
        ("%", r"\%"),
        ("$", r"\$"),
        ("#", r"\#"),
        ("_", r"\_"),
        ("{", r"\{"),
        ("}", r"\}"),
    ]

    for old, new in replacements:
        value = value.replace(old, new)

    return value


def fmt(x: str) -> str:
    return f"{float(x):.4f}"


def render_table(spec: dict) -> str:
    path = spec["path"]
    rows = load_csv(path)

    require(
        len(rows) == spec["expected_rows"],
        f"{path.name}:ROW_COUNT:{len(rows)}",
    )

    observed_variants = {r["variant_id"] for r in rows}
    require(
        observed_variants == set(spec["variants"]),
        f"{path.name}:VARIANT_SET_INVALID:{sorted(observed_variants)}",
    )

    observed_metrics = {r["metric_name"] for r in rows}
    require(
        observed_metrics == set(METRIC_LABELS),
        f"{path.name}:METRIC_SET_INVALID:{sorted(observed_metrics)}",
    )

    require(
        all(r["formal_hypothesis_test"] == "false" for r in rows),
        f"{path.name}:FORMAL_TEST_PRESENT",
    )

    require(
        all(r["p_value_computed"] == "false" for r in rows),
        f"{path.name}:P_VALUE_PRESENT",
    )

    require(
        all(r["multiplicity_family_added"] == "false" for r in rows),
        f"{path.name}:MULTIPLICITY_PRESENT",
    )

    by_key = {
        (r["variant_id"], r["metric_name"]): r
        for r in rows
    }

    lines = []

    lines.append(r"\begin{table}[htbp]")
    lines.append(r"    \centering")
    lines.append(
        r"    \caption{" + spec["caption"] + "}"
    )
    lines.append(
        r"    \label{" + spec["label"] + "}"
    )
    lines.append(r"    \small")
    lines.append(r"    \begin{tabular}{lllllll}")
    lines.append(r"        \toprule")
    lines.append(
        r"        Variant & Level & Metric & "
        r"Mean $\pm$ SD & Main mean & "
        r"$\Delta$ vs Main & 95\% CI \\"
    )
    lines.append(r"        \midrule")

    for variant in spec["variants"]:
        for metric in METRIC_LABELS:
            r = by_key[(variant, metric)]

            level = tex_escape(r["level"])
            metric_label = METRIC_LABELS[metric]

            mean_sd = (
                fmt(r["variant_mean"])
                + r" $\pm$ "
                + fmt(r["variant_sample_SD_ddof1"])
            )

            main_mean = fmt(r["main_mean"])
            delta = fmt(r["paired_mean_difference"])

            ci = (
                "["
                + fmt(r["paired_difference_two_sided_95CI_low"])
                + ", "
                + fmt(r["paired_difference_two_sided_95CI_high"])
                + "]"
            )

            lines.append(
                "        "
                + variant
                + " & "
                + level
                + " & "
                + metric_label
                + " & "
                + mean_sd
                + " & "
                + main_mean
                + " & "
                + delta
                + " & "
                + ci
                + r" \\"
            )

        if variant != spec["variants"][-1]:
            lines.append(r"        \addlinespace")

    lines.append(r"        \bottomrule")
    lines.append(r"    \end{tabular}")
    lines.append(
        r"    \begin{flushleft}\footnotesize "
        r"Giá trị được tổng hợp trên 10 training seeds. "
        r"SD là sample standard deviation với $ddof=1$. "
        r"$\Delta$ được định nghĩa là variant trừ Main cùng seed. "
        r"Khoảng tin cậy là two-sided 95\% CI của paired mean difference. "
        r"Không thực hiện formal hypothesis test cho các variant ablation. "
        r"Non-Main variants chỉ được đánh giá trên validation; "
        r"cấu hình Main không được thay đổi từ kết quả ablation."
        r"\end{flushleft}"
    )
    lines.append(r"\end{table}")

    return "\n".join(lines)


def main() -> None:
    for path, expected in EXPECTED_SHA.items():
        observed = sha256(path)

        require(
            observed == expected,
            f"{path.name}:SHA256_MISMATCH:{observed}",
        )

    rendered = [
        "% AUTO-GENERATED S10.1 ABLATION THESIS TABLES",
        "% Presentation-only artifact.",
        "% No statistics are recomputed here.",
        "% Sources:",
        "% - analysis/ablation/confidence_summary.csv",
        "% - analysis/ablation/augmentation_summary.csv",
        "% - analysis/ablation/regression_filter_summary.csv",
        "",
    ]

    for idx, spec in enumerate(TABLES):
        rendered.append(render_table(spec))

        if idx != len(TABLES) - 1:
            rendered.append("")
            rendered.append("")

    rendered.append("")
    rendered.append(
        "% FORMAL_HYPOTHESIS_TEST=FALSE"
    )
    rendered.append(
        "% P_VALUE_COMPUTED=FALSE"
    )
    rendered.append(
        "% MULTIPLICITY_FAMILY_ADDED=FALSE"
    )
    rendered.append(
        "% MAIN_RETRAINED=FALSE"
    )
    rendered.append(
        "% STATISTICS_REIMPLEMENTED=FALSE"
    )

    OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    OUTPUT.write_text(
        "\n".join(rendered) + "\n",
        encoding="utf-8",
    )

    text = OUTPUT.read_text(encoding="utf-8")

    require(
        text.count(r"\begin{table}") == 3,
        "TABLE_COUNT_NOT_3",
    )

    require(
        text.count(r"\caption{") == 3,
        "CAPTION_COUNT_NOT_3",
    )

    require(
        "C4" in text
        and "A3" in text
        and "F2" in text,
        "MAIN_ALIASES_MISSING",
    )

    require(
        r"$Q_{\mathrm{pseudo}}$" in text,
        "QPSEUDO_MISSING",
    )

    print(
        "=== S10.1 ABLATION THESIS TABLE PRODUCER ==="
    )
    print("STATUS=PASS")
    print("TABLE_COUNT=3")
    print("CONFIDENCE_CONFIGS=5")
    print("AUGMENTATION_CONFIGS=4")
    print("REGRESSION_CONFIGS=3")
    print("METRICS_PER_CONFIG=4")
    print("MAIN_ALIASES=C4|A3|F2")
    print("FORMAL_HYPOTHESIS_TEST=FALSE")
    print("P_VALUE_COMPUTED=FALSE")
    print("MULTIPLICITY_FAMILY_ADDED=FALSE")
    print("MAIN_RETRAINED=FALSE")
    print("STATISTICS_REIMPLEMENTED=FALSE")
    print(
        "OUTPUT="
        + str(OUTPUT.relative_to(ROOT))
    )
    print(
        "OUTPUT_SHA256="
        + sha256(OUTPUT)
    )


if __name__ == "__main__":
    main()