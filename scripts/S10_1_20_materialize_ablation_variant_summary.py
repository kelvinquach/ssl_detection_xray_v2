from __future__ import annotations

import csv
import hashlib
import math
import sys
from collections import OrderedDict, defaultdict
from pathlib import Path

import scipy
from scipy import stats


ROOT = Path(__file__).resolve().parents[1]

SOURCE = (
    ROOT / "analysis/ablation/ablation_seed_level_values.csv"
)

QPSEUDO_SOURCE = (
    ROOT / "analysis/ablation/ablation_qpseudo_effects.csv"
)

OUTPUT = (
    ROOT / "analysis/ablation/ablation_variant_summary.csv"
)

EXPECTED_SOURCE_SHA256 = (
    "04a63af61d7978d14384d27db8af05b330bfffd4eda29c844377d692482090cb"
)

EXPECTED_QPSEUDO_SHA256 = (
    "d67f2a595fa6f9d8949ac7022c3958818b2c0798e1f1c612aa04f7433f2877b8"
)

FROZEN_SEED_SUMMARY_SHA256 = (
    "2a404ac0f6097cad16816b550fcd1c9ca240fab5535854dc0fd5e4433e49ff3f"
)

FROZEN_RQ3_PAIRWISE_SHA256 = (
    "18c0c84fd396193e6019c9a55f051e6ab68ea9cb08e132017d92ddd017be8381"
)

FROZEN_SCIPY_VERSION = "1.15.3"

EXPECTED_VARIANTS = {
    "C0", "C1", "C2", "C3",
    "A0", "A1", "A2",
    "F0", "F1",
}

EXPECTED_METRICS = {
    "validation_bbox_mAP_50_95",
    "validation_AP50",
    "validation_AP75",
    "Q_pseudo",
}

VARIANT_ORDER = {
    "C0": 0,
    "C1": 1,
    "C2": 2,
    "C3": 3,
    "A0": 4,
    "A1": 5,
    "A2": 6,
    "F0": 7,
    "F1": 8,
}

METRIC_ORDER = {
    "validation_bbox_mAP_50_95": 0,
    "validation_AP50": 1,
    "validation_AP75": 2,
    "Q_pseudo": 3,
}

FIELDNAMES = [
    "variant_id",
    "family",
    "level",
    "metric_name",
    "n_seeds",
    "variant_mean",
    "variant_sample_SD_ddof1",
    "main_mean",
    "paired_mean_difference",
    "paired_difference_sample_SD_ddof1",
    "paired_difference_two_sided_95CI_low",
    "paired_difference_two_sided_95CI_high",
    "ci_level",
    "formal_hypothesis_test",
    "p_value_computed",
    "multiplicity_family_added",
    "replication_unit",
    "statistics_semantics",
    "source_artifact",
    "source_artifact_sha256",
    "representation_authority",
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as f:
        return list(csv.DictReader(f))


def norm(value) -> str:
    return str(value if value is not None else "").strip()


def finite_float(value, label: str) -> float:
    try:
        x = float(value)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(
            f"{label}:NON_NUMERIC:{value!r}"
        ) from exc

    if not math.isfinite(x):
        raise RuntimeError(
            f"{label}:NON_FINITE:{x!r}"
        )

    return x


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def close_enough(a: float, b: float, tol: float = 1e-15) -> bool:
    return abs(a - b) <= tol


def main() -> None:
    require(SOURCE.is_file(), f"MISSING_SOURCE:{SOURCE}")
    require(
        QPSEUDO_SOURCE.is_file(),
        f"MISSING_QPSEUDO_SOURCE:{QPSEUDO_SOURCE}",
    )

    source_sha = sha256(SOURCE)
    qpseudo_sha = sha256(QPSEUDO_SOURCE)

    require(
        source_sha == EXPECTED_SOURCE_SHA256,
        f"SOURCE_SHA_MISMATCH:{source_sha}",
    )
    require(
        qpseudo_sha == EXPECTED_QPSEUDO_SHA256,
        f"QPSEUDO_SHA_MISMATCH:{qpseudo_sha}",
    )

    sys.path.insert(0, str(ROOT))

    from src.statistics.seed_summary import (
        OFFICIAL_SEED_COUNT,
        OFFICIAL_TRAINING_SEEDS,
        REPLICATION_UNIT,
        SAMPLE_SD_DDOF,
        summarize_seed_runs,
    )

    from src.statistics.rq3_pairwise import (
        RQ3_PAIRWISE_ALPHA,
        RQ3_PAIRWISE_CI_LEVEL,
        RQ3_PAIRWISE_DF,
    )

    require(
        sha256(ROOT / "src/statistics/seed_summary.py")
        == FROZEN_SEED_SUMMARY_SHA256,
        "FROZEN_SEED_SUMMARY_SHA256_MISMATCH",
    )

    require(
        sha256(ROOT / "src/statistics/rq3_pairwise.py")
        == FROZEN_RQ3_PAIRWISE_SHA256,
        "FROZEN_RQ3_PAIRWISE_SHA256_MISMATCH",
    )

    require(
        scipy.__version__ == FROZEN_SCIPY_VERSION,
        f"SCIPY_VERSION_MISMATCH:"
        f"{scipy.__version__}!={FROZEN_SCIPY_VERSION}",
    )

    require(SAMPLE_SD_DDOF == 1, "SAMPLE_SD_DDOF_NOT_1")
    require(OFFICIAL_SEED_COUNT == 10, "OFFICIAL_SEED_COUNT_NOT_10")
    require(RQ3_PAIRWISE_ALPHA == 0.05, "CI_ALPHA_NOT_0_05")
    require(RQ3_PAIRWISE_CI_LEVEL == 0.95, "CI_LEVEL_NOT_0_95")
    require(RQ3_PAIRWISE_DF == 9, "CI_DF_NOT_9")

    rows = load_csv(SOURCE)

    require(len(rows) == 360, f"SOURCE_ROWS:{len(rows)}")

    grouped: dict[
        tuple[str, str],
        list[dict[str, str]]
    ] = defaultdict(list)

    for row in rows:
        require(
            norm(row["pair_valid"]).lower() == "true",
            "PAIR_INVALID",
        )
        require(
            norm(row["formal_hypothesis_test"]).lower()
            == "false",
            "FORMAL_HYPOTHESIS_TEST_INVALID",
        )
        require(
            norm(row["test_used"]).lower() == "false",
            "TEST_USED_INVALID",
        )
        require(
            norm(row["hidden_u_gt_used"]).lower()
            == "false",
            "HIDDEN_U_GT_USED_INVALID",
        )

        grouped[
            (
                norm(row["variant_id"]),
                norm(row["metric_name"]),
            )
        ].append(row)

    require(
        len(grouped) == 36,
        f"GROUP_COUNT:{len(grouped)}",
    )

    observed_variants = {
        k[0] for k in grouped
    }
    observed_metrics = {
        k[1] for k in grouped
    }

    require(
        observed_variants == EXPECTED_VARIANTS,
        f"VARIANT_SET:{sorted(observed_variants)}",
    )
    require(
        observed_metrics == EXPECTED_METRICS,
        f"METRIC_SET:{sorted(observed_metrics)}",
    )

    qpseudo_rows = load_csv(QPSEUDO_SOURCE)

    qpseudo_reference = {}

    for r in qpseudo_rows:
        key = norm(r["variant_id"])
        qpseudo_reference.setdefault(
            key,
            {
                "variant_mean":
                    finite_float(r["variant_mean"], "QP_VARIANT_MEAN"),
                "variant_sample_SD_ddof1":
                    finite_float(
                        r["variant_sample_SD_ddof1"],
                        "QP_VARIANT_SD",
                    ),
                "main_mean":
                    finite_float(r["main_mean"], "QP_MAIN_MEAN"),
                "paired_mean_difference":
                    finite_float(
                        r["paired_mean_difference"],
                        "QP_PAIRED_MEAN",
                    ),
                "paired_difference_sample_SD_ddof1":
                    finite_float(
                        r["paired_difference_sample_SD_ddof1"],
                        "QP_PAIRED_SD",
                    ),
                "paired_difference_two_sided_95CI_low":
                    finite_float(
                        r[
                            "paired_difference_two_sided_95CI_low"
                        ],
                        "QP_CI_LOW",
                    ),
                "paired_difference_two_sided_95CI_high":
                    finite_float(
                        r[
                            "paired_difference_two_sided_95CI_high"
                        ],
                        "QP_CI_HIGH",
                    ),
            },
        )

    require(
        set(qpseudo_reference) == EXPECTED_VARIANTS,
        "QPSEUDO_REFERENCE_VARIANT_SET_INVALID",
    )

    output_rows: list[dict[str, str]] = []

    for variant_id in sorted(
        EXPECTED_VARIANTS,
        key=VARIANT_ORDER.get,
    ):
        for metric_name in sorted(
            EXPECTED_METRICS,
            key=METRIC_ORDER.get,
        ):
            seed_rows = grouped[
                (variant_id, metric_name)
            ]

            require(
                len(seed_rows) == 10,
                f"{variant_id}:{metric_name}:N_ROWS:"
                f"{len(seed_rows)}",
            )

            row_by_seed = {
                int(r["training_seed"]): r
                for r in seed_rows
            }

            require(
                tuple(
                    seed
                    for seed in OFFICIAL_TRAINING_SEEDS
                    if seed in row_by_seed
                )
                == OFFICIAL_TRAINING_SEEDS,
                f"{variant_id}:{metric_name}:SEED_SET_INVALID",
            )

            require(
                len(row_by_seed) == 10,
                f"{variant_id}:{metric_name}:SEED_DUPLICATE",
            )

            variant_values = OrderedDict(
                (
                    seed,
                    finite_float(
                        row_by_seed[seed]["variant_value"],
                        f"{variant_id}:{metric_name}:VARIANT:{seed}",
                    ),
                )
                for seed in OFFICIAL_TRAINING_SEEDS
            )

            main_values = OrderedDict(
                (
                    seed,
                    finite_float(
                        row_by_seed[seed]["main_same_seed_value"],
                        f"{variant_id}:{metric_name}:MAIN:{seed}",
                    ),
                )
                for seed in OFFICIAL_TRAINING_SEEDS
            )

            paired_values = OrderedDict(
                (
                    seed,
                    finite_float(
                        row_by_seed[seed]["paired_difference"],
                        f"{variant_id}:{metric_name}:DELTA:{seed}",
                    ),
                )
                for seed in OFFICIAL_TRAINING_SEEDS
            )

            for seed in OFFICIAL_TRAINING_SEEDS:
                require(
                    close_enough(
                        paired_values[seed],
                        variant_values[seed] - main_values[seed],
                    ),
                    f"{variant_id}:{metric_name}:"
                    f"PAIRED_DIFFERENCE_MISMATCH:{seed}",
                )

            variant_summary = summarize_seed_runs(
                variant_values
            )
            main_summary = summarize_seed_runs(
                main_values
            )
            delta_summary = summarize_seed_runs(
                paired_values
            )

            n = int(delta_summary["n"])
            df = n - 1

            require(
                n == OFFICIAL_SEED_COUNT,
                f"{variant_id}:{metric_name}:N_INVALID:{n}",
            )
            require(
                df == RQ3_PAIRWISE_DF,
                f"{variant_id}:{metric_name}:DF_INVALID:{df}",
            )

            effect = float(delta_summary["mean"])
            paired_sd = float(
                delta_summary["sample_sd"]
            )

            standard_error = (
                paired_sd / math.sqrt(n)
            )

            critical_value = float(
                stats.t.ppf(
                    1.0 - RQ3_PAIRWISE_ALPHA / 2.0,
                    df=df,
                )
            )

            ci_low = (
                effect
                - critical_value * standard_error
            )
            ci_high = (
                effect
                + critical_value * standard_error
            )

            require(
                all(
                    math.isfinite(x)
                    for x in (
                        effect,
                        paired_sd,
                        standard_error,
                        critical_value,
                        ci_low,
                        ci_high,
                    )
                ),
                f"{variant_id}:{metric_name}:NON_FINITE_SUMMARY",
            )

            first = seed_rows[0]

            output = {
                "variant_id": variant_id,
                "family": norm(first["family"]),
                "level": norm(first["level"]),
                "metric_name": metric_name,
                "n_seeds": str(n),
                "variant_mean":
                    repr(float(variant_summary["mean"])),
                "variant_sample_SD_ddof1":
                    repr(float(variant_summary["sample_sd"])),
                "main_mean":
                    repr(float(main_summary["mean"])),
                "paired_mean_difference":
                    repr(effect),
                "paired_difference_sample_SD_ddof1":
                    repr(paired_sd),
                "paired_difference_two_sided_95CI_low":
                    repr(ci_low),
                "paired_difference_two_sided_95CI_high":
                    repr(ci_high),
                "ci_level":
                    repr(RQ3_PAIRWISE_CI_LEVEL),
                "formal_hypothesis_test": "false",
                "p_value_computed": "false",
                "multiplicity_family_added": "false",
                "replication_unit":
                    REPLICATION_UNIT,
                "statistics_semantics":
                    "FROZEN_S6_SEED_SUMMARY_PLUS_STUDENT_T_CI_NO_TEST",
                "source_artifact":
                    "analysis/ablation/ablation_seed_level_values.csv",
                "source_artifact_sha256":
                    source_sha,
                "representation_authority":
                    "S10_ABLATION_VARIANT_SUMMARY",
            }

            # Independent cross-check against the already-created
            # canonical Q_pseudo summary.
            if metric_name == "Q_pseudo":
                ref = qpseudo_reference[variant_id]

                comparisons = {
                    "variant_mean":
                        float(output["variant_mean"]),
                    "variant_sample_SD_ddof1":
                        float(output["variant_sample_SD_ddof1"]),
                    "main_mean":
                        float(output["main_mean"]),
                    "paired_mean_difference":
                        float(output["paired_mean_difference"]),
                    "paired_difference_sample_SD_ddof1":
                        float(
                            output[
                                "paired_difference_sample_SD_ddof1"
                            ]
                        ),
                    "paired_difference_two_sided_95CI_low":
                        float(
                            output[
                                "paired_difference_two_sided_95CI_low"
                            ]
                        ),
                    "paired_difference_two_sided_95CI_high":
                        float(
                            output[
                                "paired_difference_two_sided_95CI_high"
                            ]
                        ),
                }

                for field, observed in comparisons.items():
                    require(
                        close_enough(
                            observed,
                            ref[field],
                        ),
                        f"{variant_id}:QPSEUDO_CROSSCHECK:"
                        f"{field}:{observed}!={ref[field]}",
                    )

            output_rows.append(output)

    require(
        len(output_rows) == 36,
        f"OUTPUT_ROWS:{len(output_rows)}",
    )

    require(
        len({
            (
                r["variant_id"],
                r["metric_name"],
            )
            for r in output_rows
        }) == 36,
        "OUTPUT_KEYS_NOT_UNIQUE",
    )

    require(
        all(
            r["n_seeds"] == "10"
            for r in output_rows
        ),
        "N_SEEDS_NOT_10",
    )

    require(
        all(
            r["formal_hypothesis_test"] == "false"
            for r in output_rows
        ),
        "FORMAL_TEST_PRESENT",
    )

    require(
        all(
            r["p_value_computed"] == "false"
            for r in output_rows
        ),
        "P_VALUE_PRESENT",
    )

    require(
        all(
            r["multiplicity_family_added"] == "false"
            for r in output_rows
        ),
        "MULTIPLICITY_FAMILY_ADDED",
    )

    OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with OUTPUT.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=FIELDNAMES,
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(output_rows)

    print(
        "=== S10.1 ABLATION VARIANT SUMMARY MATERIALIZATION ==="
    )
    print("STATUS=PASS")
    print("SOURCE_SHA256=" + source_sha)
    print("QPSEUDO_REFERENCE_SHA256=" + qpseudo_sha)
    print(
        "FROZEN_SEED_SUMMARY_SHA256="
        + FROZEN_SEED_SUMMARY_SHA256
    )
    print(
        "FROZEN_RQ3_PAIRWISE_SHA256="
        + FROZEN_RQ3_PAIRWISE_SHA256
    )
    print("SCIPY_VERSION=" + scipy.__version__)
    print("ROWS=" + str(len(output_rows)))
    print("UNIQUE_VARIANTS=9")
    print("UNIQUE_METRICS=4")
    print("VARIANT_METRIC_GROUPS=36")
    print("N_SEEDS_PER_GROUP=10")
    print("SAMPLE_SD_DDOF=1")
    print("CI_LEVEL=0.95")
    print("CI_DF=9")
    print("FORMAL_HYPOTHESIS_TEST=FALSE")
    print("P_VALUE_COMPUTED=FALSE")
    print("MULTIPLICITY_FAMILY_ADDED=FALSE")
    print("QPSEUDO_CROSSCHECK=PASS")
    print("STATISTICS_CORE=FROZEN_S6_REUSE")
    print("STATISTICS_REIMPLEMENTED=FALSE")
    print(
        "OUTPUT="
        + str(OUTPUT.relative_to(ROOT))
    )
    print("OUTPUT_SHA256=" + sha256(OUTPUT))


if __name__ == "__main__":
    main()