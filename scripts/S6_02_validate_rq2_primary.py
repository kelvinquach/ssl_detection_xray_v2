"""Validate S6.02 RQ2 primary one-sample t-test implementation.

This validator extends the shared statistical fixture evidence while preserving
the closed S6.01 top-level evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

DEFAULT_OUTPUT = Path(
    "/workspace/ssod/project/artifacts/preflight/statistics/statistical_fixture_report.json"
)
HISTORICAL_S6_01_EVIDENCE_SHA256 = (
    "50f9f408ab3d8c4a030825c99a14fd0c055be6a61567b720150f2d4104fac368"
)
EXPECTED_S6_01_SEED_SUMMARY_SHA256 = (
    "2a404ac0f6097cad16816b550fcd1c9ca240fab5535854dc0fd5e4433e49ff3f"
)
EXPECTED_S6_01_VALIDATOR_SHA256 = (
    "99e571af06ac3db43a6673f1139d898cfa7079654b6ae8352ad81b177872cc35"
)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path("/workspace/ssod/project"))
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    repo = args.repo_root.resolve()
    sys.path.insert(0, str(repo))

    from src.statistics.rq2_primary import (  # noqa: E402
        RQ2_ALPHA,
        RQ2_ALTERNATIVE,
        RQ2_CELL_COUNT,
        RQ2_CI_LEVEL,
        RQ2_DF,
        RQ2_NULL_MEAN,
        RQ2PrimaryError,
        compute_rq2_gain_by_seed,
        rq2_primary_test,
    )
    from src.statistics.seed_summary import (  # noqa: E402
        OFFICIAL_SEED_COUNT,
        OFFICIAL_TRAINING_SEEDS,
        REPLICATION_UNIT,
    )

    evidence_path = args.output.resolve()
    if not evidence_path.exists():
        raise FileNotFoundError(f"shared S6 evidence not found: {evidence_path}")

    historical_bytes = evidence_path.read_bytes()
    historical_sha = hashlib.sha256(historical_bytes).hexdigest()
    evidence = json.loads(historical_bytes.decode("utf-8"))

    checks: list[dict[str, object]] = []

    def check(name: str, condition: bool, detail: object) -> None:
        checks.append({"name": name, "pass": bool(condition), "detail": detail})

    # Preserve the closed S6.01 proof exactly before extension.
    check(
        "historical::s6_01_evidence_sha256",
        historical_sha == HISTORICAL_S6_01_EVIDENCE_SHA256,
        historical_sha,
    )
    check("historical::s6_01_task", evidence.get("task") == "S6.01", evidence.get("task"))
    check("historical::s6_01_status", evidence.get("status") == "PASS", evidence.get("status"))
    check(
        "historical::s6_01_checks_14_of_14",
        evidence.get("checks_passed") == 14
        and evidence.get("checks_total") == 14
        and evidence.get("failed_count") == 0,
        {
            "checks_passed": evidence.get("checks_passed"),
            "checks_total": evidence.get("checks_total"),
            "failed_count": evidence.get("failed_count"),
        },
    )
    historical_sources = evidence.get("source_sha256", {})
    check(
        "historical::s6_01_seed_summary_sha256",
        historical_sources.get("src/statistics/seed_summary.py")
        == EXPECTED_S6_01_SEED_SUMMARY_SHA256,
        historical_sources.get("src/statistics/seed_summary.py"),
    )
    check(
        "historical::s6_01_validator_sha256",
        historical_sources.get("scripts/S6_01_validate_seed_summary.py")
        == EXPECTED_S6_01_VALIDATOR_SHA256,
        historical_sources.get("scripts/S6_01_validate_seed_summary.py"),
    )

    # Contract checks.
    check("contract::replication_unit", REPLICATION_UNIT == "TRAINING_SEED_TRAINED_RUN", REPLICATION_UNIT)
    check("contract::official_seed_count", OFFICIAL_SEED_COUNT == 10, OFFICIAL_SEED_COUNT)
    check("contract::cell_count_8", RQ2_CELL_COUNT == 8, RQ2_CELL_COUNT)
    check("contract::null_mean_zero", RQ2_NULL_MEAN == 0.0, RQ2_NULL_MEAN)
    check("contract::alternative_greater", RQ2_ALTERNATIVE == "greater", RQ2_ALTERNATIVE)
    check("contract::alpha_005", RQ2_ALPHA == 0.05, RQ2_ALPHA)
    check("contract::ci_level_095", RQ2_CI_LEVEL == 0.95, RQ2_CI_LEVEL)
    check("contract::df_9", RQ2_DF == 9, RQ2_DF)

    # Positive golden fixture: each seed has 8 equal-weight cell effects equal
    # to the seed ordinal, hence G_s = 1..10.
    cell_effects = {
        seed: [float(i)] * RQ2_CELL_COUNT
        for i, seed in enumerate(OFFICIAL_TRAINING_SEEDS, start=1)
    }
    seed_gains = compute_rq2_gain_by_seed(cell_effects)
    result = rq2_primary_test(seed_gains)

    expected_gains = [float(i) for i in range(1, 11)]
    observed_gains = [seed_gains[seed] for seed in OFFICIAL_TRAINING_SEEDS]
    check("positive::g_s_equal_weight_mean", np.allclose(observed_gains, expected_gains), observed_gains)
    check("positive::n_10", result["n"] == 10, result["n"])
    check("positive::df_9", result["df"] == 9, result["df"])
    check("positive::effect_estimate", np.isclose(result["effect_estimate"], 5.5), result["effect_estimate"])
    check(
        "positive::sample_sd_ddof1",
        np.isclose(result["sample_sd"], 3.0276503540974917),
        result["sample_sd"],
    )
    check("positive::sample_sd_ddof_field", result["sample_sd_ddof"] == 1, result["sample_sd_ddof"])
    check(
        "positive::t_statistic",
        np.isclose(result["t_statistic"], 5.744562646538029),
        result["t_statistic"],
    )
    check(
        "positive::one_sided_p_value",
        np.isclose(result["p_value"], 0.00013909800552409273),
        result["p_value"],
    )
    check(
        "positive::p_value_sidedness",
        result["p_value_sidedness"] == "one-sided",
        result["p_value_sidedness"],
    )
    check(
        "positive::ci_two_sided_95",
        np.isclose(result["ci_low"], 3.3341494102783162)
        and np.isclose(result["ci_high"], 7.665850589721684)
        and result["ci_sidedness"] == "two-sided"
        and np.isclose(result["ci_level"], 0.95),
        {
            "ci_low": result["ci_low"],
            "ci_high": result["ci_high"],
            "ci_level": result["ci_level"],
            "ci_sidedness": result["ci_sidedness"],
        },
    )
    check("positive::alternative", result["alternative"] == "greater", result["alternative"])

    # Fail-closed fixtures.
    negative_fixtures: dict[str, dict[str, object]] = {}

    def expect_blocked(name: str, func) -> None:
        blocked = False
        message = None
        try:
            func()
        except (RQ2PrimaryError, ValueError, TypeError) as exc:
            blocked = True
            message = str(exc)
        negative_fixtures[name] = {"blocked": blocked, "error": message}
        check(f"fail_closed::{name}", blocked, message)

    missing_seed = dict(cell_effects)
    missing_seed.pop(OFFICIAL_TRAINING_SEEDS[-1])
    expect_blocked("missing_seed", lambda: compute_rq2_gain_by_seed(missing_seed))

    reversed_seed_items = list(cell_effects.items())[::-1]
    expect_blocked(
        "reordered_seeds",
        lambda: compute_rq2_gain_by_seed(dict(reversed_seed_items)),
    )

    seven_cells = {seed: list(vals) for seed, vals in cell_effects.items()}
    seven_cells[OFFICIAL_TRAINING_SEEDS[0]] = [1.0] * 7
    expect_blocked("wrong_cell_count", lambda: compute_rq2_gain_by_seed(seven_cells))

    nonfinite_cells = {seed: list(vals) for seed, vals in cell_effects.items()}
    nonfinite_cells[OFFICIAL_TRAINING_SEEDS[0]][0] = float("nan")
    expect_blocked("nonfinite_cell_effect", lambda: compute_rq2_gain_by_seed(nonfinite_cells))

    rq2_path = repo / "src/statistics/rq2_primary.py"
    seed_summary_path = repo / "src/statistics/seed_summary.py"
    validator_path = repo / "scripts/S6_02_validate_rq2_primary.py"

    failed = [c for c in checks if not c["pass"]]

    s6_02_section = {
        "task": "S6.02",
        "tracker_action": "Implement one-sample t, alternative=greater",
        "status": "PASS" if not failed else "FAIL",
        "scientific_contract": {
            "effect": "G_s",
            "g_s_definition": "EQUAL_WEIGHT_MEAN_OF_8_PAIRED_ARCHITECTURE_BUDGET_DELTA_MAP_CELLS",
            "cell_count_per_seed": 8,
            "cell_weight": 1.0 / 8.0,
            "inferential_replication_unit": REPLICATION_UNIT,
            "n": 10,
            "df": 9,
            "null_mean": 0.0,
            "alternative": "greater",
            "p_value_sidedness": "one-sided",
            "ci_level": 0.95,
            "ci_sidedness": "two-sided",
        },
        "positive_fixture": {
            "g_s_values": observed_gains,
            "effect_estimate": result["effect_estimate"],
            "sample_sd": result["sample_sd"],
            "sample_sd_ddof": result["sample_sd_ddof"],
            "t_statistic": result["t_statistic"],
            "p_value": result["p_value"],
            "ci_low": result["ci_low"],
            "ci_high": result["ci_high"],
        },
        "negative_fixtures": negative_fixtures,
        "checks": checks,
        "checks_passed": len(checks) - len(failed),
        "checks_total": len(checks),
        "failed_count": len(failed),
        "failed_checks": [str(c["name"]) for c in failed],
        "historical_s6_01_evidence_sha256_before_extension": historical_sha,
        "source_sha256": {
            "src/statistics/rq2_primary.py": sha256_file(rq2_path),
            "src/statistics/seed_summary.py": sha256_file(seed_summary_path),
            "scripts/S6_02_validate_rq2_primary.py": sha256_file(validator_path),
        },
        "scientific_boundaries": {
            "fixture_only_no_official_results": True,
            "test_used": False,
            "hidden_unlabeled_gt_used": False,
            "official_training_authorized": False,
            "final_test_authorized": False,
            "sign_flip_implemented_in_s6_02": False,
            "loso_implemented_in_s6_02": False,
        },
    }

    evidence["s6_02_rq2_primary_t_test"] = s6_02_section
    evidence_path.parent.mkdir(parents=True, exist_ok=True)
    evidence_path.write_text(
        json.dumps(evidence, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )

    print("===== S6.02 RQ2 PRIMARY VALIDATOR =====")
    for item in checks:
        print(f'{item["name"]}={item["pass"]}')
    print(f"CHECKS_PASSED={len(checks) - len(failed)}")
    print(f"CHECKS_TOTAL={len(checks)}".replace(" ", ""))
    print(f"FAILED_COUNT={len(failed)}")
    print("FAILED_CHECKS=" + (",".join(str(c ["name"]) for c in failed)  if failed else "NONE"))
    print(f"EVIDENCE={evidence_path}")
    print(f"EVIDENCE_SHA256={sha256_file(evidence_path)}")
    print("STATUS=" + ("PASS" if not failed else "FAIL"))
    print("S6_02_RQ2_PRIMARY_FIXTURE=" + ("PASS" if not failed else "FAIL"))
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
