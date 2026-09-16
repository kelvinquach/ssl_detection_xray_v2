"""S6.03 validator: RQ3 repeated-measures ANOVA + Greenhouse-Geisser ALWAYS."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

DEFAULT_OUTPUT = Path("/workspace/ssod/project/artifacts/preflight/statistics/statistical_fixture_report.json")
HISTORICAL_EVIDENCE_SHA256 = "e9197b86831e31d008575ec61fb6fdfb378f1458667f836d61155ebd733b6e95"
RQ3_PRIMARY_SHA256 = "41ac57f7c4f9c19daef8f4f062001bcff1ff41b0af7b4683eb3459e436bf3676"
S6_01_SEED_SHA256 = "2a404ac0f6097cad16816b550fcd1c9ca240fab5535854dc0fd5e4433e49ff3f"
S6_02_PRIMARY_SHA256 = "a9dea0204ec7f38cba428736edbe9f1c70f4583f71103e2b8dbc21be83121c3a"
S6_02_VALIDATOR_SHA256 = "4cca5652003b43dbf34f44dc26ea0566f655bee368df08d51784165a23e4b468"


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path("/workspace/ssod/project"))
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    repo = args.repo_root.resolve()
    sys.path.insert(0, str(repo))

    from src.statistics.rq3_primary import (
        RQ3_ARCHITECTURE_COUNT,
        RQ3_ARCHITECTURE_WEIGHT,
        RQ3_BUDGET_COUNT,
        RQ3_BUDGET_LEVELS,
        RQ3_GREENHOUSE_GEISSER_POLICY,
        RQ3_MAUCHLY_ROLE,
        RQ3_UNCORRECTED_DF_DENOMINATOR,
        RQ3_UNCORRECTED_DF_NUMERATOR,
        RQ3PrimaryError,
        compute_rq3_budget_gain_by_seed,
        rq3_primary_anova,
    )
    from src.statistics.seed_summary import (
        OFFICIAL_SEED_COUNT,
        OFFICIAL_TRAINING_SEEDS,
        REPLICATION_UNIT,
    )

    ev_path = args.output.resolve()
    old_bytes = ev_path.read_bytes()
    old_sha = hashlib.sha256(old_bytes).hexdigest()
    evidence = json.loads(old_bytes.decode("utf-8"))
    checks: list[dict[str, object]] = []

    def check(name: str, ok: bool, detail: object) -> None:
        checks.append({"name": name, "pass": bool(ok), "detail": detail})

    # Historical closure must be preserved exactly.
    check("historical::evidence_sha", old_sha == HISTORICAL_EVIDENCE_SHA256, old_sha)
    check("historical::s6_01", evidence.get("task") == "S6.01" and evidence.get("status") == "PASS"
          and evidence.get("checks_passed") == 14 and evidence.get("checks_total") == 14
          and evidence.get("failed_count") == 0, {
              "task": evidence.get("task"), "status": evidence.get("status"),
              "checks": [evidence.get("checks_passed"), evidence.get("checks_total")],
              "failed": evidence.get("failed_count")})
    check("historical::s6_01_seed_sha",
          evidence.get("source_sha256", {}).get("src/statistics/seed_summary.py") == S6_01_SEED_SHA256,
          evidence.get("source_sha256", {}).get("src/statistics/seed_summary.py"))
    s2 = evidence.get("s6_02_rq2_primary_t_test", {})
    check("historical::s6_02", s2.get("task") == "S6.02" and s2.get("status") == "PASS"
          and s2.get("checks_passed") == 29 and s2.get("checks_total") == 29
          and s2.get("failed_count") == 0, {
              "task": s2.get("task"), "status": s2.get("status"),
              "checks": [s2.get("checks_passed"), s2.get("checks_total")],
              "failed": s2.get("failed_count")})
    check("historical::s6_02_primary_sha",
          s2.get("source_sha256", {}).get("src/statistics/rq2_primary.py") == S6_02_PRIMARY_SHA256,
          s2.get("source_sha256", {}).get("src/statistics/rq2_primary.py"))
    check("historical::s6_02_validator_sha",
          s2.get("source_sha256", {}).get("scripts/S6_02_validate_rq2_primary.py") == S6_02_VALIDATOR_SHA256,
          s2.get("source_sha256", {}).get("scripts/S6_02_validate_rq2_primary.py"))

    # Locked contract.
    contract = [
        ("replication_unit", REPLICATION_UNIT == "TRAINING_SEED_TRAINED_RUN", REPLICATION_UNIT),
        ("seed_count_10", OFFICIAL_SEED_COUNT == 10, OFFICIAL_SEED_COUNT),
        ("budget_levels", RQ3_BUDGET_LEVELS == ("1%", "5%", "10%", "20%"), RQ3_BUDGET_LEVELS),
        ("budget_count_4", RQ3_BUDGET_COUNT == 4, RQ3_BUDGET_COUNT),
        ("architecture_count_2", RQ3_ARCHITECTURE_COUNT == 2, RQ3_ARCHITECTURE_COUNT),
        ("architecture_weight_half", np.isclose(RQ3_ARCHITECTURE_WEIGHT, 0.5), RQ3_ARCHITECTURE_WEIGHT),
        ("df_num_3", RQ3_UNCORRECTED_DF_NUMERATOR == 3, RQ3_UNCORRECTED_DF_NUMERATOR),
        ("df_den_27", RQ3_UNCORRECTED_DF_DENOMINATOR == 27, RQ3_UNCORRECTED_DF_DENOMINATOR),
        ("gg_always", RQ3_GREENHOUSE_GEISSER_POLICY == "ALWAYS", RQ3_GREENHOUSE_GEISSER_POLICY),
        ("mauchly_diagnostic_only", RQ3_MAUCHLY_ROLE == "DIAGNOSTIC_ONLY", RQ3_MAUCHLY_ROLE),
    ]
    for name, ok, detail in contract:
        check("contract::" + name, ok, detail)

    # Golden fixture.
    x = np.asarray([
        [1.0, 1.4, 1.8, 2.1], [1.2, 1.5, 1.9, 2.4],
        [0.9, 1.3, 1.6, 2.0], [1.4, 1.7, 2.2, 2.5],
        [1.1, 1.6, 1.7, 2.3], [1.3, 1.8, 2.0, 2.6],
        [0.8, 1.2, 1.5, 1.9], [1.5, 1.9, 2.4, 2.7],
        [1.0, 1.5, 2.1, 2.2], [1.2, 1.6, 1.8, 2.5],
    ], dtype=float)
    effects = {
        seed: [[float(v - 0.1), float(v + 0.1)] for v in row]
        for seed, row in zip(OFFICIAL_TRAINING_SEEDS, x)
    }
    gains = compute_rq3_budget_gain_by_seed(effects)
    observed = np.asarray([gains[s] for s in OFFICIAL_TRAINING_SEEDS], dtype=float)
    r = rq3_primary_anova(gains)
    m = r["mauchly_diagnostic"]
    golden = [
        ("b_b_s_architecture_mean", np.allclose(observed, x, rtol=0.0, atol=1e-12), observed.tolist()),
        ("n_10", r["n_subjects"] == 10, r["n_subjects"]),
        ("budget_count_4", r["budget_count"] == 4, r["budget_count"]),
        ("budget_means", np.allclose(r["budget_means"], [1.14, 1.55, 1.9, 2.32], rtol=0.0, atol=1e-12), r["budget_means"]),
        ("F", np.isclose(r["f_statistic"], 306.05050505050576, rtol=1e-12, atol=1e-12), r["f_statistic"]),
        ("epsilon_GG", np.isclose(r["epsilon_gg"], 0.5303135665058788, rtol=1e-12, atol=1e-12), r["epsilon_gg"]),
        ("df_GG_num", np.isclose(r["df_gg_numerator"], 1.5909406995176365, rtol=1e-12, atol=1e-12), r["df_gg_numerator"]),
        ("df_GG_den", np.isclose(r["df_gg_denominator"], 14.318466295658729, rtol=1e-12, atol=1e-12), r["df_gg_denominator"]),
        ("p_GG", np.isclose(r["p_gg"], 5.009062437721494e-12, rtol=1e-10, atol=1e-20), r["p_gg"]),
        ("GG_applied", r["greenhouse_geisser_policy"] == "ALWAYS" and r["greenhouse_geisser_applied"] is True,
         [r["greenhouse_geisser_policy"], r["greenhouse_geisser_applied"]]),
        ("Mauchly_not_selector", r["mauchly_role"] == "DIAGNOSTIC_ONLY"
         and m["used_to_select_primary_method"] is False, m),
        ("Mauchly_fixture", m["defined"] is True
         and np.isclose(m["w"], 0.2584006846610435, rtol=1e-10, atol=1e-12)
         and np.isclose(m["chi_square"], 10.450049788164561, rtol=1e-10, atol=1e-12)
         and m["df"] == 5
         and np.isclose(m["p_value"], 0.06344250637550378, rtol=1e-10, atol=1e-12), m),
    ]
    for name, ok, detail in golden:
        check("positive::" + name, ok, detail)

    negative: dict[str, dict[str, object]] = {}

    def blocked(name: str, func) -> None:
        ok, msg = False, None
        try:
            func()
        except (RQ3PrimaryError, ValueError, TypeError) as exc:
            ok, msg = True, str(exc)
        negative[name] = {"blocked": ok, "error": msg}
        check("fail_closed::" + name, ok, msg)

    missing = dict(effects)
    missing.pop(OFFICIAL_TRAINING_SEEDS[-1])
    blocked("missing_seed", lambda: compute_rq3_budget_gain_by_seed(missing))
    blocked("reordered_seeds", lambda: compute_rq3_budget_gain_by_seed(dict(list(effects.items())[::-1])))
    wrong = {s: [list(row) for row in matrix] for s, matrix in effects.items()}
    wrong[OFFICIAL_TRAINING_SEEDS[0]] = [[1.0, 2.0]] * 3
    blocked("wrong_shape", lambda: compute_rq3_budget_gain_by_seed(wrong))
    bad = {s: [list(row) for row in matrix] for s, matrix in effects.items()}
    bad[OFFICIAL_TRAINING_SEEDS[0]][0][0] = float("nan")
    blocked("nonfinite", lambda: compute_rq3_budget_gain_by_seed(bad))
    flat = {s: (1.0, 1.0, 1.0, 1.0) for s in OFFICIAL_TRAINING_SEEDS}
    blocked("degenerate_variance", lambda: rq3_primary_anova(flat))

    rq3_path = repo / "src/statistics/rq3_primary.py"
    seed_path = repo / "src/statistics/seed_summary.py"
    validator_path = repo / "scripts/S6_03_validate_rq3_primary.py"
    check("source::rq3_primary_sha", sha256_file(rq3_path) == RQ3_PRIMARY_SHA256, sha256_file(rq3_path))

    failed = [c for c in checks if not c["pass"]]
    section = {
        "task": "S6.03",
        "tracker_action": "Implement repeated-measures ANOVA + GG ALWAYS",
        "status": "PASS" if not failed else "FAIL",
        "scientific_contract": {
            "effect": "B_b_s",
            "b_b_s_definition": "MEAN_OF_R50_AND_SWIN_T_PAIRED_DELTA_MAP_TEST_WITHIN_BUDGET_AND_SEED",
            "inferential_replication_unit": REPLICATION_UNIT,
            "n_subjects": 10,
            "budget_factor": "REPEATED_CATEGORICAL",
            "budget_levels": list(RQ3_BUDGET_LEVELS),
            "architecture_count": 2,
            "architecture_weight": 0.5,
            "primary_method": "ONE_WAY_REPEATED_MEASURES_ANOVA",
            "greenhouse_geisser_policy": "ALWAYS",
            "mauchly_role": "DIAGNOSTIC_ONLY",
        },
        "positive_fixture": {
            "budget_means": r["budget_means"],
            "f_statistic": r["f_statistic"],
            "df_uncorrected_numerator": r["df_uncorrected_numerator"],
            "df_uncorrected_denominator": r["df_uncorrected_denominator"],
            "epsilon_gg": r["epsilon_gg"],
            "df_gg_numerator": r["df_gg_numerator"],
            "df_gg_denominator": r["df_gg_denominator"],
            "p_gg": r["p_gg"],
            "mauchly_diagnostic": m,
        },
        "negative_fixtures": negative,
        "checks": checks,
        "checks_passed": len(checks) - len(failed),
        "checks_total": len(checks),
        "failed_count": len(failed),
        "failed_checks": [c["name"] for c in failed],
        "historical_s6_02_evidence_sha256_before_extension": old_sha,
        "source_sha256": {
            "src/statistics/rq3_primary.py": sha256_file(rq3_path),
            "src/statistics/seed_summary.py": sha256_file(seed_path),
            "scripts/S6_03_validate_rq3_primary.py": sha256_file(validator_path),
        },
        "scientific_boundaries": {
            "fixture_only_no_official_results": True,
            "test_used": False,
            "hidden_unlabeled_gt_used": False,
            "official_training_authorized": False,
            "final_test_authorized": False,
            "pairwise_contrasts_implemented_in_s6_03": False,
            "holm_f3_implemented_in_s6_03": False,
            "friedman_sensitivity_implemented_in_s6_03": False,
            "mauchly_used_to_select_primary_method": False,
        },
    }
    evidence["s6_03_rq3_gg_repeated_measures_anova"] = section
    ev_path.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")

    print("===== S6.03 RQ3 PRIMARY VALIDATOR =====")
    for item in checks:
        print(f'{item["name"]}={item["pass"]}')
    print(f"CHECKS_PASSED={len(checks)-len(failed)}")
    print(f"CHECKS_TOTAL={len(checks)}")
    print(f"FAILED_COUNT={len(failed)}")
    print("FAILED_CHECKS=" + (",".join(c["name"] for c in failed) if failed else "NONE"))
    print(f"F={r['f_statistic']}")
    print(f"EPSILON_GG={r['epsilon_gg']}")
    print(f"DF_GG_NUM={r['df_gg_numerator']}")
    print(f"DF_GG_DEN={r['df_gg_denominator']}")
    print(f"P_GG={r['p_gg']}")
    print(f"MAUCHLY_DEFINED={m['defined']}")
    print(f"MAUCHLY_USED_TO_SELECT_PRIMARY={m['used_to_select_primary_method']}")
    print(f"EVIDENCE={ev_path}")
    print(f"EVIDENCE_SHA256={sha256_file(ev_path)}")
    print("STATUS=" + ("PASS" if not failed else "FAIL"))
    print("S6_03_RQ3_GG_ANOVA_FIXTURE=" + ("PASS" if not failed else "FAIL"))
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
