"""S5.09 fail-closed preflight validator for final-test isolation."""

import argparse
import hashlib
import json
import sys
from copy import deepcopy
from pathlib import Path


DEFAULT_ROOT = Path("/workspace/ssod/project")
DEFAULT_EVIDENCE = (
    DEFAULT_ROOT
    / "artifacts"
    / "preflight"
    / "firewall"
    / "test_firewall_preflight.json"
)

EXPECTED_CANONICAL_SHA256 = {
    "sn-article.tex":
        "8d97840cc546a58aef7354af0c856119b1aea16f8be1cb7bbc1ab1a324d40fcc",
    "IMPLEMENTATION_HANDOFF.md":
        "540c94746571612ff819fb62a1e9fac6123c4035bc2642518eaa7740770820d0",
    "ARTIFACT_AND_EVIDENCE_CONTRACT.md":
        "885d4eeb66857e99744f6e8a27083e5bb695bbcc0d172cf4ebff722c53f655a4",
}

EXPECTED_GOVERNANCE_SHA256 = {
    "artifacts/governance/scientific_source_manifest.json":
        "108b9e64bcbe94096d2e4cf576b83a7a11380b109939c88616cecb46414b4f16",
    "artifacts/governance/implementation_contract_manifest.json":
        "d73423a0977be30924d5c1ce7ee47c132aed63d9e6aca0835faefda52659266c",
    "artifacts/governance/artifact_contract_manifest.json":
        "196c4f859e588ca8716a71414dd24fe979b4f8191fd6c20b45c9ed8f3a664950",
}

EXPECTED_SOURCE_SHA256 = {
    "src/utils/pre_run_guardrails.py":
        "aa3a874583d6bf4714e4882ef046d846f9e62526493bc754e11988eddda1e6ca",
    "src/evaluation/qpseudo.py":
        "300fc37b08e4f1e9e3e97c29bc714aef2b219386c2932cc34669923707e57bcc",
    "scripts/S5_08_validate_qpseudo_gt_firewall.py":
        "7559b3d253609adfc6ce0a5a1b3bb445ccb60c8aa3da0fe085ff9c4811df86d7",
}

QPSEUDO_EVIDENCE_REL = (
    "artifacts/preflight/evaluation/qpseudo_path_golden_test.json"
)
QPSEUDO_EVIDENCE_SHA256 = (
    "50be7f43887f039acb4bee0aec56c919cef0654e8c8cdf037e3d8bb6cabe207d"
)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", default=str(DEFAULT_ROOT))
    parser.add_argument("--output", default=str(DEFAULT_EVIDENCE))
    return parser.parse_args()


def main():
    args = parse_args()
    root = Path(args.repo_root).resolve()
    output = Path(args.output).resolve()
    sys.path.insert(0, str(root))

    import src.utils.pre_run_guardrails as pr

    print("===== S5.09 TEST FIREWALL VALIDATOR =====")
    checks = []

    def check(name, passed, detail=None):
        passed = bool(passed)
        checks.append({
            "name": name,
            "pass": passed,
            "detail": detail,
        })
        print(f"{name}={passed}")

    canonical_sha = {}
    for rel, expected_sha in EXPECTED_CANONICAL_SHA256.items():
        observed_sha = sha256(root / rel)
        canonical_sha[rel] = observed_sha
        check(
            f"canonical::{rel}",
            observed_sha == expected_sha,
            observed_sha,
        )

    governance_sha = {}
    for rel, expected_sha in EXPECTED_GOVERNANCE_SHA256.items():
        observed_sha = sha256(root / rel)
        governance_sha[rel] = observed_sha
        check(
            f"governance::{rel}",
            observed_sha == expected_sha,
            observed_sha,
        )

    source_sha = {}
    for rel, expected_sha in EXPECTED_SOURCE_SHA256.items():
        observed_sha = sha256(root / rel)
        source_sha[rel] = observed_sha
        check(
            f"source::{rel}",
            observed_sha == expected_sha,
            observed_sha,
        )

    integrity = pr.verify_locked_source_integrity(root)
    check(
        "governance::locked_source_integrity",
        all(integrity.values()),
        integrity,
    )

    qpseudo_path = root / QPSEUDO_EVIDENCE_REL
    qpseudo_sha = sha256(qpseudo_path)
    check(
        "qpseudo::evidence_sha",
        qpseudo_sha == QPSEUDO_EVIDENCE_SHA256,
        qpseudo_sha,
    )
    qpseudo = load_json(qpseudo_path)
    s5_08 = qpseudo.get("s5_08_hidden_u_test_gt_firewall", {})
    qp_boundaries = s5_08.get("scientific_boundaries", {})
    qp_rule = s5_08.get("canonical_rule", {})
    check(
        "qpseudo::fixed_validation_only",
        qp_boundaries.get("fixed_validation_only") is True
        and qp_rule.get("dataset") == "fixed_validation",
        {
            "fixed_validation_only": qp_boundaries.get(
                "fixed_validation_only"
            ),
            "dataset": qp_rule.get("dataset"),
        },
    )
    check(
        "qpseudo::test_not_used",
        qp_boundaries.get("test_used") is False
        and qp_rule.get("test_gt_prohibited") is True,
        {
            "test_used": qp_boundaries.get("test_used"),
            "test_gt_prohibited": qp_rule.get("test_gt_prohibited"),
        },
    )

    config_files = sorted((root / "configs").rglob("*.py"))
    test_dataloader_hits = []
    test_evaluator_hits = []
    for path in config_files:
        text = path.read_text(encoding="utf-8-sig")
        rel = str(path.relative_to(root)).replace("\\\\", "/")
        if "test_dataloader" in text:
            test_dataloader_hits.append(rel)
        if "test_evaluator" in text:
            test_evaluator_hits.append(rel)

    check(
        "preflight::test_dataloader_absent",
        not test_dataloader_hits,
        test_dataloader_hits,
    )
    check(
        "preflight::test_evaluator_absent",
        not test_evaluator_hits,
        test_evaluator_hits,
    )

    sup = pr.build_expected_pre_run_identity(
        root,
        method="SUP",
        architecture="R50-FPN",
        budget="10pct",
        seed_index=1,
    )
    ssl = pr.build_expected_pre_run_identity(
        root,
        method="SSL",
        architecture="R50-FPN",
        budget="10pct",
        seed_index=1,
    )

    sup_positive = pr.evaluate_pre_run_identity(root, deepcopy(sup))
    ssl_positive = pr.evaluate_pre_run_identity(root, deepcopy(ssl))
    check(
        "runtime::sup_positive_identity",
        sup_positive["all_checks_pass"]
        and sup_positive["launch_status"] == "PASS",
    )
    check(
        "runtime::ssl_positive_identity",
        ssl_positive["all_checks_pass"]
        and ssl_positive["launch_status"] == "PASS",
    )

    required_firewall = {
        "checkpoint_selection_metric": "bbox_mAP_50_95_validation",
        "test_usage": "FINAL_ONLY",
        "test_access_authorized": False,
        "test_accessed": False,
        "test_access_reason": "FINAL_ONLY",
        "final_evaluation_status": "NOT_AUTHORIZED",
    }
    for field, expected_value in required_firewall.items():
        check(
            f"firewall_expected::{field}",
            sup.get(field) == expected_value
            and ssl.get(field) == expected_value,
            {
                "SUP": sup.get(field),
                "SSL": ssl.get(field),
            },
        )

    mutation_cases = {
        "checkpoint_selection_metric": "bbox_mAP_50_95_test",
        "test_usage": "DEVELOPMENT",
        "test_access_authorized": True,
        "test_accessed": True,
        "test_access_reason": "DEVELOPMENT",
        "final_evaluation_status": "AUTHORIZED",
    }
    mutation_results = {}
    for field, bad_value in mutation_cases.items():
        observed = deepcopy(sup)
        observed[field] = bad_value
        report = pr.evaluate_pre_run_identity(root, observed)
        mismatch_fields = [
            item["field"]
            for item in report["mismatches"]
        ]
        blocked_by_assert = False
        try:
            pr.assert_pre_run_identity(root, observed)
        except pr.PreRunGuardrailError:
            blocked_by_assert = True
        passed = (
            not report["all_checks_pass"]
            and report["launch_status"] == "BLOCKED"
            and field in mismatch_fields
            and blocked_by_assert
        )
        mutation_results[field] = {
            "blocked": passed,
            "mismatch_fields": mismatch_fields,
        }
        check(
            f"fail_closed::{field}",
            passed,
            mutation_results[field],
        )

    missing_results = {}
    for field in required_firewall:
        observed = deepcopy(sup)
        del observed[field]
        report = pr.evaluate_pre_run_identity(root, observed)
        blocked_by_assert = False
        try:
            pr.assert_pre_run_identity(root, observed)
        except pr.PreRunGuardrailError:
            blocked_by_assert = True
        passed = (
            not report["all_checks_pass"]
            and report["launch_status"] == "BLOCKED"
            and field in report["missing_fields"]
            and blocked_by_assert
        )
        missing_results[field] = {
            "blocked": passed,
            "missing_fields": report["missing_fields"],
        }
        check(
            f"fail_closed_missing::{field}",
            passed,
            missing_results[field],
        )

    ablation_attempt = deepcopy(sup)
    ablation_attempt["condition_role"] = "ABLATION"
    ablation_attempt["variant"] = "NON_MAIN"
    ablation_attempt["test_access_authorized"] = True
    ablation_attempt["test_accessed"] = True
    ablation_report = pr.evaluate_pre_run_identity(
        root,
        ablation_attempt,
    )
    ablation_mismatches = [
        item["field"]
        for item in ablation_report["mismatches"]
    ]
    ablation_blocked_by_assert = False
    try:
        pr.assert_pre_run_identity(root, ablation_attempt)
    except pr.PreRunGuardrailError:
        ablation_blocked_by_assert = True
    ablation_blocked = (
        not ablation_report["all_checks_pass"]
        and ablation_report["launch_status"] == "BLOCKED"
        and "test_access_authorized" in ablation_mismatches
        and "test_accessed" in ablation_mismatches
        and ablation_blocked_by_assert
    )
    check(
        "firewall::non_main_ablation_test_access_blocked",
        ablation_blocked,
        {
            "condition_role": "ABLATION",
            "variant": "NON_MAIN",
            "mismatch_fields": ablation_mismatches,
        },
    )

    final_test_dir = root / "artifacts" / "final_test"
    check(
        "preflight::final_test_artifacts_not_materialized",
        not final_test_dir.exists(),
        str(final_test_dir),
    )

    failed = [item for item in checks if not item["pass"]]
    print(f"CHECKS_PASSED={len(checks) - len(failed)}")
    print(f"CHECKS_TOTAL={len(checks)}")
    print(f"FAILED_COUNT={len(failed)}")
    print(
        "FAILED_CHECKS="
        + (
            "NONE"
            if not failed
            else ",".join(item["name"] for item in failed)
        )
    )

    if failed:
        raise SystemExit(1)

    validator_rel = "scripts/S5_09_validate_test_firewall.py"
    source_sha[validator_rel] = sha256(root / validator_rel)

    report = {
        "schema_version": "1.0",
        "evidence_type": "test_firewall_preflight",
        "task": "S5.09",
        "status": "PASS",
        "tracker_action": "Keep test closed",
        "constraint": (
            "No checkpoint/model/tuning/ablation selection using test."
        ),
        "canonical_proofs": {
            "test_evaluator_not_invoked_in_preflight": True,
            "test_dataloader_not_available_to_checkpoint_selection_path": True,
            "qpseudo_uses_validation_only": True,
            "non_main_ablation_test_access": "BLOCKED",
        },
        "firewall_state": {
            "checkpoint_selection_metric":
                "bbox_mAP_50_95_validation",
            "test_usage": "FINAL_ONLY",
            "test_access_authorized": False,
            "test_accessed": False,
            "test_access_reason": "FINAL_ONLY",
            "final_evaluation_status": "NOT_AUTHORIZED",
        },
        "runtime_negative_tests": {
            "mutation_cases": mutation_results,
            "missing_field_cases": missing_results,
            "non_main_ablation_test_access_blocked":
                ablation_blocked,
        },
        "qpseudo_firewall": {
            "evidence_sha256": qpseudo_sha,
            "fixed_validation_only": True,
            "test_used": False,
        },
        "preflight_materialization": {
            "final_test_artifacts_created": False,
            "test_evaluator_config_present": False,
            "test_dataloader_config_present": False,
        },
        "scientific_boundaries": {
            "scientific_protocol_changed": False,
            "test_used_for_checkpoint_selection": False,
            "test_used_for_model_selection": False,
            "test_used_for_tuning": False,
            "test_used_for_ablation_selection": False,
            "qpseudo_test_used": False,
            "official_training_authorized": False,
            "final_test_authorized": False,
        },
        "source_sha256": source_sha,
        "canonical_sha256": canonical_sha,
        "governance_sha256": governance_sha,
        "checks": checks,
        "checks_passed": len(checks),
        "checks_total": len(checks),
        "failed_count": 0,
        "failed_checks": [],
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print(f"EVIDENCE={output}")
    print(f"EVIDENCE_SHA256={sha256(output)}")
    print("STATUS=PASS")
    print("S5_09_TEST_FIREWALL=PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
