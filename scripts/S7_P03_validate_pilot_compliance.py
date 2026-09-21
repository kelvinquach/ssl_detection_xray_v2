#!/usr/bin/env python3
import argparse
import hashlib
import json
from pathlib import Path

TASK = "S7.P03"
SCHEMA_VERSION = "1.0"
ARTIFACT_TYPE = "PILOT_REPRODUCIBILITY_AND_LEAKAGE_REPORT"
TECHNICAL_SEED = 204886845

EXPECTED_SHA256 = {
    "artifacts/preflight/pilot/pilot_end_to_end_report.json": "e6a5b225ec724c2ca9699e21f6d22e6ffc9799c01238d60a3c207e07b1e01401",
    "artifacts/preflight/ssl/gt_bbox_alignment_report.json": "c6822e14e53f4773779a64d21af85c8d37a73d6011134536a4987dbb8354e59e",
    "artifacts/preflight/ssl/pseudo_bbox_alignment_report.json": "27b69568818301ef21a1896fdca290ea4be2fca2ea348d3248e874e4ccc6142c",
    "artifacts/preflight/ssl/ema_timing_test.json": "2fa821dbe18baea43486bfa163ba578ca15ae71eecf54a2352e208b9cfc0eda0",
    "artifacts/preflight/resume/resume_equivalence_test.json": "f92db6919196d9f2f3098053017af57a115896c4544ef5ecc249c097a0ad5e80",
    "artifacts/preflight/seed/training_seed_invariance_audit.json": "209c9c3fd30da7771a4663e8bac5e5ab27c80cf79e912a4b36c193e0315b797a",
    "artifacts/preflight/seed/seed_propagation_audit.json": "c6e8d5d9e3ec30af0851e94e26d4b3f95b1632f10203628da0e539d313b4f1c6",
    "artifacts/preflight/seed/dataloader_worker_seed_audit.json": "f7fa227e60f62139c8ecdd48eb0356bac92873b4f7cf81963be631d069232921",
    "artifacts/preflight/firewall/hidden_u_gt_firewall_report.json": "f52fd896ea38b59fd353f191fae74a8303053eea184572973ffa18219432ac01",
    "artifacts/preflight/firewall/test_firewall_preflight.json": "f89161690013d4045cfafa8f677f0464e6c171cae9cb206bbf764bef044915b7",
    "data/manifests/leakage_check_report.json": "c794d7be3deaa9661c40f990296d03eb7f664ef81887ff62636d924f9a30d2fc",
    "data/manifests/phase2F_leakage_check.json": "c3d26982c4d63dca6bc817f01d589b5d096ff10cbb2dcaaf8557172d1e0ac9cc",
}

TEST_FIREWALL_TARGETS = {
    "preflight::test_dataloader_absent",
    "preflight::test_evaluator_absent",
    "firewall_expected::test_access_authorized",
    "firewall_expected::test_accessed",
    "firewall_expected::test_access_reason",
    "firewall_expected::final_evaluation_status",
    "firewall::non_main_ablation_test_access_blocked",
    "preflight::final_test_artifacts_not_materialized",
}

def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

def load_json(root: Path, rel: str):
    p = root / rel
    if not p.is_file():
        raise FileNotFoundError(rel)
    return json.loads(p.read_text(encoding="utf-8"))

def dict_checks_true(obj, names):
    checks = obj.get("checks")
    return isinstance(checks, dict) and all(checks.get(name) is True for name in names)

def test_firewall_targets_pass(obj):
    checks = obj.get("checks")
    if not isinstance(checks, list):
        return False
    found = {}
    for item in checks:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name") or item.get("check_id") or item.get("check") or "")
        if name in TEST_FIREWALL_TARGETS:
            found[name] = item.get("pass") is True
    return set(found) == TEST_FIREWALL_TARGETS and all(found.values())

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", default=".")
    parser.add_argument(
        "--output",
        default="artifacts/preflight/pilot/pilot_reproducibility_and_leakage_report.json",
    )
    args = parser.parse_args()

    root = Path(args.repo_root).resolve()
    output = Path(args.output)
    if not output.is_absolute():
        output = root / output

    evidence = {rel: load_json(root, rel) for rel in EXPECTED_SHA256}
    checks = {}

    for rel, expected in EXPECTED_SHA256.items():
        checks["sha256::" + rel] = sha256(root / rel) == expected

    pilot = evidence["artifacts/preflight/pilot/pilot_end_to_end_report.json"]
    gt = evidence["artifacts/preflight/ssl/gt_bbox_alignment_report.json"]
    pseudo = evidence["artifacts/preflight/ssl/pseudo_bbox_alignment_report.json"]
    ema = evidence["artifacts/preflight/ssl/ema_timing_test.json"]
    resume = evidence["artifacts/preflight/resume/resume_equivalence_test.json"]
    inv = evidence["artifacts/preflight/seed/training_seed_invariance_audit.json"]
    prop = evidence["artifacts/preflight/seed/seed_propagation_audit.json"]
    worker = evidence["artifacts/preflight/seed/dataloader_worker_seed_audit.json"]
    hidden = evidence["artifacts/preflight/firewall/hidden_u_gt_firewall_report.json"]
    testfw = evidence["artifacts/preflight/firewall/test_firewall_preflight.json"]
    leak = evidence["data/manifests/leakage_check_report.json"]
    leak2 = evidence["data/manifests/phase2F_leakage_check.json"]

    groups = {}

    groups["code_completes_intended_path"] = (
        pilot.get("status") == "PASS"
        and pilot.get("overall_pass") is True
        and pilot.get("all_checks_pass") is True
        and dict_checks_true(
            pilot,
            [
                "r50_optimizer_update_exactly_one",
                "r50_labeled_unlabeled_exposure",
                "r50_pseudo_path_exercised",
                "r50_validation_once",
                "r50_best_last",
                "swin_optimizer_update_exactly_one",
                "swin_labeled_unlabeled_exposure",
                "swin_pseudo_path_exercised",
                "swin_validation_once",
                "swin_best_last",
            ],
        )
    )

    groups["image_annotation_alignment"] = (
        gt.get("status") == "PASS"
        and gt.get("overall_pass") is True
        and gt.get("all_checks_pass") is True
        and gt.get("scope", {}).get("images_checked") == 4894
        and gt.get("scope", {}).get("annotations_checked") == 36096
        and dict_checks_true(
            gt,
            [
                "jpeg_availability_pass",
                "dimension_consistency_pass",
                "bbox_geometry_pass",
                "reference_integrity_pass",
                "automated_checks_status_pass",
            ],
        )
    )

    groups["pseudo_box_transform_alignment"] = (
        pseudo.get("all_checks_pass") is True
        and dict_checks_true(
            pseudo,
            [
                "framework_teacher_projects_inverse_homography",
                "framework_student_projects_forward_homography",
                "runtime_pseudo_box_alignment",
                "runtime_hidden_u_gt_unused",
            ],
        )
    )

    groups["ema_timing"] = (
        ema.get("all_checks_pass") is True
        and dict_checks_true(
            ema,
            [
                "counter_zero_before_optimizer_step",
                "no_ema_without_actual_optimizer_update",
                "counter_one_after_optimizer_step",
                "ema_once_after_actual_optimizer_update",
                "no_duplicate_ema_without_new_update",
            ],
        )
        and dict_checks_true(
            pilot,
            ["r50_ema_matches_update", "swin_ema_matches_update"],
        )
    )

    groups["checkpoint_save_select_resume"] = (
        dict_checks_true(pilot, ["r50_best_last", "swin_best_last"])
        and resume.get("all_checks_pass") is True
        and resume.get("training_seed") == TECHNICAL_SEED
        and dict_checks_true(
            resume,
            [
                "r50_1pct_latest_resume_interval_exact",
                "r50_1pct_native_default_checkpoint_disabled",
                "swin_t_1pct_latest_resume_interval_exact",
                "swin_t_1pct_native_default_checkpoint_disabled",
            ],
        )
    )

    groups["protocol_seed_reproducibility"] = (
        inv.get("status") == "PASS"
        and inv.get("training_seed_count") == 10
        and inv.get("partition_seed") == 42
        and dict_checks_true(
            inv,
            [
                "locked_training_seed_count_is_10",
                "all_10_training_seeds_preserve_train_val_test_membership",
                "all_10_training_seeds_preserve_labeled_membership",
                "all_10_training_seeds_preserve_unlabeled_membership",
                "all_seed_runs_pass",
            ],
        )
        and prop.get("status") == "PASS"
        and dict_checks_true(
            prop,
            [
                "locked_training_seed_count_is_10",
                "selected_seed_in_locked_mapping",
                "numpy_seed_observed_matches",
                "torch_initial_seed_observed_matches",
                "cuda_initial_seeds_match",
                "sampler_seed_controlled_by_training_seed",
            ],
        )
        and worker.get("status") == "PASS"
        and dict_checks_true(
            worker,
            [
                "locked_training_seed_count_is_10",
                "selected_seed_in_locked_mapping",
                "audit_num_workers_is_multiworker",
                "observed_expected_worker_count",
                "observed_worker_ids_match",
                "all_worker_seed_checks_pass",
            ],
        )
        and dict_checks_true(pilot, ["r50_same_seed", "swin_same_seed"])
    )

    groups["hidden_u_and_test_firewall"] = (
        hidden.get("overall_pass") is True
        and hidden.get("all_checks_pass") is True
        and hidden.get("hidden_u_master_gt_opened") is False
        and hidden.get("real_unlabeled_coco_modified") is False
        and hidden.get("official_training_authorized") is False
        and dict_checks_true(
            hidden,
            [
                "R50-FPN:1pct_current_ssl_firewall_pass",
                "Swin-T-FPN:1pct_current_ssl_firewall_pass",
                "contaminated_unlabeled_source_blocked",
                "active_ssl_pre_run_blocks_firewall_failure",
                "hidden_u_master_gt_not_opened",
                "real_unlabeled_coco_not_modified",
            ],
        )
        and testfw.get("status") == "PASS"
        and test_firewall_targets_pass(testfw)
        and pilot.get("test_accessed") is False
        and pilot.get("final_test_authorized") is False
        and pilot.get("official_training_authorized") is False
        and dict_checks_true(pilot, ["r50_test_firewall", "swin_test_firewall"])
    )

    groups["no_data_leakage"] = (
        leak.get("status") == "PASS"
        and leak.get("unit_of_split") == "image_id"
        and leak.get("completeness_pass") is True
        and leak.get("zero_overlap_pass") is True
        and leak2.get("status") == "PASS"
        and leak2.get("val_test_isolation_pass") is True
        and leak2.get("labeled_unlabeled_disjoint_and_complete_pass") is True
    )

    checks.update({"group::" + k: v for k, v in groups.items()})
    checks["governance::scientific_protocol_change_false"] = (
        pilot.get("scientific_protocol_change") is False
        and gt.get("scientific_protocol_change") is False
    )
    checks["governance::performance_driven_tuning_false"] = (
        pilot.get("performance_driven_tuning_used") is False
    )
    checks["governance::official_training_not_authorized"] = (
        pilot.get("official_training_authorized") is False
    )
    checks["governance::final_test_not_authorized"] = (
        pilot.get("final_test_authorized") is False
        and pilot.get("test_accessed") is False
    )

    failed = [name for name, value in checks.items() if value is not True]
    report = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": ARTIFACT_TYPE,
        "task": TASK,
        "status": "PASS" if not failed else "FAIL",
        "scientific_protocol_change": False,
        "official_training_authorized": False,
        "final_test_authorized": False,
        "test_accessed": False,
        "performance_driven_tuning_used": False,
        "technical_training_seed": TECHNICAL_SEED,
        "compliance_groups": groups,
        "evidence_sources": {
            rel: {"sha256": EXPECTED_SHA256[rel]} for rel in EXPECTED_SHA256
        },
        "checks": checks,
        "check_count": len(checks),
        "failed_checks": failed,
        "all_checks_pass": not failed,
        "overall_pass": not failed,
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_suffix(output.suffix + ".tmp")
    tmp.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
    tmp.replace(output)

    print("=== S7.P03 PILOT COMPLIANCE VALIDATOR ===")
    print("REPORT=" + str(output))
    print("COMPLIANCE_GROUP_COUNT=" + str(len(groups)))
    print("TOTAL_CHECKS=" + str(len(checks)))
    print("FAILED_CHECKS=" + str(len(failed)))
    if failed:
        for name in failed:
            print("FAILED=" + name)
    print("S7_P03_PILOT_COMPLIANCE=" + ("PASS" if not failed else "FAIL"))
    raise SystemExit(0 if not failed else 1)

if __name__ == "__main__":
    main()
