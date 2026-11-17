#!/usr/bin/env python3
"""G5.1 Metric/Output Contract Preflight Validator.

Scientific scope:
- validates locked metric/output schemas for B0/B1/B2/B3;
- validates mandatory provenance and selected safety/firewall fields;
- does NOT train models, run VLM inference, access evaluation images, or choose methods.

Standard-library only for CSV/JSON/JSONL checks.
Parquet schema inspection uses optional pyarrow. If unavailable and a selected mode
requires Parquet inspection, the result is BLOCKED (never silently PASS).
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
import tempfile
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

SCHEMA_VERSION = "1.0.0"
CONTRACT_FILENAME = "SSOD_VLM_METRIC_OUTPUT_CONTRACT_V1_0_LOCKED_VI.md"
CONTRACT_SHA256 = "0316e7249ccaa681fe4762c6d4ed8095f23c94e7a9e916905fa974b629003c71"

# ---------- Locked exact legacy schemas ----------
VALIDATION_HISTORY_HEADER = [
    "optimizer_update", "bbox_mAP_50_95", "AP50", "AP75",
    "AR_50_95_max100", "model_role", "checkpoint_candidate",
    "best_so_far", "checkpoint_path",
]

CLASSWISE_VALIDATION_AP_HEADER = [
    "category_id", "class_name", "AP_50_95", "AP50", "AP75",
]

TRAINING_SUMMARY_KEYS = [
    "amp_skipped_step_count", "effective_labeled_batch", "effective_unlabeled_batch",
    "ema_update_count", "ema_updates_match_optimizer_updates",
    "expected_optimizer_updates", "expected_update_budget_reached", "l_u_ratio",
    "labeled_exposure_matches_update_count", "labeled_images_consumed_total",
    "observed_optimizer_updates", "unlabeled_exposure_matches_update_count",
    "unlabeled_images_consumed_total", "validation_events_observed",
]

VALIDATION_BEST_TEACHER_KEYS = [
    "checkpoint_path", "checkpoint_sha256", "rule", "selected_metric_value",
    "selected_update", "selection_metric", "selection_model", "selection_split",
    "tie_break",
]

FINAL_TEST_MASTER_HEADER = [
    "run_id", "condition_role", "method", "architecture", "budget", "seed_index",
    "training_seed", "best_checkpoint_sha256", "best_validation_mAP_50_95",
    "test_mAP_50_95", "test_AP50", "test_AP75", "test_AR_50_95_max100",
    "test_Recall_tau_eval", "test_FP_total", "test_FP_per_image",
    "AP_Aortic_enlargement", "AP_Atelectasis", "AP_Calcification",
    "AP_Cardiomegaly", "AP_Consolidation", "AP_ILD", "AP_Infiltration",
    "AP_Lung_Opacity", "AP_Nodule_Mass", "AP_Other_lesion",
    "AP_Pleural_effusion", "AP_Pleural_thickening", "AP_Pneumothorax",
    "AP_Pulmonary_fibrosis", "FP_total_negative", "FP_per_negative",
    "negative_FAR", "Q_pseudo", "Q_pseudo_status", "final_evaluation_status",
]

PAIRED_EFFECTS_HEADER = [
    "pair_id", "architecture", "budget", "seed_index", "training_seed",
    "mAP_test_SUP", "mAP_test_SSL", "delta_mAP_test",
    "FP_per_negative_test_SUP", "FP_per_negative_test_SSL",
    "delta_FP_per_negative_test", "AP_test_Atelectasis_SUP",
    "AP_test_Atelectasis_SSL", "delta_AP_test_Atelectasis",
    "AP_test_Pneumothorax_SUP", "AP_test_Pneumothorax_SSL",
    "delta_AP_test_Pneumothorax", "Q_pseudo_validation_SSL",
]

METHOD_PAIR_EFFECTS_HEADER = [
    "schema_version", "comparison_id", "architecture", "budget", "seed_index",
    "training_seed", "reference_method", "comparison_method", "reference_run_id",
    "comparison_run_id", "metric_name", "metric_unit", "reference_value",
    "comparison_value", "delta_comparison_minus_reference",
    "evaluator_config_sha256", "checkpoint_rule_match", "training_seed_match",
    "architecture_match", "budget_match", "pair_valid", "evaluation_role",
    "evaluation_set_id", "evaluation_set_sha256",
]

QPSEUDO_REQUIRED_KEYS = [
    "PL_AP50", "PL_AP75", "PL_mAP_50_95", "acceptance", "accepted_pseudo_count",
    "checkpoint_role", "checkpoint_sha256", "dataset", "evaluator_config_sha256",
    "hidden_u_gt_used", "pseudo_acceptance_config_sha256", "pseudo_set",
    "schema_version", "test_used", "validation_image_count", "validation_split_sha256",
]

# ---------- B2/B3 locked schemas ----------
B2_REFERENCE_FIELDS = [
    "protocol_version", "run_id", "architecture", "budget", "training_seed",
    "checkpoint_role", "checkpoint_sha256", "validation_role", "logical_batch_id",
    "image_id", "native_candidate_index", "candidate_id", "bbox_original_xyxy",
    "bbox_teacher_xyxy", "class_id", "class_name", "teacher_score", "reg_unc",
    "initial_keep", "adaptive_tau", "branch_H_or_U", "K_c", "q_c", "b2_selected",
    "b2_slot_id", "selection_source", "is_mined", "classification_weight_multiplier",
    "image_cap_state", "global_cap_state", "fifo_hash_before", "fifo_hash_after",
]

TEACHER_CANDIDATE_FIELDS = [
    "protocol_version", "run_id", "architecture", "budget", "training_seed",
    "checkpoint_role", "checkpoint_sha256", "validation_role", "logical_batch_id",
    "image_id", "native_candidate_index", "candidate_id", "bbox_original_xyxy",
    "bbox_teacher_xyxy", "class_id", "class_name", "teacher_score", "reg_unc",
    "initial_keep", "adaptive_tau", "branch_H_or_U", "K_c", "q_c", "b2_selected",
    "b2_slot_id", "image_cap_state", "global_cap_state", "fifo_hash_before",
    "fifo_hash_after",
]

VLM_REQUEST_FIELDS = [
    "request_id", "candidate_id", "analysis_role", "control_id", "review_branch",
    "review_order", "model_id", "model_revision", "model_hash", "processor_hash",
    "prompt_hash", "renderer_hash", "decoding_config_hash", "rendered_image_hash",
    "vlm_cache_key", "class_term", "payload_audit_pass", "contains_teacher_score",
    "contains_teacher_rank", "contains_b2_selected", "contains_gt",
    "contains_annotation", "contains_clinical_metadata",
]

VLM_RESPONSE_FIELDS = [
    "request_id", "candidate_id", "vlm_cache_key", "response_hash",
    "raw_response_text", "parse_status", "decision", "region_relevant",
    "abstain_reason", "latency_ms", "analysis_role", "control_id",
]

B3_VARIANT_FIELDS = [
    "run_id", "validation_role", "logical_batch_id", "image_id", "candidate_id",
    "class_id", "class_name", "variant_id", "branch_H_or_U", "K_c", "q_c",
    "b2_selected", "b3_selected", "b2_slot_id", "selection_status",
    "replacement_candidate_id", "h_reviewed", "h_valid_reject",
    "h_region_valid_reject", "h_suppressed", "gate_U_state", "gate_H_state",
    "vlm_cache_key", "fallback_reason", "image_cap_state", "global_cap_state",
]

GT_MATCHING_FIELDS = [
    "candidate_id", "image_id", "class_id", "class_name", "matching_rule_id",
    "iou_threshold", "matched_gt_annotation_id", "matched_iou", "match_status", "Y_i",
]

OFFLINE_SUMMARY_HEADER = [
    "schema_version", "class_id", "class_name", "action_branch", "n_images",
    "n_reviewed", "n_actionable", "n_support", "n_replacements", "q_total",
    "n_reject", "n_suppressions", "n_false_suppressions", "PPV_U", "PPV_U_LCB",
    "PPV_U_UCB", "U_utility", "U_utility_LCB", "U_utility_UCB", "U_coverage",
    "U_abstention", "CR_H", "CR_H_LCB", "CR_H_UCB", "FSR_H", "FSR_H_LCB",
    "FSR_H_UCB", "H_coverage", "H_abstention", "harm_real_minus_sham", "harm_LCB",
    "harm_UCB", "bootstrap_reps", "cluster_unit", "family_size", "alpha_adjusted",
    "support_status", "gate_status", "reason_code",
]

FALSIFICATION_HEADER = [
    "control_id", "control_name", "action_branch", "class_id", "class_name",
    "analysis_role", "n_images", "n_pairs", "numerator", "denominator",
    "metric_name", "point_estimate", "CI_low", "CI_high", "threshold",
    "threshold_comparator", "verdict", "control_seed", "control_manifest_hash",
]

MATCHED_BURDEN_HEADER = [
    "action_branch", "class_id", "class_name", "real_intervention_count",
    "sham_intervention_count", "real_metric", "sham_metric", "real_minus_sham",
    "difference_LCB", "difference_UCB", "sham_seed", "sham_manifest_hash", "verdict",
]

OFFLINE_AUDIT_REQUIRED_KEYS = [
    "protocol_version", "candidate_pool_hash", "b2_reference_hash",
    "phase_A_invariants_pass", "all_off_parity_pass", "confidence_blind_pass",
    "historical_test_used_for_design", "hidden_unlabeled_train_gt_used",
    "U_gate_pass_classes", "U_gate_fail_classes", "U_gate_insufficient_classes",
    "H_gate_pass_classes", "H_gate_fail_classes", "H_gate_insufficient_classes",
    "critical_falsification_status", "F11_status", "intervention_coverage",
    "intervention_coverage_LCB", "intervention_coverage_UCB", "global_status",
    "global_reason", "student_training_authorized",
]

CLASS_NAMES = [
    "Aortic enlargement", "Atelectasis", "Calcification", "Cardiomegaly",
    "Consolidation", "ILD", "Infiltration", "Lung Opacity", "Nodule/Mass",
    "Other lesion", "Pleural effusion", "Pleural thickening", "Pneumothorax",
    "Pulmonary fibrosis",
]

ENUMS = {
    "variant_id": {"B2", "B3-U", "B3-H", "B3-UH", "B3-ALL-OFF"},
    "selection_status": {"SHARED", "REMOVED", "GAINED", "UNCHANGED", "NOT_APPLICABLE"},
    "decision": {"support", "reject", "uncertain"},
    "region_relevant": {"yes", "no", "uncertain"},
    "match_status": {"MATCHED", "UNMATCHED", "EVAL_IGNORED"},
    "gate_status": {"PASS", "FAIL", "INSUFFICIENT"},
    "falsification_verdict": {"PASS", "FAIL", "INSUFFICIENT", "CONTROL_UNAVAILABLE", "NOT_APPLICABLE"},
    "global_status": {"GO_TO_STUDENT_TRAINING_CONSIDERATION", "HOLD", "STOP_B3", "TECHNICAL_FAIL"},
    "selection_source": {"H", "M2", "Z2", "NOT_SELECTED"},
}

RATE_FIELDS_01 = {
    "bbox_mAP_50_95", "AP50", "AP75", "AR_50_95_max100", "test_mAP_50_95",
    "test_AP50", "test_AP75", "test_AR_50_95_max100", "test_Recall_tau_eval",
    "negative_FAR", "PL_AP50", "PL_AP75", "PL_mAP_50_95", "AP_50_95",
    "PPV_U", "PPV_U_LCB", "PPV_U_UCB", "CR_H", "CR_H_LCB", "CR_H_UCB",
    "FSR_H", "FSR_H_LCB", "FSR_H_UCB", "U_coverage", "U_abstention",
    "H_coverage", "H_abstention", "intervention_coverage", "intervention_coverage_LCB",
    "intervention_coverage_UCB",
}

COUNT_FIELDS = {
    "optimizer_update", "accepted_pseudo_count", "validation_image_count", "test_FP_total",
    "FP_total_negative", "n_images", "n_reviewed", "n_actionable", "n_support",
    "n_replacements", "q_total", "n_reject", "n_suppressions", "n_false_suppressions",
    "bootstrap_reps", "family_size", "n_pairs", "numerator", "denominator",
    "real_intervention_count", "sham_intervention_count", "latency_ms",
}

MANDATORY_B3_FILES = [
    "offline_verification_manifest.json",
    "teacher_candidate_pool.parquet",
    "b2_reference_selection.parquet",
    "vlm_request_manifest.jsonl",
    "vlm_response_cache.jsonl",
    "b3_variant_selection.parquet",
    "falsification_controls.csv",
    "matched_burden_controls.csv",
    "gt_matching.parquet",
    "offline_summary_by_class.csv",
    "offline_audit_report.json",
]

@dataclass
class Finding:
    level: str  # PASS / FAIL / BLOCKED / WARN
    check: str
    path: str = ""
    detail: str = ""

class Audit:
    def __init__(self) -> None:
        self.findings: list[Finding] = []

    def add(self, level: str, check: str, path: Path | str = "", detail: str = "") -> None:
        self.findings.append(Finding(level, check, str(path), detail))

    def pass_(self, check: str, path: Path | str = "", detail: str = "") -> None:
        self.add("PASS", check, path, detail)

    def fail(self, check: str, path: Path | str = "", detail: str = "") -> None:
        self.add("FAIL", check, path, detail)

    def blocked(self, check: str, path: Path | str = "", detail: str = "") -> None:
        self.add("BLOCKED", check, path, detail)

    def warn(self, check: str, path: Path | str = "", detail: str = "") -> None:
        self.add("WARN", check, path, detail)

    @property
    def status(self) -> str:
        if any(f.level == "FAIL" for f in self.findings):
            return "FAIL"
        if any(f.level == "BLOCKED" for f in self.findings):
            return "BLOCKED"
        return "PASS"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def is_nullish(v: Any) -> bool:
    if v is None:
        return True
    if isinstance(v, float) and math.isnan(v):
        return True
    if isinstance(v, str) and v.strip().upper() in {"", "NA", "N/A", "NULL", "NONE"}:
        return True
    return False


def to_float(v: Any) -> float | None:
    if is_nullish(v):
        return None
    try:
        x = float(v)
        return x if math.isfinite(x) else None
    except (TypeError, ValueError):
        return None


def to_int(v: Any) -> int | None:
    if is_nullish(v):
        return None
    try:
        x = int(v)
        if str(x) == str(v).strip() or float(v) == x:
            return x
    except (TypeError, ValueError):
        pass
    return None


def read_csv_header(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return next(csv.reader(f))


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8-sig") as f:
        return json.load(f)


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig") as f:
        for i, line in enumerate(f, 1):
            if not line.strip():
                continue
            obj = json.loads(line)
            if not isinstance(obj, dict):
                raise ValueError(f"line {i}: expected JSON object")
            yield obj


def check_exists(audit: Audit, path: Path, check: str) -> bool:
    if path.exists():
        audit.pass_(check, path)
        return True
    audit.fail(check, path, "required file missing")
    return False


def check_exact_csv_header(audit: Audit, path: Path, expected: Sequence[str], check: str) -> None:
    if not check_exists(audit, path, check + ":exists"):
        return
    try:
        actual = read_csv_header(path)
    except Exception as e:
        audit.fail(check, path, f"cannot read header: {e}")
        return
    if actual == list(expected):
        audit.pass_(check, path, f"{len(actual)} columns exact match")
    else:
        missing = [x for x in expected if x not in actual]
        extra = [x for x in actual if x not in expected]
        audit.fail(check, path, f"header mismatch; missing={missing}; extra={extra}; actual={actual}")


def check_json_keys(audit: Audit, path: Path, required: Sequence[str], check: str, exact: bool = False) -> dict[str, Any] | None:
    if not check_exists(audit, path, check + ":exists"):
        return None
    try:
        obj = read_json(path)
    except Exception as e:
        audit.fail(check, path, f"invalid JSON: {e}")
        return None
    if not isinstance(obj, dict):
        audit.fail(check, path, "top-level JSON must be object")
        return None
    actual = list(obj.keys())
    missing = [k for k in required if k not in obj]
    extra = [k for k in actual if k not in required]
    if missing or (exact and extra):
        audit.fail(check, path, f"missing={missing}; extra={extra if exact else 'allowed'}")
    else:
        audit.pass_(check, path, f"required keys present ({len(required)})")
    return obj


def check_jsonl_schema(audit: Audit, path: Path, required: Sequence[str], check: str, exact: bool = True) -> list[dict[str, Any]]:
    if not check_exists(audit, path, check + ":exists"):
        return []
    rows: list[dict[str, Any]] = []
    try:
        for line_no, obj in enumerate(iter_jsonl(path), 1):
            rows.append(obj)
            keys = list(obj.keys())
            missing = [k for k in required if k not in obj]
            extra = [k for k in keys if k not in required]
            if missing or (exact and extra):
                audit.fail(check, path, f"line {line_no}: missing={missing}; extra={extra if exact else 'allowed'}")
                return rows
        audit.pass_(check, path, f"{len(rows)} JSONL records conform")
    except Exception as e:
        audit.fail(check, path, f"invalid JSONL: {e}")
    return rows


def parquet_columns(path: Path) -> list[str] | None:
    try:
        import pyarrow.parquet as pq  # type: ignore
    except Exception:
        return None
    return pq.ParquetFile(path).schema_arrow.names


def check_parquet_schema(audit: Audit, path: Path, expected: Sequence[str], check: str, exact: bool = True) -> None:
    if not check_exists(audit, path, check + ":exists"):
        return
    try:
        cols = parquet_columns(path)
    except Exception as e:
        audit.fail(check, path, f"cannot inspect parquet: {e}")
        return
    if cols is None:
        audit.blocked(check, path, "pyarrow not installed; schema cannot be verified; silent PASS prohibited")
        return
    missing = [c for c in expected if c not in cols]
    extra = [c for c in cols if c not in expected]
    if missing or (exact and extra):
        audit.fail(check, path, f"missing={missing}; extra={extra if exact else 'allowed'}")
    else:
        audit.pass_(check, path, f"{len(cols)} columns conform")


def check_contract_hash(audit: Audit, path: Path) -> None:
    if not check_exists(audit, path, "contract_file"):
        return
    actual = sha256_file(path)
    if actual == CONTRACT_SHA256:
        audit.pass_("contract_sha256", path, actual)
    else:
        audit.fail("contract_sha256", path, f"expected={CONTRACT_SHA256}; actual={actual}")


def validate_numeric_csv(audit: Audit, path: Path, fields01: set[str], count_fields: set[str], check_prefix: str) -> None:
    if not path.exists():
        return
    try:
        rows = read_csv_rows(path)
    except Exception as e:
        audit.fail(check_prefix + ":read", path, str(e))
        return
    errors: list[str] = []
    for ri, row in enumerate(rows, 2):
        for k, v in row.items():
            if k in fields01 and not is_nullish(v):
                x = to_float(v)
                if x is None or not (0.0 <= x <= 1.0):
                    errors.append(f"row {ri} {k}={v!r} not in [0,1]")
            if k in count_fields and not is_nullish(v):
                x = to_int(v)
                if x is None or x < 0:
                    errors.append(f"row {ri} {k}={v!r} not nonnegative integer")
        if len(errors) >= 10:
            break
    if errors:
        audit.fail(check_prefix + ":numeric_semantics", path, "; ".join(errors[:10]))
    else:
        audit.pass_(check_prefix + ":numeric_semantics", path, f"checked {len(rows)} rows")


def validate_legacy(audit: Audit, repo: Path) -> None:
    final_master = repo / "artifacts" / "final_test" / "official_final_test_results.csv"
    paired = repo / "artifacts" / "results" / "paired_effects.csv"
    check_exact_csv_header(audit, final_master, FINAL_TEST_MASTER_HEADER, "legacy_final_test_master_header")
    check_exact_csv_header(audit, paired, PAIRED_EFFECTS_HEADER, "legacy_paired_effects_header")
    validate_numeric_csv(audit, final_master, RATE_FIELDS_01, COUNT_FIELDS, "legacy_final_test_master")


def validate_common_run(audit: Audit, run_root: Path, method: str) -> None:
    vdir = run_root / "metrics" / "validation"
    check_exact_csv_header(audit, vdir / "validation_history.csv", VALIDATION_HISTORY_HEADER, "validation_history_header")
    check_exact_csv_header(audit, vdir / "classwise_validation_ap.csv", CLASSWISE_VALIDATION_AP_HEADER, "classwise_validation_ap_header")
    ts = check_json_keys(audit, vdir / "training_summary.json", TRAINING_SUMMARY_KEYS, "training_summary_keys", exact=True)
    vb = check_json_keys(audit, vdir / "validation_best_teacher.json", VALIDATION_BEST_TEACHER_KEYS, "validation_best_teacher_keys", exact=True)
    validate_numeric_csv(audit, vdir / "validation_history.csv", RATE_FIELDS_01, COUNT_FIELDS, "validation_history")
    validate_numeric_csv(audit, vdir / "classwise_validation_ap.csv", RATE_FIELDS_01, COUNT_FIELDS, "classwise_validation_ap")

    # Class coverage check: exactly 14 named detection classes, if data rows exist.
    cap = vdir / "classwise_validation_ap.csv"
    if cap.exists():
        try:
            rows = read_csv_rows(cap)
            names = [r.get("class_name", "") for r in rows]
            if len(rows) == 14 and names == CLASS_NAMES:
                audit.pass_("classwise_validation_14_classes", cap)
            else:
                audit.fail("classwise_validation_14_classes", cap, f"rows={len(rows)} names={names}")
        except Exception as e:
            audit.fail("classwise_validation_14_classes", cap, str(e))

    qpath = run_root / "pseudo" / "qpseudo_validation.json"
    if method in {"B1", "B2", "B3"}:
        q = check_json_keys(audit, qpath, QPSEUDO_REQUIRED_KEYS, "qpseudo_required_keys", exact=False)
        if q is not None:
            for k in ("PL_AP50", "PL_AP75", "PL_mAP_50_95"):
                x = to_float(q.get(k))
                if x is None or not (0 <= x <= 1):
                    audit.fail("qpseudo_metric_range", qpath, f"{k}={q.get(k)!r}")
            if q.get("hidden_u_gt_used") is not False:
                audit.fail("qpseudo_hidden_u_gt_firewall", qpath, f"hidden_u_gt_used={q.get('hidden_u_gt_used')!r}")
            else:
                audit.pass_("qpseudo_hidden_u_gt_firewall", qpath)
            if q.get("test_used") is not False:
                audit.fail("qpseudo_test_firewall", qpath, f"test_used={q.get('test_used')!r}")
            else:
                audit.pass_("qpseudo_test_firewall", qpath)
    else:
        if qpath.exists():
            audit.warn("b0_qpseudo_unexpected", qpath, "B0 should not require Q_pseudo")


def validate_b2(audit: Audit, run_root: Path) -> None:
    validate_common_run(audit, run_root, "B2")
    # Flexible search: method_evidence is preferred canonical location.
    p = run_root / "method_evidence" / "b2_reference_selection.parquet"
    check_parquet_schema(audit, p, B2_REFERENCE_FIELDS, "b2_reference_selection_schema", exact=True)


def validate_b3_offline(audit: Audit, run_root: Path) -> None:
    # Canonical subdirectories per contract.
    method_dir = run_root / "method_evidence"
    vlm_dir = run_root / "vlm"

    manifest = check_json_keys(audit, vlm_dir / "offline_verification_manifest.json", ["protocol_version"], "offline_verification_manifest_minimum", exact=False)
    check_parquet_schema(audit, method_dir / "teacher_candidate_pool.parquet", TEACHER_CANDIDATE_FIELDS, "teacher_candidate_pool_schema", exact=True)
    check_parquet_schema(audit, method_dir / "b2_reference_selection.parquet", B2_REFERENCE_FIELDS, "b2_reference_selection_schema", exact=True)
    req_rows = check_jsonl_schema(audit, vlm_dir / "vlm_request_manifest.jsonl", VLM_REQUEST_FIELDS, "vlm_request_schema", exact=True)
    res_rows = check_jsonl_schema(audit, vlm_dir / "vlm_response_cache.jsonl", VLM_RESPONSE_FIELDS, "vlm_response_schema", exact=True)
    check_parquet_schema(audit, vlm_dir / "b3_variant_selection.parquet", B3_VARIANT_FIELDS, "b3_variant_selection_schema", exact=True)
    check_exact_csv_header(audit, vlm_dir / "falsification_controls.csv", FALSIFICATION_HEADER, "falsification_controls_header")
    check_exact_csv_header(audit, vlm_dir / "matched_burden_controls.csv", MATCHED_BURDEN_HEADER, "matched_burden_header")
    check_parquet_schema(audit, vlm_dir / "gt_matching.parquet", GT_MATCHING_FIELDS, "gt_matching_schema", exact=True)
    check_exact_csv_header(audit, vlm_dir / "offline_summary_by_class.csv", OFFLINE_SUMMARY_HEADER, "offline_summary_header")
    audit_report = check_json_keys(audit, vlm_dir / "offline_audit_report.json", OFFLINE_AUDIT_REQUIRED_KEYS, "offline_audit_report_keys", exact=False)

    validate_numeric_csv(audit, vlm_dir / "offline_summary_by_class.csv", RATE_FIELDS_01, COUNT_FIELDS, "offline_summary")
    validate_numeric_csv(audit, vlm_dir / "falsification_controls.csv", RATE_FIELDS_01, COUNT_FIELDS, "falsification_controls")
    validate_numeric_csv(audit, vlm_dir / "matched_burden_controls.csv", RATE_FIELDS_01, COUNT_FIELDS, "matched_burden_controls")

    # Confidence-blind payload must be false on every primary/control request.
    forbidden_flags = [
        "contains_teacher_score", "contains_teacher_rank", "contains_b2_selected",
        "contains_gt", "contains_annotation", "contains_clinical_metadata",
    ]
    bad = []
    for i, r in enumerate(req_rows, 1):
        if r.get("payload_audit_pass") is not True:
            bad.append(f"row{i}:payload_audit_pass={r.get('payload_audit_pass')!r}")
        for k in forbidden_flags:
            if r.get(k) is not False:
                bad.append(f"row{i}:{k}={r.get(k)!r}")
    if bad:
        audit.fail("vlm_confidence_blind_payload", vlm_dir / "vlm_request_manifest.jsonl", "; ".join(bad[:10]))
    elif req_rows:
        audit.pass_("vlm_confidence_blind_payload", vlm_dir / "vlm_request_manifest.jsonl", f"checked {len(req_rows)} requests")

    # Enum checks for VLM responses.
    enum_bad = []
    for i, r in enumerate(res_rows, 1):
        if r.get("decision") not in ENUMS["decision"]:
            enum_bad.append(f"row{i}:decision={r.get('decision')!r}")
        if r.get("region_relevant") not in ENUMS["region_relevant"]:
            enum_bad.append(f"row{i}:region_relevant={r.get('region_relevant')!r}")
    if enum_bad:
        audit.fail("vlm_response_enums", vlm_dir / "vlm_response_cache.jsonl", "; ".join(enum_bad[:10]))
    elif res_rows:
        audit.pass_("vlm_response_enums", vlm_dir / "vlm_response_cache.jsonl")

    if audit_report is not None:
        if audit_report.get("historical_test_used_for_design") is not False:
            audit.fail("historical_test_design_firewall", vlm_dir / "offline_audit_report.json", f"value={audit_report.get('historical_test_used_for_design')!r}")
        else:
            audit.pass_("historical_test_design_firewall", vlm_dir / "offline_audit_report.json")
        if audit_report.get("hidden_unlabeled_train_gt_used") is not False:
            audit.fail("hidden_unlabeled_train_gt_firewall", vlm_dir / "offline_audit_report.json", f"value={audit_report.get('hidden_unlabeled_train_gt_used')!r}")
        else:
            audit.pass_("hidden_unlabeled_train_gt_firewall", vlm_dir / "offline_audit_report.json")
        if audit_report.get("student_training_authorized") is not False:
            audit.fail("student_training_authorization_guard", vlm_dir / "offline_audit_report.json", "must remain false until separate authorization")
        else:
            audit.pass_("student_training_authorization_guard", vlm_dir / "offline_audit_report.json")
        gs = audit_report.get("global_status")
        if gs not in ENUMS["global_status"]:
            audit.fail("offline_global_status_enum", vlm_dir / "offline_audit_report.json", f"global_status={gs!r}")
        else:
            audit.pass_("offline_global_status_enum", vlm_dir / "offline_audit_report.json", str(gs))


def validate_b3_trained(audit: Audit, run_root: Path) -> None:
    validate_common_run(audit, run_root, "B3")
    validate_b3_offline(audit, run_root)


def validate_method_pair_effects(audit: Audit, path: Path) -> None:
    check_exact_csv_header(audit, path, METHOD_PAIR_EFFECTS_HEADER, "method_pair_effects_header")
    if not path.exists():
        return
    rows = read_csv_rows(path)
    errors: list[str] = []
    for i, r in enumerate(rows, 2):
        if r.get("schema_version") != SCHEMA_VERSION:
            errors.append(f"row{i}: schema_version={r.get('schema_version')!r}")
        rv, cv, dv = map(to_float, (r.get("reference_value"), r.get("comparison_value"), r.get("delta_comparison_minus_reference")))
        if None not in (rv, cv, dv) and abs((cv - rv) - dv) > 1e-12:
            errors.append(f"row{i}: delta sign/value mismatch")
        if r.get("reference_method") == r.get("comparison_method"):
            errors.append(f"row{i}: reference_method equals comparison_method")
    if errors:
        audit.fail("method_pair_effects_semantics", path, "; ".join(errors[:10]))
    else:
        audit.pass_("method_pair_effects_semantics", path, f"checked {len(rows)} rows")


def self_test() -> tuple[Audit, Path]:
    a = Audit()
    td = Path(tempfile.mkdtemp(prefix="g5_1_metric_contract_selftest_"))
    # Contract hash is not tested here because the canonical file may not be beside the script.
    # Exact legacy CSV fixture.
    f = td / "official_final_test_results.csv"
    f.write_text(",".join(FINAL_TEST_MASTER_HEADER) + "\n", encoding="utf-8")
    check_exact_csv_header(a, f, FINAL_TEST_MASTER_HEADER, "selftest_exact_csv")

    # JSON exact keys fixture.
    j = td / "training_summary.json"
    j.write_text(json.dumps({k: 0 for k in TRAINING_SUMMARY_KEYS}), encoding="utf-8")
    check_json_keys(a, j, TRAINING_SUMMARY_KEYS, "selftest_json_keys", exact=True)

    # JSONL VLM request fixture with all forbidden payload flags false.
    q = td / "vlm_request_manifest.jsonl"
    row = {k: None for k in VLM_REQUEST_FIELDS}
    row.update({
        "request_id": "r1", "candidate_id": "c1", "analysis_role": "PRIMARY",
        "control_id": "NONE", "review_branch": "U", "review_order": 1,
        "model_id": "fixture", "model_revision": "fixture", "model_hash": "a"*64,
        "processor_hash": "b"*64, "prompt_hash": "c"*64, "renderer_hash": "d"*64,
        "decoding_config_hash": "e"*64, "rendered_image_hash": "f"*64,
        "vlm_cache_key": "k", "class_term": "Atelectasis", "payload_audit_pass": True,
        "contains_teacher_score": False, "contains_teacher_rank": False,
        "contains_b2_selected": False, "contains_gt": False,
        "contains_annotation": False, "contains_clinical_metadata": False,
    })
    q.write_text(json.dumps(row) + "\n", encoding="utf-8")
    rows = check_jsonl_schema(a, q, VLM_REQUEST_FIELDS, "selftest_jsonl", exact=True)
    if rows and all(rows[0].get(k) is False for k in ["contains_teacher_score", "contains_teacher_rank", "contains_b2_selected", "contains_gt", "contains_annotation", "contains_clinical_metadata"]):
        a.pass_("selftest_confidence_blind_fixture", q)
    else:
        a.fail("selftest_confidence_blind_fixture", q)

    # Negative test: intentionally bad header must be detected as FAIL in an isolated audit.
    bad_a = Audit()
    bad = td / "bad.csv"
    bad.write_text("a,b\n", encoding="utf-8")
    check_exact_csv_header(bad_a, bad, ["a", "b", "c"], "negative_fixture")
    if bad_a.status == "FAIL":
        a.pass_("selftest_negative_fixture_detected", bad)
    else:
        a.fail("selftest_negative_fixture_detected", bad, f"unexpected={bad_a.status}")
    return a, td


def write_report(audit: Audit, path: Path | None, mode: str, repo_root: Path | None, run_root: Path | None) -> dict[str, Any]:
    report = {
        "schema_version": SCHEMA_VERSION,
        "validator": "validate_metric_output_contract.py",
        "mode": mode,
        "repo_root": str(repo_root) if repo_root else None,
        "run_root": str(run_root) if run_root else None,
        "status": audit.status,
        "counts": {lvl: sum(f.level == lvl for f in audit.findings) for lvl in ["PASS", "FAIL", "BLOCKED", "WARN"]},
        "findings": [asdict(f) for f in audit.findings],
    }
    if path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return report


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Validate locked SSOD+VLM Metric/Output Contract schemas.")
    p.add_argument("--mode", choices=["self-test", "legacy", "b2", "b3-offline", "b3-trained", "method-pairs"], required=True)
    p.add_argument("--repo-root", type=Path)
    p.add_argument("--run-root", type=Path)
    p.add_argument("--contract", type=Path)
    p.add_argument("--method-pairs", type=Path)
    p.add_argument("--report", type=Path)
    args = p.parse_args(argv)

    if args.mode == "self-test":
        audit, td = self_test()
        report = write_report(audit, args.report, args.mode, None, None)
        print(json.dumps(report, indent=2, ensure_ascii=False))
        print(f"SELF_TEST_TMP={td}", file=sys.stderr)
        return 0 if audit.status == "PASS" else 2

    audit = Audit()
    if args.contract:
        check_contract_hash(audit, args.contract)

    if args.mode == "legacy":
        if args.repo_root is None:
            p.error("--repo-root required for legacy mode")
        validate_legacy(audit, args.repo_root)
    elif args.mode == "b2":
        if args.run_root is None:
            p.error("--run-root required for b2 mode")
        validate_b2(audit, args.run_root)
    elif args.mode == "b3-offline":
        if args.run_root is None:
            p.error("--run-root required for b3-offline mode")
        validate_b3_offline(audit, args.run_root)
    elif args.mode == "b3-trained":
        if args.run_root is None:
            p.error("--run-root required for b3-trained mode")
        validate_b3_trained(audit, args.run_root)
    elif args.mode == "method-pairs":
        if args.method_pairs is None:
            p.error("--method-pairs required for method-pairs mode")
        validate_method_pair_effects(audit, args.method_pairs)

    report = write_report(audit, args.report, args.mode, args.repo_root, args.run_root)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if audit.status == "PASS" else (3 if audit.status == "BLOCKED" else 2)


if __name__ == "__main__":
    raise SystemExit(main())

