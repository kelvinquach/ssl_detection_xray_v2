#!/usr/bin/env python3
import argparse
import csv
import json
import re
from pathlib import Path

EXPECTED_ROWS = 180
EXPECTED_SSL = 80
EXPECTED_SUP = 100
EXPECTED_SUP_REFERENCE = 20

EXPECTED_CLASSES = [
    ("1",  "Aortic enlargement"),
    ("2",  "Atelectasis"),
    ("3",  "Calcification"),
    ("4",  "Cardiomegaly"),
    ("5",  "Consolidation"),
    ("6",  "ILD"),
    ("7",  "Infiltration"),
    ("8",  "Lung Opacity"),
    ("9",  "Nodule/Mass"),
    ("10", "Other lesion"),
    ("11", "Pleural effusion"),
    ("12", "Pleural thickening"),
    ("13", "Pneumothorax"),
    ("14", "Pulmonary fibrosis"),
]

BASE_FIELDS = [
    "run_id",
    "condition_role",
    "method",
    "architecture",
    "budget",
    "seed_index",
    "training_seed",
    "best_checkpoint_sha256",
    "best_validation_mAP_50_95",
    "test_mAP_50_95",
    "test_AP50",
    "test_AP75",
    "test_AR_50_95_max100",
    "test_Recall_tau_eval",
    "test_FP_total",
    "test_FP_per_image",
]

NEGATIVE_FIELDS = [
    "FP_total_negative",
    "FP_per_negative",
    "negative_FAR",
]

TAIL_FIELDS = [
    "Q_pseudo",
    "Q_pseudo_status",
    "final_evaluation_status",
]


def norm(x):
    if x is None:
        return ""
    return str(x).strip().strip('"')


def require(cond, msg):
    if not cond:
        raise RuntimeError(msg)


def load_json(path):
    require(path.is_file(), f"MISSING_FILE: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_csv(path):
    require(path.is_file(), f"MISSING_FILE: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def class_column(name):
    sanitized = re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_")
    return f"AP_{sanitized}"


CLASS_FIELDS = [class_column(name) for _, name in EXPECTED_CLASSES]

OUTPUT_FIELDS = (
    BASE_FIELDS
    + CLASS_FIELDS
    + NEGATIVE_FIELDS
    + TAIL_FIELDS
)


def require_json_value(obj, key, path):
    require(key in obj, f"MISSING_FIELD: {path}: {key}")
    require(obj[key] is not None, f"NULL_FIELD: {path}: {key}")
    return obj[key]


def resolve_qpseudo(run_root):
    matches = sorted(run_root.rglob("qpseudo_validation.json"))
    return matches


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eligible-model-inventory", required=True)
    ap.add_argument("--training-summary", required=True)
    ap.add_argument("--official-root", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    eligible_path = Path(args.eligible_model_inventory)
    training_summary_path = Path(args.training_summary)
    official_root = Path(args.official_root)
    output_path = Path(args.output)

    eligible = load_csv(eligible_path)
    training = load_csv(training_summary_path)

    require(
        len(eligible) == EXPECTED_ROWS,
        f"ELIGIBLE_ROW_COUNT: expected={EXPECTED_ROWS} observed={len(eligible)}",
    )
    require(
        len(training) == EXPECTED_ROWS,
        f"TRAINING_SUMMARY_ROW_COUNT: expected={EXPECTED_ROWS} observed={len(training)}",
    )

    eligible_by_run = {}
    for r in eligible:
        run_id = norm(r.get("s9_12.run_id"))
        require(run_id, "EMPTY_RUN_ID_IN_ELIGIBLE_INVENTORY")
        require(
            run_id not in eligible_by_run,
            f"DUPLICATE_ELIGIBLE_RUN_ID: {run_id}",
        )
        eligible_by_run[run_id] = r

    training_run_ids = [norm(r.get("run_id")) for r in training]
    require(
        len(set(training_run_ids)) == EXPECTED_ROWS,
        "TRAINING_SUMMARY_RUN_IDS_NOT_UNIQUE",
    )
    require(
        set(training_run_ids) == set(eligible_by_run),
        "TRAINING_SUMMARY_ELIGIBLE_RUN_SET_MISMATCH",
    )

    ssl_count = sum(norm(r.get("method")) == "SSL" for r in training)
    sup_count = sum(norm(r.get("method")) == "SUP" for r in training)
    sup_ref_count = sum(
        norm(r.get("method")) == "SUP"
        and norm(r.get("condition_role")) == "SUP_REFERENCE"
        for r in training
    )

    require(ssl_count == EXPECTED_SSL, f"SSL_COUNT={ssl_count}")
    require(sup_count == EXPECTED_SUP, f"SUP_COUNT={sup_count}")
    require(
        sup_ref_count == EXPECTED_SUP_REFERENCE,
        f"SUP_REFERENCE_COUNT={sup_ref_count}",
    )

    output_rows = []
    undefined_ap_count = 0
    ssl_qpseudo_count = 0

    for tr in training:
        run_id = norm(tr.get("run_id"))
        method = norm(tr.get("method"))
        condition_role = norm(tr.get("condition_role"))
        inv = eligible_by_run[run_id]

        require(method in {"SUP", "SSL"}, f"INVALID_METHOD: {run_id}: {method}")
        require(
            norm(inv.get("official_inventory.method")) == method,
            f"METHOD_MISMATCH: {run_id}",
        )
        require(
            norm(inv.get("official_inventory.condition_role")) == condition_role,
            f"CONDITION_ROLE_MISMATCH: {run_id}",
        )

        attempt = norm(inv.get("s9_12.attempt_used"))
        require(
            re.fullmatch(r"attempt_\d{3}", attempt) is not None,
            f"INVALID_CANONICAL_ATTEMPT: {run_id}: {attempt}",
        )

        run_root = official_root / run_id / attempt
        final_root = run_root / "evaluation" / "final_test"

        coco_path = final_root / "coco_metrics.json"
        operating_path = final_root / "operating_point_metrics.json"
        classwise_path = final_root / "classwise_ap.csv"
        negative_path = final_root / "negative_metrics.json"
        manifest_path = final_root / "evaluation_manifest.json"
        report_path = final_root / "final_evaluation_report.json"

        coco = load_json(coco_path)
        operating = load_json(operating_path)
        negative = load_json(negative_path)
        manifest = load_json(manifest_path)
        report = load_json(report_path)
        classwise = load_csv(classwise_path)

        # Final-test inclusion gate: fail closed.
        require(report.get("authorized") is True, f"NOT_AUTHORIZED: {run_id}")
        require(
            report.get("checkpoint_sha256_match") is True,
            f"CHECKPOINT_SHA_MISMATCH: {run_id}",
        )
        require(
            report.get("eligible_for_official_analysis") is True,
            f"NOT_ELIGIBLE_FOR_OFFICIAL_ANALYSIS: {run_id}",
        )
        require(
            report.get("eligible_model_inventory_match") is True,
            f"ELIGIBLE_INVENTORY_MISMATCH: {run_id}",
        )
        require(
            report.get("evaluation_manifest_valid") is True,
            f"EVALUATION_MANIFEST_INVALID: {run_id}",
        )
        require(
            report.get("test_split_sha256_match") is True,
            f"TEST_SPLIT_SHA_MISMATCH: {run_id}",
        )
        require(
            norm(report.get("final_evaluation_status")) == "CLOSED_VERIFIED",
            f"FINAL_EVALUATION_NOT_CLOSED_VERIFIED: {run_id}",
        )
        require(
            norm(report.get("run_id")) == run_id,
            f"FINAL_REPORT_RUN_ID_MISMATCH: {run_id}",
        )

        # Evaluation-manifest identity / checkpoint gate.
        require(
            norm(manifest.get("run_id")) == run_id,
            f"EVALUATION_MANIFEST_RUN_ID_MISMATCH: {run_id}",
        )

        best_sha = norm(tr.get("best_checkpoint_sha256"))
        require(best_sha, f"EMPTY_BEST_CHECKPOINT_SHA256: {run_id}")
        require(
            norm(manifest.get("checkpoint_sha256")) == best_sha,
            f"BEST_CHECKPOINT_SHA256_MISMATCH: {run_id}",
        )

        expected_checkpoint_role = (
            "BEST_EMA_TEACHER_CHECKPOINT"
            if method == "SSL"
            else "BEST_CHECKPOINT"
        )
        require(
            norm(manifest.get("checkpoint_role")) == expected_checkpoint_role,
            f"CHECKPOINT_ROLE_MISMATCH: {run_id}",
        )

        require(
            norm(manifest.get("evaluator_config_sha256")),
            f"EMPTY_EVALUATOR_CONFIG_SHA256: {run_id}",
        )
        require(
            norm(manifest.get("test_authorization_sha256")),
            f"EMPTY_TEST_AUTHORIZATION_SHA256: {run_id}",
        )
        require(
            norm(manifest.get("test_split_sha256")),
            f"EMPTY_TEST_SPLIT_SHA256: {run_id}",
        )

        # Canonical 14-class mapping and undefined-AP handling.
        require(
            len(classwise) == len(EXPECTED_CLASSES),
            f"CLASS_COUNT_MISMATCH: {run_id}: {len(classwise)}",
        )

        class_by_id = {}
        for cr in classwise:
            cid = norm(cr.get("class_id"))
            require(cid, f"EMPTY_CLASS_ID: {run_id}")
            require(
                cid not in class_by_id,
                f"DUPLICATE_CLASS_ID: {run_id}: {cid}",
            )
            class_by_id[cid] = cr

        class_values = {}
        for cid, expected_name in EXPECTED_CLASSES:
            require(
                cid in class_by_id,
                f"MISSING_CLASS_ID: {run_id}: {cid}",
            )
            cr = class_by_id[cid]
            actual_name = norm(cr.get("class_name"))
            require(
                actual_name == expected_name,
                f"CLASS_NAME_MISMATCH: {run_id}: "
                f"id={cid} expected={expected_name!r} observed={actual_name!r}",
            )

            status = norm(cr.get("status"))
            ap_value = norm(cr.get("AP_50_95"))

            if status == "DEFINED":
                require(
                    ap_value != "",
                    f"DEFINED_CLASS_AP_EMPTY: {run_id}: {expected_name}",
                )
                out_ap = ap_value
            else:
                # Scientific contract: undefined AP is retained as NA,
                # never numerically imputed.
                out_ap = "NA"
                undefined_ap_count += 1

            class_values[class_column(expected_name)] = out_ap

        # Q_pseudo is validation-only and SSL-only.
        qmatches = resolve_qpseudo(run_root)

        if method == "SSL":
            require(
                len(qmatches) == 1,
                f"SSL_QPSEUDO_FILE_COUNT: {run_id}: {len(qmatches)}",
            )
            qp = load_json(qmatches[0])
            require(
                qp.get("PL_mAP_50_95") is not None,
                f"SSL_QPSEUDO_MISSING_PL_MAP: {run_id}",
            )
            require(
                qp.get("test_used") is False,
                f"SSL_QPSEUDO_TEST_USED: {run_id}",
            )
            require(
                qp.get("hidden_u_gt_used") is False,
                f"SSL_QPSEUDO_HIDDEN_U_GT_USED: {run_id}",
            )
            q_pseudo = qp["PL_mAP_50_95"]
            q_status = "AVAILABLE"
            ssl_qpseudo_count += 1
        else:
            require(
                len(qmatches) == 0,
                f"SUP_QPSEUDO_UNEXPECTED_FILE_COUNT: {run_id}: {len(qmatches)}",
            )
            q_pseudo = "NA"
            q_status = "NOT_APPLICABLE"

        row = {
            "run_id": run_id,
            "condition_role": condition_role,
            "method": method,
            "architecture": norm(tr.get("architecture")),
            "budget": norm(tr.get("budget")),
            "seed_index": norm(tr.get("seed_index")),
            "training_seed": norm(tr.get("training_seed")),
            "best_checkpoint_sha256": best_sha,
            "best_validation_mAP_50_95": norm(
                tr.get("best_validation_mAP_50_95")
            ),
            "test_mAP_50_95": require_json_value(
                coco, "bbox_mAP_50_95", coco_path
            ),
            "test_AP50": require_json_value(
                coco, "AP50", coco_path
            ),
            "test_AP75": require_json_value(
                coco, "AP75", coco_path
            ),
            "test_AR_50_95_max100": require_json_value(
                coco, "AR_50_95_max100", coco_path
            ),
            "test_Recall_tau_eval": require_json_value(
                operating, "Recall_tau_eval", operating_path
            ),
            "test_FP_total": require_json_value(
                operating, "FP_total", operating_path
            ),
            "test_FP_per_image": require_json_value(
                operating, "FP_per_image", operating_path
            ),
            "FP_total_negative": require_json_value(
                negative, "FP_total_on_negative", negative_path
            ),
            "FP_per_negative": require_json_value(
                negative, "FP_per_negative", negative_path
            ),
            "negative_FAR": require_json_value(
                negative, "negative_image_FAR", negative_path
            ),
            "Q_pseudo": q_pseudo,
            "Q_pseudo_status": q_status,
            "final_evaluation_status": norm(
                report.get("final_evaluation_status")
            ),
        }

        row.update(class_values)
        output_rows.append(row)

    require(
        len(output_rows) == EXPECTED_ROWS,
        f"OUTPUT_ROW_COUNT: {len(output_rows)}",
    )
    require(
        ssl_qpseudo_count == EXPECTED_SSL,
        f"SSL_QPSEUDO_COUNT: {ssl_qpseudo_count}",
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=OUTPUT_FIELDS)
        w.writeheader()
        w.writerows(output_rows)

    print("=== S10.1.03 OFFICIAL FINAL-TEST RESULTS ===")
    print(f"INPUT_ROWS={len(training)}")
    print(f"OUTPUT_ROWS={len(output_rows)}")
    print(f"UNIQUE_RUN_IDS={len(set(r['run_id'] for r in output_rows))}")
    print(f"SSL_ROWS={ssl_count}")
    print(f"SUP_ROWS={sup_count}")
    print(f"SUP_REFERENCE_ROWS={sup_ref_count}")
    print(f"SSL_QPSEUDO_AVAILABLE={ssl_qpseudo_count}")
    print(f"UNDEFINED_CLASS_AP_ROWS={undefined_ap_count}")
    print(f"COLUMNS={len(OUTPUT_FIELDS)}")
    print("S10_1_03_STATUS=PASS")


if __name__ == "__main__":
    main()