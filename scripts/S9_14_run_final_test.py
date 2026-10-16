"""S9.14 authorized final-test launcher.

Default mode is pre-access dry-run. Test data is opened only with --execute.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INVENTORY = ROOT / "artifacts/final_test/eligible_model_inventory.csv"
AUTHORIZATION = ROOT / "artifacts/final_test/FINAL_TEST_AUTHORIZATION.json"
PRE_GATE = ROOT / "artifacts/final_test/PRE_FINAL_TEST_GATE_REPORT.json"
ACCESS_AUDIT = ROOT / "artifacts/final_test/test_access_audit.json"

EXPECTED_EVALUATOR_SHA = "43079be37ab97cbbd58d0fdeaf171b9ebfe8272399d2b32800bbfbd8505d72a9"
EXPECTED_TEST_SPLIT_SHA = "1f7903e069e872bf2e5fe13bb4d0fa257dc4a1c2c8290a621d3f7286ada66b37"
EXPECTED_AUTH_SHA = "67cb73b0c041437cb591883c8e7d762189da48d56a089919972112e087e4b448"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def record_test_access(evidence: dict, *, timestamp_utc: str, status: str, test_accessed: bool) -> None:
    """Persist campaign-level authorized test-access evidence before/after access."""
    ACCESS_AUDIT.parent.mkdir(parents=True, exist_ok=True)
    if ACCESS_AUDIT.exists():
        payload = json.loads(ACCESS_AUDIT.read_text(encoding="utf-8"))
    else:
        payload = {
            "schema_version": "S9.14",
            "authorization_sha256": evidence["authorization_sha256"],
            "eligible_model_inventory_sha256": sha256(INVENTORY),
            "evaluator_config_sha256": evidence["evaluator_config_sha256"],
            "test_split_sha256": evidence["test_split_sha256"],
            "runs": {},
        }
    for key, expected in (
        ("authorization_sha256", evidence["authorization_sha256"]),
        ("eligible_model_inventory_sha256", sha256(INVENTORY)),
        ("evaluator_config_sha256", evidence["evaluator_config_sha256"]),
        ("test_split_sha256", evidence["test_split_sha256"]),
    ):
        if payload.get(key) != expected:
            raise RuntimeError(f"test_access_audit identity mismatch for {key}")
    run = payload["runs"].setdefault(
        evidence["run_id"],
        {
            "run_id": evidence["run_id"],
            "authorized": True,
            "checkpoint_role": evidence["checkpoint_role"],
            "checkpoint_sha256": evidence["checkpoint_sha256"],
            "access_started_at_utc": timestamp_utc,
        },
    )
    if run.get("checkpoint_sha256") != evidence["checkpoint_sha256"]:
        raise RuntimeError("test_access_audit checkpoint SHA256 mismatch")
    run["status"] = status
    run["test_accessed"] = bool(test_accessed)
    run["last_updated_at_utc"] = timestamp_utc
    tmp = ACCESS_AUDIT.with_name(ACCESS_AUDIT.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(ACCESS_AUDIT)


def scalar(value: str):
    value = "" if value is None else value
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


def load_inventory() -> list[dict]:
    with INVENTORY.open(encoding="utf-8-sig", newline="") as f:
        return [{k: scalar(v) for k, v in row.items()} for row in csv.DictReader(f)]


def select_row(run_id: str) -> dict:
    rows = load_inventory()
    if len(rows) != 180:
        raise RuntimeError(f"Eligible inventory row count is {len(rows)}, expected 180")
    matches = [r for r in rows if r["s9_12.run_id"] == run_id]
    if len(matches) != 1:
        raise RuntimeError(f"Expected exactly one eligible row for {run_id}; got {len(matches)}")
    return matches[0]


def preaccess_guards(row: dict) -> dict:
    if sha256(AUTHORIZATION) != EXPECTED_AUTH_SHA:
        raise RuntimeError("FINAL_TEST_AUTHORIZATION.json SHA256 mismatch")
    auth = json.loads(AUTHORIZATION.read_text(encoding="utf-8"))
    if auth.get("researcher_authorization") is not True:
        raise RuntimeError("Researcher authorization is not TRUE")

    gate = json.loads(PRE_GATE.read_text(encoding="utf-8"))
    gate_status = gate.get("overall_status", gate.get("status"))
    if gate_status != "PASS":
        raise RuntimeError(f"PRE_FINAL_TEST_GATE_REPORT is not PASS: {gate_status!r}")

    method = row["official_inventory.method"]
    role = row["selected_checkpoint.role"]
    expected_role = "BEST_CHECKPOINT" if method == "SUP" else "BEST_EMA_TEACHER_CHECKPOINT"
    if method not in {"SUP", "SSL"} or role != expected_role:
        raise RuntimeError(f"Method/checkpoint-role mismatch: method={method!r} role={role!r}")

    if row["official_inventory.condition_role"] not in {"MAIN_LOW_LABEL", "SUP_REFERENCE"}:
        raise RuntimeError("Non-Main/non-reference row is prohibited from final test")

    evaluator_sha = row["pre_run_assertions.expected_identity.evaluator_config_sha256"]
    test_split_sha = row["pre_run_assertions.expected_identity.test_split_sha256"]
    if evaluator_sha != EXPECTED_EVALUATOR_SHA:
        raise RuntimeError("Evaluator semantic SHA mismatch")
    if test_split_sha != EXPECTED_TEST_SPLIT_SHA:
        raise RuntimeError("Test split membership SHA mismatch")

    checkpoint = Path(row["selected_checkpoint.runtime_path"])
    if not checkpoint.is_file():
        raise RuntimeError(f"Selected checkpoint missing: {checkpoint}")
    observed_checkpoint_sha = sha256(checkpoint)
    if observed_checkpoint_sha != row["selected_checkpoint.sha256"]:
        raise RuntimeError("Selected checkpoint SHA256 mismatch")

    return {
        "run_id": row["s9_12.run_id"],
        "method": method,
        "condition_role": row["official_inventory.condition_role"],
        "architecture": row["official_inventory.architecture"],
        "attempt_used": row["s9_12.attempt_used"],
        "executable_config_path": row["pre_run_assertions.expected_identity.executable_config_path"],
        "checkpoint_role": role,
        "checkpoint_path": str(checkpoint),
        "checkpoint_sha256": observed_checkpoint_sha,
        "evaluator_config_sha256": evaluator_sha,
        "test_split_sha256": test_split_sha,
        "authorization_sha256": EXPECTED_AUTH_SHA,
    }


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--run-id", required=True)
    p.add_argument("--execute", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    row = select_row(args.run_id)
    evidence = preaccess_guards(row)
    print(json.dumps({"mode": "EXECUTE" if args.execute else "DRY_RUN", **evidence}, indent=2))
    if not args.execute:
        print("S9_14_PREACCESS_DRY_RUN=PASS")
        print("FINAL_TEST_ACCESSED=False")
        return 0

    # Runtime-only imports keep Windows dry-run independent of MMEngine/MMDetection.
    from datetime import datetime, timezone

    from mmengine.runner import Runner

    from src.evaluation.s9_14_final_test_artifacts import (
        materialize_final_test_artifacts,
    )
    from src.evaluation.s9_14_final_test_execute import (
        TEST_ANN,
        build_final_test_config,
        load_selected_weights,
        verify_test_file_before_build,
    )
    from src.hooks.s9_14_final_test_retention_hook import (
        FinalTestPredictionRetentionHook,
    )
    from src.metrics.operating_point_metric import OperatingPointMetric
    from src.metrics.protocol_coco_metric import ProtocolCocoMetric
    from src.utils.s9_14_final_test_backup import FinalTestActiveBackupManager

    access_started = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    record_test_access(
        evidence,
        timestamp_utc=access_started,
        status="AUTHORIZED_ACCESS_STARTED",
        test_accessed=True,
    )

    # This ledger write occurs before the first canonical test-file open so that
    # any interruption during verification cannot under-report test access.
    verify_test_file_before_build()
    record_test_access(
        evidence,
        timestamp_utc=datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        status="TEST_FILE_VERIFIED",
        test_accessed=True,
    )

    checkpoint = Path(evidence["checkpoint_path"])
    attempt_dir = checkpoint.parent.parent
    if attempt_dir.name != evidence["attempt_used"]:
        raise RuntimeError(
            f"Attempt-directory mismatch: {attempt_dir.name!r} != "
            f"{evidence['attempt_used']!r}"
        )
    if attempt_dir.parent.name != evidence["run_id"]:
        raise RuntimeError(
            f"Run-directory mismatch: {attempt_dir.parent.name!r} != "
            f"{evidence['run_id']!r}"
        )

    output_dir = attempt_dir / "evaluation" / "final_test"
    if (output_dir / "final_evaluation_report.json").exists():
        raise RuntimeError(
            "Final-test CLOSED_VERIFIED artifact already exists; refusing duplicate test access"
        )

    cfg_path = ROOT / evidence["executable_config_path"]
    if not cfg_path.is_file():
        raise RuntimeError(f"Executable config missing: {cfg_path}")

    cfg = build_final_test_config(
        cfg_path,
        work_dir=output_dir / "runtime",
        evaluator_config_sha256=evidence["evaluator_config_sha256"],
    )

    backup = FinalTestActiveBackupManager(ROOT, attempt_dir)
    backup.start_final_test()
    try:
        runner = Runner.from_cfg(cfg)
        evaluation_model_role = load_selected_weights(
            runner.model,
            checkpoint,
            method=evidence["method"],
            checkpoint_role=evidence["checkpoint_role"],
        )

        evaluator_metrics = runner.test()

        protocol_metrics = [
            metric
            for metric in runner.test_loop.evaluator.metrics
            if isinstance(metric, ProtocolCocoMetric)
        ]
        operating_metrics = [
            metric
            for metric in runner.test_loop.evaluator.metrics
            if isinstance(metric, OperatingPointMetric)
        ]
        retention_hooks = [
            hook
            for hook in runner.hooks
            if isinstance(hook, FinalTestPredictionRetentionHook)
        ]
        if len(protocol_metrics) != 1:
            raise RuntimeError(
                f"Expected exactly one ProtocolCocoMetric; got {len(protocol_metrics)}"
            )
        if len(operating_metrics) != 1:
            raise RuntimeError(
                f"Expected exactly one OperatingPointMetric; got {len(operating_metrics)}"
            )
        if len(retention_hooks) != 1:
            raise RuntimeError(
                f"Expected exactly one FinalTestPredictionRetentionHook; got {len(retention_hooks)}"
            )

        native_records = retention_hooks[0].export_native_records()
        timestamp = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

        report = materialize_final_test_artifacts(
            output_dir=output_dir,
            run_id=evidence["run_id"],
            evaluation_model_role=evaluation_model_role,
            checkpoint_role=evidence["checkpoint_role"],
            checkpoint_sha256=evidence["checkpoint_sha256"],
            evaluator_config_sha256=evidence["evaluator_config_sha256"],
            test_split_sha256=evidence["test_split_sha256"],
            test_authorization_sha256=evidence["authorization_sha256"],
            evaluation_timestamp_utc=timestamp,
            native_records=native_records,
            evaluator_metrics=evaluator_metrics,
            protocol_metric=protocol_metrics[0],
            operating_metric=operating_metrics[0],
            eligible_model_inventory_match=True,
            checkpoint_sha256_match=(sha256(checkpoint) == evidence["checkpoint_sha256"]),
            test_split_sha256_match=True,
            authorized=True,
        )
        backup.finalize()
        record_test_access(
            evidence,
            timestamp_utc=datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            status="CLOSED_VERIFIED",
            test_accessed=True,
        )
        print(json.dumps(report, indent=2, sort_keys=True))
        print("S9_14_FINAL_TEST_RUN=CLOSED_VERIFIED")
        print("FINAL_TEST_ACCESSED=True")
        return 0
    except Exception:
        record_test_access(
            evidence,
            timestamp_utc=datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            status="TECHNICAL_FAILURE_AFTER_AUTHORIZED_ACCESS",
            test_accessed=True,
        )
        backup.record_failure_state()
        raise
    finally:
        backup.stop()


if __name__ == "__main__":
    raise SystemExit(main())
