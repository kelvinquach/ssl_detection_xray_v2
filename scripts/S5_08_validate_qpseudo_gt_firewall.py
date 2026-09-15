"""S5.08 validation for Q_pseudo hidden-U/test-GT fail-closed firewall."""

import hashlib
import json
import tempfile
from pathlib import Path

import numpy as np
from mmengine.config import Config

import src.evaluation.qpseudo as qp


ROOT = Path("/workspace/ssod/project")
EVIDENCE = ROOT / "artifacts/preflight/evaluation/qpseudo_path_golden_test.json"
QPSEUDO_IMPL = ROOT / "src/evaluation/qpseudo.py"
QPSEUDO_INIT = ROOT / "src/evaluation/__init__.py"
ACCEPTANCE_CONFIG = ROOT / "configs/evaluation/s5_06_qpseudo_acceptance.py"
S5_06_VALIDATOR = ROOT / "scripts/S5_06_validate_qpseudo_path.py"
S5_07_VALIDATOR = ROOT / "scripts/S5_07_validate_best_teacher_firewall.py"
S5_08_VALIDATOR = ROOT / "scripts/S5_08_validate_qpseudo_gt_firewall.py"
EVALUATOR_CONFIG = ROOT / "configs/evaluation/s5_02_coco_secondary_evaluator.py"
FIXED_VAL = Path("/workspace/ssod/data/coco/instances_val.json")

HISTORICAL_EVIDENCE_SHA256 = (
    "72e81b8ef7639c31b9be2cac79eed451caf9130e7e4b4ae1895cd125d0769b1d"
)
FIXED_VAL_SHA256 = (
    "33064f47ba690be13e9d418d8ec5d4a6a2bec8482c24ea3eadf83fb33d01762a"
)
EXPECTED_SOURCE_SHA256 = {
    "src/evaluation/qpseudo.py":
        "300fc37b08e4f1e9e3e97c29bc714aef2b219386c2932cc34669923707e57bcc",
    "src/evaluation/__init__.py":
        "58083297b7df31668de5fcc60323eb1b6fb14a7c5927c233e63be16622a6ac5e",
    "configs/evaluation/s5_06_qpseudo_acceptance.py":
        "bb748e521af2eaeefe197a06d75b3fff1b3200f032725d42af61cfd1e2cac1c1",
    "scripts/S5_06_validate_qpseudo_path.py":
        "63bb6393bbdb5e897a2dfd17ff947373aab08de2849fec756de3bdf7cccb3b51",
    "scripts/S5_07_validate_best_teacher_firewall.py":
        "121e489e2963f1bd15cec1b90338b56b99f8db8db58d78fd0195aaf31557f2fd",
}
CANONICAL_SHA256 = {
    "sn-article.tex":
        "8d97840cc546a58aef7354af0c856119b1aea16f8be1cb7bbc1ab1a324d40fcc",
    "IMPLEMENTATION_HANDOFF.md":
        "540c94746571612ff819fb62a1e9fac6123c4035bc2642518eaa7740770820d0",
    "ARTIFACT_AND_EVIDENCE_CONTRACT.md":
        "885d4eeb66857e99744f6e8a27083e5bb695bbcc0d172cf4ebff722c53f655a4",
}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build_fixture(path):
    categories = [
        {"id": idx, "name": f"class_{idx}"}
        for idx in range(1, 15)
    ]
    images = [
        {"id": idx, "width": 100, "height": 100, "file_name": f"{idx}.jpg"}
        for idx in range(1, 15)
    ]
    annotations = []
    predictions = []
    for idx in range(1, 15):
        annotations.append({
            "id": idx,
            "image_id": idx,
            "category_id": idx,
            "bbox": [10.0, 10.0, 20.0, 20.0],
            "area": 400.0,
            "iscrowd": 0,
        })
        predictions.append({
            "img_id": idx,
            "ori_shape": (100, 100),
            "bboxes": np.asarray([[10.0, 10.0, 30.0, 30.0]], dtype=np.float32),
            "scores": np.asarray([0.99], dtype=np.float32),
            "labels": np.asarray([idx - 1], dtype=np.int64),
            "detector_native_count": 1,
            "accepted_count": 1,
        })
    payload = {
        "images": images,
        "annotations": annotations,
        "categories": categories,
    }
    Path(path).write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return predictions


print("===== S5.08 Q_PSEUDO GT FIREWALL VALIDATOR =====")

checks = []


def check(name, passed, detail=None):
    passed = bool(passed)
    checks.append({
        "name": name,
        "pass": passed,
        "detail": detail,
    })
    print(f"{name}={passed}")


historical_sha = sha256(EVIDENCE)
historical = json.loads(EVIDENCE.read_text(encoding="utf-8"))

check(
    "historical::evidence_sha",
    historical_sha == HISTORICAL_EVIDENCE_SHA256,
    historical_sha,
)
check(
    "historical::s5_06_pass",
    historical.get("task") == "S5.06"
    and historical.get("status") == "PASS"
    and int(historical.get("checks_passed", -1)) == 64
    and int(historical.get("checks_total", -1)) == 64
    and int(historical.get("failed_count", -1)) == 0,
)
s7 = historical.get("s5_07_best_teacher_firewall")
check(
    "historical::s5_07_pass",
    isinstance(s7, dict)
    and s7.get("task") == "S5.07"
    and s7.get("status") == "PASS"
    and int(s7.get("checks_passed", -1)) == 33
    and int(s7.get("checks_total", -1)) == 33
    and int(s7.get("failed_count", -1)) == 0,
)

for rel, expected in EXPECTED_SOURCE_SHA256.items():
    check(
        f"source::{rel}",
        sha256(ROOT / rel) == expected,
        sha256(ROOT / rel),
    )

for rel, expected in CANONICAL_SHA256.items():
    check(
        f"canonical::{rel}",
        sha256(ROOT / rel) == expected,
        sha256(ROOT / rel),
    )

fixed_sha = sha256(FIXED_VAL)
check("fixed_validation::sha", fixed_sha == FIXED_VAL_SHA256, fixed_sha)
check(
    "fixed_validation::constant",
    qp.FIXED_VALIDATION_SHA256 == FIXED_VAL_SHA256,
    qp.FIXED_VALIDATION_SHA256,
)
check(
    "fixed_validation::guard_accepts",
    qp._assert_fixed_validation_identity(FIXED_VAL) == FIXED_VAL_SHA256,
)

impl_text = QPSEUDO_IMPL.read_text(encoding="utf-8")
init_text = QPSEUDO_INIT.read_text(encoding="utf-8")
check(
    "static::private_helpers_not_exported",
    "_evaluate_qpseudo_predictions_unchecked" not in init_text
    and "_write_qpseudo_artifacts_unchecked" not in init_text,
)
check(
    "static::forbidden_gt_paths_absent",
    all(
        token not in impl_text
        for token in (
            "instances_test.json",
            "instances_train.json",
            "instances_unlabeled",
            "unlabeled_splits",
        )
    ),
)
check(
    "static::public_evaluator_guard_present",
    "def evaluate_qpseudo_predictions(" in impl_text
    and "_assert_fixed_validation_identity(ann_file)" in impl_text,
)
check(
    "static::public_writer_guard_present",
    "def write_qpseudo_artifacts(" in impl_text
    and "observed_validation_sha256 = _assert_fixed_validation_identity(" in impl_text
    and "declared_validation_sha256 != FIXED_VALIDATION_SHA256" in impl_text
    and "declared_validation_sha256 != observed_validation_sha256" in impl_text,
)

eval_calls = []
writer_calls = []
orig_eval = qp._evaluate_qpseudo_predictions_unchecked
orig_writer = qp._write_qpseudo_artifacts_unchecked


def eval_stub(*args, **kwargs):
    eval_calls.append(1)
    return {"stub": 1.0}


def writer_stub(*args, **kwargs):
    writer_calls.append(kwargs)
    return {"stub": "writer"}


try:
    qp._evaluate_qpseudo_predictions_unchecked = eval_stub
    qp._write_qpseudo_artifacts_unchecked = writer_stub

    out = qp.evaluate_qpseudo_predictions([], FIXED_VAL, {})
    check(
        "runtime::public_eval_accepts_fixed",
        out == {"stub": 1.0} and len(eval_calls) == 1,
    )

    out = qp.write_qpseudo_artifacts(
        output_dir="/tmp/s5_08_validator_writer_stub",
        predictions=[],
        metrics={},
        checkpoint_sha256="x",
        validation_split_sha256=FIXED_VAL_SHA256.upper(),
        pseudo_acceptance_config_sha256="x",
        evaluator_config_sha256="x",
        ann_file=FIXED_VAL,
    )
    check(
        "runtime::public_writer_accepts_fixed",
        out == {"stub": "writer"} and len(writer_calls) == 1,
    )
    check(
        "runtime::writer_normalizes_validation_sha",
        writer_calls[0]["validation_split_sha256"] == FIXED_VAL_SHA256,
    )

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        raw = FIXED_VAL.read_bytes()
        roles = (
            ("test", "instances_test.json"),
            ("train", "instances_train.json"),
            ("hidden_u", "instances_unlabeled.json"),
        )
        for role, name in roles:
            path = td / name
            path.write_bytes(raw + b"\n")

            before = len(eval_calls)
            rejected = False
            try:
                qp.evaluate_qpseudo_predictions([], path, {})
            except ValueError:
                rejected = True
            check(
                f"runtime::public_eval_rejects_{role}",
                rejected and len(eval_calls) == before,
            )

            before = len(writer_calls)
            rejected = False
            try:
                qp.write_qpseudo_artifacts(
                    output_dir=td / f"out_{role}",
                    predictions=[],
                    metrics={},
                    checkpoint_sha256="x",
                    validation_split_sha256=FIXED_VAL_SHA256,
                    pseudo_acceptance_config_sha256="x",
                    evaluator_config_sha256="x",
                    ann_file=path,
                )
            except ValueError:
                rejected = True
            check(
                f"runtime::public_writer_rejects_{role}",
                rejected and len(writer_calls) == before,
            )

    before = len(writer_calls)
    rejected = False
    try:
        qp.write_qpseudo_artifacts(
            output_dir="/tmp/s5_08_validator_bad_declared",
            predictions=[],
            metrics={},
            checkpoint_sha256="x",
            validation_split_sha256="0" * 64,
            pseudo_acceptance_config_sha256="x",
            evaluator_config_sha256="x",
            ann_file=FIXED_VAL,
        )
    except ValueError:
        rejected = True
    check(
        "runtime::writer_rejects_declared_sha_mismatch",
        rejected and len(writer_calls) == before,
    )

finally:
    qp._evaluate_qpseudo_predictions_unchecked = orig_eval
    qp._write_qpseudo_artifacts_unchecked = orig_writer

with tempfile.TemporaryDirectory() as td:
    td = Path(td)
    fixture = td / "fixture.json"
    predictions = build_fixture(fixture)

    cfg = Config.fromfile(str(EVALUATOR_CONFIG))
    evaluator_cfg = dict(cfg.val_evaluator)

    metrics = qp._evaluate_qpseudo_predictions_unchecked(
        predictions,
        fixture,
        evaluator_cfg,
    )
    check(
        "regression::private_synthetic_map",
        float(metrics["coco/bbox_mAP"]) == 1.0
        and float(metrics["coco/bbox_mAP_50"]) == 1.0
        and float(metrics["coco/bbox_mAP_75"]) == 1.0,
    )

    artifacts = qp._write_qpseudo_artifacts_unchecked(
        output_dir=td / "pseudo",
        predictions=predictions,
        metrics=metrics,
        checkpoint_sha256="synthetic",
        validation_split_sha256=FIXED_VAL_SHA256,
        pseudo_acceptance_config_sha256="synthetic",
        evaluator_config_sha256="synthetic",
        ann_file=fixture,
    )
    artifact_path_keys = (
        "qpseudo_validation",
        "val_last_teacher_predictions",
        "qpseudo_classwise",
    )
    artifact_sha_pairs = (
        ("qpseudo_validation", "qpseudo_validation_sha256"),
        (
            "val_last_teacher_predictions",
            "val_last_teacher_predictions_sha256",
        ),
        ("qpseudo_classwise", "qpseudo_classwise_sha256"),
    )
    artifact_paths = [
        Path(artifacts[key])
        for key in artifact_path_keys
    ]
    expected_artifact_keys = (
        set(artifact_path_keys)
        | {sha_key for _, sha_key in artifact_sha_pairs}
    )
    artifact_hashes_match = all(
        artifacts[sha_key] == sha256(artifacts[path_key])
        for path_key, sha_key in artifact_sha_pairs
    )
    check(
        "regression::private_synthetic_artifacts",
        len(artifact_paths) == 3
        and set(artifacts.keys()) == expected_artifact_keys
        and {path.name for path in artifact_paths} == {
            "qpseudo_validation.json",
            "val_last_teacher_predictions.json",
            "qpseudo_classwise.csv",
        }
        and all(path.is_file() for path in artifact_paths)
        and artifact_hashes_match,
    )

failed = [item for item in checks if not item["pass"]]
print(f"CHECKS_PASSED={len(checks) - len(failed)}")
print(f"CHECKS_TOTAL={len(checks)}")
print(f"FAILED_COUNT={len(failed)}")
print(
    "FAILED_CHECKS="
    + ("NONE" if not failed else ",".join(item["name"] for item in failed))
)

if failed:
    raise SystemExit(1)

source_sha = {
    rel: sha256(ROOT / rel)
    for rel in EXPECTED_SOURCE_SHA256
}
source_sha["scripts/S5_08_validate_qpseudo_gt_firewall.py"] = sha256(S5_08_VALIDATOR)

section = {
    "task": "S5.08",
    "status": "PASS",
    "tracker_action": "Verify no hidden-U/test GT in Q_pseudo",
    "constraint": "Validation only.",
    "historical_evidence": {
        "sha256_before_s5_08_extension": HISTORICAL_EVIDENCE_SHA256,
        "s5_06_preserved": True,
        "s5_07_preserved": True,
    },
    "canonical_rule": {
        "dataset": "fixed_validation",
        "fixed_validation_sha256": FIXED_VAL_SHA256,
        "hidden_u_gt_prohibited": True,
        "test_gt_prohibited": True,
    },
    "runtime_firewall": {
        "public_evaluator_accepts_exact_fixed_validation": True,
        "public_writer_accepts_exact_fixed_validation": True,
        "public_evaluator_rejects_test_like_gt": True,
        "public_writer_rejects_test_like_gt": True,
        "public_evaluator_rejects_train_gt": True,
        "public_writer_rejects_train_gt": True,
        "public_evaluator_rejects_hidden_u_gt": True,
        "public_writer_rejects_hidden_u_gt": True,
        "writer_rejects_declared_validation_sha_mismatch": True,
        "identity_basis": "EXACT_FIXED_VALIDATION_SHA256",
    },
    "regression": {
        "private_synthetic_helper_not_exported": True,
        "synthetic_PL_mAP_50_95": float(metrics["coco/bbox_mAP"]),
        "synthetic_PL_AP50": float(metrics["coco/bbox_mAP_50"]),
        "synthetic_PL_AP75": float(metrics["coco/bbox_mAP_75"]),
        "synthetic_artifact_writer": "PASS",
    },
    "scientific_boundaries": {
        "scientific_protocol_changed": False,
        "fixed_validation_only": True,
        "hidden_u_gt_used": False,
        "test_used": False,
        "official_detection_model": "BEST_EMA_TEACHER",
        "qpseudo_model": "LAST_EMA_TEACHER",
    },
    "source_sha256": source_sha,
    "canonical_sha256": CANONICAL_SHA256,
    "checks": checks,
    "checks_passed": len(checks),
    "checks_total": len(checks),
    "failed_count": 0,
}

historical["s5_08_hidden_u_test_gt_firewall"] = section
EVIDENCE.write_text(
    json.dumps(historical, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)

print(f"UPDATED_EVIDENCE_SHA256={sha256(EVIDENCE)}")
print("STATUS=PASS")
print("S5_08_QPSEUDO_GT_FIREWALL=PASS")
