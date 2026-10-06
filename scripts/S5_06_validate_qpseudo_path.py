"""S5.06 golden validation for the Q_pseudo LAST-Teacher path."""

import csv
import hashlib
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from mmengine.config import Config

from src.evaluation.qpseudo import (
    ACCEPTANCE_COMPARATOR,
    CHECKPOINT_ROLE,
    CLS_PSEUDO_THR,
    DATASET_ROLE,
    MAX_DETS_PER_IMAGE,
    MODEL_ROLE,
    PSEUDO_SET,
    SERIALIZATION_SCOPE,
    accepted_rcnn_classification_predictions,
    evaluate_qpseudo_predictions,
    load_last_ema_teacher_checkpoint,
    write_qpseudo_artifacts,
)

ROOT = Path("/workspace/ssod/project")
EVIDENCE = ROOT / "artifacts/preflight/evaluation/qpseudo_path_golden_test.json"

ACCEPTANCE_CONFIG = ROOT / "configs/evaluation/s5_06_qpseudo_acceptance.py"
EVALUATOR_CONFIG = ROOT / "configs/evaluation/s5_02_coco_secondary_evaluator.py"
QPSEUDO_IMPL = ROOT / "src/evaluation/qpseudo.py"
QPSEUDO_INIT = ROOT / "src/evaluation/__init__.py"

LOCAL_VAL = ROOT / "data/processed/coco/instances_val.json"
RUNTIME_VAL = Path("/workspace/ssod/data/coco/instances_val.json")

CANONICAL = {
    "sn-article.tex":
        "8d97840cc546a58aef7354af0c856119b1aea16f8be1cb7bbc1ab1a324d40fcc",
    "IMPLEMENTATION_HANDOFF.md":
        "540c94746571612ff819fb62a1e9fac6123c4035bc2642518eaa7740770820d0",
    "ARTIFACT_AND_EVIDENCE_CONTRACT.md":
        "885d4eeb66857e99744f6e8a27083e5bb695bbcc0d172cf4ebff722c53f655a4",
}

EXPECTED = {
    "validation_split_sha256":
        "33064f47ba690be13e9d418d8ec5d4a6a2bec8482c24ea3eadf83fb33d01762a",
    "evaluator_config_sha256":
        "a370acb4c0e76662b515b80943e3870a514d24f6f138db71df3680f71c204f2f",
    "pseudo_acceptance_config_sha256":
        "bb748e521af2eaeefe197a06d75b3fff1b3200f032725d42af61cfd1e2cac1c1",
    "qpseudo_impl_sha256":
        "7b0bfb96fb595be24288d083576fabfb2367d48ea1d870c1c099bbe9a889f788",
    "qpseudo_init_sha256":
        "58083297b7df31668de5fcc60323eb1b6fb14a7c5927c233e63be16622a6ac5e",
}


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def make_sample(img_id, label, score, bbox=None):
    if bbox is None:
        bbox = [10.0, 10.0, 30.0, 30.0]
    pred = SimpleNamespace(
        bboxes=torch.tensor([bbox], dtype=torch.float32),
        scores=torch.tensor([score], dtype=torch.float32),
        labels=torch.tensor([label], dtype=torch.int64),
    )
    return SimpleNamespace(
        pred_instances=pred,
        metainfo={
            "img_id": int(img_id),
            "ori_shape": (100, 100),
        },
    )


def build_fixture(path):
    categories = [
        {"id": i, "name": f"class_{i}"}
        for i in range(1, 15)
    ]
    images = []
    annotations = []

    for i in range(1, 15):
        images.append(
            {
                "id": i,
                "width": 100,
                "height": 100,
                "file_name": f"{i}.jpg",
            }
        )
        annotations.append(
            {
                "id": i,
                "image_id": i,
                "category_id": i,
                "bbox": [10, 10, 20, 20],
                "area": 400,
                "iscrowd": 0,
            }
        )

    payload = {
        "images": images,
        "annotations": annotations,
        "categories": categories,
        "licenses": [],
        "info": {},
    }
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main():
    checks = []

    def check(name, ok, observed=None):
        checks.append({
            "name": name,
            "pass": bool(ok),
            "observed": observed,
        })

    print("===== S5.06 Q_PSEUDO GOLDEN TEST =====")

    for name, expected in CANONICAL.items():
        actual = sha256(ROOT / name)
        check(
            f"canonical::{name}",
            actual == expected,
            actual,
        )

    acceptance_sha = sha256(ACCEPTANCE_CONFIG)
    evaluator_sha = sha256(EVALUATOR_CONFIG)
    impl_sha = sha256(QPSEUDO_IMPL)
    init_sha = sha256(QPSEUDO_INIT)
    local_val_sha = sha256(LOCAL_VAL)
    runtime_val_sha = sha256(RUNTIME_VAL)

    check(
        "provenance::validation_split_local",
        local_val_sha == EXPECTED["validation_split_sha256"],
        local_val_sha,
    )
    check(
        "provenance::validation_split_runtime",
        runtime_val_sha == EXPECTED["validation_split_sha256"],
        runtime_val_sha,
    )
    check(
        "provenance::validation_local_runtime_identity",
        local_val_sha == runtime_val_sha,
        {
            "local": local_val_sha,
            "runtime": runtime_val_sha,
        },
    )
    check(
        "provenance::evaluator_config",
        evaluator_sha == EXPECTED["evaluator_config_sha256"],
        evaluator_sha,
    )
    check(
        "provenance::acceptance_config",
        acceptance_sha == EXPECTED["pseudo_acceptance_config_sha256"],
        acceptance_sha,
    )
    check(
        "provenance::qpseudo_impl",
        impl_sha == EXPECTED["qpseudo_impl_sha256"],
        impl_sha,
    )
    check(
        "provenance::qpseudo_init",
        init_sha == EXPECTED["qpseudo_init_sha256"],
        init_sha,
    )

    cfg = Config.fromfile(str(ACCEPTANCE_CONFIG))
    evaluator_cfg = Config.fromfile(str(EVALUATOR_CONFIG)).val_evaluator

    check(
        "config::checkpoint_role",
        cfg.checkpoint_role == "LAST_EMA_TEACHER",
        cfg.checkpoint_role,
    )
    check(
        "config::model_role",
        cfg.model_role == "EMA_TEACHER",
        cfg.model_role,
    )
    check(
        "config::serialization_scope",
        cfg.serialization_scope == "EMA_TEACHER_ONLY",
        cfg.serialization_scope,
    )
    check(
        "config::dataset",
        cfg.dataset == "fixed_validation",
        cfg.dataset,
    )
    check(
        "config::pseudo_set",
        cfg.pseudo_set == "accepted_rcnn_classification",
        cfg.pseudo_set,
    )
    check(
        "config::cls_thr",
        abs(float(cfg.cls_pseudo_thr) - 0.90) < 1e-12,
        float(cfg.cls_pseudo_thr),
    )
    check(
        "config::strict_greater",
        cfg.acceptance_comparator == "STRICT_GREATER",
        cfg.acceptance_comparator,
    )
    check(
        "config::native_max100",
        int(cfg.rcnn_max_per_img) == 100,
        int(cfg.rcnn_max_per_img),
    )
    check(
        "config::hidden_u_gt_false",
        cfg.hidden_u_gt_used is False,
        cfg.hidden_u_gt_used,
    )
    check(
        "config::test_false",
        cfg.test_used is False,
        cfg.test_used,
    )

    check(
        "constants::checkpoint_role",
        CHECKPOINT_ROLE == "LAST_EMA_TEACHER",
        CHECKPOINT_ROLE,
    )
    check(
        "constants::model_role",
        MODEL_ROLE == "EMA_TEACHER",
        MODEL_ROLE,
    )
    check(
        "constants::serialization_scope",
        SERIALIZATION_SCOPE == "EMA_TEACHER_ONLY",
        SERIALIZATION_SCOPE,
    )
    check(
        "constants::dataset",
        DATASET_ROLE == "fixed_validation",
        DATASET_ROLE,
    )
    check(
        "constants::pseudo_set",
        PSEUDO_SET == "accepted_rcnn_classification",
        PSEUDO_SET,
    )
    check(
        "constants::cls_thr",
        abs(float(CLS_PSEUDO_THR) - 0.90) < 1e-12,
        float(CLS_PSEUDO_THR),
    )
    check(
        "constants::comparator",
        ACCEPTANCE_COMPARATOR == "STRICT_GREATER",
        ACCEPTANCE_COMPARATOR,
    )
    check(
        "constants::max100",
        int(MAX_DETS_PER_IMAGE) == 100,
        int(MAX_DETS_PER_IMAGE),
    )

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)

        teacher = torch.nn.Linear(2, 2)
        original_state = {
            k: v.detach().cpu().clone()
            for k, v in teacher.state_dict().items()
        }
        model = SimpleNamespace(
            teacher=teacher,
            semi_test_cfg={"predict_on": "teacher"},
        )

        ckpt = td / "last_ema_teacher.pth"
        torch.save(
            {
                "meta": {
                    "scientific_checkpoint_role": "LAST",
                    "model_role": "EMA_TEACHER",
                    "serialization_scope": "EMA_TEACHER_ONLY",
                    "optimizer_update": 2064,
                },
                "state_dict": original_state,
            },
            ckpt,
        )

        with torch.no_grad():
            for p in teacher.parameters():
                p.zero_()

        load_info = load_last_ema_teacher_checkpoint(model, ckpt)

        restored = all(
            torch.equal(
                teacher.state_dict()[key].cpu(),
                value,
            )
            for key, value in original_state.items()
        )
        check("loader::strict_state_restored", restored, restored)
        check(
            "loader::reported_role",
            load_info["checkpoint_role"] == "LAST_EMA_TEACHER",
            load_info,
        )
        check(
            "loader::reported_scope",
            load_info["serialization_scope"] == "EMA_TEACHER_ONLY",
            load_info,
        )
        check(
            "loader::checkpoint_sha",
            load_info["checkpoint_sha256"] == sha256(ckpt),
            load_info["checkpoint_sha256"],
        )

        def rejected(payload, expected_text):
            path = td / (
                "bad_" + hashlib.sha256(
                    json.dumps(
                        sorted(payload.get("meta", {}).items())
                    ).encode("utf-8")
                ).hexdigest()[:10] + ".pth"
            )
            torch.save(payload, path)
            try:
                load_last_ema_teacher_checkpoint(model, path)
            except Exception as exc:
                return expected_text in str(exc)
            return False

        bad_best = {
            "meta": {
                "scientific_checkpoint_role": "BEST",
                "model_role": "EMA_TEACHER",
                "serialization_scope": "EMA_TEACHER_ONLY",
            },
            "state_dict": original_state,
        }
        check(
            "loader::best_role_rejected",
            rejected(
                bad_best,
                "scientific_checkpoint_role='LAST'",
            ),
            True,
        )

        bad_model_role = {
            "meta": {
                "scientific_checkpoint_role": "LAST",
                "model_role": "STUDENT",
                "serialization_scope": "EMA_TEACHER_ONLY",
            },
            "state_dict": original_state,
        }
        check(
            "loader::student_role_rejected",
            rejected(
                bad_model_role,
                "model_role='EMA_TEACHER'",
            ),
            True,
        )

        bad_scope = {
            "meta": {
                "scientific_checkpoint_role": "LAST",
                "model_role": "EMA_TEACHER",
                "serialization_scope": "FULL_MODEL",
            },
            "state_dict": original_state,
        }
        check(
            "loader::full_model_scope_rejected",
            rejected(
                bad_scope,
                "serialization_scope='EMA_TEACHER_ONLY'",
            ),
            True,
        )

        operational = {
            "meta": {
                "scientific_checkpoint_role": "LAST",
                "model_role": "EMA_TEACHER",
                "serialization_scope": "EMA_TEACHER_ONLY",
            },
            "state_dict": original_state,
            "optimizer": {"forbidden": True},
        }
        operational_path = td / "operational.pth"
        torch.save(operational, operational_path)
        operational_rejected = False
        try:
            load_last_ema_teacher_checkpoint(
                model,
                operational_path,
            )
        except ValueError as exc:
            operational_rejected = (
                "operational resume state" in str(exc)
            )
        check(
            "loader::operational_state_rejected",
            operational_rejected,
            operational_rejected,
        )

        student_model = SimpleNamespace(
            teacher=torch.nn.Linear(2, 2),
            semi_test_cfg={"predict_on": "student"},
        )
        student_predict_rejected = False
        try:
            load_last_ema_teacher_checkpoint(
                student_model,
                ckpt,
            )
        except ValueError as exc:
            student_predict_rejected = (
                "predict_on='teacher'" in str(exc)
            )
        check(
            "loader::predict_on_student_rejected",
            student_predict_rejected,
            student_predict_rejected,
        )

        boundary_pred = SimpleNamespace(
            bboxes=torch.tensor(
                [
                    [0, 0, 10, 10],
                    [20, 20, 30, 30],
                    [40, 40, 50, 50],
                ],
                dtype=torch.float32,
            ),
            scores=torch.tensor(
                [0.90, 0.9001, 0.99],
                dtype=torch.float32,
            ),
            labels=torch.tensor(
                [0, 1, 2],
                dtype=torch.int64,
            ),
        )
        boundary_sample = SimpleNamespace(
            pred_instances=boundary_pred,
            metainfo={
                "img_id": 1,
                "ori_shape": (100, 100),
            },
        )

        boundary = accepted_rcnn_classification_predictions(
            [boundary_sample]
        )[0]

        check(
            "acceptance::exact_090_rejected",
            boundary["accepted_count"] == 2
            and not np.any(
                np.isclose(
                    boundary["scores"],
                    0.90,
                    rtol=0.0,
                    atol=1e-7,
                )
            ),
            {
                "accepted_count":
                    boundary["accepted_count"],
                "scores":
                    boundary["scores"].tolist(),
            },
        )
        check(
            "acceptance::above_090_accepted",
            boundary["accepted_count"] == 2,
            boundary["accepted_count"],
        )
        check(
            "acceptance::native_count_preserved",
            boundary["detector_native_count"] == 3,
            boundary["detector_native_count"],
        )

        too_many_pred = SimpleNamespace(
            bboxes=torch.zeros((101, 4), dtype=torch.float32),
            scores=torch.full((101,), 0.99),
            labels=torch.zeros((101,), dtype=torch.int64),
        )
        too_many_sample = SimpleNamespace(
            pred_instances=too_many_pred,
            metainfo={
                "img_id": 1,
                "ori_shape": (100, 100),
            },
        )
        max100_rejected = False
        try:
            accepted_rcnn_classification_predictions(
                [too_many_sample]
            )
        except ValueError as exc:
            max100_rejected = (
                "more than 100 detector-native predictions"
                in str(exc)
            )
        check(
            "acceptance::max100_guard",
            max100_rejected,
            max100_rejected,
        )

        fixture = td / "fixture.json"
        build_fixture(fixture)

        samples = [
            make_sample(i, i - 1, 0.99)
            for i in range(1, 15)
        ]
        accepted = (
            accepted_rcnn_classification_predictions(
                samples
            )
        )

        metrics = evaluate_qpseudo_predictions(
            accepted,
            fixture,
            evaluator_cfg,
        )

        check(
            "metric::PL_mAP_50_95",
            float(metrics["coco/bbox_mAP"]) == 1.0,
            metrics["coco/bbox_mAP"],
        )
        check(
            "metric::PL_AP50",
            float(metrics["coco/bbox_mAP_50"]) == 1.0,
            metrics["coco/bbox_mAP_50"],
        )
        check(
            "metric::PL_AP75",
            float(metrics["coco/bbox_mAP_75"]) == 1.0,
            metrics["coco/bbox_mAP_75"],
        )

        class_keys = [
            key
            for key in metrics
            if "/classwise/" in key
        ]
        check(
            "metric::classwise_key_count",
            len(class_keys) == 42,
            len(class_keys),
        )

        prod = td / "pseudo"
        artifacts = write_qpseudo_artifacts(
            output_dir=prod,
            predictions=accepted,
            metrics=metrics,
            checkpoint_sha256=sha256(ckpt),
            validation_split_sha256=
                EXPECTED["validation_split_sha256"],
            pseudo_acceptance_config_sha256=
                acceptance_sha,
            evaluator_config_sha256=
                evaluator_sha,
            ann_file=fixture,
        )

        validation = json.loads(
            (prod / "qpseudo_validation.json")
            .read_text(encoding="utf-8")
        )
        predictions_json = json.loads(
            (prod / "val_last_teacher_predictions.json")
            .read_text(encoding="utf-8")
        )

        with (
            prod / "qpseudo_classwise.csv"
        ).open("r", encoding="utf-8", newline="") as handle:
            classwise_rows = list(csv.DictReader(handle))

        required_validation_fields = {
            "checkpoint_role",
            "checkpoint_sha256",
            "dataset",
            "validation_split_sha256",
            "pseudo_set",
            "pseudo_acceptance_config_sha256",
            "evaluator_config_sha256",
            "hidden_u_gt_used",
            "test_used",
            "PL_mAP_50_95",
            "PL_AP50",
            "PL_AP75",
        }

        check(
            "artifact::validation_required_fields",
            required_validation_fields.issubset(
                validation.keys()
            ),
            sorted(validation.keys()),
        )
        check(
            "artifact::validation_checkpoint_role",
            validation["checkpoint_role"]
            == "LAST_EMA_TEACHER",
            validation["checkpoint_role"],
        )
        check(
            "artifact::validation_checkpoint_sha_match",
            validation["checkpoint_sha256"]
            == sha256(ckpt),
            validation["checkpoint_sha256"],
        )
        check(
            "artifact::validation_dataset",
            validation["dataset"]
            == "fixed_validation",
            validation["dataset"],
        )
        check(
            "artifact::validation_pseudo_set",
            validation["pseudo_set"]
            == "accepted_rcnn_classification",
            validation["pseudo_set"],
        )
        check(
            "artifact::validation_hidden_u_false",
            validation["hidden_u_gt_used"] is False,
            validation["hidden_u_gt_used"],
        )
        check(
            "artifact::validation_test_false",
            validation["test_used"] is False,
            validation["test_used"],
        )
        check(
            "artifact::validation_primary_metric",
            float(validation["PL_mAP_50_95"]) == 1.0,
            validation["PL_mAP_50_95"],
        )
        check(
            "artifact::prediction_image_count",
            len(predictions_json["images"]) == 14,
            len(predictions_json["images"]),
        )
        check(
            "artifact::prediction_threshold",
            abs(
                float(
                    predictions_json[
                        "classification_threshold"
                    ]
                ) - 0.90
            ) < 1e-12,
            predictions_json["classification_threshold"],
        )
        check(
            "artifact::prediction_comparator",
            predictions_json[
                "acceptance_comparator"
            ] == "STRICT_GREATER",
            predictions_json["acceptance_comparator"],
        )
        check(
            "artifact::prediction_no_second_nms",
            predictions_json["second_nms"] is False,
            predictions_json["second_nms"],
        )
        check(
            "artifact::prediction_no_topk",
            predictions_json["topk_truncation"] is False,
            predictions_json["topk_truncation"],
        )
        check(
            "artifact::classwise_14_rows",
            len(classwise_rows) == 14,
            len(classwise_rows),
        )
        check(
            "artifact::classwise_columns",
            set(classwise_rows[0].keys())
            == {
                "category_id",
                "class_name",
                "PL_AP_50_95",
                "PL_AP50",
                "PL_AP75",
            },
            sorted(classwise_rows[0].keys()),
        )
        check(
            "artifact::returned_paths_and_hashes",
            all(
                key in artifacts
                for key in (
                    "qpseudo_validation",
                    "val_last_teacher_predictions",
                    "qpseudo_classwise",
                    "qpseudo_validation_sha256",
                    "val_last_teacher_predictions_sha256",
                    "qpseudo_classwise_sha256",
                )
            ),
            sorted(artifacts.keys()),
        )

    impl_text = QPSEUDO_IMPL.read_text(encoding="utf-8")
    acceptance_text = ACCEPTANCE_CONFIG.read_text(
        encoding="utf-8"
    )

    check(
        "firewall::best_teacher_path_absent",
        "best_ema_teacher.pth" not in impl_text
        and "best_ema_teacher.pth" not in acceptance_text,
        True,
    )
    check(
        "firewall::test_split_absent",
        "instances_test.json" not in impl_text
        and "instances_test.json" not in acceptance_text,
        True,
    )
    check(
        "firewall::latest_resume_absent",
        "latest_resume.pth" not in impl_text
        and "latest_resume.pth" not in acceptance_text,
        True,
    )

    passed = sum(1 for item in checks if item["pass"])
    total = len(checks)
    failed = [
        item["name"]
        for item in checks
        if not item["pass"]
    ]
    all_pass = passed == total

    evidence = {
        "schema_version": "1.0.0",
        "task": "S5.06",
        "evidence_type": "qpseudo_path_golden_test",
        "status": "PASS" if all_pass else "FAIL",
        "scientific_protocol": {
            "primary_metric": "PL_mAP_50_95",
            "checkpoint_role": "LAST_EMA_TEACHER",
            "model_role": "EMA_TEACHER",
            "serialization_scope": "EMA_TEACHER_ONLY",
            "dataset": "fixed_validation",
            "pseudo_set": "accepted_rcnn_classification",
            "classification_threshold": 0.90,
            "acceptance_comparator": "STRICT_GREATER",
            "score_equal_0_90_accepted": False,
            "detector_native_nms": True,
            "native_max_per_img": 100,
            "second_nms": False,
            "topk_truncation": False,
            "hidden_u_gt_used": False,
            "test_used": False,
            "best_teacher_used": False,
            "latest_resume_used": False,
        },
        "provenance": {
            "validation_split_sha256":
                local_val_sha,
            "pseudo_acceptance_config_sha256":
                acceptance_sha,
            "evaluator_config_sha256":
                evaluator_sha,
            "qpseudo_impl_sha256":
                impl_sha,
            "qpseudo_init_sha256":
                init_sha,
            "canonical_sha256": {
                name: sha256(ROOT / name)
                for name in CANONICAL
            },
        },
        "synthetic_golden": {
            "fixture": "SYNTHETIC_COCO_14_CLASS_PERFECT_PREDICTIONS",
            "PL_mAP_50_95":
                1.0,
            "PL_AP50":
                1.0,
            "PL_AP75":
                1.0,
            "classwise_row_count":
                14,
            "threshold_boundary_exact_0_90_rejected":
                True,
        },
        "checks_passed": passed,
        "checks_total": total,
        "failed_count": len(failed),
        "failed_checks": failed,
        "checks": checks,
        "source_sha256": {
            "configs/evaluation/s5_06_qpseudo_acceptance.py":
                acceptance_sha,
            "src/evaluation/__init__.py":
                init_sha,
            "src/evaluation/qpseudo.py":
                impl_sha,
            "configs/evaluation/s5_02_coco_secondary_evaluator.py":
                evaluator_sha,
            "src/metrics/protocol_coco_metric.py":
                sha256(
                    ROOT / "src/metrics/protocol_coco_metric.py"
                ),
            "scripts/S5_06_validate_qpseudo_path.py":
                sha256(Path(__file__)),
        },
    }

    EVIDENCE.parent.mkdir(parents=True, exist_ok=True)
    EVIDENCE.write_text(
        json.dumps(
            evidence,
            indent=2,
            sort_keys=True,
        ) + "\n",
        encoding="utf-8",
    )

    print("CHECKS_PASSED=" + str(passed))
    print("CHECKS_TOTAL=" + str(total))
    print("FAILED_COUNT=" + str(len(failed)))
    print(
        "FAILED_CHECKS="
        + (",".join(failed) if failed else "NONE")
    )
    print("STATUS=" + evidence["status"])
    print("EVIDENCE=" + str(EVIDENCE))
    print("EVIDENCE_SHA256=" + sha256(EVIDENCE))
    print(
        "QPSEUDO_PATH_GOLDEN_TEST="
        + ("PASS" if all_pass else "FAIL")
    )

    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())