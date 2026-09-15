"""Q_pseudo evaluation path for accepted RCNN classification pseudo labels.

Scientific contract:
- checkpoint: LAST EMA Teacher;
- dataset: fixed validation;
- pseudo set: detector-native Teacher predictions accepted by strict
  classification score > 0.90;
- primary metric: PL-mAP@[0.50:0.95];
- hidden-U GT and test GT are prohibited.
"""

import csv
import hashlib
import json
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence

import numpy as np
import torch

from src.metrics.protocol_coco_metric import ProtocolCocoMetric


CHECKPOINT_ROLE = "LAST_EMA_TEACHER"
SCIENTIFIC_CHECKPOINT_ROLE = "LAST"
MODEL_ROLE = "EMA_TEACHER"
SERIALIZATION_SCOPE = "EMA_TEACHER_ONLY"
DATASET_ROLE = "fixed_validation"
PSEUDO_SET = "accepted_rcnn_classification"
CLS_PSEUDO_THR = 0.90
ACCEPTANCE_COMPARATOR = "STRICT_GREATER"
MAX_DETS_PER_IMAGE = 100
FIXED_VALIDATION_SHA256 = "33064f47ba690be13e9d418d8ec5d4a6a2bec8482c24ea3eadf83fb33d01762a"


def sha256_file(path) -> str:
    """Return SHA-256 of exact file bytes."""
    path = Path(path)
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _assert_fixed_validation_identity(ann_file) -> str:
    """Fail closed unless ann_file is the exact locked fixed-validation COCO."""
    path = Path(ann_file)
    if not path.is_file():
        raise FileNotFoundError(
            f"Q_pseudo fixed-validation annotation file not found: {path}"
        )

    observed_sha256 = sha256_file(path)
    if observed_sha256 != FIXED_VALIDATION_SHA256:
        raise ValueError(
            "Q_pseudo ann_file is not the locked fixed validation; "
            f"expected_sha256={FIXED_VALIDATION_SHA256}, "
            f"observed_sha256={observed_sha256}"
        )

    return observed_sha256


def load_last_ema_teacher_checkpoint(model, checkpoint_path) -> Dict[str, object]:
    """Fail-closed loader for teacher-only scientific LAST checkpoint."""
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            "LAST EMA Teacher checkpoint does not exist: "
            f"{checkpoint_path}"
        )

    payload = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(payload, dict):
        raise TypeError("LAST EMA Teacher checkpoint payload must be a dict")

    meta = payload.get("meta")
    state_dict = payload.get("state_dict")

    if not isinstance(meta, dict):
        raise ValueError("LAST EMA Teacher checkpoint is missing dict meta")
    if meta.get("scientific_checkpoint_role") != SCIENTIFIC_CHECKPOINT_ROLE:
        raise ValueError(
            "Q_pseudo requires scientific_checkpoint_role='LAST'"
        )
    if meta.get("model_role") != MODEL_ROLE:
        raise ValueError("Q_pseudo requires model_role='EMA_TEACHER'")
    if meta.get("serialization_scope") != SERIALIZATION_SCOPE:
        raise ValueError(
            "Q_pseudo requires serialization_scope='EMA_TEACHER_ONLY'"
        )
    if not isinstance(state_dict, dict) or not state_dict:
        raise ValueError(
            "LAST EMA Teacher checkpoint state_dict is missing or empty"
        )

    # Scientific teacher-only checkpoints must not silently become operational
    # resume checkpoints.
    forbidden_payload_keys = {
        "optimizer",
        "optim_wrapper",
        "param_schedulers",
        "message_hub",
    }
    present_forbidden = sorted(forbidden_payload_keys.intersection(payload))
    if present_forbidden:
        raise ValueError(
            "Scientific LAST EMA Teacher checkpoint contains operational "
            "resume state: " + ",".join(present_forbidden)
        )

    teacher = getattr(model, "teacher", None)
    if teacher is None:
        raise TypeError("Q_pseudo model must expose model.teacher")

    semi_test_cfg = getattr(model, "semi_test_cfg", {})
    predict_on = semi_test_cfg.get("predict_on", "teacher")
    if predict_on != "teacher":
        raise ValueError(
            "Q_pseudo requires semi_test_cfg.predict_on='teacher'; "
            f"got {predict_on!r}"
        )

    incompatible = teacher.load_state_dict(state_dict, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(
            "Strict LAST EMA Teacher load returned incompatible keys"
        )

    teacher.eval()

    return {
        "checkpoint_role": CHECKPOINT_ROLE,
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "scientific_checkpoint_role": SCIENTIFIC_CHECKPOINT_ROLE,
        "model_role": MODEL_ROLE,
        "serialization_scope": SERIALIZATION_SCOPE,
        "optimizer_update": meta.get("optimizer_update"),
    }


def _as_numpy(value, dtype=None):
    if torch.is_tensor(value):
        value = value.detach().cpu().numpy()
    array = np.asarray(value)
    if dtype is not None:
        array = array.astype(dtype, copy=False)
    return array


def accepted_rcnn_classification_predictions(
    data_samples: Iterable,
    score_thr: float = CLS_PSEUDO_THR,
    max_dets_per_image: int = MAX_DETS_PER_IMAGE,
) -> List[Dict[str, object]]:
    """Extract the accepted RCNN classification pseudo-label set.

    Input samples must already be detector-native Teacher predictions after
    the detector's own RCNN NMS and max-per-image processing. No second NMS
    and no top-k truncation are performed here.

    MMDetection 3.3.0 SoftTeacher uses:
        scores > cls_pseudo_thr
    Therefore a score exactly equal to 0.90 is rejected.
    """
    score_thr = float(score_thr)
    max_dets_per_image = int(max_dets_per_image)

    if abs(score_thr - CLS_PSEUDO_THR) > 1e-12:
        raise ValueError(
            "Q_pseudo classification threshold must be exactly 0.90"
        )
    if max_dets_per_image != MAX_DETS_PER_IMAGE:
        raise ValueError(
            "Q_pseudo max_dets_per_image must be exactly 100"
        )

    output = []

    for sample in data_samples:
        pred = getattr(sample, "pred_instances", None)
        if pred is None:
            raise ValueError(
                "Q_pseudo Teacher prediction is missing pred_instances"
            )

        bboxes = _as_numpy(pred.bboxes, np.float32).reshape(-1, 4)
        scores = _as_numpy(pred.scores, np.float32).reshape(-1)
        labels = _as_numpy(pred.labels, np.int64).reshape(-1)

        count = int(len(scores))
        if len(bboxes) != count or len(labels) != count:
            raise ValueError(
                "Prediction bboxes/scores/labels length mismatch"
            )
        if count > max_dets_per_image:
            raise ValueError(
                "Q_pseudo received more than 100 detector-native predictions"
            )
        if not np.all(np.isfinite(bboxes)):
            raise ValueError("Q_pseudo prediction contains non-finite bbox")
        if not np.all(np.isfinite(scores)):
            raise ValueError("Q_pseudo prediction contains non-finite score")
        if np.any(labels < 0) or np.any(labels >= 14):
            raise ValueError(
                "Q_pseudo prediction label index must be in [0, 13]"
            )

        # Exact SoftTeacher classification-filter semantics.
        keep = scores > score_thr

        metainfo = getattr(sample, "metainfo", {})
        image_id = metainfo.get("img_id")
        if image_id is None and hasattr(sample, "img_id"):
            image_id = sample.img_id
        if image_id is None:
            raise ValueError(
                "Q_pseudo prediction sample is missing img_id"
            )

        ori_shape = metainfo.get("ori_shape")
        if ori_shape is None and hasattr(sample, "ori_shape"):
            ori_shape = sample.ori_shape
        if ori_shape is None:
            raise ValueError(
                "Q_pseudo prediction sample is missing ori_shape"
            )

        output.append(
            {
                "img_id": int(image_id),
                "ori_shape": tuple(int(x) for x in ori_shape[:2]),
                "bboxes": bboxes[keep],
                "scores": scores[keep],
                "labels": labels[keep],
                "detector_native_count": count,
                "accepted_count": int(np.count_nonzero(keep)),
            }
        )

    return output


def _load_validation_categories(ann_file):
    payload = json.loads(
        Path(ann_file).read_text(encoding="utf-8")
    )
    categories = sorted(
        payload.get("categories", []),
        key=lambda item: int(item["id"]),
    )
    category_ids = [int(item["id"]) for item in categories]

    if category_ids != list(range(1, 15)):
        raise ValueError(
            "Q_pseudo fixed-validation category IDs must be 1..14"
        )

    return payload, categories


def _validate_evaluator_config(evaluator_cfg: Mapping[str, object]) -> Dict:
    cfg = dict(evaluator_cfg)
    cfg.pop("_delete_", None)

    metric_type = cfg.pop("type", None)
    if metric_type != "ProtocolCocoMetric":
        raise ValueError(
            "Q_pseudo evaluator must use ProtocolCocoMetric"
        )

    if cfg.get("metric") != "bbox":
        raise ValueError("Q_pseudo evaluator metric must be bbox")
    if cfg.get("classwise") is not True:
        raise ValueError("Q_pseudo evaluator must enable classwise")
    if tuple(cfg.get("proposal_nums", ())) != (1, 10, 100):
        raise ValueError(
            "Q_pseudo evaluator proposal_nums must be (1, 10, 100)"
        )
    if list(cfg.get("metric_items", [])) != [
        "mAP",
        "mAP_50",
        "mAP_75",
        "AR@1000",
    ]:
        raise ValueError("Q_pseudo evaluator metric_items mismatch")
    if cfg.get("format_only") is not False:
        raise ValueError("Q_pseudo evaluator format_only must be False")
    if cfg.get("prefix") != "coco":
        raise ValueError("Q_pseudo evaluator prefix must be 'coco'")

    return cfg


def evaluate_qpseudo_predictions(
    predictions: Sequence[Mapping[str, object]],
    ann_file,
    evaluator_cfg: Mapping[str, object],
) -> Dict[str, float]:
    """Evaluate Q_pseudo only against the exact locked fixed validation."""
    _assert_fixed_validation_identity(ann_file)
    return _evaluate_qpseudo_predictions_unchecked(
        predictions,
        ann_file,
        evaluator_cfg,
    )


def _evaluate_qpseudo_predictions_unchecked(
    predictions: Sequence[Mapping[str, object]],
    ann_file,
    evaluator_cfg: Mapping[str, object],
) -> Dict[str, float]:
    """Evaluate accepted pseudo labels with locked ProtocolCocoMetric."""
    payload, categories = _load_validation_categories(ann_file)

    image_ids = [
        int(item["id"])
        for item in payload.get("images", [])
    ]
    prediction_ids = [
        int(item["img_id"])
        for item in predictions
    ]

    if len(predictions) != len(image_ids):
        raise ValueError(
            "Q_pseudo predictions must cover every fixed-validation image"
        )
    if prediction_ids != image_ids:
        raise ValueError(
            "Q_pseudo prediction order/IDs must exactly match "
            "fixed validation"
        )

    cfg = _validate_evaluator_config(evaluator_cfg)
    cfg["ann_file"] = str(ann_file)

    metric = ProtocolCocoMetric(**{
        key: value
        for key, value in cfg.items()
        if key != "type"
    })
    metric.dataset_meta = {
        "classes": tuple(
            str(item["name"])
            for item in categories
        )
    }

    for row in predictions:
        sample = {
            "img_id": int(row["img_id"]),
            "ori_shape": tuple(row["ori_shape"]),
            "pred_instances": {
                "bboxes": torch.as_tensor(
                    np.asarray(
                        row["bboxes"],
                        dtype=np.float32,
                    )
                ),
                "scores": torch.as_tensor(
                    np.asarray(
                        row["scores"],
                        dtype=np.float32,
                    )
                ),
                "labels": torch.as_tensor(
                    np.asarray(
                        row["labels"],
                        dtype=np.int64,
                    )
                ),
            },
        }
        metric.process({}, [sample])

    observed = metric.evaluate(size=len(predictions))

    required = (
        "coco/bbox_mAP",
        "coco/bbox_mAP_50",
        "coco/bbox_mAP_75",
    )
    missing = [
        key
        for key in required
        if key not in observed
    ]
    if missing:
        raise KeyError(
            "Q_pseudo evaluator missing required keys: "
            + ",".join(missing)
        )

    return {
        key: float(value)
        for key, value in observed.items()
    }


def _prediction_artifact_rows(predictions, category_ids):
    rows = []

    for prediction in predictions:
        bboxes = np.asarray(
            prediction["bboxes"],
            dtype=float,
        ).reshape(-1, 4)
        scores = np.asarray(
            prediction["scores"],
            dtype=float,
        ).reshape(-1)
        labels = np.asarray(
            prediction["labels"],
            dtype=int,
        ).reshape(-1)

        detections = []
        for bbox, score, label in zip(bboxes, scores, labels):
            label = int(label)
            if label < 0 or label >= len(category_ids):
                raise ValueError(
                    "Q_pseudo label index outside category range"
                )

            detections.append(
                {
                    "category_id": int(category_ids[label]),
                    "bbox_xyxy": [
                        float(x)
                        for x in bbox
                    ],
                    "score": float(score),
                }
            )

        rows.append(
            {
                "image_id": int(prediction["img_id"]),
                "ori_shape": [
                    int(x)
                    for x in prediction["ori_shape"]
                ],
                "detector_native_count":
                    int(prediction["detector_native_count"]),
                "accepted_count":
                    int(prediction["accepted_count"]),
                "detections": detections,
            }
        )

    return rows


def write_qpseudo_artifacts(
    output_dir,
    predictions: Sequence[Mapping[str, object]],
    metrics: Mapping[str, float],
    checkpoint_sha256: str,
    validation_split_sha256: str,
    pseudo_acceptance_config_sha256: str,
    evaluator_config_sha256: str,
    ann_file,
) -> Dict[str, str]:
    """Write Q_pseudo artifacts only for the exact locked fixed validation."""
    observed_validation_sha256 = _assert_fixed_validation_identity(
        ann_file
    )
    declared_validation_sha256 = str(
        validation_split_sha256
    ).lower()

    if declared_validation_sha256 != FIXED_VALIDATION_SHA256:
        raise ValueError(
            "Q_pseudo validation_split_sha256 must identify the "
            "locked fixed validation"
        )

    if declared_validation_sha256 != observed_validation_sha256:
        raise ValueError(
            "Q_pseudo validation_split_sha256 does not match ann_file"
        )

    return _write_qpseudo_artifacts_unchecked(
        output_dir=output_dir,
        predictions=predictions,
        metrics=metrics,
        checkpoint_sha256=checkpoint_sha256,
        validation_split_sha256=declared_validation_sha256,
        pseudo_acceptance_config_sha256=
            pseudo_acceptance_config_sha256,
        evaluator_config_sha256=evaluator_config_sha256,
        ann_file=ann_file,
    )


def _write_qpseudo_artifacts_unchecked(
    output_dir,
    predictions: Sequence[Mapping[str, object]],
    metrics: Mapping[str, float],
    checkpoint_sha256: str,
    validation_split_sha256: str,
    pseudo_acceptance_config_sha256: str,
    evaluator_config_sha256: str,
    ann_file,
) -> Dict[str, str]:
    """Write the three required Q_pseudo production artifacts."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    payload, categories = _load_validation_categories(ann_file)
    category_ids = [
        int(item["id"])
        for item in categories
    ]
    class_names = [
        str(item["name"])
        for item in categories
    ]

    validation = {
        "schema_version": "1.0.0",
        "checkpoint_role": CHECKPOINT_ROLE,
        "checkpoint_sha256": str(checkpoint_sha256),
        "dataset": DATASET_ROLE,
        "validation_split_sha256":
            str(validation_split_sha256),
        "pseudo_set": PSEUDO_SET,
        "pseudo_acceptance_config_sha256":
            str(pseudo_acceptance_config_sha256),
        "evaluator_config_sha256":
            str(evaluator_config_sha256),
        "hidden_u_gt_used": False,
        "test_used": False,
        "acceptance": {
            "cls_pseudo_thr": CLS_PSEUDO_THR,
            "comparator": ACCEPTANCE_COMPARATOR,
            "native_max_per_img": MAX_DETS_PER_IMAGE,
            "second_nms": False,
            "topk_truncation": False,
        },
        "validation_image_count":
            int(len(payload.get("images", []))),
        "accepted_pseudo_count":
            int(sum(
                int(row["accepted_count"])
                for row in predictions
            )),
        "PL_mAP_50_95":
            float(metrics["coco/bbox_mAP"]),
        "PL_AP50":
            float(metrics["coco/bbox_mAP_50"]),
        "PL_AP75":
            float(metrics["coco/bbox_mAP_75"]),
    }

    validation_path = (
        output_dir / "qpseudo_validation.json"
    )
    predictions_path = (
        output_dir / "val_last_teacher_predictions.json"
    )
    classwise_path = (
        output_dir / "qpseudo_classwise.csv"
    )

    validation_path.write_text(
        json.dumps(
            validation,
            indent=2,
            sort_keys=True,
        ) + "\n",
        encoding="utf-8",
    )

    prediction_payload = {
        "schema_version": "1.0.0",
        "checkpoint_role": CHECKPOINT_ROLE,
        "checkpoint_sha256": str(checkpoint_sha256),
        "dataset": DATASET_ROLE,
        "pseudo_set": PSEUDO_SET,
        "classification_threshold": CLS_PSEUDO_THR,
        "acceptance_comparator":
            ACCEPTANCE_COMPARATOR,
        "second_nms": False,
        "topk_truncation": False,
        "images": _prediction_artifact_rows(
            predictions,
            category_ids,
        ),
    }
    predictions_path.write_text(
        json.dumps(
            prediction_payload,
            indent=2,
            sort_keys=True,
        ) + "\n",
        encoding="utf-8",
    )

    with classwise_path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "category_id",
                "class_name",
                "PL_AP_50_95",
                "PL_AP50",
                "PL_AP75",
            ],
        )
        writer.writeheader()

        for category_id, class_name in zip(
            category_ids,
            class_names,
        ):
            prefix = f"coco/classwise/{class_name}/"
            writer.writerow(
                {
                    "category_id": int(category_id),
                    "class_name": class_name,
                    "PL_AP_50_95":
                        float(metrics[
                            prefix + "AP_50_95"
                        ]),
                    "PL_AP50":
                        float(metrics[
                            prefix + "AP50"
                        ]),
                    "PL_AP75":
                        float(metrics[
                            prefix + "AP75"
                        ]),
                }
            )

    return {
        "qpseudo_validation":
            str(validation_path),
        "val_last_teacher_predictions":
            str(predictions_path),
        "qpseudo_classwise":
            str(classwise_path),
        "qpseudo_validation_sha256":
            sha256_file(validation_path),
        "val_last_teacher_predictions_sha256":
            sha256_file(predictions_path),
        "qpseudo_classwise_sha256":
            sha256_file(classwise_path),
    }