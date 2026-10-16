"""S9.14 authorized final-test execution helpers.

This module adapts an eligible official run to the canonical test split without
modifying frozen training/evaluator sources. It contains no campaign fan-out.
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import torch
from mmengine.config import Config

from src.hooks.s9_14_final_test_retention_hook import FinalTestPredictionRetentionHook
from src.metrics.operating_point_metric import OperatingPointMetric
from src.metrics.protocol_coco_metric import ProtocolCocoMetric


TEST_ANN = Path("/workspace/ssod/data/coco/instances_test.json")
EXPECTED_RAW_TEST_COCO_SHA256 = "e1a73110e92af2656276d6c532035afe474b6ac6f2b2f03849834c036e1c00a4"
EXPECTED_TEST_MEMBERSHIP_SHA256 = "1f7903e069e872bf2e5fe13bb4d0fa257dc4a1c2c8290a621d3f7286ada66b37"
EXPECTED_EVALUATOR_SHA256 = "43079be37ab97cbbd58d0fdeaf171b9ebfe8272399d2b32800bbfbd8505d72a9"


def sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def phase2e_membership_sha256(ann_file: str | Path) -> str:
    payload = json.loads(Path(ann_file).read_text(encoding="utf-8"))
    image_ids = sorted((str(item["id"]) for item in payload["images"]))
    return hashlib.sha256("\n".join(image_ids).encode("utf-8")).hexdigest()


def verify_test_file_before_build() -> None:
    if not TEST_ANN.is_file():
        raise FileNotFoundError(TEST_ANN)
    raw = sha256(TEST_ANN)
    if raw != EXPECTED_RAW_TEST_COCO_SHA256:
        raise RuntimeError(f"Raw test COCO SHA256 mismatch: {raw}")
    membership = phase2e_membership_sha256(TEST_ANN)
    if membership != EXPECTED_TEST_MEMBERSHIP_SHA256:
        raise RuntimeError(f"Test membership SHA256 mismatch: {membership}")


def build_final_test_config(
    executable_config_path: str | Path,
    *,
    work_dir: str | Path,
    evaluator_config_sha256: str,
) -> Config:
    """Adapt the official config to the canonical test split and frozen metrics."""
    if str(evaluator_config_sha256) != EXPECTED_EVALUATOR_SHA256:
        raise RuntimeError(
            "Frozen evaluator semantic SHA256 mismatch: "
            f"{evaluator_config_sha256} != {EXPECTED_EVALUATOR_SHA256}"
        )
    cfg = Config.fromfile(str(executable_config_path))
    for key in ("COCO_ROOT", "MMDET_IMAGE_ROOT", "Path", "SSOD_ROOT", "annotations"):
        cfg.pop(key, None)
    cfg.default_scope = "mmdet"
    if "DATASETS" not in cfg or "test" not in cfg.DATASETS:
        raise RuntimeError("Official config has no canonical DATASETS['test']")

    dataset = copy.deepcopy(cfg.DATASETS["test"])
    if str(dataset.get("ann_file")) != str(TEST_ANN):
        raise RuntimeError(
            f"Canonical test ann_file mismatch: {dataset.get('ann_file')!r}"
        )
    if dataset.get("test_mode") is not True:
        raise RuntimeError("Canonical test dataset must have test_mode=True")

    cfg.test_dataloader = dict(
        batch_size=1,
        num_workers=2,
        persistent_workers=True,
        drop_last=False,
        sampler=dict(type="DefaultSampler", shuffle=False),
        dataset=dataset,
    )
    cfg.test_cfg = dict(type="TestLoop")
    cfg.test_evaluator = [
        dict(
            type="ProtocolCocoMetric",
            ann_file=str(TEST_ANN),
            metric="bbox",
            classwise=True,
            proposal_nums=(1, 10, 100),
            metric_items=["mAP", "mAP_50", "mAP_75", "AR@1000"],
            format_only=False,
            prefix="coco",
        ),
        dict(
            type="OperatingPointMetric",
            ann_file=str(TEST_ANN),
            tau_eval=0.50,
            iou_match_threshold=0.50,
            max_dets_per_image=100,
            prefix="operating",
        ),
    ]
    cfg.work_dir = str(work_dir)
    cfg.custom_hooks = [
        dict(type="FinalTestPredictionRetentionHook", max_dets_per_image=100)
    ]
    cfg.load_from = None
    cfg.resume = False
    return cfg


def _unwrap_model(model: Any) -> Any:
    return model.module if hasattr(model, "module") else model


def load_selected_weights(
    model: Any,
    checkpoint_path: str | Path,
    *,
    method: str,
    checkpoint_role: str,
) -> str:
    """Strict-load the selected scientific weights without resume semantics."""
    checkpoint_path = Path(checkpoint_path)
    payload = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(payload, Mapping) or "state_dict" not in payload:
        raise RuntimeError("Selected checkpoint has no state_dict mapping")

    target = _unwrap_model(model)
    if method == "SUP":
        if checkpoint_role != "BEST_CHECKPOINT":
            raise RuntimeError(f"SUP checkpoint role mismatch: {checkpoint_role}")
        target.load_state_dict(payload["state_dict"], strict=True)
        return "BEST"

    if method == "SSL":
        if checkpoint_role != "BEST_EMA_TEACHER_CHECKPOINT":
            raise RuntimeError(f"SSL checkpoint role mismatch: {checkpoint_role}")
        meta = payload.get("meta", {})
        expected = {
            "scientific_checkpoint_role": "BEST",
            "model_role": "EMA_TEACHER",
            "serialization_scope": "EMA_TEACHER_ONLY",
        }
        for key, value in expected.items():
            if meta.get(key) != value:
                raise RuntimeError(
                    f"SSL BEST EMA checkpoint metadata mismatch: {key}={meta.get(key)!r}"
                )
        if not hasattr(target, "teacher"):
            raise RuntimeError("SSL model has no teacher attribute")
        target.teacher.load_state_dict(payload["state_dict"], strict=True)
        return "BEST_EMA_TEACHER"

    raise RuntimeError(f"Unsupported eligible method: {method}")
