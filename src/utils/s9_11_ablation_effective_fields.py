"""Extract and validate effective S9.11 scientific semantics.

Purpose
-------
S9.11 is a validation-only OFAT ablation campaign anchored to the frozen
Main SSL Faster R-CNN R50-FPN 10% configuration.

This module reads the *resolved MMEngine configuration that will actually be
passed to Runner.from_cfg()* and derives the active scientific settings for:

    C family — RCNN pseudo-label classification confidence threshold
    A family — Student strong-view brightness / contrast augmentation
    F family — RCNN bbox-regression pseudo-label reliability filtering

The extractor deliberately distinguishes two evidence sources:

1. CONFIG_RESOLVED
   Values explicitly represented in the resolved MMEngine configuration.

2. IMPLEMENTATION_RUNTIME
   Scientific behaviour implemented by a registered Python class rather than
   represented as a numeric config value.

The critical example is F1.  Its bbox-regression rule is score > 0.90.
That value MUST NOT be represented as reg_pseudo_thr=0.90 because
reg_pseudo_thr is the regression-uncertainty threshold used by frozen Main.
For F1, the threshold is therefore read from the active detector class
constant at runtime.

The output is designed for run-manifest provenance and later same-seed OFAT
pair validation.  It does not authorize final-test access.
"""

from __future__ import annotations

import math
from typing import Any


MAIN_CLS_PSEUDO_THR = 0.90
MAIN_INITIAL_PSEUDO_SCORE_THR = 0.50
MAIN_RPN_PSEUDO_THR = 0.90
MAIN_REG_UNCERTAINTY_THR = 0.02

VALID_C_LEVELS = {
    "C0": 0.50,
    "C1": 0.60,
    "C2": 0.70,
    "C3": 0.80,
}

VALID_A_LEVELS = {
    "A0": (False, False),
    "A1": (True, False),
    "A2": (False, True),
}

VALID_F_LEVELS = {"F0", "F1"}

MAIN_AUGMENTATION_VARIANT = "A3"
MAIN_REGRESSION_VARIANT = "F2"


class AblationEffectiveFieldsError(RuntimeError):
    """Raised when resolved S9.11 scientific semantics violate OFAT."""


def _as_plain_mapping(cfg: Any) -> dict[str, Any]:
    """Return a plain top-level resolved-config mapping."""

    to_dict = getattr(cfg, "to_dict", None)

    if callable(to_dict):
        payload = to_dict()
    elif isinstance(cfg, dict):
        payload = cfg
    else:
        raise AblationEffectiveFieldsError(
            "RESOLVED_CONFIG_NOT_MAPPING_OR_CONFIG"
        )

    if not isinstance(payload, dict) or not payload:
        raise AblationEffectiveFieldsError(
            "RESOLVED_CONFIG_EMPTY_OR_INVALID"
        )

    return payload


def _type_name(value: Any) -> str:
    """Normalize a config ``type`` entry without changing its semantics."""

    if isinstance(value, str):
        return value

    if isinstance(value, type):
        return value.__name__

    name = getattr(value, "__name__", None)

    if isinstance(name, str) and name:
        return name

    return str(value)


def _require_mapping(
    value: Any,
    *,
    label: str,
) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise AblationEffectiveFieldsError(
            f"{label}_NOT_MAPPING"
        )
    return value


def _require_finite_float(
    value: Any,
    *,
    label: str,
) -> float:
    try:
        observed = float(value)
    except (TypeError, ValueError) as exc:
        raise AblationEffectiveFieldsError(
            f"{label}_NOT_NUMERIC"
        ) from exc

    if not math.isfinite(observed):
        raise AblationEffectiveFieldsError(
            f"{label}_NOT_FINITE"
        )

    return observed


def _same_float(
    left: float,
    right: float,
) -> bool:
    return math.isclose(
        float(left),
        float(right),
        rel_tol=0.0,
        abs_tol=1e-12,
    )


def _walk(value: Any):
    """Yield all nested config objects deterministically."""

    yield value

    if isinstance(value, dict):
        for child in value.values():
            yield from _walk(child)

    elif isinstance(value, (list, tuple)):
        for child in value:
            yield from _walk(child)


def _photometric_identity(
    payload: dict[str, Any],
) -> dict[str, Any]:
    """Resolve the active Student strong-view photometric semantics."""

    train_dataloader = _require_mapping(
        payload.get("train_dataloader"),
        label="TRAIN_DATALOADER",
    )

    candidates: list[dict[str, Any]] = []

    for node in _walk(train_dataloader):
        if not isinstance(node, dict):
            continue

        transform_type = _type_name(
            node.get("type")
        )

        if transform_type in {
            "GrayscaleBrightnessContrast",
            "S911GrayscalePhotometricAblation",
        }:
            candidates.append(node)

    if len(candidates) != 1:
        raise AblationEffectiveFieldsError(
            "EXPECTED_EXACTLY_ONE_STRONG_PHOTOMETRIC_TRANSFORM:"
            f"observed={len(candidates)}"
        )

    transform = candidates[0]
    transform_type = _type_name(
        transform.get("type")
    )

    brightness_range = transform.get(
        "brightness_range"
    )
    contrast_range = transform.get(
        "contrast_range"
    )

    if (
        not isinstance(brightness_range, (list, tuple))
        or len(brightness_range) != 2
    ):
        raise AblationEffectiveFieldsError(
            "BRIGHTNESS_RANGE_INVALID"
        )

    if (
        not isinstance(contrast_range, (list, tuple))
        or len(contrast_range) != 2
    ):
        raise AblationEffectiveFieldsError(
            "CONTRAST_RANGE_INVALID"
        )

    brightness_range = [
        _require_finite_float(
            value,
            label="BRIGHTNESS_RANGE_VALUE",
        )
        for value in brightness_range
    ]

    contrast_range = [
        _require_finite_float(
            value,
            label="CONTRAST_RANGE_VALUE",
        )
        for value in contrast_range
    ]

    if brightness_range != [0.90, 1.10]:
        raise AblationEffectiveFieldsError(
            "BRIGHTNESS_RANGE_DIFFERS_FROM_LOCKED_RANGE"
        )

    if contrast_range != [0.90, 1.10]:
        raise AblationEffectiveFieldsError(
            "CONTRAST_RANGE_DIFFERS_FROM_LOCKED_RANGE"
        )

    if transform_type == "GrayscaleBrightnessContrast":
        variant = MAIN_AUGMENTATION_VARIANT
    else:
        variant = str(
            transform.get(
                "ablation_augmentation_variant",
                "",
            )
        ).strip().upper()

        if variant not in {
            "A0",
            "A1",
            "A2",
            "A3",
        }:
            raise AblationEffectiveFieldsError(
                "ABLATION_AUGMENTATION_VARIANT_INVALID"
            )

    brightness_enabled = variant in {
        "A1",
        "A3",
    }

    contrast_enabled = variant in {
        "A2",
        "A3",
    }

    return {
        "augmentation_variant": variant,
        "augmentation_transform_type": transform_type,
        "brightness_enabled": brightness_enabled,
        "contrast_enabled": contrast_enabled,
        "brightness_range": brightness_range,
        "contrast_range": contrast_range,
        "augmentation_evidence_source": "CONFIG_RESOLVED",
    }


def _regression_identity(
    model: dict[str, Any],
    semi_train_cfg: dict[str, Any],
) -> dict[str, Any]:
    """Resolve active bbox-regression pseudo-label filtering semantics."""

    model_type = _type_name(
        model.get("type")
    )

    reg_uncertainty_threshold = (
        _require_finite_float(
            semi_train_cfg.get("reg_pseudo_thr"),
            label="REG_PSEUDO_THR",
        )
    )

    if model_type != "S911AblationRegressionFilterSoftTeacher":
        if not _same_float(
            reg_uncertainty_threshold,
            MAIN_REG_UNCERTAINTY_THR,
        ):
            raise AblationEffectiveFieldsError(
                "MAIN_REG_UNCERTAINTY_THRESHOLD_MISMATCH"
            )

        return {
            "regression_filter_variant": MAIN_REGRESSION_VARIANT,
            "regression_filter_rule": (
                "REG_UNCERTAINTY_LT_THRESHOLD"
            ),
            "regression_score_threshold": None,
            "regression_uncertainty_threshold": (
                reg_uncertainty_threshold
            ),
            "regression_evidence_source": "CONFIG_RESOLVED",
            "detector_type": model_type,
        }

    variant = str(
        model.get(
            "ablation_regression_variant",
            "",
        )
    ).strip().upper()

    if variant not in {
        "F0",
        "F1",
        "F2",
    }:
        raise AblationEffectiveFieldsError(
            "ABLATION_REGRESSION_VARIANT_INVALID"
        )

    if variant == "F0":
        return {
            "regression_filter_variant": "F0",
            "regression_filter_rule": (
                "NO_ADDITIONAL_REGRESSION_RELIABILITY_FILTER"
            ),
            "regression_score_threshold": None,
            "regression_uncertainty_threshold": None,
            "regression_evidence_source": (
                "CONFIG_RESOLVED_PLUS_IMPLEMENTATION_RUNTIME"
            ),
            "detector_type": model_type,
        }

    if variant == "F1":
        # F1's 0.90 threshold is an implementation constant, not
        # ``semi_train_cfg.reg_pseudo_thr``. Import lazily so local
        # syntax/static audits do not require MMDetection.
        try:
            from src.detectors.s9_11_ablation_regression_filter_soft_teacher import (
                S911AblationRegressionFilterSoftTeacher,
            )
        except Exception as exc:
            raise AblationEffectiveFieldsError(
                "F1_RUNTIME_DETECTOR_IMPORT_FAILED"
            ) from exc

        score_threshold = _require_finite_float(
            S911AblationRegressionFilterSoftTeacher
            .F1_REGRESSION_SCORE_THRESHOLD,
            label="F1_REGRESSION_SCORE_THRESHOLD",
        )

        if not _same_float(
            score_threshold,
            0.90,
        ):
            raise AblationEffectiveFieldsError(
                "F1_RUNTIME_SCORE_THRESHOLD_MISMATCH"
            )

        return {
            "regression_filter_variant": "F1",
            "regression_filter_rule": (
                "REGRESSION_SCORE_GT_THRESHOLD"
            ),
            "regression_score_threshold": score_threshold,
            "regression_uncertainty_threshold": None,
            "regression_evidence_source": (
                "CONFIG_RESOLVED_PLUS_IMPLEMENTATION_RUNTIME"
            ),
            "detector_type": model_type,
        }

    if not _same_float(
        reg_uncertainty_threshold,
        MAIN_REG_UNCERTAINTY_THR,
    ):
        raise AblationEffectiveFieldsError(
            "F2_REG_UNCERTAINTY_THRESHOLD_MISMATCH"
        )

    return {
        "regression_filter_variant": "F2",
        "regression_filter_rule": (
            "REG_UNCERTAINTY_LT_THRESHOLD"
        ),
        "regression_score_threshold": None,
        "regression_uncertainty_threshold": (
            reg_uncertainty_threshold
        ),
        "regression_evidence_source": (
            "CONFIG_RESOLVED_PLUS_IMPLEMENTATION_RUNTIME"
        ),
        "detector_type": model_type,
    }


def extract_s9_11_effective_scientific_fields(
    *,
    cfg: Any,
    ablation_family: str,
    ablation_config_id: str,
    ablation_level: str,
) -> dict[str, Any]:
    """Extract and OFAT-validate one resolved non-Main S9.11 config."""

    payload = _as_plain_mapping(cfg)

    family = str(
        ablation_family
    ).strip().upper()

    config_id = str(
        ablation_config_id
    ).strip().upper()

    level = str(
        ablation_level
    ).strip().upper()

    model = _require_mapping(
        payload.get("model"),
        label="MODEL",
    )

    semi_train_cfg = _require_mapping(
        model.get("semi_train_cfg"),
        label="SEMI_TRAIN_CFG",
    )

    initial_pseudo_score = _require_finite_float(
        semi_train_cfg.get(
            "pseudo_label_initial_score_thr"
        ),
        label="INITIAL_PSEUDO_SCORE",
    )

    rpn_pseudo_threshold = _require_finite_float(
        semi_train_cfg.get(
            "rpn_pseudo_thr"
        ),
        label="RPN_PSEUDO_THR",
    )

    cls_pseudo_threshold = _require_finite_float(
        semi_train_cfg.get(
            "cls_pseudo_thr"
        ),
        label="CLS_PSEUDO_THR",
    )

    if not _same_float(
        initial_pseudo_score,
        MAIN_INITIAL_PSEUDO_SCORE_THR,
    ):
        raise AblationEffectiveFieldsError(
            "INITIAL_PSEUDO_SCORE_DIFFERS_FROM_MAIN"
        )

    if not _same_float(
        rpn_pseudo_threshold,
        MAIN_RPN_PSEUDO_THR,
    ):
        raise AblationEffectiveFieldsError(
            "RPN_PSEUDO_THRESHOLD_DIFFERS_FROM_MAIN"
        )

    augmentation = _photometric_identity(
        payload
    )

    regression = _regression_identity(
        model,
        semi_train_cfg,
    )

    if family == "CONFIDENCE_THRESHOLD":
        if config_id not in VALID_C_LEVELS:
            raise AblationEffectiveFieldsError(
                "CONFIDENCE_LEVEL_INVALID"
            )

        if not _same_float(
            cls_pseudo_threshold,
            VALID_C_LEVELS[config_id],
        ):
            raise AblationEffectiveFieldsError(
                "CONFIDENCE_THRESHOLD_RESOLVED_VALUE_MISMATCH"
            )

        if (
            augmentation["augmentation_variant"]
            != MAIN_AUGMENTATION_VARIANT
        ):
            raise AblationEffectiveFieldsError(
                "CONFIDENCE_VARIANT_CHANGED_AUGMENTATION"
            )

        if (
            regression["regression_filter_variant"]
            != MAIN_REGRESSION_VARIANT
        ):
            raise AblationEffectiveFieldsError(
                "CONFIDENCE_VARIANT_CHANGED_REGRESSION_FILTER"
            )

    elif family == "STRONG_PHOTOMETRIC_AUGMENTATION":
        if config_id not in VALID_A_LEVELS:
            raise AblationEffectiveFieldsError(
                "AUGMENTATION_LEVEL_INVALID"
            )

        if not _same_float(
            cls_pseudo_threshold,
            MAIN_CLS_PSEUDO_THR,
        ):
            raise AblationEffectiveFieldsError(
                "AUGMENTATION_VARIANT_CHANGED_CONFIDENCE_THRESHOLD"
            )

        expected_brightness, expected_contrast = (
            VALID_A_LEVELS[config_id]
        )

        if (
            augmentation["augmentation_variant"]
            != config_id
        ):
            raise AblationEffectiveFieldsError(
                "AUGMENTATION_VARIANT_RESOLVED_VALUE_MISMATCH"
            )

        if (
            augmentation["brightness_enabled"]
            is not expected_brightness
        ):
            raise AblationEffectiveFieldsError(
                "AUGMENTATION_BRIGHTNESS_SEMANTICS_MISMATCH"
            )

        if (
            augmentation["contrast_enabled"]
            is not expected_contrast
        ):
            raise AblationEffectiveFieldsError(
                "AUGMENTATION_CONTRAST_SEMANTICS_MISMATCH"
            )

        if (
            regression["regression_filter_variant"]
            != MAIN_REGRESSION_VARIANT
        ):
            raise AblationEffectiveFieldsError(
                "AUGMENTATION_VARIANT_CHANGED_REGRESSION_FILTER"
            )

    elif family == "REGRESSION_PSEUDO_LABEL_FILTERING":
        if config_id not in VALID_F_LEVELS:
            raise AblationEffectiveFieldsError(
                "REGRESSION_LEVEL_INVALID"
            )

        if not _same_float(
            cls_pseudo_threshold,
            MAIN_CLS_PSEUDO_THR,
        ):
            raise AblationEffectiveFieldsError(
                "REGRESSION_VARIANT_CHANGED_CONFIDENCE_THRESHOLD"
            )

        if (
            augmentation["augmentation_variant"]
            != MAIN_AUGMENTATION_VARIANT
        ):
            raise AblationEffectiveFieldsError(
                "REGRESSION_VARIANT_CHANGED_AUGMENTATION"
            )

        if (
            regression["regression_filter_variant"]
            != config_id
        ):
            raise AblationEffectiveFieldsError(
                "REGRESSION_VARIANT_RESOLVED_VALUE_MISMATCH"
            )

    else:
        raise AblationEffectiveFieldsError(
            f"UNKNOWN_ABLATION_FAMILY:{family!r}"
        )

    return {
        "ablation_family": family,
        "ablation_level": level,
        "initial_pseudo_score": initial_pseudo_score,
        "rpn_pseudo_threshold": rpn_pseudo_threshold,
        "rcnn_classification_threshold": cls_pseudo_threshold,
        **augmentation,
        **regression,
        "ofat_effective_semantics_valid": True,
        "validation_only": True,
        "final_test_eligible": False,
    }


__all__ = [
    "AblationEffectiveFieldsError",
    "extract_s9_11_effective_scientific_fields",
]