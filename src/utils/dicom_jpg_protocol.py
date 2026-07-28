#!/usr/bin/env python3
"""Phase 2D.1B-Pilot - pure, importable protocol/transformation helpers.

WHY THIS MODULE EXISTS
----------------------
The pilot orchestrator lives in ``scripts/02D1B_pilot_dicom_to_jpg.py``. That
file name begins with a digit, so it is **not importable** as a normal Python
module (``import 02D1B_...`` is a syntax error). The Phase 2D.1B guardrail
tests, however, must import the *pure* transformation and validation functions
directly and must compute their expected outputs **independently** of the
production code (self-referential synthetic tests are forbidden by the spec,
Section 33). Placing the deterministic, side-effect-free logic here lets both
the orchestrator and the tests import the same functions without triggering any
pipeline side effects, and keeps a single source of truth for the locked
transformation branches.

SCOPE / HARD CONSTRAINTS
------------------------
* This module performs NO file I/O with side effects, NO DICOM pixel decoding,
  and NO network access at import time.
* It never runs full conversion, never selects a final JPEG quality, and never
  flips a readiness flag.
* ``pydicom``/``PIL``/``skimage`` are imported lazily where needed so importing
  this module (e.g. from the test suite) never crashes on a missing optional
  dependency. Missing dependencies are reported, not silently worked around.

The transformation math implemented here operates on plain ``numpy`` arrays and
explicit metadata values, so it is fully unit-testable with hand-computed
constants.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any, Dict, List, Optional, Sequence, Tuple

try:
    import numpy as np
except Exception as exc:  # pragma: no cover - numpy is a hard requirement
    raise ImportError(
        "numpy is required for src.utils.dicom_jpg_protocol but is not importable: "
        f"{exc!r}"
    )


# =========================================================================== #
# 0. Locked constants (Phase 2D.1A / Phase 2D evidence)                        #
# =========================================================================== #
PHASE_ID = "2D.1B-Pilot"
EXPECTED_PROTOCOL_VERSION = "1.0.0"
EXPECTED_PROTOCOL_SHA256 = (
    "1528da27758d35786847141c37d0ddb754dddb146aff116a8f3a9a7b07221229"
)
# SHA-256 of data/processed/coco/coco_master.json recorded by Phase 2D
# (reports/phase2D_coco_master_validation.json -> output_sha256).
EXPECTED_COCO_MASTER_SHA256 = (
    "36f09d1b1477ea4a63153a04d775c938752e224c26079a0d44881c14b9bb4d75"
)
QUALITY_CANDIDATES: Tuple[int, ...] = (95, 100)
TIE_BREAK_SEED = 2026

LOCKED_INPUT_COUNTS: Dict[str, int] = {
    "images": 4894,
    "abnormal_images": 4394,
    "no_finding_images": 500,
    "annotations": 36096,
    "categories": 14,
    "no_finding_annotations": 0,
}

MIN_PILOT_IMAGES = 64
MIN_PILOT_NO_FINDING = 16
MAX_PILOT_IMAGES = 256

ALLOWED_PHOTOMETRIC = ("MONOCHROME1", "MONOCHROME2")

# Number of abnormal detection classes (canonical 0..13 / COCO 1..14).
NUM_ABNORMAL_CLASSES = 14


# =========================================================================== #
# 1. Custom exceptions - every hard-fail path has a named, catchable error     #
# =========================================================================== #
class Phase2D1BError(Exception):
    """Base class for all Phase 2D.1B pilot errors."""


class ProtocolSchemaMismatch(Phase2D1BError):
    """A required nested YAML field path is absent (do not repair/guess)."""


class ProtocolDriftError(Phase2D1BError):
    """Protocol version or fingerprint does not match locked evidence."""


class CocoMasterDriftError(Phase2D1BError):
    """coco_master.json hash differs from locked Phase 2D evidence."""


class ProtocolGapError(Phase2D1BError):
    """Locked protocol does not resolve a case (BLOCKED_PROTOCOL_REVIEW).

    Carries a machine-readable ``reason`` so the orchestrator can emit the
    correct validation-JSON status without inventing a fallback.
    """

    def __init__(self, message: str, reason: str) -> None:
        super().__init__(message)
        self.reason = reason


class UnsupportedInputError(Phase2D1BError):
    """A DICOM header violates a hard structural constraint (hard fail)."""


class PaddingMetadataError(Phase2D1BError):
    """Ambiguous pixel-padding metadata (range limit without value)."""


class ModalityBranchError(Phase2D1BError):
    """Exactly-one modality branch could not be selected."""


class DegenerateRangeError(Phase2D1BError):
    """Theoretical output low == high (cannot linearly map)."""


class NonFiniteError(Phase2D1BError):
    """NaN/Inf encountered during transformation (hard fail)."""


class NonDeterministicEncodingError(Phase2D1BError):
    """Two identical encodes produced different bytes."""


class AccidentalFullConversionError(Phase2D1BError):
    """Pixel decode count reached the full controlled scope (4894)."""


class PilotScopeExplosionError(Phase2D1BError):
    """Deterministic selection exceeded the max_pilot_images guardrail."""


class PreexistingForbiddenArtifactError(Phase2D1BError):
    """A forbidden full-conversion artifact already exists (never delete)."""


# =========================================================================== #
# 2. Canonical fingerprint (identical rule to Phase 2D.1A)                      #
# =========================================================================== #
def canonical_json(obj: Any) -> str:
    """Deterministic JSON serialization used for the protocol fingerprint."""
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def protocol_sha256(protocol_dict: Dict[str, Any]) -> str:
    """SHA-256 of the canonical-JSON form of the *parsed* protocol dict."""
    return hashlib.sha256(canonical_json(protocol_dict).encode("utf-8")).hexdigest()


# =========================================================================== #
# 3. Locked nested YAML field-path map (Section 8.0 correction)                #
# --------------------------------------------------------------------------- #
# The validator MUST read every locked value from its actual nested path and    #
# MUST NOT silently search for a similarly named field elsewhere. This map is   #
# declared centrally and treated as immutable.                                  #
# =========================================================================== #
FIELD_PATHS: Dict[str, Tuple[str, ...]] = {
    "protocol_version": ("protocol_metadata", "protocol_version"),
    "quality_candidates": ("jpeg_encoding", "quality_candidates"),
    "final_quality": ("jpeg_encoding", "final_quality"),
    "final_quality_status": ("jpeg_encoding", "final_quality_status"),
    "lossless_claim": ("jpeg_encoding", "lossless_claim"),
    "resize": ("geometry_bbox_policy", "resize"),
    "crop": ("geometry_bbox_policy", "crop"),
    "rotation": ("geometry_bbox_policy", "rotation"),
    "flip": ("geometry_bbox_policy", "flip"),
    "transpose": ("geometry_bbox_policy", "transpose"),
    "bbox_scaling_expected": ("geometry_bbox_policy", "bbox_scaling_expected"),
    "direct_observed_per_image_min_max": (
        "voi_windowing_policy",
        "direct_observed_per_image_min_max",
    ),
    "automatic_percentile_clipping": (
        "voi_windowing_policy",
        "automatic_percentile_clipping",
    ),
    "final_quality_must_remain_null": (
        "final_quality_decision_rule",
        "final_quality_must_remain_null_in_this_phase",
    ),
    "locked_input_counts": ("locked_input_counts",),
    "readiness_flags": ("readiness_flags",),
    "forbidden_actions": ("forbidden_actions",),
    "jpeg_mode": ("output_channel_policy", "jpg_storage", "jpeg_mode"),
    "jpeg_channels": ("output_channel_policy", "jpg_storage", "channels"),
}

_MISSING = object()


def resolve_field(protocol_dict: Dict[str, Any], logical_name: str) -> Any:
    """Strictly resolve a locked value by its nested path.

    Raises :class:`ProtocolSchemaMismatch` if the exact nested path is absent.
    Never falls back to a similarly named key elsewhere in the document.
    """
    if logical_name not in FIELD_PATHS:
        raise ProtocolSchemaMismatch(f"Unknown locked field: {logical_name!r}")
    node: Any = protocol_dict
    path = FIELD_PATHS[logical_name]
    for key in path:
        if not isinstance(node, dict) or key not in node:
            dotted = ".".join(path)
            raise ProtocolSchemaMismatch(
                f"protocol_schema_mismatch: expected nested path '{dotted}' "
                f"is missing (stopped at segment '{key}')"
            )
        node = node[key]
    return node


def resolved_field_path(logical_name: str) -> str:
    """Human-readable dotted path recorded in evidence for a locked value."""
    return ".".join(FIELD_PATHS[logical_name])


def validate_protocol(protocol_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Validate the locked protocol using ACTUAL nested paths + fingerprint.

    Returns an evidence dict of every locked value together with the exact
    resolved field path used. Raises on drift / schema mismatch. Does not
    mutate the protocol.
    """
    evidence: Dict[str, Any] = {"resolved_paths": {}, "values": {}}

    def record(name: str) -> Any:
        value = resolve_field(protocol_dict, name)
        evidence["resolved_paths"][name] = resolved_field_path(name)
        evidence["values"][name] = value
        return value

    version = record("protocol_version")
    if version != EXPECTED_PROTOCOL_VERSION:
        raise ProtocolDriftError(
            f"protocol_drift_detected: protocol_metadata.protocol_version="
            f"{version!r} != {EXPECTED_PROTOCOL_VERSION!r}"
        )

    candidates = record("quality_candidates")
    if list(candidates) != list(QUALITY_CANDIDATES):
        raise ProtocolDriftError(
            f"protocol_drift_detected: quality_candidates={candidates!r} "
            f"!= {list(QUALITY_CANDIDATES)!r}"
        )

    final_quality = record("final_quality")
    if final_quality is not None:
        raise ProtocolDriftError(
            f"protocol_drift_detected: final_quality must be null, got "
            f"{final_quality!r}"
        )

    fq_status = record("final_quality_status")
    if fq_status != "pending_phase2D1B_pilot":
        raise ProtocolDriftError(
            f"protocol_drift_detected: final_quality_status={fq_status!r}"
        )

    for geom in ("resize", "crop", "rotation", "flip", "transpose",
                 "bbox_scaling_expected"):
        val = record(geom)
        if val is not False:
            raise ProtocolDriftError(
                f"protocol_drift_detected: geometry_bbox_policy.{geom}={val!r} "
                "(expected false)"
            )

    if record("direct_observed_per_image_min_max") != "forbidden":
        raise ProtocolDriftError(
            "protocol_drift_detected: direct observed per-image min/max must be "
            "forbidden"
        )
    if record("automatic_percentile_clipping") != "forbidden":
        raise ProtocolDriftError(
            "protocol_drift_detected: automatic percentile clipping must be "
            "forbidden"
        )

    if record("final_quality_must_remain_null") is not True:
        raise ProtocolDriftError(
            "protocol_drift_detected: final_quality_must_remain_null_in_this_phase "
            "!= true"
        )

    counts = record("locked_input_counts")
    for k, v in LOCKED_INPUT_COUNTS.items():
        if counts.get(k) != v:
            raise ProtocolDriftError(
                f"protocol_drift_detected: locked_input_counts.{k}="
                f"{counts.get(k)!r} != {v!r}"
            )

    readiness = record("readiness_flags")
    for k, v in readiness.items():
        if v is not False:
            raise ProtocolDriftError(
                f"protocol_drift_detected: readiness_flags.{k}={v!r} (must be false)"
            )

    forbidden = record("forbidden_actions")
    for k, v in forbidden.items():
        if v is not False:
            raise ProtocolDriftError(
                f"protocol_drift_detected: forbidden_actions.{k}={v!r} (must be false)"
            )

    if record("jpeg_mode") != "L":
        raise ProtocolDriftError("protocol_drift_detected: jpg_storage.jpeg_mode != L")
    if record("jpeg_channels") != 1:
        raise ProtocolDriftError("protocol_drift_detected: jpg_storage.channels != 1")

    fingerprint = protocol_sha256(protocol_dict)
    evidence["protocol_sha256"] = fingerprint
    evidence["protocol_sha256_expected"] = EXPECTED_PROTOCOL_SHA256
    evidence["protocol_sha256_match"] = fingerprint == EXPECTED_PROTOCOL_SHA256
    if fingerprint != EXPECTED_PROTOCOL_SHA256:
        raise ProtocolDriftError(
            f"protocol_drift_detected: fingerprint {fingerprint} != "
            f"{EXPECTED_PROTOCOL_SHA256}"
        )
    return evidence


# =========================================================================== #
# 4. DICOM-root CLI/environment resolution table (Section 3 correction)        #
# =========================================================================== #
ENV_VAR_NAME = "VINBIGDATA_DICOM_ROOT"


@dataclass(frozen=True)
class RootResolution:
    root: Path
    source: str          # "cli" | "env" | "cli_and_env_equal"
    cli_value: Optional[str]
    env_value: Optional[str]
    cli_resolved: Optional[str]
    env_resolved: Optional[str]


def _normalize_resolved(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    try:
        return os.path.normcase(str(Path(value).expanduser().resolve(strict=False)))
    except Exception:
        return os.path.normcase(os.path.normpath(str(value)))


def resolve_dicom_root(
    cli_value: Optional[str],
    env_value: Optional[str],
) -> RootResolution:
    """Resolve the DICOM root using the locked precedence table.

    ================================  ==================================
    CLI absent  + ENV present         use ENV
    CLI present + ENV absent          use CLI
    CLI present + ENV present + same  use that path
    CLI present + ENV present + diff  hard fail
    CLI absent  + ENV absent          hard fail
    ================================  ==================================

    Path equivalence is evaluated after OS-aware normalize + resolve. We do NOT
    treat CLI as an unconditional override; a genuine conflict is a hard fail so
    the wrong dataset can never be run silently.
    """
    cli_present = cli_value is not None and str(cli_value).strip() != ""
    env_present = env_value is not None and str(env_value).strip() != ""
    cli_res = _normalize_resolved(cli_value) if cli_present else None
    env_res = _normalize_resolved(env_value) if env_present else None

    if not cli_present and not env_present:
        raise UnsupportedInputError(
            "dicom_root_unresolved: neither --dicom-root nor "
            f"{ENV_VAR_NAME} is set"
        )
    if cli_present and not env_present:
        return RootResolution(Path(cli_value), "cli", cli_value, None, cli_res, None)
    if env_present and not cli_present:
        return RootResolution(Path(env_value), "env", None, env_value, None, env_res)
    # both present
    if cli_res == env_res:
        return RootResolution(
            Path(cli_value), "cli_and_env_equal", cli_value, env_value, cli_res, env_res
        )
    raise UnsupportedInputError(
        "dicom_root_conflict: --dicom-root and "
        f"{ENV_VAR_NAME} resolve to different paths "
        f"({cli_res!r} != {env_res!r}); refusing to guess the dataset"
    )


def safe_resolve_under_root(dicom_root: Path, coco_file_name: str) -> Path:
    """Join a relative COCO file_name under the root and reject traversal.

    ``coco_file_name`` is expected to be POSIX-relative like
    ``train/<id>.dicom``. Absolute names and ``..`` traversal are rejected.
    """
    pure = PurePosixPath(coco_file_name)
    if pure.is_absolute() or Path(coco_file_name).is_absolute():
        raise UnsupportedInputError(f"absolute file_name rejected: {coco_file_name!r}")
    if any(part == ".." for part in pure.parts):
        raise UnsupportedInputError(f"path traversal rejected: {coco_file_name!r}")
    root_res = dicom_root.expanduser().resolve(strict=False)
    candidate = (root_res / Path(*pure.parts)).resolve(strict=False)
    try:
        candidate.relative_to(root_res)
    except ValueError:
        raise UnsupportedInputError(
            f"resolved path escapes dicom_root: {candidate} not under {root_res}"
        )
    return candidate


# =========================================================================== #
# 5. Deterministic tie-break rank (Section 15)                                 #
# =========================================================================== #
def tie_break_rank(image_id: str, seed: int = TIE_BREAK_SEED) -> str:
    """Stable SHA-256 tie-break rank. Never uses Python's builtin hash()."""
    return hashlib.sha256(f"{seed}|{image_id}".encode("utf-8")).hexdigest()


# =========================================================================== #
# 6. Theoretical stored / modality range                                       #
# =========================================================================== #
def theoretical_stored_range(bits_stored: int, pixel_representation: int) -> Tuple[int, int]:
    """Theoretical stored-value range from BitsStored + PixelRepresentation."""
    if bits_stored <= 0:
        raise UnsupportedInputError(f"invalid BitsStored={bits_stored}")
    if pixel_representation == 0:
        return 0, (1 << bits_stored) - 1
    if pixel_representation == 1:
        return -(1 << (bits_stored - 1)), (1 << (bits_stored - 1)) - 1
    raise UnsupportedInputError(f"invalid PixelRepresentation={pixel_representation}")


@dataclass
class ModalityResult:
    branch: str                     # "modality_lut" | "rescale" | "identity"
    values: "np.ndarray"
    theoretical_low: float
    theoretical_high: float
    rescale_state: str              # "applied" | "present_not_applied" | "absent"


def modality_branch_name(
    modality_lut_present: bool,
    rescale_slope_present: bool,
    rescale_intercept_present: bool,
) -> str:
    """Select exactly one modality branch (Section 17.3). Never sequential."""
    if modality_lut_present:
        return "modality_lut"
    slope = bool(rescale_slope_present)
    inter = bool(rescale_intercept_present)
    if slope and inter:
        return "rescale"
    if not slope and not inter:
        return "identity"
    # Exactly one of slope/intercept present -> incomplete/invalid.
    raise ModalityBranchError(
        "rescale_incomplete_invalid: exactly one of RescaleSlope/RescaleIntercept "
        "present"
    )


def apply_rescale(
    stored: "np.ndarray",
    slope: float,
    intercept: float,
    theoretical_stored_low: int,
    theoretical_stored_high: int,
) -> Tuple["np.ndarray", float, float]:
    """Apply rescale to pixels and derive sorted theoretical modality bounds.

    The two theoretical stored endpoints are transformed and re-sorted so a
    negative slope is handled correctly.
    """
    values = stored.astype(np.float64) * float(slope) + float(intercept)
    e0 = theoretical_stored_low * float(slope) + float(intercept)
    e1 = theoretical_stored_high * float(slope) + float(intercept)
    low, high = (e0, e1) if e0 <= e1 else (e1, e0)
    return values, float(low), float(high)


# =========================================================================== #
# 7. VOI / windowing (Section 17.4). Output is a normalized [0, 1] fraction.    #
# --------------------------------------------------------------------------- #
# We map the VOI stage to a normalized fraction where 0 == darkest and 1 ==     #
# brightest in MONOCHROME2 convention. Presentation polarity then optionally    #
# inverts once (1 - frac), and uint8 conversion maps [0,1] -> [0,255]. This     #
# keeps the theoretical output bounds explicit (0.0, 1.0) and never uses        #
# observed array min/max or percentiles.                                        #
# =========================================================================== #
def window_linear(values: "np.ndarray", center: float, width: float) -> "np.ndarray":
    """DICOM VOI LUT 'LINEAR' mapped to [0, 1] (ymin=0, ymax=1).

    Per DICOM PS3.3 C.11.2.1.2, the minimum valid LINEAR WindowWidth is 1.
    * width < 1  -> invalid (block).
    * width == 1 -> threshold at Window Center:
        x <= center - 0.5 -> 0 ; x > center - 0.5 -> 1.
    * width > 1  -> standard linear ramp.
    """
    width = float(width)
    if width < 1:
        raise ProtocolGapError(
            f"invalid WindowWidth={width} for LINEAR (must be >= 1)",
            reason="invalid_window_width",
        )
    if width == 1:
        threshold = float(center) - 0.5
        return np.where(values.astype(np.float64) > threshold, 1.0, 0.0)
    c = float(center) - 0.5
    w = width - 1.0
    frac = (values.astype(np.float64) - c) / w + 0.5
    return np.clip(frac, 0.0, 1.0)


def validate_window_width_for_function(width: float,
                                       voi_lut_function: Optional[str]) -> None:
    """Header-level window-width validity by VOILUTFunction (before decode).

    LINEAR requires width >= 1 (width == 1 is a valid threshold). LINEAR_EXACT
    and SIGMOID only require width > 0. Any violation blocks for protocol review.
    """
    func = (voi_lut_function or "LINEAR").upper()
    w = float(width)
    if func == "LINEAR":
        if w < 1:
            raise ProtocolGapError(
                f"window_width_lt_1_for_linear:{w}", reason="invalid_window_width")
    elif func in ("LINEAR_EXACT", "SIGMOID"):
        if w <= 0:
            raise ProtocolGapError(
                f"window_width_le_0_for_{func.lower()}:{w}",
                reason="invalid_window_width")
    else:
        raise ProtocolGapError(
            f"unsupported_voi_lut_function:{voi_lut_function!r}",
            reason="unsupported_voi_lut_function")


def window_linear_exact(values: "np.ndarray", center: float, width: float) -> "np.ndarray":
    """DICOM VOI LUT 'LINEAR_EXACT' mapped to [0, 1]."""
    if width <= 0:
        raise ProtocolGapError(
            f"invalid WindowWidth={width} for LINEAR_EXACT",
            reason="invalid_window_width",
        )
    frac = (values.astype(np.float64) - float(center)) / float(width) + 0.5
    return np.clip(frac, 0.0, 1.0)


def window_sigmoid(values: "np.ndarray", center: float, width: float) -> "np.ndarray":
    """DICOM VOI LUT 'SIGMOID' mapped to (0, 1)."""
    if width <= 0:
        raise ProtocolGapError(
            f"invalid WindowWidth={width} for SIGMOID", reason="invalid_window_width"
        )
    z = -4.0 * (values.astype(np.float64) - float(center)) / float(width)
    return 1.0 / (1.0 + np.exp(z))


def apply_windowing(
    values: "np.ndarray",
    center: float,
    width: float,
    voi_lut_function: Optional[str],
) -> "np.ndarray":
    """Dispatch on VOILUTFunction. Default is LINEAR when absent."""
    func = (voi_lut_function or "LINEAR").upper()
    if func == "LINEAR":
        return window_linear(values, center, width)
    if func == "LINEAR_EXACT":
        return window_linear_exact(values, center, width)
    if func == "SIGMOID":
        return window_sigmoid(values, center, width)
    raise ProtocolGapError(
        f"unsupported VOILUTFunction={voi_lut_function!r}",
        reason="unsupported_voi_lut_function",
    )


def fallback_modality_fraction(
    values: "np.ndarray", low: float, high: float
) -> "np.ndarray":
    """Theoretical modality-domain fallback -> [0, 1] linear map."""
    if high == low:
        raise DegenerateRangeError(
            f"degenerate_theoretical_range: modality low==high=={low}"
        )
    frac = (values.astype(np.float64) - low) / (high - low)
    return np.clip(frac, 0.0, 1.0)


# =========================================================================== #
# 8. Presentation polarity (Section 17.5) - LOCKED ACTION TABLE + conflicts     #
# =========================================================================== #
@dataclass
class PresentationDecision:
    action: str                     # "invert_once" | "no_inversion" | "conflict" | "gap"
    inversion_count: Optional[int]  # 0, 1, or None for conflict/gap
    metadata_conflict: bool
    protocol_gap: bool
    reason: Optional[str]


def presentation_polarity_decision(
    photometric_interpretation: str,
    presentation_lut_shape: Optional[str],
    presentation_lut_sequence_present: bool = False,
) -> PresentationDecision:
    """Resolve polarity using the corrected, locked action table.

    ================================================================
    Photometric  | PresentationLUTShape | project action
    MONOCHROME1  | absent               | invert once
    MONOCHROME2  | absent               | no inversion
    MONOCHROME1  | INVERSE              | invert once
    MONOCHROME2  | IDENTITY             | no inversion
    MONOCHROME1  | IDENTITY             | metadata presentation conflict
    MONOCHROME2  | INVERSE              | metadata presentation conflict
    ================================================================

    A ``PresentationLUTSequence`` (as opposed to the scalar
    ``PresentationLUTShape``) is not defined by the locked protocol and is a
    protocol gap: BLOCKED_PROTOCOL_REVIEW. The two conflict rows are likewise
    blocked. We never infer precedence and never pick polarity from appearance.
    """
    if presentation_lut_sequence_present:
        return PresentationDecision(
            "gap", None, False, True,
            "presentation_lut_sequence_present_not_defined_by_protocol",
        )

    pi = (photometric_interpretation or "").upper()
    if pi not in ALLOWED_PHOTOMETRIC:
        raise UnsupportedInputError(
            f"unsupported PhotometricInterpretation={photometric_interpretation!r}"
        )

    shape = presentation_lut_shape
    shape_norm = shape.upper() if isinstance(shape, str) and shape.strip() else "ABSENT"
    if shape_norm not in ("ABSENT", "IDENTITY", "INVERSE"):
        return PresentationDecision(
            "gap", None, False, True,
            f"unsupported_presentation_lut_shape={presentation_lut_shape!r}",
        )

    table = {
        ("MONOCHROME1", "ABSENT"): ("invert_once", 1, False, False),
        ("MONOCHROME2", "ABSENT"): ("no_inversion", 0, False, False),
        ("MONOCHROME1", "INVERSE"): ("invert_once", 1, False, False),
        ("MONOCHROME2", "IDENTITY"): ("no_inversion", 0, False, False),
        ("MONOCHROME1", "IDENTITY"): ("conflict", None, True, True),
        ("MONOCHROME2", "INVERSE"): ("conflict", None, True, True),
    }
    action, inv, conflict, gap = table[(pi, shape_norm)]
    reason = None
    if conflict:
        reason = f"presentation_metadata_conflict:{pi}+{shape_norm}"
    return PresentationDecision(action, inv, conflict, gap, reason)


def apply_presentation(fraction: "np.ndarray", inversion_count: int) -> "np.ndarray":
    """Apply polarity in the normalized [0,1] domain. Invert at most once."""
    if inversion_count not in (0, 1):
        raise ValueError(f"inversion_count must be 0 or 1, got {inversion_count}")
    if inversion_count == 1:
        return 1.0 - fraction
    return fraction


# =========================================================================== #
# 9. Pixel padding (Section 17.2)                                              #
# =========================================================================== #
def build_padding_mask(
    stored: "np.ndarray",
    padding_value: Optional[int],
    padding_range_limit: Optional[int],
) -> "np.ndarray":
    """Padding mask on STORED pixels, before modality transformation."""
    if padding_range_limit is not None and padding_value is None:
        raise PaddingMetadataError(
            "ambiguous_padding_metadata: PixelPaddingRangeLimit present without "
            "PixelPaddingValue"
        )
    if padding_value is None:
        return np.zeros(stored.shape, dtype=bool)
    if padding_range_limit is None:
        return stored == padding_value
    low = min(int(padding_value), int(padding_range_limit))
    high = max(int(padding_value), int(padding_range_limit))
    return (stored >= low) & (stored <= high)


# =========================================================================== #
# 10. Deterministic uint8 conversion (Section 17.6)                            #
# =========================================================================== #
def fraction_to_uint8(
    fraction: "np.ndarray",
    padding_mask: Optional["np.ndarray"] = None,
) -> "np.ndarray":
    """Map a normalized [0,1] fraction to uint8 and reapply padding as 0.

    Order: (fraction already clipped) -> *255 -> rint -> clip[0,255] -> uint8
    -> padding=0.
    """
    if not np.all(np.isfinite(fraction)):
        raise NonFiniteError("nan_or_inf detected before uint8 conversion")
    scaled = np.clip(fraction, 0.0, 1.0) * 255.0
    rounded = np.rint(scaled)
    clipped = np.clip(rounded, 0, 255)
    out = clipped.astype(np.uint8)
    if padding_mask is not None:
        out = out.copy()
        out[padding_mask] = 0
    return out


def pre_jpeg_sha256(uint8_image: "np.ndarray") -> str:
    """Content hash of the pre-JPEG uint8 array (shape recorded separately)."""
    return hashlib.sha256(
        np.ascontiguousarray(uint8_image).tobytes(order="C")
    ).hexdigest()


# =========================================================================== #
# 11. Fidelity metrics (Sections 19-20). Reference = pre-JPEG uint8.            #
# =========================================================================== #
def whole_image_error_metrics(
    reference_uint8: "np.ndarray", target_uint8: "np.ndarray"
) -> Dict[str, Any]:
    """MAE/RMSE/PSNR + max/p95/p99 absolute error. PSNR None when RMSE==0."""
    if reference_uint8.shape != target_uint8.shape:
        raise UnsupportedInputError("metric shape mismatch")
    reference = reference_uint8.astype(np.float64)
    target = target_uint8.astype(np.float64)
    error = target - reference
    abs_err = np.abs(error)
    mae = float(np.mean(abs_err))
    rmse = float(np.sqrt(np.mean(error ** 2)))
    if rmse == 0.0:
        psnr: Optional[float] = None
        psnr_inf = True
    else:
        psnr = float(20.0 * math.log10(255.0) - 10.0 * math.log10(rmse ** 2))
        psnr_inf = False
    return {
        "mae": mae,
        "rmse": rmse,
        "psnr_db": psnr,
        "psnr_is_infinite": psnr_inf,
        "max_absolute_error": float(np.max(abs_err)) if abs_err.size else 0.0,
        # 'linear' interpolation is stated and used stably.
        "p95_absolute_error": float(np.percentile(abs_err, 95, method="linear"))
        if abs_err.size else 0.0,
        "p99_absolute_error": float(np.percentile(abs_err, 99, method="linear"))
        if abs_err.size else 0.0,
        "percentile_method": "linear",
    }


def roi_extraction_coords(
    x_min: float, y_min: float, x_max: float, y_max: float
) -> Tuple[int, int, int, int]:
    """Integer ROI slice coords (floor/ceil). Does NOT modify canonical bbox."""
    x0 = int(math.floor(x_min))
    y0 = int(math.floor(y_min))
    x1 = int(math.ceil(x_max))
    y1 = int(math.ceil(y_max))
    return x0, y0, x1, y1


def largest_odd_win_size(roi_height: int, roi_width: int) -> Optional[int]:
    """Largest valid odd SSIM window (>=3, <= min side). None if impossible."""
    m = min(int(roi_height), int(roi_width))
    if m < 3:
        return None
    win = m if m % 2 == 1 else m - 1
    return win if win >= 3 else None


# =========================================================================== #
# 12. Coverage feature keys for deterministic selection (Sections 13-15)        #
# =========================================================================== #
def metadata_stratum_keys(header: Dict[str, Any]) -> List[str]:
    """Categorical + pattern strata a single image contributes to.

    Continuous values (WindowCenter/Width, RescaleSlope/Intercept) are NEVER
    turned into per-value strata; only presence/pattern buckets are used.
    """
    keys: List[str] = []

    def add(kind: str, value: Any) -> None:
        keys.append(f"{kind}={value}")

    for cat in (
        "SOPClassUID", "Modality", "PhotometricInterpretation", "TransferSyntaxUID",
        "BitsAllocated", "BitsStored", "HighBit", "PixelRepresentation",
        "SamplesPerPixel", "NumberOfFrames_effective", "VOILUTFunction",
        "PresentationLUTShape",
    ):
        add(cat, header.get(cat, "ABSENT"))

    add("PresentationLUTSequence_present",
        bool(header.get("presentation_lut_sequence_present", False)))
    add("modality_lut", "present" if header.get("modality_lut_present") else "absent")
    add("voi_lut", "present" if header.get("voi_lut_present") else "absent")

    slope_p = bool(header.get("rescale_slope_present"))
    inter_p = bool(header.get("rescale_intercept_present"))
    slope = header.get("RescaleSlope")
    inter = header.get("RescaleIntercept")
    if not slope_p and not inter_p:
        add("rescale", "absent")
    elif slope_p and inter_p:
        try:
            if float(slope) == 1.0 and float(inter) == 0.0:
                add("rescale", "identity")
            else:
                add("rescale", "non_identity")
        except (TypeError, ValueError):
            add("rescale", "incomplete_invalid")
    else:
        add("rescale", "incomplete_invalid")

    wc = int(header.get("window_center_count", 0) or 0)
    ww = int(header.get("window_width_count", 0) or 0)
    if wc == 0 and ww == 0:
        add("window", "absent")
    elif wc == ww and wc == 1:
        add("window", "single_valid")
    elif wc == ww and wc > 1:
        add("window", "multi_valid")
    else:
        add("window", "incomplete_or_invalid")

    pv = bool(header.get("pixel_padding_value_present"))
    pr = bool(header.get("pixel_padding_range_present"))
    if not pv and not pr:
        add("padding", "absent")
    elif pv and not pr:
        add("padding", "single_value")
    elif pv and pr:
        add("padding", "range")
    else:
        add("padding", "range_limit_without_value")  # unsupported -> errors

    return keys


# =========================================================================== #
# 13. Window classification (Section 17.4 / blocker 7)                         #
# --------------------------------------------------------------------------- #
# A window that is PRESENT but incomplete / cardinality-mismatched /            #
# non-numeric / invalid MUST block (protocol gap). Only a genuinely ABSENT      #
# window (with no VOI LUT) is allowed to use the theoretical fallback.          #
# =========================================================================== #
@dataclass
class WindowDecision:
    state: str                       # "absent" | "valid" | "invalid"
    center: Optional[float]
    width: Optional[float]
    reason: str
    centers: List[float]
    widths: List[float]


def _split_multivalue(raw: Any) -> List[str]:
    if raw is None:
        return []
    if isinstance(raw, (list, tuple)):
        items = list(raw)
    else:
        items = [p for p in str(raw).split(";")]
    return [str(x).strip() for x in items if str(x).strip() != ""]


def classify_window(window_center: Any, window_width: Any) -> WindowDecision:
    """Classify WindowCenter/WindowWidth. Never silently drops an invalid window."""
    centers_raw = _split_multivalue(window_center)
    widths_raw = _split_multivalue(window_width)
    if not centers_raw and not widths_raw:
        return WindowDecision("absent", None, None, "window_absent", [], [])
    if len(centers_raw) != len(widths_raw):
        return WindowDecision(
            "invalid", None, None, "cardinality_mismatch", [], []
        )
    try:
        cvals = [float(x) for x in centers_raw]
        wvals = [float(x) for x in widths_raw]
    except (TypeError, ValueError):
        return WindowDecision("invalid", None, None, "non_numeric", [], [])
    if any(w <= 0 for w in wvals):
        return WindowDecision("invalid", None, None, "invalid_width", cvals, wvals)
    return WindowDecision("valid", cvals[0], wvals[0], "valid_index0", cvals, wvals)


def require_valid_window(decision: WindowDecision) -> Tuple[float, float]:
    """Return (center, width) for a valid window, else block via ProtocolGapError."""
    if decision.state != "valid":
        raise ProtocolGapError(
            f"window_{decision.reason}", reason=f"window_{decision.reason}"
        )
    return float(decision.center), float(decision.width)


# =========================================================================== #
# 14. Modality LUT output bounds from LUT DATA (Section 17.3 / blocker 8)       #
# --------------------------------------------------------------------------- #
# The theoretical modality-domain output range for a Modality LUT is defined by #
# the LUT *data* (the mapped output values), NOT the LUT descriptor's input     #
# index range and NOT the transformed image array's observed min/max.           #
# =========================================================================== #
def modality_lut_output_bounds(lut_data: Sequence[Any]) -> Tuple[float, float]:
    arr = np.asarray(list(lut_data), dtype=np.float64)
    if arr.size == 0:
        raise ModalityBranchError("empty_modality_lut_data")
    low = float(arr.min())
    high = float(arr.max())
    if low == high:
        raise DegenerateRangeError("degenerate_modality_lut_output_range")
    return low, high


# =========================================================================== #
# 15. Presentation gap enforcement (blocker 6)                                 #
# --------------------------------------------------------------------------- #
# Any PresentationDecision with protocol_gap (conflict / LUT sequence /         #
# unsupported shape) or inversion_count is None MUST raise. We never coerce      #
# ``None`` into ``inversion=0``.                                                 #
# =========================================================================== #
def require_inversion_count(decision: PresentationDecision) -> int:
    if decision.protocol_gap or decision.inversion_count is None:
        raise ProtocolGapError(
            f"presentation_gap:{decision.reason}",
            reason=decision.reason or "presentation_gap",
        )
    return int(decision.inversion_count)


# =========================================================================== #
# 16. JPEG2000 decoder-backend enforcement (Section 11 / blocker 5)            #
# --------------------------------------------------------------------------- #
# The requested backend must be available BEFORE any pixel decoding. If a       #
# JPEG2000-compressed image is in the pilot but the backend is unavailable, we  #
# hard fail rather than let pydicom silently fall back to another handler.       #
# =========================================================================== #
JPEG2000_TRANSFER_SYNTAXES = {
    "1.2.840.10008.1.2.4.90",   # JPEG 2000 Image Compression (Lossless Only)
    "1.2.840.10008.1.2.4.91",   # JPEG 2000 Image Compression
}
_JPEG2000_BACKEND_MODULES: Dict[str, Tuple[str, ...]] = {
    "pylibjpeg": ("pylibjpeg", "openjpeg"),   # pylibjpeg + pylibjpeg-openjpeg
    "gdcm": ("gdcm",),
    "pillow": ("PIL",),
}


# Mapping from our backend name to the pydicom decoding-plugin identifier.
PYDICOM_DECODING_PLUGIN: Dict[str, str] = {
    "pylibjpeg": "pylibjpeg",
    "gdcm": "gdcm",
    "pillow": "pillow",
}


def is_jpeg2000(transfer_syntax_uid: str) -> bool:
    return str(transfer_syntax_uid) in JPEG2000_TRANSFER_SYNTAXES


def pillow_jpeg2000_capable() -> bool:
    """Real Pillow JPEG2000 capability check (not merely 'import PIL')."""
    try:
        from PIL import features
        return bool(features.check("jpg_2000"))
    except Exception:
        return False


def jpeg2000_backend_available(name: str) -> bool:
    """Availability of a JPEG2000 backend.

    For 'pillow' this verifies the ACTUAL JPEG2000 codec capability, not just
    that ``PIL`` imports. For 'pylibjpeg' both pylibjpeg and the openjpeg
    plugin must import. For 'gdcm' the gdcm module must import.
    """
    import importlib

    if name == "pillow":
        return pillow_jpeg2000_capable()
    mods = _JPEG2000_BACKEND_MODULES.get(name)
    if not mods:
        return False
    for m in mods:
        try:
            importlib.import_module(m)
        except Exception:
            return False
    return True


def ensure_jpeg2000_backend(name: str) -> None:
    """Raise before any decoding if the explicit backend is unavailable."""
    if name not in _JPEG2000_BACKEND_MODULES:
        raise UnsupportedInputError(f"unknown_jpeg2000_backend:{name}")
    if not jpeg2000_backend_available(name):
        raise UnsupportedInputError(f"jpeg2000_backend_unavailable:{name}")


def resolve_decoding_plugin(transfer_syntax_uid: str, backend: str) -> Optional[str]:
    """Resolve the explicit pydicom decoding plugin for a transfer syntax.

    Returns ``None`` for uncompressed transfer syntaxes (native pydicom decode).
    For JPEG2000, verifies the requested backend is available (hard fail, no
    silent fallback) and returns the plugin identifier to PASS INTO the decode
    call. The caller must pass this plugin explicitly rather than letting
    pydicom pick a handler from its global list.
    """
    if not is_jpeg2000(transfer_syntax_uid):
        return None
    ensure_jpeg2000_backend(backend)
    return PYDICOM_DECODING_PLUGIN[backend]


# =========================================================================== #
# 17. SSIM (whole-image + ROI small-window handling) (Sections 19-20)          #
# =========================================================================== #
def whole_image_ssim(reference_uint8: "np.ndarray", target_uint8: "np.ndarray") -> Dict[str, Any]:
    """Whole-image SSIM (data_range=255, channel_axis=None). Records params."""
    from skimage.metrics import structural_similarity as ssim
    import skimage

    val = float(ssim(reference_uint8, target_uint8, data_range=255, channel_axis=None))
    return {
        "ssim": val,
        "skimage_version": getattr(skimage, "__version__", None),
        "data_range": 255,
        "channel_axis": None,
        "gaussian_weights": False,
        "use_sample_covariance": True,
    }


def roi_ssim(reference_roi: "np.ndarray", target_roi: "np.ndarray") -> Dict[str, Any]:
    """ROI SSIM with small-window handling. Never returns NaN.

    Picks the largest valid odd window <= min(roi_h, roi_w). If the ROI is too
    small (< 3 px on a side) SSIM is not evaluable and is recorded as null.
    """
    h, w = reference_roi.shape[:2]
    win = largest_odd_win_size(h, w)
    if win is None:
        return {"ssim": None, "evaluable": False, "reason": "roi_too_small_for_ssim",
                "win_size": None}
    try:
        from skimage.metrics import structural_similarity as ssim
    except Exception as exc:  # pragma: no cover - skimage is required at runtime
        return {"ssim": None, "evaluable": False,
                "reason": f"skimage_unavailable:{exc!r}", "win_size": win}
    val = float(ssim(reference_roi, target_roi, data_range=255, win_size=win,
                     channel_axis=None))
    return {"ssim": val, "evaluable": True, "reason": "ok", "win_size": win}


# =========================================================================== #
# 18. ROI-metric summaries (Section 20 / blocker 4)                            #
# --------------------------------------------------------------------------- #
# Pure, testable aggregations. We NEVER report only a global mean; we produce   #
# annotation-level micro, image-macro, class-macro, small-lesion, rare-class,   #
# worst-case ROI, and paired q100-minus-q95 views.                              #
# =========================================================================== #
def _mean(values: Sequence[float]) -> Optional[float]:
    vals = [v for v in values if v is not None]
    return float(sum(vals) / len(vals)) if vals else None


def summarize_roi_metrics(
    roi_rows: Sequence[Dict[str, Any]],
    metric_key: str = "ROI_MAE",
) -> Dict[str, Any]:
    """Micro / image-macro / class-macro summary for one metric, per quality."""
    out: Dict[str, Any] = {}
    qualities = sorted({int(r["jpeg_quality"]) for r in roi_rows})
    for q in qualities:
        rows = [r for r in roi_rows if int(r["jpeg_quality"]) == q]
        micro = _mean([r.get(metric_key) for r in rows])
        # image-macro: mean over per-image means.
        per_image: Dict[Any, List[float]] = {}
        per_class: Dict[Any, List[float]] = {}
        for r in rows:
            if r.get(metric_key) is None:
                continue
            per_image.setdefault(r["image_id"], []).append(r[metric_key])
            per_class.setdefault(r["canonical_class_id"], []).append(r[metric_key])
        image_macro = _mean([_mean(v) for v in per_image.values()])
        class_macro = _mean([_mean(v) for v in per_class.values()])
        out[q] = {
            "metric": metric_key,
            "jpeg_quality": q,
            "annotation_micro_mean": micro,
            "image_macro_mean": image_macro,
            "class_macro_mean": class_macro,
            "n_annotations": len(rows),
        }
    return out


def worst_roi_cases(
    roi_rows: Sequence[Dict[str, Any]],
    metric_key: str = "ROI_MAE",
    top: int = 5,
    largest_is_worst: bool = True,
) -> Dict[int, List[Dict[str, Any]]]:
    """Top-N worst ROI cases per quality by a metric (deterministic order)."""
    out: Dict[int, List[Dict[str, Any]]] = {}
    for q in sorted({int(r["jpeg_quality"]) for r in roi_rows}):
        rows = [r for r in roi_rows
                if int(r["jpeg_quality"]) == q and r.get(metric_key) is not None]
        rows.sort(key=lambda r: (r[metric_key], tie_break_rank(str(r.get("annotation_id", "")))),
                  reverse=largest_is_worst)
        out[q] = rows[:top]
    return out


def pairwise_q100_minus_q95(rows_in: Sequence[Dict[str, Any]],
                            metric_keys: Sequence[str] = ("ROI_MAE", "ROI_PSNR", "ROI_SSIM"),
                            key_field: str = "canonical_ann_id"
                            ) -> List[Dict[str, Any]]:
    """Paired q100 - q95 deltas per unit (annotation or image); both required."""
    by_unit: Dict[Any, Dict[int, Dict[str, Any]]] = {}
    for r in rows_in:
        by_unit.setdefault(r[key_field], {})[int(r["jpeg_quality"])] = r
    rows: List[Dict[str, Any]] = []
    for unit_id, per_q in sorted(by_unit.items(), key=lambda kv: str(kv[0])):
        if 95 not in per_q or 100 not in per_q:
            continue
        entry: Dict[str, Any] = {key_field: unit_id}
        for mk in metric_keys:
            a = per_q[100].get(mk)
            b = per_q[95].get(mk)
            entry[f"{mk}_q100_minus_q95"] = (a - b) if (a is not None and b is not None) else None
        rows.append(entry)
    return rows


def per_class_distribution(roi_rows: Sequence[Dict[str, Any]],
                           metric_key: str = "ROI_MAE") -> List[Dict[str, Any]]:
    """Per-class metric distribution (count/min/mean/max) per quality."""
    out: List[Dict[str, Any]] = []
    qualities = sorted({int(r["jpeg_quality"]) for r in roi_rows})
    classes = sorted({r["canonical_class_id"] for r in roi_rows})
    for q in qualities:
        for cid in classes:
            vals = [r[metric_key] for r in roi_rows
                    if int(r["jpeg_quality"]) == q
                    and r["canonical_class_id"] == cid
                    and r.get(metric_key) is not None]
            if not vals:
                continue
            out.append({
                "jpeg_quality": q, "canonical_class_id": cid, "metric": metric_key,
                "n": len(vals), "min": float(min(vals)),
                "mean": float(sum(vals) / len(vals)), "max": float(max(vals)),
            })
    return out


def small_lesion_ranking(roi_rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Deterministic small-lesion selection by ascending relative_bbox_area.

    Returns the smallest relative bbox overall and the smallest per class that
    appears in the pilot. Ranking basis is recorded explicitly. No clinical
    threshold is invented.
    """
    seen: Dict[Any, Dict[str, Any]] = {}
    for r in roi_rows:
        if int(r["jpeg_quality"]) != 95:
            continue  # one entry per annotation
        seen[r["canonical_ann_id"]] = r
    ranked = sorted(seen.values(),
                    key=lambda r: (float(r["relative_bbox_area"]),
                                   tie_break_rank(str(r["canonical_ann_id"]))))
    per_class_smallest: Dict[Any, Dict[str, Any]] = {}
    for r in ranked:
        cid = r["canonical_class_id"]
        if cid not in per_class_smallest:
            per_class_smallest[cid] = r
    return {
        "ranking_basis": "relative_bbox_area_ascending",
        "smallest_overall": ranked[0] if ranked else None,
        "smallest_per_class": per_class_smallest,
    }


def rare_class_ranking(class_image_count: Dict[int, int],
                       classes_present: Sequence[int],
                       top: int = 5) -> Dict[str, Any]:
    """Rare-class ranking by canonical class image_count ascending (no threshold)."""
    present = [c for c in class_image_count if c in set(classes_present)]
    ranked = sorted(present, key=lambda c: (class_image_count[c], c))
    return {
        "ranking_basis": "canonical_class_image_count_ascending",
        "rare_classes": ranked[:top],
        "counts": {c: class_image_count[c] for c in ranked[:top]},
    }


# =========================================================================== #
# 20. VOI LUT normalization + canonical<->COCO cross-check + misc pure helpers  #
# =========================================================================== #
def voi_lut_normalize(applied_values: "np.ndarray", nbits: int) -> "np.ndarray":
    """Normalize VOI-LUT output to [0,1] using the LUT bit depth (not observed)."""
    out_max = float((1 << int(nbits)) - 1)
    if out_max <= 0:
        raise DegenerateRangeError("voi_lut_nbits_degenerate")
    return np.clip(np.asarray(applied_values).astype(np.float64) / out_max, 0.0, 1.0)


def assert_extraction_in_bounds(x0: int, y0: int, x1: int, y1: int,
                                width: int, height: int) -> None:
    """Assert an integer ROI slice is within image bounds. No silent clamping."""
    if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
        raise UnsupportedInputError(
            f"roi_out_of_bounds: ({x0},{y0},{x1},{y1}) not within {width}x{height}"
        )


# Two artifacts derived from the SAME canonical source must agree to within a
# strict numerical tolerance (float round-trip only), NOT a 1-pixel slack.
BBOX_COORD_TOLERANCE = 1e-6


def crosscheck_canonical_coco_bbox(
    canon_row: Dict[str, Any],
    coco_ann: Optional[Dict[str, Any]],
    category_id_for_canonical: Dict[int, int],
    tol: float = BBOX_COORD_TOLERANCE,
) -> None:
    """Cross-check one canonical annotation against its COCO annotation.

    Verifies identifier linkage, canonical_class_id, the category_id mapping
    (from metadata, never hard-coded), and that xyxy (canonical) matches xywh
    (COCO) within tolerance. Any mismatch raises (BLOCKED); never repairs bbox.
    """
    if coco_ann is None:
        raise UnsupportedInputError(
            f"canonical_ann_id {canon_row.get('canonical_ann_id')} has no COCO match"
        )
    if str(coco_ann.get("canonical_ann_id")) != str(canon_row.get("canonical_ann_id")):
        raise UnsupportedInputError("canonical_ann_id mismatch")
    if str(coco_ann.get("original_image_id")) != str(canon_row.get("image_id")):
        raise UnsupportedInputError("image_id/original_image_id mismatch")
    ccid = int(canon_row["canonical_class_id"])
    if int(coco_ann.get("canonical_class_id")) != ccid:
        raise UnsupportedInputError("canonical_class_id mismatch")
    expected_cat = category_id_for_canonical.get(ccid)
    if expected_cat is None or int(coco_ann.get("category_id")) != int(expected_cat):
        raise UnsupportedInputError(
            f"category_id mapping mismatch for canonical_class_id {ccid}"
        )
    x_min, y_min = float(canon_row["x_min"]), float(canon_row["y_min"])
    x_max, y_max = float(canon_row["x_max"]), float(canon_row["y_max"])
    bx, by, bw, bh = [float(v) for v in coco_ann["bbox"]]
    if (abs(bx - x_min) > tol or abs(by - y_min) > tol
            or abs((bx + bw) - x_max) > tol or abs((by + bh) - y_max) > tol):
        raise UnsupportedInputError("xyxy(canonical) vs xywh(COCO) coordinate mismatch")


REFERENCE_RENDERER_STATUSES = (
    "PASS", "FAIL", "NOT_RUN_DEPENDENCY_UNAVAILABLE",
    "NOT_COMPARABLE_CONFIGURATION_UNCONTROLLED",
)


def reference_renderer_status(dependency_available: bool,
                              controlled_configuration: bool,
                              concordant: Optional[bool] = None) -> str:
    """Choose the reference-renderer status from ACTUAL evidence.

    * dependency missing            -> NOT_RUN_DEPENDENCY_UNAVAILABLE
    * dependency present but config
      not controlled                -> NOT_COMPARABLE_CONFIGURATION_UNCONTROLLED
    * controlled + concordant       -> PASS
    * controlled + not concordant   -> FAIL
    """
    if not dependency_available:
        return "NOT_RUN_DEPENDENCY_UNAVAILABLE"
    if not controlled_configuration:
        return "NOT_COMPARABLE_CONFIGURATION_UNCONTROLLED"
    return "PASS" if concordant else "FAIL"


def _reject_json_constant(token: str) -> Any:
    raise ValueError(f"non_finite_json_constant:{token}")


def strict_json_loads(text: str) -> Any:
    """json.loads that rejects NaN / Infinity / -Infinity (parse_constant)."""
    return json.loads(text, parse_constant=_reject_json_constant)


__all__ = [name for name in globals() if not name.startswith("_")]


# =========================================================================== #
# 19. Final coverage validation (Section 29 / blocker 10)                      #
# =========================================================================== #
def validate_full_coverage(covered: set, all_features: set) -> Dict[str, Any]:
    """Assert 14/14 classes + all supported strata + all extrema are covered."""
    missing = set(all_features) - set(covered)
    class_feats = {f for f in all_features if f.startswith("class=")}
    covered_classes = {f for f in covered if f.startswith("class=")}
    extrema_feats = {f for f in all_features if f.startswith("extremum=")}
    covered_extrema = {f for f in covered if f.startswith("extremum=")}
    result = {
        "all_features_total": len(all_features),
        "covered_total": len(set(covered) & set(all_features)),
        "missing_features": sorted(missing),
        "classes_expected": len(class_feats),
        "classes_covered": len(covered_classes & class_feats),
        "extrema_expected": len(extrema_feats),
        "extrema_covered": len(covered_extrema & extrema_feats),
        "fully_covered": len(missing) == 0,
    }
    if not result["fully_covered"]:
        raise UnsupportedInputError(
            f"coverage_incomplete: missing {sorted(missing)[:10]}"
        )
    if result["classes_covered"] != NUM_ABNORMAL_CLASSES:
        raise UnsupportedInputError(
            f"class_coverage_incomplete: {result['classes_covered']}/14"
        )
    return result


__all__ = [name for name in globals() if not name.startswith("_")]
