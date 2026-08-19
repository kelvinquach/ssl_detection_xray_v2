#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""PHASE 3A - DATASET DIAGNOSTICS BEFORE TRAINING.

PHASE3A_PROTOCOL = RESEARCHER_APPROVED / LOCKED
Protocol file: configs/protocol/phase3A_dataset_diagnostics.yaml

WHAT THIS SCRIPT IS
-------------------
Descriptive pre-training dataset diagnostics over the FIXED TRAINING SET
and the legitimately labeled subsets L_1pct / L_5pct / L_10pct / L_20pct.

WHAT THIS SCRIPT IS NOT
-----------------------
It is not training, not evaluation, not hyperparameter search, not
pseudo-label generation, not an ablation study and not threshold
optimization. It never loads a model, never writes a checkpoint, never
writes a pseudo-label and never uses a training seed.

DATA-USE FIREWALL (hard lock)
-----------------------------
* Detailed diagnostics are computed ONLY from
  data/processed/coco/instances_train.json and the four labeled subsets.
* The canonical masters are used for integrity/reference only.
* Validation and test are read ONLY through ``structural_summary()``,
  which returns a frozen dataclass carrying counts, category ids and a
  checksum. The parsed validation/test JSON never leaves that function,
  so no code path downstream can compute content diagnostics from them.
* No unlabeled-subset COCO file is referenced anywhere in this script or
  in the protocol. The hidden ground truth of U_b is never recovered by
  reverse-mapping unlabeled image ids into instances_train.json.

COMMENT CONVENTION (protocol section 42)
----------------------------------------
    [LOCKED SCIENTIFIC DECISION]  changing it changes the science
    [IMPLEMENTATION DETAIL]       engineering choice, no scientific meaning
    [VISUALIZATION DETAIL]        rendering only, never feeds back

CLI
---
    python scripts/03A_dataset_diagnostics.py \\
        --protocol configs/protocol/phase3A_dataset_diagnostics.yaml \\
        --mode preflight

    python scripts/03A_dataset_diagnostics.py \\
        --protocol configs/protocol/phase3A_dataset_diagnostics.yaml \\
        --mode full

EXIT CODES  [IMPLEMENTATION DETAIL]
-----------
    0  execution completed, no hard protocol violation, REVIEW STILL REQUIRED
    2  hard preflight / integrity / protocol failure
    3  runtime or output-generation failure

Exit code 0 never means Phase 3A is PASS or CLOSED. The script always
writes ``phase_status = OPEN_REVIEW_REQUIRED``.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

import numpy as np
import yaml

# =====================================================================
# LOCKED SCIENTIFIC CONSTANTS
# These are restated in the source so that a silent edit of the YAML can
# never move a scientific definition without tripping a guardrail.
# =====================================================================

PHASE_ID = "3A"
PROTOCOL_STATUS = "RESEARCHER_APPROVED_LOCKED"
PHASE_STATUS_VALUE = "OPEN_REVIEW_REQUIRED"
FORBIDDEN_PHASE_STATUS_VALUES = ("PASS", "CLOSED", "CLOSED_PASS")

CANONICAL_IMAGES = 4894
CANONICAL_ANNOTATIONS = 36096
CANONICAL_CATEGORIES = 14
CANONICAL_NO_FINDING = 500

TRAIN_IMAGES = 3426
TRAIN_ANNOTATIONS = 25260
TRAIN_NO_FINDING = 350

VAL_IMAGES = 734
VAL_ANNOTATIONS = 5399
VAL_NO_FINDING = 75

TEST_IMAGES = 734
TEST_ANNOTATIONS = 5437
TEST_NO_FINDING = 75

BUDGET_KEYS: Tuple[str, ...] = ("1pct", "5pct", "10pct", "20pct")
LABELED_IMAGE_COUNTS = {"1pct": 34, "5pct": 171, "10pct": 343, "20pct": 685}
LABELED_NO_FINDING_COUNTS = {"1pct": 3, "5pct": 17, "10pct": 35, "20pct": 70}

MASTER_JPG_SHA256 = (
    "f587152278f713460ff1e727a2912248a47052f6abc48de8f7bad6e8a63b94c0")
TRAIN_SHA256 = (
    "0f3c37a6f1b5bcc6971b01fd4c69c7a2a3a1e8bee145c11488c7a658dcbbebe3")
VAL_SHA256 = (
    "33064f47ba690be13e9d418d8ec5d4a6a2bec8482c24ea3eadf83fb33d01762a")
TEST_SHA256 = (
    "e1a73110e92af2656276d6c532035afe474b6ac6f2b2f03849834c036e1c00a4")
LABELED_SHA256 = {
    "1pct": "1e267c547a8f535f6a69088da4735bd12c0188fae1b49e9d785b3e1e6883df98",
    "5pct": "ebb98f32cad612fe48adbbd6b5da8aa5db10bec478d1c7fd9288a9a93bbc3763",
    "10pct": "6814b5b9ba8dfcd4af483d97c1a6b7e7ab26914bdd02bbb923502b43ecf2fdf6",
    "20pct": "6a8b5bf9c59baea41eefb0611e15a387257255b6fc59a3de607bfc2e80801a14",
}

# --- rare class -------------------------------------------------------
RARE_DEFINITION_TYPE = "PRE_SPECIFIED_OPERATIONAL_DEFINITION"
RARE_PRIMARY_THRESHOLD = 0.05          # [LOCKED SCIENTIFIC DECISION]
RARE_SENSITIVITY_THRESHOLDS = (0.01, 0.05, 0.10)   # [LOCKED]
RARE_BOUNDARY_MAX_RARE_COUNT = 171     # derived from 0.05 * 3426
RARE_BOUNDARY_MIN_NONRARE_COUNT = 172

# --- bbox size --------------------------------------------------------
BBOX_SIZE_DEFINITION_TYPE = "PRE_SPECIFIED_OPERATIONAL_DEFINITION"
BBOX_SMALL_PRIMARY_THRESHOLD = 0.01    # [LOCKED SCIENTIFIC DECISION]
BBOX_LARGE_PRIMARY_THRESHOLD = 0.10    # [LOCKED SCIENTIFIC DECISION]
BBOX_SMALL_SENSITIVITY_THRESHOLDS = (0.005, 0.01, 0.02)   # [LOCKED]
BBOX_LARGE_SENSITIVITY_THRESHOLDS = (0.05, 0.10, 0.20)    # [LOCKED]
BBOX_SIZE_CATEGORIES: Tuple[str, ...] = ("small", "medium", "large")

# --- bbox location ----------------------------------------------------
HEATMAP_BINS_X = 50                    # [LOCKED SCIENTIFIC DECISION]
HEATMAP_BINS_Y = 50                    # [LOCKED SCIENTIFIC DECISION]

# --- multilabel -------------------------------------------------------
EXPECTED_COOCCURRENCE_PAIRS = 91       # 14 choose 2

SD_DDOF = 1                            # [LOCKED] sample standard deviation
PERCENTILE_METHOD = "linear"           # [LOCKED] numpy linear interpolation

# --- scope ------------------------------------------------------------
PRIMARY_DIAGNOSTIC_SCOPE = "FIXED_TRAIN"
DETAILED_SCOPE_ALLOWED: Tuple[str, ...] = (
    "train", "labeled_1pct", "labeled_5pct", "labeled_10pct", "labeled_20pct")
STRUCTURAL_ONLY_SCOPES: Tuple[str, ...] = ("val", "test", "canonical")

# Paths that would give access to unlabeled-subset ground truth.
# [LOCKED SCIENTIFIC DECISION] Deliberately narrow so that Phase 2F
# *evidence* files whose names merely contain the word "unlabeled"
# (e.g. reports/02F_labeled_unlabeled_validation_report.json) are not
# confused with an unlabeled COCO instances file.
FORBIDDEN_UNLABELED_PATH_PATTERNS: Tuple[str, ...] = (
    r"instances_unlabeled",
    r"unlabeled_splits[\\/]",
    r"unlabeled_(1|5|10|20)pct\.json$",
)

# The forbidden terminology list itself lives only in the protocol YAML
# (terminology.forbidden), so that the source of this script never
# contains the forbidden phrases even as data.

TRAINING_ARTIFACT_DIRS: Tuple[str, ...] = (
    "models", "checkpoints", "pseudo_labels", "pseudo_label")
TRAINING_ARTIFACT_SUFFIXES: Tuple[str, ...] = (
    ".pth", ".pt", ".ckpt", ".onnx", ".safetensors")

EXIT_OK = 0
EXIT_HARD_FAIL = 2
EXIT_RUNTIME = 3

# --- output-path safety  [LOCKED SCIENTIFIC DECISION: the policy] -----
# Phase 3A may only ever write inside these directories, and may never
# write inside the locked directories below. Enforced at PREFLIGHT time,
# before any directory or file can be created.
WRITABLE_OUTPUT_DIRECTORIES: Tuple[str, ...] = ("reports", "plots/dataset")
FORBIDDEN_OUTPUT_DIRECTORIES: Tuple[str, ...] = (
    "data/processed/coco",
    "data/manifests",
    "configs/protocol",
    "models",
    "checkpoints",
    "pseudo_labels",
)

# --- output contract schemas  [IMPLEMENTATION DETAIL] -----------------
# The Python declaration is authoritative for what is written; the YAML
# copy is cross-checked against it during preflight so the two can never
# drift apart silently.
BBOX_DISTRIBUTION_COLUMNS: Tuple[str, ...] = (
    "annotation_id", "image_id", "category_id", "class_name", "x", "y",
    "w", "h", "image_width", "image_height", "raw_area", "normalized_width",
    "normalized_height", "normalized_area", "aspect_ratio",
    "center_x_normalized", "center_y_normalized", "primary_size_category",
)

MANDATORY_CSV_SCHEMAS: Dict[str, Tuple[str, ...]] = {
    "class_distribution": (
        "category_id", "class_name", "image_count", "image_prevalence",
        "bbox_annotation_count", "bbox_annotation_share",
        "bbox_per_positive_image", "rare_flag"),
    "class_imbalance": ("metric", "value", "unit"),
    "bbox_distribution": BBOX_DISTRIBUTION_COLUMNS,
    "bbox_summary": (
        "variable", "role", "scope", "n", "mean", "sd", "min", "p05", "p25",
        "median", "p75", "p95", "max"),
    "bbox_count_per_image": (
        "population", "n", "mean", "sd_ddof1", "median", "p95", "max"),
    "negative_distribution": (
        "scope", "diagnostic_role", "image_count", "positive_image_count",
        "negative_image_count", "negative_prevalence"),
    "label_cardinality": (
        "record_type", "cardinality", "image_count", "percentage",
        "statistic", "statistic_value"),
    "class_cooccurrence": (
        "category_id_a", "class_name_a", "category_id_b", "class_name_b",
        "n_a_images", "n_b_images", "n_both_images", "n_union_images",
        "jaccard"),
    "labeled_budget_coverage": (
        "budget", "labeled_image_count", "negative_image_count",
        "negative_prevalence", "class_coverage_out_of_14", "category_id",
        "class_name", "image_count", "image_prevalence", "train_image_count",
        "train_image_prevalence", "absolute_deviation_from_train",
        "train_rare_flag", "rare_class_support", "coverage_present",
        "budget_max_absolute_deviation", "budget_mean_absolute_deviation"),
    "threshold_sensitivity": (
        "analysis_type", "parameter", "threshold", "is_primary", "category",
        "count", "percentage", "primary_percentage",
        "delta_percentage_points", "category_id", "class_name", "image_count",
        "image_prevalence", "rare_flag", "primary_rare_flag",
        "status_changed_vs_primary", "stable_rare_flag",
        "stable_nonrare_flag", "threshold_sensitive_flag"),
    "split_distribution": (
        "split", "image_count", "image_percentage", "annotation_count",
        "negative_count", "negative_percentage", "official_json_sha256",
        "image_membership_sha256", "diagnostic_usage"),
}

# Structural row counts only. None means the row count is data dependent
# and is therefore never asserted. No observed scientific distribution is
# ever a PASS/FAIL gate here.
EXPECTED_CSV_ROW_COUNTS: Dict[str, Optional[int]] = {
    "class_distribution": CANONICAL_CATEGORIES,              # 14
    "class_imbalance": 12,
    "bbox_distribution": TRAIN_ANNOTATIONS,                  # 25260
    "bbox_summary": 7,
    "bbox_count_per_image": 2,
    "negative_distribution": 3 + len(BUDGET_KEYS),           # 7
    "label_cardinality": None,                               # data dependent
    "class_cooccurrence": EXPECTED_COOCCURRENCE_PAIRS,       # 91
    "labeled_budget_coverage": len(BUDGET_KEYS) * CANONICAL_CATEGORIES,  # 56
    # rare per-class 3 x 14 + rare summary 3 + small 3 x 3 + large 3 x 3
    "threshold_sensitivity": (
        len(RARE_SENSITIVITY_THRESHOLDS) * CANONICAL_CATEGORIES
        + len(RARE_SENSITIVITY_THRESHOLDS)
        + len(BBOX_SMALL_SENSITIVITY_THRESHOLDS) * len(BBOX_SIZE_CATEGORIES)
        + len(BBOX_LARGE_SENSITIVITY_THRESHOLDS) * len(BBOX_SIZE_CATEGORIES)),
    "split_distribution": 3,
}

MANDATORY_JSON_OUTPUT_KEYS: Tuple[str, ...] = ("validation_json",
                                               "artifact_manifest")

# --- mandatory plot keys  [IMPLEMENTATION DETAIL] ---------------------
# These are the keys of output_contract.mandatory_plots in the protocol
# YAML. resolve_output_contract() prefixes each of them with "plot_" to
# build its resolved-target key, so run_full() and audit_output_contract()
# MUST derive their expected keys from this single list. Hard-coding the
# list twice is what produced the plot_negative_distribution vs
# plot_negative_image_distribution mismatch.
MANDATORY_PLOT_KEYS: Tuple[str, ...] = (
    "class_distribution",
    "bbox_distribution",
    "bbox_location_heatmap",
    "negative_image_distribution",
)

MANDATORY_PLOT_TARGET_KEYS: Tuple[str, ...] = tuple(
    "plot_{0}".format(key) for key in MANDATORY_PLOT_KEYS)

# --- declared hard-fail contract  [LOCKED SCIENTIFIC DECISION] --------
# The runtime may only ever emit a hard-fail code that the protocol
# declares in guardrails.hard_fail_conditions, i.e. HF01..HF35. This
# keeps the machine-readable guardrail contract and the runtime evidence
# in a one-to-one relationship.
DECLARED_HARD_FAIL_CODES: Tuple[str, ...] = tuple(
    "HF{0:02d}".format(index) for index in range(1, 36))

# The single permitted sub-code and the declared code it rolls up into.
# HF09b is the fixed-train restatement of the HF09 canonical "No Finding
# is not a detection category" policy; both roll up into HF09 in the
# validation record. No other sub-code is allowed to exist.
DECLARED_HARD_FAIL_SUBCODES: Dict[str, str] = {"HF09b": "HF09"}


def undeclared_hard_fail_codes(codes: Iterable[str]) -> List[str]:
    """Return every emitted hard-fail code outside the declared contract.

    A code is declared when it is one of HF01..HF35, or when it is a
    registered sub-code of one of them. Any other ad-hoc sub-code that the
    protocol does not declare is a contract violation and fails the run.
    """
    offenders: List[str] = []
    for code in codes:
        text = str(code)
        if not text.upper().startswith("HF"):
            continue  # PF.. / W.. codes are advisory, not hard-fail codes
        if text in DECLARED_HARD_FAIL_CODES:
            continue
        if text in DECLARED_HARD_FAIL_SUBCODES:
            continue
        offenders.append(text)
    return sorted(set(offenders))


# =====================================================================
# Exceptions  [IMPLEMENTATION DETAIL]
# =====================================================================

class Phase3AError(RuntimeError):
    """Base class for every explicit Phase 3A failure."""


class ProtocolError(Phase3AError):
    """The protocol file is missing, malformed or has been altered."""


class ScopeViolationError(Phase3AError):
    """An attempt was made to use data outside the permitted scope."""


class OutputContractError(Phase3AError):
    """A mandatory output could not be produced."""


class OutputPathError(Phase3AError):
    """An output target is outside the explicitly writable directories."""


# =====================================================================
# Firewall tripwires
# Every detailed load registers its scope key here. The validation stage
# asserts that the set never contains a forbidden scope.
# =====================================================================

_DETAILED_SCOPES_USED: Set[str] = set()
_STRUCTURAL_SCOPES_USED: Set[str] = set()


def detailed_scopes_used() -> Tuple[str, ...]:
    return tuple(sorted(_DETAILED_SCOPES_USED))


def structural_scopes_used() -> Tuple[str, ...]:
    return tuple(sorted(_STRUCTURAL_SCOPES_USED))


def reset_scope_tripwires() -> None:
    """[IMPLEMENTATION DETAIL] Used by the guardrail tests."""
    _DETAILED_SCOPES_USED.clear()
    _STRUCTURAL_SCOPES_USED.clear()


# =====================================================================
# Output-path safety gate
#
# Every writer in this module refuses to touch a path that has not been
# resolved and registered by ``resolve_safe_output_path``. The whole
# output contract is resolved during PREFLIGHT, before FULL mode is
# allowed to create a single directory or file, so a bad target can
# never reach the filesystem and be caught only afterwards by HF34.
# =====================================================================

_VALIDATED_OUTPUT_PATHS: Set[Path] = set()


def reset_validated_output_paths() -> None:
    """[IMPLEMENTATION DETAIL] Used by the guardrail tests."""
    _VALIDATED_OUTPUT_PATHS.clear()


def validated_output_paths() -> Tuple[str, ...]:
    return tuple(sorted(str(path) for path in _VALIDATED_OUTPUT_PATHS))


def resolve_safe_output_path(
        repo_root: Path, candidate: Any,
        writable_directories: Sequence[str] = WRITABLE_OUTPUT_DIRECTORIES,
        forbidden_directories: Sequence[str] = FORBIDDEN_OUTPUT_DIRECTORIES,
        label: str = "output", register: bool = True) -> Path:
    """Resolve a declared output target and prove that writing it is legal.

    Rejects, in this order:
      * an empty target;
      * an absolute POSIX path, a Windows drive-letter path or a '~' path;
      * any target containing a '..' traversal segment;
      * any target that escapes the repository root once resolved;
      * any target under a forbidden directory;
      * any target that is not inside an explicitly writable directory.

    Returns the resolved absolute Path and registers it so that
    ``write_csv`` / ``write_json`` / the plot writers will accept it.
    """
    text = "" if candidate is None else str(candidate).strip()
    if not text:
        raise OutputPathError(
            "{0}: empty output target is not allowed".format(label))

    normalized = text.replace("\\", "/")
    if normalized.startswith("~"):
        raise OutputPathError(
            "{0}: home-relative output target rejected: {1}".format(
                label, text))
    if normalized.startswith("/") or re.match(r"^[A-Za-z]:", normalized):
        raise OutputPathError(
            "{0}: absolute output target rejected: {1}".format(label, text))
    parts = [part for part in normalized.split("/") if part not in ("", ".")]
    if ".." in parts:
        raise OutputPathError(
            "{0}: parent-directory traversal rejected: {1}".format(
                label, text))

    root = Path(repo_root).resolve()
    resolved = (root / Path(*parts)).resolve() if parts else root
    try:
        relative = resolved.relative_to(root)
    except ValueError as error:
        raise OutputPathError(
            "{0}: output target escapes the repository root: {1}".format(
                label, text)) from error
    relative_posix = relative.as_posix()

    for forbidden in forbidden_directories:
        blocked = str(forbidden).replace("\\", "/").strip("/")
        if not blocked:
            continue
        if relative_posix == blocked or relative_posix.startswith(
                blocked + "/"):
            raise OutputPathError(
                "{0}: output target is inside the locked directory "
                "'{1}': {2}".format(label, blocked, text))

    for writable in writable_directories:
        allowed = str(writable).replace("\\", "/").strip("/")
        if not allowed:
            continue
        if relative_posix == allowed or relative_posix.startswith(
                allowed + "/"):
            if register:
                _VALIDATED_OUTPUT_PATHS.add(resolved)
            return resolved

    raise OutputPathError(
        "{0}: output target is not inside an explicitly writable directory "
        "{1}: {2}".format(label, list(writable_directories), text))


def _assert_validated_output(path: Path) -> None:
    """Refuse to write a path that the preflight gate never approved."""
    if Path(path).resolve() not in _VALIDATED_OUTPUT_PATHS:
        raise OutputPathError(
            "refusing to write an output path that was not approved by the "
            "preflight output-path gate: {0}".format(path))


def resolve_output_contract(protocol: Mapping[str, Any], repo_root: Path
                            ) -> Tuple[Dict[str, Path], List[str]]:
    """Resolve and gate every declared output target.

    Called during PREFLIGHT. Returns the resolved targets plus the list of
    human-readable rejection messages (empty when the contract is safe).
    """
    contract = dig(protocol, "output_contract", {}) or {}
    writable = list(dig(protocol, "output_contract.writable_directories",
                        list(WRITABLE_OUTPUT_DIRECTORIES)) or [])
    forbidden = list(dig(protocol, "output_contract.forbidden_write_paths",
                         list(FORBIDDEN_OUTPUT_DIRECTORIES)) or [])
    # The Python declaration always wins: a YAML that widened the writable
    # set or dropped a forbidden directory cannot loosen the gate.
    writable = [entry for entry in writable
                if entry in WRITABLE_OUTPUT_DIRECTORIES]
    for entry in FORBIDDEN_OUTPUT_DIRECTORIES:
        if entry not in forbidden:
            forbidden.append(entry)

    declared: List[Tuple[str, Any]] = [
        ("reports_dir", contract.get("reports_dir")),
        ("plots_dir", contract.get("plots_dir")),
        ("report", contract.get("mandatory_report")),
        ("validation_json", contract.get("mandatory_validation_json")),
        ("artifact_manifest", contract.get("mandatory_artifact_manifest")),
        ("guardrail_junit", contract.get("guardrail_junit")),
    ]
    for key, value in sorted((contract.get("mandatory_csv") or {}).items()):
        declared.append(("csv_{0}".format(key), value))
    for key, value in sorted((contract.get("mandatory_plots") or {}).items()):
        declared.append(("plot_{0}".format(key), value))

    resolved: Dict[str, Path] = {}
    errors: List[str] = []
    for key, value in declared:
        try:
            resolved[key] = resolve_safe_output_path(
                repo_root, value, writable, forbidden, label=key)
        except OutputPathError as error:
            errors.append(str(error))
    return resolved, errors


# =====================================================================
# Small pure helpers - all deterministic, all unit-testable
# =====================================================================

def sha256_file(path: Path) -> str:
    """SHA-256 of the exact bytes of a file."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def utc_now_iso() -> str:
    """[IMPLEMENTATION DETAIL] Metadata only. Never enters a computation."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def normalized_width(w: float, image_width: float) -> float:
    """w_n = w / W."""
    return float(w) / float(image_width)


def normalized_height(h: float, image_height: float) -> float:
    """h_n = h / H."""
    return float(h) / float(image_height)


def normalized_area(w: float, h: float,
                    image_width: float, image_height: float) -> float:
    """a_n = (w * h) / (W * H).  [LOCKED SCIENTIFIC DECISION]"""
    return (float(w) * float(h)) / (float(image_width) * float(image_height))


def aspect_ratio(w: float, h: float) -> float:
    """AR = w / h."""
    return float(w) / float(h)


def bbox_center_normalized(x: float, y: float, w: float, h: float,
                           image_width: float,
                           image_height: float) -> Tuple[float, float]:
    """center_x_n = (x + w/2)/W ; center_y_n = (y + h/2)/H.

    Coordinate convention is top-left origin, x left->right,
    y top->bottom.  [LOCKED SCIENTIFIC DECISION]
    """
    cx = (float(x) + float(w) / 2.0) / float(image_width)
    cy = (float(y) + float(h) / 2.0) / float(image_height)
    return cx, cy


def classify_bbox_size(area_normalized: float,
                       small_threshold: float = BBOX_SMALL_PRIMARY_THRESHOLD,
                       large_threshold: float = BBOX_LARGE_PRIMARY_THRESHOLD
                       ) -> str:
    """Normalized-area-based bbox size category.

    [LOCKED SCIENTIFIC DECISION] Boundary semantics are exact:
        a_n <  small_threshold                      -> "small"
        small_threshold <= a_n < large_threshold    -> "medium"
        a_n >= large_threshold                      -> "large"
    So a_n == 0.01 is MEDIUM and a_n == 0.10 is LARGE.

    These are NOT the COCO pixel-area categories.
    """
    value = float(area_normalized)
    if value < float(small_threshold):
        return "small"
    if value >= float(large_threshold):
        return "large"
    return "medium"


def bbox_validity_violations(x: float, y: float, w: float, h: float,
                             image_width: float,
                             image_height: float) -> List[str]:
    """Return the list of violated validity rules for one annotation.

    [LOCKED SCIENTIFIC DECISION] Exact comparison, tolerance 0.0.
    No clamping, repairing, deleting, moving or resizing is ever done.
    """
    violations: List[str] = []
    if not float(w) > 0.0:
        violations.append("w_not_positive")
    if not float(h) > 0.0:
        violations.append("h_not_positive")
    if float(x) < 0.0:
        violations.append("x_negative")
    if float(y) < 0.0:
        violations.append("y_negative")
    if float(x) + float(w) > float(image_width):
        violations.append("x_plus_w_exceeds_image_width")
    if float(y) + float(h) > float(image_height):
        violations.append("y_plus_h_exceeds_image_height")
    area_n = normalized_area(w, h, image_width, image_height)
    if not (0.0 < area_n <= 1.0):
        violations.append("normalized_area_out_of_range")
    return violations


def image_prevalence(image_count: int, total_images: int) -> float:
    """P_c_img = N_c_img / total_images."""
    if total_images <= 0:
        raise ValueError("total_images must be positive")
    return float(image_count) / float(total_images)


def is_rare(prevalence: float,
            threshold: float = RARE_PRIMARY_THRESHOLD) -> bool:
    """Rare(c) <=> P_c_img < threshold.  [LOCKED SCIENTIFIC DECISION]

    Strictly less than. Rarity is image-level support inside the fixed
    training dataset. It is never defined by bounding-box counts and it
    is never a claim about clinical rarity.
    """
    return float(prevalence) < float(threshold)


def class_image_support(annotations: Iterable[Mapping[str, Any]]
                        ) -> Dict[int, Set[int]]:
    """category_id -> set of UNIQUE image ids containing that class.

    An image is counted once per class regardless of how many bounding
    boxes of that class it carries.  [LOCKED SCIENTIFIC DECISION]
    """
    support: Dict[int, Set[int]] = {}
    for annotation in annotations:
        category_id = int(annotation["category_id"])
        support.setdefault(category_id, set()).add(int(annotation["image_id"]))
    return support


def class_bbox_counts(annotations: Iterable[Mapping[str, Any]]
                      ) -> Dict[int, int]:
    """category_id -> count of COCO bounding-box annotation records.

    [LOCKED SCIENTIFIC DECISION] This is an annotation-record count. It
    must not be interpreted as a count of independently resolved
    clinical lesions.
    """
    counts: Dict[int, int] = {}
    for annotation in annotations:
        category_id = int(annotation["category_id"])
        counts[category_id] = counts.get(category_id, 0) + 1
    return counts


def image_class_presence(annotations: Iterable[Mapping[str, Any]]
                         ) -> Dict[int, Set[int]]:
    """image_id -> set of UNIQUE category ids present in that image."""
    presence: Dict[int, Set[int]] = {}
    for annotation in annotations:
        image_id = int(annotation["image_id"])
        presence.setdefault(image_id, set()).add(int(annotation["category_id"]))
    return presence


def label_cardinality_per_image(presence: Mapping[int, Set[int]],
                                all_image_ids: Iterable[int]
                                ) -> Dict[int, int]:
    """LC_i = number of UNIQUE detection classes in image i.

    [LOCKED SCIENTIFIC DECISION] A negative (zero-GT) image has LC_i = 0.
    Bounding-box counts must never be substituted for unique classes.
    """
    return {int(image_id): len(presence.get(int(image_id), set()))
            for image_id in all_image_ids}


def bbox_count_per_image(annotations: Iterable[Mapping[str, Any]],
                         all_image_ids: Iterable[int]) -> Dict[int, int]:
    """B_i = number of bounding-box annotation records of image i."""
    counts = {int(image_id): 0 for image_id in all_image_ids}
    for annotation in annotations:
        image_id = int(annotation["image_id"])
        if image_id in counts:
            counts[image_id] += 1
    return counts


def cooccurrence_count(presence: Mapping[int, Set[int]],
                       category_a: int, category_b: int) -> int:
    """N_cd = number of images containing BOTH classes (image-level)."""
    total = 0
    for classes in presence.values():
        if category_a in classes and category_b in classes:
            total += 1
    return total


def jaccard(n_c: int, n_d: int, n_cd: int) -> Optional[float]:
    """J(c,d) = N_cd / (N_c + N_d - N_cd).  [LOCKED SCIENTIFIC DECISION]"""
    union = int(n_c) + int(n_d) - int(n_cd)
    if union <= 0:
        return None
    return float(n_cd) / float(union)


def percentile(values: Sequence[float], q: float) -> Optional[float]:
    """numpy percentile with the locked 'linear' interpolation method."""
    array = np.asarray(list(values), dtype=float)
    if array.size == 0:
        return None
    try:
        return float(np.percentile(array, q, method=PERCENTILE_METHOD))
    except TypeError:  # numpy < 1.22  [IMPLEMENTATION DETAIL]
        return float(np.percentile(array, q, interpolation=PERCENTILE_METHOD))


def continuous_stats(values: Sequence[float]) -> Dict[str, Optional[float]]:
    """n / mean / sample SD (ddof=1) / min / p05 / p25 / median / p75 / p95 / max."""
    array = np.asarray(list(values), dtype=float)
    n = int(array.size)
    if n == 0:
        return {"n": 0, "mean": None, "sd": None, "min": None, "p05": None,
                "p25": None, "median": None, "p75": None, "p95": None,
                "max": None}
    return {
        "n": n,
        "mean": float(array.mean()),
        "sd": float(array.std(ddof=SD_DDOF)) if n >= 2 else None,
        "min": float(array.min()),
        "p05": percentile(array, 5),
        "p25": percentile(array, 25),
        "median": percentile(array, 50),
        "p75": percentile(array, 75),
        "p95": percentile(array, 95),
        "max": float(array.max()),
    }


def imbalance_ratio(support_counts: Sequence[int]) -> Optional[float]:
    """R_max_min = max(N_c_img) / min(N_c_img)."""
    if not support_counts:
        return None
    minimum = min(support_counts)
    if minimum <= 0:
        return None
    return float(max(support_counts)) / float(minimum)


def coefficient_of_variation(values: Sequence[float]) -> Optional[float]:
    """CV = sample_SD / mean.  Secondary descriptive statistic only."""
    array = np.asarray(list(values), dtype=float)
    if array.size < 2:
        return None
    mean = float(array.mean())
    if mean == 0.0:
        return None
    return float(array.std(ddof=SD_DDOF)) / mean


def size_category_counts(areas: Sequence[float], small_threshold: float,
                         large_threshold: float) -> Dict[str, int]:
    """Counts of the three normalized-area-based size categories."""
    counts = {name: 0 for name in BBOX_SIZE_CATEGORIES}
    for area in areas:
        counts[classify_bbox_size(area, small_threshold, large_threshold)] += 1
    return counts


def is_nested(sequence_of_sets: Sequence[Set[int]]) -> bool:
    """True when each set is a subset of the following one."""
    for earlier, later in zip(sequence_of_sets, sequence_of_sets[1:]):
        if not earlier.issubset(later):
            return False
    return True


def freedman_diaconis_bin_count(values: Sequence[float], min_bins: int,
                                max_bins: int) -> int:
    """[VISUALIZATION DETAIL] Histogram bin count for the area panel only.

    This never touches the locked 0.01 / 0.10 size thresholds; it only
    decides how the continuous panel is drawn.
    """
    array = np.asarray(list(values), dtype=float)
    if array.size < 2:
        return int(min_bins)
    q75 = percentile(array, 75)
    q25 = percentile(array, 25)
    iqr = float(q75) - float(q25)
    value_range = float(array.max()) - float(array.min())
    if iqr <= 0.0 or value_range <= 0.0:
        return int(min_bins)
    width = 2.0 * iqr / (float(array.size) ** (1.0 / 3.0))
    if width <= 0.0:
        return int(min_bins)
    bins = int(math.ceil(value_range / width))
    return int(max(int(min_bins), min(int(max_bins), bins)))


# =====================================================================
# Formatting helpers  [IMPLEMENTATION DETAIL]
# =====================================================================

def fmt_num(value: Any) -> str:
    """Deterministic, round-trip-exact CSV rendering of a number."""
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    number = float(value)
    if math.isnan(number):
        return ""
    return repr(number)


def fmt_bool(value: Optional[bool]) -> str:
    if value is None:
        return ""
    return "true" if bool(value) else "false"


# =====================================================================
# Data containers
# =====================================================================

@dataclass(frozen=True)
class StructuralSummary:
    """Everything Phase 3A is ever allowed to know about val / test.

    [LOCKED SCIENTIFIC DECISION] This dataclass deliberately carries NO
    annotation-level, class-level or geometry-level information. The
    parsed JSON is discarded inside ``structural_summary`` so that no
    downstream code path can produce content diagnostics for a
    structural-only split.
    """
    scope: str
    path: str
    sha256: str
    image_count: int
    annotation_count: int
    negative_image_count: int
    category_ids: Tuple[int, ...]
    category_names: Tuple[str, ...]


@dataclass
class CheckRecord:
    code: str
    description: str
    status: str
    expected: Any
    observed: Any
    evidence: str

    def as_dict(self) -> Dict[str, Any]:
        return {
            "code": self.code,
            "description": self.description,
            "status": self.status,
            "expected": self.expected,
            "observed": self.observed,
            "evidence": self.evidence,
        }


@dataclass
class PreflightResult:
    checks: List[CheckRecord] = field(default_factory=list)
    warnings: List[CheckRecord] = field(default_factory=list)
    flags: Dict[str, Any] = field(default_factory=dict)

    @property
    def hard_errors(self) -> List[CheckRecord]:
        return [check for check in self.checks if check.status == "FAIL"]

    @property
    def hard_error_count(self) -> int:
        return len(self.hard_errors)

    @property
    def warning_count(self) -> int:
        return len(self.warnings)

    def add(self, code: str, description: str, condition: bool, expected: Any,
            observed: Any, evidence: str) -> None:
        self.checks.append(CheckRecord(
            code=code, description=description,
            status="PASS" if condition else "FAIL",
            expected=expected, observed=observed, evidence=evidence))

    def warn(self, code: str, description: str, expected: Any, observed: Any,
             evidence: str) -> None:
        self.warnings.append(CheckRecord(
            code=code, description=description, status="WARNING",
            expected=expected, observed=observed, evidence=evidence))


# =====================================================================
# Protocol loading
# =====================================================================

def dig(obj: Any, dotted_path: str, default: Any = None) -> Any:
    """Safe nested lookup."""
    current = obj
    for part in dotted_path.split("."):
        if isinstance(current, Mapping) and part in current:
            current = current[part]
        else:
            return default
    return current


def load_protocol(protocol_path: Path) -> Tuple[Dict[str, Any], str]:
    if not protocol_path.is_file():
        raise ProtocolError(
            "Protocol file not found: {0}".format(protocol_path))
    raw = protocol_path.read_text(encoding="utf-8")
    parsed = yaml.safe_load(raw)
    if not isinstance(parsed, dict):
        raise ProtocolError(
            "Protocol file did not parse into a mapping: {0}".format(
                protocol_path))
    # Hash the exact bytes on disk, not the re-encoded text.
    return parsed, sha256_file(protocol_path)


def resolve_repo_root(protocol_path: Path,
                      explicit_root: Optional[Path]) -> Path:
    if explicit_root is not None:
        return explicit_root.resolve()
    # scripts/03A_dataset_diagnostics.py -> repository root
    return Path(__file__).resolve().parents[1]


def collect_protocol_paths(protocol: Mapping[str, Any]) -> List[str]:
    """Every path string configured under input_artifacts."""
    found: List[str] = []

    def walk(node: Any) -> None:
        if isinstance(node, Mapping):
            for value in node.values():
                walk(value)
        elif isinstance(node, (list, tuple)):
            for value in node:
                walk(value)
        elif isinstance(node, str):
            found.append(node)

    walk(protocol.get("input_artifacts", {}))
    return sorted(found)


def find_forbidden_unlabeled_paths(paths: Iterable[str]) -> List[str]:
    """Paths that would expose unlabeled-subset ground truth."""
    offenders: List[str] = []
    for candidate in paths:
        for pattern in FORBIDDEN_UNLABELED_PATH_PATTERNS:
            if re.search(pattern, candidate):
                offenders.append(candidate)
                break
    return sorted(set(offenders))


# =====================================================================
# COCO loading - firewalled
# =====================================================================

def _read_json(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        raise Phase3AError("Required input artifact not found: {0}".format(path))
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def load_detailed_coco(path: Path, scope: str) -> Dict[str, Any]:
    """Load a COCO file for DETAILED diagnostics.

    [LOCKED SCIENTIFIC DECISION] Only the fixed training set and the four
    legitimately labeled subsets may ever be opened through this
    function. Any other scope raises ScopeViolationError.
    """
    if scope not in DETAILED_SCOPE_ALLOWED:
        raise ScopeViolationError(
            "Detailed diagnostics are not permitted for scope '{0}'. "
            "Permitted scopes: {1}".format(
                scope, ", ".join(DETAILED_SCOPE_ALLOWED)))
    for pattern in FORBIDDEN_UNLABELED_PATH_PATTERNS:
        if re.search(pattern, str(path).replace("\\", "/")):
            raise ScopeViolationError(
                "Refusing to open an unlabeled-subset artifact: {0}".format(
                    path))
    _DETAILED_SCOPES_USED.add(scope)
    return _read_json(path)


def structural_summary(path: Path, scope: str) -> StructuralSummary:
    """Load a split for STRUCTURAL / INTEGRITY evidence only.

    [LOCKED SCIENTIFIC DECISION] The parsed document is discarded inside
    this function. Only counts, category identity and the checksum are
    returned, so validation and test can never contribute content
    diagnostics.
    """
    _STRUCTURAL_SCOPES_USED.add(scope)
    document = _read_json(path)
    images = document.get("images", [])
    annotations = document.get("annotations", [])
    categories = document.get("categories", [])
    negative_count = sum(1 for image in images if bool(image.get("is_negative")))
    summary = StructuralSummary(
        scope=scope,
        path=str(path),
        sha256=sha256_file(path),
        image_count=len(images),
        annotation_count=len(annotations),
        negative_image_count=negative_count,
        category_ids=tuple(sorted(int(c["id"]) for c in categories)),
        category_names=tuple(str(c["name"])
                             for c in sorted(categories,
                                             key=lambda c: int(c["id"]))),
    )
    del document, images, annotations, categories
    return summary


def negative_image_ids(coco: Mapping[str, Any]) -> Set[int]:
    """Zero-GT negative ("No Finding") image ids of a COCO document."""
    return {int(image["id"]) for image in coco.get("images", [])
            if bool(image.get("is_negative"))}


def image_dimensions(coco: Mapping[str, Any]) -> Dict[int, Tuple[int, int]]:
    return {int(image["id"]): (int(image["width"]), int(image["height"]))
            for image in coco.get("images", [])}


def category_index(coco: Mapping[str, Any]) -> Dict[int, str]:
    return {int(category["id"]): str(category["name"])
            for category in coco.get("categories", [])}


def contains_no_finding_category(coco: Mapping[str, Any]) -> List[str]:
    """Detection-category names that illegally encode 'No Finding'."""
    offenders: List[str] = []
    for category in coco.get("categories", []):
        name = str(category["name"]).strip().lower()
        squashed = name.replace("_", " ").replace("-", " ")
        squashed = " ".join(squashed.split())
        if squashed in ("no finding", "nofinding"):
            offenders.append(str(category["name"]))
    return offenders


def ordered_class_rows(category_names: Mapping[int, str],
                       support: Mapping[int, int]) -> List[int]:
    """Deterministic class order: image_count desc, then category_id asc."""
    return sorted(category_names.keys(),
                  key=lambda cid: (-int(support.get(cid, 0)), int(cid)))


# =====================================================================
# PREFLIGHT
# =====================================================================

def run_preflight(protocol: Mapping[str, Any], protocol_sha256: str,
                  repo_root: Path) -> Tuple[PreflightResult, Dict[str, Any]]:
    """Verify every locked input contract and every locked constant.

    Returns the preflight result plus a context dictionary reused by the
    FULL mode so that the train COCO is parsed only once.
    """
    result = PreflightResult()
    context: Dict[str, Any] = {}

    # ---- protocol identity ------------------------------------------
    result.add(
        "PF01", "protocol_status is RESEARCHER_APPROVED_LOCKED",
        dig(protocol, "phase.protocol_status") == PROTOCOL_STATUS,
        PROTOCOL_STATUS, dig(protocol, "phase.protocol_status"),
        "protocol::phase.protocol_status")
    result.add(
        "PF02", "phase id is 3A",
        str(dig(protocol, "phase.id")) == PHASE_ID,
        PHASE_ID, dig(protocol, "phase.id"), "protocol::phase.id")
    result.add(
        "PF03", "the script may not self-close the phase",
        dig(protocol, "phase_status_policy.script_may_self_close") is False
        and dig(protocol, "phase_status_policy.phase_status_value")
        == PHASE_STATUS_VALUE,
        {"script_may_self_close": False,
         "phase_status_value": PHASE_STATUS_VALUE},
        {"script_may_self_close":
             dig(protocol, "phase_status_policy.script_may_self_close"),
         "phase_status_value":
             dig(protocol, "phase_status_policy.phase_status_value")},
        "protocol::phase_status_policy")

    # ---- PF13 YAML <-> Python lock consistency -----------------------
    # The independently hard-coded constants in this source are
    # authoritative. Any disagreement with the machine-readable YAML fails
    # preflight, so a silent edit of the YAML can never move a locked
    # value, hash, count, threshold or output schema.
    lock_mismatches = yaml_python_lock_mismatches(protocol)
    result.add(
        "PF13", "every machine-readable YAML lock matches the source constants",
        not lock_mismatches, "no mismatch",
        {"mismatch_count": len(lock_mismatches),
         "mismatches": lock_mismatches[:20]},
        "protocol YAML vs scripts/03A_dataset_diagnostics.py constants")

    # ---- PF12 output-path safety gate --------------------------------
    # Resolved BEFORE full mode may create any directory or file. Nothing
    # in this module can write a path that is not registered here.
    reset_validated_output_paths()
    output_paths, output_path_errors = resolve_output_contract(
        protocol, repo_root)
    context["output_paths"] = output_paths
    result.add(
        "PF12", "every declared output target is inside a writable directory",
        not output_path_errors,
        {"writable_directories": list(WRITABLE_OUTPUT_DIRECTORIES),
         "forbidden_directories": list(FORBIDDEN_OUTPUT_DIRECTORIES),
         "rejections": []},
        {"rejections": output_path_errors,
         "resolved_targets": len(output_paths)},
        "protocol::output_contract resolved through resolve_safe_output_path")

    # ---- HF21..HF27 locked thresholds -------------------------------
    result.add(
        "HF21", "primary rare threshold is 0.05",
        float(dig(protocol, "rare_class_definition.primary_threshold"))
        == RARE_PRIMARY_THRESHOLD,
        RARE_PRIMARY_THRESHOLD,
        dig(protocol, "rare_class_definition.primary_threshold"),
        "protocol::rare_class_definition.primary_threshold")
    result.add(
        "HF22", "primary bbox small threshold is 0.01",
        float(dig(protocol, "bbox_size_definition.small_threshold"))
        == BBOX_SMALL_PRIMARY_THRESHOLD,
        BBOX_SMALL_PRIMARY_THRESHOLD,
        dig(protocol, "bbox_size_definition.small_threshold"),
        "protocol::bbox_size_definition.small_threshold")
    result.add(
        "HF23", "primary bbox large threshold is 0.10",
        float(dig(protocol, "bbox_size_definition.large_threshold"))
        == BBOX_LARGE_PRIMARY_THRESHOLD,
        BBOX_LARGE_PRIMARY_THRESHOLD,
        dig(protocol, "bbox_size_definition.large_threshold"),
        "protocol::bbox_size_definition.large_threshold")
    result.add(
        "HF24", "rare sensitivity thresholds are [0.01, 0.05, 0.10]",
        tuple(float(v) for v in
              dig(protocol, "sensitivity_analysis.rare.thresholds", []))
        == RARE_SENSITIVITY_THRESHOLDS,
        list(RARE_SENSITIVITY_THRESHOLDS),
        dig(protocol, "sensitivity_analysis.rare.thresholds"),
        "protocol::sensitivity_analysis.rare.thresholds")
    result.add(
        "HF25", "small sensitivity thresholds are [0.005, 0.01, 0.02]",
        tuple(float(v) for v in
              dig(protocol, "sensitivity_analysis.bbox_small.thresholds", []))
        == BBOX_SMALL_SENSITIVITY_THRESHOLDS,
        list(BBOX_SMALL_SENSITIVITY_THRESHOLDS),
        dig(protocol, "sensitivity_analysis.bbox_small.thresholds"),
        "protocol::sensitivity_analysis.bbox_small.thresholds")
    result.add(
        "HF26", "large sensitivity thresholds are [0.05, 0.10, 0.20]",
        tuple(float(v) for v in
              dig(protocol, "sensitivity_analysis.bbox_large.thresholds", []))
        == BBOX_LARGE_SENSITIVITY_THRESHOLDS,
        list(BBOX_LARGE_SENSITIVITY_THRESHOLDS),
        dig(protocol, "sensitivity_analysis.bbox_large.thresholds"),
        "protocol::sensitivity_analysis.bbox_large.thresholds")
    result.add(
        "HF27", "bbox sensitivity is one-factor-at-a-time",
        dig(protocol, "sensitivity_analysis.one_factor_at_a_time") is True
        and dig(protocol, "sensitivity_analysis.cartesian_grid_allowed")
        is False,
        {"one_factor_at_a_time": True, "cartesian_grid_allowed": False},
        {"one_factor_at_a_time":
             dig(protocol, "sensitivity_analysis.one_factor_at_a_time"),
         "cartesian_grid_allowed":
             dig(protocol, "sensitivity_analysis.cartesian_grid_allowed")},
        "protocol::sensitivity_analysis")

    # sensitivity may never reselect a primary threshold
    result.add(
        "PF04", "sensitivity analysis is secondary and non-gating",
        dig(protocol, "sensitivity_analysis.role") == "SECONDARY_NON_GATING"
        and dig(protocol,
                "sensitivity_analysis.used_for_threshold_selection") is False
        and dig(protocol,
                "sensitivity_analysis.primary_threshold_changed_after_"
                "diagnostics") is False,
        {"role": "SECONDARY_NON_GATING",
         "used_for_threshold_selection": False,
         "primary_threshold_changed_after_diagnostics": False},
        {"role": dig(protocol, "sensitivity_analysis.role"),
         "used_for_threshold_selection":
             dig(protocol,
                 "sensitivity_analysis.used_for_threshold_selection"),
         "primary_threshold_changed_after_diagnostics":
             dig(protocol,
                 "sensitivity_analysis."
                 "primary_threshold_changed_after_diagnostics")},
        "protocol::sensitivity_analysis")

    # ---- definitions are pre-specified -------------------------------
    result.add(
        "PF05", "rare and bbox-size definitions are pre-specified",
        dig(protocol, "rare_class_definition.definition_type")
        == RARE_DEFINITION_TYPE
        and dig(protocol, "bbox_size_definition.definition_type")
        == BBOX_SIZE_DEFINITION_TYPE
        and dig(protocol,
                "rare_class_definition.threshold_may_be_chosen_from_histogram")
        is False
        and dig(protocol, "rare_class_definition.bbox_count_may_define_rarity")
        is False,
        {"rare_definition_type": RARE_DEFINITION_TYPE,
         "bbox_size_definition_type": BBOX_SIZE_DEFINITION_TYPE,
         "threshold_may_be_chosen_from_histogram": False,
         "bbox_count_may_define_rarity": False},
        {"rare_definition_type":
             dig(protocol, "rare_class_definition.definition_type"),
         "bbox_size_definition_type":
             dig(protocol, "bbox_size_definition.definition_type"),
         "threshold_may_be_chosen_from_histogram":
             dig(protocol,
                 "rare_class_definition."
                 "threshold_may_be_chosen_from_histogram"),
         "bbox_count_may_define_rarity":
             dig(protocol,
                 "rare_class_definition.bbox_count_may_define_rarity")},
        "protocol::rare_class_definition, protocol::bbox_size_definition")

    # ---- HF28 / HF30 forbidden scopes are not configured -------------
    configured_paths = collect_protocol_paths(protocol)
    offenders = find_forbidden_unlabeled_paths(configured_paths)
    unlabeled_block = dig(protocol, "input_artifacts.unlabeled_subsets", {})
    result.add(
        "HF28", "no unlabeled-subset ground truth is reachable",
        not offenders and not unlabeled_block
        and dig(protocol, "scope_policy.unlabeled_hidden_gt.usage")
        == "PROHIBITED"
        and dig(protocol,
                "scope_policy.unlabeled_hidden_gt."
                "reverse_lookup_into_train_allowed") is False
        and not [scope for scope in detailed_scopes_used()
                 if scope not in DETAILED_SCOPE_ALLOWED],
        {"forbidden_unlabeled_paths": [], "unlabeled_subsets": {},
         "usage": "PROHIBITED", "reverse_lookup_into_train_allowed": False},
        {"forbidden_unlabeled_paths": offenders,
         "unlabeled_subsets": unlabeled_block,
         "usage": dig(protocol, "scope_policy.unlabeled_hidden_gt.usage"),
         "reverse_lookup_into_train_allowed":
             dig(protocol,
                 "scope_policy.unlabeled_hidden_gt."
                 "reverse_lookup_into_train_allowed"),
         "detailed_scopes_used": list(detailed_scopes_used())},
        "protocol::input_artifacts, protocol::scope_policy.unlabeled_hidden_gt")

    forbidden_ops = dig(protocol, "forbidden_operations", {}) or {}
    switched_on = sorted(
        key for key, value in forbidden_ops.items()
        if key != "all_must_be_false" and value is not False)
    result.add(
        "PF06", "every forbidden operation is disabled",
        not switched_on,
        "all forbidden_operations are false", switched_on,
        "protocol::forbidden_operations")

    result.add(
        "HF30", "no training seed is used",
        dig(protocol, "scope_policy.training_seed_used") is False
        and dig(protocol, "determinism.training_seed_used") is False
        and dig(protocol, "determinism.rng_used") is False
        and dig(protocol, "scope_policy.repeat_over_training_seeds") is False,
        {"scope_policy.training_seed_used": False,
         "determinism.training_seed_used": False,
         "determinism.rng_used": False,
         "repeat_over_training_seeds": False},
        {"scope_policy.training_seed_used":
             dig(protocol, "scope_policy.training_seed_used"),
         "determinism.training_seed_used":
             dig(protocol, "determinism.training_seed_used"),
         "determinism.rng_used": dig(protocol, "determinism.rng_used"),
         "repeat_over_training_seeds":
             dig(protocol, "scope_policy.repeat_over_training_seeds")},
        "protocol::scope_policy, protocol::determinism")

    result.add(
        "PF07", "primary diagnostic scope is the fixed training set",
        dig(protocol, "scope_policy.primary_diagnostic_scope")
        == PRIMARY_DIAGNOSTIC_SCOPE
        and tuple(dig(protocol, "scope_policy.detailed_diagnostics_allowed_on",
                      [])) == DETAILED_SCOPE_ALLOWED
        and dig(protocol, "scope_policy.validation_usage")
        == "STRUCTURAL_INTEGRITY_ONLY"
        and dig(protocol, "scope_policy.test_usage")
        == "STRUCTURAL_INTEGRITY_ONLY",
        {"primary_diagnostic_scope": PRIMARY_DIAGNOSTIC_SCOPE,
         "detailed_diagnostics_allowed_on": list(DETAILED_SCOPE_ALLOWED),
         "validation_usage": "STRUCTURAL_INTEGRITY_ONLY",
         "test_usage": "STRUCTURAL_INTEGRITY_ONLY"},
        {"primary_diagnostic_scope":
             dig(protocol, "scope_policy.primary_diagnostic_scope"),
         "detailed_diagnostics_allowed_on":
             dig(protocol, "scope_policy.detailed_diagnostics_allowed_on"),
         "validation_usage": dig(protocol, "scope_policy.validation_usage"),
         "test_usage": dig(protocol, "scope_policy.test_usage")},
        "protocol::scope_policy")

    # ---- input hashes -----------------------------------------------
    paths = {
        "coco_master": repo_root / dig(
            protocol, "input_artifacts.canonical.coco_master"),
        "coco_master_jpg": repo_root / dig(
            protocol, "input_artifacts.canonical.coco_master_jpg"),
        "train": repo_root / dig(protocol, "input_artifacts.fixed_split.train"),
        "val": repo_root / dig(protocol, "input_artifacts.fixed_split.val"),
        "test": repo_root / dig(protocol, "input_artifacts.fixed_split.test"),
        "split_lock_manifest": repo_root / dig(
            protocol, "input_artifacts.fixed_split.split_lock_manifest"),
    }
    for budget in BUDGET_KEYS:
        paths["labeled_{0}".format(budget)] = repo_root / dig(
            protocol, "input_artifacts.labeled_subsets.{0}".format(budget))
    for key, value in (dig(protocol, "input_artifacts.phase2F_evidence", {})
                       or {}).items():
        paths["phase2F_{0}".format(key)] = repo_root / value
    for key, value in (dig(protocol, "input_artifacts.phase2F1_seed_evidence",
                           {}) or {}).items():
        paths["phase2F1_{0}".format(key)] = repo_root / value

    missing = sorted(name for name, path in paths.items() if not path.is_file())
    result.add(
        "PF08", "every declared input artifact exists",
        not missing, "no missing input artifact", missing,
        "protocol::input_artifacts")
    if missing:
        context["input_hashes"] = {}
        context["paths"] = paths
        return result, context

    input_hashes = {name: sha256_file(path)
                    for name, path in sorted(paths.items())}
    context["input_hashes"] = input_hashes
    context["paths"] = paths

    result.add(
        "HF01", "coco_master_jpg SHA-256 matches the locked value",
        input_hashes["coco_master_jpg"] == MASTER_JPG_SHA256,
        MASTER_JPG_SHA256, input_hashes["coco_master_jpg"],
        str(paths["coco_master_jpg"]))
    result.add(
        "HF02", "instances_train.json SHA-256 matches the locked value",
        input_hashes["train"] == TRAIN_SHA256,
        TRAIN_SHA256, input_hashes["train"], str(paths["train"]))
    result.add(
        "HF03", "instances_val.json SHA-256 matches the locked value",
        input_hashes["val"] == VAL_SHA256,
        VAL_SHA256, input_hashes["val"], str(paths["val"]))
    result.add(
        "HF04", "instances_test.json SHA-256 matches the locked value",
        input_hashes["test"] == TEST_SHA256,
        TEST_SHA256, input_hashes["test"], str(paths["test"]))

    labeled_hash_mismatch = {
        budget: input_hashes["labeled_{0}".format(budget)]
        for budget in BUDGET_KEYS
        if input_hashes["labeled_{0}".format(budget)] != LABELED_SHA256[budget]}
    result.add(
        "HF17", "every labeled subset JSON SHA-256 matches the locked value",
        not labeled_hash_mismatch, LABELED_SHA256, labeled_hash_mismatch or
        {budget: input_hashes["labeled_{0}".format(budget)]
         for budget in BUDGET_KEYS},
        "protocol::expected_hashes.labeled")

    # The canonical non-JPG master has no locked hash; record only.
    result.flags["coco_master_sha256_observed"] = input_hashes["coco_master"]

    # ---- canonical integrity (reference only) ------------------------
    # Registered as a structural-only scope: the canonical master is used
    # for identity, counts and category mapping, never for diagnostics.
    _STRUCTURAL_SCOPES_USED.add("canonical")
    canonical = _read_json(paths["coco_master_jpg"])
    canonical_categories = category_index(canonical)
    canonical_negatives = negative_image_ids(canonical)
    result.add(
        "HF05", "canonical image count is 4894",
        len(canonical.get("images", [])) == CANONICAL_IMAGES,
        CANONICAL_IMAGES, len(canonical.get("images", [])),
        str(paths["coco_master_jpg"]))
    result.add(
        "HF06", "canonical annotation count is 36096",
        len(canonical.get("annotations", [])) == CANONICAL_ANNOTATIONS,
        CANONICAL_ANNOTATIONS, len(canonical.get("annotations", [])),
        str(paths["coco_master_jpg"]))
    result.add(
        "PF09", "canonical No Finding image count is 500",
        len(canonical_negatives) == CANONICAL_NO_FINDING,
        CANONICAL_NO_FINDING, len(canonical_negatives),
        str(paths["coco_master_jpg"]))
    result.add(
        "HF07", "the canonical category count is 14",
        len(canonical_categories) == CANONICAL_CATEGORIES,
        CANONICAL_CATEGORIES, len(canonical_categories),
        str(paths["coco_master_jpg"]))
    result.add(
        "HF08", "category ids are contiguous 1..14",
        sorted(canonical_categories.keys())
        == list(range(1, CANONICAL_CATEGORIES + 1)),
        list(range(1, CANONICAL_CATEGORIES + 1)),
        sorted(canonical_categories.keys()),
        str(paths["coco_master_jpg"]))
    canonical_no_finding_categories = contains_no_finding_category(canonical)
    result.add(
        "HF09", "No Finding does not appear as a detection category",
        not canonical_no_finding_categories,
        [], canonical_no_finding_categories,
        str(paths["coco_master_jpg"]))

    # Canonical No Finding images must be strictly zero-GT. The evidence
    # is computed here and RETAINED; it is reported later as part of the
    # single declared hard-fail record HF10, together with the fixed-train
    # evidence, so that the runtime emits exactly one HF10 record.
    # Integrity checking only; the canonical master is never used for
    # detailed diagnostics.
    canonical_annotated = {int(annotation["image_id"])
                           for annotation in canonical.get("annotations", [])}
    canonical_negative_with_gt = sorted(canonical_negatives
                                        & canonical_annotated)
    del canonical, canonical_annotated

    # ---- fixed train -------------------------------------------------
    train = load_detailed_coco(paths["train"], "train")
    train_categories = category_index(train)
    train_images = train.get("images", [])
    train_annotations = train.get("annotations", [])
    train_negatives = negative_image_ids(train)
    train_image_ids = {int(image["id"]) for image in train_images}
    train_dims = image_dimensions(train)
    annotated_image_ids = {int(a["image_id"]) for a in train_annotations}

    result.add(
        "HF11", "train image count is 3426",
        len(train_images) == TRAIN_IMAGES, TRAIN_IMAGES, len(train_images),
        str(paths["train"]))
    result.add(
        "HF12", "train annotation count is 25260",
        len(train_annotations) == TRAIN_ANNOTATIONS, TRAIN_ANNOTATIONS,
        len(train_annotations), str(paths["train"]))
    result.add(
        "HF13", "train negative (No Finding) image count is 350",
        len(train_negatives) == TRAIN_NO_FINDING, TRAIN_NO_FINDING,
        len(train_negatives), str(paths["train"]))
    result.add(
        "PF10", "train categories are identical to the canonical categories",
        train_categories == canonical_categories,
        canonical_categories, train_categories, str(paths["train"]))
    train_no_finding_categories = contains_no_finding_category(train)
    result.add(
        "HF09b", "No Finding does not appear as a train detection category",
        not train_no_finding_categories, [], train_no_finding_categories,
        str(paths["train"]))

    # HF10 - ONE declared hard-fail record covering both scopes. The
    # canonical evidence was computed above and is reported here together
    # with the fixed-train evidence, so the runtime never emits a code
    # outside the declared HF01..HF35 contract.
    train_negative_with_gt = sorted(train_negatives & annotated_image_ids)
    result.add(
        "HF10",
        "negative images carry zero ground-truth annotations in canonical "
        "and fixed train",
        not canonical_negative_with_gt and not train_negative_with_gt,
        {"canonical_negative_with_gt": [],
         "train_negative_with_gt": []},
        {"canonical_negative_with_gt_count": len(canonical_negative_with_gt),
         "canonical_first_offenders": canonical_negative_with_gt[:20],
         "train_negative_with_gt_count": len(train_negative_with_gt),
         "train_first_offenders": train_negative_with_gt[:20]},
        "coco_master_jpg.json + instances_train.json")

    positives_without_annotations = sorted(
        (train_image_ids - train_negatives) - annotated_image_ids)
    if positives_without_annotations:
        # Scientific/structural observation, not a locked hard-fail.
        result.warn(
            "W01", "images not flagged negative but carrying no annotation",
            [], positives_without_annotations[:20], str(paths["train"]))

    # ---- HF16 bbox validity (train only) ------------------------------
    invalid_records: List[Dict[str, Any]] = []
    for annotation in train_annotations:
        image_id = int(annotation["image_id"])
        if image_id not in train_dims:
            invalid_records.append({
                "annotation_id": int(annotation["id"]),
                "image_id": image_id,
                "violations": ["image_id_not_in_split"]})
            continue
        width, height = train_dims[image_id]
        x, y, w, h = (float(v) for v in annotation["bbox"])
        violations = bbox_validity_violations(x, y, w, h, width, height)
        if violations:
            invalid_records.append({
                "annotation_id": int(annotation["id"]),
                "image_id": image_id,
                "bbox": [x, y, w, h],
                "image_width": width,
                "image_height": height,
                "violations": violations})
    result.add(
        "HF16", "every train bounding box satisfies the locked validity rules",
        not invalid_records, 0,
        {"invalid_annotation_count": len(invalid_records),
         "first_offenders": invalid_records[:10]},
        str(paths["train"]))

    # ---- val / test STRUCTURAL ONLY -----------------------------------
    val_summary = structural_summary(paths["val"], "val")
    test_summary = structural_summary(paths["test"], "test")
    result.add(
        "HF14", "validation structural counts match the locked values",
        (val_summary.image_count == VAL_IMAGES
         and val_summary.annotation_count == VAL_ANNOTATIONS
         and val_summary.negative_image_count == VAL_NO_FINDING
         and list(val_summary.category_ids)
         == list(range(1, CANONICAL_CATEGORIES + 1))),
        {"images": VAL_IMAGES, "annotations": VAL_ANNOTATIONS,
         "no_finding": VAL_NO_FINDING,
         "category_ids": list(range(1, CANONICAL_CATEGORIES + 1))},
        {"images": val_summary.image_count,
         "annotations": val_summary.annotation_count,
         "no_finding": val_summary.negative_image_count,
         "category_ids": list(val_summary.category_ids)},
        str(paths["val"]))
    result.add(
        "HF15", "test structural counts match the locked values",
        (test_summary.image_count == TEST_IMAGES
         and test_summary.annotation_count == TEST_ANNOTATIONS
         and test_summary.negative_image_count == TEST_NO_FINDING
         and list(test_summary.category_ids)
         == list(range(1, CANONICAL_CATEGORIES + 1))),
        {"images": TEST_IMAGES, "annotations": TEST_ANNOTATIONS,
         "no_finding": TEST_NO_FINDING,
         "category_ids": list(range(1, CANONICAL_CATEGORIES + 1))},
        {"images": test_summary.image_count,
         "annotations": test_summary.annotation_count,
         "no_finding": test_summary.negative_image_count,
         "category_ids": list(test_summary.category_ids)},
        str(paths["test"]))

    # ---- labeled subsets ----------------------------------------------
    labeled_docs: Dict[str, Dict[str, Any]] = {}
    labeled_image_id_sets: Dict[str, Set[int]] = {}
    count_mismatch: Dict[str, Any] = {}
    outside_train: Dict[str, List[int]] = {}
    for budget in BUDGET_KEYS:
        document = load_detailed_coco(paths["labeled_{0}".format(budget)],
                                      "labeled_{0}".format(budget))
        labeled_docs[budget] = document
        ids = {int(image["id"]) for image in document.get("images", [])}
        labeled_image_id_sets[budget] = ids
        negatives = negative_image_ids(document)
        if (len(ids) != LABELED_IMAGE_COUNTS[budget]
                or len(negatives) != LABELED_NO_FINDING_COUNTS[budget]):
            count_mismatch[budget] = {
                "images": len(ids), "no_finding": len(negatives)}
        stray = sorted(ids - train_image_ids)
        if stray:
            outside_train[budget] = stray[:20]

    result.add(
        "HF18", "labeled image and negative counts match the locked values",
        not count_mismatch,
        {budget: {"images": LABELED_IMAGE_COUNTS[budget],
                  "no_finding": LABELED_NO_FINDING_COUNTS[budget]}
         for budget in BUDGET_KEYS},
        count_mismatch, "protocol::expected_counts.labeled")
    result.add(
        "HF20", "every labeled image belongs to the fixed training set",
        not outside_train, {}, outside_train,
        "labeled subsets vs instances_train.json")
    nested_ok = is_nested([labeled_image_id_sets[b] for b in BUDGET_KEYS])
    result.add(
        "HF19", "labeled subsets are nested 1pct in 5pct in 10pct in 20pct",
        nested_ok,
        "L_1pct subset_of L_5pct subset_of L_10pct subset_of L_20pct",
        {"1pct_in_5pct": labeled_image_id_sets["1pct"].issubset(
            labeled_image_id_sets["5pct"]),
         "5pct_in_10pct": labeled_image_id_sets["5pct"].issubset(
             labeled_image_id_sets["10pct"]),
         "10pct_in_20pct": labeled_image_id_sets["10pct"].issubset(
             labeled_image_id_sets["20pct"])},
        "labeled subsets")

    # ---- Phase 2F.1 seed evidence: training has not started -----------
    seed_state_path = paths.get("phase2F1_seed_state_manifest")
    seed_state: Optional[Dict[str, Any]] = None
    if seed_state_path is not None and seed_state_path.is_file():
        seed_state = _read_json(seed_state_path)
        context["seed_state"] = seed_state
        result.add(
            "PF11", "Phase 2F.1 seed state confirms training has not started",
            seed_state.get("training_started") is False
            and seed_state.get("runs") == [],
            {"training_started": False, "runs": []},
            {"training_started": seed_state.get("training_started"),
             "runs": seed_state.get("runs")},
            str(seed_state_path))

    # ---- firewall tripwire -------------------------------------------
    illegal_detailed = sorted(
        scope for scope in detailed_scopes_used()
        if scope not in DETAILED_SCOPE_ALLOWED)
    result.add(
        "HF29", "no detailed diagnostics were opened for validation or test",
        not illegal_detailed and "val" not in detailed_scopes_used()
        and "test" not in detailed_scopes_used(),
        [], {"detailed_scopes_used": list(detailed_scopes_used()),
             "illegal": illegal_detailed},
        "runtime firewall tripwire")

    # ---- PF14 declared hard-fail contract ----------------------------
    # The runtime must never emit a hard-fail code that the protocol does
    # not declare in guardrails.hard_fail_conditions.
    declared_in_yaml = sorted(
        (dig(protocol, "guardrails.hard_fail_conditions", {}) or {}).keys())
    emitted_codes = [check.code for check in result.checks]
    undeclared = undeclared_hard_fail_codes(emitted_codes)
    result.add(
        "PF14", "every emitted hard-fail code is declared by the protocol",
        not undeclared and declared_in_yaml == list(DECLARED_HARD_FAIL_CODES),
        {"declared_hard_fail_codes": list(DECLARED_HARD_FAIL_CODES),
         "permitted_subcodes": dict(DECLARED_HARD_FAIL_SUBCODES),
         "undeclared": []},
        {"declared_in_yaml": declared_in_yaml,
         "undeclared": undeclared,
         "emitted_hard_fail_codes": sorted(
             {code for code in emitted_codes if str(code).startswith("HF")})},
        "protocol::guardrails.hard_fail_conditions vs runtime checks")

    # ---- context for FULL mode ---------------------------------------
    context.update({
        "train": train,
        "train_categories": train_categories,
        "train_images": train_images,
        "train_annotations": train_annotations,
        "train_negatives": train_negatives,
        "train_image_ids": train_image_ids,
        "train_dims": train_dims,
        "labeled_docs": labeled_docs,
        "labeled_image_id_sets": labeled_image_id_sets,
        "val_summary": val_summary,
        "test_summary": test_summary,
        "protocol_sha256": protocol_sha256,
        # Snapshot used by HF31 / HF32 / HF33 to distinguish artifacts that
        # already existed from artifacts created by this run.
        "pre_run_training_artifacts": detect_training_artifacts(repo_root),
    })
    context.setdefault("seed_state", None)
    return result, context


# =====================================================================
# DIAGNOSTICS (FULL mode)
# =====================================================================

def build_class_distribution(context: Mapping[str, Any]
                             ) -> Tuple[List[Dict[str, Any]], List[int]]:
    """Per-class image support, prevalence, bbox annotation count, rare flag."""
    categories: Dict[int, str] = context["train_categories"]
    annotations = context["train_annotations"]
    support_sets = class_image_support(annotations)
    bbox_counts = class_bbox_counts(annotations)
    total_annotations = len(annotations)

    support_counts = {cid: len(support_sets.get(cid, set()))
                      for cid in categories}
    order = ordered_class_rows(categories, support_counts)

    rows: List[Dict[str, Any]] = []
    for category_id in order:
        image_count = support_counts[category_id]
        prevalence = image_prevalence(image_count, TRAIN_IMAGES)
        bbox_count = bbox_counts.get(category_id, 0)
        rows.append({
            "category_id": category_id,
            "class_name": categories[category_id],
            "image_count": image_count,
            "image_prevalence": prevalence,
            "bbox_annotation_count": bbox_count,
            "bbox_annotation_share": (float(bbox_count) / float(total_annotations)
                                      if total_annotations else None),
            "bbox_per_positive_image": (float(bbox_count) / float(image_count)
                                        if image_count else None),
            "rare_flag": is_rare(prevalence, RARE_PRIMARY_THRESHOLD),
        })
    return rows, order


def build_class_imbalance(class_rows: Sequence[Mapping[str, Any]]
                          ) -> List[Dict[str, Any]]:
    supports = [int(row["image_count"]) for row in class_rows]
    prevalences = [float(row["image_prevalence"]) for row in class_rows]
    rare_rows = [row for row in class_rows if row["rare_flag"]]
    rare_names = sorted(str(row["class_name"]) for row in rare_rows)
    metrics: List[Tuple[str, Any, str]] = [
        ("max_image_support", max(supports) if supports else None, "count"),
        ("min_image_support", min(supports) if supports else None, "count"),
        ("median_image_support", percentile(supports, 50), "count"),
        ("mean_image_support",
         float(np.mean(supports)) if supports else None, "count"),
        ("max_image_prevalence", max(prevalences) if prevalences else None,
         "fraction"),
        ("min_image_prevalence", min(prevalences) if prevalences else None,
         "fraction"),
        ("imbalance_ratio_max_over_min", imbalance_ratio(supports), "ratio"),
        ("coefficient_of_variation_image_support",
         coefficient_of_variation(supports), "ratio (secondary)"),
        ("rare_class_count", len(rare_rows), "count"),
        ("rare_class_list", "; ".join(rare_names), "class names"),
        ("rare_threshold_used", RARE_PRIMARY_THRESHOLD, "locked primary"),
        ("total_train_images", TRAIN_IMAGES, "count"),
    ]
    return [{"metric": name, "value": value, "unit": unit}
            for name, value, unit in metrics]


def build_bbox_tables(context: Mapping[str, Any],
                      class_name_by_id: Mapping[int, str]
                      ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Annotation-level derived table plus the continuous summary table."""
    annotations = context["train_annotations"]
    dims = context["train_dims"]
    rows: List[Dict[str, Any]] = []
    for annotation in annotations:
        image_id = int(annotation["image_id"])
        width, height = dims[image_id]
        x, y, w, h = (float(v) for v in annotation["bbox"])
        area_n = normalized_area(w, h, width, height)
        center_x, center_y = bbox_center_normalized(x, y, w, h, width, height)
        rows.append({
            "annotation_id": int(annotation["id"]),
            "image_id": image_id,
            "category_id": int(annotation["category_id"]),
            "class_name": class_name_by_id[int(annotation["category_id"])],
            "x": x, "y": y, "w": w, "h": h,
            "image_width": width, "image_height": height,
            "raw_area": w * h,
            "normalized_width": normalized_width(w, width),
            "normalized_height": normalized_height(h, height),
            "normalized_area": area_n,
            "aspect_ratio": aspect_ratio(w, h),
            "center_x_normalized": center_x,
            "center_y_normalized": center_y,
            "primary_size_category": classify_bbox_size(
                area_n, BBOX_SMALL_PRIMARY_THRESHOLD,
                BBOX_LARGE_PRIMARY_THRESHOLD),
        })
    rows.sort(key=lambda row: (row["image_id"], row["annotation_id"]))

    summary_rows: List[Dict[str, Any]] = []
    variables = [
        ("normalized_area", "primary"),
        ("normalized_width", "primary"),
        ("normalized_height", "primary"),
        ("aspect_ratio", "primary"),
        ("raw_area", "secondary"),
        ("w", "secondary"),
        ("h", "secondary"),
    ]
    for variable, role in variables:
        stats = continuous_stats([row[variable] for row in rows])
        record = {"variable": variable, "role": role, "scope": "fixed_train"}
        record.update(stats)
        summary_rows.append(record)
    return rows, summary_rows


def build_bbox_count_per_image(context: Mapping[str, Any]
                               ) -> List[Dict[str, Any]]:
    image_ids = sorted(context["train_image_ids"])
    negatives = context["train_negatives"]
    counts = bbox_count_per_image(context["train_annotations"], image_ids)
    all_values = [counts[i] for i in image_ids]
    positive_values = [counts[i] for i in image_ids if i not in negatives]
    rows: List[Dict[str, Any]] = []
    for population, values in (("all_train_images", all_values),
                               ("positive_train_images_only", positive_values)):
        stats = continuous_stats(values)
        rows.append({
            "population": population,
            "n": stats["n"],
            "mean": stats["mean"],
            "sd_ddof1": stats["sd"],
            "median": stats["median"],
            "p95": stats["p95"],
            "max": stats["max"],
        })
    return rows


def build_negative_distribution(context: Mapping[str, Any]
                                ) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    train_total = len(context["train_image_ids"])
    train_negative = len(context["train_negatives"])
    rows.append({
        "scope": "train",
        "diagnostic_role": "PRIMARY_DETAILED",
        "image_count": train_total,
        "positive_image_count": train_total - train_negative,
        "negative_image_count": train_negative,
        "negative_prevalence": (float(train_negative) / float(train_total)
                                if train_total else None),
    })
    for summary in (context["val_summary"], context["test_summary"]):
        rows.append({
            "scope": summary.scope,
            "diagnostic_role": "STRUCTURAL_ONLY",
            "image_count": summary.image_count,
            "positive_image_count": (summary.image_count
                                     - summary.negative_image_count),
            "negative_image_count": summary.negative_image_count,
            "negative_prevalence": (
                float(summary.negative_image_count) / float(summary.image_count)
                if summary.image_count else None),
        })
    for budget in BUDGET_KEYS:
        document = context["labeled_docs"][budget]
        total = len(document.get("images", []))
        negative = len(negative_image_ids(document))
        rows.append({
            "scope": "labeled_{0}".format(budget),
            "diagnostic_role": "LABELED_SUBSET_DETAILED",
            "image_count": total,
            "positive_image_count": total - negative,
            "negative_image_count": negative,
            "negative_prevalence": (float(negative) / float(total)
                                    if total else None),
        })
    return rows


def build_label_cardinality(context: Mapping[str, Any]
                            ) -> List[Dict[str, Any]]:
    image_ids = sorted(context["train_image_ids"])
    presence = image_class_presence(context["train_annotations"])
    cardinality = label_cardinality_per_image(presence, image_ids)
    values = [cardinality[i] for i in image_ids]
    total = len(values)

    histogram: Dict[int, int] = {}
    for value in values:
        histogram[value] = histogram.get(value, 0) + 1

    rows: List[Dict[str, Any]] = []
    for value in sorted(histogram):
        rows.append({
            "record_type": "distribution",
            "cardinality": value,
            "image_count": histogram[value],
            "percentage": (100.0 * histogram[value] / total) if total else None,
            "statistic": "",
            "statistic_value": "",
        })
    stats = continuous_stats(values)
    for name, key in (("mean", "mean"), ("sd_ddof1", "sd"),
                      ("median", "median"), ("p95", "p95"), ("max", "max")):
        rows.append({
            "record_type": "summary",
            "cardinality": "",
            "image_count": "",
            "percentage": "",
            "statistic": name,
            "statistic_value": stats[key],
        })
    return rows


def build_cooccurrence(context: Mapping[str, Any]) -> List[Dict[str, Any]]:
    categories: Dict[int, str] = context["train_categories"]
    presence = image_class_presence(context["train_annotations"])
    support = {cid: len(images) for cid, images
               in class_image_support(context["train_annotations"]).items()}
    ids = sorted(categories)
    rows: List[Dict[str, Any]] = []
    for index_a, category_a in enumerate(ids):
        for category_b in ids[index_a + 1:]:
            n_a = support.get(category_a, 0)
            n_b = support.get(category_b, 0)
            n_both = cooccurrence_count(presence, category_a, category_b)
            rows.append({
                "category_id_a": category_a,
                "class_name_a": categories[category_a],
                "category_id_b": category_b,
                "class_name_b": categories[category_b],
                "n_a_images": n_a,
                "n_b_images": n_b,
                "n_both_images": n_both,
                "n_union_images": n_a + n_b - n_both,
                "jaccard": jaccard(n_a, n_b, n_both),
            })
    rows.sort(key=lambda row: (row["category_id_a"], row["category_id_b"]))
    return rows


def build_labeled_budget_coverage(context: Mapping[str, Any],
                                  class_rows: Sequence[Mapping[str, Any]]
                                  ) -> List[Dict[str, Any]]:
    categories: Dict[int, str] = context["train_categories"]
    train_prevalence = {int(row["category_id"]): float(row["image_prevalence"])
                        for row in class_rows}
    train_rare = {int(row["category_id"]): bool(row["rare_flag"])
                  for row in class_rows}
    train_support = {int(row["category_id"]): int(row["image_count"])
                     for row in class_rows}

    rows: List[Dict[str, Any]] = []
    for budget in BUDGET_KEYS:
        document = context["labeled_docs"][budget]
        images = document.get("images", [])
        annotations = document.get("annotations", [])
        total = len(images)
        negative = len(negative_image_ids(document))
        support_sets = class_image_support(annotations)
        support = {cid: len(support_sets.get(cid, set())) for cid in categories}
        coverage = sum(1 for cid in categories if support.get(cid, 0) > 0)
        deviations = {
            cid: abs(image_prevalence(support.get(cid, 0), total)
                     - train_prevalence[cid]) for cid in categories} if total \
            else {cid: None for cid in categories}
        numeric_deviations = [value for value in deviations.values()
                              if value is not None]
        max_deviation = max(numeric_deviations) if numeric_deviations else None
        mean_deviation = (float(np.mean(numeric_deviations))
                          if numeric_deviations else None)
        for category_id in sorted(categories):
            count = support.get(category_id, 0)
            prevalence = image_prevalence(count, total) if total else None
            rows.append({
                "budget": budget,
                "labeled_image_count": total,
                "negative_image_count": negative,
                "negative_prevalence": (float(negative) / float(total)
                                        if total else None),
                "class_coverage_out_of_14": coverage,
                "category_id": category_id,
                "class_name": categories[category_id],
                "image_count": count,
                "image_prevalence": prevalence,
                "train_image_count": train_support[category_id],
                "train_image_prevalence": train_prevalence[category_id],
                "absolute_deviation_from_train": deviations[category_id],
                # Rare status always comes from the fixed train set.
                "train_rare_flag": train_rare[category_id],
                "rare_class_support": count if train_rare[category_id] else 0,
                "coverage_present": count > 0,
                "budget_max_absolute_deviation": max_deviation,
                "budget_mean_absolute_deviation": mean_deviation,
            })
    return rows


def build_threshold_sensitivity(context: Mapping[str, Any],
                                class_rows: Sequence[Mapping[str, Any]],
                                bbox_rows: Sequence[Mapping[str, Any]]
                                ) -> List[Dict[str, Any]]:
    """SECONDARY / NON-GATING robustness check.

    [LOCKED SCIENTIFIC DECISION] One factor at a time. This function
    never mutates the primary thresholds; it only re-labels copies.
    """
    rows: List[Dict[str, Any]] = []

    # ---- 20.1 rare sensitivity ---------------------------------------
    flags_by_threshold: Dict[float, Dict[int, bool]] = {}
    for threshold in RARE_SENSITIVITY_THRESHOLDS:
        flags_by_threshold[threshold] = {
            int(row["category_id"]):
                is_rare(float(row["image_prevalence"]), threshold)
            for row in class_rows}

    for threshold in RARE_SENSITIVITY_THRESHOLDS:
        is_primary = (threshold == RARE_PRIMARY_THRESHOLD)
        rare_count = 0
        for row in sorted(class_rows, key=lambda r: int(r["category_id"])):
            category_id = int(row["category_id"])
            flag = flags_by_threshold[threshold][category_id]
            primary_flag = bool(row["rare_flag"])
            statuses = [flags_by_threshold[t][category_id]
                        for t in RARE_SENSITIVITY_THRESHOLDS]
            rare_count += 1 if flag else 0
            rows.append({
                "analysis_type": "rare_threshold_per_class",
                "parameter": "rare_image_prevalence_threshold",
                "threshold": threshold,
                "is_primary": is_primary,
                "category": "rare" if flag else "non_rare",
                "count": int(row["image_count"]),
                "percentage": 100.0 * float(row["image_prevalence"]),
                "primary_percentage": 100.0 * float(row["image_prevalence"]),
                "delta_percentage_points": 0.0,
                "category_id": category_id,
                "class_name": row["class_name"],
                "image_count": int(row["image_count"]),
                "image_prevalence": float(row["image_prevalence"]),
                "rare_flag": flag,
                "primary_rare_flag": primary_flag,
                "status_changed_vs_primary": flag != primary_flag,
                "stable_rare_flag": all(statuses),
                "stable_nonrare_flag": not any(statuses),
                "threshold_sensitive_flag": len(set(statuses)) > 1,
            })
        primary_rare_count = sum(1 for row in class_rows if row["rare_flag"])
        total_classes = len(class_rows)
        rows.append({
            "analysis_type": "rare_threshold_summary",
            "parameter": "rare_image_prevalence_threshold",
            "threshold": threshold,
            "is_primary": is_primary,
            "category": "rare_class_count",
            "count": rare_count,
            "percentage": (100.0 * rare_count / total_classes
                           if total_classes else None),
            "primary_percentage": (100.0 * primary_rare_count / total_classes
                                   if total_classes else None),
            "delta_percentage_points": (
                100.0 * (rare_count - primary_rare_count) / total_classes
                if total_classes else None),
            "category_id": "", "class_name": "", "image_count": "",
            "image_prevalence": "", "rare_flag": "", "primary_rare_flag": "",
            "status_changed_vs_primary": "", "stable_rare_flag": "",
            "stable_nonrare_flag": "", "threshold_sensitive_flag": "",
        })

    # ---- 20.2 / 20.3 bbox boundary sensitivity ------------------------
    areas = [float(row["normalized_area"]) for row in bbox_rows]
    total_boxes = len(areas)
    primary_counts = size_category_counts(
        areas, BBOX_SMALL_PRIMARY_THRESHOLD, BBOX_LARGE_PRIMARY_THRESHOLD)

    def append_size_rows(analysis_type: str, parameter: str, threshold: float,
                         small_threshold: float, large_threshold: float,
                         is_primary: bool) -> None:
        counts = size_category_counts(areas, small_threshold, large_threshold)
        for category in BBOX_SIZE_CATEGORIES:
            percentage = (100.0 * counts[category] / total_boxes
                          if total_boxes else None)
            primary_percentage = (100.0 * primary_counts[category] / total_boxes
                                  if total_boxes else None)
            rows.append({
                "analysis_type": analysis_type,
                "parameter": parameter,
                "threshold": threshold,
                "is_primary": is_primary,
                "category": category,
                "count": counts[category],
                "percentage": percentage,
                "primary_percentage": primary_percentage,
                "delta_percentage_points": (
                    percentage - primary_percentage
                    if percentage is not None and primary_percentage is not None
                    else None),
                "category_id": "", "class_name": "", "image_count": "",
                "image_prevalence": "", "rare_flag": "",
                "primary_rare_flag": "", "status_changed_vs_primary": "",
                "stable_rare_flag": "", "stable_nonrare_flag": "",
                "threshold_sensitive_flag": "",
            })

    # One factor at a time: vary tau_small, keep tau_large fixed at 0.10.
    for threshold in BBOX_SMALL_SENSITIVITY_THRESHOLDS:
        append_size_rows(
            "bbox_small_boundary_sensitivity", "bbox_small_threshold",
            threshold, threshold, BBOX_LARGE_PRIMARY_THRESHOLD,
            threshold == BBOX_SMALL_PRIMARY_THRESHOLD)
    # One factor at a time: vary tau_large, keep tau_small fixed at 0.01.
    for threshold in BBOX_LARGE_SENSITIVITY_THRESHOLDS:
        append_size_rows(
            "bbox_large_boundary_sensitivity", "bbox_large_threshold",
            threshold, BBOX_SMALL_PRIMARY_THRESHOLD, threshold,
            threshold == BBOX_LARGE_PRIMARY_THRESHOLD)

    return rows


def build_split_distribution(context: Mapping[str, Any],
                             repo_root: Path,
                             protocol: Mapping[str, Any]
                             ) -> List[Dict[str, Any]]:
    lock_path = repo_root / dig(
        protocol, "input_artifacts.fixed_split.split_lock_manifest")
    membership = {}
    if lock_path.is_file():
        membership = _read_json(lock_path).get("image_id_sha256", {}) or {}

    train_total = len(context["train_image_ids"])
    train_negative = len(context["train_negatives"])
    total_images = (train_total + context["val_summary"].image_count
                    + context["test_summary"].image_count)
    entries = [
        ("train", train_total, len(context["train_annotations"]),
         train_negative, context["input_hashes"]["train"]),
        ("val", context["val_summary"].image_count,
         context["val_summary"].annotation_count,
         context["val_summary"].negative_image_count,
         context["val_summary"].sha256),
        ("test", context["test_summary"].image_count,
         context["test_summary"].annotation_count,
         context["test_summary"].negative_image_count,
         context["test_summary"].sha256),
    ]
    rows: List[Dict[str, Any]] = []
    for split, images, annotation_count, negatives, checksum in entries:
        rows.append({
            "split": split,
            "image_count": images,
            "image_percentage": (100.0 * images / total_images
                                 if total_images else None),
            "annotation_count": annotation_count,
            "negative_count": negatives,
            "negative_percentage": (100.0 * negatives / images
                                    if images else None),
            "official_json_sha256": checksum,
            "image_membership_sha256": membership.get(split, ""),
            "diagnostic_usage": ("PRIMARY_DETAILED" if split == "train"
                                 else "STRUCTURAL_INTEGRITY_ONLY"),
        })
    return rows


# =====================================================================
# CSV writing  [IMPLEMENTATION DETAIL]
# =====================================================================

def write_csv(path: Path, fieldnames: Sequence[str],
              rows: Iterable[Mapping[str, Any]]) -> None:
    _assert_validated_output(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(list(fieldnames))
        for row in rows:
            writer.writerow([_csv_cell(row.get(name)) for name in fieldnames])


def _csv_cell(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return fmt_bool(value)
    if isinstance(value, (int, float, np.integer, np.floating)):
        return fmt_num(value)
    return str(value)


# =====================================================================
# PLOTS  [VISUALIZATION DETAIL]
# Rules applied: one axis per panel (never a second y-scale), a fixed
# deterministic category order that is identical across panels, a
# single-hue sequential ramp for density, recessive grid, explicit axis
# labels and captions. Colours carry no scientific meaning.
# =====================================================================

def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def plot_class_distribution(class_rows: Sequence[Mapping[str, Any]],
                            output_path: Path, style: Mapping[str, Any]
                            ) -> None:
    plt = _plt()
    names = [str(row["class_name"]) for row in class_rows]
    image_counts = [int(row["image_count"]) for row in class_rows]
    prevalences = [float(row["image_prevalence"]) for row in class_rows]
    bbox_counts = [int(row["bbox_annotation_count"]) for row in class_rows]
    positions = list(range(len(names)))[::-1]  # same order in both panels

    figure, axes = plt.subplots(1, 2, figsize=(14, 7), sharey=True)
    axis_a, axis_b = axes

    axis_a.barh(positions, image_counts, color=style["bar_color"])
    axis_a.set_yticks(positions)
    axis_a.set_yticklabels(names)
    axis_a.set_xlabel("Number of unique training images containing the class")
    axis_a.set_title("Panel A - image-level support (fixed train, n = "
                     "{0} images)".format(TRAIN_IMAGES))
    axis_a.grid(axis="x", alpha=style["grid_alpha"])
    axis_a.set_axisbelow(True)
    span_a = max(image_counts) if image_counts else 1
    for position, count, prevalence in zip(positions, image_counts,
                                           prevalences):
        axis_a.text(count + 0.01 * span_a, position,
                    "{0} ({1:.1f}%)".format(count, 100.0 * prevalence),
                    va="center", fontsize=8)
    axis_a.set_xlim(0, span_a * 1.22)

    axis_b.barh(positions, bbox_counts, color=style["bar_color_secondary"])
    axis_b.set_xlabel("Bounding-box annotation count")
    axis_b.set_title("Panel B - bounding-box annotation count (fixed train)")
    axis_b.grid(axis="x", alpha=style["grid_alpha"])
    axis_b.set_axisbelow(True)
    span_b = max(bbox_counts) if bbox_counts else 1
    for position, count in zip(positions, bbox_counts):
        axis_b.text(count + 0.01 * span_b, position, str(count),
                    va="center", fontsize=8)
    axis_b.set_xlim(0, span_b * 1.22)

    figure.suptitle("Phase 3A - class distribution on the fixed training set",
                    fontsize=13)
    figure.text(0.5, 0.005,
                "Panels share the same class order (image support descending, "
                "category_id ascending). Panel B counts bounding-box "
                "annotation records, not unique clinical lesions.",
                ha="center", fontsize=8)
    figure.tight_layout(rect=(0, 0.03, 1, 0.96))
    _assert_validated_output(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=style["dpi"])
    plt.close(figure)


def plot_bbox_distribution(areas: Sequence[float],
                           size_counts: Mapping[str, int],
                           output_path: Path, style: Mapping[str, Any]
                           ) -> None:
    plt = _plt()
    figure, axes = plt.subplots(1, 2, figsize=(14, 6))
    axis_a, axis_b = axes

    bins = freedman_diaconis_bin_count(
        areas, style["histogram_min_bins"], style["histogram_max_bins"])
    axis_a.hist(areas, bins=bins, color=style["bar_color"])
    axis_a.axvline(BBOX_SMALL_PRIMARY_THRESHOLD, color="#333333",
                   linestyle="--", linewidth=1.2,
                   label="locked small threshold = 0.01")
    axis_a.axvline(BBOX_LARGE_PRIMARY_THRESHOLD, color="#333333",
                   linestyle=":", linewidth=1.2,
                   label="locked large threshold = 0.10")
    axis_a.set_xlabel("Normalized bounding-box area  a_n = (w*h)/(W*H)")
    axis_a.set_ylabel("Bounding-box annotation count")
    axis_a.set_title("Panel A - continuous normalized bbox-area distribution")
    axis_a.set_yscale("log")
    axis_a.grid(axis="y", alpha=style["grid_alpha"])
    axis_a.set_axisbelow(True)
    axis_a.legend(fontsize=8)

    total = sum(size_counts.values())
    categories = list(BBOX_SIZE_CATEGORIES)
    counts = [size_counts[name] for name in categories]
    shades = [style["bar_color_secondary"], style["bar_color"], "#1F3F66"]
    axis_b.bar(categories, counts, color=shades)
    axis_b.set_xlabel("Normalized-area-based bbox size category")
    axis_b.set_ylabel("Bounding-box annotation count")
    axis_b.set_title("Panel B - Small / Medium / Large "
                     "(normalized-area-based, not COCO pixel-area)")
    axis_b.grid(axis="y", alpha=style["grid_alpha"])
    axis_b.set_axisbelow(True)
    span = max(counts) if counts else 1
    for index, count in enumerate(counts):
        percentage = (100.0 * count / total) if total else 0.0
        axis_b.text(index, count + 0.02 * span,
                    "{0}\n({1:.2f}%)".format(count, percentage),
                    ha="center", fontsize=9)
    axis_b.set_ylim(0, span * 1.18)

    figure.suptitle("Phase 3A - bounding-box size distribution "
                    "(fixed training set)", fontsize=13)
    figure.text(0.5, 0.005,
                "Histogram binning is a visualization detail "
                "(Freedman-Diaconis) and does not determine any threshold. "
                "The 0.01 and 0.10 boundaries were pre-specified.",
                ha="center", fontsize=8)
    figure.tight_layout(rect=(0, 0.04, 1, 0.95))
    _assert_validated_output(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=style["dpi"])
    plt.close(figure)


def plot_bbox_location_heatmap(centers_x: Sequence[float],
                               centers_y: Sequence[float],
                               output_path: Path, style: Mapping[str, Any]
                               ) -> None:
    plt = _plt()
    histogram, _, _ = np.histogram2d(
        np.asarray(centers_x, dtype=float), np.asarray(centers_y, dtype=float),
        bins=[HEATMAP_BINS_X, HEATMAP_BINS_Y],
        range=[[0.0, 1.0], [0.0, 1.0]])

    figure, axis = plt.subplots(figsize=(8, 7.6))
    # extent = [left, right, bottom, top]; top-left origin, y downward.
    image = axis.imshow(histogram.T, origin="upper", extent=(0.0, 1.0, 1.0, 0.0),
                        cmap=style["heatmap_cmap"], aspect="equal",
                        interpolation="nearest")
    axis.set_xlabel("Normalized bbox centre x = (x + w/2) / W  "
                    "(0 = image left, 1 = image right)")
    axis.set_ylabel("Normalized bbox centre y = (y + h/2) / H  "
                    "(0 = image top, 1 = image bottom)")
    axis.set_title("Phase 3A - bbox-centre density, fixed training set\n"
                   "{0} x {1} bins, annotation-weighted".format(
                       HEATMAP_BINS_X, HEATMAP_BINS_Y))
    colorbar = figure.colorbar(image, ax=axis, shrink=0.85)
    colorbar.set_label("Bounding-box annotation count per cell")
    figure.text(0.5, 0.01,
                "The heatmap describes bounding-box annotations. It is not by "
                "itself a density of unique clinical lesions.",
                ha="center", fontsize=8)
    figure.tight_layout(rect=(0, 0.04, 1, 1))
    _assert_validated_output(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=style["dpi"])
    plt.close(figure)


def plot_negative_distribution(positive_count: int, negative_count: int,
                               output_path: Path, style: Mapping[str, Any]
                               ) -> None:
    plt = _plt()
    total = positive_count + negative_count
    labels = ["Positive / abnormal", "No Finding / negative"]
    counts = [positive_count, negative_count]

    figure, axis = plt.subplots(figsize=(7.5, 6))
    axis.bar(labels, counts,
             color=[style["bar_color"], style["bar_color_secondary"]])
    axis.set_ylabel("Number of training images")
    axis.set_title("Phase 3A - positive vs zero-GT negative images\n"
                   "fixed training set (n = {0})".format(total))
    axis.grid(axis="y", alpha=style["grid_alpha"])
    axis.set_axisbelow(True)
    span = max(counts) if counts else 1
    for index, count in enumerate(counts):
        percentage = (100.0 * count / total) if total else 0.0
        axis.text(index, count + 0.02 * span,
                  "{0}\n({1:.2f}%)".format(count, percentage),
                  ha="center", fontsize=10)
    axis.set_ylim(0, span * 1.18)
    figure.text(0.5, 0.01,
                "No Finding is a zero-ground-truth negative image, not a "
                "detection category.", ha="center", fontsize=8)
    figure.tight_layout(rect=(0, 0.04, 1, 1))
    _assert_validated_output(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=style["dpi"])
    plt.close(figure)


# =====================================================================
# REPORT
# =====================================================================

def _md_table(headers: Sequence[str],
              rows: Iterable[Sequence[Any]]) -> List[str]:
    lines = ["| " + " | ".join(str(h) for h in headers) + " |",
             "| " + " | ".join("---" for _ in headers) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(_csv_cell(cell) for cell in row) + " |")
    return lines


def build_report(protocol: Mapping[str, Any], protocol_sha256: str,
                 preflight: PreflightResult, context: Mapping[str, Any],
                 tables: Mapping[str, Any], generated_at: str) -> str:
    statements = dig(protocol, "report_statements", {}) or {}
    class_rows = tables["class_distribution"]
    imbalance_rows = tables["class_imbalance"]
    bbox_summary = tables["bbox_summary"]
    size_counts = tables["size_counts"]
    total_boxes = sum(size_counts.values())
    rare_rows = [row for row in class_rows if row["rare_flag"]]

    lines: List[str] = []
    add = lines.append

    add("# Phase 3A - Dataset Diagnostics Before Training")
    add("")
    add("| field | value |")
    add("| --- | --- |")
    add("| phase | 3A |")
    add("| protocol_version | {0} |".format(
        dig(protocol, "phase.protocol_version")))
    add("| protocol_status | `{0}` |".format(PROTOCOL_STATUS))
    add("| protocol_sha256 | `{0}` |".format(protocol_sha256))
    add("| phase_status | `{0}` |".format(PHASE_STATUS_VALUE))
    add("| generated_at_utc | `{0}` |".format(generated_at))
    add("")
    add("This report is descriptive pre-training dataset diagnostics. It is "
        "not training, not evaluation, not hyperparameter search, not "
        "pseudo-label generation and not an ablation study. `phase_status` is "
        "`{0}`; the script cannot and does not declare the phase PASS or "
        "CLOSED.".format(PHASE_STATUS_VALUE))
    add("")
    add("Throughout the report **METHOD** states a locked rule, definition or "
        "formula, and **RESULT** states an observed statistic.")
    add("")

    # 1 -----------------------------------------------------------------
    add("## 1. Scope and leakage policy")
    add("")
    add("**METHOD.** `primary_diagnostic_scope = {0}`. Detailed diagnostics "
        "are computed only on `instances_train.json` and on the labeled "
        "subsets L_1pct, L_5pct, L_10pct, L_20pct. The canonical masters are "
        "used for integrity reference only. Validation and test are read only "
        "through a structural-summary function that returns image count, "
        "annotation count, negative count, category ids and a checksum; the "
        "parsed validation/test documents are discarded inside that function. "
        "Test is reserved for final evaluation.".format(
            PRIMARY_DIAGNOSTIC_SCOPE))
    add("")
    add("**RESULT.** Detailed scopes actually opened during this run: `{0}`. "
        "Structural-only scopes opened: `{1}`.".format(
            ", ".join(detailed_scopes_used()),
            ", ".join(structural_scopes_used())))
    add("")

    # 2 -----------------------------------------------------------------
    add("## 2. Dataset lineage and integrity")
    add("")
    add("**METHOD.** Every locked input is hashed with SHA-256 before the "
        "diagnostics and re-hashed afterwards. A mismatch against the locked "
        "value, or any change during the run, is a hard failure.")
    add("")
    add("**RESULT.**")
    add("")
    lines.extend(_md_table(
        ["input", "sha256"],
        [[name, value] for name, value
         in sorted(context["input_hashes"].items())]))
    add("")
    add("Preflight hard checks: {0} executed, {1} failed, {2} warnings.".format(
        len(preflight.checks), preflight.hard_error_count,
        preflight.warning_count))
    add("")

    # 3 -----------------------------------------------------------------
    train_total = len(context["train_image_ids"])
    train_negative = len(context["train_negatives"])
    add("## 3. Fixed training-set overview")
    add("")
    add("**METHOD.** The fixed training set is the Phase 2E split "
        "`instances_train.json`, locked with `partition_seed = 42`. It is not "
        "regenerated or modified here.")
    add("")
    add("**RESULT.** {0} images, {1} bounding-box annotations, {2} zero-GT "
        "negative images, {3} positive images, {4} detection categories.".format(
            train_total, len(context["train_annotations"]), train_negative,
            train_total - train_negative, len(context["train_categories"])))
    add("")

    # 4 -----------------------------------------------------------------
    add("## 4. Class distribution")
    add("")
    add("**METHOD.** For class c, `N_c_img` is the number of unique training "
        "images with at least one annotation of class c (an image is counted "
        "once per class regardless of how many boxes it carries) and "
        "`P_c_img = N_c_img / {0}`. `N_c_bbox` is the bounding-box annotation "
        "count; `bbox_annotation_share = N_c_bbox / total_train_annotations`; "
        "`bbox_per_positive_image = N_c_bbox / N_c_img`. Class order is image "
        "support descending, then category_id ascending.".format(TRAIN_IMAGES))
    add("")
    add("**RESULT.**")
    add("")
    lines.extend(_md_table(
        ["category_id", "class_name", "image_count", "image_prevalence",
         "bbox_annotation_count", "bbox_annotation_share",
         "bbox_per_positive_image", "rare_flag"],
        [[row["category_id"], row["class_name"], row["image_count"],
          row["image_prevalence"], row["bbox_annotation_count"],
          row["bbox_annotation_share"], row["bbox_per_positive_image"],
          row["rare_flag"]] for row in class_rows]))
    add("")
    add("Figure: `plots/dataset/class_distribution.png` "
        "(Panel A image support, Panel B bounding-box annotation count, "
        "identical class order).")
    add("")

    # 5 -----------------------------------------------------------------
    add("## 5. Class-imbalance assessment")
    add("")
    add("**METHOD.** `R_max_min = max(N_c_img) / min(N_c_img)` over the 14 "
        "detection classes. `CV = sample_SD(N_c_img) / mean(N_c_img)` with "
        "`ddof = 1` is reported as a secondary descriptive statistic. Strong "
        "imbalance is a scientific finding, not a Phase 3A failure.")
    add("")
    add("**RESULT.**")
    add("")
    lines.extend(_md_table(
        ["metric", "value", "unit"],
        [[row["metric"], row["value"], row["unit"]] for row in imbalance_rows]))
    add("")

    # 6 -----------------------------------------------------------------
    add("## 6. Rare-class analysis")
    add("")
    add("**METHOD.** `rare_definition_type = {0}`. `Rare(c)` holds when "
        "`P_c_img < {1}` (strictly less than). With a fixed training set of "
        "{2} images this is `N_c_img <= {3}` rare and `N_c_img >= {4}` "
        "non-rare. Rarity is never defined by bounding-box counts and the "
        "threshold was not chosen from any histogram.".format(
            RARE_DEFINITION_TYPE, RARE_PRIMARY_THRESHOLD, TRAIN_IMAGES,
            RARE_BOUNDARY_MAX_RARE_COUNT, RARE_BOUNDARY_MIN_NONRARE_COUNT))
    add("")
    add("**RESULT.** {0} of {1} classes are rare under the locked "
        "definition.".format(len(rare_rows), len(class_rows)))
    add("")
    if rare_rows:
        lines.extend(_md_table(
            ["category_id", "class_name", "image_count", "image_prevalence"],
            [[row["category_id"], row["class_name"], row["image_count"],
              row["image_prevalence"]] for row in
             sorted(rare_rows, key=lambda r: int(r["category_id"]))]))
        add("")
    add("**Interpretation.** {0}".format(
        statements.get("rare_class_interpretation_statement", "")))
    add("")

    # 7 -----------------------------------------------------------------
    add("## 7. Bounding-box annotation distribution")
    add("")
    add("**METHOD.** `N_c_bbox` counts COCO annotation records of class c. "
        "The quantity is a bounding-box annotation count; bounding-box "
        "annotation records must not be interpreted as a count of "
        "independently resolved clinical lesions.")
    add("")
    add("**RESULT.** Total bounding-box annotations on the fixed training "
        "set: {0}. Per-class counts and shares are in section 4 and in "
        "`reports/03A_class_distribution.csv`.".format(
            len(context["train_annotations"])))
    add("")

    # 8 -----------------------------------------------------------------
    add("## 8. Bounding-box count per image")
    add("")
    add("**METHOD.** `B_i` is the number of bounding-box annotation records of "
        "training image i. Statistics are reported for all training images and "
        "for positive training images only, with sample SD (`ddof = 1`) and "
        "numpy `linear` percentiles.")
    add("")
    add("**RESULT.**")
    add("")
    lines.extend(_md_table(
        ["population", "n", "mean", "sd_ddof1", "median", "p95", "max"],
        [[row["population"], row["n"], row["mean"], row["sd_ddof1"],
          row["median"], row["p95"], row["max"]]
         for row in tables["bbox_count_per_image"]]))
    add("")

    # 9 -----------------------------------------------------------------
    add("## 9. Bounding-box geometry")
    add("")
    add("**METHOD.** COCO boxes are `[x, y, w, h]` in absolute pixels with "
        "image size `W x H`. `w_n = w/W`, `h_n = h/H`, "
        "`a_n = (w*h)/(W*H)`, `AR = w/h`. The primary scale descriptor is "
        "`normalized_area`. Every training annotation must satisfy `w > 0`, "
        "`h > 0`, `x >= 0`, `y >= 0`, `x + w <= W`, `y + h <= H` and "
        "`0 < a_n <= 1`; a violation is a hard failure and no box is clamped, "
        "repaired, deleted, moved or resized.")
    add("")
    add("**RESULT.**")
    add("")
    lines.extend(_md_table(
        ["variable", "role", "n", "mean", "sd", "min", "p05", "p25", "median",
         "p75", "p95", "max"],
        [[row["variable"], row["role"], row["n"], row["mean"], row["sd"],
          row["min"], row["p05"], row["p25"], row["median"], row["p75"],
          row["p95"], row["max"]] for row in bbox_summary]))
    add("")

    # 10 ----------------------------------------------------------------
    add("## 10. Normalized-area Small/Medium/Large analysis")
    add("")
    add("**METHOD.** `bbox_size_definition_type = {0}`. Small is "
        "`a_n < {1}`, Medium is `{1} <= a_n < {2}`, Large is `a_n >= {2}`. "
        "Boundaries are exact: `a_n == {1}` is Medium and `a_n == {2}` is "
        "Large.".format(BBOX_SIZE_DEFINITION_TYPE,
                        BBOX_SMALL_PRIMARY_THRESHOLD,
                        BBOX_LARGE_PRIMARY_THRESHOLD))
    add("")
    add("**RESULT.**")
    add("")
    lines.extend(_md_table(
        ["size_category", "count", "percentage_of_annotations"],
        [[name, size_counts[name],
          (100.0 * size_counts[name] / total_boxes) if total_boxes else None]
         for name in BBOX_SIZE_CATEGORIES]))
    add("")
    add("**Interpretation.** {0}".format(
        statements.get("bbox_size_interpretation_statement", "")))
    add("")
    add("Figure: `plots/dataset/bbox_distribution.png`.")
    add("")

    # 11 ----------------------------------------------------------------
    add("## 11. Bounding-box location analysis")
    add("")
    add("**METHOD.** `center_x_n = (x + w/2)/W` and `center_y_n = (y + h/2)/H` "
        "with a top-left origin, x from left to right and y from top to "
        "bottom. The density map uses a locked {0} x {1} grid over "
        "`[0,1] x [0,1]` and is annotation-weighted.".format(
            HEATMAP_BINS_X, HEATMAP_BINS_Y))
    add("")
    add("**RESULT.** Figure `plots/dataset/bbox_location_heatmap.png`; the "
        "per-annotation centres are in `reports/03A_bbox_distribution.csv`.")
    add("")
    add("**Interpretation.** {0}".format(
        dig(protocol, "bbox_location.required_caption_note", "")))
    add("")

    # 12 ----------------------------------------------------------------
    add("## 12. Negative / No Finding analysis")
    add("")
    add("**METHOD.** No Finding is a zero-ground-truth negative image, not a "
        "detection category and never a fifteenth class. On the fixed "
        "training set the positive count, negative count and negative "
        "prevalence are reported. Validation and test contribute structural "
        "counts only.")
    add("")
    add("**RESULT.**")
    add("")
    lines.extend(_md_table(
        ["scope", "diagnostic_role", "image_count", "positive_image_count",
         "negative_image_count", "negative_prevalence"],
        [[row["scope"], row["diagnostic_role"], row["image_count"],
          row["positive_image_count"], row["negative_image_count"],
          row["negative_prevalence"]]
         for row in tables["negative_distribution"]]))
    add("")
    add("Figure: `plots/dataset/negative_image_distribution.png`.")
    add("")

    # 13 ----------------------------------------------------------------
    add("## 13. Labeled-budget diagnostics")
    add("")
    add("**METHOD.** For each budget b and class c, `P_b,c_img` is the "
        "image-level prevalence inside L_b and "
        "`D_b,c = |P_b,c_img - P_train,c_img|`. The rare flag is always the "
        "fixed-train flag; rarity is never recomputed inside a budget. All "
        "figures are recomputed independently from the labeled COCO files; "
        "`reports/02F_class_distribution.csv` is a non-gating cross-check. The "
        "hidden ground truth of the unlabeled complements is not used. This "
        "analysis is descriptive and can never trigger a rebuild of L_b, whose "
        "membership is locked.")
    add("")
    add("**RESULT.**")
    add("")
    budget_summary_rows = []
    for budget in BUDGET_KEYS:
        budget_rows = [row for row in tables["labeled_budget_coverage"]
                       if row["budget"] == budget]
        if not budget_rows:
            continue
        first = budget_rows[0]
        budget_summary_rows.append([
            budget, first["labeled_image_count"], first["negative_image_count"],
            first["negative_prevalence"], first["class_coverage_out_of_14"],
            first["budget_max_absolute_deviation"],
            first["budget_mean_absolute_deviation"]])
    lines.extend(_md_table(
        ["budget", "labeled_image_count", "negative_image_count",
         "negative_prevalence", "class_coverage_out_of_14",
         "max_absolute_deviation", "mean_absolute_deviation"],
        budget_summary_rows))
    add("")
    add("Per-class detail is in `reports/03A_labeled_budget_coverage.csv`.")
    add("")

    # 14 ----------------------------------------------------------------
    add("## 14. Label-cardinality analysis")
    add("")
    add("**METHOD.** `LC_i` is the number of unique detection classes present "
        "in training image i; a negative image has `LC_i = 0`. Bounding-box "
        "counts are never substituted for unique class counts.")
    add("")
    add("**RESULT.**")
    add("")
    lines.extend(_md_table(
        ["cardinality", "image_count", "percentage"],
        [[row["cardinality"], row["image_count"], row["percentage"]]
         for row in tables["label_cardinality"]
         if row["record_type"] == "distribution"]))
    add("")
    lines.extend(_md_table(
        ["statistic", "value"],
        [[row["statistic"], row["statistic_value"]]
         for row in tables["label_cardinality"]
         if row["record_type"] == "summary"]))
    add("")

    # 15 ----------------------------------------------------------------
    add("## 15. Class co-occurrence")
    add("")
    add("**METHOD.** Role is SECONDARY / NON-GATING. Co-occurrence uses unique "
        "image-level class presence, never bounding-box counts. For classes c "
        "and d, `N_cd` is the number of training images containing both and "
        "`J(c,d) = N_cd / (N_c + N_d - N_cd)`. All {0} unordered pairs are "
        "emitted, sorted by `category_id_a` then `category_id_b`. "
        "Co-occurrence never changes the training design.".format(
            EXPECTED_COOCCURRENCE_PAIRS))
    add("")
    add("**RESULT.** {0} unordered pairs written to "
        "`reports/03A_class_cooccurrence.csv`. The ten highest Jaccard "
        "coefficients:".format(len(tables["cooccurrence"])))
    add("")
    top_pairs = sorted(
        [row for row in tables["cooccurrence"] if row["jaccard"] is not None],
        key=lambda row: (-float(row["jaccard"]), int(row["category_id_a"]),
                         int(row["category_id_b"])))[:10]
    lines.extend(_md_table(
        ["class_name_a", "class_name_b", "n_a_images", "n_b_images",
         "n_both_images", "jaccard"],
        [[row["class_name_a"], row["class_name_b"], row["n_a_images"],
          row["n_b_images"], row["n_both_images"], row["jaccard"]]
         for row in top_pairs]))
    add("")

    # 16 ----------------------------------------------------------------
    add("## 16. Threshold sensitivity analysis")
    add("")
    add("**METHOD.** Role is SECONDARY / NON-GATING. The rare threshold is "
        "varied over {0} with primary {1}; the small boundary over {2} with "
        "the large boundary fixed at {3}; the large boundary over {4} with the "
        "small boundary fixed at {5}. The variation is one factor at a time; "
        "no 3 x 3 Cartesian grid is run.".format(
            list(RARE_SENSITIVITY_THRESHOLDS), RARE_PRIMARY_THRESHOLD,
            list(BBOX_SMALL_SENSITIVITY_THRESHOLDS),
            BBOX_LARGE_PRIMARY_THRESHOLD,
            list(BBOX_LARGE_SENSITIVITY_THRESHOLDS),
            BBOX_SMALL_PRIMARY_THRESHOLD))
    add("")
    add("**RESULT.**")
    add("")
    lines.extend(_md_table(
        ["analysis_type", "parameter", "threshold", "is_primary", "category",
         "count", "percentage", "primary_percentage",
         "delta_percentage_points"],
        [[row["analysis_type"], row["parameter"], row["threshold"],
          row["is_primary"], row["category"], row["count"], row["percentage"],
          row["primary_percentage"], row["delta_percentage_points"]]
         for row in tables["threshold_sensitivity"]
         if row["analysis_type"] != "rare_threshold_per_class"]))
    add("")
    sensitive = sorted({
        str(row["class_name"]) for row in tables["threshold_sensitivity"]
        if row["analysis_type"] == "rare_threshold_per_class"
        and row["threshold_sensitive_flag"] is True})
    add("Classes whose rare status is not identical across {0}: {1}".format(
        list(RARE_SENSITIVITY_THRESHOLDS),
        ", ".join(sensitive) if sensitive else "none"))
    add("")
    add("`primary_threshold_changed_after_diagnostics = false`; "
        "`sensitivity_used_for_threshold_selection = false`. {0}".format(
            statements.get("threshold_statement_sensitivity", "")))
    add("")

    # 17 ----------------------------------------------------------------
    add("## 17. Fixed split structural summary")
    add("")
    add("**METHOD.** Structural fields only: image count, image percentage, "
        "annotation count, negative count, negative percentage, the official "
        "JSON SHA-256 and, where available, the locked image-membership "
        "SHA-256. No class distribution, bbox size, label cardinality, "
        "co-occurrence or heatmap is computed for validation or test.")
    add("")
    add("**RESULT.**")
    add("")
    lines.extend(_md_table(
        ["split", "image_count", "image_percentage", "annotation_count",
         "negative_count", "negative_percentage", "official_json_sha256",
         "image_membership_sha256", "diagnostic_usage"],
        [[row["split"], row["image_count"], row["image_percentage"],
          row["annotation_count"], row["negative_count"],
          row["negative_percentage"], row["official_json_sha256"],
          row["image_membership_sha256"], row["diagnostic_usage"]]
         for row in tables["split_distribution"]]))
    add("")

    # 18 ----------------------------------------------------------------
    add("## 18. Pre-training dataset risks")
    add("")
    add("**METHOD.** Risks are stated as descriptive dataset properties "
        "observed before training. None of them is a prediction about model "
        "behaviour: no claim is made about average precision, detection "
        "failure, pseudo-label quality, semi-supervised superiority, "
        "convergence or training stability.")
    add("")
    add("**RESULT.**")
    add("")
    add("- Class support is unbalanced; the observed "
        "`imbalance_ratio_max_over_min` is reported in section 5.")
    add("- {0} of {1} classes fall below the locked rare threshold and "
        "therefore have low support in the fixed training set.".format(
            len(rare_rows), len(class_rows)))
    add("- The Small share of bounding boxes under the locked "
        "normalized-area definition is reported in section 10.")
    add("- Zero-GT negative images are present at the prevalence reported in "
        "section 12 and carry no boxes by construction.")
    add("- Rare-class support inside the smallest labeled budget is reported "
        "per class in `reports/03A_labeled_budget_coverage.csv`.")
    add("")
    # Magnitudes are reported directly. Phase 3A defines no operational
    # rule for qualitative labels such as "severe", "high" or "strong",
    # so the report must not attach them to any observation.
    add("The descriptive statistics reported above characterize potential "
        "pre-training dataset risks. Their magnitude is reported directly "
        "rather than classified using additional post-hoc labels, and none of "
        "these observations constitutes a Phase 3A protocol failure.")
    add("")

    # 19 ----------------------------------------------------------------
    add("## 19. Limitations")
    add("")
    add("- Phase 3A is descriptive. It contains no model, no training run and "
        "no evaluation, so no statement about achievable detection "
        "performance can be derived from it.")
    add("- Rare, Small, Medium and Large are pre-specified operational "
        "categories of this study, not external standards.")
    add("- Bounding-box counts are annotation records. Multiple radiologist "
        "annotations of the same finding are not resolved here, so a count is "
        "not a count of unique clinical lesions.")
    add("- Validation and test were summarised structurally only, so no "
        "statement about their content distribution is made.")
    add("- The unlabeled complements U_b were not inspected, so nothing is "
        "claimed about their label distribution.")
    add("- Sensitivity analysis covers only the ranges listed in section 16 "
        "and one factor at a time.")
    add("")

    # 20 ----------------------------------------------------------------
    add("## 20. Leakage statement")
    add("")
    add(statements.get("leakage_statement", ""))
    add("")
    add(statements.get("threshold_statement_pre_specification", ""))
    add("")
    add(statements.get("threshold_statement_sensitivity", ""))
    add("")

    # 21 ----------------------------------------------------------------
    add("## 21. Conclusions before training")
    add("")
    add("**METHOD.** Conclusions are restricted to the claim boundaries of the "
        "protocol: class rarity under the locked definition, imbalance "
        "magnitude, size-category percentages, spatial concentration, negative "
        "prevalence, labeled-budget coverage and descriptive robustness under "
        "sensitivity.")
    add("")
    add("**RESULT.** The fixed training set, the four labeled budgets and the "
        "locked split all match their recorded identities and counts; the "
        "descriptive properties above are documented for use as pre-training "
        "context. `phase_status = {0}`. This report does not declare Phase 3A "
        "PASS or CLOSED; the researcher and the GPT review decide.".format(
            PHASE_STATUS_VALUE))
    add("")
    return "\n".join(lines) + "\n"


# =====================================================================
# VALIDATION JSON + ARTIFACT MANIFEST
# =====================================================================

def derive_forbidden_action_flags(
        protocol: Mapping[str, Any],
        training_artifacts_created: Sequence[str] = (),
        seed_state: Optional[Mapping[str, Any]] = None,
        detailed_scopes: Optional[Sequence[str]] = None,
        forbidden_unlabeled_paths: Optional[Sequence[str]] = None
        ) -> Dict[str, bool]:
    """Derive the forbidden-action evidence flags from runtime evidence.

    Nothing here is hard-coded to False: every flag is computed from a
    runtime tripwire, from the artifact-detection diff, or from the
    Phase 2F.1 seed-state evidence.
    """
    scopes = list(detailed_scopes if detailed_scopes is not None
                  else detailed_scopes_used())
    if forbidden_unlabeled_paths is None:
        forbidden_unlabeled_paths = find_forbidden_unlabeled_paths(
            collect_protocol_paths(protocol))

    created = [str(item) for item in training_artifacts_created]
    checkpoint_created = any(
        Path(item).suffix.lower() in TRAINING_ARTIFACT_SUFFIXES
        for item in created)
    pseudo_label_created = any(
        "pseudo_label" in item.replace("\\", "/").lower() for item in created)
    training_artifact_created = bool(created)

    illegal_scopes = [scope for scope in scopes
                      if scope not in DETAILED_SCOPE_ALLOWED]
    val_or_test_detailed = bool({"val", "test"} & set(scopes))
    unlabeled_detailed = any("unlabeled" in scope for scope in scopes)

    training_seed_used = not (
        dig(protocol, "scope_policy.training_seed_used") is False
        and dig(protocol, "determinism.training_seed_used") is False
        and dig(protocol, "determinism.rng_used") is False
        and dig(protocol, "scope_policy.repeat_over_training_seeds") is False)

    training_started = False
    if seed_state:
        training_started = bool(seed_state.get("training_started")) or bool(
            seed_state.get("runs"))

    return {
        "validation_design_usage": "val" in scopes,
        "test_design_usage": "test" in scopes,
        "test_content_diagnostics": val_or_test_detailed,
        "unlabeled_hidden_gt_diagnostics": bool(
            unlabeled_detailed or illegal_scopes
            or list(forbidden_unlabeled_paths)),
        "checkpoint_created": checkpoint_created,
        "pseudo_label_created": pseudo_label_created,
        "training_artifact_created": training_artifact_created,
        "training_seed_used": bool(training_seed_used),
        "training_started": training_started,
    }


def assert_validation_consistency(validation: Mapping[str, Any]) -> None:
    """Refuse to emit a validation record that contradicts its own evidence.

    A validation JSON may never state ``checkpoint_created = false`` while
    its ``training_artifacts_detected`` list contains a checkpoint, and it
    may never state ``test_content_diagnostics = false`` while a detailed
    val/test scope appears in ``detailed_scopes_used``.
    """
    detected = [str(item) for item in
                validation.get("training_artifacts_detected", []) or []]
    if any(Path(item).suffix.lower() in TRAINING_ARTIFACT_SUFFIXES
           for item in detected) and not validation.get("checkpoint_created"):
        raise Phase3AError(
            "validation contradiction: a checkpoint appears in "
            "training_artifacts_detected but checkpoint_created is false")
    if any("pseudo_label" in item.replace("\\", "/").lower()
           for item in detected) and not validation.get(
               "pseudo_label_created"):
        raise Phase3AError(
            "validation contradiction: a pseudo-label artifact appears in "
            "training_artifacts_detected but pseudo_label_created is false")
    if detected and not validation.get("training_artifact_created"):
        raise Phase3AError(
            "validation contradiction: training_artifacts_detected is "
            "non-empty but training_artifact_created is false")

    scopes = [str(item) for item in
              validation.get("detailed_scopes_used", []) or []]
    if ({"val", "test"} & set(scopes)) and not validation.get(
            "test_content_diagnostics"):
        raise Phase3AError(
            "validation contradiction: a detailed val/test scope was used "
            "but test_content_diagnostics is false")
    if any(scope not in DETAILED_SCOPE_ALLOWED for scope in scopes) and \
            not validation.get("unlabeled_hidden_gt_diagnostics"):
        raise Phase3AError(
            "validation contradiction: a scope outside the permitted "
            "detailed scopes was used but "
            "unlabeled_hidden_gt_diagnostics is false")

    undeclared = list(validation.get("undeclared_hard_fail_codes", []) or [])
    if undeclared:
        raise Phase3AError(
            "validation contradiction: the runtime emitted hard-fail codes "
            "that the protocol does not declare in "
            "guardrails.hard_fail_conditions: {0}".format(undeclared))

    if validation.get("phase_status") != PHASE_STATUS_VALUE:
        raise Phase3AError(
            "validation contradiction: phase_status must always be "
            "{0}".format(PHASE_STATUS_VALUE))
    if validation.get("phase_status") in FORBIDDEN_PHASE_STATUS_VALUES:
        raise Phase3AError(
            "validation contradiction: the script may never self-close the "
            "phase")


def read_csv_table(path: Path) -> Tuple[List[str], List[List[str]]]:
    """Read a written CSV back as (header, rows). Structural use only."""
    with open(path, "r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader, None) or []
        rows = [row for row in reader]
    return list(header), rows


def audit_machine_readable_outputs(output_paths: Mapping[str, Path]
                                   ) -> List[Dict[str, Any]]:
    """Structural audit of the written machine-readable artifacts.

    [LOCKED SCIENTIFIC DECISION] No observed scientific distribution is a
    PASS/FAIL gate. Only structural consistency is checked: declared CSV
    schemas, structural row counts, internal count reconciliation against
    the locked annotation total, and JSON parseability.
    """
    audits: List[Dict[str, Any]] = []

    def record(name: str, condition: bool, expected: Any, observed: Any
               ) -> None:
        audits.append({
            "check": name,
            "status": "PASS" if condition else "FAIL",
            "expected": expected,
            "observed": observed,
        })

    tables: Dict[str, Tuple[List[str], List[List[str]]]] = {}
    for key, schema in sorted(MANDATORY_CSV_SCHEMAS.items()):
        path = output_paths.get("csv_{0}".format(key))
        if path is None or not Path(path).is_file():
            record("csv_present::{0}".format(key), False, "file exists",
                   str(path))
            continue
        record("csv_present::{0}".format(key), True, "file exists", str(path))
        header, rows = read_csv_table(Path(path))
        tables[key] = (header, rows)
        record("csv_schema::{0}".format(key), header == list(schema),
               list(schema), header)
        expected_rows = EXPECTED_CSV_ROW_COUNTS.get(key)
        if expected_rows is not None:
            record("csv_row_count::{0}".format(key),
                   len(rows) == expected_rows, expected_rows, len(rows))
        else:
            record("csv_row_count::{0}".format(key), len(rows) > 0,
                   "> 0 (data dependent)", len(rows))

    # ---- internal reconciliation against the locked annotation total ----
    if "class_distribution" in tables:
        header, rows = tables["class_distribution"]
        try:
            index = header.index("bbox_annotation_count")
            total = sum(int(row[index]) for row in rows)
        except (ValueError, IndexError):
            total = None
        record("sum_class_bbox_annotation_count",
               total == TRAIN_ANNOTATIONS, TRAIN_ANNOTATIONS, total)

    if "bbox_distribution" in tables:
        header, rows = tables["bbox_distribution"]
        try:
            index = header.index("primary_size_category")
            counts = {name: 0 for name in BBOX_SIZE_CATEGORIES}
            unknown = 0
            for row in rows:
                value = row[index]
                if value in counts:
                    counts[value] += 1
                else:
                    unknown += 1
        except (ValueError, IndexError):
            counts, unknown = {}, -1
        record("sum_primary_size_category_counts",
               sum(counts.values()) == TRAIN_ANNOTATIONS and unknown == 0,
               {"total": TRAIN_ANNOTATIONS, "unknown_categories": 0},
               {"total": sum(counts.values()), "unknown_categories": unknown,
                "by_category": counts})

    # ---- JSON artifacts must parse -------------------------------------
    for key in MANDATORY_JSON_OUTPUT_KEYS:
        path = output_paths.get(key)
        if path is None or not Path(path).is_file():
            record("json_present::{0}".format(key), False, "file exists",
                   str(path))
            continue
        try:
            json.loads(Path(path).read_text(encoding="utf-8"))
            record("json_parses::{0}".format(key), True, "valid JSON",
                   "valid JSON")
        except ValueError as error:
            record("json_parses::{0}".format(key), False, "valid JSON",
                   str(error))
    return audits


def audit_output_contract(output_paths: Mapping[str, Path]) -> Dict[str, Any]:
    """Final audit over EVERY mandatory Phase 3A output.

    Includes reports/03A_dataset_diagnostics_validation.json and
    reports/03A_artifact_manifest.json, so HF35 can never be declared
    PASS before those two artifacts have been accounted for.
    """
    mandatory_keys = (
        ["csv_{0}".format(key) for key in sorted(MANDATORY_CSV_SCHEMAS)]
        + list(MANDATORY_PLOT_TARGET_KEYS)
        + ["report", "validation_json", "artifact_manifest"])
    missing: List[str] = []
    empty: List[str] = []
    for key in mandatory_keys:
        path = output_paths.get(key)
        if path is None or not Path(path).is_file():
            missing.append(key)
            continue
        if Path(path).stat().st_size <= 0:
            empty.append(key)
    return {
        "mandatory_keys": mandatory_keys,
        "missing": missing,
        "empty": empty,
        "all_present": not missing and not empty,
    }


def yaml_python_lock_mismatches(protocol: Mapping[str, Any]
                                ) -> List[Dict[str, Any]]:
    """Cross-check every machine-readable YAML lock against the source.

    The independently hard-coded Python constants are authoritative. Any
    disagreement is returned here and fails preflight, so a silent edit of
    the YAML can never move a locked value.
    """
    comparisons: List[Tuple[str, Any, Any]] = [
        ("expected_hashes.coco_master_jpg",
         MASTER_JPG_SHA256, dig(protocol, "expected_hashes.coco_master_jpg")),
        ("expected_hashes.train",
         TRAIN_SHA256, dig(protocol, "expected_hashes.train")),
        ("expected_hashes.val",
         VAL_SHA256, dig(protocol, "expected_hashes.val")),
        ("expected_hashes.test",
         TEST_SHA256, dig(protocol, "expected_hashes.test")),
        ("expected_counts.canonical.images",
         CANONICAL_IMAGES, dig(protocol, "expected_counts.canonical.images")),
        ("expected_counts.canonical.annotations",
         CANONICAL_ANNOTATIONS,
         dig(protocol, "expected_counts.canonical.annotations")),
        ("expected_counts.canonical.categories",
         CANONICAL_CATEGORIES,
         dig(protocol, "expected_counts.canonical.categories")),
        ("expected_counts.canonical.no_finding_images",
         CANONICAL_NO_FINDING,
         dig(protocol, "expected_counts.canonical.no_finding_images")),
        ("expected_counts.train.images",
         TRAIN_IMAGES, dig(protocol, "expected_counts.train.images")),
        ("expected_counts.train.annotations",
         TRAIN_ANNOTATIONS,
         dig(protocol, "expected_counts.train.annotations")),
        ("expected_counts.train.no_finding_images",
         TRAIN_NO_FINDING,
         dig(protocol, "expected_counts.train.no_finding_images")),
        ("expected_counts.val.images",
         VAL_IMAGES, dig(protocol, "expected_counts.val.images")),
        ("expected_counts.val.annotations",
         VAL_ANNOTATIONS, dig(protocol, "expected_counts.val.annotations")),
        ("expected_counts.val.no_finding_images",
         VAL_NO_FINDING,
         dig(protocol, "expected_counts.val.no_finding_images")),
        ("expected_counts.test.images",
         TEST_IMAGES, dig(protocol, "expected_counts.test.images")),
        ("expected_counts.test.annotations",
         TEST_ANNOTATIONS, dig(protocol, "expected_counts.test.annotations")),
        ("expected_counts.test.no_finding_images",
         TEST_NO_FINDING,
         dig(protocol, "expected_counts.test.no_finding_images")),
        ("rare_class_definition.primary_threshold",
         RARE_PRIMARY_THRESHOLD,
         dig(protocol, "rare_class_definition.primary_threshold")),
        ("rare_class_definition.denominator",
         TRAIN_IMAGES, dig(protocol, "rare_class_definition.denominator")),
        ("rare_class_definition.boundary_max_rare_image_count",
         RARE_BOUNDARY_MAX_RARE_COUNT,
         dig(protocol,
             "rare_class_definition.boundary_max_rare_image_count")),
        ("rare_class_definition.boundary_min_nonrare_image_count",
         RARE_BOUNDARY_MIN_NONRARE_COUNT,
         dig(protocol,
             "rare_class_definition.boundary_min_nonrare_image_count")),
        ("bbox_size_definition.small_threshold",
         BBOX_SMALL_PRIMARY_THRESHOLD,
         dig(protocol, "bbox_size_definition.small_threshold")),
        ("bbox_size_definition.large_threshold",
         BBOX_LARGE_PRIMARY_THRESHOLD,
         dig(protocol, "bbox_size_definition.large_threshold")),
        ("bbox_size_definition.category_order",
         list(BBOX_SIZE_CATEGORIES),
         list(dig(protocol, "bbox_size_definition.category_order", []) or [])),
        ("sensitivity_analysis.rare.thresholds",
         [float(v) for v in RARE_SENSITIVITY_THRESHOLDS],
         [float(v) for v in
          (dig(protocol, "sensitivity_analysis.rare.thresholds", []) or [])]),
        ("sensitivity_analysis.rare.primary",
         RARE_PRIMARY_THRESHOLD,
         dig(protocol, "sensitivity_analysis.rare.primary")),
        ("sensitivity_analysis.bbox_small.thresholds",
         [float(v) for v in BBOX_SMALL_SENSITIVITY_THRESHOLDS],
         [float(v) for v in
          (dig(protocol, "sensitivity_analysis.bbox_small.thresholds", [])
           or [])]),
        ("sensitivity_analysis.bbox_small.primary",
         BBOX_SMALL_PRIMARY_THRESHOLD,
         dig(protocol, "sensitivity_analysis.bbox_small.primary")),
        ("sensitivity_analysis.bbox_small.large_threshold_fixed_at",
         BBOX_LARGE_PRIMARY_THRESHOLD,
         dig(protocol,
             "sensitivity_analysis.bbox_small.large_threshold_fixed_at")),
        ("sensitivity_analysis.bbox_large.thresholds",
         [float(v) for v in BBOX_LARGE_SENSITIVITY_THRESHOLDS],
         [float(v) for v in
          (dig(protocol, "sensitivity_analysis.bbox_large.thresholds", [])
           or [])]),
        ("sensitivity_analysis.bbox_large.primary",
         BBOX_LARGE_PRIMARY_THRESHOLD,
         dig(protocol, "sensitivity_analysis.bbox_large.primary")),
        ("sensitivity_analysis.bbox_large.small_threshold_fixed_at",
         BBOX_SMALL_PRIMARY_THRESHOLD,
         dig(protocol,
             "sensitivity_analysis.bbox_large.small_threshold_fixed_at")),
        ("sensitivity_analysis.one_factor_at_a_time",
         True, dig(protocol, "sensitivity_analysis.one_factor_at_a_time")),
        ("bbox_location.heatmap_bins_x",
         HEATMAP_BINS_X, dig(protocol, "bbox_location.heatmap_bins_x")),
        ("bbox_location.heatmap_bins_y",
         HEATMAP_BINS_Y, dig(protocol, "bbox_location.heatmap_bins_y")),
        ("category_contract.detection_category_count",
         CANONICAL_CATEGORIES,
         dig(protocol, "category_contract.detection_category_count")),
        ("category_contract.category_ids_contiguous_from",
         1, dig(protocol, "category_contract.category_ids_contiguous_from")),
        ("category_contract.category_ids_contiguous_to",
         CANONICAL_CATEGORIES,
         dig(protocol, "category_contract.category_ids_contiguous_to")),
        ("bbox_geometry.continuous_statistics.sd_ddof",
         SD_DDOF,
         dig(protocol, "bbox_geometry.continuous_statistics.sd_ddof")),
        ("bbox_geometry.continuous_statistics.percentile_method",
         PERCENTILE_METHOD,
         dig(protocol,
             "bbox_geometry.continuous_statistics.percentile_method")),
        ("bbox_count_per_image.sd_ddof",
         SD_DDOF, dig(protocol, "bbox_count_per_image.sd_ddof")),
        ("multilabel_diagnostics.label_cardinality.sd_ddof",
         SD_DDOF,
         dig(protocol, "multilabel_diagnostics.label_cardinality.sd_ddof")),
        ("multilabel_diagnostics.class_cooccurrence.expected_pair_count",
         EXPECTED_COOCCURRENCE_PAIRS,
         dig(protocol,
             "multilabel_diagnostics.class_cooccurrence."
             "expected_pair_count")),
        ("output_contract.bbox_distribution_columns",
         list(BBOX_DISTRIBUTION_COLUMNS),
         list(dig(protocol, "output_contract.bbox_distribution_columns", [])
              or [])),
        ("output_contract.writable_directories",
         list(WRITABLE_OUTPUT_DIRECTORIES),
         list(dig(protocol, "output_contract.writable_directories", []) or [])),
        ("output_contract.forbidden_write_paths",
         list(FORBIDDEN_OUTPUT_DIRECTORIES),
         list(dig(protocol, "output_contract.forbidden_write_paths", [])
              or [])),
    ]

    for budget in BUDGET_KEYS:
        comparisons.append((
            "expected_hashes.labeled.{0}".format(budget),
            LABELED_SHA256[budget],
            dig(protocol, "expected_hashes.labeled.{0}".format(budget))))
        comparisons.append((
            "expected_counts.labeled.{0}.images".format(budget),
            LABELED_IMAGE_COUNTS[budget],
            dig(protocol,
                "expected_counts.labeled.{0}.images".format(budget))))
        comparisons.append((
            "expected_counts.labeled.{0}.no_finding_images".format(budget),
            LABELED_NO_FINDING_COUNTS[budget],
            dig(protocol, "expected_counts.labeled.{0}.no_finding_images"
                          .format(budget))))

    declared_schemas = dig(protocol, "output_contract.csv_schemas", {}) or {}
    for key, schema in sorted(MANDATORY_CSV_SCHEMAS.items()):
        comparisons.append((
            "output_contract.csv_schemas.{0}".format(key), list(schema),
            list(declared_schemas.get(key, []) or [])))
    declared_rows = dig(protocol, "output_contract.expected_row_counts",
                        {}) or {}
    for key, expected in sorted(EXPECTED_CSV_ROW_COUNTS.items()):
        comparisons.append((
            "output_contract.expected_row_counts.{0}".format(key), expected,
            declared_rows.get(key, "__ABSENT__")
            if key in declared_rows else "__ABSENT__"))

    mismatches: List[Dict[str, Any]] = []
    for field_name, expected, observed in comparisons:
        if isinstance(expected, float) or isinstance(observed, float):
            equal = (observed is not None
                     and not isinstance(observed, (list, dict, str))
                     and float(observed) == float(expected))
        else:
            equal = observed == expected
        if not equal:
            mismatches.append({"field": field_name, "expected": expected,
                               "observed": observed})
    return mismatches


def build_validation(protocol: Mapping[str, Any], protocol_sha256: str,
                     preflight: PreflightResult, context: Mapping[str, Any],
                     mode: str, generated_at: str,
                     locked_inputs_modified: bool,
                     mandatory_outputs_exist: bool,
                     machine_readable_outputs_valid: bool,
                     training_artifacts_found: Sequence[str],
                     seed_state: Optional[Mapping[str, Any]] = None,
                     output_contract_audit: Optional[Mapping[str, Any]] = None,
                     machine_readable_audit: Optional[
                         Sequence[Mapping[str, Any]]] = None,
                     provisional: bool = False
                     ) -> Dict[str, Any]:
    def passed(code: str) -> Optional[bool]:
        for check in preflight.checks:
            if check.code == code:
                return check.status == "PASS"
        return None

    canonical_identity = passed("HF01")
    canonical_counts = (bool(passed("HF05")) and bool(passed("HF06"))
                        and bool(passed("PF09")))
    category_mapping = bool(passed("HF07")) and bool(passed("HF08"))
    no_finding_policy = (bool(passed("HF09")) and bool(passed("HF09b"))
                         and bool(passed("HF10")))
    split_identity = (bool(passed("HF02")) and bool(passed("HF03"))
                      and bool(passed("HF04")) and bool(passed("HF11"))
                      and bool(passed("HF12")) and bool(passed("HF13"))
                      and bool(passed("HF14")) and bool(passed("HF15")))
    membership_identity = (bool(passed("HF17")) and bool(passed("HF18"))
                           and bool(passed("HF20")))

    # Forbidden-action evidence is DERIVED, never hard-coded.
    flags = derive_forbidden_action_flags(
        protocol,
        training_artifacts_created=training_artifacts_found,
        seed_state=seed_state)

    validation = {
        "phase": PHASE_ID,
        "protocol_version": dig(protocol, "phase.protocol_version"),
        "protocol_status": PROTOCOL_STATUS,
        "protocol_sha256": protocol_sha256,
        "mode": mode,
        "generated_at_utc": generated_at,

        "canonical_identity_pass": bool(canonical_identity),
        "train_identity_pass": bool(passed("HF02")) and bool(passed("HF11"))
                               and bool(passed("HF12")) and bool(passed("HF13")),
        "validation_identity_pass": bool(passed("HF03"))
                                    and bool(passed("HF14")),
        "test_identity_pass": bool(passed("HF04")) and bool(passed("HF15")),
        "canonical_counts_pass": canonical_counts,
        "category_mapping_pass": category_mapping,
        "no_finding_policy_pass": no_finding_policy,
        "bbox_validity_pass": bool(passed("HF16")),
        "phase2E_split_identity_pass": split_identity,
        "phase2F_membership_identity_pass": membership_identity,
        "phase2F_nested_pass": bool(passed("HF19")),

        "primary_diagnostic_scope_train_only_pass": (
            bool(passed("PF07"))
            and "val" not in detailed_scopes_used()
            and "test" not in detailed_scopes_used()),
        "validation_design_usage": flags["validation_design_usage"],
        "test_design_usage": flags["test_design_usage"],
        "test_content_diagnostics": flags["test_content_diagnostics"],
        "unlabeled_hidden_gt_diagnostics": flags[
            "unlabeled_hidden_gt_diagnostics"],

        "rare_definition_type": RARE_DEFINITION_TYPE,
        "rare_primary_threshold": RARE_PRIMARY_THRESHOLD,
        "bbox_size_definition_type": BBOX_SIZE_DEFINITION_TYPE,
        "bbox_small_primary_threshold": BBOX_SMALL_PRIMARY_THRESHOLD,
        "bbox_large_primary_threshold": BBOX_LARGE_PRIMARY_THRESHOLD,

        "sensitivity_analysis_enabled": True,
        "sensitivity_analysis_role": "SECONDARY_NON_GATING",
        "rare_sensitivity_thresholds": list(RARE_SENSITIVITY_THRESHOLDS),
        "bbox_small_sensitivity_thresholds": list(
            BBOX_SMALL_SENSITIVITY_THRESHOLDS),
        "bbox_large_sensitivity_thresholds": list(
            BBOX_LARGE_SENSITIVITY_THRESHOLDS),
        "bbox_sensitivity_one_factor_at_a_time": True,
        "primary_threshold_changed_after_diagnostics": False,
        "sensitivity_used_for_threshold_selection": False,

        "training_seed_used": flags["training_seed_used"],
        "training_started": flags["training_started"],
        "checkpoint_created": flags["checkpoint_created"],
        "pseudo_label_created": flags["pseudo_label_created"],
        "training_artifact_created": flags["training_artifact_created"],
        "training_artifacts_detected": list(training_artifacts_found),
        "locked_inputs_modified": bool(locked_inputs_modified),
        "mandatory_outputs_exist": bool(mandatory_outputs_exist),
        "machine_readable_outputs_valid": bool(machine_readable_outputs_valid),

        "undeclared_hard_fail_codes": undeclared_hard_fail_codes(
            check.code for check in preflight.checks),
        "detailed_scopes_used": list(detailed_scopes_used()),
        "structural_only_scopes_used": list(structural_scopes_used()),
        "input_sha256": dict(sorted(context.get("input_hashes", {}).items())),

        "hard_error_count": preflight.hard_error_count,
        "warning_count": preflight.warning_count,
        "checks": [check.as_dict() for check in preflight.checks],
        "warnings": [check.as_dict() for check in preflight.warnings],

        "output_contract_audit": dict(output_contract_audit or {}),
        "machine_readable_audit": [dict(entry) for entry
                                   in (machine_readable_audit or [])],
        "provisional": bool(provisional),

        "dod_candidate": (preflight.hard_error_count == 0
                          and bool(mandatory_outputs_exist)
                          and bool(machine_readable_outputs_valid)
                          and not locked_inputs_modified
                          and not provisional
                          and not flags["test_content_diagnostics"]
                          and not flags["unlabeled_hidden_gt_diagnostics"]
                          and not flags["checkpoint_created"]
                          and not flags["pseudo_label_created"]
                          and not flags["training_artifact_created"]
                          and not flags["training_seed_used"]
                          and not flags["training_started"]
                          and not undeclared_hard_fail_codes(
                              check.code for check in preflight.checks)),
        "phase_status": PHASE_STATUS_VALUE,
        "phase_status_note": (
            "OPEN_REVIEW_REQUIRED. The script never writes PASS, CLOSED or "
            "CLOSED_PASS. dod_candidate is a candidate flag only; the "
            "researcher and the GPT review decide the phase outcome."),
    }
    # A validation record must never contradict its own evidence.
    assert_validation_consistency(validation)
    return validation


def build_artifact_manifest(repo_root: Path, artifact_paths: Sequence[Path],
                            protocol_sha256: str, generated_at: str,
                            source_inputs: Mapping[str, str]
                            ) -> Dict[str, Any]:
    """Manifest of generated artifacts.

    The manifest never hashes itself: it is written last and its own path
    is excluded from ``artifact_paths``.
    """
    entries: List[Dict[str, Any]] = []
    for path in sorted(artifact_paths, key=lambda p: str(p)):
        if not path.is_file():
            continue
        entries.append({
            "artifact_path": path.relative_to(repo_root).as_posix(),
            "artifact_type": path.suffix.lstrip(".").lower(),
            "sha256": sha256_file(path),
            "byte_size": path.stat().st_size,
            "source_inputs": dict(sorted(source_inputs.items())),
            "protocol_sha256": protocol_sha256,
        })
    return {
        "phase": PHASE_ID,
        "protocol_sha256": protocol_sha256,
        "generated_at_utc": generated_at,
        "self_hash_excluded": True,
        "phase_status": PHASE_STATUS_VALUE,
        "artifact_count": len(entries),
        "artifacts": entries,
    }


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    _assert_validated_output(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=True, sort_keys=False)
        + "\n", encoding="utf-8")


def detect_training_artifacts(repo_root: Path) -> List[str]:
    """Snapshot every training artifact, checkpoint or pseudo-label present.

    Phase 3A must never CREATE any of these. HF31 / HF32 / HF33 compare a
    snapshot taken during preflight with a snapshot taken after the
    diagnostics, so files that already existed in the repository before
    the run are never reported as created by Phase 3A.
    """
    found: List[str] = []
    for directory in TRAINING_ARTIFACT_DIRS:
        candidate = repo_root / directory
        if candidate.is_dir():
            for item in sorted(candidate.rglob("*")):
                if item.is_file():
                    found.append(item.relative_to(repo_root).as_posix())
    for directory in ("reports", "plots"):
        candidate = repo_root / directory
        if candidate.is_dir():
            for item in sorted(candidate.rglob("*")):
                if (item.is_file()
                        and item.suffix.lower() in TRAINING_ARTIFACT_SUFFIXES):
                    found.append(item.relative_to(repo_root).as_posix())
    return sorted(set(found))


# =====================================================================
# FULL MODE
# =====================================================================

def run_full(protocol: Mapping[str, Any], protocol_sha256: str,
             repo_root: Path, preflight: PreflightResult,
             context: Dict[str, Any]) -> Tuple[int, Dict[str, Any]]:
    generated_at = utc_now_iso()
    style = {
        "dpi": int(dig(protocol, "plotting.dpi", 200)),
        "heatmap_cmap": str(dig(protocol, "plotting.heatmap_cmap", "Blues")),
        "bar_color": str(dig(protocol, "plotting.bar_color", "#4269A4")),
        "bar_color_secondary": str(
            dig(protocol, "plotting.bar_color_secondary", "#9EBCDA")),
        "grid_alpha": float(dig(protocol, "plotting.grid_alpha", 0.3)),
        "histogram_min_bins": int(dig(protocol, "plotting.histogram_min_bins",
                                      10)),
        "histogram_max_bins": int(dig(protocol, "plotting.histogram_max_bins",
                                      200)),
    }

    # Every output target was resolved and gated during PREFLIGHT by
    # resolve_safe_output_path. Nothing outside this map can be written.
    output_paths: Dict[str, Path] = context.get("output_paths") or {}
    required_keys = (
        ["reports_dir", "plots_dir", "report", "validation_json",
         "artifact_manifest"]
        + ["csv_{0}".format(key) for key in sorted(MANDATORY_CSV_SCHEMAS)]
        + list(MANDATORY_PLOT_TARGET_KEYS))
    missing_targets = [key for key in required_keys if key not in output_paths]
    if missing_targets:
        raise OutputContractError(
            "the output-path gate did not resolve these targets: "
            "{0}".format(missing_targets))

    output_paths["reports_dir"].mkdir(parents=True, exist_ok=True)
    output_paths["plots_dir"].mkdir(parents=True, exist_ok=True)

    class_rows, _ = build_class_distribution(context)
    class_name_by_id = context["train_categories"]
    imbalance_rows = build_class_imbalance(class_rows)
    bbox_rows, bbox_summary_rows = build_bbox_tables(context, class_name_by_id)
    per_image_rows = build_bbox_count_per_image(context)
    negative_rows = build_negative_distribution(context)
    cardinality_rows = build_label_cardinality(context)
    cooccurrence_rows = build_cooccurrence(context)
    budget_rows = build_labeled_budget_coverage(context, class_rows)
    sensitivity_rows = build_threshold_sensitivity(context, class_rows,
                                                   bbox_rows)
    split_rows = build_split_distribution(context, repo_root, protocol)

    areas = [float(row["normalized_area"]) for row in bbox_rows]
    size_counts = size_category_counts(areas, BBOX_SMALL_PRIMARY_THRESHOLD,
                                       BBOX_LARGE_PRIMARY_THRESHOLD)

    tables = {
        "class_distribution": class_rows,
        "class_imbalance": imbalance_rows,
        "bbox_distribution": bbox_rows,
        "bbox_summary": bbox_summary_rows,
        "bbox_count_per_image": per_image_rows,
        "negative_distribution": negative_rows,
        "label_cardinality": cardinality_rows,
        "cooccurrence": cooccurrence_rows,
        "labeled_budget_coverage": budget_rows,
        "threshold_sensitivity": sensitivity_rows,
        "split_distribution": split_rows,
        "size_counts": size_counts,
    }

    written: List[Path] = []

    def emit_csv(key: str, rows: Sequence[Mapping[str, Any]]) -> None:
        """Write one contract CSV using the authoritative Python schema."""
        path = output_paths["csv_{0}".format(key)]
        write_csv(path, MANDATORY_CSV_SCHEMAS[key], rows)
        written.append(path)

    emit_csv("class_distribution", class_rows)
    emit_csv("class_imbalance", imbalance_rows)
    emit_csv("bbox_distribution", bbox_rows)
    emit_csv("bbox_summary", bbox_summary_rows)
    emit_csv("bbox_count_per_image", per_image_rows)
    emit_csv("negative_distribution", negative_rows)
    emit_csv("label_cardinality", cardinality_rows)
    emit_csv("class_cooccurrence", cooccurrence_rows)
    emit_csv("labeled_budget_coverage", budget_rows)
    emit_csv("threshold_sensitivity", sensitivity_rows)
    emit_csv("split_distribution", split_rows)

    # ---- plots -------------------------------------------------------
    plot_class_distribution(class_rows, output_paths["plot_class_distribution"],
                            style)
    written.append(output_paths["plot_class_distribution"])

    plot_bbox_distribution(areas, size_counts,
                           output_paths["plot_bbox_distribution"], style)
    written.append(output_paths["plot_bbox_distribution"])

    plot_bbox_location_heatmap(
        [float(row["center_x_normalized"]) for row in bbox_rows],
        [float(row["center_y_normalized"]) for row in bbox_rows],
        output_paths["plot_bbox_location_heatmap"], style)
    written.append(output_paths["plot_bbox_location_heatmap"])

    train_total = len(context["train_image_ids"])
    train_negative = len(context["train_negatives"])
    # The resolved-target key follows the YAML key
    # output_contract.mandatory_plots.negative_image_distribution; the
    # plotting function keeps its own name and the output filename is
    # unchanged (plots/dataset/negative_image_distribution.png).
    plot_negative_distribution(
        train_total - train_negative, train_negative,
        output_paths["plot_negative_image_distribution"], style)
    written.append(output_paths["plot_negative_image_distribution"])

    # ---- report ------------------------------------------------------
    report_path = output_paths["report"]
    _assert_validated_output(report_path)
    report_text = build_report(protocol, protocol_sha256, preflight, context,
                               tables, generated_at)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report_text, encoding="utf-8")
    written.append(report_path)

    # ---- HF31 / HF32 / HF33 -------------------------------------------
    # Only artifacts that did NOT exist before the run are attributed to
    # Phase 3A.
    before = set(context.get("pre_run_training_artifacts", ()))
    training_artifacts = sorted(
        set(detect_training_artifacts(repo_root)) - before)

    # ---- HF34 input immutability --------------------------------------
    recomputed = {name: sha256_file(path)
                  for name, path in sorted(context["paths"].items())
                  if path.is_file()}
    changed = sorted(name for name, value in recomputed.items()
                     if context["input_hashes"].get(name) != value)
    locked_inputs_modified = bool(changed)

    preflight.add("HF34", "no locked input was modified during the run",
                  not changed, [], changed, "input SHA-256 before vs after")
    preflight.add("HF31", "no training artifact was created",
                  not training_artifacts, [], training_artifacts,
                  "repository scan of models/, checkpoints/, pseudo_labels/")
    preflight.add("HF32", "no checkpoint was created",
                  not [p for p in training_artifacts
                       if Path(p).suffix.lower() in TRAINING_ARTIFACT_SUFFIXES],
                  [], [p for p in training_artifacts
                       if Path(p).suffix.lower() in TRAINING_ARTIFACT_SUFFIXES],
                  "repository scan")
    preflight.add("HF33", "no pseudo-label was created",
                  not [p for p in training_artifacts
                       if "pseudo_label" in p], [],
                  [p for p in training_artifacts if "pseudo_label" in p],
                  "repository scan")

    seed_state = context.get("seed_state")
    validation_path = output_paths["validation_json"]
    manifest_path = output_paths["artifact_manifest"]

    # ------------------------------------------------------------------
    # Two-pass output-contract closure.
    #
    #   pass 1  materialise the validation JSON, then the manifest, so that
    #           both artifacts physically exist and can be audited;
    #   audit   final output-contract audit over EVERY mandatory output,
    #           including the validation JSON and the artifact manifest;
    #   pass 2  rewrite the validation JSON with the final audit result and
    #           the final HF35 verdict, then rewrite the manifest so its
    #           recorded hash of the validation JSON is the final one.
    #
    # The manifest is never hashed by anything, so this terminates: no
    # artifact depends on the manifest's own digest.
    # ------------------------------------------------------------------
    provisional_validation = build_validation(
        protocol, protocol_sha256, preflight, context, "full", generated_at,
        locked_inputs_modified, False, False, training_artifacts,
        seed_state=seed_state, provisional=True)
    write_json(validation_path, provisional_validation)
    written.append(validation_path)

    manifest = build_artifact_manifest(
        repo_root, written, protocol_sha256, generated_at,
        context["input_hashes"])
    write_json(manifest_path, manifest)

    # ---- final audit over ALL mandatory outputs -----------------------
    contract_audit = audit_output_contract(output_paths)
    machine_audit = audit_machine_readable_outputs(output_paths)
    machine_readable_ok = all(entry["status"] == "PASS"
                              for entry in machine_audit)
    mandatory_outputs_exist = bool(contract_audit["all_present"])

    preflight.add(
        "HF35", "every mandatory output of the contract exists and is "
                "structurally valid",
        mandatory_outputs_exist and machine_readable_ok,
        {"missing": [], "empty": [], "failed_structural_checks": []},
        {"missing": contract_audit["missing"],
         "empty": contract_audit["empty"],
         "failed_structural_checks": [entry["check"] for entry in machine_audit
                                      if entry["status"] == "FAIL"]},
        "final output-contract audit over "
        "{0} mandatory artifacts".format(len(contract_audit["mandatory_keys"])))

    validation = build_validation(
        protocol, protocol_sha256, preflight, context, "full", generated_at,
        locked_inputs_modified, mandatory_outputs_exist, machine_readable_ok,
        training_artifacts, seed_state=seed_state,
        output_contract_audit=contract_audit,
        machine_readable_audit=machine_audit, provisional=False)
    write_json(validation_path, validation)

    manifest = build_artifact_manifest(
        repo_root, written, protocol_sha256, generated_at,
        context["input_hashes"])
    manifest["validation_json_hashed_after_final_audit"] = True
    manifest["write_passes"] = 2
    write_json(manifest_path, manifest)

    if not manifest_path.is_file() or not validation_path.is_file():
        raise OutputContractError(
            "the validation JSON or the artifact manifest was not written")

    exit_code = EXIT_HARD_FAIL if preflight.hard_error_count else EXIT_OK
    return exit_code, validation


# =====================================================================
# Console output  [IMPLEMENTATION DETAIL] - ASCII only
# =====================================================================

def print_preflight(result: PreflightResult, mode: str) -> None:
    print("=" * 78)
    print("PHASE 3A - DATASET DIAGNOSTICS BEFORE TRAINING   mode={0}".format(
        mode))
    print("protocol_status={0}   phase_status={1}".format(
        PROTOCOL_STATUS, PHASE_STATUS_VALUE))
    print("=" * 78)
    for check in result.checks:
        print("[{0:4s}] {1:8s} {2}".format(
            check.status, check.code, check.description))
    for warning in result.warnings:
        print("[WARN] {0:8s} {1}".format(warning.code, warning.description))
    print("-" * 78)
    print("hard_error_count : {0}".format(result.hard_error_count))
    print("warning_count    : {0}".format(result.warning_count))
    if result.hard_errors:
        print("failed checks    : {0}".format(
            ", ".join(check.code for check in result.hard_errors)))
        print("")
        for check in result.hard_errors:
            print("  FAIL {0}: {1}".format(check.code, check.description))
            print("       expected : {0}".format(check.expected))
            print("       observed : {0}".format(check.observed))
            print("       evidence : {0}".format(check.evidence))
    print("-" * 78)
    print("detailed_scopes_used        : {0}".format(
        ", ".join(detailed_scopes_used()) or "none"))
    print("structural_only_scopes_used : {0}".format(
        ", ".join(structural_scopes_used()) or "none"))
    print("=" * 78)


# =====================================================================
# CLI
# =====================================================================

def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Phase 3A dataset diagnostics before training "
                    "(descriptive only; never trains, never evaluates).")
    parser.add_argument(
        "--protocol", type=Path,
        default=Path("configs/protocol/phase3A_dataset_diagnostics.yaml"),
        help="path to the Phase 3A protocol YAML")
    parser.add_argument(
        "--mode", choices=("preflight", "full"), required=True,
        help="preflight verifies inputs and locked constants; "
             "full additionally produces the diagnostics artifacts")
    parser.add_argument(
        "--repo-root", type=Path, default=None,
        help="repository root; defaults to the parent of scripts/")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    try:
        repo_root = resolve_repo_root(args.protocol, args.repo_root)
        protocol_path = args.protocol
        if not protocol_path.is_absolute():
            candidate = repo_root / protocol_path
            protocol_path = candidate if candidate.is_file() else protocol_path
        protocol, protocol_sha256 = load_protocol(protocol_path)
    except (ProtocolError, OSError) as error:
        sys.stderr.write("[FATAL] {0}\n".format(error))
        return EXIT_HARD_FAIL

    try:
        reset_scope_tripwires()
        preflight, context = run_preflight(protocol, protocol_sha256, repo_root)
    except ScopeViolationError as error:
        sys.stderr.write("[FATAL] data-use firewall: {0}\n".format(error))
        return EXIT_HARD_FAIL
    except Phase3AError as error:
        sys.stderr.write("[FATAL] {0}\n".format(error))
        return EXIT_HARD_FAIL
    except Exception as error:  # pragma: no cover - defensive
        sys.stderr.write("[FATAL] runtime failure during preflight: "
                         "{0}: {1}\n".format(type(error).__name__, error))
        return EXIT_RUNTIME

    if args.mode == "preflight":
        print_preflight(preflight, "preflight")
        print("No diagnostics artifact is produced in preflight mode.")
        print("phase_status = {0}".format(PHASE_STATUS_VALUE))
        return EXIT_HARD_FAIL if preflight.hard_error_count else EXIT_OK

    # ---- FULL --------------------------------------------------------
    if preflight.hard_error_count:
        print_preflight(preflight, "full")
        print("ABORTED before scientific diagnostics: preflight reported "
              "{0} hard error(s).".format(preflight.hard_error_count))
        print("phase_status = {0}".format(PHASE_STATUS_VALUE))
        return EXIT_HARD_FAIL

    try:
        exit_code, validation = run_full(protocol, protocol_sha256, repo_root,
                                         preflight, context)
    except ScopeViolationError as error:
        sys.stderr.write("[FATAL] data-use firewall: {0}\n".format(error))
        return EXIT_HARD_FAIL
    except OutputContractError as error:
        sys.stderr.write("[FATAL] output contract: {0}\n".format(error))
        return EXIT_RUNTIME
    except Phase3AError as error:
        sys.stderr.write("[FATAL] {0}\n".format(error))
        return EXIT_HARD_FAIL
    except Exception as error:  # pragma: no cover - defensive
        sys.stderr.write("[FATAL] runtime failure during full diagnostics: "
                         "{0}: {1}\n".format(type(error).__name__, error))
        return EXIT_RUNTIME

    print_preflight(preflight, "full")
    print("dod_candidate        : {0}".format(validation["dod_candidate"]))
    print("phase_status         : {0}".format(validation["phase_status"]))
    print("NOTE: exit code 0 does not mean Phase 3A is PASS or CLOSED. "
          "Researcher and GPT review decide.")
    return exit_code


if __name__ == "__main__":
    sys.exit(main())