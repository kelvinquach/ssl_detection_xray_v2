# -*- coding: utf-8 -*-
"""PHASE 3A - guardrail tests.

These tests protect the LOCKED scientific decisions of Phase 3A against
silent drift. They are deliberately split into two groups:

* unit / contract tests - small, deterministic, synthetic COCO fixtures
  and direct checks of the locked constants and formulas;
* integration tests     - they read the real canonical artifacts, the
  Phase 2E fixed split and the Phase 2F labeled subsets and verify their
  SHA-256 identities and counts.

Nothing here trains, evaluates, produces a pseudo-label, writes a
checkpoint or touches validation/test content.

Run (researcher, later):
    pytest -q tests/test_phase3A_dataset_diagnostics_guardrails.py \\
        --junitxml=reports/03A_guardrails_junit.xml
"""

from __future__ import annotations

import ast
import dataclasses
import hashlib
import importlib.util
import json
import re
import sys
import typing
from pathlib import Path
from typing import Any, Dict, List

import pytest
import yaml

# =====================================================================
# Repository layout and module loading
# =====================================================================

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "03A_dataset_diagnostics.py"
PROTOCOL_PATH = (REPO_ROOT / "configs" / "protocol"
                 / "phase3A_dataset_diagnostics.yaml")


MODULE_NAME = "phase3a_dataset_diagnostics"


def _load_script_module(module_name: str = MODULE_NAME):
    """Import the Phase 3A script as a module.

    The file name starts with a digit, so it cannot be imported with a
    plain ``import`` statement.

    The module MUST be registered in ``sys.modules`` under ``spec.name``
    BEFORE ``exec_module`` runs. The production script combines
    ``from __future__ import annotations`` with ``@dataclass``: dataclass
    field types are stored as strings and are resolved later through
    ``sys.modules[cls.__module__].__dict__``. Without the registration,
    ``typing.get_type_hints`` on those dataclasses raises KeyError, and
    any future ``dataclasses.fields`` introspection that resolves the
    annotations breaks in a way that is confusing to debug.
    """
    if not SCRIPT_PATH.is_file():
        pytest.fail("missing script: {0}".format(SCRIPT_PATH))
    if module_name in sys.modules:
        return sys.modules[module_name]
    spec = importlib.util.spec_from_file_location(module_name, SCRIPT_PATH)
    if spec is None or spec.loader is None:
        pytest.fail("cannot build an import spec for {0}".format(SCRIPT_PATH))
    module = importlib.util.module_from_spec(spec)
    # Register BEFORE executing the module body.
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(spec.name, None)
        raise
    return module


D = _load_script_module()


@pytest.fixture(scope="module")
def protocol() -> Dict[str, Any]:
    if not PROTOCOL_PATH.is_file():
        pytest.fail("missing protocol: {0}".format(PROTOCOL_PATH))
    return yaml.safe_load(PROTOCOL_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def script_source() -> str:
    return SCRIPT_PATH.read_text(encoding="utf-8")


# =====================================================================
# Locked constants restated independently of the script and the YAML
# =====================================================================

EXPECTED_RARE_THRESHOLD = 0.05
EXPECTED_RARE_SENSITIVITY = [0.01, 0.05, 0.10]
EXPECTED_SMALL_THRESHOLD = 0.01
EXPECTED_LARGE_THRESHOLD = 0.10
EXPECTED_SMALL_SENSITIVITY = [0.005, 0.01, 0.02]
EXPECTED_LARGE_SENSITIVITY = [0.05, 0.10, 0.20]

EXPECTED_TRAIN_IMAGES = 3426
EXPECTED_TRAIN_ANNOTATIONS = 25260
EXPECTED_TRAIN_NO_FINDING = 350
EXPECTED_VAL = (734, 5399, 75)
EXPECTED_TEST = (734, 5437, 75)
EXPECTED_CANONICAL = (4894, 36096, 14, 500)

EXPECTED_MASTER_JPG_SHA = (
    "f587152278f713460ff1e727a2912248a47052f6abc48de8f7bad6e8a63b94c0")
EXPECTED_TRAIN_SHA = (
    "0f3c37a6f1b5bcc6971b01fd4c69c7a2a3a1e8bee145c11488c7a658dcbbebe3")
EXPECTED_VAL_SHA = (
    "33064f47ba690be13e9d418d8ec5d4a6a2bec8482c24ea3eadf83fb33d01762a")
EXPECTED_TEST_SHA = (
    "e1a73110e92af2656276d6c532035afe474b6ac6f2b2f03849834c036e1c00a4")
EXPECTED_LABELED_SHA = {
    "1pct": "1e267c547a8f535f6a69088da4735bd12c0188fae1b49e9d785b3e1e6883df98",
    "5pct": "ebb98f32cad612fe48adbbd6b5da8aa5db10bec478d1c7fd9288a9a93bbc3763",
    "10pct": "6814b5b9ba8dfcd4af483d97c1a6b7e7ab26914bdd02bbb923502b43ecf2fdf6",
    "20pct": "6a8b5bf9c59baea41eefb0611e15a387257255b6fc59a3de607bfc2e80801a14",
}
EXPECTED_LABELED_IMAGES = {"1pct": 34, "5pct": 171, "10pct": 343, "20pct": 685}
EXPECTED_LABELED_NEGATIVES = {"1pct": 3, "5pct": 17, "10pct": 35, "20pct": 70}

MISSING_INPUT_HINT = (
    "Canonical input artifact not found. The integration guardrails must be "
    "run inside the repository that holds the Phase 2D/2E/2F artifacts.")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require(path: Path) -> Path:
    if not path.is_file():
        pytest.fail("{0}\nmissing: {1}".format(MISSING_INPUT_HINT, path))
    return path


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(_require(path).read_text(encoding="utf-8"))


# =====================================================================
# Synthetic COCO fixtures - deterministic, tiny, no real data needed
# =====================================================================

@pytest.fixture()
def synthetic_annotations() -> List[Dict[str, Any]]:
    """Two classes over three images, with a duplicated class per image.

    image 1: two boxes of class 1 and one box of class 2
    image 2: one box of class 1
    image 3: one box of class 2
    """
    return [
        {"id": 1, "image_id": 1, "category_id": 1, "bbox": [0.0, 0.0, 10.0, 10.0]},
        {"id": 2, "image_id": 1, "category_id": 1, "bbox": [10.0, 0.0, 10.0, 10.0]},
        {"id": 3, "image_id": 1, "category_id": 2, "bbox": [0.0, 10.0, 10.0, 10.0]},
        {"id": 4, "image_id": 2, "category_id": 1, "bbox": [0.0, 0.0, 20.0, 20.0]},
        {"id": 5, "image_id": 3, "category_id": 2, "bbox": [0.0, 0.0, 5.0, 5.0]},
    ]


# =====================================================================
# A. protocol locked status
# =====================================================================

def test_a_protocol_locked_status(protocol):
    assert protocol["phase"]["protocol_status"] == "RESEARCHER_APPROVED_LOCKED"
    assert D.PROTOCOL_STATUS == "RESEARCHER_APPROVED_LOCKED"
    assert str(protocol["phase"]["id"]) == "3A"
    assert protocol["phase"]["self_declared_closure_allowed"] is False
    assert protocol["phase"]["analysis_type"] == (
        "DESCRIPTIVE_PRE_TRAINING_DATASET_DIAGNOSTICS")
    for excluded in ("model_training", "model_evaluation",
                     "hyperparameter_search", "pseudo_label_generation",
                     "ablation_study", "threshold_optimization"):
        assert excluded in protocol["phase"]["is_not"]


# =====================================================================
# B. primary rare threshold locked
# =====================================================================

def test_b_primary_rare_threshold_locked(protocol):
    rare = protocol["rare_class_definition"]
    assert float(rare["primary_threshold"]) == EXPECTED_RARE_THRESHOLD
    assert D.RARE_PRIMARY_THRESHOLD == EXPECTED_RARE_THRESHOLD
    assert rare["definition_type"] == "PRE_SPECIFIED_OPERATIONAL_DEFINITION"
    assert rare["primary_measure"] == "IMAGE_LEVEL_SUPPORT_PREVALENCE"
    assert rare["comparison"] == "strictly_less_than"
    assert int(rare["denominator"]) == EXPECTED_TRAIN_IMAGES
    assert rare["bbox_count_may_define_rarity"] is False
    assert rare["threshold_may_be_chosen_from_histogram"] is False
    assert rare["clinical_rarity_claim_allowed"] is False
    assert rare["count_image_once_per_class"] is True


# =====================================================================
# C. primary bbox thresholds locked
# =====================================================================

def test_c_primary_bbox_thresholds_locked(protocol):
    size = protocol["bbox_size_definition"]
    assert float(size["small_threshold"]) == EXPECTED_SMALL_THRESHOLD
    assert float(size["large_threshold"]) == EXPECTED_LARGE_THRESHOLD
    assert D.BBOX_SMALL_PRIMARY_THRESHOLD == EXPECTED_SMALL_THRESHOLD
    assert D.BBOX_LARGE_PRIMARY_THRESHOLD == EXPECTED_LARGE_THRESHOLD
    assert size["definition_type"] == "PRE_SPECIFIED_OPERATIONAL_DEFINITION"
    assert size["basis"] == "normalized_area"
    assert size["coco_pixel_area_rule_used_as_primary"] is False
    assert list(size["category_order"]) == ["small", "medium", "large"]
    assert protocol["bbox_geometry"]["primary_scale_descriptor"] == \
        "normalized_area"


# =====================================================================
# D. sensitivity thresholds locked
# =====================================================================

def test_d_sensitivity_thresholds_locked(protocol):
    sensitivity = protocol["sensitivity_analysis"]
    assert [float(v) for v in sensitivity["rare"]["thresholds"]] == \
        EXPECTED_RARE_SENSITIVITY
    assert [float(v) for v in sensitivity["bbox_small"]["thresholds"]] == \
        EXPECTED_SMALL_SENSITIVITY
    assert [float(v) for v in sensitivity["bbox_large"]["thresholds"]] == \
        EXPECTED_LARGE_SENSITIVITY
    assert float(sensitivity["rare"]["primary"]) == EXPECTED_RARE_THRESHOLD
    assert float(sensitivity["bbox_small"]["primary"]) == \
        EXPECTED_SMALL_THRESHOLD
    assert float(sensitivity["bbox_large"]["primary"]) == \
        EXPECTED_LARGE_THRESHOLD
    assert float(sensitivity["bbox_small"]["large_threshold_fixed_at"]) == \
        EXPECTED_LARGE_THRESHOLD
    assert float(sensitivity["bbox_large"]["small_threshold_fixed_at"]) == \
        EXPECTED_SMALL_THRESHOLD
    assert sensitivity["enabled"] is True
    assert sensitivity["role"] == "SECONDARY_NON_GATING"
    assert sensitivity["used_for_threshold_selection"] is False
    assert sensitivity["primary_threshold_changed_after_diagnostics"] is False

    assert list(D.RARE_SENSITIVITY_THRESHOLDS) == EXPECTED_RARE_SENSITIVITY
    assert list(D.BBOX_SMALL_SENSITIVITY_THRESHOLDS) == \
        EXPECTED_SMALL_SENSITIVITY
    assert list(D.BBOX_LARGE_SENSITIVITY_THRESHOLDS) == \
        EXPECTED_LARGE_SENSITIVITY


# =====================================================================
# E. bbox sensitivity is one factor at a time
# =====================================================================

def test_e_bbox_sensitivity_one_factor_at_a_time(protocol, script_source):
    sensitivity = protocol["sensitivity_analysis"]
    assert sensitivity["one_factor_at_a_time"] is True
    assert sensitivity["cartesian_grid_allowed"] is False

    # The implementation must never emit 3 x 3 combinations: exactly three
    # rows per size category for each of the two one-factor sweeps.
    areas = [0.001, 0.005, 0.02, 0.05, 0.2]
    bbox_rows = [{"normalized_area": value} for value in areas]
    class_rows = [{
        "category_id": 1, "class_name": "A", "image_count": 100,
        "image_prevalence": 100.0 / EXPECTED_TRAIN_IMAGES, "rare_flag": True,
    }]
    rows = D.build_threshold_sensitivity({}, class_rows, bbox_rows)

    small_thresholds = sorted({
        row["threshold"] for row in rows
        if row["analysis_type"] == "bbox_small_boundary_sensitivity"})
    large_thresholds = sorted({
        row["threshold"] for row in rows
        if row["analysis_type"] == "bbox_large_boundary_sensitivity"})
    assert small_thresholds == EXPECTED_SMALL_SENSITIVITY
    assert large_thresholds == EXPECTED_LARGE_SENSITIVITY

    small_rows = [row for row in rows
                  if row["analysis_type"] == "bbox_small_boundary_sensitivity"]
    large_rows = [row for row in rows
                  if row["analysis_type"] == "bbox_large_boundary_sensitivity"]
    # 3 thresholds x 3 size categories, never 3 x 3 x 3 combinations.
    assert len(small_rows) == 9
    assert len(large_rows) == 9
    assert "itertools.product" not in script_source


# =====================================================================
# F. no training seed configured
# =====================================================================

def test_f_no_training_seed_configured(protocol, script_source):
    assert protocol["scope_policy"]["training_seed_used"] is False
    assert protocol["scope_policy"]["repeat_over_training_seeds"] is False
    assert protocol["determinism"]["training_seed_used"] is False
    assert protocol["determinism"]["rng_used"] is False
    assert protocol["forbidden_operations"]["use_of_training_seed"] is False

    # No random number generator may be seeded or drawn from.
    for forbidden in ("np.random.seed", "random.seed", "torch.manual_seed",
                      "np.random.default_rng", "import random",
                      "training_seed ="):
        assert forbidden not in script_source, \
            "forbidden RNG/seed usage in script: {0}".format(forbidden)


# =====================================================================
# G. hidden unlabeled ground truth is forbidden
# =====================================================================

def test_g_hidden_unlabeled_gt_forbidden(protocol, script_source):
    assert protocol["scope_policy"]["unlabeled_hidden_gt"]["usage"] == \
        "PROHIBITED"
    assert protocol["scope_policy"]["unlabeled_hidden_gt"][
        "reverse_lookup_into_train_allowed"] is False
    assert not protocol["input_artifacts"]["unlabeled_subsets"]

    configured = D.collect_protocol_paths(protocol)
    assert D.find_forbidden_unlabeled_paths(configured) == []

    # The detector must catch a real unlabeled COCO file ...
    assert D.find_forbidden_unlabeled_paths([
        "data/processed/coco/unlabeled_splits/instances_unlabeled_5pct.json"])
    assert D.find_forbidden_unlabeled_paths(["x/instances_unlabeled_1pct.json"])
    # ... and must NOT flag Phase 2F evidence files whose name merely
    # contains the word "unlabeled".
    assert D.find_forbidden_unlabeled_paths([
        "reports/02F_labeled_unlabeled_validation_report.json",
        "configs/protocol/phase2F_labeled_unlabeled.yaml"]) == []

    # The firewall must refuse an unlabeled scope outright.
    with pytest.raises(D.ScopeViolationError):
        D.load_detailed_coco(Path("whatever.json"), "unlabeled_5pct")

    # The script must never build a path to an unlabeled COCO file. The
    # only place the word may appear is inside the refusal patterns of
    # FORBIDDEN_UNLABELED_PATH_PATTERNS.
    for line in script_source.splitlines():
        if "instances_unlabeled" in line or "unlabeled_splits" in line:
            assert "FORBIDDEN_UNLABELED_PATH_PATTERNS" in line or \
                line.strip().startswith(("r\"", "#")), \
                "script builds an unlabeled path: {0}".format(line.strip())


# =====================================================================
# H. detailed test/validation content diagnostics are forbidden
# =====================================================================

def test_h_test_content_diagnostics_forbidden(protocol):
    assert protocol["scope_policy"]["test_usage"] == "STRUCTURAL_INTEGRITY_ONLY"
    assert protocol["scope_policy"]["validation_usage"] == \
        "STRUCTURAL_INTEGRITY_ONLY"
    assert protocol["scope_policy"]["test_role"] == "FINAL_EVALUATION_ONLY"
    assert protocol["forbidden_operations"]["detailed_test_diagnostics"] is False
    assert protocol["forbidden_operations"][
        "detailed_validation_diagnostics"] is False

    assert "val" not in D.DETAILED_SCOPE_ALLOWED
    assert "test" not in D.DETAILED_SCOPE_ALLOWED
    for scope in ("val", "test", "canonical"):
        with pytest.raises(D.ScopeViolationError):
            D.load_detailed_coco(Path("whatever.json"), scope)

    # The structural summary must not be able to carry content diagnostics.
    field_names = set(D.StructuralSummary.__dataclass_fields__)
    assert field_names == {
        "scope", "path", "sha256", "image_count", "annotation_count",
        "negative_image_count", "category_ids", "category_names"}
    for forbidden in ("annotations", "images", "bboxes", "class_distribution",
                      "areas"):
        assert forbidden not in field_names


# =====================================================================
# I. rare = image-level support formula
# =====================================================================

def test_i_rare_image_level_support_formula(synthetic_annotations):
    support = D.class_image_support(synthetic_annotations)
    # class 1 appears in images 1 and 2 -> 2 unique images, although it has
    # three bounding boxes.
    assert support[1] == {1, 2}
    assert support[2] == {1, 3}
    assert len(support[1]) == 2
    assert D.class_bbox_counts(synthetic_annotations) == {1: 3, 2: 2}

    assert D.image_prevalence(171, EXPECTED_TRAIN_IMAGES) == \
        171 / EXPECTED_TRAIN_IMAGES
    assert D.is_rare(0.049, EXPECTED_RARE_THRESHOLD) is True
    assert D.is_rare(0.05, EXPECTED_RARE_THRESHOLD) is False
    assert D.is_rare(0.051, EXPECTED_RARE_THRESHOLD) is False


# =====================================================================
# J. rare boundary 171 vs 172
# =====================================================================

def test_j_rare_boundary_171_vs_172():
    rare_at_171 = D.is_rare(
        D.image_prevalence(171, EXPECTED_TRAIN_IMAGES), EXPECTED_RARE_THRESHOLD)
    rare_at_172 = D.is_rare(
        D.image_prevalence(172, EXPECTED_TRAIN_IMAGES), EXPECTED_RARE_THRESHOLD)
    assert rare_at_171 is True
    assert rare_at_172 is False
    assert D.RARE_BOUNDARY_MAX_RARE_COUNT == 171
    assert D.RARE_BOUNDARY_MIN_NONRARE_COUNT == 172


# =====================================================================
# K. normalized area formula
# =====================================================================

def test_k_bbox_normalized_area_formula():
    assert D.normalized_area(10.0, 20.0, 100.0, 200.0) == \
        (10.0 * 20.0) / (100.0 * 200.0)
    assert D.normalized_area(50.0, 50.0, 100.0, 100.0) == 0.25
    assert D.normalized_width(10.0, 100.0) == 0.1
    assert D.normalized_height(20.0, 200.0) == 0.1
    assert D.aspect_ratio(30.0, 15.0) == 2.0
    # normalized area never depends on the raw pixel area alone
    assert D.normalized_area(10.0, 10.0, 100.0, 100.0) != \
        D.normalized_area(10.0, 10.0, 200.0, 200.0)


# =====================================================================
# L. bbox size boundary exactness
# =====================================================================

def test_l_bbox_size_boundary_exactness():
    classify = D.classify_bbox_size
    assert classify(0.0) == "small"
    assert classify(0.009999) == "small"
    assert classify(0.01) == "medium"        # exactly at the small boundary
    assert classify(0.05) == "medium"
    assert classify(0.099999) == "medium"
    assert classify(0.10) == "large"         # exactly at the large boundary
    assert classify(0.5) == "large"
    assert classify(1.0) == "large"

    counts = D.size_category_counts(
        [0.001, 0.01, 0.05, 0.10, 0.5],
        EXPECTED_SMALL_THRESHOLD, EXPECTED_LARGE_THRESHOLD)
    assert counts == {"small": 1, "medium": 2, "large": 2}


# =====================================================================
# M. bbox centre formula
# =====================================================================

def test_m_bbox_center_formula():
    center = D.bbox_center_normalized(0.0, 0.0, 100.0, 200.0, 200.0, 400.0)
    assert center == (0.25, 0.25)
    center = D.bbox_center_normalized(100.0, 200.0, 100.0, 200.0, 200.0, 400.0)
    assert center == (0.75, 0.75)
    # top-left origin: a box at the top of the image has a small centre_y
    top = D.bbox_center_normalized(0.0, 0.0, 10.0, 10.0, 100.0, 100.0)
    bottom = D.bbox_center_normalized(0.0, 90.0, 10.0, 10.0, 100.0, 100.0)
    assert top[1] < bottom[1]
    assert D.HEATMAP_BINS_X == 50
    assert D.HEATMAP_BINS_Y == 50


# =====================================================================
# N. bbox validity rules
# =====================================================================

def test_n_bbox_validity_rules():
    assert D.bbox_validity_violations(0.0, 0.0, 10.0, 10.0, 100.0, 100.0) == []
    assert D.bbox_validity_violations(90.0, 90.0, 10.0, 10.0, 100.0,
                                      100.0) == []
    assert "w_not_positive" in D.bbox_validity_violations(
        0.0, 0.0, 0.0, 10.0, 100.0, 100.0)
    assert "h_not_positive" in D.bbox_validity_violations(
        0.0, 0.0, 10.0, 0.0, 100.0, 100.0)
    assert "x_negative" in D.bbox_validity_violations(
        -1.0, 0.0, 10.0, 10.0, 100.0, 100.0)
    assert "y_negative" in D.bbox_validity_violations(
        0.0, -1.0, 10.0, 10.0, 100.0, 100.0)
    assert "x_plus_w_exceeds_image_width" in D.bbox_validity_violations(
        95.0, 0.0, 10.0, 10.0, 100.0, 100.0)
    assert "y_plus_h_exceeds_image_height" in D.bbox_validity_violations(
        0.0, 95.0, 10.0, 10.0, 100.0, 100.0)
    # no repair of any kind is offered by the API
    for forbidden in ("clamp", "repair_bbox", "fix_bbox"):
        assert not hasattr(D, forbidden)


# =====================================================================
# O. label cardinality counts unique classes
# =====================================================================

def test_o_label_cardinality_unique_classes(synthetic_annotations):
    presence = D.image_class_presence(synthetic_annotations)
    cardinality = D.label_cardinality_per_image(presence, [1, 2, 3, 4])
    # image 1 has three boxes but only two unique classes
    assert cardinality[1] == 2
    assert cardinality[2] == 1
    assert cardinality[3] == 1
    # image 4 is a zero-GT negative image
    assert cardinality[4] == 0
    counts = D.bbox_count_per_image(synthetic_annotations, [1, 2, 3, 4])
    assert counts[1] == 3
    assert cardinality[1] != counts[1]


# =====================================================================
# P. co-occurrence uses unique image-level presence
# =====================================================================

def test_p_cooccurrence_unique_image_presence(synthetic_annotations):
    presence = D.image_class_presence(synthetic_annotations)
    # only image 1 contains both class 1 and class 2, even though image 1
    # holds three bounding boxes
    assert D.cooccurrence_count(presence, 1, 2) == 1
    assert D.cooccurrence_count(presence, 2, 1) == 1
    assert D.cooccurrence_count(presence, 1, 99) == 0


# =====================================================================
# Q. Jaccard formula
# =====================================================================

def test_q_jaccard_formula():
    assert D.jaccard(2, 2, 1) == 1.0 / 3.0
    assert D.jaccard(10, 10, 10) == 1.0
    assert D.jaccard(10, 5, 0) == 0.0
    assert D.jaccard(0, 0, 0) is None
    assert D.EXPECTED_COOCCURRENCE_PAIRS == 91
    assert D.EXPECTED_COOCCURRENCE_PAIRS == (14 * 13) // 2


# =====================================================================
# R. sensitivity never mutates the primary definitions
# =====================================================================

def test_r_sensitivity_does_not_mutate_primary_definition():
    rare_before = D.RARE_PRIMARY_THRESHOLD
    small_before = D.BBOX_SMALL_PRIMARY_THRESHOLD
    large_before = D.BBOX_LARGE_PRIMARY_THRESHOLD

    class_rows = [
        {"category_id": 1, "class_name": "A", "image_count": 100,
         "image_prevalence": 100 / EXPECTED_TRAIN_IMAGES, "rare_flag": True},
        {"category_id": 2, "class_name": "B", "image_count": 2000,
         "image_prevalence": 2000 / EXPECTED_TRAIN_IMAGES, "rare_flag": False},
    ]
    bbox_rows = [{"normalized_area": value}
                 for value in (0.001, 0.02, 0.15, 0.3)]
    rows = D.build_threshold_sensitivity({}, class_rows, bbox_rows)

    assert D.RARE_PRIMARY_THRESHOLD == rare_before
    assert D.BBOX_SMALL_PRIMARY_THRESHOLD == small_before
    assert D.BBOX_LARGE_PRIMARY_THRESHOLD == large_before
    # the primary rare flag reported alongside every sensitivity row must
    # remain the fixed-train primary flag
    for row in rows:
        if row["analysis_type"] == "rare_threshold_per_class":
            expected = True if row["category_id"] == 1 else False
            assert row["primary_rare_flag"] is expected
    primaries = {row["threshold"] for row in rows if row["is_primary"] is True}
    assert primaries == {EXPECTED_RARE_THRESHOLD, EXPECTED_SMALL_THRESHOLD,
                         EXPECTED_LARGE_THRESHOLD}


# =====================================================================
# S. labeled-budget nesting logic
# =====================================================================

def test_s_labeled_budget_nesting_logic(protocol):
    assert D.is_nested([{1}, {1, 2}, {1, 2, 3}]) is True
    assert D.is_nested([{1, 4}, {1, 2}, {1, 2, 3}]) is False
    assert D.is_nested([set(), {1}]) is True
    assert list(D.BUDGET_KEYS) == ["1pct", "5pct", "10pct", "20pct"]
    assert D.LABELED_IMAGE_COUNTS == EXPECTED_LABELED_IMAGES
    assert D.LABELED_NO_FINDING_COUNTS == EXPECTED_LABELED_NEGATIVES

    budget = protocol["labeled_budget_diagnostics"]
    assert budget["rare_flag_source"] == "FIXED_TRAIN_ONLY"
    assert budget["recompute_rare_within_budget"] is False
    assert budget["uses_unlabeled_hidden_gt"] is False
    assert budget["independent_recomputation_required"] is True
    assert budget["cross_check_is_gating"] is False
    assert budget["representativeness"][
        "may_trigger_rebuild_of_labeled_subsets"] is False


# =====================================================================
# T. output contract
# =====================================================================

def test_t_output_contract(protocol):
    contract = protocol["output_contract"]
    for key in ("class_distribution", "class_imbalance", "bbox_distribution",
                "bbox_summary", "bbox_count_per_image",
                "negative_distribution", "label_cardinality",
                "class_cooccurrence", "labeled_budget_coverage",
                "threshold_sensitivity", "split_distribution"):
        assert key in contract["mandatory_csv"]
        assert contract["mandatory_csv"][key].startswith("reports/03A_")

    assert contract["mandatory_report"] == "reports/dataset_analysis_report.md"
    assert contract["mandatory_validation_json"] == (
        "reports/03A_dataset_diagnostics_validation.json")
    assert contract["mandatory_artifact_manifest"] == (
        "reports/03A_artifact_manifest.json")
    assert contract["guardrail_junit"] == "reports/03A_guardrails_junit.xml"

    for key in ("class_distribution", "bbox_distribution",
                "bbox_location_heatmap", "negative_image_distribution"):
        assert contract["mandatory_plots"][key].startswith("plots/dataset/")

    required_columns = [
        "annotation_id", "image_id", "category_id", "class_name", "x", "y",
        "w", "h", "image_width", "image_height", "raw_area",
        "normalized_width", "normalized_height", "normalized_area",
        "aspect_ratio", "center_x_normalized", "center_y_normalized",
        "primary_size_category"]
    assert list(contract["bbox_distribution_columns"]) == required_columns
    assert list(D.BBOX_DISTRIBUTION_COLUMNS) == required_columns
    assert contract["patient_identifiers_allowed_in_outputs"] is False

    # declared CSV schemas and structural row counts
    for key, schema in D.MANDATORY_CSV_SCHEMAS.items():
        assert key in contract["csv_schemas"]
        assert list(contract["csv_schemas"][key]) == list(schema)
    assert set(contract["csv_schemas"]) == set(D.MANDATORY_CSV_SCHEMAS)
    for key, count in D.EXPECTED_CSV_ROW_COUNTS.items():
        assert key in contract["expected_row_counts"]
        assert contract["expected_row_counts"][key] == count
    assert contract["expected_row_counts"]["class_distribution"] == 14
    assert contract["expected_row_counts"]["bbox_distribution"] == 25260
    assert contract["expected_row_counts"]["class_cooccurrence"] == 91
    assert contract["expected_row_counts"]["labeled_budget_coverage"] == 56

    # output-path safety policy
    safety = contract["output_path_safety"]
    assert safety["enforced_at"] == "PREFLIGHT"
    assert safety["preflight_check_id"] == "PF12"
    for key in ("absolute_paths_allowed", "parent_traversal_allowed",
                "home_relative_paths_allowed", "unregistered_write_allowed"):
        assert safety[key] is False
    for key in ("must_resolve_within_repo_root",
                "must_be_inside_writable_directory",
                "must_not_be_inside_forbidden_directory"):
        assert safety[key] is True
    assert list(contract["writable_directories"]) == \
        list(D.WRITABLE_OUTPUT_DIRECTORIES)
    for forbidden in ("data/processed/coco", "data/manifests",
                      "configs/protocol", "models", "checkpoints",
                      "pseudo_labels"):
        assert forbidden in contract["forbidden_write_paths"]

    # every hard-fail condition of the protocol section 25 is declared
    codes = protocol["guardrails"]["hard_fail_conditions"]
    assert len(codes) == 35
    for index in range(1, 36):
        assert "HF{0:02d}".format(index) in codes


# =====================================================================
# U. the phase status can never be auto-closed
# =====================================================================

def test_u_phase_status_cannot_auto_close(protocol, script_source):
    assert protocol["phase_status_policy"]["phase_status_value"] == \
        "OPEN_REVIEW_REQUIRED"
    assert protocol["phase_status_policy"]["script_may_self_close"] is False
    for forbidden in ("PASS", "CLOSED", "CLOSED_PASS"):
        assert forbidden in protocol["phase_status_policy"][
            "forbidden_status_values"]

    assert D.PHASE_STATUS_VALUE == "OPEN_REVIEW_REQUIRED"
    assert set(D.FORBIDDEN_PHASE_STATUS_VALUES) == {"PASS", "CLOSED",
                                                    "CLOSED_PASS"}
    # the script must never assign a closing status to phase_status
    assert not re.search(r"phase_status\"?\s*[:=]\s*\"?(PASS|CLOSED"
                         r"|CLOSED_PASS)\b", script_source)

    validation = D.build_validation(
        protocol, "0" * 64, D.PreflightResult(), {"input_hashes": {}},
        "full", "1970-01-01T00:00:00Z", False, True, True, [])
    assert validation["phase_status"] == "OPEN_REVIEW_REQUIRED"
    assert validation["protocol_status"] == "RESEARCHER_APPROVED_LOCKED"
    for key in ("validation_design_usage", "test_design_usage",
                "test_content_diagnostics", "unlabeled_hidden_gt_diagnostics",
                "training_seed_used", "training_started", "checkpoint_created",
                "pseudo_label_created", "locked_inputs_modified",
                "primary_threshold_changed_after_diagnostics",
                "sensitivity_used_for_threshold_selection"):
        assert validation[key] is False
    assert validation["sensitivity_analysis_enabled"] is True


# =====================================================================
# U2. the validation contract carries every mandatory field
# =====================================================================

def test_u2_validation_contract_fields(protocol):
    validation = D.build_validation(
        protocol, "0" * 64, D.PreflightResult(), {"input_hashes": {}},
        "full", "1970-01-01T00:00:00Z", False, True, True, [])
    required_fields = [
        "phase", "protocol_version", "protocol_status", "protocol_sha256",
        "canonical_identity_pass", "train_identity_pass",
        "validation_identity_pass", "test_identity_pass",
        "canonical_counts_pass", "category_mapping_pass",
        "no_finding_policy_pass", "bbox_validity_pass",
        "phase2E_split_identity_pass", "phase2F_membership_identity_pass",
        "phase2F_nested_pass", "primary_diagnostic_scope_train_only_pass",
        "validation_design_usage", "test_design_usage",
        "test_content_diagnostics", "unlabeled_hidden_gt_diagnostics",
        "rare_definition_type", "rare_primary_threshold",
        "bbox_size_definition_type", "bbox_small_primary_threshold",
        "bbox_large_primary_threshold", "sensitivity_analysis_enabled",
        "sensitivity_analysis_role", "rare_sensitivity_thresholds",
        "bbox_small_sensitivity_thresholds",
        "bbox_large_sensitivity_thresholds",
        "bbox_sensitivity_one_factor_at_a_time",
        "primary_threshold_changed_after_diagnostics",
        "sensitivity_used_for_threshold_selection", "training_seed_used",
        "training_started", "checkpoint_created", "pseudo_label_created",
        "locked_inputs_modified", "mandatory_outputs_exist",
        "machine_readable_outputs_valid", "hard_error_count", "warning_count",
        "dod_candidate", "phase_status"]
    for field_name in required_fields:
        assert field_name in validation, "missing field: {0}".format(field_name)

    assert validation["rare_definition_type"] == \
        "PRE_SPECIFIED_OPERATIONAL_DEFINITION"
    assert validation["bbox_size_definition_type"] == \
        "PRE_SPECIFIED_OPERATIONAL_DEFINITION"
    assert validation["rare_primary_threshold"] == EXPECTED_RARE_THRESHOLD
    assert validation["bbox_small_primary_threshold"] == \
        EXPECTED_SMALL_THRESHOLD
    assert validation["bbox_large_primary_threshold"] == \
        EXPECTED_LARGE_THRESHOLD
    assert validation["sensitivity_analysis_enabled"] is True
    assert validation["sensitivity_analysis_role"] == "SECONDARY_NON_GATING"
    assert validation["rare_sensitivity_thresholds"] == \
        EXPECTED_RARE_SENSITIVITY
    assert validation["bbox_small_sensitivity_thresholds"] == \
        EXPECTED_SMALL_SENSITIVITY
    assert validation["bbox_large_sensitivity_thresholds"] == \
        EXPECTED_LARGE_SENSITIVITY
    assert validation["bbox_sensitivity_one_factor_at_a_time"] is True


# =====================================================================
# Determinism and terminology
# =====================================================================

def test_determinism_of_derived_tables(protocol):
    assert protocol["determinism"]["rng_used"] is False
    assert protocol["determinism"]["explicit_sorting_required"] is True
    assert protocol["determinism"]["csv_row_order_deterministic"] is True
    assert protocol["determinism"]["timestamp_in_scientific_computation"] \
        is False
    assert D.SD_DDOF == 1
    assert D.PERCENTILE_METHOD == "linear"

    values = [1.0, 2.0, 3.0, 4.0, 5.0]
    first = D.continuous_stats(values)
    second = D.continuous_stats(values)
    assert first == second
    assert first["n"] == 5
    assert first["mean"] == 3.0
    # sample standard deviation with ddof = 1
    assert abs(first["sd"] - 1.5811388300841898) < 1e-12
    assert first["median"] == 3.0

    order_a = D.ordered_class_rows({1: "A", 2: "B", 3: "C"},
                                   {1: 10, 2: 10, 3: 5})
    # tie between category 1 and 2 is broken by ascending category_id
    assert order_a == [1, 2, 3]


def test_forbidden_terminology_absent(script_source):
    lowered = script_source.lower()
    for phrase in ("number of unique lesions", "coco small", "coco medium",
                   "coco large"):
        assert phrase not in lowered, \
            "forbidden terminology in script: {0}".format(phrase)


def test_no_training_or_inference_imports(script_source):
    for forbidden in ("import torch", "from torch", "import tensorflow",
                      "mmdet", "detectron2", "import seaborn",
                      "from seaborn"):
        assert forbidden not in script_source, \
            "forbidden dependency in script: {0}".format(forbidden)


def test_script_never_writes_into_locked_directories(script_source, protocol):
    for forbidden in protocol["output_contract"]["forbidden_write_paths"]:
        pattern = r"write_(csv|json)\([^)]*{0}".format(re.escape(forbidden))
        assert not re.search(pattern, script_source), \
            "script appears to write into a locked path: {0}".format(forbidden)


# =====================================================================
# V. integration - canonical hash and counts
# =====================================================================

def test_v_integration_canonical_hash_and_counts(protocol):
    path = REPO_ROOT / protocol["input_artifacts"]["canonical"][
        "coco_master_jpg"]
    assert _sha256(_require(path)) == EXPECTED_MASTER_JPG_SHA
    document = _load_json(path)
    images, annotations, categories, no_finding = EXPECTED_CANONICAL
    assert len(document["images"]) == images
    assert len(document["annotations"]) == annotations
    assert len(document["categories"]) == categories
    assert sum(1 for image in document["images"]
               if bool(image.get("is_negative"))) == no_finding
    assert sorted(int(c["id"]) for c in document["categories"]) == \
        list(range(1, 15))
    assert D.contains_no_finding_category(document) == []

    # canonical No Finding images must be strictly zero-GT (integrity only)
    negative_ids = D.negative_image_ids(document)
    annotated_ids = {int(annotation["image_id"])
                     for annotation in document["annotations"]}
    assert negative_ids & annotated_ids == set(), (
        "canonical No Finding images carry ground-truth annotations: "
        "{0}".format(sorted(negative_ids & annotated_ids)[:20]))
    assert len(negative_ids) == no_finding


# =====================================================================
# W. integration - fixed split hashes and counts
# =====================================================================

def test_w_integration_fixed_split_hash_and_counts(protocol):
    split_paths = protocol["input_artifacts"]["fixed_split"]
    train_path = REPO_ROOT / split_paths["train"]
    val_path = REPO_ROOT / split_paths["val"]
    test_path = REPO_ROOT / split_paths["test"]

    assert _sha256(_require(train_path)) == EXPECTED_TRAIN_SHA
    assert _sha256(_require(val_path)) == EXPECTED_VAL_SHA
    assert _sha256(_require(test_path)) == EXPECTED_TEST_SHA

    train = _load_json(train_path)
    assert len(train["images"]) == EXPECTED_TRAIN_IMAGES
    assert len(train["annotations"]) == EXPECTED_TRAIN_ANNOTATIONS
    negatives = D.negative_image_ids(train)
    assert len(negatives) == EXPECTED_TRAIN_NO_FINDING
    annotated = {int(a["image_id"]) for a in train["annotations"]}
    assert negatives & annotated == set()
    assert D.contains_no_finding_category(train) == []

    # validation and test are only ever summarised structurally
    D.reset_scope_tripwires()
    val_summary = D.structural_summary(val_path, "val")
    test_summary = D.structural_summary(test_path, "test")
    assert (val_summary.image_count, val_summary.annotation_count,
            val_summary.negative_image_count) == EXPECTED_VAL
    assert (test_summary.image_count, test_summary.annotation_count,
            test_summary.negative_image_count) == EXPECTED_TEST
    assert val_summary.sha256 == EXPECTED_VAL_SHA
    assert test_summary.sha256 == EXPECTED_TEST_SHA
    assert "val" not in D.detailed_scopes_used()
    assert "test" not in D.detailed_scopes_used()
    D.reset_scope_tripwires()

    # every train bounding box satisfies the locked validity rules
    dims = D.image_dimensions(train)
    invalid = 0
    for annotation in train["annotations"]:
        width, height = dims[int(annotation["image_id"])]
        x, y, w, h = (float(v) for v in annotation["bbox"])
        if D.bbox_validity_violations(x, y, w, h, width, height):
            invalid += 1
    assert invalid == 0


# =====================================================================
# X. integration - labeled subsets hashes, counts and nesting
# =====================================================================

def test_x_integration_labeled_hash_counts_and_nesting(protocol):
    train_path = REPO_ROOT / protocol["input_artifacts"]["fixed_split"]["train"]
    train_ids = {int(image["id"])
                 for image in _load_json(train_path)["images"]}

    id_sets = {}
    for budget in ("1pct", "5pct", "10pct", "20pct"):
        path = REPO_ROOT / protocol["input_artifacts"]["labeled_subsets"][budget]
        assert _sha256(_require(path)) == EXPECTED_LABELED_SHA[budget]
        document = _load_json(path)
        ids = {int(image["id"]) for image in document["images"]}
        id_sets[budget] = ids
        assert len(ids) == EXPECTED_LABELED_IMAGES[budget]
        assert len(D.negative_image_ids(document)) == \
            EXPECTED_LABELED_NEGATIVES[budget]
        assert ids.issubset(train_ids)
        assert sorted(int(c["id"]) for c in document["categories"]) == \
            list(range(1, 15))

    assert D.is_nested([id_sets["1pct"], id_sets["5pct"], id_sets["10pct"],
                        id_sets["20pct"]]) is True


# =====================================================================
# Y. integration - the Phase 2F.1 seed evidence confirms no training
# =====================================================================

def test_y_integration_no_training_started(protocol):
    seed_state_path = REPO_ROOT / protocol["input_artifacts"][
        "phase2F1_seed_evidence"]["seed_state_manifest"]
    seed_state = _load_json(seed_state_path)
    assert seed_state["training_started"] is False
    assert seed_state["runs"] == []


# =====================================================================
# Z0. the module loader itself is a guardrail
# =====================================================================

def test_z0_module_loader_registers_module_in_sys_modules():
    """The digit-prefixed script must be importable reliably.

    ``from __future__ import annotations`` turns every dataclass field
    annotation into a string. Resolving those strings goes through
    ``sys.modules[cls.__module__].__dict__``, so the module has to be
    registered under ``spec.name`` BEFORE ``exec_module`` runs.
    """
    assert MODULE_NAME in sys.modules
    assert sys.modules[MODULE_NAME] is D
    assert D.__name__ == MODULE_NAME

    # dataclass introspection must work, including annotation resolution
    for cls in (D.StructuralSummary, D.CheckRecord, D.PreflightResult):
        assert dataclasses.is_dataclass(cls)
        assert dataclasses.fields(cls)
        hints = typing.get_type_hints(cls)
        assert set(hints) >= {f.name for f in dataclasses.fields(cls)}

    assert typing.get_type_hints(D.StructuralSummary)["image_count"] is int
    assert D.StructuralSummary.__dataclass_params__.frozen is True

    # a fresh load under a different name must behave identically
    probe_name = "phase3a_loader_probe"
    sys.modules.pop(probe_name, None)
    try:
        probe = _load_script_module(probe_name)
        assert sys.modules[probe_name] is probe
        assert typing.get_type_hints(probe.StructuralSummary)
        assert probe.RARE_PRIMARY_THRESHOLD == D.RARE_PRIMARY_THRESHOLD
    finally:
        sys.modules.pop(probe_name, None)


# =====================================================================
# Z1. runtime output-path safety gate
# =====================================================================

UNSAFE_OUTPUT_TARGETS = [
    "data/processed/coco/instances_train.json",
    "data/manifests/x.json",
    "configs/protocol/x.yaml",
    "../outside.txt",
    "reports/../data/manifests/x.json",
    "plots/dataset/../../models/x.pth",
    "models/best.pth",
    "checkpoints/last.ckpt",
    "pseudo_labels/round1.json",
    "/etc/passwd",
    "C:\\Windows\\system32\\x.txt",
    "~/x.txt",
    "src/x.csv",
    "",
]

SAFE_OUTPUT_TARGETS = [
    "reports",
    "plots/dataset",
    "reports/03A_class_distribution.csv",
    "reports/dataset_analysis_report.md",
    "reports/03A_dataset_diagnostics_validation.json",
    "reports/03A_artifact_manifest.json",
    "plots/dataset/class_distribution.png",
]


def test_z1_output_path_gate_rejects_unsafe_targets(tmp_path):
    D.reset_validated_output_paths()
    for target in UNSAFE_OUTPUT_TARGETS:
        with pytest.raises(D.OutputPathError):
            D.resolve_safe_output_path(tmp_path, target, label="probe")
    # nothing unsafe may ever be registered as writable
    assert D.validated_output_paths() == ()


def test_z1b_output_path_gate_accepts_declared_targets(tmp_path):
    D.reset_validated_output_paths()
    for target in SAFE_OUTPUT_TARGETS:
        resolved = D.resolve_safe_output_path(tmp_path, target, label="probe")
        assert resolved.is_absolute()
        assert str(resolved).startswith(str(tmp_path.resolve()))
    assert len(D.validated_output_paths()) == len(SAFE_OUTPUT_TARGETS)
    D.reset_validated_output_paths()


def test_z1c_writers_refuse_unregistered_paths(tmp_path):
    """No file may be created before the preflight gate approved its path."""
    D.reset_validated_output_paths()
    csv_target = tmp_path / "reports" / "03A_class_distribution.csv"
    json_target = tmp_path / "reports" / "03A_artifact_manifest.json"

    with pytest.raises(D.OutputPathError):
        D.write_csv(csv_target, ["a"], [{"a": 1}])
    with pytest.raises(D.OutputPathError):
        D.write_json(json_target, {"a": 1})
    assert not csv_target.exists()
    assert not json_target.exists()

    # positive control: once gated, the write succeeds
    D.resolve_safe_output_path(tmp_path, "reports/03A_class_distribution.csv",
                               label="probe")
    D.write_csv(csv_target, ["a"], [{"a": 1}])
    assert csv_target.is_file()
    D.reset_validated_output_paths()


def test_z1d_output_contract_gate_rejects_tampered_protocol(protocol,
                                                            tmp_path):
    import copy

    D.reset_validated_output_paths()
    good, errors = D.resolve_output_contract(protocol, tmp_path)
    assert errors == []
    assert "csv_class_distribution" in good
    assert "validation_json" in good
    assert "artifact_manifest" in good

    for bad_target in ("data/processed/coco/instances_train.json",
                       "data/manifests/x.json",
                       "configs/protocol/x.yaml",
                       "../outside.txt"):
        tampered = copy.deepcopy(protocol)
        tampered["output_contract"]["mandatory_csv"]["class_distribution"] = \
            bad_target
        resolved, rejections = D.resolve_output_contract(tampered, tmp_path)
        assert rejections, "accepted an unsafe target: {0}".format(bad_target)
        assert "csv_class_distribution" not in resolved

    # widening the YAML allowlist must NOT loosen the gate
    tampered = copy.deepcopy(protocol)
    tampered["output_contract"]["writable_directories"] = [
        "reports", "plots/dataset", "data/manifests"]
    tampered["output_contract"]["forbidden_write_paths"] = []
    tampered["output_contract"]["mandatory_csv"]["class_distribution"] = \
        "data/manifests/x.csv"
    resolved, rejections = D.resolve_output_contract(tampered, tmp_path)
    assert rejections
    assert "csv_class_distribution" not in resolved
    D.reset_validated_output_paths()


# =====================================================================
# Z2. forbidden-action flags are derived from evidence
# =====================================================================

TS = "1970-01-01T00:00:00Z"


def test_z2_validation_flags_are_evidence_derived(protocol):
    D.reset_scope_tripwires()

    clean = D.build_validation(
        protocol, "0" * 64, D.PreflightResult(), {"input_hashes": {}},
        "full", TS, False, True, True, [])
    assert clean["checkpoint_created"] is False
    assert clean["pseudo_label_created"] is False
    assert clean["training_artifact_created"] is False
    assert clean["test_content_diagnostics"] is False

    with_checkpoint = D.build_validation(
        protocol, "0" * 64, D.PreflightResult(), {"input_hashes": {}},
        "full", TS, False, True, True, ["checkpoints/model_final.pth"])
    assert with_checkpoint["checkpoint_created"] is True
    assert with_checkpoint["training_artifact_created"] is True
    assert with_checkpoint["dod_candidate"] is False

    with_pseudo = D.build_validation(
        protocol, "0" * 64, D.PreflightResult(), {"input_hashes": {}},
        "full", TS, False, True, True, ["pseudo_labels/round1.json"])
    assert with_pseudo["pseudo_label_created"] is True
    assert with_pseudo["dod_candidate"] is False

    started = D.build_validation(
        protocol, "0" * 64, D.PreflightResult(), {"input_hashes": {}},
        "full", TS, False, True, True, [],
        seed_state={"training_started": True, "runs": []})
    assert started["training_started"] is True
    assert started["dod_candidate"] is False


def test_z2b_derive_flags_from_runtime_tripwires(protocol):
    flags = D.derive_forbidden_action_flags(
        protocol, training_artifacts_created=[],
        detailed_scopes=["train", "labeled_1pct"])
    assert flags["test_content_diagnostics"] is False
    assert flags["unlabeled_hidden_gt_diagnostics"] is False

    leaked = D.derive_forbidden_action_flags(
        protocol, training_artifacts_created=[],
        detailed_scopes=["train", "test"])
    assert leaked["test_content_diagnostics"] is True
    assert leaked["test_design_usage"] is True

    hidden = D.derive_forbidden_action_flags(
        protocol, training_artifacts_created=[],
        detailed_scopes=["train", "unlabeled_5pct"])
    assert hidden["unlabeled_hidden_gt_diagnostics"] is True

    from_paths = D.derive_forbidden_action_flags(
        protocol, training_artifacts_created=[], detailed_scopes=["train"],
        forbidden_unlabeled_paths=["x/instances_unlabeled_5pct.json"])
    assert from_paths["unlabeled_hidden_gt_diagnostics"] is True


def test_z2c_contradictory_validation_is_rejected():
    base = {
        "phase_status": "OPEN_REVIEW_REQUIRED",
        "training_artifacts_detected": [],
        "checkpoint_created": False,
        "pseudo_label_created": False,
        "training_artifact_created": False,
        "detailed_scopes_used": [],
        "test_content_diagnostics": False,
        "unlabeled_hidden_gt_diagnostics": False,
    }
    D.assert_validation_consistency(base)  # the clean record is accepted

    contradictions = [
        {"training_artifacts_detected": ["checkpoints/model_final.pth"],
         "training_artifact_created": True},
        {"training_artifacts_detected": ["pseudo_labels/round1.json"],
         "training_artifact_created": True},
        {"training_artifacts_detected": ["models/notes.txt"]},
        {"detailed_scopes_used": ["train", "test"]},
        {"detailed_scopes_used": ["unlabeled_5pct"],
         "test_content_diagnostics": False},
        {"phase_status": "PASS"},
        {"phase_status": "CLOSED"},
        {"phase_status": "CLOSED_PASS"},
    ]
    for override in contradictions:
        record = dict(base)
        record.update(override)
        with pytest.raises(D.Phase3AError):
            D.assert_validation_consistency(record)


# =====================================================================
# Z3. final output-contract audit (HF35) and structural validity
# =====================================================================

def _fake_output_paths(tmp_path, skip=()):
    reports = tmp_path / "reports"
    plots = tmp_path / "plots" / "dataset"
    reports.mkdir(parents=True, exist_ok=True)
    plots.mkdir(parents=True, exist_ok=True)

    paths = {}
    for key, schema in D.MANDATORY_CSV_SCHEMAS.items():
        target = reports / "03A_{0}.csv".format(key)
        paths["csv_{0}".format(key)] = target
        if "csv_{0}".format(key) not in skip:
            target.write_text(",".join(schema) + "\n", encoding="utf-8")
    for key, name in (("plot_class_distribution", "class_distribution.png"),
                      ("plot_bbox_distribution", "bbox_distribution.png"),
                      ("plot_bbox_location_heatmap",
                       "bbox_location_heatmap.png"),
                      ("plot_negative_image_distribution",
                       "negative_image_distribution.png")):
        target = plots / name
        paths[key] = target
        if key not in skip:
            target.write_bytes(b"PNG")
    for key, name in (("report", "dataset_analysis_report.md"),
                      ("validation_json",
                       "03A_dataset_diagnostics_validation.json"),
                      ("artifact_manifest", "03A_artifact_manifest.json")):
        target = reports / name
        paths[key] = target
        if key not in skip:
            if name.endswith(".json"):
                target.write_text("{}\n", encoding="utf-8")
            else:
                target.write_text("# report\n", encoding="utf-8")
    return paths


def test_z3_output_contract_audit_covers_validation_and_manifest(tmp_path):
    paths = _fake_output_paths(tmp_path)
    audit = D.audit_output_contract(paths)
    assert "validation_json" in audit["mandatory_keys"]
    assert "artifact_manifest" in audit["mandatory_keys"]
    assert audit["all_present"] is True

    for missing_key in ("validation_json", "artifact_manifest",
                        "csv_class_distribution", "report",
                        "plot_bbox_location_heatmap",
                        "plot_negative_image_distribution"):
        partial = _fake_output_paths(tmp_path / missing_key,
                                     skip=(missing_key,))
        partial_audit = D.audit_output_contract(partial)
        assert missing_key in partial_audit["missing"]
        assert partial_audit["all_present"] is False


def test_z3b_machine_readable_audit_is_structural(tmp_path):
    paths = _fake_output_paths(tmp_path)
    audits = D.audit_machine_readable_outputs(paths)
    by_check = {entry["check"]: entry for entry in audits}

    # schemas match, so every schema check passes
    for key in D.MANDATORY_CSV_SCHEMAS:
        assert by_check["csv_schema::{0}".format(key)]["status"] == "PASS"

    # header-only CSVs must fail the structural row-count reconciliation
    assert by_check["csv_row_count::class_distribution"]["status"] == "FAIL"
    assert by_check["csv_row_count::bbox_distribution"]["status"] == "FAIL"
    assert by_check["csv_row_count::class_cooccurrence"]["status"] == "FAIL"
    assert by_check["csv_row_count::labeled_budget_coverage"]["status"] == \
        "FAIL"
    assert by_check["sum_class_bbox_annotation_count"]["status"] == "FAIL"
    assert by_check["sum_primary_size_category_counts"]["status"] == "FAIL"
    assert by_check["json_parses::validation_json"]["status"] == "PASS"
    assert by_check["json_parses::artifact_manifest"]["status"] == "PASS"

    # the expected structural targets are the locked ones
    assert by_check["csv_row_count::class_distribution"]["expected"] == 14
    assert by_check["csv_row_count::bbox_distribution"]["expected"] == 25260
    assert by_check["csv_row_count::class_cooccurrence"]["expected"] == 91
    assert by_check["csv_row_count::labeled_budget_coverage"]["expected"] == 56
    assert by_check["sum_class_bbox_annotation_count"]["expected"] == 25260
    assert by_check["sum_primary_size_category_counts"]["expected"][
        "total"] == 25260

    # a drifted schema is caught
    paths["csv_class_distribution"].write_text("wrong,header\n",
                                               encoding="utf-8")
    drifted = {entry["check"]: entry for entry
               in D.audit_machine_readable_outputs(paths)}
    assert drifted["csv_schema::class_distribution"]["status"] == "FAIL"

    # unparseable JSON is caught
    paths["validation_json"].write_text("{not json", encoding="utf-8")
    broken = {entry["check"]: entry for entry
              in D.audit_machine_readable_outputs(paths)}
    assert broken["json_parses::validation_json"]["status"] == "FAIL"


def test_z3c_label_cardinality_row_count_is_not_asserted():
    """Data-dependent row counts must never become a scientific gate."""
    assert D.EXPECTED_CSV_ROW_COUNTS["label_cardinality"] is None


# =====================================================================
# Z4. YAML <-> Python lock consistency
# =====================================================================

def _set_nested(mapping, dotted_path, value):
    parts = dotted_path.split(".")
    node = mapping
    for part in parts[:-1]:
        node = node[part]
    node[parts[-1]] = value


def test_z4_yaml_matches_python_locks(protocol):
    assert D.yaml_python_lock_mismatches(protocol) == []


def test_z4b_tampered_yaml_lock_is_detected(protocol):
    import copy

    tampering = [
        ("rare_class_definition.primary_threshold", 0.07),
        ("rare_class_definition.denominator", 3000),
        ("rare_class_definition.boundary_max_rare_image_count", 170),
        ("rare_class_definition.boundary_min_nonrare_image_count", 173),
        ("bbox_size_definition.small_threshold", 0.02),
        ("bbox_size_definition.large_threshold", 0.20),
        ("sensitivity_analysis.rare.thresholds", [0.01, 0.05, 0.20]),
        ("sensitivity_analysis.bbox_small.thresholds", [0.005, 0.01, 0.05]),
        ("sensitivity_analysis.bbox_large.thresholds", [0.05, 0.10, 0.30]),
        ("sensitivity_analysis.one_factor_at_a_time", False),
        ("bbox_location.heatmap_bins_x", 25),
        ("bbox_location.heatmap_bins_y", 100),
        ("category_contract.detection_category_count", 15),
        ("category_contract.category_ids_contiguous_to", 15),
        ("expected_counts.train.images", 3000),
        ("expected_counts.train.annotations", 25000),
        ("expected_counts.canonical.no_finding_images", 501),
        ("expected_counts.test.annotations", 5400),
        ("expected_hashes.train", "deadbeef"),
        ("expected_hashes.coco_master_jpg", "deadbeef"),
        ("expected_hashes.labeled.1pct", "deadbeef"),
        ("expected_counts.labeled.20pct.images", 700),
        ("bbox_geometry.continuous_statistics.sd_ddof", 0),
        ("bbox_geometry.continuous_statistics.percentile_method", "midpoint"),
        ("multilabel_diagnostics.label_cardinality.sd_ddof", 0),
        ("multilabel_diagnostics.class_cooccurrence.expected_pair_count", 90),
        ("output_contract.writable_directories", ["reports", "data"]),
    ]
    for dotted_path, value in tampering:
        tampered = copy.deepcopy(protocol)
        _set_nested(tampered, dotted_path, value)
        mismatches = D.yaml_python_lock_mismatches(tampered)
        assert any(entry["field"] == dotted_path for entry in mismatches), \
            "undetected YAML tampering at {0}".format(dotted_path)

    # a dropped CSV schema is also a lock mismatch
    tampered = copy.deepcopy(protocol)
    tampered["output_contract"]["csv_schemas"].pop("class_distribution")
    mismatches = D.yaml_python_lock_mismatches(tampered)
    assert any(entry["field"] == "output_contract.csv_schemas."
                                 "class_distribution"
               for entry in mismatches)

    tampered = copy.deepcopy(protocol)
    tampered["output_contract"]["expected_row_counts"][
        "bbox_distribution"] = 25000
    mismatches = D.yaml_python_lock_mismatches(tampered)
    assert any(entry["field"] == "output_contract.expected_row_counts."
                                 "bbox_distribution"
               for entry in mismatches)


# =====================================================================
# Z5. the runtime hard-fail codes match the declared guardrail contract
# =====================================================================

HARD_FAIL_ADD_PATTERN = re.compile(
    r'(?:result|preflight)\.add\(\s*"([A-Za-z0-9_]+)"')


def _emitted_check_codes(source: str) -> List[str]:
    return HARD_FAIL_ADD_PATTERN.findall(source)


def test_z5_runtime_emits_only_declared_hard_fail_codes(script_source,
                                                        protocol):
    declared = list(protocol["guardrails"]["hard_fail_conditions"])
    assert declared == ["HF{0:02d}".format(index) for index in range(1, 36)]
    assert list(D.DECLARED_HARD_FAIL_CODES) == declared
    assert len(declared) == 35

    emitted = _emitted_check_codes(script_source)
    hard_fail_emitted = sorted({code for code in emitted
                                if code.startswith("HF")})

    # no runtime hard-fail code outside the declared contract
    assert D.undeclared_hard_fail_codes(emitted) == [], (
        "runtime declares hard-fail codes outside HF01..HF35: "
        "{0}".format(D.undeclared_hard_fail_codes(emitted)))

    # the only permitted sub-code is the registered one
    subcodes = [code for code in hard_fail_emitted if code not in declared]
    assert subcodes == ["HF09b"]
    assert D.DECLARED_HARD_FAIL_SUBCODES == {"HF09b": "HF09"}

    # one-to-one: every declared code is actually emitted somewhere
    for code in declared:
        assert code in hard_fail_emitted, \
            "declared hard-fail code never emitted: {0}".format(code)

    # the ad-hoc canonical sub-code introduced in the previous revision
    # must be gone from the runtime entirely
    assert 'add("HF10a"' not in script_source
    assert "HF10a" not in script_source


def test_z5b_undeclared_hard_fail_code_detector():
    assert D.undeclared_hard_fail_codes([]) == []
    assert D.undeclared_hard_fail_codes(["HF01", "HF35", "HF09b"]) == []
    # advisory PF / W codes are not hard-fail codes and are ignored
    assert D.undeclared_hard_fail_codes(["PF01", "PF14", "W01"]) == []
    # ad-hoc sub-codes and out-of-range codes are rejected
    assert D.undeclared_hard_fail_codes(["HF10a"]) == ["HF10a"]
    assert D.undeclared_hard_fail_codes(["HF36"]) == ["HF36"]
    assert D.undeclared_hard_fail_codes(["HF00"]) == ["HF00"]
    assert D.undeclared_hard_fail_codes(["HF10a", "HF10a", "HF36"]) == \
        ["HF10a", "HF36"]


def test_z5c_hf10_covers_canonical_and_fixed_train(script_source):
    """HF10 must be one record covering both zero-GT scopes."""
    # exactly one HF10 record is emitted
    occurrences = [match.start() for match in re.finditer(
        r'(?:result|preflight)\.add\(\s*"HF10"\s*,', script_source)]
    assert len(occurrences) == 1, \
        "HF10 must be emitted exactly once, found {0}".format(len(occurrences))

    body = script_source[occurrences[0]:occurrences[0] + 1200]
    assert "canonical_negative_with_gt" in body
    assert "train_negative_with_gt" in body
    assert "canonical_negative_with_gt_count" in body
    assert "train_negative_with_gt_count" in body
    assert "coco_master_jpg.json + instances_train.json" in body

    # both scopes are computed before the single record is emitted
    canonical_evidence = script_source.index("canonical_negative_with_gt =")
    train_evidence = script_source.index("train_negative_with_gt =")
    assert canonical_evidence < occurrences[0]
    assert train_evidence < occurrences[0]

    # the No Finding policy roll-up must not reference the removed sub-code
    policy_start = script_source.index("no_finding_policy = (")
    policy = script_source[policy_start:policy_start + 200]
    assert 'passed("HF09")' in policy
    assert 'passed("HF09b")' in policy
    assert 'passed("HF10")' in policy
    assert "HF10a" not in policy


def test_z5d_validation_rejects_undeclared_hard_fail_codes(protocol):
    D.reset_scope_tripwires()

    clean = D.build_validation(
        protocol, "0" * 64, D.PreflightResult(), {"input_hashes": {}},
        "full", TS, False, True, True, [])
    assert clean["undeclared_hard_fail_codes"] == []

    polluted = D.PreflightResult()
    polluted.add("HF10a", "ad-hoc sub-code", True, [], [], "probe")
    with pytest.raises(D.Phase3AError):
        D.build_validation(
            protocol, "0" * 64, polluted, {"input_hashes": {}},
            "full", TS, False, True, True, [])

    record = {
        "phase_status": "OPEN_REVIEW_REQUIRED",
        "training_artifacts_detected": [],
        "checkpoint_created": False,
        "pseudo_label_created": False,
        "training_artifact_created": False,
        "detailed_scopes_used": [],
        "test_content_diagnostics": False,
        "unlabeled_hidden_gt_diagnostics": False,
        "undeclared_hard_fail_codes": ["HF36"],
    }
    with pytest.raises(D.Phase3AError):
        D.assert_validation_consistency(record)


# =====================================================================
# Z6. YAML key -> resolver key -> run_full key must never drift
#
# Regression guard for the plot_negative_distribution /
# plot_negative_image_distribution mismatch: the YAML declared
# mandatory_plots.negative_image_distribution, the resolver therefore
# produced "plot_negative_image_distribution", but run_full() and
# audit_output_contract() expected "plot_negative_distribution" and the
# run aborted with "the output-path gate did not resolve these targets".
# =====================================================================

def test_z6_plot_keys_are_consistent_end_to_end(protocol, tmp_path):
    D.reset_validated_output_paths()
    resolved, errors = D.resolve_output_contract(protocol, tmp_path)
    assert errors == []

    # the exact regression that broke the run
    assert "plot_negative_image_distribution" in resolved
    assert "plot_negative_distribution" not in resolved

    # YAML keys -> resolver keys
    yaml_plot_keys = set(protocol["output_contract"]["mandatory_plots"])
    resolver_plot_keys = {key for key in resolved if key.startswith("plot_")}
    assert resolver_plot_keys == {
        "plot_{0}".format(key) for key in yaml_plot_keys}

    # the single source of truth used by run_full and audit_output_contract
    assert set(D.MANDATORY_PLOT_KEYS) == yaml_plot_keys
    assert set(D.MANDATORY_PLOT_TARGET_KEYS) == resolver_plot_keys
    assert list(D.MANDATORY_PLOT_TARGET_KEYS) == [
        "plot_{0}".format(key) for key in D.MANDATORY_PLOT_KEYS]

    # audit_output_contract must demand exactly those plot keys
    audit_plot_keys = {key for key
                       in D.audit_output_contract(resolved)["mandatory_keys"]
                       if key.startswith("plot_")}
    assert audit_plot_keys == resolver_plot_keys

    # the output filenames are unchanged by this fix
    assert resolved["plot_negative_image_distribution"].name == \
        "negative_image_distribution.png"
    assert resolved["plot_class_distribution"].name == \
        "class_distribution.png"
    assert resolved["plot_bbox_distribution"].name == "bbox_distribution.png"
    assert resolved["plot_bbox_location_heatmap"].name == \
        "bbox_location_heatmap.png"
    D.reset_validated_output_paths()


def test_z6b_all_contract_keys_resolve_for_run_full(protocol, tmp_path):
    """Every key run_full() indexes must exist in the resolved contract."""
    D.reset_validated_output_paths()
    resolved, errors = D.resolve_output_contract(protocol, tmp_path)
    assert errors == []

    required_by_run_full = (
        ["reports_dir", "plots_dir", "report", "validation_json",
         "artifact_manifest"]
        + ["csv_{0}".format(key) for key in sorted(D.MANDATORY_CSV_SCHEMAS)]
        + list(D.MANDATORY_PLOT_TARGET_KEYS))
    missing = [key for key in required_by_run_full if key not in resolved]
    assert missing == [], \
        "the output-path gate does not resolve: {0}".format(missing)

    for key in D.audit_output_contract(resolved)["mandatory_keys"]:
        assert key in resolved, \
            "audit demands an unresolved target: {0}".format(key)
    D.reset_validated_output_paths()


def test_z6c_run_full_indexes_no_unresolvable_output_key(script_source,
                                                         protocol, tmp_path):
    """Static guard: every output_paths[...] literal key must be resolvable."""
    D.reset_validated_output_paths()
    resolved, errors = D.resolve_output_contract(protocol, tmp_path)
    assert errors == []

    indexed_keys = set(re.findall(r'output_paths\[\s*"([A-Za-z0-9_]+)"\s*\]',
                                  script_source))
    assert indexed_keys, "no output_paths[...] access found in the script"
    unknown = sorted(key for key in indexed_keys if key not in resolved)
    assert unknown == [], \
        "run_full indexes output keys the resolver never produces: " \
        "{0}".format(unknown)

    # the stale key must be gone from the implementation entirely
    assert 'output_paths["plot_negative_distribution"]' not in script_source
    # ... while the plotting FUNCTION keeps its original name
    assert "def plot_negative_distribution(" in script_source
    assert "negative_image_distribution.png" in \
        protocol["output_contract"]["mandatory_plots"][
            "negative_image_distribution"]
    D.reset_validated_output_paths()


# =====================================================================
# Z7. section 18 must report magnitudes, not post-hoc qualitative labels
#
# Phase 3A defines no operational rule for "severe", "many", "high",
# "unusual" or "strong". The report may therefore state the observed
# magnitude and where it is tabulated, but must not attach an
# undefined qualitative label to an observation as a RESULT claim.
# =====================================================================

UNDEFINED_QUALITATIVE_CLAIMS = [
    "many rare classes",
    "high small-box share",
    "high small-bbox share",
    "unusual aspect-ratio distribution",
    "strong spatial concentration",
    "high label cardinality",
    "strong co-occurrence",
    "severe imbalance",
    "severe class imbalance",
]

REQUIRED_SECTION_18_SENTENCE = (
    "The descriptive statistics reported above characterize potential "
    "pre-training dataset risks. Their magnitude is reported directly "
    "rather than classified using additional post-hoc labels, and none of "
    "these observations constitutes a Phase 3A protocol failure."
)


def _normalise(text: str) -> str:
    return " ".join(text.split())


def _string_constants(source: str, function_name: str = None) -> List[str]:
    """Every string literal the script can emit, whitespace-normalised.

    Adjacent string literals are folded into a single Constant node by the
    parser, so this sees the sentences exactly as they reach the report,
    not as they are wrapped in the source.
    """
    tree = ast.parse(source)
    scope = tree
    if function_name is not None:
        scope = None
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and \
                    node.name == function_name:
                scope = node
                break
        assert scope is not None, \
            "function not found: {0}".format(function_name)
    return [_normalise(node.value) for node in ast.walk(scope)
            if isinstance(node, ast.Constant) and isinstance(node.value, str)]


def test_z7_report_emits_no_undefined_qualitative_claims(script_source):
    """No undefined qualitative label may be emitted as a RESULT claim."""
    emitted = [text.lower() for text in _string_constants(script_source)]
    for phrase in UNDEFINED_QUALITATIVE_CLAIMS:
        offenders = [text for text in emitted if phrase in text]
        assert offenders == [], \
            "the script emits an undefined qualitative claim {0!r}: " \
            "{1}".format(phrase, offenders[:3])


def test_z7b_section_18_uses_the_approved_wording(script_source):
    report_strings = _string_constants(script_source, "build_report")
    assert REQUIRED_SECTION_18_SENTENCE in report_strings, \
        "section 18 does not carry the approved risk-summary wording"

    # the replaced sentence must be gone
    for text in report_strings:
        assert "and do not make Phase 3A fail" not in text
        assert "are scientific findings" not in text

    # section 18 still points at the tabulated magnitudes rather than
    # asserting a qualitative level
    joined = " ".join(report_strings)
    assert "imbalance_ratio_max_over_min" in joined
    assert "reports/03A_labeled_budget_coverage.csv" in joined
    lowered = joined.lower()
    for phrase in UNDEFINED_QUALITATIVE_CLAIMS:
        assert phrase not in lowered


def test_z7c_snake_case_finding_tokens_are_unaffected(protocol):
    """The YAML finding tokens are machine keys, not narrative claims.

    guardrails.scientific_findings_are_not_failures is a machine-readable
    list of snake_case tokens. It stays exactly as declared; only the
    report narrative changed.
    """
    findings = protocol["guardrails"]["scientific_findings_are_not_failures"]
    for token in ("severe_class_imbalance", "many_rare_classes",
                  "high_small_bbox_proportion",
                  "unusual_aspect_ratio_distribution",
                  "strong_spatial_concentration", "high_label_cardinality",
                  "strong_class_cooccurrence",
                  "low_rare_class_support_in_1pct"):
        assert token in findings
    # snake_case tokens must never match the natural-language scan
    joined = " ".join(findings).lower()
    for phrase in UNDEFINED_QUALITATIVE_CLAIMS:
        assert phrase not in joined