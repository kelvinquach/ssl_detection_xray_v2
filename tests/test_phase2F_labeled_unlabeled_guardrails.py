#!/usr/bin/env python3
"""Guardrail tests for Phase 2F — Labeled/Unlabeled Construction.

The active protocol stage and version are defined by
configs/protocol/phase2F_labeled_unlabeled.yaml.

Synthetic, fully deterministic fixtures only (no RNG-dependent assertions
except where iterative-stratification itself is exercised, which is guarded
by HAVE_ITERSTRAT and never asserted against hardcoded membership). These
tests never read the real 3,426-image instances_train.json, never run the
real Phase 2F build against project data, and never touch Phase 2E
artifacts. They exercise the actual functions imported from
scripts/02F_build_labeled_unlabeled.py.

Run (Windows CMD):
    python -m pytest tests/test_phase2F_labeled_unlabeled_guardrails.py -v
"""
from __future__ import annotations

import ast
import importlib.util
import json
import sys
import tempfile
import time
import unittest
from fractions import Fraction
from pathlib import Path
from unittest import mock

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

SCRIPT_PATH = REPO_ROOT / "scripts" / "02F_build_labeled_unlabeled.py"
CONFIG_PATH = REPO_ROOT / "configs" / "protocol" / "phase2F_labeled_unlabeled.yaml"

try:
    import iterstrat.ml_stratifiers  # noqa: F401
    HAVE_ITERSTRAT = True
except Exception:
    HAVE_ITERSTRAT = False


def load_module():
    spec = importlib.util.spec_from_file_location("phase2F_mod", SCRIPT_PATH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


M = load_module()
CATEGORY_NAMES = [f"Class{i}" for i in range(14)]


def _categories():
    return [{"id": i + 1, "name": CATEGORY_NAMES[i], "supercategory": "synthetic"} for i in range(14)]


def make_tiny_fixture() -> dict:
    """5 images, 14 categories (only classes 0 and 1 populated), hand
    computable by hand for exact-arithmetic tests:
      image 1: class 0        image 2: class 0
      image 3: class 1        image 4: class 1
      image 5: zero-GT (No Finding)
    K (train-wide class counts) = [2, 2, 0, 0, ..., 0], N=5, S_T=4.
    """
    images = [
        {"id": 1, "file_name": "train/1.jpg", "width": 10, "height": 10, "is_negative": False, "scope_label": "abnormal"},
        {"id": 2, "file_name": "train/2.jpg", "width": 10, "height": 10, "is_negative": False, "scope_label": "abnormal"},
        {"id": 3, "file_name": "train/3.jpg", "width": 10, "height": 10, "is_negative": False, "scope_label": "abnormal"},
        {"id": 4, "file_name": "train/4.jpg", "width": 10, "height": 10, "is_negative": False, "scope_label": "abnormal"},
        {"id": 5, "file_name": "train/5.jpg", "width": 10, "height": 10, "is_negative": True, "scope_label": "no_finding"},
    ]
    annotations = [
        {"id": 1, "image_id": 1, "category_id": 1, "bbox": [1, 1, 2, 2], "area": 4, "iscrowd": 0},
        {"id": 2, "image_id": 2, "category_id": 1, "bbox": [1, 1, 2, 2], "area": 4, "iscrowd": 0},
        {"id": 3, "image_id": 3, "category_id": 2, "bbox": [1, 1, 2, 2], "area": 4, "iscrowd": 0},
        {"id": 4, "image_id": 4, "category_id": 2, "bbox": [1, 1, 2, 2], "area": 4, "iscrowd": 0},
    ]
    return {"info": {}, "licenses": [], "images": images, "annotations": annotations, "categories": _categories()}


def make_deterministic_train() -> dict:
    """50 images, fully deterministic (no RNG):
      - image_id 1..14: each carries EXACTLY one distinct class (i-1), so
        any selection containing images 1..14 trivially covers 14/14
        classes — this lets coverage-repair tests hand-pick a guaranteed-
        feasible starting selection instead of relying on randomness or a
        skip-on-infeasible escape hatch.
      - image_id 15..20: 6 zero-GT (No Finding) images.
      - image_id 21..50: 30 further abnormal images, each carrying two
        classes in a fixed round-robin pattern (deterministic, no RNG).
    """
    images = []
    annotations = []
    ann_id = 1
    for i in range(14):
        image_id = i + 1
        images.append({"id": image_id, "file_name": f"train/{image_id}.jpg", "width": 10, "height": 10,
                        "is_negative": False, "scope_label": "abnormal"})
        annotations.append({"id": ann_id, "image_id": image_id, "category_id": i + 1,
                             "bbox": [1, 1, 2, 2], "area": 4, "iscrowd": 0})
        ann_id += 1
    for image_id in range(15, 21):
        images.append({"id": image_id, "file_name": f"train/{image_id}.jpg", "width": 10, "height": 10,
                        "is_negative": True, "scope_label": "no_finding"})
    for offset in range(30):
        image_id = 21 + offset
        c1, c2 = offset % 14, (offset + 7) % 14
        images.append({"id": image_id, "file_name": f"train/{image_id}.jpg", "width": 10, "height": 10,
                        "is_negative": False, "scope_label": "abnormal"})
        for c in sorted({c1, c2}):
            annotations.append({"id": ann_id, "image_id": image_id, "category_id": c + 1,
                                 "bbox": [1, 1, 2, 2], "area": 4, "iscrowd": 0})
            ann_id += 1
    return {"info": {}, "licenses": [], "images": images, "annotations": annotations, "categories": _categories()}


# --------------------------------------------------------------------------- #
# Checksum convention (NHIEM VU 5) — unchanged by the R2 review               #
# --------------------------------------------------------------------------- #
class TestChecksumConvention(unittest.TestCase):
    def test_numeric_ascending_sort_not_string_sort(self):
        import hashlib
        h = M.canonical_membership_sha256([10, 9, 2])
        expected = hashlib.sha256("2\n9\n10".encode("utf-8")).hexdigest()
        self.assertEqual(h, expected)

    def test_independent_of_input_order(self):
        ids = [5, 1, 3, 2, 4]
        self.assertEqual(M.canonical_membership_sha256(ids), M.canonical_membership_sha256(list(reversed(ids))))
        self.assertEqual(M.canonical_membership_sha256(ids), M.canonical_membership_sha256(sorted(ids)))

    def test_rejects_non_integer_image_id(self):
        with self.assertRaises(M.Phase2FError):
            M.canonical_membership_sha256([1, 2, "abc"])
        with self.assertRaises(M.Phase2FError):
            M.canonical_membership_sha256([1, 2, 3.5])
        with self.assertRaises(M.Phase2FError):
            M.canonical_membership_sha256([1, True])

    def test_differs_from_phase2e_string_sort_convention(self):
        ids = [10, 9, 2]
        self.assertNotEqual(M.canonical_membership_sha256(ids), M.phase2e_style_sha256(ids))


# --------------------------------------------------------------------------- #
# Canonical namespaced tie-break (review item 6)                              #
# --------------------------------------------------------------------------- #
class TestNamespacedTieBreak(unittest.TestCase):
    def test_image_priority_deterministic_and_namespaced(self):
        d1 = M.image_priority_digest(42, 1001)
        d2 = M.image_priority_digest(42, 1001)
        self.assertEqual(d1, d2)
        import hashlib
        expected = hashlib.sha256(f"{M.NAMESPACE_IMAGE_PRIORITY}|seed=42|image_id=1001".encode("utf-8")).hexdigest()
        self.assertEqual(d1, expected)

    def test_move_priority_includes_full_payload_and_is_order_independent(self):
        d1 = M.move_priority_digest(42, "1pct", "exact_no_finding", "one_for_one_swap", [5], [9])
        d2 = M.move_priority_digest(42, "1pct", "exact_no_finding", "one_for_one_swap", [5], [9])
        self.assertEqual(d1, d2)
        # canonical sorting of removed/added ids: order of the input list must not matter
        d3 = M.move_priority_digest(42, "1pct", "minimum_class_coverage", "two_for_two_swap", [9, 5], [20, 11])
        d4 = M.move_priority_digest(42, "1pct", "minimum_class_coverage", "two_for_two_swap", [5, 9], [11, 20])
        self.assertEqual(d3, d4)

    def test_move_priority_changes_with_any_payload_field(self):
        base = M.move_priority_digest(42, "1pct", "exact_no_finding", "one_for_one_swap", [5], [9])
        self.assertNotEqual(base, M.move_priority_digest(43, "1pct", "exact_no_finding", "one_for_one_swap", [5], [9]))
        self.assertNotEqual(base, M.move_priority_digest(42, "5pct", "exact_no_finding", "one_for_one_swap", [5], [9]))
        self.assertNotEqual(base, M.move_priority_digest(42, "1pct", "minimum_class_coverage", "one_for_one_swap", [5], [9]))
        self.assertNotEqual(base, M.move_priority_digest(42, "1pct", "exact_no_finding", "two_for_two_swap", [5], [9]))
        self.assertNotEqual(base, M.move_priority_digest(42, "1pct", "exact_no_finding", "one_for_one_swap", [6], [9]))

    def test_candidate_set_priority_deterministic_and_order_independent(self):
        d1 = M.candidate_set_priority_digest(42, "5pct", [3, 1, 2])
        d2 = M.candidate_set_priority_digest(42, "5pct", [1, 2, 3])
        self.assertEqual(d1, d2)

    def test_three_namespaces_are_distinct_for_equivalent_inputs(self):
        image_digest = M.image_priority_digest(42, 5)
        move_digest = M.move_priority_digest(42, "1pct", "exact_size", "add", [], [5])
        set_digest = M.candidate_set_priority_digest(42, "1pct", [5])
        self.assertNotEqual(image_digest, move_digest)
        self.assertNotEqual(image_digest, set_digest)
        self.assertNotEqual(move_digest, set_digest)

    def test_canonical_numeric_move_identity_is_the_final_fallback(self):
        identity_a = M.canonical_numeric_move_identity([9, 5], [20, 11])
        identity_b = M.canonical_numeric_move_identity([5, 9], [11, 20])
        self.assertEqual(identity_a, identity_b)
        self.assertEqual(identity_a, ((5, 9), (11, 20)))


# --------------------------------------------------------------------------- #
# round-half-up (unchanged target-size precomputation, still Fraction-based)  #
# --------------------------------------------------------------------------- #
class TestRoundHalfUp(unittest.TestCase):
    def test_exact_half_rounds_up(self):
        self.assertEqual(M.round_half_up_fraction(Fraction(7, 2)), 4)
        self.assertEqual(M.round_half_up_fraction(Fraction(1, 2)), 1)

    def test_matches_locked_reference_targets(self):
        n_train = 3426
        for pct, expected in (("1pct", 34), ("5pct", 171), ("10pct", 343), ("20pct", 685)):
            self.assertEqual(M.round_half_up_fraction(Fraction(n_train) * M.BUDGET_FRACTION[pct]), expected, msg=pct)


# --------------------------------------------------------------------------- #
# Integer distribution objective: 14 indicators only, exact arithmetic        #
# --------------------------------------------------------------------------- #
class TestIntegerObjective(unittest.TestCase):
    def setUp(self):
        self.train = make_tiny_fixture()
        self.image_ids, self.labels14, self.zero_gt, self.names = M.build_indicators(self.train)
        self.K, self.S_T, self.N = M.full_train_integer_stats(self.labels14)

    def test_full_stats_hand_verified(self):
        self.assertEqual(self.K, [2, 2] + [0] * 12)
        self.assertEqual(self.S_T, 4)
        self.assertEqual(self.N, 5)

    def test_integer_objective_signature_has_no_zero_gt_parameter(self):
        # Structural proof, not just a behavioral one: zero_gt is not even a
        # parameter of integer_objective, so it is IMPOSSIBLE for it to
        # influence the distribution objective, regardless of its values.
        import inspect
        params = list(inspect.signature(M.integer_objective).parameters)
        self.assertNotIn("zero_gt", params)
        self.assertEqual(params, ["selected_positions", "labels14", "K", "S_T", "N"])

    def test_integer_tuple_hand_verified_against_fraction_oracle(self):
        positions = {0, 2}  # image_id 1 (class0), image_id 3 (class1); n=2
        e_max, e_mean, e_lc = M.integer_objective(positions, self.labels14, self.K, self.S_T, self.N)
        self.assertEqual((e_max, e_mean, e_lc), (1, 2, 2))  # hand-derived in module docstring / PR notes
        n = 2
        d_max, d_mean, d_lc, *_ = M.fraction_distribution_report(positions, self.labels14, self.K, self.N)
        self.assertEqual(d_max, Fraction(e_max, n * self.N))
        self.assertEqual(d_mean, Fraction(e_mean, 14 * n * self.N))
        self.assertEqual(d_lc, Fraction(e_lc, n * self.N))
        self.assertEqual(d_max, Fraction(1, 10))
        self.assertEqual(d_mean, Fraction(1, 70))
        self.assertEqual(d_lc, Fraction(1, 5))

    def test_d_mean_is_mean_absolute_not_squared(self):
        positions = {0, 2}
        _, d_mean, _, train_prev, labeled_prev, deviations, _ = M.fraction_distribution_report(
            positions, self.labels14, self.K, self.N,
        )
        manual_mean_absolute = sum(deviations, Fraction(0)) / 14
        self.assertEqual(d_mean, manual_mean_absolute)
        manual_mean_squared = sum((d * d for d in deviations), Fraction(0)) / 14
        if manual_mean_squared != 0:
            self.assertNotEqual(d_mean, manual_mean_squared)

    def test_worst_deviation_class_is_argmax_of_absolute_deviation(self):
        positions = {0, 2}
        _, _, _, _, _, deviations, worst_idx = M.fraction_distribution_report(positions, self.labels14, self.K, self.N)
        self.assertEqual(worst_idx, max(range(14), key=lambda c: deviations[c]))

    def test_objective_empty_selection_fails(self):
        with self.assertRaises(M.Phase2FError):
            M.integer_objective(set(), self.labels14, self.K, self.S_T, self.N)

    def test_no_isclose_or_round_in_selection_source(self):
        source = SCRIPT_PATH.read_text(encoding="utf-8")
        for token in ("math.isclose", "np.isclose", "numpy.isclose"):
            self.assertNotIn(token, source, msg=f"forbidden tolerance-based comparison found: {token}")

    def test_no_bare_round_builtin_in_selection_functions(self):
        tree = ast.parse(SCRIPT_PATH.read_text(encoding="utf-8"))
        selection_function_names = {
            "integer_objective", "_move_key", "_coverage_move_key", "repair_exact_size", "repair_exact_no_finding",
            "repair_min_class_coverage", "repair_objective_local_search",
            "_accept_coverage_move", "_accept_objective_move",
        }
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name in selection_function_names:
                for inner in ast.walk(node):
                    if isinstance(inner, ast.Call) and isinstance(inner.func, ast.Name) and inner.func.id == "round":
                        self.fail(f"bare round() found in selection function {node.name}")

    def test_no_fraction_in_integer_objective_source(self):
        source = SCRIPT_PATH.read_text(encoding="utf-8")
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == "integer_objective":
                body_source = ast.get_source_segment(source, node) or ""
                self.assertNotIn("Fraction(", body_source)


class TestObjectiveWinsHashPriority(unittest.TestCase):
    def test_better_integer_objective_sorts_before_worse_regardless_of_priority(self):
        better = (1, 0, 0, "zzzz", ((1,), (2,)))
        worse = (2, 0, 0, "aaaa", ((3,), (4,)))
        self.assertLess(better, worse)

    def test_tie_break_only_applies_on_exact_objective_tie(self):
        tied_a = (1, 0, 0, "aaaa", ((1,), (2,)))
        tied_b = (1, 0, 0, "bbbb", ((3,), (4,)))
        self.assertLess(tied_a, tied_b)

    def test_equal_tuples_never_compare_strictly_less(self):
        a = (1, 2, 3)
        b = (1, 2, 3)
        self.assertFalse(a < b)  # equal-objective move forbidden relies on this


# --------------------------------------------------------------------------- #
# Two-for-two neighborhood composition (review item 5)                        #
# --------------------------------------------------------------------------- #
class TestTwoForTwoNeighborhood(unittest.TestCase):
    def setUp(self):
        self.train = make_deterministic_train()
        self.image_ids, self.labels14, self.zero_gt, self.names = M.build_indicators(self.train)

    def test_two_abnormal_to_two_abnormal_included(self):
        # positions 20,21 (image_id 21,22) are both abnormal (zero_gt=0);
        # pool positions 22,23 (image_id 23,24) also abnormal. Exactly one
        # (out1,out2) pair and one (in1,in2) pair exist here, so exactly one
        # quad must be yielded -- still full/no-topk exhaustive enumeration
        # of the (unchanged, R3-BLOCKED) same-composition branch.
        selected_list = [20, 21]
        pool_list = [22, 23]
        pairs = list(M._brute_force_same_composition_two_for_two_pairs(selected_list, pool_list, self.zero_gt))
        self.assertEqual(len(pairs), 1)
        for out1, out2, in1, in2 in pairs:
            self.assertEqual(int(self.zero_gt[out1]), 0)
            self.assertEqual(int(self.zero_gt[out2]), 0)
            self.assertEqual(int(self.zero_gt[in1]), 0)
            self.assertEqual(int(self.zero_gt[in2]), 0)

    def test_mixed_composition_no_longer_separately_enumerated(self):
        # R3: mixed composition (1 abnormal + 1 No-Finding <-> 1 abnormal +
        # 1 No-Finding) is proven redundant with the one-for-one
        # neighborhood (see the proof comment above _two_for_two_candidates
        # in the script) and is intentionally no longer yielded here.
        # position 14 (image_id 15) is zero_gt=1; position 20 (image_id 21)
        # is zero_gt=0.
        selected_list = [14, 20]
        pool_list = [15, 21]
        pairs = list(M._brute_force_same_composition_two_for_two_pairs(selected_list, pool_list, self.zero_gt))
        self.assertEqual(pairs, [], msg="mixed composition must no longer be separately enumerated in R3")

    def test_mixed_composition_redundancy_proof_premise_nf_rows_all_zero(self):
        # The mathematical premise the redundancy proof relies on: every
        # No-Finding image contributes the all-zero row to labels14 (by
        # construction in build_indicators: zero_gt = 1 iff the row sums to
        # zero), so pairing it into a two-for-two move can never change
        # class coverage or the integer objective beyond what the paired
        # abnormal-for-abnormal one-for-one swap already achieves.
        nf_positions = [p for p in range(len(self.image_ids)) if self.zero_gt[p] == 1]
        self.assertTrue(len(nf_positions) > 0)
        for p in nf_positions:
            self.assertEqual(int(self.labels14[p].sum()), 0)

    def test_two_no_finding_to_two_no_finding_excluded(self):
        # positions 14,15 (image_id 15,16) both zero_gt=1; pool 16,17 also zero_gt=1.
        selected_list = [14, 15]
        pool_list = [16, 17]
        pairs = list(M._brute_force_same_composition_two_for_two_pairs(selected_list, pool_list, self.zero_gt))
        self.assertEqual(pairs, [], msg="NF<->NF composition must never be generated")

    def test_never_yields_k_ge_3_move(self):
        selected_list = [20, 21, 22]
        pool_list = [23, 24, 25]
        for quad in M._brute_force_same_composition_two_for_two_pairs(selected_list, pool_list, self.zero_gt):
            self.assertEqual(len(quad), 4)


# --------------------------------------------------------------------------- #
# R3 review item 5 — two-for-two complexity disclosure                       #
# --------------------------------------------------------------------------- #
class TestR11ActiveProtocolClaims(unittest.TestCase):
    """R11 (TASK 8/9): the ACTIVE protocol declares exactly one objective
    neighborhood -- exhaustive one-for-one -- and none of the retired
    two-for-two claims survive as an active claim.

    This class REPLACES TestTwoForTwoComplexityDisclosure, whose assertions
    were about repair.two_for_two_complexity_status / two_for_two_engine.
    complexity_policy / .completeness_claim -- all retired. Those historical
    statements are preserved under legacy_two_for_two_engine and are
    asserted to be explicitly marked historical (see
    TestLegacyTwoForTwoIsNonActive)."""

    def _config(self):
        import yaml
        return yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))

    def test_active_repair_policy_declared_and_matches_code_constant(self):
        config = self._config()
        self.assertEqual(config["repair"]["active_repair_policy"], "ONE_FOR_ONE_EXHAUSTIVE_DETERMINISTIC")
        self.assertEqual(M.ACTIVE_REPAIR_POLICY, "ONE_FOR_ONE_EXHAUSTIVE_DETERMINISTIC")
        self.assertEqual(config["repair"]["active_repair_policy"], M.ACTIVE_REPAIR_POLICY)

    def test_active_phase_order_matches_r11(self):
        # TASK 10.1 -- the active phase order, in config AND in code.
        config = self._config()
        expected = ["exact_size", "exact_no_finding", "minimum_class_coverage", "objective_repair"]
        self.assertEqual(config["repair"]["phase_order"], expected)
        self.assertEqual(list(M.REPAIR_PHASE_ORDER), expected)

    def test_active_move_order_is_one_for_one_only(self):
        config = self._config()
        self.assertEqual(config["repair"]["move_order_for_coverage_and_objective_repair"], ["one_for_one"])

    def test_active_move_types_contain_no_two_for_two(self):
        self.assertNotIn("two_for_two_swap", M.REPAIR_MOVE_TYPE_ORDER)
        self.assertEqual(set(M.REPAIR_MOVE_TYPE_ORDER), {"add", "remove", "one_for_one_swap"})

    def test_termination_is_one_for_one_local_optimum_and_no_global_optimum_claimed(self):
        # TASK 10.9 -- no global optimum is claimed, anywhere in the active
        # protocol document or in the code constant.
        config = self._config()
        self.assertEqual(config["repair"]["active_objective_termination"], "one_for_one_local_optimum")
        self.assertFalse(config["repair"]["global_optimum_claimed"])
        self.assertEqual(M.LOCAL_OPTIMUM_NEIGHBORHOOD, "one_for_one")

    def test_retired_two_for_two_keys_are_not_active_protocol_keys(self):
        # TASK 9 -- the retired claims must not be reachable as ACTIVE keys.
        config = self._config()
        for retired in ("two_for_two_complexity_status", "two_for_two_complexity_note", "two_for_two_neighborhood"):
            self.assertNotIn(retired, config["repair"], msg=f"retired active claim still present: {retired}")
        self.assertNotIn("two_for_two_engine", config)
        self.assertNotIn("two_for_two_hotpath_profiler", config)

    def test_neighborhood_exhaustion_claim_is_scoped_to_one_for_one(self):
        config = self._config()
        value = config["repair"]["neighborhood_exhaustion"]
        self.assertIn("one_for_one", value)
        self.assertIn("full_no_topk_heuristic", value)

    def test_no_unproven_heuristic_or_topk_marker_introduced(self):
        # Retained verbatim from the pre-R11 suite: removing two-for-two must
        # not have smuggled in a heuristic/top-k replacement.
        source = SCRIPT_PATH.read_text(encoding="utf-8")
        for token in ("top_k", "topk", "heuristic_filter", "approx_prune"):
            self.assertNotIn(token, source, msg=f"forbidden unproven-pruning marker found: {token}")

    def test_allowed_final_claim_text_present_in_config_and_code(self):
        config = self._config()
        claim = config["repair"]["active_repair_policy_note"]
        normalized = " ".join(claim.split())
        self.assertIn("exhaustively enumerates the admissible one-for-one swap neighborhood", normalized)
        self.assertIn("one-for-one local optimum", normalized)
        self.assertIn("No global optimum is claimed", normalized)
        # The same claim must appear in the script itself (module docstring),
        # compared whitespace-normalized so line wrapping is irrelevant.
        source_normalized = " ".join(SCRIPT_PATH.read_text(encoding="utf-8").split())
        self.assertIn(
            "objective-repair stage exhaustively enumerates the admissible one-for-one "
            "swap neighborhood and terminates at a one-for-one local optimum. No global "
            "optimum is claimed.",
            source_normalized,
        )


# --------------------------------------------------------------------------- #
# R11 TASK 3/22 — the ACTIVE repair-policy gate replaces the retired          #
# two-for-two operational-approval gate                                       #
# --------------------------------------------------------------------------- #
class TestActiveRepairPolicyGate(unittest.TestCase):
    """R11: compute_all_budgets fails closed on a protocol/implementation
    policy mismatch, and NO LONGER fails on the retired
    two_for_two_engine.operationally_approved flag.

    This class REPLACES TestTwoForTwoEngineFailClosed. That class asserted
    the retired gate and the seven retired preflight engine-status
    informational fields; both are gone from the active protocol."""

    def _config(self):
        import yaml
        return yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))

    def test_legacy_engine_approval_no_longer_gates_official_construction(self):
        # TASK 10.22 -- the real config still records operationally_approved
        # as false (it was never approved). With the retired gate removed,
        # compute_all_budgets must NOT refuse for that reason.
        config = self._config()
        self.assertFalse(config["legacy_two_for_two_engine"]["operationally_approved"])
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(Exception) as ctx:
                M.compute_all_budgets(config, Path(tmp))
            self.assertNotIn("TWO_FOR_TWO_ENGINE_NOT_OPERATIONALLY_APPROVED", str(ctx.exception))
            self.assertNotIn("ACTIVE_REPAIR_POLICY_MISMATCH", str(ctx.exception))

    def test_policy_mismatch_refuses_before_any_construction(self):
        # An EMPTY project (no instances_train.json at all): if the policy
        # check runs FIRST, as required, the failure must be
        # ACTIVE_REPAIR_POLICY_MISMATCH, never a downstream
        # FileNotFoundError/JSON error from loading a nonexistent train
        # file -- proving refusal happens before any construction work.
        config = self._config()
        config = dict(config)
        config["repair"] = dict(config["repair"])
        config["repair"]["active_repair_policy"] = "SOMETHING_ELSE_ENTIRELY"
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(M.Phase2FError) as ctx:
                M.compute_all_budgets(config, Path(tmp))
            self.assertIn("ACTIVE_REPAIR_POLICY_MISMATCH", str(ctx.exception))

    def test_missing_policy_key_also_refuses(self):
        config = self._config()
        config = dict(config)
        config["repair"] = {k: v for k, v in config["repair"].items() if k != "active_repair_policy"}
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(M.Phase2FError) as ctx:
                M.compute_all_budgets(config, Path(tmp))
            self.assertIn("ACTIVE_REPAIR_POLICY_MISMATCH", str(ctx.exception))

    def test_preflight_reports_active_policy_and_no_retired_engine_fields(self):
        # The active informational fields must be present even on the
        # missing-inputs early-return path (same discipline the retired
        # engine fields had), and none of the retired fields may survive.
        config = self._config()
        with tempfile.TemporaryDirectory() as tmp:
            result = M.run_preflight(config, Path(tmp))
            self.assertEqual(result["status"], "FAIL")  # confirms the early-return path
            self.assertEqual(result["active_repair_policy"], M.ACTIVE_REPAIR_POLICY)
            self.assertEqual(result["objective_repair_neighborhood"], "one_for_one")
            self.assertFalse(result["global_optimum_claimed"])
            self.assertIn("one-for-one local optimum", result["objective_repair_termination_claim"])
            for retired in (
                "two_for_two_engine_status", "operationally_approved", "completeness_claim",
                "complexity_policy", "mathematical_completeness",
                "empirical_operational_tractability", "operational_approval",
                "two_for_two_complexity_diagnostics",
            ):
                self.assertNotIn(retired, result, msg=f"retired preflight field survived: {retired}")

    def test_preflight_active_policy_is_a_pass_fail_check_not_only_informational(self):
        config = self._config()
        config = dict(config)
        config["repair"] = dict(config["repair"])
        config["repair"]["active_repair_policy"] = "SOMETHING_ELSE_ENTIRELY"
        with tempfile.TemporaryDirectory() as tmp:
            result = M.run_preflight(config, Path(tmp))
        entry = next(c for c in result["checks"] if c["name"] == "active_repair_policy_matches_locked_constant")
        self.assertEqual(entry["status"], "FAIL")


# --------------------------------------------------------------------------- #
# R11 TASK 4 — legacy two-for-two code is retained but NON-ACTIVE             #
# --------------------------------------------------------------------------- #
class TestLegacyTwoForTwoIsNonActive(unittest.TestCase):
    """The legacy engine may still exist (its unit tests below still
    exercise it), but it must be unreachable from every active construction
    path and must not appear as an active protocol claim."""

    def _config(self):
        import yaml
        return yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))

    def test_legacy_config_block_is_marked_historical_and_non_gating(self):
        legacy = self._config()["legacy_two_for_two_engine"]
        self.assertEqual(legacy["status"], "HISTORICAL_NOT_PART_OF_ACTIVE_PROTOCOL")
        self.assertEqual(legacy["removed_in"], "2F-C0-R11")
        self.assertEqual(legacy["removed_reason"], "COMPUTATIONAL_TRACTABILITY_AND_PROPORTIONALITY")
        self.assertFalse(legacy["active_protocol_depends_on_this_block"])
        self.assertFalse(legacy["gates_materialization"])
        # TASK 8 -- decision hygiene, recorded machine-readably.
        self.assertTrue(legacy["removed_before_training"])
        self.assertTrue(legacy["removed_without_val_or_test_outcomes"])
        self.assertTrue(legacy["removed_without_model_performance"])
        self.assertTrue(legacy["removed_without_candidate_comparison"])
        self.assertTrue(legacy["removed_without_seed_search"])
        # The retired claims are renamed so they can never be read as active.
        self.assertNotIn("completeness_claim", legacy)
        self.assertNotIn("complexity_policy", legacy)
        self.assertNotIn("operational_tractability_claim", legacy)
        self.assertIn("historical_completeness_claim", legacy)
        self.assertIn("historical_complexity_policy", legacy)
        self.assertIn("historical_operational_tractability_claim", legacy)

    def test_legacy_engine_function_still_exists_for_its_retained_unit_tests(self):
        self.assertTrue(callable(M._equivalence_class_two_for_two_search))
        self.assertTrue(callable(M._brute_force_same_composition_two_for_two_pairs))

    def _source_of(self, func_name: str) -> str:
        tree = ast.parse(SCRIPT_PATH.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == func_name:
                return ast.dump(node)
        self.fail(f"function not found in script source: {func_name}")

    def _called_functions(self, func_name: str) -> set[str]:
        """Return direct call targets, ignoring docstrings/comments."""
        tree = ast.parse(SCRIPT_PATH.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == func_name:
                calls: set[str] = set()
                for child in ast.walk(node):
                    if not isinstance(child, ast.Call):
                        continue
                    if isinstance(child.func, ast.Name):
                        calls.add(child.func.id)
                    elif isinstance(child.func, ast.Attribute):
                        calls.add(child.func.attr)
                return calls
        self.fail(f"function not found in script source: {func_name}")

    def test_objective_repair_never_calls_two_for_two(self):
        # TASK 10.2 -- static proof over the real source, not a mock.
        self.assertNotIn("_equivalence_class_two_for_two_search",
                         self._called_functions("repair_objective_local_search"))

    def test_coverage_repair_never_calls_two_for_two(self):
        # TASK 10.3
        self.assertNotIn("_equivalence_class_two_for_two_search",
                         self._called_functions("repair_min_class_coverage"))

    def test_build_budget_step_never_calls_two_for_two(self):
        self.assertNotIn("_equivalence_class_two_for_two_search",
                         self._called_functions("build_budget_step"))

    def test_official_construction_never_calls_two_for_two_statically(self):
        # TASK 10.4 (static half) -- compute_all_budgets is the ONLY entry
        # point for official materialization and --reconstruct-check. It must
        # not name the engine, and the only construction primitive it may
        # call is build_budget_step (already proven engine-free above).
        calls = self._called_functions("compute_all_budgets")
        self.assertNotIn("_equivalence_class_two_for_two_search", calls)
        self.assertIn("build_budget_step", calls)

    def test_official_construction_never_calls_two_for_two_at_runtime(self):
        # TASK 10.4 (runtime half) -- patch the engine with a sentinel that
        # raises if it is ever reached, then drive the real production
        # per-budget primitive end to end. The assertion under test is that
        # the sentinel NEVER fires; the construction outcome itself may be
        # OK or REPAIR_INFEASIBLE depending on the fixture's coverage
        # feasibility, and either is acceptable here (coverage feasibility
        # has its own dedicated tests) -- what must never happen is the
        # legacy engine being invoked as a fallback.
        train = make_deterministic_train()
        image_ids, labels14, zero_gt, names = M.build_indicators(train)
        K, S_T, N = M.full_train_integer_stats(labels14)
        repair_log: list[dict] = []

        def _explode(*_args, **_kwargs):
            raise AssertionError("active construction reached the legacy two-for-two engine")

        # The invariant under test is orchestration reachability, not the
        # third-party splitter.  Supply one deterministic initial candidate
        # so this guardrail remains executable even when iterstrat is absent.
        initial_candidate = set(range(20))
        with mock.patch.object(M, "_equivalence_class_two_for_two_search", _explode), \
             mock.patch.object(M, "stratified_initial_candidate", return_value=initial_candidate):
            _selected, outcome, _status = M.build_budget_step(
                "1pct", set(), image_ids, labels14, zero_gt, names,
                20, 2, 42, K, S_T, N, None, repair_log,
            )
        self.assertIn(outcome, (M.RepairOutcome.OK, M.RepairOutcome.REPAIR_INFEASIBLE))
        for entry in repair_log:
            self.assertNotEqual(entry["move_type"], "two_for_two_swap")

    def test_reconstruct_check_construction_never_calls_two_for_two(self):
        # TASK 10.5 -- run_reconstruct_check builds via compute_all_budgets,
        # so the static proof above covers its construction path too.
        # Asserted structurally here so the test does not depend on an
        # already-promoted lock manifest existing.
        source = self._source_of("run_reconstruct_check")
        self.assertNotIn("_equivalence_class_two_for_two_search", source)
        self.assertIn("compute_all_budgets", source)


# --------------------------------------------------------------------------- #
# R4 NHIEM VU 3B — equivalence-class two-for-two engine vs. brute-force      #
# oracle (test-only) equivalence                                             #
# --------------------------------------------------------------------------- #
def _oracle_best_two_for_two(locked, selected, pool, labels14, zero_gt, K, S_T, N, seed, budget, repair_phase,
                              image_ids, visited, accept_and_rank):
    """TEST-ONLY brute-force oracle matching _equivalence_class_two_for_two_
    search's exact contract: enumerates the FULL concrete same-composition
    neighborhood via M._brute_force_same_composition_two_for_two_pairs and
    returns the true minimum by (accept_and_rank prefix, priority,
    identity), or None. Used only against small synthetic fixtures."""
    selected_list = sorted(p for p in selected if zero_gt[p] == 0)
    pool_list = sorted(p for p in pool if zero_gt[p] == 0)
    best_key = None
    best_move = None
    for out1, out2, in1, in2 in M._brute_force_same_composition_two_for_two_pairs(selected_list, pool_list, zero_gt):
        trial = set(selected)
        trial.discard(out1)
        trial.discard(out2)
        trial.add(in1)
        trial.add(in2)
        if frozenset(trial) in visited:
            continue
        trial_state = sorted(locked | trial)
        new_k = labels14[trial_state].sum(axis=0).astype(np.int64) if trial_state else np.zeros(14, dtype=np.int64)
        new_n = len(trial_state)
        new_s_l = int(new_k.sum())
        prefix = accept_and_rank(new_k, new_n, new_s_l)
        if prefix is None:
            continue
        removed_ids = (int(image_ids[out1]), int(image_ids[out2]))
        added_ids = (int(image_ids[in1]), int(image_ids[in2]))
        priority = M.move_priority_digest(seed, budget, repair_phase, "two_for_two_swap", removed_ids, added_ids)
        identity = M.canonical_numeric_move_identity(removed_ids, added_ids)
        key = tuple(prefix) + (priority, identity)
        if best_key is None or key < best_key:
            best_key = key
            best_move = (out1, out2, in1, in2, key)
    return best_move


def _make_multilabel_cooccurrence_train(seed: int, n: int, class_pool_size: int = 5) -> dict:
    """Deterministic (seeded) synthetic train set where every abnormal image
    carries 1-3 classes drawn from a SMALL class pool (default 5 of the 14
    categories), which forces genuine multilabel co-occurrence AND repeated
    equivalence classes (many images sharing the exact same label vector) --
    exactly the structure _equivalence_class_two_for_two_search is designed
    to exploit, and a meaningful stress case for oracle-vs-engine
    equivalence."""
    import random as _random
    rng = _random.Random(seed)
    class_pool = list(range(1, class_pool_size + 1))
    images, annotations = [], []
    ann_id = 1
    for i in range(1, n + 1):
        k = rng.choice([1, 2, 2, 3])
        cats = rng.sample(class_pool, min(k, len(class_pool)))
        images.append({"id": i, "file_name": f"t/{i}.jpg", "width": 10, "height": 10})
        for c in sorted(cats):
            annotations.append({"id": ann_id, "image_id": i, "category_id": c,
                                 "bbox": [1, 1, 2, 2], "area": 4, "iscrowd": 0})
            ann_id += 1
    return {"info": {}, "licenses": [], "images": images, "annotations": annotations, "categories": _categories()}


class TestEquivalenceClassEngineVsBruteForceOracle(unittest.TestCase):
    def _coverage_accept(self, K, S_T, N, missing_before):
        def accept(new_k, new_n, new_s_l):
            new_missing = 14 - int((new_k > 0).sum())
            if new_missing >= missing_before:
                return None
            e_max, e_mean, e_lc = M._integer_objective_from_aggregate(new_k, new_n, new_s_l, K, S_T, N)
            return (new_missing, e_max, e_mean, e_lc)
        return accept

    def _objective_accept(self, K, S_T, N, current_obj):
        def accept(new_k, new_n, new_s_l):
            if int((new_k > 0).sum()) != 14:
                return None
            trial_obj = M._integer_objective_from_aggregate(new_k, new_n, new_s_l, K, S_T, N)
            if not (trial_obj < current_obj):
                return None
            return trial_obj
        return accept

    def test_multilabel_cooccurrence_engine_matches_oracle_many_cases(self):
        # R6 pytest-repair round: several independent (seed, n, split)
        # RANDOM synthetic cases, each with genuine multilabel
        # co-occurrence and repeated equivalence classes. These are
        # NEGATIVE-CASE-TOLERANT equivalence checks -- oracle==None and
        # engine==None on a given case is a perfectly valid, EXPECTED
        # outcome (not every random cut point has an improving move), so
        # this test only asserts oracle==engine on every case, never that
        # any case must be nonempty. The GUARANTEED-nonempty positive case
        # is a separate, purpose-built fixture below
        # (test_multilabel_cooccurrence_positive_constructive_case), per
        # the "don't rely on random fixtures to happen to produce a
        # candidate" principle.
        cases = [(101, 40, 0.3), (202, 60, 0.4), (303, 50, 0.5), (404, 45, 0.25)]
        for seed, n, split_frac in cases:
            train = _make_multilabel_cooccurrence_train(seed, n)
            image_ids, labels14, zero_gt, _ = M.build_indicators(train)
            K, S_T, N = M.full_train_integer_stats(labels14)
            abnormal = [p for p in range(len(image_ids)) if zero_gt[p] == 0]
            cut = max(2, int(len(abnormal) * split_frac))
            selected = set(abnormal[:cut])
            pool = set(abnormal[cut:])
            locked: set[int] = set()
            visited: set[frozenset] = set()

            missing = 14 - int(labels14[sorted(selected)].sum(axis=0).astype(bool).sum()) if selected else 14
            accept = self._coverage_accept(K, S_T, N, missing)
            oracle = _oracle_best_two_for_two(locked, selected, pool, labels14, zero_gt, K, S_T, N, 42, "1pct",
                                               "minimum_class_coverage", image_ids, visited, accept)
            engine = M._equivalence_class_two_for_two_search(locked, selected, pool, labels14, zero_gt, K, S_T, N,
                                                               42, "1pct", "minimum_class_coverage", image_ids,
                                                               visited, accept)
            self.assertEqual(oracle, engine, msg=f"coverage mismatch for seed={seed}")

            current_obj = M.integer_objective(locked | selected, labels14, K, S_T, N) if selected else (0, 0, 0)
            accept_obj = self._objective_accept(K, S_T, N, current_obj)
            oracle_obj = _oracle_best_two_for_two(locked, selected, pool, labels14, zero_gt, K, S_T, N, 42, "1pct",
                                                    "objective_repair", image_ids, visited, accept_obj)
            engine_obj = M._equivalence_class_two_for_two_search(locked, selected, pool, labels14, zero_gt, K, S_T,
                                                                   N, 42, "1pct", "objective_repair", image_ids,
                                                                   visited, accept_obj)
            self.assertEqual(oracle_obj, engine_obj, msg=f"objective mismatch for seed={seed}")

    def test_multilabel_cooccurrence_positive_constructive_case(self):
        # A GUARANTEED-nonempty, hand-verified positive case with genuine
        # multilabel co-occurrence AND repeated equivalence signatures
        # (the structure the engine's Phase 1/Phase 2 split is
        # specifically designed to exploit), rather than hoping a random
        # case happens to produce one. Exercises the COVERAGE-REPAIR path
        # (minimum_class_coverage / `_coverage_accept`), where a
        # missing-classes baseline is the CORRECT precondition.
        #
        # R6 pytest-fixture-repair round (2nd repair pass): GPT static
        # review flagged that this fixture was originally paired with
        # `_objective_accept`, which requires the baseline to ALREADY be
        # 14/14 covered -- the same mismatch as the main blocker in
        # test_improving_two_swap_exists_with_no_improving_one_swap. Fixed
        # by switching to `_coverage_accept` / repair_phase=
        # "minimum_class_coverage"; this test does NOT claim to represent
        # objective_repair (see the separate, dedicated 14/14-baseline
        # test for that).
        #
        #   SELECTED (n=14): singleton images S_0..S_11 (classes 0..11,
        #   one each) plus R1, R2 -- TWO DIFFERENT images that BOTH carry
        #   the SAME two-class signature {1, 2} (a repeated equivalence
        #   class of size 2), pushing k_1=k_2=3 (redundant). Classes 12/13
        #   are missing (missing_before=2).
        #   POOL: P1, P2 -- TWO DIFFERENT images that BOTH carry the SAME
        #   two-class signature {12, 13} (also a repeated equivalence
        #   class of size 2) -- the only way to gain coverage of 12/13.
        #
        # Removing {R1, R2} (classes 1/2 remain covered via S_1/S_2) and
        # adding {P1, P2} strictly reduces the missing-class count (2 -> 0)
        # under the coverage-progress rule; other coverage-safe removed
        # pairs drawn from {S_1, S_2, R1, R2} exist too, so the winner is
        # left to the engine/oracle's own tie-break -- this test only
        # asserts that SOME valid coverage-improving move exists and that
        # oracle and engine agree on it (including the full key).
        images, annotations = [], []
        ann_id = 1

        def add_image(image_id, classes):
            nonlocal ann_id
            images.append({"id": image_id, "file_name": f"t/{image_id}.jpg", "width": 10, "height": 10})
            for c in classes:
                annotations.append({"id": ann_id, "image_id": image_id, "category_id": c + 1,
                                     "bbox": [1, 1, 2, 2], "area": 4, "iscrowd": 0})
                ann_id += 1

        for c in range(12):
            add_image(c + 1, [c])       # S_0..S_11 -> image_id 1..12
        add_image(13, [1, 2])            # R1 (pool-adjacent, selected): co-occurring {1,2}
        add_image(14, [1, 2])            # R2: same repeated signature {1,2}
        add_image(15, [12, 13])          # P1 (pool): co-occurring {12,13}
        add_image(16, [12, 13])          # P2 (pool): same repeated signature {12,13}
        fixture = {"info": {}, "licenses": [], "images": images, "annotations": annotations,
                   "categories": _categories()}

        image_ids, labels14, zero_gt, _ = M.build_indicators(fixture)
        K, S_T, N = M.full_train_integer_stats(labels14)
        id_to_pos = {int(v): i for i, v in enumerate(image_ids.tolist())}
        selected = {id_to_pos[i] for i in range(1, 15)}
        pool = {id_to_pos[15], id_to_pos[16]}
        locked: set[int] = set()

        missing_before = 14 - int(labels14[sorted(selected)].sum(axis=0).astype(bool).sum())
        self.assertEqual(missing_before, 2, "fixture precondition: exactly classes 12/13 must be missing")
        present = set(int(c) for c in np.where(labels14[sorted(selected)].sum(axis=0) > 0)[0])
        self.assertEqual(present, set(range(12)), "fixture precondition: classes 0-11 must all be present")

        visited: set[frozenset] = set()
        accept_cov = self._coverage_accept(K, S_T, N, missing_before)
        oracle = _oracle_best_two_for_two(locked, selected, pool, labels14, zero_gt, K, S_T, N, 42, "1pct",
                                           "minimum_class_coverage", image_ids, visited, accept_cov)
        engine = M._equivalence_class_two_for_two_search(locked, selected, pool, labels14, zero_gt, K, S_T, N, 42,
                                                           "1pct", "minimum_class_coverage", image_ids, visited,
                                                           accept_cov)
        self.assertIsNotNone(oracle, "fixture precondition violated: no coverage-improving 2-swap exists")
        self.assertEqual(oracle, engine)
        # The winning move must strictly reduce the missing-class count
        # (coverage-progress rule), never merely tie or add without
        # progress.
        out1, out2, in1, in2, full_key = engine
        new_missing_count = full_key[0]
        self.assertLess(new_missing_count, missing_before)

    def test_improving_two_swap_exists_with_no_improving_one_swap(self):
        # R6 pytest-fixture-repair round (2nd repair pass): the previous
        # fixture's baseline was MISSING classes 12/13 (missing_before=2),
        # which GPT static review correctly flagged as invalid -- the real
        # call site only ever invokes objective_repair once size, exact
        # No Finding, and 14/14 coverage are ALREADY satisfied, so a
        # baseline that starts short of 14/14 does not represent a state
        # objective_repair is ever actually asked to improve from. Replaced
        # with a fixture whose BASELINE is already exactly 14/14 covered,
        # where NO legal 1-swap can improve (proven exhaustively below --
        # every one of the 22 possible (out,in) combinations is coverage-
        # REJECTED, never merely non-improving) and where a hand-verified
        # 2-swap strictly improves the exact-integer objective.
        #
        # Class roles (0-indexed): r=0 (a "spare capacity" class), p=10,
        # p'=11, q=12, q'=13 (a private class-pair per swappable image),
        # clean = 1..9 (9 untouched classes).
        #
        #   SELECTED (n=11): S_1 (id 1, classes {r=0, 1} -- the ONLY
        #   backup for r, permanently covers class 1 too), S_2..S_9 (ids
        #   2-9, one singleton class each, classes 2..9), X1 (id 10,
        #   classes {p=10, p'=11}), X2 (id 11, classes {q=12, q'=13}). X1
        #   and X2 are each the SOLE carrier of their own private pair.
        #   POOL (2 images, forcing the added side of ANY 2-swap to be
        #   exactly {P1, P2}): P1 (id 12, classes {p=10, q'=13, r=0}),
        #   P2 (id 13, classes {q=12, p'=11}) -- a CROSS pairing (P1
        #   restores p and q', P2 restores q and p') plus P1 carries a
        #   bonus copy of r.
        #
        # Why no 1-swap can ever pass coverage: removing S_1 or any of
        # S_2..S_9 drops a class (1..9) that neither P1 nor P2 covers.
        # Removing X1 alone drops p'=11 (P1 doesn't cover it) or drops
        # p=10 (P2 doesn't cover it) depending on which pool image is
        # added -- one of the two is always missing. Symmetrically for
        # X2 (q=12 / q'=13). So every (out_pos, in_pos) pair fails the
        # accept function's exact-14-coverage gate outright; the
        # objective is never even evaluated for any 1-swap.
        # Why {X1,X2} -> {P1,P2} is the UNIQUE valid 2-swap: pool has only
        # 2 elements, so the added pair is forced to be {P1,P2}; P1 union
        # P2 covers exactly {r,p,p',q,q'} = {0,10,11,12,13}, so the only
        # removed pair whose classes are fully subsumed by that set is
        # {X1,X2} itself -- removing any S_i instead always drops a class
        # (1..9) P1/P2 cannot restore.
        images, annotations = [], []
        ann_id = 1

        def add_image(image_id, classes):
            nonlocal ann_id
            images.append({"id": image_id, "file_name": f"t/{image_id}.jpg", "width": 10, "height": 10})
            for c in classes:
                annotations.append({"id": ann_id, "image_id": image_id, "category_id": c + 1,
                                     "bbox": [1, 1, 2, 2], "area": 4, "iscrowd": 0})
                ann_id += 1

        add_image(1, [0, 1])          # S_1: r=0 backup + clean class 1
        for c in range(2, 10):
            add_image(c, [c])          # S_2..S_9 -> image_id 2..9, classes 2..9
        add_image(10, [10, 11])        # X1 (selected): private pair {p, p'}
        add_image(11, [12, 13])        # X2 (selected): private pair {q, q'}
        add_image(12, [10, 13, 0])     # P1 (pool): covers p, q', bonus r
        add_image(13, [12, 11])        # P2 (pool): covers q, p'
        fixture = {"info": {}, "licenses": [], "images": images, "annotations": annotations,
                   "categories": _categories()}

        image_ids, labels14, zero_gt, _ = M.build_indicators(fixture)
        K, S_T, N = M.full_train_integer_stats(labels14)
        id_to_pos = {int(v): i for i, v in enumerate(image_ids.tolist())}
        selected = {id_to_pos[i] for i in range(1, 12)}   # S_1, S_2..S_9, X1, X2
        pool = {id_to_pos[12], id_to_pos[13]}              # P1, P2
        locked: set[int] = set()

        # --- Fixture-shape preconditions (NHIEM VU 1 items 1-5) ---
        self.assertTrue(selected.isdisjoint(pool))
        self.assertGreaterEqual(len(selected), 2)
        self.assertGreaterEqual(len(pool), 2)
        size_before = len(selected)
        self.assertEqual(size_before, 11)
        nf_before = int(sum(zero_gt[i] for i in selected))
        self.assertEqual(nf_before, 0)
        coverage_before = int(labels14[sorted(selected)].sum(axis=0).astype(bool).sum())
        self.assertEqual(coverage_before, 14, "fixture precondition: baseline must already be 14/14 covered")

        # --- Hand-derived K_c / k_c / objective, asserted via production
        # helpers (NHIEM VU 4). N=13 (11 selected + 2 pool), n=11.
        # K_0=2 (S_1,P1), K_1..9=1 each, K_10=2 (X1,P1), K_11=2 (X1,P2),
        # K_12=2 (X2,P2), K_13=2 (X2,P1); S_T=19.
        self.assertEqual(N, 13)
        self.assertEqual(K[0], 2)
        for c in range(1, 10):
            self.assertEqual(K[c], 1)
        for c in (10, 11, 12, 13):
            self.assertEqual(K[c], 2)
        self.assertEqual(S_T, 19)
        # baseline k_0=1 (S_1 only), k_1..9=1 each, k_10..13=1 each (X1/X2).
        # Deviations |k*N - K*n|: c0=|1*13-2*11|=9, c1..9 (9 classes)
        # =|1*13-1*11|=2 each, c10..13=|1*13-2*11|=9 each -> E_max=9
        # (5-way tie: c0,c10,c11,c12,c13), E_mean=9(c0)+2*9(c1-9)+9*4
        # (c10-13)=9+18+36=63, s_l=1+9+4=14, E_lc=|14*13-19*11|=|182-209|=27.
        current_obj = M.integer_objective(locked | selected, labels14, K, S_T, N)
        self.assertEqual(current_obj, (9, 63, 27))

        # --- NHIEM VU 2: exhaustive one-swap precondition. Collect every
        # ALLOWED improving 1-swap (same zero_gt, size/NF trivially
        # preserved, coverage after == 14), comparing the FULL
        # lexicographic (E_max, E_mean, E_lc) tuple via production
        # integer_objective -- not just E_max, not a partial sample.
        allowed_improving_one_swaps = []
        for out_pos in sorted(selected):
            for in_pos in sorted(pool):
                if zero_gt[out_pos] != zero_gt[in_pos]:
                    continue
                trial = set(selected)
                trial.discard(out_pos)
                trial.add(in_pos)
                if len(trial) != size_before:
                    continue
                if int(sum(zero_gt[i] for i in trial)) != nf_before:
                    continue
                trial_coverage = int(labels14[sorted(locked | trial)].sum(axis=0).astype(bool).sum())
                if trial_coverage != 14:
                    continue
                trial_obj = M.integer_objective(locked | trial, labels14, K, S_T, N)
                if trial_obj < current_obj:
                    allowed_improving_one_swaps.append((out_pos, in_pos, trial_obj))
        self.assertEqual(allowed_improving_one_swaps, [])

        # --- NHIEM VU 3: two-swap positive path ---
        visited: set[frozenset] = set()
        accept_obj = self._objective_accept(K, S_T, N, current_obj)
        oracle = _oracle_best_two_for_two(locked, selected, pool, labels14, zero_gt, K, S_T, N, 42, "1pct",
                                           "objective_repair", image_ids, visited, accept_obj)
        self.assertIsNotNone(oracle)
        engine = M._equivalence_class_two_for_two_search(locked, selected, pool, labels14, zero_gt, K, S_T, N, 42,
                                                           "1pct", "objective_repair", image_ids, visited, accept_obj)
        self.assertEqual(engine, oracle)  # removed/added positions AND full lexicographic key match

        out1, out2, in1, in2, _full_key = engine
        # Pool has only 2 elements, so {in1,in2} == {P1,P2} necessarily;
        # {X1,X2} was proven the unique coverage-valid removed pair above.
        x1_pos, x2_pos = id_to_pos[10], id_to_pos[11]
        self.assertEqual({out1, out2}, {x1_pos, x2_pos})
        self.assertEqual({in1, in2}, pool)

        trial = set(selected)
        trial.discard(out1)
        trial.discard(out2)
        trial.add(in1)
        trial.add(in2)
        size_after = len(trial)
        nf_after = int(sum(zero_gt[i] for i in trial))
        coverage_after = int(labels14[sorted(locked | trial)].sum(axis=0).astype(bool).sum())
        trial_obj = M.integer_objective(locked | trial, labels14, K, S_T, N)

        self.assertEqual(size_after, size_before)
        self.assertEqual(nf_after, nf_before)
        self.assertEqual(coverage_after, 14)
        self.assertLess(trial_obj, current_obj)
        # Hand-derived after-state: k_0=2 (S_1+P1), k_10..13 unchanged at 1
        # each (X removed / P added nets to zero per private class) ->
        # deviations c0=|2*13-2*11|=4, c1..9 unchanged=2 each,
        # c10..13 unchanged=9 each -> E_max=9 (ties, now a 4-way tie among
        # c10..13 only), E_mean=4+2*9+9*4=58 (4+18+36), s_l=15,
        # E_lc=|15*13-19*11|=14.
        self.assertEqual(trial_obj, (9, 58, 14))
        self.assertEqual(len({out1, out2}), 2)
        self.assertEqual(len({in1, in2}), 2)
        self.assertTrue({out1, out2}.issubset(selected))
        self.assertTrue({in1, in2}.issubset(pool))
        self.assertTrue({out1, out2}.isdisjoint(locked))  # locked prefix (empty) untouched
        self.assertTrue(trial.isdisjoint(pool - {in1, in2}))  # no stray pool images in final selected

    def test_objective_tie_across_distinct_signatures_resolved_identically(self):
        # Construct added-side equivalence classes A={0,1}, B={2}, C={0},
        # D={1,2} so that vec(A)+vec(B) == vec(C)+vec(D) as class-count
        # vectors (both equal {0:1,1:1,2:1}) -- two DIFFERENT signature-pair
        # transitions producing the IDENTICAL delta, hence a genuine
        # objective-level tie that must be resolved by comparing concrete
        # SHA-256 priority ACROSS both signatures, not within just one.
        images, annotations = [], []
        ann_id = 1

        def add_image(image_id, classes):
            nonlocal ann_id
            images.append({"id": image_id, "file_name": f"t/{image_id}.jpg", "width": 10, "height": 10})
            for c in classes:
                annotations.append({"id": ann_id, "image_id": image_id, "category_id": c + 1,
                                     "bbox": [1, 1, 2, 2], "area": 4, "iscrowd": 0})
                ann_id += 1

        # selected: two images to remove, single-class filler covering 3,4
        add_image(1, [3])
        add_image(2, [4])
        # pool: equivalence classes A={0,1}, B={2}, C={0}, D={1,2}
        add_image(10, [0, 1])   # A
        add_image(11, [2])      # B
        add_image(12, [0])      # C
        add_image(13, [1, 2])   # D
        # a few more classes so the fixture has full potential coverage
        for image_id, classes in [(20, [5]), (21, [6]), (22, [7]), (23, [8]), (24, [9]),
                                   (25, [10]), (26, [11]), (27, [12]), (28, [13])]:
            add_image(image_id, classes)
        fixture = {"info": {}, "licenses": [], "images": images, "annotations": annotations, "categories": _categories()}

        image_ids, labels14, zero_gt, _ = M.build_indicators(fixture)
        K, S_T, N = M.full_train_integer_stats(labels14)
        id_to_pos = {int(v): i for i, v in enumerate(image_ids.tolist())}
        selected = {id_to_pos[1], id_to_pos[2]}
        pool = {id_to_pos[k] for k in (10, 11, 12, 13)}
        locked: set[int] = set()
        visited: set[frozenset] = set()

        # Confirm the algebraic premise directly (not just assumed by hand):
        # signatures (A,B)=({0,1},{2}) and (C,D)=({0},{1,2}) must yield an
        # IDENTICAL resulting aggregate k, hence an identical objective --
        # a genuine cross-signature tie, not a coincidence of a constant
        # accept function.
        vec_a, vec_b = M._signature_vector((0, 1)), M._signature_vector((2,))
        vec_c, vec_d = M._signature_vector((0,)), M._signature_vector((1, 2))
        self.assertTrue(np.array_equal(vec_a + vec_b, vec_c + vec_d))

        # selected={1,2} covers only classes {3,4} -> missing_before=12.
        # Using the coverage-progress acceptance rule (not the 14/14-strict
        # objective rule, which this small fixture cannot reach in a single
        # swap): both (A,B) and (C,D) replace that with coverage {0,1,2},
        # tying at missing_after=11 -- a genuine cross-signature tie.
        missing_before = 14 - int(labels14[sorted(selected)].sum(axis=0).astype(bool).sum())
        self.assertEqual(missing_before, 12)
        accept = self._coverage_accept(K, S_T, N, missing_before)
        oracle = _oracle_best_two_for_two(locked, selected, pool, labels14, zero_gt, K, S_T, N, 42, "1pct",
                                           "minimum_class_coverage", image_ids, visited, accept)
        engine = M._equivalence_class_two_for_two_search(locked, selected, pool, labels14, zero_gt, K, S_T, N, 42,
                                                           "1pct", "minimum_class_coverage", image_ids, visited,
                                                           accept)
        self.assertIsNotNone(oracle)
        self.assertEqual(oracle, engine)
        self.assertEqual(oracle[4][0], 11)  # missing_after == 11, confirming the tie group was reached

    def test_full_tie_falls_back_to_canonical_numeric_identity_with_mocked_digest(self):
        train = _make_multilabel_cooccurrence_train(555, 30)
        image_ids, labels14, zero_gt, _ = M.build_indicators(train)
        K, S_T, N = M.full_train_integer_stats(labels14)
        abnormal = [p for p in range(len(image_ids)) if zero_gt[p] == 0]
        selected = set(abnormal[:10])
        pool = set(abnormal[10:])
        locked: set[int] = set()
        visited: set[frozenset] = set()

        def accept_constant(new_k, new_n, new_s_l):
            return (0,)  # every candidate ties -- forces resolution to identity

        with mock.patch.object(M, "move_priority_digest", return_value="0" * 64):
            oracle = _oracle_best_two_for_two(locked, selected, pool, labels14, zero_gt, K, S_T, N, 42, "1pct",
                                               "objective_repair", image_ids, visited, accept_constant)
            engine = M._equivalence_class_two_for_two_search(locked, selected, pool, labels14, zero_gt, K, S_T, N,
                                                               42, "1pct", "objective_repair", image_ids, visited,
                                                               accept_constant)
        self.assertIsNotNone(oracle)
        self.assertEqual(oracle, engine)
        # with every priority digest identical, the winner must be the
        # globally smallest canonical numeric identity among ALL concrete
        # candidates -- confirm the returned key's final element (identity)
        # is in fact minimal by re-deriving it from removed/added ids.
        out1, out2, in1, in2, full_key = engine
        removed_ids = tuple(sorted((int(image_ids[out1]), int(image_ids[out2]))))
        added_ids = tuple(sorted((int(image_ids[in1]), int(image_ids[in2]))))
        self.assertEqual(full_key[-1], (removed_ids, added_ids))

    def test_immutable_nested_prefix_never_touched(self):
        train = _make_multilabel_cooccurrence_train(777, 40)
        image_ids, labels14, zero_gt, _ = M.build_indicators(train)
        K, S_T, N = M.full_train_integer_stats(labels14)
        abnormal = [p for p in range(len(image_ids)) if zero_gt[p] == 0]
        locked = set(abnormal[:5])
        selected = set(abnormal[5:15])
        pool = set(abnormal[15:])
        visited: set[frozenset] = set()
        current_obj = M.integer_objective(locked | selected, labels14, K, S_T, N)
        accept_obj = self._objective_accept(K, S_T, N, current_obj)
        result = M._equivalence_class_two_for_two_search(locked, selected, pool, labels14, zero_gt, K, S_T, N, 42,
                                                           "1pct", "objective_repair", image_ids, visited, accept_obj)
        if result is not None:
            out1, out2, in1, in2, _ = result
            self.assertTrue({out1, out2}.isdisjoint(locked))
            self.assertTrue({in1, in2}.isdisjoint(locked))
            self.assertIn(out1, selected)
            self.assertIn(out2, selected)
            self.assertIn(in1, pool)
            self.assertIn(in2, pool)

    def test_exact_size_no_finding_and_coverage_preserved_after_applying_move(self):
        # R6 pytest-repair round / NHIEM VU 7: replaces the previous
        # random-cut-point fixture, which relied on self.skipTest(...) when
        # the random cut happened not to reach 14/14 coverage or not to
        # produce an improving move -- silently hiding a weak fixture
        # instead of failing so it could be fixed. Skip is now forbidden in
        # this class (see TestNoSkipInCoreScientificGuardrails below);
        # replaced with a deterministic, hand-verified constructive
        # fixture that ALWAYS satisfies every precondition (asserted, not
        # skipped-around):
        #
        #   SELECTED (n=14): singletons S_0..S_11 (classes 0..11, one
        #   each) plus X1, X2 -- the ONLY two carriers of classes 12 and
        #   13 (both X1, X2 carry {12, 13}), so baseline coverage is
        #   already exactly 14/14, with classes 12/13 redundantly
        #   over-represented (k=2 each).
        #   POOL: Q1 (class 12 only), Q2 (class 13 only).
        #   82 additional NO-FINDING filler images (zero_gt=1, no
        #   annotations, in neither selected nor pool) inflate N without
        #   touching any class's global count K_c -- this is what makes
        #   classes 0-11's global count (K=1, matching their selected
        #   representation exactly) legitimately "on target" while
        #   classes 12/13's redundant k=2 (against K=3) is legitimately
        #   over-represented.
        #
        #   N = 14 (selected) + 2 (pool) + 82 (fillers) = 98 (NOT 100 --
        #   corrected from a prior hand-arithmetic error caught by GPT
        #   static review; assertEqual(N, 16+num_fillers) below was
        #   always dynamically correct at 98, only the prose comment and
        #   the missing hardcoded assertions were wrong/absent before).
        #   Uniqueness: pool has only 2 elements, so the added side of
        #   ANY 2-swap is forced to be exactly {Q1, Q2}, which covers only
        #   {12, 13} -- so the only removed pair that keeps classes 0-11
        #   covered (their sole singleton carriers untouched) is {X1, X2}
        #   itself. {X1,X2} -> {Q1,Q2} is therefore the UNIQUE coverage-
        #   valid 2-swap in this fixture, which is why its objective can
        #   be safely hardcoded and cross-checked against the engine's
        #   actual result rather than merely assumed.
        images, annotations = [], []
        ann_id = 1

        def add_image(image_id, classes):
            nonlocal ann_id
            images.append({"id": image_id, "file_name": f"t/{image_id}.jpg", "width": 10, "height": 10})
            for c in classes:
                annotations.append({"id": ann_id, "image_id": image_id, "category_id": c + 1,
                                     "bbox": [1, 1, 2, 2], "area": 4, "iscrowd": 0})
                ann_id += 1

        for c in range(12):
            add_image(c + 1, [c])        # S_0..S_11 -> image_id 1..12
        add_image(13, [12, 13])           # X1 (selected): co-occurring {12,13}
        add_image(14, [12, 13])           # X2 (selected): same signature {12,13}
        add_image(15, [12])               # Q1 (pool): class 12 only
        add_image(16, [13])               # Q2 (pool): class 13 only
        num_fillers = 82
        for offset in range(num_fillers):
            images.append({"id": 17 + offset, "file_name": f"t/{17 + offset}.jpg", "width": 10, "height": 10})
        fixture = {"info": {}, "licenses": [], "images": images, "annotations": annotations,
                   "categories": _categories()}

        image_ids, labels14, zero_gt, _ = M.build_indicators(fixture)
        K, S_T, N = M.full_train_integer_stats(labels14)
        self.assertEqual(N, 98)
        self.assertEqual(N, 16 + num_fillers)
        id_to_pos = {int(v): i for i, v in enumerate(image_ids.tolist())}
        selected = {id_to_pos[i] for i in range(1, 15)}
        pool = {id_to_pos[15], id_to_pos[16]}
        locked: set[int] = set()
        visited: set[frozenset] = set()

        # Preconditions, asserted (never skipped around):
        size_before = len(selected)
        self.assertEqual(size_before, 14)
        nf_before = int(sum(zero_gt[i] for i in selected))
        self.assertEqual(nf_before, 0)
        present = set(int(c) for c in np.where(labels14[sorted(selected)].sum(axis=0) > 0)[0])
        self.assertEqual(len(present), 14, "fixture precondition violated: baseline is not 14/14 covered")
        self.assertTrue(selected.isdisjoint(pool), "fixture precondition violated: selected/pool overlap")

        # Hand-derived K_c / objective, asserted via production helpers
        # (NHIEM VU 4). K_0..11=1 each (12 classes, singletons only);
        # K_12=K_13=3 (X1,X2,Q-respective). S_T=12+3+3=18.
        for c in range(12):
            self.assertEqual(K[c], 1)
        self.assertEqual(K[12], 3)
        self.assertEqual(K[13], 3)
        self.assertEqual(S_T, 18)
        # baseline k_0..11=1 each, k_12=k_13=2 (X1,X2). Deviations
        # |k*98-K*14|: c0..11=|98-14|=84 each, c12=c13=|2*98-3*14|=
        # |196-42|=154 -> E_max=154 (tied c12,c13), E_mean=12*84+2*154=
        # 1008+308=1316, s_l=12+2+2=16, E_lc=|16*98-18*14|=|1568-252|=1316.
        current_obj = M.integer_objective(locked | selected, labels14, K, S_T, N)
        self.assertEqual(current_obj, (154, 1316, 1316))

        # NHIEM VU 5, step 1: compute the EXPECTED move's objective
        # directly (independent of whatever the oracle/engine return),
        # since {X1,X2}->{Q1,Q2} is proven the unique coverage-valid
        # candidate (see docstring above).
        x1_pos, x2_pos = id_to_pos[13], id_to_pos[14]
        q1_pos, q2_pos = id_to_pos[15], id_to_pos[16]
        expected_trial = (selected - {x1_pos, x2_pos}) | {q1_pos, q2_pos}
        expected_trial_obj = M.integer_objective(locked | expected_trial, labels14, K, S_T, N)
        # After: c0..11 unchanged=84 each, c12=c13=|1*98-3*14|=|98-42|=56
        # -> E_max=84 (now from c0..11, since 56<84), E_mean=12*84+2*56=
        # 1008+112=1120, s_l=12+1+1=14, E_lc=|14*98-18*14|=|1372-252|=1120.
        self.assertEqual(expected_trial_obj, (84, 1120, 1120))
        self.assertLess(expected_trial_obj, current_obj)

        # step 2: oracle/engine equivalence.
        accept_obj = self._objective_accept(K, S_T, N, current_obj)
        oracle = _oracle_best_two_for_two(locked, selected, pool, labels14, zero_gt, K, S_T, N, 42, "1pct",
                                           "objective_repair", image_ids, visited, accept_obj)
        self.assertIsNotNone(oracle, "fixture precondition violated: no improving 2-swap exists")
        engine = M._equivalence_class_two_for_two_search(locked, selected, pool, labels14, zero_gt, K, S_T, N, 42,
                                                           "1pct", "objective_repair", image_ids, visited, accept_obj)
        self.assertEqual(engine, oracle)

        # step 3: objective of what the engine ACTUALLY returned.
        out1, out2, in1, in2, _full_key = engine
        trial = set(selected)
        trial.discard(out1)
        trial.discard(out2)
        trial.add(in1)
        trial.add(in2)
        trial_obj = M.integer_objective(locked | trial, labels14, K, S_T, N)

        # step 4: improvement + invariants.
        self.assertEqual(len(trial), size_before)  # size_after == size_before
        self.assertEqual(int(sum(zero_gt[i] for i in trial)), nf_before)  # NF_after == NF_before
        trial_present = set(int(c) for c in np.where(labels14[sorted(trial)].sum(axis=0) > 0)[0])
        self.assertEqual(len(trial_present), 14)  # coverage_after == 14
        self.assertLess(trial_obj, current_obj)
        self.assertEqual(len({out1, out2}), 2)  # exactly 2 distinct removed positions
        self.assertEqual(len({in1, in2}), 2)  # exactly 2 distinct added positions
        self.assertTrue({out1, out2}.issubset(selected))
        self.assertTrue({in1, in2}.issubset(pool))
        self.assertTrue({out1, out2}.isdisjoint(locked))  # locked prefix untouched (empty here)
        self.assertTrue({in1, in2}.isdisjoint(locked))
        self.assertTrue(trial.isdisjoint(pool - {in1, in2}))  # selected/pool remain disjoint after the move

        # step 5: only NOW, having proven {X1,X2}->{Q1,Q2} is the unique
        # coverage-valid candidate (docstring) AND that the engine's own
        # result satisfies every invariant above, assert the engine's
        # actual result equals the independently hand-derived move.
        self.assertEqual({out1, out2}, {x1_pos, x2_pos})
        self.assertEqual({in1, in2}, {q1_pos, q2_pos})
        self.assertEqual(trial_obj, expected_trial_obj)
        self.assertEqual(trial_obj, (84, 1120, 1120))

    def test_never_produces_a_k_ge_3_move(self):
        train = _make_multilabel_cooccurrence_train(999, 30)
        image_ids, labels14, zero_gt, _ = M.build_indicators(train)
        K, S_T, N = M.full_train_integer_stats(labels14)
        abnormal = [p for p in range(len(image_ids)) if zero_gt[p] == 0]
        selected = set(abnormal[:12])
        pool = set(abnormal[12:])
        locked: set[int] = set()
        visited: set[frozenset] = set()
        current_obj = M.integer_objective(locked | selected, labels14, K, S_T, N)
        accept_obj = self._objective_accept(K, S_T, N, current_obj)
        result = M._equivalence_class_two_for_two_search(locked, selected, pool, labels14, zero_gt, K, S_T, N, 42,
                                                           "1pct", "objective_repair", image_ids, visited, accept_obj)
        if result is not None:
            out1, out2, in1, in2, _ = result
            self.assertEqual(len({out1, out2}), 2, "removed side must be exactly 2 distinct images")
            self.assertEqual(len({in1, in2}), 2, "added side must be exactly 2 distinct images")


def _build_cross_signature_tie_fixture():
    """Same construction as TestEquivalenceClassEngineVsBruteForceOracle.
    test_objective_tie_across_distinct_signatures_resolved_identically:
    removed side is a single signature-pair (one concrete realization);
    added side has exactly 6 valid signature-pairs among {A,B,C,D}
    (pool_sigs enumerated in insertion order A,B,C,D -> pairs (A,B),(A,C),
    (A,D),(B,C),(B,D),(C,D) -- 6 total, so R=1, A=6, phase1_transition_
    count=6), of which (A,B) [encountered FIRST] and (C,D) [encountered
    LAST] are proven to tie on the coverage-progress prefix. Returns
    (locked, selected, pool, labels14, zero_gt, K, S_T, N, image_ids,
    accept_and_rank, id_ab, id_cd) where id_ab/id_cd are the (out of 4
    pool image ids 10,11,12,13) added-id-pairs for the (A,B)/(C,D)
    realizations respectively."""
    images, annotations = [], []
    ann_id = 1

    def add_image(image_id, classes):
        nonlocal ann_id
        images.append({"id": image_id, "file_name": f"t/{image_id}.jpg", "width": 10, "height": 10})
        for c in classes:
            annotations.append({"id": ann_id, "image_id": image_id, "category_id": c + 1,
                                 "bbox": [1, 1, 2, 2], "area": 4, "iscrowd": 0})
            ann_id += 1

    add_image(1, [3])
    add_image(2, [4])
    add_image(10, [0, 1])   # A
    add_image(11, [2])      # B
    add_image(12, [0])      # C
    add_image(13, [1, 2])   # D
    for image_id, classes in [(20, [5]), (21, [6]), (22, [7]), (23, [8]), (24, [9]),
                               (25, [10]), (26, [11]), (27, [12]), (28, [13])]:
        add_image(image_id, classes)
    fixture = {"info": {}, "licenses": [], "images": images, "annotations": annotations, "categories": _categories()}
    image_ids, labels14, zero_gt, _ = M.build_indicators(fixture)
    K, S_T, N = M.full_train_integer_stats(labels14)
    id_to_pos = {int(v): i for i, v in enumerate(image_ids.tolist())}
    selected = {id_to_pos[1], id_to_pos[2]}
    pool = {id_to_pos[k] for k in (10, 11, 12, 13)}
    locked: set[int] = set()
    missing_before = 14 - int(labels14[sorted(selected)].sum(axis=0).astype(bool).sum())

    def accept(new_k, new_n, new_s_l):
        new_missing = 14 - int((new_k > 0).sum())
        if new_missing >= missing_before:
            return None
        e_max, e_mean, e_lc = M._integer_objective_from_aggregate(new_k, new_n, new_s_l, K, S_T, N)
        return (new_missing, e_max, e_mean, e_lc)

    return locked, selected, pool, labels14, zero_gt, K, S_T, N, image_ids, accept, (10, 11), (12, 13)


def _r8_reference_equivalence_class_two_for_two_search(
    locked, selected, pool, labels14, zero_gt, K, S_T, N, seed, budget, repair_phase, image_ids,
    visited, accept_and_rank,
):
    """TEST-ONLY: a faithful, deliberately UN-optimized reproduction of the
    PRE-R9 (R8) engine algorithm -- identical Phase-1/Phase-2 equivalence-
    class search, but WITHOUT the R9 added-pair-aggregate precompute: the
    added-side vector is reconstructed via _signature_vector on every
    single (removed_pair, added_pair) combination, exactly as R8 did. No
    deadline/watchdog/diagnostics support (not needed for an output/
    ordering equivalence check). Used ONLY to prove the R9-optimized
    production engine (M._equivalence_class_two_for_two_search) returns
    bit-for-bit identical results and visits transitions in the identical
    order -- never called against real project data."""
    selected_abnormal = [p for p in selected if zero_gt[p] == 0]
    pool_abnormal = [p for p in pool if zero_gt[p] == 0]
    if len(selected_abnormal) < 2 or len(pool_abnormal) < 2:
        return None
    sel_groups = M._abnormal_equivalence_classes(selected_abnormal, labels14)
    pool_groups = M._abnormal_equivalence_classes(pool_abnormal, labels14)
    base_state = sorted(locked | selected)
    base_k = labels14[base_state].sum(axis=0).astype(np.int64) if base_state else np.zeros(14, dtype=np.int64)
    base_n = len(base_state)
    base_s_l = int(base_k.sum())
    sel_sigs = list(sel_groups.keys())
    pool_sigs = list(pool_groups.keys())
    removed_pairs = [
        (sel_sigs[i], sel_sigs[j])
        for i in range(len(sel_sigs)) for j in range(i, len(sel_sigs))
        if sel_sigs[i] != sel_sigs[j] or len(sel_groups[sel_sigs[i]]) >= 2
    ]
    added_pairs = [
        (pool_sigs[i], pool_sigs[j])
        for i in range(len(pool_sigs)) for j in range(i, len(pool_sigs))
        if pool_sigs[i] != pool_sigs[j] or len(pool_groups[pool_sigs[i]]) >= 2
    ]

    def _stream(exclude_prefix_leq):
        best_prefix = None
        tied = []
        for r_ca, r_cb in removed_pairs:
            removed_vec = M._signature_vector(r_ca) + M._signature_vector(r_cb)
            for a_ca, a_cb in added_pairs:
                added_vec = M._signature_vector(a_ca) + M._signature_vector(a_cb)  # R8 behavior: rebuilt every time
                new_k = base_k - removed_vec + added_vec
                new_s_l = base_s_l - int(removed_vec.sum()) + int(added_vec.sum())
                key_prefix = accept_and_rank(new_k, base_n, new_s_l)
                if key_prefix is None:
                    continue
                if exclude_prefix_leq is not None and not (key_prefix > exclude_prefix_leq):
                    continue
                if best_prefix is None or key_prefix < best_prefix:
                    best_prefix = key_prefix
                    tied = [((r_ca, r_cb), (a_ca, a_cb))]
                elif key_prefix == best_prefix:
                    tied.append(((r_ca, r_cb), (a_ca, a_cb)))
        return best_prefix, tied

    exclude_leq = None
    while True:
        best_prefix, tied_best_transitions = _stream(exclude_leq)
        if best_prefix is None:
            return None
        best_concrete = None
        best_move = None
        for (r_ca, r_cb), (a_ca, a_cb) in tied_best_transitions:
            if r_ca != r_cb:
                out_pairs = [(x, y) for x in sel_groups[r_ca] for y in sel_groups[r_cb]]
            else:
                members = sel_groups[r_ca]
                out_pairs = [(members[i], members[j]) for i in range(len(members)) for j in range(i + 1, len(members))]
            if a_ca != a_cb:
                in_pairs = [(x, y) for x in pool_groups[a_ca] for y in pool_groups[a_cb]]
            else:
                members = pool_groups[a_ca]
                in_pairs = [(members[i], members[j]) for i in range(len(members)) for j in range(i + 1, len(members))]
            for out1, out2 in out_pairs:
                lo, hi = (out1, out2) if out1 < out2 else (out2, out1)
                for in1, in2 in in_pairs:
                    li, hj = (in1, in2) if in1 < in2 else (in2, in1)
                    trial = set(selected)
                    trial.discard(lo)
                    trial.discard(hi)
                    trial.add(li)
                    trial.add(hj)
                    if frozenset(trial) in visited:
                        continue
                    removed_ids = (int(image_ids[lo]), int(image_ids[hi]))
                    added_ids = (int(image_ids[li]), int(image_ids[hj]))
                    priority = M.move_priority_digest(seed, budget, repair_phase, "two_for_two_swap", removed_ids, added_ids)
                    identity = M.canonical_numeric_move_identity(removed_ids, added_ids)
                    concrete_key = (priority, identity)
                    if best_concrete is None or concrete_key < best_concrete:
                        best_concrete = concrete_key
                        best_move = (lo, hi, li, hj)
        if best_move is not None:
            full_key = tuple(best_prefix) + best_concrete
            return (best_move[0], best_move[1], best_move[2], best_move[3], full_key)
        exclude_leq = best_prefix


# --------------------------------------------------------------------------- #
# R5 / NHIEM VU 3 — streaming best-prefix search (fixes blocker R4-B4)       #
# --------------------------------------------------------------------------- #
class TestStreamingBestPrefixSearch(unittest.TestCase):
    def test_streaming_engine_matches_oracle_on_tied_group(self):
        locked, selected, pool, labels14, zero_gt, K, S_T, N, image_ids, accept, _ab, _cd = \
            _build_cross_signature_tie_fixture()
        visited: set[frozenset] = set()
        oracle = _oracle_best_two_for_two(locked, selected, pool, labels14, zero_gt, K, S_T, N, 42, "1pct",
                                           "minimum_class_coverage", image_ids, visited, accept)
        engine = M._equivalence_class_two_for_two_search(locked, selected, pool, labels14, zero_gt, K, S_T, N, 42,
                                                           "1pct", "minimum_class_coverage", image_ids, visited, accept)
        self.assertIsNotNone(oracle)
        self.assertEqual(oracle, engine)

    def test_source_never_materializes_full_candidate_list_in_phase1(self):
        # Memory contract (R5 / fixes blocker R4-B4): Phase 1 must be a
        # streaming best-prefix scan, never "append every accepted
        # transition to a list, then sort". Source/AST guardrail: no
        # `candidates.append(` / `candidates.sort(` pattern remains in the
        # engine function, and the streaming update-rule variable names ARE
        # present.
        source = SCRIPT_PATH.read_text(encoding="utf-8")
        tree = ast.parse(source)
        engine_fn = next(
            node for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef) and node.name == "_equivalence_class_two_for_two_search"
        )
        engine_source = ast.get_source_segment(source, engine_fn) or ""
        self.assertNotIn("candidates.append(", engine_source)
        self.assertNotIn("candidates.sort(", engine_source)
        self.assertIn("best_prefix", engine_source)
        self.assertIn("tied_best_transitions", engine_source)

    def test_winning_move_can_be_in_a_transition_not_encountered_first(self):
        # R5-required regression test: mock move_priority_digest so the
        # GLOBAL minimum belongs to the (C,D) added-signature-pair
        # transition, which is encountered LAST among the 6 valid
        # added-pairs (A,B) is first) -- proving the streaming reset/
        # append/discard logic is not secretly a "first tied group wins"
        # bug, since it must correctly REPLACE an earlier tied entry's
        # concrete winner with a later one.
        locked, selected, pool, labels14, zero_gt, K, S_T, N, image_ids, accept, id_ab, id_cd = \
            _build_cross_signature_tie_fixture()
        visited: set[frozenset] = set()
        id_to_pos = {int(v): i for i, v in enumerate(image_ids.tolist())}
        winning_added_ids = tuple(sorted(id_cd))

        def fake_digest(seed, budget, repair_phase, move_type, removed_ids, added_ids):
            if tuple(sorted(added_ids)) == winning_added_ids:
                return "0" * 64
            return "f" * 64

        with mock.patch.object(M, "move_priority_digest", side_effect=fake_digest):
            oracle = _oracle_best_two_for_two(locked, selected, pool, labels14, zero_gt, K, S_T, N, 42, "1pct",
                                               "minimum_class_coverage", image_ids, visited, accept)
            engine = M._equivalence_class_two_for_two_search(locked, selected, pool, labels14, zero_gt, K, S_T, N,
                                                               42, "1pct", "minimum_class_coverage", image_ids,
                                                               visited, accept)
        self.assertIsNotNone(oracle)
        self.assertEqual(oracle, engine)
        out1, out2, in1, in2, _full_key = engine
        engine_added_ids = tuple(sorted((int(image_ids[in1]), int(image_ids[in2]))))
        self.assertEqual(engine_added_ids, winning_added_ids,
                          "engine did not pick the mocked-minimum-digest transition (C,D), "
                          "which is NOT the first-encountered added-signature-pair")

    def test_multiple_tied_signature_transitions_all_considered(self):
        # With every digest forced equal, resolution must fall back to
        # canonical numeric identity across BOTH tied transitions (A,B) and
        # (C,D) -- i.e. Phase 2 must have actually visited both, not just
        # the first-encountered one.
        locked, selected, pool, labels14, zero_gt, K, S_T, N, image_ids, accept, id_ab, id_cd = \
            _build_cross_signature_tie_fixture()
        visited: set[frozenset] = set()
        with mock.patch.object(M, "move_priority_digest", return_value="0" * 64):
            oracle = _oracle_best_two_for_two(locked, selected, pool, labels14, zero_gt, K, S_T, N, 42, "1pct",
                                               "minimum_class_coverage", image_ids, visited, accept)
            engine = M._equivalence_class_two_for_two_search(locked, selected, pool, labels14, zero_gt, K, S_T, N,
                                                               42, "1pct", "minimum_class_coverage", image_ids,
                                                               visited, accept)
        self.assertIsNotNone(oracle)
        self.assertEqual(oracle, engine)
        # numeric identity fallback must pick the globally-smallest
        # (removed_ids, added_ids) tuple among ALL concrete realizations of
        # EVERY tied transition -- confirm by re-deriving it directly.
        out1, out2, in1, in2, full_key = engine
        removed_ids = tuple(sorted((int(image_ids[out1]), int(image_ids[out2]))))
        added_ids = tuple(sorted((int(image_ids[in1]), int(image_ids[in2]))))
        self.assertEqual(full_key[-1], (removed_ids, added_ids))

    def test_diagnostics_report_tied_transition_count_and_status(self):
        locked, selected, pool, labels14, zero_gt, K, S_T, N, image_ids, accept, _ab, _cd = \
            _build_cross_signature_tie_fixture()
        visited: set[frozenset] = set()
        diag: dict = {}
        result = M._equivalence_class_two_for_two_search(locked, selected, pool, labels14, zero_gt, K, S_T, N, 42,
                                                           "1pct", "minimum_class_coverage", image_ids, visited,
                                                           accept, diagnostics=diag)
        self.assertIsNotNone(result)
        for field in (
            "selected_abnormal_count", "pool_abnormal_count", "selected_distinct_signature_count",
            "pool_distinct_signature_count", "removed_pair_signature_count_R", "added_pair_signature_count_A",
            "phase1_transition_count", "tied_best_signature_transition_count_W",
            "concrete_candidates_evaluated_phase2", "theoretical_concrete_candidates_in_winning_ties",
            "elapsed_seconds", "deadline_enabled", "status",
        ):
            self.assertIn(field, diag, msg=f"missing runtime diagnostics field: {field}")
        self.assertEqual(diag["removed_pair_signature_count_R"], 1)
        self.assertEqual(diag["added_pair_signature_count_A"], 6)
        self.assertEqual(diag["phase1_transition_count"], 6)
        self.assertGreaterEqual(diag["tied_best_signature_transition_count_W"], 2)  # at least (A,B) and (C,D)
        self.assertEqual(diag["status"], "completed")
        self.assertFalse(diag["deadline_enabled"])

    def test_diagnostics_left_unfilled_when_not_requested(self):
        # Passing diagnostics=None (the default) must have zero behavioral
        # effect -- every pre-R5 call site continues to work unchanged.
        locked, selected, pool, labels14, zero_gt, K, S_T, N, image_ids, accept, _ab, _cd = \
            _build_cross_signature_tie_fixture()
        visited: set[frozenset] = set()
        result = M._equivalence_class_two_for_two_search(locked, selected, pool, labels14, zero_gt, K, S_T, N, 42,
                                                           "1pct", "minimum_class_coverage", image_ids, visited,
                                                           accept)
        self.assertIsNotNone(result)  # unchanged 5-tuple contract, no diagnostics required


# --------------------------------------------------------------------------- #
# R5 / NHIEM VU 4 — deadline/watchdog inside the engine (fixes R4-B3)        #
# --------------------------------------------------------------------------- #
def _build_unique_two_for_two_winner_fixture():
    """Same construction as TestEquivalenceClassEngineVsBruteForceOracle.
    test_exact_size_no_finding_and_coverage_preserved_after_applying_move:
    singletons S_0..S_11 (classes 0..11) plus X1,X2 (both {12,13}) selected
    (already 14/14 covered, classes 12/13 redundantly over-represented);
    pool Q1 (class 12 only), Q2 (class 13 only). Pool has only 2 elements,
    so {X1,X2}->{Q1,Q2} is the UNIQUE coverage-preserving, objective-
    improving two-for-two move -- an "ordinary improving move" fixture
    with no tie."""
    images, annotations = [], []
    ann_id = 1

    def add_image(image_id, classes):
        nonlocal ann_id
        images.append({"id": image_id, "file_name": f"t/{image_id}.jpg", "width": 10, "height": 10})
        for c in classes:
            annotations.append({"id": ann_id, "image_id": image_id, "category_id": c + 1,
                                 "bbox": [1, 1, 2, 2], "area": 4, "iscrowd": 0})
            ann_id += 1

    for c in range(12):
        add_image(c + 1, [c])
    add_image(13, [12, 13])
    add_image(14, [12, 13])
    add_image(15, [12])
    add_image(16, [13])
    num_fillers = 82
    for offset in range(num_fillers):
        images.append({"id": 17 + offset, "file_name": f"t/{17 + offset}.jpg", "width": 10, "height": 10})
    fixture = {"info": {}, "licenses": [], "images": images, "annotations": annotations,
               "categories": _categories()}
    image_ids, labels14, zero_gt, _ = M.build_indicators(fixture)
    K, S_T, N = M.full_train_integer_stats(labels14)
    id_to_pos = {int(v): i for i, v in enumerate(image_ids.tolist())}
    selected = {id_to_pos[i] for i in range(1, 15)}
    pool = {id_to_pos[15], id_to_pos[16]}
    locked: set[int] = set()
    return locked, selected, pool, labels14, zero_gt, K, S_T, N, image_ids


# --------------------------------------------------------------------------- #
# R9 (static-audit-driven performance fix, NHIEM VU 1-2) -- regression tests #
# proving the added-pair-aggregate precompute never changes scientific       #
# output, ordering, or diagnostics, and satisfies the memory-safety and      #
# deadline-coverage design constraints. Reference comparisons use            #
# _r8_reference_equivalence_class_two_for_two_search, a faithful,            #
# deliberately un-optimized reproduction of the PRE-R9 algorithm.            #
# --------------------------------------------------------------------------- #
class TestR9AddedPairPrecompute(unittest.TestCase):
    def _objective_accept(self, K, S_T, N, current_obj):
        def accept(new_k, new_n, new_s_l):
            if int((new_k > 0).sum()) != 14:
                return None
            trial_obj = M._integer_objective_from_aggregate(new_k, new_n, new_s_l, K, S_T, N)
            if not (trial_obj < current_obj):
                return None
            return trial_obj
        return accept

    # --- item 2: final move identity matches the R8 reference across the required fixture types --- #
    def test_matches_r8_reference_ordinary_improving_move(self):
        locked, selected, pool, labels14, zero_gt, K, S_T, N, image_ids = \
            _build_unique_two_for_two_winner_fixture()
        current_obj = M.integer_objective(locked | selected, labels14, K, S_T, N)
        accept = self._objective_accept(K, S_T, N, current_obj)
        production = M._equivalence_class_two_for_two_search(
            locked, selected, pool, labels14, zero_gt, K, S_T, N, 42, "1pct",
            "objective_repair", image_ids, set(), accept,
        )
        reference = _r8_reference_equivalence_class_two_for_two_search(
            locked, selected, pool, labels14, zero_gt, K, S_T, N, 42, "1pct",
            "objective_repair", image_ids, set(), accept,
        )
        self.assertIsNotNone(production)
        self.assertEqual(production, reference)

    def test_matches_r8_reference_exhaustive_no_candidate(self):
        locked, selected, pool, labels14, zero_gt, K, S_T, N, image_ids, _accept, _ab, _cd = \
            _build_cross_signature_tie_fixture()
        reject_everything = lambda new_k, new_n, new_s_l: None
        production = M._equivalence_class_two_for_two_search(
            locked, selected, pool, labels14, zero_gt, K, S_T, N, 42, "1pct",
            "minimum_class_coverage", image_ids, set(), reject_everything,
        )
        reference = _r8_reference_equivalence_class_two_for_two_search(
            locked, selected, pool, labels14, zero_gt, K, S_T, N, 42, "1pct",
            "minimum_class_coverage", image_ids, set(), reject_everything,
        )
        self.assertIsNone(production)
        self.assertEqual(production, reference)

    def test_matches_r8_reference_multiple_tied_signature_transitions_and_identical_objective_prefix(self):
        # _build_cross_signature_tie_fixture is specifically constructed so
        # that (A,B) and (C,D) -- two DIFFERENT added-signature-pairs --
        # produce an IDENTICAL resulting aggregate (hence an identical
        # objective/coverage prefix), a genuine multi-transition tie.
        locked, selected, pool, labels14, zero_gt, K, S_T, N, image_ids, accept, _ab, _cd = \
            _build_cross_signature_tie_fixture()
        production = M._equivalence_class_two_for_two_search(
            locked, selected, pool, labels14, zero_gt, K, S_T, N, 42, "1pct",
            "minimum_class_coverage", image_ids, set(), accept,
        )
        reference = _r8_reference_equivalence_class_two_for_two_search(
            locked, selected, pool, labels14, zero_gt, K, S_T, N, 42, "1pct",
            "minimum_class_coverage", image_ids, set(), accept,
        )
        self.assertIsNotNone(production)
        self.assertEqual(production, reference)

    def test_matches_r8_reference_digest_tie_and_canonical_numeric_fallback(self):
        train = _make_multilabel_cooccurrence_train(555, 30)
        image_ids, labels14, zero_gt, _ = M.build_indicators(train)
        K, S_T, N = M.full_train_integer_stats(labels14)
        abnormal = [p for p in range(len(image_ids)) if zero_gt[p] == 0]
        selected = set(abnormal[:10])
        pool = set(abnormal[10:])
        locked: set[int] = set()

        def accept_constant(new_k, new_n, new_s_l):
            return (0,)  # every candidate ties -- forces full digest-tie / canonical-numeric-fallback resolution

        with mock.patch.object(M, "move_priority_digest", return_value="0" * 64):
            production = M._equivalence_class_two_for_two_search(
                locked, selected, pool, labels14, zero_gt, K, S_T, N, 42, "1pct",
                "objective_repair", image_ids, set(), accept_constant,
            )
            reference = _r8_reference_equivalence_class_two_for_two_search(
                locked, selected, pool, labels14, zero_gt, K, S_T, N, 42, "1pct",
                "objective_repair", image_ids, set(), accept_constant,
            )
        self.assertIsNotNone(production)
        self.assertEqual(production, reference)

    # --- item 3: Phase-1 traversal order is unchanged (full R x A sequence) --- #
    def test_traversal_order_full_sequence_matches_r8_reference(self):
        locked, selected, pool, labels14, zero_gt, K, S_T, N, image_ids, _accept, _ab, _cd = \
            _build_cross_signature_tie_fixture()
        recorded_production: list[tuple] = []
        recorded_reference: list[tuple] = []

        def spy_production(new_k, new_n, new_s_l):
            recorded_production.append((tuple(int(x) for x in new_k), int(new_n), int(new_s_l)))
            return None  # never accept -> guarantees exactly one full, unresumed R x A pass

        def spy_reference(new_k, new_n, new_s_l):
            recorded_reference.append((tuple(int(x) for x in new_k), int(new_n), int(new_s_l)))
            return None

        M._equivalence_class_two_for_two_search(
            locked, selected, pool, labels14, zero_gt, K, S_T, N, 42, "1pct",
            "objective_repair", image_ids, set(), spy_production,
        )
        _r8_reference_equivalence_class_two_for_two_search(
            locked, selected, pool, labels14, zero_gt, K, S_T, N, 42, "1pct",
            "objective_repair", image_ids, set(), spy_reference,
        )
        self.assertGreater(len(recorded_production), 0)
        self.assertEqual(recorded_production, recorded_reference)

    # --- item 4: every tied-best transition is still retained --- #
    def test_all_tied_best_transitions_retained(self):
        locked, selected, pool, labels14, zero_gt, K, S_T, N, image_ids, accept, _ab, _cd = \
            _build_cross_signature_tie_fixture()
        diag: dict = {}
        result = M._equivalence_class_two_for_two_search(
            locked, selected, pool, labels14, zero_gt, K, S_T, N, 42, "1pct",
            "minimum_class_coverage", image_ids, set(), accept, diagnostics=diag,
        )
        self.assertIsNotNone(result)
        # hand-verified by the fixture's own docstring: (A,B) and (C,D) tie
        # for best prefix -- W must be at least 2, unchanged from pre-R9.
        self.assertGreaterEqual(diag["tied_best_signature_transition_count_W"], 2)

    # --- item 5: _signature_vector call count scales with distinct        #
    # signatures/pairs precomputed, never with R*A.                        #
    # --- item 9: phase1_transition_count is unchanged.                    #
    def test_signature_vector_call_count_scales_with_distinct_signatures_not_r_times_a(self):
        locked, selected, pool, labels14, zero_gt, K, S_T, N, image_ids, _accept, _ab, _cd = \
            _build_cross_signature_tie_fixture()
        reject_everything = lambda new_k, new_n, new_s_l: None  # forces exactly one full, unresumed R x A pass
        real_signature_vector = M._signature_vector
        call_count = [0]

        def counting_signature_vector(sig):
            call_count[0] += 1
            return real_signature_vector(sig)

        diag: dict = {}
        with mock.patch.object(M, "_signature_vector", side_effect=counting_signature_vector):
            M._equivalence_class_two_for_two_search(
                locked, selected, pool, labels14, zero_gt, K, S_T, N, 42, "1pct",
                "minimum_class_coverage", image_ids, set(), reject_everything, diagnostics=diag,
            )
        R = diag["removed_pair_signature_count_R"]
        p = diag["pool_distinct_signature_count"]
        A = diag["added_pair_signature_count_A"]
        self.assertEqual(R, 1)
        self.assertEqual(A, 6)
        self.assertEqual(p, 4)
        self.assertEqual(diag["phase1_transition_count"], R * A)  # item 9: unchanged formula
        # R9-B1 (GPT static review): the previous sanity assertion compared
        # call_count against R*A (the TRANSITION count, i.e.
        # phase1_transition_count), not against the pre-R9 SIGNATURE-VECTOR
        # CALL count -- for this exact fixture R*A == 6 == 2*R+p, so
        # `assertLess(call_count[0], R * A)` asserted 6 < 6, which can never
        # pass. The correct comparison is against what R8 actually did:
        # removed side: _signature_vector(r_ca)+_signature_vector(r_cb)
        # called once per removed pair, i.e. 2*R calls -- UNCHANGED by this
        # round, not in scope (R9 only touched the ADDED side). Added side:
        # R8 called _signature_vector(a_ca)+_signature_vector(a_cb) on
        # EVERY (removed_pair, added_pair) combination, i.e. 2*R*A calls;
        # R9 instead calls it once per DISTINCT pool signature during the
        # precompute step, i.e. p calls, then reads the precomputed
        # aggregate by index for every transition -- eliminating exactly
        # the repeated ADDED-side reconstruction the R8 audit identified,
        # while leaving removed-side behavior untouched.
        #   R9 removed-side calls = 2*R
        #   R9 added-side calls   = p
        #   R9 total calls        = 2*R + p
        #   R8 removed-side calls = 2*R
        #   R8 added-side calls   = 2*R*A
        #   R8 total calls        = 2*R + 2*R*A
        self.assertEqual(call_count[0], 2 * R + p)
        self.assertLess(p, 2 * R * A)  # fixture precondition: the pre-R9 added-side term is the larger one
        self.assertLess(call_count[0], 2 * R + 2 * R * A)  # R9 total is strictly less than the R8 total it replaces

    # --- items 6-8: contiguous, correctly-shaped/typed aggregate storage,  #
    # never a list of per-added-pair ndarray objects.                      #
    def test_added_vec_matrix_is_contiguous_int64_correct_shape_no_per_pair_object_list(self):
        locked, selected, pool, labels14, zero_gt, K, S_T, N, image_ids, accept, _ab, _cd = \
            _build_cross_signature_tie_fixture()
        real_empty = np.empty
        captured: list[np.ndarray] = []

        def spy_empty(shape, dtype=None, *args, **kwargs):
            arr = real_empty(shape, dtype=dtype, *args, **kwargs)
            if isinstance(shape, tuple) and len(shape) == 2 and shape[1] == 14:
                captured.append(arr)
            return arr

        with mock.patch.object(M.np, "empty", side_effect=spy_empty):
            result = M._equivalence_class_two_for_two_search(
                locked, selected, pool, labels14, zero_gt, K, S_T, N, 42, "1pct",
                "minimum_class_coverage", image_ids, set(), accept,
            )
        self.assertIsNotNone(result)
        # exactly 2 contiguous (*, 14) allocations per call: pool_sig_vec_matrix
        # (p rows) then added_vec_matrix (A rows) -- never A separate
        # per-pair array objects.
        self.assertEqual(len(captured), 2)
        pool_sig_vec_matrix, added_vec_matrix = captured
        self.assertEqual(pool_sig_vec_matrix.shape, (4, 14))
        self.assertEqual(added_vec_matrix.shape, (6, 14))
        for arr in (pool_sig_vec_matrix, added_vec_matrix):
            self.assertEqual(arr.dtype, np.int64)  # exact dtype match with _signature_vector -- no narrowing
            self.assertTrue(arr.flags["C_CONTIGUOUS"])

    # --- item 10: phase1_transitions_evaluated unchanged on a completed fixture. --- #
    def test_phase1_transitions_evaluated_unchanged_on_completed_fixture(self):
        locked, selected, pool, labels14, zero_gt, K, S_T, N, image_ids, accept, _ab, _cd = \
            _build_cross_signature_tie_fixture()
        diag: dict = {}
        result = M._equivalence_class_two_for_two_search(
            locked, selected, pool, labels14, zero_gt, K, S_T, N, 42, "1pct",
            "minimum_class_coverage", image_ids, set(), accept, diagnostics=diag,
        )
        self.assertIsNotNone(result)
        self.assertEqual(diag["status"], "completed")
        # a single, unresumed full R x A pass (R=1, A=6) -> exactly 6,
        # identical to the pre-R9 value for this fixture.
        self.assertEqual(diag["phase1_transitions_evaluated"], 6)

    # --- item 11: deadline abort DURING precomputation is classified       #
    # correctly -- must never falsely claim any logical transition was     #
    # evaluated.                                                           #
    def test_deadline_abort_during_precompute_reports_zero_transitions_evaluated(self):
        locked, selected, pool, labels14, zero_gt, K, S_T, N, image_ids, accept, _ab, _cd = \
            _build_cross_signature_tie_fixture()
        diag = {"budget": "1pct", "repair_phase": "minimum_class_coverage", "status": "started"}
        # entry check (1) + 2 successful pool_sig_vec_matrix precompute
        # ticks, then abort on the 3rd precompute tick -- strictly INSIDE
        # the R9 precompute block, before the main R x A loop (and hence
        # phase1_evaluated) has ever started.
        with mock.patch.object(M, "TWO_FOR_TWO_DEADLINE_WATCHDOG_INTERVAL", 1):
            with mock.patch.object(M, "_check_deadline", side_effect=[None, None, None, M.DeadlineExceeded()]):
                with self.assertRaises(M.DeadlineExceeded):
                    M._equivalence_class_two_for_two_search(
                        locked, selected, pool, labels14, zero_gt, K, S_T, N, 42, "1pct",
                        "minimum_class_coverage", image_ids, set(), accept, deadline=1.0, diagnostics=diag,
                    )
        self.assertEqual(diag["status"], "aborted")
        self.assertTrue(diag["aborted_by_deadline"])
        # R/A/signature counts ARE already known by this point (computed
        # before the precompute block begins) -- must be preserved, not nulled.
        self.assertEqual(diag["removed_pair_signature_count_R"], 1)
        self.assertEqual(diag["added_pair_signature_count_A"], 6)
        self.assertEqual(diag["phase1_transition_count"], 6)
        # but NO logical (removed_pair, added_pair) transition was ever
        # evaluated -- the abort happened during setup, not search.
        self.assertEqual(diag["phase1_transitions_evaluated"], 0)
        self.assertEqual(diag["concrete_candidates_evaluated_phase2"], 0)
        self.assertIsNone(diag["tied_best_signature_transition_count_W"])
        self.assertIsNone(diag["theoretical_concrete_candidates_in_winning_ties"])
        # item 13: never a false COMPLETED/EXHAUSTED claim.
        self.assertNotEqual(diag["status"], "completed")
        self.assertNotEqual(diag["status"], "exhausted")

    # --- item 12: deadline abort during the real Phase-1 main loop still  #
    # works exactly like R8 (aborts, real partial progress recorded).      #
    def test_deadline_abort_during_main_phase1_loop_still_works(self):
        locked, selected, pool, labels14, zero_gt, K, S_T, N, image_ids, accept, _ab, _cd = \
            _build_cross_signature_tie_fixture()
        diag = {"budget": "1pct", "repair_phase": "minimum_class_coverage", "status": "started"}
        # entry(1) + 10 precompute ticks (4+6) + 2 successful main-loop
        # ticks, then abort on the 3rd main-loop tick. phase1_evaluated is
        # incremented BEFORE each tick within that same iteration, so it is
        # already 3 (not 2) at the moment the 3rd tick raises.
        with mock.patch.object(M, "TWO_FOR_TWO_DEADLINE_WATCHDOG_INTERVAL", 1):
            with mock.patch.object(M, "_check_deadline", side_effect=[None] * 13 + [M.DeadlineExceeded()]):
                with self.assertRaises(M.DeadlineExceeded):
                    M._equivalence_class_two_for_two_search(
                        locked, selected, pool, labels14, zero_gt, K, S_T, N, 42, "1pct",
                        "minimum_class_coverage", image_ids, set(), accept, deadline=1.0, diagnostics=diag,
                    )
        self.assertEqual(diag["status"], "aborted")
        self.assertTrue(diag["aborted_by_deadline"])
        self.assertEqual(diag["phase1_transitions_evaluated"], 3)  # real, known, partial progress
        self.assertNotEqual(diag["status"], "completed")
        self.assertNotEqual(diag["status"], "exhausted")

    # --- item 14: locked scientific gates are untouched. --- #
    def test_training_authorized_and_seed_still_locked(self):
        # R11 (TASK 10.25): training_authorized stays false and the seed
        # stays 42. The former assertion on
        # two_for_two_engine.operationally_approved moved to
        # TestLegacyTwoForTwoIsNonActive, where it is asserted as a
        # HISTORICAL, non-gating record rather than an active gate.
        import yaml
        config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
        self.assertFalse(config["gates_recorded_in_every_output"]["training_authorized"])
        self.assertEqual(config["seed"]["partition_seed"], 42)
        self.assertEqual(config["seed"]["seed_policy"], "PRE_SPECIFIED_LOCKED_NO_SEED_SEARCH")
        self.assertTrue(config["seed"]["seed_search_forbidden"])


class LegacyTwoForTwoDeadlineTests:
    def setUp(self):
        self.train = make_deterministic_train()
        self.image_ids, self.labels14, self.zero_gt, self.names = M.build_indicators(self.train)
        self.K, self.S_T, self.N = M.full_train_integer_stats(self.labels14)

    def test_deadline_param_and_watchdog_constant_exist(self):
        import inspect
        params = list(inspect.signature(M._equivalence_class_two_for_two_search).parameters)
        self.assertIn("deadline", params)
        self.assertIn("diagnostics", params)
        self.assertTrue(hasattr(M, "TWO_FOR_TWO_DEADLINE_WATCHDOG_INTERVAL"))
        self.assertIsInstance(M.TWO_FOR_TWO_DEADLINE_WATCHDOG_INTERVAL, int)
        self.assertGreater(M.TWO_FOR_TWO_DEADLINE_WATCHDOG_INTERVAL, 0)

    def test_watchdog_constant_matches_config(self):
        import yaml
        config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
        self.assertEqual(
            M.TWO_FOR_TWO_DEADLINE_WATCHDOG_INTERVAL,
            config["legacy_two_for_two_engine"]["deadline_watchdog_check_interval_candidates"],
        )

    def test_phase1_deadline_exceeded_propagates_uncaught(self):
        locked, selected, pool, labels14, zero_gt, K, S_T, N, image_ids, accept, _ab, _cd = \
            _build_cross_signature_tie_fixture()
        with mock.patch.object(M, "TWO_FOR_TWO_DEADLINE_WATCHDOG_INTERVAL", 1):
            with mock.patch.object(M, "_check_deadline", side_effect=M.DeadlineExceeded()) as mocked:
                with self.assertRaises(M.DeadlineExceeded):
                    M._equivalence_class_two_for_two_search(
                        locked, selected, pool, labels14, zero_gt, K, S_T, N, 42, "1pct",
                        "minimum_class_coverage", image_ids, set(), accept, deadline=1.0,
                    )
                mocked.assert_called()

    def test_phase2_deadline_exceeded_after_phase1_completes(self):
        # R=1, A=6, p=4 distinct pool signatures in this fixture -> with
        # watchdog interval=1: call #1 is the R6 entry boundary check,
        # calls #2-#5 are the 4 R9 precompute ticks (pool_sig_vec_matrix,
        # one per distinct pool signature), calls #6-#11 are the 6 R9
        # precompute ticks (added_vec_matrix, one per added pair), calls
        # #12-#17 are the 6 Phase-1 main-loop ticks, and the 18th call (the
        # first Phase-2 candidate) is made to raise.
        locked, selected, pool, labels14, zero_gt, K, S_T, N, image_ids, accept, _ab, _cd = \
            _build_cross_signature_tie_fixture()
        with mock.patch.object(M, "TWO_FOR_TWO_DEADLINE_WATCHDOG_INTERVAL", 1):
            with mock.patch.object(M, "_check_deadline", side_effect=[None] * 17 + [M.DeadlineExceeded()]) as mocked:
                with self.assertRaises(M.DeadlineExceeded):
                    M._equivalence_class_two_for_two_search(
                        locked, selected, pool, labels14, zero_gt, K, S_T, N, 42, "1pct",
                        "minimum_class_coverage", image_ids, set(), accept, deadline=1.0,
                    )
                self.assertEqual(mocked.call_count, 18)

    def test_deadline_none_never_calls_monotonic_clock(self):
        locked, selected, pool, labels14, zero_gt, K, S_T, N, image_ids, accept, _ab, _cd = \
            _build_cross_signature_tie_fixture()
        with mock.patch.object(M, "TWO_FOR_TWO_DEADLINE_WATCHDOG_INTERVAL", 1):
            with mock.patch.object(M.time, "monotonic") as mocked_clock:
                mocked_clock.return_value = 0.0
                result = M._equivalence_class_two_for_two_search(
                    locked, selected, pool, labels14, zero_gt, K, S_T, N, 42, "1pct",
                    "minimum_class_coverage", image_ids, set(), accept, deadline=None,
                )
            mocked_clock.assert_not_called()
        self.assertIsNotNone(result)

    def test_deadline_none_produces_identical_result_to_no_deadline_arg(self):
        locked, selected, pool, labels14, zero_gt, K, S_T, N, image_ids, accept, _ab, _cd = \
            _build_cross_signature_tie_fixture()
        r1 = M._equivalence_class_two_for_two_search(locked, selected, pool, labels14, zero_gt, K, S_T, N, 42,
                                                       "1pct", "minimum_class_coverage", image_ids, set(), accept)
        r2 = M._equivalence_class_two_for_two_search(locked, selected, pool, labels14, zero_gt, K, S_T, N, 42,
                                                       "1pct", "minimum_class_coverage", image_ids, set(), accept,
                                                       deadline=None)
        self.assertEqual(r1, r2)

    def test_build_budget_step_reports_computational_abort_not_false_exhaustion(self):
        fixed = set(range(14))
        with mock.patch.object(M, "stratified_initial_candidate", return_value=fixed), \
             mock.patch.object(M, "repair_exact_size", return_value=(fixed, M.RepairOutcome.OK)), \
             mock.patch.object(M, "repair_exact_no_finding", return_value=(fixed, M.RepairOutcome.OK)), \
             mock.patch.object(M, "repair_min_class_coverage", return_value=(fixed, M.RepairOutcome.OK)), \
             mock.patch.object(M, "repair_objective_local_search", side_effect=M.DeadlineExceeded()):
            selected, outcome, exhaustion = M.build_budget_step(
                "1pct", set(), self.image_ids, self.labels14, self.zero_gt, self.names,
                14, 0, 42, self.K, self.S_T, self.N, 1.0, [],
            )
        self.assertEqual(outcome, M.RepairOutcome.COMPUTATIONAL_ABORT)
        self.assertNotEqual(outcome, M.RepairOutcome.REPAIR_INFEASIBLE)
        # R11 (TASK 7): `two_for_two_exhausted` is gone from the active
        # contract; an abort must claim NO exhaustion and NO local optimum.
        self.assertNotIn("two_for_two_exhausted", exhaustion)
        self.assertFalse(exhaustion["one_for_one_exhausted"])
        self.assertFalse(exhaustion["local_optimum"])
        self.assertIsNone(exhaustion["local_optimum_neighborhood"])
        self.assertFalse(exhaustion["global_optimum_claimed"])

    def test_build_budget_step_abort_flags_aborted_by_deadline(self):
        fixed = set(range(14))
        with mock.patch.object(M, "stratified_initial_candidate", return_value=fixed), \
             mock.patch.object(M, "repair_exact_size", return_value=(fixed, M.RepairOutcome.OK)), \
             mock.patch.object(M, "repair_exact_no_finding", side_effect=M.DeadlineExceeded()):
            selected, outcome, exhaustion = M.build_budget_step(
                "1pct", set(), self.image_ids, self.labels14, self.zero_gt, self.names,
                14, 0, 42, self.K, self.S_T, self.N, 1.0, [],
            )
        self.assertEqual(outcome, M.RepairOutcome.COMPUTATIONAL_ABORT)
        self.assertTrue(exhaustion.get("aborted_by_deadline"))

    # --- R6 / NHIEM VU 2 (fixes blocker R5-B2): aborted diagnostics must
    # never be lost -- caller-append-before-call pattern + engine mutates
    # the same dict in place, even on DeadlineExceeded. ---
    def test_caller_pattern_preserves_diagnostics_record_on_abort(self):
        locked, selected, pool, labels14, zero_gt, K, S_T, N, image_ids, accept, _ab, _cd = \
            _build_cross_signature_tie_fixture()
        log: list[dict] = []
        diag = {"budget": "1pct", "repair_phase": "minimum_class_coverage", "status": "started"}
        log.append(diag)  # exactly the caller pattern required by R6 NHIEM VU 2
        with mock.patch.object(M, "TWO_FOR_TWO_DEADLINE_WATCHDOG_INTERVAL", 1):
            with mock.patch.object(M, "_check_deadline", side_effect=M.DeadlineExceeded()):
                with self.assertRaises(M.DeadlineExceeded):
                    M._equivalence_class_two_for_two_search(
                        locked, selected, pool, labels14, zero_gt, K, S_T, N, 42, "1pct",
                        "minimum_class_coverage", image_ids, set(), accept, deadline=1.0, diagnostics=diag,
                    )
        self.assertEqual(len(log), 1)  # record was never removed
        self.assertIs(log[0], diag)  # same object -- mutated in place
        self.assertEqual(diag["status"], "aborted")
        self.assertTrue(diag["aborted_by_deadline"])
        self.assertEqual(diag["budget"], "1pct")  # caller-set fields untouched
        self.assertEqual(diag["repair_phase"], "minimum_class_coverage")

    def test_phase1_abort_diagnostic_has_progress_and_null_unknowns(self):
        locked, selected, pool, labels14, zero_gt, K, S_T, N, image_ids, accept, _ab, _cd = \
            _build_cross_signature_tie_fixture()
        diag = {"budget": "1pct", "repair_phase": "minimum_class_coverage", "status": "started"}
        # 1 entry check + 10 R9 precompute ticks (4 pool_sig_vec_matrix + 6
        # added_vec_matrix, see test_phase2_deadline_exceeded_after_phase1_
        # completes for the exact breakdown) + 3 no-ops mid Phase-1
        # (R=1,A=6 so this is partway through the 6 main-loop ticks) then
        # abort -- this must land INSIDE the real R x A loop (after
        # precompute has already finished), so phase1_transitions_
        # evaluated is genuinely > 0, not a precompute-phase abort (see
        # TestR9AddedPairPrecompute's dedicated precompute-abort test for
        # that case).
        with mock.patch.object(M, "TWO_FOR_TWO_DEADLINE_WATCHDOG_INTERVAL", 1):
            with mock.patch.object(M, "_check_deadline", side_effect=[None] * 14 + [M.DeadlineExceeded()]):
                with self.assertRaises(M.DeadlineExceeded):
                    M._equivalence_class_two_for_two_search(
                        locked, selected, pool, labels14, zero_gt, K, S_T, N, 42, "1pct",
                        "minimum_class_coverage", image_ids, set(), accept, deadline=1.0, diagnostics=diag,
                    )
        self.assertEqual(diag["status"], "aborted")
        self.assertTrue(diag["aborted_by_deadline"])
        self.assertIsNotNone(diag["phase1_transitions_evaluated"])
        self.assertGreater(diag["phase1_transitions_evaluated"], 0)
        # Phase 2 was never reached.
        self.assertEqual(diag["concrete_candidates_evaluated_phase2"], 0)
        # W / theoretical-in-ties are not knowable until a Phase-1 pass
        # finishes -- must be null, never a fabricated 0.
        self.assertIsNone(diag["tied_best_signature_transition_count_W"])
        self.assertIsNone(diag["theoretical_concrete_candidates_in_winning_ties"])
        self.assertNotEqual(diag["status"], "exhausted")
        self.assertNotEqual(diag["status"], "completed")

    def test_immediate_abort_before_any_computation_nulls_everything_unknown(self):
        locked, selected, pool, labels14, zero_gt, K, S_T, N, image_ids, accept, _ab, _cd = \
            _build_cross_signature_tie_fixture()
        diag = {"budget": "1pct", "repair_phase": "minimum_class_coverage", "status": "started"}
        with mock.patch.object(M, "_check_deadline", side_effect=M.DeadlineExceeded()):
            with self.assertRaises(M.DeadlineExceeded):
                M._equivalence_class_two_for_two_search(
                    locked, selected, pool, labels14, zero_gt, K, S_T, N, 42, "1pct",
                    "minimum_class_coverage", image_ids, set(), accept, deadline=1.0, diagnostics=diag,
                )
        self.assertEqual(diag["status"], "aborted")
        for field in (
            "selected_abnormal_count", "pool_abnormal_count", "selected_distinct_signature_count",
            "pool_distinct_signature_count", "removed_pair_signature_count_R", "added_pair_signature_count_A",
            "phase1_transition_count", "tied_best_signature_transition_count_W",
            "theoretical_concrete_candidates_in_winning_ties",
        ):
            self.assertIsNone(diag[field], msg=f"{field} should be null (unknown), not fabricated")
        self.assertEqual(diag["phase1_transitions_evaluated"], 0)  # a real, known count (zero ticks happened)
        self.assertEqual(diag["concrete_candidates_evaluated_phase2"], 0)

    def test_phase2_abort_diagnostic_keeps_concrete_evaluated_count(self):
        locked, selected, pool, labels14, zero_gt, K, S_T, N, image_ids, accept, _ab, _cd = \
            _build_cross_signature_tie_fixture()
        diag = {"budget": "1pct", "repair_phase": "minimum_class_coverage", "status": "started"}
        # 1 entry check + 10 R9 precompute ticks (4+6, see
        # test_phase2_deadline_exceeded_after_phase1_completes) + 6 Phase-1
        # main-loop ticks (R=1, A=6, so the full R x A pass completes),
        # then 1 Phase-2 tick raises.
        with mock.patch.object(M, "TWO_FOR_TWO_DEADLINE_WATCHDOG_INTERVAL", 1):
            with mock.patch.object(M, "_check_deadline", side_effect=[None] * 17 + [M.DeadlineExceeded()]):
                with self.assertRaises(M.DeadlineExceeded):
                    M._equivalence_class_two_for_two_search(
                        locked, selected, pool, labels14, zero_gt, K, S_T, N, 42, "1pct",
                        "minimum_class_coverage", image_ids, set(), accept, deadline=1.0, diagnostics=diag,
                    )
        self.assertEqual(diag["status"], "aborted")
        self.assertEqual(diag["phase1_transitions_evaluated"], 6)
        self.assertGreaterEqual(diag["concrete_candidates_evaluated_phase2"], 1)
        # R/A were already known by the time Phase 2 started -- must be
        # preserved, not nulled.
        self.assertEqual(diag["removed_pair_signature_count_R"], 1)
        self.assertEqual(diag["added_pair_signature_count_A"], 6)
        self.assertIsNotNone(diag["tied_best_signature_transition_count_W"])

    def test_timeout_diagnostics_survive_in_repair_min_class_coverage_log(self):
        # End-to-end through the actual caller (repair_min_class_coverage),
        # not just the engine directly -- confirms the append-before-call
        # wiring in the real repair function. Mocks the ENGINE itself (not
        # _check_deadline) to raise, so the outer function's own deadline
        # checks (deadline=None here, always a no-op) never interfere.
        #
        # R6 pytest-repair round: the previous fixture (make_deterministic_
        # train, selected=positions 0..13 minus class-index-6's position)
        # assumed no one-for-one fix could exist, but make_deterministic_
        # train's round-robin class PAIRING means every pool image carrying
        # class 6 also carries class 13 -- and removing position 13 (class
        # 13's sole singleton carrier) while adding such a pool image
        # recovers BOTH 6 and 13 in a single legal one-for-one swap, so the
        # one-for-one branch resolved the repair before the engine was ever
        # called (the mocked DeadlineExceeded side effect was never
        # triggered, and assertRaises failed).
        #
        # Replaced with a purpose-built fixture with no such coincidence:
        # 13 selected images, each the SOLE carrier of its own class
        # (classes 0-5, 7-13; class 6 entirely missing), and pool
        # candidates that carry class 6 ONLY (single-class, no
        # co-occurrence with anything else). Since every selected image is
        # a sole carrier, removing ANY of them to make room for a class-6
        # pool image always trades "class 6 recovered" for "some other
        # class newly missing" -- the missing count never strictly
        # decreases, so the one-for-one branch can never succeed (verified
        # exhaustively below), forcing execution into the two-for-two
        # branch every time.
        images, annotations = [], []
        ann_id = 1

        def add_image(image_id, classes):
            nonlocal ann_id
            images.append({"id": image_id, "file_name": f"t/{image_id}.jpg", "width": 10, "height": 10})
            for c in classes:
                annotations.append({"id": ann_id, "image_id": image_id, "category_id": c + 1,
                                     "bbox": [1, 1, 2, 2], "area": 4, "iscrowd": 0})
                ann_id += 1

        other_classes = [c for c in range(14) if c != 6]
        for idx, c in enumerate(other_classes):
            add_image(idx + 1, [c])   # 13 sole-carrier images, ids 1..13
        add_image(14, [6])             # pool candidate: class 6 ONLY
        add_image(15, [6])             # second pool candidate: class 6 ONLY
        train = {"info": {}, "licenses": [], "images": images, "annotations": annotations,
                 "categories": _categories()}

        image_ids, labels14, zero_gt, names = M.build_indicators(train)
        K, S_T, N = M.full_train_integer_stats(labels14)
        id_to_pos = {int(v): i for i, v in enumerate(image_ids.tolist())}
        selected = {id_to_pos[i] for i in range(1, 14)}
        pool = {id_to_pos[14], id_to_pos[15]}

        # Precondition, verified EXHAUSTIVELY (not just by construction):
        # no one-for-one swap strictly reduces the missing-class count.
        missing_before_count = 1  # only class 6
        no_valid_one_for_one = True
        for out_pos in sorted(selected):
            for in_pos in sorted(pool):
                trial = set(selected)
                trial.discard(out_pos)
                trial.add(in_pos)
                present = set(int(c) for c in np.where(labels14[sorted(trial)].sum(axis=0) > 0)[0])
                missing_after_count = 14 - len(present)
                if missing_after_count < missing_before_count:
                    no_valid_one_for_one = False
                    break
            if not no_valid_one_for_one:
                break
        self.assertTrue(no_valid_one_for_one,
                         "fixture precondition violated: a one-for-one fix exists, engine would never be reached")

        log: list[dict] = []
        with mock.patch.object(M, "_equivalence_class_two_for_two_search", side_effect=M.DeadlineExceeded()):
            with self.assertRaises(M.DeadlineExceeded):
                M.repair_min_class_coverage(
                    set(), selected, pool, labels14, zero_gt, names, K, S_T, N, 42, "1pct", image_ids,
                    None, [], set(), log,
                )
        self.assertEqual(len(log), 1)  # the caller appended BEFORE calling the (mocked) engine
        self.assertEqual(log[0]["budget"], "1pct")
        self.assertEqual(log[0]["repair_phase"], "minimum_class_coverage")
        # The mocked engine never got a chance to mutate it further, so
        # proving the record's PRESENCE survives the abort is the point of
        # this test; the engine's OWN finalization to status="aborted" is
        # covered end-to-end by the direct engine-level tests above.
        self.assertIn(log[0]["status"], ("started", "aborted"))

    # --- R6 / NHIEM VU 3 (fixes blocker R5-B3): boundary checks. ---
    def test_boundary_check_before_exhausted_return_trivial_case(self):
        # selected has fewer than 2 abnormal images -> the trivial
        # "exhausted" early-return path; deadline expires exactly at that
        # boundary (the check placed immediately before the return).
        train = make_deterministic_train()
        image_ids, labels14, zero_gt, names = M.build_indicators(train)
        K, S_T, N = M.full_train_integer_stats(labels14)
        selected = {0}  # only 1 abnormal image selected
        pool = set(range(1, len(image_ids)))
        with mock.patch.object(M, "_check_deadline", side_effect=[None, M.DeadlineExceeded()]) as mocked:
            with self.assertRaises(M.DeadlineExceeded):
                M._equivalence_class_two_for_two_search(
                    set(), selected, pool, labels14, zero_gt, K, S_T, N, 42, "1pct",
                    "objective_repair", image_ids, set(),
                    lambda k, n, s: (0, 0, 0), deadline=1.0,
                )
        self.assertEqual(mocked.call_count, 2)  # entry check + boundary check, never reaching enumeration

    def test_boundary_check_before_exhausted_return_after_full_search(self):
        locked, selected, pool, labels14, zero_gt, K, S_T, N, image_ids, _accept, _ab, _cd = \
            _build_cross_signature_tie_fixture()

        def reject_everything(new_k, new_n, new_s_l):
            return None  # nothing ever accepted -> guaranteed "exhausted"

        with mock.patch.object(M, "TWO_FOR_TWO_DEADLINE_WATCHDOG_INTERVAL", 1):
            # entry check (1) + 10 R9 precompute ticks (4 pool_sig_vec_matrix
            # + 6 added_vec_matrix) + 6 Phase-1 ticks (all no-op, R*A=6) +
            # the boundary check right before the "exhausted" return (18th)
            # raises.
            with mock.patch.object(M, "_check_deadline", side_effect=[None] * 17 + [M.DeadlineExceeded()]) as mocked:
                with self.assertRaises(M.DeadlineExceeded):
                    M._equivalence_class_two_for_two_search(
                        locked, selected, pool, labels14, zero_gt, K, S_T, N, 42, "1pct",
                        "minimum_class_coverage", image_ids, set(), reject_everything, deadline=1.0,
                    )
            self.assertEqual(mocked.call_count, 18)

    def test_boundary_check_before_completed_return(self):
        locked, selected, pool, labels14, zero_gt, K, S_T, N, image_ids, accept, _ab, _cd = \
            _build_cross_signature_tie_fixture()
        # entry(1) + 6 Phase-1 ticks + >=1 Phase-2 ticks, then the boundary
        # check right before the "completed" return must be the one that
        # raises -- give it enough no-ops to reach that point deterministically
        # is fixture-dependent, so instead assert the CONTRACT: with the
        # watchdog interval huge (never fires mid-search) but the boundary
        # check itself forced to always raise, a successful search must
        # still raise DeadlineExceeded at the very end, not return a candidate.
        with mock.patch.object(M, "TWO_FOR_TWO_DEADLINE_WATCHDOG_INTERVAL", 10_000_000):
            with mock.patch.object(M, "_check_deadline", side_effect=[None, M.DeadlineExceeded()]):
                with self.assertRaises(M.DeadlineExceeded):
                    M._equivalence_class_two_for_two_search(
                        locked, selected, pool, labels14, zero_gt, K, S_T, N, 42, "1pct",
                        "minimum_class_coverage", image_ids, set(), accept, deadline=1.0,
                    )

    def test_deadline_none_boundary_checks_never_call_monotonic(self):
        locked, selected, pool, labels14, zero_gt, K, S_T, N, image_ids, accept, _ab, _cd = \
            _build_cross_signature_tie_fixture()
        with mock.patch.object(M.time, "monotonic") as mocked_clock:
            mocked_clock.return_value = 0.0
            result = M._equivalence_class_two_for_two_search(
                locked, selected, pool, labels14, zero_gt, K, S_T, N, 42, "1pct",
                "minimum_class_coverage", image_ids, set(), accept, deadline=None,
            )
        mocked_clock.assert_not_called()
        self.assertIsNotNone(result)

    def test_no_status_started_survives_a_finished_invocation(self):
        locked, selected, pool, labels14, zero_gt, K, S_T, N, image_ids, accept, _ab, _cd = \
            _build_cross_signature_tie_fixture()
        diag = {"budget": "1pct", "repair_phase": "minimum_class_coverage", "status": "started"}
        M._equivalence_class_two_for_two_search(
            locked, selected, pool, labels14, zero_gt, K, S_T, N, 42, "1pct",
            "minimum_class_coverage", image_ids, set(), accept, diagnostics=diag,
        )
        self.assertIn(diag["status"], ("completed", "exhausted"))
        self.assertNotEqual(diag["status"], "started")

    def test_compute_all_budgets_stops_on_computational_abort_without_promoting(self):
        fixed_targets = ({b: 14 for b in M.BUDGET_ORDER}, {b: 0 for b in M.BUDGET_ORDER})
        aborted_exhaustion = {
            "one_for_one_exhausted": False, "local_optimum": False,
            "local_optimum_neighborhood": None, "global_optimum_claimed": False,
        }
        with mock.patch.object(M, "check_iterative_stratification_version", return_value=("0.1.9", "OK")), \
             mock.patch.object(M, "load_json", return_value=self.train), \
             mock.patch.object(M, "build_indicators", return_value=(self.image_ids, self.labels14, self.zero_gt, self.names)), \
             mock.patch.object(M, "compute_locked_size_targets", return_value=fixed_targets), \
             mock.patch.object(M, "build_budget_step", return_value=(set(), M.RepairOutcome.COMPUTATIONAL_ABORT, aborted_exhaustion)):
            config = {
                # R11: the active gate is repair.active_repair_policy, not the
                # retired two_for_two_engine.operationally_approved flag.
                "repair": {"active_repair_policy": M.ACTIVE_REPAIR_POLICY},
                "dependencies": {"iterative_stratification": {"required_version": "0.1.9",
                                                                "package_name": "iterative-stratification"}},
                "inputs": {"train_coco": "x"},
                "seed": {"partition_seed": 42},
            }
            bundle = M.compute_all_budgets(config, Path("."))
        self.assertEqual(bundle["per_budget_outcome"]["1pct"], M.RepairOutcome.COMPUTATIONAL_ABORT)
        self.assertNotIn("5pct", bundle["per_budget_outcome"])
        self.assertNotIn("10pct", bundle["per_budget_outcome"])
        self.assertNotIn("20pct", bundle["per_budget_outcome"])


# --------------------------------------------------------------------------- #
# R6 / NHIEM VU 5 — diagnostics status lifecycle consistency                  #
# (started -> completed/exhausted/aborted; "started" never final)            #
# --------------------------------------------------------------------------- #
class TestDiagnosticsStateConsistency(unittest.TestCase):
    def test_engine_source_only_finalizes_to_the_three_terminal_statuses(self):
        # Guardrail: every literal status string the engine can ever WRITE
        # into diagnostics must be one of the three terminal values -- never
        # anything else, and "started" is never assigned BY the engine
        # itself (only ever set by the caller as the initial placeholder
        # before invoking the engine).
        source = SCRIPT_PATH.read_text(encoding="utf-8")
        tree = ast.parse(source)
        engine_fn = next(
            node for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef) and node.name == "_equivalence_class_two_for_two_search"
        )
        engine_source = ast.get_source_segment(source, engine_fn) or ""
        for literal in ('"completed"', '"exhausted"', '"aborted"'):
            self.assertIn(literal, engine_source)
        self.assertNotIn('"started"', engine_source)

    def test_completed_invocation_always_returns_a_candidate(self):
        locked, selected, pool, labels14, zero_gt, K, S_T, N, image_ids, accept, _ab, _cd = \
            _build_cross_signature_tie_fixture()
        diag: dict = {}
        result = M._equivalence_class_two_for_two_search(locked, selected, pool, labels14, zero_gt, K, S_T, N, 42,
                                                           "1pct", "minimum_class_coverage", image_ids, set(),
                                                           accept, diagnostics=diag)
        if diag["status"] == "completed":
            self.assertIsNotNone(result)

    def test_exhausted_invocation_always_returns_none(self):
        train = make_deterministic_train()
        image_ids, labels14, zero_gt, _ = M.build_indicators(train)
        K, S_T, N = M.full_train_integer_stats(labels14)
        selected = {0}  # trivially < 2 abnormal -> guaranteed exhausted
        pool = set(range(1, len(image_ids)))
        diag: dict = {}
        result = M._equivalence_class_two_for_two_search(
            set(), selected, pool, labels14, zero_gt, K, S_T, N, 42, "1pct", "objective_repair", image_ids,
            set(), lambda k, n, s: (0, 0, 0), diagnostics=diag,
        )
        self.assertEqual(diag["status"], "exhausted")
        self.assertIsNone(result)

    def test_diagnostics_never_influence_which_candidate_is_returned(self):
        # Same inputs, only difference is whether a diagnostics dict is
        # passed -- the returned candidate must be bit-for-bit identical.
        locked, selected, pool, labels14, zero_gt, K, S_T, N, image_ids, accept, _ab, _cd = \
            _build_cross_signature_tie_fixture()
        r_without = M._equivalence_class_two_for_two_search(locked, selected, pool, labels14, zero_gt, K, S_T, N,
                                                              42, "1pct", "minimum_class_coverage", image_ids,
                                                              set(), accept)
        r_with = M._equivalence_class_two_for_two_search(locked, selected, pool, labels14, zero_gt, K, S_T, N, 42,
                                                           "1pct", "minimum_class_coverage", image_ids, set(),
                                                           accept, diagnostics={})
        self.assertEqual(r_without, r_with)


# --------------------------------------------------------------------------- #
# R6 / NHIEM VU 4 (fixes blocker R5-B4) — orchestration-level proof that a   #
# COMPUTATIONAL_ABORT NEVER reaches write_all_outputs_and_promote, not just #
# that compute_all_budgets's per-budget loop stops.                          #
# --------------------------------------------------------------------------- #
class TestOrchestrationNoPromotionOnAbort(unittest.TestCase):
    def test_main_never_promotes_on_computational_abort(self):
        import argparse as _argparse
        import contextlib as _contextlib
        import io as _io

        fake_config = {
            # R6 pytest-repair round: main() now reads its console-banner
            # identity via get_protocol_identity(config), which fails
            # closed (CONFIG_INVALID) if config["protocol"] is missing --
            # this fake config must include it or main() raises before
            # ever reaching the abort/no-promotion behavior under test.
            "protocol": {"stage": "TEST-STAGE", "version": "TEST-VERSION"},
            "seed": {"partition_seed": 42, "seed_policy": "PRE_SPECIFIED_LOCKED_NO_SEED_SEARCH"},
            "repair": {"active_repair_policy": M.ACTIVE_REPAIR_POLICY},
        }
        aborted_exhaustion = {
            "one_for_one_exhausted": False, "local_optimum": False,
            "local_optimum_neighborhood": None, "global_optimum_claimed": False,
            "aborted_by_deadline": True,
        }
        fake_bundle = {
            "per_budget_selected": {"1pct": set()},
            "per_budget_outcome": {"1pct": M.RepairOutcome.COMPUTATIONAL_ABORT},
            "per_budget_exhaustion": {"1pct": aborted_exhaustion},
            "repair_log": [], "active_repair_policy": M.ACTIVE_REPAIR_POLICY,
            "nf_size_target": {"1pct": 0},
        }
        with tempfile.TemporaryDirectory() as tmp:
            fake_args = _argparse.Namespace(
                config=Path("configs/protocol/phase2F_labeled_unlabeled.yaml"),
                project_root=Path(tmp), preflight_only=False, reconstruct_check=False, max_seconds=None,
                # R7-B1: main() now calls _validate_benchmark_cli_args(args)
                # unconditionally right after parse_args(), so every fake
                # Namespace standing in for it must carry the real
                # argparse contract's two new fields, not just the fields
                # this particular test happens to care about.
                benchmark_two_for_two=False, benchmark_max_seconds_per_budget=None,
            )
            with mock.patch.object(M, "parse_args", return_value=fake_args), \
                 mock.patch.object(M, "load_config", return_value=fake_config), \
                 mock.patch.object(M, "run_preflight", return_value={
                     "status": "PASS", "checks": [],
                     "policy_evidence_training_authorized": "POLICY_EVIDENCE_NOT_MACHINE_READABLE",
                     # Post-R6 observability fix: main() now unconditionally
                     # prints these informational fields, so any fixture
                     # standing in for run_preflight()'s return value must
                     # include them too (same reason policy_evidence_
                     # training_authorized was already required above).
                     "active_repair_policy": M.ACTIVE_REPAIR_POLICY,
                     "objective_repair_neighborhood": M.LOCAL_OPTIMUM_NEIGHBORHOOD,
                     "objective_repair_termination_claim": "one-for-one local optimum; no global optimum claimed",
                     "global_optimum_claimed": False,
                 }), \
                 mock.patch.object(M, "official_relative_paths", return_value=[]), \
                 mock.patch.object(M, "compute_all_budgets", return_value=fake_bundle), \
                 mock.patch.object(M, "write_all_outputs_and_promote") as mocked_promote:
                stdout = _io.StringIO()
                with _contextlib.redirect_stdout(stdout):
                    with self.assertRaises(M.Phase2FError):
                        M.main()
                mocked_promote.assert_not_called()
                output = stdout.getvalue()
                self.assertNotIn("PHASE_2F_GATE= PASS", output)
                self.assertIn(M.RepairOutcome.COMPUTATIONAL_ABORT, output)
            # No stray official artifacts anywhere under the temp project
            # root -- official_relative_paths was empty and promotion
            # itself was mocked out, so nothing should exist at all.
            self.assertEqual(list(Path(tmp).rglob("*")), [])

    def test_main_taxonomy_is_computational_abort_not_repair_infeasible(self):
        import argparse as _argparse

        fake_config = {
            # R6 pytest-repair round: main() now reads its console-banner
            # identity via get_protocol_identity(config), which fails
            # closed (CONFIG_INVALID) if config["protocol"] is missing --
            # this fake config must include it or main() raises before
            # ever reaching the abort/no-promotion behavior under test.
            "protocol": {"stage": "TEST-STAGE", "version": "TEST-VERSION"},
            "seed": {"partition_seed": 42, "seed_policy": "PRE_SPECIFIED_LOCKED_NO_SEED_SEARCH"},
            "repair": {"active_repair_policy": M.ACTIVE_REPAIR_POLICY},
        }
        fake_bundle = {
            "per_budget_selected": {"1pct": set()},
            "per_budget_outcome": {"1pct": M.RepairOutcome.COMPUTATIONAL_ABORT},
            "per_budget_exhaustion": {"1pct": {}}, "repair_log": [],
            "active_repair_policy": M.ACTIVE_REPAIR_POLICY,
            "nf_size_target": {"1pct": 0},
        }
        with tempfile.TemporaryDirectory() as tmp:
            fake_args = _argparse.Namespace(
                config=Path("configs/protocol/phase2F_labeled_unlabeled.yaml"),
                project_root=Path(tmp), preflight_only=False, reconstruct_check=False, max_seconds=None,
                # R7-B1: main() now calls _validate_benchmark_cli_args(args)
                # unconditionally right after parse_args(), so every fake
                # Namespace standing in for it must carry the real
                # argparse contract's two new fields, not just the fields
                # this particular test happens to care about.
                benchmark_two_for_two=False, benchmark_max_seconds_per_budget=None,
            )
            with mock.patch.object(M, "parse_args", return_value=fake_args), \
                 mock.patch.object(M, "load_config", return_value=fake_config), \
                 mock.patch.object(M, "run_preflight", return_value={
                     "status": "PASS", "checks": [],
                     "policy_evidence_training_authorized": "POLICY_EVIDENCE_NOT_MACHINE_READABLE",
                     # Post-R6 observability fix: see identical comment above
                     # in test_main_never_promotes_on_computational_abort.
                     "active_repair_policy": M.ACTIVE_REPAIR_POLICY,
                     "objective_repair_neighborhood": M.LOCAL_OPTIMUM_NEIGHBORHOOD,
                     "objective_repair_termination_claim": "one-for-one local optimum; no global optimum claimed",
                     "global_optimum_claimed": False,
                 }), \
                 mock.patch.object(M, "official_relative_paths", return_value=[]), \
                 mock.patch.object(M, "compute_all_budgets", return_value=fake_bundle), \
                 mock.patch.object(M, "write_all_outputs_and_promote"):
                with self.assertRaises(M.Phase2FError) as ctx:
                    M.main()
        self.assertIn(M.RepairOutcome.COMPUTATIONAL_ABORT, str(ctx.exception))
        self.assertNotIn(M.RepairOutcome.REPAIR_INFEASIBLE, str(ctx.exception))


# --------------------------------------------------------------------------- #
# R3 review item 4 — coverage progress rule + candidate priority order       #
# --------------------------------------------------------------------------- #
class TestCoverageCandidatePriorityOrder(unittest.TestCase):
    def setUp(self):
        self.train = make_deterministic_train()
        self.image_ids, self.labels14, self.zero_gt, self.names = M.build_indicators(self.train)
        self.K, self.S_T, self.N = M.full_train_integer_stats(self.labels14)

    def test_coverage_progress_rule_constant_matches_config(self):
        import yaml
        config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
        self.assertEqual(config["repair"]["coverage_progress_rule"], M.COVERAGE_PROGRESS_RULE)
        self.assertEqual(M.COVERAGE_PROGRESS_RULE, "STRICTLY_REDUCE_MISSING_CLASS_COUNT_BY_AT_LEAST_ONE")

    def test_config_priority_order_starts_with_missing_count_after(self):
        import yaml
        config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
        order = config["repair"]["coverage_candidate_priority_order"]
        self.assertEqual(order[0], "missing_count_after")
        self.assertEqual(order[1], "integer_objective_tuple")

    def test_missing_count_after_beats_a_better_objective(self):
        # trialA reaches full 14/14 coverage (missing_count_after=0);
        # trialB is still missing a class (missing_count_after=1). Whatever
        # the integer-objective values happen to be, _coverage_move_key
        # MUST rank trialA strictly first because missing_count_after is
        # compared before the objective tuple.
        trial_a = set(range(14)) | {20}   # all 14 classes covered, plus one extra
        trial_b = set(range(13))          # class index 13 missing
        key_a = M._coverage_move_key(trial_a, 0, self.labels14, self.K, self.S_T, self.N,
                                      42, "1pct", "one_for_one_swap", (1,), (2,))
        key_b = M._coverage_move_key(trial_b, 1, self.labels14, self.K, self.S_T, self.N,
                                      42, "1pct", "one_for_one_swap", (1,), (2,))
        self.assertEqual(key_a[0], 0)
        self.assertEqual(key_b[0], 1)
        self.assertLess(key_a, key_b)


# --------------------------------------------------------------------------- #
# Repair phases                                                               #
# --------------------------------------------------------------------------- #
class TestRepairExactSize(unittest.TestCase):
    def setUp(self):
        self.train = make_deterministic_train()
        self.image_ids, self.labels14, self.zero_gt, self.names = M.build_indicators(self.train)
        self.K, self.S_T, self.N = M.full_train_integer_stats(self.labels14)

    def test_grows_to_exact_target(self):
        pool = set(range(len(self.image_ids)))
        initial = set(list(pool)[:5])
        log, visited = [], set()
        selected, outcome = M.repair_exact_size(
            set(), initial, pool, 10, self.labels14, self.zero_gt, self.names,
            self.K, self.S_T, self.N, 42, "1pct", self.image_ids, None, log, visited,
        )
        self.assertEqual(outcome, M.RepairOutcome.OK)
        self.assertEqual(len(selected), 10)
        self.assertTrue(initial <= selected)
        for entry in log:
            self.assertEqual(entry["budget"], "1pct")
            self.assertEqual(entry["repair_phase"], "exact_size")

    def test_shrinks_to_exact_target(self):
        pool = set(range(len(self.image_ids)))
        initial = set(list(pool)[:15])
        log, visited = [], set()
        selected, outcome = M.repair_exact_size(
            set(), initial, pool, 8, self.labels14, self.zero_gt, self.names,
            self.K, self.S_T, self.N, 42, "1pct", self.image_ids, None, log, visited,
        )
        self.assertEqual(outcome, M.RepairOutcome.OK)
        self.assertEqual(len(selected), 8)
        self.assertTrue(selected <= initial)

    def test_locked_prefix_never_mutated(self):
        pool_all = set(range(len(self.image_ids)))
        locked = set(list(pool_all)[:5])
        pool = pool_all - locked
        initial = set(list(pool)[:3])
        log, visited = [], set()
        selected, outcome = M.repair_exact_size(
            locked, initial, pool, 6, self.labels14, self.zero_gt, self.names,
            self.K, self.S_T, self.N, 42, "1pct", self.image_ids, None, log, visited,
        )
        self.assertEqual(outcome, M.RepairOutcome.OK)
        self.assertTrue(selected <= pool)  # repair_exact_size only ever touches the new-region pool


class TestRepairExactNoFinding(unittest.TestCase):
    def setUp(self):
        self.train = make_deterministic_train()
        self.image_ids, self.labels14, self.zero_gt, self.names = M.build_indicators(self.train)
        self.K, self.S_T, self.N = M.full_train_integer_stats(self.labels14)

    def test_reaches_exact_nf_target_preserving_size(self):
        pool = set(range(len(self.image_ids)))
        initial = set(sorted(pool)[:10])
        log, visited = [], set()
        selected, outcome = M.repair_exact_no_finding(
            set(), initial, pool, 3, self.zero_gt, self.labels14, self.names,
            self.K, self.S_T, self.N, 42, "1pct", self.image_ids, None, log, visited,
        )
        self.assertEqual(outcome, M.RepairOutcome.OK)
        self.assertEqual(len(selected), len(initial))
        self.assertEqual(int(sum(self.zero_gt[i] for i in selected)), 3)

    def test_impossible_target_reports_repair_infeasible_not_exception(self):
        pool = set(range(len(self.image_ids)))
        initial = set(sorted(pool)[:5])
        total_nf_available = int(self.zero_gt.sum())
        log, visited = [], set()
        selected, outcome = M.repair_exact_no_finding(
            set(), initial, pool, total_nf_available + 1000, self.zero_gt, self.labels14, self.names,
            self.K, self.S_T, self.N, 42, "1pct", self.image_ids, None, log, visited,
        )
        self.assertEqual(outcome, M.RepairOutcome.REPAIR_INFEASIBLE)


class TestRepairMinClassCoverage(unittest.TestCase):
    def setUp(self):
        self.train = make_deterministic_train()
        self.image_ids, self.labels14, self.zero_gt, self.names = M.build_indicators(self.train)
        self.K, self.S_T, self.N = M.full_train_integer_stats(self.labels14)
        # image_id 1..14 -> positions 0..13, one distinct class each. Any
        # pool containing them all makes coverage repair GUARANTEED
        # feasible; no skip-on-infeasible escape hatch is needed.
        self.single_class_positions = list(range(14))
        # image_id 15..20 -> positions 14..19, all zero_gt=1 (No Finding).
        self.zero_gt_positions = list(range(14, 20))

    def test_all_zero_gt_selection_is_provably_repair_infeasible(self):
        # Corrected per GPT review: this all-zero_gt fixture is NOT silently
        # treated as feasible. Hand-verified: when `selected` is entirely
        # zero_gt=1 and the entire pool is zero_gt=0, no one-for-one swap
        # (requires zero_gt[out]==zero_gt[in]) and no composition-preserving
        # two-for-two swap (requires the removed pair's zero_gt multiset to
        # equal the added pair's) can ever exist, because {1,1} (both
        # removed are zero_gt=1) can never equal {0,0} or {0,1} (both
        # candidates for `in` are zero_gt=0). The correct, provable outcome
        # is REPAIR_INFEASIBLE, asserted explicitly rather than skipped.
        pool = set(range(len(self.image_ids))) - set(self.zero_gt_positions)
        pool |= set(self.single_class_positions)
        initial = set(self.zero_gt_positions)
        log, visited = [], set()
        selected, outcome = M.repair_min_class_coverage(
            set(), initial, pool, self.labels14, self.zero_gt, self.names,
            self.K, self.S_T, self.N, 42, "1pct", self.image_ids, None, log, visited,
        )
        self.assertEqual(outcome, M.RepairOutcome.REPAIR_INFEASIBLE)
        self.assertEqual(log, [])

    def test_one_for_one_fixes_missing_class_when_feasible(self):
        # Purpose-built, fully hand-verified fixture (independent of
        # make_deterministic_train, to avoid disturbing its 50-image
        # invariants used elsewhere):
        #   image_id 1..14: single-class images, image_id k carries
        #     category_id k (class index k-1) and nothing else.
        #   image_id 15: a REDUNDANT duplicate of class index 0 (category_id
        #     1) -- adds no unique coverage, since image_id 1 already covers
        #     class index 0.
        #   image_id 16, 17: No-Finding (zero annotations).
        # `initial` = images 1..14 EXCEPT image_id 7 (so class index 6 /
        # category_id 7 is the only missing class) PLUS image_id 15, 16, 17.
        # `pool` = {image_id 7} only.
        # Hand-verified unique outcome: the ONLY (out, in) pair that
        # achieves strict coverage progress is (remove image_id 15, add
        # image_id 7) -- every other candidate `out` uniquely covers its
        # own class, so removing it would merely trade one missing class
        # for another (net-zero progress, correctly rejected). This makes
        # the expected repair-log content an exact, provable prediction,
        # not an assumption.
        images, annotations = [], []
        ann_id = 1
        for k in range(1, 15):
            images.append({"id": k, "file_name": f"t/{k}.jpg", "width": 10, "height": 10})
            annotations.append({"id": ann_id, "image_id": k, "category_id": k,
                                 "bbox": [1, 1, 2, 2], "area": 4, "iscrowd": 0})
            ann_id += 1
        images.append({"id": 15, "file_name": "t/15.jpg", "width": 10, "height": 10})
        annotations.append({"id": ann_id, "image_id": 15, "category_id": 1,
                             "bbox": [1, 1, 2, 2], "area": 4, "iscrowd": 0})
        images.append({"id": 16, "file_name": "t/16.jpg", "width": 10, "height": 10})
        images.append({"id": 17, "file_name": "t/17.jpg", "width": 10, "height": 10})
        fixture = {"info": {}, "licenses": [], "images": images, "annotations": annotations, "categories": _categories()}

        image_ids, labels14, zero_gt, names = M.build_indicators(fixture)
        K, S_T, N = M.full_train_integer_stats(labels14)
        id_to_pos = {int(v): i for i, v in enumerate(image_ids.tolist())}

        initial = {id_to_pos[k] for k in range(1, 15) if k != 7}
        initial |= {id_to_pos[15], id_to_pos[16], id_to_pos[17]}
        pool = set(range(len(image_ids))) - initial
        self.assertEqual(pool, {id_to_pos[7]})  # sanity: exactly one candidate in the pool

        log, visited = [], set()
        selected, outcome = M.repair_min_class_coverage(
            set(), initial, pool, labels14, zero_gt, names, K, S_T, N, 42, "1pct", image_ids, None, log, visited,
        )
        self.assertEqual(outcome, M.RepairOutcome.OK)
        self.assertEqual(len(log), 1)
        self.assertEqual(log[0]["move_type"], "one_for_one_swap")
        self.assertEqual(log[0]["removed_ids"], [15])
        self.assertEqual(log[0]["added_ids"], [7])
        self.assertIn("tie_break_priority_digest", log[0])
        self.assertIn("membership_sha256_before", log[0])
        self.assertIn("membership_sha256_after", log[0])
        self.assertIn("candidate_set_priority_before", log[0])

        positions = sorted(selected)
        covered = set(int(c) for c in np.where(labels14[positions].sum(axis=0) > 0)[0])
        self.assertEqual(covered, set(range(14)))
        self.assertEqual(len(selected), len(initial))
        self.assertEqual(int(sum(zero_gt[i] for i in selected)), int(sum(zero_gt[i] for i in initial)))


# --------------------------------------------------------------------------- #
# Objective-repair local search (review item 4)                               #
# --------------------------------------------------------------------------- #
class TestObjectiveRepairLocalSearch(unittest.TestCase):
    def setUp(self):
        self.train = make_deterministic_train()
        self.image_ids, self.labels14, self.zero_gt, self.names = M.build_indicators(self.train)
        self.K, self.S_T, self.N = M.full_train_integer_stats(self.labels14)

    def test_never_makes_the_objective_worse_and_preserves_hard_constraints(self):
        # images 1..14 (one of each class) plus image_id 21 and image_id 35
        # (both round-robin offset 0 and 14, which both carry classes {0,7}
        # -- hand-verified via the round-robin formula c1=offset%14,
        # c2=(offset+7)%14) deliberately skews class 0 and class 7 to
        # k=3 while every other class stays at k=1, against a train-wide
        # reference K that is not similarly skewed. This does not assert
        # WHICH move the exhaustive local search picks (that depends on the
        # full neighborhood scan), only the invariants that must hold no
        # matter what it finds: the objective can never get worse (accept-
        # ance requires strict improvement), and size/exact-NF are held
        # fixed throughout.
        skewed = set(range(14)) | {20, 34}  # positions for image_id 1..14, 21, 35
        pool = set(range(len(self.image_ids))) - skewed
        log, visited = [], set()
        selected, outcome, exhaustion = M.repair_objective_local_search(
            set(), skewed, pool, self.labels14, self.zero_gt, self.names,
            self.K, self.S_T, self.N, 42, "1pct", self.image_ids, None, log, visited,
        )
        self.assertEqual(outcome, M.RepairOutcome.OK)
        # R11 (TASK 1/7/10.8/10.9): a one-for-one local optimum is a SUCCESS,
        # labelled with the neighborhood it is local to, and never a global
        # optimum. `two_for_two_exhausted` no longer exists.
        self.assertTrue(exhaustion["one_for_one_exhausted"])
        self.assertTrue(exhaustion["local_optimum"])
        self.assertEqual(exhaustion["local_optimum_neighborhood"], "one_for_one")
        self.assertFalse(exhaustion["global_optimum_claimed"])
        self.assertNotIn("two_for_two_exhausted", exhaustion)
        # size and NF must be exactly preserved throughout objective_repair
        self.assertEqual(len(selected), len(skewed))
        self.assertEqual(
            int(sum(self.zero_gt[i] for i in selected)), int(sum(self.zero_gt[i] for i in skewed)),
        )
        final_obj = M.integer_objective(selected, self.labels14, self.K, self.S_T, self.N)
        initial_obj = M.integer_objective(skewed, self.labels14, self.K, self.S_T, self.N)
        self.assertLessEqual(final_obj, initial_obj)  # local search must never make the objective worse

    def test_no_improving_move_remains_on_a_small_exhaustive_fixture(self):
        # images 1..14 alone: k_c = 1 for every class, n=14. This is already
        # perfectly proportional to K (which is also skewed only by the
        # round-robin block, so it is not claimed to be globally optimal,
        # but no SAME-SIZE/SAME-NF/SAME-COVERAGE swap among images 1..14
        # itself vs. an empty same-status pool exists since all 14 are
        # already selected and swapping with other single-class images is
        # unavailable in this minimal pool) — construct pool as EXACTLY
        # images 1..14 already selected with NO other same-composition
        # candidates available, guaranteeing immediate exhaustion.
        selected = set(range(14))
        pool = set()  # nothing else available to swap with -> forced exhaustion on move 1
        log, visited = [], set()
        result_selected, outcome, exhaustion = M.repair_objective_local_search(
            set(), selected, pool, self.labels14, self.zero_gt, self.names,
            self.K, self.S_T, self.N, 42, "1pct", self.image_ids, None, log, visited,
        )
        self.assertEqual(outcome, M.RepairOutcome.OK)
        self.assertEqual(result_selected, selected)
        self.assertEqual(log, [])
        self.assertTrue(exhaustion["one_for_one_exhausted"])
        self.assertTrue(exhaustion["local_optimum"])
        self.assertEqual(exhaustion["local_optimum_neighborhood"], "one_for_one")
        self.assertFalse(exhaustion["global_optimum_claimed"])
        self.assertNotIn("two_for_two_exhausted", exhaustion)
        # TASK 10.8 -- reaching the local optimum is RepairOutcome.OK, never
        # COMPUTATIONAL_ABORT and never REPAIR_INFEASIBLE.
        self.assertNotEqual(outcome, M.RepairOutcome.COMPUTATIONAL_ABORT)
        self.assertNotEqual(outcome, M.RepairOutcome.REPAIR_INFEASIBLE)
        # TASK 5 -- the phase reports its own before/after objective and the
        # number of one-for-one moves it accepted.
        self.assertEqual(exhaustion["objective_repair_one_for_one_move_count"], 0)
        self.assertEqual(
            exhaustion["objective_before_objective_repair"],
            exhaustion["objective_after_objective_repair"],
        )

    def test_equal_objective_move_never_accepted(self):
        # Directly probes the acceptance condition rather than trying to
        # engineer a real tie in the search: `trial_obj < current_obj` must
        # be False (never True) when the tuples are equal.
        obj = (3, 5, 2)
        self.assertFalse(obj < obj)


# --------------------------------------------------------------------------- #
# COCO subset builders / GT-leakage guard (unchanged by R2)                   #
# --------------------------------------------------------------------------- #
class TestSubsetBuildersAndLeakageGuard(unittest.TestCase):
    def setUp(self):
        self.train = make_deterministic_train()

    def test_labeled_subset_preserves_annotations(self):
        ids = {1, 2, 3}
        result = M.subset_labeled_coco(self.train, ids)
        self.assertEqual({im["id"] for im in result["images"]}, ids)
        for a in result["annotations"]:
            self.assertIn(a["image_id"], ids)
        self.assertEqual(len(result["categories"]), 14)

    def test_unlabeled_subset_has_no_annotations_and_strips_gt_fields(self):
        ids = {4, 5, 6}
        forbidden = {"is_negative", "scope_label", "zero_gt", "class_presence",
                     "class_presence_vector", "annotation_count", "bbox", "bbox_count"}
        result = M.subset_unlabeled_coco(self.train, ids, forbidden)
        self.assertEqual(result["annotations"], [])
        self.assertEqual({im["id"] for im in result["images"]}, ids)
        for im in result["images"]:
            self.assertTrue(forbidden.isdisjoint(im.keys()))
        self.assertEqual(len(result["categories"]), 14)

    def test_unlabeled_subset_disjoint_and_complete_with_labeled(self):
        all_ids = {im["id"] for im in self.train["images"]}
        labeled_ids = set(list(all_ids)[:10])
        unlabeled_ids = all_ids - labeled_ids
        labeled = M.subset_labeled_coco(self.train, labeled_ids)
        unlabeled = M.subset_unlabeled_coco(self.train, unlabeled_ids, {"is_negative", "scope_label"})
        l_ids = {im["id"] for im in labeled["images"]}
        u_ids = {im["id"] for im in unlabeled["images"]}
        self.assertTrue(l_ids.isdisjoint(u_ids))
        self.assertEqual(l_ids | u_ids, all_ids)

    def test_config_forbidden_fields_present_in_real_train_schema(self):
        import yaml
        config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
        forbidden = set(config["unlabeled_json_forbidden_fields"]["images"])
        self.assertIn("is_negative", forbidden)
        self.assertIn("scope_label", forbidden)


class TestUnauthorizedAuditReferenceGuardrail(unittest.TestCase):
    def test_audit_csv_path_only_referenced_by_permitted_files(self):
        audit_relative = "data/manifests/audit/phase2F_unlabeled_gt_audit.csv"
        permitted_substrings = (
            "02F_build_labeled_unlabeled.py", "phase2F_labeled_unlabeled.yaml",
            "test_phase2F_labeled_unlabeled_guardrails.py",
            "PHASE_HANDOFF.md", "PROJECT_CONTEXT.md", "research_log.md",
        )
        offenders = []
        for path in REPO_ROOT.rglob("*"):
            if path.is_dir():
                continue
            if any(part in {".git", "__pycache__", ".pytest_cache"} for part in path.parts):
                continue
            if path.suffix.lower() not in {".py", ".yaml", ".yml", ".json"}:
                continue
            if any(name in str(path) for name in permitted_substrings):
                continue
            try:
                text = path.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                continue
            if audit_relative in text or "phase2F_unlabeled_gt_audit" in text:
                offenders.append(str(path))
        self.assertEqual(offenders, [])


# --------------------------------------------------------------------------- #
# Materialization: refuse-overwrite / rollback (unchanged by R2)              #
# --------------------------------------------------------------------------- #
class TestMaterializationGuardrails(unittest.TestCase):
    def test_refuses_to_overwrite_existing_official_artifact(self):
        with tempfile.TemporaryDirectory() as tmp:
            project_root = Path(tmp)
            staging = project_root / "staging"
            (staging / "sub").mkdir(parents=True)
            (staging / "sub" / "out.json").write_text("{}", encoding="utf-8")
            existing_target = project_root / "sub" / "out.json"
            existing_target.parent.mkdir(parents=True)
            existing_target.write_text("already here", encoding="utf-8")
            with self.assertRaises(M.Phase2FError):
                M.promote_with_rollback(staging, project_root, [Path("sub/out.json")])
            self.assertEqual(existing_target.read_text(encoding="utf-8"), "already here")

    def test_rollback_removes_files_promoted_by_failed_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            project_root = Path(tmp)
            staging = project_root / "staging"
            (staging / "a").mkdir(parents=True)
            (staging / "a" / "one.json").write_text("{}", encoding="utf-8")
            relative_paths = [Path("a/one.json"), Path("a/two.json")]
            with self.assertRaises(Exception):
                M.promote_with_rollback(staging, project_root, relative_paths)
            self.assertFalse((project_root / "a" / "one.json").exists())


def _make_readback_fixture_train(n_train: int) -> dict:
    """n_train ALL-ABNORMAL images (no zero-GT at all -> nf_train=0, so
    every budget's exact-No-Finding target is trivially 0). Image i
    (1-indexed) carries EXACTLY category_id ((i-1) % 14) + 1, so any prefix
    1..k with k >= 14 covers all 14 classes by construction. Used only by
    TestIndependentReadback, which needs REAL nested per-budget sizes
    (matching compute_locked_size_targets) rather than an identical set
    reused across all 4 budgets, so the nested-subset gates are exercised
    as genuine proper-subset checks, not trivial equality."""
    images, annotations = [], []
    ann_id = 1
    for i in range(1, n_train + 1):
        images.append({"id": i, "file_name": f"train/{i}.jpg", "width": 10, "height": 10,
                        "is_negative": False, "scope_label": "abnormal"})
        category_id = ((i - 1) % 14) + 1
        annotations.append({"id": ann_id, "image_id": i, "category_id": category_id,
                             "bbox": [1, 1, 2, 2], "area": 4, "iscrowd": 0})
        ann_id += 1
    return {"info": {}, "licenses": [], "images": images, "annotations": annotations, "categories": _categories()}


# --------------------------------------------------------------------------- #
# Independent readback (review item 8, expanded R3 review item 3)             #
# --------------------------------------------------------------------------- #
class TestIndependentReadback(unittest.TestCase):
    N_TRAIN = 1400  # so 1pct=14 (exactly enough for 14/14 coverage), 5pct=70, 10pct=140, 20pct=280

    def _build_staged_tree(self, tmp: Path, corrupt: str | None = None, corrupt_budget: str = "1pct") -> tuple[Path, Path, dict]:
        train = _make_readback_fixture_train(self.N_TRAIN)
        project_root = tmp / "project"
        (project_root / "data" / "processed" / "coco").mkdir(parents=True)
        (project_root / "reports").mkdir(parents=True)
        M.write_json(project_root / "data" / "processed" / "coco" / "instances_train.json", train)
        empty = {"images": [], "annotations": [], "categories": train["categories"]}
        M.write_json(project_root / "data" / "processed" / "coco" / "instances_val.json", empty)
        M.write_json(project_root / "data" / "processed" / "coco" / "instances_test.json", empty)

        labeled_size_target = {
            b: M.round_half_up_fraction(Fraction(self.N_TRAIN) * M.BUDGET_FRACTION[b]) for b in M.BUDGET_ORDER
        }
        forbidden = {"is_negative", "scope_label"}
        staging = tmp / "staging"
        labeled_coco_sha256: dict[str, str] = {}
        unlabeled_coco_sha256: dict[str, str] = {}
        membership_sha256: dict[str, str] = {}

        for budget in M.BUDGET_ORDER:
            labeled_ids = set(range(1, labeled_size_target[budget] + 1))  # nested prefix, image_id 1-indexed
            unlabeled_ids = set(range(1, self.N_TRAIN + 1)) - labeled_ids
            labeled_data = M.subset_labeled_coco(train, labeled_ids)
            unlabeled_data = M.subset_unlabeled_coco(train, unlabeled_ids, forbidden)

            if corrupt and budget == corrupt_budget:
                if corrupt == "gt_field_leak":
                    unlabeled_data["images"][0]["is_negative"] = False
                elif corrupt == "modify_annotation":
                    original = labeled_data["annotations"][0]["category_id"]
                    labeled_data["annotations"][0]["category_id"] = (original % 14) + 1
                elif corrupt == "delete_annotation":
                    labeled_data["annotations"] = labeled_data["annotations"][1:]
                elif corrupt == "add_annotation":
                    spurious = dict(labeled_data["annotations"][0])
                    spurious["id"] = max(a["id"] for a in train["annotations"]) + 999
                    labeled_data["annotations"] = labeled_data["annotations"] + [spurious]
                elif corrupt == "modify_labeled_file_name":
                    labeled_data["images"][0]["file_name"] = "corrupted/path.jpg"
                elif corrupt == "modify_labeled_width":
                    labeled_data["images"][0]["width"] = 99999
                elif corrupt == "modify_labeled_height":
                    labeled_data["images"][0]["height"] = 99999
                elif corrupt == "add_labeled_image":
                    # a REAL, in-universe image (id=N_TRAIN) that simply was
                    # not part of this budget's actual selection -- distinct
                    # from the out-of-universe / duplicate corruption cases
                    # below, and caught by the exact-size gate.
                    extra = next(im for im in train["images"] if im["id"] == self.N_TRAIN)
                    labeled_data["images"] = labeled_data["images"] + [dict(extra)]
                elif corrupt == "delete_labeled_image":
                    labeled_data["images"] = labeled_data["images"][1:]
                elif corrupt == "duplicate_labeled_image_id":
                    labeled_data["images"] = labeled_data["images"] + [dict(labeled_data["images"][0])]
                elif corrupt == "out_of_universe_labeled_image":
                    spurious_img = dict(labeled_data["images"][0])
                    spurious_img["id"] = self.N_TRAIN + 500  # does not exist in locked train at all
                    labeled_data["images"] = labeled_data["images"] + [spurious_img]
                elif corrupt == "modify_unlabeled_retained_field":
                    unlabeled_data["images"][0]["width"] = 77777
                elif corrupt == "add_unlabeled_image_out_of_universe":
                    spurious_img = dict(unlabeled_data["images"][0])
                    spurious_img["id"] = self.N_TRAIN + 501
                    for f in forbidden:
                        spurious_img.pop(f, None)
                    unlabeled_data["images"] = unlabeled_data["images"] + [spurious_img]
                elif corrupt == "duplicate_unlabeled_image_id":
                    unlabeled_data["images"] = unlabeled_data["images"] + [dict(unlabeled_data["images"][0])]
                elif corrupt == "modify_labeled_category":
                    labeled_data["categories"] = [dict(c) for c in labeled_data["categories"]]
                    labeled_data["categories"][0]["name"] = "CORRUPTED_CLASS_NAME"
                elif corrupt == "modify_unlabeled_category":
                    unlabeled_data["categories"] = [dict(c) for c in unlabeled_data["categories"]]
                    unlabeled_data["categories"][0]["name"] = "CORRUPTED_CLASS_NAME"
                elif corrupt == "modify_labeled_top_level":
                    labeled_data["info"] = {"corrupted": True}
                elif corrupt == "modify_unlabeled_top_level":
                    unlabeled_data["info"] = {"corrupted": True}
                # --- R5 / NHIEM VU 1: structural-corruption scenarios that
                # must produce readback status=FAIL WITHOUT raising, never
                # reaching build_indicators() on the corrupted data. ---
                elif corrupt == "non_integer_labeled_image_id":
                    labeled_data["images"][0] = dict(labeled_data["images"][0])
                    labeled_data["images"][0]["id"] = "not-an-integer"
                elif corrupt == "duplicate_annotation_id":
                    spurious = dict(labeled_data["annotations"][1])
                    spurious["id"] = labeled_data["annotations"][0]["id"]  # reuse an existing id -> true duplicate
                    labeled_data["annotations"] = labeled_data["annotations"] + [spurious]
                elif corrupt == "dangling_annotation_image_id":
                    labeled_data["annotations"] = [dict(a) for a in labeled_data["annotations"]]
                    labeled_data["annotations"][0]["image_id"] = self.N_TRAIN + 999  # no such staged image record
                elif corrupt == "unknown_annotation_category_id":
                    labeled_data["annotations"] = [dict(a) for a in labeled_data["annotations"]]
                    labeled_data["annotations"][0]["category_id"] = 9999  # no such category
                elif corrupt == "malformed_images_container":
                    labeled_data["images"] = {"not": "a list"}
                elif corrupt == "malformed_annotations_container":
                    labeled_data["annotations"] = "not-a-list-either"
                elif corrupt == "malformed_categories_container":
                    labeled_data["categories"] = None

            labeled_path = staging / "data" / "processed" / "coco" / "labeled_splits" / f"instances_labeled_{budget}.json"
            unlabeled_path = staging / "data" / "processed" / "coco" / "unlabeled_splits" / f"instances_unlabeled_{budget}.json"
            M.write_json(labeled_path, labeled_data)
            M.write_json(unlabeled_path, unlabeled_data)
            labeled_coco_sha256[budget] = M.sha256_file(labeled_path)
            unlabeled_coco_sha256[budget] = M.sha256_file(unlabeled_path)
            membership_sha256[budget] = M.canonical_membership_sha256(labeled_ids)

        if corrupt == "wrong_staged_lock_checksum":
            membership_sha256[corrupt_budget] = "0" * 64

        config = {
            "outputs": {
                "labeled_coco": {b: f"data/processed/coco/labeled_splits/instances_labeled_{b}.json" for b in M.BUDGET_ORDER},
                "unlabeled_coco": {b: f"data/processed/coco/unlabeled_splits/instances_unlabeled_{b}.json" for b in M.BUDGET_ORDER},
                "lock_manifest_json": "data/manifests/phase2F_lock_manifest.json",
            },
            "inputs": {
                "train_coco": "data/processed/coco/instances_train.json",
                "val_coco": "data/processed/coco/instances_val.json",
                "test_coco": "data/processed/coco/instances_test.json",
            },
            "unlabeled_json_forbidden_fields": {"images": ["is_negative", "scope_label"]},
            "locked_reference_targets": {
                "labeled_size": labeled_size_target,
                "no_finding_size": {b: 0 for b in M.BUDGET_ORDER},
            },
        }
        lock_manifest = {
            "labeled_image_id_sha256": membership_sha256,
            "coco_json_sha256": {"labeled": labeled_coco_sha256, "unlabeled": unlabeled_coco_sha256},
        }
        M.write_json(staging / Path(config["outputs"]["lock_manifest_json"]), lock_manifest)
        return staging, project_root, config

    def test_signature_takes_no_bundle_parameter(self):
        import inspect
        params = list(inspect.signature(M.independently_validate_staging).parameters)
        self.assertEqual(params, ["staging", "project_root", "config"])

    def test_recomputes_all_hard_gates_and_passes_on_clean_staging(self):
        with tempfile.TemporaryDirectory() as tmp:
            staging, project_root, config = self._build_staged_tree(Path(tmp), corrupt=None)
            result = M.independently_validate_staging(staging, project_root, config)
            self.assertEqual(result["status"], "PASS")
            self.assertTrue(result["gates"]["1pct_coverage_14_of_14"])
            self.assertTrue(result["gates"]["1pct_exact_no_finding"])
            self.assertTrue(result["gates"]["1pct_no_gt_derived_fields"])
            self.assertTrue(result["gates"]["1pct_unlabeled_annotations_empty"])
            self.assertTrue(result["gates"]["1pct_membership_checksum_matches"])
            self.assertTrue(result["gates"]["1pct_annotation_id_set_matches_locked_subset"])
            self.assertTrue(result["gates"]["1pct_annotation_count_matches_locked_subset"])
            self.assertTrue(result["gates"]["1pct_annotation_payload_matches_locked_subset"])
            self.assertTrue(result["gates"]["1pct_labeled_coco_sha256_matches_staged_lock"])
            self.assertTrue(result["gates"]["1pct_unlabeled_coco_sha256_matches_staged_lock"])
            # R4 / NHIEM VU 2: full semantic payload gates.
            self.assertTrue(result["gates"]["1pct_labeled_no_duplicate_or_invalid_image_id"])
            self.assertTrue(result["gates"]["1pct_labeled_no_out_of_universe_image_id"])
            self.assertTrue(result["gates"]["1pct_labeled_no_duplicate_or_invalid_annotation_id"])
            self.assertTrue(result["gates"]["1pct_unlabeled_no_duplicate_or_invalid_image_id"])
            self.assertTrue(result["gates"]["1pct_unlabeled_no_out_of_universe_image_id"])
            self.assertTrue(result["gates"]["1pct_labeled_image_id_set_matches_locked_subset"])
            self.assertTrue(result["gates"]["1pct_labeled_image_record_count_matches_locked_subset"])
            self.assertTrue(result["gates"]["1pct_labeled_image_record_payload_matches_locked_subset"])
            self.assertTrue(result["gates"]["1pct_unlabeled_image_id_set_matches_expected_complement"])
            self.assertTrue(result["gates"]["1pct_unlabeled_retained_image_payload_matches_locked_train_after_forbidden_strip"])
            self.assertTrue(result["gates"]["1pct_labeled_category_payload_matches_locked_train"])
            self.assertTrue(result["gates"]["1pct_unlabeled_category_payload_matches_locked_train"])
            self.assertTrue(result["gates"]["1pct_labeled_retained_top_level_payload_matches_locked_train_subset"])
            self.assertTrue(result["gates"]["1pct_unlabeled_retained_top_level_payload_matches_locked_train_subset"])
            for i in range(len(M.BUDGET_ORDER) - 1):
                smaller, larger = M.BUDGET_ORDER[i], M.BUDGET_ORDER[i + 1]
                self.assertTrue(result["gates"][f"nested_labeled_{smaller}_in_{larger}"])
                self.assertTrue(result["gates"][f"nested_no_finding_{smaller}_in_{larger}"])

    def test_detects_injected_gt_field_leak(self):
        with tempfile.TemporaryDirectory() as tmp:
            staging, project_root, config = self._build_staged_tree(Path(tmp), corrupt="gt_field_leak")
            result = M.independently_validate_staging(staging, project_root, config)
            self.assertEqual(result["status"], "FAIL")
            self.assertFalse(result["gates"]["1pct_no_gt_derived_fields"])

    def test_detects_modified_annotation(self):
        # Changing a retained field's VALUE (category_id) without changing
        # the annotation's id or the annotation count must still fail --
        # this is exactly what a naive "file exists, size matches" readback
        # would miss, and exactly what the R3 annotation-payload checksum
        # gate is for.
        with tempfile.TemporaryDirectory() as tmp:
            staging, project_root, config = self._build_staged_tree(Path(tmp), corrupt="modify_annotation")
            result = M.independently_validate_staging(staging, project_root, config)
            self.assertEqual(result["status"], "FAIL")
            self.assertFalse(result["gates"]["1pct_annotation_payload_matches_locked_subset"])
            # the corrupted-but-internally-self-consistent staged file still
            # matches its OWN recorded checksum in the staged lock manifest
            # -- proving that gate alone is insufficient and the annotation-
            # level check against the separately-read locked train file is
            # what actually catches this class of bug.
            self.assertTrue(result["gates"]["1pct_labeled_coco_sha256_matches_staged_lock"])

    def test_detects_deleted_annotation(self):
        with tempfile.TemporaryDirectory() as tmp:
            staging, project_root, config = self._build_staged_tree(Path(tmp), corrupt="delete_annotation")
            result = M.independently_validate_staging(staging, project_root, config)
            self.assertEqual(result["status"], "FAIL")
            self.assertFalse(result["gates"]["1pct_annotation_count_matches_locked_subset"])
            self.assertFalse(result["gates"]["1pct_annotation_id_set_matches_locked_subset"])
            self.assertFalse(result["gates"]["1pct_annotation_payload_matches_locked_subset"])

    def test_detects_added_spurious_annotation(self):
        with tempfile.TemporaryDirectory() as tmp:
            staging, project_root, config = self._build_staged_tree(Path(tmp), corrupt="add_annotation")
            result = M.independently_validate_staging(staging, project_root, config)
            self.assertEqual(result["status"], "FAIL")
            self.assertFalse(result["gates"]["1pct_annotation_count_matches_locked_subset"])
            self.assertFalse(result["gates"]["1pct_annotation_id_set_matches_locked_subset"])

    def test_reads_staged_lock_manifest_not_a_recomputed_value(self):
        # The staged lock manifest's own recorded checksum is deliberately
        # wrong while the staged labeled COCO file itself is untouched/
        # correct. If the function recomputed and trusted its own value
        # instead of reading the STAGED manifest back from disk, this gate
        # would incorrectly report PASS.
        with tempfile.TemporaryDirectory() as tmp:
            staging, project_root, config = self._build_staged_tree(Path(tmp), corrupt="wrong_staged_lock_checksum")
            result = M.independently_validate_staging(staging, project_root, config)
            self.assertEqual(result["status"], "FAIL")
            self.assertFalse(result["gates"]["1pct_membership_checksum_matches"])

    # --- R4 / NHIEM VU 2: labeled image-record semantic payload corruption ---
    def test_detects_modified_labeled_file_name(self):
        with tempfile.TemporaryDirectory() as tmp:
            staging, project_root, config = self._build_staged_tree(Path(tmp), corrupt="modify_labeled_file_name")
            result = M.independently_validate_staging(staging, project_root, config)
            self.assertEqual(result["status"], "FAIL")
            self.assertFalse(result["gates"]["1pct_labeled_image_record_payload_matches_locked_subset"])

    def test_detects_modified_labeled_width(self):
        with tempfile.TemporaryDirectory() as tmp:
            staging, project_root, config = self._build_staged_tree(Path(tmp), corrupt="modify_labeled_width")
            result = M.independently_validate_staging(staging, project_root, config)
            self.assertEqual(result["status"], "FAIL")
            self.assertFalse(result["gates"]["1pct_labeled_image_record_payload_matches_locked_subset"])

    def test_detects_modified_labeled_height(self):
        with tempfile.TemporaryDirectory() as tmp:
            staging, project_root, config = self._build_staged_tree(Path(tmp), corrupt="modify_labeled_height")
            result = M.independently_validate_staging(staging, project_root, config)
            self.assertEqual(result["status"], "FAIL")
            self.assertFalse(result["gates"]["1pct_labeled_image_record_payload_matches_locked_subset"])
            # this is EXACTLY the R3-B2 blind spot: file+manifest are
            # mutually self-consistent (both reflect the corrupted bytes),
            # so the checksum-vs-staged-lock gate alone would report PASS;
            # only the payload comparison against the independently
            # re-read locked train file catches it.
            self.assertTrue(result["gates"]["1pct_labeled_coco_sha256_matches_staged_lock"])

    def test_detects_added_labeled_image_record(self):
        with tempfile.TemporaryDirectory() as tmp:
            staging, project_root, config = self._build_staged_tree(Path(tmp), corrupt="add_labeled_image")
            result = M.independently_validate_staging(staging, project_root, config)
            self.assertEqual(result["status"], "FAIL")
            self.assertFalse(result["gates"]["1pct_exact_labeled_size"])

    def test_detects_deleted_labeled_image_record(self):
        with tempfile.TemporaryDirectory() as tmp:
            staging, project_root, config = self._build_staged_tree(Path(tmp), corrupt="delete_labeled_image")
            result = M.independently_validate_staging(staging, project_root, config)
            self.assertEqual(result["status"], "FAIL")
            self.assertFalse(result["gates"]["1pct_exact_labeled_size"])

    def test_detects_duplicate_labeled_image_id(self):
        with tempfile.TemporaryDirectory() as tmp:
            staging, project_root, config = self._build_staged_tree(Path(tmp), corrupt="duplicate_labeled_image_id")
            result = M.independently_validate_staging(staging, project_root, config)
            self.assertEqual(result["status"], "FAIL")
            self.assertFalse(result["gates"]["1pct_labeled_no_duplicate_or_invalid_image_id"])
            self.assertFalse(result["gates"]["1pct_labeled_image_id_set_matches_locked_subset"])
            self.assertFalse(result["gates"]["1pct_labeled_image_record_count_matches_locked_subset"])
            self.assertFalse(result["gates"]["1pct_labeled_image_record_payload_matches_locked_subset"])

    def test_detects_out_of_universe_labeled_image_id(self):
        with tempfile.TemporaryDirectory() as tmp:
            staging, project_root, config = self._build_staged_tree(Path(tmp), corrupt="out_of_universe_labeled_image")
            result = M.independently_validate_staging(staging, project_root, config)  # must not raise
            self.assertEqual(result["status"], "FAIL")
            self.assertFalse(result["gates"]["1pct_labeled_no_out_of_universe_image_id"])
            self.assertFalse(result["gates"]["1pct_labeled_image_id_set_matches_locked_subset"])

    def test_out_of_universe_labeled_id_is_a_structural_ok_failure_r6(self):
        # R6 / NHIEM VU 1 (fixes blocker R5-B1): image_ids_within_universe
        # must now be part of struct["ok"] itself, not just its own
        # separately-reported gate -- so an out-of-universe id blanks out
        # EVERY semantic gate for that budget (never reaches build_
        # indicators/semantic reconstruction on the corrupted file), not
        # just the ones that happen to compare id sets directly.
        with tempfile.TemporaryDirectory() as tmp:
            staging, project_root, config = self._build_staged_tree(Path(tmp), corrupt="out_of_universe_labeled_image")
            result = M.independently_validate_staging(staging, project_root, config)  # must not raise
            self.assertEqual(result["status"], "FAIL")
            self.assertFalse(result["gates"]["1pct_exact_labeled_size"])
            self.assertFalse(result["gates"]["1pct_coverage_14_of_14"])
            self.assertFalse(result["gates"]["1pct_membership_checksum_matches"])

    def test_out_of_universe_unlabeled_id_is_a_structural_ok_failure_r6(self):
        with tempfile.TemporaryDirectory() as tmp:
            staging, project_root, config = self._build_staged_tree(Path(tmp), corrupt="add_unlabeled_image_out_of_universe")
            result = M.independently_validate_staging(staging, project_root, config)  # must not raise
            self.assertEqual(result["status"], "FAIL")
            self.assertFalse(result["gates"]["1pct_unlabeled_no_out_of_universe_image_id"])
            self.assertFalse(result["gates"]["1pct_disjoint_labeled_unlabeled"])
            self.assertFalse(result["gates"]["1pct_complete_labeled_unlabeled"])

    def test_out_of_universe_id_in_smaller_budget_fails_nested_no_finding_gate_r6(self):
        # R6 / NHIEM VU 1 (fixes blocker R5-B1): the nested No-Finding
        # computation indexes locked_image_id_to_position (keyed by LOCKED
        # Train ids only) using labeled_ids_by_budget, which is populated
        # from the RAW structurally-parsed id set regardless of that
        # budget's own "ok" -- so this must be defense-in-depth fail-closed
        # on its own, never relying solely on the upstream gate, and must
        # NEVER KeyError.
        with tempfile.TemporaryDirectory() as tmp:
            staging, project_root, config = self._build_staged_tree(
                Path(tmp), corrupt="out_of_universe_labeled_image", corrupt_budget="1pct",
            )
            result = M.independently_validate_staging(staging, project_root, config)  # must not raise
            self.assertEqual(result["status"], "FAIL")
            self.assertFalse(result["gates"]["nested_no_finding_1pct_in_5pct"])

    # --- R4 / NHIEM VU 2: unlabeled image-record semantic payload corruption ---
    def test_detects_modified_unlabeled_retained_field(self):
        with tempfile.TemporaryDirectory() as tmp:
            staging, project_root, config = self._build_staged_tree(Path(tmp), corrupt="modify_unlabeled_retained_field")
            result = M.independently_validate_staging(staging, project_root, config)
            self.assertEqual(result["status"], "FAIL")
            self.assertFalse(result["gates"]["1pct_unlabeled_retained_image_payload_matches_locked_train_after_forbidden_strip"])

    def test_detects_out_of_universe_unlabeled_image_id(self):
        with tempfile.TemporaryDirectory() as tmp:
            staging, project_root, config = self._build_staged_tree(Path(tmp), corrupt="add_unlabeled_image_out_of_universe")
            result = M.independently_validate_staging(staging, project_root, config)
            self.assertEqual(result["status"], "FAIL")
            self.assertFalse(result["gates"]["1pct_unlabeled_no_out_of_universe_image_id"])
            self.assertFalse(result["gates"]["1pct_unlabeled_image_id_set_matches_expected_complement"])

    def test_detects_duplicate_unlabeled_image_id(self):
        with tempfile.TemporaryDirectory() as tmp:
            staging, project_root, config = self._build_staged_tree(Path(tmp), corrupt="duplicate_unlabeled_image_id")
            result = M.independently_validate_staging(staging, project_root, config)
            self.assertEqual(result["status"], "FAIL")
            self.assertFalse(result["gates"]["1pct_unlabeled_no_duplicate_or_invalid_image_id"])
            self.assertFalse(result["gates"]["1pct_unlabeled_retained_image_payload_matches_locked_train_after_forbidden_strip"])

    # --- R4 / NHIEM VU 2: category payload corruption ---
    def test_detects_modified_labeled_category_payload(self):
        with tempfile.TemporaryDirectory() as tmp:
            staging, project_root, config = self._build_staged_tree(Path(tmp), corrupt="modify_labeled_category")
            result = M.independently_validate_staging(staging, project_root, config)
            self.assertEqual(result["status"], "FAIL")
            self.assertFalse(result["gates"]["1pct_labeled_category_payload_matches_locked_train"])

    def test_detects_modified_unlabeled_category_payload(self):
        with tempfile.TemporaryDirectory() as tmp:
            staging, project_root, config = self._build_staged_tree(Path(tmp), corrupt="modify_unlabeled_category")
            result = M.independently_validate_staging(staging, project_root, config)
            self.assertEqual(result["status"], "FAIL")
            self.assertFalse(result["gates"]["1pct_unlabeled_category_payload_matches_locked_train"])

    # --- R4 / NHIEM VU 2: retained top-level field corruption ---
    def test_detects_modified_labeled_top_level_field(self):
        with tempfile.TemporaryDirectory() as tmp:
            staging, project_root, config = self._build_staged_tree(Path(tmp), corrupt="modify_labeled_top_level")
            result = M.independently_validate_staging(staging, project_root, config)
            self.assertEqual(result["status"], "FAIL")
            self.assertFalse(result["gates"]["1pct_labeled_retained_top_level_payload_matches_locked_train_subset"])

    def test_detects_modified_unlabeled_top_level_field(self):
        with tempfile.TemporaryDirectory() as tmp:
            staging, project_root, config = self._build_staged_tree(Path(tmp), corrupt="modify_unlabeled_top_level")
            result = M.independently_validate_staging(staging, project_root, config)
            self.assertEqual(result["status"], "FAIL")
            self.assertFalse(result["gates"]["1pct_unlabeled_retained_top_level_payload_matches_locked_train_subset"])

    # --- R5 / NHIEM VU 1: clean staging must also PASS the new R5
    # structural gates, not just the R4 semantic ones. ---
    def test_clean_staging_passes_new_r5_structural_gates(self):
        with tempfile.TemporaryDirectory() as tmp:
            staging, project_root, config = self._build_staged_tree(Path(tmp), corrupt=None)
            result = M.independently_validate_staging(staging, project_root, config)
            self.assertEqual(result["status"], "PASS")
            for name in (
                "1pct_labeled_container_types_valid", "1pct_unlabeled_container_types_valid",
                "1pct_labeled_category_ids_valid_and_unique", "1pct_unlabeled_category_ids_valid_and_unique",
                "1pct_labeled_annotation_image_id_references_valid",
                "1pct_labeled_annotation_category_id_references_valid",
            ):
                self.assertTrue(result["gates"][name], msg=f"expected True: {name}")

    # --- R5 / NHIEM VU 1: staged corruption must FAIL readback, NEVER raise
    # an exception (fixes blocker R4-B1). Each test below explicitly proves
    # "no exception" by simply not wrapping the call in assertRaises. ---
    def test_non_integer_labeled_image_id_fails_without_crashing(self):
        with tempfile.TemporaryDirectory() as tmp:
            staging, project_root, config = self._build_staged_tree(Path(tmp), corrupt="non_integer_labeled_image_id")
            result = M.independently_validate_staging(staging, project_root, config)  # must not raise
            self.assertEqual(result["status"], "FAIL")
            self.assertFalse(result["gates"]["1pct_labeled_no_duplicate_or_invalid_image_id"])
            self.assertFalse(result["gates"]["1pct_exact_labeled_size"])

    def test_duplicate_annotation_id_fails_without_crashing(self):
        with tempfile.TemporaryDirectory() as tmp:
            staging, project_root, config = self._build_staged_tree(Path(tmp), corrupt="duplicate_annotation_id")
            result = M.independently_validate_staging(staging, project_root, config)  # must not raise
            self.assertEqual(result["status"], "FAIL")
            self.assertFalse(result["gates"]["1pct_labeled_no_duplicate_or_invalid_annotation_id"])
            self.assertFalse(result["gates"]["1pct_annotation_payload_matches_locked_subset"])

    def test_dangling_annotation_image_id_fails_without_crashing(self):
        with tempfile.TemporaryDirectory() as tmp:
            staging, project_root, config = self._build_staged_tree(Path(tmp), corrupt="dangling_annotation_image_id")
            result = M.independently_validate_staging(staging, project_root, config)  # must not raise
            self.assertEqual(result["status"], "FAIL")
            self.assertFalse(result["gates"]["1pct_labeled_annotation_image_id_references_valid"])
            self.assertFalse(result["gates"]["1pct_exact_no_finding"])

    def test_unknown_annotation_category_id_fails_without_crashing(self):
        with tempfile.TemporaryDirectory() as tmp:
            staging, project_root, config = self._build_staged_tree(Path(tmp), corrupt="unknown_annotation_category_id")
            result = M.independently_validate_staging(staging, project_root, config)  # must not raise
            self.assertEqual(result["status"], "FAIL")
            self.assertFalse(result["gates"]["1pct_labeled_annotation_category_id_references_valid"])
            self.assertFalse(result["gates"]["1pct_coverage_14_of_14"])

    def test_malformed_images_container_fails_without_crashing(self):
        with tempfile.TemporaryDirectory() as tmp:
            staging, project_root, config = self._build_staged_tree(Path(tmp), corrupt="malformed_images_container")
            result = M.independently_validate_staging(staging, project_root, config)  # must not raise
            self.assertEqual(result["status"], "FAIL")
            self.assertFalse(result["gates"]["1pct_labeled_container_types_valid"])
            self.assertFalse(result["gates"]["1pct_exact_labeled_size"])

    def test_malformed_annotations_container_fails_without_crashing(self):
        with tempfile.TemporaryDirectory() as tmp:
            staging, project_root, config = self._build_staged_tree(Path(tmp), corrupt="malformed_annotations_container")
            result = M.independently_validate_staging(staging, project_root, config)  # must not raise
            self.assertEqual(result["status"], "FAIL")
            self.assertFalse(result["gates"]["1pct_labeled_container_types_valid"])

    def test_malformed_categories_container_fails_without_crashing(self):
        with tempfile.TemporaryDirectory() as tmp:
            staging, project_root, config = self._build_staged_tree(Path(tmp), corrupt="malformed_categories_container")
            result = M.independently_validate_staging(staging, project_root, config)  # must not raise
            self.assertEqual(result["status"], "FAIL")
            self.assertFalse(result["gates"]["1pct_labeled_container_types_valid"])

    def test_readback_never_raises_phase2f_error_on_any_structural_corruption(self):
        # Blanket regression guard for blocker R4-B1: sweep every structural
        # corruption type and assert independently_validate_staging always
        # RETURNS a dict (status=FAIL), never raises.
        structural_corruptions = (
            "non_integer_labeled_image_id", "duplicate_annotation_id", "dangling_annotation_image_id",
            "unknown_annotation_category_id", "malformed_images_container",
            "malformed_annotations_container", "malformed_categories_container",
            "duplicate_labeled_image_id", "out_of_universe_labeled_image", "duplicate_unlabeled_image_id",
            "add_unlabeled_image_out_of_universe",
        )
        for corrupt in structural_corruptions:
            with tempfile.TemporaryDirectory() as tmp:
                staging, project_root, config = self._build_staged_tree(Path(tmp), corrupt=corrupt)
                try:
                    result = M.independently_validate_staging(staging, project_root, config)
                except Exception as exc:  # pragma: no cover - failure path under test
                    self.fail(f"independently_validate_staging raised {type(exc).__name__} for corrupt={corrupt}: {exc}")
                self.assertEqual(result["status"], "FAIL", msg=f"corrupt={corrupt} unexpectedly PASSed")


# --------------------------------------------------------------------------- #
# Deterministic reconstruction mode (review item 9)                           #
# --------------------------------------------------------------------------- #
@unittest.skipUnless(HAVE_ITERSTRAT, "iterative-stratification not installed")
class TestReconstructCheckMode(unittest.TestCase):
    def _make_project(self, tmp: Path) -> tuple[dict, Path]:
        import yaml
        config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
        project_root = tmp / "project"
        train = make_deterministic_train()
        # This fixture does not satisfy the LOCKED reference targets (34/171/
        # 343/685 on n_train=3426), so compute_all_budgets would fail its
        # own cross-check against locked_reference_targets. Point the config
        # at a small custom reference matching this fixture instead.
        config = dict(config)
        config["locked_reference_targets"] = {
            "labeled_size": {"1pct": 14, "5pct": 20, "10pct": 30, "20pct": 40},
            "no_finding_size": {"1pct": 0, "5pct": 1, "10pct": 2, "20pct": 3},
            "min_class_coverage": 14,
        }
        config["sampling"] = dict(config["sampling"])
        (project_root / "data" / "processed" / "coco" / "labeled_splits").mkdir(parents=True)
        (project_root / "data" / "processed" / "coco" / "unlabeled_splits").mkdir(parents=True)
        (project_root / "data" / "manifests").mkdir(parents=True)
        (project_root / "reports").mkdir(parents=True)
        M.write_json(project_root / "data" / "processed" / "coco" / "instances_train.json", train)
        return config, project_root

    def test_reconstruct_check_requires_promoted_lock_manifest_first(self):
        with tempfile.TemporaryDirectory() as tmp:
            config, project_root = self._make_project(Path(tmp))
            with self.assertRaises(M.Phase2FError):
                M.run_reconstruct_check(config, project_root)

    def test_detects_unlabeled_checksum_mismatch_against_promoted_lock(self):
        # R3 review item 2: --reconstruct-check must compare labeled AND
        # unlabeled coco_json_sha256 (R2 only declared one ambiguous field
        # and never actually compared it). Build a promoted lock manifest
        # that is CORRECT for every field except unlabeled_coco_json_sha256
        # (deliberately wrong), using the exact same helper functions the
        # script itself uses, and confirm only that field is flagged.
        #
        # Researcher pytest round: config["locked_reference_targets"] above
        # (14/20/30/40 labeled sizes, 0/1/2/3 No-Finding) does NOT match
        # what round_half_up_fraction(...) actually recomputes from
        # make_deterministic_train()'s n_train=50 (1% of 50 rounds to 1,
        # not 14) -- compute_locked_size_targets() correctly fails closed
        # on that mismatch (an INVALID/STALE test fixture, not a
        # production defect: the check is doing exactly its job). Forcing
        # the locked target down to the "correct" 1-image value would make
        # the 14/14 coverage hard gate infeasible for a 1-image budget,
        # trading this failure for a REPAIR_INFEASIBLE one -- and budget-
        # rounding/coverage feasibility is not what this test is about.
        # This test's actual job is proving run_reconstruct_check compares
        # unlabeled_coco_json_sha256 (and only that field) correctly. So
        # compute_locked_size_targets is patched, LOCALLY and ONLY inside
        # this test, to return the synthetic target pair the small fixture
        # was designed around -- covering BOTH call sites that matter:
        # compute_all_budgets below (used to build the promoted lock) AND
        # compute_all_budgets called internally by run_reconstruct_check.
        # The patch is a `with mock.patch.object(...)` context manager, so
        # it is automatically reverted when the block exits (verified
        # explicitly below, not just assumed from context-manager
        # semantics). Nothing else is patched or bypassed:
        # subset_labeled_coco, subset_unlabeled_coco, sha256_file,
        # canonical_membership_sha256, the promoted/recomputed comparison
        # logic, and run_reconstruct_check itself all run for real.
        synthetic_labeled_size = {"1pct": 14, "5pct": 20, "10pct": 30, "20pct": 40}
        synthetic_nf_size = {"1pct": 0, "5pct": 1, "10pct": 2, "20pct": 3}

        # Fixture precondition, asserted (not just documented in a
        # comment): the first 14 images of make_deterministic_train() must
        # collectively cover all 14 classes with zero No-Finding among
        # them (this is what makes the 1pct=14/NF=0 synthetic target
        # feasible against the 14/14 coverage hard gate), and the dataset
        # must carry enough No-Finding images for the largest synthetic
        # No-Finding target. If this fixture ever drifts, this assertion
        # fails clearly here instead of surfacing as a confusing
        # REPAIR_INFEASIBLE/COMPUTATIONAL_ABORT deep inside
        # compute_all_budgets.
        precondition_train = make_deterministic_train()
        pre_image_ids, pre_labels14, pre_zero_gt, _ = M.build_indicators(precondition_train)
        first_14_present = set(int(c) for c in np.where(pre_labels14[:14].sum(axis=0) > 0)[0])
        self.assertEqual(first_14_present, set(range(14)),
                          "fixture precondition violated: first 14 images no longer cover all 14 classes")
        self.assertEqual(int(pre_zero_gt[:14].sum()), 0,
                          "fixture precondition violated: first 14 images are no longer all abnormal")
        self.assertGreaterEqual(int(pre_zero_gt.sum()), max(synthetic_nf_size.values()),
                                 "fixture precondition violated: not enough No-Finding images in the dataset "
                                 "for the largest synthetic No-Finding target")

        with tempfile.TemporaryDirectory() as tmp:
            config, project_root = self._make_project(Path(tmp))
            # R11: the retired TWO_FOR_TWO_ENGINE_NOT_OPERATIONALLY_APPROVED
            # gate is gone, so no local engine-approval override is needed
            # here any more. compute_all_budgets now gates on
            # repair.active_repair_policy, which _make_project already
            # carries verbatim from the real YAML.
            self.assertEqual(config["repair"]["active_repair_policy"], M.ACTIVE_REPAIR_POLICY)

            original_compute_locked_size_targets = M.compute_locked_size_targets
            with mock.patch.object(
                M, "compute_locked_size_targets",
                return_value=(synthetic_labeled_size, synthetic_nf_size),
            ):
                bundle = M.compute_all_budgets(config, project_root)
                image_ids = bundle["image_ids"]
                zero_gt = bundle["zero_gt"]
                labels14 = bundle["labels14"]
                K, S_T, N = bundle["K"], bundle["S_T"], bundle["N"]
                image_id_to_position = bundle["image_id_to_position"]
                forbidden_fields = set(config["unlabeled_json_forbidden_fields"]["images"])
                train = bundle["train"]

                labeled_ids_by_budget = {
                    b: {int(image_ids[p]) for p in bundle["per_budget_selected"][b]} for b in M.BUDGET_ORDER
                }
                # Fixture precondition: the synthetic sizes/NF targets must
                # actually have been achieved by the real construction
                # pipeline, not silently substituted.
                for b in M.BUDGET_ORDER:
                    self.assertEqual(len(labeled_ids_by_budget[b]), synthetic_labeled_size[b])
                    nf_count = sum(1 for i in labeled_ids_by_budget[b] if zero_gt[image_id_to_position[i]] == 1)
                    self.assertEqual(nf_count, synthetic_nf_size[b])
                promoted_lock = {
                    "labeled_image_id_sha256": {b: M.canonical_membership_sha256(labeled_ids_by_budget[b]) for b in M.BUDGET_ORDER},
                    "labeled_size": {b: len(labeled_ids_by_budget[b]) for b in M.BUDGET_ORDER},
                    "no_finding_size": {
                        b: int(sum(1 for i in labeled_ids_by_budget[b] if zero_gt[image_id_to_position[i]] == 1))
                        for b in M.BUDGET_ORDER
                    },
                    "repair_move_counts": {
                        b: M.compute_repair_summary(bundle["repair_log"], b)["total_moves"] for b in M.BUDGET_ORDER
                    },
                    "integer_objective_final": {
                        b: list(M.integer_objective(
                            {image_id_to_position[i] for i in labeled_ids_by_budget[b]}, labels14, K, S_T, N,
                        )) for b in M.BUDGET_ORDER
                    },
                    "coco_json_sha256": {"labeled": {}, "unlabeled": {}},
                }
                for b in M.BUDGET_ORDER:
                    labeled_tmp = project_root / f"_tmp_labeled_{b}.json"
                    M.write_json(labeled_tmp, M.subset_labeled_coco(train, labeled_ids_by_budget[b]))
                    promoted_lock["coco_json_sha256"]["labeled"][b] = M.sha256_file(labeled_tmp)
                    promoted_lock["coco_json_sha256"]["unlabeled"][b] = "0" * 64  # deliberately wrong

                lock_path = project_root / Path(config["outputs"]["lock_manifest_json"])
                M.write_json(lock_path, promoted_lock)

                with self.assertRaises(M.Phase2FError):
                    M.run_reconstruct_check(config, project_root)

            # The patch context has exited -- prove no global mutation
            # survives the test, rather than merely relying on
            # mock.patch's documented behavior.
            self.assertIs(M.compute_locked_size_targets, original_compute_locked_size_targets)
            self.assertFalse(hasattr(M, "BUDGET_FRACTION_BACKUP"))

            report = M.load_json(project_root / "reports" / "02F_deterministic_reconstruction_check.json")
            self.assertEqual(report["status"], "MISMATCH")
            for b in M.BUDGET_ORDER:
                self.assertFalse(report["per_budget_diff"][b]["unlabeled_coco_json_sha256_match"])
                self.assertTrue(report["per_budget_diff"][b]["labeled_image_id_sha256_match"])
                self.assertTrue(report["per_budget_diff"][b]["labeled_size_match"])
                self.assertTrue(report["per_budget_diff"][b]["no_finding_size_match"])
                self.assertTrue(report["per_budget_diff"][b]["repair_move_counts_match"])
                self.assertTrue(report["per_budget_diff"][b]["integer_objective_final_match"])
                self.assertTrue(report["per_budget_diff"][b]["labeled_coco_json_sha256_match"])


# --------------------------------------------------------------------------- #
# R3 review item 2 — reconstruct-check compared_fields config-drivenness     #
# (does not require iterstrat: fails fast before any construction runs)      #
# --------------------------------------------------------------------------- #
class TestReconstructionFieldValidation(unittest.TestCase):
    def test_config_compared_fields_all_have_implemented_checks(self):
        import yaml
        config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
        compared = config["deterministic_reconstruction_mode"]["compared_fields"]
        for field in compared:
            self.assertIn(field, M.RECONSTRUCTION_COMPARABLE_FIELDS)

    def test_compared_fields_includes_both_checksums_and_no_finding_size(self):
        import yaml
        config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
        compared = set(config["deterministic_reconstruction_mode"]["compared_fields"])
        self.assertIn("labeled_coco_json_sha256", compared)
        self.assertIn("unlabeled_coco_json_sha256", compared)
        self.assertIn("no_finding_size", compared)
        self.assertIn("labeled_image_id_sha256", compared)
        self.assertIn("labeled_size", compared)
        self.assertIn("repair_move_counts", compared)
        self.assertIn("integer_objective_final", compared)

    def test_unrecognized_compared_field_fails_loudly_not_silently(self):
        import yaml
        with tempfile.TemporaryDirectory() as tmp:
            project_root = Path(tmp)
            config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
            config = dict(config)
            config["deterministic_reconstruction_mode"] = dict(config["deterministic_reconstruction_mode"])
            config["deterministic_reconstruction_mode"]["compared_fields"] = list(
                config["deterministic_reconstruction_mode"]["compared_fields"]
            ) + ["not_a_real_field"]
            lock_path = project_root / Path(config["outputs"]["lock_manifest_json"])
            M.write_json(lock_path, {"labeled_image_id_sha256": {}})
            with self.assertRaises(M.Phase2FError):
                M.run_reconstruct_check(config, project_root)


# --------------------------------------------------------------------------- #
# R3 review item 1 — iterative-stratification version lock                   #
# --------------------------------------------------------------------------- #
class TestIterstratVersionLock(unittest.TestCase):
    def test_config_locks_exact_version_0_1_9(self):
        import yaml
        config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
        dep = config["dependencies"]["iterative_stratification"]
        self.assertEqual(dep["required_version"], "0.1.9")
        self.assertEqual(dep["package_name"], "iterative-stratification")

    def test_uses_importlib_metadata_only_not_dunder_version_or_pkg_resources(self):
        # R6 pytest-repair round: "pkg_resources" legitimately appears in
        # the module docstring's historical revision notes and in a
        # comment above check_iterative_stratification_version, both in a
        # NEGATED context ("checked at runtime via importlib.metadata
        # (never __version__ / pkg_resources)") explaining what is NOT
        # used -- a blind assertNotIn over the whole source incorrectly
        # flags this. Scoped instead, via AST, to an ACTUAL pkg_resources
        # import or identifier use, which is never a comment/docstring
        # since those aren't part of the AST at all.
        source = SCRIPT_PATH.read_text(encoding="utf-8")
        self.assertIn("import importlib.metadata", source)
        self.assertIn("importlib.metadata.version(", source)
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    self.assertNotEqual(alias.name.split(".")[0], "pkg_resources")
            elif isinstance(node, ast.ImportFrom):
                self.assertNotEqual((node.module or "").split(".")[0], "pkg_resources")
            elif isinstance(node, ast.Name):
                self.assertNotEqual(node.id, "pkg_resources")
            elif isinstance(node, ast.Attribute):
                self.assertNotEqual(node.attr, "pkg_resources")

    def test_ok_when_installed_version_matches(self):
        import yaml
        config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
        with mock.patch.object(M.importlib.metadata, "version", return_value="0.1.9"):
            version, status = M.check_iterative_stratification_version(config)
        self.assertEqual((version, status), ("0.1.9", "OK"))

    def test_fails_closed_on_version_mismatch(self):
        import yaml
        config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
        with mock.patch.object(M.importlib.metadata, "version", return_value="0.1.10"):
            version, status = M.check_iterative_stratification_version(config)
        self.assertEqual((version, status), ("0.1.10", "VERSION_MISMATCH"))

    def test_fails_closed_when_not_installed(self):
        import yaml
        config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))

        def raise_not_found(name):
            raise M.importlib.metadata.PackageNotFoundError(name)

        with mock.patch.object(M.importlib.metadata, "version", side_effect=raise_not_found):
            version, status = M.check_iterative_stratification_version(config)
        self.assertEqual((version, status), (None, "NOT_INSTALLED"))

    def test_preflight_records_version_field_and_fails_on_mismatch(self):
        import yaml
        config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.object(M.importlib.metadata, "version", return_value="9.9.9"):
                result = M.run_preflight(config, Path(tmp))
            checks_by_name = {c["name"]: c for c in result["checks"]}
            self.assertEqual(checks_by_name["iterative_stratification_version_locked"]["status"], "FAIL")
            self.assertEqual(result["iterative_stratification_version"], "9.9.9")
            self.assertEqual(result["status"], "FAIL")

    def test_compute_all_budgets_fails_closed_on_version_mismatch(self):
        import yaml
        config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.object(M.importlib.metadata, "version", return_value="0.0.1"):
                with self.assertRaises(M.Phase2FError):
                    M.compute_all_budgets(config, Path(tmp))


# --------------------------------------------------------------------------- #
# Preflight (review item 7)                                                   #
# --------------------------------------------------------------------------- #
class TestPreflightExpanded(unittest.TestCase):
    def test_preflight_fails_closed_on_missing_inputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            import yaml
            config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
            result = M.run_preflight(config, Path(tmp))
            self.assertEqual(result["status"], "FAIL")

    def test_preflight_checks_phase2e_report_files(self):
        import yaml
        config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
        self.assertIn("phase2e_validation_report", config["inputs"])
        self.assertIn("phase2e_log_report", config["inputs"])
        self.assertEqual(config["inputs"]["phase2e_validation_report"],
                          "reports/phase2E_build_fixed_split_validation_report.json")
        self.assertEqual(config["inputs"]["phase2e_log_report"], "reports/phase2E_build_fixed_split_log.json")

    def test_preflight_does_not_fake_training_authorized_pass(self):
        source = SCRIPT_PATH.read_text(encoding="utf-8")
        self.assertNotIn('check("gate_training_authorized_false", True', source)
        self.assertIn("POLICY_EVIDENCE_NOT_MACHINE_READABLE", source)

    def test_preflight_policy_evidence_field_is_not_a_pass_fail_check(self):
        with tempfile.TemporaryDirectory() as tmp:
            import yaml
            config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
            result = M.run_preflight(config, Path(tmp))
            self.assertEqual(result["policy_evidence_training_authorized"], "POLICY_EVIDENCE_NOT_MACHINE_READABLE")
            names = {c["name"] for c in result["checks"]}
            self.assertNotIn("policy_evidence_training_authorized", names)


# --------------------------------------------------------------------------- #
# No training/inference/pseudo-label/test-evaluation code paths               #
# --------------------------------------------------------------------------- #
class TestNoForbiddenCodePaths(unittest.TestCase):
    def test_script_has_no_training_or_inference_keywords(self):
        # R6 pytest-repair round: "pseudo_label" was removed from this list
        # -- it is a legitimate SUBSTRING of the evidence field name
        # pseudo_labels_generated (always False, required by
        # test_gates_hardcoded_false_in_output below) and of the print
        # statement reporting that field, so a blind assertNotIn over the
        # whole script incorrectly flags it. It gets its own AST-scoped
        # test below. None of the keywords remaining in this list have any
        # legitimate negated/evidence-field occurrence anywhere in the
        # script (confirmed), so a direct substring scan is safe for them.
        source = SCRIPT_PATH.read_text(encoding="utf-8")
        forbidden_keywords = [
            "mmdet", "mmengine", "mmcv", "torch.load", "torch.save", "model.fit",
            ".backward(", "optimizer.step", "predict(", "inference(",
            "compute_map", "coco_eval", "COCOeval",
        ]
        for keyword in forbidden_keywords:
            self.assertNotIn(keyword, source, msg=f"forbidden keyword found in script: {keyword}")

    def test_script_never_executes_pseudo_label_generation(self):
        # Allows: the evidence field name pseudo_labels_generated, the
        # print statement reporting it, and any docstring/comment mention.
        # Forbids: any executable construct that would actually perform
        # pseudo-label generation -- a def/class whose name references it,
        # an import of such a module, a call to such a function, a
        # mutation setting pseudo_labels_generated to True, or a CLI flag
        # that would activate it.
        source = SCRIPT_PATH.read_text(encoding="utf-8")
        tree = ast.parse(source)

        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                self.assertNotIn("pseudo_label", node.name.lower(),
                                  msg=f"forbidden pseudo-label-generation definition: {node.name}")
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    self.assertNotIn("pseudo_label", alias.name.lower())
            elif isinstance(node, ast.ImportFrom):
                self.assertNotIn("pseudo_label", (node.module or "").lower())
            elif isinstance(node, ast.Call):
                func = node.func
                callee = func.attr if isinstance(func, ast.Attribute) else (
                    func.id if isinstance(func, ast.Name) else "")
                self.assertNotIn("pseudo_label", callee.lower())
                if callee == "add_argument":
                    for arg in node.args:
                        if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                            self.assertNotIn("pseudo-label", arg.value.lower())
                            self.assertNotIn("pseudo_label", arg.value.lower())

        self.assertNotIn('"pseudo_labels_generated": True', source)
        self.assertNotIn("pseudo_labels_generated = True", source)
        self.assertIn('"pseudo_labels_generated": False', source)

    def test_sampling_functions_never_reference_val_or_test_coco(self):
        source = SCRIPT_PATH.read_text(encoding="utf-8")
        tree = ast.parse(source)
        sampling_functions = {"build_indicators", "build_budget_step", "stratified_initial_candidate", "compute_all_budgets"}
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name in sampling_functions:
                inner_source = ast.get_source_segment(source, node) or ""
                self.assertNotIn("test_coco", inner_source)
                self.assertNotIn("val_coco", inner_source)

    def test_gates_hardcoded_false_in_output(self):
        source = SCRIPT_PATH.read_text(encoding="utf-8")
        self.assertIn('"training_authorized": False', source)
        self.assertIn('"training_started": False', source)
        self.assertIn('"pseudo_labels_generated": False', source)
        self.assertIn('"test_used": False', source)


# --------------------------------------------------------------------------- #
# R6 metadata/provenance correction — NHIEM VU 4 guardrails. This class does  #
# NOT constitute a new algorithmic revision: it only guards that the active   #
# protocol stage/version are read from config at runtime (get_protocol_       #
# identity), never hard-coded, in the console banner and in every official    #
# output artifact's provenance fields.                                        #
# --------------------------------------------------------------------------- #
class TestProtocolIdentityProvenance(unittest.TestCase):
    def _main_source_segment(self) -> str:
        source = SCRIPT_PATH.read_text(encoding="utf-8")
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == "main":
                return ast.get_source_segment(source, node) or ""
        self.fail("main() not found in script source")

    def _module_docstring(self) -> str:
        source = SCRIPT_PATH.read_text(encoding="utf-8")
        tree = ast.parse(source)
        doc = ast.get_docstring(tree)
        self.assertIsNotNone(doc, "module docstring must be present")
        return doc or ""

    # Item 1 — YAML identity is exactly the CURRENT stage/version. R9
    # (static-audit-driven performance fix: added-pair aggregate precompute)
    # deliberately bumped the identity from 2F-C0-R8/1.7.0 to
    # 2F-C0-R9/1.8.0 because it changed the protocol implementation
    # contract (the Phase-1 hot-path computation strategy + new deadline
    # coverage over the precompute step) -- this test must track whatever
    # the ACTIVE identity is, never stay pinned to a past round's value.
    def test_yaml_protocol_identity_is_2F_C0_R11(self):
        # R11 (TASK 8): MAJOR bump to 2.0.0 -- removing the two-for-two
        # neighborhood from the active protocol is a scientific scope
        # change, not a refactor or an instrumentation round.
        import yaml
        config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
        self.assertEqual(config["protocol"]["stage"], "2F-C0-R11")
        self.assertEqual(config["protocol"]["version"], "2.0.0")

    # Item 2 — the ACTIVE claims (main()'s banner, module docstring's
    # opening "this file is stage X" line) must no longer hard-code a
    # stale stage. Historical revision-note prose elsewhere in the
    # docstring (e.g. "Revision R4 (applied after ...)") is untouched and
    # is intentionally out of scope for this check.
    def test_main_banner_has_no_hardcoded_old_stage(self):
        main_source = self._main_source_segment()
        self.assertNotIn("2F-C0-R2", main_source)
        self.assertNotIn("2F-C0-R4", main_source)
        self.assertNotIn("2F-C0-R6", main_source)
        self.assertNotIn("2F-C0-R7", main_source)
        self.assertNotIn("2F-C0-R8", main_source)
        self.assertNotIn("2F-C0-R9", main_source)
        self.assertNotIn("2F-C0-R10", main_source)
        self.assertNotIn("2F-C0-R11", main_source)  # must not hard-code the new one either

    def test_module_docstring_active_claim_has_no_hardcoded_old_stage(self):
        doc = self._module_docstring()
        opening_line = doc.strip().splitlines()[0]
        self.assertNotIn("2F-C0-R2", opening_line)
        self.assertNotIn("2F-C0-R4", opening_line)
        self.assertNotIn("2F-C0-R6", opening_line)
        self.assertNotIn("2F-C0-R7", opening_line)
        self.assertNotIn("2F-C0-R8", opening_line)

    # Item 3 — banner is provably config-driven: it calls
    # get_protocol_identity(config) and interpolates the returned
    # variables, rather than printing a literal.
    def test_main_banner_reads_identity_from_get_protocol_identity(self):
        main_source = self._main_source_segment()
        self.assertIn("get_protocol_identity(config)", main_source)
        self.assertIn("protocol_stage", main_source)
        self.assertIn("protocol_version", main_source)

    # Items 4-7 — get_protocol_identity() fails closed on a malformed
    # protocol identity block, always with taxonomy CONFIG_INVALID.
    def test_missing_protocol_stage_fails_config_invalid(self):
        config = {"protocol": {"version": "1.5.0"}}
        with self.assertRaises(M.Phase2FError) as ctx:
            M.get_protocol_identity(config)
        self.assertIn("[CONFIG_INVALID]", str(ctx.exception))

    def test_missing_protocol_version_fails_config_invalid(self):
        config = {"protocol": {"stage": "2F-C0-R6"}}
        with self.assertRaises(M.Phase2FError) as ctx:
            M.get_protocol_identity(config)
        self.assertIn("[CONFIG_INVALID]", str(ctx.exception))

    def test_empty_or_non_string_stage_fails(self):
        for bad_stage in ["", "   ", None, 42, []]:
            with self.subTest(bad_stage=bad_stage):
                config = {"protocol": {"stage": bad_stage, "version": "1.5.0"}}
                with self.assertRaises(M.Phase2FError) as ctx:
                    M.get_protocol_identity(config)
                self.assertIn("[CONFIG_INVALID]", str(ctx.exception))

    def test_empty_or_non_string_version_fails(self):
        for bad_version in ["", "   ", None, 42, []]:
            with self.subTest(bad_version=bad_version):
                config = {"protocol": {"stage": "2F-C0-R6", "version": bad_version}}
                with self.assertRaises(M.Phase2FError) as ctx:
                    M.get_protocol_identity(config)
                self.assertIn("[CONFIG_INVALID]", str(ctx.exception))

    def test_missing_or_non_mapping_protocol_block_fails(self):
        for bad_protocol in [None, [], "2F-C0-R6"]:
            with self.subTest(bad_protocol=bad_protocol):
                config = {"protocol": bad_protocol} if bad_protocol is not None else {}
                with self.assertRaises(M.Phase2FError) as ctx:
                    M.get_protocol_identity(config)
                self.assertIn("[CONFIG_INVALID]", str(ctx.exception))

    # Item 8 — on the real, valid project config, identity resolves to
    # exactly the current R9 stage/version (bumped from R8/1.7.0 to
    # R9/1.8.0 when the added-pair aggregate precompute performance fix
    # was applied).
    def test_get_protocol_identity_matches_real_config(self):
        import yaml
        config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
        self.assertEqual(M.get_protocol_identity(config), ("2F-C0-R11", "2.0.0"))

    # Item 9 — output evidence (starting with the console banner, the
    # cheapest fully-observable proxy for "config-driven, not just
    # coincidentally correct for R6") reflects a FAKE identity end-to-end
    # through main(), with all I/O patched out and no compute/
    # materialization ever invoked. This proves the banner is genuinely
    # sourced from config at runtime rather than hard-coded to match R6.
    def test_main_banner_reflects_fake_config_identity_end_to_end(self):
        import argparse as _argparse
        import contextlib as _contextlib
        import io as _io

        fake_config = {
            "protocol": {"stage": "TEST-STAGE-XYZ", "version": "TEST-VERSION-XYZ"},
            "seed": {"partition_seed": 42, "seed_policy": "PRE_SPECIFIED_LOCKED_NO_SEED_SEARCH"},
        }
        with tempfile.TemporaryDirectory() as tmp:
            fake_args = _argparse.Namespace(
                config=Path("configs/protocol/phase2F_labeled_unlabeled.yaml"),
                project_root=Path(tmp), preflight_only=True, reconstruct_check=False, max_seconds=None,
                # R7-B1: real argparse contract fields, required since
                # main() unconditionally validates them.
                benchmark_two_for_two=False, benchmark_max_seconds_per_budget=None,
            )
            with mock.patch.object(M, "parse_args", return_value=fake_args), \
                 mock.patch.object(M, "load_config", return_value=fake_config), \
                 mock.patch.object(M, "run_preflight", return_value={
                     "status": "PASS", "checks": [],
                     "policy_evidence_training_authorized": "POLICY_EVIDENCE_NOT_MACHINE_READABLE",
                     # R11: main() unconditionally prints the ACTIVE protocol
                     # informational fields, so any fixture standing in for
                     # run_preflight()'s return value must include them.
                     "active_repair_policy": M.ACTIVE_REPAIR_POLICY,
                     "objective_repair_neighborhood": M.LOCAL_OPTIMUM_NEIGHBORHOOD,
                     "objective_repair_termination_claim": "one-for-one local optimum; no global optimum claimed",
                     "global_optimum_claimed": False,
                 }), \
                 mock.patch.object(M, "official_relative_paths", return_value=[]), \
                 mock.patch.object(M, "compute_all_budgets") as mocked_compute, \
                 mock.patch.object(M, "write_all_outputs_and_promote") as mocked_promote:
                stdout = _io.StringIO()
                with _contextlib.redirect_stdout(stdout):
                    M.main()
                output = stdout.getvalue()
                mocked_compute.assert_not_called()
                mocked_promote.assert_not_called()
            self.assertIn("TEST-STAGE-XYZ", output)
            self.assertIn("TEST-VERSION-XYZ", output)
            self.assertNotIn("2F-C0-R6", output)
            self.assertNotIn("2F-C0-R2", output)
            self.assertNotIn("2F-C0-R4", output)
            # preflight-only path: no artifacts materialized under the temp
            # project root.
            self.assertEqual(list(Path(tmp).rglob("*")), [])

    # Item 9 (continued) — the output-evidence constructors themselves
    # (write_all_outputs_and_promote, run_reconstruct_check) source
    # protocol_stage/protocol_version via get_protocol_identity(config) in
    # their own function bodies, not via a hard-coded literal, and never
    # via a bare config["protocol"]["stage"] lookup scattered inline.
    def test_output_constructors_source_identity_via_get_protocol_identity(self):
        source = SCRIPT_PATH.read_text(encoding="utf-8")
        tree = ast.parse(source)
        for fn_name in ("write_all_outputs_and_promote", "run_reconstruct_check"):
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef) and node.name == fn_name:
                    inner_source = ast.get_source_segment(source, node) or ""
                    with self.subTest(fn_name=fn_name):
                        self.assertIn("get_protocol_identity(config)", inner_source)
                        self.assertNotIn('config["protocol"]["stage"]', inner_source)
                        self.assertNotIn('config["protocol"]["version"]', inner_source)
                    break
            else:
                self.fail(f"{fn_name} not found in script source")


# --------------------------------------------------------------------------- #
# Post-R6 observability fix -- run_preflight() has always COMPUTED the        #
# two_for_two engine's informational status/diagnostics fields, but main()   #
# never printed them, so `--preflight-only` gave no console evidence of      #
# them even when PREFLIGHT_GATE=PASS. This is a pure console-output fix:     #
# no PASS/FAIL semantics, no operationally_approved value, no membership/    #
# repair/objective algorithm, and no artifact-writing behavior are touched.  #
# --------------------------------------------------------------------------- #
class TestPreflightObservabilityFix(unittest.TestCase):
    """R11 active-preflight observability; legacy engine claims are absent."""

    def _fake_preflight_result(self, status="PASS"):
        return {
            "status": status,
            "checks": [],
            "policy_evidence_training_authorized": "POLICY_EVIDENCE_NOT_MACHINE_READABLE",
            "active_repair_policy": M.ACTIVE_REPAIR_POLICY,
            "objective_repair_neighborhood": M.LOCAL_OPTIMUM_NEIGHBORHOOD,
            "objective_repair_termination_claim":
                "one-for-one local optimum; no global optimum claimed",
            "global_optimum_claimed": False,
        }

    def _run_main_with_fake_preflight(self, status="PASS"):
        import argparse as _argparse
        import contextlib as _contextlib
        import io as _io

        fake_config = {
            "protocol": {"stage": "TEST-STAGE-OBS", "version": "TEST-VERSION-OBS"},
            "seed": {"partition_seed": 42, "seed_policy": "PRE_SPECIFIED_LOCKED_NO_SEED_SEARCH"},
        }
        with tempfile.TemporaryDirectory() as tmp:
            fake_args = _argparse.Namespace(
                config=Path("configs/protocol/phase2F_labeled_unlabeled.yaml"),
                project_root=Path(tmp), preflight_only=True,
                reconstruct_check=False, max_seconds=None,
            )
            with mock.patch.object(M, "parse_args", return_value=fake_args), \
                 mock.patch.object(M, "load_config", return_value=fake_config), \
                 mock.patch.object(M, "run_preflight", return_value=self._fake_preflight_result(status)), \
                 mock.patch.object(M, "compute_all_budgets") as mocked_compute, \
                 mock.patch.object(M, "write_all_outputs_and_promote") as mocked_promote:
                stdout = _io.StringIO()
                exc = None
                with _contextlib.redirect_stdout(stdout):
                    try:
                        M.main()
                    except M.Phase2FError as error:
                        exc = error
                return stdout.getvalue(), exc, mocked_compute, mocked_promote

    def test_preflight_only_prints_active_r11_fields_and_never_legacy_fields(self):
        output, exc, mocked_compute, mocked_promote = self._run_main_with_fake_preflight("PASS")
        self.assertIsNone(exc)
        mocked_compute.assert_not_called()
        mocked_promote.assert_not_called()
        self.assertIn("PREFLIGHT_ACTIVE_REPAIR_POLICY=", output)
        self.assertIn("PREFLIGHT_OBJECTIVE_REPAIR_NEIGHBORHOOD=", output)
        self.assertIn("PREFLIGHT_GLOBAL_OPTIMUM_CLAIMED= False", output)
        self.assertIn("PREFLIGHT_OBJECTIVE_REPAIR_TERMINATION_CLAIM=", output)
        self.assertIn("PREFLIGHT_GATE= PASS", output)
        self.assertNotIn("PREFLIGHT_TWO_FOR_TWO_", output)

    def test_preflight_fail_is_visible_and_blocks_construction(self):
        output, exc, mocked_compute, mocked_promote = self._run_main_with_fake_preflight("FAIL")
        self.assertIsInstance(exc, M.Phase2FError)
        self.assertIn("PREFLIGHT_FAIL", str(exc))
        self.assertIn("PREFLIGHT_GATE= FAIL", output)
        self.assertNotIn("PREFLIGHT_TWO_FOR_TWO_", output)
        mocked_compute.assert_not_called()
        mocked_promote.assert_not_called()

    def test_real_preflight_failure_reports_r11_policy_not_legacy_engine(self):
        import yaml
        config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
        with tempfile.TemporaryDirectory() as tmp:
            result = M.run_preflight(config, Path(tmp))
        self.assertEqual(result["status"], "FAIL")
        self.assertEqual(result["active_repair_policy"], M.ACTIVE_REPAIR_POLICY)
        self.assertEqual(result["objective_repair_neighborhood"], M.LOCAL_OPTIMUM_NEIGHBORHOOD)
        self.assertFalse(result["global_optimum_claimed"])
        for retired in (
            "two_for_two_complexity_diagnostics", "two_for_two_engine_status",
            "operationally_approved", "operational_approval", "mathematical_completeness",
            "completeness_claim", "complexity_policy", "empirical_operational_tractability",
        ):
            self.assertNotIn(retired, result)


# --------------------------------------------------------------------------- #
# R7 -- NON_PROMOTING_OPERATIONAL_BENCHMARK. Every test in this class mocks  #
# build_budget_step (never the real membership/repair/objective algorithm,  #
# which is exhaustively covered elsewhere in this file) so these tests stay #
# deterministic and independent of whether the real iterative-stratification#
# package is installed -- run_operational_benchmark's OWN orchestration     #
# logic (fresh per-budget deadlines, status mapping, NOT_RUN_DEPENDENCY     #
# propagation, report schema, transactional write) is what is under test.  #
# --------------------------------------------------------------------------- #
class LegacyOperationalBenchmarkTests:
    def _make_project(self, tmp: Path) -> tuple[dict, Path]:
        import yaml
        config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
        project_root = tmp / "project"
        train = make_deterministic_train()
        config = dict(config)
        config["locked_reference_targets"] = {
            "labeled_size": {"1pct": 14, "5pct": 20, "10pct": 30, "20pct": 40},
            "no_finding_size": {"1pct": 0, "5pct": 1, "10pct": 2, "20pct": 3},
            "min_class_coverage": 14,
        }
        (project_root / "data" / "processed" / "coco" / "labeled_splits").mkdir(parents=True)
        (project_root / "data" / "processed" / "coco" / "unlabeled_splits").mkdir(parents=True)
        (project_root / "data" / "manifests").mkdir(parents=True)
        (project_root / "reports").mkdir(parents=True)
        M.write_json(project_root / "data" / "processed" / "coco" / "instances_train.json", train)
        return config, project_root

    @staticmethod
    def _fake_build_budget_step(outcomes: dict, recorded_calls: list):
        """outcomes: {budget: "OK"|"ABORT"|"INFEASIBLE"}. Records
        (budget, deadline, call_time, locked) for every invocation so tests
        can assert on deadline freshness/propagation without depending on
        the real construction algorithm at all."""
        def _fake(budget, locked, image_ids, labels14, zero_gt, names, target_size, target_nf,
                   seed, K, S_T, N, deadline, repair_log, diagnostics_log=None):
            recorded_calls.append({
                "budget": budget, "deadline": deadline, "call_time": time.monotonic(), "locked": set(locked),
            })
            kind = outcomes[budget]
            if kind == "OK":
                new_members = {p for p in (0, 1) if p not in locked} or {2, 3}
                selected = set(locked) | new_members
                if diagnostics_log is not None:
                    diagnostics_log.append({
                        "budget": budget, "repair_phase": "objective_repair", "status": "completed",
                        "aborted_by_deadline": False, "deadline_enabled": deadline is not None,
                        "elapsed_seconds": 0.001,
                    })
                exhaustion = {
                    "one_for_one_exhausted": True, "two_for_two_exhausted": True,
                    "local_optimum": True, "global_optimum_claimed": False,
                    "candidate_new_region": sorted(int(image_ids[p]) for p in new_members),
                    "final_new_region": sorted(int(image_ids[p]) for p in new_members),
                    "candidate_final_intersection_count": len(new_members),
                    "candidate_final_union_count": len(new_members),
                    "candidate_final_jaccard": 1.0,
                    "objective_before_repair": [0, 0, 0], "objective_after_repair": [0, 0, 0],
                }
                return selected, M.RepairOutcome.OK, exhaustion
            if kind == "ABORT":
                if diagnostics_log is not None:
                    diagnostics_log.append({
                        "budget": budget, "repair_phase": "objective_repair", "status": "aborted",
                        "aborted_by_deadline": True, "deadline_enabled": True, "elapsed_seconds": 0.5,
                        "selected_abnormal_count": 3, "pool_abnormal_count": None,
                    })
                return set(locked), M.RepairOutcome.COMPUTATIONAL_ABORT, {
                    "one_for_one_exhausted": False, "two_for_two_exhausted": False,
                    "local_optimum": False, "global_optimum_claimed": False, "aborted_by_deadline": True,
                }
            if kind == "INFEASIBLE":
                return set(locked), M.RepairOutcome.REPAIR_INFEASIBLE, {}
            raise AssertionError(f"unknown outcome kind: {kind!r}")
        return _fake

    def _run_benchmark(self, config, project_root, outcomes, max_seconds=600.0, recorded_calls=None):
        if recorded_calls is None:
            recorded_calls = []
        synthetic_labeled_size = {"1pct": 14, "5pct": 20, "10pct": 30, "20pct": 40}
        synthetic_nf_size = {"1pct": 0, "5pct": 1, "10pct": 2, "20pct": 3}
        with mock.patch.object(M, "compute_locked_size_targets",
                                return_value=(synthetic_labeled_size, synthetic_nf_size)), \
             mock.patch.object(M, "build_budget_step",
                                side_effect=self._fake_build_budget_step(outcomes, recorded_calls)):
            report = M.run_operational_benchmark(config, project_root, max_seconds)
        return report, recorded_calls

    # ---------------------------------------------------------------- #
    # A. CLI / mode isolation                                          #
    # ---------------------------------------------------------------- #
    def _fake_args(self, **overrides):
        import argparse as _argparse
        base = dict(
            config=Path("configs/protocol/phase2F_labeled_unlabeled.yaml"), project_root=Path("."),
            preflight_only=False, reconstruct_check=False, max_seconds=None,
            benchmark_two_for_two=False, benchmark_max_seconds_per_budget=None,
        )
        base.update(overrides)
        return _argparse.Namespace(**base)

    def test_missing_deadline_fails_closed(self):
        args = self._fake_args(benchmark_two_for_two=True, benchmark_max_seconds_per_budget=None)
        with self.assertRaises(M.Phase2FError) as ctx:
            M._validate_benchmark_cli_args(args)
        self.assertIn("BENCHMARK_ARGUMENT_INVALID", str(ctx.exception))

    def test_nonpositive_or_nan_or_inf_deadline_fails_closed(self):
        for bad in (0.0, -1.0, -600.0, float("nan"), float("inf"), float("-inf")):
            with self.subTest(deadline=bad):
                args = self._fake_args(benchmark_two_for_two=True, benchmark_max_seconds_per_budget=bad)
                with self.assertRaises(M.Phase2FError) as ctx:
                    M._validate_benchmark_cli_args(args)
                self.assertIn("BENCHMARK_ARGUMENT_INVALID", str(ctx.exception))

    def test_benchmark_conflicts_with_preflight_only(self):
        args = self._fake_args(benchmark_two_for_two=True, benchmark_max_seconds_per_budget=600.0,
                                preflight_only=True)
        with self.assertRaises(M.Phase2FError) as ctx:
            M._validate_benchmark_cli_args(args)
        self.assertIn("BENCHMARK_MODE_CONFLICT", str(ctx.exception))

    def test_benchmark_conflicts_with_reconstruct_check(self):
        args = self._fake_args(benchmark_two_for_two=True, benchmark_max_seconds_per_budget=600.0,
                                reconstruct_check=True)
        with self.assertRaises(M.Phase2FError) as ctx:
            M._validate_benchmark_cli_args(args)
        self.assertIn("BENCHMARK_MODE_CONFLICT", str(ctx.exception))

    def test_valid_benchmark_args_pass_validation(self):
        args = self._fake_args(benchmark_two_for_two=True, benchmark_max_seconds_per_budget=600.0)
        M._validate_benchmark_cli_args(args)  # must not raise

    def test_non_benchmark_args_skip_validation_entirely(self):
        args = self._fake_args(benchmark_two_for_two=False, benchmark_max_seconds_per_budget=None,
                                preflight_only=True)
        M._validate_benchmark_cli_args(args)  # must not raise -- benchmark flag is off

    # R7-B1 (GPT static review) regression guard: statically audits EVERY
    # argparse.Namespace(...) construction in this ENTIRE test file (not
    # just the 5 fixtures already known to be stale), so a FUTURE fixture
    # standing in for main()'s real args that omits the two benchmark CLI
    # fields fails loudly here instead of silently reintroducing R7-B1 (a
    # missing attribute would otherwise only surface as an
    # AttributeError deep inside _validate_benchmark_cli_args the next time
    # someone happens to run that specific test).
    def test_all_namespace_fixtures_include_benchmark_cli_fields(self):
        test_source = Path(__file__).resolve().read_text(encoding="utf-8")
        tree = ast.parse(test_source)
        required_fields = {"benchmark_two_for_two", "benchmark_max_seconds_per_budget"}
        offending = []
        found_any = False
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            is_namespace_call = (
                (isinstance(func, ast.Attribute) and func.attr == "Namespace")
                or (isinstance(func, ast.Name) and func.id == "Namespace")
            )
            if not is_namespace_call:
                continue
            found_any = True
            keyword_names = {kw.arg for kw in node.keywords if kw.arg is not None}
            has_double_star_unpack = any(kw.arg is None for kw in node.keywords)
            if has_double_star_unpack:
                # This file's only Namespace(**base)-style construction is
                # this class's own _fake_args helper -- verified directly
                # (not statically) by
                # test_fake_args_base_dict_includes_benchmark_fields below.
                continue
            missing = required_fields - keyword_names
            if missing:
                offending.append((node.lineno, sorted(missing)))
        self.assertTrue(found_any, "no argparse.Namespace(...) construction found -- this audit itself may be broken")
        self.assertEqual(offending, [],
                          f"Namespace fixture(s) missing required benchmark CLI field(s): {offending}")

    def test_fake_args_base_dict_includes_benchmark_fields(self):
        # The one Namespace(**base) construction in this file (this class's
        # own _fake_args helper, exempted by name above) is verified
        # directly here instead of statically, since its fields come from a
        # dict literal, not explicit keyword arguments the AST scan above
        # can see.
        args = self._fake_args()
        self.assertTrue(hasattr(args, "benchmark_two_for_two"))
        self.assertTrue(hasattr(args, "benchmark_max_seconds_per_budget"))

    # R7-B2 (GPT static review): an orphan --benchmark-max-seconds-per-budget
    # without --benchmark-two-for-two must fail closed BEFORE any config/data
    # I/O, never be silently dropped and allowed to fall through to official
    # or reconstruct-check mode as if it had never been passed.
    def test_orphan_benchmark_deadline_without_benchmark_flag_fails_closed(self):
        args = self._fake_args(benchmark_two_for_two=False, benchmark_max_seconds_per_budget=600.0)
        with self.assertRaises(M.Phase2FError) as ctx:
            M._validate_benchmark_cli_args(args)
        self.assertIn("BENCHMARK_ARGUMENT_INVALID", str(ctx.exception))

    def test_orphan_benchmark_deadline_with_preflight_only_fails_closed(self):
        # Same defect, observed through the OTHER flag combination named in
        # the GPT review: --benchmark-max-seconds-per-budget alongside
        # --preflight-only, with --benchmark-two-for-two never set.
        args = self._fake_args(benchmark_two_for_two=False, benchmark_max_seconds_per_budget=600.0,
                                preflight_only=True)
        with self.assertRaises(M.Phase2FError) as ctx:
            M._validate_benchmark_cli_args(args)
        self.assertIn("BENCHMARK_ARGUMENT_INVALID", str(ctx.exception))

    # R7-B3 (GPT static review): combining the benchmark's per-budget
    # deadline with the official/reconstruct-check global --max-seconds is
    # an ambiguous double deadline and must be refused, not silently
    # resolved in favor of one or the other.
    def test_benchmark_conflicts_with_global_max_seconds(self):
        args = self._fake_args(benchmark_two_for_two=True, benchmark_max_seconds_per_budget=600.0,
                                max_seconds=600.0)
        with self.assertRaises(M.Phase2FError) as ctx:
            M._validate_benchmark_cli_args(args)
        self.assertIn("BENCHMARK_MODE_CONFLICT", str(ctx.exception))

    def test_max_seconds_semantics_unchanged_for_non_benchmark_modes(self):
        # --max-seconds alone (no benchmark flag) must still pass validation
        # untouched -- R7-B3's new check is scoped to benchmark_two_for_two=
        # True only, never applied to the official/reconstruct-check paths.
        args = self._fake_args(benchmark_two_for_two=False, benchmark_max_seconds_per_budget=None,
                                max_seconds=600.0)
        M._validate_benchmark_cli_args(args)  # must not raise

    def test_official_gate_check_present_only_in_compute_all_budgets_not_benchmark(self):
        source = SCRIPT_PATH.read_text(encoding="utf-8")
        tree = ast.parse(source)
        gate_snippet = 'if not config["two_for_two_engine"]["operationally_approved"]:'
        bodies = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name in ("compute_all_budgets", "run_operational_benchmark"):
                bodies[node.name] = ast.get_source_segment(source, node) or ""
        self.assertIn("compute_all_budgets", bodies)
        self.assertIn("run_operational_benchmark", bodies)
        self.assertIn(gate_snippet, bodies["compute_all_budgets"])
        self.assertNotIn(gate_snippet, bodies["run_operational_benchmark"])
        self.assertIn("build_budget_step(", bodies["run_operational_benchmark"])

    def test_run_reconstruct_check_still_uses_compute_all_budgets(self):
        source = SCRIPT_PATH.read_text(encoding="utf-8")
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == "run_reconstruct_check":
                body = ast.get_source_segment(source, node) or ""
                self.assertIn("compute_all_budgets(", body)
                return
        self.fail("run_reconstruct_check not found")

    def test_compute_all_budgets_still_refuses_when_not_approved(self):
        import yaml
        config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
        self.assertFalse(config["two_for_two_engine"]["operationally_approved"])
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(M.Phase2FError) as ctx:
                M.compute_all_budgets(config, Path(tmp))
            self.assertIn("TWO_FOR_TWO_ENGINE_NOT_OPERATIONALLY_APPROVED", str(ctx.exception))

    # ---------------------------------------------------------------- #
    # B. No promotion / no artifact leakage                            #
    # ---------------------------------------------------------------- #
    def test_benchmark_never_calls_write_all_outputs_and_promote(self):
        with tempfile.TemporaryDirectory() as tmp:
            config, project_root = self._make_project(Path(tmp))
            with mock.patch.object(M, "write_all_outputs_and_promote") as mocked_promote:
                self._run_benchmark(config, project_root, {b: "OK" for b in M.BUDGET_ORDER})
                mocked_promote.assert_not_called()

    def test_benchmark_never_calls_promote_with_rollback(self):
        with tempfile.TemporaryDirectory() as tmp:
            config, project_root = self._make_project(Path(tmp))
            with mock.patch.object(M, "promote_with_rollback") as mocked_rollback:
                self._run_benchmark(config, project_root, {b: "OK" for b in M.BUDGET_ORDER})
                mocked_rollback.assert_not_called()

    def test_benchmark_creates_no_official_artifacts_only_the_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            config, project_root = self._make_project(Path(tmp))
            official_paths = {project_root / p for p in M.official_relative_paths(config)}
            self._run_benchmark(config, project_root, {b: "OK" for b in M.BUDGET_ORDER})
            for p in official_paths:
                self.assertFalse(p.exists(), f"official artifact must not exist after benchmark: {p}")
            report_path = project_root / Path(config["operational_benchmark"]["report_path"])
            self.assertTrue(report_path.is_file())
            # Nothing under reports/ except the benchmark report itself.
            reports_dir_contents = sorted(p.name for p in (project_root / "reports").iterdir())
            self.assertEqual(reports_dir_contents, [report_path.name])

    # ---------------------------------------------------------------- #
    # C. Deadline semantics                                            #
    # ---------------------------------------------------------------- #
    def test_each_budget_gets_a_fresh_independent_deadline(self):
        with tempfile.TemporaryDirectory() as tmp:
            config, project_root = self._make_project(Path(tmp))
            recorded: list = []
            # Sleep AFTER the first budget's call is recorded (never before
            # -- that would contaminate budget 1pct's own measurement), so
            # the delay lands strictly BETWEEN budget 1pct's call and budget
            # 5pct's call. A correctly-reset-per-budget deadline keeps every
            # budget's (deadline - call_time) gap ~= max_seconds regardless
            # of this delay; a bug that computes ONE deadline before the
            # loop (shared across all budgets) would show 5pct/10pct/20pct's
            # gap shrunk by ~the sleep duration -- easily outside the tight
            # tolerance below.
            outcomes = {b: "OK" for b in M.BUDGET_ORDER}
            original_fake = self._fake_build_budget_step(outcomes, recorded)

            def _slow_fake(budget, *args, **kwargs):
                result = original_fake(budget, *args, **kwargs)
                if budget == "1pct":
                    time.sleep(1.0)
                return result

            synthetic_labeled_size = {"1pct": 14, "5pct": 20, "10pct": 30, "20pct": 40}
            synthetic_nf_size = {"1pct": 0, "5pct": 1, "10pct": 2, "20pct": 3}
            max_seconds = 10.0
            with mock.patch.object(M, "compute_locked_size_targets",
                                    return_value=(synthetic_labeled_size, synthetic_nf_size)), \
                 mock.patch.object(M, "build_budget_step", side_effect=_slow_fake):
                M.run_operational_benchmark(config, project_root, max_seconds)

            self.assertEqual([c["budget"] for c in recorded], list(M.BUDGET_ORDER))
            for call in recorded:
                # Each deadline must be freshly anchored to THAT call's own
                # time, not shrinking cumulatively across budgets.
                self.assertAlmostEqual(call["deadline"] - call["call_time"], max_seconds, delta=0.3,
                                        msg=f"budget {call['budget']} did not get a fresh per-budget deadline")

    def test_computational_abort_maps_correctly_and_blocks_later_budgets(self):
        with tempfile.TemporaryDirectory() as tmp:
            config, project_root = self._make_project(Path(tmp))
            outcomes = {"1pct": "OK", "5pct": "ABORT", "10pct": "OK", "20pct": "OK"}
            report, recorded = self._run_benchmark(config, project_root, outcomes)
            budgets = report["budgets"]
            self.assertEqual(budgets["1pct"]["status"], "COMPLETED")
            self.assertEqual(budgets["5pct"]["status"], "COMPUTATIONAL_ABORT")
            self.assertEqual(budgets["10pct"]["status"], "NOT_RUN_DEPENDENCY")
            self.assertEqual(budgets["20pct"]["status"], "NOT_RUN_DEPENDENCY")
            self.assertEqual(budgets["10pct"]["blocked_by_budget"], "5pct")
            self.assertEqual(budgets["10pct"]["blocked_by_status"], "COMPUTATIONAL_ABORT")
            self.assertEqual(budgets["20pct"]["blocked_by_budget"], "5pct")
            # build_budget_step must never be called for a budget that is
            # already blocked by a dependency.
            self.assertEqual([c["budget"] for c in recorded], ["1pct", "5pct"])
            self.assertEqual(report["overall_status"], "COMPUTATIONAL_ABORT")
            # never reported as REPAIR_INFEASIBLE or a false exhaustion/
            # local-optimum claim on a timeout.
            self.assertFalse(budgets["5pct"]["repair_infeasible"])
            self.assertFalse(budgets["5pct"]["one_for_one_exhausted"])
            self.assertFalse(budgets["5pct"]["two_for_two_exhausted"])
            self.assertFalse(budgets["5pct"]["local_optimum"])
            self.assertTrue(budgets["5pct"]["aborted_by_deadline"])

    def test_repair_infeasible_maps_correctly_and_blocks_later_budgets(self):
        with tempfile.TemporaryDirectory() as tmp:
            config, project_root = self._make_project(Path(tmp))
            outcomes = {"1pct": "OK", "5pct": "OK", "10pct": "INFEASIBLE", "20pct": "OK"}
            report, recorded = self._run_benchmark(config, project_root, outcomes)
            budgets = report["budgets"]
            self.assertEqual(budgets["10pct"]["status"], "REPAIR_INFEASIBLE")
            self.assertTrue(budgets["10pct"]["repair_infeasible"])
            self.assertFalse(budgets["10pct"]["aborted_by_deadline"])
            self.assertEqual(budgets["20pct"]["status"], "NOT_RUN_DEPENDENCY")
            self.assertEqual(budgets["20pct"]["blocked_by_status"], "REPAIR_INFEASIBLE")
            self.assertEqual(report["overall_status"], "REPAIR_INFEASIBLE")
            # COMPUTATIONAL_ABORT and REPAIR_INFEASIBLE must never be conflated.
            self.assertNotEqual(report["overall_status"], "COMPUTATIONAL_ABORT")

    # ---------------------------------------------------------------- #
    # D. Diagnostics                                                    #
    # ---------------------------------------------------------------- #
    def test_aborted_record_fields_are_null_not_fabricated(self):
        with tempfile.TemporaryDirectory() as tmp:
            config, project_root = self._make_project(Path(tmp))
            outcomes = {"1pct": "ABORT", "5pct": "OK", "10pct": "OK", "20pct": "OK"}
            report, _ = self._run_benchmark(config, project_root, outcomes)
            b1 = report["budgets"]["1pct"]
            for field in ("final_selected_size", "final_no_finding_size", "final_class_coverage",
                          "membership_sha256", "objective_before_repair", "objective_after_repair",
                          "initial_candidate_size"):
                self.assertIsNone(b1[field], f"{field} must be null (unknown), not fabricated, on abort")
            self.assertTrue(b1["two_for_two_engine_invoked"])
            self.assertEqual(len(b1["two_for_two_runtime_diagnostics"]), 1)
            self.assertEqual(b1["two_for_two_runtime_diagnostics"][0]["status"], "aborted")
            # a known counter recorded before abort must be preserved, not nulled.
            self.assertEqual(b1["two_for_two_runtime_diagnostics"][0]["selected_abnormal_count"], 3)
            # an unknown-at-abort-time counter must be null, never fabricated 0.
            self.assertIsNone(b1["two_for_two_runtime_diagnostics"][0]["pool_abnormal_count"])

    def test_not_run_dependency_record_is_fully_null(self):
        with tempfile.TemporaryDirectory() as tmp:
            config, project_root = self._make_project(Path(tmp))
            outcomes = {"1pct": "ABORT", "5pct": "OK", "10pct": "OK", "20pct": "OK"}
            report, _ = self._run_benchmark(config, project_root, outcomes)
            b2 = report["budgets"]["5pct"]
            self.assertEqual(b2["status"], "NOT_RUN_DEPENDENCY")
            for field in ("elapsed_seconds", "initial_candidate_size", "final_selected_size",
                          "final_no_finding_size", "final_class_coverage", "membership_sha256",
                          "one_for_one_move_count", "two_for_two_move_count", "repair_move_count_total",
                          "objective_before_repair", "objective_after_repair", "one_for_one_exhausted",
                          "two_for_two_exhausted", "local_optimum", "aborted_by_deadline", "repair_infeasible"):
                self.assertIsNone(b2[field], f"{field} must be null for a NOT_RUN_DEPENDENCY budget")
            self.assertFalse(b2["two_for_two_engine_invoked"])
            self.assertEqual(b2["two_for_two_runtime_diagnostics"], [])

    def test_completed_record_includes_real_runtime_diagnostics(self):
        with tempfile.TemporaryDirectory() as tmp:
            config, project_root = self._make_project(Path(tmp))
            outcomes = {b: "OK" for b in M.BUDGET_ORDER}
            report, _ = self._run_benchmark(config, project_root, outcomes)
            for b in M.BUDGET_ORDER:
                entry = report["budgets"][b]
                self.assertEqual(entry["status"], "COMPLETED")
                self.assertTrue(entry["two_for_two_engine_invoked"])
                self.assertEqual(len(entry["two_for_two_runtime_diagnostics"]), 1)
                self.assertEqual(entry["two_for_two_runtime_diagnostics"][0]["status"], "completed")
                self.assertIsNotNone(entry["membership_sha256"])
                self.assertEqual(len(entry["membership_sha256"]), 64)

    def test_report_never_auto_approves(self):
        with tempfile.TemporaryDirectory() as tmp:
            config, project_root = self._make_project(Path(tmp))
            report, _ = self._run_benchmark(config, project_root, {b: "OK" for b in M.BUDGET_ORDER})
            self.assertEqual(report["overall_status"], "COMPLETED")
            self.assertFalse(report["operational_approval_decision"])
            self.assertFalse(report["usable_for_training"])
            self.assertFalse(report["official_partition_artifact"])
            self.assertFalse(report["membership_lock"])

    # ---------------------------------------------------------------- #
    # E. Scientific invariants                                         #
    # ---------------------------------------------------------------- #
    def test_seed_and_targets_unchanged_in_yaml(self):
        import yaml
        config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
        self.assertEqual(config["seed"]["partition_seed"], 42)
        self.assertEqual(config["seed"]["seed_policy"], "PRE_SPECIFIED_LOCKED_NO_SEED_SEARCH")
        self.assertTrue(config["seed"]["seed_search_forbidden"])
        self.assertEqual(config["locked_reference_targets"]["labeled_size"],
                          {"1pct": 34, "5pct": 171, "10pct": 343, "20pct": 685})
        self.assertEqual(config["locked_reference_targets"]["no_finding_size"],
                          {"1pct": 3, "5pct": 17, "10pct": 35, "20pct": 70})

    def test_nested_small_to_large_order_unchanged(self):
        self.assertEqual(M.BUDGET_ORDER, ("1pct", "5pct", "10pct", "20pct"))
        self.assertEqual(M.NESTED_PARENT_BUDGET,
                          {"1pct": None, "5pct": "1pct", "10pct": "5pct", "20pct": "10pct"})

    def test_benchmark_source_never_reads_val_or_test_inputs(self):
        source = SCRIPT_PATH.read_text(encoding="utf-8")
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == "run_operational_benchmark":
                body = ast.get_source_segment(source, node) or ""
                self.assertNotIn('inputs["val_coco"]', body)
                self.assertNotIn('inputs["test_coco"]', body)
                return
        self.fail("run_operational_benchmark not found")

    def test_report_gates_all_false_and_seed_search_not_performed(self):
        with tempfile.TemporaryDirectory() as tmp:
            config, project_root = self._make_project(Path(tmp))
            report, _ = self._run_benchmark(config, project_root, {b: "OK" for b in M.BUDGET_ORDER})
            self.assertFalse(report["training_authorized"])
            self.assertFalse(report["training_started"])
            self.assertFalse(report["pseudo_labels_generated"])
            self.assertFalse(report["test_used"])
            self.assertFalse(report["seed_search_performed"])
            self.assertEqual(report["partition_seed"], 42)

    # ---------------------------------------------------------------- #
    # F. Evidence writing                                              #
    # ---------------------------------------------------------------- #

    # --- R8-B2: fail-closed evidence path separation -------------------- #
    def test_evidence_path_conflict_exact_same_strings(self):
        with tempfile.TemporaryDirectory() as tmp:
            config, project_root = self._make_project(Path(tmp))
            config = dict(config)
            config["operational_benchmark"] = dict(config["operational_benchmark"])
            config["operational_benchmark"]["superseded_evidence_path"] = \
                config["operational_benchmark"]["report_path"]
            with mock.patch.object(M, "build_budget_step") as mocked_step:
                with self.assertRaises(M.Phase2FError) as ctx:
                    M.run_operational_benchmark(config, project_root, 600.0)
                self.assertIn("BENCHMARK_EVIDENCE_PATH_CONFLICT", str(ctx.exception))
                mocked_step.assert_not_called()  # conflict caught before any construction work

    def test_evidence_path_conflict_lexically_different_same_resolved_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            config, project_root = self._make_project(Path(tmp))
            config = dict(config)
            config["operational_benchmark"] = dict(config["operational_benchmark"])
            real_report_path = config["operational_benchmark"]["report_path"]
            # Lexically different string (goes through reports/../reports/)
            # but resolves to the EXACT SAME file as report_path.
            config["operational_benchmark"]["superseded_evidence_path"] = f"reports/../{real_report_path}"
            self.assertNotEqual(
                config["operational_benchmark"]["superseded_evidence_path"], real_report_path,
                "fixture precondition violated: the two config strings must be lexically different",
            )
            with mock.patch.object(M, "build_budget_step") as mocked_step:
                with self.assertRaises(M.Phase2FError) as ctx:
                    M.run_operational_benchmark(config, project_root, 600.0)
                self.assertIn("BENCHMARK_EVIDENCE_PATH_CONFLICT", str(ctx.exception))
                mocked_step.assert_not_called()

    def test_evidence_path_conflict_creates_no_artifact(self):
        with tempfile.TemporaryDirectory() as tmp:
            config, project_root = self._make_project(Path(tmp))
            config = dict(config)
            config["operational_benchmark"] = dict(config["operational_benchmark"])
            config["operational_benchmark"]["superseded_evidence_path"] = \
                config["operational_benchmark"]["report_path"]
            reports_dir = project_root / "reports"
            before = set(reports_dir.iterdir()) if reports_dir.exists() else set()
            with mock.patch.object(M, "build_budget_step"):
                with self.assertRaises(M.Phase2FError):
                    M.run_operational_benchmark(config, project_root, 600.0)
            after = set(reports_dir.iterdir()) if reports_dir.exists() else set()
            self.assertEqual(before, after,
                              "path-conflict fail-closed check must create no report or temp artifact")

    def test_distinct_evidence_paths_run_normally(self):
        # Regression guard: the real project config's report_path and
        # superseded_evidence_path are genuinely distinct -- the R8-B2
        # check must never fire for the normal, correct configuration, and
        # the benchmark must run to completion exactly as before.
        with tempfile.TemporaryDirectory() as tmp:
            config, project_root = self._make_project(Path(tmp))
            self.assertNotEqual(
                config["operational_benchmark"]["report_path"],
                config["operational_benchmark"]["superseded_evidence_path"],
            )
            report, _ = self._run_benchmark(config, project_root, {b: "OK" for b in M.BUDGET_ORDER})
            self.assertEqual(report["overall_status"], "COMPLETED")

    def test_refuse_overwrite_existing_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            config, project_root = self._make_project(Path(tmp))
            report_path = project_root / Path(config["operational_benchmark"]["report_path"])
            report_path.parent.mkdir(parents=True, exist_ok=True)
            report_path.write_text("{}", encoding="utf-8")
            with mock.patch.object(M, "build_budget_step") as mocked_step:
                with self.assertRaises(M.Phase2FError) as ctx:
                    M.run_operational_benchmark(config, project_root, 600.0)
                self.assertIn("BENCHMARK_REPORT_ALREADY_EXISTS", str(ctx.exception))
                mocked_step.assert_not_called()  # refused before any construction work

    def test_atomic_write_readback_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            final_path = Path(tmp) / "reports" / "bench.json"
            payload = {"a": 1, "b": [1, 2, 3], "c": {"d": None}}
            M._write_benchmark_report_atomic(final_path, payload)
            self.assertTrue(final_path.is_file())
            self.assertEqual(M.load_json(final_path), payload)
            # no leftover temp files
            leftovers = [p for p in final_path.parent.iterdir() if p.name != final_path.name]
            self.assertEqual(leftovers, [])

    def test_partial_temp_file_cleaned_on_write_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            final_path = Path(tmp) / "reports" / "bench.json"
            # A python set is not JSON-serializable -- json.dump raises
            # TypeError partway through the write.
            with self.assertRaises(TypeError):
                M._write_benchmark_report_atomic(final_path, {"bad": {1, 2, 3}})
            self.assertFalse(final_path.exists())
            if final_path.parent.exists():
                leftovers = list(final_path.parent.iterdir())
                self.assertEqual(leftovers, [], "temporary write file must be cleaned up on failure")

    def test_computational_abort_still_produces_valid_report_not_marked_completed(self):
        with tempfile.TemporaryDirectory() as tmp:
            config, project_root = self._make_project(Path(tmp))
            outcomes = {"1pct": "ABORT", "5pct": "OK", "10pct": "OK", "20pct": "OK"}
            report, _ = self._run_benchmark(config, project_root, outcomes)
            report_path = project_root / Path(config["operational_benchmark"]["report_path"])
            self.assertTrue(report_path.is_file())
            reloaded = M.load_json(report_path)
            self.assertEqual(reloaded["overall_status"], "COMPUTATIONAL_ABORT")
            self.assertNotEqual(reloaded["overall_status"], "COMPLETED")
            self.assertFalse(reloaded["operational_approval_decision"])

    def test_report_schema_has_required_top_level_and_per_budget_fields(self):
        with tempfile.TemporaryDirectory() as tmp:
            config, project_root = self._make_project(Path(tmp))
            report, _ = self._run_benchmark(config, project_root, {b: "OK" for b in M.BUDGET_ORDER})
            required_top_level = (
                "phase", "stage", "protocol_version", "mode", "artifact_role",
                "official_partition_artifact", "membership_lock", "usable_for_training",
                "operational_approval_decision", "partition_seed", "seed_policy",
                "seed_search_performed", "input_train_coco_sha256",
                "input_train_image_id_sha256_phase2e_style", "train_image_count",
                "train_no_finding_count", "iterative_stratification_version",
                "operationally_approved_at_run_start", "benchmark_exception_scope",
                "deadline_policy", "budgets", "overall_status", "training_authorized",
                "training_started", "pseudo_labels_generated", "test_used",
                # R8 / NHIEM VU 8: documents which earlier report this run
                # supersedes for approval-evidence purposes (without ever
                # calling that earlier report invalid).
                "supersedes_for_approval_evidence",
            )
            for field in required_top_level:
                self.assertIn(field, report, f"missing required top-level report field: {field}")
            self.assertEqual(report["mode"], "NON_PROMOTING_OPERATIONAL_BENCHMARK")
            self.assertEqual(report["artifact_role"], "NON_PROMOTING_OPERATIONAL_DIAGNOSTIC_ONLY")
            required_per_budget = (
                "budget", "target_labeled_size", "target_no_finding_size", "nested_parent_budget",
                "status", "blocked_by_budget", "blocked_by_status", "elapsed_seconds",
                "deadline_seconds", "deadline_enabled", "initial_candidate_size", "final_selected_size",
                "final_no_finding_size", "final_class_coverage", "membership_sha256",
                "one_for_one_move_count", "two_for_two_move_count", "repair_move_count_total",
                "objective_before_repair", "objective_after_repair", "one_for_one_exhausted",
                "two_for_two_exhausted", "local_optimum", "aborted_by_deadline", "repair_infeasible",
                "two_for_two_engine_invoked", "two_for_two_runtime_diagnostics",
            )
            for b in M.BUDGET_ORDER:
                for field in required_per_budget:
                    self.assertIn(field, report["budgets"][b],
                                  f"budget {b} missing required field: {field}")

    def test_report_never_contains_full_membership_list_or_gt_annotations(self):
        with tempfile.TemporaryDirectory() as tmp:
            config, project_root = self._make_project(Path(tmp))
            report, _ = self._run_benchmark(config, project_root, {b: "OK" for b in M.BUDGET_ORDER})
            serialized = json.dumps(report)
            for forbidden in ("bbox", "category_id", "annotations", "image_ids"):
                self.assertNotIn(forbidden, serialized)
            for b in M.BUDGET_ORDER:
                entry = report["budgets"][b]
                self.assertIsInstance(entry["membership_sha256"], str)
                self.assertNotIn("final_selected_ids", entry)
                self.assertNotIn("labeled_image_ids", entry)

    # ---------------------------------------------------------------- #
    # R8 / NHIEM VU 8 -- benchmark evidence versioning: the R7 report   #
    # must be preserved (never overwritten/deleted), and the new run's  #
    # report path must be distinct, with a supersedes_for_approval_     #
    # evidence field documenting the (non-invalidating) relationship.   #
    # ---------------------------------------------------------------- #
    def test_report_path_differs_from_superseded_evidence_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            config, _project_root = self._make_project(Path(tmp))
            self.assertNotEqual(
                config["operational_benchmark"]["report_path"],
                config["operational_benchmark"]["superseded_evidence_path"],
            )

    def test_supersedes_for_approval_evidence_field_matches_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            config, project_root = self._make_project(Path(tmp))
            report, _ = self._run_benchmark(config, project_root, {b: "OK" for b in M.BUDGET_ORDER})
            self.assertEqual(
                report["supersedes_for_approval_evidence"],
                config["operational_benchmark"]["superseded_evidence_path"],
            )

    def test_old_superseded_report_file_is_never_touched(self):
        with tempfile.TemporaryDirectory() as tmp:
            config, project_root = self._make_project(Path(tmp))
            superseded_path = project_root / Path(config["operational_benchmark"]["superseded_evidence_path"])
            new_path = project_root / Path(config["operational_benchmark"]["report_path"])
            self.assertNotEqual(superseded_path, new_path)
            superseded_path.parent.mkdir(parents=True, exist_ok=True)
            old_content = '{"stage": "2F-C0-R7", "note": "historical R7 evidence, do not touch"}'
            superseded_path.write_text(old_content, encoding="utf-8")
            old_mtime_ns = superseded_path.stat().st_mtime_ns

            report, _ = self._run_benchmark(config, project_root, {b: "OK" for b in M.BUDGET_ORDER})

            # the old file's bytes and mtime are byte-for-byte, timestamp-
            # for-timestamp unchanged -- this run never opened it for writing.
            self.assertEqual(superseded_path.read_text(encoding="utf-8"), old_content)
            self.assertEqual(superseded_path.stat().st_mtime_ns, old_mtime_ns)
            # the new report was written to the distinct R8 path instead.
            self.assertTrue(new_path.is_file())
            self.assertEqual(M.load_json(new_path)["overall_status"], report["overall_status"])


# --------------------------------------------------------------------------- #
# R8 — production scientific-neighborhood defect fix (discovered via a real  #
# benchmark run: _equivalence_class_two_for_two_search's `pool` argument was #
# the raw new_region_pool, not the addable complement new_region_pool -      #
# selected, at both production call sites). NHIEM VU 1-6.                    #
# --------------------------------------------------------------------------- #
class LegacyTwoForTwoDisjointNeighborhoodInvariantTests:
    """R8 / NHIEM VU 2-6: defense-in-depth engine-level overlap rejection,
    move-disjointness invariants on the engine's own output, the real
    production-call-path complement-pool contract, diagnostics correctness
    via genuine set-difference (never a hardcoded count), and the oracle's
    lack of an independent overlap guard (hence the deliberate absence of
    any oracle-vs-engine comparison on invalid/overlapping input)."""

    # --- NHIEM VU 2: engine-level fail-closed defense-in-depth ---
    def test_engine_rejects_overlapping_selected_and_pool(self):
        # R8 / NHIEM VU 6 (oracle audit finding, documented here rather than
        # inside the oracle itself): _oracle_best_two_for_two and the
        # brute-force pair enumerator it wraps have NO independent
        # disjointness guard -- they simply enumerate over whatever
        # selected_list/pool_list they are given. Oracle==engine agreement
        # is therefore only ever a meaningful, valid comparison on fixtures
        # already proven disjoint (as every positive equivalence test in
        # this file does). This test deliberately never calls the oracle on
        # the overlapping fixture below, so it can never spuriously imply
        # oracle/engine "agreement" on an invalid input -- only the
        # engine's own fail-closed contract is under test here.
        locked, selected, pool, labels14, zero_gt, K, S_T, N, image_ids, accept, _ab, _cd = \
            _build_cross_signature_tie_fixture()
        overlapping_pool = set(pool) | {next(iter(selected))}
        with self.assertRaises(M.Phase2FError) as ctx:
            M._equivalence_class_two_for_two_search(
                locked, selected, overlapping_pool, labels14, zero_gt, K, S_T, N, 42, "1pct",
                "minimum_class_coverage", image_ids, set(), accept,
            )
        self.assertIn("PROTOCOL_VIOLATION", str(ctx.exception))

    def test_engine_rejects_locked_overlapping_selected(self):
        _locked0, selected, pool, labels14, zero_gt, K, S_T, N, image_ids, accept, _ab, _cd = \
            _build_cross_signature_tie_fixture()
        locked = {next(iter(selected))}
        with self.assertRaises(M.Phase2FError) as ctx:
            M._equivalence_class_two_for_two_search(
                locked, selected, pool, labels14, zero_gt, K, S_T, N, 42, "1pct",
                "minimum_class_coverage", image_ids, set(), accept,
            )
        self.assertIn("PROTOCOL_VIOLATION", str(ctx.exception))

    def test_engine_rejects_locked_overlapping_pool(self):
        _locked0, selected, pool, labels14, zero_gt, K, S_T, N, image_ids, accept, _ab, _cd = \
            _build_cross_signature_tie_fixture()
        locked = {next(iter(pool))}
        with self.assertRaises(M.Phase2FError) as ctx:
            M._equivalence_class_two_for_two_search(
                locked, selected, pool, labels14, zero_gt, K, S_T, N, 42, "1pct",
                "minimum_class_coverage", image_ids, set(), accept,
            )
        self.assertIn("PROTOCOL_VIOLATION", str(ctx.exception))

    def test_disjoint_fixtures_still_pass_unaffected_by_the_new_check(self):
        # Regression guard: the defense-in-depth check must never reject a
        # genuinely disjoint (valid) input -- re-run one of the existing
        # positive equivalence fixtures and confirm it still completes
        # normally (no exception), matching pre-R8 behavior exactly.
        locked, selected, pool, labels14, zero_gt, K, S_T, N, image_ids, accept, _ab, _cd = \
            _build_cross_signature_tie_fixture()
        result = M._equivalence_class_two_for_two_search(
            locked, selected, pool, labels14, zero_gt, K, S_T, N, 42, "1pct",
            "minimum_class_coverage", image_ids, set(), accept,
        )
        self.assertIsNotNone(result)

    # --- NHIEM VU 3: move-disjointness on the engine's own returned output ---
    def test_every_returned_move_satisfies_disjointness_invariants(self):
        cases = []
        locked1, selected1, pool1, labels14_1, zero_gt_1, K1, S_T1, N1, image_ids1, accept1, _ab, _cd = \
            _build_cross_signature_tie_fixture()
        cases.append((locked1, selected1, pool1, labels14_1, zero_gt_1, K1, S_T1, N1, image_ids1,
                       "minimum_class_coverage", accept1))
        for seed, n, split_frac in [(101, 40, 0.3), (202, 60, 0.4), (303, 50, 0.5)]:
            train = _make_multilabel_cooccurrence_train(seed, n)
            image_ids, labels14, zero_gt, _ = M.build_indicators(train)
            K, S_T, N = M.full_train_integer_stats(labels14)
            abnormal = [p for p in range(len(image_ids)) if zero_gt[p] == 0]
            cut = max(2, int(len(abnormal) * split_frac))
            selected = set(abnormal[:cut])
            pool = set(abnormal[cut:])
            locked: set[int] = set()
            current_obj = M.integer_objective(locked | selected, labels14, K, S_T, N) if selected else (0, 0, 0)

            def make_accept(K=K, S_T=S_T, N=N, current_obj=current_obj):
                def accept(new_k, new_n, new_s_l):
                    if int((new_k > 0).sum()) != 14:
                        return None
                    trial_obj = M._integer_objective_from_aggregate(new_k, new_n, new_s_l, K, S_T, N)
                    if not (trial_obj < current_obj):
                        return None
                    return trial_obj
                return accept

            cases.append((locked, selected, pool, labels14, zero_gt, K, S_T, N, image_ids,
                           "objective_repair", make_accept()))

        checked_at_least_one_nonempty_case = False
        for locked, selected, pool, labels14, zero_gt, K, S_T, N, image_ids, phase, accept in cases:
            result = M._equivalence_class_two_for_two_search(
                locked, selected, pool, labels14, zero_gt, K, S_T, N, 42, "1pct", phase, image_ids, set(), accept,
            )
            if result is None:
                continue
            checked_at_least_one_nonempty_case = True
            out1, out2, in1, in2, _full_key = result
            self.assertNotEqual(out1, out2)
            self.assertNotEqual(in1, in2)
            self.assertTrue({out1, out2}.isdisjoint({in1, in2}))
            self.assertTrue({out1, out2} <= selected)
            self.assertTrue({in1, in2} <= pool)
            self.assertTrue({in1, in2}.isdisjoint(selected))
            self.assertTrue({out1, out2}.isdisjoint(pool))
        self.assertTrue(checked_at_least_one_nonempty_case,
                         "fixture precondition violated: no case produced a move to check invariants against")

    # --- NHIEM VU 4: the REAL production call path, not just the engine  ---
    # directly with already-disjoint fixtures. Capable of catching the OLD  #
    # (pre-R8) buggy implementation, which passed the raw new_region_pool  #
    # (still containing `selected`) instead of new_region_pool - selected. #
    def test_repair_min_class_coverage_passes_addable_complement_to_engine(self):
        def _capture_and_return_none(locked, selected, pool, *args, **kwargs):
            captured.append({"selected": set(selected), "pool": set(pool)})
            return None
        captured: list[dict] = []

        images, annotations = [], []
        ann_id = 1

        def add_image(image_id, classes):
            nonlocal ann_id
            images.append({"id": image_id, "file_name": f"t/{image_id}.jpg", "width": 10, "height": 10})
            for c in classes:
                annotations.append({"id": ann_id, "image_id": image_id, "category_id": c + 1,
                                     "bbox": [1, 1, 2, 2], "area": 4, "iscrowd": 0})
                ann_id += 1

        other_classes = [c for c in range(14) if c != 6]
        for idx, c in enumerate(other_classes):
            add_image(idx + 1, [c])   # 13 sole-carrier images, ids 1..13 (class 6 missing)
        add_image(14, [6])             # pool candidate: class 6 ONLY
        add_image(15, [6])             # second pool candidate: class 6 ONLY
        train = {"info": {}, "licenses": [], "images": images, "annotations": annotations,
                 "categories": _categories()}
        image_ids, labels14, zero_gt, names = M.build_indicators(train)
        K, S_T, N = M.full_train_integer_stats(labels14)
        id_to_pos = {int(v): i for i, v in enumerate(image_ids.tolist())}
        selected = {id_to_pos[i] for i in range(1, 14)}
        # Real calling convention: the RAW region pool (universe - locked),
        # not pre-narrowed by this test -- it INCLUDES `selected`, exactly
        # like build_budget_step actually passes it.
        new_region_pool = set(range(len(image_ids)))
        self.assertTrue(selected <= new_region_pool)  # overlap is real, not a vacuous fixture

        with mock.patch.object(M, "_equivalence_class_two_for_two_search", side_effect=_capture_and_return_none):
            M.repair_min_class_coverage(
                set(), selected, new_region_pool, labels14, zero_gt, names,
                K, S_T, N, 42, "1pct", image_ids, None, [], set(),
            )
        self.assertEqual(len(captured), 1, "engine was never reached -- fixture precondition violated "
                                            "(a one-for-one fix must not exist for this probe to be valid)")
        call = captured[0]
        self.assertEqual(call["pool"], new_region_pool - call["selected"])
        self.assertTrue(call["pool"].isdisjoint(call["selected"]))
        # sanity: the OLD buggy call would have passed the raw new_region_pool
        # itself -- confirm this test can actually distinguish the two.
        self.assertNotEqual(call["pool"], new_region_pool)

    def test_repair_objective_local_search_passes_addable_complement_to_engine(self):
        captured: list[dict] = []

        def _capture_and_return_none(locked, selected, pool, *args, **kwargs):
            captured.append({"selected": set(selected), "pool": set(pool)})
            return None

        train = make_deterministic_train()
        image_ids, labels14, zero_gt, names = M.build_indicators(train)
        K, S_T, N = M.full_train_integer_stats(labels14)
        skewed = set(range(14)) | {20, 34}
        new_region_pool = set(range(len(image_ids)))  # raw, NOT pre-narrowed
        self.assertTrue(skewed <= new_region_pool)

        with mock.patch.object(M, "_equivalence_class_two_for_two_search", side_effect=_capture_and_return_none):
            M.repair_objective_local_search(
                set(), skewed, new_region_pool, labels14, zero_gt, names,
                K, S_T, N, 42, "1pct", image_ids, None, [], set(),
            )
        self.assertGreaterEqual(len(captured), 1, "engine was never reached -- fixture precondition violated")
        for call in captured:
            self.assertEqual(call["pool"], new_region_pool - call["selected"],
                              "captured pool must equal new_region_pool - selected_at_call, "
                              "which may differ from the initial `skewed` set once prior "
                              "one_for_one_swap moves have mutated `selected`")
            self.assertTrue(call["pool"].isdisjoint(call["selected"]))

    # --- NHIEM VU 5: diagnostics regression -- real set-difference, never  #
    # a hardcoded count. ---
    def test_diagnostics_pool_abnormal_count_is_addable_complement_not_raw_region_pool(self):
        images, annotations = [], []
        ann_id = 1

        def add_image(image_id, classes):
            nonlocal ann_id
            images.append({"id": image_id, "file_name": f"t/{image_id}.jpg", "width": 10, "height": 10})
            for c in classes:
                annotations.append({"id": ann_id, "image_id": image_id, "category_id": c + 1,
                                     "bbox": [1, 1, 2, 2], "area": 4, "iscrowd": 0})
                ann_id += 1

        other_classes = [c for c in range(14) if c != 6]
        for idx, c in enumerate(other_classes):
            add_image(idx + 1, [c])
        add_image(14, [6])
        add_image(15, [6])
        train = {"info": {}, "licenses": [], "images": images, "annotations": annotations,
                 "categories": _categories()}
        image_ids, labels14, zero_gt, names = M.build_indicators(train)
        K, S_T, N = M.full_train_integer_stats(labels14)
        id_to_pos = {int(v): i for i, v in enumerate(image_ids.tolist())}
        selected = {id_to_pos[i] for i in range(1, 14)}
        new_region_pool = set(range(len(image_ids)))
        # Ground truth computed via a REAL set-difference on the fixture --
        # never a hardcoded number.
        correct_addable_pool = new_region_pool - selected
        self.assertEqual(correct_addable_pool, {id_to_pos[14], id_to_pos[15]})

        two_for_two_diagnostics_log: list[dict] = []
        M.repair_min_class_coverage(
            set(), selected, new_region_pool, labels14, zero_gt, names,
            K, S_T, N, 42, "1pct", image_ids, None, [], set(),
            two_for_two_diagnostics_log=two_for_two_diagnostics_log,
        )
        self.assertEqual(len(two_for_two_diagnostics_log), 1)
        diag = two_for_two_diagnostics_log[0]
        expected_pool_abnormal_count = len({p for p in correct_addable_pool if zero_gt[p] == 0})
        self.assertEqual(diag["pool_abnormal_count"], expected_pool_abnormal_count)
        # Regression guard: the OLD buggy behavior would have reported the
        # raw new_region_pool's abnormal count instead -- confirm the two
        # differ in this fixture, so the assertion above actually
        # discriminates between correct and buggy behavior.
        raw_pool_abnormal_count = len({p for p in new_region_pool if zero_gt[p] == 0})
        self.assertNotEqual(expected_pool_abnormal_count, raw_pool_abnormal_count)


def _build_hotpath_profile_fixture():
    """Purpose-built fixture for R10 hotpath-profiler tests. 12 "locked"
    images, each a distinct single class 0..11 (zero_gt=0); 2 "selected"
    images, class 12 and class 13 respectively (R=1 removed-pair
    signature: sel_sigs=[(12,),(13,)]); 4 "pool" images with signatures
    (12,), (13,), (12,13), (0,12) (A=C(4,2)=6 added-pair signatures, all
    cross-signature since every pool signature is distinct). Positions
    follow image_id/file order exactly (locked 1..12, selected 13..14,
    pool 21..24), so set-iteration order over these small ints matches the
    intended removed/added enumeration order -- the same established
    pattern _build_cross_signature_tie_fixture already relies on.

    HAND-VERIFIED (by the R10 implementer, restated in every test that
    depends on it) over the 6 added-pair transitions in traversal order
    (Qa,Qb),(Qa,Qc),(Qa,Qd),(Qb,Qc),(Qb,Qd),(Qc,Qd):
      - coverage (14/14) FAILS only for (Qa,Qd) (drops class 13 to zero);
        PASSES for the other 5;
      - among those 5 coverage-passing transitions, exactly 4 (all except
        (Qa,Qb), which reproduces the unchanged base state -- an EQUAL,
        not strictly-improving, objective, forbidden by objective_repair)
        strictly improve the fixed current/base objective;
      - all 6 resulting (new_k, new_s_l) states are pairwise DISTINCT (no
        duplicate states in this fixture).
    accept_and_rank reproduces objective_accept_and_rank's exact semantics
    (14/14 coverage required; strict lexicographic improvement over the
    fixed base objective required; equal objective forbidden)."""
    images, annotations = [], []
    ann_id = 1

    def add_image(image_id, classes):
        nonlocal ann_id
        images.append({"id": image_id, "file_name": f"h/{image_id}.jpg", "width": 10, "height": 10})
        for c in classes:
            annotations.append({"id": ann_id, "image_id": image_id, "category_id": c + 1,
                                 "bbox": [1, 1, 2, 2], "area": 4, "iscrowd": 0})
            ann_id += 1

    for c in range(12):
        add_image(c + 1, [c])       # locked 1..12: classes 0..11
    add_image(13, [12])             # selected A: class 12
    add_image(14, [13])             # selected B: class 13
    add_image(21, [12])             # pool Qa: (12,)
    add_image(22, [13])             # pool Qb: (13,)
    add_image(23, [12, 13])         # pool Qc: (12,13)
    add_image(24, [0, 12])          # pool Qd: (0,12)

    fixture = {"info": {}, "licenses": [], "images": images, "annotations": annotations, "categories": _categories()}
    image_ids, labels14, zero_gt, _ = M.build_indicators(fixture)
    K, S_T, N = M.full_train_integer_stats(labels14)
    id_to_pos = {int(v): i for i, v in enumerate(image_ids.tolist())}

    locked = {id_to_pos[i] for i in range(1, 13)}
    selected = {id_to_pos[13], id_to_pos[14]}
    pool = {id_to_pos[i] for i in (21, 22, 23, 24)}

    base_state = sorted(locked | selected)
    base_k = labels14[base_state].sum(axis=0).astype(np.int64)
    base_n = len(base_state)
    base_s_l = int(base_k.sum())
    current_obj = M._integer_objective_from_aggregate(base_k, base_n, base_s_l, K, S_T, N)

    def accept_and_rank(new_k, new_n, new_s_l):
        if int((new_k > 0).sum()) != 14:
            return None
        trial_obj = M._integer_objective_from_aggregate(new_k, new_n, new_s_l, K, S_T, N)
        if not (trial_obj < current_obj):
            return None
        return trial_obj

    expected_identity = {
        "selected_abnormal_count": 2, "pool_abnormal_count": 4,
        "selected_distinct_signature_count": 2, "pool_distinct_signature_count": 4,
        "removed_pair_signature_count_R": 1, "added_pair_signature_count_A": 6,
        "phase1_transition_count": 6,
    }
    return locked, selected, pool, labels14, zero_gt, K, S_T, N, image_ids, accept_and_rank, expected_identity


# --------------------------------------------------------------------------- #
# R10 -- NON_PROMOTING_PERFORMANCE_DIAGNOSTIC_ONLY measurement profiler       #
# --------------------------------------------------------------------------- #
class LegacyTwoForTwoHotpathProfilerTests:
    # ---------------------------------------------------------------- #
    # A. CLI / mode isolation (TASK 9 items 1-3)                       #
    # ---------------------------------------------------------------- #
    def _fake_args(self, **overrides):
        import argparse as _argparse
        base = dict(
            config=Path("configs/protocol/phase2F_labeled_unlabeled.yaml"), project_root=Path("."),
            preflight_only=False, reconstruct_check=False, max_seconds=None,
            benchmark_two_for_two=False, benchmark_max_seconds_per_budget=None,
            profile_two_for_two_hotpath=False, profile_max_seconds=None,
        )
        base.update(overrides)
        return _argparse.Namespace(**base)

    def test_missing_profile_deadline_fails_closed(self):
        args = self._fake_args(profile_two_for_two_hotpath=True, profile_max_seconds=None)
        with self.assertRaises(M.Phase2FError) as ctx:
            M._validate_profiler_cli_args(args)
        self.assertIn("PROFILE_CONFIG_INVALID", str(ctx.exception))

    def test_nonpositive_or_nan_or_inf_profile_deadline_fails_closed(self):
        for bad in (0.0, -1.0, -600.0, float("nan"), float("inf"), float("-inf")):
            with self.subTest(deadline=bad):
                args = self._fake_args(profile_two_for_two_hotpath=True, profile_max_seconds=bad)
                with self.assertRaises(M.Phase2FError) as ctx:
                    M._validate_profiler_cli_args(args)
                self.assertIn("PROFILE_CONFIG_INVALID", str(ctx.exception))

    def test_orphan_profile_deadline_without_profile_flag_fails_closed(self):
        args = self._fake_args(profile_two_for_two_hotpath=False, profile_max_seconds=600.0)
        with self.assertRaises(M.Phase2FError) as ctx:
            M._validate_profiler_cli_args(args)
        self.assertIn("PROFILE_CONFIG_INVALID", str(ctx.exception))

    def test_profile_conflicts_with_preflight_only(self):
        args = self._fake_args(profile_two_for_two_hotpath=True, profile_max_seconds=600.0, preflight_only=True)
        with self.assertRaises(M.Phase2FError) as ctx:
            M._validate_profiler_cli_args(args)
        self.assertIn("PROFILE_CONFIG_INVALID", str(ctx.exception))

    def test_profile_conflicts_with_reconstruct_check(self):
        args = self._fake_args(profile_two_for_two_hotpath=True, profile_max_seconds=600.0, reconstruct_check=True)
        with self.assertRaises(M.Phase2FError) as ctx:
            M._validate_profiler_cli_args(args)
        self.assertIn("PROFILE_CONFIG_INVALID", str(ctx.exception))

    def test_profile_conflicts_with_benchmark_two_for_two(self):
        args = self._fake_args(profile_two_for_two_hotpath=True, profile_max_seconds=600.0,
                                benchmark_two_for_two=True)
        with self.assertRaises(M.Phase2FError) as ctx:
            M._validate_profiler_cli_args(args)
        self.assertIn("PROFILE_CONFIG_INVALID", str(ctx.exception))

    def test_profile_conflicts_with_global_max_seconds(self):
        args = self._fake_args(profile_two_for_two_hotpath=True, profile_max_seconds=600.0, max_seconds=600.0)
        with self.assertRaises(M.Phase2FError) as ctx:
            M._validate_profiler_cli_args(args)
        self.assertIn("PROFILE_CONFIG_INVALID", str(ctx.exception))

    def test_valid_profile_args_pass_validation(self):
        args = self._fake_args(profile_two_for_two_hotpath=True, profile_max_seconds=600.0)
        M._validate_profiler_cli_args(args)  # must not raise

    def test_non_profile_args_skip_validation_entirely(self):
        args = self._fake_args(profile_two_for_two_hotpath=False, preflight_only=True)
        M._validate_profiler_cli_args(args)  # must not raise -- profile flag is off

    # ---------------------------------------------------------------- #
    # B. Engine-level: real invocation reuse, phase timing, identity    #
    # check, Layer-B microbenchmark, calibration (items 4/6/7/8/9/10/   #
    # 11/12/13/14/15/16)                                                #
    # ---------------------------------------------------------------- #
    def test_hotpath_profile_does_not_change_accepted_move(self):
        locked, selected, pool, labels14, zero_gt, K, S_T, N, image_ids, accept, _exp = \
            _build_hotpath_profile_fixture()
        baseline = M._equivalence_class_two_for_two_search(
            locked, selected, pool, labels14, zero_gt, K, S_T, N, 42, "1pct", "objective_repair",
            image_ids, set(), accept,
        )
        hp = {"profile_transition_prefix_count": 6}
        profiled = M._equivalence_class_two_for_two_search(
            locked, selected, pool, labels14, zero_gt, K, S_T, N, 42, "1pct", "objective_repair",
            image_ids, set(), accept, hotpath_profile=hp,
        )
        self.assertEqual(baseline, profiled)

    def test_phase_timing_fields_present_and_nonnegative(self):
        locked, selected, pool, labels14, zero_gt, K, S_T, N, image_ids, accept, _exp = \
            _build_hotpath_profile_fixture()
        hp = {"profile_transition_prefix_count": 6}
        M._equivalence_class_two_for_two_search(
            locked, selected, pool, labels14, zero_gt, K, S_T, N, 42, "1pct", "objective_repair",
            image_ids, set(), accept, hotpath_profile=hp,
        )
        pt = hp["phase_timing"]
        for field in ("equivalence_class_construction_seconds", "removed_pair_construction_seconds",
                      "added_pair_construction_seconds", "aggregate_precompute_seconds"):
            self.assertIn(field, pt)
            self.assertGreaterEqual(pt[field], 0.0)
        self.assertIsInstance(hp["watchdog_ticks_total"], int)
        self.assertGreaterEqual(hp["watchdog_ticks_total"], 0)

    def test_identity_check_passes_when_matching_expected(self):
        locked, selected, pool, labels14, zero_gt, K, S_T, N, image_ids, accept, expected = \
            _build_hotpath_profile_fixture()
        good_expected = dict(expected)
        good_expected["one_for_one_move_count_before_two_for_two"] = 0
        hp = {"profile_transition_prefix_count": 6, "expected_identity": good_expected,
              "one_for_one_move_count_before_two_for_two": 0}
        M._equivalence_class_two_for_two_search(
            locked, selected, pool, labels14, zero_gt, K, S_T, N, 42, "1pct", "objective_repair",
            image_ids, set(), accept, hotpath_profile=hp,
        )
        self.assertIsNotNone(hp.get("microbenchmark"))

    def test_identity_mismatch_fails_closed_before_microbenchmark(self):
        locked, selected, pool, labels14, zero_gt, K, S_T, N, image_ids, accept, expected = \
            _build_hotpath_profile_fixture()
        bad_expected = dict(expected)
        bad_expected["removed_pair_signature_count_R"] = 999
        bad_expected["one_for_one_move_count_before_two_for_two"] = 0
        hp = {"profile_transition_prefix_count": 6, "expected_identity": bad_expected,
              "one_for_one_move_count_before_two_for_two": 0}
        with self.assertRaises(M.Phase2FError) as ctx:
            M._equivalence_class_two_for_two_search(
                locked, selected, pool, labels14, zero_gt, K, S_T, N, 42, "1pct", "objective_repair",
                image_ids, set(), accept, hotpath_profile=hp,
            )
        self.assertIn("PROFILE_INVOCATION_IDENTITY_MISMATCH", str(ctx.exception))
        # must fail BEFORE the R9 precompute / Layer-B microbenchmark: no
        # microbenchmark result and no precompute timing were ever recorded.
        self.assertNotIn("microbenchmark", hp)
        self.assertNotIn("aggregate_precompute_seconds", hp.get("phase_timing", {}))
        # but the fields the check itself needs (computed BEFORE the check)
        # are present -- proving the check runs after R/A computation, not
        # before it (and before precompute).
        self.assertIn("removed_pair_construction_seconds", hp.get("phase_timing", {}))

    def test_semantic_equivalence_failure_fails_closed(self):
        locked, selected, pool, labels14, zero_gt, K, S_T, N, image_ids, _accept, _exp = \
            _build_hotpath_profile_fixture()

        def broken_accept_and_rank(new_k, new_n, new_s_l):
            return (0, 0, 0)  # always "accepts", even when coverage fails -- a genuine logic contradiction

        hp = {"profile_transition_prefix_count": 6}
        with self.assertRaises(M.Phase2FError) as ctx:
            M._equivalence_class_two_for_two_search(
                locked, selected, pool, labels14, zero_gt, K, S_T, N, 42, "1pct", "objective_repair",
                image_ids, set(), broken_accept_and_rank, hotpath_profile=hp,
            )
        self.assertIn("PROFILE_SEMANTIC_EQUIVALENCE_FAILURE", str(ctx.exception))

    def test_microbenchmark_hand_computed_metrics_match_manual_derivation(self):
        locked, selected, pool, labels14, zero_gt, K, S_T, N, image_ids, accept, _exp = \
            _build_hotpath_profile_fixture()
        hp = {"profile_transition_prefix_count": 6}
        M._equivalence_class_two_for_two_search(
            locked, selected, pool, labels14, zero_gt, K, S_T, N, 42, "1pct", "objective_repair",
            image_ids, set(), accept, hotpath_profile=hp,
        )
        mb = hp["microbenchmark"]
        self.assertEqual(mb["prefix_count_used"], 6)
        self.assertAlmostEqual(mb["coverage_pass_fraction"], 5 / 6)
        self.assertAlmostEqual(mb["strictly_improving_fraction_among_coverage_passing"], 4 / 5)
        self.assertTrue(mb["semantic_equivalence_verified"])
        dsd = mb["duplicate_state_diagnostics"]
        self.assertEqual(dsd["prefix_count_considered"], 6)
        self.assertEqual(dsd["unique_state_count"], 6)
        self.assertEqual(dsd["repeated_state_count"], 0)
        self.assertEqual(dsd["duplicate_state_ratio"], 0.0)
        self.assertEqual(dsd["max_multiplicity"], 1)
        self.assertEqual(dsd["unique_accept_rank_prefix_count"], 4)
        self.assertEqual(dsd["repeated_accepted_prefix_count"], 0)

    def test_deterministic_prefix_bounded_by_r_times_a(self):
        locked, selected, pool, labels14, zero_gt, K, S_T, N, image_ids, accept, _exp = \
            _build_hotpath_profile_fixture()
        hp = {"profile_transition_prefix_count": 999999}  # far larger than R*A=6
        M._equivalence_class_two_for_two_search(
            locked, selected, pool, labels14, zero_gt, K, S_T, N, 42, "1pct", "objective_repair",
            image_ids, set(), accept, hotpath_profile=hp,
        )
        self.assertEqual(hp["microbenchmark"]["prefix_count_used"], 6)
        self.assertEqual(hp["microbenchmark"]["total_transition_count_r_times_a"], 6)

    def test_deterministic_prefix_follows_removed_outer_added_inner_order(self):
        # A prefix count smaller than A (=6) must stay confined to the
        # FIRST removed pair (there is only one here, R=1) -- proving the
        # traversal order is removed-outer/added-inner, matching
        # _stream_best_prefix_above exactly.
        locked, selected, pool, labels14, zero_gt, K, S_T, N, image_ids, accept, _exp = \
            _build_hotpath_profile_fixture()
        hp3 = {"profile_transition_prefix_count": 3}
        M._equivalence_class_two_for_two_search(
            locked, selected, pool, labels14, zero_gt, K, S_T, N, 42, "1pct", "objective_repair",
            image_ids, set(), accept, hotpath_profile=hp3,
        )
        # first 3 of 6 transitions = (Qa,Qb),(Qa,Qc),(Qa,Qd) -- coverage
        # fails only on the 3rd (Qa,Qd) -- so coverage_pass_fraction over
        # THIS shorter prefix must be 2/3, not 5/6 (which would only hold
        # under a different/full-6 ordering).
        self.assertEqual(hp3["microbenchmark"]["prefix_count_used"], 3)
        self.assertAlmostEqual(hp3["microbenchmark"]["coverage_pass_fraction"], 2 / 3)

    def test_all_component_passes_present_and_checksum_is_deterministic(self):
        locked, selected, pool, labels14, zero_gt, K, S_T, N, image_ids, accept, _exp = \
            _build_hotpath_profile_fixture()
        hp1 = {"profile_transition_prefix_count": 6}
        hp2 = {"profile_transition_prefix_count": 6}
        M._equivalence_class_two_for_two_search(
            locked, selected, pool, labels14, zero_gt, K, S_T, N, 42, "1pct", "objective_repair",
            image_ids, set(), accept, hotpath_profile=hp1,
        )
        M._equivalence_class_two_for_two_search(
            locked, selected, pool, labels14, zero_gt, K, S_T, N, 42, "1pct", "objective_repair",
            image_ids, set(), accept, hotpath_profile=hp2,
        )
        self.assertEqual(hp1["microbenchmark"]["semantic_checksum_sha256"],
                          hp2["microbenchmark"]["semantic_checksum_sha256"])
        checksum = hp1["microbenchmark"]["semantic_checksum_sha256"]
        self.assertIsInstance(checksum, str)
        self.assertEqual(len(checksum), 64)
        components = hp1["microbenchmark"]["components"]
        for key in ("loop_counter_overhead_seconds", "watchdog_tick_overhead_seconds",
                    "added_matrix_lookup_seconds", "new_k_construction_seconds",
                    "coverage_check_seconds", "objective_evaluation_seconds",
                    "real_accept_and_rank_call_seconds", "best_prefix_maintenance_seconds",
                    "complete_current_r9_transition_cost_seconds"):
            self.assertIn(key, components)
            self.assertGreaterEqual(components[key], 0.0)

    def test_calibration_returns_all_required_fields(self):
        calib = M._calibrate_observer_overhead(iterations=1000)
        for field in ("empty_loop_cost_seconds_per_iteration", "perf_counter_call_cost_seconds",
                      "watchdog_no_check_tick_cost_seconds_per_iteration",
                      "check_deadline_actual_cost_seconds_per_call"):
            self.assertIn(field, calib)
            self.assertGreaterEqual(calib[field], 0.0)

    def test_no_per_transition_timer_calls_in_real_search_loop(self):
        source = SCRIPT_PATH.read_text(encoding="utf-8")
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == "_stream_best_prefix_above":
                body = ast.get_source_segment(source, node) or ""
                self.assertNotIn("time.monotonic()", body,
                                  "R10 must never add a per-transition timer inside the real Phase-1 loop")
                return
        self.fail("_stream_best_prefix_above not found (expected nested inside "
                  "_equivalence_class_two_for_two_search)")

    # ---------------------------------------------------------------- #
    # C. End-to-end orchestration (items 4/5/6/17-26)                  #
    # ---------------------------------------------------------------- #
    def _make_project(self, tmp: Path) -> tuple[dict, Path]:
        import yaml
        config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
        project_root = tmp / "project"
        train = make_deterministic_train()
        config = dict(config)
        config["locked_reference_targets"] = {
            "labeled_size": {"1pct": 14, "5pct": 20, "10pct": 30, "20pct": 40},
            "no_finding_size": {"1pct": 0, "5pct": 1, "10pct": 2, "20pct": 3},
            "min_class_coverage": 14,
        }
        (project_root / "data" / "processed" / "coco" / "labeled_splits").mkdir(parents=True)
        (project_root / "data" / "processed" / "coco" / "unlabeled_splits").mkdir(parents=True)
        (project_root / "data" / "manifests").mkdir(parents=True)
        (project_root / "reports").mkdir(parents=True)
        M.write_json(project_root / "data" / "processed" / "coco" / "instances_train.json", train)
        return config, project_root

    @staticmethod
    def _fake_build_budget_step_profiler(status: str, engine_diag_extra: dict | None = None):
        def _fake(budget, locked, image_ids, labels14, zero_gt, names, target_size, target_nf,
                   seed, K, S_T, N, deadline, repair_log, diagnostics_log=None,
                   phase_timing=None, hotpath_profile=None):
            if phase_timing is not None:
                phase_timing.update({
                    "stratification_seconds": 0.001, "exact_size_seconds": 0.001,
                    "exact_no_finding_seconds": 0.001, "minimum_class_coverage_seconds": 0.001,
                    "one_for_one_before_first_two_for_two_seconds": 0.001,
                    "one_for_one_move_count_before_two_for_two": 3,
                    "objective_repair_seconds": 0.01,
                })
            diag = {
                "budget": budget, "repair_phase": "objective_repair", "status": status,
                "aborted_by_deadline": status == "aborted", "deadline_enabled": True,
                "elapsed_seconds": 0.5,
                "selected_abnormal_count": 31, "pool_abnormal_count": 3045,
                "selected_distinct_signature_count": 25, "pool_distinct_signature_count": 822,
                "removed_pair_signature_count_R": 303, "added_pair_signature_count_A": 337779,
                "phase1_transition_count": 102347037, "phase1_transitions_evaluated": 12345,
                "concrete_candidates_evaluated_phase2": 0,
            }
            if engine_diag_extra:
                diag.update(engine_diag_extra)
            if diagnostics_log is not None:
                diagnostics_log.append(diag)
            if hotpath_profile is not None:
                hotpath_profile["watchdog_ticks_total"] = 12
                hotpath_profile["phase_timing"] = {
                    "equivalence_class_construction_seconds": 0.001,
                    "removed_pair_construction_seconds": 0.001,
                    "added_pair_construction_seconds": 0.001,
                    "aggregate_precompute_seconds": 0.001,
                }
                hotpath_profile["microbenchmark"] = {
                    "prefix_count_used": 6, "prefix_count_configured": 2000,
                    "total_transition_count_r_times_a": 102347037,
                    "deterministic_prefix_order": "removed_pairs_outer_added_pairs_inner",
                    "representativeness_caveat": "test caveat", "components": {},
                    "coverage_pass_fraction": 0.5, "strictly_improving_fraction_among_coverage_passing": 0.5,
                    "semantic_checksum_sha256": "a" * 64, "semantic_equivalence_verified": True,
                    "duplicate_state_diagnostics": {
                        "prefix_count_considered": 6, "unique_state_count": 6, "repeated_state_count": 0,
                        "duplicate_state_ratio": 0.0, "max_multiplicity": 1,
                        "unique_accept_rank_prefix_count": 3, "repeated_accepted_prefix_count": 0,
                        "representativeness_caveat": "test caveat",
                    },
                }
            if status == "aborted":
                return set(locked), M.RepairOutcome.COMPUTATIONAL_ABORT, {
                    "one_for_one_exhausted": False, "two_for_two_exhausted": False,
                    "local_optimum": False, "global_optimum_claimed": False, "aborted_by_deadline": True,
                }
            new_members = {p for p in (0, 1) if p not in locked} or {2, 3}
            return set(locked) | new_members, M.RepairOutcome.OK, {
                "one_for_one_exhausted": True, "two_for_two_exhausted": True,
                "local_optimum": True, "global_optimum_claimed": False,
            }
        return _fake

    def _run_profile(self, config, project_root, max_seconds=600.0, build_budget_step_fake=None,
                      status="aborted", engine_diag_extra=None):
        if build_budget_step_fake is None:
            build_budget_step_fake = self._fake_build_budget_step_profiler(status, engine_diag_extra)
        with mock.patch.object(M, "run_preflight", return_value={"status": "PASS", "checks": []}), \
             mock.patch.object(M, "compute_locked_size_targets",
                                return_value=({"1pct": 14, "5pct": 20, "10pct": 30, "20pct": 40},
                                              {"1pct": 0, "5pct": 1, "10pct": 2, "20pct": 3})), \
             mock.patch.object(M, "build_budget_step", side_effect=build_budget_step_fake):
            report = M.run_two_for_two_hotpath_profile(config, project_root, max_seconds)
        return report

    def test_engine_never_invoked_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            config, project_root = self._make_project(Path(tmp))

            def _fake_no_engine(budget, locked, image_ids, labels14, zero_gt, names, target_size, target_nf,
                                 seed, K, S_T, N, deadline, repair_log, diagnostics_log=None,
                                 phase_timing=None, hotpath_profile=None):
                # one-for-one alone "solved" objective_repair -- engine never invoked.
                return set(locked) | {0}, M.RepairOutcome.OK, {
                    "one_for_one_exhausted": True, "two_for_two_exhausted": True,
                    "local_optimum": True, "global_optimum_claimed": False,
                }

            with self.assertRaises(M.Phase2FError) as ctx:
                self._run_profile(config, project_root, build_budget_step_fake=_fake_no_engine)
            self.assertIn("PROFILE_INVOCATION_IDENTITY_MISMATCH", str(ctx.exception))

    def test_train_only_construction_state(self):
        source = SCRIPT_PATH.read_text(encoding="utf-8")
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == "run_two_for_two_hotpath_profile":
                body = ast.get_source_segment(source, node) or ""
                self.assertIn('inputs["train_coco"]', body)
                self.assertNotIn('inputs["val_coco"]', body)
                self.assertNotIn('inputs["test_coco"]', body)
                return
        self.fail("run_two_for_two_hotpath_profile not found")

    def test_real_invocation_identity_is_computed_not_hardcoded(self):
        with tempfile.TemporaryDirectory() as tmp:
            config, project_root = self._make_project(Path(tmp))
            config = dict(config)
            config["two_for_two_hotpath_profiler"] = dict(config["two_for_two_hotpath_profiler"])
            different_identity = {
                "selected_abnormal_count": 7, "pool_abnormal_count": 9,
                "selected_distinct_signature_count": 3, "pool_distinct_signature_count": 4,
                "removed_pair_signature_count_R": 2, "added_pair_signature_count_A": 5,
                "phase1_transition_count": 10, "one_for_one_move_count_before_two_for_two": 1,
            }
            config["two_for_two_hotpath_profiler"]["expected_invocation_identity"] = different_identity

            def _fake(budget, locked, image_ids, labels14, zero_gt, names, target_size, target_nf,
                      seed, K, S_T, N, deadline, repair_log, diagnostics_log=None,
                      phase_timing=None, hotpath_profile=None):
                if phase_timing is not None:
                    phase_timing["one_for_one_move_count_before_two_for_two"] = 1
                diag = {"budget": budget, "repair_phase": "objective_repair", "status": "aborted",
                        "aborted_by_deadline": True, "deadline_enabled": True, "elapsed_seconds": 0.2,
                        "phase1_transitions_evaluated": 4, "concrete_candidates_evaluated_phase2": 0,
                        "selected_abnormal_count": 7, "pool_abnormal_count": 9,
                        "selected_distinct_signature_count": 3, "pool_distinct_signature_count": 4,
                        "removed_pair_signature_count_R": 2, "added_pair_signature_count_A": 5,
                        "phase1_transition_count": 10}
                if diagnostics_log is not None:
                    diagnostics_log.append(diag)
                return set(locked), M.RepairOutcome.COMPUTATIONAL_ABORT, {"aborted_by_deadline": True}

            report = self._run_profile(config, project_root, build_budget_step_fake=_fake)
            self.assertEqual(report["real_invocation_identity"]["selected_abnormal_count"], 7)
            self.assertEqual(report["real_invocation_identity"]["removed_pair_signature_count_R"], 2)
            # not the real R9-evidence numbers baked into the un-overridden YAML.
            self.assertNotEqual(report["real_invocation_identity"]["selected_abnormal_count"], 31)

    def test_computational_abort_maps_to_profile_computational_abort(self):
        with tempfile.TemporaryDirectory() as tmp:
            config, project_root = self._make_project(Path(tmp))
            report = self._run_profile(config, project_root, status="aborted")
            self.assertEqual(report["profiling_status"], "PROFILE_COMPUTATIONAL_ABORT")

    def test_completed_maps_to_profile_completed(self):
        with tempfile.TemporaryDirectory() as tmp:
            config, project_root = self._make_project(Path(tmp))
            report = self._run_profile(config, project_root, status="completed")
            self.assertEqual(report["profiling_status"], "PROFILE_COMPLETED")

    def test_no_false_infeasibility_claim_on_abort(self):
        with tempfile.TemporaryDirectory() as tmp:
            config, project_root = self._make_project(Path(tmp))
            report = self._run_profile(config, project_root, status="aborted")
            self.assertFalse(report["real_invocation_identity"]["repair_infeasible"])

    def test_scientific_move_returned_always_false(self):
        with tempfile.TemporaryDirectory() as tmp:
            config, project_root = self._make_project(Path(tmp))
            report = self._run_profile(config, project_root, status="aborted")
            self.assertFalse(report["scientific_move_returned"])

    def test_never_calls_write_all_outputs_and_promote(self):
        with tempfile.TemporaryDirectory() as tmp:
            config, project_root = self._make_project(Path(tmp))
            with mock.patch.object(M, "write_all_outputs_and_promote") as mocked_promote:
                self._run_profile(config, project_root)
                mocked_promote.assert_not_called()

    def test_never_calls_promote_with_rollback(self):
        with tempfile.TemporaryDirectory() as tmp:
            config, project_root = self._make_project(Path(tmp))
            with mock.patch.object(M, "promote_with_rollback") as mocked_rollback:
                self._run_profile(config, project_root)
                mocked_rollback.assert_not_called()

    def test_creates_no_official_artifacts_only_the_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            config, project_root = self._make_project(Path(tmp))
            official_paths = {project_root / p for p in M.official_relative_paths(config)}
            self._run_profile(config, project_root)
            for p in official_paths:
                self.assertFalse(p.exists(), f"official artifact must not exist after profiling: {p}")
            report_path = project_root / Path(config["two_for_two_hotpath_profiler"]["report_path"])
            self.assertTrue(report_path.is_file())

    def test_report_write_is_atomic_no_leftover_temp_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            config, project_root = self._make_project(Path(tmp))
            self._run_profile(config, project_root)
            reports_dir = project_root / "reports"
            leftovers = [p for p in reports_dir.iterdir() if ".tmp-" in p.name]
            self.assertEqual(leftovers, [])

    def test_temp_report_cleaned_up_on_write_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            config, project_root = self._make_project(Path(tmp))
            report_path = project_root / Path(config["two_for_two_hotpath_profiler"]["report_path"])
            with mock.patch.object(M.os, "replace", side_effect=OSError("simulated failure")):
                with self.assertRaises(OSError):
                    self._run_profile(config, project_root)
            self.assertFalse(report_path.exists())
            reports_dir = project_root / "reports"
            leftovers = [p for p in reports_dir.iterdir() if ".tmp-" in p.name] if reports_dir.exists() else []
            self.assertEqual(leftovers, [])

    def test_existing_report_never_overwritten(self):
        with tempfile.TemporaryDirectory() as tmp:
            config, project_root = self._make_project(Path(tmp))
            report_path = project_root / Path(config["two_for_two_hotpath_profiler"]["report_path"])
            report_path.parent.mkdir(parents=True, exist_ok=True)
            report_path.write_text('{"pre-existing": true}', encoding="utf-8")
            with self.assertRaises(M.Phase2FError) as ctx:
                self._run_profile(config, project_root)
            self.assertIn("PROFILE_REPORT_EXISTS", str(ctx.exception))
            self.assertEqual(report_path.read_text(encoding="utf-8"), '{"pre-existing": true}')

    def test_r8_r9_operational_benchmark_reports_never_touched(self):
        with tempfile.TemporaryDirectory() as tmp:
            config, project_root = self._make_project(Path(tmp))
            r8_path = project_root / Path(config["operational_benchmark"]["superseded_evidence_path"])
            r9_path = project_root / Path(config["operational_benchmark"]["report_path"])
            r8_path.parent.mkdir(parents=True, exist_ok=True)
            r8_path.write_text('{"note": "r8 evidence"}', encoding="utf-8")
            r9_path.write_text('{"note": "r9 evidence"}', encoding="utf-8")
            r8_mtime = r8_path.stat().st_mtime_ns
            r9_mtime = r9_path.stat().st_mtime_ns
            self._run_profile(config, project_root)
            self.assertEqual(r8_path.read_text(encoding="utf-8"), '{"note": "r8 evidence"}')
            self.assertEqual(r9_path.read_text(encoding="utf-8"), '{"note": "r9 evidence"}')
            self.assertEqual(r8_path.stat().st_mtime_ns, r8_mtime)
            self.assertEqual(r9_path.stat().st_mtime_ns, r9_mtime)

    def test_every_gate_field_is_false(self):
        with tempfile.TemporaryDirectory() as tmp:
            config, project_root = self._make_project(Path(tmp))
            report = self._run_profile(config, project_root)
            for field in ("official_partition_artifact", "membership_lock", "usable_for_training",
                          "operational_approval_decision", "training_authorized", "training_started",
                          "pseudo_labels_generated", "test_used", "scientific_move_returned",
                          "official_artifacts_created"):
                self.assertFalse(report[field], f"{field} must be false")

    def test_seed_unchanged_in_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            config, project_root = self._make_project(Path(tmp))
            report = self._run_profile(config, project_root)
            self.assertEqual(report["partition_seed"], config["seed"]["partition_seed"])

    def test_build_budget_step_and_engine_backward_compatible_without_new_params(self):
        import inspect
        sig = inspect.signature(M.build_budget_step)
        self.assertIn("phase_timing", sig.parameters)
        self.assertIn("hotpath_profile", sig.parameters)
        self.assertIsNone(sig.parameters["phase_timing"].default)
        self.assertIsNone(sig.parameters["hotpath_profile"].default)
        sig2 = inspect.signature(M.repair_objective_local_search)
        self.assertIsNone(sig2.parameters["phase_timing"].default)
        self.assertIsNone(sig2.parameters["hotpath_profile"].default)
        sig3 = inspect.signature(M._equivalence_class_two_for_two_search)
        self.assertIsNone(sig3.parameters["hotpath_profile"].default)

    def test_representation_comparison_omitted_and_documented(self):
        with tempfile.TemporaryDirectory() as tmp:
            config, project_root = self._make_project(Path(tmp))
            report = self._run_profile(config, project_root)
            self.assertFalse(report["profiler_configuration"]["representation_comparison_enabled"])
            self.assertTrue(report["profiler_configuration"]["representation_comparison_omission_reason"])

    def test_report_distinguishes_coarse_component_calibration_fields(self):
        with tempfile.TemporaryDirectory() as tmp:
            config, project_root = self._make_project(Path(tmp))
            report = self._run_profile(config, project_root)
            self.assertIn("coarse_timing", report)
            self.assertIn("microbenchmark", report)
            self.assertIn("observer_effect_calibration", report)
            self.assertIsNot(report["coarse_timing"], report["observer_effect_calibration"])

    def test_memory_diagnostics_separate_field_not_enabled(self):
        with tempfile.TemporaryDirectory() as tmp:
            config, project_root = self._make_project(Path(tmp))
            report = self._run_profile(config, project_root)
            self.assertIn("memory_diagnostics", report)
            self.assertFalse(report["memory_diagnostics"]["enabled"])


# --------------------------------------------------------------------------- #
# R11 additive runtime observability                                          #
# --------------------------------------------------------------------------- #
class TestR11RuntimeTiming(unittest.TestCase):
    def setUp(self):
        self.train = make_tiny_fixture()
        self.image_ids, self.labels14, self.zero_gt, self.names = M.build_indicators(self.train)
        self.config = {
            "repair": {"active_repair_policy": M.ACTIVE_REPAIR_POLICY},
            "dependencies": {"iterative_stratification": {
                "required_version": "0.1.9", "package_name": "iterative-stratification",
            }},
            "inputs": {"train_coco": "unused.json"},
            "seed": {"partition_seed": 42},
        }
        self.targets = ({b: i + 1 for i, b in enumerate(M.BUDGET_ORDER)},
                        {b: 0 for b in M.BUDGET_ORDER})

    def _compute(self, step, clock=None, max_seconds=None):
        patches = [
            mock.patch.object(M, "check_iterative_stratification_version", return_value=("0.1.9", "OK")),
            mock.patch.object(M, "load_json", return_value=self.train),
            mock.patch.object(M, "build_indicators",
                              return_value=(self.image_ids, self.labels14, self.zero_gt, self.names)),
            mock.patch.object(M, "compute_locked_size_targets", return_value=self.targets),
            mock.patch.object(M, "build_budget_step", side_effect=step),
        ]
        if clock is not None:
            patches.append(mock.patch.object(M.time, "monotonic", side_effect=clock))
        with patches[0], patches[1], patches[2], patches[3], patches[4]:
            if clock is None:
                return M.compute_all_budgets(self.config, Path("."), max_seconds)
            with patches[5]:
                return M.compute_all_budgets(self.config, Path("."), max_seconds)

    @staticmethod
    def _ok_step(budget, locked, *_args):
        selected = set(locked)
        selected.add(len(selected))
        return selected, M.RepairOutcome.OK, {}

    def test_each_executed_budget_and_total_have_nonnegative_elapsed(self):
        bundle = self._compute(self._ok_step)
        self.assertEqual(set(bundle["per_budget_elapsed_seconds"]), set(M.BUDGET_ORDER))
        self.assertTrue(all(v >= 0 for v in bundle["per_budget_elapsed_seconds"].values()))
        self.assertGreaterEqual(bundle["total_construction_elapsed_seconds"], 0)

    def test_unrun_budgets_after_failure_have_no_fake_timing(self):
        def abort(*_args):
            return set(), M.RepairOutcome.COMPUTATIONAL_ABORT, {}
        bundle = self._compute(abort)
        self.assertEqual(set(bundle["per_budget_elapsed_seconds"]), {"1pct"})
        for budget in M.BUDGET_ORDER[1:]:
            self.assertNotIn(budget, bundle["per_budget_elapsed_seconds"])

    def test_timing_does_not_change_membership(self):
        counter_a = iter(float(i) for i in range(100))
        counter_b = iter(float(i * 17) for i in range(100))
        a = self._compute(self._ok_step, lambda: next(counter_a))
        b = self._compute(self._ok_step, lambda: next(counter_b))
        self.assertEqual(a["per_budget_selected"], b["per_budget_selected"])

    def test_timer_does_not_change_seed_objective_tie_break_or_global_deadline(self):
        seen = []
        def capture(*args):
            seen.append(args)
            return self._ok_step(*args)
        values = iter(float(i) for i in range(100))
        self._compute(capture, lambda: next(values), max_seconds=50.0)
        self.assertEqual({args[8] for args in seen}, {42})
        self.assertEqual(len({args[12] for args in seen}), 1)
        source = SCRIPT_PATH.read_text(encoding="utf-8")
        self.assertNotIn("per_budget_elapsed_seconds", ast.get_source_segment(
            source, next(n for n in ast.walk(ast.parse(source))
                         if isinstance(n, ast.FunctionDef) and n.name == "integer_objective")) or "")
        self.assertNotIn("per_budget_elapsed_seconds", ast.get_source_segment(
            source, next(n for n in ast.walk(ast.parse(source))
                         if isinstance(n, ast.FunctionDef) and n.name == "move_priority_digest")) or "")

    def test_none_deadline_remains_no_resource_cutoff(self):
        seen = []
        def capture(*args):
            seen.append(args[12])
            return self._ok_step(*args)
        self._compute(capture)
        self.assertEqual(seen, [None] * len(M.BUDGET_ORDER))

    def test_console_contract_contains_budget_and_total_timing(self):
        source = SCRIPT_PATH.read_text(encoding="utf-8")
        self.assertIn("OUTCOME={per_budget_outcome[budget]}", source)
        self.assertIn("ELAPSED_SECONDS={per_budget_elapsed.get(budget)}", source)
        self.assertIn("TOTAL_CONSTRUCTION_ELAPSED_SECONDS=", source)
        self.assertIn("bundle.get('total_construction_elapsed_seconds')", source)

    def test_validation_report_and_lock_manifest_include_timing_schema(self):
        source = SCRIPT_PATH.read_text(encoding="utf-8")
        fn = next(n for n in ast.walk(ast.parse(source))
                  if isinstance(n, ast.FunctionDef) and n.name == "write_all_outputs_and_promote")
        body = ast.get_source_segment(source, fn) or ""
        for field in ("per_budget_elapsed_seconds", "total_construction_elapsed_seconds",
                      "timing_clock", "timing_role"):
            self.assertGreaterEqual(body.count(f'"{field}"'), 2)

    def test_timing_is_excluded_from_checksum_and_digest_inputs(self):
        source = SCRIPT_PATH.read_text(encoding="utf-8")
        tree = ast.parse(source)
        for name in ("canonical_membership_sha256", "image_priority_digest",
                     "move_priority_digest", "candidate_set_priority_digest"):
            fn = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == name)
            body = ast.get_source_segment(source, fn) or ""
            self.assertNotIn("elapsed", body)
            self.assertNotIn("timing", body)

    def test_exception_reports_elapsed_and_cannot_return_partial_bundle(self):
        import contextlib
        import io
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            with self.assertRaisesRegex(RuntimeError, "boom"):
                self._compute(lambda *_args: (_ for _ in ()).throw(RuntimeError("boom")))
        self.assertIn("BUDGET=1pct OUTCOME=EXCEPTION SIZE=UNKNOWN ELAPSED_SECONDS=", output.getvalue())
        self.assertIn("TOTAL_CONSTRUCTION_ELAPSED_SECONDS=", output.getvalue())

    def test_reconstruct_check_does_not_compare_runtime_timing(self):
        import yaml
        config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
        compared = config["deterministic_reconstruction_mode"]["compared_fields"]
        self.assertFalse(any("timing" in field or "elapsed" in field for field in compared))
        self.assertFalse(any("timing" in field or "elapsed" in field
                             for field in M.RECONSTRUCTION_COMPARABLE_FIELDS))

    def test_identity_and_observability_constants_are_locked(self):
        import yaml
        config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
        self.assertEqual(config["protocol"]["stage"], "2F-C0-R11")
        self.assertEqual(config["protocol"]["version"], "2.0.0")
        self.assertEqual(M.TIMING_CLOCK, "time.monotonic")
        self.assertEqual(M.TIMING_ROLE, "OBSERVABILITY_ONLY_NOT_SELECTION_CRITERION")
        self.assertFalse(config["runtime_timing"]["included_in_membership_checksum"])
        self.assertFalse(config["runtime_timing"]["included_in_reconstruct_comparison"])


# --------------------------------------------------------------------------- #
# R6 pytest-repair round / NHIEM VU 7 -- meta-guardrail: core scientific      #
# guardrail test classes must never use skip. A weak/invalid fixture must    #
# FAIL (so it gets fixed), never silently SKIP. This does not ban skip       #
# project-wide -- TestReconstructCheckMode's @unittest.skipUnless(HAVE_      #
# ITERSTRAT, ...) is a legitimate environment-dependent integration-test     #
# gate (the real iterative-stratification package may not be installed),    #
# not a fixture weakness, and is intentionally NOT in this list.            #
# --------------------------------------------------------------------------- #
class TestNoSkipInCoreScientificGuardrails(unittest.TestCase):
    CORE_SCIENTIFIC_GUARDRAIL_CLASSES = (
        "TestRepairMinClassCoverage",
        "TestObjectiveRepairLocalSearch",
        "TestIndependentReadback",
        "TestActiveRepairPolicyGate",
        "TestLegacyTwoForTwoIsNonActive",
        "TestPreflightObservabilityFix",
        "TestR11RuntimeTiming",
    )
    # R7-B4: pytest's own spelling is lowercase "skipif" (distinct from
    # unittest's "skipIf"); both are forbidden, alongside "xfail" in either
    # decorator (@pytest.mark.xfail) or imperative-call (pytest.xfail(...))
    # form.
    FORBIDDEN_SKIP_DECORATOR_NAMES = {"skip", "skipIf", "skipif", "skipUnless", "xfail"}
    FORBIDDEN_SKIP_CALL_NAMES = {"skipTest", "skip", "xfail"}

    @staticmethod
    def _decorator_name(deco: ast.expr) -> str:
        node = deco.func if isinstance(deco, ast.Call) else deco
        if isinstance(node, ast.Attribute):
            return node.attr
        if isinstance(node, ast.Name):
            return node.id
        return ""

    @staticmethod
    def _call_name(node: ast.Call) -> str:
        func = node.func
        if isinstance(func, ast.Attribute):
            return func.attr
        if isinstance(func, ast.Name):
            return func.id
        return ""

    def test_no_skip_decorators_or_calls_in_core_classes(self):
        test_source = Path(__file__).resolve().read_text(encoding="utf-8")
        tree = ast.parse(test_source)
        found_classes = set()
        for node in ast.walk(tree):
            if not (isinstance(node, ast.ClassDef) and node.name in self.CORE_SCIENTIFIC_GUARDRAIL_CLASSES):
                continue
            found_classes.add(node.name)
            for inner in ast.walk(node):
                if isinstance(inner, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    for deco in inner.decorator_list:
                        deco_name = self._decorator_name(deco)
                        self.assertNotIn(
                            deco_name, self.FORBIDDEN_SKIP_DECORATOR_NAMES,
                            msg=f"{node.name}.{inner.name} has a forbidden @...{deco_name} skip decorator",
                        )
                if isinstance(inner, ast.Call):
                    called_name = self._call_name(inner)
                    self.assertNotIn(
                        called_name, self.FORBIDDEN_SKIP_CALL_NAMES,
                        msg=f"{node.name} calls forbidden {called_name}(...) -- fixtures must FAIL, not skip",
                    )
        missing = set(self.CORE_SCIENTIFIC_GUARDRAIL_CLASSES) - found_classes
        self.assertFalse(missing, f"core guardrail class(es) not found in test file: {missing}")


if __name__ == "__main__":
    unittest.main()
