"""Phase 2F — Labeled/Unlabeled Construction.

The active protocol stage/version are NEVER hard-coded here -- they are
read at runtime from configs/protocol/phase2F_labeled_unlabeled.yaml (see
get_protocol_identity()) and are what the console banner and every output
artifact's provenance fields actually report. The revision notes below are
a historical record of what changed and why in each past static-review
cycle; they are not a claim about which stage this file "is" right now.

Builds nested labeled/unlabeled SSL subsets (1% / 5% / 10% / 20%) from the
Phase 2E fixed, checksum-locked `instances_train.json`. Never reads,
modifies, or resamples `instances_val.json` / `instances_test.json`.

ACTIVE PROTOCOL (Revision R11)
------------------------------
  iterative multilabel stratification
  -> exact-size repair
  -> exact-No-Finding repair
  -> minimum-class-coverage ONE-FOR-ONE repair
  -> deterministic ONE-FOR-ONE objective repair
  -> stop at a ONE-FOR-ONE LOCAL OPTIMUM
  -> validation
  -> transactional promotion

The single allowed claim about the objective-repair stage is: "The
objective-repair stage exhaustively enumerates the admissible one-for-one
swap neighborhood and terminates at a one-for-one local optimum. No global
optimum is claimed."

R11 removes the two-for-two composition-preserving neighborhood from the
ACTIVE construction protocol. Rationale, recorded here so it is never
mistaken for a scientific-quality claim: a real, non-promoting R9 benchmark
run (partition_seed=42, budget 1pct) reached only 24,849,751 of
102,347,037 Phase-1 signature transitions (R=303, A=337,779) in 591.922 s
and ended in COMPUTATIONAL_ABORT with two_for_two_move_count=0. The static
audit of that run established that at the moment the engine was first
invoked EVERY hard constraint was already satisfied (exact labeled size 34,
exact No-Finding size 3, subset entirely inside Phase 2E TRAIN,
selected/unlabeled complement semantics, no VAL/TEST use, all 14 abnormality
classes present, seed=42, no seed search) and that the invocation happened
in `objective_repair` only AFTER the one-for-one neighborhood was already
exhausted -- i.e. it was OPTIONAL objective refinement, never required
hard-constraint repair. The removal was decided BEFORE any training, BEFORE
any pseudo-labelling, WITHOUT model metrics, WITHOUT VAL/TEST outcome
selection, WITHOUT candidate comparison and WITHOUT any seed search; the
reason is computational tractability and proportionality only.

R11 legacy discipline: `_equivalence_class_two_for_two_search` and its
helpers/oracle are RETAINED for historical reproducibility and for their
existing unit tests, but they are NON-ACTIVE. They are unreachable from
`build_budget_step`, `compute_all_budgets` (official materialization),
`run_reconstruct_check` and `run_operational_benchmark`; they never gate
materialization; and their completeness/tractability statements are no
longer active protocol claims. The revision notes below R11 are a
historical record of past static-review cycles, NOT a description of the
active protocol.

Revision R4 (applied after a third GPT static review of R3, which left two
blockers open). Summary:
  - blocker R3-B2 (independent readback did not compare exact labeled/
    unlabeled image-record PAYLOAD against the locked train file, only
    annotation-level content) is fixed: independently_validate_staging now
    performs full semantic-payload comparison of images, categories, and
    top-level fields, plus explicit duplicate-id and out-of-universe-id
    rejection gates, all against a fresh re-read of the exact expected
    subset of instances_train.json -- never relying on the staged lock
    manifest's self-consistency with a possibly-already-corrupted staged
    file;
  - blocker R3-B1 (same-composition two-for-two was O(|selected|^2 *
    |pool|^2) exhaustive, computationally infeasible at real scale) is
    addressed with an exact, equivalence-class signature search
    (_equivalence_class_two_for_two_search), mathematically PROVEN
    equivalent to exhaustive concrete enumeration on every tier of the
    locked acceptance ordering (hard-constraint progress, integer
    objective, seeded SHA-256 priority, canonical numeric identity) -- see
    the proof comment above _brute_force_same_composition_two_for_two_pairs
    (the old brute-force implementation, retained as a TEST-ONLY oracle).
    Its WORST-CASE complexity is not proven better than brute force (see
    the proof's complexity analysis), so it is gated behind
    `two_for_two_engine.operationally_approved` (default false, a locked
    human/researcher decision field, never set by Claude) -- construction
    refuses closed with taxonomy TWO_FOR_TWO_ENGINE_NOT_OPERATIONALLY_
    APPROVED before starting any search while unapproved, both for the
    official run and --reconstruct-check (both funnel through
    compute_all_budgets). --preflight-only still completes and reports the
    engine's status fields, which are informational, not part of the
    preflight PASS/FAIL gate.

Revision R3 (applied after a second GPT static review of R2). Summary:
  - iterative-stratification is now version-locked to exactly 0.1.9,
    checked at runtime via importlib.metadata (never __version__ /
    pkg_resources), fail-closed on mismatch or absence. The resolved
    version string is recorded in the validation report and the seed
    manifest;
  - --reconstruct-check now compares ALL seven fields declared in
    configs/.../deterministic_reconstruction_mode.compared_fields
    (labeled_image_id_sha256, labeled AND unlabeled coco_json_sha256,
    labeled_size, no_finding_size, repair_move_counts,
    integer_objective_final), config-driven so the field list and the
    implemented checks cannot silently drift apart;
  - independent readback (independently_validate_staging) is now FULLY
    independent: it re-reads instances_train.json and the staged
    phase2F_lock_manifest.json from disk (never an in-memory object still
    held from construction), rebuilds zero-GT/coverage from the staged
    labeled COCO's own annotation records, and verifies the exact staged
    annotation ID-set/count/payload checksum against the exact expected
    subset of the locked train file, so any added/removed/modified
    annotation flips a gate to FAIL;
  - minimum_class_coverage candidate ranking is now explicitly
    STRICTLY_REDUCE_MISSING_CLASS_COUNT_BY_AT_LEAST_ONE, and orders
    candidates by (missing_count_after, integer_objective, seeded SHA-256
    priority, canonical numeric identity) -- coverage progress always wins
    first, ahead of objective quality;
  - the two-for-two mixed-composition neighborhood (1 abnormal + 1
    No-Finding <-> 1 abnormal + 1 No-Finding) is proven redundant with the
    one-for-one neighborhood (an NF image contributes an all-zero labels14
    row, so its swap never changes coverage or the integer objective) and
    is no longer separately enumerated; see the proof comment above
    _two_for_two_candidates. The same-composition (2 abnormal <-> 2
    abnormal) branch's O(|selected|^2 * |pool|^2) exhaustive complexity is
    UNCHANGED and explicitly flagged BLOCKED -- no complete polynomial
    pruning was found or proven for this revision; see the BLOCKED comment
    block above _two_for_two_candidates and the chat handoff for the full
    analysis and options.

Revision R2 (applied after a GPT static review of R1). Summary of what
changed and why is in configs/protocol/phase2F_labeled_unlabeled.yaml's
header comment; the short version:
  - the distribution objective (D_max/D_mean/D_LC) now uses the 14 abnormal
    class-presence indicators ONLY. zero_gt is used solely for the
    stratification matrix and the exact-No-Finding hard constraint, never
    for the distribution objective;
  - D_mean is mean ABSOLUTE per-class deviation, not squared;
  - the move-selection/ranking path uses an EXACT INTEGER tuple
    (E_max, E_mean, E_LC) — no Fraction, no float, no round()/isclose() in
    that path. Fraction is used only for post-hoc reporting;
  - a fourth repair phase (objective_repair) runs after exact size/exact
    NF/14-of-14 coverage are satisfied, doing local-search improvement of
    the integer objective (one-for-one first, then two-for-two, repeating
    until neither neighborhood has an improving move — a LOCAL optimum,
    never claimed as global);
  - the two-for-two neighborhood now covers BOTH same-composition
    (2 abnormal <-> 2 abnormal) and mixed-composition (1 abnormal + 1
    No-Finding <-> 1 abnormal + 1 No-Finding) swaps;
  - tie-break uses three canonical namespaced SHA-256 payloads
    (phase2F_image_priority_v1, phase2F_repair_move_priority_v1,
    phase2F_candidate_set_priority_v1);
  - preflight, independent readback, and the required-evidence set are all
    substantially expanded (see below);
  - a non-promoting --reconstruct-check mode lets the researcher rebuild
    everything from scratch and diff it against the already-promoted lock
    manifest without touching any official artifact.

Construction order is small-to-large (L1pct first, then extended to L5pct,
L10pct, L20pct) so that L1pct subset_of L5pct subset_of L10pct subset_of
L20pct holds by construction (immutable nested prefix). No-Finding
membership is nested by the same construction.

Materialization follows stage -> validate -> independent readback ->
transactional promote, refusing to run if any official artifact already
exists, and rolling back only the files this run itself staged if promotion
fails partway through.

This script performs no training, inference, pseudo-label generation,
threshold tuning, or AP/mAP computation, and never touches val/test.

Default project inputs:
  configs/protocol/phase2F_labeled_unlabeled.yaml
  data/processed/coco/instances_train.json
  data/manifests/split_lock_manifest.json
  data/manifests/leakage_check_report.json
  reports/phase2E_build_fixed_split_validation_report.json
  reports/phase2E_build_fixed_split_log.json

Official outputs: see configs/protocol/phase2F_labeled_unlabeled.yaml -> outputs.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import shutil
import sys
import tempfile
import time
from collections import Counter
from fractions import Fraction
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import yaml


BUDGET_ORDER = ("1pct", "5pct", "10pct", "20pct")
BUDGET_FRACTION = {"1pct": Fraction(1, 100), "5pct": Fraction(5, 100),
                   "10pct": Fraction(10, 100), "20pct": Fraction(20, 100)}
# R7 -- nested-prefix parent of each budget (None for the smallest budget,
# which has no parent). Shared by run_operational_benchmark's
# NOT_RUN_DEPENDENCY bookkeeping; never used to change membership itself.
NESTED_PARENT_BUDGET = {"1pct": None, "5pct": "1pct", "10pct": "5pct", "20pct": "10pct"}
SPLIT_ORDER = ("train", "val", "test")
REPAIR_PHASE_ORDER = ("exact_size", "exact_no_finding", "minimum_class_coverage", "objective_repair")
# R11 -- every move_type the ACTIVE repair cascade can emit. There is no
# "two_for_two_swap" entry: the two-for-two neighborhood is not part of the
# active protocol (see the ACTIVE PROTOCOL section of the module docstring).
REPAIR_MOVE_TYPE_ORDER = ("add", "remove", "one_for_one_swap")

# R11 -- the ACTIVE repair policy. The objective-repair stage exhaustively
# enumerates the admissible ONE-FOR-ONE swap neighborhood and terminates at a
# ONE-FOR-ONE LOCAL OPTIMUM; no global optimum is claimed, and no heuristic,
# top-k, sampling or wider-neighborhood search replaces the removed
# two-for-two branch. Must equal configs/.../repair.active_repair_policy --
# cross-checked informationally in run_preflight and enforced FAIL-CLOSED in
# compute_all_budgets (taxonomy ACTIVE_REPAIR_POLICY_MISMATCH), so the
# protocol document and the implementation can never silently diverge on
# which neighborhood the active protocol actually searches.
ACTIVE_REPAIR_POLICY = "ONE_FOR_ONE_EXHAUSTIVE_DETERMINISTIC"
# R11 -- the neighborhood the reported local optimum is local WITH RESPECT TO.
# Recorded verbatim in every diagnostics/evidence record so a reader never has
# to infer which neighborhood was actually exhausted.
LOCAL_OPTIMUM_NEIGHBORHOOD = "one_for_one"

# R11 runtime observability. These values describe wall-clock durations only;
# they are deliberately excluded from every membership/checksum/tie-break
# input and from deterministic reconstruction comparisons.
TIMING_CLOCK = "time.monotonic"
TIMING_ROLE = "OBSERVABILITY_ONLY_NOT_SELECTION_CRITERION"

# R11 active execution purposes are OFFICIAL_MATERIALIZATION and
# RECONSTRUCT_CHECK.  The benchmark constant/function below are retained as
# historical R7--R9 implementation evidence only; parse_args/main expose no
# route to them and active construction never reads legacy engine approval.
EXECUTION_PURPOSE_OFFICIAL_MATERIALIZATION = "OFFICIAL_MATERIALIZATION"
EXECUTION_PURPOSE_RECONSTRUCT_CHECK = "RECONSTRUCT_CHECK"
EXECUTION_PURPOSE_NON_PROMOTING_OPERATIONAL_BENCHMARK = "NON_PROMOTING_OPERATIONAL_BENCHMARK"
# R11 -- the R10 NON_PROMOTING_PERFORMANCE_DIAGNOSTIC_ONLY execution purpose
# (the unfinished two-for-two hot-path profiler) has been REMOVED. It existed
# only to measure the two-for-two engine, which is no longer part of the
# active construction protocol; it never produced a report, and its CLI mode,
# config block, orchestration function and `phase_timing`/`hotpath_profile`
# plumbing are all gone. Nothing in the active protocol replaces it.

NAMESPACE_IMAGE_PRIORITY = "phase2F_image_priority_v1"
NAMESPACE_MOVE_PRIORITY = "phase2F_repair_move_priority_v1"
NAMESPACE_SET_PRIORITY = "phase2F_candidate_set_priority_v1"

# NHIEM VU R3-1: minimum_class_coverage candidate-ranking rule name, locked
# in config under repair.coverage_progress_rule and cross-checked in
# run_preflight so the code and the protocol document cannot silently
# diverge on the semantics of "progress".
COVERAGE_PROGRESS_RULE = "STRICTLY_REDUCE_MISSING_CLASS_COUNT_BY_AT_LEAST_ONE"

# LEGACY / NON-ACTIVE (R11): deadline/watchdog check-FREQUENCY constant for
# the retained-but-unreachable _equivalence_class_two_for_two_search. Purely
# a resource-cutoff polling interval (how many candidates are examined
# between _check_deadline calls) -- never a candidate cap, never a
# heuristic/pruning device, and never a factor in which candidate is
# accepted. Must equal configs/.../legacy_two_for_two_engine.
# deadline_watchdog_check_interval_candidates -- cross-checked in
# tests/test_phase2F_labeled_unlabeled_guardrails.py. Not read by any active
# construction path.
TWO_FOR_TWO_DEADLINE_WATCHDOG_INTERVAL = 1024

# NHIEM VU R3-2: the exhaustive set of fields --reconstruct-check knows how
# to compare. configs/.../deterministic_reconstruction_mode.compared_fields
# is validated against this set at runtime (run_reconstruct_check) -- any
# configured field with no entry here is a hard FAIL (CONFIG_INVALID), never
# a silent skip.
RECONSTRUCTION_COMPARABLE_FIELDS = (
    "labeled_image_id_sha256", "labeled_coco_json_sha256", "unlabeled_coco_json_sha256",
    "labeled_size", "no_finding_size", "repair_move_counts", "integer_objective_final",
)


class Phase2FError(RuntimeError):
    """Raised for any FAIL condition."""


def fail(message: str, taxonomy: str = "FAIL") -> None:
    raise Phase2FError(f"[{taxonomy}] {message}")


# --------------------------------------------------------------------------- #
# Generic IO helpers                                                          #
# --------------------------------------------------------------------------- #
def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        fail(f"Expected JSON object in {path}")
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False))
            stream.write("\n")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_config(path: Path) -> dict[str, Any]:
    if not path.is_file():
        fail(f"Protocol config not found: {path}", "CONFIG_MISSING")
    with path.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        fail(f"Protocol config did not parse to a mapping: {path}", "CONFIG_INVALID")
    return config


# --------------------------------------------------------------------------- #
# Metadata/provenance revision (fixes stale hard-coded stage strings): the
# console banner and every output artifact's provenance fields must read
# the ACTIVE protocol stage/version from THIS function, which reads
# config["protocol"]["stage"]/["version"] at runtime -- never a literal
# "2F-C0-Rn" baked into script source. Fail-closed (CONFIG_INVALID) if the
# identity is missing or malformed, so a broken/incomplete config can never
# silently produce mislabeled console output or evidence provenance.
# --------------------------------------------------------------------------- #
def get_protocol_identity(config: dict[str, Any]) -> tuple[str, str]:
    """Returns (stage, version) read from config["protocol"]["stage"] /
    config["protocol"]["version"]. Never falls back to a hard-coded
    default. Fails closed (Phase2FError, taxonomy CONFIG_INVALID) if
    config["protocol"] is missing/not a mapping, or if stage or version is
    missing, not a string, or an empty/whitespace-only string."""
    protocol = config.get("protocol")
    if not isinstance(protocol, dict):
        fail("config['protocol'] is missing or not a mapping", "CONFIG_INVALID")
    stage = protocol.get("stage")
    if not isinstance(stage, str) or not stage.strip():
        fail("config['protocol']['stage'] is missing, not a string, or empty", "CONFIG_INVALID")
    version = protocol.get("version")
    if not isinstance(version, str) or not version.strip():
        fail("config['protocol']['version'] is missing, not a string, or empty", "CONFIG_INVALID")
    return stage, version


# --------------------------------------------------------------------------- #
# NHIEM VU 5 — official membership checksum (validate-integer, numeric
# ascending sort, canonical decimal representation, LF join, UTF-8,
# SHA-256). Distinct from Phase 2E's own convention (sort key=str(value)).
# --------------------------------------------------------------------------- #
def _validate_canonical_int(value: Any) -> int:
    if isinstance(value, bool):
        fail(f"image_id must be integer, got bool: {value!r}", "CHECKSUM_INVALID_TYPE")
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.strip().lstrip("-").isdigit():
        return int(value)
    fail(f"image_id must be integer, got non-integer value: {value!r}", "CHECKSUM_INVALID_TYPE")
    raise AssertionError("unreachable")  # fail() always raises; keeps type-checkers happy


def canonical_membership_sha256(image_ids: Iterable[Any]) -> str:
    canonical_ints = sorted(_validate_canonical_int(v) for v in image_ids)
    canonical_text = "\n".join(str(v) for v in canonical_ints)
    return hashlib.sha256(canonical_text.encode("utf-8")).hexdigest()


def _record_list_has_duplicate_or_invalid_id(records: list[dict[str, Any]], id_field: str = "id") -> bool:
    """True if ANY record's id_field is not a canonical integer, or if any
    two records share an id. Never raises -- used to turn a corrupted
    staged file into a FAIL gate rather than a crash."""
    seen: set[int] = set()
    for r in records:
        try:
            v = _validate_canonical_int(r[id_field])
        except Phase2FError:
            return True
        if v in seen:
            return True
        seen.add(v)
    return False


def _canonical_record_list_sha256(records: list[dict[str, Any]], id_field: str = "id") -> str:
    """Order-independent, content-sensitive checksum of a COCO record list
    (images, annotations, or categories): FAILS if any id_field is not a
    canonical integer or if any id is duplicated (semantic identity must
    never be computed over a schema-invalid list), otherwise canonicalizes
    by sorting on id_field then serializing with sort_keys=True and fixed
    separators so every field of every record participates. Any added,
    removed, or modified record (any field, not just the id) changes this
    digest. Never depends on file/list order."""
    if _record_list_has_duplicate_or_invalid_id(records, id_field):
        fail(f"Duplicate or non-canonical-integer {id_field} in record list", "SCHEMA_DRIFT")
    canonical = sorted(records, key=lambda r: _validate_canonical_int(r[id_field]))
    payload = json.dumps(canonical, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _safe_canonical_record_list_sha256(records: list[dict[str, Any]], id_field: str = "id") -> str | None:
    """Non-raising wrapper for STAGED (externally-supplied, possibly
    corrupted) record lists: returns None -- a sentinel that can never
    equal a real digest -- instead of raising, so a duplicate/invalid id in
    a staged file flips the corresponding readback gate to FAIL instead of
    crashing the whole independent-readback pass."""
    try:
        return _canonical_record_list_sha256(records, id_field)
    except Phase2FError:
        return None


def _canonical_annotation_list_sha256(annotations: list[dict[str, Any]]) -> str:
    """Backward-compatible name for _canonical_record_list_sha256 over
    annotations specifically (id_field="id")."""
    return _canonical_record_list_sha256(annotations, id_field="id")


def phase2e_style_sha256(values: Iterable[Any]) -> str:
    """Reproduce Phase 2E's own sha256_image_ids exactly (sort key=str(v)),
    used ONLY to validate the untouched Phase 2E lock, never for Phase 2F's
    own membership checksums."""
    canonical = "\n".join(str(v) for v in sorted(values, key=lambda v: str(v)))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


# --------------------------------------------------------------------------- #
# Canonical namespaced SHA-256 tie-break (three versioned payload families)
# --------------------------------------------------------------------------- #
def _canonical_id_csv(ids: Iterable[int]) -> str:
    canonical_ints = sorted(_validate_canonical_int(v) for v in ids)
    return ",".join(str(v) for v in canonical_ints)


def image_priority_digest(seed: int, image_id: int) -> str:
    payload = f"{NAMESPACE_IMAGE_PRIORITY}|seed={seed}|image_id={_validate_canonical_int(image_id)}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def move_priority_digest(
    seed: int, budget: str, repair_phase: str, move_type: str,
    removed_ids: Iterable[int], added_ids: Iterable[int],
) -> str:
    payload = (
        f"{NAMESPACE_MOVE_PRIORITY}|seed={seed}|budget={budget}|phase={repair_phase}|"
        f"move_type={move_type}|removed={_canonical_id_csv(removed_ids)}|added={_canonical_id_csv(added_ids)}"
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def candidate_set_priority_digest(seed: int, budget: str, member_ids: Iterable[int]) -> str:
    payload = f"{NAMESPACE_SET_PRIORITY}|seed={seed}|budget={budget}|members={_canonical_id_csv(member_ids)}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def canonical_numeric_move_identity(removed_ids: Iterable[int], added_ids: Iterable[int]) -> tuple[tuple[int, ...], tuple[int, ...]]:
    return (
        tuple(sorted(_validate_canonical_int(v) for v in removed_ids)),
        tuple(sorted(_validate_canonical_int(v) for v in added_ids)),
    )


# --------------------------------------------------------------------------- #
# Round-half-up (exact; used only for target-size/count PRE-computation, not
# for the objective-selection path).
# --------------------------------------------------------------------------- #
def round_half_up_fraction(value: Fraction) -> int:
    shifted = value + Fraction(1, 2)
    return shifted.numerator // shifted.denominator  # floor for positive Fraction


# --------------------------------------------------------------------------- #
# NHIEM VU R3-1 — iterative-stratification version lock. Reads the installed
# version via importlib.metadata ONLY (never __version__ attributes, never
# pkg_resources). config-driven: the required version string lives in
# configs/.../dependencies.iterative_stratification.required_version, never
# hardcoded here.
# --------------------------------------------------------------------------- #
def check_iterative_stratification_version(config: dict[str, Any]) -> tuple[str | None, str]:
    """Returns (installed_version_or_None, status), status in
    {"OK", "NOT_INSTALLED", "VERSION_MISMATCH"}. Never raises for a missing
    or mismatched package -- callers (run_preflight, compute_all_budgets)
    decide whether/how to fail closed."""
    dep = config["dependencies"]["iterative_stratification"]
    required = dep["required_version"]
    package_name = dep["package_name"]
    try:
        installed = importlib.metadata.version(package_name)
    except importlib.metadata.PackageNotFoundError:
        return None, "NOT_INSTALLED"
    if installed != required:
        return installed, "VERSION_MISMATCH"
    return installed, "OK"


# --------------------------------------------------------------------------- #
# NHIEM VU 1 — Preflight
# --------------------------------------------------------------------------- #
def run_preflight(config: dict[str, Any], project_root: Path) -> dict[str, Any]:
    ref = config["locked_phase2e_reference"]
    inputs = config["inputs"]

    # R11: the ACTIVE repair policy is the only neighborhood claim preflight
    # reports. Read + validate it fail-closed (CONFIG_INVALID) here, BEFORE
    # any early return, so a malformed/absent policy can never silently
    # produce a result missing this field instead of a clear error. Every
    # former two_for_two_engine informational field (engine status,
    # completeness_claim, complexity_policy, mathematical_completeness,
    # empirical_operational_tractability, operational_approval,
    # operationally_approved) and the full-train two-for-two complexity
    # diagnostics have been REMOVED: they described a neighborhood that is
    # no longer part of the active protocol, and reporting them would keep a
    # retired claim alive in current evidence.
    repair_cfg = config.get("repair")
    if not isinstance(repair_cfg, dict):
        fail("config['repair'] is missing or not a mapping", "CONFIG_INVALID")
    if "active_repair_policy" not in repair_cfg:
        fail("config['repair']['active_repair_policy'] is missing", "CONFIG_INVALID")

    result: dict[str, Any] = {
        "phase": "2F-PREFLIGHT", "checks": [], "status": "PENDING",
        "policy_evidence_training_authorized": "POLICY_EVIDENCE_NOT_MACHINE_READABLE",
        # Informational, config-driven fields, populated here so they exist
        # regardless of how the input-audit checks below turn out.
        "active_repair_policy": repair_cfg["active_repair_policy"],
        "objective_repair_neighborhood": LOCAL_OPTIMUM_NEIGHBORHOOD,
        "objective_repair_termination_claim": (
            "The objective-repair stage exhaustively enumerates the admissible one-for-one "
            "swap neighborhood and terminates at a one-for-one local optimum. No global "
            "optimum is claimed."
        ),
        "global_optimum_claimed": False,
    }

    def check(name: str, ok: bool, detail: Any = None) -> None:
        result["checks"].append({"name": name, "status": "PASS" if ok else "FAIL", "detail": detail})

    required_paths = {
        "train_coco_exists": project_root / inputs["train_coco"],
        "val_coco_exists": project_root / inputs["val_coco"],
        "test_coco_exists": project_root / inputs["test_coco"],
        "split_lock_manifest_exists": project_root / inputs["split_lock_manifest"],
        "leakage_check_report_exists": project_root / inputs["leakage_check_report"],
        "phase2e_validation_report_exists": project_root / inputs["phase2e_validation_report"],
        "phase2e_log_report_exists": project_root / inputs["phase2e_log_report"],
    }
    for name, path in required_paths.items():
        check(name, path.is_file(), str(path))

    iterstrat_version, iterstrat_status = check_iterative_stratification_version(config)
    dep = config["dependencies"]["iterative_stratification"]
    check("iterative_stratification_version_locked", iterstrat_status == "OK",
          {"installed": iterstrat_version, "required": dep["required_version"], "status": iterstrat_status})
    result["iterative_stratification_version"] = iterstrat_version

    check("coverage_progress_rule_matches_locked_constant",
          config["repair"]["coverage_progress_rule"] == COVERAGE_PROGRESS_RULE,
          {"config": config["repair"]["coverage_progress_rule"], "code": COVERAGE_PROGRESS_RULE})

    # R11: the protocol document and the implementation must agree on which
    # neighborhood the ACTIVE objective-repair stage searches. Unlike the
    # retired engine-status fields, this IS part of the PASS/FAIL check list:
    # a mismatch means the running code is not the documented protocol.
    check("active_repair_policy_matches_locked_constant",
          repair_cfg["active_repair_policy"] == ACTIVE_REPAIR_POLICY,
          {"config": repair_cfg["active_repair_policy"], "code": ACTIVE_REPAIR_POLICY})

    if not all(entry["status"] == "PASS" for entry in result["checks"]):
        result["status"] = "FAIL"
        return result

    coco = {name: load_json(project_root / inputs[f"{name}_coco"]) for name in SPLIT_ORDER}
    split_lock = load_json(project_root / inputs["split_lock_manifest"])
    leakage_report = load_json(project_root / inputs["leakage_check_report"])
    phase2e_validation_report = load_json(project_root / inputs["phase2e_validation_report"])

    ann_id_sets: dict[str, set[Any]] = {}
    for name in SPLIT_ORDER:
        images = coco[name].get("images", [])
        annotations = coco[name].get("annotations", [])
        categories = coco[name].get("categories", [])
        ann_id_sets[name] = {a["id"] for a in annotations}

        check(f"{name}_image_count", len(images) == ref["split_sizes"][name],
              {"expected": ref["split_sizes"][name], "observed": len(images)})
        check(f"{name}_annotation_count", len(annotations) == ref["annotation_sizes"][name],
              {"expected": ref["annotation_sizes"][name], "observed": len(annotations)})
        img_with_ann = {a["image_id"] for a in annotations}
        zero_gt_count = sum(1 for im in images if im["id"] not in img_with_ann)
        check(f"{name}_no_finding_count", zero_gt_count == ref["no_finding_sizes"][name],
              {"expected": ref["no_finding_sizes"][name], "observed": zero_gt_count})
        check(f"{name}_category_count", len(categories) == ref["total_categories"],
              {"expected": ref["total_categories"], "observed": len(categories)})
        observed_hash = sha256_file(project_root / inputs[f"{name}_coco"])
        check(f"{name}_coco_json_sha256", observed_hash == ref["coco_json_sha256"][name],
              {"expected": ref["coco_json_sha256"][name], "observed": observed_hash})
        observed_id_hash = phase2e_style_sha256([im["id"] for im in images])
        check(f"{name}_phase2e_image_id_sha256", observed_id_hash == ref["image_id_sha256"][name],
              {"expected": ref["image_id_sha256"][name], "observed": observed_id_hash})

    # identical category id+name mapping across all three splits
    category_maps = {
        name: sorted((c["id"], c["name"]) for c in coco[name]["categories"]) for name in SPLIT_ORDER
    }
    check("identical_category_id_and_name_mapping_across_splits",
          category_maps["train"] == category_maps["val"] == category_maps["test"],
          category_maps["train"])
    check("train_categories_contiguous_1_14",
          [cid for cid, _ in category_maps["train"]] == list(range(1, 15)),
          category_maps["train"])

    image_ids_by_split = {name: {im["id"] for im in coco[name]["images"]} for name in SPLIT_ORDER}
    check("train_val_image_overlap_zero", not (image_ids_by_split["train"] & image_ids_by_split["val"]))
    check("train_test_image_overlap_zero", not (image_ids_by_split["train"] & image_ids_by_split["test"]))
    check("val_test_image_overlap_zero", not (image_ids_by_split["val"] & image_ids_by_split["test"]))
    check("train_val_annotation_id_overlap_zero", not (ann_id_sets["train"] & ann_id_sets["val"]))
    check("train_test_annotation_id_overlap_zero", not (ann_id_sets["train"] & ann_id_sets["test"]))
    check("val_test_annotation_id_overlap_zero", not (ann_id_sets["val"] & ann_id_sets["test"]))

    image_union = set().union(*image_ids_by_split.values())
    check("image_union_identity_and_count", len(image_union) == ref["total_images"],
          {"expected": ref["total_images"], "observed": len(image_union)})
    ann_union = set().union(*ann_id_sets.values())
    total_ann_expected = ref["total_annotations"]
    total_ann_observed = sum(len(coco[name]["annotations"]) for name in SPLIT_ORDER)
    check("annotation_union_identity_and_count",
          len(ann_union) == total_ann_observed == total_ann_expected,
          {"expected": total_ann_expected, "observed_union": len(ann_union), "observed_sum": total_ann_observed})

    check("split_lock_manifest_status_and_reference_match",
          split_lock.get("status") == "LOCKED"
          and split_lock.get("image_id_sha256") == ref["image_id_sha256"]
          and split_lock.get("coco_json_sha256") == ref["coco_json_sha256"],
          {"split_lock_status": split_lock.get("status")})
    check("leakage_check_report_status_pass_and_reference_match",
          leakage_report.get("status") == "PASS"
          and leakage_report.get("image_union") == ref["total_images"]
          and leakage_report.get("annotation_union") == ref["total_annotations"],
          {"leakage_status": leakage_report.get("status")})
    check("phase2e_validation_report_status_pass",
          phase2e_validation_report.get("status") == "PASS",
          {"phase2e_validation_status": phase2e_validation_report.get("status")})

    train_images = coco["train"]["images"]
    for image in train_images[:1]:
        for forbidden in ("is_negative", "scope_label"):
            if forbidden not in image:
                fail(f"Expected field '{forbidden}' missing from instances_train.json images; "
                     "unlabeled-field-stripping logic must be re-verified against the actual schema.",
                     "SCHEMA_DRIFT")

    # R11: the full-train two-for-two equivalence-class complexity
    # diagnostics that used to be computed here have been REMOVED. Their sole
    # purpose was to inform the retired
    # two_for_two_engine.operationally_approved decision; the active protocol
    # does not search that neighborhood, so continuing to publish those
    # figures would keep a retired claim alive in current evidence. No
    # replacement diagnostic is introduced here -- the active protocol's own
    # per-budget evidence is recorded at the one-for-one local optimum
    # instead (see build_final_one_for_one_diagnostics).

    result["status"] = "PASS" if all(c["status"] == "PASS" for c in result["checks"]) else "FAIL"
    result["train_image_count"] = len(train_images)
    result["train_annotation_count"] = len(coco["train"]["annotations"])
    result["train_no_finding_count"] = sum(
        1 for im in train_images if im["id"] not in {a["image_id"] for a in coco["train"]["annotations"]}
    )
    return result


# --------------------------------------------------------------------------- #
# Indicator matrix (14 class-presence + 1 zero_gt = 15 columns FOR
# STRATIFICATION ONLY; the distribution objective uses labels14 alone).
# --------------------------------------------------------------------------- #
def build_indicators(train: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    images = train["images"]
    annotations = train["annotations"]
    categories = sorted(train["categories"], key=lambda c: c["id"])
    category_ids = [c["id"] for c in categories]
    names = [str(c["name"]) for c in categories]
    if category_ids != list(range(1, 15)):
        fail("Train categories are not contiguous 1..14", "SCHEMA_DRIFT")

    image_ids = np.asarray([im["id"] for im in images], dtype=np.int64)
    if len(set(image_ids.tolist())) != len(image_ids):
        fail("Duplicate image IDs in instances_train.json", "SCHEMA_DRIFT")
    image_index = {int(image_id): position for position, image_id in enumerate(image_ids.tolist())}
    category_index = {category_id: position for position, category_id in enumerate(category_ids)}

    labels14 = np.zeros((len(images), 14), dtype=np.int64)
    for annotation in annotations:
        img_pos = image_index.get(annotation["image_id"])
        cat_pos = category_index.get(annotation["category_id"])
        if img_pos is None or cat_pos is None:
            fail("Annotation references an image/category outside instances_train.json", "SCHEMA_DRIFT")
        labels14[img_pos, cat_pos] = 1
    zero_gt = (labels14.sum(axis=1) == 0).astype(np.int64)
    return image_ids, labels14, zero_gt, names


# --------------------------------------------------------------------------- #
# EXACT INTEGER distribution objective (14 abnormal classes only). This is
# the ONLY objective used in the move-acceptance/selection path.
# --------------------------------------------------------------------------- #
def full_train_integer_stats(labels14: np.ndarray) -> tuple[list[int], int, int]:
    """Return (K_c per class [14], S_T = sum_c K_c, N = total train images)."""
    K = labels14.sum(axis=0).astype(np.int64).tolist()
    S_T = int(labels14.sum())
    N = int(labels14.shape[0])
    return K, S_T, N


def _integer_objective_from_aggregate(k: np.ndarray, n: int, s_l: int, K: list[int], S_T: int, N: int) -> tuple[int, int, int]:
    """Same formula as integer_objective(), factored out so the
    equivalence-class two-for-two engine can evaluate a candidate move from
    an incrementally-updated aggregate (k, n, s_l) without re-summing
    labels14 over the full position set on every candidate -- this is a
    pure refactor (single source of truth for the objective formula), not a
    behavior change: integer_objective() below now calls this directly."""
    deviations = np.abs(k.astype(np.int64) * N - np.asarray(K, dtype=np.int64) * n)
    e_max = int(deviations.max())
    e_mean = int(deviations.sum())
    e_lc = int(abs(s_l * N - S_T * n))
    return e_max, e_mean, e_lc


def integer_objective(
    selected_positions: Iterable[int], labels14: np.ndarray, K: list[int], S_T: int, N: int,
) -> tuple[int, int, int]:
    positions = sorted(selected_positions)
    n = len(positions)
    if n == 0:
        fail("Cannot compute objective over an empty selection", "OBJECTIVE_EMPTY")
    k = labels14[positions].sum(axis=0).astype(np.int64)
    s_l = int(k.sum())
    return _integer_objective_from_aggregate(k, n, s_l, K, S_T, N)


def fraction_distribution_report(
    selected_positions: Iterable[int], labels14: np.ndarray, K: list[int], N: int,
) -> tuple[Fraction, Fraction, Fraction, list[Fraction], list[Fraction], list[Fraction], int]:
    """Reporting-only (never used in the selection path): returns
    (D_max, D_mean, D_LC, train_prevalence[14], labeled_prevalence[14],
    absolute_deviation[14], worst_class_index)."""
    positions = sorted(selected_positions)
    n = len(positions)
    k = labels14[positions].sum(axis=0).astype(np.int64).tolist()
    s_l = int(labels14[positions].sum())
    s_t = int(sum(K))
    train_prevalence = [Fraction(K[c], N) for c in range(14)]
    labeled_prevalence = [Fraction(k[c], n) for c in range(14)]
    deviations = [abs(labeled_prevalence[c] - train_prevalence[c]) for c in range(14)]
    d_max = max(deviations)
    d_mean = sum(deviations, Fraction(0)) / 14
    d_lc = abs(Fraction(s_l, n) - Fraction(s_t, N))
    worst_class_index = max(range(14), key=lambda c: deviations[c])
    return d_max, d_mean, d_lc, train_prevalence, labeled_prevalence, deviations, worst_class_index


# --------------------------------------------------------------------------- #
# Repair primitives                                                           #
# --------------------------------------------------------------------------- #
class RepairOutcome:
    OK = "OK"
    REPAIR_INFEASIBLE = "REPAIR_INFEASIBLE"
    COMPUTATIONAL_ABORT = "COMPUTATIONAL_ABORT"
    # R10-B1: a NON-SCIENTIFIC, diagnostic-only outcome. Returned ONLY when
    # repair_objective_local_search was called with a `layer_b_state_sink`
    # (i.e. only ever by the R10 profiler's Layer-B state-reconstruction
    # pass -- see run_two_for_two_hotpath_profile), meaning the caller
    # asked to STOP right before the first two-for-two engine invocation
    # and capture the (locked, selected, pool) state instead of actually
    # running the search. Never returned by the official run, --reconstruct
    # -check, or --benchmark-two-for-two (none of which ever pass
    # layer_b_state_sink). build_budget_step passes this outcome straight
    # through unchanged, exactly like it does for OK/REPAIR_INFEASIBLE/
    # COMPUTATIONAL_ABORT, and the caller must treat it as terminal (no
    # membership was computed or repaired past this point -- `selected` is
    # a mid-repair snapshot, never a valid final candidate).
    LAYER_B_STATE_CAPTURED = "LAYER_B_STATE_CAPTURED"


class DeadlineExceeded(Exception):
    pass


def _check_deadline(deadline: float | None) -> None:
    if deadline is not None and time.monotonic() > deadline:
        raise DeadlineExceeded()


def _move_key(
    trial: set[int], labels14: np.ndarray, K: list[int], S_T: int, N: int,
    seed: int, budget: str, repair_phase: str, move_type: str,
    removed_ids: tuple[int, ...], added_ids: tuple[int, ...],
) -> tuple[Any, ...]:
    e_max, e_mean, e_lc = integer_objective(trial, labels14, K, S_T, N)
    if move_type in ("add", "remove"):
        # exact_size single-image toggle: use the image-priority namespace.
        image_id = added_ids[0] if added_ids else removed_ids[0]
        priority = image_priority_digest(seed, image_id)
    else:
        priority = move_priority_digest(seed, budget, repair_phase, move_type, removed_ids, added_ids)
    identity = canonical_numeric_move_identity(removed_ids, added_ids)
    return (e_max, e_mean, e_lc, priority, identity)


def _log_entry(
    iteration: int, budget: str, phase: str, move_type: str,
    removed_ids: list[int], added_ids: list[int],
    nf_before: int, nf_after: int, missing_before: list[str], missing_after: list[str],
    violation_before: int, violation_after: int,
    objective_before: tuple[int, int, int], objective_after: tuple[int, int, int],
    priority_digest: str, membership_before_sha256: str, membership_after_sha256: str,
    candidate_set_priority_before: str, candidate_set_priority_after: str,
) -> dict[str, Any]:
    return {
        "iteration": iteration, "budget": budget, "repair_phase": phase, "move_type": move_type,
        "removed_ids": removed_ids, "added_ids": added_ids,
        "no_finding_before": nf_before, "no_finding_after": nf_after,
        "missing_classes_before": missing_before, "missing_classes_after": missing_after,
        "violation_before": violation_before, "violation_after": violation_after,
        "integer_objective_before": list(objective_before), "integer_objective_after": list(objective_after),
        "tie_break_priority_digest": priority_digest,
        "membership_sha256_before": membership_before_sha256,
        "membership_sha256_after": membership_after_sha256,
        "candidate_set_priority_before": candidate_set_priority_before,
        "candidate_set_priority_after": candidate_set_priority_after,
    }


def _missing_class_names(members_positions: Iterable[int], labels14: np.ndarray, names: list[str]) -> list[str]:
    positions = sorted(members_positions)
    if not positions:
        return list(names)
    present = set(int(c) for c in np.where(labels14[positions].sum(axis=0) > 0)[0])
    return [names[c] for c in range(14) if c not in present]


def repair_exact_size(
    locked: set[int], new_region_selected: set[int], new_region_pool: set[int],
    target_new_size: int, labels14: np.ndarray, zero_gt: np.ndarray, names: list[str],
    K: list[int], S_T: int, N: int, seed: int, budget: str, image_ids: np.ndarray,
    deadline: float | None, log: list[dict[str, Any]], visited: set[frozenset],
) -> tuple[set[int], str]:
    selected = set(new_region_selected)
    while len(selected) != target_new_size:
        _check_deadline(deadline)
        grow = len(selected) < target_new_size
        candidates = (new_region_pool - selected) if grow else selected
        best: tuple[Any, ...] | None = None
        best_position: int | None = None
        for position in candidates:
            trial = set(selected)
            (trial.add if grow else trial.discard)(position)
            signature = frozenset(trial)
            if signature in visited:
                continue
            image_id = int(image_ids[position])
            removed_ids = () if grow else (image_id,)
            added_ids = (image_id,) if grow else ()
            key = _move_key(locked | trial, labels14, K, S_T, N, seed, budget, "exact_size",
                             "add" if grow else "remove", removed_ids, added_ids)
            if best is None or key < best:
                best, best_position = key, position
        if best is None or best_position is None:
            return selected, RepairOutcome.REPAIR_INFEASIBLE

        violation_before = abs(len(selected) - target_new_size)
        before_state = locked | selected
        before_sha = canonical_membership_sha256(int(v) for v in [image_ids[p] for p in before_state]) if before_state else ""
        before_set_digest = candidate_set_priority_digest(seed, budget, [int(image_ids[p]) for p in before_state]) if before_state else ""
        nf_before = int(sum(zero_gt[i] for i in selected))
        missing_before = _missing_class_names(locked | selected, labels14, names)
        obj_before = integer_objective(before_state, labels14, K, S_T, N) if before_state else (0, 0, 0)

        (selected.add if grow else selected.discard)(best_position)
        visited.add(frozenset(selected))

        image_id = int(image_ids[best_position])
        after_state = locked | selected
        after_sha = canonical_membership_sha256(int(image_ids[p]) for p in after_state)
        after_set_digest = candidate_set_priority_digest(seed, budget, [int(image_ids[p]) for p in after_state])
        nf_after = int(sum(zero_gt[i] for i in selected))
        missing_after = _missing_class_names(locked | selected, labels14, names)
        violation_after = abs(len(selected) - target_new_size)
        obj_after = integer_objective(after_state, labels14, K, S_T, N)

        log.append(_log_entry(
            len(log) + 1, budget, "exact_size", "add" if grow else "remove",
            [] if grow else [image_id], [image_id] if grow else [],
            nf_before, nf_after, missing_before, missing_after,
            violation_before, violation_after, obj_before, obj_after,
            best[3], before_sha, after_sha, before_set_digest, after_set_digest,
        ))
    return selected, RepairOutcome.OK


def repair_exact_no_finding(
    locked: set[int], new_region_selected: set[int], new_region_pool: set[int],
    target_nf: int, zero_gt: np.ndarray, labels14: np.ndarray, names: list[str],
    K: list[int], S_T: int, N: int, seed: int, budget: str, image_ids: np.ndarray,
    deadline: float | None, log: list[dict[str, Any]], visited: set[frozenset],
) -> tuple[set[int], str]:
    selected = set(new_region_selected)
    target_size = len(selected)

    def nf_count(members: set[int]) -> int:
        return int(sum(zero_gt[i] for i in members))

    while nf_count(selected) != target_nf:
        _check_deadline(deadline)
        need_more_nf = nf_count(selected) < target_nf
        if need_more_nf:
            out_pool = [i for i in selected if zero_gt[i] == 0]
            in_pool = [i for i in (new_region_pool - selected) if zero_gt[i] == 1]
        else:
            out_pool = [i for i in selected if zero_gt[i] == 1]
            in_pool = [i for i in (new_region_pool - selected) if zero_gt[i] == 0]
        best: tuple[Any, ...] | None = None
        best_pair: tuple[int, int] | None = None
        for out_pos in out_pool:
            for in_pos in in_pool:
                trial = set(selected)
                trial.discard(out_pos)
                trial.add(in_pos)
                if len(trial) != target_size:
                    continue
                signature = frozenset(trial)
                if signature in visited:
                    continue
                removed_ids = (int(image_ids[out_pos]),)
                added_ids = (int(image_ids[in_pos]),)
                key = _move_key(locked | trial, labels14, K, S_T, N, seed, budget, "exact_no_finding",
                                 "one_for_one_swap", removed_ids, added_ids)
                if best is None or key < best:
                    best, best_pair = key, (out_pos, in_pos)
        if best is None or best_pair is None:
            return selected, RepairOutcome.REPAIR_INFEASIBLE

        out_pos, in_pos = best_pair
        violation_before = abs(nf_count(selected) - target_nf)
        before_state = locked | selected
        before_sha = canonical_membership_sha256(int(image_ids[p]) for p in before_state)
        before_set_digest = candidate_set_priority_digest(seed, budget, [int(image_ids[p]) for p in before_state])
        nf_before = nf_count(selected)
        missing_before = _missing_class_names(locked | selected, labels14, names)
        obj_before = integer_objective(before_state, labels14, K, S_T, N)

        selected.discard(out_pos)
        selected.add(in_pos)
        visited.add(frozenset(selected))

        after_state = locked | selected
        after_sha = canonical_membership_sha256(int(image_ids[p]) for p in after_state)
        after_set_digest = candidate_set_priority_digest(seed, budget, [int(image_ids[p]) for p in after_state])
        nf_after = nf_count(selected)
        missing_after = _missing_class_names(locked | selected, labels14, names)
        violation_after = abs(nf_after - target_nf)
        obj_after = integer_objective(after_state, labels14, K, S_T, N)

        log.append(_log_entry(
            len(log) + 1, budget, "exact_no_finding", "one_for_one_swap",
            [int(image_ids[out_pos])], [int(image_ids[in_pos])],
            nf_before, nf_after, missing_before, missing_after,
            violation_before, violation_after, obj_before, obj_after,
            best[3], before_sha, after_sha, before_set_digest, after_set_digest,
        ))
    return selected, RepairOutcome.OK


# --------------------------------------------------------------------------- #
# ========================= LEGACY / NON-ACTIVE (R11) ======================= #
# Everything from here down to the end of _equivalence_class_two_for_two_
# search is RETAINED FOR HISTORICAL REPRODUCIBILITY AND ITS EXISTING UNIT
# TESTS ONLY. It is NOT part of the R11 active construction protocol:
#   - it is never called by build_budget_step, compute_all_budgets (official
#     materialization), run_reconstruct_check, or run_operational_benchmark;
#   - it never gates materialization or preflight;
#   - PROOF 1/2/3 below are HISTORICAL records of what was proven in R3-R5.
#     They are NOT active protocol claims. In particular, "exhaustive
#     two-for-two search", "two-for-two local optimum", and "mathematical
#     completeness over the two-for-two neighborhood" are NO LONGER CLAIMED
#     by this project's active protocol. The only active claim is the
#     one-for-one claim stated in the module docstring.
# Do not re-wire any active code path to this block without an explicit
# researcher protocol decision.
# --------------------------------------------------------------------------- #
# HISTORICAL: R3 review item 5 / R4 blocker R3-B1 -- two-for-two neighborhood
# complexity.
#
# PROOF 1 (mixed composition is redundant with one-for-one -- unchanged
# from R3, still holds):
#   A "mixed" move removes {a (abnormal), b (No-Finding)} and adds
#   {c (abnormal), d (No-Finding)}. By construction (build_indicators:
#   zero_gt[i] = 1 iff labels14[i, :] is the all-zero row), b and d each
#   contribute the all-zero vector to labels14. Therefore the net delta this
#   move induces in (k_1..k_14) and hence in integer_objective() and in
#   set-coverage is EXACTLY the delta of the single one-for-one swap
#   (a -> c) alone. The one-for-one search already enumerates every
#   same-zero_gt swap, which includes every abnormal-for-abnormal pair. So
#   any move a mixed two-for-two could ever usefully make is already found
#   by the existing one-for-one search; it is safe to drop with a complete
#   proof of no lost completeness, not a heuristic approximation.
#
# PROOF 2 (R4: same-composition, 2-abnormal <-> 2-abnormal -- equivalence-
# class signature search is PROVEN EXACTLY EQUIVALENT to exhaustive concrete
# enumeration):
#   Group every ABNORMAL image (zero_gt=0) by its exact 14-bit
#   class-presence vector into equivalence classes. Two images in the same
#   equivalence class are, by definition, INTERCHANGEABLE with respect to
#   labels14: vec(x) == vec(y) for any x, y in the same class. A
#   same-composition two-for-two move removes {out1, out2} (subset of
#   selected) and adds {in1, in2} (subset of pool); its effect on the
#   aggregate class-count vector k is EXACTLY
#     delta = vec(in1) + vec(in2) - vec(out1) - vec(out2).
#   This delta -- and therefore integer_objective() (which is a pure
#   function of the resulting aggregate k, n, s_l; see
#   _integer_objective_from_aggregate) and set-coverage (which classes have
#   k_c > 0) -- depends ONLY on which EQUIVALENCE CLASSES out1/out2/in1/in2
#   belong to (their "signature"), never on which concrete member is
#   chosen. Consequently: sorting the FULL concrete neighborhood by
#   key = (hard-constraint term, integer objective, priority, identity) is
#   PROVABLY EQUIVALENT to: (1) sorting all SIGNATURE-PAIR transitions by
#   (hard-constraint term, integer objective) alone -- since this prefix is
#   provably constant within a signature-pair, no concrete move belonging
#   to a signature-pair with a strictly worse prefix can ever be selected,
#   by construction of lexicographic tuple ordering; then (2) resolving the
#   SHA-256 tie-break ONLY among the concrete realizations of the
#   signature-pair(s) that are tied for the best prefix (an image_priority/
#   move_priority digest is a cryptographic hash of the CONCRETE ids and
#   has no exploitable structure, so THIS step, and only this step, must
#   still enumerate the concrete Cartesian product of the winning
#   signature(s) -- see _equivalence_class_two_for_two_search below).
#   Answers to the R4-mandated analysis questions:
#     1. Complete vs. the full concrete neighborhood? YES -- proven above;
#        every concrete quadruple is represented by exactly one
#        signature-pair, and the two-phase search never discards a
#        signature-pair that could contain the true winner.
#     2. Does keeping one representative per signature-transition lose the
#        SHA-256 tie-break winner? NO, PROVIDED tie-break resolution
#        (phase 2) is performed over ALL concrete realizations of every
#        signature-pair tied for the best objective/hard-constraint prefix,
#        not just one representative -- which is exactly what the
#        implementation below does.
#     3. Can the min-priority concrete move be found per signature without
#        enumerating its Cartesian product? NO. SHA-256 is a cryptographic
#        hash with no order-preserving structure over its inputs; the
#        minimum-digest element of a set can only be found by hashing every
#        element. This bounds phase 2's cost to the concrete size of
#        ONLY the tied-winning signature-pair(s), not the full
#        neighborhood -- still a real, sometimes large, cost in principle
#        (see complexity below), but strictly bounded by the winning
#        group(s) rather than by the entire selected x pool space.
#     4. Complexity -- CORRECTED in R5 (fixes blocker R4-B2; R4's prose here
#        was imprecise about Phase 2). In S=|selected|, P=|pool|, m=
#        |distinct label vectors in selected|, p=|distinct label vectors in
#        pool|, R=number of valid removed-pair signatures (<= m + C(m,2)),
#        A=number of valid added-pair signatures (<= p + C(p,2)), W=the set
#        of signature transitions tied for the best Phase-1 prefix, and for
#        each w in W, C_out(w)/C_in(w)=the number of CONCRETE removed/added
#        image-pairs realizing w's removed-/added-signature-pair (for a
#        same-signature pair {c,c}, that count is C(|class c|,2); for a
#        different-signature pair {c1,c2}, it is |class c1|*|class c2|):
#          Phase 1 (signature search): O(R * A), O(1) work per signature-
#          pair via the incrementally-updated aggregate (no re-summing
#          labels14). NOT O(max equivalence-class size^2) -- that described
#          neither phase correctly.
#          Phase 2 (tie-break): O(sum_{w in W} C_out(w) * C_in(w)) --
#          bounded by the TIED-BEST transitions' own concrete Cartesian
#          products only, never by all R*A of them and never by a single
#          "max class size" term (a winning transition's cost is the
#          PRODUCT of its two sides' concrete sizes, not one side alone).
#          Total: O(R*A + sum_{w in W} C_out(w)*C_in(w)).
#        See PROOF 3 (R5) below for the streaming-search memory bound and
#        the deadline/watchdog addition.
#     5. Worst-case bound: if every abnormal image has a DISTINCT label
#        vector (m=S, p=P, the adversarial case), R=O(S^2), A=O(P^2), so
#        Phase 1 is O(S^2 * P^2) -- IDENTICAL order to brute force, i.e.
#        NO WORST-CASE IMPROVEMENT IS PROVEN. However, in this exact
#        adversarial case every equivalence class has exactly 1 member, so
#        Phase 2 is trivial (each signature has exactly one concrete
#        realization). Expected bound on VinDr-CXR/VinBigData (14 fixed
#        abnormality classes, images typically carrying 1-3 co-occurring
#        findings) is that the number of DISTINCT label-vector patterns
#        actually observed is very plausibly far smaller than S or P
#        (bounded above by 2^14-1 = 16,383 possible nonempty patterns, and
#        clinically far fewer common co-occurrence patterns are expected in
#        practice) -- but this is a PLAUSIBLE EXPECTATION, NOT an empirical
#        measurement: Claude has not inspected the real instances_train.json
#        label-vector distribution in this revision (barred from running
#        code against real data), so no numeric expected-case bound is
#        asserted as fact.
#     6. Condition for COMPLETE_EXHAUSTION_EQUIVALENT_TO_CONCRETE_TWO_FOR_TWO:
#        the mathematical equivalence proof above holds UNCONDITIONALLY
#        (independent of the actual data) -- so CORRECTNESS is proven
#        regardless of dataset. What is NOT proven unconditionally is
#        OPERATIONAL TRACTABILITY (finishing in practical wall-clock time),
#        which depends on the real dataset's equivalence-class structure.
#        Per this project's own precedent for training_authorized (a
#        researcher/GPT protocol decision, never a Claude-asserted fact),
#        `two_for_two_engine.operationally_approved` is likewise a locked,
#        human-owned config field, defaulting to false, and is the gate
#        that separates "mathematically proven correct" from "cleared to
#        run against real data" -- see NHIEM VU 4 / R4-4 below.
#     7. Additional proof obligations / guardrails implemented: an
#        oracle-equivalence test suite comparing this engine against a
#        brute-force concrete enumerator (kept for synthetic-tiny-data
#        testing ONLY, see _brute_force_same_composition_two_for_two_pairs)
#        across cases with multilabel co-occurrence, an improving 2-swap
#        with no improving 1-swap, an objective tie resolved by SHA-256
#        priority, a full tie resolved by canonical numeric identity (via a
#        mocked digest function), immutable-nested-prefix preservation,
#        exact-NF/14-of-14-coverage preservation, and a check that no k>=3
#        move is ever produced. See tests/test_phase2F_labeled_unlabeled_
#        guardrails.py's TestEquivalenceClassEngineVsBruteForceOracle.
#
# PROOF 3 (R5: streaming best-prefix search is equivalent to "collect all
# accepted transitions, sort, take the first tied group", fixing blocker
# R4-B4; plus the deadline/watchdog addition, fixing blocker R4-B3):
#   Phase 1 previously appended every ACCEPTED (key_prefix, r_pair, a_pair)
#   triple to a list, sorted the whole list, then grouped by the first
#   (smallest) prefix value -- O(R*A) auxiliary memory. R5 replaces this
#   with the standard streaming-minimum/argmin pattern: maintain
#   best_prefix (initially None) and tied_best_transitions (initially
#   empty); for each accepted transition, compare its prefix against
#   best_prefix and RESET (best_prefix := prefix, tied_best_transitions :=
#   [transition]) if strictly better, APPEND if exactly tied, or DISCARD
#   (do nothing) if strictly worse. This is exactly a running-minimum scan:
#   because every accepted transition is compared against the CURRENT
#   best_prefix exactly once, and best_prefix only ever strictly decreases
#   (in the tuple order), at the end of the scan best_prefix necessarily
#   equals the true minimum prefix over every transition that was ever
#   accepted, and tied_best_transitions necessarily contains EXACTLY the
#   transitions whose prefix equals that true minimum -- a transition tied
#   with the eventual winner is never lost, because at the moment it is
#   seen it either (a) matches the best_prefix seen so far and is appended,
#   and it can only later be dropped by a RESET, which happens only when a
#   STRICTLY better prefix is found -- but a strictly better prefix would
#   also have excluded this transition from the true final tied group, so
#   dropping it is correct, not a loss; or (b) itself triggers a RESET,
#   becoming (part of) the new best_prefix group. So streaming produces an
#   IDENTICAL tied_best_transitions set to "sort everything, take the first
#   group", with O(|tied_best_transitions|) auxiliary memory instead of
#   O(R*A). If Phase 2 finds no available move within tied_best_transitions
#   (every concrete realization already in `visited`), the ORIGINAL
#   (sort-based) implementation fell through to the NEXT-smallest prefix
#   group; R5 preserves this exactly by re-running the streaming Phase-1
#   scan with an exclusive lower bound (skip any prefix <= the previous
#   best_prefix) until either a move is found or no further accepted
#   transition remains -- still never materializing more than one
#   candidate group's worth of transitions in memory at a time, at the cost
#   of re-scanning the R*A signature-pair space once per exhausted group
#   (expected to be the rare/edge-case path, not the common one).
#   Deadline/watchdog (R4-B3): both the (possibly repeated) Phase-1 scan
#   and the Phase-2 concrete enumeration now poll _check_deadline every
#   TWO_FOR_TWO_DEADLINE_WATCHDOG_INTERVAL candidates examined (a single
#   counter shared across every Phase-1 rescan and the Phase-2 loop within
#   one call, so a pathological multi-group-exhaustion case still respects
#   the deadline promptly). This interval is PURELY a check-frequency
#   optimization (avoiding a monotonic-clock read on every single
#   candidate): it never skips a candidate, never changes which candidate
#   is accepted, and never alters the returned result when deadline=None
#   (in which case _check_deadline is an unconditional no-op). On
#   expiry, DeadlineExceeded propagates out of this function uncaught (it
#   is caught only by build_budget_step) -- see repair_min_class_coverage/
#   repair_objective_local_search/build_budget_step for the full
#   propagation chain.
# --------------------------------------------------------------------------- #
def _brute_force_same_composition_two_for_two_pairs(
    selected_list: list[int], pool_list: list[int], zero_gt: np.ndarray,
) -> Iterable[tuple[int, int, int, int]]:
    """TEST-ONLY brute-force oracle: yields every (out1, out2, in1, in2)
    quadruple in the same-composition (both removed AND both added
    abnormal, zero_gt=0) two-for-two neighborhood by naive
    O(|selected|^2 * |pool|^2) enumeration. This function must NEVER be
    called against real Phase 2F construction (only against small
    synthetic fixtures in tests, to validate
    _equivalence_class_two_for_two_search for exact equivalence). The
    mixed-composition branch is intentionally not enumerated here either
    (see PROOF 1 above); no k>=3 move is ever generated."""
    for out1 in selected_list:
        for out2 in selected_list:
            if out2 <= out1:
                continue
            if zero_gt[out1] != 0 or zero_gt[out2] != 0:
                continue
            for in1 in pool_list:
                for in2 in pool_list:
                    if in2 <= in1:
                        continue
                    if zero_gt[in1] != 0 or zero_gt[in2] != 0:
                        continue
                    yield out1, out2, in1, in2


def _abnormal_equivalence_classes(positions: list[int], labels14: np.ndarray) -> dict[tuple[int, ...], list[int]]:
    """Group ABNORMAL positions (caller must pre-filter to zero_gt=0) by
    their exact 14-bit class-presence vector, represented canonically as a
    sorted tuple of present class indices. Returns
    {vector_signature: [positions sharing that exact vector]}."""
    groups: dict[tuple[int, ...], list[int]] = {}
    for p in positions:
        sig = tuple(int(c) for c in np.where(labels14[p] > 0)[0])
        groups.setdefault(sig, []).append(p)
    return groups


def _signature_vector(sig: tuple[int, ...]) -> np.ndarray:
    v = np.zeros(14, dtype=np.int64)
    for c in sig:
        v[c] = 1
    return v


# --------------------------------------------------------------------------- #
# LEGACY / NON-ACTIVE (R11) -- former R10 measurement helpers for the
# two-for-two hot path. The R10 profiler MODE (its CLI flags, its config
# block, its orchestration function run_two_for_two_hotpath_profile and its
# report writer) has been REMOVED from the active protocol in R11; these
# helper functions are retained only alongside the legacy engine they
# measured, and are now reachable ONLY from
# _equivalence_class_two_for_two_search's own legacy `hotpath_profile`
# parameter, which no active code path ever supplies. They never returned,
# promoted, or influenced a scientific move, and they never will.
# --------------------------------------------------------------------------- #
def _calibrate_observer_overhead(iterations: int = 200_000) -> dict[str, Any]:
    """Calibrates THIS profiler's own instrumentation overhead so that
    component-pass timing differences elsewhere in the hotpath profile
    report can be labeled honestly as instrumentation-affected estimates,
    never exact isolated causal costs (TASK 4). Never touches production
    state, never reads training data, never affects any scientific
    decision. `iterations` is a pure calibration-loop trip count, not a
    transition count."""
    n = max(1, int(iterations))

    t0 = time.monotonic()
    x = 0
    for _ in range(n):
        x += 1
    empty_loop_total = time.monotonic() - t0

    t0 = time.monotonic()
    for _ in range(n):
        time.monotonic()
    perf_counter_total = time.monotonic() - t0

    t0 = time.monotonic()
    wd = 0
    for _ in range(n):
        wd += 1
        _ = wd % TWO_FOR_TWO_DEADLINE_WATCHDOG_INTERVAL
    watchdog_no_check_total = time.monotonic() - t0

    far_future_deadline = time.monotonic() + 3600.0
    check_iterations = max(1, n // 20)
    t0 = time.monotonic()
    for _ in range(check_iterations):
        _check_deadline(far_future_deadline)  # never expires -- measures the real per-call cost only
    check_deadline_total = time.monotonic() - t0

    return {
        "calibration_iterations": n,
        "empty_loop_cost_seconds_per_iteration": empty_loop_total / n,
        "perf_counter_call_cost_seconds": perf_counter_total / n,
        "watchdog_no_check_tick_cost_seconds_per_iteration": watchdog_no_check_total / n,
        "check_deadline_actual_cost_seconds_per_call": check_deadline_total / check_iterations,
        "note": (
            "Coarse, machine- and run-condition-dependent calibration of this profiler's own "
            "instrumentation overhead, run fresh on every profiler invocation (never cached/reused "
            "across runs). Component-pass timing differences elsewhere in this report are "
            "instrumentation-affected estimates, not exact isolated causal costs -- this calibration "
            "exists so a small delta can be judged against plausible noise/overhead rather than "
            "asserted as a real cost difference."
        ),
    }


def _run_production_like_complete_prefix_pass(
    prefix: list[tuple[int, int]], removed_vecs: list[np.ndarray], removed_sums: list[int],
    added_vec_matrix: np.ndarray, added_sum_array: np.ndarray,
    base_k: np.ndarray, base_n: int, base_s_l: int, accept_and_rank: Any,
) -> dict[str, Any]:
    """R10-B3: ONE independently-timed pass over the SAME ordered
    deterministic prefix _run_hotpath_microbenchmark builds, performing
    every real-per-transition operation inside a SINGLE loop with NO
    per-operation timers inside it: a watchdog tick (diagnostic/no-expiry
    mode -- deadline=None, always a documented no-op, since this pass must
    never depend on or be bounded by the real search's own deadline),
    added-row/sum lookup, new_k/new_s_l construction, the REAL
    accept_and_rank call, and the exact reset-on-strictly-better/append-
    on-tie/discard-on-worse best-prefix/tied-list maintenance
    _stream_best_prefix_above performs. This is the correct, non-
    overlapping end-to-end per-transition cost measurement -- NEVER a sum
    of the separate, deliberately-overlapping component passes in
    _run_hotpath_microbenchmark. Never returns, promotes, or uses a
    scientific move; the accepted move (if any) found here is discarded,
    never surfaced to any caller."""
    n = len(prefix)
    if n == 0:
        return {"production_like_complete_prefix_seconds": None, "average_seconds_per_transition": None}
    watchdog_counter = 0
    best_prefix: tuple[Any, ...] | None = None
    tied: list[int] = []
    t0 = time.monotonic()
    for i, (_r_idx, a_idx) in enumerate(prefix):
        watchdog_counter += 1
        if watchdog_counter % TWO_FOR_TWO_DEADLINE_WATCHDOG_INTERVAL == 0:
            _check_deadline(None)
        added_vec = added_vec_matrix[a_idx]
        added_sum = int(added_sum_array[a_idx])
        new_k = base_k - removed_vecs[i] + added_vec
        new_s_l = base_s_l - removed_sums[i] + added_sum
        key_prefix = accept_and_rank(new_k, base_n, new_s_l)
        if key_prefix is None:
            continue
        if best_prefix is None or key_prefix < best_prefix:
            best_prefix = key_prefix
            tied = [i]
        elif key_prefix == best_prefix:
            tied.append(i)
        # else: strictly worse than best_prefix seen so far -- DISCARD.
    elapsed = time.monotonic() - t0
    return {
        "production_like_complete_prefix_seconds": elapsed,
        "average_seconds_per_transition": elapsed / n,
    }


def _run_hotpath_microbenchmark(
    removed_pairs: list[tuple[tuple[int, ...], tuple[int, ...]]],
    added_pairs: list[tuple[tuple[int, ...], tuple[int, ...]]],
    added_vec_matrix: np.ndarray, added_sum_array: np.ndarray,
    base_k: np.ndarray, base_n: int, base_s_l: int,
    K: list[int], S_T: int, N: int, accept_and_rank: Any,
    prefix_count_configured: int,
) -> dict[str, Any]:
    """TEST/PROFILER-ONLY controlled hot-path microbenchmark (TASK 3 Layer
    B, TASK 5 duplicate-state diagnostics). Builds a deterministic PREFIX
    of (removed_pair, added_pair) transitions in the EXACT SAME
    removed-pairs-outer / added-pairs-inner order the real Phase-1 search
    (_stream_best_prefix_above) visits them in, then runs several
    CUMULATIVE, separately-timed passes over that fixed prefix -- never a
    per-iteration timer inside one combined loop, since timer overhead
    would dominate at this granularity. Each pass adds exactly one more
    operation layer than the previous one, so consecutive-pass deltas
    isolate that layer's own (instrumentation-affected) cost. Reuses the
    REAL `accept_and_rank` closure -- never a re-derived copy -- as ground
    truth: every hand-computed component is cross-checked against it, and
    ANY disagreement is a hard fail (PROFILE_SEMANTIC_EQUIVALENCE_FAILURE),
    never a silently-ignored divergence. This function NEVER returns,
    promotes, or uses a scientific move; it only measures."""
    total_possible = len(removed_pairs) * len(added_pairs)
    prefix_count = max(0, min(int(prefix_count_configured), total_possible))

    prefix: list[tuple[int, int]] = []  # (removed_pair_index, added_pair_index)
    removed_vecs: list[np.ndarray] = []
    removed_sums: list[int] = []
    for r_idx, (r_ca, r_cb) in enumerate(removed_pairs):
        if len(prefix) >= prefix_count:
            break
        r_vec = _signature_vector(r_ca) + _signature_vector(r_cb)
        r_sum = int(r_vec.sum())
        for a_idx in range(len(added_pairs)):
            if len(prefix) >= prefix_count:
                break
            prefix.append((r_idx, a_idx))
            removed_vecs.append(r_vec)
            removed_sums.append(r_sum)

    representativeness_caveat = (
        "Deterministic-PREFIX evidence only. This is NOT a random or representative sample of the "
        "full R*A transition space and must never be treated as one -- it is the exact FIRST "
        "len(prefix) transitions the real Phase-1 search would visit, in the real traversal order, "
        "nothing more."
    )
    n = len(prefix)
    if n == 0:
        return {
            "prefix_count_used": 0, "prefix_count_configured": int(prefix_count_configured),
            "total_transition_count_r_times_a": total_possible,
            "deterministic_prefix_order": "removed_pairs_outer_added_pairs_inner",
            "representativeness_caveat": representativeness_caveat,
            "components": {},
            "component_overlap_disclosure": {
                "component_timings_overlap_semantically": True,
                "component_timings_must_not_be_summed_for_end_to_end_cost": True,
                "note": "No transitions in this (empty) prefix; no component passes ran.",
            },
            "production_like_complete_prefix_seconds": None,
            "production_like_average_seconds_per_transition": None,
            "coverage_pass_fraction": None,
            "strictly_improving_fraction_among_coverage_passing": None,
            "semantic_checksum_sha256": None, "semantic_equivalence_verified": True,
            "duplicate_state_diagnostics": {
                "prefix_count_considered": 0, "unique_state_count": 0, "repeated_state_count": 0,
                "duplicate_state_ratio": None, "max_multiplicity": 0,
                "unique_accept_rank_prefix_count": 0, "repeated_accepted_prefix_count": 0,
                "representativeness_caveat": representativeness_caveat,
            },
        }

    components: dict[str, float] = {}

    # Component 1: loop/counter overhead.
    t0 = time.monotonic()
    _c = 0
    for _ in range(n):
        _c += 1
    components["loop_counter_overhead_seconds"] = time.monotonic() - t0

    # Component 2: watchdog function-call overhead (replica tick logic --
    # mirrors _tick_watchdog's increment+modulo exactly; deadline=None
    # inside _check_deadline is always a documented no-op).
    t0 = time.monotonic()
    _wd = 0
    for _ in range(n):
        _wd += 1
        if _wd % TWO_FOR_TWO_DEADLINE_WATCHDOG_INTERVAL == 0:
            _check_deadline(None)
    components["watchdog_tick_overhead_seconds"] = time.monotonic() - t0

    # Component 3: added-matrix row lookup and scalar-sum lookup.
    t0 = time.monotonic()
    for _r_idx, a_idx in prefix:
        _ = added_vec_matrix[a_idx]
        _ = int(added_sum_array[a_idx])
    components["added_matrix_lookup_seconds"] = time.monotonic() - t0

    # Component 4: new_k / new_s_l construction (arithmetic combine only;
    # removed_vec/removed_sum were already computed once per removed pair
    # above, exactly mirroring the real hoisted-per-removed-pair cost).
    t0 = time.monotonic()
    new_ks: list[np.ndarray] = [None] * n  # type: ignore[list-item]
    new_s_ls: list[int] = [0] * n
    for i, (_r_idx, a_idx) in enumerate(prefix):
        added_vec = added_vec_matrix[a_idx]
        added_sum = int(added_sum_array[a_idx])
        new_ks[i] = base_k - removed_vecs[i] + added_vec
        new_s_ls[i] = base_s_l - removed_sums[i] + added_sum
    components["new_k_construction_seconds"] = time.monotonic() - t0

    # Component 5 (+ byproduct: percentage passing 14/14 coverage).
    t0 = time.monotonic()
    coverage_pass = [False] * n
    for i in range(n):
        coverage_pass[i] = int((new_ks[i] > 0).sum()) == 14
    components["coverage_check_seconds"] = time.monotonic() - t0
    coverage_pass_count = sum(coverage_pass)
    coverage_pass_fraction = coverage_pass_count / n

    # Component 7: integer-objective evaluation, restricted to
    # coverage-passing transitions only (matching accept_and_rank's own
    # short-circuit order). Reuses the real, unmodified
    # _integer_objective_from_aggregate -- never a re-derived formula.
    t0 = time.monotonic()
    trial_objs: list[tuple[int, int, int] | None] = [None] * n
    for i in range(n):
        if coverage_pass[i]:
            trial_objs[i] = _integer_objective_from_aggregate(new_ks[i], base_n, new_s_ls[i], K, S_T, N)
    components["objective_evaluation_seconds"] = time.monotonic() - t0

    # Ground truth: the REAL accept_and_rank closure, called once per
    # prefix transition -- never re-derived. Also the only source of the
    # "strictly improving" decision (its threshold, `current_obj`, lives
    # only inside this closure's own captured scope, deliberately never
    # duplicated here).
    t0 = time.monotonic()
    real_results: list[tuple[Any, ...] | None] = [
        accept_and_rank(new_ks[i], base_n, new_s_ls[i]) for i in range(n)
    ]
    components["real_accept_and_rank_call_seconds"] = time.monotonic() - t0
    accepted_count = sum(1 for r in real_results if r is not None)
    strictly_improving_fraction_among_coverage_passing = (
        (accepted_count / coverage_pass_count) if coverage_pass_count else None
    )

    # Component 9: best-prefix / tied-list maintenance (mirrors the
    # reset-on-strictly-better / append-on-tie / discard-on-worse logic in
    # _stream_best_prefix_above exactly, over the real accept_and_rank
    # results computed above).
    t0 = time.monotonic()
    best_prefix: tuple[Any, ...] | None = None
    tied_count = 0
    for i in range(n):
        key_prefix = real_results[i]
        if key_prefix is None:
            continue
        if best_prefix is None or key_prefix < best_prefix:
            best_prefix = key_prefix
            tied_count = 1
        elif key_prefix == best_prefix:
            tied_count += 1
    components["best_prefix_maintenance_seconds"] = time.monotonic() - t0

    # R10-B3: component timers above are SEPARATE, CUMULATIVE-BUT-
    # OVERLAPPING passes (e.g. component 4's pass re-does component 3's
    # lookup work; component 9's pass re-does component 7's real-call
    # work) -- they exist to isolate each layer's OWN incremental cost via
    # consecutive-pass deltas, never to be summed into an end-to-end
    # estimate (summing them double-counts the overlapping work every
    # later pass repeats). The previous round's
    # complete_current_r9_transition_cost_seconds field did exactly that
    # (summed disjoint-looking but actually-overlapping component times)
    # and has been REMOVED; see production_like_complete_prefix_seconds
    # below for the correct, independently-measured end-to-end figure.
    component_overlap_disclosure = {
        "component_timings_overlap_semantically": True,
        "component_timings_must_not_be_summed_for_end_to_end_cost": True,
        "note": (
            "Each component pass above re-executes some of the work already performed by an earlier "
            "pass (component N's loop calls everything component N-1's loop called, plus one more "
            "operation layer) -- this is intentional (it isolates each layer's own incremental cost via "
            "consecutive-pass deltas, per the R10 design), but it means summing these component times "
            "would double-count overlapping work and is NOT a valid way to estimate end-to-end "
            "per-transition cost. Differences between consecutive passes are instrumentation-affected "
            "estimates (see observer_effect_calibration in the top-level report), not exact isolated "
            "causal costs. Use production_like_complete_prefix_seconds below for the independently-"
            "measured, single-pass end-to-end figure."
        ),
    }

    # R10-B3: ONE independent, single-timer pass over the SAME ordered
    # prefix, performing every real-per-transition operation (counter,
    # watchdog tick, added-row/sum lookup, new_k/new_s_l construction, the
    # REAL accept_and_rank call, best-prefix/tied maintenance) inside ONE
    # loop with NO per-operation timers inside it -- see
    # _run_production_like_complete_prefix_pass. This -- not a sum of the
    # overlapping component passes above -- is the correct end-to-end
    # per-transition cost estimate.
    production_like = _run_production_like_complete_prefix_pass(
        prefix, removed_vecs, removed_sums, added_vec_matrix, added_sum_array,
        base_k, base_n, base_s_l, accept_and_rank,
    )

    # Semantic-equivalence verification (fail-closed, TASK 8
    # PROFILE_SEMANTIC_EQUIVALENCE_FAILURE): every hand-computed component
    # must agree with the real accept_and_rank closure's own decisions.
    mismatches: list[tuple[int, str]] = []
    for i in range(n):
        if not coverage_pass[i] and real_results[i] is not None:
            mismatches.append((i, "hand_computed_coverage_fail_but_real_accept_and_rank_accepted"))
        if coverage_pass[i] and real_results[i] is not None and trial_objs[i] != real_results[i]:
            mismatches.append((i, "hand_computed_objective_disagrees_with_real_accept_and_rank"))
    if mismatches:
        fail(
            f"Hotpath microbenchmark component pass disagreed with the real accept_and_rank closure "
            f"on {len(mismatches)} of {n} deterministic-prefix transitions "
            f"(first mismatch: {mismatches[0]!r}); refusing to report untrustworthy diagnostic evidence.",
            "PROFILE_SEMANTIC_EQUIVALENCE_FAILURE",
        )

    checksum_payload = json.dumps(
        [list(r) if r is not None else None for r in real_results], sort_keys=True, default=str,
    )
    semantic_checksum_sha256 = hashlib.sha256(checksum_payload.encode("utf-8")).hexdigest()

    # TASK 5 -- duplicate-state diagnostics over the SAME deterministic
    # prefix. Purely observational: never memoizes, collapses, or skips a
    # production transition; never claims the prefix is representative of
    # the full R*A space.
    state_counts: dict[tuple[Any, ...], int] = {}
    for i in range(n):
        key = (tuple(int(v) for v in new_ks[i]), new_s_ls[i])
        state_counts[key] = state_counts.get(key, 0) + 1
    unique_state_count = len(state_counts)
    repeated_state_count = sum(1 for c in state_counts.values() if c > 1)
    max_multiplicity = max(state_counts.values()) if state_counts else 0
    duplicate_state_ratio = (n - unique_state_count) / n

    accept_prefix_counts: dict[tuple[Any, ...], int] = {}
    for r in real_results:
        if r is None:
            continue
        accept_prefix_counts[r] = accept_prefix_counts.get(r, 0) + 1
    unique_accept_rank_prefix_count = len(accept_prefix_counts)
    repeated_accepted_prefix_count = sum(1 for c in accept_prefix_counts.values() if c > 1)

    return {
        "prefix_count_used": n, "prefix_count_configured": int(prefix_count_configured),
        "total_transition_count_r_times_a": total_possible,
        "deterministic_prefix_order": "removed_pairs_outer_added_pairs_inner",
        "representativeness_caveat": representativeness_caveat,
        "components": components,
        "component_overlap_disclosure": component_overlap_disclosure,
        "production_like_complete_prefix_seconds": production_like["production_like_complete_prefix_seconds"],
        "production_like_average_seconds_per_transition": production_like["average_seconds_per_transition"],
        "coverage_pass_fraction": coverage_pass_fraction,
        "strictly_improving_fraction_among_coverage_passing": strictly_improving_fraction_among_coverage_passing,
        "semantic_checksum_sha256": semantic_checksum_sha256,
        "semantic_equivalence_verified": True,
        "duplicate_state_diagnostics": {
            "prefix_count_considered": n,
            "unique_state_count": unique_state_count,
            "repeated_state_count": repeated_state_count,
            "duplicate_state_ratio": duplicate_state_ratio,
            "max_multiplicity": max_multiplicity,
            "unique_accept_rank_prefix_count": unique_accept_rank_prefix_count,
            "repeated_accepted_prefix_count": repeated_accepted_prefix_count,
            "representativeness_caveat": representativeness_caveat,
        },
    }


def _equivalence_class_two_for_two_search(
    locked: set[int], selected: set[int], pool: set[int], labels14: np.ndarray, zero_gt: np.ndarray,
    K: list[int], S_T: int, N: int, seed: int, budget: str, repair_phase: str, image_ids: np.ndarray,
    visited: set[frozenset], accept_and_rank: Any,
    deadline: float | None = None, diagnostics: dict[str, Any] | None = None,
    hotpath_profile: dict[str, Any] | None = None,
) -> tuple[int, int, int, int, tuple[Any, ...]] | None:
    """Exact, deterministic, equivalence-class search over the
    same-composition (2 abnormal <-> 2 abnormal) two-for-two neighborhood.
    Mathematically proven equivalent to exhaustive concrete enumeration --
    see PROOF 2 above. Returns (out1, out2, in1, in2, full_key) for the
    single best allowed move, or None if none exists.

    accept_and_rank(new_k: np.ndarray, new_n: int, new_s_l: int) ->
        key_prefix_tuple_or_None
    where key_prefix_tuple_or_None is None if the resulting aggregate state
    is not an allowed/accepted move, else the ranking prefix EXCLUDING
    priority/identity (those are resolved only for the tied-best group, per
    PROOF 2 answer to question 2/3).

    R5 (fixes blockers R4-B3/R4-B4, see PROOF 3 above): Phase 1 is a
    STREAMING best-prefix search (O(|tied group|) auxiliary memory, never
    O(R*A)); both Phase 1 and Phase 2 poll `deadline` (if not None) every
    TWO_FOR_TWO_DEADLINE_WATCHDOG_INTERVAL candidates.

    R6 (fixes blockers R5-B2/R5-B3): `_check_deadline`
    is now ALSO polled once at the very top of this function and once
    immediately before every normal return (the trivial "fewer than 2
    abnormal images" early return, the "exhausted" return, and the
    "completed" return) -- a BOUNDARY check, not a candidate cap, so a
    neighborhood smaller than the watchdog interval (or a deadline that
    expires in the gap between the last enumerated candidate and the
    function actually returning) is still caught. If `deadline` is None,
    every one of these extra checks is a documented no-op (see
    _check_deadline), so this never changes behavior/results.

    On DeadlineExceeded (from any of the above), this function -- unlike
    R5 -- DOES catch it, but ONLY to finalize `diagnostics` (if a dict was
    passed) with status="aborted", aborted_by_deadline=True, and every
    counter known at the moment of abort (fields not yet computed are left
    as None, never fabricated as 0 -- see the field-by-field comment in
    the except block below), before RE-RAISING the exact same exception
    unchanged. It never converts DeadlineExceeded into a return value,
    never reports REPAIR_INFEASIBLE, and never claims exhausted/completed/
    local_optimum on an aborted call.

    diagnostics: if a dict is passed, this function fills it in-place with
    the runtime diagnostics contract (selected_abnormal_count, pool_
    abnormal_count, selected/pool_distinct_signature_count, removed/
    added_pair_signature_count_R/A, phase1_transition_count, phase1_
    transitions_evaluated, tied_best_signature_transition_count_W,
    concrete_candidates_evaluated_phase2, theoretical_concrete_candidates_
    in_winning_ties, elapsed_seconds, deadline_enabled, status,
    aborted_by_deadline). Purely observational: nothing written to
    `diagnostics` ever feeds back into which candidate is accepted. Left
    unfilled (None) has zero behavioral effect.

    R10 (measurement instrumentation only, NON_PROMOTING_PERFORMANCE_
    DIAGNOSTIC_ONLY): hotpath_profile, if a dict is passed, enables THREE
    additional, purely-observational, purely-additive behaviors, all gated
    behind `hotpath_profile is not None` (every existing caller passes
    None, so this is a strict no-op for the official run, --reconstruct-
    check, and --benchmark-two-for-two):
      1. an EARLY fail-closed invocation-identity check (taxonomy
         PROFILE_INVOCATION_IDENTITY_MISMATCH), comparing this call's own
         freshly-computed selected/pool counts, distinct-signature counts,
         R, A, and phase1_transition_count -- plus
         hotpath_profile["one_for_one_move_count_before_two_for_two"], set
         by the caller before invoking -- against
         hotpath_profile["expected_identity"] (if present). This runs
         BEFORE the R9 precompute block and BEFORE the Layer-B
         microbenchmark below, so a mismatched real-data invocation is
         never silently profiled as if it were the locked reference
         invocation;
      2. coarse (bounded-phase, never per-transition) wall-clock timers
         around the equivalence-class construction, removed-pair
         construction, added-pair construction, and R9 aggregate-
         precomputation steps, written into
         hotpath_profile["phase_timing"];
      3. ONE bounded, deterministic-prefix, diagnostic-only "Layer B"
         microbenchmark (see _run_hotpath_microbenchmark), run once
         immediately after the R9 precompute block and BEFORE the real
         Phase-1 streaming search begins (so it never competes with the
         real search for the caller's `deadline` budget), written into
         hotpath_profile["microbenchmark"]. This never returns, promotes,
         or uses a scientific move; it only measures.
    None of the above changes which candidate this function accepts, the
    traversal order, or any return value -- see
    tests/test_phase2F_labeled_unlabeled_guardrails.py's
    TestTwoForTwoHotpathProfiler for the regression suite proving this."""
    # R8 / NHIEM VU 2 (fixes production defect discovered via real benchmark
    # run): defense-in-depth caller-contract check. `selected` (removable)
    # and `pool` (addable) must be disjoint -- an image must never be
    # simultaneously a removed-side and added-side candidate for the same
    # move. `locked` must also be disjoint from both, since by construction
    # (new_region_pool = universe - locked, and pool is always meant to be
    # new_region_pool - selected) neither selected nor the addable pool may
    # ever contain a locked position. This is fail-closed, not
    # auto-corrected: silently subtracting the overlap here would mask a
    # caller defect identical in kind to the one this round fixed at the
    # two production call sites (repair_min_class_coverage,
    # repair_objective_local_search) -- see NHIEM VU 1. If a caller ever
    # violates this contract again, it must be surfaced loudly, not patched
    # over invisibly inside the engine.
    overlap_selected_pool = selected & pool
    if overlap_selected_pool:
        fail(
            f"_equivalence_class_two_for_two_search caller contract violation: "
            f"selected and pool are not disjoint (overlap size "
            f"{len(overlap_selected_pool)}); pool must be exactly the addable "
            f"complement (new_region_pool - selected), never the raw region "
            f"pool. An image must never be both a removable and an addable "
            f"candidate for the same move.",
            "PROTOCOL_VIOLATION",
        )
    overlap_locked_selected = locked & selected
    if overlap_locked_selected:
        fail(
            f"_equivalence_class_two_for_two_search caller contract violation: "
            f"locked and selected are not disjoint (overlap size "
            f"{len(overlap_locked_selected)}); a locked position must never "
            f"also be a removable (selected) candidate.",
            "PROTOCOL_VIOLATION",
        )
    overlap_locked_pool = locked & pool
    if overlap_locked_pool:
        fail(
            f"_equivalence_class_two_for_two_search caller contract violation: "
            f"locked and pool are not disjoint (overlap size "
            f"{len(overlap_locked_pool)}); a locked position must never also "
            f"be an addable (pool) candidate.",
            "PROTOCOL_VIOLATION",
        )
    start_time = time.monotonic() if diagnostics is not None else None
    watchdog_counter = [0]
    phase1_evaluated = [0]
    phase2_evaluated = [0]
    # R6 / NHIEM VU 2: last_w_count/last_theoretical use None (not 0) as
    # their "not yet known" sentinel -- a real Phase-1 pass has to complete
    # before either has an actual measured value; see the except block.
    last_w_count: int | None = None
    last_theoretical: int | None = None

    def _tick_watchdog() -> None:
        watchdog_counter[0] += 1
        if watchdog_counter[0] % TWO_FOR_TWO_DEADLINE_WATCHDOG_INTERVAL == 0:
            _check_deadline(deadline)

    def _always_known_snapshot(status: str) -> dict[str, Any]:
        # Fields that are ALWAYS computable regardless of when (if ever)
        # abort happens -- pure function of the counters/clock above, never
        # "unknown".
        return {
            "status": status,
            "aborted_by_deadline": status == "aborted",
            "deadline_enabled": deadline is not None,
            "elapsed_seconds": (time.monotonic() - start_time) if start_time is not None else None,
            "phase1_transitions_evaluated": phase1_evaluated[0],
            "concrete_candidates_evaluated_phase2": phase2_evaluated[0],
        }

    def _finalize(status: str, w_count: int, theoretical_in_ties: int) -> None:
        # R10 (measurement instrumentation only): expose the raw watchdog
        # tick count for the profiler's Layer-A coarse-timing report. Only
        # ever written when hotpath_profile is not None -- never touches
        # `diagnostics` (the R7-R9 benchmark/production diagnostics
        # contract), so this has zero effect on any existing caller/test.
        if hotpath_profile is not None:
            hotpath_profile["watchdog_ticks_total"] = watchdog_counter[0]
        if diagnostics is None:
            return
        diagnostics.update(_always_known_snapshot(status))
        diagnostics["tied_best_signature_transition_count_W"] = w_count
        diagnostics["theoretical_concrete_candidates_in_winning_ties"] = theoretical_in_ties

    try:
        # R6 boundary check #1: catch an already-expired deadline before
        # any enumeration work at all (fixes blocker R5-B3 for
        # neighborhoods smaller than the watchdog interval).
        _check_deadline(deadline)

        selected_abnormal = [p for p in selected if zero_gt[p] == 0]
        pool_abnormal = [p for p in pool if zero_gt[p] == 0]
        if diagnostics is not None:
            diagnostics.update({
                "selected_abnormal_count": len(selected_abnormal),
                "pool_abnormal_count": len(pool_abnormal),
                "deadline_enabled": deadline is not None,
            })
        if len(selected_abnormal) < 2 or len(pool_abnormal) < 2:
            if diagnostics is not None:
                diagnostics.update({
                    "selected_distinct_signature_count": 0, "pool_distinct_signature_count": 0,
                    "removed_pair_signature_count_R": 0, "added_pair_signature_count_A": 0,
                    "phase1_transition_count": 0,
                })
            # R6 boundary check #2: before the trivial "exhausted" return --
            # these zeros above are REAL measurements (the neighborhood is
            # provably empty), not placeholders, so it is correct to keep
            # them even if this exact check is what raises.
            _check_deadline(deadline)
            _finalize("exhausted", 0, 0)
            return None
        # R10: coarse (bounded-phase, not per-transition) timer -- see
        # hotpath_profile docstring paragraph above. A no-op dict write
        # when hotpath_profile is None.
        _hp_t0 = time.monotonic() if hotpath_profile is not None else None
        sel_groups = _abnormal_equivalence_classes(selected_abnormal, labels14)
        pool_groups = _abnormal_equivalence_classes(pool_abnormal, labels14)
        if hotpath_profile is not None:
            hotpath_profile.setdefault("phase_timing", {})["equivalence_class_construction_seconds"] = (
                time.monotonic() - _hp_t0
            )

        base_state = sorted(locked | selected)
        base_k = labels14[base_state].sum(axis=0).astype(np.int64) if base_state else np.zeros(14, dtype=np.int64)
        base_n = len(base_state)
        base_s_l = int(base_k.sum())

        sel_sigs = list(sel_groups.keys())
        pool_sigs = list(pool_groups.keys())
        _hp_t0 = time.monotonic() if hotpath_profile is not None else None
        removed_pairs = [
            (sel_sigs[i], sel_sigs[j])
            for i in range(len(sel_sigs)) for j in range(i, len(sel_sigs))
            if sel_sigs[i] != sel_sigs[j] or len(sel_groups[sel_sigs[i]]) >= 2
        ]
        if hotpath_profile is not None:
            hotpath_profile.setdefault("phase_timing", {})["removed_pair_construction_seconds"] = (
                time.monotonic() - _hp_t0
            )
        _hp_t0 = time.monotonic() if hotpath_profile is not None else None
        added_pairs = [
            (pool_sigs[i], pool_sigs[j])
            for i in range(len(pool_sigs)) for j in range(i, len(pool_sigs))
            if pool_sigs[i] != pool_sigs[j] or len(pool_groups[pool_sigs[i]]) >= 2
        ]
        if hotpath_profile is not None:
            hotpath_profile.setdefault("phase_timing", {})["added_pair_construction_seconds"] = (
                time.monotonic() - _hp_t0
            )

        if diagnostics is not None:
            diagnostics.update({
                "selected_distinct_signature_count": len(sel_sigs),
                "pool_distinct_signature_count": len(pool_sigs),
                "removed_pair_signature_count_R": len(removed_pairs),
                "added_pair_signature_count_A": len(added_pairs),
                "phase1_transition_count": len(removed_pairs) * len(added_pairs),
            })

        # R10 / NHIEM VU 2 (measurement instrumentation only): fail-closed
        # invocation-identity cross-check, BEFORE the R9 precompute block
        # and BEFORE the Layer-B microbenchmark -- so a real-data
        # invocation that does not match the locked reference evidence
        # (e.g. instances_train.json drifted, or a different budget/phase
        # reached the engine first) is never silently profiled as if it
        # were the reference invocation. Only ever active when the caller
        # (run_two_for_two_hotpath_profile, via repair_objective_local_
        # search) passed hotpath_profile with an "expected_identity" key;
        # every existing caller passes hotpath_profile=None, so this is a
        # complete no-op for the official run, --reconstruct-check, and
        # --benchmark-two-for-two.
        if hotpath_profile is not None and hotpath_profile.get("expected_identity") is not None:
            expected = hotpath_profile["expected_identity"]
            observed = {
                "selected_abnormal_count": len(selected_abnormal),
                "pool_abnormal_count": len(pool_abnormal),
                "selected_distinct_signature_count": len(sel_sigs),
                "pool_distinct_signature_count": len(pool_sigs),
                "removed_pair_signature_count_R": len(removed_pairs),
                "added_pair_signature_count_A": len(added_pairs),
                "phase1_transition_count": len(removed_pairs) * len(added_pairs),
                "one_for_one_move_count_before_two_for_two": hotpath_profile.get(
                    "one_for_one_move_count_before_two_for_two"
                ),
            }
            hotpath_profile["observed_identity"] = observed
            mismatches = {
                field: (expected[field], observed[field])
                for field in expected
                if field in observed and observed[field] != expected[field]
            }
            if mismatches:
                fail(
                    "Real 1pct objective-repair invocation identity does not match the locked "
                    f"reference evidence (see two_for_two_hotpath_profiler.expected_invocation_identity "
                    f"in the protocol config): {mismatches!r}. Refusing to profile a different "
                    "invocation state than the one this round's evidence and claims are about.",
                    "PROFILE_INVOCATION_IDENTITY_MISMATCH",
                )

        # R9 (static-audit-driven performance fix, NHIEM VU 1-2): precompute
        # each ADDED signature pair's aggregate vector/sum EXACTLY ONCE
        # here, before the R x A loop -- the R8 static audit confirmed
        # added_vec depends ONLY on (a_ca, a_cb), never on the removed
        # pair, so _stream_best_prefix_above previously reconstructing it
        # from scratch on every (removed_pair, added_pair) combination
        # (R times per added pair, i.e. R*A total _signature_vector calls
        # for the added side alone) was structurally certain redundant
        # work. This precompute block lives in THIS enclosing scope (not
        # inside _stream_best_prefix_above) so it runs exactly ONCE per
        # engine call, even though _stream_best_prefix_above itself may be
        # re-invoked multiple times (once per exhausted tied group) via the
        # `while True` resumption loop below -- every resumption reuses the
        # same precomputed matrix/sum-array, never rebuilds it.
        #
        # Memory-safe representation: two small CONTIGUOUS ndarrays, never
        # a Python list/dict holding hundreds of thousands of individual
        # ndarray objects. pool_sig_vec_matrix has one row per DISTINCT
        # pool signature (len(pool_sigs) = p, bounded by
        # pool_distinct_signature_count -- 822 in the real R8 1pct
        # evidence, nowhere near A's scale), so _signature_vector is called
        # exactly p times here, never A times and never R*A times.
        # added_vec_matrix is a single (A, 14) int64 ndarray -- exact
        # dtype match with _signature_vector's own np.int64, so no dtype
        # narrowing and no arithmetic-semantics change; values are sums of
        # two 0/1 indicator vectors (range 0..2 per class), far below
        # int64's range, so there is no overflow risk. Memory bound:
        # pool_sig_vec_matrix is p*14*8 bytes (negligible, low hundreds of
        # KB at real scale); added_vec_matrix is A*14*8 bytes (~37.8 MB at
        # A=337,779, matching the audit's estimate exactly since it is one
        # contiguous buffer, not per-row objects); added_sum_array is an
        # additional A*8 bytes (~2.7 MB). No other new per-pair Python
        # object is created or retained.
        _hp_t0 = time.monotonic() if hotpath_profile is not None else None
        pool_sig_index = {sig: i for i, sig in enumerate(pool_sigs)}
        pool_sig_vec_matrix = np.empty((len(pool_sigs), 14), dtype=np.int64)
        for _sig_row, _sig in enumerate(pool_sigs):
            pool_sig_vec_matrix[_sig_row] = _signature_vector(_sig)
            _tick_watchdog()  # R9 / NHIEM VU 4: bounded periodic deadline coverage; never touches phase1_evaluated.
        added_vec_matrix = np.empty((len(added_pairs), 14), dtype=np.int64)
        for _pair_row, (_a_ca, _a_cb) in enumerate(added_pairs):
            added_vec_matrix[_pair_row] = pool_sig_vec_matrix[pool_sig_index[_a_ca]] + pool_sig_vec_matrix[pool_sig_index[_a_cb]]
            # R9 / NHIEM VU 4: this O(A) precompute loop must be covered by
            # the SAME bounded periodic deadline check as the R x A loop
            # below, at the SAME configured interval
            # (TWO_FOR_TWO_DEADLINE_WATCHDOG_INTERVAL, unchanged) --
            # reusing _tick_watchdog() exactly as-is, never a new/changed
            # mechanism. This deliberately does NOT increment
            # phase1_evaluated: phase1_transitions_evaluated must count
            # only actual logical (removed_pair, added_pair) transitions
            # evaluated in the loop below, never this setup step. If
            # DeadlineExceeded fires here, it propagates to the same
            # `except DeadlineExceeded:` block below exactly like a
            # Phase-1/Phase-2 abort; phase1_evaluated[0] is correctly still
            # whatever it was before this precompute call (0 on the first
            # call), so no logical transition is ever falsely claimed
            # evaluated, and no COMPLETED/EXHAUSTED status can ever be
            # reached without the exception propagating first.
            _tick_watchdog()
        # One vectorized reduction over the whole (A, 14) buffer -- not A
        # separate .sum() calls -- mirrors the exact same summation
        # _signature_vector(a_ca).sum() + _signature_vector(a_cb).sum()
        # would have produced, just computed once, in bulk.
        added_sum_array = added_vec_matrix.sum(axis=1)
        if hotpath_profile is not None:
            hotpath_profile.setdefault("phase_timing", {})["aggregate_precompute_seconds"] = (
                time.monotonic() - _hp_t0
            )

        # R10 / NHIEM VU 3: ONE bounded, deterministic-prefix, diagnostic-
        # only Layer-B microbenchmark, run here -- AFTER the R9 precompute
        # (so it can reuse pool_sig_vec_matrix/added_vec_matrix/
        # added_sum_array/base_k/base_n/base_s_l/accept_and_rank exactly as
        # the real search below will) and BEFORE the real Phase-1 streaming
        # search begins (so it never competes with the real search for
        # `deadline`'s remaining budget, and never depends on how much of
        # that budget is left). Only ever runs when hotpath_profile is not
        # None -- a strict no-op for every existing caller.
        if hotpath_profile is not None:
            hotpath_profile["microbenchmark"] = _run_hotpath_microbenchmark(
                removed_pairs, added_pairs, added_vec_matrix, added_sum_array,
                base_k, base_n, base_s_l, K, S_T, N, accept_and_rank,
                hotpath_profile.get("profile_transition_prefix_count", 2000),
            )

        def _stream_best_prefix_above(exclude_prefix_leq: tuple[Any, ...] | None):
            """PROOF 3 (R5) Phase 1: streaming best-prefix scan. Never
            materializes or sorts a full candidate list -- see PROOF 3
            above for the equivalence argument. If exclude_prefix_leq is
            not None, any candidate whose prefix is <= it is skipped (used
            to resume the search past an already-exhausted-by-`visited`
            group, preserving the pre-R5 "fall through to the next-best
            group" behavior without ever holding more than one group's
            transitions in memory).

            R9: added-pair vectors/sums are read from the precomputed
            added_vec_matrix/added_sum_array (built once in the enclosing
            scope, above) via a plain index lookup -- never reconstructed
            here. Traversal order is unchanged: the outer loop is still
            `for r_ca, r_cb in removed_pairs`, the inner loop still visits
            every element of `added_pairs` in the exact same order
            (enumerate() does not reorder); the index is used only to read
            the matching precomputed row, not to change which pair is
            visited or when."""
            best_prefix: tuple[Any, ...] | None = None
            tied: list[tuple[tuple, tuple]] = []
            for r_ca, r_cb in removed_pairs:
                removed_vec = _signature_vector(r_ca) + _signature_vector(r_cb)
                removed_sum = int(removed_vec.sum())  # R9: hoisted out of the A loop -- invariant per removed pair.
                for _added_idx, (a_ca, a_cb) in enumerate(added_pairs):
                    phase1_evaluated[0] += 1
                    _tick_watchdog()
                    added_vec = added_vec_matrix[_added_idx]           # R9: precomputed, no _signature_vector() call.
                    added_sum = int(added_sum_array[_added_idx])       # R9: precomputed.
                    new_k = base_k - removed_vec + added_vec
                    new_s_l = base_s_l - removed_sum + added_sum
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
                    # else: strictly worse than best_prefix seen so far -- DISCARD.
            return best_prefix, tied

        exclude_leq: tuple[Any, ...] | None = None
        while True:
            best_prefix, tied_best_transitions = _stream_best_prefix_above(exclude_leq)
            if best_prefix is None:
                # R6 boundary check #3: before the "exhausted" return.
                _check_deadline(deadline)
                _finalize("exhausted", last_w_count or 0, last_theoretical or 0)
                return None
            last_w_count = len(tied_best_transitions)

            # --- Phase 2: concrete SHA-256 tie-break, restricted to THIS
            # tied-best group only (see PROOF 2, answers to questions 2/3). ---
            best_concrete: tuple[Any, ...] | None = None
            best_move: tuple[int, int, int, int] | None = None
            theoretical_this_group = 0
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
                theoretical_this_group += len(out_pairs) * len(in_pairs)

                for out1, out2 in out_pairs:
                    lo, hi = (out1, out2) if out1 < out2 else (out2, out1)
                    for in1, in2 in in_pairs:
                        phase2_evaluated[0] += 1
                        _tick_watchdog()
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
                        priority = move_priority_digest(seed, budget, repair_phase, "two_for_two_swap", removed_ids, added_ids)
                        identity = canonical_numeric_move_identity(removed_ids, added_ids)
                        concrete_key = (priority, identity)
                        if best_concrete is None or concrete_key < best_concrete:
                            best_concrete = concrete_key
                            best_move = (lo, hi, li, hj)

            last_theoretical = theoretical_this_group
            if best_move is not None:
                full_key = tuple(best_prefix) + best_concrete
                # R6 boundary check #4: before the "completed" return.
                _check_deadline(deadline)
                _finalize("completed", last_w_count, theoretical_this_group)
                return (best_move[0], best_move[1], best_move[2], best_move[3], full_key)
            # entire tied group exhausted by the visited-membership cycle guard --
            # resume Phase 1, excluding every prefix <= this one, to fall
            # through to the next-best signature-pair group (R5: same
            # semantics as R4's sort-then-group fallback, without ever
            # materializing the full candidate list -- see PROOF 3 above).
            exclude_leq = best_prefix
    except DeadlineExceeded:
        # R6 / NHIEM VU 2 (fixes blocker R5-B2): the invocation that hit
        # resource exhaustion is exactly the one whose evidence matters
        # most -- preserve it rather than losing it. Caught HERE ONLY to
        # finalize diagnostics; the exception is always re-raised unchanged
        # immediately after, never converted into REPAIR_INFEASIBLE, a
        # candidate return, or any exhausted/completed/local_optimum claim.
        if hotpath_profile is not None:
            hotpath_profile["watchdog_ticks_total"] = watchdog_counter[0]
        if diagnostics is not None:
            diagnostics.update(_always_known_snapshot("aborted"))
            # Fields below may or may not have been computed yet at the
            # moment of abort. setdefault leaves an ALREADY-recorded real
            # measurement (including a genuine 0) untouched, and only fills
            # in None for whatever this specific call never got to -- zero
            # must never be fabricated for "not yet measured".
            for field in (
                "selected_abnormal_count", "pool_abnormal_count",
                "selected_distinct_signature_count", "pool_distinct_signature_count",
                "removed_pair_signature_count_R", "added_pair_signature_count_A",
                "phase1_transition_count",
            ):
                diagnostics.setdefault(field, None)
            diagnostics["tied_best_signature_transition_count_W"] = last_w_count
            diagnostics["theoretical_concrete_candidates_in_winning_ties"] = last_theoretical
        raise


def _coverage_move_key(
    trial: set[int], missing_count_after: int, labels14: np.ndarray, K: list[int], S_T: int, N: int,
    seed: int, budget: str, move_type: str, removed_ids: tuple[int, ...], added_ids: tuple[int, ...],
) -> tuple[Any, ...]:
    """Candidate ranking for minimum_class_coverage under
    COVERAGE_PROGRESS_RULE = STRICTLY_REDUCE_MISSING_CLASS_COUNT_BY_AT_LEAST_ONE.
    Priority order (first element wins, ties broken left-to-right):
      1. missing_count_after  -- more coverage progress always wins first,
         even over a worse integer objective;
      2. the integer objective tuple (e_max, e_mean, e_lc);
      3. the seeded namespaced SHA-256 move priority;
      4. the canonical numeric move identity (final, deterministic fallback).
    """
    e_max, e_mean, e_lc = integer_objective(trial, labels14, K, S_T, N)
    priority = move_priority_digest(seed, budget, "minimum_class_coverage", move_type, removed_ids, added_ids)
    identity = canonical_numeric_move_identity(removed_ids, added_ids)
    return (missing_count_after, e_max, e_mean, e_lc, priority, identity)


def repair_min_class_coverage(
    locked: set[int], new_region_selected: set[int], new_region_pool: set[int],
    labels14: np.ndarray, zero_gt: np.ndarray, names: list[str],
    K: list[int], S_T: int, N: int, seed: int, budget: str, image_ids: np.ndarray,
    deadline: float | None, log: list[dict[str, Any]], visited: set[frozenset],
) -> tuple[set[int], str]:
    """R11 ACTIVE: deterministic ONE-FOR-ONE coverage repair only.

    Enumerates the COMPLETE admissible one-for-one swap neighborhood for the
    current missing-class set (every addable pool image carrying the target
    class, crossed with every removable selected image of the SAME zero_gt
    status), accepts only moves satisfying
    STRICTLY_REDUCE_MISSING_CLASS_COUNT_BY_AT_LEAST_ONE, and preserves exact
    labeled size, exact No-Finding count and the locked nested prefix on
    every accepted move.

    Returns RepairOutcome.OK as soon as all 14 abnormality classes are
    present. If the full admissible one-for-one neighborhood is exhausted
    while classes are still missing, returns RepairOutcome.REPAIR_INFEASIBLE
    -- it NEVER falls back to the legacy two-for-two neighborhood, never
    widens the neighborhood some other way, and never searches another seed
    (seed_search_forbidden applies unconditionally). Callers must treat
    REPAIR_INFEASIBLE as terminal and report the exact failing budget/phase;
    this applies independently and identically to 1pct/5pct/10pct/20pct --
    a larger budget is never assumed to succeed because a smaller one did.
    """
    selected = set(new_region_selected)
    target_size = len(selected)
    target_nf = int(sum(zero_gt[i] for i in selected))

    def missing_positions(members: set[int]) -> set[int]:
        positions = sorted(locked | members)
        present = set(int(c) for c in np.where(labels14[positions].sum(axis=0) > 0)[0])
        return set(range(14)) - present

    missing = missing_positions(selected)
    while missing:
        _check_deadline(deadline)
        target_class = min(missing)

        best: tuple[Any, ...] | None = None
        best_pair: tuple[int, int] | None = None
        in_pool = [i for i in (new_region_pool - selected) if labels14[i, target_class] == 1]
        for in_pos in in_pool:
            same_nf_out_pool = [i for i in selected if zero_gt[i] == zero_gt[in_pos]]
            for out_pos in same_nf_out_pool:
                trial = set(selected)
                trial.discard(out_pos)
                trial.add(in_pos)
                if len(trial) != target_size or int(sum(zero_gt[i] for i in trial)) != target_nf:
                    continue
                missing_after = len(missing_positions(trial))
                if missing_after >= len(missing):  # STRICTLY_REDUCE_MISSING_CLASS_COUNT_BY_AT_LEAST_ONE
                    continue
                signature = frozenset(trial)
                if signature in visited:
                    continue
                removed_ids = (int(image_ids[out_pos]),)
                added_ids = (int(image_ids[in_pos]),)
                key = _coverage_move_key(locked | trial, missing_after, labels14, K, S_T, N, seed, budget,
                                          "one_for_one_swap", removed_ids, added_ids)
                if best is None or key < best:
                    best, best_pair = key, (out_pos, in_pos)

        if best is not None and best_pair is not None:
            out_pos, in_pos = best_pair
            _accept_coverage_move(
                selected, locked, [out_pos], [in_pos], "one_for_one_swap", target_class,
                labels14, zero_gt, names, K, S_T, N, seed, budget, image_ids, log, visited, missing,
            )
            missing = missing_positions(selected)
            continue

        # R11: the COMPLETE admissible one-for-one coverage neighborhood has
        # been enumerated above and contains no move that strictly reduces
        # the missing-class count. Under the active protocol
        # (ACTIVE_REPAIR_POLICY) this is terminal: report REPAIR_INFEASIBLE
        # and let the caller stop and name the exact failing budget/phase.
        # This deliberately does NOT fall back to the legacy two-for-two
        # neighborhood (removed from the active protocol -- see the module
        # docstring), does NOT widen the neighborhood by any other means,
        # and does NOT retry with a different seed: seed_search_forbidden
        # applies unconditionally, and a coverage failure must surface as a
        # reported failure, never be papered over by resampling.
        return selected, RepairOutcome.REPAIR_INFEASIBLE
    return selected, RepairOutcome.OK


def _accept_coverage_move(
    selected: set[int], locked: set[int], out_positions: list[int], in_positions: list[int],
    move_type: str, target_class: int, labels14: np.ndarray, zero_gt: np.ndarray, names: list[str],
    K: list[int], S_T: int, N: int, seed: int, budget: str, image_ids: np.ndarray,
    log: list[dict[str, Any]], visited: set[frozenset], missing_before_positions: set[int],
) -> None:
    before_state = locked | selected
    before_sha = canonical_membership_sha256(int(image_ids[p]) for p in before_state)
    before_set_digest = candidate_set_priority_digest(seed, budget, [int(image_ids[p]) for p in before_state])
    nf_before = int(sum(zero_gt[i] for i in selected))
    missing_before_names = [names[c] for c in sorted(missing_before_positions)]
    obj_before = integer_objective(before_state, labels14, K, S_T, N)

    for p in out_positions:
        selected.discard(p)
    for p in in_positions:
        selected.add(p)
    visited.add(frozenset(selected))

    after_state = locked | selected
    after_sha = canonical_membership_sha256(int(image_ids[p]) for p in after_state)
    after_set_digest = candidate_set_priority_digest(seed, budget, [int(image_ids[p]) for p in after_state])
    nf_after = int(sum(zero_gt[i] for i in selected))
    positions = sorted(after_state)
    present = set(int(c) for c in np.where(labels14[positions].sum(axis=0) > 0)[0])
    missing_after_names = [names[c] for c in range(14) if c not in present]
    obj_after = integer_objective(after_state, labels14, K, S_T, N)

    removed_ids = [int(image_ids[p]) for p in out_positions]
    added_ids = [int(image_ids[p]) for p in in_positions]
    priority = (move_priority_digest(seed, budget, "minimum_class_coverage", move_type, removed_ids, added_ids))

    log.append(_log_entry(
        len(log) + 1, budget, "minimum_class_coverage", move_type,
        removed_ids, added_ids, nf_before, nf_after, missing_before_names, missing_after_names,
        len(missing_before_positions), len(missing_after_names), obj_before, obj_after,
        priority, before_sha, after_sha, before_set_digest, after_set_digest,
    ))


def repair_objective_local_search(
    locked: set[int], new_region_selected: set[int], new_region_pool: set[int],
    labels14: np.ndarray, zero_gt: np.ndarray, names: list[str],
    K: list[int], S_T: int, N: int, seed: int, budget: str, image_ids: np.ndarray,
    deadline: float | None, log: list[dict[str, Any]], visited: set[frozenset],
) -> tuple[set[int], str, dict[str, Any]]:
    """R11 ACTIVE: deterministic ONE-FOR-ONE objective repair.

    Exhaustively enumerates the COMPLETE admissible one-for-one swap
    neighborhood on every iteration (every selected position crossed with
    every addable pool position of the SAME zero_gt status) -- no top-k, no
    sampling, no early cut-off, no heuristic pruning. Only STRICTLY
    improving moves are accepted: an equal-objective move is forbidden, and
    every accepted move must preserve exact labeled size, exact No-Finding
    count, full 14/14 class coverage and the locked nested prefix.

    Terminates when the one-for-one neighborhood is exhausted with no
    improving move. That is a ONE-FOR-ONE LOCAL OPTIMUM and a NORMAL,
    SUCCESSFUL termination (RepairOutcome.OK) -- never an error, never
    COMPUTATIONAL_ABORT, never REPAIR_INFEASIBLE, and never a global
    optimum. This function NEVER calls
    _equivalence_class_two_for_two_search: the two-for-two neighborhood is
    not part of the active protocol (see the module docstring), and nothing
    replaces it.

    Returns (selected, outcome, status) where `status` records
    one_for_one_exhausted / local_optimum / local_optimum_neighborhood /
    global_optimum_claimed plus the objective tuple immediately BEFORE and
    AFTER this phase and the number of one-for-one moves this phase
    accepted -- these feed build_final_one_for_one_diagnostics."""
    selected = set(new_region_selected)
    target_size = len(selected)
    target_nf = int(sum(zero_gt[i] for i in selected))
    objective_before_objective_repair = integer_objective(locked | selected, labels14, K, S_T, N)
    one_for_one_accepted_count = 0

    def coverage_ok(members: set[int]) -> bool:
        positions = sorted(locked | members)
        present = set(int(c) for c in np.where(labels14[positions].sum(axis=0) > 0)[0])
        return len(present) == 14

    while True:
        _check_deadline(deadline)
        current_state = locked | selected
        current_obj = integer_objective(current_state, labels14, K, S_T, N)

        # --- one-for-one improving swaps: same zero_gt status in/out ---
        best: tuple[Any, ...] | None = None
        best_pair: tuple[int, int] | None = None
        selected_list = sorted(selected)
        pool_list = sorted(new_region_pool - selected)
        for out_pos in selected_list:
            for in_pos in pool_list:
                if zero_gt[out_pos] != zero_gt[in_pos]:
                    continue
                trial = set(selected)
                trial.discard(out_pos)
                trial.add(in_pos)
                if len(trial) != target_size or int(sum(zero_gt[i] for i in trial)) != target_nf:
                    continue
                if not coverage_ok(trial):
                    continue
                trial_obj = integer_objective(locked | trial, labels14, K, S_T, N)
                if not (trial_obj < current_obj):  # equal-objective move forbidden; must strictly improve
                    continue
                signature = frozenset(trial)
                if signature in visited:
                    continue
                removed_ids = (int(image_ids[out_pos]),)
                added_ids = (int(image_ids[in_pos]),)
                priority = move_priority_digest(seed, budget, "objective_repair", "one_for_one_swap", removed_ids, added_ids)
                identity = canonical_numeric_move_identity(removed_ids, added_ids)
                key = (trial_obj[0], trial_obj[1], trial_obj[2], priority, identity)
                if best is None or key < best:
                    best, best_pair = key, (out_pos, in_pos)

        if best is not None and best_pair is not None:
            out_pos, in_pos = best_pair
            _accept_objective_move(
                selected, locked, [out_pos], [in_pos], "one_for_one_swap",
                labels14, zero_gt, names, K, S_T, N, seed, budget, image_ids, log, visited, current_obj,
            )
            one_for_one_accepted_count += 1
            continue

        # R11: the COMPLETE admissible one-for-one neighborhood was just
        # enumerated and contains no strictly improving move. This is the
        # ONE-FOR-ONE LOCAL OPTIMUM and the phase's normal, successful
        # termination. There is deliberately no second neighborhood to fall
        # through to.
        break

    objective_after_objective_repair = integer_objective(locked | selected, labels14, K, S_T, N)
    exhaustion_status: dict[str, Any] = {
        "one_for_one_exhausted": True,
        "local_optimum": True,
        "local_optimum_neighborhood": LOCAL_OPTIMUM_NEIGHBORHOOD,
        "global_optimum_claimed": False,
        "objective_before_objective_repair": list(objective_before_objective_repair),
        "objective_after_objective_repair": list(objective_after_objective_repair),
        "objective_repair_one_for_one_move_count": one_for_one_accepted_count,
    }
    return selected, RepairOutcome.OK, exhaustion_status


def _accept_objective_move(
    selected: set[int], locked: set[int], out_positions: list[int], in_positions: list[int],
    move_type: str, labels14: np.ndarray, zero_gt: np.ndarray, names: list[str],
    K: list[int], S_T: int, N: int, seed: int, budget: str, image_ids: np.ndarray,
    log: list[dict[str, Any]], visited: set[frozenset], obj_before: tuple[int, int, int],
) -> None:
    before_state = locked | selected
    before_sha = canonical_membership_sha256(int(image_ids[p]) for p in before_state)
    before_set_digest = candidate_set_priority_digest(seed, budget, [int(image_ids[p]) for p in before_state])
    nf_before = int(sum(zero_gt[i] for i in selected))
    missing_before = _missing_class_names(before_state, labels14, names)

    for p in out_positions:
        selected.discard(p)
    for p in in_positions:
        selected.add(p)
    visited.add(frozenset(selected))

    after_state = locked | selected
    after_sha = canonical_membership_sha256(int(image_ids[p]) for p in after_state)
    after_set_digest = candidate_set_priority_digest(seed, budget, [int(image_ids[p]) for p in after_state])
    nf_after = int(sum(zero_gt[i] for i in selected))
    missing_after = _missing_class_names(after_state, labels14, names)
    obj_after = integer_objective(after_state, labels14, K, S_T, N)

    removed_ids = [int(image_ids[p]) for p in out_positions]
    added_ids = [int(image_ids[p]) for p in in_positions]
    priority = move_priority_digest(seed, budget, "objective_repair", move_type, removed_ids, added_ids)

    log.append(_log_entry(
        len(log) + 1, budget, "objective_repair", move_type,
        removed_ids, added_ids, nf_before, nf_after, missing_before, missing_after,
        0, 0, obj_before, obj_after, priority, before_sha, after_sha, before_set_digest, after_set_digest,
    ))


# --------------------------------------------------------------------------- #
# Initial candidate via iterative multilabel stratified sampling
# --------------------------------------------------------------------------- #
def stratified_initial_candidate(
    pool_positions: list[int], labels14: np.ndarray, zero_gt: np.ndarray,
    size: int, seed: int,
) -> set[int]:
    if size <= 0:
        return set()
    if size >= len(pool_positions):
        return set(pool_positions)
    try:
        from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
    except ImportError:
        fail(
            "iterative-stratification (import name 'iterstrat') is not installed. "
            "Install it with: pip install iterative-stratification",
            "DEPENDENCY_MISSING",
        )
    # 15-column matrix (14 class-presence + zero_gt) for STRATIFICATION only.
    matrix15 = np.hstack([labels14[pool_positions], zero_gt[pool_positions].reshape(-1, 1)])
    dummy_x = np.zeros((len(pool_positions), 1), dtype=np.uint8)
    test_size = size / len(pool_positions)
    splitter = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    _, candidate_local = next(splitter.split(dummy_x, matrix15))
    return {pool_positions[i] for i in candidate_local.tolist()}


# --------------------------------------------------------------------------- #
# Budget construction (one budget step; caller drives small-to-large order)
# --------------------------------------------------------------------------- #
def build_budget_step(
    budget: str, locked: set[int], image_ids: np.ndarray, labels14: np.ndarray, zero_gt: np.ndarray,
    names: list[str], target_total_size: int, target_total_nf: int, partition_seed: int,
    K: list[int], S_T: int, N: int, deadline: float | None, repair_log: list[dict[str, Any]],
) -> tuple[set[int], str, dict[str, Any]]:
    """R11 ACTIVE per-budget construction primitive. Runs, in this exact
    order, the four locked repair phases (REPAIR_PHASE_ORDER):

        iterative multilabel stratification (initial candidate)
        -> exact-size repair
        -> exact-No-Finding repair
        -> minimum-class-coverage ONE-FOR-ONE repair
        -> deterministic ONE-FOR-ONE objective repair
        -> stop at a ONE-FOR-ONE LOCAL OPTIMUM

    It NEVER calls _equivalence_class_two_for_two_search, directly or
    indirectly: none of the four phase functions it invokes contains a
    two-for-two call site any more (see repair_min_class_coverage and
    repair_objective_local_search). The R10 profiler's `phase_timing` /
    `hotpath_profile` parameters have been removed along with the profiler
    mode itself.

    On success the returned status dict carries the R11 exhaustion flags and
    the full final one-for-one diagnostics record (see
    build_final_one_for_one_diagnostics)."""
    universe = set(range(len(image_ids)))
    new_region_pool = universe - locked
    target_new_size = target_total_size - len(locked)
    if target_new_size < 0:
        fail(f"Budget {budget}: target size smaller than already-locked prefix", "PROTOCOL_VIOLATION")

    initial_new = stratified_initial_candidate(
        sorted(new_region_pool), labels14, zero_gt, target_new_size, partition_seed
    )
    objective_before_repair = integer_objective(locked | initial_new, labels14, K, S_T, N) if initial_new or locked else (0, 0, 0)

    visited: set[frozenset] = set()
    exhaustion_status: dict[str, Any] = {}
    try:
        selected, outcome = repair_exact_size(
            locked, initial_new, new_region_pool, target_new_size, labels14, zero_gt, names,
            K, S_T, N, partition_seed, budget, image_ids, deadline, repair_log, visited,
        )
        if outcome != RepairOutcome.OK:
            exhaustion_status["failed_phase"] = "exact_size"
            return locked | selected, outcome, exhaustion_status

        target_new_nf = target_total_nf - int(sum(zero_gt[i] for i in locked))
        selected, outcome = repair_exact_no_finding(
            locked, selected, new_region_pool, target_new_nf, zero_gt, labels14, names,
            K, S_T, N, partition_seed, budget, image_ids, deadline, repair_log, visited,
        )
        if outcome != RepairOutcome.OK:
            exhaustion_status["failed_phase"] = "exact_no_finding"
            return locked | selected, outcome, exhaustion_status

        selected, outcome = repair_min_class_coverage(
            locked, selected, new_region_pool, labels14, zero_gt, names,
            K, S_T, N, partition_seed, budget, image_ids, deadline, repair_log, visited,
        )
        if outcome != RepairOutcome.OK:
            # R11: a coverage failure must name the exact failing phase so the
            # caller can report "budget B failed in minimum_class_coverage"
            # rather than an anonymous REPAIR_INFEASIBLE. Never retried with
            # another seed.
            exhaustion_status["failed_phase"] = "minimum_class_coverage"
            exhaustion_status["missing_classes_at_failure"] = _missing_class_names(
                locked | selected, labels14, names,
            )
            return locked | selected, outcome, exhaustion_status

        selected, outcome, exhaustion_status = repair_objective_local_search(
            locked, selected, new_region_pool, labels14, zero_gt, names,
            K, S_T, N, partition_seed, budget, image_ids, deadline, repair_log, visited,
        )
        if outcome != RepairOutcome.OK:
            exhaustion_status["failed_phase"] = "objective_repair"
            return locked | selected, outcome, exhaustion_status
    except DeadlineExceeded:
        # R5 / NHIEM VU 4 (fixes blocker R4-B3/timeout-taxonomy correctness):
        # explicitly report NO exhaustion/local-optimum claim on a resource
        # cutoff -- `locked | initial_new` is an ABORTED, non-repaired
        # candidate state, never a valid repaired membership; callers must
        # treat RepairOutcome.COMPUTATIONAL_ABORT as terminal (see
        # compute_all_budgets, which breaks the budget loop and never
        # promotes on any non-OK outcome) and must never read
        # exhaustion_status as if a search had actually completed.
        return locked | initial_new, RepairOutcome.COMPUTATIONAL_ABORT, {
            "one_for_one_exhausted": False,
            "local_optimum": False,
            "local_optimum_neighborhood": None,
            "global_optimum_claimed": False,
            "aborted_by_deadline": True,
        }

    objective_after_repair = integer_objective(locked | selected, labels14, K, S_T, N)
    intersection = initial_new & selected
    union = initial_new | selected
    jaccard = (len(intersection) / len(union)) if union else 1.0
    exhaustion_status.update({
        "candidate_new_region": sorted(int(image_ids[p]) for p in initial_new),
        "final_new_region": sorted(int(image_ids[p]) for p in selected),
        "candidate_final_intersection_count": len(intersection),
        "candidate_final_union_count": len(union),
        "candidate_final_jaccard": jaccard,
        "objective_before_repair": list(objective_before_repair),
        "objective_after_repair": list(objective_after_repair),
    })
    # R11 / TASK 5: the full final one-for-one local-optimum evidence record,
    # assembled here so EVERY caller of build_budget_step (official
    # materialization, --reconstruct-check, and the non-promoting operational
    # benchmark) gets identical diagnostics without re-deriving anything.
    exhaustion_status["final_one_for_one_diagnostics"] = build_final_one_for_one_diagnostics(
        budget, locked | selected, image_ids, labels14, zero_gt, names,
        K, S_T, N, target_total_size, target_total_nf, partition_seed,
        repair_log, exhaustion_status,
    )
    return locked | selected, RepairOutcome.OK, exhaustion_status


# --------------------------------------------------------------------------- #
# COCO subset builders
# --------------------------------------------------------------------------- #
def subset_labeled_coco(master: dict[str, Any], ids: set[int]) -> dict[str, Any]:
    result = {k: v for k, v in master.items() if k not in {"images", "annotations", "categories"}}
    result["images"] = [im for im in master["images"] if im["id"] in ids]
    result["annotations"] = [a for a in master["annotations"] if a["image_id"] in ids]
    result["categories"] = master["categories"]
    return result


def subset_unlabeled_coco(master: dict[str, Any], ids: set[int], forbidden_fields: set[str]) -> dict[str, Any]:
    result = {k: v for k, v in master.items() if k not in {"images", "annotations", "categories"}}
    images = []
    for im in master["images"]:
        if im["id"] not in ids:
            continue
        stripped = {k: v for k, v in im.items() if k not in forbidden_fields}
        images.append(stripped)
    result["images"] = images
    result["annotations"] = []
    result["categories"] = master["categories"]
    return result


def gt_audit_rows(
    master: dict[str, Any], ids: set[int], budget: str, labels14: np.ndarray,
    zero_gt: np.ndarray, image_id_to_position: dict[int, int], category_names: list[str],
) -> list[dict[str, Any]]:
    ann_by_image: dict[int, list[dict[str, Any]]] = {}
    for a in master["annotations"]:
        ann_by_image.setdefault(a["image_id"], []).append(a)
    rows = []
    for image_id in sorted(ids):
        position = image_id_to_position[image_id]
        present_classes = [category_names[i] for i in range(14) if labels14[position, i] == 1]
        rows.append({
            "budget": budget, "image_id": image_id, "zero_gt": int(zero_gt[position]),
            "annotation_count": len(ann_by_image.get(image_id, [])),
            "present_class_names": "|".join(present_classes),
            "present_class_count": len(present_classes),
        })
    return rows


# --------------------------------------------------------------------------- #
# Core, reusable "compute everything" pipeline — used by BOTH the official
# run and the non-promoting --reconstruct-check mode, so the two paths can
# never silently diverge in logic.
# --------------------------------------------------------------------------- #
def compute_locked_size_targets(config: dict[str, Any], n_train: int, nf_train: int) -> tuple[dict[str, int], dict[str, int]]:
    """Recompute (labeled_size_target, nf_size_target) per budget purely
    from config + (n_train, nf_train), cross-checked against
    locked_reference_targets. Shared by compute_all_budgets AND
    independently_validate_staging so the two can never silently diverge on
    what the "correct" target sizes are."""
    ref_targets = config["locked_reference_targets"]
    labeled_size_target: dict[str, int] = {}
    nf_size_target: dict[str, int] = {}
    for budget in BUDGET_ORDER:
        labeled_size_target[budget] = round_half_up_fraction(Fraction(n_train) * BUDGET_FRACTION[budget])
        expected = ref_targets["labeled_size"][budget]
        if labeled_size_target[budget] != expected:
            fail(
                f"Recomputed labeled size for {budget} ({labeled_size_target[budget]}) disagrees with the "
                f"locked reference target ({expected}). instances_train.json may have drifted from Phase 2E.",
                "PREFLIGHT_MISMATCH",
            )
        nf_prevalence_in_labeled = Fraction(labeled_size_target[budget]) * Fraction(nf_train, n_train)
        nf_size_target[budget] = round_half_up_fraction(nf_prevalence_in_labeled)
        expected_nf = ref_targets["no_finding_size"][budget]
        if nf_size_target[budget] != expected_nf:
            fail(
                f"Recomputed No-Finding target for {budget} ({nf_size_target[budget]}) disagrees with the "
                f"locked reference target ({expected_nf}).",
                "PREFLIGHT_MISMATCH",
            )
    return labeled_size_target, nf_size_target


def compute_all_budgets(config: dict[str, Any], project_root: Path, max_seconds: float | None = None) -> dict[str, Any]:
    """Serves exactly two execution purposes:
    EXECUTION_PURPOSE_OFFICIAL_MATERIALIZATION (main()'s full-run path,
    which promotes via write_all_outputs_and_promote) and
    EXECUTION_PURPOSE_RECONSTRUCT_CHECK (run_reconstruct_check, which never
    promotes). Both share ONE global deadline for the whole run
    (`max_seconds`), never a per-budget one.
    The retired R7--R9 benchmark implementation is not one of this function's
    purposes and is unreachable from the R11 CLI.

    R11: the former `two_for_two_engine.operationally_approved` gate is
    GONE. It existed to withhold permission to run a neighborhood that is no
    longer part of the active protocol, so leaving it in place would have
    blocked official materialization forever for a reason that no longer
    applies. It has NOT been flipped to true -- it has been replaced by a
    protocol-consistency check proving the active repair policy really is
    ONE_FOR_ONE_EXHAUSTIVE_DETERMINISTIC."""
    # R11 -- refuse UNCONDITIONALLY, before any construction begins, if the
    # protocol document and this implementation disagree about which
    # neighborhood the active repair policy searches. This replaces the
    # retired two-for-two operational-approval gate: it is shared by BOTH
    # the official run and --reconstruct-check (both call this function), so
    # neither path can ever construct a partition under a policy the
    # protocol config does not actually declare. It never reads, and never
    # depends on, config["legacy_two_for_two_engine"].
    active_policy = config.get("repair", {}).get("active_repair_policy")
    if active_policy != ACTIVE_REPAIR_POLICY:
        fail(
            f"Active repair policy mismatch: configs/protocol/phase2F_labeled_unlabeled.yaml "
            f"declares repair.active_repair_policy={active_policy!r} but this implementation "
            f"is {ACTIVE_REPAIR_POLICY!r}. Construction refuses to start rather than build a "
            f"partition under an undocumented repair policy. The active protocol's objective "
            f"repair exhaustively enumerates the admissible one-for-one swap neighborhood and "
            f"terminates at a one-for-one local optimum; no global optimum is claimed.",
            "ACTIVE_REPAIR_POLICY_MISMATCH",
        )

    iterstrat_version, iterstrat_status = check_iterative_stratification_version(config)
    if iterstrat_status != "OK":
        dep = config["dependencies"]["iterative_stratification"]
        fail(
            f"iterative-stratification version check failed: status={iterstrat_status}, "
            f"installed={iterstrat_version}, required={dep['required_version']}",
            "DEPENDENCY_VERSION_MISMATCH",
        )

    inputs = config["inputs"]
    train = load_json(project_root / inputs["train_coco"])
    image_ids, labels14, zero_gt, category_names = build_indicators(train)
    n_train = len(image_ids)
    nf_train = int(zero_gt.sum())
    image_id_to_position = {int(v): i for i, v in enumerate(image_ids.tolist())}

    partition_seed = int(config["seed"]["partition_seed"])
    K, S_T, N = full_train_integer_stats(labels14)

    labeled_size_target, nf_size_target = compute_locked_size_targets(config, n_train, nf_train)

    deadline = (time.monotonic() + max_seconds) if max_seconds else None

    locked: set[int] = set()
    repair_log: list[dict[str, Any]] = []
    per_budget_selected: dict[str, set[int]] = {}
    per_budget_outcome: dict[str, str] = {}
    per_budget_exhaustion: dict[str, dict[str, Any]] = {}
    per_budget_elapsed_seconds: dict[str, float] = {}
    construction_started = time.monotonic()
    for budget in BUDGET_ORDER:
        budget_started = time.monotonic()
        try:
            selected, outcome, exhaustion = build_budget_step(
                budget, locked, image_ids, labels14, zero_gt, category_names,
                labeled_size_target[budget], nf_size_target[budget], partition_seed,
                K, S_T, N, deadline, repair_log,
            )
        except Exception:
            # The exception remains authoritative and is re-raised unchanged.
            # Emit the consumed duration because no result bundle exists from
            # which main() could otherwise report the failing budget.
            elapsed = max(0.0, time.monotonic() - budget_started)
            print(f"BUDGET={budget} OUTCOME=EXCEPTION SIZE=UNKNOWN ELAPSED_SECONDS={elapsed}")
            print(f"TOTAL_CONSTRUCTION_ELAPSED_SECONDS="
                  f"{max(0.0, time.monotonic() - construction_started)}")
            raise
        per_budget_elapsed_seconds[budget] = max(0.0, time.monotonic() - budget_started)
        per_budget_selected[budget] = selected
        per_budget_outcome[budget] = outcome
        per_budget_exhaustion[budget] = exhaustion
        if outcome != RepairOutcome.OK:
            break
        locked = selected

    total_construction_elapsed_seconds = max(0.0, time.monotonic() - construction_started)

    return {
        "train": train, "image_ids": image_ids, "labels14": labels14, "zero_gt": zero_gt,
        "category_names": category_names, "image_id_to_position": image_id_to_position,
        "n_train": n_train, "nf_train": nf_train, "partition_seed": partition_seed,
        "K": K, "S_T": S_T, "N": N,
        "labeled_size_target": labeled_size_target, "nf_size_target": nf_size_target,
        "per_budget_selected": per_budget_selected, "per_budget_outcome": per_budget_outcome,
        "per_budget_exhaustion": per_budget_exhaustion, "repair_log": repair_log,
        "per_budget_elapsed_seconds": per_budget_elapsed_seconds,
        "total_construction_elapsed_seconds": total_construction_elapsed_seconds,
        "timing_clock": TIMING_CLOCK, "timing_role": TIMING_ROLE,
        "iterative_stratification_version": iterstrat_version,
        # R11: the active repair policy actually used to build this bundle,
        # carried through into every downstream evidence artifact.
        "active_repair_policy": ACTIVE_REPAIR_POLICY,
    }


# --------------------------------------------------------------------------- #
# Materialization
# --------------------------------------------------------------------------- #
def promote_with_rollback(staging: Path, project_root: Path, relative_paths: list[Path]) -> None:
    existing = [project_root / relative for relative in relative_paths if (project_root / relative).exists()]
    if existing:
        fail("Refusing to overwrite official artifact(s): " + ", ".join(map(str, existing)), "REFUSE_OVERWRITE")
    promoted: list[Path] = []
    try:
        for relative in relative_paths:
            source = staging / relative
            target = project_root / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            os.replace(source, target)
            promoted.append(target)
    except Exception:
        for target in reversed(promoted):
            if target.exists():
                target.unlink()
        raise


def official_relative_paths(config: dict[str, Any]) -> list[Path]:
    outputs = config["outputs"]
    paths = [Path(outputs["labeled_coco"][b]) for b in BUDGET_ORDER]
    paths += [Path(outputs["unlabeled_coco"][b]) for b in BUDGET_ORDER]
    paths += [
        Path(outputs["unlabeled_gt_audit_csv"]), Path(outputs["partition_manifest_csv"]),
        Path(outputs["lock_manifest_json"]), Path(outputs["nested_split_check_json"]),
        Path(outputs["leakage_check_json"]), Path(outputs["seed_manifest_json"]),
        Path(outputs["validation_report_json"]), Path(outputs["log_json"]),
        Path(outputs["class_distribution_csv"]), Path(outputs["negative_distribution_csv"]),
        Path(outputs["repair_log_jsonl"]), Path(outputs["errors_csv"]),
    ]
    return paths


def _top_level_payload(coco: dict[str, Any]) -> dict[str, Any]:
    """The COCO dict's top-level keys OTHER than images/annotations/
    categories (e.g. "info", "licenses"), which subset_labeled_coco and
    subset_unlabeled_coco both preserve verbatim from the master file."""
    return {k: v for k, v in coco.items() if k not in {"images", "annotations", "categories"}}


# --------------------------------------------------------------------------- #
# R5 / NHIEM VU 1 (fixes blocker R4-B1) -- SAFE independent-readback         #
# structural validation. Every function in this block is NON-RAISING by      #
# construction: independently_validate_staging must be able to run          #
# build_indicators()-equivalent semantic reconstruction on STAGED (possibly #
# corrupted, externally-produced) data WITHOUT ever crashing on it -- a      #
# corrupted staged artifact must turn a readback gate to False and the      #
# overall status to FAIL, never raise an uncaught Phase2FError. Pipeline:   #
#   structural validation (this block, never raises)                        #
#   -> safe semantic reconstruction (_safe_rebuild_staged_indicators)       #
#   -> semantic hard gates (only attempted if structure is safe)            #
#   -> aggregate PASS/FAIL.                                                  #
# This is distinct from "missing official input/config or an unexpected     #
# internal error", which independently_validate_staging may still let       #
# raise Phase2FError as before (e.g. the staged lock manifest file itself   #
# missing entirely is a READBACK_PRECONDITION_FAILED raise, not a gate).    #
# --------------------------------------------------------------------------- #
def _is_canonical_int_safe(value: Any) -> bool:
    """Non-raising boolean form of _validate_canonical_int."""
    try:
        _validate_canonical_int(value)
        return True
    except Phase2FError:
        return False


def _safe_canonical_int_list(records: list[Any], id_field: str) -> list[int] | None:
    """Returns [record[id_field] as int, ...] IF every element of `records`
    is a dict containing a canonical-integer id_field, else None. Never
    raises regardless of what `records` contains (wrong type elements,
    missing keys, non-integer ids, etc.)."""
    ids: list[int] = []
    for r in records:
        if not isinstance(r, dict) or id_field not in r or not _is_canonical_int_safe(r[id_field]):
            return None
        ids.append(int(_validate_canonical_int(r[id_field])))
    return ids


# The subset of _safe_validate_staged_coco_structure's checks that are
# CRASH-RISK prerequisites for calling build_indicators()-equivalent
# semantic reconstruction / for trusting the returned id sets. R6 (fixes
# blocker R5-B1): image_ids_within_universe is now INCLUDED here -- a
# staged image ID outside the locked Train universe is not just a semantic
# mismatch, it is a crash risk for ANY downstream code that indexes a
# Train-keyed mapping (e.g. locked_image_id_to_position[image_id]) by a
# staged id without first checking membership, such as the nested
# No-Finding computation in independently_validate_staging. Checks NOT
# listed here (annotations_empty, no_forbidden_fields, ...) never risk a
# crash on their own -- they only require the relevant container to
# already be a list, which IS one of these prerequisites -- and are
# therefore always safely computed and reported as their own gate
# regardless of "ok".
_STRUCTURAL_CRASH_RISK_CHECK_NAMES = (
    "images_is_list", "annotations_is_list", "categories_is_list",
    "image_ids_canonical_int", "image_ids_no_duplicate", "image_ids_within_universe",
    "annotation_ids_canonical_int", "annotation_ids_no_duplicate",
    "category_ids_canonical_int", "category_ids_no_duplicate",
    "annotation_image_id_references_valid", "annotation_category_id_references_valid",
)


def _safe_validate_staged_coco_structure(
    coco: Any, universe_ids: set[int], role: str, forbidden_fields: set[str] | None = None,
) -> dict[str, Any]:
    """NEVER RAISES. Validates the structural prerequisites of a staged
    COCO dict (role: "labeled" or "unlabeled") BEFORE any semantic
    reconstruction is attempted. Checks: images/annotations/categories
    exist and are lists; image/annotation/category IDs are canonical
    integers with no duplicates; every annotation.image_id references an
    image present in THIS staged dict; every annotation.category_id
    references a category present in THIS staged dict; image IDs are
    within the locked Train universe. For role="unlabeled", additionally:
    annotations == [] and (if forbidden_fields given) no forbidden GT-
    derived field is present on any image record.

    Returns {"checks": {name: bool}, "ok": bool, "image_ids": set[int] |
    None, "annotation_ids": set[int] | None, "category_ids": set[int] |
    None}. "ok" is True iff every _STRUCTURAL_CRASH_RISK_CHECK_NAMES entry
    is True, i.e. iff it is safe to call build_indicators()-equivalent
    reconstruction on `coco` and to trust the returned id sets."""
    checks: dict[str, bool] = {}
    is_dict = isinstance(coco, dict)
    raw_images = coco.get("images") if is_dict else None
    raw_annotations = coco.get("annotations") if is_dict else None
    raw_categories = coco.get("categories") if is_dict else None

    checks["images_is_list"] = isinstance(raw_images, list)
    checks["annotations_is_list"] = isinstance(raw_annotations, list)
    checks["categories_is_list"] = isinstance(raw_categories, list)

    images = raw_images if checks["images_is_list"] else []
    annotations = raw_annotations if checks["annotations_is_list"] else []
    categories = raw_categories if checks["categories_is_list"] else []

    image_ids = _safe_canonical_int_list(images, "id") if checks["images_is_list"] else None
    checks["image_ids_canonical_int"] = image_ids is not None
    checks["image_ids_no_duplicate"] = image_ids is not None and len(image_ids) == len(set(image_ids))
    image_id_set = set(image_ids) if image_ids is not None else set()
    checks["image_ids_within_universe"] = image_ids is not None and image_id_set <= universe_ids

    ann_ids = _safe_canonical_int_list(annotations, "id") if checks["annotations_is_list"] else None
    checks["annotation_ids_canonical_int"] = ann_ids is not None
    checks["annotation_ids_no_duplicate"] = ann_ids is not None and len(ann_ids) == len(set(ann_ids))

    cat_ids = _safe_canonical_int_list(categories, "id") if checks["categories_is_list"] else None
    checks["category_ids_canonical_int"] = cat_ids is not None
    checks["category_ids_no_duplicate"] = cat_ids is not None and len(cat_ids) == len(set(cat_ids))
    cat_id_set = set(cat_ids) if cat_ids is not None else set()

    if image_ids is not None and checks["annotations_is_list"]:
        checks["annotation_image_id_references_valid"] = all(
            isinstance(a, dict) and _is_canonical_int_safe(a.get("image_id"))
            and int(a["image_id"]) in image_id_set
            for a in annotations
        )
    else:
        checks["annotation_image_id_references_valid"] = False

    if cat_ids is not None and checks["annotations_is_list"]:
        checks["annotation_category_id_references_valid"] = all(
            isinstance(a, dict) and _is_canonical_int_safe(a.get("category_id"))
            and int(a["category_id"]) in cat_id_set
            for a in annotations
        )
    else:
        checks["annotation_category_id_references_valid"] = False

    if role == "unlabeled":
        checks["annotations_empty"] = checks["annotations_is_list"] and annotations == []
        if forbidden_fields is not None:
            checks["no_forbidden_fields"] = checks["images_is_list"] and all(
                isinstance(im, dict) and forbidden_fields.isdisjoint(im.keys()) for im in images
            )

    ok = all(checks[name] for name in _STRUCTURAL_CRASH_RISK_CHECK_NAMES)
    return {
        "checks": checks, "ok": ok,
        "image_ids": image_id_set if image_ids is not None else None,
        "annotation_ids": set(ann_ids) if ann_ids is not None else None,
        "category_ids": cat_id_set if cat_ids is not None else None,
    }


def _safe_rebuild_staged_indicators(coco: dict[str, Any]):
    """Only call AFTER _safe_validate_staged_coco_structure(coco, ...)
    reports "ok"=True for this exact dict. Defense-in-depth wrapper around
    build_indicators(): build_indicators() enforces additional invariants
    beyond the structural prerequisites above (e.g. categories must be
    EXACTLY contiguous 1..14), so this still catches Phase2FError rather
    than propagating it -- corrupted-but-structurally-valid staged data
    must still become a readback gate=False, never a crash. Returns None on
    failure, else build_indicators()'s normal (image_ids, labels14,
    zero_gt, names) tuple."""
    try:
        return build_indicators(coco)
    except Phase2FError:
        return None


def independently_validate_staging(staging: Path, project_root: Path, config: dict[str, Any]) -> dict[str, Any]:
    """Recompute every hard gate FROM THE STAGED FILES ON DISK and from a
    FRESH re-read of instances_train.json / instances_val.json /
    instances_test.json -- never from any in-memory Python object still
    held from construction (no `bundle` parameter is accepted). zero-GT and
    class coverage are rebuilt from the STAGED labeled COCO's own
    annotation records via build_indicators(labeled_data). The
    membership/coco checksums are compared against the STAGED
    phase2F_lock_manifest.json (read from `staging`, not any in-memory
    object).

    R4 (fixes blocker R3-B2): beyond annotation-level checks, this function
    now performs FULL SEMANTIC PAYLOAD comparison of every staged image
    record, category record, and top-level field against the exact expected
    subset reconstructed from the checksum-locked instances_train.json --
    so a corruption that only touches retained image metadata (file_name,
    width, height, or any other retained field) while leaving image_id and
    annotations untouched is still caught, even when the staged lock
    manifest's own recorded checksum was generated from that same corrupted
    object (i.e. this does not rely on staged-file/staged-manifest
    self-consistency, which is exactly the blind spot R3-B2 identified).
    Duplicate ids and out-of-universe ids are rejected as their own explicit
    gates, not inferred incidentally from other comparisons. All canonical
    comparisons are order-independent: sorted by canonical numeric id,
    serialized with sort_keys=True and fixed separators -- displayed JSON
    formatting or on-disk list order is never treated as semantic identity."""
    outputs = config["outputs"]
    inputs = config["inputs"]
    forbidden_fields = set(config["unlabeled_json_forbidden_fields"]["images"])

    locked_train = load_json(project_root / inputs["train_coco"])
    locked_image_ids, _, locked_zero_gt, _ = build_indicators(locked_train)
    locked_image_id_to_position = {int(v): i for i, v in enumerate(locked_image_ids.tolist())}
    train_ids_set = {int(v) for v in locked_image_ids.tolist()}
    n_train = len(locked_image_ids)
    nf_train = int(locked_zero_gt.sum())
    labeled_size_target, nf_size_target = compute_locked_size_targets(config, n_train, nf_train)
    locked_top_level = _top_level_payload(locked_train)
    locked_categories_checksum = _canonical_record_list_sha256(locked_train["categories"], id_field="id")

    val_test_ids: set[int] = set()
    for name in ("val", "test"):
        coco_vt = load_json(project_root / inputs[f"{name}_coco"])
        val_test_ids |= {im["id"] for im in coco_vt["images"]}

    staged_lock_manifest_path = staging / Path(outputs["lock_manifest_json"])
    if not staged_lock_manifest_path.is_file():
        fail("Independent readback requires the staged lock manifest to exist", "READBACK_PRECONDITION_FAILED")
    staged_lock_manifest = load_json(staged_lock_manifest_path)

    gates: dict[str, Any] = {}
    labeled_ids_by_budget: dict[str, set[int]] = {}
    unlabeled_ids_by_budget: dict[str, set[int]] = {}
    # R5 / NHIEM VU 1 (fixes blocker R4-B1): every gate name below the
    # structural checks is a "semantic hard gate" -- attempted ONLY after
    # BOTH labeled_struct["ok"] and unlabeled_struct["ok"] are True (i.e.
    # only once it is SAFE to reconstruct/trust build_indicators() and to
    # compare id sets/payloads). If structure is unsafe for either side,
    # every semantic gate name is explicitly set to False (never left
    # missing, never silently skipped) and this budget's loop iteration
    # moves on -- independently_validate_staging NEVER raises because of
    # corrupted STAGED data; it always produces a full result with
    # status=FAIL.
    _SEMANTIC_GATE_NAMES = (
        "exact_labeled_size", "exact_no_finding", "coverage_14_of_14",
        "disjoint_labeled_unlabeled", "complete_labeled_unlabeled", "val_test_isolation",
        "labeled_annotation_ownership", "membership_checksum_matches",
        "labeled_coco_sha256_matches_staged_lock", "unlabeled_coco_sha256_matches_staged_lock",
        "labeled_image_id_set_matches_locked_subset", "labeled_image_record_count_matches_locked_subset",
        "labeled_image_record_payload_matches_locked_subset",
        "annotation_id_set_matches_locked_subset", "annotation_count_matches_locked_subset",
        "annotation_payload_matches_locked_subset",
        "unlabeled_image_id_set_matches_expected_complement",
        "unlabeled_retained_image_payload_matches_locked_train_after_forbidden_strip",
        "labeled_category_payload_matches_locked_train", "unlabeled_category_payload_matches_locked_train",
        "labeled_retained_top_level_payload_matches_locked_train_subset",
        "unlabeled_retained_top_level_payload_matches_locked_train_subset",
    )

    for budget in BUDGET_ORDER:
        labeled_path = staging / Path(outputs["labeled_coco"][budget])
        unlabeled_path = staging / Path(outputs["unlabeled_coco"][budget])
        labeled_data = load_json(labeled_path)
        unlabeled_data = load_json(unlabeled_path)

        # --- Step 1: STRUCTURAL VALIDATION (never raises). ---
        labeled_struct = _safe_validate_staged_coco_structure(labeled_data, train_ids_set, "labeled")
        unlabeled_struct = _safe_validate_staged_coco_structure(unlabeled_data, train_ids_set, "unlabeled", forbidden_fields)

        # Structural gates are ALWAYS safely computed and reported, whether
        # or not "ok" -- they never depend on build_indicators() or on any
        # other id-trusting reconstruction.
        gates[f"{budget}_labeled_container_types_valid"] = (
            labeled_struct["checks"]["images_is_list"]
            and labeled_struct["checks"]["annotations_is_list"]
            and labeled_struct["checks"]["categories_is_list"]
        )
        gates[f"{budget}_labeled_no_duplicate_or_invalid_image_id"] = (
            labeled_struct["checks"]["image_ids_canonical_int"] and labeled_struct["checks"]["image_ids_no_duplicate"]
        )
        gates[f"{budget}_labeled_no_out_of_universe_image_id"] = labeled_struct["checks"]["image_ids_within_universe"]
        gates[f"{budget}_labeled_no_duplicate_or_invalid_annotation_id"] = (
            labeled_struct["checks"]["annotation_ids_canonical_int"] and labeled_struct["checks"]["annotation_ids_no_duplicate"]
        )
        gates[f"{budget}_labeled_category_ids_valid_and_unique"] = (
            labeled_struct["checks"]["category_ids_canonical_int"] and labeled_struct["checks"]["category_ids_no_duplicate"]
        )
        gates[f"{budget}_labeled_annotation_image_id_references_valid"] = labeled_struct["checks"]["annotation_image_id_references_valid"]
        gates[f"{budget}_labeled_annotation_category_id_references_valid"] = labeled_struct["checks"]["annotation_category_id_references_valid"]

        gates[f"{budget}_unlabeled_container_types_valid"] = (
            unlabeled_struct["checks"]["images_is_list"]
            and unlabeled_struct["checks"]["annotations_is_list"]
            and unlabeled_struct["checks"]["categories_is_list"]
        )
        gates[f"{budget}_unlabeled_no_duplicate_or_invalid_image_id"] = (
            unlabeled_struct["checks"]["image_ids_canonical_int"] and unlabeled_struct["checks"]["image_ids_no_duplicate"]
        )
        gates[f"{budget}_unlabeled_no_out_of_universe_image_id"] = unlabeled_struct["checks"]["image_ids_within_universe"]
        gates[f"{budget}_unlabeled_category_ids_valid_and_unique"] = (
            unlabeled_struct["checks"]["category_ids_canonical_int"] and unlabeled_struct["checks"]["category_ids_no_duplicate"]
        )
        gates[f"{budget}_unlabeled_annotations_empty"] = unlabeled_struct["checks"].get("annotations_empty", False)
        gates[f"{budget}_no_gt_derived_fields"] = unlabeled_struct["checks"].get("no_forbidden_fields", False)

        labeled_ids = labeled_struct["image_ids"] if labeled_struct["image_ids"] is not None else set()
        unlabeled_ids = unlabeled_struct["image_ids"] if unlabeled_struct["image_ids"] is not None else set()
        labeled_ids_by_budget[budget] = labeled_ids
        unlabeled_ids_by_budget[budget] = unlabeled_ids

        if not (labeled_struct["ok"] and unlabeled_struct["ok"]):
            # Structural prerequisite FAILED: do NOT call build_indicators()
            # or any other id-trusting helper on this staged data. Every
            # semantic gate is explicitly False; readback still produces a
            # full result (status computed below) instead of crashing.
            for name in _SEMANTIC_GATE_NAMES:
                gates[f"{budget}_{name}"] = False
            continue

        # --- Step 2: SAFE SEMANTIC RECONSTRUCTION (only now that structure
        # is verified safe). Reconstructs zero-GT / class coverage from the
        # STAGED labeled COCO's OWN annotation records -- fully independent
        # of any in-memory construction state. ---
        staged_indicators = _safe_rebuild_staged_indicators(labeled_data)
        if staged_indicators is None:
            # Structurally valid but build_indicators() still rejected it
            # (e.g. non-contiguous categories) -- same treatment as a
            # structural failure: every semantic gate False, no crash.
            for name in _SEMANTIC_GATE_NAMES:
                gates[f"{budget}_{name}"] = False
            continue
        staged_image_ids, staged_labels14, staged_zero_gt, _ = staged_indicators

        # --- Step 3: SEMANTIC HARD GATES (safe now: structure verified,
        # reconstruction succeeded). ---
        gates[f"{budget}_exact_labeled_size"] = len(labeled_ids) == labeled_size_target[budget]
        gates[f"{budget}_exact_no_finding"] = int(staged_zero_gt.sum()) == nf_size_target[budget]
        present = set(int(c) for c in np.where(staged_labels14.sum(axis=0) > 0)[0]) if len(staged_image_ids) else set()
        gates[f"{budget}_coverage_14_of_14"] = len(present) == 14
        gates[f"{budget}_disjoint_labeled_unlabeled"] = not (labeled_ids & unlabeled_ids)
        gates[f"{budget}_complete_labeled_unlabeled"] = (labeled_ids | unlabeled_ids) == train_ids_set
        gates[f"{budget}_val_test_isolation"] = not (labeled_ids & val_test_ids) and not (unlabeled_ids & val_test_ids)
        gates[f"{budget}_labeled_annotation_ownership"] = all(
            a["image_id"] in labeled_ids for a in labeled_data["annotations"]
        )
        gates[f"{budget}_membership_checksum_matches"] = (
            canonical_membership_sha256(labeled_ids) == staged_lock_manifest["labeled_image_id_sha256"][budget]
        )
        gates[f"{budget}_labeled_coco_sha256_matches_staged_lock"] = (
            sha256_file(labeled_path) == staged_lock_manifest["coco_json_sha256"]["labeled"][budget]
        )
        gates[f"{budget}_unlabeled_coco_sha256_matches_staged_lock"] = (
            sha256_file(unlabeled_path) == staged_lock_manifest["coco_json_sha256"]["unlabeled"][budget]
        )

        # --- R4 / NHIEM VU 2: labeled image-record SET/COUNT/PAYLOAD against
        # the exact expected subset of the locked train file.
        gates[f"{budget}_labeled_image_id_set_matches_locked_subset"] = (
            gates[f"{budget}_labeled_no_duplicate_or_invalid_image_id"]
            and gates[f"{budget}_labeled_no_out_of_universe_image_id"]
        )
        gates[f"{budget}_labeled_image_record_count_matches_locked_subset"] = len(labeled_data["images"]) == len(labeled_ids)
        expected_labeled_images = [im for im in locked_train["images"] if im["id"] in labeled_ids]
        staged_labeled_images_checksum = _safe_canonical_record_list_sha256(labeled_data["images"], "id")
        gates[f"{budget}_labeled_image_record_payload_matches_locked_subset"] = (
            staged_labeled_images_checksum is not None
            and staged_labeled_images_checksum == _canonical_record_list_sha256(expected_labeled_images, "id")
        )

        # --- annotation-ID set/count/PAYLOAD against the locked train file.
        staged_ann_ids = {a["id"] for a in labeled_data["annotations"]}
        expected_ann_ids = {a["id"] for a in locked_train["annotations"] if a["image_id"] in labeled_ids}
        gates[f"{budget}_annotation_id_set_matches_locked_subset"] = staged_ann_ids == expected_ann_ids
        gates[f"{budget}_annotation_count_matches_locked_subset"] = len(labeled_data["annotations"]) == len(expected_ann_ids)
        expected_subset = subset_labeled_coco(locked_train, labeled_ids)
        staged_ann_checksum = _safe_canonical_record_list_sha256(labeled_data["annotations"], "id")
        gates[f"{budget}_annotation_payload_matches_locked_subset"] = (
            staged_ann_checksum is not None
            and staged_ann_checksum == _canonical_record_list_sha256(expected_subset["annotations"], "id")
        )

        # --- R4 / NHIEM VU 2: unlabeled image membership, PAYLOAD (after
        # forbidden-field strip), duplicate/out-of-universe rejection.
        expected_unlabeled_ids = train_ids_set - labeled_ids
        gates[f"{budget}_unlabeled_image_id_set_matches_expected_complement"] = unlabeled_ids == expected_unlabeled_ids
        expected_unlabeled = subset_unlabeled_coco(locked_train, unlabeled_ids, forbidden_fields)
        staged_unlabeled_checksum = _safe_canonical_record_list_sha256(unlabeled_data["images"], "id")
        gates[f"{budget}_unlabeled_retained_image_payload_matches_locked_train_after_forbidden_strip"] = (
            staged_unlabeled_checksum is not None
            and staged_unlabeled_checksum == _canonical_record_list_sha256(expected_unlabeled["images"], "id")
        )

        # --- R4 / NHIEM VU 2: category payload, byte-for-byte against
        # locked train (categories are preserved verbatim, never filtered).
        staged_labeled_cat_checksum = _safe_canonical_record_list_sha256(labeled_data["categories"], "id")
        staged_unlabeled_cat_checksum = _safe_canonical_record_list_sha256(unlabeled_data["categories"], "id")
        gates[f"{budget}_labeled_category_payload_matches_locked_train"] = (
            staged_labeled_cat_checksum is not None and staged_labeled_cat_checksum == locked_categories_checksum
        )
        gates[f"{budget}_unlabeled_category_payload_matches_locked_train"] = (
            staged_unlabeled_cat_checksum is not None and staged_unlabeled_cat_checksum == locked_categories_checksum
        )

        # --- R4 / NHIEM VU 2: retained TOP-LEVEL payload (e.g. info,
        # licenses) against locked train, order-independent via sort_keys.
        gates[f"{budget}_labeled_retained_top_level_payload_matches_locked_train_subset"] = (
            json.dumps(_top_level_payload(labeled_data), sort_keys=True, separators=(",", ":"))
            == json.dumps(locked_top_level, sort_keys=True, separators=(",", ":"))
        )
        gates[f"{budget}_unlabeled_retained_top_level_payload_matches_locked_train_subset"] = (
            json.dumps(_top_level_payload(unlabeled_data), sort_keys=True, separators=(",", ":"))
            == json.dumps(locked_top_level, sort_keys=True, separators=(",", ":"))
        )

    # R6 / NHIEM VU 1 (fixes blocker R5-B1): nested No-Finding computation
    # indexes locked_image_id_to_position (a dict keyed by LOCKED Train
    # image ids only) by staged image ids. Even though image_ids_within_
    # universe is now part of the structural "ok" gate (so a staged file
    # with an out-of-universe id will already have every semantic gate
    # forced False), labeled_ids_by_budget[budget] is still populated from
    # the RAW parsed id set regardless of that budget's own "ok" status (so
    # the loop below can still run over every budget). This is therefore a
    # SEPARATE, defense-in-depth, fail-closed membership check -- never a
    # silent "skip the invalid id, compute over what's left" -- so this
    # code can NEVER KeyError on locked_image_id_to_position, no matter
    # what staged data produced labeled_ids_by_budget.
    for i in range(len(BUDGET_ORDER) - 1):
        smaller, larger = BUDGET_ORDER[i], BUDGET_ORDER[i + 1]
        smaller_ids = labeled_ids_by_budget[smaller]
        larger_ids = labeled_ids_by_budget[larger]
        gates[f"nested_labeled_{smaller}_in_{larger}"] = smaller_ids <= larger_ids

        smaller_ids_valid = smaller_ids <= train_ids_set
        larger_ids_valid = larger_ids <= train_ids_set
        if not (smaller_ids_valid and larger_ids_valid):
            # Fail-closed: at least one staged id is outside the locked
            # Train universe (or the set is otherwise untrustworthy). Never
            # silently drop the offending id(s) and compute over the rest
            # -- that could produce a false PASS. Explicitly FAIL instead.
            gates[f"nested_no_finding_{smaller}_in_{larger}"] = False
            continue
        smaller_nf = {i for i in smaller_ids if locked_zero_gt[locked_image_id_to_position[i]] == 1}
        larger_nf = {i for i in larger_ids if locked_zero_gt[locked_image_id_to_position[i]] == 1}
        gates[f"nested_no_finding_{smaller}_in_{larger}"] = smaller_nf <= larger_nf

    status = "PASS" if all(gates.values()) else "FAIL"
    return {"status": status, "gates": gates}


def compute_repair_summary(repair_log: list[dict[str, Any]], budget: str) -> dict[str, Any]:
    """R11 / TASK 6: report repair moves BY PHASE and BY TYPE.

    The pre-R11 summary collapsed exact-size add/remove toggles,
    exact-No-Finding swaps, coverage swaps and objective-repair swaps into a
    single `one_for_one_move_count`, which made a recorded count like "3"
    impossible to attribute to a phase after the fact (exactly the evidence
    gap the R11 static audit hit on the R9 benchmark report). Those two
    ambiguous fields (`one_for_one_move_count`, `two_for_two_move_count`)
    are REMOVED rather than kept under a misleading name.

    `total_moves` is retained unchanged -- it was always literally "how many
    accepted repair moves this budget recorded", it is correctly named, and
    the promoted lock manifest / --reconstruct-check comparison
    (`repair_move_counts`) depends on exactly that meaning.

    `move_count_by_phase` always contains every key in REPAIR_PHASE_ORDER
    and `move_count_by_type` every key in REPAIR_MOVE_TYPE_ORDER, including
    explicit zeros, so a consumer never has to distinguish "phase absent"
    from "phase ran zero moves". An unexpected phase/type (only reachable
    from a legacy/non-active code path writing into the same log) is
    surfaced under its own key rather than silently dropped."""
    entries = [e for e in repair_log if e["budget"] == budget]
    move_count_by_phase: dict[str, int] = {phase: 0 for phase in REPAIR_PHASE_ORDER}
    move_count_by_type: dict[str, int] = {move_type: 0 for move_type in REPAIR_MOVE_TYPE_ORDER}
    added_ids: set[int] = set()
    removed_ids: set[int] = set()
    for e in entries:
        phase = e["repair_phase"]
        move_type = e["move_type"]
        move_count_by_phase[phase] = move_count_by_phase.get(phase, 0) + 1
        move_count_by_type[move_type] = move_count_by_type.get(move_type, 0) + 1
        added_ids.update(e["added_ids"])
        removed_ids.update(e["removed_ids"])
    return {
        "move_count_by_phase": move_count_by_phase,
        "move_count_by_type": move_count_by_type,
        "unique_ids_added": sorted(added_ids), "unique_ids_removed": sorted(removed_ids),
        "total_moves": len(entries),
    }


def build_final_one_for_one_diagnostics(
    budget: str, member_positions: Iterable[int], image_ids: np.ndarray, labels14: np.ndarray,
    zero_gt: np.ndarray, names: list[str], K: list[int], S_T: int, N: int,
    target_labeled_size: int, target_no_finding_size: int, seed: int,
    repair_log: list[dict[str, Any]], objective_status: dict[str, Any],
) -> dict[str, Any]:
    """R11 / TASK 5: the evidence record describing the FINAL ONE-FOR-ONE
    LOCAL OPTIMUM for one budget.

    Every quantity here is produced by an EXISTING reference function --
    integer_objective, fraction_distribution_report, _missing_class_names,
    canonical_membership_sha256, candidate_set_priority_digest,
    compute_repair_summary -- never by a re-derived competing formula. It is
    purely descriptive: nothing in this record is ever read back to change
    membership, a move, or any selection decision.

    It is emitted for every budget on every successful construction, even
    though the legacy two-for-two neighborhood is no longer invoked -- these
    diagnostics deliberately do not depend on that engine having run."""
    positions = sorted(member_positions)
    member_image_ids = [int(image_ids[p]) for p in positions]

    actual_size = len(positions)
    actual_nf = int(sum(int(zero_gt[p]) for p in positions))
    present_class_indices = sorted(int(c) for c in np.where(labels14[positions].sum(axis=0) > 0)[0])
    missing_class_names = _missing_class_names(positions, labels14, names)

    e_max, e_mean, e_lc = integer_objective(positions, labels14, K, S_T, N)
    d_max, d_mean, d_lc, train_prev, labeled_prev, deviations, worst_idx = fraction_distribution_report(
        positions, labels14, K, N,
    )
    summary = compute_repair_summary(repair_log, budget)

    return {
        "budget": budget,
        # --- exact-size / exact-No-Finding hard constraints ---
        "target_labeled_size": target_labeled_size,
        "actual_labeled_size": actual_size,
        "labeled_size_matches_target": actual_size == target_labeled_size,
        "target_no_finding_size": target_no_finding_size,
        "actual_no_finding_size": actual_nf,
        "no_finding_size_matches_target": actual_nf == target_no_finding_size,
        # --- coverage hard constraint ---
        "class_coverage_count": len(present_class_indices),
        "present_class_ids": [c + 1 for c in present_class_indices],
        "missing_classes": missing_class_names,
        # --- objective, exact integer (selection-path arithmetic) ---
        "integer_objective_before_objective_repair":
            objective_status.get("objective_before_objective_repair"),
        "integer_objective_after_objective_repair":
            objective_status.get("objective_after_objective_repair"),
        "integer_objective_final_E_max_E_mean_E_LC": [e_max, e_mean, e_lc],
        "objective_repair_one_for_one_move_count":
            objective_status.get("objective_repair_one_for_one_move_count"),
        # --- repair accounting, unambiguous per phase and per type ---
        "repair_move_count_total": summary["total_moves"],
        "move_count_by_phase": summary["move_count_by_phase"],
        "move_count_by_type": summary["move_count_by_type"],
        # --- distributional deviation (Fraction, reporting only) ---
        "max_per_class_prevalence_deviation_D_max": str(d_max),
        "mean_absolute_per_class_prevalence_deviation_D_mean": str(d_mean),
        "label_cardinality_deviation_D_LC": str(d_lc),
        "worst_deviation_class_id": worst_idx + 1,
        "worst_deviation_class_name": names[worst_idx],
        "train_reference_prevalence_per_class": {names[c]: str(train_prev[c]) for c in range(14)},
        "labeled_prevalence_per_class": {names[c]: str(labeled_prev[c]) for c in range(14)},
        "absolute_deviation_per_class": {names[c]: str(deviations[c]) for c in range(14)},
        # --- identity ---
        "membership_sha256": canonical_membership_sha256(member_image_ids),
        "candidate_set_priority_digest": candidate_set_priority_digest(seed, budget, member_image_ids),
        # --- termination claim, verbatim ---
        "one_for_one_exhausted": True,
        "local_optimum": True,
        "local_optimum_neighborhood": LOCAL_OPTIMUM_NEIGHBORHOOD,
        "global_optimum_claimed": False,
        "active_repair_policy": ACTIVE_REPAIR_POLICY,
        "termination_claim": (
            "The objective-repair stage exhaustively enumerates the admissible one-for-one "
            "swap neighborhood and terminates at a one-for-one local optimum. No global "
            "optimum is claimed."
        ),
    }


# --------------------------------------------------------------------------- #
# NON_PROMOTING_OPERATIONAL_BENCHMARK: measures empirical operational
# tractability of the ACTIVE repair cascade on the REAL training split,
# without ever promoting or writing official artifacts. Introduced in R7 to
# measure the two-for-two engine; since R11 the active cascade it exercises
# is ONE_FOR_ONE_EXHAUSTIVE_DETERMINISTIC and the legacy engine is never
# reached. See the module-level EXECUTION_PURPOSE_* constants.
# --------------------------------------------------------------------------- #
def _write_benchmark_report_atomic(final_path: Path, report: dict[str, Any]) -> None:
    """Transactional, refuse-overwrite write for the benchmark evidence
    report: write to a temporary sibling file in the SAME directory, flush
    + fsync, independently re-parse the file's OWN bytes back off disk
    (never the in-memory `report` object) to catch a truncated/corrupted
    write, THEN atomically os.replace() it onto `final_path` -- only after
    a second refuse-overwrite check immediately before the replace. On ANY
    failure at any step, the temporary file is removed and never left
    behind; `final_path` is never touched (created, truncated, or
    partially written) until the atomic replace itself succeeds."""
    if final_path.exists():
        fail(f"Benchmark report already exists, refusing to overwrite: {final_path}",
             "BENCHMARK_REPORT_ALREADY_EXISTS")
    final_path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=final_path.name + ".tmp-", dir=str(final_path.parent))
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(report, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        with tmp_path.open("r", encoding="utf-8") as stream:
            reloaded = json.load(stream)
        if reloaded != report:
            fail("Benchmark report readback (independently re-parsed from the temporary file on disk) "
                 "did not match the intended content; refusing to promote a possibly-corrupted write.",
                 "BENCHMARK_REPORT_WRITE_VALIDATION_FAILED")
        if final_path.exists():
            fail(f"Benchmark report already exists, refusing to overwrite: {final_path}",
                 "BENCHMARK_REPORT_ALREADY_EXISTS")
        os.replace(tmp_path, final_path)
    except BaseException:
        if tmp_path.exists():
            tmp_path.unlink()
        raise


# R11: _write_profile_report_atomic (the R10 hot-path profiler's report
# writer, with its PROFILE_REPORT_EXISTS / PROFILE_REPORT_WRITE_VALIDATION_
# FAILED taxonomy) has been removed together with the profiler mode itself.
# The benchmark report writer above is unchanged.


def _benchmark_overall_status(per_budget_report: dict[str, dict[str, Any]]) -> str:
    """COMPUTATIONAL_ABORT and REPAIR_INFEASIBLE are NEVER conflated with
    each other or with a plain unattempted dependency block
    (NOT_RUN_DEPENDENCY): at most one budget can ever be the ORIGINAL
    COMPUTATIONAL_ABORT/REPAIR_INFEASIBLE (construction stops attempting
    further budgets the moment one fails), so checking for either first,
    before requiring unanimous COMPLETED, is unambiguous."""
    statuses = [per_budget_report[b]["status"] for b in BUDGET_ORDER]
    if any(s == "COMPUTATIONAL_ABORT" for s in statuses):
        return "COMPUTATIONAL_ABORT"
    if any(s == "REPAIR_INFEASIBLE" for s in statuses):
        return "REPAIR_INFEASIBLE"
    if all(s == "COMPLETED" for s in statuses):
        return "COMPLETED"
    # Defensive, fail-closed: should be unreachable given the loop logic in
    # run_operational_benchmark (every budget is exactly one of COMPLETED /
    # COMPUTATIONAL_ABORT / REPAIR_INFEASIBLE / NOT_RUN_DEPENDENCY, and a
    # NOT_RUN_DEPENDENCY budget can only follow one of the first three). If
    # this is ever hit, per Section 5's "no half-written report on an
    # uncontrolled failure" rule, the caller must NOT have written a report
    # yet -- see run_operational_benchmark, which calls this BEFORE writing.
    fail(f"Internal inconsistency in per-budget benchmark statuses: {statuses}", "BENCHMARK_INTERNAL_ERROR")
    raise AssertionError("unreachable")  # fail() always raises; keeps type-checkers happy


def run_operational_benchmark(
    config: dict[str, Any], project_root: Path, benchmark_max_seconds_per_budget: float,
) -> dict[str, Any]:
    """LEGACY R7--R9 implementation retained for historical unit tests only.

    R11 parse_args/main cannot call this function.  It must not be used to
    create new evidence because the active build_budget_step no longer invokes
    the two-for-two engine.  Historical R7/R8/R9 reports remain untouched.

    In the retired protocol this NON_PROMOTING_OPERATIONAL_BENCHMARK ran the
    exact production
    per-budget construction primitive (build_budget_step -- the SAME
    function compute_all_budgets calls for the official run, never copied
    or re-implemented) over the real training split, small-to-large,
    preserving the nested-prefix relation exactly like the
    official run, but with each budget getting its OWN fresh monotonic
    deadline instead of one shared global deadline. Never promotes and never
    writes an official artifact.

    R11: this function no longer reads two_for_two_engine.
    operationally_approved for any purpose -- that gate was retired together
    with the two-for-two neighborhood it guarded, and build_budget_step now
    never reaches the legacy engine on any call path, benchmark included.

    R8-B2: before any dependency check, data load, or construction work,
    this function resolves both the current report path and the
    superseded-evidence path and fails closed (BENCHMARK_EVIDENCE_PATH_
    CONFLICT) if they resolve to the SAME file -- lexically different
    config strings (e.g. one going through `..`) can still resolve
    identically, and writing the new report would then silently clobber
    the older evidence it is supposed to supersede, not preserve it. This
    check never requires the superseded file to exist and never opens,
    reads, stats, or deletes it -- only the resolved path STRINGS are
    ever compared."""
    # R8-B2: fail-closed evidence path separation -- must run before the
    # refuse-overwrite check below, before check_iterative_stratification_
    # version, before load_json(train), and before build_budget_step.
    report_path_resolved = (project_root / Path(config["operational_benchmark"]["report_path"])).resolve()
    superseded_path_resolved = (
        project_root / Path(config["operational_benchmark"]["superseded_evidence_path"])
    ).resolve()
    if report_path_resolved == superseded_path_resolved:
        fail(
            "Current benchmark report path resolves to the superseded evidence path",
            "BENCHMARK_EVIDENCE_PATH_CONFLICT",
        )

    report_path = project_root / Path(config["operational_benchmark"]["report_path"])
    if report_path.exists():
        fail(f"Benchmark report already exists, refusing to overwrite: {report_path}",
             "BENCHMARK_REPORT_ALREADY_EXISTS")
    # R8: this run's report path must be a NEW, distinct path from whatever
    # earlier-stage report(s) it supersedes for approval-evidence purposes --
    # this function must NEVER write to, delete, or otherwise touch that
    # older path (read only as a documentation string below, never opened).
    superseded_evidence_path = config["operational_benchmark"]["superseded_evidence_path"]

    iterstrat_version, iterstrat_status = check_iterative_stratification_version(config)
    if iterstrat_status != "OK":
        dep = config["dependencies"]["iterative_stratification"]
        fail(
            f"iterative-stratification version check failed: status={iterstrat_status}, "
            f"installed={iterstrat_version}, required={dep['required_version']}",
            "BENCHMARK_DEPENDENCY_VERSION_MISMATCH",
        )

    inputs = config["inputs"]
    train_path = project_root / inputs["train_coco"]
    train = load_json(train_path)
    image_ids, labels14, zero_gt, category_names = build_indicators(train)
    n_train = len(image_ids)
    nf_train = int(zero_gt.sum())

    partition_seed = int(config["seed"]["partition_seed"])
    K, S_T, N = full_train_integer_stats(labels14)
    labeled_size_target, nf_size_target = compute_locked_size_targets(config, n_train, nf_train)

    print("BENCHMARK_MODE=", EXECUTION_PURPOSE_NON_PROMOTING_OPERATIONAL_BENCHMARK)
    print("BENCHMARK_MAX_SECONDS_PER_BUDGET=", benchmark_max_seconds_per_budget)

    locked: set[int] = set()
    repair_log: list[dict[str, Any]] = []
    per_budget_report: dict[str, dict[str, Any]] = {}
    blocked_by_budget: str | None = None
    blocked_by_status: str | None = None

    for budget in BUDGET_ORDER:
        if blocked_by_budget is not None:
            # A prior budget did not complete: NEVER continue the nested
            # prefix on an incomplete/aborted state and NEVER pretend a
            # later budget was independently benchmarked.
            per_budget_report[budget] = {
                "budget": budget,
                "target_labeled_size": labeled_size_target[budget],
                "target_no_finding_size": nf_size_target[budget],
                "nested_parent_budget": NESTED_PARENT_BUDGET[budget],
                "status": "NOT_RUN_DEPENDENCY",
                "blocked_by_budget": blocked_by_budget,
                "blocked_by_status": blocked_by_status,
                "elapsed_seconds": None,
                "deadline_seconds": benchmark_max_seconds_per_budget,
                "deadline_enabled": True,
                "initial_candidate_size": None,
                "final_selected_size": None,
                "final_no_finding_size": None,
                "final_class_coverage": None,
                "membership_sha256": None,
                "move_count_by_phase": None,
                "move_count_by_type": None,
                "repair_move_count_total": None,
                "objective_before_repair": None,
                "objective_after_repair": None,
                "one_for_one_exhausted": None,
                "local_optimum": None,
                "local_optimum_neighborhood": None,
                "global_optimum_claimed": None,
                "aborted_by_deadline": None,
                "repair_infeasible": None,
                "failed_phase": None,
                "final_one_for_one_diagnostics": None,
            }
            print(f"BENCHMARK_BUDGET={budget} STATUS=NOT_RUN_DEPENDENCY ELAPSED_SECONDS=None")
            continue

        deadline_b = time.monotonic() + benchmark_max_seconds_per_budget  # reset EVERY budget, never shared
        budget_start = time.monotonic()
        selected, outcome, exhaustion = build_budget_step(
            budget, locked, image_ids, labels14, zero_gt, category_names,
            labeled_size_target[budget], nf_size_target[budget], partition_seed,
            K, S_T, N, deadline_b, repair_log,
        )
        elapsed = time.monotonic() - budget_start

        if outcome == RepairOutcome.OK:
            status = "COMPLETED"
            locked = selected  # advance the nested prefix ONLY on success
        elif outcome == RepairOutcome.COMPUTATIONAL_ABORT:
            status = "COMPUTATIONAL_ABORT"
            blocked_by_budget, blocked_by_status = budget, status
        elif outcome == RepairOutcome.REPAIR_INFEASIBLE:
            status = "REPAIR_INFEASIBLE"
            blocked_by_budget, blocked_by_status = budget, status
        else:
            fail(f"Unknown RepairOutcome from build_budget_step: {outcome!r}", "BENCHMARK_INTERNAL_ERROR")

        final_positions = sorted(selected)
        final_size = len(final_positions)
        final_nf = int(sum(int(zero_gt[i]) for i in final_positions)) if final_positions else 0
        present_classes = (
            set(int(c) for c in np.where(labels14[final_positions].sum(axis=0) > 0)[0]) if final_positions else set()
        )
        repair_summary = compute_repair_summary(repair_log, budget)
        candidate_new_region = exhaustion.get("candidate_new_region")

        per_budget_report[budget] = {
            "budget": budget,
            "target_labeled_size": labeled_size_target[budget],
            "target_no_finding_size": nf_size_target[budget],
            "nested_parent_budget": NESTED_PARENT_BUDGET[budget],
            "status": status,
            "blocked_by_budget": None,
            "blocked_by_status": None,
            "elapsed_seconds": elapsed,
            "deadline_seconds": benchmark_max_seconds_per_budget,
            "deadline_enabled": True,
            "initial_candidate_size": len(candidate_new_region) if candidate_new_region is not None else None,
            "final_selected_size": final_size if outcome == RepairOutcome.OK else None,
            "final_no_finding_size": final_nf if outcome == RepairOutcome.OK else None,
            "final_class_coverage": len(present_classes) if outcome == RepairOutcome.OK else None,
            "membership_sha256": (
                canonical_membership_sha256(int(image_ids[p]) for p in final_positions)
                if outcome == RepairOutcome.OK else None
            ),
            "move_count_by_phase": repair_summary["move_count_by_phase"],
            "move_count_by_type": repair_summary["move_count_by_type"],
            "repair_move_count_total": repair_summary["total_moves"],
            "objective_before_repair": exhaustion.get("objective_before_repair"),
            "objective_after_repair": exhaustion.get("objective_after_repair"),
            "one_for_one_exhausted": exhaustion.get("one_for_one_exhausted"),
            "local_optimum": exhaustion.get("local_optimum"),
            "local_optimum_neighborhood": exhaustion.get("local_optimum_neighborhood"),
            "global_optimum_claimed": exhaustion.get("global_optimum_claimed"),
            # Never fabricated as False when unknown: build_budget_step's
            # OK path never sets this key (unambiguously not aborted, so
            # False IS the correct real value); its COMPUTATIONAL_ABORT
            # path always sets it True; a REPAIR_INFEASIBLE outcome returns
            # exhaustion_status == {} (see repair_exact_size/
            # repair_exact_no_finding/repair_min_class_coverage), so False
            # here is also a correct real value (genuinely not aborted).
            "aborted_by_deadline": exhaustion.get("aborted_by_deadline", False),
            "repair_infeasible": outcome == RepairOutcome.REPAIR_INFEASIBLE,
            # R11: names the exact phase that failed (never an anonymous
            # REPAIR_INFEASIBLE), and carries the full final one-for-one
            # local-optimum evidence record when construction succeeded.
            "failed_phase": exhaustion.get("failed_phase"),
            "final_one_for_one_diagnostics": exhaustion.get("final_one_for_one_diagnostics"),
        }
        print(f"BENCHMARK_BUDGET={budget} STATUS={status} ELAPSED_SECONDS={elapsed}")

    overall_status = _benchmark_overall_status(per_budget_report)

    protocol_stage, protocol_version = get_protocol_identity(config)
    report = {
        "phase": "Phase 2F — Labeled/Unlabeled Construction",
        "stage": protocol_stage,
        "protocol_version": protocol_version,
        "mode": EXECUTION_PURPOSE_NON_PROMOTING_OPERATIONAL_BENCHMARK,
        "artifact_role": "NON_PROMOTING_OPERATIONAL_DIAGNOSTIC_ONLY",
        "official_partition_artifact": False,
        "membership_lock": False,
        "usable_for_training": False,
        "operational_approval_decision": False,
        "partition_seed": partition_seed,
        "seed_policy": config["seed"]["seed_policy"],
        "seed_search_performed": False,
        "input_train_coco_sha256": sha256_file(train_path),
        "input_train_image_id_sha256_phase2e_style": phase2e_style_sha256([im["id"] for im in train["images"]]),
        "train_image_count": n_train,
        "train_no_finding_count": nf_train,
        "iterative_stratification_version": iterstrat_version,
        # R11: what this benchmark actually exercised. The legacy two-for-two
        # engine is not part of the active protocol and is never invoked by
        # build_budget_step, so there is no per-invocation engine diagnostics
        # section any more and no operational-approval field to report.
        "active_repair_policy": ACTIVE_REPAIR_POLICY,
        "objective_repair_neighborhood": LOCAL_OPTIMUM_NEIGHBORHOOD,
        "legacy_two_for_two_engine_invoked": False,
        "global_optimum_claimed": False,
        # R8: documents WHY an earlier report exists at a different path
        # without ever calling it invalid -- it is simply not eligible for
        # the operational-approval decision, because it ran against a
        # two-for-two neighborhood implementation that has since been
        # corrected (see the R8 protocol revision note). This report never
        # deletes, overwrites, or otherwise touches that older file.
        "supersedes_for_approval_evidence": superseded_evidence_path,
        "benchmark_exception_scope": "NON_PROMOTING_OPERATIONAL_BENCHMARK_ONLY",
        "deadline_policy": {
            "type": "PER_BUDGET_MONOTONIC_DEADLINE",
            "max_seconds_per_budget": benchmark_max_seconds_per_budget,
            "scientific_move_count_cap": None,
        },
        "budgets": per_budget_report,
        "overall_status": overall_status,
        "training_authorized": False,
        "training_started": False,
        "pseudo_labels_generated": False,
        "test_used": False,
    }

    _write_benchmark_report_atomic(report_path, report)

    print("BENCHMARK_REPORT=", report_path)
    print("BENCHMARK_OVERALL_STATUS=", overall_status)
    return report


# --------------------------------------------------------------------------- #
# R11 -- the R10 NON_PROMOTING_PERFORMANCE_DIAGNOSTIC_ONLY hot-path profiler
# (run_two_for_two_hotpath_profile) has been REMOVED, together with its CLI
# flags (--profile-two-for-two-hotpath / --profile-max-seconds), its argument
# validator, its report writer, its config block
# (two_for_two_hotpath_profiler) and the phase_timing/hotpath_profile
# plumbing it required. It existed solely to measure the two-for-two engine's
# Phase-1 hot loop; that neighborhood is no longer part of the active
# construction protocol, the profiler was never finished, and it never
# produced a report (reports/02F_two_for_two_hotpath_profile_R10.json was
# never written). No profiler report is needed, and nothing in the active
# protocol replaces it. The historical R7/R8/R9 operational-benchmark reports
# are untouched and remain valid historical evidence of their own runs.
# --------------------------------------------------------------------------- #


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=Path("configs/protocol/phase2F_labeled_unlabeled.yaml"))
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--preflight-only", action="store_true",
                         help="Run NHIEM VU 1 static/input validation only. Writes no official artifacts.")
    parser.add_argument("--reconstruct-check", action="store_true",
                         help="Rebuild everything from scratch (non-promoting) and diff the result against the "
                              "already-promoted data/manifests/phase2F_lock_manifest.json. Writes only the "
                              "diagnostic report at reports/02F_deterministic_reconstruction_check.json (not an "
                              "official/gated artifact).")
    parser.add_argument("--max-seconds", type=float, default=None,
                         help="Optional wall-clock budget for the repair search. If exceeded, the run reports "
                              "COMPUTATIONAL_ABORT (never REPAIR_INFEASIBLE). Default: unlimited (no cap).")
    # R11: the R7--R9 --benchmark-two-for-two mode is retired together with
    # the active two-for-two neighborhood.  argparse therefore rejects both
    # historical benchmark flags as unknown arguments.  The legacy benchmark
    # implementation remains below solely to keep the historical reports and
    # their direct unit tests interpretable; main() cannot dispatch to it.
    # R11: --profile-two-for-two-hotpath / --profile-max-seconds (the R10
    # hot-path profiler mode) are REMOVED. argparse rejects them as unknown
    # arguments, so an old invocation fails closed and loudly rather than
    # being silently ignored.
    return parser.parse_args()


def _print_gate_lines() -> None:
    print("training_authorized=false")
    print("training_started=false")
    print("pseudo_labels_generated=false")
    print("test_used=false")


def _validate_benchmark_cli_args(args: argparse.Namespace) -> None:
    """Fail-closed CLI/argument validation for --benchmark-two-for-two.
    Called before any config/data I/O so a malformed invocation is
    rejected instantly and cheaply, never after a long benchmark run has
    already started.

    R7-B2 (GPT static review): --benchmark-max-seconds-per-budget given
    WITHOUT --benchmark-two-for-two is refused, never silently ignored --
    an orphan benchmark deadline must not be dropped on the floor and let
    the run fall through into official/reconstruct-check mode as if the
    researcher had never passed it.

    R7-B3 (GPT static review): --benchmark-two-for-two combined with
    --max-seconds (the single global deadline used by official/
    reconstruct-check) is refused as an ambiguous double deadline -- this
    function is the ONLY place that decides this, so --max-seconds's own
    semantics in the official/reconstruct-check paths (still a single
    deadline for the whole run, completely unchanged) are never touched.
    """
    if not args.benchmark_two_for_two:
        if args.benchmark_max_seconds_per_budget is not None:
            fail(
                "--benchmark-max-seconds-per-budget was given without --benchmark-two-for-two; "
                "refusing to silently ignore an orphan benchmark deadline.",
                "BENCHMARK_ARGUMENT_INVALID",
            )
        return
    if args.preflight_only:
        fail("--benchmark-two-for-two cannot be combined with --preflight-only.", "BENCHMARK_MODE_CONFLICT")
    if args.reconstruct_check:
        fail("--benchmark-two-for-two cannot be combined with --reconstruct-check.", "BENCHMARK_MODE_CONFLICT")
    if args.max_seconds is not None:
        fail(
            "--benchmark-two-for-two cannot be combined with --max-seconds (ambiguous double deadline) -- "
            "use only --benchmark-max-seconds-per-budget.",
            "BENCHMARK_MODE_CONFLICT",
        )
    deadline = args.benchmark_max_seconds_per_budget
    if deadline is None:
        fail(
            "--benchmark-two-for-two requires --benchmark-max-seconds-per-budget "
            "(a per-budget deadline; there is no implicit default).",
            "BENCHMARK_ARGUMENT_INVALID",
        )
    if (
        isinstance(deadline, bool)
        or not isinstance(deadline, (int, float))
        or math.isnan(float(deadline))
        or math.isinf(float(deadline))
        or deadline <= 0
    ):
        fail(
            f"--benchmark-max-seconds-per-budget must be a positive, finite number of seconds; got {deadline!r}.",
            "BENCHMARK_ARGUMENT_INVALID",
        )


def _refuse_inactive_profiler_args(args: argparse.Namespace) -> None:
    """R11 fail-closed guard for the REMOVED R10 hot-path profiler mode.

    `parse_args()` no longer defines --profile-two-for-two-hotpath /
    --profile-max-seconds, so a CLI invocation using them already fails at
    argparse. This guard covers the OTHER entry path: a caller (a test, a
    stale wrapper script) constructing an argparse.Namespace by hand and
    still setting those attributes. Such a caller must be refused loudly
    (PROFILE_MODE_REMOVED), never silently ignored -- silently dropping a
    requested profiling mode would let a researcher believe measurement
    evidence was produced when none was."""
    for removed_attr, removed_flag in (
        ("profile_two_for_two_hotpath", "--profile-two-for-two-hotpath"),
        ("profile_max_seconds", "--profile-max-seconds"),
    ):
        value = getattr(args, removed_attr, None)
        if value:
            fail(
                f"{removed_flag} was removed in Phase 2F R11 together with the two-for-two "
                f"hot-path profiler; the two-for-two neighborhood is no longer part of the "
                f"active construction protocol and there is nothing to profile. Refusing to "
                f"silently ignore an inactive mode.",
                "PROFILE_MODE_REMOVED",
            )


def main() -> None:
    args = parse_args()
    # Defense in depth for stale wrappers/tests that construct Namespace
    # objects directly instead of going through argparse.  A requested legacy
    # benchmark must fail closed rather than silently fall through to official
    # materialization.
    if getattr(args, "benchmark_two_for_two", False) or \
            getattr(args, "benchmark_max_seconds_per_budget", None) is not None:
        fail(
            "--benchmark-two-for-two and --benchmark-max-seconds-per-budget were retired in "
            "Phase 2F R11; the two-for-two engine is not part of the active protocol.",
            "LEGACY_TWO_FOR_TWO_MODE_INACTIVE",
        )
    _refuse_inactive_profiler_args(args)
    project_root = args.project_root.resolve()
    config_path = (project_root / args.config) if not args.config.is_absolute() else args.config
    config = load_config(config_path)
    protocol_stage, protocol_version = get_protocol_identity(config)

    print(
        f"=== PHASE 2F: LABELED/UNLABELED CONSTRUCTION "
        f"(stage {protocol_stage}, protocol_version {protocol_version}) ==="
    )
    print("PYTHON=", platform.python_version())
    print("NUMPY=", np.__version__)
    print("CONFIG=", config_path)
    print("PROJECT_ROOT=", project_root)
    print("PARTITION_SEED=", config["seed"]["partition_seed"])
    print("SEED_POLICY=", config["seed"]["seed_policy"])
    mode = "PREFLIGHT_ONLY" if args.preflight_only else \
        ("RECONSTRUCT_CHECK" if args.reconstruct_check else "FULL_RUN")
    print("MODE=", mode)
    print("ACTIVE_REPAIR_POLICY=", ACTIVE_REPAIR_POLICY)

    preflight = run_preflight(config, project_root)
    print("--- PREFLIGHT ---")
    for entry in preflight["checks"]:
        print(f"PREFLIGHT_CHECK={entry['name']} STATUS={entry['status']}")
    print("PREFLIGHT_POLICY_EVIDENCE_TRAINING_AUTHORIZED=", preflight["policy_evidence_training_authorized"])
    # R11: the ACTIVE protocol's own informational fields replace the retired
    # PREFLIGHT_TWO_FOR_TWO_* console block (engine status, completeness
    # claim, complexity policy, operational-approval flags and the full-train
    # equivalence-class complexity diagnostics). Those described a
    # neighborhood the active protocol does not search. Printed
    # unconditionally, before the gate check, so a FAIL run still surfaces
    # them -- exactly like policy_evidence_training_authorized above. They do
    # not affect PREFLIGHT PASS/FAIL semantics.
    print("PREFLIGHT_ACTIVE_REPAIR_POLICY=", preflight["active_repair_policy"])
    print("PREFLIGHT_OBJECTIVE_REPAIR_NEIGHBORHOOD=", preflight["objective_repair_neighborhood"])
    print("PREFLIGHT_GLOBAL_OPTIMUM_CLAIMED=", preflight["global_optimum_claimed"])
    print("PREFLIGHT_OBJECTIVE_REPAIR_TERMINATION_CLAIM=", preflight["objective_repair_termination_claim"])
    print("PREFLIGHT_GATE=", preflight["status"])
    if preflight["status"] != "PASS":
        _print_gate_lines()
        fail("Preflight validation failed; see PREFLIGHT_CHECK lines above.", "PREFLIGHT_FAIL")

    if args.preflight_only:
        print("PHASE_2F_GATE= PREFLIGHT_PASS_ONLY (no official artifacts written)")
        _print_gate_lines()
        return

    if args.reconstruct_check:
        run_reconstruct_check(config, project_root, args.max_seconds)
        _print_gate_lines()
        return

    output_paths = official_relative_paths(config)
    existing = [project_root / p for p in output_paths if (project_root / p).exists()]
    if existing:
        fail("Refusing to overwrite official artifact(s): " + ", ".join(map(str, existing)), "REFUSE_OVERWRITE")

    bundle = compute_all_budgets(config, project_root, args.max_seconds)
    per_budget_outcome = bundle["per_budget_outcome"]
    per_budget_elapsed = bundle.get("per_budget_elapsed_seconds", {})
    for budget in BUDGET_ORDER:
        if budget in per_budget_outcome:
            print(f"BUDGET={budget} OUTCOME={per_budget_outcome[budget]} "
                  f"SIZE={len(bundle['per_budget_selected'][budget])} "
                  f"ELAPSED_SECONDS={per_budget_elapsed.get(budget)}")
    print(f"TOTAL_CONSTRUCTION_ELAPSED_SECONDS="
          f"{bundle.get('total_construction_elapsed_seconds')}")
    overall_outcome = "OK" if all(v == RepairOutcome.OK for v in per_budget_outcome.values()) else \
        next(v for v in per_budget_outcome.values() if v != RepairOutcome.OK)
    if overall_outcome != "OK":
        print("PHASE_2F_GATE=", overall_outcome)
        _print_gate_lines()
        fail(f"Budget construction did not complete: {per_budget_outcome}", overall_outcome)

    write_all_outputs_and_promote(config, project_root, bundle, output_paths)

    print("--- SUMMARY ---")
    for budget in BUDGET_ORDER:
        selected = bundle["per_budget_selected"][budget]
        print(f"BUDGET={budget} LABELED={len(selected)} "
              f"NO_FINDING={bundle['nf_size_target'][budget]}")
    print("REPAIR_MOVES_TOTAL=", len(bundle["repair_log"]))
    print("PHASE_2F_GATE= PASS")
    _print_gate_lines()
    for relative in output_paths:
        print("OUTPUT_FILE=", project_root / relative)


def write_all_outputs_and_promote(
    config: dict[str, Any], project_root: Path, bundle: dict[str, Any], output_paths: list[Path],
) -> None:
    staging = Path(tempfile.mkdtemp(prefix=".phase2F.staging-", dir=str(project_root)))
    promoted = False
    try:
        # Metadata/provenance revision: every official artifact below
        # reports the ACTIVE protocol_stage/protocol_version, read once
        # here via get_protocol_identity() (fail-closed), never a literal.
        protocol_stage, protocol_version = get_protocol_identity(config)
        outputs_cfg = config["outputs"]
        inputs = config["inputs"]
        forbidden_fields = set(config["unlabeled_json_forbidden_fields"]["images"])
        train = bundle["train"]
        image_ids = bundle["image_ids"]
        labels14 = bundle["labels14"]
        zero_gt = bundle["zero_gt"]
        category_names = bundle["category_names"]
        image_id_to_position = bundle["image_id_to_position"]
        K, S_T, N = bundle["K"], bundle["S_T"], bundle["N"]
        partition_seed = bundle["partition_seed"]
        repair_log = bundle["repair_log"]

        labeled_ids_by_budget = {
            b: {int(image_ids[p]) for p in bundle["per_budget_selected"][b]} for b in BUDGET_ORDER
        }
        unlabeled_ids_by_budget = {
            b: {int(v) for v in image_ids.tolist()} - labeled_ids_by_budget[b] for b in BUDGET_ORDER
        }

        for budget in BUDGET_ORDER:
            write_json(staging / Path(outputs_cfg["labeled_coco"][budget]),
                       subset_labeled_coco(train, labeled_ids_by_budget[budget]))
            write_json(staging / Path(outputs_cfg["unlabeled_coco"][budget]),
                       subset_unlabeled_coco(train, unlabeled_ids_by_budget[budget], forbidden_fields))

        audit_rows: list[dict[str, Any]] = []
        for budget in BUDGET_ORDER:
            audit_rows.extend(gt_audit_rows(
                train, unlabeled_ids_by_budget[budget], budget, labels14, zero_gt,
                image_id_to_position, category_names,
            ))
        audit_path = staging / Path(outputs_cfg["unlabeled_gt_audit_csv"])
        audit_path.parent.mkdir(parents=True, exist_ok=True)
        with audit_path.open("w", encoding="utf-8", newline="") as stream:
            fields = ("budget", "image_id", "zero_gt", "annotation_count", "present_class_names", "present_class_count")
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for row in audit_rows:
                writer.writerow(row)

        partition_path = staging / Path(outputs_cfg["partition_manifest_csv"])
        partition_path.parent.mkdir(parents=True, exist_ok=True)
        with partition_path.open("w", encoding="utf-8", newline="") as stream:
            fields = ("image_id", "in_labeled_1pct", "in_labeled_5pct", "in_labeled_10pct", "in_labeled_20pct",
                      "zero_gt", "annotation_count", "class_count")
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            ann_count = Counter(a["image_id"] for a in train["annotations"])
            classes_by_image: dict[int, set[int]] = {}
            for a in train["annotations"]:
                classes_by_image.setdefault(a["image_id"], set()).add(a["category_id"])
            for image_id in sorted(int(v) for v in image_ids.tolist()):
                position = image_id_to_position[image_id]
                writer.writerow({
                    "image_id": image_id,
                    "in_labeled_1pct": int(image_id in labeled_ids_by_budget["1pct"]),
                    "in_labeled_5pct": int(image_id in labeled_ids_by_budget["5pct"]),
                    "in_labeled_10pct": int(image_id in labeled_ids_by_budget["10pct"]),
                    "in_labeled_20pct": int(image_id in labeled_ids_by_budget["20pct"]),
                    "zero_gt": int(zero_gt[position]), "annotation_count": ann_count[image_id],
                    "class_count": len(classes_by_image.get(image_id, set())),
                })

        membership_sha256 = {b: canonical_membership_sha256(labeled_ids_by_budget[b]) for b in BUDGET_ORDER}
        coco_sha256 = {
            "labeled": {b: sha256_file(staging / Path(outputs_cfg["labeled_coco"][b])) for b in BUDGET_ORDER},
            "unlabeled": {b: sha256_file(staging / Path(outputs_cfg["unlabeled_coco"][b])) for b in BUDGET_ORDER},
        }
        repair_summary = {b: compute_repair_summary(repair_log, b) for b in BUDGET_ORDER}

        lock_manifest = {
            "phase": "Phase 2F — Labeled/Unlabeled Construction", "stage": protocol_stage,
            "protocol_version": protocol_version,
            "status": "LOCKED", "partition_seed": partition_seed, "seed_policy": config["seed"]["seed_policy"],
            # R11: which repair policy actually produced this locked membership.
            "active_repair_policy": ACTIVE_REPAIR_POLICY,
            "per_budget_elapsed_seconds": bundle["per_budget_elapsed_seconds"],
            "total_construction_elapsed_seconds": bundle["total_construction_elapsed_seconds"],
            "timing_clock": bundle["timing_clock"], "timing_role": bundle["timing_role"],
            "local_optimum_neighborhood": LOCAL_OPTIMUM_NEIGHBORHOOD,
            "global_optimum_claimed": False,
            "labeled_size": {b: len(labeled_ids_by_budget[b]) for b in BUDGET_ORDER},
            "no_finding_size": {
                b: int(sum(1 for i in labeled_ids_by_budget[b] if zero_gt[image_id_to_position[i]] == 1))
                for b in BUDGET_ORDER
            },
            "labeled_image_id_sha256": membership_sha256, "coco_json_sha256": coco_sha256,
            "repair_move_counts": {b: repair_summary[b]["total_moves"] for b in BUDGET_ORDER},
            "integer_objective_final": {
                b: list(integer_objective(
                    {image_id_to_position[i] for i in labeled_ids_by_budget[b]}, labels14, K, S_T, N,
                )) for b in BUDGET_ORDER
            },
            "train_coco_sha256": sha256_file(project_root / inputs["train_coco"]),
        }
        write_json(staging / Path(outputs_cfg["lock_manifest_json"]), lock_manifest)

        nested_ok = all(
            labeled_ids_by_budget[BUDGET_ORDER[i]] <= labeled_ids_by_budget[BUDGET_ORDER[i + 1]]
            for i in range(len(BUDGET_ORDER) - 1)
        )
        nested_nf_checks = {}
        for i in range(len(BUDGET_ORDER) - 1):
            smaller, larger = BUDGET_ORDER[i], BUDGET_ORDER[i + 1]
            smaller_nf = {i for i in labeled_ids_by_budget[smaller] if zero_gt[image_id_to_position[i]] == 1}
            larger_nf = {i for i in labeled_ids_by_budget[larger] if zero_gt[image_id_to_position[i]] == 1}
            nested_nf_checks[f"{smaller}_in_{larger}"] = smaller_nf <= larger_nf
        nested_nf_ok = all(nested_nf_checks.values())
        nested_split_check = {
            "phase": protocol_stage, "protocol_version": protocol_version,
            "status": "PASS" if (nested_ok and nested_nf_ok) else "FAIL",
            "nested_relation": "1pct subset_of 5pct subset_of 10pct subset_of 20pct",
            "nested_pass": nested_ok, "nested_no_finding_checks": nested_nf_checks, "nested_no_finding_pass": nested_nf_ok,
            "labeled_sizes": {b: len(labeled_ids_by_budget[b]) for b in BUDGET_ORDER},
        }
        write_json(staging / Path(outputs_cfg["nested_split_check_json"]), nested_split_check)
        if not (nested_ok and nested_nf_ok):
            fail("Nested labeled-split or nested-No-Finding invariant failed post-construction", "NESTED_CHECK_FAIL")

        val_test_ids: set[int] = set()
        for name in ("val", "test"):
            coco_vt = load_json(project_root / inputs[f"{name}_coco"])
            val_test_ids |= {im["id"] for im in coco_vt["images"]}
        leakage_ok = all(not (labeled_ids_by_budget[b] & val_test_ids) for b in BUDGET_ORDER)
        for b in BUDGET_ORDER:
            leakage_ok = leakage_ok and not (unlabeled_ids_by_budget[b] & val_test_ids)
        train_ids_set = {int(v) for v in image_ids.tolist()}
        completeness_ok = all(
            (labeled_ids_by_budget[b] | unlabeled_ids_by_budget[b]) == train_ids_set
            and not (labeled_ids_by_budget[b] & unlabeled_ids_by_budget[b])
            for b in BUDGET_ORDER
        )
        leakage_check = {
            "phase": protocol_stage, "protocol_version": protocol_version,
            "status": "PASS" if (leakage_ok and completeness_ok) else "FAIL",
            "val_test_isolation_pass": leakage_ok, "labeled_unlabeled_disjoint_and_complete_pass": completeness_ok,
        }
        write_json(staging / Path(outputs_cfg["leakage_check_json"]), leakage_check)
        if not (leakage_ok and completeness_ok):
            fail("Leakage/isolation/completeness check failed post-construction", "LEAKAGE_CHECK_FAIL")

        write_json(staging / Path(outputs_cfg["seed_manifest_json"]), {
            "phase": protocol_stage, "protocol_version": protocol_version, "partition_seed": partition_seed,
            "seed_policy": config["seed"]["seed_policy"],
            "tie_break_sha256_priority_seed": config["tie_break"]["sha256_priority_seed"],
            "tie_break_namespaces": config["tie_break"]["namespaces"],
            "iterative_stratification_version": bundle["iterative_stratification_version"],
            "iterative_stratification_required_version": config["dependencies"]["iterative_stratification"]["required_version"],
        })

        class_dist_path = staging / Path(outputs_cfg["class_distribution_csv"])
        class_dist_path.parent.mkdir(parents=True, exist_ok=True)
        per_budget_distribution: dict[str, Any] = {}
        with class_dist_path.open("w", encoding="utf-8", newline="") as stream:
            fields = ("budget", "class_name", "labeled_image_count", "train_prevalence", "labeled_prevalence", "absolute_deviation")
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for budget in BUDGET_ORDER:
                positions = sorted(image_id_to_position[i] for i in labeled_ids_by_budget[budget])
                d_max, d_mean, d_lc, train_prev, labeled_prev, deviations, worst_idx = fraction_distribution_report(
                    positions, labels14, K, N,
                )
                per_budget_distribution[budget] = {
                    "D_max": str(d_max), "D_mean": str(d_mean), "D_LC": str(d_lc),
                    "worst_deviation_class": category_names[worst_idx],
                    "train_reference_prevalence_per_class": {category_names[c]: str(train_prev[c]) for c in range(14)},
                    "labeled_prevalence_per_class": {category_names[c]: str(labeled_prev[c]) for c in range(14)},
                    "absolute_deviation_per_class": {category_names[c]: str(deviations[c]) for c in range(14)},
                }
                for c in range(14):
                    writer.writerow({
                        "budget": budget, "class_name": category_names[c],
                        "labeled_image_count": int(labels14[positions, c].sum()),
                        "train_prevalence": str(train_prev[c]), "labeled_prevalence": str(labeled_prev[c]),
                        "absolute_deviation": str(deviations[c]),
                    })

        neg_dist_path = staging / Path(outputs_cfg["negative_distribution_csv"])
        neg_dist_path.parent.mkdir(parents=True, exist_ok=True)
        with neg_dist_path.open("w", encoding="utf-8", newline="") as stream:
            fields = ("budget", "total_images", "negative_images", "abnormal_images", "negative_fraction")
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for budget in BUDGET_ORDER:
                positions = [image_id_to_position[i] for i in labeled_ids_by_budget[budget]]
                total = len(positions)
                negative = int(sum(zero_gt[p] for p in positions))
                writer.writerow({
                    "budget": budget, "total_images": total, "negative_images": negative,
                    "abnormal_images": total - negative,
                    "negative_fraction": str(Fraction(negative, total)) if total else "0",
                })

        write_jsonl(staging / Path(outputs_cfg["repair_log_jsonl"]), repair_log)

        errors_path = staging / Path(outputs_cfg["errors_csv"])
        errors_path.parent.mkdir(parents=True, exist_ok=True)
        with errors_path.open("w", encoding="utf-8", newline="") as stream:
            csv.DictWriter(stream, fieldnames=("error_type", "detail")).writeheader()

        exposure_violations = []
        for budget in BUDGET_ORDER:
            data = load_json(staging / Path(outputs_cfg["unlabeled_coco"][budget]))
            if data["annotations"] != []:
                exposure_violations.append(f"{budget}: annotations not empty")
            for im in data["images"]:
                leaked = forbidden_fields & set(im.keys())
                if leaked:
                    exposure_violations.append(f"{budget}: image {im['id']} leaked fields {sorted(leaked)}")

        readback_gates = independently_validate_staging(staging, project_root, config)

        report = {
            "phase": "Phase 2F — Labeled/Unlabeled Construction", "stage": protocol_stage,
            "protocol_version": protocol_version,
            "status": "PASS" if (not exposure_violations and readback_gates["status"] == "PASS") else "FAIL",
            "partition_seed": partition_seed,
            "iterative_stratification_version": bundle["iterative_stratification_version"],
            "per_budget_elapsed_seconds": bundle["per_budget_elapsed_seconds"],
            "total_construction_elapsed_seconds": bundle["total_construction_elapsed_seconds"],
            "timing_clock": bundle["timing_clock"], "timing_role": bundle["timing_role"],
            "budgets": {
                b: {
                    "labeled_size": len(labeled_ids_by_budget[b]), "unlabeled_size": len(unlabeled_ids_by_budget[b]),
                    "no_finding_size": lock_manifest["no_finding_size"][b],
                    "integer_objective_E_max_E_mean_E_LC": lock_manifest["integer_objective_final"][b],
                    **per_budget_distribution[b],
                    **compute_repair_summary(repair_log, b),
                    "repair_fraction": (
                        compute_repair_summary(repair_log, b)["total_moves"] / len(labeled_ids_by_budget[b])
                        if labeled_ids_by_budget[b] else 0
                    ),
                    "candidate_final_intersection_count": bundle["per_budget_exhaustion"][b].get("candidate_final_intersection_count"),
                    "candidate_final_jaccard": bundle["per_budget_exhaustion"][b].get("candidate_final_jaccard"),
                    "objective_before_repair": bundle["per_budget_exhaustion"][b].get("objective_before_repair"),
                    "objective_after_repair": bundle["per_budget_exhaustion"][b].get("objective_after_repair"),
                    # R11: the active exhaustion contract. `two_for_two_exhausted`
                    # is gone -- the active protocol never searches that
                    # neighborhood, so reporting a flag about it would be a
                    # claim about a search that never ran.
                    "neighborhood_exhaustion_status": {
                        k: v for k, v in bundle["per_budget_exhaustion"][b].items()
                        if k in ("one_for_one_exhausted", "local_optimum",
                                 "local_optimum_neighborhood", "global_optimum_claimed")
                    },
                    "final_one_for_one_diagnostics": bundle["per_budget_exhaustion"][b].get(
                        "final_one_for_one_diagnostics"
                    ),
                    "labeled_image_id_sha256": membership_sha256[b],
                    "labeled_coco_sha256": coco_sha256["labeled"][b], "unlabeled_coco_sha256": coco_sha256["unlabeled"][b],
                } for b in BUDGET_ORDER
            },
            "nested_split_check": nested_split_check, "leakage_check": leakage_check,
            "unlabeled_gt_exposure_violations": exposure_violations,
            "independent_readback_gates": readback_gates,
            # R11: the active protocol identity of this materialization.
            "active_repair_policy": bundle.get("active_repair_policy", ACTIVE_REPAIR_POLICY),
            "objective_repair_neighborhood": LOCAL_OPTIMUM_NEIGHBORHOOD,
            "global_optimum_claimed": False,
            "objective_repair_termination_claim": (
                "The objective-repair stage exhaustively enumerates the admissible one-for-one "
                "swap neighborhood and terminates at a one-for-one local optimum. No global "
                "optimum is claimed."
            ),
            "legacy_two_for_two_engine_invoked": False,
            "repair_move_count_total": len(repair_log),
            "deterministic_rerun": "TO VERIFY BY RESEARCHER RUN (use --reconstruct-check)",
            "training_authorized": False, "training_started": False,
            "pseudo_labels_generated": False, "test_used": False,
        }
        validation_report_path = staging / Path(outputs_cfg["validation_report_json"])
        write_json(validation_report_path, report)

        write_json(staging / Path(outputs_cfg["log_json"]), {
            "phase": protocol_stage, "protocol_version": protocol_version,
            "status": report["status"], "files_written": len(output_paths),
            "official_artifacts": [str(p).replace("\\", "/") for p in output_paths],
            "overwrite_policy": "REFUSE_IF_ANY_OFFICIAL_ARTIFACT_EXISTS",
            "rollback_policy": "REMOVE_FILES_PROMOTED_BY_FAILED_RUN",
        })

        if exposure_violations:
            fail(f"Unlabeled GT exposure audit failed: {exposure_violations}", "GT_EXPOSURE_FAIL")
        if readback_gates["status"] != "PASS":
            failing = [k for k, v in readback_gates["gates"].items() if not v]
            fail(f"Independent readback validation failed hard gates: {failing}", "READBACK_FAIL")

        readback_report = load_json(validation_report_path)
        if readback_report.get("status") != "PASS":
            fail("Validation report readback did not preserve PASS status", "READBACK_FAIL")
        for relative in output_paths:
            if not (staging / relative).is_file():
                fail(f"Missing staged official artifact: {relative}", "READBACK_FAIL")

        promote_with_rollback(staging, project_root, output_paths)
        promoted = True
    finally:
        if not promoted and staging.exists():
            shutil.rmtree(staging)


def run_reconstruct_check(config: dict[str, Any], project_root: Path, max_seconds: float | None = None) -> None:
    """Non-promoting deterministic reconstruction/readback. Never subject to
    the refuse-overwrite gate; writes only the diagnostic comparison report,
    which is intentionally excluded from official_relative_paths(). Compares
    EVERY field listed in config's deterministic_reconstruction_mode.
    compared_fields against RECONSTRUCTION_COMPARABLE_FIELDS -- a configured
    field with no implemented check is a hard FAIL (CONFIG_INVALID), never a
    silent skip, so the YAML list and the implementation cannot drift apart."""
    lock_path = project_root / Path(config["outputs"]["lock_manifest_json"])
    if not lock_path.is_file():
        fail("--reconstruct-check requires an already-promoted data/manifests/phase2F_lock_manifest.json. "
             "Run the official materialization first.", "RECONSTRUCT_PRECONDITION_FAILED")
    promoted_lock = load_json(lock_path)

    # Metadata/provenance revision: this diagnostic reports the ACTIVE
    # protocol_stage/protocol_version, read via get_protocol_identity()
    # (fail-closed), never a literal baked into this function.
    protocol_stage, protocol_version = get_protocol_identity(config)

    compared_fields = list(config["deterministic_reconstruction_mode"]["compared_fields"])
    unknown_fields = [f for f in compared_fields if f not in RECONSTRUCTION_COMPARABLE_FIELDS]
    if unknown_fields:
        fail(
            f"deterministic_reconstruction_mode.compared_fields lists field(s) with no implemented check: "
            f"{unknown_fields}. Every configured field must have a corresponding check in run_reconstruct_check.",
            "CONFIG_INVALID",
        )

    bundle = compute_all_budgets(config, project_root, max_seconds)
    per_budget_outcome = bundle["per_budget_outcome"]
    if not all(v == RepairOutcome.OK for v in per_budget_outcome.values()):
        fail(f"--reconstruct-check: budget construction did not complete cleanly: {per_budget_outcome}",
             "RECONSTRUCT_CONSTRUCTION_FAILED")

    train = bundle["train"]
    image_ids = bundle["image_ids"]
    zero_gt = bundle["zero_gt"]
    labels14 = bundle["labels14"]
    K, S_T, N = bundle["K"], bundle["S_T"], bundle["N"]
    image_id_to_position = bundle["image_id_to_position"]
    forbidden_fields = set(config["unlabeled_json_forbidden_fields"]["images"])

    labeled_ids_by_budget = {b: {int(image_ids[p]) for p in bundle["per_budget_selected"][b]} for b in BUDGET_ORDER}
    unlabeled_ids_by_budget = {
        b: {int(v) for v in image_ids.tolist()} - labeled_ids_by_budget[b] for b in BUDGET_ORDER
    }

    recomputed: dict[str, dict[str, Any]] = {field: {} for field in RECONSTRUCTION_COMPARABLE_FIELDS}
    promoted: dict[str, dict[str, Any]] = {field: {} for field in RECONSTRUCTION_COMPARABLE_FIELDS}

    # Non-promoting: labeled/unlabeled COCO subsets are written to a
    # throwaway temp directory ONLY to compute their sha256 for comparison,
    # then discarded. Nothing under project_root is ever written here.
    with tempfile.TemporaryDirectory(prefix="phase2F-reconstruct-") as tmp_str:
        tmp_dir = Path(tmp_str)
        labeled_paths: dict[str, Path] = {}
        unlabeled_paths: dict[str, Path] = {}
        for b in BUDGET_ORDER:
            labeled_paths[b] = tmp_dir / f"labeled_{b}.json"
            unlabeled_paths[b] = tmp_dir / f"unlabeled_{b}.json"
            write_json(labeled_paths[b], subset_labeled_coco(train, labeled_ids_by_budget[b]))
            write_json(unlabeled_paths[b], subset_unlabeled_coco(train, unlabeled_ids_by_budget[b], forbidden_fields))

        for b in BUDGET_ORDER:
            recomputed["labeled_image_id_sha256"][b] = canonical_membership_sha256(labeled_ids_by_budget[b])
            recomputed["labeled_coco_json_sha256"][b] = sha256_file(labeled_paths[b])
            recomputed["unlabeled_coco_json_sha256"][b] = sha256_file(unlabeled_paths[b])
            recomputed["labeled_size"][b] = len(labeled_ids_by_budget[b])
            recomputed["no_finding_size"][b] = int(
                sum(1 for i in labeled_ids_by_budget[b] if zero_gt[image_id_to_position[i]] == 1)
            )
            recomputed["repair_move_counts"][b] = compute_repair_summary(bundle["repair_log"], b)["total_moves"]
            recomputed["integer_objective_final"][b] = list(integer_objective(
                {image_id_to_position[i] for i in labeled_ids_by_budget[b]}, labels14, K, S_T, N,
            ))

            promoted["labeled_image_id_sha256"][b] = promoted_lock.get("labeled_image_id_sha256", {}).get(b)
            promoted["labeled_coco_json_sha256"][b] = promoted_lock.get("coco_json_sha256", {}).get("labeled", {}).get(b)
            promoted["unlabeled_coco_json_sha256"][b] = promoted_lock.get("coco_json_sha256", {}).get("unlabeled", {}).get(b)
            promoted["labeled_size"][b] = promoted_lock.get("labeled_size", {}).get(b)
            promoted["no_finding_size"][b] = promoted_lock.get("no_finding_size", {}).get(b)
            promoted["repair_move_counts"][b] = promoted_lock.get("repair_move_counts", {}).get(b)
            promoted["integer_objective_final"][b] = promoted_lock.get("integer_objective_final", {}).get(b)

    diffs: dict[str, Any] = {}
    for b in BUDGET_ORDER:
        diffs[b] = {f"{field}_match": recomputed[field][b] == promoted[field][b] for field in compared_fields}
    overall_match = all(all(v.values()) for v in diffs.values())

    result = {
        "phase": protocol_stage, "protocol_version": protocol_version,
        "mode": "deterministic_reconstruction_check",
        "official_lock_manifest": str(lock_path), "status": "MATCH" if overall_match else "MISMATCH",
        "compared_fields": compared_fields,
        "iterative_stratification_version": bundle["iterative_stratification_version"],
        "per_budget_diff": diffs,
    }
    output_path = project_root / "reports" / "02F_deterministic_reconstruction_check.json"
    write_json(output_path, result)
    print("--- RECONSTRUCT CHECK ---")
    for b in BUDGET_ORDER:
        print(f"BUDGET={b} MATCH={all(diffs[b].values())}")
    print("RECONSTRUCT_CHECK_STATUS=", result["status"])
    print("RECONSTRUCT_CHECK_REPORT=", output_path)
    if not overall_match:
        fail("Deterministic reconstruction check found a mismatch against the promoted lock manifest.",
             "RECONSTRUCT_MISMATCH")


if __name__ == "__main__":
    try:
        main()
    except Phase2FError as error:
        print(f"PHASE_2F_GATE= {error}")
        _print_gate_lines()
        print(f"ERROR_MESSAGE= {error}", file=sys.stderr)
        sys.exit(1)
    except Exception as error:
        print("PHASE_2F_GATE= ERROR")
        _print_gate_lines()
        print(f"ERROR_TYPE= {type(error).__name__}", file=sys.stderr)
        print(f"ERROR_MESSAGE= {error}", file=sys.stderr)
        sys.exit(1)
