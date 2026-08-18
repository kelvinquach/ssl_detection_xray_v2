# -*- coding: utf-8 -*-
"""Phase 2F.1 - seed protocol guardrails.

These tests are an INDEPENDENT check of the Phase 2F.1 contract. They do not
import scripts/02F1_build_seed_protocol.py: the locked constants are restated
here and the SHA-256 derivation rule is re-implemented here, so that a bug in
the builder cannot hide itself from the guardrails.

The tests never train, never load a model, never touch a DataLoader, never
read test data and never modify any artifact.

Run:
    python -m pytest tests\\test_phase2F1_seed_protocol_guardrails.py -v \\
        --junitxml=reports\\02F1_guardrails_junit.xml

They require that the artifacts already exist, i.e. that
    python scripts\\02F1_build_seed_protocol.py --execute
has been run first.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import pytest
import yaml


# =====================================================================
# Paths
# =====================================================================

REPO_ROOT = Path(__file__).resolve().parents[1]

CONFIG_REL = "configs/protocol/phase2F1_seed_protocol.yaml"
SEED_MANIFEST_REL = "data/manifests/seed_manifest.json"
STATE_MANIFEST_REL = "data/manifests/seed_state_manifest.json"
MARKDOWN_REL = "reports/seed_protocol.md"
VALIDATION_REPORT_REL = "reports/02F1_seed_protocol_validation_report.json"

CONFIG_PATH = REPO_ROOT / CONFIG_REL
SEED_MANIFEST_PATH = REPO_ROOT / SEED_MANIFEST_REL
STATE_MANIFEST_PATH = REPO_ROOT / STATE_MANIFEST_REL
MARKDOWN_PATH = REPO_ROOT / MARKDOWN_REL
VALIDATION_REPORT_PATH = REPO_ROOT / VALIDATION_REPORT_REL


# =====================================================================
# Locked constants restated independently
# =====================================================================

EXPECTED_PARTITION_SEED = 42
EXPECTED_PARTITION_SEED_POLICY = "PRE_SPECIFIED_LOCKED_NO_SEED_SEARCH"

EXPECTED_PROTOCOL_IDENTITY = "2F-C0-R11"
EXPECTED_PROTOCOL_VERSION = "2.0.0"
EXPECTED_TRAIN_SIZE = 3426
EXPECTED_LABELED_BUDGETS = ["1pct", "5pct", "10pct", "20pct"]
EXPECTED_LABELED_SIZES = [34, 171, 343, 685]
EXPECTED_UNLABELED_SIZES = [3392, 3255, 3083, 2741]
EXPECTED_MEMBERSHIP_CHECKSUMS = {
    "1pct": "c54e7d61e84b7cfce68c04a795783b5c5d01331d2aa9c754fb1ae1dbae4ba071",
    "5pct": "c4db3b5f7a5b0f391ad883ef665c3d341af3ef6afb3fc553e71f1c4ef17ee50b",
    "10pct": "fc008d31505227544087ba474613075a8cd077587df9d6b13146b378f5a3a7d6",
    "20pct": "6f4aaba6be147983d56007c49ada234dfcabe7ec2f936f28f8cd240999b8417e",
}

EXPECTED_TRAINING_SEED_COUNT = 10
EXPECTED_DETERMINISTIC_POLICY = "CONTROLLED_BEST_EFFORT"
EXPECTED_TRAINING_SEEDS = [
    204886845,
    1480646854,
    1798418854,
    2045683682,
    1814859839,
    1603952859,
    1878351743,
    875651179,
    477581743,
    869675675,
]

EXPECTED_NAMESPACE = "ssl_detection_xray_v2|phase2F.1|training_seed"
SEED_MIN = 1
SEED_MAX = 2 ** 31 - 1

EXPECTED_RUN_FIELDS = [
    "run_id",
    "run_status",
    "method",
    "configuration_id",
    "budget",
    "partition_seed",
    "training_seed",
    "training_seed_index",
    "rng_state_id",
    "membership_checksum",
    "config_hash",
    "code_revision",
    "environment",
    "deterministic_runtime_settings",
    "checkpoint",
    "results",
    "retry_of",
    "technical_failure_reason",
]

EXPECTED_PROHIBITED_CLAIM_KEYS = [
    "statistical_optimality_of_seed_count",
    "power_analysis_performed",
    "stability_proven",
    "variance_measured_in_phase2F1",
    "bitwise_reproducibility_across_environments",
]

PROHIBITED_CLAIM_PATTERNS = [
    ("statistical_optimality_of_seed_count", r"statistically\s+optimal"),
    ("statistical_optimality_of_seed_count",
     r"optimal\s+(number|count)\s+of\s+(training\s+)?seeds"),
    ("power_analysis_performed", r"power\s+analysis"),
    ("stability_proven",
     r"(stability|robustness)\s+(is|are|was|were|has\s+been|have\s+been)"
     r"\s+(proven|proved|demonstrated|established|confirmed|verified)"),
    ("stability_proven", r"proven\s+(stability|robustness)"),
    ("variance_measured_in_phase2F1",
     r"(variance|variability)\s+(is|was|has\s+been)"
     r"\s+(measured|quantified|estimated)"),
    ("bitwise_reproducibility_across_environments",
     r"(guarantee|guarantees|guaranteed|ensure|ensures|ensured|assure|assures)"
     r"\s+(bit-?wise\s+)?(reproducibility|determinism|identical\s+results)"),
    ("bitwise_reproducibility_across_environments",
     r"bit-?wise\s+(reproducibility|determinism)\s+(is|was|has\s+been)"
     r"\s+(guaranteed|ensured|achieved|verified)"),
]

REQUIRED_GPU_LIMITATION_TERMS = [
    "gpu", "cuda", "cudnn", "driver", "hardware", "software",
]

PHASE_CLOSURE_STATUS = "PENDING_RESEARCHER_GPT_REVIEW"
STATE_TEMPLATE_VALUE = "TEMPLATE_LOCKED_NO_RUNS"

MISSING_ARTIFACT_HINT = (
    "Artifact not found. Run "
    "'python scripts\\02F1_build_seed_protocol.py --execute' first."
)


# =====================================================================
# Helpers and fixtures
# =====================================================================

def dig(obj, dotted_path, default=None):
    current = obj
    for part in dotted_path.split("."):
        if isinstance(current, dict) and part in current:
            current = current[part]
        else:
            return default
    return current


def norm_text(value):
    if value is None:
        return ""
    return " ".join(str(value).split())


def derive_seed(index, namespace=EXPECTED_NAMESPACE):
    """Independent re-implementation of the locked public rule."""
    payload = "{0}|index={1}".format(namespace, index)
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return 1 + (int(digest[:8], 16) % (2 ** 31 - 1))


def _read_json(path, rel):
    if not path.is_file():
        pytest.fail("{0} : {1}".format(rel, MISSING_ARTIFACT_HINT))
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def cfg():
    if not CONFIG_PATH.is_file():
        pytest.fail("missing contract file: {0}".format(CONFIG_REL))
    return yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def config_raw():
    if not CONFIG_PATH.is_file():
        pytest.fail("missing contract file: {0}".format(CONFIG_REL))
    return CONFIG_PATH.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def seed_manifest():
    return _read_json(SEED_MANIFEST_PATH, SEED_MANIFEST_REL)


@pytest.fixture(scope="module")
def state_manifest():
    return _read_json(STATE_MANIFEST_PATH, STATE_MANIFEST_REL)


@pytest.fixture(scope="module")
def validation_report():
    return _read_json(VALIDATION_REPORT_PATH, VALIDATION_REPORT_REL)


@pytest.fixture(scope="module")
def markdown_text():
    if not MARKDOWN_PATH.is_file():
        pytest.fail("{0} : {1}".format(MARKDOWN_REL, MISSING_ARTIFACT_HINT))
    return MARKDOWN_PATH.read_text(encoding="utf-8")


# =====================================================================
# G01 - partition seed value
# =====================================================================

def test_g01_partition_seed_is_42(cfg, seed_manifest, state_manifest):
    assert dig(cfg, "partition.partition_seed") == EXPECTED_PARTITION_SEED
    assert dig(seed_manifest, "partition.partition_seed") == \
        EXPECTED_PARTITION_SEED
    assert state_manifest["partition_seed"] == EXPECTED_PARTITION_SEED


# =====================================================================
# G02 - partition seed policy
# =====================================================================

def test_g02_partition_seed_policy_matches_exactly(cfg, seed_manifest):
    assert dig(cfg, "partition.partition_seed_policy") == \
        EXPECTED_PARTITION_SEED_POLICY
    assert dig(seed_manifest, "partition.partition_seed_policy") == \
        EXPECTED_PARTITION_SEED_POLICY
    assert dig(cfg, "partition.constraints.partition_seed_mutable") is False
    assert dig(cfg,
               "partition.constraints.partition_seed_equals_training_seed") \
        is False
    assert dig(cfg, "partition.terminology.canonical_term") == "partition_seed"
    assert "split_seed = legacy alias of partition_seed" in \
        norm_text(dig(cfg, "partition.terminology.legacy_alias_note"))


# =====================================================================
# G03 - exactly ten training seeds
# =====================================================================

def test_g03_exactly_ten_training_seeds(cfg, seed_manifest, state_manifest):
    assert dig(cfg, "training_seed.training_seed_count") == \
        EXPECTED_TRAINING_SEED_COUNT
    assert len(dig(cfg, "training_seed.ordered_training_seed_values", [])) == \
        EXPECTED_TRAINING_SEED_COUNT
    assert len(dig(cfg, "training_seed.ordered_training_seeds", [])) == \
        EXPECTED_TRAINING_SEED_COUNT
    assert len(dig(seed_manifest, "training_seed.seeds", [])) == \
        EXPECTED_TRAINING_SEED_COUNT
    assert state_manifest["training_seed_count"] == EXPECTED_TRAINING_SEED_COUNT


# =====================================================================
# G04 - the ordered list matches exactly
# =====================================================================

def test_g04_ordered_seed_list_matches_locked_list(cfg, seed_manifest,
                                                   state_manifest):
    assert list(dig(cfg, "training_seed.ordered_training_seed_values")) == \
        EXPECTED_TRAINING_SEEDS

    pairs = dig(cfg, "training_seed.ordered_training_seeds", [])
    assert [entry["index"] for entry in pairs] == \
        list(range(1, EXPECTED_TRAINING_SEED_COUNT + 1))
    assert [entry["training_seed"] for entry in pairs] == EXPECTED_TRAINING_SEEDS

    assert list(dig(seed_manifest,
                    "training_seed.ordered_training_seed_values")) == \
        EXPECTED_TRAINING_SEEDS
    assert [entry["training_seed"]
            for entry in dig(seed_manifest, "training_seed.seeds")] == \
        EXPECTED_TRAINING_SEEDS
    assert list(state_manifest["ordered_training_seed_values"]) == \
        EXPECTED_TRAINING_SEEDS
    assert dig(cfg, "training_seed.ordering") == "FIXED_ORDERED_LIST"


# =====================================================================
# G05 - uniqueness and domain
# =====================================================================

def test_g05_seeds_unique_and_in_domain(cfg):
    values = list(dig(cfg, "training_seed.ordered_training_seed_values"))
    assert len(set(values)) == len(values)
    for value in values:
        assert isinstance(value, int)
        assert SEED_MIN <= value <= SEED_MAX
    assert dig(cfg, "training_seed.seed_domain.minimum") == SEED_MIN
    assert dig(cfg, "training_seed.seed_domain.maximum") == SEED_MAX
    assert dig(cfg, "training_seed.uniqueness_required") is True


# =====================================================================
# G06 - the SHA-256 rule reproduces the locked list
# =====================================================================

def test_g06_sha256_rule_reproduces_all_ten_seeds(cfg, seed_manifest):
    generation = dig(cfg, "training_seed.generation", {})
    assert generation.get("namespace") == EXPECTED_NAMESPACE
    assert generation.get("input_encoding") == "UTF-8"
    assert generation.get("digest_algorithm") == "SHA-256"
    assert generation.get("digest_prefix_hex_chars") == 8
    assert generation.get("integer_base") == 16
    assert generation.get("modulus") == SEED_MAX
    assert generation.get("offset") == 1
    assert generation.get("index_start") == 1
    assert generation.get("index_end") == EXPECTED_TRAINING_SEED_COUNT

    recomputed = [derive_seed(index)
                  for index in range(1, EXPECTED_TRAINING_SEED_COUNT + 1)]
    assert recomputed == EXPECTED_TRAINING_SEEDS
    assert recomputed == list(
        dig(cfg, "training_seed.ordered_training_seed_values"))
    assert recomputed == list(
        dig(seed_manifest, "training_seed.ordered_training_seed_values"))

    for record in dig(seed_manifest, "training_seed.seeds"):
        payload = "{0}|index={1}".format(EXPECTED_NAMESPACE, record["index"])
        digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
        assert record["payload"] == payload
        assert record["sha256_digest"] == digest
        assert record["digest_prefix_hex"] == digest[:8]
        assert record["training_seed"] == 1 + (int(digest[:8], 16) % SEED_MAX)


# =====================================================================
# G07 - no seed search
# =====================================================================

def test_g07_no_seed_search(cfg, seed_manifest):
    assert dig(cfg, "partition.constraints.seed_search_performed") is False
    assert dig(cfg,
               "training_seed.selection_constraints.seed_search_performed") \
        is False
    assert dig(cfg, "training_seed.generation.rng_calls_used") == 0
    assert dig(cfg, "training_seed.generation.method") == \
        "SHA256_PUBLIC_DERIVATION_NO_RNG"
    assert dig(cfg, "training_seed.generation.generated_before_training") is True
    assert dig(cfg, "training_seed.selection_constraints."
                    "seed_list_mutable_after_lock") is False
    assert dig(seed_manifest, "training_seed.rng_calls_used") == 0


# =====================================================================
# G08 - validation / test data never used to select seeds
# =====================================================================

def test_g08_validation_and_test_not_used_for_seed_selection(cfg):
    assert dig(cfg, "partition.constraints."
                    "validation_data_used_for_seed_selection") is False
    assert dig(cfg, "partition.constraints."
                    "test_data_used_for_seed_selection") is False
    assert dig(cfg, "training_seed.selection_constraints."
                    "seeds_selected_using_validation_data") is False
    assert dig(cfg, "training_seed.selection_constraints."
                    "seeds_selected_using_test_data") is False
    assert dig(cfg, "training_seed.selection_constraints."
                    "seeds_selected_using_results") is False


# =====================================================================
# G09 - supervised and SSL share the ordered seed list
# =====================================================================

def test_g09_supervised_and_ssl_share_same_ordered_seed_list(cfg):
    pairing = dig(cfg, "pairing_policy", {})
    assert pairing.get(
        "supervised_and_ssl_share_same_ordered_seed_list") is True
    assert pairing.get("pairing_key") == "training_seed_index"
    assert pairing.get("per_method_seed_lists_allowed") is False
    assert pairing.get("reshuffling_allowed") is False
    assert pairing.get("subsetting_allowed") is False
    assert pairing.get("partition_seed_shared_across_all_runs") is True
    assert list(pairing.get("budgets_covered", [])) == EXPECTED_LABELED_BUDGETS
    assert norm_text(pairing.get("rule"))


# =====================================================================
# G10 - retry cannot change the seed or silently replace a run
# =====================================================================

def test_g10_retry_policy_is_seed_preserving_and_auditable(cfg):
    retry = dig(cfg, "retry_policy", {})
    assert retry.get("retry_allowed_reason") == "TECHNICAL_FAILURE_ONLY"
    assert retry.get("retry_changes_training_seed") is False
    assert retry.get("retry_must_reuse_same_training_seed") is True
    assert retry.get("retry_must_reuse_same_training_seed_index") is True
    assert retry.get("silent_replacement_allowed") is False
    assert retry.get("result_based_retry_allowed") is False
    assert retry.get("failed_run_must_be_retained") is True
    assert retry.get("failed_run_status_value") == "TECHNICAL_FAILURE"
    assert retry.get("technical_failure_reason_required") is True
    assert retry.get("retry_run_must_reference_original") is True
    assert retry.get("retry_reference_field") == "retry_of"
    assert retry.get("all_attempts_must_be_recorded") is True


# =====================================================================
# G11 - per-seed, mean, sample SD with ddof=1
# =====================================================================

def test_g11_aggregation_reports_per_seed_mean_and_sample_sd(cfg,
                                                             markdown_text):
    aggregation = dig(cfg, "aggregation_policy", {})
    assert aggregation.get("report_per_seed_results") is True
    assert aggregation.get("report_mean_across_seeds") is True
    assert aggregation.get("report_sample_standard_deviation") is True
    assert aggregation.get("standard_deviation_ddof") == 1
    assert aggregation.get("standard_deviation_type") == "SAMPLE"
    assert aggregation.get("n_for_aggregation") == EXPECTED_TRAINING_SEED_COUNT
    assert aggregation.get("seed_dropping_allowed") is False
    assert aggregation.get("outlier_removal_allowed") is False
    assert aggregation.get("best_seed_only_reporting_allowed") is False
    assert "ddof=1" in markdown_text


# =====================================================================
# G12 - both rationales exist
# =====================================================================

def test_g12_both_rationales_exist(cfg, seed_manifest, markdown_text):
    rationale_a = norm_text(
        dig(cfg, "rationale.training_seed_count_rationale"))
    rationale_b = norm_text(
        dig(cfg, "rationale.controlled_best_effort_rationale"))
    assert len(rationale_a) >= 120
    assert len(rationale_b) >= 120

    assert norm_text(
        dig(seed_manifest, "rationale.training_seed_count_rationale")) == \
        rationale_a
    assert norm_text(
        dig(seed_manifest, "rationale.controlled_best_effort_rationale")) == \
        rationale_b

    markdown_norm = norm_text(markdown_text)
    assert rationale_a in markdown_norm
    assert rationale_b in markdown_norm
    assert "training_seed_count_rationale" in markdown_text
    assert "controlled_best_effort_rationale" in markdown_text


# =====================================================================
# G13 - no prohibited claim
# =====================================================================

def test_g13_no_prohibited_claims(config_raw, markdown_text, seed_manifest,
                                  state_manifest, cfg):
    sources = [
        (CONFIG_REL, config_raw),
        (MARKDOWN_REL, markdown_text),
        (SEED_MANIFEST_REL, json.dumps(seed_manifest, sort_keys=True)),
        (STATE_MANIFEST_REL, json.dumps(state_manifest, sort_keys=True)),
    ]
    hits = []
    for name, text in sources:
        lowered = text.lower()
        for claim_key, pattern in PROHIBITED_CLAIM_PATTERNS:
            match = re.search(pattern, lowered)
            if match:
                hits.append((name, claim_key, match.group(0)))
    assert hits == [], "prohibited claim found: {0}".format(hits)

    declared = dig(cfg, "prohibited_claims", {})
    for key in EXPECTED_PROHIBITED_CLAIM_KEYS:
        assert declared.get(key) == "PROHIBITED"


# =====================================================================
# G14 - CONTROLLED_BEST_EFFORT and the cross-environment limitation
# =====================================================================

def test_g14_controlled_best_effort_and_gpu_limitation(cfg, markdown_text):
    assert dig(cfg, "training_seed.deterministic_policy") == \
        EXPECTED_DETERMINISTIC_POLICY
    assert dig(cfg, "reproducibility.deterministic_policy") == \
        EXPECTED_DETERMINISTIC_POLICY
    assert dig(cfg, "reproducibility.runtime_settings_must_be_recorded") is True
    assert dig(cfg,
               "reproducibility.policy_verified_by_runtime_in_phase2F1") \
        is False

    limitation = norm_text(
        dig(cfg, "reproducibility.limitation_statement")).lower()
    for term in REQUIRED_GPU_LIMITATION_TERMS:
        assert term in limitation, "missing '{0}' in limitation".format(term)

    for flag in ("cross_gpu_identical_output_asserted",
                 "cross_cuda_version_identical_output_asserted",
                 "cross_cudnn_version_identical_output_asserted",
                 "cross_driver_version_identical_output_asserted",
                 "cross_hardware_identical_output_asserted",
                 "cross_software_version_identical_output_asserted"):
        assert dig(cfg, "reproducibility.{0}".format(flag)) is False

    assert EXPECTED_DETERMINISTIC_POLICY in markdown_text


# =====================================================================
# G15 - inherited Phase 2F values and checksums
# =====================================================================

def test_g15_phase2f_inherited_values_and_checksums(cfg, seed_manifest,
                                                    markdown_text):
    inherited = dig(cfg, "inherited", {})
    assert inherited.get("protocol_identity") == EXPECTED_PROTOCOL_IDENTITY
    assert inherited.get("protocol_version") == EXPECTED_PROTOCOL_VERSION
    assert inherited.get("train_size") == EXPECTED_TRAIN_SIZE
    assert list(inherited.get("labeled_budgets")) == EXPECTED_LABELED_BUDGETS
    assert list(inherited.get("labeled_sizes")) == EXPECTED_LABELED_SIZES
    assert list(inherited.get("unlabeled_sizes")) == EXPECTED_UNLABELED_SIZES
    assert dict(inherited.get("labeled_membership_checksums")) == \
        EXPECTED_MEMBERSHIP_CHECKSUMS

    for labeled, unlabeled in zip(EXPECTED_LABELED_SIZES,
                                  EXPECTED_UNLABELED_SIZES):
        assert labeled + unlabeled == EXPECTED_TRAIN_SIZE

    assert dict(dig(seed_manifest,
                    "inherited.labeled_membership_checksums")) == \
        EXPECTED_MEMBERSHIP_CHECKSUMS
    for checksum in EXPECTED_MEMBERSHIP_CHECKSUMS.values():
        assert checksum in markdown_text


# =====================================================================
# G16 - training_authorized is false
# =====================================================================

def test_g16_training_not_authorized(cfg, seed_manifest, state_manifest,
                                     markdown_text):
    assert dig(cfg, "authorization.training_authorized") is False
    assert dig(seed_manifest, "authorization.training_authorized") is False
    assert state_manifest["training_authorized"] is False
    assert "training_authorized" in markdown_text


# =====================================================================
# G17 - training_started is false
# =====================================================================

def test_g17_training_not_started(cfg, seed_manifest, state_manifest,
                                  markdown_text):
    assert dig(cfg, "authorization.training_started") is False
    assert dig(seed_manifest, "authorization.training_started") is False
    assert state_manifest["training_started"] is False
    assert "training_started" in markdown_text


# =====================================================================
# G18 - state manifest template with runs == []
# =====================================================================

def test_g18_state_manifest_is_empty_template(state_manifest):
    assert state_manifest["state"] == STATE_TEMPLATE_VALUE
    assert state_manifest["training_started"] is False
    assert isinstance(state_manifest["runs"], list)
    assert state_manifest["runs"] == []
    assert state_manifest["rng_state_captured"] is False
    assert list(state_manifest["required_run_fields"]) == EXPECTED_RUN_FIELDS


# =====================================================================
# G19 - YAML, JSON and Markdown are mutually consistent
# =====================================================================

def test_g19_artifacts_are_mutually_consistent(cfg, seed_manifest,
                                               state_manifest, markdown_text,
                                               validation_report):
    assert seed_manifest["source_config"] == CONFIG_REL
    assert state_manifest["source_config"] == CONFIG_REL
    assert seed_manifest["source_config_sha256"] == \
        state_manifest["source_config_sha256"]
    assert validation_report["source_config_sha256"] == \
        seed_manifest["source_config_sha256"]

    assert seed_manifest["phase"] == "2F.1"
    assert state_manifest["phase"] == "2F.1"
    assert validation_report["phase"] == "2F.1"

    assert seed_manifest["phase_closure_status"] == PHASE_CLOSURE_STATUS
    assert state_manifest["phase_closure_status"] == PHASE_CLOSURE_STATUS
    assert validation_report["phase_closure_status"] == PHASE_CLOSURE_STATUS
    assert dig(cfg, "document.phase_closure_status") == PHASE_CLOSURE_STATUS
    assert dig(cfg, "document.self_declared_closure_allowed") is False
    assert validation_report["validation_pass_implies_phase_closure"] is False
    assert validation_report["validation_status"] in ("PASS", "FAIL")

    assert state_manifest["protocol_identity"] == EXPECTED_PROTOCOL_IDENTITY
    assert EXPECTED_PROTOCOL_IDENTITY in markdown_text
    assert PHASE_CLOSURE_STATUS in markdown_text
    assert STATE_TEMPLATE_VALUE in markdown_text
    for value in EXPECTED_TRAINING_SEEDS:
        assert str(value) in markdown_text

    schema_fields = [
        entry.get("name")
        for entry in dig(cfg, "future_run_metadata_schema.required_fields", [])
    ]
    assert schema_fields == EXPECTED_RUN_FIELDS
    for entry in dig(cfg, "future_run_metadata_schema.required_fields", []):
        assert entry.get("required") is True
        assert norm_text(entry.get("description"))
    for field in EXPECTED_RUN_FIELDS:
        assert "`{0}`".format(field) in markdown_text

    for entry in validation_report.get("checks", []):
        for key in ("check_id", "status", "expected", "observed", "evidence"):
            assert key in entry, "missing '{0}' in {1}".format(
                key, entry.get("check_id"))
        assert entry["status"] in ("PASS", "FAIL")


# =====================================================================
# G20 - no fabricated training output, checkpoint or result
# =====================================================================

def test_g20_no_fabricated_training_output(cfg, seed_manifest, state_manifest):
    assert seed_manifest["runs"] == []
    assert seed_manifest["runs_present"] == 0
    assert seed_manifest["checkpoints_present"] == 0
    assert seed_manifest["results_present"] == 0
    assert seed_manifest["rng_state_captured"] is False
    assert state_manifest["checkpoints_present"] == 0
    assert state_manifest["results_present"] == 0
    assert dig(cfg,
               "future_run_metadata_schema.runs_created_in_phase2F1") == 0
    assert dig(cfg, "future_run_metadata_schema.applies_to") == \
        "FUTURE_RUNS_ONLY"
    assert "rng_state_id" in norm_text(
        dig(cfg, "future_run_metadata_schema.rng_state_id_note"))

    for excluded in ("training", "checkpointing", "model_evaluation",
                     "rng_state_capture_or_restore"):
        assert excluded in dig(cfg, "document.scope_excludes", [])