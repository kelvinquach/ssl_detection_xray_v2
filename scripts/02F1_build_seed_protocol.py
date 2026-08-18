#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Phase 2F.1 - Seed Protocol builder and validator.

Scope
-----
This script builds and validates the Phase 2F.1 seed CONTRACT only.

It never trains, never initialises a model, never touches a DataLoader,
never augments, never pseudo-labels, never writes a checkpoint, never
evaluates a model, never captures or restores RNG state, never reads test
data, and never modifies any Phase 2E / Phase 2F artifact or dataset
membership.

Commands
--------
    python scripts\\02F1_build_seed_protocol.py --execute
    python scripts\\02F1_build_seed_protocol.py --validate-existing

--execute
    Reads configs/protocol/phase2F1_seed_protocol.yaml, recomputes the
    training-seed list from the public SHA-256 rule, runs every required
    check, and - only if every required check passes - writes:
        data/manifests/seed_manifest.json
        data/manifests/seed_state_manifest.json
        reports/seed_protocol.md
        reports/02F1_seed_protocol_validation_report.json
    If any required check fails, ONLY the validation report is written (so
    that the failure is evidenced) and the exit code is non-zero. No
    manifest and no Markdown report is emitted from a failing state.

--validate-existing
    Read-only. Loads the four existing artifacts and re-runs every check.
    Nothing on disk is modified.

Exit codes
----------
    0  every required check passed
    1  at least one required check failed
    2  usage error / environment error / missing input

A validation_status of PASS does NOT mean Phase 2F.1 is closed. Closure is
decided only by the researcher together with GPT review.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

try:
    import yaml
except ImportError:  # pragma: no cover - environment guard
    sys.stderr.write(
        "[FATAL] PyYAML is required to read "
        "configs/protocol/phase2F1_seed_protocol.yaml\n"
    )
    sys.exit(2)


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
# Locked constants. Hard-coded independently of the YAML file so that the
# YAML file itself is verified against the protocol, not trusted blindly.
# =====================================================================

LOCKED_PROTOCOL_IDENTITY = "2F-C0-R11"
LOCKED_PROTOCOL_VERSION = "2.0.0"
LOCKED_TRAIN_SIZE = 3426
LOCKED_LABELED_BUDGETS = ["1pct", "5pct", "10pct", "20pct"]
LOCKED_LABELED_SIZES = [34, 171, 343, 685]
LOCKED_UNLABELED_SIZES = [3392, 3255, 3083, 2741]

LOCKED_MEMBERSHIP_CHECKSUMS = {
    "1pct": "c54e7d61e84b7cfce68c04a795783b5c5d01331d2aa9c754fb1ae1dbae4ba071",
    "5pct": "c4db3b5f7a5b0f391ad883ef665c3d341af3ef6afb3fc553e71f1c4ef17ee50b",
    "10pct": "fc008d31505227544087ba474613075a8cd077587df9d6b13146b378f5a3a7d6",
    "20pct": "6f4aaba6be147983d56007c49ada234dfcabe7ec2f936f28f8cd240999b8417e",
}

LOCKED_PARTITION_SEED = 42
LOCKED_PARTITION_SEED_POLICY = "PRE_SPECIFIED_LOCKED_NO_SEED_SEARCH"

LOCKED_TRAINING_SEED_COUNT = 10
LOCKED_DETERMINISTIC_POLICY = "CONTROLLED_BEST_EFFORT"
LOCKED_TRAINING_SEEDS = [
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

LOCKED_NAMESPACE = "ssl_detection_xray_v2|phase2F.1|training_seed"
SEED_MIN = 1
SEED_MAX = 2147483647  # 2**31 - 1
SEED_MODULUS = 2147483647
SEED_OFFSET = 1
DIGEST_PREFIX_HEX_CHARS = 8

REQUIRED_RUN_FIELDS = [
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

REQUIRED_PROHIBITED_CLAIM_KEYS = [
    "statistical_optimality_of_seed_count",
    "power_analysis_performed",
    "stability_proven",
    "variance_measured_in_phase2F1",
    "bitwise_reproducibility_across_environments",
]

# Affirmative claim patterns. They are deliberately anchored on affirmative
# verbs and on whitespace, so that snake_case tokens such as
# `power_analysis_performed` used inside negative declarations do not match.
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


# =====================================================================
# Small helpers
# =====================================================================

def dig(obj, dotted_path, default=None):
    """Safe nested lookup: dig(cfg, 'partition.partition_seed')."""
    current = obj
    for part in dotted_path.split("."):
        if isinstance(current, dict) and part in current:
            current = current[part]
        else:
            return default
    return current


def norm_text(value):
    """Collapse all whitespace so that YAML folding cannot break equality."""
    if value is None:
        return ""
    return " ".join(str(value).split())


def sha256_of_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def utc_now_iso():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def check(check_id, description, condition, expected, observed, evidence,
          required=True):
    return {
        "check_id": check_id,
        "description": description,
        "required": bool(required),
        "status": "PASS" if condition else "FAIL",
        "expected": expected,
        "observed": observed,
        "evidence": evidence,
    }


# =====================================================================
# Seed derivation (public rule, no RNG call)
# =====================================================================

def derive_training_seeds(namespace=LOCKED_NAMESPACE,
                          count=LOCKED_TRAINING_SEED_COUNT,
                          prefix_chars=DIGEST_PREFIX_HEX_CHARS,
                          modulus=SEED_MODULUS,
                          offset=SEED_OFFSET):
    """Recompute the ordered training-seed list from the public rule.

    payload_i  = namespace + '|index=' + str(i)
    digest_i   = SHA256(UTF-8(payload_i))
    seed_i     = offset + (int(digest_i[:prefix_chars], 16) mod modulus)

    No random number generator is used anywhere in this function.
    """
    records = []
    for index in range(1, count + 1):
        payload = "{0}|index={1}".format(namespace, index)
        digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
        prefix = digest[:prefix_chars]
        value = offset + (int(prefix, 16) % modulus)
        records.append({
            "index": index,
            "training_seed": value,
            "payload": payload,
            "sha256_digest": digest,
            "digest_prefix_hex": prefix,
            "integer_of_prefix": int(prefix, 16),
        })
    return records


# =====================================================================
# Loading
# =====================================================================

def load_config():
    if not CONFIG_PATH.is_file():
        raise SystemExit(
            "[FATAL] missing contract file: {0}".format(CONFIG_REL))
    raw = CONFIG_PATH.read_text(encoding="utf-8")
    cfg = yaml.safe_load(raw)
    if not isinstance(cfg, dict):
        raise SystemExit(
            "[FATAL] {0} did not parse into a mapping".format(CONFIG_REL))
    return cfg, raw


def load_json(path, rel):
    if not path.is_file():
        return None, "missing artifact: {0}".format(rel)
    try:
        return json.loads(path.read_text(encoding="utf-8")), None
    except ValueError as exc:
        return None, "invalid JSON in {0}: {1}".format(rel, exc)


# =====================================================================
# Artifact builders
# =====================================================================

def build_seed_manifest(cfg, derived, config_sha256, timestamp):
    return {
        "manifest_id": "phase2F1_seed_manifest",
        "schema_version": "1.0.0",
        "phase": "2F.1",
        "manifest_status": "CANDIDATE_PENDING_RESEARCHER_GPT_REVIEW",
        "phase_closure_status": PHASE_CLOSURE_STATUS,
        "generated_at_utc": timestamp,
        "generated_by": "scripts/02F1_build_seed_protocol.py",
        "source_config": CONFIG_REL,
        "source_config_sha256": config_sha256,
        "inherited": dig(cfg, "inherited", {}),
        "partition": dig(cfg, "partition", {}),
        "training_seed": {
            "training_seed_count": dig(cfg, "training_seed.training_seed_count"),
            "deterministic_policy": dig(
                cfg, "training_seed.deterministic_policy"),
            "ordering": dig(cfg, "training_seed.ordering"),
            "seed_domain": dig(cfg, "training_seed.seed_domain", {}),
            "generation": dig(cfg, "training_seed.generation", {}),
            "selection_constraints": dig(
                cfg, "training_seed.selection_constraints", {}),
            "ordered_training_seed_values": [
                record["training_seed"] for record in derived],
            "seeds": derived,
            "derivation_recomputed": True,
            "derivation_matches_locked_list": True,
            "rng_calls_used": 0,
        },
        "pairing_policy": dig(cfg, "pairing_policy", {}),
        "retry_policy": dig(cfg, "retry_policy", {}),
        "aggregation_policy": dig(cfg, "aggregation_policy", {}),
        "reproducibility": dig(cfg, "reproducibility", {}),
        "future_run_metadata_schema": dig(cfg, "future_run_metadata_schema", {}),
        "rationale": dig(cfg, "rationale", {}),
        "prohibited_claims": dig(cfg, "prohibited_claims", {}),
        "authorization": dig(cfg, "authorization", {}),
        "runs": [],
        "runs_present": 0,
        "checkpoints_present": 0,
        "results_present": 0,
        "rng_state_captured": False,
    }


def build_state_manifest(cfg, derived, config_sha256, timestamp):
    return {
        "manifest_id": "phase2F1_seed_state_manifest",
        "schema_version": "1.0.0",
        "phase": "2F.1",
        "state": STATE_TEMPLATE_VALUE,
        "training_started": False,
        "training_authorized": False,
        "runs": [],
        "generated_at_utc": timestamp,
        "generated_by": "scripts/02F1_build_seed_protocol.py",
        "source_config": CONFIG_REL,
        "source_config_sha256": config_sha256,
        "protocol_identity": dig(cfg, "inherited.protocol_identity"),
        "protocol_version": dig(cfg, "inherited.protocol_version"),
        "partition_seed": dig(cfg, "partition.partition_seed"),
        "training_seed_count": dig(cfg, "training_seed.training_seed_count"),
        "ordered_training_seed_values": [
            record["training_seed"] for record in derived],
        "required_run_fields": REQUIRED_RUN_FIELDS,
        "rng_state_captured": False,
        "checkpoints_present": 0,
        "results_present": 0,
        "phase_closure_status": PHASE_CLOSURE_STATUS,
        "note": (
            "Template only. No run, no RNG state, no checkpoint and no result "
            "exists in Phase 2F.1. Runs may be appended only after the "
            "researcher and GPT review authorise training."
        ),
    }


def build_markdown(cfg, derived, config_sha256, timestamp):
    seeds_table = []
    for record in derived:
        seeds_table.append(
            "| {0} | {1} | `{2}` | `{3}` |".format(
                record["index"],
                record["training_seed"],
                record["payload"],
                record["digest_prefix_hex"],
            )
        )

    checksums = dig(cfg, "inherited.labeled_membership_checksums", {})
    checksum_rows = [
        "| {0} | `{1}` |".format(budget, checksums.get(budget, "MISSING"))
        for budget in LOCKED_LABELED_BUDGETS
    ]

    field_rows = []
    for field in dig(cfg, "future_run_metadata_schema.required_fields", []):
        field_rows.append(
            "| `{0}` | {1} | {2} |".format(
                field.get("name"),
                "yes" if field.get("required") else "no",
                field.get("description", ""),
            )
        )

    prohibited_rows = [
        "- `{0}`".format(key) for key in sorted(
            dig(cfg, "prohibited_claims", {}).keys())
    ]

    lines = []
    lines.append("# Phase 2F.1 - Seed Protocol")
    lines.append("")
    lines.append("| field | value |")
    lines.append("| --- | --- |")
    lines.append("| document_status | `{0}` |".format(
        dig(cfg, "document.document_status")))
    lines.append("| phase_closure_status | `{0}` |".format(
        PHASE_CLOSURE_STATUS))
    lines.append("| approval_authority | `{0}` |".format(
        dig(cfg, "document.approval_authority")))
    lines.append("| source_config | `{0}` |".format(CONFIG_REL))
    lines.append("| source_config_sha256 | `{0}` |".format(config_sha256))
    lines.append("| generated_at_utc | `{0}` |".format(timestamp))
    lines.append("| training_authorized | `false` |")
    lines.append("| training_started | `false` |")
    lines.append("")
    lines.append(
        "This document is a CANDIDATE contract. A `validation_status` of "
        "`PASS` in `{0}` does not change `phase_closure_status`, which "
        "remains `{1}` until the researcher and GPT review decide.".format(
            VALIDATION_REPORT_REL, PHASE_CLOSURE_STATUS))
    lines.append("")

    lines.append("## 1. Scope")
    lines.append("")
    lines.append(
        "Phase 2F.1 builds the seed contract that Phase 4-5 inherits. "
        "It excludes:")
    lines.append("")
    for item in dig(cfg, "document.scope_excludes", []):
        lines.append("- `{0}`".format(item))
    lines.append("")

    lines.append("## 2. Inherited from Phase 2F (immutable here)")
    lines.append("")
    lines.append("- `protocol_identity` = `{0}`".format(
        dig(cfg, "inherited.protocol_identity")))
    lines.append("- `protocol_version` = `{0}`".format(
        dig(cfg, "inherited.protocol_version")))
    lines.append("- `train_size` = {0}".format(dig(cfg, "inherited.train_size")))
    lines.append("- `labeled_budgets` = {0}".format(
        dig(cfg, "inherited.labeled_budgets")))
    lines.append("- `labeled_sizes` = {0}".format(
        dig(cfg, "inherited.labeled_sizes")))
    lines.append("- `unlabeled_sizes` = {0}".format(
        dig(cfg, "inherited.unlabeled_sizes")))
    lines.append("")
    lines.append("| budget | labeled membership SHA-256 |")
    lines.append("| --- | --- |")
    lines.extend(checksum_rows)
    lines.append("")

    lines.append("## 3. Partition seed")
    lines.append("")
    lines.append("- `partition_seed` = {0}".format(
        dig(cfg, "partition.partition_seed")))
    lines.append("- `partition_seed_policy` = `{0}`".format(
        dig(cfg, "partition.partition_seed_policy")))
    lines.append("- `{0}`".format(
        dig(cfg, "partition.terminology.legacy_alias_note")))
    lines.append("- `seed_search_performed` = false")
    lines.append("- `partition_seed_equals_training_seed` = false")
    lines.append(
        "- The partition seed already fixed the train / validation / test "
        "split and the labeled / unlabeled membership. Phase 2F.1 does not "
        "regenerate them.")
    lines.append("")

    lines.append("## 4. Training seeds")
    lines.append("")
    lines.append("- `training_seed_count` = {0}".format(
        dig(cfg, "training_seed.training_seed_count")))
    lines.append("- `deterministic_policy` = `{0}`".format(
        dig(cfg, "training_seed.deterministic_policy")))
    lines.append("- `seed_domain` = [{0}, {1}]".format(SEED_MIN, SEED_MAX))
    lines.append("- `rng_calls_used` = 0")
    lines.append("")
    lines.append("Generation rule (public, recomputed at every run of this "
                 "script):")
    lines.append("")
    lines.append("```")
    lines.append("namespace = {0}".format(LOCKED_NAMESPACE))
    lines.append("payload_i = namespace + |index=i")
    lines.append("digest_i  = SHA256(UTF-8(payload_i))")
    lines.append("seed_i    = 1 + (integer(first 8 hexadecimal characters of "
                 "digest_i) mod (2^31 - 1))")
    lines.append("i = 1,...,10")
    lines.append("```")
    lines.append("")
    lines.append("| index | training_seed | payload | digest prefix |")
    lines.append("| --- | --- | --- | --- |")
    lines.extend(seeds_table)
    lines.append("")

    lines.append("## 5. Pairing policy")
    lines.append("")
    lines.append("- {0}".format(dig(cfg, "pairing_policy.rule")))
    lines.append("- `supervised_and_ssl_share_same_ordered_seed_list` = true")
    lines.append("- `per_method_seed_lists_allowed` = false")
    lines.append("- `reshuffling_allowed` = false")
    lines.append("- `comparison_unit` = `{0}`".format(
        dig(cfg, "pairing_policy.comparison_unit")))
    lines.append("")

    lines.append("## 6. Retry policy")
    lines.append("")
    lines.append("- `retry_allowed_reason` = `{0}`".format(
        dig(cfg, "retry_policy.retry_allowed_reason")))
    lines.append("- `retry_must_reuse_same_training_seed` = true")
    lines.append("- `retry_changes_training_seed` = false")
    lines.append("- `silent_replacement_allowed` = false")
    lines.append("- `failed_run_must_be_retained` = true, with "
                 "`run_status` = `TECHNICAL_FAILURE` and a non-null "
                 "`technical_failure_reason`")
    lines.append("- the retry run must set `retry_of` to the `run_id` of the "
                 "original attempt")
    lines.append("- `result_based_retry_allowed` = false")
    lines.append("")

    lines.append("## 7. Aggregation policy")
    lines.append("")
    lines.append("- per-seed results are reported for every one of the "
                 "{0} seeds".format(LOCKED_TRAINING_SEED_COUNT))
    lines.append("- the mean across seeds is reported")
    lines.append("- the sample standard deviation is reported with `ddof=1`")
    lines.append("- `{0}`".format(dig(cfg, "aggregation_policy.formula_mean")))
    lines.append("- `{0}`".format(dig(cfg, "aggregation_policy.formula_sd")))
    lines.append("- `seed_dropping_allowed` = false, "
                 "`outlier_removal_allowed` = false, "
                 "`best_seed_only_reporting_allowed` = false")
    lines.append("")

    lines.append("## 8. Reproducibility policy")
    lines.append("")
    lines.append("- `deterministic_policy` = `{0}`".format(
        LOCKED_DETERMINISTIC_POLICY))
    lines.append("- controllable sources to be seeded in Phase 4-5: {0}".format(
        ", ".join("`{0}`".format(source) for source in dig(
            cfg,
            "reproducibility.controllable_sources_to_be_seeded_in_phase_4_5",
            []))))
    lines.append("- `runtime_settings_must_be_recorded` = true")
    lines.append("- limitation: {0}".format(
        dig(cfg, "reproducibility.limitation_statement")))
    lines.append("- `policy_verified_by_runtime_in_phase2F1` = false")
    lines.append("")

    lines.append("## 9. Rationale")
    lines.append("")
    lines.append("### 9.1 `training_seed_count_rationale`")
    lines.append("")
    lines.append(norm_text(dig(cfg, "rationale.training_seed_count_rationale")))
    lines.append("")
    lines.append("### 9.2 `controlled_best_effort_rationale`")
    lines.append("")
    lines.append(norm_text(
        dig(cfg, "rationale.controlled_best_effort_rationale")))
    lines.append("")

    lines.append("## 10. Claims explicitly not made in Phase 2F.1")
    lines.append("")
    lines.extend(prohibited_rows)
    lines.append("")

    lines.append("## 11. Mandatory metadata schema for every future run")
    lines.append("")
    lines.append("`runs_created_in_phase2F1` = 0. The schema below constrains "
                 "future runs only.")
    lines.append("")
    lines.append("| field | required | description |")
    lines.append("| --- | --- | --- |")
    lines.extend(field_rows)
    lines.append("")
    lines.append(dig(cfg, "future_run_metadata_schema.rng_state_id_note"))
    lines.append("")

    lines.append("## 12. Authorization")
    lines.append("")
    lines.append("- `training_authorized` = false")
    lines.append("- `training_started` = false")
    lines.append("- `state` of `{0}` = `{1}` with `runs = []`".format(
        STATE_MANIFEST_REL, STATE_TEMPLATE_VALUE))
    lines.append("- unlock condition: {0}".format(
        dig(cfg, "authorization.unlock_condition")))
    lines.append("")

    lines.append("## 13. Commands")
    lines.append("")
    lines.append("```")
    lines.append("python scripts\\02F1_build_seed_protocol.py --execute")
    lines.append("python -m pytest tests\\test_phase2F1_seed_protocol_guardrails.py "
                 "-v --junitxml=reports\\02F1_guardrails_junit.xml")
    lines.append("python scripts\\02F1_build_seed_protocol.py --validate-existing")
    lines.append("```")
    lines.append("")

    return "\n".join(lines) + "\n"


# =====================================================================
# Checks
# =====================================================================

def scan_prohibited_claims(texts):
    """Return the list of affirmative prohibited-claim matches found."""
    hits = []
    for source_name, text in texts:
        lowered = text.lower()
        for claim_key, pattern in PROHIBITED_CLAIM_PATTERNS:
            match = re.search(pattern, lowered)
            if match:
                hits.append({
                    "source": source_name,
                    "claim": claim_key,
                    "pattern": pattern,
                    "matched_text": match.group(0),
                })
    return hits


def run_checks(cfg, config_raw, derived, seed_manifest, state_manifest,
               markdown_text, mode, existing_validation_report=None):
    checks = []

    # ---- C01 partition seed value -----------------------------------
    observed = dig(cfg, "partition.partition_seed")
    checks.append(check(
        "C01", "partition_seed is the locked value 42",
        observed == LOCKED_PARTITION_SEED,
        LOCKED_PARTITION_SEED, observed,
        "{0}::partition.partition_seed".format(CONFIG_REL)))

    # ---- C02 partition seed policy ----------------------------------
    observed = dig(cfg, "partition.partition_seed_policy")
    checks.append(check(
        "C02", "partition_seed_policy matches exactly",
        observed == LOCKED_PARTITION_SEED_POLICY,
        LOCKED_PARTITION_SEED_POLICY, observed,
        "{0}::partition.partition_seed_policy".format(CONFIG_REL)))

    # ---- C03 training seed count ------------------------------------
    count_cfg = dig(cfg, "training_seed.training_seed_count")
    values_cfg = dig(cfg, "training_seed.ordered_training_seed_values", [])
    pairs_cfg = dig(cfg, "training_seed.ordered_training_seeds", [])
    condition = (
        count_cfg == LOCKED_TRAINING_SEED_COUNT
        and len(values_cfg) == LOCKED_TRAINING_SEED_COUNT
        and len(pairs_cfg) == LOCKED_TRAINING_SEED_COUNT
        and len(derived) == LOCKED_TRAINING_SEED_COUNT
    )
    checks.append(check(
        "C03", "exactly 10 training seeds are declared everywhere",
        condition,
        {"training_seed_count": LOCKED_TRAINING_SEED_COUNT,
         "list_lengths": LOCKED_TRAINING_SEED_COUNT},
        {"training_seed_count": count_cfg,
         "ordered_training_seed_values": len(values_cfg),
         "ordered_training_seeds": len(pairs_cfg),
         "derived": len(derived)},
        "{0}::training_seed".format(CONFIG_REL)))

    # ---- C04 ordered list exact match --------------------------------
    pair_values = [entry.get("training_seed") for entry in pairs_cfg]
    pair_indices = [entry.get("index") for entry in pairs_cfg]
    condition = (
        list(values_cfg) == LOCKED_TRAINING_SEEDS
        and pair_values == LOCKED_TRAINING_SEEDS
        and pair_indices == list(range(1, LOCKED_TRAINING_SEED_COUNT + 1))
    )
    checks.append(check(
        "C04", "ordered training-seed list matches the locked list exactly",
        condition,
        LOCKED_TRAINING_SEEDS,
        {"ordered_training_seed_values": list(values_cfg),
         "ordered_training_seeds_values": pair_values,
         "indices": pair_indices},
        "{0}::training_seed.ordered_training_seeds".format(CONFIG_REL)))

    # ---- C05 uniqueness and domain -----------------------------------
    seed_values = [record["training_seed"] for record in derived]
    unique = len(set(seed_values)) == len(seed_values)
    in_domain = all(SEED_MIN <= value <= SEED_MAX for value in seed_values)
    domain_cfg = dig(cfg, "training_seed.seed_domain", {})
    domain_declared = (domain_cfg.get("minimum") == SEED_MIN
                       and domain_cfg.get("maximum") == SEED_MAX)
    checks.append(check(
        "C05", "training seeds are unique and inside [1, 2**31 - 1]",
        unique and in_domain and domain_declared,
        {"unique": True, "domain": [SEED_MIN, SEED_MAX]},
        {"unique": unique, "in_domain": in_domain,
         "declared_domain": [domain_cfg.get("minimum"),
                             domain_cfg.get("maximum")],
         "duplicates": sorted(
             value for value in set(seed_values)
             if seed_values.count(value) > 1)},
        "recomputed in scripts/02F1_build_seed_protocol.py"
        "::derive_training_seeds"))

    # ---- C06 SHA-256 rule reproduces the locked list ------------------
    condition = seed_values == LOCKED_TRAINING_SEEDS
    checks.append(check(
        "C06", "the public SHA-256 rule reproduces all 10 locked seeds",
        condition,
        LOCKED_TRAINING_SEEDS,
        seed_values,
        {"namespace": LOCKED_NAMESPACE,
         "payload_template": "{namespace}|index={index}",
         "digest_prefix_hex_chars": DIGEST_PREFIX_HEX_CHARS,
         "modulus": SEED_MODULUS,
         "offset": SEED_OFFSET,
         "per_seed": [
             {"index": record["index"],
              "payload": record["payload"],
              "digest_prefix_hex": record["digest_prefix_hex"],
              "training_seed": record["training_seed"]}
             for record in derived]}))

    # ---- C07 no seed search ------------------------------------------
    flags = {
        "partition.constraints.seed_search_performed":
            dig(cfg, "partition.constraints.seed_search_performed"),
        "training_seed.selection_constraints.seed_search_performed":
            dig(cfg, "training_seed.selection_constraints."
                     "seed_search_performed"),
        "training_seed.generation.rng_calls_used":
            dig(cfg, "training_seed.generation.rng_calls_used"),
        "training_seed.generation.method":
            dig(cfg, "training_seed.generation.method"),
    }
    condition = (
        flags["partition.constraints.seed_search_performed"] is False
        and flags["training_seed.selection_constraints.seed_search_performed"]
        is False
        and flags["training_seed.generation.rng_calls_used"] == 0
        and flags["training_seed.generation.method"]
        == "SHA256_PUBLIC_DERIVATION_NO_RNG"
    )
    checks.append(check(
        "C07", "no seed search: seeds are derived by a public rule, no RNG",
        condition,
        {"seed_search_performed": False, "rng_calls_used": 0,
         "method": "SHA256_PUBLIC_DERIVATION_NO_RNG"},
        flags,
        "{0}::partition.constraints, training_seed.generation".format(
            CONFIG_REL)))

    # ---- C08 validation / test never used to select seeds -------------
    flags = {
        "partition.constraints.validation_data_used_for_seed_selection":
            dig(cfg, "partition.constraints."
                     "validation_data_used_for_seed_selection"),
        "partition.constraints.test_data_used_for_seed_selection":
            dig(cfg, "partition.constraints."
                     "test_data_used_for_seed_selection"),
        "training_seed.selection_constraints."
        "seeds_selected_using_validation_data":
            dig(cfg, "training_seed.selection_constraints."
                     "seeds_selected_using_validation_data"),
        "training_seed.selection_constraints.seeds_selected_using_test_data":
            dig(cfg, "training_seed.selection_constraints."
                     "seeds_selected_using_test_data"),
        "training_seed.selection_constraints.seeds_selected_using_results":
            dig(cfg, "training_seed.selection_constraints."
                     "seeds_selected_using_results"),
    }
    checks.append(check(
        "C08", "validation and test data are not used to select any seed",
        all(value is False for value in flags.values()),
        {key: False for key in flags},
        flags,
        "{0}::partition.constraints, "
        "training_seed.selection_constraints".format(CONFIG_REL)))

    # ---- C09 supervised and SSL share the ordered seed list -----------
    flags = {
        "supervised_and_ssl_share_same_ordered_seed_list":
            dig(cfg, "pairing_policy."
                     "supervised_and_ssl_share_same_ordered_seed_list"),
        "pairing_key": dig(cfg, "pairing_policy.pairing_key"),
        "per_method_seed_lists_allowed":
            dig(cfg, "pairing_policy.per_method_seed_lists_allowed"),
        "reshuffling_allowed": dig(cfg, "pairing_policy.reshuffling_allowed"),
        "subsetting_allowed": dig(cfg, "pairing_policy.subsetting_allowed"),
        "partition_seed_shared_across_all_runs":
            dig(cfg, "pairing_policy.partition_seed_shared_across_all_runs"),
        "budgets_covered": dig(cfg, "pairing_policy.budgets_covered"),
    }
    condition = (
        flags["supervised_and_ssl_share_same_ordered_seed_list"] is True
        and flags["pairing_key"] == "training_seed_index"
        and flags["per_method_seed_lists_allowed"] is False
        and flags["reshuffling_allowed"] is False
        and flags["subsetting_allowed"] is False
        and flags["partition_seed_shared_across_all_runs"] is True
        and list(flags["budgets_covered"] or []) == LOCKED_LABELED_BUDGETS
    )
    checks.append(check(
        "C09", "supervised and SSL use the same ordered seed list",
        condition,
        {"shared": True, "pairing_key": "training_seed_index",
         "per_method_seed_lists_allowed": False,
         "budgets_covered": LOCKED_LABELED_BUDGETS},
        flags,
        "{0}::pairing_policy".format(CONFIG_REL)))

    # ---- C10 retry policy --------------------------------------------
    flags = {
        "retry_allowed_reason": dig(cfg, "retry_policy.retry_allowed_reason"),
        "retry_changes_training_seed":
            dig(cfg, "retry_policy.retry_changes_training_seed"),
        "retry_must_reuse_same_training_seed":
            dig(cfg, "retry_policy.retry_must_reuse_same_training_seed"),
        "retry_must_reuse_same_training_seed_index":
            dig(cfg, "retry_policy.retry_must_reuse_same_training_seed_index"),
        "silent_replacement_allowed":
            dig(cfg, "retry_policy.silent_replacement_allowed"),
        "result_based_retry_allowed":
            dig(cfg, "retry_policy.result_based_retry_allowed"),
        "failed_run_must_be_retained":
            dig(cfg, "retry_policy.failed_run_must_be_retained"),
        "technical_failure_reason_required":
            dig(cfg, "retry_policy.technical_failure_reason_required"),
        "retry_run_must_reference_original":
            dig(cfg, "retry_policy.retry_run_must_reference_original"),
        "retry_reference_field":
            dig(cfg, "retry_policy.retry_reference_field"),
        "all_attempts_must_be_recorded":
            dig(cfg, "retry_policy.all_attempts_must_be_recorded"),
    }
    condition = (
        flags["retry_allowed_reason"] == "TECHNICAL_FAILURE_ONLY"
        and flags["retry_changes_training_seed"] is False
        and flags["retry_must_reuse_same_training_seed"] is True
        and flags["retry_must_reuse_same_training_seed_index"] is True
        and flags["silent_replacement_allowed"] is False
        and flags["result_based_retry_allowed"] is False
        and flags["failed_run_must_be_retained"] is True
        and flags["technical_failure_reason_required"] is True
        and flags["retry_run_must_reference_original"] is True
        and flags["retry_reference_field"] == "retry_of"
        and flags["all_attempts_must_be_recorded"] is True
    )
    checks.append(check(
        "C10", "retry keeps the same seed and cannot silently replace a run",
        condition,
        {"retry_changes_training_seed": False,
         "silent_replacement_allowed": False,
         "result_based_retry_allowed": False,
         "failed_run_must_be_retained": True,
         "retry_reference_field": "retry_of"},
        flags,
        "{0}::retry_policy".format(CONFIG_REL)))

    # ---- C11 aggregation rule ----------------------------------------
    flags = {
        "report_per_seed_results":
            dig(cfg, "aggregation_policy.report_per_seed_results"),
        "report_mean_across_seeds":
            dig(cfg, "aggregation_policy.report_mean_across_seeds"),
        "report_sample_standard_deviation":
            dig(cfg, "aggregation_policy.report_sample_standard_deviation"),
        "standard_deviation_ddof":
            dig(cfg, "aggregation_policy.standard_deviation_ddof"),
        "standard_deviation_type":
            dig(cfg, "aggregation_policy.standard_deviation_type"),
        "n_for_aggregation": dig(cfg, "aggregation_policy.n_for_aggregation"),
        "seed_dropping_allowed":
            dig(cfg, "aggregation_policy.seed_dropping_allowed"),
        "outlier_removal_allowed":
            dig(cfg, "aggregation_policy.outlier_removal_allowed"),
        "best_seed_only_reporting_allowed":
            dig(cfg, "aggregation_policy.best_seed_only_reporting_allowed"),
    }
    condition = (
        flags["report_per_seed_results"] is True
        and flags["report_mean_across_seeds"] is True
        and flags["report_sample_standard_deviation"] is True
        and flags["standard_deviation_ddof"] == 1
        and flags["standard_deviation_type"] == "SAMPLE"
        and flags["n_for_aggregation"] == LOCKED_TRAINING_SEED_COUNT
        and flags["seed_dropping_allowed"] is False
        and flags["outlier_removal_allowed"] is False
        and flags["best_seed_only_reporting_allowed"] is False
    )
    checks.append(check(
        "C11", "per-seed, mean and sample SD with ddof=1 are mandated",
        condition,
        {"per_seed": True, "mean": True, "sample_sd": True, "ddof": 1,
         "n": LOCKED_TRAINING_SEED_COUNT},
        flags,
        "{0}::aggregation_policy".format(CONFIG_REL)))

    # ---- C12 both rationales present ---------------------------------
    rationale_a = norm_text(dig(cfg, "rationale.training_seed_count_rationale"))
    rationale_b = norm_text(
        dig(cfg, "rationale.controlled_best_effort_rationale"))
    condition = len(rationale_a) >= 120 and len(rationale_b) >= 120
    checks.append(check(
        "C12", "both mandatory rationales exist and are non-trivial",
        condition,
        {"training_seed_count_rationale": ">=120 chars",
         "controlled_best_effort_rationale": ">=120 chars"},
        {"training_seed_count_rationale_len": len(rationale_a),
         "controlled_best_effort_rationale_len": len(rationale_b)},
        "{0}::rationale".format(CONFIG_REL)))

    # ---- C13 no prohibited claims ------------------------------------
    scan_sources = [
        (CONFIG_REL, config_raw),
        (MARKDOWN_REL, markdown_text or ""),
        (SEED_MANIFEST_REL, json.dumps(seed_manifest or {}, sort_keys=True)),
        (STATE_MANIFEST_REL, json.dumps(state_manifest or {}, sort_keys=True)),
    ]
    hits = scan_prohibited_claims(scan_sources)
    declared = dig(cfg, "prohibited_claims", {}) or {}
    declared_ok = all(
        declared.get(key) == "PROHIBITED"
        for key in REQUIRED_PROHIBITED_CLAIM_KEYS)
    checks.append(check(
        "C13", "no statistical-optimality, power-analysis, proven-stability "
               "or guaranteed-reproducibility claim appears",
        not hits and declared_ok,
        {"affirmative_claim_matches": 0,
         "declared_prohibited_claims": REQUIRED_PROHIBITED_CLAIM_KEYS},
        {"affirmative_claim_matches": hits,
         "declared_prohibited_claims": declared},
        "regex scan of {0}".format(
            ", ".join(name for name, _ in scan_sources))))

    # ---- C14 CONTROLLED_BEST_EFFORT and GPU limitation ---------------
    limitation = norm_text(dig(cfg, "reproducibility.limitation_statement"))
    limitation_lower = limitation.lower()
    missing_terms = [term for term in REQUIRED_GPU_LIMITATION_TERMS
                     if term not in limitation_lower]
    policy_values = {
        "training_seed.deterministic_policy":
            dig(cfg, "training_seed.deterministic_policy"),
        "reproducibility.deterministic_policy":
            dig(cfg, "reproducibility.deterministic_policy"),
        "policy_verified_by_runtime_in_phase2F1":
            dig(cfg, "reproducibility.policy_verified_by_runtime_in_phase2F1"),
    }
    condition = (
        policy_values["training_seed.deterministic_policy"]
        == LOCKED_DETERMINISTIC_POLICY
        and policy_values["reproducibility.deterministic_policy"]
        == LOCKED_DETERMINISTIC_POLICY
        and policy_values["policy_verified_by_runtime_in_phase2F1"] is False
        and not missing_terms
    )
    checks.append(check(
        "C14", "CONTROLLED_BEST_EFFORT and the cross-environment limitation "
               "are documented",
        condition,
        {"deterministic_policy": LOCKED_DETERMINISTIC_POLICY,
         "limitation_terms": REQUIRED_GPU_LIMITATION_TERMS,
         "policy_verified_by_runtime_in_phase2F1": False},
        {"policy_values": policy_values,
         "limitation_statement": limitation,
         "missing_terms": missing_terms},
        "{0}::reproducibility".format(CONFIG_REL)))

    # ---- C15 inherited Phase 2F values and checksums ------------------
    inherited_observed = {
        "protocol_identity": dig(cfg, "inherited.protocol_identity"),
        "protocol_version": dig(cfg, "inherited.protocol_version"),
        "train_size": dig(cfg, "inherited.train_size"),
        "labeled_budgets": list(dig(cfg, "inherited.labeled_budgets", []) or []),
        "labeled_sizes": list(dig(cfg, "inherited.labeled_sizes", []) or []),
        "unlabeled_sizes": list(
            dig(cfg, "inherited.unlabeled_sizes", []) or []),
        "labeled_membership_checksums": dict(
            dig(cfg, "inherited.labeled_membership_checksums", {}) or {}),
    }
    inherited_expected = {
        "protocol_identity": LOCKED_PROTOCOL_IDENTITY,
        "protocol_version": LOCKED_PROTOCOL_VERSION,
        "train_size": LOCKED_TRAIN_SIZE,
        "labeled_budgets": LOCKED_LABELED_BUDGETS,
        "labeled_sizes": LOCKED_LABELED_SIZES,
        "unlabeled_sizes": LOCKED_UNLABELED_SIZES,
        "labeled_membership_checksums": LOCKED_MEMBERSHIP_CHECKSUMS,
    }
    sums_ok = (
        sum(inherited_observed["labeled_sizes"] or [0])
        + 0 == sum(LOCKED_LABELED_SIZES)
    )
    pairs_ok = all(
        labeled + unlabeled == LOCKED_TRAIN_SIZE
        for labeled, unlabeled in zip(LOCKED_LABELED_SIZES,
                                      LOCKED_UNLABELED_SIZES)
    )
    checks.append(check(
        "C15", "Phase 2F inherited values and membership checksums match",
        inherited_observed == inherited_expected and sums_ok and pairs_ok,
        inherited_expected,
        {"inherited": inherited_observed,
         "labeled_plus_unlabeled_equals_train_size": pairs_ok},
        "{0}::inherited".format(CONFIG_REL)))

    # ---- C16 training_authorized -------------------------------------
    observed = {
        "config": dig(cfg, "authorization.training_authorized"),
        "seed_manifest": dig(seed_manifest or {},
                             "authorization.training_authorized"),
        "state_manifest": (state_manifest or {}).get("training_authorized"),
    }
    checks.append(check(
        "C16", "training_authorized is false in every artifact",
        all(value is False for value in observed.values()),
        {key: False for key in observed},
        observed,
        "{0}, {1}, {2}".format(CONFIG_REL, SEED_MANIFEST_REL,
                               STATE_MANIFEST_REL)))

    # ---- C17 training_started ----------------------------------------
    observed = {
        "config": dig(cfg, "authorization.training_started"),
        "seed_manifest": dig(seed_manifest or {},
                             "authorization.training_started"),
        "state_manifest": (state_manifest or {}).get("training_started"),
    }
    checks.append(check(
        "C17", "training_started is false in every artifact",
        all(value is False for value in observed.values()),
        {key: False for key in observed},
        observed,
        "{0}, {1}, {2}".format(CONFIG_REL, SEED_MANIFEST_REL,
                               STATE_MANIFEST_REL)))

    # ---- C18 state manifest template ----------------------------------
    state = state_manifest or {}
    observed = {
        "state": state.get("state"),
        "training_started": state.get("training_started"),
        "runs": state.get("runs"),
        "rng_state_captured": state.get("rng_state_captured"),
    }
    condition = (
        observed["state"] == STATE_TEMPLATE_VALUE
        and observed["training_started"] is False
        and isinstance(observed["runs"], list)
        and len(observed["runs"]) == 0
        and observed["rng_state_captured"] is False
    )
    checks.append(check(
        "C18", "state manifest is the empty template with runs == []",
        condition,
        {"state": STATE_TEMPLATE_VALUE, "training_started": False,
         "runs": [], "rng_state_captured": False},
        observed,
        STATE_MANIFEST_REL))

    # ---- C19 cross artifact consistency -------------------------------
    markdown_text = markdown_text or ""
    manifest_seed_values = dig(
        seed_manifest or {}, "training_seed.ordered_training_seed_values", [])
    state_seed_values = state.get("ordered_training_seed_values", [])
    missing_in_markdown = []
    for token in ([str(value) for value in LOCKED_TRAINING_SEEDS]
                  + [str(LOCKED_PARTITION_SEED)]
                  + list(LOCKED_MEMBERSHIP_CHECKSUMS.values())
                  + [LOCKED_PROTOCOL_IDENTITY, LOCKED_DETERMINISTIC_POLICY,
                     STATE_TEMPLATE_VALUE, "ddof=1", PHASE_CLOSURE_STATUS]):
        if token not in markdown_text:
            missing_in_markdown.append(token)
    markdown_norm = norm_text(markdown_text)
    rationale_in_markdown = (
        rationale_a in markdown_norm and rationale_b in markdown_norm)
    condition = (
        list(manifest_seed_values) == LOCKED_TRAINING_SEEDS
        and list(state_seed_values) == LOCKED_TRAINING_SEEDS
        and dig(seed_manifest or {}, "partition.partition_seed")
        == LOCKED_PARTITION_SEED
        and state.get("partition_seed") == LOCKED_PARTITION_SEED
        and dig(seed_manifest or {}, "inherited.labeled_membership_checksums")
        == LOCKED_MEMBERSHIP_CHECKSUMS
        and state.get("required_run_fields") == REQUIRED_RUN_FIELDS
        and not missing_in_markdown
        and rationale_in_markdown
    )
    checks.append(check(
        "C19", "YAML, JSON manifests and Markdown report agree with each other",
        condition,
        {"seed_values": LOCKED_TRAINING_SEEDS,
         "partition_seed": LOCKED_PARTITION_SEED,
         "markdown_missing_tokens": [],
         "both_rationales_in_markdown": True},
        {"seed_manifest_values": list(manifest_seed_values),
         "state_manifest_values": list(state_seed_values),
         "markdown_missing_tokens": missing_in_markdown,
         "both_rationales_in_markdown": rationale_in_markdown},
        "{0}, {1}, {2}, {3}".format(CONFIG_REL, SEED_MANIFEST_REL,
                                    STATE_MANIFEST_REL, MARKDOWN_REL)))

    # ---- C20 no fabricated training output ----------------------------
    observed = {
        "seed_manifest.runs": (seed_manifest or {}).get("runs"),
        "seed_manifest.checkpoints_present":
            (seed_manifest or {}).get("checkpoints_present"),
        "seed_manifest.results_present":
            (seed_manifest or {}).get("results_present"),
        "seed_manifest.rng_state_captured":
            (seed_manifest or {}).get("rng_state_captured"),
        "state_manifest.checkpoints_present": state.get("checkpoints_present"),
        "state_manifest.results_present": state.get("results_present"),
        "future_run_metadata_schema.runs_created_in_phase2F1":
            dig(cfg, "future_run_metadata_schema.runs_created_in_phase2F1"),
    }
    condition = (
        observed["seed_manifest.runs"] == []
        and observed["seed_manifest.checkpoints_present"] == 0
        and observed["seed_manifest.results_present"] == 0
        and observed["seed_manifest.rng_state_captured"] is False
        and observed["state_manifest.checkpoints_present"] == 0
        and observed["state_manifest.results_present"] == 0
        and observed["future_run_metadata_schema.runs_created_in_phase2F1"] == 0
    )
    checks.append(check(
        "C20", "no fabricated run, RNG state, checkpoint or result exists",
        condition,
        {"runs": [], "checkpoints_present": 0, "results_present": 0,
         "rng_state_captured": False, "runs_created_in_phase2F1": 0},
        observed,
        "{0}, {1}, {2}".format(SEED_MANIFEST_REL, STATE_MANIFEST_REL,
                               CONFIG_REL)))

    # ---- C21 future-run metadata schema completeness -------------------
    schema_fields = [
        entry.get("name")
        for entry in dig(cfg, "future_run_metadata_schema.required_fields", [])
    ]
    all_required = all(
        entry.get("required") is True
        for entry in dig(cfg, "future_run_metadata_schema.required_fields", []))
    checks.append(check(
        "C21", "the mandatory future-run metadata schema is complete",
        schema_fields == REQUIRED_RUN_FIELDS and all_required,
        REQUIRED_RUN_FIELDS,
        {"fields": schema_fields, "all_marked_required": all_required},
        "{0}::future_run_metadata_schema.required_fields".format(CONFIG_REL)))

    # ---- C22 closure status is pending --------------------------------
    observed = {
        "config": dig(cfg, "document.phase_closure_status"),
        "config.self_declared_closure_allowed":
            dig(cfg, "document.self_declared_closure_allowed"),
        "seed_manifest": (seed_manifest or {}).get("phase_closure_status"),
        "state_manifest": state.get("phase_closure_status"),
    }
    condition = (
        observed["config"] == PHASE_CLOSURE_STATUS
        and observed["config.self_declared_closure_allowed"] is False
        and observed["seed_manifest"] == PHASE_CLOSURE_STATUS
        and observed["state_manifest"] == PHASE_CLOSURE_STATUS
    )
    checks.append(check(
        "C22", "phase_closure_status stays PENDING_RESEARCHER_GPT_REVIEW",
        condition,
        {"phase_closure_status": PHASE_CLOSURE_STATUS,
         "self_declared_closure_allowed": False},
        observed,
        "{0}, {1}, {2}".format(CONFIG_REL, SEED_MANIFEST_REL,
                               STATE_MANIFEST_REL)))

    # ---- C23 existing validation report (validate mode only) ----------
    if mode == "validate-existing":
        report = existing_validation_report or {}
        observed = {
            "validation_status": report.get("validation_status"),
            "phase_closure_status": report.get("phase_closure_status"),
            "checks_present": len(report.get("checks", []) or []),
            "validation_pass_implies_phase_closure":
                report.get("validation_pass_implies_phase_closure"),
        }
        condition = (
            observed["validation_status"] in ("PASS", "FAIL")
            and observed["phase_closure_status"] == PHASE_CLOSURE_STATUS
            and observed["checks_present"] > 0
            and observed["validation_pass_implies_phase_closure"] is False
        )
        checks.append(check(
            "C23", "the existing validation report separates validation "
                   "status from phase closure status",
            condition,
            {"validation_status": "PASS or FAIL",
             "phase_closure_status": PHASE_CLOSURE_STATUS,
             "validation_pass_implies_phase_closure": False},
            observed,
            VALIDATION_REPORT_REL))

    return checks


# =====================================================================
# Reporting
# =====================================================================

def build_validation_report(checks, mode, config_sha256, timestamp,
                            artifacts_written):
    required_failed = [entry for entry in checks
                       if entry["required"] and entry["status"] == "FAIL"]
    status = "FAIL" if required_failed else "PASS"
    return {
        "report_id": "02F1_seed_protocol_validation_report",
        "schema_version": "1.0.0",
        "phase": "2F.1",
        "mode": mode,
        "generated_at_utc": timestamp,
        "generated_by": "scripts/02F1_build_seed_protocol.py",
        "source_config": CONFIG_REL,
        "source_config_sha256": config_sha256,
        "validation_status": status,
        "phase_closure_status": PHASE_CLOSURE_STATUS,
        "validation_pass_implies_phase_closure": False,
        "closure_decision_authority": "RESEARCHER_AND_GPT_REVIEW",
        "note": (
            "validation_status refers only to the internal consistency of the "
            "Phase 2F.1 contract artifacts. It does not close Phase 2F.1."
        ),
        "summary": {
            "total_checks": len(checks),
            "required_checks": sum(1 for e in checks if e["required"]),
            "passed": sum(1 for e in checks if e["status"] == "PASS"),
            "failed": sum(1 for e in checks if e["status"] == "FAIL"),
            "failed_required_check_ids": [e["check_id"] for e in required_failed],
        },
        "artifacts_written": artifacts_written,
        "checks": checks,
    }


def print_console_summary(report):
    print("=" * 70)
    print("Phase 2F.1 - Seed Protocol : mode={0}".format(report["mode"]))
    print("=" * 70)
    for entry in report["checks"]:
        print("[{0}] {1} - {2}".format(
            entry["status"], entry["check_id"], entry["description"]))
    print("-" * 70)
    summary = report["summary"]
    print("total_checks            : {0}".format(summary["total_checks"]))
    print("passed                  : {0}".format(summary["passed"]))
    print("failed                  : {0}".format(summary["failed"]))
    print("failed_required_checks  : {0}".format(
        ", ".join(summary["failed_required_check_ids"]) or "none"))
    print("validation_status       : {0}".format(report["validation_status"]))
    print("phase_closure_status    : {0}".format(report["phase_closure_status"]))
    print("NOTE: validation_status=PASS does not close Phase 2F.1.")
    print("=" * 70)


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=False, ensure_ascii=True)
        + "\n",
        encoding="utf-8",
    )


# =====================================================================
# Modes
# =====================================================================

def mode_execute():
    cfg, config_raw = load_config()
    config_sha256 = sha256_of_file(CONFIG_PATH)
    timestamp = utc_now_iso()

    derived = derive_training_seeds(
        namespace=dig(cfg, "training_seed.generation.namespace",
                      LOCKED_NAMESPACE),
        count=dig(cfg, "training_seed.training_seed_count",
                  LOCKED_TRAINING_SEED_COUNT),
        prefix_chars=dig(cfg, "training_seed.generation.digest_prefix_hex_chars",
                         DIGEST_PREFIX_HEX_CHARS),
        modulus=dig(cfg, "training_seed.generation.modulus", SEED_MODULUS),
        offset=dig(cfg, "training_seed.generation.offset", SEED_OFFSET),
    )

    seed_manifest = build_seed_manifest(cfg, derived, config_sha256, timestamp)
    state_manifest = build_state_manifest(cfg, derived, config_sha256, timestamp)
    markdown_text = build_markdown(cfg, derived, config_sha256, timestamp)

    checks = run_checks(cfg, config_raw, derived, seed_manifest,
                        state_manifest, markdown_text, mode="execute")
    required_failed = [entry for entry in checks
                       if entry["required"] and entry["status"] == "FAIL"]

    if required_failed:
        report = build_validation_report(
            checks, "execute", config_sha256, timestamp,
            artifacts_written=[VALIDATION_REPORT_REL])
        report["note_on_partial_output"] = (
            "At least one required check failed. The seed manifest, the state "
            "manifest and the Markdown report were NOT written, so that no "
            "downstream phase can inherit an inconsistent contract."
        )
        write_json(VALIDATION_REPORT_PATH, report)
        print_console_summary(report)
        print("[FAIL] artifacts not written: {0}, {1}, {2}".format(
            SEED_MANIFEST_REL, STATE_MANIFEST_REL, MARKDOWN_REL))
        return 1

    write_json(SEED_MANIFEST_PATH, seed_manifest)
    write_json(STATE_MANIFEST_PATH, state_manifest)
    MARKDOWN_PATH.parent.mkdir(parents=True, exist_ok=True)
    MARKDOWN_PATH.write_text(markdown_text, encoding="utf-8")

    report = build_validation_report(
        checks, "execute", config_sha256, timestamp,
        artifacts_written=[SEED_MANIFEST_REL, STATE_MANIFEST_REL,
                           MARKDOWN_REL, VALIDATION_REPORT_REL])
    write_json(VALIDATION_REPORT_PATH, report)
    print_console_summary(report)
    for rel in report["artifacts_written"]:
        print("[WROTE] {0}".format(rel))
    return 0


def mode_validate_existing():
    cfg, config_raw = load_config()
    config_sha256 = sha256_of_file(CONFIG_PATH)
    timestamp = utc_now_iso()

    errors = []
    seed_manifest, err = load_json(SEED_MANIFEST_PATH, SEED_MANIFEST_REL)
    if err:
        errors.append(err)
    state_manifest, err = load_json(STATE_MANIFEST_PATH, STATE_MANIFEST_REL)
    if err:
        errors.append(err)
    existing_report, err = load_json(VALIDATION_REPORT_PATH,
                                     VALIDATION_REPORT_REL)
    if err:
        errors.append(err)
    if MARKDOWN_PATH.is_file():
        markdown_text = MARKDOWN_PATH.read_text(encoding="utf-8")
    else:
        markdown_text = ""
        errors.append("missing artifact: {0}".format(MARKDOWN_REL))

    if errors:
        print("=" * 70)
        print("Phase 2F.1 - Seed Protocol : mode=validate-existing")
        print("=" * 70)
        for message in errors:
            print("[FAIL] {0}".format(message))
        print("validation_status       : FAIL")
        print("phase_closure_status    : {0}".format(PHASE_CLOSURE_STATUS))
        print("Run --execute first. No file was modified.")
        print("=" * 70)
        return 1

    derived = derive_training_seeds(
        namespace=dig(cfg, "training_seed.generation.namespace",
                      LOCKED_NAMESPACE),
        count=dig(cfg, "training_seed.training_seed_count",
                  LOCKED_TRAINING_SEED_COUNT),
        prefix_chars=dig(cfg, "training_seed.generation.digest_prefix_hex_chars",
                         DIGEST_PREFIX_HEX_CHARS),
        modulus=dig(cfg, "training_seed.generation.modulus", SEED_MODULUS),
        offset=dig(cfg, "training_seed.generation.offset", SEED_OFFSET),
    )

    checks = run_checks(cfg, config_raw, derived, seed_manifest,
                        state_manifest, markdown_text,
                        mode="validate-existing",
                        existing_validation_report=existing_report)

    report = build_validation_report(
        checks, "validate-existing", config_sha256, timestamp,
        artifacts_written=[])
    report["note_on_read_only"] = (
        "validate-existing is read-only. No artifact was created or modified."
    )
    print_console_summary(report)
    print("[READ-ONLY] no artifact was created or modified.")
    return 0 if report["validation_status"] == "PASS" else 1


# =====================================================================
# Entry point
# =====================================================================

def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Phase 2F.1 seed protocol builder / validator "
                    "(contract only, no training).")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--execute", action="store_true",
        help="build the four Phase 2F.1 artifacts and self-validate them")
    group.add_argument(
        "--validate-existing", dest="validate_existing", action="store_true",
        help="read-only validation of the existing artifacts")
    args = parser.parse_args(argv)

    try:
        if args.execute:
            return mode_execute()
        return mode_validate_existing()
    except SystemExit:
        raise
    except Exception as exc:  # pragma: no cover - defensive
        sys.stderr.write("[FATAL] {0}: {1}\n".format(type(exc).__name__, exc))
        return 2


if __name__ == "__main__":
    sys.exit(main())