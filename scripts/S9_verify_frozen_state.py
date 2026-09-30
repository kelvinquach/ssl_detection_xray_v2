#!/usr/bin/env python3
"""PRE-S9 frozen-state verification.

READ_ONLY
DETERMINISTIC
FAIL_FAST
NO_SCIENTIFIC_PROTOCOL_CHANGE
NO_FROZEN_ARTIFACT_REWRITE
NO_AUTHORIZATION_MUTATION
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable


EXPECTED_BRANCH = "implementation/s9"

EXPECTED_ARTIFACT_HASHES = {
    "artifacts/preflight/S8_CLOSURE_RECORD.json":
        "c56c57a1a14ac7941591a12cb048b1d19d3285282831ae53cee33a9707f661ff",
    "artifacts/frozen/frozen_source_manifest.json":
        "3febff4c652acdec831e977ffbaede265f253467d8ccc4f50e67327a531033e3",
    "artifacts/frozen/frozen_environment.json":
        "63692dd919845cbdbee89fb17913e1564c49d22ac509c36bfca4646e1959e82c",
    "artifacts/frozen/frozen_data_manifest.json":
        "a791e35c0cdc70d433ea719566e4f16571a06968ec5265c48cfd12e5b2496abe",
    "artifacts/frozen/FROZEN_EXPERIMENTAL_CONFIGURATION.yaml":
        "f937ec2e9bd6d5368eb5b968cbaa31ad595d18edce92f996b14963d275e58526",
    "artifacts/frozen/frozen_config_manifest.json":
        "b2cfe9d9d84755ce7c2bf239ed2dfb04f50184f4f839ad5164a11a1af2b993d2",
    "artifacts/frozen/frozen_evaluator_manifest.json":
        "01d2aede85ecded4542797b5468922aac0b01e1957420c64648b71bf71f4f69a",
    "artifacts/frozen/frozen_statistics_manifest.json":
        "8f4370a9a39c990a26df58adbe64a252edb26c7115632e7caac5a30d01f93582",
    "artifacts/frozen/TEST_FIREWALL_STATE.json":
        "325896df2f340c49437108ffaedc7118fe706c971cadf5e2df90aa37071c740d",
    "artifacts/frozen/FREEZE_AUTHORIZATION.json":
        "e8a2cf6d1373e7607e4d120d51ea8dca97db2b9b48710d0b04a1f0d3e7b2f9da",
    "artifacts/preflight/GLOBAL_PREFLIGHT_REPORT.json":
        "1c0cf21a6ab9546d320a7595e17e992721012f7129ea744913f8719b5175373e",
}

EXPECTED_AUTHORITY_HASHES = {
    "sn-article.tex":
        "b6881ce90da9194f88fd9664b51dc96bdf112145949e3a89705f16b06853c681",
    "IMPLEMENTATION_HANDOFF.md":
        "d274f98402e0492039a9da0bdfed538102fc6ca155f4f7870525c55c46aa0e1d",
    "ARTIFACT_AND_EVIDENCE_CONTRACT.md":
        "222270bfe62f64074ec1cc9c0436a68bd3b0766b92900a1457511ba4535d4ccc",
}

EXPECTED_SOURCE_PRIORITY = [
    "sn-article.tex",
    "IMPLEMENTATION_HANDOFF.md",
    "ARTIFACT_AND_EVIDENCE_CONTRACT.md",
]

EXPECTED_RUNTIME_IMAGE = "quachthainguyen/sslxray-runtime:gate-i-20260828"
EXPECTED_RUNTIME_DIGEST = (
    "sha256:fab1b1ea8850527e5728de526b3d53f063355e84cb3a0d8c75e3a7890411d989"
)

EXPECTED_SPLITS = {
    "train": "628b9bb8ba25129a928abe994b101b4c4efd5588d389feb60da6de2a371fa11a",
    "val": "87c23ebed4d1e6965731fc0b31245859f49e777119813c6152efde3531ba58c6",
    "test": "1f7903e069e872bf2e5fe13bb4d0fa257dc4a1c2c8290a621d3f7286ada66b37",
}

EXPECTED_LABELED = {
    "1pct": "c54e7d61e84b7cfce68c04a795783b5c5d01331d2aa9c754fb1ae1dbae4ba071",
    "5pct": "c4db3b5f7a5b0f391ad883ef665c3d341af3ef6afb3fc553e71f1c4ef17ee50b",
    "10pct": "fc008d31505227544087ba474613075a8cd077587df9d6b13146b378f5a3a7d6",
    "20pct": "6f4aaba6be147983d56007c49ada234dfcabe7ec2f936f28f8cd240999b8417e",
}

EXPECTED_STATS_IMAGE = "quachthainguyen/sslxray-statistics:s6-r4.5.3-20260423"
EXPECTED_STATS_REFERENCE = (
    "quachthainguyen/sslxray-statistics@"
    "sha256:bb99a07178ba662bd2e84c5e03421007668b7e35856c83cb6ee816cffab15d37"
)

GATES = [
    "S8_CLOSURE_GATE",
    "SOURCE_FREEZE_GATE",
    "RUNTIME_FREEZE_GATE",
    "DATA_FREEZE_GATE",
    "CONFIG_FREEZE_GATE",
    "EVALUATOR_FREEZE_GATE",
    "STATISTICS_FREEZE_GATE",
    "TEST_FIREWALL_GATE",
    "RESEARCHER_AUTHORIZATION_GATE",
    "GLOBAL_PREFLIGHT_GATE",
]

failures: dict[str, list[str]] = {gate: [] for gate in GATES}


class TechnicalBlocked(RuntimeError):
    pass


def run_git(*args: str) -> bytes:
    try:
        proc = subprocess.run(
            ["git", *args],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
    except OSError as exc:
        raise TechnicalBlocked(f"git invocation failed: {exc}") from exc

    if proc.returncode != 0:
        detail = proc.stderr.decode("utf-8", errors="replace").strip()
        raise TechnicalBlocked(
            f"git {' '.join(args)} failed with code {proc.returncode}: {detail}"
        )

    return proc.stdout


def git_text(*args: str) -> str:
    return run_git(*args).decode("utf-8", errors="strict").strip()


def blob_bytes(path: str) -> bytes:
    return run_git("cat-file", "blob", f"HEAD:{path}")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def fail(gate: str, message: str) -> None:
    failures[gate].append(message)


def require(gate: str, condition: bool, message: str) -> None:
    if not condition:
        fail(gate, message)


def load_frozen_json(gate: str, path: str) -> dict[str, Any] | None:
    try:
        raw = blob_bytes(path)
    except TechnicalBlocked as exc:
        fail(gate, f"required artifact unavailable: {path}: {exc}")
        return None

    expected = EXPECTED_ARTIFACT_HASHES[path]
    observed = sha256_bytes(raw)
    require(
        gate,
        observed == expected,
        f"Git-blob SHA256 mismatch: {path}: expected={expected} observed={observed}",
    )

    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        fail(gate, f"invalid JSON: {path}: {exc}")
        return None

    if not isinstance(value, dict):
        fail(gate, f"JSON root is not an object: {path}")
        return None

    return value


def verify_fixed_blob(gate: str, path: str, expected_sha256: str) -> None:
    try:
        observed = sha256_bytes(blob_bytes(path))
    except TechnicalBlocked as exc:
        fail(gate, f"required Git blob unavailable: {path}: {exc}")
        return

    require(
        gate,
        observed == expected_sha256,
        f"Git-blob SHA256 mismatch: {path}: expected={expected_sha256} observed={observed}",
    )


def verify_manifest_paths(
    gate: str,
    entries: Iterable[Any],
    label: str,
) -> None:
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            fail(gate, f"{label}[{index}] is not an object")
            continue

        path = entry.get("path")
        expected = entry.get("sha256")

        if not isinstance(path, str) or not isinstance(expected, str):
            fail(gate, f"{label}[{index}] missing path/sha256")
            continue

        verify_fixed_blob(gate, path, expected)


def verify_s8_closure() -> None:
    gate = "S8_CLOSURE_GATE"
    data = load_frozen_json(
        gate, "artifacts/preflight/S8_CLOSURE_RECORD.json"
    )
    if data is None:
        return

    require(gate, data.get("stage") == "S8", "stage != S8")
    require(gate, data.get("status") == "CLOSED / PASS", "status != CLOSED / PASS")
    require(
        gate,
        data.get("official_training_authorized") is True,
        "official_training_authorized != true",
    )
    require(
        gate,
        data.get("final_test_authorized") is False,
        "final_test_authorized != false",
    )
    require(
        gate,
        data.get("global_preflight_closed") is True,
        "global_preflight_closed != true",
    )

    scope = data.get("closure_scope")
    require(gate, isinstance(scope, dict), "closure_scope missing/not object")
    if isinstance(scope, dict):
        require(
            gate,
            scope.get("researcher_authorization") is True,
            "closure_scope.researcher_authorization != true",
        )
        require(
            gate,
            scope.get("test_access_authorized") is False,
            "closure_scope.test_access_authorized != false",
        )
        require(
            gate,
            scope.get("test_access_reason") == "FINAL_ONLY",
            "closure_scope.test_access_reason != FINAL_ONLY",
        )
        require(
            gate,
            scope.get("unresolved_protocol_relevant_warning_count") == 0,
            "closure_scope unresolved warning count != 0",
        )


def verify_source() -> None:
    gate = "SOURCE_FREEZE_GATE"
    data = load_frozen_json(
        gate, "artifacts/frozen/frozen_source_manifest.json"
    )
    if data is None:
        return

    require(gate, data.get("status") == "PASS", "source status != PASS")
    require(
        gate,
        data.get("hash_basis") == "GIT_BLOB_CONTENT_SHA256",
        "source hash_basis mismatch",
    )
    require(
        gate,
        data.get("priority_order") == EXPECTED_SOURCE_PRIORITY,
        "source priority_order mismatch",
    )
    require(
        gate,
        data.get("git_commit") == "3643459c86fd9e51bdb0a322a50f7f3703184d2a",
        "frozen implementation commit mismatch",
    )

    expected_fields = {
        "scientific_source_sha256":
            EXPECTED_AUTHORITY_HASHES["sn-article.tex"],
        "implementation_contract_sha256":
            EXPECTED_AUTHORITY_HASHES["IMPLEMENTATION_HANDOFF.md"],
        "artifact_contract_sha256":
            EXPECTED_AUTHORITY_HASHES["ARTIFACT_AND_EVIDENCE_CONTRACT.md"],
    }

    for field, expected in expected_fields.items():
        require(gate, data.get(field) == expected, f"{field} mismatch")

    for path, expected in EXPECTED_AUTHORITY_HASHES.items():
        verify_fixed_blob(gate, path, expected)


def verify_runtime() -> None:
    gate = "RUNTIME_FREEZE_GATE"
    data = load_frozen_json(
        gate, "artifacts/frozen/frozen_environment.json"
    )
    if data is None:
        return

    require(gate, data.get("status") == "PASS", "runtime status != PASS")

    docker = data.get("docker")
    require(gate, isinstance(docker, dict), "docker object missing")
    if isinstance(docker, dict):
        require(
            gate,
            docker.get("expected_image_uuid") == EXPECTED_RUNTIME_IMAGE,
            "runtime image UUID mismatch",
        )
        require(
            gate,
            docker.get("observed_image_uuid") == EXPECTED_RUNTIME_IMAGE,
            "observed runtime image UUID mismatch",
        )
        require(
            gate,
            docker.get("expected_digest") == EXPECTED_RUNTIME_DIGEST,
            "expected runtime digest mismatch",
        )
        require(
            gate,
            docker.get("observed_registry_digest") == EXPECTED_RUNTIME_DIGEST,
            "observed runtime digest mismatch",
        )
        require(
            gate,
            docker.get("digest_match") is True,
            "runtime digest_match != true",
        )


def verify_data() -> None:
    gate = "DATA_FREEZE_GATE"
    data = load_frozen_json(
        gate, "artifacts/frozen/frozen_data_manifest.json"
    )
    if data is None:
        return

    require(gate, data.get("status") == "PASS", "data status != PASS")
    require(gate, data.get("partition_seed") == 42, "partition_seed != 42")
    require(
        gate,
        data.get("seed_policy") == "PRE_SPECIFIED_LOCKED_NO_SEED_SEARCH",
        "seed_policy mismatch",
    )

    for split, expected in EXPECTED_SPLITS.items():
        obj = data.get(split)
        require(gate, isinstance(obj, dict), f"{split} object missing")
        if isinstance(obj, dict):
            require(
                gate,
                obj.get("membership_sha256") == expected,
                f"{split}.membership_sha256 mismatch",
            )

    budgets = data.get("budgets")
    require(gate, isinstance(budgets, dict), "budgets object missing")
    if isinstance(budgets, dict):
        for budget, expected in EXPECTED_LABELED.items():
            obj = budgets.get(budget)
            require(gate, isinstance(obj, dict), f"budgets.{budget} missing")
            if isinstance(obj, dict):
                require(
                    gate,
                    obj.get("labeled_membership_sha256") == expected,
                    f"budgets.{budget}.labeled_membership_sha256 mismatch",
                )

    unlabeled = data.get("unlabeled_membership_semantics")
    require(
        gate,
        isinstance(unlabeled, dict),
        "unlabeled_membership_semantics missing",
    )
    if isinstance(unlabeled, dict):
        require(
            gate,
            unlabeled.get("definition") == r"U_b = T \\ L_b",
            "unlabeled complement definition mismatch",
        )
        require(
            gate,
            unlabeled.get("all_U_equal_exact_train_complement") is True,
            "U_b is not exact train complement",
        )
        require(
            gate,
            unlabeled.get("all_L_U_disjoint") is True,
            "L/U disjoint semantic mismatch",
        )
        require(
            gate,
            unlabeled.get("all_L_U_union_equals_train") is True,
            "L/U union semantic mismatch",
        )
        require(
            gate,
            unlabeled.get("unlabeled_coco_sha256_locked") is True,
            "unlabeled COCO hashes not locked",
        )

    nesting = data.get("nesting")
    require(gate, isinstance(nesting, dict), "nesting object missing")
    if isinstance(nesting, dict):
        for field in (
            "L1_subset_L5",
            "L5_subset_L10",
            "L10_subset_L20",
            "L20_subset_train",
            "U20_subset_U10",
            "U10_subset_U5",
            "U5_subset_U1",
        ):
            require(
                gate,
                nesting.get(field) is True,
                f"nesting.{field} != true",
            )


def verify_config() -> None:
    gate = "CONFIG_FREEZE_GATE"

    verify_fixed_blob(
        gate,
        "artifacts/frozen/FROZEN_EXPERIMENTAL_CONFIGURATION.yaml",
        EXPECTED_ARTIFACT_HASHES[
            "artifacts/frozen/FROZEN_EXPERIMENTAL_CONFIGURATION.yaml"
        ],
    )

    data = load_frozen_json(
        gate, "artifacts/frozen/frozen_config_manifest.json"
    )
    if data is None:
        return

    require(gate, data.get("status") == "PASS", "config status != PASS")
    require(
        gate,
        data.get("scientific_protocol_change") is False,
        "scientific_protocol_change != false",
    )
    require(gate, data.get("executable_config_count") == 18, "config count != 18")
    require(gate, data.get("supervised_config_count") == 10, "SUP config count != 10")
    require(gate, data.get("ssl_config_count") == 8, "SSL config count != 8")

    frozen = data.get("frozen_experimental_configuration")
    require(
        gate,
        isinstance(frozen, dict),
        "frozen_experimental_configuration missing",
    )
    if isinstance(frozen, dict):
        require(
            gate,
            frozen.get("path")
            == "artifacts/frozen/FROZEN_EXPERIMENTAL_CONFIGURATION.yaml",
            "frozen experimental configuration path mismatch",
        )
        require(
            gate,
            frozen.get("sha256")
            == EXPECTED_ARTIFACT_HASHES[
                "artifacts/frozen/FROZEN_EXPERIMENTAL_CONFIGURATION.yaml"
            ],
            "frozen experimental configuration hash mismatch",
        )

    entries = data.get("executable_configs")
    require(gate, isinstance(entries, list), "executable_configs missing/not list")
    if isinstance(entries, list):
        require(gate, len(entries) == 18, "executable_configs length != 18")

        sup_count = sum(
            1 for item in entries
            if isinstance(item, dict) and item.get("method") == "SUP"
        )
        ssl_count = sum(
            1 for item in entries
            if isinstance(item, dict) and item.get("method") == "SSL"
        )

        require(gate, sup_count == 10, "executable SUP entries != 10")
        require(gate, ssl_count == 8, "executable SSL entries != 8")

        verify_manifest_paths(gate, entries, "executable_configs")


def verify_evaluator() -> None:
    gate = "EVALUATOR_FREEZE_GATE"
    data = load_frozen_json(
        gate, "artifacts/frozen/frozen_evaluator_manifest.json"
    )
    if data is None:
        return

    require(gate, data.get("status") == "PASS", "evaluator status != PASS")
    require(
        gate,
        data.get("hash_basis") == "GIT_BLOB_CONTENT_SHA256",
        "evaluator hash_basis mismatch",
    )
    require(
        gate,
        data.get("scientific_protocol_change") is False,
        "evaluator scientific_protocol_change != false",
    )

    protocol = data.get("evaluator_protocol")
    require(gate, isinstance(protocol, dict), "evaluator_protocol missing")
    if isinstance(protocol, dict):
        verify_fixed_blob(
            gate,
            str(protocol.get("path")),
            str(protocol.get("sha256")),
        )

    coco = data.get("coco")
    require(gate, isinstance(coco, dict), "coco object missing")
    if isinstance(coco, dict):
        sem = coco.get("scientific_semantics")
        require(gate, isinstance(sem, dict), "coco scientific_semantics missing")
        if isinstance(sem, dict):
            require(
                gate,
                sem.get("primary_metric") == "bbox mAP@[0.50:0.95]",
                "COCO primary metric mismatch",
            )
            require(gate, sem.get("useCats") == 1, "COCO useCats != 1")
            require(gate, sem.get("maxDets") == 100, "COCO maxDets != 100")
            require(
                gate,
                sem.get("extra_fixed_score_threshold_before_AP") is False,
                "unexpected fixed score threshold before COCO AP",
            )

        sources = coco.get("sources")
        require(gate, isinstance(sources, list), "coco.sources missing")
        if isinstance(sources, list):
            verify_manifest_paths(gate, sources, "coco.sources")

        golden = coco.get("golden_evidence")
        require(gate, isinstance(golden, dict), "coco golden_evidence missing")
        if isinstance(golden, dict):
            require(gate, golden.get("status") == "PASS", "COCO golden evidence != PASS")
            verify_fixed_blob(
                gate,
                str(golden.get("path")),
                str(golden.get("sha256")),
            )

    op = data.get("operating_point")
    require(gate, isinstance(op, dict), "operating_point object missing")
    if isinstance(op, dict):
        sem = op.get("scientific_semantics")
        require(gate, isinstance(sem, dict), "operating point semantics missing")
        if isinstance(sem, dict):
            require(gate, sem.get("tau_eval") == 0.5, "tau_eval != 0.50")
            require(
                gate,
                sem.get("iou_match_threshold") == 0.5,
                "IoU threshold != 0.50",
            )
            require(
                gate,
                sem.get("matching") == "CATEGORY_AWARE_ONE_TO_ONE",
                "operating-point matching mismatch",
            )
            require(
                gate,
                sem.get("matching_algorithm")
                == "DETERMINISTIC_SCORE_DESCENDING_GREEDY",
                "operating-point matching algorithm mismatch",
            )
            require(
                gate,
                sem.get("max_dets_per_image") == 100,
                "operating-point max detections != 100",
            )

        current = op.get("current_sources")
        require(gate, isinstance(current, list), "operating_point.current_sources missing")
        if isinstance(current, list):
            verify_manifest_paths(
                gate, current, "operating_point.current_sources"
            )

        golden = op.get("golden_evidence")
        require(gate, isinstance(golden, dict), "operating point golden evidence missing")
        if isinstance(golden, dict):
            require(
                gate,
                golden.get("status") == "PASS",
                "operating point golden evidence != PASS",
            )
            verify_fixed_blob(
                gate,
                str(golden.get("path")),
                str(golden.get("sha256")),
            )

    qp = data.get("q_pseudo")
    require(gate, isinstance(qp, dict), "q_pseudo object missing")
    if isinstance(qp, dict):
        sem = qp.get("scientific_semantics")
        require(gate, isinstance(sem, dict), "q_pseudo semantics missing")
        if isinstance(sem, dict):
            require(
                gate,
                sem.get("checkpoint") == "LAST_EMA_TEACHER",
                "Q_pseudo checkpoint mismatch",
            )
            require(
                gate,
                sem.get("dataset") == "FIXED_VALIDATION",
                "Q_pseudo dataset mismatch",
            )
            require(
                gate,
                sem.get("hidden_u_gt_used") is False,
                "Q_pseudo hidden-U GT used",
            )
            require(
                gate,
                sem.get("test_used") is False,
                "Q_pseudo test used",
            )

        current = qp.get("current_sources")
        require(gate, isinstance(current, list), "q_pseudo.current_sources missing")
        if isinstance(current, list):
            verify_manifest_paths(gate, current, "q_pseudo.current_sources")

        golden = qp.get("golden_evidence")
        require(gate, isinstance(golden, dict), "Q_pseudo golden evidence missing")
        if isinstance(golden, dict):
            require(
                gate,
                golden.get("status") == "PASS",
                "Q_pseudo golden evidence != PASS",
            )
            verify_fixed_blob(
                gate,
                str(golden.get("path")),
                str(golden.get("sha256")),
            )


def verify_statistics() -> None:
    gate = "STATISTICS_FREEZE_GATE"
    data = load_frozen_json(
        gate, "artifacts/frozen/frozen_statistics_manifest.json"
    )
    if data is None:
        return

    require(gate, data.get("status") == "PASS", "statistics status != PASS")
    require(
        gate,
        data.get("hash_basis") == "GIT_BLOB_CONTENT_SHA256",
        "statistics hash_basis mismatch",
    )
    require(
        gate,
        data.get("scientific_protocol_change") is False,
        "statistics scientific_protocol_change != false",
    )
    require(
        gate,
        data.get("statistical_recomputation_performed") is False,
        "statistics unexpectedly recomputed",
    )

    runtime = data.get("runtime_lineage")
    require(gate, isinstance(runtime, dict), "runtime_lineage missing")
    if isinstance(runtime, dict):
        successor = runtime.get("frozen_successor_r_runtime")
        require(
            gate,
            isinstance(successor, dict),
            "frozen successor R runtime missing",
        )
        if isinstance(successor, dict):
            require(
                gate,
                successor.get("image") == EXPECTED_STATS_IMAGE,
                "statistics image mismatch",
            )
            require(
                gate,
                successor.get("immutable_image_reference")
                == EXPECTED_STATS_REFERENCE,
                "statistics immutable image reference mismatch",
            )
            require(gate, successor.get("R") == "4.5.3", "statistics R != 4.5.3")

            packages = successor.get("packages")
            require(
                gate,
                isinstance(packages, dict),
                "statistics package map missing",
            )
            if isinstance(packages, dict):
                expected_packages = {
                    "lme4": "2.0.1",
                    "lmerTest": "3.2.1",
                    "pbkrtest": "0.5.5",
                    "jsonlite": "2.0.0",
                    "clubSandwich": "0.6.2",
                }
                for package, version in expected_packages.items():
                    require(
                        gate,
                        packages.get(package) == version,
                        f"statistics package {package} mismatch",
                    )

    entries = data.get("implementation_file_hashes")
    require(
        gate,
        isinstance(entries, list),
        "implementation_file_hashes missing/not list",
    )
    if isinstance(entries, list):
        require(
            gate,
            len(entries) == 29,
            "statistics implementation_file_hashes length != 29",
        )
        verify_manifest_paths(
            gate,
            entries,
            "implementation_file_hashes",
        )

    core = data.get("core_evidence")
    require(gate, isinstance(core, dict), "statistics core_evidence missing")
    if isinstance(core, dict):
        for name, entry in core.items():
            if not isinstance(entry, dict):
                continue

            path = entry.get("path")
            expected = entry.get("sha256")

            if isinstance(path, str) and isinstance(expected, str):
                verify_fixed_blob(gate, path, expected)

            status = entry.get("status")
            if status is not None:
                require(
                    gate,
                    status in ("PASS", "CLOSED / PASS"),
                    f"core_evidence.{name}.status not PASS",
                )


def verify_test_firewall() -> None:
    gate = "TEST_FIREWALL_GATE"
    data = load_frozen_json(
        gate, "artifacts/frozen/TEST_FIREWALL_STATE.json"
    )
    if data is None:
        return

    require(
        gate,
        data.get("test_access_authorized") is False,
        "test_access_authorized != false",
    )
    require(
        gate,
        data.get("reason") == "FINAL_ONLY",
        "test firewall reason != FINAL_ONLY",
    )


def verify_researcher_authorization() -> None:
    gate = "RESEARCHER_AUTHORIZATION_GATE"
    data = load_frozen_json(
        gate, "artifacts/frozen/FREEZE_AUTHORIZATION.json"
    )
    if data is None:
        return

    require(
        gate,
        data.get("researcher_authorization") is True,
        "researcher_authorization != true",
    )


def verify_global_preflight() -> None:
    gate = "GLOBAL_PREFLIGHT_GATE"
    data = load_frozen_json(
        gate, "artifacts/preflight/GLOBAL_PREFLIGHT_REPORT.json"
    )
    if data is None:
        return

    require(gate, data.get("status") == "PASS", "Global Preflight status != PASS")
    require(
        gate,
        data.get("all_required_checks_pass") is True,
        "all_required_checks_pass != true",
    )
    require(
        gate,
        data.get("unresolved_protocol_relevant_warnings") == 0,
        "unresolved protocol-relevant warnings != 0",
    )

    checks = data.get("checks")
    require(gate, isinstance(checks, list), "Global Preflight checks missing")
    if isinstance(checks, list):
        require(gate, len(checks) == 25, "Global Preflight check count != 25")

        expected_ids = [f"PF{i:02d}" for i in range(1, 26)]
        observed_ids = [
            item.get("check_id")
            for item in checks
            if isinstance(item, dict)
        ]

        require(
            gate,
            observed_ids == expected_ids,
            "Global Preflight PF01-PF25 identity/order mismatch",
        )

        for index, item in enumerate(checks):
            if not isinstance(item, dict):
                fail(gate, f"checks[{index}] is not an object")
                continue

            require(
                gate,
                item.get("status") == "PASS",
                f"{item.get('check_id', index)} status != PASS",
            )

            if item.get("check_id") == "PF25":
                require(
                    gate,
                    item.get("expected") == "TRUE",
                    "PF25 expected != TRUE",
                )
                require(
                    gate,
                    item.get("observed") == "TRUE",
                    "PF25 observed != TRUE",
                )
            else:
                require(
                    gate,
                    item.get("expected") == "PASS",
                    f"{item.get('check_id', index)} expected != PASS",
                )
                require(
                    gate,
                    item.get("observed") == "PASS",
                    f"{item.get('check_id', index)} observed != PASS",
                )


def main() -> int:
    try:
        repo_root = Path(git_text("rev-parse", "--show-toplevel")).resolve()
        cwd = Path.cwd().resolve()

        if cwd != repo_root:
            raise TechnicalBlocked(
                f"run from repository root: cwd={cwd} repo_root={repo_root}"
            )

        branch = git_text("branch", "--show-current")
        if branch != EXPECTED_BRANCH:
            raise TechnicalBlocked(
                f"expected branch {EXPECTED_BRANCH}, observed {branch}"
            )

        verify_s8_closure()
        verify_source()
        verify_runtime()
        verify_data()
        verify_config()
        verify_evaluator()
        verify_statistics()
        verify_test_firewall()
        verify_researcher_authorization()
        verify_global_preflight()

    except TechnicalBlocked as exc:
        print(f"TECHNICAL_ERROR={exc}")
        print("S9_FROZEN_STATE_VERIFICATION=BLOCKED")
        return 2
    except Exception as exc:
        print(f"TECHNICAL_ERROR={type(exc).__name__}: {exc}")
        print("S9_FROZEN_STATE_VERIFICATION=BLOCKED")
        return 2

    failure_count = sum(len(items) for items in failures.values())

    if failure_count:
        for gate in GATES:
            for message in failures[gate]:
                print(f"FROZEN_STATE_FAILURE={gate}|{message}")

    for gate in GATES:
        result = "PASS" if not failures[gate] else "FAIL"
        print(f"{gate}={result}")

    print(f"FROZEN_STATE_FAILURE_COUNT={failure_count}")

    if failure_count == 0:
        print("S9_FROZEN_STATE_VERIFICATION=PASS")
        return 0

    print("S9_FROZEN_STATE_VERIFICATION=FAIL")
    return 1


if __name__ == "__main__":
    sys.exit(main())