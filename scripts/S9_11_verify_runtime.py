"""Frozen-runtime preflight for S9.11 validation-only OFAT ablations.

This script is intentionally verification-only.  It never calls
Runner.train(), never creates final-test configuration, and never
authorizes test access.

Run only in the frozen S9 Vast runtime after the S9.11 implementation
commit has been synchronized safely.

PASS requires:
- exactly 90 non-Main ablation logical runs;
- exactly 9 executable non-Main configurations;
- all 10 frozen seeds per configuration;
- all configurations import through MMEngine;
- all 90 runtime expected identities PASS;
- effective C/A/F scientific semantics match the locked S9.11 design;
- no test dataloader/evaluator is exposed;
- same-seed frozen Official Main runs are CLOSED_VERIFIED;
- ablation runtime identity equals the corresponding Official Main
  runtime identity;
- Q_pseudo remains the fixed Main measurement contract at 0.90.

No training is executed by this verifier.
"""

from __future__ import annotations

import csv
import json
import re
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


INVENTORY_PATH = (
    REPO_ROOT
    / "artifacts"
    / "results"
    / "ablation"
    / "ablation_run_inventory.csv"
)

OFFICIAL_RUN_ROOT = (
    REPO_ROOT
    / "artifacts"
    / "runs"
    / "official"
)

MAIN_CONFIG = (
    "configs/ssl/"
    "s5_06_soft_teacher_r50_fpn_scientific_eval_10pct.py"
)

QPSEUDO_CONFIG = (
    REPO_ROOT
    / "configs"
    / "evaluation"
    / "s5_06_qpseudo_acceptance.py"
)

SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class RuntimePreflightError(RuntimeError):
    pass


def _parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value

    text = str(value).strip().lower()

    if text in {"1", "true", "yes", "y"}:
        return True

    if text in {"0", "false", "no", "n", ""}:
        return False

    raise RuntimePreflightError(
        f"Invalid boolean value: {value!r}"
    )


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(
            path.read_text(encoding="utf-8-sig")
        )
    except Exception as exc:
        raise RuntimePreflightError(
            f"Cannot parse JSON {path}: {exc}"
        ) from exc

    if not isinstance(payload, dict):
        raise RuntimePreflightError(
            f"Expected JSON object: {path}"
        )

    return payload


def _load_inventory() -> list[dict[str, str]]:
    if not INVENTORY_PATH.is_file():
        raise RuntimePreflightError(
            f"Missing ablation inventory: {INVENTORY_PATH}"
        )

    with INVENTORY_PATH.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as handle:
        rows = list(csv.DictReader(handle))

    if len(rows) != 90:
        raise RuntimePreflightError(
            f"Expected 90 ablation rows; observed={len(rows)}"
        )

    run_ids = [
        str(row["ablation_run_id"]).strip()
        for row in rows
    ]

    if len(set(run_ids)) != 90:
        raise RuntimePreflightError(
            "Duplicate ablation logical run ID."
        )

    configs: dict[str, int] = {}
    seeds_by_config: dict[str, set[int]] = {}

    for row in rows:
        run_id = str(row["ablation_run_id"]).strip()

        if str(row["condition_role"]).strip() != "ABLATION":
            raise RuntimePreflightError(
                f"{run_id}: condition_role != ABLATION"
            )

        if str(row["method"]).strip() != "SSL":
            raise RuntimePreflightError(
                f"{run_id}: method != SSL"
            )

        if _parse_bool(row["is_main"]):
            raise RuntimePreflightError(
                f"{run_id}: non-Main inventory contains Main."
            )

        if not _parse_bool(row["validation_only"]):
            raise RuntimePreflightError(
                f"{run_id}: validation_only != true"
            )

        if _parse_bool(row["final_test_eligible"]):
            raise RuntimePreflightError(
                f"{run_id}: final_test_eligible != false"
            )

        if _parse_bool(row["test_access_authorized"]):
            raise RuntimePreflightError(
                f"{run_id}: test access authorized."
            )

        config_id = str(row["ablation_config_id"]).strip()
        seed = int(str(row["training_seed"]).strip())

        configs[config_id] = configs.get(config_id, 0) + 1
        seeds_by_config.setdefault(config_id, set()).add(seed)

    if set(configs) != {
        "C0", "C1", "C2", "C3",
        "A0", "A1", "A2",
        "F0", "F1",
    }:
        raise RuntimePreflightError(
            f"Unexpected non-Main config set: {sorted(configs)}"
        )

    for config_id, count in sorted(configs.items()):
        if count != 10:
            raise RuntimePreflightError(
                f"{config_id}: expected 10 rows, observed={count}"
            )

        if len(seeds_by_config[config_id]) != 10:
            raise RuntimePreflightError(
                f"{config_id}: expected 10 unique seeds."
            )

    return rows


def _find_qpseudo_threshold(value: Any) -> list[float]:
    found: list[float] = []

    if isinstance(value, dict):
        for key, child in value.items():
            normalized = str(key).lower()

            if (
                "score" in normalized
                and "thr" in normalized
                and isinstance(child, (int, float))
                and not isinstance(child, bool)
            ):
                found.append(float(child))

            found.extend(_find_qpseudo_threshold(child))

    elif isinstance(value, (list, tuple)):
        for child in value:
            found.extend(_find_qpseudo_threshold(child))

    return found


def _official_main_attempt(
    main_run_id: str,
) -> tuple[Path, dict[str, Any]]:
    run_root = OFFICIAL_RUN_ROOT / main_run_id

    status_path = run_root / "logical_run_status.json"

    if not status_path.is_file():
        raise RuntimePreflightError(
            "BLOCKED_MAIN_LOGICAL_STATUS_MISSING:"
            + main_run_id
        )

    status = _load_json(status_path)

    if status.get("training_status") != "CLOSED_VERIFIED":
        raise RuntimePreflightError(
            "BLOCKED_MAIN_NOT_CLOSED_VERIFIED:"
            + main_run_id
            + ":"
            + str(status.get("training_status"))
        )

    if "final_test_accessed" in status:
        if status.get("final_test_accessed") is not False:
            raise RuntimePreflightError(
                "MAIN_FINAL_TEST_ACCESS_INVALID:"
                + main_run_id
            )
    else:
        if status.get("final_evaluation_status") != "NOT_AUTHORIZED":
            raise RuntimePreflightError(
                "MAIN_LEGACY_FINAL_TEST_FIREWALL_INVALID:"
                + main_run_id
            )

    attempt_id = status.get("attempt_used")

    if not isinstance(attempt_id, str) or not attempt_id:
        raise RuntimePreflightError(
            "MAIN_ATTEMPT_ID_MISSING:"
            + main_run_id
        )

    attempt_dir = run_root / attempt_id

    if not attempt_dir.is_dir():
        raise RuntimePreflightError(
            "MAIN_ATTEMPT_DIRECTORY_MISSING:"
            + str(attempt_dir)
        )

    return attempt_dir, status


def _runtime_identity_from_environment(
    path: Path,
) -> str:
    payload = _load_json(path)

    candidates: list[Any] = [
        payload.get("runtime_identity_sha256"),
    ]

    for key in (
        "environment",
        "original_environment",
        "runtime",
        "snapshot",
    ):
        child = payload.get(key)

        if isinstance(child, dict):
            candidates.append(
                child.get("runtime_identity_sha256")
            )

    for candidate in candidates:
        if (
            isinstance(candidate, str)
            and SHA256_RE.fullmatch(candidate.strip().lower())
        ):
            return candidate.strip().lower()

    raise RuntimePreflightError(
        "RUNTIME_IDENTITY_SHA256_NOT_FOUND:"
        + str(path)
    )


def main() -> int:
    from mmengine.config import Config

    from src.utils.s9_11_ablation_expected_identity import (
        build_ablation_expected_identity,
    )

    from src.utils.s9_11_ablation_effective_fields import (
        extract_s9_11_effective_scientific_fields,
    )

    from src.utils.s9_11_ablation_environment_reference import (
        _current_snapshot,
    )

    rows = _load_inventory()

    main_cfg = Config.fromfile(
        str(REPO_ROOT / MAIN_CONFIG)
    )

    if main_cfg.get("test_dataloader", None) is not None:
        raise RuntimePreflightError(
            "MAIN_CONFIG_EXPOSES_TEST_DATALOADER"
        )

    if main_cfg.get("test_evaluator", None) is not None:
        raise RuntimePreflightError(
            "MAIN_CONFIG_EXPOSES_TEST_EVALUATOR"
        )

    config_rows: dict[str, dict[str, str]] = {}

    for row in rows:
        config_id = str(row["ablation_config_id"]).strip()
        config_rows.setdefault(config_id, row)

    runtime_effective: dict[str, dict[str, Any]] = {}

    for config_id in sorted(config_rows):
        row = config_rows[config_id]

        config_path = (
            REPO_ROOT
            / str(row["executable_config_path"]).strip()
        )

        if not config_path.is_file():
            raise RuntimePreflightError(
                f"{config_id}: config missing: {config_path}"
            )

        cfg = Config.fromfile(str(config_path))

        if cfg.get("test_dataloader", None) is not None:
            raise RuntimePreflightError(
                f"{config_id}: test_dataloader exposed."
            )

        if cfg.get("test_evaluator", None) is not None:
            raise RuntimePreflightError(
                f"{config_id}: test_evaluator exposed."
            )

        effective = (
            extract_s9_11_effective_scientific_fields(
                cfg=cfg,
                ablation_family=str(row["family"]).strip(),
                ablation_config_id=str(
                    row["ablation_config_id"]
                ).strip(),
                ablation_level=str(row["level"]).strip(),
            )
        )

        if effective.get("ofat_effective_semantics_valid") is not True:
            raise RuntimePreflightError(
                f"{config_id}: OFAT effective semantics invalid."
            )

        if effective.get("validation_only") is not True:
            raise RuntimePreflightError(
                f"{config_id}: effective validation_only != true."
            )

        if effective.get("final_test_eligible") is not False:
            raise RuntimePreflightError(
                f"{config_id}: effective final-test eligibility invalid."
            )

        runtime_effective[config_id] = effective

    expected_identity_count = 0

    for row in rows:
        expected = build_ablation_expected_identity(
            REPO_ROOT,
            row,
        )

        if expected.get("final_test_accessed") is not False:
            raise RuntimePreflightError(
                "EXPECTED_IDENTITY_FINAL_TEST_ACCESS_INVALID:"
                + str(row["ablation_run_id"])
            )

        if expected.get("validation_only") is not True:
            raise RuntimePreflightError(
                "EXPECTED_IDENTITY_VALIDATION_ONLY_INVALID:"
                + str(row["ablation_run_id"])
            )

        if expected.get("final_test_eligible") is not False:
            raise RuntimePreflightError(
                "EXPECTED_IDENTITY_FINAL_TEST_ELIGIBILITY_INVALID:"
                + str(row["ablation_run_id"])
            )

        expected_identity_count += 1

    qpseudo_cfg = Config.fromfile(
        str(QPSEUDO_CONFIG)
    )

    qpseudo_initial_thr = float(
        qpseudo_cfg.get("pseudo_label_initial_score_thr")
    )
    qpseudo_rpn_thr = float(
        qpseudo_cfg.get("rpn_pseudo_thr")
    )
    qpseudo_cls_thr = float(
        qpseudo_cfg.get("cls_pseudo_thr")
    )
    qpseudo_comparator = str(
        qpseudo_cfg.get("acceptance_comparator")
    )
    qpseudo_checkpoint_role = str(
        qpseudo_cfg.get("checkpoint_role")
    )
    qpseudo_pseudo_set = str(
        qpseudo_cfg.get("pseudo_set")
    )

    if abs(qpseudo_initial_thr - 0.50) > 1e-12:
        raise RuntimePreflightError(
            "QPSEUDO_INITIAL_THRESHOLD_MISMATCH"
        )

    if abs(qpseudo_rpn_thr - 0.90) > 1e-12:
        raise RuntimePreflightError(
            "QPSEUDO_RPN_THRESHOLD_MISMATCH"
        )

    if abs(qpseudo_cls_thr - 0.90) > 1e-12:
        raise RuntimePreflightError(
            "QPSEUDO_CLS_THRESHOLD_MISMATCH"
        )

    if qpseudo_comparator != "STRICT_GREATER":
        raise RuntimePreflightError(
            "QPSEUDO_ACCEPTANCE_COMPARATOR_MISMATCH"
        )

    if qpseudo_checkpoint_role != "LAST_EMA_TEACHER":
        raise RuntimePreflightError(
            "QPSEUDO_CHECKPOINT_ROLE_MISMATCH"
        )

    if qpseudo_pseudo_set != "accepted_rcnn_classification":
        raise RuntimePreflightError(
            "QPSEUDO_PSEUDO_SET_MISMATCH"
        )

    current_snapshot = _current_snapshot(REPO_ROOT)

    current_runtime_identity = str(
        current_snapshot.get(
            "runtime_identity_sha256",
            "",
        )
    ).strip().lower()

    if SHA256_RE.fullmatch(current_runtime_identity) is None:
        raise RuntimePreflightError(
            "CURRENT_RUNTIME_IDENTITY_INVALID"
        )

    main_run_ids = sorted(
        {
            str(row["main_run_id"]).strip()
            for row in rows
        }
    )

    if len(main_run_ids) != 10:
        raise RuntimePreflightError(
            f"Expected 10 same-seed Main runs; observed={len(main_run_ids)}"
        )

    main_runtime_matches = 0

    for main_run_id in main_run_ids:
        attempt_dir, status = _official_main_attempt(
            main_run_id
        )

        environment_path = (
            attempt_dir
            / "environment_reference.json"
        )

        if not environment_path.is_file():
            raise RuntimePreflightError(
                "MAIN_ENVIRONMENT_REFERENCE_MISSING:"
                + main_run_id
            )

        main_runtime_identity = (
            _runtime_identity_from_environment(
                environment_path
            )
        )

        if main_runtime_identity != current_runtime_identity:
            raise RuntimePreflightError(
                "ABLATION_OFFICIAL_RUNTIME_IDENTITY_MISMATCH:"
                + main_run_id
            )

        if (
            int(status.get("training_seed"))
            != int(
                next(
                    row["training_seed"]
                    for row in rows
                    if str(row["main_run_id"]).strip()
                    == main_run_id
                )
            )
        ):
            raise RuntimePreflightError(
                "SAME_SEED_MAIN_BINDING_INVALID:"
                + main_run_id
            )

        main_runtime_matches += 1

    print("RUN_TYPE=ABLATION")
    print("CONDITION_ROLE=ABLATION")
    print("INVENTORY_ROW_COUNT=90")
    print("NON_MAIN_CONFIG_COUNT=9")
    print("SEEDS_PER_CONFIG=10")
    print(f"CONFIG_IMPORT_COUNT={len(runtime_effective)}")
    print(
        "RUNTIME_EXPECTED_IDENTITY_COUNT="
        f"{expected_identity_count}"
    )
    print(
        "SAME_SEED_MAIN_RUN_COUNT="
        f"{len(main_run_ids)}"
    )
    print(
        "OFFICIAL_MAIN_RUNTIME_MATCH_COUNT="
        f"{main_runtime_matches}"
    )
    print("MAIN_CONFIG_IMPORTED=TRUE")
    print("MAIN_CONFIG_SEPARATE_FROM_ABLATION_FAMILY_EXTRACTOR=TRUE")
    print("NON_MAIN_EFFECTIVE_FIELDS_EXTRACTED=TRUE")
    print("QPSEUDO_FIXED_MEASUREMENT_THRESHOLD=0.90")
    print("QPSEUDO_VARIANT_AWARE_THRESHOLD=FALSE")
    print("VALIDATION_ONLY=TRUE")
    print("FINAL_TEST_ELIGIBLE=FALSE")
    print("TEST_ACCESS_AUTHORIZED=FALSE")
    print("FINAL_TEST_ACCESSED=FALSE")
    print("TRAINING_EXECUTED=FALSE")
    print("FINAL_TEST_ACCESS=FALSE")
    print("S9_11_FROZEN_RUNTIME_PREFLIGHT=PASS")

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())

    except RuntimePreflightError as exc:
        print(
            f"S9_11_FROZEN_RUNTIME_PREFLIGHT_ERROR={exc}",
            file=sys.stderr,
        )
        raise SystemExit(1)