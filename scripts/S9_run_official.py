"""S9 single inventory-driven official training launcher.

One launcher serves all official Main/reference rows.
S9.02 integrates the exact fail-fast pre-run assertion path.
Actual official execution remains subject to later S9 runtime,
backup/recovery, qualification, and canary gates.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any


INVENTORY_RELATIVE_PATH = Path(
    "artifacts/results/official_run_inventory.csv"
)

OFFICIAL_RUN_ROOT = Path(
    "artifacts/runs/official"
)

INITIAL_ATTEMPT_ID = "attempt_001"


class OfficialLauncherError(RuntimeError):
    """Raised when an official launch must be blocked."""


def _parse_bool(value: Any) -> bool:
    normalized = str(value).strip().lower()

    if normalized == "true":
        return True

    if normalized == "false":
        return False

    raise OfficialLauncherError(
        f"Invalid boolean value: {value!r}"
    )


def _canonical_architecture(value: str) -> str:
    mapping = {
        "R50": "R50-FPN",
        "R50-FPN": "R50-FPN",
        "SWIN": "Swin-T-FPN",
        "Swin": "Swin-T-FPN",
        "Swin-T": "Swin-T-FPN",
        "Swin-T-FPN": "Swin-T-FPN",
    }

    key = str(value).strip()

    if key not in mapping:
        raise OfficialLauncherError(
            f"Unsupported inventory architecture: {value!r}"
        )

    return mapping[key]


def _canonical_budget(value: str) -> str:
    mapping = {
        "1%": "1pct",
        "1pct": "1pct",
        "5%": "5pct",
        "5pct": "5pct",
        "10%": "10pct",
        "10pct": "10pct",
        "20%": "20pct",
        "20pct": "20pct",
        "100%": "100pct",
        "100pct": "100pct",
    }

    key = str(value).strip()

    if key not in mapping:
        raise OfficialLauncherError(
            f"Unsupported inventory budget: {value!r}"
        )

    return mapping[key]


def _seed_index(value: str) -> int:
    try:
        seed_index = int(str(value).strip())
    except ValueError as exc:
        raise OfficialLauncherError(
            f"Invalid seed_index: {value!r}"
        ) from exc

    if seed_index < 1 or seed_index > 10:
        raise OfficialLauncherError(
            f"seed_index out of range: {seed_index}"
        )

    return seed_index


def _training_seed(value: str) -> int:
    try:
        return int(str(value).strip())
    except ValueError as exc:
        raise OfficialLauncherError(
            f"Invalid training_seed: {value!r}"
        ) from exc


def _load_inventory(
    repo_root: Path,
) -> list[dict[str, str]]:
    path = repo_root / INVENTORY_RELATIVE_PATH

    if not path.is_file():
        raise OfficialLauncherError(
            f"Official inventory missing: {path}"
        )

    with path.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as handle:
        rows = list(csv.DictReader(handle))

    if len(rows) != 180:
        raise OfficialLauncherError(
            "Official inventory must contain exactly "
            f"180 rows; observed={len(rows)}"
        )

    run_ids = [
        str(row.get("run_id", "")).strip()
        for row in rows
    ]

    if any(not run_id for run_id in run_ids):
        raise OfficialLauncherError(
            "Official inventory contains empty run_id."
        )

    if len(set(run_ids)) != len(run_ids):
        raise OfficialLauncherError(
            "Official inventory contains duplicate run_id."
        )

    return rows


def _select_run(
    rows: list[dict[str, str]],
    run_id: str,
) -> dict[str, str]:
    matches = [
        row
        for row in rows
        if str(row["run_id"]).strip() == run_id
    ]

    if len(matches) != 1:
        raise OfficialLauncherError(
            f"Expected exactly one inventory row for "
            f"{run_id}; observed={len(matches)}"
        )

    return matches[0]


def _select_scope(
    rows: list[dict[str, str]],
    scope: str,
) -> list[dict[str, str]]:
    if scope != "sup-low-label":
        raise OfficialLauncherError(
            f"Unsupported official scope: {scope!r}"
        )

    selected = [
        row
        for row in rows
        if str(row["method"]).strip() == "SUP"
        and str(row["condition_role"]).strip()
        == "MAIN_LOW_LABEL"
    ]

    if len(selected) != 80:
        raise OfficialLauncherError(
            "SUP low-label scope must contain exactly "
            f"80 rows; observed={len(selected)}"
        )

    if any(
        _canonical_budget(row["budget"]) == "100pct"
        for row in selected
    ):
        raise OfficialLauncherError(
            "SUP low-label scope contains 100pct row."
        )

    return selected


def _inventory_selectors(
    row: dict[str, str],
) -> dict[str, Any]:
    method = str(row["method"]).strip()
    architecture = _canonical_architecture(
        row["architecture"]
    )
    budget = _canonical_budget(
        row["budget"]
    )
    seed_index = _seed_index(
        row["seed_index"]
    )
    training_seed = _training_seed(
        row["training_seed"]
    )
    condition_role = str(
        row["condition_role"]
    ).strip()

    if method not in {"SUP", "SSL"}:
        raise OfficialLauncherError(
            f"Invalid official method: {method!r}"
        )

    if condition_role not in {
        "MAIN_LOW_LABEL",
        "SUP_REFERENCE",
    }:
        raise OfficialLauncherError(
            "Invalid Main/reference condition_role: "
            f"{condition_role!r}"
        )

    if condition_role == "SUP_REFERENCE":
        if method != "SUP" or budget != "100pct":
            raise OfficialLauncherError(
                "SUP_REFERENCE must be SUP/100pct."
            )
    else:
        if budget == "100pct":
            raise OfficialLauncherError(
                "100pct cannot use MAIN_LOW_LABEL."
            )

    return {
        "method": method,
        "architecture": architecture,
        "budget": budget,
        "seed_index": seed_index,
        "training_seed": training_seed,
        "condition_role": condition_role,
    }


def _assert_inventory_state(
    row: dict[str, str],
) -> None:
    if not _parse_bool(row["expected"]):
        raise OfficialLauncherError(
            "Selected official row is not expected."
        )

    if _parse_bool(row["training_completed"]):
        raise OfficialLauncherError(
            "Selected official row is already completed."
        )

    if _parse_bool(row["training_valid"]):
        raise OfficialLauncherError(
            "Incomplete row cannot already be training_valid."
        )

    if _parse_bool(row["eligible_for_final_test"]):
        raise OfficialLauncherError(
            "Pre-training row cannot already be "
            "eligible_for_final_test."
        )

    if (
        str(row["final_evaluation_status"]).strip()
        != "NOT_AUTHORIZED"
    ):
        raise OfficialLauncherError(
            "Final-test firewall state is not "
            "NOT_AUTHORIZED."
        )


def _build_pre_run_assertions(
    repo_root: Path,
    row: dict[str, str],
) -> tuple[dict[str, Any], dict[str, Any]]:
    from src.utils.pre_run_guardrails import (
        assert_pre_run_identity,
        build_expected_pre_run_identity,
    )

    selectors = _inventory_selectors(row)

    expected = build_expected_pre_run_identity(
        repo_root,
        method=selectors["method"],
        architecture=selectors["architecture"],
        budget=selectors["budget"],
        seed_index=selectors["seed_index"],
    )

    observed = deepcopy(expected)

    # These values come from the selected inventory row,
    # not from the expected object.
    observed["method"] = selectors["method"]
    observed["architecture"] = selectors["architecture"]
    observed["budget"] = selectors["budget"]
    observed["seed_index"] = selectors["seed_index"]
    observed["training_seed"] = selectors["training_seed"]
    observed["condition_role"] = selectors["condition_role"]

    report = assert_pre_run_identity(
        repo_root,
        observed,
    )

    payload = {
        "schema_version": "1.1",
        "stage": "S9.02",
        "run_id": str(row["run_id"]).strip(),
        "attempt_id": INITIAL_ATTEMPT_ID,
        "launch_status": report["launch_status"],
        "all_checks_pass": report["all_checks_pass"],
        "inventory_identity": selectors,
        "expected_identity": expected,
        "observed_identity": observed,
        "guardrail_report": report,
        "final_test_accessed": False,
    }

    return payload, expected


def _attempt_directory(
    repo_root: Path,
    run_id: str,
) -> Path:
    return (
        repo_root
        / OFFICIAL_RUN_ROOT
        / run_id
        / INITIAL_ATTEMPT_ID
    )


def _write_json(
    path: Path,
    payload: dict[str, Any],
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    path.write_text(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )


def _configure_mmdetection(
    repo_root: Path,
    expected: dict[str, Any],
    attempt_dir: Path,
):
    # Runtime imports are intentionally deferred so local
    # S9.02 verification does not require MMEngine/MMDetection.
    from mmengine.config import Config

    cfg = Config.fromfile(
        str(
            repo_root
            / expected["executable_config_path"]
        )
    )

    if cfg.get("test_dataloader", None) is not None:
        raise OfficialLauncherError(
            "Official training config exposes "
            "test_dataloader."
        )

    if cfg.get("test_evaluator", None) is not None:
        raise OfficialLauncherError(
            "Official training config exposes "
            "test_evaluator."
        )

    cfg.work_dir = str(attempt_dir)

    cfg.randomness = {
        "seed": expected["training_seed"],
        "diff_rank_seed": False,
        "deterministic": True,
    }

    return cfg


def _execute_one(
    repo_root: Path,
    row: dict[str, str],
) -> None:
    run_id = str(row["run_id"]).strip()

    attempt_dir = _attempt_directory(
        repo_root,
        run_id,
    )

    if attempt_dir.exists():
        raise OfficialLauncherError(
            "Initial official attempt directory "
            f"already exists: {attempt_dir}"
        )

    pre_run, expected = _build_pre_run_assertions(
        repo_root,
        row,
    )

    if (
        not pre_run["all_checks_pass"]
        or pre_run["launch_status"] != "PASS"
    ):
        raise OfficialLauncherError(
            "Pre-run assertions did not PASS."
        )

    cfg = _configure_mmdetection(
        repo_root,
        expected,
        attempt_dir,
    )

    # Artifact contract requires pre_run_assertions.json
    # in the official attempt directory immediately
    # before the training launch.
    _write_json(
        attempt_dir / "pre_run_assertions.json",
        pre_run,
    )

    from mmengine.runner import Runner

    runner = Runner.from_cfg(cfg)
    runner.train()


def _verify_one(
    repo_root: Path,
    row: dict[str, str],
) -> dict[str, Any]:
    pre_run, _ = _build_pre_run_assertions(
        repo_root,
        row,
    )

    return pre_run


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Single inventory-driven S9 official "
            "training launcher."
        )
    )

    selector = parser.add_mutually_exclusive_group(
        required=True,
    )

    selector.add_argument(
        "--run-id",
        help=(
            "Exact run_id from "
            "official_run_inventory.csv."
        ),
    )

    selector.add_argument(
        "--scope",
        choices=("sup-low-label",),
        help=(
            "Select the exact 80 SUP MAIN_LOW_LABEL "
            "official inventory rows."
        ),
    )

    parser.add_argument(
        "--execute",
        action="store_true",
        help=(
            "Launch MMDetection only after exact "
            "pre-run assertions PASS. "
            "Do not use until later S9 execution "
            "gates authorize the requested run."
        ),
    )

    args = parser.parse_args()

    repo_root = (
        Path(__file__).resolve().parents[1]
    )

    repo_root_text = str(repo_root)

    if repo_root_text not in sys.path:
        sys.path.insert(
            0,
            repo_root_text,
        )

    try:
        rows = _load_inventory(repo_root)

        if args.run_id is not None:
            row = _select_run(
                rows,
                args.run_id,
            )

            _assert_inventory_state(row)

            if args.execute:
                _execute_one(
                    repo_root,
                    row,
                )
                print(
                    "OFFICIAL_LAUNCH_RESULT="
                    "TRAIN_RETURNED"
                )
            else:
                pre_run = _verify_one(
                    repo_root,
                    row,
                )

                print(
                    "RUN_ID="
                    + str(row["run_id"]).strip()
                )
                print(
                    "LAUNCH_STATUS="
                    + str(pre_run["launch_status"])
                )
                print(
                    "ALL_CHECKS_PASS="
                    + str(pre_run["all_checks_pass"])
                )
                print(
                    "EXECUTION_REQUESTED=False"
                )
                print(
                    "MMDETECTION_TRAINING_CALLED=False"
                )
        else:
            scoped_rows = _select_scope(
                rows,
                args.scope,
            )

            for row in scoped_rows:
                _assert_inventory_state(row)

            if args.execute:
                for row in scoped_rows:
                    _execute_one(
                        repo_root,
                        row,
                    )

                print(
                    "OFFICIAL_SCOPE_RESULT="
                    "TRAIN_RETURNED"
                )
                print(
                    "SCOPE="
                    + str(args.scope)
                )
                print(
                    "SCOPE_ROW_COUNT="
                    + str(len(scoped_rows))
                )
            else:
                verify_pass_count = 0

                for row in scoped_rows:
                    pre_run = _verify_one(
                        repo_root,
                        row,
                    )

                    if (
                        not pre_run["all_checks_pass"]
                        or pre_run["launch_status"] != "PASS"
                    ):
                        raise OfficialLauncherError(
                            "Scoped pre-run verification "
                            "failed for "
                            + str(row["run_id"]).strip()
                        )

                    verify_pass_count += 1

                print(
                    "SCOPE="
                    + str(args.scope)
                )
                print(
                    "SCOPE_ROW_COUNT="
                    + str(len(scoped_rows))
                )
                print(
                    "VERIFY_PASS_COUNT="
                    + str(verify_pass_count)
                )
                print(
                    "EXECUTION_REQUESTED=False"
                )
                print(
                    "MMDETECTION_TRAINING_CALLED=False"
                )

        return 0

    except Exception as exc:
        print(
            "LAUNCH_STATUS=BLOCKED"
        )
        print(
            "ERROR="
            + f"{type(exc).__name__}: {exc}"
        )
        print(
            "MMDETECTION_TRAINING_CALLED=False"
        )
        return 2


if __name__ == "__main__":
    sys.exit(main())
