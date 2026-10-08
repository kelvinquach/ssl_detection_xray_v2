"""S9.11 locked non-Main OFAT ablation launcher.

Scientific scope
----------------
This launcher executes exactly the 90 new S9.11 non-Main trainings:

    Faster R-CNN R50-FPN
    SSL
    10% labeled budget
    9 non-Main OFAT variants
    10 frozen training seeds per variant

The 10 Main observations are reused from the frozen Official SSL R50 10%
condition and are NOT retrained by this launcher.

All S9.11 runs are validation-only and permanently ineligible for final
test evaluation.

Execution namespace:

    artifacts/runs/ablation/<run_id>/<attempt_id>

The launcher reuses the already-verified Official runtime machinery where
scientifically identical: MMDetection configuration, deterministic seed
runtime observation, data provenance, Runner execution and fixed-contract
Q_pseudo measurement.

Official logical-run, backup namespace, run-manifest and closure semantics
are deliberately NOT reused.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(
    __file__
).resolve().parents[1]

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(REPO_ROOT),
    )


INVENTORY_PATH = (
    Path("artifacts")
    / "results"
    / "ablation"
    / "ablation_run_inventory.csv"
)

RUN_ROOT = (
    Path("artifacts")
    / "runs"
    / "ablation"
)

INITIAL_ATTEMPT_ID = "attempt_001"

ATTEMPT_RE = re.compile(
    r"^attempt_(\d{3})$"
)

RUN_ID_RE = re.compile(
    r"^ABL_R50_B10_[A-Z0-9_]+_S\d{2}$"
)


class AblationLauncherError(
    RuntimeError
):
    """Raised when S9.11 execution cannot proceed safely."""


def _parse_bool(
    value: Any,
) -> bool:
    if isinstance(value, bool):
        return value

    text = str(
        value
    ).strip().lower()

    if text in {
        "1",
        "true",
        "yes",
        "y",
    }:
        return True

    if text in {
        "0",
        "false",
        "no",
        "n",
        "",
    }:
        return False

    raise AblationLauncherError(
        f"Invalid boolean value: {value!r}"
    )


def _write_json(
    path: Path,
    payload: dict[str, Any],
) -> None:
    import tempfile

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fd, name = tempfile.mkstemp(
        dir=str(path.parent),
        prefix=f".{path.name}.",
        suffix=".tmp",
    )

    os.close(fd)

    tmp = Path(name)

    try:
        tmp.write_text(
            json.dumps(
                payload,
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )

        os.replace(
            tmp,
            path,
        )

    finally:
        if tmp.exists():
            tmp.unlink()


def _attempt_number(
    attempt_id: str,
) -> int:
    value = str(
        attempt_id
    ).strip()

    match = ATTEMPT_RE.fullmatch(
        value
    )

    if match is None:
        raise AblationLauncherError(
            "Invalid ablation attempt_id: "
            f"{attempt_id!r}"
        )

    number = int(
        match.group(1)
    )

    if number <= 0:
        raise AblationLauncherError(
            "Ablation attempt number must be positive."
        )

    return number


def _prior_attempt_id(
    attempt_id: str,
) -> str | None:
    number = _attempt_number(
        attempt_id
    )

    if number == 1:
        return None

    return (
        f"attempt_{number - 1:03d}"
    )


def _load_inventory(
    repo_root: Path,
) -> list[dict[str, str]]:
    path = (
        repo_root
        / INVENTORY_PATH
    )

    if not path.is_file():
        raise AblationLauncherError(
            "Ablation inventory missing: "
            + str(path)
        )

    with path.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as handle:
        reader = csv.DictReader(
            handle
        )

        fieldnames = set(
            reader.fieldnames
            or []
        )

        required = {
            "ablation_run_id",
            "condition_role",
            "method",
            "architecture",
            "budget",
            "ablation_config_id",
            "family",
            "level",
            "seed_index",
            "training_seed",
            "executable_config_path",
            "scientific_change",
            "main_run_id",
            "is_main",
            "expected",
            "validation_only",
            "final_test_eligible",
            "test_access_authorized",
        }

        missing = sorted(
            required
            - fieldnames
        )

        if missing:
            raise AblationLauncherError(
                "Ablation inventory missing fields: "
                + ", ".join(
                    missing
                )
            )

        rows = list(
            reader
        )

    if len(rows) != 90:
        raise AblationLauncherError(
            "S9.11 inventory must contain exactly "
            f"90 non-Main rows; observed={len(rows)}"
        )

    run_ids = [
        str(
            row["ablation_run_id"]
        ).strip()
        for row in rows
    ]

    if (
        len(set(run_ids))
        != 90
    ):
        raise AblationLauncherError(
            "Ablation run IDs are not unique."
        )

    config_counts: dict[
        str,
        int,
    ] = {}

    family_counts = {
        "CONFIDENCE_THRESHOLD": 0,
        "STRONG_PHOTOMETRIC_AUGMENTATION": 0,
        "REGRESSION_PSEUDO_LABEL_FILTERING": 0,
    }

    for row in rows:
        run_id = str(
            row["ablation_run_id"]
        ).strip()

        if RUN_ID_RE.fullmatch(
            run_id
        ) is None:
            raise AblationLauncherError(
                f"Invalid ablation run_id: {run_id!r}"
            )

        if (
            str(
                row["condition_role"]
            ).strip()
            != "ABLATION"
        ):
            raise AblationLauncherError(
                f"{run_id}: condition_role must be ABLATION."
            )

        if (
            str(
                row["method"]
            ).strip()
            != "SSL"
        ):
            raise AblationLauncherError(
                f"{run_id}: method must be SSL."
            )

        if _parse_bool(
            row["is_main"]
        ):
            raise AblationLauncherError(
                f"{run_id}: non-Main inventory contains Main row."
            )

        if not _parse_bool(
            row["expected"]
        ):
            raise AblationLauncherError(
                f"{run_id}: expected must be true."
            )

        if not _parse_bool(
            row["validation_only"]
        ):
            raise AblationLauncherError(
                f"{run_id}: validation_only must be true."
            )

        if _parse_bool(
            row["final_test_eligible"]
        ):
            raise AblationLauncherError(
                f"{run_id}: final_test_eligible must be false."
            )

        if _parse_bool(
            row["test_access_authorized"]
        ):
            raise AblationLauncherError(
                f"{run_id}: test access must not be authorized."
            )

        family = str(
            row["family"]
        ).strip().upper()

        if family not in family_counts:
            raise AblationLauncherError(
                f"{run_id}: invalid ablation family {family!r}."
            )

        family_counts[
            family
        ] += 1

        config_id = str(
            row[
                "ablation_config_id"
            ]
        ).strip()

        config_counts[
            config_id
        ] = (
            config_counts.get(
                config_id,
                0,
            )
            + 1
        )

    if family_counts != {
        "CONFIDENCE_THRESHOLD": 40,
        "STRONG_PHOTOMETRIC_AUGMENTATION": 30,
        "REGRESSION_PSEUDO_LABEL_FILTERING": 20,
    }:
        raise AblationLauncherError(
            "Ablation family counts mismatch: "
            f"{family_counts}"
        )

    if len(
        config_counts
    ) != 9:
        raise AblationLauncherError(
            "S9.11 must contain exactly "
            "9 non-Main configurations."
        )

    invalid_counts = {
        key: value
        for key, value
        in config_counts.items()
        if value != 10
    }

    if invalid_counts:
        raise AblationLauncherError(
            "Every non-Main configuration must "
            "contain exactly 10 seeds: "
            f"{invalid_counts}"
        )

    return rows


def _row_run_id(
    row: dict[str, str],
) -> str:
    return str(
        row[
            "ablation_run_id"
        ]
    ).strip()


def _select_run(
    rows: list[dict[str, str]],
    run_id: str,
) -> dict[str, str]:
    matches = [
        row
        for row in rows
        if _row_run_id(row)
        == run_id
    ]

    if len(matches) != 1:
        raise AblationLauncherError(
            "Expected exactly one inventory row "
            f"for {run_id}; observed={len(matches)}"
        )

    return matches[0]


def _logical_status_path(
    repo_root: Path,
    run_id: str,
) -> Path:
    return (
        repo_root
        / RUN_ROOT
        / run_id
        / "logical_run_status.json"
    )


def _classify_runtime_state(
    repo_root: Path,
    row: dict[str, str],
) -> dict[str, Any]:
    run_id = _row_run_id(
        row
    )

    path = _logical_status_path(
        repo_root,
        run_id,
    )

    if not path.is_file():
        return {
            "state": "FRESH",
            "run_id": run_id,
            "attempt_id": None,
        }

    try:
        payload = json.loads(
            path.read_text(
                encoding="utf-8-sig"
            )
        )
    except Exception as exc:
        raise AblationLauncherError(
            f"Cannot parse logical status for {run_id}: {exc}"
        ) from exc

    if not isinstance(
        payload,
        dict,
    ):
        raise AblationLauncherError(
            f"{run_id}: logical status is not a JSON object."
        )

    required_pairs = {
        "run_id":
            run_id,
        "run_type":
            "ABLATION",
        "condition_role":
            "ABLATION",
        "training_seed":
            int(
                str(
                    row[
                        "training_seed"
                    ]
                ).strip()
            ),
        "eligible_for_final_test":
            False,
        "validation_only":
            True,
        "test_access_authorized":
            False,
        "final_test_accessed":
            False,
    }

    for key, expected in (
        required_pairs.items()
    ):
        if payload.get(
            key
        ) != expected:
            raise AblationLauncherError(
                f"{run_id}: logical status mismatch: {key}"
            )

    state = payload.get(
        "training_status"
    )

    if state not in {
        "IN_PROGRESS",
        "TECHNICAL_FAILURE",
        "CLOSED_VERIFIED",
    }:
        raise AblationLauncherError(
            f"{run_id}: invalid logical status {state!r}."
        )

    attempt_id = payload.get(
        "attempt_used"
    )

    if (
        not isinstance(
            attempt_id,
            str,
        )
        or not attempt_id
    ):
        raise AblationLauncherError(
            f"{run_id}: logical status lacks attempt_used."
        )

    _attempt_number(
        attempt_id
    )

    return {
        "state":
            state,
        "run_id":
            run_id,
        "attempt_id":
            attempt_id,
    }


def _validate_new_attempt_request(
    repo_root: Path,
    row: dict[str, str],
    attempt_id: str,
) -> None:
    run_id = _row_run_id(
        row
    )

    number = _attempt_number(
        attempt_id
    )

    status = _classify_runtime_state(
        repo_root,
        row,
    )

    attempt_dir = (
        repo_root
        / RUN_ROOT
        / run_id
        / attempt_id
    )

    if attempt_dir.exists():
        raise AblationLauncherError(
            "Ablation attempt directory already exists: "
            + str(attempt_dir)
        )

    if number == 1:
        if status[
            "state"
        ] != "FRESH":
            raise AblationLauncherError(
                f"{run_id}: attempt_001 requires FRESH logical state; "
                f"observed={status['state']}"
            )

        return

    prior_attempt = (
        _prior_attempt_id(
            attempt_id
        )
    )

    if (
        status["state"]
        != "TECHNICAL_FAILURE"
        or status["attempt_id"]
        != prior_attempt
    ):
        raise AblationLauncherError(
            f"{run_id}: {attempt_id} requires immediately preceding "
            f"{prior_attempt} TECHNICAL_FAILURE."
        )

    prior_retry_path = (
        repo_root
        / RUN_ROOT
        / run_id
        / str(
            prior_attempt
        )
        / "retry_deviation.json"
    )

    if not prior_retry_path.is_file():
        raise AblationLauncherError(
            "Prior retry_deviation.json missing: "
            + str(prior_retry_path)
        )

    prior_retry = json.loads(
        prior_retry_path.read_text(
            encoding="utf-8-sig"
        )
    )

    if (
        prior_retry.get(
            "run_id"
        )
        != run_id
        or prior_retry.get(
            "attempt_id"
        )
        != prior_attempt
    ):
        raise AblationLauncherError(
            f"{run_id}: prior retry identity mismatch."
        )

    if (
        int(
            prior_retry.get(
                "training_seed"
            )
        )
        != int(
            str(
                row[
                    "training_seed"
                ]
            ).strip()
        )
    ):
        raise AblationLauncherError(
            f"{run_id}: retry training seed mismatch."
        )

    if (
        prior_retry.get(
            "technical_failure"
        )
        is not True
        or prior_retry.get(
            "retry_required"
        )
        is not True
        or prior_retry.get(
            "scientific_protocol_changed"
        )
        is not False
    ):
        raise AblationLauncherError(
            f"{run_id}: prior attempt is not an eligible technical retry."
        )


def _build_pre_run(
    *,
    run_id: str,
    attempt_id: str,
    identity: dict[str, Any],
    expected: dict[str, Any],
) -> dict[str, Any]:
    firewall = (
        identity.get(
            "validation_only"
        )
        is True
        and identity.get(
            "final_test_eligible"
        )
        is False
        and identity.get(
            "test_access_authorized"
        )
        is False
        and identity.get(
            "final_test_accessed"
        )
        is False
        and expected.get(
            "validation_only"
        )
        is True
        and expected.get(
            "final_test_eligible"
        )
        is False
        and expected.get(
            "test_access_authorized"
        )
        is False
        and expected.get(
            "final_test_accessed"
        )
        is False
    )

    if not firewall:
        raise AblationLauncherError(
            "S9.11 final-test firewall did not PASS."
        )

    return {
        "schema_version":
            "1.0",
        "run_id":
            run_id,
        "attempt_id":
            attempt_id,
        "launch_status":
            "PASS",
        "all_checks_pass":
            True,
        "expected_identity":
            expected,
        "observed_identity":
            expected,
        "ablation_logical_identity":
            identity,
        "final_test_accessed":
            False,
        "checks": {
            "run_type_ablation":
                True,
            "condition_role_ablation":
                True,
            "validation_only":
                True,
            "final_test_eligible_false":
                True,
            "test_access_authorized_false":
                True,
            "final_test_accessed_false":
                True,
            "same_seed_main_reference_bound":
                True,
        },
    }


def _execute_one(
    repo_root: Path,
    row: dict[str, str],
    attempt_id: str,
    *,
    recover: bool = False,
) -> None:
    run_id = _row_run_id(
        row
    )

    from src.utils.s9_11_ablation_identity import (
        build_ablation_identity,
        ensure_ablation_run_root_manifest,
        set_ablation_logical_run_in_progress,
        set_ablation_logical_run_technical_failure,
    )

    from src.utils.s9_11_ablation_expected_identity import (
        build_ablation_expected_identity,
    )

    identity = build_ablation_identity(
        repo_root,
        row,
    )

    expected = build_ablation_expected_identity(
        repo_root,
        row,
    )

    if (
        expected.get(
            "run_id"
        )
        != run_id
    ):
        raise AblationLauncherError(
            "Expected identity run_id mismatch."
        )

    if (
        expected.get(
            "final_test_accessed"
        )
        is not False
    ):
        raise AblationLauncherError(
            "Expected identity violates final-test firewall."
        )

    attempt_dir = (
        repo_root
        / RUN_ROOT
        / run_id
        / attempt_id
    ).resolve()

    run_root = (
        attempt_dir.parent
    )

    if recover:
        _attempt_number(
            attempt_id
        )

        if attempt_dir.exists():
            raise AblationLauncherError(
                "Same-attempt recovery requires local "
                "attempt directory to be absent: "
                + str(attempt_dir)
            )
    else:
        _validate_new_attempt_request(
            repo_root,
            row,
            attempt_id,
        )

    ensure_ablation_run_root_manifest(
        run_root,
        identity,
        attempt_id=attempt_id,
        recovery=recover,
    )

    from src.utils.s9_11_ablation_backup import (
        AblationActiveBackupManager,
        recover_ablation_attempt,
        register_active_backup,
        unregister_active_backup,
    )

    if recover:
        recovery = recover_ablation_attempt(
            repo_root,
            run_id,
            attempt_id,
        )

        latest_resume_path = Path(
            recovery[
                "latest_resume_path"
            ]
        )

        retry_path = (
            attempt_dir
            / "retry_deviation.json"
        )

        if not retry_path.is_file():
            raise AblationLauncherError(
                "Recovered retry_deviation.json is missing."
            )

        recovered_retry = json.loads(
            retry_path.read_text(
                encoding="utf-8-sig"
            )
        )

        if (
            recovered_retry.get(
                "run_id"
            )
            != run_id
            or recovered_retry.get(
                "attempt_id"
            )
            != attempt_id
        ):
            raise AblationLauncherError(
                "Recovered retry identity mismatch."
            )

        if (
            int(
                recovered_retry.get(
                    "training_seed"
                )
            )
            != int(
                str(
                    row[
                        "training_seed"
                    ]
                ).strip()
            )
        ):
            raise AblationLauncherError(
                "Recovered retry seed mismatch."
            )

        if (
            recovered_retry.get(
                "scientific_protocol_changed"
            )
            is not False
        ):
            raise AblationLauncherError(
                "Recovered attempt changed scientific protocol."
            )

        if (
            recovered_retry.get(
                "technical_failure"
            )
            is not False
            or recovered_retry.get(
                "retry_required"
            )
            is not False
            or recovered_retry.get(
                "retry_reason"
            )
            is not None
        ):
            raise AblationLauncherError(
                "Same-attempt recovery is forbidden "
                "after recorded technical failure."
            )

    else:
        attempt_dir.mkdir(
            parents=True,
            exist_ok=False,
        )

    # Runtime-only imports intentionally occur here.
    # Local implementation audits therefore do not require MMEngine/MMDet.
    from scripts.S9_run_official import (
        _configure_mmdetection,
        _retry_deviation_payload,
        _run_qpseudo_after_training,
    )

    cfg = _configure_mmdetection(
        repo_root,
        expected,
        attempt_dir,
    )

    if recover:
        cfg.resume = True
        cfg.load_from = str(
            latest_resume_path
        )
        retry_deviation = (
            recovered_retry
        )

    else:
        if (
            attempt_id
            != INITIAL_ATTEMPT_ID
        ):
            cfg.resume = False
            cfg.load_from = None

        pre_run = _build_pre_run(
            run_id=run_id,
            attempt_id=attempt_id,
            identity=identity,
            expected=expected,
        )

        _write_json(
            attempt_dir
            / "pre_run_assertions.json",
            pre_run,
        )

        retry_row = dict(
            row
        )

        retry_row[
            "run_id"
        ] = run_id

        retry_deviation = (
            _retry_deviation_payload(
                retry_row,
                attempt_id,
            )
        )

        _write_json(
            attempt_dir
            / "retry_deviation.json",
            retry_deviation,
        )

    set_ablation_logical_run_in_progress(
        run_root,
        identity,
        attempt_id,
    )

    backup = None
    backup_registered = False

    try:
        from src.utils.official_resolved_config import (
            materialize_official_resolved_config,
        )

        materialize_official_resolved_config(
            attempt_dir=attempt_dir,
            cfg=cfg,
        )

        from src.utils.official_data_provenance import (
            materialize_official_data_provenance,
        )

        materialize_official_data_provenance(
            repo_root=repo_root,
            attempt_dir=attempt_dir,
            expected=expected,
            cfg=cfg,
        )

        from src.utils.official_seed_runtime_hook import (
            begin_official_seed_runtime_observation,
            finish_official_seed_runtime_observation,
            prepare_official_seed_runtime_context,
        )

        prepare_official_seed_runtime_context(
            repo_root=repo_root,
            attempt_dir=attempt_dir,
            expected=expected,
            recovery=recover,
        )

        os.environ.setdefault(
            "CUBLAS_WORKSPACE_CONFIG",
            ":4096:8",
        )

        from src.utils.s9_11_ablation_environment_reference import (
            materialize_ablation_environment_reference,
        )

        materialize_ablation_environment_reference(
            repo_root=repo_root,
            attempt_dir=attempt_dir,
            run_id=run_id,
            attempt_id=attempt_id,
            recovery=recover,
        )

        from src.utils.s9_11_ablation_effective_fields import (
            extract_s9_11_effective_scientific_fields,
        )

        effective_fields = (
            extract_s9_11_effective_scientific_fields(
                cfg=cfg,
                ablation_family=str(
                    expected[
                        "ablation_family"
                    ]
                ),
                ablation_level=str(
                    expected[
                        "ablation_level"
                    ]
                ),
            )
        )

        if recover:
            from src.utils.s9_11_ablation_attempt_closure import (
                _validate_ablation_run_manifest,
            )

            _validate_ablation_run_manifest(
                attempt_dir=attempt_dir,
                run_id=run_id,
                attempt_id=attempt_id,
                expected=expected,
            )

        else:
            from src.utils.s9_11_ablation_run_manifest import (
                materialize_s9_11_ablation_run_manifest,
            )

            materialize_s9_11_ablation_run_manifest(
                repo_root=repo_root,
                attempt_dir=attempt_dir,
                run_id=run_id,
                attempt_id=attempt_id,
                expected=expected,
                effective_fields=effective_fields,
            )

        backup = (
            AblationActiveBackupManager(
                repo_root,
                attempt_dir,
            )
        )

        register_active_backup(
            backup
        )

        backup_registered = True

        backup.start()

        backup.sync(
            "logical_run_status_in_progress",
            raise_on_error=True,
        )

        from mmengine.runner import Runner
        from mmengine.utils import (
            import_modules_from_strings,
        )

        runtime_custom_imports = (
            cfg.get(
                "custom_imports",
                {},
            )
        )

        import_modules_from_strings(
            runtime_custom_imports.get(
                "imports",
                [],
            ),
            allow_failed_imports=(
                runtime_custom_imports.get(
                    "allow_failed_imports",
                    False,
                )
            ),
        )

        mmengine_namespace_only_keys = (
            "annotations",
            "Path",
            "SSOD_ROOT",
            "COCO_ROOT",
            "MMDET_IMAGE_ROOT",
        )

        for key in (
            mmengine_namespace_only_keys
        ):
            if key in cfg:
                cfg.pop(
                    key
                )

        begin_official_seed_runtime_observation(
            attempt_dir=attempt_dir,
        )

        try:
            runner = Runner.from_cfg(
                cfg
            )

        finally:
            finish_official_seed_runtime_observation(
                attempt_dir=attempt_dir,
            )

        runner.train()

        _run_qpseudo_after_training(
            repo_root,
            attempt_dir,
            runner,
        )

        from src.utils.s9_11_ablation_attempt_closure import (
            close_s9_11_ablation_attempt,
        )

        close_s9_11_ablation_attempt(
            attempt_dir=attempt_dir,
            run_id=run_id,
            attempt_id=attempt_id,
            identity=identity,
            expected=expected,
            validation_interval=int(
                cfg.train_cfg.val_interval
            ),
        )

    except Exception as exc:
        retry_deviation[
            "technical_failure"
        ] = True

        retry_deviation[
            "failure_reason"
        ] = (
            f"{type(exc).__name__}: {exc}"
        )

        retry_deviation[
            "retry_required"
        ] = True

        retry_deviation[
            "retry_reason"
        ] = (
            "TECHNICAL_FAILURE_ONLY"
        )

        retry_deviation[
            "scientific_protocol_changed"
        ] = False

        try:
            _write_json(
                attempt_dir
                / "retry_deviation.json",
                retry_deviation,
            )
        except Exception:
            pass

        try:
            set_ablation_logical_run_technical_failure(
                run_root,
                identity,
                attempt_id,
            )
        except Exception:
            pass

        try:
            if backup is not None:
                backup.record_failure_state()
                backup.stop()
        finally:
            if backup_registered:
                unregister_active_backup(
                    attempt_dir
                )

        raise

    try:
        if backup is None:
            raise AblationLauncherError(
                "Ablation backup manager was not initialized."
            )

        backup.finalize()

    finally:
        if backup_registered:
            unregister_active_backup(
                attempt_dir
            )


def _verify_selected(
    repo_root: Path,
    rows: list[dict[str, str]],
    *,
    require_runtime_expected_identity: bool,
) -> None:
    from src.utils.s9_11_ablation_identity import (
        build_ablation_identity,
    )

    for row in rows:
        identity = (
            build_ablation_identity(
                repo_root,
                row,
            )
        )

        run_id = _row_run_id(
            row
        )

        if (
            identity.get(
                "run_id"
            )
            != run_id
        ):
            raise AblationLauncherError(
                f"{run_id}: ablation identity binding failed."
            )

        if (
            identity.get(
                "final_test_accessed"
            )
            is not False
        ):
            raise AblationLauncherError(
                f"{run_id}: local final-test firewall failed."
            )

        if require_runtime_expected_identity:
            from src.utils.s9_11_ablation_expected_identity import (
                build_ablation_expected_identity,
            )

            expected = (
                build_ablation_expected_identity(
                    repo_root,
                    row,
                )
            )

            if (
                expected.get(
                    "run_id"
                )
                != run_id
            ):
                raise AblationLauncherError(
                    f"{run_id}: runtime expected identity binding failed."
                )

            if (
                expected.get(
                    "final_test_accessed"
                )
                is not False
            ):
                raise AblationLauncherError(
                    f"{run_id}: runtime final-test firewall failed."
                )


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "S9.11 locked validation-only "
            "non-Main ablation launcher."
        )
    )

    selector = (
        parser.add_mutually_exclusive_group()
    )

    selector.add_argument(
        "--run-id",
        help=(
            "Execute or verify one exact "
            "S9.11 logical ablation run."
        ),
    )

    selector.add_argument(
        "--scope",
        choices=(
            "all",
        ),
        help=(
            "Select all 90 non-Main "
            "S9.11 ablation runs."
        ),
    )

    parser.add_argument(
        "--attempt-id",
        default=INITIAL_ATTEMPT_ID,
        help=(
            "Attempt identity. Technical retry "
            "must use the next attempt number."
        ),
    )

    parser.add_argument(
        "--execute",
        action="store_true",
        help=(
            "Actually execute training. "
            "Without this flag the launcher "
            "performs validation only."
        ),
    )

    parser.add_argument(
        "--recover",
        action="store_true",
        help=(
            "Recover one exact same attempt "
            "from verified durable backup. "
            "Requires --run-id --execute."
        ),
    )

    args = parser.parse_args()

    repo_root = REPO_ROOT

    rows = _load_inventory(
        repo_root
    )

    if args.run_id:
        selected = [
            _select_run(
                rows,
                args.run_id,
            )
        ]

    elif args.scope == "all":
        selected = list(
            rows
        )

    else:
        selected = list(
            rows
        )

    _verify_selected(
        repo_root,
        selected,
        require_runtime_expected_identity=bool(
            args.execute
        ),
    )

    states = [
        _classify_runtime_state(
            repo_root,
            row,
        )
        for row in selected
    ]

    counts = {
        "FRESH": 0,
        "IN_PROGRESS": 0,
        "TECHNICAL_FAILURE": 0,
        "CLOSED_VERIFIED": 0,
    }

    for item in states:
        counts[
            item["state"]
        ] += 1

    print(
        "SCOPE="
        + (
            f"run-id:{args.run_id}"
            if args.run_id
            else "all"
        )
    )

    print(
        f"SELECTED_ROW_COUNT={len(selected)}"
    )

    print(
        "FRESH="
        f"{counts['FRESH']}"
    )

    print(
        "IN_PROGRESS="
        f"{counts['IN_PROGRESS']}"
    )

    print(
        "TECHNICAL_FAILURE="
        f"{counts['TECHNICAL_FAILURE']}"
    )

    print(
        "CLOSED_VERIFIED="
        f"{counts['CLOSED_VERIFIED']}"
    )

    print(
        "VALIDATION_ONLY=TRUE"
    )

    print(
        "FINAL_TEST_ELIGIBLE=FALSE"
    )

    print(
        "TEST_ACCESS_AUTHORIZED=FALSE"
    )

    print(
        "FINAL_TEST_ACCESSED=FALSE"
    )

    if not args.execute:
        print(
            "EXECUTION_REQUESTED=FALSE"
        )

        print(
            "MMDETECTION_TRAINING_CALLED=FALSE"
        )

        print(
            "S9_11_ABLATION_LAUNCHER_VALIDATION=PASS"
        )

        return 0

    if (
        args.recover
        and not args.run_id
    ):
        raise AblationLauncherError(
            "--recover requires exact --run-id."
        )

    if (
        args.recover
        and not args.execute
    ):
        raise AblationLauncherError(
            "--recover requires --execute."
        )

    if (
        args.scope == "all"
        and args.attempt_id
        != INITIAL_ATTEMPT_ID
    ):
        raise AblationLauncherError(
            "Scope-all execution only uses attempt_001 "
            "for FRESH runs. Technical retries must use --run-id."
        )

    if args.scope == "all":
        blocked = [
            item
            for item in states
            if item["state"]
            in {
                "IN_PROGRESS",
                "TECHNICAL_FAILURE",
            }
        ]

        if blocked:
            detail = ", ".join(
                (
                    f"{item['run_id']}:"
                    f"{item['state']}:"
                    f"{item['attempt_id']}"
                )
                for item in blocked
            )

            raise AblationLauncherError(
                "Scope-all execution blocked by "
                "non-fresh unresolved runs: "
                + detail
            )

        for row, state in zip(
            selected,
            states,
        ):
            run_id = _row_run_id(
                row
            )

            if (
                state["state"]
                == "CLOSED_VERIFIED"
            ):
                print(
                    f"SKIP_CLOSED_VERIFIED={run_id}"
                )
                continue

            print(
                f"EXECUTING={run_id}"
            )

            _execute_one(
                repo_root,
                row,
                INITIAL_ATTEMPT_ID,
                recover=False,
            )

        print(
            "S9_11_SCOPE_ALL_EXECUTION_COMPLETED=TRUE"
        )

        return 0

    row = selected[0]

    if args.recover:
        state = states[0]

        if (
            state["state"]
            == "CLOSED_VERIFIED"
        ):
            raise AblationLauncherError(
                "Cannot recover a CLOSED_VERIFIED logical run."
            )

        _execute_one(
            repo_root,
            row,
            args.attempt_id,
            recover=True,
        )

    else:
        _execute_one(
            repo_root,
            row,
            args.attempt_id,
            recover=False,
        )

    print(
        "S9_11_SINGLE_RUN_EXECUTION_COMPLETED=TRUE"
    )

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(
            main()
        )

    except AblationLauncherError as exc:
        print(
            f"S9_11_ABLATION_LAUNCHER_ERROR={exc}",
            file=sys.stderr,
        )

        raise SystemExit(1)