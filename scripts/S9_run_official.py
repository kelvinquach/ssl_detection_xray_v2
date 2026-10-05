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
    if scope == "sup-low-label":
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

    if scope == "sup-reference":
        selected = [
            row
            for row in rows
            if str(row["method"]).strip() == "SUP"
            and str(row["condition_role"]).strip()
            == "SUP_REFERENCE"
        ]

        if len(selected) != 20:
            raise OfficialLauncherError(
                "SUP reference scope must contain exactly "
                f"20 rows; observed={len(selected)}"
            )

        if any(
            _canonical_budget(row["budget"]) != "100pct"
            for row in selected
        ):
            raise OfficialLauncherError(
                "SUP reference scope contains non-100pct row."
            )

        return selected

    if scope == "ssl-low-label":
        selected = [
            row
            for row in rows
            if str(row["method"]).strip() == "SSL"
            and str(row["condition_role"]).strip()
            == "MAIN_LOW_LABEL"
        ]

        if len(selected) != 80:
            raise OfficialLauncherError(
                "SSL low-label scope must contain exactly "
                f"80 rows; observed={len(selected)}"
            )

        if any(
            _canonical_budget(row["budget"]) == "100pct"
            for row in selected
        ):
            raise OfficialLauncherError(
                "SSL low-label scope contains 100pct row."
            )

        return selected

    raise OfficialLauncherError(
        f"Unsupported official scope: {scope!r}"
    )


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


def _attempt_number(attempt_id: str) -> int:
    value = str(attempt_id).strip()
    prefix = "attempt_"

    if not value.startswith(prefix):
        raise OfficialLauncherError(
            f"Invalid official attempt_id: {attempt_id!r}"
        )

    suffix = value[len(prefix):]

    if not suffix.isdigit():
        raise OfficialLauncherError(
            f"Invalid official attempt_id: {attempt_id!r}"
        )

    number = int(suffix)

    if number < 1 or f"attempt_{number:03d}" != value:
        raise OfficialLauncherError(
            f"Invalid official attempt_id: {attempt_id!r}"
        )

    return number


def _retry_deviation_payload(
    row: dict[str, str],
    attempt_id: str,
) -> dict[str, Any]:
    return {
        "run_id": str(row["run_id"]).strip(),
        "attempt_id": attempt_id,
        "training_seed": int(
            str(row["training_seed"]).strip()
        ),
        "technical_failure": False,
        "failure_reason": None,
        "retry_required": False,
        "retry_reason": None,
        "same_seed_confirmed": True,
        "scientific_protocol_changed": False,
        "controlled_revision_id": None,
        "researcher_approval": None,
    }


def _validate_attempt_request(
    repo_root: Path,
    row: dict[str, str],
    attempt_id: str,
) -> None:
    attempt_number = _attempt_number(attempt_id)

    if attempt_number == 1:
        return

    run_id = str(row["run_id"]).strip()
    training_seed = int(
        str(row["training_seed"]).strip()
    )
    prior_attempt_id = (
        f"attempt_{attempt_number - 1:03d}"
    )
    prior_retry_path = (
        repo_root
        / OFFICIAL_RUN_ROOT
        / run_id
        / prior_attempt_id
        / "retry_deviation.json"
    )

    if not prior_retry_path.is_file():
        raise OfficialLauncherError(
            "Prior attempt retry_deviation.json "
            f"is missing: {prior_retry_path}"
        )

    prior_retry = json.loads(
        prior_retry_path.read_text(
            encoding="utf-8-sig",
        )
    )

    if prior_retry.get("run_id") != run_id:
        raise OfficialLauncherError(
            "Prior retry record run_id mismatch."
        )

    if (
        prior_retry.get("attempt_id")
        != prior_attempt_id
    ):
        raise OfficialLauncherError(
            "Prior retry record attempt_id mismatch."
        )

    if (
        int(prior_retry.get("training_seed"))
        != training_seed
    ):
        raise OfficialLauncherError(
            "Prior retry record training seed mismatch."
        )

    if prior_retry.get("technical_failure") is not True:
        raise OfficialLauncherError(
            "Prior attempt is not marked technical failure."
        )

    if prior_retry.get("retry_required") is not True:
        raise OfficialLauncherError(
            "Prior attempt is not authorized for retry."
        )

    if prior_retry.get("same_seed_confirmed") is not True:
        raise OfficialLauncherError(
            "Prior retry record lacks same-seed confirmation."
        )

    if (
        prior_retry.get("scientific_protocol_changed")
        is not False
    ):
        raise OfficialLauncherError(
            "Technical retry cannot change scientific protocol."
        )


def _attempt_directory(
    repo_root: Path,
    run_id: str,
    attempt_id: str,
) -> Path:
    _attempt_number(attempt_id)

    return (
        repo_root
        / OFFICIAL_RUN_ROOT
        / run_id
        / attempt_id
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

    if expected["method"] == "SUP":
        evaluation_cfg = Config.fromfile(
            str(
                repo_root
                / "configs/evaluation/s5_02_coco_secondary_evaluator.py"
            )
        )

        custom_imports = cfg.get("custom_imports", {})
        imports = list(custom_imports.get("imports", []))
        protocol_metric_module = "src.metrics.protocol_coco_metric"
        sup_summary_module = "src.utils.sup_training_summary_hook"

        if protocol_metric_module not in imports:
            imports.append(protocol_metric_module)
        if sup_summary_module not in imports:
            imports.append(sup_summary_module)

        cfg.custom_imports = dict(
            imports=imports,
            allow_failed_imports=False,
        )
        cfg.val_dataloader = deepcopy(
            evaluation_cfg.val_dataloader
        )
        cfg.val_evaluator = deepcopy(
            evaluation_cfg.val_evaluator
        )
        cfg.val_cfg = deepcopy(
            evaluation_cfg.val_cfg
        )

        custom_hooks = list(cfg.get("custom_hooks", []))
        budget_hooks = [
            hook
            for hook in custom_hooks
            if hook.get("type") == "ActualUpdateBudgetSchedulerHook"
        ]
        if len(budget_hooks) != 1:
            raise OfficialLauncherError(
                "Official SUP config must contain exactly one "
                "ActualUpdateBudgetSchedulerHook."
            )

        target_updates = budget_hooks[0].get("target_updates")
        if (
            isinstance(target_updates, bool)
            or not isinstance(target_updates, int)
            or target_updates <= 0
        ):
            raise OfficialLauncherError(
                "Official SUP target_updates must be a positive integer."
            )

        if any(
            hook.get("type") == "SupervisedTrainingSummaryHook"
            for hook in custom_hooks
        ):
            raise OfficialLauncherError(
                "Official SUP config unexpectedly already contains "
                "SupervisedTrainingSummaryHook."
            )

        custom_hooks.append(
            dict(
                type="SupervisedTrainingSummaryHook",
                expected_optimizer_updates=int(target_updates),
                effective_labeled_batch=4,
                priority="LOW",
            )
        )
        cfg.custom_hooks = custom_hooks

    custom_imports = cfg.get("custom_imports", {})
    imports = list(custom_imports.get("imports", []))
    seed_runtime_module = "src.utils.official_seed_runtime_hook"
    if seed_runtime_module not in imports:
        imports.append(seed_runtime_module)
    cfg.custom_imports = dict(
        imports=imports,
        allow_failed_imports=False,
    )

    custom_hooks = list(cfg.get("custom_hooks", []))
    if any(
        hook.get("type") == "OfficialSeedRuntimeHook"
        for hook in custom_hooks
    ):
        raise OfficialLauncherError(
            "Official config unexpectedly already contains "
            "OfficialSeedRuntimeHook."
        )
    custom_hooks.append(dict(type="OfficialSeedRuntimeHook"))
    cfg.custom_hooks = custom_hooks

    cfg.work_dir = str(attempt_dir)

    cfg.randomness = {
        "seed": expected["training_seed"],
        "diff_rank_seed": False,
        "deterministic": True,
    }

    return cfg


def _run_qpseudo_after_training(
    repo_root: Path,
    attempt_dir: Path,
    runner,
) -> None:
    """Materialize the three required S9.09 Q_pseudo artifacts for SSL."""
    import hashlib

    import torch
    from mmengine.config import Config

    from src.evaluation.qpseudo import (
        FIXED_VALIDATION_SHA256,
        accepted_rcnn_classification_predictions,
        evaluate_qpseudo_predictions,
        load_last_ema_teacher_checkpoint,
        write_qpseudo_artifacts,
    )

    checkpoint_path = (
        attempt_dir
        / "checkpoints"
        / "last_ema_teacher.pth"
    )
    runtime_val = Path(
        "/workspace/ssod/data/coco/instances_val.json"
    )
    acceptance_config = (
        repo_root
        / "configs/evaluation/s5_06_qpseudo_acceptance.py"
    )
    evaluator_config = (
        repo_root
        / "configs/evaluation/s5_02_coco_secondary_evaluator.py"
    )

    def sha256(file_path: Path) -> str:
        digest = hashlib.sha256()
        with file_path.open("rb") as handle:
            for chunk in iter(
                lambda: handle.read(1024 * 1024),
                b"",
            ):
                digest.update(chunk)
        return digest.hexdigest()

    load_last_ema_teacher_checkpoint(
        runner.model,
        checkpoint_path,
    )

    runner.model.eval()
    data_samples = []
    with torch.no_grad():
        for batch in runner.val_loop.dataloader:
            data_samples.extend(
                runner.model.test_step(batch)
            )

    predictions = accepted_rcnn_classification_predictions(
        data_samples
    )
    evaluator_cfg = Config.fromfile(
        str(evaluator_config)
    ).val_evaluator
    metrics = evaluate_qpseudo_predictions(
        predictions,
        runtime_val,
        evaluator_cfg,
    )
    write_qpseudo_artifacts(
        output_dir=attempt_dir / "pseudo",
        predictions=predictions,
        metrics=metrics,
        checkpoint_sha256=sha256(checkpoint_path),
        validation_split_sha256=FIXED_VALIDATION_SHA256,
        pseudo_acceptance_config_sha256=sha256(
            acceptance_config
        ),
        evaluator_config_sha256=sha256(
            evaluator_config
        ),
        ann_file=runtime_val,
    )


def _execute_one(
    repo_root: Path,
    row: dict[str, str],
    attempt_id: str,
    *,
    recover: bool = False,
) -> None:
    run_id = str(row["run_id"]).strip()

    if recover:
        _attempt_number(attempt_id)
    else:
        _validate_attempt_request(
            repo_root,
            row,
            attempt_id,
        )

    attempt_dir = _attempt_directory(
        repo_root,
        run_id,
        attempt_id,
    )

    if recover:
        if attempt_dir.exists():
            raise OfficialLauncherError(
                "Same-attempt recovery requires the local "
                f"attempt directory to be absent: {attempt_dir}"
            )
    elif attempt_dir.exists():
        raise OfficialLauncherError(
            "Official attempt directory "
            f"already exists: {attempt_dir}"
        )

    pre_run, expected = _build_pre_run_assertions(
        repo_root,
        row,
    )
    pre_run["attempt_id"] = attempt_id

    if (
        not pre_run["all_checks_pass"]
        or pre_run["launch_status"] != "PASS"
    ):
        raise OfficialLauncherError(
            "Pre-run assertions did not PASS."
        )

    from src.utils.official_run_root_manifest import (
        ensure_official_run_root_manifest,
    )

    ensure_official_run_root_manifest(
        run_root=attempt_dir.parent,
        run_id=run_id,
        attempt_id=attempt_id,
        expected=expected,
        recovery=recover,
    )

    from src.utils.s9_active_backup import (
        ActiveBackupManager,
        recover_official_attempt,
        validate_backup_runtime,
    )

    if not recover:
        validate_backup_runtime(repo_root)

    if recover:
        recovery = recover_official_attempt(
            repo_root,
            run_id,
            attempt_id,
        )
        latest_resume_path = Path(
            recovery["latest_resume_path"]
        )

        retry_path = attempt_dir / "retry_deviation.json"
        if not retry_path.is_file():
            raise OfficialLauncherError(
                "Recovered retry_deviation.json is missing."
            )

        recovered_retry = json.loads(
            retry_path.read_text(encoding="utf-8-sig")
        )
        if recovered_retry.get("run_id") != run_id:
            raise OfficialLauncherError(
                "Recovered retry record run_id mismatch."
            )
        if recovered_retry.get("attempt_id") != attempt_id:
            raise OfficialLauncherError(
                "Recovered retry record attempt_id mismatch."
            )
        if (
            int(recovered_retry.get("training_seed"))
            != int(str(row["training_seed"]).strip())
        ):
            raise OfficialLauncherError(
                "Recovered retry record training seed mismatch."
            )
        if recovered_retry.get("scientific_protocol_changed") is not False:
            raise OfficialLauncherError(
                "Recovered attempt changes scientific protocol."
            )
        if recovered_retry.get("technical_failure") is not False:
            raise OfficialLauncherError(
                "Same-attempt recovery is forbidden after recorded technical failure."
            )
        if recovered_retry.get("retry_required") is not False:
            raise OfficialLauncherError(
                "Same-attempt recovery is forbidden when technical retry is required."
            )
        if recovered_retry.get("retry_reason") is not None:
            raise OfficialLauncherError(
                "Same-attempt recovery requires no recorded retry reason."
            )

        cfg = _configure_mmdetection(
            repo_root,
            expected,
            attempt_dir,
        )
        cfg.resume = True
        cfg.load_from = str(latest_resume_path)
        retry_deviation = recovered_retry
    else:
        cfg = _configure_mmdetection(
            repo_root,
            expected,
            attempt_dir,
        )

        if attempt_id != INITIAL_ATTEMPT_ID:
            cfg.resume = False
            cfg.load_from = None

        _write_json(
            attempt_dir / "pre_run_assertions.json",
            pre_run,
        )

        retry_deviation = _retry_deviation_payload(
            row,
            attempt_id,
        )
        _write_json(
            attempt_dir / "retry_deviation.json",
            retry_deviation,
        )

    from src.utils.official_logical_run_status import (
        set_official_logical_run_in_progress,
        set_official_logical_run_technical_failure,
    )

    set_official_logical_run_in_progress(
        run_root=attempt_dir.parent,
        run_id=run_id,
        attempt_id=attempt_id,
        training_seed=int(str(row["training_seed"]).strip()),
        recovery=recover,
    )

    backup = None

    try:
        if recover:
            from src.utils.official_attempt_closure import (
                validate_official_run_manifest,
            )

            validate_official_run_manifest(
                attempt_dir=attempt_dir,
                run_id=run_id,
                attempt_id=attempt_id,
                expected=expected,
            )
        else:
            from src.utils.official_run_manifest import (
                materialize_official_run_manifest,
            )

            materialize_official_run_manifest(
                repo_root=repo_root,
                attempt_dir=attempt_dir,
                run_id=run_id,
                attempt_id=attempt_id,
                expected=expected,
            )

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

        from src.utils.official_environment_reference import (
            materialize_official_environment_reference,
        )

        materialize_official_environment_reference(
            repo_root=repo_root,
            attempt_dir=attempt_dir,
            run_id=run_id,
            attempt_id=attempt_id,
            recovery=recover,
        )

        backup = ActiveBackupManager(
            repo_root,
            attempt_dir,
        )
        backup.start()
        backup.sync(
            "logical_run_status_in_progress",
            raise_on_error=True,
        )

        from mmengine.runner import Runner
        from src.utils.official_attempt_closure import (
            close_official_attempt,
        )

        from mmengine.utils import import_modules_from_strings

        runtime_custom_imports = cfg.get("custom_imports", {})
        import_modules_from_strings(
            runtime_custom_imports.get("imports", []),
            allow_failed_imports=runtime_custom_imports.get(
                "allow_failed_imports",
                False,
            ),
        )

        mmengine_namespace_only_keys = (
            "annotations",
            "Path",
            "SSOD_ROOT",
            "COCO_ROOT",
            "MMDET_IMAGE_ROOT",
        )
        for key in mmengine_namespace_only_keys:
            if key in cfg:
                cfg.pop(key)

        begin_official_seed_runtime_observation(
            attempt_dir=attempt_dir,
        )
        try:
            runner = Runner.from_cfg(cfg)
        finally:
            finish_official_seed_runtime_observation(
                attempt_dir=attempt_dir,
            )
        runner.train()
        if str(row["method"]).strip() == "SSL":
            _run_qpseudo_after_training(
                repo_root,
                attempt_dir,
                runner,
            )
        close_official_attempt(
            attempt_dir=attempt_dir,
            run_id=run_id,
            attempt_id=attempt_id,
            expected=expected,
            validation_interval=int(
                cfg.train_cfg.val_interval
            ),
        )
    except Exception as exc:
        retry_deviation["technical_failure"] = True
        retry_deviation["failure_reason"] = (
            f"{type(exc).__name__}: {exc}"
        )
        retry_deviation["retry_required"] = True
        retry_deviation["retry_reason"] = (
            "TECHNICAL_FAILURE_ONLY"
        )
        _write_json(
            attempt_dir / "retry_deviation.json",
            retry_deviation,
        )
        try:
            set_official_logical_run_technical_failure(
                run_root=attempt_dir.parent,
                run_id=run_id,
                attempt_id=attempt_id,
                training_seed=int(
                    str(row["training_seed"]).strip()
                ),
            )
        finally:
            if backup is not None:
                backup.record_failure_state()
                backup.stop()
        raise

    backup.finalize()
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
        choices=("sup-low-label", "sup-reference", "ssl-low-label"),
        help=(
            "Select a locked official inventory scope."
        ),
    )

    parser.add_argument(
        "--attempt-id",
        default=INITIAL_ATTEMPT_ID,
        help=(
            "Official attempt identity. "
            "Initial execution uses attempt_001; "
            "technical retry uses the next attempt_NNN "
            "for the same logical run and exact seed."
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

    parser.add_argument(
        "--recover",
        action="store_true",
        help=(
            "Recover the exact same official attempt "
            "from verified durable backup and resume "
            "from checkpoints/latest_resume.pth. "
            "Requires --run-id and --execute."
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
        if args.recover and args.run_id is None:
            raise OfficialLauncherError(
                "--recover requires exact --run-id; "
                "scope recovery is forbidden."
            )
        if args.recover and not args.execute:
            raise OfficialLauncherError(
                "--recover requires --execute."
            )

        rows = _load_inventory(repo_root)

        if args.run_id is not None:
            row = _select_run(
                rows,
                args.run_id,
            )

            _assert_inventory_state(row)

            if args.recover:
                _attempt_number(args.attempt_id)
            else:
                _validate_attempt_request(
                    repo_root,
                    row,
                    args.attempt_id,
                )

            if args.execute:
                _execute_one(
                    repo_root,
                    row,
                    args.attempt_id,
                    recover=args.recover,
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
            if args.attempt_id != INITIAL_ATTEMPT_ID:
                raise OfficialLauncherError(
                    "Non-initial technical retry "
                    "requires --run-id."
                )

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
                        args.attempt_id,
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
