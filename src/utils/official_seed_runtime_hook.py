"""Official S9 seed-runtime evidence producer.

The hook is observation-only. It must not reseed Python, NumPy, PyTorch,
CUDA, samplers, workers, or augmentation RNGs.

The pre-Runner phase captures the already-validated data-membership identity.
The before_train hook then observes the runtime state established by MMEngine
and materializes seed_runtime.json before the first training batch.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import tempfile
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
import torch
from mmengine.hooks import Hook
from mmdet.registry import HOOKS


class OfficialSeedRuntimeError(RuntimeError):
    """Raised when official seed-runtime evidence cannot be proven."""


_PENDING_CONTEXTS: dict[str, dict[str, Any]] = {}

_ACTIVE_SEED_OBSERVATIONS: dict[str, dict[str, Any]] = {}


def begin_official_seed_runtime_observation(
    *,
    attempt_dir: Path,
) -> None:
    """Observe MMEngine seed-application calls without changing RNG semantics."""
    key = str(Path(attempt_dir).resolve())
    if key not in _PENDING_CONTEXTS:
        raise OfficialSeedRuntimeError(
            "Seed-runtime context must exist before seed observation begins."
        )
    if key in _ACTIVE_SEED_OBSERVATIONS:
        raise OfficialSeedRuntimeError(
            "Seed-runtime observation is already active for this attempt."
        )

    calls: list[dict[str, Any]] = []
    originals = {
        "python_random_seed": random.seed,
        "numpy_seed": np.random.seed,
        "torch_manual_seed": torch.manual_seed,
    }

    def _wrap(source: str, function: Any):
        def _observed(seed: Any, *args: Any, **kwargs: Any):
            try:
                observed_seed: Any = int(seed)
            except Exception:
                observed_seed = repr(seed)
            calls.append({
                "source": source,
                "seed": observed_seed,
            })
            return function(seed, *args, **kwargs)
        return _observed

    random.seed = _wrap("python_random.seed", originals["python_random_seed"])
    np.random.seed = _wrap("numpy.random.seed", originals["numpy_seed"])
    torch.manual_seed = _wrap("torch.manual_seed", originals["torch_manual_seed"])

    _ACTIVE_SEED_OBSERVATIONS[key] = {
        "calls": calls,
        "originals": originals,
    }


def finish_official_seed_runtime_observation(
    *,
    attempt_dir: Path,
) -> None:
    """Restore RNG functions and bind observed seed calls to the attempt."""
    key = str(Path(attempt_dir).resolve())
    state = _ACTIVE_SEED_OBSERVATIONS.pop(key, None)
    if not isinstance(state, dict):
        raise OfficialSeedRuntimeError(
            "Seed-runtime observation is not active for this attempt."
        )

    originals = state["originals"]
    random.seed = originals["python_random_seed"]
    np.random.seed = originals["numpy_seed"]
    torch.manual_seed = originals["torch_manual_seed"]

    context = _PENDING_CONTEXTS.get(key)
    if not isinstance(context, dict):
        raise OfficialSeedRuntimeError(
            "Seed-runtime context disappeared during Runner construction."
        )
    context["seed_application_calls"] = list(state["calls"])


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise OfficialSeedRuntimeError(
            f"Required seed-runtime input is missing: {path}"
        )
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise OfficialSeedRuntimeError(
            f"Cannot parse JSON seed-runtime input: {path}"
        ) from exc
    if not isinstance(payload, dict):
        raise OfficialSeedRuntimeError(
            f"Seed-runtime JSON input must be an object: {path}"
        )
    return payload


def _write_json_atomic(
    path: Path,
    payload: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        dir=str(path.parent),
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(
                payload,
                handle,
                indent=2,
                sort_keys=True,
                ensure_ascii=False,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)
    except Exception:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


def _membership_identity(
    data_provenance: dict[str, Any],
    *,
    method: str,
) -> tuple[str, dict[str, Any]]:
    common_fields = (
        "condition_role",
        "budget",
        "train_image_count",
        "val_image_count",
        "test_image_count",
        "labeled_image_count",
        "unlabeled_image_count",
        "train_membership_sha256",
        "val_membership_sha256",
        "test_membership_sha256",
        "labeled_membership_sha256",
    )

    missing = [
        key
        for key in common_fields
        if key not in data_provenance
    ]
    if missing:
        raise OfficialSeedRuntimeError(
            "data_provenance.json is missing membership fields: "
            + ", ".join(missing)
        )

    identity = {
        key: data_provenance[key]
        for key in common_fields
    }

    if method == "SSL":
        ssl_fields = (
            "unlabeled_membership_sha256",
            "labeled_unlabeled_disjoint",
            "labeled_unlabeled_union_equals_train",
        )
        missing_ssl = [
            key
            for key in ssl_fields
            if key not in data_provenance
        ]
        if missing_ssl:
            raise OfficialSeedRuntimeError(
                "SSL data_provenance.json is missing membership fields: "
                + ", ".join(missing_ssl)
            )
        for key in ssl_fields:
            identity[key] = data_provenance[key]

        if identity["labeled_unlabeled_disjoint"] is not True:
            raise OfficialSeedRuntimeError(
                "SSL labeled/unlabeled membership is not disjoint."
            )
        if identity["labeled_unlabeled_union_equals_train"] is not True:
            raise OfficialSeedRuntimeError(
                "SSL labeled/unlabeled union does not equal train membership."
            )

    canonical = json.dumps(
        identity,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")

    return hashlib.sha256(canonical).hexdigest(), identity


def prepare_official_seed_runtime_context(
    *,
    repo_root: Path,
    attempt_dir: Path,
    expected: dict[str, Any],
    recovery: bool = False,
) -> None:
    """Capture or validate seed-runtime identity before Runner construction."""
    repo_root = Path(repo_root).resolve()
    attempt_dir = Path(attempt_dir).resolve()
    key = str(attempt_dir)

    if key in _PENDING_CONTEXTS:
        raise OfficialSeedRuntimeError(
            "Seed-runtime context already exists for this attempt."
        )

    method = str(expected.get("method", "")).strip()
    if method not in {"SUP", "SSL"}:
        raise OfficialSeedRuntimeError(
            f"Unsupported official method for seed runtime: {method!r}"
        )

    training_seed = expected.get("training_seed")
    if (
        isinstance(training_seed, bool)
        or not isinstance(training_seed, int)
        or training_seed < 0
    ):
        raise OfficialSeedRuntimeError(
            "Official training_seed must be a non-negative integer."
        )

    provenance_path = attempt_dir / "data_provenance.json"
    provenance = _load_json(provenance_path)
    membership_hash, membership_identity = _membership_identity(
        provenance,
        method=method,
    )

    worker_audit_path = (
        repo_root
        / "artifacts"
        / "preflight"
        / "seed"
        / "dataloader_worker_seed_audit.json"
    )
    worker_audit = _load_json(worker_audit_path)
    if (
        worker_audit.get("artifact_type")
        != "DATALOADER_WORKER_SEED_AUDIT"
        or worker_audit.get("status") != "PASS"
    ):
        raise OfficialSeedRuntimeError(
            "Canonical S2.07 worker-seed evidence is not PASS."
        )
    worker_examples_observed = worker_audit.get("worker_seed_examples")
    worker_strategy_preflight = worker_audit.get("worker_seed_strategy")
    if (
        not isinstance(worker_examples_observed, list)
        or not worker_examples_observed
        or not isinstance(worker_strategy_preflight, dict)
    ):
        raise OfficialSeedRuntimeError(
            "Canonical S2.07 worker-seed evidence is incomplete."
        )

    worker_preflight_evidence = {
        "source": "artifacts/preflight/seed/dataloader_worker_seed_audit.json",
        "base_training_seed": worker_audit.get("base_training_seed"),
        "worker_seed_strategy": worker_strategy_preflight,
        "worker_seed_examples": worker_examples_observed,
    }

    seed_runtime_path = attempt_dir / "seed_runtime.json"

    if recovery:
        if not seed_runtime_path.is_file():
            raise OfficialSeedRuntimeError(
                "Recovered attempt is missing seed_runtime.json."
            )
        restored = _load_json(seed_runtime_path)
        if int(restored.get("base_training_seed", -1)) != int(training_seed):
            raise OfficialSeedRuntimeError(
                "Recovered seed_runtime base training seed mismatch."
            )
        if restored.get("membership_unchanged") is not True:
            raise OfficialSeedRuntimeError(
                "Recovered seed_runtime reports changed membership."
            )
        if (
            restored.get("membership_hash_before_seed_application")
            != membership_hash
            or restored.get("membership_hash_after_seed_application")
            != membership_hash
        ):
            raise OfficialSeedRuntimeError(
                "Recovered seed_runtime membership identity mismatch."
            )
        _PENDING_CONTEXTS[key] = {
            "method": method,
            "base_training_seed": int(training_seed),
            "membership_hash_before_seed_application": membership_hash,
            "membership_identity_before_seed_application": membership_identity,
            "recovery_reuse_existing": True,
            "worker_preflight_evidence": worker_preflight_evidence,
        }
        return

    if seed_runtime_path.exists():
        raise OfficialSeedRuntimeError(
            "Fresh attempt unexpectedly already contains seed_runtime.json."
        )

    _PENDING_CONTEXTS[key] = {
        "method": method,
        "base_training_seed": int(training_seed),
        "membership_hash_before_seed_application": membership_hash,
        "membership_identity_before_seed_application": membership_identity,
        "recovery_reuse_existing": False,
        "worker_preflight_evidence": worker_preflight_evidence,
    }


def _augmentation_rng_strategy(method: str) -> str:
    if method == "SUP":
        return (
            "DETERMINISTIC_SUP_PIPELINE_NO_STOCHASTIC_AUGMENTATION"
        )
    if method == "SSL":
        return (
            "SSL_WEAK_VIEW_DETERMINISTIC_GEOMETRY;"
            "STRONG_GrayscaleBrightnessContrast_USES_NUMPY_GLOBAL_RNG_"
            "UNIFORM_BRIGHTNESS_AND_CONTRAST"
        )
    raise OfficialSeedRuntimeError(
        f"Unsupported method for augmentation RNG strategy: {method!r}"
    )


def _observe_sampler_seed(
    loader: Any,
    *,
    base_training_seed: int,
) -> tuple[int, str, dict[str, Any]]:
    sampler = getattr(loader, "sampler", None)
    if sampler is None:
        raise OfficialSeedRuntimeError(
            "Runtime train dataloader has no sampler."
        )

    sampler_type = type(sampler).__name__
    seed = getattr(sampler, "seed", None)

    if isinstance(seed, bool) or not isinstance(seed, int):
        raise OfficialSeedRuntimeError(
            f"{sampler_type} does not expose an integer runtime seed."
        )

    observed = int(seed)
    sampler_seed_policy: dict[str, Any] = {
        "sampler_type": sampler_type,
        "sampler_module": type(sampler).__module__,
        "base_training_seed": int(base_training_seed),
        "observed_sampler_seed": observed,
        "relation_to_base_training_seed": (
            "DIRECT_EQUAL"
            if observed == int(base_training_seed)
            else "DERIVED_OR_FRAMEWORK_MANAGED"
        ),
    }
    for attribute in ("rank", "world_size", "shuffle", "round_up"):
        value = getattr(sampler, attribute, None)
        if isinstance(value, (str, int, float, bool)) or value is None:
            sampler_seed_policy[attribute] = value

    return observed, sampler_type, sampler_seed_policy


def _observe_worker_seed_policy(
    loader: Any,
    *,
    base_training_seed: int,
) -> tuple[str, list[dict[str, int]], dict[str, Any]]:
    try:
        num_workers = int(loader.num_workers)
    except Exception as exc:
        raise OfficialSeedRuntimeError(
            "Cannot observe train dataloader num_workers."
        ) from exc

    init_fn = getattr(loader, "worker_init_fn", None)

    if num_workers <= 0:
        return (
            "NO_WORKER_SUBPROCESSES_NUM_WORKERS_0",
            [],
            {
                "num_workers": num_workers,
                "worker_init_fn_type": (
                    None if init_fn is None else type(init_fn).__name__
                ),
            },
        )

    if not isinstance(init_fn, partial):
        raise OfficialSeedRuntimeError(
            "MMEngine worker_init_fn is not an inspectable functools.partial."
        )

    keywords = dict(init_fn.keywords or {})
    observed_seed = keywords.get("seed")
    observed_num_workers = keywords.get("num_workers")
    observed_rank = keywords.get("rank", 0)

    if (
        isinstance(observed_seed, bool)
        or not isinstance(observed_seed, int)
        or int(observed_seed) != int(base_training_seed)
    ):
        raise OfficialSeedRuntimeError(
            "MMEngine worker_init_fn seed does not match training_seed."
        )

    if (
        isinstance(observed_num_workers, bool)
        or not isinstance(observed_num_workers, int)
        or int(observed_num_workers) != num_workers
    ):
        raise OfficialSeedRuntimeError(
            "MMEngine worker_init_fn num_workers mismatch."
        )

    if isinstance(observed_rank, bool) or not isinstance(observed_rank, int):
        raise OfficialSeedRuntimeError(
            "MMEngine worker_init_fn rank is not an integer."
        )

    examples = []
    for worker_id in range(min(num_workers, 4)):
        worker_seed = (
            num_workers * int(observed_rank)
            + worker_id
            + int(base_training_seed)
        )
        examples.append(
            {
                "worker_id": int(worker_id),
                "rank": int(observed_rank),
                "expected_worker_seed": int(worker_seed),
            }
        )

    strategy = (
        "mmengine.dataset.utils.worker_init_fn: "
        "worker_seed = num_workers * rank + worker_id + base_training_seed"
    )

    metadata = {
        "num_workers": num_workers,
        "rank": int(observed_rank),
        "worker_init_fn_type": type(init_fn).__name__,
        "worker_init_fn_function": getattr(
            getattr(init_fn, "func", None),
            "__qualname__",
            repr(getattr(init_fn, "func", None)),
        ),
        "worker_init_seed_observed": int(observed_seed),
        "worker_init_num_workers_observed": int(observed_num_workers),
    }

    return strategy, examples, metadata


def _observe_cuda_initial_seeds() -> list[dict[str, int]]:
    if not torch.cuda.is_available():
        return []

    result = []
    for device_index in range(int(torch.cuda.device_count())):
        with torch.cuda.device(device_index):
            result.append(
                {
                    "device_index": int(device_index),
                    "initial_seed": int(torch.cuda.initial_seed()),
                }
            )
    return result


@HOOKS.register_module()
class OfficialSeedRuntimeHook(Hook):
    """Materialize official seed_runtime.json before the first train batch."""

    priority = "VERY_HIGH"

    def before_train(self, runner: Any) -> None:
        attempt_dir = Path(runner.work_dir).resolve()
        key = str(attempt_dir)

        context = _PENDING_CONTEXTS.pop(key, None)
        if not isinstance(context, dict):
            raise OfficialSeedRuntimeError(
                "Official seed-runtime pre-Runner context is unavailable."
            )

        method = context["method"]
        base_seed = int(context["base_training_seed"])

        if context.get("recovery_reuse_existing") is True:
            return

        runner_seed = getattr(runner, "seed", None)
        if (
            isinstance(runner_seed, bool)
            or not isinstance(runner_seed, int)
            or int(runner_seed) != base_seed
        ):
            raise OfficialSeedRuntimeError(
                "MMEngine runner.seed does not match official training_seed."
            )

        seed_application_calls = context.get("seed_application_calls")
        if not isinstance(seed_application_calls, list):
            raise OfficialSeedRuntimeError(
                "MMEngine seed-application observations are unavailable."
            )

        def _observed_applied_seed(source: str) -> int:
            values = [
                item.get("seed")
                for item in seed_application_calls
                if isinstance(item, dict)
                and item.get("source") == source
            ]
            if not values:
                raise OfficialSeedRuntimeError(
                    f"No observed seed application for {source}."
                )
            if any(
                isinstance(value, bool)
                or not isinstance(value, int)
                or int(value) != base_seed
                for value in values
            ):
                raise OfficialSeedRuntimeError(
                    f"Observed {source} seed does not match training_seed."
                )
            return int(values[-1])

        python_seed_observed = _observed_applied_seed(
            "python_random.seed"
        )
        numpy_seed_observed = _observed_applied_seed(
            "numpy.random.seed"
        )
        torch_seed_application_observed = _observed_applied_seed(
            "torch.manual_seed"
        )

        numpy_state = np.random.get_state()
        numpy_state_head_observed = int(numpy_state[1][0])
        torch_seed_observed = int(torch.initial_seed())

        if torch_seed_observed != base_seed:
            raise OfficialSeedRuntimeError(
                "Observed PyTorch initial seed does not match official "
                "training_seed."
            )

        cuda_initial_seeds = _observe_cuda_initial_seeds()
        if torch.cuda.is_available():
            if not cuda_initial_seeds:
                raise OfficialSeedRuntimeError(
                    "CUDA is available but CUDA seed evidence is empty."
                )
            if any(
                item["initial_seed"] != base_seed
                for item in cuda_initial_seeds
            ):
                raise OfficialSeedRuntimeError(
                    "Observed CUDA initial seed does not match training_seed."
                )

        train_loop = getattr(runner, "train_loop", None)
        loader = getattr(train_loop, "dataloader", None)
        if loader is None:
            raise OfficialSeedRuntimeError(
                "Runtime train dataloader is unavailable in before_train."
            )

        (
            sampler_seed,
            sampler_type,
            sampler_seed_policy,
        ) = _observe_sampler_seed(
            loader,
            base_training_seed=base_seed,
        )

        (
            worker_seed_strategy,
            worker_seed_derived_examples,
            worker_runtime_metadata,
        ) = _observe_worker_seed_policy(
            loader,
            base_training_seed=base_seed,
        )

        torch_deterministic = bool(
            torch.are_deterministic_algorithms_enabled()
        )

        cudnn_backend = getattr(torch.backends, "cudnn", None)
        cudnn_deterministic = (
            None
            if cudnn_backend is None
            else bool(cudnn_backend.deterministic)
        )
        cudnn_benchmark = (
            None
            if cudnn_backend is None
            else bool(cudnn_backend.benchmark)
        )

        provenance = _load_json(
            attempt_dir / "data_provenance.json"
        )
        membership_after, membership_identity_after = _membership_identity(
            provenance,
            method=method,
        )

        membership_before = context[
            "membership_hash_before_seed_application"
        ]
        membership_unchanged = membership_before == membership_after

        if not membership_unchanged:
            raise OfficialSeedRuntimeError(
                "Data membership identity changed after seed application."
            )

        payload = {
            "base_training_seed": base_seed,
            "python_seed_observed": python_seed_observed,
            "numpy_seed_observed": numpy_seed_observed,

            "numpy_runtime_state_head_observed":
                numpy_state_head_observed,
            "torch_initial_seed_observed": torch_seed_observed,
            "cuda_seed_policy": (
                "MMEngine randomness seed controls PyTorch CPU/CUDA; "
                "official cfg uses diff_rank_seed=False and "
                "deterministic=True"
            ),
            "sampler_seed_observed": sampler_seed,
            "sampler_seed_policy": sampler_seed_policy,
            "worker_seed_strategy": worker_seed_strategy,
            "worker_seed_examples": context[
                "worker_preflight_evidence"
            ]["worker_seed_examples"],
            "worker_seed_examples_source": context[
                "worker_preflight_evidence"
            ]["source"],
            "worker_seed_examples_preflight_base_training_seed": context[
                "worker_preflight_evidence"
            ]["base_training_seed"],
            "worker_seed_preflight_strategy": context[
                "worker_preflight_evidence"
            ]["worker_seed_strategy"],
            "worker_seed_derived_examples": worker_seed_derived_examples,
            "augmentation_rng_strategy":
                _augmentation_rng_strategy(method),
            "torch_deterministic_algorithms_observed":
                torch_deterministic,
            "cudnn_deterministic_observed":
                cudnn_deterministic,
            "cudnn_benchmark_observed":
                cudnn_benchmark,
            "membership_hash_before_seed_application":
                membership_before,
            "membership_hash_after_seed_application":
                membership_after,
            "membership_unchanged": membership_unchanged,
            "python_seed_observation_method": (
                "Observed argument to Python random.seed during MMEngine "
                "Runner construction; no evidence-only reseed performed"
            ),
            "numpy_seed_observation_method": (
                "Observed argument to numpy.random.seed during MMEngine "
                "Runner construction; current NumPy state recorded separately"
            ),
            "torch_seed_application_observed":
                torch_seed_application_observed,
            "seed_application_calls": seed_application_calls,
            "cuda_available": bool(torch.cuda.is_available()),
            "cuda_device_count": int(torch.cuda.device_count()),
            "cuda_initial_seeds_observed": cuda_initial_seeds,
            "sampler_type_observed": sampler_type,
            "sampler_seed_observation_method":
                "runtime train_loop.dataloader.sampler.seed",
            "worker_runtime_metadata": worker_runtime_metadata,
            "membership_identity_before_seed_application":
                context[
                    "membership_identity_before_seed_application"
                ],
            "membership_identity_after_seed_application":
                membership_identity_after,
            "observation_point":
                "OfficialSeedRuntimeHook.before_train_before_first_batch",
            "evidence_mode":
                "OBSERVATION_ONLY_NO_RUNTIME_RESEED",
        }

        _write_json_atomic(
            attempt_dir / "seed_runtime.json",
            payload,
        )


__all__ = [
    "OfficialSeedRuntimeError",
    "OfficialSeedRuntimeHook",
    "begin_official_seed_runtime_observation",
    "finish_official_seed_runtime_observation",
    "prepare_official_seed_runtime_context",
]
