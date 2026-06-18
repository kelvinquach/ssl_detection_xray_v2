"""Reproducibility utilities for SSL chest X-ray detection.

Provides global seed setting across Python `random`, NumPy and PyTorch
(CPU + CUDA), determinism flag configuration, and serializable RNG-state
summaries / manifests for experiment provenance.

Phase 0 — no dataset access. This module only touches RNG state.
"""

from __future__ import annotations

import json
import os
import platform
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

# Optional heavy deps are imported lazily / defensively so this module
# never crashes on a partially-installed environment.
try:
    import numpy as np
except Exception:  # pragma: no cover
    np = None  # type: ignore

try:
    import torch
except Exception:  # pragma: no cover
    torch = None  # type: ignore


def set_global_seed(seed: int, deterministic: bool = True) -> Dict[str, Any]:
    """Set the global RNG seed for all common libraries.

    Args:
        seed: Integer seed applied to Python, NumPy and PyTorch RNGs.
        deterministic: If True, request deterministic algorithms and disable
            cuDNN autotuning. This trades speed for reproducibility.

    Returns:
        A JSON-serializable dict describing exactly which flags were applied.
    """
    config: Dict[str, Any] = {
        "seed": int(seed),
        "deterministic": bool(deterministic),
        "applied": {},
    }

    # Python hash seed (only affects new interpreter processes, but recorded).
    os.environ["PYTHONHASHSEED"] = str(seed)
    config["applied"]["PYTHONHASHSEED"] = str(seed)

    # Python built-in RNG.
    random.seed(seed)
    config["applied"]["python_random"] = True

    # NumPy.
    if np is not None:
        np.random.seed(seed)
        config["applied"]["numpy"] = True
    else:
        config["applied"]["numpy"] = False

    # PyTorch CPU + CUDA.
    if torch is not None:
        torch.manual_seed(seed)
        config["applied"]["torch_cpu"] = True

        cuda_available = bool(getattr(torch, "cuda", None) and torch.cuda.is_available())
        if cuda_available:
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            config["applied"]["torch_cuda"] = True
        else:
            config["applied"]["torch_cuda"] = False

        # Determinism flags.
        if deterministic:
            # cuBLAS workspace config required for deterministic CUDA matmuls.
            os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
            config["applied"]["CUBLAS_WORKSPACE_CONFIG"] = os.environ[
                "CUBLAS_WORKSPACE_CONFIG"
            ]

        try:
            torch.use_deterministic_algorithms(deterministic)
            config["applied"]["use_deterministic_algorithms"] = deterministic
        except Exception as exc:  # pragma: no cover
            config["applied"]["use_deterministic_algorithms_error"] = repr(exc)

        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.deterministic = bool(deterministic)
            torch.backends.cudnn.benchmark = not bool(deterministic)
            config["applied"]["cudnn_deterministic"] = bool(deterministic)
            config["applied"]["cudnn_benchmark"] = not bool(deterministic)
    else:
        config["applied"]["torch_cpu"] = False
        config["applied"]["torch_cuda"] = False

    return config


def get_rng_state_summary() -> Dict[str, Any]:
    """Return a JSON-serializable summary of current RNG state.

    We intentionally summarize (hashes / first elements) rather than dump the
    full RNG state, so the manifest stays small and human-readable while still
    giving a fingerprint to detect drift.
    """
    summary: Dict[str, Any] = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python_random": {},
        "numpy": {},
        "torch": {},
    }

    # Python random: state is a large tuple; record a stable fingerprint.
    py_state = random.getstate()
    summary["python_random"] = {
        "version": py_state[0],
        "state_len": len(py_state[1]),
        "fingerprint": hash(py_state[1]) & 0xFFFFFFFF,
    }

    if np is not None:
        np_state = np.random.get_state()
        # np_state = (str, ndarray, int, int, float)
        try:
            keys = np_state[1]
            summary["numpy"] = {
                "bit_generator": str(np_state[0]),
                "state_len": int(len(keys)),
                "pos": int(np_state[2]),
                "fingerprint": int(keys[:8].sum()) & 0xFFFFFFFF,
            }
        except Exception as exc:  # pragma: no cover
            summary["numpy"] = {"error": repr(exc)}
    else:
        summary["numpy"] = {"available": False}

    if torch is not None:
        try:
            cpu_state = torch.get_rng_state()
            summary["torch"]["cpu"] = {
                "state_len": int(cpu_state.numel()),
                "fingerprint": int(cpu_state[:8].sum().item()) & 0xFFFFFFFF,
            }
        except Exception as exc:  # pragma: no cover
            summary["torch"]["cpu"] = {"error": repr(exc)}

        cuda_available = bool(getattr(torch, "cuda", None) and torch.cuda.is_available())
        summary["torch"]["cuda_available"] = cuda_available
        if cuda_available:
            try:
                cuda_state = torch.cuda.get_rng_state()
                summary["torch"]["cuda"] = {
                    "device_count": int(torch.cuda.device_count()),
                    "state_len": int(cuda_state.numel()),
                    "fingerprint": int(cuda_state[:8].sum().item()) & 0xFFFFFFFF,
                }
            except Exception as exc:  # pragma: no cover
                summary["torch"]["cuda"] = {"error": repr(exc)}
    else:
        summary["torch"]["available"] = False

    return summary


def save_seed_manifest(
    path: str | os.PathLike,
    seed: int,
    deterministic_config: Dict[str, Any],
    rng_state_summary: Optional[Dict[str, Any]] = None,
) -> str:
    """Write a seed-state manifest to `path` as JSON.

    Args:
        path: Destination file path.
        seed: The seed used.
        deterministic_config: The dict returned by `set_global_seed`.
        rng_state_summary: Optional summary; computed if not provided.

    Returns:
        The string path written.
    """
    if rng_state_summary is None:
        rng_state_summary = get_rng_state_summary()

    manifest = {
        "manifest_type": "seed_state_manifest",
        "phase": "phase0",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "host": platform.node(),
        "seed": int(seed),
        "deterministic_config": deterministic_config,
        "rng_state_summary": rng_state_summary,
    }

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, sort_keys=False)
    return str(out)


if __name__ == "__main__":
    cfg = set_global_seed(2026, deterministic=True)
    print(json.dumps(cfg, indent=2))
    print(json.dumps(get_rng_state_summary(), indent=2))
