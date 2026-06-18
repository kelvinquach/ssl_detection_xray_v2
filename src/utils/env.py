"""Environment introspection for reproducibility.

Collects Python / OS info, package versions, import-availability of the key
dependencies, and CUDA status. All probing is defensive: a missing or broken
package is reported as unavailable, never raised.

Phase 0 — no dataset access.
"""

from __future__ import annotations

import importlib
import importlib.metadata as importlib_metadata
import platform
import sys
from datetime import datetime, timezone
from typing import Any, Dict, List

# Packages whose import we explicitly probe. Order is informative.
PROBE_IMPORTS: List[str] = [
    "torch",
    "torchvision",
    "numpy",
    "pandas",
    "cv2",
    "pydicom",
    "pycocotools",
    "mmengine",
    "mmcv",
    "mmdet",
]

# Map import name -> distribution name (when they differ) for version lookup.
_DIST_NAME = {
    "cv2": "opencv-python",
    "pydicom": "pydicom",
    "pycocotools": "pycocotools",
    "mmcv": "mmcv",
    "mmdet": "mmdet",
    "mmengine": "mmengine",
}


def _dist_version(import_name: str) -> str | None:
    """Best-effort version lookup via importlib.metadata."""
    candidates = [import_name, _DIST_NAME.get(import_name, import_name)]
    for name in candidates:
        try:
            return importlib_metadata.version(name)
        except Exception:
            continue
    return None


def probe_import(import_name: str) -> Dict[str, Any]:
    """Try to import a module and report status without raising."""
    result: Dict[str, Any] = {
        "import_ok": False,
        "version": None,
        "error": None,
    }
    try:
        mod = importlib.import_module(import_name)
        result["import_ok"] = True
        result["version"] = (
            getattr(mod, "__version__", None) or _dist_version(import_name)
        )
    except Exception as exc:
        result["import_ok"] = False
        result["error"] = repr(exc)
        # Still try a metadata-based version (installed but failing import).
        result["version"] = _dist_version(import_name)
    return result


def collect_imports() -> Dict[str, Dict[str, Any]]:
    """Probe every package in PROBE_IMPORTS."""
    return {name: probe_import(name) for name in PROBE_IMPORTS}


def collect_cuda() -> Dict[str, Any]:
    """Report CUDA availability and GPU info via torch, defensively."""
    info: Dict[str, Any] = {
        "torch_available": False,
        "cuda_available": False,
        "torch_cuda_version": None,
        "device_count": 0,
        "devices": [],
        "error": None,
    }
    try:
        import torch  # noqa: WPS433 (local import is intentional)

        info["torch_available"] = True
        info["torch_cuda_version"] = getattr(torch.version, "cuda", None)
        info["cuda_available"] = bool(torch.cuda.is_available())
        if info["cuda_available"]:
            info["device_count"] = int(torch.cuda.device_count())
            for i in range(info["device_count"]):
                try:
                    info["devices"].append(
                        {
                            "index": i,
                            "name": torch.cuda.get_device_name(i),
                            "capability": ".".join(
                                map(str, torch.cuda.get_device_capability(i))
                            ),
                        }
                    )
                except Exception as exc:  # pragma: no cover
                    info["devices"].append({"index": i, "error": repr(exc)})
    except Exception as exc:
        info["error"] = repr(exc)
    return info


def collect_platform() -> Dict[str, Any]:
    """Collect Python / OS / machine info."""
    return {
        "python_version": sys.version,
        "python_version_info": list(sys.version_info[:3]),
        "executable": sys.executable,
        "platform": platform.platform(),
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "node": platform.node(),
    }


def collect_environment() -> Dict[str, Any]:
    """Assemble the full environment report (JSON-serializable)."""
    imports = collect_imports()
    # Overall import health: core SSL-detection stack present?
    core = ["torch", "torchvision", "numpy"]
    framework = ["mmengine", "mmcv", "mmdet"]
    report: Dict[str, Any] = {
        "report_type": "phase0_environment_check",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "platform": collect_platform(),
        "imports": imports,
        "cuda": collect_cuda(),
        "summary": {
            "core_import_ok": all(imports[p]["import_ok"] for p in core),
            "framework_import_ok": all(
                imports[p]["import_ok"] for p in framework
            ),
            "primary_framework": "mmdetection",
            "detectron2": "optional",
        },
    }
    return report


if __name__ == "__main__":
    import json

    print(json.dumps(collect_environment(), indent=2))
