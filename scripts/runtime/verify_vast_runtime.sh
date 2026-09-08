#!/usr/bin/env bash
set -uo pipefail

SCRIPT_VERSION="1.1.0"


# ---------------------------------------------------------------------------
# LOCAL ONE-COMMAND BOOTSTRAP MODE
#
# From Windows Git Bash / PowerShell:
#   bash verify_vast_runtime.sh --remote <HOST> <PORT>
#
# This mode copies this exact verifier to the new Vast instance, verifies the
# transfer hash, runs the verifier remotely, and returns one final PASS/FAIL.
# ---------------------------------------------------------------------------
if [ "${1:-}" = "--remote" ]; then
    REMOTE_HOST="${2:-}"
    REMOTE_PORT="${3:-}"
    REMOTE_USER="${4:-root}"
    REMOTE_KEY="${5:-${HOME}/.ssh/id_ed25519}"

    if [ -z "$REMOTE_HOST" ] || [ -z "$REMOTE_PORT" ]; then
        echo "USAGE: $0 --remote <HOST> <PORT> [USER] [SSH_PRIVATE_KEY]"
        echo "NEW_VAST_VERIFICATION=FAIL"
        exit 20
    fi

    for cmd in ssh scp sha256sum; do
        if ! command -v "$cmd" >/dev/null 2>&1; then
            echo "[FAIL] local command missing: $cmd"
            echo "NEW_VAST_VERIFICATION=FAIL"
            exit 21
        fi
    done

    if [ ! -f "$REMOTE_KEY" ]; then
        echo "[FAIL] SSH private key not found: $REMOTE_KEY"
        echo "NEW_VAST_VERIFICATION=FAIL"
        exit 22
    fi

    SELF_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"
    TARGET="${REMOTE_USER}@${REMOTE_HOST}"
    REMOTE_VERIFIER="/root/verify_vast_runtime.sh"

    echo "========================================"
    echo "SSOD NEW VAST INSTANCE ONE-COMMAND CHECK"
    echo "========================================"
    echo "target=$TARGET"
    echo "port=$REMOTE_PORT"
    echo "script_version=$SCRIPT_VERSION"
    echo

    echo "=== TRANSFER VERIFIER ==="
    if ! scp \
        -i "$REMOTE_KEY" \
        -P "$REMOTE_PORT" \
        -o StrictHostKeyChecking=accept-new \
        "$SELF_PATH" \
        "${TARGET}:${REMOTE_VERIFIER}"; then
        echo "[FAIL] verifier transfer failed"
        echo "NEW_VAST_VERIFICATION=FAIL"
        exit 23
    fi

    echo
    echo "=== VERIFY TRANSFER HASH ==="
    LOCAL_HASH="$(sha256sum "$SELF_PATH" | awk '{print $1}')"

    REMOTE_HASH="$(
        ssh \
          -i "$REMOTE_KEY" \
          -p "$REMOTE_PORT" \
          -o StrictHostKeyChecking=accept-new \
          "$TARGET" \
          "sha256sum '$REMOTE_VERIFIER' | awk '{print \$1}'" \
          2>/dev/null
    )"

    echo "LOCAL_SHA256=$LOCAL_HASH"
    echo "REMOTE_SHA256=$REMOTE_HASH"

    if [ -z "$REMOTE_HASH" ] || [ "$LOCAL_HASH" != "$REMOTE_HASH" ]; then
        echo "HASH_MATCH=False"
        echo "NEW_VAST_VERIFICATION=FAIL"
        exit 24
    fi

    echo "HASH_MATCH=True"

    echo
    echo "=== RUN REMOTE RUNTIME VERIFIER ==="
    ssh \
      -i "$REMOTE_KEY" \
      -p "$REMOTE_PORT" \
      -o StrictHostKeyChecking=accept-new \
      "$TARGET" \
      "bash '$REMOTE_VERIFIER'"

    REMOTE_RC=$?

    echo
    if [ "$REMOTE_RC" -eq 0 ]; then
        echo "========================================"
        echo "NEW_VAST_VERIFICATION=PASS"
        echo "========================================"
        exit 0
    else
        echo "========================================"
        echo "NEW_VAST_VERIFICATION=FAIL"
        echo "========================================"
        exit "$REMOTE_RC"
    fi
fi

EXPECTED_IMAGE="quachthainguyen/sslxray-runtime"
EXPECTED_TAG="gate-i-20260828"
EXPECTED_IMAGE_UUID="${EXPECTED_IMAGE}:${EXPECTED_TAG}"
EXPECTED_DIGEST="sha256:fab1b1ea8850527e5728de526b3d53f063355e84cb3a0d8c75e3a7890411d989"

EXPECTED_TORCH="2.1.0"
EXPECTED_TORCH_CUDA="11.8"
EXPECTED_MMENGINE="0.10.7"
EXPECTED_MMCV="2.1.0"
EXPECTED_MMDET="3.3.0"

PASS_COUNT=0
FAIL_COUNT=0
WARN_COUNT=0

pass() {
    echo "[PASS] $1"
    PASS_COUNT=$((PASS_COUNT + 1))
}

fail() {
    echo "[FAIL] $1"
    FAIL_COUNT=$((FAIL_COUNT + 1))
}

warn() {
    echo "[WARN] $1"
    WARN_COUNT=$((WARN_COUNT + 1))
}

echo "========================================"
echo "SSOD VAST RUNTIME VERIFIER"
echo "========================================"
echo "script_version=$SCRIPT_VERSION"
echo "expected_image_uuid=$EXPECTED_IMAGE_UUID"
echo "expected_digest=$EXPECTED_DIGEST"
echo

echo "=== BASIC RUNTIME ==="

if command -v python >/dev/null 2>&1; then
    pass "python command exists"
    echo "python_version=$(python --version 2>&1)"
else
    fail "python command missing"
fi

if command -v nvidia-smi >/dev/null 2>&1; then
    pass "nvidia-smi exists"
    nvidia-smi --query-gpu=name,memory.total,driver_version \
        --format=csv,noheader,nounits 2>/dev/null || true
else
    fail "nvidia-smi missing"
fi

echo "kernel=$(uname -srmo 2>/dev/null || echo UNKNOWN)"
echo "hostname=$(hostname 2>/dev/null || echo UNKNOWN)"

echo
echo "=== PYTORCH / CUDA ==="

RUNTIME_INFO="$(
python - <<'PY' 2>/dev/null
try:
    import torch
    print("torch=" + str(torch.__version__))
    print("torch_cuda=" + str(torch.version.cuda))
    print("cuda_available=" + str(torch.cuda.is_available()))
    print("cuda_device_count=" + str(torch.cuda.device_count()))
    if torch.cuda.is_available() and torch.cuda.device_count() > 0:
        print("gpu0=" + str(torch.cuda.get_device_name(0)))
    print("cudnn_version=" + str(torch.backends.cudnn.version()))
    print("cudnn_enabled=" + str(torch.backends.cudnn.enabled))
except Exception as e:
    print("runtime_error=" + repr(e))
PY
)"

echo "$RUNTIME_INFO"

TORCH_VERSION="$(printf '%s\n' "$RUNTIME_INFO" | awk -F= '$1=="torch"{print $2}')"
TORCH_CUDA="$(printf '%s\n' "$RUNTIME_INFO" | awk -F= '$1=="torch_cuda"{print $2}')"
CUDA_AVAILABLE="$(printf '%s\n' "$RUNTIME_INFO" | awk -F= '$1=="cuda_available"{print $2}')"
CUDA_DEVICE_COUNT="$(printf '%s\n' "$RUNTIME_INFO" | awk -F= '$1=="cuda_device_count"{print $2}')"

if [ "$TORCH_VERSION" = "$EXPECTED_TORCH" ]; then
    pass "PyTorch version = $EXPECTED_TORCH"
else
    fail "PyTorch version expected=$EXPECTED_TORCH observed=${TORCH_VERSION:-MISSING}"
fi

if [ "$TORCH_CUDA" = "$EXPECTED_TORCH_CUDA" ]; then
    pass "PyTorch CUDA runtime = $EXPECTED_TORCH_CUDA"
else
    fail "PyTorch CUDA expected=$EXPECTED_TORCH_CUDA observed=${TORCH_CUDA:-MISSING}"
fi

if [ "$CUDA_AVAILABLE" = "True" ]; then
    pass "torch.cuda.is_available() = True"
else
    fail "CUDA unavailable to PyTorch"
fi

if [ "${CUDA_DEVICE_COUNT:-0}" -ge 1 ] 2>/dev/null; then
    pass "at least one CUDA GPU visible"
else
    fail "no CUDA GPU visible"
fi

echo
echo "=== MM STACK ==="

MM_INFO="$(
python - <<'PY' 2>/dev/null
try:
    import mmengine, mmcv, mmdet
    print("mmengine=" + str(mmengine.__version__))
    print("mmcv=" + str(mmcv.__version__))
    print("mmdet=" + str(mmdet.__version__))
except Exception as e:
    print("mm_error=" + repr(e))
PY
)"

echo "$MM_INFO"

MMENGINE_VERSION="$(printf '%s\n' "$MM_INFO" | awk -F= '$1=="mmengine"{print $2}')"
MMCV_VERSION="$(printf '%s\n' "$MM_INFO" | awk -F= '$1=="mmcv"{print $2}')"
MMDET_VERSION="$(printf '%s\n' "$MM_INFO" | awk -F= '$1=="mmdet"{print $2}')"

[ "$MMENGINE_VERSION" = "$EXPECTED_MMENGINE" ] \
    && pass "MMEngine version = $EXPECTED_MMENGINE" \
    || fail "MMEngine expected=$EXPECTED_MMENGINE observed=${MMENGINE_VERSION:-MISSING}"

[ "$MMCV_VERSION" = "$EXPECTED_MMCV" ] \
    && pass "MMCV version = $EXPECTED_MMCV" \
    || fail "MMCV expected=$EXPECTED_MMCV observed=${MMCV_VERSION:-MISSING}"

[ "$MMDET_VERSION" = "$EXPECTED_MMDET" ] \
    && pass "MMDetection version = $EXPECTED_MMDET" \
    || fail "MMDetection expected=$EXPECTED_MMDET observed=${MMDET_VERSION:-MISSING}"

echo
echo "=== VAST INSTANCE IDENTITY ==="

pid1_env_get() {
    local key="$1"

    if [ ! -r /proc/1/environ ]; then
        return 0
    fi

    tr '\0' '\n' < /proc/1/environ |
        awk -F= -v k="$key" '
            $1 == k {
                sub(/^[^=]*=/, "")
                print
                exit
            }
        '
}

OBS_CONTAINER_ID="${CONTAINER_ID:-}"
OBS_CONTAINER_API_KEY="${CONTAINER_API_KEY:-}"
OBS_CONTAINER_LABEL="${VAST_CONTAINERLABEL:-${CONTAINER_LABEL:-}}"
OBS_VAST_DEVICE_IDXS="${VAST_DEVICE_IDXS:-}"

if [ -z "$OBS_CONTAINER_ID" ]; then
    OBS_CONTAINER_ID="$(pid1_env_get CONTAINER_ID || true)"
fi

if [ -z "$OBS_CONTAINER_API_KEY" ]; then
    OBS_CONTAINER_API_KEY="$(pid1_env_get CONTAINER_API_KEY || true)"
fi

if [ -z "$OBS_CONTAINER_LABEL" ]; then
    OBS_CONTAINER_LABEL="$(pid1_env_get VAST_CONTAINERLABEL || true)"
fi

if [ -z "$OBS_CONTAINER_LABEL" ]; then
    OBS_CONTAINER_LABEL="$(pid1_env_get CONTAINER_LABEL || true)"
fi

if [ -z "$OBS_VAST_DEVICE_IDXS" ]; then
    OBS_VAST_DEVICE_IDXS="$(pid1_env_get VAST_DEVICE_IDXS || true)"
fi

echo "CONTAINER_ID=${OBS_CONTAINER_ID:-UNSET}"
echo "VAST_CONTAINERLABEL=${OBS_CONTAINER_LABEL:-UNSET}"
echo "VAST_DEVICE_IDXS=${OBS_VAST_DEVICE_IDXS:-UNSET}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-UNSET}"

if [ -n "$OBS_CONTAINER_API_KEY" ]; then
    echo "CONTAINER_API_KEY_PRESENT=YES"
else
    echo "CONTAINER_API_KEY_PRESENT=NO"
fi

if [ -n "$OBS_CONTAINER_ID" ] && [ -n "$OBS_CONTAINER_API_KEY" ]; then
    INSTANCE_JSON="$(
        curl -fsSL \
          -H "Authorization: Bearer $OBS_CONTAINER_API_KEY" \
          "https://console.vast.ai/api/v0/instances/$OBS_CONTAINER_ID/" \
          2>/dev/null || true
    )"

    IMAGE_UUID="$(
        printf '%s' "$INSTANCE_JSON" |
        python -c 'import sys,json
try:
    j=json.load(sys.stdin)
    x=j.get("instances",j)
    print(x.get("image_uuid",""))
except Exception:
    print("")' 2>/dev/null
    )"

    ACTUAL_STATUS="$(
        printf '%s' "$INSTANCE_JSON" |
        python -c 'import sys,json
try:
    j=json.load(sys.stdin)
    x=j.get("instances",j)
    print(x.get("actual_status",""))
except Exception:
    print("")' 2>/dev/null
    )"

    echo "vast_actual_status=$ACTUAL_STATUS"
    echo "vast_image_uuid=$IMAGE_UUID"

    if [ "$ACTUAL_STATUS" = "running" ]; then
        pass "Vast instance status = running"
    else
        fail "Vast instance status observed=${ACTUAL_STATUS:-MISSING}"
    fi

    if [ "$IMAGE_UUID" = "$EXPECTED_IMAGE_UUID" ]; then
        pass "Vast image UUID matches locked repository/tag"
    else
        fail "Vast image UUID expected=$EXPECTED_IMAGE_UUID observed=${IMAGE_UUID:-MISSING}"
    fi
else
    fail "Vast instance metadata unavailable from session and PID 1 environment"
fi

echo
echo "=== DOCKER REGISTRY DIGEST ==="

TOKEN="$(
    curl -fsSL \
      "https://auth.docker.io/token?service=registry.docker.io&scope=repository:${EXPECTED_IMAGE}:pull" \
      2>/dev/null |
    python -c 'import sys,json
try:
    print(json.load(sys.stdin)["token"])
except Exception:
    print("")' 2>/dev/null
)"

if [ -n "$TOKEN" ]; then
    REGISTRY_DIGEST="$(
        curl -fsSI \
          -H "Authorization: Bearer $TOKEN" \
          -H "Accept: application/vnd.docker.distribution.manifest.v2+json, application/vnd.docker.distribution.manifest.list.v2+json, application/vnd.oci.image.manifest.v1+json, application/vnd.oci.image.index.v1+json" \
          "https://registry-1.docker.io/v2/${EXPECTED_IMAGE}/manifests/${EXPECTED_TAG}" \
          2>/dev/null |
        tr -d '\r' |
        awk -F': ' 'tolower($1)=="docker-content-digest"{print $2}'
    )"

    echo "registry_digest=$REGISTRY_DIGEST"

    if [ "$REGISTRY_DIGEST" = "$EXPECTED_DIGEST" ]; then
        pass "Docker registry digest matches locked digest"
    else
        fail "Docker digest expected=$EXPECTED_DIGEST observed=${REGISTRY_DIGEST:-MISSING}"
    fi
else
    fail "could not obtain Docker registry token"
fi

echo
echo "=== STARTUP WARNING AUDIT ==="

if [ -f /root/onstart.sh ] &&
   grep -qx 'entrypoint.sh' /root/onstart.sh 2>/dev/null &&
   ! command -v entrypoint.sh >/dev/null 2>&1; then
    warn "/root/onstart.sh references missing entrypoint.sh (observed non-gating Vast template warning)"
else
    pass "no known missing entrypoint.sh startup warning detected"
fi

echo
echo "=== PROJECT / STORAGE FOUNDATION ==="

SSOD_ROOT="${SSOD_ROOT:-/workspace/ssod}"
SSOD_PROJECT_DIR="$SSOD_ROOT/project"
SSOD_DATA_DIR="$SSOD_ROOT/data"
SSOD_IMAGES_DIR="$SSOD_ROOT/data/images"
SSOD_COCO_DIR="$SSOD_ROOT/data/coco"
SSOD_ARTIFACTS_DIR="$SSOD_ROOT/artifacts"
SSOD_CHECKPOINTS_DIR="$SSOD_ROOT/checkpoints"
SSOD_CACHE_DIR="$SSOD_ROOT/cache"

STORAGE_FAIL_COUNT_BEFORE="$FAIL_COUNT"

if mkdir -p \
    "$SSOD_PROJECT_DIR" \
    "$SSOD_IMAGES_DIR" \
    "$SSOD_COCO_DIR" \
    "$SSOD_ARTIFACTS_DIR" \
    "$SSOD_CHECKPOINTS_DIR" \
    "$SSOD_CACHE_DIR"; then
    pass "SSOD project/data/artifact directory structure created or already present"
else
    fail "cannot create SSOD project/data/artifact directory structure under $SSOD_ROOT"
fi

echo "ssod_root=$SSOD_ROOT"
echo "=== DIRECTORY STRUCTURE ==="
find "$SSOD_ROOT" -maxdepth 2 -type d 2>/dev/null | sort || true

echo "=== WRITE TEST ==="
for d in \
    "$SSOD_PROJECT_DIR" \
    "$SSOD_DATA_DIR" \
    "$SSOD_IMAGES_DIR" \
    "$SSOD_COCO_DIR" \
    "$SSOD_ARTIFACTS_DIR" \
    "$SSOD_CHECKPOINTS_DIR" \
    "$SSOD_CACHE_DIR"
do
    test_file="$d/.s1_write_test"
    if printf 'S1_STORAGE_WRITE_TEST\n' > "$test_file" 2>/dev/null &&
       [ -s "$test_file" ]; then
        pass "writable: $d"
    else
        fail "not writable: $d"
    fi
    rm -f "$test_file" 2>/dev/null || true
done

echo "=== FILESYSTEM ==="
if df -h "$SSOD_ROOT"; then
    :
else
    fail "df failed for $SSOD_ROOT"
fi

echo "=== STORAGE TYPE ==="
if command -v findmnt >/dev/null 2>&1; then
    if findmnt -T "$SSOD_ROOT" -o TARGET,SOURCE,FSTYPE,OPTIONS; then
        :
    else
        fail "findmnt failed for $SSOD_ROOT"
    fi
else
    fail "findmnt command missing"
fi

STORAGE_FS_DEVICE="$(df -P -B1 "$SSOD_ROOT" 2>/dev/null | awk 'NR==2 {print $1}')"
STORAGE_TOTAL_BYTES="$(df -P -B1 "$SSOD_ROOT" 2>/dev/null | awk 'NR==2 {print $2}')"
STORAGE_USED_BYTES="$(df -P -B1 "$SSOD_ROOT" 2>/dev/null | awk 'NR==2 {print $3}')"
STORAGE_AVAILABLE_BYTES="$(df -P -B1 "$SSOD_ROOT" 2>/dev/null | awk 'NR==2 {print $4}')"
STORAGE_USE_PERCENT="$(df -P -B1 "$SSOD_ROOT" 2>/dev/null | awk 'NR==2 {print $5}')"
STORAGE_DF_MOUNTPOINT="$(df -P -B1 "$SSOD_ROOT" 2>/dev/null | awk 'NR==2 {print $6}')"

STORAGE_MOUNT_TARGET=""
STORAGE_SOURCE=""
STORAGE_FSTYPE=""
STORAGE_OPTIONS=""
if command -v findmnt >/dev/null 2>&1; then
    STORAGE_MOUNT_TARGET="$(findmnt -T "$SSOD_ROOT" -n -o TARGET 2>/dev/null | head -n1 || true)"
    STORAGE_SOURCE="$(findmnt -T "$SSOD_ROOT" -n -o SOURCE 2>/dev/null | head -n1 || true)"
    STORAGE_FSTYPE="$(findmnt -T "$SSOD_ROOT" -n -o FSTYPE 2>/dev/null | head -n1 || true)"
    STORAGE_OPTIONS="$(findmnt -T "$SSOD_ROOT" -n -o OPTIONS 2>/dev/null | head -n1 || true)"
fi

if [ "$STORAGE_FSTYPE" = "overlay" ]; then
    STORAGE_CLASSIFICATION="local_container_overlay"
else
    STORAGE_CLASSIFICATION="observed_non_overlay"
fi

if [ "$FAIL_COUNT" -eq "$STORAGE_FAIL_COUNT_BEFORE" ]; then
    STORAGE_STATUS="PASS"
    pass "project/storage foundation verification"
else
    STORAGE_STATUS="FAIL"
fi


echo
echo "=== MACHINE-READABLE EVIDENCE ==="

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

REPO_ROOT="${SSOD_REPO_ROOT:-}"

if [ -z "$REPO_ROOT" ]; then
    REPO_ROOT="$(
        git -C "$SCRIPT_DIR" rev-parse --show-toplevel 2>/dev/null || true
    )"
fi

if [ -z "$REPO_ROOT" ]; then
    REPO_ROOT="$(
        git -C "$PWD" rev-parse --show-toplevel 2>/dev/null || true
    )"
fi

if [ -n "$REPO_ROOT" ]; then
    EVIDENCE_MODE="CANONICAL_REPO"
    EVIDENCE_DIR="${SSOD_RUNTIME_EVIDENCE_DIR:-$REPO_ROOT/artifacts/preflight/environment}"
else
    EVIDENCE_MODE="STANDALONE_BOOTSTRAP"
    EVIDENCE_DIR="${SSOD_RUNTIME_EVIDENCE_DIR:-/root/ssod_runtime_evidence}"
    warn "repository root unavailable; using standalone bootstrap evidence directory"
fi

echo "evidence_mode=$EVIDENCE_MODE"
echo "evidence_dir=$EVIDENCE_DIR"

if [ -n "${EVIDENCE_DIR:-}" ]; then
    if mkdir -p "$EVIDENCE_DIR" 2>/dev/null; then
        EVIDENCE_DIR_READY=true
    else
        EVIDENCE_DIR_READY=false
        fail "cannot create runtime evidence directory: $EVIDENCE_DIR"
    fi
else
    EVIDENCE_DIR_READY=false
fi

VERIFIER_SHA256="$(
    sha256sum "$0" 2>/dev/null |
    awk '{print $1}'
)"

STARTUP_WARNING_PRESENT=false
if [ -f /root/onstart.sh ] &&
   grep -qx 'entrypoint.sh' /root/onstart.sh 2>/dev/null &&
   ! command -v entrypoint.sh >/dev/null 2>&1; then
    STARTUP_WARNING_PRESENT=true
fi

export EVIDENCE_DIR
export EVIDENCE_MODE
export EXPECTED_IMAGE
export EXPECTED_TAG
export EXPECTED_IMAGE_UUID
export EXPECTED_DIGEST
export OBS_CONTAINER_ID
export OBS_CONTAINER_LABEL
export OBS_VAST_DEVICE_IDXS
export IMAGE_UUID
export ACTUAL_STATUS
export REGISTRY_DIGEST
export VERIFIER_SHA256
export STARTUP_WARNING_PRESENT
export SSOD_ROOT
export SSOD_PROJECT_DIR
export SSOD_DATA_DIR
export SSOD_IMAGES_DIR
export SSOD_COCO_DIR
export SSOD_ARTIFACTS_DIR
export SSOD_CHECKPOINTS_DIR
export SSOD_CACHE_DIR
export STORAGE_STATUS
export STORAGE_FS_DEVICE
export STORAGE_TOTAL_BYTES
export STORAGE_USED_BYTES
export STORAGE_AVAILABLE_BYTES
export STORAGE_USE_PERCENT
export STORAGE_DF_MOUNTPOINT
export STORAGE_MOUNT_TARGET
export STORAGE_SOURCE
export STORAGE_FSTYPE
export STORAGE_OPTIONS
export STORAGE_CLASSIFICATION
export PASS_COUNT
export FAIL_COUNT
export WARN_COUNT

if [ "$EVIDENCE_DIR_READY" = true ]; then
python - <<'PY'
import json
import os
import platform
import re
import socket
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import torch
import mmengine
import mmcv
import mmdet


def run(cmd):
    try:
        return subprocess.check_output(
            cmd,
            text=True,
            stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return ""


def read_os_release():
    result = {}
    try:
        with open("/etc/os-release", "r", encoding="utf-8") as f:
            for line in f:
                line = line.rstrip("\n")
                if "=" not in line:
                    continue
                key, value = line.split("=", 1)
                result[key] = value.strip().strip('"')
    except Exception:
        pass
    return result


def first_gpu_field(field):
    value = run([
        "nvidia-smi",
        f"--query-gpu={field}",
        "--format=csv,noheader,nounits",
    ])
    return value.splitlines()[0].strip() if value else None


os_release = read_os_release()
nvidia_smi_text = run(["nvidia-smi"])

driver_supported_cuda = None
m = re.search(r"CUDA Version:\s*([0-9.]+)", nvidia_smi_text)
if m:
    driver_supported_cuda = m.group(1)

gpu_name = first_gpu_field("name")
gpu_memory_mib = first_gpu_field("memory.total")
driver_version = first_gpu_field("driver_version")

try:
    gpu_memory_mib = int(float(gpu_memory_mib)) if gpu_memory_mib else None
except Exception:
    gpu_memory_mib = None

expected_digest = os.environ.get("EXPECTED_DIGEST", "")
observed_digest = os.environ.get("REGISTRY_DIGEST", "")

runtime_fail_count = int(os.environ.get("FAIL_COUNT", "0"))
runtime_status = "PASS" if runtime_fail_count == 0 else "FAIL"

timestamp = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

environment = {
    "schema_version": "1.1",
    "artifact_type": "VAST_RUNTIME_ENVIRONMENT",
    "status": runtime_status,
    "captured_at_utc": timestamp,

    "instance": {
        "container_id": os.environ.get("OBS_CONTAINER_ID") or None,
        "container_label": os.environ.get("OBS_CONTAINER_LABEL") or None,
        "hostname": socket.gethostname(),
        "vast_device_idxs": os.environ.get("OBS_VAST_DEVICE_IDXS") or None,
        "vast_actual_status": os.environ.get("ACTUAL_STATUS") or None,
    },

    "os": {
        "pretty_name": os_release.get("PRETTY_NAME"),
        "version_id": os_release.get("VERSION_ID"),
        "kernel": platform.release(),
        "platform": platform.platform(),
    },

    "python": {
        "version": platform.python_version(),
    },

    "pytorch": {
        "version": str(torch.__version__),
        "cuda_runtime": str(torch.version.cuda),
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_device_count": int(torch.cuda.device_count()),
        "cudnn_version": torch.backends.cudnn.version(),
        "cudnn_enabled": bool(torch.backends.cudnn.enabled),
    },

    "gpu": {
        "name": gpu_name,
        "memory_total_mib": gpu_memory_mib,
        "nvidia_driver": driver_version,
        "driver_supported_cuda": driver_supported_cuda,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "vast_device_idxs": os.environ.get("OBS_VAST_DEVICE_IDXS") or None,
    },

    "openmmlab": {
        "mmengine": str(mmengine.__version__),
        "mmcv": str(mmcv.__version__),
        "mmdet": str(mmdet.__version__),
    },

    "docker": {
        "expected_image": os.environ.get("EXPECTED_IMAGE"),
        "expected_tag": os.environ.get("EXPECTED_TAG"),
        "expected_image_uuid": os.environ.get("EXPECTED_IMAGE_UUID"),
        "observed_image_uuid": os.environ.get("IMAGE_UUID") or None,
        "expected_digest": expected_digest,
        "observed_registry_digest": observed_digest or None,
        "digest_match": observed_digest == expected_digest,
    },

    "verifier": {
        "evidence_mode": os.environ.get("EVIDENCE_MODE") or None,
        "script_sha256": os.environ.get("VERIFIER_SHA256") or None,
        "pass_count_before_evidence_write": int(
            os.environ.get("PASS_COUNT", "0")
        ),
        "fail_count_before_evidence_write": runtime_fail_count,
        "warning_count": int(os.environ.get("WARN_COUNT", "0")),
        "startup_entrypoint_warning_present":
            os.environ.get("STARTUP_WARNING_PRESENT", "false") == "true",
    },
}

docker_identity = {
    "schema_version": "1.1",
    "artifact_type": "DOCKER_IMAGE_IDENTITY",
    "status": (
        "PASS"
        if (
            os.environ.get("IMAGE_UUID") ==
            os.environ.get("EXPECTED_IMAGE_UUID")
            and observed_digest == expected_digest
        )
        else "FAIL"
    ),
    "verified_at_utc": timestamp,
    "instance_id": os.environ.get("OBS_CONTAINER_ID") or None,

    "expected": {
        "repository": os.environ.get("EXPECTED_IMAGE"),
        "tag": os.environ.get("EXPECTED_TAG"),
        "image_uuid": os.environ.get("EXPECTED_IMAGE_UUID"),
        "digest": expected_digest,
    },

    "observed": {
        "vast_image_uuid": os.environ.get("IMAGE_UUID") or None,
        "registry_digest": observed_digest or None,
    },

    "checks": {
        "vast_image_uuid_match":
            os.environ.get("IMAGE_UUID") ==
            os.environ.get("EXPECTED_IMAGE_UUID"),

        "registry_digest_match":
            observed_digest == expected_digest,
    },

    "verification_sources": [
        "Vast instance API image_uuid",
        "Docker Hub registry Docker-Content-Digest",
    ],

    "verifier_sha256": os.environ.get("VERIFIER_SHA256") or None,
}


def env_int(name):
    value = os.environ.get(name)
    try:
        return int(value) if value not in (None, "") else None
    except Exception:
        return None


storage_paths = {
    "project": os.environ.get("SSOD_PROJECT_DIR"),
    "data": os.environ.get("SSOD_DATA_DIR"),
    "images": os.environ.get("SSOD_IMAGES_DIR"),
    "coco": os.environ.get("SSOD_COCO_DIR"),
    "artifacts": os.environ.get("SSOD_ARTIFACTS_DIR"),
    "checkpoints": os.environ.get("SSOD_CHECKPOINTS_DIR"),
    "cache": os.environ.get("SSOD_CACHE_DIR"),
}

storage_path_checks = {}
for role, path in storage_paths.items():
    storage_path_checks[role] = {
        "path": path,
        "exists": bool(path and Path(path).is_dir()),
        "writable": bool(path and os.access(path, os.W_OK)),
    }

storage_environment = {
    "schema_version": "1.1",
    "artifact_type": "STORAGE_ENVIRONMENT",
    "status": os.environ.get("STORAGE_STATUS", "FAIL"),
    "captured_at_utc": timestamp,
    "instance_id": os.environ.get("OBS_CONTAINER_ID") or None,
    "ssod_root": os.environ.get("SSOD_ROOT") or None,
    "storage_classification":
        os.environ.get("STORAGE_CLASSIFICATION") or None,
    "paths": storage_path_checks,
    "filesystem": {
        "df_device": os.environ.get("STORAGE_FS_DEVICE") or None,
        "total_bytes": env_int("STORAGE_TOTAL_BYTES"),
        "used_bytes": env_int("STORAGE_USED_BYTES"),
        "available_bytes": env_int("STORAGE_AVAILABLE_BYTES"),
        "use_percent": os.environ.get("STORAGE_USE_PERCENT") or None,
        "df_mountpoint": os.environ.get("STORAGE_DF_MOUNTPOINT") or None,
        "mount_target": os.environ.get("STORAGE_MOUNT_TARGET") or None,
        "source": os.environ.get("STORAGE_SOURCE") or None,
        "fstype": os.environ.get("STORAGE_FSTYPE") or None,
        "options": os.environ.get("STORAGE_OPTIONS") or None,
    },
    "checks": {
        "all_required_paths_exist": all(
            x["exists"] for x in storage_path_checks.values()
        ),
        "all_required_paths_writable": all(
            x["writable"] for x in storage_path_checks.values()
        ),
    },
    "verifier_sha256": os.environ.get("VERIFIER_SHA256") or None,
}

out_dir = Path(os.environ["EVIDENCE_DIR"])
out_dir.mkdir(parents=True, exist_ok=True)

outputs = {
    out_dir / "vast_environment.json": environment,
    out_dir / "docker_image_identity.json": docker_identity,
    out_dir / "storage_environment.json": storage_environment,
}

for path, obj in outputs.items():
    temp = path.with_suffix(path.suffix + ".tmp")

    with open(temp, "w", encoding="utf-8", newline="\n") as f:
        json.dump(
            obj,
            f,
            indent=2,
            ensure_ascii=False,
            sort_keys=True,
        )
        f.write("\n")

    temp.replace(path)

print(f"vast_environment_json={out_dir / 'vast_environment.json'}")
print(
    "docker_image_identity_json="
    f"{out_dir / 'docker_image_identity.json'}"
)
print(
    "storage_environment_json="
    f"{out_dir / 'storage_environment.json'}"
)
PY

    JSON_WRITE_EXIT=$?

    if [ "$JSON_WRITE_EXIT" -eq 0 ] &&
       [ -s "$EVIDENCE_DIR/vast_environment.json" ] &&
       [ -s "$EVIDENCE_DIR/docker_image_identity.json" ] &&
       [ -s "$EVIDENCE_DIR/storage_environment.json" ]; then
        pass "machine-readable runtime evidence written"

        if python - \
            "$EVIDENCE_DIR/vast_environment.json" \
            "$EVIDENCE_DIR/docker_image_identity.json" \
            "$EVIDENCE_DIR/storage_environment.json" <<'PY'
import json
import sys

vast_path, docker_path, storage_path = sys.argv[1], sys.argv[2], sys.argv[3]

try:
    with open(vast_path, "r", encoding="utf-8") as f:
        vast = json.load(f)

    with open(docker_path, "r", encoding="utf-8") as f:
        docker = json.load(f)

    with open(storage_path, "r", encoding="utf-8") as f:
        storage = json.load(f)

    checks = {
        "VAST_JSON_PARSE": True,
        "VAST_STATUS": vast.get("status") == "PASS",
        "DOCKER_JSON_PARSE": True,
        "DOCKER_STATUS": docker.get("status") == "PASS",
        "DIGEST_MATCH":
            docker.get("checks", {}).get("registry_digest_match") is True,
        "IMAGE_UUID_MATCH":
            docker.get("checks", {}).get("vast_image_uuid_match") is True,
        "STORAGE_JSON_PARSE": True,
        "STORAGE_STATUS": storage.get("status") == "PASS",
        "STORAGE_PATHS_EXIST":
            storage.get("checks", {}).get("all_required_paths_exist") is True,
        "STORAGE_PATHS_WRITABLE":
            storage.get("checks", {}).get("all_required_paths_writable") is True,
    }

except Exception as exc:
    print("JSON_VALIDATION_ERROR=" + repr(exc))
    sys.exit(1)

for key, value in checks.items():
    if key.endswith("_PARSE"):
        print(f"{key}=PASS" if value else f"{key}=FAIL")
    else:
        print(f"{key}={value}")

sys.exit(0 if all(checks.values()) else 1)
PY
        then
            pass "machine-readable runtime evidence validated"
        else
            fail "machine-readable runtime evidence validation failed"
        fi
    else
        fail "machine-readable runtime evidence generation failed"
    fi
fi


echo
echo "=== CANONICAL RUNTIME EVIDENCE MATERIALIZATION ==="

# The verifier may have to generate bootstrap evidence under
# /root/ssod_runtime_evidence when no Git repository is present on a fresh
# Vast instance. Regardless of that bootstrap mode, S1 canonical runtime
# evidence must be available under /workspace/ssod/artifacts/preflight/environment.
#
# Only the two runtime identity artifacts are materialized here:
#   - vast_environment.json
#   - docker_image_identity.json
#
# storage_environment.json is intentionally NOT copied here because
# prepare_vast_backup.sh later extends/materializes the canonical storage
# evidence with S1.13 backup information. Re-running this runtime verifier
# must not overwrite that richer storage evidence.

CANONICAL_RUNTIME_EVIDENCE_DIR="${SSOD_RUNTIME_CANONICAL_EVIDENCE_DIR:-$SSOD_ROOT/artifacts/preflight/environment}"
CANONICAL_MATERIALIZATION_OK=true

if ! mkdir -p "$CANONICAL_RUNTIME_EVIDENCE_DIR" 2>/dev/null; then
    fail "cannot create canonical runtime evidence directory: $CANONICAL_RUNTIME_EVIDENCE_DIR"
    CANONICAL_MATERIALIZATION_OK=false
fi

if [ "$CANONICAL_MATERIALIZATION_OK" = true ]; then
    for evidence_name in vast_environment.json docker_image_identity.json; do
        src_path="$EVIDENCE_DIR/$evidence_name"
        dst_path="$CANONICAL_RUNTIME_EVIDENCE_DIR/$evidence_name"

        echo "--- materialize $evidence_name ---"
        echo "source_path=$src_path"
        echo "canonical_path=$dst_path"

        if [ ! -s "$src_path" ]; then
            fail "source runtime evidence missing or empty: $src_path"
            CANONICAL_MATERIALIZATION_OK=false
            continue
        fi

        # If source and canonical paths are already identical, no copy is
        # necessary. Otherwise copy bytes exactly.
        if [ "$src_path" != "$dst_path" ]; then
            if ! cp "$src_path" "$dst_path"; then
                fail "failed to materialize runtime evidence: $evidence_name"
                CANONICAL_MATERIALIZATION_OK=false
                continue
            fi
        fi

        SOURCE_SHA256="$(sha256sum "$src_path" 2>/dev/null | awk '{print $1}')"
        CANONICAL_SHA256="$(sha256sum "$dst_path" 2>/dev/null | awk '{print $1}')"

        echo "SOURCE_SHA256=$SOURCE_SHA256"
        echo "CANONICAL_SHA256=$CANONICAL_SHA256"

        if [ -z "$SOURCE_SHA256" ] ||
           [ -z "$CANONICAL_SHA256" ] ||
           [ "$SOURCE_SHA256" != "$CANONICAL_SHA256" ]; then
            echo "HASH_MATCH=False"
            fail "canonical runtime evidence hash mismatch: $evidence_name"
            CANONICAL_MATERIALIZATION_OK=false
        else
            echo "HASH_MATCH=True"
        fi
    done
fi

if [ "$CANONICAL_MATERIALIZATION_OK" = true ]; then
    if python - \
        "$CANONICAL_RUNTIME_EVIDENCE_DIR/vast_environment.json" \
        "$CANONICAL_RUNTIME_EVIDENCE_DIR/docker_image_identity.json" <<'PY'
import json
import sys

vast_path, docker_path = sys.argv[1], sys.argv[2]

try:
    with open(vast_path, "r", encoding="utf-8") as f:
        vast = json.load(f)

    with open(docker_path, "r", encoding="utf-8") as f:
        docker = json.load(f)

    checks = {
        "VAST_STATUS":
            vast.get("status") == "PASS",

        "DOCKER_STATUS":
            docker.get("status") == "PASS",

        "REGISTRY_DIGEST_MATCH":
            docker.get("checks", {}).get("registry_digest_match") is True,

        "IMAGE_UUID_MATCH":
            docker.get("checks", {}).get("vast_image_uuid_match") is True,
    }

except Exception as exc:
    print("CANONICAL_RUNTIME_JSON_VALIDATION_ERROR=" + repr(exc))
    sys.exit(1)

for key, value in checks.items():
    print(f"{key}={value}")

sys.exit(0 if all(checks.values()) else 1)
PY
    then
        pass "canonical runtime evidence materialized and validated"
        echo "CANONICAL_RUNTIME_EVIDENCE_MATERIALIZATION=PASS"
    else
        fail "canonical runtime evidence validation failed"
        echo "CANONICAL_RUNTIME_EVIDENCE_MATERIALIZATION=FAIL"
    fi
else
    echo "CANONICAL_RUNTIME_EVIDENCE_MATERIALIZATION=FAIL"
fi

echo
echo "========================================"
echo "PASS_COUNT=$PASS_COUNT"
echo "FAIL_COUNT=$FAIL_COUNT"
echo "WARN_COUNT=$WARN_COUNT"

if [ "$FAIL_COUNT" -eq 0 ]; then
    echo "RUNTIME_VERIFICATION=PASS"
    exit 0
else
    echo "RUNTIME_VERIFICATION=FAIL"
    exit 1
fi
