#!/usr/bin/env bash
set -euo pipefail

REPO="${1:-/content/ssl_detection_xray_v2}"
CONDA_ROOT="/content/miniconda"
ENV_NAME="mmdet330"
ENV_DIR="${CONDA_ROOT}/envs/${ENV_NAME}"
PY="${ENV_DIR}/bin/python"

EVIDENCE_DIR="${REPO}/reports/environment"
FREEZE_FILE="${EVIDENCE_DIR}/mmdet330_pip_freeze_2026-07-30.txt"
HASH_FILE="${EVIDENCE_DIR}/mmdet330_environment_sha256_2026-07-30.txt"

PYTORCH_INDEX="https://download.pytorch.org/whl/cu118"
MMCV_WHEEL_INDEX="https://download.openmmlab.com/mmcv/dist/cu118/torch2.1/index.html"
MINICONDA_URL="https://repo.anaconda.com/miniconda/Miniconda3-py310_25.1.1-2-Linux-x86_64.sh"

echo "Repository: ${REPO}"
echo "Environment: ${ENV_DIR}"

test -d "${REPO}" || {
    echo "ERROR: repository does not exist: ${REPO}" >&2
    exit 1
}

test -s "${FREEZE_FILE}" || {
    echo "ERROR: missing environment snapshot: ${FREEZE_FILE}" >&2
    exit 1
}

test -s "${HASH_FILE}" || {
    echo "ERROR: missing SHA-256 manifest: ${HASH_FILE}" >&2
    exit 1
}

echo "[1/8] Verify committed environment evidence"
(
    cd "${EVIDENCE_DIR}"
    sha256sum -c "$(basename "${HASH_FILE}")"
)

if [[ ! -x "${CONDA_ROOT}/bin/conda" ]]; then
    echo "[2/8] Install pinned Miniconda"
    INSTALLER="/tmp/miniconda-py310.sh"

    wget -q "${MINICONDA_URL}" -O "${INSTALLER}"
    test -s "${INSTALLER}"

    bash "${INSTALLER}" -b -p "${CONDA_ROOT}"
else
    echo "[2/8] Existing Miniconda detected"
fi

if [[ -e "${ENV_DIR}" ]]; then
    echo "ERROR: environment already exists: ${ENV_DIR}" >&2
    echo "This script intentionally refuses to overwrite it." >&2
    exit 2
fi

echo "[3/8] Create pinned Python environment"
"${CONDA_ROOT}/bin/conda" create -y \
    -n "${ENV_NAME}" \
    python=3.10.16 \
    pip=25.1

echo "[4/8] Install PyTorch CUDA 11.8 wheels"
"${PY}" -m pip install --no-cache-dir \
    "torch==2.1.0" \
    "torchvision==0.16.0" \
    --index-url "${PYTORCH_INDEX}"

echo "[5/8] Install full MMCV wheel"
"${PY}" -m pip install --no-cache-dir \
    --only-binary=:all: \
    "mmcv==2.1.0" \
    -f "${MMCV_WHEEL_INDEX}"

echo "[6/8] Reproduce exact validated package snapshot"
"${PY}" -m pip install --no-cache-dir \
    --extra-index-url "${PYTORCH_INDEX}" \
    -f "${MMCV_WHEEL_INDEX}" \
    -r "${FREEZE_FILE}"

echo "[7/8] Check dependencies"
"${PY}" -m pip check

echo "[8/8] Run training-environment validation"
export MPLBACKEND=Agg
"${REPO}/scripts/validate_mmdet330_environment.sh" "${REPO}"

echo "MMDETECTION ENVIRONMENT REPRODUCTION: PASS"
