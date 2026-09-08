#!/usr/bin/env bash
set -Eeuo pipefail

# ============================================================================
# SSOD S1 Backup / Recovery Preflight — Vast -> Google Drive
#
# Scope
#   S1.13  Durable-backup foundation:
#           - pinned rclone binary identity
#           - rclone config present with mode 600
#           - gdrive remote configured
#           - remote backup path writable
#           - canonical storage_environment.json materialized/updated
#
#   S1.14  Pre-training checksum verification:
#           - upload deterministic probe
#           - independent round-trip download
#           - SHA-256 + byte-count equality
#           - backup_checksum_verification_test.json
#
#   S1.15  Restore foundation:
#           - durable-source scientific artifact + checkpoint-like probe
#           - restore into a clean path
#           - SHA-256 + byte-count equality
#           - storage_restore_test.json
#
# Important contract boundary
#   This script MUST NOT create the canonical per-attempt:
#     runs/<run_type>/<run_id>/<attempt_id>/backup_sync_manifest.json
#   That artifact belongs to a CLOSED training attempt and remains mandatory
#   later. S1 is a pre-training operational backup/recovery preflight.
#
# Typical local invocation (Windows Git Bash / PowerShell):
#   bash scripts/backup/prepare_vast_backup.sh --remote <HOST> <PORT>
#
# Internal remote invocation:
#   prepare_vast_backup.sh --on-remote
#
# This script intentionally does NOT:
#   - store/copy OAuth client secret, access token, or refresh token in evidence
#   - commit or back up rclone.conf
#   - train SUP/SSL
#   - create a trained-model checkpoint
#   - create per-attempt backup_sync_manifest.json
#   - change scientific protocol, split membership, or L/U membership
# ============================================================================

SCRIPT_VERSION="1.0.0"

REMOTE_MODE="false"
REMOTE_HOST=""
REMOTE_PORT=""
REMOTE_USER="${REMOTE_USER:-root}"

SSOD_ROOT="${SSOD_ROOT:-/workspace/ssod}"

RCLONE_VERSION="1.75.1"
RCLONE_EXPECTED_SHA256="f66d8c1d552ad90296a11bc8b46d56a7fa5da1a7fa05e7ca522d95df92c4a4c0"
RCLONE_DIR="$SSOD_ROOT/tools/rclone-v${RCLONE_VERSION}"
RCLONE="$SSOD_ROOT/tools/rclone"
RCLONE_CONF="${RCLONE_CONF:-$SSOD_ROOT/config/rclone/rclone.conf}"

RCLONE_REMOTE_NAME="${RCLONE_REMOTE_NAME:-gdrive:}"
BACKUP_BASE="${BACKUP_BASE:-gdrive:SSOD_Backup}"

EVIDENCE_DIR="$SSOD_ROOT/artifacts/preflight/environment"
STANDALONE_ENV_DIR="${STANDALONE_ENV_DIR:-/root/ssod_runtime_evidence}"

S1_14_EVIDENCE="$EVIDENCE_DIR/backup_checksum_verification_test.json"
S1_15_EVIDENCE="$EVIDENCE_DIR/storage_restore_test.json"
STORAGE_ENV_CANONICAL="$EVIDENCE_DIR/storage_environment.json"
STORAGE_ENV_STANDALONE="$STANDALONE_ENV_DIR/storage_environment.json"

usage() {
  cat <<'EOF'
Usage:
  prepare_vast_backup.sh --remote HOST PORT
  prepare_vast_backup.sh --on-remote

Environment overrides:
  SSOD_ROOT             Default: /workspace/ssod
  REMOTE_USER           Default: root (local wrapper)
  RCLONE_CONF           Default: /workspace/ssod/config/rclone/rclone.conf
  RCLONE_REMOTE_NAME    Default: gdrive:
  BACKUP_BASE           Default: gdrive:SSOD_Backup
  INSTANCE_ID           Optional override for Vast instance ID

Authentication:
  OAuth is intentionally NOT automated or embedded.
  If rclone.conf is missing or gdrive: is not configured, this script exits
  BLOCKED with RCLONE_AUTH_REQUIRED=YES. Configure OAuth separately, then
  rerun the same command.
EOF
}

log()  { printf '%s\n' "$*"; }
pass() { printf '[PASS] %s\n' "$*"; }
warn() { printf '[WARN] %s\n' "$*"; }
fail() {
  printf '[FAIL] %s\n' "$*" >&2
  printf 'S1_BACKUP_PREPARATION=FAIL\n' >&2
  exit 2
}

auth_block() {
  printf '[BLOCKED] %s\n' "$*" >&2
  printf 'RCLONE_AUTH_REQUIRED=YES\n' >&2
  printf 'S1_BACKUP_PREPARATION=BLOCKED\n' >&2
  exit 30
}

sha256_of() {
  sha256sum "$1" | awk '{print $1}'
}

pid1_env_get() {
  local key="$1"
  if [[ ! -r /proc/1/environ ]]; then
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

detect_instance_id() {
  local value="${INSTANCE_ID:-}"

  if [[ -z "$value" ]]; then
    value="${CONTAINER_ID:-}"
  fi

  if [[ -z "$value" ]]; then
    value="$(pid1_env_get CONTAINER_ID || true)"
  fi

  if [[ -z "$value" ]]; then
    local host
    host="$(hostname 2>/dev/null || true)"
    if [[ "$host" =~ ^C\.([0-9]+)$ ]]; then
      value="${BASH_REMATCH[1]}"
    fi
  fi

  if [[ -z "$value" ]]; then
    fail "cannot determine Vast instance ID; set INSTANCE_ID explicitly"
  fi

  printf '%s' "$value"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --remote)
      [[ $# -ge 3 ]] || { usage; exit 2; }
      REMOTE_HOST="$2"
      REMOTE_PORT="$3"
      shift 3
      ;;
    --on-remote)
      REMOTE_MODE="true"
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "[FAIL] unknown argument: $1"
      usage
      exit 2
      ;;
  esac
done

# ----------------------------------------------------------------------------
# LOCAL WRAPPER
# ----------------------------------------------------------------------------
if [[ "$REMOTE_MODE" != "true" ]]; then
  [[ -n "$REMOTE_HOST" && -n "$REMOTE_PORT" ]] || {
    usage
    exit 2
  }

  for cmd in ssh scp sha256sum; do
    command -v "$cmd" >/dev/null 2>&1 || {
      echo "[FAIL] local command missing: $cmd"
      echo "NEW_VAST_BACKUP_PREPARATION=FAIL"
      exit 21
    }
  done

  SCRIPT_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"
  LOCAL_SHA="$(sha256_of "$SCRIPT_PATH")"
  TARGET="${REMOTE_USER}@${REMOTE_HOST}"
  REMOTE_SCRIPT="/tmp/ssod_s1/prepare_vast_backup.sh"

  echo "========================================"
  echo "SSOD NEW VAST BACKUP/RECOVERY PREPARER"
  echo "========================================"
  echo "target=$TARGET"
  echo "port=$REMOTE_PORT"
  echo "script_version=$SCRIPT_VERSION"
  echo "LOCAL_SCRIPT_SHA256=$LOCAL_SHA"

  echo
  echo "=== PREPARE REMOTE STAGING ==="
  ssh \
    -p "$REMOTE_PORT" \
    -o StrictHostKeyChecking=accept-new \
    "$TARGET" \
    "mkdir -p /tmp/ssod_s1"

  echo
  echo "=== TRANSFER SCRIPT ==="
  scp \
    -P "$REMOTE_PORT" \
    -o StrictHostKeyChecking=accept-new \
    "$SCRIPT_PATH" \
    "${TARGET}:${REMOTE_SCRIPT}"

  REMOTE_SHA="$(
    ssh \
      -p "$REMOTE_PORT" \
      -o StrictHostKeyChecking=accept-new \
      "$TARGET" \
      "sha256sum '$REMOTE_SCRIPT' | awk '{print \$1}'"
  )"

  echo "REMOTE_SCRIPT_SHA256=$REMOTE_SHA"

  if [[ "$LOCAL_SHA" != "$REMOTE_SHA" ]]; then
    echo "SCRIPT_HASH_MATCH=False"
    echo "NEW_VAST_BACKUP_PREPARATION=FAIL"
    exit 22
  fi

  echo "SCRIPT_HASH_MATCH=True"

  echo
  echo "=== RUN REMOTE BACKUP/RECOVERY PREPARER ==="

  set +e
  ssh \
    -p "$REMOTE_PORT" \
    -o StrictHostKeyChecking=accept-new \
    "$TARGET" \
    "chmod +x '$REMOTE_SCRIPT' && SSOD_ROOT='$SSOD_ROOT' '$REMOTE_SCRIPT' --on-remote"
  REMOTE_RC=$?
  set -e

  echo
  if [[ "$REMOTE_RC" -eq 0 ]]; then
    echo "========================================"
    echo "NEW_VAST_BACKUP_PREPARATION=PASS"
    echo "========================================"
    exit 0
  elif [[ "$REMOTE_RC" -eq 30 ]]; then
    echo "========================================"
    echo "NEW_VAST_BACKUP_PREPARATION=BLOCKED"
    echo "RCLONE_AUTH_REQUIRED=YES"
    echo "========================================"
    exit 30
  else
    echo "========================================"
    echo "NEW_VAST_BACKUP_PREPARATION=FAIL"
    echo "========================================"
    exit "$REMOTE_RC"
  fi
fi

# ----------------------------------------------------------------------------
# REMOTE EXECUTION
# ----------------------------------------------------------------------------
for cmd in python curl sha256sum stat hostname; do
  command -v "$cmd" >/dev/null 2>&1 || fail "remote command missing: $cmd"
done

INSTANCE_ID="$(detect_instance_id)"
BACKUP_ROOT="${BACKUP_BASE}/vast_${INSTANCE_ID}"
PREFLIGHT_REMOTE="${BACKUP_ROOT}/preflight"

mkdir -p \
  "$SSOD_ROOT/tools" \
  "$SSOD_ROOT/config/rclone" \
  "$EVIDENCE_DIR"

SCRIPT_SHA256="$(sha256_of "$0")"

echo "========================================"
echo "SSOD S1.13-S1.15 BACKUP/RECOVERY PREFLIGHT"
echo "========================================"
echo "script_version=$SCRIPT_VERSION"
echo "script_sha256=$SCRIPT_SHA256"
echo "instance_id=$INSTANCE_ID"
echo "backup_root=$BACKUP_ROOT"

# ----------------------------------------------------------------------------
# RCLONE PINNED INSTALL / IDENTITY
# ----------------------------------------------------------------------------
echo
echo "=== RCLONE PINNED INSTALL / IDENTITY ==="

if [[ ! -x "$RCLONE" ]] || [[ "$(sha256_of "$RCLONE" 2>/dev/null || true)" != "$RCLONE_EXPECTED_SHA256" ]]; then
  log "rclone missing or identity mismatch; installing pinned standalone rclone v${RCLONE_VERSION}"

  ZIP="/tmp/rclone-v${RCLONE_VERSION}-linux-amd64.zip"
  EXTRACT="/tmp/rclone_extract_${RCLONE_VERSION}"

  rm -rf "$EXTRACT"
  mkdir -p "$EXTRACT"

  curl -L --fail --retry 5 --retry-delay 3 \
    -o "$ZIP" \
    "https://downloads.rclone.org/v${RCLONE_VERSION}/rclone-v${RCLONE_VERSION}-linux-amd64.zip"

  python - "$ZIP" "$EXTRACT" <<'PY'
from pathlib import Path
import sys
import zipfile

src = Path(sys.argv[1])
dst = Path(sys.argv[2])

with zipfile.ZipFile(src) as z:
    members = z.infolist()
    unsafe = [
        m.filename
        for m in members
        if Path(m.filename).is_absolute()
        or ".." in Path(m.filename).parts
    ]
    if unsafe:
        raise SystemExit("unsafe path in rclone ZIP")
    z.extractall(dst)
PY

  EXTRACTED="$EXTRACT/rclone-v${RCLONE_VERSION}-linux-amd64/rclone"
  [[ -f "$EXTRACTED" ]] || fail "extracted rclone binary missing"

  EXTRACTED_SHA="$(sha256_of "$EXTRACTED")"
  echo "extracted_rclone_sha256=$EXTRACTED_SHA"

  [[ "$EXTRACTED_SHA" == "$RCLONE_EXPECTED_SHA256" ]] \
    || fail "pinned rclone binary SHA-256 mismatch"

  mkdir -p "$RCLONE_DIR"
  cp "$EXTRACTED" "$RCLONE_DIR/rclone"
  chmod 0755 "$RCLONE_DIR/rclone"
  ln -sfn "$RCLONE_DIR/rclone" "$RCLONE"
fi

RCLONE_SHA="$(sha256_of "$RCLONE")"
RCLONE_VERSION_LINE="$("$RCLONE" version | head -n1)"

echo "rclone_version=$RCLONE_VERSION_LINE"
echo "rclone_binary_sha256=$RCLONE_SHA"

[[ "$RCLONE_VERSION_LINE" == "rclone v${RCLONE_VERSION}" ]] \
  || fail "rclone version mismatch"

[[ "$RCLONE_SHA" == "$RCLONE_EXPECTED_SHA256" ]] \
  || fail "rclone binary SHA-256 mismatch"

pass "pinned rclone identity verified"

# ----------------------------------------------------------------------------
# AUTH CONFIG — never emit secrets/tokens
# ----------------------------------------------------------------------------
echo
echo "=== RCLONE AUTH CONFIG CHECK ==="

[[ -f "$RCLONE_CONF" ]] \
  || auth_block "rclone config missing: $RCLONE_CONF"

chmod 600 "$RCLONE_CONF"
CONFIG_PERMISSIONS="$(stat -c '%a' "$RCLONE_CONF")"

echo "rclone_config_path=$RCLONE_CONF"
echo "rclone_config_permissions=$CONFIG_PERMISSIONS"

[[ "$CONFIG_PERMISSIONS" == "600" ]] \
  || fail "rclone config permissions must be 600"

REMOTES="$("$RCLONE" --config "$RCLONE_CONF" listremotes 2>/dev/null || true)"
printf '%s\n' "$REMOTES" | grep -qx "$RCLONE_REMOTE_NAME" \
  || auth_block "required rclone remote not configured: $RCLONE_REMOTE_NAME"

echo "configured_remote=$RCLONE_REMOTE_NAME"
pass "rclone auth configuration available without exposing credentials"

# ============================================================================
# S1.13 — DURABLE BACKUP FOUNDATION
# ============================================================================
echo
echo "============================================================"
echo "S1.13 — DURABLE BACKUP FOUNDATION"
echo "============================================================"

S1_13_LOCAL="/tmp/s1_13_write_test.txt"
S1_13_REMOTE="${PREFLIGHT_REMOTE}/s1_13_write_test.txt"

"$RCLONE" --config "$RCLONE_CONF" mkdir "$PREFLIGHT_REMOTE"

printf 'S1.13 write test - Vast %s\n' "$INSTANCE_ID" > "$S1_13_LOCAL"

"$RCLONE" \
  --config "$RCLONE_CONF" \
  copyto \
  "$S1_13_LOCAL" \
  "$S1_13_REMOTE"

S1_13_LISTING="$(
  "$RCLONE" \
    --config "$RCLONE_CONF" \
    lsf "$S1_13_REMOTE"
)"

echo "write_test_object=$S1_13_LISTING"

[[ "$S1_13_LISTING" == "s1_13_write_test.txt" ]] \
  || fail "S1.13 remote write-test object not found"

# Materialize/update canonical storage_environment.json.
BASE_STORAGE_ENV=""

if [[ -f "$STORAGE_ENV_STANDALONE" ]]; then
  BASE_STORAGE_ENV="$STORAGE_ENV_STANDALONE"
elif [[ -f "$STORAGE_ENV_CANONICAL" ]]; then
  BASE_STORAGE_ENV="$STORAGE_ENV_CANONICAL"
else
  fail "storage_environment.json missing; run verify_vast_runtime.sh first"
fi

export INSTANCE_ID
export BACKUP_ROOT
export S1_13_REMOTE
export S1_13_LISTING
export RCLONE
export RCLONE_CONF
export RCLONE_SHA
export RCLONE_VERSION_LINE
export CONFIG_PERMISSIONS
export BASE_STORAGE_ENV
export STORAGE_ENV_CANONICAL
export SCRIPT_SHA256
export RCLONE_REMOTE_NAME

python - <<'PY'
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import os

base = Path(os.environ["BASE_STORAGE_ENV"])
out = Path(os.environ["STORAGE_ENV_CANONICAL"])

def sha256_file(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

obj = json.loads(base.read_text(encoding="utf-8"))
base_sha = sha256_file(base)

checks = obj.setdefault("checks", {})

backup_checks = {
    "rclone_binary_sha256_matches":
        os.environ["RCLONE_SHA"] ==
        "f66d8c1d552ad90296a11bc8b46d56a7fa5da1a7fa05e7ca522d95df92c4a4c0",
    "rclone_version_is_1_75_1":
        os.environ["RCLONE_VERSION_LINE"] == "rclone v1.75.1",
    "rclone_config_exists":
        Path(os.environ["RCLONE_CONF"]).is_file(),
    "rclone_config_permissions_600":
        os.environ["CONFIG_PERMISSIONS"] == "600",
    "gdrive_remote_configured":
        os.environ["RCLONE_REMOTE_NAME"] == "gdrive:",
    "backup_write_test_object_exists":
        os.environ["S1_13_LISTING"] == "s1_13_write_test.txt",
    "backup_source_artifacts_exists":
        Path("/workspace/ssod/artifacts").is_dir(),
    "backup_source_checkpoints_exists":
        Path("/workspace/ssod/checkpoints").is_dir(),
}

checks.update(backup_checks)

obj["canonical_materialization"] = {
    "source_path": str(base),
    "source_sha256_before_update": base_sha,
    "canonical_path": str(out),
}

obj["backup"] = {
    "direction": "Vast -> Google Drive",
    "purpose":
        "second durable copy of scientific artifacts and checkpoints outside the Vast instance",
    "source_paths": {
        "artifacts": "/workspace/ssod/artifacts",
        "checkpoints": "/workspace/ssod/checkpoints",
    },
    "destination": {
        "remote_name": os.environ["RCLONE_REMOTE_NAME"],
        "backup_root": os.environ["BACKUP_ROOT"],
        "preflight_test_object": os.environ["S1_13_REMOTE"],
        "write_test_verified": True,
    },
    "tool": {
        "name": "rclone",
        "version": os.environ["RCLONE_VERSION_LINE"],
        "binary_path": os.environ["RCLONE"],
        "binary_sha256": os.environ["RCLONE_SHA"],
    },
    "authentication": {
        "provider": "Google Drive OAuth via rclone",
        "configuration_path": os.environ["RCLONE_CONF"],
        "configuration_permissions": os.environ["CONFIG_PERMISSIONS"],
        "credentials_embedded_in_evidence": False,
    },
    "training_io_policy": {
        "training_uses_google_drive_directly": False,
        "google_drive_role": "durable backup / transport only",
    },
    "write_test": {
        "status": "PASS",
        "remote_object_verified": os.environ["S1_13_LISTING"],
        "verified_at_utc": datetime.now(timezone.utc).isoformat(),
    },
}

obj["s1_13_backup_path_writable"] = (
    "PASS" if all(backup_checks.values()) else "FAIL"
)

obj["backup_preflight_script"] = {
    "version": "1.0.0",
    "sha256": os.environ["SCRIPT_SHA256"],
}

obj["updated_at_utc"] = datetime.now(timezone.utc).isoformat()
obj["status"] = "PASS" if all(checks.values()) else "FAIL"

out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(
    json.dumps(obj, indent=2) + "\n",
    encoding="utf-8",
)

parsed = json.loads(out.read_text(encoding="utf-8"))
valid = (
    parsed.get("status") == "PASS"
    and parsed.get("s1_13_backup_path_writable") == "PASS"
    and all(parsed.get("checks", {}).values())
)

print("storage_environment_path =", out)
print("storage_environment_status =", parsed.get("status"))
print("S1_13_STORAGE_ENVIRONMENT_VALIDATION =", "PASS" if valid else "FAIL")

if not valid:
    raise SystemExit(2)
PY

S1_13_STORAGE_SHA="$(sha256_of "$STORAGE_ENV_CANONICAL")"
echo "storage_environment_sha256=$S1_13_STORAGE_SHA"
echo "S1_13_STATUS=PASS"

# ============================================================================
# S1.14 — CHECKSUM ROUND-TRIP VERIFICATION
# ============================================================================
echo
echo "============================================================"
echo "S1.14 — BACKUP CHECKSUM ROUND-TRIP VERIFICATION"
echo "============================================================"

S1_14_LOCAL_SRC="/tmp/s1_14_checksum_source.bin"
S1_14_LOCAL_VERIFY="/tmp/s1_14_checksum_verify.bin"
S1_14_REMOTE="${PREFLIGHT_REMOTE}/s1_14_checksum_probe.bin"

python - "$S1_14_LOCAL_SRC" "$INSTANCE_ID" <<'PY'
from pathlib import Path
import sys

p = Path(sys.argv[1])
instance_id = sys.argv[2]

p.write_bytes(
    b"SSOD_S1_14_BACKUP_CHECKSUM_VERIFICATION\n"
    + f"instance={instance_id}\n".encode("utf-8")
    + b"purpose=pretraining_backup_integrity_probe\n"
)
PY

S1_14_SOURCE_SHA="$(sha256_of "$S1_14_LOCAL_SRC")"
S1_14_SOURCE_BYTES="$(stat -c '%s' "$S1_14_LOCAL_SRC")"

echo "source_bytes=$S1_14_SOURCE_BYTES"
echo "source_sha256=$S1_14_SOURCE_SHA"

"$RCLONE" \
  --config "$RCLONE_CONF" \
  copyto \
  "$S1_14_LOCAL_SRC" \
  "$S1_14_REMOTE"

S1_14_LISTING="$(
  "$RCLONE" \
    --config "$RCLONE_CONF" \
    lsf "$S1_14_REMOTE"
)"

echo "remote_listing=$S1_14_LISTING"

[[ "$S1_14_LISTING" == "s1_14_checksum_probe.bin" ]] \
  || fail "S1.14 remote checksum object not found"

rm -f "$S1_14_LOCAL_VERIFY"

"$RCLONE" \
  --config "$RCLONE_CONF" \
  copyto \
  "$S1_14_REMOTE" \
  "$S1_14_LOCAL_VERIFY"

S1_14_VERIFY_SHA="$(sha256_of "$S1_14_LOCAL_VERIFY")"
S1_14_VERIFY_BYTES="$(stat -c '%s' "$S1_14_LOCAL_VERIFY")"

echo "roundtrip_bytes=$S1_14_VERIFY_BYTES"
echo "roundtrip_sha256=$S1_14_VERIFY_SHA"

[[ "$S1_14_SOURCE_SHA" == "$S1_14_VERIFY_SHA" ]] \
  || fail "S1.14 SHA-256 round-trip mismatch"

[[ "$S1_14_SOURCE_BYTES" == "$S1_14_VERIFY_BYTES" ]] \
  || fail "S1.14 byte-count round-trip mismatch"

export S1_14_SOURCE_SHA
export S1_14_VERIFY_SHA
export S1_14_SOURCE_BYTES
export S1_14_VERIFY_BYTES
export S1_14_REMOTE
export S1_14_EVIDENCE

python - <<'PY'
from datetime import datetime, timezone
from pathlib import Path
import json
import os

checks = {
    "remote_object_present": True,
    "source_and_roundtrip_byte_count_match":
        int(os.environ["S1_14_SOURCE_BYTES"]) ==
        int(os.environ["S1_14_VERIFY_BYTES"]),
    "source_and_roundtrip_sha256_match":
        os.environ["S1_14_SOURCE_SHA"] ==
        os.environ["S1_14_VERIFY_SHA"],
}

obj = {
    "schema_version": "1.1",
    "artifact_type": "PREFLIGHT_BACKUP_CHECKSUM_VERIFICATION",
    "lifecycle_scope": "S1_PRETRAINING_OPERATIONAL_TEST",
    "status": "PASS" if all(checks.values()) else "FAIL",
    "verification_status": "PASS" if all(checks.values()) else "FAIL",
    "instance_id": os.environ["INSTANCE_ID"],
    "source": {
        "path": "/tmp/s1_14_checksum_source.bin",
        "file_count": 1,
        "bytes": int(os.environ["S1_14_SOURCE_BYTES"]),
        "sha256": os.environ["S1_14_SOURCE_SHA"],
    },
    "destination": {
        "identity": os.environ["S1_14_REMOTE"],
        "provider": "Google Drive via rclone",
    },
    "roundtrip_verification": {
        "path": "/tmp/s1_14_checksum_verify.bin",
        "bytes": int(os.environ["S1_14_VERIFY_BYTES"]),
        "sha256": os.environ["S1_14_VERIFY_SHA"],
    },
    "checks": checks,
    "contract_boundary": {
        "is_per_attempt_backup_sync_manifest": False,
        "reason":
            "S1 occurs before pilot/official/ablation run attempts. "
            "Canonical per-attempt backup_sync_manifest.json is created only "
            "for a closed run attempt.",
    },
    "backup_preflight_script": {
        "version": "1.0.0",
        "sha256": os.environ["SCRIPT_SHA256"],
    },
    "verified_at_utc": datetime.now(timezone.utc).isoformat(),
}

path = Path(os.environ["S1_14_EVIDENCE"])
path.write_text(
    json.dumps(obj, indent=2) + "\n",
    encoding="utf-8",
)

parsed = json.loads(path.read_text(encoding="utf-8"))

valid = (
    parsed.get("status") == "PASS"
    and parsed.get("verification_status") == "PASS"
    and all(parsed.get("checks", {}).values())
    and parsed.get("contract_boundary", {}).get(
        "is_per_attempt_backup_sync_manifest"
    ) is False
)

print("S1_14_EVIDENCE_PATH =", path)
print("S1_14_EVIDENCE_VALIDATION =", "PASS" if valid else "FAIL")

if not valid:
    raise SystemExit(2)
PY

S1_14_SHA="$(sha256_of "$S1_14_EVIDENCE")"
echo "backup_checksum_verification_test_sha256=$S1_14_SHA"
echo "CHECKSUM_MATCH=True"
echo "BYTE_COUNT_MATCH=True"
echo "S1_14_STATUS=PASS"

# ============================================================================
# S1.15 — CLEAN-PATH RESTORE FOUNDATION
# ============================================================================
echo
echo "============================================================"
echo "S1.15 — CLEAN-PATH RESTORE FOUNDATION"
echo "============================================================"

S1_15_ART_SRC="$S1_14_EVIDENCE"
S1_15_CKPT_SRC="/tmp/s1_15_checkpoint_probe.pth"

S1_15_REMOTE_ROOT="${PREFLIGHT_REMOTE}/s1_15_restore_source"
S1_15_RESTORE_ROOT="/tmp/s1_15_restore_clean"

S1_15_ART_RESTORED="$S1_15_RESTORE_ROOT/backup_checksum_verification_test.json"
S1_15_CKPT_RESTORED="$S1_15_RESTORE_ROOT/s1_15_checkpoint_probe.pth"

python - "$S1_15_CKPT_SRC" "$INSTANCE_ID" <<'PY'
from pathlib import Path
import sys

p = Path(sys.argv[1])
instance_id = sys.argv[2]

p.write_bytes(
    b"SSOD_S1_15_CHECKPOINT_RESTORE_PROBE\n"
    + f"instance={instance_id}\n".encode("utf-8")
    + b"purpose=pretraining_restore_foundation_test\n"
)
PY

[[ -s "$S1_15_ART_SRC" ]] || fail "S1.15 scientific artifact source missing"
[[ -s "$S1_15_CKPT_SRC" ]] || fail "S1.15 checkpoint-like probe missing"

S1_15_ART_SHA="$(sha256_of "$S1_15_ART_SRC")"
S1_15_CKPT_SHA="$(sha256_of "$S1_15_CKPT_SRC")"

S1_15_ART_BYTES="$(stat -c '%s' "$S1_15_ART_SRC")"
S1_15_CKPT_BYTES="$(stat -c '%s' "$S1_15_CKPT_SRC")"

echo "artifact_source_sha256=$S1_15_ART_SHA"
echo "checkpoint_probe_source_sha256=$S1_15_CKPT_SHA"

"$RCLONE" --config "$RCLONE_CONF" mkdir "$S1_15_REMOTE_ROOT"

"$RCLONE" \
  --config "$RCLONE_CONF" \
  copyto \
  "$S1_15_ART_SRC" \
  "$S1_15_REMOTE_ROOT/backup_checksum_verification_test.json"

"$RCLONE" \
  --config "$RCLONE_CONF" \
  copyto \
  "$S1_15_CKPT_SRC" \
  "$S1_15_REMOTE_ROOT/s1_15_checkpoint_probe.pth"

echo "--- remote restore source ---"
"$RCLONE" --config "$RCLONE_CONF" lsf "$S1_15_REMOTE_ROOT"

rm -rf "$S1_15_RESTORE_ROOT"
mkdir -p "$S1_15_RESTORE_ROOT"

[[ ! -e "$S1_15_ART_RESTORED" ]] \
  || fail "clean restore target unexpectedly contains artifact"

[[ ! -e "$S1_15_CKPT_RESTORED" ]] \
  || fail "clean restore target unexpectedly contains checkpoint probe"

"$RCLONE" \
  --config "$RCLONE_CONF" \
  copyto \
  "$S1_15_REMOTE_ROOT/backup_checksum_verification_test.json" \
  "$S1_15_ART_RESTORED"

"$RCLONE" \
  --config "$RCLONE_CONF" \
  copyto \
  "$S1_15_REMOTE_ROOT/s1_15_checkpoint_probe.pth" \
  "$S1_15_CKPT_RESTORED"

S1_15_ART_RESTORE_SHA="$(sha256_of "$S1_15_ART_RESTORED")"
S1_15_CKPT_RESTORE_SHA="$(sha256_of "$S1_15_CKPT_RESTORED")"

S1_15_ART_RESTORE_BYTES="$(stat -c '%s' "$S1_15_ART_RESTORED")"
S1_15_CKPT_RESTORE_BYTES="$(stat -c '%s' "$S1_15_CKPT_RESTORED")"

echo "artifact_restored_sha256=$S1_15_ART_RESTORE_SHA"
echo "checkpoint_probe_restored_sha256=$S1_15_CKPT_RESTORE_SHA"

[[ "$S1_15_ART_SHA" == "$S1_15_ART_RESTORE_SHA" ]] \
  || fail "S1.15 scientific artifact restore SHA-256 mismatch"

[[ "$S1_15_CKPT_SHA" == "$S1_15_CKPT_RESTORE_SHA" ]] \
  || fail "S1.15 checkpoint probe restore SHA-256 mismatch"

[[ "$S1_15_ART_BYTES" == "$S1_15_ART_RESTORE_BYTES" ]] \
  || fail "S1.15 scientific artifact restore byte-count mismatch"

[[ "$S1_15_CKPT_BYTES" == "$S1_15_CKPT_RESTORE_BYTES" ]] \
  || fail "S1.15 checkpoint probe restore byte-count mismatch"

export S1_15_ART_SHA
export S1_15_CKPT_SHA
export S1_15_ART_RESTORE_SHA
export S1_15_CKPT_RESTORE_SHA
export S1_15_ART_BYTES
export S1_15_CKPT_BYTES
export S1_15_ART_RESTORE_BYTES
export S1_15_CKPT_RESTORE_BYTES
export S1_15_REMOTE_ROOT
export S1_15_RESTORE_ROOT
export S1_15_EVIDENCE

python - <<'PY'
from datetime import datetime, timezone
from pathlib import Path
import json
import os

checks = {
    "restore_target_was_clean": True,
    "artifact_sha256_match":
        os.environ["S1_15_ART_SHA"] ==
        os.environ["S1_15_ART_RESTORE_SHA"],
    "checkpoint_probe_sha256_match":
        os.environ["S1_15_CKPT_SHA"] ==
        os.environ["S1_15_CKPT_RESTORE_SHA"],
    "artifact_byte_count_match":
        int(os.environ["S1_15_ART_BYTES"]) ==
        int(os.environ["S1_15_ART_RESTORE_BYTES"]),
    "checkpoint_probe_byte_count_match":
        int(os.environ["S1_15_CKPT_BYTES"]) ==
        int(os.environ["S1_15_CKPT_RESTORE_BYTES"]),
}

obj = {
    "schema_version": "1.1",
    "artifact_type": "STORAGE_RESTORE_TEST",
    "lifecycle_scope": "S1_PRETRAINING_OPERATIONAL_TEST",
    "status": "PASS" if all(checks.values()) else "FAIL",
    "instance_id": os.environ["INSTANCE_ID"],
    "durable_source": {
        "provider": "Google Drive via rclone",
        "remote_root": os.environ["S1_15_REMOTE_ROOT"],
    },
    "clean_restore_root": os.environ["S1_15_RESTORE_ROOT"],
    "restored_objects": {
        "scientific_artifact": {
            "source_sha256": os.environ["S1_15_ART_SHA"],
            "restored_sha256": os.environ["S1_15_ART_RESTORE_SHA"],
            "source_bytes": int(os.environ["S1_15_ART_BYTES"]),
            "restored_bytes": int(os.environ["S1_15_ART_RESTORE_BYTES"]),
        },
        "checkpoint_probe": {
            "is_trained_model_checkpoint": False,
            "purpose": "pretraining restore-foundation probe",
            "source_sha256": os.environ["S1_15_CKPT_SHA"],
            "restored_sha256": os.environ["S1_15_CKPT_RESTORE_SHA"],
            "source_bytes": int(os.environ["S1_15_CKPT_BYTES"]),
            "restored_bytes": int(os.environ["S1_15_CKPT_RESTORE_BYTES"]),
        },
    },
    "checks": checks,
    "interpretation":
        "S1 verifies that durable backup content can be rehydrated into a "
        "clean path with exact SHA-256 and byte-count preservation before "
        "model training. Actual training checkpoint restore/resume remains "
        "subject to later training-stage checkpoint tests.",
    "backup_preflight_script": {
        "version": "1.0.0",
        "sha256": os.environ["SCRIPT_SHA256"],
    },
    "verified_at_utc": datetime.now(timezone.utc).isoformat(),
}

path = Path(os.environ["S1_15_EVIDENCE"])
path.write_text(
    json.dumps(obj, indent=2) + "\n",
    encoding="utf-8",
)

parsed = json.loads(path.read_text(encoding="utf-8"))
valid = (
    parsed.get("status") == "PASS"
    and all(parsed.get("checks", {}).values())
    and parsed.get("restored_objects", {})
              .get("checkpoint_probe", {})
              .get("is_trained_model_checkpoint") is False
)

print("S1_15_EVIDENCE_PATH =", path)
print("S1_15_STORAGE_RESTORE_TEST_VALIDATION =", "PASS" if valid else "FAIL")

if not valid:
    raise SystemExit(2)
PY

S1_15_SHA="$(sha256_of "$S1_15_EVIDENCE")"
echo "storage_restore_test_sha256=$S1_15_SHA"
echo "ARTIFACT_RESTORE_HASH_MATCH=True"
echo "CHECKPOINT_PROBE_RESTORE_HASH_MATCH=True"
echo "S1_15_STATUS=PASS"

# ----------------------------------------------------------------------------
# FINAL SCRIPT SUMMARY
# ----------------------------------------------------------------------------
echo
echo "========================================"
echo "S1.13-S1.15 BACKUP/RECOVERY SUMMARY"
echo "========================================"
echo "S1_13_STATUS=PASS"
echo "S1_14_STATUS=PASS"
echo "S1_15_STATUS=PASS"
echo "STORAGE_ENVIRONMENT_SHA256=$S1_13_STORAGE_SHA"
echo "BACKUP_CHECKSUM_EVIDENCE_SHA256=$S1_14_SHA"
echo "STORAGE_RESTORE_EVIDENCE_SHA256=$S1_15_SHA"
echo "PER_ATTEMPT_BACKUP_SYNC_MANIFEST_CREATED=False"
echo "S1_BACKUP_PREPARATION=PASS"
